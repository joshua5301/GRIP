"""Real tiny CPU connection and AST controls; no native source/data/GPU."""
import ast
import hashlib
import json
import math
import platform
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src.ce_gradient_alignment import POLICY, gradient_alignment_partials, source_gradient_targets
from src.finite_student_outer_v2 import geom_uniform_initial
from src.io import array_digest, cpu_state
from src.low_rank_assignment import LowRankMoments, initialize_factors, logit_block
from src.moments import decode_moments, make_material
from src.sweep_utils import representative
from src.transforms import FeatureTransform

REPO = Path(__file__).resolve().parents[1]
PATH = REPO / "src/citation_gradient_probe.py"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def tensor(value, shape, dtype, message):
    require(torch.is_tensor(value) and value.layout == torch.strided and tuple(value.shape) == tuple(shape)
            and value.dtype == dtype and bool(torch.isfinite(value).all()), message)
    return value


def extract(path, names, namespace):
    nodes = [n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == names
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def namespace():
    def count(evidence, key, completed=False):
        key += "_completed" if completed else "_attempts"
        evidence["counts"][key] = evidence["counts"].get(key, 0)+1

    def stop(callback):
        if callback():
            raise InterruptedError("stopped")

    inherited = extract(REPO/"src/citeseer_finite_student_v2.py", {"_adam_step", "_optimizer"},
                        dict(torch=torch, math=math, _require=require, _tensor=tensor, HORIZON=25))
    digest = extract(REPO/"src/finite_student_probe.py", {"_digest"},
                     dict(torch=torch, _require=require, array_digest=array_digest))["_digest"]
    material = extract(REPO/"src/citation_macro_probe.py", {"_material"}, dict(torch=torch,
        math=math, _require=require, _tensor=tensor, decode_moments=decode_moments,
        make_material=make_material, logit_block=logit_block))["_material"]
    ns = dict(torch=torch, math=math, _require=require, _tensor=tensor, _count=count, _stop=stop,
        POLICY=POLICY, source_gradient_targets=source_gradient_targets,
        gradient_alignment_partials=gradient_alignment_partials, representative=representative, _material=material,
        inherited=SimpleNamespace(**inherited), probe=SimpleNamespace(_digest=digest, cpu_state=cpu_state,
            _runtime_precision_guard=lambda: None, _transform=lambda b: FeatureTransform(**b["transform"]),
            _moments=lambda b,p: LowRankMoments.apply(*p,b["hard"],make_material(b["z"],b["q"]),.05,4096)))
    return extract(PATH, {"anchor_initial", "_targets", "_one_update", "_source_buffer_digest"}, ns)


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_actual_seed_geometry_parity_and_global_rng_preservation(seed, dtype):
    ns = namespace()
    rng = torch.random.get_rng_state().clone()
    ours = ns["anchor_initial"](4, 3, 5, seed, dtype=dtype)
    assert torch.equal(torch.random.get_rng_state(), rng)
    stock = extract(REPO/"src/evaluation.py", {"_initialize_geom_uniform"}, dict(torch=torch, math=math))
    layers = [SimpleNamespace(lin=SimpleNamespace(weight=torch.empty(o,i,dtype=dtype)),
                             bias=torch.empty(o,dtype=dtype)) for i,o in ((4,5),(5,3))]
    with torch.random.fork_rng():
        stock["_initialize_geom_uniform"](SimpleNamespace(layers=layers), seed)
    actual = tuple(v for layer in layers for v in (layer.lin.weight, layer.bias))
    assert all(torch.equal(a,b) and not a.requires_grad for a,b in zip(ours,actual,strict=True))
    assert torch.equal(torch.random.get_rng_state(), rng)
    if seed == 0:
        assert all(torch.equal(a,b) for a,b in zip(ours,geom_uniform_initial(4,3,5,dtype=dtype),strict=True))


def toy(ns):
    gen = torch.Generator().manual_seed(808)
    z = torch.randn(15,4,generator=gen,dtype=torch.float64)*.3
    q = (torch.randn(15,3,generator=gen,dtype=torch.float64)+torch.tensor([1.5,0.,-.7])).softmax(1)
    hard = torch.arange(15)%3
    u,v = initialize_factors(hard,3,2,0)
    transform = dict(kind="rms", matrix=None, center=torch.zeros(4,dtype=torch.float64),
                     output_center=torch.zeros(4,dtype=torch.float64), scale=torch.tensor(1.,dtype=torch.float64),eps=1e-8)
    buffers = dict(z=z,q=q,hard=hard,initial=[u.detach(),v.detach()], transform=transform,
        x=z.float(),h=z.float(),S=torch.eye(15)*.7+torch.ones(15,15)*.01,
        model_initial=ns["anchor_initial"](4,3,5,0))
    buffers["original_S"] = buffers["S"].to_sparse_csr()
    m0=ns["probe"]._moments(buffers,buffers["initial"])
    X,Q,mass=representative(m0,FeatureTransform(**transform),4,"cpu")
    reference=dict(actual_FP32_P0_X_Q_uniform_equal_reference=True,
        student_inputs=ns["probe"]._digest([X,Q,torch.full_like(mass,1/3)]))
    return buffers,dict(moments=m0.clone()),reference


def test_real_connected_P0_P1_one_backward_stock_Adam_and_immutable_cache(monkeypatch):
    ns=namespace(); b,saved,reference=toy(ns)
    before=ns["_source_buffer_digest"](b); evidence=dict(counts={}); calls=[]
    backward=LowRankMoments.backward
    monkeypatch.setattr(LowRankMoments,"backward",staticmethod(lambda ctx,g: calls.append(1) or backward(ctx,g)))
    states,targets=ns["_one_update"](b,saved,reference,evidence,lambda:False)
    assert calls==[1] and ns["_source_buffer_digest"](b)==before
    assert torch.equal(states[0]["moments"],saved["moments"])
    assert not torch.equal(states[0]["parameters"][0],states[1]["parameters"][0])
    assert torch.equal(states[0]["parameters"][1],states[1]["parameters"][1])
    assert states[0]["optimizer"]["state"]=={} and set(states[1]["optimizer"]["state"])=={0,1}
    assert evidence["target_model_digest_before"]==evidence["target_model_digest_after"]
    assert states[0]["scale"]==float(states[0]["alignment"]["loss"])>0
    assert len(targets["anchors"])==3
    for key,total in (("source_CE_gradient_target",3),("synthetic_alignment_partial",2),
                      ("additional_anchor_GEOM_factory",2),("original_moment_backward",1),("P_update",1),
                      ("P_optimizer_constructor",1)):
        assert evidence["counts"][key+"_attempts"]==evidence["counts"][key+"_completed"]==total


@pytest.mark.parametrize("mutation",["cached_moment","reference","source_zero","target_mutation","stop"])
def test_connected_flow_rejects_and_preserves_original_inputs(mutation):
    ns=namespace();b,saved,reference=toy(ns);before=ns["_source_buffer_digest"](b);e=dict(counts={})
    if mutation=="cached_moment": saved["moments"][0,0]+=.001
    elif mutation=="reference": reference["student_inputs"]="wrong"
    elif mutation=="source_zero":
        b["model_initial"]=tuple(torch.zeros_like(p) for p in b["model_initial"]);before=ns["_source_buffer_digest"](b)
    elif mutation=="target_mutation":
        actual=gradient_alignment_partials;calls=[]
        def altered(*args,**kwargs):
            result=actual(*args,**kwargs);calls.append(1)
            if len(calls)==2:args[2]["anchors"][0]["parameters"][0].add_(.01)
            return result
        ns["gradient_alignment_partials"]=altered
    with pytest.raises((ValueError,InterruptedError)):
        ns["_one_update"](b,saved,reference,e,lambda:mutation=="stop")
    assert ns["_source_buffer_digest"](b)==before
    assert e["counts"].get("P_update_attempts",0)==(1 if mutation=="target_mutation" else 0)


def spec_fixture(tmp_path):
    repo=tmp_path/"repo";folder=repo/"results/proposals";folder.mkdir(parents=True)
    tree=ast.parse(PATH.read_text());fixednode=next(n for n in tree.body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=="_FIXED" for t in n.targets))
    fixed={k.arg:ast.literal_eval(k.value) for k in fixednode.value.keywords}
    policy=json.loads(json.dumps(POLICY));sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    asset=repo/"results/asset";asset.write_text("fixed");pins={str(asset):sha(asset)}
    sci=dict(candidate=fixed,original_files_sha256=pins,parents=[])
    science=folder/"science.json";science.write_text(json.dumps(sci));output=str(repo/"results/research_loop/AY")
    spec=dict(schema=1,scientific_preregistration=dict(path=str(science),sha256=sha(science)),source={"actual":True},
        numerical_source={"versions":"fixed"},python_version=platform.python_version(),files_sha256=pins,
        roots=[{"cells":30},{"cells":120}],candidate=fixed,artifacts_sha256=pins,output_root=output,
        probe_outputs={"30":output+"/c30.json","120":output+"/c120.json"},
        gradient_policy=json.loads(json.dumps(policy)))
    path=folder/"spec.json"
    def checked(files):require(all(Path(p).is_file() and sha(p)==h for p,h in files.items()),"changed files")
    from src.citation_source_preflight import _exact
    ns=extract(PATH,{"_preserve","_load_spec"},dict(__file__=str(repo/"src/citation_gradient_probe.py"),
        Path=Path,json=json,platform=platform,torch=SimpleNamespace(get_num_threads=lambda:4),_FIXED=fixed,
        _POLICY_SPEC=policy,SCIENCE="science.json",SCIENCE_SHA=sha(science),_require=require,_sha=sha,_exact=_exact,
        implementation_provenance=lambda:{"actual":True},numerical_source=lambda:{"versions":"fixed"},
        probe=SimpleNamespace(_checked_files=checked),original=SimpleNamespace(_roots=lambda r:spec["roots"])))
    return spec,path,ns,sha,asset


@pytest.mark.parametrize("mutation",[None,"unknown","bool_schema","candidate","policy","source","duplicate","asset"])
def test_spec_guards_before_source_or_native(tmp_path,mutation):
    spec,path,ns,sha,asset=spec_fixture(tmp_path)
    if mutation=="unknown":spec["external_Q"]="override"
    elif mutation=="bool_schema":spec["schema"]=True
    elif mutation=="candidate":spec["candidate"]["rho"]*=10
    elif mutation=="policy":spec["gradient_policy"]["blocks"]=["W1","W2"]
    elif mutation=="source":spec["source"]={"actual":False}
    elif mutation=="duplicate":spec["probe_outputs"]["120"]=spec["probe_outputs"]["30"]
    elif mutation=="asset":asset.write_text("bad")
    path.write_text(json.dumps(spec))
    if mutation is None:assert ns["_load_spec"](path,sha(path))[0]==spec
    else:
        with pytest.raises(ValueError):ns["_load_spec"](path,sha(path))


def test_lazy_worker_dispatch_preserves_options_and_stop(monkeypatch):
    calls=[];expected=object();stop=lambda:False
    monkeypatch.setitem(sys.modules,"src.citation_gradient_probe",SimpleNamespace(
        prepare_probe=lambda **kw:calls.append(kw) or expected))
    ns=extract(REPO/"src/research_loop.py",{"dispatch"},{})
    options=dict(cells=30,spec_path="fixed",spec_sha256="sha",output_path="new")
    assert ns["dispatch"](dict(kind="citation_gradient_probe",options=options),stop) is expected
    assert calls==[dict(options,stop=stop)] and "stop" not in options
