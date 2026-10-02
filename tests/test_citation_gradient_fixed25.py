"""Tiny CPU fixed25/cache proof; no realdata/GPU/head/student calls."""
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src import citation_gradient_fixed25 as exp
from src.low_rank_assignment import initialize_factors
from src.moments import make_material
from src.sweep_utils import representative
from src.transforms import FeatureTransform


def toy(root):
    gen=torch.Generator().manual_seed(808)
    z=torch.randn(15,4,generator=gen,dtype=torch.float64)*.3
    q=(torch.randn(15,3,generator=gen,dtype=torch.float64)+torch.tensor([1.5,0.,-.7])).softmax(1)
    hard=torch.arange(15)%3;u,v=initialize_factors(hard,3,2,0)
    transform=dict(kind="rms",matrix=None,center=torch.zeros(4,dtype=torch.float64),
        output_center=torch.zeros(4,dtype=torch.float64),scale=torch.tensor(1.,dtype=torch.float64),eps=1e-8)
    b=dict(root=root,cells=3,z=z,q=q,hard=hard,initial=[u.detach(),v.detach()],transform=transform,x=z.float(),h=z.float(),
        S=torch.eye(15)*.7+torch.ones(15,15)*.01,model_initial=exp.ay.anchor_initial(4,3,5,0))
    b["original_S"]=b["S"].to_sparse_csr()
    m=exp.probe._moments(b,b["initial"])
    x,y,mass=representative(m,FeatureTransform(**transform),4,"cpu")
    ref=dict(actual_FP32_P0_X_Q_uniform_equal_reference=True,
        student_inputs=exp.probe._digest([x,y,torch.full_like(mass,1/3)]))
    old=root/exp.original.REFERENCE/"condensation_0/checkpoints";old.mkdir(parents=True)
    torch.save(dict(moments=m),old/"step_000000.pt")
    return b,ref


@pytest.fixture(scope="module")
def complete(tmp_path_factory):
    monkeypatch=pytest.MonkeyPatch()
    monkeypatch.setattr(exp.probe,"_runtime_precision_guard",lambda:None)
    monkeypatch.setattr(exp.inherited,"_stable_files",lambda b:None)
    root=tmp_path_factory.mktemp("alignment25");b,ref=toy(root/"source")
    spec=dict(output_root=str(root/"study"),candidate_id="toy")
    context=dict(candidate={"method":"toyCPU"},reference=ref)
    e=dict(counts={});before=exp.ay._source_buffer_digest(b)
    result=exp._prepare(spec,b,context,e,lambda:False)
    folder=exp._folder(spec,3)
    assert result["frontier"]==25 and result["cached"] is False
    assert exp.ay._source_buffer_digest(b)==before
    yield SimpleNamespace(b=b,ref=ref,spec=spec,context=context,folder=folder,evidence=e)
    monkeypatch.undo()


def load(path):return torch.load(path,map_location="cpu",weights_only=False)


def test_real_tiny_25_updates_and_complete_mathematical_replay(complete):
    c=complete;e=dict(counts={});before=exp._cache_files(c.folder,complete=True)
    bundle,pins=exp._load_progress(c.folder,c.b,c.context,e,lambda:False)
    assert set(bundle["states"])==set(range(26)) and pins==before
    for step,state in bundle["states"].items():
        assert state["optimizer"]["state"]=={} if step==0 else set(state["optimizer"]["state"])=={0,1}
        assert ("scaled_factor_gradients" in state)==(step<25)
        assert state["moment_partial"].shape==state["moments"].shape
    assert c.evidence["counts"]["P_update_completed"]==25
    assert c.evidence["counts"]["source_CE_gradient_target_completed"]==3
    assert c.evidence["counts"]["alignment_partial_completed"]==26
    assert c.evidence["counts"]["original_moment_backward_completed"]==25
    assert e["counts"]["target_replay_source_CE_gradient_target_completed"]==3
    assert e["counts"]["alignment_partial_completed"]==26
    assert e["counts"]["original_moment_backward_completed"]==25 and e["counts"].get("P_update_attempts",0)==0
    assert torch.equal(bundle["states"][0]["parameters"][1],bundle["states"][1]["parameters"][1])
    assert abs(float(bundle["states"][25]["moments"].sum(0).sub(make_material(c.b["z"],c.b["q"]).mean(0)).abs().max()))<=1e-12


@pytest.mark.parametrize("mutation",["counter","first","second","parameter","gradient","partial","J0","history","missing","bool_frontier","container"])
def test_coupled_prefix_and_midstep_resealed_tamper(complete,mutation):
    c=complete;bundle=load(c.folder/"progress.pt");state=bundle["states"][7]
    if mutation=="counter":state["optimizer"]["state"][0]["step"]+=1
    elif mutation=="first":state["optimizer"]["state"][0]["exp_avg"].flatten()[0]+=.01
    elif mutation=="second":state["optimizer"]["state"][0]["exp_avg_sq"].flatten()[0]+=.01
    elif mutation=="parameter":state["parameters"][0].flatten()[0]+=.01
    elif mutation=="gradient":state["scaled_factor_gradients"][0].flatten()[0]+=.01
    elif mutation=="partial":state["moment_partial"].flatten()[0]+=.01
    elif mutation=="J0":
        bundle["J0"]+=.1
        for s in bundle["states"].values():s["J0"]=bundle["J0"]
        bundle["states"][0]["alignment_J"]+=.1;bundle["history"]=exp._history(bundle["states"])
    elif mutation=="history":bundle["history"][7]["alignment_J"]+=.1
    elif mutation=="missing":del bundle["states"][7]
    elif mutation=="bool_frontier":bundle["frontier"]=True
    else:bundle["states"][0]=None
    bundle=exp.probe._attach({k:v for k,v in bundle.items() if k!="content_sha256"})
    targets=exp._native_targets(load(c.folder/"source_gradient_targets.pt")["targets"],torch.device("cpu"))
    with pytest.raises(ValueError):exp._check_progress(bundle,c.b,c.context,targets,
        exp._sha(c.folder/"source_gradient_targets.pt"),dict(counts={}),lambda:False)


@pytest.mark.parametrize("mutation",["gradient","model","policy","norm","context"])
def test_forged_target_cache_rechecked_against_original_source(complete,tmp_path,mutation):
    c=complete;payload=load(c.folder/"source_gradient_targets.pt")
    if mutation=="gradient":payload["targets"]["anchors"][0]["gradients"][0].flatten()[0]+=.01
    elif mutation=="model":payload["targets"]["anchors"][1]["parameters"][0].flatten()[0]+=.01
    elif mutation=="policy":payload["targets"]["policy"]["rho"]*=10
    elif mutation=="norm":payload["targets"]["anchors"][0]["source_norms"][0]+=.01
    else:payload["context"]["reference"]={}
    torch.save(exp.probe._attach({k:v for k,v in payload.items() if k!="content_sha256"}),tmp_path/"source_gradient_targets.pt")
    before=exp._sha(tmp_path/"source_gradient_targets.pt")
    with pytest.raises(ValueError):exp._verify_targets(tmp_path,c.b,c.context,dict(counts={}),lambda:False)
    assert exp._sha(tmp_path/"source_gradient_targets.pt")==before


@pytest.mark.parametrize("mutation",["missing_target","missing_terminal","orphan","history","endpoint"])
def test_partial_or_tampered_cache_never_repaired_or_restarted(complete,tmp_path,mutation):
    c=complete;folder=tmp_path/"cache";shutil.copytree(c.folder,folder)
    if mutation=="missing_target":(folder/"source_gradient_targets.pt").unlink()
    elif mutation=="missing_terminal":(folder/"step_000025.pt").unlink()
    elif mutation=="orphan":(folder/"step_bad.pt").write_text("bad")
    elif mutation=="history":(folder/"history.json").write_text("[]")
    else:
        state=load(folder/"step_000025.pt");state["alignment_J"]+=.01;torch.save(state,folder/"step_000025.pt")
    before={p.name:exp._sha(p) for p in folder.iterdir()};e=dict(counts={})
    with pytest.raises(ValueError):exp._load_progress(folder,c.b,c.context,e,lambda:False)
    assert {p.name:exp._sha(p) for p in folder.iterdir()}==before and e["counts"].get("P_update_attempts",0)==0


def test_source_and_zero_initial_cache_origin_are_bound(complete,tmp_path):
    c=complete;bundle=load(c.folder/"progress.pt");changed=c.b["z"].clone();changed[0,0]+=.001
    buffers=dict(c.b,z=changed)
    with pytest.raises(ValueError):exp._origin_check(bundle["states"][0]["moments"]+.001,buffers,c.ref,dict(counts={}))
    targets=exp._native_targets(load(c.folder/"source_gradient_targets.pt")["targets"],torch.device("cpu"))
    with pytest.raises(ValueError):exp._check_progress(bundle,buffers,c.context,targets,bundle["target_sha256"],dict(counts={}),lambda:False)


@pytest.mark.parametrize("method",["prepare","certify","validate"])
def test_real_lazy_dispatch_forwarding(monkeypatch,method):
    calls=[];expected=object();stop=lambda:False
    monkeypatch.setitem(sys.modules,"src.citation_gradient_fixed25",SimpleNamespace(**{
        method:lambda **kw:calls.append(kw) or expected}))
    from src.research_loop import dispatch
    options=dict(cells=30,spec_path="fixed",spec_sha256="sha",output_path="new")
    if method=="validate":options.update(arm="gradient",gate_sha256="gate")
    assert dispatch(dict(kind="citation_gradient_fixed25_"+method,options=options),stop) is expected
    assert calls==[dict(options,stop=stop)] and "stop" not in options


@pytest.mark.parametrize("mutation",["failed","prefix","source","targets"])
def test_native_gate_is_required_before_student_access(tmp_path,monkeypatch,mutation):
    spec=dict(certificate_outputs={"30":str(tmp_path/"gate.json")},source={"x":1},numerical_source={"version":1},
        candidate={},candidate_id="cid",scientific_preregistration={},gradient_policy={})
    gate=dict(passed=True,schema=1,assignment_steps=25,source=spec["source"],numerical_source=spec["numerical_source"],
        candidate={},candidate_id="cid",cells=30,full25_cache_replay_passed=True,actual_FP32_P0_X_Q_uniform_equal_reference=True,
        source_target_cache_replay_passed=True,gradient_policy={},scientific_preregistration={},prefix_sha256="a"*64,
        spec_sha256="b"*64,files_sha256={})
    if mutation=="failed":gate["passed"]=False
    elif mutation=="prefix":del gate["prefix_sha256"]
    elif mutation=="source":gate["source"]={"x":2}
    else:gate["source_target_cache_replay_passed"]=False
    path=Path(spec["certificate_outputs"]["30"]);path.write_text(json.dumps(gate))
    monkeypatch.setattr(exp.probe,"_checked_files",lambda pins:None)
    with pytest.raises(ValueError):exp._gate(spec,30,exp._sha(path))
