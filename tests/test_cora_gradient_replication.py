"""New condensation/source/gate/dispatch controls only; AST and opaque fakes."""
import ast
import copy
import hashlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

BASE = Path(__file__).resolve().parents[1]
MODULE = BASE / "src/cora_gradient_replication.py"


def require(value, message):
    if not value: raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def exp(tmp_path):
    functions = {"_seed", "_case", "_folder", "_load_source", "_gate", "_load_spec", "canonical_candidate"}
    body = [n for n in ast.parse(MODULE.read_text()).body if isinstance(n, ast.FunctionDef) and n.name in functions]
    ns = dict(Path=Path, json=json, __file__=str(tmp_path / "src/cora_gradient_replication.py"),
              _require=require, _exact=lambda a, b: type(a) is type(b) and a == b,
              _sha=sha, _seal=digest, _fingerprint=lambda v: digest(v)[:12], HORIZON=25,
              SCIENCE="science.json", SCIENCE_SHA="sciencepin", _POLICY_SPEC={"rho":.001}, _SEEDS=(4200,4201,4202),
              _stop=lambda stop: require(not stop(), "Stopped"), _count=lambda e,k,complete=False: None)
    exec(compile(ast.Module(body=body, type_ignores=[]), str(MODULE), "exec"), ns)
    calls=[]
    source=dict(case={"cells":70,"reference_id":"ownNODE","source_root":str(tmp_path / "source")},
                baseline_candidate={"method":"low_rank","lr":.05},
                hard_assignments={str(c):{"path":str(tmp_path/f"assignment_own{c}.pt")} for c in (1,2)},
                existing_only_reference_dirs={str(c):str(tmp_path/f"NODE/condensation_{c}") for c in (1,2)},
                BG_certificates={str(c):{"path":f"receipt{c}","sha256":f"pin{c}"} for c in (1,2)})
    fields=("source_context","native_buffers_before","pre_optimizer_P0","student_recipe_origin",
            "endpoint_certificates","certified_files_sha256","history_metadata")
    source["BG_certified_own_origins"]={str(c):{k:{"own":c,"kind":k} for k in fields} for c in (1,2)}
    def own(spec, science, c, evidence, stop):
        calls.append(("source",c,copy.deepcopy(spec["case"])))
        evidence.update(copy.deepcopy(science["BG_certified_own_origins"][str(c)]))
        return dict(root=tmp_path,graph={"x":"X","adj":"CSR"},dense="dense",z=SimpleNamespace(device="fakeCUDA"),
                    transform="RMS",source={"own":c}), dict(parameters=(f"U{c}",f"V{c}"),inputs=["XQ",c],input_digest=f"input{c}"), {"own":c}
    def endpoints(buffers,origin,expected,folder,evidence):
        calls.append(("endpoints",str(folder)))
        evidence.update(complete25_native_certificate_passed=True,actual_P0_moments_bitwise_equal_core_step0=True,
                        actual_FP32_P0_X_Q_F64_uniform_equal_core_step0=True)
    ns.update(_own_source=own,_baseline_endpoints=endpoints,
              probe=SimpleNamespace(_frozen_transform=lambda v:{"frozen":v}, _digest=lambda v:v,
                                    _checked_files=lambda pins:calls.append(("pins",pins))),
              torch=SimpleNamespace(float32="FP32",load=lambda *a,**k:{"moments":"cachedP0"}),
              ay=SimpleNamespace(anchor_initial=lambda *a,**k:calls.append(("anchor",a)) or "GEOM0"))
    return SimpleNamespace(ns=ns,science=source,calls=calls,tmp=tmp_path)


@pytest.mark.parametrize("condensation_seed", [1,2])
def test_own_seed_source_BG_origins_before_anchors_and_reference_pairing(exp, condensation_seed):
    evidence={"counts":{}}
    buffers, saved, reference = exp.ns["_load_source"]({},exp.science,condensation_seed,evidence,lambda:False)
    assert [x[0] for x in exp.calls] == ["source","endpoints","anchor"]
    assert exp.calls[0][2]["condensation_seed"] == condensation_seed
    assert exp.calls[0][2]["hard_path"] == f"assignment_own{condensation_seed}.pt"
    assert buffers["initial"] == (f"U{condensation_seed}",f"V{condensation_seed}") and saved["moments"] == "cachedP0"
    assert reference["own_condensation_seed"] == condensation_seed
    assert reference["actual_current_M0_bitwise_equal_own_NODE0"] is True
    assert reference["own_BG_certificate"] == exp.science["BG_certificates"][str(condensation_seed)]
    assert reference["historical_UV_available"] is False
    assert evidence["original_recipe_origin"] == evidence["student_recipe_origin"]


@pytest.mark.parametrize("field", ["source_context","native_buffers_before","pre_optimizer_P0","endpoint_certificates"])
def test_coherent_borrowed_source_origin_or_reference_rejected_before_anchor_or_P(exp, field):
    old=exp.ns["_baseline_endpoints"]
    def mutate(*args):
        old(*args);args[-1][field]=copy.deepcopy(exp.science["BG_certified_own_origins"]["2"][field])
    exp.ns["_baseline_endpoints"]=mutate
    with pytest.raises(ValueError): exp.ns["_load_source"]({},exp.science,1,{"counts":{}},lambda:False)
    assert [x[0] for x in exp.calls] == ["source","endpoints"]


@pytest.mark.parametrize("value", [0,True,"1",3])
def test_only_new_condensation1_2_are_supported(exp,value):
    with pytest.raises(ValueError): exp.ns["_seed"](value)
    assert not exp.calls


@pytest.mark.parametrize("mutation", [None,"crosscond_gate","prefix_missing"])
def test_owncond_complete_gate_and_file_pins_before_student_access(exp,mutation):
    ns=exp.ns;spec=dict(certificate_outputs={"1":str(exp.tmp/"gate1.json")},source={"full":68},numerical_source={"native":"strict"},
        candidates={"1":{"seed":1}},candidate_ids={"1":"own1"},gradient_policy={"rho":.001},scientific_preregistration={"pin":"science"})
    gate=dict(passed=True,schema=1,assignment_steps=25,source=spec["source"],numerical_source=spec["numerical_source"],
        candidate=spec["candidates"]["1"],candidate_id="own1",cells=70,condensation_seed=1,full25_cache_replay_passed=True,
        actual_FP32_P0_X_Q_uniform_equal_reference=True,source_target_cache_replay_passed=True,source_reference_certificate_passed=True,
        gradient_policy=spec["gradient_policy"],scientific_preregistration=spec["scientific_preregistration"],spec_sha256="specpin",
        source_assets_spec_science_unchanged=True,test_enabled=False,prefix_sha256="a"*64,files_sha256={"opaque":"pin"})
    if mutation=="crosscond_gate":gate["condensation_seed"]=2
    elif mutation=="prefix_missing":gate.pop("prefix_sha256")
    path=Path(spec["certificate_outputs"]["1"]);path.write_text(json.dumps(gate))
    if mutation:
        with pytest.raises(ValueError):ns["_gate"](spec,1,sha(path),"specpin")
        assert not exp.calls
    else:
        assert ns["_gate"](spec,1,sha(path),"specpin")==gate and exp.calls[0][0]=="pins"


@pytest.mark.parametrize("operation", ["prepare","certify","student"])
def test_three_lazy_dispatches_preserve_owncond_options_callback_result(monkeypatch,operation):
    dispatch=next(n for n in ast.parse((BASE/"src/research_loop.py").read_text()).body if isinstance(n,ast.FunctionDef) and n.name=="dispatch")
    module=ModuleType("src.cora_gradient_replication");calls=[];result={"opaque":"receipt"}
    for name in ("prepare","certify","validate"):setattr(module,name,lambda **kwargs:calls.append(kwargs) or result)
    monkeypatch.setitem(sys.modules,module.__name__,module)
    ns={};exec(compile(ast.Module(body=[dispatch],type_ignores=[]),"lazy-worker","exec"),ns)
    callback=lambda:False;options={"condensation_seed":2,"spec_path":"frozen","spec_sha256":"a"*64,"output_path":"new"}
    assert ns["dispatch"]({"kind":f"citation_cora_gradient_replication_{operation}","options":options},callback) is result
    assert calls==[dict(options,stop=callback)] and "stop" not in options
