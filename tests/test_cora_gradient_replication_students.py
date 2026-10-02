"""Own 4200/cond-indexed recipe/folder/gate controls; no numerical imports."""
import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

BASE = Path(__file__).resolve().parents[1]
MODULE = BASE / "src/cora_gradient_replication_students.py"


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(v):
    return hashlib.sha256(json.dumps(v,sort_keys=True).encode()).hexdigest()[:12]


@pytest.fixture
def state(tmp_path):
    tree=ast.parse(MODULE.read_text());body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in {"_contract","_student_header"}]
    recipe=dict(epochs=600,eval_every=1,hidden=256,dropout=0.,lr=.001,weight_decay=.001,lr_schedule="constant",initialization="geom_uniform",input_scale=1.)
    recipe_id=fingerprint(recipe);source=tmp_path/"original";source.mkdir();rp=source/f"student_recipe_{recipe_id}.json";rp.write_text(json.dumps(recipe))
    common=dict(cells=70,source_root=str(source),reference_id="ownNODE",recipe_id=recipe_id,recipe=recipe)
    science=dict(case=common,hard_assignments={str(c):{"path":str(source/f"assignment_{c}.pt")} for c in (1,2)},student_seeds=[4200,4201,4202])
    sp=tmp_path/"results/proposals/science.json";sp.parent.mkdir(parents=True);sp.write_text(json.dumps(science))
    gates={str(c):str(tmp_path/f"gate{c}.json") for c in (1,2)}
    for c,p in gates.items():Path(p).write_text(f"own-native-{c}")
    spec=dict(output_root=str(tmp_path/"new"),candidate_ids={"1":"new1","2":"new2"},certificate_outputs=gates)
    origin=dict(path=str(rp),sha256=sha(rp),recipe_id=recipe_id,contents=copy.deepcopy(recipe))
    calls=[];ns=dict(Path=Path,json=json,SCIENCE="science.json",SCIENCE_SHA=sha(sp),SEEDS=(4200,4201,4202),
        __file__=str(tmp_path/"src/module.py"),_require=require,_sha=sha,_fingerprint=fingerprint,
        _exact=lambda a,b:type(a)is type(b) and a==b,_seal=lambda v:hashlib.sha256(json.dumps(v,sort_keys=True).encode()).hexdigest(),
        _source_unchanged=lambda context:calls.append("sourceguard"))
    exec(compile(ast.Module(body=body,type_ignores=[]),str(MODULE),"exec"),ns)
    def own(c):
        case=dict(common,condensation_seed=c,hard_path=f"assignment_{c}.pt")
        buffers=dict(case=case,recipe_origin=origin)
        context=dict(case=copy.deepcopy(case),cells=70,condensation_seed=c,assignment_steps=25,
            scientific_preregistration=dict(path=str(sp),sha256=sha(sp)),student_recipe_origin=copy.deepcopy(origin),spec=spec,
            implementation={"fullsource":68},numerical_source={"strict":"native"})
        folder=Path(spec["output_root"])/f"condensation_{c}"/spec["candidate_ids"][str(c)]/"shared_P0_validation"
        return folder,buffers,context,sha(gates[str(c)])
    return SimpleNamespace(ns=ns,own=own,calls=calls)


@pytest.mark.parametrize("condensation_seed",[1,2])
def test_own_recipe4200_gate_folder_and_header_bind_without_arm_or_step(state,condensation_seed):
    folder,buffers,context,gate=state.own(condensation_seed)
    actual,settings=state.ns["_contract"](folder,buffers,context,gate,4200)
    assert actual==folder and settings["weight_decay"]==.001 and "input_scale" not in settings
    header=state.ns["_student_header"](folder,buffers,context,gate,"exact-XQ-uniform",4200,settings)
    assert header["condensation_seed"]==condensation_seed and header["seed"]==4200
    assert header["physical_folder"]==str(folder) and header["native_gate_sha256"]==gate
    assert "arm" not in header and "step" not in header
    assert header["shared_P0_origin_reference_id"]=="ownNODE" and header["test_enabled"] is False
    assert state.calls==["sourceguard"]


@pytest.mark.parametrize("mutation",["old_student_seed","crosscond_context","crosscond_folder","crosscond_gate","recipe_decay","partial_horizon"])
def test_wrong_cond_seed_recipe_gate_or_horizon_rejected_before_any_student(state,mutation):
    folder,buffers,context,gate=state.own(1);seed=4200
    if mutation=="old_student_seed":seed=4100
    elif mutation=="crosscond_context":context["condensation_seed"]=2
    elif mutation=="crosscond_folder":folder=state.own(2)[0]
    elif mutation=="crosscond_gate":gate=state.own(2)[3]
    elif mutation=="recipe_decay":buffers["recipe_origin"]["contents"]["weight_decay"]=.0005
    elif mutation=="partial_horizon":context["assignment_steps"]=1
    with pytest.raises(ValueError):state.ns["_contract"](folder,buffers,context,gate,seed)
    assert not state.calls
