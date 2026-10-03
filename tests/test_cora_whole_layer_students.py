"""New cohort/context/physicalnamespace controls, not old student/math suites."""
import ast
import copy
import json
from pathlib import Path

import pytest

REPO=next(p for p in Path(__file__).resolve().parents if (p/'src/cora_gradient_students.py').is_file())
BASE=Path(__file__).resolve().parents[1];SOURCE=BASE/'src/cora_whole_layer_students.py'
if not SOURCE.is_file():SOURCE=REPO/'src/cora_whole_layer_students.py'
SCIENCE=REPO/'results/proposals/Cora70_two_whole_layer_fixed25_matched_NODE_scientific_stageBR_v2.json';S=json.loads(SCIENCE.read_text())


def require(v,m):
    if not v:raise ValueError(m)


def contract_fixture(tmp_path):
    case=copy.deepcopy(S['case']);science=copy.deepcopy(S)
    recipe=tmp_path/f"student_recipe_{case['recipe_id']}.json";recipe.write_text(json.dumps(case['recipe']))
    case['source_root']=str(tmp_path);science['case']=case
    science['candidate_folder']=str(tmp_path/'0b17a334ca58'/'whole_layer25')
    source=tmp_path/'science.json';source.write_text(json.dumps(science))
    gate=tmp_path/'gate.json';gate.write_text('{}')
    origin=dict(path=str(recipe),sha256='recipepin',recipe_id=case['recipe_id'],contents=case['recipe'])
    context=dict(case=case,cells=70,assignment_steps=25,student_recipe_origin=origin,
        scientific_preregistration=dict(path=str(REPO/'results/proposals'/SCIENCE.name),sha256='sciencepin'),
        spec=dict(candidate=S['candidate'],candidate_id=S['candidate_id'],certificate_outputs={'70':str(gate)}))
    # Frozen scientific file is redirected only in AST-generated fixture globals.
    tree=ast.parse(SOURCE.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_contract')
    # Only replace the derived scientific Path expression by the fixture's known file.
    fn=copy.deepcopy(fn)
    for n in ast.walk(fn):
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='science_path' for t in n.targets):n.value=ast.Name(id='fixture_science_path',ctx=ast.Load())
    context['scientific_preregistration']['path']=str(source)
    env=dict(Path=Path,__file__=str(REPO/'src/cora_whole_layer_students.py'),fixture_science_path=source,json=json,
        SCIENCE=SCIENCE.name,SCIENCE_SHA='sciencepin',SEEDS=(4400,4401,4402),_require=require,_exact=lambda a,b:type(a)==type(b) and a==b,
        _sha=lambda p:'gatepin' if Path(p)==gate else ('recipepin' if Path(p)==recipe else 'sciencepin'),
        _fingerprint=lambda x:case['recipe_id'],_source_unchanged=lambda c:None)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(SOURCE),'exec'),env)
    return env,science,dict(case=case,recipe_origin=origin),context,gate,source


@pytest.mark.parametrize('mutation',['oldseed','boolseed','wrongcontext','wrongrecipe','wronggate','wrongfolder'])
def test_owned_new_cohort_recipe_joint_gate_and_namespace(tmp_path,mutation):
    env,s,b,c,g,source=contract_fixture(tmp_path);seed=4400;gate='gatepin';folder=Path(s['candidate_folder']).parent/'shared_P0_validation'
    gate='a'*64;env['_sha']=lambda p,expected=gate:expected if Path(p)==g else ('recipepin' if Path(p).name.startswith('student_recipe') else 'sciencepin')
    if mutation=='oldseed':seed=4100
    elif mutation=='boolseed':seed=True
    elif mutation=='wrongcontext':c['spec']['candidate_id']='old'
    elif mutation=='wrongrecipe':b['recipe_origin']['recipe_id']='wrong'
    elif mutation=='wronggate':gate='b'*64
    else:folder=folder.parent/'gradient25_validation'
    with pytest.raises(ValueError):env['_contract'](folder,b,c,gate,seed)


def test_same_source_context_shared_P0_contract_all_fresh_seeds(tmp_path):
    env,s,b,c,g,source=contract_fixture(tmp_path);gate='a'*64
    env['_sha']=lambda p,expected=gate:expected if Path(p)==g else ('recipepin' if Path(p).name.startswith('student_recipe') else 'sciencepin')
    for seed in (4400,4401,4402):
        folder=Path(s['candidate_folder']).parent/'shared_P0_validation'
        actual,settings=env['_contract'](folder,b,c,gate,seed)
        assert actual==folder and settings['epochs']==600 and 'input_scale' not in settings


def test_private_numeric_header_completion_and_fit_AST_unchanged():
    old=ast.parse((REPO/'src/cora_gradient_students.py').read_text());new=ast.parse(SOURCE.read_text())
    for name in ('_validate_inputs','_student_header','_student_certificate','evaluate_student'):
        a=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name==name)
        b=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name==name)
        assert ast.dump(a)==ast.dump(b)
    text=SOURCE.read_text();assert 'SEEDS = (4400, 4401, 4402)' in text and 'first strict GCN validation' in text
