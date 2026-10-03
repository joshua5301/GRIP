"""New hybrid4500/cohort/recipe/shared-P0 contracts; no numerical imports."""
import ast
import copy
import json
from pathlib import Path

import pytest

REPO = next(p for p in Path(__file__).resolve().parents if (p/'src/cora_whole_layer_students.py').is_file())
BASE = Path(__file__).resolve().parents[1]
SOURCE = BASE/'src/cora_hybrid_students.py'
if not SOURCE.is_file(): SOURCE = REPO/'src/cora_hybrid_students.py'
SCIENCE = REPO/'results/proposals/Cora70_half_teacher_CE_whole_layer_fixed25_scientific_stageBV_v1.json'
S = json.loads(SCIENCE.read_text())


def require(value, message):
    if not value: raise ValueError(message)


def contract_fixture(tmp_path):
    case,science = copy.deepcopy(S['case']),copy.deepcopy(S)
    recipe = tmp_path/f"student_recipe_{case['recipe_id']}.json"
    recipe.write_text(json.dumps(case['recipe']))
    case['source_root'] = str(tmp_path)
    science['case'] = case
    science['candidate_folder'] = str(tmp_path/'freshhybrid'/'hybrid25')
    source,gate = tmp_path/'science.json',tmp_path/'gate.json'
    source.write_text(json.dumps(science))
    gate.write_text('{}')
    origin = dict(path=str(recipe),sha256='recipepin',recipe_id=case['recipe_id'],contents=case['recipe'])
    context = dict(case=case,cells=70,assignment_steps=25,student_recipe_origin=origin,
        scientific_preregistration=dict(path=str(source),sha256='sciencepin'),
        spec=dict(candidate=S['candidate'],candidate_id=S['candidate_id'],certificate_outputs={'70':str(gate)}))
    tree = ast.parse(SOURCE.read_text())
    fn = copy.deepcopy(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == '_contract'))
    for n in ast.walk(fn):
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id == 'science_path' for t in n.targets):
            n.value = ast.Name(id='fixture_science_path',ctx=ast.Load())
    env = dict(Path=Path,__file__=str(REPO/'src/cora_hybrid_students.py'),fixture_science_path=source,json=json,
        SCIENCE=SCIENCE.name,SCIENCE_SHA='sciencepin',SEEDS=(4500,4501,4502),_require=require,
        _exact=lambda a,b:type(a)==type(b) and a==b,_fingerprint=lambda _:case['recipe_id'],_source_unchanged=lambda _:None,
        _sha=lambda p:'a'*64 if Path(p)==gate else ('recipepin' if Path(p)==recipe else 'sciencepin'))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(SOURCE),'exec'),env)
    return env,science,dict(case=case,recipe_origin=origin),context


@pytest.mark.parametrize('mutation',['old_seed','bool_seed','wrong_context','wrong_recipe','wrong_gate','wrong_folder'])
def test_own_hybrid_cohort_joint_gate_and_physical_namespace(tmp_path,mutation):
    env,science,buffers,context = contract_fixture(tmp_path)
    seed,gate,folder = 4500,'a'*64,Path(science['candidate_folder']).parent/'shared_P0_validation'
    if mutation == 'old_seed': seed = 4400
    elif mutation == 'bool_seed': seed = True
    elif mutation == 'wrong_context': context['spec']['candidate_id'] = 'old'
    elif mutation == 'wrong_recipe': buffers['recipe_origin']['recipe_id'] = 'wrong'
    elif mutation == 'wrong_gate': gate = 'b'*64
    else: folder = folder.parent/'whole_layer25_validation'
    with pytest.raises(ValueError): env['_contract'](folder,buffers,context,gate,seed)


def test_exact_P0_and_hybrid_contract_preserve_fixed600_recipe(tmp_path):
    env,science,buffers,context = contract_fixture(tmp_path)
    for seed in (4500,4501,4502):
        for name in ('shared_P0_validation','node_reference25_validation','hybrid25_validation'):
            folder = Path(science['candidate_folder']).parent/name
            actual,settings = env['_contract'](folder,buffers,context,'a'*64,seed)
            assert actual == folder and settings['epochs'] == 600 and 'input_scale' not in settings


def test_protected_fit_completion_AST_and_private_aliases_unchanged():
    old = ast.parse((REPO/'src/cora_whole_layer_students.py').read_text())
    new = ast.parse(SOURCE.read_text())
    get = lambda t,n:next(v for v in t.body if isinstance(v,ast.FunctionDef) and v.name == n)
    assert ast.dump(get(old,'evaluate_student')) == ast.dump(get(new,'evaluate_student'))
    pairs = {}
    for n in new.body:
        if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Tuple):
            pairs.update(zip([ast.unparse(x) for x in n.targets[0].elts],[ast.unparse(x) for x in n.value.elts],strict=True))
    for name in ('_source_unchanged','_validate_inputs','_student_header','_student_certificate'):
        assert pairs[name] == 'old_students.'+name
    assert not any(isinstance(n,ast.Assign) and any(isinstance(t,ast.Attribute) for t in n.targets) for n in new.body)
