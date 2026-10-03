"""AST-extracted new confirmation guards; no production or numerical imports."""
import ast
import copy
import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

BASE = Path(__file__).resolve().parents[1]
CANDIDATE = BASE / 'src/citeseer_gradient_confirmation.py'
WORKER = BASE / 'src/research_loop.py'
REPO = next(p for p in Path(__file__).resolve().parents if (p/'src/ce_gradient_alignment.py').is_file())
SCIENCE = 'Citeseer120_CE_gradient_alignment_fixed25_confirmation_scientific_stageBP_v3.json'
SCI = json.loads((REPO/'results/proposals'/SCIENCE).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def extract(*names, **extra):
    tree = ast.parse(CANDIDATE.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    require(len(nodes) == len(names), 'Actual owned function missing')
    env = dict(Path=Path, json=json, math=math, _require=require, _cond=lambda c: c,
               HORIZON=25, _SEEDS=(4300, 4301, 4302, 4303, 4304), _sha=sha,
               _exact=lambda a, b: type(a) is type(b) and a == b,
               _seal=lambda v: hashlib.sha256(json.dumps(v, sort_keys=True).encode()).hexdigest(),
               _fingerprint=lambda v: hashlib.sha256(json.dumps(v, sort_keys=True).encode()).hexdigest()[:12])
    env.update(extra)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(CANDIDATE), 'exec'), env)
    return SimpleNamespace(**{name: env[name] for name in names}), env


@pytest.mark.parametrize('c', [1, 2, 0, True, 3., '3'])
def test_reject_wrong_or_coercible_condensation(c):
    f, _ = extract('_cond')
    with pytest.raises(ValueError):
        f._cond(c)


@pytest.mark.parametrize('c', [3, 4, 5])
def test_canonical_confirmation_namespace_and_own_folder(c):
    f, env = extract('_cond', 'canonical_candidate', '_folder',
        az=SimpleNamespace(_FIXED={k:v for k,v in SCI['candidates'][str(c)].items()
                                 if k not in ('method', 'confirmation_schema', 'condensation_seed')}),
        numerical_source=lambda: {'source': 'new immutable module'})
    candidate = f.canonical_candidate(c)
    assert {k:v for k,v in candidate.items() if k != 'gradient_source_digest'} == SCI['candidates'][str(c)]
    assert candidate['gradient_source_digest'] == env['_seal']({'source': 'new immutable module'})
    assert f._folder({'output_root':'/toy', 'candidate_ids':{str(c):'own'}},c) == Path(f'/toy/condensation_{c}/own')


@pytest.fixture
def manifest(tmp_path):
    repo = tmp_path/'repo'
    proposals = repo/'results/proposals'
    proposals.mkdir(parents=True)
    science = copy.deepcopy(SCI)
    science['output_root'] = str(repo/'results/research_loop/fixed_confirmation')
    sci_path = proposals/SCIENCE
    sci_path.write_text(json.dumps(science))
    candidates = {str(c):dict(science['candidates'][str(c)], gradient_source_digest='new') for c in (3,4,5)}
    preserved = []
    f, env = extract('_load_spec', __file__=str(repo/'src/citeseer_gradient_confirmation.py'),
        SCIENCE=SCIENCE, SCIENCE_SHA=sha(sci_path), POLICY={'blocks':['W1','b1','W2','b2'], 'rho':.001},
        canonical_candidate=lambda c:candidates[str(c)],
        probe=SimpleNamespace(_checked_files=lambda files:None),
        _preserve=lambda *args:preserved.append(args))
    spec = dict(schema=1, scientific_preregistration={'path':str(sci_path),'sha256':sha(sci_path)},
        source={}, numerical_source={}, python_version='toy', files_sha256=science['original_files_sha256'],
        fixed=science['fixed'], candidates=candidates,
        candidate_ids={k:env['_fingerprint'](v) for k,v in candidates.items()}, gradient_policy=env['POLICY'],
        output_root=science['output_root'], certificate_outputs={str(c):str(Path(science['output_root'])/f'condensation_{c}/native_certificate25_v1.json') for c in (3,4,5)},
        artifacts_sha256={str(repo/'src/citeseer_gradient_confirmation.py'):'owned', str(repo/'src/research_loop.py'):'worker',
                          str(repo/'tests/test_citeseer_gradient_confirmation.py'):'test'},
        condensation_seeds=[3,4,5], source_certificate_refs=science['source_certificate_refs'])
    path = proposals/'execution.json'
    return f, env, repo, path, spec, preserved


@pytest.mark.parametrize('mutation', ['none','legacy_seed','candidate','gate','reference','recipe','unknown','undeclared_src'])
def test_frozen_manifest_controls_and_exact_artifact_allowlist(manifest,mutation):
    f, _, repo, path, spec, preserved = manifest
    if mutation == 'legacy_seed':spec['condensation_seeds'] = [1,2,5]
    elif mutation == 'candidate':spec['candidates']['3']['confirmation_schema'] = 2
    elif mutation == 'gate':spec['certificate_outputs']['3'] = spec['certificate_outputs']['4']
    elif mutation == 'reference':spec['source_certificate_refs']['3'] = spec['source_certificate_refs']['4']
    elif mutation == 'recipe':spec['fixed']['student_recipe']['sha256'] = 'changed'
    elif mutation == 'unknown':spec['unused_override'] = True
    elif mutation == 'undeclared_src':spec['artifacts_sha256'][str(repo/'src/other.py')] = 'unknown'
    path.write_text(json.dumps(spec))
    if mutation == 'none':
        assert f._load_spec(path, sha(path))[0] == spec and len(preserved) == 1
    else:
        with pytest.raises(ValueError):f._load_spec(path,sha(path))
        assert preserved == []


@pytest.mark.parametrize('mutation', ['false','cross_cond','native_buffer','recipe','cache_bytes'])
def test_own_BO_receipt_and_source_binding_before_any_new_target_or_update(tmp_path,mutation):
    path = tmp_path/'certificate.json'
    ref = {'path':str(path),'sha256':'checked'}
    raw = dict(source={'own':'three'}, root=tmp_path, ghost={}, graph={'x':SimpleNamespace(detach=lambda:'X')},
               dense='S', transform='mapping')
    accepted = dict(passed=True,source_assets_spec_science_unchanged=True,condensation_seed=3,
        complete25_native_certificate_passed=True,actual_P0_moments_bitwise_equal_core_step0=True,
        actual_FP32_P0_X_Q_F64_uniform_equal_core_step0=True,source_context=raw['source'],native_buffers_after={'exact':'source'},
        student_recipe_origin={'recipe':'frozen'},certified_files_sha256={'cache':'old'})
    if mutation == 'false':accepted['complete25_native_certificate_passed'] = False
    elif mutation == 'cross_cond':accepted['condensation_seed'] = 4
    elif mutation == 'native_buffer':accepted['native_buffers_after'] = {'different':'source'}
    elif mutation == 'recipe':accepted['student_recipe_origin'] = {'different':'recipe'}
    path.write_text(json.dumps(accepted))
    e = {'counts':{},'student_recipe_origin':{'recipe':'frozen'}}
    reached = []
    def endpoints(*args):
        reached.append('endpoint')
        e['certified_files_sha256'] = {'cache':'different'}
    f,_ = extract('_load_source',
        _source_load=lambda *args:raw, source_certificate=SimpleNamespace(_native_buffers=lambda b:{'exact':'source'}, _core_config=lambda *args:{}),
        probe=SimpleNamespace(_checked_files=lambda pins:None),
        _origin=lambda *args:reached.append('origin') or {}, _endpoints=endpoints,
        torch=SimpleNamespace(load=lambda *args,**kw:{'config':{'save_assignment':False}}),
        original=SimpleNamespace(REFERENCE='own'), _count=lambda *args:pytest.fail('no model/target/update access'))
    with pytest.raises(ValueError):f._load_source(3,{'source_certificate_refs':{'3':ref}}, {},e,lambda:False)
    assert reached == (['origin','endpoint'] if mutation == 'cache_bytes' else [])
    assert e['counts'] == {}


@pytest.mark.parametrize('mutation',['cross_cond','source','prefix','policy'])
def test_exact_native_gate_before_cache_or_student_access(tmp_path,mutation):
    path = tmp_path/'gate.json'
    spec = dict(certificate_outputs={'3':str(path)},source={},numerical_source={},candidates={'3':{}},candidate_ids={'3':'own'},
                gradient_policy={},scientific_preregistration={})
    gate = dict(passed=True,schema=1,assignment_steps=25,source={},numerical_source={},candidate={},candidate_id='own',cells=120,
        condensation_seed=3,full25_cache_replay_passed=True,actual_FP32_P0_X_Q_uniform_equal_reference=True,
        source_reference_certificate_passed=True,common_AT_native_buffers_exact_excluding_hard=True,source_target_cache_replay_passed=True,
        gradient_policy={},scientific_preregistration={},prefix_sha256='a'*64,spec_sha256='b'*64,files_sha256={'opaque':'pin'})
    if mutation == 'cross_cond':gate['condensation_seed'] = 4
    elif mutation == 'source':gate['source_reference_certificate_passed'] = False
    elif mutation == 'prefix':gate['prefix_sha256'] = ''
    else:gate['gradient_policy'] = {'altered':'smoothing'}
    path.write_text(json.dumps(gate))
    f,_=extract('_gate','_validate',probe=SimpleNamespace(_checked_files=lambda *args:None),
               _load_progress=lambda *args:pytest.fail('cache/student before gate'))
    with pytest.raises(ValueError):f._validate(spec,{'condensation_seed':3},{},'gradient',sha(path),{'counts':{}},lambda:False)


def test_partial_namespace_is_preserved_without_target_or_optimizer_access(tmp_path):
    folder = tmp_path/'condensation_3/own'
    folder.mkdir(parents=True)
    marker = folder/'partial';marker.write_text('failed bytes')
    f,_=extract('_prepare',_folder=lambda *args:folder,
        _load_progress=lambda *args:require(False,'Partial trajectory preserved; no resume/fallback'))
    with pytest.raises(ValueError):f._prepare({}, {'condensation_seed':3},{},{'counts':{}},lambda:False)
    assert marker.read_text() == 'failed bytes' and list(folder.iterdir()) == [marker]


@pytest.mark.parametrize('arm',['node_reference','gradient'])
def test_five_paired_students_and_same_own_P0_physical_namespace(tmp_path,arm):
    seen=[]
    b=dict(condensation_seed=4,cells=120,root=tmp_path)
    spec=dict(candidate_ids={'4':'new'},output_root=str(tmp_path/'new'))
    directory=tmp_path/'new/condensation_4/new'
    states={0:{'moments':0},25:{'moments':25}}
    def student(folder,inputs,buffers,context,gate,seed,stop):
        seen.append((folder,seed,inputs))
        fit=0 if arm=='gradient' and inputs[0]==0 else 1
        return dict(actual_student_fits=fit,physical_route_outputs=2*fit,diagnostic_route_forwards=0)
    f,_=extract('_validate',_gate=lambda *args:{'prefix_sha256':'prefix'},_load_progress=lambda *args:({'states':states},{}),
        _folder=lambda *args:directory,_seal=lambda *args:'prefix',original=SimpleNamespace(REFERENCE='original'),
        probe=SimpleNamespace(_transform=lambda *args:None),representative=lambda m,*args:(m,'Q','mass'),
        torch=SimpleNamespace(load=lambda p,**kw:states[0 if '000000' in str(p) else 25],full_like=lambda *args:'equal weights',equal=lambda a,b:a==b),
        inherited=SimpleNamespace(_evaluate_student=student,_settings=lambda cells:{'epochs':1000}))
    e={'counts':{}}
    result=f._validate(spec,b,{},arm,'gate',e,lambda:False)
    assert len(result['rows'])==10 and result['student_seeds']==[4300,4301,4302,4303,4304]
    assert [seed for _,seed,_ in seen]==list(range(4300,4305))*2
    assert {folder for folder,_,inputs in seen if inputs[0]==0}=={directory/'shared_P0_validation'}
    assert {folder for folder,_,inputs in seen if inputs[0]==25}=={directory/f'{arm}25_validation'}
    assert e['counts']['physical_final_GCN_fits']==(10 if arm=='node_reference' else 5)
    assert e['counts']['physical_sameweights_serving_outputs']==(20 if arm=='node_reference' else 10)
    assert e['counts']['logical_student_conditions']==10 and e['counts']['logical_serving_conditions']==20


@pytest.mark.parametrize('operation',['prepare','certify','validate'])
def test_actual_three_lazy_dispatches_preserve_options_return_and_stop(operation,monkeypatch):
    tree=ast.parse(WORKER.read_text());dispatch=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='dispatch')
    env={};exec(compile(ast.Module(body=[dispatch],type_ignores=[]),str(WORKER),'exec'),env)
    seen=[];callback=lambda:False
    monkeypatch.setitem(sys.modules,'src.citeseer_gradient_confirmation',SimpleNamespace(**{operation:lambda **kw:seen.append(kw) or 'returned'}))
    assert env['dispatch']({'kind':'citeseer_gradient_confirmation_'+operation,'options':{'condensation_seed':5}},callback)=='returned'
    assert seen==[{'condensation_seed':5,'stop':callback}]


def test_no_new_mathematical_engine_or_old_public_wrapper_retarget():
    tree=ast.parse(CANDIDATE.read_text())
    aliases={name:ast.unparse(n.value) for n in tree.body if isinstance(n,ast.Assign) for target in n.targets
             for name in ([target.id] if isinstance(target,ast.Name) else [])}
    direct=next(n for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Tuple)
                and '_state' in ast.unparse(n.targets[0]))
    assert ast.unparse(direct.value)=='(az._state, az._check_progress, az._verify_targets, az._store, az._cache_files)'
    assert aliases['_origin_check']=='replication._origin_check'
    calls={ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call)}
    assert not {'replication.prepare','az.prepare','source_certificate.prepare_node_reference','gradient_alignment_partials'} & calls
    assert '_source_load' in calls and '_endpoints' in calls and 'inherited._evaluate_student' in calls
    run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_run')
    assert any(isinstance(n,ast.keyword) and n.arg=='validation_only' and isinstance(n.value,ast.Constant) and n.value.value is True for n in ast.walk(run))


@pytest.mark.parametrize('cleanup_failure',[False,True])
def test_primary_native_failure_kept_with_exclusive_failure_evidence(tmp_path,cleanup_failure):
    output=tmp_path/'condensation_3/native_prepare25_v1.json'
    spec=dict(output_root=str(tmp_path),certificate_outputs={},source={},numerical_source={},candidates={'3':{}},candidate_ids={'3':'own'},
              scientific_preregistration={},fixed={'source_backend':'BA dense_original_S'})
    source_error=ValueError('own source failed before targets')
    def source(*args):
        raise source_error
    def preserve(*args):
        if cleanup_failure:raise RuntimeError('immutable file drift')
    def write(path,value):
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('x') as stream:json.dump(value,stream)
    cuda=SimpleNamespace(is_initialized=lambda:False,reset_peak_memory_stats=lambda:None,synchronize=lambda:None,
        current_device=lambda:0,get_device_properties=lambda *args:SimpleNamespace(total_memory=8),
        max_memory_allocated=lambda:2,max_memory_reserved=lambda:3)
    f,_=extract('_run',_load_spec=lambda *args:(spec,{}),_load_source=source,_preserve=preserve,_write_new=write,
        _stop=lambda callback:require(not callback(),'stopped'),POLICY={},
        torch=SimpleNamespace(cuda=cuda,get_num_threads=lambda:4),original=SimpleNamespace(CUDA_CAPACITY=8),
        probe=SimpleNamespace(_native=lambda device:{},_runtime_precision_guard=lambda:None),
        platform=SimpleNamespace(python_version=lambda:'toy'),time=SimpleNamespace(monotonic=lambda:1.))
    with pytest.raises(ValueError) as raised:f._run('prepare',3,tmp_path/'spec','a'*64,output,None,None,lambda:False)
    assert raised.value is source_error
    report=json.loads(output.read_text())
    assert report['passed'] is False and report['error']==str(source_error) and report['counts']['P_update_completed']==0
    assert report['source_unchanged'] is (not cleanup_failure)
    assert ('preservation_error' in report) is cleanup_failure
