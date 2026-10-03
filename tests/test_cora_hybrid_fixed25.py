"""New hybrid25 coupled-head/cache/API controls through stdlib AST and fakes."""
import ast
import copy
import hashlib
import json
import sys
import types
from pathlib import Path

import pytest

REPO = next(p for p in Path(__file__).resolve().parents if (p/'src/cora_hybrid_probe.py').is_file())
BASE = Path(__file__).resolve().parents[1]
SOURCE, WORKER = BASE/'src/cora_hybrid_fixed25.py', BASE/'src/research_loop.py'
if not SOURCE.is_file():
    SOURCE, WORKER = REPO/'src/cora_hybrid_fixed25.py', REPO/'src/research_loop.py'
SCIENCE = REPO/'results/proposals/Cora70_half_teacher_CE_whole_layer_fixed25_scientific_stageBV_v1.json'
S = json.loads(SCIENCE.read_text())


def require(value, message):
    if not value:
        raise ValueError(message)


class A:
    shape = (1,)

    def __init__(self, value):
        self.value = value

    def detach(self): return self
    def clone(self): return A(self.value)
    def requires_grad_(self, value=True): return self
    def to(self, other): return self


def normalize(value):
    if isinstance(value, A): return value.value
    if isinstance(value, dict): return {k:normalize(v) for k,v in value.items()}
    if isinstance(value, (tuple,list)): return [normalize(v) for v in value]
    return value


def seal(value):
    return hashlib.sha256(json.dumps(normalize(value), sort_keys=True).encode()).hexdigest()


def tree(): return ast.parse(SOURCE.read_text())
def function(name): return next(n for n in tree().body if isinstance(n,ast.FunctionDef) and n.name == name)
def calls(node, name): return [n for n in ast.walk(node) if isinstance(n,ast.Call) and ast.unparse(n.func) == name]


def extract(*names, **values):
    env = dict(Path=Path,json=json,_require=require,_exact=lambda a,b:normalize(a)==normalize(b),_seal=seal,
        SCIENCE=SCIENCE.name,SCIENCE_SHA=hashlib.sha256(SCIENCE.read_bytes()).hexdigest(),SEEDS=(4500,4501,4502),
        _POLICY_SPEC=S['grouping_policy'])
    fields = next(n for n in tree().body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id == '_FIELDS' for t in n.targets))
    env['_FIELDS'] = ast.literal_eval(fields.value)
    env.update(values)
    exec(compile(ast.Module(body=[function(n) for n in names],type_ignores=[]),str(SOURCE),'exec'),env)
    return env


def spec():
    fixed = {k:copy.deepcopy(S[k]) for k in ('fixed','case','candidate','candidate_id','matched_NODE_candidate','output_root','certificate_outputs','student_seeds')}
    files = dict(S['source_before']['files'])
    files.update({p:'owned' for p in S['source_protection']['owned'] if p.startswith('src/')})
    return dict(fixed,schema=1,source=dict(files=files),
        numerical_source=dict(versions=S['version_binding']['runtime_versions_from_pinned_native_metadata']),
        python_version=S['version_binding']['python_version'],files_sha256=S['original_files_sha256'],
        grouping_policy=S['grouping_policy'],artifacts_sha256={str(REPO/'src/cora_hybrid_fixed25.py'):'owned'},
        scientific_preregistration=dict(path=str(SCIENCE),sha256=hashlib.sha256(SCIENCE.read_bytes()).hexdigest()))


@pytest.mark.parametrize('mutation',['source_token','old_NODE_lr','bool_schema','old_cohort','unknown_artifact'])
def test_hybrid_identity_and_readonly_baseline_controls(mutation):
    value = spec()
    if mutation == 'source_token': value['candidate']['source_token'] = 'new'
    elif mutation == 'old_NODE_lr': value['matched_NODE_candidate']['lr'] = .05
    elif mutation == 'bool_schema': value['schema'] = True
    elif mutation == 'old_cohort': value['student_seeds'] = [4400,4401,4402]
    else: value['artifacts_sha256'] = {str(REPO/'src/soft_ce_partition.py'):'undeclared'}
    with pytest.raises(ValueError):
        extract('_spec_controls',_fingerprint=lambda _:S['candidate_id'])['_spec_controls'](value,S,REPO)


def test_exact_fixed_hybrid_spec_and_owned_artifacts_pass():
    value = spec()
    value['artifacts_sha256'] = {str(REPO/p):'pin' for p in S['source_protection']['owned']}
    extract('_spec_controls',_fingerprint=lambda _:S['candidate_id'])['_spec_controls'](value,S,REPO)


def test_partial_hybrid_namespace_never_resumes(tmp_path):
    science = copy.deepcopy(S)
    science['candidate_folder'] = str(tmp_path/'partial')
    science['operation_outputs']['prepare'] = str(tmp_path/'prepare.json')
    Path(science['candidate_folder']).mkdir()
    with pytest.raises(ValueError,match='Fresh hybrid25'):
        extract('_request')['_request']('prepare',70,None,None,science['operation_outputs']['prepare'],{},science)


def progress_fixture(tmp_path):
    folder = tmp_path/'hybrid25'
    folder.mkdir()
    context = dict(spec=dict(candidate={}),bound_source='current84')
    scales = dict(F0=.8,G0=.2)
    states = {}
    for step in range(26):
        states[step] = dict(step=step,parameters=[A(step),A(step+10)],optimizer=dict(counter=step),moments=A(5),
            theta=A(100+step),joint=dict(teacher_CE=.8,alignment=.2,objective=1.,normalizers=dict(scales)),
            normalizers=dict(scales),head_work={'historical_calls':int(step > 0)},terminal_no_update=step==25)
        if step < 25: states[step]['scaled_factor_gradients'] = [A(1),A(1)]
    canonical = copy.deepcopy(states)
    history = lambda values:[dict(step=k) for k in values]
    bundle = dict(schema=1,context=context,frontier=25,states=states,normalizers=scales,history=history(states),target_digest='targets')
    bundle['content_sha256'] = seal(bundle)
    (folder/'candidate.json').write_text('{}')
    (folder/'history.json').write_text(json.dumps(bundle['history']))
    seen = []

    def load(path, **kwargs):
        if Path(path).name == 'progress.pt': return bundle
        step = 0 if '000000' in Path(path).name else 25
        return dict(context=context,state=states[step])

    def optimizer(saved, parameters, first, second, step):
        require(saved['counter'] == step,'Adam slot/counter differs')

    def state(buffers,parameters,slots,step,theta,frozen,features,anchors,targets,evidence,stop,head_work,fit_head):
        require(fit_head is False,'Readonly fitted a head')
        require(theta.value == 100+step,'Saved stationary theta differs')
        require(frozen is None if step == 0 else frozen == scales,'Frozen normalizers differ')
        seen.append((step,fit_head,head_work))
        return copy.deepcopy(canonical[step]),dict(scales)

    fake_torch = types.SimpleNamespace(float32='f32',float64='f64',load=load,zeros_like=lambda _:A(0),equal=lambda a,b:a.value==b.value)
    fake_probe = types.SimpleNamespace(_tensor=lambda v,*args:v,_digest=lambda v:v,_attach=lambda v:v)
    env = extract('_load_progress',torch=fake_torch,probe=fake_probe,
        bq=types.SimpleNamespace(bn=types.SimpleNamespace(bi=types.SimpleNamespace(_sealed_payload=lambda _:None))),
        _hybrid_files=lambda _:dict(pins='fixed'),_state=state,_history=history,_bump=lambda *args:None,
        augmented=lambda _: 'source-features',inherited=types.SimpleNamespace(_optimizer=optimizer,
            _adam_step=lambda p,m,v,g,k:([A(x.value+1) for x in p],m,v)))
    args = (dict(candidate_folder=str(folder)),dict(initial=[A(0),A(10)],z='z'),dict(theta=A(100)),dict(moments=A(5)),context,'anchors','targets',dict(operation='certify'),lambda:False)
    return env,bundle,seen,args


@pytest.mark.parametrize('mutation',['missing_state','middle_theta','middle_scale','middle_gradient','middle_slot'])
def test_full_joint_theta_two_scales_Pgrad_and_Adam_cache_bound(tmp_path,mutation):
    env,bundle,seen,args = progress_fixture(tmp_path)
    state = bundle['states'][13]
    if mutation == 'missing_state': bundle['states'].pop(13)
    elif mutation == 'middle_theta': state['theta'] = A(999)
    elif mutation == 'middle_scale': state['normalizers']['F0'] = .9
    elif mutation == 'middle_gradient': state['scaled_factor_gradients'][0] = A(9)
    else: state['optimizer']['counter'] = 12
    with pytest.raises(ValueError): env['_load_progress'](*args)


def test_readonly_all26_saved_heads_no_refit_and_terminal_full_joint(tmp_path):
    env,bundle,seen,args = progress_fixture(tmp_path)
    states = env['_load_progress'](*args)
    assert [(k,fit) for k,fit,_ in seen] == [(k,False) for k in range(26)]
    assert 'scaled_factor_gradients' not in states[25]
    assert args[-2]['full25_cache_replay_passed'] is True
    assert args[-2]['normalizers'] == dict(F0=.8,G0=.2)


def test_joint_alias_counts_partial_failure_without_planned_completion():
    def count(e,key,complete=False):
        name = key+('_completed' if complete else '_attempts')
        e['counts'][name] = e['counts'].get(name,0)+1

    def fail(*args):
        count(args[-2],'teacher_outer')
        count(args[-2],'teacher_outer',True)
        count(args[-2],'adjoint_solve')
        raise RuntimeError('actual-adjoint-failure')

    env = extract('_joint_call',inherited=types.SimpleNamespace(_count=count),bu=types.SimpleNamespace(_joint=fail))
    evidence = dict(operation='certify',counts={})
    with pytest.raises(RuntimeError,match='actual-adjoint-failure'):
        env['_joint_call']('M',{},'theta','anchors','targets',None,'features',evidence,lambda:False)
    assert evidence['counts'] == dict(replay_hybrid_joint_partial_attempts=1,replay_hybrid_teacher_outer_attempts=1,
        replay_hybrid_teacher_outer_completed=1,replay_hybrid_adjoint_solve_attempts=1)


def test_fresh_heads_terminal_joint_no_baseline_optimizer_and_transform_AST():
    state,prepare,replay = function('_state'),function('_hybrid_prepare'),function('_load_progress')
    assert len(calls(state,'bu._joint')) == 0 and len(calls(function('_joint_call'),'bu._joint')) == 1
    assert len(calls(state,'solve_inner_newton_first')) == 1 and not calls(replay,'solve_inner_newton_first')
    assert 'fit_head=step > 0' in ast.unparse(prepare) and 'fit_head=False' in ast.unparse(replay)
    assert len(calls(state,'moments.backward')) == 1 and 'if step < 25:' in ast.unparse(state)
    assert 'range(26)' in ast.unparse(prepare) and len(calls(prepare,'optimizer.step')) == 1
    assert len(calls(prepare,'inherited._adam_step')) == 1 and len(calls(replay,'inherited._adam_step')) == 1
    assert ast.unparse(prepare).count("_bump(evidence, 'Adam_recursion'") == 2
    assert ast.unparse(replay).count("_bump(evidence, 'Adam_recursion'") == 2
    assert "head_work=s['head_work']" in ast.unparse(function('_history'))
    assert 'optimize_ce_assignment' not in SOURCE.read_text() and '_node_prepare' not in SOURCE.read_text()
    assert ast.unparse(calls(function('_validate'),'representative')[0].args[1]) == 'probe._transform(buffers)'
    fit = calls(state,'solve_inner_newton_first')[0]
    assert {k.arg:ast.unparse(k.value) for k in fit.keywords} == dict(initial='theta',max_iter='2000',grad_tol='1e-07',cg_max_iter='512',newton_steps='8',cg_check_interval='1')
    assert 'from src.whole_layer_gradient_alignment import POLICY' in SOURCE.read_text()


def test_direct_private_aliases_and_joint_are_not_retargeted():
    pairs = {}
    for n in tree().body:
        if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Tuple):
            pairs.update(zip([ast.unparse(x) for x in n.targets[0].elts],[ast.unparse(x) for x in n.value.elts],strict=True))
    assert {k:pairs[k] for k in ('_origin','_node_config','_node_files','_node_certificate')} == {k:'bs.'+k for k in ('_origin','_node_config','_node_files','_node_certificate')}
    assert pairs['_cached_targets'] == 'bq._cached_targets' and pairs['_retained'] == 'bq._retained'
    assert not any(isinstance(n,ast.Assign) and any(isinstance(t,ast.Attribute) for t in n.targets) for n in tree().body)


def test_late_retained_failure_clears_qualified_counts_and_preserves_receipt(tmp_path):
    science,value = copy.deepcopy(S),spec()
    science['candidate_folder'] = str(tmp_path/'hybrid25')
    science['matched_NODE_folder'] = str(tmp_path/'old-NODE25')
    output = tmp_path/'failed-prepare.json'
    seen = []

    def source(*args):
        args[-2]['original_recipe_origin'] = {'bound':'recipe'}
        return dict(model_initial='model0',source='source'),{},{}

    def cached(*args):
        args[-2]['BN_cache'] = {'historical':'BN74'}
        return {},'anchors','targets'

    def retained(*args):
        seen.append('retained')
        if len(seen) == 2: raise RuntimeError('late-retained-failure')

    cuda = types.SimpleNamespace(is_initialized=lambda:False,reset_peak_memory_stats=lambda:None,current_device=lambda:0,
        get_device_properties=lambda _:types.SimpleNamespace(total_memory=100),synchronize=lambda:None,
        max_memory_allocated=lambda:5,max_memory_reserved=lambda:10)
    fake_bq = types.SimpleNamespace(_source_buffer_digest=lambda _:'source',bc=types.SimpleNamespace(
        original=types.SimpleNamespace(CUDA_CAPACITY=100,_science=lambda _:dict(expected_success_counts={}))))
    fake_probe = types.SimpleNamespace(_native=lambda _:dict(device='cuda'),_digest=lambda _:'digest',
        _runtime_precision_guard=lambda:None,_checked_files=lambda _:None)
    env = extract('run',__file__=str(REPO/'src/cora_hybrid_fixed25.py'),
        time=types.SimpleNamespace(monotonic=lambda:1.),platform=types.SimpleNamespace(python_version=lambda:'3.12.7'),
        torch=types.SimpleNamespace(get_num_threads=lambda:4,cuda=cuda),bq=fake_bq,probe=fake_probe,
        _load_spec=lambda *args:(value,science),_request=lambda *args:output,_stop=lambda _:None,_load_source=source,
        _origin=lambda *args:{},_node_files=lambda _:{},_node_config=lambda *args:{},_node_certificate=lambda *args:None,
        _cached_targets=cached,_context=lambda *args:{},_hybrid_prepare=lambda *args:{},_retained=retained,
        _success_counts=lambda *args:S['counts_contract']['prepare'],_preserve=lambda *args:None,
        _write_new=lambda p,v:p.write_text(json.dumps(v)),_sha=lambda _:'pin')
    with pytest.raises(RuntimeError,match='late-retained-failure'):
        env['run']('prepare',70,'spec','sha',str(output))
    receipt = json.loads(output.read_text())
    assert receipt['passed'] is False and receipt['success_counts'] is None
    assert receipt['retained_buffer_preservation_error'] == "RuntimeError('late-retained-failure')"
    assert receipt['source_assets_spec_science_unchanged'] is True


def test_worker_exact_new_lazy_run_dispatch(monkeypatch):
    parsed = ast.parse(WORKER.read_text())
    dispatch = next(n for n in parsed.body if isinstance(n,ast.FunctionDef) and n.name == 'dispatch')
    branch = next(n for n in dispatch.body if isinstance(n,ast.If) and 'cora_hybrid_fixed25' in ast.unparse(n.test))
    mod = types.ModuleType('src.cora_hybrid_fixed25')
    marker,stop = object(),lambda:False
    mod.run = lambda **kwargs:(marker,kwargs)
    monkeypatch.setitem(sys.modules,'src',types.ModuleType('src'))
    monkeypatch.setitem(sys.modules,'src.cora_hybrid_fixed25',mod)
    mini = ast.FunctionDef(name='dispatch',args=dispatch.args,body=[dispatch.body[0],branch],decorator_list=[])
    env = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[mini],type_ignores=[])),str(WORKER),'exec'),env)
    result,options = env['dispatch'](dict(kind='cora_hybrid_fixed25',options=dict(operation='certify',cells=70)),stop)
    assert result is marker and options == dict(operation='certify',cells=70,stop=stop)
