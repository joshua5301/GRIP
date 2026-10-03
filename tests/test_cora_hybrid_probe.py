"""New hybrid wiring/spec/failure controls via AST and fakes; no Torch imports."""
import ast
import copy
import json
import math
import sys
import types
from pathlib import Path

import pytest

REPO = next(p for p in Path(__file__).resolve().parents if (p/'src/finite_student_outer_v2.py').is_file())
BASE = Path(__file__).resolve().parents[1]
SOURCE = BASE/'src/cora_hybrid_probe.py'
WORKER = BASE/'src/research_loop.py'
if not SOURCE.is_file():
    SOURCE, WORKER = REPO/'src/cora_hybrid_probe.py', REPO/'src/research_loop.py'
SCIENCE = REPO/'results/proposals/Cora70_half_teacher_CE_whole_layer_one_update_scientific_stageBU_v1.json'
SCIENCE_SHA = 'aec8aa3bb433e0ef398164684def190e5c5fe4e3eaeb781c4a284f3d55018ecb'
S = json.loads(SCIENCE.read_text())
POLICY = dict(rho=.001, layer_blocks=['concat(vec(W1),b1)', 'concat(vec(W2),b2)'], layer_weighting='equal1/2',
    anchor_weighting='equal1/3', anchors=3, accumulation='FP64', smoothing='symmetric',
    target_gradients='detached', clipping=False, renormalization=False, dropped_blocks=False)


def require(value, message):
    if not value:
        raise ValueError(message)


def tree():
    return ast.parse(SOURCE.read_text())


def function(name):
    return next(n for n in tree().body if isinstance(n, ast.FunctionDef) and n.name == name)


def extract(*names, **values):
    nodes = [function(n) for n in names]
    env = dict(Path=Path, math=math, json=json, _require=require,
        _exact=lambda a,b: type(a) is type(b) and a == b, _POLICY_SPEC=POLICY,
        SCIENCE=SCIENCE.name, SCIENCE_SHA=SCIENCE_SHA)
    fields = next(n for n in tree().body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_FIELDS' for t in n.targets))
    env['_FIELDS'] = ast.literal_eval(fields.value)
    env.update(values)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(SOURCE),'exec'),env)
    return env


def spec():
    v = {k:copy.deepcopy(S[k]) for k in ('fixed','case','output_root','output_path','raw_cache_path','BN_source')}
    files = dict(S['source_before']['files'])
    files.update({p:'owned' for p in S['source_protection']['owned'] if p.startswith('src/')})
    return dict(v,schema=1,source=dict(files=files,git_head='current82'),
        numerical_source=dict(versions=S['version_binding']['runtime_versions_from_pinned_native_metadata']),
        python_version=S['version_binding']['python_version'],files_sha256=S['original_files_sha256'],
        scientific_preregistration=dict(path=str(SCIENCE),sha256=SCIENCE_SHA),
        artifacts_sha256={str(REPO/'src/cora_hybrid_probe.py'):'owned'},grouping_policy=POLICY)


@pytest.mark.parametrize('mutation',['extra','bool_schema','oldsource','hybrid_weight','learned_normalizer'])
def test_native_hybrid_fixed_spec_fail_closed(mutation):
    v=spec()
    if mutation=='extra': v['resume']=True
    elif mutation=='bool_schema': v['schema']=True
    elif mutation=='oldsource': v['source']['files'].pop('src/cora_hybrid_probe.py')
    elif mutation=='hybrid_weight': v['fixed']['hybrid_weights']['teacher_outer_CE']=.4
    else: v['fixed']['normalizers']='learned'
    with pytest.raises(ValueError): extract('_spec_controls')['_spec_controls'](v,S,REPO)


def test_native_hybrid_exact_owned_paths_no_generic_source_waiver():
    v=spec()
    v['artifacts_sha256']={str(REPO/p):'pin' for p in S['source_protection']['owned']}
    extract('_spec_controls')['_spec_controls'](v,S,REPO)
    v['artifacts_sha256'][str(REPO/'src/soft_ce_partition.py')]='undeclared'
    with pytest.raises(ValueError): extract('_spec_controls')['_spec_controls'](v,S,REPO)


@pytest.mark.parametrize('bad',[0.,-1.,float('nan'),True])
def test_native_frozen_scales_reject_invalid_without_clamp(bad):
    f=extract('_normalizers')['_normalizers']
    with pytest.raises(ValueError): f(bad,1.)
    with pytest.raises(ValueError): f(1.,bad)
    assert f(.4,.8)==dict(F0=.4,G0=.8)


@pytest.mark.parametrize('existing',['receipt','raw'])
def test_exclusive_namespace_rejects_partial_or_completed_before_source(tmp_path,existing):
    v=spec();v.update(output_path=str(tmp_path/'receipt.json'),raw_cache_path=str(tmp_path/'raw.pt'))
    Path(v['output_path'] if existing=='receipt' else v['raw_cache_path']).touch()
    with pytest.raises(ValueError): extract('_request')['_request'](v['output_path'],v)


def test_protected_source_cache_and_retained_are_direct_private_aliases():
    pairs={}
    for n in tree().body:
        if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Tuple):
            pairs.update(zip([ast.unparse(x) for x in n.targets[0].elts],[ast.unparse(x) for x in n.value.elts],strict=True))
    assert pairs['_load_source']=='bc._load_source'
    assert pairs['_cached_targets']=='bq._cached_targets' and pairs['_retained']=='bq._retained'
    assert all(not (isinstance(n,ast.Assign) and any(isinstance(t,ast.Attribute) for t in n.targets)) for n in tree().body)
    assert 'source_gradient_targets' not in SOURCE.read_text()


def calls(node,name):
    return [n for n in ast.walk(node) if isinstance(n,ast.Call) and ast.unparse(n.func)==name]


def test_joint_native_solver_tolerances_and_implicit_minus_not_toy_policy():
    j=function('_joint')
    adj=calls(j,'solve_head_system')[0]
    assert {k.arg:ast.literal_eval(k.value) for k in adj.keywords}==dict(rtol=1e-6,atol=1e-12,max_iter=512,initial=None,cg_check_interval=1)
    assert len(calls(j,'head_gradient'))==len(calls(j,'outer_value_gradient'))==len(calls(j,'implicit_moment_gradient'))==len(calls(j,'moment_partials'))==1
    implicit=calls(j,'implicit_moment_gradient')[0]
    assert {k.arg:ast.literal_eval(k.value) for k in implicit.keywords}==dict(loss_weighting='uniform')
    teacher=calls(j,'outer_value_gradient')[0]
    assert ast.literal_eval(teacher.args[3])==65536
    gradient=next(n for n in ast.walk(j) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='gradient' for t in n.targets))
    assert ast.dump(gradient.value)==ast.dump(ast.parse('.5*Fpartial/scales["F0"] + .5*Gresult["moment_gradient"]/scales["G0"]',mode='eval').body)
    assert 'maximum <= 1e-07' in ast.unparse(j)
    assert len(calls(j,'moments.backward'))==0


def test_two_full_joint_states_one_first_backward_one_update_and_current_transform():
    one=function('_one_update')
    assert len(calls(one,'_joint'))==2 and len(calls(one,'moments0.backward'))==1 and len(calls(one,'optimizer.step'))==1
    backward=calls(one,'moments0.backward')[0]
    assert ast.unparse(backward.args[0])=="joint0['moment_gradient']"
    assert not calls(one,'moments1.backward')
    fit=calls(one,'solve_inner_newton_first')[0]
    assert {k.arg:ast.unparse(k.value) for k in fit.keywords}==dict(initial='theta0',max_iter='2000',grad_tol='1e-07',cg_max_iter='512',newton_steps='8',cg_check_interval='1')
    assert ast.unparse(calls(one,'representative')[0].args[1])=='probe._transform(buffers)'
    assert len(calls(one,'augmented'))==1
    assert len(calls(one,'inherited._adam_step'))==1 and len(calls(one,'inherited._optimizer'))==2
    first,second=calls(one,'_joint')
    assert ast.unparse(first.args[5])=='None' and ast.unparse(second.args[5])=='scales'
    assert '_normalizers(F, G)' in ast.unparse(function('_joint'))


def test_actual_success_counts_require_both_adjoints_terminal_head_and_only_one_P():
    raw=dict(P_update_completed=1,grouping_partial_completed=2,original_moment_backward_completed=1,
        accepted_BN_cache_load_completed=1,endpoint_head_solve_completed=1,adjoint_solve_completed=2,
        uniform_head_gradient_certificate_completed=2,teacher_outer_completed=2,P_updates=0)
    f=extract('_success_counts')['_success_counts']
    assert f(dict(counts=raw),S)==S['counts_contract']['planned_success']
    raw['adjoint_solve_completed']=1
    with pytest.raises(ValueError): f(dict(counts=raw),S)


@pytest.mark.parametrize('failure',['head','adjoint'])
def test_failed_native_joint_guard_blocks_downstream_implicit_and_grouping(failure):
    events=[]
    class Scalar:
        def __init__(self,value=0.): self.value=value
        def abs(self): return self
        def max(self): return self
        def norm(self): return self
        def all(self): return True
        def __float__(self): return float(self.value)
        def __sub__(self,other): return Scalar()
    device=types.SimpleNamespace(type='cuda')
    m=types.SimpleNamespace(device=device,dtype='f64',detach=lambda:'M')
    theta=types.SimpleNamespace(device=device,dtype='f64',requires_grad=False)
    buffers=dict(z=types.SimpleNamespace(shape=(6,3)),q='Q',transform='frozen')
    def count(e,key,complete=False): events.append((key,complete))
    def solver(*args,**kwargs): return Scalar(),dict(cg_converged=False,cg_residual=0.)
    fake_torch=types.SimpleNamespace(float64='f64',full_like=lambda m,v:'uniform',isfinite=lambda x:Scalar())
    env=extract('_joint',torch=fake_torch,PENALTY=.0001,decode_moments=lambda *a:('centers','labels',[1,2]),
        augmented=lambda x:'features',head_gradient=lambda *a:Scalar(1e-4 if failure=='head' else 0.),
        outer_value_gradient=lambda *a:(.5,Scalar(1.)),solve_head_system=solver,hessian_operator=lambda *a:(lambda v:Scalar(),None),
        _count=count,_stop=lambda s:None,implicit_moment_gradient=lambda *a,**k:events.append('IMPLICIT'),
        moment_partials=lambda *a:events.append('GROUPING'))
    with pytest.raises(ValueError): env['_joint'](m,buffers,theta,'anchors','targets',None,'sourcefeatures',dict(counts={}),lambda:False)
    assert 'IMPLICIT' not in events and 'GROUPING' not in events
    if failure=='head': assert ('adjoint_solve',False) not in events


def test_late_preservation_failure_clears_qualified_counts_retains_primary_and_raw(tmp_path):
    v=spec();v.update(output_path=str(tmp_path/'receipt.json'),raw_cache_path=str(tmp_path/'raw.pt'))
    calls_seen=[]
    def preserve(*a):
        calls_seen.append('preserve')
        if len(calls_seen)==2: raise RuntimeError('late-preservation-error')
    cuda=types.SimpleNamespace(is_initialized=lambda:False,reset_peak_memory_stats=lambda:None,current_device=lambda:0,
        get_device_properties=lambda _:types.SimpleNamespace(total_memory=100),synchronize=lambda:None,
        max_memory_allocated=lambda:5,max_memory_reserved=lambda:10)
    fake_torch=types.SimpleNamespace(get_num_threads=lambda:4,cuda=cuda,save=lambda p,stream:stream.write(b'raw'))
    fake_probe=types.SimpleNamespace(_native=lambda _:dict(device='cuda'),_runtime_precision_guard=lambda:None,
        _digest=lambda v:'digest',_attach=lambda v:dict(v,content_sha256='content'))
    fake_bc=types.SimpleNamespace(original=types.SimpleNamespace(CUDA_CAPACITY=100,_science=lambda _:dict(expected_success_counts={})))
    def cache(*a):
        a[-2]['BN_cache']={'historical_context':'old74'}
        return {},'anchors','targets'
    def one(*a):
        a[-2]['normalizers']=dict(F0=.5,G0=.2)
        return {0:dict(conservation={}),1:dict(conservation={})}
    env=extract('prepare',__file__=str(REPO/'src/cora_hybrid_probe.py'),time=types.SimpleNamespace(monotonic=lambda:1.),torch=fake_torch,
        platform=types.SimpleNamespace(python_version=lambda:'3.12.7'),probe=fake_probe,bc=fake_bc,bn=types.SimpleNamespace(_finite_tree=lambda _:None),
        _load_spec=lambda *a:(v,S),_request=lambda *a:(Path(v['output_path']),Path(v['raw_cache_path'])),
        _load_source=lambda *a:(dict(model_initial='model0',source='context'),{},{}),_source_buffer_digest=lambda b:'source',
        _cached_targets=cache,_one_update=one,_retained=lambda *a:None,_preserve=preserve,_stop=lambda s:None,
        _write_new=lambda p,x:p.write_text(json.dumps(x)),_sha=lambda p:'sha',_seal=lambda x:'seal',
        _success_counts=lambda *a:S['counts_contract']['planned_success'])
    with pytest.raises(RuntimeError,match='late-preservation-error'): env['prepare']('spec','sha',v['output_path'])
    receipt=json.loads(Path(v['output_path']).read_text())
    assert receipt['passed'] is False and receipt['success_counts'] is None
    assert receipt['source_assets_spec_science_unchanged'] is False
    assert Path(v['raw_cache_path']).read_bytes()==b'raw' and receipt['raw_cache_sha256']=='sha'


def test_new_worker_dispatch_exact_options_stop_and_return(monkeypatch):
    module=ast.parse(WORKER.read_text()); dispatch=next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='dispatch')
    branch=next(n for n in dispatch.body if isinstance(n,ast.If) and 'cora_hybrid_probe' in ast.unparse(n.test))
    fake=types.ModuleType('src.cora_hybrid_probe');marker=object();stop=lambda:False
    fake.prepare=lambda **kwargs:(marker,kwargs)
    monkeypatch.setitem(sys.modules,'src',types.ModuleType('src'))
    monkeypatch.setitem(sys.modules,'src.cora_hybrid_probe',fake)
    mini=ast.FunctionDef(name='dispatch',args=dispatch.args,body=[dispatch.body[0],branch],decorator_list=[])
    env={};exec(compile(ast.fix_missing_locations(ast.Module(body=[mini],type_ignores=[])),str(WORKER),'exec'),env)
    got,kwargs=env['dispatch'](dict(kind='cora_hybrid_probe',options=dict(spec_path='s',spec_sha256='h',output_path='o')),stop)
    assert got is marker and kwargs==dict(spec_path='s',spec_sha256='h',output_path='o',stop=stop)
