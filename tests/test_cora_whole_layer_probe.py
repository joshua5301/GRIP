"""New BQ AST/mock orchestration guards; no Torch/data/PT/numerical imports."""
import ast
import copy
import json
import sys
import types
from pathlib import Path

import pytest

REPO = next(p for p in Path(__file__).resolve().parents if (p/'src/finite_student_outer_v2.py').is_file())
BASE = Path(__file__).resolve().parents[1]
SOURCE = BASE/'src/cora_whole_layer_probe.py'
WORKER = BASE/'src/research_loop.py'
if not SOURCE.is_file():
    SOURCE, WORKER = REPO/'src/cora_whole_layer_probe.py', REPO/'src/research_loop.py'
SCIENCE = REPO/'results/proposals/Cora70_two_whole_layer_one_update_scientific_stageBQ_v1.json'
S = json.loads(SCIENCE.read_text())
POLICY = dict(rho=.001, layer_blocks=['concat(vec(W1),b1)', 'concat(vec(W2),b2)'], layer_weighting='equal1/2',
    anchor_weighting='equal1/3', anchors=3, accumulation='FP64', smoothing='symmetric',
    target_gradients='detached', clipping=False, renormalization=False, dropped_blocks=False)


def require(value, message):
    if not value:
        raise ValueError(message)


def extract(*names, **values):
    tree = ast.parse(SOURCE.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    env = dict(Path=Path, _require=require, _exact=lambda a,b: type(a) is type(b) and a == b,
               json=json, _POLICY_SPEC=POLICY, SCIENCE=SCIENCE.name, SCIENCE_SHA='ac99118497b6e4756d22f0e010da98e10a104921be2f03ed1553b1b371de784a')
    fields = next(n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_FIELDS' for t in n.targets))
    env['_FIELDS'] = ast.literal_eval(fields.value)
    env.update(values)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(SOURCE),'exec'),env)
    return env


def spec():
    v = {k:copy.deepcopy(S[k]) for k in ('fixed','case','output_root','output_path','raw_cache_path','BN_source')}
    files = dict(S['source_before']['files'])
    files.update({p:'owned' for p in S['source_protection']['owned'] if p.startswith('src/')})
    return dict(v,schema=1,source=dict(files=files,git_head='current78'),numerical_source=dict(versions=S['version_binding']['runtime_versions_from_pinned_native_metadata']),
        python_version=S['version_binding']['python_version'],files_sha256=S['original_files_sha256'],
        scientific_preregistration=dict(path=str(SCIENCE),sha256='ac99118497b6e4756d22f0e010da98e10a104921be2f03ed1553b1b371de784a'),
        artifacts_sha256={str(REPO/'src/cora_whole_layer_probe.py'):'owned'},grouping_policy=POLICY)


@pytest.mark.parametrize('mutation',['extra','bool_schema','group4','case','assets','oldsource','undeclared_artifact'])
def test_fixed_contract_rejects_mutations(mutation):
    v=spec()
    if mutation=='extra': v['resume']=True
    elif mutation=='bool_schema': v['schema']=True
    elif mutation=='group4': v['grouping_policy']=dict(POLICY,layer_weighting='equal1/4')
    elif mutation=='case': v['case']=dict(v['case'],cells=35)
    elif mutation=='assets': v['files_sha256']={}
    elif mutation=='oldsource': v['source']['files'].pop('src/cora_whole_layer_probe.py')
    else: v['artifacts_sha256']={str(REPO/'src/ce_gradient_alignment.py'):'old'}
    with pytest.raises(ValueError): extract('_spec_controls')['_spec_controls'](v,S,REPO)


def test_owned_paths_allowed_without_generic_repo_waiver():
    v=spec()
    v['artifacts_sha256']={str(REPO/p):'pin' for p in S['source_protection']['owned']}
    v['artifacts_sha256'][str(REPO/'results/proposals/example.json')]='pin'
    extract('_spec_controls')['_spec_controls'](v,S,REPO)


def cache_fixture():
    bindings=dict(source_backend='segment',target_caches={'2':{'path':'raw2','sha256':'pin2'}},spec={'sha256':'oldspec'})
    common=dict(digest='common',seal='seal')
    old=dict(source={'files':{'old74':'pin'}})
    row=dict(passed=True,source_assets_spec_science_unchanged=True,source_reference_provenance_passed=True,test_enabled=False,
             source_backend='segment',common_numerical_digest='common',common_numerical_seal='seal')
    second=dict(row,within_three_anchor_segment_exact_repeat_passed=True,three_anchor_segment_backend_qualified=True,
                raw_cache_path='raw2',raw_cache_sha256='pin2',spec_sha256='oldspec',source=old['source'])
    context={'historical74':old['source']}
    payload=dict(pass_index=2,context=context,common=common)
    env=extract('_cache_metadata',bn=types.SimpleNamespace(_context=lambda a,b:context),probe=types.SimpleNamespace(_digest=lambda x:x['digest']),_seal=lambda x:x['seal'])
    return env,payload,row,second,old,bindings


@pytest.mark.parametrize('mutation',['bool_pass','unqualified_repeat','context','common','rawsha'])
def test_accepted_cache_metadata_fail_closed(mutation):
    env,payload,first,second,old,bindings=cache_fixture()
    if mutation=='bool_pass': payload['pass_index']=True
    elif mutation=='unqualified_repeat': second['three_anchor_segment_backend_qualified']=False
    elif mutation=='context': payload['context']={'current78':True}
    elif mutation=='common': payload['common']['digest']='changed'
    else: second['raw_cache_sha256']='changed'
    with pytest.raises(ValueError): env['_cache_metadata'](payload,first,second,old,bindings)


def test_historical_context_stays_separate_from_current_source():
    env,payload,first,second,old,bindings=cache_fixture()
    env['_cache_metadata'](payload,first,second,old,bindings)
    assert old['source'] != spec()['source']


@pytest.mark.parametrize('mutation',['source','target'])
def test_retained_failure_cleanup_detects_mutation(mutation):
    buffers=dict(model_initial='model0',source='source')
    payload=dict(common='common')
    evidence=dict(source_buffer_digest_before='source',anchor0_digest_before='model0',
        BN_cache=dict(common_digest='common',anchor_digest_before='anchors',target_digest_before='targets'))
    if mutation=='source': buffers['source']='changed'
    else: targets='changed'
    env=extract('_retained',_source_buffer_digest=lambda b:b['source'],probe=types.SimpleNamespace(_digest=lambda v:v))
    with pytest.raises(ValueError): env['_retained'](buffers,payload,'anchors',targets if mutation=='target' else 'targets',evidence)


def test_success_count_uses_completed_interfaces_and_no_raw_legacy_zero():
    counts=dict(P_update_completed=1,grouping_partial_completed=2,original_moment_backward_completed=1,
                accepted_BN_cache_load_completed=1,P_updates=0)
    f=extract('_success_counts')['_success_counts']
    result=f(dict(counts=counts),S)
    assert result==S['counts_contract']['planned_success']
    counts['grouping_partial_completed']=1
    with pytest.raises(ValueError): f(dict(counts=counts),S)


def test_new_route_calls_grouping_not_old_four_block_or_source_target_generator():
    tree=ast.parse(SOURCE.read_text())
    one=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_one_update')
    calls=[ast.unparse(n.func) for n in ast.walk(one) if isinstance(n,ast.Call)]
    assert calls.count('moment_partials')==2 and calls.count('moments0.backward')==1 and calls.count('optimizer.step')==1
    assert 'source_gradient_targets' not in SOURCE.read_text() and 'gradient_alignment_partials' not in SOURCE.read_text()
    targets=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_cached_targets')
    assert sum(isinstance(n,ast.Call) and ast.unparse(n.func)=='torch.load' for n in ast.walk(targets))==1
    assert '["target"]["gradients"]' in ast.unparse(targets).replace("'",'"')


@pytest.mark.parametrize('failure',['cache','source_and_finalization','fresh_cuda'])
def test_prepare_primary_failure_preserved_and_no_dependent_update(tmp_path,failure):
    events=[]
    v=spec()
    v.update(output_path=str(tmp_path/'receipt.json'),raw_cache_path=str(tmp_path/'raw.pt'))
    def load_source(*a):
        events.append('source')
        if failure=='source_and_finalization': raise LookupError('primary-source-error')
        return dict(model_initial='model0'),{},{}
    def cached(*a):
        events.append('cache')
        raise LookupError('primary-cache-error')
    def sync():
        events.append('sync')
        if failure=='source_and_finalization': raise RuntimeError('secondary-sync-error')
    cuda=types.SimpleNamespace(is_initialized=lambda:failure=='fresh_cuda',reset_peak_memory_stats=lambda:None,
        current_device=lambda:0,get_device_properties=lambda _:types.SimpleNamespace(total_memory=100),synchronize=sync,
        max_memory_allocated=lambda:5,max_memory_reserved=lambda:10)
    fake_torch=types.SimpleNamespace(get_num_threads=lambda:4,cuda=cuda)
    fake_probe=types.SimpleNamespace(_native=lambda _:dict(device='cuda'),_runtime_precision_guard=lambda:None,_digest=lambda x:x)
    fake_bc=types.SimpleNamespace(original=types.SimpleNamespace(CUDA_CAPACITY=100,_science=lambda _:dict(expected_success_counts={})))
    env=extract('prepare',__file__=str(REPO/'src/cora_whole_layer_probe.py'),time=types.SimpleNamespace(monotonic=lambda:1.),torch=fake_torch,
        platform=types.SimpleNamespace(python_version=lambda:'3.12.7'),probe=fake_probe,bc=fake_bc,
        _load_spec=lambda *a:(v,S),_request=lambda *a:(Path(v['output_path']),Path(v['raw_cache_path'])),_load_source=load_source,
        _source_buffer_digest=lambda x:'source',_cached_targets=cached,_one_update=lambda *a:events.append('UPDATE'),
        _retained=lambda *a:events.append('retained'),_preserve=lambda *a:None,_stop=lambda f:require(not f(),'stop'),
        _write_new=lambda p,x:p.write_text(json.dumps(x)),_sha=lambda _: 'sha')
    error=ValueError if failure=='fresh_cuda' else LookupError
    with pytest.raises(error): env['prepare']('spec','sha',v['output_path'])
    r=json.loads(Path(v['output_path']).read_text())
    assert r['passed'] is False and 'UPDATE' not in events and r['source_assets_spec_science_unchanged'] is True
    if failure=='source_and_finalization':
        assert r['error']=='primary-source-error' and 'secondary-sync-error' in r['native_finalization_error']
    if failure=='fresh_cuda': assert 'source' not in events


def test_worker_lazy_dispatch_preserves_options_callback_and_return(monkeypatch):
    tree=ast.parse(WORKER.read_text())
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='dispatch')
    branch=next(n for n in function.body if isinstance(n,ast.If) and 'cora_whole_layer_probe' in ast.unparse(n.test))
    fake=types.ModuleType('src.cora_whole_layer_probe')
    marker=object()
    callback=lambda:False
    fake.prepare=lambda **kwargs:(marker,kwargs)
    monkeypatch.setitem(sys.modules,'src',types.ModuleType('src'))
    monkeypatch.setitem(sys.modules,'src.cora_whole_layer_probe',fake)
    env={}
    mini=ast.FunctionDef(name='dispatch',args=function.args,body=[function.body[0],branch],decorator_list=[])
    exec(compile(ast.fix_missing_locations(ast.Module(body=[mini],type_ignores=[])),str(WORKER),'exec'),env)
    result,kwargs=env['dispatch'](dict(kind='cora_whole_layer_probe',options=dict(spec_path='s',spec_sha256='h',output_path='o')),callback)
    assert result is marker and kwargs==dict(spec_path='s',spec_sha256='h',output_path='o',stop=callback)
