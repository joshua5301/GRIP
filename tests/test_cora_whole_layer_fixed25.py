"""BR owned metadata/coupled-prefix control mocks only; no numerical imports."""
import ast
import copy
import hashlib
import json
import sys
import types
from pathlib import Path

import pytest

REPO = next(p for p in Path(__file__).resolve().parents if (p/'src/cora_whole_layer_probe.py').is_file())
BASE = Path(__file__).resolve().parents[1]
SOURCE, WORKER = BASE/'src/cora_whole_layer_fixed25.py', BASE/'src/research_loop.py'
if not SOURCE.is_file(): SOURCE,WORKER=REPO/'src/cora_whole_layer_fixed25.py',REPO/'src/research_loop.py'
SCIENCE=REPO/'results/proposals/Cora70_two_whole_layer_fixed25_matched_NODE_scientific_stageBR_v2.json'
S=json.loads(SCIENCE.read_text())


def require(v,m):
    if not v: raise ValueError(m)


def normalize(v):
    if isinstance(v,A): return v.value
    if isinstance(v,dict): return {k:normalize(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)): return [normalize(x) for x in v]
    return v


def seal(v):
    return hashlib.sha256(json.dumps(normalize(v),sort_keys=True).encode()).hexdigest()


def extract(*names,**values):
    tree=ast.parse(SOURCE.read_text()); nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    fields=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_FIELDS' for t in n.targets))
    env=dict(Path=Path,json=json,_require=require,_exact=lambda a,b:normalize(a)==normalize(b),_seal=seal,
        SCIENCE=SCIENCE.name,SCIENCE_SHA='1cf9d80ded2415fb27e91e9605fdc03bfd6e7575cd87137e759cd0a7eea288ee',_FIELDS=ast.literal_eval(fields.value),_POLICY_SPEC=S['grouping_policy'],SEEDS=(4400,4401,4402))
    env.update(values); exec(compile(ast.Module(body=nodes,type_ignores=[]),str(SOURCE),'exec'),env); return env


def spec():
    v={k:copy.deepcopy(S[k]) for k in ('fixed','case','candidate','candidate_id','matched_NODE_candidate','output_root','certificate_outputs','student_seeds')}
    files=dict(S['source_before']['files']);files.update({p:'owned' for p in S['source_protection']['owned'] if p.startswith('src/')})
    return dict(v,schema=1,source=dict(files=files),numerical_source=dict(versions=S['version_binding']['runtime_versions_from_pinned_native_metadata']),
        python_version=S['version_binding']['python_version'],files_sha256=S['original_files_sha256'],grouping_policy=S['grouping_policy'],artifacts_sha256={str(REPO/'src/cora_whole_layer_fixed25.py'):'own'},
        scientific_preregistration=dict(path=str(SCIENCE),sha256='1cf9d80ded2415fb27e91e9605fdc03bfd6e7575cd87137e759cd0a7eea288ee'))


@pytest.mark.parametrize('mutation',['source_augmented','NODE_lr','extra','boolschema','oldcohort','unknown_artifact'])
def test_fixed_plain_identity_and_matched_source_controls(mutation):
    v=spec()
    if mutation=='source_augmented': v['candidate']['whole_layer_source_digest']='new'
    elif mutation=='NODE_lr': v['matched_NODE_candidate']['lr']=.05
    elif mutation=='extra': v['resume']='BQ'
    elif mutation=='boolschema': v['schema']=True
    elif mutation=='oldcohort': v['student_seeds']=[4100,4101,4102]
    else: v['artifacts_sha256']={str(REPO/'src/ce_gradient_alignment.py'):'old'}
    f=extract('_spec_controls',_fingerprint=lambda v:S['candidate_id'])['_spec_controls']
    with pytest.raises(ValueError): f(v,S,REPO)


def test_plain_candidate_all_owned_paths_valid():
    v=spec();v['artifacts_sha256']={str(REPO/p):'pin' for p in S['source_protection']['owned']}
    extract('_spec_controls',_fingerprint=lambda v:S['candidate_id'])['_spec_controls'](v,S,REPO)


def test_matched_core_nested_kwargs_exact_no_disposable_resume(tmp_path):
    v=spec();s=copy.deepcopy(S);s['matched_NODE_folder']=str(tmp_path/'matched')
    s['matched_NODE_core_call']['kwargs']['folder']=s['matched_NODE_folder']
    seen=[];counts={}
    def core(*args,**kwargs): seen.append((args,kwargs))
    def count(e,k,completed=False): e['counts'][k+('_completed' if completed else '_attempts')]=1
    e=dict(counts=counts)
    env=extract('_node_prepare',_stop=lambda f:None,inherited=types.SimpleNamespace(_count=count),optimize_ce_assignment=core,
        torch=types.SimpleNamespace(load=lambda *a,**kw:dict(step=25)),_node_config=lambda *a:{},_node_certificate=lambda *a:seen.append('qualified'))
    env['_node_prepare'](v,s,dict(z='z',q='q',hard='hard'),{},e,lambda:False)
    args,kwargs=seen[0]; assert args==('z','q','hard') and seen[1]=='qualified'
    expected=dict(s['matched_NODE_core_call']['kwargs'],folder=Path(s['matched_NODE_folder']),checkpoint_steps=(0,25),stop=kwargs['stop'])
    assert kwargs==expected and kwargs['lr']==.01 and kwargs['factor_seed']==0 and kwargs['resume_state'] is None
    assert e['counts']['matched_NODE_P_updates_completed']==25


def test_partial_namespace_cannot_resume(tmp_path):
    s=copy.deepcopy(S);s['candidate_folder']=str(tmp_path/'gradient');s['matched_NODE_folder']=str(tmp_path/'node')
    s['operation_outputs']['prepare']=str(tmp_path/'prepare.json');Path(s['matched_NODE_folder']).mkdir()
    with pytest.raises(ValueError): extract('_request')['_request']('prepare',70,None,None,s['operation_outputs']['prepare'],{},s)


class A:
    shape=(1,)
    def __init__(self,v): self.value=v
    def detach(self): return self
    def clone(self): return A(self.value)
    def requires_grad_(self,v=True): return self
    def to(self,other): return self


def progress_fixture():
    context={'bound_source':'current80'};targets='targets';states={}
    for step in range(26):
        states[step]=dict(step=step,parameters=[A(step),A(step+10)],optimizer=dict(counter=step),moments=A(5),alignment={'loss':2.},J0=2.,conservation={},terminal_no_update=step==25)
        if step<25:states[step]['scaled_factor_gradients']=[A(1),A(1)]
    history=lambda states:[dict(step=k) for k in states]
    bundle=dict(schema=1,context=context,frontier=25,states=states,J0=2.,history=history(states),target_digest='targets')
    bundle['content_sha256']=seal(bundle)
    calls=[]
    def optimizer(saved,parameters,first,second,step):
        require(saved['counter']==step,'Adam intermediate counter differs');calls.append(step)
    def adam(p,m,v,g,step):return [A(x.value+1) for x in p],m,v
    def load(path,**kwargs):
        if Path(path).name=='progress.pt':return bundle
        step=0 if '000000' in Path(path).name else 25
        return {'context':context,'state':states[step]}
    fake_torch=types.SimpleNamespace(float32='float32',load=load,zeros_like=lambda p:A(0),equal=lambda a,b:a.value==b.value)
    probe=types.SimpleNamespace(_tensor=lambda a,*args:a,_digest=lambda x:x,_attach=lambda x:x)
    env=extract('_load_progress',torch=fake_torch,probe=probe,bq=types.SimpleNamespace(bn=types.SimpleNamespace(bi=types.SimpleNamespace(_sealed_payload=lambda x:x))),
        _gradient_files=lambda f:{'pins':'fixed'},_state=lambda b,p,o,step,scale,t,u,e,stop:(states[step],2.),_history=history,_stop=lambda s:None,
        inherited=types.SimpleNamespace(_optimizer=optimizer,_adam_step=adam),_bump=lambda *args:None)
    # candidate/history are ordinary metadata only.
    return env,bundle,states,context,targets,calls


@pytest.mark.parametrize('mutation',['missing_middle','middle_counter','middle_parameter','frontier','target_digest'])
def test_complete_prefix_each_actual_Adam_state_required(tmp_path,mutation):
    env,bundle,states,context,targets,calls=progress_fixture()
    folder=tmp_path/'gradient';folder.mkdir();(folder/'candidate.json').write_text('{}');(folder/'history.json').write_text(json.dumps(bundle['history']))
    context['spec']={'candidate':{}}
    if mutation=='missing_middle':states.pop(13)
    elif mutation=='middle_counter':states[13]['optimizer']['counter']=12
    elif mutation=='middle_parameter':states[13]['parameters'][0]=A(90)
    elif mutation=='frontier':bundle['frontier']=24
    else:bundle['target_digest']='changed'
    with pytest.raises(ValueError):env['_load_progress']({'candidate_folder':str(folder)},dict(initial=[A(0),A(10)]),dict(moments=A(5)),context,None,targets,dict(operation='certify'),lambda:False)


def test_all26_counters_and_no_terminal_update_contract(tmp_path):
    env,bundle,states,context,targets,calls=progress_fixture()
    folder=tmp_path/'gradient';folder.mkdir();(folder/'candidate.json').write_text('{}');(folder/'history.json').write_text(json.dumps(bundle['history']));context['spec']={'candidate':{}}
    env['_load_progress']({'candidate_folder':str(folder)},dict(initial=[A(0),A(10)]),dict(moments=A(5)),context,None,targets,{},lambda:False)
    assert calls==list(range(26)) and 'scaled_factor_gradients' not in states[25]


def test_terminal_partial_and_exact_native_count_structure():
    tree=ast.parse(SOURCE.read_text());state=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_state')
    backwards=[n for n in ast.walk(state) if isinstance(n,ast.Call) and ast.unparse(n.func)=='moments.backward']
    assert len(backwards)==1 and any(isinstance(n,ast.If) and ast.unparse(n.test)=='step < 25' and backwards[0] in list(ast.walk(n)) for n in ast.walk(state))
    prepare=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_gradient_prepare')
    assert 'range(26)' in ast.unparse(prepare) and ast.unparse(prepare).count('optimizer.step()')==1
    assert 'bq._cached_targets' in SOURCE.read_text() and 'gradient_alignment_partials' not in SOURCE.read_text() and 'source_gradient_targets(' not in SOURCE.read_text()


def test_endpoint_numeric_AST_only_lr_and_receipt_instrumentation_changed():
    old=ast.parse((REPO/'src/cora_node_reference_certificate_v2.py').read_text());new=ast.parse(SOURCE.read_text())
    a=copy.deepcopy(next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name=='_endpoints'))
    b=copy.deepcopy(next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name=='_node_certificate'))
    b.name=a.name
    class Normalize(ast.NodeTransformer):
        def visit_Call(self,n):
            self.generic_visit(n)
            if isinstance(n.func,ast.Attribute) and ast.unparse(n.func)=='inherited._count':n.func=ast.Name(id='_count',ctx=ast.Load())
            if isinstance(n.func,ast.Name) and n.func.id=='representative' and len(n.args)==4 and ast.unparse(n.args[1])=='probe._transform(buffers)':
                n.args[1]=ast.parse("buffers['transform']",mode='eval').body
            return n
        def visit_Compare(self,n):
            self.generic_visit(n)
            if ast.unparse(n.left)=="group['lr']":n.comparators=[ast.Constant(value=.05)]
            return n
    b=Normalize().visit(b)
    a.body=a.body[:-1];b.body=b.body[:-1]
    assert ast.dump(a)==ast.dump(b)


def test_worker_lazy_dispatch_callback_options_and_return(monkeypatch):
    tree=ast.parse(WORKER.read_text());f=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='dispatch')
    branch=next(n for n in f.body if isinstance(n,ast.If) and 'cora_whole_layer_fixed25' in ast.unparse(n.test))
    mod=types.ModuleType('src.cora_whole_layer_fixed25');marker=object();callback=lambda:False
    mod.run=lambda **kw:(marker,kw);monkeypatch.setitem(sys.modules,'src',types.ModuleType('src'));monkeypatch.setitem(sys.modules,'src.cora_whole_layer_fixed25',mod)
    env={};mini=ast.FunctionDef(name='dispatch',args=f.args,body=[f.body[0],branch],decorator_list=[])
    exec(compile(ast.fix_missing_locations(ast.Module(body=[mini],type_ignores=[])),str(WORKER),'exec'),env)
    result,kw=env['dispatch'](dict(kind='cora_whole_layer_fixed25',options=dict(operation='certify',cells=70)),callback)
    assert result is marker and kw==dict(operation='certify',cells=70,stop=callback)


def test_failed_core_interface_does_not_invent_completed25(tmp_path):
    science=copy.deepcopy(S);science['matched_NODE_folder']=str(tmp_path/'node');science['matched_NODE_core_call']['kwargs']['folder']=science['matched_NODE_folder']
    evidence=dict(counts={})
    def count(e,k,completed=False):e['counts'][k+('_completed' if completed else '_attempts')]=1
    def fail(*a,**k):raise RuntimeError('core-primary-error')
    f=extract('_node_prepare',_stop=lambda x:None,inherited=types.SimpleNamespace(_count=count),optimize_ce_assignment=fail)['_node_prepare']
    with pytest.raises(RuntimeError,match='core-primary-error'):f({},science,dict(z='z',q='q',hard='hard'),{},evidence,lambda:False)
    assert evidence['counts']==dict(matched_NODE_core_attempts=1) and 'observed_matched_NODE_durable_frontier' not in evidence


@pytest.mark.parametrize('mutation',['illegal_arm','float_cells','wrong_operation'])
def test_phase_API_before_native_access(tmp_path,mutation):
    science=copy.deepcopy(S);science['operation_outputs']['certify']=str(tmp_path/'cert.json')
    operation,cells,arm,gate='certify',70,None,None
    if mutation=='illegal_arm':arm='whole_layer'
    elif mutation=='float_cells':cells=70.
    else:operation='resume'
    with pytest.raises(ValueError):extract('_request')['_request'](operation,cells,arm,gate,science['operation_outputs']['certify'],{},science)


# StageBS adds only the four frozen-transform interface controls below.
FROZEN_BR = REPO / "results/implementation_drafts/cora_whole_layer_fixed25_v1/proposed/src/cora_whole_layer_fixed25.py"
FUNCTIONS = ("_origin", "_node_certificate", "_validate")


def representative_calls(path):
    calls = {}
    for fn in ast.parse(path.read_text()).body:
        if isinstance(fn, ast.FunctionDef):
            selected = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
                        and isinstance(n.func, ast.Name) and n.func.id == "representative"]
            if selected:
                assert len(selected) == 1
                calls[fn.name] = selected[0]
    return calls


def bridge():
    # Execute only the unchanged constructor and bridge AST, with its local import
    # replaced by the injected constructor. No production module is imported.
    tree = ast.parse((REPO / "src/transforms.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FeatureTransform")
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    cls.body = [init]
    tree = ast.parse((REPO / "src/finite_student_probe.py").read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_transform")
    assert isinstance(fn.body[0], ast.ImportFrom) and fn.body[0].module == "src.transforms"
    fn.body = fn.body[1:]
    env = {}
    exec(compile(ast.Module(body=[cls, fn], type_ignores=[]), "unchanged_transform_constructor_AST", "exec"), env)
    return env["_transform"]


def test_exactly_three_readout_bridges_and_no_other_function_change():
    calls = representative_calls(SOURCE)
    assert tuple(calls) == FUNCTIONS
    tree = ast.parse(SOURCE.read_text())
    for call in calls.values():
        assert len(call.args) == 4 and not call.keywords
        assert ast.unparse(call.args[1]) == "probe._transform(buffers)"
        assert ast.literal_eval(call.args[2]) == 1433 and ast.literal_eval(call.args[3]) == "cuda"
    class Undo(ast.NodeTransformer):
        def visit_Call(self, node):
            self.generic_visit(node)
            if isinstance(node.func, ast.Name) and node.func.id == "representative":
                node.args[1] = ast.parse("buffers['transform']", mode="eval").body
            return node
    tree = Undo().visit(tree)
    # Science/peer constants are the only other permitted new-BS changes.
    constants = {"SCIENCE", "SCIENCE_SHA", "SCIENCE_REVIEW", "SCIENCE_REVIEW_SHA"}
    def body_without_pins(tree):
        return [n for n in tree.body if not (isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id in constants for t in n.targets))]
    assert ast.dump(ast.Module(body=body_without_pins(tree), type_ignores=[])) == ast.dump(
        ast.Module(body=body_without_pins(ast.parse(FROZEN_BR.read_text())), type_ignores=[]))


@pytest.mark.parametrize("function", FUNCTIONS)
def test_each_live_readout_call_wraps_same_frozen_mapping(function):
    frozen = dict(center=object(), matrix=None, output_center=object(), scale=object(), kind="rms", eps=1e-12)
    before = dict(frozen)
    moments = object()
    seen = []
    def readout(value, transform, dimension, device):
        # This is the actual representative object's attribute contract; no
        # moment decoding, numerical arithmetic, tensor or device call occurs.
        seen.append((value, transform, dimension, device))
        assert transform.scale is frozen["scale"]
        assert transform.output_center is frozen["output_center"]
        assert transform.center is frozen["center"]
        return ("FP32_X", "FP32_Q", "FP64_mass")
    env = dict(buffers={"transform": frozen}, moments=moments, endpoint=moments,
               probe=types.SimpleNamespace(_transform=bridge()), representative=readout)
    call = representative_calls(SOURCE)[function]
    result = eval(compile(ast.Expression(body=call), "actual_representative_call_AST", "eval"), env)
    assert result == ("FP32_X", "FP32_Q", "FP64_mass")
    assert len(seen) == 1 and seen[0][0] is moments and seen[0][2:] == (1433, "cuda")
    assert seen[0][1].matrix is None and seen[0][1].kind == "rms" and seen[0][1].eps == 1e-12
    assert frozen == before and all(frozen[key] is before[key] for key in frozen)
    # Retain the observed failure shape for the historical direct-dict call.
    with pytest.raises(AttributeError):
        readout(moments, frozen, 1433, "cuda")
