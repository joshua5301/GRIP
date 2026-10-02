"""AST-only fixed-Citeseer controls/cache/phase tests; no production imports.

Tensor/native/evaluator operations below are explicit fakes. Numerical engine,
CUDA Adam ordering and real source qualification remain separate native gates.
"""
import ast
import copy
import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

BASE = next(parent for parent in Path(__file__).resolve().parents if (parent / 'src/citeseer_finite_student.py').is_file())
MODULE = BASE / 'src/citeseer_finite_student.py'
WORKER = BASE / 'src/research_loop.py'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def seal(value):
    def plain(item):
        if isinstance(item, Tensor):
            return dict(data=item.data, dtype=item.dtype, device=item.device.type)
        if isinstance(item, dict):
            return {str(k): plain(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [plain(x) for x in item]
        return item
    return hashlib.sha256(json.dumps(plain(value), sort_keys=True).encode()).hexdigest()


class Tensor:
    def __init__(self, data, dtype='f32', device='cpu'):
        self.data, self.dtype = data, dtype
        self.device = SimpleNamespace(type=device)
        self.shape = () if isinstance(data, (int, float)) else (len(data),)

    def detach(self):
        return self

    def clone(self):
        return Tensor(copy.deepcopy(self.data), self.dtype, self.device.type)

    def to(self, other):
        return self

    def __float__(self):
        return float(self.data)


def tensor(value, shape, dtype, message):
    require(isinstance(value, Tensor) and value.shape == shape and value.dtype == dtype, message)
    require(all(math.isfinite(x) for x in (value.data if isinstance(value.data, list) else [value.data])), message)
    return value


def attach(value):
    result = copy.deepcopy(value)
    result['content_sha256'] = seal(value)
    return result


def load(names):
    parsed = ast.parse(MODULE.read_text())
    definitions = [n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name in names]
    constants = [n for n in parsed.body if isinstance(n, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id in {'SCHEMA', 'HORIZON', 'SCIENCE', 'SCIENCE_SHA', 'SOURCE_FIELD', '_FIXED', '_SEEDS', '_SETTINGS'} for t in n.targets)]
    ns = dict(__file__=str(MODULE), math=math, json=json, Path=Path, _require=require, _seal=seal,
              _fingerprint=lambda x: seal(x)[:12], numerical_source=lambda: {'frozen': 1},
              _tensor=tensor, _stop=lambda stop: require(not stop(), 'stopped'), _attach=attach,
              torch=SimpleNamespace(float32='f32', is_tensor=lambda x: isinstance(x, Tensor),
                  zeros_like=lambda x: Tensor([0.] * len(x.data)), equal=lambda a, b: seal(a) == seal(b)),
              original=SimpleNamespace(REFERENCE='a2d47970c967', _roots=lambda repo: [dict(cells=c, root=str(c)) for c in (30, 120)]))
    exec(compile(ast.Module(constants+definitions, type_ignores=[]), str(MODULE), 'exec'), ns)
    return ns


@pytest.mark.parametrize('change', [dict(lr=.05), dict(T=.3), dict(rank=32), dict(penalty=.0001),
    dict(outer_route='sgc_mlp'), dict(finite_assignment_steps=1), dict(finite_student_schema=True),
    dict(width=256), dict(mass_mode='initial'), dict(mixing=.1), dict(finite_model_steps=6),
    dict(external_F=[]), dict(teacher_kernel='ntk'), dict(finite_student_source_digest='stale')])
def test_fixed_unknown_or_changed_controls_reject(change):
    ns = load({'canonical_candidate'})
    with pytest.raises(ValueError):
        ns['canonical_candidate'](dict(ns['_FIXED'], **change))


def test_candidate_matches_frozen_science_and_same_source_namespace():
    ns = load({'canonical_candidate'})
    science = next(parent for parent in MODULE.parents if (parent / 'results/proposals').is_dir()) / 'results/proposals' / ns['SCIENCE']
    assert json.loads(science.read_text())['candidate'] == ns['_FIXED']
    a = ns['canonical_candidate'](ns['_FIXED'])
    assert ns['canonical_candidate'](a) == a
    assert 'target' not in a and 'frontier' not in a
    assert a['finite_assignment_steps'] == 25 and a['outer_route'] == 'gcn'


@pytest.mark.parametrize('value', [False, 0, 2, 1.5, 26, '25'])
def test_invalid_phase_rejects(value):
    with pytest.raises(ValueError):
        load({'_phase'})['_phase'](value)


@pytest.mark.parametrize('cells,epochs,lr,dropout,wd', [(30,500,.003,.3,5e-5), (120,1000,.001,0.,.0005)])
def test_supported_evaluator_settings_without_input_scale(cells, epochs, lr, dropout, wd):
    ns = load({'_budget', '_settings'})
    settings = ns['_settings'](cells)
    assert settings == dict(epochs=epochs, eval_every=1, hidden=256, lr=lr, dropout=dropout,
                           weight_decay=wd, lr_schedule='constant', initialization='geom_uniform')
    assert 'input_scale' not in settings


def fixture_bundle(ns, frontier=3):
    initial = [Tensor([0., 0.]), Tensor([1., 2.])]
    def evaluate(buffers, candidate, context, parameters, step, scale, stop):
        value = 2. + step/10
        scale = value if scale is None else scale
        return dict(schema=1, context=context, step=step, parameters=copy.deepcopy(parameters),
                    teacher_ce=value, scale=scale, scaled_factor_gradients=[Tensor([.2,.3]),Tensor([.4,.5])],
                    model='actual-fake', moment_partial='actual-fake'), scale
    def recurrence(parameters, first, second, gradients, step):
        p = [Tensor([x-y for x,y in zip(a.data,g.data,strict=True)]) for a,g in zip(parameters,gradients,strict=True)]
        m = [Tensor([float(step)]*len(a.data)) for a in parameters]
        v = [Tensor([float(step*2)]*len(a.data)) for a in parameters]
        return p,m,v
    def optimizer(parameters, first, second, step):
        return dict(state={} if step == 0 else {i: dict(step=Tensor(float(step)),exp_avg=m,exp_avg_sq=v)
            for i,(m,v) in enumerate(zip(first,second,strict=True))},param_groups=[dict(lr=.01,betas=[.9,.999],eps=1e-12,
            weight_decay=0,amsgrad=False,maximize=False,foreach=False,capturable=False,differentiable=False,fused=False,params=[0,1])])
    ns.update(_evaluate=evaluate,_adam_step=recurrence,_history=lambda s: dict(step=s['step'],J=s['teacher_ce']))
    p=copy.deepcopy(initial); m=v=[Tensor([0.,0.]),Tensor([0.,0.])]; snapshots={}
    for step in range(frontier+1):
        if step:
            p,m,v=recurrence(p,m,v,snapshots[step-1]['scaled_factor_gradients'],step)
        snapshot,scale=evaluate({}, {}, {}, p,step,None if step == 0 else 2.,lambda:False)
        snapshot['optimizer']=optimizer(p,m,v,step)
        snapshots[step]=snapshot
    bundle=attach(dict(schema=1,context={},step=frontier,parameters=p,optimizer=snapshots[frontier]['optimizer'],
                       snapshots=snapshots,scale=2.,history=[ns['_history'](s) for s in snapshots.values()]))
    return bundle,dict(initial=initial)


@pytest.mark.parametrize('corruption', ['mid_counter','mid_moment','terminal_slot','coupled_J0','partial_snapshots','bool_frontier','model_partial'])
def test_resealed_corruption_fails_even_with_latest_optimizer_valid(corruption):
    ns=load({'_validate_bundle','_optimizer'})
    bundle,buffers=fixture_bundle(ns)
    assert ns['_validate_bundle'](bundle,buffers,{}, {},lambda:False) == 4
    bad=copy.deepcopy(bundle)
    if corruption=='mid_counter': bad['snapshots'][1]['optimizer']['state'][0]['step']=Tensor(True, dtype='bool')
    elif corruption=='mid_moment': bad['snapshots'][1]['optimizer']['state'][0]['exp_avg']=Tensor([9.,9.])
    elif corruption=='terminal_slot': bad['optimizer']['state'][1]['exp_avg_sq']=Tensor([9.,9.])
    elif corruption=='coupled_J0': bad['scale']=2.1; bad['history'][0]['J']=2.1; bad['snapshots'][0]['teacher_ce']=2.1
    elif corruption=='partial_snapshots': del bad['snapshots'][1]
    elif corruption=='bool_frontier': bad['step']=True
    else: bad['snapshots'][2]['moment_partial']='forged'
    bad=attach({k:v for k,v in bad.items() if k!='content_sha256'})
    with pytest.raises(ValueError): ns['_validate_bundle'](bad,buffers,{}, {},lambda:False)


def test_prefix_one_reconstructed_after25_is_identical_and_bias_is_not_frozen():
    ns=load({'_prefix'})
    bundle,_=fixture_bundle(ns,25)
    origin=ns['_prefix'](bundle,1)
    later=copy.deepcopy(bundle)
    assert seal(ns['_prefix'](later,1)) == seal(origin)
    later['snapshots'][1]['teacher_ce'] += .1
    assert seal(ns['_prefix'](later,1)) != seal(origin)


@pytest.mark.parametrize('missing', ['step_000001.pt','optimization.csv'])
def test_certificate_requires_all_mirrors_readonly(tmp_path,missing):
    ns=load({'_cache_files'})
    folder=tmp_path/'candidate/condensation_0'; (folder/'checkpoints').mkdir(parents=True)
    for p in [folder.parent/'candidate.json',folder/'resume.pt',folder/'optimization.csv',
              folder/'checkpoints/step_000000.pt',folder/'checkpoints/step_000001.pt']:
        p.write_text('fixed')
    ns['_sha']=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    assert len(ns['_cache_files'](folder,1)) == 5
    target=folder/'checkpoints'/missing if missing.endswith('.pt') else folder/missing
    target.unlink(); before={str(p):p.read_bytes() for p in folder.rglob('*') if p.is_file()}
    with pytest.raises(ValueError): ns['_cache_files'](folder,1)
    assert before == {str(p):p.read_bytes() for p in folder.rglob('*') if p.is_file()}


def test_one_to25_same_context_and_outer_lr():
    ns=load({'_context'})
    ns['probe']=SimpleNamespace(_context=lambda *args: {'model_initial':'same','source_initial':'same'})
    buffers=dict(cells=30,recipe_origin={'identity_scale':1.})
    a=ns['_context']({},buffers,{}, {},{})
    b=ns['_context']({},buffers,{}, {},{})
    assert a==b and a['allowed_steps']==list(range(26)) and a['outer_optimizer']['lr']==.01
    assert 'target' not in a and 'frontier' not in a


def test_shared_P0_has_nine_physical_fits_not_twelve(tmp_path):
    ns=load({'_validate_students','_settings','_budget'})
    folder=tmp_path/'finite/condensation_0'; folder.mkdir(parents=True)
    buffers=dict(root=tmp_path,cells=30,counts={})
    bundle=dict(step=25,snapshots={0:{'moments':'same0'},25:{'moments':'finite25'}})
    gate=dict(prefix_seal=seal(bundle)); seen=set()
    ns.update(_gate=lambda *args,**kw:gate,_bundle_paths=lambda *args:(folder,folder.parent/'candidate.json',folder/'resume.pt'),
        _load_bundle=lambda *args,**kw:(bundle,26),_stop=lambda stop:None,
        representative=lambda moments,*args:(moments+'X',moments+'Q','mass64'),
        _stable_files=lambda b:None,_source_unchanged=lambda c:None,
        probe=SimpleNamespace(_transform=lambda b:{}),
        torch=SimpleNamespace(load=lambda p,**kw:{'moments':'same0' if '000000' in str(p) else 'node25'},
                              full_like=lambda mass,value:('weights64',value),equal=lambda a,b:a==b))
    def student(directory,inputs,*args):
        seed=args[-2]; key=(str(directory),seed); fresh=key not in seen; seen.add(key)
        return dict(actual_student_fits=int(fresh),physical_route_outputs=2*int(fresh),diagnostic_route_forwards=2,inputs=inputs)
    ns['_evaluate_student']=student
    candidate={ns['SOURCE_FIELD']:'fixedsource'}; spec={'certificate_outputs':{'30':{'25':'proof'}}}
    rows=[]; physical=0
    for arm in ('node_reference','gcn'):
        evidence={'counts':{}}
        result=ns['_validate_students'](candidate,arm,buffers,{},spec,'sha',evidence,lambda:False)
        rows+=result['rows'];physical+=evidence['counts']['physical_final_GCN_fits']
    assert len(rows)==12 and physical==9 and len(seen)==9
    assert [r['inputs'] for r in rows if r['shared_P0']][:3] == [r['inputs'] for r in rows if r['shared_P0']][3:]


@pytest.mark.parametrize('kind,entry', [('prepare','prepare'),('certify','certify'),('validate','validate')])
def test_actual_worker_lazy_dispatch_keeps_options_stop_and_return(monkeypatch,kind,entry):
    tree=ast.parse(WORKER.read_text()); fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='dispatch')
    ns={};exec(compile(ast.Module([fn],type_ignores=[]),str(WORKER),'exec'),ns)
    called=[]; sentinel=object();stop=lambda:False
    def fake(**kwargs): called.append(kwargs);return sentinel
    module=SimpleNamespace(**{entry:fake})
    monkeypatch.setitem(sys.modules,'src.citeseer_finite_student',module)
    assert ns['dispatch']({'kind':'citeseer_finite_student_'+kind,'options':{'cells':30}},stop) is sentinel
    assert called==[{'cells':30,'stop':stop}]


@pytest.mark.parametrize("secondary", [None, "sync", "preservation"])
def test_source_failure_preserves_primary_before_initializer_or_objective(tmp_path, secondary):
    ns = load({"_run", "_budget", "_phase", "canonical_candidate"})
    report = tmp_path / "new_evidence.json"
    spec = dict(source={"frozen": 1}, numerical_source={}, python_version="fixed", output_root=str(tmp_path),
                files_sha256={"source": "fixed"}, artifacts_sha256={"review": "fixed"},
                scientific_preregistration={"path": "science", "sha256": ns["SCIENCE_SHA"]})
    ns.update(_load_spec=lambda *a: (spec, {}), time=SimpleNamespace(monotonic=lambda: 0.),
              _write_new=lambda p, x: Path(p).write_text(json.dumps(x)),
              platform=SimpleNamespace(python_version=lambda: "fixed"))
    checks = []
    def unchanged(*args):
        if secondary == "preservation":
            raise ValueError("secondary source drift")
    def source(cells, spec, science, evidence, stop):
        evidence["stage"] = "strict_RMS_source_guard"
        evidence["counts"]["source_load_attempts"] = 1
        raise ValueError("original strict RMS failed")
    def sync():
        if secondary == "sync":
            raise RuntimeError("secondary sync failure")
    ns.update(_source_unchanged=unchanged, _load_source=source)
    ns["original"].CUDA_CAPACITY = 100
    ns["probe"] = SimpleNamespace(_native=lambda device: {"device": device}, _checked_files=lambda pins: checks.append(pins),
                                   _runtime_precision_guard=lambda: None)
    ns["torch"] = SimpleNamespace(get_num_threads=lambda: 4, cuda=SimpleNamespace(is_initialized=lambda: False,
        reset_peak_memory_stats=lambda: None, get_device_properties=lambda d: SimpleNamespace(total_memory=100),
        current_device=lambda: 0, synchronize=sync, max_memory_allocated=lambda: 10, max_memory_reserved=lambda: 20))
    with pytest.raises(ValueError, match="original strict RMS failed"):
        ns["_run"]("prepare", 30, ns["_FIXED"], "frozen_spec", "frozen", report, 1, None, lambda: False)
    saved = json.loads(report.read_text())
    assert saved["passed"] is False and saved["error"] == "original strict RMS failed"
    assert saved["failed_stage"] == "strict_RMS_source_guard"
    assert saved["counts"]["P_update_attempts"] == saved["counts"]["final_student_fit_attempts"] == 0
    assert saved["counts"].get("finite_objective_attempts", 0) == 0
    assert len(checks) == 3
    if secondary == "sync":
        assert "native_finalization_error" in saved
    if secondary == "preservation":
        assert saved["source_unchanged"] is False and "preservation_error" in saved


@pytest.mark.parametrize("operation,phase,sha", [("prepare", 25, None), ("certify", 25, None),
                                                ("prepare", 1, "external"), ("validate", "gcn", None)])
def test_required_phase_gate_rejects_before_source_or_spec(operation, phase, sha):
    ns = load({"_run", "_budget", "_phase", "canonical_candidate"})
    ns["_load_spec"] = lambda *a: pytest.fail("spec/source reached before gate-control rejection")
    with pytest.raises(ValueError):
        ns["_run"](operation, 30, ns["_FIXED"], "spec", "sha", "output", phase, sha, lambda: False)
