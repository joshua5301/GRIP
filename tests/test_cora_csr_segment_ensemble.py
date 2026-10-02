"""BN ensemble-only AST/metadata orchestration controls; no tensor imports."""
import ast
import copy
import hashlib
import json
import types
from pathlib import Path

import pytest

BASE = Path(__file__).resolve().parents[1]
PATH = BASE / 'src/cora_csr_segment_ensemble.py'
WORKER = BASE / 'src/research_loop.py'
REPO = next(p for p in Path(__file__).resolve().parents if (p / 'src/transforms.py').is_file())
SCIENCE = json.loads((REPO / 'results/proposals/Cora70_three_anchor_original_CSR_segment_scientific_stageBN_v1.json').read_text())
SHA = 'c847f01b7a0df032a3a1b23deeb4f0213ecce82b0b00a673f9d8f48c8f78c74c'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seal(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def extract(names, **extra):
    tree = ast.parse(PATH.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    nodes += [n for n in tree.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
              and n.targets[0].id in ('_FIELDS', 'SCIENCE', 'SCIENCE_SHA', '_BACKEND')]
    ns = dict(Path=Path, json=json, _require=require, _sha=sha, _seal=seal,
              _exact=lambda a, b: type(a) is type(b) and a == b,
              probe=types.SimpleNamespace(_digest=copy.deepcopy, _attach=lambda v: dict(v, content_sha256=seal(v))))
    ns.update(extra)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(PATH), 'exec'), ns)
    return ns


def api(tmp):
    ns = extract({'_load_spec', '_artifact_paths'}, __file__=str(tmp / 'src/module.py'),
                 _science=lambda r: SCIENCE, _preserve=lambda *a: None)
    spec = {k: copy.deepcopy(SCIENCE['original_files_sha256'] if k == 'files_sha256' else SCIENCE[k]) for k in
            ('fixed', 'files_sha256', 'output_root', 'pass_outputs', 'target_outputs', 'comparison_policy')}
    spec.update(schema=1, source={'files': dict(SCIENCE['source_before']['files']) | {'src/cora_csr_segment_ensemble.py': 'future'}},
                numerical_source={'versions': SCIENCE['version_binding']['runtime_versions_from_pinned_native_metadata']},
                python_version=SCIENCE['version_binding']['python_version'],
                artifacts_sha256={str(tmp / 'src/cora_csr_segment_ensemble.py'): 'future'},
                scientific_preregistration={'path': str(tmp / 'results/proposals' / ns['SCIENCE']), 'sha256': SHA})
    path = tmp / 'results/proposals/spec.json'
    path.parent.mkdir(parents=True)
    return ns, spec, path


def complete_digest():
    zero = copy.deepcopy(SCIENCE['source_binding']['seed0_expected_complete_common_digest'])
    shared = {k: zero.pop(k) for k in ('source_buffers', 'S', 'T')}
    records = {str(seed): copy.deepcopy(zero) for seed in (0, 1, 2)}
    for key in ('1', '2'):
        records[key]['anchor'] = copy.deepcopy(SCIENCE['source_binding']['parameter_descriptors_by_seed'][key])
    return dict(shared=shared, anchors_by_seed=records)


def test_new_source74_namespace_exact_three_owned_artifacts(tmp_path):
    ns, spec, path = api(tmp_path)
    path.write_text(json.dumps(spec))
    value, science = ns['_load_spec'](path, sha(path))
    assert value == spec and len(value['source']['files']) == 74 and science is SCIENCE
    allowed = ['src/cora_csr_segment_ensemble.py', 'tests/test_cora_csr_segment_ensemble.py', 'src/research_loop.py']
    ns['_artifact_paths']({str(tmp_path / p): 'pin' for p in allowed}, SCIENCE, tmp_path)
    for path in ('src/undeclared.py', 'tests/undeclared.py'):
        with pytest.raises(ValueError):
            ns['_artifact_paths']({str(tmp_path / path): 'pin'}, SCIENCE, tmp_path)


@pytest.mark.parametrize('mutation', ['old_science', 'unknown_control', 'backend', 'seeds', 'missing_source', 'extra_source'])
def test_changed_scientific_namespace_or_source_rejected(tmp_path, mutation):
    ns, spec, path = api(tmp_path)
    if mutation == 'old_science':
        spec['scientific_preregistration']['sha256'] = 'oldBM'
    elif mutation == 'unknown_control':
        spec['retry'] = True
    elif mutation == 'backend':
        spec['fixed']['source_backend'] = 'torch_sparse_mm'
    elif mutation == 'seeds':
        spec['fixed']['anchor_seeds'] = [0, 2, 1]
    elif mutation == 'missing_source':
        spec['source']['files'].pop('src/cora_csr_segment_ensemble.py')
    else:
        spec['source']['files']['src/undeclared.py'] = 'unreviewed'
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError):
        ns['_load_spec'](path, sha(path))


@pytest.mark.parametrize('mutation', [None, 'seed0_boundary', 'seed1_anchor', 'missing_seed', 'source', 'extra_anchor_field'])
def test_complete_common_and_stronger_seed0_BM_projection(mutation):
    value = complete_digest()
    if mutation == 'seed0_boundary':
        value['anchors_by_seed']['0']['boundaries']['D']['tensor'] = 'changed'
    elif mutation == 'seed1_anchor':
        value['anchors_by_seed']['1']['anchor'] = value['anchors_by_seed']['0']['anchor']
    elif mutation == 'missing_seed':
        value['anchors_by_seed'].pop('2')
    elif mutation == 'source':
        value['shared']['source_buffers']['Q']['tensor'] = 'changed'
    elif mutation == 'extra_anchor_field':
        value['anchors_by_seed']['2']['drop_block'] = True
    ns = extract({'_common_contract'})
    if mutation is None:
        ns['_common_contract'](value, SCIENCE)
    else:
        with pytest.raises(ValueError):
            ns['_common_contract'](value, SCIENCE)


def ensemble(fail_seed=None, transpose_seed=None):
    calls = []
    evidence = dict(counts={}, operation_attempts={})
    parameters = SCIENCE['source_binding']['parameter_descriptors_by_seed']

    def invoke(record, key, function, *args, **kwargs):
        record['operation_attempts'][key] = record['operation_attempts'].get(key, 0) + 1
        result = function(*args, **kwargs)
        record['counts'][key] = record['counts'].get(key, 0) + 1
        return result

    def factory(d, c, h, seed, **kwargs):
        assert (d, c, h, kwargs) == (1433, 7, 256, {'dtype': 'FP32', 'device': 'cuda'})
        calls.append(('factory', seed))
        return copy.deepcopy(parameters[str(seed)])

    def gradient(anchor, x, s, q, record):
        seed = next(i for i in (0, 1, 2) if anchor == parameters[str(i)])
        calls.append(('gradient', seed))
        if seed == fail_seed:
            raise ValueError('incomplete anchor')
        return dict(boundaries={'seed': seed}, transpose={'mock_T': seed if seed == transpose_seed else 'fixed'}, loss=seed)

    ns = extract({'_collect'}, explicit=types.SimpleNamespace(_invoke=invoke), anchor_initial=factory,
                 torch=types.SimpleNamespace(float32='FP32'), _native_inputs=lambda *a: None,
                 gradient_boundaries=gradient, target_packet=lambda r, e: {'seed': r['boundaries']['seed']},
                 _isolated=lambda s, b, e: {'seed': b['seed']}, _comparison=lambda b, i: {'descriptive_only': b['seed']})
    buffers = dict(h='H', z='z', q='Q', hard='hard', transform='transform', x='X', S='S')
    return ns, buffers, evidence, calls


def test_fixed_three_anchor_calls_share_source_and_preserve_inputs():
    ns, buffers, evidence, calls = ensemble()
    before = copy.deepcopy(buffers)
    common, isolated = ns['_collect'](buffers, SCIENCE, evidence)
    assert calls == [(kind, seed) for seed in (0, 1, 2) for kind in ('factory', 'gradient')]
    assert list(common['anchors_by_seed']) == list(isolated) == ['0', '1', '2']
    assert evidence['counts'] == dict(private_GEOM_anchor_factory_calls=3, explicit_source_parameter_gradient_assemblies=3, source_target_packet_assemblies=3)
    assert {k: buffers[k] for k in before} == before


def test_failed_anchor_preserves_prefix_and_never_replaces_or_runs_next_seed():
    ns, buffers, evidence, calls = ensemble(fail_seed=1)
    with pytest.raises(ValueError, match='incomplete anchor'):
        ns['_collect'](buffers, SCIENCE, evidence)
    assert calls == [('factory', 0), ('gradient', 0), ('factory', 1), ('gradient', 1)]
    assert list(buffers['common']['anchors_by_seed']) == list(evidence['descriptive_comparison']) == ['0']
    assert evidence['operation_attempts']['explicit_source_parameter_gradient_assemblies'] == 2
    assert evidence['counts']['explicit_source_parameter_gradient_assemblies'] == 1


def test_per_anchor_transpose_mismatch_is_rejected():
    ns, buffers, evidence, _ = ensemble(transpose_seed=1)
    with pytest.raises(ValueError, match='transpose representation'):
        ns['_collect'](buffers, SCIENCE, evidence)
    assert evidence['counts']['source_target_packet_assemblies'] == 2
    assert list(buffers['common']['anchors_by_seed']) == ['0']


def test_complete_raw_capture_survives_exact_repeat_failure(tmp_path):
    common = complete_digest()
    previous = copy.deepcopy(common)
    common['anchors_by_seed']['1']['CE']['tensor'] = 'new-backend-repeat-failure'
    target = tmp_path / 'new-pass/cache.json'
    evidence = dict(source_context={}, native_source_buffers_before={}, spec_sha256='spec', native_environment={},
                    within_three_anchor_segment_exact_repeat_passed=False, three_anchor_segment_backend_qualified=False)
    spec = dict(source={}, numerical_source={}, scientific_preregistration={})
    context = {k: spec[k] for k in ('source', 'numerical_source', 'scientific_preregistration')}
    context.update(spec_sha256='spec', source_context={}, native_source_buffers={}, source_backend='original_CSR_coefficients_rank2_weighted_segment_SUM')
    old_target = tmp_path / 'old.json'
    old_target.write_text(json.dumps(dict(context=context, pass_index=1, common=previous)))
    first = dict(native_environment={}, common_numerical_digest=previous, common_numerical_seal=seal(previous))
    ns = extract({'_capture', '_repeat', '_context'},
                 torch=types.SimpleNamespace(save=lambda value, stream: stream.write(json.dumps(value).encode()),
                                             load=lambda path, **kwargs: json.loads(Path(path).read_text())),
                 explicit=types.SimpleNamespace(_invoke=lambda e, k, f, *a, **kw: f(*a, **kw)),
                 bi=types.SimpleNamespace(_sealed_payload=lambda p: p),
                 _previous=lambda *a: (first, tmp_path / 'old-receipt.json', old_target))
    ns['_capture'](spec, target, context, 2, common, {}, evidence)
    with pytest.raises(ValueError, match='exact common repeat'):
        ns['_repeat'](spec, SCIENCE, common, evidence, 'priorSHA')
    assert target.is_file() and evidence['raw_cache_sha256'] == sha(target)
    assert evidence['raw_cache_is_qualification'] is False and evidence['three_anchor_segment_backend_qualified'] is False


def test_actual_public_order_and_immutable_math_aliases():
    tree = ast.parse(PATH.read_text())
    aliases = {ast.dump(n, include_attributes=False) for n in tree.body if isinstance(n, ast.Assign)}
    assert ast.dump(ast.parse('gradient_boundaries, target_packet = explicit.gradient_boundaries, explicit.target_packet').body[0], include_attributes=False) in aliases
    assert ast.dump(ast.parse('_native_inputs, _isolated, _comparison = bm._native_inputs, bm._isolated, bm._comparison').body[0], include_attributes=False) in aliases
    assert not any(isinstance(n, ast.FunctionDef) and n.name in ('gradient_boundaries', 'target_packet', '_product') for n in tree.body)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'prepare_isolation')
    text = ast.unparse(fn)
    assert text.index('_capture(') < text.index('_common_contract(') < text.index('_repeat(')
    assert 'within_three_anchor_segment_exact_repeat_passed=False' in text and 'three_anchor_segment_backend_qualified=False' in text


def test_actual_worker_lazy_dispatch_preserves_callback_and_result(monkeypatch):
    import sys
    tree = ast.parse(WORKER.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'dispatch')
    branch = next(n for n in fn.body if isinstance(n, ast.If) and 'citation_cora_csr_segment_ensemble' in ast.unparse(n.test))
    call = ast.FunctionDef(name='call', args=ast.arguments(posonlyargs=[], args=[ast.arg(arg='job'), ast.arg(arg='stop')],
                          kwonlyargs=[], kw_defaults=[], defaults=[]), body=[fn.body[0], branch], decorator_list=[])
    ns = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[call], type_ignores=[])), str(WORKER), 'exec'), ns)
    marker, calls = object(), []
    stop = lambda: False
    module = types.ModuleType('src.cora_csr_segment_ensemble')
    module.prepare_isolation = lambda **kw: (calls.append(kw), marker)[1]
    package = types.ModuleType('src')
    package.__path__ = []
    monkeypatch.setitem(sys.modules, 'src', package)
    monkeypatch.setitem(sys.modules, 'src.cora_csr_segment_ensemble', module)
    assert ns['call']({'kind': 'citation_cora_csr_segment_ensemble', 'options': {'pass_index': 2}}, stop) is marker
    assert calls == [{'pass_index': 2, 'stop': stop}]


def test_recursive_complete_finiteness_precedes_raw_capture():
    class Tensor:
        def __init__(self, finite, layout='dense'):
            self.finite, self.layout = finite, layout
        def values(self):
            return Tensor(self.finite)
    fake = types.SimpleNamespace(is_tensor=lambda x: isinstance(x, Tensor), strided='dense', sparse_csr='CSR',
                                 isfinite=lambda x: types.SimpleNamespace(all=lambda: x.finite))
    ns = extract({'_finite_tree'}, torch=fake)
    ns['_finite_tree']({'source': Tensor(True, 'CSR'), 'mask': [Tensor(True)], 'metadata': None})
    with pytest.raises(ValueError, match='Nonfinite complete'):
        ns['_finite_tree']({'anchor': [{'A': Tensor(False)}]})
    tree = ast.parse(PATH.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'prepare_isolation')
    text = ast.unparse(fn)
    assert text.index('_finite_tree(common)') < text.index('_capture(')
    assert text.index('_finite_tree(isolated)') < text.index('_capture(')
