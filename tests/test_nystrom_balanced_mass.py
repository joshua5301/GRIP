"""Balanced Nyström uses fixed source H/Q and an exact constrained hypergradient."""
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import src.citation_search as search
import src.nystrom_ce as nys
from src.balanced_assignment import BalancedMoments
from src.io import _fingerprint, save_state
from src.low_rank_assignment import LowRankLogits, initialize_factors, logit_block
from src.moments import augmented, decode_moments, make_material
from src.soft_ce_partition import solve_head_system, solve_inner_newton_first


@pytest.fixture(scope='module', autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)

def problem():
    generator = torch.Generator().manual_seed(7)
    h = torch.randn(24, 3, dtype=torch.double, generator=generator) + 0.2
    q = torch.randn(24, 3, dtype=torch.double, generator=generator).softmax(1)
    assignment = torch.tensor([0] * 9 + [1] * 6 + [2] * 5 + [3] * 4)
    feature_map = nys.NystromMap.fit(h, basis=8)
    return (h, q, assignment, feature_map)

def balanced(u, v, a, material, tol=1e-12, dual=None):
    logits = LowRankLogits.apply(u, v, a, 0.05, 7)
    return (BalancedMoments.apply(logits, material, 7, 1000, tol, dual), logits)

def load(path):
    return torch.load(path, map_location='cpu', weights_only=False)

def fit_objective(value, h, q, feature_map, phi, weighting):
    x, y, mass = decode_moments(value, h.shape[1])
    weights = nys._inner_weights(mass, weighting)
    mapped = feature_map(x).detach()
    inner = solve_inner_newton_first(mapped, y, weights, 0.1, grad_tol=1e-10)
    assert inner['inner_converged']
    loss, rhs = nys.outer_gradient(phi, q, inner['theta'], 7)
    return (loss, rhs, inner['theta'], mapped, y, weights)

@pytest.mark.parametrize('factor', ['u', 'v', 'both'])
def test_full_nonlinear_constrained_head_hypergradient_fd(factor):
    h, q, a, feature_map = problem()
    material = make_material(h, q)
    phi = feature_map(h).detach().numpy()
    g = torch.Generator().manual_seed(19)
    u = (0.2 * torch.randn(24, 3, dtype=torch.double, generator=g)).requires_grad_()
    v = torch.randn(4, 3, dtype=torch.double, generator=g).requires_grad_()
    du = 0.2 * torch.randn(u.shape, dtype=u.dtype, generator=g)
    dv = 0.2 * torch.randn(v.shape, dtype=v.dtype, generator=g)
    if factor == 'u':
        dv.zero_()
    if factor == 'v':
        du.zero_()
    (moments, dual, info), logits = balanced(u, v, a, material)
    logits.retain_grad()
    loss, rhs, theta, mapped, y, weights = fit_objective(moments.detach(), h, q, feature_map, phi, 'uniform')
    vector, diagnostic = solve_head_system(augmented(mapped), y, weights, theta, 0.1, rhs, rtol=1e-10)
    assert diagnostic['cg_converged']
    derivative = nys.moment_gradient(moments, 3, feature_map, theta, vector, 0.1, 'uniform')
    moments.backward(derivative)
    analytic = float((u.grad * du).sum() + (v.grad * dv).sum())
    epsilon = 0.001
    plus = balanced(u.detach() + epsilon * du, v.detach() + epsilon * dv, a, material)[0][0]
    minus = balanced(u.detach() - epsilon * du, v.detach() - epsilon * dv, a, material)[0][0]
    numerical = (fit_objective(plus, h, q, feature_map, phi, 'uniform')[0] - fit_objective(minus, h, q, feature_map, phi, 'uniform')[0]) / (2 * epsilon)
    np.testing.assert_allclose(analytic, numerical, rtol=0.0002, atol=1e-07)
    torch.testing.assert_close(logits.grad.sum(0), torch.zeros(4, dtype=torch.double), atol=1e-12, rtol=0)
    torch.testing.assert_close(logits.grad.sum(1), torch.zeros(24, dtype=torch.double), atol=1e-12, rtol=0)

def test_actual_p_material_conservation_and_gauge_invariance():
    h, q, a, feature_map = problem()
    material = make_material(h, q)
    u, v = initialize_factors(a, 4, 3, 0)
    u = u.detach().double().requires_grad_()
    v = v.detach().double().requires_grad_()
    (moments, dual, info), logits = balanced(u, v, a, material)
    p = (logits.detach() + dual).softmax(1)
    torch.testing.assert_close(p.sum(1), torch.ones(24, dtype=torch.double), atol=1e-12, rtol=0)
    torch.testing.assert_close(p.mean(0), torch.full((4,), 0.25, dtype=torch.double), atol=3e-13, rtol=0)
    torch.testing.assert_close(moments, p.T @ material / 24, atol=1e-15, rtol=1e-14)
    torch.testing.assert_close(moments.sum(0), material.mean(0), atol=1e-12, rtol=0)
    x, y, mass = decode_moments(moments, 3)
    torch.testing.assert_close(y.sum(1), torch.ones(4, dtype=torch.double), atol=1e-12, rtol=0)
    torch.testing.assert_close(x.mean(0), h.mean(0), atol=1e-12, rtol=0)
    torch.testing.assert_close(y.mean(0), q.mean(0), atol=1e-12, rtol=0)
    row = torch.linspace(-0.7, 0.5, 24, dtype=torch.double)[:, None]
    col = torch.tensor([0.5, -0.2, 0.1, -0.4], dtype=torch.double)
    shifted, shift_dual, _ = BalancedMoments.apply(logits.detach() + row + col, material, 7, 1000, 1e-12, None)
    torch.testing.assert_close(shifted, moments, atol=1e-12, rtol=1e-11)
    torch.testing.assert_close((logits.detach() + row + col + shift_dual).softmax(1), p, atol=2e-12, rtol=1e-11)
    assert abs(float(dual.mean())) < 1e-14

def test_equal_mass_uniform_and_mass_heads_and_constrained_hypergradients_coincide():
    h, q, a, feature_map = problem()
    material = make_material(h, q)
    phi = feature_map(h).detach().numpy()
    gradients = []
    heads = []
    for weighting in ('mass', 'uniform'):
        u, v = initialize_factors(a, 4, 3, 2)
        u = u.detach().double().requires_grad_()
        v = v.detach().double().requires_grad_()
        (moments, dual, info), _ = balanced(u, v, a, material)
        loss, rhs, theta, mapped, y, weights = fit_objective(moments.detach(), h, q, feature_map, phi, weighting)
        vector, diagnostic = solve_head_system(augmented(mapped), y, weights, theta, 0.1, rhs, rtol=1e-10)
        assert diagnostic['cg_converged']
        moments.backward(nys.moment_gradient(moments, 3, feature_map, theta, vector, 0.1, weighting))
        heads.append(theta)
        gradients.append((u.grad, v.grad))
    torch.testing.assert_close(heads[0], heads[1], atol=1e-10, rtol=1e-09)
    for left, right in zip(gradients[0], gradients[1]):
        torch.testing.assert_close(left, right, atol=1e-10, rtol=1e-07)

def test_balanced_short_optimization_resume_and_interrupted_parity(tmp_path):
    h, q, a, feature_map = problem()
    phi = feature_map(h).detach().numpy()
    options = dict(penalty=0.1, rank=3, chunk=7, checkpoint_every=1, mass_mode='uniform', inner_loss_weighting='uniform')
    nys.optimize(h, q, a, feature_map, phi, tmp_path / 'full', 2, **options)
    nys.optimize(h, q, a, feature_map, phi, tmp_path / 'extended', 1, **options)
    nys.optimize(h, q, a, feature_map, phi, tmp_path / 'extended', 2, **options)
    halted = {'stop': False}

    def progress(row):
        if row['step'] == 0:
            halted['stop'] = True
    with pytest.raises(InterruptedError):
        nys.optimize(h, q, a, feature_map, phi, tmp_path / 'interrupted', 2, stop=lambda: halted['stop'], progress=progress, **options)
    assert load(tmp_path / 'interrupted/resume.pt')['step'] == 1
    nys.optimize(h, q, a, feature_map, phi, tmp_path / 'interrupted', 2, **options)
    ref = load(tmp_path / 'full/step_000002.pt')
    for name in ('extended', 'interrupted'):
        actual = load(tmp_path / name / 'step_000002.pt')
        state = load(tmp_path / name / 'resume.pt')
        nys.validate_assignment_snapshot(actual, 'uniform')
        assert state['config']['mass_mode'] == 'uniform' and state['config']['balance_backend'] == 'chunked'
        torch.testing.assert_close(actual['moments'], ref['moments'], rtol=1e-07, atol=1e-09)
        torch.testing.assert_close(actual['theta'], ref['theta'], rtol=1e-05, atol=1e-07)
        u, v = (state['u'], state['v'])
        p = (logit_block(u, v, a, 0.05).double() + state['dual']).softmax(1)
        torch.testing.assert_close(actual['moments'], p.T @ make_material(h, q) / 24, rtol=1e-13, atol=1e-14)
        assert state['dual'] is not None and actual['balance']['balance_backward_checked']

def test_same_initial_factors_but_distinct_own_p0(tmp_path):
    h, q, a, feature_map = problem()
    phi = feature_map(h).detach().numpy()
    states = {}
    snapshots = {}
    for mode in ('free', 'uniform'):
        nys.optimize(h, q, a, feature_map, phi, tmp_path / mode, 0, penalty=0.1, rank=3, chunk=7, mass_mode=mode, inner_loss_weighting='uniform')
        states[mode] = load(tmp_path / mode / 'resume.pt')
        snapshots[mode] = load(tmp_path / mode / 'step_000000.pt')
    for key in ('u', 'v'):
        torch.testing.assert_close(states['free'][key], states['uniform'][key], atol=0, rtol=0)
    assert not torch.allclose(snapshots['free']['moments'], snapshots['uniform']['moments'])
    torch.testing.assert_close(snapshots['uniform']['moments'][:, 0], torch.full((4,), 0.25, dtype=torch.double), atol=3e-09, rtol=0)

@pytest.mark.parametrize('options', [{'mass_mode': True}, {'mass_mode': 'balanced'}, {'balance_steps': 0}, {'balance_steps': True}, {'balance_steps': 1.5}, {'balance_tol': 0}, {'balance_tol': True}, {'balance_tol': float('nan')}, {'mass_mode': 'free', 'balance_tol': 1e-06}, {'mass_mode': 'uniform', 'balance_backend': 'cached'}])
def test_invalid_controls_rejected_prewrite(tmp_path, options):
    h, q, a, feature_map = problem()
    phi = feature_map(h).detach().numpy()
    folder = tmp_path / 'absent'
    with pytest.raises(ValueError):
        nys.optimize(h, q, a, feature_map, phi, folder, 0, **options)
    assert not folder.exists()

@pytest.mark.parametrize('old,new', [('free', 'uniform'), ('uniform', 'free')])
def test_cross_mode_resume_rejected_without_parameter_or_cache_mutation(tmp_path, old, new):
    h, q, a, feature_map = problem()
    path = tmp_path / 'phi.npy'
    phi = nys.cache_features(h, feature_map, path, 7)
    nys.optimize(h, q, a, feature_map, phi, tmp_path / 'condensation', 0, penalty=0.1, rank=3, chunk=7, mass_mode=old)
    path.with_suffix('.meta.json').unlink()
    before = {p.relative_to(tmp_path).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.rglob('*') if p.is_file()}
    with pytest.raises(ValueError, match='mass_mode'):
        nys.optimize(h, q, a, feature_map, phi, tmp_path / 'condensation', 1, penalty=0.1, rank=3, chunk=7, mass_mode=new)
    after = {p.relative_to(tmp_path).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.rglob('*') if p.is_file()}
    assert before == after

def test_unconverged_balancing_does_not_persist_endpoint_or_resume(tmp_path):
    h, q, a, feature_map = problem()
    phi = feature_map(h).detach().numpy()
    folder = tmp_path / 'failure'
    with pytest.raises(RuntimeError, match='did not converge'):
        nys.optimize(h, q, a, feature_map, phi, folder, 0, mass_mode='uniform', balance_steps=1, penalty=0.1, rank=3, chunk=7)
    assert not list(folder.glob('*.pt')) and (not (folder / 'history.json').exists())

@pytest.mark.parametrize('failure', ['adjoint', 'backward'])
def test_derivative_failure_does_not_persist_balanced_endpoint(tmp_path, monkeypatch, failure):
    h, q, a, feature_map = problem()
    phi = feature_map(h).detach().numpy()
    folder = tmp_path / 'failure'
    if failure == 'adjoint':
        monkeypatch.setattr(nys, 'solve_head_system', lambda *args, **kw: (torch.zeros_like(args[3]), {'cg_converged': False}))
    else:

        def fail(*args):
            raise RuntimeError('synthetic derivative failure')
        monkeypatch.setattr(BalancedMoments, 'backward', fail)
    with pytest.raises(RuntimeError):
        nys.optimize(h, q, a, feature_map, phi, folder, 0, mass_mode='uniform', penalty=0.1, rank=3, chunk=7)
    assert not list(folder.glob('*.pt'))

@pytest.fixture
def citation_mock(tmp_path, monkeypatch):
    h, q, a, feature_map = problem()
    h = h.float()
    n = len(h)
    graph = {'x': h, 'y': torch.arange(n) % 3, 'adj': torch.eye(n).to_sparse_csr()}
    train = torch.arange(n) < 8
    validation = (graph, (torch.arange(n) >= 8) & (torch.arange(n) < 16))
    testing = (graph, torch.arange(n) >= 16)
    record = {'data_calls': 0, 'fits': [], 'optimizer_kwargs': [], 'teacher_calls': 0}

    def prepare(*args):
        record['data_calls'] += 1
        return (graph, train, validation, testing, h)

    def teacher(*args):
        record['teacher_calls'] += 1
        save_state({'logits': q.log(), 'gamma': 0.01}, args[-1] / 'teacher.pt')
        return (q.log(), 0.01)

    def evaluate(x, y, mass, eg, outerq, masks, seed, **kwargs):
        assert set(masks) in ({'train', 'val'}, {'train', 'val', 'test'})
        torch.testing.assert_close(mass, torch.full_like(mass, 0.25), atol=0, rtol=0)
        record['fits'].append({'x': x.clone(), 'y': y.clone(), 'mass': mass.clone(), 'q': outerq.clone(), 'seed': seed})
        return {'seed': seed, 'epoch': 1, 'val_acc': 50.0, 'val_ce': 1.2, **({'test_acc': 50.0} if 'test' in masks else {})}
    engine = nys.optimize

    def optimize(*args, **kwargs):
        record['optimizer_kwargs'].append(dict(kwargs))
        return engine(*args, **kwargs)
    monkeypatch.setattr(search, '_prepare_dataset', prepare)
    monkeypatch.setattr(search, 'teacher_logits', teacher)
    monkeypatch.setattr(search, 'feature_kmeans', lambda *args: a.clone())
    monkeypatch.setattr(search, 'fit_gcn_diagnostic', evaluate)
    monkeypatch.setattr(search, 'get_shared_map', lambda *args, **kwargs: nys.NystromMap.fit(h.double(), basis=8))
    monkeypatch.setattr(nys, 'optimize', optimize)
    monkeypatch.setitem(search.BUDGET, ('cora', 0.013), 4)
    record['h'] = h
    record['q'] = q
    record['a'] = a
    return record

def candidate(**extras):
    return dict(method='nystrom', width=0, lr=0.01, T=1.0, rank=3, penalty=0.1, inner_loss_weighting='uniform', **extras)

def run(tmp_path, candidates, **extra):
    return search.run_screen('cora', 0.013, tmp_path, candidates, steps=1, student_seeds=(1600,), condensation_seed=0, epochs=1, dropout=0, device='cpu', **extra)

@pytest.mark.parametrize('extra', [{'mass_mode': 'bad'}, {'mass_mode': True}, {'mass_mode': 'uniform', 'balance_steps': 0}, {'mass_mode': 'uniform', 'balance_tol': float('nan')}, {'mass_mode': 'uniform', 'balance_backend': 'cached'}, {'mass_mode': 'uniform', 'balance_cg_steps': 10}])
def test_citation_preload_rejects_invalid_balanced_controls(tmp_path, citation_mock, extra):
    with pytest.raises(ValueError):
        run(tmp_path, [candidate(**extra)])
    assert citation_mock['data_calls'] == 0 and (not list(tmp_path.iterdir()))

def test_citation_balanced_forwarding_identity_and_actual_endpoint_guard(tmp_path, citation_mock):
    ranking, root = run(tmp_path, [candidate(), candidate(mass_mode='uniform', balance_backend='chunked')])
    assert len(ranking) == 4 and len(citation_mock['fits']) == 4
    default, uniform = citation_mock['optimizer_kwargs']
    assert 'mass_mode' not in default and 'balance_steps' not in default and ('balance_backend' not in default)
    assert uniform['mass_mode'] == 'uniform' and uniform['balance_steps'] == 300 and (uniform['balance_tol'] == 1e-08)
    assert uniform['balance_backend'] == 'chunked'
    choice = ranking[(ranking.mass_mode == 'uniform') & (ranking.step == 1)].iloc[0].to_dict()
    folder = Path(choice['candidate_path']) / 'condensation_0'
    identity = json.loads((folder.parent / 'candidate.json').read_text())
    assert identity['nystrom_schema'] == 3 and identity['balance_backend'] == 'chunked' and (identity['balance_steps'] == 300)
    assert _fingerprint(identity) == folder.parent.name
    snap = load(folder / 'step_000001.pt')
    nys.validate_assignment_snapshot(snap, 'uniform')
    centers, labels, mass = decode_moments(snap['moments'], 3)
    matched = [r for r in citation_mock['fits'] if torch.allclose(r['x'], centers.float())]
    assert matched and torch.allclose(matched[-1]['y'], labels.float())
    assert torch.allclose(matched[-1]['q'], citation_mock['q'])
    snapshot_digest = hashlib.sha256((folder / 'step_000001.pt').read_bytes()).hexdigest()
    before_calls = len(citation_mock['optimizer_kwargs'])
    run(tmp_path, [candidate(mass_mode='uniform')])
    assert len(citation_mock['optimizer_kwargs']) == before_calls
    assert hashlib.sha256((folder / 'step_000001.pt').read_bytes()).hexdigest() == snapshot_digest
    snap['mass_mode'] = 'free'
    save_state(snap, folder / 'step_000001.pt')
    before = len(citation_mock['fits'])
    with pytest.raises(ValueError, match='mass_mode'):
        run(tmp_path, [candidate(mass_mode='uniform')])
    assert len(citation_mock['fits']) == before + 1
    before = citation_mock['data_calls']
    with pytest.raises(ValueError, match='mass_mode'):
        search.selected_test(root, choice, condensation_seeds=(0,), student_seeds=(1700,), epochs=1, device='cpu', report_routes=False)
    assert citation_mock['data_calls'] == before and (not (root / 'selected.json').exists())

def test_citation_default_and_explicit_free_candidate_identity_unchanged(tmp_path, citation_mock):
    a, root = run(tmp_path, [candidate()])
    before = len(citation_mock['optimizer_kwargs'])
    b, same = run(tmp_path, [candidate(mass_mode='free', balance_steps=300, balance_tol=1e-08, balance_backend='chunked')])
    assert root == same and a.candidate_path.tolist() == b.candidate_path.tolist()
    assert len(citation_mock['optimizer_kwargs']) == before
    assert 'mass_mode' not in json.loads((Path(a.iloc[0].candidate_path) / 'candidate.json').read_text())

@pytest.mark.parametrize('bad_dual', [True, 1.0, [0.0, 0.0, 0.0, 0.0], torch.tensor(float('nan')), torch.tensor([float('nan')] * 4)])
def test_malformed_resume_duals_are_value_errors(bad_dual):
    with pytest.raises(ValueError, match='dual'):
        nys.validate_assignment_resume({'config': {'mass_mode': 'uniform', 'balance_steps': 300, 'balance_tol': 1e-08, 'balance_backend': 'chunked'}, 'step': 1, 'theta': torch.zeros(3, 9), 'dual': bad_dual}, 'uniform')

@pytest.mark.parametrize('phase', ['balance', 'adjoint'])
def test_stop_after_bounded_helper_preserves_latest_resume_and_parity(tmp_path, monkeypatch, phase):
    h, q, a, feature_map = problem()
    phi = feature_map(h).detach().numpy()
    options = dict(penalty=0.1, rank=3, chunk=7, checkpoint_every=1, mass_mode='uniform', inner_loss_weighting='uniform')
    nys.optimize(h, q, a, feature_map, phi, tmp_path / 'reference', 2, **options)
    halted = {'stop': False}
    if phase == 'balance':
        helper = BalancedMoments.apply

        def stopped_helper(*args):
            result = helper(*args)
            halted['stop'] = True
            return result
        with monkeypatch.context() as local:
            local.setattr(BalancedMoments, 'apply', stopped_helper)
            with pytest.raises(InterruptedError):
                nys.optimize(h, q, a, feature_map, phi, tmp_path / 'stopped', 2, stop=lambda: halted['stop'], **options)
    else:
        helper = nys.solve_head_system

        def stopped_helper(*args, **kwargs):
            result = helper(*args, **kwargs)
            halted['stop'] = True
            return result
        with monkeypatch.context() as local:
            local.setattr(nys, 'solve_head_system', stopped_helper)
            with pytest.raises(InterruptedError):
                nys.optimize(h, q, a, feature_map, phi, tmp_path / 'stopped', 2, stop=lambda: halted['stop'], **options)
    state = load(tmp_path / 'stopped/resume.pt')
    assert state['step'] == 0 and state['dual'] is not None
    nys.validate_assignment_resume(state, 'uniform')
    assert not (tmp_path / 'stopped/step_000000.pt').exists()
    nys.optimize(h, q, a, feature_map, phi, tmp_path / 'stopped', 2, **options)
    actual = load(tmp_path / 'stopped/step_000002.pt')
    expected = load(tmp_path / 'reference/step_000002.pt')
    torch.testing.assert_close(actual['moments'], expected['moments'], atol=1e-09, rtol=1e-07)
    torch.testing.assert_close(actual['theta'], expected['theta'], atol=1e-07, rtol=1e-05)

def test_bad_marginals_returned_by_helper_never_persist_even_if_stop(tmp_path, monkeypatch):
    h, q, a, feature_map = problem()
    phi = feature_map(h).detach().numpy()
    folder = tmp_path / 'invalid'
    original_apply = BalancedMoments.apply

    def corrupt(*args):
        value, dual, diagnostic = original_apply(*args)
        value = value.clone()
        value[0, 0] += 0.1
        return (value, dual, diagnostic)
    monkeypatch.setattr(BalancedMoments, 'apply', corrupt)
    with pytest.raises(FloatingPointError):
        nys.optimize(h, q, a, feature_map, phi, folder, 0, mass_mode='uniform', penalty=0.1, rank=3, chunk=7)
    assert not list(folder.glob('*.pt'))

@pytest.mark.parametrize('field,value', [('mass_mode', 'free'), ('balance_backend', 'cached'), ('balance_steps', 301), ('balance_tol', 1e-06), ('dual', None), ('balance', None)])
def test_balanced_snapshot_controls_and_required_evidence(tmp_path, field, value):
    h, q, a, feature_map = problem()
    phi = feature_map(h).detach().numpy()
    nys.optimize(h, q, a, feature_map, phi, tmp_path, 0, mass_mode='uniform', penalty=0.1, rank=3, chunk=7)
    snap = load(tmp_path / 'step_000000.pt')
    snap[field] = value
    with pytest.raises(ValueError):
        nys.validate_assignment_snapshot(snap, 'uniform')

def test_bad_resume_gauge_is_rejected_pre_copy():
    with pytest.raises(ValueError, match='gauge'):
        nys.validate_assignment_resume({'config': {'mass_mode': 'uniform', 'balance_steps': 300, 'balance_tol': 1e-08, 'balance_backend': 'chunked'}, 'step': 1, 'theta': torch.zeros(3, 9), 'dual': torch.full((4,), 0.2)}, 'uniform')

def test_balanced_resume_control_change_rejects_without_mutation(tmp_path):
    h, q, a, feature_map = problem()
    phi = feature_map(h).detach().numpy()
    nys.optimize(h, q, a, feature_map, phi, tmp_path, 0, mass_mode='uniform', penalty=0.1, rank=3, chunk=7)
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.iterdir() if p.is_file()}
    with pytest.raises(ValueError, match='controls'):
        nys.optimize(h, q, a, feature_map, phi, tmp_path, 1, mass_mode='uniform', balance_tol=1e-06, penalty=0.1, rank=3, chunk=7)
    assert before == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.iterdir() if p.is_file()}

def test_selected_test_preflight_rejects_wrong_inner_weighting_without_writes(tmp_path, citation_mock):
    ranking, root = run(tmp_path, [candidate(mass_mode='uniform')])
    choice = ranking[ranking.step == 1].iloc[0].to_dict()
    folder = Path(choice['candidate_path']) / 'condensation_0'
    snapshot = load(folder / 'step_000001.pt')
    snapshot['inner_loss_weighting'] = 'mass'
    save_state(snapshot, folder / 'step_000001.pt')
    before = citation_mock['data_calls']
    with pytest.raises(ValueError, match='inner weighting'):
        search.selected_test(root, choice, condensation_seeds=(0,), student_seeds=(1700,), epochs=1, device='cpu', report_routes=False)
    assert citation_mock['data_calls'] == before and (not (root / 'selected.json').exists())


def test_map_callable_portable_source_contract():
    assert nys.NystromMap.__call__.__code__.co_firstlineno == 39
    assert hashlib.sha256(inspect.getsource(nys.NystromMap.__call__).encode()).hexdigest() == (
        "9f5e0dbd5a0b3e7e3dcbdb7b31ee5f5a204db0497210d72f045fde3ab940c56a")


def test_default_and_explicit_free_optimizer_identity_and_values(tmp_path):
    h, q, assignment, feature_map = problem()
    phi = feature_map(h).detach().numpy()
    options = dict(penalty=.1, rank=3, chunk=7, checkpoint_every=1)
    nys.optimize(h, q, assignment, feature_map, phi, tmp_path / "default", 1, **options)
    nys.optimize(h, q, assignment, feature_map, phi, tmp_path / "explicit", 1, mass_mode="free", **options)
    original_state = load(tmp_path / "default" / "resume.pt")
    explicit_state = load(tmp_path / "explicit" / "resume.pt")
    original_snapshot = load(tmp_path / "default" / "step_000001.pt")
    explicit_snapshot = load(tmp_path / "explicit" / "step_000001.pt")
    assert original_state["config"] == explicit_state["config"] == dict(
        steps_schema=2, penalty=.1, lr=.01, rank=3, seed=0, cells=4, chunk=7)
    assert set(original_state) == set(explicit_state)
    assert set(original_snapshot) == set(explicit_snapshot)
    for field in ("mass_mode", "balance_steps", "balance_tol", "balance_backend"):
        assert field not in original_state["config"] and field not in original_snapshot
    assert "dual" not in original_state
    for key in ("u", "v", "theta", "vector"):
        torch.testing.assert_close(original_state[key], explicit_state[key], atol=0, rtol=0)
    torch.testing.assert_close(original_snapshot["moments"], explicit_snapshot["moments"], atol=0, rtol=0)
    assert original_snapshot["input_fingerprint"] == explicit_snapshot["input_fingerprint"]


@pytest.mark.parametrize("field,value", [("balance_backend", "cached"), ("balance_steps", 1.5),
                                       ("balance_tol", float("nan")), ("mass_mode", "invalid")])
def test_selected_test_preload_rejects_bad_balancing_controls(tmp_path, citation_mock, field, value):
    ranking, root = run(tmp_path, [candidate(mass_mode="uniform")])
    choice = ranking[ranking.step == 1].iloc[0].to_dict()
    choice[field] = value
    calls = citation_mock["data_calls"]
    with pytest.raises(ValueError):
        search.selected_test(root, choice, condensation_seeds=(0,), student_seeds=(1700,),
                             epochs=1, device="cpu", report_routes=False)
    assert citation_mock["data_calls"] == calls
    assert not (root / "selected.json").exists()


@pytest.mark.parametrize('field', ['balance_steps', 'balance_tol', 'balance_backend'])
@pytest.mark.parametrize('missing', [True, False])
def test_selected_balanced_control_absence_or_nan_cannot_fall_back_to_defaults(tmp_path, citation_mock, field, missing):
    ranking, root = run(tmp_path, [candidate(mass_mode='uniform')])
    choice = ranking[ranking.step == 1].iloc[0].to_dict()
    if missing:
        choice.pop(field)
    else:
        choice[field] = float('nan')
    calls = citation_mock['data_calls']
    with pytest.raises(ValueError):
        search.selected_test(root, choice, condensation_seeds=(0,), student_seeds=(1700,),
                             epochs=1, device='cpu', report_routes=False)
    assert citation_mock['data_calls'] == calls
    assert not (root / 'selected.json').exists()
