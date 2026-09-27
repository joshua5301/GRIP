import pytest
import torch

from src.risk_partition import _local_deltas, risk_partition


def direct(x, q, ids, m, B, local):
    errors, variance = [], x.new_tensor(0.)
    for cell in range(m):
        xx, qq = x[ids == cell], q[ids == cell]
        dx, dq = xx - xx.mean(0), qq - qq.mean(0)
        variance += dx.square().sum() / len(x)
        errors.append(dx.T @ dq / len(x))
    errors = torch.stack(errors)
    moment = errors.flatten(1).norm(dim=1).sum() if local else errors.sum(0).norm()
    return B * B / 4 * variance + 2 * B * moment, errors


def test_local_move_deltas_match_every_legal_full_recomputation():
    generator = torch.Generator().manual_seed(19)
    x = torch.randn(8, 4, dtype=torch.double, generator=generator)
    q = torch.randn(8, 3, dtype=torch.double, generator=generator).softmax(1)
    assignment = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2])
    counts = torch.bincount(assignment).double()
    c = torch.stack([x[assignment == j].mean(0) for j in range(3)])
    y = torch.stack([q[assignment == j].mean(0) for j in range(3)])
    before, errors = direct(x, q, assignment, 3, 1.7, True)
    delta = _local_deltas(x, q, torch.arange(8), assignment, counts, c, y, errors, 1.7 ** 2 / 4, 3.4)
    for node in range(8):
        for target in range(3):
            if assignment[node] != target:
                changed = assignment.clone()
                changed[node] = target
                after, _ = direct(x, q, changed, 3, 1.7, True)
                torch.testing.assert_close(delta[node, target], after - before, atol=1e-8, rtol=1e-7)


def test_local_penalty_detects_canceling_cell_moments():
    h = torch.tensor([[-1.], [1.], [3.], [5.]], dtype=torch.double)
    q = torch.tensor([[1., 0.], [0., 1.], [0., 1.], [1., 0.]], dtype=torch.double)
    initial = dict(assignment=torch.tensor([0, 0, 1, 1]), generator_state=torch.Generator().get_state())
    global_result = risk_partition(h, q, 2, 1., max_sweeps=0, initial_state=initial)
    local_result = risk_partition(h, q, 2, 1., max_sweeps=0, initial_state=initial, objective_mode='local')
    assert global_result['moment_error'] < 1e-12
    assert local_result['local_moment_error'] > .1
    assert local_result['J'] > global_result['J']
    assert local_result['global_moment_error'] < 1e-12
    assert local_result['V'] == pytest.approx(global_result['V'])
    torch.testing.assert_close(local_result['x'], global_result['x'])
    torch.testing.assert_close(local_result['y'], global_result['y'])


def test_both_modes_start_with_identical_assignment_and_move_rng():
    generator = torch.Generator().manual_seed(8)
    h = torch.randn(18, 5, generator=generator)
    q = torch.randn(18, 3, generator=generator).softmax(1)
    outputs = [risk_partition(h, q, 4, 2., seed=17, max_sweeps=0, return_initial_state=True,
                               objective_mode=mode) for mode in ('combined', 'local')]
    for key in ('assignment', 'generator_state'):
        assert torch.equal(outputs[0]['initial_state'][key], outputs[1]['initial_state'][key])


@pytest.mark.parametrize('mode', ['combined', 'local'])
@pytest.mark.parametrize('B', [.1, 2., 15.])
def test_relocation_preserves_means_budget_and_decreases_exact_objective(mode, B):
    generator = torch.Generator().manual_seed(23)
    h = torch.randn(20, 4, dtype=torch.double, generator=generator)
    q = torch.randn(20, 3, dtype=torch.double, generator=generator).softmax(1)
    result = risk_partition(h, q, 4, B, seed=3, max_sweeps=8, block_size=5,
                            objective_mode=mode, return_assignment=True)
    assignment = result['assignment']
    x = h - h.mean(0)
    x = x / x.square().sum(1).mean().sqrt()
    expected, _ = direct(x, q, assignment, 4, B, mode == 'local')
    assert result['J'] == pytest.approx(float(expected), abs=1e-8)
    assert all(b <= a + 1e-8 for a, b in zip(result['history'], result['history'][1:]))
    assert result['counts'].sum() == len(h) and result['counts'].min() > 0
    for j in range(4):
        torch.testing.assert_close(result['x'][j], h[assignment == j].mean(0).float())
        torch.testing.assert_close(result['y'][j], q[assignment == j].mean(0).float())


@pytest.mark.parametrize('m', [1, 5])
def test_local_degenerate_budget(m):
    h = torch.zeros(5, 3)
    q = torch.full((5, 2), .5)
    result = risk_partition(h, q, m, 1., objective_mode='local')
    assert result['J'] == 0 and result['converged']
    assert len(result['x']) == m and result['counts'].min() > 0


def test_runner_changes_only_objective_mode_and_keeps_uniform_student(tmp_path, monkeypatch):
    import src.risk_experiment as experiment

    h = torch.tensor([[0., 1.], [1., 0.], [2., 1.], [3., 0.]])
    graph = dict(x=h, y=torch.tensor([0, 1, 0, 1]), adj=torch.eye(4))
    mask = torch.tensor([True, True, False, False])
    monkeypatch.setattr(experiment, '_prepare_dataset', lambda *args: (graph, mask, (graph, ~mask), (graph, ~mask), h))
    monkeypatch.setattr(experiment, 'get_kernel_features', lambda h, *args: h)
    monkeypatch.setattr(experiment, 'fit_logistic', lambda x, y, gamma: torch.zeros(x.shape[1], y.shape[1]))
    monkeypatch.setitem(experiment.BUDGET, ('cora', .026), 2)
    calls, students = [], []

    def partition(H, Q, m, B, **kwargs):
        calls.append((H.clone(), Q.clone(), B, kwargs))
        return dict(x=h[:2], y=Q[:2], counts=torch.tensor([2, 2]), history=[1.], J=.5,
                    converged=True, sweeps=1, seconds=0., objective_name='local_moment_surrogate',
                    global_moment_error=.1, local_moment_error=.2, cancellation_ratio=.5,
                    variance_term=.1, moment_term=.4)

    def student(cx, cy, validation, params, seed, settings, testing=None, counts=None, adjacency=None):
        assert settings['loss_weighting'] == 'uniform'
        assert adjacency is None
        students.append((seed, testing is not None))
        return .8, (.75 if testing is not None else None), 1

    monkeypatch.setattr(experiment, 'risk_partition', partition)
    monkeypatch.setattr(experiment, '_train_student', student)
    space = dict(teacher_kernel=['relu'], gamma=[.01], T=[1.], basis=[3000], B=[1.], dropout=[.9])
    for method in ('risk', 'risk_local'):
        result = experiment.run_experiments({'cora': [.026]}, tmp_path / method, space=space,
                                            method=method, search='grid', device='cpu',
                                            search_seeds=(0,), final_seeds=(100,), seed=0)
        assert result.iloc[0]['test_mean'] == pytest.approx(75.)
    torch.testing.assert_close(calls[0][0], calls[1][0])
    torch.testing.assert_close(calls[0][1], calls[1][1])
    options = calls[1][3].copy()
    assert options.pop('objective_mode') == 'local'
    assert options == calls[0][3]
    assert students == [(0, False), (100, True)] * 2
