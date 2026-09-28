import pandas as pd
import pytest
import torch

import src.trajectory_clustering as experiment


def problem():
    generator = torch.Generator().manual_seed(19)
    z = torch.randn(13, 3, dtype=torch.double, generator=generator)
    q = torch.randn(13, 4, dtype=torch.double, generator=generator).softmax(1)
    assignment = torch.arange(13) % 3
    centers = experiment.mean_centers(z, assignment, 3)
    heads = torch.randn(4, 4, 4, dtype=torch.double, generator=generator)
    targets = experiment.original_losses(z, q, heads, 5)
    return z, q, assignment, centers, heads, targets


@pytest.mark.parametrize('loss', ['absolute', 'squared'])
def test_chunked_loss_and_center_gradient_match_direct_autograd(loss):
    z, q, assignment, centers, heads, targets = problem()
    variable = centers.clone().requires_grad_()
    actual = experiment.assignment_objective(variable, heads, targets, q, assignment, 5, loss)
    derivative, = torch.autograd.grad(actual, variable)
    full = experiment.head_log_probability(heads, variable)
    predicted = -(full[:, assignment] * q[None]).sum(-1).T
    delta = predicted - targets
    expected = (delta.abs() if loss == 'absolute' else delta.square()).mean()
    direct, = torch.autograd.grad(expected, variable)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(derivative, direct)


@pytest.mark.parametrize('loss', ['absolute', 'squared'])
def test_actual_condensed_risk_is_bounded_with_mean_labels_and_mass(loss):
    z, q, assignment, centers, heads, targets = problem()
    labels, mass = experiment.cell_targets(q, assignment, len(centers))
    torch.testing.assert_close((mass[:, None] * labels).sum(0), q.mean(0))
    lp = experiment.head_log_probability(heads, centers)
    condensed = -(lp * labels[None] * mass[None, :, None]).sum((1, 2))
    substituted = -(lp[:, assignment] * q[None]).sum(-1).mean(1)
    torch.testing.assert_close(condensed, substituted)
    diagnostic = experiment.risk_diagnostics(centers, heads, targets, q, assignment, 4, loss)
    assert diagnostic['risk_gap'] <= diagnostic['risk_bound'] + 1e-12


@pytest.mark.parametrize('loss', ['absolute', 'squared'])
def test_assignment_retains_nonempty_cells_and_never_increases_cost(loss):
    q = torch.tensor([[1., 0.]], dtype=torch.double).repeat(12, 1)
    z = torch.zeros(12, 1, dtype=torch.double)
    centers = torch.arange(3, dtype=torch.double).reshape(3, 1)
    heads = torch.tensor([[[1., 0.], [0., 0.]]], dtype=torch.double)
    previous = torch.arange(12) % 3
    targets = experiment.original_losses(z, q, heads, 5)
    updated = experiment.assign_nodes(centers, heads, targets, q, previous, 5, loss)
    assert (torch.bincount(updated, minlength=3) > 0).all()
    assert (updated != previous).any()
    old = experiment.assignment_objective(centers, heads, targets, q, previous, 5, loss)
    new = experiment.assignment_objective(centers, heads, targets, q, updated, 5, loss)
    assert new <= old
    duplicate = torch.zeros_like(centers)
    tied = experiment.assign_nodes(duplicate, heads, targets, q, previous, 5, loss)
    torch.testing.assert_close(tied, previous)


def test_center_optimization_changes_features_and_keeps_best_objective():
    z, q, assignment, centers, heads, targets = problem()
    frozen = heads.clone()
    updated, record = experiment.update_centers(centers, heads, targets, q, assignment,
                                               5, .001, 5, 'squared')
    assert record['center_final'] < record['center_initial']
    assert not torch.equal(updated, centers)
    torch.testing.assert_close(heads, frozen)


def test_round_boundary_resume_reproduces_alternation(tmp_path, monkeypatch):
    z, q, assignment, centers, _, _ = problem()
    options = dict(rounds=2, lloyd_steps=1, center_steps=2, center_lr=.001,
                   trajectory_epochs=3, trajectory_lr=.01, trajectory_checkpoints=[0, 1, 3],
                   trajectory_seeds=[7], retain_rounds=2, chunk_size=5)
    experiment.optimize_trajectory(z, q, assignment, centers, tmp_path / 'full', **options)
    original = experiment.fit_ce_trajectory
    calls = [0]
    def interrupted(*args, **kwargs):
        calls[0] += 1
        if calls[0] == 2:
            raise RuntimeError('interrupted')
        return original(*args, **kwargs)
    monkeypatch.setattr(experiment, 'fit_ce_trajectory', interrupted)
    with pytest.raises(RuntimeError, match='interrupted'):
        experiment.optimize_trajectory(z, q, assignment, centers, tmp_path / 'resume', **options)
    monkeypatch.setattr(experiment, 'fit_ce_trajectory', original)
    experiment.optimize_trajectory(z, q, assignment, centers, tmp_path / 'resume', **options)
    states = [torch.load(tmp_path / mode / 'round_002.pt', weights_only=False) for mode in ('full', 'resume')]
    for key in ('centers', 'labels', 'mass', 'assignment'):
        torch.testing.assert_close(states[0][key], states[1][key])
    history = pd.read_csv(tmp_path / 'resume' / 'optimization.csv')
    for _, group in history.groupby('round'):
        assert (group.objective.diff().dropna() <= 1e-12).all()
    training = pd.read_csv(tmp_path / 'resume' / 'trajectory_training.csv')
    assert training.groupby('round').epoch.apply(list).tolist() == [[0, 1, 3], [0, 1, 3]]


def test_selection_uses_validation_and_breaks_ties_toward_earlier_rounds():
    table = pd.DataFrame([
        dict(candidate=0, round=1, val=72., test=80.),
        dict(candidate=1, round=0, val=73., test=60.),
        dict(candidate=2, round=2, val=73., test=90.),
    ])
    selected = experiment.select_validation(table)
    assert selected['candidate'] == 1
    assert selected['round'] == 0
