import pytest
import torch

from src.soft_ce_partition import solve_inner
from src.soft_ridge_partition import AssignmentMoments, augmented, decode_moments, initial_logits, make_material
from src.stationarity_risk import head_objective
from src.stationary_assignment import optimize_stationary, stationary_objective


def problem():
    generator = torch.Generator().manual_seed(17)
    z = torch.randn(12, 3, dtype=torch.double, generator=generator)
    q = torch.randn(12, 3, dtype=torch.double, generator=generator).softmax(1)
    assignment = torch.arange(12) % 3
    theta = torch.randn(3, 4, dtype=torch.double, generator=generator)
    return z, q, assignment, theta


def test_stationary_objective_and_assignment_derivative_match_autograd():
    z, q, assignment, theta = problem()
    logits = initial_logits(assignment, 3, dtype=torch.double).requires_grad_()
    moments = AssignmentMoments.apply(logits, make_material(z, q), 4)
    penalty = .2
    loss, gradient = stationary_objective(moments, 3, theta, penalty)
    actual, = torch.autograd.grad(loss, logits)
    variable = logits.detach().clone().requires_grad_()
    direct = variable.softmax(1).T @ make_material(z, q) / len(z)
    centers, labels, mass = decode_moments(direct, 3)
    weight = theta.clone().requires_grad_()
    ce = head_objective(augmented(centers), labels, mass, weight, penalty)
    expected_gradient, = torch.autograd.grad(ce, weight, create_graph=True)
    expected, = torch.autograd.grad(expected_gradient.square().sum(), variable)
    torch.testing.assert_close(gradient, expected_gradient)
    torch.testing.assert_close(actual, expected)


def test_residual_controls_distance_to_condensed_optimum():
    z, q, assignment, theta = problem()
    moments = initial_logits(assignment, 3, dtype=torch.double).softmax(1).T @ make_material(z, q) / len(z)
    centers, labels, mass = decode_moments(moments, 3)
    _, gradient = stationary_objective(moments, 3, theta, .3)
    fitted = solve_inner(centers, labels, mass, .3, grad_tol=1e-10)
    assert fitted['inner_converged']
    assert float((fitted['theta'] - theta).norm()) <= float(gradient.norm()) / .3 + 1e-8


def test_fixed_head_optimization_resumes_without_fitting_students(tmp_path, monkeypatch):
    z, q, assignment, theta = problem()
    frozen = theta.clone()
    args = dict(steps=4, lr=.001, chunk_size=4, checkpoint_steps=[0, 2, 4])
    optimize_stationary(z, q, assignment, theta, .2, tmp_path / 'full', **args)
    original_step = torch.optim.Adam.step
    counter = [0]
    def interrupted(optimizer, *a, **kw):
        counter[0] += 1
        if counter[0] == 3:
            raise RuntimeError('simulated interruption')
        return original_step(optimizer, *a, **kw)
    monkeypatch.setattr(torch.optim.Adam, 'step', interrupted)
    with pytest.raises(RuntimeError, match='simulated interruption'):
        optimize_stationary(z, q, assignment, theta, .2, tmp_path / 'resume', **args)
    monkeypatch.setattr(torch.optim.Adam, 'step', original_step)
    optimize_stationary(z, q, assignment, theta, .2, tmp_path / 'resume', **args)
    states = [torch.load(tmp_path / name / 'checkpoints' / 'step_000004.pt', weights_only=False)
              for name in ('full', 'resume')]
    torch.testing.assert_close(states[0]['moments'], states[1]['moments'])
    torch.testing.assert_close(theta, frozen)
    initial = torch.load(tmp_path / 'full' / 'checkpoints' / 'step_000000.pt', weights_only=False)
    assert states[0]['objective'] < initial['objective']
