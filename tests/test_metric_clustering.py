import pytest
import torch

from src.metric_clustering import metric_partition, metric_probability, normalize_metric, optimize_metric
from src.soft_ce_partition import implicit_moment_gradient, outer_value_gradient, solve_head_system, solve_inner
from src.soft_ridge_partition import augmented, decode_moments, make_material


def problem():
    generator = torch.Generator().manual_seed(21)
    z = torch.randn(11, 3, dtype=torch.double, generator=generator)
    q = torch.randn(11, 2, dtype=torch.double, generator=generator).softmax(1)
    centers = z[[0, 4, 8]].clone()
    weight = torch.eye(3, dtype=torch.double) + .1 * torch.randn(3, 3, dtype=torch.double, generator=generator)
    return z, q, centers, weight


def test_streamed_unrolled_metric_gradient_matches_dense_autograd():
    z, q, centers, weight = problem()
    material = make_material(z, q)
    weight.requires_grad_()
    actual = metric_partition(weight, z, material, centers, .7, 3, 4)
    multiplier = torch.arange(actual.numel(), dtype=z.dtype).reshape_as(actual) / actual.numel()
    derivative, = torch.autograd.grad((actual * multiplier).sum(), weight)
    current = centers
    for _ in range(3):
        p = metric_probability(z, current, normalize_metric(weight), .7)
        expected = p.T @ material / len(z)
        current, _, _ = decode_moments(expected, z.shape[1])
    dense, = torch.autograd.grad((expected * multiplier).sum(), weight)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(derivative, dense)
    torch.testing.assert_close(actual.sum(0), material.mean(0))
    torch.testing.assert_close(normalize_metric(weight).square().sum(), z.new_tensor(3.))
    scaled = metric_partition(4 * weight.detach(), z, material, centers, .7, 3, 4)
    torch.testing.assert_close(actual, scaled)


def test_metric_bilevel_derivative_matches_refitted_finite_difference():
    z, q, centers, weight = problem()
    material = make_material(z, q)
    weight.requires_grad_()
    def fit(w):
        moments = metric_partition(w, z, material, centers, .7, 2, 4)
        c, s, mass = decode_moments(moments.detach(), z.shape[1])
        fitted = solve_inner(c, s, mass, .2, grad_tol=1e-11)
        assert fitted['inner_converged']
        value, rhs = outer_value_gradient(z, q, fitted['theta'], 4)
        return moments, c, s, mass, fitted['theta'], value, rhs
    moments, c, s, mass, theta, _, rhs = fit(weight)
    vector, diagnostic = solve_head_system(augmented(c), s, mass, theta, .2, rhs, rtol=1e-10)
    assert diagnostic['cg_converged']
    derivative = implicit_moment_gradient(moments, z.shape[1], theta, vector, .2)
    gradient, = torch.autograd.grad(moments, weight, derivative)
    direction = torch.arange(1, 10, dtype=z.dtype).reshape(3, 3)
    direction /= direction.norm()
    eps = 1e-4
    plus = fit(weight.detach() + eps * direction)[5]
    minus = fit(weight.detach() - eps * direction)[5]
    expected = z.new_tensor((plus - minus) / (2 * eps))
    torch.testing.assert_close((gradient * direction).sum(), expected, atol=2e-6, rtol=2e-3)


def test_optimizer_resumes_after_checkpoint_update(tmp_path, monkeypatch):
    z, q, centers, _ = problem()
    settings = dict(steps=3, checkpoint_steps=[0, 1, 3], lloyd_steps=2,
                    tau=.7, metric_lr=.001, penalty=.2, chunk_size=4, inner_tol=1e-9)
    optimize_metric(z, q, centers, tmp_path / 'full', **settings)
    original = torch.optim.Adam.step
    calls = [0]
    def interrupted(optimizer, *args, **kwargs):
        calls[0] += 1
        if calls[0] == 3:
            raise RuntimeError('interrupted')
        return original(optimizer, *args, **kwargs)
    monkeypatch.setattr(torch.optim.Adam, 'step', interrupted)
    with pytest.raises(RuntimeError, match='interrupted'):
        optimize_metric(z, q, centers, tmp_path / 'resume', **settings)
    monkeypatch.setattr(torch.optim.Adam, 'step', original)
    optimize_metric(z, q, centers, tmp_path / 'resume', **settings)
    saved = [torch.load(tmp_path / name / 'step_000003.pt', weights_only=False) for name in ('full', 'resume')]
    for key in ('moments', 'metric', 'theta'):
        torch.testing.assert_close(saved[0][key], saved[1][key])
