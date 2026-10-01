import pytest
import torch

from src.low_rank_assignment import CachedLowRankMoments, LowRankMoments
from src.soft_ce_partition import optimize_ce_assignment
from src.soft_init_sweep import distance_base, random_cost_base


def problem():
    generator = torch.Generator().manual_seed(43)
    z = torch.randn(12, 3, dtype=torch.double, generator=generator)
    q = torch.randn(12, 2, dtype=torch.double, generator=generator).softmax(1)
    return z, q, torch.arange(12) % 3


def test_random_cost_initialization():
    state = torch.random.get_rng_state()
    base = random_cost_base(100, 5, 7, 0.3, "cpu", torch.double)
    torch.testing.assert_close(torch.random.get_rng_state(), state)
    torch.testing.assert_close(base, random_cost_base(100, 5, 7, 0.3, "cpu", torch.double))
    assert not torch.equal(base, random_cost_base(100, 5, 8, 0.3, "cpu", torch.double))
    torch.testing.assert_close(base.mean(1), torch.zeros(100, dtype=torch.double), atol=1e-12, rtol=0)
    torch.testing.assert_close((base * 0.3).square().mean(), torch.ones((), dtype=torch.double))


def test_distance_probabilities():
    z, _, assignment = problem()
    centers = torch.stack([z[assignment == i].mean(0) for i in range(3)])
    expected = (-torch.cdist(z, centers).square() / 0.2).softmax(1)
    torch.testing.assert_close(distance_base(z, assignment, 0.2).softmax(1), expected)


def test_normalized_distance_scale():
    z, _, assignment = problem()
    base = distance_base(z, assignment, 1, normalized=True)
    centers = torch.stack([z[assignment == i].mean(0) for i in range(3)])
    distance = torch.cdist(z, centers).square()
    centered = distance - distance.mean(1, keepdim=True)
    torch.testing.assert_close(base, -centered / centered.square().mean().sqrt())
    torch.testing.assert_close(base.mean(1), torch.zeros(12, dtype=z.dtype), atol=1e-12, rtol=0)
    torch.testing.assert_close(base.square().mean(), torch.ones((), dtype=z.dtype))
    torch.testing.assert_close(base, distance_base(7 * z, assignment, 1, normalized=True))


@pytest.mark.parametrize("operation", [LowRankMoments, CachedLowRankMoments])
def test_cost_correction_gradient(operation):
    z, q, assignment = problem()
    t = 0.3
    base = distance_base(z, assignment, t, normalized=True)
    u = (z[:, :2] * 0.1).clone().requires_grad_()
    v = z[:3, :2].clone().requires_grad_()
    material = torch.cat((torch.ones(12, 1, dtype=z.dtype), z, q), dim=1)
    actual = operation.apply(u, -v / t, base, material, 0.05, 5)
    expected = (base - u @ v.T / (t * 2**0.5)).softmax(1).T @ material / 12
    torch.testing.assert_close(actual, expected)
    ga = torch.autograd.grad(actual.square().sum(), (u, v), retain_graph=True)
    ge = torch.autograd.grad(expected.square().sum(), (u, v))
    for a, e in zip(ga, ge):
        torch.testing.assert_close(a, e)


def test_fixed_base_low_rank_gradient():
    z, q, assignment = problem()
    base = distance_base(z, assignment, 0.5)
    u = (z[:, :2] * 0.1).clone().requires_grad_()
    v = z[:3, :2].clone().requires_grad_()
    material = torch.cat((torch.ones(12, 1, dtype=z.dtype), z, q), dim=1)
    actual = LowRankMoments.apply(u, v, base, material, 0.05, 5)
    expected = (base + u @ v.T / 2**0.5).softmax(1).T @ material / 12
    torch.testing.assert_close(actual, expected)
    ga = torch.autograd.grad(actual.square().sum(), (u, v), retain_graph=True)
    ge = torch.autograd.grad(expected.square().sum(), (u, v))
    for a, e in zip(ga, ge):
        torch.testing.assert_close(a, e)


@pytest.mark.parametrize("scale", [1.0, -3.0])
def test_fixed_base_resume(tmp_path, scale):
    z, q, assignment = problem()
    base = distance_base(z, assignment, 0.5)
    options = dict(base_logits=base, assignment_rank=2, penalty=0.2, inner_method="newton_first",
                   checkpoint_steps=(0, 1), save_resume=True, inner_tol=1e-8, correction_scale=scale)
    optimize_ce_assignment(z, q, assignment, steps=1, folder=tmp_path, **options)
    state = torch.load(tmp_path / "resume.pt", weights_only=False)
    resumed = optimize_ce_assignment(z, q, assignment, steps=2, resume_state=state, **options)
    full = optimize_ce_assignment(z, q, assignment, steps=2, **options)
    torch.testing.assert_close(resumed["checkpoints"][2]["moments"], full["checkpoints"][2]["moments"])
    saved = torch.load(tmp_path / "best_assignment_factors.pt", weights_only=False)
    torch.testing.assert_close(saved["base_logits"], base)
    assert saved["correction_scale"] == scale
    with pytest.raises(ValueError, match="Resume state"):
        optimize_ce_assignment(z, q, assignment, steps=2, resume_state=state,
                               **{**options, "base_logits": base * 2})
    with pytest.raises(ValueError, match="Resume state"):
        optimize_ce_assignment(z, q, assignment, steps=2, resume_state=state,
                               **{**options, "correction_scale": scale * 2})
