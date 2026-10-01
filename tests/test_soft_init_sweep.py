import pytest
import torch

from src.low_rank_assignment import LowRankMoments
from src.soft_ce_partition import optimize_ce_assignment
from src.soft_init_sweep import distance_base


def problem():
    generator = torch.Generator().manual_seed(43)
    z = torch.randn(12, 3, dtype=torch.double, generator=generator)
    q = torch.randn(12, 2, dtype=torch.double, generator=generator).softmax(1)
    return z, q, torch.arange(12) % 3


def test_distance_probabilities():
    z, _, assignment = problem()
    centers = torch.stack([z[assignment == i].mean(0) for i in range(3)])
    expected = (-torch.cdist(z, centers).square() / 0.2).softmax(1)
    torch.testing.assert_close(distance_base(z, assignment, 0.2).softmax(1), expected)


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


def test_fixed_base_resume(tmp_path):
    z, q, assignment = problem()
    base = distance_base(z, assignment, 0.5)
    options = dict(base_logits=base, assignment_rank=2, penalty=0.2, inner_method="newton_first",
                   checkpoint_steps=(0, 1), save_resume=True, inner_tol=1e-8)
    optimize_ce_assignment(z, q, assignment, steps=1, folder=tmp_path, **options)
    state = torch.load(tmp_path / "resume.pt", weights_only=False)
    resumed = optimize_ce_assignment(z, q, assignment, steps=2, resume_state=state, **options)
    full = optimize_ce_assignment(z, q, assignment, steps=2, **options)
    torch.testing.assert_close(resumed["checkpoints"][2]["moments"], full["checkpoints"][2]["moments"])
    saved = torch.load(tmp_path / "best_assignment_factors.pt", weights_only=False)
    torch.testing.assert_close(saved["base_logits"], base)
    with pytest.raises(ValueError, match="Resume state"):
        optimize_ce_assignment(z, q, assignment, steps=2, resume_state=state,
                               **{**options, "base_logits": base * 2})
