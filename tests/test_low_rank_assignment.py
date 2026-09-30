import pytest
import torch

from src.balanced_assignment import CachedBalancedMoments
from src.low_rank_assignment import (
    LowRankLogits,
    LowRankMoments,
    initialize_factors,
    logit_block,
)
from src.moments import AssignmentMoments, initial_logits, make_material
from src.soft_ce_partition import optimize_ce_assignment


def problem():
    generator = torch.Generator().manual_seed(91)
    z = torch.randn(12, 3, generator=generator, dtype=torch.double)
    q = torch.randn(12, 2, generator=generator, dtype=torch.double).softmax(1)
    assignment = torch.arange(12) % 4
    u = (0.2 * torch.randn(12, 2, generator=generator, dtype=torch.double)).requires_grad_()
    v = torch.randn(4, 2, generator=generator, dtype=torch.double).requires_grad_()
    return z, q, assignment, u, v


def test_initialization_preserves_dense_moments_and_has_live_gradient():
    z, q, assignment, _, _ = problem()
    material = make_material(z, q)
    u, v = initialize_factors(assignment, 4, 2, 7)
    actual = LowRankMoments.apply(u, v, assignment, material, 0.05, 5)
    expected = AssignmentMoments.apply(initial_logits(assignment, 4, 0.05), material, 5)
    torch.testing.assert_close(actual, expected, atol=1e-15, rtol=1e-14)
    actual.square().sum().backward()
    assert u.grad.norm() > 0
    torch.testing.assert_close(v.grad, torch.zeros_like(v))
    u2, v2 = initialize_factors(assignment, 4, 2, 7)
    torch.testing.assert_close(u, u2)
    torch.testing.assert_close(v, v2)


@pytest.mark.parametrize("chunk_size", [1, 5, 20])
def test_streamed_moments_match_dense_autograd(chunk_size):
    z, q, assignment, u, v = problem()
    material = make_material(z, q)
    actual = LowRankMoments.apply(u, v, assignment, material, 0.05, chunk_size)
    logits = initial_logits(assignment, 4, 0.05, torch.double) + u @ v.T / 2**0.5
    expected = logits.softmax(1).T @ material / len(z)
    gradient = torch.arange(actual.numel(), dtype=torch.double).reshape_as(actual) / actual.numel()
    actual_grad = torch.autograd.grad(actual, (u, v), gradient)
    expected_grad = torch.autograd.grad(expected, (u, v), gradient)
    torch.testing.assert_close(actual, expected)
    for a, b in zip(actual_grad, expected_grad):
        torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-10)


def test_factor_moments_and_logit_gradcheck():
    z, q, assignment, u, v = problem()
    material = make_material(z, q)
    assert torch.autograd.gradcheck(
        lambda a, b: LowRankMoments.apply(a, b, assignment, material, 0.05, 5), (u, v)
    )
    assert torch.autograd.gradcheck(lambda a, b: LowRankLogits.apply(a, b, assignment, 0.05, 5), (u, v))


def test_uniform_factor_gradient_and_marginals():
    z, q, assignment, u, v = problem()
    material = make_material(z, q)

    def balanced(logits):
        return CachedBalancedMoments.apply(logits, material, 5, 300, 1e-11, None, 512, 1e-10)[0]

    actual = balanced(LowRankLogits.apply(u, v, assignment, 0.05, 5))
    expected = balanced(logit_block(u, v, assignment, 0.05))
    actual_grad = torch.autograd.grad(actual.square().sum(), (u, v))
    expected_grad = torch.autograd.grad(expected.square().sum(), (u, v))
    torch.testing.assert_close(actual[:, 0], z.new_full((4,), 0.25), atol=1e-11, rtol=0)
    torch.testing.assert_close(actual.sum(0), material.mean(0))
    for a, b in zip(actual_grad, expected_grad):
        torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-9)


@pytest.mark.parametrize("mass_mode", ["free", "uniform"])
def test_bilevel_factor_checkpoint_roundtrip(tmp_path, mass_mode):
    z, q, assignment, _, _ = problem()
    result = optimize_ce_assignment(
        z,
        q,
        assignment,
        penalty=0.2,
        steps=2,
        lr=0.02,
        chunk_size=5,
        inner_tol=1e-9,
        cg_rtol=1e-9,
        folder=tmp_path,
        checkpoint_steps=[1],
        mass_mode=mass_mode,
        assignment_rank=2,
        factor_seed=7,
    )
    saved = torch.load(tmp_path / "best_assignment_factors.pt", weights_only=True)
    logits = logit_block(saved["u"], saved["v"], saved["assignment"], saved["mixing"]).double()
    if mass_mode == "uniform":
        logits += torch.load(tmp_path / "best_assignment_column_dual.pt", weights_only=True)
    moments = logits.softmax(1).T @ make_material(z, q) / len(z)
    torch.testing.assert_close(moments, result["best_moments"], atol=1e-8, rtol=1e-7)
    assert not (tmp_path / "best_assignment_logits.pt").exists()
    assert result["assignment_parameters"] == (12 + 4) * 2
    assert set(result["checkpoints"]) == {0, 1, 2}
    assert all(row["inner_converged"] for row in result["history"])
    assert result["best_J"] <= result["history"][0]["J"]
