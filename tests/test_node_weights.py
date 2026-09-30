import pytest
import torch

from src.low_rank_assignment import (
    LowRankMoments,
    WeightedLowRankMoments,
    logit_block,
    normalized_node_weights,
)
from src.moments import make_material
from src.soft_ce_partition import optimize_ce_assignment


def problem():
    g = torch.Generator().manual_seed(41)
    z = torch.randn(9, 3, generator=g, dtype=torch.double)
    q = torch.randn(9, 2, generator=g, dtype=torch.double).softmax(1)
    assignment = torch.arange(9) % 3
    u = torch.randn(9, 2, generator=g, dtype=torch.double).requires_grad_()
    v = torch.randn(3, 2, generator=g, dtype=torch.double).requires_grad_()
    a = torch.randn(9, generator=g, dtype=torch.double).requires_grad_()
    return z, q, assignment, u, v, a


@pytest.mark.parametrize("chunk", [1, 4, 20])
def test_weighted_gradients_and_mass(chunk):
    z, q, assignment, u, v, a = problem()
    w, kl = normalized_node_weights(a)
    material = make_material(z, q)
    actual = WeightedLowRankMoments.apply(u, v, w, assignment, material, 0.05, chunk)
    expected = logit_block(u, v, assignment, 0.05).softmax(1).T @ (w[:, None] * material) / len(z)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual[:, 0].sum(), z.new_tensor(1.0))
    torch.testing.assert_close(actual.sum(0), (w[:, None] * material).mean(0))
    target = torch.arange(actual.numel(), dtype=torch.double).reshape_as(actual) / actual.numel()
    ga = torch.autograd.grad((actual * target).sum() + 0.3 * kl, (u, v, a), retain_graph=True)
    ge = torch.autograd.grad((expected * target).sum() + 0.3 * kl, (u, v, a))
    for left, right in zip(ga, ge):
        torch.testing.assert_close(left, right, atol=1e-12, rtol=1e-10)


def test_uniform_and_shift_invariance():
    z, q, assignment, u, v, a = problem()
    w, kl = normalized_node_weights(torch.zeros_like(a))
    torch.testing.assert_close(w, torch.ones_like(w))
    torch.testing.assert_close(kl, z.new_tensor(0.0), atol=1e-14, rtol=0)
    material = make_material(z, q)
    actual = WeightedLowRankMoments.apply(u, v, w, assignment, material, 0.05, 4)
    expected = LowRankMoments.apply(u, v, assignment, material, 0.05, 4)
    torch.testing.assert_close(actual, expected)
    first = normalized_node_weights(a)
    second = normalized_node_weights(a + 20)
    for left, right in zip(first, second):
        torch.testing.assert_close(left, right)


def test_weighted_gradcheck():
    z, q, assignment, u, v, a = problem()

    def forward(u, v, a):
        w, _ = normalized_node_weights(a)
        return WeightedLowRankMoments.apply(u, v, w, assignment, make_material(z, q), 0.05, 4)

    assert torch.autograd.gradcheck(forward, (u, v, a))


def test_optimizer_resume_and_saved_weights(tmp_path):
    z, q, assignment, _, _, _ = problem()
    options = dict(
        penalty=0.3,
        lr=0.01,
        assignment_rank=2,
        factor_seed=4,
        node_weighting=True,
        node_weight_penalty=0.1,
        node_weight_lr=0.005,
        save_resume=True,
        inner_tol=1e-8,
        cg_rtol=1e-9,
    )
    full = optimize_ce_assignment(
        z, q, assignment, steps=2, checkpoint_steps=[0, 1, 2], folder=tmp_path / "full", **options
    )
    optimize_ce_assignment(
        z, q, assignment, steps=1, checkpoint_steps=[0, 1], folder=tmp_path / "split", **options
    )
    resume = torch.load(tmp_path / "split" / "resume.pt", weights_only=False)
    continued = optimize_ce_assignment(
        z,
        q,
        assignment,
        steps=2,
        checkpoint_steps=[0, 1, 2],
        folder=tmp_path / "split",
        resume_state=resume,
        **options,
    )
    torch.testing.assert_close(
        full["checkpoints"][2]["moments"], continued["checkpoints"][2]["moments"], atol=2e-7, rtol=2e-6
    )
    saved = torch.load(tmp_path / "full" / "best_assignment_factors.pt", weights_only=True)
    w, _ = normalized_node_weights(saved["node_weight_logits"])
    probability = (
        logit_block(saved["u"], saved["v"], saved["assignment"], saved["mixing"]).double().softmax(1)
    )
    reconstructed = probability.T @ (w[:, None] * make_material(z, q)) / len(z)
    torch.testing.assert_close(reconstructed, full["best_moments"], atol=1e-10, rtol=1e-8)
    assert full["assignment_parameters"] == (9 + 3) * 2 + 9
    assert full["checkpoints"][2]["node_weight_logits"].abs().max() > 0
    assert full["history"][-1]["node_ess_fraction"] <= 1 + 1e-12


def test_old_resume_configuration_still_loads(tmp_path):
    z, q, assignment, _, _, _ = problem()
    options = dict(penalty=0.3, assignment_rank=2, save_resume=True, save_assignment=False)
    optimize_ce_assignment(z, q, assignment, steps=0, folder=tmp_path, **options)
    resume = torch.load(tmp_path / "resume.pt", weights_only=False)
    assert "node_weighting" not in resume["config"]
    optimize_ce_assignment(z, q, assignment, steps=1, resume_state=resume, **options)
