import numpy as np
import pytest
import torch

from src.low_rank_assignment import CachedLowRankMoments, LowRankMoments, logit_block
from src.moments import augmented, decode_moments, make_material
from src.soft_ce_partition import (
    implicit_moment_gradient,
    optimize_ce_assignment,
    outer_value_gradient,
    solve_head_system,
    solve_inner,
    temperature_labels,
)


def problem():
    g = torch.Generator().manual_seed(37)
    z = torch.randn(12, 3, generator=g, dtype=torch.double)
    logits = torch.randn(12, 3, generator=g, dtype=torch.double)
    u = torch.randn(12, 2, generator=g, dtype=torch.double).requires_grad_()
    v = torch.randn(4, 2, generator=g, dtype=torch.double).requires_grad_()
    return z, logits, u, v, torch.arange(12) % 4


@pytest.mark.parametrize("operation", [LowRankMoments, CachedLowRankMoments])
def test_temperature_moment_backward(operation):
    z, logits, u, v, assignment = problem()
    t = z.new_tensor(np.log(0.7)).requires_grad_()
    material = make_material(z, temperature_labels(logits, t))
    actual = operation.apply(u, v, assignment, material, 0.05, 5)
    expected = logit_block(u, v, assignment, 0.05).softmax(1).T @ material / len(z)
    weight = torch.arange(actual.numel(), dtype=z.dtype).reshape_as(actual).sin()
    left = torch.autograd.grad((actual * weight).sum(), (u, v, t), retain_graph=True)
    right = torch.autograd.grad((expected * weight).sum(), (u, v, t))
    for a, b in zip(left, right):
        torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-8)


def test_implicit_temperature_gradient_finite_difference():
    z, logits, u, v, assignment = problem()
    q_reference = temperature_labels(logits, z.new_tensor(np.log(0.3))).detach()
    t = z.new_tensor(np.log(0.7)).requires_grad_()

    def fitted(value):
        material = make_material(z, temperature_labels(logits, value))
        moments = CachedLowRankMoments.apply(u, v, assignment, material, 0.05, 5)
        centers, labels, mass = decode_moments(moments.detach(), z.shape[1])
        head = solve_inner(centers, labels, mass, 0.2, grad_tol=1e-11)
        assert head["inner_converged"]
        loss, gradient = outer_value_gradient(z, q_reference, head["theta"])
        return moments, centers, labels, mass, head["theta"], loss, gradient

    moments, centers, labels, mass, theta, _, gradient = fitted(t)
    vector, info = solve_head_system(augmented(centers), labels, mass, theta, 0.2, gradient, rtol=1e-10)
    assert info["cg_converged"]
    direction = implicit_moment_gradient(moments, z.shape[1], theta, vector, 0.2)
    (actual,) = torch.autograd.grad(moments, t, direction)
    eps = 1e-3
    finite = (fitted(t.detach() + eps)[5] - fitted(t.detach() - eps)[5]) / (2 * eps)
    torch.testing.assert_close(actual, actual.new_tensor(finite), atol=2e-6, rtol=2e-3)


def test_temperature_resume_and_fixed_outer(tmp_path):
    z, logits, _, _, assignment = problem()
    q = temperature_labels(logits, z.new_tensor(np.log(0.3))).detach()
    q_before = q.clone()
    options = dict(
        penalty=0.2,
        assignment_rank=2,
        inner_method="newton_first",
        cache_assignment=True,
        temperature_logits=logits,
        temperature_initial=0.3,
        temperature_lr=0.01,
        save_assignment=False,
        save_resume=True,
        checkpoint_steps=(0, 2),
    )
    result = optimize_ce_assignment(z, q, assignment, steps=2, folder=tmp_path, **options)
    state = torch.load(tmp_path / "resume.pt", weights_only=False)
    expected = float(state["parameters"][-1].exp())
    resumed = optimize_ce_assignment(z, q, assignment, steps=2, resume_state=state, **options)
    assert resumed["history"][-1]["temperature"] == expected
    assert all(np.isfinite(row["temperature"]) and row["temperature"] > 0 for row in result["history"])
    assert any(abs(row["log_temperature_gradient"]) > 1e-10 for row in result["history"] if row["step"] < 2)
    torch.testing.assert_close(q, q_before, atol=0, rtol=0)
    last = result["checkpoints"][2]
    loss, _ = outer_value_gradient(z, q, last["theta"])
    assert abs(loss - last["teacher_ce"]) < 1e-10
