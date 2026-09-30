import pytest
import torch

from src.moments import augmented, decode_moments, make_material
from src.soft_ce_partition import (
    implicit_moment_gradient,
    optimize_ce_assignment,
    outer_value_gradient,
    solve_head_system,
    solve_inner,
)


def test_uniform_hypergradient_finite_difference():
    g = torch.Generator().manual_seed(77)
    z = torch.randn(11, 3, generator=g, dtype=torch.double)
    q = torch.randn(11, 3, generator=g, dtype=torch.double).softmax(1)
    logits = torch.randn(11, 4, generator=g, dtype=torch.double)
    perturbation = torch.randn(11, 4, generator=g, dtype=torch.double)
    t = z.new_tensor(0.2).requires_grad_()

    def fitted(value):
        moments = (logits + value * perturbation).softmax(1).T @ make_material(z, q) / len(z)
        centers, labels, mass = decode_moments(moments.detach(), z.shape[1])
        weights = torch.full_like(mass, 1 / len(mass))
        head = solve_inner(centers, labels, weights, 0.2, grad_tol=1e-11)
        assert head["inner_converged"]
        loss, gradient = outer_value_gradient(z, q, head["theta"])
        return moments, centers, labels, weights, head["theta"], loss, gradient

    moments, centers, labels, weights, theta, _, gradient = fitted(t)
    vector, diagnostic = solve_head_system(
        augmented(centers), labels, weights, theta, 0.2, gradient, rtol=1e-10
    )
    assert diagnostic["cg_converged"]
    direction = implicit_moment_gradient(moments, z.shape[1], theta, vector, 0.2, "uniform")
    (actual,) = torch.autograd.grad(moments, t, direction)
    eps = 1e-3
    finite = (fitted(t.detach() + eps)[5] - fitted(t.detach() - eps)[5]) / (2 * eps)
    torch.testing.assert_close(actual, actual.new_tensor(finite), atol=2e-6, rtol=2e-3)


def test_uniform_initial_moments_and_resume(tmp_path):
    g = torch.Generator().manual_seed(22)
    z = torch.randn(12, 3, generator=g, dtype=torch.double)
    q = torch.randn(12, 3, generator=g, dtype=torch.double).softmax(1)
    assignment = torch.tensor([0] * 7 + [1] * 3 + [2] * 2)
    options = dict(
        steps=0,
        penalty=0.2,
        assignment_rank=2,
        checkpoint_steps=(0,),
        save_assignment=False,
        save_resume=True,
        inner_method="newton_first",
    )
    mass = optimize_ce_assignment(z, q, assignment, **options)
    uniform = optimize_ce_assignment(
        z, q, assignment, inner_loss_weighting="uniform", folder=tmp_path, **options
    )
    torch.testing.assert_close(mass["initial_moments"], uniform["initial_moments"], atol=0, rtol=0)
    state = torch.load(tmp_path / "resume.pt", weights_only=False)
    assert state["config"]["inner_loss_weighting"] == "uniform"
    with pytest.raises(ValueError, match="Resume state"):
        optimize_ce_assignment(z, q, assignment, resume_state=state, **options)
    resumed = optimize_ce_assignment(
        z, q, assignment, resume_state=state, inner_loss_weighting="uniform", **dict(options, steps=1)
    )
    assert all(row["inner_converged"] for row in resumed["history"])
