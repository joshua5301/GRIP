import pandas as pd
import pytest
import torch

import src.soft_ce_partition as model
from src.head import head_objective
from src.moments import (
    AssignmentMoments,
    augmented,
    decode_moments,
    initial_logits,
    make_material,
)


def problem():
    generator = torch.Generator().manual_seed(41)
    z = torch.randn(15, 3, generator=generator, dtype=torch.double)
    q = torch.randn(15, 3, generator=generator, dtype=torch.double).softmax(1)
    assignment = torch.arange(15) % 4
    return z, q, assignment


def test_hessian_and_pcg_match_dense_autograd():
    z, q, _ = problem()
    x = augmented(z)
    q = q * torch.linspace(0.9, 1.1, len(q), dtype=q.dtype)[:, None]
    mass = torch.arange(1, len(z) + 1, dtype=z.dtype)
    mass = mass / mass.sum()
    theta = torch.linspace(-0.4, 0.5, 12, dtype=z.dtype).reshape(3, 4)
    penalty = 0.03
    hessian = torch.autograd.functional.hessian(
        lambda value: head_objective(x, q, mass, value, penalty),
        theta,
    ).reshape(theta.numel(), theta.numel())
    multiply, diagonal = model.hessian_operator(x, q, mass, theta, penalty)
    rhs = theta.cos()
    torch.testing.assert_close(multiply(rhs).flatten(), hessian @ rhs.flatten())
    torch.testing.assert_close(diagonal.flatten(), hessian.diag())
    solution, diagnostic = model.conjugate_gradient(multiply, rhs, diagonal, rtol=1e-11)
    assert diagnostic["cg_converged"]
    expected = torch.linalg.solve(hessian, rhs.flatten()).reshape_as(rhs)
    torch.testing.assert_close(solution, expected, atol=1e-9, rtol=1e-9)
    zero, diagnostic = model.conjugate_gradient(multiply, rhs * 0, diagonal)
    assert diagnostic["cg_converged"] and diagnostic["cg_iterations"] == 0
    torch.testing.assert_close(zero, rhs * 0)


def test_extended_pcg_failure_uses_direct_fallback_without_relaxing_tolerance():
    z, q, _ = problem()
    x = augmented(z)
    mass = z.new_full((len(z),), 1 / len(z))
    theta = z.new_zeros(3, 4)
    rhs = torch.arange(12, dtype=z.dtype).reshape_as(theta).sin()
    penalty = 3e-6
    solution, diagnostic = model.solve_head_system(
        x, q, mass, theta, penalty, rhs, max_iter=0, reduced_limit=1, rtol=1e-8
    )
    hessian = torch.autograd.functional.hessian(
        lambda value: head_objective(x, q, mass, value, penalty),
        theta,
    ).reshape(theta.numel(), theta.numel())
    expected = torch.linalg.solve(hessian, rhs.flatten()).reshape_as(rhs)
    assert diagnostic["hessian_solver"] == "reduced_direct_fallback"
    assert diagnostic["cg_converged"] and diagnostic["cg_relative_residual"] <= 1e-8
    torch.testing.assert_close(solution, expected, atol=1e-6, rtol=1e-8)
    _, limited = model.solve_head_system(
        x, q, mass, theta, penalty, rhs, max_iter=0, reduced_limit=1, direct_limit=1
    )
    assert not limited["cg_converged"] and limited["hessian_solver"] == "pcg_extended"


def test_chunked_outer_gradient_matches_autograd():
    z, q, _ = problem()
    theta = torch.linspace(-0.3, 0.5, 12, dtype=z.dtype).reshape(3, 4).requires_grad_()
    expected = -(q * (augmented(z) @ theta.T).log_softmax(1)).sum(1).mean()
    (gradient,) = torch.autograd.grad(expected, theta)
    value, actual = model.outer_value_gradient(z, q, theta.detach(), chunk_size=4)
    assert abs(value - float(expected.detach())) < 1e-12
    torch.testing.assert_close(actual, gradient)


def test_implicit_assignment_gradient_matches_resolved_finite_difference():
    z, q, assignment = problem()
    material = make_material(z, q)
    logits = initial_logits(assignment, 4, 0.3, dtype=torch.double).requires_grad_()
    penalty = 0.2

    def solve(value):
        moments = AssignmentMoments.apply(value, material, 4)
        centers, labels, mass = decode_moments(moments.detach(), z.shape[1])
        fitted = model.solve_inner(centers, labels, mass, penalty, grad_tol=1e-11)
        assert fitted["inner_converged"]
        loss, gradient = model.outer_value_gradient(z, q, fitted["theta"], 4)
        return moments, fitted["theta"], loss, gradient

    moments, theta, _, outer_gradient = solve(logits)
    centers, labels, mass = decode_moments(moments.detach(), z.shape[1])
    multiply, diagonal = model.hessian_operator(augmented(centers), labels, mass, theta, penalty)
    vector, diagnostic = model.conjugate_gradient(multiply, outer_gradient, diagonal, rtol=1e-11, atol=1e-14)
    assert diagnostic["cg_converged"]
    gradient = model.implicit_moment_gradient(moments, z.shape[1], theta, vector, penalty)
    (actual,) = torch.autograd.grad(moments, logits, grad_outputs=gradient)
    for seed in (17, 31):
        direction = torch.randn(
            logits.shape, generator=torch.Generator().manual_seed(seed), dtype=logits.dtype
        )
        direction /= direction.norm()
        epsilon = 1e-3
        plus = solve(logits.detach() + epsilon * direction)[2]
        minus = solve(logits.detach() - epsilon * direction)[2]
        expected = (plus - minus) / (2 * epsilon)
        torch.testing.assert_close((actual * direction).sum(), z.new_tensor(expected), atol=1e-7, rtol=3e-3)


def test_checkpoint_reproduces_best_converged_ce_solution(tmp_path):
    z, q, assignment = problem()
    result = model.optimize_ce_assignment(
        z,
        q,
        assignment,
        penalty=0.2,
        steps=3,
        lr=0.02,
        chunk_size=4,
        inner_tol=1e-9,
        cg_rtol=1e-9,
        folder=tmp_path,
    )
    logits = torch.load(tmp_path / "best_assignment_logits.pt", weights_only=True)
    moments = AssignmentMoments.apply(logits, make_material(z, q), 4)
    torch.testing.assert_close(moments, result["best_moments"])
    centers, labels, mass = decode_moments(moments, z.shape[1])
    fitted = model.solve_inner(centers, labels, mass, 0.2, grad_tol=1e-10)
    value, _ = model.outer_value_gradient(z, q, fitted["theta"], 4)
    assert fitted["inner_converged"]
    assert abs(value - result["best_J"]) < 1e-7
    history = result["history"]
    assert all(row["inner_converged"] for row in history)
    assert all(row["cg_converged"] for row in history[:-1])
    assert all(b["best_J"] <= a["best_J"] for a, b in zip(history, history[1:]))


def test_unconverged_inner_cannot_update_assignments(tmp_path, monkeypatch):
    z, q, assignment = problem()

    def failed(*args, **kwargs):
        return dict(
            theta=z.new_zeros(q.shape[1], z.shape[1] + 1),
            inner_grad_max=1.0,
            inner_converged=False,
            inner_iterations=1,
            inner_polish_steps=0,
        )

    monkeypatch.setattr(model, "solve_inner", failed)
    with pytest.raises(RuntimeError, match="Inner CE did not converge"):
        model.optimize_ce_assignment(z, q, assignment, steps=1, folder=tmp_path)
    assert (tmp_path / "failure.json").exists()
    assert not (tmp_path / "best_assignment_logits.pt").exists()


def test_unconverged_implicit_solve_cannot_update_assignments(tmp_path, monkeypatch):
    z, q, assignment = problem()

    def fitted(*args, **kwargs):
        return dict(
            theta=z.new_zeros(q.shape[1], z.shape[1] + 1),
            inner_grad_max=0.0,
            inner_converged=True,
            inner_iterations=1,
            inner_polish_steps=0,
        )

    def failed(x, labels, mass, theta, penalty, rhs, **kwargs):
        return torch.zeros_like(rhs), dict(
            cg_converged=False,
            cg_iterations=1,
            cg_residual=1.0,
            cg_relative_residual=1.0,
            hessian_solver="failed",
        )

    monkeypatch.setattr(model, "solve_inner", fitted)
    monkeypatch.setattr(model, "solve_head_system", failed)
    with pytest.raises(RuntimeError, match="Implicit Hessian solve did not converge"):
        model.optimize_ce_assignment(z, q, assignment, steps=1, folder=tmp_path)
    assert (tmp_path / "failure.json").exists()
    assert not (tmp_path / "best_assignment_logits.pt").exists()


def test_checkpoints_store_current_iterates_without_changing_trajectory(tmp_path):
    z, q, assignment = problem()
    settings = dict(penalty=0.2, steps=3, lr=0.02, chunk_size=4, inner_tol=1e-9, cg_rtol=1e-9)
    plain = model.optimize_ce_assignment(z, q, assignment, **settings)
    traced = model.optimize_ce_assignment(
        z, q, assignment, folder=tmp_path, checkpoint_steps=[1, 2], **settings
    )
    assert set(traced["checkpoints"]) == {0, 1, 2, 3}
    torch.testing.assert_close(plain["best_moments"], traced["best_moments"], atol=0, rtol=0)
    for step, snapshot in traced["checkpoints"].items():
        saved = torch.load(tmp_path / "checkpoints" / f"step_{step:06d}.pt", weights_only=False)
        torch.testing.assert_close(saved["moments"], snapshot["moments"])
        value, _ = model.outer_value_gradient(z, q, saved["theta"], 4)
        assert abs(value - traced["history"][step]["J"]) < 1e-12
        assert abs(value - snapshot["teacher_ce"]) < 1e-12
    torch.testing.assert_close(traced["checkpoints"][0]["moments"], traced["initial_moments"])


def test_checkpoint_selection_uses_validation_and_earlier_step_tiebreak():
    table = pd.DataFrame(
        [
            dict(method="hard", checkpoint_step=float("nan"), gcn_val=99.0, gcn_test=99.0),
            dict(method="late", checkpoint_step=300, gcn_val=71.0, gcn_test=99.0),
            dict(method="early", checkpoint_step=100, gcn_val=71.0, gcn_test=50.0),
            dict(method="test_best", checkpoint_step=200, gcn_val=70.0, gcn_test=100.0),
        ]
    )
    assert model.select_checkpoint(table).method.iloc[0] == "early"


def test_uniform_mass_bilevel_checkpoints_and_saved_assignment(tmp_path):
    z, q, assignment = problem()
    result = model.optimize_ce_assignment(
        z,
        q,
        assignment,
        penalty=0.2,
        steps=2,
        lr=0.02,
        chunk_size=4,
        inner_tol=1e-9,
        cg_rtol=1e-9,
        folder=tmp_path,
        checkpoint_steps=[1],
        mass_mode="uniform",
        balance_tol=1e-11,
        balance_steps=1000,
    )
    for snapshot in result["checkpoints"].values():
        torch.testing.assert_close(snapshot["moments"][:, 0], z.new_full((4,), 0.25), atol=1e-11, rtol=0)
        torch.testing.assert_close(snapshot["moments"].sum(0), make_material(z, q).mean(0))
    logits = torch.load(tmp_path / "best_assignment_logits.pt", weights_only=True)
    dual = torch.load(tmp_path / "best_assignment_column_dual.pt", weights_only=True)
    probability = (logits.double() + dual).softmax(1)
    moments = probability.T @ make_material(z, q) / len(z)
    torch.testing.assert_close(moments, result["best_moments"])
    assert all(row["column_residual"] <= 1e-11 for row in result["history"])


def test_recovery_loads_only_existing_steps(tmp_path):
    folder = tmp_path / "checkpoints"
    folder.mkdir()
    for step in (0, 100, 750):
        torch.save(
            dict(step=step, moments=torch.ones(2, 4, dtype=torch.double)), folder / f"step_{step:06d}.pt"
        )
    snapshots = model.load_ce_snapshots(tmp_path)
    assert sorted(snapshots) == [0, 100, 750]


def test_recovery_requires_initial_checkpoint(tmp_path):
    with pytest.raises(ValueError, match="checkpoint zero"):
        model.load_ce_snapshots(tmp_path)


@pytest.mark.parametrize("rank_deficient", [False, True])
def test_reduced_hessian_fallback_matches_full_solve(rank_deficient):
    generator = torch.Generator().manual_seed(78)
    x = torch.randn(4, 11, generator=generator, dtype=torch.double)
    if rank_deficient:
        x[3] = x[0]
    labels = torch.randn(4, 3, generator=generator, dtype=torch.double).softmax(1)
    mass = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.double)
    theta = 0.2 * torch.randn(3, 11, generator=generator, dtype=torch.double)
    rhs = torch.randn(3, 11, generator=generator, dtype=torch.double)
    penalty = 1e-5
    hessian = torch.autograd.functional.hessian(
        lambda value: head_objective(x, labels, mass, value, penalty),
        theta,
    ).reshape(theta.numel(), theta.numel())
    expected = torch.linalg.solve(hessian, rhs.flatten()).reshape_as(rhs)
    actual, diagnostic = model.solve_head_system(x, labels, mass, theta, penalty, rhs, rtol=1e-8, max_iter=1)
    assert diagnostic["hessian_solver"] == "reduced_direct"
    assert diagnostic["cg_converged"]
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-8)
    assert diagnostic["cg_relative_residual"] <= 1e-8


def test_failed_fallback_is_not_accepted(monkeypatch):
    z, q, _ = problem()
    x = augmented(z)
    theta = z.new_zeros(3, 4)
    mass = z.new_full((len(z),), 1 / len(z))
    rhs = torch.arange(12, dtype=z.dtype).reshape_as(theta)
    monkeypatch.setattr(torch.linalg, "solve", lambda matrix, vector: torch.zeros_like(vector))
    _, diagnostic = model.solve_head_system(x, q, mass, theta, 1e-5, rhs, max_iter=1)
    assert diagnostic["hessian_solver"] == "reduced_direct"
    assert not diagnostic["cg_converged"]
