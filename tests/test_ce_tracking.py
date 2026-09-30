import pytest
import torch

import src.soft_ce_partition as model
from src.head import head_objective
from src.moments import augmented


def problem():
    generator = torch.Generator().manual_seed(27)
    z = torch.randn(18, 4, generator=generator, dtype=torch.double)
    q = torch.randn(18, 3, generator=generator, dtype=torch.double).softmax(1)
    return z, q, torch.arange(18) % 5


def options():
    return dict(
        penalty=0.2,
        lr=0.003,
        chunk_size=6,
        inner_tol=1e-9,
        cg_rtol=1e-9,
        assignment_rank=2,
        assignment_input="features",
        assignment_encoder="mlp",
        encoder_hidden=6,
        save_assignment=False,
        solver_mode="tracking",
        tracking_inner_steps=1,
        tracking_cg_steps=2,
        tracking_refresh=3,
    )


def test_pcg_reuses_previous_solution_without_mutating_it():
    matrix = torch.tensor([[3.0, 1.0], [1.0, 2.0]], dtype=torch.double)
    rhs = torch.tensor([1.0, -2.0], dtype=torch.double)
    initial = torch.linalg.solve(matrix, rhs)
    saved = initial.clone()
    answer, diagnostic = model.conjugate_gradient(lambda v: matrix @ v, rhs, matrix.diag(), initial=initial)
    assert diagnostic["cg_iterations"] == 0
    torch.testing.assert_close(answer, saved)
    shifted = rhs + 0.01
    answer, diagnostic = model.conjugate_gradient(
        lambda v: matrix @ v, shifted, matrix.diag(), initial=initial, rtol=1e-12
    )
    assert diagnostic["cg_converged"]
    torch.testing.assert_close(answer, torch.linalg.solve(matrix, shifted))
    torch.testing.assert_close(initial, saved)


def test_tracking_head_decreases_inner_objective_with_bounded_work(monkeypatch):
    z, q, _ = problem()
    mass = z.new_full((len(z),), 1 / len(z))
    initial = z.new_zeros(q.shape[1], z.shape[1] + 1)

    def forbidden(*args, **kwargs):
        raise AssertionError("Tracking must not invoke an exact solve")

    monkeypatch.setattr(model, "solve_head_system", forbidden)
    fitted = model.track_inner(z, q, mass, 0.2, initial, steps=2, cg_steps=2)
    assert not fitted["tracking_failed"]
    assert 0 < fitted["inner_iterations"] <= 2
    before = head_objective(augmented(z), q, mass, initial, 0.2)
    after = head_objective(augmented(z), q, mass, fitted["theta"], 0.2)
    assert after < before
    torch.testing.assert_close(initial, torch.zeros_like(initial))


def test_tracking_checkpoints_are_exact_and_noncheckpoints_use_pcg(tmp_path):
    z, q, assignment = problem()
    result = model.optimize_ce_assignment(
        z, q, assignment, steps=5, checkpoint_steps=[2], folder=tmp_path, **options()
    )
    tracked = [row for row in result["history"] if not row["exact_refresh"]]
    assert tracked
    assert all(row["hessian_solver"] == "tracking_pcg" and row["cg_iterations"] <= 2 for row in tracked)
    for step, snapshot in result["checkpoints"].items():
        row = result["history"][step]
        assert row["exact_refresh"] and row["inner_converged"] and snapshot["J_exact"]
        value, _ = model.outer_value_gradient(z, q, snapshot["theta"])
        assert abs(value - snapshot["teacher_ce"]) < 1e-12
    assert result["history"][result["best_step"]]["J_exact"]


def test_tracking_resume_preserves_head_and_adjoint_trajectory(tmp_path):
    z, q, assignment = problem()
    args = dict(**options(), checkpoint_steps=[3], save_resume=True)
    full = model.optimize_ce_assignment(z, q, assignment, steps=6, **args)
    model.optimize_ce_assignment(z, q, assignment, steps=3, folder=tmp_path, **args)
    state = torch.load(tmp_path / "resume.pt", weights_only=False)
    assert state["tracking_vector_before"] is not None
    resumed = model.optimize_ce_assignment(
        z, q, assignment, steps=6, folder=tmp_path, resume_state=state, **args
    )
    torch.testing.assert_close(
        full["checkpoints"][6]["moments"], resumed["checkpoints"][6]["moments"], atol=2e-7, rtol=2e-6
    )
    torch.testing.assert_close(
        full["checkpoints"][6]["theta"], resumed["checkpoints"][6]["theta"], atol=2e-7, rtol=2e-6
    )
    assert [row["step"] for row in resumed["history"]] == list(range(7))


def test_unconverged_tracking_values_do_not_replace_exact_best(monkeypatch):
    z, q, assignment = problem()
    track, outer = model.track_inner, model.outer_value_gradient
    active = {"approximate": False}

    def imperfect(*args, **kwargs):
        fitted = track(*args, **kwargs)
        fitted.update(inner_converged=False, tracking_failed=False)
        active["approximate"] = True
        return fitted

    def fake_value(*args, **kwargs):
        value, gradient = outer(*args, **kwargs)
        if active.pop("approximate", False):
            value = -100.0
        return value, gradient

    monkeypatch.setattr(model, "track_inner", imperfect)
    monkeypatch.setattr(model, "outer_value_gradient", fake_value)
    result = model.optimize_ce_assignment(z, q, assignment, steps=2, **options())
    assert result["history"][1]["J"] == -100.0
    assert not result["history"][1]["J_exact"]
    assert result["best_J"] >= 0 and result["best_step"] != 1


def test_tracking_line_search_failure_falls_back_to_exact(monkeypatch):
    z, q, assignment = problem()

    def failed(centers, labels, mass, penalty, initial, *args):
        return dict(
            theta=initial,
            inner_grad_max=1.0,
            inner_converged=False,
            inner_iterations=0,
            inner_polish_steps=0,
            tracking_failed=True,
        )

    monkeypatch.setattr(model, "track_inner", failed)
    result = model.optimize_ce_assignment(z, q, assignment, steps=2, **options())
    row = result["history"][1]
    assert row["head_fallback"] and row["exact_refresh"] and row["inner_converged"]
    assert row["cg_converged"]


def test_refresh_failures_still_stop_optimization(monkeypatch):
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
        model.optimize_ce_assignment(z, q, assignment, steps=2, **options())


def test_refresh_every_step_matches_original_exact_solver():
    z, q, assignment = problem()
    args = options()
    args["tracking_refresh"] = 1
    exact = model.optimize_ce_assignment(z, q, assignment, steps=3, **{**args, "solver_mode": "exact"})
    tracked = model.optimize_ce_assignment(z, q, assignment, steps=3, **args)
    torch.testing.assert_close(exact["best_moments"], tracked["best_moments"], atol=0, rtol=0)
    assert exact["best_J"] == tracked["best_J"]


def test_nonfinite_tracked_adjoint_is_repaired(monkeypatch):
    z, q, assignment = problem()
    pcg = model.conjugate_gradient

    def broken_warm_start(*args, **kwargs):
        if kwargs.get("initial") is not None:
            return torch.full_like(args[1], float("nan")), dict(
                cg_iterations=1,
                cg_residual=float("nan"),
                cg_relative_residual=float("nan"),
                cg_converged=False,
            )
        return pcg(*args, **kwargs)

    monkeypatch.setattr(model, "conjugate_gradient", broken_warm_start)
    result = model.optimize_ce_assignment(z, q, assignment, steps=2, **options())
    assert result["history"][1]["implicit_fallback"]
    assert result["history"][1]["cg_converged"]
