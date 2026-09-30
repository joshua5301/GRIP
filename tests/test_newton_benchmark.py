import torch

from src.head import head_objective
from src.moments import augmented
from src.soft_ce_partition import (
    head_gradient,
    optimize_ce_assignment,
    solve_inner,
    solve_inner_newton_first,
)
from src.solver_benchmark import PairedInnerSolver


def problem():
    g = torch.Generator().manual_seed(31)
    x = torch.randn(10, 3, generator=g, dtype=torch.double)
    labels = torch.randn(10, 3, generator=g, dtype=torch.double).softmax(1)
    mass = torch.ones(10, dtype=torch.double) / 10
    return x, labels, mass


def test_newton_first_matches_reference():
    x, labels, mass = problem()
    prior = solve_inner(x, labels, mass, 0.2, grad_tol=1e-9)["theta"]
    changed = x + 0.025 * torch.sin(x)
    initial = prior.clone()
    reference = solve_inner(changed, labels, mass, 0.2, prior, grad_tol=1e-9)
    actual = solve_inner_newton_first(changed, labels, mass, 0.2, prior, grad_tol=1e-9)
    assert reference["inner_converged"] and actual["inner_converged"]
    torch.testing.assert_close(prior, initial, atol=0, rtol=0)
    torch.testing.assert_close(actual["theta"], reference["theta"], atol=1e-7, rtol=1e-6)
    gradient = head_gradient(augmented(changed), labels, mass, actual["theta"], 0.2)
    assert gradient.abs().max() <= 1e-9
    assert (
        head_objective(augmented(changed), labels, mass, actual["theta"], 0.2)
        <= head_objective(augmented(changed), labels, mass, initial, 0.2) + 1e-12
    )


def test_missing_initial_and_zero_budget_fallback():
    x, labels, mass = problem()
    fitted = solve_inner_newton_first(x, labels, mass, 0.2, grad_tol=1e-9)
    assert fitted["inner_lbfgs_fallback"] and fitted["inner_converged"]
    initial = torch.zeros_like(fitted["theta"])
    fallback = solve_inner_newton_first(x, labels, mass, 0.2, initial, grad_tol=1e-9, newton_steps=0)
    assert fallback["inner_lbfgs_fallback"] and fallback["inner_converged"]
    assert fallback["inner_newton_steps"] == 0


def test_paired_solver_returns_reference():
    x, labels, mass = problem()
    initial = solve_inner(x, labels, mass, 0.2)["theta"]
    changed = x * 1.02
    reference = solve_inner(changed, labels, mass, 0.2, initial)
    paired = PairedInnerSolver(repeats=1)
    actual = paired(changed, labels, mass, 0.2, initial)
    torch.testing.assert_close(actual["theta"], reference["theta"])
    assert actual["benchmark_lbfgs_inner_converged"]
    assert actual["benchmark_newton_inner_converged"]
    assert "benchmark_newton_seconds" in actual


def test_newton_first_optimizer_resume(tmp_path):
    x, labels, _ = problem()
    assignment = torch.arange(len(x)) % 3
    options = dict(
        penalty=0.2, assignment_rank=2, inner_method="newton_first", save_resume=True, save_assignment=False
    )
    result = optimize_ce_assignment(x, labels, assignment, steps=1, folder=tmp_path, **options)
    assert all(r["inner_converged"] for r in result["history"])
    state = torch.load(tmp_path / "resume.pt", weights_only=False)
    assert state["config"]["inner_method"] == "newton_first"
    resumed = optimize_ce_assignment(x, labels, assignment, steps=2, resume_state=state, **options)
    assert all(r["inner_converged"] for r in resumed["history"])


def test_independent_solver_trajectories_start_identically():
    x, labels, _ = problem()
    assignment = torch.arange(len(x)) % 3
    results = [
        optimize_ce_assignment(
            x,
            labels,
            assignment,
            steps=3,
            penalty=0.2,
            assignment_rank=2,
            factor_seed=7,
            inner_tol=1e-9,
            inner_method=method,
            checkpoint_steps=(0, 3),
            save_assignment=False,
        )
        for method in ("lbfgs", "newton_first")
    ]
    for result in results:
        assert all(row["inner_converged"] for row in result["history"])
    torch.testing.assert_close(results[0]["initial_moments"], results[1]["initial_moments"], atol=0, rtol=0)
