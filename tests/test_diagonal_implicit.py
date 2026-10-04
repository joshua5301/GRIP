import torch

from src.soft_ce_partition import hessian_operator, optimize_ce_assignment, solve_head_diagonal


def test_diagonal_solve_reports_actual_full_hessian_residual():
    x = torch.tensor([[1., 2., 1.], [2., 1., 1.]], dtype=torch.float64)
    labels = torch.tensor([[0.8, 0.2], [0.3, 0.7]], dtype=torch.float64)
    mass = torch.tensor([0.5, 0.5], dtype=torch.float64)
    theta = torch.zeros(2, 3, dtype=torch.float64)
    rhs = torch.tensor([[1., 2., 3.], [-1., -2., -3.]], dtype=torch.float64)
    multiply, diagonal = hessian_operator(x, labels, mass, theta, 0.1)
    solution, diagnostic = solve_head_diagonal(x, labels, mass, theta, 0.1, rhs)
    torch.testing.assert_close(solution, rhs / diagonal)
    expected = float((multiply(solution) - rhs).norm() / rhs.norm())
    assert abs(diagnostic["cg_relative_residual"] - expected) < 1e-12
    assert not diagnostic["cg_converged"]
    assert diagnostic["implicit_approximate"]


def test_explicit_diagonal_approximation_can_update_without_cg_convergence():
    generator = torch.Generator().manual_seed(5)
    z = torch.randn(12, 3, generator=generator, dtype=torch.float64)
    q = torch.randn(12, 2, generator=generator, dtype=torch.float64).softmax(1)
    assignment = torch.arange(12) % 3
    result = optimize_ce_assignment(
        z, q, assignment, steps=2, penalty=0.1, save_assignment=False,
        inner_tol=1e-5, inner_method="newton_first", assignment_rank=2,
        implicit_solver=solve_head_diagonal, cg_rtol=1e-12,
        checkpoint_steps=[0, 2],
    )
    updates = [row for row in result["history"] if row["step"] < 2]
    assert all(row["hessian_solver"] == "diagonal" for row in updates)
    assert all(row["implicit_approximate"] for row in updates)
    assert 2 in result["checkpoints"]
