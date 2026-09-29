import torch

from src.hessian_benchmark import PairedHessianSolver
from src.soft_ce_partition import solve_head_system


def problem():
    g = torch.Generator().manual_seed(19)
    x = torch.randn(12, 4, generator=g, dtype=torch.double)
    labels = torch.randn(12, 3, generator=g, dtype=torch.double).softmax(1)
    mass = torch.ones(12, dtype=torch.double) / 12
    theta = torch.randn(3, 4, generator=g, dtype=torch.double)
    rhs = torch.randn(3, 4, generator=g, dtype=torch.double)
    return x, labels, mass, theta, .2, rhs


def test_warm_solver_preserves_accuracy_and_input():
    args = problem()
    cold, dc = solve_head_system(*args, rtol=1e-10)
    previous = cold.clone()
    warm, dw = solve_head_system(*args, rtol=1e-10, initial=previous)
    assert dc['cg_converged'] and dw['cg_converged']
    assert dw['cg_iterations'] == 0
    torch.testing.assert_close(previous, cold, atol=0, rtol=0)
    torch.testing.assert_close(warm, cold)


def test_paired_solver_returns_cold_solution():
    args = problem()
    expected, _ = solve_head_system(*args, rtol=1e-10)
    paired = PairedHessianSolver(repeats=2)
    actual, first = paired(*args, rtol=1e-10)
    torch.testing.assert_close(actual, expected)
    assert not first['benchmark_has_previous']
    actual, second = paired(*args, rtol=1e-10)
    torch.testing.assert_close(actual, expected)
    assert second['benchmark_has_previous']
    assert second['benchmark_warm_cg_iterations'] == 0
    assert first['benchmark_first_mode'] != second['benchmark_first_mode']
    assert second['benchmark_warm_cg_converged']


def test_new_runs_warm_start_and_legacy_resume_stays_cold(tmp_path):
    from src.soft_ce_partition import optimize_ce_assignment
    g = torch.Generator().manual_seed(81)
    z = torch.randn(9, 3, generator=g, dtype=torch.double)
    q = torch.randn(9, 2, generator=g, dtype=torch.double).softmax(1)
    assignment = torch.arange(9) % 3
    settings = dict(penalty=.3, assignment_rank=2, save_assignment=False, save_resume=True)
    result = optimize_ce_assignment(z, q, assignment, steps=2, checkpoint_steps=[0, 1, 2],
                                    folder=tmp_path, **settings)
    assert result['implicit_warm_start']
    assert not result['history'][0]['implicit_warm_start']
    assert result['history'][1]['implicit_warm_start']
    state = torch.load(tmp_path / 'resume.pt', weights_only=False)
    resumed = optimize_ce_assignment(z, q, assignment, steps=3, resume_state=state, **settings)
    assert resumed['history'][2]['implicit_warm_start']
    legacy_dir = tmp_path / 'legacy'
    optimize_ce_assignment(z, q, assignment, steps=0, folder=legacy_dir,
                           implicit_warm_start=False, **settings)
    legacy = torch.load(legacy_dir / 'resume.pt', weights_only=False)
    legacy['config'].pop('implicit_warm_start')
    continued = optimize_ce_assignment(z, q, assignment, steps=1, resume_state=legacy, **settings)
    assert not continued['implicit_warm_start']
    assert not continued['history'][0]['implicit_warm_start']
