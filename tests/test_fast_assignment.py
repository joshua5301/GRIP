import pytest
import torch

from src.low_rank_assignment import CachedLowRankMoments, LowRankMoments
from src.soft_ce_partition import conjugate_gradient, optimize_ce_assignment


@pytest.mark.parametrize("interval", [1, 8])
@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_cg_actual_residual(interval, device):
    g = torch.Generator().manual_seed(14)
    a = torch.randn(25, 25, generator=g, dtype=torch.double)
    a = a.T @ a + 0.1 * torch.eye(25, dtype=torch.double)
    rhs = torch.randn(25, generator=g, dtype=torch.double)
    a, rhs = a.to(device), rhs.to(device)
    actual, info = conjugate_gradient(
        lambda x: a @ x, rhs, a.diag(), rtol=1e-9, max_iter=512, check_interval=interval
    )
    assert info["cg_converged"]
    assert (a @ actual - rhs).norm() <= 1e-9 * rhs.norm()
    torch.testing.assert_close(actual, torch.linalg.solve(a, rhs), atol=1e-7, rtol=1e-7)


def test_cg_zero_rhs_and_breakdown():
    rhs = torch.ones(4, dtype=torch.double)
    zero, info = conjugate_gradient(lambda x: x, rhs * 0, rhs, check_interval=8)
    assert info["cg_converged"] and zero.norm() == 0
    solution, info = conjugate_gradient(lambda x: -x, rhs, rhs, check_interval=8)
    assert not info["cg_converged"]
    assert torch.isfinite(solution).all()
    solution, info = conjugate_gradient(lambda x: x, rhs, rhs, check_interval=8)
    assert info["cg_converged"]
    torch.testing.assert_close(solution, rhs)


@pytest.mark.parametrize("chunk", [3, 32])
@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_cached_moments_values_and_gradients(chunk, device):
    g = torch.Generator().manual_seed(12)
    u = torch.randn(17, 2, generator=g, requires_grad=True)
    v = torch.randn(4, 2, generator=g, requires_grad=True)
    material = torch.randn(17, 7, generator=g, dtype=torch.double)
    gradient = torch.randn(4, 7, generator=g, dtype=torch.double)
    assignment = torch.arange(17) % 4
    u, v, material, gradient, assignment = [t.to(device) for t in (u, v, material, gradient, assignment)]
    outputs, derivatives = [], []
    for operation in (LowRankMoments, CachedLowRankMoments):
        value = operation.apply(u, v, assignment, material, 0.05, chunk)
        outputs.append(value)
        derivatives.append(torch.autograd.grad(value, (u, v), gradient))
    torch.testing.assert_close(*outputs, atol=0, rtol=0)
    for left, right in zip(*derivatives):
        torch.testing.assert_close(left, right, atol=0, rtol=0)


def test_fast_optimizer_resume(tmp_path):
    g = torch.Generator().manual_seed(19)
    z = torch.randn(15, 3, generator=g, dtype=torch.double)
    q = torch.randn(15, 3, generator=g, dtype=torch.double).softmax(1)
    assignment = torch.arange(15) % 4
    options = dict(
        penalty=0.1,
        assignment_rank=2,
        inner_method="newton_first",
        cg_check_interval=8,
        cache_assignment=True,
        save_resume=True,
        save_assignment=False,
    )
    optimize_ce_assignment(z, q, assignment, steps=2, folder=tmp_path, **options)
    state = torch.load(tmp_path / "resume.pt", weights_only=False)
    result = optimize_ce_assignment(z, q, assignment, steps=3, resume_state=state, **options)
    assert all(row["inner_converged"] for row in result["history"])
    assert all(row["cg_converged"] for row in result["history"] if row["step"] < 3)
