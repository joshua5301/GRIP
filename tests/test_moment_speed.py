import pytest
import torch

from src.moment_speed import BACKENDS, AssignmentStatistics, _trajectory, gradient_snapshot


def example():
    generator = torch.Generator().manual_seed(41)
    x = torch.randn(19, 5, generator=generator, dtype=torch.double)
    q = torch.randn(19, 3, generator=generator, dtype=torch.double).softmax(1)
    u = torch.randn(19, 4, generator=generator, dtype=torch.double)
    v = torch.randn(6, 4, generator=generator, dtype=torch.double)
    material = torch.cat((torch.ones(19, 1, dtype=torch.double), x, q), 1)
    return (u, v), (x, q, material, x.square().sum(1).mean(), x.T @ q / len(x))


@pytest.mark.parametrize("cached", [False, True])
def test_direct_statistics_gradcheck(cached):
    factors, data = example()
    u, v = [t.requires_grad_() for t in factors]
    assert torch.autograd.gradcheck(
        lambda a, b: AssignmentStatistics.apply(a, b, data[2], 7, cached),
        (u, v),
        fast_mode=True,
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_gradient_and_adam_trajectory_match_baseline(backend):
    factors, data = example()
    expected = gradient_snapshot(factors, data, 0.8, "baseline", 7)
    actual = gradient_snapshot(factors, data, 0.8, backend, 7)
    assert actual["J"] == pytest.approx(expected["J"], abs=1e-12)
    for name in ("du", "dv"):
        torch.testing.assert_close(actual[name], expected[name], atol=1e-12, rtol=1e-10)
    baseline = _trajectory(factors, data, 0.8, "baseline", 7, 0.01, 5)
    result = _trajectory(factors, data, 0.8, backend, 7, 0.01, 5)
    assert result[0]["J_final"] == pytest.approx(baseline[0]["J_final"], abs=1e-10)
    for actual, expected in zip(result[2], baseline[2]):
        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-9)
