import torch

from src.soft_ce_partition import head_gradient
from src.soft_ridge_partition import augmented, make_material
from src.stationary_curvature import measure_curvature, path_curvature


def test_integrated_hessian_recovers_gradient_difference():
    torch.manual_seed(14)
    x = augmented(torch.randn(9, 3, dtype=torch.double))
    q = torch.randn(9, 3, dtype=torch.double).softmax(1)
    mass = torch.full((9,), 1 / 9, dtype=torch.double)
    start = torch.randn(3, 4, dtype=torch.double) * .2
    end = start + torch.randn_like(start) * .3
    expected = head_gradient(x, q, mass, end, .1) - head_gradient(x, q, mass, start, .1)
    actual = path_curvature(x, q, mass, end, start, .1, 24)
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-9)


def test_audit_bounds_and_positive_curvature():
    torch.manual_seed(15)
    z = torch.randn(8, 2, dtype=torch.double)
    q = torch.randn(8, 3, dtype=torch.double).softmax(1)
    moments = make_material(z, q) / len(z)
    reference = torch.randn(3, 3, dtype=torch.double) * .1
    row = measure_curvature(moments, reference, .3, head_tol=1e-10)
    assert row['head_distance'] <= row['distance_bound'] + 1e-8
    assert row['path_direction_curvature'] >= .3 - 1e-8
    assert row['objective_gap'] >= -1e-10
    assert row['path_identity_error'] < 1e-7
    assert row['cg_converged']
