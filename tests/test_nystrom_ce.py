import numpy as np
import torch

from src.moments import augmented, decode_moments, initial_logits, make_material
from src.nystrom_ce import NystromMap, moment_gradient, optimize, outer_gradient
from src.soft_ce_partition import solve_head_system, solve_inner_newton_first


def problem():
    torch.manual_seed(7)
    h = torch.randn(24, 3, dtype=torch.double) + 0.2
    q = torch.randn(24, 3, dtype=torch.double).softmax(1)
    assignment = torch.arange(24) % 4
    feature_map = NystromMap.fit(h, basis=8)
    return h, q, assignment, feature_map


def test_nonlinear_implicit_gradient_matches_refitted_finite_difference():
    h, q, assignment, feature_map = problem()
    moments = initial_logits(assignment, 4).double().softmax(1).T @ make_material(h, q) / len(h)
    phi = feature_map(h).detach().numpy()

    def objective(value):
        centers, labels, mass = decode_moments(value, h.shape[1])
        mapped = feature_map(centers).detach()
        fitted = solve_inner_newton_first(mapped, labels, mass, 0.1, grad_tol=1e-10)
        assert fitted["inner_converged"]
        loss, rhs = outer_gradient(phi, q, fitted["theta"], chunk=7)
        return loss, rhs, fitted["theta"], mapped, labels, mass

    _, rhs, theta, mapped, labels, mass = objective(moments)
    vector, info = solve_head_system(augmented(mapped), labels, mass, theta, 0.1, rhs, rtol=1e-10)
    assert info["cg_converged"]
    derivative = moment_gradient(moments, 3, feature_map, theta, vector, 0.1)
    direction = torch.randn_like(moments) * 0.01
    epsilon = 1e-3
    numerical = (objective(moments + epsilon * direction)[0] -
                 objective(moments - epsilon * direction)[0]) / (2 * epsilon)
    np.testing.assert_allclose(float((derivative * direction).sum()), numerical, rtol=2e-4, atol=1e-7)


def test_checkpoint_resume_matches_uninterrupted_updates(tmp_path):
    h, q, assignment, feature_map = problem()
    phi = feature_map(h).detach().numpy()
    kwargs = dict(penalty=0.1, rank=3, chunk=7, checkpoint_every=1)
    optimize(h, q, assignment, feature_map, phi, tmp_path / "full", 2, **kwargs)
    optimize(h, q, assignment, feature_map, phi, tmp_path / "resumed", 1, **kwargs)
    optimize(h, q, assignment, feature_map, phi, tmp_path / "resumed", 2, **kwargs)
    a = torch.load(tmp_path / "full" / "step_000002.pt", weights_only=False)
    b = torch.load(tmp_path / "resumed" / "step_000002.pt", weights_only=False)
    torch.testing.assert_close(a["moments"], b["moments"], rtol=1e-8, atol=1e-10)
    torch.testing.assert_close(a["theta"], b["theta"], rtol=1e-5, atol=1e-7)
