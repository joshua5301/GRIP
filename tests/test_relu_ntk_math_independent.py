"""Independent tiny CPU checks for the frozen both-layer ReLU NTK surrogate.

The tests derive references from the stated finite network and head objective,
without invoking teacher/data fitting or any dataset-dependent cache.
"""

import math

import pytest
import torch

from src import nystrom_ce, relu_ntk
from src.low_rank_assignment import LowRankMoments, logit_block
from src.moments import augmented, decode_moments, make_material
from src.soft_ce_partition import solve_head_system, solve_inner_newton_first


@pytest.fixture(scope="module", autouse=True)
def tiny_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("theta", [0., math.pi / 6, math.pi / 3,
                                    math.pi / 2, 2 * math.pi / 3, math.pi])
def test_known_angle_both_layer_ntk_with_fixed_diagonal_normalization(theta):
    # Variances and angle references come from the explicit network, not the API.
    left = torch.tensor([[2., 0.]], dtype=torch.double)
    right = torch.tensor([[3 * math.cos(theta), 3 * math.sin(theta)]], dtype=torch.double)
    scale = 2.5
    expected = 6 / (2 * math.pi * scale) * (
        math.sin(theta) + 2 * (math.pi - theta) * math.cos(theta))
    torch.testing.assert_close(relu_ntk.kernel_values(left, right, scale),
                               torch.tensor([[expected]], dtype=torch.double),
                               rtol=1e-13, atol=2e-15)


def test_forward_symmetry_positive_semidefiniteness_and_diagonal():
    generator = torch.Generator().manual_seed(331)
    queries = torch.randn(13, 4, dtype=torch.double, generator=generator)
    queries = torch.cat((queries, -queries[:1], queries[:1], torch.zeros(1, 4)))
    scale = 1.75
    gram = relu_ntk.kernel_values(queries, queries, scale)
    torch.testing.assert_close(gram, gram.T, rtol=0, atol=3e-15)
    # Dot/norm roundoff can put an exact parallel pair infinitesimally below c=1.
    # This endpoint-sensitive formula has O(sqrt(machine epsilon)) forward error;
    # keep the exact axis-endpoint test below and do not manufacture smoothing.
    endpoint_roundoff = 8 * math.sqrt(torch.finfo(torch.double).eps)
    torch.testing.assert_close(gram.diagonal(), queries.square().sum(1) / scale,
                               rtol=endpoint_roundoff, atol=2e-15)
    assert float(torch.linalg.eigvalsh((gram + gram.T) / 2).min()) >= -1e-12
    assert bool(torch.isfinite(gram).all())
    torch.testing.assert_close(gram[-1], torch.zeros(len(queries), dtype=torch.double),
                               rtol=0, atol=0)


def test_exact_zero_and_parallel_antiparallel_forward_are_finite():
    queries = torch.tensor([[0., 0.], [2., 0.], [-2., 0.]], dtype=torch.double)
    anchors = torch.tensor([[0., 0.], [3., 0.], [-3., 0.]], dtype=torch.double)
    expected = torch.tensor([[0., 0., 0.], [0., 3., 0.], [0., 0., 3.]], dtype=torch.double)
    torch.testing.assert_close(relu_ntk.kernel_values(queries, anchors, 2.), expected, rtol=0, atol=0)


@pytest.mark.parametrize("query", [[0., 0.], [1e-14, 2e-14], [1., 0.], [-1., 0.],
                                   [1., 1e-7]])
def test_frozen_guard_rejects_zero_tiny_and_angular_cusp_gradients(query):
    variable = torch.tensor([query], dtype=torch.double, requires_grad=True)
    anchors = torch.tensor([[1., 0.], [0., 0.]], dtype=torch.double)
    with pytest.raises(FloatingPointError):
        relu_ntk.kernel_values(variable, anchors, 1.).sum().backward()


def test_zero_fixed_anchor_is_excluded_from_differentiable_query_guard():
    variable = torch.tensor([[.8, .6]], dtype=torch.double, requires_grad=True)
    anchors = torch.tensor([[1., 0.], [0., 0.]], dtype=torch.double)
    relu_ntk.validate_gradient_domain(variable.detach(), anchors, 1.)
    values = relu_ntk.kernel_values(variable, anchors, 1.)
    derivative = torch.autograd.grad(values.sum(), variable)[0]
    assert bool(torch.isfinite(derivative).all())
    torch.testing.assert_close(values[:, 1], torch.zeros(1, dtype=torch.double), rtol=0, atol=0)


def test_differentiable_anchor_cusp_is_rejected_too():
    anchors = torch.tensor([[1., 0.]], dtype=torch.double, requires_grad=True)
    with pytest.raises(FloatingPointError):
        relu_ntk.kernel_values(torch.tensor([[2., 0.]], dtype=torch.double), anchors, 1.)


def test_interior_input_derivative_matches_independently_derived_angular_formula():
    query = torch.tensor([[.8, .6, -.3]], dtype=torch.double, requires_grad=True)
    anchor = torch.tensor([[.1, -.7, .9]], dtype=torch.double)
    scale = 1.3
    derivative = torch.autograd.grad(relu_ntk.kernel_values(query, anchor, scale).sum(), query)[0]
    radius, aradius = query.detach().norm(), anchor.norm()
    cosine = float((query.detach() * anchor).sum() / (radius * aradius))
    theta = math.acos(cosine)
    angular = (math.sqrt(1 - cosine * cosine) + 2 * (math.pi - theta) * cosine) / (2 * math.pi)
    angular_derivative = (cosine / math.sqrt(1 - cosine * cosine)
                          + 2 * (math.pi - theta)) / (2 * math.pi)
    expected = aradius / scale * (
        angular * query.detach() / radius
        + angular_derivative * (anchor / aradius - cosine * query.detach() / radius))
    torch.testing.assert_close(derivative, expected, rtol=1e-12, atol=1e-13)


def test_standard_relu_finite_network_full_parameter_jacobian_matches_kernel():
    # This checks both trainable layers and 1/sqrt(width), 1/sqrt(scale) directly.
    generator = torch.Generator().manual_seed(7321)
    width, scale = 65536, 1.7
    weight = torch.randn(width, 3, dtype=torch.double, generator=generator, requires_grad=True)
    output_weight = torch.randn(width, dtype=torch.double, generator=generator, requires_grad=True)
    queries = torch.tensor([[1., .2, -.4], [.3, 1., .1], [-.4, .2, 1.],
                            [.8, -.5, .6]], dtype=torch.double)
    outputs = (torch.relu(queries @ weight.T / math.sqrt(scale))
               @ output_weight / math.sqrt(width))
    gradients = []
    for output in outputs:
        ga, gw = torch.autograd.grad(output, (output_weight, weight), retain_graph=True)
        gradients.append(torch.cat((ga.flatten(), gw.flatten())))
    full_jacobian = torch.stack(gradients)
    empirical = full_jacobian @ full_jacobian.T
    limit = relu_ntk.kernel_values(queries, queries, scale)
    # A fixed deterministic Monte Carlo approximation, not an exact identity.
    torch.testing.assert_close(empirical, limit, rtol=.035, atol=.008)


def tiny_ntk_problem():
    generator = torch.Generator().manual_seed(782)
    h = torch.randn(18, 3, dtype=torch.double, generator=generator) + .3
    q = torch.randn(18, 3, dtype=torch.double, generator=generator).softmax(1)
    assignment = torch.tensor([0] * 7 + [1] * 5 + [2] * 4 + [3] * 2)
    anchors = torch.randn(5, 3, dtype=torch.double, generator=generator) + .1
    scale = float(anchors.square().sum(1).mean())
    gram = relu_ntk.kernel_values(anchors, anchors, scale)
    chol = torch.linalg.cholesky((gram + gram.T) / 2 + 1e-8 * gram.diagonal().mean() * torch.eye(5))
    mapping = torch.linalg.solve_triangular(chol, torch.eye(5, dtype=torch.double), upper=False).T
    return h, q, assignment, relu_ntk.NTKNystromMap(anchors, mapping, scale, "tiny_CPU_fixture")


@pytest.mark.parametrize("changed_factor", ["u", "v", "both"])
def test_full_uniform_ce_implicit_head_derivative_through_p_derived_centroids(changed_factor):
    h, q, assignment, feature_map = tiny_ntk_problem()
    original_h, original_q = h.clone(), q.clone()
    generator = torch.Generator().manual_seed(719)
    u = (.2 * torch.randn(18, 3, dtype=torch.double, generator=generator)).requires_grad_()
    v = torch.randn(4, 3, dtype=torch.double, generator=generator).requires_grad_()
    du = .2 * torch.randn(u.shape, dtype=torch.double, generator=generator)
    dv = .2 * torch.randn(v.shape, dtype=torch.double, generator=generator)
    if changed_factor == "u":
        dv.zero_()
    elif changed_factor == "v":
        du.zero_()
    material = make_material(h, q)
    phi = feature_map(h).detach().numpy()
    moments = LowRankMoments.apply(u, v, assignment, material, .05, 5)
    penalty = .2

    def reference_moments(left, right):
        probability = logit_block(left, right, assignment, .05).softmax(1)
        torch.testing.assert_close(probability.sum(1), torch.ones(len(h), dtype=torch.double))
        return probability.T @ material / len(h)

    def refitted_outer(value):
        centers, labels, mass = decode_moments(value, 3)
        torch.testing.assert_close(labels.sum(1), torch.ones(4, dtype=torch.double))
        relu_ntk.validate_gradient_domain(centers, feature_map.anchors, feature_map.scale)
        mapped = feature_map(centers).detach()
        weights = torch.full_like(mass, .25)
        fitted = solve_inner_newton_first(mapped, labels, weights, penalty,
                                          initial=torch.zeros(3, 6, dtype=torch.double),
                                          grad_tol=1e-10, newton_steps=20)
        assert fitted["inner_converged"]
        outer, rhs = nystrom_ce.outer_gradient(phi, q, fitted["theta"], chunk=5)
        return outer, rhs, fitted["theta"], mapped, labels, weights

    torch.testing.assert_close(moments, reference_moments(u.detach(), v.detach()), rtol=1e-13, atol=1e-14)
    _, rhs, theta, mapped, labels, weights = refitted_outer(moments.detach())
    vector, diagnostics = solve_head_system(augmented(mapped), labels, weights, theta,
                                            penalty, rhs, rtol=1e-11)
    assert diagnostics["cg_converged"]
    cotangent = nystrom_ce.moment_gradient(moments, 3, feature_map, theta, vector,
                                          penalty, inner_loss_weighting="uniform")
    moments.backward(cotangent)
    analytic = float((u.grad * du).sum() + (v.grad * dv).sum())
    assert abs(analytic) > 1e-7
    for epsilon in [5e-4, 2e-4]:
        plus = reference_moments(u.detach() + epsilon * du, v.detach() + epsilon * dv)
        minus = reference_moments(u.detach() - epsilon * du, v.detach() - epsilon * dv)
        plus_x, plus_q, plus_m = decode_moments(plus, 3)
        minus_x, minus_q, minus_m = decode_moments(minus, 3)
        assert float((plus_m - minus_m).abs().max()) > 1e-9
        assert float((plus_x - minus_x).abs().max()) > 1e-9
        assert float((plus_q - minus_q).abs().max()) > 1e-9
        numerical = (refitted_outer(plus)[0] - refitted_outer(minus)[0]) / (2 * epsilon)
        assert abs(analytic - numerical) <= 1e-8 + 2e-4 * abs(numerical)
    probability = logit_block(u.detach(), v.detach(), assignment, .05).softmax(1)
    mean_phi = probability.T @ torch.from_numpy(phi) / probability.sum(0)[:, None]
    assert not torch.allclose(mapped, mean_phi, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(h, original_h, rtol=0, atol=0)
    torch.testing.assert_close(q, original_q, rtol=0, atol=0)
