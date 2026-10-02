"""Tiny independent fixed-mass MLP full-CE checks, with no graph datasets.

Directional differences use FP64 encoder arithmetic; the separate P0 fixture
uses FP32 initialization and does not certify production CUDA/cache bytes.
"""
import math

import pytest
import torch

from src.fixed_mass_assignment import FixedMassMoments
from src.head import head_objective
from src.low_rank_assignment import LowRankLogits, LowRankMoments, encode_nodes, initialize_mlp
from src.moments import augmented, decode_moments, initial_logits, make_material
from src.soft_ce_partition import (
    head_gradient,
    implicit_moment_gradient,
    outer_value_gradient,
    solve_head_system,
    solve_inner_newton_first,
)


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def problem(kind="source_P0"):
    generator = torch.Generator().manual_seed(314)
    z = torch.randn(24, 3, generator=generator, dtype=torch.double)
    q = torch.stack((.8 * z[:, 0] - .5 * z[:, 1], -.4 * z[:, 0] + .7 * z[:, 2]), 1).softmax(1)
    assignment = torch.tensor([0] * 12 + [1] * 7 + [2] * 4 + [3])
    first = torch.tensor([[.45, -.2, .4, .1, -.35], [.1, .5, -.25, .4, .2],
                          [-.3, .1, .2, -.45, .3]], dtype=torch.double)
    # Pick a gap in the middle of each toy projection, so central differences
    # stay inside one ReLU region while each hidden unit sees both signs.
    ordered = (z @ first).sort(0).values
    gap_index = (ordered[9:15] - ordered[8:14]).argmax(0) + 8
    columns = torch.arange(first.shape[1])
    bias = -(ordered[gap_index, columns] + ordered[gap_index + 1, columns]) / 2
    last = .2 * torch.randn(5, 3, generator=generator, dtype=torch.double)
    output_bias = torch.tensor([.02, -.03, .01], dtype=torch.double)
    v = torch.randn(4, 3, generator=generator, dtype=torch.double).requires_grad_()
    parameters = [value.clone().requires_grad_() for value in (first, bias, last, output_bias)]
    if kind == "source_P0":
        target = initial_logits(assignment, 4, .05).double().softmax(1).mean(0).detach()
    else:
        target = torch.tensor([.001, .019, .28, .7], dtype=torch.double)
    assert not target.requires_grad and bool((target > 0).all())
    assert not torch.allclose(target, torch.full_like(target, .25))
    preactivation = z @ first + bias
    assert float(preactivation.abs().min()) > .004
    assert bool((preactivation > 0).any()) and bool((preactivation < 0).any())
    return z, q, assignment, parameters, v, target


def moments(z, q, assignment, parameters, v, target):
    logits = LowRankLogits.apply(encode_nodes(z, parameters), v, assignment, .05, 7)
    value, dual, diagnostic = FixedMassMoments.apply(
        logits, make_material(z, q), target, 7, 10000, 1e-12, None)
    return value, logits, dual, diagnostic


def fit(value, z, q, initial=None):
    centers, labels, mass = decode_moments(value.detach(), z.shape[1])
    weights = torch.full_like(mass, 1 / len(mass))
    if initial is None:
        initial = z.new_zeros(q.shape[1], z.shape[1] + 1)
    result = solve_inner_newton_first(
        centers, labels, weights, .2, initial=initial, max_iter=2000,
        grad_tol=1e-11, cg_max_iter=512, newton_steps=12)
    # At this stricter-than-production reference tolerance, an objective line
    # search can round to a tie with residual ~2.6e-11. One certified Newton
    # correction avoids relaxing the FD reference's stationarity requirement.
    assert result["inner_grad_max"] <= 5e-9
    theta = result["theta"].clone()
    x = augmented(centers)
    gradient = head_gradient(x, labels, weights, theta, .2)
    polish = 0
    if float(gradient.abs().max()) > 1e-11:
        correction, diagnostic = solve_head_system(
            x, labels, weights, theta, .2, gradient, rtol=1e-11, atol=1e-15, max_iter=512)
        assert diagnostic["cg_converged"]
        proposal = theta - correction
        assert float(head_objective(x, labels, weights, proposal, .2)) <= float(
            head_objective(x, labels, weights, theta, .2)) + 1e-14
        theta = proposal
        polish = 1
    maximum = float(head_gradient(x, labels, weights, theta, .2).abs().max())
    assert maximum <= 1e-11
    result = dict(result, reference_grad_max=maximum, reference_polish_steps=polish)
    loss, rhs = outer_value_gradient(z, q, theta, 7)
    return loss, rhs, theta, centers, labels, weights, result


@pytest.mark.parametrize("kind", ["source_P0", "tiny_nonuniform"])
@pytest.mark.parametrize("parameter", ["W1", "b1", "W2", "V", "combined"])
@pytest.mark.parametrize("epsilon", [1e-3, 3e-4])
def test_full_uniform_ce_implicit_encoder_head_fd(kind, parameter, epsilon, record_property):
    z, q, assignment, parameters, v, target = problem(kind)
    value, logits, dual, diagnostic = moments(z, q, assignment, parameters, v, target)
    logits.retain_grad()
    _, rhs, theta, centers, labels, weights, inner = fit(value, z, q)
    vector, adjoint = solve_head_system(
        augmented(centers), labels, weights, theta, .2, rhs, rtol=1e-11, atol=1e-14, max_iter=512)
    assert adjoint["cg_converged"]
    value.backward(implicit_moment_gradient(value, z.shape[1], theta, vector, .2, "uniform"))
    names = ["W1", "b1", "W2", "b2", "V"]
    variables = [*parameters, v]
    for name, variable in zip(names, variables):
        assert variable.grad is not None and bool(torch.isfinite(variable.grad).all())
        if name != "b2":
            assert float(variable.grad.norm()) > 1e-8
    assert float(parameters[3].grad.abs().max()) < 1e-12
    active = ["W1", "b1", "W2", "V"] if parameter == "combined" else [parameter]
    generator = torch.Generator().manual_seed(98)
    directions = [.2 * torch.randn(x.shape, generator=generator, dtype=torch.double) for x in variables]
    for name, direction in zip(names, directions):
        if name not in active:
            direction.zero_()
    analytic = sum(float((x.grad * d).sum()) for x, d in zip(variables, directions))
    objectives, probabilities, reference_polishes = [], [], inner["reference_polish_steps"]
    original_sign = (z @ parameters[0].detach() + parameters[1].detach()) > 0
    for sign in [1, -1]:
        changed = [x.detach() + sign * epsilon * d for x, d in zip(parameters, directions[:4])]
        changed_v = v.detach() + sign * epsilon * directions[4]
        assert torch.equal(original_sign, (z @ changed[0] + changed[1]) > 0)
        updated, updated_logits, updated_dual, _ = moments(z, q, assignment, changed, changed_v, target)
        # Use the same cold reference start on both sides rather than relying
        # on warm-start line searches near machine precision.
        updated_fit = fit(updated, z, q)
        objectives.append(updated_fit[0])
        reference_polishes += updated_fit[-1]["reference_polish_steps"]
        probabilities.append((updated_logits.detach() + updated_dual).softmax(1))
    finite = (objectives[0] - objectives[1]) / (2 * epsilon)
    error = abs(analytic - finite)
    record_property("full_head_fd_absolute_error", error)
    record_property("full_head_fd_relative_error", error / max(abs(analytic), abs(finite), 1e-15))
    record_property("inner_grad_max", inner["reference_grad_max"])
    record_property("original_inner_grad_max", inner["inner_grad_max"])
    record_property("reference_polish_steps", reference_polishes)
    record_property("adjoint_residual", adjoint["cg_residual"])
    record_property("tiny_synthetic_head_solves", 3)
    assert error < 2e-8 + 2e-4 * abs(finite)
    assert float((value.detach()[:, 0] / target - 1).abs().max()) <= 1e-12
    assert float(diagnostic[1:].max()) <= 1e-12
    assert abs(float(dual.mean())) < 1e-14
    assert float(logits.grad.sum(1).abs().max()) < 1e-12
    assert float(logits.grad.sum(0).abs().max()) < 1e-12
    tangent = (probabilities[0] - probabilities[1]) / (2 * epsilon)
    assert float(tangent.sum(1).abs().max()) < 2e-11
    assert float(tangent.mean(0).abs().max()) < 5e-10
    torch.testing.assert_close(value.detach().sum(0), make_material(z, q).mean(0), atol=2e-13, rtol=2e-12)


@pytest.mark.parametrize("kind", ["source_P0", "tiny_nonuniform"])
def test_output_bias_is_global_column_gauge(kind, record_property):
    z, q, assignment, parameters, v, target = problem(kind)
    value, logits, dual, _ = moments(z, q, assignment, parameters, v, target)
    generator = torch.Generator().manual_seed(21)
    cotangent = torch.randn(value.shape, generator=generator, dtype=torch.double)
    (gradient,) = torch.autograd.grad((value * cotangent).sum(), parameters[3])
    assert float(gradient.abs().max()) < 1e-12
    shifted = [p.detach().clone() for p in parameters]
    bias_shift = torch.tensor([.4, -.6, .2], dtype=torch.double)
    shifted[3].add_(bias_shift)
    other, other_logits, other_dual, _ = moments(z, q, assignment, shifted, v.detach(), target)
    torch.testing.assert_close(other, value.detach(), atol=2e-13, rtol=2e-11)
    torch.testing.assert_close((other_logits + other_dual).softmax(1),
                               (logits.detach() + dual).softmax(1), atol=3e-12, rtol=2e-11)
    expected = -(v.detach() @ bias_shift) / math.sqrt(v.shape[1])
    expected -= expected.mean()
    torch.testing.assert_close(other_dual - dual, expected, atol=3e-12, rtol=2e-11)
    record_property("gauge_bias_gradient_max", float(gradient.abs().max()))


def test_native_float32_original_zero_output_p0_preserved(record_property):
    z, q, assignment, _, _, _ = problem()
    inputs = z.float()
    parameters, v = initialize_mlp(inputs, 4, 3, hidden=5, seed=0)
    u = encode_nodes(inputs, parameters)
    assert torch.equal(u, torch.zeros_like(u))
    logits = LowRankLogits.apply(u, v, assignment, .05, 7)
    assert logits.dtype == torch.float32
    original_logits = initial_logits(assignment, 4, .05)
    assert torch.equal(logits, original_logits)
    target = original_logits.double().softmax(1).mean(0).detach()
    material = make_material(z, q)
    value, dual, diagnostic = FixedMassMoments.apply(logits, material, target, 7, 10000, 1e-12, None)
    reference = LowRankMoments.apply(u, v, assignment, material, .05, 7)
    assert torch.equal(dual, torch.zeros_like(dual))
    assert int(diagnostic[0]) == 1
    torch.testing.assert_close(value, reference, atol=3e-16, rtol=2e-14)
    original_x, original_q, original_mass = decode_moments(reference, z.shape[1])
    fixed_x, fixed_q, fixed_mass = decode_moments(value, z.shape[1])
    assert torch.equal(fixed_x.float(), original_x.float())
    assert torch.equal(fixed_q.float(), original_q.float())
    assert torch.equal(torch.full_like(fixed_mass.float(), .25), torch.full_like(original_mass.float(), .25))
    torch.testing.assert_close(target, fixed_mass, atol=1e-16, rtol=1e-14)
    free_fit = fit(reference, z, q)
    fixed_fit = fit(value, z, q, free_fit[2])
    torch.testing.assert_close(fixed_fit[2], free_fit[2], atol=1e-13, rtol=1e-12)
    assert abs(fixed_fit[0] - free_fit[0]) < 1e-14
    record_property("P0_moment_max_abs", float((value.detach() - reference.detach()).abs().max()))
    record_property("P0_head_max_abs", float((fixed_fit[2] - free_fit[2]).abs().max()))
    record_property("tiny_synthetic_head_solves", 2)
