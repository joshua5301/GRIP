"""Independent all-source centering Jacobian and full uniform-CE head checks.

FP64 nonlinear directional FD is separate from the native FP32 P0 fixture.
There are no datasets, teacher/student fits, CUDA or source-cache operations.
"""
import pytest
import torch

from src.head import head_objective
from src.low_rank_assignment import (
    LowRankLogits,
    LowRankMoments,
    encode_nodes,
    initialize_factors,
    initialize_mlp,
)
from src.mlp_source_centering import centered_nodes
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


def problem():
    generator = torch.Generator().manual_seed(314)
    z = torch.randn(24, 3, generator=generator, dtype=torch.double)
    q = torch.stack((.8 * z[:, 0] - .5 * z[:, 1], -.4 * z[:, 0] + .7 * z[:, 2]), 1).softmax(1)
    assignment = torch.tensor([0] * 12 + [1] * 7 + [2] * 4 + [3])
    first = torch.tensor([[.45, -.2, .4, .1, -.35], [.1, .5, -.25, .4, .2],
                          [-.3, .1, .2, -.45, .3]], dtype=torch.double)
    ordered = (z @ first).sort(0).values
    gap_index = (ordered[9:15] - ordered[8:14]).argmax(0) + 8
    columns = torch.arange(first.shape[1])
    bias = -(ordered[gap_index, columns] + ordered[gap_index + 1, columns]) / 2
    last = .2 * torch.randn(5, 3, generator=generator, dtype=torch.double)
    output_bias = torch.zeros(3, dtype=torch.double)
    parameters = [x.clone().requires_grad_() for x in (first, bias, last, output_bias)]
    v = torch.randn(4, 3, generator=generator, dtype=torch.double).requires_grad_()
    preactivation = z @ first + bias
    assert float(preactivation.abs().min()) > .004
    assert bool((preactivation > 0).any()) and bool((preactivation < 0).any())
    return z, q, assignment, parameters, v


def moments(z, q, assignment, parameters, v):
    u, mean, diagnostic = centered_nodes(z, parameters)
    logits = LowRankLogits.apply(u, v, assignment, .05, 7)
    value = LowRankMoments.apply(u, v, assignment, make_material(z, q), .05, 7)
    return value, logits, u, mean, diagnostic


def fit(value, z, q):
    centers, labels, mass = decode_moments(value.detach(), z.shape[1])
    weights = torch.full_like(mass, 1 / len(mass))
    result = solve_inner_newton_first(
        centers, labels, weights, .2, initial=z.new_zeros(q.shape[1], z.shape[1] + 1),
        max_iter=2000, grad_tol=1e-11, cg_max_iter=512, newton_steps=12)
    assert result["inner_grad_max"] <= 5e-9
    theta = result["theta"].clone()
    x = augmented(centers)
    gradient = head_gradient(x, labels, weights, theta, .2)
    polishes = 0
    if float(gradient.abs().max()) > 1e-11:
        correction, diagnostic = solve_head_system(
            x, labels, weights, theta, .2, gradient, rtol=1e-11, atol=1e-15, max_iter=512)
        assert diagnostic["cg_converged"]
        proposed = theta - correction
        assert float(head_objective(x, labels, weights, proposed, .2)) <= float(
            head_objective(x, labels, weights, theta, .2)) + 1e-14
        theta = proposed
        polishes = 1
    maximum = float(head_gradient(x, labels, weights, theta, .2).abs().max())
    assert maximum <= 1e-11
    result = dict(result, reference_grad_max=maximum, reference_polish_steps=polishes)
    loss, rhs = outer_value_gradient(z, q, theta, 7)
    return loss, rhs, theta, centers, labels, weights, result


@pytest.mark.parametrize("parameter", ["W1", "b1", "W2", "V", "combined"])
@pytest.mark.parametrize("epsilon", [1e-3, 3e-4])
def test_full_uniform_ce_head_centered_encoder_fd(parameter, epsilon, record_property):
    z, q, assignment, parameters, v = problem()
    value, _, u, _, _ = moments(z, q, assignment, parameters, v)
    u.retain_grad()
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
    assert torch.equal(parameters[3].grad, torch.zeros_like(parameters[3]))
    active = ["W1", "b1", "W2", "V"] if parameter == "combined" else [parameter]
    generator = torch.Generator().manual_seed(98)
    directions = [.2 * torch.randn(x.shape, generator=generator, dtype=torch.double) for x in variables]
    for name, direction in zip(names, directions):
        if name not in active:
            direction.zero_()
    analytic = sum(float((x.grad * d).sum()) for x, d in zip(variables, directions))
    objectives, reference_polishes = [], inner["reference_polish_steps"]
    original_sign = (z @ parameters[0].detach() + parameters[1].detach()) > 0
    for sign in (1, -1):
        changed = [x.detach() + sign * epsilon * d for x, d in zip(parameters, directions[:4])]
        changed_v = v.detach() + sign * epsilon * directions[4]
        assert torch.equal(original_sign, (z @ changed[0] + changed[1]) > 0)
        updated, _, updated_u, _, _ = moments(z, q, assignment, changed, changed_v)
        assert float(updated_u.mean(0).abs().max()) < 1e-14
        fitted = fit(updated, z, q)
        objectives.append(fitted[0])
        reference_polishes += fitted[-1]["reference_polish_steps"]
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
    torch.testing.assert_close(value.detach().sum(0), make_material(z, q).mean(0), atol=2e-13, rtol=2e-12)
    assert float(u.detach().mean(0).abs().max()) < 1e-14


def test_all_node_jacobian_matches_manual_centering_cotangent(record_property):
    z, _, _, parameters, _ = problem()
    u, mean, _ = centered_nodes(z, parameters)
    generator = torch.Generator().manual_seed(333)
    cotangent = torch.randn(u.shape, generator=generator, dtype=torch.double) + .7
    gradients = torch.autograd.grad((u * cotangent).sum(), parameters)
    first, bias, last, _ = parameters
    hidden = (z @ first + bias).relu()
    centered_g = cotangent - cotangent.mean(0, keepdim=True)
    preactivation_g = (centered_g @ last.T) * ((z @ first + bias) > 0)
    expected = [z.T @ preactivation_g, preactivation_g.sum(0),
                (hidden - hidden.mean(0, keepdim=True)).T @ cotangent,
                torch.zeros_like(parameters[3])]
    maximum = max(float((a - b).detach().abs().max()) for a, b in zip(gradients, expected))
    for actual, manual in zip(gradients, expected):
        torch.testing.assert_close(actual, manual, atol=3e-14, rtol=3e-13)
    torch.testing.assert_close(mean, (hidden @ last).mean(0, keepdim=True), atol=0, rtol=0)
    record_property("manual_C_N_jacobian_max_abs", maximum)


def test_detached_stale_and_per_chunk_means_are_counterexamples():
    z, _, _, parameters, _ = problem()
    first, bias, last, output_bias = parameters
    hidden = (z @ first + bias).relu()
    raw = hidden @ last
    u, mean, _ = centered_nodes(z, parameters)
    generator = torch.Generator().manual_seed(47)
    cotangent = torch.randn(u.shape, generator=generator, dtype=torch.double) + 1
    correct = torch.autograd.grad((u * cotangent).sum(), last, retain_graph=True)[0]
    detached = torch.autograd.grad(((raw - raw.mean(0, keepdim=True).detach()) * cotangent).sum(), last)[0]
    assert float((correct - detached).norm()) > .1
    # A source-mean cached before parameter movement cannot center the new raw U.
    changed_last = last.detach() + .02
    changed_raw = hidden.detach() @ changed_last
    stale_u = changed_raw - mean.detach()
    assert float(stale_u.mean(0).abs().max()) > .01
    # Centering each chunk separately projects out additional local directions.
    chunks = []
    for block in z.split(7):
        block_u, _, _ = centered_nodes(block, [first.detach(), bias.detach(), last.detach(), output_bias.detach()])
        chunks.append(block_u)
    chunk_centered = torch.cat(chunks)
    assert float((chunk_centered - u.detach()).norm()) > .1
    assert float(chunk_centered.mean(0).abs().max()) < 1e-14


def test_output_bias_is_exact_null_and_global_offset_removed():
    z, q, assignment, parameters, v = problem()
    value, logits, u, _, _ = moments(z, q, assignment, parameters, v)
    generator = torch.Generator().manual_seed(29)
    cotangent = torch.randn(value.shape, generator=generator, dtype=torch.double)
    (gradient,) = torch.autograd.grad((value * cotangent).sum(), parameters[3])
    assert torch.equal(gradient, torch.zeros_like(gradient))
    moved = [p.detach().clone() for p in parameters]
    moved[3].add_(torch.tensor([.7, -.3, 2.], dtype=torch.double))
    other, other_logits, other_u, _, _ = moments(z, q, assignment, moved, v.detach())
    assert torch.equal(other_u, u.detach()) and torch.equal(other_logits, logits.detach())
    assert torch.equal(other, value.detach())


@pytest.mark.parametrize("cells", [3, 70])
def test_zero_mean_residual_logits_do_not_freeze_column_mass(cells, record_property):
    assignment = torch.arange(cells)
    u = torch.zeros(cells, dtype=torch.double)
    u[:3] = torch.tensor([1., 1., -2.], dtype=torch.double)
    v = torch.zeros(cells, dtype=torch.double)
    v[0], v[2] = 1., -1.
    residual = u[:, None] * v[None, :]
    assert torch.equal(residual.mean(0), torch.zeros_like(v))
    prior = initial_logits(assignment, cells, .05, torch.double)
    scalar = torch.tensor(0., dtype=torch.double, requires_grad=True)
    columns = (prior + scalar * residual).softmax(1).mean(0)
    derivative = torch.stack([torch.autograd.grad(columns[j], scalar, retain_graph=True)[0] for j in range(cells)])
    p0 = prior.softmax(1)
    analytic = (p0 * (residual - (p0 * residual).sum(1, keepdim=True))).mean(0)
    torch.testing.assert_close(derivative, analytic, atol=1e-16, rtol=1e-12)
    assert float(derivative.abs().max()) > 1e-5
    assert abs(float(derivative.sum())) < 1e-16
    record_property("free_mass_derivative_max_abs", float(derivative.abs().max()))
    record_property("hard_prior_mixing", .05)


def test_first_zero_output_update_is_covariance_not_free_cotangent(record_property):
    z, _, assignment, parameters, v = problem()
    parameters[2].data.zero_()
    parameters[3].data.zero_()
    inputs = z
    u, _, _ = centered_nodes(inputs, parameters)
    u.retain_grad()
    generator = torch.Generator().manual_seed(86)
    coefficient = torch.randn(len(z), len(v), generator=generator, dtype=torch.double)
    logits = LowRankLogits.apply(u, v, assignment, .05, 7)
    (logits.softmax(1) * coefficient).sum().backward()
    hidden = (inputs @ parameters[0].detach() + parameters[1].detach()).relu()
    expected = (hidden - hidden.mean(0, keepdim=True)).T @ u.grad
    torch.testing.assert_close(parameters[2].grad, expected, atol=2e-14, rtol=2e-12)
    free_w2 = hidden.T @ u.grad
    difference = float((free_w2 - expected).norm())
    assert difference > 1e-6 and float(parameters[2].grad.norm()) > 1e-6
    assert float(u.grad.sum(0).norm()) > 1e-6
    for variable in (parameters[0], parameters[1], parameters[3], v):
        assert torch.equal(variable.grad, torch.zeros_like(variable))
    record_property("first_W2_free_vs_centered_gradient_difference", difference)


def test_native_float32_original_zero_output_P0_and_student_inputs_preserved(record_property):
    z, q, assignment, _, _ = problem()
    inputs = z.float()
    parameters, v = initialize_mlp(inputs, 4, 3, hidden=5, seed=0)
    originals = [p.detach().clone() for p in [*parameters, v]]
    free_u = encode_nodes(inputs, parameters)
    u, mean, _ = centered_nodes(inputs, parameters)
    node_u, node_v = initialize_factors(assignment, 4, 3, 0)
    assert u.dtype == torch.float32 and mean.dtype == torch.float32
    assert torch.equal(u, free_u) and torch.equal(u, node_u) and torch.equal(v, node_v)
    for before, after in zip(originals, [*parameters, v]):
        assert torch.equal(before, after)
    prior = initial_logits(assignment, 4, .05, torch.float32)
    assert torch.equal(LowRankLogits.apply(u, v, assignment, .05, 7), prior)
    material = make_material(z, q)
    centered_value = LowRankMoments.apply(u, v, assignment, material, .05, 7)
    free_value = LowRankMoments.apply(free_u, v, assignment, material, .05, 7)
    node_value = LowRankMoments.apply(node_u, node_v, assignment, material, .05, 7)
    assert torch.equal(centered_value, free_value) and torch.equal(free_value, node_value)
    for value in (free_value, node_value):
        centers, labels, mass = decode_moments(value, z.shape[1])
        cx, cy, cmass = decode_moments(centered_value, z.shape[1])
        assert torch.equal(cx.float(), centers.float()) and torch.equal(cy.float(), labels.float())
        assert torch.equal(torch.full_like(cmass, .25), torch.full_like(mass, .25))
    record_property("P0_moment_max_abs", 0.)
    record_property("P0_probability_max_abs", 0.)


def test_native_float32_center_residual_uses_dtype_error_bound(record_property):
    z, _, _, parameters, _ = problem()
    inputs = z.float()
    parameters = [p.detach().float() for p in parameters]
    u, mean, _ = centered_nodes(inputs, parameters)
    raw = (inputs @ parameters[0] + parameters[1]).relu() @ parameters[2]
    assert torch.equal(mean, raw.mean(0, keepdim=True))
    assert torch.equal(u, raw - mean + 0 * parameters[3])
    residual = float(u.double().mean(0).abs().max())
    n_eps = (len(inputs) + 2) * torch.finfo(inputs.dtype).eps
    bound = 4 * n_eps / (1 - n_eps) * max(1., float(raw.abs().max()))
    assert residual <= bound
    record_property("native_FP32_mean_residual", residual)
    record_property("native_FP32_IEEE_bound", bound)
