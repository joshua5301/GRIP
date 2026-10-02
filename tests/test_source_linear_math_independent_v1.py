"""Independent conceptual source-linear math QA; no future helper API assumed.

The NumPy dense CE/Newton reference does not call production autograd or solvers.
Existing production moment/implicit VJPs are only the analytic comparison side.
The parent-frozen scientific preregistration and preimplementation tolerances
are bound by the companion independent review, not a future helper API.
"""
import math

import numpy as np
import torch

from src.low_rank_assignment import LowRankMoments, initialize_factors
from src.soft_ce_partition import implicit_moment_gradient

ATOL = 5e-8
RTOL = 3e-5
EPSILONS = (1e-3, 3e-4)
COUNTS = dict(reference_head_fits=0, dense_linear_adjoint_solves=0, newton_steps=0)


def softmax(logits):
    exponent = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exponent / exponent.sum(axis=1, keepdims=True)


def log_softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def base_logits(hard, cells, mixing=.05):
    value = np.full((len(hard), cells), math.log(mixing / cells))
    value[np.arange(len(hard)), hard] = math.log(1 - mixing + mixing / cells)
    return value


def problem(coordinate_dimension):
    generator = np.random.default_rng(846)
    z = generator.normal(size=(19, 3))
    q = softmax(z @ np.array([[.6, -.4, .1], [-.3, .2, .7], [.1, .5, -.5]]))
    hard = np.array([0] * 10 + [1] * 5 + [2] * 3 + [3], dtype=np.int64)
    # Coordinates are fixed and separate from original material/outer features.
    coordinates = generator.normal(size=(19, coordinate_dimension))
    a = .12 * generator.normal(size=(coordinate_dimension, 3))
    bias = .08 * generator.normal(size=(3,))
    v = .5 * generator.normal(size=(4, 3))
    return coordinates, z, q, hard, a, bias, v


def dense_moments(coordinates, z, q, hard, a, bias, v):
    u = coordinates @ a + bias
    probability = softmax(base_logits(hard, len(v)) + u @ v.T / math.sqrt(v.shape[1]))
    material = np.column_stack((np.ones(len(z)), z, q))
    moment = probability.T @ material / len(z)
    assert np.max(np.abs(probability.sum(1) - 1)) <= 1e-12
    assert np.max(np.abs(moment.sum(0) - material.mean(0))) <= 1e-12
    assert np.all(moment[:, 0] > 0)
    return moment, probability, u, material


def decode(moment, dimension):
    mass = moment[:, 0]
    x = moment[:, 1:1 + dimension] / mass[:, None]
    labels = moment[:, 1 + dimension:] / mass[:, None]
    return np.column_stack((x, np.ones(len(x)))), labels, mass


def head_equations(theta, features, labels, penalty):
    """Exact dense CE objective/gradient/Hessian, including labels.sum and bias."""
    probability = softmax(features @ theta.T)
    sums = labels.sum(1)
    error = sums[:, None] * probability - labels
    objective = -(labels * log_softmax(features @ theta.T)).sum() / len(features)
    objective += penalty * np.square(theta).sum() / 2
    gradient = error.T @ features / len(features) + penalty * theta
    covariance = np.einsum("kc,cd->kcd", probability, np.eye(theta.shape[0]))
    covariance -= np.einsum("kc,kd->kcd", probability, probability)
    hessian = np.einsum("k,kcd,ka,kb->cadb", sums / len(features), covariance, features, features)
    hessian = hessian.reshape(theta.size, theta.size) + penalty * np.eye(theta.size)
    return float(objective), gradient, hessian, probability


def reference_fit(moment, z, q, penalty=.3):
    """Fully re-solve a tiny independent head; never uses a production fitter."""
    COUNTS["reference_head_fits"] += 1
    features, labels, mass = decode(moment, z.shape[1])
    theta = np.zeros((q.shape[1], z.shape[1] + 1))
    for _ in range(40):
        objective, gradient, hessian, _ = head_equations(theta, features, labels, penalty)
        if np.max(np.abs(gradient)) <= 3e-13:
            break
        correction = np.linalg.solve(hessian, gradient.ravel()).reshape(theta.shape)
        length = 1.
        while length >= 2 ** -30:
            updated = theta - length * correction
            next_objective, _, _, _ = head_equations(updated, features, labels, penalty)
            # Preserve Newton's last accurate step despite scalar-loss last bits.
            if next_objective <= objective - 1e-4 * length * np.sum(gradient * correction) + 1e-15:
                break
            length /= 2
        assert length >= 2 ** -30, "Independent Newton line search failed"
        theta = updated
        COUNTS["newton_steps"] += 1
    _, gradient, hessian, probability = head_equations(theta, features, labels, penalty)
    maximum = float(np.max(np.abs(gradient)))
    assert maximum <= 1e-11, maximum
    outer_features = np.column_stack((z, np.ones(len(z))))
    outer_probability = softmax(outer_features @ theta.T)
    outer = float(-(q * log_softmax(outer_features @ theta.T)).sum() / len(z))
    rhs = (q.sum(1)[:, None] * outer_probability - q).T @ outer_features / len(z)
    return dict(theta=theta, features=features, labels=labels, mass=mass, hessian=hessian,
                probability=probability, outer=outer, rhs=rhs, gradient_max=maximum)


def independent_implicit(moment, z, q, fitted, coordinates, probability, u, material, v):
    """Manual mixed derivative -D_M<g_theta,v_adjoint>; no autograd."""
    COUNTS["dense_linear_adjoint_solves"] += 1
    theta, x, labels, p = (fitted[key] for key in ("theta", "features", "labels", "probability"))
    vector = np.linalg.solve(fitted["hessian"], fitted["rhs"].ravel()).reshape(theta.shape)
    residual = float(np.linalg.norm(fitted["hessian"] @ vector.ravel() - fitted["rhs"].ravel())
                     / max(np.linalg.norm(fitted["rhs"]), 1e-30))
    assert residual <= 1e-11, residual
    projected = x @ vector.T
    error = labels.sum(1)[:, None] * p - labels
    covariance_projected = p * (projected - (p * projected).sum(1, keepdims=True))
    # r_x and r_y are d<g_theta,vector>/d prototype and /d target.
    r_x = (labels.sum(1)[:, None] * covariance_projected @ theta[:, :-1]
           + error @ vector[:, :-1]) / len(x)
    r_y = ((p * projected).sum(1, keepdims=True) - projected) / len(x)
    mass = moment[:, 0]
    g_mass = (np.sum(r_x * x[:, :-1], axis=1) + np.sum(r_y * labels, axis=1)) / mass
    g_moment = np.column_stack((g_mass, -r_x / mass[:, None], -r_y / mass[:, None]))
    g_probability = material @ g_moment.T / len(z)
    g_logits = probability * (g_probability - (probability * g_probability).sum(1, keepdims=True))
    g_u = g_logits @ v / math.sqrt(v.shape[1])
    parameter_gradients = (coordinates.T @ g_u, g_u.sum(0), g_logits.T @ u / math.sqrt(v.shape[1]))
    assert np.max(np.abs(g_logits.sum(1))) <= 1e-12
    return g_moment, parameter_gradients, vector, residual


def production_vjp_comparison(coordinates, z, q, hard, a, bias, v, moment, fitted, expected, vector):
    # Analytic comparison only. The FD reference above is independent NumPy.
    f, original_z, original_q = (torch.from_numpy(value.copy()) for value in (coordinates, z, q))
    aa, bb, vv = [torch.from_numpy(value.copy()).requires_grad_() for value in (a, bias, v)]
    material = torch.cat((original_z.new_ones(len(z), 1), original_z, original_q), 1)
    observed = LowRankMoments.apply(f @ aa + bb, vv, torch.from_numpy(hard), material, .05, 6)
    assert torch.allclose(observed.detach(), torch.from_numpy(moment), atol=2e-12, rtol=0)
    cotangent = implicit_moment_gradient(observed, z.shape[1], torch.from_numpy(fitted["theta"]),
                                        torch.from_numpy(vector), .3, "uniform")
    g_moment, grads = expected
    moment_error = float(np.max(np.abs(cotangent.numpy() - g_moment)))
    assert moment_error <= 2e-12, moment_error
    observed.backward(cotangent)
    parameter_errors = [float(np.max(np.abs(actual.grad.numpy() - gradient)))
                        for actual, gradient in zip((aa, bb, vv), grads, strict=True)]
    assert max(parameter_errors) <= 2e-12, parameter_errors
    assert f.grad is None and not f.requires_grad
    # Proto material retains original D, regardless of coordinate dimension.
    assert observed.shape == (len(v), 1 + z.shape[1] + q.shape[1])
    return dict(moment_cotangent_maxerror=moment_error, parameter_vjp_maxerrors=parameter_errors)


def run_checks():
    fd_rows, vjp_rows, zero_rows, bias_rows = [], [], [], []
    head_max = adjoint_max = 0.
    for coordinate_dimension in (2, 7):
        coordinates, z, q, hard, a, bias, v = problem(coordinate_dimension)
        moment, probability, u, material = dense_moments(coordinates, z, q, hard, a, bias, v)
        fitted = reference_fit(moment, z, q)
        g_moment, grads, vector, residual = independent_implicit(
            moment, z, q, fitted, coordinates, probability, u, material, v)
        head_max, adjoint_max = max(head_max, fitted["gradient_max"]), max(adjoint_max, residual)
        comparison = production_vjp_comparison(coordinates, z, q, hard, a, bias, v, moment,
                                              fitted, (g_moment, grads), vector)
        vjp_rows.append(dict(coordinate_dimension=coordinate_dimension, **comparison))
        generator = np.random.default_rng(57)
        all_directions = [.2 * generator.normal(size=value.shape) for value in (a, bias, v)]
        for parameter in ("A", "b", "V", "combined"):
            directions = [d.copy() if parameter in (name, "combined") else np.zeros_like(d)
                          for name, d in zip(("A", "b", "V"), all_directions, strict=True)]
            analytic = sum(float(np.sum(gradient * direction)) for gradient, direction in zip(grads, directions, strict=True))
            assert abs(analytic) > 1e-8, (coordinate_dimension, parameter, analytic)
            for epsilon in EPSILONS:
                values = []
                for sign in (1, -1):
                    changed = [value + sign * epsilon * direction
                               for value, direction in zip((a, bias, v), directions, strict=True)]
                    changed_moment, _, _, _ = dense_moments(coordinates, z, q, hard, *changed)
                    perturbed = reference_fit(changed_moment, z, q)
                    head_max = max(head_max, perturbed["gradient_max"])
                    values.append(perturbed["outer"])
                finite = (values[0] - values[1]) / (2 * epsilon)
                absolute = abs(analytic - finite)
                relative = absolute / max(abs(analytic), abs(finite), 1e-30)
                assert absolute <= ATOL and relative <= RTOL, (analytic, finite, absolute, relative)
                fd_rows.append(dict(coordinate_dimension=coordinate_dimension, parameter=parameter, epsilon=epsilon,
                                    analytic=analytic, finite_difference=finite, absolute_error=absolute, relative_error=relative))
        # Genuine original U0: V receives no first gradient; A/b need not vanish.
        moment0, p0, u0, material0 = dense_moments(coordinates, z, q, hard, np.zeros_like(a), np.zeros_like(bias), v)
        fit0 = reference_fit(moment0, z, q)
        _, grads0, _, residual0 = independent_implicit(moment0, z, q, fit0, coordinates, p0, u0, material0, v)
        assert np.array_equal(grads0[2], np.zeros_like(v))
        assert np.linalg.norm(grads0[0]) > 1e-8 and np.linalg.norm(grads0[1]) > 1e-8
        zero_rows.append(dict(coordinate_dimension=coordinate_dimension, A_gradient_norm=float(np.linalg.norm(grads0[0])),
                              bias_gradient_norm=float(np.linalg.norm(grads0[1])), V_gradient_exact_zero=True))
        head_max, adjoint_max = max(head_max, fit0["gradient_max"]), max(adjoint_max, residual0)
        changed_moment, changed_p, _, _ = dense_moments(coordinates, z, q, hard, a, bias + np.array([.1, -.08, .06]), v)
        bias_mass_delta = float(np.max(np.abs(changed_moment[:, 0] - moment[:, 0])))
        assert bias_mass_delta > 1e-5 and np.max(np.abs(changed_p - probability)) > 1e-5
        bias_rows.append(dict(coordinate_dimension=coordinate_dimension, generic_bias_changes_mass_max=bias_mass_delta,
                              no_forced_centering_or_fixed_marginal=True))
    # Native FP32 initialization uses actual unchanged legacy V0; no new helper API.
    generator = torch.Generator().manual_seed(66)
    hard = torch.tensor([0] * 10 + [1] * 5 + [2] * 3 + [3])
    native_z = torch.randn(19, 3, generator=generator, dtype=torch.float64)
    native_q = torch.randn(19, 3, generator=generator, dtype=torch.float64).softmax(1)
    legacy_u, native_v = initialize_factors(hard, 4, 3, seed=9)
    native_material = torch.cat((native_z.new_ones(19, 1), native_z, native_q), 1)
    original = LowRankMoments.apply(legacy_u, native_v, hard, native_material, .05, 6)
    native_rows = []
    for coordinate_dimension in (2, 7):
        native_f = torch.randn(19, coordinate_dimension, generator=generator, dtype=torch.float32)
        native_a, native_b = torch.zeros(coordinate_dimension, 3), torch.zeros(3)
        native_u = native_f @ native_a + native_b
        observed = LowRankMoments.apply(native_u, native_v, hard, native_material, .05, 6)
        assert torch.equal(native_u, legacy_u) and torch.equal(observed, original)
        mass = original[:, 0]
        native_rows.append(dict(coordinate_dimension=coordinate_dimension, U0_bitwise_equal=True, moments_bitwise_equal=True,
                                student_X_bitwise_equal=torch.equal((observed[:, 1:4] / observed[:, :1]).float(),
                                                                    (original[:, 1:4] / original[:, :1]).float()),
                                student_Q_bitwise_equal=torch.equal((observed[:, 4:] / observed[:, :1]).float(),
                                                                    (original[:, 4:] / original[:, :1]).float()),
                                uniform_weights_bitwise_equal=torch.equal(torch.full_like(observed[:, 0], .25),
                                                                           torch.full_like(mass, .25))))
    return dict(fd=fd_rows, vjp=vjp_rows, zero_U=zero_rows, free_bias=bias_rows, native_CPU_P0=native_rows,
                reference_head_gradient_max=head_max, dense_adjoint_relative_residual_max=adjoint_max, counts=dict(COUNTS))



def test_source_linear_dense_reference_and_existing_pure_vjp_contract(record_property):
    """16 FD comparisons plus routing/zero/bias/nativeCPU assertions; no new API."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    COUNTS.update(reference_head_fits=0, dense_linear_adjoint_solves=0, newton_steps=0)
    try:
        report = run_checks()
    finally:
        torch.set_num_threads(previous)
    assert len(report["fd"]) == 16
    for key, value in report["counts"].items():
        record_property(key, value)
    record_property("fd_comparisons", len(report["fd"]))
    record_property("fd_max_absolute_error", max(row["absolute_error"] for row in report["fd"]))
    record_property("fd_max_relative_error", max(row["relative_error"] for row in report["fd"]))
    record_property("reference_head_gradient_max", report["reference_head_gradient_max"])
    record_property("dense_adjoint_relative_residual_max", report["dense_adjoint_relative_residual_max"])
