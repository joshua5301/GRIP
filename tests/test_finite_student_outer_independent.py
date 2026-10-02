"""Independent synthetic finite-horizon QA; run only after root authorization.

The production pure engine is imported directly.  NumPy below
implements the mathematical SGD reference without autograd, torch optimizers,
the engine decoder, or any production moment/head routine.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

EPSILONS = (1e-5, 2e-6)
FD_ATOL = 2e-8
FD_RTOL = 2e-5
PARITY_ATOL = 2e-12
PARITY_RTOL = 2e-12
ROUTES = ("sgc_mlp", "gcn")


@pytest.fixture(scope="module")
def engine():
    from src import finite_student_outer

    return finite_student_outer


def _softmax(values):
    shifted = values - values.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _logsoftmax(values):
    shifted = values - values.max(axis=1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def _forward(parameters, inputs, adjacency=None):
    w1, b1, w2, b2 = parameters
    first = inputs @ w1.T
    if adjacency is not None:
        first = adjacency @ first
    hidden = np.maximum(first + b1, 0)
    second = hidden @ w2.T
    if adjacency is not None:
        second = adjacency @ second
    return _logsoftmax(second + b2)


def _decode(moment, transform, dimension):
    mass = moment[:, 0]
    normalized_features = moment[:, 1:1 + dimension] / mass[:, None]
    physical_features = (normalized_features * transform["scale"]
                         + transform["output_center"] + transform["center"])
    targets = moment[:, 1 + dimension:] / mass[:, None]
    return physical_features, targets, mass


def _manual_unroll(moment, transform, dimension, initial):
    """Five simultaneous ordinary SGD updates, including decay on both biases."""
    features, targets, _ = _decode(moment, transform, dimension)
    parameters = tuple(value.copy() for value in initial)
    states = [parameters]
    trace = []
    smallest_margin = np.inf
    for _ in range(5):
        w1, b1, w2, b2 = parameters
        preactivation = features @ w1.T + b1
        smallest_margin = min(smallest_margin, float(np.abs(preactivation).min()))
        hidden = np.maximum(preactivation, 0)
        logits = hidden @ w2.T + b2
        ce = -float(np.sum(targets * _logsoftmax(logits)) / len(features))
        decay = .001 / 2 * sum(float(np.sum(value * value)) for value in parameters)
        # This generalized residual also differentiates targets correctly off
        # the simplex.  Our finite-difference paths preserve the simplex.
        error = (_softmax(logits) * targets.sum(axis=1, keepdims=True) - targets) / len(features)
        hidden_error = (error @ w2) * (preactivation > 0)
        gradients = (
            hidden_error.T @ features + .001 * w1,
            hidden_error.sum(axis=0) + .001 * b1,
            error.T @ hidden + .001 * w2,
            error.sum(axis=0) + .001 * b2,
        )
        trace.append((ce, decay, ce + decay))
        parameters = tuple(value - .1 * gradient for value, gradient in zip(parameters, gradients))
        states.append(parameters)
    return parameters, states, trace, smallest_margin


def _manual_losses(moment, fixture, initial=None):
    initial = fixture["initial"] if initial is None else initial
    final, _, _, margin = _manual_unroll(moment, fixture["transform"], 3, initial)
    targets = fixture["q"]
    losses = {
        "sgc_mlp": -float(np.mean(np.sum(targets * _forward(final, fixture["h"]), axis=1))),
        "gcn": -float(np.mean(np.sum(targets * _forward(final, fixture["x"], fixture["s"]), axis=1))),
    }
    return losses, margin


@pytest.fixture
def tiny():
    random = np.random.default_rng(20261002)
    x = random.uniform(-.25, .25, (7, 3))
    adjacency = np.eye(7)
    for a, b in ((0, 1), (1, 2), (2, 3), (3, 4), (3, 5), (5, 6)):
        adjacency[a, b] = adjacency[b, a] = 1
    inv_degree = adjacency.sum(axis=1) ** -.5
    s = inv_degree[:, None] * adjacency * inv_degree[None, :]
    h = s @ (s @ x)  # idealized toy only, not a real cached-H equality claim
    q = _softmax(random.uniform(-.4, .4, (7, 2)))
    probability = _softmax(random.uniform(-.5, .5, (7, 3)))
    transform = dict(kind="rms", matrix=None, center=np.array([-.10, .05, .02]),
                     output_center=np.array([.03, -.02, .04]), scale=np.array(1.3))
    z = (h - transform["output_center"] - transform["center"]) / transform["scale"]
    material = np.column_stack((np.ones(7), z, q))
    moment = probability.T @ material / len(x)
    initial = (random.uniform(-.04, .04, (5, 3)), np.array([.7, .9, 1.1, .8, 1.2]),
               random.uniform(-.10, .10, (2, 5)), np.array([.07, -.04]))
    return dict(x=x, s=s, h=h, q=q, p=probability, material=material,
                moment=moment, transform=transform, initial=initial)


def _tensor(array):
    return torch.tensor(array, dtype=torch.float64)


def _arguments(fixture):
    transform = {key: _tensor(value) if isinstance(value, np.ndarray) else value
                 for key, value in fixture["transform"].items()}
    return dict(moments=_tensor(fixture["moment"]), transform=transform,
                initial_parameters=tuple(_tensor(value) for value in fixture["initial"]),
                source_x=_tensor(fixture["x"]), source_adjacency=_tensor(fixture["s"]),
                source_h=_tensor(fixture["h"]), source_q=_tensor(fixture["q"]))


def _invoke(engine, fixture):
    return engine.finite_student_outer_partials(**_arguments(fixture))


def _array(value):
    return value.detach().cpu().numpy()


def test_manual_simultaneous_sgd_all_four_parameters_and_trace(engine, tiny):
    result = _invoke(engine, tiny)
    expected, states, trace, margin = _manual_unroll(tiny["moment"], tiny["transform"], 3,
                                                  tiny["initial"])
    assert margin > .2
    assert len(result["inner_parameter_states"]) == 6
    assert len(result["inner_trace"]) == 5
    for actual_state, expected_state in zip(result["inner_parameter_states"], states):
        for actual, wanted in zip(actual_state, expected_state):
            np.testing.assert_allclose(_array(actual), wanted, atol=PARITY_ATOL, rtol=PARITY_RTOL)
    for actual, wanted in zip(result["adapted_parameters"], expected):
        np.testing.assert_allclose(_array(actual), wanted, atol=PARITY_ATOL, rtol=PARITY_RTOL)
    for index, (actual, wanted) in enumerate(zip(result["inner_trace"], trace)):
        assert actual["step"] == index
        for key, value in zip(("uniform_soft_ce", "weight_decay_objective", "regularized_objective"), wanted):
            assert float(actual[key]) == pytest.approx(value, abs=PARITY_ATOL, rel=PARITY_RTOL)
    losses, _ = _manual_losses(tiny["moment"], tiny)
    for route in ROUTES:
        assert float(result["outer_losses"][route]) == pytest.approx(losses[route], abs=PARITY_ATOL,
                                                                    rel=PARITY_RTOL)


def _moment_direction(fixture, kind):
    moment = fixture["moment"]
    direction = np.zeros_like(moment)
    if kind == "features":
        direction[:, 1:4] = np.array([[.08, -.04, .03], [-.03, .07, .01], [.02, -.03, -.06]])
    elif kind == "targets":
        direction[:, 4:] = np.array([[.08, -.08], [-.05, .05], [.03, -.03]])
    elif kind == "mass_denominator":
        dm = np.array([.04, -.03, -.01])
        direction[:, 0] = dm
        direction[:, 4:] = dm[:, None] * (moment[:, 4:] / moment[:, :1])
    elif kind == "source_probability":
        logits_direction = np.arange(21).reshape(7, 3) * .02 - .18
        p = fixture["p"]
        dp = p * (logits_direction - (p * logits_direction).sum(axis=1, keepdims=True))
        direction = dp.T @ fixture["material"] / len(p)
    elif kind == "row_scale_null":
        direction = moment * np.array([.10, -.07, .03])[:, None]
    else:
        raise AssertionError(kind)
    return direction


@pytest.mark.parametrize("kind", ("features", "targets", "mass_denominator",
                                  "source_probability", "row_scale_null"))
def test_full_five_step_moment_cotangent_matches_manual_fd(engine, tiny, kind):
    result = _invoke(engine, tiny)
    direction = _moment_direction(tiny, kind)
    for epsilon in EPSILONS:
        positive = tiny["moment"] + epsilon * direction
        negative = tiny["moment"] - epsilon * direction
        for value in (positive, negative):
            _, targets, mass = _decode(value, tiny["transform"], 3)
            assert (mass > 0).all()
            np.testing.assert_allclose(targets.sum(axis=1), 1, atol=2e-14, rtol=0)
        plus, plus_margin = _manual_losses(positive, tiny)
        minus, minus_margin = _manual_losses(negative, tiny)
        assert min(plus_margin, minus_margin) > .2
        for route in ROUTES:
            cotangent = _array(result["moment_gradients"][route])
            assert cotangent.shape == direction.shape and np.isfinite(cotangent).all()
            analytic = float(np.sum(cotangent * direction))
            numerical = (plus[route] - minus[route]) / (2 * epsilon)
            assert analytic == pytest.approx(numerical, abs=FD_ATOL, rel=FD_RTOL)
            if kind == "row_scale_null":
                assert abs(analytic) < 2e-12


@pytest.mark.parametrize("bias_index", (1, 3))
def test_initial_bias_is_trainable_and_decayed_not_silently_frozen(engine, tiny, bias_index):
    modified = dict(tiny)
    initial = [value.copy() for value in tiny["initial"]]
    initial[bias_index] += np.linspace(.02, .05, len(initial[bias_index]))
    modified["initial"] = tuple(initial)
    result = _invoke(engine, modified)
    wanted, _, _, _ = _manual_unroll(modified["moment"], modified["transform"], 3, initial)
    np.testing.assert_allclose(_array(result["adapted_parameters"][bias_index]), wanted[bias_index],
                               atol=PARITY_ATOL, rtol=PARITY_RTOL)
    assert not np.array_equal(wanted[bias_index], initial[bias_index])
    losses, _ = _manual_losses(modified["moment"], modified)
    for route in ROUTES:
        assert float(result["outer_losses"][route]) == pytest.approx(losses[route], abs=PARITY_ATOL,
                                                                    rel=PARITY_RTOL)


def test_decode_inverse_affine_rms_barycenters(engine, tiny):
    args = _arguments(tiny)
    hc, qc, mass = engine.decode_rms_moments(args["moments"], 3, args["transform"])
    wanted_h = tiny["p"].T @ tiny["h"] / tiny["p"].sum(axis=0)[:, None]
    wanted_q = tiny["p"].T @ tiny["q"] / tiny["p"].sum(axis=0)[:, None]
    np.testing.assert_allclose(_array(hc), wanted_h, atol=PARITY_ATOL, rtol=PARITY_RTOL)
    np.testing.assert_allclose(_array(qc), wanted_q, atol=PARITY_ATOL, rtol=PARITY_RTOL)
    np.testing.assert_allclose(_array(mass), tiny["p"].mean(axis=0), atol=PARITY_ATOL, rtol=PARITY_RTOL)


def test_original_graph_bias_order_noncommutation_and_identity(engine, tiny):
    params = tuple(_tensor(value) for value in tiny["initial"])
    actual = engine.functional_forward(params, _tensor(tiny["x"]), _tensor(tiny["s"]))
    np.testing.assert_allclose(_array(actual), _forward(tiny["initial"], tiny["x"], tiny["s"]),
                               atol=PARITY_ATOL, rtol=PARITY_RTOL)
    assert np.max(np.abs(tiny["s"].sum(axis=1) - 1)) > .01
    assert np.max(np.abs(_forward(tiny["initial"], tiny["h"]) - _array(actual))) > 1e-5
    # Moving the first bias inside S is wrong when S1 != 1.
    w1, b1, w2, b2 = tiny["initial"]
    incorrect_hidden = np.maximum(tiny["s"] @ (tiny["x"] @ w1.T + b1), 0)
    incorrect = _logsoftmax(tiny["s"] @ (incorrect_hidden @ w2.T) + b2)
    assert np.max(np.abs(incorrect - _array(actual))) > 1e-5
    identity_fixture = dict(tiny, s=np.eye(7), h=tiny["x"].copy())
    result = _invoke(engine, identity_fixture)
    assert torch.equal(result["outer_losses"]["sgc_mlp"], result["outer_losses"]["gcn"])
    assert torch.equal(result["moment_gradients"]["sgc_mlp"], result["moment_gradients"]["gcn"])


def test_source_row_permutation_and_distinct_cached_h_are_not_canonicalized(engine, tiny):
    baseline = _invoke(engine, tiny)
    permutation = np.array([4, 1, 6, 0, 5, 2, 3])
    altered = dict(tiny, x=tiny["x"][permutation], h=tiny["h"][permutation],
                   q=tiny["q"][permutation], s=tiny["s"][permutation][:, permutation])
    permuted = _invoke(engine, altered)
    for route in ROUTES:
        assert float(permuted["outer_losses"][route]) == pytest.approx(
            float(baseline["outer_losses"][route]), abs=PARITY_ATOL, rel=PARITY_RTOL)
        np.testing.assert_allclose(_array(permuted["moment_gradients"][route]),
                                   _array(baseline["moment_gradients"][route]),
                                   atol=PARITY_ATOL, rtol=PARITY_RTOL)
    changed_h = dict(tiny, h=tiny["h"] + np.array([.07, -.04, .03]))
    changed = _invoke(engine, changed_h)
    manual, _ = _manual_losses(tiny["moment"], changed_h)
    assert float(changed["outer_losses"]["sgc_mlp"]) == pytest.approx(manual["sgc_mlp"],
                                                                   abs=PARITY_ATOL, rel=PARITY_RTOL)
    assert torch.equal(changed["outer_losses"]["gcn"], baseline["outer_losses"]["gcn"])


def test_reset_input_immutability_and_no_hidden_warm_state(engine, tiny):
    args = _arguments(tiny)
    original = {key: value.clone() for key, value in args.items() if isinstance(value, torch.Tensor)}
    initial = tuple(value.clone() for value in args["initial_parameters"])
    rng_before = torch.get_rng_state().clone()
    first = engine.finite_student_outer_partials(**args)
    engine.finite_student_outer_partials(**_arguments(dict(tiny, q=tiny["q"][:, ::-1].copy())))
    second = engine.finite_student_outer_partials(**args)
    assert torch.equal(torch.get_rng_state(), rng_before)
    for key, value in original.items():
        assert torch.equal(args[key], value)
    for actual, old in zip(args["initial_parameters"], initial):
        assert torch.equal(actual, old) and actual.grad is None
    assert args["moments"].grad is None
    for route in ROUTES:
        assert torch.equal(first["outer_losses"][route], second["outer_losses"][route])
        assert torch.equal(first["moment_gradients"][route], second["moment_gradients"][route])
    for left, right in zip(first["adapted_parameters"], second["adapted_parameters"]):
        assert torch.equal(left, right)


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_private_geom_draw_matches_actual_stock_and_restores_global_rng(engine, dtype):
    from src.evaluation import _initialize_geom_uniform

    before = torch.get_rng_state().clone()
    params = engine.geom_uniform_initial(3, 2, hidden=5, dtype=dtype, device="cpu")
    assert torch.equal(torch.get_rng_state(), before)
    model = SimpleNamespace(layers=[
        SimpleNamespace(lin=SimpleNamespace(weight=torch.empty(5, 3, dtype=dtype)),
                        bias=torch.empty(5, dtype=dtype)),
        SimpleNamespace(lin=SimpleNamespace(weight=torch.empty(2, 5, dtype=dtype)),
                        bias=torch.empty(2, dtype=dtype)),
    ])
    with torch.random.fork_rng(devices=[]):
        _initialize_geom_uniform(model, 0)
    expected = (model.layers[0].lin.weight, model.layers[0].bias,
                model.layers[1].lin.weight, model.layers[1].bias)
    for index, (actual, wanted) in enumerate(zip(params, expected)):
        assert torch.equal(actual, wanted)
        assert not actual.requires_grad
        fan_in = 3 if index < 2 else 5
        assert float(actual.abs().max()) <= fan_in ** -.5
    assert torch.equal(torch.get_rng_state(), before)


@pytest.mark.parametrize("bad", ("mass_zero", "mass_negative", "nan_moment", "qc_not_simplex",
                                  "source_q_bad", "source_shape", "adjacency_shape", "param_nan",
                                  "param_shape", "rms_scale_zero", "rms_kind", "rms_matrix"))
def test_malformed_mathematical_inputs_fail_cleanly(engine, tiny, bad):
    args = _arguments(tiny)
    if bad == "mass_zero":
        args["moments"][0, 0] = 0
    elif bad == "mass_negative":
        args["moments"][0, 0] = -.1
    elif bad == "nan_moment":
        args["moments"][0, 1] = float("nan")
    elif bad == "qc_not_simplex":
        args["moments"][0, -1] += .05
    elif bad == "source_q_bad":
        args["source_q"][0, 0] = -.1
    elif bad == "source_shape":
        args["source_h"] = args["source_h"][:-1]
    elif bad == "adjacency_shape":
        args["source_adjacency"] = args["source_adjacency"][:-1]
    elif bad == "param_nan":
        args["initial_parameters"][2][0, 0] = float("nan")
    elif bad == "param_shape":
        params = list(args["initial_parameters"])
        params[3] = params[3][:-1]
        args["initial_parameters"] = tuple(params)
    elif bad == "rms_scale_zero":
        args["transform"]["scale"] = _tensor(np.array(0.))
    elif bad == "rms_kind":
        args["transform"]["kind"] = "identity"
    elif bad == "rms_matrix":
        args["transform"]["matrix"] = torch.eye(3, dtype=torch.float64)
    with pytest.raises(ValueError):
        engine.finite_student_outer_partials(**args)


def test_stop_before_and_during_unroll_leaves_inputs_intact(engine, tiny):
    args = _arguments(tiny)
    old = tuple(value.clone() for value in args["initial_parameters"])
    with pytest.raises(InterruptedError):
        engine.finite_student_outer_partials(**args, stop=lambda: True)
    polls = 0

    def stop():
        nonlocal polls
        polls += 1
        return polls >= 3

    with pytest.raises(InterruptedError):
        engine.finite_student_outer_partials(**args, stop=stop)
    assert polls == 3
    for actual, wanted in zip(args["initial_parameters"], old):
        assert torch.equal(actual, wanted)
