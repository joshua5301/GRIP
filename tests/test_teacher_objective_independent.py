"""Independent synthetic checks of the unchanged no-bias teacher objectives.

Optimizer interception evaluates the real closures at prescribed weights. It
does not replace their loss/gradient arithmetic, train real datasets, or modify
production source. Full solver and stop/cache tests belong to separate reviewers.
"""

import pytest
import torch

from src.nystrom_ce import fit_streaming_teacher
from src.teacher import fit_logistic


@pytest.fixture(scope="module", autouse=True)
def small_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def synthetic_inputs(dtype=torch.double):
    generator = torch.Generator().manual_seed(4713)
    x = torch.randn(20, 4, generator=generator, dtype=torch.double).to(dtype)
    selected = (0, 1, 2, 4, 5, 7, 8, 9, 12, 13, 15, 17, 18, 19)
    train = torch.tensor([i in selected for i in range(len(x))])
    # The core's class vocabulary comes from its entire input labels tensor.
    # Only training labels are supplied; excluded positions are deliberately 0.
    labels = torch.zeros(len(x), dtype=torch.long)
    labels[train] = torch.arange(int(train.sum())) % 3
    weight = .2 * torch.randn(4, 3, generator=generator, dtype=torch.double)
    direction = torch.randn(4, 3, generator=generator, dtype=torch.double)
    return x, labels, train, weight, direction


class ProbeComplete(Exception):
    """End after inspecting a real closure, before the real optimizer iterates."""


def inspect_closure(monkeypatch, kind, x, labels, train, gamma, prescribed):
    observed = {}

    class ProbeOptimizer:
        def __init__(self, parameters, **controls):
            parameters = list(parameters)
            assert len(parameters) == 1  # No additional learnable bias.
            self.weight = parameters[0]
            observed["initial"] = self.weight.detach().clone()
            observed["controls"] = controls

        def zero_grad(self):
            self.weight.grad = None

        def step(self, closure):
            with torch.no_grad():
                self.weight.copy_(prescribed)
            value = closure()
            observed["value"] = value.detach().clone()
            observed["gradient"] = self.weight.grad.detach().clone()
            raise ProbeComplete

    with monkeypatch.context() as patch:
        patch.setattr(torch.optim, "LBFGS", ProbeOptimizer)
        with pytest.raises(ProbeComplete):
            if kind == "original":
                targets = torch.nn.functional.one_hot(labels[train], 3).double()
                fit_logistic(x[train], targets, gamma, steps=1000)
            else:
                fit_streaming_teacher(x.numpy(), labels, train, gamma, chunk=3,
                                      max_iter=1000, resident_training=kind == "resident")
    return observed


def independent_objective(x, labels, train, weight, gamma):
    # Use logsumexp and integer indexing rather than either implementation's
    # cross_entropy or manually supplied softmax-error matrix.
    scores = x[train].double() @ weight
    row = torch.arange(len(scores))
    data_term = (torch.logsumexp(scores, 1) - scores[row, labels[train]]).mean()
    return data_term + gamma / len(scores) * weight.square().sum()


@pytest.mark.parametrize("kind", ["original", "streaming", "resident"])
@pytest.mark.parametrize("gamma", [1e-5, 1e-4, .001, .01])
def test_real_teacher_closure_matches_independent_objective_and_directional_gradient(
        monkeypatch, record_property, kind, gamma):
    x, labels, train, weight, direction = synthetic_inputs()
    before = (x.clone(), labels.clone(), train.clone())
    observed = inspect_closure(monkeypatch, kind, x, labels, train, gamma, weight)
    variable = weight.detach().requires_grad_()
    reference = independent_objective(x, labels, train, variable, gamma)
    derivative = torch.autograd.grad(reference, variable)[0]
    value_error = float((observed["value"] - reference.detach()).abs())
    gradient_error = float((observed["gradient"] - derivative).abs().max())
    assert value_error < 3e-15 and gradient_error < 3e-15
    epsilon = 1e-5
    finite = float((independent_objective(x, labels, train, weight + epsilon * direction, gamma)
                    - independent_objective(x, labels, train, weight - epsilon * direction, gamma))
                   / (2 * epsilon))
    analytic = float((observed["gradient"] * direction).sum())
    fd_error = abs(finite - analytic)
    assert fd_error < 2e-9
    record_property("objective_max_abs_error", value_error)
    record_property("gradient_max_abs_error", gradient_error)
    record_property("directional_fd_abs_error", fd_error)
    assert observed["initial"].dtype == torch.double
    assert observed["initial"].shape == (4, 3)
    assert torch.equal(observed["initial"], torch.zeros_like(observed["initial"]))
    assert observed["controls"] == {"max_iter": 1000, "line_search_fn": "strong_wolfe"}
    assert all(torch.equal(a, b) for a, b in zip(before, (x, labels, train)))


@pytest.mark.parametrize("kind", ["original", "resident"])
def test_float32_input_is_promoted_before_loss_and_gradient_arithmetic(monkeypatch, kind):
    x, labels, train, weight, _ = synthetic_inputs(torch.float32)
    observed = inspect_closure(monkeypatch, kind, x, labels, train, .001, weight)
    variable = weight.detach().requires_grad_()
    expected = independent_objective(x, labels, train, variable, .001)
    gradient = torch.autograd.grad(expected, variable)[0]
    torch.testing.assert_close(observed["value"], expected.detach(), rtol=1e-14, atol=1e-15)
    torch.testing.assert_close(observed["gradient"], gradient, rtol=1e-13, atol=1e-15)


def test_streaming_route_requires_float64_phi_cache(monkeypatch):
    # The existing streaming blocks move device without promoting dtype. The
    # proposed strict helper must reject a float32 feature cache before calling
    # this route, including an automatic resident-memory fallback.
    x, labels, train, weight, _ = synthetic_inputs(torch.float32)
    with pytest.raises(RuntimeError, match="same dtype"):
        inspect_closure(monkeypatch, "streaming", x, labels, train, .001, weight)


@pytest.mark.parametrize("kind", ["original", "streaming", "resident"])
def test_duplicating_training_rows_changes_only_gamma_over_n_regularization(monkeypatch, kind):
    x, labels, train, weight, _ = synthetic_inputs()
    gamma = .01
    once = inspect_closure(monkeypatch, kind, x, labels, train, gamma, weight)
    twice = inspect_closure(monkeypatch, kind, x.repeat_interleave(2, 0),
                            labels.repeat_interleave(2), train.repeat_interleave(2), gamma, weight)
    n = int(train.sum())
    # Mean CE is unchanged. gamma/n is halved, rather than using source node
    # count or silently interpreting gamma as an unnormalized fixed penalty.
    expected_value = gamma / (2 * n) * weight.square().sum()
    expected_gradient = gamma / n * weight
    torch.testing.assert_close(once["value"] - twice["value"], expected_value,
                               rtol=1e-11, atol=3e-15)
    torch.testing.assert_close(once["gradient"] - twice["gradient"], expected_gradient,
                               rtol=1e-10, atol=3e-15)


@pytest.mark.parametrize("kind", ["original", "streaming", "resident"])
def test_common_class_weight_shift_has_only_regularization_derivative(monkeypatch, kind):
    x, labels, train, weight, _ = synthetic_inputs()
    gamma = .01
    observed = inspect_closure(monkeypatch, kind, x, labels, train, gamma, weight)
    direction = torch.tensor([.2, -.1, .3, .4], dtype=torch.double)[:, None].expand_as(weight)
    actual = float((observed["gradient"] * direction).sum())
    expected = float(2 * gamma / int(train.sum()) * (weight * direction).sum())
    # Adding the same input-dependent score to all classes preserves softmax CE;
    # the no-bias parameter L2 still regularizes this common-class direction.
    assert abs(actual - expected) < 3e-15


@pytest.mark.parametrize("change", ["empty", "shape", "dtype", "zero_chunk", "float_chunk"])
def test_invalid_training_support_rejects_before_optimizer_allocation(monkeypatch, change):
    x, labels, train, _, _ = synthetic_inputs()
    chunk = 3
    if change == "empty":
        train = torch.zeros_like(train)
    elif change == "shape":
        train = train[:-1]
    elif change == "dtype":
        train = train.long()
    elif change == "zero_chunk":
        chunk = 0
    else:
        chunk = 1.5
    monkeypatch.setattr(torch.optim, "LBFGS", lambda *args, **kwargs: pytest.fail("invalid support reached optimizer"))
    with pytest.raises(ValueError):
        fit_streaming_teacher(x.numpy(), labels, train, chunk=chunk, resident_training=True)
