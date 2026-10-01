"""Actual tiny CPU teacher-solver contract, separate from closure math proofs.

These synthetic tests do not fit a real graph, write cache files, or certify
the convergence of the historical dataset teachers whose weights were lost.
"""

import inspect

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src import ntk_teacher, nystrom_ce
from src.teacher import fit_logistic

PARITY_METRICS = []


@pytest.fixture(scope="module", autouse=True)
def tiny_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def synthetic_problem():
    generator = torch.Generator().manual_seed(9812)
    phi = torch.randn(24, 4, dtype=torch.double, generator=generator)
    mask = torch.zeros(24, dtype=torch.bool)
    mask[torch.tensor([0, 1, 2, 3, 5, 7, 8, 10, 12, 14, 16, 18, 20, 23])] = True
    labels = torch.zeros(24, dtype=torch.long)
    labels[mask] = torch.tensor([0, 1, 2, 0, 1, 2, 0, 2, 1, 0, 1, 2, 1, 0])
    return phi, labels, mask


def objective_and_gradient(phi, labels, mask, weight, gamma):
    variable = weight.detach().clone().requires_grad_()
    loss = F.cross_entropy(phi[mask] @ variable, labels[mask])
    loss = loss + gamma / int(mask.sum()) * variable.square().sum()
    gradient = torch.autograd.grad(loss, variable)[0]
    return float(loss.detach()), gradient


@pytest.mark.parametrize("gamma", [1e-5, 1e-4, 1e-3, .01])
@pytest.mark.parametrize("resident", [False, True])
def test_actual_converged_solver_parity_all_frozen_gammas(gamma, resident):
    phi, labels, mask = synthetic_problem()
    original_phi, original_labels = phi.clone(), labels.clone()
    targets = F.one_hot(labels[mask], num_classes=3).double()
    historical = fit_logistic(phi[mask], targets, gamma)
    logits, weight = nystrom_ce.fit_streaming_teacher(
        phi.numpy(), labels, mask, gamma=gamma, chunk=5,
        max_iter=1000, resident_training=resident)
    assert nystrom_ce.fit_streaming_teacher.last_route["actual"] == (
        "resident" if resident else "streaming")
    old_objective, old_gradient = objective_and_gradient(phi, labels, mask, historical, gamma)
    new_objective, new_gradient = objective_and_gradient(phi, labels, mask, weight, gamma)
    old_logits = phi @ historical
    maximum_logits = float((logits - old_logits).abs().max())
    maximum_weight = float((weight - historical).abs().max())
    objective_difference = abs(new_objective - old_objective)
    assert float(new_gradient.abs().max()) <= 1e-5
    assert float(old_gradient.abs().max()) <= 1e-5
    assert objective_difference <= 1e-8
    assert maximum_logits <= 2e-4
    torch.testing.assert_close(weight, historical, rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(logits, phi @ weight, rtol=1e-14, atol=1e-14)
    assert logits.shape == (24, 3) and logits.dtype == torch.double
    assert weight.shape == (4, 3) and weight.dtype == torch.double
    torch.testing.assert_close(phi, original_phi, rtol=0, atol=0)
    torch.testing.assert_close(labels, original_labels, rtol=0, atol=0)
    PARITY_METRICS.append(dict(gamma=gamma, resident=resident,
                              maximum_logit_error=maximum_logits,
                              maximum_weight_error=maximum_weight,
                              objective_difference=objective_difference,
                              maximum_new_gradient=float(new_gradient.abs().max()),
                              maximum_old_gradient=float(old_gradient.abs().max())))


def test_helper_excluded_label_poisoning_never_reaches_vocabulary_or_context():
    phi, labels, train = synthetic_problem()
    validation = torch.zeros(len(phi), dtype=torch.bool)
    validation[torch.tensor([4, 6, 9, 11])] = True
    graph_labels = labels.clone()
    graph_labels[validation] = torch.tensor([2, 1, 0, 2])
    ignored = ~(train | validation)
    poisoned = graph_labels.clone()
    poisoned[ignored] = 1000000
    graph = dict(y=graph_labels)
    poison_graph = dict(y=poisoned)
    first, first_val = ntk_teacher.teacher_inputs(graph, train, validation)
    second, second_val = ntk_teacher.teacher_inputs(poison_graph, train, validation)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    torch.testing.assert_close(first_val, second_val, rtol=0, atol=0)
    assert bool((first[~train] == 0).all())
    first_context = ntk_teacher._inputs(phi, train, first, validation, first_val)
    second_context = ntk_teacher._inputs(phi, train, second, validation, second_val)
    assert first_context == second_context and first_context["classes"] == 3
    _, weight = nystrom_ce.fit_streaming_teacher(
        phi.numpy(), second, train, gamma=.01, chunk=5, resident_training=True)
    assert weight.shape[1] == 3
    torch.testing.assert_close(graph["y"], graph_labels, rtol=0, atol=0)
    torch.testing.assert_close(poison_graph["y"], poisoned, rtol=0, atol=0)


def test_stop_before_fit_prevents_optimizer_construction(monkeypatch):
    phi, labels, mask = synthetic_problem()

    def forbidden_optimizer(*args, **kwargs):
        raise AssertionError("Optimizer created after a pre-fit stop")

    monkeypatch.setattr(torch.optim, "LBFGS", forbidden_optimizer)
    with pytest.raises(InterruptedError, match="interrupted"):
        nystrom_ce.fit_streaming_teacher(phi.numpy(), labels, mask, stop=lambda: True)


def test_stop_during_resident_transfer_prevents_optimizer_construction(monkeypatch):
    phi, labels, mask = synthetic_problem()
    calls = 0

    def stop():
        nonlocal calls
        calls += 1
        return calls == 3

    def forbidden_optimizer(*args, **kwargs):
        raise AssertionError("Optimizer created after an interrupted resident transfer")

    monkeypatch.setattr(torch.optim, "LBFGS", forbidden_optimizer)
    with pytest.raises(InterruptedError, match="transfer interrupted"):
        nystrom_ce.fit_streaming_teacher(phi.numpy(), labels, mask, chunk=5,
                                        resident_training=True, stop=stop)
    assert calls == 3


@pytest.mark.parametrize("resident", [False, True])
def test_stop_during_first_closure_propagates_instead_of_returning_teacher(resident):
    phi, labels, mask = synthetic_problem()

    def stop():
        return inspect.currentframe().f_back.f_code.co_name == "closure"

    with pytest.raises(InterruptedError, match="fitting interrupted"):
        nystrom_ce.fit_streaming_teacher(phi.numpy(), labels, mask, chunk=5,
                                        resident_training=resident, stop=stop)


def test_stop_during_streamed_training_blocks_propagates():
    phi, labels, mask = synthetic_problem()
    blocks_seen = 0

    def stop():
        nonlocal blocks_seen
        if inspect.currentframe().f_back.f_code.co_name == "training_blocks":
            blocks_seen += 1
            return blocks_seen == 2
        return False

    with pytest.raises(InterruptedError, match="during streaming"):
        nystrom_ce.fit_streaming_teacher(phi.numpy(), labels, mask, chunk=5,
                                        resident_training=False, stop=stop)
    assert blocks_seen == 2


def test_stop_during_prediction_propagates_after_converged_fit():
    phi, labels, mask = synthetic_problem()

    def stop():
        frame = inspect.currentframe().f_back
        return frame.f_code.co_name == "fit_streaming_teacher" and "prediction" in frame.f_locals

    with pytest.raises(InterruptedError, match="prediction interrupted"):
        nystrom_ce.fit_streaming_teacher(phi.numpy(), labels, mask, gamma=.01, chunk=5,
                                        resident_training=True, stop=stop)


def test_no_optimizer_update_cannot_be_accepted_as_a_converged_teacher(monkeypatch):
    phi, labels, mask = synthetic_problem()
    monkeypatch.setattr(torch.optim.LBFGS, "step", lambda self, closure: None)
    with pytest.raises(RuntimeError, match="did not converge"):
        nystrom_ce.fit_streaming_teacher(phi.numpy(), labels, mask, gamma=.01,
                                        resident_training=True)


def test_nonfinite_weight_cannot_be_accepted_as_a_teacher(monkeypatch):
    phi, labels, mask = synthetic_problem()

    def poison_step(optimizer, closure):
        with torch.no_grad():
            optimizer.param_groups[0]["params"][0].fill_(np.nan)

    monkeypatch.setattr(torch.optim.LBFGS, "step", poison_step)
    with pytest.raises(RuntimeError, match="did not converge"):
        nystrom_ce.fit_streaming_teacher(phi.numpy(), labels, mask, gamma=.01,
                                        resident_training=True)
