"""Focused FP64 synthetic QA; no production/data/cache/evaluator imports."""
import pytest
import torch

from src import macro_teacher_outer as helper


def toy():
    generator = torch.Generator().manual_seed(20261002)
    z = torch.randn(11, 4, generator=generator, dtype=torch.float64) * .3
    logits = torch.randn(11, 3, generator=generator, dtype=torch.float64)
    logits[:, 0] += 1.8  # A synthetic unequal-prior case, not real data.
    q = logits.softmax(1)
    theta = torch.randn(3, 5, generator=generator, dtype=torch.float64) * .2
    return z, q, theta


def objective(z, q, theta):
    x = torch.cat((z, z.new_ones(len(z), 1)), dim=1)
    lp = (x @ theta.T).log_softmax(1)
    # Independent literal classwise conditional-risk reference.
    return sum(-(q[:, c]*lp[:, c]).sum()/q[:, c].sum() for c in range(q.shape[1]))/q.shape[1]


def test_analytic_RHS_matches_autograd_and_classwise_risk(record_property):
    z, q, theta = toy()
    theta.requires_grad_()
    actual, rhs = helper.macro_teacher_outer(z, q, theta)
    reference = objective(z, q, theta)
    expected, = torch.autograd.grad(reference, theta)
    error = float((rhs-expected).abs().max())
    record_property("RHS_max_abs", error)
    assert actual == pytest.approx(float(reference.detach()), abs=2e-15, rel=2e-15)
    torch.testing.assert_close(rhs, expected, atol=2e-15, rtol=2e-15)
    assert not rhs.requires_grad


@pytest.mark.parametrize("epsilon", [1e-5, 2e-6])
@pytest.mark.parametrize("direction", ["weights_and_bias", "bias_only"])
def test_two_epsilon_directional_FD(epsilon, direction, record_property):
    z, q, theta = toy()
    delta = torch.arange(theta.numel(), dtype=theta.dtype).reshape_as(theta)/theta.numel()-.3
    if direction == "bias_only":
        delta[:, :-1] = 0
    _, rhs = helper.macro_teacher_outer(z, q, theta)
    measured = float((objective(z, q, theta+epsilon*delta)-objective(z, q, theta-epsilon*delta))/(2*epsilon))
    predicted = float((rhs*delta).sum())
    record_property("FD_abs", abs(measured-predicted))
    record_property("FD_predicted", predicted)
    assert abs(measured-predicted) <= 2e-10+2e-7*abs(measured)


@pytest.mark.parametrize("chunk", [1, 2, 4, 11, 23])
def test_chunked_dense_FP64_agreement(chunk, record_property):
    z, q, theta = toy()
    dense_value, dense_rhs = helper.macro_teacher_outer(z, q, theta, chunk=11)
    value, rhs = helper.macro_teacher_outer(z, q, theta, chunk=chunk)
    record_property("chunk_RHS_abs", float((rhs-dense_rhs).abs().max()))
    assert value == pytest.approx(dense_value, abs=2e-15, rel=2e-15)
    torch.testing.assert_close(rhs, dense_rhs, atol=2e-15, rtol=2e-15)


def test_equal_class_mass_equals_original_node_uniform_CE():
    z, _, theta = toy()
    z = z[:6]
    q = torch.tensor([[.6, .3, .1], [.1, .6, .3], [.3, .1, .6]]*2, dtype=torch.float64)
    weights, mass = helper.macro_teacher_weights(q)
    torch.testing.assert_close(mass, torch.full_like(mass, 2.), atol=0, rtol=0)
    assert torch.equal(weights, q/6)
    x = torch.cat((z, z.new_ones(6, 1)), dim=1)
    variable = theta.clone().requires_grad_()
    risk = -(q*(x @ variable.T).log_softmax(1)).sum()/6
    original_rhs, = torch.autograd.grad(risk, variable)
    value, rhs = helper.macro_teacher_outer(z, q, theta)
    assert value == pytest.approx(float(risk.detach()), abs=2e-15, rel=2e-15)
    torch.testing.assert_close(rhs, original_rhs, atol=2e-15, rtol=2e-15)


def test_equal_soft_class_risk_and_unequal_node_weights_no_renormalization():
    _, q, _ = toy()
    before = q.clone()
    weights, mass = helper.macro_teacher_weights(q)
    assert torch.equal(mass, q.sum(0)) and torch.equal(weights, q/(q.shape[1]*mass))
    torch.testing.assert_close(weights.sum(0), torch.full_like(mass, 1/3), atol=2e-16, rtol=2e-16)
    assert float((weights.sum(1)-1/len(q)).abs().max()) > .005
    assert torch.equal(q, before) and not weights.requires_grad
    assert weights.data_ptr() != q.data_ptr()


def test_constant_teacher_prior_has_uniform_macro_bias_rhs_at_zero_head():
    z, _, theta = toy()
    q = torch.tensor([.82, .15, .03], dtype=torch.float64).repeat(len(z), 1)
    _, macro_rhs = helper.macro_teacher_outer(z, q, torch.zeros_like(theta))
    assert float(macro_rhs[:, -1].abs().max()) < 2e-16
    node_uniform_bias_rhs = torch.full((3,), 1/3, dtype=torch.float64)-q.mean(0)
    assert float(node_uniform_bias_rhs.abs().max()) > .4


def test_logit_common_shift_invariance_and_rhs_gauge(record_property):
    z, q, theta = toy()
    common = torch.tensor([.14, -.25, .07, .31, -.11], dtype=torch.float64)
    value, rhs = helper.macro_teacher_outer(z, q, theta)
    shifted_value, shifted_rhs = helper.macro_teacher_outer(z, q, theta+common[None, :])
    record_property("shift_RHS_abs", float((rhs-shifted_rhs).abs().max()))
    assert value == pytest.approx(shifted_value, abs=2e-15, rel=2e-15)
    torch.testing.assert_close(rhs, shifted_rhs, atol=2e-15, rtol=2e-15)
    torch.testing.assert_close(rhs.sum(0), torch.zeros(5, dtype=torch.float64), atol=2e-16, rtol=0)


def test_source_Q_z_theta_not_mutated_and_aligned_permutation_invariant():
    z, q, theta = toy()
    originals = [value.clone() for value in (z, q, theta)]
    first, rhs = helper.macro_teacher_outer(z, q, theta, chunk=2)
    order = torch.tensor([10, 2, 8, 1, 6, 9, 0, 5, 4, 7, 3])
    second, permuted_rhs = helper.macro_teacher_outer(z[order], q[order], theta, chunk=4)
    assert first == pytest.approx(second, abs=2e-15, rel=2e-15)
    torch.testing.assert_close(rhs, permuted_rhs, atol=2e-15, rtol=2e-15)
    assert all(torch.equal(current, original) for current, original in zip((z, q, theta), originals, strict=True))


@pytest.mark.parametrize("case", ["negative", "nan", "inf", "nonprobability", "strict32eps", "missing_class",
                                "requires_grad", "FP32", "empty", "dimension"])
def test_malformed_Q_rejected_without_mutation(case):
    _, q, _ = toy()
    if case in ("negative", "nan", "inf"):
        q[0, 0] = {"negative": -.1, "nan": float("nan"), "inf": float("inf")}[case]
    elif case == "nonprobability":
        q[0, 0] += .01
    elif case == "strict32eps":
        q[0, 0] += 64*torch.finfo(q.dtype).eps
    elif case == "missing_class":
        q[:, 2] = 0
        q[:, 0] = 1-q[:, 1]
    elif case == "requires_grad":
        q.requires_grad_()
    elif case == "FP32":
        q = q.float()
    elif case == "empty":
        q = q[:0]
    elif case == "dimension":
        q = q[0]
    before = q.clone()
    with pytest.raises(ValueError):
        helper.macro_teacher_weights(q)
    torch.testing.assert_close(q, before, atol=0, rtol=0, equal_nan=True)


@pytest.mark.parametrize("case", ["theta_shape", "theta_nan", "z_shape", "z_nan", "z_requires_grad", "FP32_theta", "rows"])
def test_invalid_feature_head_shapes_and_values_reject(case):
    z, q, theta = toy()
    if case == "theta_shape":
        theta = theta[:, :-1]
    elif case == "theta_nan":
        theta[0, 0] = float("nan")
    elif case == "z_shape":
        z = z.flatten()
    elif case == "z_nan":
        z[0, 0] = float("nan")
    elif case == "z_requires_grad":
        z.requires_grad_()
    elif case == "FP32_theta":
        theta = theta.float()
    elif case == "rows":
        z = z[:-1]
    with pytest.raises(ValueError):
        helper.macro_teacher_outer(z, q, theta)


@pytest.mark.parametrize("chunk", [True, 0, -1, 1.5])
def test_invalid_chunk_rejects(chunk):
    with pytest.raises(ValueError):
        helper.macro_teacher_outer(*toy(), chunk=chunk)


def test_stops_before_entry_and_between_chunks_preserve_inputs():
    arguments = toy()
    before = [value.clone() for value in arguments]
    for stop in (lambda: True, iter([False, False, True]).__next__):
        with pytest.raises(InterruptedError):
            helper.macro_teacher_outer(*arguments, chunk=2, stop=stop)
    assert all(torch.equal(a, b) for a, b in zip(arguments, before, strict=True))
    with pytest.raises(ValueError, match="callable"):
        helper.macro_teacher_outer(*arguments, stop=None)


def test_tiny_positive_class_mass_not_clamped_or_replaced():
    q = torch.tensor([[1.-2e-14, 1e-14, 1e-14], [1.-4e-14, 3e-14, 1e-14]], dtype=torch.float64)
    weights, mass = helper.macro_teacher_weights(q)
    assert 0 < float(mass.min()) < 1e-12
    assert torch.equal(weights, q/(3*mass))
    torch.testing.assert_close(weights.sum(0), torch.full_like(mass, 1/3), atol=1e-16, rtol=1e-16)
