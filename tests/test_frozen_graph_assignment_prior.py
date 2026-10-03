"""Four new CPU graph-prior domains; no earlier suites or student fits."""
import copy
import hashlib
import math

import pytest
import torch

from src.frozen_graph_assignment_prior import (
    MODE, POLICY, build_prior_packet, load_frozen_prior, packet_context,
)
from src.initial_assignment_row_kl import InitialAssignmentKLMoments
from src.io import array_digest
from src.moments import augmented, decode_moments, make_material
from src.soft_ce_partition import implicit_moment_gradient, outer_value_gradient, solve_head_system

EPSILONS = (1e-5, 3e-6)
FD_ATOL, FD_RTOL = 2e-9, 2e-6
PENALTY, NEWTON_STEPS, STATIONARITY = .2, 16, 1e-11


def base(hard, cells, dtype):
    value = torch.full((len(hard), cells), math.log(.05 / cells), dtype=dtype)
    value.scatter_(1, hard[:, None], math.log(.95 + .05 / cells))
    return value


def packed(binary):
    degree = binary.sum(1)
    inverse = degree.rsqrt()
    return (inverse[:, None] * binary * inverse[None, :]).float().to_sparse_csr()


def fixture():
    binary = torch.eye(6, dtype=torch.float64)
    for a, b in ((0, 1), (0, 2), (0, 3), (3, 4)):
        binary[a, b] = binary[b, a] = 1
    hard = torch.tensor([0, 0, 1, 2, 2, 1])
    p0 = base(hard, 3, torch.float32).double().softmax(1)
    u0 = torch.zeros(6, 2, dtype=torch.float32)
    v0 = torch.tensor([[.2, -.3], [-.25, .15], [.1, .25]], dtype=torch.float32)
    refs = dict(data_digest='d' * 64, factor_seed=17, mixing=.05, native_fixture_SHA256='f' * 64)
    return binary, hard, p0, (u0, v0), refs


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def direction(value):
    result = torch.arange(1, value.numel() + 1, dtype=value.dtype).reshape_as(value)
    result -= result.mean()
    return result / result.norm()


def dense(u, v, hard, material, log_r):
    logits = base(hard, len(v), u.dtype) + u @ v.T / math.sqrt(u.shape[1])
    p, logp = logits.double().softmax(1), logits.double().log_softmax(1)
    return p.T @ material / len(p), (p * (logp - log_r)).sum() / len(p)


def derivative_fixture():
    binary, hard, p0, initial, refs = fixture()
    packet = build_prior_packet(packed(binary), p0, initial, refs)
    z = torch.tensor([[.25, -.5], [-.125, .25], [.5, .125], [-.25, .5],
                      [.75, -.125], [.125, .75]], dtype=torch.float64)
    q = torch.tensor([[.75, .125, .125], [.625, .25, .125], [.125, .75, .125],
                      [.125, .125, .75], [.25, .125, .625], [.125, .625, .25]], dtype=torch.float64)
    u = torch.tensor([[.1, -.2], [-.15, .05], [.2, .1], [-.1, .25], [.05, -.15], [.125, .2]], dtype=torch.float64)
    v = torch.tensor([[.2, -.3], [-.25, .15], [.1, .25]], dtype=torch.float64)
    return z, q, hard, u, v, packet['log_probability']


def assert_fd(prediction, plus, minus, epsilon, record_property, name):
    measured = float((plus - minus) / (2 * epsilon))
    error = abs(float(prediction) - measured)
    assert error <= FD_ATOL + FD_RTOL * abs(measured)
    record_property(name, error)


def test_binary_random_walk_unequal_degree_identity_isolates_and_native_P0(record_property):
    binary, hard, p0, initial, refs = fixture()
    before = [p0.clone(), initial[0].clone(), initial[1].clone()]
    packet = build_prior_packet(packed(binary), p0, initial, refs)
    expected = (binary / binary.sum(1, keepdim=True)) @ p0
    actual = packet['log_probability'].exp()
    torch.testing.assert_close(actual, expected, atol=2e-15, rtol=2e-14)
    torch.testing.assert_close(actual.sum(1), torch.ones(6, dtype=torch.float64), atol=1e-12, rtol=0)
    assert bool((actual > 0).all()) and not packet['log_probability'].requires_grad
    assert torch.equal(packet['log_probability'][5], p0[5].log())
    assert packet['diagnostics']['initial_KL_P0_to_R'] > .01
    assert max(abs(a - b) for a, b in zip(packet['diagnostics']['original_cell_mass'],
        packet['diagnostics']['diffused_cell_mass'])) > .1
    assert packet['policy'] == POLICY and packet['mode'] == MODE
    identity = build_prior_packet(torch.eye(6).to_sparse_csr(), p0, initial, refs)
    assert torch.equal(identity['log_probability'], p0.log())
    assert identity['diagnostics']['initial_KL_P0_to_R'] == 0
    assert torch.equal(p0, before[0]) and all(torch.equal(a, b) for a, b in zip(initial, before[1:]))
    assert packet['native_parameter_digests'] == [array_digest(value.numpy()) for value in initial]
    record_property('binary_RW_dense_oracle_max_abs', float((actual - expected).abs().max()))
    record_property('initial_KL_P0_to_R', packet['diagnostics']['initial_KL_P0_to_R'])


def test_canonical_CSR_source_packet_loader_and_mutations(tmp_path, monkeypatch, record_property):
    binary, hard, p0, initial, refs = fixture()
    adjacency = packed(binary)
    malformed = []
    for change in ('duplicate', 'unsorted', 'negative', 'nan', 'pointers', 'normalization'):
        crow, col, values = adjacency.crow_indices().clone(), adjacency.col_indices().clone(), adjacency.values().clone()
        if change == 'duplicate': col[1] = col[0]
        elif change == 'unsorted': col[0], col[1] = col[1].clone(), col[0].clone()
        elif change == 'negative': values[0] = -.1
        elif change == 'nan': values[0] = float('nan')
        elif change == 'pointers': crow[1] = crow[0]
        else: values *= 1.01
        malformed.append(torch.sparse_csr_tensor(crow, col, values, size=adjacency.shape))
    asymmetric = binary.clone(); asymmetric[0, 1] = 0
    malformed.append(packed(asymmetric))
    no_loop = binary.clone(); no_loop[0, 0] = 0
    malformed.append(packed(no_loop))
    for candidate in malformed:
        with pytest.raises(ValueError): build_prior_packet(candidate, p0, initial, refs)
    for bad_p0 in (p0 * .9, p0.float(), p0.clone().requires_grad_()):
        with pytest.raises(ValueError): build_prior_packet(adjacency, bad_p0, initial, refs)
    bad_p0 = p0.clone(); bad_p0[0, 0] = 0
    with pytest.raises(ValueError): build_prior_packet(adjacency, bad_p0, initial, refs)
    with pytest.raises(ValueError): build_prior_packet(adjacency, p0, (initial[0] + .01, initial[1]), refs)
    with pytest.raises(ValueError): build_prior_packet(adjacency, p0, initial, dict(refs, mixing=.1))
    packet = build_prior_packet(adjacency, p0, initial, refs)
    artifact = tmp_path / 'prior.pt'; torch.save(packet, artifact)
    pin, context, dimensions = sha(artifact), packet_context(packet), packet['dimensions']
    assert set(context) == {'schema', 'mode', 'dimensions', 'source_refs', 'native_parameter_digests',
                            'log_probability_digest', 'context_digest'}
    with monkeypatch.context() as guard:
        def forbidden(*args, **kwargs): raise AssertionError('Loader recreated a graph/prior/RNG')
        guard.setattr(torch.sparse, 'mm', forbidden)
        guard.setattr(torch, 'randn', forbidden)
        guard.setattr('src.large_quotient_pilot.raw_looped_support', forbidden)
        loaded = load_frozen_prior(artifact, pin, dimensions, 'cpu', context)
    assert loaded.dtype == torch.float64 and not loaded.requires_grad
    assert torch.equal(loaded, packet['log_probability'])
    loaded[0, 0] += 1
    assert torch.equal(load_frozen_prior(artifact, pin, dimensions, 'cpu', context), packet['log_probability'])
    for bad_context in (dict(context, schema=True), dict(context, mode='original_P0_anchor'),
        dict(context, native_parameter_digests=['0' * 64, context['native_parameter_digests'][1]]),
        dict(context, source_refs=dict(refs, factor_seed=99))):
        with pytest.raises(ValueError): load_frozen_prior(artifact, pin, dimensions, 'cpu', bad_context)
    with pytest.raises(ValueError): load_frozen_prior(artifact, pin, dict(dimensions, rank=True), 'cpu', context)
    altered = tmp_path / 'mutated.pt'
    tensor_change = copy.deepcopy(packet); tensor_change['log_probability'][0, 0] += .1; torch.save(tensor_change, altered)
    with pytest.raises(ValueError, match='artifact bytes'):
        load_frozen_prior(altered, pin, dimensions, 'cpu', context)
    with pytest.raises(ValueError, match='logR tensor changed'):
        load_frozen_prior(altered, sha(altered), dimensions, 'cpu', context)
    csr_change = copy.deepcopy(packet); csr_change['input_descriptors']['packed_CSR']['crow']['tensor'] = '0' * 64
    torch.save(csr_change, altered)
    with pytest.raises(ValueError, match='metadata changed'):
        load_frozen_prior(altered, sha(altered), dimensions, 'cpu', context)
    policy_change = copy.deepcopy(packet); policy_change['policy']['coefficient'] = 2.; torch.save(policy_change, altered)
    with pytest.raises(ValueError, match='schema or policy'):
        load_frozen_prior(altered, sha(altered), dimensions, 'cpu', context)
    record_property('malformed_CSR_cases_rejected', len(malformed))
    record_property('loader_graph_prior_RNG_reconstruction_calls', 0)


def test_frozen_graph_KL_streamed_joint_VJP_and_two_epsilon_FD(record_property):
    z, q, hard, u, v, log_r = derivative_fixture()
    material = make_material(z, q)
    cotangent = torch.tensor([[.2, -.3, .1, .125, -.15, .05], [-.1, .25, -.2, -.075, .1, .15],
                              [.15, -.05, .3, .2, -.125, -.05]], dtype=torch.float64)
    scalar, scale = .7, .9
    d_u, d_v = direction(u), direction(v)
    old_log_r = log_r.clone()
    for chunk in (1, 3, 6):
        au, av, am = u.clone().requires_grad_(), v.clone().requires_grad_(), material.clone().requires_grad_()
        m, r = InitialAssignmentKLMoments.apply(au, av, hard, am, .05, chunk, log_r)
        actual = torch.autograd.grad((m, r), (au, av, am), grad_outputs=(cotangent / scale, r.new_tensor(scalar / scale)))
        ou, ov, om = u.clone().requires_grad_(), v.clone().requires_grad_(), material.clone().requires_grad_()
        dm, dr = dense(ou, ov, hard, om, log_r)
        expected = torch.autograd.grad(((dm * cotangent).sum() + scalar * dr) / scale, (ou, ov, om))
        torch.testing.assert_close(m, dm, atol=2e-14, rtol=2e-10)
        assert float(r) == pytest.approx(float(dr), abs=2e-14)
        for a, b in zip(actual, expected): torch.testing.assert_close(a, b, atol=2e-12, rtol=2e-10)
        projection = (actual[0] * d_u).sum() + (actual[1] * d_v).sum()
        for epsilon in EPSILONS:
            values = []
            for sign in (1, -1):
                fm, fr = dense(u + sign * epsilon * d_u, v + sign * epsilon * d_v, hard, material, log_r)
                values.append(((fm * cotangent).sum() + scalar * fr) / scale)
            assert_fd(projection, values[0], values[1], epsilon, record_property, f'joint_UV_chunk{chunk}_FD_{epsilon}')
    logits = (base(hard, 3, u.dtype) + u @ v.T / math.sqrt(2)).detach().requires_grad_()
    p, lp = logits.softmax(1), logits.log_softmax(1)
    loss = (p * (lp - log_r)).sum() / len(p)
    independent, = torch.autograd.grad(loss, logits)
    ell = lp.detach() - log_r
    formula = p.detach() * (ell - (p.detach() * ell).sum(1, keepdim=True)) / len(p)
    torch.testing.assert_close(independent, formula, atol=2e-12, rtol=2e-10)
    shifted = logits.detach() + torch.arange(6, dtype=torch.float64)[:, None] * .1
    shifted_loss = (shifted.softmax(1) * (shifted.log_softmax(1) - log_r)).sum() / 6
    assert float(shifted_loss) == pytest.approx(float(loss), abs=2e-14)
    order = torch.tensor([2, 0, 1]); permuted = logits.detach()[:, order]
    permuted_loss = (permuted.softmax(1) * (permuted.log_softmax(1) - log_r[:, order])).sum() / 6
    assert float(permuted_loss) == pytest.approx(float(loss), abs=2e-14)
    assert torch.equal(log_r, old_log_r) and not log_r.requires_grad
    record_property('graph_KL_row_cotangent_max_abs', float((independent - formula).abs().max()))


def solve_oracle(moments):
    centers, labels, _ = decode_moments(moments.detach(), 2)
    x = augmented(centers)
    theta = moments.new_zeros(3, 3).requires_grad_()
    def inner(parameter):
        return -(labels * (x @ parameter.T).log_softmax(1)).sum() / len(labels) + PENALTY * parameter.square().sum() / 2
    for _ in range(NEWTON_STEPS):
        gradient, = torch.autograd.grad(inner(theta), theta)
        hessian = torch.autograd.functional.hessian(inner, theta).reshape(9, 9)
        theta = (theta - torch.linalg.solve(hessian, gradient.flatten()).reshape_as(theta)).detach().requires_grad_()
    residual, = torch.autograd.grad(inner(theta), theta)
    assert float(residual.abs().max()) <= STATIONARITY
    return theta.detach()


def teacher_value(moments, z, q):
    theta = solve_oracle(moments)
    return -(q * (augmented(z) @ theta.T).log_softmax(1)).sum() / len(q)


def test_full_uniform_CE_implicit_head_coupling_with_fixed_graph_R_and_CE0_FD(record_property):
    z, q, hard, u, v, log_r = derivative_fixture()
    material = make_material(z, q)
    p0_moments, _ = dense(torch.zeros_like(u), v, hard, material, log_r)
    fixed_scale = max(float(teacher_value(p0_moments, z, q)), 1e-12)
    au, av = u.clone().requires_grad_(), v.clone().requires_grad_()
    moments, row_kl = InitialAssignmentKLMoments.apply(au, av, hard, material, .05, 3, log_r)
    theta = solve_oracle(moments)
    ce, rhs = outer_value_gradient(z, q, theta)
    centers, labels, mass = decode_moments(moments.detach(), 2)
    vector, diagnostic = solve_head_system(augmented(centers), labels, torch.full_like(mass, 1 / len(mass)),
        theta, PENALTY, rhs, rtol=1e-11, atol=1e-14)
    assert diagnostic['cg_converged']
    partial = implicit_moment_gradient(moments.detach(), 2, theta, vector, PENALTY, 'uniform')
    du, dv = torch.autograd.grad((moments, row_kl), (au, av), grad_outputs=(partial / fixed_scale, row_kl.new_tensor(1 / fixed_scale)))
    assert abs(ce - float(teacher_value(moments, z, q))) < 2e-14
    d_u, d_v = direction(u), direction(v)
    projection = (du * d_u).sum() + (dv * d_v).sum()
    old_log_r, old_scale = log_r.clone(), fixed_scale
    for epsilon in EPSILONS:
        values = []
        for sign in (1, -1):
            m, r = dense(u + sign * epsilon * d_u, v + sign * epsilon * d_v, hard, material, log_r)
            values.append((teacher_value(m, z, q) + r) / fixed_scale)
        assert_fd(projection, values[0], values[1], epsilon, record_property, f'full_CE_graphKL_headcoupling_FD_{epsilon}')
    assert torch.equal(log_r, old_log_r) and fixed_scale == old_scale
    record_property('fixed_original_teacher_CE0_scale', fixed_scale)
    record_property('head_oracle_stationarity_bound', STATIONARITY)
