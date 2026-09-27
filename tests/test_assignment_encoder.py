import pytest
import torch

from src.low_rank_assignment import (LowRankMoments, assignment_inputs,
                                     encode_nodes, initialize_encoder, initialize_mlp,
                                     logit_block, saved_encoder_nodes)
from src.soft_ce_partition import optimize_ce_assignment
from src.soft_ridge_partition import initial_logits, make_material


def problem():
    generator = torch.Generator().manual_seed(13)
    z = torch.randn(12, 3, generator=generator, dtype=torch.double)
    q = torch.randn(12, 2, generator=generator, dtype=torch.double).softmax(1)
    return z, q, torch.arange(12) % 4


@pytest.mark.parametrize('mode', ['features', 'features_labels'])
def test_encoder_initialization_and_gradient(mode):
    z, q, assignment = problem()
    inputs = assignment_inputs(z.requires_grad_(), q.requires_grad_(), mode)
    weight, v = initialize_encoder(inputs, 4, 2, 0)
    assert not inputs.requires_grad
    material = make_material(z.detach(), q.detach())
    initial = LowRankMoments.apply(inputs @ weight, v, assignment, material, .05, 5)
    expected = initial_logits(assignment, 4, .05).double().softmax(1).T @ material / len(z)
    torch.testing.assert_close(initial, expected)
    dw, dv = torch.autograd.grad(initial.square().sum(), (weight, v))
    assert dw.norm() > 0
    torch.testing.assert_close(dv, torch.zeros_like(v))
    if mode == 'features_labels':
        torch.testing.assert_close(inputs[:, 3:], q.detach().float())
        assert dw[3:].norm() > 0


@pytest.mark.parametrize('mode', ['features', 'features_labels'])
def test_encoder_derivative_matches_dense_and_finite_difference(mode):
    z, q, assignment = problem()
    inputs = assignment_inputs(z, q, mode).double()
    weight, v = initialize_encoder(inputs, 4, 2, 5)
    with torch.no_grad():
        weight.copy_(torch.linspace(-.2, .2, weight.numel()).reshape_as(weight))
    material = make_material(z, q)
    def moments(w, centers):
        return LowRankMoments.apply(inputs @ w, centers, assignment, material, .05, 5)
    actual = moments(weight, v)
    logits = initial_logits(assignment, 4, .05, torch.double) + inputs @ weight @ v.T / 2 ** .5
    expected = logits.softmax(1).T @ material / len(z)
    actual_grad = torch.autograd.grad(actual.square().sum(), (weight, v))
    expected_grad = torch.autograd.grad(expected.square().sum(), (weight, v))
    for a, b in zip(actual_grad, expected_grad):
        torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-10)
    assert torch.autograd.gradcheck(moments, (weight, v))


@pytest.mark.parametrize('mode', ['features', 'features_labels'])
@pytest.mark.parametrize('mass_mode', ['free', 'uniform'])
def test_encoder_bilevel_checkpoint_roundtrip(tmp_path, mode, mass_mode):
    z, q, assignment = problem()
    result = optimize_ce_assignment(
        z, q, assignment, penalty=.2, steps=2, lr=.01, chunk_size=5,
        inner_tol=1e-9, cg_rtol=1e-9, folder=tmp_path, checkpoint_steps=[1],
        assignment_rank=2, assignment_input=mode, mass_mode=mass_mode,
    )
    saved = torch.load(tmp_path / 'best_assignment_encoder.pt', weights_only=True)
    inputs = assignment_inputs(z, q, saved['assignment_input'])
    logits = logit_block(inputs @ saved['weight'], saved['v'], saved['assignment'], saved['mixing']).double()
    if mass_mode == 'uniform':
        logits += torch.load(tmp_path / 'best_assignment_column_dual.pt', weights_only=True)
    moments = logits.softmax(1).T @ make_material(z, q) / len(z)
    torch.testing.assert_close(moments, result['best_moments'], atol=1e-8, rtol=1e-7)
    assert result['assignment_parameters'] == (inputs.shape[1] + 4) * 2
    assert set(result['checkpoints']) == {0, 1, 2}
    assert all(row['inner_converged'] for row in result['history'])
    assert result['best_J'] <= result['history'][0]['J']
    assert not (tmp_path / 'best_assignment_factors.pt').exists()


def test_input_modes_share_random_centers():
    z, q, _ = problem()
    _, a = initialize_encoder(assignment_inputs(z, q, 'features'), 4, 2, 7)
    _, b = initialize_encoder(assignment_inputs(z, q, 'features_labels'), 4, 2, 7)
    torch.testing.assert_close(a, b, atol=0, rtol=0)


@pytest.mark.parametrize('mode', ['features', 'features_labels'])
def test_mlp_initialization_and_live_output_layer(mode):
    z, q, assignment = problem()
    inputs = assignment_inputs(z, q, mode)
    state = torch.random.get_rng_state().clone()
    parameters, v = initialize_mlp(inputs, 4, 2, 8, 7)
    torch.testing.assert_close(torch.random.get_rng_state(), state, atol=0, rtol=0)
    _, linear_v = initialize_encoder(inputs, 4, 2, 7)
    torch.testing.assert_close(v, linear_v, atol=0, rtol=0)
    u = encode_nodes(inputs, parameters)
    torch.testing.assert_close(u, torch.zeros_like(u), atol=0, rtol=0)
    material = make_material(z, q)
    moments = LowRankMoments.apply(u, v, assignment, material, .05, 5)
    expected = initial_logits(assignment, 4, .05).double().softmax(1).T @ material / len(z)
    torch.testing.assert_close(moments, expected)
    gradients = torch.autograd.grad(moments.square().sum(), [*parameters, v])
    for index in (0, 1, 4):
        torch.testing.assert_close(gradients[index], torch.zeros_like(gradients[index]), atol=0, rtol=0)
    assert gradients[2].norm() > 0
    with torch.no_grad():
        parameters[2].sub_(gradients[2], alpha=.1)
    updated = LowRankMoments.apply(encode_nodes(inputs, parameters), v, assignment, material, .05, 5)
    gradients = torch.autograd.grad(updated.square().sum(), [*parameters, v])
    assert gradients[0].norm() > 0
    assert gradients[-1].norm() > 0


def test_mlp_moment_gradients_match_dense_autograd():
    z, q, assignment = problem()
    inputs = assignment_inputs(z, q, 'features_labels').double()
    parameters, v = initialize_mlp(inputs, 4, 2, 8, 7)
    with torch.no_grad():
        parameters[2].copy_(torch.linspace(-.2, .3, 16).reshape(8, 2))
    material = make_material(z, q)
    actual = LowRankMoments.apply(encode_nodes(inputs, parameters), v, assignment, material, .05, 5)
    logits = logit_block(encode_nodes(inputs, parameters), v, assignment, .05)
    expected = logits.softmax(1).T @ material / len(z)
    actual_grad = torch.autograd.grad(actual.square().sum(), [*parameters, v])
    expected_grad = torch.autograd.grad(expected.square().sum(), [*parameters, v])
    for a, b in zip(actual_grad, expected_grad):
        torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-10)


@pytest.mark.parametrize('mode', ['features', 'features_labels'])
@pytest.mark.parametrize('mass_mode', ['free', 'uniform'])
def test_mlp_bilevel_saved_encoder_roundtrip(tmp_path, mode, mass_mode):
    z, q, assignment = problem()
    result = optimize_ce_assignment(
        z, q, assignment, penalty=.2, steps=2, lr=.01, chunk_size=5,
        inner_tol=1e-9, cg_rtol=1e-9, folder=tmp_path, checkpoint_steps=[1],
        assignment_rank=2, assignment_input=mode, mass_mode=mass_mode,
        assignment_encoder='mlp', encoder_hidden=8,
    )
    saved = torch.load(tmp_path / 'best_assignment_encoder.pt', weights_only=True)
    logits = logit_block(saved_encoder_nodes(z, q, saved), saved['v'], assignment, saved['mixing']).double()
    if mass_mode == 'uniform':
        logits += torch.load(tmp_path / 'best_assignment_column_dual.pt', weights_only=True)
    moments = logits.softmax(1).T @ make_material(z, q) / len(z)
    torch.testing.assert_close(moments, result['best_moments'], atol=1e-8, rtol=1e-7)
    dimension = 3 if mode == 'features' else 5
    assert result['assignment_parameters'] == dimension * 8 + 8 + 8 * 2 + 2 + 4 * 2
    assert set(result['checkpoints']) == {0, 1, 2}
    assert all(row['inner_converged'] for row in result['history'])
