"""New CPU integration proof for the separately activated frozen graph prior."""
import copy
import hashlib
import math

import pytest
import torch

from src.frozen_graph_assignment_prior import build_prior_packet, packet_context
from src.io import array_digest, save_state
from src.low_rank_assignment import initialize_factors, logit_block
from src.moments import make_material
from src.soft_ce_partition import optimize_ce_assignment, implicit_moment_gradient


def test_native_P0_combined_first_Adam_and_bound_resume(tmp_path, record_property):
    torch.set_num_threads(1)
    z = torch.tensor([[.3, -.4], [-.2, .25], [.6, .1], [-.3, .5],
                      [.7, -.15], [.15, .8], [-.4, -.2]], dtype=torch.float64)
    q = torch.tensor([[.7, .2, .1], [.6, .3, .1], [.8, .1, .1],
                      [.1, .8, .1], [.2, .6, .2], [.1, .1, .8], [.2, .2, .6]], dtype=torch.float64)
    a = torch.tensor([0, 0, 0, 1, 1, 2, 2], dtype=torch.int64)
    u0, v0 = initialize_factors(a, 3, 2, 19)
    p0 = logit_block(u0, v0, a, .05).double().softmax(1).detach()
    b = torch.eye(7, dtype=torch.float64)
    for first, last in ((0, 1), (0, 3), (1, 2), (2, 4), (3, 5)):
        b[first, last] = b[last, first] = 1
    inverse = b.sum(1).rsqrt()
    packed = (inverse[:, None] * b * inverse[None, :]).to_sparse_csr()
    refs = dict(data_digest=array_digest(z.numpy(), q.numpy(), a.numpy()),
                factor_seed=19, mixing=.05, chunk_size=7)
    prior = build_prior_packet(packed, p0, [u0.detach(), v0.detach()], refs)
    path = tmp_path / 'prior.pt'
    save_state(prior, path)
    pin = hashlib.sha256(path.read_bytes()).hexdigest()
    options = dict(penalty=.2, lr=.02, assignment_rank=2, factor_seed=19, mixing=.05,
        mass_mode='free', inner_loss_weighting='uniform', inner_method='newton_first',
        implicit_warm_start=True, chunk_size=7, save_resume=True, save_assignment=False, inner_max_iter=1000)
    active = dict(graph_assignment_kl_mode='binary_rw_diffused_native_P0_v1',
        graph_assignment_kl_artifact=str(path), graph_assignment_kl_sha256=pin,
        graph_assignment_kl_context=packet_context(prior))
    # These two tiny new fixtures prove unchanged source moments and head, and
    # do not execute any saved experiment or pre-existing numerical test.
    baseline = optimize_ce_assignment(z, q, a, steps=0, folder=tmp_path / 'base',
                                       checkpoint_steps=(0,), **options)
    optimize_ce_assignment(z, q, a, steps=1, folder=tmp_path / 'active',
                           checkpoint_steps=(0, 1), **options, **active)
    original = torch.load(tmp_path / 'base/checkpoints/step_000000.pt', weights_only=False)
    initial = torch.load(tmp_path / 'active/checkpoints/step_000000.pt', weights_only=False)
    state = torch.load(tmp_path / 'active/resume.pt', weights_only=False)
    assert torch.equal(initial['moments'], original['moments'])
    assert torch.equal(initial['theta'], original['theta'])
    assert initial['teacher_ce'] == original['teacher_ce']
    assert initial['graph_assignment_kl_KL'] > 0
    assert state['scale'] == max(original['teacher_ce'], 1e-12)
    assert initial['objective'] == initial['teacher_ce'] + initial['graph_assignment_kl_KL']
    assert not any(k.startswith('graph_assignment_kl_') for k in torch.load(
        tmp_path / 'base/resume.pt', weights_only=False)['config'])
    # Dense autograd cotangent wiring oracle; the separate math proof includes
    # independent convex-head finite differences rather than this core partial.
    u, v = u0.detach().clone().requires_grad_(), v0.detach().clone().requires_grad_()
    logits = logit_block(u, v, a, .05).double()
    p = logits.softmax(1)
    m = p.T @ make_material(z, q) / len(z)
    direction = implicit_moment_gradient(initial['moments'], 2, initial['theta'],
        state['tracking_vector_before'], .2, 'uniform')
    kl = (p * (logits.log_softmax(1) - prior['log_probability'])).sum() / len(z)
    gradients = torch.autograd.grad(((m * direction).sum() + kl) / state['scale'], (u, v))
    errors = []
    for native, gradient, actual in zip((u0.detach(), v0.detach()), gradients, state['parameters'], strict=True):
        expected = native - .02 * gradient / (gradient.abs() + 1e-12)
        errors.append(float((expected - actual).abs().max()))
        torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)
    assert torch.equal(gradients[1], torch.zeros_like(gradients[1]))
    record_property('native_initial_moments_head_CE_bit_exact', True)
    record_property('first_Adam_dense_cotangent_max_error', max(errors))
    record_property('initial_graph_KL', initial['graph_assignment_kl_KL'])
    # Reject changed contexts, native anchors, scale or terminal coefficients
    # before performing any resumed optimization.
    for field in ('initial_moments', 'scale', 'graph_assignment_kl_context', 'parameters', 'snapshots'):
        broken = copy.deepcopy(state)
        if field == 'initial_moments':
            broken[field][0, 0] += .01
        elif field == 'scale':
            broken[field] += .01
        elif field == 'graph_assignment_kl_context':
            broken[field]['source_refs']['factor_seed'] += 1
        elif field == 'parameters':
            broken[field][0][0, 0] += .01
        else:
            broken[field][0]['objective'] += .01
        with pytest.raises(ValueError):
            optimize_ce_assignment(z, q, a, steps=2, folder=tmp_path / 'active',
                checkpoint_steps=(0, 2), resume_state=broken, **options, **active)
    # One authorized continuation, same fixed artifact and initial CE scale.
    optimize_ce_assignment(z, q, a, steps=2, folder=tmp_path / 'active',
        checkpoint_steps=(0, 2), resume_state=state, **options, **active)
    terminal = torch.load(tmp_path / 'active/resume.pt', weights_only=False)
    assert terminal['step'] == 2 and terminal['scale'] == state['scale']
    assert terminal['graph_assignment_kl_context'] == state['graph_assignment_kl_context']
    assert torch.equal(prior['log_probability'], torch.load(path, weights_only=True)['log_probability'])
