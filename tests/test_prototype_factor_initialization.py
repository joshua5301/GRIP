"""Four new frozen-factor domains; no earlier suites are replayed."""
import copy
import hashlib
import math

import pytest
import torch

from src.io import array_digest
from src.low_rank_assignment import LowRankMoments
from src.moments import initial_logits, make_material
from src.prototype_factor_initialization import (
    MODES, build_factor_packet, generate_factor, load_frozen_factor, packet_context,
)


def fixture():
    centers = torch.tensor([[3., 0., 0.], [-3., 0., 0.], [0., 1., 0.], [0., -1., 0.]], dtype=torch.float64)
    mass = torch.tensor([.1, .2, .3, .4], dtype=torch.float64)
    original_v = torch.tensor([[.1, .5], [-.3, .2], [.9, -.8], [-1.1, .7]], dtype=torch.float32)
    return dict(mass=mass, features=mass[:, None] * centers), original_v, centers


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def test_distinct_spectrum_native_moment_scaling_and_normalized_control(record_property):
    moments, original_v, centers = fixture()
    old_mass, old_features, old_v = moments['mass'].clone(), moments['features'].clone(), original_v.clone()
    geometry, report = generate_factor(moments, original_v, MODES[0], 2)
    assert geometry.dtype == torch.float32 and geometry.device.type == 'cpu' and not geometry.requires_grad
    assert report['numerical_rank'] == 2
    torch.testing.assert_close(torch.tensor(report['eigenvalues'][:2]), torch.tensor([18., 2.]), atol=1e-6, rtol=0)
    assert report['rank_threshold'] == torch.finfo(torch.float64).eps * 4 * report['eigenvalues'][0]
    torch.testing.assert_close(geometry.double() @ geometry.double().T, .4 * (centers @ centers.T), atol=1e-6, rtol=1e-7)
    assert abs(float(geometry.double().norm()) - math.sqrt(8)) < 1e-7
    assert float(geometry.double().mean(0).abs().max()) < 1e-8
    assert bool((geometry[torch.tensor(report['sign_pivot_rows']), torch.arange(2)] > 0).all())
    for translated_scaled in (centers + torch.tensor([5., -2., .8]), centers * 3):
        changed = dict(mass=moments['mass'], features=moments['mass'][:, None] * translated_scaled)
        other, other_report = generate_factor(changed, original_v, MODES[0], 2)
        # A tied maximum can change which signed eigenvector pivot wins after
        # rounding. The reconstructed geometry is the affine-invariant object.
        torch.testing.assert_close(other.double() @ other.double().T,
                                   geometry.double() @ geometry.double().T, atol=1e-7, rtol=1e-7)
        assert bool((other[torch.tensor(other_report['sign_pivot_rows']), torch.arange(2)] > 0).all())
    gaussian, _ = generate_factor(moments, original_v, MODES[1], 2)
    centered = original_v.double() - original_v.double().mean(0)
    expected = (centered * (math.sqrt(8) / centered.norm())).float()
    assert torch.equal(gaussian, expected)
    assert abs(float(gaussian.double().norm()) - math.sqrt(8)) < 2e-7
    assert float(gaussian.double().mean(0).abs().max()) < 3e-8
    assert torch.equal(moments['mass'], old_mass) and torch.equal(moments['features'], old_features)
    assert torch.equal(original_v, old_v)
    record_property('geometry_actual_FP32_Frobenius_norm', report['actual_FP32_Frobenius_norm'])


def test_deficient_zero_nonfinite_rank_and_dtype_reject_without_fallback():
    moments, original_v, _ = fixture()
    for mode in MODES:
        with pytest.raises(ValueError, match='numerical rank'):
            generate_factor(moments, torch.cat((original_v, original_v[:, :1]), 1), mode, 3)
        zero = dict(mass=moments['mass'], features=moments['mass'][:, None].repeat(1, 3))
        with pytest.raises(ValueError, match='numerical rank'):
            generate_factor(zero, original_v, mode, 2)
        invalid = copy.deepcopy(moments); invalid['features'][0, 0] = float('nan')
        with pytest.raises(ValueError): generate_factor(invalid, original_v, mode, 2)
        invalid = copy.deepcopy(moments); invalid['mass'][0] = 0
        with pytest.raises(ValueError): generate_factor(invalid, original_v, mode, 2)
    with pytest.raises(ValueError): generate_factor(moments, original_v.double(), MODES[0], 2)
    with pytest.raises(ValueError): generate_factor(moments, original_v, MODES[0], True)
    with pytest.raises(ValueError): generate_factor(moments, original_v, 'unknown', 2)
    with pytest.raises(ValueError, match='no finite nonzero energy'):
        generate_factor(moments, torch.ones_like(original_v), MODES[1], 2)


def test_U0_moment_equality_independent_cotangent_and_first_Adam(record_property):
    moments, original_v, _ = fixture()
    factor, _ = generate_factor(moments, original_v, MODES[0], 2)
    assignment = torch.tensor([0, 1, 2, 3, 1])
    z = torch.tensor([[.3, -.7, .2], [.5, .3, -.1], [-.4, .8, .6], [2., -3., .7], [-.2, -.5, .8]], dtype=torch.float64)
    q = torch.tensor([[.7, .3], [.4, .6], [.2, .8], [.9, .1], [.5, .5]], dtype=torch.float64)
    material = make_material(z, q)
    cotangent = torch.tensor([[.1, -.2, .3, -.4, .5, .6], [.2, -.3, .4, -.5, .6, -.7],
                              [.3, .4, -.5, .6, -.7, .8], [-.4, .5, .6, -.7, .8, .9]], dtype=torch.float64)
    u = torch.zeros(5, 2, requires_grad=True); v = factor.clone().requires_grad_()
    native = LowRankMoments.apply(u, v, assignment, material, .05, 3)
    original = LowRankMoments.apply(torch.zeros_like(u), original_v, assignment, material, .05, 3)
    assert torch.equal(native, original)
    direct_logits = initial_logits(assignment, 4, .05) + u @ v.T / math.sqrt(2)
    probability = direct_logits.double().softmax(1)
    direct = probability.T @ material / len(u)
    du, dv = torch.autograd.grad((native * cotangent).sum(), (u, v), retain_graph=True)
    ref_du, ref_dv = torch.autograd.grad((direct * cotangent).sum(), (u, v))
    torch.testing.assert_close(native, direct, atol=2e-16, rtol=2e-15)
    torch.testing.assert_close(du, ref_du, atol=1e-8, rtol=2e-6)
    assert torch.equal(dv, torch.zeros_like(dv)) and torch.equal(ref_dv, torch.zeros_like(ref_dv))
    direction = material @ cotangent.T / len(u)
    g_logits = probability.detach() * (direction - (probability.detach() * direction).sum(1, keepdim=True))
    formula = (g_logits.float() / math.sqrt(2)) @ factor
    torch.testing.assert_close(du, formula, atol=1e-8, rtol=2e-6)
    optimizer = torch.optim.Adam([u, v], lr=.03, eps=1e-12, foreach=False)
    (native * cotangent).sum().backward(); optimizer.step()
    expected_u = -.03 * du / (du.abs() + 1e-12)
    torch.testing.assert_close(u, expected_u, atol=1e-8, rtol=2e-6)
    assert torch.equal(v.detach(), factor)
    record_property('independent_cotangent_absolute_error', float((du - ref_du).abs().max()))
    record_property('first_Adam_parameter_absolute_error', float((u.detach() - expected_u).abs().max()))


def test_frozen_artifact_context_mutation_and_new_core_P0_resume_integration(tmp_path, record_property):
    from src.soft_ce_partition import optimize_ce_assignment
    z = torch.tensor([[3., 0., .2], [3., 0., -.2], [-3., 0., .2], [-3., 0., -.2],
                      [0., 1., .2], [0., 1., -.2], [0., -1., .2], [0., -1., -.2]], dtype=torch.float64)
    q = torch.tensor([[.8, .2], [.7, .3], [.3, .7], [.2, .8], [.6, .4], [.5, .5], [.4, .6], [.3, .7]], dtype=torch.float64)
    assignment = torch.arange(4).repeat_interleave(2)
    options = dict(penalty=.1, steps=0, lr=.03, assignment_rank=2, factor_seed=3,
        assignment_input='node', assignment_encoder='linear', inner_loss_weighting='uniform',
        solver_mode='exact', inner_method='newton_first', implicit_warm_start=True,
        inner_max_iter=200, inner_tol=1e-10, cg_max_iter=200, cg_rtol=1e-8,
        save_resume=True, save_assignment=False, checkpoint_steps=(0,), log_every=1)
    original = optimize_ce_assignment(z, q, assignment, folder=tmp_path / 'original', **options)
    old_state = torch.load(tmp_path / 'original' / 'resume.pt', weights_only=False)
    assert not any(key.startswith('node_factor_') for key in old_state['config'])
    m0 = original['checkpoints'][0]['moments']
    initial = dict(mass=m0[:, 0].clone(), features=m0[:, 1:4].clone())
    refs = dict(data_digest=array_digest(z.numpy(), q.numpy(), assignment.numpy()), mixing=.05,
                factor_seed=3, pinned_native_fixture='new_CPU_only', original_snapshot_SHA256='f' * 64)
    packet = build_factor_packet(initial, old_state['parameters'][1].detach().cpu(), MODES[0], 2, refs)
    artifact = tmp_path / 'factor.pt'; torch.save(packet, artifact)
    digest, context, dimensions = sha(artifact), packet_context(packet), packet['dimensions']
    assert set(context) == {'schema', 'mode', 'rank', 'dimensions', 'source_refs', 'context_digest', 'factor_digest'}
    loaded = load_frozen_factor(artifact, digest, MODES[0], dimensions, 'cpu', expected_context=context)
    assert torch.equal(loaded, packet['factor']) and loaded.data_ptr() != packet['factor'].data_ptr()
    for mode, dims, expected_context in ((MODES[1], dimensions, context),
        (MODES[0], dict(cells=5, feature_dimension=3), context),
        (MODES[0], dimensions, dict(context, rank=1)),
        (MODES[0], dimensions, dict(context, rank=True))):
        with pytest.raises(ValueError):
            load_frozen_factor(artifact, digest, mode, dims, 'cpu', expected_context=expected_context)
    modified = copy.deepcopy(packet); modified['factor'][0, 0] += 1
    altered = tmp_path / 'modified.pt'; torch.save(modified, altered)
    with pytest.raises(ValueError, match='artifact bytes'):
        load_frozen_factor(altered, digest, MODES[0], dimensions, 'cpu', expected_context=context)
    with pytest.raises(ValueError, match='tensor changed'):
        load_frozen_factor(altered, sha(altered), MODES[0], dimensions, 'cpu', expected_context=context)
    modified = copy.deepcopy(packet); modified['diagnostics']['numerical_rank'] += 1; torch.save(modified, altered)
    with pytest.raises(ValueError, match='metadata changed'):
        load_frozen_factor(altered, sha(altered), MODES[0], dimensions, 'cpu', expected_context=context)
    activated = dict(node_factor_mode=MODES[0], node_factor_artifact=str(artifact),
                     node_factor_sha256=digest, node_factor_context=context)
    current = optimize_ce_assignment(z, q, assignment, folder=tmp_path / 'new', **options, **activated)
    snapshot = current['checkpoints'][0]
    assert torch.equal(snapshot['moments'], original['checkpoints'][0]['moments'])
    assert torch.equal(snapshot['theta'], original['checkpoints'][0]['theta'])
    assert torch.equal(snapshot['node_factor_parameters'][0], torch.zeros(len(z), 2))
    assert torch.equal(snapshot['node_factor_parameters'][1], packet['factor'])
    state = torch.load(tmp_path / 'new' / 'resume.pt', weights_only=False)
    assert state['node_factor_context'] == context and state['config']['node_factor_context'] == context
    wrong = copy.deepcopy(state); wrong['node_factor_context']['source_refs']['factor_seed'] = 99
    with pytest.raises(ValueError, match='initializer context'):
        optimize_ce_assignment(z, q, assignment, folder=tmp_path / 'resume_rejected', resume_state=wrong, **options, **activated)
    wrong = copy.deepcopy(state); wrong['snapshots'][0]['node_factor_context']['rank'] = 1
    with pytest.raises(ValueError, match='checkpoint context'):
        optimize_ce_assignment(z, q, assignment, folder=tmp_path / 'checkpoint_rejected', resume_state=wrong, **options, **activated)
    record_property('CPU_new_core_P0_moments_bit_exact', True)
    record_property('CPU_new_core_P0_head_theta_bit_exact', True)
