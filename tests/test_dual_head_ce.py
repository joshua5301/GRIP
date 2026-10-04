"""Resume lineage guards only: no heads, P updates, datasets or seed draws."""
import copy
import math

import pytest
import torch

from src.dual_head_ce import (
    MODE, OBJECTIVE, POLICY, WORK_KEYS, _record_digest, validate_core_resume,
)
from src.io import array_digest


def _metadata_state():
    # A constructed uniform-target P0 metadata envelope, independent of the
    # numerical proof fixtures. No forward model or optimizer step is run.
    initial = [torch.zeros(4, 1), torch.ones(2, 1)]
    context = dict(schema=1, policy=copy.deepcopy(POLICY),
        source_refs=dict(nodes=4, cells=2, dimension=2, classes=2, rank=1,
            chunk_size=4, factor_seed=0, mixing=.05, device='cpu',
            data_digest='metadata-unit-fixture', original_options=dict(lr=.01, inner_tol=1e-7)),
        native_parameter_digests=[array_digest(v.numpy()) for v in initial],
        asset_descriptors=dict(H={}, transform={}, anchors={}, mapping={},
            Phi_identity=dict(schema=1, h_digest='a' * 64, map_digest='b' * 64,
                phi_digest='c' * 64, shape=[4, 2], dtype='float64')))
    config = dict(lr=.01, inner_tol=1e-7, data_digest='metadata-unit-fixture', dual_head_mode=MODE,
                  dual_head_context=copy.deepcopy(context))
    moments = torch.tensor([[.5, 0., 0., .25, .25], [.5, 0., 0., .25, .25]], dtype=torch.float64)
    ce0 = math.log(2)
    record = dict(step=0, moments=moments, theta_linear=torch.zeros(2, 3, dtype=torch.float64),
        theta_Nystrom=torch.zeros(2, 3, dtype=torch.float64), vector_linear=None,
        vector_Nystrom=None, vector_step=None, CE_linear=ce0, CE_Nystrom=ce0,
        CE_linear0=ce0, CE_Nystrom0=ce0, objective=2., objective_name=OBJECTIVE,
        J_exact=True, parameters=[v.clone() for v in initial],
        initial_parameters=[v.clone() for v in initial], config=copy.deepcopy(config),
        context=copy.deepcopy(context), head_work={m:dict(inner_converged=True, inner_grad_max=0.)
                                                 for m in ('linear', 'Nystrom')})
    record['record_digest'] = _record_digest(record)
    history = [dict(step=0, J_exact=True, objective_name=OBJECTIVE, J=2.,
                    objective=2., CE_linear=ce0, CE_Nystrom=ce0,
                    CE_linear0=ce0, CE_Nystrom0=ce0, linear_inner_converged=True,
                    Nystrom_inner_converged=True, linear_inner_grad_max=0., Nystrom_inner_grad_max=0.,
                    solver_mode='exact', implicit_warm_start=True, status='evaluated')]
    state = dict(step=0, config=copy.deepcopy(config), context=copy.deepcopy(context),
        parameters=[v.clone() for v in initial], initial_parameters=[v.clone() for v in initial],
        initial_moments=moments.clone(), moments=moments.clone(),
        theta_linear=record['theta_linear'].clone(), theta_Nystrom=record['theta_Nystrom'].clone(),
        vector_linear=None, vector_Nystrom=None, CE_linear0=ce0, CE_Nystrom0=ce0,
        snapshots={0:copy.deepcopy(record)}, current=copy.deepcopy(record), prefix_best=None,
        origin_digest=record['record_digest'], history=history, best=2., best_step=0,
        best_moments=moments.clone(), best_theta_linear=record['theta_linear'].clone(),
        best_theta_Nystrom=record['theta_Nystrom'].clone(),
        best_parameters=[v.clone() for v in initial], work={key:0 for key in WORK_KEYS},
        optimizer=dict(state={}, param_groups=[dict(params=[0,1], lr=.01, eps=1e-12,
            foreach=False, betas=(.9,.999), weight_decay=0., amsgrad=False,
            maximize=False, capturable=False, differentiable=False, fused=None)]))
    state['work'].update(moment_forward_calls=1, linear_head_interfaces=1,
                         Nystrom_head_interfaces=1, checkpoint_writes=1, resume_writes=1)
    return state, config, context


def test_valid_metadata_envelope_and_reject_cross_head_scale_origin_optimizer_corruption():
    state, config, context = _metadata_state()
    assert validate_core_resume(state, config, context) is state
    # Checkpoint zero is also saved before the first update of an invocation.
    # Its frontier row is 'update', with two evaluated current adjoints. This
    # differs from the completed endpoint envelope and must be resumable.
    frontier = copy.deepcopy(state)
    current = frontier['current']
    current['vector_linear'] = torch.zeros_like(current['theta_linear'])
    current['vector_Nystrom'] = torch.zeros_like(current['theta_Nystrom'])
    current['vector_step'] = 0
    current['record_digest'] = _record_digest(current)
    frontier['snapshots'][0] = copy.deepcopy(current)
    frontier['origin_digest'] = current['record_digest']
    for mode in ('linear', 'Nystrom'):
        frontier['vector_' + mode] = current['vector_' + mode].clone()
        frontier['history'][0][mode + '_evaluated'] = True
        frontier['history'][0][mode + '_cg_converged'] = True
        frontier['work'][mode + '_adjoint_solves'] = 1
    frontier['history'][0]['status'] = 'update'
    assert validate_core_resume(frontier, config, context) is frontier
    corruptions = []
    changed = copy.deepcopy(state); changed['CE_Nystrom0'] *= 1.01; corruptions.append(changed)
    changed = copy.deepcopy(state); changed['snapshots'][0]['theta_Nystrom'][0,0] = 1.; corruptions.append(changed)
    changed = copy.deepcopy(state); changed['current']['theta_linear'][0,0] = 1.; corruptions.append(changed)
    changed = copy.deepcopy(state); changed['history'][0]['CE_linear0'] *= 1.01; corruptions.append(changed)
    changed = copy.deepcopy(state); changed['initial_parameters'][1][0,0] = 2.; corruptions.append(changed)
    changed = copy.deepcopy(state); changed['parameters'][0][0,0] = 1.; corruptions.append(changed)
    changed = copy.deepcopy(state); changed['context']['asset_descriptors']['Phi_identity']['map_digest'] = 'd'*64; corruptions.append(changed)
    changed = copy.deepcopy(state); changed['optimizer']['param_groups'][0]['eps'] = 1e-8; corruptions.append(changed)
    changed = copy.deepcopy(state); changed['optimizer']['param_groups'][0]['capturable'] = True; corruptions.append(changed)
    changed = copy.deepcopy(state); del changed['snapshots'][0]; corruptions.append(changed)
    for changed in corruptions:
        with pytest.raises(ValueError):
            validate_core_resume(changed, config, context)
