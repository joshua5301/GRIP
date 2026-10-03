"""New production domains: native fused precision and frozen resume links."""
import copy
import math

import pytest
import torch

from src.io import array_digest, cpu_state
from src.kernel_commutation_moments import JointKernelMoments, POLICY, attach_resume, attach_snapshot, validate_resume
from src.low_rank_assignment import LowRankMoments, logit_block


def test_three_material_native_FP32_single_cast_and_unchanged_base():
    generator = torch.Generator().manual_seed(5196)
    u = (.2 * torch.randn(9, 2, generator=generator)).requires_grad_()
    v = (.3 * torch.randn(4, 2, generator=generator)).requires_grad_()
    assignment = torch.arange(9) % 4
    material = torch.randn(9, 7, generator=generator, dtype=torch.float64)
    h = 3 * material[:, 1:4] + .17
    phi = torch.randn(9, 5, generator=generator, dtype=torch.float64)
    gb = torch.randn(4, 7, generator=generator, dtype=torch.float64)
    gh = torch.randn(4, 3, generator=generator, dtype=torch.float64)
    gp = torch.randn(4, 5, generator=generator, dtype=torch.float64)
    for chunk in (3, 7):
        base, hs, ps = JointKernelMoments.apply(u, v, assignment, material, .05, chunk, h, phi)
        old_base = LowRankMoments.apply(u, v, assignment, material, .05, chunk)
        assert torch.equal(base, old_base)
        du, dv = torch.autograd.grad((base, hs, ps), (u, v), (gb, gh, gp))
        expected_u, expected_v = torch.empty_like(u), torch.zeros_like(v)
        for first in range(0, len(u), chunk):
            last = first + chunk
            p = logit_block(u[first:last], v, assignment[first:last], .05).double().softmax(1)
            direction = material[first:last] @ gb.T / len(u)
            direction = direction + h[first:last] @ gh.T / len(u)
            direction = direction + phi[first:last] @ gp.T / len(u)
            tangent = (p * (direction - (p * direction).sum(1, keepdim=True))).float() / math.sqrt(2)
            expected_u[first:last] = tangent @ v
            expected_v += tangent.T @ u[first:last]
        assert torch.equal(du, expected_u) and torch.equal(dv, expected_v)


def test_resume_frozen_scales_and_initial_terminal_links():
    u0, v0 = torch.zeros(6, 2), torch.arange(6, dtype=torch.float32).reshape(3, 2) / 10
    parameters0, parameters1 = [u0, v0], [u0 + .01, v0 + .02]
    context = dict(schema=1, policy=POLICY, source_refs={},
        native_parameter_digests=[array_digest(v.numpy()) for v in parameters0],
        input_descriptors={'physical_H': dict(shape=[6, 4]), 'source_phi': dict(shape=[6, 5])})
    config = dict(kernel_commutation_mode=POLICY['mode'], kernel_commutation_context=context)
    snapshots, history = {}, []
    for step, parameters, CE, G in ((0, parameters0, .8, 1.25), (1, parameters1, .7, .4)):
        moments = torch.full((3, 7), .1 + step * .01, dtype=torch.float64)
        theta = torch.full((2, 5), .01 + step * .01, dtype=torch.float64)
        snapshot = dict(step=step, moments=moments, theta=theta, teacher_ce=CE, J_exact=True)
        attach_snapshot(snapshot, context, config, parameters,
            torch.full((3, 4), .2, dtype=torch.float64), torch.full((3, 5), .3, dtype=torch.float64),
            .8, 1.25, G, CE / .8 + G / 1.25)
        snapshots[step] = snapshot
        history.append(dict(step=step, J=CE, teacher_ce=CE, kernel_commutation_G=G,
            kernel_commutation_CE0=.8, kernel_commutation_G0=1.25,
            objective=CE / .8 + G / 1.25, normalized_objective=CE / .8 + G / 1.25,
            best_J=min(s['objective'] for s in snapshots.values())))
    saved = dict(config=config, step=1, parameters=cpu_state(parameters1), snapshots=snapshots,
        theta=snapshots[1]['theta'].clone(), initial_moments=snapshots[0]['moments'].clone(), scale=.8,
        history=history, best=snapshots[1]['objective'], best_step=1,
        best_moments=snapshots[1]['moments'].clone(), best_theta=snapshots[1]['theta'].clone())
    attach_resume(saved, context, parameters0, .8, 1.25)
    validate_resume(saved, context, config, 25, 6, 3, 2, 4, 2)
    for key, value in (('kernel_commutation_CE0', .9), ('kernel_commutation_G0', 0.0),
                       ('kernel_commutation_G0', 1.3), ('step', True), ('parameters', cpu_state(parameters0))):
        corrupted = copy.deepcopy(saved)
        corrupted[key] = value
        with pytest.raises(ValueError):
            validate_resume(corrupted, context, config, 25, 6, 3, 2, 4, 2)
    corrupted = copy.deepcopy(saved)
    corrupted['kernel_commutation_initial_H_moments'][0, 0] += .01
    with pytest.raises(ValueError):
        validate_resume(corrupted, context, config, 25, 6, 3, 2, 4, 2)
