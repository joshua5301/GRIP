import pandas as pd
import pytest
import torch

import src.soft_ce_partition as model
from src.feature_control_sweep import paired_grid
from src.soft_ridge_partition import AssignmentMoments, augmented, decode_moments, initial_logits, make_material


def problem():
    generator = torch.Generator().manual_seed(41)
    z = torch.randn(15, 3, generator=generator, dtype=torch.double)
    q = torch.randn(15, 3, generator=generator, dtype=torch.double).softmax(1)
    return z, q, torch.arange(15) % 4


def options():
    return dict(penalty=.2, lr=.01, chunk_size=4, inner_tol=1e-9, cg_rtol=1e-9,
                assignment_rank=2, assignment_input='features', assignment_encoder='mlp',
                encoder_hidden=6, save_assignment=False, solver_mode='tracking',
                tracking_inner_steps=1, tracking_cg_steps=2, tracking_refresh=2)


def test_controls_share_initial_state_and_freeze_targets_through_updates():
    z, q, assignment = problem()
    outputs = {control: model.optimize_ce_assignment(
        z, q, assignment, feature_control=control, steps=3, checkpoint_steps=[1, 2], **options())
        for control in ('joint', 'assignment', 'direct')}
    initial = outputs['joint']['initial_moments']
    initial_x, initial_y, initial_mass = decode_moments(initial, z.shape[1])
    for control in ('assignment', 'direct'):
        result = outputs[control]
        torch.testing.assert_close(result['initial_moments'], initial, atol=1e-14, rtol=1e-14)
        for snapshot in result['checkpoints'].values():
            _, labels, mass = decode_moments(snapshot['moments'], z.shape[1])
            torch.testing.assert_close(labels, initial_y, atol=1e-14, rtol=1e-14)
            torch.testing.assert_close(mass, initial_mass, atol=0, rtol=0)
        last = decode_moments(result['checkpoints'][3]['moments'], z.shape[1])[0]
        assert float((last - initial_x).norm()) > 1e-6
    assert outputs['direct']['assignment_parameters'] == initial_x.numel()


@pytest.mark.parametrize('control', ['assignment', 'direct'])
def test_fixed_target_hypergradient_matches_resolved_finite_difference(control):
    z, q, assignment = problem()
    material = make_material(z, q)
    logits = initial_logits(assignment, 4, .3, torch.double)
    initial = AssignmentMoments.apply(logits, material, 4)
    cx, labels, mass = decode_moments(initial, z.shape[1])
    variable = (logits if control == 'assignment' else cx).detach().requires_grad_()
    def moments(value):
        centers = decode_moments(AssignmentMoments.apply(value, material, 4), z.shape[1])[0] \
            if control == 'assignment' else value
        return model.fixed_target_moments(centers, labels, mass)
    def solve(value):
        current = moments(value)
        centers, _, _ = decode_moments(current.detach(), z.shape[1])
        fitted = model.solve_inner(centers, labels, mass, .2, grad_tol=1e-11)
        assert fitted['inner_converged']
        loss, gradient = model.outer_value_gradient(z, q, fitted['theta'])
        return current, centers, fitted['theta'], loss, gradient
    current, centers, theta, _, rhs = solve(variable)
    vector, diagnostic = model.solve_head_system(augmented(centers), labels, mass, theta, .2,
                                                  rhs, rtol=1e-11, atol=1e-14)
    assert diagnostic['cg_converged']
    direction = model.implicit_moment_gradient(current, z.shape[1], theta, vector, .2)
    gradient, = torch.autograd.grad(current, variable, grad_outputs=direction)
    tangent = torch.randn(variable.shape, generator=torch.Generator().manual_seed(17), dtype=z.dtype)
    tangent /= tangent.norm()
    epsilon = 1e-3
    expected = (solve(variable.detach() + epsilon * tangent)[3]
                - solve(variable.detach() - epsilon * tangent)[3]) / (2 * epsilon)
    torch.testing.assert_close((gradient * tangent).sum(), z.new_tensor(expected), atol=1e-7, rtol=3e-3)


@pytest.mark.parametrize('control', ['assignment', 'direct'])
def test_control_resume_preserves_parameters_and_tracking_state(tmp_path, control):
    z, q, assignment = problem()
    args = dict(**options(), feature_control=control, save_resume=True, checkpoint_steps=[2])
    full = model.optimize_ce_assignment(z, q, assignment, steps=4, **args)
    model.optimize_ce_assignment(z, q, assignment, steps=2, folder=tmp_path, **args)
    state = torch.load(tmp_path / 'resume.pt', weights_only=False)
    resumed = model.optimize_ce_assignment(z, q, assignment, steps=4, resume_state=state, **args)
    torch.testing.assert_close(full['checkpoints'][4]['moments'], resumed['checkpoints'][4]['moments'],
                               atol=2e-7, rtol=2e-6)
    with pytest.raises(ValueError, match='Resume state'):
        model.optimize_ce_assignment(z, q, assignment, steps=4, resume_state=state,
                                     **{**args, 'feature_control': 'joint'})


def test_paired_grid_matches_settings_and_reports_signed_deltas():
    row = dict(candidate=0, step=25, dropout=.9, gamma=.001, T=2., penalty=3e-5,
               assignment_lr=.003, val=80., val_std=.2, teacher_ce=.4)
    first = pd.DataFrame([row])
    second = pd.DataFrame([{**row, 'val': 82., 'teacher_ce': .3}])
    paired = paired_grid(first, second)
    assert paired.val_delta_C_minus_B.iloc[0] == 2.
    assert abs(paired.ce_delta_C_minus_B.iloc[0] + .1) < 1e-12
    with pytest.raises(ValueError, match='same evaluated configurations'):
        paired_grid(first, second.assign(T=4.))
