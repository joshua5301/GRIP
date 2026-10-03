"""New CU input-routing domain; do not collect prior experimental proof suites."""
import copy
import pytest
import torch
import torch.nn.functional as F

from src.citation_training_outer import hard_training_outer, core_options, validate_resume
from src.io import array_digest
from src.soft_ce_partition import outer_value_gradient


def candidate():
    return dict(method='low_rank', width=0, lr=.05, T=.3, rank=2, penalty=.0001,
                initialization='teacher_balanced', alpha=.3, inner_loss_weighting='uniform')


def fixture():
    q = torch.tensor([[.8, .2], [.3, .7], [.4, .6], [.6, .4]], dtype=torch.float64)
    train = torch.tensor([True, False, True, False])
    labels = torch.tensor([1, -99999, 0, 99999])
    return q, labels, train


def test_hard_train_factory_cannot_read_or_infer_from_heldout_labels():
    q, labels, train = fixture()
    original = q.clone()
    mask, target = hard_training_outer(q.requires_grad_(), labels, train)
    changed = labels.clone(); changed[~train] = torch.tensor([2**60, -(2**60)])
    second_mask, second_target = hard_training_outer(q, changed, train)
    assert torch.equal(mask, train) and torch.equal(second_mask, mask)
    assert torch.equal(target, second_target) and torch.equal(target[~mask], q.detach()[~mask])
    assert torch.equal(target[mask], torch.tensor([[0., 1.], [1., 0.]], dtype=q.dtype))
    assert torch.equal(q.detach(), original) and not target.requires_grad
    mask[0] = False; target[0, 0] = 99
    assert torch.equal(train, torch.tensor([True, False, True, False])) and torch.equal(q.detach(), original)


def test_new_route_rejects_empty_training_and_unmatched_controls():
    q, labels, train = fixture()
    with pytest.raises(ValueError): hard_training_outer(q, labels, torch.zeros_like(train))
    with pytest.raises(ValueError): hard_training_outer(q, labels, train.long())
    bad = labels.clone(); bad[train] = 999
    with pytest.raises(ValueError): hard_training_outer(q, bad, train)
    mask, target = hard_training_outer(q, labels, train)
    for delta in [dict(rank=None), dict(method='mlp'), dict(inner_loss_weighting='mass'),
                  dict(train_target_mix=.5), dict(initialization='teacher_joint')]:
        with pytest.raises(ValueError): core_options(dict(candidate(), **delta), 0, mask, target)
    options = core_options(candidate(), 0, mask, target)
    assert options['outer_indices'] is mask and options['outer_targets'] is target
    assert options['inner_loss_weighting'] == 'uniform' and options['solver_mode'] == 'exact'
    assert options['save_assignment'] is False and options['mixing'] == .05


def test_training_only_outer_value_and_gradient_against_autograd_and_FD(record_property):
    q, labels, train = fixture()
    mask, target = hard_training_outer(q, labels, train)
    z = torch.tensor([[.2, -.7], [.5, .3], [-.4, .8], [2., -3.]], dtype=torch.float64)
    theta = torch.tensor([[.12, .35, -.4], [-.21, .5, .7]], dtype=torch.float64, requires_grad=True)
    x = torch.cat([z[mask], torch.ones(2, 1, dtype=z.dtype)], 1)
    expected_loss = F.cross_entropy(x @ theta.T, labels[train])
    expected_gradient = torch.autograd.grad(expected_loss, theta)[0]
    loss, gradient = outer_value_gradient(z[mask], target[mask], theta.detach(), chunk_size=1)
    assert abs(loss - float(expected_loss.detach())) < 1e-14
    torch.testing.assert_close(gradient, expected_gradient, atol=1e-14, rtol=1e-14)
    direction = torch.tensor([[.3, -.2, .4], [.7, .1, -.5]], dtype=theta.dtype)
    projected = float((gradient * direction).sum())
    errors = []
    for epsilon in (1e-5, 3e-6):
        plus = F.cross_entropy(x @ (theta.detach() + epsilon * direction).T, labels[train])
        minus = F.cross_entropy(x @ (theta.detach() - epsilon * direction).T, labels[train])
        error = abs(float((plus - minus) / (2 * epsilon)) - projected)
        errors.append(error); assert error < 1e-9
    record_property('FD_absolute_errors', errors)
    altered_z = z.clone(); altered_z[~mask] += 1e5
    same_loss, same_gradient = outer_value_gradient(altered_z[mask], target[mask], theta.detach())
    assert abs(same_loss - loss) < 1e-14
    torch.testing.assert_close(same_gradient, gradient, atol=1e-14, rtol=1e-14)


def test_resume_cannot_skip_changed_inner_Q_outer_target_or_train_mask():
    q, labels, train = fixture()
    mask, target = hard_training_outer(q, labels, train)
    z = torch.arange(8, dtype=torch.float64).reshape(4, 2)
    assignment = torch.tensor([0, 0, 1, 1])
    context = dict(data_digest=array_digest(z.numpy(), q.numpy(), assignment.numpy()),
                   outer_digest=array_digest(mask.numpy()), outer_targets_digest=array_digest(target.numpy()))
    state = dict(step=1, config=context)
    validate_resume(state, z, q, assignment, mask, target, 25)
    for key in context:
        altered = copy.deepcopy(state); altered['config'][key] = 'wrong'
        with pytest.raises(ValueError): validate_resume(altered, z, q, assignment, mask, target, 25)
    with pytest.raises(ValueError): validate_resume(state, z, q.flip(1), assignment, mask, target, 25)
    with pytest.raises(ValueError): validate_resume(state, z, q, assignment, ~mask, target, 25)
    with pytest.raises(ValueError): validate_resume(dict(state, step=26), z, q, assignment, mask, target, 25)
