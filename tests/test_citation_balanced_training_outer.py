"""Two new CV domains; prior CU tests are not replayed."""
import pytest
import torch
import torch.nn.functional as F
from src.citation_training_outer import balanced_source_training_outer
from src.soft_ce_partition import outer_value_gradient


def inputs():
    q = torch.tensor([[.8, .2], [.3, .7], [.4, .6], [.6, .4], [.1, .9]], dtype=torch.float64)
    labels = torch.tensor([1, -99999, 0, 99999, -1234])
    train = torch.tensor([True, False, True, False, False])
    return q, labels, train


def test_balanced_rows_preserve_source_and_never_read_heldout_labels():
    q, labels, train = inputs()
    old_q, old_mask = q.clone(), train.clone()
    indices, target, repeats = balanced_source_training_outer(q.requires_grad_(), labels, train)
    assert repeats == 3 and len(indices) == 11 and indices.dtype == torch.long
    assert torch.equal(torch.bincount(indices, minlength=len(q)), torch.tensor([4, 1, 4, 1, 1]))
    changed = labels.clone(); changed[~train] = torch.tensor([2**60, -(2**60), 42])
    other_indices, other_target, other_repeats = balanced_source_training_outer(q, changed, train)
    assert torch.equal(target, other_target) and torch.equal(indices, other_indices) and repeats == other_repeats
    assert torch.equal(target[~train], q.detach()[~train]) and not target.requires_grad
    assert torch.equal(q.detach(), old_q) and torch.equal(train, old_mask)
    with pytest.raises(ValueError): balanced_source_training_outer(q, labels, torch.zeros_like(train))


def test_coalesced_outer_matches_independent_two_loss_value_gradient_FD(record_property):
    q, labels, train = inputs()
    indices, targets, repeats = balanced_source_training_outer(q, labels, train)
    z = torch.tensor([[.2, -.7], [.5, .3], [-.4, .8], [2., -3.], [-.2, -.5]], dtype=q.dtype)
    x = torch.cat((z, torch.ones(len(z), 1, dtype=z.dtype)), 1)
    theta = torch.tensor([[.12, .35, -.4], [-.21, .5, .7]], dtype=q.dtype, requires_grad=True)
    def reference(parameter):
        lp = (x @ parameter.T).log_softmax(1)
        soft_sum = -(q * lp).sum()
        hard_sum = F.nll_loss(lp[train], labels[train], reduction='sum')
        return (soft_sum + repeats * hard_sum) / (len(z) + repeats * int(train.sum()))
    loss = reference(theta); gradient = torch.autograd.grad(loss, theta)[0]
    actual, actual_gradient = outer_value_gradient(z[indices], targets[indices], theta.detach(), chunk_size=3)
    assert abs(actual - float(loss.detach())) < 2e-14
    torch.testing.assert_close(actual_gradient, gradient, atol=2e-14, rtol=2e-14)
    direction = torch.tensor([[.3, -.2, .4], [.7, .1, -.5]], dtype=q.dtype)
    projection = float((actual_gradient * direction).sum()); errors = []
    for epsilon in (1e-5, 3e-6):
        numeric = float((reference(theta.detach() + epsilon*direction) - reference(theta.detach() - epsilon*direction)) / (2*epsilon))
        errors.append(abs(numeric - projection)); assert errors[-1] < 1e-9
    record_property('balanced_CE_FD_absolute_errors', errors)
