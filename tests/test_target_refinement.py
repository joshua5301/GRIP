import pytest
import torch

from src.target_refinement import training_refined_targets


def problem():
    logits = torch.tensor([[0.3, -0.5, 0.2], [0.2, 0.4, -0.1], [0.5, -0.4, 0.0], [0.2, 0.3, -0.2]])
    labels = torch.tensor([2, 999, 1, -999])
    mask = torch.tensor([True, False, True, False])
    return logits, labels, mask


@pytest.mark.parametrize("mixing", [0.0, 0.25, 1.0])
def test_only_training_targets_change_and_heldout_labels_are_ignored(mixing):
    logits, labels, mask = problem()
    baseline = (logits / 0.3).softmax(1).double()
    actual = training_refined_targets(logits, 0.3, labels, mask, mixing)
    expected = baseline.clone()
    anchors = torch.nn.functional.one_hot(labels[mask], num_classes=3).double()
    expected[mask] = (1 - mixing) * expected[mask] + mixing * anchors
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(actual[~mask], baseline[~mask], atol=0, rtol=0)
    torch.testing.assert_close(actual.sum(1), torch.ones(4, dtype=torch.double), atol=1e-7, rtol=0)
    changed = labels.clone()
    changed[~mask] = torch.tensor([-100000, 100000])
    torch.testing.assert_close(
        training_refined_targets(logits, 0.3, changed, mask, mixing), actual, atol=0, rtol=0
    )
    assert actual.dtype == torch.double


def test_zero_mixing_is_bitwise_legacy_compatible_and_does_not_inspect_any_label_value():
    logits, labels, mask = problem()
    labels.fill_(999)  # Even training labels are unused at zero mixing.
    expected = (logits / 0.3).softmax(1).double()
    assert torch.equal(training_refined_targets(logits, 0.3, labels, mask, 0.0), expected)


def test_empty_training_split_leaves_targets_unchanged():
    logits, labels, mask = problem()
    mask.zero_()
    labels.fill_(999)
    assert torch.equal(
        training_refined_targets(logits, 0.3, labels, mask, 1.0), (logits / 0.3).softmax(1).double()
    )


def test_refinement_preserves_inputs_and_teacher_logit_gradient():
    logits, labels, mask = problem()
    before = logits.clone()
    logits.requires_grad_()
    refined = training_refined_targets(logits, 0.3, labels, mask, 0.25)
    (refined[:, 0].sum()).backward()
    assert logits.grad is not None and bool(torch.isfinite(logits.grad).all())
    assert bool((logits.grad.abs().sum(1) > 0).all())
    assert torch.equal(logits.detach(), before)


@pytest.mark.parametrize("temperature", [0.0, -0.1, float("nan"), float("inf"), True])
def test_invalid_temperature_is_rejected(temperature):
    logits, labels, mask = problem()
    with pytest.raises(ValueError, match="temperature"):
        training_refined_targets(logits, temperature, labels, mask)


@pytest.mark.parametrize("mixing", [-0.1, 1.1, float("nan"), float("inf"), True])
def test_invalid_mixing_is_rejected(mixing):
    logits, labels, mask = problem()
    with pytest.raises(ValueError, match="mixing"):
        training_refined_targets(logits, 0.3, labels, mask, mixing)


@pytest.mark.parametrize(
    "invalid",
    [
        "logits_shape",
        "nonfinite",
        "mask_shape",
        "mask_dtype",
        "labels_shape",
        "labels_dtype",
        "train_label_range",
    ],
)
def test_invalid_inputs_are_rejected(invalid):
    logits, labels, mask = problem()
    if invalid == "logits_shape":
        logits = logits[:, 0]
    elif invalid == "nonfinite":
        logits[0, 0] = torch.inf
    elif invalid == "mask_shape":
        mask = mask[:3]
    elif invalid == "mask_dtype":
        mask = mask.long()
    elif invalid == "labels_shape":
        labels = labels[:3]
    elif invalid == "labels_dtype":
        labels = labels.float()
    else:
        labels[0] = 3
    with pytest.raises(ValueError):
        training_refined_targets(logits, 0.3, labels, mask, 0.25)
