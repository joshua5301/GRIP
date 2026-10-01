"""Training-label anchors for teacher targets without reading held-out labels."""

import math
from numbers import Real

import torch
import torch.nn.functional as F


def training_refined_targets(logits, temperature, labels, train_mask, mixing=0.0):
    """Blend teacher Q with one-hot ground truth on training nodes only.

    Q is exactly ``(logits / temperature).softmax(1).double()`` when mixing=0.
    Otherwise Q[train]=(1-mixing)*Q[train]+mixing*one_hot(labels[train]).
    Classes come exclusively from logits.shape[1]; excluded label values are
    never indexed, checked, or used to infer a class count.  The input tensors
    are not modified.  Gradients through teacher logits are retained.
    """
    if (
        not isinstance(logits, torch.Tensor)
        or logits.ndim != 2
        or min(logits.shape) < 1
        or not logits.is_floating_point()
        or not bool(torch.isfinite(logits).all())
    ):
        raise ValueError("logits must be a finite nonempty floating-point nodes-by-classes matrix")
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, Real)
        or not math.isfinite(temperature)
        or temperature <= 0
    ):
        raise ValueError("temperature must be positive and finite")
    if (
        isinstance(mixing, bool)
        or not isinstance(mixing, Real)
        or not math.isfinite(mixing)
        or not 0 <= mixing <= 1
    ):
        raise ValueError("mixing must be finite and lie in [0, 1]")
    if (
        not isinstance(train_mask, torch.Tensor)
        or train_mask.dtype != torch.bool
        or train_mask.shape != (len(logits),)
    ):
        raise ValueError("train_mask must be a boolean vector with one entry per node")
    if (
        not isinstance(labels, torch.Tensor)
        or labels.shape != (len(logits),)
        or labels.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
    ):
        raise ValueError("labels must be an integer vector with one entry per node")
    probability = (logits / temperature).softmax(1).double()
    if mixing == 0 or not bool(train_mask.any()):
        return probability
    # Only this masked selection may inspect label values. Different devices
    # are supported without copying the complete label vector onto the GPU.
    selected = labels[train_mask.to(labels.device)].to(device=logits.device, dtype=torch.long)
    if bool((selected < 0).any()) or bool((selected >= logits.shape[1]).any()):
        raise ValueError("Training labels must index the classes supplied by logits")
    mask = train_mask.to(logits.device)
    anchors = F.one_hot(selected, num_classes=logits.shape[1]).double()
    result = probability.clone()
    result[mask] = (1 - mixing) * probability[mask] + mixing * anchors
    return result
