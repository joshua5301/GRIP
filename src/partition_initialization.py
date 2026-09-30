"""Teacher-aware partitions without using original-node ground-truth labels."""

import math
from numbers import Integral

import torch

from src.initialization import feature_kmeans
from src.transforms import fit_transform


def _validate_inputs(h, q, alpha):
    if (
        h.ndim != 2
        or q.ndim != 2
        or h.shape[0] != q.shape[0]
        or min(*h.shape, q.shape[1]) < 1
        or not h.is_floating_point()
        or not q.is_floating_point()
        or not bool(torch.isfinite(h).all())
        or not bool(torch.isfinite(q).all())
    ):
        raise ValueError("Require finite, nonempty, node-aligned floating-point H and teacher probabilities")
    if (
        bool((q < 0).any())
        or not torch.allclose(q.sum(1), torch.ones_like(q[:, 0]), atol=1e-6, rtol=1e-5)
        or not math.isfinite(alpha)
        or alpha < 0
    ):
        raise ValueError("Teacher rows must be probabilities and alpha must be finite and nonnegative")


def _validate_budget(cells, nodes, seed):
    if isinstance(cells, bool) or not isinstance(cells, Integral) or not 1 <= cells <= nodes:
        raise ValueError("Cell budget must be an integer between one and the number of nodes")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed < 2**31:
        raise ValueError("Seed must be an integer in [0, 2**31)")


@torch.no_grad()
def teacher_representation(h, q, alpha=1.0, feature_weights=None):
    """Concatenate global RMS-normalized H and alpha times centered teacher Q.

    ``feature_weights`` optionally specifies a nonnegative diagonal distance
    metric. Multiplication by its square root precedes global RMS normalization.
    Q must come from the teacher; no ground-truth-label argument is accepted.
    The returned representation stays on H's device and uses double precision.
    """
    _validate_inputs(h, q, alpha)
    features = h.detach().double()
    if feature_weights is not None:
        weights = torch.as_tensor(feature_weights, device=h.device, dtype=torch.double)
        if (
            weights.shape != (h.shape[1],)
            or not bool(torch.isfinite(weights).all())
            or bool((weights < 0).any())
            or not bool((weights > 0).any())
        ):
            raise ValueError("Feature weights must be a finite nonnegative vector with at least one positive entry")
        features = features * weights.sqrt()
    z, _ = fit_transform(features, kind="rms")
    probabilities = q.detach().to(device=h.device, dtype=torch.double)
    return torch.cat((z, alpha * (probabilities - probabilities.mean(0))), dim=1)


@torch.no_grad()
def teacher_aware_kmeans(h, q, cells, seed=0, alpha=1.0, feature_weights=None):
    """Seeded joint feature/teacher k-means with exactly ``cells`` nonempty cells."""
    _validate_budget(cells, len(h), seed)
    representation = teacher_representation(h, q, alpha, feature_weights)
    return feature_kmeans(representation.cpu(), int(cells), int(seed)).to(h.device)


def allocate_balanced_cells(group_sizes, cells):
    """Allocate nearly equal cell quotas to nonempty groups, respecting capacity.

    Empty groups receive zero cells. Every nonempty group receives at least one;
    ties favor the smaller group index. A budget below the number of nonempty
    groups is rejected because it cannot preserve every inferred class.
    """
    sizes = torch.as_tensor(group_sizes)
    if (
        sizes.ndim != 1
        or not len(sizes)
        or sizes.dtype == torch.bool
        or sizes.is_floating_point()
        or sizes.is_complex()
        or bool((sizes < 0).any())
    ):
        raise ValueError("Group sizes must be a nonempty vector of nonnegative integers")
    sizes = sizes.to(device="cpu", dtype=torch.long)
    _validate_budget(cells, int(sizes.sum()), 0)
    nonempty = sizes > 0
    if cells < int(nonempty.sum()):
        raise ValueError("Cell budget must cover every nonempty teacher class")
    allocation = nonempty.long()
    for _ in range(int(cells) - int(allocation.sum())):
        eligible = allocation < sizes
        current = allocation.masked_fill(~eligible, torch.iinfo(torch.long).max)
        allocation[int(current.argmin())] += 1
    return allocation


@torch.no_grad()
def teacher_balanced_kmeans(h, q, cells, seed=0, alpha=1.0, feature_weights=None):
    """Run joint k-means separately inside teacher-argmax inferred classes.

    Class quotas are balanced rather than proportional to inferred class size;
    very small classes give unused quotas back to the other classes. Cell IDs
    are contiguous in ascending inferred-class order. Class labels from the
    dataset are never read. Assignment outputs stay on H's device.
    """
    _validate_budget(cells, len(h), seed)
    representation = teacher_representation(h, q, alpha, feature_weights).cpu()
    groups = q.detach().argmax(1).cpu()
    sizes = torch.bincount(groups, minlength=q.shape[1])
    allocation = allocate_balanced_cells(sizes, cells)
    assignment = torch.empty(len(h), dtype=torch.long)
    offset = 0
    for label, count in enumerate(allocation.tolist()):
        if count == 0:
            continue
        indices = (groups == label).nonzero().flatten()
        group_seed = (int(seed) + 104729 * label) % 2**31
        assignment[indices] = feature_kmeans(representation[indices], count, group_seed) + offset
        offset += count
    return assignment.to(h.device)
