import pytest
import torch

from src.partition_initialization import (
    allocate_balanced_cells,
    teacher_aware_kmeans,
    teacher_balanced_kmeans,
    teacher_representation,
)


def teacher_problem():
    generator = torch.Generator().manual_seed(23)
    h = torch.randn(24, 4, generator=generator, dtype=torch.double)
    q = torch.randn(24, 3, generator=generator, dtype=torch.double).softmax(1)
    return h, q


def test_joint_distance_is_invariant_to_feature_translation_and_scale():
    h, q = teacher_problem()
    representation = teacher_representation(h, q, alpha=2.0)
    transformed = teacher_representation(3.0 * h + torch.arange(4), q, alpha=2.0)
    torch.testing.assert_close(torch.cdist(representation, representation), torch.cdist(transformed, transformed))
    assert not representation.requires_grad
    assert representation.shape == (24, 7)
    torch.testing.assert_close(representation.mean(0), torch.zeros(7, dtype=torch.double), atol=1e-15, rtol=0)


def test_diagonal_metric_removes_a_feature_from_partition_distances():
    h, q = teacher_problem()
    weights = torch.tensor([1.0, 0.0, 2.0, 1.0])
    expected = teacher_representation(h, q, feature_weights=weights)
    changed = h.clone()
    changed[:, 1] = torch.arange(len(h)) * 100.0
    actual = teacher_representation(changed, q, feature_weights=weights)
    torch.testing.assert_close(actual, expected)
    assert torch.count_nonzero(actual[:, 1]) == 0


@pytest.mark.parametrize("initializer", [teacher_aware_kmeans, teacher_balanced_kmeans])
def test_partitions_are_seeded_nonempty_and_do_not_mutate_inputs_or_rng(initializer):
    h, q = teacher_problem()
    initial_h, initial_q = h.clone(), q.clone()
    rng = torch.random.get_rng_state().clone()
    first = initializer(h, q, cells=6, seed=7, alpha=1.0)
    second = initializer(h, q, cells=6, seed=7, alpha=1.0)
    torch.testing.assert_close(first, second, atol=0, rtol=0)
    assert first.dtype == torch.long
    assert first.device == h.device
    assert torch.bincount(first, minlength=6).min() > 0
    assert first.unique().tolist() == list(range(6))
    torch.testing.assert_close(h, initial_h, atol=0, rtol=0)
    torch.testing.assert_close(q, initial_q, atol=0, rtol=0)
    torch.testing.assert_close(torch.random.get_rng_state(), rng, atol=0, rtol=0)


def test_capacity_redistribution_and_empty_teacher_classes():
    allocation = allocate_balanced_cells([1, 0, 2, 20, 50], cells=12)
    assert allocation.tolist() == [1, 0, 2, 5, 4]
    assert allocate_balanced_cells([1, 0, 2, 20, 50], cells=73).tolist() == [1, 0, 2, 20, 50]


def test_balanced_partition_preserves_inferred_classes_with_identical_features():
    groups = torch.tensor([0, 2, 2] + [3] * 9)
    q = torch.nn.functional.one_hot(groups, 4).double()
    h = torch.zeros(12, 2, dtype=torch.double)
    assignment = teacher_balanced_kmeans(h, q, cells=6, seed=0)
    assert assignment.unique().tolist() == list(range(6))
    assert assignment[groups == 0].unique().tolist() == [0]
    assert assignment[groups == 2].unique().tolist() == [1, 2]
    assert assignment[groups == 3].unique().tolist() == [3, 4, 5]
    for cell in range(6):
        assert groups[assignment == cell].unique().numel() == 1


@pytest.mark.parametrize("initializer", [teacher_aware_kmeans, teacher_balanced_kmeans])
def test_each_node_can_form_its_own_cell_even_with_duplicate_representations(initializer):
    h = torch.zeros(6, 2)
    q = torch.ones(6, 2) / 2
    assignment = initializer(h, q, cells=6)
    assert sorted(assignment.tolist()) == list(range(6))


@pytest.mark.parametrize("sizes,cells", [([0, 0], 1), ([2, 3], 1), ([2, 3], 6), ([-1, 4], 2), ([1.5, 3], 2)])
def test_invalid_group_budgets_are_rejected(sizes, cells):
    with pytest.raises(ValueError):
        allocate_balanced_cells(sizes, cells)


@pytest.mark.parametrize("field", ["probabilities", "alpha", "weights", "cells", "seed"])
def test_invalid_partition_inputs_are_rejected(field):
    h, q = teacher_problem()
    arguments = dict(cells=6, seed=0, alpha=1.0)
    if field == "probabilities":
        q[0, 0] = -1.0
    elif field == "alpha":
        arguments["alpha"] = -1.0
    elif field == "weights":
        arguments["feature_weights"] = [1.0, -1.0, 1.0, 1.0]
    elif field == "cells":
        arguments["cells"] = 25
    else:
        arguments["seed"] = -1
    with pytest.raises(ValueError):
        teacher_aware_kmeans(h, q, **arguments)
