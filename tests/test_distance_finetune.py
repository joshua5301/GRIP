import pytest
import torch

from src.distance_finetune import (
    distance_logits,
    evaluation_splits,
    factorized_distance,
    factorized_feature_distance,
    factorized_moment_cost,
    factorized_svd,
    initialization_assignment,
    svd_factors,
)
from src.low_rank_assignment import FactorizedBaseMoments, logit_block
from src.moment_seeding import normalized_variance_features
from src.moments import make_material
from src.soft_ce_partition import optimize_ce_assignment


def test_random_initialization_is_reproducible_nonempty_and_seeded():
    z = torch.zeros(31, 4)
    a = initialization_assignment(z, 7, "random", 0)
    assert torch.equal(a, initialization_assignment(z, 7, "random", 0))
    assert not torch.equal(a, initialization_assignment(z, 7, "random", 1))
    counts = torch.bincount(a)
    assert len(counts) == 7 and int(counts.max() - counts.min()) == 1


def test_feature_distance_matches_squared_euclidean_softmax():
    generator = torch.Generator().manual_seed(5)
    z = torch.randn(19, 4, generator=generator, dtype=torch.float64)
    assignment = torch.arange(19) % 3
    centers = torch.stack([z[assignment == j].mean(0) for j in range(3)])
    distance = (z[:, None] - centers[None]).square().sum(2)
    left, right, scale = factorized_feature_distance(z, assignment, chunk_size=5)
    torch.testing.assert_close(left @ right.T, -(distance - distance.mean(1, keepdim=True)) / scale)
    torch.testing.assert_close((left @ right.T).softmax(1), (-distance / scale).softmax(1))


def test_moment_cost_factors_match_direct_assignment_derivative():
    from src.moment_lloyd import assignment_cost
    generator = torch.Generator().manual_seed(17)
    z = torch.randn(15, 4, generator=generator, dtype=torch.float64)
    z -= z.mean(0)
    z /= z.square().sum(1).mean().sqrt()
    q = torch.randn(15, 3, generator=generator, dtype=z.dtype).softmax(1)
    assignment = torch.arange(15) % 3
    centers = torch.stack([z[assignment == j].mean(0) for j in range(3)])
    labels = torch.stack([q[assignment == j].mean(0) for j in range(3)])
    moment = z.T @ q / len(z) - centers.T @ labels / 3
    for weight in (0., 0.3, 3.):
        left, right, scale = factorized_moment_cost(z, q, assignment, weight, chunk_size=4)
        cost = assignment_cost(z, q, centers, labels, moment, weight)
        torch.testing.assert_close(left @ right.T, -(cost - cost.mean(1, keepdim=True)) / scale)
        torch.testing.assert_close((left @ right.T).softmax(1), (-cost / scale).softmax(1))


def test_inductive_evaluation_keeps_separate_graphs_and_full_split_masks():
    train_graph, val_graph, test_graph = {}, {}, {}
    train_mask = torch.tensor([True, False])
    splits = evaluation_splits(train_graph, train_mask, (val_graph, None), (test_graph, None))
    assert splits["train"][0] is train_graph
    assert splits["train"][1] is train_mask
    assert splits["val"][0] is val_graph and splits["val"][1] is None
    assert splits["test"][0] is test_graph and splits["test"][1] is None


def test_transductive_evaluation_uses_original_masks():
    graph = {}
    train, val, test = (torch.tensor([i == j for i in range(3)]) for j in range(3))
    masks = evaluation_splits(graph, train, (graph, val), (graph, test))
    assert masks["train"] is train
    assert masks["val"] is val
    assert masks["test"] is test


def test_svd_initialization_matches_low_rank_logit_scaling():
    generator = torch.Generator().manual_seed(4)
    logits = torch.randn(17, 5, generator=generator, dtype=torch.float64)
    logits -= logits.mean(1, keepdim=True)
    factors = svd_factors(logits, 5)
    recovered = logit_block(*factors, torch.zeros_like(logits), 0.05)
    torch.testing.assert_close(recovered, logits, atol=1e-6, rtol=1e-6)
    errors = []
    for rank in (1, 2, 4):
        u, v = svd_factors(logits, rank)
        errors.append(float((u.double() @ v.double().T / rank**0.5 - logits).norm()))
    assert errors[0] >= errors[1] >= errors[2]


def test_distance_initialization_matches_normalized_variance_metric():
    generator = torch.Generator().manual_seed(5)
    h = torch.randn(15, 4, generator=generator, dtype=torch.float64)
    q = torch.randn(15, 3, generator=generator, dtype=torch.float64).softmax(1)
    assignment = torch.arange(15) % 3
    logits, scale = distance_logits(h, q, assignment, 0.2)
    x = h - h.mean(0)
    x /= x.square().sum(1).mean().sqrt()
    z, _ = normalized_variance_features(x, q, 0.2)
    centers = torch.stack([z[assignment == j].mean(0) for j in range(3)])
    distance = torch.cdist(z, centers).square()
    expected = -(distance - distance.mean(1, keepdim=True)) / scale
    torch.testing.assert_close(logits, expected)
    torch.testing.assert_close(logits.softmax(1), (-distance / scale).softmax(1))
    assert scale > 0
    with pytest.raises(ValueError):
        svd_factors(logits, 4)


@pytest.mark.parametrize("factorized", [False, True])
def test_optimizer_uses_supplied_factors_at_step_zero(tmp_path, factorized):
    z = torch.tensor([[-1.0], [-0.5], [0.5], [1.0]], dtype=torch.float64)
    q = torch.tensor([[0.8, 0.2], [0.7, 0.3], [0.3, 0.7], [0.2, 0.8]], dtype=z.dtype)
    assignment = torch.tensor([0, 0, 1, 1])
    logits = torch.tensor([[1., -1.], [.5, -.5], [-.5, .5], [-1., 1.]], dtype=z.dtype)
    u, v = svd_factors(logits, 2)
    base = dict(base_factors=(z.new_zeros(4, 1), z.new_zeros(2, 1))) if factorized else dict(base_logits=torch.zeros_like(logits))
    result = optimize_ce_assignment(
        z, q, assignment, assignment_rank=2, initial_factors=(u, v),
        **base, steps=0, penalty=0.1,
        checkpoint_steps=[0], folder=tmp_path, save_assignment=False,
    )
    expected = logits.softmax(1).T @ make_material(z, q) / len(z)
    torch.testing.assert_close(result["checkpoints"][0]["moments"], expected, atol=1e-7, rtol=1e-7)


def test_factorized_distance_and_svd_equal_dense_construction():
    generator = torch.Generator().manual_seed(8)
    h = torch.randn(18, 4, generator=generator, dtype=torch.float64)
    q = torch.randn(18, 3, generator=generator, dtype=torch.float64).softmax(1)
    assignment = torch.arange(18) % 3
    expected, expected_scale = distance_logits(h, q, assignment, 0.2)
    left, right, scale = factorized_distance(h, q, assignment, 0.2, chunk_size=5)
    assert scale == pytest.approx(expected_scale)
    torch.testing.assert_close(left @ right.T, expected)
    a, s, b = factorized_svd(left, right)
    torch.testing.assert_close((a * s) @ b.T, expected)


def test_factorized_base_moments_gradient_matches_dense_autograd():
    generator = torch.Generator().manual_seed(9)
    left = torch.randn(7, 3, generator=generator, dtype=torch.float64)
    right = torch.randn(4, 3, generator=generator, dtype=torch.float64)
    material = torch.randn(7, 5, generator=generator, dtype=torch.float64, requires_grad=True)
    u = torch.randn(7, 2, generator=generator, dtype=torch.float64, requires_grad=True)
    v = torch.randn(4, 2, generator=generator, dtype=torch.float64, requires_grad=True)
    gradient = torch.randn(4, 5, generator=generator, dtype=torch.float64)
    actual = FactorizedBaseMoments.apply(u, v, left, right, material, 3)
    expected = (left @ right.T + u @ v.T / 2**0.5).softmax(1).T @ material / len(u)
    torch.testing.assert_close(actual, expected)
    actual_grads = torch.autograd.grad((actual * gradient).sum(), (u, v, material))
    expected_grads = torch.autograd.grad((expected * gradient).sum(), (u, v, material))
    for a, b in zip(actual_grads, expected_grads):
        torch.testing.assert_close(a, b)
