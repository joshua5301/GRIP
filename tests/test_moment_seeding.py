import pytest
import torch

from src.moment_lloyd import initialize, moment_lloyd_partition
from src.moment_seeding import bound_features, minimum_scatter_split, pca_partition


def example():
    generator = torch.Generator().manual_seed(11)
    x = torch.randn(24, 5, generator=generator, dtype=torch.float64)
    q = torch.randn(24, 3, generator=generator, dtype=torch.float64).softmax(1)
    return x, q


def test_bound_matches_augmented_variance():
    x, q = example()
    z, info = bound_features(x, q, 3.0)
    assignment = torch.arange(len(x)) % 4
    cx = torch.stack([x[assignment == j].mean(0) for j in range(4)])
    cq = torch.stack([q[assignment == j].mean(0) for j in range(4)])
    cz = torch.stack([z[assignment == j].mean(0) for j in range(4)])
    rx, rq = x - cx[assignment], q - cq[assignment]
    objective = rx.square().sum(1).mean() + 3 * (rx.T @ rq / len(x)).norm()
    bound = (
        info["feature_weight"] * rx.square().sum(1).mean() + info["label_weight"] * rq.square().sum(1).mean()
    )
    torch.testing.assert_close(bound, (z - cz[assignment]).square().sum(1).mean())
    assert objective <= bound + 1e-12


@pytest.mark.parametrize("constant", [False, True])
@pytest.mark.parametrize("method", ["pca", "var", "pca_sse"])
def test_pca_repeatability_and_nonempty_cells(constant, method):
    x, _ = example()
    if constant:
        x.zero_()
    a, _ = pca_partition(x, 7, method=method)
    b, _ = pca_partition(x, 7, method=method)
    assert torch.equal(a, b)
    assert (torch.bincount(a, minlength=7) > 0).all()


def test_greedy_one_trial_matches_standard():
    x, _ = example()
    assert torch.equal(
        initialize(x, 4, 0, 8, iterations=1), initialize(x, 4, 0, 8, iterations=1, local_trials=1)
    )


@pytest.mark.parametrize("seeding", ["feature", "bound", "bound_greedy", "bound_pca", "bound_var", "bound_pca_sse"])
def test_seeding_preserves_objective_and_cell_means(seeding):
    x, q = example()
    result = moment_lloyd_partition(
        x, q, 4, moment_weight=3, mode="filtered_batch", initialization_steps=1, max_sweeps=5, seeding=seeding
    )
    assert all(b < a for a, b in zip(result["history"], result["history"][1:]))
    for j in range(4):
        mask = result["assignment"] == j
        assert mask.any()
        torch.testing.assert_close(result["x"][j], x[mask].mean(0).float())
        torch.testing.assert_close(result["y"][j], q[mask].mean(0).float())


def test_minimum_scatter_split_matches_brute_force():
    x, _ = example()
    x -= x.mean(0)
    projection = x[:, 0].round(decimals=1)
    left = minimum_scatter_split(x, projection)

    def cost(mask):
        return sum((part - part.mean(0)).square().sum() for part in (x[mask], x[~mask]))

    candidates = [projection <= value for value in projection.unique().sort().values[:-1]]
    expected = torch.stack([cost(mask) for mask in candidates]).min()
    torch.testing.assert_close(cost(left), expected)


def test_var_part_uses_largest_variance_axis():
    x = torch.tensor([[-10., 0.], [-2., 1.], [3., -1.], [5., 0.]], dtype=torch.float64)
    assignment, info = pca_partition(x, 2, method="var")
    assert torch.equal(assignment, torch.tensor([0, 0, 1, 1]))
    assert info["pca_splits"][0]["power_steps"] == 0
