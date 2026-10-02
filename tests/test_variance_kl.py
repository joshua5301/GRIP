import pytest
import torch

from src.io import array_digest
from src.variance_kl import variance_kl_partition
from src.variance_moment_sweep import _load_reference


@pytest.mark.parametrize("B", [0.1, 1.0, 10.0])
def test_lloyd_means_monotonicity_and_moment_upper_bound(B):
    generator = torch.Generator().manual_seed(17)
    h = torch.randn(31, 5, generator=generator, dtype=torch.float64)
    q = torch.randn(31, 3, generator=generator, dtype=torch.float64).softmax(1)
    _, modules = _load_reference()
    initialize = modules["risk_partition"]["seed_partition"]
    result = variance_kl_partition(h, q, 4, B, initialize, seed=2, block_size=7)
    a = result["assignment"]
    x = h - h.mean(0)
    x /= x.square().sum(1).mean().sqrt()
    initial = initialize(x, q, 4, B, torch.Generator().manual_seed(2), 7)
    assert result["initial_assignment_digest"] == array_digest(initial.numpy())
    for j in range(4):
        assert (a == j).any()
        torch.testing.assert_close(result["x"][j], h[a == j].mean(0).float())
        torch.testing.assert_close(result["y"][j], q[a == j].mean(0).float())
    assert all(b <= a + 1e-10 for a, b in zip(result["history"], result["history"][1:]))
    assert B * B / 4 * result["V"] + 2 * B * result["moment_error"] <= result["J"] + 1e-10
    assert result["converged"]


def test_empty_cell_repair_and_zero_probability_support():
    h = torch.tensor([[0.0], [0.0], [0.0], [0.0], [4.0], [4.0]], dtype=torch.float64)
    q = torch.tensor([[1.0, 0.0]] * 4 + [[0.0, 1.0]] * 2, dtype=torch.float64)

    def initialize(x, q, m, B, generator, block_size):
        return torch.tensor([0, 0, 1, 1, 2, 2])

    result = variance_kl_partition(h, q, 3, 1.0, initialize, block_size=2)
    assert (result["counts"] > 0).all()
    assert result["label_kl"] == pytest.approx(0.0)
    assert result["J"] == pytest.approx(0.0)


def test_empty_cell_is_reseeded_without_increasing_cost():
    h = torch.tensor([[-10.0], [10.0], [-10.0], [-10.0], [10.0], [10.0]], dtype=torch.float64)
    q = torch.ones(6, 1, dtype=torch.float64)

    def initialize(x, q, m, B, generator, block_size):
        return torch.tensor([0, 0, 1, 1, 2, 2])

    result = variance_kl_partition(h, q, 3, 1.0, initialize)
    assert (result["counts"] > 0).all()
    assert result["J"] <= result["history"][0]
    assert result["J"] == pytest.approx(0.0)
