import pytest
import torch

from src.moment_alternating import compare_optimizer
from src.variance_moment_low_rank import low_rank_partition


def test_joint_matches_reference_and_alternating_accepts_descent():
    g = torch.Generator().manual_seed(4)
    x = torch.randn(17, 5, generator=g, dtype=torch.double)
    q = torch.randn(17, 3, generator=g, dtype=torch.double).softmax(1)
    options = dict(rank=4, seed=2, steps=12, lr=0.01)
    joint = compare_optimizer(x, q, 4, 0.8, method="joint", **options)
    alternating = compare_optimizer(x, q, 4, 0.8, method="alternating", block_steps=3, **options)
    reference = low_rank_partition(
        x, q, 4, None, None, initialization="random", moment_weight=0.8, backend="full", **options
    )
    assert joint["J_final"] == pytest.approx(reference["J"], abs=1e-10)
    assert joint["J_initial"] == alternating["J_initial"]
    assert alternating["J_final"] <= alternating["J_initial"]
    assert all(b["accepted_J"] <= b["initial_J"] + 1e-12 for b in alternating["blocks"])
    assert alternating["history"][-1]["step"] == 12
    assert {r["active"] for r in alternating["history"]} >= {"U", "V"}
