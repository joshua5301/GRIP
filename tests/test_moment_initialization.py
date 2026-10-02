import pytest
import torch

from src.moment_initialization import distance_factors, optimize_initialization
from src.variance_moment_low_rank import low_rank_partition, moment_objective


def example():
    generator = torch.Generator().manual_seed(17)
    x = torch.randn(23, 5, generator=generator, dtype=torch.double)
    q = torch.randn(23, 3, generator=generator, dtype=torch.double).softmax(1)
    return x, q


def test_distance_logits_are_reproducible_and_temperature_scaled():
    x, _ = example()
    u, v = distance_factors(x[:, :3], 6, 4, 2)
    a, b = distance_factors(x[:, :3], 6, 4, 2, temperature=2)
    logits = u @ v.T / 2
    assert logits.mean(1).abs().max() < 1e-10
    assert logits.square().mean().sqrt() == pytest.approx(1.0)
    torch.testing.assert_close(a @ b.T / 2, logits / 2)
    c, d = distance_factors(x[:, :3], 6, 4, 2)
    torch.testing.assert_close(c @ d.T, u @ v.T)


def test_random_matches_existing_optimizer():
    x, q = example()
    options = dict(rank=4, seed=2, steps=9, lr=0.01)
    result = optimize_initialization(x, q, 6, 0.8, **options)
    reference = low_rank_partition(
        x, q, 6, None, None, initialization="random", moment_weight=0.8, backend="full", **options
    )
    assert result["J_final"] == pytest.approx(reference["J"], abs=1e-10)
    torch.testing.assert_close(result["x"], reference["x"])
    torch.testing.assert_close(result["y"], reference["y"])


def test_annealing_returns_original_objective_and_valid_barycenters():
    x, q = example()
    result = optimize_initialization(
        x,
        q,
        6,
        0.8,
        rank=4,
        seed=2,
        steps=10,
        initialization="distance",
        projected=x[:, :3],
        entropy_fraction=0.1,
    )
    rows = result["history"]
    assert rows[0]["beta"] > 0
    assert all(row["beta"] == 0 for row in rows[8:])
    assert result["J_final"] == min(row["J"] for row in rows)
    mass = result["counts"] / len(x)
    torch.testing.assert_close(mass @ result["x"].double(), x.mean(0), atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(mass @ result["y"].double(), q.mean(0), atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(result["y"].sum(1), torch.ones(6))
    assert bool((result["counts"] > 0).all())


def test_alternating_first_update_matches_u_only_adam():
    x, q = example()
    z = x - x.mean(0)
    z = z / z.square().sum(1).mean().sqrt()
    projected = z[:, :3]
    u, v = distance_factors(projected, 6, 4, 2)
    u = torch.nn.Parameter(u)
    optimizer = torch.optim.Adam([u], lr=0.01)

    def objective():
        p = (u @ v.T / 2).softmax(1)
        stats = p.mean(0), p.T @ z / len(z), p.T @ q / len(z)
        return moment_objective(stats, z.square().sum(1).mean(), z.T @ q / len(z), None, moment_weight=0.8)

    loss, initial, _, _ = objective()
    initial = float(initial.detach())
    loss.backward()
    optimizer.step()
    expected = min(initial, float(objective()[1].detach()))
    result = optimize_initialization(
        x,
        q,
        6,
        0.8,
        rank=4,
        seed=2,
        steps=1,
        initialization="distance",
        projected=projected,
        optimizer_mode="alternating",
        block_steps=2,
    )
    assert result["J_final"] == pytest.approx(expected, abs=1e-10)


@pytest.mark.parametrize("entropy", [0.0, 0.1])
def test_alternating_schedule_and_matched_initialization(entropy):
    x, q = example()
    options = dict(
        rank=4,
        seed=2,
        steps=12,
        initialization="distance",
        projected=x[:, :3],
        entropy_fraction=entropy,
        block_steps=3,
    )
    joint = optimize_initialization(x, q, 6, 0.8, **options)
    alternating = optimize_initialization(x, q, 6, 0.8, optimizer_mode="alternating", **options)
    assert joint["J_initial"] == alternating["J_initial"]
    assert [r["beta"] for r in joint["history"]] == [r["beta"] for r in alternating["history"]]
    assert [r["next_update"] for r in alternating["history"][:-1]] == ["U"] * 3 + ["V"] * 3 + ["U"] * 3 + [
        "V"
    ] * 3
    assert alternating["J_final"] <= alternating["J_initial"]


@pytest.mark.parametrize("momentum", [0.0, 0.9])
def test_joint_sgd_matches_explicit_updates(momentum):
    x, q = example()
    z = x - x.mean(0)
    z = z / z.square().sum(1).mean().sqrt()
    projected = z[:, :3]
    u, v = [torch.nn.Parameter(t) for t in distance_factors(projected, 6, 4, 2)]
    optimizer = torch.optim.SGD([u, v], lr=1.0, momentum=momentum)
    best = float("inf")
    for step in range(5):
        optimizer.zero_grad()
        p = (u @ v.T / 2).softmax(1)
        stats = p.mean(0), p.T @ z / len(z), p.T @ q / len(z)
        loss, exact, _, _ = moment_objective(
            stats, z.square().sum(1).mean(), z.T @ q / len(z), None, moment_weight=0.8
        )
        best = min(best, float(exact.detach()))
        if step < 4:
            loss.backward()
            optimizer.step()
    result = optimize_initialization(
        x,
        q,
        6,
        0.8,
        rank=4,
        seed=2,
        steps=4,
        lr=1.0,
        initialization="distance",
        projected=projected,
        optimizer_name="sgd",
        momentum=momentum,
    )
    assert result["J_final"] == pytest.approx(best, abs=1e-10)


def test_distance_partition_matches_comparison_and_chunked_backend(tmp_path):
    x, q = example()
    options = dict(rank=4, seed=2, steps=6, lr=0.01, initialization="distance")
    reference = optimize_initialization(x, q, 6, 0.8, **options)
    full = low_rank_partition(
        x, q, 6, None, None, moment_weight=0.8, backend="full", initialization_cache=tmp_path, **options
    )
    chunked = low_rank_partition(
        x,
        q,
        6,
        None,
        None,
        moment_weight=0.8,
        backend="chunked",
        block_size=7,
        initialization_cache=tmp_path,
        **options,
    )
    assert full["J"] == pytest.approx(reference["J_final"], abs=1e-10)
    assert chunked["J"] == pytest.approx(full["J"], abs=1e-10)
    torch.testing.assert_close(chunked["x"], full["x"])
    torch.testing.assert_close(chunked["y"], full["y"])
    assert (tmp_path / "rank_4_cells_6_seed_2.pt").exists()
