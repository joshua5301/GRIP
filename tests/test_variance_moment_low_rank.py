import pytest
import torch

from src.variance_moment_low_rank import low_rank_partition, moment_objective


def initialize(x, q, m, B, generator, block_size):
    return torch.arange(len(x), device=x.device) % m


def test_soft_statistics_equal_explicit_pairwise_objective():
    g = torch.Generator().manual_seed(8)
    x = torch.randn(13, 4, generator=g, dtype=torch.double)
    q = torch.randn(13, 3, generator=g, dtype=torch.double).softmax(1)
    logits = torch.randn(13, 3, generator=g, dtype=torch.double, requires_grad=True)
    p = logits.softmax(1)
    mass, sx, sq = p.mean(0), p.T @ x / len(x), p.T @ q / len(x)
    _, exact, variance, _ = moment_objective((mass, sx, sq), x.square().sum(1).mean(), x.T @ q / len(x), 2.0)
    centers, labels = sx / mass[:, None], sq / mass[:, None]
    explicit_v = (p * (x[:, None] - centers[None]).square().sum(2)).sum() / len(x)
    explicit_m = x.T @ q / len(x) - centers.T @ (mass[:, None] * labels)
    explicit = explicit_v + 4 * explicit_m.norm()
    torch.testing.assert_close(variance, explicit_v)
    torch.testing.assert_close(exact, explicit)
    a = torch.autograd.grad(exact, logits, retain_graph=True)[0]
    b = torch.autograd.grad(explicit, logits)[0]
    torch.testing.assert_close(a, b)


def test_chunked_optimizer_and_initial_mass_conservation():
    g = torch.Generator().manual_seed(5)
    h = torch.randn(15, 4, generator=g, dtype=torch.double)
    q = torch.randn(15, 3, generator=g, dtype=torch.double).softmax(1)
    outputs = [low_rank_partition(h, q, 3, 1.0, initialize, steps=4, block_size=b) for b in (4, 15)]
    for result in outputs:
        assert result["J"] <= result["history"][0]
        assert result["J"] == min(result["history"])
        mass = result["counts"].double() / len(h)
        torch.testing.assert_close(mass.sum(), torch.tensor(1.0, dtype=torch.double))
        torch.testing.assert_close(mass @ result["x"].double(), h.mean(0), atol=1e-7, rtol=1e-6)
        torch.testing.assert_close(mass @ result["y"].double(), q.mean(0), atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(outputs[0]["x"], outputs[1]["x"], atol=1e-6, rtol=1e-6)
    assert abs(outputs[0]["J"] - outputs[1]["J"]) < 1e-9


@pytest.mark.parametrize("rank", [4, 8, 16])
def test_random_initialization_has_no_partition_or_fixed_cost(rank):
    generator = torch.Generator().manual_seed(2)
    h = torch.randn(15, 4, generator=generator, dtype=torch.double)
    q = torch.randn(15, 3, generator=generator, dtype=torch.double).softmax(1)

    def forbidden(*args):
        raise AssertionError("Random initialization must not use a seed partition")

    result = low_rank_partition(
        h, q, 3, 1.0, forbidden, seed=7, rank=rank, initialization="random", mixing=0.0, steps=2
    )
    g = torch.Generator().manual_seed(7)
    u = torch.randn(15, rank, generator=g, dtype=torch.double)
    v = torch.randn(3, rank, generator=g, dtype=torch.double)
    p = (u @ v.T / rank**0.5).softmax(1)
    x = h - h.mean(0)
    x /= x.square().sum(1).mean().sqrt()
    stats = p.mean(0), p.T @ x / len(x), p.T @ q / len(x)
    _, initial, _, _ = moment_objective(stats, x.square().sum(1).mean(), x.T @ q / len(x), 1.0)
    assert result["history"][0] == pytest.approx(float(initial), abs=1e-12)
    assert result["initial_assignment_digest"] is None
    assert result["mixing"] is None
    assert result["J"] <= result["history"][0]
