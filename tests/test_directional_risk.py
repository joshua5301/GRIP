import torch

from src.directional_risk import audit_heads, directional_partition, statistics, score_statistics


def test_directional_bound_and_exact_risk_decomposition():
    torch.manual_seed(5)
    u = torch.randn(12, 3, 4, dtype=torch.double)
    q = torch.randn(12, 4, dtype=torch.double).softmax(1)
    assignment = torch.arange(12) // 3
    bias = torch.randn(3, 4, dtype=torch.double)
    table = audit_heads(u, q, assignment, bias)
    assert table.bound_violation.max() < 1e-12
    n, sums, labels = statistics(u, q, assignment, 4)
    centers = sums / n[:, None, None]
    energy = u.square().sum(2).mean(0)
    uq = (u * q[:, None]).sum(2).mean(0)
    _, _, moment = score_statistics(energy, uq, n, sums, labels)
    jensen = (u + bias).logsumexp(2).mean(0) - ((centers + bias).logsumexp(2) * (n / n.sum())[:, None]).sum(0)
    torch.testing.assert_close(torch.tensor(table.signed_gap.to_numpy()), jensen - moment)


def test_refinement_monotonic_nonempty_and_label_means():
    torch.manual_seed(2)
    u = torch.randn(18, 2, 3, dtype=torch.double)
    q = torch.randn(18, 3, dtype=torch.double).softmax(1)
    initial = torch.arange(18) % 3
    result = directional_partition(u, q, initial, max_sweeps=10, block_size=4)
    history = [r['J'] for r in result['history']]
    assert all(b <= a + 1e-12 for a, b in zip(history, history[1:]))
    assert bool((result['counts'] > 0).all())
    assert int(result['counts'].sum()) == 18
    for cluster in range(3):
        expected = q[result['assignment'] == cluster].mean(0).float()
        torch.testing.assert_close(result['y'][cluster], expected)
    repeated = directional_partition(u, q, initial, max_sweeps=10, block_size=4)
    torch.testing.assert_close(result['assignment'], repeated['assignment'])


def test_one_cell_stays_fixed():
    u = torch.ones(5, 2, 3, dtype=torch.double)
    q = torch.ones(5, 3, dtype=torch.double) / 3
    result = directional_partition(u, q, torch.zeros(5, dtype=torch.long), max_sweeps=2)
    assert result['converged']
    assert result['J'] < 1e-12
