import torch

import src.moment_lloyd as module


def example():
    generator = torch.Generator().manual_seed(7)
    x = torch.randn(18, 4, generator=generator, dtype=torch.float64)
    q = torch.randn(18, 3, generator=generator, dtype=torch.float64).softmax(1)
    return x, q, torch.arange(18) % 3


def test_exact_move_matches_reaggregation():
    x, q, assignment = example()
    energy, original = x.square().sum(1).mean(), x.T @ q / len(x)
    state = module.statistics(x, q, assignment, 3, energy, original)
    weight = 1.7
    before = state[3] + weight * state[4].norm()
    for i in range(len(x)):
        delta = module.move_deltas(x[i], q[i], int(assignment[i]), state, len(x), weight)
        for target in range(3):
            if target == assignment[i]:
                continue
            moved = assignment.clone()
            moved[i] = target
            after = module.statistics(x, q, moved, 3, energy, original)
            torch.testing.assert_close(delta[target], after[3] + weight * after[4].norm() - before)


def test_cost_is_assignment_derivative_up_to_row_constant():
    x, q, _ = example()
    p = torch.randn(18, 3, dtype=x.dtype).softmax(1).requires_grad_()
    mass = p.sum(0)
    centers, labels = p.T @ x / mass[:, None], p.T @ q / mass[:, None]
    moment = x.T @ q / len(x) - centers.T @ (mass[:, None] * labels) / len(x)
    loss = x.square().sum(1).mean() - (mass * centers.square().sum(1)).sum() / len(x) + 2 * moment.norm()
    (gradient,) = torch.autograd.grad(loss, p)
    cost = module.assignment_cost(x, q, centers, labels, moment, 2)
    torch.testing.assert_close((gradient - gradient[:, :1]) * len(x), cost - cost[:, :1])


def test_paired_initialization_and_monotonicity():
    x, q, _ = example()
    outputs = [
        module.moment_lloyd_partition(x, q, 3, moment_weight=2, mode=mode, max_sweeps=10)
        for mode in ("hybrid", "full_only")
    ]
    assert outputs[0]["initial_assignment_digest"] == outputs[1]["initial_assignment_digest"]
    for result in outputs:
        assert all(b < a for a, b in zip(result["history"], result["history"][1:]))
        assert (result["counts"] > 0).all()
        assert result["counts"].sum() == len(x)
        torch.testing.assert_close(result["y"].sum(1), torch.ones(3))


def test_full_rejection_keeps_initial_state(monkeypatch):
    x = torch.tensor([[0.0], [0.1], [10.0], [10.1]], dtype=torch.float64)
    q = torch.ones(4, 1, dtype=x.dtype)
    monkeypatch.setattr(
        module, "assignment_cost", lambda *args: x.new_tensor([[0, 1], [1, 0], [0, 1], [1, 0]])
    )
    initial = module.moment_lloyd_partition(x, q, 2, max_sweeps=0)
    result = module.moment_lloyd_partition(x, q, 2, mode="full_only")
    assert result["status"] == "full_rejected"
    assert not result["converged"]
    assert len(result["history"]) == 1
    torch.testing.assert_close(initial["assignment"], result["assignment"])
