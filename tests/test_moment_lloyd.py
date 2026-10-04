import pytest
import torch

import src.moment_lloyd as module


def example():
    generator = torch.Generator().manual_seed(7)
    x = torch.randn(18, 4, generator=generator, dtype=torch.float64)
    q = torch.randn(18, 3, generator=generator, dtype=torch.float64).softmax(1)
    return x, q, torch.arange(18) % 3


def test_normalized_variance_objective_and_scale_invariance():
    x, q, _ = example()
    options = dict(mode="normalized_variance", seeding="bound_var", moment_weight=2.0, max_sweeps=300)
    result = module.moment_lloyd_partition(x, q, 3, **options)
    assignment = result["assignment"]
    cx = torch.stack([x[assignment == j].mean(0) for j in range(3)])
    cq = torch.stack([q[assignment == j].mean(0) for j in range(3)])
    gx = (x - x.mean(0)).square().sum(1).mean()
    gq = (q - q.mean(0)).square().sum(1).mean()
    expected = (x - cx[assignment]).square().sum(1).mean() / gx
    expected += 2 * (q - cq[assignment]).square().sum(1).mean() / gq
    assert result["J"] == pytest.approx(float(expected), abs=1e-10)
    assert result["converged"]
    assert all(b <= a + 1e-10 for a, b in zip(result["history"], result["history"][1:]))
    changed = module.moment_lloyd_partition(7 * x + 3, 0.2 * q + 0.8 / q.shape[1], 3, **options)
    assert torch.equal(assignment, changed["assignment"])
    assert changed["J"] == pytest.approx(result["J"], abs=1e-10)


@pytest.mark.parametrize("constant_features", [False, True])
def test_normalized_variance_constant_labels(constant_features):
    x, q, _ = example()
    if constant_features:
        x = torch.ones_like(x)
    q = torch.full_like(q, 1 / q.shape[1])
    result = module.moment_lloyd_partition(
        x, q, 3, mode="normalized_variance", seeding="bound_var", max_sweeps=300,
    )
    assert result["label_weight"] == 0
    assert result["J"] >= -1e-10
    assert bool((result["counts"] > 0).all())


def test_feature_only_initialization_preserves_normalized_objective():
    x, q, _ = example()
    options = dict(mode="normalized_variance", seeding="feature_var", max_sweeps=0)
    a = module.moment_lloyd_partition(x, q, 3, moment_weight=0.0, **options)
    b = module.moment_lloyd_partition(x, q, 3, moment_weight=2.0, **options)
    assert torch.equal(a["assignment"], b["assignment"])
    assert b["label_weight"] > 0
    expected = b["V"] * b["feature_weight"] + b["label_variance"] * b["label_weight"]
    assert b["J"] == pytest.approx(expected)


def test_variance_sum_without_label_normalization():
    x, q, _ = example()
    result = module.moment_lloyd_partition(
        x, q, 3, mode="variance_sum", seeding="bound_var", moment_weight=0.5, max_sweeps=300,
    )
    assert result["label_weight"] == 0.5
    assert result["feature_weight"] == 1.0
    assert result["J"] == pytest.approx(result["V"] + 0.5 * result["label_variance"])
    assert result["converged"]


def test_raw_variance_uses_original_feature_units():
    x, q, _ = example()
    x = 7 * x + 3
    weight = 0.7
    result = module.moment_lloyd_partition(
        x, q, 3, mode="raw_variance", seeding="bound_var", moment_weight=weight, max_sweeps=300,
    )
    assignment = result["assignment"]
    cx = torch.stack([x[assignment == j].mean(0) for j in range(3)])
    cq = torch.stack([q[assignment == j].mean(0) for j in range(3)])
    expected = (x - cx[assignment]).square().sum(1).mean()
    expected += weight * (q - cq[assignment]).square().sum(1).mean()
    assert result["J"] == pytest.approx(float(expected), abs=1e-9)
    assert torch.allclose(result["x"].double(), cx, atol=1e-6)
    assert result["mode"] == "raw_variance"
    gx = (x - x.mean(0)).square().sum(1).mean()
    gq = (q - q.mean(0)).square().sum(1).mean()
    normalized = module.moment_lloyd_partition(
        x, q, 3, mode="normalized_variance", seeding="bound_var",
        moment_weight=float(weight * gq / gx), max_sweeps=300,
    )
    assert torch.equal(assignment, normalized["assignment"])
    assert result["J"] == pytest.approx(normalized["J"] * float(gx), abs=1e-9)


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
        for mode in ("hybrid", "full_only", "filtered_batch")
    ]
    assert outputs[0]["initial_assignment_digest"] == outputs[1]["initial_assignment_digest"]
    assert outputs[0]["initial_assignment_digest"] == outputs[2]["initial_assignment_digest"]
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


def test_initialization_stability_and_limit():
    x, q, _ = example()
    a, info = module.initialize(x, 3, 0, 8, iterations=100, return_info=True)
    assert info["initialization_converged"]
    centers = x.new_zeros(3, x.shape[1]).index_add_(0, a, x) / torch.bincount(a)[:, None]
    assert torch.equal(torch.cdist(x, centers).argmin(1), a)
    with pytest.raises(RuntimeError, match="Initial k-means"):
        module.moment_lloyd_partition(
            x, q, 3, initialization_steps=1, require_initialization_convergence=True
        )


def test_one_assignment_uses_same_kmeans_plus_plus_centers():
    x, q, _ = example()
    once = module.moment_lloyd_partition(x, q, 3, max_sweeps=0, initialization_steps=1)
    converged = module.moment_lloyd_partition(
        x, q, 3, max_sweeps=0, initialization_steps=100, require_initialization_convergence=True
    )
    assert once["initial_center_indices"] == converged["initial_center_indices"]
    assert once["initialization_steps"] == 1
    assert not once["initialization_converged"]
    for j in range(3):
        mask = once["assignment"] == j
        torch.testing.assert_close(once["x"][j], x[mask].mean(0).float())
        torch.testing.assert_close(once["y"][j], q[mask].mean(0).float())


def test_zero_moment_initialization_only_control():
    x, q, _ = example()
    result = module.moment_lloyd_partition(
        x, q, 3, moment_weight=0, max_sweeps=0,
        initialization_steps=100, require_initialization_convergence=True,
    )
    assert result["status"] == "initialization_only"
    assert result["initialization_converged"]
    assert result["J"] == result["V"]
    assert result["records"] == []
    assert result["sweeps"] == 0


def test_filtered_backtracks_without_exact_scan(monkeypatch):
    x, q, _ = example()
    x = x - x.mean(0)
    x = x / x.square().sum(1).mean().sqrt()
    initial = module.moment_lloyd_partition(x, q, 3, max_sweeps=0)["assignment"]
    real_statistics = module.statistics

    def forced_cost(xb, *args):
        cost = xb.new_zeros(len(xb), 3)
        target = (initial + 1) % 3
        cost.scatter_(1, target[:, None], -torch.arange(1, len(xb) + 1, dtype=xb.dtype)[:, None])
        return cost

    def controlled_score(xb, qb, a, cells, energy, original):
        state = real_statistics(xb, qb, initial, cells, energy, original)
        changed = int((a != initial).sum())
        value = 2.0 if changed == len(a) else (0.0 if changed else 1.0)
        return (*state[:3], xb.new_tensor(value), torch.zeros_like(state[4]))

    def forbidden(*args):
        raise AssertionError("Filtered batches must not scan exact single moves")

    monkeypatch.setattr(module, "assignment_cost", forced_cost)
    monkeypatch.setattr(module, "statistics", controlled_score)
    monkeypatch.setattr(module, "move_deltas", forbidden)
    result = module.moment_lloyd_partition(x, q, 3, mode="filtered_batch", max_sweeps=1)
    assert result["records"][0]["kind"] == "partial"
    assert result["records"][0]["accepted_fraction"] == 0.5
    assert result["records"][0]["checks"] == 2
    stopped = module.moment_lloyd_partition(x, q, 3, mode="filtered_batch", max_sweeps=1, backtrack_steps=0)
    assert stopped["status"] == "filtered_rejected"
    assert not stopped["converged"]
