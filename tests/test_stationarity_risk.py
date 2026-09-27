import torch

from src.representation_learning_audit import cell_means
from src.stationarity_risk import (augment, contributions, fit_head, head_objective,
                                   move_deltas, refine_partition, statistics)


def problem():
    generator = torch.Generator().manual_seed(17)
    z = torch.randn(15, 4, generator=generator, dtype=torch.double)
    q = torch.randn(15, 3, generator=generator, dtype=torch.double).softmax(1)
    assignment = torch.arange(15) % 3
    theta = torch.randn(3, 5, generator=generator, dtype=torch.double) / 3
    return z, q, assignment, theta


def test_analytic_gradient_includes_bias_and_regularization():
    z, q, assignment, theta = problem()
    centers, labels, mass = cell_means(z, q, assignment)
    theta.requires_grad_()
    gradient, = torch.autograd.grad(head_objective(augment(centers), labels, mass, theta, .03), theta)
    n, sums, targets = statistics(augment(z), q, assignment, 3)
    analytic = contributions(n, sums, targets, theta, len(z)).sum(0) + .03 * theta
    torch.testing.assert_close(gradient, analytic, atol=1e-12, rtol=1e-12)


def test_move_scores_equal_full_objective_recomputation():
    z, q, assignment, theta = problem()
    x = augment(z)
    n, sums, targets = statistics(x, q, assignment, 3)
    nodes = torch.tensor([0, 2, 7])
    destinations = torch.tensor([1, 0, 2])
    delta = move_deltas(x, q, theta, .03, n, sums, targets, nodes, assignment[nodes], destinations)

    def score(a):
        nn, ss, ll = statistics(x, q, a, 3)
        return (contributions(nn, ss, ll, theta, len(z)).sum(0) + .03 * theta).square().sum()

    original = score(assignment)
    for i, (node, target) in enumerate(zip(nodes, destinations)):
        candidate = assignment.clone()
        candidate[node] = target
        torch.testing.assert_close(delta[i], score(candidate) - original, atol=1e-12, rtol=1e-12)


def test_refinement_decreases_exact_score_and_preserves_mass_labels():
    z, q, assignment, theta = problem()
    result = refine_partition(z, q, assignment, theta, .03, max_sweeps=5, block_size=4, pair_batch=5)
    values = [row['J'] for row in result['history']]
    assert all(b <= a for a, b in zip(values, values[1:]))
    assert bool((result['counts'] > 0).all())
    assert int(result['counts'].sum()) == len(z)
    centers, labels, mass = cell_means(z, q, result['assignment'])
    torch.testing.assert_close(labels.float(), result['y'])
    x = augment(centers)
    gradient = (mass[:, None] * ((x @ theta.T).softmax(1) - labels)).T @ x + .03 * theta
    torch.testing.assert_close(gradient.square().sum(), torch.tensor(result['J'], dtype=z.dtype))
    repeated = refine_partition(z, q, assignment, theta, .03, max_sweeps=5, block_size=4, pair_batch=5)
    torch.testing.assert_close(result['assignment'], repeated['assignment'])


def test_strong_convexity_distance_bound_with_fit_tolerance():
    z, q, assignment, _ = problem()
    penalty = .03
    reference = fit_head(z, q, z.new_full((len(z),), 1 / len(z)), penalty)
    centers, labels, mass = cell_means(z, q, assignment)
    fitted = fit_head(centers, labels, mass, penalty)
    x, theta = augment(centers), reference['theta']
    residual = (mass[:, None] * ((x @ theta.T).softmax(1) - labels)).T @ x + penalty * theta
    distance = (theta - fitted['theta']).norm()
    assert distance <= (residual.norm() + fitted['grad_norm']) / penalty + 1e-9


def test_singletons_cannot_be_emptied_and_zero_sweeps_preserves_assignment():
    z, q, _, theta = problem()
    singleton = torch.arange(len(z))
    result = refine_partition(z, q, singleton, theta, .03, max_sweeps=2)
    torch.testing.assert_close(result['assignment'], singleton)
    assert result['converged']
    initial = torch.arange(len(z)) % 3
    result = refine_partition(z, q, initial, theta, .03, max_sweeps=0)
    torch.testing.assert_close(result['assignment'], initial)
    assert result['status'] == 'iteration_limit'


def test_cached_move_scores_match_uncached_scores():
    z, q, assignment, theta = problem()
    x = augment(z)
    n, sums, labels = statistics(x, q, assignment, 3)
    nodes, targets = torch.tensor([0, 2]), torch.tensor([1, 0])
    parts = contributions(n, sums, labels, theta, len(z))
    residual = parts.sum(0) + .03 * theta
    arguments = (x, q, theta, .03, n, sums, labels, nodes, assignment[nodes], targets)
    torch.testing.assert_close(move_deltas(*arguments), move_deltas(*arguments, parts, residual))


def test_warm_start_preserves_convex_optimum():
    z, q, _, _ = problem()
    mass = z.new_full((len(z),), 1 / len(z))
    previous = fit_head(z, q, mass, .1)
    warm = fit_head(z, q, mass, .03, initial_theta=previous['theta'])
    cold = fit_head(z, q, mass, .03)
    x = augment(z)
    torch.testing.assert_close(head_objective(x, q, mass, warm['theta'], .03),
                               head_objective(x, q, mass, cold['theta'], .03), atol=1e-10, rtol=1e-10)


def test_candidate_refinement_exact_score_and_no_global_convergence_claim():
    z, q, assignment, theta = problem()
    result = refine_partition(z, q, assignment, theta, .03, max_sweeps=5, block_size=5,
                              pair_batch=7, candidate_k=1, random_candidates=1)
    n, sums, labels = statistics(augment(z), q, result['assignment'], 3)
    score = (contributions(n, sums, labels, theta, len(z)).sum(0) + .03 * theta).square().sum()
    torch.testing.assert_close(score, torch.tensor(result['J'], dtype=z.dtype))
    assert not result['converged']
    assert result['status'] in ('candidate_stalled', 'iteration_limit')
    history = [r['J'] for r in result['history']]
    assert all(b <= a + 1e-14 for a, b in zip(history, history[1:]))
