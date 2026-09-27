import torch

from src.soft_ridge_partition import (AssignmentMoments, augmented, decode_moments,
                                      excess_objective, initial_logits, make_material,
                                      moment_objective, optimize_assignment, ridge_head)


def problem():
    generator = torch.Generator().manual_seed(29)
    z = torch.randn(12, 4, generator=generator, dtype=torch.double)
    q = torch.randn(12, 3, generator=generator, dtype=torch.double).softmax(1)
    assignment = torch.arange(12) % 3
    x = augmented(z)
    system = x.T @ x / len(z) + .02 * torch.eye(x.shape[1], dtype=z.dtype)
    reference = torch.linalg.solve(system, x.T @ q / len(z))
    return z, q, assignment, system, reference


def test_chunked_aggregation_and_gradient_match_dense_autograd():
    z, q, assignment, system, reference = problem()
    material = make_material(z, q)
    logits = initial_logits(assignment, 3, .1, dtype=torch.double).requires_grad_()
    chunked = AssignmentMoments.apply(logits, material, 5)
    dense = logits.softmax(1).T @ material / len(z)
    torch.testing.assert_close(chunked, dense)
    a = moment_objective(chunked, z.shape[1], .02, reference, system)
    b = moment_objective(dense, z.shape[1], .02, reference, system)
    ga, = torch.autograd.grad(a, logits)
    gb, = torch.autograd.grad(b, logits)
    torch.testing.assert_close(ga, gb, atol=1e-11, rtol=1e-9)


def test_assignment_gradient_finite_differences():
    z, q, assignment, system, reference = problem()
    material = make_material(z, q)
    logits = initial_logits(assignment, 3, .1, dtype=torch.double).requires_grad_()
    def objective(value):
        moments = AssignmentMoments.apply(value, material, 5)
        return moment_objective(moments, z.shape[1], .02, reference, system)
    assert torch.autograd.gradcheck(objective, (logits,), eps=1e-6, atol=1e-6, rtol=1e-4)


def test_mixing_preserves_total_mass_feature_and_label_means():
    z, q, assignment, _, _ = problem()
    logits = initial_logits(assignment, 3, .1, dtype=torch.double)
    expected = .9 * torch.nn.functional.one_hot(assignment, 3).double() + .1 / 3
    torch.testing.assert_close(logits.softmax(1), expected)
    moments = AssignmentMoments.apply(logits, make_material(z, q), 4)
    centers, labels, mass = decode_moments(moments, z.shape[1])
    torch.testing.assert_close(mass.sum(), z.new_tensor(1.))
    torch.testing.assert_close((mass[:, None] * centers).sum(0), z.mean(0))
    torch.testing.assert_close((mass[:, None] * labels).sum(0), q.mean(0))
    assert bool((mass > 0).all())


def test_excess_equals_actual_regularized_full_loss_difference():
    z, q, assignment, system, reference = problem()
    moments = AssignmentMoments.apply(initial_logits(assignment, 3, .1, dtype=torch.double), make_material(z, q), 5)
    centers, labels, mass = decode_moments(moments, z.shape[1])
    weight = ridge_head(centers, labels, mass, .02)
    def objective(w):
        return (augmented(z) @ w - q).square().sum() / (2 * len(z)) + .02 * w.square().sum() / 2
    torch.testing.assert_close(excess_objective(weight, reference, system), objective(weight) - objective(reference))


def test_best_assignment_checkpoint_matches_representatives(tmp_path):
    z, q, assignment, system, reference = problem()
    result = optimize_assignment(z, q, assignment, .02, reference, system, steps=4,
                                 chunk_size=5, log_every=1, folder=tmp_path)
    best_logits = torch.load(tmp_path / 'best_assignment_logits.pt', weights_only=True)
    reproduced = AssignmentMoments.apply(best_logits, make_material(z, q), 5)
    torch.testing.assert_close(reproduced, result['best_moments'])
    value = moment_objective(reproduced, z.shape[1], .02, reference, system)
    assert abs(float(value) - result['best_J']) < 1e-12
    history = [r['best_J'] for r in result['history']]
    assert all(b <= a for a, b in zip(history, history[1:]))
