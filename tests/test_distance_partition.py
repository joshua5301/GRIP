import numpy as np
import pytest
import torch

from src.distance_partition import (
    _prepare,
    distance_implicit_gradient,
    distance_probability,
    optimize_distance_ce,
    pairwise_metric_distance,
    positive_metric_weights,
)
from src.moments import augmented, decode_moments, initial_logits, make_material
from src.soft_ce_partition import outer_value_gradient, solve_head_system, solve_inner_newton_first


def problem():
    generator = torch.Generator().manual_seed(31)
    z = torch.randn(15, 3, generator=generator, dtype=torch.double)
    q = torch.randn(15, 2, generator=generator, dtype=torch.double).softmax(1)
    return z, q, torch.arange(15) % 3


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_diagonal_distance_matches_explicit_pairs_and_weights_stay_positive(dtype):
    z, _, _ = problem()
    anchors = z[:4]
    weights = positive_metric_weights(torch.tensor([-1000.0, 0.0, 1000.0], dtype=dtype))
    assert bool((weights > 0).all())
    torch.testing.assert_close(weights.mean(), torch.tensor(1.0, dtype=dtype))
    ordinary = positive_metric_weights(torch.tensor([-0.2, 0.1, 0.4], dtype=torch.double))
    explicit = ((z[:, None] - anchors[None]) ** 2 * ordinary).sum(2)
    torch.testing.assert_close(pairwise_metric_distance(z, anchors, ordinary), explicit)


@pytest.mark.parametrize("loss_weighting", ["mass", "uniform"])
def test_initial_probability_and_representatives_exactly_match_baseline(loss_weighting):
    z, q, assignment = problem()
    inputs = torch.cat((z, 0.5 * (q - q.mean(0))), dim=1)
    result = optimize_distance_ce(z, q, assignment, metric_inputs=inputs, penalty=0.1,
                                  steps=0, inner_loss_weighting=loss_weighting)
    snapshot = result["checkpoints"][0]
    expected = initial_logits(assignment, 3, dtype=torch.double).softmax(1)
    torch.testing.assert_close(snapshot["probability"], expected, atol=0, rtol=0)
    torch.testing.assert_close(snapshot["moments"], expected.T @ make_material(z, q) / len(z), atol=0, rtol=0)
    torch.testing.assert_close(snapshot["metric_weights"], torch.ones(5, dtype=torch.double), atol=0, rtol=0)


@pytest.mark.parametrize("loss_weighting", ["mass", "uniform"])
def test_implicit_distance_gradient_matches_fully_refitted_head(loss_weighting):
    z, q, assignment = problem()
    _, _, inputs, baseline, anchors, initial_distance, material, _ = _prepare(z, q, assignment, None, 0.05)
    log_weights = torch.tensor([-0.3, 0.2, 0.1], dtype=torch.double, requires_grad=True)
    penalty = 0.07

    def objective(parameters):
        p = distance_probability(inputs, anchors, initial_distance, baseline, parameters)
        moments = p.T @ material / len(z)
        centers, labels, mass = decode_moments(moments.detach(), z.shape[1])
        if loss_weighting == "uniform":
            mass = torch.full_like(mass, 1 / len(mass))
        fitted = solve_inner_newton_first(centers, labels, mass, penalty, grad_tol=1e-11)
        assert fitted["inner_converged"]
        value, rhs = outer_value_gradient(z, q, fitted["theta"])
        return value, rhs, fitted["theta"], moments, centers, labels, mass

    _, rhs, theta, moments, centers, labels, mass = objective(log_weights)
    vector, diagnostic = solve_head_system(augmented(centers), labels, mass, theta, penalty, rhs, rtol=1e-11)
    assert diagnostic["cg_converged"]
    derivative = distance_implicit_gradient(moments, 3, theta, vector, penalty, log_weights, loss_weighting)
    direction = torch.tensor([0.4, -0.2, 0.1], dtype=torch.double)
    epsilon = 1e-4
    numerical = (objective(log_weights.detach() + epsilon * direction)[0]
                 - objective(log_weights.detach() - epsilon * direction)[0]) / (2 * epsilon)
    np.testing.assert_allclose(float(derivative @ direction), numerical, rtol=2e-4, atol=1e-8)


def test_optimization_retains_mean_constraints_and_changes_live_metric():
    z, q, assignment = problem()
    result = optimize_distance_ce(z, q, assignment, penalty=0.1, steps=3, checkpoint_steps=[1, 2], lr=0.1)
    assert result["log_weights"].norm() > 0
    assert result["best_J"] <= result["history"][0]["J"]
    for snapshot in result["checkpoints"].values():
        probability = snapshot["probability"]
        torch.testing.assert_close(probability.sum(1), torch.ones(len(z), dtype=torch.double))
        torch.testing.assert_close(snapshot["moments"], probability.T @ make_material(z, q) / len(z))
        centers, labels, _ = decode_moments(snapshot["moments"], 3)
        torch.testing.assert_close(centers, probability.T @ z / probability.sum(0)[:, None])
        torch.testing.assert_close(labels, probability.T @ q / probability.sum(0)[:, None])


@pytest.mark.parametrize("loss_weighting", ["mass", "uniform"])
def test_explicit_resume_matches_uninterrupted_updates(tmp_path, loss_weighting):
    z, q, assignment = problem()
    settings = dict(penalty=0.1, inner_loss_weighting=loss_weighting,
                    inner_tol=1e-9, cg_rtol=1e-10)
    full = optimize_distance_ce(z, q, assignment, steps=4, folder=tmp_path / "full", **settings)
    first = optimize_distance_ce(z, q, assignment, steps=2, folder=tmp_path / "resumed", **settings)
    resumed = optimize_distance_ce(z, q, assignment, steps=4, folder=tmp_path / "resumed",
                                  resume_state=first["resume_state"], **settings)
    torch.testing.assert_close(full["log_weights"], resumed["log_weights"], rtol=1e-8, atol=1e-10)
    torch.testing.assert_close(full["checkpoints"][4]["moments"], resumed["checkpoints"][4]["moments"],
                               rtol=1e-8, atol=1e-10)
    assert [row["step"] for row in resumed["history"]] == [0, 1, 2, 3, 4]
    assert (tmp_path / "resumed" / "resume.pt").exists()


def test_resume_rejects_different_data_initial_partition_or_metric(tmp_path):
    z, q, assignment = problem()
    original = optimize_distance_ce(z, q, assignment, steps=1, penalty=0.1)
    for changed in (dict(z=z + 0.01), dict(assignment=assignment.roll(1)), dict(metric_inputs=z * 2),
                    dict(penalty=0.2)):
        arguments = dict(z=z, q=q, assignment=assignment, steps=2, penalty=0.1,
                         resume_state=original["resume_state"])
        arguments.update(changed)
        with pytest.raises(ValueError, match="Resume state differs"):
            optimize_distance_ce(**arguments)


@pytest.mark.parametrize("invalid", [dict(penalty=0), dict(strength=-1), dict(steps=-1),
                                    dict(inner_loss_weighting="mse")])
def test_invalid_objective_or_budget_is_rejected(invalid):
    z, q, assignment = problem()
    with pytest.raises(ValueError):
        optimize_distance_ce(z, q, assignment, **invalid)
