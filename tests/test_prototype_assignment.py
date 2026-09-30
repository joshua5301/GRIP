import pytest
import torch

from src.moments import AssignmentMoments, augmented, decode_moments, make_material
from src.multiseed_sweep import aggregate_search
from src.prototype_assignment import initialize_prototypes, prototype_logits
from src.soft_ce_partition import (
    conjugate_gradient,
    hessian_operator,
    implicit_moment_gradient,
    optimize_ce_assignment,
    outer_value_gradient,
    solve_inner,
)


def problem():
    generator = torch.Generator().manual_seed(41)
    z = torch.randn(12, 3, generator=generator, dtype=torch.double)
    q = torch.randn(12, 2, generator=generator, dtype=torch.double).softmax(1)
    return z, q, torch.arange(12) % 3


def test_prototype_probabilities_equal_squared_distance_softmax():
    z, _, assignment = problem()
    centers = initialize_prototypes(z, assignment, 3)
    expected = (-(z[:, None] - centers[None]).square().sum(2) / 0.3).softmax(1)
    torch.testing.assert_close(prototype_logits(z, centers, 0.3).softmax(1), expected)


def test_implicit_prototype_gradient_matches_resolved_difference():
    z, q, assignment = problem()
    centers = initialize_prototypes(z, assignment, 3)
    material = make_material(z, q)

    def solve(prototypes):
        moments = AssignmentMoments.apply(prototype_logits(z, prototypes, 0.7), material, 4)
        x, y, mass = decode_moments(moments.detach(), z.shape[1])
        fitted = solve_inner(x, y, mass, 0.2, grad_tol=1e-11)
        assert fitted["inner_converged"]
        value, gradient = outer_value_gradient(z, q, fitted["theta"])
        return moments, fitted["theta"], value, gradient

    moments, theta, _, gradient = solve(centers)
    x, y, mass = decode_moments(moments.detach(), z.shape[1])
    multiply, diagonal = hessian_operator(augmented(x), y, mass, theta, 0.2)
    vector, info = conjugate_gradient(multiply, gradient, diagonal, rtol=1e-11, atol=1e-14)
    assert info["cg_converged"]
    adjoint = implicit_moment_gradient(moments, z.shape[1], theta, vector, 0.2)
    (derivative,) = torch.autograd.grad(moments, centers, grad_outputs=adjoint)
    direction = torch.linspace(-0.4, 0.7, centers.numel(), dtype=z.dtype).reshape_as(centers)
    direction /= direction.norm()
    eps = 1e-3
    finite = (solve(centers.detach() + eps * direction)[2] - solve(centers.detach() - eps * direction)[2]) / (
        2 * eps
    )
    torch.testing.assert_close((derivative * direction).sum(), z.new_tensor(finite), atol=1e-7, rtol=3e-3)


def test_prototype_resume_and_saved_parameters(tmp_path):
    z, q, assignment = problem()
    options = dict(
        assignment_input="features",
        assignment_encoder="prototype",
        prototype_temperature=0.7,
        penalty=0.2,
        inner_method="newton_first",
        checkpoint_steps=(0, 1),
        save_resume=True,
        inner_tol=1e-9,
    )
    optimize_ce_assignment(z, q, assignment, steps=1, folder=tmp_path, **options)
    state = torch.load(tmp_path / "resume.pt", weights_only=False)
    resumed = optimize_ce_assignment(z, q, assignment, steps=2, resume_state=state, **options)
    full = optimize_ce_assignment(z, q, assignment, steps=2, **options)
    torch.testing.assert_close(resumed["checkpoints"][2]["moments"], full["checkpoints"][2]["moments"])
    saved = torch.load(tmp_path / "best_assignment_prototypes.pt", weights_only=False)
    assert saved["prototypes"].shape == (3, 3)
    with pytest.raises(ValueError, match="Resume state"):
        optimize_ce_assignment(
            z, q, assignment, steps=2, resume_state=state, **{**options, "prototype_temperature": 0.8}
        )


@pytest.mark.parametrize(
    "parameters",
    [{"T": 1.0, "tau": 0.1, "penalty": 0.001}, {"T": 1.0, "rank": 4, "width": 32, "penalty": 0.001}],
)
def test_method_grid_aggregation(parameters):
    rows = [
        dict(candidate=0, **parameters, step=100, condensation_seed=a, seed=b, val_acc=80 + a + b)
        for a in (0, 1, 2)
        for b in (0, 1, 2)
    ]
    result = aggregate_search(rows, [0, 1, 2], [0, 1, 2], tuple(parameters))
    assert result.iloc[0].val == 82
    assert result.iloc[0].evaluations == 9
