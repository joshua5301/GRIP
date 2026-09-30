import pytest
import torch

from src.coarsening_ce import (
    coarsened_problem,
    gcn_inputs,
    implicit_low_rank_gradient,
    optimize_coarsening_ce,
)
from src.evaluation import fit_gcn_diagnostic
from src.low_rank_assignment import logit_block
from src.moments import augmented
from src.soft_ce_partition import head_gradient, outer_value_gradient, solve_inner_newton_first
from src.transforms import fit_transform


def problem():
    generator = torch.Generator().manual_seed(814)
    x = torch.randn(9, 3, generator=generator, dtype=torch.double)
    q = torch.randn(9, 2, generator=generator, dtype=torch.double).softmax(1)
    adjacency = torch.eye(9, dtype=torch.double)
    for i in range(8):
        adjacency[i, i + 1] = adjacency[i + 1, i] = 0.5 + i / 4
    inverse = adjacency.sum(1).rsqrt()
    adjacency = inverse[:, None] * adjacency * inverse[None, :]
    h = adjacency @ (adjacency @ x)
    outer_z, transform = fit_transform(h)
    assignment = torch.arange(9) % 3
    u = (0.4 * torch.randn(9, 2, generator=generator, dtype=torch.double)).requires_grad_()
    v = torch.randn(3, 2, generator=generator, dtype=torch.double).requires_grad_()
    return x, q, adjacency.to_sparse_csr(), h, outer_z, transform, assignment, u, v


@pytest.mark.parametrize("mass_scaling", [False, True])
def test_implicit_gradient_matches_refitted_objective_including_edge_derivative(mass_scaling):
    x, q, adjacency, _, outer_z, transform, assignment, u, v = problem()
    fitted = implicit_low_rank_gradient(
        u,
        v,
        assignment,
        x,
        q,
        adjacency,
        outer_z,
        transform,
        0.1,
        mass_scaling=mass_scaling,
        inner_tol=1e-10,
        cg_rtol=1e-10,
    )
    direction_u = torch.arange(u.numel(), dtype=u.dtype).reshape_as(u).sin()
    direction_v = torch.arange(v.numel(), dtype=v.dtype).reshape_as(v).cos()

    def objective(left, right):
        probability = logit_block(left, right, assignment, 0.05).softmax(1)
        condensed = coarsened_problem(probability, x, q, adjacency, transform, mass_scaling=mass_scaling)
        inner = solve_inner_newton_first(
            condensed["head_features"], condensed["labels"], condensed["mass"], 0.1, grad_tol=1e-11
        )
        assert inner["inner_converged"]
        return outer_value_gradient(outer_z, q, inner["theta"])[0]

    epsilon = 1e-3
    numerical = (
        objective(u.detach() + epsilon * direction_u, v.detach() + epsilon * direction_v)
        - objective(u.detach() - epsilon * direction_u, v.detach() - epsilon * direction_v)
    ) / (2 * epsilon)
    actual = sum(
        (gradient * direction).sum()
        for gradient, direction in zip(fitted["gradients"], (direction_u, direction_v), strict=True)
    )
    torch.testing.assert_close(actual, actual.new_tensor(numerical), rtol=5e-4, atol=3e-8)

    # Detaching Sc loses a material part of the derivative, even though Xc/Qc remain differentiable.
    probability = logit_block(u, v, assignment, 0.05).softmax(1)
    condensed = coarsened_problem(probability, x, q, adjacency, transform, mass_scaling=mass_scaling)
    fixed_edges = condensed["adj"].detach()
    features = transform(fixed_edges @ (fixed_edges @ condensed["x"]))
    derivative = head_gradient(
        augmented(features), condensed["labels"], condensed["mass"], fitted["theta"], 0.1
    )
    detached_edge_gradients = torch.autograd.grad(-(derivative * fitted["vector"]).sum(), (u, v))
    missing = sum(
        (gradient * direction).sum()
        for gradient, direction in zip(detached_edge_gradients, (direction_u, direction_v), strict=True)
    )
    assert abs(float(actual - missing)) > 1e-5


def test_checkpoint_resume_matches_uninterrupted_updates_and_is_gcn_ready(tmp_path):
    x, q, adjacency, h, _, transform, assignment, _, _ = problem()
    kwargs = dict(
        h=h,
        transform=transform,
        penalty=0.1,
        rank=2,
        seed=2,
        checkpoint_steps=(0, 1, 2),
        inner_tol=1e-9,
        cg_rtol=1e-9,
    )
    full = optimize_coarsening_ce(x, q, assignment, adjacency, steps=2, folder=tmp_path / "full", **kwargs)
    short_options = dict(kwargs, checkpoint_steps=(0, 1))
    optimize_coarsening_ce(x, q, assignment, adjacency, steps=1, folder=tmp_path / "resumed", **short_options)
    state = torch.load(tmp_path / "resumed" / "resume.pt", weights_only=False)
    resumed = optimize_coarsening_ce(
        x, q, assignment, adjacency, steps=2, folder=tmp_path / "resumed", resume_state=state, **kwargs
    )
    for key in ("x", "labels", "mass", "adj", "probability", "u", "v", "theta"):
        torch.testing.assert_close(
            full["checkpoints"][2][key], resumed["checkpoints"][2][key], rtol=2e-6, atol=2e-8
        )
    endpoint = resumed["checkpoints"][2]
    assert endpoint["x"].shape == (3, 3)
    torch.testing.assert_close(endpoint["labels"].sum(1), torch.ones(3, dtype=torch.double))
    torch.testing.assert_close(endpoint["mass"].sum(), torch.tensor(1.0, dtype=torch.double))
    torch.testing.assert_close(
        endpoint["head_features"], transform(endpoint["adj"] @ (endpoint["adj"] @ endpoint["x"]))
    )
    assert set(resumed["checkpoints"]) == {0, 1, 2}
    assert all(row["inner_converged"] for row in resumed["history"])
    assert [row["step"] for row in resumed["history"]] == [0, 1, 2]


def test_resume_rejects_changed_graph_or_targets(tmp_path):
    x, q, adjacency, h, _, _, assignment, _, _ = problem()
    result = optimize_coarsening_ce(x, q, assignment, adjacency, h=h, steps=0, rank=2, penalty=0.1)
    changed = (q * 0.9 + 0.1 / q.shape[1]).detach()
    with pytest.raises(ValueError, match="Resume state"):
        optimize_coarsening_ce(
            x,
            changed,
            assignment,
            adjacency,
            h=h,
            steps=1,
            rank=2,
            penalty=0.1,
            resume_state=result["resume_state"],
        )


def test_existing_gcn_training_and_original_graph_evaluation(tmp_path):
    x, q, adjacency, h, _, _, assignment, _, _ = problem()
    result = optimize_coarsening_ce(x, q, assignment, adjacency, h=h, steps=1, rank=2, penalty=0.1)
    cx, cy, mass, condensed_adjacency = gcn_inputs(result["checkpoints"][1])
    graph = dict(x=x.float(), y=q.argmax(1), adj=adjacency.float())
    masks = dict(train=torch.arange(9) < 3, val=(torch.arange(9) >= 3) & (torch.arange(9) < 6))
    kwargs = dict(
        cx=cx,
        cy=cy,
        mass=torch.ones_like(mass),
        graph=graph,
        q=q.float(),
        masks=masks,
        seed=3,
        epochs=4,
        eval_every=1,
        hidden=4,
        dropout=0.0,
        lr=0.01,
        weight_decay=0.0005,
        training_adjacency=condensed_adjacency,
    )
    first = fit_gcn_diagnostic(folder=tmp_path / "first", **kwargs)
    second = fit_gcn_diagnostic(folder=tmp_path / "second", **kwargs)
    assert first == second
    assert 0 <= first["val_acc"] <= 100
    assert first["full_teacher_ce"] > 0
