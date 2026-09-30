import pytest
import torch

from src.coarsening import assignment_probability, feature_centroids, quotient_adjacency
from src.low_rank_assignment import (
    assignment_inputs,
    encode_nodes,
    initialize_factors,
    initialize_mlp,
    logit_block,
)


def problem():
    generator = torch.Generator().manual_seed(71)
    logits = torch.randn(6, 3, generator=generator, dtype=torch.double)
    adjacency = torch.tensor(
        [
            [0, 1, 0, 0, 0, 0],
            [1, 0, 2, 0, 0, 0],
            [0, 2, 0, 1, 0, 0],
            [0, 0, 1, 0, 3, 0],
            [0, 0, 0, 3, 0, 1],
            [0, 0, 0, 0, 1, 0],
        ],
        dtype=torch.double,
    )
    return logits, adjacency


@pytest.mark.parametrize("layout", ["dense", "coo", "csr"])
@pytest.mark.parametrize("mass_scaling", [False, True])
def test_quotient_matches_dense_reference_and_preserves_edge_mass(layout, mass_scaling):
    logits, adjacency = problem()
    probability = logits.softmax(1)
    supplied = {"dense": adjacency, "coo": adjacency.to_sparse(), "csr": adjacency.to_sparse_csr()}[layout]
    expected = probability.T @ adjacency @ probability
    if mass_scaling:
        inverse = probability.sum(0).rsqrt()
        expected = inverse[:, None] * expected * inverse[None, :]
    actual = quotient_adjacency(
        probability,
        supplied,
        normalization="none",
        mass_scaling=mass_scaling,
        chunk_size=2,
    )
    torch.testing.assert_close(actual, expected)
    if not mass_scaling:
        torch.testing.assert_close(actual.sum(), adjacency.sum())


def test_singleton_hard_assignment_reproduces_graph_features_and_gcn_normalization():
    _, adjacency = problem()
    permutation = torch.tensor([3, 0, 5, 1, 4, 2])
    probability = torch.eye(6, dtype=torch.double)[:, permutation]
    features = torch.arange(18, dtype=torch.double).reshape(6, 3)
    centers, mass = feature_centroids(probability, features)
    torch.testing.assert_close(centers, features[permutation])
    torch.testing.assert_close(mass, torch.ones(6, dtype=torch.double))
    actual = quotient_adjacency(probability, adjacency, normalization="none", mass_scaling=True)
    torch.testing.assert_close(actual, adjacency[permutation][:, permutation])
    expected = adjacency + torch.eye(6, dtype=torch.double)
    inverse = expected.sum(1).rsqrt()
    expected = inverse[:, None] * expected * inverse[None, :]
    normalized = quotient_adjacency(probability, adjacency, self_loops="add")
    torch.testing.assert_close(normalized, expected[permutation][:, permutation])


@pytest.mark.parametrize("self_loops", ["retain", "remove", "add", "unit"])
def test_self_loop_policy_and_row_normalization(self_loops):
    logits, adjacency = problem()
    probability = logits.softmax(1)
    expected = probability.T @ adjacency @ probability
    if self_loops in ("remove", "unit"):
        expected = expected - torch.diag_embed(expected.diagonal())
    if self_loops in ("add", "unit"):
        expected = expected + torch.eye(3, dtype=torch.double)
    expected = expected / expected.sum(1, keepdim=True)
    actual = quotient_adjacency(probability, adjacency, self_loops=self_loops, normalization="row")
    torch.testing.assert_close(actual, expected)


def test_coarsening_gradient_finite_differences():
    logits, adjacency = problem()
    logits.requires_grad_()
    features = torch.arange(12, dtype=torch.double).reshape(6, 2).requires_grad_()

    def coarsened(value, original_features):
        probability = value.softmax(1)
        quotient = quotient_adjacency(
            probability,
            adjacency.to_sparse_csr(),
            mass_scaling=True,
            self_loops="add",
            chunk_size=2,
        )
        centers, _ = feature_centroids(probability, original_features)
        return quotient @ centers

    assert torch.autograd.gradcheck(coarsened, (logits, features), eps=1e-6, atol=2e-6, rtol=2e-4)


def test_snapshot_replay_for_low_rank_dense_mlp_and_balanced_assignments():
    z, _ = problem()
    q = z.softmax(1)
    assignment = torch.arange(6) % 3
    u, v = initialize_factors(assignment, 3, 2, seed=4)
    saved = dict(u=u, v=v, assignment=assignment, mixing=0.05)
    logits = logit_block(u, v, assignment, 0.05).double()
    torch.testing.assert_close(assignment_probability(saved, chunk_size=2), logits.softmax(1))
    dual = torch.tensor([0.3, 0.0, -0.3], dtype=torch.double)
    torch.testing.assert_close(
        assignment_probability(logits, column_dual=dual),
        (logits + dual).softmax(1),
    )
    inputs = assignment_inputs(z, q, "features_labels")
    parameters, v = initialize_mlp(inputs, 3, 2, hidden=5)
    with torch.no_grad():
        parameters[-2].fill_(0.2)
        parameters[-1].fill_(0.1)
    saved = dict(
        v=v,
        assignment=assignment,
        mixing=0.05,
        assignment_input="features_labels",
        encoder_parameters=parameters,
    )
    replay = assignment_probability(saved, z=z, q=q, chunk_size=2)
    torch.testing.assert_close(replay.sum(1), torch.ones(6, dtype=torch.double))
    expected = logit_block(encode_nodes(inputs, parameters), v, assignment, 0.05).double().softmax(1)
    torch.testing.assert_close(replay, expected)


def test_isolated_cells_stay_zero_and_empty_means_are_rejected():
    probability = torch.eye(3, dtype=torch.double).requires_grad_()
    adjacency = torch.zeros(3, 3, dtype=torch.double)
    result = quotient_adjacency(probability, adjacency)
    torch.testing.assert_close(result, adjacency)
    (gradient,) = torch.autograd.grad(result.sum(), probability)
    assert bool(torch.isfinite(gradient).all())
    with pytest.raises(ValueError, match="nonempty"):
        feature_centroids(torch.tensor([[1.0, 0.0], [1.0, 0.0]]), torch.ones(2, 2))
    with pytest.raises(ValueError, match="sum to one"):
        quotient_adjacency(probability * 2, adjacency)
