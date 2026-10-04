import torch

from src.propagation_kmeans import kernel_labels, push_average_operator, push_labels


def test_push_matches_fixed_point_and_keeps_train_soft():
    adjacency = torch.tensor([[1., 1., 1.], [1., 1., 0.], [1., 0., 1.]], dtype=torch.float64)
    operator = push_average_operator(adjacency.to_sparse_csr())
    p = adjacency / adjacency.sum(1, keepdim=True)
    expected_operator = p.T / p.sum(0)[:, None]
    torch.testing.assert_close(operator.to_dense(), expected_operator)
    source = torch.tensor([[1., 0.], [.5, .5], [0., 1.]], dtype=torch.float64)
    q, info = push_labels(operator, source, .8, tolerance=1e-10)
    expected = torch.linalg.solve(torch.eye(3, dtype=q.dtype) - .8 * expected_operator, .2 * source)
    torch.testing.assert_close(q, expected)
    torch.testing.assert_close(q.sum(1), torch.ones(3, dtype=q.dtype))
    assert info["converged"] and 0 < q[0, 1] < 1


def test_kernel_prior_and_conflicting_evidence():
    d = torch.tensor([[0., 0.], [1e8, 1e8]])
    labels = torch.tensor([[0, 1], [0, 0]])
    q = kernel_labels(d, labels, classes=2, sigma=1., beta=1.)
    torch.testing.assert_close(q, torch.full((2, 2), .5))
