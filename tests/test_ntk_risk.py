import numpy as np
import torch

from src.gcn_kernel_features import gcn_two_layer_kernels
from src.ntk_risk import fit_representatives, graph_kernel, spectral_features, tangent_kernel
from torch_geometric.nn.conv.gcn_conv import gcn_norm


def propagation(edges, nodes):
    edges, weights = gcn_norm(edges, num_nodes=nodes, dtype=torch.float64)
    return torch.sparse_coo_tensor(edges.flip(0), weights, (nodes, nodes)).coalesce()


def test_kernel_matches_existing_gcn_ntk():
    x = torch.tensor([[1., .2], [.1, 2.], [-2., 1.], [0., 0.]], dtype=torch.float64)
    edges = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    actual = graph_kernel(x, propagation(edges, len(x)))
    expected = gcn_two_layer_kernels(x.numpy(), edges.numpy(), device='cpu')['gcn2_ntk']
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
    features = spectral_features(actual)
    torch.testing.assert_close(features @ features.T, actual)


def test_cross_graph_kernel_matches_disjoint_union():
    x = torch.tensor([[1., .2], [.1, 2.], [-2., 1.]], dtype=torch.float64)
    synthetic = torch.tensor([[.5, .7], [.2, .9]], dtype=torch.float64)
    edges = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    s = propagation(edges, len(x))
    union = graph_kernel(torch.cat([x, synthetic]), propagation(edges, len(x) + len(synthetic)))
    cross = tangent_kernel(synthetic / np.sqrt(2), torch.sparse.mm(s, x) / np.sqrt(2)) @ s.to_dense().T
    torch.testing.assert_close(cross, union[3:, :3])
    torch.testing.assert_close(union.diag()[3:], synthetic.square().sum(1))


def test_endpoint_gradients_and_zero_vectors_are_finite():
    x = torch.tensor([[1., 0.], [-1., 0.], [0., 0.]], dtype=torch.float64, requires_grad=True)
    values = tangent_kernel(x, x.detach())
    values.sum().backward()
    assert torch.isfinite(x.grad).all()
    torch.testing.assert_close(values.diag(), torch.tensor([2., 2., 0.], dtype=torch.float64))


def test_reconstruction_matches_explicit_rkhs_distance_and_stays_in_cell():
    x = torch.tensor([[1., .2], [.1, 2.], [-2., 1.]], dtype=torch.float64)
    edges = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    s = propagation(edges, len(x))
    kernel = graph_kernel(x, s)
    assignment = torch.tensor([0, 0, 1])
    result = fit_representatives(x, s, kernel, assignment, steps=4)
    weights = result['weights']
    torch.testing.assert_close(torch.zeros(2, dtype=x.dtype).index_add_(0, assignment, weights), torch.ones(2, dtype=x.dtype))
    assert bool((weights >= 0).all())
    expected_x = torch.zeros(2, 2, dtype=x.dtype).index_add_(0, assignment, weights[:, None] * x)
    torch.testing.assert_close(result['x'].double(), expected_x, atol=1e-6, rtol=1e-6)
    means = torch.stack([x[:2].mean(0), x[2]])
    union = graph_kernel(torch.cat([x, means]), propagation(edges, 5))
    membership = torch.tensor([[.5, 0], [.5, 0], [0, 1.]], dtype=x.dtype)
    error = union.diag()[3:] - 2 * (union[3:, :3] @ membership).diag() + (membership.T @ kernel @ membership).diag()
    expected = float(error @ torch.tensor([2 / 3, 1 / 3], dtype=x.dtype) / kernel.diag().mean())
    np.testing.assert_allclose(result['reconstruction_initial'], expected, atol=1e-10)
    assert result['reconstruction_final'] <= result['reconstruction_initial'] + 1e-12


def test_identity_singletons_have_zero_reconstruction_error():
    x = torch.tensor([[1., .2], [.1, 2.]], dtype=torch.float64)
    s = propagation(torch.empty(2, 0, dtype=torch.long), 2)
    result = fit_representatives(x, s, graph_kernel(x, s), torch.arange(2), steps=2)
    np.testing.assert_allclose(result['reconstruction_final'], 0, atol=1e-10)
    torch.testing.assert_close(result['x'], x.float())
