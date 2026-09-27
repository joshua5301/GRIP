import numpy as np
import pytest
import torch
from torch_geometric.nn.conv.gcn_conv import gcn_norm

import src.ntk_risk as experiment
from src.ntk_nystrom import NystromGCN, nystrom_graph_features


def graph():
    x = torch.tensor([[1., .2], [.1, 2.], [-2., 1.], [.8, .6]], dtype=torch.float64)
    edges = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    edges, weights = gcn_norm(edges, num_nodes=len(x), dtype=x.dtype)
    s = torch.sparse_coo_tensor(edges.flip(0), weights, (len(x), len(x))).coalesce()
    return x, s


def test_all_landmarks_recover_full_graph_kernel_and_cross_kernel():
    x, s = graph()
    features, mapping, details = nystrom_graph_features(x, s, landmarks=len(x), block_size=2)
    torch.testing.assert_close(features @ features.T, experiment.graph_kernel(x, s), atol=1e-9, rtol=1e-9)
    synthetic = torch.tensor([[.6, .9], [.4, .2]], dtype=x.dtype)
    propagated = torch.sparse.mm(s, x) / np.sqrt(x.shape[1])
    expected = experiment.tangent_kernel(synthetic / np.sqrt(x.shape[1]), propagated) @ s.to_dense().T
    torch.testing.assert_close(mapping(synthetic) @ features.T, expected, atol=1e-9, rtol=1e-9)
    assert details['base_probe_relative_fro'] < 1e-8
    restored = NystromGCN.from_state(mapping.state_dict(), 'cpu')
    torch.testing.assert_close(restored(synthetic), mapping(synthetic))


def test_low_rank_features_and_input_gradients_are_finite():
    x, s = graph()
    features, mapping, details = nystrom_graph_features(x, s, landmarks=2, block_size=1)
    again, _, _ = nystrom_graph_features(x, s, landmarks=2, block_size=3)
    torch.testing.assert_close(features, again)
    assert features.shape == (len(x), details['rank'])
    assert details['rank'] <= 2
    synthetic = x[:2].clone().requires_grad_(True)
    mapping(synthetic).square().sum().backward()
    assert torch.isfinite(synthetic.grad).all()


def test_nystrom_reconstruction_avoids_dense_membership(monkeypatch):
    x, s = graph()
    features, mapping, _ = nystrom_graph_features(x, s, landmarks=2)
    assignment = torch.tensor([0, 0, 1, 1])

    def forbidden(*args, **kwargs):
        raise RuntimeError('Dense membership is forbidden in approximate reconstruction')

    monkeypatch.setattr(experiment.F, 'one_hot', forbidden)
    result = experiment.fit_representatives(x, s, None, assignment, steps=3,
                                            feature_map=mapping, mapped_features=features)
    means = torch.stack([x[:2].mean(0), x[2:].mean(0)])
    targets = torch.stack([features[:2].mean(0), features[2:].mean(0)])
    expected = (mapping(means) - targets).square().sum(1).mean() / features.square().sum(1).mean()
    np.testing.assert_allclose(result['reconstruction_initial'], expected, atol=1e-10)
    assert result['reconstruction_final'] <= result['reconstruction_initial'] + 1e-12
    weights = result['weights']
    torch.testing.assert_close(torch.zeros(2, dtype=x.dtype).index_add_(0, assignment, weights),
                               torch.ones(2, dtype=x.dtype))


def test_large_exact_kernel_is_rejected(tmp_path):
    with pytest.raises(ValueError, match='Nyström'):
        experiment.run_ntk_risk({'arxiv': [.0025]}, {}, tmp_path)


def test_two_main_methods_support_inductive_splits_and_resume(tmp_path, monkeypatch):
    x, s = graph()
    train = dict(x=x.float(), y=torch.tensor([0, 1, 0, 1]), adj=s.float().to_sparse_csr())
    evaluation = dict(x=x[:2].float(), y=torch.tensor([0, 1]), adj=torch.eye(2).to_sparse_csr())
    mask = torch.ones(len(x), dtype=torch.bool)
    h = torch.sparse.mm(train['adj'], torch.sparse.mm(train['adj'], train['x']))
    monkeypatch.setattr(experiment, '_prepare_dataset', lambda *args: (train, mask, (evaluation, None), (evaluation, None), h))
    monkeypatch.setattr(experiment, 'get_kernel_features', lambda h, *args: h.double())
    monkeypatch.setattr(experiment, 'fit_logistic', lambda x, y, gamma: x.new_zeros(x.shape[1], y.shape[1]))
    monkeypatch.setitem(experiment.BUDGET, ('flickr', .001), 2)
    calls = []

    def student(cx, cy, validation, params, seed, settings, testing=None, counts=None):
        calls.append((testing, settings['loss_weighting']))
        assert validation[1] is None
        assert int(counts.sum()) == len(x)
        return .5, .5 if testing is not None else None, 1

    monkeypatch.setattr(experiment, '_train_student', student)
    space = dict(teacher_kernel=['relu'], basis=[4], gamma=[.01], T=[1.], B=[1.],
                 dropout=[.5], lr=[.01], weight_decay=[.0005])
    options = dict(datasets={'flickr': [.001]}, space=space, output_dir=tmp_path,
                   nystrom=dict(landmarks=2), device='cpu', search_seeds=[0], final_seeds=[100],
                   max_sweeps=1, reconstruction_steps=2)
    result = experiment.run_main_risk(**options)
    assert list(zip(result.representation, result['mode'])) == [('s2x', 's2x_mean'), ('ntk', 'raw_convex')]
    assert len(calls) == 4 and all(weighting == 'mass' for _, weighting in calls)
    experiment.run_main_risk(**options)
    assert len(calls) == 4
