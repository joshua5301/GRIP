import json
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

import src.graph_moment_condensation as gm
from src.models import GCN
from src.risk_experiment import _forward


def test_readout_contains_both_graph_propagations_and_matches_student():
    torch.manual_seed(3)
    model = GCN(3, 5, 2, 2, .5).double().eval()
    x = torch.randn(4, 3, dtype=torch.double)
    edges = torch.randn(4, 4, dtype=torch.double)
    adjacency = gm.normalized_graph(edges)
    z = gm.readout_embedding(model, x, adjacency)
    expected = F.log_softmax(model.layers[-1].lin(z) + model.layers[-1].bias, dim=1)
    torch.testing.assert_close(expected, _forward(model, x, adjacency))
    sparse = adjacency.to_sparse_coo()
    torch.testing.assert_close(z, gm.readout_embedding(model, x, sparse))
    permutation = torch.tensor([2, 0, 3, 1])
    torch.testing.assert_close(gm.readout_embedding(model, x[permutation], adjacency[permutation][:, permutation]),
                               z[permutation])


def test_loss_detects_label_location_and_missing_variance():
    z = torch.tensor([[-1., 0.], [1., 0.]])
    q = torch.eye(2)
    mass = torch.tensor([.5, .5])
    first, second = gm.moments(z, q, mass)
    reference = dict(center=torch.zeros(2), scale=torch.tensor(1.), first=first, second=second)
    loss, _, _ = gm.moment_loss(z, q, mass, reference, 1.)
    torch.testing.assert_close(loss, torch.tensor(0.))
    _, label_error, variance_error = gm.moment_loss(z, q.flip(0), mass, reference, 1.)
    assert float(label_error) > 0
    torch.testing.assert_close(variance_error, torch.tensor(0.))
    neutral_q = torch.full((2, 2), .5)
    neutral_first, _ = gm.moments(z, neutral_q, mass)
    reference['first'] = neutral_first
    _, first_error, second_error = gm.moment_loss(torch.zeros(1, 2), torch.full((1, 2), .5),
                                                torch.ones(1), reference, 1.)
    torch.testing.assert_close(first_error, torch.tensor(0.))
    assert float(second_error) > 0


def test_graph_and_raw_features_receive_gradients_with_frozen_probe():
    torch.manual_seed(8)
    model = GCN(3, 5, 2, 2, 0.).double().eval().requires_grad_(False)
    x = torch.randn(4, 3, dtype=torch.double, requires_grad=True)
    edges = torch.randn(4, 4, dtype=torch.double, requires_grad=True)
    q = torch.tensor([[1., 0.], [0., 1.], [.8, .2], [.2, .8]], dtype=torch.double)
    mass = torch.tensor([.1, .2, .3, .4], dtype=torch.double)
    adjacency = gm.normalized_graph(edges)
    z = gm.readout_embedding(model, x, adjacency)
    reference = dict(center=torch.zeros(5, dtype=torch.double), scale=torch.tensor(1., dtype=torch.double),
                     first=torch.zeros(5, 2, dtype=torch.double), second=torch.eye(5, dtype=torch.double))
    gm.moment_loss(z, q, mass, reference, 1.)[0].backward()
    assert x.grad.abs().sum() > 0
    assert edges.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.parameters())
    torch.testing.assert_close(adjacency, adjacency.T)
    assert torch.linalg.eigvalsh(adjacency).max() <= 1 + 1e-10


def test_mean_labels_preserve_mass_with_unequal_cells():
    q = torch.tensor([[1., 0.], [.8, .2], [.6, .4], [0., 1.]])
    assignment = torch.tensor([0, 0, 0, 1])
    labels = gm.cell_means(q, assignment, 2)
    mass = torch.tensor([.75, .25])
    torch.testing.assert_close(mass @ labels, q.mean(0))
    assert not torch.allclose(labels.mean(0), q.mean(0))


def test_sweep_passes_graph_and_mass_and_resumes_without_test_selection(tmp_path, monkeypatch):
    graph = SimpleNamespace(x=torch.tensor([[1., 0.], [.8, .2], [.6, .4], [0., 1.]]),
                            edge_index=torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]]),
                            y=torch.tensor([0, 0, 1, 1]),
                            train_mask=torch.tensor([True, True, False, False]),
                            val_mask=torch.tensor([False, False, True, False]),
                            test_mask=torch.tensor([False, False, False, True]))
    teacher = tmp_path / 'teacher'
    teacher.mkdir()
    protocol = dict(dataset='cora', graph_sha256=gm.array_digest(graph.x.numpy(), graph.edge_index.numpy()),
                    supervision_sha256=gm.array_digest(graph.train_mask.numpy(), graph.val_mask.numpy(),
                                                       graph.y[graph.train_mask].numpy(), graph.y[graph.val_mask].numpy()))
    (teacher / 'protocol.json').write_text(json.dumps(protocol))
    np.savez(teacher / 'teacher_predictions.npz', logits=np.array([[2., 0.], [1., 0.], [0., 1.], [0., 2.]], dtype=np.float32))
    train = dict(x=graph.x, y=graph.y, adj=torch.eye(4))
    validation, testing = (train, graph.val_mask), (train, graph.test_mask)
    monkeypatch.setitem(gm.BUDGET, ('cora', .026), 2)
    monkeypatch.setattr(gm, 'get_dataset', lambda args: graph)
    monkeypatch.setattr(gm, '_prepare_dataset', lambda *args: (train, graph.train_mask, validation, testing, graph.x))
    monkeypatch.setattr(gm, 'kmeans_init', lambda *args: torch.tensor([0, 0, 0, 1]))
    monkeypatch.setattr(gm, '_probe_bank', lambda *args: [])
    monkeypatch.setattr(gm, '_load_probes', lambda *args: [])
    fits, students = [], []

    def fit(x, edges, labels, mass, probes, **kwargs):
        fits.append(kwargs)
        state = dict(x=x, adjacency=gm.normalized_graph(edges), objective_initial=1., objective_final=.5,
                     best_step=1, first_final=.2, second_final=.3, probe_ce_gap=.1)
        return state, gm.pd.DataFrame([dict(step=1, objective=.5)])

    def student(x, y, val, params, seed, settings, testing=None, counts=None, adjacency=None):
        torch.testing.assert_close(counts, torch.tensor([3, 1]))
        assert settings['loss_weighting'] == 'mass'
        assert adjacency.shape == (2, 2) and float(adjacency[0, 1]) > 0
        assert val is validation
        students.append((seed, testing is not None))
        return (.8 if params['T'] == 1. else .7), (.75 if testing is not None else None), 1

    monkeypatch.setattr(gm, 'fit_graph_moments', fit)
    monkeypatch.setattr(gm, '_train_student', student)
    grid = dict(T=[1., 2.], second_weight=[1.], feature_lr=[.01], edge_lr=[.01],
                dropout=[.1, .5], lr=[.01], weight_decay=[.0005])
    arguments = dict(dataset='cora', ratio=.026, output_dir=tmp_path / 'run', space=grid,
                     teacher_run=teacher, device='cpu', variants=('moments',),
                     search_seeds=(0,), final_seeds=(100,), probe_seeds=(5000,), steps=1)
    report = gm.run_graph_moment_sweep(**arguments)
    assert len(fits) == 2
    assert students == [(0, False)] * 4 + [(100, True)]
    assert report['summary'].iloc[0]['T'] == 1.
    gm.run_graph_moment_sweep(**arguments)
    assert len(fits) == 2 and len(students) == 5
