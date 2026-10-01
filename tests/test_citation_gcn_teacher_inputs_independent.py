"""Independent tiny CPU contracts; no datasets, fitting, or production mutations."""

import pytest
import torch
import torch.nn.functional as F

import src.citation_gcn_teacher as helper
from src.gcn_teacher import _raw_forward
from src.models import GCN


def toy_source():
    x = torch.tensor(
        [[0.2, 0.8, -0.1, 0.3], [0.5, -0.4, 0.7, 0.1],
         [-0.3, 0.2, 0.6, -0.7], [0.9, 0.1, -0.5, 0.2],
         [0.4, -0.8, 0.3, 0.6], [-0.1, 0.7, 0.2, -0.3],
         [0.8, -0.2, 0.5, 0.4], [0.3, 0.6, -0.4, 0.9]],
        dtype=torch.float32,
    )
    edges = torch.diag(torch.tensor([1., 2., 1., 3., 1., 2., 4., 1.]))
    for i, j, weight in [(0, 1, .4), (1, 2, 1.1), (2, 4, .7),
                         (3, 4, .2), (4, 5, 1.6), (5, 7, .9), (6, 7, .3)]:
        edges[i, j] = edges[j, i] = weight
    inv = edges.sum(1).rsqrt()
    adjacency = inv[:, None] * edges * inv[None, :]
    graph = dict(x=x, adj=adjacency.to_sparse_csr(),
                 y=torch.tensor([0, 2, 1, 0, 2, -999, 123456, -7]))
    train = torch.tensor([True, False, True, False, True, False, False, False])
    validation = torch.tensor([False, True, False, True, False, False, False, False])
    return graph, train, validation


def fixed_model():
    model = GCN(4, 3, 3, 2, .5)
    with torch.no_grad():
        model.layers[0].lin.weight.copy_(torch.tensor(
            [[.4, -.3, .7, .2], [-.6, .8, .1, -.4], [.5, .2, -.3, .9]]))
        model.layers[0].bias.copy_(torch.tensor([-.2, .05, .11]))
        model.layers[1].lin.weight.copy_(torch.tensor(
            [[.6, -.2, .3], [-.4, .7, .5], [.2, -.6, .8]]))
        model.layers[1].bias.copy_(torch.tensor([.03, -.07, .09]))
    return model


def clone_graph(graph):
    return {key: value.clone() for key, value in graph.items()}


def test_training_vocabulary_and_digest_exclude_all_unknown_source_labels():
    graph, train, validation = toy_source()
    before = helper.teacher_inputs(graph, train, (graph, validation))
    changed = clone_graph(graph)
    changed["y"][~(train | validation)] = torch.tensor([987654321, -1234567, 7654321])
    after = helper.teacher_inputs(changed, train, (changed, validation))
    assert torch.equal(before[0], torch.tensor([0, 1, 2]))
    assert before[1] is graph and before[2] is validation
    assert torch.equal(before[3], torch.tensor([2, 0]))
    assert torch.equal(before[4], torch.tensor([0, 1, 2]))
    for index in (0, 3, 4):
        assert torch.equal(before[index], after[index])
    assert before[5] == after[5]


@pytest.mark.parametrize("change", ["train_labels", "validation_labels", "source_x", "source_adj"])
def test_known_labels_and_all_source_features_and_edges_bind_context(change):
    graph, train, validation = toy_source()
    before = helper.teacher_inputs(graph, train, (graph, validation))[-1]
    changed = clone_graph(graph)
    if change == "train_labels":
        changed["y"][[0, 2]] = torch.tensor([1, 0])
    elif change == "validation_labels":
        changed["y"][1] = 1
    elif change == "source_x":
        changed["x"][6, 0] += .25
    else:
        changed["adj"].values()[0] += .01
    after = helper.teacher_inputs(changed, train, (changed, validation))[-1]
    assert before != after


@pytest.mark.parametrize("change", ["overlap", "different_validation_graph", "unknown_validation",
                                    "missing_training_class", "negative_training", "float_labels",
                                    "empty_train", "empty_validation", "nonboolean_train",
                                    "double_x", "nonfinite_x", "double_adj", "dense_adj"])
def test_invalid_source_boundaries_are_rejected_before_any_fit(change):
    graph, train, validation = toy_source()
    validation_graph = graph
    if change == "overlap":
        validation[0] = True
    elif change == "different_validation_graph":
        validation_graph = clone_graph(graph)
    elif change == "unknown_validation":
        graph["y"][1] = 3
    elif change == "missing_training_class":
        graph["y"][train] = torch.tensor([0, 0, 2])
    elif change == "negative_training":
        graph["y"][0] = -1
    elif change == "float_labels":
        graph["y"] = graph["y"].float()
    elif change == "empty_train":
        train[:] = False
    elif change == "empty_validation":
        validation[:] = False
    elif change == "nonboolean_train":
        train = train.long()
    elif change == "double_x":
        graph["x"] = graph["x"].double()
    elif change == "nonfinite_x":
        graph["x"][0, 0] = float("nan")
    elif change == "double_adj":
        graph["adj"] = graph["adj"].double()
    else:
        graph["adj"] = graph["adj"].to_dense()
    with pytest.raises(ValueError):
        helper.teacher_inputs(graph, train, (validation_graph, validation))


@pytest.mark.parametrize("layout", ["dense", "csr", "coo"])
def test_raw_forward_uses_supplied_packed_adjacency_once_per_layer(layout):
    graph, _, _ = toy_source()
    adjacency = graph["adj"].to_dense()
    supplied = adjacency if layout == "dense" else (
        adjacency.to_sparse_csr() if layout == "csr" else adjacency.to_sparse())
    model = fixed_model().eval()
    first, second = model.layers
    hidden = (adjacency @ (graph["x"] @ first.lin.weight.T) + first.bias).relu()
    expected = adjacency @ (hidden @ second.lin.weight.T) + second.bias
    actual = _raw_forward(model, graph["x"], supplied)
    torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-6)
    # This S has unequal degrees: applying a second normalization changes behavior.
    inv = adjacency.sum(1).rsqrt()
    twice_normalized = inv[:, None] * adjacency * inv[None, :]
    wrong_hidden = (twice_normalized @ (graph["x"] @ first.lin.weight.T) + first.bias).relu()
    wrong = twice_normalized @ (wrong_hidden @ second.lin.weight.T) + second.bias
    assert float((expected - wrong).detach().abs().max()) > 1e-4


def test_training_dropout_is_applied_to_first_relu_only_with_fixed_rng():
    graph, _, _ = toy_source()
    model = fixed_model().train()
    adjacency = graph["adj"].to_dense()
    first, second = model.layers
    torch.manual_seed(801)
    actual = _raw_forward(model, graph["x"], graph["adj"])
    torch.manual_seed(801)
    hidden = (adjacency @ (graph["x"] @ first.lin.weight.T) + first.bias).relu()
    hidden = F.dropout(hidden, p=.5, training=True)
    expected = adjacency @ (hidden @ second.lin.weight.T) + second.bias
    torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-6)


@pytest.mark.parametrize("temperature", [.3, 1.])
def test_native_fp32_logits_form_detached_double_simplex_and_exact_head_gradient(temperature):
    graph, _, _ = toy_source()
    model = fixed_model().eval()
    with torch.no_grad():
        native = _raw_forward(model, graph["x"], graph["adj"])
    assert native.dtype == torch.float32
    q = (native.double() / temperature).softmax(1)
    assert q.dtype == torch.float64 and not q.requires_grad
    torch.testing.assert_close(q.sum(1), torch.ones(len(q), dtype=torch.double), atol=3e-16, rtol=0)
    assert bool((q > 0).all())
    assert not torch.equal(q, (native / temperature).softmax(1).double())
    inputs = torch.cat((graph["x"].double(), torch.ones(len(q), 1, dtype=torch.double)), dim=1)
    theta = torch.arange(15, dtype=torch.double).reshape(5, 3).div(40).requires_grad_()
    scores = inputs @ theta
    loss = -(q * scores.log_softmax(1)).sum(1).mean()
    (gradient,) = torch.autograd.grad(loss, theta)
    expected = inputs.T @ (scores.detach().softmax(1) - q) / len(q)
    torch.testing.assert_close(gradient, expected, atol=1e-15, rtol=1e-12)
    assert all(parameter.grad is None for parameter in model.parameters())


def test_recipe_pins_existing_gcn_training_and_double_q_contract():
    recipe = helper.recipe()
    assert recipe["epochs"] == 200 and recipe["seed"] == 0
    assert recipe["q_formation"] == "float64_softmax_from_native_fp32_GCN_raw_logits"
    for key, value in dict(epochs=200, seed=0, hidden=256, dropout=.5, lr=.01,
                           weight_decay=.0005, layers=2, lr_schedule="constant",
                           initialization="pyg", selection="first maximum validation accuracy").items():
        assert recipe["settings"][key] == value
    assert len(recipe["helper_sha256"]) == 64
