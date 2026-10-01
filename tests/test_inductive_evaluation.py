import hashlib
import json

import pytest
import torch

import src.inductive_evaluation as evaluation
from src.models import GCN


def graph(nodes, offset=0):
    x = torch.tensor([[2.0 + offset, 0.1], [0.1, 2.0 + offset]] * ((nodes + 1) // 2))[:nodes]
    return dict(x=x, y=torch.arange(nodes) % 2, adj=torch.eye(nodes).to_sparse_csr())


def recipe():
    return dict(epochs=4, eval_every=1, hidden=4, dropout=0.0, lr=0.01, weight_decay=0.0005)


def inputs():
    cx = torch.tensor([[2.0, 0.1], [0.1, 2.0]])
    cy = torch.eye(2)
    return cx, cy, torch.tensor([0.2, 0.8]), graph(5), (graph(3, 0.3), None)


def test_each_split_uses_its_own_features_adjacency_and_mask():
    model = GCN(2, 2, 2, 2, 0.0)
    model.eval()
    with torch.no_grad():
        for layer in model.layers:
            layer.lin.weight.copy_(torch.eye(2))
            layer.bias.zero_()
    validation = graph(3)
    testing = graph(4)
    testing["adj"] = torch.zeros(4, 4)
    assert evaluation.graph_metrics(model, validation)["acc"] == 100
    assert evaluation.graph_metrics(model, testing)["acc"] == 50
    assert evaluation.graph_metrics(model, testing, torch.tensor([False, True, False, True]))["acc"] == 0


def test_validation_only_run_never_evaluates_a_testing_split(tmp_path, monkeypatch):
    values = inputs()
    evaluated = []
    original = evaluation.graph_metrics

    def track(model, supplied, mask=None):
        evaluated.append(supplied)
        return original(model, supplied, mask)

    monkeypatch.setattr(evaluation, "graph_metrics", track)
    result = evaluation.fit_inductive_gcn(*values, settings=recipe(), folder=tmp_path)
    assert len(evaluated) == recipe()["epochs"] + 1
    assert all(supplied is values[3] or supplied is values[4][0] for supplied in evaluated)
    assert not any(key.startswith("test_") for key in result)
    cached = json.loads((tmp_path / "seed_0.json").read_text())
    assert cached["recipe"]["test_enabled"] is False
    assert "test" not in (tmp_path / "seed_0_epochs.csv").read_text()


def test_test_labels_cannot_select_epoch_and_testing_runs_once(tmp_path, monkeypatch):
    values = inputs()
    testing = graph(4, 0.7)
    changed = dict(testing, y=1 - testing["y"])
    calls = []
    original = evaluation.graph_metrics

    def track(model, supplied, mask=None):
        calls.append(supplied)
        return original(model, supplied, mask)

    monkeypatch.setattr(evaluation, "graph_metrics", track)
    first = evaluation.fit_inductive_gcn(
        *values, testing=(testing, None), seed=2, settings=recipe(), folder=tmp_path / "first"
    )
    assert sum(supplied is testing for supplied in calls) == 1
    second = evaluation.fit_inductive_gcn(
        *values, testing=(changed, None), seed=2, settings=recipe(), folder=tmp_path / "second"
    )
    assert first["epoch"] == second["epoch"]
    assert first["val_acc"] == second["val_acc"]
    assert first["val_ce"] == second["val_ce"]
    assert first["test_acc"] + second["test_acc"] == 100
    assert (tmp_path / "first" / "seed_2_epochs.csv").read_text() == (
        tmp_path / "second" / "seed_2_epochs.csv"
    ).read_text()
    selected = torch.load(tmp_path / "first" / "seed_2_selected.pt", weights_only=False)
    model = GCN(2, recipe()["hidden"], 2, 2, recipe()["dropout"])
    model.load_state_dict(selected["model_state"])
    model.eval()
    expected = evaluation.graph_metrics(model, testing)
    assert first["test_acc"] == expected["acc"]
    assert first["test_ce"] == expected["ce"]


@pytest.mark.parametrize("change", ["settings", "features", "validation_mask", "adjacency", "weighting"])
def test_cache_rejects_mismatched_recipe_inputs_and_graph_splits(tmp_path, change):
    cx, cy, mass, train, validation = inputs()
    kwargs = dict(settings=recipe(), folder=tmp_path)
    first = evaluation.fit_inductive_gcn(cx, cy, mass, train, validation, **kwargs)
    assert first == evaluation.fit_inductive_gcn(cx, cy, mass, train, validation, **kwargs)
    if change == "settings":
        kwargs["settings"] = dict(recipe(), dropout=0.2)
    elif change == "features":
        cx = cx + 0.01
    elif change == "validation_mask":
        validation = validation[0], torch.tensor([True, True, False])
    elif change == "adjacency":
        validation = dict(validation[0], adj=torch.ones(3, 3) / 3), None
    else:
        kwargs["weighting"] = "mass"
    with pytest.raises(ValueError, match="Cached student differs"):
        evaluation.fit_inductive_gcn(cx, cy, mass, train, validation, **kwargs)


def test_transductive_train_mask_excludes_nontrain_labels_from_diagnostics_selection_and_cache(tmp_path):
    cx, cy, mass, shared_graph, _ = inputs()
    train_mask = torch.tensor([True, True, False, False, False])
    val_mask = torch.tensor([False, False, True, True, False])
    shared_graph["y"][4] = 999  # Outside the class vocabulary; it must never enter CE.
    first = evaluation.fit_inductive_gcn(
        cx,
        cy,
        mass,
        shared_graph,
        (shared_graph, val_mask),
        train_mask=train_mask,
        settings=recipe(),
        folder=tmp_path / "first",
    )
    assert first["train_nodes"] == first["val_nodes"] == 2
    changed = dict(shared_graph, y=shared_graph["y"].clone())
    changed["y"][4] = 888
    second = evaluation.fit_inductive_gcn(
        cx,
        cy,
        mass,
        changed,
        (changed, val_mask),
        train_mask=train_mask,
        settings=recipe(),
        folder=tmp_path / "second",
    )
    assert first == second
    assert first == evaluation.fit_inductive_gcn(
        cx,
        cy,
        mass,
        changed,
        (changed, val_mask),
        train_mask=train_mask,
        settings=recipe(),
        folder=tmp_path / "first",
    )
    with pytest.raises(ValueError, match="Cached student differs"):
        evaluation.fit_inductive_gcn(
            cx,
            cy,
            mass,
            shared_graph,
            (shared_graph, val_mask),
            train_mask=torch.tensor([True, True, True, False, False]),
            settings=recipe(),
            folder=tmp_path / "first",
        )


def test_none_train_mask_preserves_legacy_cache_digest():
    cx, cy, mass, training, validation = inputs()
    legacy = hashlib.sha256()
    for name, value in (("cx", cx), ("cy", cy), ("mass", mass), ("training_adjacency", None)):
        evaluation._update_tensor_digest(legacy, name, value)
    for name, (supplied, mask) in (("train", (training, None)), ("validation", validation)):
        for key in ("x", "y", "adj"):
            evaluation._update_tensor_digest(legacy, name + "." + key, supplied[key])
        evaluation._update_tensor_digest(legacy, name + ".mask", mask)
    assert evaluation._input_digest(cx, cy, mass, None, training, validation, None) == legacy.hexdigest()
