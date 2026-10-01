import json

import pandas as pd
import pytest
import torch

import src.evaluation as evaluation
from src.models import GCN


@pytest.fixture(scope="module", autouse=True)
def small_cpu_jobs():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def inputs(folder, final=True):
    x = torch.tensor([[2.0, 0.1], [0.1, 2.0]] * 3)
    y = torch.arange(6) % 2
    masks = dict(train=torch.arange(6) < 2, val=(torch.arange(6) >= 2) & (torch.arange(6) < 4))
    if final:
        masks["test"] = torch.arange(6) >= 4
    return dict(
        cx=x[:2].clone(), cy=torch.eye(2), mass=torch.tensor([0.2, 0.8]),
        graph=dict(x=x, y=y, adj=torch.eye(6).to_sparse_csr()), q=torch.eye(2)[y], masks=masks,
        seed=0, epochs=4, eval_every=1, hidden=4, dropout=0.0, lr=0.01, weight_decay=0.0005,
        folder=folder, training_adjacency=torch.eye(2).to_sparse_csr(),
    )


def test_cache_hit_returns_scores_without_retraining(tmp_path, monkeypatch):
    kwargs = inputs(tmp_path)
    first = evaluation.fit_gcn_diagnostic(**kwargs)
    cached = json.loads((tmp_path / "seed_0.json").read_text())
    assert cached["result"] == first
    assert cached["recipe"]["seed"] == 0
    assert cached["recipe"]["settings"]["epochs"] == 4
    assert len(cached["recipe"]["input_digest"]) == 64
    assert cached["recipe"]["test_enabled"] is True

    def forbid_training(*args, **kwargs):
        raise AssertionError("A valid cache must not retrain")

    monkeypatch.setattr(evaluation, "GCN", forbid_training)
    assert evaluation.fit_gcn_diagnostic(**kwargs) == first


@pytest.mark.parametrize("change", [
    "cx", "cy", "mass", "graph_x", "graph_y", "graph_adj_values", "graph_adj_indices",
    "q", "train_mask", "val_mask", "test_mask", "synthetic_adj", "adjacency_layout",
    "settings", "seed", "test_enabled",
])
def test_cache_rejects_changed_inputs_and_recipe(tmp_path, change):
    kwargs = inputs(tmp_path)
    evaluation.fit_gcn_diagnostic(**kwargs)
    if change in ("cx", "cy", "mass", "q"):
        value = kwargs[change]
        kwargs[change] = value.flip(0) if change in ("cy", "mass") else value + 0.01
    elif change == "graph_x":
        kwargs["graph"]["x"] = kwargs["graph"]["x"] + 0.01
    elif change == "graph_y":
        kwargs["graph"]["y"] = 1 - kwargs["graph"]["y"]
    elif change == "graph_adj_values":
        kwargs["graph"]["adj"] = (torch.eye(6) * 0.9).to_sparse_csr()
    elif change == "graph_adj_indices":
        kwargs["graph"]["adj"] = torch.eye(6).roll(1, 1).to_sparse_csr()
    elif change.endswith("_mask"):
        name = change.removesuffix("_mask")
        kwargs["masks"][name] = kwargs["masks"][name].roll(1)
    elif change == "synthetic_adj":
        kwargs["training_adjacency"] = (torch.eye(2) * 0.9).to_sparse_csr()
    elif change == "adjacency_layout":
        kwargs["graph"]["adj"] = kwargs["graph"]["adj"].to_sparse_coo()
    elif change == "settings":
        kwargs["dropout"] = 0.2
    elif change == "seed":
        (tmp_path / "seed_1.json").write_bytes((tmp_path / "seed_0.json").read_bytes())
        kwargs["seed"] = 1
    else:
        del kwargs["masks"]["test"]
    with pytest.raises(ValueError, match="Cached student differs"):
        evaluation.fit_gcn_diagnostic(**kwargs)


def test_legacy_score_is_retrained_and_preserved(tmp_path):
    old = dict(seed=0, epoch=1, val_acc=999.0, test_acc=999.0)
    (tmp_path / "seed_0.json").write_text(json.dumps(old))
    old_history = "epoch,val_acc,test_acc\n1,999,999\n"
    (tmp_path / "seed_0_epochs.csv").write_text(old_history)
    result = evaluation.fit_gcn_diagnostic(**inputs(tmp_path))
    assert 0 <= result["val_acc"] <= 100
    archives = list(tmp_path.glob("seed_0_legacy_*.json"))
    assert len(archives) == 1
    assert json.loads(archives[0].read_text()) == old
    assert next(tmp_path.glob("seed_0_legacy_*_epochs.csv")).read_text() == old_history
    assert json.loads((tmp_path / "seed_0.json").read_text())["result"] == result
    assert "test" not in (tmp_path / "seed_0_epochs.csv").read_text()


def test_search_history_and_result_never_evaluate_test(tmp_path, monkeypatch):
    kwargs = inputs(tmp_path)
    testing_mask = kwargs["masks"].pop("test")
    original = evaluation._mask_metrics

    def track(log_probability, y, q, mask):
        assert mask is not testing_mask
        return original(log_probability, y, q, mask)

    monkeypatch.setattr(evaluation, "_mask_metrics", track)
    result = evaluation.fit_gcn_diagnostic(**kwargs)
    assert not any("test" in key for key in result)
    assert "test" not in (tmp_path / "seed_0_epochs.csv").read_text()
    assert json.loads((tmp_path / "seed_0.json").read_text())["recipe"]["test_enabled"] is False


def test_test_runs_once_at_validation_selected_weights(tmp_path, monkeypatch):
    kwargs = inputs(tmp_path / "first")
    testing_mask = kwargs["masks"]["test"]
    calls = []
    original = evaluation._mask_metrics

    def track(log_probability, y, q, mask):
        if mask is testing_mask:
            calls.append(mask)
        return original(log_probability, y, q, mask)

    monkeypatch.setattr(evaluation, "_mask_metrics", track)
    first = evaluation.fit_gcn_diagnostic(**kwargs)
    assert len(calls) == 1
    assert not any(key.startswith("last_test_") for key in first)
    history = pd.read_csv(tmp_path / "first" / "seed_0_epochs.csv")
    assert first["epoch"] == history.loc[history.val_acc.idxmax(), "epoch"]
    assert not any("test" in name for name in history.columns)

    selected = torch.load(tmp_path / "first" / "seed_0_selected.pt", weights_only=False)
    model = GCN(2, kwargs["hidden"], 2, 2, kwargs["dropout"])
    model.load_state_dict(selected["model_state"])
    model.eval()
    with torch.no_grad():
        prediction = evaluation._forward(model, kwargs["graph"]["x"], kwargs["graph"]["adj"])
    expected = original(prediction, kwargs["graph"]["y"], kwargs["q"], testing_mask)
    assert first["test_acc"] == expected["acc"]
    assert first["test_ce"] == expected["ce"]
    assert first["test_teacher_ce"] == expected["teacher_ce"]

    changed_labels = kwargs["graph"]["y"].clone()
    changed_labels[testing_mask] = 1 - changed_labels[testing_mask]
    kwargs["graph"] = dict(kwargs["graph"], y=changed_labels)
    kwargs["folder"] = tmp_path / "second"
    calls.clear()
    second = evaluation.fit_gcn_diagnostic(**kwargs)
    assert len(calls) == 1
    assert first["epoch"] == second["epoch"]
    assert first["val_acc"] == second["val_acc"]
    assert first["val_ce"] == second["val_ce"]
    assert first["test_acc"] + second["test_acc"] == 100
    assert (tmp_path / "first" / "seed_0_epochs.csv").read_bytes() == (
        tmp_path / "second" / "seed_0_epochs.csv"
    ).read_bytes()


def test_interruption_does_not_publish_partial_student(tmp_path, monkeypatch):
    halted = False
    original_step = torch.optim.Adam.step

    def step(*args, **kwargs):
        nonlocal halted
        result = original_step(*args, **kwargs)
        halted = True
        return result

    monkeypatch.setattr(torch.optim.Adam, "step", step)
    with pytest.raises(InterruptedError, match="Student evaluation interrupted"):
        evaluation.fit_gcn_diagnostic(**inputs(tmp_path), stop=lambda: halted)
    assert not list(tmp_path.glob("seed_0*"))


def test_stopped_legacy_migration_preserves_original(tmp_path):
    old = dict(seed=0, val_acc=999.0)
    path = tmp_path / "seed_0.json"
    path.write_text(json.dumps(old))
    with pytest.raises(InterruptedError, match="Student evaluation interrupted"):
        evaluation.fit_gcn_diagnostic(**inputs(tmp_path), stop=lambda: True)
    assert json.loads(path.read_text()) == old
    assert not list(tmp_path.glob("seed_0_legacy*"))
