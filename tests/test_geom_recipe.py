import json
import math

import pandas as pd
import pytest
import torch
from torch_geometric import seed_everything

import src.evaluation as evaluation
from src.io import _fingerprint
from src.models import GCN


class OfficialGraphConvolution(torch.nn.Module):
    """Direct constructor/reset reference from GEOM models/gcn.py, lines 14–26.

    Source commit: 3cc01601633a14fc8829fe2e62f2d78fe83ca9a3. GEOM's GCN
    constructor adds layers in order; eval_condg.py does not reset them again.
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = torch.nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1 / math.sqrt(self.weight.T.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)


def inputs():
    generator = torch.Generator().manual_seed(51)
    cx = torch.randn(4, 5, generator=generator)
    cy = torch.randn(4, 2, generator=generator).softmax(1)
    mass = torch.tensor([0.1, 0.2, 0.3, 0.4])
    graph = dict(x=torch.randn(12, 5, generator=generator), y=torch.arange(12) % 2,
                 adj=torch.eye(12).to_sparse_csr())
    q = torch.randn(12, 2, generator=generator).softmax(1)
    masks = dict(train=torch.arange(12) < 4,
                 val=(torch.arange(12) >= 4) & (torch.arange(12) < 8))
    return cx, cy, mass, graph, q, masks


def settings():
    return dict(seed=73, epochs=6, eval_every=1, hidden=4, dropout=0.0,
                lr=0.01, weight_decay=0.001)


@pytest.mark.parametrize("dimensions,seed", [((5, 4, 2), 3), ((1433, 256, 7), 211)])
def test_geom_initializer_matches_official_cpu_constructor_weights_biases_and_rng(dimensions, seed):
    nin, hidden, classes = dimensions
    torch.manual_seed(seed)
    reference = [OfficialGraphConvolution(nin, hidden), OfficialGraphConvolution(hidden, classes)]
    expected_rng = torch.random.get_rng_state().clone()
    # PyG constructor consumes several different Glorot draws; GEOM must ignore them.
    torch.manual_seed(seed + 19)
    actual = GCN(nin, hidden, classes, 2, dropout=0.0)
    evaluation._initialize_geom_uniform(actual, seed)
    assert torch.equal(torch.random.get_rng_state(), expected_rng)
    for layer, official in zip(actual.layers, reference, strict=True):
        assert torch.equal(layer.lin.weight, official.weight.T)
        assert torch.equal(layer.bias, official.bias)
        assert bool((layer.bias != 0).any())


def test_default_recipe_fingerprint_and_selected_state_preserve_original_evaluator(tmp_path):
    values, options = inputs(), settings()
    cx, cy, mass, graph, q, masks = values
    result = evaluation.fit_gcn_diagnostic(*values, folder=tmp_path, **options)
    saved = json.loads((tmp_path / f"seed_{options['seed']}.json").read_text())
    plain_settings = {key: value for key, value in options.items() if key != "seed"}
    expected_recipe = dict(
        version=2, seed=options["seed"], settings=plain_settings, layers=2,
        optimizer="Adam; reset to lr/10 at epochs//2", weighting="normalized supplied mass",
        selection="first maximum validation accuracy", test_enabled=False,
        test_evaluation="selected weights once", torch_version=torch.__version__,
        input_digest=evaluation._input_digest(cx, cy, mass, graph, q, masks, None, lambda: False),
    )
    assert saved["recipe"] == expected_recipe and saved["fingerprint"] == _fingerprint(expected_recipe)
    assert "lr_schedule" not in saved["recipe"]["settings"]
    assert "initialization" not in saved["recipe"]["settings"]

    # The pre-option evaluator's model construction, optimizer reset, loss and
    # strict validation selection give the very same selected parameters.
    seed_everything(options["seed"])
    model = GCN(cx.shape[1], options["hidden"], cy.shape[1], 2, options["dropout"])
    optimizer = torch.optim.Adam(model.parameters(), lr=options["lr"], weight_decay=options["weight_decay"])
    weights, best, selected = mass / mass.sum(), -float("inf"), None
    for epoch in range(1, options["epochs"] + 1):
        if epoch == options["epochs"] // 2:
            optimizer = torch.optim.Adam(model.parameters(), lr=options["lr"] * 0.1,
                                         weight_decay=options["weight_decay"])
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = -(weights[:, None] * cy * evaluation._forward(model, cx)).sum()
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            probability = evaluation._forward(model, graph["x"], graph["adj"])
            accuracy = float((probability[masks["val"]].argmax(1) == graph["y"][masks["val"]]).double().mean())
        if accuracy > best:
            best, selected = accuracy, {key: value.clone() for key, value in model.state_dict().items()}
    checkpoint = torch.load(tmp_path / f"seed_{options['seed']}_selected.pt", weights_only=False)
    for key, expected in selected.items():
        assert torch.equal(checkpoint["model_state"][key], expected)
    assert result["val_acc"] == 100 * best


def test_constant_adam_keeps_moments_and_diverges_only_when_half_reset_occurs(tmp_path, monkeypatch):
    original = torch.optim.Adam
    traces, current = {}, None

    class CaptureAdam(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            traces[current]["initial_lrs"].append(self.param_groups[0]["lr"])

        def step(self, *args, **kwargs):
            value = super().step(*args, **kwargs)
            traces[current]["updates"].append([parameter.detach().clone()
                                               for parameter in self.param_groups[0]["params"]])
            return value

    monkeypatch.setattr(torch.optim, "Adam", CaptureAdam)
    for schedule in ("half_reset", "constant"):
        current = schedule
        traces[current] = dict(initial_lrs=[], updates=[])
        evaluation.fit_gcn_diagnostic(*inputs(), folder=tmp_path / schedule,
                                      initialization="geom_uniform", lr_schedule=schedule, **settings())
    assert traces["half_reset"]["initial_lrs"] == [0.01, 0.001]
    assert traces["constant"]["initial_lrs"] == [0.01]
    for a, b in zip(traces["half_reset"]["updates"][:2], traces["constant"]["updates"][:2], strict=True):
        assert all(torch.equal(left, right) for left, right in zip(a, b, strict=True))
    assert any(not torch.equal(left, right) for left, right in
               zip(traces["half_reset"]["updates"][2], traces["constant"]["updates"][2], strict=True))
    assert len(traces["constant"]["updates"]) == settings()["epochs"]


@pytest.mark.parametrize("changed", [dict(lr_schedule="constant"), dict(initialization="geom_uniform")])
def test_changed_schedule_or_initializer_is_fingerprinted_and_cannot_reuse_old_cache(tmp_path, changed):
    values, options = inputs(), settings()
    evaluation.fit_gcn_diagnostic(*values, folder=tmp_path / "old", **options)
    path = tmp_path / "old" / f"seed_{options['seed']}.json"
    original = path.read_bytes()
    with pytest.raises(ValueError, match="Cached student differs"):
        evaluation.fit_gcn_diagnostic(*values, folder=tmp_path / "old", **changed, **options)
    assert path.read_bytes() == original
    result = evaluation.fit_gcn_diagnostic(*values, folder=tmp_path / "new", **changed, **options)
    new_cache = json.loads((tmp_path / "new" / path.name).read_text())
    assert all(new_cache["recipe"]["settings"][key] == value for key, value in changed.items())
    assert new_cache["fingerprint"] != json.loads(original)["fingerprint"]
    assert new_cache["result"] == result


def test_geom_recipe_uses_validation_only_epoch_search_and_evaluates_test_once(tmp_path, monkeypatch):
    cx, cy, mass, graph, q, masks = inputs()
    masks["test"] = torch.arange(12) >= 8
    options = dict(settings(), lr_schedule="constant", initialization="geom_uniform")
    calls, original = [], evaluation._mask_metrics

    def record(probability, labels, teacher, mask):
        if mask is masks["test"]:
            calls.append("test")
        return original(probability, labels, teacher, mask)

    monkeypatch.setattr(evaluation, "_mask_metrics", record)
    first = evaluation.fit_gcn_diagnostic(cx, cy, mass, graph, q, masks, folder=tmp_path / "first", **options)
    assert calls == ["test"]
    changed = dict(graph, y=graph["y"].clone())
    changed["y"][masks["test"]] = 1 - changed["y"][masks["test"]]
    second = evaluation.fit_gcn_diagnostic(cx, cy, mass, changed, q, masks, folder=tmp_path / "second", **options)
    assert calls == ["test", "test"]
    assert first["epoch"] == second["epoch"] and first["val_acc"] == second["val_acc"]
    assert first["test_acc"] + second["test_acc"] == 100
    assert not any(key.startswith("last_test") for key in first)
    for name in ("first", "second"):
        history = pd.read_csv(tmp_path / name / f"seed_{options['seed']}_epochs.csv")
        assert len(history) == options["epochs"]
        assert not any("test" in column for column in history.columns)
    a = torch.load(tmp_path / "first" / f"seed_{options['seed']}_selected.pt", weights_only=False)
    b = torch.load(tmp_path / "second" / f"seed_{options['seed']}_selected.pt", weights_only=False)
    assert all(torch.equal(a["model_state"][key], b["model_state"][key]) for key in a["model_state"])


@pytest.mark.parametrize("changed", [dict(lr_schedule="invalid"), dict(initialization="invalid")])
def test_invalid_student_policies_are_rejected_before_cache_or_training(tmp_path, changed):
    folder = tmp_path / "absent"
    with pytest.raises(ValueError):
        evaluation.fit_gcn_diagnostic(*inputs(), folder=folder, **changed, **settings())
    assert not folder.exists()
