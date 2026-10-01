"""CPU replay of a frozen student on one explicit held-out test split."""

import hashlib
import json

import pytest
import torch
import torch.nn.functional as F

import src.evaluation as evaluation
import src.student_routes as routes
from src.evaluation import _update_tensor_digest
from src.io import _fingerprint
from src.models import GCN


@pytest.fixture(scope="module", autouse=True)
def small_cpu_jobs():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def inputs(tmp_path, test_only=True, layout="csr"):
    model = GCN(2, 3, 3, 2, dropout=0.4)
    with torch.no_grad():
        model.layers[0].lin.weight.copy_(torch.tensor([[1.0, -0.5], [-0.3, 1.0], [0.4, 0.7]]))
        model.layers[0].bias.copy_(torch.tensor([0.2, -0.1, 0.05]))
        model.layers[1].lin.weight.copy_(torch.tensor([[0.9, -0.2, 0.1], [-0.1, 0.7, 0.4], [0.2, 0.1, -0.5]]))
        model.layers[1].bias.copy_(torch.tensor([-0.05, 0.1, 0.2]))
    selected = tmp_path / "seed_17_selected.pt"
    torch.save(dict(epoch=11, fingerprint="frozen-validation-source", model_state=model.state_dict()), selected)
    x = torch.tensor([[2.0, 0.1], [0.2, 1.5], [-0.4, 0.9], [1.2, -0.2], [0.3, 1.9]])
    s = 0.6 * torch.eye(5) + 0.2 * torch.eye(5).roll(1, 1) + 0.2 * torch.eye(5).roll(-1, 1)
    adjacency = {"dense": lambda: s, "coo": s.to_sparse_coo, "csr": s.to_sparse_csr}[layout]()
    h = s @ (s @ x)
    test = torch.tensor([False, False, False, True, True])
    masks = dict(test=test)
    y = torch.tensor([-999, 0, 1, 2, 1])
    if test_only:
        y[:3] = torch.tensor([-999, 10_000_000, -333])
    else:
        masks["val"] = torch.tensor([False, True, True, False, False])
    kwargs = dict(selected_path=selected, graph=dict(x=x, adj=adjacency, y=y), propagated=h,
                  masks=masks, settings=dict(hidden=3, dropout=0.4, lr=0.003),
                  output_path=tmp_path / "routes.json", seed=17)
    return model, kwargs


def reference_scores(model, graph, propagated, masks):
    """Explicit two-layer X/S and S²X equations, independent of _forward."""
    w1, w2 = (layer.lin.weight.detach() for layer in model.layers)
    b1, b2 = (layer.bias.detach() for layer in model.layers)
    s = graph["adj"].to_dense()
    gcn = F.log_softmax(s @ (F.relu(s @ (graph["x"] @ w1.T) + b1) @ w2.T) + b2, dim=1)
    mlp = F.log_softmax(F.relu(propagated @ w1.T + b1) @ w2.T + b2, dim=1)
    result = {}
    for route, probability in (("gcn", gcn), ("mlp", mlp)):
        for name, mask in masks.items():
            target = graph["y"][mask]
            result[f"{route}_{name}_acc"] = 100 * float((probability[mask].argmax(1) == target).double().mean())
            result[f"{route}_{name}_ce"] = float(F.nll_loss(probability[mask], target))
    return result


@pytest.mark.parametrize("layout", ["dense", "coo", "csr"])
def test_test_only_matches_explicit_graph_and_sgc_equations_without_training(tmp_path, monkeypatch, layout):
    model, kwargs = inputs(tmp_path, layout=layout)
    expected = reference_scores(model, kwargs["graph"], kwargs["propagated"], kwargs["masks"])

    def forbid_training(*args, **kwargs):
        raise AssertionError("Test replay must not train or select an epoch")

    monkeypatch.setattr(torch.optim, "Adam", forbid_training)
    monkeypatch.setattr(evaluation, "fit_gcn_diagnostic", forbid_training)
    original_train = torch.nn.Module.train

    def evaluation_only(self, mode=True):
        assert mode is False, "Test replay must only enter evaluation mode"
        return original_train(self, mode)

    monkeypatch.setattr(torch.nn.Module, "train", evaluation_only)
    rng = torch.random.get_rng_state().clone()
    result = routes.replay_routes(**kwargs, test_only=True)
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert result["epoch"] == 11 and result["selection"] == routes.SELECTION
    assert not any("val" in key or "train" in key for key in result)
    for key, value in expected.items():
        assert result[key] == pytest.approx(value, abs=1e-6)
    cache = json.loads(kwargs["output_path"].read_text())
    assert cache["recipe"]["test_only"] is True and cache["recipe"]["test_enabled"] is True
    # The class count is read from weights even with enormous unused labels.
    assert cache["recipe"]["architecture"] == dict(nin=2, hidden=3, nout=3, layers=2)
    assert cache["recipe"]["source_fingerprint"] == "frozen-validation-source"


def test_only_requested_labels_are_read_and_outside_sentinels_do_not_change_cache(tmp_path, monkeypatch):
    _, kwargs = inputs(tmp_path)
    allowed = kwargs["masks"]["test"]
    raw = kwargs["graph"]["y"]
    reads = []

    class GuardedLabels:
        def __getitem__(self, mask):
            assert mask is allowed, "Only the requested test labels may be accessed"
            reads.append(mask)
            return raw[mask]

    kwargs["graph"]["y"] = GuardedLabels()
    first = routes.replay_routes(**kwargs, test_only=True)
    first_cache = kwargs["output_path"].read_bytes()
    raw[~allowed] = torch.tensor([1_000_000_000, -999_999, 456_789])

    def forbid_forward(*args, **kwargs):
        raise AssertionError("A valid cache must not serve the model again")

    monkeypatch.setattr(routes, "_forward", forbid_forward)
    monkeypatch.setattr(routes, "GCN", forbid_forward)
    rng = torch.random.get_rng_state().clone()
    assert routes.replay_routes(**kwargs, test_only=True) == first
    assert kwargs["output_path"].read_bytes() == first_cache
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert len(reads) == 2


@pytest.mark.parametrize("change", ["x", "adj", "h", "weights", "epoch", "source", "settings", "mask", "labels", "seed"])
def test_test_only_cache_rejects_changed_inputs_and_selected_provenance(tmp_path, change):
    _, kwargs = inputs(tmp_path)
    routes.replay_routes(**kwargs, test_only=True)
    original_cache = kwargs["output_path"].read_bytes()
    if change in ("weights", "epoch", "source"):
        saved = torch.load(kwargs["selected_path"], weights_only=False)
        if change == "weights":
            saved["model_state"]["layers.0.lin.weight"] += 0.01
        elif change == "epoch":
            saved["epoch"] += 1
        else:
            saved["fingerprint"] = "another-selected-source"
        torch.save(saved, kwargs["selected_path"])
    elif change == "x":
        kwargs["graph"]["x"] += 0.01
    elif change == "adj":
        kwargs["graph"]["adj"] = torch.eye(5).to_sparse_csr()
    elif change == "h":
        kwargs["propagated"] += 0.01
    elif change == "settings":
        kwargs["settings"]["lr"] = 0.01
    elif change == "mask":
        kwargs["masks"]["test"] = torch.tensor([3], dtype=torch.long)
    elif change == "labels":
        kwargs["graph"]["y"][kwargs["masks"]["test"]] = torch.tensor([1, 2])
    else:
        kwargs["seed"] += 1
    with pytest.raises(ValueError, match="Cached route replay differs"):
        routes.replay_routes(**kwargs, test_only=True)
    assert kwargs["output_path"].read_bytes() == original_cache


def test_changed_test_labels_do_not_reselect_the_saved_epoch(tmp_path):
    _, kwargs = inputs(tmp_path)
    first = routes.replay_routes(**kwargs, test_only=True)
    kwargs["graph"]["y"][kwargs["masks"]["test"]] = torch.tensor([0, 0])
    kwargs["output_path"] = tmp_path / "changed_labels.json"
    second = routes.replay_routes(**kwargs, test_only=True)
    assert first["epoch"] == second["epoch"] == 11
    assert first["selection"] == second["selection"] == routes.SELECTION


def test_default_recipe_and_explicit_false_preserve_legacy_fingerprint_and_scores(tmp_path, monkeypatch):
    model, kwargs = inputs(tmp_path, test_only=False)
    state = model.state_dict()
    digest = hashlib.sha256()
    for name, value in (("graph.x", kwargs["graph"]["x"]), ("graph.adj", kwargs["graph"]["adj"]),
                        ("propagated", kwargs["propagated"])):
        _update_tensor_digest(digest, name, value, lambda: False)
    for name, value in sorted(state.items()):
        _update_tensor_digest(digest, "state." + name, value, lambda: False)
    for name, mask in sorted(kwargs["masks"].items()):
        _update_tensor_digest(digest, "mask." + name, mask, lambda: False)
        _update_tensor_digest(digest, "labels." + name, kwargs["graph"]["y"][mask], lambda: False)
    legacy_recipe = dict(
        version=1, seed=17, settings=kwargs["settings"], architecture=dict(nin=2, hidden=3, nout=3, layers=2),
        epoch=11, source_fingerprint="frozen-validation-source", selection=routes.SELECTION,
        routes=dict(gcn="original X and normalized adjacency", mlp="supplied S²X; no adjacency"),
        test_enabled=True, input_digest=digest.hexdigest(), torch_version=torch.__version__,
    )
    result = routes.replay_routes(**kwargs)
    cached = json.loads(kwargs["output_path"].read_text())
    assert cached["recipe"] == legacy_recipe
    assert cached["fingerprint"] == _fingerprint(legacy_recipe)
    assert "test_only" not in cached["recipe"]
    for key, value in reference_scores(model, kwargs["graph"], kwargs["propagated"], kwargs["masks"]).items():
        assert result[key] == pytest.approx(value, abs=1e-6)

    def forbid_forward(*args, **kwargs):
        raise AssertionError("Explicit False must reuse the legacy cache")

    monkeypatch.setattr(routes, "_forward", forbid_forward)
    assert routes.replay_routes(**kwargs, test_only=False) == result


@pytest.mark.parametrize("flag,mask_names", [
    (True, []), (True, ["val"]), (True, ["val", "test"]), (True, ["train", "test"]),
    (False, ["test"]), (False, ["train", "val"]),
    (1, ["test"]), (0, ["val"]), (None, ["test"]), ("true", ["test"]),
])
def test_invalid_mask_policy_or_flag_is_rejected_before_checkpoint_access(tmp_path, monkeypatch, flag, mask_names):
    _, kwargs = inputs(tmp_path)
    kwargs["masks"] = {name: torch.ones(5, dtype=torch.bool) for name in mask_names}

    def forbid_load(*args, **kwargs):
        raise AssertionError("Invalid mask policy/flag must fail before loading a checkpoint")

    monkeypatch.setattr(torch, "load", forbid_load)
    with pytest.raises(ValueError):
        routes.replay_routes(**kwargs, test_only=flag)
    assert not kwargs["output_path"].exists()


@pytest.mark.parametrize("mask", [None, torch.zeros(5, dtype=torch.bool),
                                   torch.ones(4, dtype=torch.bool), torch.tensor([[3, 4]]),
                                   torch.tensor([3.0, 4.0])])
def test_invalid_test_split_mask_is_rejected_without_publishing(tmp_path, mask):
    _, kwargs = inputs(tmp_path)
    kwargs["masks"] = dict(test=mask)
    with pytest.raises(ValueError):
        routes.replay_routes(**kwargs, test_only=True)
    assert not kwargs["output_path"].exists()
