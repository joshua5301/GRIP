import json

import pytest
import torch
import torch.nn.functional as F

import src.student_routes as routes
from src.evaluation import _forward
from src.models import GCN


@pytest.fixture(scope="module", autouse=True)
def small_cpu_jobs():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def inputs(tmp_path, final=True):
    model = GCN(2, 3, 2, 2, 0.5)
    with torch.no_grad():
        model.layers[0].lin.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]))
        model.layers[1].lin.weight.copy_(torch.tensor([[1.0, 0.1, 0.2], [0.1, 1.0, 0.3]]))
        for layer in model.layers:
            layer.bias.zero_()
    selected = tmp_path / "seed_7_selected.pt"
    torch.save(dict(epoch=3, fingerprint="selected-source", model_state=model.state_dict()), selected)
    x = torch.tensor([[2.0, 0.1], [0.1, 2.0]] * 3)
    adjacency = (0.7 * torch.eye(6) + 0.3 * torch.eye(6).roll(1, 1)).to_sparse_csr()
    h = torch.sparse.mm(adjacency, torch.sparse.mm(adjacency, x))
    # Invalid unused train labels ensure that serving does not need them.
    labels = torch.tensor([-999, -999, 0, 1, 0, 1])
    masks = dict(val=torch.tensor([False, False, True, True, False, False]))
    if final:
        masks["test"] = torch.tensor([False, False, False, False, True, True])
    kwargs = dict(selected_path=selected, graph=dict(x=x, y=labels, adj=adjacency), propagated=h,
                  masks=masks, settings=dict(hidden=3, dropout=0.5), output_path=tmp_path / "routes.json", seed=7)
    return model, kwargs


def test_both_routes_match_direct_forward_with_identical_weights(tmp_path, monkeypatch):
    model, kwargs = inputs(tmp_path)
    model.eval()

    def forbid_training(*args, **kwargs):
        raise AssertionError("Replay must never train a student")

    monkeypatch.setattr(torch.optim, "Adam", forbid_training)
    rng = torch.random.get_rng_state().clone()
    result = routes.replay_routes(**kwargs)
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert result["epoch"] == 3
    assert result["selection"] == "same weights at GCN validation-selected epoch"
    with torch.no_grad():
        probabilities = dict(gcn=_forward(model, kwargs["graph"]["x"], kwargs["graph"]["adj"]),
                             mlp=_forward(model, kwargs["propagated"], None))
    for route, probability in probabilities.items():
        for name, mask in kwargs["masks"].items():
            target = kwargs["graph"]["y"][mask]
            expected_acc = 100 * float((probability[mask].argmax(1) == target).double().mean())
            assert result[f"{route}_{name}_acc"] == expected_acc
            assert result[f"{route}_{name}_ce"] == float(F.nll_loss(probability[mask], target))
    cached = json.loads(kwargs["output_path"].read_text())
    assert cached["result"] == result
    assert cached["recipe"]["architecture"] == dict(nin=2, hidden=3, nout=2, layers=2)

    def forbid_forward(*args, **kwargs):
        raise AssertionError("An identical replay must use the safe cache")

    monkeypatch.setattr(routes, "_forward", forbid_forward)
    assert routes.replay_routes(**kwargs) == result


def test_val_only_does_not_read_train_or_test_labels(tmp_path):
    _, kwargs = inputs(tmp_path, final=False)
    allowed = kwargs["masks"]["val"]
    raw = kwargs["graph"]["y"]

    class GuardedLabels:
        def __getitem__(self, mask):
            assert mask is allowed, "Validation serving must read only requested labels"
            return raw[mask]

    kwargs["graph"]["y"] = GuardedLabels()
    first = routes.replay_routes(**kwargs)
    assert not any("test" in name for name in first)
    assert json.loads(kwargs["output_path"].read_text())["recipe"]["test_enabled"] is False
    raw[4:] = 1 - raw[4:]
    assert routes.replay_routes(**kwargs) == first


def test_test_labels_do_not_select_an_epoch_or_route(tmp_path):
    _, kwargs = inputs(tmp_path)
    first = routes.replay_routes(**kwargs)
    changed = kwargs["graph"]["y"].clone()
    changed[kwargs["masks"]["test"]] = 1 - changed[kwargs["masks"]["test"]]
    kwargs["graph"] = dict(kwargs["graph"], y=changed)
    kwargs["output_path"] = tmp_path / "changed_labels.json"
    second = routes.replay_routes(**kwargs)
    assert first["epoch"] == second["epoch"] == 3
    assert first["selection"] == second["selection"] == routes.SELECTION
    for route in ("gcn", "mlp"):
        assert first[f"{route}_val_acc"] == second[f"{route}_val_acc"]
        assert first[f"{route}_val_ce"] == second[f"{route}_val_ce"]
        assert first[f"{route}_test_acc"] + second[f"{route}_test_acc"] == 100


@pytest.mark.parametrize("change", ["h", "state", "settings", "epoch", "graph", "adj", "mask", "labels", "seed"])
def test_replay_cache_rejects_changed_scientific_inputs(tmp_path, change):
    _, kwargs = inputs(tmp_path)
    routes.replay_routes(**kwargs)
    if change == "h":
        kwargs["propagated"] = kwargs["propagated"] + 0.01
    elif change in ("state", "epoch"):
        saved = torch.load(kwargs["selected_path"], weights_only=False)
        if change == "state":
            saved["model_state"]["layers.0.lin.weight"] += 0.01
        else:
            saved["epoch"] += 1
        torch.save(saved, kwargs["selected_path"])
    elif change == "settings":
        kwargs["settings"]["dropout"] = 0.2
    elif change == "graph":
        kwargs["graph"]["x"] += 0.01
    elif change == "adj":
        kwargs["graph"]["adj"] = torch.eye(6).to_sparse_csr()
    elif change == "mask":
        kwargs["masks"]["test"] = kwargs["masks"]["test"].roll(-1)
    elif change == "labels":
        kwargs["graph"]["y"][4:] = 1 - kwargs["graph"]["y"][4:]
    else:
        kwargs["seed"] += 1
    with pytest.raises(ValueError, match="Cached route replay differs"):
        routes.replay_routes(**kwargs)


def test_interruption_does_not_publish_a_route_cache(tmp_path):
    _, kwargs = inputs(tmp_path)
    with pytest.raises(InterruptedError, match="Student route replay interrupted"):
        routes.replay_routes(**kwargs, stop=lambda: True)
    assert not kwargs["output_path"].exists()
