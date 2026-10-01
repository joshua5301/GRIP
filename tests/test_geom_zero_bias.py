"""CPU-only checks for a bias-only control of GEOM initialization."""

import json

import pytest
import torch
import torch.nn.functional as F

import src.evaluation as evaluation
from src.io import _fingerprint
from src.models import GCN


@pytest.fixture(scope="module", autouse=True)
def small_cpu_jobs():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def inputs(folder):
    generator = torch.Generator().manual_seed(47)
    graph = dict(
        x=torch.randn(8, 5, generator=generator),
        y=torch.arange(8) % 2,
        adj=torch.eye(8).to_sparse_csr(),
    )
    return dict(
        cx=torch.randn(3, 5, generator=generator),
        cy=torch.randn(3, 2, generator=generator).softmax(1),
        mass=torch.tensor([0.2, 0.3, 0.5]),
        graph=graph,
        q=torch.randn(8, 2, generator=generator).softmax(1),
        masks=dict(train=torch.arange(8) < 4, val=torch.arange(8) >= 4),
        seed=19, epochs=4, eval_every=1, hidden=4, dropout=0.3,
        lr=0.01, weight_decay=0.00005, folder=folder,
    )


@pytest.mark.parametrize("dimensions,layers,seed", [
    ((5, 4, 2), 2, 0),
    ((3703, 256, 6), 2, 2),
    ((7, 5, 3), 3, 73),
])
def test_zero_bias_keeps_all_geom_weights_and_post_initialization_rng(dimensions, layers, seed):
    nin, hidden, classes = dimensions
    baseline = GCN(nin, hidden, classes, layers)
    evaluation._initialize_geom_uniform(baseline, seed)
    expected_rng = torch.random.get_rng_state().clone()
    expected_next_dropout = F.dropout(torch.ones(64), p=0.3, training=True)

    # Different constructor draws must be discarded in the control too.
    torch.manual_seed(seed + 17)
    control = GCN(nin, hidden, classes, layers)
    evaluation._initialize_geom_uniform_zero_bias(control, seed)
    assert torch.equal(torch.random.get_rng_state(), expected_rng)
    for original, zero_bias in zip(baseline.layers, control.layers, strict=True):
        assert torch.equal(original.lin.weight, zero_bias.lin.weight)
        assert bool((original.bias != 0).any())
        assert torch.equal(zero_bias.bias, torch.zeros_like(zero_bias.bias))
    # Consuming all original bias draws also preserves the next dropout draw.
    assert torch.equal(F.dropout(torch.ones(64), p=0.3, training=True), expected_next_dropout)


@pytest.mark.parametrize("old_initialization", ["pyg", "geom_uniform"])
def test_new_policy_has_distinct_cache_and_preserves_existing_recipe_identity(
    tmp_path, monkeypatch, old_initialization,
):
    kwargs = inputs(tmp_path / "old")
    evaluation.fit_gcn_diagnostic(**kwargs, initialization=old_initialization)
    old_folder = kwargs["folder"]
    old_files = {path.name: path.read_bytes() for path in old_folder.iterdir()}
    old_cache = json.loads(old_files["seed_19.json"])
    # Existing optional policies retain precisely the pre-control identity.
    expected_settings = {key: kwargs[key] for key in
                         ("epochs", "eval_every", "hidden", "dropout", "lr", "weight_decay")}
    if old_initialization != "pyg":
        expected_settings["initialization"] = old_initialization
    assert old_cache["recipe"]["settings"] == expected_settings
    assert old_cache["fingerprint"] == _fingerprint(old_cache["recipe"])

    with pytest.raises(ValueError, match="Cached student differs"):
        evaluation.fit_gcn_diagnostic(**kwargs, initialization="geom_uniform_zero_bias")
    assert {path.name: path.read_bytes() for path in old_folder.iterdir()} == old_files

    calls = []
    original_initializer = evaluation._initialize_geom_uniform_zero_bias

    def record_control(model, seed):
        original_initializer(model, seed)
        assert all(torch.count_nonzero(layer.bias).item() == 0 for layer in model.layers)
        calls.append(seed)

    monkeypatch.setattr(evaluation, "_initialize_geom_uniform_zero_bias", record_control)
    kwargs["folder"] = tmp_path / "new"
    result = evaluation.fit_gcn_diagnostic(**kwargs, initialization="geom_uniform_zero_bias")
    assert calls == [kwargs["seed"]]
    new_cache = json.loads((kwargs["folder"] / "seed_19.json").read_text())
    expected_recipe = dict(old_cache["recipe"], settings=dict(
        expected_settings, initialization="geom_uniform_zero_bias",
    ))
    assert new_cache["recipe"] == expected_recipe
    assert new_cache["fingerprint"] == _fingerprint(expected_recipe)
    assert new_cache["fingerprint"] != old_cache["fingerprint"]
    assert new_cache["result"] == result

    with pytest.raises(ValueError, match="Cached student differs"):
        evaluation.fit_gcn_diagnostic(**kwargs, initialization=old_initialization)

    def forbid_training(*args, **kwargs):
        raise AssertionError("An identical zero-bias recipe must reuse its cache")

    monkeypatch.setattr(evaluation, "GCN", forbid_training)
    assert evaluation.fit_gcn_diagnostic(**kwargs, initialization="geom_uniform_zero_bias") == result


def test_existing_policies_do_not_enter_the_zero_bias_initializer(tmp_path, monkeypatch):
    def forbid_control(*args, **kwargs):
        raise AssertionError("The new control must not change existing trajectories")

    monkeypatch.setattr(evaluation, "_initialize_geom_uniform_zero_bias", forbid_control)
    evaluation.fit_gcn_diagnostic(**inputs(tmp_path / "default"))
    evaluation.fit_gcn_diagnostic(**inputs(tmp_path / "geom"), initialization="geom_uniform")
