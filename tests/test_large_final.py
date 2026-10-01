"""CPU integration checks for frozen large-data student test replay."""

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
import torch.nn.functional as F

import src.large_final as final
import src.research_loop as research_loop
import src.student_routes as routes
from src.evaluation import _forward
from src.inductive_evaluation import fit_inductive_gcn
from src.io import _fingerprint, save_json
from src.models import GCN


@pytest.fixture(scope="module", autouse=True)
def small_cpu_jobs():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def packed_graph(x, labels, adjacency=None):
    x = torch.tensor(x, dtype=torch.float32)
    adjacency = torch.eye(len(x)) if adjacency is None else torch.tensor(adjacency)
    return dict(x=x, y=torch.tensor(labels), adj=adjacency.to_sparse_csr())


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    settings = dict(epochs=4, eval_every=1, hidden=3, dropout=0.0,
                    lr=0.01, weight_decay=0.0005)
    train = packed_graph([[2., .1], [.1, 2.], [1.8, .2], [.2, 1.8]], [0, 1, 0, 1])
    validation = packed_graph([[1.5, .2], [.2, 1.5], [1.2, .1]], [0, 1, 0])
    testing = packed_graph(
        [[.2, 2.5], [2.5, .2], [.8, 1.4], [1.4, .8], [.1, 3.]],
        [1, 0, 1, 0, 1],
        [[.7, .3, 0., 0., 0.], [.3, .7, 0., 0., 0.],
         [0., 0., .5, .5, 0.], [0., 0., .5, .5, 0.], [0., 0., 0., 0., 1.]],
    )
    candidate = tmp_path / "pilot" / "candidate"
    folder = candidate / "validation" / f"step_20_{_fingerprint(settings)}"
    folder.mkdir(parents=True)
    save_json(dict(data_digest="training-validation-source"), candidate / "protocol.json")
    seeds = (910, 911)
    students = []
    for seed in seeds:
        result = fit_inductive_gcn(
            train["x"][:2], torch.eye(2), torch.tensor([.4, .6]), train,
            (validation, None), seed=seed, settings=settings, folder=folder,
            train_mask=torch.ones(len(train["x"]), dtype=torch.bool),
        )
        val_h = torch.sparse.mm(validation["adj"], validation["x"])
        val_h = torch.sparse.mm(validation["adj"], val_h)
        score = routes.replay_routes(
            folder / f"seed_{seed}_selected.pt", validation, val_h,
            {"val": torch.ones(len(val_h), dtype=torch.bool)}, settings,
            folder / f"seed_{seed}_validation_routes_v1.json", seed=seed,
        )
        students.append(dict(result, mlp_val_acc=score["mlp_val_acc"]))
    options = dict(dataset="flickr", ratio=.001, steps=20, epochs=4, hidden=3, dropout=0.)
    selection = dict(version=1, selection_basis="validation_only", options=options,
                     source_data_digest="training-validation-source",
                     required_source_digest=research_loop.implementation_provenance()["source_digest"],
                     validation_student_seeds=[0, 1, 2], validation_job_ids=["chosen-screen"])
    selection_path = tmp_path / "selection.json"
    save_json(selection, selection_path)
    state = SimpleNamespace(
        settings=settings, candidate=candidate, folder=folder, train=train,
        validation=validation, testing=testing, test_mask=None, seeds=seeds,
        students=students, selection=selection, selection_path=selection_path,
        selection_sha256=digest(selection_path), events=[], pilot_status="complete",
        validation_complete=False, test_loads=0, pilot_calls=0,
    )

    def fake_pilot(**kwargs):
        state.pilot_calls += 1
        state.events.append("validation-only-pilot")
        assert "testing" not in kwargs and "test" not in kwargs
        assert kwargs["report_routes"] is True
        assert tuple(kwargs["student_seeds"]) == state.seeds
        state.validation_complete = state.pilot_status == "complete"
        report = dict(status=state.pilot_status, student_recipe=dict(state.settings),
                      students=copy.deepcopy(state.students), teacher=dict(val_acc=75.))
        if state.pilot_status != "complete":
            report["reason"] = "bounded validation did not complete"
        return report, state.candidate

    def fake_prepare(dataset, data_dir, device):
        assert state.validation_complete, "Testing must follow all validation-only fits"
        state.test_loads += 1
        state.events.append("test-loader")
        # Every requested model must already be frozen before labels are exposed.
        root = (tmp_path / "final" / state.selection["options"]["dataset"]
                / f"selection_{state.selection_sha256[:16]}")
        for seed in state.seeds:
            assert (root / "condensation_0" / "selected_weights"
                    / f"seed_{seed}_selected.pt").exists()
        if dataset == "arxiv":
            graph = state.testing
            train_mask = torch.tensor([True, False, False, False, False])
            val_mask = torch.tensor([False, True, False, False, False])
            return graph, train_mask, (graph, val_mask), (graph, state.test_mask), graph["x"]
        return state.train, torch.ones(4, dtype=torch.bool), (state.validation, None), (
            state.testing, state.test_mask), state.train["x"]

    monkeypatch.setattr(final, "run_pilot", fake_pilot)
    monkeypatch.setattr(final, "_prepare_dataset", fake_prepare)

    def call(**kwargs):
        arguments = dict(selection_path=state.selection_path,
                         selection_sha256=state.selection_sha256,
                         output_dir=tmp_path / "final", condensation_seed=0,
                         student_seeds=state.seeds, pilot_output_dir=tmp_path / "pilot",
                         data_dir=tmp_path / "data", device="cpu", deadline_seconds=10.)
        arguments.update(kwargs)
        return final.run_final(**arguments)

    state.call = call
    state.native_hashes = {p.name: digest(p) for p in state.folder.iterdir() if p.is_file()}
    return state


def direct_scores(path, graph, mask, hidden):
    selected = torch.load(path, map_location="cpu", weights_only=False)
    with torch.random.fork_rng(devices=[]):
        model = GCN(graph["x"].shape[1], hidden, 2, 2, 0.)
    model.load_state_dict(selected["model_state"])
    model.eval()
    h = torch.sparse.mm(graph["adj"], torch.sparse.mm(graph["adj"], graph["x"]))
    output = {}
    with torch.no_grad():
        for name, x, adj in [("gcn", graph["x"], graph["adj"]), ("mlp", h, None)]:
            probability = _forward(model, x, adj)[mask]
            labels = graph["y"][mask]
            output[name + "_test_acc"] = 100 * float((probability.argmax(1) == labels).double().mean())
            output[name + "_test_ce"] = float(F.nll_loss(probability, labels))
    return output


@pytest.mark.parametrize("dataset", ["flickr", "reddit"])
def test_final_uses_own_test_graph_and_validation_selected_weights(experiment, dataset):
    state = experiment
    state.selection["options"]["dataset"] = dataset
    save_json(state.selection, state.selection_path)
    state.selection_sha256 = digest(state.selection_path)
    report, root = state.call()
    assert report["status"] == "complete"
    assert state.events == ["validation-only-pilot", "test-loader"]
    assert len(report["rows"]) == len(state.seeds)
    mask = torch.ones(len(state.testing["x"]), dtype=torch.bool)
    h = torch.sparse.mm(state.testing["adj"], torch.sparse.mm(state.testing["adj"], state.testing["x"]))
    cached_h = torch.load(root / "test_S2X.pt", map_location="cpu", weights_only=False)["h"]
    assert torch.equal(cached_h, h)
    for row in report["rows"]:
        seed = row["seed"]
        frozen = root / "condensation_0" / "selected_weights" / f"seed_{seed}_selected.pt"
        original = state.folder / f"seed_{seed}_selected.pt"
        assert digest(frozen) == digest(original)
        selected = torch.load(frozen, map_location="cpu", weights_only=False)
        assert row["epoch"] == selected["epoch"]
        expected = direct_scores(frozen, state.testing, mask, state.settings["hidden"])
        for key, value in expected.items():
            assert row[key] == pytest.approx(value, abs=1e-8)
        cache = json.loads((root / "condensation_0" / f"student_{seed}_test_routes.json").read_text())
        assert cache["recipe"]["test_only"] is True
        assert cache["recipe"]["test_enabled"] is True
        assert not any(name.endswith("_val_ce") for name in cache["result"])
    assert state.native_hashes == {p.name: digest(p) for p in state.folder.iterdir() if p.is_file()}


def test_arxiv_full_graph_uses_only_test_mask_labels(experiment):
    state = experiment
    state.selection["options"]["dataset"] = "arxiv"
    state.selection["options"]["ratio"] = .0005
    save_json(state.selection, state.selection_path)
    state.selection_sha256 = digest(state.selection_path)
    state.test_mask = torch.tensor([False, False, True, True, True])
    raw = state.testing["y"].clone()
    raw[:2] = 999  # No train/validation class inference is allowed during replay.

    class TestLabelsOnly:
        def __getitem__(self, mask):
            assert mask is state.test_mask, "Only the test mask may index labels"
            return raw[mask]

    state.testing["y"] = TestLabelsOnly()
    report, _ = state.call()
    assert report["status"] == "complete"
    assert all(row["epoch"] == student["epoch"]
               for row, student in zip(report["rows"], state.students))


def test_test_replay_never_optimizes_selects_epoch_or_changes_rng(experiment, monkeypatch):
    state = experiment

    def forbid_training(*args, **kwargs):
        raise AssertionError("Test replay must not optimize or refit")

    monkeypatch.setattr(torch.optim, "Adam", forbid_training)
    monkeypatch.setattr(torch.Tensor, "backward", forbid_training)
    rng = torch.random.get_rng_state().clone()
    report, _ = state.call()
    assert report["status"] == "complete"
    assert torch.equal(rng, torch.random.get_rng_state())
    assert [row["epoch"] for row in report["rows"]] == [s["epoch"] for s in state.students]


@pytest.mark.parametrize("status", ["stopped", "failed"])
def test_incomplete_validation_never_loads_test(experiment, status):
    state = experiment
    state.pilot_status = status
    report, root = state.call()
    assert report["status"] == status
    assert state.test_loads == 0 and report["rows"] == []
    assert not list(root.rglob("student_*_test_routes.json"))


@pytest.mark.parametrize("change", ["missing_student", "test_metric"])
def test_incomplete_or_test_contaminated_student_report_is_refused(experiment, change):
    state = experiment
    if change == "missing_student":
        state.students.pop()
    else:
        state.students[0]["test_acc"] = 100.
    report, _ = state.call()
    assert report["status"] == "failed"
    assert "every fresh student without test" in report["reason"]
    assert state.test_loads == 0


def test_changed_selection_is_refused_before_any_pilot_or_test(experiment):
    state = experiment
    state.selection["options"]["steps"] += 5
    save_json(state.selection, state.selection_path)
    with pytest.raises(ValueError, match="Frozen large selection changed"):
        state.call()
    assert state.pilot_calls == 0 and state.test_loads == 0


def test_changed_source_data_digest_is_refused_before_test(experiment):
    state = experiment
    save_json(dict(data_digest="different-training-validation-source"), state.candidate / "protocol.json")
    report, _ = state.call()
    assert report["status"] == "failed"
    assert "inputs differ" in report["reason"]
    assert state.test_loads == 0


def test_changed_source_code_is_refused_before_any_pilot_or_test(experiment, monkeypatch):
    state = experiment
    monkeypatch.setattr(research_loop, "implementation_provenance",
                        lambda: dict(source_digest="different-implementation"))
    with pytest.raises(ValueError, match="source differs"):
        state.call()
    assert state.pilot_calls == 0 and state.test_loads == 0


def test_source_code_change_during_validation_is_refused_before_test(experiment, monkeypatch):
    state = experiment
    original = final.run_pilot

    def pilot_then_change_source(**kwargs):
        result = original(**kwargs)
        monkeypatch.setattr(research_loop, "implementation_provenance",
                            lambda: dict(source_digest="changed-during-validation"))
        return result

    monkeypatch.setattr(final, "run_pilot", pilot_then_change_source)
    report, _ = state.call()
    assert report["status"] == "failed"
    assert "source differs" in report["reason"]
    assert state.pilot_calls == 1 and state.test_loads == 0


@pytest.mark.parametrize("seeds", [(0,), (910, 910), (-1,), (True,)])
def test_nonfresh_or_invalid_student_seeds_never_fit_or_test(experiment, seeds):
    state = experiment
    with pytest.raises(ValueError, match="distinct fresh student seeds"):
        state.call(student_seeds=seeds)
    assert state.pilot_calls == 0 and state.test_loads == 0


@pytest.mark.parametrize("change", ["checkpoint_fingerprint", "checkpoint_epoch", "cache_recipe",
                                   "cache_test_enabled", "cache_epoch", "missing_history",
                                   "history_first_max"])
def test_validation_selection_provenance_is_verified_before_test(experiment, change):
    state = experiment
    seed = state.seeds[0]
    selected_path = state.folder / f"seed_{seed}_selected.pt"
    cache_path = state.folder / f"seed_{seed}.json"
    history_path = state.folder / f"seed_{seed}_epochs.csv"
    if change.startswith("checkpoint"):
        selected = torch.load(selected_path, map_location="cpu", weights_only=False)
        if change == "checkpoint_fingerprint":
            selected["fingerprint"] = "wrong-input-provenance"
        else:
            selected["epoch"] += 1
        torch.save(selected, selected_path)
    elif change.startswith("cache"):
        cache = json.loads(cache_path.read_text())
        if change == "cache_recipe":
            cache["recipe"]["settings"]["dropout"] = .8
        elif change == "cache_test_enabled":
            cache["recipe"]["test_enabled"] = True
        else:
            cache["result"]["epoch"] += 1
        save_json(cache, cache_path)
    elif change == "missing_history":
        history_path.unlink()
    else:
        history = pd.read_csv(history_path)
        # Select an epoch distinct from the real validation winner.
        real_epoch = state.students[0]["epoch"]
        row = history.index[history.epoch != real_epoch][0]
        history.loc[row, "val_acc"] = 101.
        history.to_csv(history_path, index=False)
    report, _ = state.call()
    assert report["status"] == "failed", f"Unverified {change} reached testing"
    assert state.test_loads == 0


@pytest.mark.parametrize("field,value", [("seed", 123), ("weighting", "mass"),
                                       ("layers", 3), ("selection", "test maximum"),
                                       ("test_enabled", True),
                                       ("settings", dict(epochs=4, eval_every=1, hidden=3,
                                                         dropout=.5, lr=.01,
                                                         weight_decay=.0005))])
def test_self_consistent_cache_must_match_fresh_uniform_validation_recipe(experiment, field, value):
    state = experiment
    seed = state.seeds[0]
    selected_path = state.folder / f"seed_{seed}_selected.pt"
    cache_path = state.folder / f"seed_{seed}.json"
    cache = json.loads(cache_path.read_text())
    cache["recipe"][field] = value
    cache["fingerprint"] = _fingerprint(cache["recipe"])
    selected = torch.load(selected_path, map_location="cpu", weights_only=False)
    selected["fingerprint"] = cache["fingerprint"]
    save_json(cache, cache_path)
    torch.save(selected, selected_path)
    report, _ = state.call()
    assert report["status"] == "failed"
    assert "Selected student" in report["reason"]
    assert state.test_loads == 0


@pytest.mark.parametrize("change", ["result_budget", "truncated_history", "tied_later_epoch"])
def test_selected_epoch_requires_complete_budget_and_first_validation_maximum(experiment, change):
    state = experiment
    seed = state.seeds[0]
    cache_path = state.folder / f"seed_{seed}.json"
    history_path = state.folder / f"seed_{seed}_epochs.csv"
    selected_path = state.folder / f"seed_{seed}_selected.pt"
    cache = json.loads(cache_path.read_text())
    history = pd.read_csv(history_path)
    if change == "result_budget":
        cache["result"]["last_epoch"] -= 1
    elif change == "truncated_history":
        history = history.iloc[:-1]
    else:
        first = int(history.iloc[0].epoch)
        later = int(history.iloc[-1].epoch)
        history.loc[:, "val_acc"] = cache["result"]["val_acc"]
        selected = torch.load(selected_path, map_location="cpu", weights_only=False)
        selected["epoch"] = later
        cache["result"]["epoch"] = later
        state.students[0]["epoch"] = later
        torch.save(selected, selected_path)
        assert first != later
    save_json(cache, cache_path)
    history.to_csv(history_path, index=False)
    report, _ = state.call()
    assert report["status"] == "failed"
    assert "Selected epoch" in report["reason"]
    assert state.test_loads == 0


@pytest.mark.parametrize("which", ["native", "frozen"])
def test_changed_selected_state_is_refused_on_resume(experiment, which):
    state = experiment
    first, root = state.call()
    assert first["status"] == "complete"
    path = (state.folder / f"seed_{state.seeds[0]}_selected.pt" if which == "native"
            else root / "condensation_0" / "selected_weights" / f"seed_{state.seeds[0]}_selected.pt")
    selected = torch.load(path, map_location="cpu", weights_only=False)
    selected["model_state"]["layers.0.lin.weight"] += .25
    torch.save(selected, path)
    previous_loads = state.test_loads
    second, _ = state.call()
    assert second["status"] == "failed"
    assert "weights differ" in second["reason"]
    assert state.test_loads == previous_loads


def test_completed_resume_uses_frozen_models_and_route_cache(experiment, monkeypatch):
    state = experiment
    first, root = state.call()
    frozen = {p.name: digest(p) for p in (root / "condensation_0" / "selected_weights").iterdir()}

    def forbid_forward(*args, **kwargs):
        raise AssertionError("Completed test replay must use its verified cache")

    monkeypatch.setattr(routes, "_forward", forbid_forward)
    second, _ = state.call()
    assert second["status"] == "complete"
    assert first["rows"] == second["rows"]
    assert first["selected_weights"] == second["selected_weights"]
    assert frozen == {p.name: digest(p) for p in (root / "condensation_0" / "selected_weights").iterdir()}


def test_interrupted_replay_resumes_without_replacing_selected_weights(experiment, monkeypatch):
    state = experiment
    real_replay = final.replay_routes
    interrupted = [False]

    def replay_then_stop(*args, **kwargs):
        result = real_replay(*args, **kwargs)
        interrupted[0] = True
        return result

    monkeypatch.setattr(final, "replay_routes", replay_then_stop)
    first, root = state.call(stop=lambda: interrupted[0])
    assert first["status"] == "stopped" and len(first["rows"]) == 1
    frozen = {p.name: digest(p) for p in (root / "condensation_0" / "selected_weights").iterdir()}
    first_cache = root / "condensation_0" / f"student_{state.seeds[0]}_test_routes.json"
    first_cache_digest = digest(first_cache)
    interrupted[0] = False
    monkeypatch.setattr(final, "replay_routes", real_replay)
    calls = []
    original_forward = routes._forward

    def track_forward(*args, **kwargs):
        calls.append(1)
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(routes, "_forward", track_forward)
    second, _ = state.call()
    assert second["status"] == "complete" and len(second["rows"]) == 2
    assert len(calls) == 2  # Only the incomplete seed's two routes ran again.
    assert digest(first_cache) == first_cache_digest
    assert frozen == {p.name: digest(p) for p in (root / "condensation_0" / "selected_weights").iterdir()}


def test_changed_test_labels_reject_cached_result_without_changing_weights(experiment):
    state = experiment
    first, root = state.call()
    frozen = {p.name: digest(p) for p in (root / "condensation_0" / "selected_weights").iterdir()}
    state.testing["y"] = 1 - state.testing["y"]
    second, _ = state.call()
    assert second["status"] == "failed"
    assert "Cached route replay differs" in second["reason"]
    assert frozen == {p.name: digest(p) for p in (root / "condensation_0" / "selected_weights").iterdir()}
    assert first["selected_weights"] == second["selected_weights"]
