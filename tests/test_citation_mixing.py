"""Per-candidate background priors preserve citation caches and frozen choices."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

import src.citation_search as search
import src.coarsening_ce as coarsening
import src.distance_partition as distance
import src.nystrom_ce as nystrom
import src.student_routes as student_routes
from src.evaluation import fit_gcn_diagnostic
from src.io import _fingerprint, save_state
from src.moments import initial_logits, make_material

METHODS = ("low_rank", "mlp", "distance", "nystrom", "coarsening")


@pytest.fixture(scope="module", autouse=True)
def small_cpu_jobs():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def candidate(method="low_rank", **changes):
    return dict(method=method, width=4 if method == "mlp" else 0, lr=.01,
                T=.3, rank=2, penalty=.1, **changes)


@pytest.fixture
def source(monkeypatch):
    # The rare first cell exposes contamination by the background prior.
    x = torch.tensor([[10., .1], [0., .4], [0., .8], [0., .2], [0., .6],
                      [0., 1.], [0., .3], [0., .7], [0., .9]])
    graph = dict(x=x, y=torch.arange(9) % 2, adj=torch.eye(9).to_sparse_csr())
    train = torch.arange(9) < 3
    validation = graph, (torch.arange(9) >= 3) & (torch.arange(9) < 6)
    testing = graph, torch.arange(9) >= 6
    assignment = torch.tensor([0] + [1] * 8)
    logits = torch.tensor([[1.5, -.1], [-.3, 1.], [.7, .2], [.2, .9], [.8, .1],
                           [.1, .8], [.5, .3], [.3, .6], [.4, .7]], dtype=torch.double)
    state = SimpleNamespace(graph=graph, train=train, validation=validation, testing=testing,
                            h=x, logits=logits, assignment=assignment, data_calls=0,
                            teacher_fits=0, hard_assignment_fits=0, optimizer_calls=[], evaluations=[],
                            map_fits=0, phi_fits=0)

    def prepare(*args):
        state.data_calls += 1
        return graph, train, validation, testing, x

    def teacher(*args):
        path = args[-1] / "teacher.pt"
        if not path.exists():
            state.teacher_fits += 1
            save_state(dict(logits=logits, gamma=.01), path)
        return torch.load(path, weights_only=False)["logits"], .01

    def hard_assignment(*args):
        state.hard_assignment_fits += 1
        return assignment.clone()

    def evaluate(cx, cy, mass, evaluation_graph, q, masks, seed, **kwargs):
        assert evaluation_graph["adj"] is graph["adj"]
        assert set(masks) in ({"train", "val"}, {"train", "val", "test"})
        assert masks["val"] is validation[1]
        assert torch.equal(mass, torch.full_like(mass, .5)), "Student CE must remain uniform"
        state.evaluations.append(dict(x=cx.clone(), labels=cy.clone(), mass=mass.clone(),
                                     q=q.clone(), masks=masks, seed=seed,
                                     folder=Path(kwargs["folder"]), adjacency=kwargs["training_adjacency"]))
        return dict(seed=seed, epoch=1, val_acc=50.)

    monkeypatch.setattr(search, "_prepare_dataset", prepare)
    monkeypatch.setattr(search, "teacher_logits", teacher)
    monkeypatch.setattr(search, "feature_kmeans", hard_assignment)
    monkeypatch.setattr(search, "fit_gcn_diagnostic", evaluate)
    monkeypatch.setitem(search.BUDGET, ("cora", .026), 2)
    return state


@pytest.fixture
def mocked_optimizers(source, monkeypatch):
    state = source

    def save_checkpoints(features, q, assignment, method, folder, checks, mixing, kwargs):
        state.optimizer_calls.append(dict(method=method, mixing=mixing, folder=folder,
                                          assignment=assignment.clone(), q=q.clone(), kwargs=kwargs))
        p = initial_logits(assignment, 2, mixing).double().softmax(1)
        moments = p.T @ make_material(features.double(), q.double()) / len(p)
        checkpoint_folder = folder if method == "nystrom" else folder / "checkpoints"
        checkpoint_folder.mkdir(parents=True, exist_ok=True)
        for step in checks:
            snapshot = dict(step=step, moments=moments, J_exact=True)
            if method == "coarsening":
                mass = p.sum(0)
                snapshot.update(x=p.T @ features.double() / mass[:, None],
                                labels=p.T @ q.double() / mass[:, None], mass=mass / len(p),
                                adj=torch.eye(2, dtype=torch.double))
            save_state(snapshot, checkpoint_folder / f"step_{step:06d}.pt")
        save_state(dict(step=max(checks), config=dict(mixing=mixing)), folder / "resume.pt")

    def linear(features, q, assignment, **kwargs):
        method = "mlp" if kwargs["assignment_encoder"] == "mlp" else "low_rank"
        save_checkpoints(features, q, assignment, method, kwargs["folder"],
                         kwargs["checkpoint_steps"], kwargs.get("mixing", .05), kwargs)

    def distance_optimizer(features, q, assignment, **kwargs):
        save_checkpoints(features, q, assignment, "distance", kwargs["folder"],
                         kwargs["checkpoint_steps"], kwargs.get("mixing", .05), kwargs)

    def nystrom_optimizer(h, q, assignment, feature_map, phi, folder, steps, **kwargs):
        save_checkpoints(h, q, assignment, "nystrom", folder, [0, steps], kwargs.get("mixing", .05), kwargs)

    def coarsening_optimizer(x, q, assignment, adjacency, **kwargs):
        save_checkpoints(x, q, assignment, "coarsening", kwargs["folder"],
                         kwargs["checkpoint_steps"], kwargs.get("mixing", .05), kwargs)

    def shared_map(h, path, **kwargs):
        if not path.exists():
            state.map_fits += 1
            save_state(dict(kind="mock-shared-map"), path)
        return "same-map"

    def features(h, feature_map, path, **kwargs):
        assert feature_map == "same-map"
        if not path.exists():
            state.phi_fits += 1
            np.save(path, h.numpy())
        return np.load(path, mmap_mode="r")

    monkeypatch.setattr(search, "optimize_ce_assignment", linear)
    monkeypatch.setattr(distance, "optimize_distance_ce", distance_optimizer)
    monkeypatch.setattr(nystrom, "optimize", nystrom_optimizer)
    monkeypatch.setattr(coarsening, "optimize_coarsening_ce", coarsening_optimizer)
    monkeypatch.setattr(search, "get_shared_map", shared_map)
    monkeypatch.setattr(nystrom, "cache_features", features)
    return state


def run(output_dir, candidates, **kwargs):
    options = dict(dataset="cora", ratio=.026, output_dir=output_dir, candidates=candidates,
                   steps=1, epochs=2, student_seeds=(0,), device="cpu")
    options.update(kwargs)
    return search.run_screen(**options)


@pytest.mark.parametrize("method", METHODS)
def test_per_candidate_prior_is_forwarded_and_separates_only_candidate_inputs(tmp_path, mocked_optimizers, method):
    state = mocked_optimizers
    first, root = run(tmp_path, [candidate(method)])
    saved_h = digest(root / "propagated_H.pt")
    saved_assignment = digest(root / "inputs_0.pt")
    saved_teacher = digest(root / "teacher.pt")
    second, same_root = run(tmp_path, [candidate(method, mixing=.005)])
    assert same_root == root
    assert state.teacher_fits == state.hard_assignment_fits == 1
    assert saved_h == digest(root / "propagated_H.pt")
    assert saved_assignment == digest(root / "inputs_0.pt")
    assert saved_teacher == digest(root / "teacher.pt")
    assert [call["mixing"] for call in state.optimizer_calls] == [.05, .005]
    assert set(first.candidate_path).isdisjoint(set(second.candidate_path))
    old = json.loads((Path(first.iloc[0].candidate_path) / "candidate.json").read_text())
    new = json.loads((Path(second.iloc[0].candidate_path) / "candidate.json").read_text())
    assert "mixing" not in old and new.pop("mixing") == .005
    assert new == old
    assert torch.equal(state.optimizer_calls[0]["assignment"], state.optimizer_calls[1]["assignment"])
    assert torch.equal(state.optimizer_calls[0]["q"], state.optimizer_calls[1]["q"])
    assert all(set(call["masks"]) == {"train", "val"} for call in state.evaluations)
    assert not any("test" in column for column in (*first.columns, *second.columns))
    if method == "nystrom":
        assert state.map_fits == state.phi_fits == 1


@pytest.mark.parametrize("method", METHODS)
def test_explicit_default_reuses_legacy_candidate_resume_and_evaluation_paths(tmp_path, mocked_optimizers, method):
    state = mocked_optimizers
    first, root = run(tmp_path, [candidate(method)])
    candidate_path = Path(first.iloc[0].candidate_path)
    snapshot_hash = digest(candidate_path / "condensation_0" / "resume.pt")
    prior_evaluation_folders = {call["folder"] for call in state.evaluations}
    second, same_root = run(tmp_path, [candidate(method, mixing=.05)])
    assert root == same_root and set(first.candidate_path) == set(second.candidate_path)
    assert len(state.optimizer_calls) == 1
    assert state.teacher_fits == state.hard_assignment_fits == 1
    assert digest(candidate_path / "condensation_0" / "resume.pt") == snapshot_hash
    assert {call["folder"] for call in state.evaluations} == prior_evaluation_folders
    assert "mixing" not in json.loads((candidate_path / "candidate.json").read_text())


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("mixing", [0, 1, -.005, float("nan"), float("inf"), True, "0.005"])
def test_invalid_candidate_mixing_fails_before_loading_or_creating_any_output(tmp_path, source, method, mixing):
    with pytest.raises(ValueError, match="mixing"):
        run(tmp_path, [candidate(method, mixing=mixing)])
    assert source.data_calls == source.teacher_fits == 0
    assert not list(tmp_path.iterdir())


def test_every_candidate_is_validated_before_loading_data(tmp_path, source):
    with pytest.raises(ValueError, match="mixing"):
        run(tmp_path, [candidate(), candidate("nystrom", mixing=float("nan"))])
    assert source.data_calls == source.teacher_fits == 0


@pytest.mark.parametrize("method", METHODS)
def test_nondefault_selected_choice_recreates_exact_candidate_and_centroids(tmp_path, mocked_optimizers, method):
    state = mocked_optimizers
    ranking, root = run(tmp_path, [candidate(method, mixing=.005)])
    choice = ranking.iloc[0].to_dict()
    expected_path = Path(choice["candidate_path"])
    expected_x = state.evaluations[-1]["x"].clone()
    expected_labels = state.evaluations[-1]["labels"].clone()
    result = search.selected_test(root, choice, condensation_seeds=(0, 1), student_seeds=(100,),
                                  epochs=2, device="cpu", report_routes=False)
    assert set(result.condensation_seed) == {0, 1}
    selected = json.loads((root / "selected.json").read_text())
    assert selected["candidate"]["mixing"] == .005
    assert root / _fingerprint(selected["candidate"]) == expected_path
    assert [call["mixing"] for call in state.optimizer_calls] == [.005, .005]
    for evaluation in state.evaluations:
        if "test" in evaluation["masks"]:
            assert evaluation["seed"] == 100
            torch.testing.assert_close(evaluation["x"], expected_x)
            torch.testing.assert_close(evaluation["labels"], expected_labels)
            assert evaluation["folder"].parent.parent.parent == expected_path


@pytest.mark.parametrize("mixing", [0, 1, float("nan"), float("inf"), True, "0.005"])
def test_selected_malformed_prior_is_refused_before_source_loading(tmp_path, mocked_optimizers, mixing):
    ranking, root = run(tmp_path, [candidate()])
    choice = dict(ranking.iloc[0], mixing=mixing)
    before = mocked_optimizers.data_calls
    with pytest.raises(ValueError, match="mixing"):
        search.selected_test(root, choice, device="cpu", report_routes=False)
    assert mocked_optimizers.data_calls == before
    assert not (root / "selected.json").exists()


def test_default_selection_identity_is_identical_with_explicit_or_implicit_prior(tmp_path, mocked_optimizers):
    ranking, root = run(tmp_path, [candidate()])
    choice = ranking.iloc[0].to_dict()
    options = dict(condensation_seeds=(0,), student_seeds=(100,), epochs=2, device="cpu", report_routes=False)
    search.selected_test(root, choice, **options)
    implicit = json.loads((root / "selected.json").read_text())
    paths = set(root.glob("selected_*.json"))
    search.selected_test(root, dict(choice, mixing=.05), **options)
    assert json.loads((root / "selected.json").read_text()) == implicit
    assert set(root.glob("selected_*.json")) == paths
    assert "mixing" not in implicit["candidate"]


def test_mixed_default_and_nondefault_ranking_is_selectable_without_nan_prior(tmp_path, mocked_optimizers):
    ranking, root = run(tmp_path, [candidate(), candidate(mixing=.005)])
    assert set(ranking.mixing) == {.05, .005}
    assert not bool(ranking.mixing.isna().any())
    for row in ranking.to_dict("records"):
        search.selected_test(root, row, condensation_seeds=(0,), student_seeds=(100,),
                             epochs=2, device="cpu", report_routes=False)
        selected = json.loads((root / "selected.json").read_text())
        assert root / _fingerprint(selected["candidate"]) == Path(row["candidate_path"])


@pytest.mark.parametrize("method", METHODS)
def test_real_step_zero_optimizer_uses_nondefault_analytical_p_weighted_means(tmp_path, source, method):
    # No optimizer is mocked here: every branch produces its real step-zero snapshot.
    _, root = run(tmp_path, [candidate(method, mixing=.005)], steps=0)
    assert len(source.evaluations) == 1
    supplied = source.evaluations[0]
    p = .995 * torch.nn.functional.one_hot(source.assignment, 2).double() + .005 / 2
    expected_x = p.T @ source.h.double() / p.sum(0)[:, None]
    q = (source.logits / .3).softmax(1)
    expected_labels = p.T @ q / p.sum(0)[:, None]
    torch.testing.assert_close(supplied["x"], expected_x.float(), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(supplied["labels"], expected_labels.float(), atol=1e-6, rtol=1e-6)
    legacy_p = .95 * torch.nn.functional.one_hot(source.assignment, 2).double() + .05 / 2
    legacy_mean = legacy_p.T @ source.h.double() / legacy_p.sum(0)[:, None]
    assert not torch.allclose(supplied["x"], legacy_mean.float())
    assert set(supplied["masks"]) == {"train", "val"}
    assert source.teacher_fits == source.hard_assignment_fits == 1
    identity_path = next(root.glob("*/candidate.json"))
    assert json.loads(identity_path.read_text())["mixing"] == .005


@pytest.mark.parametrize("method", ["low_rank", "mlp", "distance", "nystrom", "coarsening"])
def test_cross_mixing_resume_state_is_kept_out_of_new_candidate(tmp_path, mocked_optimizers, method):
    state = mocked_optimizers
    first, root = run(tmp_path, [candidate(method, mixing=.005)])
    first_folder = Path(first.iloc[0].candidate_path) / "condensation_0"
    before = digest(first_folder / "resume.pt")
    second, same = run(tmp_path, [candidate(method, mixing=.02)], steps=2)
    assert same == root
    assert set(first.candidate_path).isdisjoint(set(second.candidate_path))
    assert digest(first_folder / "resume.pt") == before
    assert state.optimizer_calls[0]["mixing"] == .005
    assert state.optimizer_calls[1]["mixing"] == .02
    assert state.optimizer_calls[1]["kwargs"].get("resume_state") is None


@pytest.mark.parametrize("report_routes", [None, 0, 1, "True"])
def test_invalid_validation_route_flag_is_refused_before_loading(tmp_path, source, report_routes):
    with pytest.raises(ValueError, match="report_routes"):
        run(tmp_path, [candidate()], report_routes=report_routes)
    assert source.data_calls == source.teacher_fits == 0


def test_optional_validation_routes_reuse_selected_weights_and_never_fit_again(tmp_path, mocked_optimizers,
                                                                            monkeypatch):
    state = mocked_optimizers
    state.graph["y"][state.testing[1]] = 999  # Excluded labels cannot enter scoring or class inference.
    native_calls, route_calls = [], []

    def fit(*args, **kwargs):
        assert set(args[5]) == {"train", "val"}
        native_calls.append(Path(kwargs["folder"]))
        return fit_gcn_diagnostic(*args, **kwargs)

    original_replay = student_routes.replay_routes

    def replay(path, graph, h, masks, settings, output_path, **kwargs):
        assert set(masks) == {"val"} and masks["val"] is state.validation[1]
        assert torch.equal(h, state.h.float() * 1.6)
        assert torch.equal(graph["x"], state.graph["x"] * 1.6)
        route_calls.append(dict(path=Path(path), masks=masks, graph=graph, h=h.clone()))
        return original_replay(path, graph, h, masks, settings, output_path, **kwargs)

    monkeypatch.setattr(search, "fit_gcn_diagnostic", fit)
    monkeypatch.setattr(student_routes, "replay_routes", replay)
    options = dict(steps=0, epochs=3, dropout=0., input_scale=1.6,
                   student_settings=dict(eval_every=1, hidden=3))
    first, root = run(tmp_path, [candidate(mixing=.005)], **options)
    folder = native_calls[0]
    original_native = {path.name: digest(path) for path in folder.glob("seed_0*")}
    original_teacher = digest(root / "teacher.pt")

    def forbid_optimizer(*args, **kwargs):
        pytest.fail("Validation route replay must reuse the selected student without another fit")

    monkeypatch.setattr(torch.optim, "Adam", forbid_optimizer)
    second, same_root = run(tmp_path, [candidate(mixing=.005)], report_routes=True, **options)
    assert same_root == root and set(first.candidate_path) == set(second.candidate_path)
    assert set(first.student_recipe) == set(second.student_recipe)
    assert native_calls == [folder, folder] and len(route_calls) == 1
    assert len(state.optimizer_calls) == 1
    assert digest(root / "teacher.pt") == original_teacher
    assert original_native == {path.name: digest(path) for path in folder.glob("seed_0*")
                               if not path.name.endswith("validation_routes_v1.json")}
    route = json.loads((folder / "seed_0_validation_routes_v1.json").read_text())
    selected = torch.load(folder / "seed_0_selected.pt", weights_only=False)
    assert route["recipe"]["epoch"] == selected["epoch"]
    assert route["recipe"]["source_fingerprint"] == selected["fingerprint"]
    assert route["recipe"]["test_enabled"] is False
    assert not any("test_" in key for key in route["result"])
    raw_frames = [pd.read_csv(path) for path in root.glob("screen_*.csv")]
    with_routes = [frame for frame in raw_frames if "mlp_val_acc" in frame]
    assert len(with_routes) == 1
    row = with_routes[0].iloc[0]
    assert row["val_acc"] == row["gcn_val_acc"]
    assert row["gcn_val_acc"] == route["result"]["gcn_val_acc"]
    assert row["mlp_val_acc"] == route["result"]["mlp_val_acc"]
    assert not any("test_" in key for key in row.index)
    protocols = [json.loads(path.read_text()) for path in root.glob("screen_protocol_*.json")]
    assert len(protocols) == 2 and sum(item.get("report_routes", False) for item in protocols) == 1


@pytest.mark.parametrize("change", ["test_metric", "different_epoch", "different_gcn_accuracy", "different_gcn_ce"])
def test_validation_route_replay_refuses_test_scores_or_different_selection(tmp_path, mocked_optimizers,
                                                                          monkeypatch, change):
    state = mocked_optimizers

    def fit(*args, **kwargs):
        return dict(seed=args[6], epoch=2, val_acc=50., val_ce=.7)

    def replay(*args, **kwargs):
        result = dict(seed=kwargs["seed"], epoch=2, gcn_val_acc=50., gcn_val_ce=.7,
                      mlp_val_acc=55., mlp_val_ce=.6)
        if change == "test_metric":
            result["mlp_test_acc"] = 100.
        elif change == "different_epoch":
            result["epoch"] = 1
        elif change == "different_gcn_accuracy":
            result["gcn_val_acc"] = 51.
        else:
            result["gcn_val_ce"] = .8
        return result

    monkeypatch.setattr(search, "fit_gcn_diagnostic", fit)
    monkeypatch.setattr(student_routes, "replay_routes", replay)
    with pytest.raises(ValueError, match="route replay differs"):
        run(tmp_path, [candidate(mixing=.005)], steps=0, report_routes=True)
    assert state.teacher_fits == 1
