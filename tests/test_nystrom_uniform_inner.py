"""Uniform Nyström CE changes head weights while retaining P-derived material."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import test_citation_mixing as citation_fixtures
import test_large_pilot as large_fixtures
import torch

import src.citation_search as search
import src.large_pilot as pilot
import src.nystrom_ce as nystrom
import src.student_routes as student_routes
from src.evaluation import fit_gcn_diagnostic
from src.io import _fingerprint, save_state
from src.low_rank_assignment import LowRankMoments, logit_block
from src.moments import augmented, decode_moments, initial_logits, make_material
from src.soft_ce_partition import solve_head_system, solve_inner_newton_first

source = citation_fixtures.source
mocked_optimizers = citation_fixtures.mocked_optimizers
mocked_pipeline = large_fixtures.mocked_pipeline
digest = citation_fixtures.digest
run = citation_fixtures.run


@pytest.fixture(scope="module", autouse=True)
def small_cpu_jobs():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def candidate(**changes):
    return {**citation_fixtures.candidate("nystrom"), **changes}


def problem():
    generator = torch.Generator().manual_seed(7)
    h = torch.randn(24, 3, dtype=torch.double, generator=generator) + .2
    q = torch.randn(24, 3, dtype=torch.double, generator=generator).softmax(1)
    assignment = torch.tensor([0] * 9 + [1] * 6 + [2] * 5 + [3] * 4)
    return h, q, assignment, nystrom.NystromMap.fit(h, basis=8)


@pytest.mark.parametrize("changed_factor", ["u", "v", "both"])
def test_uniform_implicit_derivative_matches_refitted_head_along_valid_assignment_path(changed_factor):
    h, q, assignment, feature_map = problem()
    original_q = q.clone()
    generator = torch.Generator().manual_seed(19)
    u = (.2 * torch.randn(24, 3, dtype=torch.double, generator=generator)).requires_grad_()
    v = torch.randn(4, 3, dtype=torch.double, generator=generator).requires_grad_()
    du = torch.randn(u.shape, dtype=u.dtype, generator=generator) * .2
    dv = torch.randn(v.shape, dtype=v.dtype, generator=generator) * .2
    if changed_factor == "u":
        dv.zero_()
    elif changed_factor == "v":
        du.zero_()
    material = make_material(h, q)
    phi = feature_map(h).detach().numpy()
    moments = LowRankMoments.apply(u, v, assignment, material, .05, 7)

    def reference_moments(left, right):
        p = logit_block(left, right, assignment, .05).softmax(1)
        torch.testing.assert_close(p.sum(1), torch.ones(len(p), dtype=p.dtype))
        return p.T @ material / len(p)

    def refitted_objective(value):
        centers, labels, mass = decode_moments(value, 3)
        # Perturb P, so labels stay normalized while masses and both barycenters vary.
        torch.testing.assert_close(labels.sum(1), torch.ones(4, dtype=labels.dtype))
        mapped = feature_map(centers).detach()
        weights = torch.full_like(mass, 1 / len(mass))
        fitted = solve_inner_newton_first(mapped, labels, weights, .1, grad_tol=1e-10)
        assert fitted["inner_converged"]
        loss, rhs = nystrom.outer_gradient(phi, q, fitted["theta"], chunk=7)
        return loss, rhs, fitted["theta"], mapped, labels, weights

    _, rhs, theta, mapped, labels, weights = refitted_objective(moments.detach())
    vector, information = solve_head_system(augmented(mapped), labels, weights, theta,
                                             .1, rhs, rtol=1e-10)
    assert information["cg_converged"]
    derivative = nystrom.moment_gradient(moments, 3, feature_map, theta, vector,
                                        .1, inner_loss_weighting="uniform")
    moments.backward(derivative)
    analytic = float((u.grad * du).sum() + (v.grad * dv).sum())
    epsilon = 1e-3
    plus = reference_moments(u.detach() + epsilon * du, v.detach() + epsilon * dv)
    minus = reference_moments(u.detach() - epsilon * du, v.detach() - epsilon * dv)
    plus_centers, plus_labels, plus_mass = decode_moments(plus, 3)
    minus_centers, minus_labels, minus_mass = decode_moments(minus, 3)
    assert (plus_mass - minus_mass).abs().max() > 1e-8
    assert (plus_centers - minus_centers).abs().max() > 1e-8
    assert (plus_labels - minus_labels).abs().max() > 1e-8
    numerical = (refitted_objective(plus)[0] - refitted_objective(minus)[0]) / (2 * epsilon)
    assert abs(numerical) > 1e-7
    np.testing.assert_allclose(analytic, numerical, rtol=2e-4, atol=1e-8)
    # A mean of kernel features is a different geometry from kernel(mean H).
    p = logit_block(u.detach(), v.detach(), assignment, .05).softmax(1)
    mean_phi = p.T @ torch.from_numpy(phi) / p.sum(0)[:, None]
    assert not torch.allclose(mapped, mean_phi, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(q, original_q, atol=0, rtol=0)


def test_uniform_weights_are_constant_in_inner_head_adjoint_and_material_gradient(tmp_path, monkeypatch):
    h, q, assignment, feature_map = problem()
    recorded = {name: [] for name in ("solve_inner_newton_first", "solve_head_system", "head_gradient")}
    for name in recorded:
        original = getattr(nystrom, name)

        def capture(*args, _name=name, _original=original, **kwargs):
            recorded[_name].append(args[2].detach().clone())
            return _original(*args, **kwargs)

        monkeypatch.setattr(nystrom, name, capture)
    nystrom.optimize(h, q, assignment, feature_map, feature_map(h).numpy(), tmp_path, 2,
                     penalty=.1, rank=3, chunk=7, checkpoint_every=1,
                     inner_loss_weighting="uniform")
    mass = decode_moments(load(tmp_path / "step_000002.pt")["moments"], 3)[2]
    assert not torch.allclose(mass, torch.full_like(mass, .25))
    for calls in recorded.values():
        assert calls
        for weights in calls:
            torch.testing.assert_close(weights, torch.full_like(weights, .25), atol=0, rtol=0)


def test_uniform_checkpoint_resume_and_endpoint_are_original_p_weighted_averages(tmp_path):
    h, q, assignment, feature_map = problem()
    before_q = q.clone()
    phi = nystrom.cache_features(h, feature_map, tmp_path / "phi.npy", chunk=7)
    shared_hashes = {path: digest(path) for path in (tmp_path / "phi.npy", tmp_path / "phi.meta.json")}
    options = dict(penalty=.1, rank=3, chunk=7, checkpoint_every=1, inner_loss_weighting="uniform")
    nystrom.optimize(h, q, assignment, feature_map, phi, tmp_path / "full", 2, **options)
    nystrom.optimize(h, q, assignment, feature_map, phi, tmp_path / "resumed", 1, **options)
    nystrom.optimize(h, q, assignment, feature_map, phi, tmp_path / "resumed", 2, **options)
    full, resumed = (load(tmp_path / name / "step_000002.pt") for name in ("full", "resumed"))
    torch.testing.assert_close(full["moments"], resumed["moments"], rtol=1e-8, atol=1e-10)
    torch.testing.assert_close(full["theta"], resumed["theta"], rtol=1e-5, atol=1e-7)
    saved = load(tmp_path / "resumed" / "resume.pt")
    assert saved["step"] == 2 and saved["config"]["inner_loss_weighting"] == "uniform"
    assert full["inner_loss_weighting"] == resumed["inner_loss_weighting"] == "uniform"
    # Stored factors remain float32: promote after their matrix product, as at training time.
    p = logit_block(saved["u"], saved["v"], assignment, .05).double().softmax(1)
    x, labels, mass = decode_moments(resumed["moments"], h.shape[1])
    torch.testing.assert_close(x, p.T @ h / p.sum(0)[:, None], rtol=1e-8, atol=1e-9)
    torch.testing.assert_close(labels, p.T @ q / p.sum(0)[:, None], rtol=1e-8, atol=1e-9)
    torch.testing.assert_close(mass, p.sum(0) / len(p), rtol=1e-8, atol=1e-10)
    torch.testing.assert_close(q, before_q, atol=0, rtol=0)
    assert {path: digest(path) for path in shared_hashes} == shared_hashes


def test_uniform_interrupted_update_resumes_to_the_same_checkpoint(tmp_path):
    h, q, assignment, feature_map = problem()
    phi = feature_map(h).numpy()
    options = dict(penalty=.1, rank=3, chunk=7, checkpoint_every=1, inner_loss_weighting="uniform")
    nystrom.optimize(h, q, assignment, feature_map, phi, tmp_path / "full", 2, **options)
    flag = dict(stopped=False)

    def progress(row):
        flag["stopped"] = True  # First update is complete; the next loop must persist step 1.

    with pytest.raises(InterruptedError, match="resumable state"):
        nystrom.optimize(h, q, assignment, feature_map, phi, tmp_path / "interrupted", 2,
                         stop=lambda: flag["stopped"], progress=progress, **options)
    stopped = load(tmp_path / "interrupted" / "resume.pt")
    assert stopped["step"] == 1 and stopped["config"]["inner_loss_weighting"] == "uniform"
    assert load(tmp_path / "interrupted" / "step_000000.pt")["inner_loss_weighting"] == "uniform"
    nystrom.optimize(h, q, assignment, feature_map, phi, tmp_path / "interrupted", 2, **options)
    full = load(tmp_path / "full" / "step_000002.pt")
    resumed = load(tmp_path / "interrupted" / "step_000002.pt")
    torch.testing.assert_close(full["moments"], resumed["moments"], rtol=1e-8, atol=1e-10)
    torch.testing.assert_close(full["theta"], resumed["theta"], rtol=1e-5, atol=1e-7)


def test_default_and_explicit_mass_retain_legacy_config_and_share_uniform_p0_cache(tmp_path):
    h, q, assignment, feature_map = problem()
    phi = nystrom.cache_features(h, feature_map, tmp_path / "phi.npy", chunk=7)
    options = dict(penalty=.1, rank=3, chunk=7, checkpoint_every=1)
    for mode in ("default", "mass", "uniform"):
        extra = {} if mode == "default" else dict(inner_loss_weighting=mode)
        nystrom.optimize(h, q, assignment, feature_map, phi, tmp_path / mode, 0, **options, **extra)
    states = {mode: load(tmp_path / mode / "resume.pt") for mode in ("default", "mass", "uniform")}
    snapshots = {mode: load(tmp_path / mode / "step_000000.pt") for mode in states}
    assert states["default"]["config"] == states["mass"]["config"]
    assert "inner_loss_weighting" not in states["mass"]["config"]
    assert "inner_loss_weighting" not in snapshots["default"]
    assert "inner_loss_weighting" not in snapshots["mass"]
    assert {json.dumps(saved["input_fingerprint"], sort_keys=True) for saved in states.values()} == {
        json.dumps(states["default"]["input_fingerprint"], sort_keys=True)}
    for mode in ("mass", "uniform"):
        torch.testing.assert_close(snapshots[mode]["moments"], snapshots["default"]["moments"], atol=0, rtol=0)
    assert not torch.allclose(snapshots["mass"]["theta"], snapshots["uniform"]["theta"])
    nystrom.optimize(h, q, assignment, feature_map, phi, tmp_path / "default", 1,
                     inner_loss_weighting="mass", **options)
    assert "inner_loss_weighting" not in load(tmp_path / "default" / "resume.pt")["config"]


@pytest.mark.parametrize("old,new", [("mass", "uniform"), ("uniform", "mass")])
def test_cross_weighting_resume_is_rejected_before_parameter_copy_fit_or_persistence(tmp_path, monkeypatch, old, new):
    h, q, assignment, feature_map = problem()
    phi = feature_map(h).numpy()
    options = dict(penalty=.1, rank=3, chunk=7, checkpoint_every=1)
    nystrom.optimize(h, q, assignment, feature_map, phi, tmp_path, 1,
                     inner_loss_weighting=old, **options)
    assert load(tmp_path / "resume.pt")["u"].abs().max() > 0
    before = {path: digest(path) for path in tmp_path.iterdir() if path.is_file()}
    initialized = []
    original_initialize = nystrom.initialize_factors

    def initialize(*args):
        factors = original_initialize(*args)
        initialized.append((factors, [factor.detach().clone() for factor in factors]))
        return factors

    def forbidden(*args, **kwargs):
        pytest.fail("Incompatible resume must be rejected before any fit/update/persistence")

    monkeypatch.setattr(nystrom, "initialize_factors", initialize)
    monkeypatch.setattr(nystrom, "solve_inner_newton_first", forbidden)
    monkeypatch.setattr(nystrom, "save_state", forbidden)
    with pytest.raises(ValueError, match="Resume configuration differs"):
        nystrom.optimize(h, q, assignment, feature_map, phi, tmp_path, 2,
                         inner_loss_weighting=new, **options)
    assert {path: digest(path) for path in before} == before
    for factors, originals in initialized:
        for factor, original in zip(factors, originals):
            torch.testing.assert_close(factor, original, atol=0, rtol=0)


@pytest.mark.parametrize("weighting", ["mse", None, True, 0, float("nan")])
def test_invalid_core_weighting_is_rejected_before_output(tmp_path, weighting):
    h, q, assignment, feature_map = problem()
    with pytest.raises(ValueError, match="weighting"):
        nystrom.optimize(h, q, assignment, feature_map, feature_map(h).numpy(),
                         tmp_path / "absent", 0, inner_loss_weighting=weighting)
    assert not (tmp_path / "absent").exists()


@pytest.fixture
def citation_optimizer(mocked_optimizers, monkeypatch):
    original = nystrom.optimize

    def optimize(*args, **kwargs):
        original(*args, **kwargs)
        if kwargs.get("inner_loss_weighting") == "uniform":
            for path in Path(args[5]).glob("step_*.pt"):
                snapshot = load(path)
                snapshot["inner_loss_weighting"] = "uniform"
                save_state(snapshot, path)

    monkeypatch.setattr(nystrom, "optimize", optimize)
    return mocked_optimizers


def test_citation_uniform_separates_candidate_but_preserves_teacher_geometry_assignment_and_student_recipe(
        tmp_path, citation_optimizer):
    state = citation_optimizer
    first, root = run(tmp_path, [candidate()])
    shared = {path: digest(path) for path in (root / "teacher.pt", root / "propagated_H.pt", root / "inputs_0.pt")}
    second, same = run(tmp_path, [candidate(inner_loss_weighting="uniform")])
    assert same == root and set(first.candidate_path).isdisjoint(second.candidate_path)
    assert set(first.student_recipe) == set(second.student_recipe)
    assert state.teacher_fits == state.hard_assignment_fits == state.map_fits == state.phi_fits == 1
    assert {path: digest(path) for path in shared} == shared
    default, uniform = state.optimizer_calls
    assert "inner_loss_weighting" not in default["kwargs"]
    assert uniform["kwargs"]["inner_loss_weighting"] == "uniform"
    for key in ("assignment", "q"):
        torch.testing.assert_close(default[key], uniform[key], atol=0, rtol=0)
    p = initial_logits(state.assignment, 2).double().softmax(1)
    q = (state.logits / .3).softmax(1)
    for evaluation in state.evaluations:
        torch.testing.assert_close(evaluation["x"], (p.T @ state.h.double() / p.sum(0)[:, None]).float())
        torch.testing.assert_close(evaluation["labels"], (p.T @ q / p.sum(0)[:, None]).float())
        assert evaluation["adjacency"] is None and set(evaluation["masks"]) == {"train", "val"}
    assert not any("test" in column for column in second.columns)
    uniform_folder = Path(second.iloc[0].candidate_path) / "condensation_0"
    saved = {path: digest(path) for path in uniform_folder.glob("*.pt")}
    cached, _ = run(tmp_path, [candidate(inner_loss_weighting="uniform")])
    pd.testing.assert_frame_equal(second, cached)
    assert len(state.optimizer_calls) == 2
    assert {path: digest(path) for path in saved} == saved


def test_citation_explicit_mass_preserves_its_existing_identity_without_new_optimizer_keyword(tmp_path,
                                                                                          citation_optimizer):
    state = citation_optimizer
    mass = candidate(inner_loss_weighting="mass")
    first, root = run(tmp_path, [mass])
    folder = Path(first.iloc[0].candidate_path)
    expected = dict(mass, nystrom_schema=3)
    assert folder == root / _fingerprint(expected)
    assert json.loads((folder / "candidate.json").read_text()) == expected
    assert "inner_loss_weighting" not in state.optimizer_calls[0]["kwargs"]
    before = digest(folder / "condensation_0" / "resume.pt")
    second, _ = run(tmp_path, [mass])
    pd.testing.assert_frame_equal(first, second)
    assert len(state.optimizer_calls) == 1
    assert digest(folder / "condensation_0" / "resume.pt") == before


@pytest.mark.parametrize("weighting", ["mse", None, True, 0, float("nan")])
def test_invalid_citation_mode_is_checked_for_all_candidates_before_data(tmp_path, source, weighting):
    with pytest.raises(ValueError, match="inner_loss_weighting"):
        run(tmp_path, [candidate(), candidate(inner_loss_weighting=weighting)])
    assert source.data_calls == source.teacher_fits == 0
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("weighting,marker", [("uniform", None), ("uniform", "mass"), ("mass", "uniform")])
def test_citation_cached_checkpoint_cannot_bypass_mode_validation(tmp_path, citation_optimizer, weighting, marker):
    state = citation_optimizer
    ranking, _ = run(tmp_path, [candidate(inner_loss_weighting=weighting)])
    folder = Path(ranking.iloc[0].candidate_path) / "condensation_0"
    path = folder / "step_000000.pt"
    snapshot = load(path)
    if marker is None:
        snapshot.pop("inner_loss_weighting")
    else:
        snapshot["inner_loss_weighting"] = marker
    save_state(snapshot, path)
    before = len(state.evaluations)
    with pytest.raises(ValueError, match="checkpoint inner weighting"):
        run(tmp_path, [candidate(inner_loss_weighting=weighting)])
    assert len(state.evaluations) == before and len(state.optimizer_calls) == 1


def test_citation_mixed_ranking_selected_uniform_recreates_identity_and_uses_fresh_student_seeds(
        tmp_path, citation_optimizer):
    state = citation_optimizer
    ranking, root = run(tmp_path, [candidate(), candidate(inner_loss_weighting="uniform")])
    choice = ranking[ranking.inner_loss_weighting == "uniform"].iloc[-1].to_dict()
    folder = Path(choice["candidate_path"])
    selected = search.selected_test(root, choice, condensation_seeds=(0, 1), student_seeds=(100, 101),
                                    epochs=2, device="cpu", report_routes=False)
    identity = json.loads((root / "selected.json").read_text())["candidate"]
    assert identity["inner_loss_weighting"] == "uniform"
    assert root / _fingerprint(identity) == folder
    assert set(selected.seed) == {100, 101} and set(selected.condensation_seed) == {0, 1}
    assert len(state.optimizer_calls) == 3 and state.teacher_fits == 1
    assert state.optimizer_calls[-1]["kwargs"]["inner_loss_weighting"] == "uniform"
    assert state.optimizer_calls[-1]["kwargs"]["seed"] == 1
    final_calls = [evaluation for evaluation in state.evaluations if evaluation["seed"] >= 100]
    assert len(final_calls) == 4 and all(set(call["masks"]) == {"train", "val", "test"} for call in final_calls)


@pytest.fixture
def large_optimizer(mocked_pipeline, monkeypatch):
    original = pilot.optimize_nystrom
    mocked_pipeline["nystrom_options"] = []

    def optimize(*args, **kwargs):
        mocked_pipeline["nystrom_options"].append(dict(kwargs))
        original(*args, **kwargs)
        if kwargs.get("inner_loss_weighting") == "uniform":
            path = Path(args[5]) / f"step_{args[6]:06d}.pt"
            snapshot = load(path)
            snapshot["inner_loss_weighting"] = "uniform"
            save_state(snapshot, path)

    monkeypatch.setattr(pilot, "optimize_nystrom", optimize)
    return mocked_pipeline


@pytest.mark.parametrize("dataset,ratio", [("arxiv", .0005), ("flickr", .001), ("reddit", .0005)])
def test_large_uniform_forwarding_preserves_shared_teacher_and_legacy_mass_cache(tmp_path, large_optimizer,
                                                                               dataset, ratio):
    state = large_optimizer
    options = dict(device="cpu", basis=2, rank=2, surrogate="nystrom", steps=1, student_seeds=(0, 1))
    mass, mass_root = pilot.run_pilot(dataset, ratio, tmp_path, **options)
    shared = {path: digest(path) for path in (mass_root.parent / "teacher.pt", mass_root.parent / "propagated_H.pt")
              if path.exists()}
    uniform, uniform_root = pilot.run_pilot(dataset, ratio, tmp_path, inner_loss_weighting="uniform", **options)
    assert mass["status"] == uniform["status"] == "complete"
    assert mass_root != uniform_root and mass_root.parent == uniform_root.parent
    assert mass["candidate"]["inner_loss"] == "exact_mass_ce"
    assert uniform["candidate"]["inner_loss"] == "exact_uniform_ce"
    assert mass["student_recipe"] == uniform["student_recipe"]
    assert state["teacher"] == 1 and state["nystrom"] == 2
    assert "inner_loss_weighting" not in state["nystrom_options"][0]
    assert state["nystrom_options"][1]["inner_loss_weighting"] == "uniform"
    assert all(call["weighting"] == "uniform" and "testing" not in call for call in state["students"])
    assert {path: digest(path) for path in shared} == shared
    for root in (mass_root, uniform_root):
        snapshot = load(root / "condensation" / "step_000001.pt")
        assignment = load(root / "initial_assignment.pt")
        p = torch.nn.functional.one_hot(assignment, 4).double()
        expected = p.T @ state["graph"]["x"].double() / p.sum(0)[:, None]
        torch.testing.assert_close(decode_moments(snapshot["moments"], 3)[0], expected)
    original = digest(mass_root / "condensation" / "resume.pt")
    cached, same = pilot.run_pilot(dataset, ratio, tmp_path, inner_loss_weighting="mass", **options)
    assert cached["status"] == "complete" and same == mass_root
    assert state["nystrom"] == 2 and digest(mass_root / "condensation" / "resume.pt") == original
    assert not any(key.startswith("test_") for key in uniform)


@pytest.mark.parametrize("weighting", ["mse", None, True, 0, float("nan")])
def test_invalid_large_nystrom_mode_fails_before_data(tmp_path, mocked_pipeline, weighting):
    with pytest.raises(ValueError, match="inner_loss_weighting"):
        pilot.run_pilot("arxiv", .0005, tmp_path, device="cpu", basis=2, rank=2,
                        surrogate="nystrom", inner_loss_weighting=weighting)
    assert mocked_pipeline["data"] == mocked_pipeline["teacher"] == 0
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("weighting,marker", [("uniform", None), ("uniform", "mass"), ("mass", "uniform")])
def test_large_cached_endpoint_rejects_missing_or_wrong_weighting_before_students(tmp_path, large_optimizer,
                                                                               weighting, marker):
    state = large_optimizer
    options = dict(device="cpu", basis=2, rank=2, surrogate="nystrom", inner_loss_weighting=weighting, steps=1)
    first, root = pilot.run_pilot("arxiv", .0005, tmp_path, **options)
    assert first["status"] == "complete"
    path = root / "condensation" / "step_000001.pt"
    snapshot = load(path)
    if marker is None:
        snapshot.pop("inner_loss_weighting")
    else:
        snapshot["inner_loss_weighting"] = marker
    save_state(snapshot, path)
    before = len(state["students"])
    report, _ = pilot.run_pilot("arxiv", .0005, tmp_path, **options)
    assert report["status"] == "failed" and "checkpoint inner weighting" in report["reason"]
    assert len(state["students"]) == before and state["nystrom"] == 1


def test_actual_uniform_citation_endpoint_and_same_weight_validation_replay(tmp_path, source, monkeypatch):
    state = source
    state.graph["y"][state.testing[1]] = 999
    original_optimize = nystrom.optimize
    original_replay = student_routes.replay_routes
    optimizer_calls, route_calls = [], []

    def optimize(*args, **kwargs):
        assert kwargs["inner_loss_weighting"] == "uniform"
        optimizer_calls.append(Path(args[5]))
        return original_optimize(*args, **kwargs)

    def replay(path, graph, h, masks, settings, output_path, **kwargs):
        assert set(masks) == {"val"} and masks["val"] is state.validation[1]
        torch.testing.assert_close(h, state.h.float() * 1.6, atol=0, rtol=0)
        route_calls.append(Path(path))
        return original_replay(path, graph, h, masks, settings, output_path, **kwargs)

    monkeypatch.setattr(nystrom, "optimize", optimize)
    monkeypatch.setattr(search, "fit_gcn_diagnostic", fit_gcn_diagnostic)
    monkeypatch.setattr(student_routes, "replay_routes", replay)
    options = dict(steps=2, epochs=3, dropout=0., input_scale=1.6,
                   student_settings=dict(eval_every=1, hidden=3))
    first, root = run(tmp_path, [candidate(inner_loss_weighting="uniform")], **options)
    folder = optimizer_calls[0]
    saved = load(folder / "resume.pt")
    snapshot = load(folder / "step_000002.pt")
    p = logit_block(saved["u"], saved["v"], state.assignment, .05).double().softmax(1)
    q = (state.logits / .3).softmax(1)
    torch.testing.assert_close(snapshot["moments"], p.T @ make_material(state.h.double(), q) / len(p),
                               atol=1e-9, rtol=1e-8)
    selected_paths = list(folder.glob("validation/*/seed_0_selected.pt"))
    original_selected = {path: digest(path) for path in selected_paths}

    def forbid_fit(*args, **kwargs):
        pytest.fail("Cached student route replay cannot create an optimizer or refit the teacher")

    monkeypatch.setattr(torch.optim, "Adam", forbid_fit)
    second, same = run(tmp_path, [candidate(inner_loss_weighting="uniform")], report_routes=True, **options)
    assert same == root and len(optimizer_calls) == 1 and state.teacher_fits == state.hard_assignment_fits == 1
    assert {path: digest(path) for path in selected_paths} == original_selected
    assert set(first.student_recipe) == set(second.student_recipe)
    assert len(route_calls) == len(selected_paths) == 2
    for path in folder.glob("validation/*/seed_0_validation_routes_v1.json"):
        route = json.loads(path.read_text())
        selected = load(path.parent / "seed_0_selected.pt")
        assert route["recipe"]["epoch"] == selected["epoch"]
        assert route["recipe"]["source_fingerprint"] == selected["fingerprint"]
        assert route["recipe"]["test_enabled"] is False
        assert not any("test_" in key for key in route["result"])
    torch.testing.assert_close(torch.tensor(second["mean"].to_numpy()),
                               torch.tensor(second["gcn_val_acc_mean"].to_numpy()), atol=0, rtol=0)
    assert not any("test" in column for column in second.columns)
