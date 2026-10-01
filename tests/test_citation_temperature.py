"""Learnable calibration keeps source geometry and validation selection fixed."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import test_citation_mixing as mixing_fixtures
import torch
from test_learnable_temperature import problem

import src.citation_search as search
import src.student_routes as student_routes
from src.evaluation import fit_gcn_diagnostic
from src.io import _fingerprint, save_state
from src.low_rank_assignment import CachedLowRankMoments, logit_block
from src.moments import augmented, decode_moments, initial_logits, make_material
from src.soft_ce_partition import (
    implicit_moment_gradient,
    optimize_ce_assignment,
    outer_value_gradient,
    solve_head_system,
    solve_inner,
    temperature_labels,
)

source = mixing_fixtures.source
digest = mixing_fixtures.digest
run = mixing_fixtures.run


@pytest.fixture(scope="module", autouse=True)
def small_cpu_jobs():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def candidate(**changes):
    return {**mixing_fixtures.candidate(), **changes}


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


@pytest.fixture
def temperature_optimizer(source, monkeypatch):
    """Keep fitted checkpoint metadata realistic while exposing wrapper inputs."""
    def optimize(z, q, assignment, **kwargs):
        source.optimizer_calls.append(dict(z=z.clone(), q=q.clone(), assignment=assignment.clone(),
                                           kwargs=kwargs))
        learned = "temperature_logits" in kwargs
        p = initial_logits(assignment, 2, kwargs["mixing"]).double().softmax(1)
        folder = kwargs["folder"]
        (folder / "checkpoints").mkdir(parents=True, exist_ok=True)
        checks = kwargs["checkpoint_steps"]
        snapshots = {}
        for step in checks:
            temperature = kwargs.get("temperature_initial", .3) * np.exp(.01 * step)
            labels = ((kwargs["temperature_logits"] / temperature).softmax(1) if learned else q)
            snapshot = dict(step=step, moments=p.T @ make_material(z, labels) / len(p),
                            J_exact=True, teacher_ce=.7 + .001 * step)
            if learned:
                snapshot["temperature"] = float(temperature)
            snapshots[step] = snapshot
            save_state(snapshot, folder / "checkpoints" / f"step_{step:06d}.pt")
        save_state(dict(step=max(checks), snapshots=snapshots,
                        history=[dict(step=step, log_temperature_gradient=.1 + step)
                                 for step in range(max(checks))] if learned else []), folder / "resume.pt")

    monkeypatch.setattr(search, "optimize_ce_assignment", optimize)
    return source


@pytest.fixture
def real_optimizer(source, monkeypatch):
    def optimize(z, q, assignment, **kwargs):
        before = q.clone()
        source.optimizer_calls.append(dict(z=z.clone(), q=before, assignment=assignment.clone(), kwargs=kwargs))
        result = optimize_ce_assignment(z, q, assignment, **kwargs)
        torch.testing.assert_close(q, before, atol=0, rtol=0)
        return result

    monkeypatch.setattr(search, "optimize_ce_assignment", optimize)
    return source


@pytest.mark.parametrize("explicit", [dict(learn_temperature=False),
                                      dict(learn_temperature=False, temperature_lr=.003),
                                      dict(temperature_lr=.003)])
def test_fixed_temperature_defaults_preserve_legacy_source_candidate_recipe_and_resume(
        tmp_path, temperature_optimizer, explicit):
    state = temperature_optimizer
    first, root = run(tmp_path, [candidate()])
    folder = Path(first.iloc[0].candidate_path)
    saved = {path: digest(path) for path in (root / "teacher.pt", root / "propagated_H.pt",
                                            root / "inputs_0.pt", folder / "condensation_0" / "resume.pt")}
    protocols = set(root.glob("screen_protocol_*.json"))
    evaluations = {call["folder"] for call in state.evaluations}
    second, same_root = run(tmp_path, [candidate(**explicit)])
    assert same_root == root and set(first.candidate_path) == set(second.candidate_path)
    pd.testing.assert_frame_equal(first, second)
    assert set(root.glob("screen_protocol_*.json")) == protocols
    assert {call["folder"] for call in state.evaluations} == evaluations
    assert {path: digest(path) for path in saved} == saved
    assert len(state.optimizer_calls) == state.teacher_fits == state.hard_assignment_fits == 1
    assert not ({"temperature_logits", "temperature_initial", "temperature_lr", "outer_targets"}
                & state.optimizer_calls[0]["kwargs"].keys())
    identity = json.loads((folder / "candidate.json").read_text())
    assert "learn_temperature" not in identity and "temperature_lr" not in identity
    assert not any("test" in column for column in first.columns)


def test_learning_forwards_double_logits_and_frozen_outer_q_without_refitting_sources(tmp_path,
                                                                                    temperature_optimizer):
    state = temperature_optimizer
    first, root = run(tmp_path, [candidate()])
    shared = {path: digest(path) for path in (root / "teacher.pt", root / "inputs_0.pt", root / "propagated_H.pt")}
    second, same_root = run(tmp_path, [candidate(learn_temperature=True, temperature_lr=.007)])
    assert same_root == root and set(first.candidate_path).isdisjoint(set(second.candidate_path))
    assert state.teacher_fits == state.hard_assignment_fits == 1
    assert {path: digest(path) for path in shared} == shared
    old, learned = state.optimizer_calls
    options = learned["kwargs"]
    assert options["temperature_logits"].dtype == torch.double
    torch.testing.assert_close(options["temperature_logits"], state.logits, atol=0, rtol=0)
    assert options["temperature_initial"] == .3 and options["temperature_lr"] == .007
    assert options["mass_mode"] == "free" and options["solver_mode"] == "exact"
    assert options["assignment_input"] == "node" and options["assignment_encoder"] == "linear"
    assert not options["outer_targets"].requires_grad
    for key in ("q", "z", "assignment"):
        torch.testing.assert_close(old[key], learned[key], atol=0, rtol=0)
    torch.testing.assert_close(options["outer_targets"], (state.logits / .3).softmax(1), atol=0, rtol=0)
    assert all(set(call["masks"]) == {"train", "val"} for call in state.evaluations)
    assert not any("test" in column for column in second.columns)


@pytest.mark.parametrize("key", ["T", "lr", "temperature_lr"])
@pytest.mark.parametrize("value", [0., -1., float("nan"), float("inf"), -float("inf"), True, "0.003"])
def test_invalid_numeric_controls_are_rejected_before_data_or_output(tmp_path, source, key, value):
    with pytest.raises(ValueError, match=key):
        run(tmp_path, [candidate(learn_temperature=True, **{key: value})])
    assert source.data_calls == source.teacher_fits == 0
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("value", [0, 1, "True", "False", None, float("nan")])
def test_mode_requires_boolean_before_loading(tmp_path, source, value):
    with pytest.raises(ValueError, match="learn_temperature"):
        run(tmp_path, [candidate(learn_temperature=value)])
    assert source.data_calls == 0 and not list(tmp_path.iterdir())


@pytest.mark.parametrize("changes", [dict(method=method) for method in ("mlp", "distance", "nystrom", "coarsening")]
                         + [dict(mass_mode="uniform"), dict(node_weighting=True), dict(solver_mode="tracking"),
                            dict(assignment_input="features"), dict(assignment_encoder="mlp"),
                            dict(feature_control="free"), dict(train_target_mix=.1),
                            dict(train_target_mix=True), dict(train_target_mix=float("nan"))])
def test_unsupported_temperature_modes_fail_before_loading(tmp_path, source, changes):
    with pytest.raises(ValueError, match="Learnable temperature"):
        run(tmp_path, [candidate(learn_temperature=True, **changes)])
    assert source.data_calls == source.teacher_fits == 0 and not list(tmp_path.iterdir())


def test_all_candidates_are_validated_before_first_source_fit(tmp_path, source):
    with pytest.raises(ValueError, match="temperature_lr"):
        run(tmp_path, [candidate(), candidate(learn_temperature=False, temperature_lr=.004)])
    assert source.data_calls == source.teacher_fits == 0
    assert not list(tmp_path.iterdir())


def test_mixed_ranking_roundtrips_default_and_learned_identity_and_selection_cache(tmp_path, temperature_optimizer):
    state = temperature_optimizer
    ranking, root = run(tmp_path, [candidate(), candidate(learn_temperature=True)])
    assert set(ranking.learn_temperature) == {False, True}
    assert set(ranking.temperature_lr) == {.003} and not ranking.temperature_lr.isna().any()
    assert {"learned_temperature_mean", "log_temperature_gradient_mean", "surrogate_ce_mean",
            "surrogate_J_exact"} <= set(ranking)
    for row in ranking.to_dict("records"):
        # A pandas row yields a numpy boolean; selected_test must retain its meaning.
        choice = dict(row, learn_temperature=np.bool_(row["learn_temperature"]))
        options = dict(condensation_seeds=(0, 1), student_seeds=(100, 101), epochs=2,
                       device="cpu", report_routes=False)
        result = search.selected_test(root, choice, **options)
        assert set(result.condensation_seed) == {0, 1} and set(result.seed) == {100, 101}
        selection = json.loads((root / "selected.json").read_text())
        assert root / _fingerprint(selection["candidate"]) == Path(row["candidate_path"])
        assert selection["selection"] == "validation_only"
        if row["learn_temperature"]:
            assert selection["candidate"]["learn_temperature"] is True
            assert selection["candidate"]["temperature_lr"] == .003
        else:
            assert "learn_temperature" not in selection["candidate"]
            assert "temperature_lr" not in selection["candidate"]
        before = set(root.glob("selected_*.json"))
        calls = len(state.optimizer_calls)
        search.selected_test(root, choice, **options)
        assert set(root.glob("selected_*.json")) == before and len(state.optimizer_calls) == calls
        fresh = [evaluation for evaluation in state.evaluations
                 if evaluation["folder"].parent.parent.parent == Path(row["candidate_path"])
                 and "test" in evaluation["masks"]]
        assert fresh and all(evaluation["seed"] in (100, 101) for evaluation in fresh)
    assert state.teacher_fits == 1 and state.hard_assignment_fits == 2


@pytest.mark.parametrize("changes", [dict(learn_temperature=1), dict(learn_temperature="True"),
                                      dict(learn_temperature=float("nan")),
                                      dict(temperature_lr=float("nan")), dict(temperature_lr=True),
                                      dict(temperature_lr=".003"), dict(T=float("inf")),
                                      dict(learn_temperature=False, temperature_lr=.004)])
def test_malformed_selected_controls_fail_before_loading_source(tmp_path, temperature_optimizer, changes):
    ranking, root = run(tmp_path, [candidate(learn_temperature=True)])
    before = temperature_optimizer.data_calls
    with pytest.raises(ValueError):
        search.selected_test(root, dict(ranking.iloc[0], **changes), device="cpu", report_routes=False)
    assert temperature_optimizer.data_calls == before
    assert not (root / "selected.json").exists()


def test_fixed_and_learned_real_p0_have_identical_u_v_geometry_q_and_moments(tmp_path, real_optimizer):
    fixed, root = run(tmp_path, [candidate()], steps=0)
    learned, same = run(tmp_path, [candidate(learn_temperature=True)], steps=0)
    assert same == root
    fixed_state = load(Path(fixed.iloc[0].candidate_path) / "condensation_0" / "resume.pt")
    learned_state = load(Path(learned.iloc[0].candidate_path) / "condensation_0" / "resume.pt")
    assert len(fixed_state["parameters"]) == 2 and len(learned_state["parameters"]) == 3
    for left, right in zip(fixed_state["parameters"], learned_state["parameters"][:2], strict=True):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    torch.testing.assert_close(fixed_state["initial_moments"], learned_state["initial_moments"], atol=1e-14, rtol=0)
    for key in ("x", "labels", "q", "mass"):
        torch.testing.assert_close(real_optimizer.evaluations[0][key], real_optimizer.evaluations[1][key], atol=0, rtol=0)
    row = learned.iloc[0]
    assert row.learned_temperature_mean == pytest.approx(.3)
    assert row.surrogate_J_exact and np.isnan(row.log_temperature_gradient_mean)


@pytest.mark.parametrize("inner_loss", ["mass", "uniform"])
def test_real_two_steps_derive_learned_qc_from_p_but_hold_outer_reference_fixed(tmp_path, real_optimizer, inner_loss):
    ranking, _ = run(tmp_path, [candidate(learn_temperature=True, inner_loss_weighting=inner_loss)], steps=2)
    state = load(Path(ranking.iloc[0].candidate_path) / "condensation_0" / "resume.pt")
    snapshot = state["snapshots"][2]
    z = real_optimizer.optimizer_calls[0]["z"]
    q_reference = (real_optimizer.logits / .3).softmax(1)
    u, v, log_t = state["parameters"]
    p = logit_block(u, v, real_optimizer.assignment, .05).double().softmax(1)
    learned_q = temperature_labels(real_optimizer.logits, log_t)
    assert snapshot["temperature"] == pytest.approx(float(log_t.exp()), rel=1e-14)
    assert not torch.equal(learned_q, q_reference)
    expected = p.T @ make_material(z, learned_q) / len(z)
    torch.testing.assert_close(snapshot["moments"], expected, atol=1e-12, rtol=1e-12)
    centers, labels, mass = decode_moments(snapshot["moments"], z.shape[1])
    torch.testing.assert_close(labels, p.T @ learned_q / p.sum(0)[:, None], atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(mass, p.mean(0), atol=1e-12, rtol=1e-12)
    assert not torch.allclose(labels, p.T @ q_reference / p.sum(0)[:, None], atol=1e-7, rtol=0)
    loss, _ = outer_value_gradient(z, q_reference, snapshot["theta"])
    assert snapshot["teacher_ce"] == pytest.approx(loss, abs=1e-12)
    for evaluation in real_optimizer.evaluations:
        torch.testing.assert_close(evaluation["q"], q_reference, atol=0, rtol=0)
        assert set(evaluation["masks"]) == {"train", "val"}
    endpoint = ranking.loc[ranking.step == 2].iloc[0]
    assert endpoint.learned_temperature_mean == pytest.approx(snapshot["temperature"])
    assert endpoint.surrogate_ce_mean == pytest.approx(loss)
    assert endpoint.surrogate_J_exact
    assert endpoint.log_temperature_gradient_mean == pytest.approx(state["history"][1]["log_temperature_gradient"])
    assert torch.isfinite(centers).all()


def test_real_wrapper_resume_matches_uninterrupted_and_respects_temperature_cache_boundary(tmp_path, real_optimizer):
    proposal = candidate(learn_temperature=True)
    short, root = run(tmp_path / "split", [proposal], steps=1)
    folder = Path(short.iloc[0].candidate_path) / "condensation_0"
    old_state = load(folder / "resume.pt")
    continued, same = run(tmp_path / "split", [proposal], steps=2)
    assert same == root and set(short.candidate_path) == set(continued.candidate_path)
    assert real_optimizer.optimizer_calls[-1]["kwargs"]["resume_state"]["step"] == 1
    resumed = load(folder / "resume.pt")
    full, _ = run(tmp_path / "full", [proposal], steps=2)
    uninterrupted = load(Path(full.iloc[0].candidate_path) / "condensation_0" / "resume.pt")
    for actual, expected in zip(resumed["parameters"], uninterrupted["parameters"], strict=True):
        torch.testing.assert_close(actual, expected, atol=1e-9, rtol=1e-7)
    torch.testing.assert_close(resumed["snapshots"][2]["moments"], uninterrupted["snapshots"][2]["moments"],
                               atol=1e-9, rtol=1e-7)
    q = real_optimizer.optimizer_calls[0]["q"]
    z = real_optimizer.optimizer_calls[0]["z"]
    kwargs = dict(real_optimizer.optimizer_calls[0]["kwargs"])
    kwargs.update(steps=2, folder=None, save_resume=False, resume_state=old_state)
    for changes in (dict(temperature_lr=.004), dict(temperature_logits=None, outer_targets=None)):
        with pytest.raises(ValueError, match="Resume state does not match"):
            optimize_ce_assignment(z, q, real_optimizer.assignment, **{**kwargs, **changes})
    saved = digest(folder / "resume.pt")
    other, _ = run(tmp_path / "split", [candidate(learn_temperature=True, temperature_lr=.004)], steps=0)
    assert set(other.candidate_path).isdisjoint(set(continued.candidate_path))
    assert real_optimizer.optimizer_calls[-1]["kwargs"]["resume_state"] is None
    assert digest(folder / "resume.pt") == saved


def test_real_selected_recreation_keeps_learned_endpoint_labels_and_frozen_teacher_reference(tmp_path, real_optimizer):
    ranking, root = run(tmp_path, [candidate(learn_temperature=True)], steps=2)
    choice = ranking.loc[ranking.step == 2].iloc[0]
    candidate_path = Path(choice.candidate_path)
    original = digest(candidate_path / "condensation_0" / "resume.pt")
    result = search.selected_test(root, choice, condensation_seeds=(0, 1), student_seeds=(100, 101),
                                  epochs=2, device="cpu", report_routes=False)
    assert set(result.condensation_seed) == {0, 1} and set(result.seed) == {100, 101}
    assert len(real_optimizer.optimizer_calls) == 2
    assert digest(candidate_path / "condensation_0" / "resume.pt") == original
    assert real_optimizer.optimizer_calls[-1]["kwargs"]["factor_seed"] == 1
    assert real_optimizer.optimizer_calls[-1]["kwargs"]["temperature_lr"] == .003
    assert real_optimizer.optimizer_calls[-1]["kwargs"]["resume_state"] is None
    reference = (real_optimizer.logits / .3).softmax(1)
    for cond_seed in (0, 1):
        folder = candidate_path / f"condensation_{cond_seed}"
        endpoint = load(folder / "checkpoints" / "step_000002.pt")
        _, expected_labels, _ = decode_moments(endpoint["moments"], real_optimizer.h.shape[1])
        final = [item for item in real_optimizer.evaluations
                 if item["folder"].parent.parent == folder and "test" in item["masks"]]
        assert {item["seed"] for item in final} == {100, 101}
        for item in final:
            torch.testing.assert_close(item["labels"], expected_labels.float(), atol=0, rtol=0)
            torch.testing.assert_close(item["q"], reference, atol=0, rtol=0)
    selected = json.loads((root / "selected.json").read_text())
    assert selected["step"] == 2 and selected["candidate"]["learn_temperature"] is True
    assert root / _fingerprint(selected["candidate"]) == candidate_path


def test_learning_requires_persisted_resume_before_evaluation(tmp_path, temperature_optimizer, monkeypatch):
    original = search.optimize_ce_assignment

    def missing_resume(*args, **kwargs):
        original(*args, **kwargs)
        (kwargs["folder"] / "resume.pt").unlink()

    monkeypatch.setattr(search, "optimize_ce_assignment", missing_resume)
    with pytest.raises(ValueError, match="resume state is missing"):
        run(tmp_path, [candidate(learn_temperature=True)])
    assert not temperature_optimizer.evaluations


@pytest.mark.parametrize("changes", [dict(J_exact=False), dict(J_exact=1), dict(temperature=0),
                                      dict(temperature=float("nan")), dict(temperature=True),
                                      dict(teacher_ce=float("inf")), dict(teacher_ce=".7")])
def test_bad_saved_exact_temperature_diagnostics_are_refused_before_student_fit(tmp_path,
                                                                             temperature_optimizer, changes):
    ranking, _ = run(tmp_path, [candidate(learn_temperature=True)])
    folder = Path(ranking.iloc[0].candidate_path) / "condensation_0"
    path = folder / "checkpoints" / "step_000000.pt"
    snapshot = load(path)
    save_state({**snapshot, **changes}, path)
    before = len(temperature_optimizer.evaluations)
    with pytest.raises(ValueError, match="checkpoint diagnostics"):
        run(tmp_path, [candidate(learn_temperature=True)])
    assert len(temperature_optimizer.evaluations) == before


def test_history_reports_latest_gradient_at_selected_step_and_ignores_future_updates(tmp_path, temperature_optimizer):
    ranking, _ = run(tmp_path, [candidate(learn_temperature=True)], steps=2, checkpoints=(1,))
    folder = Path(ranking.iloc[0].candidate_path) / "condensation_0"
    state = load(folder / "resume.pt")
    # An extended resume can have future rows unrelated to a selected earlier checkpoint.
    state["history"].append(dict(step=3, log_temperature_gradient=float("nan")))
    save_state(state, folder / "resume.pt")
    selected, root = run(tmp_path, [candidate(learn_temperature=True)], steps=1)
    assert set(selected.step) == {0, 1}
    assert selected.loc[selected.step == 0, "log_temperature_gradient_mean"].item() == pytest.approx(.1)
    assert selected.loc[selected.step == 1, "log_temperature_gradient_mean"].item() == pytest.approx(1.1)
    frames = [pd.read_csv(path) for path in root.glob("screen_*.csv")]
    assert any(set(frame.temperature_gradient_step) == {0, 1} for frame in frames if set(frame.step) == {0, 1})
    state["history"][0]["log_temperature_gradient"] = float("nan")
    save_state(state, folder / "resume.pt")
    before = len(temperature_optimizer.evaluations)
    with pytest.raises(ValueError, match="temperature gradient"):
        run(tmp_path, [candidate(learn_temperature=True)], steps=1)
    assert len(temperature_optimizer.evaluations) == before


def test_learned_routes_reuse_selected_weights_report_validation_only_and_aggregate(tmp_path,
                                                                                real_optimizer, monkeypatch):
    state = real_optimizer
    state.graph["y"][state.testing[1]] = 999
    native = []
    routes = []

    def fit(*args, **kwargs):
        assert set(args[5]) == {"train", "val"}
        native.append(Path(kwargs["folder"]))
        return fit_gcn_diagnostic(*args, **kwargs)

    original = student_routes.replay_routes

    def replay(path, graph, h, masks, settings, output_path, **kwargs):
        assert set(masks) == {"val"} and masks["val"] is state.validation[1]
        torch.testing.assert_close(h, state.h.float() * 1.5, atol=0, rtol=0)
        routes.append(Path(path))
        return original(path, graph, h, masks, settings, output_path, **kwargs)

    monkeypatch.setattr(search, "fit_gcn_diagnostic", fit)
    monkeypatch.setattr(student_routes, "replay_routes", replay)
    options = dict(steps=2, student_seeds=(0, 1), epochs=3, dropout=0., input_scale=1.5,
                   student_settings=dict(eval_every=1, hidden=3))
    first, root = run(tmp_path, [candidate(learn_temperature=True)], **options)
    folder = Path(first.iloc[0].candidate_path) / "condensation_0"
    immutable = {path: digest(path) for path in [root / "teacher.pt", folder / "resume.pt",
                                               *folder.glob("validation/*/seed_*_selected.pt")]}

    def forbid_optimizer(*args, **kwargs):
        pytest.fail("Calibrated validation routes must replay already selected weights")

    monkeypatch.setattr(torch.optim, "Adam", forbid_optimizer)
    second, same = run(tmp_path, [candidate(learn_temperature=True)], report_routes=True, **options)
    assert same == root and set(first.candidate_path) == set(second.candidate_path)
    assert set(first.student_recipe) == set(second.student_recipe)
    assert len(native) == 8 and len(routes) == 4 and len(state.optimizer_calls) == 1
    assert {path: digest(path) for path in immutable} == immutable
    assert state.teacher_fits == state.hard_assignment_fits == 1
    for metric in ("gcn_val_acc", "mlp_val_acc", "gcn_val_ce", "mlp_val_ce"):
        assert {f"{metric}_mean", f"{metric}_std"} <= set(second)
    for step in (0, 2):
        endpoint = second.loc[second.step == step].iloc[0]
        assert endpoint["gcn_val_acc_mean"] == endpoint["mean"]
        assert endpoint["surrogate_J_exact"]
        raw = [load(path) for path in routes if f"step_{step}_" in str(path)]
        assert len(raw) == 2
        for path in [path for path in routes if f"step_{step}_" in str(path)]:
            route = json.loads(path.with_name(path.name.replace("_selected.pt", "_validation_routes_v1.json")).read_text())
            selected = load(path)
            assert route["recipe"]["epoch"] == selected["epoch"]
            assert route["recipe"]["source_fingerprint"] == selected["fingerprint"]
            assert route["recipe"]["test_enabled"] is False
            assert not any("test_" in key for key in route["result"])
    assert not any("test_" in column for column in second.columns)


def test_uniform_inner_implicit_temperature_gradient_matches_resolved_finite_difference():
    z, logits, u, v, assignment = problem()
    reference = temperature_labels(logits, z.new_tensor(np.log(.3))).detach()
    log_t = z.new_tensor(np.log(.7)).requires_grad_()

    def fitted(value):
        moments = CachedLowRankMoments.apply(u, v, assignment,
                                           make_material(z, temperature_labels(logits, value)), .05, 5)
        centers, labels, mass = decode_moments(moments.detach(), z.shape[1])
        uniform = torch.full_like(mass, 1 / len(mass))
        head = solve_inner(centers, labels, uniform, .2, grad_tol=1e-11)
        assert head["inner_converged"]
        loss, gradient = outer_value_gradient(z, reference, head["theta"])
        return moments, centers, labels, uniform, head["theta"], loss, gradient

    moments, centers, labels, uniform, theta, _, gradient = fitted(log_t)
    vector, diagnostic = solve_head_system(augmented(centers), labels, uniform, theta, .2, gradient, rtol=1e-10)
    assert diagnostic["cg_converged"]
    direction = implicit_moment_gradient(moments, z.shape[1], theta, vector, .2, "uniform")
    (actual,) = torch.autograd.grad(moments, log_t, direction)
    epsilon = 1e-3
    finite = (fitted(log_t.detach() + epsilon)[5] - fitted(log_t.detach() - epsilon)[5]) / (2 * epsilon)
    assert abs(float(actual)) > .01  # A zero-gradient path cannot satisfy this check.
    torch.testing.assert_close(actual, actual.new_tensor(finite), atol=2e-7, rtol=2e-5)
