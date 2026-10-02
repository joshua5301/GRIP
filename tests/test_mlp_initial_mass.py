import copy

import pytest
import torch

from src import citation_search as citation
from src import mlp_initial_mass as helper
from src import soft_ce_partition as core
from src.io import _fingerprint, save_json, save_state
from src.low_rank_assignment import LowRankMoments, assignment_inputs, encode_nodes
from src.moments import make_material
from src.transforms import fit_transform


def problem():
    generator = torch.Generator().manual_seed(42)
    z = torch.randn(12, 3, generator=generator, dtype=torch.float64)
    q = torch.randn(12, 2, generator=generator, dtype=torch.float64).softmax(1)
    assignment = torch.tensor([0] * 6 + [1] * 3 + [2] * 3)
    return z, q, assignment


def options():
    return dict(penalty=.1, lr=.005, assignment_rank=2, assignment_input="features",
                assignment_encoder="mlp", encoder_hidden=5, mass_mode="initial",
                balance_steps=5000, balance_tol=1e-8, balance_backend="chunked",
                inner_loss_weighting="uniform", save_resume=True, save_assignment=False,
                mlp_initial_mass_schema=1, inner_tol=1e-9, inner_method="newton_first")


def load(folder):
    return torch.load(folder / "resume.pt", map_location="cpu", weights_only=False)


def expected(state, values):
    z, q, assignment = values
    return helper.original_context(z, q, assignment, 2, 5, 0, .05, 4096, state["config"],
                                   state["config"]["mlp_initial_mass_source"])


def reseal(state):
    state["initial_mass_content_sha256"] = helper.seal({key: value for key, value in state.items()
                                                       if key != "initial_mass_content_sha256"})
    return state


def test_real_core_p0_exact_and_source_target(tmp_path):
    values = problem()
    core.optimize_ce_assignment(*values, steps=0, folder=tmp_path, **options())
    state = load(tmp_path)
    target, context, params = expected(state, values)
    z, q, assignment = values
    u = encode_nodes(assignment_inputs(z, q, "features"), params[:-1])
    moments = LowRankMoments.apply(u, params[-1], assignment, make_material(z, q), .05, 4096)
    snapshot = state["snapshots"][0]
    assert torch.equal(snapshot["moments"], moments)
    assert torch.equal(snapshot["column_dual"], torch.zeros_like(target))
    assert float((moments[:, 0] / target - 1).abs().max()) < 1e-15
    assert target.max() > 1.9 * target.min()
    helper.validate_resume(state, *values, target, context, params, 0, tmp_path)
    assert all(torch.equal(a, b) for a, b in zip(params, snapshot["parameters"], strict=True))


def test_real_one_update_resume_to_25_same_trajectory_rng_and_checkpoints(tmp_path):
    values = problem()
    torch.manual_seed(900)
    rng = torch.get_rng_state().clone()
    full_folder, resumed_folder = tmp_path / "full", tmp_path / "resumed"
    full = core.optimize_ce_assignment(*values, steps=25, checkpoint_steps=(0, 1, 25), folder=full_folder, **options())
    core.optimize_ce_assignment(*values, steps=1, checkpoint_steps=(0, 1), folder=resumed_folder, **options())
    one = load(resumed_folder)
    zero_bytes = (resumed_folder / "checkpoints" / "step_000000.pt").read_bytes()
    stop_calls = 0

    def stop():
        nonlocal stop_calls
        stop_calls += 1
        return stop_calls >= 4

    before = (resumed_folder / "resume.pt").read_bytes()
    with pytest.raises(InterruptedError):
        core.optimize_ce_assignment(*values, steps=25, checkpoint_steps=(0, 25), folder=resumed_folder,
                                    resume_state=one, stop=stop, **options())
    assert (resumed_folder / "resume.pt").read_bytes() == before
    resumed = core.optimize_ce_assignment(*values, steps=25, checkpoint_steps=(0, 25), folder=resumed_folder,
                                          resume_state=one, **options())
    assert torch.equal(torch.get_rng_state(), rng)
    assert set(resumed["checkpoints"]) == {0, 1, 25}
    assert (resumed_folder / "checkpoints" / "step_000000.pt").read_bytes() == zero_bytes
    for key in ("moments", "theta", "column_dual"):
        assert torch.equal(full["checkpoints"][25][key], resumed["checkpoints"][25][key])
    assert all(torch.equal(a, b) for a, b in zip(load(full_folder)["parameters"], load(resumed_folder)["parameters"], strict=True))


@pytest.mark.parametrize("mutation", ["target", "initializer", "source", "device", "mode", "schema", "dual", "parameter", "moments", "head", "checksum",
                                      "adam_lr", "adam_wd", "adam_betas", "adam_count", "adam_missing", "adam_nan", "adam_negative", "history", "initial_moments", "scale"])
def test_cache_corruption_rejects_before_core_mutation(tmp_path, monkeypatch, mutation):
    values = problem()
    folder = tmp_path / "valid"
    core.optimize_ce_assignment(*values, steps=1, folder=folder, **options())
    state = copy.deepcopy(load(folder))
    if mutation == "target":
        state["mass_target"] = torch.full_like(state["mass_target"], 1 / 3)
        state["initial_mass_context"]["target"] = helper.digest(state["mass_target"])
    elif mutation in ("initializer", "source", "device"):
        state["initial_mass_context"][mutation] = "forged"
    elif mutation == "mode":
        state["mass_mode"] = "free"
    elif mutation == "schema":
        state["mlp_initial_mass_schema"] = True
    elif mutation == "dual":
        state["dual"] = torch.zeros(1, dtype=torch.float64)
    elif mutation == "parameter":
        state["parameters"][0][0, 0] += .1
    elif mutation == "moments":
        state["snapshots"][1]["moments"] = state["snapshots"][1]["moments"][:1]
        reseal(state["snapshots"][1])
    elif mutation == "head":
        state["snapshots"][1]["theta"].add_(1)
        reseal(state["snapshots"][1])
    elif mutation == "checksum":
        state["scale"] += .1
    elif mutation == "history":
        state["history"].pop(0)
    elif mutation == "initial_moments":
        state["initial_moments"] = state["initial_moments"] + .1
    elif mutation == "scale":
        state["scale"] += .1
    elif mutation.startswith("adam_"):
        group = state["optimizer"]["param_groups"][0]
        if mutation in ("adam_lr", "adam_wd", "adam_betas"):
            group[{"adam_lr": "lr", "adam_wd": "weight_decay", "adam_betas": "betas"}[mutation]] = .7
        else:
            adam = state["optimizer"]["state"][0]
            if mutation == "adam_count":
                adam["step"] += 1
            elif mutation == "adam_missing":
                adam.pop("exp_avg")
            elif mutation == "adam_nan":
                adam["exp_avg"].fill_(float("nan"))
            else:
                adam["exp_avg_sq"].fill_(-1)
    if mutation != "checksum":
        reseal(state)
    monkeypatch.setattr(core, "solve_inner_newton_first", lambda *a, **kw: pytest.fail("inner fit ran"))
    absent = tmp_path / "absent"
    with pytest.raises((ValueError, RuntimeError)):
        core.optimize_ce_assignment(*values, steps=2, folder=absent, resume_state=state, **options())
    assert not absent.exists()


@pytest.mark.parametrize("field", ["z", "q", "assignment"])
def test_changed_current_input_rejects_even_self_signed_cache(tmp_path, field):
    values = list(problem())
    core.optimize_ce_assignment(*values, steps=0, folder=tmp_path / "old", **options())
    state = load(tmp_path / "old")
    values[{"z": 0, "q": 1, "assignment": 2}[field]] = values[{"z": 0, "q": 1, "assignment": 2}[field]].clone()
    if field == "z":
        values[0][0, 0] += .1
    elif field == "q":
        values[1] = values[1].roll(1, 0)
    else:
        values[2] = values[2].roll(1, 0)
    with pytest.raises(ValueError):
        core.optimize_ce_assignment(*values, steps=1, folder=tmp_path / "new", resume_state=state, **options())
    assert not (tmp_path / "new").exists()


def test_orphan_checkpoint_and_stopped_entry_reject_without_mutation(tmp_path):
    values = problem()
    core.optimize_ce_assignment(*values, steps=0, folder=tmp_path, **options())
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="verifiable resume"):
        core.optimize_ce_assignment(*values, steps=1, folder=tmp_path, **options())
    with pytest.raises(InterruptedError):
        core.optimize_ce_assignment(*values, steps=1, folder=tmp_path / "absent", stop=lambda: True, **options())
    assert before == {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}


def candidate():
    return helper.candidate_controls(dict(method="mlp", width=5, rank=2, penalty=.1, lr=.005, T=.3,
                                          inner_loss_weighting="uniform", mass_mode="initial", mlp_initial_mass_schema=1))


@pytest.mark.parametrize("change", [dict(method="low_rank"), dict(mlp_initial_mass_schema=True), dict(balance_steps=20000),
                                  dict(balance_tol=1e-7), dict(balance_backend="cached"), dict(inner_loss_weighting="mass"),
                                  dict(train_target_mix=.5), dict(learn_temperature=True), dict(mixing=.1), dict(mass_target=[.5, .5]), dict(rank=True), dict(width=0), dict(penalty=float("nan")),
                                  dict(mlp_initial_mass_source_digest="forged"), dict(outer_targets="forged")])
def test_unsupported_preload_before_dataset(tmp_path, monkeypatch, change):
    supplied = {**candidate(), **change}
    monkeypatch.setattr(citation, "_prepare_dataset", lambda *a, **kw: pytest.fail("dataset loaded"))
    with pytest.raises(ValueError):
        citation.run_screen("cora", .013, tmp_path, [supplied], device="cpu")


def source_fixture(tmp_path, monkeypatch):
    monkeypatch.setitem(citation.BUDGET, ("cora", .013), 3)
    generator = torch.Generator().manual_seed(88)
    x = torch.randn(12, 3, generator=generator)
    graph = dict(x=x, y=torch.arange(12) % 2, adj=torch.eye(12).to_sparse_csr())
    train = torch.arange(12) < 4
    val = (None, (torch.arange(12) >= 4) & (torch.arange(12) < 8))
    test = (None, torch.arange(12) >= 8)
    h = x.clone()
    monkeypatch.setattr(citation, "_prepare_dataset", lambda *a, **kw: (graph, train, val, test, h))
    config = citation._legacy_teacher_config("cora", .013, graph, train, val, test, "default")
    root = tmp_path / "cora" / "ratio_0.013" / _fingerprint(config)
    root.mkdir(parents=True)
    save_json(config, root / "config.json")
    citation.fixed_propagated_features(h, config, root)
    logits = torch.randn(12, 2, generator=generator, dtype=torch.float64)
    save_state(dict(logits=logits, gamma=.01), root / "teacher.pt")
    z, transform = fit_transform(h.double())
    save_state(dict(z=z, assignment=problem()[2], transform=vars(transform)), root / "inputs_0.pt")
    return root, graph, train, val, test, h


def test_actual_toy_probe_to_screen_25_same_config_no_refits(tmp_path, monkeypatch):
    root, *_ = source_fixture(tmp_path, monkeypatch)
    source_before = {p: p.read_bytes() for p in root.iterdir() if p.is_file()}
    control = candidate()
    probe = helper.prepare_probe("cora", .013, tmp_path, control, device="cpu")
    assert probe["step"] == 1 and probe["student_fits"] == 0 and not probe["cached"]
    folder = root / _fingerprint(control) / "condensation_0"
    one = load(folder)
    calls = []
    monkeypatch.setattr(citation, "teacher_logits", lambda *a, **kw: pytest.fail("teacher fit ran"))
    monkeypatch.setattr(citation, "feature_kmeans", lambda *a, **kw: pytest.fail("initializer fit ran"))

    def student(*args, **kwargs):
        calls.append((kwargs["seed"] if "seed" in kwargs else args[6], args[0].clone()))
        return dict(val_acc=75., val_ce=.5)

    monkeypatch.setattr(citation, "fit_gcn_diagnostic", student)
    ranking, actual = citation.run_screen("cora", .013, tmp_path, [control], steps=25, checkpoints=(0, 25),
                                          student_seeds=(3100,), dropout=0, epochs=1, device="cpu")
    assert actual == root and len(ranking) == 2 and len(calls) == 2
    state = load(folder)
    assert state["config"] == one["config"]
    assert state["step"] == 25 and set(state["snapshots"]) == {0, 1, 25}
    assert all(p.read_bytes() == contents for p, contents in source_before.items())
    monkeypatch.setattr(core, "solve_inner_newton_first", lambda *a, **kw: pytest.fail("core fit ran"))
    again = helper.prepare_probe("cora", .013, tmp_path, control, device="cpu")
    assert again["cached"] is True


@pytest.mark.parametrize("tamper", ["teacher", "hard", "z", "config", "snapshot", "target", "optimizer", "orphan", "candidate"])
def test_cached_screen_and_selected_preflight_reject_before_fits_or_selection(tmp_path, monkeypatch, tamper):
    root, *_ = source_fixture(tmp_path, monkeypatch)
    control = candidate()
    helper.prepare_probe("cora", .013, tmp_path, control, device="cpu")
    folder = root / _fingerprint(control) / "condensation_0"
    save_json(control, folder.parent / "candidate.json")
    if tamper == "candidate":
        save_json({**control, "lr": .99}, folder.parent / "candidate.json")
    elif tamper == "teacher":
        state = torch.load(root / "teacher.pt", weights_only=False)
        state["logits"][0, 0] += .1
        save_state(state, root / "teacher.pt")
    elif tamper in ("hard", "z"):
        state = torch.load(root / "inputs_0.pt", weights_only=False)
        if tamper == "hard":
            state["assignment"] = state["assignment"].roll(1, 0)
        else:
            state["z"][0, 0] += .1
        save_state(state, root / "inputs_0.pt")
    elif tamper == "config":
        state = __import__("json").loads((root / "config.json").read_text())
        state["data_digest"] = "forged"
        save_json(state, root / "config.json")
    elif tamper == "snapshot":
        state = torch.load(folder / "checkpoints" / "step_000001.pt", weights_only=False)
        state["moments"][0, 1] += .1
        save_state(reseal(state), folder / "checkpoints" / "step_000001.pt")
    elif tamper == "orphan":
        (folder / "resume.pt").unlink()
    else:
        state = load(folder)
        if tamper == "target":
            state["mass_target"] = torch.full_like(state["mass_target"], 1 / 3)
            state["initial_mass_context"]["target"] = helper.digest(state["mass_target"])
        else:
            state["optimizer"]["param_groups"][0]["lr"] = .99
        save_state(reseal(state), folder / "resume.pt")
    monkeypatch.setattr(citation, "fit_gcn_diagnostic", lambda *a, **kw: pytest.fail("student fit ran"))
    monkeypatch.setattr(citation, "optimize_ce_assignment", lambda *a, **kw: pytest.fail("core fit ran"))
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    with pytest.raises(ValueError):
        citation.run_screen("cora", .013, tmp_path, [control], steps=1, student_seeds=(0,), device="cpu")
    assert before == {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    choice = dict(control, step=1, candidate_path=str(folder.parent))
    with pytest.raises(ValueError):
        citation.selected_test(root, choice, condensation_seeds=(0,), student_seeds=(0,), device="cpu")
    assert before == {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}


@pytest.mark.parametrize("change", [dict(step=True), dict(step=1.5), dict(mass_target=[.5, .5]),
                                  dict(mass_mode="free"), dict(mlp_initial_mass_schema=False)])
def test_selected_raw_initial_controls_reject_before_dataset(tmp_path, monkeypatch, change):
    root, *_ = source_fixture(tmp_path, monkeypatch)
    control = candidate()
    folder = root / _fingerprint(control)
    folder.mkdir()
    save_json(control, folder / "candidate.json")
    choice = {**control, "step": 1, "candidate_path": str(folder), **change}
    monkeypatch.setattr(citation, "_prepare_dataset", lambda *a, **kw: pytest.fail("dataset loaded"))
    with pytest.raises(ValueError):
        citation.selected_test(root, choice, device="cpu")


def test_probe_dispatch_forwards_stop(monkeypatch):
    from src import research_loop
    seen = []
    stop = lambda: False
    monkeypatch.setattr(helper, "prepare_probe", lambda **kwargs: seen.append(kwargs) or {"step": 1})
    assert research_loop.dispatch(dict(kind="citation_mlp_initial_probe", options=dict(dataset="cora")), stop) == {"step": 1}
    assert seen == [dict(dataset="cora", stop=stop)]



def test_legacy_candidate_ids_and_explicit_free_unchanged():
    raw = dict(method="mlp", width=256, lr=.05, T=.3, rank=32, penalty=1e-4,
               initialization="teacher_balanced", alpha=.3, inner_loss_weighting="uniform")
    assert citation._candidate_nystrom_mass(raw) == raw
    assert _fingerprint(raw) == "2755e0bc56ec"
    assert citation._candidate_nystrom_mass({**raw, "mass_mode": "free"}) == {**raw, "mass_mode": "free"}


def test_valid_selected_preflight_reaches_generation_only_after_checks(tmp_path, monkeypatch):
    root, *_ = source_fixture(tmp_path, monkeypatch)
    control = candidate()
    helper.prepare_probe("cora", .013, tmp_path, control, device="cpu")
    folder = root / _fingerprint(control)
    save_json(control, folder / "candidate.json")
    seen = []
    def generated(*args, **kwargs):
        assert (root / "selected.json").exists()
        seen.append(kwargs)
        raise InterruptedError("toy preflight completed before any student")
    monkeypatch.setattr(citation, "run_screen", generated)
    with pytest.raises(InterruptedError, match="toy preflight"):
        citation.selected_test(root, dict(control, step=1, candidate_path=str(folder)),
                               condensation_seeds=(0,), student_seeds=(0,), device="cpu")
    assert len(seen) == 1
    assert seen[0]["steps"] == 1


def test_numerical_source_token_is_canonical_and_stale_rejected(monkeypatch):
    original = candidate()
    assert citation._candidate_nystrom_mass(original) == original
    monkeypatch.setattr(helper, "source_digest", lambda: {"source": "changed"})
    with pytest.raises(ValueError, match="source changed"):
        citation._candidate_nystrom_mass(original)
    raw = {key: value for key, value in original.items() if key != "mlp_initial_mass_source_digest"}
    assert _fingerprint(citation._candidate_nystrom_mass(raw)) != _fingerprint(original)


@pytest.mark.parametrize("mutation", ["bool_counter", "int_counter", "double_counter", "vector_counter", "negative_counter", "nan_counter", "parameters_none", "parameters_mapping", "parameters_short", "snapshot_none", "snapshot_list", "snapshots_list", "history_none", "adam_groups_none", "adam_group_none", "adam_state_none", "bool_identifier", "bool_wd"])
def test_v2_resealed_malformed_native_resume_rejected_cleanly_before_mutation(tmp_path, monkeypatch, mutation):
    values = problem()
    folder = tmp_path / "valid"
    core.optimize_ce_assignment(*values, steps=1, folder=folder, **options())
    state = copy.deepcopy(load(folder))
    if mutation.endswith("counter"):
        value = {"bool_counter": torch.tensor(True), "int_counter": torch.tensor(1),
                 "double_counter": torch.tensor(1., dtype=torch.float64), "vector_counter": torch.tensor([1.]),
                 "negative_counter": torch.tensor(-1.), "nan_counter": torch.tensor(float("nan"))}[mutation]
        state["optimizer"]["state"][0]["step"] = value
    elif mutation.startswith("parameters_"):
        state["parameters"] = {"parameters_none": None, "parameters_mapping": {}, "parameters_short": []}[mutation]
    elif mutation.startswith("snapshot_"):
        state["snapshots"][1] = None if mutation == "snapshot_none" else []
    elif mutation == "snapshots_list":
        state["snapshots"] = []
    elif mutation == "history_none":
        state["history"] = None
    elif mutation == "adam_groups_none":
        state["optimizer"]["param_groups"] = None
    elif mutation == "adam_group_none":
        state["optimizer"]["param_groups"] = [None]
    elif mutation == "adam_state_none":
        state["optimizer"]["state"] = None
    elif mutation == "bool_identifier":
        state["optimizer"]["param_groups"][0]["params"][0] = False
    else:
        state["optimizer"]["param_groups"][0]["weight_decay"] = False
    reseal(state)
    monkeypatch.setattr(core, "solve_inner_newton_first", lambda *a, **kw: pytest.fail("inner fit ran"))
    absent = tmp_path / "absent"
    with pytest.raises(ValueError):
        core.optimize_ce_assignment(*values, steps=2, folder=absent, resume_state=state, **options())
    assert not absent.exists()


def test_v2_non_native_default_dtype_rejected_before_output(tmp_path):
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        with pytest.raises(ValueError, match="Initial mass"):
            core.optimize_ce_assignment(*problem(), steps=0, folder=tmp_path / "absent", **options())
    finally:
        torch.set_default_dtype(previous)
    assert not (tmp_path / "absent").exists()
