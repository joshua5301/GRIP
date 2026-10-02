import copy
import hashlib
import inspect
import json

import numpy as np
import pytest
import torch

from src import citation_search
from src import source_linear_assignment as linear
from src.io import _fingerprint, save_json, save_state
from src.low_rank_assignment import initialize_factors, logit_block
from src.mlp_initial_mass import seal
from src.moments import make_material
from src.nystrom_ce import cache_features
from src.shared_features import get_shared_map
from src.soft_ce_partition import optimize_ce_assignment
from src.transforms import fit_transform


def candidate(mode="raw_rms"):
    return linear.candidate_controls(dict(method="source_linear", width=0, lr=.05, T=.3, rank=2,
        penalty=.2, initialization="teacher_balanced", alpha=.3, inner_loss_weighting="uniform",
        assignment_coordinates=mode, source_linear_schema=1))


@pytest.fixture
def source_fixture(tmp_path, monkeypatch):
    gen = torch.Generator().manual_seed(56)
    h = torch.randn(13, 3, generator=gen)
    logits = torch.randn(13, 2, generator=gen, dtype=torch.float64)
    q = (logits / .3).softmax(1)
    z, transform = fit_transform(h.double())
    assignment = torch.arange(13) % 4
    graph = dict(x=h, y=torch.arange(13) % 2, adj=torch.eye(13).to_sparse_csr())
    mask = torch.arange(13) < 7
    validation, testing = (graph, ~mask), (graph, ~mask)
    monkeypatch.setitem(citation_search.BUDGET, ("cora", .026), 4)
    monkeypatch.setattr(citation_search, "_prepare_dataset", lambda *a, **k: (graph, mask, validation, testing, h))
    config = citation_search._legacy_teacher_config("cora", .026, graph, mask, validation, testing, "row")
    root = tmp_path / "cora" / "ratio_0.026" / _fingerprint(config)
    root.mkdir(parents=True)
    save_json(config, root / "config.json")
    frozen_h = citation_search.fixed_propagated_features(h, config, root)
    save_state(dict(logits=logits, gamma=.01), root / "teacher.pt")
    save_state(dict(z=z, transform=vars(transform), assignment=assignment), root / "inputs_0.pt")
    init = dict(mode="teacher_balanced", alpha=.3, T=.3, seed=0)
    save_state(assignment, root / f"assignment_{_fingerprint(init)}.pt")
    feature_map = get_shared_map(frozen_h, root / "nystrom_map_schema3.pt")
    cache_features(frozen_h, feature_map, root / "nystrom_phi_schema3.npy")
    return tmp_path, root, h, q, z, assignment, config


def loaded(fixture, mode):
    _, root, h, q, _, _, config = fixture
    c = candidate(mode)
    _, z, _, hard, inputs, source = linear.cached_source(root, c, 0, h, q, config)
    return c, z, q, hard, inputs, source


def run_core(fixture, mode, steps, folder, resume=None):
    c, z, q, hard, inputs, source = loaded(fixture, mode)
    result = optimize_ce_assignment(z, q, hard, steps=steps, folder=folder,
        checkpoint_steps=(0, 1, steps), resume_state=resume,
        **linear.citation_core_kwargs(c, 0, inputs, source))
    state = torch.load(folder / "resume.pt", weights_only=False)
    cfg = linear.expected_citation_config(c, 0, z, q, hard, inputs, source)
    context, params = linear.original_context(z, q, hard, inputs, c["rank"], 0, .05, 4096, cfg, source)
    linear.validate_resume(state, z, q, hard, inputs, context, params, steps, folder)
    return result, state, (z, q, hard, inputs, context, params)


def reseal(state):
    state["source_linear_content_sha256"] = seal({k: v for k, v in state.items() if k != "source_linear_content_sha256"})


@pytest.mark.parametrize("mode", linear.MODES)
def test_native_initialization_and_actual_encoder_VJP(source_fixture, mode):
    c, z, q, hard, f, source = loaded(source_fixture, mode)
    enc, v = linear.initialize_linear(f, 4, 2, 0)
    old_u, old_v = initialize_factors(hard, 4, 2, 0)
    assert torch.equal(v, old_v)
    u = linear.linear_nodes(f, enc)
    assert torch.equal(u, old_u)
    assert torch.equal(logit_block(u, v, hard, .05), logit_block(old_u, old_v, hard, .05))
    assert make_material(z, q).shape[1] == 6  # Phi is not the representative material.
    enc[0].data.add_(.13)
    enc[1].data.add_(.07)
    variable = f.clone().requires_grad_()
    cotangent = torch.randn_like(u)
    current = linear.linear_nodes(variable, enc)
    a_grad, b_grad, f_grad = torch.autograd.grad((current * cotangent).sum(), (*enc, variable), allow_unused=True)
    torch.testing.assert_close(a_grad, f.T @ cotangent)
    torch.testing.assert_close(b_grad, cotangent.sum(0))
    assert f_grad is None
    assert bool(enc[1].ne(0).all())  # Trainable bias is legal.
    assert source["coordinate_mode"] == mode
    assert c["source_linear_source_digest"] == seal(linear.source_digest())


@pytest.mark.parametrize("mode", linear.MODES)
def test_actual_core_1plus24_equals25_and_current_checkpoints(source_fixture, mode, tmp_path):
    torch.manual_seed(938)
    _, first, _ = run_core(source_fixture, mode, 1, tmp_path / "split")
    torch.randn(27)  # unrelated global RNG never changes source/factor initialization.
    _, extended, _ = run_core(source_fixture, mode, 25, tmp_path / "split", first)
    _, full, _ = run_core(source_fixture, mode, 25, tmp_path / "full")
    for x, y in zip(extended["parameters"], full["parameters"], strict=True):
        assert torch.equal(x, y)
    for index in (0, 1, 25):
        for key in ("moments", "theta"):
            assert torch.equal(extended["snapshots"][index][key], full["snapshots"][index][key])
        assert extended["snapshots"][index]["encoded_u_digest"] == full["snapshots"][index]["encoded_u_digest"]
    assert extended["config"] == full["config"]
    assert bool(extended["parameters"][1].ne(0).any())
    assert set(extended["optimizer"]["state"]) == {0, 1, 2}
    for state in extended["optimizer"]["state"].values():
        assert state["step"].dtype == torch.float32 and state["step"].device.type == "cpu"
        assert float(state["step"]) == 25
    assert "mass_target" not in extended and extended["dual"] is None


@pytest.mark.parametrize("mode", linear.MODES)
def test_probe_to_screen_cached_endpoint_bypass(source_fixture, mode, monkeypatch):
    tmp, root, *_ = source_fixture
    c = candidate(mode)
    pin = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir() if p.is_file()}
    result = linear.prepare_probe("cora", .026, tmp, c, device="cpu", citation_features="row")
    assert result["step"] == 1 and result["student_fits"] == 0 and result["cached"] is False
    again = linear.prepare_probe("cora", .026, tmp, c, device="cpu", citation_features="row")
    assert again["cached"] is True
    # The real condensation core extends the exact probe config/trajectory.
    monkeypatch.setattr(citation_search, "fit_gcn_diagnostic", lambda *a, **k: dict(val_acc=.4))
    citation_search.run_screen("cora", .026, tmp, [c], steps=25, checkpoints=(0, 25),
        student_seeds=(11,), device="cpu", citation_features="row", epochs=1)
    folder = root / _fingerprint(c) / "condensation_0"
    state = torch.load(folder / "resume.pt", weights_only=False)
    assert state["step"] == 25 and state["config"]["save_assignment"] is False
    monkeypatch.setattr(citation_search, "optimize_ce_assignment", lambda *a, **k: pytest.fail("cached endpoint optimized"))
    citation_search.run_screen("cora", .026, tmp, [c], steps=25, checkpoints=(0, 25),
        student_seeds=(12,), device="cpu", citation_features="row", epochs=1)
    for p, digest in pin.items():
        assert hashlib.sha256(p.read_bytes()).hexdigest() == digest


@pytest.mark.parametrize("field,value", [
    ("assignment_coordinates", "typo"), ("assignment_coordinate", "raw_rms"),
    ("source_linear_schema", True), ("source_linear_schema", 2), ("source_linear_source_digest", "forged"),
    ("external_coordinates", []), ("source_linear_inputs", []), ("source_linear_source", {}),
    ("outer_targets", []), ("outer_indices", []), ("mass_mode", "uniform"), ("mass_target", []),
    ("mlp_output_centering", "source_mean_v1"), ("surrogate_kernel", "relu_ntk1_diagmatch_v1"),
    ("assignment_encoder", "linear"), ("train_target_mix", 0), ("learn_temperature", False),
    ("width", 256), ("rank", True), ("lr", float("nan")), ("alpha", -1), ("penalty", False),
])
def test_preload_invalid_controls_before_dataset(field, value, tmp_path, monkeypatch):
    bad = dict(candidate(), **{field: value})
    monkeypatch.setattr(citation_search, "_prepare_dataset", lambda *a, **k: pytest.fail("dataset loaded"))
    with pytest.raises(ValueError):
        citation_search.run_screen("cora", .026, tmp_path, [bad], steps=25, device="cpu")
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("missing", ["propagated_H.pt", "nystrom_map_schema3.pt", "nystrom_phi_schema3.npy", "nystrom_phi_schema3.meta.json"])
def test_source_missing_assets_no_creation(source_fixture, missing):
    _, root, *_ = source_fixture
    (root / missing).unlink()
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir() if p.is_file()}
    with pytest.raises(ValueError):
        loaded(source_fixture, "raw_rms")
    assert before == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir() if p.is_file()}


@pytest.mark.parametrize("corruption", ["phi", "phi_meta", "map", "z", "Q", "source_config", "node_order"])
def test_current_source_content_tamper(source_fixture, corruption):
    _, root, h, q, _, _, config = source_fixture
    if corruption == "phi":
        arr = np.load(root / "nystrom_phi_schema3.npy", mmap_mode="r+")
        arr[0, 0] += .1; arr.flush(); del arr
    elif corruption == "phi_meta":
        meta = json.loads((root / "nystrom_phi_schema3.meta.json").read_text())
        meta["h_digest"] = "bad"; save_json(meta, root / "nystrom_phi_schema3.meta.json")
    elif corruption == "map":
        obj = torch.load(root / "nystrom_map_schema3.pt", weights_only=False)
        obj["anchors"][0, 0] += 1; save_state(obj, root / "nystrom_map_schema3.pt")
    elif corruption == "z":
        obj = torch.load(root / "inputs_0.pt", weights_only=False)
        obj["z"][0, 0] += .1; save_state(obj, root / "inputs_0.pt")
    elif corruption == "Q":
        q = q.flip(0)
    elif corruption == "source_config":
        config = dict(config, data_digest="bad")
    else:
        h = h.flip(0)
        config = dict(config, data_digest="permuted_source")
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir() if p.is_file()}
    with pytest.raises(ValueError):
        linear.cached_source(root, candidate("nystrom_relu"), 0, h, q, config)
    assert before == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir() if p.is_file()}


@pytest.mark.parametrize("mutation", ["coupled_J0_scale", "coupled_snapshot_CE", "snapshot_grad", "params", "coordinate_digest", "context", "counter_bool", "counter_dtype", "counter_device", "AdamIDs", "AdamLR", "AdamNaN", "dual", "history", "snapshot_cells", "container", "U_digest", "tracking_container", "tracking_shape", "tracking_nan"])
def test_resealed_cache_rejects_actual_source_and_state(source_fixture, tmp_path, mutation):
    _, state, data = run_core(source_fixture, "raw_rms", 1, tmp_path / "valid")
    bad = copy.deepcopy(state)
    if mutation.startswith("coupled"):
        bad["history"][0]["J"] += .1; bad["scale"] = bad["history"][0]["J"]
        if mutation == "coupled_snapshot_CE":
            bad["snapshots"][0]["teacher_ce"] += .1; reseal(bad["snapshots"][0])
    elif mutation == "snapshot_grad":
        bad["snapshots"][0]["inner_grad_max"] = .5e-7; reseal(bad["snapshots"][0])
    elif mutation == "params": bad["parameters"][1][0] += .01
    elif mutation == "coordinate_digest": bad["source_linear_context"]["coordinates"] = "fake"
    elif mutation == "context": bad["source_linear_context"]["device"] = "cuda:0"
    elif mutation == "counter_bool": bad["optimizer"]["state"][0]["step"] = torch.tensor(True)
    elif mutation == "counter_dtype": bad["optimizer"]["state"][0]["step"] = torch.tensor(1, dtype=torch.int64)
    elif mutation == "counter_device": bad["optimizer"]["state"][0]["step"] = torch.tensor(1., device="meta")
    elif mutation == "AdamIDs": bad["optimizer"]["param_groups"][0]["params"] = [False, 1, 2]
    elif mutation == "AdamLR": bad["optimizer"]["param_groups"][0]["lr"] = .1
    elif mutation == "AdamNaN": bad["optimizer"]["state"][1]["exp_avg"][0] = float("nan")
    elif mutation == "dual": bad["dual"] = torch.zeros(4)
    elif mutation == "history": bad["history"] = []
    elif mutation == "snapshot_cells": bad["snapshots"][1]["moments"] = bad["snapshots"][1]["moments"][:1]; reseal(bad["snapshots"][1])
    elif mutation == "container": bad["optimizer"] = []
    elif mutation == "tracking_container": bad["tracking_vector_before"] = []
    elif mutation == "tracking_shape": bad["tracking_theta_before"] = torch.zeros(1, dtype=torch.float64)
    elif mutation == "tracking_nan": bad["tracking_vector_before"] = torch.full_like(bad["theta"], float("nan"))
    else: bad["snapshots"][1]["encoded_u_digest"] = "bad"; reseal(bad["snapshots"][1])
    if mutation != "counter_device": reseal(bad)
    z, q, hard, inputs, context, params = data
    with pytest.raises((ValueError, RuntimeError)):
        linear.validate_resume(bad, z, q, hard, inputs, context, params, 1)
    # Rejection precedes any core folder or parameter mutation.
    before = [p.clone() for p in params]
    c, _, _, _, _, source = loaded(source_fixture, "raw_rms")
    with pytest.raises((ValueError, RuntimeError)):
        optimize_ce_assignment(z, q, hard, steps=1, folder=tmp_path / "rejected", resume_state=bad,
            **linear.citation_core_kwargs(c, 0, inputs, source))
    assert not (tmp_path / "rejected").exists()
    assert all(torch.equal(x, y) for x, y in zip(before, params, strict=True))


def test_original_source_defaults_signature_unchanged():
    signature = inspect.signature(optimize_ce_assignment)
    assert str(signature.parameters["mlp_initial_options"].kind) == "VAR_KEYWORD"
    assert "source_linear_inputs" not in signature.parameters


@pytest.mark.parametrize("entry", ["screen", "probe"])
def test_native_autocast_rejects_before_dataset(entry, tmp_path, monkeypatch):
    monkeypatch.setattr(citation_search, "_prepare_dataset", lambda *a, **k: pytest.fail("dataset touched under AMP"))
    with torch.autocast("cpu", dtype=torch.bfloat16), pytest.raises(ValueError, match="autocast"):
        if entry == "screen":
            citation_search.run_screen("cora", .026, tmp_path, [candidate()], steps=25, device="cpu")
        else:
            linear.prepare_probe("cora", .026, tmp_path, candidate(), device="cpu")
    assert not list(tmp_path.iterdir())


def test_native_autocast_rejects_core_before_folder(source_fixture, tmp_path):
    c, z, q, hard, inputs, source = loaded(source_fixture, "raw_rms")
    with torch.autocast("cpu", dtype=torch.bfloat16), pytest.raises(ValueError, match="autocast"):
        optimize_ce_assignment(z, q, hard, steps=1, folder=tmp_path / "rejected",
            **linear.citation_core_kwargs(c, 0, inputs, source))
    assert not (tmp_path / "rejected").exists()


@pytest.mark.parametrize("entry", ["screen", "probe", "core"])
def test_native_tf32_policy_rejects_before_scientific_mutation(source_fixture, tmp_path, monkeypatch, entry):
    c, z, q, hard, inputs, source = loaded(source_fixture, "raw_rms")
    monkeypatch.setattr(citation_search, "_prepare_dataset", lambda *a, **k: pytest.fail("dataset touched with TF32"))
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", True)
    with pytest.raises(ValueError, match="TF32"):
        if entry == "screen":
            citation_search.run_screen("cora", .026, tmp_path / "rejected", [c], steps=25, device="cpu")
        elif entry == "probe":
            linear.prepare_probe("cora", .026, tmp_path / "rejected", c, device="cpu")
        else:
            optimize_ce_assignment(z, q, hard, steps=1, folder=tmp_path / "rejected",
                **linear.citation_core_kwargs(c, 0, inputs, source))
    assert not (tmp_path / "rejected").exists()


@pytest.mark.parametrize("mode", linear.MODES)
@pytest.mark.parametrize("override", ["F_and_digest", "z_and_digest", "q_and_digest", "hard_and_digest"])
def test_fresh_core_rejects_coupled_source_overrides(source_fixture, tmp_path, mode, override):
    c, z, q, hard, inputs, source = loaded(source_fixture, mode)
    source = copy.deepcopy(source)
    if override == "F_and_digest":
        inputs = inputs.clone(); inputs[0, 0] += .1
        source["coordinates_fp32"] = linear.digest(inputs)
    elif override == "z_and_digest":
        z = z.clone(); z[0, 0] += .1
        source["z"] = linear.digest(z)
        if mode == "raw_rms":
            inputs = z.float(); source["coordinates_fp32"] = linear.digest(inputs)
    elif override == "q_and_digest":
        q = q.flip(0); source["q"] = linear.digest(q)
    else:
        hard = hard.roll(1); source["hard_assignment"] = linear.digest(hard)
    assets = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_fixture[1].iterdir() if p.is_file()}
    with pytest.raises(ValueError):
        optimize_ce_assignment(z, q, hard, steps=1, folder=tmp_path / "rejected",
            **linear.citation_core_kwargs(c, 0, inputs, source))
    assert not (tmp_path / "rejected").exists()
    assert assets == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_fixture[1].iterdir() if p.is_file()}


@pytest.mark.parametrize("mode", linear.MODES)
def test_actual_interrupted_resume_matches_uninterrupted(source_fixture, tmp_path, mode):
    c, z, q, hard, inputs, source = loaded(source_fixture, mode)
    folder = tmp_path / "interrupted"
    def stop_after_first_saved_update():
        resume = folder / "resume.pt"
        return resume.exists() and torch.load(resume, weights_only=False)["step"] == 1
    with pytest.raises(InterruptedError):
        optimize_ce_assignment(z, q, hard, steps=3, folder=folder, checkpoint_steps=(0, 1, 3),
            stop=stop_after_first_saved_update, **linear.citation_core_kwargs(c, 0, inputs, source))
    saved = torch.load(folder / "resume.pt", weights_only=False)
    assert saved["step"] == 1
    torch.manual_seed(473); torch.randn(31)
    _, resumed, _ = run_core(source_fixture, mode, 3, folder, saved)
    _, full, _ = run_core(source_fixture, mode, 3, tmp_path / "uninterrupted")
    assert all(torch.equal(x, y) for x, y in zip(resumed["parameters"], full["parameters"], strict=True))
    assert torch.equal(resumed["snapshots"][3]["moments"], full["snapshots"][3]["moments"])
    assert torch.equal(resumed["snapshots"][3]["theta"], full["snapshots"][3]["theta"])


@pytest.mark.parametrize("entry", ["probe", "screen", "selected"])
@pytest.mark.parametrize("corruption", ["orphan", "coupled_J0_scale"])
def test_cached_public_bypass_rejects_before_new_writes(source_fixture, monkeypatch, entry, corruption):
    tmp, root, *_ = source_fixture
    c = candidate("nystrom_relu")
    linear.prepare_probe("cora", .026, tmp, c, device="cpu", citation_features="row")
    folder = root / _fingerprint(c) / "condensation_0"
    if corruption == "orphan":
        (folder / "resume.pt").unlink()
    else:
        state = torch.load(folder / "resume.pt", weights_only=False)
        state["history"][0]["J"] += .1; state["scale"] = state["history"][0]["J"]
        reseal(state); save_state(state, folder / "resume.pt")
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp.rglob("*") if p.is_file()}
    monkeypatch.setattr(citation_search, "fit_gcn_diagnostic", lambda *a, **k: pytest.fail("student fit on corrupt cache"))
    monkeypatch.setattr(citation_search, "optimize_ce_assignment", lambda *a, **k: pytest.fail("P update on corrupt cache"))
    with pytest.raises(ValueError):
        if entry == "probe":
            linear.prepare_probe("cora", .026, tmp, c, device="cpu", citation_features="row")
        elif entry == "screen":
            citation_search.run_screen("cora", .026, tmp, [c], steps=1, checkpoints=(0, 1),
                student_seeds=(12,), device="cpu", citation_features="row", epochs=1)
        else:
            citation_search.selected_test(root, dict(c, step=1, candidate_path=str(folder.parent)),
                condensation_seeds=(0,), student_seeds=(12,), device="cpu", epochs=1)
    assert before == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp.rglob("*") if p.is_file()}
