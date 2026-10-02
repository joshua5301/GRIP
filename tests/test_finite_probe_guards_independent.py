"""Pure-definition guard QA; no adapter import or real data/cache/native paths.

Run only under explicit root CPU authorization against the actual src file.  AST extraction omits every top-level import and source-loader
function, injects deterministic tiny certificates, and uses temporary files.
It tests control/state authority, not real numerical/native CUDA correctness.
"""

import ast
import copy
import csv
import hashlib
import io
import json
import math
import os
import uuid
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

PURE = {"_require", "_stop", "_digest", "_seal", "canonical_candidate", "_scalar", "_tensor",
        "_first_adam", "_optimizer", "_attach", "_validate_bundle", "_history", "_atomic",
        "_mirrors", "prepare_probe", "_public_options"}
CONSTANTS = {"SCHEMA", "_FIXED", "_SOURCE_FIELD", "_ROUTES"}


def _cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_cpu(item) for item in value]
    return value


def _array_digest(array):
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str((array.shape, array.dtype.str)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _inventory(folder):
    return {str(path.relative_to(folder)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(folder.rglob("*")) if path.is_file()}


@pytest.fixture
def harness(tmp_path):
    path = (Path(__file__).resolve().parents[1] / "src/finite_student_probe.py").resolve(strict=True)
    tree = ast.parse(path.read_text())
    selected = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in PURE:
            selected.append(node)
        elif isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id in CONSTANTS
                                                for target in node.targets):
            selected.append(node)
    assert {node.name for node in selected if isinstance(node, ast.FunctionDef)} == PURE
    code = compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec")
    space = dict(torch=torch, np=np, math=math, json=json, hashlib=hashlib, csv=csv, io=io, os=os,
                 uuid=uuid, Path=Path, array_digest=_array_digest, cpu_state=_cpu,
                 STEPS=5, LEARNING_RATE=.1, WEIGHT_DECAY=.001, INITIAL_SEED=0, __file__=str(path))
    exec(code, space)
    owner = SimpleNamespace(space=space, root=tmp_path / "fake_source",
                            events=[], cancel=False, replay_hook=None, commit_hook=None)
    owner.source = dict(files={"fake_source.py": "source-A"}, versions={"torch": "fixture"})
    space["numerical_source"] = lambda: copy.deepcopy(owner.source)
    space["_fingerprint"] = lambda obj: hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:12]
    owner.initial = [torch.zeros(3, 2), torch.tensor([[.05, -.04], [.03, .02]])]
    owner.gradients = [torch.tensor([[.1, -.2], [.2, -.1], [.3, -.4]]),
                       torch.tensor([[.03, -.04], [.02, -.01]])]
    owner.buffers = dict(root=owner.root, initial=owner.initial, pins={"mock": "mock"})
    space["_native"] = lambda device: owner.events.append("native") or {"device": "fake CPU"}
    space["_load_source"] = lambda *args: owner.events.append("load_source") or owner.buffers
    space["_p0_reference"] = lambda buffers: owner.events.append("reference") or {"P0_proven": True}
    space["_checked_files"] = lambda pins: owner.events.append("pins")
    space["_source_unchanged"] = lambda context: space["_require"](owner.source == context["source"], "fixture source changed")
    space["_context"] = lambda candidate, *args: {"candidate": candidate, "source": copy.deepcopy(owner.source),
                                                "actual_Q_digest": "frozen-Q", "native": "fake-context"}

    def evaluate(buffers, candidate, context, parameters, step, scale, stop):
        space["_stop"](stop)
        owner.events.append(f"replay{step}")
        if owner.replay_hook:
            owner.replay_hook(step)
        space["_stop"](stop)
        scale = .7 if scale is None else scale
        snapshot = dict(schema=space["SCHEMA"], context=context, step=step,
                        parameters=_cpu(parameters), teacher_ce=.7 if step == 0 else .67,
                        scale=scale, scaled_factor_gradients=_cpu(owner.gradients),
                        moments=torch.tensor([[.4, .08, .12], [.6, .18, .12]], dtype=torch.float64),
                        min_mass=.4, max_mass=.6, effective_cells=1/(.4**2+.6**2),
                        finite_model_steps=5, selected_route=candidate["outer_route"],
                        moment_partial=torch.ones(2, 3, dtype=torch.float64),
                        inner_trace=[{"step": index, "uniform_soft_ce": .6} for index in range(5)],
                        adapted_model=[torch.tensor([1.])])
        return snapshot, scale

    space["_evaluate"] = evaluate
    native_atomic = space["_atomic"]

    def atomic(path, value, binary):
        owner.events.append("write:" + path.name)
        native_atomic(path, value, binary)
        if owner.commit_hook:
            owner.commit_hook(path, value)

    space["_atomic"] = atomic

    class FakeAdam:
        def __init__(self, parameters, **kwargs):
            owner.events.append("optimizer_construct")
            self.parameters = parameters
            self.saved = dict(state={}, param_groups=[dict(lr=.05, betas=(.9, .999), eps=1e-12,
                              weight_decay=0, amsgrad=False, maximize=False, foreach=False,
                              capturable=False, differentiable=False, fused=False, params=[0, 1])])
        def state_dict(self):
            return _cpu(self.saved)
        def load_state_dict(self, state):
            owner.events.append("optimizer_load")
            self.saved = _cpu(state)
        def step(self):
            owner.events.append("optimizer_update")
            for index, parameter in enumerate(self.parameters):
                updated, first, second = space["_first_adam"](parameter, parameter.grad)
                with torch.no_grad():
                    parameter.copy_(updated)
                self.saved["state"][index] = dict(step=torch.tensor(1., dtype=torch.float32),
                                                  exp_avg=first.detach(), exp_avg_sq=second.detach())

    class TorchProxy:
        optim = SimpleNamespace(Adam=FakeAdam)
        cuda = SimpleNamespace(
            reset_peak_memory_stats=lambda: owner.events.append("mock_memory_reset"),
            get_device_properties=lambda index: SimpleNamespace(total_memory=4096),
            current_device=lambda: 0,
            synchronize=lambda: owner.events.append("mock_memory_sync"),
            max_memory_allocated=lambda: 128,
            max_memory_reserved=lambda: 256,
        )
        def __getattr__(self, name):
            return getattr(torch, name)

    space["torch"] = TorchProxy()
    owner.candidate = space["canonical_candidate"](dict(space["_FIXED"], outer_route="sgc_mlp"))
    owner.context = space["_context"](owner.candidate, None, None, None)
    owner.folder = owner.root / space["_fingerprint"](owner.candidate) / "condensation_0"
    owner.fake_adam = FakeAdam

    def make_bundle(step=1):
        snap0, scale = evaluate(owner.buffers, owner.candidate, owner.context, owner.initial, 0, None, lambda: False)
        optimizer = FakeAdam([value.detach().clone().requires_grad_(True) for value in owner.initial])
        snapshots = {0: snap0}
        if step:
            for parameter, gradient in zip(optimizer.parameters, owner.gradients):
                parameter.grad = gradient.clone()
            optimizer.step()
            snap1, _ = evaluate(owner.buffers, owner.candidate, owner.context, optimizer.parameters, 1, scale, lambda: False)
            snapshots[1] = snap1
        bundle = space["_attach"](dict(schema=space["SCHEMA"], context=owner.context, step=step,
                                      parameters=optimizer.parameters, optimizer=optimizer.state_dict(),
                                      snapshots=snapshots, scale=scale,
                                      history=[space["_history"](snapshots[index]) for index in range(step + 1)]))
        owner.events.clear()
        return bundle

    def persist(bundle):
        owner.folder.mkdir(parents=True)
        (owner.folder.parent / "candidate.json").write_text(json.dumps(owner.candidate))
        torch.save(bundle, owner.folder / "resume.pt")
        space["_mirrors"](owner.folder, bundle, export=True)
        owner.events.clear()

    owner.make_bundle, owner.persist = make_bundle, persist
    owner.public = dict(dataset="cora", ratio=.026,
                        output_dir=str(path.resolve().parents[1] / "results/citation_search_v1"),
                        condensation_seed=0, data_dir=str(path.resolve().parents[1] / "data"),
                        device="cuda", citation_features="row")
    owner.call = lambda **kwargs: space["prepare_probe"](**owner.public, candidate=owner.candidate,
                                 stop=lambda: owner.cancel, **kwargs)
    return owner


@pytest.mark.parametrize("field", ("train_target_mix", "assignment_coordinates", "temperature_lr", "steps",
                                   "initial_parameters", "source_q", "hidden", "deadline_seconds"))
def test_unknown_or_legacy_controls_reject_before_source_or_optimizer(harness, field):
    candidate = dict(harness.candidate, **{field: 1})
    with pytest.raises(ValueError):
        harness.space["prepare_probe"]("fixture", .1, "unused", candidate, device="fake")
    assert harness.events == [] and not harness.root.exists()


@pytest.mark.parametrize("key,value", (("rank", True), ("lr", float("nan")), ("T", ".3"),
                                      ("finite_model_steps", 6), ("finite_model_lr", .01),
                                      ("outer_route", "linear"), ("finite_student_source_digest", "old")))
def test_fixed_policy_and_source_token_are_strict(harness, key, value):
    candidate = dict(harness.candidate, **{key: value})
    with pytest.raises(ValueError):
        harness.space["canonical_candidate"](candidate)
    assert harness.events == []


def test_route_names_have_distinct_canonical_identity_same_source(harness):
    left = harness.candidate
    right = harness.space["canonical_candidate"](dict(left, outer_route="gcn"))
    assert left["finite_student_source_digest"] == right["finite_student_source_digest"]
    assert harness.space["_fingerprint"](left) != harness.space["_fingerprint"](right)
    assert {key for key in left if left[key] != right[key]} == {"outer_route"}


def _reseal(space, bundle):
    bundle.pop("content_sha256", None)
    return space["_attach"](bundle)


@pytest.mark.parametrize("tamper", ("seal", "context", "current_parameters", "initial_parameters",
                                    "snapshot0_none", "snapshots_list", "history", "coupled_J0_scale",
                                    "finite_trace", "moment_partial", "orphan_snapshot", "step_bool",
                                    "adam_counter_bool", "adam_counter_vector", "adam_first_moment", "adam_lr"))
def test_resealed_malformed_or_selfconsistent_cache_is_not_authority(harness, tamper):
    bundle = harness.make_bundle()
    if tamper == "seal":
        bundle["content_sha256"] = "bad"
    elif tamper == "context":
        bundle["context"]["actual_Q_digest"] = "forged"
    elif tamper == "current_parameters":
        bundle["parameters"][0][0, 0] += .1
    elif tamper == "initial_parameters":
        bundle["snapshots"][0]["parameters"][1][0, 0] += .1
    elif tamper == "snapshot0_none":
        bundle["snapshots"][0] = None
    elif tamper == "snapshots_list":
        bundle["snapshots"] = [bundle["snapshots"][0], bundle["snapshots"][1]]
    elif tamper == "history":
        bundle["history"][1]["teacher_ce"] += .1
    elif tamper == "coupled_J0_scale":
        bundle["scale"] *= 2
        for snapshot in bundle["snapshots"].values():
            snapshot["scale"] *= 2
            snapshot["teacher_ce"] *= 2
        for row in bundle["history"]:
            row["teacher_ce"] *= 2
            row["J0_scale"] *= 2
    elif tamper == "finite_trace":
        bundle["snapshots"][1]["inner_trace"][2]["uniform_soft_ce"] += .1
    elif tamper == "moment_partial":
        bundle["snapshots"][1]["moment_partial"][0, 0] += .1
    elif tamper == "orphan_snapshot":
        bundle["snapshots"][2] = copy.deepcopy(bundle["snapshots"][1])
    elif tamper == "step_bool":
        bundle["step"] = True
    elif tamper == "adam_counter_bool":
        bundle["optimizer"]["state"][0]["step"] = torch.tensor(True)
    elif tamper == "adam_counter_vector":
        bundle["optimizer"]["state"][0]["step"] = torch.tensor([1.])
    elif tamper == "adam_first_moment":
        bundle["optimizer"]["state"][0]["exp_avg"][0, 0] += .1
    elif tamper == "adam_lr":
        bundle["optimizer"]["param_groups"][0]["lr"] = .01
    if tamper != "seal":
        bundle = _reseal(harness.space, bundle)
    with pytest.raises(ValueError):
        harness.space["_validate_bundle"](bundle, harness.buffers, harness.candidate, harness.context,
                                          lambda: False)
    assert not any(event.startswith("write:") or event.startswith("optimizer_") for event in harness.events)


@pytest.mark.parametrize("tamper", ("wrong_filename", "orphan_step", "payload", "non_mapping", "csv_value", "csv_column"))
def test_present_corrupt_mirror_blocks_all_repairs_preserving_bytes(harness, tamper):
    bundle = harness.make_bundle()
    harness.persist(bundle)
    checkpoint = harness.folder / "checkpoints/step_000001.pt"
    if tamper == "wrong_filename":
        checkpoint.rename(checkpoint.with_name("step_000099.pt"))
    elif tamper == "orphan_step":
        orphan = copy.deepcopy(bundle["snapshots"][1])
        orphan["step"] = 2
        torch.save(orphan, checkpoint.with_name("step_000002.pt"))
    elif tamper == "payload":
        snapshot = copy.deepcopy(bundle["snapshots"][1])
        snapshot["teacher_ce"] += .1
        torch.save(snapshot, checkpoint)
    elif tamper == "non_mapping":
        torch.save(None, checkpoint)
    elif tamper == "csv_value":
        path = harness.folder / "optimization.csv"
        path.write_text(path.read_text().replace("0.67", "0.99"))
    elif tamper == "csv_column":
        path = harness.folder / "optimization.csv"
        path.write_text(path.read_text().replace("teacher_ce", "fake_ce"))
    (harness.folder / "checkpoints/step_000000.pt").unlink()
    before = _inventory(harness.root)
    with pytest.raises(ValueError):
        harness.space["_mirrors"](harness.folder, bundle, export=True)
    assert _inventory(harness.root) == before
    assert not any(event.startswith("write:") for event in harness.events)


def test_authoritative_resume_repairs_missing_mirrors_without_an_update(harness):
    bundle = harness.make_bundle()
    harness.persist(bundle)
    (harness.folder / "checkpoints/step_000001.pt").unlink()
    (harness.folder / "optimization.csv").unlink()
    old_resume = (harness.folder / "resume.pt").read_bytes()
    result = harness.call()
    assert result["cached"] is True and result["P_updates"] == 0 and result["student_fits"] == 0
    assert (harness.folder / "resume.pt").read_bytes() == old_resume
    assert (harness.folder / "checkpoints/step_000001.pt").exists()
    assert (harness.folder / "optimization.csv").exists()
    assert "optimizer_update" not in harness.events


def test_orphan_mirror_without_resume_cannot_restart_or_overwrite(harness):
    bundle = harness.make_bundle()
    harness.persist(bundle)
    (harness.folder / "resume.pt").unlink()
    before = _inventory(harness.root)
    with pytest.raises(ValueError):
        harness.call()
    assert _inventory(harness.root) == before
    assert not any(event.startswith("optimizer_") or event.startswith("write:") for event in harness.events)


def test_cached_stop_and_changed_current_source_reject_before_export(harness):
    bundle = harness.make_bundle()
    harness.persist(bundle)
    before = _inventory(harness.root)
    harness.cancel = True
    with pytest.raises(InterruptedError):
        harness.call()
    assert harness.events == [] and _inventory(harness.root) == before
    harness.cancel = False
    harness.source["files"]["fake_source.py"] = "source-B"
    with pytest.raises(ValueError):
        harness.call()
    assert _inventory(harness.root) == before
    assert not any(event.startswith("optimizer_") or event.startswith("write:") for event in harness.events)


def test_stop_after_atomic_resume0_preserves_valid_frontier_and_resumes_once(harness):
    def cancel_after_initial_commit(path, value):
        if path.name == "resume.pt" and value["step"] == 0:
            harness.cancel = True
    harness.commit_hook = cancel_after_initial_commit
    with pytest.raises(InterruptedError):
        harness.call()
    saved = torch.load(harness.folder / "resume.pt", map_location="cpu", weights_only=False)
    assert saved["step"] == 0
    assert "optimizer_update" not in harness.events
    harness.commit_hook = None
    harness.cancel = False
    harness.events.clear()
    result = harness.call()
    assert result["P_updates"] == 1 and result["cached"] is False
    assert harness.events.count("optimizer_update") == 1


def test_deadline_stop_inside_replay_prevents_first_commit(harness):
    clock = SimpleNamespace(now=0.)
    def replay_time(step):
        clock.now += 1.
        harness.cancel = clock.now >= .5
    harness.replay_hook = replay_time
    with pytest.raises(InterruptedError):
        harness.call()
    assert not harness.root.exists()
    assert "optimizer_update" not in harness.events
    assert not any(event.startswith("write:") for event in harness.events)


@pytest.mark.parametrize("key,value", (("dataset", "citeseer"), ("ratio", .052),
                                      ("condensation_seed", True), ("device", "cpu"),
                                      ("citation_features", "raw"), ("output_dir", "/tmp/external")))
def test_public_fixed_probe_options_reject_before_native_or_loader(harness, key, value):
    public = dict(harness.public, **{key: value})
    with pytest.raises(ValueError):
        harness.space["prepare_probe"](**public, candidate=harness.candidate)
    assert harness.events == [] and not harness.root.exists()
