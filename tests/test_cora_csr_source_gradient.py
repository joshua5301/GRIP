"""StageBI orchestration guards only: AST-extracted defs, no production imports."""
import ast
import copy
import hashlib
import json
import platform
import sys
import time
import traceback
import types
from pathlib import Path

import pytest

BASE = Path(__file__).resolve().parents[1]
MODULE = BASE / "src/cora_csr_source_gradient.py"
WORKER = BASE / "src/research_loop.py"
REPO = next(p for p in Path(__file__).resolve().parents if (p / "src/transforms.py").is_file())
SCIENCE = json.loads((REPO / "results/proposals/Cora70_original_CSR_source_gradient_readonly_scientific_stageBI_v1.json").read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def exact(value, expected):
    if isinstance(expected, dict):
        return isinstance(value, dict) and set(value) == set(expected) and all(exact(value[k], v) for k, v in expected.items())
    return type(value) is type(expected) and value == expected


@pytest.fixture
def api():
    selected = {"_count", "_request", "_bind_source", "_retained", "_previous_receipt", "_repeat", "_sealed_payload", "_preserve", "prepare_targets", "_artifact_paths"}
    tree = ast.parse(MODULE.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in selected]
    ns = dict(Path=Path, json=json, time=time, traceback=traceback, platform=platform,
              _require=require, _exact=exact, _sha=sha, _stop=lambda stop: require(not stop(), "stop"),
              _source_digest=lambda b: copy.deepcopy(b["source"]),
              _seal=lambda value: hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest(), probe=types.SimpleNamespace(_digest=copy.deepcopy))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(MODULE), "exec"), ns)
    return ns


def outputs(tmp_path):
    return dict(pass_outputs={str(i): str(tmp_path / f"pass_{i}" / "native.json") for i in (1, 2)},
                target_outputs={str(i): str(tmp_path / f"pass_{i}" / "targets.pt") for i in (1, 2)})


@pytest.mark.parametrize("index,previous", [(True, None), (0, None), (3, None), (1, "a" * 64), (2, None), (2, "short")])
def test_request_rejects_phase_or_previous_sha(api, tmp_path, index, previous):
    spec = outputs(tmp_path)
    with pytest.raises(ValueError):
        api["_request"](index, previous, spec["pass_outputs"]["1"], spec)


def test_request_exclusive_both_passes(api, tmp_path):
    spec = outputs(tmp_path)
    for i, previous in ((1, None), (2, "a" * 64)):
        output, target = api["_request"](i, previous, spec["pass_outputs"][str(i)], spec)
        assert output.parent == target.parent
    Path(spec["pass_outputs"]["1"]).parent.mkdir()
    with pytest.raises(ValueError, match="exclusive"):
        api["_request"](1, None, spec["pass_outputs"]["1"], spec)


def test_original_source_uses_current_ghost_token_only(api):
    old = SCIENCE["original_source_binding"]["source_context_descriptor"]
    ghost = dict(old["candidate"], source_linear_source_digest="current_token")
    current = dict(old, candidate=ghost)
    native = SCIENCE["original_source_binding"]["native_buffers"]
    api["_bind_source"](current, native, ghost, SCIENCE)
    with pytest.raises(ValueError):
        api["_bind_source"](old, native, ghost, SCIENCE)


@pytest.mark.parametrize("key", ["X", "original_CSR", "Q", "H", "hard", "transform"])
def test_source_descriptor_override_fails(api, key):
    old = SCIENCE["original_source_binding"]["source_context_descriptor"]
    native = copy.deepcopy(SCIENCE["original_source_binding"]["native_buffers"])
    native[key] = {"altered": True}
    with pytest.raises(ValueError):
        api["_bind_source"](old, native, old["candidate"], SCIENCE)


@pytest.mark.parametrize("key", ["source", "anchors", "targets"])
def test_retained_source_model_targets_mutation(api, key):
    buffers = dict(source={"X": "fixed"}, anchors=["fixed"], targets={"fixed": True})
    evidence = dict(native_source_buffers_before=copy.deepcopy(buffers["source"]),
                    anchor_digests_before=copy.deepcopy(buffers["anchors"]), target_numerical_digest=copy.deepcopy(buffers["targets"]))
    api["_retained"](buffers, evidence)
    buffers[key] = {"mutated": True}
    with pytest.raises(ValueError):
        api["_retained"](buffers, evidence)


def first_receipt(tmp_path):
    spec = outputs(tmp_path)
    target = Path(spec["target_outputs"]["1"])
    target.parent.mkdir()
    target.write_bytes(b"opaque target cache; no deserialization")
    evidence = dict(source={"files": "current"}, numerical_source={"versions": "current"}, python_version="fixed",
                    scientific_preregistration={"sha256": "fixed"}, spec_path="fixed", spec_sha256="fixed")
    first = dict(evidence, passed=True, pass_index=1, operation="prepare_targets", source_backend="existing_original_CSR",
                 source_assets_spec_science_unchanged=True, strict_CSR_source_gradient_backend_qualified=False,
                 within_CSR_exact_repeat_passed=False, test_enabled=False, validation_only=True,
                 source_reference_provenance_passed=True, success_counts=SCIENCE["counts_contract"]["per_successful_worker"],
                 target_cache_path=str(target), target_cache_sha256=sha(target))
    path = Path(spec["pass_outputs"]["1"])
    return spec, evidence, first, path


@pytest.mark.parametrize("key,value", [("passed", False), ("strict_CSR_source_gradient_backend_qualified", True),
    ("within_CSR_exact_repeat_passed", True), ("source_reference_provenance_passed", False),
    ("source", {"wrong": True}), ("pass_index", True), ("test_enabled", True), ("success_counts", {})])
def test_resealed_first_receipt_malformed_protocol_fails(api, tmp_path, key, value):
    spec, evidence, first, path = first_receipt(tmp_path)
    first[key] = value
    path.write_text(json.dumps(first))
    with pytest.raises(ValueError):
        api["_previous_receipt"](spec, SCIENCE, evidence, sha(path))


def test_prior_opaque_cache_sha_is_checked_before_native(api, tmp_path):
    spec, evidence, first, path = first_receipt(tmp_path)
    path.write_text(json.dumps(first))
    api["_previous_receipt"](spec, SCIENCE, evidence, sha(path))
    Path(spec["target_outputs"]["1"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="bytes"):
        api["_previous_receipt"](spec, SCIENCE, evidence, sha(path))


def test_partial_operation_attempt_is_not_completion(api):
    evidence = {"_stop": lambda: False, "operation_attempts": {}, "counts": {}}
    def failure():
        raise ValueError("observed source failure")
    with pytest.raises(ValueError):
        api["_count"](evidence, "source_gradient_target_helper_calls", failure)
    assert evidence["operation_attempts"] == {"source_gradient_target_helper_calls": 1}
    assert evidence["counts"] == {}
    assert api["_count"](evidence, "private_anchor_factory_calls", lambda: "return") == "return"
    assert evidence["counts"] == {"private_anchor_factory_calls": 1}


def run_fixture(api, tmp_path, *, initialized=False, source_failure=False, preservation_failure=False, memory_failure=False):
    spec = dict(outputs(tmp_path), source={"files": "current"}, numerical_source={"versions": "current"}, python_version="fixed",
                scientific_preregistration={"sha256": "fixed"}, files_sha256={"fixed": "fixed"})
    science = dict(SCIENCE)
    calls = {"native": 0, "source": 0, "preserve": 0}
    source = {"X": "frozen"}
    def preserve(*args):
        calls["preserve"] += 1
        if preservation_failure and calls["preserve"] > 1:
            raise RuntimeError("preservation changed")
    def native(device):
        calls["native"] += 1
        return dict(GPU="NVIDIA GeForce RTX 4060", cuda_runtime="12.8")
    def load(*args):
        calls["source"] += 1
        if source_failure:
            raise ValueError("PRIMARY source failure")
        e = args[2]
        e.update(source_context={"current_token": True}, native_source_buffers_before=source,
                 source_reference_provenance_passed=True)
        return dict(source=source)
    def collect(buffers, dense, e):
        e["counts"].update(SCIENCE["counts_contract"]["per_successful_worker"])
        e.update(target_numerical_digest={"fixed": True}, target_numerical_seal="fixed")
        return {"fixed": True}
    def reserved():
        if memory_failure:
            raise RuntimeError("memory finalization failed")
        return 128
    cuda = types.SimpleNamespace(is_initialized=lambda: initialized, reset_peak_memory_stats=lambda: None,
        synchronize=lambda: None, current_device=lambda: 0,
        get_device_properties=lambda device: types.SimpleNamespace(total_memory=8316977152),
        max_memory_allocated=lambda: 64, max_memory_reserved=reserved)
    fake_torch = types.SimpleNamespace(cuda=cuda, get_num_threads=lambda: 4,
                                     save=lambda payload, stream: stream.write(b"mock target envelope"))
    probe = types.SimpleNamespace(_native=native, _runtime_precision_guard=lambda: None, _attach=lambda x: x, _digest=copy.deepcopy)
    def write(output, e):
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x") as stream:
            json.dump(e, stream, allow_nan=False)
    def load_spec(*args):
        preserve()
        return spec, science
    api.update(torch=fake_torch, probe=probe, _load_spec=load_spec, _load_source=load,
        _preserve=preserve, _historical_targets=lambda *args: {"fixed": True}, _target_valid=lambda *args: None,
        _collect=collect, _descriptive_comparison=lambda *args: {"thresholds": None, "dense_CSR_equivalent": False},
        _write_new=write, __file__=str(MODULE))
    output = spec["pass_outputs"]["1"]
    return lambda: api["prepare_targets"](1, str(tmp_path / "spec.json"), "a" * 64, output), Path(output), calls


def test_mock_public_success_is_local_only(api, tmp_path):
    run, path, calls = run_fixture(api, tmp_path)
    result = run()
    report = json.loads(path.read_text())
    assert result["validation_only"] is True and result["evidence_sha256"] == sha(path)
    assert report["passed"] is True and report["strict_CSR_source_gradient_backend_qualified"] is False
    assert report["within_CSR_exact_repeat_passed"] is False
    assert report["success_counts"] == SCIENCE["counts_contract"]["per_successful_worker"]
    assert calls["native"] == calls["source"] == 1


def test_fresh_cuda_failure_precedes_any_source_access(api, tmp_path):
    run, path, calls = run_fixture(api, tmp_path, initialized=True)
    with pytest.raises(ValueError, match="fresh CUDA"):
        run()
    report = json.loads(path.read_text())
    assert calls["native"] == calls["source"] == 0
    assert report["passed"] is False and "success_counts" not in report
    assert all(v == 0 for v in report["counts"].values())


def test_primary_error_preserved_over_final_preservation_error(api, tmp_path):
    run, path, _ = run_fixture(api, tmp_path, source_failure=True, preservation_failure=True)
    with pytest.raises(ValueError, match="PRIMARY source failure"):
        run()
    report = json.loads(path.read_text())
    assert report["passed"] is False and report["source_assets_spec_science_unchanged"] is False
    assert "preservation changed" in report["preservation_error"]
    assert "success_counts" not in report


def test_memory_failure_cannot_qualify_saved_local_target(api, tmp_path):
    run, path, _ = run_fixture(api, tmp_path, memory_failure=True)
    with pytest.raises(RuntimeError, match="memory finalization"):
        run()
    report = json.loads(path.read_text())
    assert report["passed"] is False and report["strict_CSR_source_gradient_backend_qualified"] is False
    assert Path(report["target_cache_path"]).is_file() and "success_counts" not in report


def test_draft_dispatch_is_lazy_and_preserves_options_and_stop(monkeypatch):
    tree = ast.parse(WORKER.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "dispatch")
    ns = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(WORKER), "exec"), ns)
    module = types.ModuleType("src.cora_csr_source_gradient")
    package = types.ModuleType("src")
    package.__path__ = []
    received = {}
    def fake(**kwargs):
        received.update(kwargs)
        return {"mock": True}
    module.prepare_targets = fake
    monkeypatch.setitem(sys.modules, "src", package)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    callback = lambda: False
    options = dict(pass_index=2, previous_sha256="a" * 64, output_path="fixed")
    assert ns["dispatch"]({"kind": "citation_cora_csr_source_gradient", "options": options}, callback) == {"mock": True}
    assert received == dict(options, stop=callback) and options == dict(pass_index=2, previous_sha256="a" * 64, output_path="fixed")


def test_thin_source_structure_has_no_dense_or_research_calls():
    tree = ast.parse(MODULE.read_text())
    load = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_load_source")
    calls = [n.func.attr if isinstance(n.func, ast.Attribute) else n.func.id if isinstance(n.func, ast.Name) else "" for n in ast.walk(load) if isinstance(n, ast.Call)]
    assert not set(calls) & {"to_dense", "initialize_factors", "backward", "_reference", "_prepare_graph", "optimize_ce_assignment"}
    assert "cached_source" in ast.unparse(load) and "atol=2e-07" in ast.unparse(load) and "rtol=1e-06" in ast.unparse(load)


def repeat_fixture(api, tmp_path, mutation=None):
    spec, evidence, first, path = first_receipt(tmp_path)
    evidence.update(_stop=lambda: False, operation_attempts={}, counts={}, native_environment={"CSR": True},
                    source_context={"current_token": "own"}, native_source_buffers_before={"X": "fixed"})
    context = dict(source=spec.get("source", evidence["source"]), numerical_source=evidence["numerical_source"],
        scientific_preregistration=evidence["scientific_preregistration"], spec_sha256=evidence["spec_sha256"],
        source_context=evidence["source_context"], native_source_buffers=evidence["native_source_buffers_before"],
        source_backend="existing_original_CSR")
    spec.update(source=evidence["source"], numerical_source=evidence["numerical_source"],
                scientific_preregistration=evidence["scientific_preregistration"])
    target = {"v": "fixed"}
    payload = dict(context=context, pass_index=1, targets=copy.deepcopy(target))
    first.update(native_environment=evidence["native_environment"], target_numerical_digest=copy.deepcopy(target),
                 target_numerical_seal=api["_seal"](target))
    if mutation == "targets":
        payload["targets"] = {"v": "resealed different"}
    elif mutation == "context":
        payload["context"] = dict(context, source_context={"wrong_cond": True})
    elif mutation == "bool_pass":
        payload["pass_index"] = True
    elif mutation == "environment":
        first["native_environment"] = {"CSR": False}
    payload["content_sha256"] = api["_seal"](payload)
    path.write_text(json.dumps(first))
    api["torch"] = types.SimpleNamespace(load=lambda *args, **kwargs: copy.deepcopy(payload))
    return spec, evidence, target, path


def test_exact_second_pass_qualifies_only_after_cache_and_context_match(api, tmp_path):
    spec, evidence, targets, path = repeat_fixture(api, tmp_path)
    api["_repeat"](spec, SCIENCE, evidence, targets, sha(path))
    assert evidence["within_CSR_exact_repeat_passed"] is True
    assert evidence["strict_CSR_source_gradient_backend_qualified"] is True
    assert evidence["counts"] == {"previous_CSR_target_cache_loads": 1}
    assert evidence["previous_receipt_sha256"] == sha(path)


@pytest.mark.parametrize("mutation", ["targets", "context", "bool_pass", "environment"])
def test_resealed_repeat_disagreement_cannot_qualify(api, tmp_path, mutation):
    spec, evidence, targets, path = repeat_fixture(api, tmp_path, mutation)
    with pytest.raises(ValueError):
        api["_repeat"](spec, SCIENCE, evidence, targets, sha(path))
    assert "strict_CSR_source_gradient_backend_qualified" not in evidence


@pytest.mark.parametrize("drift", [False, True])
def test_preservation_distribution_and_runtime_versions_are_distinct_guards(api, drift):
    expected = SCIENCE["version_binding"]["installed_distribution_versions"]
    spec = dict(source={"current": True}, numerical_source={"runtime_torch": "2.9.1+cu128"},
        python_version=platform.python_version(), files_sha256=SCIENCE["original_files_sha256"],
        artifacts_sha256={"frozen": "frozen"}, scientific_preregistration={"path": "science"})
    checked = []
    api.update(distribution_version=lambda k: "changed" if drift and k == "torch" else expected[k],
        implementation_provenance=lambda: spec["source"], numerical_source=lambda: spec["numerical_source"],
        torch=types.SimpleNamespace(get_num_threads=lambda: 4),
        probe=types.SimpleNamespace(_checked_files=lambda pins: checked.append(pins)),
        SCIENCE_SHA="science", SCIENCE_REVIEW="review", SCIENCE_REVIEW_SHA="review", _sha=lambda p: "spec")
    if drift:
        with pytest.raises(ValueError, match="distribution versions"):
            api["_preserve"](spec, Path("spec"), "spec", SCIENCE, REPO)
    else:
        api["_preserve"](spec, Path("spec"), "spec", SCIENCE, REPO)
    assert SCIENCE["original_files_sha256"] in checked


@pytest.mark.parametrize("relative,allowed", [
    ("src/cora_csr_source_gradient.py", True),
    ("tests/test_cora_csr_source_gradient.py", True),
    ("src/research_loop.py", True),
    ("src/undeclared.py", False),
    ("tests/test_undeclared.py", False),
])
def test_artifact_owned_paths_allowlist(api, tmp_path, relative, allowed):
    pins = {str(tmp_path / relative): "a" * 64}
    if allowed:
        api["_artifact_paths"](pins, SCIENCE, tmp_path)
    else:
        with pytest.raises(ValueError, match="immutable reviewed artifact"):
            api["_artifact_paths"](pins, SCIENCE, tmp_path)
