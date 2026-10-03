"""Six new stdlib AST/fake groups; no Torch, datasets, PT or numerical imports."""
import ast
import copy
import hashlib
import json
import math
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src/large_current_origin.py").is_file())
DRAFT = (ROOT / "src/large_feature_tile_source_gradient.py" if Path(__file__).resolve().parent == ROOT / "tests" else
         ROOT / "results/implementation_drafts/large_feature_tile_source_gradient_stageBY_v1/proposed/src/large_feature_tile_source_gradient.py")
WORKER = DRAFT.with_name("research_loop.py")
TREE = ast.parse(DRAFT.read_text())
PROTECTED_WORKER_AST_SHA = "b130a4a3b0f4b0e52e8bd33aa63431061e0fe8e3df034f0b2d0ed0810535b982"


def seal(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def function(name):
    return copy.deepcopy(next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == name))


def literal(name):
    return next(ast.literal_eval(n.value) for n in TREE.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))


def load(*names, **extra):
    namespace = dict(Path=Path, json=json, math=math, os=os, time=time, traceback=__import__("traceback"),
                     _require=require, _seal=seal, _sha=lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest())
    namespace.update({name: literal(name) for name in ("FIELDS", "SCIENCE", "SCIENCE_SHA", "BACKEND")})
    namespace.update(extra)
    module = ast.Module(body=[function(n) for n in names], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(DRAFT), "exec"), namespace)
    return namespace


def invoke(evidence, key, fn, *args, **kwargs):
    evidence["_guard"]()
    evidence["operation_attempts"][key] = evidence["operation_attempts"].get(key, 0) + 1
    value = fn(*args, **kwargs)
    evidence["counts"][key] = evidence["counts"].get(key, 0) + 1
    return value


def begin(evidence, key):
    evidence["operation_attempts"][key] = evidence["operation_attempts"].get(key, 0) + 1


def end(evidence, key):
    evidence["counts"][key] = evidence["counts"].get(key, 0) + 1


def evidence():
    return dict(_guard=lambda: None, counts={}, operation_attempts={})


def test_group1_exact_spec_owned_paths_namespace_and_worker_AST(tmp_path):
    n = load("_controls", "_request")
    owned = ["src/large_feature_tile_source_gradient.py", "src/research_loop.py", "tests/test_large_feature_tile_source_gradient.py"]
    contract = dict(owned_paths=owned, output_root=str(tmp_path / "results/out"), pass_outputs={"1": str(tmp_path / "results/out/pass1/receipt.json"),
                    "2": str(tmp_path / "results/out/pass2/receipt.json")}, target_outputs={"1": str(tmp_path / "results/out/pass1/cache.pt"),
                    "2": str(tmp_path / "results/out/pass2/cache.pt")})
    science = dict(implementation_contract=contract, fixed={"anchor_seed": 0}, original_files_sha256={"asset": "a"},
                   comparison_policy={"exact": True}, source_before={"files": {"src/old.py": "o", "src/research_loop.py": "w"}},
                   native_policy={"versions": {"torch": "literal"}, "python_version": "literal"})
    spec = dict(schema=1, fixed=science["fixed"], source={"files": dict(science["source_before"]["files"], **{owned[0]: "n"})},
                numerical_source={"versions": science["native_policy"]["versions"]}, python_version="literal", files_sha256=science["original_files_sha256"],
                scientific_preregistration=dict(path=str(tmp_path / "results/proposals" / n["SCIENCE"]), sha256=n["SCIENCE_SHA"]),
                artifacts_sha256={str(tmp_path / p): "h" for p in owned}, output_root=contract["output_root"],
                pass_outputs=contract["pass_outputs"], target_outputs=contract["target_outputs"], comparison_policy=science["comparison_policy"])
    n["_controls"](spec, science, tmp_path)
    for path in ("src/undeclared.py", "tests/test_undeclared.py"):
        changed = copy.deepcopy(spec)
        changed["artifacts_sha256"][str(tmp_path / path)] = "h"
        with pytest.raises(ValueError, match="owned artifacts"):
            n["_controls"](changed, science, tmp_path)
    for change in ({"schema": True}, {"extra": 0}, {"fixed": {"anchor_seed": 1}}):
        with pytest.raises(ValueError):
            n["_controls"](dict(spec, **change), science, tmp_path)
    n["_request"](1, None, spec["pass_outputs"]["1"], spec)
    for index, previous in ((True, None), (1, "h" * 64), (2, None)):
        with pytest.raises(ValueError):
            n["_request"](index, previous, spec["pass_outputs"]["1"], spec)
    Path(spec["pass_outputs"]["1"]).parent.mkdir(parents=True)
    with pytest.raises(ValueError, match="exclusive"):
        n["_request"](1, None, spec["pass_outputs"]["1"], spec)
    worker = ast.parse(WORKER.read_text())
    dispatch = next(fn for fn in worker.body if isinstance(fn, ast.FunctionDef) and fn.name == "dispatch")
    matches = [fn for fn in dispatch.body if isinstance(fn, ast.If) and "'large_feature_tile_source_gradient'" in ast.unparse(fn.test)]
    assert len(matches) == 1 and len(matches[0].body) == 2
    assert ast.unparse(matches[0].body[1]) == "return prepare_source_gradient(**options, stop=stop)"
    dispatch.body.remove(matches[0])
    assert hashlib.sha256(ast.dump(worker, include_attributes=False).encode()).hexdigest() == PROTECTED_WORKER_AST_SHA
    assert [a.arg for a in function("prepare_source_gradient").args.args] == ["pass_index", "spec_path", "spec_sha256", "output_path", "previous_sha256", "stop"]


def test_group2_single_BW_CPU_load_validator_and_historical_current_separation(tmp_path):
    keys = ["X", "original_CSR", "graph_y_identity", "train_mask", "validation_mask", "saved_H", "H_double", "Q", "hard", "z", "transform", "U0", "V0", "M0", "readout", "theta0"]
    arrays, descriptors = {k: "array-" + k for k in keys}, {k: "bits-" + k for k in keys}
    context = dict(source={"source_count": 85}, numerical_source={"historical": 85}, python_version="3.12.7", spec={"path": "old", "sha256": "o"})
    packet = dict(schema=1, context=context, arrays=arrays, descriptors=descriptors, content_sha256="content")
    capture = tmp_path / "capture.json"
    capture.write_text(json.dumps(dict(origin_context=context)))
    binding = dict(origin_path="opaque.pt", origin_file_sha256="file", origin_content_sha256="content", origin_context_sha256="context",
                   full16_descriptors=descriptors, historical_source=context["source"], historical_numerical_source=context["numerical_source"],
                   historical_python_version="3.12.7", historical_spec=context["spec"], capture_receipt={"path": str(capture)})
    calls = []
    def fake_load(path, **kwargs):
        calls.append((path, kwargs))
        return packet
    def validate(libs, value, bound):
        assert value is packet and bound == dict(origin_content_sha256="content", origin_context_sha256="context")
    libs = {"torch": SimpleNamespace(load=fake_load), "src.csr_explicit_adjoint": SimpleNamespace(_invoke=invoke, _csr=lambda s: require(s == arrays["original_CSR"], "CSR"))}
    n = load("_origin", _validate_packet=validate, _descriptor=lambda libs, a: dict(descriptors))
    e, retained = evidence(), {}
    result = n["_origin"](libs, {"origin_binding": binding}, e, retained)
    assert result is arrays and calls == [("opaque.pt", dict(map_location="cpu", weights_only=False))]
    assert e["counts"] == {"accepted_BW_origin_CPU_loads": 1, "BW_full_packet_validation_calls": 1}
    assert e["BW_origin"]["historical_source"] == {"source_count": 85}
    context["source"] = {"source_count": 87}
    with pytest.raises(ValueError, match="Historical"):
        n["_origin"](libs, {"origin_binding": binding}, evidence(), {})
    alias = next(n for n in TREE.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "_validate_packet" for t in n.targets))
    assert ast.unparse(alias.value) == "bw._validate_packet"


class Array:
    def __init__(self, bits, device="cpu", finite=True):
        self.bits, self.device, self.finite = bits, SimpleNamespace(type=device), finite
        self.layout, self.requires_grad = "strided", False
        self.transfers = []
    def to(self, **kwargs):
        self.transfers.append(kwargs)
        return Array(self.bits, kwargs["device"], self.finite)
    def detach(self):
        return self
    def cpu(self):
        return Array(self.bits, "cpu", self.finite)


def descriptor(libs, value):
    if isinstance(value, Array):
        return value.bits
    if isinstance(value, dict):
        return {k: descriptor(libs, v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [descriptor(libs, v) for v in value]
    return value


def fake_torch():
    return SimpleNamespace(is_tensor=lambda v: isinstance(v, Array), strided="strided", sparse_csr="csr",
                           isfinite=lambda v: SimpleNamespace(all=lambda: v.finite), float32="torch.float32")


def test_group3_only_original_X_S_Q_CUDA_and_one_frozen_anchor_gradient_packet():
    arrays = {k: Array(k) for k in ("X", "original_CSR", "Q", "saved_H", "z")}
    calls = []
    anchor = tuple(Array(k, "cuda") for k in ("W1", "b1", "W2", "b2"))
    result = dict(transpose=Array("T", "cuda"), boundaries={"A": Array("A", "cuda"), "logp": Array("logp-bits", "cuda"),
                  "full_logp": Array("logp-bits", "cuda"), "full_CE": 2.}, loss=2.)
    packet = dict(blocks=("W1", "b1", "W2", "b2"), rho=.001, gradients=anchor, source_norms=(1., 2., 3., 4.), delta=(.001, .002, .003, .004))
    def factory(*args, **kwargs):
        calls.append(("anchor", args, kwargs))
        return anchor
    def gradient(*args):
        calls.append(("gradient", args))
        assert args[1].bits == "X" and args[2].bits == "original_CSR" and args[3].bits == "Q"
        return result
    def target(*args):
        calls.append(("target", args))
        return packet
    libs = {"torch": fake_torch(), "src.csr_explicit_adjoint": SimpleNamespace(_invoke=invoke, _begin=begin, _end=end),
            "src.csr_feature_tile_adjoint": SimpleNamespace(gradient_boundaries=gradient, target_packet=target),
            "src.citation_gradient_probe": SimpleNamespace(anchor_initial=factory)}
    n = load("_collect", _descriptor=descriptor, _finite_tree=lambda *args: None, _common_contract=lambda *args: None)
    e, retained = evidence(), {}
    common = n["_collect"](libs, arrays, {"origin_binding": {"full16_descriptors": {k: k for k in arrays}}}, e, retained)
    assert calls[0] == ("anchor", (128, 40, 256, 0), dict(dtype="torch.float32", device="cuda"))
    assert [c[0] for c in calls] == ["anchor", "gradient", "target"]
    assert set(retained["inputs"]) == {"X", "original_CSR", "Q"}
    assert arrays["saved_H"].transfers == arrays["z"].transfers == []
    assert e["counts"]["source_X_S_Q_CUDA_transfer_calls"] == 3 and e["counts"]["source_GCN_gradient_targets_completed"] == 1
    assert common["S"] is arrays["original_CSR"] and e["source_gradient_norm_values"] == [1., 2., 3., 4.]
    result["boundaries"]["full_logp"] = Array("different-signed-zero-bits", "cuda")
    with pytest.raises(ValueError, match="descriptor bits"):
        n["_collect"](libs, arrays, {"origin_binding": {"full16_descriptors": {k: k for k in arrays}}}, evidence(), {})
    body = ast.unparse(function("_collect"))
    assert "helper.gradient_boundaries" in body and "helper.target_packet" in body and "_product(" not in body


def test_group4_full_finite_CPU_capture_before_repeat_and_raw_preservation(tmp_path):
    stored, calls = [], []
    t = fake_torch()
    def save(payload, stream):
        stored.append(payload)
        calls.append("save")
        stream.write(b"opaque fake snapshot")
    t.save = save
    libs = {"torch": t, "src.csr_explicit_adjoint": SimpleNamespace(_invoke=invoke)}
    n = load("_finite_tree", "_copy_cpu", "_capture", "_observe_raw", _descriptor=descriptor)
    common = dict(A=Array("negative-pre-ReLU", "cuda"), mask=Array("mask", "cuda"))
    e = evidence()
    e["common_numerical_digest"] = descriptor(libs, common)
    target = tmp_path / "pass2/raw.pt"
    n["_capture"](libs, target, {"current": 87}, 2, common, e)
    assert target.read_bytes() == b"opaque fake snapshot" and calls == ["save"]
    assert e["current_finite_CPU_raw_capture_completed"] is True
    assert stored[0]["common"]["A"].device.type == "cpu" and not hasattr(stored[0]["common"]["A"], "clone")
    with pytest.raises(ValueError, match="Nonfinite"):
        n["_finite_tree"](libs, {"A": Array("overflow", finite=False)})
    with pytest.raises(ValueError, match="CPU only"):
        n["_finite_tree"](libs, common, cpu_only=True)
    body = ast.unparse(function("prepare_source_gradient"))
    assert body.index("_capture(") < body.index("_repeat(")
    # A comparison failure never deletes the already finite exclusive cache.
    with pytest.raises(ValueError):
        require(False, "repeat differs")
    assert target.exists() and e["raw_cache_is_qualification"] is False
    n["_observe_raw"](target, e)
    assert e["observed_raw_cache_complete"] is True and e["observed_raw_cache_bytes"] > 0
    def partial_save(payload, stream):
        stream.write(b"partial raw bytes")
        raise OSError("write interrupted")
    t.save = partial_save
    partial, failed = tmp_path / "partial/raw.pt", evidence()
    failed["common_numerical_digest"] = descriptor(libs, common)
    with pytest.raises(OSError, match="interrupted"):
        n["_capture"](libs, partial, {"current": 87}, 2, common, failed)
    n["_observe_raw"](partial, failed)
    assert failed["observed_raw_cache_partial_unqualified"] is True and partial.read_bytes() == b"partial raw bytes"
    assert failed["operation_attempts"]["raw_cache_writes"] == 1 and "raw_cache_writes" not in failed["counts"]


def test_group5_pass2_CPU_only_reseal_complete_repeat_no_second_gradient(tmp_path):
    common = {"complete": ["all", "bits"]}
    context = {"current": 87, "BW_historical": 85}
    digest = common
    payload = dict(schema=1, pass_index=1, context=context, common=common, common_numerical_digest=digest, common_numerical_seal=seal(digest))
    payload["content_sha256"] = seal(dict(schema=1, pass_index=1, context=context, common_numerical_digest=digest))
    first = dict(raw_cache_context_sha256=seal(context), common_numerical_digest=digest, common_numerical_seal=seal(digest),
                 raw_cache_content_sha256=payload["content_sha256"], raw_cache_sha256="opaque")
    loads = []
    def fake_load(path, **kwargs):
        loads.append(kwargs)
        return payload
    libs = {"torch": SimpleNamespace(load=fake_load), "src.csr_explicit_adjoint": SimpleNamespace(_invoke=invoke)}
    n = load("_repeat", _previous=lambda *args: (first, tmp_path / "receipt.json", tmp_path / "raw.pt"),
             _finite_tree=lambda *args, **kwargs: None, _common_contract=lambda *args: None,
             _context=lambda *args: context, _descriptor=lambda libs, value: copy.deepcopy(value))
    e = evidence()
    e.update(common_numerical_digest=digest, common_numerical_seal=seal(digest))
    n["_repeat"](libs, {}, {}, e, "previous-SHA")
    assert loads == [dict(map_location="cpu", weights_only=False)] and e["within_anchor0_feature_band16_exact_repeat_passed"] is True
    changed = evidence()
    changed.update(common_numerical_digest={"complete": ["changed"]}, common_numerical_seal=seal({"complete": ["changed"]}))
    with pytest.raises(ValueError, match="exact repeat"):
        n["_repeat"](libs, {}, {}, changed, "previous-SHA")
    assert changed["counts"] == {"previous_raw_cache_CPU_loads": 1}
    payload["schema"] = True
    with pytest.raises(ValueError, match="Malformed"):
        n["_repeat"](libs, {}, {}, changed, "previous-SHA")
    text = ast.unparse(function("_repeat"))
    assert "gradient_boundaries" not in text and "target_packet" not in text and "'cuda'" not in text


def test_group6_failure_counts_deadline_resources_and_quarantine_finally(tmp_path):
    e = evidence()
    with pytest.raises(RuntimeError):
        invoke(e, "actual", lambda: (_ for _ in ()).throw(RuntimeError("partial")))
    assert e["operation_attempts"] == {"actual": 1} and e["counts"] == {}
    n = load("_guard", _precision=lambda t: None)
    with pytest.raises(ValueError, match="Stopped"):
        n["_guard"]({"torch": None}, time.monotonic(), lambda: True)
    with pytest.raises(ValueError, match="deadline"):
        n["_guard"]({"torch": None}, time.monotonic() - 301, lambda: False)
    cuda = SimpleNamespace(synchronize=lambda: None, max_memory_allocated=lambda i: 2, max_memory_reserved=lambda i: 3,
                           get_device_properties=lambda i: SimpleNamespace(total_memory=8), get_rng_state=lambda: SimpleNamespace(clone=lambda: "rng"))
    resource = load("_resources")
    resource["_resources"]({"torch": SimpleNamespace(cuda=cuda)}, time.monotonic(), {}, {"native_policy": {"CUDA_total_bytes": 8}})
    cuda.max_memory_reserved = lambda i: 9
    with pytest.raises(ValueError, match="memory"):
        resource["_resources"]({"torch": SimpleNamespace(cuda=cuda)}, time.monotonic(), {}, {"native_policy": {"CUDA_total_bytes": 8}})
    output, raw = tmp_path / "receipt.json", tmp_path / "raw.pt"
    spec = dict(source={"current": 87}, numerical_source={"versions": {}}, python_version="literal", scientific_preregistration={}, fixed={}, files_sha256={})
    science = {"counts_contract": {"zero_work_per_worker": {"P_updates": 0, "student_fits": 0}, "scope": "actual"}}
    torch = SimpleNamespace(random=SimpleNamespace(get_rng_state=lambda: SimpleNamespace(clone=lambda: "rng")), cuda=cuda, equal=lambda a, b: True)
    def capture(libs, target, context, pass_index, common, evidence):
        target.write_bytes(b"kept")
        evidence["raw_cache_path"] = str(target)
        evidence["counts"]["source_GCN_gradient_targets_completed"] = 1
    def late_failure(*args):
        raise ValueError("retained mutation")
    runtime = load("prepare_source_gradient", "_observe_raw", _load_spec=lambda *args: (spec, science, tmp_path), _request=lambda *args: (output, raw),
                   _libraries=lambda: {"torch": torch}, _native=lambda *args: dict(versions={}), _guard=lambda *args: None,
                   _preserve=lambda *args: None, _origin=lambda *args: {}, _collect=lambda *args: {}, _context=lambda *args: {},
                   _capture=capture, _retained=late_failure, _precision=lambda *args: None, _resources=lambda *args: None)
    with pytest.raises(ValueError, match="retained mutation"):
        runtime["prepare_source_gradient"](1, tmp_path / "spec.json", "h" * 64, output)
    receipt = json.loads(output.read_text())
    assert receipt["passed"] is False and receipt["success_counts"] is None and raw.read_bytes() == b"kept"
    assert receipt["counts"]["source_GCN_gradient_targets_completed"] == 1 and receipt["counts"]["P_updates"] == 0
    for flag in ("actual_source_gradient_accuracy_qualified", "large_gradient_backend_qualified", "production_alignment_target_consumption_allowed",
                 "anchor0_resource_reproducibility_qualified", "local_resource_finite_passed"):
        assert receipt[flag] is False
