"""AST-only BK path, exact-repeat and failure controls; no production import."""
import ast
import copy
import hashlib
import json
import platform
import time
import traceback
import types
from pathlib import Path

import pytest

BASE = Path(__file__).resolve().parents[1]
MODULE = BASE / "src/cora_csr_adjoint_isolation.py"
WORKER = BASE / "src/research_loop.py"
REPO = next(p for p in Path(__file__).resolve().parents if (p / "src/transforms.py").is_file())
SCIENCE = json.loads((REPO / "results/proposals/Cora70_original_CSR_explicit_adjoint_fixedanchor0_scientific_stageBK_v1.json").read_text())


def require(condition, message):
    if not condition: raise ValueError(message)


def seal(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def api():
    names = {"_artifact_paths", "_request", "_previous", "_repeat", "_capture", "_context", "prepare_isolation"}
    tree = ast.parse(MODULE.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    probe = types.SimpleNamespace(_digest=copy.deepcopy, _attach=lambda v: dict(v, content_sha256=seal(v)))
    invoke = lambda evidence, key, fn, *a, **kw: fn(*a, **kw)
    ns = dict(__file__=str(MODULE), Path=Path, json=json, time=time, traceback=traceback, platform=platform,
              _require=require, _sha=sha, _seal=seal, _exact=lambda a, b: type(a) is type(b) and a == b,
              _stop=lambda stop: require(not stop(), "stopped"), probe=probe,
              explicit=types.SimpleNamespace(_invoke=invoke), _BACKEND="existing_original_CSR_with_explicit_transpose_CSR")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(MODULE), "exec"), ns)
    return ns


def outputs(tmp):
    return dict(pass_outputs={str(i): str(tmp / f"pass{i}/receipt.json") for i in (1, 2)},
                target_outputs={str(i): str(tmp / f"pass{i}/cache.pt") for i in (1, 2)})


@pytest.mark.parametrize("relative", SCIENCE["helper_boundary"]["next_owned_sources"] + SCIENCE["helper_boundary"]["next_owned_tests"] + ["src/research_loop.py"])
def test_exact_five_owned_artifacts_accepted(api, relative):
    api["_artifact_paths"]({str(REPO / relative): "pin"}, SCIENCE, REPO)


@pytest.mark.parametrize("relative", ["src/undeclared.py", "tests/undeclared.py", "README.md"])
def test_undeclared_production_artifacts_rejected(api, relative):
    with pytest.raises(ValueError): api["_artifact_paths"]({str(REPO / relative): "pin"}, SCIENCE, REPO)


@pytest.mark.parametrize("index,previous", [(True, None), (1, "a" * 64), (2, None)])
def test_phase_protocol_rejects_before_native(api, tmp_path, index, previous):
    spec = outputs(tmp_path)
    with pytest.raises(ValueError): api["_request"](index, previous, spec["pass_outputs"]["1"], spec)


def fixture_first(api, tmp_path):
    spec = outputs(tmp_path)
    evidence = dict(source={"files": "fixed"}, numerical_source={"versions": "fixed"}, python_version="fixed",
                    scientific_preregistration={"sha256": "fixed"}, spec_path="fixed", spec_sha256="fixed",
                    source_context={"ghost": "current"}, native_source_buffers_before={"X": "fixed"}, native_environment={"threads": 4})
    common = {"boundaries": {"D": "fixed"}, "target": "fixed"}
    payload = dict(context=api["_context"](spec | evidence, evidence), pass_index=1, common=common, isolated={"B": "old diagnostic"})
    path, target = Path(spec["pass_outputs"]["1"]), Path(spec["target_outputs"]["1"])
    target.parent.mkdir()
    target.write_text(json.dumps(payload))
    first = dict(evidence, passed=True, pass_index=1, operation="prepare_isolation", source_backend=api["_BACKEND"],
                 source_assets_spec_science_unchanged=True, source_reference_provenance_passed=True,
                 within_explicit_adjoint_exact_repeat_passed=False, fixedanchor0_explicit_adjoint_backend_qualified=False,
                 test_enabled=False, validation_only=True, success_counts=SCIENCE["counts_contract"]["planned_per_fully_successful_worker"],
                 raw_cache_path=str(target), raw_cache_sha256=sha(target), common_numerical_digest=common, common_numerical_seal=seal(common))
    path.write_text(json.dumps(first))
    api["torch"] = types.SimpleNamespace(load=lambda *a, **k: payload, save=lambda value, stream: stream.write(json.dumps(value).encode()))
    api["bi"] = types.SimpleNamespace(_sealed_payload=lambda v: v)
    return spec | evidence, evidence, common, path


def test_repeat_excludes_isolated_outputs_and_administration(api, tmp_path):
    spec, evidence, common, path = fixture_first(api, tmp_path)
    evidence["pass_index"] = 2
    api["_repeat"](spec, SCIENCE, common, evidence, sha(path))
    assert evidence["within_explicit_adjoint_exact_repeat_passed"] and evidence["fixedanchor0_explicit_adjoint_backend_qualified"]
    assert "isolated" not in common


def test_changed_D_repeat_fails_after_raw_capture_is_preserved(api, tmp_path):
    spec, evidence, common, path = fixture_first(api, tmp_path)
    common = copy.deepcopy(common)
    common["boundaries"]["D"] = "changed"
    target = Path(spec["target_outputs"]["2"])
    api["_capture"](spec, target, api["_context"](spec, evidence), 2, common, {"B": "new diagnostic"}, evidence)
    with pytest.raises(ValueError, match="exact common repeat"):
        api["_repeat"](spec, SCIENCE, common, evidence, sha(path))
    assert target.is_file() and evidence["raw_cache_sha256"] == sha(target)
    assert evidence["raw_cache_is_qualification"] is False
    assert "fixedanchor0_explicit_adjoint_backend_qualified" not in evidence


@pytest.mark.parametrize("mutated", ["passed", "success_counts", "raw_cache_sha256"])
def test_resealed_previous_receipt_rejects_unknown_authority(api, tmp_path, mutated):
    spec, evidence, common, path = fixture_first(api, tmp_path)
    first = json.loads(path.read_text())
    first[mutated] = False if mutated == "passed" else {} if mutated == "success_counts" else "wrong"
    path.write_text(json.dumps(first))
    with pytest.raises(ValueError): api["_previous"](spec, SCIENCE, evidence, sha(path))


def test_primary_native_failure_survives_preservation_failure(api, tmp_path):
    spec = outputs(tmp_path)
    spec.update(source={}, numerical_source={}, python_version="fixed", scientific_preregistration={}, files_sha256={})
    api["_load_spec"] = lambda *a: (spec, SCIENCE)
    api["_preserve"] = lambda *a: (_ for _ in ()).throw(ValueError("preservation failed"))
    api["probe"]._runtime_precision_guard = lambda: None
    api["probe"]._native = lambda *a: (_ for _ in ()).throw(RuntimeError("native primary"))
    api["torch"] = types.SimpleNamespace(cuda=types.SimpleNamespace(is_initialized=lambda: False))
    def write(path, value):
        path.parent.mkdir()
        path.write_text(json.dumps(value))
    api["_write_new"] = write
    with pytest.raises(RuntimeError, match="native primary"):
        api["prepare_isolation"](1, tmp_path / "spec.json", "a" * 64, spec["pass_outputs"]["1"])
    receipt = json.loads(Path(spec["pass_outputs"]["1"]).read_text())
    assert not receipt["passed"] and receipt["error"] == "native primary" and "preservation failed" in receipt["preservation_error"]
    assert not receipt["fixedanchor0_explicit_adjoint_backend_qualified"]


def test_worker_one_lazy_branch_preserves_stop_and_return():
    tree = ast.parse(WORKER.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "dispatch")
    branch = next(n for n in fn.body if isinstance(n, ast.If) and ast.unparse(n.test) == "job['kind'] == 'citation_cora_csr_adjoint_isolation'")
    assert ast.unparse(branch.body[0]) == "from src.cora_csr_adjoint_isolation import prepare_isolation"
    assert ast.unparse(branch.body[1]) == "return prepare_isolation(**options, stop=stop)"
    # Check branch syntax, not any production import.
