"""AST-only fake-source/path tests. No adapter/data/Torch/model import or calls."""
import ast
import copy
import hashlib
import json
import os
import sys
import time
import uuid
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _require(value, message):
    if not value:
        raise ValueError(message)


@pytest.fixture
def scope():
    module = Path(__file__).resolve().parents[1] / "citation_source_preflight.py"
    if not module.is_file():
        module = Path(__file__).resolve().parents[1] / "src/citation_source_preflight.py"
    source = ast.parse(module.read_text())
    calls = []
    def checked_files(mapping):
        for path, digest in mapping.items():
            _require(Path(path).is_file() and _sha(path) == digest, "Frozen file changed")
    def stop(callback):
        if callback():
            raise InterruptedError("fake stop")
    probe = SimpleNamespace(_require=_require, _sha=_sha, _checked_files=checked_files, _stop=stop,
                            numerical_source=lambda: {"mock_fullsource_versions": True},
                            _native=lambda device: calls.append("native") or {"mock_only": True},
                            _tensor=lambda value, *args: value,
                            _frozen_original_dense_S=lambda *args: calls.append("FORBIDDEN_dense"))
    cuda = SimpleNamespace(is_initialized=lambda: False, reset_peak_memory_stats=lambda: calls.append("memory_reset"),
                           synchronize=lambda: calls.append("synchronize"), max_memory_allocated=lambda: 0,
                           max_memory_reserved=lambda: 0, current_device=lambda: 0,
                           get_device_properties=lambda device: SimpleNamespace(total_memory=8316977152))
    namespace = dict(Path=Path, json=json, os=os, time=time, uuid=uuid, probe=probe,
                     torch=SimpleNamespace(cuda=cuda, float64="fake_float64", load=lambda *args, **kwargs: {"logits": "fake_logits"}),
                     implementation_provenance=lambda: {"mock_immutable_source": True},
                     _fingerprint=lambda value: hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:12])
    nodes = [node for node in source.body if isinstance(node, (ast.FunctionDef, ast.Assign))]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(module), "exec"), namespace)
    namespace["__file__"] = str(module)
    namespace["calls"] = calls
    return namespace


@pytest.fixture
def frozen(scope, tmp_path):
    repo = tmp_path / "repo"
    (repo / "data/citeseer/processed").mkdir(parents=True)
    (repo / "data/citeseer/processed/data.pt").write_text("fake existing processed input")
    root, files = scope["_paths"](repo)
    for path in files:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text("fake immutable " + path.name)
    science = repo / "results/proposals" / scope["SCIENCE"]
    science.parent.mkdir(parents=True)
    science.write_text('{"mock_science_only": true}')
    scope["SCIENCE_SHA"] = _sha(science)  # controlled mock certificate, not native science evidence
    output = repo / "results/research_loop/new_source_stage/evidence.json"
    spec = dict(schema=1, fixed=copy.deepcopy(scope["FIXED"]), candidate=copy.deepcopy(scope["CANDIDATE"]),
                source_root=str(root), reference_candidate_id=scope["REFERENCE"],
                source=scope["implementation_provenance"](), numerical_source=scope["probe"].numerical_source(),
                files_sha256={str(path): _sha(path) for path in files},
                scientific_preregistration=dict(path=str(science), sha256=scope["SCIENCE_SHA"]), output_path=str(output))
    path = repo / "results/proposals/frozen_spec.json"
    def save():
        path.write_text(json.dumps(spec))
        return str(path), _sha(path), str(output), repo
    return SimpleNamespace(repo=repo, root=root, spec=spec, path=path, output=output, save=save, files=files)


def test_exact_frozen_source_spec_is_readonly(scope, frozen):
    args = frozen.save()
    before = dict(frozen.spec["files_sha256"])
    spec, root, output = scope["_load_spec"](*args)
    assert spec == frozen.spec and root == frozen.root and output == frozen.output
    assert all(_sha(path) == digest for path, digest in before.items())
    assert not output.parent.exists() and scope["calls"] == []


@pytest.mark.parametrize("kind", ["schema_bool", "dimension", "seed", "device", "penalty", "external_F",
                                 "target", "root", "reference", "source", "versions", "science", "output"])
def test_resealed_unsupported_controls_or_context_reject_before_native(scope, frozen, kind):
    spec = frozen.spec
    if kind == "schema_bool":
        spec["schema"] = True
    elif kind in ("dimension", "seed", "device"):
        key, value = {"dimension": ("dimension", 3704), "seed": ("condensation_seed", 1), "device": ("device", "cpu")}[kind]
        spec["fixed"][key] = value
    elif kind == "penalty":
        spec["candidate"]["penalty"] = .0001
    elif kind in ("external_F", "target"):
        spec[kind] = "forbidden"
    elif kind == "root":
        spec["source_root"] = str(frozen.root.parent)
    elif kind == "reference":
        spec["reference_candidate_id"] = "other"
    elif kind == "source":
        spec["source"]["mock_immutable_source"] = False
    elif kind == "versions":
        spec["numerical_source"]["mock_fullsource_versions"] = False
    elif kind == "science":
        spec["scientific_preregistration"]["sha256"] = "wrong"
    else:
        spec["output_path"] = str(frozen.repo / "src/rewrite.json")
    with pytest.raises(ValueError):
        scope["_load_spec"](*frozen.save())
    assert scope["calls"] == [] and not frozen.output.exists()


@pytest.mark.parametrize("fragment", ["teacher.pt", "propagated_H.pt", "inputs_0.pt", "assignment_", "nystrom_map_",
                                     "nystrom_phi_schema3.npy", "nystrom_phi_schema3.meta.json", "resume.pt",
                                     "step_000000.pt", "step_000025.pt", "ind.citeseer.x", "processed/data.pt"])
def test_missing_existing_asset_never_creates_or_downloads(scope, frozen, fragment):
    removed = next(path for path in frozen.files if fragment in str(path))
    removed.unlink()
    with pytest.raises(ValueError):
        scope["_load_spec"](*frozen.save())
    assert not removed.exists() and not frozen.output.parent.exists() and scope["calls"] == []


def test_existing_report_wrong_spec_SHA_and_cache_tamper_preserved(scope, frozen):
    args = frozen.save()
    with pytest.raises(ValueError, match="spec path/SHA"):
        scope["_load_spec"](args[0], "wrong", args[2], args[3])
    frozen.output.parent.mkdir(parents=True)
    frozen.output.write_text("old evidence bytes")
    with pytest.raises(ValueError, match="absent"):
        scope["_load_spec"](*args)
    assert frozen.output.read_text() == "old evidence bytes"
    frozen.output.unlink()
    frozen.files[0].write_text("corrupted asset retained")
    with pytest.raises(ValueError, match="Frozen file changed"):
        scope["_load_spec"](*args)
    assert frozen.files[0].read_text() == "corrupted asset retained"


def test_atomic_report_cannot_overwrite(scope, tmp_path):
    report = tmp_path / "new/report.json"
    scope["_write_new"](report, {"passed": False, "mock_only": True})
    old = report.read_bytes()
    with pytest.raises(FileExistsError):
        scope["_write_new"](report, {"passed": True})
    assert report.read_bytes() == old and not list(report.parent.glob("*.tmp"))


def test_cached_source_RMS_error_is_recorded_and_reraised_before_initializer_heads(scope, frozen, monkeypatch):
    # Invoke actual AST flow with fake dataset/teacher, never production loaders.
    args = frozen.save()
    scope["__file__"] = str(frozen.repo / "src/citation_source_preflight.py")
    (frozen.root / "config.json").write_text('{"mock_graph_context": true}')
    frozen.spec["files_sha256"][str(frozen.root / "config.json")] = _sha(frozen.root / "config.json")
    args = frozen.save()
    fingerprint = scope["_fingerprint"]
    scope["_fingerprint"] = lambda value: scope["ROOT"] if value == {"mock_graph_context": True} else fingerprint(value)
    prepared = []
    monkeypatch.chdir(frozen.repo.parent)
    def prepare(*args):
        prepared.append(args)
        return {"y": "mock_GT"}, "mock_train", ("mock_mask", "mock_val"), "mock_test", "mock_current_H"
    scope["citation_search"] = SimpleNamespace(_prepare_dataset=prepare,
                                              _legacy_teacher_config=lambda *args: {"mock_graph_context": True})
    scope["training_refined_targets"] = lambda *args: "mock_Q"
    scope["_rms_diagnostic"] = lambda *args: {"strict_allclose": False, "max_abs": 3.6e-8, "atol": 1e-12, "rtol": 1e-12}
    def reject(*args):
        scope["calls"].append("cached_source_guard")
        raise ValueError("Original RMS z differs from frozen H/transform")
    scope["source_helper"] = SimpleNamespace(candidate_controls=lambda value: value, cached_source=reject)
    scope["_reference"] = lambda *args: scope["calls"].append("FORBIDDEN_initializer_moments_heads")
    before = copy.deepcopy(frozen.spec["files_sha256"])
    with pytest.raises(ValueError, match="Original RMS z differs from frozen H/transform"):
        scope["prepare_preflight"](*args[:3])
    evidence = json.loads(frozen.output.read_text())
    assert not evidence["passed"] and evidence["source_error"]["message"] == "Original RMS z differs from frozen H/transform"
    assert evidence["RMS_diagnostic_before_unchanged_guard"]["atol"] == 1e-12
    assert evidence["original_assets_unchanged"] and all(_sha(path) == digest for path, digest in before.items())
    assert prepared == [("citeseer", str(frozen.repo / "data"), "cuda", "default")]
    assert not any("FORBIDDEN" in call for call in scope["calls"])
    assert all(evidence[key] == 0 for key in ["P_updates", "head_solves", "student_fits", "optimizer_steps",
               "finite_engine_unrolls", "accuracy_forwards", "native_initializer_calls", "cached_head_gradient_evaluations"])


@pytest.mark.parametrize("kind", ["already_initialized", "stopped"])
def test_stopped_or_already_initialized_native_context_records_failure_without_data(scope, frozen, kind):
    args = frozen.save()
    scope["__file__"] = str(frozen.repo / "src/citation_source_preflight.py")
    scope["torch"].cuda.is_initialized = lambda: kind == "already_initialized"
    scope["_execute"] = lambda *args: pytest.fail("Scientific data work must not run")
    with pytest.raises(ValueError if kind == "already_initialized" else InterruptedError):
        scope["prepare_preflight"](*args[:3], stop=lambda: kind == "stopped")
    assert not json.loads(frozen.output.read_text())["passed"] and scope["calls"] == []


def test_fake_success_report_has_zero_research_counts(scope, frozen):
    args = frozen.save()
    scope["__file__"] = str(frozen.repo / "src/citation_source_preflight.py")
    scope["_execute"] = lambda spec, root, evidence, stop: evidence.update(mock_valid_source_only=True)
    result = scope["prepare_preflight"](*args[:3])
    report = json.loads(frozen.output.read_text())
    assert result["passed"] and result["evidence_sha256"] == _sha(frozen.output)
    assert report["original_assets_unchanged"] and report["mock_valid_source_only"]
    assert all(report[key] == 0 for key in ("P_updates", "student_fits", "head_solves", "finite_engine_unrolls", "accuracy_forwards"))


@pytest.mark.parametrize("kind", ["asset", "spec", "science", "source", "versions"])
def test_failure_also_checks_frozen_assets_spec_science_and_source(scope, frozen, kind):
    args = frozen.save()
    scope["__file__"] = str(frozen.repo / "src/citation_source_preflight.py")
    def reject(spec, root, evidence, stop):
        if kind == "asset":
            frozen.files[0].write_text("tampered fake asset preserved")
        elif kind == "spec":
            frozen.path.write_text("tampered fake spec preserved")
        elif kind == "science":
            Path(spec["scientific_preregistration"]["path"]).write_text("tampered fake science preserved")
        elif kind == "source":
            scope["implementation_provenance"] = lambda: {"mock_immutable_source": False}
        else:
            scope["probe"].numerical_source = lambda: {"mock_fullsource_versions": False}
        raise ValueError("Original RMS z differs from frozen H/transform")
    scope["_execute"] = reject
    with pytest.raises(ValueError, match="Original RMS z differs from frozen H/transform"):
        scope["prepare_preflight"](*args[:3])
    report = json.loads(frozen.output.read_text())
    assert not report["passed"] and report["source_error"]["message"] == "Original RMS z differs from frozen H/transform"
    if kind == "asset":
        assert not report["original_assets_unchanged"]
    else:
        assert not report["source_spec_science_unchanged"] and not report["source_unchanged"]


@pytest.mark.parametrize("kind", ["capacity", "allocated", "reserved"])
def test_fake_native_memory_mismatch_cannot_pass(scope, frozen, kind):
    args = frozen.save()
    scope["__file__"] = str(frozen.repo / "src/citation_source_preflight.py")
    scope["_execute"] = lambda *args: None
    cuda = scope["torch"].cuda
    if kind == "capacity":
        cuda.get_device_properties = lambda device: SimpleNamespace(total_memory=123)
    elif kind == "allocated":
        cuda.max_memory_allocated = lambda: scope["CUDA_CAPACITY"]+1
    else:
        cuda.max_memory_reserved = lambda: scope["CUDA_CAPACITY"]+1
    with pytest.raises(ValueError, match="preservation failed"):
        scope["prepare_preflight"](*args[:3])
    report = json.loads(frozen.output.read_text())
    assert not report["passed"] and not report["native_memory_capacity_passed"]


def test_actual_dispatch_passes_exact_options_stop_and_return(monkeypatch):
    worker = Path(__file__).resolve().parents[1] / "research_loop.py"
    if not worker.is_file():
        worker = Path(__file__).resolve().parents[1] / "src/research_loop.py"
    function = next(node for node in ast.parse(worker.read_text()).body if isinstance(node, ast.FunctionDef) and node.name == "dispatch")
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(worker), "exec"), namespace)
    fake = ModuleType("src.citation_source_preflight")
    returned, calls = object(), []
    fake.prepare_preflight = lambda **kwargs: calls.append(kwargs) or returned
    monkeypatch.setitem(sys.modules, "src", ModuleType("src"))
    monkeypatch.setitem(sys.modules, "src.citation_source_preflight", fake)
    stop = lambda: False
    options = {"spec_path": "mock_spec", "spec_sha256": "mock_SHA", "output_path": "no_file"}
    assert namespace["dispatch"]({"kind": "citation_source_preflight", "options": options}, stop) is returned
    assert calls == [dict(options, stop=stop)] and set(options) == {"spec_path", "spec_sha256", "output_path"}
