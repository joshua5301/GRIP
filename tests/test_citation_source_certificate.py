"""AST/fake-source guards only; zero data/Torch/model/API/native executions."""
import ast
import copy
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


@pytest.fixture
def scope():
    base = Path(__file__).resolve().parents[1]
    module = base / "citation_source_certificate.py"
    if not module.is_file():
        module = base / "src/citation_source_certificate.py"
    repo = next(parent for parent in (base, *Path(__file__).resolve().parents)
                if (parent / "src/citation_source_preflight.py").is_file())
    exact = next(node for node in ast.parse((repo / "src/citation_source_preflight.py").read_text()).body
                 if isinstance(node, ast.FunctionDef) and node.name == "_exact")
    calls = []
    def checked_files(pins):
        for path, digest in pins.items():
            require(Path(path).is_file() and sha(path) == digest, "Frozen file changed")
    namespace = dict(Path=Path, json=json, math=math, platform=platform, time=time, __file__=str(module),
                     implementation_provenance=lambda: {"mock_source_Git": True},
                     _fingerprint=lambda value: hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:12],
                     probe=SimpleNamespace(_require=require, _sha=sha, _checked_files=checked_files,
                         numerical_source=lambda: {"mock_fullsource_versions": True},
                         _stop=lambda callback: require(not callback(), "Stopped"),
                         _native=lambda device: calls.append("native") or {"mock": True},
                         _runtime_precision_guard=lambda: calls.append("precision")))
    nodes = [node for node in ast.parse(module.read_text()).body if isinstance(node, (ast.FunctionDef, ast.Assign))]
    exec(compile(ast.Module(body=[exact, *nodes], type_ignores=[]), str(module), "exec"), namespace)
    namespace["module"], namespace["calls"] = module, calls
    return namespace


@pytest.fixture
def frozen(scope, tmp_path):
    repo = tmp_path / "repo"
    (repo / "data/citeseer/processed").mkdir(parents=True)
    (repo / "data/citeseer/processed/data.pt").write_text("mock processed")
    (repo / "data/citeseer/processed/pre_filter.pt").write_text("mock prefilter")
    (repo / "data/citeseer/processed/pre_transform.pt").write_text("mock pretransform")
    files = scope["_paths"](repo)
    for path in files:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("mock immutable " + str(path))
    parents = []
    for index in range(3):
        path = repo / "results/proposals" / f"parent_{index}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("mock frozen lineage")
        parents.append(dict(path=str(path), sha256=sha(path)))
    science = repo / "results/proposals" / scope["SCIENCE"]
    science.write_text(json.dumps(dict(fixed=scope["FIXED"], roots=scope["_roots"](repo),
                                      candidate=scope["CANDIDATE"], reference_candidate_id=scope["REFERENCE"], parents=parents)))
    scope["SCIENCE_SHA"] = sha(science)
    output = repo / "results/research_loop/new_AT/evidence.json"
    spec = dict(schema=1, fixed=copy.deepcopy(scope["FIXED"]), roots=scope["_roots"](repo),
                candidate=copy.deepcopy(scope["CANDIDATE"]), reference_candidate_id=scope["REFERENCE"],
                source=scope["implementation_provenance"](), numerical_source=scope["probe"].numerical_source(),
                python_version=platform.python_version(), files_sha256={str(path): sha(path) for path in files},
                scientific_preregistration=dict(path=str(science), sha256=scope["SCIENCE_SHA"]), output_path=str(output))
    path = repo / "results/proposals/frozen_spec.json"
    scope["_config"] = lambda root: {"mock": root["root"]}
    def save():
        path.write_text(json.dumps(spec))
        return str(path), sha(path), str(output), repo
    return SimpleNamespace(repo=repo, spec=spec, path=path, output=output, parents=parents, files=files, save=save)


def test_exact_fixed_inventory_and_spec_does_not_initialize_cuda(scope, frozen):
    spec, parents, output = scope["_load_spec"](*frozen.save())
    assert spec == frozen.spec and parents == frozen.parents and output == frozen.output
    assert len(frozen.files) == len(set(frozen.files)) == 35
    assert [row["cells"] for row in spec["roots"]] == [30, 120]
    assert not any("e6669760f3b3" in str(path) for path in frozen.files)
    assert not output.parent.exists() and not scope["calls"]


@pytest.mark.parametrize("kind", ["schema_bool", "seed", "features", "root60", "root_order", "root_bool", "candidateT",
                                  "candidate_penalty", "candidate_bool", "reference", "external", "source", "versions",
                                  "python", "science", "missing", "tamper", "output_exists"])
def test_resealed_spec_and_asset_mutations_reject_before_native(scope, frozen, kind):
    spec = frozen.spec
    if kind == "schema_bool":
        spec["schema"] = True
    elif kind == "seed":
        spec["fixed"]["condensation_seed"] = 1
    elif kind == "features":
        spec["fixed"]["citation_features"] = "default"
    elif kind == "root60":
        spec["roots"][0]["root"] = "e6669760f3b3"
    elif kind == "root_order":
        spec["roots"].reverse()
    elif kind == "root_bool":
        spec["roots"][0]["cells"] = True
    elif kind.startswith("candidate"):
        key, value = {"candidateT": ("T", .3), "candidate_penalty": ("penalty", 1e-4),
                      "candidate_bool": ("T", True)}[kind]
        spec["candidate"][key] = value
    elif kind == "reference":
        spec["reference_candidate_id"] = "other"
    elif kind == "external":
        spec["external_Q"] = "forbidden"
    elif kind == "source":
        spec["source"]["mock_source_Git"] = False
    elif kind == "versions":
        spec["numerical_source"]["mock_fullsource_versions"] = False
    elif kind == "python":
        spec["python_version"] = "other"
    elif kind == "science":
        spec["scientific_preregistration"]["sha256"] = "0"*64
    elif kind == "missing":
        frozen.files[0].unlink()
    elif kind == "tamper":
        frozen.files[0].write_text("tampered")
    else:
        frozen.output.parent.mkdir(parents=True)
        frozen.output.write_text("prior evidence")
    with pytest.raises(ValueError):
        scope["_load_spec"](*frozen.save())
    assert not scope["calls"]


def test_lineage_and_current_source_drift_reject(scope, frozen):
    scope["_load_spec"](*frozen.save())
    Path(frozen.parents[0]["path"]).write_text("changed lineage")
    with pytest.raises(ValueError):
        scope["_load_spec"](*frozen.save())
    scope["implementation_provenance"] = lambda: {"drift": True}
    with pytest.raises(ValueError):
        scope["_source_unchanged"](frozen.spec)


@pytest.mark.parametrize("failure", [None, "loader_first", "loader_second", "loader_memory", "source", "threads", "fresh", "capacity", "counter"])
def test_fake_two_root_execution_preserves_counts_original_failure_and_source(scope, frozen, failure):
    scope["__file__"] = str(frozen.repo / "src/citation_source_certificate.py")
    frozen.save()
    reports, visits = [], []
    def synchronize():
        if failure == "loader_memory":
            raise RuntimeError("secondary memory error")
    cuda = SimpleNamespace(is_initialized=lambda: failure == "fresh", reset_peak_memory_stats=lambda: None,
                           synchronize=synchronize, max_memory_allocated=lambda: 10, max_memory_reserved=lambda: 20,
                           current_device=lambda: 0, get_device_properties=lambda device: SimpleNamespace(total_memory=1 if failure == "capacity" else scope["CUDA_CAPACITY"]))
    scope["torch"] = SimpleNamespace(cuda=cuda, get_num_threads=lambda: 1 if failure == "threads" else 4)
    def execute(row, evidence, stop):
        visits.append(row["cells"])
        if failure in ("loader_first", "loader_memory") or (failure == "loader_second" and row["cells"] == 120):
            evidence["stage"] = "unchanged_cached_source_guard_" + row["root"]
            raise ValueError("original strict cached_source error")
        for key, count in (("native_initializer_calls", 2), ("native_initializer_completed_calls", 2),
                           ("native_P0_probability_material_evaluations", 1),
                           ("cached_head_gradient_evaluations", 2), ("cached_outer_CE_evaluations", 2),
                           ("readonly_native_dense_graph_materializations", 1)):
            evidence[key] += count
        if failure == "counter":
            evidence["native_initializer_calls"] -= 1
        if failure == "source":
            scope["implementation_provenance"] = lambda: {"later_drift": True}
        return {"mock_passed": True}
    scope["_execute_root"] = execute
    def write(path, value):
        reports.append(copy.deepcopy(value))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    scope["_write_new"] = write
    if failure is None:
        result = scope["prepare_certificate"](*frozen.save()[:3])
        assert result["passed"] and visits == [30, 120] and scope["calls"].count("native") == 1
    else:
        with pytest.raises(ValueError, match="original strict cached_source error" if failure.startswith("loader") else ".*"):
            scope["prepare_certificate"](*frozen.save()[:3])
    report = reports[0]
    assert report["passed"] is (failure is None)
    assert all(report[key] == 0 for key in ("P_updates", "optimizer_steps", "student_fits", "head_solves",
                                           "teacher_map_Phi_fits", "model_forwards", "finite_unrolls"))
    if failure and failure.startswith("loader"):
        assert report["source_error"]["message"] == "original strict cached_source error"
        assert report["failed_stage"].startswith("unchanged_cached_source_guard_")
        assert report["native_initializer_calls"] == (2 if failure == "loader_second" else 0)
        if failure == "loader_memory":
            assert report["memory_error"]["message"] == "secondary memory error"
    if failure == "source":
        assert not report["source_assets_spec_science_unchanged"]


@pytest.mark.parametrize("retained", [None, 0, 1, "false"])
def test_actual_reference_rejects_malformed_retention_before_factories(scope, tmp_path, retained):
    root = tmp_path / scope["REFERENCE"]
    root.mkdir()
    (root / "candidate.json").write_text(json.dumps(scope["CANDIDATE"]))
    z = SimpleNamespace(detach=lambda: SimpleNamespace(float=lambda: "mock_F"))
    buffers = dict(root=tmp_path, z=z, q="mock_Q", hard="mock_hard", source="mock_source")
    expected = dict(source_linear_coordinates="raw_rms", source_linear_schema=1, source_linear_source="mock_source", save_assignment=False)
    scope["source_helper"] = SimpleNamespace(expected_citation_config=lambda *args: copy.deepcopy(expected))
    scope["torch"] = SimpleNamespace(load=lambda *args, **kwargs: dict(config=dict(save_assignment=retained), step=25, snapshots={}))
    scope["initialize_factors"] = lambda *args: pytest.fail("Factory called after invalid config")
    with pytest.raises(ValueError, match="artifact retention"):
        scope["_reference"](buffers, "ghost", 30, {})


@pytest.mark.parametrize("kind", ["penalty", "tol", "data_digest", "assignment_input"])
def test_actual_reference_config_mismatch_rejects_before_factories(scope, tmp_path, kind):
    root = tmp_path / scope["REFERENCE"]
    root.mkdir()
    (root / "candidate.json").write_text(json.dumps(scope["CANDIDATE"]))
    expected = dict(source_linear_coordinates="raw_rms", source_linear_schema=1, source_linear_source="source",
                    save_assignment=False, penalty=.001, inner_tol=1e-7, data_digest="actual", assignment_input="features")
    saved = {key: value for key, value in expected.items() if not key.startswith("source_linear")}
    saved["assignment_input"] = "node"
    saved[{"penalty": "penalty", "tol": "inner_tol", "data_digest": "data_digest", "assignment_input": "assignment_input"}[kind]] = "changed"
    z = SimpleNamespace(detach=lambda: SimpleNamespace(float=lambda: "mock_F"))
    scope["source_helper"] = SimpleNamespace(expected_citation_config=lambda *args: copy.deepcopy(expected))
    scope["torch"] = SimpleNamespace(load=lambda *args, **kwargs: dict(config=saved, step=25, snapshots={}))
    scope["initialize_factors"] = lambda *args: pytest.fail("Factory called after config mismatch")
    with pytest.raises(ValueError, match="config/current source"):
        scope["_reference"](dict(root=tmp_path, z=z, q="Q", hard="hard", source="source"), "ghost", 120, {})


def test_actual_worker_lazy_dispatch_preserves_options_stop_return(scope, monkeypatch):
    worker = scope["module"].parent / "research_loop.py"
    dispatch = next(node for node in ast.parse(worker.read_text()).body if isinstance(node, ast.FunctionDef) and node.name == "dispatch")
    calls = []
    module = ModuleType("src.citation_source_certificate")
    module.prepare_certificate = lambda **options: calls.append(options) or {"mock": True}
    monkeypatch.setitem(sys.modules, "src.citation_source_certificate", module)
    namespace = {}
    exec(compile(ast.Module(body=[dispatch], type_ignores=[]), str(worker), "exec"), namespace)
    stop = lambda: False
    assert namespace["dispatch"](dict(kind="citation_source_certificate", options=dict(spec_path="fixed", spec_sha256="digest", output_path="new")), stop) == {"mock": True}
    assert calls == [dict(spec_path="fixed", spec_sha256="digest", output_path="new", stop=stop)]


@pytest.mark.parametrize("cells", [30, 120])
def test_actual_root_loader_error_is_preserved_before_dense_factory_or_head(scope, frozen, cells):
    row = next(root for root in frozen.spec["roots"] if root["cells"] == cells)
    calls = []
    config = {"mock_exact_original_config": cells}
    scope["_config"] = lambda root: config
    scope["_fingerprint"] = lambda value: row["root"]
    def dataset(*args):
        calls.append(("dataset", args))
        return dict(y="Y", adj="S", x="X"), "train", ("unused", "val"), "test", "fresh_H"
    scope["citation_search"] = SimpleNamespace(_prepare_dataset=dataset, _legacy_teacher_config=lambda *args: config)
    scope["torch"] = SimpleNamespace(float64="mock64", load=lambda *args, **kwargs: dict(logits="mock_logits"))
    scope["probe"]._tensor = lambda value, *args: value
    scope["training_refined_targets"] = lambda *args: "Q"
    def loader(*args):
        calls.append(("loader", args))
        raise ValueError("Original RMS z differs from frozen H/transform")
    scope["source_helper"] = SimpleNamespace(candidate_controls=lambda value: value, cached_source=loader)
    scope["probe"]._frozen_original_dense_S = lambda *args: pytest.fail("Dense conversion after loader failure")
    scope["_reference"] = lambda *args: pytest.fail("P/head checks after loader failure")
    evidence = {key: 0 for key in ("native_initializer_calls", "native_P0_probability_material_evaluations",
                                  "cached_head_gradient_evaluations", "cached_outer_CE_evaluations")}
    with pytest.raises(ValueError, match="Original RMS z differs from frozen H/transform"):
        scope["_execute_root"](row, evidence, lambda: False)
    assert calls[0][1][1] == str(scope["module"].resolve().parents[1] / "data")
    assert calls[0][1][3] == "row"
    assert calls[1][1][0] == Path(row["source_root"])
    assert calls[1][1][1]["T"] == 1. and calls[1][1][1]["alpha"] == .3
    assert evidence["stage"] == "unchanged_cached_source_guard_" + row["root"]
    assert all(evidence[key] == 0 for key in ("native_initializer_calls", "native_P0_probability_material_evaluations",
                                             "cached_head_gradient_evaluations", "cached_outer_CE_evaluations"))


@pytest.mark.parametrize("failed_call,attempts,completed", [(1, 1, 0), (2, 2, 1)])
def test_actual_reference_factory_failure_counters_are_honest(scope, tmp_path, failed_call, attempts, completed):
    root = tmp_path / scope["REFERENCE"]
    root.mkdir()
    (root / "candidate.json").write_text(json.dumps(scope["CANDIDATE"]))
    expected = dict(source_linear_coordinates="raw_rms", source_linear_schema=1, source_linear_source="source",
                    save_assignment=False, penalty=.001, inner_tol=1e-7, data_digest="actual", assignment_input="features")
    saved = {key: value for key, value in expected.items() if not key.startswith("source_linear")}
    saved["assignment_input"] = "node"
    z = SimpleNamespace(detach=lambda: SimpleNamespace(float=lambda: "mock_F"))
    scope["source_helper"] = SimpleNamespace(expected_citation_config=lambda *args: copy.deepcopy(expected))
    scope["torch"] = SimpleNamespace(load=lambda *args, **kwargs: dict(config=saved, step=25, snapshots={}))
    scope["probe"]._scalar = lambda value, message: value
    scope["make_material"] = lambda *args: "mock_material"
    calls = []
    def factory(*args):
        calls.append(args)
        if len(calls) == failed_call:
            raise ValueError("original native factory failure")
        return "mock_U", "mock_V"
    scope["initialize_factors"] = factory
    evidence = dict(native_initializer_calls=0, native_initializer_completed_calls=0)
    with pytest.raises(ValueError, match="original native factory failure"):
        scope["_reference"](dict(root=tmp_path, z=z, q="Q", hard="hard", source="source"), "ghost", 30, evidence)
    assert evidence["native_initializer_calls"] == attempts
    assert evidence["native_initializer_completed_calls"] == completed
    assert len(calls) == attempts and all(args == ("hard", 30, 8, 0) for args in calls)
