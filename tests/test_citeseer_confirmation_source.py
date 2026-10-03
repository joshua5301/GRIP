"""New BO controls only: AST extraction + stdlib/fake native operations."""
import ast
import copy
import csv
import hashlib
import json
import math
import sys
import types
from pathlib import Path

import pytest

HERE = Path(__file__).resolve()
REPO = next(p for p in HERE.parents if (p / "src/citation_source_certificate.py").is_file())
CANDIDATE = HERE.parents[1] / "src/citeseer_confirmation_source.py"
if not CANDIDATE.is_file():
    CANDIDATE = REPO / "src/citeseer_confirmation_source.py"
WORKER = CANDIDATE.parent / "research_loop.py"
TEXT = CANDIDATE.read_text()
TREE = ast.parse(TEXT)
SCIENCE = json.loads((REPO / "results/proposals/Citeseer120_source345_and_NODE5_prerequisite_scientific_stageBO_v2.json").read_text())


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seal(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def extract(names, **values):
    selected = [n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name in names]
    scope = dict(Path=Path, json=json, csv=csv, math=math, _require=require, _sha=sha, _seal=seal,
                 _exact=lambda a, b: type(a) is type(b) and a == b)
    for node in TREE.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in
                {"_NAN", "_MISSING", "_SPEC_FIELDS", "SCIENCE", "SCIENCE_SHA", "_INTEGER_CSV_POLICY", "CUDA_CAPACITY"} for t in node.targets):
            exec(compile(ast.Module([node], type_ignores=[]), "constants", "exec"), scope)
    scope.update(values)
    exec(compile(ast.Module(selected, type_ignores=[]), str(CANDIDATE), "exec"), scope)
    return scope


@pytest.mark.parametrize("operation,seed", [("certify_existing", 3), ("certify_existing", 4),
    ("prepare_initializer", 5), ("prepare_node_reference", 5)])
def test_fixed_operation_seed(operation, seed):
    assert extract({"_operation"})["_operation"](operation, seed) == seed


@pytest.mark.parametrize("operation,seed", [("certify_existing", 5), ("prepare_initializer", 3),
    ("prepare_node_reference", True), ("prepare_node_reference", 5.), ("retry", 5)])
def test_operation_cannot_retarget(operation, seed):
    with pytest.raises(ValueError):
        extract({"_operation"})["_operation"](operation, seed)


def history_fixture(tmp_path):
    rows = []
    for step in range(26):
        row = dict(step=step, inner_converged=True, J_exact=True, status="evaluated" if step == 25 else "update",
                   J=1., inner_grad_max=1e-9, seconds=.1,
                   row_residual=float("nan"), column_residual=float("nan"),
                   head_correction_relative=float("nan") if step == 0 else .01,
                   implicit_correction_relative=float("nan") if step in (0, 25) else .01,
                   cg_residual=float("nan") if step == 25 else 1e-8,
                   cg_relative_residual=float("nan") if step == 25 else 1e-8, cg_converged=True)
        if step < 25:
            row.update(hessian_solver="reduced_direct", hessian_reduced_dimension=720)
        rows.append(row)
    columns = list(dict.fromkeys(k for row in rows for k in row))
    csv_rows = [{k: "" if k not in r or (type(r[k]) is float and math.isnan(r[k])) else
                 "720.0" if k == "hessian_reduced_dimension" else str(r[k]) for k in columns} for r in rows]
    path = tmp_path / "optimization.csv"
    return rows, columns, csv_rows, path


def write_csv(path, columns, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, columns)
        writer.writeheader()
        writer.writerows(rows)


@pytest.mark.parametrize("mutation", [None, "native_float", "native_bool", "wrong_integer", "wrong_csv", "misplaced_NaN", "infinity", "missing_nonterminal"])
def test_actual_typed_history_policy(tmp_path, mutation):
    rows, columns, disk, path = history_fixture(tmp_path)
    if mutation == "native_float":
        rows[7]["hessian_reduced_dimension"] = 720.
    elif mutation == "native_bool":
        rows[7]["hessian_reduced_dimension"] = True
    elif mutation == "wrong_integer":
        rows[7]["hessian_reduced_dimension"] = 721
    elif mutation == "wrong_csv":
        disk[7]["hessian_reduced_dimension"] = "720"
    elif mutation == "misplaced_NaN":
        rows[7]["J"], disk[7]["J"] = float("nan"), ""
    elif mutation == "infinity":
        rows[7]["cg_residual"], disk[7]["cg_residual"] = float("inf"), "inf"
    elif mutation == "missing_nonterminal":
        del rows[7]["hessian_solver"]
        disk[7]["hessian_solver"] = ""
    write_csv(path, columns, disk)
    scope = extract({"_nullable_integer_csv", "_history_metadata"}, _history=lambda resume, p: resume["history"])
    if mutation:
        with pytest.raises(ValueError):
            scope["_history_metadata"]({"history": rows}, path)
    else:
        result, metadata = scope["_history_metadata"]({"history": rows}, path)
        assert result is rows and len(metadata["NaN_positions"]) == 57
        assert metadata["nullable_integer_CSV_serialization_count"] == 25
        assert metadata["nullable_integer_CSV_serialization_policy"]["resume_value"] == 720
        assert all(math.isnan(r["row_residual"]) for r in rows)


def test_endpoint_equations_and_metadata_are_only_scoped_literal_changes():
    old = (REPO / "src/cora_node_reference_certificate_v2.py").read_text()
    old_tree = ast.parse(old)
    old_function = next(n for n in old_tree.body if isinstance(n, ast.FunctionDef) and n.name == "_endpoints")
    expected = ast.get_source_segment(old, old_function)
    for before, after in [("(70, 1441)", "(120, 3710)"), ("1433", "3703"), ("(7, 1434)", "(6, 3704)"),
            ("1 / 70", "1 / 120"), ("theta, .0001", "theta, .001"), ("(2708, 32)", "(3327, 8)"),
            ("(70, 32)", "(120, 8)"), ('group["lr"] == .05', 'group["lr"] == .01')]:
        expected = expected.replace(before, after)
    expected = expected.replace("                    new_NODE_baseline_P_updates=0, prior_completed_baseline_P_updates=25)",
                                "                    prior_completed_baseline_P_updates=25)")
    actual = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == "_endpoints")
    assert ast.dump(ast.parse(expected).body[0], include_attributes=False) == ast.dump(actual, include_attributes=False)
    metadata = next(n for n in old_tree.body if isinstance(n, ast.FunctionDef) and n.name == "_history_metadata")
    new_metadata = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == "_history_metadata")
    assert ast.dump(metadata, include_attributes=False) == ast.dump(new_metadata, include_attributes=False)


@pytest.mark.parametrize("mutation", [None, "receipt_failed", "wrong_seed", "file_drift", "own_origin", "preP0", "bool_schema", "bool_condseed"])
def test_phase2_requires_exact_passed_receipt_and_own_origin(tmp_path, mutation):
    paths = {k: str(tmp_path / k) for k in ("inputs", "hard_assignment", "pre_optimizer_P0")}
    for p in paths.values():
        Path(p).write_text("opaque new publication")
    source, numerical, science = {"source": 1}, {"files": {}}, {"path": "frozen", "sha256": "a" * 64}
    buffers, origin = {"source": {"initializer_origin": {"seed": 5}}}, {"moments": [1], "input_digest": "i"}
    native = {"own": 5}
    saved = dict(schema=1, condensation_seed=5, source=source, numerical_source=numerical,
                 scientific_preregistration=science, source_context=buffers["source"], native_buffers=native, origin=origin)
    prior = dict(passed=True, operation="prepare_initializer", condensation_seed=5,
                 source_assets_spec_science_unchanged=True, genuine_seed5_creation_passed=True,
                 pre_optimizer_P0_certificate_passed=True, source=source, numerical_source=numerical,
                 scientific_preregistration=science, spec_sha256="s" * 64,
                 created_files_sha256={p: sha(p) for p in paths.values()}, own_source_context=buffers["source"],
                 own_native_buffers=native, pre_optimizer_P0=origin)
    if mutation == "receipt_failed":
        prior["passed"] = False
    elif mutation == "wrong_seed":
        prior["condensation_seed"] = 4
    elif mutation == "own_origin":
        prior["pre_optimizer_P0"] = {"moments": [2]}
    elif mutation == "preP0":
        saved = dict(saved, origin={"moments": [2]})
    elif mutation == "bool_schema":
        saved = dict(saved, schema=True)
    elif mutation == "bool_condseed":
        saved = dict(saved, condensation_seed=True)
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps(prior))
    checksum = sha(receipt)
    if mutation == "file_drift":
        Path(paths["hard_assignment"]).write_text("changed")
    spec = dict(source=source, numerical_source=numerical, scientific_preregistration=science,
                creation_paths=paths, operation_outputs={"prepare_initializer": {"5": str(receipt)}})
    scope = extract({"_bind_initializer"}, _native_buffers=lambda b: native,
                    probe=types.SimpleNamespace(_digest=lambda v: v),
                    torch=types.SimpleNamespace(load=lambda *a, **k: saved))
    evidence = {}
    if mutation:
        with pytest.raises(ValueError):
            scope["_bind_initializer"](spec, "s" * 64, checksum, buffers, origin, evidence)
    else:
        scope["_bind_initializer"](spec, "s" * 64, checksum, buffers, origin, evidence)
        assert evidence["initializer_dependency_passed"] is True
        assert evidence["initializer_created_files_sha256"] == prior["created_files_sha256"]


def test_genuine_two_initializers_and_exact_source_fields(tmp_path):
    calls, published = [], {}
    paths = {k: str(tmp_path / k) for k in ("inputs", "hard_assignment", "pre_optimizer_P0")}
    z = types.SimpleNamespace(detach=lambda: types.SimpleNamespace(clone=lambda: "exact original z"))
    h = types.SimpleNamespace(cpu=lambda: "original H CPU")
    buffers = {"h": h, "q": "original Q", "z": z, "transform": "original transform", "source": {"seed": 0}}
    native = {"H": "H", "z": "z", "Q": "Q", "transform": "T", "X": "X", "original_CSR": "S", "dense_original_S": "dense"}
    def count(evidence, key, complete=False):
        k = key + ("_completed" if complete else "_attempts")
        evidence["counts"][k] = evidence["counts"].get(k, 0) + 1
    def feature(*args):
        calls.append(("feature", args)); return "genuine feature5"
    def teacher(*args, **kwargs):
        calls.append(("teacher", args, kwargs)); return "genuine hard5"
    def save(p, v):
        published[p] = v; Path(p).write_text("opaque newly produced")
    scope = extract({"_initialize"}, _count=count, _partition=lambda a: a in ("genuine feature5", "genuine hard5"),
        _save_new=save, feature_kmeans=feature, teacher_aware_kmeans=teacher,
        _cached_parts=lambda seed, b, e: dict(b, source={"seed": seed}),
        _native_buffers=lambda b: native, _origin=lambda b, seed, e: {"current_P0_seed": seed},
        replication=types.SimpleNamespace(_COMMON=tuple(native)),
        probe=types.SimpleNamespace(_runtime_precision_guard=lambda: None, _stop=lambda s: None,
                                   _frozen_transform=lambda t: "exact original frozen fields", _digest=lambda v: v))
    spec = dict(creation_paths=paths, source={}, numerical_source={}, scientific_preregistration={})
    evidence = {"counts": {}, "native_buffers_before": native}
    scope["_initialize"](spec, {}, buffers, evidence, lambda: False)
    assert calls[0] == ("feature", ("original H CPU", 120, 5))
    assert calls[1] == ("teacher", (h, "original Q", 120, 5), {"alpha": .3, "feature_weights": None})
    assert published[paths["inputs"]] == {"z": "exact original z", "transform": "exact original frozen fields", "assignment": "genuine feature5"}
    assert published[paths["hard_assignment"]] == "genuine hard5"
    assert evidence["counts"]["integrated_nested_feature_KMeans_completed"] == 1
    assert evidence["counts"]["initializer_representation_RMS_completed"] == 1
    assert evidence["cached_NODE0_equality_available"] is False


def test_fixed_core_call_once_without_resume_or_students(tmp_path):
    calls = []
    def count(e, k, complete=False):
        e["counts"][k + ("_completed" if complete else "_attempts")] = 1
    folder = Path(SCIENCE["NODE5_exact_core_call"]["kwargs"]["folder"])
    fake = types.SimpleNamespace(exists=lambda: False, __str__=lambda: str(folder))
    scope = extract({"_run_core"}, _count=count,
        optimize_ce_assignment=lambda *a, **k: calls.append((a, k)),
        probe=types.SimpleNamespace(_runtime_precision_guard=lambda: None))
    # Do not examine the real absent cache; use a fake Path wrapper for this exact fixed string.
    class Absent:
        def exists(self): return False
        def __str__(self): return str(folder)
    own_folder = Absent()
    evidence = {"counts": {}}
    stop = lambda: False
    scope["_run_core"]({"z": "own z5", "q": "own Q5", "hard": "own hard5"}, SCIENCE, own_folder, evidence, stop)
    args, kwargs = calls[0]
    assert args == ("own z5", "own Q5", "own hard5") and len(calls) == 1
    expected = dict(SCIENCE["NODE5_exact_core_call"]["kwargs"], folder=own_folder, checkpoint_steps=(0, 25), stop=stop)
    assert kwargs == expected and kwargs["resume_state"] is None
    assert "not observed" in evidence["core_head_and_adjoint_work"]
    assert fake.exists() is False


@pytest.mark.parametrize("kind,function", [("citation_citeseer_certify_existing", "certify_existing"),
    ("citation_citeseer_prepare_initializer", "prepare_initializer"),
    ("citation_citeseer_prepare_node_reference", "prepare_node_reference")])
def test_real_worker_lazy_dispatch_preserves_options_return_and_stop(monkeypatch, kind, function):
    tree = ast.parse(WORKER.read_text())
    dispatch = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "dispatch")
    module = types.ModuleType("src.citeseer_confirmation_source")
    calls = []
    setattr(module, function, lambda **kwargs: calls.append(kwargs) or {"receipt": function})
    monkeypatch.setitem(sys.modules, "src", types.ModuleType("src"))
    monkeypatch.setitem(sys.modules, "src.citeseer_confirmation_source", module)
    scope = {}
    exec(compile(ast.Module([dispatch], type_ignores=[]), str(WORKER), "exec"), scope)
    stop = lambda: False
    options = {"condensation_seed": 5, "spec_sha256": "fixed"}
    assert scope["dispatch"]({"kind": kind, "options": options}, stop) == {"receipt": function}
    assert calls == [dict(options, stop=stop)] and options == {"condensation_seed": 5, "spec_sha256": "fixed"}


def test_cleanup_preserves_primary_error_and_reports_source_drift():
    primary = RuntimeError("original native failure")
    def drift(*args):
        raise ValueError("frozen bytes changed")
    scope = extract({"_finish"}, _preserve=drift, time=types.SimpleNamespace(monotonic=lambda: 1))
    evidence = {"passed": True}
    returned = scope["_finish"]({}, {}, "spec", "sha", None, False, 0, evidence, primary)
    assert returned is primary and evidence["passed"] is False
    assert evidence["preservation_error"]["message"] == "frozen bytes changed"
    assert evidence["source_assets_spec_science_unchanged"] is False


def test_no_old_public_wrapper_or_forbidden_fit_invocation():
    calls = [ast.unparse(n.func) for n in ast.walk(TREE) if isinstance(n, ast.Call)]
    assert not any(n in calls for n in ("replication._load_source", "replication._reference", "original._reference",
        "citation_search.run_screen", "fit_transform", "get_shared_h", "fit_gcn_diagnostic"))
    assert calls.count("optimize_ce_assignment") == 1
    assert calls.count("feature_kmeans") == 1 and calls.count("teacher_aware_kmeans") == 1


@pytest.mark.parametrize("mutation", [None, "schema_bool", "unknown", "asset_override", "wrong_output", "undeclared_artifact"])
def test_fixed_spec_paths_controls_and_artifact_allowlist(tmp_path, mutation):
    science = copy.deepcopy(SCIENCE)
    science["fixed"]["source_root"] = str(tmp_path / "original")
    science["output_root"] = str(tmp_path / "results/research_loop/BO")
    for job in science["minimal_four_fresh_jobs"]:
        job["outputs"] = [str(tmp_path / "results/research_loop/BO" / f"{job['operation']}{job['condensation_seed']}" / Path(p).name)
                          for p in job["outputs"]]
    (tmp_path / "data/citeseer/processed").mkdir(parents=True)
    (tmp_path / "data/citeseer/processed/data.pt").write_text("opaque sentinel")
    folder = Path(science["fixed"]["source_root"]) / "a2d47970c967/condensation_3"
    folder.mkdir(parents=True)
    proposals = tmp_path / "results/proposals"
    proposals.mkdir(parents=True)
    scope = extract({"_load_spec", "_operation", "_outputs", "_creation_paths"},
                    _science=lambda repo: science, _preserve=lambda *args: None)
    spec = dict(schema=1, fixed=science["fixed"], source={}, numerical_source={}, python_version="frozen",
                files_sha256=science["original_files_sha256"],
                scientific_preregistration=dict(path=str(proposals / scope["SCIENCE"]), sha256=scope["SCIENCE_SHA"]),
                artifacts_sha256={str(tmp_path / "src/citeseer_confirmation_source.py"): "a" * 64,
                    str(tmp_path / "src/research_loop.py"): "b" * 64,
                    str(tmp_path / "tests/test_citeseer_confirmation_source.py"): "c" * 64},
                output_root=science["output_root"], operations=science["minimal_four_fresh_jobs"],
                operation_outputs=scope["_outputs"](science), creation_paths=scope["_creation_paths"](science))
    output = spec["operation_outputs"]["certify_existing"]["3"]
    if mutation == "schema_bool":
        spec["schema"] = True
    elif mutation == "unknown":
        spec["resume"] = True
    elif mutation == "asset_override":
        spec["files_sha256"] = {}
    elif mutation == "wrong_output":
        output = str(tmp_path / "results/research_loop/foreign.json")
    elif mutation == "undeclared_artifact":
        spec["artifacts_sha256"][str(tmp_path / "src/obsolete.py")] = "d" * 64
    path = proposals / "spec.json"
    path.write_text(json.dumps(spec))
    if mutation:
        with pytest.raises(ValueError):
            scope["_load_spec"]("certify_existing", 3, str(path), sha(path), output, tmp_path)
    else:
        result, returned_science, returned_output, own = scope["_load_spec"]("certify_existing", 3, str(path), sha(path), output, tmp_path)
        assert result == spec and returned_science == science and str(returned_output) == output and own == folder


@pytest.mark.parametrize("failure", ["strict_source", "initializer_dependency", "cleanup_publication_hash"])
def test_primary_failure_stops_before_optimizer_and_preserves_evidence(tmp_path, failure):
    calls, receipts = [], []
    spec = dict(source={}, numerical_source={}, python_version="frozen", scientific_preregistration={},
                fixed=dict(source_root="original", source_backend="BA dense_original_S"), files_sha256={}, creation_paths={})
    cuda = types.SimpleNamespace(is_initialized=lambda: False, reset_peak_memory_stats=lambda: None,
        current_device=lambda: 0, get_device_properties=lambda d: types.SimpleNamespace(total_memory=8316977152))
    torch = types.SimpleNamespace(cuda=cuda, get_num_threads=lambda: 4, load=lambda *a, **k: {})
    error = ValueError(failure)
    def source(*args):
        calls.append("source")
        if failure in ("strict_source", "cleanup_publication_hash"):
            raise error
        return {"ghost": {}}
    def dependency(*args):
        calls.append("dependency")
        raise error
    scope = extract({"_run"}, __file__=str(CANDIDATE), torch=torch, time=types.SimpleNamespace(monotonic=lambda: 0),
        platform=types.SimpleNamespace(python_version=lambda: "frozen"), original=types.SimpleNamespace(CANDIDATE={}, REFERENCE="a2"),
        _load_spec=lambda *a: (spec, {}, tmp_path / "report.json", tmp_path / "newbaseline"),
        _load_source=source, _origin=lambda *a: {}, _core_config=lambda *a: {}, _bind_initializer=dependency,
        _run_core=lambda *a: calls.append("FORBIDDEN optimizer"), _endpoints=lambda *a: calls.append("FORBIDDEN endpoints"),
        _finish=lambda *a: a[-1], _write_new=lambda p, e: receipts.append(dict(e)),
        probe=types.SimpleNamespace(_stop=lambda s: None, _native=lambda d: {}, _runtime_precision_guard=lambda: None))
    if failure == "cleanup_publication_hash":
        owned = tmp_path / "new-owned-publication"
        owned.write_text("opaque newly created")
        spec["creation_paths"] = {"inputs": str(owned)}
        scope["_sha"] = lambda p: (_ for _ in ()).throw(OSError("publication hash failed"))
    with pytest.raises(ValueError) as caught:
        scope["_run"]("prepare_node_reference", 5, "spec", "s" * 64, "report", "a" * 64, lambda: False)
    assert caught.value is error
    assert calls == (["source"] if failure in ("strict_source", "cleanup_publication_hash") else ["source", "dependency"])
    assert receipts[0]["passed"] is False and receipts[0]["counts"]["P_update_completed"] == 0
    assert receipts[0]["primary_error"]["message"] == failure
    if failure == "cleanup_publication_hash":
        assert receipts[0]["publication_observation_error"]["message"] == "publication hash failed"
