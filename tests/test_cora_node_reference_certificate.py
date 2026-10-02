"""Focused typed-metadata/readonly guards; AST only, no Torch/source imports."""
import ast
import copy
import csv
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

BASE = Path(__file__).resolve().parents[1]
MODULE = BASE / "src/cora_node_reference_certificate.py"
REPO = next(p for p in Path(__file__).resolve().parents if (p / "src/transforms.py").is_file())


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exact(actual, expected):
    if isinstance(expected, dict):
        return isinstance(actual, dict) and set(actual) == set(expected) and all(exact(actual[k], v) for k, v in expected.items())
    return type(actual) is type(expected) and actual == expected


@pytest.fixture
def exp():
    tree = ast.parse(MODULE.read_text())
    code = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)
                           or isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                           and n.targets[0].id in {"SCIENCE", "SCIENCE_SHA", "BE_SHA", "CUDA_CAPACITY", "_NAN", "_MISSING"}], type_ignores=[])
    base_tree = ast.parse((REPO / "src/cora_node_reference.py").read_text())
    history = next(n for n in base_tree.body if isinstance(n, ast.FunctionDef) and n.name == "_history")
    history_code = ast.Module(body=[history], type_ignores=[])
    checked = []

    def check_files(pins):
        checked.append(dict(pins))
        require(all(Path(p).is_file() and sha(p) == h for p, h in pins.items()), "Immutable bytes differ")

    def scalar(v, label):
        require(type(v) in (float, int) and math.isfinite(v), label)
        return v

    ns = dict(Path=Path, json=json, csv=csv, math=math, platform=platform, time=time, __file__=str(MODULE),
              _require=require, _sha=sha, _exact=exact, _seal=lambda v: hashlib.sha256(json.dumps(v, sort_keys=True, allow_nan=False).encode()).hexdigest(),
              _seed=lambda c: c if type(c) is int and c in (1, 2) else require(False, "Invalid seed"),
              implementation_provenance=lambda: {"git": "current"},
              torch=SimpleNamespace(get_num_threads=lambda: 4),
              probe=SimpleNamespace(_checked_files=check_files, _scalar=scalar, _runtime_precision_guard=lambda: None,
                                    _stop=lambda stop: require(not stop(), "Stopped")))
    exec(compile(history_code, "protected-base-history", "exec"), ns)
    exec(compile(code, str(MODULE), "exec"), ns)
    ns["numerical_source"] = lambda: {"files": "current"}
    return SimpleNamespace(ns=ns, checked=checked)


def history():
    rows = []
    for i in range(26):
        row = dict(step=i, J=.8, inner_grad_max=1e-8, seconds=float(i), inner_converged=True,
                   J_exact=True, status="evaluated" if i == 25 else "update", cg_converged=i < 25,
                   row_residual=float("nan"), column_residual=float("nan"),
                   head_correction_relative=float("nan") if i == 0 else .01,
                   implicit_correction_relative=float("nan") if i in (0, 25) else .02,
                   cg_residual=float("nan") if i == 25 else 1e-9,
                   cg_relative_residual=float("nan") if i == 25 else 1e-8)
        if i < 25:
            row.update(hessian_solver="reduced_direct", hessian_reduced_dimension=490)
        rows.append(row)
    return rows


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: "" if type(v) is float and math.isnan(v) else v for k, v in row.items()} for row in rows)


def test_genuine_legacy_NaN_metadata_is_valid_without_sealing_raw_history(exp, tmp_path):
    rows = history()
    path = tmp_path / "history.csv"
    write_csv(path, rows)
    with pytest.raises(ValueError): exp.ns["_seal"](rows)
    original = [[math.isnan(v) if type(v) is float else False for v in row.values()] for row in rows]
    replay, meta = exp.ns["_history_metadata"](dict(history=rows), path)
    assert replay is rows and meta["full_CSV_resume_metadata_equal"] is True
    assert len(meta["NaN_positions"]) == 57
    assert len(meta["normalized_metadata_sha256"]) == 64
    assert original == [[math.isnan(v) if type(v) is float else False for v in row.values()] for row in rows]


@pytest.mark.parametrize("key,step,value", [
    ("J", 12, float("nan")), ("inner_grad_max", 12, float("inf")),
    ("head_correction_relative", 1, float("nan")), ("implicit_correction_relative", 1, float("nan")),
    ("cg_residual", 1, float("nan")), ("column_residual", 2, float("inf")),
    ("row_residual", 0, 0.), ("row_residual", 0, "nan"), ("row_residual", 0, None),
    ("cg_relative_residual", 25, "NaN"), ("hessian_solver", 25, float("nan")),
    ("extra_computed_scalar", 2, -float("inf"))])
def test_NaN_policy_rejects_wrong_position_type_infinity_or_missing_diagnostic(exp, tmp_path, key, step, value):
    rows = history()
    rows[step][key] = value
    path = tmp_path / "history.csv"
    write_csv(path, rows)
    with pytest.raises(ValueError): exp.ns["_history_metadata"](dict(history=rows), path)


@pytest.mark.parametrize("mutation", ["all_residual_missing", "Hessian_nonterminal_missing", "Hessian_terminal_present",
                                     "CSV_float", "CSV_bool", "CSV_placeholder", "CSV_extra_cell"])
def test_full_CSV_metadata_and_only_two_terminal_missing_fields(exp, tmp_path, mutation):
    rows = history()
    path = tmp_path / "history.csv"
    if mutation == "all_residual_missing":
        for row in rows: row.pop("row_residual")
    elif mutation == "Hessian_nonterminal_missing": rows[3].pop("hessian_solver")
    elif mutation == "Hessian_terminal_present": rows[25]["hessian_solver"] = "invented"
    write_csv(path, rows)
    if mutation.startswith("CSV_"):
        lines = path.read_text().splitlines()
        data = list(csv.reader(lines))
        if mutation == "CSV_float": data[2][data[0].index("head_correction_relative")] = "0.02"
        elif mutation == "CSV_bool": data[2][data[0].index("cg_converged")] = "true"
        elif mutation == "CSV_placeholder": data[2][data[0].index("row_residual")] = "nan"
        elif mutation == "CSV_extra_cell": data[2].append("orphan")
        with path.open("w", newline="") as stream: csv.writer(stream).writerows(data)
    with pytest.raises(ValueError): exp.ns["_history_metadata"](dict(history=rows), path)


@pytest.fixture
def frozen(tmp_path, exp):
    ns = exp.ns
    science = json.loads((REPO / "results/proposals" / ns["SCIENCE"]).read_text())
    science = json.loads(json.dumps(science).replace(str(REPO), str(tmp_path)))
    for group in ("original_files_sha256", "baseline_cache_files_sha256"):
        for p in science[group]:
            path = Path(p); path.parent.mkdir(parents=True, exist_ok=True); path.write_text("opaque generated bytes")
        science[group] = {p: sha(p) for p in science[group]}
    for ref in [*science["parents"], science["original_BB_certificate"]]:
        path = Path(ref["path"]); path.parent.mkdir(parents=True, exist_ok=True); path.write_text("frozen generated parent")
        ref["sha256"] = sha(path)
    science_path = tmp_path / "results/proposals" / ns["SCIENCE"]
    science_path.write_text(json.dumps(science))
    ns["SCIENCE_SHA"] = sha(science_path)
    ns["_science"] = lambda repo: science
    review = tmp_path / "results/proposals/review.json"; review.write_text("review")
    spec = dict(schema=1, fixed=copy.deepcopy(science["fixed"]), case=copy.deepcopy(science["case"]),
        baseline_candidate=copy.deepcopy(science["baseline_candidate"]), source={"git": "current"},
        numerical_source={"files": "current"}, python_version=platform.python_version(),
        files_sha256=dict(science["original_files_sha256"]), baseline_cache_files_sha256=dict(science["baseline_cache_files_sha256"]),
        scientific_preregistration=dict(path=str(science_path), sha256=ns["SCIENCE_SHA"]),
        artifacts_sha256={str(review): sha(review)}, outputs=ns["_outputs"](science))
    ns["__file__"] = str(tmp_path / "src/cora_node_reference_certificate.py")
    path = tmp_path / "results/proposals/spec.json"
    def save(seed=1):
        path.write_text(json.dumps(spec))
        return seed, path, sha(path), Path(spec["outputs"][str(seed)]), tmp_path
    return SimpleNamespace(ns=ns, science=science, spec=spec, path=path, save=save)


@pytest.mark.parametrize("mutation", [None, "cache_pin", "cache_missing", "unknown_control", "source", "oldscience"])
def test_existing_eight_baseline_bytes_and_failed_lineage_pinned_before_readonly_access(frozen, mutation):
    if mutation == "cache_pin": frozen.spec["baseline_cache_files_sha256"][next(iter(frozen.spec["baseline_cache_files_sha256"]))] = "0" * 64
    elif mutation == "cache_missing": Path(next(iter(frozen.spec["baseline_cache_files_sha256"]))).unlink()
    elif mutation == "unknown_control": frozen.spec["resume"] = True
    elif mutation == "source": frozen.spec["source"] = {"git": "old"}
    elif mutation == "oldscience": frozen.spec["scientific_preregistration"]["sha256"] = "0" * 64
    if mutation is None:
        _, science, _, folder = frozen.ns["_load_spec"](*frozen.save())
        assert folder.is_dir() and len(science["baseline_cache_files_sha256"]) == 8
    else:
        with pytest.raises(ValueError): frozen.ns["_load_spec"](*frozen.save())


@pytest.mark.parametrize("failure", [False, True])
def test_public_certificate_has_zero_optimizer_calls_and_never_recondenses(frozen, failure):
    ns = frozen.ns
    args = frozen.save()
    calls = []
    ns["torch"].cuda = SimpleNamespace(is_initialized=lambda: False, reset_peak_memory_stats=lambda: None,
        get_device_properties=lambda device: SimpleNamespace(total_memory=ns["CUDA_CAPACITY"]), current_device=lambda: 0)
    ns["probe"]._native = lambda device: {"native": "fixture"}
    ns["_load_source"] = lambda *a: calls.append("readonly source") or ({}, {}, {})
    ns["_finish"] = lambda spec, science, path, checksum, buffers, native, started, evidence, primary: primary
    def endpoints(buffers, origin, expected, folder, evidence):
        calls.append("readonly endpoints")
        if failure: raise RuntimeError("endpoint failure")
        evidence["counts"].update(certificate_head_gradient_completed=2, certificate_teacher_CE_completed=2, terminal_factor_moments_completed=1)
        evidence["source_assets_spec_science_unchanged"] = True
    ns["_endpoints"] = endpoints
    ns["_write_new"] = lambda path, evidence: path.parent.mkdir(parents=True, exist_ok=True) or path.write_text(json.dumps(evidence))
    if failure:
        with pytest.raises(RuntimeError, match="endpoint failure"): ns["certify"](*args[:4])
    else:
        result = ns["certify"](*args[:4])
        assert result["validation_only"] is True and result["qualified_own_baseline_count"] == 1
    evidence = json.loads(args[3].read_text())
    assert {k:evidence["counts"][k] for k in ("P_update_completed", "optimizer_calls", "head_solves")} == dict(P_update_completed=0, optimizer_calls=0, head_solves=0)
    assert calls == ["readonly source", "readonly endpoints"] and evidence["passed"] is not failure


def test_protected_aliases_and_endpoint_equations_are_not_public_loader_retargeting():
    tree = ast.parse(MODULE.read_text())
    text = MODULE.read_text()
    for name in ("_load_source", "_native_buffers", "_history", "_core_config", "_origin"):
        assert f"baseline.{name}" in text
    imports = [alias.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for alias in n.names]
    assert "optimize_ce_assignment" not in imports
    endpoints = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_endpoints")
    assert "result" not in {n.id for n in ast.walk(endpoints) if isinstance(n, ast.Name)}
    assert "_seal(rows)" not in ast.get_source_segment(text, endpoints)


def test_lazy_readonly_dispatch_preserves_options_and_stop(monkeypatch):
    tree = ast.parse((BASE / "src/research_loop.py").read_text())
    dispatch = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "dispatch")
    calls = []
    monkeypatch.setitem(sys.modules, "src.cora_node_reference_certificate", SimpleNamespace(certify=lambda **kw: calls.append(kw) or "certificate"))
    ns = {}; exec(compile(ast.Module(body=[dispatch], type_ignores=[]), "draft-worker", "exec"), ns)
    stop = lambda: False
    options = dict(condensation_seed=2, spec_path="spec", spec_sha256="sha", output_path="receipt")
    assert ns["dispatch"](dict(kind="citation_cora_node_reference_certify", options=options), stop) == "certificate"
    assert calls == [dict(options, stop=stop)] and "stop" not in options
