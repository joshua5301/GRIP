"""Only the new named integer serialization/namespace contract; AST and opaque CSV."""
import ast
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from types import ModuleType

import pytest

BASE = Path(__file__).resolve().parents[1]
MODULE = BASE / "src/cora_node_reference_certificate_v2.py"
REPO = next(p for p in Path(__file__).resolve().parents if (p / "src/transforms.py").is_file())
SCIENCE_PATH = REPO / "results/proposals/Cora70_NODE_reference_readonly_certificate_scientific_stageBG_v2.json"
SCIENCE_SHA = "3d8b243fb4d9101ab708f325583f735efafde3ff5f1974dc86ad57b71fa2e238"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


@pytest.fixture
def exp():
    tree = ast.parse(MODULE.read_text())
    names = {"SCIENCE", "SCIENCE_SHA", "_NAN", "_MISSING", "_INTEGER_CSV_POLICY"}
    definitions = [node for node in tree.body
                   if isinstance(node, ast.FunctionDef) and node.name in {"_nullable_integer_csv", "_history_metadata"}
                   or isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in names]
    ns = dict(csv=csv, math=math, _require=require,
              _history=lambda resume, path: resume["history"],
              _seal=lambda value: hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest())
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(MODULE), "exec"), ns)
    return ns


def rows_and_csv(path):
    rows = []
    for step in range(26):
        row = dict(step=step, other_integer=step, computed_float=.125, flag=True, status="readonly",
                   row_residual=float("nan"), column_residual=float("nan"),
                   head_correction_relative=float("nan") if step == 0 else .01,
                   implicit_correction_relative=float("nan") if step in (0, 25) else .02,
                   cg_residual=float("nan") if step == 25 else 1e-9,
                   cg_relative_residual=float("nan") if step == 25 else 1e-8)
        if step < 25:
            row.update(hessian_solver="reduced_direct", hessian_reduced_dimension=490)
        rows.append(row)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    disk = [{key: ("" if key not in row or type(row[key]) is float and math.isnan(row[key])
                   else "490.0" if key == "hessian_reduced_dimension" else str(row[key]))
             for key in columns} for row in rows]
    save_csv(path, columns, disk)
    return rows, columns, disk


def save_csv(path, columns, disk):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(disk)


def test_real_pandas_token_full_typed_guard_and_policy_without_mutating_rows(exp, tmp_path):
    assert sha(SCIENCE_PATH) == SCIENCE_SHA
    science = json.loads(SCIENCE_PATH.read_text())
    real_path = next(Path(p) for p in science["baseline_cache_files_sha256"]
                     if "/condensation_1/" in p and p.endswith("optimization.csv"))
    assert sha(real_path) == science["baseline_cache_files_sha256"][str(real_path)]
    with real_path.open(newline="") as stream:
        actual_tokens = [row["hessian_reduced_dimension"] for row in csv.DictReader(stream)]
    assert actual_tokens == ["490.0"] * 25 + [""]
    path = tmp_path / "metadata.csv"
    rows, columns, disk = rows_and_csv(path)
    for step, token in enumerate(actual_tokens):
        disk[step]["hessian_reduced_dimension"] = token
    save_csv(path, columns, disk)
    before = sha(path)
    replay, metadata = exp["_history_metadata"]({"history": rows}, path)
    assert replay is rows and sha(path) == before
    assert len(metadata["NaN_positions"]) == 57
    assert all(type(row["hessian_reduced_dimension"]) is int for row in rows[:25])
    assert all("hessian_reduced_dimension" not in row for row in rows[25:])
    assert metadata["nullable_integer_CSV_exact_serialization_passed"] is True
    assert metadata["nullable_integer_CSV_serialization_count"] == 25
    assert metadata["nullable_integer_CSV_serialization_policy"] == science["history_metadata_policy"]["nullable_integer_CSV_serialization"]
    assert exp["_nullable_integer_csv"](0, 490, "490.0")
    assert exp["_nullable_integer_csv"](24, 490, "490.0")
    assert not exp["_nullable_integer_csv"](25, 490, "490.0")
    assert not exp["_nullable_integer_csv"](True, 490, "490.0")


@pytest.mark.parametrize("mutation", ["native_float", "native_bool", "native_missing", "native_wrong_value",
                                      "integer_text", "extra_decimal", "other_integer_decimal", "terminal_present"])
def test_full_metadata_rejects_wrong_types_tokens_positions_and_global_integer_relaxation(exp, tmp_path, mutation):
    path = tmp_path / "metadata.csv"
    rows, columns, disk = rows_and_csv(path)
    if mutation == "native_float": rows[1]["hessian_reduced_dimension"] = 490.0
    elif mutation == "native_bool": rows[1]["hessian_reduced_dimension"] = True
    elif mutation == "native_missing": rows[1].pop("hessian_reduced_dimension")
    elif mutation == "native_wrong_value": rows[1]["hessian_reduced_dimension"] = 489
    elif mutation == "integer_text": disk[1]["hessian_reduced_dimension"] = "490"
    elif mutation == "extra_decimal": disk[1]["hessian_reduced_dimension"] = "490.00"
    elif mutation == "other_integer_decimal": disk[1]["other_integer"] = "1.0"
    elif mutation == "terminal_present":
        rows[25]["hessian_reduced_dimension"] = 490
        disk[25]["hessian_reduced_dimension"] = "490.0"
    save_csv(path, columns, disk)
    before = sha(path)
    with pytest.raises(ValueError):
        exp["_history_metadata"]({"history": rows}, path)
    assert sha(path) == before


def test_new_namespace_lazy_dispatch_preserves_options_callback_and_result(monkeypatch, exp):
    assert exp["SCIENCE"] == SCIENCE_PATH.name and exp["SCIENCE_SHA"] == SCIENCE_SHA
    source = MODULE.read_text()
    assert 'value["files"].get("cora_node_reference_certificate_v2.py")' in source
    assert 'value["files"].get("cora_node_reference_certificate.py")' not in source
    dispatch = next(node for node in ast.parse((BASE / "src/research_loop.py").read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == "dispatch")
    module = ModuleType("src.cora_node_reference_certificate_v2")
    calls, expected = [], {"opaque": "receipt"}
    module.certify = lambda **kwargs: calls.append(kwargs) or expected
    monkeypatch.setitem(sys.modules, module.__name__, module)
    ns = {}
    exec(compile(ast.Module(body=[dispatch], type_ignores=[]), "owned-lazy-dispatch", "exec"), ns)
    callback = lambda: False
    options = dict(condensation_seed=2, spec_path="frozen-spec", spec_sha256="a" * 64, output_path="new-evidence")
    job = dict(kind="citation_cora_node_reference_certify_v2", options=dict(options))
    assert ns["dispatch"](job, callback) is expected
    assert calls == [dict(options, stop=callback)] and job["options"] == options
