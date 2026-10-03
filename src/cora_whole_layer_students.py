"""Cora StageBD completion guards around unchanged student fit and serving.

The fixed25 adapter owns native source/origin/prefix qualification. This module
binds its accepted gate to one physical student cache; it performs no P update.
"""
import csv
import json
import math
import platform
from pathlib import Path

import torch

from src import cora_whole_layer_probe as bq
from src import finite_student_experiment as cora_routes
from src import finite_student_probe as probe
from src.citation_source_preflight import _exact, _write_new
from src.evaluation import _input_digest, fit_gcn_diagnostic
from src.io import _fingerprint
from src.research_loop import implementation_provenance
from src.student_routes import replay_routes

SCIENCE = "Cora70_two_whole_layer_fixed25_matched_NODE_scientific_stageBS_v1.json"
SCIENCE_SHA = "b894eab2fa03ac656a5c2b46fb738ff0277ab67c21d1713c63a7f2b964e08508"
SEEDS = (4400, 4401, 4402)
_require, _stop, _tensor = probe._require, probe._stop, probe._tensor
_sha, _seal = probe._sha, probe._seal
_student_paths = cora_routes._student_paths
_selected_route_counts = cora_routes._selected_route_counts


def _source_unchanged(context):
    _require(implementation_provenance() == context["implementation"]
             and bq.numerical_source() == context["numerical_source"]
             and platform.python_version() == context["spec"]["python_version"]
             and torch.get_num_threads() == 4, "Student source/Git/versions/Python/threads changed")


def _contract(folder, buffers, context, gate_sha, seed):
    _require(type(seed) is int and seed in SEEDS, "Only fresh BR student seeds4400..4402")
    _require(type(gate_sha) is str and len(gate_sha) == 64 and all(c in "0123456789abcdef" for c in gate_sha), "Require exact native joint25 gate SHA")
    repo = Path(__file__).resolve().parents[1]
    science_path = repo / "results/proposals" / SCIENCE
    _require(_sha(science_path) == SCIENCE_SHA, "Frozen BR student science changed")
    science = json.loads(science_path.read_text())
    case, origin = buffers["case"], buffers["recipe_origin"]
    _require(_exact(case, science["case"]) and _exact(context["case"], case) and type(context["cells"]) is int
             and context["cells"] == 70 and type(context["assignment_steps"]) is int and context["assignment_steps"] == 25
             and context["scientific_preregistration"] == dict(path=str(science_path), sha256=SCIENCE_SHA)
             and _exact(context["spec"]["candidate"], science["candidate"])
             and context["spec"]["candidate_id"] == science["candidate_id"], "Own BR case/full25/source-bound context differs")
    recipe_path = Path(case["source_root"]) / f"student_recipe_{case['recipe_id']}.json"
    _require(origin["path"] == str(recipe_path) and origin["sha256"] == _sha(recipe_path)
             and origin["recipe_id"] == case["recipe_id"] and _exact(origin["contents"], case["recipe"])
             and _exact(json.loads(recipe_path.read_text()), origin["contents"])
             and _fingerprint(origin["contents"]) == origin["recipe_id"]
             and _exact(context["student_recipe_origin"], origin), "Own original student recipe changed")
    settings = dict(origin["contents"])
    _require(type(settings.pop("input_scale")) is float and case["recipe"]["input_scale"] == 1., "Require identity scale")
    _require(science["student_seeds"] == list(SEEDS) and _exact(settings, {k:v for k,v in science["student_protocol"]["recipe"].items() if k != "input_scale"}), "Frozen600epoch student settings differ")
    folder = Path(folder).resolve()
    base = Path(science["candidate_folder"]).parent
    _require(folder.parent == base and folder.name in ("shared_P0_validation", "node_reference25_validation", "whole_layer25_validation"), "Student physical namespace differs")
    gate_path = Path(context["spec"]["certificate_outputs"]["70"])
    _require(gate_path.is_file() and _sha(gate_path) == gate_sha, "Accepted joint native25 gate changed")
    _source_unchanged(context)
    return folder, settings


def _validate_inputs(inputs, buffers):
    cx, cy, weights = inputs
    cells = buffers["case"]["cells"]
    _tensor(cx, (cells, 1433), torch.float32, "Invalid Cora student representative X")
    _tensor(cy, (cells, 7), torch.float32, "Invalid Cora student representative Q")
    _tensor(weights, (cells,), torch.float64, "Invalid supplied Cora uniform weights")
    _require(torch.equal(weights, torch.full_like(weights, 1 / cells)), "Supplied student weights must be exact uniform")


def _student_header(folder, buffers, context, gate_sha, digest, seed, settings):
    case = buffers["case"]
    return dict(schema=1, source=context["implementation"], numerical_source=context["numerical_source"],
        native_gate_sha256=gate_sha, input_digest=digest, seed=seed, settings=settings,
        cells=case["cells"], recipe_origin=buffers["recipe_origin"],
        uniform_supplied_weights=True, synthetic_adjacency=None,
        shared_P0_origin_reference_id=case["reference_id"], physical_source_root=case["source_root"],
        source_context_digest=_seal(context), physical_folder=str(folder),
        selection="first strict maximum GCN validation accuracy", test_enabled=False)


def _student_certificate(paths, header, buffers, stop):
    settings = header["settings"]
    _require(all(paths[k].is_file() for k in ("metadata", "epochs", "selected", "routes")),
             "Partial student cache preserved; no retraining fallback")
    metadata = json.loads(paths["metadata"].read_text())
    routes = json.loads(paths["routes"].read_text())
    selected = torch.load(paths["selected"], map_location="cpu", weights_only=False)
    _require(isinstance(metadata, dict) and isinstance(metadata.get("recipe"), dict)
             and isinstance(metadata.get("result"), dict) and isinstance(selected, dict)
             and isinstance(routes, dict) and isinstance(routes.get("recipe"), dict)
             and isinstance(routes.get("result"), dict), "Malformed completed student files")
    expected = dict(version=2, seed=header["seed"], settings=settings, layers=2,
        optimizer="Adam; constant lr", weighting="normalized supplied mass",
        selection="first maximum validation accuracy", test_enabled=False,
        test_evaluation="selected weights once", torch_version=str(torch.__version__), input_digest=header["input_digest"])
    _require(_exact(metadata["recipe"], expected) and metadata.get("fingerprint") == _fingerprint(expected)
             and selected.get("fingerprint") == metadata["fingerprint"], "Student recipe/input/selected fingerprint changed")
    with paths["epochs"].open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    _require(len(rows) == 600, "Incomplete600epoch student history")
    best, winner = -math.inf, None
    for epoch, row in enumerate(rows, 1):
        try:
            accuracy = float(row["val_acc"])
            valid = row["epoch"] == str(epoch) and all(math.isfinite(float(v)) for v in row.values())
            valid = valid and 0 <= accuracy <= 100
        except (KeyError, TypeError, ValueError):
            valid = False
        _require(valid, "Malformed ordered finite student history")
        if accuracy > best:
            best, winner = accuracy, epoch
    _require(type(selected.get("epoch")) is int and selected["epoch"] == winner
             and type(metadata["result"].get("epoch")) is int and metadata["result"]["epoch"] == winner
             and metadata["result"].get("val_acc") == best, "Selected state is not first strict GCN validation maximum")
    recipe = routes["recipe"]
    _require(routes.get("fingerprint") == _fingerprint(recipe)
             and recipe.get("source_fingerprint") == metadata["fingerprint"]
             and type(recipe.get("epoch")) is int and recipe["epoch"] == winner
             and type(recipe.get("seed")) is int and recipe["seed"] == header["seed"]
             and _exact(recipe.get("settings"), settings) and recipe.get("test_enabled") is False,
             "Same selected-state route cache provenance changed")
    replay_routes(paths["selected"], buffers["graph"], buffers["h"], {"val": buffers["val"]}, settings,
                  paths["routes"], seed=header["seed"], stop=stop)
    counts, actual = _selected_route_counts(selected, buffers, stop)
    _require(actual["gcn_val_acc"] == best == metadata["result"]["val_acc"], "Selected-state actual GCN correct count differs")
    for route in ("gcn", "mlp"):
        _require(type(counts[route]) is int and 0 <= counts[route] <= 500
                 and routes["result"].get(f"{route}_val_acc") == actual[f"{route}_val_acc"],
                 "Actual same-state route correct count/scalar changed")
        value = routes["result"].get(f"{route}_val_ce")
        _require(type(value) in (int, float) and math.isfinite(value)
                 and math.isclose(value, actual[f"{route}_val_ce"], abs_tol=2e-6, rel_tol=2e-5),
                 "Actual same-state route CE changed")
    return dict(schema=1, header=header,
        files_sha256={k: _sha(paths[k]) for k in ("metadata", "epochs", "selected", "routes")},
        selected_epoch=winner, route_correct_counts=counts, validation_nodes=500)


def evaluate_student(folder, inputs, buffers, context, gate_sha, seed, stop):
    """One own-recipe fit, or exact complete-cache reuse; no logical arm key."""
    _require(callable(stop), "Require stop callback")
    _stop(stop)
    folder, settings = _contract(folder, buffers, context, gate_sha, seed)
    _validate_inputs(inputs, buffers)
    cx, cy, weights = inputs
    digest = _input_digest(cx, cy, weights, buffers["graph"], buffers["q"],
                           {"train": buffers["train"], "val": buffers["val"]}, None, stop)
    header = _student_header(folder, buffers, context, gate_sha, digest, seed, settings)
    paths = _student_paths(folder, seed)
    prefix = list(folder.glob(f"seed_{seed}*"))
    cached = paths["certificate"].is_file()
    if prefix:
        _require(cached, "Partial/score-only student cache preserved; no fallback")
        saved = json.loads(paths["certificate"].read_text())
        _require(isinstance(saved, dict) and type(saved.get("schema")) is int and saved["schema"] == 1
                 and _exact(saved.get("header"), header) and isinstance(saved.get("files_sha256"), dict)
                 and set(saved["files_sha256"]) == {"metadata", "epochs", "selected", "routes"},
                 "Student completion/source/recipe header changed")
        _require(set(prefix) == set(paths.values()) and all(paths[k].is_file()
                 and _sha(paths[k]) == saved["files_sha256"][k] for k in saved["files_sha256"]),
                 "Student physical files changed or orphan files exist")
        _require(_exact(_student_certificate(paths, header, buffers, stop), saved), "Student completion certificate changed")
    else:
        _stop(stop)
        counts = buffers["counts"]
        counts["final_student_fit_attempted"] = counts.get("final_student_fit_attempted", 0) + 1
        fit_gcn_diagnostic(cx, cy, weights, buffers["graph"], buffers["q"],
            {"train": buffers["train"], "val": buffers["val"]}, seed=seed, folder=folder,
            training_adjacency=None, stop=stop, **settings)
        counts["final_student_fit_completed"] = counts.get("final_student_fit_completed", 0) + 1
        replay_routes(paths["selected"], buffers["graph"], buffers["h"], {"val": buffers["val"]}, settings,
                      paths["routes"], seed=seed, stop=stop)
        saved = _student_certificate(paths, header, buffers, stop)
        _stop(stop)
        _source_unchanged(context)
        _write_new(paths["certificate"], saved)
    _source_unchanged(context)
    result = json.loads(paths["routes"].read_text())["result"]
    return dict(seed=seed, selected_epoch=saved["selected_epoch"], input_digest=digest,
        gcn_correct=saved["route_correct_counts"]["gcn"], mlp_correct=saved["route_correct_counts"]["mlp"],
        result=result, cached=cached, actual_student_fits=0 if cached else 1,
        physical_route_outputs=0 if cached else 2, diagnostic_route_forwards=2,
        student_completion_path=str(paths["certificate"].resolve()), student_completion_sha256=_sha(paths["certificate"]))
