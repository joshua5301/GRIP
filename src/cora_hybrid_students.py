"""Cora StageBV completion guards around unchanged student fit and serving.

The fixed25 adapter owns native source/origin/prefix qualification. This module
binds its accepted gate to one physical student cache; it performs no P update.
"""
import json
from pathlib import Path

from src import cora_whole_layer_students as old_students
from src import finite_student_experiment as cora_routes
from src import finite_student_probe as probe
from src.citation_source_preflight import _exact, _write_new
from src.evaluation import _input_digest, fit_gcn_diagnostic
from src.io import _fingerprint
from src.student_routes import replay_routes

SCIENCE = "Cora70_half_teacher_CE_whole_layer_fixed25_scientific_stageBV_v1.json"
SCIENCE_SHA = "83f6a4789eb80f3ebb1b8d156e49ea9d499f90bf8768fbe584818440c710db0e"
SEEDS = (4500, 4501, 4502)
_require, _stop, _tensor = probe._require, probe._stop, probe._tensor
_sha, _seal = probe._sha, probe._seal
_student_paths = cora_routes._student_paths
_selected_route_counts = cora_routes._selected_route_counts
_source_unchanged, _validate_inputs = old_students._source_unchanged, old_students._validate_inputs
_student_header, _student_certificate = old_students._student_header, old_students._student_certificate


def _contract(folder, buffers, context, gate_sha, seed):
    _require(type(seed) is int and seed in SEEDS, "Only fresh hybrid25 student seeds4500..4502")
    _require(type(gate_sha) is str and len(gate_sha) == 64 and all(c in "0123456789abcdef" for c in gate_sha), "Require exact native joint25 gate SHA")
    repo = Path(__file__).resolve().parents[1]
    science_path = repo / "results/proposals" / SCIENCE
    _require(_sha(science_path) == SCIENCE_SHA, "Frozen hybrid25 student science changed")
    science = json.loads(science_path.read_text())
    case, origin = buffers["case"], buffers["recipe_origin"]
    _require(_exact(case, science["case"]) and _exact(context["case"], case) and type(context["cells"]) is int
             and context["cells"] == 70 and type(context["assignment_steps"]) is int and context["assignment_steps"] == 25
             and context["scientific_preregistration"] == dict(path=str(science_path), sha256=SCIENCE_SHA)
             and _exact(context["spec"]["candidate"], science["candidate"])
             and context["spec"]["candidate_id"] == science["candidate_id"], "Own hybrid25 case/full25/source-bound context differs")
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
    _require(folder.parent == base and folder.name in ("shared_P0_validation", "node_reference25_validation", "hybrid25_validation"), "Student physical namespace differs")
    gate_path = Path(context["spec"]["certificate_outputs"]["70"])
    _require(gate_path.is_file() and _sha(gate_path) == gate_sha, "Accepted joint native25 gate changed")
    _source_unchanged(context)
    return folder, settings


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
