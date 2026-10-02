"""Cora StageBH owncond completion guards around unchanged student fit and serving.

The fixed25 adapter owns native source/origin/prefix qualification. This module
binds its accepted gate to one physical student cache; it performs no P update.
"""
import json
import platform
from pathlib import Path

import torch

from src import citation_gradient_probe as ay
from src import cora_gradient_students as baseline_students
from src import finite_student_probe as probe
from src.citation_source_preflight import _exact, _write_new
from src.evaluation import _input_digest, fit_gcn_diagnostic
from src.io import _fingerprint
from src.research_loop import implementation_provenance
from src.student_routes import replay_routes

SCIENCE = "Cora70_CE_gradient_alignment_fixed25_replication_scientific_stageBH_v1.json"
SCIENCE_SHA = "6d84f4f498866aa40f244b3fdc7277e72d75d843e1b83a18774d6c36e85bb5ec"
SEEDS = (4200, 4201, 4202)
_require, _stop, _tensor = probe._require, probe._stop, probe._tensor
_sha, _seal = probe._sha, probe._seal
_student_paths = baseline_students._student_paths
_selected_route_counts = baseline_students._selected_route_counts
_validate_inputs, _student_certificate = baseline_students._validate_inputs, baseline_students._student_certificate


def _source_unchanged(context):
    _require(implementation_provenance() == context["implementation"]
             and ay.numerical_source() == context["numerical_source"]
             and platform.python_version() == context["spec"]["python_version"]
             and torch.get_num_threads() == 4, "Student source/Git/versions/Python/threads changed")


def _contract(folder, buffers, context, gate_sha, seed):
    _require(type(seed) is int and seed in SEEDS, "Only fixed fresh student seeds4200..4202")
    _require(type(gate_sha) is str and len(gate_sha) == 64
             and all(c in "0123456789abcdef" for c in gate_sha), "Require exact native25 gate SHA")
    repo = Path(__file__).resolve().parents[1]
    science_path = repo / "results/proposals" / SCIENCE
    _require(_sha(science_path) == SCIENCE_SHA, "Frozen StageBH student science changed")
    science = json.loads(science_path.read_text())
    case = buffers["case"]
    _require(type(case.get("cells")) is int and case["cells"] == 70 and type(case.get("condensation_seed")) is int and case["condensation_seed"] in (1, 2), "Invalid Cora budget")
    condensation_seed = case["condensation_seed"]
    expected = dict(science["case"], condensation_seed=condensation_seed,
                    hard_path=Path(science["hard_assignments"][str(condensation_seed)]["path"]).name)
    _require(_exact(case, expected) and _exact(context["case"], case)
             and context["cells"] == case["cells"] and type(context["condensation_seed"]) is int
             and context["condensation_seed"] == condensation_seed and type(context["assignment_steps"]) is int
             and context["assignment_steps"] == 25
             and context["scientific_preregistration"] == dict(path=str(science_path), sha256=SCIENCE_SHA),
             "Own Cora case/full25 context differs")
    origin = buffers["recipe_origin"]
    recipe_path = Path(case["source_root"]) / f"student_recipe_{case['recipe_id']}.json"
    _require(origin["path"] == str(recipe_path) and origin["sha256"] == _sha(recipe_path)
             and origin["recipe_id"] == case["recipe_id"]
             and _exact(origin["contents"], case["recipe"])
             and _exact(json.loads(recipe_path.read_text()), origin["contents"])
             and _fingerprint(origin["contents"]) == origin["recipe_id"]
             and _exact(context["student_recipe_origin"], origin), "Own original student recipe changed")
    settings = dict(origin["contents"])
    _require(type(settings.pop("input_scale")) is float and case["recipe"]["input_scale"] == 1.,
             "Require own identity input scale")
    _require(science["student_seeds"] == list(SEEDS)
             and settings["epochs"] == 600 and settings["eval_every"] == 1
             and settings["hidden"] == 256 and settings["dropout"] == 0.
             and settings["lr_schedule"] == "constant" and settings["initialization"] == "geom_uniform",
             "Frozen Cora student controls differ")
    folder = Path(folder).resolve()
    base = Path(context["spec"]["output_root"]) / f"condensation_{condensation_seed}" / context["spec"]["candidate_ids"][str(condensation_seed)]
    _require(folder.parent == base and folder.name in
             ("shared_P0_validation", "node_reference25_validation", "gradient25_validation"),
             "Student physical folder differs from fixed own-case namespace")
    gate_path = Path(context["spec"]["certificate_outputs"][str(condensation_seed)])
    _require(gate_path.is_file() and _sha(gate_path) == gate_sha,
             "Accepted own native25 gate file changed")
    _source_unchanged(context)
    return folder, settings




def _student_header(folder, buffers, context, gate_sha, digest, seed, settings):
    case = buffers["case"]
    return dict(schema=1, source=context["implementation"], numerical_source=context["numerical_source"],
        native_gate_sha256=gate_sha, input_digest=digest, seed=seed, settings=settings,
        cells=case["cells"], condensation_seed=case["condensation_seed"], recipe_origin=buffers["recipe_origin"],
        uniform_supplied_weights=True, synthetic_adjacency=None,
        shared_P0_origin_reference_id=case["reference_id"], physical_source_root=case["source_root"],
        source_context_digest=_seal(context), physical_folder=str(folder),
        selection="first strict maximum GCN validation accuracy", test_enabled=False)




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
