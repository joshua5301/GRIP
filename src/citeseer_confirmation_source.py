"""Fixed Citeseer120 source345 / genuinely new initializer5 / NODE5 prerequisite.

Only seed5's two partitions and one legacy NODE trajectory may be created.
Existing source, RMS, teacher/map/Phi and seed0..4 caches are immutable.
"""
import csv
import json
import math
import os
import platform
import time
import uuid
from pathlib import Path

import torch

from src import citation_gradient_replication as replication
from src import citation_search
from src import citation_source_certificate as original
from src import cora_node_reference as baseline
from src import finite_student_probe as probe
from src import source_linear_assignment as source_helper
from src.citation_source_preflight import _exact, _write_new
from src.evaluation import _input_digest
from src.initialization import feature_kmeans
from src.io import _fingerprint, cpu_state
from src.low_rank_assignment import LowRankLogits, LowRankMoments, initialize_factors
from src.moments import augmented, decode_moments, make_material
from src.partition_initialization import teacher_aware_kmeans
from src.research_loop import implementation_provenance
from src.soft_ce_partition import head_gradient, optimize_ce_assignment, outer_value_gradient
from src.sweep_utils import representative
from src.target_refinement import training_refined_targets

SCIENCE = "Citeseer120_source345_and_NODE5_prerequisite_scientific_stageBO_v2.json"
SCIENCE_SHA = "b8c38966ce6cf3cbea324f37e3c1f6724700ab49cf8f5be8200abdaee77a5eab"
CUDA_CAPACITY = 8316977152
_require, _sha, _seal = probe._require, probe._sha, probe._seal
_count, _history, _core_config = baseline._count, baseline._history, baseline._core_config
_common_certificate = replication._common_certificate
_NAN = {"row_residual": tuple(range(26)), "column_residual": tuple(range(26)),
        "head_correction_relative": (0,), "implicit_correction_relative": (0, 25),
        "cg_residual": (25,), "cg_relative_residual": (25,)}
_MISSING = {"hessian_solver": (25,), "hessian_reduced_dimension": (25,)}
_SPEC_FIELDS = {"schema", "fixed", "source", "numerical_source", "python_version", "files_sha256",
                "scientific_preregistration", "artifacts_sha256", "output_root", "operations",
                "operation_outputs", "creation_paths"}


def numerical_source():
    value = probe.numerical_source()
    _require(value["files"].get("citeseer_confirmation_source.py") == _sha(__file__), "Own numerical source absent/changed")
    return value


def _science(repo):
    path = repo / "results/proposals" / SCIENCE
    probe._checked_files({str(path): SCIENCE_SHA})
    return json.loads(path.read_text())


def _paths(repo):
    return [Path(p) for p in _science(repo)["original_files_sha256"]]


def _operation(operation, seed):
    allowed = {"certify_existing": (3, 4), "prepare_initializer": (5,), "prepare_node_reference": (5,)}
    _require(operation in allowed and type(seed) is int and seed in allowed[operation], "Wrong fixed operation/seed")
    return seed


def _outputs(science):
    outputs = {key: {} for key in ("certify_existing", "prepare_initializer", "prepare_node_reference")}
    for job in science["minimal_four_fresh_jobs"]:
        outputs[job["operation"]][str(job["condensation_seed"])] = job["outputs"][-1]
    return outputs


def _creation_paths(science):
    paths = science["minimal_four_fresh_jobs"][2]["outputs"][:3]
    return dict(zip(("inputs", "hard_assignment", "pre_optimizer_P0"), paths, strict=True))


def _source_unchanged(spec):
    _require(implementation_provenance() == spec["source"] and numerical_source() == spec["numerical_source"]
             and platform.python_version() == spec["python_version"] and torch.get_num_threads() == 4,
             "Source/Git/numerical versions/Python/threads changed")


def _reference_pins(value):
    pins = {}
    if isinstance(value, dict):
        if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
            pins[value["path"]] = value["sha256"]
        for item in value.values():
            pins.update(_reference_pins(item))
    elif isinstance(value, list):
        for item in value:
            pins.update(_reference_pins(item))
    return pins


def _preserve(spec, path, checksum, science):
    _require(_sha(path) == checksum, "Base spec changed")
    for key in ("files_sha256", "artifacts_sha256"):
        probe._checked_files(spec[key])
    probe._checked_files(_reference_pins(science))
    probe._checked_files({spec["scientific_preregistration"]["path"]: SCIENCE_SHA})
    _source_unchanged(spec)


def _load_spec(operation, seed, path, checksum, output, repo):
    _operation(operation, seed)
    path, output = Path(path).resolve(), Path(output).resolve()
    _require(type(checksum) is str and len(checksum) == 64 and path.is_relative_to(repo / "results/proposals")
             and path.is_file() and _sha(path) == checksum, "Require frozen base spec path/SHA")
    science, spec = _science(repo), json.loads(path.read_text())
    _require(isinstance(spec, dict) and set(spec) == _SPEC_FIELDS and type(spec["schema"]) is int and spec["schema"] == 1
             and _exact(spec["fixed"], science["fixed"]) and _exact(spec["operations"], science["minimal_four_fresh_jobs"])
             and spec["files_sha256"] == science["original_files_sha256"]
             and spec["output_root"] == science["output_root"] and spec["operation_outputs"] == _outputs(science)
             and spec["creation_paths"] == _creation_paths(science), "Changed/unknown fixed prerequisite controls")
    _require(spec["scientific_preregistration"] == dict(path=str(repo / "results/proposals" / SCIENCE), sha256=SCIENCE_SHA)
             and str(output) == spec["operation_outputs"][operation][str(seed)] and not output.exists(), "Wrong/occupied evidence output")
    owned = {repo / "src/citeseer_confirmation_source.py", repo / "src/research_loop.py",
             repo / "tests/test_citeseer_confirmation_source.py"}
    _require(isinstance(spec["artifacts_sha256"], dict) and spec["artifacts_sha256"]
             and all(Path(p).is_absolute() and (Path(p).resolve().is_relative_to(repo / "results")
                     or Path(p).resolve() in owned) for p in spec["artifacts_sha256"]), "Wrong reviewed artifact namespace")
    _require((repo / "data/citeseer/processed/data.pt").is_file(), "Existing dataset required")
    folder = Path(spec["fixed"]["source_root"]) / spec["fixed"]["reference_id"] / f"condensation_{seed}"
    if operation == "certify_existing":
        _require(folder.is_dir(), "Existing own baseline required")
    else:
        _require(not folder.exists(), "Absent ownNODE5 folder required; no retry/resume")
    if operation == "prepare_initializer":
        _require(all(not Path(p).exists() for p in spec["creation_paths"].values()), "Absent own5 publications required")
    _preserve(spec, path, checksum, science)
    return spec, science, output, folder


def _native_buffers(buffers):
    return probe._digest(dict(H=buffers["h"], z=buffers["z"], Q=buffers["q"], hard=buffers["hard"],
        transform=probe._frozen_transform(buffers["transform"]), X=buffers["graph"]["x"],
        original_CSR=buffers["graph"]["adj"], dense_original_S=buffers["dense"]))


def _cached_parts(seed, shared, evidence):
    _count(evidence, "cached_source")
    parts = source_helper.cached_source(shared["root"], shared["ghost"], seed,
                                       shared["fresh_h"], shared["q"], shared["config"])
    _count(evidence, "cached_source", True)
    h, z, transform, hard, _, source = parts
    _require(z.shape == (3327, 3703) and z.dtype == torch.float64 and h.dtype == torch.float32
             and q_shape(shared["q"]) and hard.shape == (3327,) and hard.dtype == torch.int64
             and bool((hard >= 0).all()) and len(torch.unique(hard)) == 120 and int(hard.max()) + 1 == 120,
             "Malformed own source/partition")
    expected_hard = f"assignment_{_fingerprint(dict(mode='teacher_joint',alpha=.3,T=1.,seed=seed))}.pt"
    _require(source["initializer_origin"] == dict(mode="teacher_joint", alpha=.3, T=1., seed=seed)
             and expected_hard in source["assets"], "Own initializer identity/path differs")
    return dict(shared, h=h, z=z, transform=transform, hard=hard, source=source)


def q_shape(q):
    return q.shape == (3327, 6) and q.dtype == torch.float64 and bool(torch.isfinite(q).all())


def _load_source(seed, spec, science, evidence, stop):
    repo, root = Path(__file__).resolve().parents[1], Path(spec["fixed"]["source_root"])
    _count(evidence, "dataset_preparation")
    graph, train, val, test, fresh_h = citation_search._prepare_dataset("citeseer", str(repo / "data"), "cuda", "row")
    _count(evidence, "dataset_preparation", True)
    evidence["counts"]["integrated_dataset_H_products_completed"] = 2
    _require(graph["x"].shape == (3327, 3703) and graph["adj"].shape == (3327, 3327)
             and torch.is_tensor(val[1]) and val[1].dtype == torch.bool and val[1].shape == (3327,)
             and int(val[1].sum()) == 500, "Original graph/500-node validation mask differs")
    config = citation_search._legacy_teacher_config("citeseer", .036, graph, train, val, test, "row")
    row = dict(source_root=str(root), root="2e2127c2bc3f", ratio=.036)
    _require(config == original._config(row), "Original graph/config differs")
    _require(json.loads((root / original.REFERENCE / "candidate.json").read_text()) == original.CANDIDATE,
             "Original NODE candidate differs")
    recipe_ref = spec["fixed"]["student_recipe"]
    recipe = json.loads(Path(recipe_ref["path"]).read_text())
    _require(_fingerprint(recipe) == "5bde67b7c172" and recipe == dict(epochs=1000, eval_every=1, hidden=256,
             dropout=0., lr=.001, weight_decay=.0005, lr_schedule="constant", initialization="geom_uniform", input_scale=1.),
             "Original recipe/identity scaling differs")
    _count(evidence, "teacher_cache_load")
    teacher = torch.load(root / "teacher.pt", map_location="cuda", weights_only=False)
    _count(evidence, "teacher_cache_load", True)
    logits = probe._tensor(teacher.get("logits") if isinstance(teacher, dict) else None,
                           (3327, 6), torch.float64, "Malformed original teacher")
    q = training_refined_targets(logits, 1., graph["y"], train, 0)
    ghost = source_helper.candidate_controls(dict(original.CANDIDATE, method="source_linear",
               assignment_coordinates="raw_rms", source_linear_schema=1))
    shared = dict(root=root, graph=graph, train=train, val=val[1], fresh_h=fresh_h, config=config, q=q, ghost=ghost)
    buffers = _cached_parts(seed, shared, evidence)
    probe._stop(stop)
    _count(evidence, "dense_S_materialization")
    buffers["dense"] = probe._frozen_original_dense_S(graph["adj"])
    _count(evidence, "dense_S_materialization", True)
    at_ref = science["refs"]["AT_actual_common_source_certificate"]
    at = json.loads(Path(at_ref["path"]).read_text())
    certified = at["roots"]["2e2127c2bc3f"]
    _require(at.get("passed") is True and certified.get("source_P0_linear_certificate_passed") is True,
             "Original AT source unqualified")
    common = _common_certificate(dict(buffers, transform=probe._frozen_transform(buffers["transform"]),
               x=graph["x"], S=buffers["dense"], original_S=graph["adj"]), certified["native_buffers"])
    evidence.update(common_AT_native_buffers_exact_excluding_hard=True, common_source_buffers=common,
                    source_context=buffers["source"], student_recipe_origin=dict(**recipe_ref, contents=recipe),
                    native_buffers_before=_native_buffers(buffers), source_reference_passed=True)
    return buffers


def _origin(buffers, seed, evidence):
    material = make_material(buffers["z"], buffers["q"])
    values = []
    for _ in range(2):
        _count(evidence, "native_factor_factory")
        values.append(initialize_factors(buffers["hard"], 120, 8, seed))
        _count(evidence, "native_factor_factory", True)
    _require(all(torch.equal(a, b) for a, b in zip(values[0], values[1], strict=True)), "Current paired U0/V0 differ")
    u, v = values[0]
    _count(evidence, "native_P0_logits")
    probability = LowRankLogits.apply(u, v, buffers["hard"], .05, 4096).double().softmax(1).detach()
    _count(evidence, "native_P0_logits", True)
    _count(evidence, "native_P0_moments")
    moments = LowRankMoments.apply(u, v, buffers["hard"], material, .05, 4096).detach()
    _count(evidence, "native_P0_moments", True)
    _require(float((probability.sum(1) - 1).abs().max()) <= 1e-12
             and torch.allclose(moments, probability.T @ material / len(u), atol=1e-12, rtol=1e-12), "Current P0 conservation differs")
    x, q, mass = representative(moments, buffers["transform"], 3703, "cuda")
    weights = torch.full_like(mass, 1 / 120)
    evidence["manual_current_native_U0_V0_exact"] = True
    return dict(material=material, probability=probability, moments=moments, inputs=(x, q, weights), parameters=(u, v),
        input_digest=_input_digest(x, q, weights, buffers["graph"], buffers["q"],
                     dict(train=buffers["train"], val=buffers["val"]), None, lambda: False))


def _save_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("xb") as stream:
            torch.save(cpu_state(value), stream)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _partition(value):
    return (torch.is_tensor(value) and value.dtype == torch.int64 and value.shape == (3327,)
            and bool((value >= 0).all()) and len(torch.unique(value)) == 120 and int(value.max()) == 119)


def _initialize(spec, science, buffers, evidence, stop):
    paths = spec["creation_paths"]
    _require(all(not Path(p).exists() for p in paths.values()), "Owned initializer5 outputs already exist")
    probe._runtime_precision_guard()
    _count(evidence, "feature_KMeans")
    feature = feature_kmeans(buffers["h"].cpu(), 120, 5)
    _count(evidence, "feature_KMeans", True)
    probe._stop(stop)
    _count(evidence, "teacher_joint_KMeans")
    hard = teacher_aware_kmeans(buffers["h"], buffers["q"], 120, 5, alpha=.3, feature_weights=None)
    _count(evidence, "teacher_joint_KMeans", True)
    evidence["counts"].update(integrated_nested_feature_KMeans_completed=1, initializer_representation_RMS_completed=1)
    _require(_partition(feature) and _partition(hard), "Genuine partitions malformed")
    _save_new(paths["inputs"], dict(z=buffers["z"].detach().clone(),
              transform=probe._frozen_transform(buffers["transform"]), assignment=feature))
    _save_new(paths["hard_assignment"], hard)
    own = _cached_parts(5, buffers, evidence)
    _require(all(_native_buffers(own)[k] == evidence["native_buffers_before"][k] for k in replication._COMMON),
             "Genuine initializer altered common source")
    origin = _origin(own, 5, evidence)
    _save_new(paths["pre_optimizer_P0"], dict(schema=1, condensation_seed=5, source=spec["source"],
              numerical_source=spec["numerical_source"], scientific_preregistration=spec["scientific_preregistration"],
              source_context=own["source"], native_buffers=_native_buffers(own), origin=origin))
    evidence.update(genuine_seed5_creation_passed=True, pre_optimizer_P0_certificate_passed=True,
        created_files_sha256={p: _sha(p) for p in paths.values()}, pre_optimizer_P0=probe._digest(origin),
        own_source_context=own["source"], own_native_buffers=_native_buffers(own),
        cached_NODE0_equality_available=False, source_RMS_refits=0)


def _bind_initializer(spec, spec_sha, checksum, buffers, origin, evidence):
    path = Path(spec["operation_outputs"]["prepare_initializer"]["5"])
    _require(type(checksum) is str and len(checksum) == 64 and path.is_file() and _sha(path) == checksum,
             "Passed initializer receipt exactSHA required")
    prior = json.loads(path.read_text())
    _require(prior.get("passed") is True and prior.get("operation") == "prepare_initializer"
             and type(prior.get("condensation_seed")) is int and prior["condensation_seed"] == 5
             and prior.get("source_assets_spec_science_unchanged") is True
             and prior.get("genuine_seed5_creation_passed") is True and prior.get("pre_optimizer_P0_certificate_passed") is True
             and all(prior.get(k) == spec[k] for k in ("source", "numerical_source", "scientific_preregistration"))
             and prior.get("spec_sha256") == spec_sha
             and prior.get("created_files_sha256") == {p: _sha(p) for p in spec["creation_paths"].values()},
             "Initializer dependency/createdbytes unqualified")
    _require(prior.get("own_source_context") == buffers["source"] and prior.get("own_native_buffers") == _native_buffers(buffers)
             and prior.get("pre_optimizer_P0") == probe._digest(origin), "Current own5 origin differs from phase1")
    saved = torch.load(spec["creation_paths"]["pre_optimizer_P0"], map_location="cuda", weights_only=False)
    _require(isinstance(saved, dict) and type(saved.get("schema")) is int and saved["schema"] == 1
             and type(saved.get("condensation_seed")) is int and saved["condensation_seed"] == 5
             and all(saved.get(k) == spec[k] for k in ("source", "numerical_source", "scientific_preregistration"))
             and saved.get("source_context") == buffers["source"]
             and saved.get("native_buffers") == _native_buffers(buffers) and probe._digest(saved.get("origin")) == probe._digest(origin),
             "Durable preP0 differs from current own origin")
    evidence.update(initializer_dependency_passed=True, initializer_evidence_path=str(path), initializer_evidence_sha256=checksum,
                    initializer_created_files_sha256=prior["created_files_sha256"])


def _run_core(buffers, science, folder, evidence, stop):
    _require(not folder.exists(), "Fresh ownNODE5 folder required")
    probe._runtime_precision_guard()
    kwargs = dict(science["NODE5_exact_core_call"]["kwargs"])
    _require(kwargs["folder"] == str(folder), "Wrong fixed core folder")
    kwargs["folder"], kwargs["checkpoint_steps"] = folder, tuple(kwargs["checkpoint_steps"])
    _count(evidence, "baseline_optimizer")
    optimize_ce_assignment(buffers["z"], buffers["q"], buffers["hard"], **kwargs, stop=stop)
    _count(evidence, "baseline_optimizer", True)
    evidence["core_head_and_adjoint_work"] = "Allowed unchanged core work; internal call counts are not observed by adapter"


def _observe_frontier(folder, evidence):
    path = folder / "resume.pt"
    if not path.exists():
        return
    saved = torch.load(path, map_location="cpu", weights_only=False)
    step = saved.get("step") if isinstance(saved, dict) else None
    _require(type(step) is int and 0 <= step <= 25, "Malformed durable raw frontier")
    evidence["counts"]["P_update_completed"] = step
    evidence["observed_durable_frontier"] = step
    evidence["baseline_files_sha256"] = {str(p): _sha(p) for p in
        [folder / "resume.pt", folder / "optimization.csv", *sorted((folder / "checkpoints").glob("*.pt"))] if p.is_file()}


_INTEGER_CSV_POLICY = {'field': 'hessian_reduced_dimension', 'positions': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24], 'resume_type': 'exact Python int', 'resume_value': 720, 'CSV_text': '720.0', 'terminal25': 'dictionary absent and CSV blank, unchanged', 'other_integer_fields': 'Exact str(integer) unchanged', 'normalization': 'None: preserve native integer value; match only exact frozen representation', 'tolerance': 'No numeric tolerance or float parsing needed'}


def _nullable_integer_csv(step, value, text):
    """Only the frozen pandas nullable-int token; no parsing or conversion."""
    return (type(step) is int and 0 <= step < 25 and type(value) is int
            and value == 720 and type(text) is str and text == "720.0")


def _history_metadata(resume, csv_path):
    """Validate full typed metadata before encoding explicit absent diagnostics."""
    rows = _history(resume, csv_path)
    with csv_path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        disk, columns = list(reader), reader.fieldnames
    normalized, placeholders, nullable_integer_count = [], [], 0
    for step, (row, csv_row) in enumerate(zip(rows, disk, strict=True)):
        _require(set(_NAN) <= set(row) and set(csv_row) == set(columns)
                 and all(type(text) is str for text in csv_row.values()), "Incomplete/malformed full metadata row")
        current = {}
        for key in columns:
            text = csv_row[key]
            if key not in row:
                _require(step in _MISSING.get(key, ()) and text == "", "Undeclared missing history field")
                current[key] = {"dictionary_field_absent": key}
                continue
            value = row[key]
            absent = step in _NAN.get(key, ())
            if absent:
                _require(type(value) is float and math.isnan(value) and text == "", "Expected genuine NaN diagnostic/blank CSV")
                current[key] = {"diagnostic_absent": key}
                placeholders.append([step, key])
            elif key == "hessian_reduced_dimension":
                _require(_nullable_integer_csv(step, value, text), "Integer metadata differs")
                current[key] = value
                nullable_integer_count += 1
            elif type(value) is bool:
                _require(text == str(value), "Boolean metadata differs")
                current[key] = value
            elif type(value) is int:
                _require(text == str(value), "Integer metadata differs")
                current[key] = value
            elif type(value) is float:
                _require(math.isfinite(value) and text != "" and math.isfinite(float(text))
                         and math.isclose(float(text), value, abs_tol=1e-12, rel_tol=1e-12), "Nonfinite/mismatched numeric metadata")
                current[key] = value
            elif type(value) is str:
                _require(text == value, "String metadata differs")
                current[key] = value
            else:
                raise ValueError("Unsupported typed history metadata")
        _require(all((key in row) == (step not in positions) for key, positions in _MISSING.items()),
                 "Only terminal Hessian fields may be absent")
        normalized.append(current)
    _require(nullable_integer_count == 25, "Incomplete nullable-integer metadata")
    return rows, dict(full_CSV_resume_metadata_equal=True, NaN_positions=placeholders,
                      nullable_integer_CSV_exact_serialization_passed=True,
                      nullable_integer_CSV_serialization_count=nullable_integer_count,
                      nullable_integer_CSV_serialization_policy=_INTEGER_CSV_POLICY,
                      missing_terminal_hessian_fields=True, normalized_metadata_sha256=_seal(normalized),
                      policy="Genuine float NaN/blank CSV only at frozen positions; computed values stay finite")


def _endpoints(buffers, origin, expected, folder, evidence):
    resume = torch.load(folder / "resume.pt", map_location="cpu", weights_only=False)
    _require(isinstance(resume, dict) and type(resume.get("step")) is int and resume["step"] == 25
             and resume.get("config") == expected and isinstance(resume.get("snapshots"), dict)
             and set(resume["snapshots"]) == {0, 25}, "Own complete NODE resume/config differs")
    rows, metadata = _history_metadata(resume, folder / "optimization.csv")
    _require(sorted(p.name for p in (folder / "checkpoints").iterdir()) == ["step_000000.pt", "step_000025.pt"], "Extra/missing endpoint files")
    certificates = {}
    for step in (0, 25):
        snapshot = torch.load(folder / "checkpoints" / f"step_{step:06d}.pt", map_location="cpu", weights_only=False)
        _require(isinstance(snapshot, dict) and type(snapshot.get("step")) is int and snapshot["step"] == step
                 and _seal(snapshot) == _seal(resume["snapshots"][step]), "Checkpoint/resume mismatch")
        moments = probe._tensor(snapshot.get("moments"), (120, 3710), torch.float64, "Malformed own moments").to(buffers["z"])
        _require(bool((moments[:, 0] > 0).all()) and torch.allclose(moments.sum(0), origin["material"].mean(0), atol=1e-12, rtol=1e-12),
                 "Own material conservation differs")
        centers, labels, mass = decode_moments(moments, 3703)
        _require(bool((labels >= 0).all()) and float((labels.sum(1) - 1).abs().max()) <= 1e-12
                 and abs(float(mass.sum()) - 1) <= 1e-12, "Own material simplex/mass differs")
        theta = probe._tensor(snapshot.get("theta"), (6, 3704), torch.float64, "Malformed own head").to(buffers["z"])
        _count(evidence, "certificate_head_gradient")
        gradient = float(head_gradient(augmented(centers), labels, torch.full_like(mass, 1 / 120), theta, .001).abs().max())
        _count(evidence, "certificate_head_gradient", True)
        _count(evidence, "certificate_teacher_CE")
        value, _ = outer_value_gradient(buffers["z"], buffers["q"], theta, 65536, augmented(buffers["z"]))
        _count(evidence, "certificate_teacher_CE", True)
        _require(snapshot.get("J_exact") is True and math.isfinite(gradient) and gradient <= 1e-7
                 and math.isclose(gradient, probe._scalar(snapshot["inner_grad_max"], "Head gradient certificate"), abs_tol=1e-12, rel_tol=1e-12)
                 and math.isclose(value, probe._scalar(snapshot["teacher_ce"], "Teacher CE certificate"), abs_tol=1e-12, rel_tol=1e-12)
                 and rows[step]["J"] == snapshot["teacher_ce"] and rows[step]["inner_grad_max"] == snapshot["inner_grad_max"],
                 "Own linear head/value/history certificate differs")
        if step == 0:
            _require(torch.equal(moments, origin["moments"]), "Core native step0 differs from own manual P0")
            x, q, mass = representative(moments, buffers["transform"], 3703, "cuda")
            _require(all(torch.equal(a, b) for a, b in zip((x, q, torch.full_like(mass, 1 / 120)), origin["inputs"], strict=True)),
                     "Own core FP32 X/Q and supplied F64 uniform differ")
        else:
            parameters = resume.get("parameters")
            _require(isinstance(parameters, list) and len(parameters) == 2, "Malformed final native factors")
            u = probe._tensor(parameters[0], (3327, 8), torch.float32, "Final U").to(buffers["hard"].device)
            v = probe._tensor(parameters[1], (120, 8), torch.float32, "Final V").to(u)
            _count(evidence, "terminal_factor_moments")
            replay = LowRankMoments.apply(u, v, buffers["hard"], origin["material"], .05, 4096).detach()
            _count(evidence, "terminal_factor_moments", True)
            _require(torch.equal(replay, moments) and torch.equal(resume["theta"].to(theta), theta), "Terminal current factors/head differ from snapshot")
        certificates[str(step)] = dict(snapshot=probe._digest(snapshot), head_gradient_max=gradient, teacher_ce=value)
    _require(torch.equal(resume["initial_moments"].to(origin["moments"]), origin["moments"])
             and resume.get("scale") == max(rows[0]["J"], 1e-12), "Own initial objective/normalization differs")
    state = resume.get("optimizer")
    _require(isinstance(state, dict) and set(state.get("state", {})) == {0, 1} and len(state.get("param_groups", [])) == 1,
             "Incomplete final two-slot Adam")
    group = state["param_groups"][0]
    _require(group["params"] == [0, 1] and group["lr"] == .01 and tuple(group["betas"]) == (.9, .999)
             and group["eps"] == 1e-12 and group["weight_decay"] == 0 and group["foreach"] is False
             and group.get("fused") is None and group.get("capturable") is False
             and group.get("differentiable") is False and group.get("maximize") is False
             and group.get("amsgrad") is False,
             "Final legacy Adam policy differs")
    for index, shape in enumerate(((3327, 8), (120, 8))):
        slot = state["state"][index]
        counter = probe._tensor(slot.get("step"), (), torch.float32, "Final Adam counter")
        _require(counter.device.type == "cpu" and float(counter) == 25, "Final Adam count differs")
        for key in ("exp_avg", "exp_avg_sq"):
            value = probe._tensor(slot.get(key), shape, torch.float32, "Final Adam slot")
            _require(key != "exp_avg_sq" or bool((value >= 0).all()), "Negative Adam square slot")
    paths = [folder / "resume.pt", folder / "optimization.csv", *sorted((folder / "checkpoints").iterdir())]
    evidence.update(complete25_native_certificate_passed=True, actual_P0_moments_bitwise_equal_core_step0=True,
                    actual_FP32_P0_X_Q_F64_uniform_equal_core_step0=True, endpoint_certificates=certificates,
                    certified_files_sha256={str(p): _sha(p) for p in paths}, history_metadata=metadata,
                    prior_completed_baseline_P_updates=25)


def _finish(spec, science, spec_path, checksum, buffers, native, started, evidence, primary):
    def check(name, action):
        nonlocal primary
        try:
            action()
        except BaseException as error:
            evidence.update(passed=False)
            evidence[name] = dict(type=type(error).__name__, message=str(error))
            if primary is None:
                primary = error

    def source_buffers():
        evidence["native_buffers_after"] = _native_buffers(buffers)
        _require(evidence["native_buffers_before"] == evidence["native_buffers_after"], "Retained source buffers changed")

    def memory():
        torch.cuda.synchronize()
        total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
        allocated, reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
        evidence.update(CUDA_peak_allocated_bytes=allocated, CUDA_peak_reserved_bytes=reserved, CUDA_total_bytes=total)
        probe._runtime_precision_guard()
        _require(total == CUDA_CAPACITY and 0 <= allocated <= total and 0 <= reserved <= total, "Native memory policy changed")

    def preserve():
        _preserve(spec, Path(spec_path).resolve(), checksum, science)
        if "initializer_created_files_sha256" in evidence:
            probe._checked_files(evidence["initializer_created_files_sha256"])
            probe._checked_files({evidence["initializer_evidence_path"]: evidence["initializer_evidence_sha256"]})
        if "created_files_sha256" in evidence:
            probe._checked_files(evidence["created_files_sha256"])
        evidence["source_assets_spec_science_unchanged"] = True

    if buffers is not None:
        check("native_buffer_preservation_error", source_buffers)
    if native:
        check("memory_error", memory)
    evidence["source_assets_spec_science_unchanged"] = False
    check("preservation_error", preserve)
    evidence["seconds"] = time.monotonic() - started
    check("deadline_error", lambda: _require(evidence["seconds"] <= 300, "Prerequisite job exceeded300seconds"))
    return primary


def _run(operation, seed, spec_path, spec_sha256, output_path, initializer_sha, stop):
    _require(callable(stop), "Require stop callback")
    _require((operation == "prepare_node_reference") == (initializer_sha is not None), "DependencySHA only for NODE5")
    repo = Path(__file__).resolve().parents[1]
    spec, science, output, folder = _load_spec(operation, seed, spec_path, spec_sha256, output_path, repo)
    evidence = dict(passed=False, operation=operation, condensation_seed=seed, source=spec["source"],
        numerical_source=spec["numerical_source"], python_version=spec["python_version"],
        spec_path=str(Path(spec_path).resolve()), spec_sha256=spec_sha256,
        scientific_preregistration=spec["scientific_preregistration"], source_root=spec["fixed"]["source_root"],
        source_backend=spec["fixed"]["source_backend"],
        baseline_candidate=original.CANDIDATE, baseline_candidate_id=original.REFERENCE, baseline_folder=str(folder),
        files_sha256=spec["files_sha256"], counts=dict(P_update_completed=0), qualified_NODE_P_updates=0,
        students=0, source_GCN_gradient_targets=0, serving_outputs=0, test_enabled=False,
        source_RMS_refits=0, teacher_map_Phi_fits=0, accuracy_forwards=0,
        count_scope="Own interface attempted/completed diagnostics; dataset/nested initializer internals unknown on failure. Only NODE5 durable frontier counts new P updates; head/adjoint core work unknown, never assumed zero.",
        heldout_labels_scope="Protected graph identity hashes only; no heldout loss/accuracy/selection")
    started, native, buffers, primary = time.monotonic(), False, None, None
    bounded = lambda: stop() or time.monotonic() - started >= 300
    try:
        evidence["stage"] = "fresh_native_policy"
        probe._stop(bounded)
        _require(torch.get_num_threads() == 4 and not torch.cuda.is_initialized(), "Require threads4/fresh CUDA worker")
        evidence["native_environment"] = dict(**probe._native("cuda"), python_version=platform.python_version(), threads=4)
        native = True
        torch.cuda.reset_peak_memory_stats()
        _require(torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory == CUDA_CAPACITY, "GPU capacity changed")
        evidence["stage"] = "own_original_cached_source"
        buffers = _load_source(0 if operation == "prepare_initializer" else seed, spec, science, evidence, bounded)
        probe._runtime_precision_guard()
        if operation == "prepare_initializer":
            evidence["stage"] = "genuine_two_partitions_and_own_pre_optimizer_P0"
            _initialize(spec, science, buffers, evidence, bounded)
        else:
            evidence["stage"] = "own_native_factor_P0"
            origin = _origin(buffers, seed, evidence)
            expected = _core_config(buffers["ghost"], seed, buffers)
            if operation == "certify_existing":
                resume = torch.load(folder / "resume.pt", map_location="cpu", weights_only=False)
                retained = resume.get("config", {}).get("save_assignment") if isinstance(resume, dict) else None
                _require(type(retained) is bool, "Malformed original artifact retention flag")
                expected["save_assignment"] = retained
            else:
                evidence["stage"] = "exact_passed_initializer_dependency"
                _bind_initializer(spec, spec_sha256, initializer_sha, buffers, origin, evidence)
                evidence["stage"] = "once_fresh_NODE5_25_updates"
                try:
                    _run_core(buffers, science, folder, evidence, bounded)
                finally:
                    try:
                        _observe_frontier(folder, evidence)
                    except BaseException as error:
                        evidence["raw_frontier_observation_error"] = dict(type=type(error).__name__, message=str(error))
            probe._stop(bounded)
            probe._runtime_precision_guard()
            evidence["stage"] = "own_complete_NODE25_endpoint_and_typed_history_certificate"
            _endpoints(buffers, origin, expected, folder, evidence)
            if operation == "prepare_node_reference":
                _require(evidence["counts"]["P_update_completed"] == 25, "Durable25 update frontier required")
                evidence["qualified_NODE_P_updates"] = 25
        evidence.update(passed=True, stage="qualified_" + operation)
    except BaseException as error:
        primary = error
        evidence.update(primary_error=dict(type=type(error).__name__, message=str(error)), failed_stage=evidence.get("stage"))
    finally:
        primary = _finish(spec, science, spec_path, spec_sha256, buffers, native, started, evidence, primary)
        try:
            evidence["observed_owned_creation_files_sha256"] = {p: _sha(p) for p in spec["creation_paths"].values() if Path(p).is_file()}
        except BaseException as error:
            evidence.update(passed=False, publication_observation_error=dict(type=type(error).__name__, message=str(error)))
            if primary is None:
                primary = error
        try:
            _write_new(output, evidence)
        except BaseException:
            if primary is not None:
                raise primary
            raise
    if primary is not None:
        raise primary
    return dict(evidence, evidence_path=str(output), evidence_sha256=_sha(output), validation_only=True)


def certify_existing(condensation_seed, spec_path, spec_sha256, output_path, stop=lambda: False):
    return _run("certify_existing", condensation_seed, spec_path, spec_sha256, output_path, None, stop)


def prepare_initializer(condensation_seed, spec_path, spec_sha256, output_path, stop=lambda: False):
    return _run("prepare_initializer", condensation_seed, spec_path, spec_sha256, output_path, None, stop)


def prepare_node_reference(condensation_seed, spec_path, spec_sha256, output_path, initializer_evidence_sha256, stop=lambda: False):
    return _run("prepare_node_reference", condensation_seed, spec_path, spec_sha256, output_path, initializer_evidence_sha256, stop)
