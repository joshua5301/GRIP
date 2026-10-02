"""Fresh own Cora70 NODE cond1/2 baselines, without student evaluation.

The unchanged legacy optimizer owns all 25 updates. This adapter checks source
and native origin before entering it, then certifies its durable endpoints.
Failed/partial folders are preserved and never resumed by this entry point.
"""
import csv
import json
import math
import platform
import time
from pathlib import Path

import torch

from src import citation_search
from src import cora_source_certificate as original
from src import finite_student_probe as probe
from src import source_linear_assignment as source_helper
from src.citation_source_preflight import _exact, _write_new
from src.evaluation import _input_digest
from src.io import _fingerprint
from src.low_rank_assignment import LowRankLogits, LowRankMoments, initialize_factors
from src.moments import augmented, decode_moments, make_material
from src.research_loop import implementation_provenance
from src.soft_ce_partition import head_gradient, optimize_ce_assignment, outer_value_gradient
from src.sweep_utils import representative
from src.target_refinement import training_refined_targets

SCIENCE = "Cora70_NODE_reference_cond1_2_scientific_stageBE_v2.json"
SCIENCE_SHA = "d9d95d0feb73183d281b97e53bb0d085a018874b726b7a9d6867f9fcc1ce7b6d"
CUDA_CAPACITY = 8316977152
_require, _sha, _seal = probe._require, probe._sha, probe._seal
_COMMON = ("H", "z", "Q", "transform", "X", "original_CSR", "dense_original_S")


def numerical_source():
    value = probe.numerical_source()
    _require(value["files"].get("cora_node_reference.py") == _sha(__file__),
             "Current own NODE adapter missing from numerical source")
    return value


def _science(repo):
    path = repo / "results/proposals" / SCIENCE
    probe._checked_files({str(path): SCIENCE_SHA})
    return json.loads(path.read_text())


def _paths(repo):
    return [Path(p) for p in _science(repo)["original_files_sha256"]]


def _source_unchanged(spec):
    _require(implementation_provenance() == spec["source"]
             and numerical_source() == spec["numerical_source"]
             and platform.python_version() == spec["python_version"]
             and torch.get_num_threads() == 4, "Source/Git/versions/Python/threads changed")


def _preserve(spec, path, checksum, science):
    _require(_sha(path) == checksum, "Frozen own NODE spec changed")
    probe._checked_files(spec["files_sha256"])
    probe._checked_files(spec["artifacts_sha256"])
    probe._checked_files({spec["scientific_preregistration"]["path"]: SCIENCE_SHA})
    refs = [*science["parents"], science["original_BB_certificate"]]
    refs.append(science["preregistration_revision"]["previous_unexecuted"])
    probe._checked_files({r["path"]: r["sha256"] for r in refs})
    _source_unchanged(spec)


def _seed(value):
    _require(type(value) is int and value in (1, 2), "Require condensation seed1 or2")
    return value


def _outputs(science):
    return {str(c): str(Path(science["output_root"]) / f"native_condensation{c}_v1.json") for c in (1, 2)}


def _load_spec(seed, path, checksum, output, repo):
    seed = _seed(seed)
    path, output = Path(path).resolve(), Path(output).resolve()
    _require(type(checksum) is str and len(checksum) == 64 and path.is_file()
             and path.is_relative_to(repo / "results/proposals") and _sha(path) == checksum,
             "Require frozen prospective spec path/SHA")
    science = _science(repo)
    spec = json.loads(path.read_text())
    fields = {"schema", "fixed", "case", "baseline_candidate", "source", "numerical_source", "python_version",
              "files_sha256", "scientific_preregistration", "artifacts_sha256", "outputs"}
    _require(isinstance(spec, dict) and set(spec) == fields and type(spec["schema"]) is int and spec["schema"] == 1
             and _exact(spec["fixed"], science["fixed"]) and _exact(spec["case"], science["case"])
             and _exact(spec["baseline_candidate"], science["baseline_candidate"]), "Unknown or changed own NODE controls")
    _require(_fingerprint(spec["baseline_candidate"]) == science["baseline_candidate_id"]
             and spec["baseline_candidate"] == spec["case"]["reference_candidate"], "Wrong original NODE identity")
    _require(spec["scientific_preregistration"] == dict(path=str(repo / "results/proposals" / SCIENCE), sha256=SCIENCE_SHA)
             and spec["outputs"] == _outputs(science), "Scientific lineage/output namespace changed")
    pins = spec["files_sha256"]
    _require(isinstance(pins, dict) and set(pins) == {str(p) for p in _paths(repo)}
             and all(pins.get(p) == h for p, h in science["original_files_sha256"].items()),
             "Require original assets and both own inputs pinned")
    _require(isinstance(spec["artifacts_sha256"], dict) and spec["artifacts_sha256"]
             and all(Path(p).is_absolute() and Path(p).resolve().is_relative_to(repo / "results")
                     for p in spec["artifacts_sha256"]), "Require reviewed metadata pins")
    folder = Path(science["new_only_reference_dirs"][str(seed)])
    _require(folder == Path(spec["case"]["source_root"]) / science["baseline_candidate_id"] / f"condensation_{seed}"
             and not folder.exists() and str(output) == spec["outputs"][str(seed)] and not output.exists(),
             "Require absent own condensation/evidence; partial state is never resumed")
    _require((repo / "data/cora/processed/data.pt").is_file(), "Require existing original dataset")
    _preserve(spec, path, checksum, science)
    original._config(spec["case"])
    original._recipe(spec["case"])
    return spec, science, output, folder


def _count(evidence, key, complete=False):
    suffix = "completed" if complete else "attempts"
    name = f"{key}_{suffix}"
    evidence["counts"][name] = evidence["counts"].get(name, 0) + 1


def _native_buffers(buffers):
    return probe._digest(dict(H=buffers["h"], z=buffers["z"], Q=buffers["q"], hard=buffers["hard"],
        transform=probe._frozen_transform(buffers["transform"]), X=buffers["graph"]["x"],
        original_CSR=buffers["graph"]["adj"], dense_original_S=buffers["dense"]))


def _bind_common(science, case, native, recipe):
    ref = science["original_BB_certificate"]
    probe._checked_files({ref["path"]: ref["sha256"]})
    bb = json.loads(Path(ref["path"]).read_text())
    _require(bb.get("passed") is True and bb.get("source_assets_spec_science_unchanged") is True,
             "Original BB certificate is unqualified")
    certified = bb["roots"][case["root"]]
    _require(certified.get("passed") is True and certified.get("source_P0_linear_certificate_passed") is True
             and all(native[k] == certified["native_buffers"][k] for k in _COMMON)
             and _exact(recipe, certified["student_recipe_origin"]), "Common own original source/recipe differs from BB")


def _core_config(ghost, seed, buffers):
    expected = source_helper.expected_citation_config(ghost, seed, buffers["z"], buffers["q"], buffers["hard"],
                  buffers["z"].detach().float(), buffers["source"])
    for key in ("source_linear_coordinates", "source_linear_schema", "source_linear_source"):
        expected.pop(key)
    expected.update(assignment_input="node", save_assignment=False)
    return expected


def _origin(buffers, seed, evidence):
    material = make_material(buffers["z"], buffers["q"])
    values = []
    for _ in range(2):
        _count(evidence, "native_factor_factory")
        values.append(initialize_factors(buffers["hard"], 70, 32, seed))
        _count(evidence, "native_factor_factory", True)
    _require(all(torch.equal(a, b) for a, b in zip(values[0], values[1], strict=True)), "Paired current native U0/V0 differ")
    u, v = values[0]
    _count(evidence, "native_P0_logits")
    probability = LowRankLogits.apply(u, v, buffers["hard"], .05, 4096).double().softmax(1).detach()
    _count(evidence, "native_P0_logits", True)
    _count(evidence, "native_P0_moments")
    moments = LowRankMoments.apply(u, v, buffers["hard"], material, .05, 4096).detach()
    _count(evidence, "native_P0_moments", True)
    _require(float((probability.sum(1) - 1).abs().max()) <= 1e-12
             and torch.allclose(moments, probability.T @ material / len(u), atol=1e-12, rtol=1e-12), "Current P0 material conservation differs")
    x, q, mass = representative(moments, buffers["transform"], 1433, "cuda")
    weights = torch.full_like(mass, 1 / 70)
    return dict(material=material, moments=moments, inputs=(x, q, weights), parameters=(u, v),
        input_digest=_input_digest(x, q, weights, buffers["graph"], buffers["q"],
                                  dict(train=buffers["train"], val=buffers["val"]), None, lambda: False))


def _load_source(spec, science, seed, evidence, stop):
    repo, case = Path(__file__).resolve().parents[1], spec["case"]
    shared = original._prepare_graph(repo, evidence, stop)
    graph, root = shared["graph"], Path(case["source_root"])
    config = citation_search._legacy_teacher_config("cora", case["ratio"], graph, shared["train"],
                                                   (None, shared["val"]), shared["test"], "row")
    _require(_exact(config, original._config(case)), "Original graph/config changed")
    recipe = original._recipe(case)
    teacher = torch.load(root / "teacher.pt", map_location="cuda", weights_only=False)
    logits = probe._tensor(teacher.get("logits") if isinstance(teacher, dict) else None,
                           (2708, 7), torch.float64, "Original teacher logits malformed")
    q = training_refined_targets(logits, .3, graph["y"], shared["train"], 0)
    ghost = source_helper.candidate_controls(dict(spec["baseline_candidate"], method="source_linear",
                         assignment_coordinates="raw_rms", source_linear_schema=1))
    evidence["stage"] = "strict_own_seed_cached_source"
    _count(evidence, "cached_source")
    h, z, transform, hard, _, source = source_helper.cached_source(root, ghost, seed, shared["fresh_h"], q, config)
    _count(evidence, "cached_source", True)
    _require(z.shape == (2708, 1433) and q.shape == (2708, 7) and int(hard.max()) + 1 == 70,
             "Original source shape/budget differs")
    _require(_fingerprint(dict(mode="teacher_balanced", alpha=.3, T=.3, seed=seed))
             == Path(science["hard_assignments"][str(seed)]["path"]).stem.removeprefix("assignment_"), "Own hard initializer path differs")
    _require(json.loads((root / science["baseline_candidate_id"] / "candidate.json").read_text())
             == spec["baseline_candidate"], "Own NODE candidate header differs")
    buffers = dict(root=root, graph=graph, train=shared["train"], val=shared["val"], h=h, z=z, q=q,
                   hard=hard, transform=transform, source=source, dense=shared["dense"])
    native = _native_buffers(buffers)
    _bind_common(science, case, native, recipe)
    origin = _origin(buffers, seed, evidence)
    evidence.update(source_context=source, native_buffers_before=native, student_recipe_origin=recipe,
                    source_reference_passed_before_P=True, manual_current_native_U0_V0_exact=True,
                    historical_condensation_UV_available=False, pre_optimizer_P0=probe._digest(origin))
    return buffers, origin, _core_config(ghost, seed, buffers)


def _run_core(buffers, seed, folder, evidence, stop):
    _require(not folder.exists(), "Own optimizer folder must remain absent before fresh call")
    probe._runtime_precision_guard()
    _count(evidence, "baseline_optimizer")
    result = optimize_ce_assignment(buffers["z"], buffers["q"], buffers["hard"], penalty=.0001, steps=25,
        mixing=.05, lr=.05, assignment_rank=32, factor_seed=seed, assignment_input="node",
        assignment_encoder="linear", encoder_hidden=64, solver_mode="exact", inner_method="newton_first",
        implicit_warm_start=True, mass_mode="free", inner_loss_weighting="uniform", inner_max_iter=2000,
        inner_tol=1e-7, cg_max_iter=512, cg_rtol=1e-6, cache_assignment=False, folder=folder,
        checkpoint_steps=(0, 25), resume_state=None, save_resume=True, save_assignment=False, stop=stop)
    _count(evidence, "baseline_optimizer", True)
    evidence["counts"]["P_update_completed"] = 25
    return result


def _history(resume, csv_path):
    rows = resume.get("history")
    _require(isinstance(rows, list) and len(rows) == 26 and all(isinstance(r, dict) and type(r.get("step")) is int
             and r["step"] == i for i, r in enumerate(rows)), "Require contiguous 0..25 history")
    with csv_path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        disk = list(reader)
        _require(reader.fieldnames == list(dict.fromkeys(k for row in rows for k in row)), "History CSV columns differ")
    _require(len(disk) == 26, "Incomplete history CSV")
    for i, (saved, row) in enumerate(zip(rows, disk, strict=True)):
        _require(row.get("step") == str(i) and saved.get("inner_converged") is True and saved.get("J_exact") is True
                 and saved.get("status") == ("evaluated" if i == 25 else "update"), "Uncertified history frontier")
        for key in ("J", "inner_grad_max", "seconds"):
            value = probe._scalar(saved.get(key), "Malformed history scalar")
            _require(value >= 0 and math.isclose(float(row[key]), value, abs_tol=1e-12, rel_tol=1e-12), "CSV/resume scalar differs")
        _require(saved["inner_grad_max"] <= 1e-7, "History head not converged")
        if i < 25:
            _require(saved.get("cg_converged") is True and probe._scalar(saved.get("cg_relative_residual"),
                     "Malformed update CG residual") <= 1e-6, "Uncertified implicit update")
    return rows


def _finish(spec, science, spec_path, checksum, buffers, native, started, evidence, primary):
    """Independent cleanup checks preserve both the primary error and all pins."""
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
        evidence["source_assets_spec_science_unchanged"] = True

    if buffers is not None:
        check("native_buffer_preservation_error", source_buffers)
    if native:
        check("memory_error", memory)
    evidence["source_assets_spec_science_unchanged"] = False
    check("preservation_error", preserve)
    evidence["seconds"] = time.monotonic() - started
    check("deadline_error", lambda: _require(evidence["seconds"] <= 300, "Baseline job exceeded300seconds"))
    return primary


def _qualify(buffers, origin, expected, folder, result, evidence):
    resume = torch.load(folder / "resume.pt", map_location="cpu", weights_only=False)
    _require(isinstance(resume, dict) and type(resume.get("step")) is int and resume["step"] == 25
             and resume.get("config") == expected and isinstance(resume.get("snapshots"), dict)
             and set(resume["snapshots"]) == {0, 25}, "Own complete NODE resume/config differs")
    rows = _history(resume, folder / "optimization.csv")
    _require(isinstance(result, dict) and _seal(result.get("history")) == _seal(rows)
             and _seal(result.get("checkpoints")) == _seal(resume["snapshots"]), "Returned core/current durable frontier differs")
    _require(sorted(p.name for p in (folder / "checkpoints").iterdir()) == ["step_000000.pt", "step_000025.pt"], "Extra/missing endpoint files")
    certificates = {}
    for step in (0, 25):
        snapshot = torch.load(folder / "checkpoints" / f"step_{step:06d}.pt", map_location="cpu", weights_only=False)
        _require(isinstance(snapshot, dict) and type(snapshot.get("step")) is int and snapshot["step"] == step
                 and _seal(snapshot) == _seal(resume["snapshots"][step]), "Checkpoint/resume mismatch")
        moments = probe._tensor(snapshot.get("moments"), (70, 1441), torch.float64, "Malformed own moments").to(buffers["z"])
        _require(bool((moments[:, 0] > 0).all()) and torch.allclose(moments.sum(0), origin["material"].mean(0), atol=1e-12, rtol=1e-12),
                 "Own material conservation differs")
        centers, labels, mass = decode_moments(moments, 1433)
        _require(bool((labels >= 0).all()) and float((labels.sum(1) - 1).abs().max()) <= 1e-12
                 and abs(float(mass.sum()) - 1) <= 1e-12, "Own material simplex/mass differs")
        theta = probe._tensor(snapshot.get("theta"), (7, 1434), torch.float64, "Malformed own head").to(buffers["z"])
        _count(evidence, "certificate_head_gradient")
        gradient = float(head_gradient(augmented(centers), labels, torch.full_like(mass, 1 / 70), theta, .0001).abs().max())
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
            x, q, mass = representative(moments, buffers["transform"], 1433, "cuda")
            _require(all(torch.equal(a, b) for a, b in zip((x, q, torch.full_like(mass, 1 / 70)), origin["inputs"], strict=True)),
                     "Own core FP32 X/Q and supplied F64 uniform differ")
        else:
            parameters = resume.get("parameters")
            _require(isinstance(parameters, list) and len(parameters) == 2, "Malformed final native factors")
            u = probe._tensor(parameters[0], (2708, 32), torch.float32, "Final U").to(buffers["hard"].device)
            v = probe._tensor(parameters[1], (70, 32), torch.float32, "Final V").to(u)
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
    _require(group["params"] == [0, 1] and group["lr"] == .05 and tuple(group["betas"]) == (.9, .999)
             and group["eps"] == 1e-12 and group["weight_decay"] == 0 and group["foreach"] is False
             and group.get("fused") is None and group.get("capturable") is False
             and group.get("differentiable") is False and group.get("maximize") is False
             and group.get("amsgrad") is False,
             "Final legacy Adam policy differs")
    for index, shape in enumerate(((2708, 32), (70, 32))):
        slot = state["state"][index]
        counter = probe._tensor(slot.get("step"), (), torch.float32, "Final Adam counter")
        _require(counter.device.type == "cpu" and float(counter) == 25, "Final Adam count differs")
        for key in ("exp_avg", "exp_avg_sq"):
            value = probe._tensor(slot.get(key), shape, torch.float32, "Final Adam slot")
            _require(key != "exp_avg_sq" or bool((value >= 0).all()), "Negative Adam square slot")
    paths = [folder / "resume.pt", folder / "optimization.csv", *sorted((folder / "checkpoints").iterdir())]
    evidence.update(complete25_native_certificate_passed=True, actual_P0_moments_bitwise_equal_core_step0=True,
                    actual_FP32_P0_X_Q_F64_uniform_equal_core_step0=True, endpoint_certificates=certificates,
                    new_files_sha256={str(p): _sha(p) for p in paths}, new_NODE_baseline_P_updates=25,
                    core_work=dict(head_solves=None, inner_CG_solves=None, scope="Not instrumented by unchanged core; never inferred zero"))


def prepare(condensation_seed, spec_path, spec_sha256, output_path, stop=lambda: False):
    """One fresh worker creates exactly one own NODE25 baseline; no retries."""
    _require(callable(stop), "Require stop callback")
    repo = Path(__file__).resolve().parents[1]
    spec, science, output, folder = _load_spec(condensation_seed, spec_path, spec_sha256, output_path, repo)
    evidence = dict(passed=False, source=spec["source"], numerical_source=spec["numerical_source"], python_version=spec["python_version"],
        spec_path=str(Path(spec_path).resolve()), spec_sha256=spec_sha256, scientific_preregistration=spec["scientific_preregistration"],
        operation="prepare", condensation_seed=condensation_seed, source_root=spec["case"]["source_root"], baseline_candidate=spec["baseline_candidate"],
        baseline_candidate_id=science["baseline_candidate_id"], baseline_folder=str(folder), counts={}, operation_attempts={},
        test_enabled=False, students=0, source_GCN_gradient_targets=0, serving_outputs=0,
        count_scope="Completed baseline updates are qualified only after durable step25; raw failed partial history remains separately observable")
    evidence["heldout_labels_scope"] = "Unchanged graph identity hashes only; never loss, accuracy or selection"
    for key in ("dataset_preparations", "dense_S_materializations"):
        evidence["counts"][key] = 0
    started, native, buffers, primary = time.monotonic(), False, None, None
    bounded = lambda: stop() or time.monotonic() - started >= 300
    try:
        probe._stop(bounded)
        evidence["stage"] = "fresh_native_policy"
        _require(torch.get_num_threads() == 4 and not torch.cuda.is_initialized(), "Require threads4/fresh CUDA worker")
        evidence["native_environment"] = dict(**probe._native("cuda"), python_version=platform.python_version(), threads=4)
        native = True
        torch.cuda.reset_peak_memory_stats()
        _require(torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory == CUDA_CAPACITY, "GPU capacity changed")
        buffers, origin, expected = _load_source(spec, science, condensation_seed, evidence, bounded)
        evidence["stage"] = "fresh_own_NODE25_optimizer"
        result = _run_core(buffers, condensation_seed, folder, evidence, bounded)
        evidence["stage"] = "durable_own_NODE25_qualification"
        _qualify(buffers, origin, expected, folder, result, evidence)
        evidence.update(passed=True, stage="qualified_own_NODE25_baseline")
    except BaseException as error:
        primary = error
        evidence.update(primary_error=dict(type=type(error).__name__, message=str(error)), failed_stage=evidence.get("stage"))
    finally:
        primary = _finish(spec, science, spec_path, spec_sha256, buffers, native, started, evidence, primary)
        evidence["partial_assets_preserved"] = True
        try:
            if folder.exists():
                evidence["observed_new_folder_files_sha256"] = {str(p): _sha(p) for p in sorted(folder.rglob("*")) if p.is_file()}
        except BaseException as error:
            evidence.update(passed=False, partial_inventory_error=dict(type=type(error).__name__, message=str(error)))
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
