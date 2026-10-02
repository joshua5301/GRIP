"""Readonly own Cora70 cond1/2 NODE25 certificates; no optimizer or fits.

NaN is permitted only as a genuine legacy diagnostic placeholder at frozen
positions. Computed moments, heads, scalar objectives and parameters stay finite.
"""
import csv
import json
import math
import platform
import time
from pathlib import Path

import torch

from src import cora_node_reference as baseline
from src import finite_student_probe as probe
from src.citation_source_preflight import _exact, _write_new
from src.low_rank_assignment import LowRankMoments
from src.moments import augmented, decode_moments
from src.research_loop import implementation_provenance
from src.soft_ce_partition import head_gradient, outer_value_gradient
from src.sweep_utils import representative

SCIENCE = "Cora70_NODE_reference_readonly_certificate_scientific_stageBF_v1.json"
SCIENCE_SHA = "a2a32672ff3a1bd15c577dd179256d35d80de81b12a7eca75b1b9b2f39f724f1"
BE_SHA = "a65410d48de3de3670227744d4b0d71252c9bcec903634eb17028ccc1a941698"
CUDA_CAPACITY = 8316977152
_require, _sha, _seal = probe._require, probe._sha, probe._seal
_load_source, _native_buffers = baseline._load_source, baseline._native_buffers
_history, _core_config, _origin = baseline._history, baseline._core_config, baseline._origin
_count, _seed = baseline._count, baseline._seed
_NAN = {"row_residual": tuple(range(26)), "column_residual": tuple(range(26)),
        "head_correction_relative": (0,), "implicit_correction_relative": (0, 25),
        "cg_residual": (25,), "cg_relative_residual": (25,)}
_MISSING = {"hessian_solver": (25,), "hessian_reduced_dimension": (25,)}


def numerical_source():
    value = probe.numerical_source()
    _require(value["files"].get("cora_node_reference.py") == BE_SHA
             and value["files"].get("cora_node_reference_certificate.py") == _sha(__file__),
             "Protected BE/new readonly numerical source changed")
    return value


def _science(repo):
    path = repo / "results/proposals" / SCIENCE
    probe._checked_files({str(path): SCIENCE_SHA})
    return json.loads(path.read_text())


def _paths(repo):
    return [Path(p) for p in _science(repo)["original_files_sha256"]]


def _outputs(science):
    return {str(c): str(Path(science["output_root"]) / f"native_condensation{c}_v1.json") for c in (1, 2)}


def _source_unchanged(spec):
    _require(implementation_provenance() == spec["source"] and numerical_source() == spec["numerical_source"]
             and platform.python_version() == spec["python_version"] and torch.get_num_threads() == 4,
             "Source/Git/versions/Python/threads changed")


def _preserve(spec, path, checksum, science):
    _require(_sha(path) == checksum, "Frozen readonly spec changed")
    for key in ("files_sha256", "baseline_cache_files_sha256", "artifacts_sha256"):
        probe._checked_files(spec[key])
    probe._checked_files({spec["scientific_preregistration"]["path"]: SCIENCE_SHA})
    probe._checked_files({r["path"]: r["sha256"] for r in [*science["parents"], science["original_BB_certificate"]]})
    _source_unchanged(spec)


def _load_spec(seed, path, checksum, output, repo):
    seed = _seed(seed)
    path, output = Path(path).resolve(), Path(output).resolve()
    _require(type(checksum) is str and len(checksum) == 64 and path.is_file()
             and path.is_relative_to(repo / "results/proposals") and _sha(path) == checksum, "Require frozen readonly spec path/SHA")
    science = _science(repo)
    spec = json.loads(path.read_text())
    fields = {"schema", "fixed", "case", "baseline_candidate", "source", "numerical_source", "python_version",
              "files_sha256", "baseline_cache_files_sha256", "scientific_preregistration", "artifacts_sha256", "outputs"}
    _require(isinstance(spec, dict) and set(spec) == fields and type(spec["schema"]) is int and spec["schema"] == 1
             and all(_exact(spec[k], science[k]) for k in ("fixed", "case", "baseline_candidate")), "Changed readonly controls")
    _require(spec["files_sha256"] == science["original_files_sha256"]
             and spec["baseline_cache_files_sha256"] == science["baseline_cache_files_sha256"]
             and spec["scientific_preregistration"] == dict(path=str(repo / "results/proposals" / SCIENCE), sha256=SCIENCE_SHA)
             and spec["outputs"] == _outputs(science), "Original cached bytes/lineage/output changed")
    _require(isinstance(spec["artifacts_sha256"], dict) and spec["artifacts_sha256"]
             and all(Path(p).is_absolute() and Path(p).resolve().is_relative_to(repo / "results")
                     for p in spec["artifacts_sha256"]), "Require reviewed immutable metadata pins")
    folder = Path(science["existing_only_reference_dirs"][str(seed)])
    _require(folder == Path(spec["case"]["source_root"]) / science["baseline_candidate_id"] / f"condensation_{seed}"
             and folder.is_dir() and str(output) == spec["outputs"][str(seed)] and not output.exists(),
             "Require existing own baseline and absent new readonly evidence")
    _preserve(spec, path, checksum, science)
    return spec, science, output, folder


def _history_metadata(resume, csv_path):
    """Validate full typed metadata before encoding explicit absent diagnostics."""
    rows = _history(resume, csv_path)
    with csv_path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        disk, columns = list(reader), reader.fieldnames
    normalized, placeholders = [], []
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
    return rows, dict(full_CSV_resume_metadata_equal=True, NaN_positions=placeholders,
                      missing_terminal_hessian_fields=True, normalized_metadata_sha256=_seal(normalized),
                      policy="Genuine float NaN/blank CSV only at frozen positions; computed values stay finite")


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
                    certified_files_sha256={str(p): _sha(p) for p in paths}, history_metadata=metadata,
                    new_NODE_baseline_P_updates=0, prior_completed_baseline_P_updates=25)


def certify(condensation_seed, spec_path, spec_sha256, output_path, stop=lambda: False):
    """Qualify existing complete25 bytes only, with zero scientific P updates."""
    _require(callable(stop), "Require stop callback")
    repo = Path(__file__).resolve().parents[1]
    spec, science, output, folder = _load_spec(condensation_seed, spec_path, spec_sha256, output_path, repo)
    evidence = dict(passed=False, source=spec["source"], numerical_source=spec["numerical_source"], python_version=spec["python_version"],
        spec_path=str(Path(spec_path).resolve()), spec_sha256=spec_sha256, scientific_preregistration=spec["scientific_preregistration"],
        operation="certify", condensation_seed=condensation_seed, source_root=spec["case"]["source_root"],
        baseline_candidate=spec["baseline_candidate"], baseline_candidate_id=science["baseline_candidate_id"], baseline_folder=str(folder),
        files_sha256=spec["files_sha256"], baseline_cache_files_sha256=spec["baseline_cache_files_sha256"],
        counts=dict(P_update_completed=0, optimizer_calls=0, head_solves=0, dataset_preparations=0, dense_S_materializations=0),
        operation_attempts={}, students=0, source_GCN_gradient_targets=0, serving_outputs=0, test_enabled=False,
        observed_prior_global_baseline_P_updates=50,
        count_scope="Readonly source/factory/material/head-gradient/teacher-CE diagnostics only; old baseline updates never counted again",
        heldout_labels_scope="Unchanged graph identity hashes only; never loss, accuracy or selection")
    started, native, buffers, primary = time.monotonic(), False, None, None
    bounded = lambda: stop() or time.monotonic() - started >= 300
    try:
        probe._stop(bounded)
        evidence["stage"] = "fresh_readonly_native_policy"
        _require(torch.get_num_threads() == 4 and not torch.cuda.is_initialized(), "Require threads4/fresh CUDA worker")
        evidence["native_environment"] = dict(**probe._native("cuda"), python_version=platform.python_version(), threads=4)
        native = True
        torch.cuda.reset_peak_memory_stats()
        _require(torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory == CUDA_CAPACITY, "GPU capacity changed")
        buffers, origin, expected = _load_source(spec, science, condensation_seed, evidence, bounded)
        probe._stop(bounded)
        probe._runtime_precision_guard()
        evidence["stage"] = "readonly_complete_NODE25_endpoint_and_metadata_certificate"
        _endpoints(buffers, origin, expected, folder, evidence)
        _require(evidence["counts"].get("certificate_head_gradient_completed") == 2
                 and evidence["counts"].get("certificate_teacher_CE_completed") == 2
                 and evidence["counts"].get("terminal_factor_moments_completed") == 1,
                 "Unexpected readonly endpoint diagnostic counts")
        evidence.update(passed=True, qualified_own_baseline_count=1, stage="qualified_existing_own_NODE25_baseline")
    except BaseException as error:
        primary = error
        evidence.update(primary_error=dict(type=type(error).__name__, message=str(error)), failed_stage=evidence.get("stage"))
    finally:
        primary = _finish(spec, science, spec_path, spec_sha256, buffers, native, started, evidence, primary)
        evidence["baseline_bytes_preserved"] = evidence.get("source_assets_spec_science_unchanged", False)
        try:
            _write_new(output, evidence)
        except BaseException:
            if primary is not None:
                raise primary
            raise
    if primary is not None:
        raise primary
    return dict(evidence, evidence_path=str(output), evidence_sha256=_sha(output), validation_only=True)
