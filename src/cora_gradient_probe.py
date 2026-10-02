"""Three disposable Cora CE-gradient runtime probes, one update per fresh worker.

The numerical core is the unchanged AY internal helper. Own Cora source and
historical readout certificates precede its stronger exact-M0 optimizer gate.
There are no students, scores, resume, continuation or fixed25 entry points.
"""
import json
import platform
import time
from pathlib import Path

import torch

from src import citation_gradient_probe as ay
from src import citation_search
from src import cora_source_certificate as original
from src import finite_student_probe as probe
from src import source_linear_assignment as source_helper
from src.ce_gradient_alignment import POLICY
from src.citation_source_preflight import _exact, _write_new
from src.io import _fingerprint
from src.low_rank_assignment import initialize_factors
from src.research_loop import implementation_provenance
from src.target_refinement import training_refined_targets

SCIENCE = "Cora35_70_140_CE_gradient_alignment_one_update_scientific_stageBC_v2.json"
SCIENCE_SHA = "80014039c269b2ea70109e1fc24d4724b25addaa4ea02581b2ca3d49db2bf5f6"
FIXED = dict(original.FIXED)
_POLICY_SPEC = json.loads(json.dumps(POLICY))
_require, _sha, _seal = probe._require, probe._sha, probe._seal
_one_update, _targets = ay._one_update, ay._targets
_source_buffer_digest, anchor_initial = ay._source_buffer_digest, ay.anchor_initial


def numerical_source():
    result = ay.numerical_source()
    _require(result["files"].get("cora_gradient_probe.py") == _sha(__file__),
             "Current Cora probe source missing or changed")
    return result


def _science(repo):
    path = repo / "results/proposals" / SCIENCE
    probe._checked_files({str(path): SCIENCE_SHA})
    return json.loads(path.read_text())


def _paths(repo):
    return [Path(p) for p in _science(repo)["original_files_sha256"]]


def _budget(cells, cases):
    _require(type(cells) is int and cells in (35, 70, 140), "Require fixed Cora35/70/140 budget")
    return next(case for case in cases if case["cells"] == cells)


def _source_unchanged(spec):
    _require(implementation_provenance() == spec["source"]
             and numerical_source() == spec["numerical_source"]
             and platform.python_version() == spec["python_version"]
             and torch.get_num_threads() == 4, "Current source/Git/Python/versions/threads changed")


def _preserve(spec, path, checksum, science):
    _require(_sha(path) == checksum, "Frozen spec changed")
    probe._checked_files(spec["files_sha256"])
    probe._checked_files(spec["artifacts_sha256"])
    probe._checked_files({spec["scientific_preregistration"]["path"]: SCIENCE_SHA})
    probe._checked_files({p["path"]: p["sha256"] for p in science["parents"]})
    previous = science["preregistration_revision"]["previous_unexecuted"]
    probe._checked_files({previous["path"]: previous["sha256"]})
    probe._checked_files(science["readonly_helper_source_pins"])
    _source_unchanged(spec)


def _load_spec(cells, path, checksum, output, repo):
    path, output = Path(path).resolve(), Path(output).resolve()
    _require(type(checksum) is str and len(checksum) == 64
             and path.is_relative_to(repo / "results/proposals") and path.is_file()
             and _sha(path) == checksum, "Require exact frozen Cora probe spec path/SHA")
    science = _science(repo)
    _require(_exact(science["fixed"], FIXED) and _exact(science["gradient_policy"], _POLICY_SPEC),
             "Frozen Cora controls/gradient policy changed")
    original._validate_cases(science["cases"], repo)
    spec = json.loads(path.read_text())
    fields = {"schema", "fixed", "cases", "candidate", "source", "numerical_source", "python_version",
              "files_sha256", "scientific_preregistration", "artifacts_sha256", "output_root",
              "probe_outputs", "gradient_policy"}
    _require(isinstance(spec, dict) and set(spec) == fields and type(spec["schema"]) is int
             and spec["schema"] == 1 and _exact(spec["fixed"], FIXED)
             and _exact(spec["cases"], science["cases"])
             and _exact(spec["candidate"], science["candidate"])
             and _exact(spec["gradient_policy"], _POLICY_SPEC), "Unknown or changed Cora probe spec")
    _require(spec["files_sha256"] == science["original_files_sha256"]
             and spec["scientific_preregistration"] == dict(
                 path=str(repo / "results/proposals" / SCIENCE), sha256=SCIENCE_SHA),
             "Original inventory/scientific lineage changed")
    _require(isinstance(spec["artifacts_sha256"], dict) and spec["artifacts_sha256"]
             and all(Path(p).is_absolute() and Path(p).resolve().is_relative_to(repo / "results")
                     for p in spec["artifacts_sha256"]), "Require reviewed immutable artifact pins")
    contract = science["implementation_contract"]
    _require(spec["output_root"] == contract["output_root"]
             and _exact(spec["probe_outputs"], contract["probe_outputs"]), "Changed probe output namespace")
    case = _budget(cells, spec["cases"])
    folder = output.with_suffix("")
    _require(str(output) == spec["probe_outputs"][str(cells)] and not output.exists() and not folder.exists(),
             "Require absent declared evidence/tensor namespace; partial caches are preserved")
    _require((repo / "data/cora/processed/data.pt").is_file(), "Require existing processed dataset")
    _preserve(spec, path, checksum, science)
    for row in spec["cases"]:
        original._config(row)
        original._recipe(row)
    return spec, science, case, output, folder


def _bb_certificate(science, case):
    ref = science["original_source_certificate"]
    probe._checked_files({ref["path"]: ref["sha256"]})
    bb = json.loads(Path(ref["path"]).read_text())
    _require(bb.get("passed") is True and bb.get("source_assets_spec_science_unchanged") is True
             and bb.get("test_enabled") is False, "Original BB certificate is unqualified")
    value = bb["roots"][case["root"]]
    _require(value.get("passed") is True and value.get("source_P0_linear_certificate_passed") is True
             and type(value.get("cells")) is int and value["cells"] == case["cells"],
             "Own BB source/reference certificate is unqualified")
    return value


def _bind_reference(source, native, reference, recipe, certified):
    _require(_exact(source, certified["source_context"])
             and _exact(native, certified["native_buffers"])
             and _exact(reference, certified["reference"])
             and _exact(recipe, certified["student_recipe_origin"]), "Own BB source/reference/recipe differs")
    _require(reference["current_native_M0_bitwise_equal_cached_NODE0"] is True
             and reference["actual_FP32_P0_X_Q_uniform_equal_reference"] is True,
             "New optimizer gate requires exact historical M0 and actual FP32/F64 readout; no fallback")


def _load_source(case, spec, science, evidence, stop):
    repo = Path(__file__).resolve().parents[1]
    shared = original._prepare_graph(repo, evidence, stop)
    root, graph = Path(case["source_root"]), shared["graph"]
    config = citation_search._legacy_teacher_config("cora", case["ratio"], graph, shared["train"],
                                                   (None, shared["val"]), shared["test"], "row")
    _require(_exact(config, original._config(case)) and _fingerprint(config) == case["root"],
             "Original Cora graph/masks/preprocessing changed")
    recipe = original._recipe(case)
    evidence["stage"] = "unchanged_original_cached_source"
    teacher = torch.load(root / "teacher.pt", map_location="cuda", weights_only=False)
    _require(isinstance(teacher, dict), "Malformed original teacher")
    logits = probe._tensor(teacher.get("logits"), (2708, 7), torch.float64, "Malformed Cora teacher logits")
    q = training_refined_targets(logits, case["reference_candidate"]["T"], graph["y"], shared["train"], 0)
    ghost = source_helper.candidate_controls(dict(case["reference_candidate"], method="source_linear",
        assignment_coordinates="raw_rms", source_linear_schema=1))
    original._attempt(evidence, "per_root_cached_source_calls")
    h, z, transform, hard, _, source = source_helper.cached_source(
        root, ghost, 0, shared["fresh_h"], q, config)
    original._complete(evidence, "per_root_cached_source_calls")
    _require(z.shape == (2708, 1433) and q.shape == (2708, 7) and int(hard.max()) + 1 == case["cells"],
             "Original Cora shape/budget changed")
    original._attempt(evidence, "per_root_CSR_dense_binding_checks")
    _require(_seal(graph["adj"]) == _seal(shared["dense"].to_sparse_csr()), "Original CSR/dense values differ")
    original._complete(evidence, "per_root_CSR_dense_binding_checks")
    base = dict(root=root, graph=graph, train=shared["train"], val=shared["val"], h=h, z=z, q=q,
                hard=hard, transform=transform, source=source)
    frozen = probe._frozen_transform(transform)
    native = probe._digest(dict(H=h, z=z, Q=q, hard=hard, transform=frozen, X=graph["x"],
                               original_CSR=graph["adj"], dense_original_S=shared["dense"]))
    reference = original._reference(base, ghost, case, evidence)
    certified = _bb_certificate(science, case)
    _bind_reference(source, native, reference, recipe, certified)
    evidence["source_reference_certificate_passed"] = True
    evidence["original_recipe_origin"] = recipe
    evidence["original_source_context"] = source
    evidence["native_source_buffers"] = native
    probe._stop(stop)
    ay._count(evidence, "own_native_factor_factory")
    u, v = initialize_factors(hard, case["cells"], 32, 0)
    ay._count(evidence, "own_native_factor_factory", True)
    _require(probe._digest([u, v]) == reference["P0_parameters"], "Own current initializer differs from BB")
    ay._count(evidence, "anchor0_GEOM_factory")
    model = anchor_initial(1433, 7, 256, 0, dtype=torch.float32, device="cuda")
    ay._count(evidence, "anchor0_GEOM_factory", True)
    for value, shape in zip(model, ((256, 1433), (256,), (7, 256), (7,)), strict=True):
        probe._tensor(value, shape, torch.float32, "Malformed Cora GEOM anchor0")
        _require(value.device.type == "cuda", "Anchor0 native device differs")
    buffers = dict(base, transform=frozen, cells=case["cells"], x=graph["x"], S=shared["dense"],
                   original_S=graph["adj"], initial=(u, v), model_initial=model)
    checkpoint = root / case["reference_id"] / "condensation_0/checkpoints/step_000000.pt"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return buffers, saved, reference


def _success_counts(evidence, science):
    raw = evidence["counts"]
    actual = {k: 0 for k in science["counts_contract"]["per_successful_case"]}
    actual.update(scientific_P_trajectories=raw.get("P_update_completed", 0),
        P_updates=raw.get("P_update_completed", 0),
        source_CE_gradient_targets=raw.get("source_CE_gradient_target_completed", 0),
        synthetic_alignment_partials=raw.get("synthetic_alignment_partial_completed", 0),
        outside_original_moment_first_backwards=raw.get("original_moment_backward_completed", 0),
        model_anchor_factories=raw.get("anchor0_GEOM_factory_completed", 0)
                             + raw.get("additional_anchor_GEOM_factory_completed", 0),
        cached_reference_head_gradient_diagnostics=raw["cached_head_gradients"],
        cached_reference_outer_CE_diagnostics=raw["cached_outer_teacher_CE"])
    _require(actual == science["counts_contract"]["per_successful_case"], "Unexpected actual probe counts")
    return actual


def prepare_probe(cells, spec_path, spec_sha256, output_path, stop=lambda: False):
    """One disposable update, zero students, in a fresh native process."""
    _require(callable(stop), "Require stop callback")
    repo = Path(__file__).resolve().parents[1]
    spec, science, case, output, folder = _load_spec(cells, spec_path, spec_sha256, output_path, repo)
    candidate = dict(spec["candidate"], gradient_source_digest=_seal(spec["numerical_source"]))
    evidence = dict(passed=False, cells=cells, candidate=candidate, candidate_id=_fingerprint(candidate),
        source=spec["source"], numerical_source=spec["numerical_source"], python_version=spec["python_version"],
        scientific_preregistration=spec["scientific_preregistration"], spec_path=str(Path(spec_path).resolve()),
        spec_sha256=spec_sha256, files_sha256=spec["files_sha256"], parents=science["parents"],
        gradient_policy=_POLICY_SPEC, test_enabled=False, validation_only=True, counts={
            k: 0 for k in original._science(repo)["expected_success_counts"] if k != "P_updates"}, operation_attempts={},
        count_scope="BB reference diagnostics plus actual AY attempt/completion counters; success_counts is the scientific authority. Unused BB zero-P counter omitted.",
        stage="fresh_native_policy", source_reference_certificate_passed=False,
        no_resume_or_continuation=True, heldout_labels_scope="Immutable graph identity hashes only; never loss/accuracy/selection")
    native, primary, buffers = False, None, None
    started = time.monotonic()
    bounded = lambda: stop() or time.monotonic() - started >= 300
    try:
        probe._stop(bounded)
        _require(torch.get_num_threads() == 4 and not torch.cuda.is_initialized(), "Require threads4/fresh CUDA worker")
        environment = probe._native("cuda")
        native = True
        torch.cuda.reset_peak_memory_stats()
        _require(torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory == original.CUDA_CAPACITY,
                 "Frozen physical GPU capacity changed")
        evidence["native_environment"] = dict(environment, python_version=platform.python_version(), threads=4)
        buffers, saved, reference = _load_source(case, spec, science, evidence, bounded)
        evidence["source_buffer_digest_before"] = _source_buffer_digest(buffers)
        evidence["anchor0_model_digest_before"] = probe._digest(buffers["model_initial"])
        evidence["stage"] = "unchanged_AY_one_update"
        states, targets = _one_update(buffers, saved, reference, evidence, bounded)
        evidence["source_buffer_digest_after"] = _source_buffer_digest(buffers)
        _require(evidence["source_buffer_digest_before"] == evidence["source_buffer_digest_after"],
                 "Immutable original source buffers changed")
        context = dict(schema=1, candidate=candidate, source=spec["source"], numerical_source=spec["numerical_source"],
            scientific_preregistration=spec["scientific_preregistration"], spec_sha256=spec_sha256,
            native_environment=evidence["native_environment"], case=case, original_source=buffers["source"],
            reference=reference, original_recipe_origin=evidence["original_recipe_origin"],
            source_buffers=evidence["source_buffer_digest_before"], target_model_digest=evidence["target_model_digest_before"],
            gradient_policy=_POLICY_SPEC, no_resume_or_continuation=True)
        probe._stop(bounded)
        _preserve(spec, Path(spec_path).resolve(), spec_sha256, science)
        probe._runtime_precision_guard()
        folder.mkdir(parents=True, exist_ok=False)
        payloads = {"source_gradient_targets.pt": dict(context=probe.cpu_state(context), targets=targets)}
        for step, state in states.items():
            payloads[f"step_{step:06d}.pt"] = dict(state, context=probe.cpu_state(context))
        for name, payload in payloads.items():
            probe._atomic(folder / name, probe._attach(payload), True)
        evidence.update(checkpoints_sha256={str((folder / name).resolve()): _sha(folder / name) for name in payloads},
            context_digest=_seal(context), reference=reference,
            alignment_J0=float(states[0]["alignment"]["loss"]), alignment_J1=float(states[1]["alignment"]["loss"]),
            own_scale=states[0]["scale"], conservation_by_step={k: s["conservation"] for k, s in states.items()},
            success_counts=_success_counts(evidence, science))
        evidence["passed"] = True
    except BaseException as error:
        primary = error
        evidence.update(error_type=type(error).__name__, error=str(error), failed_stage=evidence["stage"],
            failed_ephemeral_target_scope="Unchanged AY locals are not retained on every helper failure; failed evidence never qualifies or resumes")
    finally:
        if buffers is not None and "source_buffer_digest_before" in evidence:
            try:
                evidence["source_buffer_digest_after"] = _source_buffer_digest(buffers)
                evidence["anchor0_model_digest_after"] = probe._digest(buffers["model_initial"])
                _require(evidence["source_buffer_digest_before"] == evidence["source_buffer_digest_after"]
                         and evidence["anchor0_model_digest_before"] == evidence["anchor0_model_digest_after"],
                         "Immutable native source/model0 buffers changed")
            except BaseException as error:
                evidence.update(passed=False, native_source_preservation_error=repr(error))
                if primary is None:
                    primary = error
        try:
            _preserve(spec, Path(spec_path).resolve(), spec_sha256, science)
            evidence.update(source_unchanged=True, source_assets_spec_science_unchanged=True)
        except BaseException as error:
            evidence.update(passed=False, source_unchanged=False, source_assets_spec_science_unchanged=False,
                            preservation_error=repr(error))
            if primary is None:
                primary = error
        if native:
            try:
                torch.cuda.synchronize()
                capacity = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
                allocated, reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
                evidence.update(CUDA_peak_allocated_bytes=allocated, CUDA_peak_reserved_bytes=reserved, CUDA_total_bytes=capacity)
                _require(capacity == original.CUDA_CAPACITY and 0 <= allocated <= capacity and 0 <= reserved <= capacity
                         and torch.get_num_threads() == 4, "Native capacity/memory/threads changed")
                probe._runtime_precision_guard()
            except BaseException as error:
                evidence.update(passed=False, native_finalization_error=repr(error))
                if primary is None:
                    primary = error
        evidence["seconds"] = time.monotonic() - started
        if evidence["seconds"] > 300:
            error = ValueError("Runtime probe exceeded300seconds")
            evidence.update(passed=False, deadline_error=str(error))
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
