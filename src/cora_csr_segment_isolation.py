"""Readonly fixed-anchor CSR explicit-adjoint isolation; no backend fallback."""
import json
import math
import platform
import time
import traceback
from importlib.metadata import version as distribution_version
from pathlib import Path

import torch

from src import cora_csr_source_gradient as bi
from src import csr_segment_adjoint as explicit
from src import finite_student_probe as probe
from src.citation_source_preflight import _exact, _write_new
from src.research_loop import implementation_provenance

SCIENCE = "Cora70_original_CSR_weighted_segment_fixedanchor0_scientific_stageBM_v1.json"
SCIENCE_SHA = "37c5581f3ee9220d8217050c05936f265da26c4931e6922bc00b8660a6d0522c"
SCIENCE_REVIEW = "Cora70_original_CSR_weighted_segment_fixedanchor0_independent_scientific_review_stageBM_v1.json"
SCIENCE_REVIEW_SHA = "bb5084cdfb98215e52d5e6f4dd4c797381e7210008b72c36a8950cf73985f4fd"
_load_source, _source_digest, anchor_initial = bi._load_source, bi._source_digest, bi.anchor_initial
_require, _sha, _seal, _stop = probe._require, probe._sha, probe._seal, probe._stop
_FIELDS = {"schema", "fixed", "source", "numerical_source", "python_version", "files_sha256",
           "scientific_preregistration", "artifacts_sha256", "output_root", "pass_outputs", "target_outputs", "comparison_policy"}
_BACKEND = "original_CSR_coefficients_rank2_weighted_segment_SUM"


def numerical_source():
    value = bi.numerical_source()
    for name in ("csr_segment_adjoint.py", "cora_csr_segment_isolation.py"):
        _require(value["files"].get(name) == _sha(Path(__file__).with_name(name)), "Owned explicit-adjoint source changed")
    return value


def _science(repo):
    path = repo / "results/proposals" / SCIENCE
    probe._checked_files({str(path): SCIENCE_SHA})
    return json.loads(path.read_text())


def _paths(repo):
    return [Path(p) for p in _science(repo)["original_files_sha256"]]


def _artifact_paths(pins, science, repo):
    owned = {(repo / p).resolve() for p in science["helper_boundary"]["next_owned_sources"] + science["helper_boundary"]["next_owned_tests"]}
    owned.add((repo / "src/research_loop.py").resolve())
    _require(isinstance(pins, dict) and pins and all(type(p) is str and Path(p).is_absolute()
             and (Path(p).resolve().is_relative_to(repo / "results") or Path(p).resolve() in owned) for p in pins),
             "Require reviewed results or exact five owned artifact paths")


def _preserve(spec, path, checksum, science, repo):
    _require(_sha(path) == checksum, "Execution spec changed")
    for pins in (spec["files_sha256"], spec["artifacts_sha256"],
                 {spec["scientific_preregistration"]["path"]: SCIENCE_SHA,
                  str(repo / "results/proposals" / SCIENCE_REVIEW): SCIENCE_REVIEW_SHA},
                 {p["path"]: p["sha256"] for p in science["parents"].values()},
                 {str(repo / p): s for p, s in science["old_tests_sha256"].items()},
                 {str(repo / p): s for p, s in science["source_before"]["files"].items() if p != "src/research_loop.py"}):
        probe._checked_files(pins)
    version = science["version_binding"]["torch_version_file"]
    probe._checked_files({version["path"]: version["sha256"]})
    expected = science["version_binding"]["installed_distribution_versions"]
    _require({k: distribution_version(k) for k in expected} == expected, "Distribution versions changed")
    _require(implementation_provenance() == spec["source"] and numerical_source() == spec["numerical_source"]
             and platform.python_version() == spec["python_version"] and torch.get_num_threads() == 4,
             "Source/Git/versions/Python/threads changed")


def _load_spec(path, checksum):
    repo, path = Path(__file__).resolve().parents[1], Path(path).resolve()
    _require(type(checksum) is str and len(checksum) == 64 and path.is_file()
             and path.is_relative_to(repo / "results/proposals") and _sha(path) == checksum, "Require pinned prospective spec")
    science, spec = _science(repo), json.loads(path.read_text())
    _require(isinstance(spec, dict) and set(spec) == _FIELDS and type(spec["schema"]) is int and spec["schema"] == 1,
             "Unknown/malformed isolation spec")
    for key in ("fixed", "files_sha256", "output_root", "pass_outputs", "target_outputs", "comparison_policy"):
        expected = science["original_files_sha256"] if key == "files_sha256" else science[key]
        _require(_exact(spec[key], expected), "Changed fixed isolation spec: " + key)
    _require(_exact(spec["scientific_preregistration"], dict(path=str(repo / "results/proposals" / SCIENCE), sha256=SCIENCE_SHA)),
             "Wrong scientific reference")
    _artifact_paths(spec["artifacts_sha256"], science, repo)
    _require(isinstance(spec["source"], dict) and isinstance(spec["source"].get("files"), dict)
             and set(spec["source"]["files"]) == set(science["source_before"]["files"]) | set(science["helper_boundary"]["next_owned_sources"])
             and spec["python_version"] == science["version_binding"]["python_version"]
             and spec["numerical_source"].get("versions") == science["version_binding"]["runtime_versions_from_pinned_native_metadata"],
             "Require exact promoted73-source/version binding")
    _preserve(spec, path, checksum, science, repo)
    return spec, science


def _request(pass_index, previous, output, spec):
    _require(type(pass_index) is int and pass_index in (1, 2), "Require pass1 or2")
    _require((pass_index == 1 and previous is None) or (pass_index == 2 and type(previous) is str and len(previous) == 64),
             "Wrong previous-pass protocol")
    output, target = Path(output).resolve(), Path(spec["target_outputs"][str(pass_index)])
    _require(str(output) == spec["pass_outputs"][str(pass_index)] and output.parent == target.parent
             and not output.exists() and not output.parent.exists(), "Require exclusive declared pass namespace")
    return output, target


def _previous(spec, science, evidence, checksum):
    path = Path(spec["pass_outputs"]["1"])
    _require(_sha(path) == checksum, "Previous receipt SHA differs")
    first = json.loads(path.read_text())
    for key in ("source", "numerical_source", "python_version", "scientific_preregistration", "spec_path", "spec_sha256"):
        _require(_exact(first.get(key), evidence[key]), "Previous provenance differs: " + key)
    _require(first.get("passed") is True and type(first.get("pass_index")) is int and first["pass_index"] == 1
             and first.get("operation") == "prepare_isolation" and first.get("source_backend") == _BACKEND
             and first.get("source_assets_spec_science_unchanged") is True and first.get("source_reference_provenance_passed") is True
             and first.get("within_explicit_adjoint_exact_repeat_passed") is False
             and first.get("fixedanchor0_explicit_adjoint_backend_qualified") is False
             and first.get("test_enabled") is False and first.get("validation_only") is True
             and _exact(first.get("success_counts"), science["counts_contract"]["planned_per_fully_successful_worker"]),
             "First local receipt is not accepted")
    target = Path(spec["target_outputs"]["1"])
    _require(first.get("raw_cache_path") == str(target) and _sha(target) == first.get("raw_cache_sha256"), "First raw cache differs")
    return first, path, target


def _native_inputs(buffers, anchor):
    explicit._inputs(anchor, buffers["x"], buffers["S"], buffers["q"])
    _require(buffers["x"].device == buffers["q"].device == buffers["S"].device == torch.device("cuda:0")
             and buffers["x"].dtype == buffers["S"].dtype == torch.float32 and buffers["q"].dtype == torch.float64
             and tuple(buffers["x"].shape) == (2708, 1433) and tuple(buffers["q"].shape) == (2708, 7)
             and buffers["S"]._nnz() == 13264 and all(p.device == buffers["x"].device and p.dtype == torch.float32 for p in anchor),
             "Require exact native FP32 original CSR/X/anchor and FP64 Q; no fallback")


def _isolated(source, boundaries, evidence):
    values = {}
    for name, input_name, cotangent_name in (("B", "U2", "D"), ("C", "U1", "E")):
        leaf = boundaries[input_name].detach().clone().requires_grad_(True)
        output = explicit._invoke(evidence, "isolated_original_CSR_SpMM_forward_calls", torch.sparse.mm, source, leaf)
        value = explicit._invoke(evidence, "isolated_original_CSR_right_input_VJP_calls", torch.autograd.grad,
                                 output, leaf, grad_outputs=boundaries[cotangent_name])[0].detach().clone()
        _require(bool(torch.isfinite(output).all()) and bool(torch.isfinite(value).all()), "Nonfinite isolated sparse VJP")
        values[name] = value
    return values


def _comparison(boundaries, isolated):
    rows = []
    for name in ("B", "C"):
        reference, observed = boundaries[name].double().reshape(-1), isolated[name].double().reshape(-1)
        nr, no = torch.linalg.vector_norm(reference), torch.linalg.vector_norm(observed)
        _require(bool(torch.isfinite(nr)) and float(nr) > 0 and bool(torch.isfinite(no)), "Invalid descriptive norms")
        row = dict(boundary=name, explicit_norm=float(nr), isolated_norm=float(no), isolated_exact_zero_norm=bool(no == 0),
                   max_abs=float((observed - reference).abs().max()), relative_L2=float(torch.linalg.vector_norm(observed - reference) / nr),
                   raw_cosine=None if bool(no == 0) else float(torch.dot(reference, observed) / (nr * no)))
        _require(all(v is None or type(v) in (str, bool) or math.isfinite(v) for v in row.values()), "Nonfinite descriptive metric")
        rows.append(row)
    return dict(status="DESCRIPTIVE_ONLY_NO_EQUIVALENCE_BOUND_OR_CAUSE", boundaries=rows,
                equivalence_thresholds=None, cross_algorithm_equivalence_claim=False)


def _context(spec, evidence):
    return dict(source=spec["source"], numerical_source=spec["numerical_source"], scientific_preregistration=spec["scientific_preregistration"],
                spec_sha256=evidence["spec_sha256"], source_context=evidence["source_context"],
                native_source_buffers=evidence["native_source_buffers_before"], source_backend=_BACKEND)


def _capture(spec, target, context, pass_index, common, isolated, evidence):
    target.parent.mkdir(parents=True, exist_ok=False)
    with target.open("xb") as stream:
        torch.save(probe._attach(dict(context=context, pass_index=pass_index, common=common, isolated=isolated)), stream)
    evidence.update(raw_cache_path=str(target), raw_cache_sha256=_sha(target), raw_cache_is_qualification=False)


def _repeat(spec, science, common, evidence, previous):
    first, path, target = _previous(spec, science, evidence, previous)
    _require(_exact(first.get("native_environment"), evidence["native_environment"]), "Previous native environment differs")
    payload = explicit._invoke(evidence, "previous_explicit_target_cache_loads", torch.load, target, map_location="cuda", weights_only=False)
    bi._sealed_payload(payload)
    _require(type(payload.get("pass_index")) is int and payload["pass_index"] == 1 and _exact(payload.get("context"), _context(spec, evidence))
             and probe._digest(payload.get("common")) == first.get("common_numerical_digest")
             and _seal(payload["common"]) == first.get("common_numerical_seal"), "Previous sealed common payload differs")
    evidence.update(previous_receipt_path=str(path), previous_receipt_sha256=previous,
                    previous_raw_cache_path=str(target), previous_raw_cache_sha256=_sha(target),
                    previous_common_numerical_digest=first["common_numerical_digest"])
    _require(_seal(common) == first["common_numerical_seal"], "Within explicit-adjoint exact common repeat differs")
    evidence.update(within_explicit_adjoint_exact_repeat_passed=True, fixedanchor0_explicit_adjoint_backend_qualified=True)


def _retained(buffers, evidence):
    evidence["native_source_buffers_after"] = _source_digest(buffers)
    _require(_exact(evidence["native_source_buffers_after"], evidence["native_source_buffers_before"]), "Source buffers changed")
    for key, before in (("anchor", "anchor_digest_before"), ("common", "common_numerical_digest"), ("isolated", "isolated_VJP_digest")):
        if key in buffers:
            value = probe._digest(buffers[key])
            evidence[key + "_digest_after"] = value
            _require(value == evidence[before], "Completed " + key + " buffers changed")


def prepare_isolation(pass_index, spec_path, spec_sha256, output_path, previous_sha256=None, stop=lambda: False):
    _require(callable(stop), "Require stop callback")
    repo, spec_path = Path(__file__).resolve().parents[1], Path(spec_path).resolve()
    spec, science = _load_spec(spec_path, spec_sha256)
    output, target = _request(pass_index, previous_sha256, output_path, spec)
    started = time.monotonic()
    bounded = lambda: stop() or time.monotonic() - started >= 300
    evidence = dict(passed=False, operation="prepare_isolation", pass_index=pass_index, source=spec["source"],
        numerical_source=spec["numerical_source"], python_version=spec["python_version"], scientific_preregistration=spec["scientific_preregistration"],
        spec_path=str(spec_path), spec_sha256=spec_sha256, files_sha256=spec["files_sha256"], source_backend=_BACKEND,
        validation_only=True, test_enabled=False, counts={k: 0 for k in science["counts_contract"]["planned_per_fully_successful_worker"]}, operation_attempts={},
        count_scope="Actual interface attempts/returns. Loader H2 and four output blocks are integrated after full return; unfinished internals unknown and completions lower bounds.",
        within_explicit_adjoint_exact_repeat_passed=False, fixedanchor0_explicit_adjoint_backend_qualified=False,
        old_BI_original_autograd_backend_qualified=False, old_BK_backend_qualified=False, three_anchor_target_backend_qualified=False, large_trigger_satisfied=False,
        _stop=bounded, stage="fresh_native_precision")
    evidence["_guard"] = lambda: (_stop(bounded), _preserve(spec, spec_path, spec_sha256, science, repo), probe._runtime_precision_guard())
    native, buffers, primary, rng = False, None, None, None
    try:
        _stop(bounded)
        if pass_index == 2:
            _previous(spec, science, evidence, previous_sha256)
        _require(not torch.cuda.is_initialized(), "Require fresh CUDA worker")
        initializer = probe._native("cuda")
        native = True
        torch.cuda.reset_peak_memory_stats()
        capacity = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
        expected = science["native_policy"]
        _require(capacity == expected["CUDA_total_bytes"] and initializer["GPU"] == expected["GPU"]
                 and initializer["cuda_runtime"] == expected["cuda_runtime"] and torch.get_num_threads() == 4, "Native runtime/capacity/threads differ")
        evidence["native_environment"] = dict(source_backend=_BACKEND, actual_source_layout="torch.sparse_csr", actual_transpose_layout="torch.sparse_csr",
            threads=torch.get_num_threads(), python_version=platform.python_version(), inherited_precision_initializer=initializer,
            inherited_graph_label_scope="precision initializer only; actual four model products use weighted rank2 segment SUM with original S/T coefficients")
        rng = (torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone())
        buffers = _load_source(spec, science, evidence, repo)
        evidence["stage"] = "fixedanchor0_explicit_adjoint"
        anchor = explicit._invoke(evidence, "private_GEOM_anchor_factory_calls", anchor_initial, 1433, 7, 256, 0, dtype=torch.float32, device="cuda")
        _native_inputs(buffers, anchor)
        buffers["anchor"] = anchor
        evidence["anchor_digest_before"] = probe._digest(anchor)
        _require(evidence["anchor_digest_before"] == science["source_binding"]["anchor0_parameter_descriptors"], "Published anchor0 differs")
        result = explicit._invoke(evidence, "explicit_source_parameter_gradient_assemblies", explicit.gradient_boundaries,
                                  anchor, buffers["x"], buffers["S"], buffers["q"], evidence)
        packet = explicit._invoke(evidence, "source_target_packet_assemblies", explicit.target_packet, result, evidence)
        isolated = _isolated(buffers["S"], result["boundaries"], evidence)
        common = dict(source_buffers=dict(H=buffers["h"], z=buffers["z"], Q=buffers["q"], hard=buffers["hard"], transform=buffers["transform"],
                                          X=buffers["x"], original_CSR=buffers["S"]),
                      S=buffers["S"], T=result["transpose"], anchor=anchor, boundaries=result["boundaries"], CE=result["loss"], target=packet)
        buffers.update(common=common, isolated=isolated)
        evidence.update(common_numerical_digest=probe._digest(common), common_numerical_seal=_seal(common),
                        isolated_VJP_digest=probe._digest(isolated), descriptive_comparison=_comparison(result["boundaries"], isolated))
        _retained(buffers, evidence)
        evidence["_guard"]()
        _capture(spec, target, _context(spec, evidence), pass_index, common, isolated, evidence)
        if pass_index == 2:
            evidence["stage"] = "exact_common_repeat"
            _repeat(spec, science, common, evidence, previous_sha256)
        evidence["_guard"]()
        evidence["passed"] = True
    except BaseException as error:
        primary = error
        evidence.update(passed=False, error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc(),
            partial_internal_work_scope="Only actual completed interface calls are known. Failed primitive internals remain unknown; any raw capture remains unqualified.")
    finally:
        try:
            _preserve(spec, spec_path, spec_sha256, science, repo)
            if buffers is not None:
                _retained(buffers, evidence)
            if rng is not None:
                _require(torch.equal(rng[0], torch.get_rng_state()) and torch.equal(rng[1], torch.cuda.get_rng_state()), "Global RNG changed")
                evidence["global_RNG_unchanged"] = True
            evidence["source_assets_spec_science_unchanged"] = True
        except BaseException as error:
            evidence.update(passed=False, source_assets_spec_science_unchanged=False, preservation_error=repr(error))
            primary = primary or error
        if native:
            try:
                torch.cuda.synchronize()
                capacity = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
                allocated, reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
                evidence.update(CUDA_total_bytes=capacity, CUDA_peak_allocated_bytes=allocated, CUDA_peak_reserved_bytes=reserved)
                _require(capacity == 8316977152 and 0 <= allocated <= capacity and 0 <= reserved <= capacity, "Native memory/capacity differs")
                probe._runtime_precision_guard()
            except BaseException as error:
                evidence.update(passed=False, native_finalization_error=repr(error))
                primary = primary or error
        evidence["seconds"] = time.monotonic() - started
        if evidence["seconds"] > 300:
            error = ValueError("Isolation exceeded300seconds")
            evidence.update(passed=False, deadline_error=str(error))
            primary = primary or error
        if evidence["passed"]:
            expected = science["counts_contract"]["planned_per_fully_successful_worker"]
            actual = {k: evidence["counts"][k] for k in expected}
            if actual != expected:
                primary = ValueError("Unexpected actual isolation counts")
                evidence.update(passed=False, counter_error=str(primary))
            else:
                evidence["success_counts"] = actual
        if not evidence["passed"]:
            evidence.update(within_explicit_adjoint_exact_repeat_passed=False, fixedanchor0_explicit_adjoint_backend_qualified=False)
        evidence.pop("_stop")
        evidence.pop("_guard")
        try:
            _write_new(output, evidence)
        except BaseException:
            if primary is not None:
                raise primary
            raise
    if primary is not None:
        raise primary
    return dict(evidence, evidence_path=str(output), evidence_sha256=_sha(output), validation_only=True)
