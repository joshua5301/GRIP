"""One disposable whole-layer P update using the accepted BN source targets.

Current source/origin checks and the historical target-cache context are distinct.
No source gradient regeneration, students, resume or continuation is available.
"""
import json
import math
import platform
import time
from importlib.metadata import version as distribution_version
from pathlib import Path

import torch
from src.whole_layer_gradient_alignment import POLICY, moment_partials

from src import citeseer_finite_student_v2 as inherited
from src import cora_csr_segment_ensemble as bn
from src import cora_gradient_probe as bc
from src import finite_student_probe as probe
from src.citation_macro_probe import _material
from src.citation_source_preflight import _exact, _write_new
from src.citeseer_confirmation_source import _reference_pins
from src.research_loop import implementation_provenance
from src.sweep_utils import representative

SCIENCE = "Cora70_two_whole_layer_one_update_scientific_stageBQ_v1.json"
SCIENCE_SHA = "ac99118497b6e4756d22f0e010da98e10a104921be2f03ed1553b1b371de784a"
SCIENCE_REVIEW = "Cora70_two_whole_layer_one_update_independent_scientific_review_stageBQ_v1.json"
SCIENCE_REVIEW_SHA = "edfb16a4dcbb1bcedbbf5c1a9a7bfca14d3e7e2b94a39c4823c01fcbcc1709ef"
GROUPING_REVIEW = "results/implementation_drafts/cora_whole_layer_grouping_v1/independent_grouping_static_review_v1.json"
GROUPING_REVIEW_SHA = "8e364e925a88aa6a5c6b3d91da905909e6d6e472d6153b99daae6ca2fd245d23"
HELPER_SHA = "38c649ddeb004982dd5b1c4f6f128232c0652a171686844c549a4ff817b51fa7"
_FIELDS = {"schema", "fixed", "case", "source", "numerical_source", "python_version", "files_sha256",
           "scientific_preregistration", "artifacts_sha256", "output_root", "output_path", "raw_cache_path",
           "BN_source", "grouping_policy"}
_POLICY_SPEC = json.loads(json.dumps(POLICY))
_require, _sha, _seal, _stop = probe._require, probe._sha, probe._seal, probe._stop
_load_source, _source_buffer_digest = bc._load_source, bc._source_buffer_digest
_count = inherited._count


def numerical_source():
    result = bc.numerical_source()
    _require(result["files"].get("whole_layer_gradient_alignment.py") == HELPER_SHA
             and result["files"].get("cora_whole_layer_probe.py") == _sha(__file__),
             "Whole-layer helper/module source changed")
    return result


def _science(repo):
    path = repo / "results/proposals" / SCIENCE
    probe._checked_files({str(path): SCIENCE_SHA})
    return json.loads(path.read_text())


def _paths(repo):
    return [Path(p) for p in _science(repo)["original_files_sha256"]]


def _preserve(spec, path, checksum, science, repo):
    _require(_sha(path) == checksum, "Execution spec changed")
    for pins in (spec["files_sha256"], spec["artifacts_sha256"], _reference_pins(science),
                 {spec["scientific_preregistration"]["path"]: SCIENCE_SHA,
                  str(repo / "results/proposals" / SCIENCE_REVIEW): SCIENCE_REVIEW_SHA,
                  str(repo / GROUPING_REVIEW): GROUPING_REVIEW_SHA},
                 {str(repo / p): h for p, h in science["source_before"]["files"].items() if p != "src/research_loop.py"},
                 {str(repo / p): h for p, h in science["old_tests_sha256"].items()}):
        probe._checked_files(pins)
    expected = science["version_binding"]["installed_distribution_versions"]
    _require({k: distribution_version(k) for k in expected} == expected, "Distribution versions changed")
    _require(implementation_provenance() == spec["source"] and numerical_source() == spec["numerical_source"]
             and platform.python_version() == spec["python_version"] and torch.get_num_threads() == 4,
             "Source/Git/numerical versions/Python/threads changed")


def _spec_controls(spec, science, repo):
    _require(isinstance(spec, dict) and set(spec) == _FIELDS and type(spec["schema"]) is int and spec["schema"] == 1,
             "Unknown or malformed whole-layer probe spec")
    for key in ("fixed", "case", "output_root", "output_path", "raw_cache_path", "BN_source"):
        _require(_exact(spec[key], science[key]), "Changed frozen spec field: " + key)
    _require(_exact(spec["grouping_policy"], _POLICY_SPEC)
             and _exact(spec["files_sha256"], science["original_files_sha256"])
             and _exact(spec["scientific_preregistration"], dict(path=str(repo / "results/proposals" / SCIENCE), sha256=SCIENCE_SHA)),
             "Grouping policy/assets/science changed")
    owned = {(repo / p).resolve() for p in science["source_protection"]["owned"]}
    pins = spec["artifacts_sha256"]
    _require(isinstance(pins, dict) and pins and all(type(p) is str and Path(p).is_absolute()
             and (Path(p).resolve().is_relative_to(repo / "results") or Path(p).resolve() in owned) for p in pins),
             "Require results or exact five owned artifact paths")
    _require(isinstance(spec["source"], dict) and isinstance(spec["source"].get("files"), dict)
             and set(spec["source"]["files"]) == set(science["source_before"]["files"]) | {
                 p for p in science["source_protection"]["owned"] if p.startswith("src/")}
             and spec["python_version"] == science["version_binding"]["python_version"]
             and spec["numerical_source"].get("versions") == science["version_binding"]["runtime_versions_from_pinned_native_metadata"],
             "Require actual promoted78 source/version binding")


def _load_spec(path, checksum):
    repo, path = Path(__file__).resolve().parents[1], Path(path).resolve()
    _require(type(checksum) is str and len(checksum) == 64 and path.is_file()
             and path.is_relative_to(repo / "results/proposals") and _sha(path) == checksum, "Require exact frozen prospective spec")
    spec, science = json.loads(path.read_text()), _science(repo)
    _spec_controls(spec, science, repo)
    _preserve(spec, path, checksum, science, repo)
    _require((repo / "data/cora/processed/data.pt").is_file(), "Require existing processed Cora data")
    bc.original._config(spec["case"])
    bc.original._recipe(spec["case"])
    return spec, science


def _request(output, spec):
    output, raw = Path(output).resolve(), Path(spec["raw_cache_path"])
    _require(str(output) == spec["output_path"] and raw.is_absolute() and raw.parent == output.parent
             and str(raw.resolve()) == str(raw) and not output.exists() and not raw.exists(),
             "Require absent declared evidence/raw namespace; partial cache never resumes")
    return output, raw


def _cache_metadata(payload, first, second, oldspec, bindings):
    _require(type(payload.get("pass_index")) is int and payload["pass_index"] == 2,
             "Require the accepted complete BN pass2 cache")
    for row in (first, second):
        _require(row.get("passed") is True and row.get("source_assets_spec_science_unchanged") is True
                 and row.get("source_reference_provenance_passed") is True and row.get("test_enabled") is False
                 and row.get("source_backend") == bindings["source_backend"], "Unqualified BN source receipt")
    _require(second.get("within_three_anchor_segment_exact_repeat_passed") is True
             and second.get("three_anchor_segment_backend_qualified") is True
             and _exact(first.get("common_numerical_digest"), second.get("common_numerical_digest"))
             and first.get("common_numerical_seal") == second.get("common_numerical_seal"), "BN strict repeat is not accepted")
    _require(second.get("raw_cache_path") == bindings["target_caches"]["2"]["path"]
             and second.get("raw_cache_sha256") == bindings["target_caches"]["2"]["sha256"]
             and second.get("spec_sha256") == bindings["spec"]["sha256"]
             and _exact(second.get("source"), oldspec["source"])
             and _exact(payload.get("context"), bn._context(oldspec, second)), "BN historical source/context/cache lineage differs")
    _require(probe._digest(payload.get("common")) == second["common_numerical_digest"]
             and _seal(payload["common"]) == second["common_numerical_seal"], "BN complete common payload differs")


def _cached_targets(buffers, spec, science, evidence, stop):
    bindings = science["BN_source"]
    oldspec = json.loads(Path(bindings["spec"]["path"]).read_text())
    oldscience = json.loads(Path(bindings["science"]["path"]).read_text())
    first, second = (json.loads(Path(bindings["receipts"][str(i)]["path"]).read_text()) for i in (1, 2))
    path = Path(bindings["target_caches"]["2"]["path"])
    _stop(stop)
    _count(evidence, "accepted_BN_cache_load")
    payload = torch.load(path, map_location="cuda", weights_only=False)
    _count(evidence, "accepted_BN_cache_load", True)
    bn.bi._sealed_payload(payload)
    _cache_metadata(payload, first, second, oldspec, bindings)
    bn._finite_tree(payload["common"])
    digest = probe._digest(payload["common"])
    bn._common_contract(digest, oldscience)
    shared = probe._digest(dict(H=buffers["h"], z=buffers["z"], Q=buffers["q"], hard=buffers["hard"],
        transform=buffers["transform"], X=buffers["x"], original_CSR=buffers["original_S"]))
    _require(shared == bindings["native_buffer_descriptors"] == digest["shared"]["source_buffers"],
             "Current source buffers differ from accepted BN targets")
    rows = payload["common"]["anchors_by_seed"]
    _require(list(rows) == ["0", "1", "2"], "Require ordered complete BN anchors")
    anchors, targets = [], []
    for seed, row in rows.items():
        anchor, gradient = row["anchor"], row["target"]["gradients"]
        _require(probe._digest(anchor) == bindings["parameter_descriptors_by_seed"][seed]
                 and len(anchor) == 4 and len(gradient) == 4, "Frozen BN anchor/target shape differs")
        for parameter, target in zip(anchor, gradient, strict=True):
            _require(parameter.device.type == target.device.type == "cuda" and parameter.dtype == target.dtype == torch.float32
                     and parameter.shape == target.shape and not parameter.requires_grad and not target.requires_grad,
                     "Require detached native FP32 BN parameters/gradients")
        anchors.append(anchor)
        targets.append(gradient)
    _require(all(torch.equal(a, b) for a, b in zip(anchors[0], buffers["model_initial"], strict=True)),
             "Current private seed0 anchor differs from BN")
    evidence["BN_cache"] = dict(path=str(path), sha256=_sha(path), historical_context=payload["context"],
        common_digest=digest, common_seal=second["common_numerical_seal"],
        current_source_buffers_equal=True, historical_context_validated=True,
        anchor_digest_before=probe._digest(anchors), target_digest_before=probe._digest(targets),
        accepted_three_anchor_backend=bindings["source_backend"], no_target_regeneration=True)
    return payload, tuple(anchors), tuple(targets)


def _one_update(buffers, saved, reference, anchors, targets, evidence, stop):
    probe._runtime_precision_guard()
    parameters = [p.detach().clone().requires_grad_() for p in buffers["initial"]]
    _count(evidence, "P_optimizer_constructor")
    optimizer = torch.optim.Adam(parameters, lr=.01, betas=(.9, .999), eps=1e-12, weight_decay=0, foreach=False, fused=False)
    _count(evidence, "P_optimizer_constructor", True)
    _count(evidence, "P0_connected_moment")
    moments0 = probe._moments(buffers, parameters)
    _count(evidence, "P0_connected_moment", True)
    _require(torch.equal(moments0.detach(), probe._tensor(saved.get("moments"), moments0.shape, torch.float64,
             "Malformed original NODE0 moments").to(moments0)), "Connected P0 differs from own cached NODE0")
    _count(evidence, "material_certificate")
    _, _, _, logits0, conservation0 = _material(moments0.detach(), buffers, parameters)
    _count(evidence, "material_certificate", True)
    _count(evidence, "P0_representative_inputs")
    X, Q, mass = representative(moments0.detach(), probe._transform(buffers), 1433, moments0.device)
    uniform = torch.full_like(mass, 1/len(mass))
    _count(evidence, "P0_representative_inputs", True)
    _require(reference["actual_FP32_P0_X_Q_uniform_equal_reference"] is True
             and probe._digest([X, Q, uniform]) == reference["student_inputs"], "Actual FP32 P0 inputs differ")
    _stop(stop)
    _count(evidence, "grouping_partial")
    result0 = moment_partials(moments0, buffers["transform"], anchors, targets)
    _count(evidence, "grouping_partial", True)
    scale = float(result0["loss"])
    _require(math.isfinite(scale) and scale > 0, "Own whole-layer J0 must be finite and positive")
    _stop(stop)
    _count(evidence, "original_moment_backward")
    moments0.backward(result0["moment_gradient"]/scale)
    _count(evidence, "original_moment_backward", True)
    _require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in parameters), "Nonfinite/missing factor gradient")
    gradients = [p.grad.detach().clone() for p in parameters]
    expected, first, second = inherited._adam_step(parameters, [torch.zeros_like(p) for p in parameters],
        [torch.zeros_like(p) for p in parameters], gradients, 1)
    inherited._optimizer(optimizer.state_dict(), parameters, first, second, 0)
    state0 = probe.cpu_state(dict(step=0, parameters=parameters, moments=moments0, alignment=result0, scale=scale,
        scaled_factor_gradients=gradients, optimizer=optimizer.state_dict(), raw_logits=logits0, conservation=conservation0))
    _stop(stop)
    probe._runtime_precision_guard()
    _count(evidence, "P_update")
    optimizer.step()
    _count(evidence, "P_update", True)
    _require(all(torch.equal(p.detach(), e) for p, e in zip(parameters, expected, strict=True)), "Native Adam parameter recurrence differs")
    inherited._optimizer(optimizer.state_dict(), parameters, first, second, 1)
    evidence["native_Adam_recurrence_passed"] = True
    _stop(stop)
    _count(evidence, "P1_moment")
    with torch.no_grad():
        moments1 = probe._moments(buffers, parameters).detach()
    _count(evidence, "P1_moment", True)
    _count(evidence, "material_certificate")
    _, _, _, logits1, conservation1 = _material(moments1, buffers, parameters)
    _count(evidence, "material_certificate", True)
    _count(evidence, "grouping_partial")
    result1 = moment_partials(moments1, buffers["transform"], anchors, targets)
    _count(evidence, "grouping_partial", True)
    _require(math.isfinite(float(result1["loss"])), "Terminal whole-layer J1 must be finite")
    state1 = probe.cpu_state(dict(step=1, parameters=parameters, moments=moments1, alignment=result1,
        optimizer=optimizer.state_dict(), raw_logits=logits1, conservation=conservation1, terminal_partial_diagnostic_only=True))
    evidence.update(native_P0_moments_exactly_equal_cached_NODE0=True, actual_FP32_P0_X_Q_uniform_equal_reference=True,
        P0_student_inputs=probe._digest([X, Q, uniform]), alignment_J0=scale, alignment_J1=float(result1["loss"]),
        J1_decrease_descriptive_only=float(result1["loss"]) < scale)
    return {0: state0, 1: state1}


def _retained(buffers, payload, anchors, targets, evidence):
    if buffers is not None and "source_buffer_digest_before" in evidence:
        evidence["source_buffer_digest_after"] = _source_buffer_digest(buffers)
        evidence["anchor0_digest_after"] = probe._digest(buffers["model_initial"])
        _require(evidence["source_buffer_digest_before"] == evidence["source_buffer_digest_after"]
                 and evidence["anchor0_digest_before"] == evidence["anchor0_digest_after"], "Retained source/model0 changed")
    if payload is not None:
        info = evidence["BN_cache"]
        info.update(common_digest_after=probe._digest(payload["common"]), anchor_digest_after=probe._digest(anchors),
                    target_digest_after=probe._digest(targets))
        _require(info["common_digest"] == info["common_digest_after"] and info["anchor_digest_before"] == info["anchor_digest_after"]
                 and info["target_digest_before"] == info["target_digest_after"], "Retained BN common/targets/anchors changed")


def _success_counts(evidence, science):
    raw = evidence["counts"]
    actual = {k: 0 for k in science["counts_contract"]["planned_success"]}
    actual.update(P_updates=raw.get("P_update_completed", 0), native_Adam_steps=raw.get("P_update_completed", 0),
        grouping_partials=raw.get("grouping_partial_completed", 0),
        synthetic_CE_parameter_gradients=3*raw.get("grouping_partial_completed", 0),
        outside_original_P_backwards=raw.get("original_moment_backward_completed", 0),
        accepted_BN_cache_loads=raw.get("accepted_BN_cache_load_completed", 0))
    _require(actual == science["counts_contract"]["planned_success"], "Actual completed operation counts differ")
    return actual


def prepare(spec_path, spec_sha256, output_path, stop=lambda: False):
    _require(callable(stop), "Stop callback must be callable")
    repo, spec_path = Path(__file__).resolve().parents[1], Path(spec_path).resolve()
    spec, science = _load_spec(spec_path, spec_sha256)
    output, raw = _request(output_path, spec)
    started = time.monotonic()
    bounded = lambda: stop() or time.monotonic()-started >= 300
    evidence = dict(passed=False, operation="prepare", cells=70, condensation_seed=0, source=spec["source"],
        numerical_source=spec["numerical_source"], python_version=spec["python_version"], scientific_preregistration=spec["scientific_preregistration"],
        spec_path=str(spec_path), spec_sha256=spec_sha256, grouping_policy=_POLICY_SPEC, fixed=spec["fixed"],
        case=spec["case"], test_enabled=False, validation_only=True, stage="fresh_native_policy",
        counts={k: 0 for k in bc.original._science(repo)["expected_success_counts"] if k != "P_updates"}, operation_attempts={},
        count_scope="Inherited source-certificate diagnostics are separate. Actual attempts/completions count interfaces; failed helper internals are unknown. No new source gradients or students.")
    native, primary, buffers, payload, anchors, targets = False, None, None, None, None, None
    try:
        _stop(bounded)
        _require(torch.get_num_threads() == 4 and not torch.cuda.is_initialized(), "Require threads4/fresh CUDA")
        environment = probe._native("cuda")
        native = True
        torch.cuda.reset_peak_memory_stats()
        _require(torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory == bc.original.CUDA_CAPACITY,
                 "Frozen GPU physical capacity changed")
        evidence["native_environment"] = dict(environment, python_version=platform.python_version(), threads=4,
            target_backend=spec["BN_source"]["source_backend"], inherited_graph_label_scope="precision/source-certificate only; source targets loaded unchanged from accepted BN CSR segment cache")
        buffers, saved, reference = _load_source(spec["case"], spec, science, evidence, bounded)
        evidence.update(reference=reference, source_buffer_digest_before=_source_buffer_digest(buffers),
                        anchor0_digest_before=probe._digest(buffers["model_initial"]))
        evidence["stage"] = "accepted_BN_target_cache"
        payload, anchors, targets = _cached_targets(buffers, spec, science, evidence, bounded)
        evidence["stage"] = "whole_layer_one_update"
        states = _one_update(buffers, saved, reference, anchors, targets, evidence, bounded)
        _retained(buffers, payload, anchors, targets, evidence)
        _stop(bounded)
        _preserve(spec, spec_path, spec_sha256, science, repo)
        probe._runtime_precision_guard()
        context = dict(schema=1, source=spec["source"], numerical_source=spec["numerical_source"], spec_sha256=spec_sha256,
            scientific_preregistration=spec["scientific_preregistration"], case=spec["case"], fixed=spec["fixed"],
            grouping_policy=_POLICY_SPEC, source_context=buffers["source"], source_buffers=evidence["source_buffer_digest_before"],
            reference=reference, BN_cache=evidence["BN_cache"], native_environment=evidence["native_environment"], no_resume_or_continuation=True)
        packet = probe._attach(dict(context=context, states=states))
        bn._finite_tree(packet)
        raw.parent.mkdir(parents=True, exist_ok=True)
        with raw.open("xb") as stream:
            torch.save(packet, stream)
        evidence.update(raw_cache_path=str(raw), raw_cache_sha256=_sha(raw), raw_cache_content_sha256=packet["content_sha256"],
            context_digest=_seal(context), states_digest=probe._digest(states), conservation_by_step={str(k):s["conservation"] for k,s in states.items()},
            coupled_one_update_certificate_passed=True, success_counts=_success_counts(evidence, science), passed=True)
    except BaseException as error:
        primary = error
        evidence.update(error_type=type(error).__name__, error=str(error), failed_stage=evidence["stage"])
    finally:
        try:
            _retained(buffers, payload, anchors, targets, evidence)
        except BaseException as error:
            evidence.update(passed=False, retained_buffer_preservation_error=repr(error))
            if primary is None:
                primary = error
        try:
            _preserve(spec, spec_path, spec_sha256, science, repo)
            evidence.update(source_unchanged=True, source_assets_spec_science_unchanged=True)
        except BaseException as error:
            evidence.update(passed=False, source_unchanged=False, source_assets_spec_science_unchanged=False, preservation_error=repr(error))
            if primary is None:
                primary = error
        if native:
            try:
                torch.cuda.synchronize()
                capacity = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
                allocated, reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
                evidence.update(CUDA_peak_allocated_bytes=allocated, CUDA_peak_reserved_bytes=reserved, CUDA_total_bytes=capacity)
                _require(capacity == bc.original.CUDA_CAPACITY and 0 <= allocated <= capacity and 0 <= reserved <= capacity
                         and torch.get_num_threads() == 4, "Native memory/capacity/threads changed")
                probe._runtime_precision_guard()
            except BaseException as error:
                evidence.update(passed=False, native_finalization_error=repr(error))
                if primary is None:
                    primary = error
        evidence["seconds"] = time.monotonic()-started
        if evidence["seconds"] > 300:
            error = ValueError("Whole-layer runtime probe exceeded300seconds")
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
