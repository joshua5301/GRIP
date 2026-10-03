"""One disposable coupled teacher-CE / whole-layer P update.

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

from src import citeseer_finite_student_v2 as inherited
from src import cora_csr_segment_ensemble as bn
from src import cora_gradient_probe as bc
from src import cora_whole_layer_probe as bq
from src import finite_student_probe as probe
from src.citation_macro_probe import _material
from src.citation_source_preflight import _exact, _write_new
from src.citeseer_confirmation_source import _reference_pins
from src.moments import augmented, decode_moments
from src.research_loop import implementation_provenance
from src.soft_ce_partition import (
    head_gradient,
    hessian_operator,
    implicit_moment_gradient,
    outer_value_gradient,
    solve_head_system,
    solve_inner_newton_first,
)
from src.sweep_utils import representative
from src.whole_layer_gradient_alignment import POLICY, moment_partials

SCIENCE = "Cora70_half_teacher_CE_whole_layer_one_update_scientific_stageBU_v1.json"
SCIENCE_SHA = "aec8aa3bb433e0ef398164684def190e5c5fe4e3eaeb781c4a284f3d55018ecb"
SCIENCE_REVIEW = "independent_scientific_review_stageBU_v1.json"
SCIENCE_REVIEW_SHA = "6b519d13add5944c642cc8b3ea663e0f6b148b0cf4da6f4a1f7d4e33fc642044"
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
_cached_targets, _retained = bq._cached_targets, bq._retained
PENALTY = .0001


def numerical_source():
    result = bc.numerical_source()
    _require(result["files"].get("whole_layer_gradient_alignment.py") == HELPER_SHA
             and result["files"].get("cora_hybrid_probe.py") == _sha(__file__),
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
             "Require results or exact three owned artifact paths")
    _require(isinstance(spec["source"], dict) and isinstance(spec["source"].get("files"), dict)
             and set(spec["source"]["files"]) == set(science["source_before"]["files"]) | {
                 p for p in science["source_protection"]["owned"] if p.startswith("src/")}
             and spec["python_version"] == science["version_binding"]["python_version"]
             and spec["numerical_source"].get("versions") == science["version_binding"]["runtime_versions_from_pinned_native_metadata"],
             "Require actual promoted82 source/version binding")


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












def _normalizers(teacher_CE, alignment):
    _require(all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in (teacher_CE, alignment)),
             "Own P0 F0/G0 must be finite strictly positive detached scalars")
    return dict(F0=float(teacher_CE), G0=float(alignment))


def _joint(moments, buffers, theta, anchors, targets, scales, source_features, evidence, stop):
    """Full native joint partial; no P backward and no head fit here."""
    dimension = buffers["z"].shape[1]
    _require(moments.device.type == "cuda" and moments.dtype == torch.float64
             and theta.device == moments.device and theta.dtype == torch.float64 and not theta.requires_grad,
             "Require native FP64 moments and detached stationary head")
    centers, labels, mass = decode_moments(moments.detach(), dimension)
    weights, x = torch.full_like(mass, 1/len(mass)), augmented(centers)
    _count(evidence, "uniform_head_gradient_certificate")
    maximum = float(head_gradient(x, labels, weights, theta, PENALTY).abs().max())
    _require(math.isfinite(maximum) and maximum <= 1e-7, "Native uniform head not stationary")
    _count(evidence, "uniform_head_gradient_certificate", True)
    _stop(stop)
    _count(evidence, "teacher_outer")
    F, rhs = outer_value_gradient(buffers["z"], buffers["q"], theta, 65536, source_features)
    _count(evidence, "teacher_outer", True)
    _stop(stop)
    _count(evidence, "adjoint_solve")
    vector, diagnostic = solve_head_system(x, labels, weights, theta, PENALTY, rhs,
        rtol=1e-6, atol=1e-12, max_iter=512, initial=None, cg_check_interval=1)
    _count(evidence, "adjoint_solve", True)
    _count(evidence, "adjoint_residual_certificate")
    multiply, _ = hessian_operator(x, labels, weights, theta, PENALTY)
    residual = float((multiply(vector)-rhs).norm())
    rhs_norm = float(rhs.norm())
    _require(diagnostic.get("cg_converged") is True and bool(torch.isfinite(vector).all())
             and math.isfinite(residual) and residual <= 1e-12 + 1e-6*rhs_norm
             and math.isfinite(diagnostic["cg_residual"]) and diagnostic["cg_residual"] <= 1e-12 + 1e-6*rhs_norm,
             "Native teacher adjoint failed")
    _count(evidence, "adjoint_residual_certificate", True)
    _stop(stop)
    _count(evidence, "implicit_moment_partial")
    Fpartial = implicit_moment_gradient(moments, dimension, theta, vector, PENALTY, loss_weighting="uniform")
    _count(evidence, "implicit_moment_partial", True)
    _stop(stop)
    _count(evidence, "grouping_partial")
    Gresult = moment_partials(moments, buffers["transform"], anchors, targets)
    _count(evidence, "grouping_partial", True)
    G = float(Gresult["loss"])
    if scales is None:
        scales = _normalizers(F, G)
    else:
        _require(type(scales) is dict and set(scales) == {"F0", "G0"}, "Exactly frozen F0/G0 required")
        _require(scales == _normalizers(scales["F0"], scales["G0"]), "Frozen normalizers changed")
    gradient = .5*Fpartial/scales["F0"] + .5*Gresult["moment_gradient"]/scales["G0"]
    value = .5*F/scales["F0"] + .5*G/scales["G0"]
    _require(math.isfinite(F) and math.isfinite(G) and math.isfinite(value)
             and bool(torch.isfinite(Fpartial).all()) and bool(torch.isfinite(gradient).all()), "Nonfinite native joint value/partial")
    _count(evidence, "joint_partial", True)
    return dict(objective=value, teacher_CE=F, alignment=G, normalizers=dict(scales), theta_gradient_max=maximum,
        rhs=rhs, rhs_norm=rhs_norm, vector=vector, adjoint=diagnostic, explicit_adjoint_residual=residual,
        teacher_partial=Fpartial.detach(), grouping=Gresult, moment_gradient=gradient.detach(), outside_P_backwards=0)


def _one_update(buffers, saved, reference, anchors, targets, evidence, stop):
    probe._runtime_precision_guard()
    parameters = [p.detach().clone().requires_grad_() for p in buffers["initial"]]
    _count(evidence, "P_optimizer_constructor")
    optimizer = torch.optim.Adam(parameters, lr=.01, betas=(.9, .999), eps=1e-12, weight_decay=0, foreach=False, fused=False)
    _count(evidence, "P_optimizer_constructor", True)
    _count(evidence, "source_augmented_features")
    source_features = augmented(buffers["z"])
    _count(evidence, "source_augmented_features", True)
    _count(evidence, "P0_connected_moment")
    moments0 = probe._moments(buffers, parameters)
    _count(evidence, "P0_connected_moment", True)
    _require(torch.equal(moments0.detach(), probe._tensor(saved.get("moments"), moments0.shape, torch.float64,
             "Malformed original NODE0 moments").to(moments0)), "Connected P0 differs from own cached NODE0")
    _count(evidence, "material_certificate")
    centers0, labels0, _, logits0, conservation0 = _material(moments0.detach(), buffers, parameters)
    _count(evidence, "material_certificate", True)
    theta0 = probe._tensor(saved.get("theta"), (labels0.shape[1], centers0.shape[1]+1), torch.float64,
        "Malformed original NODE0 theta").to(moments0).detach()
    _count(evidence, "P0_representative_inputs")
    X, Q, mass = representative(moments0.detach(), probe._transform(buffers), 1433, moments0.device)
    uniform = torch.full_like(mass, 1/len(mass))
    _count(evidence, "P0_representative_inputs", True)
    _require(reference["actual_FP32_P0_X_Q_uniform_equal_reference"] is True
             and probe._digest([X, Q, uniform]) == reference["student_inputs"], "Actual FP32 P0 inputs differ")
    _count(evidence, "joint_partial")
    joint0 = _joint(moments0, buffers, theta0, anchors, targets, None, source_features, evidence, stop)
    scales = dict(joint0["normalizers"])
    _require(abs(joint0["objective"]-1.) <= 1e-12, "Own normalized joint J0 must equal1")
    _stop(stop)
    _count(evidence, "original_moment_backward")
    moments0.backward(joint0["moment_gradient"])
    _count(evidence, "original_moment_backward", True)
    _require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in parameters), "Nonfinite/missing factor gradient")
    gradients = [p.grad.detach().clone() for p in parameters]
    expected, first, second = inherited._adam_step(parameters, [torch.zeros_like(p) for p in parameters],
        [torch.zeros_like(p) for p in parameters], gradients, 1)
    inherited._optimizer(optimizer.state_dict(), parameters, first, second, 0)
    state0 = probe.cpu_state(dict(step=0, parameters=parameters, moments=moments0, theta=theta0, joint=joint0,
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
    centers1, labels1, weights1, logits1, conservation1 = _material(moments1, buffers, parameters)
    _count(evidence, "material_certificate", True)
    _count(evidence, "endpoint_head_solve")
    fitted = solve_inner_newton_first(centers1, labels1, weights1, PENALTY, initial=theta0,
        max_iter=2000, grad_tol=1e-7, cg_max_iter=512, newton_steps=8, cg_check_interval=1)
    _count(evidence, "endpoint_head_solve", True)
    _require(fitted.get("inner_converged") is True, "P1 original uniform head solver did not converge")
    theta1 = fitted["theta"].detach()
    evidence["head1_work"] = {k:v for k,v in fitted.items() if k != "theta"}
    _stop(stop)
    _count(evidence, "joint_partial")
    joint1 = _joint(moments1, buffers, theta1, anchors, targets, scales, source_features, evidence, stop)
    _require(joint1["normalizers"] == scales, "P1 changed own P0 normalizers")
    state1 = probe.cpu_state(dict(step=1, parameters=parameters, moments=moments1, theta=theta1, joint=joint1,
        optimizer=optimizer.state_dict(), raw_logits=logits1, conservation=conservation1,
        head_work=evidence["head1_work"], terminal_full_joint_partial_diagnostic_only=True))
    evidence.update(native_P0_moments_exactly_equal_cached_NODE0=True, actual_FP32_P0_X_Q_uniform_equal_reference=True,
        P0_student_inputs=probe._digest([X, Q, uniform]), normalizers=scales, hybrid_J0=joint0["objective"],
        hybrid_J1=joint1["objective"], teacher_F0=joint0["teacher_CE"], teacher_F1=joint1["teacher_CE"],
        alignment_G0=joint0["alignment"], alignment_G1=joint1["alignment"],
        joint_J1_decrease_descriptive_only=joint1["objective"] < joint0["objective"],
        joint_by_step={str(k):{name:j[name] for name in ("objective", "teacher_CE", "alignment", "normalizers",
            "theta_gradient_max", "rhs_norm", "adjoint", "explicit_adjoint_residual")} for k,j in ((0,joint0),(1,joint1))})
    return {0:state0, 1:state1}


def _success_counts(evidence, science):
    raw = evidence["counts"]
    actual = {k:0 for k in science["counts_contract"]["planned_success"]}
    actual.update(P_updates=raw.get("P_update_completed", 0), native_Adam_steps=raw.get("P_update_completed", 0),
        grouping_partials=raw.get("grouping_partial_completed", 0),
        synthetic_CE_parameter_gradients=3*raw.get("grouping_partial_completed", 0),
        outside_original_P_backwards=raw.get("original_moment_backward_completed", 0),
        accepted_BN_cache_loads=raw.get("accepted_BN_cache_load_completed", 0),
        updated_head_solve_calls=raw.get("endpoint_head_solve_completed", 0),
        teacher_CE_adjoint_calls=raw.get("adjoint_solve_completed", 0),
        joint_head_gradient_diagnostics=raw.get("uniform_head_gradient_certificate_completed", 0),
        joint_teacher_CE_diagnostics=raw.get("teacher_outer_completed", 0))
    _require(actual == science["counts_contract"]["planned_success"], "Actual completed operation counts differ")
    return actual


def prepare(spec_path, spec_sha256, output_path, stop=lambda: False):
    _require(callable(stop), "Stop callback must be callable")
    repo, spec_path = Path(__file__).resolve().parents[1], Path(spec_path).resolve()
    spec, science = _load_spec(spec_path, spec_sha256)
    output, raw = _request(output_path, spec)
    started = time.monotonic()
    bounded = lambda: stop() or time.monotonic()-started >= 300
    evidence = dict(passed=False, success_counts=None, operation="prepare", cells=70, condensation_seed=0, source=spec["source"],
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
        evidence["stage"] = "hybrid_one_update"
        states = _one_update(buffers, saved, reference, anchors, targets, evidence, bounded)
        _retained(buffers, payload, anchors, targets, evidence)
        _stop(bounded)
        _preserve(spec, spec_path, spec_sha256, science, repo)
        probe._runtime_precision_guard()
        context = dict(schema=1, source=spec["source"], numerical_source=spec["numerical_source"], spec_sha256=spec_sha256,
            scientific_preregistration=spec["scientific_preregistration"], case=spec["case"], fixed=spec["fixed"],
            grouping_policy=_POLICY_SPEC, normalizers=dict(evidence["normalizers"]), source_context=buffers["source"], source_buffers=evidence["source_buffer_digest_before"],
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
            error = ValueError("Hybrid runtime probe exceeded300seconds")
            evidence.update(passed=False, deadline_error=str(error))
            if primary is None:
                primary = error
        if evidence["passed"] is not True:
            evidence["success_counts"] = None
        try:
            _write_new(output, evidence)
        except BaseException:
            if primary is not None:
                raise primary
            raise
    if primary is not None:
        raise primary
    return dict(evidence, evidence_path=str(output), evidence_sha256=_sha(output), validation_only=True)
