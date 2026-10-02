"""One fresh macro-teacher CE assignment update on certified Citeseer sources.

This runtime probe has no students, resume, continuation or validation scores.
The original uniform converged inner head and moment map remain unchanged.
"""
import json
import math
import platform
import time
from pathlib import Path

import torch

from src import citation_source_certificate as original
from src import citeseer_finite_student_v2 as inherited
from src import finite_student_probe as probe
from src.citation_source_preflight import _exact, _write_new
from src.io import _fingerprint
from src.low_rank_assignment import logit_block
from src.macro_teacher_outer import POLICY, macro_teacher_outer, macro_teacher_weights
from src.moments import augmented, decode_moments, make_material
from src.research_loop import implementation_provenance
from src.soft_ce_partition import (
    head_gradient,
    hessian_operator,
    implicit_moment_gradient,
    solve_head_system,
    solve_inner_newton_first,
)
from src.sweep_utils import representative

SCIENCE = "Citeseer30_120_macro_teacher_CE_one_update_scientific_stageAW_v1.json"
SCIENCE_SHA = "09fed6e719853739988d57bc7684df26e9d628172a93770b18845bd39a6ea8b2"
HELPER_SHA = "a91cafe63b8ac37d24c579c16e2d598b7a35fbc04b6758de4d4edfab37265596"
_FIXED = dict(method="macro_source_teacher_ce_linear_probe", macro_schema=1, probe_updates=1, width=0,
    lr=.01, T=1., rank=8, penalty=.001, initialization="teacher_joint", alpha=.3,
    inner_loss_weighting="uniform", mass_mode="free", mixing=.05, condensation_seed=0,
    outer_risk="teacher_soft_class_macro", inner_tol=1e-7, cg_rtol=1e-6, cg_atol=1e-12,
    cg_max_iter=512, inner_max_iter=2000, newton_steps=8, cg_check_interval=1)
_require, _stop, _tensor, _sha, _seal = probe._require, probe._stop, probe._tensor, probe._sha, probe._seal
_count = inherited._count


def numerical_source():
    result = inherited.numerical_source()
    _require(result["files"].get("macro_teacher_outer.py") == HELPER_SHA
             and result["files"].get("citation_macro_probe.py") == _sha(__file__),
             "Macro helper/module source changed")
    return result


def _preserve(spec, path, checksum, science):
    _require(_sha(path) == checksum, "Frozen execution spec changed")
    probe._checked_files(spec["files_sha256"])
    probe._checked_files(spec["artifacts_sha256"])
    probe._checked_files({spec["scientific_preregistration"]["path"]: SCIENCE_SHA})
    probe._checked_files({row["path"]: row["sha256"] for row in science["parents"]})
    _require(implementation_provenance() == spec["source"] and numerical_source() == spec["numerical_source"]
             and platform.python_version() == spec["python_version"], "Source/Git/Python/versions changed")
    _require(torch.get_num_threads() == 4, "Macro probe requires unchanged threads4")


def _load_spec(path, checksum):
    repo = Path(__file__).resolve().parents[1]
    path = Path(path).resolve()
    _require(type(checksum) is str and len(checksum) == 64 and path.is_file()
             and path.is_relative_to(repo / "results/proposals") and _sha(path) == checksum,
             "Require frozen prospective execution spec path/SHA")
    spec = json.loads(path.read_text())
    fields = {"schema", "scientific_preregistration", "source", "numerical_source", "python_version",
              "files_sha256", "roots", "candidate", "artifacts_sha256", "output_root", "probe_outputs"}
    _require(isinstance(spec, dict) and set(spec) == fields and type(spec["schema"]) is int and spec["schema"] == 1
             and _exact(spec["candidate"], _FIXED) and spec["roots"] == original._roots(repo),
             "Malformed or changed fixed macro probe spec")
    reference = dict(path=str(repo / "results/proposals" / SCIENCE), sha256=SCIENCE_SHA)
    _require(spec["scientific_preregistration"] == reference, "Wrong macro scientific policy")
    probe._checked_files({reference["path"]: SCIENCE_SHA})
    science = json.loads(Path(reference["path"]).read_text())
    _require(_exact(science["candidate"], _FIXED) and spec["files_sha256"] == science["original_files_sha256"],
             "Changed macro science or original assets")
    _require(isinstance(spec["artifacts_sha256"], dict) and spec["artifacts_sha256"]
             and all(Path(p).resolve().is_relative_to(repo / "results") for p in spec["artifacts_sha256"]),
             "Require frozen implementation/review artifacts")
    output = Path(spec["output_root"]).resolve()
    _require(str(output) == spec["output_root"] and output.is_relative_to(repo / "results/research_loop"),
             "Require frozen research output root")
    outputs = spec["probe_outputs"]
    _require(isinstance(outputs, dict) and set(outputs) == {"30", "120"}, "Require both fixed probe outputs")
    for value in outputs.values():
        _require(type(value) is str and str(Path(value).resolve()) == value
                 and Path(value).is_relative_to(output) and Path(value).suffix == ".json", "Invalid probe output")
    _require(len(set(outputs.values())) == 2, "Probe outputs must be distinct")
    _preserve(spec, path, checksum, science)
    return spec, science


@torch.no_grad()
def _material(moments, buffers, parameters):
    dimension = buffers["z"].shape[1]
    _tensor(moments, (len(parameters[1]), dimension+1+buffers["q"].shape[1]), torch.float64, "Malformed original moments")
    centers, labels, mass = decode_moments(moments, dimension)
    logits = logit_block(*parameters, buffers["hard"], .05)
    probability = logits.double().softmax(1)
    diagnostics = dict(row=float((probability.sum(1)-1).abs().max()), mass_sum=float((mass.sum()-1).abs()),
        material=float((moments.sum(0)-make_material(buffers["z"], buffers["q"]).mean(0)).abs().max()),
        Qc_simplex=float((labels.sum(1)-1).abs().max()), min_mass=float(mass.min()))
    _require(bool((mass > 0).all()) and bool((labels >= 0).all()) and bool(torch.isfinite(centers).all())
             and all(math.isfinite(v) and v <= 1e-12 for k, v in diagnostics.items() if k != "min_mass"),
             "Original P/material conservation failed")
    return centers, labels, torch.full_like(mass, 1/len(mass)), logits, diagnostics


def _head_certificate(theta, centers, labels, weights):
    _tensor(theta, (labels.shape[1], centers.shape[1]+1), torch.float64, "Malformed uniform linear head")
    gradient = head_gradient(augmented(centers), labels, weights, theta, .001)
    error = float(gradient.abs().max())
    _require(math.isfinite(error) and error <= 1e-7, "Original uniform inner head is not stationary")
    return error


def _one_update(buffers, saved, evidence, stop):
    """Original moment VJP, uniform-inner implicit head, and one native Adam step."""
    _require(isinstance(saved, dict), "Malformed cached NODE0 snapshot")
    probe._runtime_precision_guard()
    parameters = [value.detach().clone().requires_grad_() for value in buffers["initial"]]
    optimizer = torch.optim.Adam(parameters, lr=.01, betas=(.9, .999), eps=1e-12, weight_decay=0,
                                 foreach=False, fused=False)
    _stop(stop)
    _count(evidence, "P0_connected_moment")
    moments0 = probe._moments(buffers, parameters)
    _count(evidence, "P0_connected_moment", True)
    cached = _tensor(saved.get("moments"), moments0.shape, torch.float64, "Malformed original NODE0 moments")
    _require(torch.equal(moments0.detach(), cached.to(moments0)), "Connected native P0 moments differ from cached NODE0")
    _count(evidence, "material_certificate")
    centers, labels, weights, logits0, conservation0 = _material(moments0.detach(), buffers, parameters)
    _count(evidence, "material_certificate", True)
    theta0 = _tensor(saved.get("theta"), (labels.shape[1], centers.shape[1]+1), torch.float64,
                     "Malformed original NODE0 theta").to(moments0)
    _count(evidence, "uniform_head_gradient_certificate")
    gradient0 = _head_certificate(theta0, centers, labels, weights)
    _count(evidence, "uniform_head_gradient_certificate", True)
    _count(evidence, "macro_outer")
    loss0, rhs = macro_teacher_outer(buffers["z"], buffers["q"], theta0, stop=stop)
    _count(evidence, "macro_outer", True)
    _require(math.isfinite(loss0) and loss0 > 0, "Macro J0 must be finite and strictly positive")
    _stop(stop)
    _count(evidence, "adjoint_solve")
    vector, adjoint = solve_head_system(augmented(centers), labels, weights, theta0, .001, rhs,
        rtol=1e-6, atol=1e-12, max_iter=512, initial=None, cg_check_interval=1)
    _count(evidence, "adjoint_solve", True)
    multiply, _ = hessian_operator(augmented(centers), labels, weights, theta0, .001)
    residual = float((multiply(vector)-rhs).norm())
    _require(adjoint["cg_converged"] is True and bool(torch.isfinite(vector).all())
             and math.isfinite(residual) and residual <= max(1e-12, 1e-6*float(rhs.norm())), "Macro adjoint certificate failed")
    _stop(stop)
    _count(evidence, "implicit_moment_partial")
    partial = implicit_moment_gradient(moments0, centers.shape[1], theta0, vector, .001, loss_weighting="uniform")
    _count(evidence, "implicit_moment_partial", True)
    _require(bool(torch.isfinite(partial).all()), "Nonfinite macro moment partial")
    _count(evidence, "original_moment_backward")
    moments0.backward(partial/loss0)
    _count(evidence, "original_moment_backward", True)
    _require(all(value.grad is not None for value in parameters), "Missing original factor gradient")
    gradients = [value.grad.detach().clone() for value in parameters]
    _require(all(bool(torch.isfinite(value).all()) for value in gradients), "Nonfinite original factor gradient")
    expected, first, second = inherited._adam_step(parameters, [torch.zeros_like(v) for v in parameters],
        [torch.zeros_like(v) for v in parameters], gradients, 1)
    snapshot0 = dict(step=0, parameters=[v.detach().clone() for v in parameters], moments=moments0.detach(),
        theta=theta0, macro_ce=loss0, scale=loss0, uniform_inner_grad_max=gradient0, rhs=rhs, vector=vector,
        adjoint=adjoint, explicit_adjoint_residual=residual, moment_partial=partial, scaled_factor_gradients=gradients,
        optimizer=optimizer.state_dict(), raw_logits=logits0.detach(), conservation=conservation0)
    _stop(stop)
    _count(evidence, "P_update")
    optimizer.step()
    _count(evidence, "P_update", True)
    _require(all(torch.equal(v.detach(), e) for v, e in zip(parameters, expected, strict=True)), "Native Adam parameter recurrence differs")
    inherited._optimizer(optimizer.state_dict(), parameters, first, second, 1)
    _stop(stop)
    _count(evidence, "P1_connected_moment")
    moments1 = probe._moments(buffers, parameters).detach()
    _count(evidence, "P1_connected_moment", True)
    _count(evidence, "material_certificate")
    centers1, labels1, weights1, logits1, conservation1 = _material(moments1, buffers, parameters)
    _count(evidence, "material_certificate", True)
    _count(evidence, "endpoint_head_solve")
    fitted = solve_inner_newton_first(centers1, labels1, weights1, .001, initial=theta0, max_iter=2000,
        grad_tol=1e-7, cg_max_iter=512, newton_steps=8, cg_check_interval=1)
    _count(evidence, "endpoint_head_solve", True)
    _stop(stop)
    _count(evidence, "uniform_head_gradient_certificate")
    gradient1 = _head_certificate(fitted["theta"], centers1, labels1, weights1)
    _count(evidence, "uniform_head_gradient_certificate", True)
    _require(fitted["inner_converged"] is True, "P1 uniform inner solver did not converge")
    _count(evidence, "macro_outer")
    loss1, rhs1 = macro_teacher_outer(buffers["z"], buffers["q"], fitted["theta"], stop=stop)
    _count(evidence, "macro_outer", True)
    snapshot1 = dict(step=1, parameters=[v.detach().clone() for v in parameters], moments=moments1,
        theta=fitted["theta"], macro_ce=loss1, macro_rhs=rhs1, uniform_inner_grad_max=gradient1,
        optimizer=optimizer.state_dict(), raw_logits=logits1.detach(), conservation=conservation1,
        head_work={k: v for k, v in fitted.items() if k != "theta"})
    evidence["head1_work"] = snapshot1["head_work"]
    return {0: probe.cpu_state(snapshot0), 1: probe.cpu_state(snapshot1)}


def prepare_probe(cells, spec_path, spec_sha256, output_path, stop=lambda: False):
    """Exactly one new P update; require a fresh CUDA process per invocation."""
    inherited._budget(cells)
    _require(callable(stop), "Stop callback must be callable")
    spec_path = Path(spec_path).resolve()
    spec, science = _load_spec(spec_path, spec_sha256)
    output = Path(output_path).resolve()
    folder = output.with_suffix("")
    _require(str(output) == spec["probe_outputs"][str(cells)] and not output.exists() and not folder.exists(),
             "Require new declared one-update evidence and tensor namespace")
    started = time.monotonic()
    bounded_stop = lambda: stop() or time.monotonic()-started >= 300
    candidate = dict(_FIXED, macro_source_digest=_seal(spec["numerical_source"]))
    evidence = dict(passed=False, cells=cells, candidate=candidate, candidate_id=_fingerprint(candidate),
        source=spec["source"], numerical_source=spec["numerical_source"], python_version=spec["python_version"],
        scientific_preregistration=spec["scientific_preregistration"], spec_path=str(spec_path), spec_sha256=spec_sha256,
        policy=POLICY, test_enabled=False, validation_only=True, stage="fresh_native_policy",
        files_sha256=spec["files_sha256"], counts=dict(P_update_attempts=0, P_update_completed=0, student_fits=0,
        teacher_map_Phi_hardinit_fits=0, functional_inner_SGD_steps=0, accuracy_forwards=0))
    native, primary = False, None
    try:
        _stop(bounded_stop)
        _require(torch.get_num_threads() == 4 and not torch.cuda.is_initialized(), "Require threads4/fresh CUDA worker")
        environment = probe._native("cuda")
        native = True
        torch.cuda.reset_peak_memory_stats()
        _require(torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory == original.CUDA_CAPACITY,
                 "Frozen physical GPU capacity changed")
        evidence["native_environment"] = dict(environment, python_version=platform.python_version(), threads=4)
        buffers, reference = inherited._load_source(cells, spec, science, evidence, bounded_stop)
        evidence["stage"] = "macro_implicit_one_update"
        checkpoint = buffers["root"] / original.REFERENCE / "condensation_0/checkpoints/step_000000.pt"
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        states = _one_update(buffers, saved, evidence, bounded_stop)
        X, Q, mass = representative(states[0]["moments"], probe._transform(buffers), 3703, "cuda")
        uniform = torch.full_like(mass, 1/cells)
        _require(reference["actual_FP32_P0_X_Q_uniform_equal_reference"] is True
                 and probe._digest([X, Q, uniform]) == reference["student_inputs"],
                 "Actual FP32 P0 X/Q/uniform inputs differ from AT reference")
        class_weights, class_mass = macro_teacher_weights(buffers["q"])
        context = dict(schema=1, candidate=candidate, source=spec["source"], numerical_source=spec["numerical_source"],
            scientific_preregistration=spec["scientific_preregistration"], spec_sha256=spec_sha256,
            native_environment=evidence["native_environment"], original_source=buffers["source"], reference=reference,
            source_buffers=probe._digest(dict(z=buffers["z"], Q=buffers["q"], H=buffers["h"],
                hard=buffers["hard"], transform=buffers["transform"], original_CSR=buffers["original_S"],
                dense_original_S=buffers["S"], initial=buffers["initial"])), macro_class_mass=class_mass,
            macro_weights_digest=probe._digest(class_weights), unused_GEOM_factory=True, no_resume_or_continuation=True)
        _stop(bounded_stop)
        _preserve(spec, spec_path, spec_sha256, science)
        inherited._stable_files(buffers)
        probe._runtime_precision_guard()
        folder.mkdir(parents=True, exist_ok=False)
        for step, state in states.items():
            state.update(context=probe.cpu_state(context))
            probe._atomic(folder / f"step_{step:06d}.pt", probe._attach(state), True)
        pins = {str(p.resolve()): _sha(p) for p in folder.glob("*.pt")}
        evidence.update(checkpoints_sha256=pins, context_digest=_seal(context), reference=reference,
            native_P0_moments_exactly_equal_cached_NODE0=True, actual_FP32_P0_XQ_uniform_equal_reference=True,
            P0_student_inputs=probe._digest(dict(X=X, Q=Q, uniform_weights=uniform)),
            macro_J0=states[0]["macro_ce"], macro_J1=states[1]["macro_ce"], macro_change=states[1]["macro_ce"]-states[0]["macro_ce"],
            own_scale=states[0]["scale"], adjoint=states[0]["adjoint"], conservation_by_step={k: s["conservation"] for k, s in states.items()})
        evidence["passed"] = True
    except BaseException as error:
        primary = error
        evidence.update(error_type=type(error).__name__, error=str(error), failed_stage=evidence["stage"])
    finally:
        try:
            _preserve(spec, spec_path, spec_sha256, science)
            evidence["source_unchanged"] = True
        except BaseException as error:
            evidence.update(passed=False, source_unchanged=False, preservation_error=repr(error))
            if primary is None:
                primary = error
        if native:
            try:
                torch.cuda.synchronize()
                capacity = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
                allocated, reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
                _require(capacity == original.CUDA_CAPACITY and allocated <= capacity and reserved <= capacity,
                         "Native memory/capacity certificate failed")
                evidence.update(CUDA_peak_allocated_bytes=allocated, CUDA_peak_reserved_bytes=reserved, CUDA_total_bytes=capacity)
                probe._runtime_precision_guard()
            except BaseException as error:
                evidence.update(passed=False, native_finalization_error=repr(error))
                if primary is None:
                    primary = error
        evidence["seconds"] = time.monotonic()-started
        try:
            _write_new(output, evidence)
        except BaseException:
            if primary is not None:
                raise primary
            raise
    if primary is not None:
        raise primary
    return dict(evidence, evidence_path=str(output), evidence_sha256=_sha(output))
