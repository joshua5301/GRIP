"""One frozen CE-gradient alignment P update on certified Citeseer sources.

This runtime probe has no students, resume, continuation or validation scores.
Frozen models are never trained; the original moment map remains unchanged.
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
from src.ce_gradient_alignment import POLICY, gradient_alignment_partials, source_gradient_targets
from src.citation_macro_probe import _material
from src.citation_source_preflight import _exact, _write_new
from src.io import _fingerprint
from src.research_loop import implementation_provenance
from src.sweep_utils import representative

SCIENCE = "Citeseer30_120_CE_gradient_alignment_one_update_scientific_stageAY_v1.json"
SCIENCE_SHA = "57b4318c24941fc992526c8f57c7dcbb7a8e1d8249e11a355858b3fa80c73431"
HELPER_SHA = "75f6ef10cd1f1e1e15aa2b10ec4e05a924f8bb3634df7b0d3f8f28020cf98bec"
_FIXED = dict(method="frozen_GCN_CE_gradient_alignment_probe", gradient_schema=1, probe_updates=1,
    lr=.01, T=1., rank=8, initialization="teacher_joint", alpha=.3, mass_mode="free", mixing=.05,
    condensation_seed=0, outer_risk="equal_four_block_symmetric_smoothed_CE_gradient_direction",
    rho=.001, anchor_seeds=[0, 1, 2], anchor_count=3, hidden=256, proxy_dropout=0.,
    source_ce_weighting="node_uniform", synthetic_ce_weighting="uniform", source_model_dtype="float32",
    source_Q_dtype="float64", moment_dtype="float64", direction_dtype="float64")
_POLICY_SPEC = json.loads(json.dumps(POLICY))
_require, _stop, _tensor, _sha, _seal = probe._require, probe._stop, probe._tensor, probe._sha, probe._seal
_count = inherited._count


def numerical_source():
    result = inherited.numerical_source()
    _require(result["files"].get("ce_gradient_alignment.py") == HELPER_SHA
             and result["files"].get("citation_gradient_probe.py") == _sha(__file__),
             "Gradient helper/module source changed")
    return result


def _preserve(spec, path, checksum, science):
    _require(_sha(path) == checksum, "Frozen execution spec changed")
    probe._checked_files(spec["files_sha256"])
    probe._checked_files(spec["artifacts_sha256"])
    probe._checked_files({spec["scientific_preregistration"]["path"]: SCIENCE_SHA})
    probe._checked_files({row["path"]: row["sha256"] for row in science["parents"]})
    _require(implementation_provenance() == spec["source"] and numerical_source() == spec["numerical_source"]
             and platform.python_version() == spec["python_version"], "Source/Git/Python/versions changed")
    _require(torch.get_num_threads() == 4, "Gradient probe requires unchanged threads4")


def _load_spec(path, checksum):
    repo = Path(__file__).resolve().parents[1]
    path = Path(path).resolve()
    _require(type(checksum) is str and len(checksum) == 64 and path.is_file()
             and path.is_relative_to(repo / "results/proposals") and _sha(path) == checksum,
             "Require frozen prospective execution spec path/SHA")
    spec = json.loads(path.read_text())
    fields = {"schema", "scientific_preregistration", "source", "numerical_source", "python_version",
              "files_sha256", "roots", "candidate", "artifacts_sha256", "output_root", "probe_outputs", "gradient_policy"}
    _require(isinstance(spec, dict) and set(spec) == fields and type(spec["schema"]) is int and spec["schema"] == 1
             and _exact(spec["candidate"], _FIXED) and _exact(spec["gradient_policy"], _POLICY_SPEC)
             and spec["roots"] == original._roots(repo),
             "Malformed or changed fixed gradient probe spec")
    reference = dict(path=str(repo / "results/proposals" / SCIENCE), sha256=SCIENCE_SHA)
    _require(spec["scientific_preregistration"] == reference, "Wrong gradient scientific policy")
    probe._checked_files({reference["path"]: SCIENCE_SHA})
    science = json.loads(Path(reference["path"]).read_text())
    _require(_exact(science["candidate"], _FIXED) and spec["files_sha256"] == science["original_files_sha256"],
             "Changed gradient science or original assets")
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


def anchor_initial(nin, classes, hidden, seed, *, dtype=torch.float32, device="cpu"):
    """Private CPU-GEOM draw order, weight[in,out] then bias, seeds0/1/2."""
    _require(all(type(v) is int and v > 0 for v in (nin, classes, hidden))
             and type(seed) is int and seed in (0, 1, 2) and dtype in (torch.float32, torch.float64),
             "Require fixed GEOM anchor dimensions/seed/precision")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    parameters = []
    for inputs, outputs in ((nin, hidden), (hidden, classes)):
        bound = 1/math.sqrt(inputs)
        weight = torch.empty(inputs, outputs, dtype=dtype, device="cpu")
        weight.uniform_(-bound, bound, generator=generator)
        bias = torch.empty(outputs, dtype=dtype, device="cpu")
        bias.uniform_(-bound, bound, generator=generator)
        parameters.extend((weight.T.contiguous().to(device).detach(), bias.to(device).detach()))
    return tuple(parameters)


def _targets(buffers, evidence, stop):
    ensemble = [buffers["model_initial"]]
    nin, classes = buffers["z"].shape[1], buffers["q"].shape[1]
    for seed in (1, 2):
        _stop(stop)
        _count(evidence, "additional_anchor_GEOM_factory")
        ensemble.append(anchor_initial(nin, classes, buffers["model_initial"][0].shape[0], seed,
                                      dtype=buffers["x"].dtype, device=buffers["x"].device))
        _count(evidence, "additional_anchor_GEOM_factory", True)
    targets = dict(policy=dict(POLICY), anchors=[])
    for initial in ensemble:
        _stop(stop)
        _count(evidence, "source_CE_gradient_target")
        cache = source_gradient_targets((initial,), buffers["x"], buffers["S"], buffers["q"], stop=stop)
        _count(evidence, "source_CE_gradient_target", True)
        targets["anchors"].extend(cache["anchors"])
    targets["anchors"] = tuple(targets["anchors"])
    return targets


def _one_update(buffers, saved, reference, evidence, stop):
    """Two synthetic partials, one outside custom moment VJP and one Adam step."""
    _require(isinstance(saved, dict), "Malformed original NODE0 snapshot")
    probe._runtime_precision_guard()
    parameters = [value.detach().clone().requires_grad_() for value in buffers["initial"]]
    _count(evidence, "P_optimizer_constructor")
    optimizer = torch.optim.Adam(parameters, lr=.01, betas=(.9, .999), eps=1e-12, weight_decay=0,
                                 foreach=False, fused=False)
    _count(evidence, "P_optimizer_constructor", True)
    _stop(stop)
    _count(evidence, "P0_connected_moment")
    moments0 = probe._moments(buffers, parameters)
    _count(evidence, "P0_connected_moment", True)
    cached = _tensor(saved.get("moments"), moments0.shape, torch.float64, "Malformed original NODE0 moments")
    _require(torch.equal(moments0.detach(), cached.to(moments0)), "Connected P0 differs from original cached NODE0")
    _count(evidence, "material_certificate")
    _, _, _, logits0, conservation0 = _material(moments0.detach(), buffers, parameters)
    _count(evidence, "material_certificate", True)
    _count(evidence, "P0_representative_inputs")
    X, Q, mass = representative(moments0.detach(), probe._transform(buffers), buffers["z"].shape[1], moments0.device)
    uniform = torch.full_like(mass, 1/len(mass))
    _count(evidence, "P0_representative_inputs", True)
    _require(reference["actual_FP32_P0_X_Q_uniform_equal_reference"] is True
             and probe._digest([X, Q, uniform]) == reference["student_inputs"],
             "Actual FP32 P0 X/Q/uniform differ from AT reference")
    targets = _targets(buffers, evidence, stop)
    frozen_digest = probe._digest(targets)
    original_model_digest = probe._digest(buffers["model_initial"])
    _count(evidence, "synthetic_alignment_partial")
    result0 = gradient_alignment_partials(moments0, buffers["transform"], targets, stop=stop)
    _count(evidence, "synthetic_alignment_partial", True)
    scale = float(result0["loss"])
    _require(math.isfinite(scale) and scale > 0, "Own alignment J0 must be finite and positive")
    _stop(stop)
    _count(evidence, "original_moment_backward")
    moments0.backward(result0["moment_gradient"]/scale)
    _count(evidence, "original_moment_backward", True)
    _require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in parameters),
             "Missing/nonfinite original factor gradients")
    gradients = [p.grad.detach().clone() for p in parameters]
    expected, first, second = inherited._adam_step(parameters, [torch.zeros_like(p) for p in parameters],
        [torch.zeros_like(p) for p in parameters], gradients, 1)
    state0 = dict(step=0, parameters=[p.detach().clone() for p in parameters], moments=moments0.detach(),
        alignment=result0, scale=scale, scaled_factor_gradients=gradients, optimizer=optimizer.state_dict(),
        raw_logits=logits0.detach(), conservation=conservation0)
    _stop(stop)
    probe._runtime_precision_guard()
    _count(evidence, "P_update")
    optimizer.step()
    _count(evidence, "P_update", True)
    _require(all(torch.equal(p.detach(), e) for p, e in zip(parameters, expected, strict=True)),
             "Native Adam parameter recurrence differs")
    inherited._optimizer(optimizer.state_dict(), parameters, first, second, 1)
    _stop(stop)
    probe._runtime_precision_guard()
    _count(evidence, "P1_moment")
    with torch.no_grad():
        moments1 = probe._moments(buffers, parameters).detach()
    _count(evidence, "P1_moment", True)
    _count(evidence, "material_certificate")
    _, _, _, logits1, conservation1 = _material(moments1, buffers, parameters)
    _count(evidence, "material_certificate", True)
    _count(evidence, "synthetic_alignment_partial")
    result1 = gradient_alignment_partials(moments1, buffers["transform"], targets, stop=stop)
    _count(evidence, "synthetic_alignment_partial", True)
    _require(probe._digest(targets) == frozen_digest and probe._digest(buffers["model_initial"]) == original_model_digest,
             "Immutable target/model cache changed")
    state1 = dict(step=1, parameters=[p.detach().clone() for p in parameters], moments=moments1,
        alignment=result1, optimizer=optimizer.state_dict(), raw_logits=logits1.detach(), conservation=conservation1,
        terminal_partial_diagnostic_only=True)
    evidence.update(target_model_digest_before=frozen_digest, target_model_digest_after=probe._digest(targets),
                    inherited_seed0_model_digest=original_model_digest, anchor_seeds=[0, 1, 2],
                    P0_student_inputs=probe._digest(dict(X=X, Q=Q, uniform_weights=uniform)),
                    actual_FP32_P0_X_Q_uniform_equal_reference=True,
                    native_P0_moments_exactly_equal_cached_NODE0=True)
    return {0: probe.cpu_state(state0), 1: probe.cpu_state(state1)}, probe.cpu_state(targets)


def _source_buffer_digest(buffers):
    return probe._digest(dict(z=buffers["z"], Q=buffers["q"], H=buffers["h"], X=buffers["x"],
        hard=buffers["hard"], transform=buffers["transform"], original_CSR=buffers["original_S"],
        dense_original_S=buffers["S"], initial=buffers["initial"]))


def prepare_probe(cells, spec_path, spec_sha256, output_path, stop=lambda: False):
    """Runtime-only; a fresh CUDA process is mandatory for each budget."""
    inherited._budget(cells)
    _require(callable(stop), "Stop callback must be callable")
    spec_path = Path(spec_path).resolve()
    spec, science = _load_spec(spec_path, spec_sha256)
    output = Path(output_path).resolve()
    folder = output.with_suffix("")
    _require(str(output) == spec["probe_outputs"][str(cells)] and not output.exists() and not folder.exists(),
             "Require new declared probe evidence/tensor namespace")
    started = time.monotonic()
    bounded_stop = lambda: stop() or time.monotonic()-started >= 300
    candidate = dict(_FIXED, gradient_source_digest=_seal(spec["numerical_source"]))
    evidence = dict(passed=False, cells=cells, candidate=candidate, candidate_id=_fingerprint(candidate),
        source=spec["source"], numerical_source=spec["numerical_source"], python_version=spec["python_version"],
        scientific_preregistration=spec["scientific_preregistration"], spec_path=str(spec_path), spec_sha256=spec_sha256,
        gradient_policy=_POLICY_SPEC, test_enabled=False, validation_only=True, stage="fresh_native_policy",
        files_sha256=spec["files_sha256"], counts=dict(P_update_attempts=0, P_update_completed=0, student_fits=0,
            teacher_map_Phi_hardinit_fits=0, functional_inner_SGD_steps=0, accuracy_forwards=0,
            fresh_head_fits=0, adjoint_or_Hessian_solves=0, custom_double_backwards=0))
    native, primary, buffers = False, None, None
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
        evidence["stage"] = "frozen_gradient_alignment_one_update"
        checkpoint = buffers["root"] / original.REFERENCE / "condensation_0/checkpoints/step_000000.pt"
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        evidence["source_buffer_digest_before"] = _source_buffer_digest(buffers)
        states, targets = _one_update(buffers, saved, reference, evidence, bounded_stop)
        evidence["source_buffer_digest_after"] = _source_buffer_digest(buffers)
        _require(evidence["source_buffer_digest_before"] == evidence["source_buffer_digest_after"],
                 "Immutable original source buffers changed")
        context = dict(schema=1, candidate=candidate, source=spec["source"], numerical_source=spec["numerical_source"],
            scientific_preregistration=spec["scientific_preregistration"], spec_sha256=spec_sha256,
            native_environment=evidence["native_environment"], original_source=buffers["source"], reference=reference,
            source_buffers=evidence["source_buffer_digest_before"],
            target_model_digest=evidence["target_model_digest_before"], anchor_seeds=[0, 1, 2],
            gradient_policy=_POLICY_SPEC, no_resume_or_continuation=True)
        _stop(bounded_stop)
        _preserve(spec, spec_path, spec_sha256, science)
        inherited._stable_files(buffers)
        probe._runtime_precision_guard()
        folder.mkdir(parents=True, exist_ok=False)
        target_payload = dict(context=probe.cpu_state(context), targets=targets)
        probe._atomic(folder / "source_gradient_targets.pt", probe._attach(target_payload), True)
        for step, state in states.items():
            state.update(context=probe.cpu_state(context))
            probe._atomic(folder / f"step_{step:06d}.pt", probe._attach(state), True)
        evidence.update(checkpoints_sha256={str(p.resolve()): _sha(p) for p in folder.glob("*.pt")},
            context_digest=_seal(context), reference=reference, alignment_J0=float(states[0]["alignment"]["loss"]),
            alignment_J1=float(states[1]["alignment"]["loss"]), own_scale=states[0]["scale"],
            conservation_by_step={k: s["conservation"] for k, s in states.items()})
        evidence["passed"] = True
    except BaseException as error:
        primary = error
        evidence.update(error_type=type(error).__name__, error=str(error), failed_stage=evidence["stage"])
    finally:
        if buffers is not None and "source_buffer_digest_before" in evidence:
            try:
                if "source_buffer_digest_after" not in evidence:
                    evidence["source_buffer_digest_after"] = _source_buffer_digest(buffers)
                _require(evidence["source_buffer_digest_before"] == evidence["source_buffer_digest_after"],
                         "Immutable native source buffers changed")
            except BaseException as error:
                evidence.update(passed=False, native_source_preservation_error=repr(error))
                if primary is None:
                    primary = error
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
