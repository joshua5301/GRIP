"""Fixed25 macro source CE, converged uniform inner head, and gated students.

Each budget starts a fresh AX trajectory. AW state and incomplete AX caches
are never resumed. All actual Adam frames and mathematical states are replayed.
"""
import json
import math
import platform
import time
from pathlib import Path

import torch

from src import citation_macro_probe as aw
from src import citation_source_certificate as original
from src import citeseer_finite_student_v2 as inherited
from src import finite_student_probe as probe
from src.citation_source_preflight import _exact, _write_new
from src.io import _fingerprint
from src.macro_teacher_outer import POLICY, macro_teacher_outer
from src.moments import augmented
from src.research_loop import implementation_provenance
from src.soft_ce_partition import (
    hessian_operator,
    implicit_moment_gradient,
    solve_head_system,
    solve_inner_newton_first,
)
from src.sweep_utils import representative

SCIENCE = "Citeseer30_120_macro_teacher_outer_CE_fixed25_scientific_stageAX_v1.json"
SCIENCE_SHA = "198192cf1e99039f86e625fd6441b02810d3b7039429d7901229c2b12174bb93"
HORIZON = 25
_FIXED = {key: value for key, value in aw._FIXED.items() if key != "probe_updates"}
_FIXED.update(method="macro_source_teacher_ce_linear_fixed25", assignment_steps=HORIZON)
_SEEDS = (3800, 3801, 3802)
_require, _stop, _tensor, _sha, _seal = probe._require, probe._stop, probe._tensor, probe._sha, probe._seal
_count = inherited._count


def numerical_source():
    result = aw.numerical_source()
    _require(result["files"].get("citation_macro_fixed25.py") == _sha(__file__), "AX module source changed")
    return result


def canonical_candidate():
    return dict(_FIXED, macro_source_digest=_seal(numerical_source()))


def _preserve(spec, path, checksum, science):
    _require(_sha(path) == checksum, "Frozen AX spec changed")
    probe._checked_files(spec["files_sha256"])
    probe._checked_files(spec["artifacts_sha256"])
    probe._checked_files({spec["scientific_preregistration"]["path"]: SCIENCE_SHA})
    probe._checked_files({row["path"]: row["sha256"] for row in science["parents"]})
    _require(implementation_provenance() == spec["source"] and numerical_source() == spec["numerical_source"]
             and platform.python_version() == spec["python_version"], "Source/Git/Python/versions changed")
    _require(torch.get_num_threads() == 4, "Require unchanged threads4")


def _load_spec(path, checksum):
    repo = Path(__file__).resolve().parents[1]
    path = Path(path).resolve()
    _require(type(checksum) is str and len(checksum) == 64 and path.is_file()
             and path.is_relative_to(repo / "results/proposals") and _sha(path) == checksum, "Require frozen AX spec path/SHA")
    spec = json.loads(path.read_text())
    fields = {"schema", "scientific_preregistration", "source", "numerical_source", "python_version", "files_sha256",
              "roots", "candidate", "candidate_id", "artifacts_sha256", "output_root", "certificate_outputs"}
    _require(isinstance(spec, dict) and set(spec) == fields and type(spec["schema"]) is int and spec["schema"] == 1
             and _exact(spec["candidate"], canonical_candidate()) and spec["candidate_id"] == _fingerprint(spec["candidate"])
             and spec["roots"] == original._roots(repo), "Changed AX candidate/roots/schema or unknown controls")
    reference = dict(path=str(repo / "results/proposals" / SCIENCE), sha256=SCIENCE_SHA)
    _require(spec["scientific_preregistration"] == reference, "Wrong fixed AX science")
    probe._checked_files({reference["path"]: SCIENCE_SHA})
    science = json.loads(Path(reference["path"]).read_text())
    _require(_exact(science["candidate"], _FIXED) and spec["files_sha256"] == science["original_files_sha256"]
             and science["evaluation"]["student_seeds"] == list(_SEEDS), "Changed AX science or original assets")
    _require(isinstance(spec["artifacts_sha256"], dict) and spec["artifacts_sha256"]
             and all(Path(p).resolve().is_relative_to(repo / "results") for p in spec["artifacts_sha256"]), "Require reviewed artifact pins")
    root = Path(spec["output_root"]).resolve()
    _require(str(root) == spec["output_root"] and root.is_relative_to(repo / "results/research_loop"), "Invalid AX output root")
    expected = {str(c): str(root / f"citeseer{c}/native_certificate25_v1.json") for c in (30, 120)}
    _require(spec["certificate_outputs"] == expected, "Wrong declared native25 gate outputs")
    _preserve(spec, path, checksum, science)
    return spec, science


def _folder(spec, cells):
    return Path(spec["output_root"]) / f"citeseer{cells}" / spec["candidate_id"]


def _context(spec, cells, buffers, reference, environment):
    return dict(schema=1, candidate=spec["candidate"], cells=cells, implementation=spec["source"],
        numerical_source=spec["numerical_source"], spec=spec, scientific_preregistration=spec["scientific_preregistration"],
        native_environment=environment, ghost_source=buffers["source"], reference=reference,
        student_recipe_origin=buffers["recipe_origin"], policy=POLICY, assignment_steps=HORIZON,
        source_buffers=probe._digest(dict(z=buffers["z"], Q=buffers["q"], H=buffers["h"], hard=buffers["hard"],
            transform=buffers["transform"], X=buffers["x"], original_CSR=buffers["original_S"], dense_original_S=buffers["S"],
            initial=buffers["initial"])), no_AW_state_reuse=True)


def _adjoint(centers, labels, weights, theta, rhs, vector=None, diagnostic=None):
    if vector is None:
        vector, diagnostic = solve_head_system(augmented(centers), labels, weights, theta, .001, rhs,
            rtol=1e-6, atol=1e-12, max_iter=512, initial=None, cg_check_interval=1)
    vector = _tensor(vector, theta.shape, torch.float64, "Malformed macro adjoint").to(theta)
    multiply, _ = hessian_operator(augmented(centers), labels, weights, theta, .001)
    residual = float((multiply(vector)-rhs).norm())
    _require(isinstance(diagnostic, dict) and diagnostic.get("cg_converged") is True
             and math.isfinite(residual) and residual <= max(1e-12, 1e-6*float(rhs.norm())), "Macro adjoint certificate failed")
    return vector, diagnostic, residual


def _state(buffers, parameters, theta, optimizer, step, scale, head_work, evidence, stop, *, replay=None):
    """Current objective/partial certificate; terminal25 has no update adjoint."""
    _stop(stop)
    probe._runtime_precision_guard()
    for parameter in parameters:
        parameter.grad = None
    _count(evidence, "connected_moment")
    moments = probe._moments(buffers, parameters)
    _count(evidence, "connected_moment", True)
    centers, labels, weights, logits, conservation = aw._material(moments.detach(), buffers, parameters)
    _count(evidence, "uniform_head_gradient_certificate")
    grad_max = aw._head_certificate(theta, centers, labels, weights)
    _count(evidence, "uniform_head_gradient_certificate", True)
    _count(evidence, "macro_outer")
    value, rhs = macro_teacher_outer(buffers["z"], buffers["q"], theta, stop=stop)
    _count(evidence, "macro_outer", True)
    scale = value if scale is None else scale
    _require(type(scale) in (int, float) and math.isfinite(scale) and scale > 0, "Invalid own macro J0")
    snapshot = dict(step=step, parameters=parameters, optimizer=optimizer, theta=theta, macro_ce=value, J0=scale,
        uniform_inner_grad_max=grad_max, head_work=head_work, moments_digest=probe._digest(moments),
        macro_rhs_digest=probe._digest(rhs), raw_logits_digest=probe._digest(logits), conservation=conservation,
        terminal_no_update=step == HORIZON)
    if step < HORIZON:
        if replay is None:
            _count(evidence, "adjoint_solve")
        vector, diagnostic, residual = _adjoint(centers, labels, weights, theta, rhs,
            None if replay is None else replay.get("vector"), None if replay is None else replay.get("adjoint"))
        if replay is None:
            _count(evidence, "adjoint_solve", True)
        _count(evidence, "implicit_moment_partial")
        partial = implicit_moment_gradient(moments, centers.shape[1], theta, vector, .001, loss_weighting="uniform")
        _count(evidence, "implicit_moment_partial", True)
        _require(bool(torch.isfinite(partial).all()), "Nonfinite macro moment partial")
        _count(evidence, "original_moment_backward")
        moments.backward(partial/scale)
        _count(evidence, "original_moment_backward", True)
        _require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in parameters), "Invalid factor gradient")
        snapshot.update(vector=vector, adjoint=diagnostic, explicit_adjoint_residual=residual,
                        moment_partial_digest=probe._digest(partial), scaled_factor_gradients=[p.grad for p in parameters])
        if step == 0:
            snapshot.update(moment_partial=partial, macro_rhs=rhs)
    if step in (0, HORIZON):
        snapshot["moments"] = moments
    probe._runtime_precision_guard()
    return probe.cpu_state(snapshot), scale


def _history(states):
    return [dict(step=step, macro_ce=s["macro_ce"], uniform_inner_grad_max=s["uniform_inner_grad_max"],
                 J0=s["J0"], update_adjoint_residual=s.get("explicit_adjoint_residual")) for step, s in sorted(states.items())]


def _check_progress(bundle, buffers, context, evidence, stop):
    _require(isinstance(bundle, dict) and set(bundle) == {"schema", "context", "frontier", "states", "J0", "history", "content_sha256"}
             and type(bundle["schema"]) is int and bundle["schema"] == 1 and bundle["context"] == context
             and _seal({k: v for k, v in bundle.items() if k != "content_sha256"}) == bundle["content_sha256"], "Changed AX progress seal/context")
    frontier, states = bundle["frontier"], bundle["states"]
    _require(type(frontier) is int and 0 <= frontier <= HORIZON and isinstance(states, dict)
             and all(type(k) is int for k in states) and set(states) == set(range(frontier+1)), "Incomplete/malformed AX prefix")
    expected = [p.detach().clone() for p in buffers["initial"]]
    first, second = [torch.zeros_like(p) for p in expected], [torch.zeros_like(p) for p in expected]
    scale = None
    for step, saved in sorted(states.items()):
        _stop(stop)
        _require(isinstance(saved, dict) and type(saved.get("step")) is int and saved["step"] == step
                 and isinstance(saved.get("parameters"), list) and len(saved["parameters"]) == 2
                 and isinstance(saved.get("optimizer"), dict), "Malformed AX state")
        for actual, target in zip(saved["parameters"], expected, strict=True):
            _tensor(actual, target.shape, torch.float32, "Malformed AX native factor")
            _require(torch.equal(actual.to(target), target), "AX factor differs from original/previous Adam state")
        inherited._optimizer(saved["optimizer"], expected, first, second, step)
        parameters = [p.detach().clone().requires_grad_() for p in expected]
        theta = _tensor(saved.get("theta"), (buffers["q"].shape[1], buffers["z"].shape[1]+1), torch.float64,
                        "Malformed AX uniform head").to(buffers["z"])
        actual, scale = _state(buffers, parameters, theta, saved["optimizer"], step, scale,
                               saved.get("head_work"), evidence, stop, replay=saved)
        _require(_seal(actual) == _seal(saved), "AX coupled moments/head/J0/partial/gradient certificate changed")
        if step < frontier:
            expected, first, second = inherited._adam_step(expected, first, second,
                [g.to(p) for g, p in zip(actual["scaled_factor_gradients"], expected, strict=True)], step+1)
    history = bundle["history"]
    _require(type(bundle["J0"]) in (int, float) and bundle["J0"] == scale and isinstance(history, list)
             and len(history) == len(states) and all(_exact(a, b) for a, b in zip(history, _history(states), strict=True)),
             "AX history/original J0 changed")
    return bundle


def _cache_files(folder, *, complete):
    names = {"candidate.json", "progress.pt", "history.json", "step_000000.pt"} | ({"step_000025.pt"} if complete else set())
    present = {p.name for p in folder.iterdir() if p.is_file()}
    _require(present == names, "Partial/orphan AX cache files preserved")
    return {str((folder/name).resolve()): _sha(folder/name) for name in names}


def _load_progress(folder, buffers, context, evidence, stop):
    _require(folder.is_dir(), "Require completed AX candidate cache")
    pins = _cache_files(folder, complete=True)
    _require(json.loads((folder/"candidate.json").read_text()) == context["candidate"], "Changed AX candidate")
    bundle = torch.load(folder/"progress.pt", map_location="cpu", weights_only=False)
    _check_progress(bundle, buffers, context, evidence, stop)
    _require(bundle["frontier"] == HORIZON, "Incomplete AX trajectory preserved; no resume/fallback")
    origin = torch.load(buffers["root"] / original.REFERENCE / "condensation_0/checkpoints/step_000000.pt",
                        map_location="cpu", weights_only=False)
    zero = bundle["states"][0]
    _require(isinstance(origin, dict), "Malformed original cached NODE0")
    old_moments = _tensor(origin.get("moments"), zero["moments"].shape, torch.float64, "Malformed original NODE0 moments")
    old_theta = _tensor(origin.get("theta"), zero["theta"].shape, torch.float64, "Malformed original NODE0 theta")
    _require(torch.equal(zero["moments"], old_moments) and torch.equal(zero["theta"], old_theta),
             "AX P0 moments/theta differ from exact original cached NODE0")
    inputs = []
    for moments in (zero["moments"], origin["moments"]):
        _count(evidence, "readonly_P0_representative")
        x, q, mass = representative(moments, probe._transform(buffers), 3703, "cuda")
        inputs.append((x, q, torch.full_like(mass, 1/buffers["cells"])))
        _count(evidence, "readonly_P0_representative", True)
    _require(all(torch.equal(a, b) for a, b in zip(*inputs, strict=True))
             and probe._digest(list(inputs[0])) == context["reference"]["student_inputs"], "Actual native shared P0 inputs differ from AT")
    _require(json.loads((folder/"history.json").read_text()) == bundle["history"], "Changed AX progress history mirror")
    for step in (0, HORIZON):
        mirror = torch.load(folder/f"step_{step:06d}.pt", map_location="cpu", weights_only=False)
        _require(_seal(mirror) == _seal(bundle["states"][step]), "AX endpoint mirror differs from progress authority")
    _require(_cache_files(folder, complete=True) == pins, "Readonly AX prefix validation changed files")
    return bundle, pins


def _store(folder, context, states, scale):
    bundle = probe._attach(dict(schema=1, context=context, frontier=max(states), states=states, J0=scale, history=_history(states)))
    probe._atomic(folder/"progress.pt", bundle, True)
    probe._atomic(folder/"history.json", json.dumps(bundle["history"], indent=2), False)
    for step in (0, HORIZON):
        if step == max(states):
            probe._atomic(folder/f"step_{step:06d}.pt", states[step], True)
    return bundle


def _prepare(spec, buffers, context, evidence, stop):
    folder = _folder(spec, buffers["cells"])
    if folder.exists():
        bundle, pins = _load_progress(folder, buffers, context, evidence, stop)
        return dict(cached=True, frontier=HORIZON, prefix_sha256=_seal(bundle), cache_files_sha256=pins)
    _stop(stop)
    path = buffers["root"] / original.REFERENCE / "condensation_0/checkpoints/step_000000.pt"
    origin = torch.load(path, map_location="cpu", weights_only=False)
    parameters = [p.detach().clone().requires_grad_() for p in buffers["initial"]]
    _count(evidence, "origin_connected_moment")
    current = probe._moments(buffers, parameters)
    _count(evidence, "origin_connected_moment", True)
    _require(torch.equal(current.detach(), origin["moments"].to(current)), "AX current native P0 differs from original cached NODE0")
    theta = origin["theta"].to(current)
    optimizer = torch.optim.Adam(parameters, lr=.01, betas=(.9, .999), eps=1e-12, weight_decay=0, foreach=False, fused=False)
    states, scale, head_work = {}, None, dict(cached_original_NODE0=True, head_fits=0)
    first, second = [torch.zeros_like(p) for p in parameters], [torch.zeros_like(p) for p in parameters]
    folder.mkdir(parents=True, exist_ok=False)
    probe._atomic(folder/"candidate.json", json.dumps(context["candidate"], indent=2), False)
    for step in range(HORIZON+1):
        _stop(stop)
        inherited._stable_files(buffers)
        state, scale = _state(buffers, parameters, theta, optimizer.state_dict(), step, scale, head_work, evidence, stop)
        states[step] = state
        probe._runtime_precision_guard()
        bundle = _store(folder, context, states, scale)
        if step == HORIZON:
            break
        expected, first, second = inherited._adam_step(parameters, first, second,
            [g.to(p) for g, p in zip(state["scaled_factor_gradients"], parameters, strict=True)], step+1)
        _stop(stop)
        _count(evidence, "P_update")
        optimizer.step()
        _count(evidence, "P_update", True)
        _require(all(torch.equal(p.detach(), target) for p, target in zip(parameters, expected, strict=True)), "Native AX Adam recurrence differs")
        inherited._optimizer(optimizer.state_dict(), parameters, first, second, step+1)
        _stop(stop)
        probe._runtime_precision_guard()
        _count(evidence, "head_prefit_moment")
        moments = probe._moments(buffers, parameters).detach()
        _count(evidence, "head_prefit_moment", True)
        centers, labels, weights, _, _ = aw._material(moments, buffers, parameters)
        _count(evidence, "endpoint_head_solve")
        result = solve_inner_newton_first(centers, labels, weights, .001, initial=theta, max_iter=2000,
            grad_tol=1e-7, cg_max_iter=512, newton_steps=8, cg_check_interval=1)
        _count(evidence, "endpoint_head_solve", True)
        _require(result["inner_converged"] is True, "AX uniform inner head did not converge")
        theta, head_work = result["theta"], {k: v for k, v in result.items() if k != "theta"}
    return dict(cached=False, frontier=HORIZON, own_J0=scale, prefix_sha256=_seal(bundle), cache_files_sha256=_cache_files(folder, complete=True))


def _gate(spec, cells, checksum):
    path = Path(spec["certificate_outputs"][str(cells)])
    _require(type(checksum) is str and len(checksum) == 64 and path.is_file() and _sha(path) == checksum, "Require exact accepted native25 gate SHA")
    gate = json.loads(path.read_text())
    expected = dict(passed=True, schema=1, assignment_steps=HORIZON, source=spec["source"], numerical_source=spec["numerical_source"],
        candidate=spec["candidate"], candidate_id=spec["candidate_id"], cells=cells, full25_cache_replay_passed=True,
        actual_FP32_P0_X_Q_uniform_equal_reference=True, scientific_preregistration=spec["scientific_preregistration"])
    _require(isinstance(gate, dict) and all(_exact(gate.get(k), v) for k, v in expected.items()), "Changed/partial AX native25 gate")
    _require(all(type(gate.get(k)) is str and len(gate[k]) == 64 for k in ("prefix_sha256", "spec_sha256")),
             "Gate lacks exact source-prefix/spec hashes")
    _require(isinstance(gate.get("files_sha256"), dict) and gate["files_sha256"], "Gate lacks exact input/cache pins")
    probe._checked_files(gate["files_sha256"])
    return gate


def _validate(spec, buffers, context, arm, gate_sha, evidence, stop):
    gate = _gate(spec, buffers["cells"], gate_sha)
    bundle, _ = _load_progress(_folder(spec, buffers["cells"]), buffers, context, evidence, stop)
    _require(_seal(bundle) == gate["prefix_sha256"], "Students require accepted actual AX25 prefix")
    reference = {s: torch.load(buffers["root"] / original.REFERENCE / f"condensation_0/checkpoints/step_{s:06d}.pt",
                              map_location="cpu", weights_only=False) for s in (0, HORIZON)}
    snapshots = reference if arm == "node_reference" else bundle["states"]
    rows = []
    for step in (0, HORIZON):
        cx, cq, mass = representative(snapshots[step]["moments"], probe._transform(buffers), 3703, "cuda")
        supplied = torch.full_like(mass, 1/buffers["cells"])
        if step == 0:
            rx, rq, rm = representative(reference[0]["moments"], probe._transform(buffers), 3703, "cuda")
            _require(all(torch.equal(a, b) for a, b in zip((cx, cq, supplied), (rx, rq, torch.full_like(rm, 1/buffers["cells"])), strict=True)),
                     "Actual FP32 shared P0 X/Q/uniform differs")
            directory = _folder(spec, buffers["cells"])/"shared_P0_validation"
        else:
            directory = _folder(spec, buffers["cells"])/f"{arm}25_validation"
        for seed in _SEEDS:
            row = inherited._evaluate_student(directory, (cx, cq, supplied), buffers, context, gate_sha, seed, stop)
            rows.append(dict(arm=arm, step=step, candidate_id=original.REFERENCE if arm == "node_reference" else spec["candidate_id"],
                             shared_P0=step == 0, physical_condition=str(directory.resolve())+f":{seed}", **row))
    evidence["counts"].update(logical_student_conditions=6, logical_serving_conditions=12,
        physical_final_GCN_fits=sum(r["actual_student_fits"] for r in rows),
        physical_sameweights_serving_outputs=sum(r["physical_route_outputs"] for r in rows),
        selected_state_diagnostic_route_forwards=sum(r["diagnostic_route_forwards"] for r in rows),
        final_optimizer_epochs=inherited._settings(buffers["cells"])["epochs"]*sum(r["actual_student_fits"] for r in rows))
    return dict(rows=rows, arm=arm, checkpoints=[0, HORIZON], student_seeds=list(_SEEDS), test_enabled=False,
                shared_P0_physical_reuse=True, secondary_same_selected_GCN_weights_and_epoch=True)


def _run(operation, cells, spec_path, spec_sha256, output_path, arm, gate_sha256, stop):
    inherited._budget(cells)
    _require(operation in ("prepare", "certify", "validate") and callable(stop), "Unsupported AX operation")
    _require((operation != "validate" and arm is None and gate_sha256 is None) or
             (operation == "validate" and type(arm) is str and arm in ("node_reference", "macro") and type(gate_sha256) is str), "Invalid fixed AX phase controls")
    spec_path = Path(spec_path).resolve()
    spec, science = _load_spec(spec_path, spec_sha256)
    output = Path(output_path).resolve()
    expected = Path(spec["certificate_outputs"][str(cells)]) if operation == "certify" else Path(spec["output_root"])/f"citeseer{cells}"/(
        "native_prepare25_v1.json" if operation == "prepare" else f"validation_{arm}_v1.json")
    _require(output == expected and not output.exists(), "Require new declared AX output")
    if operation == "validate":
        gate = _gate(spec, cells, gate_sha256)
        _require(gate["spec_sha256"] == spec_sha256, "Gate refers to another spec")
    started = time.monotonic()
    bounded = lambda: stop() or time.monotonic()-started >= 300
    evidence = dict(passed=False, operation=operation, arm=arm, cells=cells, schema=1, assignment_steps=HORIZON,
        source=spec["source"], numerical_source=spec["numerical_source"], candidate=spec["candidate"], candidate_id=spec["candidate_id"],
        scientific_preregistration=spec["scientific_preregistration"], spec_path=str(spec_path), spec_sha256=spec_sha256,
        counts=dict(P_update_attempts=0, P_update_completed=0, final_student_fit_attempts=0, final_student_fit_completed=0,
                    teacher_map_Phi_hardinit_fits=0, functional_inner_SGD_steps=0, extra_MLP_fits=0), test_enabled=False,
        stage="fresh_native_policy", gate_sha256=gate_sha256)
    native, primary = False, None
    try:
        _stop(bounded)
        _require(torch.get_num_threads() == 4 and not torch.cuda.is_initialized(), "Require threads4/fresh CUDA process")
        environment = dict(**probe._native("cuda"), python_version=platform.python_version(), threads=4)
        native = True
        torch.cuda.reset_peak_memory_stats()
        _require(torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory == original.CUDA_CAPACITY, "GPU capacity changed")
        evidence["native_environment"] = environment
        buffers, reference = inherited._load_source(cells, spec, science, evidence, bounded)
        context = _context(spec, cells, buffers, reference, environment)
        evidence["stage"] = operation
        if operation == "prepare":
            result = _prepare(spec, buffers, context, evidence, bounded)
        elif operation == "certify":
            bundle, pins = _load_progress(_folder(spec, cells), buffers, context, evidence, bounded)
            repo = Path(__file__).resolve().parents[1]
            inputs = {str(repo/p): h for p, h in spec["source"]["files"].items()}
            inputs.update(spec["files_sha256"])
            inputs.update(spec["artifacts_sha256"])
            inputs.update({str(spec_path): spec_sha256, spec["scientific_preregistration"]["path"]: SCIENCE_SHA})
            inputs.update({row["path"]: row["sha256"] for row in science["parents"]})
            inputs.update(pins)
            result = dict(full25_cache_replay_passed=True, actual_FP32_P0_X_Q_uniform_equal_reference=True,
                prefix_sha256=_seal(bundle), own_J0=bundle["J0"], files_sha256=inputs, frontier=HORIZON)
        else:
            result = _validate(spec, buffers, context, arm, gate_sha256, evidence, bounded)
        evidence.update(result)
        _stop(bounded)
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
            if primary is None: primary = error
        if native:
            try:
                torch.cuda.synchronize()
                capacity = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
                allocated, reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
                _require(capacity == original.CUDA_CAPACITY and allocated <= capacity and reserved <= capacity, "Invalid native memory/capacity")
                evidence.update(CUDA_peak_allocated_bytes=allocated, CUDA_peak_reserved_bytes=reserved, CUDA_total_bytes=capacity)
                probe._runtime_precision_guard()
            except BaseException as error:
                evidence.update(passed=False, native_finalization_error=repr(error))
                if primary is None: primary = error
        evidence["seconds"] = time.monotonic()-started
        try:
            _write_new(output, evidence)
        except BaseException:
            if primary is not None: raise primary
            raise
    if primary is not None: raise primary
    return dict(evidence, evidence_path=str(output), evidence_sha256=_sha(output))


def prepare(cells, spec_path, spec_sha256, output_path, stop=lambda: False):
    return _run("prepare", cells, spec_path, spec_sha256, output_path, None, None, stop)


def certify(cells, spec_path, spec_sha256, output_path, stop=lambda: False):
    return _run("certify", cells, spec_path, spec_sha256, output_path, None, None, stop)


def validate(cells, arm, spec_path, spec_sha256, output_path, gate_sha256, stop=lambda: False):
    return _run("validate", cells, spec_path, spec_sha256, output_path, arm, gate_sha256, stop)
