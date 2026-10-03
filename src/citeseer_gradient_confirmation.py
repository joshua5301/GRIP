"""Fresh Citeseer120 condensation3/4/5 confirmation; isolated prospective draft.

Only new namespace/context/cohort wrappers; protected gradient and student math.
Frozen scientific controls; native qualification remains a separate experiment.
"""
import json
import math
import platform
import time
from pathlib import Path

import torch

from src import citation_gradient_fixed25 as az
from src import citation_gradient_probe as ay
from src import citation_gradient_replication as replication
from src import citation_source_certificate as original
from src import citeseer_confirmation_source as source_certificate
from src import citeseer_finite_student_v2 as inherited
from src import finite_student_probe as probe
from src.ce_gradient_alignment import POLICY
from src.citation_source_preflight import _exact, _write_new
from src.io import _fingerprint
from src.research_loop import implementation_provenance
from src.sweep_utils import representative

SCIENCE = "Citeseer120_CE_gradient_alignment_fixed25_confirmation_scientific_stageBP_v3.json"
SCIENCE_SHA = "82fcc248e9a12233fa513f8a6e383141a0c906658f3647068a424e12ec1a070e"
HORIZON = 25
_SEEDS = (4300, 4301, 4302, 4303, 4304)
_require, _stop, _tensor, _sha, _seal = probe._require, probe._stop, probe._tensor, probe._sha, probe._seal
_count = inherited._count
_state, _check_progress, _verify_targets, _store, _cache_files = az._state, az._check_progress, az._verify_targets, az._store, az._cache_files
_origin_check = replication._origin_check
_source_load, _origin, _endpoints = source_certificate._load_source, source_certificate._origin, source_certificate._endpoints


def _cond(value):
    _require(type(value) is int and value in (3, 4, 5), "Only fixed condensation3/4/5")
    return value


def numerical_source():
    value = az.numerical_source()
    _require(value["files"].get("citeseer_gradient_confirmation.py") == _sha(__file__), "Own confirmation source changed")
    return value


def canonical_candidate(condensation_seed):
    c = _cond(condensation_seed)
    fixed = dict(az._FIXED, method="frozen_GCN_CE_gradient_alignment_fixed25_confirmation", confirmation_schema=1, condensation_seed=c)
    return dict(fixed, gradient_source_digest=_seal(numerical_source()))


def _preserve(spec, path, checksum, science):
    _require(_sha(path) == checksum, "Confirmation spec changed")
    probe._checked_files(spec["files_sha256"])
    probe._checked_files(spec["artifacts_sha256"])
    probe._checked_files({spec["scientific_preregistration"]["path"]: SCIENCE_SHA})
    probe._checked_files(source_certificate._reference_pins(science))
    _require(implementation_provenance() == spec["source"] and numerical_source() == spec["numerical_source"]
             and platform.python_version() == spec["python_version"] and torch.get_num_threads() == 4,
             "Current source/Git/versions/Python/threads changed")


def _load_spec(path, checksum):
    repo, path = Path(__file__).resolve().parents[1], Path(path).resolve()
    _require(type(checksum) is str and len(checksum) == 64 and path.is_file()
             and path.is_relative_to(repo / "results/proposals") and _sha(path) == checksum, "Frozen confirmation spec required")
    spec = json.loads(path.read_text())
    fields = {"schema", "scientific_preregistration", "source", "numerical_source", "python_version", "files_sha256", "fixed",
              "candidates", "candidate_ids", "gradient_policy", "output_root", "certificate_outputs", "artifacts_sha256",
              "condensation_seeds", "source_certificate_refs"}
    candidates = {str(c): canonical_candidate(c) for c in (3, 4, 5)}
    reference = dict(path=str(repo / "results/proposals" / SCIENCE), sha256=SCIENCE_SHA)
    _require(isinstance(spec, dict) and set(spec) == fields and type(spec["schema"]) is int and spec["schema"] == 1
             and _exact(spec["candidates"], candidates) and spec["candidate_ids"] == {k: _fingerprint(v) for k, v in candidates.items()}
             and _exact(spec["condensation_seeds"], [3, 4, 5])
             and _exact(spec["gradient_policy"], json.loads(json.dumps(POLICY)))
             and spec["scientific_preregistration"] == reference, "Changed fixed confirmation controls")
    probe._checked_files({reference["path"]: SCIENCE_SHA})
    science = json.loads(Path(reference["path"]).read_text())
    fixed = {c: {k: v for k, v in candidate.items() if k != "gradient_source_digest"} for c, candidate in candidates.items()}
    _require(_exact(science["candidates"], fixed) and _exact(spec["fixed"], science["fixed"])
             and spec["files_sha256"] == science["original_files_sha256"]
             and spec["source_certificate_refs"] == science["source_certificate_refs"]
             and science["evaluation"]["student_seeds"] == list(_SEEDS), "Changed science/source/cohort certificates")
    owned = {repo / "src/citeseer_gradient_confirmation.py", repo / "src/research_loop.py", repo / "tests/test_citeseer_gradient_confirmation.py"}
    _require(isinstance(spec["artifacts_sha256"], dict) and spec["artifacts_sha256"]
             and all(Path(p).is_absolute() and (Path(p).resolve().is_relative_to(repo / "results") or Path(p).resolve() in owned)
                     for p in spec["artifacts_sha256"]), "Undeclared reviewed artifact")
    root = Path(spec["output_root"]).resolve()
    _require(str(root) == spec["output_root"] == science["output_root"]
             and root.is_relative_to(repo / "results/research_loop")
             and spec["certificate_outputs"] == {str(c): str(root / f"condensation_{c}/native_certificate25_v1.json") for c in (3, 4, 5)},
             "Changed fixed output/gate namespace")
    _preserve(spec, path, checksum, science)
    return spec, science


def _load_source(condensation_seed, spec, science, evidence, stop):
    c = _cond(condensation_seed)
    raw = _source_load(c, spec, science, evidence, stop)
    ref = spec["source_certificate_refs"][str(c)]
    probe._checked_files({ref["path"]: ref["sha256"]})
    accepted = json.loads(Path(ref["path"]).read_text())
    _require(accepted.get("passed") is True and accepted.get("source_assets_spec_science_unchanged") is True
             and type(accepted.get("condensation_seed")) is int and accepted["condensation_seed"] == c
             and accepted.get("complete25_native_certificate_passed") is True
             and accepted.get("actual_P0_moments_bitwise_equal_core_step0") is True
             and accepted.get("actual_FP32_P0_X_Q_F64_uniform_equal_core_step0") is True
             and accepted.get("source_context") == raw["source"]
             and accepted.get("native_buffers_after") == source_certificate._native_buffers(raw)
             and accepted.get("student_recipe_origin") == evidence["student_recipe_origin"], "Current own source differs from qualified BO")
    probe._checked_files(accepted["certified_files_sha256"])
    origin = _origin(raw, c, evidence)
    folder = raw["root"] / original.REFERENCE / f"condensation_{c}"
    resume = torch.load(folder / "resume.pt", map_location="cpu", weights_only=False)
    retained = resume.get("config", {}).get("save_assignment") if isinstance(resume, dict) else None
    _require(type(retained) is bool, "Original retention policy malformed")
    expected = source_certificate._core_config(raw["ghost"], c, raw)
    expected["save_assignment"] = retained
    _endpoints(raw, origin, expected, folder, evidence)
    _require(evidence["certified_files_sha256"] == accepted["certified_files_sha256"], "Own endpoint certificate bytes changed")
    _count(evidence, "model_GEOM_factory")
    model = probe.geom_uniform_initial(3703, 6, hidden=256, dtype=torch.float32, device="cuda")
    _count(evidence, "model_GEOM_factory", True)
    buffers = dict(raw, cells=120, condensation_seed=c, x=raw["graph"]["x"].detach(), S=raw["dense"],
        original_S=raw["graph"]["adj"].detach(), transform=probe._frozen_transform(raw["transform"]),
        initial=[p.detach() for p in origin["parameters"]], model_initial=model,
        recipe_origin=spec["fixed"]["student_recipe"], counts=evidence["counts"], pins=spec["files_sha256"])
    buffers["file_stats"] = {p: (Path(p).stat().st_size, Path(p).stat().st_mtime_ns, Path(p).stat().st_ino) for p in buffers["pins"]}
    reference = dict(condensation_seed=c, source_reference_certificate_passed=True,
        actual_FP32_P0_X_Q_uniform_equal_reference=True, student_inputs=probe._digest(origin["inputs"]),
        input_digest=origin["input_digest"], own_BO_certificate=ref, original_linear_certificates=evidence["endpoint_certificates"],
        current_native_M0_exactly_equal_own_cached_NODE0=True, historical_checkpoint_UV_available=False)
    evidence.update(source_reference_certificate_passed=True, source_reference=reference,
                    own_BO_certificate=ref, original_source_certificate_only=True)
    return buffers, reference



def _folder(spec, condensation_seed):
    c = _cond(condensation_seed)
    return Path(spec["output_root"]) / f"condensation_{c}" / spec["candidate_ids"][str(c)]


def _context(spec, condensation_seed, buffers, reference, environment):
    c = _cond(condensation_seed)
    _require(reference.get("source_reference_certificate_passed") is True and reference.get("condensation_seed") == c,
             "Own condensation source/reference certificate required")
    return dict(schema=1,candidate=spec["candidates"][str(c)],cells=120,condensation_seed=c,implementation=spec["source"],
        numerical_source=spec["numerical_source"],spec=spec,scientific_preregistration=spec["scientific_preregistration"],
        native_environment=environment,ghost_source=buffers["source"],reference=reference,student_recipe_origin=buffers["recipe_origin"],
        policy=json.loads(json.dumps(POLICY)),assignment_steps=25,
        source_buffers=ay._source_buffer_digest(buffers),no_AZ_AY_state_reuse=True)


def _load_progress(folder, buffers, context, evidence, stop):
    _require(folder.is_dir(),"Require completed BP candidate cache")
    pins=_cache_files(folder,complete=True)
    _require(json.loads((folder/"candidate.json").read_text())==context["candidate"],"Changed BP candidate")
    targets,target_sha=_verify_targets(folder,buffers,context,evidence,stop)
    bundle=torch.load(folder/"progress.pt",map_location="cpu",weights_only=False)
    _check_progress(bundle,buffers,context,targets,target_sha,evidence,stop)
    _require(bundle["frontier"]==HORIZON,"Partial trajectory preserved; no resume/fallback")
    _origin_check(bundle["states"][0]["moments"].to(buffers["z"]),buffers,context["reference"],evidence)
    _require(json.loads((folder/"history.json").read_text())==bundle["history"],"Changed history mirror")
    for step in (0,HORIZON):
        mirror=torch.load(folder/f"step_{step:06d}.pt",map_location="cpu",weights_only=False)
        _require(_seal(mirror)==_seal(bundle["states"][step]),"Endpoint mirror differs from authority")
    _require(_cache_files(folder,complete=True)==pins,"Readonly cache replay changed files")
    return bundle,pins


def _prepare(spec, buffers, context, evidence, stop):
    folder=_folder(spec,buffers["condensation_seed"])
    if folder.exists():
        bundle,pins=_load_progress(folder,buffers,context,evidence,stop)
        return dict(cached=True,frontier=HORIZON,prefix_sha256=_seal(bundle),cache_files_sha256=pins)
    _require(context["reference"]["source_reference_certificate_passed"] is True
             and context["condensation_seed"] == buffers["condensation_seed"], "Own source certificate required before any P update")
    parameters=[p.detach().clone().requires_grad_() for p in buffers["initial"]]
    _count(evidence,"origin_connected_moment")
    current=probe._moments(buffers,parameters)
    _count(evidence,"origin_connected_moment",True)
    _origin_check(current,buffers,context["reference"],evidence)
    targets=ay._targets(buffers,evidence,stop)
    target_digest=probe._digest(targets)
    _count(evidence,"P_optimizer_constructor")
    optimizer=torch.optim.Adam(parameters,lr=.01,betas=(.9,.999),eps=1e-12,weight_decay=0,foreach=False,fused=False)
    _count(evidence,"P_optimizer_constructor",True)
    first,second=[torch.zeros_like(p) for p in parameters],[torch.zeros_like(p) for p in parameters]
    folder.mkdir(parents=True,exist_ok=False)
    probe._atomic(folder/"candidate.json",json.dumps(context["candidate"],indent=2),False)
    probe._atomic(folder/"source_gradient_targets.pt",probe._attach(dict(schema=1,context=context,
                  targets=probe.cpu_state(targets))),True)
    target_sha=_sha(folder/"source_gradient_targets.pt")
    states,scale={},None
    for step in range(HORIZON+1):
        _stop(stop);inherited._stable_files(buffers)
        state,scale=_state(buffers,parameters,optimizer.state_dict(),step,scale,targets,evidence,stop)
        states[step]=state
        _require(probe._digest(targets)==target_digest,"Immutable source targets/models changed")
        bundle=_store(folder,context,states,scale,target_sha,targets)
        if step==HORIZON:break
        expected,first,second=inherited._adam_step(parameters,first,second,
            [g.to(p) for g,p in zip(state["scaled_factor_gradients"],parameters,strict=True)],step+1)
        _stop(stop);probe._runtime_precision_guard()
        _count(evidence,"P_update");optimizer.step();_count(evidence,"P_update",True)
        _require(all(torch.equal(p.detach(),t) for p,t in zip(parameters,expected,strict=True)),"Native Adam recurrence differs")
        inherited._optimizer(optimizer.state_dict(),parameters,first,second,step+1)
    _require(_sha(folder/"source_gradient_targets.pt")==target_sha,"Frozen targetcache file changed")
    return dict(cached=False,frontier=HORIZON,own_J0=scale,prefix_sha256=_seal(bundle),
        cache_files_sha256=_cache_files(folder,complete=True),target_model_digest_before=target_digest,
        target_model_digest_after=probe._digest(targets))


def _gate(spec, condensation_seed, checksum):
    c = _cond(condensation_seed)
    path = Path(spec["certificate_outputs"][str(c)])
    _require(type(checksum) is str and len(checksum) == 64 and path.is_file() and _sha(path) == checksum, "Require exact accepted native25 gate SHA")
    gate = json.loads(path.read_text())
    expected = dict(passed=True, schema=1, assignment_steps=HORIZON, source=spec["source"], numerical_source=spec["numerical_source"],
        candidate=spec["candidates"][str(c)], candidate_id=spec["candidate_ids"][str(c)], cells=120, condensation_seed=c, full25_cache_replay_passed=True,
        actual_FP32_P0_X_Q_uniform_equal_reference=True, source_reference_certificate_passed=True, common_AT_native_buffers_exact_excluding_hard=True, source_target_cache_replay_passed=True,
        gradient_policy=spec["gradient_policy"], scientific_preregistration=spec["scientific_preregistration"])
    _require(isinstance(gate, dict) and all(_exact(gate.get(k), v) for k, v in expected.items()), "Changed/partial BP native25 gate")
    _require(all(type(gate.get(k)) is str and len(gate[k]) == 64 for k in ("prefix_sha256", "spec_sha256")),
             "Gate lacks exact source-prefix/spec hashes")
    _require(isinstance(gate.get("files_sha256"), dict) and gate["files_sha256"], "Gate lacks exact input/cache pins")
    probe._checked_files(gate["files_sha256"])
    return gate


def _validate(spec, buffers, context, arm, gate_sha, evidence, stop):
    gate = _gate(spec, buffers["condensation_seed"], gate_sha)
    bundle, _ = _load_progress(_folder(spec, buffers["condensation_seed"]), buffers, context, evidence, stop)
    _require(_seal(bundle) == gate["prefix_sha256"], "Students require accepted actual BP25 prefix")
    reference = {s: torch.load(buffers["root"] / original.REFERENCE / f"condensation_{buffers['condensation_seed']}/checkpoints/step_{s:06d}.pt",
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
            directory = _folder(spec, buffers["condensation_seed"])/"shared_P0_validation"
        else:
            directory = _folder(spec, buffers["condensation_seed"])/f"{arm}25_validation"
        for seed in _SEEDS:
            row = inherited._evaluate_student(directory, (cx, cq, supplied), buffers, context, gate_sha, seed, stop)
            rows.append(dict(arm=arm, step=step, condensation_seed=buffers["condensation_seed"], candidate_id=original.REFERENCE if arm == "node_reference" else spec["candidate_ids"][str(buffers["condensation_seed"])],
                             shared_P0=step == 0, physical_condition=str(directory.resolve())+f":{seed}", **row))
    evidence["counts"].update(logical_student_conditions=10, logical_serving_conditions=20,
        physical_final_GCN_fits=sum(r["actual_student_fits"] for r in rows),
        physical_sameweights_serving_outputs=sum(r["physical_route_outputs"] for r in rows),
        selected_state_diagnostic_route_forwards=sum(r["diagnostic_route_forwards"] for r in rows),
        final_optimizer_epochs=inherited._settings(buffers["cells"])["epochs"]*sum(r["actual_student_fits"] for r in rows))
    return dict(rows=rows, arm=arm, condensation_seed=buffers["condensation_seed"], checkpoints=[0, HORIZON], student_seeds=list(_SEEDS), test_enabled=False,
                shared_P0_physical_reuse=True, secondary_same_selected_GCN_weights_and_epoch=True)


def _run(operation, condensation_seed, spec_path, spec_sha256, output_path, arm, gate_sha256, stop):
    c = _cond(condensation_seed)
    _require(operation in ("prepare", "certify", "validate") and callable(stop), "Unsupported BP operation")
    _require((operation != "validate" and arm is None and gate_sha256 is None) or
             (operation == "validate" and type(arm) is str and arm in ("node_reference", "gradient") and type(gate_sha256) is str), "Invalid fixed BP phase controls")
    spec_path = Path(spec_path).resolve()
    spec, science = _load_spec(spec_path, spec_sha256)
    output = Path(output_path).resolve()
    expected = Path(spec["certificate_outputs"][str(c)]) if operation == "certify" else Path(spec["output_root"])/f"condensation_{c}"/(
        "native_prepare25_v1.json" if operation == "prepare" else f"validation_{arm}_v1.json")
    _require(output == expected and not output.exists(), "Require new declared BP output")
    if operation == "validate":
        gate = _gate(spec, c, gate_sha256)
        _require(gate["spec_sha256"] == spec_sha256, "Gate refers to another spec")
    started = time.monotonic()
    bounded = lambda: stop() or time.monotonic()-started >= 300
    evidence = dict(passed=False, operation=operation, arm=arm, cells=120, condensation_seed=c, schema=1, assignment_steps=HORIZON,
        source=spec["source"], numerical_source=spec["numerical_source"], candidate=spec["candidates"][str(c)], candidate_id=spec["candidate_ids"][str(c)],
        scientific_preregistration=spec["scientific_preregistration"], spec_path=str(spec_path), spec_sha256=spec_sha256,
        source_backend=spec["fixed"]["source_backend"],
        counts=dict(P_update_attempts=0, P_update_completed=0, final_student_fit_attempts=0, final_student_fit_completed=0,
                    teacher_map_Phi_hardinit_fits=0, functional_inner_SGD_steps=0, extra_MLP_fits=0,
                    fresh_head_fits=0, adjoint_or_Hessian_solves=0), test_enabled=False,
        gradient_policy=json.loads(json.dumps(POLICY)),
        stage="fresh_native_policy", gate_sha256=gate_sha256)
    native, primary, buffers = False, None, None
    try:
        _stop(bounded)
        _require(torch.get_num_threads() == 4 and not torch.cuda.is_initialized(), "Require threads4/fresh CUDA process")
        environment = dict(**probe._native("cuda"), python_version=platform.python_version(), threads=4)
        native = True
        torch.cuda.reset_peak_memory_stats()
        _require(torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory == original.CUDA_CAPACITY, "GPU capacity changed")
        evidence["native_environment"] = environment
        buffers, reference = _load_source(c, spec, science, evidence, bounded)
        evidence["source_buffer_digest_before"] = ay._source_buffer_digest(buffers)
        evidence["seed0_model_digest_before"] = probe._digest(buffers["model_initial"])
        context = _context(spec, c, buffers, reference, environment)
        evidence["stage"] = operation
        if operation == "prepare":
            result = _prepare(spec, buffers, context, evidence, bounded)
        elif operation == "certify":
            bundle, pins = _load_progress(_folder(spec, c), buffers, context, evidence, bounded)
            repo = Path(__file__).resolve().parents[1]
            inputs = {str(repo/p): h for p, h in spec["source"]["files"].items()}
            inputs.update(spec["files_sha256"])
            inputs.update(spec["artifacts_sha256"])
            inputs.update({str(spec_path): spec_sha256, spec["scientific_preregistration"]["path"]: SCIENCE_SHA})
            inputs.update(source_certificate._reference_pins(science))
            inputs.update(pins)
            result = dict(full25_cache_replay_passed=True, actual_FP32_P0_X_Q_uniform_equal_reference=True,
                prefix_sha256=_seal(bundle), own_J0=bundle["J0"], files_sha256=inputs, frontier=HORIZON,
                source_target_cache_replay_passed=True, target_model_digest=bundle["target_digest"],
                target_cache_sha256=bundle["target_sha256"])
        else:
            result = _validate(spec, buffers, context, arm, gate_sha256, evidence, bounded)
        evidence.update(result)
        _stop(bounded)
        evidence["passed"] = True
    except BaseException as error:
        primary = error
        evidence.update(error_type=type(error).__name__, error=str(error), failed_stage=evidence["stage"])
    finally:
        if buffers is not None:
            try:
                evidence["source_buffer_digest_after"] = ay._source_buffer_digest(buffers)
                evidence["seed0_model_digest_after"] = probe._digest(buffers["model_initial"])
                _require(evidence["source_buffer_digest_after"] == evidence["source_buffer_digest_before"]
                         and evidence["seed0_model_digest_after"] == evidence["seed0_model_digest_before"],
                         "Immutable native source/model buffers changed")
            except BaseException as error:
                evidence.update(passed=False,native_source_preservation_error=repr(error))
                if primary is None: primary = error
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
            _require(math.isfinite(evidence["seconds"]) and evidence["seconds"] <= 300, "Native job exceeded300seconds")
        except BaseException as error:
            evidence.update(passed=False, deadline_error=repr(error))
            if primary is None: primary = error
        try:
            _write_new(output, evidence)
        except BaseException:
            if primary is not None: raise primary
            raise
    if primary is not None: raise primary
    return dict(evidence, evidence_path=str(output), evidence_sha256=_sha(output), validation_only=True)


def prepare(condensation_seed, spec_path, spec_sha256, output_path, stop=lambda: False):
    return _run("prepare", condensation_seed, spec_path, spec_sha256, output_path, None, None, stop)


def certify(condensation_seed, spec_path, spec_sha256, output_path, stop=lambda: False):
    return _run("certify", condensation_seed, spec_path, spec_sha256, output_path, None, None, stop)


def validate(condensation_seed, arm, spec_path, spec_sha256, output_path, gate_sha256, stop=lambda: False):
    return _run("validate", condensation_seed, spec_path, spec_sha256, output_path, arm, gate_sha256, stop)
