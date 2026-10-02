"""Fixed25 frozen CE-gradient alignment and native-gated students.

Each budget starts a fresh AZ trajectory. AY state and incomplete AZ caches
are never resumed. All actual Adam frames and mathematical states are replayed.
"""
import json
import math
import platform
import time
from pathlib import Path

import torch

from src import citation_gradient_probe as ay
from src import citation_source_certificate as original
from src import citeseer_finite_student_v2 as inherited
from src import finite_student_probe as probe
from src.ce_gradient_alignment import POLICY, gradient_alignment_partials
from src.citation_source_preflight import _exact, _write_new
from src.io import _fingerprint
from src.research_loop import implementation_provenance
from src.sweep_utils import representative

SCIENCE = "Citeseer30_120_CE_gradient_alignment_fixed25_scientific_stageAZ_v2.json"
SCIENCE_SHA = "f10a36b8dccf2fd50ce7d9f2a72aa4dc599e190b7b137cd3fdc490491b8740d3"
HORIZON = 25
_FIXED = {key: value for key, value in ay._FIXED.items() if key != "probe_updates"}
_FIXED.update(method="frozen_GCN_CE_gradient_alignment_fixed25", assignment_steps=HORIZON)
_SEEDS = (3900, 3901, 3902)
_require, _stop, _tensor, _sha, _seal = probe._require, probe._stop, probe._tensor, probe._sha, probe._seal
_count = inherited._count


def numerical_source():
    result = ay.numerical_source()
    _require(result["files"].get("citation_gradient_fixed25.py") == _sha(__file__), "AZ module source changed")
    return result


def canonical_candidate():
    return dict(_FIXED, gradient_source_digest=_seal(numerical_source()))


def _preserve(spec, path, checksum, science):
    _require(_sha(path) == checksum, "Frozen AZ spec changed")
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
             and path.is_relative_to(repo / "results/proposals") and _sha(path) == checksum, "Require frozen AZ spec path/SHA")
    spec = json.loads(path.read_text())
    fields = {"schema", "scientific_preregistration", "source", "numerical_source", "python_version", "files_sha256",
              "roots", "candidate", "candidate_id", "artifacts_sha256", "output_root", "certificate_outputs", "gradient_policy"}
    _require(isinstance(spec, dict) and set(spec) == fields and type(spec["schema"]) is int and spec["schema"] == 1
             and _exact(spec["candidate"], canonical_candidate()) and spec["candidate_id"] == _fingerprint(spec["candidate"])
             and _exact(spec["gradient_policy"], json.loads(json.dumps(POLICY)))
             and spec["roots"] == original._roots(repo), "Changed AZ candidate/roots/schema or unknown controls")
    reference = dict(path=str(repo / "results/proposals" / SCIENCE), sha256=SCIENCE_SHA)
    _require(spec["scientific_preregistration"] == reference, "Wrong fixed AZ science")
    probe._checked_files({reference["path"]: SCIENCE_SHA})
    science = json.loads(Path(reference["path"]).read_text())
    _require(_exact(science["candidate"], _FIXED) and spec["files_sha256"] == science["original_files_sha256"]
             and science["evaluation"]["student_seeds"] == list(_SEEDS), "Changed AZ science or original assets")
    _require(isinstance(spec["artifacts_sha256"], dict) and spec["artifacts_sha256"]
             and all(Path(p).resolve().is_relative_to(repo / "results") for p in spec["artifacts_sha256"]), "Require reviewed artifact pins")
    root = Path(spec["output_root"]).resolve()
    _require(str(root) == spec["output_root"] and root.is_relative_to(repo / "results/research_loop"), "Invalid AZ output root")
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
        student_recipe_origin=buffers["recipe_origin"], policy=json.loads(json.dumps(POLICY)), assignment_steps=HORIZON,
        source_buffers=probe._digest(dict(z=buffers["z"], Q=buffers["q"], H=buffers["h"], hard=buffers["hard"],
            transform=buffers["transform"], X=buffers["x"], original_CSR=buffers["original_S"], dense_original_S=buffers["S"],
            initial=buffers["initial"])), no_AY_state_reuse=True)


















def _native_targets(saved, device):
    _require(isinstance(saved, dict) and set(saved) == {"policy", "anchors"}
             and _exact(saved["policy"], json.loads(json.dumps(POLICY))), "Changed saved target policy")
    def move(value):
        if torch.is_tensor(value):
            return value.detach().to(device)
        if isinstance(value, dict):
            return {key: move(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [move(item) for item in value]
        return value
    result = move(saved)
    result["policy"] = dict(POLICY)  # Strict serialized policy checked above.
    return result


def _verify_targets(folder, buffers, context, evidence, stop):
    path = folder / "source_gradient_targets.pt"
    checksum = _sha(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    _require(isinstance(payload, dict) and set(payload) == {"schema", "context", "targets", "content_sha256"}
             and type(payload["schema"]) is int and payload["schema"] == 1 and payload["context"] == context
             and _seal({k:v for k,v in payload.items() if k != "content_sha256"}) == payload["content_sha256"],
             "Changed source-gradient target cache/context")
    targets = _native_targets(payload["targets"], buffers["z"].device)
    diagnostic = dict(counts={})
    try:
        replay = ay._targets(buffers, diagnostic, stop)
    finally:
        for key, value in diagnostic["counts"].items():
            evidence["counts"]["target_replay_"+key] = evidence["counts"].get("target_replay_"+key,0)+value
    _require(probe._digest(targets) == probe._digest(replay) and _sha(path) == checksum,
             "Saved source gradients/models differ from actual original source/GEOM targets")
    return targets, checksum


def _state(buffers, parameters, optimizer, step, scale, targets, evidence, stop):
    _stop(stop)
    probe._runtime_precision_guard()
    for parameter in parameters:
        parameter.grad = None
    _count(evidence, "connected_moment")
    moments = probe._moments(buffers, parameters)
    _count(evidence, "connected_moment", True)
    _count(evidence, "material_certificate")
    _, _, _, logits, conservation = ay._material(moments.detach(), buffers, parameters)
    _count(evidence, "material_certificate", True)
    _count(evidence, "alignment_partial")
    result = gradient_alignment_partials(moments, buffers["transform"], targets, stop=stop)
    _count(evidence, "alignment_partial", True)
    value = float(result["loss"])
    scale = value if scale is None else scale
    _require(type(scale) in (int,float) and math.isfinite(scale) and scale > 0, "Invalid fresh original J0")
    snapshot = dict(step=step, parameters=parameters, optimizer=optimizer, alignment_J=value, J0=scale,
        moments=moments, moment_partial=result["moment_gradient"], anchor_diagnostics=result["anchors"],
        representative_digests=probe._digest(dict(Hc=result["Hc"],Qc=result["Qc"],mass=result["mass"])),
        raw_logits_digest=probe._digest(logits), conservation=conservation, terminal_no_update=step==HORIZON)
    if step < HORIZON:
        _count(evidence, "original_moment_backward")
        moments.backward(result["moment_gradient"]/scale)
        _count(evidence, "original_moment_backward", True)
        _require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in parameters),
                 "Missing/nonfinite original factor gradient")
        snapshot["scaled_factor_gradients"] = [p.grad for p in parameters]
    probe._runtime_precision_guard()
    return probe.cpu_state(snapshot), scale


def _history(states):
    return [dict(step=step,alignment_J=s["alignment_J"],J0=s["J0"],conservation=s["conservation"],
                 terminal_no_update=s["terminal_no_update"]) for step,s in sorted(states.items())]


def _check_progress(bundle, buffers, context, targets, target_sha, evidence, stop):
    _require(isinstance(bundle,dict) and set(bundle)=={
        "schema","context","frontier","states","J0","history","target_sha256","target_digest","content_sha256"}
        and type(bundle["schema"]) is int and bundle["schema"]==1 and bundle["context"]==context
        and bundle["target_sha256"]==target_sha and bundle["target_digest"]==probe._digest(targets)
        and _seal({k:v for k,v in bundle.items() if k!="content_sha256"})==bundle["content_sha256"],
        "Changed AZ progress/target/context seal")
    frontier,states=bundle["frontier"],bundle["states"]
    _require(type(frontier) is int and 0<=frontier<=HORIZON and isinstance(states,dict)
        and all(type(k) is int for k in states) and set(states)==set(range(frontier+1)),"Malformed AZ prefix")
    expected=[p.detach().clone() for p in buffers["initial"]]
    first,second=[torch.zeros_like(p) for p in expected],[torch.zeros_like(p) for p in expected]
    scale=None; before=probe._digest(targets)
    for step,saved in sorted(states.items()):
        _stop(stop)
        _require(isinstance(saved,dict) and type(saved.get("step")) is int and saved["step"]==step
                 and isinstance(saved.get("parameters"),list) and len(saved["parameters"])==2
                 and isinstance(saved.get("optimizer"),dict),"Malformed AZ state")
        for actual,target in zip(saved["parameters"],expected,strict=True):
            _tensor(actual,target.shape,torch.float32,"Malformed native factor")
            _require(torch.equal(actual.to(target),target),"Factor differs from initial/previous actual Adam")
        inherited._optimizer(saved["optimizer"],expected,first,second,step)
        actual,scale=_state(buffers,[p.clone().requires_grad_() for p in expected],saved["optimizer"],
                            step,scale,targets,evidence,stop)
        _require(_seal(actual)==_seal(saved),"Coupled M/J0/partial/gradient/Adam state changed")
        if step<frontier:
            expected,first,second=inherited._adam_step(expected,first,second,
                [g.to(p) for g,p in zip(actual["scaled_factor_gradients"],expected,strict=True)],step+1)
    _require(type(bundle["J0"]) in (int,float) and bundle["J0"]==scale and isinstance(bundle["history"],list)
        and len(bundle["history"])==len(states) and all(_exact(a,b) for a,b in zip(bundle["history"],_history(states),strict=True))
        and probe._digest(targets)==before,"Changed original J0/history/frozen targets")
    return bundle


def _cache_files(folder, *, complete):
    names={"candidate.json","progress.pt","history.json","source_gradient_targets.pt","step_000000.pt"}
    if complete:names.add("step_000025.pt")
    _require({p.name for p in folder.iterdir() if p.is_file()}==names,"Partial/orphan AZ cache preserved")
    return {str((folder/name).resolve()):_sha(folder/name) for name in names}


def _origin_check(moments, buffers, reference, evidence):
    origin=torch.load(buffers["root"]/original.REFERENCE/"condensation_0/checkpoints/step_000000.pt",
                      map_location="cpu",weights_only=False)
    _require(isinstance(origin,dict),"Malformed original NODE0")
    cached=_tensor(origin.get("moments"),moments.shape,torch.float64,"Malformed original moments")
    _require(torch.equal(moments.detach(),cached.to(moments)),"Current native P0 differs from exact cached NODE0")
    _count(evidence,"readonly_P0_representative")
    x,q,mass=representative(moments.detach(),probe._transform(buffers),buffers["z"].shape[1],buffers["z"].device)
    _count(evidence,"readonly_P0_representative",True)
    _require(reference["actual_FP32_P0_X_Q_uniform_equal_reference"] is True and
        probe._digest([x,q,torch.full_like(mass,1/buffers["cells"])])==reference["student_inputs"],
        "Actual native shared P0 inputs differ from AT")


def _load_progress(folder, buffers, context, evidence, stop):
    _require(folder.is_dir(),"Require completed AZ candidate cache")
    pins=_cache_files(folder,complete=True)
    _require(json.loads((folder/"candidate.json").read_text())==context["candidate"],"Changed AZ candidate")
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


def _store(folder, context, states, scale, target_sha, targets):
    bundle=probe._attach(dict(schema=1,context=context,frontier=max(states),states=states,J0=scale,
        history=_history(states),target_sha256=target_sha,target_digest=probe._digest(targets)))
    probe._atomic(folder/"progress.pt",bundle,True)
    probe._atomic(folder/"history.json",json.dumps(bundle["history"],indent=2),False)
    if max(states) in (0,HORIZON):probe._atomic(folder/f"step_{max(states):06d}.pt",states[max(states)],True)
    return bundle


def _prepare(spec, buffers, context, evidence, stop):
    folder=_folder(spec,buffers["cells"])
    if folder.exists():
        bundle,pins=_load_progress(folder,buffers,context,evidence,stop)
        return dict(cached=True,frontier=HORIZON,prefix_sha256=_seal(bundle),cache_files_sha256=pins)
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


def _gate(spec, cells, checksum):
    path = Path(spec["certificate_outputs"][str(cells)])
    _require(type(checksum) is str and len(checksum) == 64 and path.is_file() and _sha(path) == checksum, "Require exact accepted native25 gate SHA")
    gate = json.loads(path.read_text())
    expected = dict(passed=True, schema=1, assignment_steps=HORIZON, source=spec["source"], numerical_source=spec["numerical_source"],
        candidate=spec["candidate"], candidate_id=spec["candidate_id"], cells=cells, full25_cache_replay_passed=True,
        actual_FP32_P0_X_Q_uniform_equal_reference=True, source_target_cache_replay_passed=True,
        gradient_policy=spec["gradient_policy"], scientific_preregistration=spec["scientific_preregistration"])
    _require(isinstance(gate, dict) and all(_exact(gate.get(k), v) for k, v in expected.items()), "Changed/partial AZ native25 gate")
    _require(all(type(gate.get(k)) is str and len(gate[k]) == 64 for k in ("prefix_sha256", "spec_sha256")),
             "Gate lacks exact source-prefix/spec hashes")
    _require(isinstance(gate.get("files_sha256"), dict) and gate["files_sha256"], "Gate lacks exact input/cache pins")
    probe._checked_files(gate["files_sha256"])
    return gate


def _validate(spec, buffers, context, arm, gate_sha, evidence, stop):
    gate = _gate(spec, buffers["cells"], gate_sha)
    bundle, _ = _load_progress(_folder(spec, buffers["cells"]), buffers, context, evidence, stop)
    _require(_seal(bundle) == gate["prefix_sha256"], "Students require accepted actual AZ25 prefix")
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
    _require(operation in ("prepare", "certify", "validate") and callable(stop), "Unsupported AZ operation")
    _require((operation != "validate" and arm is None and gate_sha256 is None) or
             (operation == "validate" and type(arm) is str and arm in ("node_reference", "gradient") and type(gate_sha256) is str), "Invalid fixed AZ phase controls")
    spec_path = Path(spec_path).resolve()
    spec, science = _load_spec(spec_path, spec_sha256)
    output = Path(output_path).resolve()
    expected = Path(spec["certificate_outputs"][str(cells)]) if operation == "certify" else Path(spec["output_root"])/f"citeseer{cells}"/(
        "native_prepare25_v1.json" if operation == "prepare" else f"validation_{arm}_v1.json")
    _require(output == expected and not output.exists(), "Require new declared AZ output")
    if operation == "validate":
        gate = _gate(spec, cells, gate_sha256)
        _require(gate["spec_sha256"] == spec_sha256, "Gate refers to another spec")
    started = time.monotonic()
    bounded = lambda: stop() or time.monotonic()-started >= 300
    evidence = dict(passed=False, operation=operation, arm=arm, cells=cells, schema=1, assignment_steps=HORIZON,
        source=spec["source"], numerical_source=spec["numerical_source"], candidate=spec["candidate"], candidate_id=spec["candidate_id"],
        scientific_preregistration=spec["scientific_preregistration"], spec_path=str(spec_path), spec_sha256=spec_sha256,
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
        buffers, reference = inherited._load_source(cells, spec, science, evidence, bounded)
        evidence["source_buffer_digest_before"] = ay._source_buffer_digest(buffers)
        evidence["seed0_model_digest_before"] = probe._digest(buffers["model_initial"])
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
