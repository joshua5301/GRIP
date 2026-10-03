"""Fresh whole-layer25 versus fresh matched-.01 NODE25, with one paired cohort.

No disposable-probe resume, partial-cache repair or score-dependent operation.
Historical BN source context remains separate from the current frozen source.
"""
import json
import math
import platform
import time
from importlib.metadata import version as distribution_version
from pathlib import Path

import torch
from src.cora_whole_layer_students import evaluate_student

from src import citeseer_finite_student_v2 as inherited
from src import cora_node_reference_certificate_v2 as bg
from src import cora_whole_layer_probe as bq
from src import finite_student_probe as probe
from src import source_linear_assignment as source_helper
from src.citation_macro_probe import _material
from src.citation_source_preflight import _exact, _write_new
from src.citeseer_confirmation_source import _reference_pins
from src.io import _fingerprint
from src.low_rank_assignment import LowRankMoments
from src.moments import augmented, decode_moments, make_material
from src.research_loop import implementation_provenance
from src.soft_ce_partition import head_gradient, optimize_ce_assignment, outer_value_gradient
from src.sweep_utils import representative
from src.whole_layer_gradient_alignment import POLICY, moment_partials

SCIENCE = "Cora70_two_whole_layer_fixed25_matched_NODE_scientific_stageBR_v2.json"
SCIENCE_SHA = "1cf9d80ded2415fb27e91e9605fdc03bfd6e7575cd87137e759cd0a7eea288ee"
SCIENCE_REVIEW = "independent_scientific_review_stageBR_v2.json"
SCIENCE_REVIEW_SHA = "2a9f5fb5d66b3fc42a80685e8e0d0251cca9ba8e1dfdc13587ab0febe53d0bca"
_FIELDS = {"schema", "fixed", "case", "candidate", "candidate_id", "matched_NODE_candidate", "source", "numerical_source",
           "python_version", "files_sha256", "scientific_preregistration", "artifacts_sha256", "output_root",
           "certificate_outputs", "grouping_policy", "student_seeds"}
_POLICY_SPEC = json.loads(json.dumps(POLICY))
HORIZON, SEEDS = 25, (4400, 4401, 4402)
_require, _sha, _seal, _stop = probe._require, probe._sha, probe._seal, probe._stop
_load_source, _cached_targets, _retained = bq._load_source, bq._cached_targets, bq._retained
_history_metadata = bg._history_metadata


def numerical_source():
    value = bq.numerical_source()
    for name in ("cora_whole_layer_fixed25.py", "cora_whole_layer_students.py"):
        _require(value["files"].get(name) == _sha(Path(__file__).with_name(name)), "Owned BR source changed")
    return value


def _science(repo):
    path = repo / "results/proposals" / SCIENCE
    probe._checked_files({str(path): SCIENCE_SHA})
    return json.loads(path.read_text())


def canonical_candidate():
    return _science(Path(__file__).resolve().parents[1])["candidate"]


def _paths(repo):
    return [Path(p) for p in _science(repo)["original_files_sha256"]]


def _preserve(spec, path, checksum, science, repo):
    _require(_sha(path) == checksum, "Frozen spec changed")
    for pins in (spec["files_sha256"], spec["artifacts_sha256"], _reference_pins(science),
                 {spec["scientific_preregistration"]["path"]: SCIENCE_SHA, str(repo/"results/proposals"/SCIENCE_REVIEW): SCIENCE_REVIEW_SHA},
                 {str(repo/p):h for p,h in science["source_before"]["files"].items() if p != "src/research_loop.py"},
                 {str(repo/p):h for p,h in science["old_tests_sha256"].items()}):
        probe._checked_files(pins)
    expected = science["version_binding"]["installed_distribution_versions"]
    _require({k:distribution_version(k) for k in expected} == expected, "Distribution versions changed")
    _require(implementation_provenance() == spec["source"] and numerical_source() == spec["numerical_source"]
             and platform.python_version() == spec["python_version"] and torch.get_num_threads() == 4,
             "Source/Git/numerical/Python/threads changed")


def _spec_controls(spec, science, repo):
    _require(isinstance(spec,dict) and set(spec) == _FIELDS and type(spec["schema"]) is int and spec["schema"] == 1, "Unknown BR spec")
    for key in ("fixed","case","candidate","candidate_id","matched_NODE_candidate","output_root","certificate_outputs","student_seeds"):
        _require(_exact(spec[key],science[key]), "Changed fixed BR control: " + key)
    _require(_fingerprint(spec["candidate"]) == spec["candidate_id"] and _exact(spec["grouping_policy"],_POLICY_SPEC)
             and _exact(spec["files_sha256"],science["original_files_sha256"])
             and spec["scientific_preregistration"] == dict(path=str(repo/"results/proposals"/SCIENCE),sha256=SCIENCE_SHA), "Candidate/policy/assets/science changed")
    owned = {(repo/p).resolve() for p in science["source_protection"]["owned"]}
    pins = spec["artifacts_sha256"]
    _require(isinstance(pins,dict) and pins and all(type(p) is str and Path(p).is_absolute()
             and (Path(p).resolve().is_relative_to(repo/"results") or Path(p).resolve() in owned) for p in pins), "Require exact owned or results artifact pins")
    _require(set(spec["source"].get("files",{})) == set(science["source_before"]["files"]) | {p for p in science["source_protection"]["owned"] if p.startswith("src/")}
             and spec["python_version"] == science["version_binding"]["python_version"]
             and spec["numerical_source"].get("versions") == science["version_binding"]["runtime_versions_from_pinned_native_metadata"], "Promoted80 source/version binding changed")


def _load_spec(path, checksum):
    repo, path = Path(__file__).resolve().parents[1], Path(path).resolve()
    _require(type(checksum) is str and len(checksum) == 64 and path.is_file()
             and path.is_relative_to(repo/"results/proposals") and _sha(path) == checksum, "Require frozen prospective execution spec")
    spec, science = json.loads(path.read_text()), _science(repo)
    _spec_controls(spec,science,repo)
    _preserve(spec,path,checksum,science,repo)
    _require((repo/"data/cora/processed/data.pt").is_file(), "Require existing Cora processed data")
    return spec, science


def _request(operation,cells,arm,gate,output,spec,science):
    _require(type(cells) is int and cells == 70 and operation in ("prepare","certify","validate"), "Only fixed Cora70 operations")
    _require((operation != "validate" and arm is None and gate is None) or
             (operation == "validate" and arm in ("node_reference","whole_layer") and type(gate) is str and len(gate) == 64), "Changed phase/arm/gate protocol")
    key = arm if operation == "validate" else operation
    output = Path(output).resolve()
    _require(str(output) == science["operation_outputs"][key] and not output.exists(), "Require absent declared receipt")
    if operation == "prepare":
        _require(not Path(science["candidate_folder"]).exists() and not Path(science["matched_NODE_folder"]).exists(), "Fresh both25 namespaces required; partial caches never resume")
    return output


def _bump(evidence,key,complete=False):
    prefix = "gradient_" if evidence["operation"] == "prepare" else "replay_gradient_"
    inherited._count(evidence,prefix+key,complete)


def _origin(buffers,saved,reference,evidence):
    inherited._count(evidence,"origin_moment")
    with torch.no_grad():
        moments = probe._moments(buffers,buffers["initial"]).detach()
    inherited._count(evidence,"origin_moment",True)
    _require(torch.equal(moments,probe._tensor(saved.get("moments"),(70,1441),torch.float64,"Malformed own original NODE0").to(moments)), "Own current M0 differs from original")
    inputs = representative(moments,buffers["transform"],1433,"cuda")
    weights = torch.full_like(inputs[2],1/70)
    inputs = (inputs[0],inputs[1],weights)
    _require(reference["actual_FP32_P0_X_Q_uniform_equal_reference"] is True and probe._digest(list(inputs)) == reference["student_inputs"], "Own actual P0 readout differs")
    return dict(material=make_material(buffers["z"],buffers["q"]),moments=moments,inputs=inputs,parameters=buffers["initial"])


def _node_config(spec,buffers):
    ghost = source_helper.candidate_controls(dict(spec["matched_NODE_candidate"],method="source_linear",assignment_coordinates="raw_rms",source_linear_schema=1))
    config = source_helper.expected_citation_config(ghost,0,buffers["z"],buffers["q"],buffers["hard"],buffers["z"].detach().float(),buffers["source"])
    for key in ("source_linear_coordinates","source_linear_schema","source_linear_source"):
        config.pop(key)
    config.update(assignment_input="node",save_assignment=False)
    return config


def _node_files(folder):
    paths = [folder/"resume.pt",folder/"optimization.csv",folder/"checkpoints/step_000000.pt",folder/"checkpoints/step_000025.pt"]
    _require(all(p.is_file() for p in paths), "Incomplete matched NODE25 cache preserved")
    return {str(p):_sha(p) for p in paths}


def _node_prepare(spec,science,buffers,origin,evidence,stop):
    folder = Path(science["matched_NODE_folder"])
    _require(not folder.exists(), "Matched NODE namespace must be absent")
    _stop(stop)
    inherited._count(evidence,"matched_NODE_core")
    kwargs = dict(science["matched_NODE_core_call"]["kwargs"])
    _require(kwargs["folder"] == str(folder), "Declared matched NODE folder differs")
    kwargs["folder"] = folder
    kwargs["checkpoint_steps"] = tuple(kwargs["checkpoint_steps"])
    optimize_ce_assignment(buffers["z"],buffers["q"],buffers["hard"],**kwargs,stop=stop)
    inherited._count(evidence,"matched_NODE_core",True)
    resume = torch.load(folder/"resume.pt",map_location="cpu",weights_only=False)
    _require(type(resume.get("step")) is int and resume["step"] == 25, "Matched NODE returned incomplete durable frontier")
    evidence.update(observed_matched_NODE_durable_frontier=25)
    evidence["counts"]["matched_NODE_P_updates_completed"] = 25
    _node_certificate(buffers,origin,_node_config(spec,buffers),folder,evidence)


def _node_certificate(buffers, origin, expected, folder, evidence):
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
        inherited._count(evidence, "certificate_head_gradient")
        gradient = float(head_gradient(augmented(centers), labels, torch.full_like(mass, 1 / 70), theta, .0001).abs().max())
        inherited._count(evidence, "certificate_head_gradient", True)
        inherited._count(evidence, "certificate_teacher_CE")
        value, _ = outer_value_gradient(buffers["z"], buffers["q"], theta, 65536, augmented(buffers["z"]))
        inherited._count(evidence, "certificate_teacher_CE", True)
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
            inherited._count(evidence, "terminal_factor_moments")
            replay = LowRankMoments.apply(u, v, buffers["hard"], origin["material"], .05, 4096).detach()
            inherited._count(evidence, "terminal_factor_moments", True)
            _require(torch.equal(replay, moments) and torch.equal(resume["theta"].to(theta), theta), "Terminal current factors/head differ from snapshot")
        certificates[str(step)] = dict(snapshot=probe._digest(snapshot), head_gradient_max=gradient, teacher_ce=value)
    _require(torch.equal(resume["initial_moments"].to(origin["moments"]), origin["moments"])
             and resume.get("scale") == max(rows[0]["J"], 1e-12), "Own initial objective/normalization differs")
    state = resume.get("optimizer")
    _require(isinstance(state, dict) and set(state.get("state", {})) == {0, 1} and len(state.get("param_groups", [])) == 1,
             "Incomplete final two-slot Adam")
    group = state["param_groups"][0]
    _require(group["params"] == [0, 1] and group["lr"] == .01 and tuple(group["betas"]) == (.9, .999)
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
    evidence["matched_NODE"] = dict(complete25_native_certificate_passed=True, actual_P0_moments_bitwise_equal_core_step0=True,
                    actual_FP32_P0_X_Q_F64_uniform_equal_core_step0=True, endpoint_certificates=certificates,
                    certified_files_sha256={str(p): _sha(p) for p in paths}, history_metadata=metadata,
                    readonly_diagnostics_only=True, certified_baseline_frontier=25)


def _context(spec,science,buffers,reference,evidence):
    return dict(schema=1,implementation=spec["source"],numerical_source=spec["numerical_source"],
        scientific_preregistration=spec["scientific_preregistration"],spec_sha256=evidence["spec_sha256"],spec=spec,
        case=spec["case"],cells=70,assignment_steps=25,student_recipe_origin=buffers["recipe_origin"],
        source_context=buffers["source"],source_buffers=evidence["source_buffer_digest_before"],reference=reference,
        BN_cache=probe.cpu_state(evidence["BN_cache"]),grouping_policy=_POLICY_SPEC,
        matched_NODE_candidate=spec["matched_NODE_candidate"],no_BQ_resume=True)


def _state(buffers,parameters,optimizer,step,scale,anchors,targets,evidence,stop):
    _stop(stop)
    probe._runtime_precision_guard()
    for parameter in parameters:
        parameter.grad = None
    _bump(evidence,"connected_moment")
    moments = probe._moments(buffers,parameters)
    _bump(evidence,"connected_moment",True)
    _bump(evidence,"material_certificate")
    _,_,_,logits,conservation = _material(moments.detach(),buffers,parameters)
    _bump(evidence,"material_certificate",True)
    _bump(evidence,"grouping_partial")
    result = moment_partials(moments,buffers["transform"],anchors,targets)
    _bump(evidence,"grouping_partial",True)
    value = float(result["loss"])
    scale = value if scale is None else scale
    _require(math.isfinite(value) and type(scale) in (int,float) and math.isfinite(scale) and scale > 0, "Finite loss and fresh positive J0 required")
    snapshot = dict(step=step,parameters=parameters,optimizer=optimizer,moments=moments,alignment=result,J0=scale,
                    raw_logits=logits,conservation=conservation,terminal_no_update=step==25)
    if step < 25:
        _bump(evidence,"original_moment_backward")
        moments.backward(result["moment_gradient"]/scale)
        _bump(evidence,"original_moment_backward",True)
        _require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in parameters), "Missing/nonfinite factor gradients")
        snapshot["scaled_factor_gradients"] = [p.grad for p in parameters]
    bq.bn._finite_tree(snapshot)
    return probe.cpu_state(snapshot),scale


def _history(states):
    return [dict(step=k,J=float(s["alignment"]["loss"]),J0=s["J0"],conservation=s["conservation"],terminal_no_update=s["terminal_no_update"]) for k,s in sorted(states.items())]


def _store(folder,context,states,scale,target_digest):
    bundle = probe._attach(dict(schema=1,context=context,frontier=max(states),states=states,J0=scale,
                                history=_history(states),target_digest=target_digest))
    probe._atomic(folder/"progress.pt",bundle,True)
    probe._atomic(folder/"history.json",json.dumps(bundle["history"],indent=2),False)
    if max(states) in (0,25):
        probe._atomic(folder/f"step_{max(states):06d}.pt",probe._attach(dict(context=context,state=states[max(states)])),True)
    return bundle


def _gradient_files(folder):
    names = {"candidate.json","progress.pt","history.json","step_000000.pt","step_000025.pt"}
    _require(folder.is_dir() and {p.name for p in folder.iterdir()} == names and all((folder/n).is_file() for n in names), "Require complete exact whole-layer25 files; no partial fallback")
    return {str(folder/n):_sha(folder/n) for n in sorted(names)}


def _gradient_prepare(spec,science,buffers,origin,context,anchors,targets,evidence,stop):
    folder = Path(science["candidate_folder"])
    _require(not folder.exists(), "Whole-layer25 namespace must be absent")
    folder.mkdir(parents=True,exist_ok=False)
    _write_new(folder/"candidate.json",spec["candidate"])
    parameters = [p.detach().clone().requires_grad_() for p in buffers["initial"]]
    _bump(evidence,"optimizer_constructor")
    optimizer = torch.optim.Adam(parameters,lr=.01,betas=(.9,.999),eps=1e-12,weight_decay=0,foreach=False,fused=False)
    _bump(evidence,"optimizer_constructor",True)
    first,second = [torch.zeros_like(p) for p in parameters],[torch.zeros_like(p) for p in parameters]
    states,scale = {},None
    digest = probe._digest(targets)
    for step in range(26):
        inherited._optimizer(optimizer.state_dict(),parameters,first,second,step)
        state,scale = _state(buffers,parameters,optimizer.state_dict(),step,scale,anchors,targets,evidence,stop)
        if step == 0:
            _require(torch.equal(state["moments"].to(origin["moments"]),origin["moments"]), "Fresh whole-layer M0 differs from paired NODE")
        states[step] = state
        _store(folder,context,states,scale,digest)
        if step < 25:
            expected,first,second = inherited._adam_step(parameters,first,second,[p.grad for p in parameters],step+1)
            _stop(stop)
            probe._runtime_precision_guard()
            _bump(evidence,"P_update")
            optimizer.step()
            _bump(evidence,"P_update",True)
            _require(all(torch.equal(p.detach(),e) for p,e in zip(parameters,expected,strict=True)), "Actual whole-layer Adam parameters differ")
            inherited._optimizer(optimizer.state_dict(),parameters,first,second,step+1)
    _require(probe._digest(targets) == digest, "Frozen targets changed")
    evidence.update(gradient_native_Adam_recurrence_passed=True,gradient_fresh25_complete=True,
                    gradient_cache_files_sha256=_gradient_files(folder),gradient_prefix_sha256=_seal(states),
                    alignment_J0=scale,alignment_J25=float(states[25]["alignment"]["loss"]))
    return states


def _load_progress(science,buffers,origin,context,anchors,targets,evidence,stop):
    folder = Path(science["candidate_folder"])
    pins = _gradient_files(folder)
    _require(_exact(json.loads((folder/"candidate.json").read_text()),context["spec"]["candidate"]), "Cached candidate changed")
    bundle = torch.load(folder/"progress.pt",map_location="cpu",weights_only=False)
    bq.bn.bi._sealed_payload(bundle)
    _require(set(bundle) == {"schema","context","frontier","states","J0","history","target_digest","content_sha256"}
             and type(bundle["schema"]) is int and bundle["schema"] == 1 and _exact(bundle["context"],context)
             and type(bundle["frontier"]) is int and bundle["frontier"] == 25
             and isinstance(bundle["states"],dict) and all(type(k) is int for k in bundle["states"])
             and set(bundle["states"]) == set(range(26)) and bundle["target_digest"] == probe._digest(targets), "Changed full26 coupled cache/context")
    expected = [p.detach().clone() for p in buffers["initial"]]
    first,second = [torch.zeros_like(p) for p in expected],[torch.zeros_like(p) for p in expected]
    scale = None
    for step,saved in sorted(bundle["states"].items()):
        _require(type(saved.get("step")) is int and saved["step"] == step and isinstance(saved.get("parameters"),list) and len(saved["parameters"]) == 2, "Malformed coupled state")
        _require(all(torch.equal(probe._tensor(a,p.shape,torch.float32,"Cached factor").to(p),p) for a,p in zip(saved["parameters"],expected,strict=True)), "Cached initial/Adam transition differs")
        inherited._optimizer(saved["optimizer"],expected,first,second,step)
        actual,scale = _state(buffers,[p.clone().requires_grad_() for p in expected],saved["optimizer"],step,scale,anchors,targets,evidence,stop)
        _require(_seal(actual) == _seal(saved), "Coupled M/J0/groupingpartial/factorgrad/Adam state differs")
        if step < 25:
            _bump(evidence,"Adam_recursion")
            expected,first,second = inherited._adam_step(expected,first,second,[g.to(p) for g,p in zip(actual["scaled_factor_gradients"],expected,strict=True)],step+1)
            _bump(evidence,"Adam_recursion",True)
    _require(bundle["J0"] == scale and _exact(bundle["history"],_history(bundle["states"]))
             and _exact(json.loads((folder/"history.json").read_text()),bundle["history"])
             and torch.equal(bundle["states"][0]["moments"].to(origin["moments"]),origin["moments"]), "Own origin/J0/history differs")
    for step in (0,25):
        checkpoint = torch.load(folder/f"step_{step:06d}.pt",map_location="cpu",weights_only=False)
        _require(_seal(checkpoint) == _seal(probe._attach(dict(context=context,state=bundle["states"][step]))), "Endpoint/progress mirror differs")
    _require(_gradient_files(folder) == pins, "Readonly whole-layer cache changed")
    evidence.update(full25_cache_replay_passed=True,gradient_native_Adam_recurrence_passed=True,
        gradient_cache_files_sha256=pins,gradient_prefix_sha256=_seal(bundle["states"]),alignment_J0=scale,
        alignment_J25=float(bundle["states"][25]["alignment"]["loss"]))
    return bundle["states"]


def _gate(spec,science,checksum):
    path = Path(spec["certificate_outputs"]["70"])
    _require(type(checksum) is str and len(checksum) == 64 and _sha(path) == checksum, "Accepted joint gate SHA differs")
    gate = json.loads(path.read_text())
    _require(gate.get("passed") is True and gate.get("operation") == "certify" and gate.get("cells") == 70
             and gate.get("full25_cache_replay_passed") is True and gate.get("gradient_native_Adam_recurrence_passed") is True
             and gate.get("source_assets_spec_science_unchanged") is True and gate.get("test_enabled") is False
             and _exact(gate.get("source"),spec["source"]) and _exact(gate.get("numerical_source"),spec["numerical_source"])
             and _exact(gate.get("scientific_preregistration"),spec["scientific_preregistration"])
             and all(type(gate.get(k)) is str and len(gate[k]) == 64 for k in ("spec_sha256","gradient_prefix_sha256","context_digest")),
             "Joint native whole-layer gate is unqualified")
    node = gate.get("matched_NODE",{})
    _require(node.get("complete25_native_certificate_passed") is True and node.get("actual_P0_moments_bitwise_equal_core_step0") is True
             and node.get("actual_FP32_P0_X_Q_F64_uniform_equal_core_step0") is True, "Matched NODE native gate is unqualified")
    _require(_gradient_files(Path(science["candidate_folder"])) == gate["gradient_cache_files_sha256"]
             and _node_files(Path(science["matched_NODE_folder"])) == node["certified_files_sha256"], "Native cache byte pins changed")
    return gate


def _validate(spec,science,buffers,origin,states,context,arm,gate_sha,evidence,stop):
    gate = _gate(spec,science,gate_sha)
    _require(gate["gradient_prefix_sha256"] == _seal(states) and gate["context_digest"] == _seal(context)
             and gate["spec_sha256"] == evidence["spec_sha256"], "Gate coupled prefix/context/spec differs")
    if arm == "node_reference":
        saved = torch.load(Path(science["matched_NODE_folder"])/"checkpoints/step_000025.pt",map_location="cpu",weights_only=False)
        endpoint = saved["moments"].to(origin["moments"])
    else:
        endpoint = states[25]["moments"].to(origin["moments"])
    endpoint_inputs = representative(endpoint,buffers["transform"],1433,"cuda")
    endpoint_inputs = (endpoint_inputs[0],endpoint_inputs[1],torch.full_like(endpoint_inputs[2],1/70))
    base = Path(science["candidate_folder"]).parent
    rows = []
    for step,inputs in ((0,origin["inputs"]),(25,endpoint_inputs)):
        folder = base/("shared_P0_validation" if step == 0 else ("node_reference25_validation" if arm == "node_reference" else "whole_layer25_validation"))
        if arm == "whole_layer" and step == 0:
            _require(folder.is_dir() and all((folder/f"seed_{seed}_completion_v1.json").is_file() for seed in SEEDS), "NODE P0 complete physical cohort must precede whole-layer reuse")
        for seed in SEEDS:
            row = evaluate_student(folder,inputs,buffers,context,gate_sha,seed,stop)
            if arm == "whole_layer" and step == 0:
                _require(row["cached"] is True and row["actual_student_fits"] == 0, "Shared P0 must be exact hot reuse")
            rows.append(dict(row,arm=arm,step=step,cells=70,candidate_id=spec["candidate_id"]))
    evidence.update(validation_rows=rows,arm=arm,gate_sha256=gate_sha,
        actual_student_fits=sum(r["actual_student_fits"] for r in rows),physical_route_outputs=sum(r["physical_route_outputs"] for r in rows),
        logical_student_conditions=6,logical_serving_conditions=12,actual_student_epochs=600*sum(r["actual_student_fits"] for r in rows))


def run(operation,cells,spec_path,spec_sha256,output_path,arm=None,gate_sha256=None,stop=lambda:False):
    _require(callable(stop), "Require stop callback")
    repo,spec_path = Path(__file__).resolve().parents[1],Path(spec_path).resolve()
    spec,science = _load_spec(spec_path,spec_sha256)
    output = _request(operation,cells,arm,gate_sha256,output_path,spec,science)
    if operation == "validate":
        _gate(spec,science,gate_sha256)
    started = time.monotonic()
    bounded = lambda:stop() or time.monotonic()-started >= 300
    evidence = dict(passed=False,operation=operation,cells=70,candidate=spec["candidate"],candidate_id=spec["candidate_id"],
        source=spec["source"],numerical_source=spec["numerical_source"],python_version=spec["python_version"],
        scientific_preregistration=spec["scientific_preregistration"],spec_path=str(spec_path),spec_sha256=spec_sha256,
        grouping_policy=_POLICY_SPEC,test_enabled=False,validation_only=True,stage="fresh_native_policy",
        counts={k:0 for k in bq.bc.original._science(repo)["expected_success_counts"] if k != "P_updates"},operation_attempts={},
        count_scope="Source-certificate diagnostics separate; gradient/replay interfaces attempted/completed. Matched core is one interface; durable25 proof determines observed updates. Unobserved head/adjoint internals unknown.")
    evidence["counts"].update(final_student_fit_attempted=0,final_student_fit_completed=0)
    native,primary,buffers,payload,anchors,targets = False,None,None,None,None,None
    cache_pins = None
    try:
        _stop(bounded)
        _require(torch.get_num_threads() == 4 and not torch.cuda.is_initialized(), "Fresh threads4 CUDA worker required")
        environment = probe._native("cuda")
        native = True
        torch.cuda.reset_peak_memory_stats()
        _require(torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory == bq.bc.original.CUDA_CAPACITY, "GPU capacity differs")
        evidence["native_environment"] = dict(environment,python_version=platform.python_version(),threads=4,target_backend=science["BN_source"]["source_backend"],inherited_graph_label_scope="precision and source-certificate only; source gradients are accepted BN segment cache")
        evidence["stage"] = "own_original_source"
        buffers,saved,reference = _load_source(spec["case"],spec,science,evidence,bounded)
        buffers.update(case=spec["case"],recipe_origin=evidence["original_recipe_origin"],counts=evidence["counts"])
        evidence.update(reference=reference,source_buffer_digest_before=bq._source_buffer_digest(buffers),anchor0_digest_before=probe._digest(buffers["model_initial"]))
        origin = _origin(buffers,saved,reference,evidence)
        evidence["stage"] = "accepted_BN_cache"
        payload,anchors,targets = _cached_targets(buffers,spec,science,evidence,bounded)
        context = _context(spec,science,buffers,reference,evidence)
        if operation == "prepare":
            evidence["stage"] = "fresh_matched_NODE25"
            _node_prepare(spec,science,buffers,origin,evidence,bounded)
            evidence["stage"] = "fresh_whole_layer25"
            states = _gradient_prepare(spec,science,buffers,origin,context,anchors,targets,evidence,bounded)
        else:
            cache_pins = dict(_gradient_files(Path(science["candidate_folder"])),**_node_files(Path(science["matched_NODE_folder"])))
            evidence["stage"] = "readonly_matched_NODE25"
            _node_certificate(buffers,origin,_node_config(spec,buffers),Path(science["matched_NODE_folder"]),evidence)
            evidence["stage"] = "readonly_whole_layer26states"
            states = _load_progress(science,buffers,origin,context,anchors,targets,evidence,bounded)
            if operation == "validate":
                evidence["stage"] = "paired_student_validation"
                _validate(spec,science,buffers,origin,states,context,arm,gate_sha256,evidence,bounded)
        _retained(buffers,payload,anchors,targets,evidence)
        evidence.update(context_digest=_seal(context),native_P0_moments_exactly_equal_cached_NODE0=True,
            actual_FP32_P0_X_Q_uniform_equal_reference=True,passed=True)
    except BaseException as error:
        primary = error
        evidence.update(error_type=type(error).__name__,error=str(error),failed_stage=evidence["stage"])
    finally:
        try:
            _retained(buffers,payload,anchors,targets,evidence)
            if cache_pins is not None:
                probe._checked_files(cache_pins)
                evidence["readonly_native_cache_unchanged"] = True
        except BaseException as error:
            evidence.update(passed=False,retained_buffer_preservation_error=repr(error))
            if primary is None: primary = error
        try:
            _preserve(spec,spec_path,spec_sha256,science,repo)
            evidence.update(source_unchanged=True,source_assets_spec_science_unchanged=True)
        except BaseException as error:
            evidence.update(passed=False,source_unchanged=False,source_assets_spec_science_unchanged=False,preservation_error=repr(error))
            if primary is None: primary = error
        if native:
            try:
                torch.cuda.synchronize()
                capacity = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
                allocated,reserved = torch.cuda.max_memory_allocated(),torch.cuda.max_memory_reserved()
                evidence.update(CUDA_total_bytes=capacity,CUDA_peak_allocated_bytes=allocated,CUDA_peak_reserved_bytes=reserved)
                _require(capacity == bq.bc.original.CUDA_CAPACITY and 0 <= allocated <= capacity and 0 <= reserved <= capacity and torch.get_num_threads() == 4, "GPU capacity/memory/threads differ")
                probe._runtime_precision_guard()
            except BaseException as error:
                evidence.update(passed=False,native_finalization_error=repr(error))
                if primary is None: primary = error
        evidence["seconds"] = time.monotonic()-started
        if evidence["seconds"] > 300:
            error = ValueError("BR worker exceeded300seconds")
            evidence.update(passed=False,deadline_error=str(error))
            if primary is None: primary = error
        try:
            _write_new(output,evidence)
        except BaseException:
            if primary is not None: raise primary
            raise
    if primary is not None: raise primary
    return dict(evidence,evidence_path=str(output),evidence_sha256=_sha(output),validation_only=True)
