"""Fresh hybrid25 with readonly exact BS matched-.01 NODE25 and one paired cohort.

No disposable-probe resume, partial-cache repair or score-dependent operation.
Historical BN source context remains separate from the current frozen source.
"""
import json
import platform
import time
from importlib.metadata import version as distribution_version
from pathlib import Path

import torch
from src.cora_hybrid_students import evaluate_student

from src import citeseer_finite_student_v2 as inherited
from src import cora_hybrid_probe as bu
from src import cora_whole_layer_fixed25 as bs
from src import cora_whole_layer_probe as bq
from src import finite_student_probe as probe
from src.citation_macro_probe import _material
from src.citation_source_preflight import _exact, _write_new
from src.citeseer_confirmation_source import _reference_pins
from src.io import _fingerprint
from src.moments import augmented
from src.research_loop import implementation_provenance
from src.soft_ce_partition import solve_inner_newton_first
from src.sweep_utils import representative
from src.whole_layer_gradient_alignment import POLICY

SCIENCE = "Cora70_half_teacher_CE_whole_layer_fixed25_scientific_stageBV_v1.json"
SCIENCE_SHA = "83f6a4789eb80f3ebb1b8d156e49ea9d499f90bf8768fbe584818440c710db0e"
SCIENCE_REVIEW = "independent_scientific_review_stageBV_v1.json"
SCIENCE_REVIEW_SHA = "44e4439e7a6d022fdf256afa66b5d45581816d3b485169d6d432d557b693ef36"
_FIELDS = {"schema", "fixed", "case", "candidate", "candidate_id", "matched_NODE_candidate", "source", "numerical_source",
           "python_version", "files_sha256", "scientific_preregistration", "artifacts_sha256", "output_root",
           "certificate_outputs", "grouping_policy", "student_seeds"}
_POLICY_SPEC = json.loads(json.dumps(POLICY))
HORIZON, SEEDS = 25, (4500, 4501, 4502)
_require, _sha, _seal, _stop = probe._require, probe._sha, probe._seal, probe._stop
_load_source, _cached_targets, _retained = bq._load_source, bq._cached_targets, bq._retained
_origin, _node_config, _node_files, _node_certificate = bs._origin, bs._node_config, bs._node_files, bs._node_certificate


def numerical_source():
    value = bu.numerical_source()
    for name in ("cora_hybrid_fixed25.py", "cora_hybrid_students.py"):
        _require(value["files"].get(name) == _sha(Path(__file__).with_name(name)), "Owned BV source changed")
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
    _require(isinstance(spec,dict) and set(spec) == _FIELDS and type(spec["schema"]) is int and spec["schema"] == 1, "Unknown hybrid25 spec")
    for key in ("fixed","case","candidate","candidate_id","matched_NODE_candidate","output_root","certificate_outputs","student_seeds"):
        _require(_exact(spec[key],science[key]), "Changed fixed BV control: " + key)
    _require(_fingerprint(spec["candidate"]) == spec["candidate_id"] and _exact(spec["grouping_policy"],_POLICY_SPEC)
             and _exact(spec["files_sha256"],science["original_files_sha256"])
             and spec["scientific_preregistration"] == dict(path=str(repo/"results/proposals"/SCIENCE),sha256=SCIENCE_SHA), "Candidate/policy/assets/science changed")
    owned = {(repo/p).resolve() for p in science["source_protection"]["owned"]}
    pins = spec["artifacts_sha256"]
    _require(isinstance(pins,dict) and pins and all(type(p) is str and Path(p).is_absolute()
             and (Path(p).resolve().is_relative_to(repo/"results") or Path(p).resolve() in owned) for p in pins), "Require exact owned or results artifact pins")
    _require(set(spec["source"].get("files",{})) == set(science["source_before"]["files"]) | {p for p in science["source_protection"]["owned"] if p.startswith("src/")}
             and spec["python_version"] == science["version_binding"]["python_version"]
             and spec["numerical_source"].get("versions") == science["version_binding"]["runtime_versions_from_pinned_native_metadata"], "Promoted84 source/version binding changed")


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
             (operation == "validate" and arm in ("node_reference","hybrid") and type(gate) is str and len(gate) == 64), "Changed phase/arm/gate protocol")
    key = arm if operation == "validate" else operation
    output = Path(output).resolve()
    _require(str(output) == science["operation_outputs"][key] and not output.exists(), "Require absent declared receipt")
    if operation == "prepare":
        _require(not Path(science["candidate_folder"]).exists(), "Fresh hybrid25 namespace required; no disposable resume")
    return output


def _bump(evidence,key,complete=False):
    prefix = "hybrid_" if evidence["operation"] == "prepare" else "replay_hybrid_"
    inherited._count(evidence,prefix+key,complete)


def _context(spec,science,buffers,reference,evidence):
    return dict(schema=1,implementation=spec["source"],numerical_source=spec["numerical_source"],
        scientific_preregistration=spec["scientific_preregistration"],spec_sha256=evidence["spec_sha256"],spec=spec,
        case=spec["case"],cells=70,assignment_steps=25,student_recipe_origin=buffers["recipe_origin"],
        source_context=buffers["source"],source_buffers=evidence["source_buffer_digest_before"],reference=reference,
        BN_cache=probe.cpu_state(evidence["BN_cache"]),grouping_policy=_POLICY_SPEC,
        matched_NODE_candidate=spec["matched_NODE_candidate"],no_BU_resume=True)


def _hybrid_files(folder):
    names = {"candidate.json","progress.pt","history.json","step_000000.pt","step_000025.pt"}
    _require(folder.is_dir() and {p.name for p in folder.iterdir()} == names and all((folder/n).is_file() for n in names), "Require complete exact hybrid25 files; no partial fallback")
    return {str(folder/n):_sha(folder/n) for n in sorted(names)}


def _joint_call(moments,buffers,theta,anchors,targets,scales,features,evidence,stop):
    """Call immutable BU algebra; qualify/count diagnostics under own phase."""
    local = {"counts":{}}
    inherited._count(local,"joint_partial")
    try:
        return bu._joint(moments,buffers,theta,anchors,targets,scales,features,local,stop)
    finally:
        prefix = "hybrid_" if evidence["operation"] == "prepare" else "replay_hybrid_"
        for key,value in local["counts"].items():
            evidence["counts"][prefix+key] = evidence["counts"].get(prefix+key,0)+value


def _state(buffers,parameters,optimizer,step,theta,scales,features,anchors,targets,evidence,stop,head_work=None,fit_head=False):
    _stop(stop)
    probe._runtime_precision_guard()
    for parameter in parameters:
        parameter.grad = None
    _bump(evidence,"connected_moment")
    moments = probe._moments(buffers,parameters)
    _bump(evidence,"connected_moment",True)
    _bump(evidence,"material_certificate")
    centers,labels,weights,logits,conservation = _material(moments.detach(),buffers,parameters)
    _bump(evidence,"material_certificate",True)
    _require(type(step) is int and 0 <= step <= 25 and type(fit_head) is bool
             and (not fit_head or evidence["operation"] == "prepare" and step > 0), "No readonly/P0 head refit allowed")
    if fit_head:
        _bump(evidence,"endpoint_head_solve")
        fitted = solve_inner_newton_first(centers,labels,weights,.0001,initial=theta,
            max_iter=2000,grad_tol=1e-7,cg_max_iter=512,newton_steps=8,cg_check_interval=1)
        _bump(evidence,"endpoint_head_solve",True)
        _require(fitted.get("inner_converged") is True,"Updated original uniform head did not converge")
        theta = fitted["theta"].detach()
        head_work = {k:v for k,v in fitted.items() if k != "theta"}
    theta = probe._tensor(theta,(7,1434),torch.float64,"Malformed saved stationary hybrid head").to(moments).detach()
    result = _joint_call(moments,buffers,theta,anchors,targets,scales,features,evidence,stop)
    if scales is None:
        scales = dict(result["normalizers"])
        _require(abs(result["objective"]-1.) <= 1e-12,"Own fresh normalized J0 differs")
    _require(result["normalizers"] == scales,"Own fixed F0/G0 changed")
    snapshot = dict(step=step,parameters=parameters,optimizer=optimizer,moments=moments,theta=theta,joint=result,
        normalizers=dict(scales),head_work=head_work,raw_logits=logits,conservation=conservation,terminal_no_update=step==25)
    if step < 25:
        _bump(evidence,"original_moment_backward")
        moments.backward(result["moment_gradient"])
        _bump(evidence,"original_moment_backward",True)
        _require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in parameters),"Missing/nonfinite hybrid factor gradient")
        snapshot["scaled_factor_gradients"] = [p.grad for p in parameters]
    bq.bn._finite_tree(snapshot)
    return probe.cpu_state(snapshot),dict(scales)


def _history(states):
    return [dict(step=k,F=s["joint"]["teacher_CE"],G=s["joint"]["alignment"],J=s["joint"]["objective"],
        normalizers=s["normalizers"],head_gradient=s["joint"]["theta_gradient_max"],adjoint=s["joint"]["adjoint"],
        explicit_adjoint_residual=s["joint"]["explicit_adjoint_residual"],rhs_norm=s["joint"]["rhs_norm"],
        conservation=s["conservation"],head_work=s["head_work"],terminal_no_update=s["terminal_no_update"]) for k,s in sorted(states.items())]


def _store(folder,context,states,scales,target_digest):
    bundle = probe._attach(dict(schema=1,context=context,frontier=max(states),states=states,normalizers=dict(scales),
        history=_history(states),target_digest=target_digest))
    probe._atomic(folder/"progress.pt",bundle,True)
    probe._atomic(folder/"history.json",json.dumps(bundle["history"],indent=2),False)
    if max(states) in (0,25):
        probe._atomic(folder/f"step_{max(states):06d}.pt",probe._attach(dict(context=context,state=states[max(states)])),True)
    return bundle


def _hybrid_prepare(spec,science,buffers,saved,origin,context,anchors,targets,evidence,stop):
    folder = Path(science["candidate_folder"])
    _require(not folder.exists(),"Hybrid25 namespace must be absent")
    folder.mkdir(parents=True,exist_ok=False)
    _write_new(folder/"candidate.json",spec["candidate"])
    parameters = [p.detach().clone().requires_grad_() for p in buffers["initial"]]
    _bump(evidence,"optimizer_constructor")
    optimizer = torch.optim.Adam(parameters,lr=.01,betas=(.9,.999),eps=1e-12,weight_decay=0,foreach=False,fused=False)
    _bump(evidence,"optimizer_constructor",True)
    first,second = [torch.zeros_like(p) for p in parameters],[torch.zeros_like(p) for p in parameters]
    _bump(evidence,"source_augmented_features")
    features = augmented(buffers["z"])
    _bump(evidence,"source_augmented_features",True)
    theta = probe._tensor(saved.get("theta"),(7,1434),torch.float64,"Malformed original NODE0theta").to(buffers["z"]).detach()
    states,scales,digest = {},None,probe._digest(targets)
    for step in range(26):
        inherited._optimizer(optimizer.state_dict(),parameters,first,second,step)
        state,scales = _state(buffers,parameters,optimizer.state_dict(),step,theta,scales,features,anchors,targets,evidence,stop,
            head_work=None if step else {"cached_original_NODE0":True},fit_head=step>0)
        theta = state["theta"].to(buffers["z"])
        if step == 0:
            _require(torch.equal(state["moments"].to(origin["moments"]),origin["moments"])
                     and torch.equal(theta,saved["theta"].to(theta)),"Fresh hybrid M0/theta0 differs from own original")
        states[step] = state
        _store(folder,context,states,scales,digest)
        evidence["observed_hybrid_durable_frontier"] = step
        if step < 25:
            _bump(evidence,"Adam_recursion")
            expected,first,second = inherited._adam_step(parameters,first,second,[p.grad for p in parameters],step+1)
            _bump(evidence,"Adam_recursion",True)
            _stop(stop)
            probe._runtime_precision_guard()
            _bump(evidence,"P_update")
            optimizer.step()
            _bump(evidence,"P_update",True)
            _require(all(torch.equal(p.detach(),e) for p,e in zip(parameters,expected,strict=True)),"Actual hybrid Adam recurrence differs")
            inherited._optimizer(optimizer.state_dict(),parameters,first,second,step+1)
    _require(probe._digest(targets)==digest,"Frozen targets changed")
    evidence.update(hybrid_native_Adam_recurrence_passed=True,hybrid_fresh25_complete=True,
        hybrid_cache_files_sha256=_hybrid_files(folder),hybrid_prefix_sha256=_seal(states),normalizers=dict(scales),
        hybrid_J0=states[0]["joint"]["objective"],hybrid_J25=states[25]["joint"]["objective"],history=_history(states))
    return states


def _load_progress(science,buffers,saved0,origin,context,anchors,targets,evidence,stop):
    folder = Path(science["candidate_folder"])
    pins = _hybrid_files(folder)
    _require(_exact(json.loads((folder/"candidate.json").read_text()),context["spec"]["candidate"]),"Cached hybrid candidate changed")
    bundle = torch.load(folder/"progress.pt",map_location="cpu",weights_only=False)
    bq.bn.bi._sealed_payload(bundle)
    _require(set(bundle)=={"schema","context","frontier","states","normalizers","history","target_digest","content_sha256"}
        and type(bundle["schema"]) is int and bundle["schema"]==1 and _exact(bundle["context"],context)
        and type(bundle["frontier"]) is int and bundle["frontier"]==25 and isinstance(bundle["states"],dict)
        and all(type(k) is int for k in bundle["states"]) and set(bundle["states"])==set(range(26))
        and bundle["target_digest"]==probe._digest(targets),"Changed full26hybrid cache/context")
    expected = [p.detach().clone() for p in buffers["initial"]]
    first,second = [torch.zeros_like(p) for p in expected],[torch.zeros_like(p) for p in expected]
    _bump(evidence,"source_augmented_features")
    features = augmented(buffers["z"])
    _bump(evidence,"source_augmented_features",True)
    scales = None
    for step,saved in sorted(bundle["states"].items()):
        _require(type(saved.get("step")) is int and saved["step"]==step and isinstance(saved.get("parameters"),list)
            and len(saved["parameters"])==2,"Malformed coupled hybrid state")
        _require(all(torch.equal(probe._tensor(a,p.shape,torch.float32,"Cached factor").to(p),p)
            for a,p in zip(saved["parameters"],expected,strict=True)),"Cached hybrid initial/Adamtransition differs")
        inherited._optimizer(saved["optimizer"],expected,first,second,step)
        theta = probe._tensor(saved.get("theta"),(7,1434),torch.float64,"Malformed cached stationary head").to(buffers["z"])
        if step == 0:
            _require(torch.equal(theta,saved0["theta"].to(theta)),"Cached hybridtheta0 differs from originalNODE0")
        actual,scales = _state(buffers,[p.clone().requires_grad_() for p in expected],saved["optimizer"],step,theta,scales,
            features,anchors,targets,evidence,stop,head_work=saved.get("head_work"),fit_head=False)
        _require(_seal(actual)==_seal(saved),"Coupled hybridM/theta/F0G0/jointpartial/Pgrad/Adamstate differs")
        if step < 25:
            _bump(evidence,"Adam_recursion")
            expected,first,second = inherited._adam_step(expected,first,second,
                [g.to(p) for g,p in zip(actual["scaled_factor_gradients"],expected,strict=True)],step+1)
            _bump(evidence,"Adam_recursion",True)
    _require(bundle["normalizers"]==scales and _exact(bundle["history"],_history(bundle["states"]))
        and _exact(json.loads((folder/"history.json").read_text()),bundle["history"])
        and torch.equal(bundle["states"][0]["moments"].to(origin["moments"]),origin["moments"]),"Own original/scales/history differs")
    for step in (0,25):
        checkpoint = torch.load(folder/f"step_{step:06d}.pt",map_location="cpu",weights_only=False)
        _require(_seal(checkpoint)==_seal(probe._attach(dict(context=context,state=bundle["states"][step]))),"Endpoint/progress mirror differs")
    _require(_hybrid_files(folder)==pins,"Readonly hybrid cache changed")
    evidence.update(full25_cache_replay_passed=True,hybrid_native_Adam_recurrence_passed=True,
        hybrid_cache_files_sha256=pins,hybrid_prefix_sha256=_seal(bundle["states"]),normalizers=dict(scales),
        hybrid_J0=bundle["states"][0]["joint"]["objective"],hybrid_J25=bundle["states"][25]["joint"]["objective"],history=bundle["history"])
    return bundle["states"]


def _success_counts(evidence,science):
    raw = evidence["counts"]
    preparing = evidence["operation"] == "prepare"
    prefix = "hybrid_" if preparing else "replay_hybrid_"
    result = dict(P_updates=raw.get("hybrid_P_update_completed",0),baseline_P_updates=0,
        native_Adam_steps=raw.get("hybrid_P_update_completed",0),full_joint_partials=raw.get(prefix+"joint_partial_completed",0),
        CE_adjoints=raw.get(prefix+"adjoint_solve_completed",0),grouping_partials=raw.get(prefix+"grouping_partial_completed",0),
        synthetic_CE_parameter_gradients=3*raw.get(prefix+"grouping_partial_completed",0),
        outside_original_P_backwards=raw.get(prefix+"original_moment_backward_completed",0),
        updated_head_solve_calls=raw.get("hybrid_endpoint_head_solve_completed",0),
        manual_Adam_recursions=raw.get(prefix+"Adam_recursion_completed",0),
        accepted_BN_cache_loads=raw.get("accepted_BN_cache_load_completed",0),
        student_fits=raw.get("final_student_fit_completed",0),new_source_gradient_targets=0)
    expected = science["counts_contract"][evidence["operation"] if evidence["operation"]!="validate" else evidence["arm"]]
    _require(result==expected,"Actual fullhybrid phase counts differ")
    return result


def _gate(spec,science,checksum):
    path = Path(spec["certificate_outputs"]["70"])
    _require(type(checksum) is str and len(checksum) == 64 and _sha(path) == checksum, "Accepted joint gate SHA differs")
    gate = json.loads(path.read_text())
    _require(gate.get("passed") is True and gate.get("operation") == "certify" and gate.get("cells") == 70
             and gate.get("full25_cache_replay_passed") is True and gate.get("hybrid_native_Adam_recurrence_passed") is True
             and gate.get("source_assets_spec_science_unchanged") is True and gate.get("test_enabled") is False
             and _exact(gate.get("source"),spec["source"]) and _exact(gate.get("numerical_source"),spec["numerical_source"])
             and _exact(gate.get("scientific_preregistration"),spec["scientific_preregistration"])
             and all(type(gate.get(k)) is str and len(gate[k]) == 64 for k in ("spec_sha256","hybrid_prefix_sha256","context_digest")),
             "Joint native hybrid gate is unqualified")
    node = gate.get("matched_NODE",{})
    _require(node.get("complete25_native_certificate_passed") is True and node.get("actual_P0_moments_bitwise_equal_core_step0") is True
             and node.get("actual_FP32_P0_X_Q_F64_uniform_equal_core_step0") is True, "Matched NODE native gate is unqualified")
    _require(_hybrid_files(Path(science["candidate_folder"])) == gate["hybrid_cache_files_sha256"]
             and _node_files(Path(science["matched_NODE_folder"])) == node["certified_files_sha256"], "Native cache byte pins changed")
    return gate


def _validate(spec,science,buffers,origin,states,context,arm,gate_sha,evidence,stop):
    gate = _gate(spec,science,gate_sha)
    _require(gate["hybrid_prefix_sha256"] == _seal(states) and gate["context_digest"] == _seal(context)
             and gate["spec_sha256"] == evidence["spec_sha256"], "Gate coupled prefix/context/spec differs")
    if arm == "node_reference":
        saved = torch.load(Path(science["matched_NODE_folder"])/"checkpoints/step_000025.pt",map_location="cpu",weights_only=False)
        endpoint = saved["moments"].to(origin["moments"])
    else:
        endpoint = states[25]["moments"].to(origin["moments"])
    endpoint_inputs = representative(endpoint,probe._transform(buffers),1433,"cuda")
    endpoint_inputs = (endpoint_inputs[0],endpoint_inputs[1],torch.full_like(endpoint_inputs[2],1/70))
    base = Path(science["candidate_folder"]).parent
    rows = []
    for step,inputs in ((0,origin["inputs"]),(25,endpoint_inputs)):
        folder = base/("shared_P0_validation" if step == 0 else ("node_reference25_validation" if arm == "node_reference" else "hybrid25_validation"))
        if arm == "hybrid" and step == 0:
            _require(folder.is_dir() and all((folder/f"seed_{seed}_completion_v1.json").is_file() for seed in SEEDS), "NODE P0 complete physical cohort must precede hybrid reuse")
        for seed in SEEDS:
            row = evaluate_student(folder,inputs,buffers,context,gate_sha,seed,stop)
            if arm == "hybrid" and step == 0:
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
    evidence = dict(passed=False,success_counts=None,operation=operation,cells=70,candidate=spec["candidate"],candidate_id=spec["candidate_id"],
        source=spec["source"],numerical_source=spec["numerical_source"],python_version=spec["python_version"],
        scientific_preregistration=spec["scientific_preregistration"],spec_path=str(spec_path),spec_sha256=spec_sha256,
        grouping_policy=_POLICY_SPEC,test_enabled=False,validation_only=True,stage="fresh_native_policy",
        counts={k:0 for k in bq.bc.original._science(repo)["expected_success_counts"] if k != "P_updates"},operation_attempts={},
        count_scope="Source and readonly matched-NODE diagnostics separate; hybrid/replay interfaces attempted/completed. Only fresh hybrid preparation performs P updates. Historical saved head_work is sealed provenance; readonly phases perform no head refits. Unobserved solver internals unknown.")
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
        cache_pins = _node_files(Path(science["matched_NODE_folder"]))
        evidence["stage"] = "accepted_BN_cache"
        payload,anchors,targets = _cached_targets(buffers,spec,science,evidence,bounded)
        context = _context(spec,science,buffers,reference,evidence)
        if operation == "prepare":
            evidence["stage"] = "readonly_matched_NODE25"
            _node_certificate(buffers,origin,_node_config(spec,buffers),Path(science["matched_NODE_folder"]),evidence)
            evidence["stage"] = "fresh_hybrid25"
            states = _hybrid_prepare(spec,science,buffers,saved,origin,context,anchors,targets,evidence,bounded)
        else:
            cache_pins = dict(_hybrid_files(Path(science["candidate_folder"])),**_node_files(Path(science["matched_NODE_folder"])))
            evidence["stage"] = "readonly_matched_NODE25"
            _node_certificate(buffers,origin,_node_config(spec,buffers),Path(science["matched_NODE_folder"]),evidence)
            evidence["stage"] = "readonly_hybrid26states"
            states = _load_progress(science,buffers,saved,origin,context,anchors,targets,evidence,bounded)
            if operation == "validate":
                evidence["stage"] = "paired_student_validation"
                _validate(spec,science,buffers,origin,states,context,arm,gate_sha256,evidence,bounded)
        _retained(buffers,payload,anchors,targets,evidence)
        evidence.update(context_digest=_seal(context),native_P0_moments_exactly_equal_cached_NODE0=True,
            actual_FP32_P0_X_Q_uniform_equal_reference=True,success_counts=_success_counts(evidence,science),passed=True)
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
            error = ValueError("BV worker exceeded300seconds")
            evidence.update(passed=False,deadline_error=str(error))
            if primary is None: primary = error
        if evidence["passed"] is not True:
            evidence["success_counts"] = None
        try:
            _write_new(output,evidence)
        except BaseException:
            if primary is not None: raise primary
            raise
    if primary is not None: raise primary
    return dict(evidence,evidence_path=str(output),evidence_sha256=_sha(output),validation_only=True)
