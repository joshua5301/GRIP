"""Unqualified EZ draft: physical tape, composed head, immutable evaluation records.

The caller must authenticate actual source/native-math admission before use.
This is a separate engine; no historical optimizer/resume engine is changed.
"""
import csv
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import time

import torch

from src import composed_centroid_joint_ce as joint
from src.dual_head_ce import _factor_digests, _files, _validate_optimizer
from src.kernel_mean_ce import _cpu, _plain, _seal, _write_state
from src.low_rank_assignment import LowRankMoments
from src.moments import augmented, decode_moments, make_material
from src.nystrom_ce import outer_gradient
from src.shared_features import _tensor_identity
from src.soft_ce_partition import solve_head_system, solve_inner_newton_first

SCHEMA = 1
TRAJECTORY = "cold_once_composed_centroid_stationary_head_v1"
POLICY = dict(schema=1, kind=TRAJECTORY, head_evaluation="exactly_once_per_P_step",
    continuation="cached_head_features_rhs_plus_one_exact_physical_tape_forward",
    frozen_scale="own_positive_raw_CE0", native_Adam_eps=1e-12, native_Adam_foreach=False)
WORK = ("physical_forward", "endpoint_map", "G_map", "head", "outer", "adjoint", "complete_G",
    "backward", "Adam", "P_updates", "reattachment", "checkpoint_writes", "resume_writes", "checkpoint_copies")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def integer(value, minimum=0):
    require(type(value) is int and value >= minimum, "Expected typed integer")
    return value


def scalar(value, positive=False):
    return joint._num(value, positive=positive)


def same(a, b):
    return _seal(a) == _seal(b)


def seal_record(value):
    value = _cpu(value)
    return dict(value, record_digest=_seal(value))


def pair(params, refs):
    return _factor_digests(params, refs["nodes"], refs["cells"], refs["rank"])


def owned(value):
    if torch.is_tensor(value):
        require(value.device.type == "cpu" and not value.requires_grad and value.grad_fn is None
            and value._base is None, "Record tensors must own detached CPU storage")
    elif isinstance(value, dict):
        for item in value.values(): owned(item)
    elif isinstance(value, (list,tuple)):
        for item in value: owned(item)


def optimizer(saved, config, refs, step):
    require(_validate_optimizer(saved,config,refs) == step, "Adam slot steps differ")
    require(set(saved["state"]) == (set() if step == 0 else {0,1}), "Both native Adam slots are required after an update")
    owned(saved)
    require(all(s["step"].dtype == torch.float32 for s in saved["state"].values()), "Original Adam step dtype differs")


def decoded(M, width, cells, classes, retain=lambda value:None):
    joint._matrix(M, (cells, 1+width+classes))
    centers, targets, mass = decode_moments(M, width)
    retain((centers,targets,mass))
    require(bool((mass > 0).all()) and bool(torch.isfinite(centers).all())
        and bool(torch.isfinite(targets).all()) and bool((targets >= 0).all())
        and bool((targets.sum(1) > 0).all()) and bool((targets.sum(0) > 0).all()),
        "Moment positive-mass/general-Q domain differs")
    return centers, targets, mass


def origin(M, centers, targets):
    return dict(moments=_tensor_identity(M), centers=_tensor_identity(centers), labels=_tensor_identity(targets))


def evaluation_link(E):
    return dict(step=E["step"], raw_CE=E["raw_CE"], CE0=E["CE0"], objective=E["objective"],
        evaluation_digest=E["record_digest"], parameters_digest=_seal(E["parameters"]),
        theta_digest=_seal(E["theta"]), raw_rhs_digest=_seal(E["raw_rhs"]),
        theta_initial_digest=_seal(E["theta_initial"]), parent_update_digest=E["parent_update_digest"],
        update_digest=None, update_after_parameters_digest=None, raw_adjoint_digest=None)


def validate_evaluation(E, config, context):
    refs, _, layout = joint.validate_context(context)
    fields = {"schema", "kind", "step", "config_digest", "context_digest", "source_admission_digest", "initial_parameter_digests",
        "parameters", "physical_moments", "physical_centers", "physical_targets", "physical_mass", "head_features",
        "theta", "theta_initial", "parent_update_digest", "raw_CE", "raw_rhs", "CE0", "objective", "head_work", "record_digest"}
    require(isinstance(E,dict) and set(E)==fields and type(E["schema"]) is int and E["schema"]==SCHEMA
        and E["kind"]=="composed_joint_head_evaluation_v1", "Composed evaluation schema differs")
    integer(E["step"]); require(E["record_digest"]==_seal({k:v for k,v in E.items() if k!="record_digest"})
        and E["config_digest"]==_seal(config) and E["context_digest"]==_seal(context)
        and E["source_admission_digest"]==_seal(refs["source_admission"])
        and E["initial_parameter_digests"]==context["native_parameter_digests"], "Evaluation seal/source changed")
    pair(E["parameters"],refs);owned(E);k,c=refs["cells"],refs["classes"];dev=torch.device("cpu")
    for name,shape in (("physical_moments",(k,layout.material_width)),("physical_centers",(k,layout.physical_dimension)),
        ("physical_targets",(k,c)),("head_features",(k,layout.critic_dimension)),("theta",(c,layout.critic_dimension+1))):
        joint._matrix(E[name],shape,dev)
    mass=E["physical_mass"]
    require(torch.is_tensor(mass) and mass.dtype==torch.float64 and mass.shape==(k,) and mass.device==dev
        and bool(torch.isfinite(mass).all()) and bool((mass>0).all()) and bool((E["physical_targets"]>=0).all())
        and bool((E["physical_targets"].sum(1)>0).all()) and bool((E["physical_targets"].sum(0)>0).all())
        and torch.equal(E["head_features"][:,:layout.physical_dimension],E["physical_centers"]), "Physical general-Q/head-prefix domain differs")
    joint._matrix(E["raw_rhs"],E["theta"].shape,dev)
    if E["theta_initial"] is not None: joint._matrix(E["theta_initial"],E["theta"].shape,dev)
    scalar(E["raw_CE"],True);scalar(E["CE0"],True);scalar(E["objective"])
    require(isinstance(E["head_work"],dict) and E["objective"]==E["raw_CE"]/E["CE0"]
        and E["head_work"].get("inner_converged") is True,"Head/scale is not accepted")
    require(0<=scalar(E["head_work"].get("inner_grad_max"))<=config["inner_tol"],"Head stationarity is not accepted")
    if E["step"]==0:
        require(E["theta_initial"] is None and E["parent_update_digest"] is None and E["raw_CE"]==E["CE0"]
            and pair(E["parameters"],refs)==context["native_parameter_digests"] and bool(E["parameters"][0].eq(0).all())
            and origin(E["physical_moments"],E["physical_centers"],E["physical_targets"])==context["native_origin"],
            "Physical origin evaluation differs from admitted native buffers")


def validate_evaluated_state(state, expected_config, context, folder=None):
    """State consistency only; ROOT wrapper authenticates accepted input files."""
    refs, _, layout = joint.validate_context(context)
    keys={"schema","trajectory_policy","step","scientific_endpoint","config","context","initial_parameters",
        "parameters","optimizer","CE0","current_evaluation","last_completed_update","earliest_best_evaluation",
        "snapshots","history","work","attempts","checkpoint_files_sha256","history_sha256","artifact_folder","state_digest"}
    require(isinstance(state,dict) and set(state) == keys and type(state["schema"]) is int and state["schema"] == SCHEMA
        and state["trajectory_policy"] == POLICY and same(state["config"],expected_config)
        and same(state["context"],context) and state["state_digest"] == _seal({k:v for k,v in state.items() if k != "state_digest"}),
        "State schema/source/config/seal differs")
    end=integer(state["step"]); limit=integer(state["scientific_endpoint"],1)
    require(end <= limit and limit == expected_config["scientific_endpoint"], "Scientific horizon changed")
    scalar(state["CE0"],True)
    require(pair(state["initial_parameters"],refs) == context["native_parameter_digests"]
        and bool(state["initial_parameters"][0].eq(0).all()), "Initial native factors changed")
    pair(state["parameters"],refs); owned(state)
    optimizer(state["optimizer"],expected_config,refs,end)
    history=state["history"]; snaps=state["snapshots"]; work=state["work"]
    require(isinstance(history,list) and len(history) == end+1 and isinstance(snaps,dict)
        and 0 in snaps and end in snaps and (end == 0 or 1 in snaps), "Endpoint/history missing")
    require(set(work) == set(state["attempts"]) == set(WORK), "Counter roles differ")
    for name in WORK:
        integer(work[name]); integer(state["attempts"][name]); require(state["attempts"][name] >= work[name], "Returned count exceeds attempts")
    require(work["physical_forward"] == end+1+work["reattachment"]
        and all(work[k] == end+1 for k in ("endpoint_map","head","outer")) and work["G_map"] == end
        and all(work[k] == end for k in ("adjoint","complete_G","backward","Adam","P_updates"))
        and work["checkpoint_writes"] == len(snaps) and work["resume_writes"] >= len(snaps), "Cold-once work differs")
    for i,row in enumerate(history):
        require(type(row["step"]) is int and row["step"] == i and scalar(row["CE0"],True) == state["CE0"], "History order/scale changed")
        scalar(row["raw_CE"],True); scalar(row["objective"])
        require(row["objective"] == row["raw_CE"]/state["CE0"], "History objective units changed")
        if i == 0:
            require(row["theta_initial_digest"] == _seal(None) and row["parent_update_digest"] is None, "Cold head initial changed")
        else:
            require(row["theta_initial_digest"] == history[i-1]["theta_digest"]
                and row["parent_update_digest"] == history[i-1]["update_digest"]
                and row["parameters_digest"] == history[i-1]["update_after_parameters_digest"], "Head/update/warm linkage differs")
        require((row["update_digest"] is None) == (i == end), "Accepted frontier update was already applied")
    for step,E in snaps.items():
        require(type(step) is int and 0 <= step <= end and E["step"] == step, "Snapshot step changed")
        validate_evaluation(E,expected_config,context)
        require(all(evaluation_link(E)[k] == history[step][k] for k in evaluation_link(E)
            if k not in ("update_digest","update_after_parameters_digest","raw_adjoint_digest")), "Snapshot/history changed")
    E=state["current_evaluation"]; validate_evaluation(E,expected_config,context)
    require(E["step"] == end and same(E,snaps[end]) and same(E["parameters"],state["parameters"])
        and E["CE0"] == state["CE0"], "Current endpoint/factors/scale differs")
    best=state["earliest_best_evaluation"]; validate_evaluation(best,expected_config,context)
    best_step=min(range(len(history)),key=lambda i:history[i]["objective"])
    require(best["step"] == best_step and best["record_digest"] == history[best_step]["evaluation_digest"], "Earliest best record changed")
    A=state["last_completed_update"]
    if end == 0:
        require(A is None, "Cold origin cannot carry an update")
    else:
        fields={"schema","kind","step","evaluation_digest","previous_update_digest","raw_adjoint_initial_digest",
            "adjoint_work","raw_adjoint","normalized_G","native_grad_U","native_grad_V","parameters_after","optimizer_after","record_digest"}
        require(isinstance(A,dict) and set(A) == fields and type(A["schema"]) is int and A["schema"] == SCHEMA
            and A["kind"] == "composed_joint_pending_update_v1" and type(A["step"]) is int and A["step"] == end-1
            and A["record_digest"] == _seal({k:v for k,v in A.items() if k != "record_digest"})
            and A["record_digest"] == history[end-1]["update_digest"]
            and A["evaluation_digest"] == history[end-1]["evaluation_digest"]
            and same(A["parameters_after"],state["parameters"]) and same(A["optimizer_after"],state["optimizer"])
            and A["previous_update_digest"] == (None if end == 1 else history[end-2]["update_digest"])
            and A["raw_adjoint_initial_digest"] == (_seal(None) if end == 1 else history[end-2]["raw_adjoint_digest"])
            and A["adjoint_work"].get("cg_converged") is True, "Last update/warm vector/slots differs")
        joint._matrix(A["raw_adjoint"],E["theta"].shape,torch.device("cpu"))
        joint._matrix(A["normalized_G"],E["physical_moments"].shape,torch.device("cpu"))
        for name,shape in (("native_grad_U",(refs["nodes"],refs["rank"])),("native_grad_V",(refs["cells"],refs["rank"]))):
            value=A[name]; require(torch.is_tensor(value) and value.dtype == torch.float32 and value.shape == shape
                and bool(torch.isfinite(value).all()), "Update gradient shape/dtype/domain differs")
        require(_seal(A["raw_adjoint"]) == history[end-1]["raw_adjoint_digest"], "Raw vector identity changed")
    require(set(state["checkpoint_files_sha256"]) == {f"step_{i:06d}.pt" for i in snaps}, "Snapshot file closure differs")
    for digest in [state["history_sha256"],*state["checkpoint_files_sha256"].values()]: joint._hex(digest)
    require(type(state["artifact_folder"]) is str and Path(state["artifact_folder"]).is_absolute()
        and str(Path(state["artifact_folder"]).resolve()) == state["artifact_folder"], "Artifact folder is not normalized ABS")
    if folder is not None:
        folder=Path(folder)
        _files({str(folder/"checkpoints"/name):digest for name,digest in state["checkpoint_files_sha256"].items()})
        require(hashlib.sha256((folder/"optimization.csv").read_bytes()).hexdigest() == state["history_sha256"], "History bytes changed")
    return True


def evaluate_at_current_factors(step, params, hard, material, rows, q, composed,
        options, context, config, call, theta_initial=None, parent_update=None, CE0=None, check=lambda:None, retain=lambda key,value:None):
    refs,_,layout=joint.validate_context(context);k,c=refs["cells"],refs["classes"]
    M=call("physical_forward",LowRankMoments.apply,*params,hard,material,.05,options["chunk_size"])
    centers,targets,mass=decoded(M.detach(),layout.physical_dimension,k,c,lambda value:retain("physical_quotients",value))
    if step==0: require(origin(M.detach(),centers,targets)==context["native_origin"],"Original physical origin differs before first head")
    features=call("endpoint_map",composed,centers)
    fit=call("head",solve_inner_newton_first,features,targets,torch.full_like(mass,1/k),options["penalty"],
        theta_initial,options["inner_max_iter"],options["inner_tol"],cg_max_iter=options["cg_max_iter"],
        cg_check_interval=options.get("cg_check_interval",1))
    theta=fit["theta"].detach();head=_plain({n:v for n,v in fit.items() if n!="theta"})
    require(head.get("inner_converged") is True and 0<=scalar(head.get("inner_grad_max"))<=options["inner_tol"],
        "Original stationary head failed");check()
    CE,rhs=call("outer",outer_gradient,rows,q,theta,options["outer_chunk_size"])
    scalar(CE,True);CE0=CE if CE0 is None else scalar(CE0,True)
    E=seal_record(dict(schema=SCHEMA,kind="composed_joint_head_evaluation_v1",step=step,config_digest=_seal(config),
        context_digest=_seal(context),source_admission_digest=_seal(refs["source_admission"]),
        initial_parameter_digests=context["native_parameter_digests"],parameters=params,
        physical_moments=M.detach(),physical_centers=centers,physical_targets=targets,physical_mass=mass,
        head_features=features,theta=theta,theta_initial=theta_initial,
        parent_update_digest=None if parent_update is None else parent_update["record_digest"],raw_CE=CE,raw_rhs=rhs,
        CE0=CE0,objective=CE/CE0,head_work=head))
    retain("returned_evaluation",E);validate_evaluation(E,config,context);check();return E,M


def reattach_physical_tape(E,params,hard,material,options,call):
    require(same(params,E["parameters"]),"Accepted frontier factors changed")
    M=call("physical_forward",LowRankMoments.apply,*params,hard,material,.05,options["chunk_size"])
    require(_tensor_identity(M.detach())==_tensor_identity(E["physical_moments"]),"Accepted physical tape changed bytes")
    call("reattachment",lambda:None);return M


def apply_pending_update(E,M,params,adam,composed,options,context,call,previous_update=None,check=lambda:None,retain=lambda key,value:None):
    refs,_,layout=joint.validate_context(context);device=params[0].device
    require(same(params,E["parameters"]) and _tensor_identity(M.detach())==_tensor_identity(E["physical_moments"])
        and M.requires_grad,"Pending update has no exact live physical tape")
    features,targets,mass=(E[n].to(device) for n in ("head_features","physical_targets","physical_mass"))
    theta,rhs=E["theta"].to(device),E["raw_rhs"].to(device)
    initial=None if previous_update is None else previous_update["raw_adjoint"].to(device)
    vector,diagnostic=call("adjoint",solve_head_system,augmented(features),targets,torch.full_like(mass,1/refs["cells"]),
        theta,options["penalty"],rhs,rtol=options["cg_rtol"],max_iter=options["cg_max_iter"],initial=initial,
        cg_check_interval=options.get("cg_check_interval",1))
    vector=vector.detach();require(diagnostic.get("cg_converged") is True and bool(torch.isfinite(vector).all()),"Raw adjoint failed")
    G=call("complete_G",joint.complete_moment_cotangent,M.detach(),layout,composed,theta,vector,options["penalty"],E["CE0"],
        map_call=lambda f,x:call("G_map",f,x))
    adam.zero_grad(set_to_none=True);call("backward",M.backward,G)
    require(all(p.grad is not None and p.grad.dtype==torch.float32 and bool(torch.isfinite(p.grad).all()) for p in params),
        "Native physical pullback differs")
    retained=_cpu(dict(raw_adjoint=vector,normalized_G=G,native_grad_U=params[0].grad,native_grad_V=params[1].grad))
    retain("returned_pullback",retained);check();call("Adam",adam.step);call("P_updates",lambda:None)
    A=seal_record(dict(schema=SCHEMA,kind="composed_joint_pending_update_v1",step=E["step"],evaluation_digest=E["record_digest"],
        previous_update_digest=None if previous_update is None else previous_update["record_digest"],raw_adjoint_initial_digest=_seal(initial),
        adjoint_work=_plain(diagnostic),**retained,parameters_after=params,optimizer_after=adam.state_dict()))
    retain("returned_update",A);require(all(bool(torch.isfinite(p).all()) for p in params),"Native Adam factors nonfinite");check();return A


def optimize_composed_joint_ce(z,q,hard,phi,initial_parameters,options,context,folder,target_step,composed_features,source_features,
        checkpoint_steps=(0,1,25),resume_state=None,stop=lambda:False):
    """Caller must bind actual source/native math and watchdog; no initializer."""
    started=time.monotonic(); refs,_,layout=joint.validate_context(context)
    limit=integer(refs.get("scientific_endpoint",25),1); target=integer(target_step)
    require(0 <= target <= limit, "Unfrozen horizon is unsupported")
    config=joint.validate_input_metadata(z,q,hard,initial_parameters,phi,options,context,source_features,composed_features)
    config.update(trajectory_policy=POLICY,scientific_endpoint=limit)
    folder=Path(folder); require(folder.is_absolute() and str(folder.resolve()) == str(folder), "Folder must be normalized ABS")
    if resume_state is not None:
        validate_evaluated_state(resume_state,config,context,resume_state["artifact_folder"])
        require(resume_state["step"] < target, "No endpoint re-evaluation or extra horizon")
    folder.mkdir(parents=True,exist_ok=False); (folder/"checkpoints").mkdir()
    params=[]; adam=None
    work=dict.fromkeys(WORK,0); attempts=dict.fromkeys(WORK,0); snaps={}; history=[]; file_pins={}; E=A=M=best=None
    checkpoints={integer(i) for i in checkpoint_steps if i <= target} | {0,target} | ({1} if target else set())
    returned={}
    def check():
        if stop() or time.monotonic()-started > 300:
            raise InterruptedError("Composed physical trajectory stopped or exceeded300s")
        require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 <= 16*1024**3, "Trajectory peak RSS exceeds16GiB")
        if z.device.type == "cuda":
            require(torch.cuda.max_memory_allocated(z.device) <= 4*1024**3
                and torch.cuda.max_memory_reserved(z.device) <= 6*1024**3, "Trajectory CUDA peaks exceed4/6GiB")
    def call(key,func,*args,**kwargs):
        attempts[key]+=1; value=func(*args,**kwargs); work[key]+=1
        if key in ("physical_forward","endpoint_map","G_map","head","outer","adjoint","complete_G"):
            returned[key]=_cpu(value)
        if key in ("endpoint_map","G_map"): returned["composed_components"]=composed_features.last_returned()
        elif key == "backward": returned[key]=_cpu(dict(grad_U=params[0].grad,grad_V=params[1].grad))
        elif key == "Adam": returned[key]=_cpu(dict(parameters=params,optimizer=adam.state_dict()))
        if key in ("physical_forward","endpoint_map","G_map","head","outer","adjoint","complete_G","backward","P_updates"):
            check()
        return value
    def retain(key,value):
        returned[key]=_cpu(value)
    try:
        params=[p.detach().clone().requires_grad_() for p in initial_parameters]
        material=make_material(z,q);rows=source_features.outer_rows()
        retain("physical_material",material);retain("literal_source_outer_rows",torch.from_numpy(rows.copy()))
        adam=torch.optim.Adam(params,lr=options["lr"],eps=1e-12,foreach=False); check()
        if resume_state is not None:
            for p,saved in zip(params,resume_state["parameters"],strict=True):
                with torch.no_grad(): p.copy_(saved.to(p))
            adam.load_state_dict(_cpu(resume_state["optimizer"])); E=resume_state["current_evaluation"]
            A=resume_state["last_completed_update"]; best=resume_state["earliest_best_evaluation"]
            snaps=dict(resume_state["snapshots"]); history=[dict(r) for r in resume_state["history"]]
            work=dict(resume_state["work"]); attempts=dict(resume_state["attempts"])
            file_pins=dict(resume_state["checkpoint_files_sha256"])
            for name,digest in file_pins.items():
                call("checkpoint_copies",shutil.copyfile,Path(resume_state["artifact_folder"])/"checkpoints"/name,folder/"checkpoints"/name)
                _files({str(folder/"checkpoints"/name):digest})
            M=reattach_physical_tape(E,params,hard,material,options,call); check()
        else:
            E,M=evaluate_at_current_factors(0,params,hard,material,rows,q,composed_features,options,context,config,call,check=check,retain=retain)
            history.append(evaluation_link(E)); best=E
        while True:
            step=E["step"]; validate_evaluation(E,config,context)
            if E["objective"] < best["objective"]: best=E
            if step in checkpoints and step not in snaps:
                name=f"step_{step:06d}.pt"; file_pins[name]=call("checkpoint_writes",_write_state,E,folder/"checkpoints"/name,exclusive=True)
                snaps[step]=E
            if step in checkpoints and (resume_state is None or step != resume_state["step"]):
                csv_path=folder/"optimization.csv"; temporary=csv_path.with_suffix(".tmp.csv")
                with temporary.open("x",newline="") as stream:
                    writer=csv.DictWriter(stream,fieldnames=list(history[0])); writer.writeheader(); writer.writerows(history)
                    stream.flush(); os.fsync(stream.fileno())
                temporary.replace(csv_path)
                next_work=dict(work,resume_writes=work["resume_writes"]+1)
                next_attempts=dict(attempts,resume_writes=attempts["resume_writes"]+1)
                state=_cpu(dict(schema=SCHEMA,trajectory_policy=POLICY,step=step,scientific_endpoint=limit,config=config,context=context,
                    initial_parameters=initial_parameters,parameters=params,optimizer=adam.state_dict(),CE0=E["CE0"],current_evaluation=E,
                    last_completed_update=A,earliest_best_evaluation=best,snapshots=snaps,history=history,work=next_work,attempts=next_attempts,
                    checkpoint_files_sha256=file_pins,history_sha256=hashlib.sha256(csv_path.read_bytes()).hexdigest(),artifact_folder=str(folder)))
                state["state_digest"]=_seal(state); validate_evaluated_state(state,config,context,folder)
                call("resume_writes",_write_state,state,folder/"resume.pt")
            if step == target:
                check(); _files(refs["files_sha256"]); _files(refs["current_source"]["files"])
                require(joint._runtime(z.device) == refs["runtime"] and source_features.descriptor()==context["feature_contract"]["source_outer"]
                    and composed_features.descriptor()==context["feature_contract"]["composed"], "Runtime/owning source changed at exit")
                check()
                return state
            A=apply_pending_update(E,M,params,adam,composed_features,options,context,call,A,check,retain)
            history[-1].update(update_digest=A["record_digest"],update_after_parameters_digest=_seal(A["parameters_after"]),
                raw_adjoint_digest=_seal(A["raw_adjoint"]))
            theta=E["theta"].to(z.device); CE0=E["CE0"]
            E,M=evaluate_at_current_factors(step+1,params,hard,material,rows,q,composed_features,options,context,config,call,
                theta,A,CE0,check,retain); history.append(evaluation_link(E))
    except BaseException as error:
        returned["last_composed_components"]=composed_features.last_returned()
        failure=dict(schema=1,kind=TRAJECTORY,passed=False,error_type=type(error).__name__,error=str(error),
            seconds=time.monotonic()-started,work=work,attempts=attempts,inflight_operation_interiors="unknown",
            no_retry_or_rescue=True,partial_state_is_not_successful_resume=True)
        raw=folder/"failure_partial.pt"; failure["raw_evidence"]=dict(path=str(raw),exists=False,bytes=None,sha256=None,hash_unknown=True)
        try:
            failure["raw_evidence"]["sha256"]=_write_state(_cpu(dict(current_evaluation=E,last_update=A,parameters=params,
                optimizer=None if adam is None else adam.state_dict(),history=history,work=work,attempts=attempts,returned=returned)),raw,exclusive=True)
            failure["raw_evidence"]["hash_unknown"]=False
        except BaseException as observation:
            failure["raw_evidence"]["error"]=repr(observation)
        finally:
            failure["raw_evidence"]["exists"]=raw.exists()
            if raw.exists(): failure["raw_evidence"]["bytes"]=raw.stat().st_size
        with (folder/"failure.json").open("x") as stream:
            json.dump(failure,stream,indent=2,allow_nan=False); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        raise
