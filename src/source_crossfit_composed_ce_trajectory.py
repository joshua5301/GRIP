"""Two fixed source folds, one native packed physical tape, two cold-once heads.

Original head/adjoint/generalQ/LowRank/Adam operations are unchanged suppliers.
The first live F0 creates this mode's origin; no historical CE0 or state is used.
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

from src import source_crossfit_composed_ce_native as native
from src import composed_centroid_joint_ce as joint
from src.dual_head_ce import _factor_digests, _files, _validate_optimizer
from src.kernel_mean_ce import _cpu, _plain, _seal, _write_state
from src.low_rank_assignment import LowRankMoments
from src.moments import augmented, decode_moments
from src.nystrom_ce import outer_gradient
from src.shared_features import _tensor_identity
from src.soft_ce_partition import solve_head_system, solve_inner_newton_first

SCHEMA=1
TRAJECTORY="cold_once_two_fixed_source_fold_original_composed_CE_v1"
POLICY=dict(schema=1,kind=TRAJECTORY,head_evaluation="two_once_per_P_step",continuation="cached_pair_plus_one_packed_reattachment",
    frozen_scale="own_positive_symmetric_raw_CE0",native_Adam_eps=1e-12,native_Adam_foreach=False)
WORK=("packed_physical_forward","fold_physical_decode","endpoint_map","head","held_outer_CE_RHS","raw_adjoint",
    "raw_moment_G","G_map","packed_complete_G","backward","Adam","P_updates","reattachment","checkpoint_writes","resume_writes","checkpoint_copies")


def require(ok,message):
    if not ok:raise ValueError(message)


def integer(value,minimum=0):
    require(type(value) is int and value>=minimum,"Typed integer required");return value


def scalar(value,positive=False):return joint._num(value,positive)
def same(a,b):return _seal(a)==_seal(b)
def pair(params,refs):return _factor_digests(params,refs["nodes"],refs["cells"],refs["rank"])


def owned(value):
    if torch.is_tensor(value):
        require(value.device.type=="cpu" and not value.requires_grad and value.grad_fn is None and value._base is None,"Record must own detached CPU tensors")
    elif isinstance(value,dict):
        for item in value.values():owned(item)
    elif isinstance(value,(list,tuple)):
        for item in value:owned(item)


def seal_record(value):
    value=_cpu(value);return dict(value,record_digest=_seal(value))


def evaluation_link(E):
    return dict(step=E["step"],raw_CE=E["raw_CE"],CE0=E["CE0"],objective=E["objective"],evaluation_digest=E["record_digest"],
        parameters_digest=_seal(E["parameters"]),theta_digest=_seal({f:E["folds"][f]["theta"] for f in ("A","B")}),
        raw_rhs_digest=_seal({f:E["folds"][f]["raw_rhs"] for f in ("A","B")}),
        theta_initial_digest=_seal({f:E["folds"][f]["theta_initial"] for f in ("A","B")}),parent_update_digest=E["parent_update_digest"],
        update_digest=None,update_after_parameters_digest=None,raw_adjoint_digest=None)


def validate_quotients(centers,targets,mass,layout,k,device):
    joint._matrix(centers,(k,layout.physical_dimension),device);joint._matrix(targets,(k,layout.classes),device)
    require(mass.shape==(k,) and mass.dtype==torch.float64 and mass.device==device and bool(torch.isfinite(mass).all())
        and bool((mass>0).all()) and bool((targets>=0).all()) and bool((targets.sum(1)>0).all())
        and bool((targets.sum(0)>0).all()),"Fold physical positive mass/generalQ differs")


def validate_evaluation(E,config,context):
    refs,_,layout=native.validate_context(context);k,c=refs["cells"],refs["classes"];dev=torch.device("cpu")
    fields={"schema","kind","step","config_digest","context_digest","source_admission_digest","initial_parameter_digests",
        "parameters","packed_moments","folds","parent_update_digest","raw_CE","CE0","objective","record_digest"}
    require(isinstance(E,dict) and set(E)==fields and E["schema"]==SCHEMA and E["kind"]=="two_fold_composed_head_pair_evaluation_v1",
        "New two-fold E schema differs")
    integer(E["step"]);owned(E);pair(E["parameters"],refs)
    require(E["record_digest"]==_seal({k:v for k,v in E.items() if k!="record_digest"}) and E["config_digest"]==_seal(config)
        and E["context_digest"]==_seal(context) and E["source_admission_digest"]==_seal(refs["source_admission"])
        and E["initial_parameter_digests"]==context["pre_origin_contract"]["native_parameter_digests"],"Evaluation seal/source changed")
    joint._matrix(E["packed_moments"],(k,2*layout.material_width),dev);require(set(E["folds"])=={"A","B"},"Pair absent")
    foldkeys={"physical_moments","physical_centers","physical_targets","physical_mass","head_features","theta","theta_initial","raw_CE","raw_rhs","head_work"}
    for i,f in enumerate(("A","B")):
        fold=E["folds"][f];require(set(fold)==foldkeys,"Fold fields differ")
        joint._matrix(fold["physical_moments"],(k,layout.material_width),dev)
        require(torch.equal(fold["physical_moments"],E["packed_moments"][:,i*layout.material_width:(i+1)*layout.material_width]),"Packed/fold moment bits differ")
        validate_quotients(fold["physical_centers"],fold["physical_targets"],fold["physical_mass"],layout,k,dev)
        joint._matrix(fold["head_features"],(k,layout.critic_dimension),dev);joint._matrix(fold["theta"],(c,layout.critic_dimension+1),dev)
        joint._matrix(fold["raw_rhs"],fold["theta"].shape,dev)
        if fold["theta_initial"] is not None:joint._matrix(fold["theta_initial"],fold["theta"].shape,dev)
        require(torch.equal(fold["head_features"][:,:layout.physical_dimension],fold["physical_centers"])
            and fold["head_work"].get("inner_converged") is True
            and 0<=scalar(fold["head_work"].get("inner_grad_max"))<=config["inner_tol"],"Both stationary full heads required")
        scalar(fold["raw_CE"],True)
    scalar(E["CE0"],True);scalar(E["raw_CE"],True)
    require(E["raw_CE"]==.5*(E["folds"]["A"]["raw_CE"]+E["folds"]["B"]["raw_CE"])
        and E["objective"]==E["raw_CE"]/E["CE0"],"Symmetric CE/own scale differs")
    if E["step"]==0:
        org=context["native_origin"]
        require(E["parent_update_digest"] is None and all(x["theta_initial"] is None for x in E["folds"].values())
            and E["raw_CE"]==E["CE0"] and bool(E["parameters"][0].eq(0).all())
            and pair(E["parameters"],refs)==E["initial_parameter_digests"]
            and _tensor_identity(E["packed_moments"])==org["packed_moments"],"New cold packed origin differs")
        for f,v in E["folds"].items():
            require({name:_tensor_identity(v[key]) for name,key in (("centers","physical_centers"),("targets","physical_targets"),("mass","physical_mass"))}==org["folds"][f],"Own native quotient origin differs")


def validate_evaluated_state(state,expected_config,context,folder=None):
    """Pure typed consistency and immutable artifact seals; no current-source replay."""
    refs,_,layout=native.validate_context(context)
    keys={"schema","trajectory_policy","step","scientific_endpoint","config","context","initial_parameters","parameters","optimizer","CE0",
        "current_evaluation","last_completed_update","earliest_best_evaluation","snapshots","history","work","attempts","checkpoint_files_sha256","history_sha256","artifact_folder","state_digest"}
    require(isinstance(state,dict) and set(state)==keys and state["schema"]==1 and state["trajectory_policy"]==POLICY
        and same(state["config"],expected_config) and same(state["context"],context)
        and state["state_digest"]==_seal({k:v for k,v in state.items() if k!="state_digest"}),"Typed pair state/config/context seal differs")
    end=integer(state["step"]);require(end<=state["scientific_endpoint"]==expected_config["scientific_endpoint"]==25,"Horizon differs")
    owned(state);scalar(state["CE0"],True);pair(state["parameters"],refs)
    require(pair(state["initial_parameters"],refs)==context["pre_origin_contract"]["native_parameter_digests"] and bool(state["initial_parameters"][0].eq(0).all()),"Native cold pair differs")
    require(_validate_optimizer(state["optimizer"],expected_config,refs)==end and set(state["optimizer"]["state"])==(set() if end==0 else {0,1})
        and all(v["step"].dtype==torch.float32 for v in state["optimizer"]["state"].values()),"Original Adam slot steps/types differ")
    h=state["history"];snaps=state["snapshots"];w=state["work"]
    require(len(h)==end+1 and 0 in snaps and end in snaps and (end==0 or 1 in snaps) and set(w)==set(state["attempts"])==set(WORK),"History/snapshots/count roles differ")
    for name in WORK:integer(w[name]);integer(state["attempts"][name]);require(w[name]<=state["attempts"][name],"Return exceeds attempt")
    require(w["packed_physical_forward"]==end+1+w["reattachment"] and all(w[x]==2*(end+1) for x in ("fold_physical_decode","endpoint_map","head","held_outer_CE_RHS"))
        and all(w[x]==2*end for x in ("raw_adjoint","raw_moment_G","G_map"))
        and all(w[x]==end for x in ("packed_complete_G","backward","Adam","P_updates"))
        and w["checkpoint_writes"]==len(snaps) and w["resume_writes"]>=len(snaps),"Cold-once pair work differs")
    cold_theta=_seal(dict(A=None,B=None))
    for i,row in enumerate(h):
        require(row["step"]==i and row["CE0"]==state["CE0"] and row["objective"]==row["raw_CE"]/row["CE0"],"History scale/order differs")
        scalar(row["raw_CE"],True)
        require((row["update_digest"] is None)==(i==end),"Cached frontier was already applied")
        if i==0:require(row["theta_initial_digest"]==cold_theta and row["parent_update_digest"] is None,"Cold heads differ")
        else:require(row["theta_initial_digest"]==h[i-1]["theta_digest"] and row["parent_update_digest"]==h[i-1]["update_digest"]
            and row["parameters_digest"]==h[i-1]["update_after_parameters_digest"],"Two head warm/update links differ")
    for step,E in snaps.items():
        require(type(step) is int and E["step"]==step and 0<=step<=end,"Snapshot step differs");validate_evaluation(E,expected_config,context)
        require(all(evaluation_link(E)[k]==h[step][k] for k in evaluation_link(E) if k not in ("update_digest","update_after_parameters_digest","raw_adjoint_digest")),"Snapshot/history link differs")
    E=state["current_evaluation"];validate_evaluation(E,expected_config,context)
    require(E["step"]==end and same(E,snaps[end]) and same(E["parameters"],state["parameters"]) and E["CE0"]==state["CE0"],"Current frontier differs")
    best=state["earliest_best_evaluation"];validate_evaluation(best,expected_config,context)
    first=min(range(len(h)),key=lambda i:h[i]["objective"])
    require(best["step"]==first and best["record_digest"]==h[first]["evaluation_digest"],"Earliest best differs")
    A=state["last_completed_update"]
    if end==0:require(A is None,"Cold origin has an update")
    else:
        fields={"schema","kind","step","evaluation_digest","previous_update_digest","raw_adjoint_initial_digest","folds","normalized_G","native_grad_U","native_grad_V","parameters_after","optimizer_after","record_digest"}
        require(set(A)==fields and A["schema"]==1 and A["kind"]=="two_fold_composed_pending_update_v1" and A["step"]==end-1
            and A["record_digest"]==_seal({k:v for k,v in A.items() if k!="record_digest"}) and A["record_digest"]==h[end-1]["update_digest"]
            and A["evaluation_digest"]==h[end-1]["evaluation_digest"] and same(A["parameters_after"],state["parameters"])
            and same(A["optimizer_after"],state["optimizer"]) and A["previous_update_digest"]==(None if end==1 else h[end-2]["update_digest"])
            and A["raw_adjoint_initial_digest"]==(cold_theta if end==1 else h[end-2]["raw_adjoint_digest"]),"Last update/slots/warm links differ")
        require(set(A["folds"])=={"A","B"},"Two raw adjoints absent")
        for f,a in A["folds"].items():
            require(set(a)=={"raw_adjoint","raw_moment_G","adjoint_work"} and a["adjoint_work"].get("cg_converged") is True,"Raw adjoint not accepted")
            joint._matrix(a["raw_adjoint"],E["folds"][f]["theta"].shape,torch.device("cpu"));joint._matrix(a["raw_moment_G"],(refs["cells"],layout.material_width),torch.device("cpu"))
        joint._matrix(A["normalized_G"],E["packed_moments"].shape,torch.device("cpu"))
        require(_seal({f:A["folds"][f]["raw_adjoint"] for f in ("A","B")})==h[end-1]["raw_adjoint_digest"],"Raw pair identity differs")
        for key,shape in (("native_grad_U",(refs["nodes"],refs["rank"])),("native_grad_V",(refs["cells"],refs["rank"]))):
            value=A[key];require(value.dtype==torch.float32 and value.shape==shape and bool(torch.isfinite(value).all()),"Native gradient differs")
    require(set(state["checkpoint_files_sha256"])=={f"step_{i:06d}.pt" for i in snaps},"Checkpoint closure differs")
    for digest in [state["history_sha256"],*state["checkpoint_files_sha256"].values()]:joint._hex(digest)
    require(type(state["artifact_folder"]) is str and Path(state["artifact_folder"]).is_absolute() and str(Path(state["artifact_folder"]).resolve())==state["artifact_folder"],"Artifact folder not ABS")
    if folder is not None:
        folder=Path(folder);_files({str(folder/"checkpoints"/name):digest for name,digest in state["checkpoint_files_sha256"].items()})
        require(hashlib.sha256((folder/"optimization.csv").read_bytes()).hexdigest()==state["history_sha256"],"History bytes changed")
    return True


def evaluate_pair(step,params,hard,material,owner,composed,options,contract,context,config,call,retain,check,
        theta_initial=None,parent_update=None,CE0=None):
    refs,_,layout=native.validate_pre_origin_contract(contract);k=refs["cells"];width=layout.material_width
    M=call("packed_physical_forward",LowRankMoments.apply,*params,hard,material,.05,options["chunk_size"])
    quotients={};parts={}
    for i,f in enumerate(("A","B")):
        part=M.detach()[:,i*width:(i+1)*width]
        centers,targets,mass=call("fold_physical_decode",decode_moments,part,layout.physical_dimension)
        retain("native_quotients_"+f,dict(centers=centers,targets=targets,mass=mass));validate_quotients(centers,targets,mass,layout,k,M.device)
        quotients[f]=(centers,targets,mass);parts[f]=part
    if context is None:
        require(step==0,"Only first F0 creates origin");context=native.bind_origin_context(contract,M.detach(),quotients);retain("native_context",context)
    else:native.validate_context(context)
    folds={}
    for f in ("A","B"):
        centers,targets,mass=quotients[f];features=call("endpoint_map",composed,centers)
        initial=None if theta_initial is None else theta_initial[f]
        fit=call("head",solve_inner_newton_first,features,targets,torch.full_like(mass,1/k),options["penalty"],initial,
            options["inner_max_iter"],options["inner_tol"],cg_max_iter=options["cg_max_iter"],cg_check_interval=options.get("cg_check_interval",1))
        theta=fit["theta"].detach();head=_plain({n:v for n,v in fit.items() if n!="theta"})
        require(head.get("inner_converged") is True and 0<=scalar(head.get("inner_grad_max"))<=options["inner_tol"],"Original stationary fold head failed");check()
        opposite="B" if f=="A" else "A"
        CE,rhs=call("held_outer_CE_RHS",outer_gradient,owner.held_numpy_rows(opposite),owner.held_Q(opposite,theta.device),theta,options["outer_chunk_size"])
        scalar(CE,True)
        folds[f]=dict(physical_moments=parts[f],physical_centers=centers,physical_targets=targets,physical_mass=mass,
            head_features=features,theta=theta,theta_initial=initial,raw_CE=CE,raw_rhs=rhs,head_work=head)
    CE=.5*(folds["A"]["raw_CE"]+folds["B"]["raw_CE"]);CE0=CE if CE0 is None else scalar(CE0,True)
    E=seal_record(dict(schema=1,kind="two_fold_composed_head_pair_evaluation_v1",step=step,config_digest=_seal(config),context_digest=_seal(context),
        source_admission_digest=_seal(refs["source_admission"]),initial_parameter_digests=contract["native_parameter_digests"],parameters=params,
        packed_moments=M.detach(),folds=folds,parent_update_digest=None if parent_update is None else parent_update["record_digest"],raw_CE=CE,CE0=CE0,objective=CE/CE0))
    retain("returned_evaluation",E);validate_evaluation(E,config,context);check();return E,M,context


def apply_pending_update(E,M,params,adam,composed,options,context,call,retain,check,previous_update=None):
    refs,_,layout=native.validate_context(context);device=params[0].device;k=refs["cells"]
    require(same(params,E["parameters"]) and _tensor_identity(M.detach())==_tensor_identity(E["packed_moments"]) and M.requires_grad,"Exact live packed tape absent")
    vectors={};initials={};folds={}
    for f in ("A","B"):
        e=E["folds"][f];features,targets,mass=(e[x].to(device) for x in ("head_features","physical_targets","physical_mass"))
        theta,rhs=e["theta"].to(device),e["raw_rhs"].to(device)
        initial=None if previous_update is None else previous_update["folds"][f]["raw_adjoint"].to(device);initials[f]=initial
        vector,diag=call("raw_adjoint",solve_head_system,augmented(features),targets,torch.full_like(mass,1/k),theta,options["penalty"],rhs,
            rtol=options["cg_rtol"],max_iter=options["cg_max_iter"],initial=initial,cg_check_interval=options.get("cg_check_interval",1))
        vector=vector.detach();retain("raw_adjoint_"+f,dict(vector=vector,diagnostic=diag))
        require(diag.get("cg_converged") is True and bool(torch.isfinite(vector).all()),"Original raw fold adjoint failed")
        vectors[f]=vector;folds[f]=dict(raw_adjoint=vector,adjoint_work=_plain(diag))
    raw={}
    def observed(key,value):
        retain("packed_G_"+key,value)
        if key in ("raw_A","raw_B"):raw[key[-1]]=value
    G=call("packed_complete_G",native.complete_packed_cotangent,M.detach(),layout,composed,
        E["folds"]["A"]["theta"].to(device),E["folds"]["B"]["theta"].to(device),vectors["A"],vectors["B"],options["penalty"],E["CE0"],
        map_call=lambda f,x:call("G_map",f,x),raw_call=lambda f,*a,**kw:call("raw_moment_G",f,*a,**kw),observe_return=observed)
    for f in ("A","B"):folds[f]["raw_moment_G"]=raw[f]
    adam.zero_grad(set_to_none=True);call("backward",M.backward,G)
    require(all(p.grad is not None and p.grad.dtype==torch.float32 and bool(torch.isfinite(p.grad).all()) for p in params),"Native packed pullback failed")
    retained=_cpu(dict(folds=folds,normalized_G=G,native_grad_U=params[0].grad,native_grad_V=params[1].grad));retain("returned_pullback",retained)
    check();call("Adam",adam.step);call("P_updates",lambda:None)
    A=seal_record(dict(schema=1,kind="two_fold_composed_pending_update_v1",step=E["step"],evaluation_digest=E["record_digest"],
        previous_update_digest=None if previous_update is None else previous_update["record_digest"],raw_adjoint_initial_digest=_seal(initials),
        **retained,parameters_after=params,optimizer_after=adam.state_dict()))
    retain("returned_update",A);require(all(bool(torch.isfinite(p).all()) for p in params),"Native Adam nonfinite");check();return A


def optimize_source_crossfit_composed_ce(initial_parameters,hard,options,pre_origin_contract,source_owner,composed_features,
        packed_material,folder,target_step,checkpoint_steps=(0,1,25),resume_state=None,stop=lambda:False):
    started=time.monotonic();refs,_,_=native.validate_pre_origin_contract(pre_origin_contract)
    config=native.validate_inputs(initial_parameters,hard,options,pre_origin_contract,source_owner,composed_features,packed_material)
    config.update(trajectory_policy=POLICY,scientific_endpoint=25);target=integer(target_step)
    require(target in (1,25) and (resume_state is None)==(target==1),"Cold1 then authenticated24 only")
    context=None if resume_state is None else resume_state["context"]
    if resume_state is not None:
        require(context["pre_origin_contract"]==pre_origin_contract and resume_state["step"]==1,"Accepted input/origin frontier differs")
        validate_evaluated_state(resume_state,config,context,resume_state["artifact_folder"])
    folder=Path(folder);require(folder.is_absolute() and str(folder.resolve())==str(folder),"Folder must be normalized ABS")
    folder.mkdir(parents=True,exist_ok=False);(folder/"checkpoints").mkdir()
    params=[];adam=None;E=A=M=best=None;history=[];snaps={};file_pins={};state=None
    work=dict.fromkeys(WORK,0);attempts=dict.fromkeys(WORK,0);returned={};checkpoints={0,1,target}
    def check():
        require(not stop() and time.monotonic()-started<=300,"Two-fold trajectory stopped or exceeded300s")
        require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=16*1024**3,"RSS exceeds16GiB")
        if initial_parameters[0].device.type=="cuda":require(torch.cuda.max_memory_allocated(0)<=4*1024**3 and torch.cuda.max_memory_reserved(0)<=6*1024**3,"CUDA exceeds4/6GiB")
    def retain(key,value):
        returned[key]=value;returned[key]=_cpu(value)
    def call(key,func,*args,**kwargs):
        attempts[key]+=1;value=func(*args,**kwargs);work[key]+=1
        if key in WORK[:9]:retain(key,value)
        if key in ("endpoint_map","G_map"):retain("composed_components",composed_features.last_returned())
        elif key=="backward":retain(key,dict(grad_U=params[0].grad,grad_V=params[1].grad))
        elif key=="Adam":retain(key,dict(parameters=params,optimizer=adam.state_dict()))
        check();return value
    try:
        params=[p.detach().clone().requires_grad_() for p in initial_parameters]
        retain("supplied_packed_material",packed_material);adam=torch.optim.Adam(params,lr=options["lr"],eps=1e-12,foreach=False);check()
        if resume_state is not None:
            for p,saved in zip(params,resume_state["parameters"],strict=True):
                with torch.no_grad():p.copy_(saved.to(p))
            adam.load_state_dict(_cpu(resume_state["optimizer"]))
            E=_cpu(resume_state["current_evaluation"]);A=_cpu(resume_state["last_completed_update"]);best=_cpu(resume_state["earliest_best_evaluation"])
            snaps=_cpu(resume_state["snapshots"]);history=_cpu(resume_state["history"]);work=dict(resume_state["work"]);attempts=dict(resume_state["attempts"])
            file_pins=dict(resume_state["checkpoint_files_sha256"])
            for name,digest in file_pins.items():
                call("checkpoint_copies",shutil.copyfile,Path(resume_state["artifact_folder"])/"checkpoints"/name,folder/"checkpoints"/name)
                _files({str(folder/"checkpoints"/name):digest})
            M=call("packed_physical_forward",LowRankMoments.apply,*params,hard,packed_material,.05,options["chunk_size"])
            require(_tensor_identity(M.detach())==_tensor_identity(E["packed_moments"]),"Cached E1 packed tape differs");call("reattachment",lambda:None)
        else:
            E,M,context=evaluate_pair(0,params,hard,packed_material,source_owner,composed_features,options,pre_origin_contract,None,config,call,retain,check)
            history.append(evaluation_link(E));best=E
        while True:
            step=E["step"];validate_evaluation(E,config,context)
            if E["objective"]<best["objective"]:best=E
            if step in checkpoints and step not in snaps:
                name=f"step_{step:06d}.pt";file_pins[name]=call("checkpoint_writes",_write_state,E,folder/"checkpoints"/name,exclusive=True);snaps[step]=E
            if step in checkpoints and (resume_state is None or step!=1):
                csv_path=folder/"optimization.csv";temporary=csv_path.with_suffix(".tmp.csv")
                with temporary.open("x",newline="") as stream:
                    writer=csv.DictWriter(stream,fieldnames=list(history[0]));writer.writeheader();writer.writerows(history);stream.flush();os.fsync(stream.fileno())
                temporary.replace(csv_path)
                state=_cpu(dict(schema=1,trajectory_policy=POLICY,step=step,scientific_endpoint=25,config=config,context=context,
                    initial_parameters=initial_parameters,parameters=params,optimizer=adam.state_dict(),CE0=E["CE0"],current_evaluation=E,last_completed_update=A,
                    earliest_best_evaluation=best,snapshots=snaps,history=history,work=dict(work,resume_writes=work["resume_writes"]+1),
                    attempts=dict(attempts,resume_writes=attempts["resume_writes"]+1),checkpoint_files_sha256=file_pins,
                    history_sha256=hashlib.sha256(csv_path.read_bytes()).hexdigest(),artifact_folder=str(folder)))
                state["state_digest"]=_seal(state);validate_evaluated_state(state,config,context,folder);call("resume_writes",_write_state,state,folder/"resume.pt")
            if step==target:
                native.check_current_contract(pre_origin_contract)
                require(joint._runtime(params[0].device)==refs["runtime"] and source_owner.descriptor()==pre_origin_contract["feature_contract"]["source_outer"]
                    and composed_features.descriptor()==pre_origin_contract["feature_contract"]["composed"],"Runtime/source owner changed at exit");check();return state
            A=apply_pending_update(E,M,params,adam,composed_features,options,context,call,retain,check,A)
            history[-1].update(update_digest=A["record_digest"],update_after_parameters_digest=_seal(A["parameters_after"]),
                raw_adjoint_digest=_seal({f:A["folds"][f]["raw_adjoint"] for f in ("A","B")}))
            theta={f:E["folds"][f]["theta"].to(params[0].device) for f in ("A","B")}
            E,M,context=evaluate_pair(step+1,params,hard,packed_material,source_owner,composed_features,options,pre_origin_contract,context,config,call,retain,check,theta,A,E["CE0"])
            history.append(evaluation_link(E))
    except BaseException as error:
        failure=dict(schema=1,kind=TRAJECTORY,passed=False,error_type=type(error).__name__,error=str(error),seconds=time.monotonic()-started,
            work=work,attempts=attempts,inflight_operation_interiors="unknown",partial_state_is_not_successful_resume=True,no_retry_or_rescue=True)
        raw=folder/"failure_partial.pt";failure["raw_evidence"]=dict(path=str(raw),exists=False,bytes=None,sha256=None,hash_unknown=True)
        try:
            failure["raw_evidence"]["sha256"]=_write_state(_cpu(dict(context=context,current_evaluation=E,last_update=A,parameters=params,
                optimizer=None if adam is None else adam.state_dict(),history=history,work=work,attempts=attempts,returned=returned,
                last_composed_components=composed_features.last_returned())),raw,exclusive=True)
            failure["raw_evidence"]["hash_unknown"]=False
        except BaseException as observation:failure["raw_evidence"]["error"]=repr(observation)
        finally:
            failure["raw_evidence"]["exists"]=raw.exists()
            if raw.exists():failure["raw_evidence"]["bytes"]=raw.stat().st_size
        failure["partial_files"]=[dict(path=str(p),bytes=p.stat().st_size,hash_unknown=True) for p in folder.rglob("*") if p.is_file()]
        with (folder/"failure.json").open("x") as stream:
            json.dump(failure,stream,indent=2,allow_nan=False);stream.write("\n");stream.flush();os.fsync(stream.fileno())
        raise
