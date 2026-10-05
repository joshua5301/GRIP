"""New cold-once streaming trajectory: physical moments, one composed head."""
import csv
import hashlib
import os
from pathlib import Path
import shutil

import torch
from src import composed_centroid_joint_ce as joint
from src import fullwidth_coupled_row_backend as row
from src import Flickr446_fullwidth_composed_joint_ce_streaming as provider
from src.dual_head_ce import _validate_optimizer
from src.kernel_mean_ce import _cpu, _plain, _seal
from src.moments import augmented, decode_moments
from src.shared_features import _tensor_identity
from src.soft_ce_partition import solve_head_system, solve_inner_newton_first

TRAJECTORY="Flickr446_own_filebacked_composed_physical_cold_once_v1"
POLICY=dict(schema=1,kind=TRAJECTORY,head_evaluation="once_per_P",origin="own_streamed_M0_native_quotients",
    scale="own_positive_raw_CE0",Adam_eps=1e-12,foreach=False,continuation="cached_E1_plus_one_physical_reattachment")
WORK=("physical_forward","endpoint_map","G_map","head","outer","adjoint","complete_G","backward",
    "Adam","P_updates","reattachment","checkpoint_writes","resume_writes","checkpoint_copies")
require=provider.require


def sealed(value,budget,role):
    value=budget.own("sealed_"+role,value);return dict(value,record_digest=_seal(value))


def valid_record(value):
    require(value["record_digest"]==_seal({k:v for k,v in value.items() if k!="record_digest"}),"Owning record digest")


def linked(E):
    return dict(step=E["step"],raw_CE=E["raw_CE"],CE0=E["CE0"],objective=E["objective"],evaluation_digest=E["record_digest"],
        parameters_digest=_seal(E["parameters"]),theta_digest=_seal(E["theta"]),raw_rhs_digest=_seal(E["raw_rhs"]),
        theta_initial_digest=_seal(E["theta_initial"]),parent_update_digest=E["parent_update_digest"],
        update_digest=None,update_after_parameters_digest=None,raw_adjoint_digest=None)


def validate_evaluation(E,config,context):
    provider.validate_context(context);valid_record(E)
    require(E["kind"]=="Flickr446_own_streaming_composed_evaluation_v1" and type(E["step"]) is int and 0<=E["step"]<=25
        and E["context_digest"]==context["context_digest"] and E["config_digest"]==_seal(config),"Own streaming evaluation schema")
    for name,shape in (("physical_moments",(446,508)),("physical_centers",(446,500)),("physical_targets",(446,7)),
        ("head_features",(446,1012)),("theta",(7,1013)),("raw_rhs",(7,1013))):
        v=E[name];require(v.shape==shape and v.dtype==torch.float64 and v.device.type=="cpu" and bool(torch.isfinite(v).all()),"Evaluation finite fullwidth domain")
    mass=E["physical_mass"]
    require(mass.shape==(446,) and bool((mass>0).all()) and bool((E["physical_targets"]>=0).all())
        and bool((E["physical_targets"].sum(1)>0).all()) and bool((E["physical_targets"].sum(0)>0).all())
        and torch.equal(E["head_features"][:,:500],E["physical_centers"]),"Physical quotient general-Q/head prefix")
    require(E["CE0"]>0 and E["raw_CE"]>0 and E["objective"]==E["raw_CE"]/E["CE0"]
        and E["head_work"]["inner_converged"] is True and 0<=E["head_work"]["inner_grad_max"]<=config["inner_tol"],"Original stationary head/units")
    for v,shape in zip(E["parameters"],((44625,16),(446,16)),strict=True):
        require(v.shape==shape and v.dtype==torch.float32 and v.device.type=="cpu" and bool(torch.isfinite(v).all()),"Native owning endpoint factor domain")
    if E["step"]==0:
        own=dict(moments=_tensor_identity(E["physical_moments"]),centers=_tensor_identity(E["physical_centers"]),
            targets=_tensor_identity(E["physical_targets"]),mass=_tensor_identity(mass))
        require(own==context["native_origin"] and E["raw_CE"]==E["CE0"] and E["theta_initial"] is None
            and bool(E["parameters"][0].eq(0).all()),"New native streaming origin/CE0")


def validate_evaluated_state(state,expected_config,context=None,folder=None):
    require(state["schema"]==1 and state["trajectory_policy"]==POLICY
        and state["state_digest"]==_seal({k:v for k,v in state.items() if k!="state_digest"})
        and _seal(state["config"])==_seal(expected_config),"New streaming state/config/seal")
    provider.validate_context(state["context"],state["input_contract"])
    if context is not None:require(_seal(state["context"])==_seal(context),"Accepted context changed")
    step=state["step"];require(type(step) is int and 0<=step<=25 and len(state["history"])==step+1,"Endpoint/history")
    refs=dict(nodes=44625,cells=446,rank=16)
    require(_validate_optimizer(state["optimizer"],state["config"],refs)==step
        and set(state["optimizer"]["state"])==(set() if step==0 else {0,1}),"Original two Adam slots")
    require(set(state["work"])==set(state["attempts"])==set(WORK),"Count roles")
    w=state["work"]
    require(w["physical_forward"]==step+1+w["reattachment"] and all(w[k]==step+1 for k in ("head","endpoint_map","outer"))
        and all(w[k]==step for k in ("G_map","adjoint","complete_G","backward","Adam","P_updates"))
        and all(type(w[k]) is int and 0<=w[k]<=state["attempts"][k] for k in WORK),"Cold-once counts")
    require(set(state["snapshots"])==({0} if step==0 else {0,1} if step==1 else {0,1,25})
        and w["checkpoint_writes"]==len(state["snapshots"]) and w["resume_writes"]==len(state["snapshots"])
        and set(state["checkpoint_files_sha256"])=={f"step_{i:06d}.pt" for i in state["snapshots"]},"Exact selected snapshot/write closure")
    require(all(_tensor_identity(p)==state["input_contract"]["native_descriptors"][k]
        for p,k in zip(state["initial_parameters"],("U0","V0"),strict=True)),"Own initial factor descriptors")
    E=state["current_evaluation"];validate_evaluation(E,state["config"],state["context"])
    require(E["step"]==step and _seal(E["parameters"])==_seal(state["parameters"])
        and E["CE0"]==state["CE0"] and _seal(E)==_seal(state["snapshots"][step]),"Current accepted frontier")
    for i,h in enumerate(state["history"]):
        require(h["step"]==i and h["CE0"]==state["CE0"] and h["objective"]==h["raw_CE"]/h["CE0"]
            and (h["update_digest"] is None)==(i==step),"History scalar/frontier links")
        if i:require(h["theta_initial_digest"]==state["history"][i-1]["theta_digest"]
            and h["parent_update_digest"]==state["history"][i-1]["update_digest"]
            and h["parameters_digest"]==state["history"][i-1]["update_after_parameters_digest"],"Warm head/update chain")
    for i,S in state["snapshots"].items():
        validate_evaluation(S,state["config"],state["context"])
        require(S["step"]==i and all(linked(S)[k]==state["history"][i][k] for k in linked(S)
            if k not in ("update_digest","update_after_parameters_digest","raw_adjoint_digest")),"Preserved endpoint/head columns")
    A=state["last_completed_update"]
    if step:
        valid_record(A)
        require(A["kind"]=="Flickr446_own_streaming_composed_update_v1" and A["step"]==step-1
            and A["record_digest"]==state["history"][step-1]["update_digest"]
            and A["evaluation_digest"]==state["history"][step-1]["evaluation_digest"]
            and _seal(A["parameters_after"])==_seal(state["parameters"])
            and _seal(A["optimizer_after"])==_seal(state["optimizer"])
            and A["adjoint_work"]["cg_converged"] is True,"Accepted update slots/adjoint")
    else:require(A is None,"No origin update")
    if folder is not None:
        p=Path(folder)
        require(row.file_sha(p/"optimization.csv")==state["history_sha256"],"Accepted CSV immutable")
        require(all(row.file_sha(p/"checkpoints"/n)==h for n,h in state["checkpoint_files_sha256"].items()),"Accepted checkpoint bytes")
    return True


def optimize_streaming_composed_joint_mean(source_owner,hard,initial_parameters,composed,options,input_contract,folder,target_step,
        resume_state=None,stop=lambda:False):
    provider.validate_input_contract(input_contract);require(type(source_owner) is provider.ObservedSource,"Exact supplied streaming owner")
    require(source_owner.descriptor()==input_contract["source_owner"] and composed.descriptor()==input_contract["composed_owner"]
        and options==input_contract["original_options"],"Same supplied source/map/options")
    require(target_step in (1,25) and (resume_state is None and target_step==1 or resume_state is not None and target_step==25),"Only cold1 or guarded accepted1→25")
    config=dict(options,mode=provider.MODE,trajectory_policy=POLICY,scientific_endpoint=25,input_digest=input_contract["input_digest"])
    folder=Path(folder);folder.mkdir(exist_ok=False);(folder/"checkpoints").mkdir()
    budget=source_owner.budget;work=dict.fromkeys(WORK,0);attempts=dict.fromkeys(WORK,0);returned={};E=A=M=context=state=None
    history=[];snapshots={};file_pins={};best=None;previous=None;params=[];adam=None
    def live():
        budget.retained["live_engine"]=dict(E=E,A=A,M=M,context=context,state=state,snapshots=snapshots,best=best,
            parameters=params,optimizer=None if adam is None else adam.state_dict(),returned=returned,resume_input=resume_state,previous_update=previous)
    def check():
        live();require(not stop(),"Stopped streaming trajectory");budget.charge("core_check",returned)
    def retain(key,value):
        returned[key]=value;live();returned[key]=budget.own("core_"+key,value)
    def call(key,f,*args,**kwargs):
        def invoke(phase=None):
            attempts[key]+=1;value=f(*args,**kwargs);work[key]+=1
            if phase is not None:phase["returned"]=True
            if key in ("physical_forward","endpoint_map","G_map","head","outer","adjoint","complete_G","backward"):
                retain(key,value)
            if key in ("endpoint_map","G_map"):retain("composed_components",composed.last_returned())
            if key=="head":
                require(value["inner_converged"] is True and 0<=value["inner_grad_max"]<=options["inner_tol"],"Original head stationarity")
            if key=="adjoint":
                require(value[1]["cg_converged"] is True and bool(torch.isfinite(value[0]).all()),"Original raw adjoint")
            check();return value
        if key in ("head","adjoint"):
            live()
            with budget.phase_reservation("core_"+key+"_"+str(work[key])) as phase:return invoke(phase)
        return invoke()
    def evaluate(step,theta_initial=None,CE0=None):
        nonlocal context
        source_owner.phase="core";M,evidence=call("physical_forward",row.physical_moments,*params,hard,source_owner,.05,provider.PARTITION)
        centers,targets,mass=decode_moments(M,500);retain("native_quotients",(centers,targets,mass))
        require(all(bool(torch.isfinite(v).all()) for v in (centers,targets,mass)) and bool((mass>0).all())
            and bool((targets>=0).all()) and bool((targets.sum(1)>0).all()) and bool((targets.sum(0)>0).all()),"Native physical quotient general-Q domain")
        if context is None:context=provider.build_streaming_context(input_contract,M,(centers,targets,mass))
        features=call("endpoint_map",composed,centers)
        fit=call("head",solve_inner_newton_first,features,targets,torch.full_like(mass,1/446),options["penalty"],theta_initial,
            options["inner_max_iter"],options["inner_tol"],cg_max_iter=options["cg_max_iter"],cg_check_interval=options["cg_check_interval"])
        head=_plain({k:v for k,v in fit.items() if k!="theta"});theta=fit["theta"].detach()
        require(head["inner_converged"] is True and 0<=head["inner_grad_max"]<=options["inner_tol"],"Original head stationarity")
        CE,rhs,outer_evidence=call("outer",row.source_ce_rhs,theta,source_owner,provider.PARTITION)
        CE=float(CE);CE0=CE if CE0 is None else CE0
        result=sealed(dict(schema=1,kind="Flickr446_own_streaming_composed_evaluation_v1",step=step,config_digest=_seal(config),
            context_digest=context["context_digest"],parameters=params,physical_moments=M,physical_centers=centers,
            physical_targets=targets,physical_mass=mass,head_features=features,theta=theta,theta_initial=theta_initial,
            parent_update_digest=None if A is None else A["record_digest"],raw_CE=CE,raw_rhs=rhs,CE0=CE0,objective=CE/CE0,head_work=head),budget,"evaluation")
        retain("evaluation",result);validate_evaluation(result,config,context);return result,M
    try:
        params=[p.detach().clone().requires_grad_() for p in initial_parameters]
        adam=torch.optim.Adam(params,lr=.01,eps=1e-12,foreach=False)
        budget.retained["core_initial_parameters"]=initial_parameters
        if resume_state is not None:
            validate_evaluated_state(resume_state,config,folder=resume_state["artifact_folder"])
            require(resume_state["step"]==1 and _seal(resume_state["input_contract"])==_seal(input_contract),"Only same-source accepted E1")
            context=resume_state["context"];E=resume_state["current_evaluation"];A=resume_state["last_completed_update"]
            for p,s in zip(params,resume_state["parameters"],strict=True):
                with torch.no_grad():p.copy_(s.to(p))
            adam.load_state_dict(_cpu(resume_state["optimizer"]))
            history=[dict(h) for h in resume_state["history"]];snapshots=dict(resume_state["snapshots"])
            work=dict(resume_state["work"]);attempts=dict(resume_state["attempts"]);best=resume_state["earliest_best_evaluation"]
            for name,digest in resume_state["checkpoint_files_sha256"].items():
                call("checkpoint_copies",shutil.copyfile,Path(resume_state["artifact_folder"])/"checkpoints"/name,folder/"checkpoints"/name)
                require(row.file_sha(folder/"checkpoints"/name)==digest,"Copied old checkpoint");file_pins[name]=digest
            source_owner.phase="core";M,_=call("physical_forward",row.physical_moments,*params,hard,source_owner,.05,provider.PARTITION)
            require(_tensor_identity(M)==_tensor_identity(E["physical_moments"]),"Exact accepted E1 streamed reattachment")
            call("reattachment",lambda:None)
        else:E,M=evaluate(0);history.append(linked(E));best=E
        while True:
            step=E["step"]
            if E["objective"]<best["objective"]:best=E
            if step in (0,1,25) and step not in snapshots:
                path=folder/"checkpoints"/f"step_{step:06d}.pt"
                meta=call("checkpoint_writes",budget.write,E,path,exclusive=True);file_pins[path.name]=meta["sha256"];snapshots[step]=E
            if step in (0,1,25) and (resume_state is None or step!=1):
                csvpath=folder/"optimization.csv";tmp=csvpath.with_suffix(".tmp.csv")
                with tmp.open("x",newline="") as stream:
                    writer=csv.DictWriter(stream,fieldnames=list(history[0]));writer.writeheader();writer.writerows(history)
                    stream.flush();os.fsync(stream.fileno())
                tmp.replace(csvpath)
                live();state=budget.own("generated_state",dict(schema=1,trajectory_policy=POLICY,step=step,config=config,input_contract=input_contract,context=context,
                    initial_parameters=initial_parameters,parameters=params,optimizer=adam.state_dict(),CE0=E["CE0"],current_evaluation=E,
                    last_completed_update=A,earliest_best_evaluation=best,snapshots=snapshots,history=history,
                    work=dict(work,resume_writes=work["resume_writes"]+1),attempts=dict(attempts,resume_writes=attempts["resume_writes"]+1),
                    checkpoint_files_sha256=file_pins,history_sha256=row.file_sha(csvpath),artifact_folder=str(folder)))
                state["state_digest"]=_seal(state);retain("state",state);validate_evaluated_state(state,config,folder=folder)
                call("resume_writes",budget.write,state,folder/"resume.pt")
            if step==target_step:check();return state
            device=params[0].device;features,targets,mass=(E[k].to(device) for k in ("head_features","physical_targets","physical_mass"))
            theta,rhs=E["theta"].to(device),E["raw_rhs"].to(device)
            previous=A;warm=None if A is None else A["raw_adjoint"].to(device)
            vector,diag=call("adjoint",solve_head_system,augmented(features),targets,torch.full_like(mass,1/446),theta,options["penalty"],rhs,
                rtol=options["cg_rtol"],atol=1e-12,max_iter=options["cg_max_iter"],initial=warm,cg_check_interval=1)
            vector=vector.detach();require(diag["cg_converged"] is True and bool(torch.isfinite(vector).all()),"Original raw adjoint")
            G=call("complete_G",joint.complete_moment_cotangent,M,joint.ComposedJointLayout(500,512,7),composed,theta,vector,options["penalty"],E["CE0"],
                map_call=lambda f,x:call("G_map",f,x))
            source_owner.phase="core";du,dv,evidence=call("backward",row.physical_factor_adjoint,*params,hard,source_owner,G,.05,provider.PARTITION,True)
            adam.zero_grad(set_to_none=True);params[0].grad=du.detach().clone();params[1].grad=dv.detach().clone()
            call("Adam",adam.step);call("P_updates",lambda:None)
            A=sealed(dict(schema=1,kind="Flickr446_own_streaming_composed_update_v1",step=step,evaluation_digest=E["record_digest"],
                previous_update_digest=None if previous is None else previous["record_digest"],raw_adjoint_initial_digest=_seal(warm),
                raw_adjoint=vector,adjoint_work=_plain(diag),normalized_G=G,native_grad_U=du,native_grad_V=dv,
                parameters_after=params,optimizer_after=adam.state_dict()),budget,"update")
            retain("update",A)
            history[-1].update(update_digest=A["record_digest"],update_after_parameters_digest=_seal(A["parameters_after"]),raw_adjoint_digest=_seal(A["raw_adjoint"]))
            E,M=evaluate(step+1,theta,E["CE0"]);history.append(linked(E))
    except BaseException:
        partial=dict(E=E,A=A,context=context,parameters=params,optimizer=None if adam is None else adam.state_dict(),
            history=history,work=work,attempts=attempts,returned=returned,last_operator=source_owner.last_evidence)
        try:budget.write(budget.own("failed_core",partial),folder/"failure_partial.pt",exclusive=True)
        except BaseException:pass
        raise
