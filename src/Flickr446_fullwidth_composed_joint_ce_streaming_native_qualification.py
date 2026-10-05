"""Bounded first native qualifier; later continuation requires real immutable admission."""
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import resource
import signal
import time

KIND="Flickr446_original_fullwidth_composed_streaming_native_qualification_and_continuation_v1"
SCIENCE_SHA="49d6b339552b42bd6633849fa12fdce8cbacff575e605c73cb93552dd5309f53"


def require(ok,message):
    if not ok:raise ValueError(message)


def read(ref):
    return json.loads(Path(ref["path"]).read_text())


def sha(path,check=lambda:None):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(4194304),b""):check();h.update(block)
    return h.hexdigest()


def refs(value,result=None):
    result={} if result is None else result
    if isinstance(value,dict):
        if set(value)=={"path","sha256"}:
            p,h=value["path"],value["sha256"]
            require(Path(p).is_absolute() and str(Path(p).resolve())==p and len(h)==64,"Normalized ABS ref")
            require(p not in result or result[p]==h,"Conflicting ref");result[p]=h
        for v in value.values():refs(v,result)
    elif isinstance(value,list):
        for v in value:refs(v,result)
    return result


def run(protocol_path,protocol_sha256,budget,mode,stop=lambda:False):
    started=time.monotonic();require(Path(protocol_path).is_absolute() and str(Path(protocol_path).resolve())==protocol_path
        and sha(protocol_path)==protocol_sha256,"Exact protocol")
    packet=json.loads(Path(protocol_path).read_text());science=read(packet["scientific_contract"])
    require(packet["scientific_contract"]["sha256"]==sha(packet["scientific_contract"]["path"])==SCIENCE_SHA
        and packet["kind"]==science["kind"]==KIND and budget=="flickr446" and mode in ("qualify","continue")
        and packet["authorized_mode"]==mode,"Frozen science/budget/mode")
    out=Path(packet["output_folders"][mode]);require(out.is_absolute() and str(out.resolve())==str(out),"Normalized ABS output")
    out.mkdir(parents=True,exist_ok=False)
    report_path=out/"native_admission_report.json";arrays_path=out/"native_admission_arrays.pt"
    report=dict(schema=1,kind=KIND,budget=budget,mode=mode,passed=False,completed=False,source=packet["source"],
        scientific_contract=packet["scientific_contract"],protocol=dict(path=protocol_path,sha256=protocol_sha256),gates=[],failure=None,
        work=dict(PT_load_attempts=0,PT_loads=0,component_owner_attempts=0,component_owners=0,core_attempts=0,core_returns=0,
            external_adjoint_attempts=0,external_adjoint=0,external_G_attempts=0,external_G=0,external_G_map=0,
            diagnostic_forward_attempts=0,diagnostic_forward=0,diagnostic_pullback_attempts=0,diagnostic_pullback=0,
            CPU_reference_attempts=0,CPU_reference=0,RNG=0,factor_factory=0,Q_decode=0,RMS_fit=0,map_fit=0,Phi_rebuild=0,
            SGC=0,external_head=0,external_outer=0,external_Adam=0,external_P=0,student=0,test=0),raw_evidence={})
    torch=owner=storage=None;saved={};oldhandler=signal.getsignal(signal.SIGALRM);oldtimer=signal.getitimer(signal.ITIMER_REAL)
    def capture(key,value,role=None):
        saved[key]=value
        storage.retained["live_wrapper_evidence"]=saved
        saved[key]=storage.own(role or key,value);return saved[key]
    def peaks():
        d=dict(seconds=time.monotonic()-started,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            peak_allocated_bytes=0,peak_reserved_bytes=0)
        if torch is not None and torch.cuda.is_initialized():
            d.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(0),peak_reserved_bytes=torch.cuda.max_memory_reserved(0))
        return d
    def check():
        require(not stop() and all(v<=science["resources"][k] for k,v in peaks().items()),"Frozen300s/RSS/CUDA boundary")
    def gate(name,ok,**detail):
        report["gates"].append(dict(name=name,passed=bool(ok),**detail));require(ok,name)
    def pins():
        expected=packet["readonly_files_sha256"]
        require(all(Path(p).is_absolute() and str(Path(p).resolve())==p for p in expected),"Normalized pin map")
        require(all(expected.get(p)==h for p,h in refs(dict(packet=packet,science=science)).items()),"Required readonly ref omitted")
        return {p:sha(p,check) for p in expected}
    def alarm(signum,frame):raise TimeoutError("Frozen300s deadline")
    def write_report():
        with report_path.open("w") as stream:json.dump(report,stream,indent=2,allow_nan=False);stream.write("\n");stream.flush();os.fsync(stream.fileno())
    try:
        signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,max(.000001,300-(time.monotonic()-started)))
        report["readonly_entry"]=pins();gate("readonly_entry_exact",report["readonly_entry"]==packet["readonly_files_sha256"])
        gate("own_entrypoint",Path(packet["entrypoint"]["path"]).resolve()==Path(__file__).resolve())
        import numpy as np
        import torch
        from src import Flickr446_fullwidth_composed_joint_ce_streaming as provider, Flickr446_fullwidth_composed_joint_ce_streaming_trajectory as engine
        from src import composed_centroid_joint_ce as joint, fullwidth_coupled_row_backend as row
        from src.kernel_mean_ce import _cpu,_plain,_seal
        from src.dual_head_ce import _transform,_map
        from src.transforms import FeatureTransform
        from src.nystrom_ce import NystromMap,_map_token
        from src.moments import augmented
        from src.soft_ce_partition import solve_head_system
        from src.shared_features import _tensor_identity
        from src.research_loop import implementation_provenance
        gate("production_modules_exact",all(Path(m.__file__).resolve()==Path(packet[k]["path"]).resolve()
            and sha(m.__file__)==packet[k]["sha256"] for m,k in ((provider,"provider_entrypoint"),(engine,"engine_entrypoint"))))
        report["source_entry"]=implementation_provenance();gate("current_source_Git",report["source_entry"]==packet["source"])
        bridge=read(packet["prerequisite_admission"])
        gate("ROOT_static_bridge",bridge["passed"] is True and bridge["source"]==packet["source"]
            and all(bridge[k]==packet[k] for k in ("entrypoint","provider_entrypoint","engine_entrypoint","scientific_contract")))
        torch.set_num_threads(4)
        if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
        torch.set_default_dtype(torch.float32);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision("highest");torch.use_deterministic_algorithms(True)
        gate("environment_exact",{k:os.environ.get(k) for k in science["runtime"]["environment"]}==science["runtime"]["environment"])
        gate("pre_CUDA_versions",platform.python_version()=="3.12.7" and str(torch.__version__)=="2.9.1+cu128" and np.__version__=="1.26.4")
        torch.cuda.init();device=torch.device("cuda:0")
        report["runtime"]=dict(joint._runtime(device),interop_threads=torch.get_num_interop_threads(),GPU=torch.cuda.get_device_name(0))
        expected={k:v for k,v in science["runtime"].items() if k!="environment"}
        gate("runtime_exact",report["runtime"]==expected and not torch.is_autocast_enabled("cpu") and not torch.is_autocast_enabled("cuda"))
        req=science["required_refs"];coordinate=read(req["source122_report"]);cap=read(req["K446_report"]);manifest=read(req["FE_manifest"])
        gate("own128_native_actual_admitted",cap["passed"] is True and cap["budget"]==budget and cap["source"]==science["suppliers"]["native_producer"]
            and cap["new_data_digest"]==science["suppliers"]["own_native_data_digest"]
            and all(read(req[k])["passed"] is True and read(req[k])["actual_report"]==req["K446_report"] and read(req[k])["source"]==cap["source"] and read(req[k])["new_data_digest"]==cap["new_data_digest"] for k in ("K446_ROOT","K446_independent"))
            and read(req["K446_ROOT"])["actual_arrays"]==req["K446_arrays"]
            and read(req["K446_independent"])["actual_arrays"]==req["K446_arrays"] and read(req["K446_independent"])["array_descriptors"]==cap["array_descriptors"])
        gate("shared122_coordinate_Q_actual_admitted",coordinate["passed"] is True and coordinate["source"]==science["suppliers"]["coordinate_Q_producer"]
            and all(read(req[k])["passed"] is True and read(req[k])["actual_report"]==req["source122_report"] for k in ("source122_ROOT","source122_independent"))
            and cap["reused_coordinate_Q_producer"]==coordinate["source"] and cap["transform_descriptors"]==coordinate["transform_descriptors"]
            and cap["original_Phi_identity"]==coordinate["original_Phi_identity"]
            and all(cap["array_descriptors"][k]==manifest["components"][k]["original_Torch_descriptor"] for k in ("z","Q")))
        gate("FE_actual_components_admitted",all(read(req[k])["passed"] is True and read(req[k])["actual_report"]==req["FE_actual_report"]
            and read(req[k])["source_component_manifest"]==req["FE_manifest"] for k in ("FE_actual_ROOT","FE_actual_independent","FE_combined_admission"))
            and manifest["source122"]["source"]==coordinate["source"] and manifest["partition"]==provider.PARTITION)
        gate("original_Phi_sidecar",read(req["original_asset_Phi_sidecar"])==science["suppliers"]["original_Phi_identity"]==manifest["original_Phi"]["identity"])
        storage=provider.ActiveStorageBudget(check=check)
        report["work"]["PT_load_attempts"]+=1
        mapped=torch.load(req["K446_arrays"]["path"],map_location="cpu",weights_only=False,mmap=True)
        report["work"]["PT_loads"]+=1
        small=storage.own("loaded_small_native",{k:mapped[k] for k in ("hard","U0","V0")})
        td=storage.own("loaded_transform",mapped["transform"]);md=storage.own("loaded_map",mapped["map_state"])
        gate("selected_native_original_bytes",{k:_tensor_identity(v) for k,v in small.items()}==science["suppliers"]["selected_native_descriptors"])
        require(set(td)=={"center","matrix","output_center","scale","kind","eps"} and set(md)=={"anchors","mapping","kernel"},"Literal original RMS/map states")
        tc=FeatureTransform(**td);mc=NystromMap(**md)
        gate("original_RMS_and_map_bytes",_transform(tc,500,torch.device("cpu"))==cap["transform_descriptors"]
            and hashlib.sha256(json.dumps(_map_token(mc,2048,stop),sort_keys=True).encode()).hexdigest()==manifest["original_Phi"]["identity"]["map_digest"])
        del mapped
        transform=FeatureTransform(**{k:v.to(device) if torch.is_tensor(v) else v for k,v in td.items()})
        fmap=NystromMap(md["anchors"].to(device),md["mapping"].to(device),kernel="relu")
        initial=[small[k].to(device) for k in ("U0","V0")];hard=small["hard"].to(device)
        admissions={k:req[k] for k in ("source122_report","source122_arrays","source122_ROOT","source122_independent",
            "FE_actual_report","FE_manifest","FE_actual_ROOT","FE_actual_independent","FE_combined_admission",
            "K446_report","K446_arrays","K446_ROOT","K446_independent")}
        report["work"]["component_owner_attempts"]+=1
        input_contract,owner,composed=provider.build_streaming_inputs(manifest,admissions,dict(hard=hard,U0=initial[0],V0=initial[1]),
            (transform,fmap),row.CoupledTileLayout(**provider.LAYOUT),provider.PARTITION,packet["source"],science["original_options"],
            budget=storage,witness_folder=out/"witnesses",source_refs=dict(source122_data_digest=science["suppliers"]["source122_data_digest"],own_native_data_digest=cap["new_data_digest"],admissions=admissions))
        report["work"]["component_owners"]+=1;report["input_contract"]=input_contract
        gate("initial_component_verification66",all(p.work["verifications"]==1 and p.work["hash_rows"]==44625 for p in (owner.z,owner.Q,owner.Phi)))
        old=None;oldseal=None;oldfiles={}
        if mode=="continue":
            authorization=read(packet["continuation_authorization"]);accepted=packet["accepted_qualifier"]
            gate("separate_ROOT_continuation_authorization",authorization["passed"] is True and authorization["mode"]=="continue"
                and authorization["source"]==packet["source"] and authorization["scientific_contract"]==packet["scientific_contract"]
                and authorization["accepted_qualifier"]==accepted and packet["target_step"]==25)
            for role in ("ROOT","independent"):
                a=read(accepted[role]);gate(role+"_actual_qualifier_accepted",a["passed"] is True and a["budgets"][budget]["report"]==accepted["report"]
                    and a["budgets"][budget]["state"]==accepted["state"])
            accepted_report=read(accepted["report"])
            gate("actual_E1_unchanged_source_science",accepted_report["passed"] is True and accepted_report["mode"]=="qualify"
                and accepted_report["source"]==packet["source"] and accepted_report["input_contract"]==input_contract)
            report["work"]["PT_load_attempts"]+=1;old=torch.load(accepted["state"]["path"],map_location="cpu",weights_only=False)
            report["work"]["PT_loads"]+=1;storage.retained["accepted_input_state"]=old;oldseal=_seal(old)
            report["accepted_input_state_seal_before"]=oldseal
            oldfiles={str(Path(old["artifact_folder"])/"optimization.csv"):old["history_sha256"],
                **{str(Path(old["artifact_folder"])/"checkpoints"/n):h for n,h in old["checkpoint_files_sha256"].items()}}
        else:gate("FIRST_cold1_only",packet["target_step"]==1 and packet["accepted_qualifier"] is None and packet["continuation_authorization"] is None)
        storage.retained["live_wrapper"]=dict(saved=saved,accepted_input=old,small=small,transform=td,map=md)
        report["work"]["core_attempts"]+=1
        state=engine.optimize_streaming_composed_joint_mean(owner,hard,initial,composed,science["original_options"],input_contract,
            out/"core",packet["target_step"],resume_state=old,stop=stop)
        report["work"]["core_returns"]+=1;saved["core_state"]=state;storage.retained["wrapper_state"]=state
        report["core_work"]=state["work"];report["core_attempts"]=state["attempts"];report["context"]=state["context"]
        engine.validate_evaluated_state(state,state["config"],folder=state["artifact_folder"])
        E=state["current_evaluation"];report["endpoint"]=_plain(E);report["last_update"]=_plain(state["last_completed_update"])
        corepins={str(out/"core"/"resume.pt"):sha(out/"core"/"resume.pt"),str(out/"core"/"optimization.csv"):state["history_sha256"],
            **{str(out/"core"/"checkpoints"/n):h for n,h in state["checkpoint_files_sha256"].items()}}
        report["state_ref"]=dict(path=str(out/"core"/"resume.pt"),sha256=corepins[str(out/"core"/"resume.pt")]);report["core_artifacts"]=corepins
        end=1 if mode=="qualify" else 25;w=state["work"]
        gate("exact_core_cold_once_counts",w["physical_forward"]==end+1+(0 if mode=="qualify" else 1)
            and all(w[k]==end+1 for k in ("head","endpoint_map","outer"))
            and all(w[k]==end for k in ("G_map","adjoint","complete_G","backward","Adam","P_updates"))
            and w["checkpoint_writes"]==w["resume_writes"]==(2 if mode=="qualify" else 3)
            and w["checkpoint_copies"]==(0 if mode=="qualify" else 2))
        coreseal=_seal(state);report["core_state_seal_before_diagnostics"]=coreseal
        if mode=="qualify":
            features,targets,mass=(E[k].to(device) for k in ("head_features","physical_targets","physical_mass"))
            theta,rhs=E["theta"].to(device),E["raw_rhs"].to(device)
            report["work"]["external_adjoint_attempts"]+=1
            storage.retained["live_wrapper"].update(state=state,saved=saved,external_CPU_evaluation=E)
            with storage.phase_reservation("external_E1_raw_adjoint") as phase:
                vector,diag=solve_head_system(augmented(features),targets,torch.full_like(mass,1/446),theta,1e-4,rhs,
                    rtol=1e-6,atol=1e-12,max_iter=512,initial=state["last_completed_update"]["raw_adjoint"].to(device),cg_check_interval=1)
                phase["returned"]=True
                report["work"]["external_adjoint"]+=1;capture("external_adjoint",dict(vector=vector,diagnostic=diag))
                gate("external_raw_adjoint",diag["cg_converged"] is True and bool(torch.isfinite(vector).all()))
            def mapped_call(f,x):
                value=f(x);report["work"]["external_G_map"]+=1;capture("external_map",composed.last_returned());return value
            report["work"]["external_G_attempts"]+=1
            G=joint.complete_moment_cotangent(E["physical_moments"].to(device),joint.ComposedJointLayout(500,512,7),composed,
                theta,vector.detach(),1e-4,E["CE0"],map_call=mapped_call)
            report["work"]["external_G"]+=1;capture("held_complete_G",G,"held_G")
            params=[p.to(device).detach().clone() for p in E["parameters"]];heldG=G.detach().clone();storage.retained["diagnostic_params"]=(params,heldG)
            gate("legitimate_nonzero_E1_factors",all(bool(torch.isfinite(p).all()) and float(p.norm())>0 for p in params))
            records=[];first_grad=None
            for i in range(3):
                owner.phase="diagnostic";report["work"]["diagnostic_forward_attempts"]+=1
                M,ev=row.physical_moments(*params,hard,owner,.05,provider.PARTITION);report["work"]["diagnostic_forward"]+=1
                returned=capture("repeat_M_"+str(i),dict(M=M,evidence=ev))
                report["work"]["diagnostic_pullback_attempts"]+=1
                du,dv,ev=row.physical_factor_adjoint(*params,hard,owner,heldG,.05,provider.PARTITION,True)
                report["work"]["diagnostic_pullback"]+=1;capture("repeat_gradient_"+str(i),dict(U=du,V=dv,evidence=ev),"repeat_grad_"+str(i))
                record=dict(M=_tensor_identity(M),U=_tensor_identity(du),V=_tensor_identity(dv));records.append(record)
                gate("repeat_"+str(i)+"_finite_nonzero",all(bool(torch.isfinite(v).all()) and float(v.norm())>0 for v in (du,dv)))
                gate("repeat_"+str(i)+"_own_E1_M",record["M"]==_tensor_identity(E["physical_moments"]))
                if first_grad is None:first_grad=storage.own("first_gradient",(du,dv))
            report["same_G_repeats"]=records;gate("three_byte_equal_repeats",records[0]==records[1]==records[2])
            report["work"]["CPU_reference_attempts"]+=1
            cpu=storage.own("CPU_reference_inputs",dict(parameters=params,G=heldG,hard=hard))
            u,v=cpu["parameters"];gh=cpu["G"];hh=cpu["hard"]
            refu,refv=torch.zeros_like(u),torch.zeros_like(v);storage.retained["CPU_reference_live"]=(u,v,gh,hh,refu,refv)
            owner.phase="CPU_reference"
            for start,end in owner.blocks(provider.PARTITION):
                s=owner.physical_block(start,end,"cpu")
                prior=torch.full((end-start,446),float(np.log(.05/446)),dtype=torch.float32)
                prior.scatter_(1,hh[start:end,None],float(np.log(1-.05+.05/446)))
                probability=(prior+u[start:end]@v.T/math.sqrt(16)).double().softmax(1)
                direction=s@gh.T/44625;whole=probability*(direction-(probability*direction).sum(1,keepdim=True))
                cast=whole.float()/math.sqrt(16);bu=cast@v;bv=cast.T@u[start:end]
                refu[start:end]=bu;refv+=bv
                storage.retained["CPU_reference_row_live"]=(s,prior,probability,direction,whole,cast,bu,bv)
                owner.witness("reference",start,end,material=s,probability=probability,R=direction,whole_W=whole,cast_scaled_W=cast,du=bu,dv_contribution=bv)
                del s,prior,probability,direction,whole,cast,bu,bv
                storage.retained["CPU_reference_row_live"]=None  # Original scoped row locals are now deleted.
            report["work"]["CPU_reference"]+=1;capture("CPU_reference",dict(U=refu,V=refv))
            errors={}
            storage.charge("before_CPU_comparison_pairs",future_cpu_bytes=sum(v.numel()*v.element_size() for v in (*first_grad,refu,refv)))
            comparison_pairs=(("U",first_grad[0],refu),("V",first_grad[1],refv),("full",torch.cat((first_grad[0].flatten(),first_grad[1].flatten())),torch.cat((refu.flatten(),refv.flatten()))))
            storage.retained["CPU_comparison_pairs"]=comparison_pairs
            for name,a,b in comparison_pairs:
                delta=(a.double()-b.double())
                storage.retained["CPU_comparison_live"]=(a,b,delta);storage.charge("CPU_comparison_"+name,(a,b,delta))
                absolute=float(delta.abs().max());den=float(b.double().norm());num=float(delta.norm())
                relative=num/den if den>0 else (0.0 if num==0 else None)
                finite=math.isfinite(absolute) and math.isfinite(den) and math.isfinite(num) and relative is not None and math.isfinite(relative)
                errors[name]=dict(absmax=absolute if math.isfinite(absolute) else None,
                    relativeL2=relative if relative is not None and math.isfinite(relative) else None,nonfinite_diagnostic=not finite)
                gate(name+"_BOTH_gradient_bounds",finite and absolute<=5e-7 and relative<=2e-5,**errors[name])
            report["gradient_errors"]=errors
        else:
            report["accepted_input_state_seal_after"]=_seal(old)
            gate("accepted_input_whole_state_unchanged",_seal(old)==oldseal and all(sha(p)==h for p,h in oldfiles.items()))
            gate("cached_E0_E1_head_records_unchanged",all(_seal(state["snapshots"][i])==_seal(old["snapshots"][i]) for i in (0,1)))
        report["core_state_seal_after_diagnostics"]=_seal(state)
        gate("core_state_and_files_unchanged_by_diagnostics",_seal(state)==coreseal and all(sha(p)==h for p,h in corepins.items()))
        expectedw=264 if mode=="qualify" else 1606
        report["witnesses"]=owner.witness_records;gate("exact_witness_files",owner.witness_attempts==owner.witness_returns==len(owner.witness_records)==expectedw
            and all(r["write_completed"] and not r["hash_unknown"] for r in owner.witness_records))
        report["operator_work"]=owner.work;report["component_work"]={k:v.work for k,v in (("z",owner.z),("Q",owner.Q),("Phi",owner.Phi))}
        forward,pullback,outer=(5,4,2) if mode=="qualify" else (25,24,24)
        physical_blocks=(forward+pullback+(1 if mode=="qualify" else 0))*22
        operator_expected=dict(outer_blocks=outer*22,physical_blocks=physical_blocks,moment_attempts=forward,moments=forward,
            adjoint_attempts=pullback,adjoints=pullback,source_attempts=outer,source_returns=outer)
        gate("exact_operator_counts",owner.work==operator_expected)
        read_counts=dict(z=physical_blocks+outer*22,Q=physical_blocks+outer*22,Phi=outer*22)
        gate("exact_component_reads",all(report["component_work"][k]["read_attempts"]==report["component_work"][k]["reads"]==n
            and report["component_work"][k]["rows_read"]==n//22*44625 for k,n in read_counts.items()))
        expected_diag=dict(external_adjoint=1,external_G=1,external_G_map=1,diagnostic_forward=3,diagnostic_pullback=3,CPU_reference=1)
        gate("exact_external_diagnostic_counts",all(report["work"][k]==(v if mode=="qualify" else 0) for k,v in expected_diag.items())
            and report["work"]["PT_loads"]==report["work"]["PT_load_attempts"]==(1 if mode=="qualify" else 2)
            and report["work"]["core_attempts"]==report["work"]["core_returns"]==report["work"]["component_owners"]==1)
        report["row_pass_accounting"]=dict(component_verification_blocks=66,operator_witness_bundles=expectedw,
            component_owning_read_blocks=read_counts,component_verification_byte_hash_passes=6)
        report["witness_role_counts"]={label:sum(r["phase"]+"_"+r["role"]==label for r in owner.witness_records)
            for label in ("core_forward","core_adjoint","core_source","diagnostic_forward","diagnostic_adjoint","CPU_reference_reference")}
        expected_roles=dict(core_forward=44,core_adjoint=22,core_source=44,diagnostic_forward=66,diagnostic_adjoint=66,CPU_reference_reference=22)
        if mode=="continue":expected_roles=dict(core_forward=550,core_adjoint=528,core_source=528,diagnostic_forward=0,diagnostic_adjoint=0,CPU_reference_reference=0)
        gate("exact_witness_roles",report["witness_role_counts"]==expected_roles)
        report["solver_phase_observations"]=storage.solver_phases
        gate("fixed_solver_phase_policy",len(storage.solver_phases)==(4 if mode=="qualify" else 48)
            and storage.reserve==64*1024**2 and all(p["entry_passed"] and p["returned"] and p["completed"] and p["restored64"]
                and p["managed_plus_opaque_bytes"]<=768*1024**2 and p["observed_active_upper_bytes"]<=1536*1024**2 for p in storage.solver_phases))
        saved["streaming_context"]=state["context"];report["completed"]=True;report["passed"]=True
    except BaseException as error:
        report["failure"]=dict(error_type=type(error).__name__,error=str(error),no_retry=True,unknown_inflight_interiors=True)
    finally:
        if storage is not None:
            if (out/"core").exists():
                report["preserved_core_output_files"]={str(p):dict(exists=True,bytes=p.stat().st_size,sha256=None,hash_unknown=True)
                    for p in (out/"core").rglob("*") if p.is_file()}
            if owner is not None:
                report["witnesses"]=owner.witness_records;report["witness_attempts"]=owner.witness_attempts;report["witness_returns"]=owner.witness_returns
                saved["last_operator_evidence"]=owner.last_evidence
            try:
                storage.retained["live_wrapper_evidence"]=saved
                if owner is not None:
                    saved["last_operator_evidence"]=storage.own("last_operator_evidence",saved["last_operator_evidence"])
                storage.retained["wrapper_evidence"]=saved
                require(all(v.device.type=="cpu" and not v.requires_grad and v.grad_fn is None for v in provider.tensors(saved)),
                    "Final raw evidence must already be detached owning CPU records; no copying fallback")
                report["raw_evidence"]=storage.write(saved,arrays_path,exclusive=True)
            except BaseException as error:
                report["passed"]=False;report["failure"]=report["failure"] or dict(error_type=type(error).__name__,error=str(error))
                last=getattr(storage,"last_write",None)
                report["raw_evidence"]=last if last is not None and last["path"]==str(arrays_path) else dict(
                    path=str(arrays_path),exists=arrays_path.exists(),bytes=arrays_path.stat().st_size if arrays_path.exists() else None,
                    hash_unknown=True,sha256=None,write_completed=False)
                report["raw_evidence"]["error"]=repr(error)
            report["storage_accounting"]=dict(peak_charged_bytes=storage.peak,observations=storage.observations,
                scope="All current CUDA allocations plus deduplicated retained/live CPU/copies; fixed768MiB original head/adjoint phases and64MiB otherwise +4MiB scratch. No CUDA peak resets; selected allowance is not private-library proof.",solver_phases=storage.solver_phases)
        try:
            report["readonly_exit"]=pins();require(report["readonly_exit"]==packet["readonly_files_sha256"],"Readonly exit exact")
            if torch is not None:
                from src.research_loop import implementation_provenance
                report["source_exit"]=implementation_provenance();require(report["source_exit"]==packet["source"],"Source exit exact")
            check()
        except BaseException as error:
            report["passed"]=False;report["failure"]=report["failure"] or dict(error_type=type(error).__name__,error=str(error))
        if not report["passed"]:report["completed"]=False
        report["resources"]=peaks()
        try:
            write_report();check()
        except BaseException as error:
            report["passed"]=False;report["failure"]=report["failure"] or dict(error_type=type(error).__name__,error=str(error))
            report["completed"]=False;report["resources"]=peaks();write_report()
        finally:
            signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,oldhandler)
            if oldtimer[0]>0:signal.setitimer(signal.ITIMER_REAL,*oldtimer)
    require(report["passed"],"Terminal native qualification failure; inspect preserved report")
    return str(report_path)
