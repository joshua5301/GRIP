"""FM: two original-source folds, one original packed native pullback; cold1 then authorized24."""
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import resource
import signal
import time

KIND="two_fixed_source_fold_original135_composed_uniform_CE_native_qualification_and_continuation_stageFM_v1"
SCIENCE_SHA="70369d3630415c280ebd6a7049f6baa4fd317b3e46157d9e776e4a57c455ab45"
ENGINE_SHA="a3f6dbd64b0a62dd44fb029b577b6022ee50c08d2a89be73c061906e5a2be9e7"
HELPER_SHA="a9eb0d522713f6c8928b2bcd50d9bdc5440bca07ca49ab1c47d3c78721fac10c"


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path,check=lambda:None):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(4194304),b""):check();digest.update(block)
    return digest.hexdigest()


def refs(value,found=None):
    found={} if found is None else found
    if isinstance(value,dict):
        if set(value)=={"path","sha256"}:
            p,h=value["path"],value["sha256"]
            require(type(p) is str and Path(p).is_absolute() and str(Path(p).resolve())==p and type(h) is str
                and len(h)==64 and all(c in "0123456789abcdef" for c in h),"Normalized ABS ref required")
            require(p not in found or found[p]==h,"Conflicting ref");found[p]=h
        for child in value.values():refs(child,found)
    elif isinstance(value,list):
        for child in value:refs(child,found)
    return found


def run(protocol_path,protocol_sha256,budget,mode,stop=lambda:False):
    started=time.monotonic();require(Path(protocol_path).is_absolute() and str(Path(protocol_path).resolve())==protocol_path
        and sha(protocol_path)==protocol_sha256,"Protocol path/SHA differs")
    packet=json.loads(Path(protocol_path).read_text());science_ref=packet["scientific_contract"]
    require(science_ref["sha256"]==SCIENCE_SHA and sha(science_ref["path"])==SCIENCE_SHA,"Science differs")
    science=json.loads(Path(science_ref["path"]).read_text())
    require(packet["kind"]==science["kind"]==KIND and budget in science["cases"] and mode in ("qualify","continue"),"Typed kind/budget/mode differs")
    require(packet["mode"]==mode,"Invocation authorizes a different phase")
    case=science["cases"][budget];out=Path(packet["output_folders"][mode][budget]);require(out.is_absolute() and str(out.resolve())==str(out),"Output ABS required")
    out.mkdir(parents=True,exist_ok=False);report_path=out/"native_admission_report.json";arrays_path=out/"native_admission_arrays.pt"
    wrapper=dict(science["exact_counts"]["wrapper_per_invocation"])
    wrapper["accepted_state_PT_loads"]=0
    work=dict.fromkeys(wrapper,0);external=dict.fromkeys(science["exact_counts"]["external_E1"],0)
    report=dict(schema=1,kind=KIND,budget=budget,mode=mode,source=packet["source"],scientific_contract=science_ref,
        protocol=dict(path=protocol_path,sha256=protocol_sha256),passed=False,completed=False,failure=None,gates=[],work=work,
        external_work=external,attempts=dict(PT_loads=0,source_owner_builds=0,composed_owner_builds=0,parity_owner_builds=0,packed_material_builds=0),
        external_attempts=dict.fromkeys(external,0),engine_invocation=dict(attempts=0,returns=0,interiors="not_entered"),
        raw_evidence=dict(path=str(arrays_path),exists=False,bytes=None,sha256=None,hash_unknown=True,write_completed=False))
    saved={};torch=None;source_owner=composed=None;old_handler=signal.getsignal(signal.SIGALRM);old_timer=signal.getitimer(signal.ITIMER_REAL)
    def peaks():
        value=dict(seconds=time.monotonic()-started,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,peak_allocated_bytes=0,peak_reserved_bytes=0)
        if torch is not None and torch.cuda.is_initialized():value.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(0),peak_reserved_bytes=torch.cuda.max_memory_reserved(0))
        return value
    def check():require(not stop() and all(v<=science["resources"][k] for k,v in peaks().items()),"Frozen stop/time/RSS/CUDA boundary")
    def gate(name,ok,**detail):report["gates"].append(dict(name=name,passed=bool(ok),**detail));require(ok,name)
    def alarm(signum,frame):raise TimeoutError("Frozen300s native boundary")
    def pins():
        expected=packet["readonly_files_sha256"]
        require(all(Path(p).is_absolute() and str(Path(p).resolve())==p for p in expected),"Pin path differs")
        require(all(expected.get(p)==h for p,h in refs(dict(packet=packet,science=science)).items()),"Direct ref pin absent")
        return {p:sha(p,check) for p in expected}
    try:
        signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,max(1e-6,300-(time.monotonic()-started)))
        report["readonly_entry"]=pins();gate("readonly_entry_exact",report["readonly_entry"]==packet["readonly_files_sha256"])
        gate("own_entrypoint_exact",Path(packet["entrypoint"]["path"]).resolve()==Path(__file__).resolve())
        import numpy as np
        import torch
        from src import source_crossfit_composed_ce_native as native,source_crossfit_composed_ce_trajectory as engine
        from src import composed_centroid_joint_ce as joint,source_crossfit_composed_ce as CPU_math
        from src.dual_head_ce import _factor_digests
        from src.kernel_mean_ce import _cpu,_plain,_seal
        from src.low_rank_assignment import LowRankMoments
        from src.moments import augmented
        from src.nystrom_ce import NystromMap
        from src.research_loop import implementation_provenance
        from src.shared_features import _tensor_identity
        from src.soft_ce_partition import solve_head_system
        from src.transforms import FeatureTransform
        def retain(key,value):saved[key]=value;saved[key]=_cpu(value)
        gate("new_engine_helper_and_original_math_exact",Path(engine.__file__).resolve()==Path(packet["engine_entrypoint"]["path"]).resolve()
            and sha(engine.__file__)==packet["engine_entrypoint"]["sha256"]==ENGINE_SHA
            and Path(native.__file__).resolve()==Path(packet["helper_entrypoint"]["path"]).resolve()
            and sha(native.__file__)==packet["helper_entrypoint"]["sha256"]==HELPER_SHA
            and sha(joint.__file__)==science["required_refs"]["original_helper"]["sha256"]
            and sha(CPU_math.__file__)==science["required_refs"]["qualified_two_fold_math"]["sha256"])
        report["source_entry"]=implementation_provenance();gate("current_full_source_Git_exact",report["source_entry"]==packet["source"])
        bridge=json.loads(Path(packet["prerequisite_admission"]["path"]).read_text())
        gate("ROOT_current_static_source_prerequisites",bridge["passed"] is True and bridge["source"]==packet["source"]
            and bridge["scientific_contract"]==science_ref and bridge["helper_entrypoint"]==packet["helper_entrypoint"]
            and bridge["engine_entrypoint"]==packet["engine_entrypoint"] and bridge["driver_entrypoint"]==packet["entrypoint"])
        gate("actual_FL_CPU_admission_fixed",packet["CPU_admission"]==science["CPU_admission"])
        for role in ("ROOT","independent"):
            admission=json.loads(Path(science["CPU_admission"][role]["path"]).read_text())
            gate(role+"_two_branch_CPU_complete_chain_admitted",admission["passed"] is True
                and admission["actual_report"]==science["CPU_admission"]["report"] and admission["actual_arrays"]==science["CPU_admission"]["arrays"]
                and admission["scientific_contract"]==science["required_refs"]["CPU_science"])
        cp=case["source135"];root=json.loads(Path(cp["ROOT"]["path"]).read_text());peer=json.loads(Path(cp["peer"]["path"]).read_text())
        cap=json.loads(Path(cp["report"]["path"]).read_text());producer=science["source135_literal_producer"]
        gate("actual135_source_admitted",root["passed"] is True and peer["passed"] is True and cap["passed"] is True and cap["capture_completed"] is True
            and root["budgets"][budget]["report"]==peer["budgets"][budget]["actual_report"]==cp["report"]
            and root["budgets"][budget]["arrays"]==peer["budgets"][budget]["arrays"]==cp["arrays"]
            and cap["source"]==cap["source_entry"]==cap["source_exit"]==producer==case["source_capture_producer"]
            and cap["scientific_contract"]==science["required_refs"]["source135_science"] and Path(cp["arrays"]["path"]).stat().st_size==cp["expected_array_bytes"])
        torch.set_num_threads(4)
        if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision("highest");torch.use_deterministic_algorithms(False);torch.cuda.init();device=torch.device("cuda:0")
        report["runtime"]=joint._runtime(device)
        gate("original_native_runtime",report["runtime"]==case["original_runtime"] and platform.python_version()=="3.12.7"
            and np.__version__=="1.26.4" and torch.get_num_interop_threads()==1
            and all(os.environ.get(k)=="4" for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"))
            and os.environ.get("CUBLAS_WORKSPACE_CONFIG")==":4096:8" and os.environ.get("CUDA_VISIBLE_DEVICES")=="0")
        def load(reference,role):
            check();report["attempts"]["PT_loads"]+=1;value=torch.load(reference["path"],map_location="cpu",weights_only=False)
            work[role]+=1;return value
        payload=load(cp["arrays"],"source135_PT_loads");retain("loaded_original_source135_payload",payload)
        gate("own_original_source135_payload_header",payload["schema"]==1 and payload["kind"]==cap["kind"] and payload["budget"]==budget
            and payload["source"]==producer and payload["scientific_contract"]==science["required_refs"]["source135_science"])
        captured=payload["captured"];dims=case["dimensions"];n,d,b,c,k,rank=(dims[x] for x in ("nodes","physical_dimension","original_Phi_basis","classes","cells","rank"))
        assets=case["asset_descriptors"];z=captured["z_transform"]["z"].to(device);q=captured["Q"].to(device);hard=captured["hard"].to(device);phi=captured["original_Phi"]
        initial=[captured["native_factors"][x].to(device) for x in ("u","v")]
        gate("own_initial_native_parameters",_factor_digests(initial,n,k,rank,device)==case["native_initial_parameter_digests"]
            and bool(initial[0].eq(0).all()) and captured["native_factors"]["data_digest"]==case["original_data_digest"]
            and captured["native_factors"]["factor_seed"]==0 and captured["native_factors"]["mixing"]==.05)
        native_physical=dict(moments=captured["physical_M0"],**captured["native_physical"])
        gate("historical135_physical_metadata_only",all(_tensor_identity(value)=={x:cap["descriptors"]["native_physical"][name][x] for x in ("shape","dtype","digest")}
            for name,value in native_physical.items()))
        transform=FeatureTransform(**{key:value.to(device) if torch.is_tensor(value) else value for key,value in captured["z_transform"]["transform"].items()})
        feature_map=NystromMap(captured["map_cache"]["anchors"].to(device),captured["map_cache"]["mapping"].to(device),"relu")
        layout=joint.ComposedJointLayout(d,b,c)
        report["attempts"]["source_owner_builds"]+=1;report["attempts"]["parity_owner_builds"]+=1
        source_owner=native.TwoFoldOriginalSourceOwner(z,phi,q,layout,cp,original_components=dict(z=assets["z"],Phi_identity=assets["Phi_identity"]))
        work["source_owner_builds"]+=1;work["parity_owner_builds"]+=1;retain("source_owner_descriptor",source_owner.descriptor())
        report["attempts"]["composed_owner_builds"]+=1
        composed=joint.OriginalRMSComposedCentroidFeatures(transform,feature_map,layout,{x:assets[x] for x in ("transform","anchors","mapping")})
        work["composed_owner_builds"]+=1;retain("composed_owner_descriptor",composed.descriptor())
        gate("original135_H_Q_native_descriptors",all(_tensor_identity(value)==assets[name] for name,value in (("H",captured["H_cache"]["h"]),("z",z),("Q",q),("assignment",hard))))
        report["attempts"]["packed_material_builds"]+=1;material=source_owner.material_on(device);work["packed_material_builds"]+=1;retain("shared_packed_material",material)
        gate("fixed_intrinsic_logical_budget",material.numel()*8+n*(d+b+c)*8==case["memory"]["intrinsic_source_plus_packed"]<=science["logical_cap_bytes"])
        repo=Path(packet["repository"]).resolve();binding=packet["trajectory_context_bindings"][budget]
        gate("current_context_source_ABS",Path(packet["repository"]).is_absolute() and str(repo)==packet["repository"]
            and set(binding)=={"files_sha256","current_source"} and binding["current_source"]==dict(packet["source"],files={str(repo/path):h for path,h in packet["source"]["files"].items()})
            and all(packet["readonly_files_sha256"].get(p)==h for p,h in binding["files_sha256"].items()))
        gate("all_eight_original_assets_both_domains",all(binding["files_sha256"].get(p)==h and packet["readonly_files_sha256"].get(p)==h
            for B in science["cases"].values() for p,h in B["mandatory_original_asset_pins"].items()))
        source_refs=dict(nodes=n,cells=k,rank=rank,dimension=d,basis=b,classes=c,chunk_size=case["original_options"]["chunk_size"],factor_seed=0,mixing=.05,
            data_digest=case["original_data_digest"],helper_source_sha256=HELPER_SHA,inner_helper_source_sha256=science["required_refs"]["original_helper"]["sha256"],
            CPU_math_source_sha256=science["required_refs"]["qualified_two_fold_math"]["sha256"],source_admission=cp,CPU_admission=science["CPU_admission"],
            historical_source135_producer=producer,original_options=case["original_options"],runtime=case["original_runtime"],device="cuda:0",asset_paths=case["asset_paths"],scientific_endpoint=25,**binding)
        contract=native.build_pre_origin_contract(source_refs,assets,case["native_initial_parameter_digests"],source_owner.descriptor(),composed.descriptor())
        report["pre_origin_contract"]=contract;accepted_state=None;accepted_payload_seal=_seal(payload);old_files={}
        if mode=="continue":
            accepted=packet["accepted_qualifier"][budget];qualifying=json.loads(Path(accepted["report"]["path"]).read_text())
            for role in ("ROOT","independent"):
                admission=json.loads(Path(accepted[role]["path"]).read_text())
                gate(role+"_native1_accepted",admission["passed"] is True and all(admission["budgets"][budget][name]==accepted[name] for name in ("report","state","arrays")))
            gate("accepted_qualifier_same_source_input",qualifying["passed"] is True and qualifying["mode"]=="qualify" and qualifying["kind"]==KIND
                and qualifying["source"]==qualifying["source_entry"]==qualifying["source_exit"]==packet["source"] and qualifying["scientific_contract"]==science_ref
                and qualifying["pre_origin_contract"]==contract and qualifying["state"]==accepted["state"]
                and qualifying["raw_evidence"]["path"]==accepted["arrays"]["path"] and qualifying["raw_evidence"]["sha256"]==accepted["arrays"]["sha256"])
            accepted_state=load(accepted["state"],"accepted_state_PT_loads");retain("loaded_accepted_state",accepted_state)
            gate("accepted_state_exact_frontier1",accepted_state["step"]==1 and accepted_state["context"]==qualifying["context"]
                and accepted_state["context"]["pre_origin_contract"]==contract)
            engine.validate_evaluated_state(accepted_state,accepted_state["config"],accepted_state["context"],accepted_state["artifact_folder"])
            report["accepted_input_seal"]=_seal(accepted_state);old_folder=Path(accepted_state["artifact_folder"])
            old_files={str(old_folder/"optimization.csv"):accepted_state["history_sha256"],**{str(old_folder/"checkpoints"/name):digest for name,digest in accepted_state["checkpoint_files_sha256"].items()}}
            gate("accepted_old_artifact_pins",old_folder==Path(accepted["state"]["path"]).parent and all(packet["readonly_files_sha256"].get(p)==h for p,h in old_files.items()))
            report["accepted_origin_files"]=old_files
        report["engine_invocation"].update(attempts=1,interiors="unknown_until_return")
        state=engine.optimize_source_crossfit_composed_ce(initial,hard,dict(case["original_options"],save_resume=True),contract,source_owner,composed,
            material,str(out/"core"),1 if mode=="qualify" else 25,resume_state=accepted_state,stop=stop)
        report["engine_invocation"].update(returns=1,interiors="returned_state_counts");retain("returned_core_state",state)
        report["context"]=state["context"];report["core_work"]=state["work"];report["core_attempts"]=state["attempts"]
        report["state"]=dict(path=str(out/"core"/"resume.pt"),sha256=sha(out/"core"/"resume.pt",check))
        report["evaluation_summaries"]={str(step):dict(scalars=_plain({key:E[key] for key in ("step","config_digest","context_digest","source_admission_digest","initial_parameter_digests","parent_update_digest","raw_CE","CE0","objective","record_digest")}),
            packed_moments=_tensor_identity(E["packed_moments"]),parameters_digest=_seal(E["parameters"]),folds={f:dict(raw_CE=e["raw_CE"],head_work=e["head_work"],theta_initial_digest=_seal(e["theta_initial"]),
                identities={key:_tensor_identity(e[key]) for key in ("theta","raw_rhs","head_features","physical_moments","physical_centers","physical_targets","physical_mass")}) for f,e in E["folds"].items()}) for step,E in state["snapshots"].items()}
        A=state["last_completed_update"]
        report["last_update_summary"]=dict(scalars=_plain({key:A[key] for key in ("step","evaluation_digest","previous_update_digest","raw_adjoint_initial_digest","record_digest")}),
            folds={f:dict(adjoint_work=a["adjoint_work"],raw_adjoint=_tensor_identity(a["raw_adjoint"]),raw_moment_G=_tensor_identity(a["raw_moment_G"])) for f,a in A["folds"].items()},
            identities={key:_tensor_identity(A[key]) for key in ("normalized_G","native_grad_U","native_grad_V")},parameters_after_digest=_seal(A["parameters_after"]),optimizer_after_digest=_seal(A["optimizer_after"]))
        report["core_artifacts"]=dict(history=dict(path=str(out/"core"/"optimization.csv"),sha256=state["history_sha256"]),resume=report["state"],
            checkpoints={name:dict(path=str(out/"core"/"checkpoints"/name),sha256=digest) for name,digest in state["checkpoint_files_sha256"].items()})
        expected=science["exact_counts"]["core_cold0to1" if mode=="qualify" else "cumulative_cold1_then_resume24"]
        gate("exact_core_counts_and_attempts",state["work"]==state["attempts"]==expected)
        if mode=="qualify":
            E=state["current_evaluation"];params=[p.to(device) for p in E["parameters"]];vectors={}
            for f in ("A","B"):
                e=E["folds"][f];features,targets,mass=(e[x].to(device) for x in ("head_features","physical_targets","physical_mass"))
                report["external_attempts"]["raw_adjoint"]+=1
                vector,diag=solve_head_system(augmented(features),targets,torch.full_like(mass,1/k),e["theta"].to(device),case["original_options"]["penalty"],e["raw_rhs"].to(device),
                    rtol=case["original_options"]["cg_rtol"],max_iter=case["original_options"]["cg_max_iter"],initial=A["folds"][f]["raw_adjoint"].to(device),cg_check_interval=case["original_options"].get("cg_check_interval",1))
                external["raw_adjoint"]+=1;retain("external_raw_adjoint_"+f,dict(vector=vector,diagnostic=diag));vectors[f]=vector.detach()
                gate("external_raw_adjoint_"+f+"_converged",diag.get("cg_converged") is True and bool(torch.isfinite(vector).all()))
            def observed_map(func,x):
                report["external_attempts"]["G_map"]+=1;value=func(x);external["G_map"]+=1;retain("external_G_composed_components",composed.last_returned());return value
            def observed_raw(func,*a,**kw):
                report["external_attempts"]["raw_moment_G"]+=1;value=func(*a,**kw);external["raw_moment_G"]+=1;retain("external_latest_raw_G",value);return value
            report["external_attempts"]["packed_complete_G"]+=1
            G=native.complete_packed_cotangent(E["packed_moments"].to(device),layout,composed,E["folds"]["A"]["theta"].to(device),E["folds"]["B"]["theta"].to(device),vectors["A"],vectors["B"],
                case["original_options"]["penalty"],E["CE0"],map_call=observed_map,raw_call=observed_raw,observe_return=lambda key,value:retain("external_"+key,value))
            external["packed_complete_G"]+=1;retain("external_packed_G",G);check();repeats=[];saved["native_repeats"]=repeats
            for repeat in range(3):
                fresh=[p.detach().clone().requires_grad_() for p in params];report["external_attempts"]["same_G_forward"]+=1
                M=LowRankMoments.apply(*fresh,hard,material,.05,case["original_options"]["chunk_size"]);external["same_G_forward"]+=1
                item=dict(M=_cpu(M));repeats.append(item)
                gate("sameG_forward_equals_E1_"+str(repeat),_tensor_identity(M.detach())==_tensor_identity(E["packed_moments"]))
                report["external_attempts"]["same_G_backward"]+=1;M.backward(G);external["same_G_backward"]+=1
                item.update(U=_cpu(fresh[0].grad),V=_cpu(fresh[1].grad));check()
            gate("three_native_sameG_byte_repeats",all(_seal(item)==_seal(repeats[0]) for item in repeats[1:]))
            report["external_attempts"]["full_K_CPU_native_cast_reference"]+=1
            with torch.no_grad():
                U,V=(p.detach().cpu() for p in params);QG=G.detach().cpu();X=material.detach().cpu();assignment=hard.cpu()
                prior=torch.full((n,k),float(np.log(.05/k)),dtype=torch.float32)
                prior.scatter_(1,assignment[:,None],float(np.log(1-.05+.05/k)))
                logits=prior+U@V.T/math.sqrt(rank);probability=logits.double().softmax(1)
                direction=X@QG.T/n
                block=(probability*(direction-(probability*direction).sum(1,keepdim=True))).to(torch.float32)/math.sqrt(rank)
                reference=dict(U=block@V,V=block.T@U)
                external["full_K_CPU_native_cast_reference"]+=1;retain("CPU_reference",dict(logits=logits,probability=probability,direction=direction,whole_GP_native_block=block,gradients=reference))
                errors={};pairs=dict(U=(repeats[0]["U"],reference["U"]),V=(repeats[0]["V"],reference["V"]))
                pairs["full"]=(torch.cat([pairs[x][0].flatten() for x in ("U","V")]),torch.cat([pairs[x][1].flatten() for x in ("U","V")]))
                for name,(actual,ref) in pairs.items():
                    delta=actual.double()-ref.double();absolute=float(delta.abs().max());error_norm=float(torch.linalg.vector_norm(delta));ref_norm=float(torch.linalg.vector_norm(ref.double()))
                    relative=error_norm/ref_norm if ref_norm>0 else (0.0 if error_norm==0 else None)
                    relative=relative if relative is not None and math.isfinite(relative) else None
                    errors[name]=dict(absolute_max=absolute if math.isfinite(absolute) else None,difference_L2=error_norm if math.isfinite(error_norm) else None,
                        reference_L2=ref_norm if math.isfinite(ref_norm) else None,relative_L2=relative,both_passed=math.isfinite(absolute) and absolute<=5e-7 and relative is not None and relative<=2e-5)
                saved["gradient_errors"]=report["gradient_errors"]=errors
                norms={"U1":float(torch.linalg.vector_norm(U)),"V1":float(torch.linalg.vector_norm(V)),"native_dU":float(torch.linalg.vector_norm(repeats[0]["U"])),"native_dV":float(torch.linalg.vector_norm(repeats[0]["V"]))}
                report["nonzero_norms"]={name:value if math.isfinite(value) else None for name,value in norms.items()}
            gate("nonzero_finite_endpoint1_and_native_gradients",all(math.isfinite(v) and v>0 for v in norms.values()) and all(bool(torch.isfinite(item[key]).all()) for item in repeats for key in ("M","U","V")))
            for name,value in errors.items():gate(name+"_BOTH_gradient_bounds",value["both_passed"],**value)
            gate("diagnostic_core_state_unchanged",_seal(state)==_seal(saved["returned_core_state"]) and sha(out/"core"/"resume.pt",check)==report["state"]["sha256"])
        else:
            gate("entire_accepted_input_unchanged",_seal(accepted_state)==report["accepted_input_seal"])
            gate("cached_E0_E1_and_old_checkpoint_copies_unchanged",all(_seal(accepted_state["snapshots"][step])==_seal(state["snapshots"][step]) for step in (0,1))
                and all(sha(out/"core"/"checkpoints"/name,check)==digest for name,digest in accepted_state["checkpoint_files_sha256"].items()))
            gate("accepted_old_CSV_checkpoints_unchanged",all(sha(p,check)==h for p,h in old_files.items()))
        gate("own_source135_payload_unchanged",_seal(payload)==accepted_payload_seal)
        expected_wrapper=dict(science["exact_counts"]["wrapper_per_invocation"],accepted_state_PT_loads=0 if mode=="qualify" else 1)
        gate("exact_wrapper_counts",work==expected_wrapper and report["attempts"]==dict(PT_loads=1 if mode=="qualify" else 2,source_owner_builds=1,composed_owner_builds=1,parity_owner_builds=1,packed_material_builds=1))
        expected_external=science["exact_counts"]["external_E1"] if mode=="qualify" else dict.fromkeys(external,0)
        gate("exact_external_counts",external==report["external_attempts"]==expected_external)
        saved["latest_composed_components"]=composed.last_returned();torch.cuda.synchronize(device);check();report["completed"]=True
    except BaseException as error:
        report["failure"]=dict(type=type(error).__name__,message=str(error));report["inflight_operation_interiors"]="unknown; attempts and returned evidence remain distinct"
        core_failure=out/"core"/"failure.json"
        if core_failure.exists():report["core_failure_evidence"]=dict(path=str(core_failure),sha256=sha(core_failure))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        try:
            if torch is not None:
                raw=report["raw_evidence"]
                try:
                    if source_owner is not None:saved["source_owner_last_returned"]=source_owner.last_returned()
                    if composed is not None:saved["last_composed_components"]=composed.last_returned()
                    with arrays_path.open("xb") as stream:
                        torch.save(dict(schema=1,kind=KIND,budget=budget,mode=mode,source=packet["source"],scientific_contract=science_ref,saved=saved),stream);stream.flush();os.fsync(stream.fileno())
                    raw["write_completed"]=True
                except BaseException as error:report["raw_write_error"]=repr(error)
                finally:
                    raw["exists"]=arrays_path.exists()
                    if raw["exists"]:raw["bytes"]=arrays_path.stat().st_size
                if raw["exists"]:
                    try:raw["sha256"]=sha(arrays_path);raw["hash_unknown"]=False
                    except BaseException as error:raw["hash_error"]=repr(error)
            try:
                report["readonly_exit"],report["source_exit"]=pins(),implementation_provenance()
                require(report["readonly_exit"]==packet["readonly_files_sha256"] and report["source_exit"]==packet["source"],"Exit pins/source changed")
            except BaseException as error:report["exit_verification_error"]=repr(error)
            report["resources"]=peaks();report["passed"]=bool(report["completed"] and report["failure"] is None and not any(k in report for k in ("raw_write_error","exit_verification_error"))
                and report["raw_evidence"]["write_completed"] and not report["raw_evidence"]["hash_unknown"] and all(v<=science["resources"][k] for k,v in report["resources"].items()))
            if not report["passed"]:report["completed"]=False
            with report_path.open("x") as stream:json.dump(report,stream,indent=2,allow_nan=False);stream.write("\n");stream.flush();os.fsync(stream.fileno())
            final=peaks()
            if report["passed"] and any(v>science["resources"][k] for k,v in final.items()):
                report.update(passed=False,completed=False,resources=final,failure=dict(type="FinalSerializationResourceBoundary",message="Preserve terminal namespace; no retry"))
                with report_path.open("w") as stream:json.dump(report,stream,indent=2,allow_nan=False);stream.write("\n");stream.flush();os.fsync(stream.fileno())
        finally:signal.signal(signal.SIGALRM,old_handler);signal.setitimer(signal.ITIMER_REAL,*old_timer)
    require(report["passed"],"Terminal native qualification failure; inspect owning evidence, no retry")
    return str(report_path)
