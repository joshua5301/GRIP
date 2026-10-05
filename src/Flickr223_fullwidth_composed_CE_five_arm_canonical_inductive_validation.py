"""Canonical Flickr223 serving on exact old controls and one held composed P25.

Only preparation constructs a new raw-H readout. Original GCN sparse graph
products occur normally in fitting/replay; cached source-SGC is never rebuilt.
"""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import signal
import time

KIND="Flickr223_fullwidth_composed_CE_five_arm_canonical_inductive_validation_v1"
SCIENCE_SHA="4884784b563f51caeb3469edda8f79bb7482eccc844d2be6ea00f72d1e8b68f3"
ARMS=("own_canonical_P0","uniform_linear25","uniform_centroid_Ny25","meanPhi25","singleComposed25")


def require(ok,message):
    if not ok:raise ValueError(message)


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


def run(protocol_path,protocol_sha256,phase,arm=None,stop=lambda:False):
    started=time.monotonic();require(Path(protocol_path).is_absolute() and sha(protocol_path)==protocol_sha256,"Frozen protocol")
    packet=json.loads(Path(protocol_path).read_text());science=json.loads(Path(packet["scientific_contract"]["path"]).read_text())
    require(packet["scientific_contract"]["sha256"]==sha(packet["scientific_contract"]["path"])==SCIENCE_SHA
        and packet["kind"]==science["kind"]==KIND and packet["budget"]==science["budget"]=="flickr223"
        and packet["authorized_phase"]==phase and phase in ("prepare","evaluate")
        and (arm is None if phase=="prepare" else arm in ARMS),"Selected science/phase/arm")
    folder=Path(packet["prepare_folder"] if phase=="prepare" else packet["evaluation_folders"][arm])
    require(folder.is_absolute() and str(folder.resolve())==str(folder),"Normalized output");folder.mkdir(parents=True,exist_ok=False)
    report_path=folder/("prepare_report.json" if phase=="prepare" else "evaluation_report.json")
    counts=dict.fromkeys(science["expected_prepare_counts"],0)
    counts.update(PT_load_attempts=0,old_cache_loads=0,native_state_loads=0,signed_own125_mmap_loads=0,
        own_shared_cache_loads=0,rawH_forward_attempts=0,rawH_decode_attempts=0,shared_cache_write_attempts=0,
        fit_attempts=0,student_fits=0,student_epochs=0,completed_fit_CSV_rows=0,route_attempts=0,
        paired_replay_calls=0,validation_routes=0,selected_checkpoint_replay_loads=0)
    report=dict(schema=1,kind=KIND,budget="flickr223",phase=phase,arm=arm,passed=False,completed=False,
        source=packet["source"],scientific_contract=packet["scientific_contract"],protocol=dict(path=protocol_path,sha256=protocol_sha256),
        counts=counts,gates=[],records=[],failure=None,test_enabled=False,goal_complete=False,
        ordinary_GCN_graph_SpMM="Unchanged fitter and replay products occur normally; uninstrumented",
        secondary=science["evaluation"]["secondary"],memory_policy=science["prepare"]["memory"])
    cache={};saved={};torch=None;gpu=False;failure=None
    oldhandler=signal.getsignal(signal.SIGALRM);oldtimer=signal.getitimer(signal.ITIMER_REAL)
    def peaks():
        return dict(seconds=time.monotonic()-started,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(0) if gpu else 0,
            peak_reserved_bytes=torch.cuda.max_memory_reserved(0) if gpu else 0)
    def guard():
        require(not stop() and all(v<=science["resources"][k] for k,v in peaks().items()),"Frozen time/RSS/CUDA boundary")
        return False
    def gate(name,ok):
        report["gates"].append(dict(name=name,passed=bool(ok)));require(ok,name)
    def read(r):return json.loads(Path(r["path"]).read_text())
    def pins():
        expected=packet["readonly_files_sha256"]
        require(all(Path(p).is_absolute() and str(Path(p).resolve())==p for p in expected),"Normalized pins")
        require(all(expected.get(p)==h for p,h in refs(dict(packet=packet,science=science)).items()),"Required immutable pin omitted")
        observed={p:sha(p,guard) for p in expected};require(observed==expected,"Readonly bytes changed")
        return observed
    def opaque(path,record):
        record.update(path=str(path),exists=path.exists(),bytes=path.stat().st_size if path.exists() else None,sha256=None,hash_unknown=True)
        record["sha256"]=sha(path,guard);record["hash_unknown"]=False;return record
    def expire(signum,frame):raise TimeoutError("Frozen300s deadline")
    def write_report():
        with report_path.open("w") as stream:json.dump(observed(report),stream,indent=2,allow_nan=False);stream.write("\n");stream.flush();os.fsync(stream.fileno())
    def observed(v):
        if isinstance(v,float) and not math.isfinite(v):return dict(nonfinite=repr(v))
        if isinstance(v,dict):return {str(k):observed(x) for k,x in v.items()}
        if isinstance(v,(tuple,list)):return [observed(x) for x in v]
        return v
    try:
        signal.signal(signal.SIGALRM,expire);signal.setitimer(signal.ITIMER_REAL,max(.000001,300-(time.monotonic()-started)))
        report["readonly_entry"]=pins()
        gate("own_entrypoint",Path(packet["entrypoint"]["path"]).resolve()==Path(__file__).resolve())
        import numpy as np
        import torch
        from src.research_loop import implementation_provenance
        from src.large_canonical_inductive_validation import _identities,_graph_domain
        from src.large_kernel_mean_source_capture import _matrix
        from src.kernel_mean_ce import _plain,_seal
        from src.io import cpu_state
        from src.shared_features import _tensor_identity
        from src.low_rank_assignment import LowRankMoments
        from src.moments import decode_moments
        from src.Flickr223_fullwidth_composed_joint_ce_streaming_trajectory import validate_evaluated_state
        from src.inductive_evaluation import fit_inductive_gcn
        from src.student_routes import replay_routes
        report["source_entry"]=implementation_provenance();gate("current_source",report["source_entry"]==packet["source"])
        bridge=read(packet["prerequisite_admission"])
        gate("ROOT_static_bridge",bridge["passed"] is True and all(bridge[k]==packet[k] for k in ("source","entrypoint","scientific_contract")))
        torch.set_num_threads(4)
        if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
        torch.set_default_dtype(torch.float32);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision("highest");torch.use_deterministic_algorithms(True)
        rt=science["runtime"]
        gate("CPU_runtime",platform.python_version()==rt["python"] and str(torch.__version__)==rt["torch"] and np.__version__==rt["numpy"]
            and all(os.environ.get(k)==rt[k] for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","CUBLAS_WORKSPACE_CONFIG","CUDA_VISIBLE_DEVICES")))
        torch.cuda.init();torch.cuda.reset_peak_memory_stats(0);gpu=True;device=torch.device("cuda:0")
        report["runtime"]=dict(python=platform.python_version(),torch=str(torch.__version__),numpy=np.__version__,device=str(device),GPU=torch.cuda.get_device_name(0),
            default_dtype=str(torch.get_default_dtype()),threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),AMP=torch.is_autocast_enabled("cuda"),
            TF32=torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32,float32_matmul_precision=torch.get_float32_matmul_precision(),deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
        gate("GPU_runtime",report["runtime"]=={k:v for k,v in rt.items() if k not in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","CUBLAS_WORKSPACE_CONFIG","CUDA_VISIBLE_DEVICES")} and not torch.is_autocast_enabled("cpu"))
        settings=science["evaluation"]["settings"]
        gate("recipe_cohort_arms",read(science["student_recipe_file"])["recipe"]["settings"]==settings
            and settings==dict(epochs=300,eval_every=10,hidden=256,dropout=.5,lr=.01,weight_decay=.0005)
            and science["evaluation"]["student_seeds"]==[621000,621001,621002] and science["arms"]==list(ARMS))
        expected_providers=dict(science["old_cache"]["arm_providers"],singleComposed25=science["native25"]["provider"])
        if phase=="prepare":
            oldspec=science["old_cache"];r=oldspec["provider"];prior,root,peer=map(read,(r["actual_report"],r["ROOT"],r["independent"]))
            gate("old127_metadata_admitted",prior["passed"] is root["passed"] is peer["passed"] is True
                and prior["source"]==root["source"]==peer["source"]==r["producer"]
                and prior["scientific_contract"]==root["scientific_contract"]==peer["scientific_contract"]==r["scientific_contract"]
                and prior["shared_cache"]==root["shared_cache"]==peer["shared_cache"]==r["arrays"]
                and root["shared_prepare_report"]==peer["actual_report"]==r["actual_report"]
                and Path(r["arrays"]["path"]).stat().st_size==oldspec["file_bytes"]
                and prior["cache_identities"]==root["cache_identities"]==dict(peer["shared_identities"],arms=peer["canonical_arm_descriptors"])==oldspec["identities"]
                and prior["graph_provider"]==root["graph_provider"]==peer["graph_provider"]==oldspec["graph_provider"])
            counts["PT_load_attempts"]+=1;old=torch.load(r["arrays"]["path"],map_location="cpu",weights_only=False)
            saved["returned_old_cache"]=old;counts["PT_loads"]+=1;counts["old_cache_loads"]+=1
            report["returned_old_cache_header"]={k:old.get(k) for k in ("schema","kind","source","scientific_contract","source_tokens")}
            report["returned_old_cache_identities"]=_identities(old)
            gate("whole_old127_owner",old["schema"]==1 and old["kind"]==oldspec["kind"] and old["budget"]=="flickr223" and old["source"]==r["producer"]
                and old["scientific_contract"]==r["scientific_contract"] and old["identities"]==_identities(old)==oldspec["identities"]
                and old["arm_providers"]==oldspec["arm_providers"] and old["graph_provider"]==oldspec["graph_provider"] and old["source_tokens"]==oldspec["source_tokens"])
            cache=dict(schema=1,kind=KIND,budget="flickr223",source=packet["source"],scientific_contract=packet["scientific_contract"],
                graph_provider=oldspec["graph_provider"],old_cache_provider=r,arm_providers=expected_providers,arms=cpu_state(old["arms"]))
            cache.update({k:cpu_state(old[k]) for k in oldspec["shared_keys"]});guard()
            n=science["native25"];nr=read(n["provider"]["report"])
            for role in ("ROOT","independent"):
                a=read(n["provider"][role]);gate(role+"_actual25_admitted",a["passed"] is True and a["budgets"]["flickr223"]["report"]==n["provider"]["report"] and a["budgets"]["flickr223"]["state"]==n["provider"]["state"])
            gate("actual25_native_producer",nr["passed"] is nr["completed"] is True and nr["mode"]=="continue"
                and nr["source"]==n["provider"]["producer"] and nr["endpoint"]==n["endpoint"] and nr["context"]==n["context"] and nr["input_contract"]==n["input_contract"])
            gate("accepted_native_core_artifact_pins",packet["native25_core_artifacts"]=={p:dict(path=p,sha256=h) for p,h in nr["core_artifacts"].items()})
            counts["PT_load_attempts"]+=1;state=torch.load(n["provider"]["state"]["path"],map_location="cpu",weights_only=False)
            saved["returned_native_state"]=state;counts["PT_loads"]+=1;counts["native_state_loads"]+=1
            gate("accepted_native_artifact_folder",state["artifact_folder"]==str(Path(n["provider"]["state"]["path"]).parent))
            validate_evaluated_state(state,n["expected_config"],folder=state["artifact_folder"])
            gate("own_pure_E25_state",state["step"]==25 and _plain(state["current_evaluation"])==n["endpoint"]
                and _plain(state["context"])==n["context"] and _plain(state["input_contract"])==n["input_contract"] and state["work"]==n["core_work"]
                and json.loads(json.dumps(_plain(state["last_completed_update"]),sort_keys=True,allow_nan=False))==n["last_update"])
            counts["PT_load_attempts"]+=1;mapped=torch.load(science["required_refs"]["owning125_arrays"]["path"],map_location="cpu",weights_only=False,mmap=True)
            counts["PT_loads"]+=1;counts["signed_own125_mmap_loads"]+=1;hard=mapped["hard"].detach().clone();saved["returned_small_hard"]=hard;del mapped
            cap=read(science["required_refs"]["owning125_report"])
            gate("original125_H_Q_native",cap["passed"] is True and cap["source"]==science["source125"]["producer"]
                and _tensor_identity(cache["train_H"])==cap["array_descriptors"]["saved_H_FP32"]
                and _tensor_identity(cache["Q"])==cap["array_descriptors"]["Q"]==n["input_contract"]["original_Torch_components"]["Q"]
                and _tensor_identity(hard)==cap["array_descriptors"]["hard"]==n["input_contract"]["native_descriptors"]["hard"]
                and all(_tensor_identity(t)==cap["array_descriptors"][k]==n["input_contract"]["native_descriptors"][k] for t,k in zip(state["initial_parameters"],("U0","V0"),strict=True)))
            _matrix(cache["train_H"],(44625,500),torch.float32);_matrix(cache["Q"],(44625,7),torch.float64)
            report["memory_observations"]=dict(before_rawH_material=peaks())
            material=torch.empty((44625,508),dtype=torch.float64,device=device);report["pack_records"]=[]
            for start in range(0,44625,2048):
                end=min(start+2048,44625);material[start:end,0]=1
                material[start:end,1:501].copy_(cache["train_H"][start:end].to(device=device,dtype=torch.float64))
                material[start:end,501:].copy_(cache["Q"][start:end].to(device))
                counts["fixed2048_rawH_material_pack_blocks"]+=1;report["pack_records"].append(dict(start=start,end=end));guard()
            u,v=[t.detach().to(device).clone() for t in state["current_evaluation"]["parameters"]];hd=hard.to(device)
            saved["held_native_parameters"]=cpu_state((u,v));counts["rawH_forward_attempts"]+=1
            with torch.no_grad():M=LowRankMoments.apply(u,v,hd,material,.05,2048)
            saved["returned_rawH_M25"]=cpu_state(M);counts["original_rawH_LowRank_forwards"]+=1
            counts["rawH_decode_attempts"]+=1;x,y,mass=decode_moments(M,500)
            saved["returned_native_rawH_quotient_tuple"]=cpu_state(dict(X=x,Q=y,mass=mass));counts["native_rawH_quotient_decodes"]+=1
            cache["owning_canonical_export"]=dict(rawH_M25=saved["returned_rawH_M25"],quotients=saved["returned_native_rawH_quotient_tuple"],parameters=saved["held_native_parameters"],hard=hard)
            cache["arms"]["singleComposed25"]=dict(X=cpu_state(x.float()),Qbar=cpu_state(y.float()),uniform_weights=cpu_state(cache["arms"]["own_canonical_P0"]["uniform_weights"]))
            report["canonical_export_identities"]=_plain(cache["owning_canonical_export"])
            gate("native_rawH_return_domain",M.shape==(223,508) and M.dtype==torch.float64 and x.shape==(223,500) and y.shape==(223,7) and mass.shape==(223,)
                and x.dtype==y.dtype==mass.dtype==torch.float64 and bool(torch.isfinite(M).all()) and bool((mass>0).all())
                and all(bool(torch.isfinite(t).all()) for t in (x,y,mass)) and bool((y>=0).all()) and bool((y.sum(1)>0).all()))
            report["memory_observations"]["after_rawH_quotient_before_release"]=peaks()
            del material;guard()
        else:
            prior=read(packet["shared_prepare_report"])
            for role in ("shared_prepare_ROOT","shared_prepare_independent"):
                a=read(packet[role]);gate(role+"_actual_cache_admitted",a["passed"] is True and a["budget"]=="flickr223"
                    and a["actual_report"]==packet["shared_prepare_report"] and a["shared_cache"]==packet["shared_cache"])
            gate("current_prepare_producer",prior["passed"] is True and prior["phase"]=="prepare" and prior["source"]==packet["source"]
                and prior["scientific_contract"]==packet["scientific_contract"] and prior["shared_cache"]==packet["shared_cache"])
            counts["PT_load_attempts"]+=1;cache=torch.load(packet["shared_cache"]["path"],map_location="cpu",weights_only=False)
            saved["returned_own_cache"]=cache;counts["PT_loads"]+=1;counts["own_shared_cache_loads"]+=1
            gate("own_shared_cache_identity",cache["source"]==packet["source"] and cache["scientific_contract"]==packet["scientific_contract"]
                and cache["identities"]==_identities(cache)==prior["cache_identities"]
                and _plain(cache["owning_canonical_export"])==prior["canonical_export_identities"])
        gate("cache_provider_scope",cache["schema"]==1 and cache["kind"]==KIND and cache["budget"]=="flickr223"
            and set(cache["arms"])==set(ARMS) and cache["graph_provider"]==science["old_cache"]["graph_provider"]
            and cache["old_cache_provider"]==science["old_cache"]["provider"] and cache["arm_providers"]==expected_providers and cache["source_tokens"]==science["old_cache"]["source_tokens"])
        for name,nodes in (("train",44625),("val",22312)):
            _graph_domain(cache[name+"_graph"],cache[name+"_mask"],nodes);_matrix(cache[name+"_H"],(nodes,500),torch.float32)
        _matrix(cache["Q"],(44625,7),torch.float64)
        gate("raw_Q_domain",bool((cache["Q"]>=0).all()) and bool((cache["Q"].sum(1)>0).all()))
        for name,row in cache["arms"].items():
            for key,shape,dtype in (("X",(223,500),torch.float32),("Qbar",(223,7),torch.float32),("uniform_weights",(223,),torch.float64)):_matrix(row[key],shape,dtype)
            require(bool((row["Qbar"]>=0).all()) and bool((row["Qbar"].sum(1)>0).all()) and bool((row["uniform_weights"]>0).all())
                and torch.equal(row["uniform_weights"],cache["arms"]["own_canonical_P0"]["uniform_weights"]),"GeneralQ/common uniform weights")
        identities=_identities(cache)
        gate("old_four_and_shared_bytes_unchanged",all(identities[k]==v for k,v in science["old_cache"]["identities"].items() if k!="arms")
            and all(identities["arms"][name]==science["old_cache"]["identities"]["arms"][name] for name in ARMS[:4]))
        cache["identities"]=identities;report["cache_identities"]=identities;report["arm_providers"]=cache["arm_providers"];report["graph_provider"]=cache["graph_provider"]
        report["old_cache_provider"]=cache["old_cache_provider"];report["source_tokens"]=cache["source_tokens"];report["canonical_export_identities"]=_plain(cache["owning_canonical_export"]);guard()
        if phase=="evaluate":
            report.update(shared_cache=packet["shared_cache"],shared_prepare_report=packet["shared_prepare_report"])
            train={k:t.to(device) for k,t in cache["train_graph"].items()};val={k:t.to(device) for k,t in cache["val_graph"].items()}
            train_mask=cache["train_mask"].to(device);val_mask=cache["val_mask"].to(device);val_H=cache["val_H"].to(device)
            x,y,w=[cache["arms"][arm][k].to(device) for k in ("X","Qbar","uniform_weights")];students=folder/"students"
            for seed in science["evaluation"]["student_seeds"]:
                guard();counts["fit_attempts"]+=1
                fit=fit_inductive_gcn(x,y,w,train,(val,None),testing=None,seed=seed,settings=settings,folder=students,training_adjacency=None,stop=guard,weighting="uniform",train_mask=train_mask)
                record=dict(seed=seed,fit=fit);report["records"].append(record);counts["student_fits"]+=1;counts["students"]+=1;counts["student_epochs"]+=300
                files={k:students/f"seed_{seed}{suffix}" for k,suffix in (("json",".json"),("csv","_epochs.csv"),("selected","_selected.pt"))}
                record["fit_files"]={k:{} for k in files}
                for k,p in files.items():opaque(p,record["fit_files"][k])
                with files["csv"].open() as stream:rows=list(csv.DictReader(stream))
                counts["completed_fit_CSV_rows"]+=len(rows);saved_fit=json.loads(files["json"].read_text());best=max(float(r["val_acc"]) for r in rows);first=next(r for r in rows if float(r["val_acc"])==best)
                gate("selected_fit_"+str(seed),[int(r["epoch"]) for r in rows]==list(range(10,301,10)) and all(int(r["val_nodes"])==22312 for r in rows)
                    and fit["epoch"]==int(first["epoch"]) and fit["val_acc"]==best and fit["last_epoch"]==300 and fit["train_nodes"]==44625
                    and saved_fit["result"]==fit and saved_fit["recipe"]["settings"]==settings and saved_fit["recipe"]["seed"]==seed and saved_fit["recipe"]["weighting"]=="uniform" and saved_fit["recipe"]["test_enabled"] is False)
                correct=round(fit["val_acc"]*22312/100);record["selected_GCN_correct"]=correct
                gate("integer_correct_"+str(seed),0<=correct<=22312 and abs(fit["val_acc"]-100*correct/22312)<=1e-10 and not any("test" in k for k in fit) and all(not isinstance(v,float) or math.isfinite(v) for v in fit.values()))
                counts["route_attempts"]+=1;route_path=students/f"seed_{seed}_validation_routes.json"
                route=replay_routes(files["selected"],val,val_H,{"val":val_mask},settings,route_path,seed=seed,stop=guard)
                record["routes"]=route;counts["paired_replay_calls"]+=1;counts["validation_routes"]+=2;counts["selected_checkpoint_replay_loads"]+=1
                record["route_file"]={};opaque(route_path,record["route_file"]);saved_route=json.loads(route_path.read_text())
                gate("same_selected_weight_routes_"+str(seed),route["gcn_val_acc"]==fit["val_acc"] and route["gcn_val_ce"]==fit["val_ce"] and route["epoch"]==fit["epoch"]
                    and saved_route["result"]==route and saved_route["recipe"]["source_fingerprint"]==saved_fit["fingerprint"] and saved_route["recipe"]["settings"]==settings
                    and saved_route["recipe"]["test_enabled"] is False and not any("test" in k for k in route) and all(not isinstance(v,float) or math.isfinite(v) for v in route.values()));guard()
            gate("complete_arm",counts["student_fits"]==3 and counts["student_epochs"]==900 and counts["completed_fit_CSV_rows"]==90 and counts["validation_routes"]==6 and counts["selected_checkpoint_replay_loads"]==3)
    except BaseException as error:
        failure=error;report["failure"]=dict(error_type=type(error).__name__,error=str(error),terminal_no_retry=True)
    finally:
        if phase=="prepare":
            path=folder/"serving_cache.pt";report["shared_cache"]=dict(path=str(path),sha256=None);raw=dict(path=str(path),exists=False,bytes=None,write_completed=False,hash_unknown=True);report["raw_cache_evidence"]=raw
            try:
                if failure is not None:cache["failed_owning_evidence"]=saved
                counts["shared_cache_write_attempts"]+=1
                with path.open("xb") as stream:torch.save(cpu_state(cache),stream);stream.flush();os.fsync(stream.fileno())
                raw["write_completed"]=True;counts["shared_cache_writes"]+=1
            except BaseException as error:
                failure=failure or error;raw["write_error"]=repr(error)
            finally:
                raw.update(exists=path.exists(),bytes=path.stat().st_size if path.exists() else None)
                report.setdefault("memory_observations",{})["after_cache_serialization"]=peaks()
            try:
                if raw["exists"]:opaque(path,raw);report["shared_cache"]["sha256"]=raw["sha256"]
            except BaseException as error:failure=failure or error;raw["hash_error"]=repr(error)
        report["output_files"]={str(p):dict(exists=True,bytes=p.stat().st_size,sha256=None,hash_unknown=True) for p in folder.rglob("*") if p.is_file()}
        try:
            if phase=="prepare" and failure is None:gate("exact_prepare_counts",all(counts[k]==v for k,v in science["expected_prepare_counts"].items()) and counts["PT_load_attempts"]==3 and counts["rawH_forward_attempts"]==counts["rawH_decode_attempts"]==1)
            report["readonly_exit"]=pins()
            if torch is not None:report["source_exit"]=implementation_provenance();require(report["source_exit"]==packet["source"],"Source exit changed")
            guard()
        except BaseException as error:failure=failure or error
        report["passed"]=report["completed"]=failure is None;report["resources"]=peaks()
        if failure is not None:report["failure"]=report["failure"] or dict(error_type=type(failure).__name__,error=str(failure),terminal_no_retry=True)
        try:write_report();guard()
        except BaseException as error:
            report["passed"]=report["completed"]=False;report["failure"]=report["failure"] or dict(error_type=type(error).__name__,error=str(error),terminal_no_retry=True);report["resources"]=peaks();write_report()
        finally:
            signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,oldhandler)
            if oldtimer[0]>0:signal.setitimer(signal.ITIMER_REAL,max(.000001,oldtimer[0]-(time.monotonic()-started)),oldtimer[1])
    require(report["passed"],"Terminal canonical serving failure; inspect preserved report")
    return str(report_path)
