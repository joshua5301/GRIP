"""Unexecuted seven-arm cache-only original physical-moment serving adapter for EZ.

Actual endpoint/source/lineage science is fixed. ROOT binds current source and
exclusive outputs; no fit before ROOT and independent shared-prepare admission.
"""
import ast
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import time

KIND = "single_composed_centroid_joint_CE_common_RMS_inverse_physical_seven_arm_validation_v1"
SCIENCE_SHA = "548bcf89153b34d7f78fe5188536873f683079b17db9097f1f126be221760009"
ARMS = ("P0", "linear25", "centroidNy25", "unscaledjoint25", "centeredbalanced25", "physicalOuter25", "composedjoint25")
OLD_ARM_KEYS = {name:name for name in ARMS[:-1]}
SEEDS = [620400, 620401, 620402]


def require(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream: return hashlib.file_digest(stream, "sha256").hexdigest()


def refs(value, found=None):
    found = {} if found is None else found
    if isinstance(value, dict):
        if set(value) == {"path", "sha256"}:
            p,h = value["path"],value["sha256"]
            require(type(p) is str and Path(p).is_absolute() and str(Path(p).resolve()) == p
                and type(h) is str and len(h) == 64 and all(c in "0123456789abcdef" for c in h), "Invalid ABS ref")
            require(p not in found or found[p] == h, "Conflicting ref"); found[p] = h
        for child in value.values(): refs(child, found)
    elif isinstance(value, list):
        for child in value: refs(child, found)
    return found


def write(value, path):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())


def observed(value):
    if isinstance(value, dict): return {k:observed(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [observed(v) for v in value]
    if type(value) is float and not math.isfinite(value): return dict(nonfinite_observation=repr(value))
    return value


def run(protocol_path, protocol_sha256, budget, phase, arm=None, stop=lambda:False):
    started=time.monotonic(); path=Path(protocol_path)
    require(path.is_absolute() and str(path.resolve()) == str(path) and sha(path) == protocol_sha256, "Protocol differs")
    packet=json.loads(path.read_text()); science_ref=packet["scientific_contract"]
    require(science_ref["sha256"] == SCIENCE_SHA and sha(science_ref["path"]) == SCIENCE_SHA, "Science not exactly bound")
    science=json.loads(Path(science_ref["path"]).read_text()); B=science["budgets"][budget]
    require(packet["kind"] == science["kind"] == KIND and phase in ("prepare","evaluate")
        and (arm is None if phase == "prepare" else arm in ARMS), "Domain/phase/arm differs")
    out=Path(packet["prepare_folders"][budget] if phase == "prepare" else packet["evaluation_folders"][budget][arm])
    require(out.is_absolute() and str(out.resolve()) == str(out), "Output must be normalized ABS"); out.mkdir(parents=True,exist_ok=False)
    report=dict(schema=1,kind=KIND,phase=phase,budget=budget,arm=arm,source=packet["source"],scientific_contract=science_ref,
        protocol=dict(path=str(path),sha256=protocol_sha256),passed=False,completed=False,failure=None,gates=[],records=[],
        work=dict(own_PT_load_attempts=0,own_PT_loads=0,dataset_get_attempts=0,dataset_gets=0,graph_pack_attempts=0,graph_packs=0,
            representative_attempts=0,representatives=0,student_fit_attempts=0,student_fits=0,student_epochs=0,route_attempts=0,
            routes=0,sameweight_GCN_routes=0,sameweight_SGC_routes=0,heads=0,adjoints=0,P_updates=0,LowRank_forwards=0,
            factor_factories=0,source_SGC=0,Q_decodes=0,map_fits=0,RMS_refits=0,test_evaluations=0),
        PT_count_scope="Explicit owning payload reads only: prepare3/evaluate1. No dataset loader or graph packing. Original replay_routes adds one selected-checkpoint read per paired replay invocation (GCN and SGC+MLP routes together).",
        secondary="Supplied original cached source-SGC H and exact GCN-selected weights; descriptive SGC+MLP, no independent raw-X MLP or equality to newly packed S²X.")
    work=report["work"]; saved={}; cache={}; torch=None
    raw=out/("shared_physical_serving_cache.pt" if phase == "prepare" else "partial_evaluation_evidence.pt")
    report["raw_evidence"]=dict(path=str(raw),exists=False,bytes=None,sha256=None,hash_unknown=True,hash_error=None,write_completed=False)
    old_handler,old_timer=signal.getsignal(signal.SIGALRM),signal.getitimer(signal.ITIMER_REAL)
    def peaks():
        p=dict(seconds=time.monotonic()-started,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            peak_allocated_bytes=0,peak_reserved_bytes=0)
        if torch is not None and torch.cuda.is_initialized():
            p.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(0),peak_reserved_bytes=torch.cuda.max_memory_reserved(0))
        return p
    def guard():
        require(not stop() and all(v <= science["resources"][k] for k,v in peaks().items()), "Bounded terminal stop/resources")
        return False
    def alarm(signum,frame): raise TimeoutError("Seven-arm reuse serving300s deadline")
    def gate(name,ok,**detail):
        report["gates"].append(dict(name=name,passed=bool(ok),**detail)); require(ok,name)
    def read(reference):
        require(sha(reference["path"]) == reference["sha256"], "Pinned JSON differs"); return json.loads(Path(reference["path"]).read_text())
    def pins():
        expected=packet["readonly_files_sha256"]
        require(all(expected.get(p) == h for p,h in refs(dict(packet=packet,science=science)).items()), "Required ref omitted")
        actual={}
        for p,h in expected.items():
            guard();require(Path(p).is_absolute() and str(Path(p).resolve()) == p, "Pin not normalized ABS");actual[p]=sha(p)
        require(actual == expected, "Readonly bytes changed");return actual
    try:
        signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,max(1e-6,300-(time.monotonic()-started)))
        report["readonly_entry"]=pins()
        import torch
        from src import joint_mean_ce as joint
        from src.citation_graph_factor import _transform
        from src.kernel_mean_ce import _cpu,_plain,_seal
        from src.research_loop import implementation_provenance
        from src.shared_features import _tensor_identity
        from src.sweep_utils import representative
        from src.transforms import FeatureTransform
        report["source_entry"]=implementation_provenance();gate("current_source_and_entrypoint",report["source_entry"] == packet["source"]
            and Path(packet["entrypoint"]["path"]).resolve() == Path(__file__).resolve())
        bridge=read(packet["prerequisite_admission"])
        gate("ROOT_current_serving_static_bridge",bridge["passed"] is True and bridge["source"]==packet["source"]
            and bridge["entrypoint"]==packet["entrypoint"] and bridge["scientific_contract"]==science_ref)
        torch.set_num_threads(4)
        if torch.get_num_interop_threads() != 1: torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision("highest")
        torch.cuda.init();torch.cuda.reset_peak_memory_stats(0);device=torch.device("cuda:0")
        report["runtime"]=joint._runtime(device);gate("frozen_original_runtime",report["runtime"] == B["original_runtime"])
        def load(reference,role):
            guard();work["own_PT_load_attempts"]+=1;value=torch.load(reference["path"],map_location="cpu",weights_only=False)
            work["own_PT_loads"]+=1;saved["loaded_"+role]=value;saved["loaded_"+role]=_cpu(value);return value
        def identity(value):
            if torch.is_tensor(value) and value.layout == torch.sparse_csr:
                return dict(shape=list(value.shape),dtype=str(value.dtype),layout=str(value.layout),
                    crow=_tensor_identity(value.crow_indices()),col=_tensor_identity(value.col_indices()),values=_tensor_identity(value.values()))
            return _tensor_identity(value)
        def tensor_metadata(value):
            if torch.is_tensor(value): return identity(value)
            if isinstance(value,dict): return {key:tensor_metadata(v) for key,v in value.items()}
            if isinstance(value,(list,tuple)): return [tensor_metadata(v) for v in value]
            return value
        def cache_ids(value):
            return dict(graph={k:identity(v) for k,v in value["graph"].items()},masks={k:identity(v) for k,v in value["masks"].items()},
                H=identity(value["H"]),Q=identity(value["Q"]),arms={a:{k:identity(v) for k,v in row.items()} for a,row in value["arms"].items()})
        if phase == "prepare":
            legacy=B["historical147"]
            prior,root,peer=(read(legacy[role]) for role in ("report","ROOT","independent"))
            gate("historical147_actual_whole_cache_admitted",prior["passed"] is True and root["passed"] is True and peer["passed"] is True
                and root["actual_report"]==peer["actual_report"]==legacy["report"]
                and prior["shared_cache"]==root["shared_cache"]==peer["shared_cache"]==legacy["cache"]
                and prior["source"]==prior["source_entry"]==prior["source_exit"]==B["historical147_producer"]
                and prior["scientific_contract"]==B["historical147_science"])
            historical=load(legacy["cache"],"historical147_cache")
            gate("whole_historical147_payload_exact",historical["schema"]==1 and historical["kind"]==B["historical147_kind"]
                and historical["budget"]==budget and historical["source"]==B["historical147_producer"]
                and historical["scientific_contract"]==B["historical147_science"]
                and historical["identities"]==prior["cache_identities"]==B["historical147_identities"]==cache_ids(historical)
                and set(historical["masks"])=={"train","val"} and set(historical["arms"])==set(OLD_ARM_KEYS.values())
                and _seal(historical["upstream_providers"])==B["historical147_upstream_providers_seal"])
            admitted=packet["composed_candidate"][budget]
            source_refs=B["original135"];endpoint_refs=admitted["endpoint25"]
            source_report=read(source_refs["report"]);native_report=read(endpoint_refs["report"])
            for role in ("ROOT","independent"):
                source_accept=read(source_refs[role]);native_accept=read(endpoint_refs[role])
                gate(role+"_actual_source135_and_EZ25_admitted",source_accept["passed"] is True and native_accept["passed"] is True
                    and source_accept["budgets"][budget]["report" if role=="ROOT" else "actual_report"]==source_refs["report"]
                    and source_accept["budgets"][budget]["arrays"]==source_refs["arrays"]
                    and native_accept["budgets"][budget]["report"]==endpoint_refs["report"])
            gate("actual_original135_owner",source_report["passed"] is True and source_report["budget"]==budget
                and source_report["source"]==source_report["source_entry"]==source_report["source_exit"]==B["source135"]
                and source_report["scientific_contract"]==science["required_refs"]["source135_science"]
                and source_report["raw_evidence"]["sha256"]==source_refs["arrays"]["sha256"])
            gate("EZ_actual_complete25_physical_composed_joint",native_report["passed"] is True and native_report["completed"] is True
                and native_report["kind"]=="single_composed_centroid_joint_CE_native_qualification_and_continuation_v1"
                and set(native_report["context"])=={"schema","mode","policy","source_refs","native_parameter_digests","asset_descriptors","native_origin","feature_contract"}
                and native_report["context"]["mode"]==B["EZ_mode"]
                and native_report["core_work"]["P_updates"]==25 and native_report["core_work"]["head"]==26
                and native_report["core_work"]["physical_forward"]==27 and native_report["core_work"]["reattachment"]==1
                and native_report["core_work"]["endpoint_map"]==26 and native_report["core_work"]["G_map"]==25
                and native_report["mode"]=="continue" and native_report["budget"]==budget
                and native_report["source"]==native_report["source_entry"]==native_report["source_exit"]==admitted["native_producer"]
                and native_report["scientific_contract"]==science["required_refs"]["EZ_native_science"]
                and native_report["context"]["asset_descriptors"]==B["assets"]
                and native_report["context"]["source_refs"]["data_digest"]==B["original_data_digest"]
                and native_report["core_artifacts"]["checkpoints"]["step_000025.pt"]==endpoint_refs["checkpoint"])
            source_payload=load(source_refs["arrays"],"own_original135_source")
            gate("original135_payload_header",source_payload["schema"]==1 and source_payload["kind"]==source_report["kind"]
                and source_payload["budget"]==budget and source_payload["source"]==source_report["source"]
                and source_payload["scientific_contract"]==source_report["scientific_contract"])
            captured=source_payload["captured"]
            n,d,c,k=(B["dimensions"][x] for x in ("nodes","dimension","classes","cells"))
            transform=FeatureTransform(**{name:value.to(device) if torch.is_tensor(value) else value for name,value in captured["z_transform"]["transform"].items()})
            gate("original_RMS_H_Q_hard_and_cache_link",_transform(transform,d)==B["assets"]["transform"]
                and all(identity(v)==B["assets"][name] for name,v in (("H",captured["H_cache"]["h"]),("Q",captured["Q"]),("assignment",captured["hard"])))
                and identity(captured["H_cache"]["h"])==historical["identities"]["H"]
                and identity(captured["Q"])==historical["identities"]["Q"])
            endpoint=load(endpoint_refs["checkpoint"],"own_EZ_E25")
            gate("new_own_EZ_physical_E25",type(endpoint["step"]) is int and endpoint["step"]==25
                and endpoint["kind"]=="composed_joint_head_evaluation_v1"
                and endpoint["record_digest"]==_seal({key:value for key,value in endpoint.items() if key!="record_digest"})
                and endpoint["context_digest"]==_seal(native_report["context"])
                and endpoint["source_admission_digest"]==_seal(native_report["context"]["source_refs"]["source_admission"])
                and endpoint["initial_parameter_digests"]==native_report["context"]["native_parameter_digests"]
                and all(_plain(endpoint[key])==value for key,value in native_report["evaluation_summaries"]["25"]["scalars"].items())
                and all(identity(endpoint[key])==value for key,value in native_report["evaluation_summaries"]["25"]["identities"].items()))
            M=endpoint["physical_moments"];saved["own_EZ_physical_M25"]=_cpu(M)
            gate("own_original_D_physical25_domain",M.dtype==torch.float64 and list(M.shape)==[k,1+d+c]
                and not M.requires_grad and bool(torch.isfinite(M).all()) and bool((M[:,0]>0).all())
                and bool((M[:,1+d:]>=0).all()) and bool((M[:,1+d:].sum(0)>0).all()) and bool((M[:,1+d:].sum(1)>0).all()))
            work["representative_attempts"]+=1;x,y,mass=representative(M,transform,d,device);work["representatives"]+=1
            saved["returned_EZ_representative"]=_cpu(dict(X=x,Q=y,mass=mass))
            rows={name:_cpu(historical["arms"][oldname]) for name,oldname in OLD_ARM_KEYS.items()}
            rows["composedjoint25"]=_cpu(dict(X=x,Q=y,uniform_weights=torch.full_like(mass,1/k)));saved["readouts"]=_cpu(rows)
            report["new_readout"]=dict(step=25,moment=identity(M),returned={name:identity(v) for name,v in rows["composedjoint25"].items()})
            gate("EZ_common_original_RMS_readout_domain",list(x.shape)==[k,d] and list(y.shape)==[k,c] and x.dtype==y.dtype==torch.float32
                and rows["composedjoint25"]["uniform_weights"].dtype==torch.float64
                and all(bool(torch.isfinite(v).all()) for v in rows["composedjoint25"].values()) and bool((y>=0).all()) and bool((y.sum(1)>0).all()))
            gate("new_uniform_weights_exact_original_FP64_P0",rows["composedjoint25"]["uniform_weights"].dtype==torch.float64
                and identity(rows["composedjoint25"]["uniform_weights"])==historical["identities"]["arms"]["P0"]["uniform_weights"])
            report["historical_arm_key_mapping"]=OLD_ARM_KEYS
            gate("old_six_tensor_readouts_cloned_unchanged",all({key:identity(v) for key,v in rows[name].items()}==historical["identities"]["arms"][oldname]
                for name,oldname in OLD_ARM_KEYS.items()))
            cache=dict(schema=1,kind=KIND,budget=budget,source=packet["source"],scientific_contract=science_ref,
                upstream_providers=dict(historical147=dict(admission=legacy,source=historical["source"],upstream_providers=historical["upstream_providers"],identities=historical["identities"],arm_key_mapping=OLD_ARM_KEYS),
                    original135=source_refs,original135_producer=source_report["source"],composedjoint25=endpoint_refs,composedjoint25_producer=native_report["source"]),
                graph=_cpu(historical["graph"]),masks=_cpu(historical["masks"]),H=_cpu(historical["H"]),Q=_cpu(historical["Q"]),arms=rows)
            cache["identities"]=cache_ids(cache);report["cache_identities"]=cache["identities"]
            gate("unchanged_shared_graph_H_Q_masks",all(cache["identities"][key]==historical["identities"][key] for key in ("graph","masks","H","Q")))
            gate("all_three_own_inputs_unchanged",all(_seal(tensor_metadata(value))==_seal(tensor_metadata(saved["loaded_"+name])) for value,name in
                ((historical,"historical147_cache"),(source_payload,"own_original135_source"),(endpoint,"own_EZ_E25"))))
            gate("prepare_exact_reuse_counts",work["own_PT_loads"]==work["own_PT_load_attempts"]==3
                and work["representative_attempts"]==work["representatives"]==1
                and all(work[key]==0 for key in ("dataset_get_attempts","dataset_gets","graph_pack_attempts","graph_packs","student_fit_attempts","student_fits")))
        else:
            admission=packet["shared"][budget];prior,root,peer=(read(admission[x]) for x in ("report","ROOT","independent"))
            gate("shared_cache_ROOT_independent_admitted",prior["passed"] is True and root["passed"] is True and peer["passed"] is True
                and prior["shared_cache"] == root["shared_cache"] == peer["shared_cache"] == admission["cache"]
                and root["actual_report"] == peer["actual_report"] == admission["report"])
            cache=load(admission["cache"],"shared_cache")
            gate("shared_cache_whole_identity",cache["source"] == prior["source"] == packet["source"] and cache["scientific_contract"] == science_ref
                and cache["budget"] == budget and cache["kind"] == KIND and cache["identities"] == prior["cache_identities"] == cache_ids(cache)
                and set(cache["arms"]) == set(ARMS) and set(cache["masks"]) == {"train","val"})
            from src.evaluation import fit_gcn_diagnostic
            from src.student_routes import replay_routes
            settings=dict(B["original_recipe"]);gate("frozen_shared_student_cohort",science["student_seeds"] == SEEDS and settings.pop("input_scale") == 1.0)
            graph={k:t.to(device) for k,t in cache["graph"].items()};masks={k:t.to(device) for k,t in cache["masks"].items()}
            q,h=cache["Q"].to(device),cache["H"].to(device);x,y,mass=(cache["arms"][arm][k].to(device) for k in ("X","Q","uniform_weights"))
            gate("admitted_uniform_and_valonly_labels",bool(torch.equal(mass,torch.full_like(mass,1/len(mass))))
                and bool((graph["y"][~(masks["train"]|masks["val"])] == -1).all()))
            fit_folder=out/("step_0" if arm == "P0" else "step_25")
            for seed in SEEDS:
                guard();work["student_fit_attempts"]+=1;fit=fit_gcn_diagnostic(x,y,mass,graph,q,masks,seed,folder=fit_folder,training_adjacency=None,stop=guard,**settings)
                work["student_fits"]+=1;record=dict(seed=seed,fit=fit);report["records"].append(record)
                fit_paths={key:fit_folder/f"seed_{seed}{suffix}" for key,suffix in (("json",".json"),("csv","_epochs.csv"),("selected","_selected.pt"))}
                record["fit_files"]={key:dict(path=str(p),sha256=sha(p)) for key,p in fit_paths.items()}
                history=list(csv.DictReader(fit_paths["csv"].open()));work["student_epochs"]+=len(history);fit_JSON=json.loads(fit_paths["json"].read_text())
                best=max(float(row["val_acc"]) for row in history);first=next(row for row in history if float(row["val_acc"]) == best)
                gate("fit_"+str(seed)+"_full_firstmax_original_recipe",[int(row["epoch"]) for row in history] == list(range(1,settings["epochs"]+1))
                    and fit_JSON["result"] == fit and fit_JSON["recipe"]["settings"] == settings and fit_JSON["recipe"]["test_enabled"] is False
                    and fit["epoch"] == int(first["epoch"]) and fit["val_acc"] == best and not any("test" in key for key in fit))
                route_path=fit_folder/f"seed_{seed}_sameweights_routes.json";work["route_attempts"]+=1
                route=replay_routes(fit_paths["selected"],graph,h.float(),dict(val=masks["val"]),settings,route_path,seed=seed,stop=guard)
                work["routes"]+=1;work["sameweight_GCN_routes"]+=1;work["sameweight_SGC_routes"]+=1
                record["routes"]=route;record["route_file"]=dict(path=str(route_path),sha256=sha(route_path))
                route_JSON=json.loads(route_path.read_text());gate("route_"+str(seed)+"_same_selected_weights",route_JSON["result"] == route and route["gcn_val_acc"] == fit["val_acc"]
                    and route["epoch"] == fit["epoch"] and route_JSON["recipe"]["source_fingerprint"] == fit_JSON["fingerprint"]
                    and route_JSON["recipe"]["test_enabled"] is False and not any("test" in key for key in route));guard()
            report["shared_cache"]=admission["cache"];report["internal_route_checkpoint_reads"]=work["routes"]
            gate("evaluate_exact_counts",work["own_PT_loads"] == work["own_PT_load_attempts"] == 1 and work["student_fits"] == work["student_fit_attempts"] == 3
                and work["student_epochs"] == settings["epochs"]*3 and work["routes"] == work["route_attempts"] == 3)
        gate("no_new_condensation_or_test_work",all(work[key] == 0 for key in ("heads","adjoints","P_updates","LowRank_forwards",
            "factor_factories","source_SGC","Q_decodes","map_fits","RMS_refits","test_evaluations")))
        torch.cuda.synchronize(device);guard();report["completed"]=True
    except BaseException as error:
        report["failure"]=dict(type=type(error).__name__,message=str(error));report["inflight_interiors"]="Unknown; only owning returned evidence and completed calls are counted"
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        try:
            evidence=cache if phase == "prepare" and report["completed"] else dict(partial_unqualified=True,returned=saved,cache=cache,records=report["records"])
            if torch is not None:
                raw_info=report["raw_evidence"]
                try:
                    with raw.open("xb") as stream:torch.save(evidence,stream);stream.flush();os.fsync(stream.fileno())
                    raw_info["write_completed"]=True
                except BaseException as error:report["raw_write_error"]=repr(error)
                try:
                    raw_info["exists"]=raw.exists()
                    if raw.exists():
                        raw_info["bytes"]=raw.stat().st_size
                        try:raw_info["sha256"]=sha(raw);raw_info["hash_unknown"]=False
                        except BaseException as error:raw_info["hash_error"]=repr(error)
                except BaseException as error:report["raw_metadata_error"]=repr(error)
                if phase == "prepare" and report["completed"]:report["shared_cache"]=dict(path=str(raw),sha256=raw_info["sha256"])
            try:
                report["readonly_exit"]=pins();report["source_exit"]=implementation_provenance();require(report["source_exit"] == packet["source"], "Exit source differs")
            except BaseException as error:report["exit_verification_error"]=repr(error)
            report["resources"]=peaks();report["passed"]=bool(report["completed"] and report["failure"] is None
                and not any(k in report for k in ("raw_write_error","raw_metadata_error","exit_verification_error"))
                and report["raw_evidence"]["write_completed"] and not report["raw_evidence"]["hash_unknown"]
                and all(v <= science["resources"][k] for k,v in report["resources"].items()))
            rp=out/("prepare_report.json" if phase == "prepare" else "evaluation_report.json");write(observed(report),rp);final=peaks()
            if report["passed"] and any(v > science["resources"][k] for k,v in final.items()):
                report.update(passed=False,resources=final,failure=dict(type="FinalSerializationResourceBoundary",message="Preserve failure; no retry"))
                with rp.open("w") as stream:json.dump(observed(report),stream,indent=2,allow_nan=False);stream.write("\n");stream.flush();os.fsync(stream.fileno())
        finally:signal.signal(signal.SIGALRM,old_handler);signal.setitimer(signal.ITIMER_REAL,*old_timer)
    require(report["passed"],"Terminal seven-arm physical serving failure; no retry/recipe/seed/secondary rescue")
    return str(rp)
