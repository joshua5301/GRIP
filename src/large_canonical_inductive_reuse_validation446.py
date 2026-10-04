"""Validation-only K446 canonical readouts on an unchanged owning source124 cache.

Preparation loads only one admitted old cache plus four accepted readout artifacts.
No raw IO, scaler, graph pack, source/validation SGC, target or moment regeneration.
Original GCN graph products occur in unchanged student fitting and route replay.
"""
import csv
import json
import math
import os
import platform
import resource
import signal
import time
from pathlib import Path

import numpy as np
import torch

from src.inductive_evaluation import fit_inductive_gcn
from src.io import cpu_state
from src.large_canonical_inductive_validation import _identities, _graph_domain, _refs
from src.large_kernel_mean_source_capture import _matrix, _observed, _require, _sha
from src.shared_features import _tensor_identity
from src.student_routes import replay_routes

KIND = "large_canonical_inductive_reuse_validation446_v1"
CONTRACT_SHA = "ec1ea63c1f3a154d5682dd6c7aee082688de9e8f1c5e2dcae829d0ad548ebf83"
ARMS = ("own_canonical_P0", "uniform_linear25", "uniform_centroid_Ny25", "meanPhi25")


def run(protocol_path, protocol_sha256, phase, arm=None, stop=lambda: False):
    from src.research_loop import implementation_provenance

    _require(callable(stop) and _sha(protocol_path)==protocol_sha256, "Frozen invocation changed")
    packet=json.loads(Path(protocol_path).read_text()); ref=packet["scientific_contract"]
    _require(ref["sha256"]==CONTRACT_SHA and _sha(ref["path"])==CONTRACT_SHA, "Scientific contract changed")
    contract=json.loads(Path(ref["path"]).read_text())
    _require(type(packet["schema"]) is int and packet["schema"]==1 and packet["kind"]==contract["kind"]==KIND
        and packet["budget"]==contract["budget"]=="flickr446" and packet["test_enabled"] is contract["test_enabled"] is False
        and packet["dimensions"]==contract["dimensions"]
        and packet["runtime"]==contract["runtime"] and packet["resource_limits"]==contract["resource_limits"]
        and packet["source"]==implementation_provenance() and phase in ("prepare","evaluate")
        and (arm is None if phase=="prepare" else arm in ARMS), "Frozen phase/source/domain differs")
    required=_refs(contract); required.update(_refs(ref))
    if phase=="evaluate":
        for key in ("shared_cache","shared_prepare_report","shared_prepare_acceptance"):
            required.update(_refs(packet[key]))
    _require(all(Path(p).is_absolute() and str(Path(p).resolve())==p
        and type(s) is str and len(s)==64 and packet["readonly_files_sha256"].get(p)==s
        for p,s in required.items()),
        "Required readonly admission omitted")
    folder=Path(packet["prepare_folder"] if phase=="prepare" else packet["evaluation_folders"][arm])
    _require(folder.is_absolute() and not folder.exists(), "Fresh exclusive namespace required")
    folder.mkdir(parents=True); started=time.monotonic(); cache={}; failure=None; gpu=False
    counts={name:0 for name in contract["expected_prepare_counts"]}
    counts.update(old_source124_serving_cache_load_attempts=0,canonical_readout_load_attempts=0,
        total_PT_payload_load_attempts=0,shared_cache_write_attempts=0,own_shared_cache_load_attempts=0,
        own_shared_cache_loads=0,fit_attempts=0,student_fits=0,student_epochs=0,
        completed_fit_CSV_rows=0,route_attempts=0,route_calls=0,validation_routes=0)
    report=dict(schema=1,kind=KIND,phase=phase,arm=arm,passed=False,source=packet["source"],
        scientific_contract=ref,protocol=dict(path=str(Path(protocol_path).resolve()),sha256=protocol_sha256),
        counts=counts,records=[],test_enabled=False,goal_complete=False,
        secondary=contract["student_recipe"]["secondary"],raw_H_equivalence_claim=False)
    limits=contract["resource_limits"]
    def guard():
        _require(not stop() and time.monotonic()-started<=300
            and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=limits["peak_RSS_bytes"], "Terminal stop/time/RSS")
        if gpu:
            _require(torch.cuda.max_memory_allocated()<=limits["peak_allocated_bytes"]
                and torch.cuda.max_memory_reserved()<=limits["peak_reserved_bytes"], "Terminal GPU resource bound")
        return False
    def pins(label):
        report[label]={}
        for path,digest in {**packet["readonly_files_sha256"],**packet["source"]["files"]}.items():
            report[label][path]=_sha(path,guard)
            _require(report[label][path]==digest, "Readonly input/source changed: "+path)
        _require(implementation_provenance()==packet["source"], "Whole source changed")
    def read(reference):
        _require(_sha(reference["path"],guard)==reference["sha256"], "Pinned metadata changed")
        return json.loads(Path(reference["path"]).read_text())
    def expire(signum,frame): raise InterruptedError("Frozen validation phase300s deadline")
    old_handler=signal.getsignal(signal.SIGALRM);old_timer=signal.getitimer(signal.ITIMER_REAL)
    try:
        signal.signal(signal.SIGALRM,expire);signal.setitimer(signal.ITIMER_REAL,min(300,old_timer[0]) if old_timer[0] else 300)
        runtime=contract["runtime"]
        _require(platform.python_version()==runtime["python"] and str(torch.__version__)==runtime["torch"]
            and np.__version__==runtime["numpy"] and str(torch.get_default_dtype())==runtime["default_dtype"]
            and not torch.is_autocast_enabled("cuda") and all(os.environ.get(k)==runtime[k]
            for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","CUBLAS_WORKSPACE_CONFIG","CUDA_VISIBLE_DEVICES")),
            "Frozen runtime differs")
        torch.set_num_threads(4)
        if torch.get_num_interop_threads()!=1: torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision("highest");torch.use_deterministic_algorithms(True)
        torch.cuda.init();_require(torch.cuda.get_device_name(0)==runtime["GPU"], "Frozen GPU differs")
        torch.cuda.reset_peak_memory_stats();gpu=True;device=torch.device("cuda:0");pins("entry_files_sha256")
        settings=contract["student_recipe"]["settings"]
        _require(read(contract["student_recipe"]["file"])["recipe"]["settings"]==settings
            and settings==dict(epochs=300,eval_every=10,hidden=256,dropout=.5,lr=.01,weight_decay=.0005)
            and contract["student_seeds"]==[20400,20401,20402]
            and all(type(s) is int for s in contract["student_seeds"])
            and [item["id"] for item in contract["arms"]]==list(ARMS), "Original recipe/cohort/arms differ")
        reuse=contract["reuse_source124"]
        graph_provider={k:reuse[k] for k in ("producer","arrays","actual_report","acceptance",
            "independent_acceptance","protocol","scientific_contract")}
        expected_providers={item["id"]:dict(producer=item["producer"],arrays=item["arrays"],
            actual_report=item["actual_report"],actual_acceptance=item["actual_acceptance"],step=item["step"])
            for item in contract["arms"]}
        if phase=="prepare":
            cache.update(schema=1,kind=KIND,budget="flickr446",source=packet["source"],scientific_contract=ref,
                arms={},arm_providers={},graph_provider=graph_provider)
            prior=read(reuse["actual_report"]);accepted=read(reuse["acceptance"]);old_packet=read(reuse["protocol"])
            _require(prior["passed"] is True and prior["phase"]=="prepare" and accepted["passed"] is True
                and prior["source"]==accepted["source"]==old_packet["source"]==reuse["producer"]
                and prior["scientific_contract"]==accepted["scientific_contract"]==old_packet["scientific_contract"]==reuse["scientific_contract"]
                and prior["shared_cache"]==accepted["shared_cache"]==reuse["arrays"]
                and accepted["shared_prepare_report"]==reuse["actual_report"]
                and accepted["protocol"]==reuse["protocol"], "Source124 owning cache admission differs")
            counts["old_source124_serving_cache_load_attempts"]+=1;counts["total_PT_payload_load_attempts"]+=1
            old=torch.load(reuse["arrays"]["path"],map_location="cpu",weights_only=False)
            counts["old_source124_serving_cache_loads"]+=1;counts["total_PT_payload_loads"]+=1
            report["loaded_graph_provider_header"]={k:old.get(k) for k in ("schema","kind","source","scientific_contract","source_tokens")}
            observed=_identities(old);report["loaded_graph_provider_identities"]=observed
            _require(type(old["schema"]) is int and old["schema"]==1
                and old["kind"]=="large_canonical_inductive_validation_v1" and old["source"]==reuse["producer"]
                and old["scientific_contract"]==reuse["scientific_contract"]
                and old["identities"]==observed==prior["cache_identities"]==accepted["cache_identities"]
                ==reuse["expected_old_cache_identities"] and old["source_tokens"]==reuse["source_tokens"],
                "Whole source124 cache content/lineage differs; no fallback")
            keys=("train_graph","val_graph","train_mask","val_mask","train_H","val_H","Q","source_tokens")
            _require(reuse["shared_keys"]==list(keys), "Only eight original shared keys may be copied")
            cache.update({k:cpu_state(old[k]) for k in keys});del old;guard()
            for item in contract["arms"]:
                actual=read(item["actual_report"]);accepted=read(item["actual_acceptance"])
                _require(actual["passed"] is True and accepted["passed"] is True
                    and actual["source"]==accepted["source"]==item["producer"]
                    and accepted["actual_report"]==item["actual_report"]
                    and accepted["actual_arrays" if item["step"]==0 else "canonical_endpoint25"]==item["arrays"],
                    "Canonical producer/admission differs")
                counts["canonical_readout_load_attempts"]+=1;counts["total_PT_payload_load_attempts"]+=1
                supplied=torch.load(item["arrays"]["path"],map_location="cpu",weights_only=False)
                cache["arms"][item["id"]]={key:cpu_state(supplied[old]) for key,old in item["keys"].items()}
                counts["canonical_readout_loads"]+=1;counts["total_PT_payload_loads"]+=1
                cache["arm_providers"][item["id"]]=expected_providers[item["id"]]
                report["returned_canonical_providers"]=dict(cache["arm_providers"])
                if item["step"]==0:
                    _require(item["producer"]==contract["capture_producer"]
                        and item["actual_report"]==contract["capture_report"]
                        and item["actual_acceptance"]==contract["capture_acceptance"]
                        and item["arrays"]==contract["capture_arrays"]
                        and actual["new_data_digest"]==accepted["new_data_digest"]==contract["new_source_data_digest"]
                        and actual["reused_coordinate_Q_producer"]==accepted["reused_coordinate_Q_producer"]==contract["coordinate_Q_producer"]
                        and actual["rawgraph_H_admission_producer"]==accepted["rawgraph_H_admission_producer"]==contract["rawgraph_H_admission_producer"]
                        and _tensor_identity(supplied["Q"])==_tensor_identity(cache["Q"])
                        ==actual["array_descriptors"]["Q"]==accepted["asset_descriptors"]["Q"]
                        and _tensor_identity(cache["train_H"])==actual["array_descriptors"]["saved_H_FP32"]
                        ==accepted["asset_descriptors"]["saved_H_FP32"], "Captured source128 Q/H provenance differs")
                    descriptors={key:actual["array_descriptors"][old] for key,old in item["keys"].items()}
                else:
                    report.setdefault("returned_endpoint_headers",{})[item["id"]]={k:supplied.get(k) for k in ("source","budget","arm","step","capture_arrays","new_data_digest")}
                    _require(type(supplied["step"]) is int and supplied["step"]==25
                        and supplied["source"]==item["producer"] and supplied["budget"]=="flickr446"
                        and supplied["arm"]==item["id"] and supplied["capture_arrays"]==contract["capture_arrays"]
                        and supplied["new_data_digest"]==actual["new_data_digest"]==contract["new_source_data_digest"]
                        and actual["arm"]==accepted["arm"]==item["id"]
                        and actual["budget"]==accepted["budget"]=="flickr446"
                        and type(actual["actual_P_updates"]) is int and actual["actual_P_updates"]==25
                        and type(accepted["P_updates"]) is int and accepted["P_updates"]==25
                        and actual["capture_source"]==accepted["capture_source"]==contract["capture_producer"],
                        "Canonical25 owning producer/native origin differs")
                    descriptors={key:actual["canonical_descriptors"][key] for key in item["keys"]}
                    _require(descriptors=={key:accepted["canonical_descriptors"][key] for key in item["keys"]},
                        "Actual25 ROOT descriptor admission differs")
                _require(descriptors==item["expected_descriptors"], "Frozen canonical descriptor differs")
                del supplied;guard()
        else:
            prior=read(packet["shared_prepare_report"]);accepted=read(packet["shared_prepare_acceptance"])
            _require(prior["passed"] is True and prior["phase"]=="prepare" and accepted["passed"] is True
                and prior["source"]==accepted["source"]==packet["source"]
                and prior["scientific_contract"]==accepted["scientific_contract"]==ref
                and prior["shared_cache"]==accepted["shared_cache"]==packet["shared_cache"]
                and accepted["shared_prepare_report"]==packet["shared_prepare_report"], "Shared cache not ROOT-admitted")
            counts["own_shared_cache_load_attempts"]+=1;counts["total_PT_payload_load_attempts"]+=1
            cache=torch.load(packet["shared_cache"]["path"],map_location="cpu",weights_only=False)
            counts["own_shared_cache_loads"]+=1;counts["total_PT_payload_loads"]+=1
            report["loaded_new_cache_header"]={k:cache.get(k) for k in ("schema","kind","budget","source","scientific_contract","graph_provider","source_tokens")}
            _require(type(cache["schema"]) is int and cache["schema"]==1 and cache["kind"]==KIND
                and cache["budget"]=="flickr446" and cache["source"]==packet["source"] and cache["scientific_contract"]==ref
                and cache["identities"]==_identities(cache)==prior["cache_identities"]
                and prior["graph_provider"]==graph_provider, "Owning shared-cache integrity differs")
        _require(set(cache["arms"])==set(ARMS) and cache["graph_provider"]==graph_provider
            and cache["arm_providers"]==expected_providers and cache["source_tokens"]==reuse["source_tokens"],
            "Shared canonical arms or retained producer lineage differs")
        for name,n in (("train",44625),("val",22312)):
            _graph_domain(cache[name+"_graph"],cache[name+"_mask"],n)
            _matrix(cache[name+"_H"],(n,500),torch.float32)
        for item in contract["arms"]:
            row=cache["arms"][item["id"]]
            _matrix(row["X"],(446,500),torch.float32);_matrix(row["Qbar"],(446,7),torch.float32)
            _matrix(row["uniform_weights"],(446,),torch.float64)
            _require({k:_tensor_identity(t) for k,t in row.items()}==item["expected_descriptors"]
                and bool((row["Qbar"]>=0).all()) and bool((row["Qbar"].sum(1)>0).all())
                and bool((row["uniform_weights"]>0).all()), "Canonical readout domain/bytes differ")
        _matrix(cache["Q"],(44625,7),torch.float64)
        _require(bool((cache["Q"]>=0).all()) and bool((cache["Q"].sum(1)>0).all()), "Captured rawQ domain differs")
        cache["identities"]=_identities(cache)
        shared={k:v for k,v in cache["identities"].items() if k!="arms"}
        _require(shared==reuse["expected_shared_descriptors"], "Copied shared source124 bytes differ")
        report["cache_identities"]=cache["identities"];report["shared_identities"]=shared
        report["graph_provider"]=graph_provider
        report["coordinate_Q_producer"]=contract["coordinate_Q_producer"]
        report["capture_producer"]=contract["capture_producer"]
        report["arm_providers"]=cache["arm_providers"];report["source_tokens"]=cache["source_tokens"];guard()
        if phase=="evaluate":
            report.update(shared_cache=packet["shared_cache"],shared_prepare_report=packet["shared_prepare_report"],
                shared_prepare_acceptance=packet["shared_prepare_acceptance"])
            train={k:t.to(device) for k,t in cache["train_graph"].items()}
            val={k:t.to(device) for k,t in cache["val_graph"].items()}
            train_mask=cache["train_mask"].to(device);val_mask=cache["val_mask"].to(device)
            val_H=cache["val_H"].to(device);x,y,w=[cache["arms"][arm][k].to(device) for k in ("X","Qbar","uniform_weights")]
            students=folder/"students"
            for seed in contract["student_seeds"]:
                guard();counts["fit_attempts"]+=1
                fit=fit_inductive_gcn(x,y,w,train,(val,None),testing=None,seed=seed,settings=settings,
                    folder=students,training_adjacency=None,stop=guard,weighting="uniform",train_mask=train_mask)
                counts["student_fits"]+=1;counts["students"]+=1;counts["student_epochs"]+=300
                record=dict(seed=seed,fit=dict(fit));report["records"].append(record)
                files={key:students/f"seed_{seed}{suffix}" for key,suffix in (("json",".json"),("csv","_epochs.csv"),("selected","_selected.pt"))}
                record["fit_files"]={key:dict(path=str(p),sha256=_sha(p,guard)) for key,p in files.items()}
                with files["csv"].open() as stream: rows=list(csv.DictReader(stream))
                counts["completed_fit_CSV_rows"]+=len(rows);saved=json.loads(files["json"].read_text())
                best=max(float(r["val_acc"]) for r in rows);first=next(r for r in rows if float(r["val_acc"])==best)
                _require([int(r["epoch"]) for r in rows]==list(range(10,301,10))
                    and all(int(r["val_nodes"])==22312 for r in rows) and fit["epoch"]==int(first["epoch"])
                    and fit["val_acc"]==best and fit["last_epoch"]==300 and fit["train_nodes"]==44625
                    and saved["result"]==fit and saved["recipe"]["settings"]==settings
                    and saved["recipe"]["seed"]==seed and saved["recipe"]["weighting"]=="uniform"
                    and saved["recipe"]["test_enabled"] is False and not any("test" in k for k in fit)
                    and all(not isinstance(v,float) or math.isfinite(v) for v in fit.values()), "Fresh selected fit/CSV differs")
                guard();counts["route_attempts"]+=1;route_path=students/f"seed_{seed}_validation_routes.json"
                route=replay_routes(files["selected"],val,val_H,{"val":val_mask},settings,route_path,seed=seed,stop=guard)
                counts["route_calls"]+=1;counts["validation_routes"]+=2;record["routes"]=dict(route)
                record["route_file"]=dict(path=str(route_path),sha256=_sha(route_path,guard))
                saved_route=json.loads(route_path.read_text())
                _require(route["gcn_val_acc"]==fit["val_acc"] and route["gcn_val_ce"]==fit["val_ce"]
                    and route["epoch"]==fit["epoch"] and saved_route["result"]==route
                    and saved_route["recipe"]["source_fingerprint"]==saved["fingerprint"]
                    and saved_route["recipe"]["settings"]==settings and saved_route["recipe"]["test_enabled"] is False
                    and not any("test" in k for k in route) and all(not isinstance(v,float) or math.isfinite(v) for v in route.values()),
                    "Same selected-weight route differs")
                guard()
            _require(counts["student_fits"]==3 and counts["student_epochs"]==900
                and counts["completed_fit_CSV_rows"]==90 and counts["validation_routes"]==6, "Incomplete arm")
    except BaseException as exc:
        failure=exc;report["error"]=dict(type=type(exc).__name__,message=str(exc))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old_handler)
        if old_timer[0]>0:signal.setitimer(signal.ITIMER_REAL,max(1e-6,old_timer[0]-(time.monotonic()-started)),old_timer[1])
        if phase=="prepare":
            path=folder/"serving_cache.pt";report["shared_cache"]=dict(path=str(path),sha256=None)
            report["raw_cache_evidence"]=dict(path=str(path),exists=False,write_complete=False)
            try:
                counts["shared_cache_write_attempts"]+=1
                with path.open("xb") as stream:
                    torch.save(cpu_state(cache),stream);stream.flush();os.fsync(stream.fileno())
                counts["shared_cache_writes"]+=1
                report["raw_cache_evidence"].update(exists=True,write_complete=True,bytes=path.stat().st_size)
                report["shared_cache"]["sha256"]=_sha(path,guard)
            except BaseException as exc:
                failure=failure or exc;report["raw_cache_evidence"].update(exists=path.exists(),
                    bytes=path.stat().st_size if path.exists() else 0,error=str(exc))
        report["output_files_sha256"]={}
        try:
            if phase=="prepare" and failure is None:
                _require(all(counts[k]==v for k,v in contract["expected_prepare_counts"].items()), "Prepare counts differ")
            for p in sorted(folder.rglob("*")):
                if p.is_file():report["output_files_sha256"][str(p)]=_sha(p,guard)
            pins("exit_files_sha256");guard()
        except BaseException as exc:
            failure=failure or exc;report["preservation_exit_error"]=dict(type=type(exc).__name__,message=str(exc))
        report.update(seconds=time.monotonic()-started,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            peak_allocated_bytes=torch.cuda.max_memory_allocated() if gpu else 0,
            peak_reserved_bytes=torch.cuda.max_memory_reserved() if gpu else 0)
        if report["seconds"]>300 or report["peak_RSS_bytes"]>limits["peak_RSS_bytes"] or report["peak_allocated_bytes"]>limits["peak_allocated_bytes"] or report["peak_reserved_bytes"]>limits["peak_reserved_bytes"]:
            failure=failure or RuntimeError("Final preservation resource bound failed")
        report["passed"]=failure is None
        if failure is not None and "error" not in report:report["error"]=dict(type=type(failure).__name__,message=str(failure))
        with (folder/("prepare_report.json" if phase=="prepare" else "evaluation_report.json")).open("x") as stream:
            json.dump(_observed(report),stream,indent=2,allow_nan=False);stream.write("\n")
    if failure is not None: raise RuntimeError("Terminal reuse-only canonical validation phase; no fallback/retry") from failure
    return report
