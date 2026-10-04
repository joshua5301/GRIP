"""Validation-only shared Flickr cache and unchanged canonical-readout students.

Cached train H and validation packed-graph S2X are admitted by exact source and
content fingerprints. No propagation, critic, moment or target decoding occurs.
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
import scipy.sparse as sp
import torch
import torch_geometric.transforms as T
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import subgraph

from src.inductive_evaluation import fit_inductive_gcn
from src.io import _fingerprint, cpu_state
from src.large_kernel_mean_source_capture import _matrix, _observed, _require, _sha
from src.large_pilot import _data_digest
from src.shared_features import _load_state, _tensor_identity
from src.student_routes import replay_routes

KIND = "large_canonical_inductive_validation_v1"
CONTRACT_SHA = "98d1b9c6cb36f0ecd3cae3ef6babbd47e66b3b3bf6f53a21fb3525144b490e6f"
ARMS = ("own_canonical_P0", "uniform_linear25", "uniform_centroid_Ny25", "meanPhi25")


def _refs(value):
    result = {}
    if isinstance(value, dict):
        if {"path", "sha256"} <= value.keys(): result[value["path"]] = value["sha256"]
        for child in value.values(): result.update(_refs(child))
    elif isinstance(value, list):
        for child in value: result.update(_refs(child))
    return result


def _identity(value):
    if value.layout == torch.sparse_csr:
        return dict(shape=list(value.shape), dtype=str(value.dtype), layout=str(value.layout),
            crow=_tensor_identity(value.crow_indices()), col=_tensor_identity(value.col_indices()),
            values=_tensor_identity(value.values()))
    return _tensor_identity(value)


def _identities(cache):
    return dict(train_graph={k:_identity(t) for k,t in cache["train_graph"].items()},
        val_graph={k:_identity(t) for k,t in cache["val_graph"].items()},
        train_mask=_identity(cache["train_mask"]), val_mask=_identity(cache["val_mask"]),
        train_H=_identity(cache["train_H"]), val_H=_identity(cache["val_H"]), Q=_identity(cache["Q"]),
        arms={name:{k:_identity(t) for k,t in row.items()} for name,row in cache["arms"].items()})


def _graph_domain(graph, mask, nodes):
    _matrix(graph["x"],(nodes,500),torch.float32)
    y,adj=graph["y"],graph["adj"]
    _require(y.dtype==torch.int64 and y.shape==(nodes,) and bool((y>=0).all()) and bool((y<7).all())
        and mask.dtype==torch.bool and mask.shape==(nodes,) and bool(mask.all())
        and adj.layout==torch.sparse_csr and adj.dtype==torch.float32 and adj.shape==(nodes,nodes),
        "Packed graph/label/mask domain differs")
    crow,col,values=adj.crow_indices(),adj.col_indices(),adj.values()
    _require(crow.dtype==col.dtype==torch.int64 and crow.shape==(nodes+1,) and col.shape==values.shape
        and crow[0].item()==0 and crow[-1].item()==len(col) and bool((crow[1:]>=crow[:-1]).all())
        and bool((col>=0).all()) and bool((col<nodes).all()) and bool(torch.isfinite(values).all()),
        "Packed CSR buffers invalid")


def run(protocol_path, protocol_sha256, phase, arm=None, stop=lambda: False):
    from src.research_loop import implementation_provenance

    _require(callable(stop) and _sha(protocol_path)==protocol_sha256, "Frozen invocation changed")
    packet=json.loads(Path(protocol_path).read_text()); ref=packet["scientific_contract"]
    _require(ref["sha256"]==CONTRACT_SHA and _sha(ref["path"])==CONTRACT_SHA, "Scientific contract changed")
    contract=json.loads(Path(ref["path"]).read_text())
    _require(type(packet["schema"]) is int and packet["schema"]==1 and packet["kind"]==contract["kind"]==KIND
        and packet["budget"]==contract["budget"]=="flickr44" and packet["test_enabled"] is contract["test_enabled"] is False
        and packet["runtime"]==contract["runtime"] and packet["resource_limits"]==contract["resource_limits"]
        and packet["source"]==implementation_provenance() and phase in ("prepare","evaluate")
        and (arm is None if phase=="prepare" else arm in ARMS), "Frozen phase/source/domain differs")
    required=_refs(contract); required.update(_refs(ref))
    if phase=="evaluate":
        for key in ("shared_cache","shared_prepare_report","shared_prepare_acceptance"):
            required.update(_refs(packet[key]))
    _require(all(Path(p).is_absolute() and packet["readonly_files_sha256"].get(p)==s for p,s in required.items()),
        "Required readonly admission omitted")
    folder=Path(packet["prepare_folder"] if phase=="prepare" else packet["evaluation_folders"][arm])
    _require(folder.is_absolute() and not folder.exists(), "Fresh exclusive namespace required")
    folder.mkdir(parents=True); started=time.monotonic(); cache={}; failure=None; gpu=False
    counts={name:0 for name in contract["expected_prepare_counts"]}
    counts.update(raw_data_IO_attempts=0,train_StandardScaler_fit_attempts=0,induced_graph_pack_attempts=0,
        train_H_cache_load_attempts=0,validation_H_cache_load_attempts=0,canonical_readout_load_attempts=0,
        shared_cache_write_attempts=0,fit_attempts=0,student_fits=0,student_epochs=0,
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
            and contract["student_seeds"]==[20200,20201,20202]
            and all(type(s) is int for s in contract["student_seeds"])
            and [item["id"] for item in contract["arms"]]==list(ARMS), "Original recipe/cohort/arms differ")
        if phase=="prepare":
            cache.update(schema=1,kind=KIND,source=packet["source"],scientific_contract=ref,arms={},arm_providers={})
            raw=contract["raw_sources"];counts["raw_data_IO_attempts"]+=1
            adjacency=sp.load_npz(raw["adj_full.npz"]["path"])
            role=json.loads(Path(raw["role.json"]["path"]).read_text())
            class_map=json.loads(Path(raw["class_map.json"]["path"]).read_text())
            features=np.load(raw["feats.npy"]["path"],allow_pickle=False);nodes=adjacency.shape[0]
            counts["raw_data_IO"]+=1;guard()
            _require(adjacency.shape==(nodes,nodes) and features.shape==(nodes,500), "Raw graph shape differs")
            masks={name:torch.zeros(nodes,dtype=torch.bool) for name in ("train","val")}
            for name,key,count in (("train","tr",44625),("val","va",22312)):
                ids=role[key];_require(len(ids)==count and len(set(ids))==count
                    and all(type(i) is int and 0<=i<nodes for i in ids), "Original split IDs differ")
                masks[name][ids]=True
            _require(not bool((masks["train"] & masks["val"]).any()), "Train/val overlap")
            labels=torch.full((nodes,),-1,dtype=torch.int64)
            for i in torch.where(masks["train"] | masks["val"])[0].tolist():
                label=class_map[str(i)];_require(type(label) is int and 0<=label<7, "Allowed scalar class differs")
                labels[i]=label
            _require(bool((labels[masks["train"]]==0).any()), "Original zero-based class convention differs")
            counts["train_StandardScaler_fit_attempts"]+=1
            scaler=StandardScaler().fit(features[role["tr"]]);counts["train_StandardScaler_fits"]+=1;guard()
            graph=Data(x=torch.FloatTensor(scaler.transform(features)),
                edge_index=torch.LongTensor(np.array(adjacency.nonzero())),y=labels)
            graph=T.ToUndirected()(graph);guard()
            for name in ("train","val"):
                mask=masks[name];edges,_=subgraph(mask,graph.edge_index,relabel_nodes=True)
                induced=Data(x=graph.x[mask],y=graph.y[mask],edge_index=edges)
                counts["induced_graph_pack_attempts"]+=1
                edges,weights=gcn_norm(induced.edge_index,induced.edge_attr,induced.num_nodes,dtype=induced.x.dtype)
                packed=dict(x=cpu_state(induced.x),y=cpu_state(induced.y),
                    adj=cpu_state(torch.sparse_coo_tensor(edges.flip(0),weights,(induced.num_nodes,induced.num_nodes)).coalesce().to_sparse_csr()))
                cache[name+"_graph"]=packed;cache[name+"_mask"]=torch.ones(induced.num_nodes,dtype=torch.bool)
                counts["induced_graph_packs"]+=1;guard()
            for name,n in (("train",44625),("val",22312)):
                _graph_domain(cache[name+"_graph"],cache[name+"_mask"],n)
            del graph,induced,packed,features,scaler,adjacency,class_map,role,masks,labels
            train={k:t.to(device) for k,t in cache["train_graph"].items()}
            val={k:t.to(device) for k,t in cache["val_graph"].items()};train_mask=cache["train_mask"].to(device)
            fresh=_data_digest(train,train_mask,(val,None),None,guard)
            train_token=_fingerprint(dict(data=fresh,propagation="SGC2-normalize-adj-v1",torch=str(torch.__version__),dtype="torch.float32"))
            report["fresh_no_H_data_digest"]=fresh;report["expected_train_H_source_token"]=train_token
            _require(train_token==contract["accepted_train_H"]["source_digest"], "Rawgraph/train-H source mismatch")
            counts["train_H_cache_load_attempts"]+=1
            saved=_load_state(contract["original_serving_metadata_refs"]["saved_train_H"]["path"],"shared_h")
            report["train_cache_lineage"]={k:saved.get(k) for k in ("schema","kind","source_digest","identity")}
            cache["train_H"]=cpu_state(saved["h"]);counts["train_H_cache_loads"]+=1
            _matrix(cache["train_H"],(44625,500),torch.float32)
            _require(type(saved["schema"]) is int and saved["source_digest"]==train_token
                and saved["identity"]==_tensor_identity(cache["train_H"])==contract["accepted_train_H"]["identity"], "Saved train-H metadata/content mismatch")
            historical=_data_digest(train,train_mask,(val,None),cache["train_H"].to(device),guard)
            report["fresh_with_cached_H_data_digest"]=historical
            _require(historical==contract["historical_data_digest"], "Original full-H source differs")
            val_token=_fingerprint(dict(data_digest=fresh,propagation="validation-packed-adjacency-S2X-v1",
                dtype="torch.float32",torch_version=str(torch.__version__)))
            report["expected_validation_H_source_token"]=val_token
            guard();counts["validation_H_cache_load_attempts"]+=1
            saved=_load_state(contract["original_serving_metadata_refs"]["validation_route_H_opaque_candidate"]["path"],"shared_h")
            report["validation_cache_lineage"]={k:saved.get(k) for k in ("schema","kind","source_digest","identity")}
            cache["val_H"]=cpu_state(saved["h"]);counts["validation_H_cache_loads"]+=1
            _matrix(cache["val_H"],(22312,500),torch.float32)
            _require(type(saved["schema"]) is int and saved["source_digest"]==val_token
                and saved["identity"]==_tensor_identity(cache["val_H"]), "Validation shared-H source/content mismatch; no fallback")
            cache["source_tokens"]=dict(no_H_data_digest=fresh,with_H_data_digest=historical,train_H=train_token,val_H=val_token)
            guard()
            for item in contract["arms"]:
                actual=read(item["actual_report"]);accepted=read(item["actual_acceptance"])
                _require(actual["passed"] is True and accepted["passed"] is True
                    and actual["source"]==accepted["source"]==item["producer"]
                    and accepted["actual_report"]==item["actual_report"]
                    and accepted["actual_arrays" if item["step"]==0 else "canonical_endpoint25"]==item["arrays"], "Canonical producer/admission differs")
                counts["canonical_readout_load_attempts"]+=1
                supplied=torch.load(item["arrays"]["path"],map_location="cpu",weights_only=False)
                cache["arms"][item["id"]]={key:cpu_state(supplied[old]) for key,old in item["keys"].items()}
                counts["canonical_readout_loads"]+=1;cache["arm_providers"][item["id"]]=dict(producer=item["producer"],
                    arrays=item["arrays"],actual_report=item["actual_report"],actual_acceptance=item["actual_acceptance"],step=item["step"])
                if item["step"]==0:
                    cache["Q"]=cpu_state(supplied["Q"])
                    _require(_tensor_identity(cache["Q"])==actual["array_descriptors"]["Q"]
                        and _tensor_identity(cache["train_H"])==actual["array_descriptors"]["saved_H_FP32"], "Captured rawQ/H provenance differs")
                else:
                    _require(type(supplied["step"]) is int and supplied["step"]==25 and supplied["source"]==item["producer"]
                        and supplied["arm"]==item["id"], "Canonical25 owning producer differs")
                guard()
        else:
            prior=read(packet["shared_prepare_report"]);accepted=read(packet["shared_prepare_acceptance"])
            _require(prior["passed"] is True and prior["phase"]=="prepare" and accepted["passed"] is True
                and prior["source"]==accepted["source"]==packet["source"]
                and prior["scientific_contract"]==accepted["scientific_contract"]==ref
                and prior["shared_cache"]==accepted["shared_cache"]==packet["shared_cache"]
                and accepted["shared_prepare_report"]==packet["shared_prepare_report"], "Shared cache not ROOT-admitted")
            cache=torch.load(packet["shared_cache"]["path"],map_location="cpu",weights_only=False)
            _require(type(cache["schema"]) is int and cache["schema"]==1 and cache["kind"]==KIND
                and cache["source"]==packet["source"] and cache["scientific_contract"]==ref
                and cache["identities"]==_identities(cache)==prior["cache_identities"], "Owning shared-cache integrity differs")
        _require(set(cache["arms"])==set(ARMS), "Shared canonical arms differ")
        for name,n in (("train",44625),("val",22312)):
            _graph_domain(cache[name+"_graph"],cache[name+"_mask"],n)
            _matrix(cache[name+"_H"],(n,500),torch.float32)
        for item in contract["arms"]:
            row=cache["arms"][item["id"]]
            _matrix(row["X"],(44,500),torch.float32);_matrix(row["Qbar"],(44,7),torch.float32)
            _matrix(row["uniform_weights"],(44,),torch.float64)
            _require({k:_tensor_identity(t) for k,t in row.items()}==item["expected_descriptors"]
                and bool((row["Qbar"]>=0).all()) and bool((row["Qbar"].sum(1)>0).all())
                and bool((row["uniform_weights"]>0).all()), "Canonical readout domain/bytes differ")
        _matrix(cache["Q"],(44625,7),torch.float64)
        _require(bool((cache["Q"]>=0).all()) and bool((cache["Q"].sum(1)>0).all()), "Captured rawQ domain differs")
        cache["identities"]=_identities(cache);report["cache_identities"]=cache["identities"]
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
