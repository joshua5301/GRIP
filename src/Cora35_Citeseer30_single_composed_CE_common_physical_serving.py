"""Own two original-family graph caches and five accepted physical readouts.

FA153 coordinates/Q and native158 endpoints remain historical suppliers. Source
promotion authenticates this wrapper separately; no old native runtime replay.
"""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import time

KIND = "Cora35_Citeseer30_common_physical_five_arm_serving_stageFD_v1"
SCIENCE_SHA = "2301bacb8983212c95a39b1e33d694b000ea7b32cada56b9b39d4a1d00ca8337"
ARMS = ("P0", "linear25", "centroidNy25", "meanPhi25", "singleComposed25")
SEEDS = [620600, 620601, 620602]


def require(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream: return hashlib.file_digest(stream, "sha256").hexdigest()


def refs(value, found=None):
    found = {} if found is None else found
    if isinstance(value, dict):
        if {"path","sha256"} <= value.keys():
            p,h=value["path"],value["sha256"]
            require(type(p) is str and Path(p).is_absolute() and str(Path(p).resolve()) == p
                and type(h) is str and len(h)==64 and all(c in "0123456789abcdef" for c in h), "Invalid ABS ref")
            require(p not in found or found[p]==h, "Conflicting ref"); found[p]=h
        for child in value.values(): refs(child,found)
    elif isinstance(value,list):
        for child in value: refs(child,found)
    return found


def observed(value):
    if isinstance(value,dict): return {k:observed(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [observed(v) for v in value]
    if type(value) is float and not math.isfinite(value): return dict(nonfinite_observation=repr(value))
    return value


def run(protocol_path, protocol_sha256, budget, phase, arm=None, stop=lambda:False):
    started=time.monotonic(); path=Path(protocol_path)
    require(path.is_absolute() and str(path.resolve())==str(path) and sha(path)==protocol_sha256, "Protocol differs")
    packet=json.loads(path.read_text()); science_ref=packet["scientific_contract"]
    require(science_ref["sha256"]==SCIENCE_SHA and sha(science_ref["path"])==SCIENCE_SHA, "Frozen science differs")
    science=json.loads(Path(science_ref["path"]).read_text()); B=science["budgets"][budget]; n,d,c,k=(B["dimensions"][x] for x in ("N","D","C","K"))
    require(packet["kind"]==science["kind"]==KIND and budget in ("cora35","citeseer30") and phase in ("prepare","evaluate")
        and (arm is None if phase=="prepare" else arm in ARMS) and B["student_seeds"]==SEEDS, "Budget/phase/arm/cohort differs")
    out=Path(packet["prepare_folders"][budget] if phase=="prepare" else packet["evaluation_folders"][budget][arm])
    require(out.is_absolute() and str(out.resolve())==str(out), "Exclusive normalized ABS output required"); out.mkdir(parents=True,exist_ok=False)
    report=dict(schema=1,kind=KIND,budget=budget,phase=phase,arm=arm,source=packet["source"],scientific_contract=science_ref,
        protocol=dict(path=str(path),sha256=protocol_sha256),passed=False,completed=False,failure=None,gates=[],records=[],
        work=dict(own_PT_load_attempts=0,own_PT_loads=0,dataset_get_attempts=0,dataset_gets=0,graph_pack_attempts=0,graph_packs=0,
            dataset_digest_attempts=0,dataset_digests=0,representative_attempts=0,representatives=0,raw_H_decode_attempts=0,raw_H_decodes=0,
            student_fit_attempts=0,student_fits=0,student_epochs=0,route_attempts=0,routes=0,sameweight_GCN_routes=0,sameweight_SGC_routes=0,
            evidence_write_attempts=0,evidence_writes=0,shared_cache_writes=0,heads=0,adjoints=0,P_updates=0,LowRank_forwards=0,
            factor_factories=0,source_SGC=0,Q_decodes=0,map_fits=0,RMS_refits=0,test_evaluations=0),
        PT_count_scope="Six explicit owning supplier loads in prepare. Existing Planetoid internal processed reads are dataset-loader work; replay_routes adds one selected-weight load per paired invocation.",
        secondary="Supplied original cached source-SGC H, same GCN-selected weights; descriptive. No independent raw-X MLP or assertion that H equals newly packed gcn_norm S2X.",
        test_scope="Original full labels and test mask hashed once as source provenance; cache labels sentinel outside train|val, test mask omitted, zero test evaluations.")
    work=report["work"]; saved={}; cache={}; torch=None
    raw=out/("shared_physical_serving_cache.pt" if phase=="prepare" else "partial_evaluation_evidence.pt")
    report["raw_evidence"]=dict(path=str(raw),exists=False,bytes=None,sha256=None,hash_unknown=True,hash_error=None,write_completed=False)
    old_handler,old_timer=signal.getsignal(signal.SIGALRM),signal.getitimer(signal.ITIMER_REAL)
    def peaks():
        values=dict(seconds=time.monotonic()-started,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,peak_allocated_bytes=0,peak_reserved_bytes=0)
        if torch is not None and torch.cuda.is_initialized(): values.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(0),peak_reserved_bytes=torch.cuda.max_memory_reserved(0))
        return values
    def guard():
        require(not stop() and all(v<=science["resources"][key] for key,v in peaks().items()), "Terminal stop/resource boundary"); return False
    def alarm(signum,frame): raise TimeoutError("FD serving300s deadline")
    def gate(name,ok,**detail):
        report["gates"].append(dict(name=name,passed=bool(ok),**detail)); require(ok,name)
    def read(reference):
        require(sha(reference["path"])==reference["sha256"], "Pinned JSON changed"); return json.loads(Path(reference["path"]).read_text())
    def pins():
        expected=packet["readonly_files_sha256"]; require(all(expected.get(p)==h for p,h in refs(dict(packet=packet,science=science)).items()), "Required ref omitted")
        actual={}
        for p,h in expected.items():
            guard(); require(Path(p).is_absolute() and str(Path(p).resolve())==p, "Pin not normalized ABS"); actual[p]=sha(p)
        require(actual==expected, "Readonly bytes changed"); return actual
    try:
        signal.signal(signal.SIGALRM,alarm); signal.setitimer(signal.ITIMER_REAL,max(1e-6,300-(time.monotonic()-started)))
        report["readonly_entry"]=pins()
        import torch
        from src import composed_centroid_joint_ce as joint
        from src.kernel_mean_ce import _cpu,_plain,_seal
        from src.research_loop import implementation_provenance
        from src.shared_features import _tensor_identity
        from src.sweep_utils import representative
        from src.transforms import FeatureTransform
        report["source_entry"]=implementation_provenance()
        gate("current_source_entrypoint_static_admitted",report["source_entry"]==packet["source"] and Path(packet["entrypoint"]["path"]).resolve()==Path(__file__).resolve()
            and all(read(packet[name])["passed"] is True and read(packet[name])["source"]==packet["source"]
                and read(packet[name])["entrypoint"]==packet["entrypoint"] and read(packet[name])["scientific_contract"]==science_ref for name in ("ROOT_static","independent_static")))
        torch.set_num_threads(4)
        if torch.get_num_interop_threads()!=1: torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False; torch.set_float32_matmul_precision("highest")
        torch.cuda.init(); torch.cuda.reset_peak_memory_stats(0); device=torch.device("cuda:0")
        report["runtime"]=joint._runtime(device)
        gate("frozen_original_runtime",report["runtime"]==B["original_runtime"] and not torch.is_autocast_enabled("cuda")
            and all(os.environ.get(key)=="4" for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS")))
        def identity(value):
            if torch.is_tensor(value) and value.layout==torch.sparse_csr:
                return dict(shape=list(value.shape),dtype=str(value.dtype),layout=str(value.layout),crow=_tensor_identity(value.crow_indices()),col=_tensor_identity(value.col_indices()),values=_tensor_identity(value.values()))
            return _tensor_identity(value)
        def metadata(value):
            if torch.is_tensor(value): return identity(value)
            if isinstance(value,dict): return {key:metadata(v) for key,v in value.items()}
            if isinstance(value,(list,tuple)): return [metadata(v) for v in value]
            return value
        def cache_ids(value):
            return dict(graph={key:identity(v) for key,v in value["graph"].items()},masks={key:identity(v) for key,v in value["masks"].items()},
                H=identity(value["H"]),Q=identity(value["Q"]),arms={a:{key:identity(v) for key,v in row.items()} for a,row in value["arms"].items()})
        def load(reference,role):
            guard(); work["own_PT_load_attempts"]+=1; value=torch.load(reference["path"],map_location="cpu",weights_only=False)
            work["own_PT_loads"]+=1; saved["loaded_"+role]=value; saved["loaded_"+role]=_cpu(value); return value
        def moment_domain(M):
            return M.dtype==torch.float64 and list(M.shape)==[k,1+d+c] and not M.requires_grad and bool(torch.isfinite(M).all())
        def domains(value):
            graph,masks=value["graph"],value["masks"]; adj=graph["adj"]; train,val=masks["train"],masks["val"]
            return set(graph)=={"x","y","adj"} and set(masks)=={"train","val"} and graph["x"].dtype==torch.float32 and list(graph["x"].shape)==[n,d] and bool(torch.isfinite(graph["x"]).all()) \
                and graph["y"].dtype==torch.int64 and list(graph["y"].shape)==[n] and train.dtype==val.dtype==torch.bool and list(train.shape)==list(val.shape)==[n] \
                and int(train.sum())==B["dimensions"]["train_nodes"] and int(val.sum())==500 and not bool((train&val).any()) \
                and bool((graph["y"][~(train|val)]==-1).all()) and bool((graph["y"][train|val]>=0).all()) and bool((graph["y"][train|val]<c).all()) \
                and adj.layout==torch.sparse_csr and adj.dtype==torch.float32 and list(adj.shape)==[n,n] and adj.crow_indices().dtype==adj.col_indices().dtype==torch.int64 \
                and list(adj.crow_indices().shape)==[n+1] and int(adj.crow_indices()[0])==0 and int(adj.crow_indices()[-1])==len(adj.values()) \
                and bool((adj.crow_indices()[1:]>=adj.crow_indices()[:-1]).all()) and bool((adj.col_indices()>=0).all()) and bool((adj.col_indices()<n).all()) and bool(torch.isfinite(adj.values()).all()) \
                and value["H"].dtype==torch.float32 and list(value["H"].shape)==[n,d] and bool(torch.isfinite(value["H"]).all()) \
                and value["Q"].dtype==torch.float64 and list(value["Q"].shape)==[n,c] and bool(torch.isfinite(value["Q"]).all()) and bool((value["Q"]>=0).all()) and bool((value["Q"].sum(1)>0).all()) \
                and set(value["arms"])==set(ARMS) and all(set(row)=={"X","Q","uniform_weights"} and row["X"].dtype==row["Q"].dtype==torch.float32 \
                    and list(row["X"].shape)==[k,d] and list(row["Q"].shape)==[k,c] and row["uniform_weights"].dtype==torch.float64 and list(row["uniform_weights"].shape)==[k] \
                    and all(bool(torch.isfinite(v).all()) for v in row.values()) and bool((row["Q"]>=0).all()) and bool((row["Q"].sum(1)>0).all()) \
                    and torch.equal(row["uniform_weights"],torch.full_like(row["uniform_weights"],1/k)) for row in value["arms"].values())
        providers=dict(FA153=B["source_FA"],old_controls=B["control_suppliers"],native25=B["native25"])
        settings={key:v for key,v in B["student_recipe"]["settings"].items() if key!="input_scale"}
        gate("literal_original_recipe",read(B["student_recipe"]["file"])==B["student_recipe"]["settings"] and B["student_recipe"]["settings"]["input_scale"]==1)
        if phase=="prepare":
            from types import SimpleNamespace
            from src.data import get_dataset
            from src.citation_search import dataset_digest
            from src.moments import decode_moments
            from torch_geometric.nn.conv.gcn_conv import gcn_norm
            F=B["source_FA"]; fa_report,fa_root,fa_peer=(read(F[name]) for name in ("report","ROOT","independent"))
            gate("own_FA153_admitted",fa_report["passed"] is True and fa_root["passed"] is True and fa_peer["passed"] is True
                and fa_report["source"]==F["producer"] and fa_root["budgets"][budget]["report"]==fa_peer["budgets"][budget]["actual_report"]==F["report"]
                and fa_root["budgets"][budget]["arrays"]==fa_peer["budgets"][budget]["arrays"]==F["arrays"] and fa_report["descriptors"]==B["FA_descriptor_authority"])
            N=B["native25"]; native,native_root,native_peer=(read(N[name]) for name in ("report","ROOT","independent"))
            gate("own_native158_complete25_admitted",native["passed"] is True and native["completed"] is True and native_root["passed"] is True and native_peer["passed"] is True
                and native["source"]==N["producer"] and native["context"]==N["context"] and native["mode"]=="continue"
                and native_root["budgets"][budget]["report"]==native_peer["budgets"][budget]["report"]==N["report"]
                and native_root["budgets"][budget]["state"]==native_peer["budgets"][budget]["state"]==N["state"])
            payload=load(F["arrays"],"FA153"); captured=payload["captured"]
            selected={key:captured[key] for key in ("physical_M0","native_physical","loaded_native","loaded_inputs","loaded_H","loaded_hard","raw_Q")}; saved["FA_selected"]=_cpu(selected)
            gate("literal_FA_payload_producer",payload["kind"]==fa_report["kind"] and payload["budget"]==budget and payload["source"]==F["producer"] and payload["scientific_contract"]==fa_report["scientific_contract"])
            transform=FeatureTransform(**{key:v.to(device) if torch.is_tensor(v) else v for key,v in captured["loaded_inputs"]["transform"].items()})
            initial=[captured["loaded_native"][key] for key in ("u","v")]; M0=captured["physical_M0"]
            gate("exact_FA_RMS_H_Q_native_and_M0",joint._transform(transform,d,device)==B["FA_descriptor_authority"]["transform"]
                and identity(captured["loaded_H"]["h"])==B["FA_descriptor_authority"]["H"] and identity(captured["raw_Q"])==B["FA_descriptor_authority"]["Q"]
                and identity(captured["loaded_hard"])==B["FA_descriptor_authority"]["assignment"] and captured["loaded_native"]["data_digest"]==B["original_array_digest"]
                and captured["loaded_native"]["factor_seed"]==0 and captured["loaded_native"]["mixing"]==.05
                and joint._factor_digests(initial,n,k,B["dimensions"]["r"],torch.device("cpu"))==B["native_parameter_digests"] and bool(initial[0].eq(0).all())
                and identity(M0)==N["context"]["native_origin"]["moments"])
            controls=B["control_suppliers"]; linear0=load(controls["linear0"]["payload"],"linear0"); linear=load(controls["linear25"]["payload"],"linear25")
            ny=load(controls["centroidNy25"]["payload"],"centroidNy25"); mean=load(controls["meanPhi25"]["payload"],"meanPhi25"); state=load(N["state"],"singleComposed25")
            gate("original_linear_M0_observed_historical_V0_unavailable",type(linear0["step"]) is int and linear0["step"]==0 and moment_domain(linear0["moments"]) and torch.equal(linear0["moments"],M0))
            report["historical_linear_U0_V0_recovered"]=False
            gate("original_linear_uniform_endpoint25",type(linear["step"]) is int and linear["step"]==25 and linear["J_exact"] is True
                and read(controls["linear25"]["candidate"])==controls["linear25"]["settings"])
            gate("original_raw_H_Ny_uniform_endpoint25",type(ny["step"]) is int and ny["step"]==25 and ny["inner_loss_weighting"]=="uniform"
                and read(controls["centroidNy25"]["candidate"])==controls["centroidNy25"]["settings"])
            mr,mroot=(read(controls["meanPhi25"][name]) for name in ("report","ROOT"))
            gate("original_meanPhi_uniform_physical_endpoint25",mr["passed"] is True and mroot["passed"] is True and mroot["actual_report"]==controls["meanPhi25"]["report"]
                and mr["observed_final_step"]==25 and mr["kernel_mean_context"]["native_parameter_digests"]==B["native_parameter_digests"]
                and mean["step"]==25 and mean["J_exact"] is True and mean["context"]==mr["kernel_mean_context"]
                and mean["record_digest"]==_seal({key:v for key,v in mean.items() if key!="record_digest"})
                and mean["context"]["native_origin"]["moments"]==identity(M0) and mean["context"]["source_refs"]["data_digest"]==B["original_array_digest"])
            E=state["current_evaluation"]; E0=state["snapshots"][0]; saved["accepted_E0_E25"]=_cpu(dict(E0=E0,E25=E))
            gate("pure_accepted_state_E0_E25_seals",state["step"]==state["scientific_endpoint"]==25 and state["context"]==native["context"]
                and state["state_digest"]==_seal({key:v for key,v in state.items() if key!="state_digest"}) and _seal(E)==_seal(state["snapshots"][25])
                and _seal(state["parameters"])==_seal(E["parameters"]) and _seal(state["initial_parameters"])==_seal(initial)
                and E["kind"]==E0["kind"]=="composed_joint_head_evaluation_v1" and all(endpoint["record_digest"]==_seal({key:v for key,v in endpoint.items() if key!="record_digest"})
                    and endpoint["context_digest"]==_seal(native["context"]) and endpoint["config_digest"]==_seal(state["config"])
                    and all(_plain(endpoint[key])==v for key,v in native["evaluation_summaries"][str(t)]["scalars"].items())
                    and all(identity(endpoint[key])==v for key,v in native["evaluation_summaries"][str(t)]["identities"].items())
                    and _seal(endpoint["parameters"])==native["evaluation_summaries"][str(t)]["parameters_digest"]
                    and _seal(endpoint["theta_initial"])==native["evaluation_summaries"][str(t)]["theta_initial_digest"] for t,endpoint in ((0,E0),(25,E)))
                and torch.equal(E0["physical_moments"],M0) and _seal(E0["parameters"])==_seal(initial))
            moments=dict(P0=M0,linear25=linear["moments"],centroidNy25=ny["moments"],meanPhi25=mean["physical_moments"],singleComposed25=E["physical_moments"])
            rows={}; saved["owning_physical_endpoints"]=_cpu(dict(moments,linear0=linear0["moments"])); report["readouts"]={}
            for name,M in moments.items():
                gate(name+"_own_physical_moment_domain",moment_domain(M) and bool((M[:,0]>0).all()) and bool((M[:,1+d:]>=0).all()) and bool((M[:,1+d:].sum(1)>0).all()))
                if name=="centroidNy25":
                    work["raw_H_decode_attempts"]+=1; x,y,mass=decode_moments(M.to(device),d); work["raw_H_decodes"]+=1
                    saved["raw_H_native_decode"]=_cpu(dict(X=x,Q=y,mass=mass)); x,y=x.float(),y.float()
                else:
                    work["representative_attempts"]+=1; x,y,mass=representative(M,transform,d,device); work["representatives"]+=1
                saved["returned_"+name]=_cpu(dict(X=x,Q=y,mass=mass)); rows[name]=_cpu(dict(X=x,Q=y,uniform_weights=torch.full_like(mass,1/k))); saved["readouts"]=_cpu(rows)
                report["readouts"][name]=dict(moment=identity(M),returned={key:identity(v) for key,v in rows[name].items()}); guard()
            work["dataset_get_attempts"]+=1
            data=get_dataset(SimpleNamespace(dataset_name=B["dataset"],raw_data_dir=B["data_dir"].rstrip("/")+"/",citation_features="row")); work["dataset_gets"]+=1
            saved["cached_dataset_return"]=_cpu({key:getattr(data,key) for key in ("x","y","edge_index","edge_attr","train_mask","val_mask","test_mask")})
            work["graph_pack_attempts"]+=1; edges,weights=gcn_norm(data.edge_index,data.edge_attr,data.num_nodes,dtype=data.x.dtype)
            graph=dict(x=data.x.detach().cpu().clone(),y=data.y.detach().cpu().clone(),adj=torch.sparse_coo_tensor(edges.flip(0),weights,(data.num_nodes,data.num_nodes)).coalesce().to_sparse_csr()); work["graph_packs"]+=1
            saved["packed_graph_before_guards"]=_cpu(graph); train,val,test=(data.train_mask.cpu(),data.val_mask.cpu(),data.test_mask.cpu())
            work["dataset_digest_attempts"]+=1; graph_digest=dataset_digest(graph,train,(graph,val),(graph,test)); work["dataset_digests"]+=1; report["original_graph_dataset_digest"]=graph_digest
            gate("original_family_graph_provenance",graph_digest==read(B["family_config"])["data_digest"]==B["original_packed_graph_dataset_digest"])
            graph["y"]=torch.where(train|val,graph["y"],torch.full_like(graph["y"],-1))
            cache=dict(schema=1,kind=KIND,budget=budget,source=packet["source"],scientific_contract=science_ref,upstream_providers=providers,
                graph=_cpu(graph),masks=_cpu(dict(train=train,val=val)),H=_cpu(captured["loaded_H"]["h"]),Q=_cpu(captured["raw_Q"]),arms=rows,
                owning_physical_endpoints=_cpu(dict(moments,linear0=linear0["moments"])),owning_native_source=_cpu(dict(parameters=initial,hard=captured["loaded_hard"],transform=captured["loaded_inputs"]["transform"],native_physical=captured["native_physical"])),own_E25_parameters=_cpu(E["parameters"]),owning_raw_H_decode=_cpu(saved["raw_H_native_decode"]))
            gate("own_graph_labels_H_Q_all_five_readout_domains",domains(cache))
            cache["identities"]=cache_ids(cache); cache["owning_content_metadata"]=metadata({key:cache[key] for key in ("owning_physical_endpoints","owning_native_source","own_E25_parameters","owning_raw_H_decode")})
            report.update(cache_identities=cache["identities"],owning_content_metadata=cache["owning_content_metadata"],upstream_providers=providers)
            gate("source_H_Q_and_six_loaded_inputs_unchanged",cache["identities"]["H"]==B["FA_descriptor_authority"]["H"] and cache["identities"]["Q"]==B["FA_descriptor_authority"]["Q"]
                and all(metadata(value)==metadata(saved["loaded_"+role]) for role,value in (("FA153",payload),("linear0",linear0),("linear25",linear),("centroidNy25",ny),("meanPhi25",mean),("singleComposed25",state))))
            gate("prepare_exact_six_load_four_RMS_one_rawH_one_graph_counts",work["own_PT_loads"]==work["own_PT_load_attempts"]==6 and work["representatives"]==work["representative_attempts"]==4
                and work["raw_H_decodes"]==work["raw_H_decode_attempts"]==1 and work["dataset_gets"]==work["dataset_get_attempts"]==work["graph_packs"]==work["graph_pack_attempts"]==work["dataset_digests"]==work["dataset_digest_attempts"]==1)
        else:
            admission=packet["shared"][budget]; prior,root,peer=(read(admission[name]) for name in ("report","ROOT","independent"))
            gate("new_cache_ROOT_and_peer_admitted",prior["passed"] is True and root["passed"] is True and peer["passed"] is True and root["budget"]==peer["budget"]==budget
                and root["actual_report"]==peer["actual_report"]==admission["report"] and prior["shared_cache"]==root["shared_cache"]==peer["shared_cache"]==admission["cache"])
            cache=load(admission["cache"],"shared_cache")
            gate("whole_own_cache_and_provider_seals",cache["schema"]==1 and cache["kind"]==KIND and cache["budget"]==budget and cache["source"]==prior["source"]==packet["source"]
                and cache["scientific_contract"]==science_ref and cache["identities"]==cache_ids(cache)==prior["cache_identities"] and cache["upstream_providers"]==prior["upstream_providers"]==providers
                and metadata({key:cache[key] for key in ("owning_physical_endpoints","owning_native_source","own_E25_parameters","owning_raw_H_decode")})==cache["owning_content_metadata"]==prior["owning_content_metadata"]
                and cache["identities"]["H"]==B["FA_descriptor_authority"]["H"] and cache["identities"]["Q"]==B["FA_descriptor_authority"]["Q"] and domains(cache))
            from src.evaluation import fit_gcn_diagnostic
            from src.student_routes import replay_routes
            graph={key:t.to(device) for key,t in cache["graph"].items()}; masks={key:t.to(device) for key,t in cache["masks"].items()}
            q,h=cache["Q"].to(device),cache["H"].to(device); x,y,mass=(cache["arms"][arm][key].to(device) for key in ("X","Q","uniform_weights"))
            fit_folder=out/("step_0" if arm=="P0" else "step_25")
            for seed in SEEDS:
                guard();work["student_fit_attempts"]+=1;fit=fit_gcn_diagnostic(x,y,mass,graph,q,masks,seed,folder=fit_folder,training_adjacency=None,stop=guard,**settings)
                work["student_fits"]+=1;record=dict(seed=seed,fit=fit);report["records"].append(record)
                fit_paths={key:fit_folder/f"seed_{seed}{suffix}" for key,suffix in (("json",".json"),("csv","_epochs.csv"),("selected","_selected.pt"))}
                record["fit_files"]={key:dict(path=str(p),sha256=sha(p)) for key,p in fit_paths.items()}
                history=list(csv.DictReader(fit_paths["csv"].open()));work["student_epochs"]+=len(history);fit_JSON=json.loads(fit_paths["json"].read_text())
                best=max(float(row["val_acc"]) for row in history);first=next(row for row in history if float(row["val_acc"])==best)
                gate("fit_"+str(seed)+"_full_firstmax_original_recipe",[int(row["epoch"]) for row in history]==list(range(1,settings["epochs"]+1))
                    and fit_JSON["result"]==fit and fit_JSON["recipe"]["settings"]==settings and fit_JSON["recipe"]["test_enabled"] is False
                    and fit["epoch"]==int(first["epoch"]) and fit["val_acc"]==best and not any("test" in key for key in fit) and all(not isinstance(v,float) or math.isfinite(v) for v in fit.values()))
                route_path=fit_folder/f"seed_{seed}_sameweights_routes.json";work["route_attempts"]+=1
                route=replay_routes(fit_paths["selected"],graph,h.float(),dict(val=masks["val"]),settings,route_path,seed=seed,stop=guard)
                work["routes"]+=1;work["sameweight_GCN_routes"]+=1;work["sameweight_SGC_routes"]+=1;record["routes"]=route
                record["route_file"]=dict(path=str(route_path),sha256=sha(route_path));route_JSON=json.loads(route_path.read_text())
                gate("route_"+str(seed)+"_same_selected_weights",route_JSON["result"]==route and route["gcn_val_acc"]==fit["val_acc"] and route["epoch"]==fit["epoch"]
                    and route_JSON["recipe"]["source_fingerprint"]==fit_JSON["fingerprint"] and route_JSON["recipe"]["test_enabled"] is False and not any("test" in key for key in route));guard()
            report.update(shared_cache=admission["cache"],shared_prepare_report=admission["report"],internal_route_checkpoint_reads=work["routes"],cache_identities=cache["identities"])
            gate("evaluate_exact_one_cache_three_fit_route_counts",work["own_PT_loads"]==work["own_PT_load_attempts"]==1 and work["student_fits"]==work["student_fit_attempts"]==3
                and work["student_epochs"]==settings["epochs"]*3 and work["routes"]==work["route_attempts"]==3
                and all(work[key]==0 for key in ("dataset_gets","graph_packs","dataset_digests","representatives","raw_H_decodes")))
        gate("no_new_condensation_or_source_test_work",all(work[key]==0 for key in ("heads","adjoints","P_updates","LowRank_forwards","factor_factories","source_SGC","Q_decodes","map_fits","RMS_refits","test_evaluations")))
        report["ordinary_GCN_graph_SpMM"]="Normal fitter/replay operations, uninstrumented; source/cache SGC products remain zero."
        torch.cuda.synchronize(device);guard();report["completed"]=True
    except BaseException as error:
        report["failure"]=dict(type=type(error).__name__,message=str(error));report["inflight_interiors"]="Unknown; returned owning evidence and attempts counted separately"
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        try:
            evidence=cache if phase=="prepare" and report["completed"] else dict(partial_unqualified=True,returned=saved,cache=cache,records=report["records"])
            if torch is not None:
                info=report["raw_evidence"]
                try:
                    work["evidence_write_attempts"]+=1
                    with raw.open("xb") as stream:torch.save(evidence,stream);stream.flush();os.fsync(stream.fileno())
                    info["write_completed"]=True;work["evidence_writes"]+=1
                    if phase=="prepare" and report["completed"]:work["shared_cache_writes"]+=1
                except BaseException as error:report["raw_write_error"]=repr(error)
                try:
                    info["exists"]=raw.exists()
                    if raw.exists():
                        info["bytes"]=raw.stat().st_size
                        try:info["sha256"]=sha(raw);info["hash_unknown"]=False
                        except BaseException as error:info["hash_error"]=repr(error)
                except BaseException as error:report["raw_metadata_error"]=repr(error)
                if phase=="prepare" and report["completed"]:report["shared_cache"]=dict(path=str(raw),sha256=info["sha256"])
            try:
                report["readonly_exit"]=pins();report["source_exit"]=implementation_provenance();require(report["source_exit"]==packet["source"], "Exit source differs")
            except BaseException as error:report["exit_verification_error"]=repr(error)
            report["resources"]=peaks();report["passed"]=bool(report["completed"] and report["failure"] is None
                and not any(key in report for key in ("raw_write_error","raw_metadata_error","exit_verification_error")) and report["raw_evidence"]["write_completed"]
                and not report["raw_evidence"]["hash_unknown"] and all(v<=science["resources"][key] for key,v in report["resources"].items()))
            rp=out/("prepare_report.json" if phase=="prepare" else "evaluation_report.json")
            with rp.open("x") as stream:json.dump(observed(report),stream,indent=2,allow_nan=False);stream.write("\n");stream.flush();os.fsync(stream.fileno())
            final=peaks()
            if report["passed"] and any(v>science["resources"][key] for key,v in final.items()):
                report.update(passed=False,resources=final,failure=dict(type="FinalSerializationResourceBoundary",message="Preserve failure; no retry"))
                with rp.open("w") as stream:json.dump(observed(report),stream,indent=2,allow_nan=False);stream.write("\n");stream.flush();os.fsync(stream.fileno())
        finally:signal.signal(signal.SIGALRM,old_handler);signal.setitimer(signal.ITIMER_REAL,*old_timer)
    require(report["passed"],"Terminal FD serving failure; no retry/recipe/seed/secondary rescue")
    return str(rp)
