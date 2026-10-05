"""Unexecuted FN two-fold endpoint serving through a new full-source physical export.

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

KIND = "source_crossfit_composed_CE_fullsource_RMS_five_arm_common_physical_validation_stageFN_v1"
SCIENCE_SHA = "2687264b85018605c3730c6e8db8b869d9f18ffa1325efcf3d8a9807d81c493a"
ARMS = ("P0", "linear25", "centroidNy25", "composedjoint25", "crossFit25")
SOURCE152_ARMS = ("P0", "linear25", "centroidNy25", "unscaledjoint25", "centeredbalanced25", "physicalOuter25", "composedjoint25")
OLD_ARM_KEYS = {name:name for name in ARMS[:-1]}
SEEDS = [621200, 621201, 621202]


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
            factor_factories=0,source_SGC=0,Q_decodes=0,map_fits=0,RMS_refits=0,test_evaluations=0,
            material_build_attempts=0,material_builds=0,native_quotient_attempts=0,native_quotient_decode=0,RMS_inverse_attempts=0,RMS_inverse=0,
            LowRank_forward_attempts=0,evidence_write_attempts=0,evidence_writes=0,cache_write=0,RNG=0,new_dataset_get=0,graph_pack=0,teacher_Q_decode=0,map_fits_or_calls=0),
        PT_count_scope="Explicit owning payload reads only: prepare3/evaluate1. All three owning returned inputs retained before guards; no dataset loader or graph packing. Owning CPU copies and loaded native GPU export buffers are inside measured whole RSS/CUDA caps. Original replay_routes adds one selected-checkpoint read per paired replay invocation (GCN and SGC+MLP routes together).",
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
    def alarm(signum,frame): raise TimeoutError("FN full-source RMS serving300s deadline")
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
        from src.dual_head_ce import _factor_digests
        from src.low_rank_assignment import LowRankMoments
        from src.moments import make_material,decode_moments
        from src.source_crossfit_composed_ce_trajectory import validate_evaluated_state
        from src.transforms import FeatureTransform
        report["source_entry"]=implementation_provenance();gate("current_source_and_entrypoint",report["source_entry"] == packet["source"]
            and Path(packet["entrypoint"]["path"]).resolve() == Path(__file__).resolve())
        bridge=read(packet["prerequisite_admission"])
        gate("ROOT_current_serving_static_bridge",bridge["passed"] is True and bridge["source"]==packet["source"]
            and bridge["entrypoint"]==packet["entrypoint"] and bridge["scientific_contract"]==science_ref)
        torch.set_num_threads(4)
        if torch.get_num_interop_threads() != 1: torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision("highest")
        torch.use_deterministic_algorithms(False);torch.cuda.init();device=torch.device("cuda:0")
        report["runtime"]=joint._runtime(device);gate("frozen_original_runtime",report["runtime"] == B["original_runtime"]
            and torch.get_num_interop_threads()==1 and all(os.environ.get(key)==value for key,value in science["runtime"].items() if key.isupper() and isinstance(value,str)))
        def load(reference,role):
            guard();work["own_PT_load_attempts"]+=1;value=torch.load(reference["path"],map_location="cpu",weights_only=False)
            work["own_PT_loads"]+=1;saved["loaded_"+role]=value;saved["loaded_"+role]=_cpu(value);return value
        def retain(key,value): saved[key]=value;saved[key]=_cpu(value)
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
        def domains(value):
            n,d,c,k=(B["dimensions"][x] for x in ("nodes","dimension","classes","cells"))
            require(set(value["graph"])=={"x","y","adj"} and set(value["masks"])=={"train","val"},"Val-only graph keys differ")
            train,val=(value["masks"][key] for key in ("train","val"));labels=value["graph"]["y"]
            return (train.dtype==val.dtype==torch.bool and list(train.shape)==list(val.shape)==[n] and int(val.sum())==500
                and not bool((train&val).any()) and bool((labels[~(train|val)]==-1).all())
                and list(value["H"].shape)==[n,d] and list(value["Q"].shape)==[n,c]
                and all(list(row["X"].shape)==[k,d] and list(row["Q"].shape)==[k,c] and list(row["uniform_weights"].shape)==[k]
                    and row["X"].dtype==row["Q"].dtype==torch.float32 and row["uniform_weights"].dtype==torch.float64
                    and all(bool(torch.isfinite(v).all()) for v in row.values()) and bool((row["Q"]>=0).all()) and bool((row["Q"].sum(1)>0).all())
                    and torch.equal(row["uniform_weights"],torch.full_like(row["uniform_weights"],1/k)) for row in value["arms"].values()))
        if phase == "prepare":
            L=B["source152_admission"];prior,root,peer=(read(L[role]) for role in ("report","ROOT","independent"))
            gate("actual152_whole_cache_ROOT_peer_admitted",prior["passed"] is True and root["passed"] is True and peer["passed"] is True
                and root["budget"]==peer["budget"]==budget and root["actual_report"]==peer["actual_report"]==L["report"]
                and prior["shared_cache"]==root["shared_cache"]==peer["shared_cache"]==L["cache"]
                and prior["source"]==prior["source_entry"]==prior["source_exit"]==root["source"]==peer["source"]==B["expected_source152_header"]["source"]
                and prior["scientific_contract"]==root["scientific_contract"]==peer["scientific_contract"]==L["scientific_contract"]
                and prior["cache_identities"]==root["cache_identities"]==peer["cache_identities"]==B["expected_source152_identities"])
            C=B["source135"];cap=read(C["report"]);N=B["FM_native25"];native=read(N["report"])
            for role in ("ROOT","independent"):
                cp=read(C["ROOT" if role=="ROOT" else "peer"]);np=read(N[role])
                gate(role+"_source135_and_FM25_admitted",cp["passed"] is True and np["passed"] is True
                    and cp["budgets"][budget]["report" if role=="ROOT" else "actual_report"]==C["report"] and cp["budgets"][budget]["arrays"]==C["arrays"]
                    and all(np["budgets"][budget][key]==N[key] for key in ("report","state","arrays")))
            gate("actual135_fullproducer_and_complete177_endpoint",cap["passed"] is True and cap["capture_completed"] is True and cap["budget"]==budget
                and cap["source"]==cap["source_entry"]==cap["source_exit"]==B["source135_literal_producer"] and cap["scientific_contract"]==B["source135_science"]
                and Path(C["arrays"]["path"]).stat().st_size==C["expected_array_bytes"] and cap["raw_evidence"]["sha256"]==C["arrays"]["sha256"]
                and native["passed"] is True and native["completed"] is True and native["budget"]==budget and native["mode"]=="continue"
                and native["source"]==native["source_entry"]==native["source_exit"]==N["native_producer"] and native["scientific_contract"]==N["scientific_contract"]
                and native["state"]==N["state"] and native["context"]==N["context"] and native["pre_origin_contract"]==N["pre_origin_contract"]
                and native["core_work"]==N["core_work"] and native["core_artifacts"]==N["core_artifacts"]
                and native["evaluation_summaries"]["25"]==N["E25_summary"] and native["last_update_summary"]==N["last_update_summary"])
            combined=read(N["combined"]);gate("combined_FM25_authority",combined["passed"] is True and combined["source"]==N["native_producer"]
                and combined["scientific_contract"]==N["scientific_contract"] and combined["ROOT"]==N["ROOT"] and combined["independent"]==N["independent"]
                and all(combined["budgets"][budget][key]==N[key] for key in ("report","state","arrays")))
            gate("all8_original_assets_pinned",all(packet["readonly_files_sha256"].get(p)==h for case in science["budgets"].values() for p,h in case["mandatory_original_asset_pins"].items()))
            historical=load(L["cache"],"source152_cache");payload=load(C["arrays"],"original135_source");state=load(N["state"],"FM25_state")
            gate("whole152_payload_and_provider_exact",set(historical)==set(science["source152_cache_fields"])
                and all(historical[key]==value for key,value in B["expected_source152_header"].items())
                and historical["identities"]==cache_ids(historical)==B["expected_source152_identities"] and set(historical["arms"])==set(SOURCE152_ARMS)
                and tensor_metadata(historical["upstream_providers"])==B["source152_expected_upstream_providers"]
                and _seal(tensor_metadata(historical["upstream_providers"]))==B["source152_upstream_providers_seal"] and domains(historical))
            gate("original135_payload_header",payload["schema"]==1 and payload["kind"]==cap["kind"] and payload["budget"]==budget
                and payload["source"]==B["source135_literal_producer"] and payload["scientific_contract"]==B["source135_science"])
            captured=payload["captured"];n,d,c,k,r=(B["dimensions"][x] for x in ("nodes","dimension","classes","cells","rank"))
            transform=FeatureTransform(**{name:value.to(device) if torch.is_tensor(value) else value for name,value in captured["z_transform"]["transform"].items()})
            initial=[captured["native_factors"][name] for name in ("u","v")]
            gate("original135_RMS_H_Q_z_native_P0_frame",_transform(transform,d)==B["asset_descriptors"]["transform"]
                and all(identity(v)==B["asset_descriptors"][name] for name,v in (("H",captured["H_cache"]["h"]),("z",captured["z_transform"]["z"]),("Q",captured["Q"]),("assignment",captured["hard"])))
                and identity(captured["H_cache"]["h"])==historical["identities"]["H"] and identity(captured["Q"])==historical["identities"]["Q"]
                and _factor_digests(initial,n,k,r,torch.device("cpu"))==B["matched_initialization"]["native_initial_digests"] and bool(initial[0].eq(0).all())
                and captured["native_factors"]["data_digest"]==B["original_data_digest"] and captured["native_factors"]["factor_seed"]==0 and captured["native_factors"]["mixing"]==.05
                and identity(captured["physical_M0"])==B["matched_initialization"]["parent_P0_M"])
            report["historical_linear_V0_recovered"]=False
            validate_evaluated_state(state,N["expected_config"],N["context"],folder=state["artifact_folder"])
            E=state["current_evaluation"];A=state["last_completed_update"]
            ES=dict(scalars=_plain({key:E[key] for key in ("step","config_digest","context_digest","source_admission_digest","initial_parameter_digests","parent_update_digest","raw_CE","CE0","objective","record_digest")}),
                packed_moments=identity(E["packed_moments"]),parameters_digest=_seal(E["parameters"]),folds={f:dict(raw_CE=e["raw_CE"],head_work=e["head_work"],theta_initial_digest=_seal(e["theta_initial"]),
                    identities={key:identity(e[key]) for key in ("theta","raw_rhs","head_features","physical_moments","physical_centers","physical_targets","physical_mass")}) for f,e in E["folds"].items()})
            AS=dict(scalars=_plain({key:A[key] for key in ("step","evaluation_digest","previous_update_digest","raw_adjoint_initial_digest","record_digest")}),
                folds={f:dict(adjoint_work=a["adjoint_work"],raw_adjoint=identity(a["raw_adjoint"]),raw_moment_G=identity(a["raw_moment_G"])) for f,a in A["folds"].items()},
                identities={key:identity(A[key]) for key in ("normalized_G","native_grad_U","native_grad_V")},parameters_after_digest=_seal(A["parameters_after"]),optimizer_after_digest=_seal(A["optimizer_after"]))
            gate("own_pure_FM_E25_state",state["step"]==25 and E["kind"]=="two_fold_composed_head_pair_evaluation_v1" and state["work"]==N["core_work"]
                and ES==N["E25_summary"] and json.loads(json.dumps(AS,allow_nan=False))==N["last_update_summary"]
                and _seal(state["initial_parameters"])==_seal(initial) and _seal(state["parameters"])==_seal(E["parameters"])==_seal(state["snapshots"][25]["parameters"])
                and str(Path(state["artifact_folder"]).resolve())==str(Path(N["core_artifacts"]["resume"]["path"]).parent))
            report["memory_observations"]=dict(before_fullphysical_material=peaks())
            z,q=captured["z_transform"]["z"].to(device),captured["Q"].to(device);params=[p.to(device) for p in E["parameters"]];hard=captured["hard"].to(device)
            retain("held_native_export_inputs",dict(parameters=params,hard=hard));work["material_build_attempts"]+=1;material=make_material(z,q);work["material_builds"]+=1;retain("fullphysical_material",material)
            work["LowRank_forward_attempts"]+=1;M=LowRankMoments.apply(*params,hard,material,N["expected_config"]["mixing"],N["expected_config"]["chunk_size"])
            work["LowRank_forwards"]+=1;retain("returned_fullsource_M25",M)
            gate("new_fullsource_physical_domain",M.dtype==torch.float64 and list(M.shape)==[k,1+d+c] and not M.requires_grad
                and bool(torch.isfinite(M).all()) and bool((M[:,0]>0).all()) and bool((M[:,1+d:]>=0).all()) and bool((M[:,1+d:].sum(0)>0).all()))
            work["native_quotient_attempts"]+=1;centers,targets,mass=decode_moments(M,d);work["native_quotient_decode"]+=1
            retain("returned_native_fullsource_quotients",dict(centers=centers,targets=targets,mass=mass))
            work["RMS_inverse_attempts"]+=1;features=centers * transform.scale + transform.output_center + transform.center;work["RMS_inverse"]+=1
            x,y=features.float(),targets.float();retain("returned_RMS_readout",dict(X=x,Q=y,mass=mass))
            rows={name:_cpu(historical["arms"][oldname]) for name,oldname in OLD_ARM_KEYS.items()};rows["crossFit25"]=_cpu(dict(X=x,Q=y,uniform_weights=torch.full_like(mass,1/k)));retain("readouts",rows)
            report["memory_observations"]["after_native_quotient_before_release"]=peaks()
            gate("new_uniform_FP64_equal_P0_and_four_controls_unchanged",identity(rows["crossFit25"]["uniform_weights"])==historical["identities"]["arms"]["P0"]["uniform_weights"]
                and all({key:identity(v) for key,v in rows[name].items()}==B["selected_four_control_identities"][name] for name in OLD_ARM_KEYS))
            export=_cpu(dict(full_M25=M,parameters=params,hard=hard,native_quotients=saved["returned_native_fullsource_quotients"],readout=rows["crossFit25"]));retain("owning_canonical_export",export)
            providers=dict(source152=dict(admission=L,header=B["expected_source152_header"],upstream_providers=historical["upstream_providers"],all7_identities=historical["identities"],arm_key_mapping=OLD_ARM_KEYS),
                original135=C,original135_producer=B["source135_literal_producer"],FM_native25=N)
            cache=dict(schema=1,kind=KIND,budget=budget,source=packet["source"],scientific_contract=science_ref,upstream_providers=providers,
                graph=_cpu(historical["graph"]),masks=_cpu(historical["masks"]),H=_cpu(historical["H"]),Q=_cpu(historical["Q"]),arms=rows,owning_canonical_export=export)
            cache["identities"]=cache_ids(cache);cache["owning_export_metadata"]=tensor_metadata(export)
            report.update(cache_identities=cache["identities"],upstream_providers=tensor_metadata(providers),owning_export_metadata=cache["owning_export_metadata"],native25_core_artifacts=N["core_artifacts"])
            gate("new_domain_and_shared_graph_H_Q_masks_unchanged",domains(cache) and all(cache["identities"][key]==B["shared_identities"][key] for key in ("graph","masks","H","Q")))
            gate("all_three_owning_inputs_unchanged",all(tensor_metadata(value)==tensor_metadata(saved["loaded_"+name]) for value,name in
                ((historical,"source152_cache"),(payload,"original135_source"),(state,"FM25_state"))))
            gate("prepare_exact_selected_calls_before_write",work["own_PT_loads"]==work["own_PT_load_attempts"]==3
                and work["material_builds"]==work["material_build_attempts"]==work["LowRank_forwards"]==work["LowRank_forward_attempts"]==1
                and work["native_quotient_decode"]==work["native_quotient_attempts"]==work["RMS_inverse"]==work["RMS_inverse_attempts"]==1)
        else:
            admission=packet["shared"][budget];prior,root,peer=(read(admission[x]) for x in ("report","ROOT","independent"))
            gate("shared_cache_ROOT_independent_admitted",prior["passed"] is True and root["passed"] is True and peer["passed"] is True
                and prior["shared_cache"] == root["shared_cache"] == peer["shared_cache"] == admission["cache"]
                and root["actual_report"] == peer["actual_report"] == admission["report"])
            cache=load(admission["cache"],"shared_cache")
            gate("shared_cache_whole_identity",cache["source"] == prior["source"] == packet["source"] and cache["scientific_contract"] == science_ref
                and cache["budget"] == budget and cache["kind"] == KIND and cache["identities"] == prior["cache_identities"] == cache_ids(cache)
                and set(cache["arms"]) == set(ARMS) and set(cache["masks"]) == {"train","val"}
                and tensor_metadata(cache["upstream_providers"])==prior["upstream_providers"]
                and tensor_metadata(cache["owning_canonical_export"])==cache["owning_export_metadata"]==prior["owning_export_metadata"] and domains(cache))
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
            report["shared_cache"]=admission["cache"];report["internal_route_checkpoint_reads"]=work["routes"];report["cache_identities"]=cache["identities"]
            report["upstream_providers"]=tensor_metadata(cache["upstream_providers"])
            gate("evaluate_exact_counts",work["own_PT_loads"] == work["own_PT_load_attempts"] == 1 and work["student_fits"] == work["student_fit_attempts"] == 3
                and work["student_epochs"] == settings["epochs"]*3 and work["routes"] == work["route_attempts"] == 3)
        gate("no_new_condensation_or_test_work",all(work[key] == 0 for key in ("heads","adjoints","P_updates",
            "factor_factories","source_SGC","Q_decodes","map_fits","RMS_refits","test_evaluations","dataset_gets","graph_packs")))
        report["ordinary_GCN_graph_SpMM"]="Original fitter/replay graph products uninstrumented; source SGC products remain zero."
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
                    work["evidence_write_attempts"]+=1
                    with raw.open("xb") as stream:torch.save(evidence,stream);stream.flush();os.fsync(stream.fileno())
                    raw_info["write_completed"]=True;work["evidence_writes"]+=1
                    if phase=="prepare" and report["completed"]:work["cache_write"]+=1
                except BaseException as error:report["raw_write_error"]=repr(error)
                try:
                    raw_info["exists"]=raw.exists()
                    if raw.exists():
                        raw_info["bytes"]=raw.stat().st_size
                        try:raw_info["sha256"]=sha(raw);raw_info["hash_unknown"]=False
                        except BaseException as error:raw_info["hash_error"]=repr(error)
                except BaseException as error:report["raw_metadata_error"]=repr(error)
                if phase == "prepare" and report["completed"]:report["shared_cache"]=dict(path=str(raw),sha256=raw_info["sha256"])
            if phase=="prepare" and report["completed"]:
                report["memory_observations"]["after_cache_serialization"]=peaks()
                try:gate("prepare_exact_final_selected_counts",all(work[key]==value for key,value in science["prepare_expected_counts"].items()) and work["evidence_write_attempts"]==work["evidence_writes"]==1)
                except BaseException as error:report["failure"]=dict(type=type(error).__name__,message=str(error));report["completed"]=False
            try:
                report["readonly_exit"]=pins();report["source_exit"]=implementation_provenance();require(report["source_exit"] == packet["source"], "Exit source differs")
            except BaseException as error:report["exit_verification_error"]=repr(error)
            report["resources"]=peaks();report["passed"]=bool(report["completed"] and report["failure"] is None
                and not any(k in report for k in ("raw_write_error","raw_metadata_error","exit_verification_error"))
                and report["raw_evidence"]["write_completed"] and not report["raw_evidence"]["hash_unknown"]
                and all(v <= science["resources"][k] for k,v in report["resources"].items()))
            if not report["passed"]:report["completed"]=False
            rp=out/("prepare_report.json" if phase == "prepare" else "evaluation_report.json");write(observed(report),rp);final=peaks()
            if report["passed"] and any(v > science["resources"][k] for k,v in final.items()):
                report.update(passed=False,completed=False,resources=final,failure=dict(type="FinalSerializationResourceBoundary",message="Preserve failure; no retry"))
                with rp.open("w") as stream:json.dump(observed(report),stream,indent=2,allow_nan=False);stream.write("\n");stream.flush();os.fsync(stream.fileno())
        finally:signal.signal(signal.SIGALRM,old_handler);signal.setitimer(signal.ITIMER_REAL,*old_timer)
    require(report["passed"],"Terminal FN five-arm fullsource RMS serving failure; no retry/recipe/seed/secondary rescue")
    return str(rp)
