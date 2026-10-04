"""Unexecuted EZ: one original-map composed centroid joint CE head, physical tape; cold1 then exact25."""
import hashlib
import json
import math
import os
import platform
import resource
import signal
import time
from pathlib import Path

KIND = "single_composed_centroid_joint_CE_native_qualification_and_continuation_v1"
SCIENCE_SHA = "968b0f1c4a72ab4d75b8c78f2cad0581b570405877489c5847912065220e9459"
ENGINE_SHA = "0299164068e1f52caac6d254bcf0d6cf1bb10029ade37e627e9bcbf648e8cefe"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path, check=lambda: None):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4194304), b""):
            check(); digest.update(block)
    return digest.hexdigest()


def refs(value, found=None):
    found = {} if found is None else found
    if isinstance(value, dict):
        if set(value) == {"path", "sha256"}:
            path, digest = value["path"], value["sha256"]
            require(type(path) is str and Path(path).is_absolute() and str(Path(path).resolve()) == path,
                "Ref must be normalized ABS")
            require(type(digest) is str and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest), "Ref SHA differs")
            require(path not in found or found[path] == digest, "Conflicting ref")
            found[path] = digest
        for child in value.values(): refs(child, found)
    elif isinstance(value, list):
        for child in value: refs(child, found)
    return found


def run(protocol_path, protocol_sha256, budget, mode, stop=lambda: False):
    started = time.monotonic()
    require(Path(protocol_path).is_absolute() and str(Path(protocol_path).resolve()) == protocol_path
        and sha(protocol_path) == protocol_sha256, "Protocol path/SHA differs")
    packet = json.loads(Path(protocol_path).read_text()); science_ref = packet["scientific_contract"]
    require(science_ref["sha256"] == SCIENCE_SHA and sha(science_ref["path"]) == SCIENCE_SHA, "Science differs")
    science = json.loads(Path(science_ref["path"]).read_text())
    require(packet["kind"] == science["kind"] == KIND and budget in science["cases"] and mode in ("qualify", "continue"), "Kind/mode/budget differs")
    case = science["cases"][budget]; out = Path(packet["output_folders"][mode][budget])
    require(out.is_absolute() and str(out.resolve()) == str(out), "Output must be normalized ABS")
    out.mkdir(parents=True, exist_ok=False)
    report_path, arrays_path = out / "native_admission_report.json", out / "native_admission_arrays.pt"
    report = dict(schema=1, kind=KIND, budget=budget, mode=mode, source=packet["source"],
        scientific_contract=science_ref, protocol=dict(path=protocol_path, sha256=protocol_sha256),
        passed=False, completed=False, failure=None, gates=[],
        work=dict(PT_load_attempts=0, PT_loads=0, external_raw_adjoint=0, external_complete_G=0,
            external_material=0, external_physical_forward=0, external_backward=0, CPU_native_cast_reference=0,
            external_heads=0, external_outer=0, external_physical=0, external_Adam=0, external_P=0,
            factor_factory=0, source_SGC=0, map_fit_forward=0, RMS_refits=0, RNG=0, students=0, tests=0,
            source_owner_build_attempts=0, source_owner_builds=0, composed_owner_build_attempts=0, composed_owner_builds=0, external_G_map_attempts=0, external_G_map=0, clamp_diagnostic_attempts=0, clamp_diagnostics=0),
        raw_evidence=dict(path=str(arrays_path), exists=False, bytes=None, sha256=None,
            hash_unknown=True, hash_error=None, write_completed=False))
    report["attempts"] = dict(external_raw_adjoint=0,external_complete_G=0,
        external_physical_forward=0,external_backward=0,CPU_native_cast_reference=0)
    report["engine_invocation"] = dict(attempts=0,returns=0,interiors="not_entered")
    saved, torch = {}, None
    old_handler, old_timer = signal.getsignal(signal.SIGALRM), signal.getitimer(signal.ITIMER_REAL)

    def peaks():
        values = dict(seconds=time.monotonic()-started, peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            peak_allocated_bytes=0, peak_reserved_bytes=0)
        if torch is not None and torch.cuda.is_initialized():
            values.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(0), peak_reserved_bytes=torch.cuda.max_memory_reserved(0))
        return values

    def check():
        require(not stop() and all(v <= science["resources"][k] for k,v in peaks().items()), "Frozen stop/time/RSS/GPU boundary")

    def alarm(signum, frame): raise TimeoutError("Frozen300s native boundary")

    def gate(name, ok, **detail):
        report["gates"].append(dict(name=name, passed=bool(ok), **detail)); require(ok, name)

    def pins():
        expected = packet["readonly_files_sha256"]
        require(all(Path(p).is_absolute() and str(Path(p).resolve()) == p for p in expected), "Pin path differs")
        require(all(expected.get(p) == h for p,h in refs(dict(packet=packet, science=science)).items()), "Required ref omitted")
        return {p: sha(p, check) for p in expected}

    try:
        signal.signal(signal.SIGALRM, alarm); signal.setitimer(signal.ITIMER_REAL, max(1e-6, 300-(time.monotonic()-started)))
        report["readonly_entry"] = pins(); gate("readonly_entry_exact", report["readonly_entry"] == packet["readonly_files_sha256"])
        gate("own_entrypoint_exact", Path(packet["entrypoint"]["path"]).resolve() == Path(__file__).resolve())
        import numpy as np
        import torch
        from src import composed_centroid_joint_ce as joint, composed_centroid_joint_ce_trajectory as engine
        from src.nystrom_ce import NystromMap
        from src.transforms import FeatureTransform
        from src.dual_head_ce import _factor_digests
        from src.kernel_mean_ce import _cpu, _plain, _seal
        from src.low_rank_assignment import LowRankMoments
        from src.moments import augmented
        from src.research_loop import implementation_provenance
        from src.shared_features import _tensor_identity
        from src.soft_ce_partition import solve_head_system
        gate("engine_and_helper_exact", Path(engine.__file__).resolve() == Path(packet["engine_entrypoint"]["path"]).resolve()
            and sha(engine.__file__) == packet["engine_entrypoint"]["sha256"] == ENGINE_SHA
            and Path(joint.__file__).resolve() == Path(packet["helper_entrypoint"]["path"]).resolve()
            and sha(joint.__file__) == packet["helper_entrypoint"]["sha256"] == science["required_refs"]["helper_draft"]["sha256"]
)
        report["source_entry"] = implementation_provenance()
        gate("current_full_source_Git_exact", report["source_entry"] == packet["source"])
        req = science["required_refs"]
        bridge = json.loads(Path(packet["prerequisite_admission"]["path"]).read_text())
        gate("ROOT_current_static_source_prerequisites", bridge["passed"] is True and bridge["source"] == packet["source"]
            and bridge["engine_entrypoint"] == packet["engine_entrypoint"] and bridge["driver_entrypoint"] == packet["entrypoint"]
            and bridge["helper_entrypoint"] == packet["helper_entrypoint"]
            and bridge["CPU_science"] == req["CPU_science"])
        for role in ("ROOT","independent"):
            admission=json.loads(Path(packet["CPU_admission"][role]["path"]).read_text())
            gate(role+"_new_CPU_complete_chain_admitted",admission["passed"] is True
                and admission["actual_report"]==packet["CPU_admission"]["report"]
                and admission["scientific_contract"]==req["CPU_science"])
        root = json.loads(Path(req["actual_source135_capture_ROOT"]["path"]).read_text())
        peer = json.loads(Path(req["actual_source135_capture_independent"]["path"]).read_text())
        capture_report = json.loads(Path(case["actual_capture_report"]["path"]).read_text())
        gate("actual135_source_admitted", root["passed"] is True and peer["passed"] is True and capture_report["passed"] is True
            and root["budgets"][budget]["report"] == peer["budgets"][budget]["actual_report"] == case["actual_capture_report"]
            and root["budgets"][budget]["arrays"] == peer["budgets"][budget]["arrays"] == case["actual_capture_arrays"]
            and capture_report["source"] == capture_report["source_entry"] == capture_report["source_exit"] == case["source_capture_producer"]
            and capture_report["scientific_contract"] == req["actual_source135_capture_science"]
            and Path(case["actual_capture_arrays"]["path"]).stat().st_size == case["actual_array_bytes"])
        torch.set_num_threads(4)
        if torch.get_num_interop_threads() != 1: torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest"); torch.cuda.init(); torch.cuda.reset_peak_memory_stats(0)
        device = torch.device("cuda:0")
        report["runtime"] = joint._runtime(device)
        gate("original_native_runtime", report["runtime"] == case["original_runtime"] and platform.python_version() == "3.12.7"
            and np.__version__ == "1.26.4" and torch.get_num_interop_threads() == 1
            and all(os.environ.get(k) == "4" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")))
        def load(reference):
            check(); report["work"]["PT_load_attempts"] += 1
            value = torch.load(reference["path"], map_location="cpu", weights_only=False)
            report["work"]["PT_loads"] += 1
            return value
        payload = load(case["actual_capture_arrays"])
        saved["loaded_original_source135_payload"]=_cpu(payload)
        gate("own_original_source135_payload_header",payload["schema"]==1 and payload["kind"]==capture_report["kind"]
            and payload["budget"]==budget and payload["source"]==case["source_capture_producer"]
            and payload["scientific_contract"]==req["actual_source135_capture_science"])
        captured=payload["captured"];dims=case["dimensions"]
        n,d,b,c,k,rank=(dims[x] for x in ("nodes","physical_dimension","original_Phi_basis","classes","cells","rank"))
        native_physical=dict(moments=captured["physical_M0"],**captured["native_physical"])
        gate("own135_physical_descriptors",all(_tensor_identity(value)=={x:capture_report["descriptors"]["native_physical"][name][x]
            for x in ("shape","dtype","digest")} for name,value in native_physical.items()))
        z=captured["z_transform"]["z"].to(device);q=captured["Q"].to(device)
        hard=captured["hard"].to(device);phi=captured["original_Phi"]
        initial=[captured["native_factors"][x].to(device) for x in ("u","v")]
        gate("own_native_parameters",_factor_digests(initial,n,k,rank,device)==case["native_parameter_digests"]
            and captured["native_factors"]["data_digest"]==case["original_data_digest"]
            and captured["native_factors"]["factor_seed"]==0 and captured["native_factors"]["mixing"]==.05)
        assets=case["asset_descriptors"]
        transform=FeatureTransform(**{key:value.to(device) if torch.is_tensor(value) else value
            for key,value in captured["z_transform"]["transform"].items()})
        feature_map=NystromMap(captured["map_cache"]["anchors"].to(device),captured["map_cache"]["mapping"].to(device),"relu")
        report["work"]["source_owner_build_attempts"]+=1
        layout=joint.ComposedJointLayout(d,b,c)
        source_owner=joint.ResidentOriginalJointSource(z,phi,layout,
            dict(z=assets["z"],Phi_identity=assets["Phi_identity"]))
        report["work"]["source_owner_builds"]+=1
        saved["source_owner_descriptor"]=source_owner.descriptor()
        report["work"]["composed_owner_build_attempts"]+=1
        composed=joint.OriginalRMSComposedCentroidFeatures(transform,feature_map,layout,
            {x:assets[x] for x in ("transform","anchors","mapping")})
        report["work"]["composed_owner_builds"]+=1
        saved["composed_owner_descriptor"]=composed.descriptor()
        gate("original_physical_source_Q_map_RMS_descriptors",_tensor_identity(q)==assets["Q"]
            and _tensor_identity(z)==assets["z"] and _tensor_identity(hard)==assets["assignment"]
            and _tensor_identity(captured["H_cache"]["h"])==assets["H"]
            and source_owner.descriptor()["components"]==dict(z=assets["z"],original_Phi_identity=assets["Phi_identity"]))
        report["actual_source135"]=case["source135"]
        accepted_payload_seal=_seal(payload)
        binding = packet["trajectory_context_bindings"][budget]
        repo = Path(packet["repository"]).resolve()
        gate("repository_normalized_ABS", type(packet["repository"]) is str
            and Path(packet["repository"]).is_absolute() and str(repo) == packet["repository"])
        current_ABS = dict(packet["source"], files={str(repo/path): h for path,h in packet["source"]["files"].items()})
        gate("fixed_context_current_source_ABS", set(binding) == {"files_sha256","current_source"}
            and binding["current_source"] == current_ABS
            and all(Path(p).is_absolute() and str(Path(p).resolve()) == p for p in binding["files_sha256"])
            and all(packet["readonly_files_sha256"].get(p) == h for p,h in binding["files_sha256"].items()))
        gate("all_eight_original_assets_in_context_and_invocation",all(
            binding["files_sha256"].get(path)==digest and packet["readonly_files_sha256"].get(path)==digest
            for B in science["cases"].values() for path,digest in B["mandatory_original_asset_pins"].items()))
        source_refs = dict(nodes=n,cells=k,rank=rank,dimension=d,basis=b,classes=c,chunk_size=case["original_options"]["chunk_size"],
            factor_seed=0,mixing=.05,data_digest=case["original_data_digest"],helper_source_sha256=packet["helper_entrypoint"]["sha256"],
            original_options=case["original_options"],runtime=case["original_runtime"],device="cuda:0",asset_paths=case["asset_paths"],
            source_admission=dict(report=case["actual_capture_report"],arrays=case["actual_capture_arrays"],root_acceptance=req["actual_source135_capture_ROOT"]),
            scientific_endpoint=25,**binding)
        context=joint.build_context(source_refs,case["asset_descriptors"],case["native_parameter_digests"],
            case["native_physical_origin"],layout,source_owner.descriptor(),composed.descriptor())
        report["context"] = context; accepted_state = None
        if mode == "continue":
            accepted = packet["accepted_qualifier"][budget]
            qualifying = json.loads(Path(accepted["report"]["path"]).read_text())
            for role in ("ROOT","independent"):
                admission = json.loads(Path(accepted[role]["path"]).read_text())
                gate(role+"_qualifier_accepted", admission["passed"] is True
                    and admission["budgets"][budget]["report"] == accepted["report"] and admission["budgets"][budget]["state"] == accepted["state"])
            gate("accepted_qualifier_same_source_and_context", qualifying["passed"] is True and qualifying["kind"] == KIND and qualifying["mode"] == "qualify"
                and qualifying["source"] == qualifying["source_entry"] == qualifying["source_exit"] == packet["source"]
                and qualifying["scientific_contract"] == science_ref and qualifying["context"] == context
                and qualifying["state"] == accepted["state"])
            accepted_state = load(accepted["state"])
            saved["loaded_accepted_state"] = _cpu(accepted_state)
            gate("accepted_state_exact_frontier1", accepted_state["step"] == 1 and accepted_state["context"] == context)
            engine.validate_evaluated_state(accepted_state,accepted_state["config"],context,accepted_state["artifact_folder"])
            report["accepted_input_seal"] = _seal(accepted_state)
            old_folder=Path(accepted_state["artifact_folder"])
            old_files={str(old_folder/"optimization.csv"):accepted_state["history_sha256"],
                **{str(old_folder/"checkpoints"/name):digest for name,digest in accepted_state["checkpoint_files_sha256"].items()}}
            gate("accepted_original_CSV_checkpoint_pins_required", old_folder == Path(accepted["state"]["path"]).parent
                and all(packet["readonly_files_sha256"].get(path)==digest for path,digest in old_files.items()))
            report["accepted_origin_files"] = old_files
        execution_options = dict(case["original_options"],save_resume=True)
        report["engine_invocation"].update(attempts=1,interiors="unknown_until_return")
        state = engine.optimize_composed_joint_ce(z,q,hard,phi,initial,execution_options,context,str(out/"core"),
            1 if mode == "qualify" else 25,composed,source_owner,resume_state=accepted_state,stop=stop)
        report["engine_invocation"].update(returns=1,interiors="returned_state_counts")
        saved["returned_core_state"] = _cpu(state); report["core_work"],report["core_attempts"] = state["work"],state["attempts"]
        report["state"] = dict(path=str(out/"core"/"resume.pt"),sha256=sha(out/"core"/"resume.pt",check))
        report["evaluation_summaries"] = {str(step):dict(scalars=_plain({key:E[key] for key in
            ("step","config_digest","context_digest","source_admission_digest","initial_parameter_digests","parent_update_digest",
             "raw_CE","CE0","objective","head_work","record_digest")}),identities={key:_tensor_identity(E[key]) for key in
            ("theta","raw_rhs","head_features","physical_moments","physical_centers","physical_targets","physical_mass")},
            theta_initial_digest=_seal(E["theta_initial"]),parameters_digest=_seal(E["parameters"])) for step,E in state["snapshots"].items()}
        last_A=state["last_completed_update"]
        report["last_update_summary"] = dict(scalars=_plain({key:last_A[key] for key in
            ("step","evaluation_digest","previous_update_digest","raw_adjoint_initial_digest","adjoint_work","record_digest")}),
            identities={key:_tensor_identity(last_A[key]) for key in ("raw_adjoint","normalized_G","native_grad_U","native_grad_V")},
            parameters_after_digest=_seal(last_A["parameters_after"]),optimizer_after_digest=_seal(last_A["optimizer_after"]))
        report["core_artifacts"] = dict(history=dict(path=str(out/"core"/"optimization.csv"),sha256=state["history_sha256"]),
            checkpoints={name:dict(path=str(out/"core"/"checkpoints"/name),sha256=digest)
                for name,digest in state["checkpoint_files_sha256"].items()},resume=report["state"])
        gate("full_joint_head_and_physical_tape_domains", all(tuple(E["theta"].shape)==(c,d+b+1)
            and tuple(E["head_features"].shape)==(k,d+b) and tuple(E["physical_moments"].shape)==(k,1+d+c)
            and torch.equal(E["head_features"][:,:d],E["physical_centers"])
            and bool(torch.isfinite(E["raw_rhs"]).all()) for E in state["snapshots"].values()))
        saved["latest_composed_components"]=composed.last_returned()
        branch_observations={}
        with torch.no_grad():
            for step,E in state["snapshots"].items():
                report["work"]["clamp_diagnostic_attempts"]+=1
                centers=E["physical_centers"].to(device)
                raw=centers*transform.scale+transform.output_center+transform.center
                norm_raw=raw.norm(dim=1,keepdim=True);anchor_norm_raw=feature_map.anchors.norm(dim=1).unsqueeze(0)
                norm=norm_raw.clamp(min=1e-12);anchor_norm=anchor_norm_raw.clamp(min=1e-12)
                raw_cos=(raw@feature_map.anchors.T)/(norm*anchor_norm)
                cos=raw_cos.clamp(-1+1e-12,1-1e-12);angle=cos.acos()
                raw_relu=(angle.sin()+(math.pi-angle)*angle.cos())/math.pi
                branch_observations[str(step)]=_cpu(dict(raw_H_centroids=raw,norm_unclamped=norm_raw,
                    anchor_norm_unclamped=anchor_norm_raw,cosine_unclamped=raw_cos,relu_unclamped=raw_relu))
                saved["original_clamp_observations"]=branch_observations
                report.setdefault("original_clamp_scalar_observations",{})[str(step)]=dict(
                    input_norm_clamped=int((norm_raw<1e-12).sum()),anchor_norm_clamped=int((anchor_norm_raw<1e-12).sum()),
                    cosine_lower_clamped=int((raw_cos<(-1+1e-12)).sum()),cosine_upper_clamped=int((raw_cos>(1-1e-12)).sum()),
                    relu_lower_clamped=int((raw_relu<(-1+1e-12)).sum()),relu_upper_clamped=int((raw_relu>(1-1e-12)).sum()),
                    norm_min=float(norm_raw.min()),norm_max=float(norm_raw.max()),cosine_raw_min=float(raw_cos.min()),
                    cosine_raw_max=float(raw_cos.max()),relu_raw_min=float(raw_relu.min()),relu_raw_max=float(raw_relu.max()),
                    observational_only=True,smoothness_required=False)
                report["work"]["clamp_diagnostics"]+=1
                check()
        expected = science["native_core"]["qualify_cold0_to1" if mode == "qualify" else "total_qualifier_plus_continuation"]
        mapped = {"physical_forward":"physical_forward","endpoint_map":"endpoint_map","G_map":"G_map",
            "stationary_heads":"head","raw_outer_CE_RHS":"outer","core_raw_adjoints":"adjoint",
            "complete_G":"complete_G","backward":"backward","Adam":"Adam","P_updates":"P_updates",
            "reattachment":"reattachment","checkpoint_writes":"checkpoint_writes",
            "resume_writes":"resume_writes","checkpoint_copies":"checkpoint_copies"}
        gate("exact_core_role_counts", all(state["work"][mapped[name]] == value for name,value in expected.items()))
        if mode == "qualify":
            E, A = state["current_evaluation"], state["last_completed_update"]
            params = [p.to(device) for p in E["parameters"]]
            centers,targets,mass = (E[x].to(device) for x in ("head_features","physical_targets","physical_mass"))
            report["attempts"]["external_raw_adjoint"] += 1
            vector, diagnostic = solve_head_system(augmented(centers),targets,torch.full_like(mass,1/k),E["theta"].to(device),
                case["original_options"]["penalty"],E["raw_rhs"].to(device),rtol=case["original_options"]["cg_rtol"],
                max_iter=case["original_options"]["cg_max_iter"],initial=A["raw_adjoint"].to(device),
                cg_check_interval=case["original_options"].get("cg_check_interval",1))
            saved["external_raw_adjoint"] = _cpu(dict(vector=vector,diagnostic=diagnostic))
            report["work"]["external_raw_adjoint"] += 1
            gate("external_raw_adjoint_converged", diagnostic.get("cg_converged") is True and bool(torch.isfinite(vector).all()))
            report["attempts"]["external_complete_G"] += 1
            def observed_map(callable_map,physical_centers):
                report["work"]["external_G_map_attempts"]+=1
                mapped=callable_map(physical_centers)
                report["work"]["external_G_map"]+=1
                saved["external_G_composed_components"]=composed.last_returned()
                return mapped
            G = joint.complete_moment_cotangent(E["physical_moments"].to(device),layout,composed,
                E["theta"].to(device),vector,case["original_options"]["penalty"],E["CE0"],map_call=observed_map)
            report["work"]["external_complete_G"] += 1; saved["external_complete_G"] = _cpu(G); check()
            material = source_owner.material_on(q,device)
            report["work"]["external_material"] += 1; saved["external_material"] = _cpu(material); repeats = []
            for repeat in range(3):
                fresh = [p.detach().clone().requires_grad_() for p in params]
                report["attempts"]["external_physical_forward"] += 1
                M = LowRankMoments.apply(*fresh,hard,material,.05,case["original_options"]["chunk_size"])
                report["work"]["external_physical_forward"] += 1
                item = dict(M=_cpu(M)); repeats.append(item); saved["native_repeats"] = repeats
                gate("fixedG_forward_matches_E1_"+str(repeat), _tensor_identity(M.detach()) == _tensor_identity(E["physical_moments"]))
                report["attempts"]["external_backward"] += 1
                M.backward(G); report["work"]["external_backward"] += 1
                item.update(U=_cpu(fresh[0].grad),V=_cpu(fresh[1].grad)); check()
            gate("three_native_fixedG_byte_repeats", all(_seal(item) == _seal(repeats[0]) for item in repeats[1:]))
            report["attempts"]["CPU_native_cast_reference"] += 1
            with torch.no_grad():
                U,V = (p.detach().cpu() for p in params); QG = G.detach().cpu(); X = material.detach().cpu(); assignment=hard.cpu()
                prior = torch.full((n,k),float(np.log(.05/k)),dtype=torch.float32)
                prior.scatter_(1,assignment[:,None],float(np.log(1-.05+.05/k)))
                logits = prior+U@V.T/math.sqrt(rank); probability=logits.double().softmax(1)
                direction=X@QG.T/n
                block=(probability*(direction-(probability*direction).sum(1,keepdim=True))).to(torch.float32)/math.sqrt(rank)
                reference=dict(U=block@V,V=block.T@U)
                report["work"]["CPU_native_cast_reference"] += 1
                saved["CPU_reference"] = _cpu(dict(logits=logits,probability=probability,direction=direction,whole_GP_native_block=block,gradients=reference))
                errors = {}; pairs=dict(U=(repeats[0]["U"],reference["U"]),V=(repeats[0]["V"],reference["V"]))
                pairs["full"] = (torch.cat([pairs[x][0].flatten() for x in ("U","V")]),torch.cat([pairs[x][1].flatten() for x in ("U","V")]))
                for name,(native,ref) in pairs.items():
                    delta=native.double()-ref.double(); absolute=float(delta.abs().max()); error_norm=float(torch.linalg.vector_norm(delta)); ref_norm=float(torch.linalg.vector_norm(ref.double()))
                    relative=error_norm/ref_norm if ref_norm>0 else (0.0 if error_norm==0 else None)
                    relative=relative if relative is not None and math.isfinite(relative) else None
                    errors[name]=dict(absolute_max=absolute if math.isfinite(absolute) else None,
                        difference_L2=error_norm if math.isfinite(error_norm) else None,reference_L2=ref_norm if math.isfinite(ref_norm) else None,
                        relative_L2=relative,both_passed=math.isfinite(absolute) and absolute<=5e-7 and relative is not None and relative<=2e-5)
                saved["gradient_errors"] = report["gradient_errors"] = errors
                norms={"U1":float(torch.linalg.vector_norm(U)),"V1":float(torch.linalg.vector_norm(V)),
                    "native_dU":float(torch.linalg.vector_norm(repeats[0]["U"])),"native_dV":float(torch.linalg.vector_norm(repeats[0]["V"]))}
                report["nonzero_norms"]={name:value if math.isfinite(value) else None for name,value in norms.items()}
            gate("nonzero_finite_endpoint1_native_parameters_and_gradients", all(math.isfinite(v) and v>0 for v in norms.values())
                and all(bool(torch.isfinite(item[name]).all()) for item in repeats for name in ("M","U","V")))
            for name,value in errors.items(): gate(name+"_BOTH_gradient_bounds",value["both_passed"],**value)
            gate("external_diagnostic_did_not_change_core_E_A_state", _seal(state) == _seal(saved["returned_core_state"])
                and sha(out/"core"/"resume.pt",check) == report["state"]["sha256"])
        else:
            gate("entire_accepted_input_state_unchanged", _seal(accepted_state)==report["accepted_input_seal"])
            gate("accepted_evaluation_and_old_checkpoints_unchanged", _seal(accepted_state["current_evaluation"]) == _seal(state["snapshots"][1])
                and all(sha(Path(state["artifact_folder"])/"checkpoints"/name,check) == digest for name,digest in accepted_state["checkpoint_files_sha256"].items()))
            gate("accepted_original_CSV_and_checkpoint_files_unchanged", all(sha(path,check)==digest for path,digest in old_files.items()))
        gate("own_original135_payload_unchanged",_seal(payload)==accepted_payload_seal)
        expected_work=dict.fromkeys(report["work"],0)
        expected_work.update(PT_load_attempts=1 if mode=="qualify" else 2,PT_loads=1 if mode=="qualify" else 2,
            external_raw_adjoint=1 if mode=="qualify" else 0,external_complete_G=1 if mode=="qualify" else 0,
            external_material=1 if mode=="qualify" else 0,external_physical_forward=3 if mode=="qualify" else 0,
            external_backward=3 if mode=="qualify" else 0,CPU_native_cast_reference=1 if mode=="qualify" else 0,
            source_owner_build_attempts=1,source_owner_builds=1,composed_owner_build_attempts=1,composed_owner_builds=1,
            external_G_map_attempts=1 if mode=="qualify" else 0,external_G_map=1 if mode=="qualify" else 0,
            clamp_diagnostic_attempts=2 if mode=="qualify" else 3,clamp_diagnostics=2 if mode=="qualify" else 3)
        gate("exact_wrapper_return_counts", report["work"] == expected_work)
        gate("exact_external_attempt_return_counts", report["attempts"] == {name:expected_work[name] for name in report["attempts"]})
        torch.cuda.synchronize(device); check(); report["completed"] = True
    except BaseException as error:
        report["failure"] = dict(type=type(error).__name__,message=str(error))
        report["inflight_operation_interiors"] = "unknown; attempts and owning returned values remain distinct"
        core_failure=out/"core"/"failure.json"
        if core_failure.exists(): report["core_failure_evidence"] = dict(path=str(core_failure),sha256=sha(core_failure))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        try:
            if torch is not None:
                raw=report["raw_evidence"]
                try:
                    with arrays_path.open("xb") as stream:
                        torch.save(dict(schema=1,kind=KIND,budget=budget,mode=mode,source=packet["source"],scientific_contract=science_ref,saved=saved),stream)
                        stream.flush(); os.fsync(stream.fileno())
                    raw["write_completed"]=True
                except BaseException as error: report["raw_write_error"]=repr(error)
                try:
                    raw["exists"]=arrays_path.exists()
                    if raw["exists"]:
                        raw["bytes"]=arrays_path.stat().st_size
                        try: raw["sha256"]=sha(arrays_path);raw["hash_unknown"]=False
                        except BaseException as error: raw["hash_error"]=repr(error)
                except BaseException as error: report["raw_metadata_error"]=repr(error)
            try:
                report["readonly_exit"],report["source_exit"]=pins(),implementation_provenance()
                require(report["readonly_exit"]==packet["readonly_files_sha256"] and report["source_exit"]==packet["source"],"Exit pins/source changed")
            except BaseException as error: report["exit_verification_error"]=repr(error)
            report["resources"]=peaks()
            report["passed"]=bool(report["completed"] and report["failure"] is None
                and not any(k in report for k in ("raw_write_error","raw_metadata_error","exit_verification_error"))
                and report["raw_evidence"]["write_completed"] and not report["raw_evidence"]["hash_unknown"]
                and all(v<=science["resources"][k] for k,v in report["resources"].items()))
            with report_path.open("x") as stream:
                json.dump(report,stream,indent=2,allow_nan=False);stream.write("\n");stream.flush();os.fsync(stream.fileno())
            final=peaks()
            if report["passed"] and any(v>science["resources"][k] for k,v in final.items()):
                report.update(passed=False,resources=final,failure=dict(type="FinalSerializationResourceBoundary",message="Preserve namespace; no retry"))
                with report_path.open("w") as stream:
                    json.dump(report,stream,indent=2,allow_nan=False);stream.write("\n");stream.flush();os.fsync(stream.fileno())
        finally:
            signal.signal(signal.SIGALRM,old_handler);signal.setitimer(signal.ITIMER_REAL,*old_timer)
    require(report["passed"],"Terminal native admission failure; inspect owning evidence, no retry")
    return str(report_path)
