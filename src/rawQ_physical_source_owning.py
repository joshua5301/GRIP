"""Methods-agnostic cache-only original raw-Q/physical source owner; no critic."""
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import signal
import sys
import time

KIND = "original_rawQ_physical_source_owning_capture_v1"
CONTRACT_SHA = "f59afe40a114fd68daf3d5dd2ef6bf4434c8021a80694a98b96795e870b3c4bb"


def require(ok, message):
    if not ok: raise ValueError(message)


def sha(path, check=lambda: None):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block); check()
    return digest.hexdigest()


def refs(value):
    found = {}
    def visit(x):
        if isinstance(x, dict):
            if isinstance(x.get("path"), str) and isinstance(x.get("sha256"), str):
                p, h = Path(x["path"]), x["sha256"]
                require(p.is_absolute() and str(p) == str(p.resolve()), "Reference must be normalized ABS")
                require(len(h) == 64 and all(c in "0123456789abcdef" for c in h), "Reference SHA invalid")
                require(str(p) not in found or found[str(p)] == h, "Reference conflict")
                found[str(p)] = h
            for item in x.values(): visit(item)
        elif isinstance(x, list):
            for item in x: visit(item)
    visit(value)
    return found


def run(protocol_path, protocol_sha256, budget, stop=lambda: False):
    started = time.monotonic(); path = Path(protocol_path)
    require(callable(stop) and path.is_absolute() and str(path) == str(path.resolve())
        and sha(path) == protocol_sha256, "Protocol/stop differs")
    packet = json.loads(path.read_text()); science = packet["scientific_contract"]
    require(type(packet["schema"]) is int and packet["schema"] == 1 and packet["kind"] == KIND
        and science["sha256"] == CONTRACT_SHA and sha(science["path"]) == CONTRACT_SHA, "Packet/science differs")
    contract = json.loads(Path(science["path"]).read_text())
    require(budget in contract["budgets"], "Unknown original cond0 budget")
    B = contract["cases"][budget]; folder = Path(packet["output_folders"][budget])
    require(folder.is_absolute() and str(folder) == str(folder.resolve()) and not folder.exists(), "Fresh ABS output required")
    folder.mkdir(parents=True); raw_path, report_path = folder/"source_capture_arrays.pt", folder/"source_capture_report.json"
    work = {k: 0 for k in B["expected_counts"]}; captured = {}; torch = None
    report = dict(schema=1, kind=KIND, budget=budget, passed=False, source=packet["source"],
        scientific_contract=science, protocol=dict(path=str(path), sha256=protocol_sha256), work=work,
        gates=[], descriptors={}, supplier_producers={}, failure=None, scope=contract["scope"],
        no_critic_context_or_CE0_or_resume=True, new_aggregation_head_update_student_qualified=False,
        raw_evidence=dict(path=str(raw_path), exists=False, bytes=None, sha256=None,
            hash_unknown=True, hash_error=None, write_completed=False))
    old_profile = sys.getprofile(); old_handler = signal.getsignal(signal.SIGALRM)
    old_timer = signal.getitimer(signal.ITIMER_REAL); profile_work = {}
    def peaks():
        p = dict(seconds=time.monotonic()-started, peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            peak_allocated_bytes=0, peak_reserved_bytes=0)
        if torch is not None and torch.cuda.is_initialized():
            p.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(0), peak_reserved_bytes=torch.cuda.max_memory_reserved(0))
        return p
    def check():
        p = peaks()
        if stop() or any(p[k] > contract["limits"][k] for k in p): raise InterruptedError("Frozen resource/stop boundary")
    def alarm(signum, frame): raise TimeoutError("Frozen300s deadline")
    def gate(name, ok, **details):
        report["gates"].append(dict(name=name, passed=bool(ok), **details)); require(ok, name)
    def read(ref): return json.loads(Path(ref["path"]).read_text())
    def pins():
        expected = packet["readonly_files_sha256"]
        require(all(Path(p).is_absolute() and str(Path(p)) == str(Path(p).resolve()) for p in expected), "Readonly keys not ABS")
        needed = refs(dict(packet=packet, science=contract))
        require(all(expected.get(p) == h for p,h in needed.items()), "Required readonly ref omitted")
        return {p:sha(p,check) for p in expected}
    try:
        signal.signal(signal.SIGALRM, alarm); signal.setitimer(signal.ITIMER_REAL, max(1e-6,300-(time.monotonic()-started)))
        report["readonly_entry"] = pins(); gate("readonly_entry_exact", report["readonly_entry"] == packet["readonly_files_sha256"])
        gate("entrypoint_exact", Path(packet["entrypoint"]["path"]).resolve() == Path(__file__).resolve())
        from src.research_loop import implementation_provenance
        report["source_entry"] = implementation_provenance(); gate("current_full_source_Git", report["source_entry"] == packet["source"])
        bridge = read(packet["prerequisite_admission"])
        expected = {k:B[k] for k in ("prepare_report","prepare_ROOT","prepare_independent","geometry_report","geometry_ROOT","original_certificate")}
        gate("ROOT_actual_historical_schema_bridge", bridge["passed"] is True and bridge["source"] == packet["source"]
            and bridge["scientific_contract"] == science and bridge["methods_agnostic_source_only"] is True
            and bridge["budgets"][budget] == expected)
        prep, root, geo, groot = (read(B[k]) for k in ("prepare_report","prepare_ROOT","geometry_report","geometry_ROOT"))
        gate("literal_prepare_producer_and_packet", prep["passed"] is True and prep["budget"] == budget
            and prep["source"] == root["source"] == B["prepare_producer"] and root["passed"] is True
            and root[B["prepare_ROOT_report_key"]] == B["prepare_report"] and prep["native_kernel_mean_packet"] == B["packet"]
            and prep["kernel_mean_context"] == B["old_context"])
        gate("literal_original_geometry_supplier", geo["passed"] is True and groot["passed"] is True
            and groot["actual_report"] == B["geometry_report"] and groot["source"] == geo["source"] == B["geometry_producer"])
        cert = read(B["original_certificate"])["roots"][B["certificate_root"]]
        gate("original_P0_certificate_supplier", cert["source_P0_linear_certificate_passed"] is True
            and cert["native_buffers"]["Q"] == B["certificate_Q_array_identity"])
        report["supplier_producers"] = dict(physical_packet=B["prepare_producer"], geometry=B["geometry_producer"],
            raw_Q=B["Q"], original_certificate=B["original_certificate"])
        gate("intrinsic_source_domain_cap", B["logical_source_bytes"] <= contract["limits"]["intrinsic_source_cap_bytes"])
        import numpy as np
        import torch
        from src.citation_graph_factor import _transform
        from src.dual_head_ce import _factor_digests
        from src.io import array_digest
        from src.kernel_mean_ce import _seal
        from src.low_rank_assignment import LowRankMoments, initialize_factors
        from src.moments import decode_moments, make_material
        from src.nystrom_ce import NystromMap, _cache_identity, _content_digest
        from src.shared_features import _map_identity, _tensor_identity, _validate_map_state
        from src.soft_ce_partition import head_gradient, solve_inner, solve_inner_newton_first, solve_head_system
        from src.transforms import FeatureTransform, fit_transform
        torch.set_num_threads(4)
        if torch.get_num_interop_threads() != 1: torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest"); torch.use_deterministic_algorithms(B["historical_runtime"]["deterministic"])
        device = torch.device("cuda:0"); torch.cuda.init(); torch.cuda.reset_peak_memory_stats(device)
        report["runtime"] = dict(Python=platform.python_version(), Torch=torch.__version__, NumPy=np.__version__, threads=torch.get_num_threads(),
            device=str(device), default_dtype=str(torch.get_default_dtype()), AMP=torch.is_autocast_enabled("cuda"),
            TF32_matmul=torch.backends.cuda.matmul.allow_tf32, TF32_cudnn=torch.backends.cudnn.allow_tf32,
            matmul_precision=torch.get_float32_matmul_precision(), deterministic=torch.are_deterministic_algorithms_enabled())
        gate("exact_original_runtime", report["runtime"] == B["historical_runtime"] and torch.get_num_interop_threads() == 1
            and all(os.environ.get(k) == "4" for k in contract["runtime"]["environment_threads"]))
        def own(value):
            if torch.is_tensor(value): return value.detach().cpu().clone()
            if isinstance(value,dict): return {k:own(v) for k,v in value.items()}
            if isinstance(value,(tuple,list)): return [own(v) for v in value]
            return value
        def load(role, ref, where, keys=None):
            check(); work["PT_load_attempts"] += 1
            value = torch.load(ref["path"], map_location=where, weights_only=False); work["PT_loads"] += 1
            captured["loaded_"+role] = own(value if keys is None else {k:value[k] for k in keys if k in value})
            return value
        def matrix(value, shape, dtype=torch.float64, where=device):
            return torch.is_tensor(value) and list(value.shape) == shape and value.dtype == dtype and value.device == where
        methods = {LowRankMoments.forward.__code__:"LowRank_forwards", LowRankMoments.backward.__code__:"LowRank_backwards",
            initialize_factors.__code__:"factor_factories", decode_moments.__code__:"physical_decodes", make_material.__code__:"make_material",
            NystromMap.fit.__func__.__code__:"map_fits", NystromMap.__call__.__code__:"map_forwards", fit_transform.__wrapped__.__code__:"RMS_refits",
            head_gradient.__code__:"head_gradient", solve_inner.__code__:"solve_inner", solve_inner_newton_first.__code__:"solve_inner_newton_first",
            solve_head_system.__code__:"solve_head_system"}
        profile_work = {v:0 for v in methods.values()}
        def profile(frame,event,arg):
            if event == "call" and frame.f_code in methods: profile_work[methods[frame.f_code]] += 1
        sys.setprofile(profile)
        n,d,b,c,k,r = (B["dimensions"][x] for x in ("N","D","B","C","K","r")); assets = B["expected_assets"]
        inputs = load("inputs",B["inputs"],device,("z","transform")); z = inputs["z"]; transform = FeatureTransform(**inputs["transform"])
        hard = load("hard",B["hard"],device); qsource = load("Q_source",B["Q"]["input"],device,
            ("logits","gamma") if B["Q"]["kind"] == "cached_original_logits_once" else ("new_source_Q",))
        if B["Q"]["kind"] == "cached_original_logits_once":
            logits = qsource["logits"]; report["stored_logits"] = dict(_tensor_identity(logits),device=str(logits.device))
            gate("original_logits_domain", matrix(logits,[n,c]) and str(logits.dtype) == B["Q"]["expected_logits_dtype"]
                and not logits.requires_grad and bool(torch.isfinite(logits).all()) and B["Q"]["train_target_mix"] == 0.0)
            check(); work["Q_decode_attempts"] += 1; q = (logits/B["Q"]["T"]).softmax(1).double(); work["Q_decodes"] += 1
            captured["raw_Q"] = own(q)
        else:
            qroot,qreport,qpeer = (read(B["Q"][x]) for x in ("ROOT","report","independent"))
            gate("source120_own_Q_admitted", qroot["passed"] is True and qpeer["passed"] is True and qreport["passed"] is True
                and qroot["actual_report"] == B["Q"]["report"] and qroot["actual_arrays"] == B["Q"]["input"]
                and qreport["raw_evidence"]["sha256"] == B["Q"]["input"]["sha256"])
            check(); work["Q_copy_attempts"] += 1; q = qsource[B["Q"]["selector"]].detach().to(device).clone(); work["Q_copies"] += 1
            captured["raw_Q"] = own(q)
        report["descriptors"]["Q"] = _tensor_identity(q)
        gate("exact_original_full_Q", matrix(q,[n,c]) and not q.requires_grad and bool(torch.isfinite(q).all())
            and bool((q>=0).all()) and bool((q.sum(1)>0).all()) and bool((q.sum(0)>0).all())
            and report["descriptors"]["Q"] == B["Q"]["expected_Q"])
        native = load("native",B["native"],device); parameters = [native["u"],native["v"]]
        hstate = load("H",B["H"],"cpu",("schema","kind","source_digest","identity","h")); h = hstate["h"]
        mstate = load("map",B["map"],"cpu")
        artifact = load("physical_packet",B["packet"],"cpu",("schema","kind","source","budget","context","initial_parameters","original_moments","original_serving","content_seal"))
        gate("original_inputs_RMS_H_hard", matrix(z,[n,d]) and not z.requires_grad and bool(torch.isfinite(z).all())
            and matrix(h,[n,d],torch.float32,torch.device("cpu")) and hstate["schema"] == 1 and hstate["kind"] == "shared_h"
            and _transform(transform,d) == assets["transform"] and matrix(hard,[n],torch.int64) and int(hard.min()) == 0
            and int(hard.max()) == k-1 and len(torch.unique(hard)) == k
            and all(_tensor_identity(v)==assets[name] for name,v in (("z",z),("H",h),("assignment",hard))) and hstate["identity"] == assets["H"])
        gate("literal_packet_and_native_owner", artifact["schema"] == 1 and artifact["kind"] == "fixed_original_Phi_kernel_mean_uniform_CE_native_NODE_v1"
            and artifact["source"] == B["prepare_producer"] and artifact["budget"] == budget and artifact["context"] == B["old_context"]
            and artifact["content_seal"] == _seal({x:v for x,v in artifact.items() if x != "content_seal"})
            and _factor_digests(parameters,n,k,r,device) == B["expected_native_parameter_digests"]
            and _factor_digests(artifact["initial_parameters"],n,k,r,torch.device("cpu")) == B["expected_native_parameter_digests"]
            and bool(parameters[0].eq(0).all()) and native["factor_seed"] == B["factor_seed"] and native["mixing"] == B["mixing"]
            and native["data_digest"] == B["original_data_digest"]
            and array_digest(captured["loaded_inputs"]["z"].numpy(),captured["raw_Q"].numpy(),captured["loaded_hard"].numpy()) == B["original_data_digest"])
        anchors,mapping = _validate_map_state(mstate,_map_identity(h,B["map_request_basis"],B["map_seed"],B["kernel"]))
        gate("original_map_owner", _tensor_identity(anchors)==assets["anchors"] and _tensor_identity(mapping)==assets["mapping"])
        check(); work["Phi_mmap_attempts"] += 1; phi = np.load(B["Phi"]["path"],mmap_mode="r",allow_pickle=False); work["Phi_mmaps"] += 1
        captured["original_Phi"] = torch.from_numpy(np.array(phi,copy=True,order="C"))
        identity = _cache_identity(h,NystromMap(anchors,mapping,B["kernel"]),(n,b),B["chunk_size"],check)
        identity["phi_digest"] = _content_digest(phi,B["chunk_size"],check)
        gate("mandatory_original_Phi_identity", phi.shape==(n,b) and phi.dtype==np.float64 and not phi.flags.writeable
            and bool(np.isfinite(phi).all()) and identity==assets["Phi_identity"]==read(B["Phi_metadata"]))
        M = artifact["original_moments"].detach().to(device).clone(); captured["physical_M0"] = own(M)
        gate("original_physical_M0_exact", matrix(M,[k,1+d+c]) and _tensor_identity(M)==B["expected_physical_origin"]["moments"])
        check(); work["physical_decode_attempts"] += 1; centers,targets,mass = decode_moments(M,d); work["physical_decodes"] += 1
        captured["native_physical"] = own(dict(centers=centers,targets=targets,mass=mass))
        report["descriptors"]["native_physical"] = {name:dict(_tensor_identity(value),device=str(value.device),requires_grad=value.requires_grad)
            for name,value in dict(moments=M,centers=centers,targets=targets,mass=mass).items()}
        gate("own_native_quotient_domain", all(v.device==device and not v.requires_grad and v.grad_fn is None and bool(torch.isfinite(v).all())
            for v in (M,centers,targets,mass)) and bool((mass>0).all()) and bool((targets>=0).all())
            and bool((targets.sum(1)>0).all()) and bool((targets.sum(0)>0).all()))
        report["descriptors"].update(z=_tensor_identity(z),H=_tensor_identity(h),assignment=_tensor_identity(hard),
            transform=_transform(transform,d),anchors=_tensor_identity(anchors),mapping=_tensor_identity(mapping),Phi_identity=identity)
        report["original_data_digest"] = B["original_data_digest"]
        report["historical_CPU_quotients_supplier_only"] = B["expected_physical_origin"]
        gate("exact_capture_counts", work==B["expected_counts"])
        gate("observed_original_decode_and_no_forbidden_calls", all(v==(1 if name=="physical_decodes" else 0) for name,v in profile_work.items()))
        torch.cuda.synchronize(device); check(); report["capture_completed"] = True
    except BaseException as exc:
        report["failure"] = dict(type=type(exc).__name__,message=str(exc))
    finally:
        sys.setprofile(old_profile); signal.setitimer(signal.ITIMER_REAL,0); report["observed_profile_calls"] = profile_work
        if torch is not None:
            raw = report["raw_evidence"]
            try:
                with raw_path.open("xb") as stream:
                    torch.save(dict(schema=1,kind=KIND,budget=budget,source=packet["source"],scientific_contract=science,captured=captured),stream)
                    stream.flush(); os.fsync(stream.fileno())
                raw["write_completed"] = True
            except BaseException as exc: report["raw_write_error"] = repr(exc)
            try:
                raw["exists"] = raw_path.exists()
                if raw["exists"]:
                    raw["bytes"] = raw_path.stat().st_size
                    try: raw["sha256"] = sha(raw_path); raw["hash_unknown"] = False
                    except BaseException as exc: raw["hash_error"] = repr(exc)
            except BaseException as exc: report["raw_metadata_error"] = repr(exc)
        try:
            report["readonly_exit"] = pins(); report["source_exit"] = implementation_provenance()
            require(report["readonly_exit"]==packet["readonly_files_sha256"] and report["source_exit"]==packet["source"], "Exit source/pin differs")
        except BaseException as exc: report["exit_verification_error"] = repr(exc)
        report["resources"] = peaks()
        report["passed"] = bool(report.get("capture_completed") and report["failure"] is None
            and not any(k in report for k in ("exit_verification_error","raw_write_error","raw_metadata_error"))
            and report["raw_evidence"]["write_completed"] and not report["raw_evidence"]["hash_unknown"]
            and all(report["resources"][k]<=contract["limits"][k] for k in report["resources"]))
        try:
            with report_path.open("x") as stream:
                json.dump(report,stream,indent=2,allow_nan=False); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
            final = peaks()
            if report["passed"] and any(final[k]>contract["limits"][k] for k in final):
                report.update(passed=False,resources=final,failure=dict(type="FinalSerializationResourceBoundary",message="Final report exceeds frozen resources"))
                with report_path.open("w") as stream: json.dump(report,stream,indent=2,allow_nan=False); stream.write("\n")
        finally:
            signal.signal(signal.SIGALRM,old_handler); signal.setitimer(signal.ITIMER_REAL,*old_timer)
    require(report["passed"], "Terminal methods-agnostic source capture failure; preserve namespace, no retry")
    return str(report_path)
