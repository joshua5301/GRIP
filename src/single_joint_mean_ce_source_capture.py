"""Cache-only original-Q and owning, unqualified literal joint-M0 capture."""
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import signal
import sys
import time

KIND = "single_joint_mean_CE_cache_only_source_admission_capture_v1"
CONTRACT_SHA = "88a37d983562d4ac24810af8542de067f6842930ba5585d57ed20460a50edf87"
HELPER_SHA = "91a7a5f29f4293e28f4b94c3271692502bdba3c1e51ec9cddbd07d0e0d633cfc"


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _sha(path, check=lambda: None):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block); check()
    return h.hexdigest()


def _refs(value):
    found = {}
    def visit(x):
        if isinstance(x, dict):
            if "path" in x and "sha256" in x:
                p, h = Path(x["path"]), x["sha256"]
                _require(p.is_absolute() and str(p) == str(p.resolve()), "Reference must be normalized ABS")
                _require(type(h) is str and len(h) == 64 and all(c in "0123456789abcdef" for c in h), "Invalid reference SHA")
                _require(str(p) not in found or found[str(p)] == h, "Conflicting reference")
                found[str(p)] = h
            for v in x.values(): visit(v)
        elif isinstance(x, list):
            for v in x: visit(v)
    visit(value)
    return found


def run(protocol_path, protocol_sha256, budget, stop=lambda: False):
    started = time.monotonic()
    path = Path(protocol_path)
    _require(callable(stop) and path.is_absolute() and str(path) == str(path.resolve())
        and _sha(path) == protocol_sha256, "Protocol path/SHA or stop differs")
    packet = json.loads(path.read_text())
    _require(type(packet["schema"]) is int and packet["schema"] == 1 and packet["kind"] == KIND, "Schema/kind differs")
    science = packet["scientific_contract"]
    _require(science["sha256"] == CONTRACT_SHA and _sha(science["path"]) == CONTRACT_SHA, "Science differs")
    contract = json.loads(Path(science["path"]).read_text())
    _require(budget in contract["budgets"], "Unknown fixed budget")
    B = contract["cases"][budget]
    folder = Path(packet["output_folders"][budget])
    _require(folder.is_absolute() and str(folder) == str(folder.resolve()) and not folder.exists(), "Fresh normalized ABS namespace required")
    folder.mkdir(parents=True)
    arrays_path, report_path = folder / "source_capture_arrays.pt", folder / "source_capture_report.json"
    work = {k: 0 for k in contract["exact_counts"]}
    captured = {}
    report = dict(schema=1, kind=KIND, budget=budget, passed=False, source=packet["source"],
        scientific_contract=science, protocol=dict(path=str(path), sha256=protocol_sha256), work=work,
        gates=[], descriptors={}, failure=None, mathematical_joint_M0_qualified=False,
        native_head_gradient_update_student_qualified=False, helper_context_built=False,
        scope=contract["scope"], raw_evidence=dict(path=str(arrays_path), exists=False, bytes=None,
            sha256=None, hash_unknown=True, hash_error=None, write_completed=False))
    torch = None
    old_profile = sys.getprofile()
    old_handler, old_timer = signal.getsignal(signal.SIGALRM), signal.getitimer(signal.ITIMER_REAL)
    profile_work = {k: 0 for k in ("LowRank_forward", "LowRank_backward", "initialize_factors",
        "make_material", "decode_moments", "map_fit", "map_forward", "RMS_refit")}
    def peaks():
        p = dict(seconds=time.monotonic()-started, peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            peak_allocated_bytes=0, peak_reserved_bytes=0)
        if torch is not None and torch.cuda.is_initialized():
            p.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(0), peak_reserved_bytes=torch.cuda.max_memory_reserved(0))
        return p
    def check():
        p = peaks()
        if stop() or any(p[k] > contract["resource_limits"][k] for k in p):
            raise InterruptedError("Frozen stop/time/RSS/GPU boundary reached")
    def alarm(signum, frame): raise TimeoutError("Frozen 300s deadline reached")
    def pins():
        expected = packet["readonly_files_sha256"]
        _require(all(Path(p).is_absolute() and str(Path(p)) == str(Path(p).resolve()) for p in expected), "Readonly keys must be normalized ABS")
        needed = _refs(dict(packet=packet, contract=contract))
        _require(all(expected.get(p) == h for p, h in needed.items()), "Required readonly ref omitted")
        return {p: _sha(p, check) for p in expected}
    def gate(name, ok, **details):
        report["gates"].append(dict(name=name, passed=bool(ok), **details))
        _require(ok, name)
    try:
        signal.signal(signal.SIGALRM, alarm)
        signal.setitimer(signal.ITIMER_REAL, max(1e-6, 300-(time.monotonic()-started)))
        report["readonly_entry"] = pins()
        gate("readonly_entry_exact", report["readonly_entry"] == packet["readonly_files_sha256"])
        gate("entrypoint_exact", Path(packet["entrypoint"]["path"]).resolve() == Path(__file__).resolve())
        from src.research_loop import implementation_provenance
        report["source_entry"] = implementation_provenance()
        gate("current_full_source_and_Git_exact", report["source_entry"] == packet["source"])
        refs = contract["required_refs"]
        admission = json.loads(Path(packet["prerequisite_admission"]["path"]).read_text())
        parity = {k: refs["helper_parity_"+k] for k in ("report", "ROOT", "independent")}
        gate("ROOT_prerequisites_admitted", admission["passed"] is True
            and admission["helper_entrypoint"] == packet["helper_entrypoint"]
            and admission["helper_parity"] == parity and type(admission["positive_cases_passed"]) is int
            and admission["positive_cases_passed"] == 3 and type(admission["negative_cases_passed"]) is int
            and admission["negative_cases_passed"] == 8 and admission["cache_only_source_capture_admitted"] is True)
        typed_refs = admission["typed_behavior"]
        gate("typed_behavior_actual_refs_pinned", set(typed_refs) == {"report", "ROOT", "independent"}
            and all(packet["readonly_files_sha256"].get(p) == h for p, h in _refs(typed_refs).items()))
        for label, ref in dict(parity=parity, typed=typed_refs).items():
            for role, r in ref.items():
                gate(label+"_"+role+"_passed", json.loads(Path(r["path"]).read_text())["passed"] is True)
        old_report = json.loads(Path(B["historical_prepare_report"]["path"]).read_text())
        old_ROOT = json.loads(Path(refs["historical_kernel_prepare_ROOT"]["path"]).read_text())
        old_producer = json.loads(Path(refs["historical_kernel_prepare_source"]["path"]).read_text())
        gate("historical_owning_packet_admitted", old_report["passed"] is True and old_report["new_kernel_P0_reference_passed"] is True
            and old_report["budget"] == budget and old_report["source"] == old_producer
            and old_report["native_kernel_mean_packet"] == B["old_origin_packet_reference_only"]
            and old_ROOT["passed"] is True and old_ROOT["source"] == old_producer
            and old_ROOT["budget_acceptances"][budget]["report"] == B["historical_prepare_report"]
            and old_ROOT["budget_acceptances"][budget]["native_kernel_mean_packet"] == B["old_origin_packet_reference_only"])
        geometry = json.loads(Path(B["source_geometry_acceptance"]["path"]).read_text())
        geometry_report = json.loads(Path(B["historical_geometry_report"]["path"]).read_text())
        gate("original_geometry_metadata_admitted", geometry["both_original_RMS_inverse_gates_passed"] is True
            and geometry["original_native_P0_and_serving_bitexact"] is True
            and geometry["reports"][budget] == B["historical_geometry_report"]
            and geometry["protocol"] == B["historical_geometry_protocol"] and geometry_report["passed"] is True
            and geometry_report["affine_inverse_vs_direct_H_centroid_passed"] is True
            and geometry_report["dual_head_context"]["source_refs"]["data_digest"] == B["original_data_digest"])
        import numpy as np
        import torch
        from src import joint_mean_ce as helper
        from src.citation_graph_factor import _transform
        from src.dual_head_ce import _factor_digests
        from src.io import array_digest
        from src.kernel_mean_ce import _seal
        from src.low_rank_assignment import LowRankMoments, initialize_factors
        from src.moments import decode_moments, make_material
        from src.nystrom_ce import NystromMap, _cache_identity, _content_digest
        from src.shared_features import _map_identity, _tensor_identity, _validate_map_state
        from src.transforms import FeatureTransform, fit_transform
        gate("promoted_409_helper_exact", packet["helper_entrypoint"]["sha256"] == HELPER_SHA
            and Path(helper.__file__).resolve() == Path(packet["helper_entrypoint"]["path"]).resolve() and _sha(helper.__file__, check) == HELPER_SHA)
        torch.set_num_threads(4)
        if torch.get_num_interop_threads() != 1: torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        device = torch.device("cuda:0")
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(device)
        report["runtime"] = dict(helper._runtime(device), interop_threads=torch.get_num_interop_threads())
        expected_runtime = dict(B["historical_runtime"], interop_threads=1)
        gate("frozen_native_runtime", report["runtime"] == expected_runtime and platform.python_version() == "3.12.7"
            and np.__version__ == "1.26.4" and torch.get_num_threads() == 4
            and all(os.environ.get(k) == "4" for k in contract["runtime"]["environment_threads"]))
        def own(v):
            if torch.is_tensor(v): return v.detach().cpu().clone()
            if isinstance(v, dict): return {k: own(x) for k, x in v.items()}
            if isinstance(v, (list, tuple)): return [own(x) for x in v]
            return v
        def load(role, ref, location, keys=None, weights_only=True):
            check(); work["PT_load_attempts"] += 1
            value = torch.load(ref["path"], map_location=location, weights_only=weights_only)
            work["PT_loads"] += 1
            captured[role] = own(value if keys is None else {k: value[k] for k in keys if k in value})
            return value
        def matrix(value, shape, dtype=torch.float64, dev=device):
            return torch.is_tensor(value) and list(value.shape) == shape and value.dtype == dtype and value.device == dev
        D = B["dimensions"]; n,d,b,c,k,r = (D[x] for x in ("nodes","physical_dimension","original_Phi_basis","classes","cells","rank"))
        inputs = load("z_transform", B["owning_z_transform"], device, ("z","transform"), False)
        z, transform = inputs["z"], FeatureTransform(**inputs["transform"])
        hard = load("hard", B["teacher_balanced_hard"], device)
        teacher = load("teacher", B["Q_admission_required"]["cached_teacher"], device, ("logits","gamma"), False)
        logits = teacher["logits"]
        report["logits_before_Q_decode"] = dict(_tensor_identity(logits), device=str(logits.device), requires_grad=logits.requires_grad)
        gate("original_logits_domain", logits.is_floating_point() and list(logits.shape) == [n,c]
            and logits.device == device and not logits.requires_grad and bool(torch.isfinite(logits).all()))
        work["Q_decode_attempts"] += 1
        q = (logits / B["Q_admission_required"]["T"]).softmax(1).double()
        work["Q_decode_completions"] += 1
        captured["Q"] = own(q)
        report["descriptors"]["Q"] = _tensor_identity(q)
        gate("own_original_Q_exact", report["descriptors"]["Q"] == B["expected_source_descriptors"]["Q"])
        native = load("native_factors", B["owning_native_factors"], device)
        parameters = [native["u"], native["v"]]
        h_state = load("H_cache", B["frozen_source_assets"]["H"], "cpu", ("h","identity","source_digest","schema","kind"))
        h = h_state["h"]
        map_state = load("map_cache", B["frozen_source_assets"]["map"], "cpu")
        artifact = load("historical_packet", B["old_origin_packet_reference_only"], "cpu",
            ("schema","kind","source","budget","context","initial_parameters","original_moments","original_serving","content_seal"), False)
        gate("historical_packet_source_context_seal", artifact["schema"] == 1
            and artifact["kind"] == "fixed_original_Phi_kernel_mean_uniform_CE_native_NODE_v1" and artifact["budget"] == budget
            and artifact["source"] == old_producer and artifact["context"] == old_report["kernel_mean_context"]
            and artifact["content_seal"] == _seal({x:y for x,y in artifact.items() if x != "content_seal"}))
        assets = B["historical_asset_descriptors"]
        gate("original_z_H_hard_exact", matrix(z,[n,d]) and matrix(h,[n,d],torch.float32,torch.device("cpu"))
            and matrix(hard,[n],torch.int64) and int(hard.min()) == 0 and int(hard.max()) == k-1
            and len(torch.unique(hard)) == k and all(_tensor_identity(v) == assets[x] for x,v in (("z",z),("H",h),("assignment",hard)))
            and h_state["schema"] == 1 and h_state["kind"] == "shared_h" and h_state["identity"] == assets["H"])
        gate("original_RMS_transform_exact", _transform(transform,d) == assets["transform"])
        gate("native_U0V0_and_data_exact", _factor_digests(parameters,n,k,r,device) == B["owning_native_factors"]["native_parameter_digests"]
            and bool(parameters[0].eq(0).all()) and native["factor_seed"] == 0 and native["mixing"] == .05
            and native["data_digest"] == B["original_data_digest"]
            and _factor_digests(artifact["initial_parameters"],n,k,r,torch.device("cpu")) == B["owning_native_factors"]["native_parameter_digests"]
            and array_digest(captured["z_transform"]["z"].numpy(),captured["Q"].numpy(),captured["hard"].numpy()) == B["original_data_digest"])
        gate("historical_serving_preserved", {x:_tensor_identity(v) for x,v in artifact["original_serving"].items()} == B["historical_original_student"])
        anchors, mapping = _validate_map_state(map_state, _map_identity(h,3000,0,"relu"))
        gate("original_map_descriptors_exact", map_state["schema"] == 1 and map_state["kind"] == "shared_nystrom_map"
            and _tensor_identity(anchors) == assets["anchors"] and _tensor_identity(mapping) == assets["mapping"])
        phi = np.load(B["frozen_source_assets"]["Phi"]["path"], mmap_mode="r", allow_pickle=False)
        captured["original_Phi"] = torch.from_numpy(np.array(phi,copy=True,order="C"))
        identity = _cache_identity(h,NystromMap(anchors,mapping,"relu"),(n,b),4096,check)
        identity["phi_digest"] = _content_digest(phi,4096,check)
        gate("mandatory_original_Phi_sidecar_exact", phi.shape == (n,b) and phi.dtype == np.float64
            and not phi.flags.writeable and identity == B["original_Phi_identity_unchanged"]
            and json.loads(Path(B["frozen_source_assets"]["Phi_metadata"]["path"]).read_text()) == identity)
        physical = artifact["original_moments"].to(device=device)
        captured["physical_M0"] = own(physical)
        gate("physical_M0_exact_original_shape", matrix(physical,[k,1+d+c])
            and _tensor_identity(physical) == B["old_physical_M0_reference_only"]["moments"])
        report["historical_CPU_quotients_supplier_only"] = B["old_physical_M0_reference_only"]
        functions = {LowRankMoments.forward.__code__:"LowRank_forward", LowRankMoments.backward.__code__:"LowRank_backward",
            initialize_factors.__code__:"initialize_factors", make_material.__code__:"make_material",
            decode_moments.__code__:"decode_moments", NystromMap.fit.__func__.__code__:"map_fit",
            NystromMap.__call__.__code__:"map_forward", fit_transform.__wrapped__.__code__:"RMS_refit"}
        def profile(frame,event,arg):
            if event == "call" and frame.f_code in functions: profile_work[functions[frame.f_code]] += 1
        sys.setprofile(profile)
        with torch.no_grad():
            layout = helper.JointMeanLayout(d,b,c)
            provider = helper.ResidentJointFeatures(z,phi,layout,dict(z=assets["z"],Phi_identity=identity))
            work["resident_joint_provider_constructions"] += 1
            report["layout"], report["joint_provider"] = layout.descriptor(), provider.descriptor()
            material = provider.material_on(q,device)
            work["joint_material_constructions"] += 1
            report["material_shape"] = list(material.shape)
            gate("literal_joint_material_shape", list(material.shape) == [n,1+d+b+c])
            work["joint_moment_forward_attempts"] += 1
            joint = LowRankMoments.apply(*parameters,hard,material,.05,4096)
            work["joint_moment_forwards"] += 1
            captured["joint_M0"] = own(joint)
            for label,value,width in (("native_physical",physical,d),("native_joint",joint,d+b)):
                centers, labels, mass = decode_moments(value,width)
                work["native_device_moment_decodes"] += 1
                captured[label] = own(dict(centers=centers,labels=labels,mass=mass))
                report["descriptors"][label] = {x:dict(_tensor_identity(y),device=str(y.device),requires_grad=y.requires_grad)
                    for x,y in dict(moments=value,centers=centers,labels=labels,mass=mass).items()}
                gate(label+"_frozen_finite_positive_domain", matrix(value,[k,1+width+c])
                    and all(not v.requires_grad and v.grad_fn is None and v.device == device and bool(torch.isfinite(v).all())
                        for v in (value,centers,labels,mass)) and bool((mass > 0).all())
                    and bool((labels >= 0).all()) and bool((labels.sum(1) > 0).all()) and bool((labels.sum(0) > 0).all()))
        sys.setprofile(old_profile)
        report["observed_profile_calls"] = profile_work
        gate("exact_public_and_original_call_counts", profile_work == dict(LowRank_forward=1,LowRank_backward=0,
            initialize_factors=0,make_material=1,decode_moments=2,map_fit=0,map_forward=0,RMS_refit=0))
        gate("exact_capture_counts", work == contract["exact_counts"])
        torch.cuda.synchronize(device); check()
        report["capture_completed"] = True
    except BaseException as exc:
        report["failure"] = dict(type=type(exc).__name__,message=str(exc))
    finally:
        sys.setprofile(old_profile)
        signal.setitimer(signal.ITIMER_REAL,0)
        report["observed_profile_calls"] = profile_work
        if torch is not None:
            raw = report["raw_evidence"]
            try:
                with arrays_path.open("xb") as f:
                    torch.save(dict(schema=1,kind=KIND,budget=budget,source=packet["source"],
                        scientific_contract=science,captured=captured),f); f.flush(); os.fsync(f.fileno())
                raw["write_completed"] = True
            except BaseException as exc: report["raw_write_error"] = repr(exc)
            try:
                raw["exists"] = arrays_path.exists()
                if raw["exists"]:
                    raw["bytes"] = arrays_path.stat().st_size
                    try:
                        raw["sha256"] = _sha(arrays_path); raw["hash_unknown"] = False
                    except BaseException as exc: raw["hash_error"] = repr(exc)
            except BaseException as exc: report["raw_metadata_error"] = repr(exc)
        try:
            report["readonly_exit"] = pins()
            report["source_exit"] = implementation_provenance()
            _require(report["readonly_exit"] == packet["readonly_files_sha256"] and report["source_exit"] == packet["source"], "Exit pin/source differs")
        except BaseException as exc:
            report["exit_verification_error"] = repr(exc)
        report["resources"] = peaks()
        report["passed"] = bool(report.get("capture_completed") and report["failure"] is None
            and "exit_verification_error" not in report and "raw_write_error" not in report
            and "raw_metadata_error" not in report and not report["raw_evidence"]["hash_unknown"]
            and all(report["resources"][k] <= contract["resource_limits"][k] for k in report["resources"]))
        try:
            with report_path.open("x") as f:
                json.dump(report,f,indent=2,allow_nan=False); f.write("\n"); f.flush(); os.fsync(f.fileno())
            final = peaks()
            if report["passed"] and any(final[k] > contract["resource_limits"][k] for k in final):
                report.update(passed=False,resources=final,failure=dict(type="FinalSerializationResourceBoundary",
                    message="Final report serialization exceeded frozen capture resources; no retry"))
                with report_path.open("w") as f:
                    json.dump(report,f,indent=2,allow_nan=False); f.write("\n"); f.flush(); os.fsync(f.fileno())
        finally:
            signal.signal(signal.SIGALRM,old_handler); signal.setitimer(signal.ITIMER_REAL,*old_timer)
    _require(report["passed"], "Terminal source capture failure; preserve namespace and do not retry")
    return str(report_path)
