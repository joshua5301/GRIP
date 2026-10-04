"""Own K223 native origin on byte-preserved source122 coordinates and targets."""
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

from src.io import array_digest, cpu_state
from src.large_kernel_mean_source_capture import _matrix, _observed, _require, _sha
from src.low_rank_assignment import LowRankMoments, initialize_factors
from src.moments import decode_moments, make_material
from src.nystrom_ce import NystromMap, _cache_identity, _content_digest
from src.shared_features import _tensor_identity

KIND = "large_kernel_mean_source_reuse_capture_v1"
CONTRACT_SHA = "22edc25559b094bb82bf7de12990c2854e63c9bc71e603f082d04d44cd51f7e0"
DIMS = dict(nodes=44625, features=500, classes=7, basis=512, cells=223, rank=16)


def run(protocol_path, protocol_sha256, budget, stop=lambda: False):
    from src.research_loop import implementation_provenance

    _require(callable(stop) and _sha(protocol_path) == protocol_sha256, "Frozen invocation differs")
    packet = json.loads(Path(protocol_path).read_text()); ref = packet["scientific_contract"]
    _require(ref["sha256"] == CONTRACT_SHA and _sha(ref["path"]) == CONTRACT_SHA, "Capture contract differs")
    contract = json.loads(Path(ref["path"]).read_text())
    _require(type(packet["schema"]) is int and packet["schema"] == 1
        and packet["kind"] == contract["kind"] == KIND
        and budget == packet["budget"] == contract["budget"] == "flickr223"
        and contract["dimensions"] == DIMS and packet["test_enabled"] is False
        and packet["required_refs"] == contract["required_refs"]
        and packet["runtime"] == contract["runtime"]
        and packet["resource_limits"] == contract["resource_limits"]
        and packet["source"] == implementation_provenance(), "Frozen source/domain differs")
    required = dict(contract["source_pins_sha256"])
    def collect(value):
        if isinstance(value, dict):
            if "path" in value and "sha256" in value:
                required[value["path"]] = value["sha256"]
            for child in value.values(): collect(child)
        elif isinstance(value, list):
            for child in value: collect(child)
    collect(contract); collect(ref)
    _require(all(Path(p).is_absolute() and packet["readonly_files_sha256"].get(p) == s
        for p, s in required.items()), "Required absolute readonly admission omitted")
    _require(len(packet["source"]["files"]) == 125
        and contract["resident_source"]["bytes"] == 44625*(500+512+7)*8 == 363783000
        and contract["resident_source"]["cap"] == 512*1024**2
        and 363783000 <= contract["resident_source"]["cap"], "Resident/source domain differs")
    folder = Path(packet["output_folder"])
    _require(folder.is_absolute() and not folder.exists(), "Fresh exclusive absolute namespace required")
    folder.mkdir(parents=True); started = time.monotonic(); arrays = {}; failure = None; gpu = False
    limits = contract["resource_limits"]; counts = dict.fromkeys(contract["new_first_capture_counts"], 0)
    counts.update({k+"_attempts": 0 for k,v in contract["new_first_capture_counts"].items() if v})
    counts.update(own_source122_array_load_attempts=0, own_source122_array_loads=0)
    report = dict(schema=1, kind=KIND, budget=budget, passed=False, source=packet["source"],
        protocol=dict(path=str(Path(protocol_path).resolve()), sha256=protocol_sha256),
        scientific_contract=ref, required_refs=packet["required_refs"], dimensions=DIMS, counts=counts,
        reused_coordinate_Q_producer=contract["source122_reused_producer"],
        rawgraph_H_admission_producer=contract["source124_rawgraph_admission_producer"],
        historical_dataset_digest=contract["historical_dataset_digest"],
        source122_K44_data_digest=contract["source122_K44_data_digest"],
        historical_RMS_equality_claim=False, fresh_raw_data_source_reconstruction=False,
        old_K44_native_moments_readouts_adopted=False, conditional_head_control_status="NOT_QUALIFIED")
    def guard():
        _require(not stop() and time.monotonic()-started <= limits["max_seconds"]
            and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 <= limits["peak_RSS_bytes"],
            "Terminal stop/time/RSS boundary")
        if gpu:
            _require(torch.cuda.max_memory_allocated() <= limits["peak_allocated_bytes"]
                and torch.cuda.max_memory_reserved() <= limits["peak_reserved_bytes"], "Terminal GPU resource boundary")
    def pins(label):
        report[label] = {}
        for p,s in {**packet["readonly_files_sha256"], **packet["source"]["files"]}.items():
            report[label][p] = _sha(p, guard)
            _require(report[label][p] == s, "Readonly/source bytes changed: "+p)
        _require(implementation_provenance() == packet["source"], "Whole producer source changed")
    def read(reference):
        _require(_sha(reference["path"], guard) == reference["sha256"], "Pinned metadata changed")
        return json.loads(Path(reference["path"]).read_text())
    def call(name, fn, *args, **kwargs):
        guard(); counts[name+"_attempts"] += 1
        returned = fn(*args, **kwargs); counts[name] += 1
        return returned
    def expire(signum, frame): raise InterruptedError("Frozen capture300s deadline")
    old_handler = signal.getsignal(signal.SIGALRM); old_timer = signal.getitimer(signal.ITIMER_REAL)
    try:
        signal.signal(signal.SIGALRM, expire)
        signal.setitimer(signal.ITIMER_REAL, min(300, old_timer[0]) if old_timer[0] else 300)
        rt = contract["runtime"]
        _require(platform.python_version() == rt["python"] and str(torch.__version__) == rt["torch"]
            and np.__version__ == rt["numpy"] and str(torch.get_default_dtype()) == rt["default_dtype"]
            and not torch.is_autocast_enabled("cuda") and all(os.environ.get(k) == rt[k] for k in
                ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "CUBLAS_WORKSPACE_CONFIG", "CUDA_VISIBLE_DEVICES")),
            "Frozen runtime/environment differs")
        torch.set_num_threads(4)
        if torch.get_num_interop_threads() != 1: torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest"); torch.use_deterministic_algorithms(True)
        torch.cuda.init(); _require(torch.cuda.get_device_name(0) == rt["GPU"], "Frozen GPU differs")
        torch.cuda.reset_peak_memory_stats(); gpu = True; device = torch.device("cuda:0")
        report["runtime"] = dict(rt, actual_threads=torch.get_num_threads(),
            actual_interop_threads=torch.get_num_interop_threads(),
            actual_deterministic=torch.are_deterministic_algorithms_enabled())
        pins("entry_files_sha256"); refs = packet["required_refs"]
        accepted = read(refs["own_source122_acceptance"]); parent = read(refs["own_source122_report"])
        report["source122_parent_lineage"] = dict(source=parent.get("source"), passed=parent.get("passed"),
            new_data_digest=parent.get("new_data_digest"), original_Phi_identity=parent.get("original_Phi_identity"))
        graph_accepted = read(refs["own_source124_rawgraph_H_admission"]); graph_report = read(refs["own_source124_prepare_report"])
        report["source124_parent_lineage"] = dict(source=graph_report.get("source"), passed=graph_report.get("passed"),
            source_tokens=graph_report.get("source_tokens"), shared_cache=graph_report.get("shared_cache"))
        _require(accepted["passed"] is parent["passed"] is graph_accepted["passed"] is graph_report["passed"] is True
            and accepted["source"] == parent["source"] == contract["source122_reused_producer"]
            and accepted["actual_report"] == refs["own_source122_report"]
            and accepted["actual_arrays"] == refs["own_source122_arrays"]
            and parent["raw_evidence"]["sha256"] == refs["own_source122_arrays"]["sha256"]
            and graph_accepted["source"] == graph_report["source"] == contract["source124_rawgraph_admission_producer"]
            and graph_accepted["shared_prepare_report"] == refs["own_source124_prepare_report"]
            and graph_accepted["shared_cache"] == graph_report["shared_cache"] == refs["own_source124_shared_cache"],
            "Owning source122 or rawgraph124 admission differs")
        _require(read(refs["hard223_policy_protocol"]) == contract["original_hard_policy"], "Original hard223 policy differs")
        shared = contract["shared_reuse"]
        _require(graph_accepted["cache_identities"] == graph_report["cache_identities"]
            and graph_report["cache_identities"]["train_H"] == shared["expected_array_descriptors"]["saved_H_FP32"]
            and graph_report["cache_identities"]["Q"] == shared["expected_array_descriptors"]["Q"],
            "Accepted rawgraph/H/Q lineage differs from shared source122")
        counts["own_source122_array_load_attempts"] += 1
        supplied = torch.load(refs["own_source122_arrays"]["path"], map_location="cpu", weights_only=False)
        counts["own_source122_array_loads"] += 1
        for key in shared["strict_owning_keys"]: arrays[key] = cpu_state(supplied[key])
        del supplied
        report["reused_array_descriptors"] = {k:_tensor_identity(arrays[k]) for k in shared["expected_array_descriptors"]}
        report["transform_descriptors"] = {k:_tensor_identity(v) if torch.is_tensor(v) else v for k,v in arrays["transform"].items()}
        _require(report["reused_array_descriptors"] == shared["expected_array_descriptors"]
            == {k:parent["array_descriptors"][k] for k in shared["expected_array_descriptors"]}
            and report["transform_descriptors"] == shared["expected_transform_descriptors"] == parent["transform_descriptors"],
            "Reused owning array/transform bytes differ")
        _matrix(arrays["z"], (44625,500), torch.float64); _matrix(arrays["Q"], (44625,7), torch.float64)
        _matrix(arrays["saved_H_FP32"], (44625,500), torch.float32)
        _require(bool((arrays["Q"]>=0).all()) and bool((arrays["Q"].sum(1)>0).all()), "Reused rawQ domain differs")
        t = arrays["transform"]
        _require(t["kind"] == "rms" and t["matrix"] is None and t["eps"] == 1e-12
            and bool(torch.isfinite(t["scale"])) and bool(t["scale"]>0), "Reused RMS transform differs")
        _matrix(t["center"], (500,), torch.float64); _matrix(t["output_center"], (500,), torch.float64)
        _matrix(t["scale"], (), torch.float64)
        map_state = arrays["map_state"]
        _require(isinstance(map_state, dict) and set(map_state) == {"anchors", "mapping", "kernel"}
            and map_state["kernel"] == "relu", "Reused literal map differs")
        _matrix(map_state["anchors"], (512,500), torch.float64); _matrix(map_state["mapping"], (512,512), torch.float64)
        feature_map = NystromMap(**map_state)
        z = arrays["z"].to(device); q = arrays["Q"].to(device); h = arrays["saved_H_FP32"].to(device).double()
        sidecar = read(refs["Phi_sidecar"])
        phi = np.load(refs["Phi"]["path"], mmap_mode="r", allow_pickle=False)
        _require(phi.shape == (44625,512) and phi.dtype == np.float64 and phi.dtype.isnative
            and phi.flags.c_contiguous and not phi.flags.writeable, "Original Phi header differs")
        for start in range(0,len(phi),2048):
            _require(bool(np.isfinite(phi[start:start+2048]).all()), "Original Phi nonfinite"); guard()
        identity = _cache_identity(h, feature_map, phi.shape, 2048, lambda: (guard() or False))
        _require(sidecar == dict(**identity, phi_digest=_content_digest(phi,2048,lambda: (guard() or False)))
            == shared["original_Phi_identity"] == parent["original_Phi_identity"], "Mandatory original Phi/map/H identity differs")
        report["original_Phi_identity"] = sidecar; guard()
        hard = call("own_K223_hard_loads", torch.load, refs["hard223"]["path"], map_location="cpu", weights_only=False)
        arrays["hard"] = cpu_state(hard)
        _require(torch.is_tensor(hard) and hard.dtype == torch.int64 and tuple(hard.shape) == (44625,)
            and int(hard.min()) == 0 and int(hard.max()) == 222 and len(torch.unique(hard)) == 223, "Hard223 domain differs")
        hard = hard.detach().to(device); guard()
        u,v = call("native_factor_factory_calls", initialize_factors, hard, 223, 16, seed=0)
        arrays.update(U0=cpu_state(u), V0=cpu_state(v))
        _matrix(u, (44625,16), torch.float32); _matrix(v, (223,16), torch.float32)
        _require(bool(u.eq(0).all()), "New native zeroU origin differs"); guard()
        phi_gpu = torch.tensor(phi, dtype=torch.float64, device=device)
        for name,key,features in (("RMS_z_Q_LowRankMoments_forwards","RMS_M0",z),
            ("raw_H_Q_LowRankMoments_forwards","raw_H_M0",h),
            ("cached_Phi_Q_LowRankMoments_forwards","Phi_M0",phi_gpu)):
            material = make_material(features,q)
            with torch.no_grad(): moments = call(name,LowRankMoments.apply,u,v,hard,material,.05,2048)
            arrays[key] = cpu_state(moments)
            _matrix(moments, (223,material.shape[1]), torch.float64)
            _require(bool((moments[:,0]>0).all()), "New M0 mass invalid")
            del material; guard()
        raw = arrays["raw_H_M0"].to(device); rms = arrays["RMS_M0"].to(device)
        centers,labels,mass = decode_moments(raw,500)
        arrays.update(canonical_X=cpu_state(centers.float()),canonical_Qbar=cpu_state(labels.float()),
            uniform_weights=cpu_state(mass.new_full((223,),1/223)))
        _matrix(arrays["canonical_X"], (223,500), torch.float32)
        _matrix(arrays["canonical_Qbar"], (223,7), torch.float32); _matrix(arrays["uniform_weights"], (223,), torch.float64)
        _require(bool((arrays["canonical_Qbar"]>=0).all()) and bool((arrays["canonical_Qbar"].sum(1)>0).all())
            and bool((arrays["uniform_weights"]>0).all()), "New canonical readout domain differs")
        def compare():
            inverse = decode_moments(rms,500)[0]*t["scale"].to(device)+t["output_center"].to(device)+t["center"].to(device)
            return inverse,inverse-centers
        inverse,residual = call("FP64_affine_geometry_comparisons", compare)
        arrays.update(new_RMS_inverse_centroids=cpu_state(inverse),new_affine_residual=cpu_state(residual))
        error = float(residual.abs().max())
        report["new_FP64_geometry"] = dict(Linf=_observed(error),bound=1e-12,passed=math.isfinite(error) and error<=1e-12)
        _require(bool(torch.isfinite(residual).all()) and error<=1e-12, "New K223 affine geometry failed"); guard()
        report["new_data_digest"] = array_digest(arrays["z"].numpy(),arrays["Q"].numpy(),arrays["hard"].numpy())
        report["array_descriptors"] = {k:_tensor_identity(v) for k,v in arrays.items() if torch.is_tensor(v)}
        _require(set(arrays) == set(contract["outputs"]["array_keys"])
            and all(counts[k] == v for k,v in contract["new_first_capture_counts"].items())
            and counts["own_source122_array_loads"] == counts["own_source122_array_load_attempts"] == 1,
            "Capture outputs/actual work differ")
    except BaseException as exc:
        failure = exc; report["error"] = dict(type=type(exc).__name__,message=str(exc))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0); signal.signal(signal.SIGALRM,old_handler)
        if old_timer[0]>0: signal.setitimer(signal.ITIMER_REAL,max(1e-6,old_timer[0]-(time.monotonic()-started)),old_timer[1])
        path = folder/"capture_arrays.pt"
        report["returned_array_headers"] = {k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in arrays.items() if torch.is_tensor(v)}
        report["raw_evidence"] = dict(path=str(path),exists=False,complete=False,sha256=None)
        try:
            with path.open("xb") as stream:
                torch.save(arrays,stream);stream.flush();os.fsync(stream.fileno())
            report["raw_evidence"].update(exists=True,complete=True,bytes=path.stat().st_size)
            report["raw_evidence"]["sha256"] = _sha(path,guard)
        except BaseException as exc:
            failure = failure or exc
            report["raw_evidence"].update(exists=path.exists(),bytes=path.stat().st_size if path.exists() else 0,error=str(exc))
        try: pins("exit_files_sha256"); guard()
        except BaseException as exc:
            failure = failure or exc; report["preservation_exit_error"] = dict(type=type(exc).__name__,message=str(exc))
        report.update(seconds=time.monotonic()-started,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            peak_allocated_bytes=torch.cuda.max_memory_allocated() if gpu else 0,
            peak_reserved_bytes=torch.cuda.max_memory_reserved() if gpu else 0)
        if report["seconds"]>300 or report["peak_RSS_bytes"]>limits["peak_RSS_bytes"] or report["peak_allocated_bytes"]>limits["peak_allocated_bytes"] or report["peak_reserved_bytes"]>limits["peak_reserved_bytes"]:
            failure = failure or RuntimeError("Final preservation resource bound failed")
        report["passed"] = failure is None
        if failure is not None and "error" not in report:
            report["error"] = dict(type=type(failure).__name__,message=str(failure))
        with (folder/"capture_report.json").open("x") as stream:
            json.dump(_observed(report),stream,indent=2,allow_nan=False);stream.write("\n")
    if failure is not None: raise RuntimeError("Terminal new K223 capture; no fallback/retry") from failure
    return report
