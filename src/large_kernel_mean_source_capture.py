"""NEW owning Flickr44 source/native capture; no head or assignment update.

This entry point requires existing caches. Its one RMS fit is a new owning
coordinate origin, never a claim of equality to a historical unsaved transform.
"""
import hashlib
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
from src.low_rank_assignment import LowRankMoments, initialize_factors
from src.moments import decode_moments, make_material
from src.nystrom_ce import NystromMap, _cache_identity, _content_digest
from src.shared_features import _load_state, _tensor_identity
from src.transforms import fit_transform

KIND = "large_kernel_mean_new_source_capture_v1"
CONTRACT_SHA = "fdf55ed0f3719667e869a91528bfa68a27367a158bcb6bb32e2a2a33e71787f9"
DIMS = dict(nodes=44625, features=500, classes=7, cells=44, rank=16, basis=512)
DATA = "8515edc243be54315563c8f122f1416bd9e2979c554d6a23d6a0a9e0b541acde"


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _sha(path, guard=lambda: None):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            value.update(block)
            guard()
    return value.hexdigest()


def _observed(value):
    if isinstance(value, float) and not math.isfinite(value):
        return dict(nonfinite=repr(value))
    if isinstance(value, dict):
        return {str(k): _observed(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_observed(v) for v in value]
    return value


def _matrix(value, shape, dtype=None):
    _require(torch.is_tensor(value) and tuple(value.shape) == shape
        and value.is_floating_point() and (dtype is None or value.dtype == dtype)
        and bool(torch.isfinite(value).all()), "Cached matrix shape/dtype/finite domain differs")


def run(protocol_path, protocol_sha256, budget, stop=lambda: False):
    from src.research_loop import implementation_provenance

    _require(callable(stop) and _sha(protocol_path) == protocol_sha256, "Frozen invocation differs")
    packet = json.loads(Path(protocol_path).read_text())
    ref = packet["scientific_contract"]
    _require(ref["sha256"] == CONTRACT_SHA and _sha(ref["path"]) == CONTRACT_SHA,
        "Prospective first-capture contract differs")
    contract = json.loads(Path(ref["path"]).read_text())
    _require(type(packet["schema"]) is int and packet["schema"] == 1 and packet["kind"] == KIND
        and budget == packet["budget"] == contract["budget"] == "flickr44"
        and contract["dimensions"] == DIMS and packet["test_enabled"] is False
        and packet["required_cache_refs"] == contract["required_cache_refs"]
        and packet["runtime"] == contract["runtime"]
        and packet["resource_limits"] == contract["resource_limits"]
        and packet["source"] == implementation_provenance(), "Frozen source/capture domain differs")
    expected_pins = {}
    def collect(value):
        if isinstance(value, dict):
            if "path" in value and "sha256" in value:
                path, digest = value["path"], value["sha256"]
                _require(Path(path).is_absolute() and packet["readonly_files_sha256"].get(path) == digest,
                    "Required contract/cache pin omitted or changed")
                expected_pins[path] = digest
            for child in value.values():
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)
    collect(contract); collect(ref)
    _require(isinstance(packet["readonly_files_sha256"], dict) and expected_pins
        and isinstance(packet["source"].get("files"), dict), "Readonly/source manifest missing")
    folder = Path(packet["output_folder"])
    _require(folder.is_absolute() and not folder.exists(), "Exclusive fresh absolute output folder required")
    folder.mkdir(parents=True)
    started = time.monotonic(); arrays = {}; failure = None; gpu = False
    counts = {k: 0 for k in contract["new_first_capture_counts"]}
    counts.update({k + "_attempts": 0 for k, v in contract["new_first_capture_counts"].items() if v})
    report = dict(schema=1, kind=KIND, budget=budget, passed=False, source=packet["source"],
        protocol=dict(path=str(Path(protocol_path).resolve()), sha256=protocol_sha256),
        scientific_contract=ref, dimensions=DIMS, counts=counts,
        historical_dataset_digest=DATA, historical_RMS_equality_claim=False,
        fresh_raw_data_source_certification=False, conditional_head_control_status="NOT_QUALIFIED")
    limits = contract["resource_limits"]
    def guard():
        _require(not stop() and time.monotonic() - started <= limits["max_seconds"]
            and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 <= limits["peak_RSS_bytes"],
            "Terminal stop/time/RSS boundary")
        if gpu:
            _require(torch.cuda.max_memory_allocated() <= limits["peak_allocated_bytes"]
                and torch.cuda.max_memory_reserved() <= limits["peak_reserved_bytes"], "Terminal GPU peak boundary")
    def pins(label):
        actual = {}
        report[label] = actual
        for path, digest in {**packet["readonly_files_sha256"], **packet["source"]["files"]}.items():
            actual[path] = _sha(path, guard)
            _require(actual[path] == digest, "Immutable source/cache bytes differ: " + path)
        _require(implementation_provenance() == packet["source"], "Whole producer source changed")
    def call(name, function, *args, **kwargs):
        guard(); counts[name + "_attempts"] += 1
        result = function(*args, **kwargs); counts[name] += 1
        return result
    def expired(signum, frame):
        raise InterruptedError("Frozen capture wall deadline reached")
    previous_handler = signal.getsignal(signal.SIGALRM); previous_timer = signal.getitimer(signal.ITIMER_REAL)
    try:
        signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, min(300, previous_timer[0]) if previous_timer[0] else 300)
        runtime = contract["runtime"]
        _require(platform.python_version() == runtime["python"] and str(torch.__version__) == runtime["torch"]
            and str(torch.get_default_dtype()) == runtime["default_dtype"]
            and not torch.is_autocast_enabled("cuda")
            and all(os.environ.get(k) == runtime[k] for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS", "CUBLAS_WORKSPACE_CONFIG", "CUDA_VISIBLE_DEVICES")), "Frozen runtime/environment differs")
        torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False; torch.set_float32_matmul_precision("highest")
        torch.use_deterministic_algorithms(True); torch.cuda.init()
        _require(torch.cuda.get_device_name(0) == runtime["GPU"], "Frozen native GPU differs")
        torch.cuda.reset_peak_memory_stats(0); gpu = True; device = torch.device("cuda:0")
        report["runtime"] = dict(runtime, actual_threads=torch.get_num_threads(),
            actual_deterministic=torch.are_deterministic_algorithms_enabled(),
            actual_matmul_precision=torch.get_float32_matmul_precision())
        pins("entry_files_sha256"); cache = packet["required_cache_refs"]
        teacher_protocol = json.loads(Path(cache["teacher_protocol"]["path"]).read_text())
        _require(all(teacher_protocol.get(k) == v for k, v in dict(dataset="flickr", basis=512,
            teacher_seed=0, kernel="relu", gamma=1.0, chunk=2048, data_digest=DATA).items()), "Gamma1 teacher source differs")
        for name, loss in (("baseline_protocol", "exact_mass_ce"), ("uniform_Ny_protocol", "exact_uniform_ce")):
            old = json.loads(Path(cache[name]["path"]).read_text()); candidate = old["candidate"]
            _require(old["data_digest"] == DATA and all(candidate.get(k) == v for k, v in
                dict(cells=44, temperature=.3, rank=16, penalty=1e-4, lr=.01, condensation_seed=0,
                    initialization="teacher_balanced", alpha=1.0, assignment="low_rank", inner_loss=loss,
                    student_loss="uniform_ce", synthetic_adjacency="identity").items())
                and candidate.get("train_target_mix", 0.0) == 0.0, "Historical Q/native policy differs")
        saved = _load_state(cache["H"]["path"], "shared_h")
        arrays["saved_H_FP32"] = cpu_state(saved["h"])
        _matrix(saved["h"], (44625,500), torch.float32)
        _require(saved["identity"] == _tensor_identity(saved["h"])
            and type(saved.get("source_digest")) is str and bool(saved["source_digest"]), "Saved H identity/lineage differs")
        report["saved_H_lineage"] = dict(source_digest=saved["source_digest"], identity=saved["identity"],
            raw_source_reconstruction_performed=False)
        h = saved["h"].detach().to(device).double(); guard()
        map_state = torch.load(cache["map"]["path"], map_location="cpu", weights_only=False)
        arrays["map_state"] = cpu_state(map_state)
        _require(isinstance(map_state, dict) and set(map_state) == {"anchors", "mapping", "kernel"}
            and map_state["kernel"] == "relu", "Legacy literal map state differs")
        _matrix(map_state["anchors"], (512,500), torch.float64)
        _matrix(map_state["mapping"], (512,512), torch.float64)
        feature_map = NystromMap(**map_state)
        sidecar = json.loads(Path(cache["Phi_sidecar"]["path"]).read_text())
        phi = np.load(cache["Phi"]["path"], mmap_mode="r", allow_pickle=False)
        _require(phi.shape == (44625,512) and phi.dtype == np.float64 and phi.dtype.isnative
            and phi.flags.c_contiguous, "Original Phi header differs")
        for start in range(0, len(phi), 2048):
            _require(bool(np.isfinite(phi[start:start+2048]).all()), "Original Phi nonfinite"); guard()
        identity = _cache_identity(h, feature_map, phi.shape, 2048, lambda: (guard() or False))
        _require(sidecar == dict(**identity, phi_digest=_content_digest(phi, 2048, lambda: (guard() or False))),
            "Original mandatory Phi/H/map sidecar identity differs")
        report["original_Phi_identity"] = sidecar
        teacher = torch.load(cache["teacher"]["path"], map_location="cpu", weights_only=False)
        _require(isinstance(teacher, dict) and teacher.get("converged") is True
            and teacher.get("gamma") == 1.0 and teacher.get("data_digest") == DATA, "Teacher cached source differs")
        _matrix(teacher["logits"], (44625,7)); _matrix(teacher["weight"], (512,7))
        logits = teacher["logits"].detach().to(device)
        report["teacher_logits_dtype"] = str(logits.dtype)
        q = call("cached_teacher_target_decodes", lambda: (logits/.3).softmax(1).double())
        arrays["Q"] = cpu_state(q)
        _require(bool(torch.isfinite(q).all()) and bool((q>=0).all()) and bool((q.sum(1)>0).all()), "Raw Q domain differs"); guard()
        hard = torch.load(cache["hard"]["path"], map_location="cpu", weights_only=False)
        arrays["hard"] = cpu_state(hard)
        _require(cache["hard"]["sha256"] == cache["uniform_Ny_hard"]["sha256"]
            and torch.is_tensor(hard) and hard.dtype == torch.int64 and tuple(hard.shape) == (44625,)
            and int(hard.min()) == 0 and int(hard.max()) == 43 and len(torch.unique(hard)) == 44, "Hard source cells differ")
        hard = hard.detach().to(device)
        z, transform = call("RMS_fit_calls", fit_transform, h, kind="rms")
        arrays.update(z=cpu_state(z), transform=cpu_state(transform.state_dict()))
        _matrix(z, (44625,500), torch.float64)
        _require(transform.kind == "rms" and transform.matrix is None and transform.eps == 1e-12
            and bool(torch.isfinite(transform.scale)) and bool(transform.scale>0), "New RMS source invalid"); guard()
        u, v = call("native_factor_factory_calls", initialize_factors, hard, 44, 16, seed=0)
        arrays.update(U0=cpu_state(u), V0=cpu_state(v))
        _matrix(u, (44625,16), torch.float32); _matrix(v, (44,16), torch.float32)
        _require(bool(u.eq(0).all()), "Native zero-U origin differs"); guard()
        phi_gpu = torch.tensor(phi, dtype=torch.float64, device=device)
        for name, key, features in (("RMS_z_Q_LowRankMoments_forwards", "RMS_M0", z),
            ("raw_H_Q_LowRankMoments_forwards", "raw_H_M0", h),
            ("cached_Phi_Q_LowRankMoments_forwards", "Phi_M0", phi_gpu)):
            material = make_material(features, q)
            with torch.no_grad():
                moments = call(name, LowRankMoments.apply, u, v, hard, material, .05, 2048)
            arrays[key] = cpu_state(moments)
            _require(bool(torch.isfinite(moments).all()) and bool((moments[:,0]>0).all()), "Captured M0 invalid")
            del material; guard()
        rms = arrays["RMS_M0"].to(device); raw = arrays["raw_H_M0"].to(device)
        centers, labels, mass = decode_moments(raw, 500)
        arrays.update(canonical_X=cpu_state(centers.float()), canonical_Qbar=cpu_state(labels.float()),
            uniform_weights=cpu_state(mass.new_full((44,), 1/44)))
        _matrix(arrays["canonical_X"], (44,500), torch.float32)
        _matrix(arrays["canonical_Qbar"], (44,7), torch.float32)
        _matrix(arrays["uniform_weights"], (44,), torch.float64)
        _require(bool((arrays["canonical_Qbar"]>=0).all())
            and bool((arrays["canonical_Qbar"].sum(1)>0).all())
            and bool((arrays["uniform_weights"]>0).all()), "Canonical owning readout invalid")
        def compare_geometry():
            inverse = decode_moments(rms, 500)[0] * transform.scale + transform.output_center + transform.center
            return inverse, inverse-centers
        inverse, residual = call("FP64_affine_geometry_comparisons", compare_geometry)
        arrays.update(new_RMS_inverse_centroids=cpu_state(inverse), new_affine_residual=cpu_state(residual))
        error = float(residual.abs().max())
        report["new_FP64_geometry"] = dict(Linf=_observed(error), bound=1e-12,
            passed=math.isfinite(error) and error<=1e-12, historical_FP32_parity_gate_waived=False)
        _require(bool(torch.isfinite(residual).all()) and error<=1e-12, "New owning FP64 geometry failed"); guard()
        report["new_data_digest"] = array_digest(arrays["z"].numpy(), arrays["Q"].numpy(), arrays["hard"].numpy())
        report["array_descriptors"] = {k:_tensor_identity(t) for k,t in arrays.items() if torch.is_tensor(t)}
        report["transform_descriptors"] = {k:_tensor_identity(t) if torch.is_tensor(t) else t
            for k,t in arrays["transform"].items()}
        _require(all(counts[k] == v for k,v in contract["new_first_capture_counts"].items()), "Actual capture work differs")
        report["passed"] = True
    except BaseException as error:
        failure = error; report["error"] = dict(type=type(error).__name__, message=str(error))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0); signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, max(1e-6, previous_timer[0]-(time.monotonic()-started)), previous_timer[1])
        raw_path = folder/"capture_arrays.pt"
        report["returned_array_headers"] = {k:dict(shape=list(t.shape), dtype=str(t.dtype))
            for k,t in arrays.items() if torch.is_tensor(t)}
        report["raw_evidence"] = dict(path=str(raw_path), exists=False, complete=False, sha256=None)
        try:
            with raw_path.open("xb") as stream:
                torch.save(arrays, stream); stream.flush(); os.fsync(stream.fileno())
            report["raw_evidence"].update(exists=True, bytes=raw_path.stat().st_size, complete=True)
            report["raw_evidence"]["sha256"] = _sha(raw_path, guard)
        except BaseException as error:
            failure = failure or error; report["passed"] = False
            report["raw_evidence"].update(exists=raw_path.exists(), bytes=raw_path.stat().st_size if raw_path.exists() else 0,
                error=dict(type=type(error).__name__, message=str(error)))
        try:
            pins("exit_files_sha256"); guard()
        except BaseException as error:
            failure = failure or error; report["passed"] = False
            report["exit_error"] = dict(type=type(error).__name__, message=str(error))
        report.update(seconds=time.monotonic()-started, peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            peak_allocated_bytes=torch.cuda.max_memory_allocated() if gpu else 0,
            peak_reserved_bytes=torch.cuda.max_memory_reserved() if gpu else 0)
        if (report["seconds"]>300 or report["peak_RSS_bytes"]>limits["peak_RSS_bytes"]
            or report["peak_allocated_bytes"]>limits["peak_allocated_bytes"]
            or report["peak_reserved_bytes"]>limits["peak_reserved_bytes"]):
            failure = failure or RuntimeError("Final preservation resource bound failed"); report["passed"] = False
        if failure is not None and "error" not in report:
            report["error"] = dict(type=type(failure).__name__, message=str(failure))
        with (folder/"capture_report.json").open("x") as stream:
            json.dump(_observed(report), stream, indent=2, allow_nan=False); stream.write("\n")
    if failure is not None:
        raise RuntimeError("Terminal new-source capture; preserve actual evidence, no retry") from failure
    return report
