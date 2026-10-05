"""Source-only, bounded, immutable Flickr z/Q export; no native/critic admission."""
import hashlib
import json
import math
import os
import platform
import resource
import signal
import subprocess
import time
from pathlib import Path

KIND = "Flickr_shared_original_source122_filebacked_component_export_stageFE_v1"
CONTRACT_SHA = "237ca4811a68d1f8c7ac6afe51d26552946c88bc6cf5ad4b15eec9e294ddd49f"
PARTITION = (2048,) * 21 + (1617,)
SCRATCH = 4194304


def require(ok, message):
    if not ok: raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path, guard=lambda: None):
    digest, buffer = hashlib.sha256(), bytearray(SCRATCH)
    with Path(path).open("rb", buffering=0) as stream:
        while True:
            guard(); count = stream.readinto(buffer)
            if not count: break
            digest.update(memoryview(buffer)[:count])
    return digest.hexdigest()


def refs(value, result=None):
    result = {} if result is None else result
    if isinstance(value, dict):
        if "path" in value and "sha256" in value:
            p, h = value["path"], value["sha256"]
            require(isinstance(p, str) and Path(p).is_absolute() and str(Path(p).resolve()) == p, "Normalized ABS ref")
            require(isinstance(h, str) and len(h) == 64 and all(c in "0123456789abcdef" for c in h), "SHA256 ref")
            require(p not in result or result[p] == h, "Conflicting immutable ref")
            result[p] = h
        for item in value.values(): refs(item, result)
    elif isinstance(value, (list, tuple)):
        for item in value: refs(item, result)
    return result


def source(repo, guard):
    files = {str(p.relative_to(repo)): sha(p, guard) for p in sorted((repo / "src").glob("*.py"))}
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, text=True, capture_output=True, check=True)
    return dict(git_head=revision.stdout.strip(), source_digest=hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()[:12], files=files)


def safe(value):
    if isinstance(value, float) and not math.isfinite(value): return dict(nonfinite=repr(value))
    if isinstance(value, dict): return {k: safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [safe(v) for v in value]
    return value


def write(path, value):
    with Path(path).open("w") as stream:
        json.dump(safe(value), stream, indent=2, allow_nan=False); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())


def run(protocol_path, protocol_sha256, stop=lambda: False):
    started, torch, folder, error, namespace_created = time.monotonic(), None, None, None, False
    report = dict(schema=1, kind=KIND, passed=False, gates={}, work={}, components={}, output_evidence={}, scope=dict(real_native_admission=False,critic_admission=False,student_or_efficacy_admission=False))
    work, maps, active = report["work"], [], dict(peak_conservative_bytes=SCRATCH, current_bytes=SCRATCH)
    work.update({key:0 for key in ("source_packet_mmap_deserialize","new_z_npy","new_Q_npy","source_export_blocks_per_component","output_readonly_verify_blocks_per_component","existing_Phi_verify_blocks","metadata_manifest_writes","full_source_clones","source_tensor_to_GPU","source_fits","Q_decodes","source_SGC","Phi_writes","kernel_map_forward","heads","adjoints","P_updates","students","tests")})
    limits = dict(seconds=300, active_and_transient_row_storage_bytes=536870912, peak_RSS_bytes=17179869184)
    def resource_ok():
        return time.monotonic() - started <= 300 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 <= limits["peak_RSS_bytes"] and active["current_bytes"] <= limits["active_and_transient_row_storage_bytes"] and (torch is None or not torch.cuda.is_initialized())
    def guard():
        if stop(): raise InterruptedError("stop requested; terminal namespace")
        if not resource_ok(): raise TimeoutError("Frozen wall/RSS/active-storage/CUDA boundary")
    def gate(name, ok):
        report["gates"][name] = bool(ok); require(ok, name)
    def attempt(key): work[key + "_attempted"] = work.get(key + "_attempted", 0) + 1
    def completed(key): work[key] = work.get(key, 0) + 1
    def storage(block):
        terms = dict(mapped_row=block.nbytes, owning_row=block.nbytes, mapped_output_row=block.nbytes, conservative_hash_alias=block.nbytes, boolean_scratch=2 * block.size, row_totals=len(block) * 8, class_totals=56, file_hash_scratch=SCRATCH)
        active["current_bytes"] = sum(terms.values()); active["peak_conservative_bytes"] = max(active["peak_conservative_bytes"], active["current_bytes"])
        report["active_storage_accounting"]["largest_charge_terms"] = terms
    def finish_file(path, evidence):
        evidence["exists"] = path.exists()
        if evidence["exists"]:
            evidence["bytes"] = path.stat().st_size; evidence["hash_unknown"] = True
            evidence["wholebyte_hash_attempts"] = evidence.get("wholebyte_hash_attempts",0)+1
            try: evidence["sha256"] = sha(path, guard); evidence["hash_unknown"] = False
            except BaseException as exc: evidence["hash_error"] = f"{type(exc).__name__}: {exc}"
    old_handler = signal.getsignal(signal.SIGALRM)
    def deadline(_signum, _frame): raise TimeoutError("Frozen 300-second alarm")
    signal.signal(signal.SIGALRM, deadline); signal.setitimer(signal.ITIMER_REAL, 300)
    try:
        p = Path(protocol_path)
        require(p.is_absolute() and str(p.resolve()) == str(p) and sha(p, guard) == protocol_sha256, "Protocol ABS/hash")
        packet = read(p); folder = Path(packet["output_folder"])
        require(folder.is_absolute() and str(folder.resolve()) == str(folder), "Exclusive ABS output namespace")
        folder.mkdir(parents=True, exist_ok=False); namespace_created = True
        report.update(protocol=dict(path=str(p), sha256=protocol_sha256), scientific_contract=packet["scientific_contract"], source=packet["source"], active_storage_accounting=dict(**active, scope="Conservative active mapped/owning/output/hash/boolean row overlap plus fixed4MiB scratch; full mapped virtual lengths and process RSS are separate disclosed measurements."))
        gate("kind_and_contract", packet["kind"] == KIND and packet["scientific_contract"]["sha256"] == CONTRACT_SHA)
        contract = read(packet["scientific_contract"]["path"]); required = refs(packet); refs(contract,required)
        pins = packet["readonly_files_sha256"]
        gate("normalized_pin_closure", all(Path(q).is_absolute() and str(Path(q).resolve()) == q for q in pins) and all(pins.get(q) == h for q, h in required.items()))
        report["entry_files_sha256"] = {q: sha(q, guard) for q in pins}; gate("readonly_entry_exact", report["entry_files_sha256"] == pins)
        repo = Path(packet["repository_root"]); gate("repository_ABS", repo.is_absolute() and str(repo.resolve()) == str(repo))
        report["source_entry"] = source(repo, guard); gate("current_source_entry_exact", report["source_entry"] == packet["source"])
        gate("entrypoint_exact", str(Path(__file__).resolve()) == packet["entrypoint"]["path"] and sha(__file__, guard) == packet["entrypoint"]["sha256"])
        for role in ("ROOT_static_acceptance", "independent_static_acceptance"):
            bridge = read(packet[role]["path"])
            gate(role, bridge.get("passed") is True and all(bridge.get(k) == packet[k] for k in ("source", "entrypoint", "scientific_contract")))
        supplier = contract["supplier"]; cap = read(supplier["producer_full_source_authority"]["path"])
        root = read(contract["source122_ROOT"]["path"]); peer = read(contract["source122_independent"]["path"])
        report["historical_source122"] = dict(source=cap["source"], report=supplier["producer_full_source_authority"], arrays=supplier["owning_packet"], ROOT=contract["source122_ROOT"], independent=contract["source122_independent"], transform_descriptors=supplier["transform_descriptors"], original_assets=supplier["original_assets"], native_origins_are_metadata_only=contract["original_K_specific_producers"])
        gate("source122_actual_admissions", cap["passed"] is True and cap["kind"] == "large_kernel_mean_new_source_capture_v1" and cap["source"] == contract["source122_full_producer"] and root["passed"] is True and root["source"] == cap["source"] and root["actual_report"] == supplier["producer_full_source_authority"] and root["actual_arrays"] == supplier["owning_packet"] and peer["passed"] is True and peer["actual_report"] == supplier["producer_full_source_authority"] and all(peer["actual_producer"][k] == cap["source"][k] for k in ("git_head", "source_digest")) and all(peer["raw_opaque_evidence"][k] == supplier["owning_packet"][k] for k in ("path", "sha256")))
        gate("literal_descriptors_and_policy", cap["array_descriptors"]["z"] == supplier["z_descriptor"] and cap["array_descriptors"]["Q"] == supplier["Q_descriptor"] and cap["transform_descriptors"] == supplier["transform_descriptors"] and cap["original_Phi_identity"] == supplier["original_Phi_identity"] and contract["partition"] == list(PARTITION) and contract["dimensions"] == dict(N=44625,D=500,B=512,C=7))
        import numpy as np
        import torch as torch_module
        torch = torch_module
        torch.set_num_threads(4)
        if torch.get_num_interop_threads() != 1: torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False; torch.set_float32_matmul_precision("highest")
        runtime = dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__, default_dtype=str(torch.get_default_dtype()), threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(), autocast_cpu=torch.is_autocast_enabled("cpu"), autocast_cuda=torch.is_autocast_enabled("cuda"), deterministic_algorithms=torch.are_deterministic_algorithms_enabled(), allow_tf32_matmul=torch.backends.cuda.matmul.allow_tf32, allow_tf32_cudnn=torch.backends.cudnn.allow_tf32, float32_matmul_precision=torch.get_float32_matmul_precision(), CUDA_initialized=torch.cuda.is_initialized(), environment={k:os.environ.get(k) for k in contract["runtime"]["environment"]})
        report["runtime"] = runtime; gate("runtime_exact_CPU_only", runtime == contract["runtime"]); guard()
        gate("prospective_max_row_overlap", 4*2048*512*8+2*2048*512+2048*8+56+SCRATCH <= limits["active_and_transient_row_storage_bytes"])
        for key, value in contract["expected_work"].items(): work[key] = 0
        for name in ("z", "Q"):
            for role in ("source_export_blocks_", "output_readonly_verify_blocks_"): work[role + name] = 0
        attempt("source_packet_mmap_deserialize")
        payload = torch.load(supplier["owning_packet"]["path"], mmap=True, map_location="cpu", weights_only=False)
        completed("source_packet_mmap_deserialize"); report["mapped_packet"] = dict(path=supplier["owning_packet"]["path"], logical_file_bytes=Path(supplier["owning_packet"]["path"]).stat().st_size, type=type(payload).__name__, load=dict(mmap=True,map_location="cpu",weights_only=False), no_eager_fallback=True)
        gate("owning_packet_dictionary", type(payload) is dict)
        def header(value):
            result = dict(type=type(value).__name__)
            if isinstance(value, torch.Tensor): result.update(shape=list(value.shape),dtype=str(value.dtype),device=str(value.device),layout=str(value.layout),stride=list(value.stride()),storage_offset=value.storage_offset(),storage_bytes=value.untyped_storage().nbytes(),requires_grad=value.requires_grad)
            return result
        def digest(shape, dtype): return hashlib.sha256(json.dumps(dict(shape=tuple(shape),dtype=dtype),sort_keys=True).encode())
        def domain(block, name):
            gate(f"{name}_finite", bool(np.isfinite(block).all()))
            if name == "Q":
                gate("Q_nonnegative_positive_rows", not bool((block < 0).any()) and bool((block.sum(1) > 0).all()))
        for name, width in (("z",500),("Q",7)):
            value = payload[name]; report["components"][name] = dict(mapped_tensor=header(value), source_selector=f"payload.{name}", original_descriptor=supplier[name + "_descriptor"], source_blocks=[], verification_blocks=[])
            item = report["components"][name]; path = folder / f"{name}.npy"
            gate(f"{name}_mapped_tensor_domain", isinstance(value,torch.Tensor) and value.device.type == "cpu" and value.dtype == torch.float64 and value.layout == torch.strided and list(value.shape) == [44625,width] and value.is_contiguous() and value.stride() == (width,1) and value.storage_offset() == 0 and value.untyped_storage().nbytes() == 44625*width*8 and not value.requires_grad)
            original, exported, class_totals = digest(value.shape,"torch.float64"), digest(value.shape,"float64"), np.zeros(7,dtype=np.float64)
            evidence = report["output_evidence"][name] = dict(path=str(path), exists=False, write_complete=False, accepted=False, hash_unknown=True)
            attempt(f"new_{name}_npy"); out = np.lib.format.open_memmap(path,mode="w+",dtype=np.float64,shape=(44625,width),fortran_order=False,version=(1,0)); maps.append(out)
            evidence["exists"] = path.exists(); evidence["bytes"] = path.stat().st_size; completed(f"new_{name}_npy")
            start = 0
            for rows in PARTITION:
                guard(); attempt("source_export_blocks_" + name)
                view = value[start:start+rows]; numpy_view = view.numpy(); block = np.array(numpy_view,copy=True,order="C"); storage(block)
                raw = memoryview(block).cast("B"); original.update(raw); exported.update(raw)
                row_record = dict(start=start,rows=rows,bytes=block.nbytes,row_bytes_sha256=hashlib.sha256(raw).hexdigest(),write_attempted=True,returned_write=False); item["source_blocks"].append(row_record)
                out[start:start+rows] = block; completed("source_export_blocks_" + name); row_record["returned_write"] = True
                evidence["rows_written"] = start+rows; guard(); domain(block,name)
                if name == "Q": class_totals += block.sum(0)
                guard(); start += rows; del raw,block,numpy_view,view; active["current_bytes"] = SCRATCH
            out.flush(); out._mmap.close(); maps.pop(); del out
            with path.open("rb") as stream: os.fsync(stream.fileno())
            evidence["write_complete"] = True; item.update(original_Torch_content=dict(shape=[44625,width],dtype="torch.float64",digest=original.hexdigest()),exported_NumPy_content=dict(shape=[44625,width],dtype="float64",digest=exported.hexdigest()))
            gate(f"{name}_original_content_exact", original.hexdigest() == supplier[name+"_descriptor"]["digest"])
            if name == "Q": item["class_aggregates"] = class_totals.tolist(); gate("Q_all_supported_classes", bool(np.isfinite(class_totals).all()) and bool((class_totals>0).all()))
            os.chmod(path,0o444); finish_file(path,evidence); gate(f"{name}_file_hash_known", not evidence["hash_unknown"])
            del value,class_totals; guard()
        def verify(path, name, shape, expected_digest, original=False):
            attempt("readonly_open_" + name); array = np.load(path,mmap_mode="r",allow_pickle=False); maps.append(array); completed("readonly_open_" + name)
            info = dict(path=str(path),shape=list(array.shape),dtype=str(array.dtype),descr=array.dtype.str,fortran_order=not array.flags.c_contiguous,data_offset=int(array.offset),logical_data_bytes=int(array.nbytes),writeable=bool(array.flags.writeable))
            report["components"].setdefault(name,dict(verification_blocks=[]))["npy_header"] = info
            gate(name+"_readonly_header", isinstance(array,np.memmap) and list(array.shape) == shape and array.dtype == np.dtype("float64") and array.flags.c_contiguous and not array.flags.writeable and array.dtype.str == "<f8" and info["data_offset"] == 128)
            hashed, start = digest(shape,"float64"), 0
            for rows in PARTITION:
                guard(); key = "existing_Phi_verify_blocks" if original else "output_readonly_verify_blocks_" + name; attempt(key)
                block = np.array(array[start:start+rows],copy=True,order="C"); storage(block); raw = memoryview(block).cast("B"); hashed.update(raw); completed(key)
                report["components"][name]["verification_blocks"].append(dict(start=start,rows=rows,bytes=block.nbytes,row_bytes_sha256=hashlib.sha256(raw).hexdigest()))
                domain(block,name); guard(); start += rows; del raw,block; active["current_bytes"] = SCRATCH
            result = dict(shape=shape,dtype="float64",digest=hashed.hexdigest()); report["components"][name]["readonly_verified_content"] = result
            array._mmap.close(); maps.pop(); del array; gate(name+"_verified_content_exact", result["digest"] == expected_digest); return result
        for name,width in (("z",500),("Q",7)):
            verify(folder/f"{name}.npy",name,[44625,width],report["components"][name]["exported_NumPy_content"]["digest"])
            report["output_evidence"][name]["accepted"] = True
        sidecar = read(supplier["original_assets"]["Phi_sidecar"]["path"]); report["original_Phi_sidecar"] = sidecar
        gate("original_Phi_sidecar_exact", sidecar == supplier["original_Phi_identity"])
        verify(Path(supplier["original_assets"]["Phi"]["path"]),"Phi",[44625,512],sidecar["phi_digest"],True)
        work["source_export_blocks_per_component"] = 22; work["output_readonly_verify_blocks_per_component"] = 22
        gate("ordered_pass_counts", work["source_packet_mmap_deserialize"] == 1 and work["new_z_npy"] == work["new_Q_npy"] == 1 and all(work[role+name] == 22 for role in ("source_export_blocks_","output_readonly_verify_blocks_") for name in ("z","Q")) and work["existing_Phi_verify_blocks"] == 22)
        report["scope"] = contract["source_scope"]; report["mapped_virtual_lengths"] = dict(source_packet=report["mapped_packet"]["logical_file_bytes"],z=178500000,Q=2499000,Phi=182784000,not_residency_exemption=True)
        manifest = dict(schema=1,kind="Flickr_shared_original_source122_immutable_fullwidth_components_v1",exporter_source=packet["source"],scientific_contract=packet["scientific_contract"],source122=report["historical_source122"],partition=list(PARTITION),row_order=contract["row_order"],components={name:dict(file=report["output_evidence"][name],original_Torch_descriptor=report["components"][name]["original_Torch_content"],NumPy_descriptor=report["components"][name]["readonly_verified_content"],header=report["components"][name]["npy_header"]) for name in ("z","Q")},original_Phi=dict(file=supplier["original_assets"]["Phi"],sidecar=supplier["original_assets"]["Phi_sidecar"],identity=sidecar,header=report["components"]["Phi"]["npy_header"]),scope=contract["source_scope"])
        attempt("metadata_manifest_writes"); write(folder/"source_component_manifest.json",manifest); completed("metadata_manifest_writes"); report["manifest"] = dict(path=str(folder/"source_component_manifest.json"))
        report["manifest_evidence"] = dict(path=report["manifest"]["path"],exists=True,bytes=(folder/"source_component_manifest.json").stat().st_size,write_complete=True,hash_unknown=True); finish_file(folder/"source_component_manifest.json",report["manifest_evidence"])
        gate("manifest_hash_known", not report["manifest_evidence"]["hash_unknown"]); report["manifest"]["sha256"] = report["manifest_evidence"]["sha256"]
        report["exit_files_sha256"] = {q:sha(q,guard) for q in pins}; gate("readonly_exit_exact", report["exit_files_sha256"] == pins)
        report["source_exit"] = source(repo,guard); gate("current_source_exit_exact", report["source_exit"] == packet["source"])
        gate("frozen_work_exact", all(work[k] == v for k,v in contract["expected_work"].items()))
        guard(); gate("final_pre_report_resources", resource_ok()); report["passed"] = True
    except BaseException as exc:
        error = exc; report["failure"] = dict(error=f"{type(exc).__name__}: {exc}",terminal=True,no_retry=True)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        for mapped in maps:
            try: mapped.flush(); mapped._mmap.close()
            except BaseException as exc: report.setdefault("cleanup_errors",[]).append(f"{type(exc).__name__}: {exc}"); report["passed"] = False
        if namespace_created:
            for evidence in report["output_evidence"].values():
                path = Path(evidence["path"])
                try:
                    if path.exists(): os.chmod(path,0o444)
                    if evidence.get("hash_unknown",True): finish_file(path,evidence)
                except BaseException as exc: evidence["preservation_error"] = f"{type(exc).__name__}: {exc}"; report["passed"] = False
            report["seconds"] = time.monotonic()-started; report["peak_RSS_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
            report["active_storage_accounting"].update(active); report["peak_allocated_bytes"] = report["peak_reserved_bytes"] = 0
            report["CUDA_initialized_exit"] = False if torch is None else torch.cuda.is_initialized()
            report["passed"] = report["passed"] and resource_ok() and all(not e.get("hash_unknown",True) for e in report["output_evidence"].values())
            if not report["passed"] and "failure" not in report: report["failure"] = dict(error="Final resource/ownership preservation boundary",terminal=True,no_retry=True)
            target = folder/"actual_export_report.json"
            try:
                write(target,report)
                report["seconds"] = time.monotonic()-started; report["peak_RSS_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
                if not resource_ok(): report["passed"] = False; report["failure"] = dict(error="Final serialization resource boundary",terminal=True,no_retry=True)
                write(target,report)
                if not resource_ok():
                    report["passed"] = False; report["seconds"] = time.monotonic()-started; report["peak_RSS_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
                    report["failure"] = dict(error="Final report serialization resource boundary",terminal=True,no_retry=True); write(target,report)
            except BaseException as exc: error = exc
        signal.setitimer(signal.ITIMER_REAL,0); signal.signal(signal.SIGALRM,old_handler)
    if error is not None: raise error
    require(report["passed"], "Terminal source-only export did not pass")
    return str(folder/"actual_export_report.json")
