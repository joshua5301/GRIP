"""Narrow public-helper parity on the admitted NEW132 saved CPU oracle only."""
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

KIND = "single_joint_mean_CE_NEW132_saved_oracle_helper_parity_v1"
CONTRACT_SHA = "586cd63ec76044405d40740e885465a584c39137068f349e81d5692b2fdc8e13"
HELPER_SHA = "91a7a5f29f4293e28f4b94c3271692502bdba3c1e51ec9cddbd07d0e0d633cfc"


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _sha(path, check=lambda: None):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
            check()
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
            for v in x.values():
                visit(v)
        elif isinstance(x, list):
            for v in x:
                visit(v)
    visit(value)
    return found


def _observed(x):
    if isinstance(x, float) and not math.isfinite(x):
        return {"nonfinite_observation": repr(x)}
    if isinstance(x, dict):
        return {str(k): _observed(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_observed(v) for v in x]
    return x


def run(protocol_path, protocol_sha256, stop=lambda: False):
    started = time.monotonic()
    _require(callable(stop), "Stop must be callable")
    path = Path(protocol_path)
    _require(path.is_absolute() and str(path) == str(path.resolve()) and _sha(path) == protocol_sha256, "Protocol path/SHA differs")
    packet = json.loads(path.read_text())
    _require(type(packet["schema"]) is int and packet["schema"] == 1 and packet["kind"] == KIND, "Protocol schema/kind differs")
    science = packet["scientific_contract"]
    _require(science["sha256"] == CONTRACT_SHA and _sha(science["path"]) == CONTRACT_SHA, "Frozen science differs")
    contract = json.loads(Path(science["path"]).read_text())
    folder = Path(packet["output_folder"])
    _require(folder.is_absolute() and str(folder) == str(folder.resolve()) and not folder.exists(), "Fresh normalized ABS namespace required")
    folder.mkdir(parents=True)
    arrays_path, report_path = folder / "helper_parity_arrays.pt", folder / "helper_parity_report.json"
    work = {k: 0 for k in contract["exact_counts"]}
    evidence = dict(inputs_before={}, inputs_after={}, returned={})
    report = dict(schema=1, kind=KIND, passed=False, source=packet["source"],
        scientific_contract=science, protocol=dict(path=str(path), sha256=protocol_sha256),
        helper_entrypoint=packet["helper_entrypoint"], accepted_oracle=contract["required_refs"]["accepted_NEW132_oracle"],
        work=work, gates=[], returned_descriptors={}, failure=None,
        scope="Saved NEW132 synthetic CPU helper parity only; not source/native/kernel/student qualification",
        raw_evidence=dict(path=str(arrays_path), exists=False, bytes=None, sha256=None, hash_unknown=True, hash_error=None, write_completed=False))
    torch = None
    old_profile = sys.getprofile()
    old_handler, old_timer = signal.getsignal(signal.SIGALRM), signal.getitimer(signal.ITIMER_REAL)
    def peaks():
        return dict(seconds=time.monotonic()-started, peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    def check():
        p = peaks()
        if stop() or p["seconds"] > 300 or p["peak_RSS_bytes"] > 16*1024**3:
            raise InterruptedError("Frozen CPU time/stop/RSS boundary reached")
    def alarm(signum, frame):
        raise TimeoutError("Frozen 300s CPU deadline reached")
    def pins():
        required = _refs(dict(packet=packet, contract=contract))
        expected = packet["readonly_files_sha256"]
        _require(all(Path(p).is_absolute() and str(Path(p)) == str(Path(p).resolve()) for p in expected), "Readonly keys must be normalized ABS")
        _require(all(expected.get(p) == h for p, h in required.items()), "Required readonly admission omitted")
        return {p: _sha(p, check) for p in expected}
    def gate(name, passed, **details):
        report["gates"].append(dict(name=name, passed=bool(passed), **details))
    try:
        signal.signal(signal.SIGALRM, alarm)
        signal.setitimer(signal.ITIMER_REAL, max(1e-6, 300-(time.monotonic()-started)))
        report["readonly_entry"] = pins()
        _require(report["readonly_entry"] == packet["readonly_files_sha256"], "Readonly entry bytes differ")
        _require(Path(packet["entrypoint"]["path"]).resolve() == Path(__file__).resolve(), "Parity entrypoint differs")
        from src.research_loop import implementation_provenance
        report["source_entry"] = implementation_provenance()
        _require(report["source_entry"] == packet["source"], "Current full source/Git differs")
        _require(platform.python_version() == "3.12.7" and all(os.environ.get(k) == "4" for k in
            ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")), "Frozen CPU environment differs")
        refs = contract["required_refs"]
        proof = json.loads(Path(refs["proof_report"]["path"]).read_text())
        root = json.loads(Path(refs["proof_ROOT_acceptance"]["path"]).read_text())
        peer = json.loads(Path(refs["proof_independent_actual_acceptance"]["path"]).read_text())
        producer = json.loads(Path(contract["accepted_oracle_producer_source"]["manifest"]["path"]).read_text())
        _require(proof["passed"] and proof["math_completed"] and proof["failure"] is None
            and proof["seed"] == 510052 and proof["source"] == producer
            and proof["scientific_contract"] == refs["proof_science"] and proof["protocol"] == refs["proof_protocol"], "NEW132 proof admission differs")
        _require(root["passed"] and root["source"] == producer and root["attempts"] == 1
            and root["report"] == refs["proof_report"] and root["owning_oracle_arrays"] == refs["accepted_NEW132_oracle"], "ROOT NEW132 admission differs")
        _require(peer["passed"] and peer["synthetic_CPU_algebra_metadata_accepted"]
            and not peer["production_native_source_student_qualification"]
            and peer["actual_report"] == refs["proof_report"] and peer["actual_arrays_opaque_only"] == refs["accepted_NEW132_oracle"]
            and peer["source"]["git_head"] == producer["git_head"] and peer["source"]["source_digest"] == producer["source_digest"], "Independent NEW132 admission differs")
        oracle = refs["accepted_NEW132_oracle"]
        _require(proof["raw_evidence"]["path"] == oracle["path"] and proof["raw_evidence"]["sha256"] == oracle["sha256"]
            and proof["raw_evidence"]["write_completed"] and proof["raw_evidence"]["bytes"] == 255551
            and Path(oracle["path"]).stat().st_size == 255551, "Owning NEW132 payload differs")
        report["admitted_proof_source"] = contract["accepted_oracle_producer_source"]
        import torch
        _require(str(torch.__version__) == "2.9.1+cu128", "Torch version differs")
        torch.set_num_threads(4)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        _require(torch.get_num_threads() == 4 and torch.get_num_interop_threads() == 1, "Thread counts differ")
        report["runtime"] = dict(python=platform.python_version(), torch=str(torch.__version__),
            threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(), device="cpu", dtype="torch.float64")
        def own(x):
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().clone()
            if isinstance(x, dict):
                return {k: own(v) for k, v in x.items()}
            return x
        def descriptor(x):
            if not isinstance(x, torch.Tensor):
                return dict(tensor=False, type=type(x).__name__)
            return dict(tensor=True, shape=list(x.shape), dtype=str(x.dtype), device=str(x.device),
                requires_grad=x.requires_grad, grad_fn_none=x.grad_fn is None, finite=bool(torch.isfinite(x).all()))
        def valid(value, shape):
            return value == dict(tensor=True, shape=shape, dtype="torch.float64", device="cpu",
                requires_grad=False, grad_fn_none=True, finite=True)
        from src import joint_mean_ce as helper
        _require(Path(helper.__file__).resolve() == Path(packet["helper_entrypoint"]["path"]).resolve()
            and packet["helper_entrypoint"]["sha256"] == HELPER_SHA and _sha(helper.__file__, check) == HELPER_SHA, "Promoted helper differs")
        work["oracle_PT_load_attempts"] += 1
        payload = torch.load(oracle["path"], map_location="cpu", weights_only=True)
        work["oracle_PT_loads"] += 1
        _require(type(payload) is dict and type(payload.get("oracle")) is dict, "NEW132 oracle structure differs")
        inputs = {}
        for key in ("current_M", "theta", "raw_CE_adjoint", "CE0", "raw_moment_cotangent", "complete_moment_cotangent"):
            inputs[key] = payload["oracle"][key]
            evidence["inputs_before"][key] = own(inputs[key])
            work["oracle_selected_fields"] += 1
        del payload
        shapes = dict(current_M=[4,12], theta=[3,9], raw_CE_adjoint=[3,9],
            raw_moment_cotangent=[4,12], complete_moment_cotangent=[4,12])
        report["input_descriptors"] = {k: descriptor(inputs[k]) for k in shapes}
        _require(all(valid(report["input_descriptors"][k], shape) for k, shape in shapes.items()), "Selected matrices differ")
        CE0 = inputs["CE0"]
        report["CE0"] = CE0
        _require(type(CE0) in (int, float) and math.isfinite(CE0) and CE0 > 1e-12, "Frozen own CE0 invalid")
        layout = helper.JointMeanLayout(3,5,3)
        work["layout_instances"] += 1
        report["layout"] = layout.descriptor()
        _require(report["layout"]["material_offsets"] == dict(mass=[0,1], z=[1,4], original_Phi=[4,9], raw_Q=[9,12])
            and layout.critic_dimension == 8 and layout.material_width == 12
            and report["layout"]["block_scales"] == [1,1] and report["layout"]["head_bias"] == "last", "Joint layout differs")
        codes = {helper.raw_moment_cotangent.__code__: "profile_raw_moment_cotangent_calls",
            helper.complete_moment_cotangent.__code__: "profile_complete_moment_cotangent_calls",
            helper.moment_gradient.__code__: "profile_moment_gradient_calls",
            helper.moment_gradient.__globals__["head_gradient"].__code__: "profile_head_gradient_calls",
            helper.decode_moments.__code__: "profile_decode_moments_calls"}
        report["profile_functions"] = {label: dict(name=code.co_name, path=code.co_filename) for code, label in codes.items()}
        _require(old_profile is None and sys.getprofile() is None, "Existing profiler cannot be replaced")
        def profile(frame, event, arg):
            if event == "call":
                label = codes.get(frame.f_code)
                if label is not None:
                    work[label] += 1
        sys.setprofile(profile)
        try:
            work["explicit_raw_attempts"] += 1
            raw = helper.raw_moment_cotangent(inputs["current_M"], layout, inputs["theta"], inputs["raw_CE_adjoint"], .07)
            evidence["returned"]["raw"] = own(raw)
            work["explicit_raw_returns"] += 1
            work["explicit_complete_attempts"] += 1
            complete = helper.complete_moment_cotangent(inputs["current_M"], layout, inputs["theta"], inputs["raw_CE_adjoint"], .07, CE0)
            evidence["returned"]["complete"] = own(complete)
            work["explicit_complete_returns"] += 1
        finally:
            sys.setprofile(old_profile)
        gate("profiler_restored", sys.getprofile() is old_profile)
        for name, value in (("raw", raw), ("complete", complete)):
            report["returned_descriptors"][name] = descriptor(value)
            gate(name+"_shape_detached_finite_CPUFP64", valid(report["returned_descriptors"][name], [4,12]))
        _require(all(valid(v, [4,12]) for v in report["returned_descriptors"].values()), "Returned matrix domain differs")
        check()
        bounds = contract["gates"]["blockwise_BOTH"]
        def compare(name, actual, expected):
            for block, (lo, hi) in contract["blocks"].items():
                a, b = actual[:,lo:hi], expected[:,lo:hi]
                absolute = float((a-b).abs().max())
                relative = absolute/max(float(b.abs().max()),1e-12)
                work["numerical_block_gates"] += 1
                gate(name+"_"+block, math.isfinite(absolute) and math.isfinite(relative)
                    and absolute <= bounds["abs"] and relative <= bounds["rel"], max_abs=absolute,
                    relative_Linf=relative, bounds=dict(abs=bounds["abs"],rel=bounds["rel"],require="BOTH"))
        compare("raw_vs_saved", raw, inputs["raw_moment_cotangent"])
        compare("complete_vs_saved", complete, inputs["complete_moment_cotangent"])
        compare("whole_CE0_units", complete, raw/CE0)
        gate("saved_complete_exact_whole_raw_over_CE0", torch.equal(inputs["complete_moment_cotangent"], inputs["raw_moment_cotangent"]/CE0))
        gate("returned_complete_exact_whole_raw_over_CE0", torch.equal(complete, raw/CE0))
        evidence["inputs_after"] = own(inputs)
        gate("all_selected_inputs_torch_equal", all(torch.equal(inputs[k], evidence["inputs_before"][k]) for k in shapes)
            and inputs["CE0"] == evidence["inputs_before"]["CE0"] == CE0)
        for key, count in contract["exact_counts"].items():
            gate("exact_count_"+key, work[key] == count, actual=work[key], expected=count)
        report["helper_parity_completed"] = True
    except BaseException as error:
        report["failure"] = dict(type=type(error).__name__, message=str(error))
    finally:
        sys.setprofile(old_profile)
        report["profiler_restored_at_exit"] = sys.getprofile() is old_profile
        signal.setitimer(signal.ITIMER_REAL,0)
        signal.signal(signal.SIGALRM,old_handler)
        try:
            if torch is not None:
                with arrays_path.open("xb") as f:
                    torch.save(evidence,f); f.flush(); os.fsync(f.fileno())
                report["raw_evidence"]["write_completed"] = True
            if arrays_path.exists():
                report["raw_evidence"]["exists"] = True
                report["raw_evidence"]["bytes"] = arrays_path.stat().st_size
                try:
                    report["raw_evidence"]["sha256"] = _sha(arrays_path)
                    report["raw_evidence"]["hash_unknown"] = False
                except BaseException as error:
                    report["raw_evidence"]["hash_error"] = dict(type=type(error).__name__,message=str(error))
                    raise
            report["readonly_exit"] = pins()
            from src.research_loop import implementation_provenance
            report["source_exit"] = implementation_provenance()
            _require(report["readonly_exit"] == packet["readonly_files_sha256"] and report["source_exit"] == packet["source"], "Source/readonly exit bytes differ")
            check()
        except BaseException as error:
            report["preservation_or_exit_error"] = dict(type=type(error).__name__,message=str(error))
            if report["failure"] is None:
                report["failure"] = report["preservation_or_exit_error"]
        report["resources"] = peaks()
        if report["failure"] is None and any(not g["passed"] for g in report["gates"]):
            report["failure"] = dict(type="GateFailure",gates=[g["name"] for g in report["gates"] if not g["passed"]])
        if report["failure"] is None and (report["resources"]["seconds"] > 300 or report["resources"]["peak_RSS_bytes"] > 16*1024**3):
            report["failure"] = dict(type="ResourceBoundary",message="Output preservation exceeded boundary")
        report["passed"] = bool(report.get("helper_parity_completed") and report["failure"] is None
            and report["raw_evidence"]["write_completed"] and not report["raw_evidence"]["hash_unknown"]
            and report["profiler_restored_at_exit"] and all(g["passed"] for g in report["gates"]))
        with report_path.open("x") as f:
            json.dump(_observed(report),f,indent=2,allow_nan=False); f.write("\n"); f.flush(); os.fsync(f.fileno())
        final = peaks()
        if report["passed"] and (final["seconds"] > 300 or final["peak_RSS_bytes"] > 16*1024**3):
            report.update(passed=False,resources=final,failure=dict(type="ResourceBoundary",message="Final report preservation exceeded boundary"))
            with report_path.open("w") as f:
                json.dump(_observed(report),f,indent=2,allow_nan=False); f.write("\n"); f.flush(); os.fsync(f.fileno())
        if old_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL,*old_timer)
    if not report["passed"]:
        raise RuntimeError("NEW132 helper parity failed terminally; inspect owning report and partial evidence")
    return report
