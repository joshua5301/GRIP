"""Pinned physical-endpoint student evaluation only; no condensation or teacher work.

The ROOT protocol binds the current evaluator source separately from the source
that produced each immutable endpoint. Each invocation fits exactly one new GCN.
"""
import csv
import hashlib
import json
import math
import os
import platform
import resource
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

SETTINGS = dict(epochs=300, eval_every=10, hidden=256,
                dropout=0.31881090213944857, lr=0.01, weight_decay=0.0005)
COHORT = (20000, 20001, 20002)
ARMS = {"P0", "linear25", "kernel_mean25"}
POLICY = "Arxiv90_original_physical_uniform_GCN_paired_validation_v1"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def observed(value):
    if isinstance(value, float) and not math.isfinite(value):
        return dict(nonfinite_observation=repr(value))
    if isinstance(value, dict):
        return {str(k): observed(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [observed(v) for v in value]
    return value


def run(operation, protocol_path, protocol_sha256, arm, seed, output_dir,
        stop=lambda: False):
    require(operation == "evaluate" and arm in ARMS, "Evaluation-only operation/arm required")
    require(type(seed) is int and seed in COHORT, "Frozen fresh student cohort required")
    started = time.monotonic()
    counts = dict(student_fit_attempts=0, student_fits=0, route_replay_attempts=0,
                  route_replay_completed=0, physical_serving_routes=0,
                  origin_payload_loads=0, endpoint_payload_loads=0,
                  condensed_physical_readouts=0, completed_training_epochs=0,
                  teacher_fits=0, head_solves=0, adjoints=0, P_updates=0,
                  factor_factories=0, new_kernel_features=0, teacher_SGC_replays=0,
                  serving_SpMM_attempts=0, serving_SpMM_completed=0, tests=0)
    report = dict(schema=1, policy=POLICY, status="starting", passed=False,
                  arm=arm, seed=seed, test_enabled=False, efficacy_qualified=False,
                  primary="GCN validation at its first maximum epoch",
                  secondary="SGC+MLP with exactly the same selected GCN weights")
    folder = Path(output_dir).resolve()
    require(not folder.exists(), "Fresh exclusive evaluation namespace required")
    folder.mkdir(parents=True)
    torch = None
    pinned = {}
    error = None

    def memory():
        m = dict(peak_RSS_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024)
        if torch is not None and torch.cuda.is_initialized():
            m.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
                     peak_reserved_bytes=torch.cuda.max_memory_reserved(0))
        return m

    def stopped():
        if torch is not None and torch.cuda.is_initialized():
            torch.cuda.synchronize(0)
        m = memory()
        require(m["peak_RSS_bytes"] <= 16*1024**3, "Student RSS16GiB limit exceeded")
        require(m.get("peak_allocated_bytes", 0) <= 4*1024**3 and
                m.get("peak_reserved_bytes", 0) <= 6*1024**3, "Student GPU4/6GiB limit exceeded")
        return stop() or time.monotonic()-started >= 300

    def guard():
        if stopped():
            raise InterruptedError("Bounded student evaluation stopped")

    def sha(path, bounded=True):
        h = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for block in iter(lambda: stream.read(8*1024**2), b""):
                if bounded:
                    guard()
                h.update(block)
        return h.hexdigest()

    def pin(ref):
        require(set(ref) == {"path", "sha256"} and sha(ref["path"]) == ref["sha256"],
                "Pinned input changed: "+str(ref.get("path")))
        pinned[str(Path(ref["path"]).resolve())] = ref["sha256"]

    def save_report():
        report.update(counts=dict(counts), resources=memory(), seconds=time.monotonic()-started)
        (folder/"report.json").write_text(json.dumps(observed(report), indent=2, allow_nan=False)+"\n")

    try:
        pin(dict(path=str(Path(protocol_path).resolve()), sha256=protocol_sha256))
        protocol = json.loads(Path(protocol_path).read_text())
        require(protocol["schema"] == 1 and type(protocol["schema"]) is int and
                protocol["policy"] == POLICY and protocol["seeds"] == list(COHORT) and
                protocol["settings"] == SETTINGS and protocol["test_enabled"] is False,
                "Frozen validation protocol changed")
        require(set(protocol["arms"]) == ARMS, "Three fixed physical arms required")
        require(protocol["limits"] == dict(seconds=300, allocated_bytes=4*1024**3,
                reserved_bytes=6*1024**3, RSS_bytes=16*1024**3), "Bounds changed")
        repo = Path(protocol["repo"]).resolve()
        source = protocol["evaluator_source"]
        require(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo,
                text=True).strip() == source["git_head"], "Current evaluator Git changed")
        pin(protocol["evaluator_source_manifest"])
        require(json.loads(Path(protocol["evaluator_source_manifest"]["path"]).read_text()) == source,
                "Evaluator manifest changed")
        # Bind the implementation actually used here. Unused legacy prototypes
        # are neither imported nor read by this operation.
        required_sources = {"src/large_kernel_mean_validation.py", "src/inductive_evaluation.py",
                            "src/student_routes.py", "src/evaluation.py", "src/models.py",
                            "src/io.py", "src/shared_features.py", "src/sweep_utils.py",
                            "src/moments.py"}
        require(required_sources <= source["files"].keys(), "Missing evaluation source identities")
        for name in sorted(required_sources):
            pin(dict(path=str(repo/name), sha256=source["files"][name]))
        require(Path(__file__).resolve() == repo/"src/large_kernel_mean_validation.py",
                "Use the promoted, ROOT-pinned evaluator")
        for ref in protocol["references"].values():
            pin(ref)
        for ref in protocol["admissions"].values():
            pin(ref["reference"])
            accepted = json.loads(Path(ref["reference"]["path"]).read_text())
            for key, value in ref["required_fields"].items():
                require(accepted.get(key) == value and type(accepted.get(key)) is type(value),
                        "ROOT endpoint admission not passed: "+key)
        require({"native_fixed25", "matched_linear25"} <= protocol["admissions"].keys(),
                "Native candidate and exact-initialization linear control need ROOT admission")
        item = protocol["arms"][arm]
        require(item["step"] == (0 if arm == "P0" else 25) and
                item["origin"] == protocol["references"]["origin"], "Wrong endpoint/origin")
        require(set(item["producer_source"]) == {"git_head", "source_digest", "files"},
                "Each endpoint must retain its actual producer source")
        authority = protocol["admissions"][item["producer_admission"]]["reference"]
        accepted_source = json.loads(Path(authority["path"]).read_text())[item["producer_source_field"]]
        require(accepted_source == item["producer_source"], "Endpoint producer differs from its admitted actual source")
        if arm != "P0":
            pin(item["endpoint"])
            require(item["moment_key"] == ("moments" if arm == "linear25" else "physical_moments"),
                    "Critic-space moments cannot replace physical serving moments")
        require(all(os.environ.get(k) == v for k,v in protocol["environment"].items()),
                "Student environment changed")
        import numpy as np
        import torch as imported_torch
        torch = imported_torch
        from src.inductive_evaluation import fit_inductive_gcn
        from src.student_routes import replay_routes
        from src.shared_features import _tensor_identity
        from src.sweep_utils import representative
        require(platform.python_version() == protocol["runtime"]["Python"] and
                np.__version__ == protocol["runtime"]["NumPy"] and
                str(torch.__version__) == protocol["runtime"]["Torch"] and
                torch.version.cuda == protocol["runtime"]["CUDA"], "Evaluation runtime changed")
        require(torch.cuda.get_device_name(0) == protocol["GPU"], "Evaluation GPU changed")
        torch.set_num_threads(4)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        require(torch.get_default_dtype() == torch.float32 and not torch.is_autocast_enabled("cuda"),
                "Student precision changed")
        device = torch.device("cuda:0")
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(0)
        packet = torch.load(protocol["references"]["origin"]["path"], map_location="cpu", weights_only=True)
        counts["origin_payload_loads"] += 1
        arrays = packet["arrays"]
        capture = json.loads(Path(protocol["references"]["origin_capture"]["path"]).read_text())
        require(packet["context"] == capture["origin_context"] and capture["passed"] is True and
                packet["content_sha256"] == capture["origin_content_sha256"], "BW origin identity changed")
        require(arrays["X"].shape == (169343,128) and arrays["X"].dtype == torch.float32 and
                arrays["original_CSR"].layout == torch.sparse_csr and
                arrays["original_CSR"].dtype == torch.float32 and
                arrays["saved_H"].shape == arrays["X"].shape and
                arrays["saved_H"].dtype == torch.float32, "Original graph/SGC buffers changed")
        train, val = arrays["train_mask"], arrays["validation_mask"]
        require(train.dtype == val.dtype == torch.bool and train.shape == val.shape == (169343,) and
                not bool((train & val).any()) and bool(train.any()) and bool(val.any()), "Explicit disjoint splits required")
        allowed = train | val
        # The captured tensor is opaque outside the explicitly allowed splits.
        # Neither evaluation nor cache identities need held-out labels.
        labels = torch.zeros(169343, dtype=torch.long)
        require(arrays["graph_y_identity"].dtype == torch.int64 and
                arrays["graph_y_identity"].shape == (169343,), "Captured label tensor changed")
        labels[allowed] = arrays["graph_y_identity"][allowed]
        require(bool(((labels[allowed] >= 0) & (labels[allowed] < 40)).all()), "Allowed labels invalid")
        graph = dict(x=arrays["X"].to(device), adj=arrays["original_CSR"].to(device), y=labels.to(device))
        train, val = train.to(device), val.to(device)
        # Match the existing validation route: packed gcn_norm CSR, not the
        # teacher's independently normalized cached H. No cache/factory call.
        counts["serving_SpMM_attempts"] += 1
        with torch.no_grad():
            route_intermediate = torch.sparse.mm(graph["adj"],graph["x"])
        counts["serving_SpMM_completed"] += 1
        guard()
        counts["serving_SpMM_attempts"] += 1
        with torch.no_grad():
            route_h = torch.sparse.mm(graph["adj"],route_intermediate)
        counts["serving_SpMM_completed"] += 1
        del route_intermediate
        report["route_propagated_identity"] = _tensor_identity(route_h.detach().cpu())
        report["route_source"] = dict(operation="original packed CSR @ (original packed CSR @ X)",
                original_X_identity=packet["descriptors"]["X"],
                original_CSR_identity=packet["descriptors"]["original_CSR"],
                dtype="torch.float32",teacher_H_used=False,graph_products=2)
        guard()
        if arm == "P0":
            moment = arrays["M0"]
        else:
            snapshot = torch.load(item["endpoint"]["path"], map_location="cpu", weights_only=False)
            counts["endpoint_payload_loads"] += 1
            require(snapshot["step"] == 25 and type(snapshot["step"]) is int and snapshot["J_exact"] is True,
                    "Exact stationary fixed25 endpoint required")
            moment = snapshot[item["moment_key"]]
        require(moment.shape == (90,169) and moment.dtype == torch.float64 and
                bool(torch.isfinite(moment).all()) and bool((moment[:,0] > 0).all()), "Physical moments invalid")
        require(_tensor_identity(moment) == item["physical_moment_identity"], "Accepted physical endpoint identity changed")
        transform = SimpleNamespace(**{k:v.to(device) if torch.is_tensor(v) else v
                                      for k,v in arrays["transform"].items()})
        require(transform.kind == "rms" and transform.matrix is None, "Use original stored RMS inverse")
        cx, cy, mass = representative(moment, transform, 128, device)
        counts["condensed_physical_readouts"] += 1
        if arm == "P0":
            initial_readout = [cx, cy, torch.full_like(mass,1/90)]
            require(all(_tensor_identity(x.detach().cpu()) == _tensor_identity(arrays["readout"][i])
                        for i,x in enumerate(initial_readout)), "P0 serving differs from accepted original")
        guard()
        save_report()
        counts["student_fit_attempts"] += 1
        fitted = fit_inductive_gcn(cx,cy,mass,graph,(graph,val),testing=None,seed=seed,
                settings=SETTINGS,folder=folder,training_adjacency=None,stop=stopped,
                weighting="uniform",train_mask=train)
        counts["student_fits"] += 1
        counts["completed_training_epochs"] = SETTINGS["epochs"]
        report["GCN"] = fitted
        selected = folder/f"seed_{seed}_selected.pt"
        counts["route_replay_attempts"] += 1
        routes = replay_routes(selected,graph,route_h,{"val":val},SETTINGS,
                               folder/f"seed_{seed}_validation_routes_v1.json",seed=seed,stop=stopped)
        counts["route_replay_completed"] += 1
        counts["physical_serving_routes"] = 2
        report["sameweights_routes"] = routes
        require(all(math.isfinite(float(v)) for k,v in fitted.items()
                    if k.endswith(("_acc", "_ce"))) and
                all(math.isfinite(float(v)) for k,v in routes.items()
                    if k.endswith(("_acc", "_ce"))), "Nonfinite student diagnostics")
        require(not any("test_" in key for key in fitted) and not any("test_" in key for key in routes),
                "Validation-only evaluator produced test metrics")
        require(abs(routes["gcn_val_acc"]-fitted["val_acc"]) <= 1e-8 and
                math.isclose(routes["gcn_val_ce"],fitted["val_ce"],rel_tol=1e-6,abs_tol=1e-6),
                "Same selected GCN route differs from fitter")
        history = list(csv.DictReader((folder/f"seed_{seed}_epochs.csv").open()))
        require([int(row["epoch"]) for row in history] == list(range(10,301,10)), "Incomplete evaluation history")
        first_max = max(history,key=lambda row:float(row["val_acc"]))
        require(int(first_max["epoch"]) == fitted["epoch"] and
                float(first_max["val_acc"]) == fitted["val_acc"] and
                fitted["val_nodes"] == int(val.sum()), "First-max selection or validation split changed")
        cached = json.loads((folder/f"seed_{seed}.json").read_text())
        routed = json.loads((folder/f"seed_{seed}_validation_routes_v1.json").read_text())
        require(cached["recipe"]["test_enabled"] is False and routed["recipe"]["test_enabled"] is False and
                cached["recipe"]["settings"] == SETTINGS and cached["recipe"]["weighting"] == "uniform" and
                routed["recipe"]["source_fingerprint"] == cached["fingerprint"] and
                routed["recipe"]["epoch"] == fitted["epoch"], "Student/route provenance mismatch")
        correct = round(fitted["val_acc"]*fitted["val_nodes"]/100)
        require(abs(correct-fitted["val_acc"]*fitted["val_nodes"]/100) <= 1e-7, "Validation score off integer grid")
        report.update(status="complete",passed=True,GCN_validation_correct=correct,
                GCN_validation_nodes=fitted["val_nodes"],protocol=dict(path=str(Path(protocol_path).resolve()),sha256=protocol_sha256),
                evaluator_source=source,endpoint_producer_source=item["producer_source"],
                origin=protocol["references"]["origin"],endpoint=item.get("endpoint"),
                selected_weights=dict(path=str(selected),sha256=sha(selected)),
                selected_epoch=fitted["epoch"],cache_input_digest=cached["recipe"]["input_digest"])
        guard()
    except BaseException as exc:
        error = exc
        report.update(status="failed",passed=False,error=type(exc).__name__+": "+str(exc),
                      partial_student_epochs="unknown unless the fitter returned; existing output files retained")
    finally:
        changed = []
        for p, expected in pinned.items():
            try:
                if sha(p,bounded=False) != expected:
                    changed.append(p)
            except BaseException as exc:
                changed.append(p+": "+type(exc).__name__)
        report["input_bytes_unchanged"] = not changed
        report["input_changes"] = changed
        if changed:
            report.update(passed=False,status="failed")
            if error is None:
                error = ValueError("Immutable evaluation inputs changed")
        report["output_files"] = {str(p):dict(bytes=p.stat().st_size,sha256=sha(p,bounded=False))
                                  for p in folder.glob("*") if p.is_file() and p.name != "report.json"}
        final_memory = memory()
        if time.monotonic()-started > 300 or final_memory["peak_RSS_bytes"] > 16*1024**3 or \
                final_memory.get("peak_allocated_bytes",0) > 4*1024**3 or \
                final_memory.get("peak_reserved_bytes",0) > 6*1024**3:
            report.update(status="failed",passed=False,final_resource_or_time_limit_exceeded=True)
            if error is None:
                error = InterruptedError("Student including final integrity checks exceeded limits")
        save_report()
    if error is not None:
        if isinstance(error, (InterruptedError, TimeoutError)):
            raise RuntimeError("Terminal bounded student failure; retained exclusive outputs: "+str(error)) from error
        raise error
    return dict(report=report,report_path=str(folder/"report.json"),
                report_sha256=sha(folder/"report.json",bounded=False),validation_only=True)
