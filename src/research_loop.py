"""Run a saved queue of bounded research jobs on one local GPU.

The worker executes a concrete plan; the thread heartbeat reviews its results
and appends the next plan. Empty queues mean awaiting analysis, never SOTA success.
"""
import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import time
import traceback
from pathlib import Path

from src.io import _fingerprint, save_json


def job_key(job):
    return _fingerprint({key: value for key, value in job.items() if key != "id"})


def implementation_provenance():
    repo = Path(__file__).resolve().parents[1]
    files = {str(path.relative_to(repo)): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in sorted((repo / "src").glob("*.py"))}
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                              capture_output=True, text=True, check=False)
    return dict(git_head=revision.stdout.strip(), source_digest=_fingerprint(files), files=files)


def dispatch(job, stop):
    options = dict(job["options"])
    if job["kind"] == "citation_gradient_fixed25_prepare":
        from src.citation_gradient_fixed25 import prepare
        return prepare(**options, stop=stop)
    if job["kind"] == "citation_gradient_fixed25_certify":
        from src.citation_gradient_fixed25 import certify
        return certify(**options, stop=stop)
    if job["kind"] == "citation_gradient_fixed25_validate":
        from src.citation_gradient_fixed25 import validate
        return validate(**options, stop=stop)
    if job["kind"] == "citation_gradient_probe":
        from src.citation_gradient_probe import prepare_probe
        return prepare_probe(**options, stop=stop)
    if job["kind"] == "citation_macro_fixed25_prepare":
        from src.citation_macro_fixed25 import prepare
        return prepare(**options, stop=stop)
    if job["kind"] == "citation_macro_fixed25_certify":
        from src.citation_macro_fixed25 import certify
        return certify(**options, stop=stop)
    if job["kind"] == "citation_macro_fixed25_validate":
        from src.citation_macro_fixed25 import validate
        return validate(**options, stop=stop)
    if job["kind"] == "citation_macro_teacher_probe":
        from src.citation_macro_probe import prepare_probe
        return prepare_probe(**options, stop=stop)
    if job["kind"] == "citeseer_finite_student_v2_prepare":
        from src.citeseer_finite_student_v2 import prepare
        return prepare(**options, stop=stop)
    if job["kind"] == "citeseer_finite_student_v2_certify":
        from src.citeseer_finite_student_v2 import certify
        return certify(**options, stop=stop)
    if job["kind"] == "citeseer_finite_student_v2_validate":
        from src.citeseer_finite_student_v2 import validate
        return validate(**options, stop=stop)
    if job["kind"] == "citeseer_finite_student_prepare":
        from src.citeseer_finite_student import prepare
        return prepare(**options, stop=stop)
    if job["kind"] == "citeseer_finite_student_certify":
        from src.citeseer_finite_student import certify
        return certify(**options, stop=stop)
    if job["kind"] == "citeseer_finite_student_validate":
        from src.citeseer_finite_student import validate
        return validate(**options, stop=stop)
    if job["kind"] == "citation_source_certificate":
        from src.citation_source_certificate import prepare_certificate
        return prepare_certificate(**options, stop=stop)
    if job["kind"] == "citation_source_provenance":
        from src.citation_source_provenance import prepare_provenance
        return prepare_provenance(**options, stop=stop)
    if job["kind"] == "citation_source_preflight":
        from src.citation_source_preflight import prepare_preflight
        return prepare_preflight(**options, stop=stop)
    if job["kind"] == "citation_screen":
        from src.citation_search import run_screen
        ranking, root = run_screen(**options, stop=stop)
        return dict(root=str(root.resolve()), ranking=ranking.to_dict("records"),
                    validation_only=True)
    if job["kind"] == "citation_finite_student_full25":
        from src.finite_student_experiment import prepare_full25
        return prepare_full25(**options, stop=stop)
    if job["kind"] == "citation_finite_student_validation":
        from src.finite_student_experiment import evaluate_cached
        return evaluate_cached(**options, stop=stop)
    if job["kind"] == "citation_finite_student_probe":
        from src.finite_student_probe import prepare_probe
        return prepare_probe(**options, stop=stop)
    if job["kind"] == "citation_source_linear_probe":
        from src.source_linear_assignment import prepare_probe
        return prepare_probe(**options, stop=stop)
    if job["kind"] == "citation_mlp_centered_probe":
        from src.mlp_source_centering import prepare_probe
        return prepare_probe(**job["options"], stop=stop)
    if job["kind"] == "citation_mlp_initial_probe":
        from src.mlp_initial_mass import prepare_probe
        return prepare_probe(**options, stop=stop)
    if job["kind"] == "citation_gcn_teacher":
        from src.citation_gcn_teacher import prepare_job
        return prepare_job(**options, stop=stop)
    if job["kind"] == "citation_teacher_gamma":
        from src.ntk_teacher import prepare_gamma_job
        return prepare_gamma_job(**options, stop=stop)
    if job["kind"] == "citation_test":
        from src.citation_search import selected_test
        result = selected_test(**options, stop=stop)
        return dict(rows=result.to_dict("records"),
                    test_mean=float(result.test_acc.mean()),
                    by_condensation=result.groupby("condensation_seed").test_acc.mean().to_dict())
    if job["kind"] == "large_pilot":
        from src.large_pilot import run_pilot
        result = run_pilot(**options, stop=stop)
        if isinstance(result, tuple):
            report, root = result
            return dict(report=report, root=str(Path(root).resolve()), validation_only=True)
        return result
    if job["kind"] == "large_final":
        from src.large_final import run_final
        report, root = run_final(**options, stop=stop)
        return dict(report=report, root=str(Path(root).resolve()),
                    validation_only=False, fixed_configuration_test=True)
    if job["kind"] == "large_quotient":
        from src.large_quotient_pilot import run_quotient_pilot
        report, root = run_quotient_pilot(**options, stop=stop)
        return dict(report=report, root=str(Path(root).resolve()), validation_only=True)
    raise ValueError(f"Unknown research job kind: {job['kind']}")


def run_plan(path, max_seconds=1800, dispatch_fn=dispatch):
    path = Path(path).resolve()
    root = path.parent
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / "worker.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return dict(phase="already_running")
    status_path = root / "status.json"
    ledger_path = root / "jobs.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
    implementation = implementation_provenance()
    save_json(implementation, root / f"implementation_{implementation['source_digest']}.json")
    started = time.monotonic()
    interrupted = False
    previous_handlers = {}

    def interrupt(signum, frame):
        nonlocal interrupted
        interrupted = True

    def alarm(signum, frame):
        raise InterruptedError("Bounded job reached its wall-clock limit")

    for sig, handler in ((signal.SIGTERM, interrupt), (signal.SIGINT, interrupt), (signal.SIGALRM, alarm)):
        previous_handlers[sig] = signal.signal(sig, handler)

    def stop():
        return interrupted or (root / "STOP").exists() or time.monotonic() - started >= max_seconds

    def status(phase, **details):
        value = dict(phase=phase, pid=os.getpid(), updated=time.time(),
                     queue_is_goal_completion=False, **details)
        save_json(value, status_path)
        print(json.dumps(value), flush=True)
        return value

    try:
        status("running")
        while not stop():
            plan = json.loads(path.read_text())
            jobs = plan["jobs"]
            ids = [job.get("id", job_key(job)) for job in jobs]
            if len(set(ids)) != len(ids):
                raise ValueError("Plan contains duplicate job IDs")
            pending = None
            for key, job in zip(ids, jobs):
                fingerprint = job_key(job)
                saved = ledger.get(key)
                if saved is not None and saved["fingerprint"] != fingerprint:
                    raise ValueError("A queued job was changed in place; use a fresh ID")
                if saved is None or saved["phase"] in ("running", "interrupted"):
                    if saved is None or saved.get("attempts", 0) < 3:
                        pending = (key, job, fingerprint)
                        break
            if pending is None:
                return status("awaiting_next_plan", completed=sum(row["phase"] == "completed" for row in ledger.values()))
            key, job, fingerprint = pending
            previous = ledger.get(key, {})
            record = dict(fingerprint=fingerprint, job=job, phase="running", started=time.time(),
                          attempts=previous.get("attempts", 0) + 1,
                          implementation=dict(git_head=implementation["git_head"],
                                              source_digest=implementation["source_digest"]))
            ledger[key] = record
            save_json(ledger, ledger_path)
            status("running_job", job_id=key, kind=job["kind"], options=job["options"])
            budget = min(float(job.get("max_seconds", 300)), max_seconds - (time.monotonic() - started))
            if budget <= 0:
                break
            signal.setitimer(signal.ITIMER_REAL, budget)
            try:
                record["result"] = dispatch_fn(job, stop)
                record["phase"] = "completed"
                if job["kind"] in ("large_pilot", "large_final", "large_quotient"):
                    report = record["result"].get("report", record["result"])
                    if report.get("status") != "complete":
                        record["phase"] = "pilot_stopped" if report.get("status") == "stopped" else "failed"
                        record["error"] = report.get("reason", "Pilot did not complete")
            except InterruptedError as error:
                record.update(phase="interrupted", error=str(error))
            except Exception as error:
                record.update(phase="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
                record["seconds"] = time.time() - record["started"]
                record["finished"] = time.time()
                save_json(ledger, ledger_path)
                status("job_finished", job_id=key, result_phase=record["phase"], seconds=record["seconds"])
                # Let memory from one graph go before loading the next dataset.
                import gc
                gc.collect()
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        return status("stopped", reason="stop file, signal or invocation time limit")
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        lock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("plan")
    parser.add_argument("--max-seconds", type=int, default=1800)
    args = parser.parse_args()
    run_plan(args.plan, args.max_seconds)
