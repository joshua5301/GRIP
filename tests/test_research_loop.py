import fcntl
import json

import pytest

from src.io import save_json
from src.research_loop import run_plan


def prepare(tmp_path, jobs):
    path = tmp_path / "plan.json"
    save_json(dict(jobs=jobs), path)
    return path


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)


def job(identity="first"):
    return dict(id=identity, kind="citation_screen", options=dict(dataset="cora"), max_seconds=10)


def test_completed_jobs_are_not_repeated_and_queue_is_not_goal_completion(tmp_path):
    path = prepare(tmp_path, [job()])
    calls = []
    dispatch = lambda spec, stop: calls.append(spec["id"]) or dict(validation_only=True)
    first = run_plan(path, dispatch_fn=dispatch)
    second = run_plan(path, dispatch_fn=dispatch)
    assert calls == ["first"]
    assert first["phase"] == second["phase"] == "awaiting_next_plan"
    assert first["queue_is_goal_completion"] is False


def test_rejects_mutation_of_an_existing_job(tmp_path):
    path = prepare(tmp_path, [job()])
    run_plan(path, dispatch_fn=lambda spec, stop: {})
    changed = job()
    changed["options"]["dataset"] = "citeseer"
    prepare(tmp_path, [changed])
    with pytest.raises(ValueError, match="changed in place"):
        run_plan(path, dispatch_fn=lambda spec, stop: {})


def test_interrupted_job_is_bounded_to_three_attempts_then_next_job_runs(tmp_path):
    path = prepare(tmp_path, [job(), job("second")])
    calls = []

    def dispatch(spec, stop):
        calls.append(spec["id"])
        if spec["id"] == "first":
            raise InterruptedError("budget")
        return {}

    run_plan(path, dispatch_fn=dispatch)
    assert calls == ["first", "first", "first", "second"]
    ledger = json.loads((tmp_path / "jobs.json").read_text())
    assert ledger["first"]["phase"] == "interrupted"
    assert ledger["first"]["attempts"] == 3


def test_stop_file_prevents_dispatch_and_worker_lock_prevents_second_worker(tmp_path):
    path = prepare(tmp_path, [job()])
    lock = (tmp_path / "worker.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert run_plan(path)["phase"] == "already_running"
    lock.close()
    (tmp_path / "STOP").touch()
    assert run_plan(path, dispatch_fn=lambda *_: pytest.fail("must not launch"))["phase"] == "stopped"


def test_pilot_deadline_is_recorded_as_incomplete_without_blind_retries(tmp_path):
    spec = dict(job(), kind="large_pilot")
    path = prepare(tmp_path, [spec])
    calls = []

    def dispatch(spec, stop):
        calls.append(spec["id"])
        return dict(report=dict(status="stopped", reason="teacher cost limit"))

    result = run_plan(path, dispatch_fn=dispatch)
    saved = json.loads((tmp_path / "jobs.json").read_text())["first"]
    assert calls == ["first"]
    assert saved["phase"] == "pilot_stopped"
    assert result["completed"] == 0
