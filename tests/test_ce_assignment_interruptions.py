from pathlib import Path

import pytest
import torch

import src.soft_ce_partition as model


def problem():
    generator = torch.Generator().manual_seed(619)
    z = torch.randn(18, 3, generator=generator, dtype=torch.double)
    q = torch.randn(18, 3, generator=generator, dtype=torch.double).softmax(1)
    return z, q, torch.arange(18) % 4


def settings():
    return dict(penalty=0.2, lr=0.02, chunk_size=4, assignment_rank=3,
                inner_method="newton_first", inner_tol=1e-9, cg_rtol=1e-9,
                checkpoint_steps=(0,), save_resume=True, save_assignment=False)


def test_stop_before_work_does_not_call_solver_or_create_output(tmp_path, monkeypatch):
    monkeypatch.setattr(model, "solve_inner_newton_first", lambda *args, **kwargs: pytest.fail("solver ran"))
    folder = tmp_path / "absent"
    with pytest.raises(InterruptedError, match="optimization interrupted"):
        model.optimize_ce_assignment(*problem(), steps=2, folder=folder, stop=lambda: True, **settings())
    assert not folder.exists()


@pytest.mark.parametrize("stage", ["inner", "adjoint"])
def test_solver_stop_preserves_resume_and_resumes_same_numerical_trajectory(tmp_path, stage):
    values, options = problem(), settings()
    full = model.optimize_ce_assignment(*values, steps=3, folder=tmp_path / "full", **options)
    folder = tmp_path / "interrupted"
    model.optimize_ce_assignment(*values, steps=1, folder=folder, **options)
    resume_path = folder / "resume.pt"
    previous_bytes = resume_path.read_bytes()
    state = torch.load(resume_path, weights_only=False)
    assert "stop" not in state["config"]
    stopped = False
    calls = []

    def inner(*args, **kwargs):
        nonlocal stopped
        result = model.solve_inner_newton_first(*args, **kwargs)
        calls.append("inner")
        if stage == "inner":
            stopped = True
        return result

    def adjoint(*args, **kwargs):
        nonlocal stopped
        result = model.solve_head_system(*args, **kwargs)
        calls.append("adjoint")
        stopped = True
        return result

    with pytest.raises(InterruptedError, match="optimization interrupted"):
        model.optimize_ce_assignment(*values, steps=3, folder=folder, resume_state=state,
                                     inner_solver=inner, implicit_solver=adjoint,
                                     stop=lambda: stopped, **options)
    assert calls == (["inner"] if stage == "inner" else ["inner", "adjoint"])
    assert resume_path.read_bytes() == previous_bytes
    resumed = model.optimize_ce_assignment(*values, steps=3, folder=folder, resume_state=state,
                                          stop=lambda: False, **options)
    torch.testing.assert_close(resumed["checkpoints"][3]["moments"], full["checkpoints"][3]["moments"],
                               atol=1e-11, rtol=1e-9)
    torch.testing.assert_close(resumed["checkpoints"][3]["theta"], full["checkpoints"][3]["theta"],
                               atol=1e-8, rtol=1e-6)
    assert [row["step"] for row in resumed["history"]] == [0, 1, 2, 3]


@pytest.mark.parametrize("previous_checkpoint", [False, True])
def test_interrupted_checkpoint_write_never_exposes_partial_endpoint(tmp_path, monkeypatch, previous_checkpoint):
    values, options = problem(), settings()
    endpoint = tmp_path / "checkpoints" / "step_000000.pt"
    original = None
    if previous_checkpoint:
        model.optimize_ce_assignment(*values, steps=0, folder=tmp_path, **options)
        original = endpoint.read_bytes()
    save = torch.save
    destinations = []

    def interrupted_save(value, path, *args, **kwargs):
        path = Path(path)
        destinations.append(path.name)
        if path.name == ".step_000000.pt.tmp":
            path.write_bytes(b"partial serialization")
            raise InterruptedError("alarm during checkpoint serialization")
        return save(value, path, *args, **kwargs)

    monkeypatch.setattr(torch, "save", interrupted_save)
    with pytest.raises(InterruptedError, match="alarm during checkpoint serialization"):
        model.optimize_ce_assignment(*values, steps=0, folder=tmp_path, **options)
    assert destinations == [".step_000000.pt.tmp"]
    assert not (endpoint.parent / ".step_000000.pt.tmp").exists()
    if previous_checkpoint:
        assert endpoint.read_bytes() == original
        assert set(model.load_ce_snapshots(tmp_path)) == {0}
    else:
        assert not endpoint.exists()
    monkeypatch.setattr(torch, "save", save)
    model.optimize_ce_assignment(*values, steps=0, folder=tmp_path, **options)
    assert set(model.load_ce_snapshots(tmp_path)) == {0}
