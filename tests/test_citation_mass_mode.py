import json

import pytest
import torch

import src.citation_search as search
from src.io import _fingerprint, save_state
from src.moments import initial_logits, make_material


@pytest.fixture
def screen(monkeypatch):
    x = torch.arange(24, dtype=torch.float32).reshape(8, 3) / 24
    graph = dict(x=x, y=torch.arange(8) % 2, adj=torch.eye(8).to_sparse_csr())
    train = torch.arange(8) < 2
    validation = graph, (torch.arange(8) >= 2) & (torch.arange(8) < 4)
    testing = graph, torch.arange(8) >= 4
    prepared = graph, train, validation, testing, x
    calls = []

    def teacher(*args):
        logits = torch.arange(16, dtype=torch.float32).reshape(8, 2) / 16
        save_state(dict(logits=logits, gamma=0.01), args[-1] / "teacher.pt")
        return logits, 0.01

    def optimize(z, q, assignment, **kwargs):
        calls.append(kwargs)
        p = initial_logits(assignment, 2).double().softmax(1)
        moments = p.T @ make_material(z, q) / len(z)
        folder = kwargs["folder"] / "checkpoints"
        folder.mkdir(exist_ok=True)
        for step in kwargs["checkpoint_steps"]:
            save_state(dict(step=step, moments=moments), folder / f"step_{step:06d}.pt")

    monkeypatch.setattr(search, "_prepare_dataset", lambda *args: prepared)
    monkeypatch.setattr(search, "teacher_logits", teacher)
    monkeypatch.setattr(search, "feature_kmeans", lambda *args: torch.arange(8) % 2)
    monkeypatch.setattr(search, "optimize_ce_assignment", optimize)
    monkeypatch.setattr(search, "fit_gcn_diagnostic", lambda *args, **kwargs: dict(epoch=1, val_acc=50.0))
    return calls


def candidate(method="low_rank", **changed):
    return dict(method=method, T=1.0, rank=2, penalty=0.1, **changed)


@pytest.mark.parametrize("method", ["low_rank", "mlp"])
def test_uniform_mass_mode_and_stop_are_forwarded_and_ranked(tmp_path, screen, method):
    stop = lambda: False
    option = candidate(method, mass_mode="uniform")
    ranking, _ = search.run_screen(
        "cora", 0.026, tmp_path, [option], steps=1, epochs=2, student_seeds=(0,), device="cpu", stop=stop
    )
    assert screen[0]["mass_mode"] == "uniform"
    assert screen[0]["stop"] is stop
    assert set(ranking["mass_mode"]) == {"uniform"}


@pytest.mark.parametrize("method", ["distance", "coarsening"])
def test_unsupported_assignments_reject_explicit_mass_mode(tmp_path, screen, method):
    with pytest.raises(ValueError, match="supported only"):
        search.run_screen(
            "cora",
            0.026,
            tmp_path,
            [candidate(method, mass_mode="free")],
            steps=1,
            epochs=2,
            student_seeds=(0,),
            device="cpu",
        )
    assert not screen


def test_omitted_mass_mode_preserves_existing_candidate_and_cache_key(tmp_path, screen):
    option = candidate()
    ranking, root = search.run_screen(
        "cora", 0.026, tmp_path, [option], steps=1, epochs=2, student_seeds=(0,), device="cpu"
    )
    expected = dict(method="low_rank", width=0, lr=0.01, T=1.0, rank=2, penalty=0.1)
    saved = json.loads((root / _fingerprint(expected) / "candidate.json").read_text())
    assert saved == expected
    assert screen[0]["mass_mode"] == "free"
    assert "mass_mode" not in ranking


def test_selected_final_keeps_uniform_mass_mode_in_candidate_and_recreated_screen(tmp_path, screen):
    ranking, root = search.run_screen(
        "cora",
        0.026,
        tmp_path,
        [candidate(mass_mode="uniform")],
        steps=1,
        epochs=2,
        student_seeds=(0,),
        device="cpu",
    )
    result = search.selected_test(
        root,
        ranking.iloc[0].to_dict(),
        condensation_seeds=(0,),
        student_seeds=(100,),
        epochs=2,
        device="cpu",
        report_routes=False,
    )
    assert not result.empty
    selected = json.loads((root / "selected.json").read_text())
    assert selected["candidate"]["mass_mode"] == "uniform"
    assert all(call["mass_mode"] == "uniform" for call in screen)


def test_invalid_mass_mode_is_rejected():
    with pytest.raises(ValueError, match="free or uniform"):
        search._assignment_mass_mode(dict(method="low_rank", mass_mode="unknown"))


def test_student_overrides_keep_condensate_and_separate_evaluation_recipe(tmp_path, screen, monkeypatch):
    seen = []

    def evaluate(*args, **kwargs):
        seen.append(kwargs)
        return dict(epoch=1, val_acc=50.0)

    monkeypatch.setattr(search, "fit_gcn_diagnostic", evaluate)
    defaults, root = search.run_screen("cora", .026, tmp_path, [candidate()], steps=1,
        epochs=2, student_seeds=(0,), device="cpu")
    overrides = dict(lr=.001, weight_decay=.001, eval_every=1,
                     lr_schedule="constant", initialization="geom_uniform")
    changed, changed_root = search.run_screen("cora", .026, tmp_path, [candidate()], steps=1,
        epochs=2, dropout=0., student_seeds=(0,), device="cpu", student_settings=overrides)
    assert root == changed_root
    assert set(defaults.candidate_path) == set(changed.candidate_path)
    assert set(defaults.student_recipe).isdisjoint(set(changed.student_recipe))
    assert seen[0]["lr"] == .01
    assert seen[-1]["lr_schedule"] == "constant"
    assert seen[-1]["initialization"] == "geom_uniform"
    assert seen[-1]["eval_every"] == 1

    search.selected_test(root, changed.iloc[0].to_dict(), condensation_seeds=(0,),
        student_seeds=(100,), epochs=2, dropout=0., device="cpu", report_routes=False,
        student_settings=overrides)
    assert seen[-1]["lr"] == .001
    assert seen[-1]["lr_schedule"] == "constant"
    assert seen[-1]["initialization"] == "geom_uniform"
    selected = json.loads((root / "selected.json").read_text())
    assert selected["student"]["overrides"] == overrides


def test_explicit_student_defaults_preserve_historical_recipe():
    default = search._student_settings(.9, 1000, None)
    assert search._student_settings(.9, 1000, dict(lr_schedule="half_reset", initialization="pyg")) == default
    with pytest.raises(ValueError, match="Unknown student setting"):
        search._student_settings(.9, 1000, dict(epochs=600))
