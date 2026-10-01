"""GCN teacher isolation without touching the legacy kernel/geometry identities."""
import json

import pytest
import test_teacher_gamma as gamma
import torch

import src.large_pilot as pilot
from src.io import save_state

options, pipeline = gamma.options, gamma.pipeline


@pytest.fixture
def gcn_pipeline(pipeline, monkeypatch):
    pipeline["gcn_teachers"] = []

    def fit_teacher(graph, train_mask, validation, folder, *, epochs, seed, guard):
        assert torch.equal(graph["y"][train_mask], torch.tensor([0, 1] * 3))
        val_graph, val_mask = validation
        assert int(val_mask.sum()) == 4
        assert torch.equal(val_graph["y"][val_mask], torch.tensor([0, 1] * 2))
        assert len(graph["x"]) == 12
        guard()
        path = folder / "teacher.pt"
        if path.exists():
            return torch.load(path, weights_only=False)
        pipeline["gcn_teachers"].append(dict(folder=folder, epochs=epochs, seed=seed,
                                              graph=graph, validation=validation, train_mask=train_mask))
        logits = torch.stack((graph["x"][:, 0], -graph["x"][:, 0]), dim=1)
        fitted = dict(training_complete=True, logits=logits,
                      selected_validation=dict(epoch=epochs, val_acc=72.0, val_ce=0.4, val_nodes=4),
                      timings=dict(training_seconds=0.1, validation_seconds=0.1))
        save_state(fitted, path)
        return fitted

    monkeypatch.setattr(pilot, "fit_gcn_teacher", fit_teacher)
    return pipeline


def test_gcn_teacher_reuses_legacy_geometry_and_has_distinct_full_budget_roots(tmp_path, gcn_pipeline):
    baseline, root = pilot.run_pilot(**options(tmp_path), report_routes=True)
    assert baseline["status"] == "complete"
    geometry = root.parent
    leaves = ("propagated_H.pt", "feature_map.pt", "phi.npy", "validation_routes_S2X_v1.pt")
    frozen = {leaf: (geometry / leaf).read_bytes() for leaf in leaves}
    frozen_kernel = {leaf: (geometry / leaf).read_bytes() for leaf in ("teacher.pt", "protocol.json")}
    roots = []
    for epochs in (10, 200):
        result, new = pilot.run_pilot(**options(tmp_path), report_routes=True,
                                       teacher_type="gcn", gcn_teacher_epochs=epochs)
        assert result["status"] == "complete"
        assert new.parent.parent == geometry and new != root
        assert result["teacher_type"] == "gcn"
        assert result["gcn_teacher_epochs"] == epochs
        assert result["geometry_cache_root"] == str(geometry.resolve())
        assert result["teacher"]["training_complete"] is True
        assert result["teacher"]["epoch"] == epochs
        assert "converged" not in result["teacher"]
        assert "teacher_gamma" not in result
        protocol = json.loads((new.parent / "protocol.json").read_text())
        assert protocol["teacher_type"] == "gcn"
        assert protocol["gcn_settings"] == pilot.teacher_settings(epochs, 0)
        assert not any((new.parent / leaf).exists() for leaf in leaves)
        assert result["candidate"] == baseline["candidate"]
        assert result["selection"] == "validation_only"
        assert not any("test" in key for key in result)
        assert not any("test" in key for student in result["students"] for key in student)
        roots.append(new)
    assert roots[0] != roots[1]
    assert [row["epochs"] for row in gcn_pipeline["gcn_teachers"]] == [10, 200]
    assert gcn_pipeline["teachers"] == [0.01]
    assert gcn_pipeline["optimizers"] == 3
    assert gcn_pipeline["map"] == 1 and len(gcn_pipeline["phi_created"]) == 1
    for leaf, value in {**frozen, **frozen_kernel}.items():
        assert (geometry / leaf).read_bytes() == value
    report, same = pilot.run_pilot(**options(tmp_path), teacher_type="gcn", gcn_teacher_epochs=10,
                                   report_routes=True)
    assert report["status"] == "complete" and same == roots[0]
    assert len(gcn_pipeline["gcn_teachers"]) == 2
    assert gcn_pipeline["optimizers"] == 3


def test_gcn_first_does_not_create_fake_kernel_teacher_and_kernel_reuses_geometry(tmp_path, gcn_pipeline):
    result, root = pilot.run_pilot(**options(tmp_path), teacher_type="gcn", gcn_teacher_epochs=10)
    assert result["status"] == "complete"
    geometry = root.parent.parent
    assert not (geometry / "teacher.pt").exists()
    assert not (geometry / "protocol.json").exists()
    phi = (geometry / "phi.npy").read_bytes()
    kernel, old = pilot.run_pilot(**options(tmp_path))
    explicit, same = pilot.run_pilot(**options(tmp_path), teacher_type="kernel")
    assert kernel["status"] == explicit["status"] == "complete"
    assert old == same and old.parent == geometry
    assert kernel["candidate"] == explicit["candidate"]
    assert "teacher_type" not in kernel and "gcn_teacher_epochs" not in kernel
    assert (geometry / "phi.npy").read_bytes() == phi
    assert gcn_pipeline["teachers"] == [0.01]
    assert gcn_pipeline["map"] == 1 and len(gcn_pipeline["phi_created"]) == 1


def test_stopped_teacher_does_not_initialize_or_condense_partial_logits(tmp_path, gcn_pipeline, monkeypatch):
    def stopped(*args, **kwargs):
        folder = args[3]
        save_state(dict(epoch=1), folder / "teacher_resume.pt")
        raise InterruptedError("Teacher interrupted after saved epoch")

    monkeypatch.setattr(pilot, "fit_gcn_teacher", stopped)
    monkeypatch.setattr(pilot, "feature_kmeans", lambda *a, **k: pytest.fail("Incomplete teacher initialized P"))
    result, root = pilot.run_pilot(**options(tmp_path), teacher_type="gcn", gcn_teacher_epochs=10)
    assert result["status"] == "stopped" and result["stage"] == "fitting_teacher"
    assert "saved epoch" in result["reason"]
    assert (root.parent / "teacher_resume.pt").exists()
    assert not (root.parent / "teacher.pt").exists()
    assert not (root / "initial_assignment.pt").exists()
    assert not (root / "condensation").exists()
    assert gcn_pipeline["optimizers"] == 0 and not gcn_pipeline["students"]


def test_teacher_failure_is_recorded_without_condensation(tmp_path, gcn_pipeline, monkeypatch):
    def failed(*args, **kwargs):
        raise ValueError("GCN teacher cache differs from requested fit")

    monkeypatch.setattr(pilot, "fit_gcn_teacher", failed)
    result, root = pilot.run_pilot(**options(tmp_path), teacher_type="gcn", gcn_teacher_epochs=10)
    assert result["status"] == "failed" and result["error_type"] == "ValueError"
    assert result["stage"] == "fitting_teacher"
    assert not (root / "condensation").exists()
    assert gcn_pipeline["optimizers"] == 0 and not gcn_pipeline["students"]


@pytest.mark.parametrize("change,match", [
    (dict(teacher_type="gcn", teacher_gamma=0.001), "leave teacher_gamma"),
    (dict(teacher_type="kernel", gcn_teacher_epochs=10), "only applies"),
    (dict(teacher_type="gcn", gcn_teacher_epochs=0), "positive integer"),
    (dict(teacher_type="gcn", gcn_teacher_epochs=True), "positive integer"),
    (dict(teacher_type="gcn", gcn_teacher_epochs=1.5), "positive integer"),
    (dict(teacher_type="unknown"), "teacher_type must"),
])
def test_invalid_teacher_route_combination_fails_before_data_or_cache(tmp_path, gcn_pipeline, change, match):
    with pytest.raises(ValueError, match=match):
        pilot.run_pilot(**options(tmp_path), **change)
    assert gcn_pipeline["data"] == 0
    assert not list(tmp_path.iterdir())
