"""Teacher gamma state isolation and legacy/shared-geometry cache regression."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import src.large_pilot as pilot
from src.io import save_state
from src.moments import make_material


@pytest.fixture
def pipeline(monkeypatch):
    calls = dict(data=0, map=0, phi_created=[], teachers=[], optimizers=0, students=[])
    x = torch.arange(24, dtype=torch.float32).reshape(12, 2) / 24 + 0.1
    graph = dict(x=x, y=torch.tensor([0, 1] * 5 + [999, 999]), adj=torch.eye(12).to_sparse_csr())
    train = torch.arange(12) < 6
    val = (torch.arange(12) >= 6) & (torch.arange(12) < 10)

    def dataset(*args):
        calls["data"] += 1
        return graph, train, (graph, val), object(), x

    class FeatureMap:
        def __init__(self, anchors, mapping, kernel="linear"):
            self.anchors, self.mapping, self.kernel = anchors, mapping, kernel

        @classmethod
        def fit(cls, h, basis, seed):
            calls["map"] += 1
            return cls(h[:2], torch.eye(2, dtype=torch.double))

        def __call__(self, h):
            return h.double()

    def features(h, feature_map, path, **kwargs):
        if not path.exists():
            calls["phi_created"].append(path)
            np.save(path, feature_map(h).numpy())
        return np.load(path, mmap_mode="r")

    def teacher(phi, labels, mask, gamma, **kwargs):
        assert torch.equal(mask, train)
        assert bool((labels[~train] == 0).all())
        calls["teachers"].append(gamma)
        weight = torch.eye(2, dtype=torch.double) * (0.01 / gamma)
        return torch.from_numpy(np.array(phi[:])) @ weight, weight

    def optimizer(z, q, assignment, folder, steps, **kwargs):
        calls["optimizers"] += 1
        p = torch.nn.functional.one_hot(assignment, 2).double()
        moments = p.T @ make_material(z, q) / len(z)
        path = folder / "checkpoints" / f"step_{steps:06d}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        save_state(dict(step=steps, moments=moments, J_exact=True, teacher_ce=0.5), path)
        save_state(dict(step=steps), folder / "resume.pt")

    def student(*args, **kwargs):
        assert "testing" not in kwargs
        assert kwargs["weighting"] == "uniform"
        calls["students"].append(kwargs)
        return dict(seed=kwargs["seed"], epoch=10, val_acc=61.0, val_ce=0.5)

    def routes(*args, **kwargs):
        assert set(args[3]) == {"val"}
        return dict(selection="same weights at GCN validation-selected epoch", gcn_val_acc=61.0,
                    gcn_val_ce=0.5, mlp_val_acc=60.0, mlp_val_ce=0.6)

    for module in (pilot,):
        monkeypatch.setattr(module, "_prepare_dataset", dataset)
        monkeypatch.setattr(module, "NystromMap", FeatureMap)
        monkeypatch.setattr(module, "cache_features", features)
        monkeypatch.setattr(module, "fit_streaming_teacher", teacher)
        monkeypatch.setattr(module, "optimize_ce_assignment", optimizer)
        monkeypatch.setattr(module, "fit_inductive_gcn", student)
        monkeypatch.setattr(module, "replay_routes", routes)
        monkeypatch.setattr(module, "feature_kmeans", lambda h, cells, seed: torch.arange(len(h)) % cells)
        monkeypatch.setitem(module.BUDGET, ("flickr", 0.001), 2)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *args: pytest.fail("GPU accessed"))
    return calls


def options(tmp_path):
    return dict(dataset="flickr", ratio=0.001, output_dir=tmp_path, device="cpu", basis=2, rank=2,
                steps=0, epochs=10)


def test_default_preserves_all_legacy_roots_protocols_and_endpoints(tmp_path, pipeline):
    old, root = pilot.run_pilot(**options(tmp_path))
    legacy_teacher_protocol = dict(
        version=2, dataset="flickr", data_dir=str(Path("data").resolve()), basis=2,
        teacher_seed=0, kernel="relu", gamma=0.01,
        teacher_feature_route="resident training rows with memory-checked streaming fallback",
        chunk=2048, torch_version=str(torch.__version__),
    )
    expected_teacher_root = tmp_path / "flickr" / f"teacher_{pilot._fingerprint(legacy_teacher_protocol)}"
    assert root.parent == expected_teacher_root
    saved_protocol = json.loads((root.parent / "protocol.json").read_text())
    assert {k: v for k, v in saved_protocol.items() if k != "data_digest"} == legacy_teacher_protocol
    legacy_candidate = dict(
        version=1, ratio=0.001, cells=2, temperature=0.3, rank=2, penalty=1e-4, lr=0.01,
        condensation_seed=0, initialization="feature", alpha=1.0, assignment="low_rank",
        inner_loss="exact_mass_ce", student_loss="uniform_ce", synthetic_adjacency="identity",
    )
    assert old["candidate"] == legacy_candidate
    assert root.name == f"candidate_{pilot._fingerprint(legacy_candidate)}"
    protocol = (root.parent / "protocol.json").read_bytes()
    teacher = (root.parent / "teacher.pt").read_bytes()
    candidate = (root / "protocol.json").read_bytes()
    new, same = pilot.run_pilot(**options(tmp_path))
    explicit, again = pilot.run_pilot(**options(tmp_path), teacher_gamma=0.01)
    assert old["status"] == new["status"] == explicit["status"] == "complete"
    assert root == same == again
    assert (root.parent / "protocol.json").read_bytes() == protocol
    assert (root.parent / "teacher.pt").read_bytes() == teacher
    assert (root / "protocol.json").read_bytes() == candidate
    assert old["candidate"] == new["candidate"] == explicit["candidate"]
    assert "teacher_gamma" not in new and "geometry_cache_root" not in new
    assert len(pipeline["teachers"]) == pipeline["optimizers"] == pipeline["map"] == 1
    assert len(pipeline["phi_created"]) == 1
    assert len({s["folder"] for s in pipeline["students"]}) == 1
    assert (root / f"report_step_0_{pilot._fingerprint(new['student_recipe'])}.json").exists()


def test_nondefault_teacher_states_are_separate_but_all_geometry_is_reused(tmp_path, pipeline):
    baseline, root = pilot.run_pilot(**options(tmp_path), report_routes=True)
    geometry = root.parent
    leaves = ["propagated_H.pt", "feature_map.pt", "phi.npy", "validation_routes_S2X_v1.pt"]
    frozen = {leaf: (geometry / leaf).read_bytes() for leaf in leaves}
    roots = [root]
    for gamma in (0.001, 0.1):
        result, new = pilot.run_pilot(**options(tmp_path), teacher_gamma=gamma, report_routes=True)
        assert result["status"] == "complete"
        assert new.parent.parent == geometry
        assert new != root
        assert result["geometry_cache_root"] == str(geometry.resolve())
        assert result["teacher_gamma"] == gamma
        assert json.loads((new.parent / "protocol.json").read_text())["gamma"] == gamma
        assert torch.load(new.parent / "teacher.pt", weights_only=False)["gamma"] == gamma
        assert not any((new.parent / leaf).exists() for leaf in leaves)
        roots.append(new)
    assert len(set(roots)) == 3
    assert pipeline["teachers"] == [0.01, 0.001, 0.1]
    assert pipeline["map"] == 1 and len(pipeline["phi_created"]) == 1
    assert pipeline["optimizers"] == 3
    for leaf in leaves:
        assert (geometry / leaf).read_bytes() == frozen[leaf]
    repeated, same = pilot.run_pilot(**options(tmp_path), teacher_gamma=0.001, report_routes=True)
    assert repeated["status"] == "complete" and same == roots[1]
    assert pipeline["teachers"] == [0.01, 0.001, 0.1]
    assert pipeline["optimizers"] == 3


def test_gamma_first_does_not_invent_a_default_teacher_and_default_can_reuse_geometry(tmp_path, pipeline):
    first, root = pilot.run_pilot(**options(tmp_path), teacher_gamma=0.003)
    assert first["status"] == "complete"
    geometry = root.parent.parent
    assert not (geometry / "teacher.pt").exists()
    assert not (geometry / "protocol.json").exists()
    phi = (geometry / "phi.npy").read_bytes()
    default, old_root = pilot.run_pilot(**options(tmp_path))
    assert default["status"] == "complete" and old_root.parent == geometry
    assert (geometry / "phi.npy").read_bytes() == phi
    assert pipeline["map"] == 1 and len(pipeline["phi_created"]) == 1
    assert pipeline["teachers"] == [0.003, 0.01]


def test_loaded_teacher_gamma_mismatch_is_refused_without_overwrite(tmp_path, pipeline):
    _, root = pilot.run_pilot(**options(tmp_path), teacher_gamma=0.003)
    path = root.parent / "teacher.pt"
    bad = torch.load(path, weights_only=False)
    bad["gamma"] = 0.001
    save_state(bad, path)
    before = path.read_bytes()
    result, same = pilot.run_pilot(**options(tmp_path), teacher_gamma=0.003)
    assert result["status"] == "failed" and same == root
    assert "Teacher cache" in result["reason"]
    assert path.read_bytes() == before
    assert pipeline["teachers"] == [0.003]


@pytest.mark.parametrize("gamma", [0, -0.001, float("nan"), float("inf"), True, "0.003"])
def test_invalid_gamma_rejected_before_data_or_cache_creation(tmp_path, pipeline, gamma):
    with pytest.raises(ValueError, match="teacher_gamma must be positive and finite"):
        pilot.run_pilot(**options(tmp_path), teacher_gamma=gamma)
    assert pipeline["data"] == 0
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("change", [dict(basis=3), dict(teacher_seed=1)])
def test_geometry_request_change_gets_distinct_cache(tmp_path, pipeline, change):
    _, root = pilot.run_pilot(**options(tmp_path), teacher_gamma=0.003)
    changed = dict(options(tmp_path), **change)
    result, new = pilot.run_pilot(**changed, teacher_gamma=0.003)
    assert result["status"] == "complete"
    assert root.parent.parent != new.parent.parent
    assert pipeline["map"] == 2 and len(pipeline["phi_created"]) == 2


def test_gamma_and_nondefault_mixing_have_distinct_candidates_but_share_teacher_geometry(tmp_path, pipeline):
    baseline, root = pilot.run_pilot(**options(tmp_path), teacher_gamma=0.001)
    changed, new = pilot.run_pilot(**options(tmp_path), teacher_gamma=0.001, mixing=0.005)
    assert baseline["status"] == changed["status"] == "complete"
    assert root != new and root.parent == new.parent
    assert "mixing" not in baseline["candidate"]
    assert changed["candidate"]["mixing"] == 0.005
    assert pipeline["teachers"] == [0.001]
    assert pipeline["map"] == 1 and len(pipeline["phi_created"]) == 1
