import json
import time

import numpy as np
import pytest
import torch

import src.large_pilot as pilot
from src.io import save_state
from src.moments import make_material
from src.partition_initialization import allocate_balanced_cells, teacher_balanced_kmeans


@pytest.fixture
def mocked_pipeline(monkeypatch):
    calls = dict(teacher=0, optimizer=0, nystrom=0, students=[], data=0)
    x = torch.arange(54, dtype=torch.float32).reshape(18, 3) / 54 + 0.1
    y = torch.arange(18) % 2
    y[15:] = 9  # Deliberately outside the train/validation class vocabulary.
    graph = dict(x=x, y=y, adj=torch.eye(18).to_sparse_csr())
    train = torch.arange(18) < 12
    mask = (torch.arange(18) >= 12) & (torch.arange(18) < 15)
    validation = graph, mask
    calls.update(graph=graph, validation=validation)

    def dataset(name, data_dir, device):
        calls["data"] += 1
        assert device == "cpu"
        val = (
            validation
            if name == "arxiv"
            else (dict(x=x[12:15], y=y[12:15], adj=torch.eye(3).to_sparse_csr()), None)
        )
        calls["validation"] = val
        return graph, train, val, object(), x

    class FakeMap:
        def __init__(self, anchors, mapping, kernel="linear"):
            self.anchors, self.mapping, self.kernel = anchors, mapping, kernel

        @classmethod
        def fit(cls, h, basis, seed):
            return cls(h[:2].double(), torch.eye(2, dtype=torch.double))

        def __call__(self, h):
            return h.double()[:, :2]

    def features(h, feature_map, path, chunk, stop):
        if stop():
            raise InterruptedError("features stopped")
        if not path.exists():
            np.save(path, feature_map(h).numpy())
        return np.load(path, mmap_mode="r")

    def teacher(phi, labels, mask, **kwargs):
        calls["teacher"] += 1
        assert bool((labels[~mask] == 0).all())
        weight = torch.eye(2, dtype=torch.double)
        logits = torch.from_numpy(np.array(phi[:])) @ weight
        calls["logits"] = logits
        return logits, weight

    def optimizer(z, q, assignment, *, folder, steps, resume_state, **kwargs):
        calls["optimizer"] += 1
        assert kwargs["solver_mode"] == "exact"
        assert kwargs["inner_loss_weighting"] in ("mass", "uniform")
        calls["optimizer_q"] = q.clone()
        calls["optimizer_loss"] = kwargs["inner_loss_weighting"]
        assert not kwargs["save_assignment"]
        if resume_state is not None:
            calls["resume_step"] = resume_state["step"]
        probability = torch.nn.functional.one_hot(assignment, 4).double()
        moments = probability.T @ make_material(z, q) / len(z)
        (folder / "checkpoints").mkdir(parents=True, exist_ok=True)
        save_state(
            dict(step=steps, moments=moments, J_exact=True, teacher_ce=0.5),
            folder / "checkpoints" / f"step_{steps:06d}.pt",
        )
        save_state(dict(step=steps), folder / "resume.pt")

    def nystrom(h, q, assignment, feature_map, phi, folder, steps, **kwargs):
        calls["nystrom"] += 1
        assert torch.equal(h, x.double())
        assert not kwargs["stop"]()
        np.testing.assert_array_equal(np.array(phi[:]), feature_map(h).numpy())
        probability = torch.nn.functional.one_hot(assignment, 4).double()
        moments = probability.T @ make_material(h, q) / len(h)
        folder.mkdir(parents=True, exist_ok=True)
        if (folder / "resume.pt").exists():
            calls["nystrom_resume_step"] = torch.load(folder / "resume.pt", weights_only=False)["step"]
        save_state(dict(step=steps, moments=moments, outer_ce=0.4), folder / f"step_{steps:06d}.pt")
        save_state(dict(step=steps), folder / "resume.pt")
        calls["nystrom_expected_x"] = (probability.T @ h / probability.sum(0)[:, None]).float()

    def student(cx, cy, mass, supplied_graph, supplied_validation, **kwargs):
        assert "testing" not in kwargs
        assert supplied_graph is graph
        assert supplied_validation is calls["validation"]
        assert kwargs["weighting"] == "uniform"
        assert not kwargs["stop"]()
        calls["student_x"] = cx.clone()
        calls["students"].append(kwargs)
        return dict(seed=kwargs["seed"], epoch=10, val_acc=61.0, val_ce=0.5)

    monkeypatch.setattr(pilot, "_prepare_dataset", dataset)
    monkeypatch.setattr(pilot, "NystromMap", FakeMap)
    monkeypatch.setattr(pilot, "cache_features", features)
    monkeypatch.setattr(pilot, "fit_streaming_teacher", teacher)
    monkeypatch.setattr(pilot, "optimize_ce_assignment", optimizer)
    monkeypatch.setattr(pilot, "optimize_nystrom", nystrom)
    monkeypatch.setattr(pilot, "fit_inductive_gcn", student)
    monkeypatch.setattr(pilot, "feature_kmeans", lambda h, cells, seed: torch.arange(len(h)) % cells)
    for name, ratio in (("arxiv", 0.0005), ("flickr", 0.001), ("reddit", 0.0005)):
        monkeypatch.setitem(pilot.BUDGET, (name, ratio), 4)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *args: pytest.fail("GPU accessed by CPU test"))
    return calls


@pytest.mark.parametrize("dataset,ratio", [("arxiv", 0.0005), ("flickr", 0.001), ("reddit", 0.0005)])
def test_full_graph_protocol_is_training_label_only_and_validation_only(
    tmp_path, mocked_pipeline, dataset, ratio
):
    report, root = pilot.run_pilot(dataset, ratio, tmp_path, device="cpu", basis=2, rank=2, steps=20)
    assert report["status"] == "complete"
    assert report["completed_steps"] == 20
    assert report["train_nodes"] == 18
    assert report["peak_gpu_bytes"] == 0
    assert report["val_acc_mean"] == 61
    assert not any(key.startswith("test_") for key in report)
    assert not any(key.startswith("test_") for key in report["students"][0])
    assert json.loads((root / "report.json").read_text())["status"] == "complete"
    assert mocked_pipeline["teacher"] == mocked_pipeline["optimizer"] == 1


def test_pilot_resumes_steps_and_reuses_shared_teacher(tmp_path, mocked_pipeline):
    first, root = pilot.run_pilot("arxiv", 0.0005, tmp_path, device="cpu", basis=2, rank=2, steps=20)
    second, same = pilot.run_pilot("arxiv", 0.0005, tmp_path, device="cpu", basis=2, rank=2, steps=50)
    assert first["status"] == second["status"] == "complete"
    assert same == root
    assert mocked_pipeline["teacher"] == 1
    assert mocked_pipeline["optimizer"] == 2
    assert mocked_pipeline["resume_step"] == 20


def test_stop_callback_prevents_data_or_training_and_persists_report(tmp_path, mocked_pipeline):
    report, root = pilot.run_pilot(
        "arxiv", 0.0005, tmp_path, device="cpu", basis=2, rank=2, stop=lambda: True
    )
    assert report["status"] == "stopped"
    assert mocked_pipeline["data"] == mocked_pipeline["teacher"] == mocked_pipeline["optimizer"] == 0
    assert json.loads((root / "report.json").read_text())["status"] == "stopped"


def test_deadline_interrupts_blocking_solver_and_leaves_teacher_cache(tmp_path, mocked_pipeline, monkeypatch):
    def slow_optimizer(*args, **kwargs):
        time.sleep(2)

    monkeypatch.setattr(pilot, "optimize_ce_assignment", slow_optimizer)
    started = time.monotonic()
    report, root = pilot.run_pilot(
        "arxiv", 0.0005, tmp_path, device="cpu", basis=2, rank=2, deadline_seconds=0.15
    )
    assert time.monotonic() - started < 1.0
    assert report["status"] == "stopped"
    assert (root.parent / "teacher.pt").exists()
    assert not mocked_pipeline["students"]


def test_nonconverged_teacher_is_reported_and_condensation_refused(tmp_path, mocked_pipeline, monkeypatch):
    def failed(*args, **kwargs):
        raise RuntimeError("Teacher logistic fit did not converge")

    monkeypatch.setattr(pilot, "fit_streaming_teacher", failed)
    report, root = pilot.run_pilot("arxiv", 0.0005, tmp_path, device="cpu", basis=2, rank=2)
    assert report["status"] == "failed"
    assert "did not converge" in report["reason"]
    assert mocked_pipeline["optimizer"] == 0
    assert not (root.parent / "teacher.pt").exists()


def test_changed_data_rejects_shared_cache(tmp_path, mocked_pipeline):
    pilot.run_pilot("arxiv", 0.0005, tmp_path, device="cpu", basis=2, rank=2)
    mocked_pipeline["graph"]["x"][0, 0] += 0.1
    report, _ = pilot.run_pilot("arxiv", 0.0005, tmp_path, device="cpu", basis=2, rank=2)
    assert report["status"] == "failed"
    assert "source digest differs" in report["reason"]
    assert mocked_pipeline["teacher"] == 1


def test_streamed_feature_proxy_checks_each_chunk():
    array = np.ones((4, 2), dtype=np.float64)
    accesses = []
    proxy = pilot._BoundedFeatures(array, lambda: accesses.append(1))
    assert proxy.shape == (4, 2) and proxy.dtype == np.float64
    assert len(proxy) == 4
    np.testing.assert_array_equal(proxy[:2], array[:2])
    np.testing.assert_array_equal(proxy[2:], array[2:])
    assert len(accesses) == 2


def test_refined_training_targets_leave_heldout_q_unchanged_and_final_student_uniform(
    tmp_path, mocked_pipeline
):
    report, _ = pilot.run_pilot(
        "arxiv",
        0.0005,
        tmp_path,
        device="cpu",
        basis=2,
        rank=2,
        temperature=0.5,
        train_target_mix=0.7,
        inner_loss_weighting="uniform",
    )
    assert report["status"] == "complete"
    mask = torch.arange(18) < 12
    teacher_q = (mocked_pipeline["logits"] / 0.5).softmax(1).double()
    expected = teacher_q.clone()
    expected[mask] = (
        0.3 * teacher_q[mask]
        + 0.7 * torch.nn.functional.one_hot(mocked_pipeline["graph"]["y"][mask], num_classes=2).double()
    )
    torch.testing.assert_close(mocked_pipeline["optimizer_q"], expected, atol=1e-15, rtol=1e-14)
    assert torch.equal(mocked_pipeline["optimizer_q"][~mask], teacher_q[~mask])
    assert mocked_pipeline["optimizer_loss"] == "uniform"
    assert all(call["weighting"] == "uniform" for call in mocked_pipeline["students"])


def test_all_training_flickr_hard_targets_recover_classes_for_balanced_initialization(
    tmp_path,
    mocked_pipeline,
    monkeypatch,
):
    supplied = mocked_pipeline["graph"]
    supplied["y"] = torch.arange(18) % 2
    train = torch.ones(18, dtype=torch.bool)
    validation = dict(x=supplied["x"][:3], y=supplied["y"][:3], adj=torch.eye(3).to_sparse_csr()), None

    def full_train_dataset(*args):
        mocked_pipeline["validation"] = validation
        return supplied, train, validation, object(), supplied["x"]

    def balanced(h, q, cells, seed, alpha):
        mocked_pipeline["balanced_q"] = q.clone()
        sizes = torch.bincount(q.argmax(1), minlength=q.shape[1])
        mocked_pipeline["class_allocation"] = allocate_balanced_cells(sizes, cells)
        return teacher_balanced_kmeans(h, q, cells, seed, alpha)

    monkeypatch.setattr(pilot, "_prepare_dataset", full_train_dataset)
    monkeypatch.setattr(pilot, "teacher_balanced_kmeans", balanced)
    monkeypatch.setattr(
        "src.partition_initialization.feature_kmeans", lambda h, cells, seed: torch.arange(len(h)) % cells
    )
    report, _ = pilot.run_pilot(
        "flickr",
        0.001,
        tmp_path,
        device="cpu",
        basis=2,
        rank=2,
        train_target_mix=1.0,
        initialization="teacher_balanced",
    )
    assert report["status"] == "complete"
    hard_q = torch.nn.functional.one_hot(supplied["y"], num_classes=2).double()
    assert torch.equal(mocked_pipeline["balanced_q"], hard_q)
    assert torch.equal(mocked_pipeline["optimizer_q"], hard_q)
    assert torch.equal(mocked_pipeline["class_allocation"], torch.tensor([2, 2]))
    assert all(call["weighting"] == "uniform" for call in mocked_pipeline["students"])


def test_temperature_mix_and_inner_loss_have_distinct_candidates_but_share_teacher(tmp_path, mocked_pipeline):
    options = ({}, dict(temperature=0.5), dict(train_target_mix=1.0), dict(inner_loss_weighting="uniform"))
    roots = []
    for changed in options:
        report, root = pilot.run_pilot("arxiv", 0.0005, tmp_path, device="cpu", basis=2, rank=2, **changed)
        assert report["status"] == "complete"
        roots.append(root)
    assert len(set(roots)) == 4
    assert len({root.parent for root in roots}) == 1
    assert mocked_pipeline["teacher"] == 1
    default = json.loads((roots[0] / "protocol.json").read_text())["candidate"]
    legacy = dict(
        version=1,
        ratio=0.0005,
        cells=4,
        temperature=0.3,
        rank=2,
        penalty=1e-4,
        lr=0.01,
        condensation_seed=0,
        initialization="feature",
        alpha=1.0,
        assignment="low_rank",
        inner_loss="exact_mass_ce",
        student_loss="uniform_ce",
        synthetic_adjacency="identity",
    )
    assert default == legacy
    assert roots[0].name == f"candidate_{pilot._fingerprint(legacy)}"


@pytest.mark.parametrize(
    "changed",
    [
        dict(temperature=0),
        dict(temperature=float("nan")),
        dict(train_target_mix=-0.1),
        dict(train_target_mix=1.1),
        dict(inner_loss_weighting="mse"),
    ],
)
def test_invalid_target_ablation_options_are_rejected_before_loading(tmp_path, mocked_pipeline, changed):
    with pytest.raises(ValueError):
        pilot.run_pilot("arxiv", 0.0005, tmp_path, device="cpu", basis=2, rank=2, **changed)
    assert mocked_pipeline["data"] == 0


def test_nystrom_endpoint_uses_raw_h_centroids_and_shares_teacher_with_linear(tmp_path, mocked_pipeline):
    linear, linear_root = pilot.run_pilot("arxiv", 0.0005, tmp_path, device="cpu", basis=2, rank=2)
    report, root = pilot.run_pilot(
        "arxiv", 0.0005, tmp_path, device="cpu", basis=2, rank=2, surrogate="nystrom"
    )
    assert linear["status"] == report["status"] == "complete"
    assert root != linear_root and root.parent == linear_root.parent
    assert mocked_pipeline["teacher"] == mocked_pipeline["optimizer"] == mocked_pipeline["nystrom"] == 1
    assert report["candidate"]["surrogate"] == "nystrom"
    assert "surrogate" not in linear["candidate"]
    assert report["outer_ce"] == 0.4
    assert (root / "condensation" / "step_000020.pt").exists()
    assert not (root / "condensation" / "checkpoints").exists()
    torch.testing.assert_close(
        mocked_pipeline["student_x"], mocked_pipeline["nystrom_expected_x"], atol=0, rtol=0
    )
    assert all(call["weighting"] == "uniform" for call in mocked_pipeline["students"])


def test_nystrom_resumes_internally_and_reuses_existing_endpoint(tmp_path, mocked_pipeline):
    options = dict(device="cpu", basis=2, rank=2, surrogate="nystrom")
    first, root = pilot.run_pilot("arxiv", 0.0005, tmp_path, steps=20, **options)
    second, same = pilot.run_pilot("arxiv", 0.0005, tmp_path, steps=50, **options)
    cached, repeated = pilot.run_pilot("arxiv", 0.0005, tmp_path, steps=50, **options)
    assert first["status"] == second["status"] == cached["status"] == "complete"
    assert root == same == repeated
    assert mocked_pipeline["nystrom"] == 2
    assert mocked_pipeline["teacher"] == 1
    assert mocked_pipeline["nystrom_resume_step"] == 20


@pytest.mark.parametrize(
    "changed", [dict(surrogate="unknown"), dict(surrogate="nystrom", inner_loss_weighting="uniform")]
)
def test_invalid_surrogate_or_uniform_nystrom_is_rejected_before_loading(tmp_path, mocked_pipeline, changed):
    with pytest.raises(ValueError):
        pilot.run_pilot("arxiv", 0.0005, tmp_path, device="cpu", basis=2, rank=2, **changed)
    assert mocked_pipeline["data"] == 0


def test_nystrom_deadline_and_stop_guard_are_preserved(tmp_path, mocked_pipeline, monkeypatch):
    def slow(*args, **kwargs):
        assert not kwargs["stop"]()
        time.sleep(2)

    monkeypatch.setattr(pilot, "optimize_nystrom", slow)
    started = time.monotonic()
    report, _ = pilot.run_pilot(
        "arxiv", 0.0005, tmp_path, device="cpu", basis=2, rank=2, surrogate="nystrom", deadline_seconds=0.15
    )
    assert report["status"] == "stopped"
    assert time.monotonic() - started < 1.0
