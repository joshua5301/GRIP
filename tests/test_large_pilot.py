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


def test_default_routes_option_preserves_legacy_cache_identity(tmp_path, mocked_pipeline, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Default pilots must not prepare or replay extra serving routes")

    monkeypatch.setattr(pilot, "_validation_route_inputs", forbidden)
    monkeypatch.setattr(pilot, "replay_routes", forbidden)
    options = dict(device="cpu", basis=2, rank=2, steps=20)
    default, root = pilot.run_pilot("arxiv", 0.0005, tmp_path, **options)
    explicit, same = pilot.run_pilot("arxiv", 0.0005, tmp_path, report_routes=False, **options)
    assert default["status"] == explicit["status"] == "complete"
    assert same == root
    settings = dict(
        epochs=300, eval_every=10, hidden=256, dropout=0.31881090213944857,
        lr=0.01, weight_decay=0.0005,
    )
    assert default["student_recipe"] == explicit["student_recipe"] == settings
    folder = root / "validation" / f"step_20_{pilot._fingerprint(settings)}"
    assert all(call["folder"] == folder for call in mocked_pipeline["students"])
    assert (root / f"report_step_20_{pilot._fingerprint(settings)}.json").exists()
    assert "serving_comparison" not in default and "serving_comparison" not in explicit
    assert "report_routes" not in default["candidate"] and "report_routes" not in settings
    assert mocked_pipeline["teacher"] == mocked_pipeline["optimizer"] == 1


@pytest.mark.parametrize("dataset,ratio", [("arxiv", 0.0005), ("flickr", 0.001), ("reddit", 0.0005)])
def test_validation_routes_use_own_graph_and_explicit_validation_mask_only(
    tmp_path, mocked_pipeline, monkeypatch, dataset, ratio,
):
    replays = []

    def replay(selected_path, graph, propagated, masks, settings, output_path, seed, stop):
        expected_graph, original_mask = mocked_pipeline["validation"]
        assert graph is expected_graph
        assert set(masks) == {"val"}
        if original_mask is None:
            assert torch.equal(masks["val"], torch.ones(len(graph["x"]), dtype=torch.bool))
            assert len(graph["x"]) != len(mocked_pipeline["graph"]["x"])
        else:
            assert masks["val"] is original_mask
        expected = torch.sparse.mm(graph["adj"], torch.sparse.mm(graph["adj"], graph["x"]))
        assert torch.equal(propagated, expected)
        assert selected_path == output_path.parent / f"seed_{seed}_selected.pt"
        assert output_path.name == f"seed_{seed}_validation_routes_v1.json"
        assert settings == mocked_pipeline["students"][-1]["settings"]
        assert not stop()
        replays.append(seed)
        return dict(
            seed=seed, epoch=10, selection="same weights at GCN validation-selected epoch",
            gcn_val_acc=61.0, gcn_val_ce=0.5, mlp_val_acc=55.0, mlp_val_ce=0.9,
        )

    monkeypatch.setattr(pilot, "replay_routes", replay)
    options = dict(device="cpu", basis=2, rank=2, steps=20, student_seeds=(2, 7))
    original, root = pilot.run_pilot(dataset, ratio, tmp_path, **options)
    legacy_report = root / f"report_step_20_{pilot._fingerprint(original['student_recipe'])}.json"
    legacy_bytes = legacy_report.read_bytes()
    protocol_bytes = (root / "protocol.json").read_bytes()
    report, same = pilot.run_pilot(dataset, ratio, tmp_path, report_routes=True, **options)
    assert report["status"] == "complete"
    assert same == root
    assert replays == [2, 7]
    assert report["val_acc_mean"] == report["gcn_val_acc_mean"] == 61
    assert report["mlp_val_acc_mean"] == 55
    assert report["gcn_minus_mlp_val_acc_mean"] == 6
    assert report["gcn_minus_mlp_val_ce_mean"] == pytest.approx(-0.4)
    assert all(not any("test_" in key for key in row) for row in report["students"])
    assert legacy_report.read_bytes() == legacy_bytes
    assert (root / "protocol.json").read_bytes() == protocol_bytes
    route_report_key = pilot._fingerprint(dict(student=original["student_recipe"], report_routes=True, version=1))
    assert (root / f"report_step_20_{route_report_key}.json").exists()
    assert mocked_pipeline["teacher"] == mocked_pipeline["optimizer"] == 1


@pytest.mark.parametrize("sparse", [False, True])
def test_validation_h_uses_own_packed_adjacency(tmp_path, sparse):
    x = torch.tensor([[1.0, 0.1], [0.1, 2.0], [0.5, 1.0]])
    adjacency = 0.7 * torch.eye(3) + 0.3 * torch.eye(3).roll(1, 1)
    graph = dict(x=x, y=torch.tensor([0, 1, 1]), adj=adjacency.to_sparse_csr() if sparse else adjacency)
    checks = []
    supplied, propagated, masks = pilot._validation_route_inputs(
        (graph, None), "validation-source", tmp_path, lambda: checks.append(1),
    )
    assert supplied is graph
    torch.testing.assert_close(propagated, adjacency @ adjacency @ x)
    assert not torch.equal(propagated, x)
    assert set(masks) == {"val"}
    assert torch.equal(masks["val"], torch.ones(3, dtype=torch.bool))
    assert len(checks) == 3


def test_validation_h_freezes_original_source_coordinates_and_refuses_changed_source(tmp_path, monkeypatch):
    x = torch.tensor([[1.0, 0.1], [0.1, 2.0], [0.5, 1.0]])
    graph = dict(x=x, y=torch.tensor([0, 1, 1]), adj=torch.eye(3).to_sparse_csr())
    validation = graph, torch.tensor([True, False, True])
    _, first, masks = pilot._validation_route_inputs(validation, "source-v1", tmp_path, lambda: None)
    saved_path = tmp_path / "validation_routes_S2X_v1.pt"
    saved_bytes = saved_path.read_bytes()
    original_mm = torch.sparse.mm

    def slightly_rounded(adjacency, features):
        return original_mm(adjacency, features) + 1e-7

    monkeypatch.setattr(torch.sparse, "mm", slightly_rounded)
    _, frozen, same_masks = pilot._validation_route_inputs(validation, "source-v1", tmp_path, lambda: None)
    assert torch.equal(frozen, first)
    assert same_masks["val"] is masks["val"] is validation[1]
    assert saved_path.read_bytes() == saved_bytes
    with pytest.raises(ValueError, match="source digest differs"):
        pilot._validation_route_inputs(validation, "different-source", tmp_path, lambda: None)
    assert saved_path.read_bytes() == saved_bytes


def test_existing_selected_student_is_replayed_without_refit_or_weight_changes(
    tmp_path, mocked_pipeline, monkeypatch,
):
    import src.student_routes as routes
    from src.inductive_evaluation import fit_inductive_gcn

    monkeypatch.setattr(pilot, "fit_inductive_gcn", fit_inductive_gcn)
    graph = mocked_pipeline["graph"]
    graph["adj"] = (0.7 * torch.eye(18) + 0.3 * torch.eye(18).roll(1, 1)).to_sparse_csr()
    options = dict(device="cpu", basis=2, rank=2, steps=20, epochs=4, hidden=4, dropout=0.5)
    first, root = pilot.run_pilot("arxiv", 0.0005, tmp_path, **options)
    assert first["status"] == "complete"
    folder = root / "validation" / f"step_20_{pilot._fingerprint(first['student_recipe'])}"
    selected_path = folder / "seed_0_selected.pt"
    selected_bytes = selected_path.read_bytes()
    selected = torch.load(selected_path, map_location="cpu", weights_only=False)
    assert set(selected) == {"epoch", "model_state", "fingerprint"}
    forward = routes._forward
    seen = []

    def same_weights(model, features, adjacency):
        for name, value in model.state_dict().items():
            assert torch.equal(value, selected["model_state"][name])
        seen.append("mlp" if adjacency is None else "gcn")
        return forward(model, features, adjacency)

    def no_training(*args, **kwargs):
        pytest.fail("An existing validation-selected student must not be refit")

    monkeypatch.setattr(routes, "_forward", same_weights)
    monkeypatch.setattr(torch.optim, "Adam", no_training)
    original_replay = pilot.replay_routes

    def only_validation(selected_path, graph, propagated, masks, *args, **kwargs):
        assert set(masks) == {"val"}
        raw_labels = graph["y"]

        class GuardedLabels:
            def __getitem__(self, mask):
                assert mask is masks["val"], "Replay may read only validation labels"
                return raw_labels[mask]

        return original_replay(selected_path, dict(graph, y=GuardedLabels()), propagated, masks, *args, **kwargs)

    monkeypatch.setattr(pilot, "replay_routes", only_validation)
    paired, same = pilot.run_pilot("arxiv", 0.0005, tmp_path, report_routes=True, **options)
    assert paired["status"] == "complete"
    assert same == root
    assert seen == ["gcn", "mlp"]
    assert selected_path.read_bytes() == selected_bytes
    assert paired["students"][0]["epoch"] == selected["epoch"]
    assert paired["students"][0]["gcn_val_acc"] == first["students"][0]["val_acc"]
    assert paired["students"][0]["gcn_val_ce"] == first["students"][0]["val_ce"]
    assert paired["students"][0]["serving_selection"] == routes.SELECTION
    assert json.loads((folder / "seed_0_validation_routes_v1.json").read_text())["recipe"]["test_enabled"] is False
    paired_cached, _ = pilot.run_pilot("arxiv", 0.0005, tmp_path, report_routes=True, **options)
    assert paired_cached["students"] == paired["students"]
    assert seen == ["gcn", "mlp"]


@pytest.mark.parametrize("mismatch", ["accuracy", "ce", "test_mask"])
def test_inconsistent_or_test_enabled_replay_is_refused(tmp_path, mocked_pipeline, monkeypatch, mismatch):
    def replay(*args, **kwargs):
        result = dict(
            selection="same weights at GCN validation-selected epoch",
            gcn_val_acc=61.0, gcn_val_ce=0.5, mlp_val_acc=55.0, mlp_val_ce=0.9,
        )
        if mismatch == "accuracy":
            result["gcn_val_acc"] = 62
        elif mismatch == "ce":
            result["gcn_val_ce"] = 0.7
        else:
            result["gcn_test_acc"] = 99
        return result

    monkeypatch.setattr(pilot, "replay_routes", replay)
    report, _ = pilot.run_pilot("arxiv", 0.0005, tmp_path, device="cpu", basis=2, rank=2, report_routes=True)
    assert report["status"] == "failed"
    assert not report["students"]
    if mismatch == "test_mask":
        assert "test metrics" in report["reason"]
    else:
        assert "differs" in report["reason"]


def test_validation_route_preparation_checks_stop_before_propagation_or_save(tmp_path, monkeypatch):
    def stopped():
        raise InterruptedError("paused")

    def forbidden(*args):
        pytest.fail("A stopped route preparation must not compute propagation")

    monkeypatch.setattr(torch.sparse, "mm", forbidden)
    graph = dict(x=torch.ones(3, 2), y=torch.tensor([0, 1, 0]), adj=torch.eye(3).to_sparse_csr())
    with pytest.raises(InterruptedError, match="paused"):
        pilot._validation_route_inputs((graph, None), "source", tmp_path, stopped)
    assert not (tmp_path / "validation_routes_S2X_v1.pt").exists()


def test_interrupted_route_replay_preserves_selected_student_and_checkpoint(
    tmp_path, mocked_pipeline, monkeypatch,
):
    from src.inductive_evaluation import fit_inductive_gcn

    def paused(*args, **kwargs):
        raise InterruptedError("Serving replay paused")

    monkeypatch.setattr(pilot, "fit_inductive_gcn", fit_inductive_gcn)
    options = dict(device="cpu", basis=2, rank=2, epochs=4, hidden=4)
    complete, root = pilot.run_pilot("arxiv", 0.0005, tmp_path, **options)
    assert complete["status"] == "complete"
    selected_path = (
        root / "validation" / f"step_20_{pilot._fingerprint(complete['student_recipe'])}"
        / "seed_0_selected.pt"
    )
    selected_bytes = selected_path.read_bytes()
    monkeypatch.setattr(pilot, "replay_routes", paused)
    report, same = pilot.run_pilot("arxiv", 0.0005, tmp_path, report_routes=True, **options)
    assert report["status"] == "stopped"
    assert same == root
    assert report["stage"] == "validation_serving_routes"
    assert selected_path.read_bytes() == selected_bytes
    assert (root / "condensation" / "checkpoints" / "step_000020.pt").exists()
    assert (root.parent / "teacher.pt").exists()
    assert not report["students"]
    assert not list(root.rglob("*_validation_routes_v1.json"))


def test_invalid_routes_option_is_rejected_before_loading(tmp_path, mocked_pipeline):
    with pytest.raises(ValueError, match="report_routes must be boolean"):
        pilot.run_pilot("arxiv", 0.0005, tmp_path, device="cpu", basis=2, rank=2, report_routes="true")
    assert mocked_pipeline["data"] == 0
