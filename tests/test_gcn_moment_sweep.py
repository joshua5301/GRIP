import json

import pytest
import torch

import src.gcn_moment_sweep as module


def test_dropout_comparison_reuses_saved_teachers_and_trains_only_missing(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(json.dumps(dict(
        dataset="cora", data_digest="abc", seed=0,
        training=dict(lr=.01, weight_decay=.0005, epochs=1000, eval_every=10))))
    (source / "features.pt").write_bytes(b"features")
    module.pd.DataFrame([dict(candidate=i, hidden=256, dropout=p) for i, p in enumerate((.1, .5))]).to_csv(
        source / "teacher_grid.csv", index=False)
    for i in range(2):
        folder = source / f"candidate_{i:03d}"
        folder.mkdir()
        (folder / "teacher.pt").write_bytes(b"saved teacher")
    calls = []

    def fit(graph, train, validation, testing, settings, seed, folder):
        assert testing is None
        assert settings["hidden"] == 256
        assert settings["weight_decay"] == .0005
        calls.append(settings["dropout"])
        result = dict(val_acc=80., val_ce=.7, epoch=10)
        folder.mkdir(parents=True)
        module.save_state(result, folder / "teacher.pt")
        return result

    monkeypatch.setattr(module, "_prepare_dataset", lambda *args: ({}, None, None, None, None))
    monkeypatch.setattr(module, "_data_digest", lambda splits: "abc")
    monkeypatch.setattr(module, "fit_teacher", fit)
    paths = module.prepare_dropout_teachers(source, tmp_path / "output", device="cpu")
    assert calls == [0.]
    assert paths[.1] == dict(source=str(source), candidate=0)
    assert paths[.5] == dict(source=str(source), candidate=1)
    assert paths[0.]["source"] != str(source)
    assert module.prepare_dropout_teachers(source, tmp_path / "output", device="cpu") == paths
    assert calls == [0.]


def test_width_teacher_identity_keeps_student_settings(tmp_path):
    config = dict(dataset="cora", data_digest="abc", seed=0, training={"lr": .01})
    (tmp_path / "config.json").write_text(json.dumps(config))
    module.pd.DataFrame([dict(candidate=0, hidden=128, dropout=.9)]).to_csv(
        tmp_path / "teacher_grid.csv", index=False)
    folder = tmp_path / "candidate_000"
    folder.mkdir()
    (folder / "teacher.pt").write_bytes(b"teacher")
    (tmp_path / "features.pt").write_bytes(b"features")
    request = dict(dataset="cora", data_digest="abc", teacher_seed=0, settings={"hidden": 256})
    identity = module.width_source_identity(tmp_path, 0, request)
    assert identity["hidden"] == 128
    assert request["settings"]["hidden"] == 256
    with pytest.raises(ValueError, match="data_digest"):
        module.width_source_identity(tmp_path, 0, dict(request, data_digest="other"))
    with pytest.raises(ValueError, match="exactly one"):
        module.width_source_identity(tmp_path, 1, request)


def test_saved_width_teacher_uses_teacher_architecture(tmp_path, monkeypatch):
    model = module.GCN(3, 4, 2, 2, .5)
    source = tmp_path / "source.pt"
    module.save_state(dict(state=model.state_dict(), logits=torch.zeros(3, 2),
                           val_acc=80., epoch=1), source)
    identity = dict(teacher_path=str(source), hidden=4, dropout=.5)
    graph = dict(x=torch.eye(3), y=torch.tensor([0, 1, 0]))
    monkeypatch.setattr(module, "_metrics", lambda model, pair: dict(acc=70., ce=1.))
    result = module.load_width_teacher(identity, tmp_path, graph, None)
    assert result["hidden"] == 4
    assert result["test_acc"] == 70.
    assert torch.equal(result["logits"], torch.zeros(3, 2))


def test_shared_teacher_identity_checks_protocol_and_file_contents(tmp_path):
    config = dict(dataset="cora", data_digest="abc", teacher_seed=0, settings={"lr": .01})
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "teacher.pt").write_bytes(b"teacher")
    (tmp_path / "features.pt").write_bytes(b"features")
    first = module.shared_source_identity(tmp_path, config)
    (tmp_path / "teacher.pt").write_bytes(b"different teacher")
    second = module.shared_source_identity(tmp_path, config)
    assert first["teacher_sha256"] != second["teacher_sha256"]
    assert first["features_sha256"] == second["features_sha256"]
    with pytest.raises(ValueError, match="data_digest"):
        module.shared_source_identity(tmp_path, dict(config, data_digest="other"))


def test_teacher_restores_validation_checkpoint_and_tests_once(tmp_path, monkeypatch):
    graph = dict(x=torch.eye(3), adj=torch.eye(3), y=torch.tensor([0, 1, 0]))
    validation, testing = (graph, torch.tensor([1])), (graph, torch.tensor([2]))
    values, states, tests = iter([90., 80., 70., 60.]), [], []

    def metrics(model, pair):
        if pair is validation:
            if len(states) < 4:
                states.append({key: value.detach().clone() for key, value in model.state_dict().items()})
                return dict(acc=next(values), ce=1.)
            return dict(acc=90., ce=1.)
        if pair is testing:
            tests.append(len(states))
        return dict(acc=50., ce=1.)

    monkeypatch.setattr(module, "_metrics", metrics)
    settings = dict(hidden=4, dropout=0., lr=0.01, weight_decay=0.0005, epochs=4, eval_every=1)
    result = module.fit_teacher(graph, torch.tensor([0, 1]), validation, testing, settings, 0, tmp_path)
    assert result["epoch"] == 1
    assert tests == [4]
    for key in states[0]:
        assert torch.equal(result["state"][key], states[0][key])
    cached = module.fit_teacher(graph, torch.tensor([0, 1]), validation, testing, settings, 0, tmp_path)
    assert torch.equal(cached["logits"], result["logits"])
    assert tests == [4]
    assert torch.allclose(result["logits"].exp().sum(1), torch.ones(3))


def test_invalid_temperature_rejected_before_loading(tmp_path):
    with pytest.raises(ValueError, match="Temperatures"):
        module.run_gcn_moment_sweep("cora", 0.026, tmp_path, [0], [1.])


def test_teacher_grid_selects_validation_and_preserves_student_settings(tmp_path, monkeypatch):
    graph = dict(x=torch.eye(3), adj=torch.eye(3), y=torch.tensor([0, 1, 0]))
    settings = dict(hidden=4, dropout=0.9, lr=0.01, weight_decay=0.0005, epochs=4, eval_every=1)
    calls = []

    def fit(graph, train, validation, testing, options, seed, folder):
        assert testing is None
        calls.append(dict(options))
        model = module.GCN(3, 4, 2, 2, options["dropout"])
        result = dict(state=model.state_dict(), logits=torch.zeros(3, 2), epoch=1, seed=seed,
                      val_acc=80., val_ce=options["dropout"] + options["weight_decay"])
        folder.mkdir(parents=True, exist_ok=True)
        module.save_state(result, folder / "teacher.pt")
        return result

    monkeypatch.setattr(module, "fit_teacher", fit)
    monkeypatch.setattr(module, "_metrics", lambda model, pair: dict(acc=70., ce=1.))
    result = module.select_teacher(graph, torch.tensor([0]), (graph, torch.tensor([1])),
                                   (graph, torch.tensor([2])), settings, 0, tmp_path,
                                   [0.5, 0.1], [0.001, 0.0001])
    assert len(calls) == 4
    assert result["dropout"] == 0.1
    assert result["weight_decay"] == 0.0001
    assert settings["dropout"] == 0.9
    assert settings["weight_decay"] == 0.0005
    assert all(row["lr"] == 0.01 for row in calls)
def test_zero_weight_var_part_lloyd_is_label_independent():
    import torch

    from src.moment_lloyd import moment_lloyd_partition
    generator = torch.Generator().manual_seed(12)
    x = torch.randn(40, 5, generator=generator)
    q1 = torch.randn(40, 3, generator=generator).softmax(1)
    q2 = torch.randn(40, 3, generator=generator).softmax(1)
    options = dict(m=4, moment_weight=0., mode="variance_sum", seeding="feature_var", max_sweeps=200)
    first = moment_lloyd_partition(x, q1, **options)
    second = moment_lloyd_partition(x, q2, **options)
    assert first["converged"] and second["converged"]
    assert torch.equal(first["assignment"], second["assignment"])
    torch.testing.assert_close(first["x"], second["x"])
    assert all(b <= a + 1e-10 for a, b in zip(first["history"], first["history"][1:]))


def test_label_objectives_share_feature_initialization():
    from src.moment_lloyd import moment_lloyd_partition

    generator = torch.Generator().manual_seed(42)
    x = torch.randn(40, 5, generator=generator)
    q = torch.randn(40, 3, generator=generator).softmax(1)
    partitions = [
        moment_lloyd_partition(x, q, 4, mode=mode, moment_weight=0.3,
                               seeding="feature_var", max_sweeps=200)
        for mode in ("filtered_batch", "kl", "variance_sum")
    ]
    assert len({p["initial_assignment_digest"] for p in partitions}) == 1
    for p in partitions[1:]:
        assert p["converged"]
        assert all(b <= a + 1e-10 for a, b in zip(p["history"], p["history"][1:]))


def test_joint_variance_initialization_matches_objective_space():
    from src.moment_lloyd import moment_lloyd_partition
    from src.moment_seeding import pca_partition

    generator = torch.Generator().manual_seed(42)
    x = torch.randn(40, 5, generator=generator, dtype=torch.float64)
    q = torch.randn(40, 3, generator=generator, dtype=torch.float64).softmax(1)
    centered = x - x.mean(0)
    z = centered / centered.square().sum(1).mean().sqrt()
    weight = 0.3
    joint = torch.cat((z, weight**0.5 * (q - q.mean(0))), dim=1)
    expected, _ = pca_partition(joint, 4, method="var")
    result = moment_lloyd_partition(x, q, 4, mode="variance_sum", moment_weight=weight,
                                    seeding="bound_var", max_sweeps=0)
    assert torch.equal(result["assignment"], expected)


def test_joint_initialization_rejects_other_objectives(tmp_path):
    with pytest.raises(ValueError, match="requires variance_sum"):
        module.run_gcn_moment_sweep("cora", 0.026, tmp_path, [1.], [1.],
                                    partition_method="kl", initialization_space="joint")


