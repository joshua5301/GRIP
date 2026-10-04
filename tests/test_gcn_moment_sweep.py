import pytest
import torch

import src.gcn_moment_sweep as module


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
