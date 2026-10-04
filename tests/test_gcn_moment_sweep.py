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
