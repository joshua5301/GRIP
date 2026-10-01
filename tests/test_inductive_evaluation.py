import torch

import src.evaluation as evaluation


def test_inductive_metrics_use_each_split_graph(monkeypatch):
    train = dict(x=torch.tensor([[0.0], [1.0]]), y=torch.tensor([0, 1]), adj=None)
    val = dict(x=torch.tensor([[2.0], [3.0]]), y=torch.tensor([1, 0]), adj=None)
    test = dict(x=torch.tensor([[4.0], [5.0]]), y=torch.tensor([0, 1]), adj=None)
    logits = {
        0: torch.tensor([[5.0, 0.0], [0.0, 5.0]]),
        2: torch.tensor([[0.0, 5.0], [5.0, 0.0]]),
        4: torch.tensor([[5.0, 0.0], [0.0, 5.0]]),
    }
    monkeypatch.setattr(
        evaluation, "_forward",
        lambda model, x, adjacency: logits[int(x[0, 0])].log_softmax(1),
    )
    masks = dict(train=(train, torch.ones(2, dtype=torch.bool)), val=(val, None), test=(test, None))
    metrics = evaluation.inductive_metrics(None, train, torch.eye(2), masks)
    assert metrics["train_acc"] == metrics["val_acc"] == metrics["test_acc"] == 100
    assert "full_teacher_ce" in metrics
    assert "val_teacher_ce" not in metrics
