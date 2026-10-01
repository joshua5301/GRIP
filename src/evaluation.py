import json

import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric import seed_everything

from src.io import save_json
from src.models import GCN


def _forward(model, x, adjacency=None):
    for i, layer in enumerate(model.layers):
        x = layer.lin(x)
        if adjacency is not None:
            x = adjacency @ x if adjacency.layout == torch.strided else torch.sparse.mm(adjacency, x)
        if layer.bias is not None:
            x = x + layer.bias
        if i + 1 < len(model.layers):
            x = F.dropout(F.relu(x), p=model.dropout, training=model.training)
    return F.log_softmax(x, dim=1)


@torch.no_grad()
def split_metrics(log_probability, y, q, masks):
    values = {} if q is None else dict(full_teacher_ce=float(-(q * log_probability).sum(1).mean()))
    for name, mask in masks.items():
        prediction = log_probability[mask]
        values[f"{name}_acc"] = 100 * float((prediction.argmax(1) == y[mask]).double().mean())
        values[f"{name}_ce"] = float(F.nll_loss(prediction, y[mask]))
        if q is not None:
            values[f"{name}_teacher_ce"] = float(-(q[mask] * prediction).sum(1).mean())
    values["val_minus_train_ce"] = values["val_ce"] - values["train_ce"]
    values["train_minus_val_acc"] = values["train_acc"] - values["val_acc"]
    return values


def inductive_metrics(model, graph, q, masks):
    predictions = {}
    values = {}
    for name, (split_graph, mask) in masks.items():
        key = id(split_graph)
        if key not in predictions:
            predictions[key] = _forward(model, split_graph["x"], split_graph["adj"])
        prediction = predictions[key] if mask is None else predictions[key][mask]
        labels = split_graph["y"] if mask is None else split_graph["y"][mask]
        values[f"{name}_acc"] = 100 * float((prediction.argmax(1) == labels).double().mean())
        values[f"{name}_ce"] = float(F.nll_loss(prediction, labels))
        if split_graph is graph and q is not None:
            targets = q if mask is None else q[mask]
            values[f"{name}_teacher_ce"] = float(-(targets * prediction).sum(1).mean())
    if q is not None:
        values["full_teacher_ce"] = float(-(q * predictions[id(graph)]).sum(1).mean())
    values["val_minus_train_ce"] = values["val_ce"] - values["train_ce"]
    values["train_minus_val_acc"] = values["train_acc"] - values["val_acc"]
    return values


def fit_gcn_diagnostic(
    cx,
    cy,
    mass,
    graph,
    q,
    masks,
    seed,
    epochs,
    eval_every,
    hidden,
    dropout,
    lr,
    weight_decay,
    folder,
    training_adjacency=None,
):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"seed_{seed}.json"
    if path.exists():
        return json.loads(path.read_text())
    seed_everything(seed)
    model = GCN(cx.shape[1], hidden, cy.shape[1], 2, dropout).to(cx.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    weights = (mass / mass.sum()).to(cx)
    history, best, best_value = [], None, -float("inf")
    for epoch in range(1, epochs + 1):
        if epoch == epochs // 2:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr * 0.1, weight_decay=weight_decay)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = -(weights[:, None] * cy * _forward(model, cx, training_adjacency)).sum()
        loss.backward()
        optimizer.step()
        if epoch % eval_every != 0 and epoch != epochs:
            continue
        model.eval()
        with torch.no_grad():
            metrics = (
                inductive_metrics(model, graph, q, masks)
                if isinstance(masks["val"], tuple)
                else split_metrics(_forward(model, graph["x"], graph["adj"]), graph["y"], q, masks)
            )
            row = dict(
                epoch=epoch,
                **metrics,
                condensed_ce=float(-(weights[:, None] * cy * _forward(model, cx, training_adjacency)).sum()),
            )
        history.append(row)
        if row["val_acc"] > best_value:
            best, best_value = dict(row), row["val_acc"]
    result = dict(seed=seed, **best, **{f"last_{key}": value for key, value in history[-1].items()})
    pd.DataFrame(history).to_csv(folder / f"seed_{seed}_epochs.csv", index=False)
    save_json(result, path)
    return result
