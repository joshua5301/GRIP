"""Fingerprint-checked GCN evaluation with validation-only epoch selection.

Legacy score-only caches are retrained and preserved under a ``legacy`` name.
Test metrics use the selected model once and never appear in epoch histories.
"""

import hashlib
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric import seed_everything

from src.io import _fingerprint, save_json, save_state, write_table
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
def _mask_metrics(log_probability, y, q, mask):
    prediction, labels, teacher = (
        (log_probability, y, q) if mask is None else (log_probability[mask], y[mask], q[mask])
    )
    if prediction.ndim != 2 or labels.ndim != 1 or len(prediction) != len(labels) or len(labels) == 0:
        raise ValueError("Evaluation requires a nonempty split of integer class labels")
    return dict(
        acc=100 * float((prediction.argmax(1) == labels).double().mean()),
        ce=float(F.nll_loss(prediction, labels)),
        teacher_ce=float(-(teacher * prediction).sum(1).mean()),
    )


@torch.no_grad()
def split_metrics(log_probability, y, q, masks):
    values = dict(full_teacher_ce=float(-(q * log_probability).sum(1).mean()))
    for name, mask in masks.items():
        values.update({f"{name}_{key}": value for key, value in _mask_metrics(log_probability, y, q, mask).items()})
    values["val_minus_train_ce"] = values["val_ce"] - values["train_ce"]
    values["train_minus_val_acc"] = values["train_acc"] - values["val_acc"]
    return values


def _update_tensor_digest(digest, name, tensor, stop, chunk_size=1_048_576):
    if stop():
        raise InterruptedError("Student evaluation interrupted while fingerprinting inputs")
    if tensor is None:
        digest.update(f"{name}:None".encode())
        return
    digest.update(f"{name}:{tuple(tensor.shape)}:{tensor.dtype}:{tensor.layout}".encode())
    if tensor.layout == torch.sparse_coo:
        tensor = tensor.coalesce()
        _update_tensor_digest(digest, name + ".indices", tensor.indices(), stop, chunk_size)
        _update_tensor_digest(digest, name + ".values", tensor.values(), stop, chunk_size)
        return
    if tensor.layout == torch.sparse_csr:
        _update_tensor_digest(digest, name + ".crow", tensor.crow_indices(), stop, chunk_size)
        _update_tensor_digest(digest, name + ".col", tensor.col_indices(), stop, chunk_size)
        _update_tensor_digest(digest, name + ".values", tensor.values(), stop, chunk_size)
        return
    if tensor.layout != torch.strided:
        raise ValueError("Use dense, sparse COO or sparse CSR adjacency")
    values = tensor.detach().reshape(-1)
    for start in range(0, len(values), chunk_size):
        if stop():
            raise InterruptedError("Student evaluation interrupted while fingerprinting inputs")
        digest.update(values[start : start + chunk_size].cpu().contiguous().numpy().tobytes())


def _input_digest(cx, cy, mass, graph, q, masks, training_adjacency, stop):
    digest = hashlib.sha256()
    for name, tensor in (("cx", cx), ("cy", cy), ("mass", mass), ("q", q),
                         ("training_adjacency", training_adjacency)):
        _update_tensor_digest(digest, name, tensor, stop)
    for name in ("x", "y", "adj"):
        _update_tensor_digest(digest, "graph." + name, graph[name], stop)
    for name, mask in sorted(masks.items()):
        _update_tensor_digest(digest, "mask." + name, mask, stop)
    return digest.hexdigest()


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
    stop=lambda: False,
):
    """Return scores for the first maximum-validation epoch of one student.

    Reusing a folder/seed with changed inputs or settings raises ValueError.
    Score-only legacy caches are retrained; their JSON and epoch history are
    archived after the new fit succeeds. Optional test metrics are evaluated
    only after restoring validation-selected weights; ``last_test_*`` values
    are intentionally absent. ``stop`` interrupts without writing a new cache.
    """
    settings = dict(epochs=epochs, eval_every=eval_every, hidden=hidden, dropout=dropout,
                    lr=lr, weight_decay=weight_decay)
    if any(not isinstance(settings[key], int) or settings[key] < 1 for key in ("epochs", "eval_every", "hidden")):
        raise ValueError("epochs, eval_every and hidden must be positive integers")
    if not 0 <= dropout < 1 or lr <= 0 or weight_decay < 0:
        raise ValueError("Invalid dropout, learning rate or weight decay")
    if set(masks) not in ({"train", "val"}, {"train", "val", "test"}):
        raise ValueError("masks must contain train, val and optionally test")
    if cx.ndim != 2 or cy.ndim != 2 or len(cx) != len(cy) or mass.shape != (len(cx),) or len(cx) == 0:
        raise ValueError("Condensed features, labels and masses must have matching nonempty rows")
    if not bool(torch.isfinite(mass).all()) or bool((mass < 0).any()) or float(mass.sum()) <= 0:
        raise ValueError("mass must be nonnegative, finite and have positive total")
    recipe = dict(
        version=2, seed=seed, settings=settings, layers=2,
        optimizer="Adam; reset to lr/10 at epochs//2", weighting="normalized supplied mass",
        selection="first maximum validation accuracy", test_enabled="test" in masks,
        test_evaluation="selected weights once", torch_version=torch.__version__,
        input_digest=_input_digest(cx, cy, mass, graph, q, masks, training_adjacency, stop),
    )
    fingerprint = _fingerprint(recipe)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"seed_{seed}.json"
    legacy = None
    if path.exists():
        cached = json.loads(path.read_text())
        if "fingerprint" in cached:
            if cached.get("fingerprint") != fingerprint or cached.get("recipe") != recipe:
                raise ValueError("Cached student differs from the recipe, condensed inputs or graph splits")
            return cached["result"]
        legacy = cached
    if stop():
        raise InterruptedError("Student evaluation interrupted before training")
    seed_everything(seed)
    model = GCN(cx.shape[1], hidden, cy.shape[1], 2, dropout).to(cx.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    weights = (mass / mass.sum()).to(cx)
    search_masks = {name: masks[name] for name in ("train", "val")}
    history, best, best_state, best_value = [], None, None, -float("inf")
    for epoch in range(1, epochs + 1):
        if stop():
            raise InterruptedError("Student evaluation interrupted")
        if epoch == epochs // 2:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr * 0.1, weight_decay=weight_decay)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = -(weights[:, None] * cy * _forward(model, cx, training_adjacency)).sum()
        loss.backward()
        optimizer.step()
        if stop():
            raise InterruptedError("Student evaluation interrupted before epoch diagnostics")
        if epoch % eval_every != 0 and epoch != epochs:
            continue
        model.eval()
        with torch.no_grad():
            log_probability = _forward(model, graph["x"], graph["adj"])
            row = dict(
                epoch=epoch,
                **split_metrics(log_probability, graph["y"], q, search_masks),
                condensed_ce=float(-(weights[:, None] * cy * _forward(model, cx, training_adjacency)).sum()),
            )
        history.append(row)
        if row["val_acc"] > best_value:
            best, best_value = dict(row), row["val_acc"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if stop():
        raise InterruptedError("Student evaluation interrupted before selected-model evaluation")
    model.load_state_dict(best_state)
    model.eval()
    result = dict(seed=seed, **best, **{f"last_{key}": value for key, value in history[-1].items()})
    if "test" in masks:
        with torch.no_grad():
            log_probability = _forward(model, graph["x"], graph["adj"])
            result.update({f"test_{key}": value for key, value in
                           _mask_metrics(log_probability, graph["y"], q, masks["test"]).items()})
    history_path = folder / f"seed_{seed}_epochs.csv"
    if legacy is not None:
        legacy_key = _fingerprint(legacy)
        save_json(legacy, folder / f"seed_{seed}_legacy_{legacy_key}.json")
        if history_path.exists():
            archive = folder / f"seed_{seed}_legacy_{legacy_key}_epochs.csv"
            if not archive.exists():
                archive.write_bytes(history_path.read_bytes())
    save_state(dict(epoch=best["epoch"], model_state=best_state, fingerprint=fingerprint),
               folder / f"seed_{seed}_selected.pt")
    write_table(pd.DataFrame(history), history_path)
    save_json(dict(fingerprint=fingerprint, recipe=recipe, result=result), path)
    return result
