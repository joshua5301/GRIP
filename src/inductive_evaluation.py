"""GCN student evaluation on separate induced train/validation/test graphs.

Flickr and Reddit must use each split's own features and normalized adjacency.
Validation chooses the student epoch; optional test evaluation happens once,
after restoring those selected weights.  Validation-only runs do not require
or inspect a testing graph.  The training recipe matches the existing fitter:
two GCN layers, Adam, and an optimizer reset to lr/10 halfway through training.
"""

import hashlib
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric import seed_everything

from src.evaluation import _forward
from src.io import _fingerprint, save_json, save_state, write_table
from src.models import GCN


def _update_tensor_digest(digest, name, tensor, chunk_size=1_048_576):
    """Hash sparse/dense tensors without a full large-graph CPU allocation."""
    if tensor is None:
        digest.update(f"{name}:None".encode())
        return
    digest.update(f"{name}:{tuple(tensor.shape)}:{tensor.dtype}:{tensor.layout}".encode())
    if tensor.layout == torch.sparse_coo:
        tensor = tensor.coalesce()
        _update_tensor_digest(digest, name + ".indices", tensor.indices(), chunk_size)
        _update_tensor_digest(digest, name + ".values", tensor.values(), chunk_size)
        return
    if tensor.layout == torch.sparse_csr:
        _update_tensor_digest(digest, name + ".crow", tensor.crow_indices(), chunk_size)
        _update_tensor_digest(digest, name + ".col", tensor.col_indices(), chunk_size)
        _update_tensor_digest(digest, name + ".values", tensor.values(), chunk_size)
        return
    if tensor.layout != torch.strided:
        raise ValueError("Use dense, sparse COO or sparse CSR graph adjacency")
    values = tensor.detach().reshape(-1)
    for start in range(0, len(values), chunk_size):
        block = values[start : start + chunk_size].cpu().contiguous().numpy()
        digest.update(block.tobytes())


def _input_digest(cx, cy, mass, training_adjacency, train_graph, validation, testing, train_mask=None):
    digest = hashlib.sha256()
    for name, value in (("cx", cx), ("cy", cy), ("mass", mass), ("training_adjacency", training_adjacency)):
        _update_tensor_digest(digest, name, value)
    splits = [("train", (train_graph, train_mask)), ("validation", validation)]
    if testing is not None:
        splits.append(("testing", testing))
    for name, (graph, mask) in splits:
        for key in ("x", "y", "adj"):
            value = graph[key]
            # Explicit transductive masks exclude held-out labels from both
            # diagnostics and cache identity. Preserve legacy None-mask hashes.
            if key == "y" and train_mask is not None and mask is not None:
                value = value[mask]
            _update_tensor_digest(digest, name + "." + key, value)
        _update_tensor_digest(digest, name + ".mask", mask)
    return digest.hexdigest()


@torch.no_grad()
def graph_metrics(model, graph, mask=None):
    """Evaluate the supplied graph's own X/A; mask=None means its whole graph."""
    parameter = next(model.parameters())
    x = graph["x"].to(parameter)
    adjacency = graph["adj"].to(parameter)
    labels = graph["y"].to(device=parameter.device, dtype=torch.long)
    probability = _forward(model, x, adjacency)
    if mask is not None:
        mask = mask.to(device=parameter.device)
        probability, labels = probability[mask], labels[mask]
    if probability.ndim != 2 or labels.ndim != 1 or len(probability) != len(labels) or len(labels) == 0:
        raise ValueError("Evaluation requires a nonempty split of integer class labels")
    return dict(
        acc=100 * float((probability.argmax(1) == labels).double().mean()),
        ce=float(F.nll_loss(probability, labels)),
        nodes=len(labels),
    )


def fit_inductive_gcn(
    cx,
    cy,
    mass,
    train_graph,
    validation,
    testing=None,
    seed=0,
    settings=None,
    folder=None,
    training_adjacency=None,
    stop=lambda: False,
    weighting="uniform",
    train_mask=None,
):
    """Fit one fresh student and evaluate the validation-selected weights.

    ``validation`` and optional ``testing`` are (graph, mask_or_None) tuples.  Graph
    dictionaries contain x, y and adj.  Supply already normalized adjacencies, and
    use None for synthetic adjacency to train the existing identity-graph student.
    Uniform condensed CE is the default; weighting='mass' uses normalized masses.
    For transductive graphs, supply train_mask to restrict train diagnostics to
    training nodes. Explicit masks also keep excluded labels out of cache hashes.
    train_mask=None preserves the previous whole-train-graph behavior and cache.

    settings must contain epochs, eval_every, hidden, dropout, lr and weight_decay.
    The first epoch attaining maximum validation accuracy is selected.  History
    contains validation metrics only; testing labels never affect epoch selection.
    Returned test_acc/test_ce, when requested, use that selected epoch's weights.

    JSON caches include recipe, inputs, all supplied split graphs and masks.  Reuse
    of the same folder/seed with different settings or inputs raises ValueError;
    use a distinct folder for a changed experiment or a final test-enabled run.
    The selected model is saved for review, alongside validation history.
    """
    required = {"epochs", "eval_every", "hidden", "dropout", "lr", "weight_decay"}
    if settings is None or set(settings) != required:
        raise ValueError("settings must contain epochs, eval_every, hidden, dropout, lr and weight_decay")
    settings = dict(settings)
    if any(
        not isinstance(settings[key], int) or settings[key] < 1 for key in ("epochs", "eval_every", "hidden")
    ):
        raise ValueError("epochs, eval_every and hidden must be positive integers")
    if not 0 <= settings["dropout"] < 1 or settings["lr"] <= 0 or settings["weight_decay"] < 0:
        raise ValueError("Invalid dropout, learning rate or weight decay")
    if weighting not in ("uniform", "mass"):
        raise ValueError("weighting must be uniform or mass")
    if cx.ndim != 2 or cy.ndim != 2 or len(cx) != len(cy) or mass.shape != (len(cx),) or len(cx) == 0:
        raise ValueError("Condensed features, labels and masses must have matching nonempty rows")
    if not bool(torch.isfinite(mass).all()) or bool((mass < 0).any()) or float(mass.sum()) <= 0:
        raise ValueError("mass must be nonnegative, finite and have positive total")
    if folder is None:
        raise ValueError("A result folder is required for fingerprinted evaluation caching")
    recipe = dict(
        version=1,
        seed=seed,
        settings=settings,
        weighting=weighting,
        layers=2,
        selection="first maximum validation accuracy",
        test_enabled=testing is not None,
        torch_version=torch.__version__,
        input_digest=_input_digest(
            cx, cy, mass, training_adjacency, train_graph, validation, testing, train_mask=train_mask
        ),
    )
    fingerprint = _fingerprint(recipe)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    cache_path = folder / f"seed_{seed}.json"
    if cache_path.exists():
        cached = json.loads(cache_path.read_text())
        if cached.get("fingerprint") != fingerprint or cached.get("recipe") != recipe:
            raise ValueError("Cached student differs from the recipe, condensed inputs or graph splits")
        return cached["result"]
    seed_everything(seed)
    cx = cx.detach().float()
    cy = cy.detach().to(cx)
    training_adjacency = training_adjacency.to(cx) if training_adjacency is not None else None
    weights = cx.new_full((len(cx),), 1 / len(cx)) if weighting == "uniform" else (mass / mass.sum()).to(cx)
    model = GCN(cx.shape[1], settings["hidden"], cy.shape[1], 2, settings["dropout"]).to(cx.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"])
    history, best, best_state, best_value = [], None, None, -float("inf")
    for epoch in range(1, settings["epochs"] + 1):
        if stop():
            raise InterruptedError("Inductive student evaluation interrupted")
        if epoch == settings["epochs"] // 2:
            optimizer = torch.optim.Adam(
                model.parameters(), lr=settings["lr"] * 0.1, weight_decay=settings["weight_decay"]
            )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = -(weights[:, None] * cy * _forward(model, cx, training_adjacency)).sum()
        loss.backward()
        optimizer.step()
        if epoch % settings["eval_every"] != 0 and epoch != settings["epochs"]:
            continue
        model.eval()
        validation_metrics = graph_metrics(model, *validation)
        row = dict(epoch=epoch, **{f"val_{key}": value for key, value in validation_metrics.items()})
        history.append(row)
        if row["val_acc"] > best_value:
            best, best_value = dict(row), row["val_acc"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if stop():
        raise InterruptedError("Inductive student evaluation interrupted before selected-model evaluation")
    model.load_state_dict(best_state)
    model.eval()
    result = dict(
        seed=seed, weighting=weighting, **best, **{f"last_{key}": value for key, value in history[-1].items()}
    )
    train_metrics = graph_metrics(model, train_graph, train_mask)
    result.update({f"train_{key}": value for key, value in train_metrics.items()})
    if testing is not None:
        test_metrics = graph_metrics(model, *testing)
        result.update({f"test_{key}": value for key, value in test_metrics.items()})
    with torch.no_grad():
        result["condensed_ce"] = float(
            -(weights[:, None] * cy * _forward(model, cx, training_adjacency)).sum()
        )
    save_state(
        dict(epoch=best["epoch"], model_state=best_state, fingerprint=fingerprint),
        folder / f"seed_{seed}_selected.pt",
    )
    write_table(pd.DataFrame(history), folder / f"seed_{seed}_epochs.csv")
    save_json(dict(fingerprint=fingerprint, recipe=recipe, result=result), cache_path)
    return result
