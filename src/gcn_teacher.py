"""Resumable full-graph GCN teacher with training-only labels and own-graph validation."""
import hashlib
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric import seed_everything

from src.inductive_evaluation import _update_tensor_digest
from src.io import _fingerprint, cpu_state, save_state
from src.models import GCN


def teacher_settings(epochs, seed):
    return dict(epochs=epochs, seed=seed, hidden=256, dropout=0.5, lr=0.01,
                weight_decay=0.0005, layers=2, lr_schedule="constant",
                initialization="pyg", selection="first maximum validation accuracy",
                implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                model_implementation_sha256=hashlib.sha256(Path(__file__).with_name("models.py").read_bytes()).hexdigest(),
                torch_geometric_version=str(torch_geometric.__version__))


def _raw_forward(model, x, adjacency):
    for i, layer in enumerate(model.layers):
        x = layer.lin(x)
        if adjacency is not None:
            x = adjacency @ x if adjacency.layout == torch.strided else torch.sparse.mm(adjacency, x)
        if layer.bias is not None:
            x = x + layer.bias
        if i + 1 < len(model.layers):
            x = F.dropout(F.relu(x), p=model.dropout, training=model.training)
    return x


def _rng_state(device):
    return dict(torch=torch.get_rng_state(), python=random.getstate(), numpy=np.random.get_state(),
                cuda=torch.cuda.get_rng_state_all() if device.type == "cuda" else [])


def _restore_rng(state, device):
    torch.set_rng_state(state["torch"].cpu())
    python = state["python"]
    # cpu_state serializes tuples as lists; Python's internal state requires a tuple.
    random.setstate((python[0], tuple(python[1]), python[2]))
    np.random.set_state(tuple(state["numpy"]))
    if device.type == "cuda":
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def _inputs(graph, train_mask, validation, guard):
    """The excluded source labels never enter class inference or cache identity."""
    if (train_mask.dtype != torch.bool or train_mask.shape != (len(graph["x"]),)
            or not bool(train_mask.any())):
        raise ValueError("GCN teacher requires a nonempty explicit boolean training mask")
    train_labels = graph["y"][train_mask]
    classes = torch.unique(train_labels, sorted=True)
    if (train_labels.dtype != torch.long or not torch.equal(
            classes, torch.arange(len(classes), device=classes.device))):
        raise ValueError("Training vocabulary must be contiguous integer classes starting at zero")
    val_graph, val_mask = validation
    if val_mask is None:
        val_mask = torch.ones(len(val_graph["x"]), dtype=torch.bool, device=val_graph["x"].device)
    if val_mask.dtype != torch.bool or val_mask.shape != (len(val_graph["x"]),) or not bool(val_mask.any()):
        raise ValueError("GCN teacher requires a nonempty boolean validation mask")
    val_labels = val_graph["y"][val_mask]
    if (val_labels.dtype != torch.long or bool((val_labels < 0).any())
            or bool((val_labels >= len(classes)).any())):
        raise ValueError("Validation labels must belong to the training class vocabulary")
    digest = hashlib.sha256()
    for prefix, g, mask, labels in (("train", graph, train_mask, train_labels),
                                    ("validation", val_graph, val_mask, val_labels)):
        for key, tensor in (("x", g["x"]), ("adj", g["adj"]), ("mask", mask), ("labels", labels)):
            guard()
            _update_tensor_digest(digest, prefix + "." + key, tensor)
    return train_labels, val_graph, val_mask, val_labels, classes, digest.hexdigest()


def fit_gcn_teacher(graph, train_mask, validation, folder, *, epochs=200, seed=0, guard=lambda: None):
    """Return selected raw source logits, weights, validation metrics and timings.

    No testing graph is accepted. Source scores cover every node in the supplied
    training graph; only train_mask labels fit the teacher. A stopped requested
    fit leaves an atomic epoch checkpoint and raises InterruptedError, so targets
    from partially completed training cannot enter condensation. Cache identity
    includes source/validation inputs, known class vocabulary and full recipe.
    """
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise ValueError("GCN teacher epochs must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("GCN teacher seed must be a nonnegative integer")
    guard()
    train_labels, val_graph, val_mask, val_labels, classes, digest = _inputs(
        graph, train_mask, validation, guard)
    settings = teacher_settings(epochs, seed)
    recipe = dict(version=1, settings=settings, input_digest=digest,
                  class_vocabulary=classes.cpu().tolist(), torch_version=str(torch.__version__))
    fingerprint = _fingerprint(recipe)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path, resume_path = folder / "teacher.pt", folder / "teacher_resume.pt"
    device = graph["x"].device
    if path.exists():
        saved = torch.load(path, map_location=device, weights_only=False)
        if (saved.get("fingerprint") != fingerprint or saved.get("recipe") != recipe
                or saved.get("training_complete") is not True
                or saved["logits"].shape != (len(graph["x"]), len(classes))
                or not bool(torch.isfinite(saved["logits"]).all())):
            raise ValueError("GCN teacher cache differs from the completed training-only recipe")
        guard()
        return saved
    seed_everything(seed)
    model = GCN(graph["x"].shape[1], settings["hidden"], len(classes), 2, settings["dropout"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"])
    state = dict(epoch=0, best=None, best_state=None, history=[],
                 timings=dict(training_seconds=0.0, validation_seconds=0.0))
    if resume_path.exists():
        state = torch.load(resume_path, map_location=device, weights_only=False)
        if (state.get("fingerprint") != fingerprint or state.get("recipe") != recipe
                or not 0 <= state["epoch"] <= epochs):
            raise ValueError("GCN teacher resume differs from the requested recipe")
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer"])
        _restore_rng(state["rng"], device)
    for epoch in range(state["epoch"] + 1, epochs + 1):
        guard()
        started = time.monotonic()
        model.train()
        optimizer.zero_grad()
        scores = _raw_forward(model, graph["x"], graph["adj"])
        loss = F.cross_entropy(scores[train_mask], train_labels)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Nonfinite GCN teacher training loss")
        loss.backward()
        optimizer.step()
        del scores, loss
        state["timings"]["training_seconds"] += time.monotonic() - started
        guard()
        started = time.monotonic()
        model.eval()
        with torch.no_grad():
            scores = _raw_forward(model, val_graph["x"], val_graph["adj"])[val_mask]
            metrics = dict(epoch=epoch, val_acc=100 * float((scores.argmax(1) == val_labels).double().mean()),
                           val_ce=float(F.cross_entropy(scores, val_labels)), val_nodes=len(val_labels))
        del scores
        state["timings"]["validation_seconds"] += time.monotonic() - started
        if not math.isfinite(metrics["val_ce"]):
            raise RuntimeError("Nonfinite GCN teacher validation loss")
        state["history"].append(metrics)
        if state["best"] is None or metrics["val_acc"] > state["best"]["val_acc"]:
            state.update(best=metrics, best_state=cpu_state(model.state_dict()))
        state.update(epoch=epoch, model_state=model.state_dict(), optimizer=optimizer.state_dict(),
                     rng=_rng_state(device), recipe=recipe, fingerprint=fingerprint)
        save_state(state, resume_path)
    guard()
    model.load_state_dict(state["best_state"])
    model.eval()
    started = time.monotonic()
    with torch.no_grad():
        logits = _raw_forward(model, graph["x"], graph["adj"])
    guard()
    result = dict(training_complete=True, fingerprint=fingerprint, recipe=recipe, logits=logits,
                  selected_state=state["best_state"], selected_validation=state["best"],
                  history=state["history"], timings=dict(state["timings"],
                                                       source_logits_seconds=time.monotonic() - started))
    save_state(result, path)
    guard()
    return result
