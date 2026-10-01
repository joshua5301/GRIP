"""Replay two serving routes with one saved, validation-selected GCN student.

No training or route-specific epoch selection occurs. Supply propagated S²X in
the same input scaling as graph X; both routes restore the very same weights.
"""

import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from src.evaluation import _forward, _update_tensor_digest
from src.io import _fingerprint, save_json
from src.models import GCN

SELECTION = "same weights at GCN validation-selected epoch"


def replay_routes(selected_path, graph, propagated, masks, settings, output_path, seed=0,
                  stop=lambda: False, test_only=False):
    """Return paired MLP/GCN scores and save a fingerprinted JSON cache.

    ``selected_path`` is seed_*_selected.pt from fit_gcn_diagnostic. ``masks``
    contains an explicit val mask and optionally test; no training labels are
    needed. A validation-only call does not read test labels. Settings require
    hidden and dropout; layers, when provided, must be 2. Additional original
    training settings are retained as provenance. Input/output widths come from
    the saved weights. ``output_path`` names the JSON file, not a directory.

    Epoch selection is inherited from the checkpoint, even if MLP validation
    accuracy is lower. Reusing an output path with changed inputs, weights,
    selected epoch or settings raises ValueError. ``test_only=True`` instead
    requires exactly one explicit test mask and reads/hashes only its labels;
    the saved weights determine the class count. It records a distinct recipe
    flag without changing default replay identities or epoch selection.
    """
    if stop():
        raise InterruptedError("Student route replay interrupted")
    if not isinstance(test_only, bool):
        raise ValueError("test_only must be a boolean")
    if test_only and set(masks) != {"test"}:
        raise ValueError("test_only requires exactly a test mask")
    if not test_only and set(masks) not in ({"val"}, {"val", "test"}):
        raise ValueError("masks must contain val and optionally test")
    settings = dict(settings)
    if not {"hidden", "dropout"} <= settings.keys() or settings.get("layers", 2) != 2:
        raise ValueError("Replay settings require hidden, dropout and two layers")
    if not isinstance(settings["hidden"], int) or settings["hidden"] < 1 or not 0 <= settings["dropout"] < 1:
        raise ValueError("Invalid hidden width or dropout")
    selected_path = Path(selected_path)
    selected = torch.load(selected_path, map_location="cpu", weights_only=False)
    if not isinstance(selected, dict) or not {"model_state", "epoch", "fingerprint"} <= selected.keys():
        raise ValueError("Expected a fingerprinted validation-selected model checkpoint")
    state = selected["model_state"]
    weight_keys = sorted(name for name in state if name.endswith(".lin.weight"))
    if weight_keys != ["layers.0.lin.weight", "layers.1.lin.weight"]:
        raise ValueError("Route replay requires the saved two-layer GCN")
    first, last = (state[key] for key in weight_keys)
    if first.ndim != 2 or last.ndim != 2 or first.shape[0] != last.shape[1]:
        raise ValueError("Incompatible saved GCN layer dimensions")
    nin, hidden, nout = first.shape[1], first.shape[0], last.shape[0]
    if hidden != settings["hidden"] or not isinstance(selected["epoch"], int) or selected["epoch"] < 1:
        raise ValueError("Settings or selected epoch differ from the saved architecture")
    x, adjacency = graph["x"], graph["adj"]
    if x.ndim != 2 or x.shape[1] != nin or propagated.shape != x.shape or adjacency.shape != (len(x), len(x)):
        raise ValueError("Original X, propagated S²X and adjacency must match the saved model")

    digest = hashlib.sha256()
    for name, value in (("graph.x", x), ("graph.adj", adjacency), ("propagated", propagated)):
        _update_tensor_digest(digest, name, value, stop)
    for name, tensor in sorted(state.items()):
        _update_tensor_digest(digest, "state." + name, tensor, stop)
    labels = {}
    for name, mask in sorted(masks.items()):
        if mask is None or not torch.is_tensor(mask) or mask.ndim != 1:
            raise ValueError("Use explicit nonempty split masks or node indices")
        if mask.dtype == torch.bool and len(mask) != len(x):
            raise ValueError("Boolean masks must match the original nodes")
        if mask.dtype not in (torch.bool, torch.int64, torch.int32):
            raise ValueError("Split masks must be boolean or integer node indices")
        split_labels = graph["y"][mask]
        if split_labels.ndim != 1 or len(split_labels) == 0:
            raise ValueError("Each requested split must contain integer class labels")
        if split_labels.dtype not in (torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8):
            raise ValueError("Each requested split must contain integer class labels")
        if bool((split_labels < 0).any()) or bool((split_labels >= nout).any()):
            raise ValueError("Requested split labels exceed the saved model's classes")
        labels[name] = split_labels
        _update_tensor_digest(digest, "mask." + name, mask, stop)
        _update_tensor_digest(digest, "labels." + name, split_labels, stop)
    recipe = dict(
        version=1, seed=seed, settings=settings, architecture=dict(nin=nin, hidden=hidden, nout=nout, layers=2),
        epoch=selected["epoch"], source_fingerprint=selected["fingerprint"],
        selection=SELECTION, routes=dict(gcn="original X and normalized adjacency", mlp="supplied S²X; no adjacency"),
        test_enabled="test" in masks, input_digest=digest.hexdigest(), torch_version=torch.__version__,
    )
    if test_only:
        recipe["test_only"] = True
    fingerprint = _fingerprint(recipe)
    output_path = Path(output_path)
    if output_path.exists():
        cached = json.loads(output_path.read_text())
        if cached.get("fingerprint") != fingerprint or cached.get("recipe") != recipe:
            raise ValueError("Cached route replay differs from the selected state, inputs or settings")
        return cached["result"]
    if stop():
        raise InterruptedError("Student route replay interrupted before serving")
    # GCN's constructor initializes parameters; do not perturb the experiment's RNG.
    with torch.random.fork_rng(devices=[]):
        model = GCN(nin, hidden, nout, 2, settings["dropout"])
    model = model.to(device=x.device, dtype=first.dtype)
    model.load_state_dict(state)
    model.eval()
    parameter = next(model.parameters())
    result = dict(seed=seed, epoch=selected["epoch"], selection=SELECTION)
    with torch.no_grad():
        for route, features, adj in (("gcn", x, adjacency), ("mlp", propagated, None)):
            if stop():
                raise InterruptedError("Student route replay interrupted before serving")
            probability = _forward(model, features.to(parameter), adj.to(parameter) if adj is not None else None)
            for name, mask in masks.items():
                prediction = probability[mask.to(device=parameter.device)]
                target = labels[name].to(device=parameter.device, dtype=torch.long)
                result[f"{route}_{name}_acc"] = 100 * float((prediction.argmax(1) == target).double().mean())
                result[f"{route}_{name}_ce"] = float(F.nll_loss(prediction, target))
    if stop():
        raise InterruptedError("Student route replay interrupted before saving")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(dict(fingerprint=fingerprint, recipe=recipe, result=result), output_path)
    return result
