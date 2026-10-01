"""Freeze shared SGC coordinates and Nyström maps across condensation seeds.

The source digest for H must cover the raw graph/features AND preprocessing
settings (normalization, propagation depth, etc.). Equal source provenance
deliberately reuses the first H even if a later GPU sparse product rounds
slightly differently. Nyström anchors and the Cholesky-derived mapping are
then persisted together, before any original-node feature cache is created.
"""
import re
from pathlib import Path

import torch

from src.io import save_state
from src.nystrom_ce import NystromMap, _content_digest


def _validate_matrix(value, name):
    if not torch.is_tensor(value) or value.ndim != 2 or min(value.shape) < 1:
        raise ValueError(f"{name} must be a nonempty two-dimensional tensor")
    if not value.is_floating_point():
        raise ValueError(f"{name} must have a floating dtype")
    for block in value.detach().split(2048):
        if not torch.isfinite(block).all():
            raise ValueError(f"{name} must contain only finite values")


def _load_state(path, kind):
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"Invalid or unknown legacy {kind} cache; preserve it and use a new path") from exc
    if not isinstance(state, dict) or state.get("schema") != 1 or state.get("kind") != kind:
        raise ValueError(f"Invalid or unknown legacy {kind} cache; preserve it and use a new path")
    return state


def _tensor_identity(value):
    return dict(shape=list(value.shape), dtype=str(value.dtype), digest=_content_digest(value))


@torch.no_grad()
def get_shared_h(h, path, source_digest):
    """Return the first persisted H for this exact source, on ``h.device``.

    Current H shape/dtype/finite values are checked; its numerical contents are
    intentionally ignored on reuse. ``source_digest`` must identify raw data
    and the complete preprocessing protocol, rather than recomputed H bytes.
    Existing incompatible or unverifiable files are never overwritten.
    """
    _validate_matrix(h, "H")
    if not isinstance(source_digest, str) or not source_digest.strip():
        raise ValueError("source_digest must be a nonempty stable source/protocol digest")
    path = Path(path)
    if path.exists():
        state = _load_state(path, "shared_h")
        if state.get("source_digest") != source_digest:
            raise ValueError("Shared H source digest differs; preserve it and use a new path")
        saved = state.get("h")
        _validate_matrix(saved, "Saved H")
        if state.get("identity") != _tensor_identity(saved):
            raise ValueError("Shared H content or metadata fingerprint differs")
        if saved.shape != h.shape or saved.dtype != h.dtype:
            raise ValueError("Shared H shape or dtype differs from the requested H")
        return saved.to(device=h.device)
    frozen = h.detach().cpu().clone()
    state = dict(schema=1, kind="shared_h", source_digest=source_digest,
                 identity=_tensor_identity(frozen), h=frozen)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_state(state, path)
    return frozen.to(device=h.device)


def _map_identity(h, basis, seed, kernel):
    if not isinstance(basis, int) or isinstance(basis, bool) or basis < 1:
        raise ValueError("Nyström basis must be a positive integer")
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**63:
        raise ValueError("Nyström seed must be an integer in [0, 2**63)")
    if not isinstance(kernel, str) or not (
        kernel in {"erf", "rbf", "linear"} or re.fullmatch(r"relu\d*", kernel)
    ):
        raise ValueError("Unsupported Nyström kernel")
    return dict(h_digest=_content_digest(h, canonical_double=True), h_shape=list(h.shape),
                basis=basis, seed=seed, kernel=kernel)


def _validate_map_state(state, identity):
    if state.get("identity") != identity:
        raise ValueError("Shared Nyström map H/basis/seed/kernel fingerprint differs")
    anchors, mapping = state.get("anchors"), state.get("mapping")
    _validate_matrix(anchors, "Saved Nyström anchors")
    _validate_matrix(mapping, "Saved Nyström mapping")
    count = min(identity["basis"], identity["h_shape"][0])
    if (tuple(anchors.shape) != (count, identity["h_shape"][1]) or
            tuple(mapping.shape) != (count, count) or
            anchors.dtype != torch.double or mapping.dtype != torch.double):
        raise ValueError("Shared Nyström map shape or dtype differs")
    if (state.get("anchors_identity") != _tensor_identity(anchors) or
            state.get("mapping_identity") != _tensor_identity(mapping)):
        raise ValueError("Shared Nyström map content or metadata fingerprint differs")
    return anchors, mapping


@torch.no_grad()
def get_shared_map(h, path, basis=3000, seed=0, kernel="relu"):
    """Load exact saved anchors/mapping, fitting only when ``path`` is absent.

    Reuse validates H and all requested map settings without recomputing a
    kernel Gram matrix or Cholesky decomposition. Unknown legacy maps and
    incompatible/corrupt files are preserved and rejected.
    """
    _validate_matrix(h, "H")
    identity = _map_identity(h, basis, seed, kernel)
    path = Path(path)
    if path.exists():
        state = _load_state(path, "shared_nystrom_map")
    else:
        fitted = NystromMap.fit(h, basis=basis, seed=seed, kernel=kernel)
        anchors, mapping = fitted.anchors.detach().cpu().clone(), fitted.mapping.detach().cpu().clone()
        state = dict(schema=1, kind="shared_nystrom_map", identity=identity,
                     anchors=anchors, mapping=mapping,
                     anchors_identity=_tensor_identity(anchors), mapping_identity=_tensor_identity(mapping))
        _validate_map_state(state, identity)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_state(state, path)
    anchors, mapping = _validate_map_state(state, identity)
    return NystromMap(anchors.to(device=h.device), mapping.to(device=h.device), kernel)
