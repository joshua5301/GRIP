"""Frozen source-P0 prototype geometry for native NODE factor initialization.

Preparation is CPU-only and consumes pinned native moment/factor tensors. The
production loader copies the immutable FP32 factor; it never recreates an RNG,
centroid or eigenspace. No labels or free representative features are accepted.
"""
import hashlib
import json
import math
from pathlib import Path

import torch

from src.io import array_digest

SCHEMA = 1
MODES = ("prototype_gram_v1", "centered_gaussian_v1")
_CONTEXT_KEYS = ("schema", "mode", "rank", "dimensions", "source_refs", "context_digest", "factor_digest")
_PACKET_KEYS = set(_CONTEXT_KEYS) | {"factor", "input_digests", "diagnostics"}


def _json_copy(value):
    try:
        return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError("Factor context must be finite JSON metadata") from error


def _seal(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _tensor(value, shape, dtype, name):
    if (not torch.is_tensor(value) or value.shape != shape or value.dtype != dtype
            or value.device.type != "cpu" or value.requires_grad
            or not bool(torch.isfinite(value).all())):
        raise ValueError(f"{name} must be a detached, finite CPU {dtype} tensor of shape {tuple(shape)}")


def _inputs(initial_moments, original_v, mode, rank):
    if mode not in MODES or type(rank) is not int or rank < 1:
        raise ValueError("Require an explicit supported factor mode and positive integer rank")
    if not isinstance(initial_moments, dict) or set(initial_moments) != {"mass", "features"}:
        raise ValueError("Require native P0 mass and mass-times-feature moments only")
    mass, features = initial_moments["mass"], initial_moments["features"]
    if (not torch.is_tensor(features) or features.ndim != 2 or min(features.shape) < 1):
        raise ValueError("Require a nonempty cell-by-feature native moment matrix")
    cells, dimension = features.shape
    _tensor(mass, torch.Size((cells,)), torch.float64, "Initial mass")
    _tensor(features, torch.Size((cells, dimension)), torch.float64, "Initial feature totals")
    _tensor(original_v, torch.Size((cells, rank)), torch.float32, "Original native V")
    if cells < 2 or rank > min(cells - 1, dimension) or bool((mass <= 0).any()):
        raise ValueError("Require positive P0 masses and rank no larger than min(cells-1, features)")
    return mass.detach().clone(), features.detach().clone(), original_v.detach().clone()


def generate_factor(initial_moments, original_v, mode, rank):
    """Return CPU FP32 V and diagnostics from native P0 moments and pinned V.

    ``features`` contains native mass-times-centroid totals, not centroids.
    Both arms require the same qualified full-rank source geometry. The fixed
    numerical-rank threshold is eps64*max(K,D)*lambda_max; no fallback or rank
    reduction is permitted. The Gaussian arm centers the supplied native V.
    """
    mass, feature_totals, original_v = _inputs(initial_moments, original_v, mode, rank)
    cells, dimension = feature_totals.shape
    centers = feature_totals / mass[:, None]
    centered = centers - centers.mean(0, keepdim=True)
    gram = centered @ centered.T
    if not bool(torch.isfinite(gram).all()):
        raise ValueError("Source prototype Gram is nonfinite")
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    eigenvalues, eigenvectors = eigenvalues.flip(0), eigenvectors.flip(1)
    maximum = float(eigenvalues[0])
    threshold = torch.finfo(torch.float64).eps * max(cells, dimension) * maximum
    if (not math.isfinite(maximum) or maximum <= 0 or float(eigenvalues[-1]) < -threshold
            or float(eigenvalues[rank - 1]) <= threshold):
        raise ValueError("Source prototype Gram cannot support the requested numerical rank")
    pivots = None
    if mode == "prototype_gram_v1":
        vectors = eigenvectors[:, :rank].clone()
        pivots = vectors.abs().argmax(0)
        signs = vectors[pivots, torch.arange(rank)].sign()
        vectors *= signs[None, :]
        unscaled = vectors * eigenvalues[:rank].sqrt()[None, :]
    else:
        unscaled = original_v.double() - original_v.double().mean(0, keepdim=True)
    norm = float(unscaled.norm())
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("Centered factor has no finite nonzero energy")
    factor = (unscaled * (math.sqrt(cells * rank) / norm)).float().contiguous()
    if not bool(torch.isfinite(factor).all()):
        raise ValueError("Normalized factor is nonfinite in native FP32")
    diagnostics = dict(
        definition="P0_feature_totals_divided_by_mass_then_uniform_cell_center",
        factor_definition=("descending_Gram_eigenvectors_times_sqrt_eigenvalues_sign_pivot"
                           if mode == MODES[0] else "pinned_native_Gaussian_minus_uniform_cell_mean"),
        preparation_device="cpu", preparation_dtype="torch.float64", factor_dtype="torch.float32",
        sign_pivot_rows=None if pivots is None else pivots.tolist(),
        torch_version=str(torch.__version__), rank_threshold_definition="eps64*max(K,D)*lambda_max",
        rank_threshold=threshold, numerical_rank=int((eigenvalues > threshold).sum()),
        eigenvalues=eigenvalues.tolist(), retained_energy=float(eigenvalues[:rank].sum()),
        cutoff_gap=float(eigenvalues[rank - 1] - eigenvalues[rank]),
        centroid_digest=array_digest(centers.numpy()), centered_centroid_digest=array_digest(centered.numpy()),
        gram_digest=array_digest(gram.numpy()), original_native_V_norm=float(original_v.double().norm()),
        original_centered_V_norm=float((original_v.double() - original_v.double().mean(0)).norm()),
        target_Frobenius_norm=math.sqrt(cells * rank), actual_FP32_Frobenius_norm=float(factor.double().norm()),
        actual_FP32_column_mean_max=float(factor.double().mean(0).abs().max()), mass_sum=float(mass.sum()),
    )
    return factor.detach().clone(), diagnostics


def build_factor_packet(initial_moments, original_v, mode, rank, source_refs):
    """Freeze tensors plus root-owned native source refs in a self-bound packet."""
    if not isinstance(source_refs, dict) or not source_refs:
        raise ValueError("Require nonempty root-owned pinned source references")
    source_refs = _json_copy(source_refs)
    factor, diagnostics = generate_factor(initial_moments, original_v, mode, rank)
    packet = dict(schema=SCHEMA, mode=mode, rank=rank,
        dimensions=dict(cells=factor.shape[0], feature_dimension=initial_moments["features"].shape[1]),
        source_refs=source_refs, factor=factor,
        factor_digest=array_digest(factor.numpy()), diagnostics=diagnostics,
        input_digests=dict(mass=array_digest(initial_moments["mass"].numpy()),
                          feature_totals=array_digest(initial_moments["features"].numpy()),
                          original_native_V=array_digest(original_v.numpy())))
    packet["context_digest"] = _seal({key: value for key, value in packet.items() if key != "factor"})
    return packet


def packet_context(packet):
    """Return exactly the seven JSON context fields, without tensor recompute."""
    if not isinstance(packet, dict) or not set(_CONTEXT_KEYS).issubset(packet):
        raise ValueError("Frozen factor packet lacks its context")
    return _json_copy({key: packet[key] for key in _CONTEXT_KEYS})


def load_frozen_factor(path, expected_digest, mode, dimensions, device, *, expected_context):
    """Validate immutable file/context/tensor digests and clone to the consumer."""
    if (not isinstance(expected_digest, str) or len(expected_digest) != 64
            or any(character not in "0123456789abcdef" for character in expected_digest)):
        raise ValueError("Require a pinned lowercase SHA256 artifact digest")
    with Path(path).open("rb") as stream:
        actual_digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual_digest != expected_digest:
        raise ValueError("Frozen factor artifact bytes changed")
    packet = torch.load(path, map_location="cpu", weights_only=True)
    if (not isinstance(packet, dict) or set(packet) != _PACKET_KEYS or type(packet["schema"]) is not int
            or packet["schema"] != SCHEMA or packet["mode"] not in MODES
            or type(packet["rank"]) is not int or packet["rank"] < 1
            or not isinstance(packet["dimensions"], dict)
            or set(packet["dimensions"]) != {"cells", "feature_dimension"}
            or any(type(value) is not int or value < 1 for value in packet["dimensions"].values())
            or not isinstance(packet["source_refs"], dict) or not packet["source_refs"]
            or not isinstance(packet["diagnostics"], dict)
            or not isinstance(packet["input_digests"], dict)):
        raise ValueError("Frozen factor packet has an unsupported schema")
    metadata = {key: value for key, value in packet.items() if key not in ("factor", "context_digest")}
    if packet["context_digest"] != _seal(metadata):
        raise ValueError("Frozen factor metadata changed")
    if (mode != packet["mode"] or _seal(_json_copy(dimensions)) != _seal(packet["dimensions"])
            or _seal(_json_copy(expected_context)) != _seal(packet_context(packet))):
        raise ValueError("Frozen factor differs from the consumer's mode, dimensions or source context")
    cells, rank = packet["dimensions"]["cells"], packet["rank"]
    if rank > min(cells - 1, packet["dimensions"]["feature_dimension"]):
        raise ValueError("Frozen factor rank exceeds its centered source dimensions")
    _tensor(packet["factor"], torch.Size((cells, rank)), torch.float32, "Frozen factor")
    if array_digest(packet["factor"].numpy()) != packet["factor_digest"]:
        raise ValueError("Frozen factor tensor changed")
    return packet["factor"].detach().to(device=device).clone()
