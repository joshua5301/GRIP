"""Frozen one-hidden-layer, bias-free ReLU NTK Nyström surrogate features.

Teacher Q is unchanged. With fixed s=mean(||original anchors||²), this is the
infinite-width NTK of f(x)=m**(-1/2) sum a_j ReLU(w_j·x/sqrt(s)), training both
weight layers. Its theoretical diagonal matches the existing normalized ReLU
covariance. Forward endpoints are exact; gradient cusps are rejected explicitly.
"""
import hashlib
import math
from pathlib import Path

import torch

from src.io import save_state
from src.nystrom_ce import _cache_identity, _content_digest, _metadata_path, cache_features
from src.shared_features import get_shared_map

KIND = "relu_ntk1_diagmatch_v1"
ANGLE_GUARD = 1e-12
NORM_GUARD = 1e-12
JITTER = 1e-8


def source_digest():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _matrix(value, name):
    if (not torch.is_tensor(value) or value.ndim != 2 or min(value.shape) < 1
            or not value.is_floating_point() or not bool(torch.isfinite(value).all())):
        raise ValueError(f"{name} must be a finite nonempty floating matrix")


def validate_gradient_domain(a, b, scale, angle_guard=ANGLE_GUARD, norm_guard=NORM_GUARD):
    """Do not manufacture a smoothed or straight-through derivative at a cusp."""
    _matrix(a, "NTK queries")
    _matrix(b, "NTK anchors")
    if a.shape[1] != b.shape[1] or a.device != b.device:
        raise ValueError("NTK query/anchor coordinates or devices differ")
    if not isinstance(scale, (int, float)) or isinstance(scale, bool) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("NTK scale must be a fixed positive finite number")
    na, nb = a.double().norm(dim=1), b.double().norm(dim=1)
    if bool((na <= norm_guard * math.sqrt(scale)).any()):
        raise FloatingPointError("NTK centroid derivative is undefined/unstable at a zero or tiny query")
    denominator = na[:, None] * torch.where(nb > 0, nb, torch.ones_like(nb))[None, :]
    cosine = (a.double() @ b.double().T) / denominator
    if bool(((cosine.abs() >= 1 - angle_guard) & (nb > 0)[None, :]).any()):
        raise FloatingPointError("NTK centroid derivative is undefined/unstable at an angular endpoint")


def kernel_values(a, b, scale, angle_guard=ANGLE_GUARD, norm_guard=NORM_GUARD):
    _matrix(a, "NTK queries")
    _matrix(b, "NTK anchors")
    if a.shape[1] != b.shape[1] or a.device != b.device:
        raise ValueError("NTK query/anchor coordinates or devices differ")
    if not isinstance(scale, (int, float)) or isinstance(scale, bool) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("NTK scale must be a fixed positive finite number")
    if angle_guard != ANGLE_GUARD or norm_guard != NORM_GUARD:
        raise ValueError("Only the versioned NTK cusp-guard policy is supported")
    if a.requires_grad:
        validate_gradient_domain(a.detach(), b.detach(), scale, angle_guard, norm_guard)
    if b.requires_grad:
        validate_gradient_domain(b.detach(), a.detach(), scale, angle_guard, norm_guard)
    a, b = a.double(), b.double()
    na, nb = a.norm(dim=1), b.norm(dim=1)
    denominator = torch.where(na > 0, na, torch.ones_like(na))[:, None] * torch.where(nb > 0, nb, torch.ones_like(nb))[None, :]
    cosine = ((a @ b.T) / denominator).clamp(-1, 1)
    theta = torch.acos(cosine)
    covariance = (torch.sqrt((1 - cosine.square()).clamp_min(0)) + (math.pi - theta) * cosine) / math.pi
    derivative_correlation = (math.pi - theta) / math.pi
    return (na[:, None] * nb[None, :] / scale) * (covariance + cosine * derivative_correlation) / 2


class NTKNystromMap:
    def __init__(self, anchors, mapping, scale, source):
        self.anchors, self.mapping, self.scale = anchors, mapping, float(scale)
        self.kernel, self.schema, self.source = KIND, 1, source
        self.angle_guard, self.norm_guard, self.jitter = ANGLE_GUARD, NORM_GUARD, JITTER

    def __call__(self, h):
        return kernel_values(h.double(), self.anchors, self.scale,
                             self.angle_guard, self.norm_guard) @ self.mapping


def cache_paths(root):
    tag = f"{KIND}_{source_digest()[:12]}"
    return Path(root) / f"nystrom_map_{tag}.pt", Path(root) / f"nystrom_phi_{tag}.npy"


def _identity(h, anchors, scale):
    return dict(schema=1, kind=KIND, source_digest=source_digest(),
                h_digest=_content_digest(h, canonical_double=True), h_shape=list(h.shape),
                anchor_digest=_content_digest(anchors), anchor_shape=list(anchors.shape),
                scale=float(scale), angle_guard=ANGLE_GUARD, norm_guard=NORM_GUARD, jitter=JITTER,
                torch_version=str(torch.__version__), anchor_origin="existing_shared_relu_map_exact_order",
                scale_calibration="original_saved_anchors_cpu_float64_mean_squared_norm_v1")


@torch.no_grad()
def get_shared_ntk_map(h, root, *, require_existing=False):
    """Reuse actual saved ReLU anchors, not a new random or learned dictionary."""
    _matrix(h, "NTK source H")
    root = Path(root)
    original_path = root / "nystrom_map_schema3.pt"
    if not original_path.exists():
        raise ValueError("NTK surrogate requires the existing frozen ReLU anchor cache")
    original = get_shared_map(h, original_path, basis=3000, seed=0)
    anchors = original.anchors.detach().double()
    # Canonical CPU calibration uses the exact saved anchor bytes regardless of
    # the serving device; never recalibrate from a query batch or GPU reduction.
    scale = float(anchors.cpu().square().sum(1).mean())
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("NTK anchor energy must be finite and positive")
    identity = _identity(h, anchors, scale)
    path, _ = cache_paths(root)
    if path.exists():
        try:
            saved = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise ValueError("Invalid NTK map cache; preserve it and use a new namespace") from exc
        if not isinstance(saved, dict) or saved.get("identity") != identity:
            raise ValueError("NTK map identity/source/H/anchors/config differs")
        frozen_anchors, mapping = saved.get("anchors"), saved.get("mapping")
        _matrix(frozen_anchors, "Saved NTK anchors")
        _matrix(mapping, "Saved NTK mapping")
        if (frozen_anchors.dtype != torch.double or mapping.dtype != torch.double
                or tuple(mapping.shape) != (len(anchors), len(anchors))
                or tuple(frozen_anchors.shape) != tuple(anchors.shape)
                or _content_digest(frozen_anchors) != identity["anchor_digest"]
                or saved.get("mapping_digest") != _content_digest(mapping)):
            raise ValueError("NTK map content differs")
        return NTKNystromMap(frozen_anchors.to(h.device), mapping.to(h.device), scale, identity["source_digest"])
    if require_existing:
        raise ValueError("Cached NTK condensate lacks its frozen NTK map")
    gram = kernel_values(anchors, anchors, scale)
    gram = (gram + gram.T) / 2
    eye = torch.eye(len(anchors), device=anchors.device, dtype=torch.double)
    diagonal = gram.diagonal().mean()
    if not bool(torch.isfinite(gram).all()) or not float(diagonal) > 0:
        raise FloatingPointError("Invalid NTK Gram matrix")
    chol = torch.linalg.cholesky(gram + JITTER * diagonal * eye)
    mapping = torch.linalg.solve_triangular(chol, eye, upper=False).T
    state = dict(identity=identity, anchors=anchors.cpu().clone(), mapping=mapping.cpu().clone(),
                 mapping_digest=_content_digest(mapping))
    save_state(state, path)
    return NTKNystromMap(anchors, mapping, scale, identity["source_digest"])


def prepare_features(h, root, *, require_existing=False, stop=lambda: False):
    """The generic phi cache stays separate from all legacy map/phi filenames."""
    feature_map = get_shared_ntk_map(h, root, require_existing=require_existing)
    _, path = cache_paths(root)
    if require_existing and (not path.exists() or not _metadata_path(path).exists()):
        raise ValueError("Cached NTK condensate lacks verifiable phi metadata")
    phi = cache_features(h, feature_map, path, stop=stop)
    return feature_map, phi


def expected_fingerprint(h, q, assignment, root):
    feature_map, phi = prepare_features(h, root, require_existing=True)
    identity = _cache_identity(h, feature_map, tuple(phi.shape), 2048, lambda: False)
    fingerprint = dict(**identity, phi_digest=_content_digest(phi),
                       q_digest=_content_digest(q, canonical_double=True), assignment_digest=_content_digest(assignment))
    return fingerprint, feature_map


def validate_cached(saved, candidate, h, q, assignment, root, seed, *, resume=False):
    fingerprint, feature_map = expected_fingerprint(h, q, assignment, root)
    if saved.get("input_fingerprint") != fingerprint:
        raise ValueError("NTK cached H/Q/assignment/map/phi fingerprint differs")
    config = dict(steps_schema=2, penalty=candidate["penalty"], lr=candidate["lr"], rank=candidate["rank"],
                  seed=seed, cells=int(assignment.max()) + 1, chunk=2048, inner_loss_weighting="uniform")
    if candidate.get("mixing", .05) != .05:
        config["mixing"] = candidate["mixing"]
    head_shape = (q.shape[1], len(feature_map.anchors) + 1)
    if resume:
        if saved.get("config") != config:
            raise ValueError("NTK resume effective configuration differs")
        for name, shape in (("u", (len(h), config["rank"])), ("v", (config["cells"], config["rank"]))):
            tensor = saved.get(name)
            if not torch.is_tensor(tensor) or tuple(tensor.shape) != shape or not bool(torch.isfinite(tensor).all()):
                raise ValueError("NTK resume factor shape/content differs")
        for name in ("theta", "vector"):
            tensor = saved.get(name)
            if tensor is not None and (not torch.is_tensor(tensor) or tuple(tensor.shape) != head_shape
                                       or not bool(torch.isfinite(tensor).all())):
                raise ValueError("NTK resume head shape/content differs")
    else:
        from src.moments import decode_moments
        moments = saved.get("moments")
        if (saved.get("inner_loss_weighting", "mass") != "uniform"
                or not torch.is_tensor(moments) or tuple(moments.shape) != (config["cells"], 1 + h.shape[1] + q.shape[1])
                or not bool(torch.isfinite(moments).all()) or bool((moments[:, 0] <= 0).any())):
            raise ValueError("NTK checkpoint moments are invalid")
        theta = saved.get("theta")
        if not torch.is_tensor(theta) or tuple(theta.shape) != head_shape or not bool(torch.isfinite(theta).all()):
            raise ValueError("NTK checkpoint head shape/content differs")
        centers, _, _ = decode_moments(moments, h.shape[1])
        validate_gradient_domain(centers.detach(), feature_map.anchors, feature_map.scale)


def candidate_controls(candidate):
    candidate = dict(candidate)
    fields = {"surrogate_kernel", "surrogate_schema", "surrogate_source_digest", "ntk_angle_guard", "ntk_norm_guard", "ntk_jitter"}
    unsupported = {"kernel", "teacher_kernel", "ntk_layers", "ntk_bias", "ntk_scale", "ntk_basis", "ntk_guard", "ntk_smoothing"}
    supplied = (set(candidate) & (fields | unsupported)) | {key for key in candidate if key.startswith("ntk_")}
    if not supplied:
        return candidate
    if supplied - fields:
        raise ValueError("Unsupported NTK setting; frozen anchors/scale/guard are mandatory")
    if candidate.get("method", "low_rank") != "nystrom":
        raise ValueError("surrogate_kernel controls require a Nyström assignment")
    kind = candidate.get("surrogate_kernel", "relu")
    if kind == "relu":
        if set(candidate) & (fields - {"surrogate_kernel"}):
            raise ValueError("NTK controls require the NTK surrogate")
        candidate.pop("surrogate_kernel", None)
        return candidate
    if kind != KIND:
        raise ValueError("Unsupported surrogate_kernel")
    if candidate.get("mass_mode", "free") != "free" or candidate.get("inner_loss_weighting", "mass") != "uniform":
        raise ValueError("This NTK pilot requires free masses and uniform inner CE")
    if candidate.get("learn_temperature", False) or candidate.get("train_target_mix", 0) != 0:
        raise ValueError("NTK surrogate pilot requires the frozen existing teacher Q")
    values = dict(surrogate_schema=1, surrogate_source_digest=source_digest(),
                  ntk_angle_guard=ANGLE_GUARD, ntk_norm_guard=NORM_GUARD, ntk_jitter=JITTER)
    for key, default in values.items():
        if key in candidate and (isinstance(candidate[key], bool) or candidate[key] != default):
            raise ValueError(f"Unsupported or stale NTK control {key}")
    unsupported = {"kernel", "teacher_kernel", "ntk_layers", "ntk_bias", "ntk_scale", "ntk_basis", "ntk_guard", "ntk_smoothing"}
    if set(candidate) & unsupported:
        raise ValueError("Unsupported NTK setting; frozen anchors/scale/guard are mandatory")
    candidate.update(values)
    return candidate
