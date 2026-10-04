"""Narrow dual stationary-CE heads on one unchanged native NODE assignment.

Only the P-derived original [1,z,Q] moments are learned. The second head maps
the explicit original RMS inverse of each centroid with the existing ReLU
Nyström map. No factory, teacher fit, extra moment stream or free feature is
used. This module is an opt-in standalone path; original core defaults stay
unchanged. Inclusive resumed endpoints must preserve any existing checkpoint
bits. A bounded prefix-best record handles earliest normalized-objective minima.
"""
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from src.io import array_digest, save_json
from src.low_rank_assignment import LowRankMoments
from src.moments import augmented, decode_moments, make_material
from src.nystrom_ce import NystromMap, _content_digest, moment_gradient
from src.shared_features import _tensor_identity
from src.soft_ce_partition import (
    _save_checkpoint_atomic,
    implicit_moment_gradient,
    outer_value_gradient,
    solve_head_system,
    solve_inner_newton_first,
)
from src.transforms import FeatureTransform

SCHEMA = 1
MODE = "normalized_dual_linear_Nystrom_CE_v1"
OBJECTIVE = "CE_linear_over_CE_linear0_plus_CE_Nystrom_over_CE_Nystrom0"
POLICY = dict(schema=SCHEMA, mode=MODE, objective=OBJECTIVE,
    coefficients=dict(linear=1, Nystrom=1), mass_mode="free", head_weighting="uniform",
    teacher="original_frozen_soft_Q", kernel="relu", inverse="explicit_original_RMS_affine",
    source_space="original_H", linear_space="original_z", original_base_moment_operator=True,
    single_original_P_backward=True, scale_floor=False, probability_floor=False,
    free_features=False, MSE_regression=False, head_bias_regularized=True)
WORK_KEYS = ("moment_forward_calls", "moment_backward_calls", "P_updates",
    "linear_head_interfaces", "Nystrom_head_interfaces", "linear_adjoint_solves",
    "Nystrom_adjoint_solves", "checkpoint_writes", "resume_writes")
PROBABILITY_ATOL = 1e-12
OPTION_KEYS = {"penalty", "lr", "mixing", "chunk_size", "inner_max_iter", "inner_tol",
    "cg_max_iter", "cg_rtol", "save_assignment", "mass_mode", "balance_steps",
    "balance_tol", "balance_backend", "balance_cg_steps", "balance_cg_rtol",
    "outer_chunk_size", "assignment_rank", "factor_seed", "assignment_input",
    "assignment_encoder", "encoder_hidden", "save_resume", "solver_mode",
    "tracking_inner_steps", "tracking_cg_steps", "tracking_refresh", "feature_control",
    "implicit_warm_start", "inner_method", "cg_check_interval", "temperature_initial",
    "temperature_lr", "inner_loss_weighting", "node_weighting", "node_weight_penalty",
    "node_weight_lr", "cache_assignment", "data_digest"}


def _seal(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _number(value, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value) or (positive and value <= 0):
        raise ValueError("Dual-head scalar must be finite and frozen scales positive")
    return value


def _integer(value, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError("Dual-head counter/dimension must be a typed integer")
    return value


def _json_value(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise ValueError("Dual-head metadata must be finite JSON primitives")


def _owned_cpu_state(value):
    """Every saved tensor owns its storage even when optimization runs on CPU."""
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _owned_cpu_state(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_owned_cpu_state(item) for item in value]
    return value


def _matrix(value, shape, dtype=torch.float64, device=None, detached=True):
    if (not torch.is_tensor(value) or value.layout != torch.strided
            or value.ndim != 2 or tuple(value.shape) != tuple(shape) or min(value.shape) < 1
            or value.dtype != dtype or (device is not None and value.device != device)
            or (detached and value.requires_grad) or not bool(torch.isfinite(value).all())):
        raise ValueError("Dual-head matrix shape/dtype/device/finiteness differs")
    return value


def _no_autocast(device):
    if device.type not in ("cpu", "cuda") or torch.is_autocast_enabled(device.type):
        raise ValueError("Dual-head native path supports typed CPU/CUDA without autocast")


def _transform(transform, dimension, device):
    if (type(transform) is not FeatureTransform or set(vars(transform)) !=
            {"center", "matrix", "output_center", "scale", "kind", "eps"}
            or transform.kind != "rms" or transform.matrix is not None
            or type(transform.eps) is not float or not math.isfinite(transform.eps) or transform.eps <= 0):
        raise ValueError("Dual-head mapping requires the original explicit affine RMS inverse")
    for name, shape in (("center", (dimension,)), ("output_center", (dimension,)), ("scale", ())):
        value = getattr(transform, name)
        if (not torch.is_tensor(value) or value.shape != shape or value.dtype != torch.float64
                or value.device != device or value.requires_grad or not bool(torch.isfinite(value).all())):
            raise ValueError("Dual-head frozen RMS tensor descriptor differs")
    if float(transform.scale) <= 0:
        raise ValueError("Dual-head RMS scale must be positive without a floor")
    return dict(kind="rms", matrix=None, eps=transform.eps,
                **{name: _tensor_identity(getattr(transform, name))
                   for name in ("center", "output_center", "scale")})


def _map(feature_map, dimension, device):
    if type(feature_map) is not NystromMap or feature_map.kernel != "relu":
        raise ValueError("Dual-head uses only the original pinned literal ReLU Nyström map")
    anchors, mapping = feature_map.anchors, feature_map.mapping
    if not torch.is_tensor(anchors) or anchors.ndim != 2 or len(anchors) < 1:
        raise ValueError("Dual-head frozen anchor dimensions differ")
    width = len(anchors)
    _matrix(anchors, (width, dimension), device=device)
    _matrix(mapping, (width, width), device=device)
    return width, dict(anchors=_tensor_identity(anchors), mapping=_tensor_identity(mapping))


def _decoded(moments, dimension):
    if type(dimension) is not int or dimension < 1 or not torch.is_tensor(moments) or moments.ndim != 2:
        raise ValueError("Dual-head base moments or physical dimension differ")
    classes = moments.shape[1] - dimension - 1
    if len(moments) < 2 or classes < 2:
        raise ValueError("Dual-head needs nonempty original cells and teacher classes")
    _no_autocast(moments.device)
    _matrix(moments, (len(moments), 1 + dimension + classes))
    centers, labels, mass = decode_moments(moments, dimension)
    if (not bool((mass > 0).all()) or bool((labels < 0).any())
            or not bool(torch.isfinite(centers).all()) or not bool(torch.isfinite(labels).all())
            or float((labels.sum(1) - 1).abs().max()) > PROBABILITY_ATOL
            or abs(float(mass.sum()) - 1) > PROBABILITY_ATOL):
        raise FloatingPointError("Dual-head base moments leave the original positive probability domain")
    return centers, labels, mass, classes


class _RMSFeatureMap:
    def __init__(self, feature_map, transform):
        self.feature_map, self.transform = feature_map, transform

    def __call__(self, cz):
        physical = cz * self.transform.scale + self.transform.output_center + self.transform.center
        return self.feature_map(physical)


def mapped_centroids(moments, dimension, feature_map, transform):
    """Return original-map features after the explicit original RMS inverse."""
    centers, _, _, _ = _decoded(moments, dimension)
    _transform(transform, dimension, moments.device)
    width, _ = _map(feature_map, dimension, moments.device)
    result = _RMSFeatureMap(feature_map, transform)(centers)
    return _matrix(result, (len(moments), width), device=moments.device, detached=False)


def complete_moment_cotangent(moments, dimension, theta_linear, vector_linear,
        theta_Nystrom, vector_Nystrom, penalty, feature_map, transform, CE_linear0, CE_Nystrom0):
    """Return combined FP64 base-M cotangent; original backward supplies 1/N."""
    _, _, _, classes = _decoded(moments, dimension)
    _number(penalty, positive=True); _number(CE_linear0, positive=True); _number(CE_Nystrom0, positive=True)
    _transform(transform, dimension, moments.device)
    width, _ = _map(feature_map, dimension, moments.device)
    for theta, vector, count in ((theta_linear, vector_linear, dimension), (theta_Nystrom, vector_Nystrom, width)):
        _matrix(theta, (classes, count + 1), device=moments.device)
        _matrix(vector, (classes, count + 1), device=moments.device)
    linear = implicit_moment_gradient(moments, dimension, theta_linear, vector_linear,
                                     penalty, loss_weighting="uniform")
    nonlinear = moment_gradient(moments, dimension, _RMSFeatureMap(feature_map, transform),
        theta_Nystrom, vector_Nystrom, penalty, inner_loss_weighting="uniform")
    combined = linear / CE_linear0 + nonlinear / CE_Nystrom0
    return _matrix(combined.detach(), moments.shape, device=moments.device)


@torch.no_grad()
def outer_gradient(phi, q, theta, chunk=65536, stop=lambda: False):
    """Existing cached Phi and original Q mean CE, with bias and stop checks."""
    _integer(chunk, 1)
    value, gradient = theta.new_zeros(()), torch.zeros_like(theta)
    for start in range(0, len(phi), chunk):
        if stop():
            raise InterruptedError("Dual-head stopped during original cached-Phi outer CE")
        block = phi[start:start + chunk]
        if torch.is_tensor(block):
            block = block.detach().to(theta)
        else:
            block = torch.from_numpy(np.array(block, copy=True)).to(theta)
        x = augmented(block)
        target = q[start:start + len(block)]
        lp = (x @ theta.T).log_softmax(1)
        value -= (target * lp).sum() / len(phi)
        error = target.sum(1, keepdim=True) * lp.exp() - target
        gradient += error.T @ x / len(phi)
    if not bool(torch.isfinite(value)) or not bool(torch.isfinite(gradient).all()):
        raise FloatingPointError("Dual-head mapped outer CE/RHS is nonfinite")
    return float(value), gradient


def _files(pins):
    if not isinstance(pins, dict) or not pins:
        raise ValueError("Dual-head immutable file pins are missing")
    for name, digest in pins.items():
        if (type(name) is not str or type(digest) is not str or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)):
            raise ValueError("Dual-head immutable file pin is malformed")
        with Path(name).open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != digest:
            raise ValueError("Dual-head immutable source/cache bytes changed")


def _factor_digests(parameters, nodes, cells, rank, device=None):
    if not isinstance(parameters, (list, tuple)) or len(parameters) != 2:
        raise ValueError("Dual-head requires exactly the native U/V pair")
    for value, shape in zip(parameters, ((nodes, rank), (cells, rank)), strict=True):
        _matrix(value, shape, dtype=torch.float32, device=device)
        if value.device.type not in ("cpu", "cuda"):
            raise ValueError("Dual-head native factor device differs")
    return [array_digest(p.detach().cpu().numpy()) for p in parameters]


def _native_options(options):
    if not isinstance(options, dict) or set(options) - OPTION_KEYS:
        raise ValueError("Dual-head rejects unknown options and simultaneous auxiliary methods")
    _seal(options)
    required = dict(assignment_input="node", assignment_encoder="linear", mass_mode="free",
        inner_loss_weighting="uniform", solver_mode="exact", inner_method="newton_first",
        implicit_warm_start=True, mixing=.05, save_resume=True, save_assignment=False, feature_control="joint")
    if any(options.get(k) != v for k, v in required.items()):
        raise ValueError("Dual-head needs the unchanged native free/uniform/exact NODE recipe")
    for key, value in (("implicit_warm_start", True), ("save_resume", True), ("save_assignment", False)):
        if options.get(key) is not value:
            raise ValueError("Dual-head native Boolean policy must remain typed")
    _number(options.get("mixing"), positive=True)
    if options.get("node_weighting", False) is not False or options.get("cache_assignment", False) is not False:
        raise ValueError("Dual-head cannot reweight nodes or cache a different assignment path")
    if options.get("node_weight_penalty", 0.0) != 0.0 or options.get("node_weight_lr") is not None:
        raise ValueError("Disabled native node-weight options cannot carry an active parameter")
    for key in ("penalty", "lr", "inner_tol", "cg_rtol"):
        _number(options.get(key), positive=True)
    for key in ("chunk_size", "outer_chunk_size", "inner_max_iter", "cg_max_iter", "cg_check_interval", "assignment_rank"):
        _integer(options.get(key), 1)
    _integer(options.get("factor_seed"))
    old = {k: v for k, v in options.items() if k not in ("save_resume", "data_digest")}
    return old


def _validate_context(context):
    if (not isinstance(context, dict) or set(context) != {"schema", "policy", "source_refs",
            "native_parameter_digests", "asset_descriptors"} or type(context.get("schema")) is not int
            or context["schema"] != SCHEMA or _seal(context.get("policy")) != _seal(POLICY)):
        raise ValueError("Dual-head context policy/schema/fields differ")
    _seal(context)
    refs = context["source_refs"]
    if not isinstance(refs, dict):
        raise ValueError("Dual-head source references must be a sealed JSON object")
    for key in ("nodes", "cells", "dimension", "classes", "rank", "chunk_size"):
        _integer(refs.get(key), 1)
    _integer(refs.get("factor_seed"))
    if (refs["cells"] < 2 or refs["classes"] < 2 or refs["rank"] > min(refs["nodes"], refs["cells"])
            or refs.get("mixing") != .05 or type(refs.get("data_digest")) is not str
            or not refs["data_digest"] or type(refs.get("device")) is not str
            or not isinstance(refs.get("original_options"), dict)):
        raise ValueError("Dual-head native dimensions/data/recipe context differs")
    native = context["native_parameter_digests"]
    if (not isinstance(native, list) or len(native) != 2
            or any(type(x) is not str or len(x) != 64 or any(c not in "0123456789abcdef" for c in x) for x in native)):
        raise ValueError("Dual-head original native origin digests are missing")
    assets = context["asset_descriptors"]
    if not isinstance(assets, dict) or set(assets) != {"H", "transform", "anchors", "mapping", "Phi_identity"}:
        raise ValueError("Dual-head original asset descriptors differ")
    phi = assets["Phi_identity"]
    if (not isinstance(phi, dict) or set(phi) != {"schema", "h_digest", "map_digest", "shape", "dtype", "phi_digest"}
            or type(phi["schema"]) is not int or phi["schema"] != 1 or phi["dtype"] != "float64"
            or not isinstance(phi["shape"], list) or len(phi["shape"]) != 2
            or type(phi["shape"][0]) is not int
            or phi["shape"][0] != refs["nodes"] or type(phi["shape"][1]) is not int or phi["shape"][1] < 1):
        raise ValueError("Dual-head mandatory original Phi sidecar identity differs")
    for key in ("h_digest", "map_digest", "phi_digest"):
        pin = phi[key]
        if type(pin) is not str or len(pin) != 64 or any(c not in "0123456789abcdef" for c in pin):
            raise ValueError("Dual-head mandatory original Phi identity digest is malformed")
    return refs, assets, phi["shape"][1]


def _record_digest(record):
    arrays = ("moments", "theta_linear", "theta_Nystrom", "vector_linear", "vector_Nystrom")
    content = {key: _tensor_identity(record[key]) if record.get(key) is not None else None for key in arrays}
    content["parameters"] = [_tensor_identity(p) for p in record["parameters"]]
    content["initial_parameters"] = [_tensor_identity(p) for p in record["initial_parameters"]]
    for key in ("step", "config", "context", "CE_linear", "CE_Nystrom", "CE_linear0",
                "CE_Nystrom0", "objective", "objective_name", "J_exact", "head_work", "vector_step"):
        content[key] = record[key]
    return _seal(content)


def _validate_record(record, config, context):
    refs, _, width = _validate_context(context)
    nodes, cells, rank, dimension, classes = (refs[k] for k in ("nodes", "cells", "rank", "dimension", "classes"))
    required = {"step", "moments", "theta_linear", "theta_Nystrom", "vector_linear", "vector_Nystrom",
        "vector_step", "CE_linear", "CE_Nystrom", "CE_linear0", "CE_Nystrom0", "objective", "objective_name",
        "parameters", "initial_parameters", "config", "context", "J_exact", "head_work", "record_digest"}
    if (not isinstance(record, dict) or set(record) != required or _seal(record.get("config")) != _seal(config)
            or _seal(record.get("context")) != _seal(context) or record.get("J_exact") is not True
            or record.get("objective_name") != OBJECTIVE):
        raise ValueError("Dual-head checkpoint/record lost its source/config/heads")
    _integer(record.get("step"))
    _factor_digests(record.get("parameters"), nodes, cells, rank)
    if _factor_digests(record.get("initial_parameters"), nodes, cells, rank) != context["native_parameter_digests"]:
        raise ValueError("Dual-head record changed its original native U0/V0")
    _decoded(record.get("moments"), dimension)
    _matrix(record["moments"], (cells, 1 + dimension + classes))
    if not isinstance(record.get("head_work"), dict) or set(record["head_work"]) != {"linear", "Nystrom"}:
        raise ValueError("Dual-head record lost its two independent head diagnostics")
    for mode, count in (("linear", dimension), ("Nystrom", width)):
        _matrix(record.get(f"theta_{mode}"), (classes, count + 1))
        vector = record.get(f"vector_{mode}")
        if vector is not None:
            _matrix(vector, (classes, count + 1))
        head = record.get("head_work", {}).get(mode)
        if (not isinstance(head, dict) or head.get("inner_converged") is not True
                or not 0 <= _number(head.get("inner_grad_max")) <= config["inner_tol"]):
            raise ValueError("Dual-head record lost a stationary head diagnostic")
        _number(record.get(f"CE_{mode}"))
        _number(record.get(f"CE_{mode}0"), positive=True)
    objective = record["CE_linear"] / record["CE_linear0"] + record["CE_Nystrom"] / record["CE_Nystrom0"]
    if _number(record.get("objective")) != objective or record.get("record_digest") != _record_digest(record):
        raise ValueError("Dual-head record scalar/tensor/lineage seal differs")
    if record.get("vector_step") is not None:
        _integer(record["vector_step"])
        if record["vector_step"] > record["step"] or any(record.get(f"vector_{m}") is None for m in ("linear", "Nystrom")):
            raise ValueError("Dual-head record adjoints lost their endpoint binding")
    elif any(record.get(f"vector_{m}") is not None for m in ("linear", "Nystrom")):
        raise ValueError("Dual-head adjoints cannot lose their typed endpoint")


def _validate_optimizer(saved, config, refs):
    if not isinstance(saved, dict):
        raise ValueError("Dual-head Adam state must be a complete saved object")
    groups, slots = saved.get("param_groups"), saved.get("state")
    if not isinstance(groups, list) or len(groups) != 1 or not isinstance(slots, dict):
        raise ValueError("Dual-head native Adam structure differs")
    group = groups[0]
    if (not isinstance(group, dict) or group.get("params") != [0, 1] or group.get("lr") != config["lr"]
            or group.get("eps") != 1e-12 or group.get("foreach") is not False
            or tuple(group.get("betas", ())) != (.9, .999) or group.get("weight_decay") != 0
            or group.get("amsgrad") is not False or group.get("maximize") is not False
            or group.get("capturable") is not False or group.get("differentiable") is not False
            or group.get("fused") is not None or group.get("decoupled_weight_decay", False) is not False):
        raise ValueError("Dual-head original Adam recipe changed")
    if not slots:
        return 0
    if set(slots) != {0, 1}:
        raise ValueError("Dual-head Adam lost native U/V states")
    iterations = []
    for index, shape in enumerate(((refs["nodes"], refs["rank"]), (refs["cells"], refs["rank"]))):
        slot = slots[index]
        if not isinstance(slot, dict) or set(slot) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError("Dual-head Adam state keys differ")
        for key in ("exp_avg", "exp_avg_sq"):
            _matrix(slot[key], shape, dtype=torch.float32)
        if bool((slot["exp_avg_sq"] < 0).any()):
            raise ValueError("Dual-head Adam second moment is negative")
        step = slot["step"]
        if not torch.is_tensor(step) or step.shape != () or not bool(torch.isfinite(step)):
            raise ValueError("Dual-head Adam step is not finite scalar state")
        value = float(step)
        if value < 0 or int(value) != value:
            raise ValueError("Dual-head Adam step must be an exact nonnegative integer")
        iterations.append(int(value))
    if iterations[0] != iterations[1]:
        raise ValueError("Dual-head U/V Adam iterations differ")
    return iterations[0]


def validate_core_resume(state, expected_config, context):
    """Validate both heads/scales and immutable native prefix without refitting."""
    refs, _, _ = _validate_context(context)
    bound_config = dict(refs["original_options"], data_digest=refs["data_digest"],
                        dual_head_mode=MODE, dual_head_context=context)
    required = {"step", "config", "context", "parameters", "initial_parameters", "optimizer", "snapshots",
        "history", "current", "prefix_best", "initial_moments", "origin_digest", "moments", "theta_linear",
        "theta_Nystrom", "vector_linear", "vector_Nystrom", "CE_linear0", "CE_Nystrom0", "best", "best_step",
        "best_moments", "best_theta_linear", "best_theta_Nystrom", "best_parameters", "work"}
    if (not isinstance(state, dict) or set(state) != required or not isinstance(expected_config, dict)
            or _seal(expected_config) != _seal(bound_config)
            or _seal(state.get("config")) != _seal(expected_config)
            or _seal(state.get("context")) != _seal(context)):
        raise ValueError("Dual-head resume config/context differs")
    end = _integer(state.get("step"))
    snapshots = state.get("snapshots")
    if not isinstance(snapshots, dict) or 0 not in snapshots or end not in snapshots or (end >= 1 and 1 not in snapshots):
        raise ValueError("Dual-head resume lost original or accepted-prefix/end checkpoint")
    for step, record in snapshots.items():
        if type(step) is not int or not 0 <= step <= end or not isinstance(record, dict) or record.get("step") != step:
            raise ValueError("Dual-head snapshot endpoint key differs")
        _validate_record(record, expected_config, context)
    zero, terminal = snapshots[0], snapshots[end]
    if (_factor_digests(zero["parameters"], refs["nodes"], refs["cells"], refs["rank"]) != context["native_parameter_digests"]
            or bool(zero["parameters"][0].ne(0).any()) or zero["CE_linear"] != zero["CE_linear0"]
            or zero["CE_Nystrom"] != zero["CE_Nystrom0"] or zero["objective"] != 2.0
            or state.get("origin_digest") != zero["record_digest"]):
        raise ValueError("Dual-head original P0/origin/scales changed")
    _validate_record(state.get("current"), expected_config, context)
    _matrix(state.get("initial_moments"), zero["moments"].shape)
    for key in ("CE_linear0", "CE_Nystrom0"):
        if state.get(key) != zero[key] or any(record[key] != zero[key] for record in snapshots.values()):
            raise ValueError("Dual-head frozen CE0 scale changed")
    if (_factor_digests(state.get("initial_parameters"), refs["nodes"], refs["cells"], refs["rank"]) != context["native_parameter_digests"]
            or _tensor_identity(state.get("initial_moments")) != _tensor_identity(zero["moments"])
            or state["current"]["step"] != end):
        raise ValueError("Dual-head resume lost original moments/factors/current endpoint")
    for key in ("moments", "theta_linear", "theta_Nystrom", "vector_linear", "vector_Nystrom"):
        value, current = state.get(key), state["current"].get(key)
        if (value is None) != (current is None) or (value is not None and _tensor_identity(value) != _tensor_identity(current)):
            raise ValueError("Dual-head resume current head/moment/vector link changed")
    if (_factor_digests(state.get("parameters"), refs["nodes"], refs["cells"], refs["rank"]) !=
            _factor_digests(terminal["parameters"], refs["nodes"], refs["cells"], refs["rank"])
            or _factor_digests(state["current"]["parameters"], refs["nodes"], refs["cells"], refs["rank"]) !=
            _factor_digests(terminal["parameters"], refs["nodes"], refs["cells"], refs["rank"])):
        raise ValueError("Dual-head resume parameters differ from terminal checkpoint")
    for key in ("moments", "theta_linear", "theta_Nystrom"):
        if _tensor_identity(state["current"][key]) != _tensor_identity(terminal[key]):
            raise ValueError("Dual-head current/terminal checkpoint material/head differs")
    for key in ("CE_linear", "CE_Nystrom", "objective", "CE_linear0", "CE_Nystrom0"):
        if state["current"][key] != terminal[key]:
            raise ValueError("Dual-head current/terminal checkpoint scalar differs")
    history = state.get("history")
    if (not isinstance(history, list) or any(not isinstance(row, dict) for row in history)
            or [row.get("step") for row in history] != list(range(end + 1))):
        raise ValueError("Dual-head history lost its complete typed prefix")
    for row in history:
        if (type(row.get("step")) is not int or row.get("J_exact") is not True
                or row.get("objective_name") != OBJECTIVE or row.get("CE_linear0") != zero["CE_linear0"]
                or row.get("CE_Nystrom0") != zero["CE_Nystrom0"]
                or _number(row.get("objective")) != _number(row.get("CE_linear")) / zero["CE_linear0"]
                    + _number(row.get("CE_Nystrom")) / zero["CE_Nystrom0"]
                or row.get("J") != row["objective"]):
            raise ValueError("Dual-head history changed heads/scales/normalized objective")
        for mode in ("linear", "Nystrom"):
            if (row.get(f"{mode}_inner_converged") is not True
                    or not 0 <= _number(row.get(f"{mode}_inner_grad_max")) <= expected_config["inner_tol"]):
                raise ValueError("Dual-head history lost an exact head")
            if (row["step"] < end or row.get("status") == "update") and (
                    row.get(f"{mode}_evaluated") is not True or row.get(f"{mode}_cg_converged") is not True):
                raise ValueError("Dual-head update history lost its converged original adjoint")
        if (row.get("solver_mode") != "exact" or row.get("implicit_warm_start") is not True
                or (row["step"] < end and row.get("status") != "update")
                or (row["step"] == end and row.get("status") not in ("update", "evaluated"))):
            raise ValueError("Dual-head history changed its fixed exact endpoint policy")
        # Persist also runs BEFORE an update at a checkpoint frontier. Such a
        # frontier is complete only when both solved adjoints bind this exact
        # current endpoint; an invocation's evaluated endpoint needs no update.
        if (row["step"] == end and row.get("status") == "update"
                and (state["current"]["vector_step"] != end
                     or any(state["current"].get(f"vector_{mode}") is None for mode in ("linear", "Nystrom")))):
            raise ValueError("Dual-head saved update frontier lost its current two-adjoint binding")
    for step, record in snapshots.items():
        if any(record[key] != history[step][key] for key in ("CE_linear", "CE_Nystrom", "objective")):
            raise ValueError("Dual-head checkpoint/history scalar links changed")
    prefix = state.get("prefix_best")
    if end == 0:
        if prefix is not None:
            raise ValueError("Dual-head step0 cannot carry a historical best prefix")
    else:
        if prefix is None:
            raise ValueError("Dual-head bounded historical best prefix missing")
        _validate_record(prefix, expected_config, context)
        expected_prefix = min(history[:-1], key=lambda row: row["objective"])
        if (prefix["step"] != expected_prefix["step"] or prefix["objective"] != expected_prefix["objective"]
                or any(prefix[key] != expected_prefix[key] for key in ("CE_linear", "CE_Nystrom", "CE_linear0", "CE_Nystrom0"))):
            raise ValueError("Dual-head prefix best lost earliest normalized history minimum")
    best = state["current"] if prefix is None or state["current"]["objective"] < prefix["objective"] else prefix
    if state.get("best_step") != best["step"] or state.get("best") != best["objective"]:
        raise ValueError("Dual-head best must select earliest normalized objective, not students")
    for stored, source in (("best_moments", "moments"), ("best_theta_linear", "theta_linear"), ("best_theta_Nystrom", "theta_Nystrom")):
        _matrix(state.get(stored), best[source].shape)
        if _tensor_identity(state.get(stored)) != _tensor_identity(best[source]):
            raise ValueError("Dual-head bounded best material/head link changed")
    if _factor_digests(state.get("best_parameters"), refs["nodes"], refs["cells"], refs["rank"]) != _factor_digests(best["parameters"], refs["nodes"], refs["cells"], refs["rank"]):
        raise ValueError("Dual-head bounded best factor link changed")
    work = state.get("work")
    if not isinstance(work, dict) or set(work) != set(WORK_KEYS):
        raise ValueError("Dual-head actual-work counters missing")
    for value in work.values():
        _integer(value)
    if work["P_updates"] != end or _validate_optimizer(state.get("optimizer"), expected_config, refs) != end:
        raise ValueError("Dual-head trajectory/Adam/native update count differs")
    if (work["moment_backward_calls"] != end or work["checkpoint_writes"] != len(snapshots)
            or work["linear_head_interfaces"] != work["moment_forward_calls"]
            or work["Nystrom_head_interfaces"] != work["moment_forward_calls"]
            or work["moment_forward_calls"] < end + 1 or work["resume_writes"] < len(snapshots)
            or work["linear_adjoint_solves"] < end or work["Nystrom_adjoint_solves"] < end):
        raise ValueError("Dual-head actual two-head work counters lost their trajectory links")
    return state


def _validate_inputs(z, q, assignment, initial_parameters, feature_map, phi, transform, options, context):
    if not torch.is_tensor(z):
        raise ValueError("Dual-head original z must be a detached matrix")
    _no_autocast(z.device)
    if torch.get_default_dtype() != torch.float32:
        raise ValueError("Dual-head native initialization/Adam requires the original FP32 default")
    refs, assets, width = _validate_context(context)
    old = _native_options(options)
    n, d, c, k, rank = (refs[key] for key in ("nodes", "dimension", "classes", "cells", "rank"))
    _matrix(z, (n, d), device=z.device)
    _matrix(q, (n, c), device=z.device)
    if (not torch.is_tensor(assignment) or assignment.ndim != 1 or assignment.shape != (n,)
            or assignment.dtype != torch.int64 or assignment.device != z.device or assignment.requires_grad
            or int(assignment.min()) < 0 or int(assignment.max()) != k - 1):
        raise ValueError("Dual-head original hard assignment differs")
    if bool((q < 0).any()) or float((q.sum(1) - 1).abs().max()) > PROBABILITY_ATOL:
        raise ValueError("Dual-head original fixed teacher Q differs")
    data = array_digest(z.cpu().numpy(), q.cpu().numpy(), assignment.cpu().numpy())
    if (data != refs["data_digest"] or refs["device"] != str(z.device)
            or options["assignment_rank"] != rank or options["factor_seed"] != refs["factor_seed"]
            or options["chunk_size"] != refs["chunk_size"] or old != refs["original_options"]
            or ("data_digest" in options and options["data_digest"] != data)):
        raise ValueError("Dual-head native source/options/provenance binding differs")
    native = _factor_digests(initial_parameters, n, k, rank)
    if native != context["native_parameter_digests"] or bool(initial_parameters[0].ne(0).any()):
        raise ValueError("Dual-head original zeroU/nativeGaussianV origin differs")
    actual_width, map_descriptors = _map(feature_map, d, z.device)
    if (width != actual_width or map_descriptors["anchors"] != assets["anchors"]
            or map_descriptors["mapping"] != assets["mapping"]
            or _transform(transform, d, z.device) != assets["transform"]):
        raise ValueError("Dual-head frozen map/RMS descriptors changed")
    hdesc = assets["H"]
    if not isinstance(hdesc, dict) or hdesc.get("shape") != [n, d] or hdesc.get("dtype") not in ("torch.float32", "torch.float64"):
        raise ValueError("Dual-head original physical H descriptor differs")
    if torch.is_tensor(phi):
        _matrix(phi, (n, width), device=phi.device)
    elif (not isinstance(phi, np.ndarray) or phi.shape != (n, width) or phi.dtype != np.float64
            or phi.flags.writeable):
        raise ValueError("Dual-head requires detached FP64 Phi or original read-only FP64 cache")
    if _content_digest(phi, canonical_double=True) != assets["Phi_identity"]["phi_digest"]:
        raise ValueError("Dual-head original cached Phi content changed")
    paths, pins = refs.get("asset_paths"), refs.get("files_sha256")
    if (not isinstance(paths, dict) or set(paths) != {"H", "map", "Phi", "Phi_metadata"}
            or not isinstance(pins, dict) or any(type(p) is not str or p not in pins for p in paths.values())
            or not Path(paths["Phi_metadata"]).is_file()):
        raise ValueError("Dual-head H/map/Phi/sidecar/native source file roles are unpinned")
    _files(pins)
    if json.loads(Path(paths["Phi_metadata"]).read_text()) != assets["Phi_identity"]:
        raise ValueError("Dual-head mandatory existing Phi sidecar bytes/identity changed")
    source = refs.get("current_source")
    if (not isinstance(source, dict) or not isinstance(source.get("git_head"), str) or not source["git_head"]
            or not isinstance(source.get("source_digest"), str) or not source["source_digest"]):
        raise ValueError("Dual-head current implementation binding missing")
    _files(source.get("files"))
    config = dict(old, data_digest=data, dual_head_mode=MODE, dual_head_context=context)
    return config


def optimize(z, q, assignment, initial_parameters, feature_map, phi, transform,
        folder, steps, options, context, checkpoint_steps=(0, 1, 25), resume_state=None, stop=lambda: False):
    """Run only the original P with two exact heads; return the saved CPU state."""
    if not callable(stop):
        raise ValueError("Dual-head stop must be callable")
    _integer(steps)
    if stop():
        raise InterruptedError("Dual-head stopped before input/context validation")
    config = _validate_inputs(z, q, assignment, initial_parameters, feature_map, phi, transform, options, context)
    refs, _, _ = _validate_context(context)
    if not isinstance(checkpoint_steps, (list, tuple)) or any(type(x) is not int or x < 0 for x in checkpoint_steps):
        raise ValueError("Dual-head checkpoint schedule must contain nonnegative integer steps")
    checkpoints = {0, 1, steps, *checkpoint_steps}
    folder = Path(folder)
    if (folder / "failure.json").exists():
        raise ValueError("Preserve failed dual-head namespace; no automatic numerical rescue")
    if resume_state is not None:
        validate_core_resume(resume_state, config, context)
        if resume_state["step"] > steps:
            raise ValueError("Dual-head cannot resume to an earlier endpoint")
    elif (folder / "resume.pt").exists() or any((folder / "checkpoints").glob("step_*.pt")):
        raise ValueError("Dual-head existing checkpoints require explicit verifiable resume")
    context_path = folder / "context.json"
    if context_path.exists() and _seal(json.loads(context_path.read_text())) != _seal(context):
        raise ValueError("Dual-head trajectory context file changed")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "checkpoints").mkdir(exist_ok=True)
    if not context_path.exists():
        save_json(context, context_path)
    parameters = [p.detach().to(device=z.device).clone().requires_grad_() for p in initial_parameters]
    initial = _owned_cpu_state(initial_parameters)
    optimizer = torch.optim.Adam(parameters, lr=options["lr"], eps=1e-12, foreach=False)
    material, full_features = make_material(z, q), augmented(z)
    start, snapshots, history = 0, {}, []
    work = {key: 0 for key in WORK_KEYS}
    theta = dict(linear=None, Nystrom=None)
    vector = dict(linear=None, Nystrom=None)
    vector_step, prefix_best, current, CE_linear0, CE_Nystrom0 = None, None, None, None, None
    if resume_state is not None:
        with torch.no_grad():
            for parameter, saved in zip(parameters, resume_state["parameters"], strict=True):
                parameter.copy_(saved.to(parameter))
        optimizer.load_state_dict(resume_state["optimizer"])
        start = resume_state["step"]
        snapshots, history, work = dict(resume_state["snapshots"]), list(resume_state["history"]), dict(resume_state["work"])
        for mode in ("linear", "Nystrom"):
            theta[mode] = resume_state[f"theta_{mode}"].to(z.device)
            value = resume_state[f"vector_{mode}"]
            vector[mode] = None if value is None else value.to(z.device)
        vector_step, prefix_best = resume_state["current"]["vector_step"], resume_state["prefix_best"]
        CE_linear0, CE_Nystrom0 = resume_state["CE_linear0"], resume_state["CE_Nystrom0"]

    def check_stop():
        if stop():
            raise InterruptedError("Dual-head bounded invocation stopped; preserve last complete checkpoint")

    def make_record(step, moments, values, fitted, objective):
        record = _owned_cpu_state(dict(step=step, moments=moments.detach(), theta_linear=theta["linear"],
            theta_Nystrom=theta["Nystrom"], vector_linear=vector["linear"], vector_Nystrom=vector["Nystrom"],
            vector_step=vector_step, CE_linear=values["linear"], CE_Nystrom=values["Nystrom"],
            CE_linear0=CE_linear0, CE_Nystrom0=CE_Nystrom0, objective=objective,
            objective_name=OBJECTIVE, parameters=parameters, initial_parameters=initial,
            config=config, context=context, J_exact=True, head_work=fitted))
        record["record_digest"] = _record_digest(record)
        return record

    def persist(step, record):
        nonlocal current
        current = record
        if step in snapshots:
            original = snapshots[step]
            # Accepted checkpoints0/1 retain their bytes/diagnostics; only the
            # current vector/work may advance on the inclusive endpoint read.
            for key in ("moments", "theta_linear", "theta_Nystrom"):
                if _tensor_identity(record[key]) != _tensor_identity(original[key]):
                    raise ValueError("Dual-head resumed endpoint changed accepted material/head bits")
            for key in ("CE_linear", "CE_Nystrom", "CE_linear0", "CE_Nystrom0", "objective"):
                if record[key] != original[key]:
                    raise ValueError("Dual-head resumed endpoint changed accepted scalar bits")
            if _factor_digests(record["parameters"], refs["nodes"], refs["cells"], refs["rank"]) != _factor_digests(original["parameters"], refs["nodes"], refs["cells"], refs["rank"]):
                raise ValueError("Dual-head resumed endpoint changed accepted native parameters")
        else:
            snapshots[step] = record
            work["checkpoint_writes"] += 1
            _save_checkpoint_atomic(record, folder / "checkpoints" / f"step_{step:06d}.pt")
        best = record if prefix_best is None or record["objective"] < prefix_best["objective"] else prefix_best
        work["resume_writes"] += 1
        state = _owned_cpu_state(dict(step=step, config=config, context=context, parameters=parameters,
            initial_parameters=initial, optimizer=optimizer.state_dict(), snapshots=snapshots,
            history=history, current=record, prefix_best=prefix_best,
            initial_moments=snapshots[0]["moments"], origin_digest=snapshots[0]["record_digest"],
            moments=record["moments"], theta_linear=theta["linear"], theta_Nystrom=theta["Nystrom"],
            vector_linear=vector["linear"], vector_Nystrom=vector["Nystrom"],
            CE_linear0=CE_linear0, CE_Nystrom0=CE_Nystrom0,
            best=best["objective"], best_step=best["step"], best_moments=best["moments"],
            best_theta_linear=best["theta_linear"], best_theta_Nystrom=best["theta_Nystrom"],
            best_parameters=best["parameters"], work=work))
        validate_core_resume(state, config, context)
        _save_checkpoint_atomic(state, folder / "resume.pt")
        pd.DataFrame(history).to_csv(folder / "optimization.csv", index=False)
        return state

    step = start
    try:
        for step in range(start, steps + 1):
            check_stop()
            if current is not None:
                if prefix_best is None or current["objective"] < prefix_best["objective"]:
                    prefix_best = current
            tick = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            work["moment_forward_calls"] += 1
            moments = LowRankMoments.apply(*parameters, assignment, material, .05, options["chunk_size"])
            centers, labels, mass, _ = _decoded(moments.detach(), z.shape[1])
            weights = torch.full_like(mass, 1 / len(mass))
            features = dict(linear=centers, Nystrom=mapped_centroids(moments.detach(), z.shape[1], feature_map, transform))
            fitted, values, rhs = {}, {}, {}
            for mode in ("linear", "Nystrom"):
                check_stop()
                work[f"{mode}_head_interfaces"] += 1
                fit = solve_inner_newton_first(features[mode], labels, weights, options["penalty"],
                    theta[mode], options["inner_max_iter"], options["inner_tol"],
                    cg_max_iter=options["cg_max_iter"], cg_check_interval=options["cg_check_interval"])
                theta[mode] = fit.pop("theta").detach()
                fitted[mode] = _json_value(fit)
                check_stop()
                if fitted[mode].get("inner_converged") is not True or fitted[mode].get("inner_grad_max", math.inf) > options["inner_tol"]:
                    raise RuntimeError(f"Dual-head {mode} stationary head did not converge")
            check_stop()
            values["linear"], rhs["linear"] = outer_value_gradient(z, q, theta["linear"], options["outer_chunk_size"], full_features)
            check_stop()
            values["Nystrom"], rhs["Nystrom"] = outer_gradient(phi, q, theta["Nystrom"], options["outer_chunk_size"], stop)
            check_stop()
            if step == 0:
                for value in values.values():
                    _number(value, positive=True)
                if CE_linear0 is None and CE_Nystrom0 is None:
                    CE_linear0, CE_Nystrom0 = values["linear"], values["Nystrom"]
                elif CE_linear0 != values["linear"] or CE_Nystrom0 != values["Nystrom"]:
                    raise ValueError("Dual-head resumed P0 changed its frozen CE0 scalars")
            _number(CE_linear0, positive=True); _number(CE_Nystrom0, positive=True)
            objective = values["linear"] / CE_linear0 + values["Nystrom"] / CE_Nystrom0
            _number(objective)
            diagnostics = {mode: dict(cg_converged=False, evaluated=False) for mode in ("linear", "Nystrom")}
            if step < steps:
                for mode in ("linear", "Nystrom"):
                    check_stop()
                    work[f"{mode}_adjoint_solves"] += 1
                    vector[mode], diagnostic = solve_head_system(augmented(features[mode]), labels, weights,
                        theta[mode], options["penalty"], rhs[mode], rtol=options["cg_rtol"],
                        max_iter=options["cg_max_iter"], initial=vector[mode],
                        cg_check_interval=options["cg_check_interval"])
                    diagnostics[mode] = dict(_json_value(diagnostic), evaluated=True)
                    check_stop()
                    if diagnostics[mode].get("cg_converged") is not True or not bool(torch.isfinite(vector[mode]).all()):
                        raise RuntimeError(f"Dual-head {mode} implicit adjoint did not converge")
                    vector[mode] = vector[mode].detach()
                vector_step = step
            row = dict(step=step, CE_linear=values["linear"], CE_Nystrom=values["Nystrom"],
                CE_linear0=CE_linear0, CE_Nystrom0=CE_Nystrom0, objective=objective, J=objective,
                objective_name=OBJECTIVE, J_exact=True, status="evaluated" if step == steps else "update",
                min_mass=float(mass.min()), max_mass=float(mass.max()), effective_cells=float(1 / mass.square().sum()),
                solver_mode="exact", implicit_warm_start=True, seconds=time.perf_counter() - tick)
            for mode in ("linear", "Nystrom"):
                row.update({f"{mode}_{key}": value for key, value in fitted[mode].items()})
                row.update({f"{mode}_{key}": value for key, value in diagnostics[mode].items()})
            history = [entry for entry in history if entry["step"] < step] + [row]
            record = make_record(step, moments, values, fitted, objective)
            if step in checkpoints or step == steps:
                state = persist(step, record)
            else:
                current = record
            if step == steps:
                check_stop()
                return state
            check_stop()
            cotangent = complete_moment_cotangent(moments.detach(), z.shape[1],
                theta["linear"], vector["linear"], theta["Nystrom"], vector["Nystrom"],
                options["penalty"], feature_map, transform, CE_linear0, CE_Nystrom0)
            work["moment_backward_calls"] += 1
            moments.backward(cotangent)
            if any(p.grad is None or p.grad.dtype != torch.float32 or not bool(torch.isfinite(p.grad).all()) for p in parameters):
                raise FloatingPointError("Dual-head combined native factor gradient is nonfinite")
            check_stop()
            optimizer.step()
            work["P_updates"] += 1
            if any(not bool(torch.isfinite(p).all()) for p in parameters):
                raise FloatingPointError("Dual-head original Adam produced nonfinite native factors")
    except Exception as error:
        save_json(dict(schema=SCHEMA, step=step, mode=MODE, reason=str(error),
                       error_type=type(error).__name__, work=work, no_automatic_rescue=True), folder / "failure.json")
        raise
    raise AssertionError("Dual-head fixed endpoint was not evaluated")
