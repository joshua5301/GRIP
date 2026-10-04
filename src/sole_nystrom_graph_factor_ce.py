"""Sole frozen ReLU Nyström stationary CE on native direct or graph NODE factors.

Only the original assignment is optimized. The graph map is composed with the
unchanged LowRankMoments, and its transpose follows the original native factor
pullback. Saved checkpoints are immutable; inclusive prefix reevaluation is
recorded separately. This module contains no initializer or map/cache factory.
"""
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from src.dual_head_ce import (
    _factor_digests, _files, _map, _native_options, _transform, _validate_optimizer,
)
from src.graph_factor import FrozenCPUCSR, GraphProduct
from src.io import array_digest
from src.low_rank_assignment import LowRankMoments
from src.moments import augmented, decode_moments, make_material
from src.nystrom_ce import _content_digest, moment_gradient, outer_gradient
from src.shared_features import _tensor_identity
from src.soft_ce_partition import solve_head_system, solve_inner_newton_first

SCHEMA = 1
DIRECT = "normalized_sole_original_Nystrom_CE_direct_U_v1"
GRAPH = "normalized_sole_original_Nystrom_CE_graph_S_U_v1"
OBJECTIVE = "sole_original_Nystrom_outer_CE_over_frozen_own_positive_CE0"
POLICY = dict(schema=SCHEMA, objective=OBJECTIVE, coefficient=1, mass_mode="free",
    head_weighting="uniform", teacher="original_frozen_raw_Q", kernel="relu",
    inverse="explicit_original_RMS_affine", bias_regularized=True,
    scale_floor=False, target_renormalization=False, original_moment_operator=True,
    original_native_cast=True, Adam_base_parameters="U,V", free_features=False)
WORK_KEYS = ("moment_forward_calls", "moment_backward_calls", "head_interfaces",
    "adjoint_solves", "Adam_steps", "P_updates", "S_forward_products",
    "T_transpose_products", "checkpoint_writes", "resume_writes")
ATTEMPT_KEYS = ("moment_forward_calls", "moment_backward_calls", "head_interfaces",
    "adjoint_solves", "Adam_steps", "checkpoint_writes", "resume_writes")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _int(value, minimum=0):
    _require(type(value) is int and value >= minimum, "Expected typed nonnegative integer")
    return value


def _num(value, positive=False):
    _require(type(value) in (int, float) and math.isfinite(value)
             and (not positive or value > 0), "Expected finite scalar or positive frozen scale")
    return value


def _plain(value):
    if isinstance(value, np.generic):
        value = value.item()
    if torch.is_tensor(value):
        return _tensor_identity(value.detach())
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    _require(value is None or type(value) in (str, bool, int)
             or type(value) is float and math.isfinite(value), "Nonfinite/non-JSON metadata")
    return value


def _seal(value):
    return hashlib.sha256(json.dumps(_plain(value), sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _cpu(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_cpu(v) for v in value]
    return value


def _matrix(value, shape, dtype=torch.float64, device=None):
    _require(torch.is_tensor(value) and value.layout == torch.strided
        and tuple(value.shape) == tuple(shape) and value.dtype == dtype
        and (device is None or value.device == device) and bool(torch.isfinite(value).all()),
        "Matrix shape/dtype/device/finiteness differs")
    return value


def _decoded(moments, dimension):
    _int(dimension, 1)
    _require(torch.is_tensor(moments) and moments.ndim == 2, "Expected FP64 base moments")
    k, width = moments.shape
    _require(k >= 2 and width > dimension + 2, "Cell/class dimensions differ")
    _matrix(moments, (k, width))
    centers, labels, mass = decode_moments(moments, dimension)
    _require(bool((mass > 0).all()) and bool(torch.isfinite(centers).all())
        and bool(torch.isfinite(labels).all()) and bool((labels >= 0).all())
        and bool((labels.sum(1) > 0).all()), "Positive-mass/general-target domain violated")
    return centers, labels, mass


class _RMSMap:
    def __init__(self, feature_map, transform):
        self.feature_map, self.transform = feature_map, transform

    def __call__(self, centers):
        return self.feature_map(centers * self.transform.scale
            + self.transform.output_center + self.transform.center)


def mapped_centroids(moments, dimension, feature_map, transform):
    """Original literal map of the explicit RMS inverse; no unit-target guard."""
    centers, _, _ = _decoded(moments, dimension)
    _transform(transform, dimension, moments.device)
    width, _ = _map(feature_map, dimension, moments.device)
    return _matrix(_RMSMap(feature_map, transform)(centers),
        (len(moments), width), device=moments.device)


def raw_moment_cotangent(moments, dimension, theta, vector, penalty, feature_map, transform):
    """Raw outer-CE implicit base-M partial; original backward supplies 1/N."""
    _decoded(moments, dimension)
    width, _ = _map(feature_map, dimension, moments.device)
    _transform(transform, dimension, moments.device)
    classes = moments.shape[1] - dimension - 1
    _matrix(theta, (classes, width + 1), device=moments.device)
    _matrix(vector, theta.shape, device=moments.device)
    _num(penalty, positive=True)
    value = moment_gradient(moments, dimension, _RMSMap(feature_map, transform),
        theta.detach(), vector.detach(), penalty, inner_loss_weighting="uniform")
    return _matrix(value.detach(), moments.shape, device=moments.device)


def complete_moment_cotangent(moments, dimension, theta, vector, penalty,
        feature_map, transform, CE0):
    """Detached FP64 complete cotangent normalized by own frozen CE0 once."""
    _num(CE0, positive=True)
    value = raw_moment_cotangent(moments, dimension, theta, vector, penalty,
        feature_map, transform) / CE0
    return _matrix(value.detach(), moments.shape, device=moments.device)


def _runtime(device):
    return dict(Python=platform.python_version(), Torch=str(torch.__version__),
        NumPy=np.__version__, threads=torch.get_num_threads(), device=str(device),
        default_dtype=str(torch.get_default_dtype()), AMP=torch.is_autocast_enabled(device.type),
        TF32_matmul=torch.backends.cuda.matmul.allow_tf32,
        TF32_cudnn=torch.backends.cudnn.allow_tf32,
        matmul_precision=torch.get_float32_matmul_precision(),
        deterministic=torch.are_deterministic_algorithms_enabled())


def _context(context):
    keys = {"schema", "mode", "policy", "source_refs", "native_parameter_digests",
        "asset_descriptors", "native_origin", "operator"}
    _require(isinstance(context, dict) and set(context) == keys
        and type(context["schema"]) is int and context["schema"] == SCHEMA
        and context["mode"] in (DIRECT, GRAPH) and _seal(context["policy"]) == _seal(POLICY),
        "Sole-head context policy/schema/keys differ")
    refs, assets = context["source_refs"], context["asset_descriptors"]
    _require(isinstance(refs, dict), "Source references must be a sealed object")
    for key in ("nodes", "cells", "rank", "dimension", "classes", "chunk_size"):
        _int(refs.get(key), 1)
    _int(refs.get("factor_seed"))
    _require(refs["cells"] >= 2 and refs["classes"] >= 2
        and refs["rank"] <= min(refs["nodes"], refs["cells"]) and refs.get("mixing") == .05
        and isinstance(refs.get("original_options"), dict)
        and isinstance(refs.get("runtime"), dict) and type(refs.get("device")) is str
        and type(refs.get("data_digest")) is str and len(refs["data_digest"]) == 64,
        "Native source/recipe references differ")
    native = context["native_parameter_digests"]
    _require(isinstance(native, list) and len(native) == 2 and all(type(h) is str
        and len(h) == 64 and all(c in "0123456789abcdef" for c in h) for h in native),
        "Native U0/V0 digests missing")
    _require(isinstance(assets, dict) and set(assets) == {"H", "transform", "anchors",
        "mapping", "Phi_identity", "z", "Q", "assignment"}, "Frozen assets differ")
    phi = assets["Phi_identity"]
    _require(isinstance(phi, dict) and set(phi) == {"schema", "h_digest", "map_digest",
        "shape", "dtype", "phi_digest"} and type(phi["schema"]) is int and phi["schema"] == 1
        and phi["dtype"] == "float64" and isinstance(phi["shape"], list)
        and len(phi["shape"]) == 2 and phi["shape"][0] == refs["nodes"], "Phi identity differs")
    width = _int(phi["shape"][1], 1)
    _require(isinstance(context["native_origin"], dict)
        and set(context["native_origin"]) == {"moments", "centers", "labels"}
        and ((context["mode"] == DIRECT and context["operator"] is None)
             or (context["mode"] == GRAPH and isinstance(context["operator"], dict))),
        "Native physical P0 or graph descriptor missing")
    return refs, assets, width


def _validate_inputs(z, q, assignment, initial, feature_map, phi, transform, options, context, operator):
    refs, assets, width = _context(context)
    old = _native_options(options)
    _require(isinstance(z, torch.Tensor), "Source z must be a matrix")
    n, k, r, d, c = (refs[x] for x in ("nodes", "cells", "rank", "dimension", "classes"))
    _matrix(z, (n, d), device=z.device); _matrix(q, (n, c), device=z.device)
    _require(not z.requires_grad and not q.requires_grad and bool((q >= 0).all())
        and bool((q.sum(1) > 0).all()) and z.device.type in ("cpu", "cuda"), "Raw source Q/z domain differs")
    _require(torch.is_tensor(assignment) and assignment.shape == (n,)
        and assignment.dtype == torch.int64 and assignment.device == z.device
        and not assignment.requires_grad and int(assignment.min()) >= 0
        and int(assignment.max()) == k - 1, "Original hard-cell ordering differs")
    data = array_digest(z.cpu().numpy(), q.cpu().numpy(), assignment.cpu().numpy())
    _require(data == refs["data_digest"] and old == refs["original_options"]
        and ("data_digest" not in options or options["data_digest"] == data)
        and refs["device"] == str(z.device) and options["assignment_rank"] == r
        and options["factor_seed"] == refs["factor_seed"] and options["chunk_size"] == refs["chunk_size"]
        and _runtime(z.device) == refs["runtime"] and torch.get_default_dtype() == torch.float32
        and not torch.is_autocast_enabled(z.device.type), "Source/options/runtime changed")
    for key, value in (("z", z), ("Q", q), ("assignment", assignment)):
        _require(_tensor_identity(value) == assets[key], "Original source array identity changed")
    _require(_factor_digests(initial, n, k, r) == context["native_parameter_digests"]
        and bool(initial[0].eq(0).all()), "Original zeroU/native GaussianV origin changed")
    actual_width, desc = _map(feature_map, d, z.device)
    _require(width == actual_width and desc["anchors"] == assets["anchors"]
        and desc["mapping"] == assets["mapping"]
        and _transform(transform, d, z.device) == assets["transform"], "Original map/RMS changed")
    if torch.is_tensor(phi):
        _matrix(phi, (n, width), device=torch.device("cpu")); _require(not phi.requires_grad, "Phi is not detached CPU cache")
    else:
        _require(isinstance(phi, np.ndarray) and phi.shape == (n, width)
            and phi.dtype == np.float64 and not phi.flags.writeable, "Phi must be original read-only FP64 cache")
    _require(_content_digest(phi, canonical_double=True) == assets["Phi_identity"]["phi_digest"], "Phi content changed")
    paths, pins = refs.get("asset_paths"), refs.get("files_sha256")
    _require(isinstance(paths, dict) and set(paths) == {"H", "map", "Phi", "Phi_metadata"}
        and isinstance(pins, dict) and all(type(p) is str and p in pins for p in paths.values())
        and Path(paths["Phi_metadata"]).is_file(), "Original asset paths/mandatory sidecar unpinned")
    _files(pins)
    _require(json.loads(Path(paths["Phi_metadata"]).read_text()) == assets["Phi_identity"], "Phi sidecar changed")
    source = refs.get("current_source")
    _require(isinstance(source, dict) and type(source.get("git_head")) is str
        and type(source.get("source_digest")) is str, "Current implementation is unbound")
    _files(source.get("files"))
    _require(refs.get("optimizer_source_sha256") == hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "Optimizer code pin changed")
    _require((context["mode"] == DIRECT and operator is None) or
        (context["mode"] == GRAPH and isinstance(operator, FrozenCPUCSR)
         and operator.descriptor() == context["operator"]), "Exact graph mode/operator changed")
    return dict(old, data_digest=data, sole_nystrom_mode=context["mode"], sole_nystrom_context=context)


def _record(record, config, context):
    refs, _, width = _context(context)
    keys = {"step", "config", "context", "moments", "theta", "vector", "vector_step",
        "parameters", "initial_parameters", "effective_u", "CE", "CE0", "objective",
        "objective_name", "J_exact", "head_work", "warm_before", "last_pullback", "record_digest"}
    _require(isinstance(record, dict) and set(record) == keys and record["J_exact"] is True
        and record["objective_name"] == OBJECTIVE and _seal(record["config"]) == _seal(config)
        and _seal(record["context"]) == _seal(context), "Record/config/context changed")
    step = _int(record["step"])
    n, k, r, d, c = (refs[x] for x in ("nodes", "cells", "rank", "dimension", "classes"))
    _factor_digests(record["parameters"], n, k, r)
    _require(_factor_digests(record["initial_parameters"], n, k, r)
        == context["native_parameter_digests"], "Record lost native P0 parameters")
    _require(all(p.device.type == "cpu" and not p.requires_grad
        for pair in (record["parameters"], record["initial_parameters"]) for p in pair), "Saved factors do not own detached CPU state")
    _matrix(record["moments"], (k, 1 + d + c), device=torch.device("cpu"))
    _decoded(record["moments"], d)
    _matrix(record["theta"], (c, width + 1), device=torch.device("cpu"))
    _matrix(record["effective_u"], (n, r), dtype=torch.float32, device=torch.device("cpu"))
    if context["mode"] == DIRECT:
        _require(_seal(record["effective_u"]) == _seal(record["parameters"][0]), "Direct W is not its owning base U")
    if record["vector"] is not None:
        _matrix(record["vector"], (c, width + 1), device=torch.device("cpu"))
        _require(_int(record["vector_step"]) <= step, "Adjoint frontier is ahead of factors")
    else:
        _require(record["vector_step"] is None, "Missing adjoint retained a step")
    head = record["head_work"]
    _require(isinstance(head, dict) and head.get("inner_converged") is True
        and 0 <= _num(head.get("inner_grad_max")) <= config["inner_tol"], "Head is not stationary")
    _num(record["CE"], positive=True); _num(record["CE0"], positive=True)
    _require(_num(record["objective"]) == record["CE"] / record["CE0"], "Fixed CE0 units changed")
    warm = record["warm_before"]
    _require(isinstance(warm, dict) and set(warm) == {"theta", "vector", "vector_step", "parent_digest"}, "Warm lineage missing")
    for name in ("theta", "vector"):
        if warm[name] is not None:
            _matrix(warm[name], (c, width + 1), device=torch.device("cpu"))
    _require((warm["vector"] is None) == (warm["vector_step"] is None), "Warm vector lost its endpoint")
    if warm["vector_step"] is not None:
        _require(_int(warm["vector_step"]) <= step, "Warm vector is ahead of frontier")
    pullback = record["last_pullback"]
    if step == 0:
        _require(pullback is None, "P0 cannot carry a previous P update")
    else:
        _require(isinstance(pullback, dict) and set(pullback) == {"step", "gW", "gU", "gV"}
            and type(pullback["step"]) is int and pullback["step"] == step - 1, "Actual pullback endpoint missing")
        for name, shape in (("gW", (n, r)), ("gU", (n, r)), ("gV", (k, r))):
            _matrix(pullback[name], shape, dtype=torch.float32, device=torch.device("cpu"))
    _require(record["record_digest"] == _seal({k: v for k, v in record.items() if k != "record_digest"}), "Record seal changed")


def _same_material(old, new):
    _require(all(_seal(old[key]) == _seal(new[key]) for key in
        ("parameters", "initial_parameters", "effective_u", "moments", "config", "context", "CE0")),
        "Inclusive frontier changed accepted factors/W/M/source or CE0")


def _record_link(record):
    return dict(record_digest=record["record_digest"], record_identity=_plain({key: record[key]
        for key in ("parameters", "initial_parameters", "moments", "effective_u", "theta",
            "vector", "vector_step", "CE0", "warm_before", "last_pullback")}))


def _history_link(row, record):
    _require(all(row.get(key) == value for key, value in _record_link(record).items())
        and all(record[key] == row.get(key) for key in ("CE", "CE0", "objective", "head_work")),
        "Full record/history owning-factor/head/vector linkage changed")


def _warm_link(record, parent_row):
    expected = dict(theta=None, vector=None, vector_step=None, parent_digest=None)
    if parent_row is not None:
        identity = parent_row["record_identity"]
        expected = dict(theta=identity["theta"], vector=identity["vector"],
            vector_step=identity["vector_step"], parent_digest=parent_row["record_digest"])
    _require(_plain(record["warm_before"]) == expected, "Actual warm head/vector/parent provenance changed")


def _history_identity(row, refs, width):
    identity = row.get("record_identity")
    keys = {"parameters", "initial_parameters", "moments", "effective_u", "theta", "vector",
        "vector_step", "CE0", "warm_before", "last_pullback"}
    _require(isinstance(identity, dict) and set(identity) == keys, "History tensor descriptors missing")
    def descriptor(value, shape, dtype):
        _require(isinstance(value, dict) and set(value) == {"shape", "dtype", "digest"}
            and value["shape"] == list(shape) and value["dtype"] == dtype
            and type(value["digest"]) is str and len(value["digest"]) == 64
            and all(c in "0123456789abcdef" for c in value["digest"]), "History shape/dtype/digest changed")
    n, k, r, d, c = (refs[x] for x in ("nodes", "cells", "rank", "dimension", "classes"))
    for name in ("parameters", "initial_parameters"):
        _require(isinstance(identity[name], list) and len(identity[name]) == 2, "History U/V pair lost")
        descriptor(identity[name][0], (n, r), "torch.float32"); descriptor(identity[name][1], (k, r), "torch.float32")
    for name, shape, dtype in (("moments", (k, 1 + d + c), "torch.float64"),
            ("effective_u", (n, r), "torch.float32"), ("theta", (c, width + 1), "torch.float64")):
        descriptor(identity[name], shape, dtype)
    if identity["vector"] is None:
        _require(identity["vector_step"] is None, "History lost vector endpoint")
    else:
        descriptor(identity["vector"], (c, width + 1), "torch.float64")
        _require(_int(identity["vector_step"]) <= row["step"], "History vector is ahead of factors")
    return identity


def validate_core_resume(state, expected_config, context, folder=None):
    """Validate typed source/origin/Adam/history/immutable prefix, without fits."""
    refs, _, width = _context(context)
    bound = dict(refs["original_options"], data_digest=refs["data_digest"],
        sole_nystrom_mode=context["mode"], sole_nystrom_context=context)
    keys = {"schema", "step", "config", "context", "parameters", "initial_parameters",
        "optimizer", "snapshots", "history", "current", "best_record", "frontiers",
        "initial_moments", "CE0", "work", "attempts", "checkpoint_files_sha256",
        "history_sha256", "state_digest"}
    _require(isinstance(state, dict) and set(state) == keys and type(state["schema"]) is int
        and state["schema"] == SCHEMA and _seal(expected_config) == _seal(bound)
        and _seal(state["config"]) == _seal(bound) and _seal(state["context"]) == _seal(context), "Resume schema/config/context changed")
    end = _int(state["step"])
    snaps, history, frontiers = state["snapshots"], state["history"], state["frontiers"]
    _require(isinstance(snaps, dict) and 0 in snaps and end in snaps
        and (end == 0 or 1 in snaps) and isinstance(history, list) and len(history) == end + 1
        and all(isinstance(row, dict) for row in history)
        and [row.get("step") for row in history] == list(range(end + 1)), "Complete typed prefix missing")
    for step, rec in snaps.items():
        _require(type(step) is int and 0 <= step <= end and rec.get("step") == step, "Checkpoint key changed")
        _record(rec, bound, context)
    for rec in (state["current"], state["best_record"]):
        _record(rec, bound, context)
    zero, current = snaps[0], state["current"]
    _require(current["step"] == end and _num(state["CE0"], positive=True) == zero["CE0"]
        and zero["CE"] == zero["CE0"] and zero["objective"] == 1.0
        and _seal(state["initial_moments"]) == _seal(zero["moments"])
        and _seal(state["parameters"]) == _seal(current["parameters"])
        and _factor_digests(state["initial_parameters"], refs["nodes"], refs["cells"], refs["rank"])
            == context["native_parameter_digests"]
        and _seal(state["initial_parameters"]) == _seal(zero["initial_parameters"])
        and _seal(zero["parameters"]) == _seal(zero["initial_parameters"])
        and bool(zero["parameters"][0].eq(0).all()), "Original P0/current factor lineage changed")
    cz0, qc0, _ = _decoded(zero["moments"], refs["dimension"])
    _require(context["native_origin"] == dict(moments=_tensor_identity(zero["moments"]),
        centers=_tensor_identity(cz0), labels=_tensor_identity(qc0)), "Original physical M0/serving bits changed")
    _same_material(snaps[end], current)
    _require(isinstance(frontiers, dict), "Inclusive frontier provenance missing")
    canonical = list(history)
    for step, provenance in frontiers.items():
        _require(type(step) is int and step in snaps and 0 <= step <= end
            and isinstance(provenance, dict) and set(provenance) == {"accepted_history",
                "warm_parent_history", "warm_parent_record", "reevaluated_record"}, "Frontier provenance keys changed")
        rec = provenance["reevaluated_record"]
        _record(rec, bound, context); _same_material(snaps[step], rec)
        parent = provenance["warm_parent_record"]
        _record(parent, bound, context); _same_material(snaps[step], parent)
        _require(rec["step"] == parent["step"] == step
            and provenance["accepted_history"]["step"] == provenance["warm_parent_history"]["step"] == step,
            "Accepted frontier endpoint changed")
        _history_link(provenance["accepted_history"], snaps[step])
        _history_link(provenance["warm_parent_history"], parent)
        _warm_link(rec, provenance["warm_parent_history"])
        canonical[step] = provenance["accepted_history"]
    for step, row in enumerate(history):
        _require(type(row.get("step")) is int and row["step"] == step
            and row.get("J_exact") is True and row.get("objective_name") == OBJECTIVE
            and _num(row.get("CE0"), positive=True) == zero["CE0"]
            and _num(row.get("CE"), positive=True) / zero["CE0"] == _num(row.get("objective"))
            and _num(row.get("J")) == row["objective"] and row.get("status") in ("update", "evaluated")
            and (step == end or row["status"] == "update")
            and row.get("head_work", {}).get("inner_converged") is True
            and 0 <= _num(row["head_work"].get("inner_grad_max")) <= bound["inner_tol"], "History lost stationary CE/scales/status")
        if row["status"] == "update":
            _require(row.get("adjoint_work", {}).get("evaluated") is True
                and row["adjoint_work"].get("cg_converged") is True, "Update lost converged adjoint")
        rec = frontiers[step]["reevaluated_record"] if step in frontiers else snaps.get(step)
        if step == end:
            rec = current
            if row["status"] == "update":
                _require(current["vector"] is not None and current["vector_step"] == end, "Persisted update frontier lost solved vector")
        if rec is not None:
            _require(rec["CE0"] == zero["CE0"], "Record changed the frozen own scale")
            _history_link(row, rec)
            if step not in frontiers:
                _warm_link(rec, None if step == 0 else history[step - 1])
        identity = _history_identity(row, refs, width)
        _require(type(row.get("record_digest")) is str and len(row["record_digest"]) == 64
            and all(c in "0123456789abcdef" for c in row["record_digest"])
            and identity["CE0"] == zero["CE0"], "History record descriptor missing")
        if step not in frontiers:
            expected_warm = dict(theta=None, vector=None, vector_step=None, parent_digest=None) if step == 0 else dict(
                theta=history[step - 1]["record_identity"]["theta"], vector=history[step - 1]["record_identity"]["vector"],
                vector_step=history[step - 1]["record_identity"]["vector_step"], parent_digest=history[step - 1]["record_digest"])
            _require(identity["warm_before"] == expected_warm, "Intermediate history warm lineage changed")
    best_step = min(range(end + 1), key=lambda s: canonical[s]["objective"])
    _require(state["best_record"]["step"] == best_step and all(state["best_record"][k]
        == canonical[best_step][k] for k in ("CE", "CE0", "objective")), "Earliest canonical objective best changed")
    _history_link(canonical[best_step], state["best_record"])
    for step, rec in snaps.items():
        _require(rec["CE0"] == zero["CE0"], "Snapshot changed the frozen own scale")
        _history_link(canonical[step], rec)
        _warm_link(rec, None if step == 0 else history[step - 1])
    work, attempts = state["work"], state["attempts"]
    _require(isinstance(work, dict) and set(work) == set(WORK_KEYS)
        and isinstance(attempts, dict) and set(attempts) == set(ATTEMPT_KEYS), "Actual work counters missing")
    for value in (*work.values(), *attempts.values()):
        _int(value)
    _require(all(attempts[k] >= work[k] for k in ATTEMPT_KEYS)
        and work["P_updates"] == work["Adam_steps"] == work["moment_backward_calls"] == end
        and _validate_optimizer(state["optimizer"], bound, refs) == end
        and work["head_interfaces"] == work["moment_forward_calls"] >= end + 1
        and work["adjoint_solves"] >= end and work["checkpoint_writes"] == len(snaps)
        and work["resume_writes"] >= len(snaps), "Actual optimizer/one-head/trajectory counts changed")
    _require((context["mode"] == DIRECT and work["S_forward_products"] == work["T_transpose_products"] == 0)
        or (context["mode"] == GRAPH and work["S_forward_products"] == work["moment_forward_calls"]
            and work["T_transpose_products"] == work["moment_backward_calls"]), "Graph actual product counts changed")
    pins = state["checkpoint_files_sha256"]
    _require(isinstance(pins, dict) and set(pins) == {f"step_{s:06d}.pt" for s in snaps}
        and state["state_digest"] == _seal({k: v for k, v in state.items() if k != "state_digest"}), "Resume/file/array seal changed")
    if folder is not None:
        _files({str(Path(folder) / "checkpoints" / p): h for p, h in pins.items()})
        _files({str(Path(folder) / "optimization.csv"): state["history_sha256"]})
    return state


def _write_state(value, path, exclusive=False):
    path = Path(path)
    target = path if exclusive else path.with_suffix(path.suffix + ".tmp")
    with target.open("xb") as stream:
        torch.save(value, stream); stream.flush(); os.fsync(stream.fileno())
    if not exclusive:
        target.replace(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def optimize(z, q, assignment, initial_parameters, feature_map, phi, transform,
        folder, steps, options, context, operator=None, checkpoint_steps=(0, 1, 25),
        resume_state=None, stop=lambda: False):
    """One stationary mapped head and original Adam on base U,V; saved CPU state."""
    _int(steps); _require(callable(stop), "Stop must be callable")
    started = time.monotonic()
    def check():
        if stop() or time.monotonic() - started > 300:
            raise InterruptedError("Bounded sole-head invocation stopped; preserve last accepted prefix")
    check()
    config = _validate_inputs(z, q, assignment, initial_parameters, feature_map, phi, transform, options, context, operator)
    refs, _, _ = _context(context)
    _require(isinstance(checkpoint_steps, (list, tuple)) and all(type(s) is int and s >= 0 for s in checkpoint_steps), "Invalid checkpoint schedule")
    checkpoints, folder = {0, 1, steps, *checkpoint_steps}, Path(folder)
    _require(not (folder / "failure.json").exists(), "Failed namespace cannot be resumed or rescued")
    if resume_state is not None:
        validate_core_resume(resume_state, config, context, folder)
        disk = torch.load(folder / "resume.pt", map_location="cpu", weights_only=False)
        _require(_seal(disk) == _seal(resume_state) and resume_state["step"] <= steps, "Stale/earlier-horizon resume cannot overwrite current trajectory")
    else:
        _require(not (folder / "resume.pt").exists() and not any((folder / "checkpoints").glob("step_*.pt")), "Existing trajectory needs explicit verified resume")
    folder.mkdir(parents=True, exist_ok=True); (folder / "checkpoints").mkdir(exist_ok=True)
    cp = folder / "context.json"
    if cp.exists():
        _require(_seal(json.loads(cp.read_text())) == _seal(context), "Owned context changed")
    else:
        with cp.open("x") as stream:
            json.dump(context, stream, indent=2, allow_nan=False)
    params = [p.detach().to(device=z.device).clone().requires_grad_() for p in initial_parameters]
    initial, material = _cpu(initial_parameters), make_material(z, q)
    adam = torch.optim.Adam(params, lr=options["lr"], eps=1e-12, foreach=False)
    work, attempts = dict.fromkeys(WORK_KEYS, 0), dict.fromkeys(ATTEMPT_KEYS, 0)
    snaps, history, frontiers, pins = {}, [], {}, {}
    start, theta, vector, vector_step, CE0, last_pullback, best, parent = 0, None, None, None, None, None, None, None
    if resume_state is not None:
        with torch.no_grad():
            for p, saved in zip(params, resume_state["parameters"], strict=True):
                p.copy_(saved.to(p))
        adam.load_state_dict(resume_state["optimizer"])
        start, CE0 = resume_state["step"], resume_state["CE0"]
        snaps, history, frontiers, pins = (dict(resume_state[k]) if k != "history" else list(resume_state[k]) for k in ("snapshots", "history", "frontiers", "checkpoint_files_sha256"))
        work, attempts = dict(resume_state["work"]), dict(resume_state["attempts"])
        parent, best = resume_state["current"], resume_state["best_record"]
        theta = parent["theta"].to(z.device)
        vector = None if parent["vector"] is None else parent["vector"].to(z.device)
        vector_step, last_pullback = parent["vector_step"], parent["last_pullback"]
        if start not in frontiers:
            frontiers[start] = dict(accepted_history=history[start], warm_parent_history=history[start],
                warm_parent_record=parent, reevaluated_record=None)
        else:
            frontiers[start] = dict(frontiers[start], warm_parent_history=history[start], warm_parent_record=parent)
    baseline_counts = None if operator is None else operator.counts()
    prior_products = {k: work[k] for k in ("S_forward_products", "T_transpose_products")}
    def sync_products():
        if operator is not None:
            now = operator.counts()
            for key in prior_products:
                work[key] = prior_products[key] + now[key] - baseline_counts[key]
    def call(key, function, *args, **kwargs):
        attempts[key] += 1
        value = function(*args, **kwargs)
        work[key] += 1
        return value
    step = start
    try:
        for step in range(start, steps + 1):
            check(); adam.zero_grad(set_to_none=True)
            warm = _cpu(dict(theta=theta, vector=vector, vector_step=vector_step,
                parent_digest=None if parent is None else parent["record_digest"]))
            W = params[0] if operator is None else GraphProduct.apply(params[0], operator)
            W.retain_grad()
            M = call("moment_forward_calls", LowRankMoments.apply, W, params[1], assignment, material, .05, options["chunk_size"])
            sync_products()
            centers, labels, mass = _decoded(M.detach(), z.shape[1])
            if step == 0:
                _require(context["native_origin"] == dict(moments=_tensor_identity(M.detach()),
                    centers=_tensor_identity(centers), labels=_tensor_identity(labels)), "Original M0/student serving differs before any head/Adam")
            features = mapped_centroids(M.detach(), z.shape[1], feature_map, transform).detach()
            weights = torch.full_like(mass, 1 / len(mass)); check()
            fit = call("head_interfaces", solve_inner_newton_first, features, labels, weights,
                options["penalty"], theta, options["inner_max_iter"], options["inner_tol"],
                cg_max_iter=options["cg_max_iter"], cg_check_interval=options.get("cg_check_interval", 1))
            theta, head = fit["theta"].detach(), _plain({k: v for k, v in fit.items() if k != "theta"})
            _require(head.get("inner_converged") is True and 0 <= head.get("inner_grad_max", math.inf) <= options["inner_tol"], "Mapped stationary head failed")
            check(); CE, rhs = outer_gradient(phi, q, theta, options["outer_chunk_size"]); _num(CE, positive=True)
            if step == 0:
                if CE0 is None:
                    CE0 = CE
                _require(CE0 == CE, "Resumed own mapped P0 CE0 changed")
            _num(CE0, positive=True); objective = CE / CE0; check()
            diagnostic = dict(evaluated=False)
            if step < steps:
                vector, diagnostic = call("adjoint_solves", solve_head_system, augmented(features), labels,
                    weights, theta, options["penalty"], rhs, rtol=options["cg_rtol"],
                    max_iter=options["cg_max_iter"], initial=vector, cg_check_interval=options.get("cg_check_interval", 1))
                vector = vector.detach(); diagnostic = dict(_plain(diagnostic), evaluated=True)
                _require(diagnostic.get("cg_converged") is True and bool(torch.isfinite(vector).all()), "Mapped adjoint failed")
                vector_step = step
            row = dict(step=step, CE=CE, CE0=CE0, objective=objective, J=objective,
                objective_name=OBJECTIVE, J_exact=True, status="update" if step < steps else "evaluated",
                head_work=head, adjoint_work=diagnostic, min_mass=float(mass.min()),
                max_mass=float(mass.max()), effective_cells=float(1 / mass.square().sum()))
            history = history[:step] + [row]
            record = _cpu(dict(step=step, config=config, context=context, moments=M.detach(), theta=theta,
                vector=vector, vector_step=vector_step, parameters=params, initial_parameters=initial,
                effective_u=W, CE=CE, CE0=CE0, objective=objective, objective_name=OBJECTIVE,
                J_exact=True, head_work=head, warm_before=warm, last_pullback=last_pullback))
            record["record_digest"] = _seal(record)
            row.update(_record_link(record))
            if step in frontiers:
                _same_material(snaps[step], record); frontiers[step]["reevaluated_record"] = record
            eligible = snaps[step] if step in frontiers else record
            if best is None or eligible["objective"] < best["objective"]:
                best = eligible
            if step in checkpoints:
                if step not in snaps:
                    pins[f"step_{step:06d}.pt"] = call("checkpoint_writes", _write_state,
                        record, folder / "checkpoints" / f"step_{step:06d}.pt", exclusive=True)
                    snaps[step] = record
                else:
                    _same_material(snaps[step], record)
                csv_path = folder / "optimization.csv"; temp = csv_path.with_suffix(".csv.tmp")
                with temp.open("x") as stream:
                    pd.DataFrame(history).to_csv(stream, index=False); stream.flush(); os.fsync(stream.fileno())
                temp.replace(csv_path)
                next_work, next_attempts = dict(work), dict(attempts)
                next_work["resume_writes"] += 1; next_attempts["resume_writes"] += 1
                state = _cpu(dict(schema=SCHEMA, step=step, config=config, context=context, parameters=params,
                    initial_parameters=initial, optimizer=adam.state_dict(), snapshots=snaps, history=history,
                    current=record, best_record=best, frontiers=frontiers, initial_moments=snaps[0]["moments"],
                    CE0=CE0, work=next_work, attempts=next_attempts, checkpoint_files_sha256=pins,
                    history_sha256=hashlib.sha256(csv_path.read_bytes()).hexdigest()))
                state["state_digest"] = _seal(state)
                validate_core_resume(state, config, context, folder)
                call("resume_writes", _write_state, state, folder / "resume.pt")
            if step == steps:
                check(); _files(refs["files_sha256"]); _files(refs["current_source"]["files"])
                _require(_runtime(z.device) == refs["runtime"], "Runtime changed during invocation")
                return state
            check()
            G = complete_moment_cotangent(M.detach(), z.shape[1], theta, vector,
                options["penalty"], feature_map, transform, CE0)
            call("moment_backward_calls", M.backward, G); sync_products()
            _require(all(p.grad is not None and p.grad.dtype == torch.float32
                and bool(torch.isfinite(p.grad).all()) for p in params)
                and W.grad is not None and bool(torch.isfinite(W.grad).all()), "Native factor pullback is nonfinite")
            last_pullback = _cpu(dict(step=step, gW=W.grad, gU=params[0].grad, gV=params[1].grad))
            check(); call("Adam_steps", adam.step); work["P_updates"] += 1
            _require(all(bool(torch.isfinite(p).all()) for p in params), "Native Adam factors nonfinite")
            parent = record
    except BaseException as error:
        try:
            sync_products()
        except BaseException as observation_error:
            error.add_note("Product counter observation also failed: " + repr(observation_error))
        failure = dict(schema=SCHEMA, mode=context["mode"], step=step, error_type=type(error).__name__,
            error=str(error), elapsed_seconds=time.monotonic() - started, work=work, attempts=attempts,
            completed_returned_calls_only=True, interrupted_operation_interiors="unknown",
            qualification=False, no_retry_or_rescue=True)
        try:
            with (folder / "failure.json").open("x") as stream:
                json.dump(failure, stream, indent=2, allow_nan=False)
        except BaseException as observation_error:
            error.add_note("Failure metadata write also failed: " + repr(observation_error))
        raise
    raise AssertionError("Sole mapped fixed endpoint was not evaluated")
