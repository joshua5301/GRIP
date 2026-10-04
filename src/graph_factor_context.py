"""Pinned graph-factor origin and original-CE checkpoint lineage.

This module never evaluates probabilities, moments, heads or graph products.
It consumes the actual original-core outputs and owns their CPU snapshots.
GraphProduct and the unchanged LowRankMoments remain the numerical operators.
"""
import hashlib
import json
import math
from pathlib import Path
import weakref

import torch

from src.graph_factor import FrozenCPUCSR, MODE
from src.io import array_digest, cpu_state
from src.shared_features import _tensor_identity


SCHEMA = 1
OBJECTIVE = "original_teacher_CE_over_original_positive_CE0"
POLICY = dict(schema=SCHEMA, mode=MODE, parameterization="effective_U=S@base_U",
              critic="original_single_physical_linear_CE", mass_mode="free",
              inner_loss_weighting="uniform", source_Q="original_frozen_teacher",
              initial_U="pinned_native_exact_zero", initial_V="pinned_native_Gaussian",
              original_LowRankMoments=True, original_Adam=True, mixing=.05,
              graph_gradient=False, extra_normalization=False, fallback=False,
              save_assignment=False, original_scale_floor_inactive_above_1e_12=True,
              best_policy="original_strict_first_improvement_including_accepted_frontier")
CONTEXT_KEYS = {"schema", "mode", "source_refs", "native_parameter_digests", "operator", "native_origin"}
OPTION_KEYS = {"graph_factor_mode", "graph_factor_artifact", "graph_factor_sha256", "graph_factor_context"}
_COUNT_OFFSETS = weakref.WeakKeyDictionary()
_ACCEPTED_BEST = weakref.WeakKeyDictionary()
_ACCEPTED_FRONTIERS = weakref.WeakKeyDictionary()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _json(value):
    try:
        return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError("Graph factor context must be finite JSON") from error


def _seal(value, *, history=False):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=history).encode()).hexdigest()


def _number(value, *, positive=False):
    _require(type(value) in (int, float) and math.isfinite(value)
             and (not positive or value > 0), "Graph factor scalar is nonfinite or invalid")
    return value


def _hash(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _matrix(value, shape, *, dtype, device=None, detached=True):
    _require(torch.is_tensor(value) and value.layout == torch.strided
             and tuple(value.shape) == tuple(shape) and value.dtype == dtype
             and (device is None or value.device == device)
             and (not detached or not value.requires_grad) and bool(torch.isfinite(value).all()),
             "Graph factor tensor shape/dtype/device/finiteness differs")
    return value


def _dimensions(context):
    refs = context["source_refs"]
    result = {key: refs.get(key) for key in ("nodes", "cells", "rank", "dimension", "classes")}
    _require(all(type(value) is int and value > 0 for value in result.values())
             and result["cells"] >= 2 and result["classes"] >= 2
             and result["rank"] <= min(result["nodes"], result["cells"]),
             "Graph factor source dimensions are invalid")
    return result


def _factors(parameters, dims, *, device=None, detached=True):
    _require(isinstance(parameters, (list, tuple)) and len(parameters) == 2,
             "Graph factor requires exactly native baseU/V")
    for value, shape in zip(parameters, ((dims["nodes"], dims["rank"]),
                                       (dims["cells"], dims["rank"])), strict=True):
        _matrix(value, shape, dtype=torch.float32, device=device, detached=detached)
    return [array_digest(value.detach().cpu().numpy()) for value in parameters]


def _moments(value, dims):
    _matrix(value, (dims["cells"], 1 + dims["dimension"] + dims["classes"]),
            dtype=torch.float64, device=torch.device("cpu"))
    mass, totals = value[:, :1], value[:, 1 + dims["dimension"]:]
    _require(bool((mass > 0).all()) and bool((totals >= 0).all())
             and abs(float(mass.sum()) - 1) <= 1e-12
             and bool((totals.sum(1) > 0).all()),
             "Graph factor original moment probability domain differs")
    return value


def _theta(value, dims):
    return _matrix(value, (dims["classes"], dims["dimension"] + 1),
                   dtype=torch.float64, device=torch.device("cpu"))


def _files(pins):
    _require(isinstance(pins, dict) and bool(pins), "Graph factor source/input file pins are missing")
    for name, expected in pins.items():
        _require(isinstance(name, str) and _hash(expected), "Graph factor file pin is malformed")
        with Path(name).open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        _require(actual == expected, "Graph factor immutable source/input bytes changed")


def _context(context, *, files=False):
    _require(isinstance(context, dict) and set(context) == CONTEXT_KEYS
             and type(context["schema"]) is int and context["schema"] == SCHEMA
             and context["mode"] == MODE and isinstance(context["source_refs"], dict),
             "Graph factor context/schema/mode differs")
    _json(context)
    dims = _dimensions(context)
    refs, operator, origin = context["source_refs"], context["operator"], context["native_origin"]
    _require(isinstance(operator, dict) and operator.get("schema") == SCHEMA
             and operator.get("mode") == MODE and operator.get("nodes") == dims["nodes"]
             and operator.get("rank") == dims["rank"]
             and operator.get("shape") == [dims["nodes"], dims["nodes"]]
             and _seal(operator.get("source_refs")) == _seal(refs)
             and isinstance(origin, dict)
             and set(origin) == {"moments", "theta", "teacher_ce", "J_exact", "inner_grad_max"}
             and origin["J_exact"] is True and _number(origin["teacher_ce"], positive=True) > 1e-12
             and _number(origin["inner_grad_max"]) >= 0
             and isinstance(context["native_parameter_digests"], list)
             and len(context["native_parameter_digests"]) == 2
             and all(_hash(value) for value in context["native_parameter_digests"]),
             "Graph factor operator/native origin declaration differs")
    _require(_hash(refs.get("data_digest")) and type(refs.get("factor_seed")) is int
             and refs.get("mixing") == .05 and type(refs.get("chunk_size")) is int
             and refs["chunk_size"] > 0 and isinstance(refs.get("device"), str)
             and isinstance(refs.get("original_config"), dict)
             and isinstance(refs.get("source_graph"), dict) and bool(refs["source_graph"]),
             "Graph factor source graph/native config declaration is missing")
    source = refs.get("current_source")
    _require(isinstance(source, dict) and isinstance(source.get("git_head"), str)
             and bool(source["git_head"]) and isinstance(source.get("source_digest"), str)
             and bool(source["source_digest"])
             and source["git_head"] == operator.get("runtime", {}).get("implementation", {}).get("git_head"),
             "Graph factor current source/Git declaration differs")
    if files:
        _files(refs.get("files_sha256"))
        _files(source.get("files"))
    return dims


def _origin(snapshot, dims):
    _require(isinstance(snapshot, dict) and snapshot.get("step") == 0
             and type(snapshot.get("step")) is int and snapshot.get("J_exact") is True,
             "Graph factor native origin must be the original exact step0")
    moments, theta = _moments(snapshot.get("moments"), dims), _theta(snapshot.get("theta"), dims)
    ce = _number(snapshot.get("teacher_ce"), positive=True)
    grad = _number(snapshot.get("inner_grad_max"))
    _require(ce > 1e-12 and grad >= 0, "Graph factor original CE0 must strictly exceed the legacy floor")
    return dict(moments=_tensor_identity(moments), theta=_tensor_identity(theta),
                teacher_ce=ce, J_exact=True, inner_grad_max=grad)


def build_packet(source_CSR, native_parameters, native_snapshot, source_refs):
    """Copy the already-pinned native S/U0/V0/M0/head; no factory or numerical replay."""
    _require(isinstance(source_refs, dict) and bool(source_refs), "Graph factor source refs are missing")
    _require(isinstance(native_snapshot, dict) and all(key in native_snapshot for key in
             ("step", "moments", "theta", "teacher_ce", "inner_grad_max", "J_exact")),
             "Graph factor original native checkpoint0 is incomplete")
    refs = _json(source_refs)
    dims = {key: refs.get(key) for key in ("nodes", "cells", "rank", "dimension", "classes")}
    _require(all(type(v) is int and v > 0 for v in dims.values()), "Graph factor packet needs typed dimensions")
    native = cpu_state(native_parameters)
    digests = _factors(native, dims, device=torch.device("cpu"))
    _require(bool(native[0].eq(0).all()), "Graph factor native U0 must be exact zero")
    origin_snapshot = cpu_state({key: native_snapshot[key] for key in
                               ("step", "moments", "theta", "teacher_ce", "inner_grad_max", "J_exact")})
    origin = _origin(origin_snapshot, dims)
    operator = FrozenCPUCSR.from_torch(source_CSR, dims["rank"], source_refs=refs)
    context = dict(schema=SCHEMA, mode=MODE, source_refs=refs,
                   native_parameter_digests=digests, operator=operator.descriptor(), native_origin=origin)
    _context(context, files=True)
    buffers = [value.detach().cpu().contiguous().numpy().copy(order="C")
               for value in (source_CSR.crow_indices(), source_CSR.col_indices(), source_CSR.values())]
    return dict(schema=SCHEMA, mode=MODE, context=_json(context),
                S=dict(zip(("rowptr", "col", "values"), buffers, strict=True)),
                native_parameters=native, native_snapshot=origin_snapshot)


def packet_context(packet):
    _require(isinstance(packet, dict) and type(packet.get("schema")) is int
             and packet["schema"] == SCHEMA and packet.get("mode") == MODE,
             "Graph factor artifact schema/mode differs")
    _context(packet.get("context"))
    return _json(packet["context"])


def load_frozen_graph(path, sha256, dimensions, device, expected_context):
    """Consume sealed buffers and native factors; never reconstruct S, P0 or RNG."""
    _require(isinstance(path, str) and _hash(sha256), "Graph factor artifact path/hash is invalid")
    path, device = Path(path), torch.device(device)
    _require(device.type in ("cpu", "cuda") and torch.get_default_dtype() == torch.float32
             and not torch.is_autocast_enabled(device.type), "Graph factor requires native FP32 CPU/CUDA")
    with path.open("rb") as stream:
        _require(hashlib.file_digest(stream, "sha256").hexdigest() == sha256,
                 "Graph factor artifact bytes differ")
    packet = torch.load(path, map_location="cpu", weights_only=False)
    _require(isinstance(packet, dict) and set(packet) ==
             {"schema", "mode", "context", "S", "native_parameters", "native_snapshot"},
             "Graph factor artifact packet fields differ")
    context = packet_context(packet)
    dims = _context(context, files=True)
    _require(_seal(context) == _seal(expected_context) and dimensions == dims,
             "Graph factor frozen artifact/context/dimensions differ")
    buffers = packet["S"]
    _require(isinstance(buffers, dict) and set(buffers) == {"rowptr", "col", "values"},
             "Graph factor frozen CSR fields differ")
    operator = FrozenCPUCSR(buffers["rowptr"], buffers["col"], buffers["values"], dims["rank"],
                            source_refs=context["source_refs"])
    _require(_seal(operator.descriptor()) == _seal(context["operator"]),
             "Graph factor sealed CSR/runtime/operator differs")
    native = cpu_state(packet["native_parameters"])
    _require(_factors(native, dims, device=torch.device("cpu")) == context["native_parameter_digests"]
             and bool(native[0].eq(0).all()) and _origin(packet["native_snapshot"], dims) == context["native_origin"],
             "Graph factor pinned native factors/M0/head/CE0 differ")
    with path.open("rb") as stream:
        _require(hashlib.file_digest(stream, "sha256").hexdigest() == sha256,
                 "Graph factor artifact changed while loading")
    return dict(operator=operator, initial_parameters=[p.detach().to(device=device).clone() for p in native],
                native_snapshot=cpu_state(packet["native_snapshot"]), context=_json(context))


def _config(config, context):
    _require(isinstance(config, dict) and config.get("graph_factor_mode") == MODE
             and isinstance(config.get("graph_factor_artifact"), str) and _hash(config.get("graph_factor_sha256"))
             and _seal(config.get("graph_factor_context")) == _seal(context),
             "Graph factor activated config binding differs")
    original = {key: value for key, value in config.items() if key not in OPTION_KEYS}
    _require(_seal(original) == _seal(context["source_refs"]["original_config"]),
             "Graph factor changed the original native config or default-key omissions")
    expected = dict(assignment_input="node", assignment_encoder="linear", feature_control="joint",
                    mass_mode="free", inner_loss_weighting="uniform", solver_mode="exact",
                    inner_method="newton_first", implicit_warm_start=True, save_assignment=False, mixing=.05)
    _require(all(type(config.get(key)) is type(value) and config.get(key) == value
                 for key, value in expected.items())
             and config.get("node_weighting", False) is False and config.get("cache_assignment", False) is False
             and type(config.get("cg_check_interval", 1)) is int and config.get("cg_check_interval", 1) > 0,
             "Graph factor only supports unchanged original free/uniform exact native CE")
    closed = ("mlp_initial", "mlp_output", "mlp_source", "source_linear", "uniform_cell_q_prior",
              "assignment_kl", "node_factor", "graph_assignment_kl", "kernel_commutation",
              "conditional_label_entropy", "soft_cell_mass_KL", "dual_head")
    _require(not any(key.startswith(closed) for key in config),
             "Graph factor cannot combine closed auxiliary methods")


def _counts(operator):
    current = operator.counts()
    offset = _COUNT_OFFSETS.get(operator, dict(S_forward_products=0, T_transpose_products=0))
    _require(set(current) == set(offset) == {"S_forward_products", "T_transpose_products"}
             and all(type(value) is int and value >= 0 for value in (*current.values(), *offset.values())),
             "Graph factor operator work counts are invalid")
    return {key: current[key] + offset[key] for key in current}


def validate_context_and_resume(z, q, assignment, initial_parameters, operator, context, config,
                                steps, resume_state=None, folder=None):
    """Run before loading updated base parameters or Adam slots."""
    dims = _context(context, files=True)
    _config(config, context)
    device, refs = z.device, context["source_refs"]
    _require(type(steps) is int and steps >= 0 and isinstance(operator, FrozenCPUCSR)
             and device.type in ("cpu", "cuda") and not torch.is_autocast_enabled(device.type)
             and torch.get_default_dtype() == torch.float32, "Graph factor native invocation differs")
    _matrix(z, (dims["nodes"], dims["dimension"]), dtype=torch.float64, device=device)
    _matrix(q, (dims["nodes"], dims["classes"]), dtype=torch.float64, device=device)
    _require(bool((q >= 0).all()) and bool((q.sum(0) > 0).all())
             and bool((q.sum(1) > 0).all())
             and torch.is_tensor(assignment) and assignment.layout == torch.strided
             and assignment.dtype == torch.int64 and tuple(assignment.shape) == (dims["nodes"],)
             and assignment.device == device and not assignment.requires_grad
             and bool(((assignment >= 0) & (assignment < dims["cells"])).all()),
             "Graph factor original teacher/assignment domain differs")
    digest = array_digest(z.detach().cpu().numpy(), q.detach().cpu().numpy(), assignment.cpu().numpy())
    _require(refs["data_digest"] == config.get("data_digest") == digest
             and type(config.get("factor_seed")) is int and type(config.get("assignment_rank")) is int
             and type(config.get("chunk_size")) is int
             and refs["factor_seed"] == config.get("factor_seed")
             and refs["rank"] == config.get("assignment_rank")
             and refs["chunk_size"] == config.get("chunk_size")
             and refs["device"] == str(device)
             and _factors(initial_parameters, dims, device=device, detached=False) == context["native_parameter_digests"]
             and bool(initial_parameters[0].eq(0).all())
             and _seal(operator.descriptor()) == _seal(context["operator"]),
             "Graph factor loaded native origin/operator/data/config binding differs")
    _require(operator.counts() == dict(S_forward_products=0, T_transpose_products=0),
             "Graph factor operator must not run products before source/native validation")
    if resume_state is not None:
        validate_resume(resume_state, context, config, steps)
        _COUNT_OFFSETS[operator] = dict(resume_state["graph_factor_operator_counts"])
        _ACCEPTED_BEST[operator] = dict(record=_json(resume_state["graph_factor_best_record"]),
                                      frontier=resume_state["step"],
                                      history_digest=resume_state["graph_factor_history_digest"])
        accepted = _json(resume_state["graph_factor_accepted_frontiers"])
        frontier = resume_state["step"]
        if str(frontier) not in accepted:
            old_row = resume_state["history"][frontier]
            old_snapshot = resume_state["snapshots"][frontier]
            _require(old_row["J"] == old_snapshot["teacher_ce"],
                     "Graph factor accepted new frontier lacks an exact old row/checkpoint link")
            accepted[str(frontier)] = dict(step=frontier,
                snapshot_record=_json(resume_state["graph_factor_current_record"]),
                original_history_J=old_row["J"],
                original_history_row_digest=_seal(old_row, history=True),
                original_history_digest=resume_state["graph_factor_history_digest"],
                original_state_digest=resume_state["graph_factor_state_digest"])
        _ACCEPTED_FRONTIERS[operator] = accepted
    else:
        _COUNT_OFFSETS[operator] = dict(S_forward_products=0, T_transpose_products=0)
        _ACCEPTED_BEST[operator] = None
        _ACCEPTED_FRONTIERS[operator] = {}
        if folder is not None:
            folder = Path(folder)
            _require(not (folder / "resume.pt").exists()
                     and not any((folder / "checkpoints").glob("step_*.pt")),
                     "Graph factor cached prefix needs its verifiable resume")
    return cpu_state(initial_parameters)


def _pullback(value, step, dims):
    if step == 0:
        _require(value is None, "Graph factor P0 cannot carry an earlier pullback")
        return None
    _require(isinstance(value, dict) and set(value) == {"step", "gW", "gU", "gV"}
             and type(value["step"]) is int and value["step"] == step - 1,
             "Graph factor pullback must be the actual immediately preceding update")
    for key, shape in (("gW", (dims["nodes"], dims["rank"])),
                       ("gU", (dims["nodes"], dims["rank"])),
                       ("gV", (dims["cells"], dims["rank"]))):
        _matrix(value[key], shape, dtype=torch.float32, device=torch.device("cpu"))
    return cpu_state(value)


def _pullback_descriptor(value):
    return None if value is None else dict(step=value["step"],
        **{key: _tensor_identity(value[key]) for key in ("gW", "gU", "gV")})


def _snapshot_record(snapshot, context, config, dims):
    return dict(step=snapshot["step"], context_digest=_seal(context), config_digest=_seal(config),
                parameter_digests=_factors(snapshot["graph_factor_parameters"], dims,
                                           device=torch.device("cpu")),
                effective_u=_tensor_identity(snapshot["graph_factor_effective_u"]),
                moments=_tensor_identity(snapshot["moments"]), theta=_tensor_identity(snapshot["theta"]),
                teacher_ce=snapshot["teacher_ce"], inner_grad_max=snapshot["inner_grad_max"],
                J_exact=snapshot["J_exact"], scale=snapshot["graph_factor_scale"],
                normalized_objective=snapshot["normalized_objective"],
                operator_counts=snapshot["graph_factor_operator_counts"],
                last_pullback=_pullback_descriptor(snapshot["graph_factor_last_pullback"]))


def attach_snapshot(snapshot, context, config, parameters, effective_u, operator, scale, pullback=None):
    dims = _context(context)
    _config(config, context)
    _require(isinstance(snapshot, dict) and type(snapshot.get("step")) is int and snapshot["step"] >= 0
             and snapshot.get("J_exact") is True, "Graph factor requires an exact actual checkpoint")
    step = snapshot["step"]
    native = cpu_state(parameters)
    effective = cpu_state(effective_u)
    _factors(native, dims, device=torch.device("cpu"))
    _matrix(effective, (dims["nodes"], dims["rank"]), dtype=torch.float32, device=torch.device("cpu"))
    _moments(snapshot.get("moments"), dims); _theta(snapshot.get("theta"), dims)
    ce, grad = _number(snapshot.get("teacher_ce"), positive=True), _number(snapshot.get("inner_grad_max"))
    _require(_number(scale, positive=True) > 1e-12 and scale == context["native_origin"]["teacher_ce"]
             and grad >= 0 and _seal(operator.descriptor()) == _seal(context["operator"]),
             "Graph factor checkpoint scale/operator/stationarity declaration differs")
    if step == 0:
        _require(_origin(snapshot, dims) == context["native_origin"]
                 and _factors(native, dims) == context["native_parameter_digests"]
                 and bool(native[0].eq(0).all()) and bool(effective.eq(0).all()),
                 "Graph factor changed native P0 moments/head/CE/factors before its first update")
    snapshot.update(graph_factor_context=_json(context), graph_factor_config=_json(config),
                    graph_factor_parameters=native, graph_factor_effective_u=effective,
                    graph_factor_operator_counts=_counts(operator), graph_factor_scale=scale,
                    graph_factor_last_pullback=_pullback(pullback, step, dims),
                    normalized_objective=ce / scale, objective_name=OBJECTIVE)
    snapshot["graph_factor_record_digest"] = _seal(_snapshot_record(snapshot, context, config, dims))
    return snapshot


def _warm(value, dims):
    if value is None:
        return None
    _theta(value, dims)
    return _tensor_identity(value)


def _best_record(state, context, config, dims):
    return dict(step=state["best_step"], teacher_ce=state["best"],
                context_digest=_seal(context), config_digest=_seal(config),
                parameters=_factors(state["graph_factor_best_parameters"], dims, device=torch.device("cpu")),
                moments=_tensor_identity(state["best_moments"]), theta=_tensor_identity(state["best_theta"]))


def _optimizer_descriptor(saved):
    return dict(param_groups=_json(saved["param_groups"]), slots={str(key):
        {name: _tensor_identity(value) for name, value in slot.items()}
        for key, slot in saved["state"].items()})


def attach_resume(state, context, initial_parameters, best_parameters, operator, pullback=None):
    dims = _context(context)
    _require(isinstance(state, dict) and isinstance(state.get("config"), dict),
             "Graph factor resume envelope is missing")
    _config(state["config"], context)
    state.update(graph_factor_context=_json(context), graph_factor_initial_parameters=cpu_state(initial_parameters),
                 graph_factor_best_parameters=cpu_state(best_parameters),
                 graph_factor_operator_counts=_counts(operator),
                 graph_factor_last_pullback=_pullback(pullback, state["step"], dims),
                 graph_factor_policy=_json(POLICY))
    state["graph_factor_current_record"] = _snapshot_record(state["snapshots"][state["step"]],
                                                            context, state["config"], dims)
    state["graph_factor_best_record"] = _best_record(state, context, state["config"], dims)
    accepted = _ACCEPTED_BEST.get(operator)
    state["graph_factor_best_provenance"] = (_json(accepted) if accepted is not None
        and accepted["record"] == state["graph_factor_best_record"] else None)
    state["graph_factor_accepted_frontiers"] = {
        key: _json(value) for key, value in _ACCEPTED_FRONTIERS.get(operator, {}).items()
        if int(key) in state["snapshots"] and value["snapshot_record"] ==
        _snapshot_record(state["snapshots"][int(key)], context, state["config"], dims)}
    state["graph_factor_warm_before"] = {
        key: _warm(state.get(key), dims) for key in ("tracking_theta_before", "tracking_vector_before")}
    state["graph_factor_optimizer_descriptor"] = _optimizer_descriptor(state["optimizer"])
    state["graph_factor_history_digest"] = _seal(state["history"], history=True)
    state["graph_factor_state_digest"] = _seal(dict(
        current=state["graph_factor_current_record"], best=state["graph_factor_best_record"],
        best_provenance=state["graph_factor_best_provenance"],
        accepted_frontiers=state["graph_factor_accepted_frontiers"],
        initial=context["native_origin"], warm_before=state["graph_factor_warm_before"],
        optimizer=state["graph_factor_optimizer_descriptor"],
        history=state["graph_factor_history_digest"], counts=state["graph_factor_operator_counts"],
        last_pullback=_pullback_descriptor(state["graph_factor_last_pullback"])))
    validate_resume(state, context, state["config"], state["step"])
    return state


def _validate_optimizer(saved, config, parameters, step):
    _require(isinstance(saved, dict) and set(saved) == {"state", "param_groups"}
             and isinstance(saved["param_groups"], list) and len(saved["param_groups"]) == 1
             and isinstance(saved["state"], dict), "Graph factor original Adam envelope differs")
    group = saved["param_groups"][0]
    _require(isinstance(group, dict) and group.get("params") == [0, 1]
             and group.get("lr") == config["lr"] and group.get("eps") == 1e-12
             and tuple(group.get("betas", ())) == (.9, .999)
             and group.get("weight_decay") == 0 and group.get("foreach") is False
             and group.get("amsgrad") is False and group.get("maximize") is False
             and group.get("capturable") is False and group.get("differentiable") is False
             and group.get("fused") is None
             and ("decoupled_weight_decay" not in group or group["decoupled_weight_decay"] is False),
             "Graph factor resume changes the original Adam recurrence")
    _require(set(group) <= {"params", "lr", "eps", "betas", "weight_decay", "foreach", "amsgrad",
                            "maximize", "capturable", "differentiable", "fused", "decoupled_weight_decay"},
             "Graph factor Adam has unsupported recurrence fields")
    if step == 0:
        _require(not saved["state"], "Graph factor native P0 Adam slots must be empty")
        return
    _require(set(saved["state"]) == {0, 1}, "Graph factor Adam U/V slots are incomplete")
    for index, parameter in enumerate(parameters):
        slot = saved["state"][index]
        _require(isinstance(slot, dict) and set(slot) == {"step", "exp_avg", "exp_avg_sq"},
                 "Graph factor Adam slot schema differs")
        tick = slot["step"]
        _require(torch.is_tensor(tick) and tick.ndim == 0 and tick.device.type == "cpu"
                 and tick.dtype == torch.float32 and not tick.requires_grad and bool(torch.isfinite(tick))
                 and float(tick) == step, "Graph factor Adam completed-step counter differs")
        for key in ("exp_avg", "exp_avg_sq"):
            _matrix(slot[key], parameter.shape, dtype=torch.float32, device=torch.device("cpu"))
        _require(bool((slot["exp_avg_sq"] >= 0).all()), "Graph factor Adam second moment is negative")


def validate_resume(state, context, config, steps):
    """Check saved native/current/best/warm lineage; no endpoint or head replay."""
    dims = _context(context)
    _config(config, context)
    _require(isinstance(state, dict) and type(state.get("step")) is int
             and type(steps) is int and 0 <= state["step"] <= steps
             and _seal(state.get("config")) == _seal(config)
             and _seal(state.get("graph_factor_context")) == _seal(context)
             and _seal(state.get("graph_factor_policy")) == _seal(POLICY),
             "Graph factor resume lost source/config/policy/context")
    end = state["step"]
    origin_digests = _factors(state.get("graph_factor_initial_parameters"), dims, device=torch.device("cpu"))
    parameters = state.get("parameters")
    parameter_digests = _factors(parameters, dims, device=torch.device("cpu"))
    _require(origin_digests == context["native_parameter_digests"]
             and bool(state["graph_factor_initial_parameters"][0].eq(0).all())
             and state.get("dual") is None and state.get("best_dual") is None
             and state.get("best_parameters") is None,
             "Graph factor native origin or disabled assignment exporter differs")
    _validate_optimizer(state.get("optimizer"), config, parameters, end)
    optimizer_descriptor = _optimizer_descriptor(state["optimizer"])
    _require(state.get("graph_factor_optimizer_descriptor") == optimizer_descriptor,
             "Graph factor original Adam slot bytes or group descriptor changed")
    scale = _number(state.get("scale"), positive=True)
    _require(scale > 1e-12 and scale == context["native_origin"]["teacher_ce"],
             "Graph factor must retain original positive CE0 without floor activation")
    snapshots = state.get("snapshots")
    _require(isinstance(snapshots, dict) and 0 in snapshots and end in snapshots,
             "Graph factor resume needs original and frontier checkpoints")
    for step, snapshot in snapshots.items():
        _require(type(step) is int and 0 <= step <= end and isinstance(snapshot, dict)
                 and type(snapshot.get("step")) is int and snapshot["step"] == step
                 and snapshot.get("J_exact") is True
                 and _seal(snapshot.get("graph_factor_context")) == _seal(context)
                 and _seal(snapshot.get("graph_factor_config")) == _seal(config)
                 and snapshot.get("graph_factor_scale") == scale and snapshot.get("objective_name") == OBJECTIVE,
                 "Graph factor checkpoint context/config/scale differs")
        _moments(snapshot.get("moments"), dims); _theta(snapshot.get("theta"), dims)
        _factors(snapshot.get("graph_factor_parameters"), dims, device=torch.device("cpu"))
        _matrix(snapshot.get("graph_factor_effective_u"), (dims["nodes"], dims["rank"]),
                dtype=torch.float32, device=torch.device("cpu"))
        ce = _number(snapshot.get("teacher_ce"), positive=True)
        _require(_number(snapshot.get("inner_grad_max")) >= 0
                 and snapshot.get("normalized_objective") == ce / scale,
                 "Graph factor checkpoint changed its original CE-only objective")
        _pullback(snapshot.get("graph_factor_last_pullback"), step, dims)
        _require(snapshot.get("graph_factor_record_digest") == _seal(_snapshot_record(snapshot, context, config, dims)),
                 "Graph factor checkpoint actual tensor/scalar record changed")
    zero, terminal = snapshots[0], snapshots[end]
    _require(_origin(zero, dims) == context["native_origin"]
             and _factors(zero["graph_factor_parameters"], dims) == context["native_parameter_digests"]
             and bool(zero["graph_factor_effective_u"].eq(0).all())
             and _tensor_identity(_moments(state.get("initial_moments"), dims)) == context["native_origin"]["moments"]
             and _tensor_identity(_theta(state.get("theta"), dims)) == _tensor_identity(terminal["theta"])
             and parameter_digests == _factors(terminal["graph_factor_parameters"], dims),
             "Graph factor P0 or current original M/head/base-factor links differ")
    counts = state.get("graph_factor_operator_counts")
    _require(isinstance(counts, dict) and set(counts) == {"S_forward_products", "T_transpose_products"}
             and all(type(value) is int and value >= 0 for value in counts.values())
             and counts == terminal["graph_factor_operator_counts"]
             and counts["S_forward_products"] >= end + 1 and counts["T_transpose_products"] >= end,
             "Graph factor actual cumulative operator work is incomplete")
    for snapshot in snapshots.values():
        old_counts = snapshot["graph_factor_operator_counts"]
        _require(isinstance(old_counts, dict) and set(old_counts) == set(counts)
                 and all(type(v) is int and 0 <= v <= counts[k] for k, v in old_counts.items()),
                 "Graph factor checkpoint cumulative product counts differ")
    previous_counts = dict(S_forward_products=0, T_transpose_products=0)
    for step in sorted(snapshots):
        snapshot_counts = snapshots[step]["graph_factor_operator_counts"]
        _require(all(snapshot_counts[key] >= previous_counts[key] for key in previous_counts),
                 "Graph factor saved product work is not a monotone prefix")
        previous_counts = snapshot_counts
    last = _pullback(state.get("graph_factor_last_pullback"), end, dims)
    _require(_pullback_descriptor(last) == _pullback_descriptor(terminal["graph_factor_last_pullback"]),
             "Graph factor actual last pullback lost its terminal link")
    warm = {key: _warm(state.get(key), dims) for key in ("tracking_theta_before", "tracking_vector_before")}
    _require(warm == state.get("graph_factor_warm_before")
             and all((value is None) == (end == 0) for value in warm.values()),
             "Graph factor original actual warm-before provenance differs")
    history = state.get("history")
    _require(isinstance(history, list) and all(isinstance(row, dict) for row in history)
             and [row.get("step") for row in history] == list(range(end + 1))
             and state.get("graph_factor_history_digest") == _seal(history, history=True),
             "Graph factor complete typed scalar history changed")
    accepted_frontiers = state.get("graph_factor_accepted_frontiers")
    _require(isinstance(accepted_frontiers, dict), "Graph factor accepted frontier provenance is missing")
    for key, accepted in accepted_frontiers.items():
        _require(isinstance(key, str) and key.isdecimal() and str(int(key)) == key
                 and isinstance(accepted, dict) and set(accepted) ==
                 {"step", "snapshot_record", "original_history_J", "original_history_row_digest",
                  "original_history_digest", "original_state_digest"}
                 and type(accepted["step"]) is int and accepted["step"] == int(key)
                 and int(key) in snapshots and 0 <= int(key) <= end
                 and accepted["snapshot_record"] == _snapshot_record(snapshots[int(key)], context, config, dims)
                 and accepted["original_history_J"] == snapshots[int(key)]["teacher_ce"]
                 and all(_hash(accepted[name]) for name in
                         ("original_history_row_digest", "original_history_digest", "original_state_digest")),
                 "Graph factor accepted frontier lost its exact old snapshot/history/state links")
    for row in history:
        step, ce = row["step"], _number(row.get("J"), positive=True)
        _require(type(step) is int and row.get("J_exact") is True and row.get("inner_converged") is True
                 and row.get("solver_mode") == "exact" and row.get("exact_refresh") is True
                 and row.get("head_fallback") is False and row.get("implicit_fallback") is False
                 and _number(row.get("inner_grad_max")) >= 0
                 and row.get("status") in ("update", "evaluated")
                 and (step == end or (row["status"] == "update" and row.get("cg_converged") is True))
                 and (row["status"] != "update" or row.get("cg_converged") is True),
                 "Graph factor history lost exact original head/adjoint completion")
        _number(row.get("best_J"), positive=True)
        if step in snapshots:
            # Inclusive continuation may re-evaluate its accepted frontier. Its
            # old checkpoint remains exact and externally pinned; only that
            # validated frontier can link a new row to a different old CE.
            _require(ce == snapshots[step]["teacher_ce"] or str(step) in accepted_frontiers,
                     "Graph factor non-frontier checkpoint/history CE differs")
    _require(history[0]["J"] == scale and _number(state.get("best"), positive=True) <= min(row["J"] for row in history)
             and type(state.get("best_step")) is int and 0 <= state["best_step"] <= end,
             "Graph factor original strict best CE selection differs")
    best_step, best = state["best_step"], state["best"]
    _moments(state.get("best_moments"), dims); _theta(state.get("best_theta"), dims)
    _factors(state.get("graph_factor_best_parameters"), dims, device=torch.device("cpu"))
    provenance = state.get("graph_factor_best_provenance")
    if provenance is not None:
        _require(isinstance(provenance, dict) and set(provenance) == {"record", "frontier", "history_digest"}
                 and type(provenance["frontier"]) is int and best_step <= provenance["frontier"] <= end
                 and _hash(provenance["history_digest"])
                 and provenance["record"] == _best_record(state, context, config, dims),
                 "Graph factor retained accepted-frontier best provenance differs")
    _require(all(row["J"] > best for row in history[:best_step])
             and (best == history[best_step]["J"] or provenance is not None),
             "Graph factor retained best is not the original first strict improvement")
    if best_step in snapshots and snapshots[best_step]["teacher_ce"] == best:
        selected = snapshots[best_step]
        _require(_tensor_identity(state["best_moments"]) == _tensor_identity(selected["moments"])
                 and _tensor_identity(state["best_theta"]) == _tensor_identity(selected["theta"])
                 and _factors(state["graph_factor_best_parameters"], dims) == _factors(selected["graph_factor_parameters"], dims),
                 "Graph factor best M/head/base factors lost checkpoint linkage")
    current_record = _snapshot_record(terminal, context, config, dims)
    best_record = _best_record(state, context, config, dims)
    _require(state.get("graph_factor_current_record") == current_record
             and state.get("graph_factor_best_record") == best_record,
             "Graph factor bounded current/best records changed")
    expected = _seal(dict(current=current_record, best=best_record, best_provenance=provenance,
                         accepted_frontiers=accepted_frontiers,
                         initial=context["native_origin"],
                         warm_before=warm, optimizer=optimizer_descriptor,
                         history=state["graph_factor_history_digest"], counts=counts,
                         last_pullback=_pullback_descriptor(last)))
    _require(state.get("graph_factor_state_digest") == expected,
             "Graph factor original core state lineage seal changed")
    return state
