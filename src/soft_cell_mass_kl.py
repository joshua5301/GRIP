"""Opt-in source-cell occupancy KL to uniform on original moments."""
import hashlib
import json
import math
from pathlib import Path

import torch

from src.io import array_digest, cpu_state
from src.shared_features import _tensor_identity

SCHEMA = 1
MODE = "normalized_soft_cell_mass_KL_v1"
OBJECTIVE = "teacher_CE_over_CE0_plus_soft_cell_mass_KL_over_Omega0"
PROBABILITY_ATOL = 1e-12
POLICY = dict(schema=SCHEMA, mode=MODE, objective=OBJECTIVE, coefficient=1,
              cell_mass="original_source_node_count_marginals", reference="uniform_over_cells",
              mass_mode="free", teacher="original_frozen_soft_Q", head_weighting="uniform",
              scale_floor=False, probability_floor=False, mass_renormalization=False,
              feature_cotangent="zero", target_cotangent="zero",
              mass_cotangent="log(K*m)+1", original_base_moment_operator=True,
              single_original_P_backward=True)


def _seal(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _number(value, *, positive=False):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or (positive and value <= 0)):
        raise ValueError("Soft cell mass KL scalar must be finite and scales strictly positive")
    return value


def _matrix(value, shape, *, dtype=torch.float64, device=None, detached=True):
    if (not torch.is_tensor(value) or value.layout != torch.strided or value.ndim != 2
            or min(value.shape) < 1 or tuple(value.shape) != tuple(shape) or value.dtype != dtype
            or (device is not None and value.device != device)
            or (detached and value.requires_grad) or not bool(torch.isfinite(value).all())):
        raise ValueError("Soft cell mass KL tensor shape/dtype/device/finiteness differs")
    return value


def _factor_digests(values, nodes, cells, rank, *, detached=True):
    if not isinstance(values, (list, tuple)) or len(values) != 2:
        raise ValueError("Soft cell mass KL needs exactly native U/V")
    for value, shape in zip(values, ((nodes, rank), (cells, rank)), strict=True):
        _matrix(value, shape, dtype=torch.float32, detached=detached)
    return [array_digest(value.detach().cpu().numpy()) for value in values]


def _files(pins):
    if not isinstance(pins, dict) or not pins:
        raise ValueError("Soft cell mass KL immutable source/input file pins are missing")
    for name, expected in pins.items():
        if (not isinstance(name, str) or not isinstance(expected, str) or len(expected) != 64
                or any(c not in "0123456789abcdef" for c in expected)):
            raise ValueError("Soft cell mass KL immutable file pin is malformed")
        with Path(name).open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise ValueError("Soft cell mass KL immutable source/input bytes changed")


def _decoded(moments, dimension, classes):
    if (type(dimension) is not int or dimension < 1 or type(classes) is not int or classes < 2
            or not torch.is_tensor(moments) or moments.ndim != 2 or len(moments) < 2
            or moments.device.type not in ("cpu", "cuda")
            or torch.is_autocast_enabled(moments.device.type)):
        raise ValueError("Soft cell mass KL requires the original typed FP64 CPU/CUDA base moments")
    _matrix(moments, (len(moments), 1 + dimension + classes))
    mass, totals = moments[:, :1], moments[:, 1 + dimension:]
    if not bool((mass > 0).all()) or bool((totals < 0).any()):
        raise FloatingPointError("Soft cell mass KL needs positive masses and unchanged nonnegative teacher totals")
    probabilities = totals / mass
    if (not bool(torch.isfinite(probabilities).all()) or bool((probabilities < 0).any())
            or float((probabilities.sum(1) - 1).abs().max()) > PROBABILITY_ATOL
            or abs(float(mass.sum()) - 1) > PROBABILITY_ATOL):
        raise FloatingPointError("Soft cell mass KL moments are outside the unchanged probability domain")
    return mass, totals, probabilities


def mass_kl_partials(moments, dimension, classes):
    """FP64 ambient derivative on mass only; no extra material or 1/N here."""
    mass, _, _ = _decoded(moments, dimension, classes)
    log_mass_ratio = (len(moments) * mass).log()
    value = (mass * log_mass_ratio).sum()
    gradient = torch.zeros_like(moments)
    # Keep +1 in independent M coordinates; only the original row-softmax
    # adjoint projects the shared constant away. Original backward supplies 1/N.
    gradient[:, :1] = log_mass_ratio + 1
    if (not bool(torch.isfinite(value)) or float(value) < 0
            or not bool(torch.isfinite(gradient).all())):
        raise FloatingPointError("Soft cell mass KL or its analytic moment cotangent is nonfinite/negative")
    return dict(value=value.detach(), moment_gradient=gradient.detach())


def validate_context(z, q, assignment, parameters, mixing, chunk_size, context, config,
                     steps, resume_state=None, folder=None):
    """Validate native origin once, before updated resume parameters are loaded."""
    if (not isinstance(context, dict)
            or set(context) != {"schema", "policy", "source_refs", "native_parameter_digests"}
            or type(context["schema"]) is not int or context["schema"] != SCHEMA
            or _seal(context["policy"]) != _seal(POLICY)
            or type(steps) is not int or steps < 0 or type(chunk_size) is not int or chunk_size < 1
            or mixing != .05 or not isinstance(config, dict)
            or config.get("soft_cell_mass_KL_mode") != MODE
            or _seal(config.get("soft_cell_mass_KL_context")) != _seal(context)
            or not isinstance(parameters, (list, tuple)) or len(parameters) != 2
            or any(not torch.is_tensor(v) or v.ndim != 2 or min(v.shape) < 1
                   for v in (z, q, *parameters))
            or z.device.type not in ("cpu", "cuda")
            or torch.is_autocast_enabled(z.device.type)):
        raise ValueError("Soft cell mass KL source/context/schema/control differs")
    nodes, cells, rank = len(z), len(parameters[1]), parameters[0].shape[1]
    dimension, classes = z.shape[1], q.shape[1]
    _matrix(z, (nodes, dimension), device=z.device)
    _matrix(q, (nodes, classes), device=z.device)
    if (classes < 2 or cells < 2 or rank < 1 or rank > min(nodes, cells)
            or bool((q < 0).any()) or not bool((q.sum(0) > 0).all())
            or float((q.sum(1) - 1).abs().max()) > PROBABILITY_ATOL
            or not torch.is_tensor(assignment) or assignment.dtype != torch.int64
            or assignment.shape != (nodes,) or assignment.device != z.device or assignment.requires_grad
            or not bool(((assignment >= 0) & (assignment < cells)).all())):
        raise ValueError("Soft cell mass KL original soft teacher/hard cell order differs")
    refs = context["source_refs"]
    data = array_digest(z.detach().cpu().numpy(), q.detach().cpu().numpy(), assignment.cpu().numpy())
    if (not isinstance(refs, dict) or refs.get("data_digest") != data or config.get("data_digest") != data
            or type(refs.get("factor_seed")) is not int or type(config.get("factor_seed")) is not int
            or refs["factor_seed"] != config["factor_seed"]
            or type(config.get("assignment_rank")) is not int or config["assignment_rank"] != rank
            or any(type(refs.get(key)) is not int or refs[key] != value for key, value in
                   dict(nodes=nodes, cells=cells, rank=rank, physical_dimensions=dimension, classes=classes).items())
            or type(refs.get("chunk_size")) is not int or type(config.get("chunk_size")) is not int
            or refs["chunk_size"] != chunk_size or config["chunk_size"] != chunk_size
            or refs.get("mixing") != mixing or config.get("mixing") != mixing
            or refs.get("device") != str(z.device)):
        raise ValueError("Soft cell mass KL native source dimensions/data/seed/config binding differs")
    _files(refs.get("files_sha256"))
    source = refs.get("current_source")
    if (not isinstance(source, dict) or not isinstance(source.get("git_head"), str)
            or not source["git_head"] or not isinstance(source.get("source_digest"), str)
            or not source["source_digest"]):
        raise ValueError("Soft cell mass KL complete current source binding is missing")
    _files(source.get("files"))
    native = _factor_digests(parameters, nodes, cells, rank, detached=False)
    if (native != context["native_parameter_digests"] or bool(parameters[0].ne(0).any())
            or any(p.device != z.device for p in parameters)):
        raise ValueError("Soft cell mass KL original zeroU/GaussianV native origin differs")
    initial = cpu_state(parameters)
    if resume_state is not None:
        validate_resume(resume_state, context, config, steps, nodes, cells, rank, dimension, classes)
    elif folder is not None and any((Path(folder) / "checkpoints").glob("step_*.pt")):
        raise ValueError("Soft cell mass KL checkpoints require a verifiable activated resume")
    return initial


def attach_snapshot(snapshot, context, config, parameters, CE0, Omega0, Omega, objective):
    _number(CE0, positive=True); _number(Omega0, positive=True)
    _number(Omega); _number(snapshot.get("teacher_ce")); _number(objective)
    if Omega < 0 or objective != snapshot["teacher_ce"] / CE0 + Omega / Omega0:
        raise ValueError("Soft cell mass KL checkpoint lost its normalized CE/Omega objective")
    snapshot.update(soft_cell_mass_KL_context=context, soft_cell_mass_KL_config=config,
                    soft_cell_mass_KL_parameters=cpu_state(parameters),
                    soft_cell_mass_KL_CE0=CE0, soft_cell_mass_KL_Omega0=Omega0,
                    soft_cell_mass_KL_Omega=Omega, objective=objective, objective_name=OBJECTIVE)
    return snapshot


def attach_resume(state, context, initial_parameters, CE0, Omega0):
    _number(CE0, positive=True); _number(Omega0, positive=True)
    state.update(soft_cell_mass_KL_context=context,
                 soft_cell_mass_KL_initial_parameters=cpu_state(initial_parameters),
                 soft_cell_mass_KL_CE0=CE0, soft_cell_mass_KL_Omega0=Omega0)
    return state


def validate_resume(saved, context, config, steps, nodes, cells, rank, dimension, classes):
    if (not isinstance(saved, dict) or type(saved.get("step")) is not int or not 0 <= saved["step"] <= steps
            or _seal(saved.get("config")) != _seal(config)
            or _seal(saved.get("soft_cell_mass_KL_context")) != _seal(context)
            or _factor_digests(saved.get("soft_cell_mass_KL_initial_parameters"), nodes, cells, rank)
               != context["native_parameter_digests"]):
        raise ValueError("Soft cell mass KL resume lost its native origin/config/context")
    CE0 = _number(saved.get("soft_cell_mass_KL_CE0"), positive=True)
    Omega0 = _number(saved.get("soft_cell_mass_KL_Omega0"), positive=True)
    snapshots, end = saved.get("snapshots"), saved["step"]
    if not isinstance(snapshots, dict) or 0 not in snapshots or end not in snapshots:
        raise ValueError("Soft cell mass KL resume requires initial and terminal snapshots")
    for step, snapshot in snapshots.items():
        if (type(step) is not int or not 0 <= step <= end or type(snapshot.get("step")) is not int
                or snapshot["step"] != step or snapshot.get("J_exact") is not True
                or _seal(snapshot.get("soft_cell_mass_KL_context")) != _seal(context)
                or _seal(snapshot.get("soft_cell_mass_KL_config")) != _seal(config)
                or snapshot.get("soft_cell_mass_KL_CE0") != CE0
                or snapshot.get("soft_cell_mass_KL_Omega0") != Omega0 or snapshot.get("objective_name") != OBJECTIVE):
            raise ValueError("Soft cell mass KL checkpoint binding differs")
        _factor_digests(snapshot.get("soft_cell_mass_KL_parameters"), nodes, cells, rank)
        _matrix(snapshot.get("theta"), (classes, dimension + 1))
        _decoded(snapshot.get("moments"), dimension, classes)
        Omega, CE = _number(snapshot.get("soft_cell_mass_KL_Omega")), _number(snapshot.get("teacher_ce"))
        if Omega < 0 or _number(snapshot.get("objective")) != CE / CE0 + Omega / Omega0:
            raise ValueError("Soft cell mass KL checkpoint scalar objective differs")
    zero, terminal = snapshots[0], snapshots[end]
    if (_factor_digests(zero["soft_cell_mass_KL_parameters"], nodes, cells, rank)
            != context["native_parameter_digests"] or zero["teacher_ce"] != CE0
            or zero["soft_cell_mass_KL_Omega"] != Omega0 or zero["objective"] != 2.0
            or saved.get("scale") != CE0
            or _tensor_identity(saved["initial_moments"]) != _tensor_identity(zero["moments"])
            or _tensor_identity(saved["theta"]) != _tensor_identity(terminal["theta"])
            or _factor_digests(saved["parameters"], nodes, cells, rank)
               != _factor_digests(terminal["soft_cell_mass_KL_parameters"], nodes, cells, rank)):
        raise ValueError("Soft cell mass KL initial/terminal moment/head/parameter links differ")
    history = saved.get("history")
    if (not isinstance(history, list) or any(not isinstance(row, dict) for row in history)
            or [row.get("step") for row in history] != list(range(end + 1))):
        raise ValueError("Soft cell mass KL history must retain the complete typed prefix")
    for row in history:
        CE, Omega = _number(row.get("teacher_ce")), _number(row.get("soft_cell_mass_KL_Omega"))
        if (type(row.get("step")) is not int or row.get("J") != CE or Omega < 0
                or row.get("J_exact") is not True or row.get("inner_converged") is not True
                or row.get("soft_cell_mass_KL_CE0") != CE0
                or row.get("soft_cell_mass_KL_Omega0") != Omega0
                or _number(row.get("objective")) != CE / CE0 + Omega / Omega0
                or row.get("normalized_objective") != row["objective"] or row.get("objective_name") != OBJECTIVE):
            raise ValueError("Soft cell mass KL history CE/Omega/scale/objective differs")
    for step, snapshot in snapshots.items():
        row = history[step]
        if any(row[key] != snapshot[key] for key in ("teacher_ce", "soft_cell_mass_KL_Omega", "objective")):
            raise ValueError("Soft cell mass KL checkpoint/history scalar links differ")
    best = min(history, key=lambda row: row["objective"])
    if (type(saved.get("best_step")) is not int or saved["best_step"] != best["step"]
            or _number(saved.get("best")) != best["objective"]):
        raise ValueError("Soft cell mass KL saved best must use the normalized joint objective")
    _matrix(saved.get("best_moments"), (cells, 1 + dimension + classes))
    _matrix(saved.get("best_theta"), (classes, dimension + 1))
    if best["step"] in snapshots:
        checkpoint = snapshots[best["step"]]
        if (_tensor_identity(saved["best_moments"]) != _tensor_identity(checkpoint["moments"])
                or _tensor_identity(saved["best_theta"]) != _tensor_identity(checkpoint["theta"])):
            raise ValueError("Soft cell mass KL best moment/head checkpoint link differs")
