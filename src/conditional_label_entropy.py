"""Opt-in source-mass conditional teacher-label entropy on original moments."""
import hashlib
import json
import math
from pathlib import Path

import torch

from src.io import array_digest, cpu_state
from src.shared_features import _tensor_identity

SCHEMA = 1
MODE = "normalized_conditional_label_entropy_v1"
OBJECTIVE = "teacher_CE_over_CE0_plus_conditional_entropy_over_E0"
PROBABILITY_ATOL = 1e-12
POLICY = dict(schema=SCHEMA, mode=MODE, objective=OBJECTIVE, coefficient=1,
              cell_weighting="original_source_cell_mass", teacher="original_frozen_soft_Q",
              head_weighting="uniform", scale_floor=False, probability_floor=False,
              feature_cotangent="zero", mass_quotient_derivative=True,
              original_base_moment_operator=True, single_original_P_backward=True)


def _seal(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _number(value, *, positive=False):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or (positive and value <= 0)):
        raise ValueError("Entropy scalar must be finite and scales strictly positive")
    return value


def _matrix(value, shape, *, dtype=torch.float64, device=None, detached=True):
    if (not torch.is_tensor(value) or value.layout != torch.strided or value.ndim != 2
            or min(value.shape) < 1 or tuple(value.shape) != tuple(shape) or value.dtype != dtype
            or (device is not None and value.device != device)
            or (detached and value.requires_grad) or not bool(torch.isfinite(value).all())):
        raise ValueError("Entropy tensor shape/dtype/device/finiteness differs")
    return value


def _factor_digests(values, nodes, cells, rank, *, detached=True):
    if not isinstance(values, (list, tuple)) or len(values) != 2:
        raise ValueError("Entropy needs exactly native U/V")
    for value, shape in zip(values, ((nodes, rank), (cells, rank)), strict=True):
        _matrix(value, shape, dtype=torch.float32, detached=detached)
    return [array_digest(value.detach().cpu().numpy()) for value in values]


def _files(pins):
    if not isinstance(pins, dict) or not pins:
        raise ValueError("Entropy immutable source/input file pins are missing")
    for name, expected in pins.items():
        if (not isinstance(name, str) or not isinstance(expected, str) or len(expected) != 64
                or any(c not in "0123456789abcdef" for c in expected)):
            raise ValueError("Entropy immutable file pin is malformed")
        with Path(name).open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise ValueError("Entropy immutable source/input bytes changed")


def _decoded(moments, dimension, classes):
    if (type(dimension) is not int or dimension < 1 or type(classes) is not int or classes < 2
            or not torch.is_tensor(moments) or moments.ndim != 2):
        raise ValueError("Entropy requires typed physical dimension and full class count")
    _matrix(moments, (len(moments), 1 + dimension + classes))
    mass, totals = moments[:, :1], moments[:, 1 + dimension:]
    if not bool((mass > 0).all()) or not bool((totals > 0).all()):
        raise FloatingPointError("Entropy mass and every teacher target total must be positive")
    probabilities = totals / mass
    if (not bool(torch.isfinite(probabilities).all()) or not bool((probabilities > 0).all())
            or float((probabilities.sum(1) - 1).abs().max()) > PROBABILITY_ATOL
            or abs(float(mass.sum()) - 1) > PROBABILITY_ATOL):
        raise FloatingPointError("Entropy moments are outside the unchanged probability domain")
    return mass, totals, probabilities


def entropy_partials(moments, dimension, classes):
    """Analytic FP64 partials; both label totals and cell-mass quotients vary."""
    mass, totals, probabilities = _decoded(moments, dimension, classes)
    log_probability = probabilities.log()
    value = -(totals * log_probability).sum()
    gradient = torch.zeros_like(moments)
    gradient[:, :1] = (totals / mass).sum(1, keepdim=True)
    gradient[:, 1 + dimension:] = -log_probability - 1
    if (not bool(torch.isfinite(value)) or float(value) < 0
            or not bool(torch.isfinite(gradient).all())):
        raise FloatingPointError("Conditional entropy or analytic moment cotangent is nonfinite")
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
            or config.get("conditional_label_entropy_mode") != MODE
            or _seal(config.get("conditional_label_entropy_context")) != _seal(context)
            or not isinstance(parameters, (list, tuple)) or len(parameters) != 2
            or any(not torch.is_tensor(v) or v.ndim != 2 or min(v.shape) < 1
                   for v in (z, q, *parameters))):
        raise ValueError("Entropy source/context/schema/control differs")
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
        raise ValueError("Entropy original soft teacher/hard cell order differs")
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
        raise ValueError("Entropy native source dimensions/data/seed/config binding differs")
    _files(refs.get("files_sha256"))
    source = refs.get("current_source")
    if (not isinstance(source, dict) or not isinstance(source.get("git_head"), str)
            or not source["git_head"] or not isinstance(source.get("source_digest"), str)
            or not source["source_digest"]):
        raise ValueError("Entropy complete current source binding is missing")
    _files(source.get("files"))
    native = _factor_digests(parameters, nodes, cells, rank, detached=False)
    if (native != context["native_parameter_digests"] or bool(parameters[0].ne(0).any())
            or any(p.device != z.device for p in parameters)):
        raise ValueError("Entropy original zeroU/GaussianV native origin differs")
    initial = cpu_state(parameters)
    if resume_state is not None:
        validate_resume(resume_state, context, config, steps, nodes, cells, rank, dimension, classes)
    elif folder is not None and any((Path(folder) / "checkpoints").glob("step_*.pt")):
        raise ValueError("Entropy checkpoints require a verifiable activated resume")
    return initial


def attach_snapshot(snapshot, context, config, parameters, CE0, E0, E, objective):
    _number(CE0, positive=True); _number(E0, positive=True)
    _number(E); _number(snapshot.get("teacher_ce")); _number(objective)
    if E < 0 or objective != snapshot["teacher_ce"] / CE0 + E / E0:
        raise ValueError("Entropy checkpoint lost its normalized CE/E objective")
    snapshot.update(conditional_label_entropy_context=context, conditional_label_entropy_config=config,
                    conditional_label_entropy_parameters=cpu_state(parameters),
                    conditional_label_entropy_CE0=CE0, conditional_label_entropy_E0=E0,
                    conditional_label_entropy_E=E, objective=objective, objective_name=OBJECTIVE)
    return snapshot


def attach_resume(state, context, initial_parameters, CE0, E0):
    _number(CE0, positive=True); _number(E0, positive=True)
    state.update(conditional_label_entropy_context=context,
                 conditional_label_entropy_initial_parameters=cpu_state(initial_parameters),
                 conditional_label_entropy_CE0=CE0, conditional_label_entropy_E0=E0)
    return state


def validate_resume(saved, context, config, steps, nodes, cells, rank, dimension, classes):
    if (not isinstance(saved, dict) or type(saved.get("step")) is not int or not 0 <= saved["step"] <= steps
            or _seal(saved.get("config")) != _seal(config)
            or _seal(saved.get("conditional_label_entropy_context")) != _seal(context)
            or _factor_digests(saved.get("conditional_label_entropy_initial_parameters"), nodes, cells, rank)
               != context["native_parameter_digests"]):
        raise ValueError("Entropy resume lost its native origin/config/context")
    CE0 = _number(saved.get("conditional_label_entropy_CE0"), positive=True)
    E0 = _number(saved.get("conditional_label_entropy_E0"), positive=True)
    snapshots, end = saved.get("snapshots"), saved["step"]
    if not isinstance(snapshots, dict) or 0 not in snapshots or end not in snapshots:
        raise ValueError("Entropy resume requires initial and terminal snapshots")
    for step, snapshot in snapshots.items():
        if (type(step) is not int or not 0 <= step <= end or type(snapshot.get("step")) is not int
                or snapshot["step"] != step or snapshot.get("J_exact") is not True
                or _seal(snapshot.get("conditional_label_entropy_context")) != _seal(context)
                or _seal(snapshot.get("conditional_label_entropy_config")) != _seal(config)
                or snapshot.get("conditional_label_entropy_CE0") != CE0
                or snapshot.get("conditional_label_entropy_E0") != E0 or snapshot.get("objective_name") != OBJECTIVE):
            raise ValueError("Entropy checkpoint binding differs")
        _factor_digests(snapshot.get("conditional_label_entropy_parameters"), nodes, cells, rank)
        _matrix(snapshot.get("theta"), (classes, dimension + 1))
        _decoded(snapshot.get("moments"), dimension, classes)
        E, CE = _number(snapshot.get("conditional_label_entropy_E")), _number(snapshot.get("teacher_ce"))
        if E < 0 or _number(snapshot.get("objective")) != CE / CE0 + E / E0:
            raise ValueError("Entropy checkpoint scalar objective differs")
    zero, terminal = snapshots[0], snapshots[end]
    if (_factor_digests(zero["conditional_label_entropy_parameters"], nodes, cells, rank)
            != context["native_parameter_digests"] or zero["teacher_ce"] != CE0
            or zero["conditional_label_entropy_E"] != E0 or zero["objective"] != 2.0
            or saved.get("scale") != CE0
            or _tensor_identity(saved["initial_moments"]) != _tensor_identity(zero["moments"])
            or _tensor_identity(saved["theta"]) != _tensor_identity(terminal["theta"])
            or _factor_digests(saved["parameters"], nodes, cells, rank)
               != _factor_digests(terminal["conditional_label_entropy_parameters"], nodes, cells, rank)):
        raise ValueError("Entropy initial/terminal moment/head/parameter links differ")
    history = saved.get("history")
    if (not isinstance(history, list) or any(not isinstance(row, dict) for row in history)
            or [row.get("step") for row in history] != list(range(end + 1))):
        raise ValueError("Entropy history must retain the complete typed prefix")
    for row in history:
        CE, E = _number(row.get("teacher_ce")), _number(row.get("conditional_label_entropy_E"))
        if (type(row.get("step")) is not int or row.get("J") != CE or E < 0
                or row.get("J_exact") is not True or row.get("inner_converged") is not True
                or row.get("conditional_label_entropy_CE0") != CE0
                or row.get("conditional_label_entropy_E0") != E0
                or _number(row.get("objective")) != CE / CE0 + E / E0
                or row.get("normalized_objective") != row["objective"] or row.get("objective_name") != OBJECTIVE):
            raise ValueError("Entropy history CE/E/scale/objective differs")
    for step, snapshot in snapshots.items():
        row = history[step]
        if any(row[key] != snapshot[key] for key in ("teacher_ce", "conditional_label_entropy_E", "objective")):
            raise ValueError("Entropy checkpoint/history scalar links differ")
    best = min(history, key=lambda row: row["objective"])
    if (type(saved.get("best_step")) is not int or saved["best_step"] != best["step"]
            or _number(saved.get("best")) != best["objective"]):
        raise ValueError("Entropy saved best must use the normalized joint objective")
    _matrix(saved.get("best_moments"), (cells, 1 + dimension + classes))
    _matrix(saved.get("best_theta"), (classes, dimension + 1))
    if best["step"] in snapshots:
        checkpoint = snapshots[best["step"]]
        if (_tensor_identity(saved["best_moments"]) != _tensor_identity(checkpoint["moments"])
                or _tensor_identity(saved["best_theta"]) != _tensor_identity(checkpoint["theta"])):
            raise ValueError("Entropy best moment/head checkpoint link differs")
