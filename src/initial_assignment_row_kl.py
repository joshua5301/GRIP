"""Source-row KL to the immutable initial assignment, fused with P moments.

Only the activated NODE route uses this operator. The head objective is unchanged;
both CE and row-KL directions use the original frozen teacher-CE0 scale.
"""
import hashlib
import inspect
import json
import math
from numbers import Real
from pathlib import Path

import torch

from src.io import array_digest, cpu_state
from src.low_rank_assignment import initialize_factors, logit_block
from src.moments import initial_logits

SCHEMA = 1
KEYS = ("assignment_kl_weight", "assignment_kl_schema", "assignment_kl_source_digest")
POLICY = dict(schema=SCHEMA, objective="teacher_CE_plus_source_row_KL_current_to_initial",
              coefficient=1.0, reduction="mean_source_rows_sum_cells",
              normalization="(teacher_CE_plus_row_KL)/max_teacher_CE_P0_1e-12",
              reference="actual_initial_FP32_UV_residual_plus_hard_logit_prior_then_FP64_log_softmax",
              mixing=.05, teacher_temperature=1.0, source_teacher="original_ReLU",
              columns="free", inner_loss="uniform_CE", probability_floor=False)


def source_files():
    local = Path(__file__).resolve().parent
    repository = next(p for p in local.parents if (p / "src/low_rank_assignment.py").is_file())
    paths = {name: local/name for name in
             ("initial_assignment_row_kl.py", "soft_ce_partition.py", "citation_search.py")}
    paths.update({name: repository/"src"/name for name in ("low_rank_assignment.py", "moments.py")})
    return {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}


def _seal(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def source_digest():
    return _seal(source_files())


def coefficient(value):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value not in (0, 1):
        raise ValueError("Assignment KL supports only disabled0 or fixed1")
    return float(value)


def candidate_controls(candidate):
    result = dict(candidate)
    if any(key.startswith("assignment_kl_") and key not in KEYS for key in result):
        raise ValueError("Unknown initial-assignment KL control")
    if not set(result).intersection(KEYS):
        return result
    weight = coefficient(result.get(KEYS[0], 0))
    if weight == 0:
        if any(result.get(key) is not None for key in KEYS[1:]):
            raise ValueError("Disabled assignment KL cannot carry activated provenance")
        for key in KEYS:
            result.pop(key, None)
        return result
    allowed = {"method", "width", "lr", "T", "rank", "penalty", "initialization", "alpha",
               "inner_loss_weighting", "mixing", "mass_mode", *KEYS}
    if (set(result)-allowed or result.get("method") != "low_rank" or result.get("width", 0) != 0
            or isinstance(result.get("T"), bool) or result.get("T") != 1
            or result.get("mass_mode", "free") != "free" or result.get("mixing", .05) != .05
            or result.get("inner_loss_weighting") != "uniform"
            or result.get("initialization") not in ("teacher_balanced", "teacher_joint")):
        raise ValueError("Assignment KL requires original fixedT1/free NODE/uniform ReLU CE")
    if type(result.get(KEYS[1])) is not int or result[KEYS[1]] != SCHEMA:
        raise ValueError("Assignment KL requires exact schema1")
    pin = source_digest()
    if result.get(KEYS[2], pin) != pin:
        raise ValueError("Assignment KL source changed; preserve its old cache namespace")
    result.update({KEYS[0]: weight, KEYS[1]: SCHEMA, KEYS[2]: pin})
    return result


def core_options(candidate):
    canonical = candidate_controls(candidate)
    if canonical.get(KEYS[0], 0) == 0:
        return {}
    return dict(assignment_kl_weight=1.0, assignment_kl_schema=SCHEMA,
                assignment_kl_source=canonical[KEYS[2]], assignment_kl_teacher_T=1.0)


def _finite(value, name, ndim, dtype=None):
    if (not torch.is_tensor(value) or value.layout != torch.strided or value.ndim != ndim
            or min(value.shape) <= 0 or (dtype is not None and value.dtype != dtype)
            or not bool(torch.isfinite(value).all())):
        raise ValueError(f"Require finite nonempty {name}")


def _descriptor(value):
    return dict(shape=list(value.shape), dtype=str(value.dtype),
                tensor=array_digest(value.detach().cpu().numpy()))


def _inputs(u, v, assignment, material, mixing, chunk_size, logp0):
    _finite(u, "U", 2)
    _finite(v, "V", 2)
    _finite(material, "material", 2, torch.float64)
    _finite(logp0, "detached initial logP", 2, torch.float64)
    if (u.dtype not in (torch.float32, torch.float64) or v.dtype != u.dtype or u.shape[1] != v.shape[1]
            or material.shape[0] != len(u) or logp0.shape != (len(u), len(v)) or logp0.requires_grad
            or not torch.is_tensor(assignment) or assignment.dtype != torch.int64
            or assignment.shape != (len(u),) or bool((assignment < 0).any()) or bool((assignment >= len(v)).any())
            or len({u.device, v.device, assignment.device, material.device, logp0.device}) != 1
            or isinstance(mixing, bool) or not isinstance(mixing, Real) or not 0 < mixing < 1
            or type(chunk_size) is not int or chunk_size < 1):
        raise ValueError("Assignment KL operands/layout/reference do not match")


class InitialAssignmentKLMoments(torch.autograd.Function):
    """Original streamed moments and row-KL with one combined low-rank VJP.

FP64 factors are allowed only for independent CPU derivative proofs. The native
core's activated contract separately requires its original FP32 factor factory.
    """
    @staticmethod
    def forward(ctx, u, v, assignment, material, mixing, chunk_size, logp0):
        _inputs(u, v, assignment, material, mixing, chunk_size, logp0)
        ctx.save_for_backward(u, v, assignment, material, logp0)
        ctx.mixing, ctx.chunk_size = mixing, chunk_size
        result = material.new_zeros(len(v), material.shape[1])
        value = material.new_zeros(())
        for start in range(0, len(u), chunk_size):
            end = start + chunk_size
            probability = (
                logit_block(u[start:end], v, assignment[start:end], mixing).to(material.dtype).softmax(1)
            )
            result += probability.T @ material[start:end] / len(u)
            logp = logit_block(u[start:end], v, assignment[start:end], mixing).to(material.dtype).log_softmax(1)
            if not bool(torch.isfinite(logp).all()):
                raise FloatingPointError("Nonfinite current assignment log-probabilities")
            value += (probability*(logp-logp0[start:end])).sum()/len(u)
        if not bool(torch.isfinite(result).all()) or not bool(torch.isfinite(value)):
            raise FloatingPointError("Nonfinite assignment KL/moments")
        return result, value

    @staticmethod
    def backward(ctx, gradient, value_gradient):
        u, v, assignment, material, logp0 = ctx.saved_tensors
        du, dv = torch.empty_like(u), torch.zeros_like(v)
        dm = torch.empty_like(material) if ctx.needs_input_grad[3] else None
        scale = math.sqrt(u.shape[1])
        if gradient is None:
            gradient = material.new_zeros(len(v), material.shape[1])
        if value_gradient is None:
            value_gradient = material.new_zeros(())
        if not bool(torch.isfinite(gradient).all()) or not bool(torch.isfinite(value_gradient)):
            raise FloatingPointError("Nonfinite assignment KL upstream cotangent")
        for start in range(0, len(u), ctx.chunk_size):
            end = start + ctx.chunk_size
            probability = (
                logit_block(u[start:end], v, assignment[start:end], ctx.mixing).to(material.dtype).softmax(1)
            )
            if dm is not None:
                dm[start:end] = probability @ gradient / len(u)
            direction = material[start:end] @ gradient.T / len(u)
            logp = logit_block(u[start:end], v, assignment[start:end], ctx.mixing).to(material.dtype).log_softmax(1)
            ell = logp-logp0[start:end]
            if not bool(torch.isfinite(ell).all()):
                raise FloatingPointError("Nonfinite assignment KL log-ratio")
            direction = direction + value_gradient*ell/len(u)
            block = (probability * (direction - (probability * direction).sum(1, keepdim=True))).to(
                u.dtype
            ) / scale
            du[start:end] = block @ v
            dv += block.T @ u[start:end]
        if not bool(torch.isfinite(du).all()) or not bool(torch.isfinite(dv).all()):
            raise FloatingPointError("Nonfinite combined assignment KL low-rank gradient")
        return du, dv, None, dm, None, None, None


@torch.no_grad()
def original_reference(z, q, assignment, rank, seed, mixing, chunk_size, config, pin, parameters=None):
    if (pin != source_digest() or not isinstance(config, dict) or type(seed) is not int
            or type(rank) is not int or rank < 1 or not torch.is_tensor(assignment)
            or assignment.dtype != torch.int64 or assignment.ndim != 1 or not len(assignment)
            or bool((assignment < 0).any())):
        raise ValueError("Assignment KL requires current source/config/seed/rank/hard assignment")
    _finite(z, "source z", 2, torch.float64)
    _finite(q, "original source Q", 2, torch.float64)
    if (z.requires_grad or q.requires_grad or len(z) != len(q) or z.device != q.device
            or bool((q < 0).any()) or float((q.sum(1)-1).abs().max()) > 32*torch.finfo(q.dtype).eps):
        raise ValueError("Require detached original FP64 ReLU source/probabilities")
    if parameters is None:
        parameters = initialize_factors(assignment, int(assignment.max())+1, rank, seed)
    if (not isinstance(parameters, (tuple, list)) or len(parameters) != 2
            or any(not torch.is_tensor(p) for p in parameters)):
        raise ValueError("Initial assignment requires actual U0/V0")
    u0, v0 = (p.detach() for p in parameters)
    _finite(u0, "native initial U0", 2, torch.float32)
    _finite(v0, "native initial V0", 2, torch.float32)
    if (type(rank) is not int or u0.shape != (len(z), rank) or v0.shape[1] != rank
            or bool(u0.ne(0).any()) or mixing != .05 or assignment.device != z.device):
        raise ValueError("Require original native node-factor P0 factory and mixing.05")
    prior = initial_logits(assignment, len(v0), mixing, u0.dtype)
    residual = u0.new_empty(len(u0), len(v0))
    a0 = torch.empty_like(residual)
    logp0 = z.new_empty(len(u0), len(v0))
    p0 = torch.empty_like(logp0)
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("Initial reference requires the original positive chunk size")
    for start in range(0, len(u0), chunk_size):
        end = start+chunk_size
        residual[start:end] = u0[start:end] @ v0.T / math.sqrt(rank)
        a0[start:end] = prior[start:end]+residual[start:end]
        logp0[start:end] = a0[start:end].to(z.dtype).log_softmax(1)
        p0[start:end] = a0[start:end].to(z.dtype).softmax(1)
    _inputs(u0, v0, assignment, z, mixing, chunk_size, logp0)
    if not bool((p0 > 0).all()):
        raise ValueError("Initial assignment requires strictly positive actual P0")
    reference = dict(schema=SCHEMA, initial_parameters=[u0.clone(), v0.clone()], log_probability=logp0.clone())
    context = dict(POLICY, implementation_sha256=pin, implementation_files_sha256=source_files(),
                   source_data_digest=array_digest(z.cpu().numpy(), q.cpu().numpy(), assignment.cpu().numpy()),
                   config_sha256=_seal({key: value for key, value in config.items() if key != "assignment_kl_context"}),
                   nodes=len(z), cells=len(v0), rank=rank, seed=seed, chunk_size=chunk_size,
                   device=str(z.device), material_dtype=str(z.dtype), factor_dtype=str(u0.dtype),
                   initial_U0=_descriptor(u0), initial_V0=_descriptor(v0), hard_assignment=_descriptor(assignment),
                   hard_logit_prior=_descriptor(prior), initial_residual=_descriptor(residual),
                   actual_A0_FP32=_descriptor(a0), log_P0=_descriptor(logp0), actual_P0=_descriptor(p0))
    return reference, context


def attach(saved, reference, context, config):
    saved.update(assignment_kl_reference=cpu_state(reference), assignment_kl_context=context,
                 assignment_kl_config=config)
    return saved


def _reference_equal(saved, expected):
    if (not isinstance(saved, dict) or set(saved) != set(expected) or type(saved.get("schema")) is not int
            or saved["schema"] != SCHEMA or not isinstance(saved.get("initial_parameters"), (tuple, list))
            or len(saved["initial_parameters"]) != 2):
        raise ValueError("Assignment KL cached reference schema changed")
    for actual, target in zip([*saved["initial_parameters"], saved["log_probability"]],
                              [*expected["initial_parameters"], expected["log_probability"]], strict=True):
        if not torch.is_tensor(actual) or actual.requires_grad or _descriptor(actual) != _descriptor(target):
            raise ValueError("Assignment KL cached actual U0/V0/logP0 bytes changed")


def validate_core_resume(saved, reference, context, config, steps, *, resume=True, step=None):
    if not isinstance(saved, dict) or type(steps) is not int or steps < 0:
        raise ValueError("Require typed assignment KL cache and horizon")
    saved_step = saved.get("step")
    if (type(saved_step) is not int or not 0 <= saved_step <= steps
            or (step is not None and (type(step) is not int or saved_step != step))):
        raise ValueError("Assignment KL checkpoint/frontier differs")
    actual_config = saved.get("config" if resume else "assignment_kl_config")
    if _seal(actual_config) != _seal(config) or _seal(saved.get("assignment_kl_context")) != _seal(context):
        raise ValueError("Assignment KL cached source/config/reference context differs")
    _reference_equal(saved.get("assignment_kl_reference"), reference)
    if resume:
        snapshots = saved.get("snapshots")
        if not isinstance(snapshots, dict) or saved_step not in snapshots:
            raise ValueError("Assignment KL resume lacks its current checkpoint")
        for key, snapshot in snapshots.items():
            if type(key) is not int or not 0 <= key <= saved_step:
                raise ValueError("Assignment KL resume contains a changed checkpoint")
            validate_core_resume(snapshot, reference, context, config, steps, resume=False, step=key)
    return saved


def _expected_citation_inputs(candidate, seed, z, q, assignment):
    from src.soft_ce_partition import optimize_ce_assignment
    canonical = candidate_controls(candidate)
    if canonical.get(KEYS[0]) != 1 or type(seed) is not int:
        raise ValueError("Require active canonical assignment KL and integer seed")
    values = {key: value.default for key, value in inspect.signature(optimize_ce_assignment).parameters.items()
              if value.default is not inspect.Parameter.empty}
    values.update(penalty=canonical["penalty"], lr=canonical["lr"], mixing=.05,
                  assignment_rank=canonical["rank"], factor_seed=seed, assignment_input="node",
                  assignment_encoder="linear", encoder_hidden=64, solver_mode="exact",
                  inner_method="newton_first", implicit_warm_start=True, mass_mode="free",
                  inner_loss_weighting="uniform", inner_max_iter=2000, inner_tol=1e-7,
                  cg_max_iter=512, cg_rtol=1e-6, cache_assignment=False, save_assignment=False)
    for key in ("steps", "folder", "checkpoint_steps", "resume_state", "save_resume", "log_every",
                "initial_representatives", "outer_indices", "implicit_solver", "inner_solver", "temperature_logits",
                "outer_targets", "stop", "temperature_initial", "temperature_lr", "cg_check_interval",
                "cache_assignment", "node_weighting", "node_weight_penalty", "node_weight_lr"):
        values.pop(key, None)
    values["data_digest"] = array_digest(z.detach().cpu().numpy(), q.detach().cpu().numpy(), assignment.cpu().numpy())
    values.update(assignment_kl_weight=1.0, assignment_kl_schema=SCHEMA,
                  assignment_kl_source=canonical[KEYS[2]], assignment_kl_teacher_T=1.0)
    reference, context = original_reference(z, q, assignment, canonical["rank"], seed, .05,
                                           values["chunk_size"], values, canonical[KEYS[2]])
    values["assignment_kl_context"] = context
    return values, reference, context


def expected_citation_config(candidate, seed, z, q, assignment):
    """Exact activated public core config; keeps signature-introspection defaults."""
    return _expected_citation_inputs(candidate, seed, z, q, assignment)[0]


def validate_cached(saved, candidate, z, q, assignment, seed, steps, *, resume=False, step=None):
    if candidate.get(KEYS[0], 0) == 0:
        return saved
    canonical = candidate_controls(candidate)
    config, reference, context = _expected_citation_inputs(canonical, seed, z, q, assignment)
    return validate_core_resume(saved, reference, context, config, steps, resume=resume, step=step)
