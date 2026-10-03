"""Direct uniform-cell label-prior KL; no source loading or P updates.

For moments [m, m*zbar, m*Qbar], the cell prior is mean_k(Qbar_k),
not the conserved mass-weighted prior. Both label and mass derivatives matter.
The term is independent of the fitted head; the CE implicit VJP is unchanged.
"""
import hashlib
import inspect
import json
import math
from numbers import Real
from pathlib import Path

import torch

from src.io import array_digest

SCHEMA = 1
KEYS = ("uniform_cell_q_prior_weight", "uniform_cell_q_prior_schema",
        "uniform_cell_q_prior_source_digest")
POLICY = dict(schema=SCHEMA, objective="teacher_CE_plus_KL_source_to_uniform_cell_Q_prior",
              coefficient=1.0, source_prior="detached_original_Q_mean_no_renormalization",
              cell_prior="arithmetic_mean_of_label_moment_divided_by_cell_mass",
              normalization="existing_teacher_CE_P0_scale_max_F0_1e-12",
              class_support="all_source_and_cell_prior_classes_strictly_positive_no_pseudocount")


def source_files():
    """Bind the new term, its core hook and public cache/identity semantics."""
    root = Path(__file__).resolve().parent
    return {name: hashlib.sha256((root/name).read_bytes()).hexdigest()
            for name in ("uniform_cell_q_prior.py", "soft_ce_partition.py", "citation_search.py")}


def source_digest():
    return hashlib.sha256(json.dumps(source_files(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def coefficient(value):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value not in (0, 1):
        raise ValueError("Uniform-cell prior supports only explicit coefficient0 or fixed1")
    return float(value)


def candidate_controls(candidate):
    """Only an explicit prior key changes the ordinary low-rank candidate."""
    result = dict(candidate)
    if any(key.startswith("uniform_cell_q_prior_") and key not in KEYS for key in result):
        raise ValueError("Unknown uniform-cell prior candidate control")
    if not set(result).intersection(KEYS):
        return result
    weight = coefficient(result.get(KEYS[0], 0))
    if weight == 0:
        if any(result.get(key) is not None for key in KEYS[1:]):
            raise ValueError("Disabled prior cannot carry activated provenance")
        for key in KEYS:
            result.pop(key, None)
        return result
    if (result.get("method") != "low_rank" or result.get("width", 0) != 0
            or result.get("inner_loss_weighting") != "uniform"
            or result.get("mass_mode", "free") != "free" or result.get("mixing", .05) != .05
            or any(key.startswith(("source_linear", "assignment_coordinate", "mlp_")) for key in result)
            or result.get("learn_temperature", False) or result.get("train_target_mix", 0.0) != 0):
        raise ValueError("Prior objective requires the original free NODE uniform-inner CE route")
    if type(result.get(KEYS[1])) is not int or result[KEYS[1]] != SCHEMA:
        raise ValueError("Activated prior requires exact schema1")
    pin = source_digest()
    if result.get(KEYS[2], pin) != pin:
        raise ValueError("Prior implementation changed; preserve its old candidate/cache namespace")
    result.update({KEYS[0]: weight, KEYS[1]: SCHEMA, KEYS[2]: pin})
    return result


def core_options(candidate):
    canonical = candidate_controls(candidate)
    if not canonical.get(KEYS[0], 0):
        return {}
    return dict(uniform_cell_q_prior_weight=1.0, uniform_cell_q_prior_schema=SCHEMA,
                uniform_cell_q_prior_source=canonical[KEYS[2]])


def _finite(value, name, ndim):
    if (not torch.is_tensor(value) or value.layout != torch.strided or value.dtype != torch.float64
            or value.ndim != ndim or min(value.shape) <= 0 or not bool(torch.isfinite(value).all())):
        raise ValueError(f"Require finite nonempty FP64 {name}")


@torch.no_grad()
def source_prior(q):
    _finite(q, "frozen original Q", 2)
    if (q.requires_grad or bool((q < 0).any())
            or float((q.sum(1)-1).abs().max()) > 32*torch.finfo(q.dtype).eps):
        raise ValueError("Source Q must be detached full-class probabilities without normalization")
    result = q.mean(0)
    if not bool((result > 0).all()):
        raise ValueError("Every original soft teacher class must have strictly positive support")
    return result.detach().clone()


def _decoded(moments, dimension, pi):
    _finite(moments, "moments", 2)
    _finite(pi, "detached source prior", 1)
    if (type(dimension) is not int or dimension < 0 or moments.shape[1] != 1+dimension+len(pi)
            or moments.device != pi.device or pi.requires_grad or not bool((pi > 0).all())
            or abs(float(pi.sum())-1) > 32*torch.finfo(pi.dtype).eps):
        raise ValueError("Moment layout/source prior must retain the frozen full class support")
    mass = moments[:, 0]
    totals = moments[:, dimension+1:]
    if bool((mass <= 0).any()) or bool((totals < 0).any()):
        raise ValueError("Require positive cell mass and nonnegative label moments")
    labels = totals/mass[:, None]
    mean = labels.mean(0)
    if not bool(torch.isfinite(labels).all()) or not bool(torch.isfinite(mean).all()) or not bool((mean > 0).all()):
        raise ValueError("Every uniform-cell prior class must be finite and strictly positive")
    # The ambient moment function deliberately does not renormalize labels.
    # Reachable P/material probability/conservation guards belong to the caller.
    return mass, labels, mean


def prior_value(moments, dimension, pi):
    """Differentiable ambient function; no clamp, smoothing or input mutation."""
    _, _, mean = _decoded(moments, dimension, pi)
    value = (pi*(pi.log()-mean.log())).sum()
    if not bool(torch.isfinite(value)):
        raise FloatingPointError("Nonfinite prior KL")
    return value


@torch.no_grad()
def prior_partials(moments, dimension, pi):
    """Return exact analytic dKL/d[m,mz,mQ], including the denominator VJP."""
    mass, labels, mean = _decoded(moments, dimension, pi)
    value = (pi*(pi.log()-mean.log())).sum()
    derivative_mean = -pi/mean
    gradient = torch.zeros_like(moments)
    gradient[:, dimension+1:] = derivative_mean[None, :]/(len(mass)*mass[:, None])
    gradient[:, 0] = -(labels*derivative_mean[None, :]).sum(1)/(len(mass)*mass)
    if not bool(torch.isfinite(value)) or not bool(torch.isfinite(gradient).all()):
        raise FloatingPointError("Nonfinite prior value or moment partial")
    return dict(value=value.detach(), moment_gradient=gradient.detach(),
                cell_prior=mean.detach().clone())


def combine_direction(teacher_direction, moments, dimension, pi, weight=1.0):
    """Disabled route returns the identical legacy direction, without prior algebra."""
    weight = coefficient(weight)
    if weight == 0:
        return teacher_direction
    partial = prior_partials(moments, dimension, pi)
    if (teacher_direction.shape != moments.shape or teacher_direction.dtype != moments.dtype
            or teacher_direction.device != moments.device or not bool(torch.isfinite(teacher_direction).all())):
        raise ValueError("Teacher direction must be finite and match moments")
    result = teacher_direction+partial["moment_gradient"]
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError("Nonfinite combined CE/prior moment direction")
    return result


def config_context(q, schema, pin):
    if type(schema) is not int or schema != SCHEMA or pin != source_digest():
        raise ValueError("Activated prior requires exact schema/implementation source pin")
    pi = source_prior(q)
    context = dict(POLICY, implementation_sha256=pin, implementation_files_sha256=source_files(),
                   source_prior_values=pi.cpu().tolist(),
                   source_prior_sha256=array_digest(pi.cpu().numpy()), source_prior_dtype=str(pi.dtype),
                   classes=len(pi))
    return pi, context


def expected_citation_config(candidate, seed, z, q, assignment):
    """Bind activated cached public-NODE controls to the unchanged core signature."""
    from src.soft_ce_partition import optimize_ce_assignment
    canonical = candidate_controls(candidate)
    if canonical.get(KEYS[0]) != 1 or type(seed) is not int:
        raise ValueError("Require activated canonical prior and integer condensation seed")
    values = {key: value.default for key, value in inspect.signature(optimize_ce_assignment).parameters.items()
              if value.default is not inspect.Parameter.empty}
    values.update(penalty=canonical["penalty"], lr=canonical["lr"], mixing=canonical.get("mixing", .05),
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
    _, context = config_context(q, canonical[KEYS[1]], canonical[KEYS[2]])
    values.update(uniform_cell_q_prior_weight=1.0, uniform_cell_q_prior_schema=SCHEMA,
                  uniform_cell_q_prior_source=canonical[KEYS[2]], uniform_cell_q_prior_context=context)
    return values


def validate_cached(saved, candidate, z, q, assignment, seed, steps, *, resume=False, step=None):
    """Validate activated cache reads even when a completed run skips the optimizer.

The data digest binds the actual z, full Q and hard assignment; the prior
context binds detached Q.mean, policy and this helper. This is current-cache
compatibility, not equivalence to a historical full source manifest.
    """
    if candidate.get(KEYS[0], 0) == 0:
        return saved
    if (type(steps) is not int or steps < 0 or not isinstance(saved, dict)
            or (step is not None and type(step) is not int)):
        raise ValueError("Require typed prior cache and horizon")
    expected = expected_citation_config(candidate, seed, z, q, assignment)
    saved_step = saved.get("step")
    if type(saved_step) is not int or not 0 <= saved_step <= steps or (step is not None and saved_step != step):
        raise ValueError("Prior cache step differs from the requested horizon/checkpoint")
    config = saved.get("config" if resume else "uniform_cell_q_prior_config")
    if (not isinstance(config, dict) or config != expected
            or any(type(config.get(key)) is not type(value) for key, value in expected.items()
                   if type(value) in (bool, int))):
        raise ValueError("Prior cache configuration/source data differs from the current candidate")
    context = expected["uniform_cell_q_prior_context"]
    if json.dumps(config["uniform_cell_q_prior_context"], sort_keys=True, allow_nan=False) != json.dumps(context, sort_keys=True, allow_nan=False):
        raise ValueError("Prior cache context changed typed policy/source-prior metadata")
    if resume:
        snapshots = saved.get("snapshots")
        if not isinstance(snapshots, dict) or saved_step not in snapshots:
            raise ValueError("Prior resume is missing its exact current snapshot")
        for key, snapshot in snapshots.items():
            if type(key) is not int or not 0 <= key <= saved_step:
                raise ValueError("Prior resume contains a changed checkpoint step")
            validate_cached(snapshot, candidate, z, q, assignment, seed, steps, step=key)
    elif json.dumps(saved.get("uniform_cell_q_prior_context"), sort_keys=True, allow_nan=False) != json.dumps(context, sort_keys=True, allow_nan=False):
        raise ValueError("Prior checkpoint lost or changed its prior context")
    return saved
