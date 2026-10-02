import json
import time
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import trange

from src.balanced_assignment import BalancedMoments, CachedBalancedMoments
from src.head import fit_head, head_objective
from src.io import array_digest, cpu_state
from src.low_rank_assignment import (
    CachedLowRankMoments,
    LowRankLogits,
    LowRankMoments,
    WeightedLowRankMoments,
    assignment_inputs,
    encode_nodes,
    initialize_encoder,
    initialize_factors,
    initialize_mlp,
    normalized_node_weights,
)
from src.moments import (
    AssignmentMoments,
    augmented,
    decode_moments,
    initial_logits,
    make_material,
)


def head_gradient(x, labels, mass, theta, penalty):
    error = labels.sum(1, keepdim=True) * (x @ theta.T).softmax(1) - labels
    return (mass[:, None] * error).T @ x + penalty * theta


@torch.no_grad()
def hessian_operator(x, labels, mass, theta, penalty):
    probability = (x @ theta.T).softmax(1)
    weight = mass * labels.sum(1)
    diagonal = (weight[:, None] * probability * (1 - probability)).T @ x.square() + penalty

    def multiply(vector):
        direction = x @ vector.T
        projected = probability * (direction - (probability * direction).sum(1, keepdim=True))
        return (weight[:, None] * projected).T @ x + penalty * vector

    return multiply, diagonal


@torch.no_grad()
def conjugate_gradient(
    multiply,
    rhs,
    diagonal,
    rtol=1e-6,
    atol=1e-12,
    max_iter=512,
    initial=None,
    restart_interval=50,
    check_interval=1,
):
    if check_interval > 1:
        return grouped_conjugate_gradient(
            multiply, rhs, diagonal, rtol, atol, max_iter, initial, restart_interval, check_interval
        )
    solution = torch.zeros_like(rhs) if initial is None else initial.detach().clone()
    residual = rhs.clone() if initial is None else rhs - multiply(solution)
    norm = float(rhs.norm())
    target = max(atol, rtol * norm)
    direction = residual / diagonal
    product = (residual * direction).sum()
    iterations = 0
    for iteration in range(max_iter):
        if float(residual.norm()) <= target:
            break
        image = multiply(direction)
        curvature = (direction * image).sum()
        if not torch.isfinite(curvature) or float(curvature) <= 0:
            break
        alpha = product / curvature
        solution.add_(direction, alpha=float(alpha))
        residual.sub_(image, alpha=float(alpha))
        iterations = iteration + 1
        restart = iterations % restart_interval == 0 or float(residual.norm()) <= target
        if restart:
            residual = rhs - multiply(solution)
        preconditioned = residual / diagonal
        next_product = (residual * preconditioned).sum()
        direction = preconditioned if restart else preconditioned + (next_product / product) * direction
        product = next_product
    residual_norm = float((rhs - multiply(solution)).norm())
    return solution, dict(
        cg_iterations=iterations,
        cg_residual=residual_norm,
        cg_relative_residual=residual_norm / max(norm, 1e-30),
        cg_converged=np.isfinite(residual_norm) and residual_norm <= target,
    )


@torch.no_grad()
def grouped_conjugate_gradient(
    multiply, rhs, diagonal, rtol, atol, max_iter, initial, restart_interval, check_interval
):
    solution = torch.zeros_like(rhs) if initial is None else initial.detach().clone()
    residual = rhs - multiply(solution)
    norm = float(rhs.norm())
    target = max(atol, rtol * norm)
    direction = residual / diagonal
    product = (residual * direction).sum()
    iterations = 0
    broken = rhs.new_tensor(False, dtype=torch.bool)
    residual_norm = float(residual.norm())
    while iterations < max_iter and residual_norm > target and np.isfinite(residual_norm):
        for _ in range(min(check_interval, max_iter - iterations)):
            active = (residual.norm() > target) & ~broken
            image = multiply(direction)
            curvature = (direction * image).sum()
            valid = torch.isfinite(curvature) & (curvature > 0) & torch.isfinite(product) & (product > 0)
            broken = broken | (active & ~valid)
            take = active & valid
            alpha = torch.where(
                take, product / torch.where(valid, curvature, torch.ones_like(curvature)), 0.0
            )
            solution.add_(torch.where(take, alpha * direction, torch.zeros_like(direction)))
            residual.sub_(torch.where(take, alpha * image, torch.zeros_like(image)))
            iterations += 1
            if iterations % restart_interval == 0:
                residual = rhs - multiply(solution)
                direction = residual / diagonal
                product = (residual * direction).sum()
            else:
                preconditioned = residual / diagonal
                next_product = (residual * preconditioned).sum()
                beta = torch.where(
                    take, next_product / torch.where(valid, product, torch.ones_like(product)), 0.0
                )
                direction = preconditioned + beta * direction
                product = next_product
        residual_norm = float(residual.norm())
        if bool(broken):
            break
        if residual_norm <= target:
            residual = rhs - multiply(solution)
            residual_norm = float(residual.norm())
            direction = residual / diagonal
            product = (residual * direction).sum()
    residual_norm = float((rhs - multiply(solution)).norm())
    return solution, dict(
        cg_iterations=iterations,
        cg_residual=residual_norm,
        cg_relative_residual=residual_norm / max(norm, 1e-30),
        cg_converged=np.isfinite(residual_norm) and residual_norm <= target,
    )


@torch.no_grad()
def solve_head_system(
    x,
    labels,
    mass,
    theta,
    penalty,
    rhs,
    rtol=1e-6,
    atol=1e-12,
    max_iter=512,
    reduced_limit=2048,
    direct_limit=8192,
    initial=None,
    cg_check_interval=1,
):
    multiply, diagonal = hessian_operator(x, labels, mass, theta, penalty)
    classes = theta.shape[0]
    dimension = min(x.shape) * classes
    diagnostic = dict(cg_iterations=0)
    if not (x.shape[1] > 2 * x.shape[0] and dimension <= reduced_limit):
        solution, diagnostic = conjugate_gradient(
            multiply,
            rhs,
            diagonal,
            rtol=rtol,
            atol=atol,
            max_iter=max_iter,
            initial=initial,
            check_interval=cg_check_interval,
        )
        diagnostic.update(hessian_solver="pcg", hessian_reduced_dimension=0)
        if diagnostic["cg_converged"]:
            return solution, diagnostic
    if dimension > reduced_limit:
        solution, retry = conjugate_gradient(
            multiply,
            rhs,
            diagonal,
            rtol=rtol,
            atol=atol,
            max_iter=4 * max_iter,
            initial=solution if bool(torch.isfinite(solution).all()) else None,
            restart_interval=512,
            check_interval=cg_check_interval,
        )
        retry.update(
            hessian_solver="pcg_extended",
            hessian_reduced_dimension=0,
            cg_iterations=diagnostic["cg_iterations"] + retry["cg_iterations"],
        )
        if retry["cg_converged"] or dimension > direct_limit:
            return solution, retry
        diagnostic = retry
    _, singular, basis = torch.linalg.svd(x, full_matrices=False)
    projected_x = x @ basis.T
    projected_rhs = rhs @ basis.T
    probability = (x @ theta.T).softmax(1)
    system = x.new_zeros(dimension, dimension)
    for start in range(0, len(x), 128):
        p = probability[start : start + 128]
        block = projected_x[start : start + 128]
        covariance = torch.diag_embed(p) - p[:, :, None] * p[:, None, :]
        covariance *= (mass[start : start + 128] * labels[start : start + 128].sum(1))[:, None, None]
        system.add_(torch.einsum("icd,ia,ib->cadb", covariance, block, block).reshape(dimension, dimension))
    system = (system + system.T) / 2 + penalty * torch.eye(dimension, dtype=x.dtype, device=x.device)
    projected_solution = torch.linalg.solve(system, projected_rhs.flatten()).reshape(classes, len(singular))
    solution = projected_solution @ basis + (rhs - projected_rhs @ basis) / penalty
    norm = float(rhs.norm())
    residual = float((multiply(solution) - rhs).norm())
    diagnostic.update(
        hessian_solver="reduced_direct_fallback" if dimension > reduced_limit else "reduced_direct",
        hessian_reduced_dimension=dimension,
        cg_residual=residual,
        cg_relative_residual=residual / max(norm, 1e-30),
        cg_converged=bool(torch.isfinite(solution).all())
        and np.isfinite(residual)
        and residual <= max(atol, rtol * norm),
    )
    return solution, diagnostic


def solve_inner(
    centers,
    labels,
    mass,
    penalty,
    initial=None,
    max_iter=2000,
    grad_tol=1e-7,
    polish_steps=8,
    cg_max_iter=512,
    cg_check_interval=1,
):
    fitted = fit_head(
        centers.detach(),
        labels.detach(),
        mass.detach(),
        penalty,
        max_iter=max_iter,
        grad_tol=grad_tol,
        initial_theta=initial,
    )
    theta = fitted["theta"]
    x = augmented(centers.detach())
    labels, mass = labels.detach(), mass.detach()
    polished = 0
    with torch.no_grad():
        for _ in range(polish_steps):
            gradient = head_gradient(x, labels, mass, theta, penalty)
            if float(gradient.abs().max()) <= grad_tol:
                break
            direction, diagnostic = solve_head_system(
                x,
                labels,
                mass,
                theta,
                penalty,
                gradient,
                rtol=1e-8,
                atol=1e-14,
                max_iter=cg_max_iter,
                cg_check_interval=cg_check_interval,
            )
            if not diagnostic["cg_converged"]:
                break
            objective = head_objective(x, labels, mass, theta, penalty)
            decrease = (gradient * direction).sum()
            accepted = False
            for exponent in range(24):
                step = 0.5**exponent
                candidate = theta - step * direction
                value = head_objective(x, labels, mass, candidate, penalty)
                if value <= objective - 1e-4 * step * decrease:
                    theta, accepted = candidate, True
                    polished += 1
                    break
            if not accepted:
                break
        gradient = head_gradient(x, labels, mass, theta, penalty)
        maximum = float(gradient.abs().max())
    return dict(
        theta=theta.detach(),
        inner_grad_max=maximum,
        inner_converged=np.isfinite(maximum) and maximum <= grad_tol,
        inner_iterations=fitted["iterations"],
        inner_polish_steps=polished,
    )


def solve_inner_newton_first(
    centers,
    labels,
    mass,
    penalty,
    initial=None,
    max_iter=2000,
    grad_tol=1e-7,
    cg_max_iter=512,
    newton_steps=8,
    cg_check_interval=1,
):
    if newton_steps < 0:
        raise ValueError("Newton budget must be nonnegative")
    theta = initial.detach().to(centers).clone() if initial is not None else None
    accepted, attempts, cg_iterations, line_evaluations = 0, 0, 0, 0
    with torch.no_grad():
        x = augmented(centers.detach())
        if theta is not None:
            for _ in range(newton_steps):
                gradient = head_gradient(x, labels, mass, theta, penalty)
                maximum = float(gradient.abs().max())
                if not np.isfinite(maximum) or maximum <= grad_tol:
                    break
                attempts += 1
                tolerance = min(0.1, max(1e-4, maximum**0.5))
                direction, diagnostic = solve_head_system(
                    x,
                    labels,
                    mass,
                    theta,
                    penalty,
                    gradient,
                    rtol=tolerance,
                    atol=1e-14,
                    max_iter=cg_max_iter,
                    cg_check_interval=cg_check_interval,
                )
                cg_iterations += diagnostic["cg_iterations"]
                decrease = float((gradient * direction).sum())
                if not diagnostic["cg_converged"] or not np.isfinite(decrease) or decrease <= 0:
                    break
                objective = head_objective(x, labels, mass, theta, penalty)
                for exponent in range(24):
                    step = 0.5**exponent
                    proposal = theta - step * direction
                    value = head_objective(x, labels, mass, proposal, penalty)
                    line_evaluations += 1
                    if bool(torch.isfinite(value)) and bool(value <= objective - 1e-4 * step * decrease):
                        theta = proposal
                        accepted += 1
                        break
                else:
                    break
            maximum = float(head_gradient(x, labels, mass, theta, penalty).abs().max())
        else:
            maximum = float("inf")
    converged = np.isfinite(maximum) and maximum <= grad_tol
    if converged:
        result = dict(
            theta=theta.detach(),
            inner_grad_max=maximum,
            inner_converged=True,
            inner_iterations=0,
            inner_polish_steps=0,
        )
    else:
        result = solve_inner(
            centers,
            labels,
            mass,
            penalty,
            theta,
            max_iter,
            grad_tol,
            cg_max_iter=cg_max_iter,
            cg_check_interval=cg_check_interval,
        )
    result.update(
        inner_newton_steps=accepted,
        inner_newton_attempts=attempts,
        inner_newton_cg_iterations=cg_iterations,
        inner_newton_line_evaluations=line_evaluations,
        inner_lbfgs_fallback=not converged,
    )
    return result


@torch.no_grad()
def track_inner(centers, labels, mass, penalty, initial, steps=2, cg_steps=8, grad_tol=1e-7):
    x, theta = augmented(centers), initial.detach().clone()
    accepted, failed = 0, False
    for _ in range(steps):
        gradient = head_gradient(x, labels, mass, theta, penalty)
        if float(gradient.abs().max()) <= grad_tol:
            break
        multiply, diagonal = hessian_operator(x, labels, mass, theta, penalty)
        direction, _ = conjugate_gradient(multiply, gradient, diagonal, max_iter=cg_steps, rtol=1e-2)
        decrease = (gradient * direction).sum()
        if not bool(torch.isfinite(direction).all()) or float(decrease) <= 0:
            direction = gradient / diagonal
            decrease = (gradient * direction).sum()
        objective = head_objective(x, labels, mass, theta, penalty)
        for exponent in range(16):
            step = 0.5**exponent
            candidate = theta - step * direction
            value = head_objective(x, labels, mass, candidate, penalty)
            if torch.isfinite(value) and value <= objective - 1e-4 * step * decrease:
                theta = candidate
                accepted += 1
                break
        else:
            failed = True
            break
    maximum = float(head_gradient(x, labels, mass, theta, penalty).abs().max())
    return dict(
        theta=theta,
        inner_grad_max=maximum,
        inner_converged=np.isfinite(maximum) and maximum <= grad_tol,
        inner_iterations=accepted,
        inner_polish_steps=0,
        tracking_failed=failed or not np.isfinite(maximum),
    )


@torch.no_grad()
def outer_value_gradient(z, q, theta, chunk_size=4096, features=None):
    value, gradient = z.new_zeros(()), torch.zeros_like(theta)
    for start in range(0, len(z), chunk_size):
        x = (
            augmented(z[start : start + chunk_size])
            if features is None
            else features[start : start + chunk_size]
        )
        labels = q[start : start + chunk_size]
        log_probability = (x @ theta.T).log_softmax(1)
        value -= (labels * log_probability).sum() / len(z)
        error = labels.sum(1, keepdim=True) * log_probability.exp() - labels
        gradient += error.T @ x / len(z)
    return float(value), gradient


def implicit_moment_gradient(moments, dimension, theta, vector, penalty, loss_weighting="mass"):
    variable = moments.detach().requires_grad_()
    centers, labels, mass = decode_moments(variable, dimension)
    if loss_weighting == "uniform":
        mass = torch.full_like(mass, 1 / len(mass))
    gradient = head_gradient(augmented(centers), labels, mass, theta.detach(), penalty)
    (result,) = torch.autograd.grad(-(gradient * vector.detach()).sum(), variable)
    return result


def temperature_labels(logits, log_temperature):
    return (logits.detach() / log_temperature.exp().to(logits.dtype)).softmax(1).double()


def _check_assignment_stop(stop):
    if stop():
        raise InterruptedError("CE assignment optimization interrupted")


def _save_checkpoint_atomic(snapshot, path):
    # A hidden temporary file also stays outside load_ce_snapshots' step_*.pt glob
    # if a hard process kill prevents cleanup.
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        torch.save(snapshot, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def optimize_ce_assignment(
    z,
    q,
    assignment,
    penalty=3e-05,
    steps=300,
    lr=0.01,
    mixing=0.05,
    chunk_size=4096,
    inner_max_iter=2000,
    inner_tol=1e-07,
    cg_max_iter=512,
    cg_rtol=1e-06,
    save_assignment=True,
    folder=None,
    checkpoint_steps=(),
    mass_mode="free",
    balance_steps=300,
    balance_tol=1e-08,
    balance_backend="cached",
    balance_cg_steps=512,
    balance_cg_rtol=1e-07,
    outer_chunk_size=65536,
    log_every=10,
    assignment_rank=None,
    factor_seed=0,
    assignment_input="node",
    assignment_encoder="linear",
    encoder_hidden=64,
    resume_state=None,
    save_resume=False,
    solver_mode="exact",
    tracking_inner_steps=2,
    tracking_cg_steps=8,
    tracking_refresh=20,
    feature_control="joint",
    initial_representatives=None,
    outer_indices=None,
    node_weighting=False,
    node_weight_penalty=0.0,
    node_weight_lr=None,
    implicit_solver=None,
    implicit_warm_start=None,
    inner_method="lbfgs",
    inner_solver=None,
    cg_check_interval=1,
    cache_assignment=False,
    temperature_logits=None,
    temperature_initial=0.3,
    temperature_lr=0.003,
    outer_targets=None,
    inner_loss_weighting="mass",
    stop=lambda: False,
    **mlp_initial_options,
):
    if not callable(stop):
        raise ValueError("stop must be callable")
    _check_assignment_stop(stop)
    if feature_control != "joint" or initial_representatives is not None:
        raise ValueError("Only clustering-derived features and labels are supported")
    if inner_loss_weighting not in ("mass", "uniform"):
        raise ValueError("Unknown inner CE weighting")
    if not isinstance(cg_check_interval, int) or cg_check_interval < 1:
        raise ValueError("cg_check_interval must be a positive integer")
    if cache_assignment and (assignment_rank is None or mass_mode != "free" or node_weighting):
        raise ValueError("Assignment caching requires unweighted free-mass low-rank assignments")
    if implicit_warm_start is None:
        implicit_warm_start = (
            True if resume_state is None else resume_state["config"].get("implicit_warm_start", False)
        )
    resume_config = {
        key: value
        for key, value in locals().copy().items()
        if key
        not in (
            "z",
            "q",
            "assignment",
            "steps",
            "folder",
            "checkpoint_steps",
            "resume_state",
            "save_resume",
            "log_every",
            "initial_representatives",
            "outer_indices",
            "implicit_solver",
            "inner_solver",
            "temperature_logits",
            "outer_targets",
            "stop",
            "mlp_initial_options",
        )
    }
    if set(mlp_initial_options) - {"mlp_initial_mass_schema", "mlp_initial_mass_source",
                                   "mlp_output_centering", "mlp_source_centering_schema", "mlp_source_centering_source"}:
        raise ValueError("Unknown MLP initial-mass control")
    mlp_initial_mass_schema = mlp_initial_options.get("mlp_initial_mass_schema")
    mlp_initial_mass_source = mlp_initial_options.get("mlp_initial_mass_source")
    centering_mode = mlp_initial_options.get("mlp_output_centering")
    centering_schema = mlp_initial_options.get("mlp_source_centering_schema")
    centering_source = mlp_initial_options.get("mlp_source_centering_source")
    centered_active = centering_mode is not None
    if inner_loss_weighting == "mass":
        resume_config.pop("inner_loss_weighting")
    if outer_targets is not None:
        if (
            outer_targets.shape != q.shape
            or not bool(torch.isfinite(outer_targets).all())
            or bool((outer_targets < 0).any())
            or (not torch.allclose(outer_targets.sum(1), torch.ones_like(outer_targets[:, 0]), atol=1e-06))
        ):
            raise ValueError("Outer targets must be node-aligned probability labels")
        resume_config["outer_targets_digest"] = array_digest(outer_targets.detach().cpu().numpy())
    if temperature_logits is None:
        resume_config.pop("temperature_initial")
        resume_config.pop("temperature_lr")
    else:
        if (
            assignment_rank is None
            or assignment_input != "node"
            or mass_mode != "free"
            or node_weighting
            or (solver_mode != "exact")
        ):
            raise ValueError(
                "Learnable temperature requires exact, unweighted, free-mass node-factor optimization"
            )
        if (
            temperature_logits.shape != q.shape
            or not bool(torch.isfinite(temperature_logits).all())
            or any((not np.isfinite(t) or t <= 0 for t in (temperature_initial, temperature_lr)))
        ):
            raise ValueError("Invalid temperature logits or settings")
        resume_config["temperature_logits_digest"] = array_digest(temperature_logits.detach().cpu().numpy())
        temperature_logits = temperature_logits.detach().to(z.device)
    if inner_method not in ("lbfgs", "newton_first"):
        raise ValueError("Unknown exact inner solver")
    if inner_method == "lbfgs":
        resume_config.pop("inner_method")
    if cg_check_interval == 1:
        resume_config.pop("cg_check_interval")
    if not cache_assignment:
        resume_config.pop("cache_assignment")
    if outer_indices is not None:
        resume_config["outer_digest"] = array_digest(outer_indices.cpu().numpy())
    if resume_state is not None or save_resume:
        resume_config["data_digest"] = array_digest(
            z.detach().cpu().numpy(), q.detach().cpu().numpy(), assignment.cpu().numpy()
        )
    if not node_weighting:
        for key in ("node_weighting", "node_weight_penalty", "node_weight_lr"):
            resume_config.pop(key)
    if node_weighting and (assignment_rank is None or assignment_input != "node" or mass_mode != "free"):
        raise ValueError("Node weighting requires free-mass node-factor joint optimization")
    if (
        not np.isfinite(node_weight_penalty)
        or node_weight_penalty < 0
        or (node_weight_lr is not None and (not np.isfinite(node_weight_lr) or node_weight_lr < 0))
    ):
        raise ValueError("Invalid node weight penalty or learning rate")
    if steps < 0 or any(
        (
            not np.isfinite(v) or v <= 0
            for v in (penalty, lr, chunk_size, inner_max_iter, inner_tol, cg_max_iter, cg_rtol)
        )
    ):
        raise ValueError("Require positive finite solver settings and nonnegative steps")
    if mass_mode not in ("free", "uniform", "initial") or balance_steps < 1 or (not 0 < balance_tol < 1):
        raise ValueError("Invalid mass constraint settings")
    if (
        balance_backend not in ("cached", "chunked")
        or min(balance_cg_steps, outer_chunk_size, log_every) < 1
        or (not 0 < balance_cg_rtol < 1)
    ):
        raise ValueError("Invalid performance settings")
    if assignment_input not in ("node", "features", "features_labels") or (
        assignment_input != "node" and assignment_rank is None
    ):
        raise ValueError("Feature assignment requires assignment_rank and a supported input mode")
    if assignment_encoder not in ("linear", "mlp") or (
        assignment_input == "node" and assignment_encoder != "linear"
    ):
        raise ValueError("MLP encoder requires feature-conditioned assignments")
    if solver_mode not in ("exact", "tracking") or any(
        (
            not isinstance(v, (int, np.integer)) or v < 1
            for v in (tracking_inner_steps, tracking_cg_steps, tracking_refresh)
        )
    ):
        raise ValueError("Invalid tracking solver settings")
    if mass_mode == "initial":
        if (torch.get_default_dtype() != torch.float32
                or type(mlp_initial_mass_schema) is not int or mlp_initial_mass_schema != 1
                or assignment_input != "features" or assignment_encoder != "mlp"
                or assignment_rank is None or inner_loss_weighting != "uniform"
                or solver_mode != "exact" or node_weighting or temperature_logits is not None
                or outer_targets is not None or outer_indices is not None or implicit_solver is not None
                or inner_solver is not None or cache_assignment or save_assignment or not save_resume
                or mixing != .05 or balance_backend != "chunked" or balance_steps != 5000
                or balance_tol != 1e-8):
            raise ValueError("Initial mass is an MLP-only fixed-P0 uniform-CE pilot with verifiable checkpoints")
        resume_config.update(mlp_initial_mass_schema=1, mlp_initial_mass_source=mlp_initial_mass_source)
    elif mlp_initial_mass_schema is not None or mlp_initial_mass_source is not None:
        raise ValueError("MLP initial provenance controls require mass_mode='initial'")
    if centered_active:
        if (centering_mode != "source_mean_v1" or type(centering_schema) is not int or centering_schema != 1
                or torch.get_default_dtype() != torch.float32 or assignment_input != "features"
                or assignment_encoder != "mlp" or assignment_rank is None or mass_mode != "free"
                or inner_loss_weighting != "uniform" or solver_mode != "exact" or node_weighting
                or temperature_logits is not None or outer_targets is not None or outer_indices is not None
                or implicit_solver is not None or inner_solver is not None or cache_assignment or save_assignment
                or not save_resume or mixing != .05):
            raise ValueError("Source centering requires native MLP/free-mass/uniform-CE verifiable checkpoints")
        resume_config.update(mlp_output_centering=centering_mode, mlp_source_centering_schema=centering_schema,
                             mlp_source_centering_source=centering_source)
    elif centering_schema is not None or centering_source is not None:
        raise ValueError("Source centering provenance requires its explicit output mode")
    checkpoints = set(checkpoint_steps)
    if any((not isinstance(step, (int, np.integer)) or not 0 <= step <= steps for step in checkpoints)):
        raise ValueError("Checkpoint steps must be integers within the optimization budget")
    if checkpoints or mass_mode == "initial" or centered_active:
        checkpoints.update((0, steps))
    snapshots = {}
    _check_assignment_stop(stop)
    material = make_material(z, q)
    outer_z = z if outer_indices is None else z[outer_indices]
    outer_q = q if outer_targets is None else outer_targets.detach().to(q)
    outer_q = (outer_q if outer_indices is None else outer_q[outer_indices]).detach()
    if not len(outer_z):
        raise ValueError("Outer loss requires at least one node")
    full_features = augmented(outer_z)
    clusters = int(assignment.max()) + 1
    if assignment_rank is None:
        logits = initial_logits(assignment, clusters, mixing).requires_grad_()
        parameters = [logits]
    elif assignment_input != "node":
        inputs = assignment_inputs(z, q, assignment_input)
        if assignment_encoder == "mlp":
            encoder_parameters, v = initialize_mlp(
                inputs, clusters, assignment_rank, encoder_hidden, factor_seed
            )
        else:
            weight, v = initialize_encoder(inputs, clusters, assignment_rank, factor_seed)
            encoder_parameters = [weight]
        parameters = [*encoder_parameters, v]
    else:
        u, v = initialize_factors(assignment, clusters, assignment_rank, factor_seed)
        parameters = [u, v]
    if mass_mode == "initial":
        from src.mlp_initial_mass import original_context, validate_resume
        mass_target, initial_mass_context, initial_parameters = original_context(
            z, q, assignment, assignment_rank, encoder_hidden, factor_seed, mixing,
            chunk_size, resume_config, mlp_initial_mass_source)
        if [array_digest(p.detach().cpu().numpy()) for p in parameters] != [
                array_digest(p.detach().cpu().numpy()) for p in initial_parameters]:
            raise ValueError("Native initial MLP parameters differ from the source target derivation")
        if resume_state is not None:
            validate_resume(resume_state, z, q, assignment, mass_target, initial_mass_context,
                            initial_parameters, steps, folder)
        elif folder is not None and (Path(folder) / "checkpoints").exists() and any(
                (Path(folder) / "checkpoints").glob("step_*.pt")):
            raise ValueError("Initial-mass cached checkpoints require a verifiable resume state")
    if centered_active:
        from src.mlp_source_centering import original_context as centered_context
        from src.mlp_source_centering import validate_resume as validate_centered_resume
        centering_context, centering_initial = centered_context(
            z, q, assignment, assignment_rank, encoder_hidden, factor_seed, mixing, chunk_size,
            resume_config, centering_source)
        if [array_digest(p.detach().cpu().numpy()) for p in parameters] != [
                array_digest(p.detach().cpu().numpy()) for p in centering_initial]:
            raise ValueError("Centered initializer differs from its native source provenance")
        if resume_state is not None:
            validate_centered_resume(resume_state, z, q, assignment, centering_context,
                                     centering_initial, steps, folder)
        elif folder is not None and any((Path(folder) / "checkpoints").glob("step_*.pt")):
            raise ValueError("Centered cached checkpoints require a verifiable resume")
    if node_weighting:
        node_logits = z.new_zeros(len(z), dtype=torch.float32).requires_grad_()
        groups = [
            dict(params=parameters),
            dict(params=[node_logits], lr=lr if node_weight_lr is None else node_weight_lr),
        ]
        parameters = [*parameters, node_logits]
    else:
        groups = parameters
    if temperature_logits is not None:
        log_temperature = z.new_tensor(np.log(temperature_initial)).requires_grad_()
        groups = [dict(params=parameters), dict(params=[log_temperature], lr=temperature_lr)]
        parameters = [*parameters, log_temperature]
    optimizer = torch.optim.Adam(groups, lr=lr, eps=1e-12, foreach=False)
    folder = Path(folder) if folder is not None else None
    if folder is not None:
        folder.mkdir(parents=True, exist_ok=True)
    history, theta, best, dual = ([], None, float("inf"), None)
    vector = None
    start_step = 0
    solve_student = inner_solver or partial(
        solve_inner_newton_first if inner_method == "newton_first" else solve_inner,
        cg_check_interval=cg_check_interval,
    )
    if resume_state is not None:
        saved_config = dict(resume_state["config"])
        saved_config.setdefault("implicit_warm_start", False)
        if saved_config != resume_config or resume_state["step"] > steps:
            raise ValueError("Resume state does not match data, solver settings or step budget")
        with torch.no_grad():
            for parameter, saved in zip(parameters, resume_state["parameters"], strict=True):
                parameter.copy_(saved.to(parameter))
        optimizer.load_state_dict(resume_state["optimizer"])
        start_step = resume_state["step"]
        theta = resume_state["theta"].to(z)
        if solver_mode == "tracking":
            theta = resume_state["tracking_theta_before"]
            theta = theta.to(z) if theta is not None else None
            vector = resume_state["tracking_vector_before"]
            vector = vector.to(z) if vector is not None else None
        elif implicit_warm_start:
            vector = resume_state.get("tracking_vector_before")
            vector = vector.to(z) if vector is not None else None
        dual = resume_state["dual"].to(z) if resume_state["dual"] is not None else None
        best, best_step = (resume_state["best"], resume_state["best_step"])
        best_moments, best_theta = (resume_state["best_moments"], resume_state["best_theta"])
        initial_moments, scale = (resume_state["initial_moments"], resume_state["scale"])
        history = [dict(row) for row in resume_state["history"] if row["step"] < start_step]
        snapshots = dict(resume_state["snapshots"])
        if save_assignment:
            best_parameters = [p.to(z.device) for p in resume_state["best_parameters"]]
            best_dual = resume_state["best_dual"]
    elapsed = resume_state["elapsed"] if resume_state is not None else 0.0
    started = time.perf_counter()

    def timestamp():
        if z.is_cuda:
            torch.cuda.synchronize(z.device)
        return time.perf_counter()

    for step in trange(start_step, steps + 1, desc="CE inner + CE outer"):
        _check_assignment_stop(stop)
        tick = timestamp()
        optimizer.zero_grad(set_to_none=True)
        if temperature_logits is not None:
            temperature = float(log_temperature.detach().exp())
            if not np.isfinite(temperature) or temperature <= 0:
                raise FloatingPointError("Nonfinite or nonpositive learned temperature")
            material = make_material(z, temperature_labels(temperature_logits, log_temperature))
        if assignment_input != "node":
            if centered_active:
                from src.mlp_source_centering import centered_nodes
                u, centering_mean, centering_diagnostic = centered_nodes(inputs, encoder_parameters)
            else:
                u = encode_nodes(inputs, encoder_parameters)
        balance = dict(balance_iterations=0, row_residual=np.nan, column_residual=np.nan)
        if node_weighting:
            node_weights, weight_kl = normalized_node_weights(node_logits)
            moments = WeightedLowRankMoments.apply(
                u, v, node_weights, assignment, material, mixing, chunk_size
            )
        elif mass_mode == "initial":
            from src.fixed_mass_assignment import FixedMassMoments
            logits = LowRankLogits.apply(u, v, assignment, mixing, chunk_size)
            moments, dual, diagnostic = FixedMassMoments.apply(
                logits, material, mass_target, chunk_size, balance_steps, balance_tol, dual)
            balance = dict(balance_iterations=int(diagnostic[0]), row_residual=float(diagnostic[1]),
                           column_residual=float(diagnostic[2]))
        elif mass_mode == "uniform":
            if assignment_rank is not None:
                logits = LowRankLogits.apply(u, v, assignment, mixing, chunk_size)
            if balance_backend == "cached":
                moments, dual, diagnostic = CachedBalancedMoments.apply(
                    logits,
                    material,
                    chunk_size,
                    balance_steps,
                    balance_tol,
                    dual,
                    balance_cg_steps,
                    balance_cg_rtol,
                )
            else:
                moments, dual, diagnostic = BalancedMoments.apply(
                    logits, material, chunk_size, balance_steps, balance_tol, dual
                )
            balance = dict(
                balance_iterations=int(diagnostic[0]),
                row_residual=float(diagnostic[1]),
                column_residual=float(diagnostic[2]),
            )
        elif assignment_rank is not None:
            operation = CachedLowRankMoments if cache_assignment else LowRankMoments
            moments = operation.apply(u, v, assignment, material, mixing, chunk_size)
        else:
            moments = AssignmentMoments.apply(logits, material, chunk_size)
        if not bool(torch.isfinite(moments).all()) or bool((moments[:, 0] <= 0).any()):
            raise FloatingPointError("Nonfinite moments or empty soft cell")
        centers, labels, mass = decode_moments(moments.detach(), z.shape[1])
        student_mass = torch.full_like(mass, 1 / len(mass)) if inner_loss_weighting == "uniform" else mass
        after_assignment = timestamp()
        theta_before, vector_before = (theta, vector)
        refresh = (
            solver_mode == "exact"
            or theta is None
            or step == steps
            or (step in checkpoints)
            or (step % tracking_refresh == 0)
        )
        head_fallback = False
        _check_assignment_stop(stop)
        if refresh:
            fitted = solve_student(
                centers,
                labels,
                student_mass,
                penalty,
                theta,
                inner_max_iter,
                inner_tol,
                cg_max_iter=cg_max_iter,
            )
        else:
            fitted = track_inner(
                centers,
                labels,
                student_mass,
                penalty,
                theta,
                tracking_inner_steps,
                tracking_cg_steps,
                inner_tol,
            )
            _check_assignment_stop(stop)
            head_fallback = fitted.pop("tracking_failed")
            if head_fallback:
                refresh = True
                fitted = solve_student(
                    centers,
                    labels,
                    student_mass,
                    penalty,
                    theta,
                    inner_max_iter,
                    inner_tol,
                    cg_max_iter=cg_max_iter,
                )
        _check_assignment_stop(stop)
        theta = fitted.pop("theta")
        after_inner = timestamp()
        value, outer_gradient = outer_value_gradient(outer_z, outer_q, theta, outer_chunk_size, full_features)
        _check_assignment_stop(stop)
        after_outer = timestamp()
        row = dict(
            step=step,
            J=value,
            **fitted,
            **balance,
            cg_iterations=0,
            cg_residual=np.nan,
            cg_relative_residual=np.nan,
            cg_converged=False,
            solver_mode=solver_mode,
            exact_refresh=refresh,
            head_fallback=head_fallback,
            J_exact=fitted["inner_converged"],
            implicit_fallback=False,
            implicit_warm_start=False,
            head_correction_relative=float((theta - theta_before).norm() / theta.norm().clamp_min(1e-30))
            if theta_before is not None
            else np.nan,
            implicit_correction_relative=np.nan,
            min_mass=float(mass.min()),
            max_mass=float(mass.max()),
            effective_cells=float(1 / mass.square().sum()),
        )
        row.update(
            assignment_seconds=after_assignment - tick,
            inner_seconds=after_inner - after_assignment,
            outer_seconds=after_outer - after_inner,
            implicit_seconds=0.0,
            backward_seconds=0.0,
        )
        if centered_active:
            row.update(centering_diagnostic)
        if temperature_logits is not None:
            row["temperature"] = temperature
        if "benchmark_lbfgs_seconds" in fitted:
            row["inner_benchmark_seconds"] = row["inner_seconds"]
            row["inner_seconds"] = fitted["benchmark_lbfgs_seconds"]
        objective = value
        if node_weighting:
            objective += node_weight_penalty * float(weight_kl.detach())
            row.update(
                weight_kl=float(weight_kl.detach()),
                weight_min=float(node_weights.min().detach()),
                weight_max=float(node_weights.max().detach()),
                weight_mean=float(node_weights.mean().detach()),
                node_ess_fraction=float((1 / node_weights.square().mean()).detach()),
                objective=objective,
            )
        failure = None
        if refresh and (not fitted["inner_converged"]) or not np.isfinite(value):
            failure = "Inner CE did not converge; increase inner_max_iter or inspect inner_tol"
        else:
            if step == 0:
                initial_moments = moments.detach().cpu()
                scale = max(value, 1e-12)
            if fitted["inner_converged"] and objective < best:
                best, best_step = (objective, step)
                best_moments, best_theta = (moments.detach().cpu(), theta.cpu())
                if save_assignment:
                    best_parameters = [parameter.detach().clone() for parameter in parameters]
                    best_dual = dual.clone() if dual is not None else None
            if step < steps:
                _check_assignment_stop(stop)
                implicit_start = timestamp()
                if refresh:
                    if implicit_solver is not None:
                        vector, diagnostic = implicit_solver(
                            augmented(centers),
                            labels,
                            student_mass,
                            theta,
                            penalty,
                            outer_gradient,
                            rtol=cg_rtol,
                            max_iter=cg_max_iter,
                        )
                    else:
                        row["implicit_warm_start"] = implicit_warm_start and vector is not None
                        vector, diagnostic = solve_head_system(
                            augmented(centers),
                            labels,
                            student_mass,
                            theta,
                            penalty,
                            outer_gradient,
                            rtol=cg_rtol,
                            max_iter=cg_max_iter,
                            initial=vector if implicit_warm_start else None,
                            cg_check_interval=cg_check_interval,
                        )
                else:
                    multiply, diagonal = hessian_operator(
                        augmented(centers), labels, student_mass, theta, penalty
                    )
                    vector, diagnostic = conjugate_gradient(
                        multiply,
                        outer_gradient,
                        diagonal,
                        rtol=cg_rtol,
                        max_iter=tracking_cg_steps,
                        initial=vector,
                        check_interval=cg_check_interval,
                    )
                    _check_assignment_stop(stop)
                    diagnostic.update(hessian_solver="tracking_pcg", hessian_reduced_dimension=0)
                    if not bool(torch.isfinite(vector).all()) or not np.isfinite(diagnostic["cg_residual"]):
                        row["implicit_fallback"] = True
                        vector, diagnostic = solve_head_system(
                            augmented(centers),
                            labels,
                            student_mass,
                            theta,
                            penalty,
                            outer_gradient,
                            rtol=cg_rtol,
                            max_iter=cg_max_iter,
                            cg_check_interval=cg_check_interval,
                        )
                _check_assignment_stop(stop)
                row.update(diagnostic)
                row["implicit_seconds"] = timestamp() - implicit_start
                if "benchmark_cold_seconds" in diagnostic:
                    row["implicit_benchmark_seconds"] = row["implicit_seconds"]
                    row["implicit_seconds"] = diagnostic["benchmark_cold_seconds"]
                if vector_before is not None:
                    row["implicit_correction_relative"] = float(
                        (vector - vector_before).norm() / vector.norm().clamp_min(1e-30)
                    )
                if (refresh or row["implicit_fallback"]) and (not diagnostic["cg_converged"]):
                    failure = f"Implicit Hessian solve did not converge: solver={diagnostic['hessian_solver']}, relative residual={diagnostic['cg_relative_residual']:.3g}, target={cg_rtol:.3g}"
        row.update(
            best_J=best,
            seconds=elapsed + time.perf_counter() - started,
            status="failed" if failure else "evaluated" if step == steps else "update",
        )
        history.append(row)
        if folder is not None and (step == steps or failure):
            pd.DataFrame(history).to_csv(folder / "optimization.csv", index=False)
        if failure:
            if folder is not None:
                (folder / "failure.json").write_text(json.dumps(dict(step=step, reason=failure), indent=2))
            raise RuntimeError(f"Step {step}: {failure}")
        if step in checkpoints:
            snapshot = dict(
                step=step,
                moments=moments.detach().cpu().clone(),
                theta=theta.cpu().clone(),
                teacher_ce=value,
                inner_grad_max=fitted["inner_grad_max"],
                J_exact=fitted["inner_converged"],
            )
            if temperature_logits is not None:
                snapshot["temperature"] = temperature
            if node_weighting:
                snapshot.update(
                    node_weight_logits=node_logits.detach().cpu().clone(),
                    weight_kl=row["weight_kl"],
                    node_ess_fraction=row["node_ess_fraction"],
                    weight_min=row["weight_min"],
                    weight_max=row["weight_max"],
                )
            if mass_mode == "initial":
                from src.mlp_initial_mass import attach
                snapshot["column_dual"] = dual.detach().cpu().clone()
                attach(snapshot, mass_target, initial_mass_context, parameters)
            if centered_active:
                from src.mlp_source_centering import attach as attach_centered
                attach_centered(snapshot, centering_context, parameters, u, centering_mean, centering_diagnostic)
            snapshots[step] = snapshot
            if folder is not None:
                checkpoint_dir = folder / "checkpoints"
                checkpoint_dir.mkdir(exist_ok=True)
                _save_checkpoint_atomic(snapshot, checkpoint_dir / f"step_{step:06d}.pt")
        if save_resume and (step in checkpoints or step == steps):
            state = cpu_state(
                dict(
                    config=resume_config,
                    step=step,
                    parameters=parameters,
                    optimizer=optimizer.state_dict(),
                    theta=theta,
                    dual=dual,
                    tracking_theta_before=theta_before,
                    tracking_vector_before=vector_before,
                    best=best,
                    best_step=best_step,
                    best_moments=best_moments,
                    best_theta=best_theta,
                    initial_moments=initial_moments,
                    scale=scale,
                    history=history,
                    snapshots=snapshots,
                    elapsed=row["seconds"],
                    best_parameters=best_parameters if save_assignment else None,
                    best_dual=best_dual if save_assignment else None,
                )
            )
            if mass_mode == "initial":
                from src.mlp_initial_mass import attach
                attach(state, mass_target, initial_mass_context)
            if centered_active:
                from src.mlp_source_centering import attach as attach_centered
                attach_centered(state, centering_context)
            if folder is not None:
                torch.save(state, folder / "resume.tmp.pt")
                (folder / "resume.tmp.pt").replace(folder / "resume.pt")
        _check_assignment_stop(stop)
        if step == steps:
            break
        _check_assignment_stop(stop)
        backward_start = timestamp()
        direction = implicit_moment_gradient(
            moments, z.shape[1], theta, vector, penalty, inner_loss_weighting
        )
        if not bool(torch.isfinite(direction).all()):
            raise FloatingPointError("Nonfinite implicit gradient")
        try:
            if node_weighting:
                torch.autograd.backward(
                    (moments, weight_kl),
                    (direction / scale, weight_kl.new_tensor(node_weight_penalty / scale)),
                )
            else:
                moments.backward(direction / scale)
        except RuntimeError as error:
            row.update(status="failed", backward_seconds=timestamp() - backward_start)
            if folder is not None:
                pd.DataFrame(history).to_csv(folder / "optimization.csv", index=False)
                (folder / "failure.json").write_text(json.dumps(dict(step=step, reason=str(error)), indent=2))
            raise
        if temperature_logits is not None:
            if log_temperature.grad is None or not bool(torch.isfinite(log_temperature.grad)):
                raise FloatingPointError("Missing or nonfinite temperature gradient")
            row["log_temperature_gradient"] = float(log_temperature.grad)
        if mass_mode == "initial" and any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in parameters):
            raise FloatingPointError("Missing or nonfinite constrained MLP gradient")
        if centered_active:
            if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in parameters):
                raise FloatingPointError("Missing or nonfinite centered MLP gradient")
            if bool(encoder_parameters[3].grad.ne(0).any()):
                raise FloatingPointError("Centered output-bias gradient is not exactly zero")
        _check_assignment_stop(stop)
        optimizer.step()
        if centered_active and bool(encoder_parameters[3].ne(0).any()):
            raise FloatingPointError("Centered output bias left its zero gauge")
        row["backward_seconds"] = timestamp() - backward_start
        row["seconds"] = elapsed + time.perf_counter() - started
        if folder is not None and step % log_every == 0:
            pd.DataFrame(history).to_csv(folder / "optimization.csv", index=False)
    result = dict(
        initial_moments=initial_moments,
        best_moments=best_moments,
        theta=best_theta,
        best_J=best,
        best_step=best_step,
        history=history,
        penalty=penalty,
        steps=steps,
        checkpoints=snapshots,
        mass_mode=mass_mode,
        assignment_rank=assignment_rank,
        factor_seed=factor_seed,
        assignment_input=assignment_input,
        assignment_encoder=assignment_encoder,
        solver_mode=solver_mode,
        feature_control=feature_control,
        implicit_warm_start=implicit_warm_start,
        inner_method=inner_method,
        inner_loss_weighting=inner_loss_weighting,
        node_weighting=node_weighting,
        node_weight_penalty=node_weight_penalty,
        assignment_parameters=sum((parameter.numel() for parameter in parameters)),
    )
    if folder is not None and save_assignment:
        if assignment_rank is None:
            torch.save(best_parameters[0].cpu(), folder / "best_assignment_logits.pt")
        elif assignment_input != "node":
            saved = dict(
                v=best_parameters[-1].cpu(),
                assignment=assignment.cpu(),
                mixing=mixing,
                rank=assignment_rank,
                assignment_input=assignment_input,
                factor_seed=factor_seed,
                step=best_step,
                assignment_encoder=assignment_encoder,
            )
            if assignment_encoder == "mlp":
                saved.update(
                    encoder_parameters=[p.cpu() for p in best_parameters[:-1]], encoder_hidden=encoder_hidden
                )
            else:
                saved["weight"] = best_parameters[0].cpu()
            torch.save(saved, folder / "best_assignment_encoder.pt")
        else:
            saved = dict(
                u=best_parameters[0].cpu(),
                v=best_parameters[1].cpu(),
                assignment=assignment.cpu(),
                mixing=mixing,
                rank=assignment_rank,
                factor_seed=factor_seed,
                step=best_step,
            )
            if node_weighting:
                saved["node_weight_logits"] = best_parameters[2].cpu()
            if temperature_logits is not None:
                saved["temperature"] = float(best_parameters[-1].exp())
            torch.save(saved, folder / "best_assignment_factors.pt")
        if best_dual is not None:
            torch.save(best_dual.cpu(), folder / "best_assignment_column_dual.pt")
    return result


@torch.no_grad()
def classification(z, y, mask, theta):
    logits = augmented(z[mask]) @ theta.T
    return 100 * float((logits.argmax(1) == y[mask]).double().mean()), float(F.cross_entropy(logits, y[mask]))


def select_checkpoint(table):
    candidates = table[table.checkpoint_step.notna()]
    return candidates.sort_values(["gcn_val", "checkpoint_step"], ascending=[False, True]).iloc[[0]]


def load_ce_snapshots(directory):
    snapshots = {}
    for path in sorted((Path(directory) / "checkpoints").glob("step_*.pt")):
        snapshot = torch.load(path, map_location="cpu", weights_only=False)
        step = int(path.stem.split("_")[-1])
        if snapshot["step"] != step or not bool(torch.isfinite(snapshot["moments"]).all()):
            raise ValueError(f"Invalid saved checkpoint: {path}")
        snapshots[step] = snapshot
    if 0 not in snapshots:
        raise ValueError("Recovery requires saved checkpoint zero")
    return snapshots
