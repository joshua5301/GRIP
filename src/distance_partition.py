"""Diagonal-distance assignments with convex CE and exact implicit updates.

Only a positive diagonal distance metric is learned. Representatives are always
P-weighted means of the provided original normalized features and teacher targets.
This dense node-by-cell implementation is intended for small citation graphs.
"""

import json
import math
import time
from numbers import Integral
from pathlib import Path

import torch

from src.io import _fingerprint, array_digest, cpu_state, save_json, save_state
from src.moments import augmented, decode_moments, initial_logits, make_material
from src.soft_ce_partition import (
    head_gradient,
    outer_value_gradient,
    solve_head_system,
    solve_inner_newton_first,
)


def positive_metric_weights(log_weights):
    """Return strictly positive diagonal weights of mean one, stably."""
    shifted = log_weights - log_weights.max()
    unnormalized = shifted.clamp_min(math.log(torch.finfo(log_weights.dtype).tiny)).exp()
    return unnormalized / unnormalized.mean()


def pairwise_metric_distance(inputs, anchors, weights):
    """Squared diagonal Mahalanobis distances without a node×cell×feature tensor."""
    node_norm = (inputs.square() * weights).sum(1, keepdim=True)
    anchor_norm = (anchors.square() * weights).sum(1)[None]
    return (node_norm + anchor_norm - 2 * (inputs * weights) @ anchors.T).clamp_min(0)


def distance_probability(inputs, anchors, initial_distance, baseline_logits, log_weights, strength=3.0):
    weights = positive_metric_weights(log_weights)
    distance = pairwise_metric_distance(inputs, anchors, weights)
    return (baseline_logits + strength * (initial_distance - distance)).softmax(1)


def distance_implicit_gradient(moments, dimension, theta, vector, penalty, log_weights,
                               inner_loss_weighting="mass"):
    """Differentiate the stationary CE condition through P-derived moments."""
    if inner_loss_weighting not in ("mass", "uniform"):
        raise ValueError("Unknown inner CE weighting")
    centers, labels, mass = decode_moments(moments, dimension)
    if inner_loss_weighting == "uniform":
        mass = torch.full_like(mass, 1 / len(mass))
    gradient = head_gradient(augmented(centers), labels, mass, theta.detach(), penalty)
    return torch.autograd.grad(-(gradient * vector.detach()).sum(), log_weights)[0]


def _prepare(z, q, assignment, metric_inputs, mixing):
    if (
        z.ndim != 2
        or q.ndim != 2
        or len(z) != len(q)
        or min(*z.shape, q.shape[1]) < 1
        or not z.is_floating_point()
        or not q.is_floating_point()
        or not bool(torch.isfinite(z).all())
        or not bool(torch.isfinite(q).all())
        or bool((q < 0).any())
        or not torch.allclose(q.sum(1), torch.ones_like(q[:, 0]), atol=1e-6, rtol=1e-5)
    ):
        raise ValueError("Require finite node-aligned features and teacher probabilities")
    if assignment.ndim != 1 or len(assignment) != len(z) or assignment.dtype != torch.long:
        raise ValueError("Initial assignment must be a node-aligned long tensor")
    if bool((assignment < 0).any()) or assignment.unique().tolist() != list(range(int(assignment.max()) + 1)):
        raise ValueError("Initial assignment must contain contiguous nonempty cells")
    z, q, assignment = z.detach().double(), q.detach().to(z.device).double(), assignment.to(z.device)
    inputs = z if metric_inputs is None else metric_inputs.detach().to(device=z.device, dtype=torch.double)
    if inputs.ndim != 2 or len(inputs) != len(z) or inputs.shape[1] < 1 or not bool(torch.isfinite(inputs).all()):
        raise ValueError("Metric inputs must be finite nonempty node-aligned features")
    cells = int(assignment.max()) + 1
    baseline = initial_logits(assignment, cells, mixing, dtype=torch.double)
    probability = baseline.softmax(1)
    anchors = probability.T @ inputs / probability.sum(0)[:, None]
    initial_distance = pairwise_metric_distance(inputs, anchors, inputs.new_ones(inputs.shape[1]))
    material = make_material(z, q)
    initial_moments = probability.T @ material / len(z)
    return z, q, inputs, baseline, anchors, initial_distance, material, initial_moments


def optimize_distance_ce(
    z,
    q,
    assignment,
    metric_inputs=None,
    penalty=1e-4,
    steps=100,
    lr=0.03,
    strength=3.0,
    mixing=0.05,
    inner_loss_weighting="mass",
    inner_max_iter=2000,
    inner_tol=1e-7,
    cg_max_iter=512,
    cg_rtol=1e-6,
    outer_chunk_size=4096,
    folder=None,
    checkpoint_steps=(),
    resume_state=None,
    save_resume=True,
    log_every=10,
    stop=lambda: False,
    progress=lambda row: None,
):
    """Learn a distance metric while keeping synthetic features/labels tied to P.

    z is the original normalized H-space feature matrix and q contains only
    teacher probabilities. Optional metric_inputs may concatenate z and centered
    teacher probabilities, without changing the space of synthetic representatives.
    Resume is explicit via resume_state; target steps can be extended while the
    data and all optimization settings must match the saved configuration.
    Checkpoints use the existing normalized-z ``moments`` representation.
    """
    positive = (penalty, lr, strength, inner_tol, cg_rtol)
    counts = (inner_max_iter, cg_max_iter, outer_chunk_size, log_every)
    if (
        isinstance(steps, bool)
        or not isinstance(steps, Integral)
        or steps < 0
        or any(not math.isfinite(v) or v <= 0 for v in positive)
        or any(isinstance(v, bool) or not isinstance(v, Integral) or v < 1 for v in counts)
        or inner_loss_weighting not in ("mass", "uniform")
    ):
        raise ValueError("Require nonnegative integer steps, positive solver settings and mass/uniform CE")
    checkpoints = {0, int(steps), *checkpoint_steps}
    if any(isinstance(s, bool) or not isinstance(s, Integral) or not 0 <= s <= steps for s in checkpoints):
        raise ValueError("Checkpoint steps must be integers within the step budget")
    z, q, inputs, baseline, anchors, initial_distance, material, initial_moments = _prepare(
        z, q, assignment, metric_inputs, mixing
    )
    data_digest = array_digest(*(t.cpu().numpy() for t in (z, q, inputs, baseline, anchors)))
    config = dict(
        version=1, method="distance", penalty=penalty, lr=lr, strength=strength, mixing=mixing,
        inner_loss_weighting=inner_loss_weighting, inner_max_iter=inner_max_iter, inner_tol=inner_tol,
        cg_max_iter=cg_max_iter, cg_rtol=cg_rtol, outer_chunk_size=outer_chunk_size,
        data_digest=data_digest, nodes=len(z), cells=len(anchors), dimension=z.shape[1],
        metric_dimension=inputs.shape[1], objective="uniform_full_node_teacher_ce",
    )
    fingerprint = _fingerprint(config)
    folder = Path(folder) if folder is not None else None
    if folder is not None:
        folder.mkdir(parents=True, exist_ok=True)
        config_path = folder / "config.json"
        if config_path.exists() and json.loads(config_path.read_text()) != config:
            raise ValueError("Distance result folder belongs to a different configuration or dataset")
        save_json(config, config_path)
    log_weights = inputs.new_zeros(inputs.shape[1]).requires_grad_()
    optimizer = torch.optim.Adam([log_weights], lr=lr, eps=1e-12, foreach=False)
    full_features = augmented(z)
    start, theta, vector, history, snapshots = 0, None, None, [], {}
    best, best_step, best_moments, best_theta, best_log_weights = math.inf, 0, None, None, None
    if resume_state is not None:
        if resume_state["config"] != config or resume_state["fingerprint"] != fingerprint:
            raise ValueError("Resume state differs in data, initial assignment, metric or solver configuration")
        start = int(resume_state["step"])
        if not 0 <= start <= steps:
            raise ValueError("Cannot resume to an earlier or negative step")
        with torch.no_grad():
            log_weights.copy_(resume_state["log_weights"].to(log_weights))
        optimizer.load_state_dict(resume_state["optimizer"])
        theta = resume_state["theta"]
        theta = theta.to(z) if theta is not None else None
        vector = resume_state["vector"]
        vector = vector.to(z) if vector is not None else None
        history = [dict(row) for row in resume_state["history"]]
        snapshots = dict(resume_state["snapshots"])
        best, best_step = resume_state["best_J"], resume_state["best_step"]
        best_moments, best_theta, best_log_weights = (
            resume_state[key] for key in ("best_moments", "best_theta", "best_log_weights")
        )

    def state(step):
        return cpu_state(dict(
            config=config, fingerprint=fingerprint, step=step, log_weights=log_weights,
            optimizer=optimizer.state_dict(), theta=theta, vector=vector, history=history,
            snapshots=snapshots, best_J=best, best_step=best_step, best_moments=best_moments,
            best_theta=best_theta, best_log_weights=best_log_weights,
        ))

    def persist(step):
        saved = state(step)
        if folder is not None and save_resume:
            save_state(saved, folder / "resume.pt")
        if folder is not None:
            save_json(history, folder / "optimization.json")
        return saved

    for step in range(start, steps + 1):
        if stop():
            persist(step)
            raise InterruptedError("Distance condensation stopped with a resumable state")
        started = time.monotonic()
        probability = distance_probability(inputs, anchors, initial_distance, baseline, log_weights, strength)
        moments = probability.T @ material / len(z)
        centers, labels, mass = decode_moments(moments.detach(), z.shape[1])
        student_mass = mass if inner_loss_weighting == "mass" else torch.full_like(mass, 1 / len(mass))
        fitted = solve_inner_newton_first(
            centers, labels, student_mass, penalty, initial=theta, max_iter=inner_max_iter,
            grad_tol=inner_tol, cg_max_iter=cg_max_iter,
        )
        theta = fitted["theta"]
        if not fitted["inner_converged"]:
            persist(step)
            raise RuntimeError(f"Step {step}: inner CE did not converge; refusing an inexact distance update")
        value, rhs = outer_value_gradient(z, q, theta, outer_chunk_size, full_features)
        if not math.isfinite(value):
            persist(step)
            raise FloatingPointError("Nonfinite outer teacher CE")
        row = dict(step=step, teacher_ce=value, J=value, inner_grad_max=fitted["inner_grad_max"],
                   inner_converged=True, metric_min=float(positive_metric_weights(log_weights).min().detach()),
                   metric_max=float(positive_metric_weights(log_weights).max().detach()))
        if value < best:
            best, best_step = value, step
            best_moments, best_theta, best_log_weights = cpu_state((moments, theta, log_weights))
        if step < steps:
            vector, diagnostic = solve_head_system(
                augmented(centers), labels, student_mass, theta, penalty, rhs,
                rtol=cg_rtol, max_iter=cg_max_iter, initial=vector,
            )
            row.update(diagnostic)
            if not diagnostic["cg_converged"]:
                persist(step)
                raise RuntimeError(f"Step {step}: implicit Hessian solve did not converge")
        history = [r for r in history if r["step"] < step] + [row]
        if step in checkpoints:
            snapshot = cpu_state(dict(
                step=step, moments=moments, theta=theta, teacher_ce=value, J_exact=True,
                inner_grad_max=fitted["inner_grad_max"], probability=probability,
                metric_weights=positive_metric_weights(log_weights), log_weights=log_weights,
                anchors=anchors, fingerprint=fingerprint,
            ))
            snapshots[step] = snapshot
            if folder is not None:
                checkpoint_dir = folder / "checkpoints"
                checkpoint_dir.mkdir(exist_ok=True)
                save_state(snapshot, checkpoint_dir / f"step_{step:06d}.pt")
        if step in checkpoints or step % log_every == 0:
            saved = persist(step)
        if step == steps:
            row["seconds"] = time.monotonic() - started
            progress(dict(row))
            saved = persist(step)
            break
        derivative = distance_implicit_gradient(
            moments, z.shape[1], theta, vector, penalty, log_weights, inner_loss_weighting
        )
        if not bool(torch.isfinite(derivative).all()):
            persist(step)
            raise FloatingPointError("Nonfinite distance metric gradient")
        optimizer.zero_grad(set_to_none=True)
        log_weights.grad = derivative
        optimizer.step()
        row.update(metric_gradient_norm=float(derivative.norm()), seconds=time.monotonic() - started)
        progress(dict(row))
    return dict(
        config=config, fingerprint=fingerprint, initial_moments=initial_moments.detach().cpu(),
        best_moments=best_moments, theta=best_theta, best_J=best, best_step=best_step,
        best_metric_weights=positive_metric_weights(best_log_weights), metric_weights=positive_metric_weights(
            log_weights.detach()).cpu(), log_weights=log_weights.detach().cpu(), history=history,
        checkpoints=snapshots, resume_state=saved, steps=steps,
    )
