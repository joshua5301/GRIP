"""Shared teacher Nyström map and exact implicit CE condensation.

Representatives are averages in SGC input space; the nonlinear map is applied
AFTER averaging. Original-node feature blocks live in a CPU memory map.
"""
import time
from pathlib import Path

import numpy as np
import torch

from src.io import save_json, save_state
from src.low_rank_assignment import LowRankMoments, initialize_factors
from src.moments import augmented, decode_moments, make_material
from src.soft_ce_partition import head_gradient, solve_head_system, solve_inner_newton_first
from src.teacher import get_kernel_values


class NystromMap:
    def __init__(self, anchors, mapping, kernel="relu"):
        self.anchors, self.mapping, self.kernel = anchors, mapping, kernel

    @classmethod
    def fit(cls, h, basis=3000, seed=0, kernel="relu"):
        generator = torch.Generator(device=h.device).manual_seed(seed)
        indices = torch.randperm(len(h), generator=generator, device=h.device)[:basis] if basis < len(h) else None
        anchors = h[indices].double() if indices is not None else h.double()
        gram = get_kernel_values(anchors, anchors, kernel)
        gram = (gram + gram.T) / 2
        eye = torch.eye(len(anchors), dtype=gram.dtype, device=gram.device)
        chol = torch.linalg.cholesky(gram + 1e-8 * gram.diagonal().mean() * eye)
        return cls(anchors, torch.linalg.solve_triangular(chol, eye, upper=False).T, kernel)

    def __call__(self, h):
        return get_kernel_values(h.double(), self.anchors, self.kernel) @ self.mapping


@torch.no_grad()
def cache_features(h, feature_map, path, chunk=2048, stop=lambda: False):
    path = Path(path)
    if path.exists():
        phi = np.load(path, mmap_mode="r")
        if phi.shape != (len(h), len(feature_map.anchors)) or phi.dtype != np.float64:
            raise ValueError("Incompatible feature cache")
        return phi
    temporary = path.with_suffix(".partial.npy")
    phi = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float64,
                                  shape=(len(h), len(feature_map.anchors)))
    for start in range(0, len(h), chunk):
        if stop():
            raise InterruptedError("Feature preparation interrupted")
        phi[start:start + chunk] = feature_map(h[start:start + chunk]).cpu().numpy()
    phi.flush()
    del phi
    temporary.replace(path)
    return np.load(path, mmap_mode="r")


def blocks(phi, device, chunk=2048):
    for start in range(0, len(phi), chunk):
        # Copy avoids writing through a read-only NumPy mapping.
        yield start, torch.from_numpy(np.array(phi[start:start + chunk])).to(device)


@torch.no_grad()
def outer_gradient(phi, q, theta, chunk=2048):
    value, gradient = theta.new_zeros(()), torch.zeros_like(theta)
    for start, block in blocks(phi, theta.device, chunk):
        x = augmented(block)
        target = q[start:start + len(block)]
        lp = (x @ theta.T).log_softmax(1)
        value -= (target * lp).sum() / len(phi)
        error = target.sum(1, keepdim=True) * lp.exp() - target
        gradient += error.T @ x / len(phi)
    return float(value), gradient


def fit_streaming_teacher(phi, labels, mask, gamma=0.01, chunk=2048,
                          max_iter=1000, stop=lambda: False):
    """Same no-bias logistic objective as teacher.fit_logistic, streamed."""
    n = int(mask.sum())
    classes = int(labels.max()) + 1
    weight = torch.zeros(phi.shape[1], classes, dtype=torch.double,
                         device=labels.device, requires_grad=True)
    optimizer = torch.optim.LBFGS([weight], max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        if stop():
            raise InterruptedError("Teacher fitting interrupted")
        value = weight.new_zeros(())
        gradient = torch.zeros_like(weight)
        with torch.no_grad():
            for start, block in blocks(phi, weight.device, chunk):
                chosen = mask[start:start + len(block)]
                x = block[chosen]
                if not len(x):
                    continue
                y = labels[start:start + len(block)][chosen]
                lp = (x @ weight).log_softmax(1)
                value -= lp[torch.arange(len(y), device=y.device), y].sum() / n
                error = lp.exp()
                error[torch.arange(len(y), device=y.device), y] -= 1
                gradient += x.T @ error / n
            value += gamma / n * weight.square().sum()
            gradient += 2 * gamma / n * weight
        weight.grad = gradient
        return value

    optimizer.step(closure)
    closure()
    if not torch.isfinite(weight).all() or float(weight.grad.abs().max()) > 1e-5:
        raise RuntimeError("Teacher logistic fit did not converge")
    with torch.no_grad():
        logits = torch.cat([x @ weight for _, x in blocks(phi, weight.device, chunk)])
    return logits, weight.detach()


def moment_gradient(moments, dimension, feature_map, theta, vector, penalty):
    variable = moments.detach().requires_grad_()
    centers, labels, mass = decode_moments(variable, dimension)
    gradient = head_gradient(augmented(feature_map(centers)), labels, mass,
                             theta.detach(), penalty)
    return torch.autograd.grad(-(gradient * vector.detach()).sum(), variable)[0]


def optimize(h, q, assignment, feature_map, phi, folder, steps, penalty=1e-4,
             lr=0.01, rank=16, seed=0, chunk=2048, stop=lambda: False,
             progress=lambda row: None, checkpoint_every=25):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    config = dict(steps_schema=1, penalty=penalty, lr=lr, rank=rank, seed=seed,
                  cells=int(assignment.max()) + 1, chunk=chunk)
    u, v = initialize_factors(assignment, config["cells"], rank, seed)
    optimizer = torch.optim.Adam([u, v], lr=lr)
    material = make_material(h.double(), q.double())
    start, theta, vector, history = 0, None, None, []
    resume = folder / "resume.pt"
    if resume.exists():
        saved = torch.load(resume, map_location=h.device, weights_only=False)
        if saved["config"] != config:
            raise ValueError("Resume configuration differs")
        with torch.no_grad():
            u.copy_(saved["u"])
            v.copy_(saved["v"])
        optimizer.load_state_dict(saved["optimizer"])
        start, theta, vector, history = (saved[k] for k in ("step", "theta", "vector", "history"))
    if start > steps:
        raise ValueError("Cannot resume to an earlier step")

    def persist(step):
        save_state(dict(config=config, step=step, u=u, v=v, optimizer=optimizer.state_dict(),
                        theta=theta, vector=vector, history=history), resume)

    for step in range(start, steps + 1):
        if stop():
            persist(step)
            raise InterruptedError("Condensation stopped with resumable state")
        started = time.monotonic()
        moments = LowRankMoments.apply(u, v, assignment, material, 0.05, chunk)
        centers, labels, mass = decode_moments(moments.detach(), h.shape[1])
        mapped = feature_map(centers).detach()
        inner = solve_inner_newton_first(mapped, labels, mass, penalty, initial=theta)
        theta = inner["theta"]
        if not inner["inner_converged"]:
            persist(step)
            raise RuntimeError("Inner CE fit did not converge; refusing inexact update")
        value, rhs = outer_gradient(phi, q, theta, chunk)
        row = dict(step=step, outer_ce=value, inner_grad=inner["inner_grad_max"])
        history = [r for r in history if r["step"] < step] + [row]
        if step % checkpoint_every == 0 or step == steps:
            save_state(dict(step=step, moments=moments, theta=theta, outer_ce=value),
                       folder / f"step_{step:06d}.pt")
            persist(step)
            save_json(history, folder / "history.json")
        if step == steps:
            progress(dict(**row, seconds=time.monotonic() - started))
            return folder / f"step_{step:06d}.pt"
        vector, diagnostic = solve_head_system(augmented(mapped), labels, mass, theta,
                                               penalty, rhs, initial=vector)
        if not diagnostic["cg_converged"]:
            persist(step)
            raise RuntimeError("Implicit Hessian solve did not converge")
        derivative = moment_gradient(moments, h.shape[1], feature_map, theta, vector, penalty)
        optimizer.zero_grad(set_to_none=True)
        moments.backward(derivative)
        if not all(torch.isfinite(p.grad).all() for p in (u, v)):
            persist(step)
            raise RuntimeError("Nonfinite assignment gradient")
        optimizer.step()
        progress(dict(**row, seconds=time.monotonic() - started))
    raise AssertionError("Unreachable")
