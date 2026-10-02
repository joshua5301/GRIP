import math
from time import perf_counter

import torch

from src.io import array_digest


def moment_objective(statistics, energy, original_moment, B, epsilon=1e-12):
    mass, features, labels = statistics
    centers = features / mass[:, None]
    variance = (energy - (features * centers).sum()).clamp_min(0)
    moment = original_moment - centers.T @ labels
    norm = moment.norm()
    exact = B * B / 4 * variance + 2 * B * norm
    smooth = B * B / 4 * variance + 2 * B * ((moment.square().sum() + epsilon**2).sqrt() - epsilon)
    return smooth, exact, variance, norm


def low_rank_partition(
    H,
    Q,
    m,
    B,
    seed_partition,
    seed=0,
    max_sweeps=100,
    block_size=1024,
    rank=8,
    steps=1000,
    lr=0.01,
    mixing=0.05,
    initialization="historical",
):
    if initialization not in ("historical", "random"):
        raise ValueError("Use historical or random initialization")
    if not 1 <= m <= len(H) or not math.isfinite(B) or B <= 0:
        raise ValueError("Require 1 <= m <= N and finite B > 0")
    if (
        (initialization == "historical" and not 0 < mixing < 1)
        or min(rank, block_size, steps) < 1
        or not math.isfinite(lr)
        or lr <= 0
    ):
        raise ValueError("Invalid low-rank settings")
    if H.is_cuda:
        torch.cuda.synchronize(H.device)
    started = perf_counter()
    X, Q = H.detach().double(), Q.detach().double()
    offset = X.mean(0)
    X = X - offset
    scale = X.square().sum(1).mean().sqrt()
    scale = scale if float(scale) > 0 else X.new_tensor(1.0)
    X = X / scale
    n, d = X.shape
    classes = Q.shape[1]
    generator = torch.Generator(device=X.device).manual_seed(seed)
    assignment, initial_digest = None, None
    if initialization == "historical":
        assignment = seed_partition(X, Q, m, B, generator, block_size)
        initial_digest = array_digest(assignment.cpu().numpy())
    energy, original = X.square().sum(1).mean(), X.T @ Q / n
    bias = math.log((1 - mixing + mixing / m) / (mixing / m)) if assignment is not None else None
    U = torch.nn.Parameter(torch.randn(n, rank, generator=generator, device=X.device, dtype=X.dtype))
    if initialization == "random":
        V = torch.nn.Parameter(torch.randn(m, rank, generator=generator, device=X.device, dtype=X.dtype))
    else:
        U.data.div_(rank**0.5)
        V = torch.nn.Parameter(X.new_zeros(m, rank))
    optimizer = torch.optim.Adam([U, V], lr=lr)

    def probabilities(start, end):
        logits = U[start:end] @ V.T
        if assignment is None:
            logits = logits / rank**0.5
        else:
            logits = logits.scatter_add(
                1, assignment[start:end, None], logits.new_full((end - start, 1), bias)
            )
        return logits.softmax(1)

    def chunk_statistics(start, end):
        p = probabilities(start, end) / n
        return p.sum(0), p.T @ X[start:end], p.T @ Q[start:end]

    history, best, best_value, best_step = [], None, math.inf, 0
    for step in range(steps + 1):
        with torch.no_grad():
            statistics = [X.new_zeros(m), X.new_zeros(m, d), X.new_zeros(m, classes)]
            for start in range(0, n, block_size):
                for total, chunk in zip(statistics, chunk_statistics(start, min(start + block_size, n))):
                    total.add_(chunk)
        if not all(bool(torch.isfinite(s).all()) for s in statistics) or bool((statistics[0] <= 0).any()):
            raise FloatingPointError(f"Invalid cell statistics at step {step}")
        statistics = [s.requires_grad_() for s in statistics]
        smooth, exact, variance, moment = moment_objective(statistics, energy, original, B)
        value = float(exact.detach())
        if not math.isfinite(value):
            raise FloatingPointError(f"Nonfinite objective at step {step}")
        history.append(value)
        if value < best_value:
            best_value, best_step = value, step
            best = [s.detach().clone() for s in statistics]
        if step == steps:
            break
        derivatives = torch.autograd.grad(smooth, statistics)
        optimizer.zero_grad(set_to_none=True)
        for start in range(0, n, block_size):
            chunks = chunk_statistics(start, min(start + block_size, n))
            sum((s * g).sum() for s, g in zip(chunks, derivatives)).backward()
        if not all(bool(torch.isfinite(p.grad).all()) for p in (U, V)):
            raise FloatingPointError(f"Nonfinite gradient at step {step}")
        optimizer.step()
    mass, features, labels = best
    _, exact, variance, moment = moment_objective(best, energy, original, B)
    if H.is_cuda:
        torch.cuda.synchronize(H.device)
    return dict(
        x=(features / mass[:, None] * scale + offset).float().cpu(),
        y=(labels / mass[:, None]).float().cpu(),
        counts=(mass * n).cpu(),
        J=float(exact),
        V=float(variance),
        moment_error=float(moment),
        history=history,
        sweeps=steps,
        converged=False,
        status="fixed_step_budget",
        best_step=best_step,
        rank=rank,
        mixing=mixing if initialization == "historical" else None,
        initialization=initialization,
        lr=lr,
        B=B,
        seed=seed,
        initial_assignment_digest=initial_digest,
        seconds=perf_counter() - started,
    )
