from copy import deepcopy
from time import perf_counter

import torch

from src.variance_moment_low_rank import moment_objective


def compare_optimizer(
    H,
    Q,
    cells,
    weight,
    rank=32,
    seed=0,
    lr=0.01,
    steps=1000,
    method="joint",
    block_steps=50,
    window=10,
    tolerance=1e-7,
):
    if method not in ("joint", "alternating") or min(cells, rank, steps, block_steps, window) < 1:
        raise ValueError("Invalid optimizer or budget")
    x, q = H.detach().double(), Q.detach().double()
    offset = x.mean(0)
    x = x - offset
    scale = x.square().sum(1).mean().sqrt().clamp_min(torch.finfo(x.dtype).tiny)
    x = x / scale
    n = len(x)
    energy, original = x.square().sum(1).mean(), x.T @ q / n
    generator = torch.Generator(device=x.device).manual_seed(seed)
    u = torch.nn.Parameter(torch.randn(n, rank, generator=generator, device=x.device, dtype=x.dtype))
    v = torch.nn.Parameter(torch.randn(cells, rank, generator=generator, device=x.device, dtype=x.dtype))
    optimizers = (
        [torch.optim.Adam([u, v], lr=lr)]
        if method == "joint"
        else [torch.optim.Adam([u], lr=lr), torch.optim.Adam([v], lr=lr)]
    )

    def evaluate():
        p = (u @ v.T / rank**0.5).softmax(1) / n
        statistics = p.sum(0), p.T @ x, p.T @ q
        smooth, exact, variance, moment = moment_objective(
            statistics, energy, original, None, moment_weight=weight
        )
        return smooth, exact, variance, moment, statistics

    def elapsed():
        if x.is_cuda:
            torch.cuda.synchronize(x.device)
        return perf_counter() - started

    if x.is_cuda:
        torch.cuda.synchronize(x.device)
    started = perf_counter()
    with torch.no_grad():
        _, exact, variance, moment, statistics = evaluate()
    best = float(exact)
    best_stats = [s.clone() for s in statistics]
    best_step = 0
    rows = [dict(step=0, seconds=elapsed(), J=best, best_J=best, block=0, active="initial")]
    blocks, step, cycle = [], 0, 0
    while step < steps:
        active = cycle % 2
        parameters = [u, v] if method == "joint" else [[u], [v]][active]
        u.requires_grad_(method == "joint" or active == 0)
        v.requires_grad_(method == "joint" or active == 1)
        optimizer = optimizers[0 if method == "joint" else active]
        block_initial = float(exact)
        block_best = block_initial
        snapshot = [p.detach().clone() for p in parameters]
        state = deepcopy(optimizer.state_dict()) if method == "alternating" else None
        recent = [block_best]
        for _ in range(min(block_steps if method == "alternating" else steps, steps - step)):
            optimizer.zero_grad(set_to_none=True)
            smooth, _, _, _, _ = evaluate()
            smooth.backward()
            if not bool(torch.stack([torch.isfinite(p.grad).all() for p in parameters]).all()):
                raise FloatingPointError("Nonfinite optimizer gradient")
            optimizer.step()
            step += 1
            with torch.no_grad():
                _, exact, variance, moment, statistics = evaluate()
            value = float(exact)
            if not bool(torch.isfinite(exact)):
                raise FloatingPointError("Nonfinite objective")
            if value < best:
                best, best_step = value, step
                best_stats = [s.clone() for s in statistics]
            if method == "alternating" and value < block_best:
                block_best = value
                snapshot = [p.detach().clone() for p in parameters]
                state = deepcopy(optimizer.state_dict())
            rows.append(
                dict(
                    step=step,
                    seconds=elapsed(),
                    J=value,
                    best_J=best,
                    block=cycle,
                    active="UV" if method == "joint" else ("U" if active == 0 else "V"),
                )
            )
            recent.append(min(recent[-1], value))
            if method == "alternating" and len(recent) > window:
                if recent[-window - 1] - recent[-1] <= tolerance * max(abs(recent[-window - 1]), 1e-12):
                    break
        if method == "alternating":
            with torch.no_grad():
                for p, saved in zip(parameters, snapshot):
                    p.copy_(saved)
            optimizer.load_state_dict(state)
            with torch.no_grad():
                _, exact, _, _, _ = evaluate()
            blocks.append(
                dict(
                    block=cycle,
                    step=step,
                    initial_J=block_initial,
                    accepted_J=float(exact),
                    seconds=elapsed(),
                )
            )
            rows.append(
                dict(
                    step=step, seconds=elapsed(), J=float(exact), best_J=best, block=cycle, active="accepted"
                )
            )
        cycle += 1
    mass, sx, sq = best_stats
    _, exact, variance, moment = moment_objective(best_stats, energy, original, None, moment_weight=weight)
    return dict(
        method=method,
        seed=seed,
        rank=rank,
        weight=weight,
        best_step=best_step,
        J_initial=rows[0]["J"],
        J_final=float(exact),
        variance=float(variance),
        moment=float(moment),
        seconds=elapsed(),
        history=rows,
        blocks=blocks,
        x=(sx / mass[:, None] * scale + offset).float().cpu(),
        y=(sq / mass[:, None]).float().cpu(),
        counts=(mass * n).cpu(),
    )
