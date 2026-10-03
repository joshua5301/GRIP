import math
from time import perf_counter

import torch

from src.io import array_digest


def statistics(x, q, assignment, cells, energy, original):
    counts = torch.bincount(assignment, minlength=cells).to(x)
    if bool((counts == 0).any()):
        return None
    centers = x.new_zeros(cells, x.shape[1]).index_add_(0, assignment, x) / counts[:, None]
    labels = q.new_zeros(cells, q.shape[1]).index_add_(0, assignment, q) / counts[:, None]
    variance = energy - (counts * centers.square().sum(1)).sum() / len(x)
    moment = original - centers.T @ (counts[:, None] * labels) / len(x)
    return counts, centers, labels, variance, moment


def assignment_cost(x, q, centers, labels, moment, weight):
    direction = moment / moment.norm().clamp_min(1e-30)
    cg = centers @ direction
    return (
        centers.square().sum(1)
        - 2 * x @ centers.T
        + weight * (-x @ (direction @ labels.T) - q @ cg.T + (cg * labels).sum(1))
    )


def move_deltas(x, q, source, state, n, weight):
    counts, centers, labels, _, moment = state
    if counts[source] <= 1:
        return counts.new_full(counts.shape, torch.inf)
    u, v = x - centers, q - labels
    remove = counts[source] / (n * (counts[source] - 1))
    add = counts / (n * (counts + 1))
    u2, v2 = u.square().sum(1), v.square().sum(1)
    dot = ((u @ moment) * v).sum(1)
    norm2 = (
        moment.square().sum()
        - 2 * remove * dot[source]
        + remove.square() * u2[source] * v2[source]
        + 2 * add * dot
        + add.square() * u2 * v2
        - 2 * remove * add * (u @ u[source]) * (v @ v[source])
    )
    delta = add * u2 - remove * u2[source] + weight * (norm2.clamp_min(0).sqrt() - moment.norm())
    delta[source] = torch.inf
    return delta


def initialize(x, cells, seed, block_size, iterations=20, return_info=False):
    if iterations < 1:
        raise ValueError("Initialization iterations must be positive")
    generator = torch.Generator(device=x.device).manual_seed(seed)
    index = int(torch.randint(len(x), (1,), generator=generator, device=x.device))
    selected = torch.zeros(len(x), device=x.device, dtype=torch.bool)
    nearest = x.new_full((len(x),), torch.inf)
    indices = []
    for _ in range(cells):
        indices.append(index)
        selected[index] = True
        nearest = torch.minimum(nearest, (x - x[index]).square().sum(1))
        nearest[selected] = 0
        if len(indices) < cells:
            index = (
                int(torch.multinomial(nearest, 1, generator=generator))
                if float(nearest.sum()) > 0
                else int(torch.where(~selected)[0][0])
            )
    centers = x[indices].clone()
    assignment = torch.full((len(x),), -1, device=x.device, dtype=torch.long)
    converged = False
    for step in range(iterations):
        previous = assignment.clone()
        distances = x.new_empty(len(x))
        for start in range(0, len(x), block_size):
            chunk = x[start : start + block_size]
            cost = (
                chunk.square().sum(1, keepdim=True) + centers.square().sum(1) - 2 * chunk @ centers.T
            ).clamp_min(0)
            distances[start : start + len(chunk)], assignment[start : start + len(chunk)] = cost.min(1)
        counts = torch.bincount(assignment, minlength=cells)
        for empty in torch.where(counts == 0)[0].tolist():
            donor = distances.masked_fill(counts[assignment] <= 1, -torch.inf).argmax()
            counts[assignment[donor]] -= 1
            assignment[donor] = empty
            counts[empty] += 1
            distances[donor] = -torch.inf
        centers = x.new_zeros(centers.shape).index_add_(0, assignment, x) / counts[:, None]
        if torch.equal(previous, assignment):
            converged = True
            break
    info = dict(
        initialization_steps=step + 1, initialization_converged=converged, initial_center_indices=indices
    )
    return (assignment, info) if return_info else assignment


@torch.no_grad()
def moment_lloyd_partition(
    H,
    Q,
    m,
    B=None,
    seed=0,
    max_sweeps=100,
    block_size=1024,
    moment_weight=1.0,
    mode="hybrid",
    atol=1e-12,
    rtol=1e-10,
    initialization_steps=20,
    require_initialization_convergence=False,
    backtrack_steps=8,
):
    if mode not in ("hybrid", "full_only", "filtered_batch") or not 1 <= m <= len(H):
        raise ValueError("Invalid mode or cell count")
    if not math.isfinite(moment_weight) or moment_weight < 0 or max_sweeps < 0 or block_size < 1:
        raise ValueError("Invalid weight or budget")
    if backtrack_steps < 0:
        raise ValueError("Backtracking budget must be nonnegative")
    if H.is_cuda:
        torch.cuda.synchronize(H.device)
    started = perf_counter()
    x, q = H.detach().double(), Q.detach().double()
    offset = x.mean(0)
    x = x - offset
    scale = x.square().sum(1).mean().sqrt().clamp_min(1e-30)
    x = x / scale
    energy, original = x.square().sum(1).mean(), x.T @ q / len(x)
    assignment, initialization_info = initialize(
        x, m, seed, block_size, initialization_steps, return_info=True
    )
    if require_initialization_convergence and not initialization_info["initialization_converged"]:
        raise RuntimeError("Initial k-means reached its iteration limit; increase initialization_steps")
    digest = array_digest(assignment.cpu().numpy())

    def aggregate(a):
        return statistics(x, q, a, m, energy, original)

    def score(s):
        return float(s[3] + moment_weight * s[4].norm())

    state = aggregate(assignment)
    history, records = [score(state)], []
    status, converged = "initialization_only" if max_sweeps == 0 else "iteration_limit", False
    for sweep in range(max_sweeps):
        tolerance = atol + rtol * max(1.0, abs(history[-1]))
        proposed, gains = assignment.clone(), x.new_zeros(len(x))
        for start in range(0, len(x), block_size):
            end = min(start + block_size, len(x))
            cost = assignment_cost(x[start:end], q[start:end], state[1], state[2], state[4], moment_weight)
            best, target = cost.min(1)
            old = cost.gather(1, assignment[start:end, None]).squeeze(1)
            proposed[start:end] = torch.where(old <= best, assignment[start:end], target)
            gains[start:end] = old - best
        ids = torch.where(proposed != assignment)[0]
        checks, accepted, kind = 0, None, "full"

        def record(kind, moved, value):
            records.append(
                dict(
                    sweep=sweep + 1,
                    kind=kind,
                    checks=checks,
                    moves=moved,
                    J=value,
                    candidates=len(ids),
                    accepted_fraction=moved / max(1, len(ids)),
                )
            )

        def acceptable(candidate):
            nonlocal checks
            checks += 1
            updated = aggregate(candidate)
            return updated if updated is not None and score(updated) < history[-1] - tolerance else None

        if len(ids):
            accepted = acceptable(proposed)
        if accepted is None and mode == "full_only":
            status = "full_rejected" if len(ids) else "no_proposal"
            record(status, 0, history[-1])
            break
        if accepted is None:
            kind = "partial"
            ordered = ids[torch.argsort(gains[ids], descending=True, stable=True)]
            size = len(ordered) // 2
            for _ in range(backtrack_steps):
                if not size:
                    break
                candidate = assignment.clone()
                chosen = ordered[:size]
                candidate[chosen] = proposed[chosen]
                accepted = acceptable(candidate)
                if accepted is not None:
                    proposed = candidate
                    break
                size //= 2
        if accepted is None and mode == "filtered_batch":
            status = "filtered_rejected" if len(ids) else "no_proposal"
            record(status, 0, history[-1])
            break
        if accepted is None:
            kind = "exact_single"
            for i in range(len(x)):
                delta = move_deltas(x[i], q[i], int(assignment[i]), state, len(x), moment_weight)
                value, target = delta.min(0)
                if float(value) < -tolerance:
                    candidate = assignment.clone()
                    candidate[i] = target
                    accepted = acceptable(candidate)
                    if accepted is not None:
                        proposed = candidate
                        break
            if accepted is None:
                status, converged = "single_move_stationary", True
                record(status, 0, history[-1])
                break
        moved = int((assignment != proposed).sum())
        assignment, state = proposed, accepted
        history.append(score(state))
        record(kind, moved, history[-1])
    counts, centers, labels, variance, moment = state
    if H.is_cuda:
        torch.cuda.synchronize(H.device)
    return dict(
        x=(centers * scale + offset).float().cpu(),
        y=labels.float().cpu(),
        counts=counts.long().cpu(),
        assignment=assignment.cpu(),
        initial_assignment_digest=digest,
        **initialization_info,
        J=history[-1],
        V=float(variance),
        moment_error=float(moment.norm()),
        history=history,
        records=records,
        sweeps=len(records),
        converged=converged,
        status=status,
        mode=mode,
        seconds=perf_counter() - started,
        seed=seed,
        moment_weight=moment_weight,
    )
