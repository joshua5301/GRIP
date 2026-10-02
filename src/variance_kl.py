import math
from time import perf_counter

import torch

from src.io import array_digest


@torch.no_grad()
def variance_kl_partition(
    H,
    Q,
    m,
    B,
    seed_partition,
    seed=0,
    max_sweeps=100,
    block_size=1024,
    atol=1e-12,
    rtol=1e-10,
):
    if not 1 <= m <= len(H) or not math.isfinite(B) or B <= 0:
        raise ValueError("Require 1 <= m <= N and finite B > 0")
    if max_sweeps < 0 or block_size < 1:
        raise ValueError("Invalid iteration budget")
    if H.is_cuda:
        torch.cuda.synchronize(H.device)
    started = perf_counter()
    X, Q = H.detach().double(), Q.detach().double()
    offset = X.mean(0)
    X = X - offset
    scale = X.square().sum(1).mean().sqrt()
    scale = scale if float(scale) > 0 else X.new_tensor(1.0)
    X = X / scale
    n = len(X)
    alpha, beta = B * B / 2, 8.0
    generator = torch.Generator(device=X.device).manual_seed(seed)
    assignment = seed_partition(X, Q, m, B, generator, block_size)
    initial_digest = array_digest(assignment.cpu().numpy())
    entropy = torch.special.xlogy(Q, Q).sum(1)
    tiny = torch.finfo(Q.dtype).tiny

    def statistics(a):
        counts = torch.bincount(a, minlength=m).to(X)
        centers = X.new_zeros(m, X.shape[1]).index_add_(0, a, X) / counts[:, None]
        labels = Q.new_zeros(m, Q.shape[1]).index_add_(0, a, Q) / counts[:, None]
        variance = ((X - centers[a]).square().sum(1)).mean()
        kl = (entropy - (Q * labels[a].clamp_min(tiny).log()).sum(1)).mean().clamp_min(0)
        objective = alpha * variance + beta * kl
        return counts, centers, labels, variance, kl, objective

    state = statistics(assignment)
    history, moves, converged = [float(state[-1])], [], False
    for _ in range(max_sweeps):
        counts, centers, labels, _, _, _ = state
        proposed = torch.empty_like(assignment)
        assigned_cost = X.new_empty(n)
        log_labels = labels.clamp_min(tiny).log()
        zero_labels = (labels == 0).to(X)
        for start in range(0, n, block_size):
            end = min(start + block_size, n)
            x, q = X[start:end], Q[start:end]
            distance = (
                x.square().sum(1, keepdim=True) + centers.square().sum(1) - 2 * x @ centers.T
            ).clamp_min(0)
            kl = (entropy[start:end, None] - q @ log_labels.T).clamp_min(0)
            cost = alpha * distance + beta * kl
            cost.masked_fill_((q > 0).to(X) @ zero_labels.T > 0, torch.inf)
            best, target = cost.min(1)
            old = assignment[start:end]
            old_cost = cost.gather(1, old[:, None]).squeeze(1)
            target = torch.where(old_cost <= best, old, target)
            proposed[start:end] = target
            assigned_cost[start:end] = cost.gather(1, target[:, None]).squeeze(1)
        counts = torch.bincount(proposed, minlength=m)
        for empty in torch.where(counts == 0)[0].tolist():
            eligible = counts[proposed] > 1
            donor = assigned_cost.masked_fill(~eligible, -torch.inf).argmax()
            counts[proposed[donor]] -= 1
            proposed[donor] = empty
            counts[empty] += 1
            assigned_cost[donor] = 0
        updated = statistics(proposed)
        value = float(updated[-1])
        if not math.isfinite(value) or value > history[-1] + atol + rtol * abs(history[-1]):
            raise RuntimeError("Variance-KL Lloyd objective increased or became nonfinite")
        moved = int((proposed != assignment).sum())
        assignment, state = proposed, updated
        history.append(value)
        moves.append(moved)
        if moved == 0:
            converged = True
            break
    counts, centers, labels, variance, kl, objective = state
    moment = X.T @ Q / n - centers.T @ (counts[:, None] * labels) / n
    if H.is_cuda:
        torch.cuda.synchronize(H.device)
    seconds = perf_counter() - started
    return dict(
        x=(centers * scale + offset).float().cpu(),
        y=labels.float().cpu(),
        counts=counts.cpu(),
        assignment=assignment.cpu(),
        initial_assignment_digest=initial_digest,
        J=float(objective),
        V=float(variance),
        label_kl=float(kl),
        moment_error=float(moment.norm()),
        history=history,
        moves=moves,
        sweeps=len(moves),
        converged=converged,
        B=B,
        seed=seed,
        seconds=seconds,
        variance_weight=alpha,
        kl_weight=beta,
    )
