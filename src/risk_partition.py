import math
import time

import torch
from src.risk_split import split_partition


def _local_deltas(X, Q, ids, sources, counts, c, y, error, alpha, beta):
    N = len(X)
    xb, qb = X[ids], Q[ids]
    ua, va = xb - c[sources], qb - y[sources]
    ka = counts[sources] / (N * (counts[sources] - 1).clamp_min(1))
    kb = counts / (N * (counts + 1))
    ua2, va2 = ua.square().sum(1), va.square().sum(1)
    ub2 = (xb.square().sum(1)[:, None] + c.square().sum(1)[None, :] - 2 * xb @ c.T).clamp_min(0)
    vb2 = (qb.square().sum(1)[:, None] + y.square().sum(1)[None, :] - 2 * qb @ y.T).clamp_min(0)
    norms2 = error.square().sum((1, 2))
    ta = torch.einsum('bd,bdk,bk->b', ua, error[sources], va)
    xe = torch.einsum('bd,mdk->bmk', xb, error)
    ce = torch.einsum('md,mdk->mk', c, error)
    tb = (torch.einsum('bmk,bk->bm', xe, qb) - torch.einsum('bmk,mk->bm', xe, y)
          - qb @ ce.T + (ce * y).sum(1)[None, :])
    removed = (norms2[sources] - 2 * ka * ta + ka.square() * ua2 * va2).clamp_min(0).sqrt()
    added = (norms2[None, :] + 2 * kb[None, :] * tb
             + kb.square()[None, :] * ub2 * vb2).clamp_min(0).sqrt()
    delta_m = (removed - norms2[sources].sqrt())[:, None] + added - norms2.sqrt()[None, :]
    delta_v = -ka[:, None] * ua2[:, None] + kb[None, :] * ub2
    return alpha * delta_v + beta * delta_m


def _uniform_deltas(X, Q, ids, sources, counts, c, y, error, alpha, beta):
    N, m = len(X), len(c)
    xb, qb = X[ids], Q[ids] - 1 / Q.shape[1]
    y = y - 1 / Q.shape[1]
    na = counts[sources]
    ca, ya = c[sources], y[sources]
    denominator = (na - 1).clamp_min(1)[:, None]
    next_ca = (na[:, None] * ca - xb) / denominator
    next_ya = (na[:, None] * ya - qb) / denominator
    removed = error[None, :, :] + (
        ca[:, :, None] * ya[:, None, :] - next_ca[:, :, None] * next_ya[:, None, :]) / m
    old_products = c[:, :, None] * y[:, None, :]
    old_cross = removed.flatten(1) @ old_products.flatten(1).T
    eq = torch.einsum('bdk,bk->bd', removed, qb)
    xe = torch.einsum('bd,bdk->bk', xb, removed)
    nb, denominator_b = counts[None, :], (counts + 1)[None, :]
    new_cross = (nb.square() * old_cross + nb * (eq @ c.T + xe @ y.T)
                 + (xe * qb).sum(1)[:, None]) / denominator_b.square()
    c2, y2 = c.square().sum(1)[None, :], y.square().sum(1)[None, :]
    x2, q2 = xb.square().sum(1)[:, None], qb.square().sum(1)[:, None]
    xc, qy = xb @ c.T, qb @ y.T
    next_c2 = (nb.square() * c2 + 2 * nb * xc + x2) / denominator_b.square()
    next_y2 = (nb.square() * y2 + 2 * nb * qy + q2) / denominator_b.square()
    cross_c, cross_y = (nb * c2 + xc) / denominator_b, (nb * y2 + qy) / denominator_b
    change_norm2 = (c2 * y2 + next_c2 * next_y2 - 2 * cross_c * cross_y).clamp_min(0) / m ** 2
    next_norm2 = (removed.square().sum((1, 2))[:, None]
                  + 2 * (old_cross - new_cross) / m + change_norm2)
    delta_v = (-na / (N * (na - 1).clamp_min(1)) * (xb - ca).square().sum(1))[:, None]
    delta_v = delta_v + nb / (N * denominator_b) * (x2 + c2 - 2 * xc).clamp_min(0)
    correction_a = (((na - 1) / N - 1 / m).abs() * next_ca.square().sum(1)
                    - (na / N - 1 / m).abs() * ca.square().sum(1))
    correction_b = ((nb + 1) / N - 1 / m).abs() * next_c2 - (nb / N - 1 / m).abs() * c2
    return alpha * (delta_v + correction_a[:, None] + correction_b) + beta * (
        next_norm2.clamp_min(0).sqrt() - error.norm())


def seed_partition(X, Q, m, B, generator, block_size):
    N = len(X)
    if m == N:
        return torch.arange(N, device=X.device)
    if m == 1 or float(X.square().sum(1).mean()) == 0:
        return torch.arange(N, device=X.device) % m

    centered_q = Q - Q.mean(0)
    vh = X.square().sum(1).mean()
    vq = centered_q.square().sum(1).mean()
    if float(vq) > 0:
        eta = (vq / vh).sqrt()
        wx, wq = B * B / 4 + B * eta, B / eta
    else:
        wx, wq = X.new_tensor(B * B / 4), X.new_tensor(0.0)

    Z = torch.cat((wx.sqrt() * X, wq.sqrt() * centered_q), dim=1)
    z2 = Z.square().sum(1)
    nearest = torch.full((N,), torch.inf, dtype=X.dtype, device=X.device)
    selected = torch.zeros(N, dtype=torch.bool, device=X.device)
    seeds = []
    current = int(torch.randint(N, (1,), generator=generator, device=X.device))
    for j in range(m):
        seeds.append(current)
        selected[current] = True
        distance = (z2 + z2[current] - 2 * (Z @ Z[current])).clamp_min(0)
        nearest = torch.minimum(nearest, distance)
        nearest[selected] = 0
        if j + 1 < m:
            if float(nearest.sum()) > 0:
                current = int(torch.multinomial(nearest, 1, generator=generator))
            else:
                current = int(torch.nonzero(~selected)[0, 0])

    seeds = torch.tensor(seeds, device=X.device)
    centers = Z[seeds]
    assignment = torch.empty(N, dtype=torch.long, device=X.device)
    for start in range(0, N, block_size):
        end = min(start + block_size, N)
        cost = (z2[start:end, None] + z2[seeds][None, :]
                - 2 * Z[start:end] @ centers.T).clamp_min(0)
        assignment[start:end] = cost.argmin(1)
    assignment[seeds] = torch.arange(m, device=X.device)
    return assignment


@torch.no_grad()
def risk_partition(H, Q, m, B, seed=0, max_sweeps=30, block_size=1024,
                   atol=1e-12, rtol=1e-10, checkpoints=(), objective_mode='combined',
                   initial_state=None, return_initial_state=False, verify_deltas=True,
                   return_assignment=False, init='surrogate', split_random_directions=2,
                   move_seed=None):
    if init not in ('surrogate', 'split') or split_random_directions < 0:
        raise ValueError('Invalid Risk initialization')
    if init == 'split' and objective_mode != 'combined':
        raise ValueError('Risk splitting requires the original combined objective')
    if not 1 <= m <= len(H) or not math.isfinite(B) or B <= 0:
        raise ValueError('Require 1 <= m <= N and finite B > 0')
    if max_sweeps < 0 or block_size < 1 or min(atol, rtol) < 0:
        raise ValueError('Invalid partition solver settings')
    if objective_mode not in ('combined', 'variance', 'uniform', 'frobenius', 'local'):
        raise ValueError('Require combined, variance, uniform, frobenius or local objective')

    if H.is_cuda:
        torch.cuda.synchronize(H.device)
    started = time.perf_counter()
    X, Q = H.detach().double(), Q.detach().double()
    offset = X.mean(0)
    X = X - offset
    raw_scale = X.square().sum(1).mean().sqrt()
    scale = raw_scale if float(raw_scale) > 0 else X.new_tensor(1.0)
    X = X / scale

    N, d = X.shape
    K = Q.shape[1]
    original_d = d
    alpha, beta = B * B / 4, 0.0 if objective_mode == 'variance' else 2 * B
    generator = torch.Generator(device=X.device).manual_seed(seed)
    initialization_info = {}
    if initial_state is None:
        if init == 'split':
            assignment, initialization_info = split_partition(X, Q, m, B, generator, split_random_directions)
        else:
            assignment = seed_partition(X, Q, m, B, generator, block_size)
    else:
        assignment = initial_state['assignment'].to(device=X.device, dtype=torch.long).clone()
        if assignment.shape != (N,) or int(assignment.min()) < 0 or int(assignment.max()) >= m:
            raise ValueError('Invalid initial assignment')
        if bool((torch.bincount(assignment, minlength=m) == 0).any()):
            raise ValueError('Initial partition must have no empty cells')
        generator.set_state(initial_state['generator_state'].cpu())
    if move_seed is not None:
        generator.manual_seed(move_seed)
    saved_initial = dict(assignment=assignment.cpu().clone(),
                         generator_state=generator.get_state()) if return_initial_state else None
    if H.is_cuda:
        torch.cuda.synchronize(H.device)
    initialization_seconds = time.perf_counter() - started
    if objective_mode == 'uniform':
        X = torch.cat((X, X.new_ones(N, 1)), dim=1)
        d = X.shape[1]
    x2, q2 = X.square().sum(1), Q.square().sum(1)
    energy, moment = x2.mean(), X.T @ Q / N
    if objective_mode == 'uniform':
        moment = X.T @ (Q - 1 / K) / N
    second_moment = X.T @ X / N if objective_mode == 'frobenius' else None

    def covariance(n, s):
        within = second_moment - (s / n[:, None]).T @ s / N
        return (within + within.T) / 2

    def aggregate():
        n = torch.bincount(assignment, minlength=m).double()
        s = X.new_zeros(m, d).index_add_(0, assignment, X)
        r = Q.new_zeros(m, K).index_add_(0, assignment, Q)
        cross = None
        if objective_mode == 'local':
            cross = X.new_zeros(m, d, K)
            for start in range(0, N, block_size):
                end = min(start + block_size, N)
                cross.index_add_(0, assignment[start:end], X[start:end, :, None] * Q[start:end, None, :])
        return n, s, r, cross

    def moment_norm(value):
        return value.flatten(1).norm(dim=1).sum() if objective_mode == 'local' else value.norm()

    def score(n, s, r, cross=None):
        c = s / n[:, None]
        error = moment - c.T @ r / N
        if objective_mode == 'local':
            error = (cross - c[:, :, None] * r[:, None, :]) / N
        variance = (energy - (s * c).sum() / N).clamp_min(0)
        correction = X.new_tensor(0.0)
        if objective_mode == 'uniform':
            error = moment - c.T @ (r / n[:, None] - 1 / K) / m
            correction = ((n / N - 1 / m).abs() * c.square().sum(1)).sum()
        feature_error = covariance(n, s).norm() if objective_mode == 'frobenius' else variance + correction
        value = float(alpha * feature_error + beta * moment_norm(error))
        if not math.isfinite(value):
            raise FloatingPointError('Nonfinite partition objective')
        return error, variance, value

    counts, sums, label_sums, cross_sums = aggregate()
    error, variance, objective = score(counts, sums, label_sums, cross_sums)
    initial_variance, initial_moment = float(variance), float(moment_norm(error))
    history, moves_history = [objective], []
    snapshots = {}

    def snapshot(sweep):
        return dict(x=((sums[:, :original_d] / counts[:, None]) * scale + offset).float().cpu(),
                    y=(label_sums / counts[:, None]).float().cpu(),
                    counts=counts.long().cpu(), J=objective, V=float(variance),
                    moment_error=float(moment_norm(error)), sweeps=sweep,
                    seconds=time.perf_counter() - started)

    if 0 in checkpoints:
        snapshots[0] = snapshot(0)

    def check_deltas(ids, sources, delta):
        eligible = torch.nonzero(counts[sources] > 1).flatten()[:4]
        for local in eligible.tolist():
            source = int(sources[local])
            target = (source + 1) % m
            node = ids[local]
            next_counts, next_sums, next_labels = counts.clone(), sums.clone(), label_sums.clone()
            next_counts[source] -= 1
            next_counts[target] += 1
            next_sums[source] -= X[node]
            next_sums[target] += X[node]
            next_labels[source] -= Q[node]
            next_labels[target] += Q[node]
            next_cross = cross_sums
            if objective_mode == 'local':
                next_cross = cross_sums.clone()
                product = X[node, :, None] * Q[node, None, :]
                next_cross[source] -= product
                next_cross[target] += product
            direct = score(next_counts, next_sums, next_labels, next_cross)[2] - objective
            if abs(float(delta[local, target]) - direct) > 1e-8 * max(1.0, abs(objective)):
                raise FloatingPointError('Move delta disagrees with full objective recomputation')
        return len(eligible) > 0

    def apply_moves(ids, destinations):
        nonlocal counts, sums, label_sums, cross_sums, error, variance, objective
        sources = assignment[ids]
        next_counts = (counts + torch.bincount(destinations, minlength=m)
                       - torch.bincount(sources, minlength=m))
        if bool((next_counts > 0).all()):
            xb, qb = X[ids], Q[ids]
            next_sums, next_labels = sums.clone(), label_sums.clone()
            next_sums.index_add_(0, sources, -xb)
            next_sums.index_add_(0, destinations, xb)
            next_labels.index_add_(0, sources, -qb)
            next_labels.index_add_(0, destinations, qb)
            next_cross = cross_sums
            if objective_mode == 'local':
                next_cross = cross_sums.clone()
                products = xb[:, :, None] * qb[:, None, :]
                next_cross.index_add_(0, sources, -products)
                next_cross.index_add_(0, destinations, products)
            next_error, next_variance, next_objective = score(
                next_counts, next_sums, next_labels, next_cross)
            tolerance = atol + rtol * max(1.0, abs(objective))
            if next_objective < objective - tolerance:
                counts, sums, label_sums = next_counts, next_sums, next_labels
                cross_sums = next_cross
                error, variance, objective = next_error, next_variance, next_objective
                assignment[ids] = destinations
                return len(ids)
        if len(ids) == 1:
            return 0
        middle = len(ids) // 2
        left = apply_moves(ids[:middle], destinations[:middle])
        right = apply_moves(ids[middle:], destinations[middle:])
        return left + right

    converged = m in (1, N) or (float(raw_scale) == 0 and objective_mode != 'uniform')
    checked = False
    for _ in range(0 if converged else max_sweeps):
        moved = 0
        order = torch.randperm(N, generator=generator, device=X.device)
        for ids in order.split(block_size):
            c, y = sums / counts[:, None], label_sums / counts[:, None]
            if objective_mode in ('uniform', 'local'):
                sources = assignment[ids]
                deltas = _local_deltas if objective_mode == 'local' else _uniform_deltas
                delta = deltas(X, Q, ids, sources, counts, c, y, error, alpha, beta)
                if verify_deltas and not checked:
                    checked = check_deltas(ids, sources, delta)
                delta.scatter_(1, sources[:, None], torch.inf)
                delta[counts[sources] <= 1] = torch.inf
                best_delta, destinations = delta.min(1)
                take = best_delta < -(atol + rtol * max(1.0, abs(objective)))
                if bool(take.any()):
                    moved += apply_moves(ids[take], destinations[take])
                continue
            xb, qb = X[ids], Q[ids]
            sources = assignment[ids]
            ua, va = xb - c[sources], qb - y[sources]
            ua2, va2 = ua.square().sum(1), va.square().sum(1)
            ka = counts[sources] / (N * (counts[sources] - 1).clamp_min(1))
            kb = counts / (N * (counts + 1))
            ub2 = (x2[ids, None] + c.square().sum(1)[None, :]
                   - 2 * xb @ c.T).clamp_min(0)
            vb2 = (q2[ids, None] + y.square().sum(1)[None, :]
                   - 2 * qb @ y.T).clamp_min(0)
            xe, ce = xb @ error, c @ error
            ta = ((xe - ce[sources]) * va).sum(1)
            tb = ((xe * qb).sum(1)[:, None] - xe @ y.T - qb @ ce.T
                  + (ce * y).sum(1)[None, :])
            cross_x = (ua * xb).sum(1)[:, None] - ua @ c.T
            cross_q = (va * qb).sum(1)[:, None] - va @ y.T
            next_norm2 = (error.square().sum() - 2 * (ka * ta)[:, None]
                          + (ka.square() * ua2 * va2)[:, None]
                          + 2 * kb[None, :] * tb
                          + kb.square()[None, :] * ub2 * vb2
                          - 2 * ka[:, None] * kb[None, :] * cross_x * cross_q)
            delta_v = -ka[:, None] * ua2[:, None] + kb[None, :] * ub2
            if objective_mode == 'frobenius':
                within = covariance(counts, sums)
                xs, cs = xb @ within, c @ within
                source_quadratic = ((ua @ within) * ua).sum(1)
                target_quadratic = ((xs * xb).sum(1)[:, None] - 2 * xs @ c.T
                                    + (cs * c).sum(1)[None, :])
                next_covariance_norm2 = (within.square().sum()
                    - 2 * (ka * source_quadratic)[:, None]
                    + 2 * kb[None, :] * target_quadratic
                    + (ka.square() * ua2.square())[:, None]
                    + kb.square()[None, :] * ub2.square()
                    - 2 * ka[:, None] * kb[None, :] * cross_x.square())
                delta_v = next_covariance_norm2.clamp_min(0).sqrt() - within.norm()
            delta = alpha * delta_v + beta * (next_norm2.clamp_min(0).sqrt()
                                              - error.norm())
            if objective_mode == 'frobenius' and verify_deltas and not checked:
                checked = check_deltas(ids, sources, delta)
            delta.scatter_(1, sources[:, None], torch.inf)
            delta[counts[sources] <= 1] = torch.inf
            best_delta, destinations = delta.min(1)
            tolerance = atol + rtol * max(1.0, abs(objective))
            take = best_delta < -tolerance
            if bool(take.any()):
                moved += apply_moves(ids[take], destinations[take])

        counts, sums, label_sums, cross_sums = aggregate()
        error, variance, objective = score(counts, sums, label_sums, cross_sums)
        if objective > history[-1] + 1e-8 * max(1.0, abs(history[-1])):
            raise FloatingPointError('Partition objective increased')
        history.append(objective)
        moves_history.append(moved)
        if len(moves_history) in checkpoints:
            snapshots[len(moves_history)] = snapshot(len(moves_history))
        if moved == 0:
            converged = True
            break

    result = dict(x=((sums[:, :original_d] / counts[:, None]) * scale + offset).float().cpu(),
                  y=(label_sums / counts[:, None]).float().cpu(),
                  counts=counts.long().cpu(), J=objective, V=float(variance),
                  moment_error=float(moment_norm(error)), history=history,
                  moves=moves_history, sweeps=len(moves_history),
                  converged=converged, B=float(B), seed=int(seed))
    result['objective_mode'] = objective_mode
    result.update(initialization=init, initialization_seconds=initialization_seconds,
                  initial_V=initial_variance, initial_moment_error=initial_moment, **initialization_info)
    result['bound_J'] = B * B / 4 * result['V'] + 2 * B * result['moment_error']
    if objective_mode == 'local':
        global_error = float(error.sum(0).norm())
        result.update(objective_name='local_moment_surrogate', global_moment_error=global_error,
                      local_moment_error=result['moment_error'], variance_term=alpha * result['V'],
                      moment_term=beta * result['moment_error'],
                      cancellation_ratio=global_error / result['moment_error'] if result['moment_error'] > 0 else 1.)
    if objective_mode == 'frobenius':
        frobenius = float(covariance(counts, sums).norm())
        if frobenius > float(variance) + 1e-8 * max(1.0, float(variance)):
            raise FloatingPointError('Covariance Frobenius norm exceeds its trace')
        result.update(objective_name='frobenius_risk_bound', covariance_fro=frobenius,
                      covariance_trace=float(variance), trace_bound=result['bound_J'],
                      bound_J=objective, feature_term=alpha * frobenius,
                      moment_term=beta * result['moment_error'],
                      trace_to_fro=float(variance) / frobenius if frobenius > 0 else 1.0)
    if objective_mode == 'uniform':
        correction = float(((counts / N - 1 / m).abs()
                            * (sums / counts[:, None]).square().sum(1)).sum())
        result.update(mass_correction=correction, bound_J=objective,
                      mass_tv=float((counts / N - 1 / m).abs().sum() / 2),
                      variance_term=alpha * result['V'], correction_term=alpha * correction,
                      moment_term=beta * result['moment_error'])
    if return_initial_state:
        result['initial_state'] = saved_initial
    if return_assignment:
        result['assignment'] = assignment.cpu()
    if checkpoints:
        result['snapshots'] = snapshots
    result['seconds'] = time.perf_counter() - started
    return result
