import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import trange

from src.balanced_assignment import BalancedMoments, CachedBalancedMoments
from src.low_rank_assignment import (LowRankLogits, LowRankMoments, assignment_inputs,
                                     encode_nodes, initialize_encoder, initialize_factors, initialize_mlp)
from src.node_distances import array_digest
from src.ntk_transforms import FeatureTransform
from src.risk_experiment import _fingerprint, _prepare_dataset, _train_student
from src.soft_ridge_partition import AssignmentMoments, augmented, decode_moments, initial_logits, make_material
from src.stationarity_risk import fit_head, head_objective


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
def conjugate_gradient(multiply, rhs, diagonal, rtol=1e-6, atol=1e-12, max_iter=512, initial=None):
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
        restart = iterations % 50 == 0 or float(residual.norm()) <= target
        if restart:
            residual = rhs - multiply(solution)
        preconditioned = residual / diagonal
        next_product = (residual * preconditioned).sum()
        direction = preconditioned if restart else preconditioned + (next_product / product) * direction
        product = next_product
    residual_norm = float((rhs - multiply(solution)).norm())
    return solution, dict(cg_iterations=iterations, cg_residual=residual_norm,
                          cg_relative_residual=residual_norm / max(norm, 1e-30),
                          cg_converged=np.isfinite(residual_norm) and residual_norm <= target)


@torch.no_grad()
def solve_head_system(x, labels, mass, theta, penalty, rhs, rtol=1e-6, atol=1e-12,
                      max_iter=512, reduced_limit=2048):
    multiply, diagonal = hessian_operator(x, labels, mass, theta, penalty)
    classes = theta.shape[0]
    dimension = min(x.shape) * classes
    diagnostic = dict(cg_iterations=0)
    if not (x.shape[1] > 2 * x.shape[0] and dimension <= reduced_limit):
        solution, diagnostic = conjugate_gradient(multiply, rhs, diagonal, rtol=rtol,
                                                   atol=atol, max_iter=max_iter)
        diagnostic.update(hessian_solver='pcg', hessian_reduced_dimension=0)
        if diagnostic['cg_converged']:
            return solution, diagnostic
    if dimension > reduced_limit:
        solution, retry = conjugate_gradient(multiply, rhs, diagonal, rtol=rtol,
                                              atol=atol, max_iter=4 * max_iter)
        retry.update(hessian_solver='pcg_extended', hessian_reduced_dimension=0,
                     cg_iterations=diagnostic['cg_iterations'] + retry['cg_iterations'])
        return solution, retry
    _, singular, basis = torch.linalg.svd(x, full_matrices=False)
    projected_x = x @ basis.T
    projected_rhs = rhs @ basis.T
    probability = (x @ theta.T).softmax(1)
    covariance = torch.diag_embed(probability) - probability[:, :, None] * probability[:, None, :]
    covariance *= (mass * labels.sum(1))[:, None, None]
    system = torch.einsum('icd,ia,ib->cadb', covariance, projected_x, projected_x).reshape(dimension, dimension)
    system = (system + system.T) / 2 + penalty * torch.eye(dimension, dtype=x.dtype, device=x.device)
    projected_solution = torch.linalg.solve(system, projected_rhs.flatten()).reshape(classes, len(singular))
    solution = projected_solution @ basis + (rhs - projected_rhs @ basis) / penalty
    norm = float(rhs.norm())
    residual = float((multiply(solution) - rhs).norm())
    diagnostic.update(hessian_solver='reduced_direct', hessian_reduced_dimension=dimension,
                      cg_residual=residual, cg_relative_residual=residual / max(norm, 1e-30),
                      cg_converged=bool(torch.isfinite(solution).all()) and np.isfinite(residual)
                                   and residual <= max(atol, rtol * norm))
    return solution, diagnostic


def solve_inner(centers, labels, mass, penalty, initial=None, max_iter=2000, grad_tol=1e-7,
                polish_steps=8, cg_max_iter=512):
    fitted = fit_head(centers.detach(), labels.detach(), mass.detach(), penalty,
                      max_iter=max_iter, grad_tol=grad_tol, initial_theta=initial)
    theta = fitted['theta']
    x = augmented(centers.detach())
    labels, mass = labels.detach(), mass.detach()
    polished = 0
    with torch.no_grad():
        for _ in range(polish_steps):
            gradient = head_gradient(x, labels, mass, theta, penalty)
            if float(gradient.abs().max()) <= grad_tol:
                break
            direction, diagnostic = solve_head_system(x, labels, mass, theta, penalty, gradient,
                                                       rtol=1e-8, atol=1e-14, max_iter=cg_max_iter)
            if not diagnostic['cg_converged']:
                break
            objective = head_objective(x, labels, mass, theta, penalty)
            decrease = (gradient * direction).sum()
            accepted = False
            for exponent in range(24):
                step = .5 ** exponent
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
    return dict(theta=theta.detach(), inner_grad_max=maximum,
                inner_converged=np.isfinite(maximum) and maximum <= grad_tol,
                inner_iterations=fitted['iterations'], inner_polish_steps=polished)


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
            step = .5 ** exponent
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
    return dict(theta=theta, inner_grad_max=maximum,
                inner_converged=np.isfinite(maximum) and maximum <= grad_tol,
                inner_iterations=accepted, inner_polish_steps=0,
                tracking_failed=failed or not np.isfinite(maximum))


@torch.no_grad()
def outer_value_gradient(z, q, theta, chunk_size=4096, features=None):
    value, gradient = z.new_zeros(()), torch.zeros_like(theta)
    for start in range(0, len(z), chunk_size):
        x = augmented(z[start:start + chunk_size]) if features is None else features[start:start + chunk_size]
        labels = q[start:start + chunk_size]
        log_probability = (x @ theta.T).log_softmax(1)
        value -= (labels * log_probability).sum() / len(z)
        error = labels.sum(1, keepdim=True) * log_probability.exp() - labels
        gradient += error.T @ x / len(z)
    return float(value), gradient


def implicit_moment_gradient(moments, dimension, theta, vector, penalty):
    variable = moments.detach().requires_grad_()
    centers, labels, mass = decode_moments(variable, dimension)
    gradient = head_gradient(augmented(centers), labels, mass, theta.detach(), penalty)
    result, = torch.autograd.grad(-(gradient * vector.detach()).sum(), variable)
    return result


def fixed_target_moments(centers, labels, mass):
    return torch.cat((mass[:, None], mass[:, None] * centers, mass[:, None] * labels), dim=1)


def cpu_state(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_state(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [cpu_state(item) for item in value]
    return value


def optimize_ce_assignment(z, q, assignment, penalty=3e-5, steps=300, lr=.01,
                           mixing=.05, chunk_size=4096, inner_max_iter=2000,
                           inner_tol=1e-7, cg_max_iter=512, cg_rtol=1e-6,
                           save_assignment=True, folder=None, checkpoint_steps=(),
                           mass_mode='free', balance_steps=300, balance_tol=1e-8,
                           balance_backend='cached', balance_cg_steps=512, balance_cg_rtol=1e-7,
                           outer_chunk_size=65536, log_every=10, assignment_rank=None, factor_seed=0,
                           assignment_input='node', assignment_encoder='linear', encoder_hidden=64,
                           resume_state=None, save_resume=False, solver_mode='exact',
                           tracking_inner_steps=2, tracking_cg_steps=8, tracking_refresh=20,
                           feature_control='joint'):
    resume_config = {key: value for key, value in locals().copy().items()
                     if key not in ('z', 'q', 'assignment', 'steps', 'folder', 'checkpoint_steps',
                                    'resume_state', 'save_resume', 'log_every')}
    if resume_state is not None or save_resume:
        resume_config['data_digest'] = array_digest(z.detach().cpu().numpy(), q.detach().cpu().numpy(),
                                                    assignment.cpu().numpy())
    if steps < 0 or any(not np.isfinite(v) or v <= 0 for v in
                        (penalty, lr, chunk_size, inner_max_iter, inner_tol, cg_max_iter, cg_rtol)):
        raise ValueError('Require positive finite solver settings and nonnegative steps')
    if mass_mode not in ('free', 'uniform') or balance_steps < 1 or not 0 < balance_tol < 1:
        raise ValueError('Invalid mass constraint settings')
    if feature_control not in ('joint', 'assignment', 'direct'):
        raise ValueError('Unknown feature control')
    if feature_control != 'joint' and mass_mode != 'free':
        raise ValueError('Fixed-target controls use free geometric assignments and fixed initial loss mass')
    if (balance_backend not in ('cached', 'chunked') or min(balance_cg_steps, outer_chunk_size, log_every) < 1
            or not 0 < balance_cg_rtol < 1):
        raise ValueError('Invalid performance settings')
    if assignment_input not in ('node', 'features', 'features_labels') or (assignment_input != 'node' and assignment_rank is None):
        raise ValueError('Feature assignment requires assignment_rank and a supported input mode')
    if assignment_encoder not in ('linear', 'mlp') or (assignment_input == 'node' and assignment_encoder != 'linear'):
        raise ValueError('MLP encoder requires feature-conditioned assignments')
    if (solver_mode not in ('exact', 'tracking') or any(
            not isinstance(v, (int, np.integer)) or v < 1
            for v in (tracking_inner_steps, tracking_cg_steps, tracking_refresh))):
        raise ValueError('Invalid tracking solver settings')
    checkpoints = set(checkpoint_steps)
    if any(not isinstance(step, (int, np.integer)) or not 0 <= step <= steps for step in checkpoints):
        raise ValueError('Checkpoint steps must be integers within the optimization budget')
    if checkpoints:
        checkpoints.update((0, steps))
    snapshots = {}
    material = make_material(z, q)
    full_features = augmented(z)
    clusters = int(assignment.max()) + 1
    if assignment_rank is None:
        logits = initial_logits(assignment, clusters, mixing).requires_grad_()
        parameters = [logits]
    elif assignment_input != 'node':
        inputs = assignment_inputs(z, q, assignment_input)
        if assignment_encoder == 'mlp':
            encoder_parameters, v = initialize_mlp(inputs, clusters, assignment_rank, encoder_hidden, factor_seed)
        else:
            weight, v = initialize_encoder(inputs, clusters, assignment_rank, factor_seed)
            encoder_parameters = [weight]
        parameters = [*encoder_parameters, v]
    else:
        u, v = initialize_factors(assignment, clusters, assignment_rank, factor_seed)
        parameters = [u, v]
    if feature_control != 'joint':
        with torch.no_grad():
            if assignment_rank is None:
                initial = AssignmentMoments.apply(logits, material, chunk_size)
            else:
                initial_u = encode_nodes(inputs, encoder_parameters) if assignment_input != 'node' else u
                initial = LowRankMoments.apply(initial_u, v, assignment, material, mixing, chunk_size)
            initial_centers, fixed_labels, fixed_mass = decode_moments(initial, z.shape[1])
        if feature_control == 'direct':
            free_centers = initial_centers.clone().requires_grad_()
            parameters = [free_centers]
    optimizer = torch.optim.Adam(parameters, lr=lr, eps=1e-12, foreach=False)
    folder = Path(folder) if folder is not None else None
    if folder is not None:
        folder.mkdir(parents=True, exist_ok=True)
    history, theta, best, dual = [], None, float('inf'), None
    vector = None
    start_step = 0
    if resume_state is not None:
        if resume_state['config'] != resume_config or resume_state['step'] > steps:
            raise ValueError('Resume state does not match data, solver settings or step budget')
        with torch.no_grad():
            for parameter, saved in zip(parameters, resume_state['parameters'], strict=True):
                parameter.copy_(saved.to(parameter))
        optimizer.load_state_dict(resume_state['optimizer'])
        start_step = resume_state['step']
        theta = resume_state['theta'].to(z)
        if solver_mode == 'tracking':
            theta = resume_state['tracking_theta_before']
            theta = theta.to(z) if theta is not None else None
            vector = resume_state['tracking_vector_before']
            vector = vector.to(z) if vector is not None else None
        dual = resume_state['dual'].to(z) if resume_state['dual'] is not None else None
        best, best_step = resume_state['best'], resume_state['best_step']
        best_moments, best_theta = resume_state['best_moments'], resume_state['best_theta']
        initial_moments, scale = resume_state['initial_moments'], resume_state['scale']
        history = [dict(row) for row in resume_state['history'] if row['step'] < start_step]
        snapshots = dict(resume_state['snapshots'])
        if save_assignment:
            best_parameters = [p.to(z.device) for p in resume_state['best_parameters']]
            best_dual = resume_state['best_dual']
    elapsed = resume_state['elapsed'] if resume_state is not None else 0.
    started = time.perf_counter()
    def timestamp():
        if z.is_cuda:
            torch.cuda.synchronize(z.device)
        return time.perf_counter()
    for step in trange(start_step, steps + 1, desc='CE inner + CE outer'):
        tick = timestamp()
        optimizer.zero_grad(set_to_none=True)
        if assignment_input != 'node' and feature_control != 'direct':
            u = encode_nodes(inputs, encoder_parameters)
        balance = dict(balance_iterations=0, row_residual=np.nan, column_residual=np.nan)
        if feature_control == 'direct':
            moments = fixed_target_moments(free_centers, fixed_labels, fixed_mass)
        elif mass_mode == 'uniform':
            if assignment_rank is not None:
                logits = LowRankLogits.apply(u, v, assignment, mixing, chunk_size)
            if balance_backend == 'cached':
                moments, dual, diagnostic = CachedBalancedMoments.apply(
                    logits, material, chunk_size, balance_steps, balance_tol, dual,
                    balance_cg_steps, balance_cg_rtol)
            else:
                moments, dual, diagnostic = BalancedMoments.apply(
                    logits, material, chunk_size, balance_steps, balance_tol, dual)
            balance = dict(balance_iterations=int(diagnostic[0]), row_residual=float(diagnostic[1]),
                           column_residual=float(diagnostic[2]))
        elif assignment_rank is not None:
            moments = LowRankMoments.apply(u, v, assignment, material, mixing, chunk_size)
        else:
            moments = AssignmentMoments.apply(logits, material, chunk_size)
        if feature_control == 'assignment':
            moving_centers, _, _ = decode_moments(moments, z.shape[1])
            moments = fixed_target_moments(moving_centers, fixed_labels, fixed_mass)
        if not bool(torch.isfinite(moments).all()) or bool((moments[:, 0] <= 0).any()):
            raise FloatingPointError('Nonfinite moments or empty soft cell')
        centers, labels, mass = decode_moments(moments.detach(), z.shape[1])
        after_assignment = timestamp()
        theta_before, vector_before = theta, vector
        refresh = (solver_mode == 'exact' or theta is None or step == steps
                   or step in checkpoints or step % tracking_refresh == 0)
        head_fallback = False
        if refresh:
            fitted = solve_inner(centers, labels, mass, penalty, theta, inner_max_iter,
                                 inner_tol, cg_max_iter=cg_max_iter)
        else:
            fitted = track_inner(centers, labels, mass, penalty, theta,
                                 tracking_inner_steps, tracking_cg_steps, inner_tol)
            head_fallback = fitted.pop('tracking_failed')
            if head_fallback:
                refresh = True
                fitted = solve_inner(centers, labels, mass, penalty, theta, inner_max_iter,
                                     inner_tol, cg_max_iter=cg_max_iter)
        theta = fitted.pop('theta')
        after_inner = timestamp()
        value, outer_gradient = outer_value_gradient(z, q, theta, outer_chunk_size, full_features)
        after_outer = timestamp()
        row = dict(step=step, J=value, **fitted, **balance, cg_iterations=0, cg_residual=np.nan,
                   cg_relative_residual=np.nan, cg_converged=False,
                   solver_mode=solver_mode, exact_refresh=refresh, head_fallback=head_fallback,
                   J_exact=fitted['inner_converged'], implicit_fallback=False,
                   head_correction_relative=(float((theta - theta_before).norm() / theta.norm().clamp_min(1e-30))
                                             if theta_before is not None else np.nan),
                   implicit_correction_relative=np.nan,
                   min_mass=float(mass.min()), max_mass=float(mass.max()),
                   effective_cells=float(1 / mass.square().sum()))
        row.update(assignment_seconds=after_assignment - tick, inner_seconds=after_inner - after_assignment,
                   outer_seconds=after_outer - after_inner, implicit_seconds=0., backward_seconds=0.)
        failure = None
        if (refresh and not fitted['inner_converged']) or not np.isfinite(value):
            failure = 'Inner CE did not converge; increase inner_max_iter or inspect inner_tol'
        else:
            if step == 0:
                initial_moments = moments.detach().cpu()
                scale = max(value, 1e-12)
            if fitted['inner_converged'] and value < best:
                best, best_step = value, step
                best_moments, best_theta = moments.detach().cpu(), theta.cpu()
                if save_assignment:
                    best_parameters = [parameter.detach().clone() for parameter in parameters]
                    best_dual = dual.clone() if dual is not None else None
            if step < steps:
                implicit_start = timestamp()
                if refresh:
                    vector, diagnostic = solve_head_system(augmented(centers), labels, mass, theta, penalty,
                                                            outer_gradient, rtol=cg_rtol, max_iter=cg_max_iter)
                else:
                    multiply, diagonal = hessian_operator(augmented(centers), labels, mass, theta, penalty)
                    vector, diagnostic = conjugate_gradient(multiply, outer_gradient, diagonal,
                                                              rtol=cg_rtol, max_iter=tracking_cg_steps,
                                                              initial=vector)
                    diagnostic.update(hessian_solver='tracking_pcg', hessian_reduced_dimension=0)
                    if not bool(torch.isfinite(vector).all()) or not np.isfinite(diagnostic['cg_residual']):
                        row['implicit_fallback'] = True
                        vector, diagnostic = solve_head_system(augmented(centers), labels, mass, theta, penalty,
                                                                outer_gradient, rtol=cg_rtol, max_iter=cg_max_iter)
                row.update(diagnostic)
                row['implicit_seconds'] = timestamp() - implicit_start
                if vector_before is not None:
                    row['implicit_correction_relative'] = float(
                        (vector - vector_before).norm() / vector.norm().clamp_min(1e-30))
                if (refresh or row['implicit_fallback']) and not diagnostic['cg_converged']:
                    failure = (f'Implicit Hessian solve did not converge: solver={diagnostic["hessian_solver"]}, '
                               f'relative residual={diagnostic["cg_relative_residual"]:.3g}, target={cg_rtol:.3g}')
        row.update(best_J=best, seconds=elapsed + time.perf_counter() - started,
                   status='failed' if failure else 'evaluated' if step == steps else 'update')
        history.append(row)
        if folder is not None and (step == steps or failure):
            pd.DataFrame(history).to_csv(folder / 'optimization.csv', index=False)
        if failure:
            if folder is not None:
                (folder / 'failure.json').write_text(json.dumps(dict(step=step, reason=failure), indent=2))
            raise RuntimeError(f'Step {step}: {failure}')
        if step in checkpoints:
            snapshot = dict(step=step, moments=moments.detach().cpu().clone(),
                            theta=theta.cpu().clone(), teacher_ce=value, inner_grad_max=fitted['inner_grad_max'],
                            J_exact=fitted['inner_converged'])
            snapshots[step] = snapshot
            if folder is not None:
                checkpoint_dir = folder / 'checkpoints'
                checkpoint_dir.mkdir(exist_ok=True)
                torch.save(snapshot, checkpoint_dir / f'step_{step:06d}.pt')
        if save_resume and (step in checkpoints or step == steps):
            state = cpu_state(dict(config=resume_config, step=step, parameters=parameters,
                                   optimizer=optimizer.state_dict(), theta=theta, dual=dual,
                                   tracking_theta_before=theta_before, tracking_vector_before=vector_before,
                                   best=best, best_step=best_step, best_moments=best_moments,
                                   best_theta=best_theta, initial_moments=initial_moments, scale=scale,
                                   history=history, snapshots=snapshots, elapsed=row['seconds'],
                                   best_parameters=best_parameters if save_assignment else None,
                                   best_dual=best_dual if save_assignment else None))
            if folder is not None:
                torch.save(state, folder / 'resume.tmp.pt')
                (folder / 'resume.tmp.pt').replace(folder / 'resume.pt')
        if step == steps:
            break
        backward_start = timestamp()
        direction = implicit_moment_gradient(moments, z.shape[1], theta, vector, penalty)
        if not bool(torch.isfinite(direction).all()):
            raise FloatingPointError('Nonfinite implicit gradient')
        try:
            moments.backward(direction / scale)
        except RuntimeError as error:
            row.update(status='failed', backward_seconds=timestamp() - backward_start)
            if folder is not None:
                pd.DataFrame(history).to_csv(folder / 'optimization.csv', index=False)
                (folder / 'failure.json').write_text(json.dumps(dict(step=step, reason=str(error)), indent=2))
            raise
        optimizer.step()
        row['backward_seconds'] = timestamp() - backward_start
        row['seconds'] = elapsed + time.perf_counter() - started
        if folder is not None and step % log_every == 0:
            pd.DataFrame(history).to_csv(folder / 'optimization.csv', index=False)
    result = dict(initial_moments=initial_moments, best_moments=best_moments,
                  theta=best_theta, best_J=best, best_step=best_step, history=history,
                  penalty=penalty, steps=steps, checkpoints=snapshots, mass_mode=mass_mode,
                  assignment_rank=assignment_rank, factor_seed=factor_seed,
                  assignment_input=assignment_input,
                  assignment_encoder=assignment_encoder,
                  solver_mode=solver_mode, feature_control=feature_control,
                  assignment_parameters=sum(parameter.numel() for parameter in parameters))
    if folder is not None and save_assignment:
        if feature_control != 'joint':
            torch.save(dict(labels=fixed_labels.cpu(), mass=fixed_mass.cpu(), feature_control=feature_control),
                       folder / 'fixed_targets.pt')
        if feature_control == 'direct':
            torch.save(dict(centers=best_parameters[0].cpu(), labels=fixed_labels.cpu(),
                            mass=fixed_mass.cpu(), step=best_step), folder / 'best_free_features.pt')
        elif assignment_rank is None:
            torch.save(best_parameters[0].cpu(), folder / 'best_assignment_logits.pt')
        elif assignment_input != 'node':
            saved = dict(v=best_parameters[-1].cpu(), assignment=assignment.cpu(), mixing=mixing,
                         rank=assignment_rank, assignment_input=assignment_input, factor_seed=factor_seed,
                         step=best_step, assignment_encoder=assignment_encoder)
            if assignment_encoder == 'mlp':
                saved.update(encoder_parameters=[p.cpu() for p in best_parameters[:-1]], encoder_hidden=encoder_hidden)
            else:
                saved['weight'] = best_parameters[0].cpu()
            torch.save(saved, folder / 'best_assignment_encoder.pt')
        else:
            torch.save(dict(u=best_parameters[0].cpu(), v=best_parameters[1].cpu(),
                            assignment=assignment.cpu(), mixing=mixing, rank=assignment_rank,
                            factor_seed=factor_seed, step=best_step), folder / 'best_assignment_factors.pt')
        if best_dual is not None:
            torch.save(best_dual.cpu(), folder / 'best_assignment_column_dual.pt')
    return result


@torch.no_grad()
def classification(z, y, mask, theta):
    logits = augmented(z[mask]) @ theta.T
    return 100 * float((logits.argmax(1) == y[mask]).double().mean()), float(F.cross_entropy(logits, y[mask]))


def select_checkpoint(table):
    candidates = table[table.checkpoint_step.notna()]
    return candidates.sort_values(['gcn_val', 'checkpoint_step'], ascending=[False, True]).iloc[[0]]


def load_ce_snapshots(directory):
    snapshots = {}
    for path in sorted((Path(directory) / 'checkpoints').glob('step_*.pt')):
        snapshot = torch.load(path, map_location='cpu', weights_only=False)
        step = int(path.stem.split('_')[-1])
        if snapshot['step'] != step or not bool(torch.isfinite(snapshot['moments']).all()):
            raise ValueError(f'Invalid saved checkpoint: {path}')
        snapshots[step] = snapshot
    if 0 not in snapshots:
        raise ValueError('Recovery requires saved checkpoint zero')
    return snapshots


def run_soft_ce(previous_run, output_dir=None, penalty=3e-5, steps=300, lr=.01,
                 chunk_size=4096, inner_max_iter=2000, inner_tol=1e-7,
                 cg_max_iter=512, cg_rtol=1e-6, save_assignment=True,
                 data_dir='/content/data/', device='cuda', checkpoint_steps=(),
                 mass_mode='free', balance_steps=300, balance_tol=1e-8,
                 balance_backend='cached', balance_cg_steps=512, balance_cg_rtol=1e-7,
                 outer_chunk_size=65536, log_every=10, checkpoint_source=None,
                 assignment_rank=None, factor_seed=0, assignment_input='node',
                 assignment_encoder='linear', encoder_hidden=64, solver_mode='exact',
                 tracking_inner_steps=2, tracking_cg_steps=8, tracking_refresh=20):
    previous_run = Path(previous_run)
    prior = json.loads((previous_run / 'config.json').read_text())
    original, source = prior['original'], Path(prior['source'])
    if original['dataset'] != 'arxiv' or prior.get('outer_loss') != 'ce':
        raise ValueError('Require the previous Arxiv ridge-inner / CE-outer run')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    train, mask, validation, testing, h = _prepare_dataset('arxiv', data_dir, device)
    adj = train['adj'].to_sparse_csr()
    digest = array_digest(train['x'].cpu().numpy(), train['y'].cpu().numpy(), adj.crow_indices().cpu().numpy(),
                          adj.col_indices().cpu().numpy(), adj.values().cpu().numpy(), mask.cpu().numpy(),
                          validation[1].cpu().numpy(), testing[1].cpu().numpy())
    if digest != original['data_digest']:
        raise ValueError('Source graph or splits differ')
    teacher = torch.load(source / 'teacher_logits.pt', map_location=device, weights_only=True)
    baseline = torch.load(source / 'baseline_partition.pt', map_location=device, weights_only=False)
    assignment = baseline['assignment']
    if (array_digest(teacher.cpu().numpy()) != prior['logits_digest'] or
            array_digest(assignment.cpu().numpy()) != prior['assignment_digest']):
        raise ValueError('Source teacher or partition differs')
    q = (teacher / original['params']['T']).softmax(1).double()
    reference = torch.load(previous_run / 'reference.pt', map_location=device, weights_only=False)
    transform = FeatureTransform(**reference['transform'])
    z = transform(h.double())
    options = dict(penalty=penalty, steps=steps, lr=lr, mixing=prior['mixing'], chunk_size=chunk_size,
                   inner_max_iter=inner_max_iter, inner_tol=inner_tol, cg_max_iter=cg_max_iter,
                   cg_rtol=cg_rtol, save_assignment=save_assignment,
                   checkpoint_steps=sorted(set(checkpoint_steps)), mass_mode=mass_mode,
                   balance_steps=balance_steps, balance_tol=balance_tol, balance_backend=balance_backend,
                   balance_cg_steps=balance_cg_steps, balance_cg_rtol=balance_cg_rtol,
                   outer_chunk_size=outer_chunk_size, log_every=log_every)
    if solver_mode != 'exact':
        options.update(solver_mode=solver_mode, tracking_inner_steps=tracking_inner_steps,
                       tracking_cg_steps=tracking_cg_steps, tracking_refresh=tracking_refresh)
    if assignment_rank is not None:
        options.update(assignment_rank=assignment_rank, factor_seed=factor_seed)
    if assignment_input != 'node':
        options.update(assignment_input=assignment_input)
    if assignment_encoder != 'linear':
        options.update(assignment_encoder=assignment_encoder, encoder_hidden=encoder_hidden)
    methods = ['hard_baseline', 'soft_initial', 'ridge_optimized', 'ce_optimized']
    controls = {name: torch.load(previous_run / f'{name}.pt', map_location='cpu', weights_only=False)
                for name in methods}
    control_digests = {name: array_digest(*(saved[key].numpy() for key in ('x', 'y', 'mass')))
                       for name, saved in controls.items()}
    config = dict(previous_run=str(previous_run), prior=prior, options=options, control_digests=control_digests,
                  transform_digest=array_digest(z.cpu().numpy()),
                  revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
    recovered = None
    if checkpoint_source is not None:
        saved_config = json.loads((Path(checkpoint_source) / 'config.json').read_text())
        if saved_config['prior'] != prior or saved_config['options'] != options or not checkpoint_steps:
            raise ValueError('Recovery settings differ from the saved experiment')
        recovered = load_ce_snapshots(checkpoint_source)
        config.update(checkpoint_source=str(checkpoint_source), evaluation_only=True,
                      recovered_steps=sorted(recovered),
                      recovered_digest=array_digest(*(entry['moments'].numpy() for entry in recovered.values())))
    root = Path(output_dir or previous_run / 'ce_bilevel') / _fingerprint(config)
    root.mkdir(parents=True, exist_ok=True)
    (root / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    optimized_path = root / 'optimized.pt'
    if recovered is not None:
        optimized = dict(initial_moments=recovered[0]['moments'], checkpoints=recovered)
    elif optimized_path.exists():
        optimized = torch.load(optimized_path, map_location='cpu', weights_only=False)
    else:
        optimized = optimize_ce_assignment(z, q, assignment, folder=root, **options)
        torch.save(optimized, optimized_path)
    old_initial = torch.load(previous_run / 'optimized.pt', map_location='cpu', weights_only=False)['initial_moments']
    if mass_mode == 'free' and not torch.allclose(optimized['initial_moments'], old_initial, atol=1e-10, rtol=1e-8):
        raise ValueError('Initialization differs from the previous experiment')
    state = {key: value.cpu() if torch.is_tensor(value) else value for key, value in reference['transform'].items()}
    checkpoint_numbers = {}
    if checkpoint_steps:
        representatives = {}
        for step, snapshot in sorted(optimized['checkpoints'].items()):
            method = f'ce_step_{step:06d}'
            representatives[method] = snapshot['moments']
            checkpoint_numbers[method] = step
    else:
        representatives = {'ce_bilevel': optimized['best_moments']}
    for method, moments in representatives.items():
        centers, labels, mass = decode_moments(moments, z.shape[1])
        cx = (centers * state['scale'] + state['output_center'] + state['center']).float()
        controls[method] = dict(x=cx, y=labels.float(), mass=mass)
    summaries, student_tables = [], []
    settings = {key: original[key] for key in ('epochs', 'eval_every', 'hidden')}
    for method, saved in controls.items():
        cx, labels, mass = (saved[key].to(device) for key in ('x', 'y', 'mass'))
        fitted = solve_inner(transform(cx.double()), labels.double(), mass.double(), penalty,
                             max_iter=inner_max_iter, grad_tol=inner_tol, cg_max_iter=cg_max_iter)
        theta = fitted.pop('theta')
        if not fitted['inner_converged']:
            raise RuntimeError(f'{method}: evaluation CE head did not converge')
        ce, _ = outer_value_gradient(z, q, theta, outer_chunk_size)
        val, val_ce = classification(z, train['y'], validation[1], theta)
        test, test_ce = classification(z, train['y'], testing[1], theta) if prior['evaluate_test'] else (np.nan, np.nan)
        torch.save(dict(x=cx.cpu(), y=labels.cpu(), mass=mass.cpu(), theta=theta.cpu()), root / f'{method}.pt')
        student_path = root / f'{method}_students.csv'
        if student_path.exists():
            students = pd.read_csv(student_path)
        elif method in methods:
            students = pd.read_csv(previous_run / f'{method}_students.csv')
        else:
            records = []
            for seed in prior['final_seeds']:
                accuracy, test_accuracy, epoch = _train_student(
                    cx.float(), labels.float(), validation, original['params'], seed,
                    {**settings, 'loss_weighting': 'mass'}, counts=mass,
                    testing=testing if prior['evaluate_test'] else None)
                records.append(dict(seed=seed, val=100 * accuracy,
                                    test=100 * test_accuracy if test_accuracy is not None else np.nan, epoch=epoch))
                pd.DataFrame(records).to_csv(root / f'{method}_students.partial.csv', index=False)
            students = pd.DataFrame(records)
        if sorted(students.seed.tolist()) != sorted(prior['final_seeds']):
            raise ValueError('Cached GCN seeds differ')
        students.to_csv(student_path, index=False)
        student_tables.append(students.assign(method=method))
        summaries.append(dict(method=method, nodes=len(cx), penalty=penalty, teacher_ce=ce,
                              mass_mode=mass_mode if method in representatives else 'free',
                              assignment_rank=assignment_rank if method in representatives else np.nan,
                              assignment_input=assignment_input if method in representatives else 'control',
                              assignment_encoder=assignment_encoder if method in representatives else 'control',
                              mass_tv=float((mass - 1 / len(mass)).abs().sum() / 2),
                              mass_relative_residual=float((mass * len(mass) - 1).abs().max()),
                              linear_val=val, linear_test=test, linear_val_ce=val_ce, linear_test_ce=test_ce,
                              **fitted, head_norm=float(theta.norm()), effective_cells=float(1 / mass.square().sum()),
                              gcn_val=students.val.mean(), gcn_val_std=students.val.std(ddof=0),
                              gcn_test=students.test.mean(), gcn_test_std=students.test.std(ddof=0),
                              checkpoint_step=checkpoint_numbers.get(method, np.nan),
                              best_step=optimized['best_step'] if method == 'ce_bilevel' else np.nan,
                              output_dir=str(root)))
        pd.DataFrame(summaries).to_csv(root / 'summary.csv', index=False)
        pd.concat(student_tables).to_csv(root / 'students.csv', index=False)
    result = pd.DataFrame(summaries)
    if checkpoint_steps:
        selected = select_checkpoint(result)
        selected.to_csv(root / 'selected_checkpoint.csv', index=False)
        result['selected_by_gcn_val'] = result.method.eq(selected.method.iloc[0])
        result.to_csv(root / 'summary.csv', index=False)
    return result


def evaluate_saved_ce_checkpoints(failed_run, output_dir=None, data_dir='/content/data/', device='cuda'):
    failed_run = Path(failed_run)
    config = json.loads((failed_run / 'config.json').read_text())
    options = dict(config['options'])
    options.pop('mixing')
    return run_soft_ce(config['previous_run'], output_dir=output_dir or failed_run / 'recovered_evaluation',
                        data_dir=data_dir, device=device, checkpoint_source=failed_run, **options)
