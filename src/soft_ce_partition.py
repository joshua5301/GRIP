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
def conjugate_gradient(multiply, rhs, diagonal, rtol=1e-6, atol=1e-12, max_iter=512):
    solution = torch.zeros_like(rhs)
    residual = rhs.clone()
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
            multiply, diagonal = hessian_operator(x, labels, mass, theta, penalty)
            direction, diagnostic = conjugate_gradient(multiply, gradient, diagonal,
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


def optimize_ce_assignment(z, q, assignment, penalty=3e-5, steps=300, lr=.01,
                           mixing=.05, chunk_size=4096, inner_max_iter=2000,
                           inner_tol=1e-7, cg_max_iter=512, cg_rtol=1e-6,
                           save_assignment=True, folder=None, checkpoint_steps=(),
                           mass_mode='free', balance_steps=300, balance_tol=1e-8,
                           balance_backend='cached', balance_cg_steps=512, balance_cg_rtol=1e-7,
                           outer_chunk_size=65536, log_every=10):
    if steps < 0 or any(not np.isfinite(v) or v <= 0 for v in
                        (penalty, lr, chunk_size, inner_max_iter, inner_tol, cg_max_iter, cg_rtol)):
        raise ValueError('Require positive finite solver settings and nonnegative steps')
    if mass_mode not in ('free', 'uniform') or balance_steps < 1 or not 0 < balance_tol < 1:
        raise ValueError('Invalid mass constraint settings')
    if (balance_backend not in ('cached', 'chunked') or min(balance_cg_steps, outer_chunk_size, log_every) < 1
            or not 0 < balance_cg_rtol < 1):
        raise ValueError('Invalid performance settings')
    checkpoints = set(checkpoint_steps)
    if any(not isinstance(step, (int, np.integer)) or not 0 <= step <= steps for step in checkpoints):
        raise ValueError('Checkpoint steps must be integers within the optimization budget')
    if checkpoints:
        checkpoints.update((0, steps))
    snapshots = {}
    material = make_material(z, q)
    full_features = augmented(z)
    logits = initial_logits(assignment, int(assignment.max()) + 1, mixing).requires_grad_()
    optimizer = torch.optim.Adam([logits], lr=lr, eps=1e-12, foreach=False)
    folder = Path(folder) if folder is not None else None
    if folder is not None:
        folder.mkdir(parents=True, exist_ok=True)
    history, theta, best, dual = [], None, float('inf'), None
    started = time.perf_counter()
    def timestamp():
        if z.is_cuda:
            torch.cuda.synchronize(z.device)
        return time.perf_counter()
    for step in trange(steps + 1, desc='CE inner + CE outer'):
        tick = timestamp()
        optimizer.zero_grad(set_to_none=True)
        balance = dict(balance_iterations=0, row_residual=np.nan, column_residual=np.nan)
        if mass_mode == 'uniform':
            if balance_backend == 'cached':
                moments, dual, diagnostic = CachedBalancedMoments.apply(
                    logits, material, chunk_size, balance_steps, balance_tol, dual,
                    balance_cg_steps, balance_cg_rtol)
            else:
                moments, dual, diagnostic = BalancedMoments.apply(
                    logits, material, chunk_size, balance_steps, balance_tol, dual)
            balance = dict(balance_iterations=int(diagnostic[0]), row_residual=float(diagnostic[1]),
                           column_residual=float(diagnostic[2]))
        else:
            moments = AssignmentMoments.apply(logits, material, chunk_size)
        if not bool(torch.isfinite(moments).all()) or bool((moments[:, 0] <= 0).any()):
            raise FloatingPointError('Nonfinite moments or empty soft cell')
        centers, labels, mass = decode_moments(moments.detach(), z.shape[1])
        after_assignment = timestamp()
        fitted = solve_inner(centers, labels, mass, penalty, theta, inner_max_iter,
                             inner_tol, cg_max_iter=cg_max_iter)
        theta = fitted.pop('theta')
        after_inner = timestamp()
        value, outer_gradient = outer_value_gradient(z, q, theta, outer_chunk_size, full_features)
        after_outer = timestamp()
        row = dict(step=step, J=value, **fitted, **balance, cg_iterations=0, cg_residual=np.nan,
                   cg_relative_residual=np.nan, cg_converged=False,
                   min_mass=float(mass.min()), max_mass=float(mass.max()),
                   effective_cells=float(1 / mass.square().sum()))
        row.update(assignment_seconds=after_assignment - tick, inner_seconds=after_inner - after_assignment,
                   outer_seconds=after_outer - after_inner, implicit_seconds=0., backward_seconds=0.)
        failure = None
        if not fitted['inner_converged'] or not np.isfinite(value):
            failure = 'Inner CE did not converge; increase inner_max_iter or inspect inner_tol'
        else:
            if step == 0:
                initial_moments = moments.detach().cpu()
                scale = max(value, 1e-12)
            if value < best:
                best, best_step = value, step
                best_moments, best_theta = moments.detach().cpu(), theta.cpu()
                if save_assignment:
                    best_logits = logits.detach().clone()
                    best_dual = dual.clone() if dual is not None else None
            if step < steps:
                implicit_start = timestamp()
                multiply, diagonal = hessian_operator(augmented(centers), labels, mass, theta, penalty)
                vector, diagnostic = conjugate_gradient(multiply, outer_gradient, diagonal,
                                                         rtol=cg_rtol, max_iter=cg_max_iter)
                row.update(diagnostic)
                row['implicit_seconds'] = timestamp() - implicit_start
                if not diagnostic['cg_converged']:
                    failure = 'Implicit Hessian solve did not converge; increase cg_max_iter'
        row.update(best_J=best, seconds=time.perf_counter() - started,
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
                            theta=theta.cpu().clone(), teacher_ce=value, inner_grad_max=fitted['inner_grad_max'])
            snapshots[step] = snapshot
            if folder is not None:
                checkpoint_dir = folder / 'checkpoints'
                checkpoint_dir.mkdir(exist_ok=True)
                torch.save(snapshot, checkpoint_dir / f'step_{step:06d}.pt')
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
        row['seconds'] = time.perf_counter() - started
        if folder is not None and step % log_every == 0:
            pd.DataFrame(history).to_csv(folder / 'optimization.csv', index=False)
    result = dict(initial_moments=initial_moments, best_moments=best_moments,
                  theta=best_theta, best_J=best, best_step=best_step, history=history,
                  penalty=penalty, steps=steps, checkpoints=snapshots, mass_mode=mass_mode)
    if folder is not None and save_assignment:
        torch.save(best_logits.cpu(), folder / 'best_assignment_logits.pt')
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


def run_soft_ce(previous_run, output_dir=None, penalty=3e-5, steps=300, lr=.01,
                 chunk_size=4096, inner_max_iter=2000, inner_tol=1e-7,
                 cg_max_iter=512, cg_rtol=1e-6, save_assignment=True,
                 data_dir='/content/data/', device='cuda', checkpoint_steps=(),
                 mass_mode='free', balance_steps=300, balance_tol=1e-8,
                 balance_backend='cached', balance_cg_steps=512, balance_cg_rtol=1e-7,
                 outer_chunk_size=65536, log_every=10):
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
    methods = ['hard_baseline', 'soft_initial', 'ridge_optimized', 'ce_optimized']
    controls = {name: torch.load(previous_run / f'{name}.pt', map_location='cpu', weights_only=False)
                for name in methods}
    control_digests = {name: array_digest(*(saved[key].numpy() for key in ('x', 'y', 'mass')))
                       for name, saved in controls.items()}
    config = dict(previous_run=str(previous_run), prior=prior, options=options, control_digests=control_digests,
                  transform_digest=array_digest(z.cpu().numpy()),
                  revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
    root = Path(output_dir or previous_run / 'ce_bilevel') / _fingerprint(config)
    root.mkdir(parents=True, exist_ok=True)
    (root / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    optimized_path = root / 'optimized.pt'
    if optimized_path.exists():
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
