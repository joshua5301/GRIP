import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from src.condensation_diagnostics import save_json
from src.node_distances import array_digest
from src.risk_experiment import _fingerprint
from src.soft_ce_partition import head_gradient, hessian_operator, solve_head_system, solve_inner
from src.soft_ridge_partition import augmented, decode_moments
from src.stationarity_risk import head_objective


@torch.no_grad()
def path_curvature(x, labels, mass, reference, optimum, penalty, quadrature=16):
    delta = reference - optimum
    points, weights = np.polynomial.legendre.leggauss(quadrature)
    image = torch.zeros_like(delta)
    for point, weight in zip((points + 1) / 2, weights / 2):
        multiply, _ = hessian_operator(x, labels, mass, optimum + float(point) * delta, penalty)
        image.add_(multiply(delta), alpha=float(weight))
    return image


def measure_curvature(moments, reference, penalty, head_tol=1e-8, cg_rtol=1e-8,
                      cg_steps=2048, quadrature=16):
    centers, labels, mass = decode_moments(moments, reference.shape[1] - 1)
    fitted = solve_inner(centers, labels, mass, penalty, initial=reference,
                         max_iter=3000, grad_tol=head_tol, cg_max_iter=cg_steps)
    if not fitted['inner_converged']:
        raise RuntimeError('Diagnostic head did not converge')
    with torch.no_grad():
        optimum = fitted['theta']
        x = augmented(centers)
        gradient = head_gradient(x, labels, mass, reference, penalty)
        optimum_gradient = head_gradient(x, labels, mass, optimum, penalty)
        newton, diagnostic = solve_head_system(x, labels, mass, reference, penalty, gradient,
                                               rtol=cg_rtol, max_iter=cg_steps)
        if not diagnostic['cg_converged']:
            raise RuntimeError(f'Hessian solve failed: {diagnostic}')
        delta = reference - optimum
        norm, distance = gradient.norm(), delta.norm()
        multiply, _ = hessian_operator(x, labels, mass, reference, penalty)
        difference = gradient - optimum_gradient
        integrated = path_curvature(x, labels, mass, reference, optimum, penalty, quadrature)
        coarse = path_curvature(x, labels, mass, reference, optimum, penalty, max(2, quadrature // 2))
        gap = head_objective(x, labels, mass, reference, penalty) - head_objective(x, labels, mass, optimum, penalty)
        row = dict(
            gradient_norm=float(norm), head_distance=float(distance), head_norm=float(reference.norm()),
            relative_head_distance=float(distance / reference.norm().clamp_min(1e-30)),
            distance_bound=float(norm / penalty), optimum_gradient_norm=float(optimum_gradient.norm()),
            optimum_gradient_max=float(optimum_gradient.abs().max()),
            newton_distance=float(newton.norm()),
            newton_relative_error=float((newton - delta).norm() / distance.clamp_min(1e-30)),
            newton_cosine=float((newton * delta).sum() / (newton.norm() * distance).clamp_min(1e-30)),
            gradient_direction_curvature=float((gradient * multiply(gradient)).sum() / norm.square().clamp_min(1e-30)),
            displacement_direction_curvature=float((delta * multiply(delta)).sum() / distance.square().clamp_min(1e-30)),
            path_direction_curvature=float((delta * difference).sum() / distance.square().clamp_min(1e-30)),
            path_identity_error=float((integrated - difference).norm() / difference.norm().clamp_min(1e-30)),
            quadrature_change=float((integrated - coarse).norm() / difference.norm().clamp_min(1e-30)),
            objective_gap=float(gap), quadratic_gap_prediction=float((gradient * newton).sum() / 2),
            **diagnostic)
    return row


def run_curvature_audit(previous_run, initializations=('kmeans', 'risk'), steps=None,
                        head_tol=1e-8, cg_rtol=1e-8, cg_steps=2048, quadrature=16, device='cuda'):
    if min(head_tol, cg_rtol, cg_steps) <= 0 or quadrature < 4:
        raise ValueError('Require positive solver tolerances and at least four quadrature points')
    root = Path(previous_run)
    options = dict(head_tol=head_tol, cg_rtol=cg_rtol, cg_steps=cg_steps, quadrature=quadrature)
    output = root / 'curvature' / _fingerprint(dict(version=1, **options))
    output.mkdir(parents=True, exist_ok=True)
    save_json(options, output / 'config.json')
    rows = []
    for name in initializations:
        folder = root / name
        config = json.loads((folder / 'config.json').read_text())
        reference = torch.load(folder / 'fixed_head.pt', map_location=device, weights_only=False)['theta'].double()
        search = pd.read_csv(folder / 'search.csv').set_index('step')
        paths = sorted((folder / 'checkpoints').glob('step_*.pt'))
        available = {int(path.stem.split('_')[1]) for path in paths}
        if steps is not None and set(steps) - available:
            raise ValueError(f'Missing {name} checkpoints: {set(steps) - available}')
        for path in tqdm(paths, desc=f'{name}: curvature audit'):
            step = int(path.stem.split('_')[1])
            if steps is not None and step not in steps:
                continue
            moments = torch.load(path, map_location=device, weights_only=False)['moments'].double()
            digest = array_digest(moments.cpu().numpy(), reference.cpu().numpy())
            cache = output / f'{name}_{step:06d}.json'
            row = json.loads(cache.read_text()) if cache.exists() else None
            if row is None or row['digest'] != digest or row['penalty'] != config['penalty']:
                row = dict(initialization=name, step=step, digest=digest, penalty=config['penalty'],
                           **measure_curvature(moments, reference, config['penalty'], **options))
                save_json(row, cache)
            if step in search.index:
                row.update(gcn_val=float(search.loc[step, 'val']), gcn_val_std=float(search.loc[step, 'val_std']))
            rows.append(row)
            pd.DataFrame(rows).to_csv(output / 'curvature.csv', index=False)
    if not rows:
        raise ValueError('No saved checkpoints found')
    return pd.DataFrame(rows), output
