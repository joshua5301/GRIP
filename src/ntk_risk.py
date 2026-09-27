import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric import seed_everything

from src.convex_representatives import cluster_softmax, convex_features
from src.grid_search import GridStudy
from src.node_distances import array_digest
from src.risk_experiment import _fingerprint, _prepare_dataset, _train_student
from src.risk_partition import risk_partition
from src.teacher import fit_logistic, get_kernel_features
from src.utils import BUDGET


def tangent_kernel(left, right):
    product = left.norm(dim=1)[:, None] * right.norm(dim=1)[None, :]
    cosine = (left @ right.T / product.clamp_min(1e-30)).clamp(-1, 1)
    safe = cosine.clamp(-1 + 1e-10, 1 - 1e-10)
    angle = safe.acos()
    value = product * ((1 - safe.square()).sqrt() + 2 * (torch.pi - angle) * safe) / torch.pi
    value = torch.where(cosine >= 1 - 1e-10, 2 * product, value)
    return torch.where(cosine <= -1 + 1e-10, torch.zeros_like(value), value)


@torch.no_grad()
def graph_kernel(x, propagation):
    propagated = torch.sparse.mm(propagation, x) / np.sqrt(x.shape[1])
    base = tangent_kernel(propagated, propagated)
    left = torch.sparse.mm(propagation, base)
    kernel = torch.sparse.mm(propagation, left.T).T
    return (kernel + kernel.T) / 2


@torch.no_grad()
def spectral_features(kernel):
    values, vectors = torch.linalg.eigh(kernel)
    if float(values.min()) < -1e-8 * max(float(values.max()), 1e-30):
        raise FloatingPointError('NTK is not positive semidefinite')
    return vectors * values.clamp_min(0).sqrt()


def fit_representatives(x, propagation, kernel, assignment, steps=1000, lr=.05):
    x, propagation = x.double(), propagation.double().to_sparse_coo().coalesce()
    assignment = assignment.to(x.device)
    clusters = int(assignment.max()) + 1
    counts = torch.bincount(assignment, minlength=clusters).to(x.dtype)
    if bool((counts == 0).any()):
        raise ValueError('Every cell must have a member')
    membership = F.one_hot(assignment, clusters).to(x.dtype) / counts
    target_norm = (membership * (kernel @ membership)).sum(0)
    support = torch.sparse.mm(propagation.transpose(0, 1), membership)
    propagated = torch.sparse.mm(propagation, x) / np.sqrt(x.shape[1])
    mass = counts / counts.sum()
    scale = kernel.diag().mean().clamp_min(1e-30)
    scores = torch.zeros(len(x), dtype=x.dtype, device=x.device, requires_grad=True)
    optimizer = torch.optim.Adam([scores], lr=lr)
    history, best_loss, best_weights = [], float('inf'), None
    for step in range(steps + 1):
        weights = cluster_softmax(scores, assignment, clusters)
        representatives = convex_features(x, assignment, weights, clusters)
        inputs = representatives / np.sqrt(x.shape[1])
        cross = (tangent_kernel(inputs, propagated) * support.T).sum(1)
        error = 2 * inputs.square().sum(1) - 2 * cross + target_norm
        loss = (mass * error).sum() / scale
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Nonfinite NTK reconstruction loss')
        value = float(loss.detach())
        if value < best_loss:
            best_loss, best_weights = value, weights.detach().clone()
        history.append(dict(step=step, relative_error=value, best_relative_error=best_loss))
        if step < steps:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return dict(x=convex_features(x, assignment, best_weights, clusters).float().cpu(),
                weights=best_weights.cpu(), reconstruction_history=history,
                reconstruction_initial=history[0]['relative_error'],
                reconstruction_final=best_loss, reconstruction_step=min(history, key=lambda r: r['relative_error'])['step'])


def run_ntk_risk(datasets, space, output_dir, modes=('raw_mean', 'raw_convex'),
                 data_dir='/content/data/', device='cuda', seed=0,
                 search_seeds=(0, 1, 2), final_seeds=tuple(range(100, 110)),
                 loss_weighting='uniform', epochs=1000, eval_every=10, hidden=256,
                 max_sweeps=30, reconstruction_steps=1000, reconstruction_lr=.05):
    if set(datasets) - {'cora', 'citeseer'}:
        raise ValueError('Full NTK is implemented for Cora and CiteSeer only')
    if set(modes) - {'raw_mean', 'raw_convex', 's2x_mean'}:
        raise ValueError('Unknown representative mode')
    if not modes or not search_seeds or not final_seeds:
        raise ValueError('Require modes and student seeds')
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    settings = dict(epochs=epochs, eval_every=eval_every, hidden=hidden, loss_weighting=loss_weighting)
    config = dict(datasets=datasets, space=space, modes=modes, seed=seed,
                  search_seeds=list(search_seeds), final_seeds=list(final_seeds), settings=settings,
                  max_sweeps=max_sweeps, reconstruction_steps=reconstruction_steps,
                  reconstruction_lr=reconstruction_lr, revision=revision, schema=1)
    root = Path(output_dir) / _fingerprint(config)
    root.mkdir(parents=True, exist_ok=True)
    (root / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    rows = []
    for name, ratios in datasets.items():
        train, mask, validation, testing, h = _prepare_dataset(name, data_dir, device)
        x = train['x'].double()
        propagation = train['adj'].double().to_sparse_coo().coalesce()
        digest = array_digest(x.cpu().numpy(), propagation.indices().cpu().numpy(),
                              propagation.values().cpu().numpy(), train['y'].cpu().numpy(),
                              mask.cpu().numpy(), validation[1].cpu().numpy(), testing[1].cpu().numpy())
        directory = root / f'{name}_{digest[:12]}'
        directory.mkdir(exist_ok=True)
        kernel_path = directory / 'kernel.pt'
        if kernel_path.exists():
            cached = torch.load(kernel_path, map_location=device, weights_only=False)
            kernel, features = cached['kernel'], cached['features']
        else:
            kernel = graph_kernel(x, propagation)
            features = spectral_features(kernel)
            torch.save(dict(kernel=kernel.cpu(), features=features.cpu()), kernel_path)
        feature_cache, logit_cache = {}, {}

        def labels(params):
            feature_key = (params['teacher_kernel'], params['basis'])
            logit_key = (*feature_key, params['gamma'])
            if logit_key not in logit_cache:
                if feature_key not in feature_cache:
                    seed_everything(seed)
                    feature_cache[feature_key] = get_kernel_features(h, *feature_key)
                teacher_features = feature_cache[feature_key]
                target = F.one_hot(train['y'][mask], int(train['y'].max()) + 1).double()
                weight = fit_logistic(teacher_features[mask], target, params['gamma'])
                logit_cache[logit_key] = (teacher_features @ weight).detach()
            return (logit_cache[logit_key] / params['T']).softmax(1)

        for ratio in ratios:
            clusters = BUDGET[(name, ratio)]
            folder = directory / str(ratio)
            folder.mkdir(exist_ok=True)

            def condensed(params, mode):
                key = {k: params[k] for k in ('teacher_kernel', 'basis', 'gamma', 'T', 'B')}
                path = folder / f'partition_{_fingerprint(key)}.pt'
                if path.exists():
                    result = torch.load(path, weights_only=False)
                else:
                    result = risk_partition(features, labels(params), clusters, params['B'],
                                            seed=seed, max_sweeps=max_sweeps, return_assignment=True)
                    assignment = result['assignment'].to(device)
                    weights = 1 / result['counts'].to(device)[assignment].double()
                    result['raw_mean'] = convex_features(x, assignment, weights, clusters).float().cpu()
                    result['s2x_mean'] = convex_features(h.double(), assignment, weights, clusters).float().cpu()
                    torch.save(result, path)
                if mode == 'raw_convex' and 'raw_convex' not in result:
                    fitted = fit_representatives(x, propagation, kernel, result['assignment'],
                                                reconstruction_steps, reconstruction_lr)
                    result['raw_convex'] = fitted.pop('x')
                    result.update(fitted)
                    torch.save(result, path)
                return result, path

            for mode in modes:
                mode_dir = folder / mode
                mode_dir.mkdir(exist_ok=True)
                study = GridStudy(space, mode_dir)

                def objective(trial):
                    data, _ = condensed(trial.params, mode)
                    scores = [_train_student(data[mode].to(device), data['y'].to(device), validation,
                                             trial.params, s, settings, counts=data['counts'])[0]
                              for s in search_seeds]
                    trial.set_user_attr('validation_seeds', scores)
                    return np.mean(scores)

                study.optimize(objective)
                best = study.best_trial
                study.trials_dataframe().to_csv(mode_dir / 'trials.csv', index=False)
                data, path = condensed(best.params, mode)
                final_path = mode_dir / 'final.csv'
                if final_path.exists():
                    final = pd.read_csv(final_path)
                else:
                    records = []
                    for s in final_seeds:
                        val, test, epoch = _train_student(data[mode].to(device), data['y'].to(device), validation,
                                                         best.params, s, settings, testing=testing, counts=data['counts'])
                        records.append(dict(seed=s, val=100 * val, test=100 * test, epoch=epoch))
                    final = pd.DataFrame(records)
                    final.to_csv(final_path, index=False)
                (mode_dir / 'best.json').write_text(json.dumps(dict(params=best.params, value=best.value,
                                                   partition_path=str(path)), indent=2), encoding='utf-8')
                row = dict(dataset=name, ratio=ratio, mode=mode, nodes=clusters, loss_weighting=loss_weighting,
                           **best.params, search_val=100 * best.value, final_val=final.val.mean(),
                           final_val_std=final.val.std(ddof=0), test_mean=final.test.mean(), test_std=final.test.std(ddof=0),
                           J_initial=data['history'][0], J_final=data['J'], converged=data['converged'],
                           partition_path=str(path), output_dir=str(root))
                if mode == 'raw_convex':
                    row.update({k: data[k] for k in ('reconstruction_initial', 'reconstruction_final', 'reconstruction_step')})
                    pd.DataFrame(data['reconstruction_history']).to_csv(mode_dir / 'reconstruction.csv', index=False)
                rows.append(row)
                pd.DataFrame(rows).to_csv(root / 'summary.csv', index=False)
    return pd.DataFrame(rows)
