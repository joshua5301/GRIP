import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.anil_probe import fit_probe
from src.anil_representation import accuracy, sample_support
from src.grid_search import GridStudy
from src.risk_experiment import _fingerprint, _prepare_dataset


class FeatureTransform:
    def __init__(self, center, matrix, output_center, scale, kind, eps):
        self.center, self.matrix = center, matrix
        self.output_center, self.scale, self.kind, self.eps = output_center, scale, kind, eps

    def __call__(self, z):
        z = z - self.center
        if self.matrix is not None:
            z = z @ self.matrix
        if self.kind == 'l2':
            z = z / z.norm(dim=1, keepdim=True).clamp_min(self.eps)
        return (z - self.output_center) / self.scale

    def state_dict(self):
        return {k: v.cpu() if torch.is_tensor(v) else v for k, v in vars(self).items()}


class TransformedMap:
    def __init__(self, mapping, transform):
        self.mapping, self.transform = mapping, transform

    def __call__(self, x):
        return self.transform(self.mapping(x))


@torch.no_grad()
def fit_transform(features, kind='rms', power=.5, ridge=.01, eps=1e-12):
    if kind not in ('rms', 'l2', 'whiten') or not 0 <= power <= 1 or ridge <= 0 or eps <= 0:
        raise ValueError('Invalid NTK transform settings')
    center = features.mean(0)
    z = features - center
    matrix = None
    if kind == 'whiten':
        covariance = z.T @ z / len(z)
        values, vectors = torch.linalg.eigh(covariance)
        stabilizer = ridge * values.max().clamp_min(eps)
        factors = (values.clamp_min(0) + stabilizer).pow(-power / 2)
        matrix = (vectors * factors) @ vectors.T
        z = z @ matrix
    if kind == 'l2':
        z = z / z.norm(dim=1, keepdim=True).clamp_min(eps)
    output_center = z.mean(0)
    scale = (z - output_center).square().sum(1).mean().sqrt().clamp_min(eps)
    transform = FeatureTransform(center, matrix, output_center, scale, kind, eps)
    return transform(features), transform


def run_transform_probes(risk_results, space, output_dir, search_seeds=(0, 1, 2),
                         final_seeds=tuple(range(100, 110)), steps=500, polish_steps=300,
                         data_dir='/content/data/', device='cuda'):
    if set(risk_results.dataset) != {'cora'}:
        raise ValueError('This probe protocol currently requires Cora')
    train, mask, validation, testing, _ = _prepare_dataset('cora', data_dir, device)
    y, ids = train['y'], mask.nonzero().flatten()
    val_ids, test_ids = validation[1].nonzero().flatten(), testing[1].nonzero().flatten()
    classes = int(y.max()) + 1
    config = dict(runs=sorted(risk_results.output_dir.unique()), space=space,
                  search_seeds=list(search_seeds), final_seeds=list(final_seeds),
                  steps=steps, polish_steps=polish_steps)
    root = Path(output_dir) / ('probes_' + _fingerprint(config))
    root.mkdir(parents=True, exist_ok=True)
    (root / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    rows, records = [], []
    for run in risk_results.drop_duplicates('output_dir').itertuples():
        path, = list(Path(run.output_dir).glob('cora_*/transformed_features.pt'))
        saved = torch.load(path, map_location='cpu', weights_only=False)
        z = saved['features'].to(device).double()
        folder = root / _fingerprint(run.output_dir)
        folder.mkdir(exist_ok=True)

        def evaluate(params, seed, testing_enabled=False):
            support = sample_support(ids, y, len(ids), seed)
            fitted = fit_probe(z[support], y[support], classes, seed, **params,
                               steps=steps, polish_steps=polish_steps)
            head = (fitted['weight'], fitted['bias'])
            val, ce = accuracy(z, y, val_ids, head)
            record = dict(seed=seed, val=100 * val, val_ce=ce,
                          head_norm=float(fitted['weight'].norm()), grad=fitted['grad_after'],
                          converged=fitted['converged'])
            if testing_enabled:
                test, test_ce = accuracy(z, y, test_ids, head)
                record.update(test=100 * test, test_ce=test_ce)
            return record

        study = GridStudy(space, folder)

        def objective(trial):
            values = [evaluate(trial.params, seed) for seed in search_seeds]
            trial.set_user_attr('runs', values)
            return -np.mean([v['val_ce'] for v in values])

        study.optimize(objective)
        study.trials_dataframe().to_csv(folder / 'trials.csv', index=False)
        params = study.best_trial.params
        final_path = folder / 'final.csv'
        if final_path.exists():
            final = pd.read_csv(final_path)
        else:
            final = pd.DataFrame([evaluate(params, seed, True) for seed in final_seeds])
            final.to_csv(final_path, index=False)
        records.append(final.assign(transform=run.transform))
        rows.append(dict(transform=run.transform, **params, val=final.val.mean(),
                         test=final.test.mean(), val_ce=final.val_ce.mean(), test_ce=final.test_ce.mean(),
                         head_norm=final.head_norm.mean(), converged_fraction=final.converged.mean(),
                         grad_max=final.grad.max(), output_dir=str(root)))
        pd.DataFrame(rows).to_csv(root / 'summary.csv', index=False)
        pd.concat(records).to_csv(root / 'probes.csv', index=False)
    return pd.DataFrame(rows)
