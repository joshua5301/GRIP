import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric import seed_everything

from src.anil_representation import accuracy, sample_support
from src.grid_search import GridStudy
from src.node_distances import array_digest
from src.risk_experiment import _fingerprint, _prepare_dataset


def normalize_features(features, train_ids):
    center = features[train_ids].mean(0)
    scale = (features[train_ids] - center).square().sum(1).mean().sqrt().clamp_min(1e-12)
    return (features - center) / scale, center, scale


def fit_probe(z, y, classes, seed, lr, penalty, steps=2000, polish_steps=300,
              grad_tol=1e-6, record_every=100):
    seed_everything(seed)
    head = torch.nn.Linear(z.shape[1], classes, device=z.device, dtype=z.dtype)
    optimizer = torch.optim.SGD(head.parameters(), lr=lr)

    def objective():
        return F.cross_entropy(head(z), y) + penalty * head.weight.square().sum() / 2

    def diagnostics():
        loss = objective()
        gradient = torch.autograd.grad(loss, tuple(head.parameters()))
        return float(loss.detach()), max(float(g.abs().max()) for g in gradient)

    history = []
    for step in range(steps + 1):
        if step % record_every == 0 or step == steps:
            loss, grad = diagnostics()
            if not np.isfinite(loss + grad):
                raise FloatingPointError('Nonfinite linear probe objective or gradient')
            history.append(dict(phase='sgd', step=step, objective=loss, grad_inf=grad))
            if grad <= grad_tol or step == steps:
                break
        optimizer.zero_grad(set_to_none=True)
        objective().backward()
        optimizer.step()
    before_weight = head.weight.detach().clone()
    before_bias = head.bias.detach().clone()
    before_loss, before_grad = diagnostics()
    if polish_steps:
        optimizer = torch.optim.LBFGS(head.parameters(), lr=1., max_iter=polish_steps,
                                     tolerance_grad=grad_tol, tolerance_change=1e-12,
                                     line_search_fn='strong_wolfe')

        def closure():
            optimizer.zero_grad(set_to_none=True)
            loss = objective()
            loss.backward()
            return loss

        optimizer.step(closure)
        after_loss, after_grad = diagnostics()
        if not np.isfinite(after_loss + after_grad) or after_loss > before_loss + 1e-10:
            raise FloatingPointError('L-BFGS did not produce a finite nonincreasing objective')
        iterations = optimizer.state[head.weight].get('n_iter', 0)
        history.append(dict(phase='lbfgs', step=iterations, objective=after_loss, grad_inf=after_grad))
    else:
        after_loss, after_grad = before_loss, before_grad
    return dict(weight=head.weight.detach(), bias=head.bias.detach(),
                before_weight=before_weight, before_bias=before_bias, history=history,
                objective_before=before_loss, objective_after=after_loss,
                grad_before=before_grad, grad_after=after_grad, converged=after_grad <= grad_tol)


def run_probe_grid(source, space, output_dir=None, support_sizes=(35, 70, 140),
                   search_seeds=(0, 1, 2), final_seeds=tuple(range(100, 110)),
                   steps=2000, polish_steps=300, grad_tol=1e-6, select_by='val_ce',
                   data_dir='/content/data/', device='cuda'):
    if select_by not in ('val', 'val_ce') or set(space) != {'lr', 'penalty'}:
        raise ValueError('Require val or val_ce selection and lr/penalty grid')
    if not search_seeds or not final_seeds or min(steps, polish_steps) < 0:
        raise ValueError('Require seeds and nonnegative iteration counts')
    source = Path(source)
    original = json.loads((source / 'config.json').read_text())
    train, mask, validation, testing, _ = _prepare_dataset('cora', data_dir, device)
    y, ids = train['y'], mask.nonzero().flatten()
    val_ids, test_ids = validation[1].nonzero().flatten(), testing[1].nonzero().flatten()
    adjacency = train['adj'].to_sparse_coo().coalesce()
    digest = array_digest(train['x'].cpu().numpy(), y.cpu().numpy(), adjacency.indices().cpu().numpy(),
                          adjacency.values().cpu().numpy(), ids.cpu().numpy(), val_ids.cpu().numpy(),
                          test_ids.cpu().numpy())
    if original['data'] != digest:
        raise ValueError('Saved encoder data/splits do not match the current Cora dataset')
    loaded = {m: torch.load(source / m / 'features.pt', map_location='cpu', weights_only=False)
              for m in ('supervised', 'anil')}
    config = dict(source=str(source), space=space, support_sizes=list(support_sizes),
                  search_seeds=list(search_seeds), final_seeds=list(final_seeds), steps=steps,
                  polish_steps=polish_steps, grad_tol=grad_tol, select_by=select_by, data=digest,
                  features={m: array_digest(v['features'].numpy()) for m, v in loaded.items()},
                  revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
    root = Path(output_dir or source / 'probe_grid') / _fingerprint(config)
    root.mkdir(parents=True, exist_ok=True)
    (root / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    rows, runs = [], []
    classes = int(y.max()) + 1
    for method, saved in loaded.items():
        z, center, scale = normalize_features(saved['features'].to(device).double(), ids)
        torch.save(dict(center=center.cpu(), scale=scale.cpu()), root / f'{method}_normalization.pt')
        for size in support_sizes:
            folder = root / method / str(size)
            folder.mkdir(parents=True, exist_ok=True)

            def evaluate(params, seed, include_test=False):
                support = sample_support(ids, y, size, seed)
                fitted = fit_probe(z[support], y[support], classes, seed, **params,
                                   steps=steps, polish_steps=polish_steps, grad_tol=grad_tol)
                val, ce = accuracy(z, y, val_ids, (fitted['weight'], fitted['bias']))
                pre_val, pre_ce = accuracy(z, y, val_ids, (fitted['before_weight'], fitted['before_bias']))
                record = dict(seed=seed, val=100 * val, val_ce=ce, before_val=100 * pre_val,
                              before_val_ce=pre_ce, **{k: fitted[k] for k in
                              ('objective_before', 'objective_after', 'grad_before', 'grad_after', 'converged')})
                if include_test:
                    test, test_ce = accuracy(z, y, test_ids, (fitted['weight'], fitted['bias']))
                    record.update(test=100 * test, test_ce=test_ce)
                    pd.DataFrame(fitted['history']).to_csv(folder / f'history_{seed}.csv', index=False)
                    torch.save(dict(weight=fitted['weight'].cpu(), bias=fitted['bias'].cpu(),
                                    support=support.cpu()), folder / f'head_{seed}.pt')
                return record

            study = GridStudy(space, folder)

            def objective(trial):
                results = [evaluate(trial.params, seed) for seed in search_seeds]
                trial.set_user_attr('runs', results)
                score = float(np.mean([r[select_by] for r in results]))
                return -score if select_by == 'val_ce' else score

            study.optimize(objective)
            study.trials_dataframe().to_csv(folder / 'trials.csv', index=False)
            params = study.best_trial.params
            (folder / 'best.json').write_text(json.dumps(params, indent=2), encoding='utf-8')
            final_path = folder / 'final.csv'
            if final_path.exists():
                final = pd.read_csv(final_path)
            else:
                final = pd.DataFrame([evaluate(params, seed, True) for seed in final_seeds])
                final.to_csv(final_path, index=False)
            runs.append(final.assign(method=method, support_size=size))
            rows.append(dict(method=method, support_size=size, **params, val=final.val.mean(),
                             val_std=final.val.std(ddof=0), test=final.test.mean(),
                             test_std=final.test.std(ddof=0), val_ce=final.val_ce.mean(),
                             test_ce=final.test_ce.mean(), before_val_ce=final.before_val_ce.mean(),
                             converged_fraction=final.converged.mean(), grad_max=final.grad_after.max(),
                             encoder_episode=saved['epoch'], output_dir=str(root)))
            pd.DataFrame(rows).to_csv(root / 'summary.csv', index=False)
            pd.concat(runs).to_csv(root / 'probes.csv', index=False)
    return pd.DataFrame(rows)
