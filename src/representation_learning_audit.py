import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.anil_probe import fit_probe
from src.anil_representation import sample_support
from src.directional_risk import directional_partition
from src.node_distances import array_digest
from src.ntk_nystrom import NystromGCN
from src.ntk_transforms import FeatureTransform, TransformedMap, fit_transform
from src.risk_experiment import _fingerprint, _prepare_dataset, _train_student
from src.risk_partition import risk_partition


def cell_means(z, q, assignment):
    n = torch.bincount(assignment).to(z.dtype)
    centers = z.new_zeros(len(n), z.shape[1]).index_add_(0, assignment, z) / n[:, None]
    labels = q.new_zeros(len(n), q.shape[1]).index_add_(0, assignment, q) / n[:, None]
    return centers, labels, n / n.sum()


def fit_soft_head(z, q, mass, penalty, max_iter=1000, grad_tol=1e-7):
    z, q, mass = z.double(), q.double(), mass.double()
    weight = z.new_zeros(q.shape[1], z.shape[1], requires_grad=True)
    bias = z.new_zeros(q.shape[1], requires_grad=True)
    optimizer = torch.optim.LBFGS([weight, bias], lr=1., max_iter=max_iter,
                                 tolerance_grad=grad_tol, tolerance_change=1e-13, line_search_fn='strong_wolfe')

    def objective():
        losses = -(q * F.linear(z, weight, bias).log_softmax(1)).sum(1)
        return (mass * losses).sum() + penalty * weight.square().sum() / 2

    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss = objective()
        loss.backward()
        return loss

    optimizer.step(closure)
    loss = objective()
    gradients = torch.autograd.grad(loss, (weight, bias))
    gradient = max(float(g.abs().max()) for g in gradients)
    if not np.isfinite(float(loss.detach()) + gradient):
        raise FloatingPointError('Nonfinite linear-head optimization')
    return dict(weight=weight.detach(), bias=bias.detach(), grad=gradient, converged=gradient <= grad_tol)


@torch.no_grad()
def evaluate_head(z, q, y, val_ids, test_ids, fitted, penalty, with_test=False):
    logits = F.linear(z, fitted['weight'], fitted['bias'])
    teacher_ce = float(-(q * logits.log_softmax(1)).sum(1).mean())
    result = dict(val=100 * float((logits[val_ids].argmax(1) == y[val_ids]).float().mean()),
                  val_ce=float(F.cross_entropy(logits[val_ids], y[val_ids])),
                  full_teacher_ce=teacher_ce,
                  full_objective=teacher_ce + penalty * float(fitted['weight'].square().sum()) / 2,
                  head_norm=float(fitted['weight'].norm()), grad=fitted['grad'], converged=fitted['converged'])
    if with_test:
        result.update(test=100 * float((logits[test_ids].argmax(1) == y[test_ids]).float().mean()),
                      test_ce=float(F.cross_entropy(logits[test_ids], y[test_ids])))
    return result


def run_learning_audit(directional_root, output_dir=None, transforms=('rms', 'l2', 'whiten_half'),
                       include_s2x=True, penalties=(.0001, .001, .01, .1), max_iter=1000,
                       final_seeds=tuple(range(100, 110)), data_dir='/content/data/', device='cuda'):
    source_root = Path(directional_root)
    source_config = json.loads((source_root / 'config.json').read_text())
    source = pd.read_csv(source_config['source'])
    if set(source.dataset) != {'cora'} or not penalties or min(penalties) <= 0:
        raise ValueError('Require Cora results and positive penalties')
    config = dict(directional_root=str(source_root), source=source_config, transforms=list(transforms),
                  include_s2x=include_s2x, penalties=list(penalties), max_iter=max_iter,
                  final_seeds=list(final_seeds),
                  revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
    root = Path(output_dir or source_root / 'learning_audit') / _fingerprint(config)
    root.mkdir(parents=True, exist_ok=True)
    (root / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    train, mask, validation, testing, h = _prepare_dataset('cora', data_dir, device)
    ids, y = mask.nonzero().flatten(), train['y']
    val_ids, test_ids = validation[1].nonzero().flatten(), testing[1].nonzero().flatten()
    signatures = []
    for graph, split in ((train, mask), validation, testing):
        adjacency = graph['adj'].to_sparse_csr()
        signatures.append(array_digest(graph['x'].cpu().numpy(), graph['y'].cpu().numpy(),
                          adjacency.crow_indices().cpu().numpy(), adjacency.col_indices().cpu().numpy(),
                          adjacency.values().cpu().numpy(), split.cpu().numpy()))
    jobs = [(row, 'ntk_' + row.transform) for row in source.itertuples() if row.transform in transforms]
    if include_s2x:
        jobs += [(row, 's2x') for row in source.itertuples() if row.transform == 'rms']
    if not jobs:
        raise ValueError('No requested conditions found')
    summaries, gcn_records = [], []
    for row, name in jobs:
        path = Path(row.partition_path)
        directory = path.parent.parent
        if directory.name != 'cora_' + _fingerprint(signatures)[:12]:
            raise ValueError('Saved graph/splits do not match the current Cora data')
        folder = root / name / str(row.ratio)
        folder.mkdir(parents=True, exist_ok=True)
        original_config = json.loads((Path(row.output_dir) / 'config.json').read_text())
        params = json.loads((path.parent / row.mode / 'best.json').read_text())['params']
        label_root = Path(row.output_dir).parent / ('labels_' + _fingerprint(dict(
            dataset='cora', data=signatures, seed=original_config['seed'], revision=original_config['revision'])))
        key = (params['teacher_kernel'], params['basis'], params['gamma'])
        logits = torch.load(label_root / f'{_fingerprint(key)}.pt', map_location=device, weights_only=True)
        q = (logits / params['T']).softmax(1).double()
        if name == 's2x':
            z, mapping = fit_transform(h.double(), kind='rms')
            baseline_path = folder / 'baseline_partition.pt'
            if baseline_path.exists():
                baseline = torch.load(baseline_path, map_location='cpu', weights_only=False)
            else:
                baseline = risk_partition(h, q, row.nodes, params['B'], seed=original_config['seed'],
                                          max_sweeps=100, return_assignment=True)
                torch.save(baseline, baseline_path)
            refined_path = folder / 'directional_partition.pt'
            if refined_path.exists():
                refined = torch.load(refined_path, map_location='cpu', weights_only=False)
            else:
                head_weights, head_records = [], []
                for penalty in source_config['train_penalties']:
                    for seed in source_config['train_seeds']:
                        support = sample_support(ids, y, source_config['train_support'], seed)
                        head = fit_probe(z[support], y[support], q.shape[1], seed, .2, penalty,
                                         steps=500, polish_steps=300)
                        head_weights.append(head['weight'].T)
                        head_records.append(dict(seed=seed, penalty=penalty, grad=head['grad_after'],
                                                 converged=head['converged']))
                pd.DataFrame(head_records).to_csv(folder / 'optimization_heads.csv', index=False)
                u = torch.einsum('nd,hdc->nhc', z, torch.stack(head_weights))
                refined = directional_partition(u, q, baseline['assignment'],
                                                  source_config['max_sweeps'], source_config['block_size'])
                torch.save(refined, refined_path)
        else:
            cached = torch.load(directory / 'transformed_features.pt', map_location=device, weights_only=False)
            z = cached['features'].double()
            transform = FeatureTransform(**cached['transform'])
            kernel = torch.load(directory / 'nystrom.pt', map_location='cpu', weights_only=False)
            mapping = TransformedMap(NystromGCN.from_state(kernel['mapping'], device), transform)
            baseline = torch.load(path, map_location='cpu', weights_only=False)
            refined = torch.load(source_root / row.transform / str(row.ratio) / 'directional_partition.pt',
                                 map_location='cpu', weights_only=False)
        full_mass = z.new_full((len(z),), 1 / len(z))
        candidates, fits = [], []
        for penalty in penalties:
            fitted = fit_soft_head(z, q, full_mass, penalty, max_iter)
            fits.append(fitted)
            candidates.append(dict(penalty=penalty, **evaluate_head(z, q, y, val_ids, test_ids, fitted, penalty)))
        pd.DataFrame(candidates).to_csv(folder / 'reference_grid.csv', index=False)
        choice = min(range(len(candidates)), key=lambda i: candidates[i]['val_ce'])
        penalty, reference = penalties[choice], fits[choice]
        full_metrics = evaluate_head(z, q, y, val_ids, test_ids, reference, penalty, True)
        summaries.append(dict(representation=name, ratio=row.ratio, partition='full', stage='original',
                              penalty=penalty, excess_full_objective=0., realization_error=0., **full_metrics))
        for method, data in (('baseline', baseline), ('directional', refined)):
            assignment = data['assignment'].to(device)
            centers, labels, mass = cell_means(z, q, assignment)
            if float((labels - data['y'].to(device)).abs().max()) > 1e-5:
                raise ValueError('Cached teacher labels differ from saved partition means')
            if name == 's2x':
                raw_centers, _, _ = cell_means(h.double(), q, assignment)
                cx = raw_centers.float()
                realized = mapping(cx.double())
                final_path = folder / f'{method}_gcn.csv'
                if final_path.exists():
                    final = pd.read_csv(final_path)
                else:
                    records = []
                    for seed in final_seeds:
                        val, test, epoch = _train_student(cx, labels.float(), validation, params, seed,
                                                          {**original_config['settings'], 'loss_weighting': 'mass'},
                                                          testing=testing, counts=data['counts'])
                        records.append(dict(seed=seed, val=100 * val, test=100 * test, epoch=epoch))
                    final = pd.DataFrame(records)
                    final.to_csv(final_path, index=False)
                gcn_records.append(final.assign(representation=name, ratio=row.ratio, method=method))
            else:
                fitted = torch.load(source_root / row.transform / str(row.ratio) / f'{method}_representatives.pt',
                                    map_location=device, weights_only=False)
                with torch.no_grad():
                    realized = mapping(fitted['x'].double())
            error = float((mass * (realized - centers).square().sum(1)).sum())
            for stage, features in (('ideal', centers), ('realized', realized)):
                fitted = fit_soft_head(features.detach(), labels, mass, penalty, max_iter)
                values = evaluate_head(z, q, y, val_ids, test_ids, fitted, penalty, True)
                summaries.append(dict(representation=name, ratio=row.ratio, partition=method, stage=stage,
                                      penalty=penalty, realization_error=error,
                                      excess_full_objective=values['full_objective'] - full_metrics['full_objective'],
                                      **values))
                torch.save({k: v.cpu() if torch.is_tensor(v) else v for k, v in fitted.items()},
                           folder / f'{method}_{stage}_head.pt')
        pd.DataFrame(summaries).to_csv(root / 'summary.csv', index=False)
        if gcn_records:
            pd.concat(gcn_records).to_csv(root / 's2x_students.csv', index=False)
    return pd.DataFrame(summaries).assign(output_dir=str(root))
