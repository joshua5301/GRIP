import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import trange

from src.anil_probe import fit_probe
from src.anil_representation import sample_support
from src.node_distances import array_digest
from src.ntk_nystrom import NystromGCN
from src.ntk_risk import fit_representatives
from src.ntk_transforms import FeatureTransform, TransformedMap
from src.risk_experiment import _fingerprint, _prepare_dataset, _train_student


def statistics(u, q, assignment, clusters):
    counts = torch.bincount(assignment, minlength=clusters).to(u.dtype)
    sums = u.new_zeros(clusters, *u.shape[1:]).index_add_(0, assignment, u)
    labels = q.new_zeros(clusters, q.shape[1]).index_add_(0, assignment, q)
    return counts, sums, labels


def score_statistics(u_energy, uq, n, sums, labels):
    total = n.sum()
    centers = sums / n[:, None, None]
    variance = (u_energy - (sums * centers).sum((0, 2)) / total).clamp_min(0)
    moment = uq - (centers * labels[:, None, :]).sum((0, 2)) / total
    return (variance / 4 + moment.abs()).mean(), variance, moment


@torch.no_grad()
def directional_partition(u, q, initial_assignment, max_sweeps=100, block_size=128, seed=0, tol=1e-12):
    u, q = u.double(), q.double()
    assignment = initial_assignment.to(u.device).clone()
    clusters = int(assignment.max()) + 1
    n, sums, labels = statistics(u, q, assignment, clusters)
    if bool((n == 0).any()):
        raise ValueError('Initial partition contains an empty cell')
    energy = u.square().sum(2).mean(0)
    uq = (u * q[:, None, :]).sum(2).mean(0)
    objective, variance, moment = score_statistics(energy, uq, n, sums, labels)
    history = [dict(sweep=0, J=float(objective), moves=0)]
    generator = torch.Generator(device=u.device).manual_seed(seed)

    def apply_moves(ids, destinations):
        nonlocal n, sums, labels, objective, variance, moment
        candidate = assignment.clone()
        candidate[ids] = destinations
        nn, ss, ll = statistics(u, q, candidate, clusters)
        if bool((nn > 0).all()):
            value, vv, ee = score_statistics(energy, uq, nn, ss, ll)
            if value < objective - tol:
                assignment.copy_(candidate)
                n, sums, labels = nn, ss, ll
                objective, variance, moment = value, vv, ee
                return len(ids)
        if len(ids) == 1:
            return 0
        middle = len(ids) // 2
        return apply_moves(ids[:middle], destinations[:middle]) + apply_moves(ids[middle:], destinations[middle:])

    converged = False
    for sweep in trange(1, max_sweeps + 1, desc='Directional partition'):
        moved = 0
        for ids in torch.randperm(len(u), generator=generator, device=u.device).split(block_size):
            centers, means = sums / n[:, None, None], labels / n[:, None]
            source = assignment[ids]
            du = u[ids] - centers[source]
            dq = q[ids] - means[source]
            ka = n[source] / (len(u) * (n[source] - 1).clamp_min(1))
            kb = n / (len(u) * (n + 1))
            target_u = u[ids, None] - centers[None]
            target_q = q[ids, None] - means[None]
            dv = -ka[:, None, None] * du.square().sum(2)[:, None]
            dv = dv + kb[None, :, None] * target_u.square().sum(3)
            de = -ka[:, None, None] * (du * dq[:, None]).sum(2)[:, None]
            de = de + kb[None, :, None] * (target_u * target_q[:, :, None]).sum(3)
            delta = (dv / 4 + (moment[None, None] + de).abs() - moment.abs()[None, None]).mean(2)
            delta.scatter_(1, source[:, None], torch.inf)
            delta[n[source] <= 1] = torch.inf
            values, destinations = delta.min(1)
            take = values < -tol
            if bool(take.any()):
                moved += apply_moves(ids[take], destinations[take])
        history.append(dict(sweep=sweep, J=float(objective), moves=moved))
        if moved == 0:
            converged = True
            break
    return dict(assignment=assignment.cpu(), counts=n.long().cpu(), y=(labels / n[:, None]).float().cpu(),
                J=float(objective), history=history, converged=converged)


@torch.no_grad()
def audit_heads(u, q, assignment, biases):
    n, sums, labels = statistics(u, q, assignment, int(assignment.max()) + 1)
    centers, means = sums / n[:, None, None], labels / n[:, None]
    energy = u.square().sum(2).mean(0)
    uq = (u * q[:, None]).sum(2).mean(0)
    _, variance, moment = score_statistics(energy, uq, n, sums, labels)
    original = -(q[:, None] * (u + biases).log_softmax(2)).sum(2).mean(0)
    condensed = -((means[:, None] * (centers + biases).log_softmax(2)).sum(2)
                  * (n / n.sum())[:, None]).sum(0)
    gap = original - condensed
    bound = variance / 4 + moment.abs()
    return pd.DataFrame(dict(head=range(u.shape[1]), bound=bound.cpu().numpy(),
                             absolute_gap=gap.abs().cpu().numpy(), signed_gap=gap.cpu().numpy(),
                             variance_term=(variance / 4).cpu().numpy(),
                             moment_term=moment.abs().cpu().numpy(),
                             bound_violation=(gap.abs() - bound).clamp_min(0).cpu().numpy()))


def run_directional_study(source, output_dir, transforms=('rms',),
                          train_support=70, train_penalties=(.001, .01, .1), train_seeds=(10, 11),
                          audit_support=105, audit_penalties=(.003, .03), audit_seeds=(200, 201),
                          final_seeds=tuple(range(100, 110)), max_sweeps=100, block_size=128,
                          data_dir='/content/data/', device='cuda'):
    options = {k: v for k, v in locals().copy().items() if k != 'output_dir'}
    source = Path(source)
    selected = pd.read_csv(source)
    selected = selected[selected['transform'].isin(transforms)]
    if selected.empty or set(selected.dataset) != {'cora'}:
        raise ValueError('Require saved Cora transformed NTK results')
    if train_support == audit_support and set(train_penalties) & set(audit_penalties) and set(train_seeds) & set(audit_seeds):
        raise ValueError('Optimization and audit head configurations overlap')
    options['source'] = str(source)
    options['revision'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    options['source_digest'] = _fingerprint(selected.to_dict('records'))
    options['partition_digests'] = [array_digest(torch.load(p, map_location='cpu', weights_only=False)['assignment'].numpy())
                                    for p in selected.partition_path]
    root = Path(output_dir) / _fingerprint(options)
    root.mkdir(parents=True, exist_ok=True)
    (root / 'config.json').write_text(json.dumps(options, indent=2), encoding='utf-8')
    train, mask, validation, testing, _ = _prepare_dataset('cora', data_dir, device)
    signatures = []
    for graph, split in ((train, mask), validation, testing):
        adjacency = graph['adj'].to_sparse_csr()
        signatures.append(array_digest(graph['x'].cpu().numpy(), graph['y'].cpu().numpy(),
                          adjacency.crow_indices().cpu().numpy(), adjacency.col_indices().cpu().numpy(),
                          adjacency.values().cpu().numpy(), split.cpu().numpy()))
    ids, y = mask.nonzero().flatten(), train['y']
    summaries, audits, students, head_records = [], [], [], []
    for row in selected.itertuples():
        path = Path(row.partition_path)
        directory = path.parent.parent
        if directory.name != 'cora_' + _fingerprint(signatures)[:12]:
            raise ValueError('Original graph data or split differs from saved partition')
        folder = root / row.transform / str(row.ratio)
        folder.mkdir(parents=True, exist_ok=True)
        config = json.loads((Path(row.output_dir) / 'config.json').read_text())
        params = json.loads((path.parent / row.mode / 'best.json').read_text())['params']
        cached = torch.load(directory / 'transformed_features.pt', map_location=device, weights_only=False)
        z = cached['features'].double()
        transform = FeatureTransform(**cached['transform'])
        base = torch.load(directory / 'nystrom.pt', map_location='cpu', weights_only=False)
        mapping = TransformedMap(NystromGCN.from_state(base['mapping'], device), transform)
        label_root = Path(row.output_dir).parent / ('labels_' + _fingerprint(dict(
            dataset='cora', data=signatures, seed=config['seed'], revision=config['revision'])))
        key = (params['teacher_kernel'], params['basis'], params['gamma'])
        logits = torch.load(label_root / f'{_fingerprint(key)}.pt', map_location=device, weights_only=True)
        q = (logits / params['T']).softmax(1).double()
        original = torch.load(path, map_location='cpu', weights_only=False)
        old_assignment = original['assignment'].to(device)
        old_counts = original['counts'].to(device).double()
        check_labels = q.new_zeros(len(old_counts), q.shape[1]).index_add_(0, old_assignment, q) / old_counts[:, None]
        if float((check_labels - original['y'].to(device)).abs().max()) > 1e-5:
            raise ValueError('Cached teacher probabilities do not match the partition labels')
        heads = {}
        for group, size, penalties, seeds in (
            ('optimization', train_support, train_penalties, train_seeds),
            ('heldout', audit_support, audit_penalties, audit_seeds),
        ):
            weights, biases = [], []
            for penalty in penalties:
                for seed in seeds:
                    head_path = folder / f'{group}_{penalty}_{seed}.pt'
                    if head_path.exists():
                        fitted = torch.load(head_path, map_location=device, weights_only=False)
                    else:
                        support = sample_support(ids, y, size, seed)
                        fitted = fit_probe(z[support], y[support], q.shape[1], seed, .2, penalty,
                                           steps=500, polish_steps=300)
                        fitted['support'] = support
                        torch.save({k: v.cpu() if torch.is_tensor(v) else v for k, v in fitted.items()}, head_path)
                    weights.append(fitted['weight'].T)
                    biases.append(fitted['bias'])
                    head_records.append(dict(transform=row.transform, ratio=row.ratio, group=group,
                                             penalty=penalty, seed=seed, support_size=size,
                                             converged=fitted['converged'], grad=fitted['grad_after']))
            heads[group] = (torch.einsum('nd,hdc->nhc', z, torch.stack(weights)), torch.stack(biases))
        refined_path = folder / 'directional_partition.pt'
        if refined_path.exists():
            refined = torch.load(refined_path, map_location='cpu', weights_only=False)
        else:
            refined = directional_partition(heads['optimization'][0], q, original['assignment'],
                                              max_sweeps, block_size, seed=0)
            torch.save(refined, refined_path)
        pd.DataFrame(refined['history']).to_csv(folder / 'partition_history.csv', index=False)
        for method, data in (('baseline', original), ('directional', refined)):
            assignment = data['assignment'].to(device)
            for group, (u, bias) in heads.items():
                table = audit_heads(u, q, assignment, bias)
                audits.append(table.assign(transform=row.transform, ratio=row.ratio, method=method, group=group))
            fit_path = folder / f'{method}_representatives.pt'
            if fit_path.exists():
                fitted = torch.load(fit_path, map_location='cpu', weights_only=False)
            else:
                fitted = fit_representatives(train['x'], train['adj'], None, assignment,
                                             config['reconstruction_steps'], config['reconstruction_lr'],
                                             feature_map=mapping, mapped_features=z)
                torch.save(fitted, fit_path)
            pd.DataFrame(fitted['reconstruction_history']).to_csv(folder / f'{method}_reconstruction.csv', index=False)
            final_path = folder / f'{method}_students.csv'
            if final_path.exists():
                final = pd.read_csv(final_path)
            else:
                records = []
                for seed in final_seeds:
                    val, test, epoch = _train_student(fitted['x'].to(device), data['y'].to(device), validation,
                                                     params, seed, {**config['settings'], 'loss_weighting': 'mass'},
                                                     testing=testing, counts=data['counts'])
                    records.append(dict(seed=seed, val=100 * val, test=100 * test, epoch=epoch))
                final = pd.DataFrame(records)
                final.to_csv(final_path, index=False)
            students.append(final.assign(transform=row.transform, ratio=row.ratio, method=method))
            summaries.append(dict(transform=row.transform, ratio=row.ratio, method=method,
                                  val=final.val.mean(), val_std=final.val.std(ddof=0),
                                  test=final.test.mean(), test_std=final.test.std(ddof=0),
                                  reconstruction_error=fitted['reconstruction_final'],
                                  changed_fraction=float((data['assignment'] != original['assignment']).float().mean()),
                                  converged=data['converged'], output_dir=str(root)))
        pd.DataFrame(summaries).to_csv(root / 'summary.csv', index=False)
        pd.concat(audits).to_csv(root / 'head_audit.csv', index=False)
        pd.concat(students).to_csv(root / 'students.csv', index=False)
        pd.DataFrame(head_records).to_csv(root / 'heads.csv', index=False)
    return pd.DataFrame(summaries)
