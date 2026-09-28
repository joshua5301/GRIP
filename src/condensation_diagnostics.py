import json
import subprocess
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric import seed_everything
from tqdm.auto import tqdm

from src.assignment_sweep import boundary_profile, grid_rows, representative, teacher_logits
from src.models import GCN
from src.node_distances import array_digest
from src.ntk_transforms import fit_transform
from src.partition import cell_means, kmeans_init
from src.risk_experiment import _fingerprint, _forward, _prepare_dataset
from src.soft_ce_partition import optimize_ce_assignment, solve_inner
from src.soft_ridge_partition import augmented, decode_moments
from src.utils import BUDGET


def save_json(value, path):
    path = Path(path)
    temporary = path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def train_prior_initial(z, y, mask, cells, seed, classes):
    ids = mask.nonzero().flatten().cpu()
    labels = y[ids.to(y.device)].cpu()
    counts = torch.bincount(labels, minlength=classes)
    desired = counts.double() * cells / len(ids)
    quota = desired.floor().long()
    remaining = cells - int(quota.sum())
    order = torch.argsort(desired - quota, descending=True, stable=True)
    quota[order[:remaining]] += 1
    generator = torch.Generator().manual_seed(seed)
    sampled, targets = [], []
    for k, count in enumerate(quota.tolist()):
        if not count:
            continue
        pool = ids[labels == k]
        chosen = torch.randperm(len(pool), generator=generator)[:count] if count <= len(pool) else \
            torch.randint(len(pool), (count,), generator=generator)
        sampled.append(pool[chosen])
        targets.append(torch.full((count,), k, dtype=torch.long))
    selected, labels = torch.cat(sampled), torch.cat(targets)
    return dict(centers=z[selected.to(z.device)].clone(),
                labels=F.one_hot(labels.to(z.device), classes).to(z),
                mass=z.new_full((cells,), 1 / cells)), selected, quota


def feature_kmeans(h, cells, seed):
    threads = faiss.omp_get_max_threads()
    faiss.omp_set_num_threads(1)
    try:
        assignment = kmeans_init(h.cpu(), cells, seed=seed, plus_plus=True)
    finally:
        faiss.omp_set_num_threads(threads)
    counts = torch.bincount(assignment, minlength=cells)
    residual = (h.cpu() - cell_means(h.cpu(), assignment, cells)[assignment]).square().sum(1)
    for empty in (counts == 0).nonzero().flatten().tolist():
        node = int(residual.masked_fill(counts[assignment] <= 1, -torch.inf).argmax())
        counts[assignment[node]] -= 1
        assignment[node] = empty
        counts[empty] += 1
        residual[node] = 0
    return assignment


@torch.no_grad()
def split_metrics(log_probability, y, q, masks):
    values = dict(full_teacher_ce=float(-(q * log_probability).sum(1).mean()))
    for name, mask in masks.items():
        prediction = log_probability[mask]
        values[f'{name}_acc'] = 100 * float((prediction.argmax(1) == y[mask]).double().mean())
        values[f'{name}_ce'] = float(F.nll_loss(prediction, y[mask]))
        values[f'{name}_teacher_ce'] = float(-(q[mask] * prediction).sum(1).mean())
    values['val_minus_train_ce'] = values['val_ce'] - values['train_ce']
    values['train_minus_val_acc'] = values['train_acc'] - values['val_acc']
    return values


def sgc_metrics(snapshot, z, q, y, masks):
    theta = snapshot['theta'].to(z)
    centers, labels, mass = decode_moments(snapshot['moments'].to(z), z.shape[1])
    values = split_metrics((augmented(z) @ theta.T).log_softmax(1), y, q, masks)
    values['condensed_ce'] = float(-(mass[:, None] * labels
                                    * (augmented(centers) @ theta.T).log_softmax(1)).sum())
    return dict(**values, outer_ce=snapshot['teacher_ce'], inner_grad_max=snapshot['inner_grad_max'])


def fit_gcn_diagnostic(cx, cy, mass, graph, q, masks, seed, epochs, eval_every,
                       hidden, dropout, lr, weight_decay, folder, training_adjacency=None):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f'seed_{seed}.json'
    if path.exists():
        return json.loads(path.read_text())
    seed_everything(seed)
    model = GCN(cx.shape[1], hidden, cy.shape[1], 2, dropout).to(cx.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    weights = (mass / mass.sum()).to(cx)
    history, best, best_value = [], None, -float('inf')
    for epoch in range(1, epochs + 1):
        if epoch == epochs // 2:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr * .1, weight_decay=weight_decay)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = -(weights[:, None] * cy * _forward(model, cx, training_adjacency)).sum()
        loss.backward()
        optimizer.step()
        if epoch % eval_every != 0 and epoch != epochs:
            continue
        model.eval()
        with torch.no_grad():
            log_probability = _forward(model, graph['x'], graph['adj'])
            row = dict(epoch=epoch, **split_metrics(log_probability, graph['y'], q, masks),
                       condensed_ce=float(-(weights[:, None] * cy * _forward(model, cx, training_adjacency)).sum()))
        history.append(row)
        if row['val_acc'] > best_value:
            best, best_value = dict(row), row['val_acc']
    result = dict(seed=seed, **best, **{f'last_{key}': value for key, value in history[-1].items()})
    pd.DataFrame(history).to_csv(folder / f'seed_{seed}_epochs.csv', index=False)
    save_json(result, path)
    return result


def select_source(table):
    return table.sort_values(['val_acc', 'candidate', 'step'], ascending=[False, True, True]).iloc[0].to_dict()


def checkpoint_seeds(step, selected_step, final_seeds, trajectory_seeds):
    return final_seeds if step in (0, selected_step) else trajectory_seeds


def full_data_reference(z, graph, q, masks, penalties, final_seeds, epochs, eval_every,
                        hidden, dropout, lr, weight_decay, root, solver):
    path = root / 'full_reference.csv'
    if path.exists():
        return pd.read_csv(path)
    labels = F.one_hot(graph['y'][masks['train']], q.shape[1]).to(z)
    features = z[masks['train']]
    mass = z.new_full((len(features),), 1 / len(features))
    rows, heads = [], []
    for penalty in penalties:
        fitted = solve_inner(features, labels, mass, penalty,
                            max_iter=solver.get('inner_max_iter', 2000), grad_tol=solver.get('inner_tol', 1e-7),
                            cg_max_iter=solver.get('cg_max_iter', 512))
        if not fitted['inner_converged']:
            raise RuntimeError('Full-train SGC fit did not converge')
        metrics = split_metrics((augmented(z) @ fitted['theta'].T).log_softmax(1), graph['y'], q,
                                {key: masks[key] for key in ('train', 'val')})
        rows.append(dict(penalty=penalty, **metrics))
        heads.append(fitted['theta'])
    pd.DataFrame(rows).to_csv(root / 'full_sgc_grid.csv', index=False)
    selected = max(range(len(rows)), key=lambda index: rows[index]['val_acc'])
    metrics = split_metrics((augmented(z) @ heads[selected].T).log_softmax(1), graph['y'], q, masks)
    records = [dict(architecture='SGC', seed=-1, **metrics)]
    targets = graph['x'].new_zeros(len(z), q.shape[1])
    targets[masks['train']] = labels.float()
    for student_seed in tqdm(final_seeds, desc='Full-train GCN reference'):
        row = fit_gcn_diagnostic(graph['x'], targets, masks['train'].float(), graph, q, masks,
                                 student_seed, epochs, eval_every, hidden, dropout, lr, weight_decay,
                                 root / 'full_gcn', training_adjacency=graph['adj'])
        records.append(dict(architecture='GCN', **row))
    result = pd.DataFrame(records)
    result.to_csv(path, index=False)
    return result


def run_condensation_diagnostics(method, output_dir, space, dataset='cora', ratio=.052,
                                 steps=200, checkpoint_steps=(10, 25, 50, 100, 150, 200),
                                 teacher_kernel='relu', basis=3000, seed=0, mixing=.05,
                                 rank=16, encoder_hidden=64, outer_scope='all', final_seeds=tuple(range(100, 110)),
                                 epochs=1000, eval_every=10, hidden=256, dropout=.9,
                                 student_lr=.01, weight_decay=.0005, data_dir='/content/data/',
                                 device='cuda', solver=None, full_baseline=True, trajectory_seeds=None):
    if method not in ('A', 'C-mean', 'C-prior') or dataset not in ('cora', 'citeseer', 'arxiv'):
        raise ValueError('Use A, C-mean or C-prior on Cora/Citeseer/Arxiv')
    if (dataset, ratio) not in BUDGET:
        raise ValueError('Unsupported dataset ratio; use a configured representative budget')
    if set(space) != {'gamma', 'T', 'penalty', 'assignment_lr'} or outer_scope not in ('all', 'train'):
        raise ValueError('Specify gamma/T/penalty/assignment_lr and all or train outer nodes')
    candidates = grid_rows(space)
    if any(not np.isfinite(value) or value <= 0 for row in candidates for value in row.values()):
        raise ValueError('Grid values must be positive and finite')
    checkpoints = sorted({0, steps, *checkpoint_steps})
    trajectory_seeds = list(final_seeds) if trajectory_seeds is None else list(trajectory_seeds)
    if (steps < 1 or any(step < 0 or step > steps for step in checkpoints)
            or epochs < 1 or eval_every < 1 or not final_seeds or not trajectory_seeds):
        raise ValueError('Invalid optimization or evaluation budget')
    solver = dict(solver or {})
    allowed = {'inner_max_iter', 'inner_tol', 'cg_max_iter', 'cg_rtol', 'chunk_size', 'outer_chunk_size',
               'solver_mode', 'tracking_inner_steps', 'tracking_cg_steps', 'tracking_refresh', 'log_every'}
    if set(solver) - allowed:
        raise ValueError('Unknown solver option')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    graph, train_mask, validation, testing, h_cpu = _prepare_dataset(dataset, data_dir, 'cpu')
    masks_cpu = dict(train=train_mask, val=validation[1], test=testing[1])
    adjacency = graph['adj']
    digest = array_digest(*(tensor.cpu().numpy() for tensor in
                           (graph['x'], graph['y'], adjacency.crow_indices(), adjacency.col_indices(),
                            adjacency.values(), *masks_cpu.values(), h_cpu)))
    config = dict(dataset=dataset, ratio=ratio, space=space, steps=steps, checkpoint_steps=checkpoints,
                  teacher_kernel=teacher_kernel, basis=basis, seed=seed, mixing=mixing,
                  rank=rank, encoder_hidden=encoder_hidden, outer_scope=outer_scope,
                  final_seeds=list(final_seeds), epochs=epochs, eval_every=eval_every, hidden=hidden,
                  trajectory_seeds=trajectory_seeds,
                  dropout=dropout, student_lr=student_lr, weight_decay=weight_decay, solver=solver,
                  full_baseline=full_baseline,
                  data_digest=digest, selection='sgc_validation_only',
                  revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
    protocol = Path(output_dir) / _fingerprint(config)
    root = protocol / method
    root.mkdir(parents=True, exist_ok=True)
    save_json(dict(**config, method=method), root / 'config.json')
    graph = {key: value.to(device) for key, value in graph.items()}
    masks = {key: value.to(device) for key, value in masks_cpu.items()}
    search_masks = {key: masks[key] for key in ('train', 'val')}
    h = h_cpu.to(device)
    z, transform = fit_transform(h.double(), kind='rms')
    torch.save(transform.state_dict(), root / 'transform.pt')
    cells, classes = BUDGET[(dataset, ratio)], int(graph['y'].max()) + 1
    init_path = root / 'initialization.pt'
    if init_path.exists():
        initial = torch.load(init_path, map_location='cpu', weights_only=False)
    else:
        initial = dict(assignment=feature_kmeans(h_cpu, cells, seed))
        if method == 'C-prior':
            representatives, ids, quota = train_prior_initial(z, graph['y'], masks['train'], cells, seed, classes)
            initial.update(representatives={key: value.cpu() for key, value in representatives.items()},
                           node_ids=ids, class_counts=quota)
        torch.save(initial, init_path)
    assignment = initial['assignment'].to(device)
    teachers = teacher_logits(h, graph, masks['train'], (graph, masks['val']), teacher_kernel,
                              space['gamma'], basis, seed, root, return_all=True)
    outer_indices = masks['train'].nonzero().flatten() if outer_scope == 'train' else None
    rows = []
    for index, params in enumerate(tqdm(candidates, desc=f'{method}: SGC validation grid')):
        folder = root / f'candidate_{index:04d}'
        folder.mkdir(exist_ok=True)
        save_json(params, folder / 'params.json')
        q = (teachers[params['gamma']].to(device) / params['T']).softmax(1).double()
        state_path = folder / 'resume.pt'
        state = torch.load(state_path, map_location='cpu', weights_only=False) if state_path.exists() else None
        if state is None or state['step'] < steps:
            optimize_ce_assignment(
                z, q, assignment, penalty=params['penalty'], lr=params['assignment_lr'], steps=steps,
                folder=folder, checkpoint_steps=checkpoints, mixing=mixing,
                assignment_rank=rank, assignment_input='features', assignment_encoder='mlp',
                encoder_hidden=encoder_hidden, factor_seed=seed, feature_control='joint' if method == 'A' else 'direct',
                initial_representatives=initial.get('representatives'), outer_indices=outer_indices,
                save_assignment=False, save_resume=True, resume_state=state, **solver)
        for step in checkpoints:
            snapshot = torch.load(folder / 'checkpoints' / f'step_{step:06d}.pt', map_location='cpu', weights_only=False)
            metrics = sgc_metrics(snapshot, z, q, graph['y'], search_masks)
            rows.append(dict(candidate=index, step=step, **params, **metrics))
        pd.DataFrame(rows).to_csv(root / 'search_sgc.csv', index=False)
    search = pd.DataFrame(rows)
    profile = boundary_profile(search.rename(columns={'val_acc': 'val'}), space)
    sgc_penalty_profile = boundary_profile(search.rename(columns={'val_acc': 'val'}).assign(step=1),
                                           {'penalty': space['penalty']})
    profile = pd.concat([profile[profile.parameter != 'penalty'], sgc_penalty_profile], ignore_index=True)
    profile.to_csv(root / 'boundaries.csv', index=False)
    choice = select_source(search)
    save_json(choice, root / 'selected.json')
    folder = root / f'candidate_{int(choice["candidate"]):04d}'
    q = (teachers[choice['gamma']].to(device) / choice['T']).softmax(1).double()
    records = []
    for step in tqdm(checkpoints, desc=f'{method}: frozen selection, SGC + GCN curves'):
        snapshot = torch.load(folder / 'checkpoints' / f'step_{step:06d}.pt', map_location='cpu', weights_only=False)
        common = dict(method=method, step=step, selected=step == int(choice['step']))
        records.append(dict(**common, architecture='SGC', seed=-1,
                            **sgc_metrics(snapshot, z, q, graph['y'], masks)))
        cx, cy, mass = representative(snapshot['moments'], transform, z.shape[1], device)
        for student_seed in checkpoint_seeds(step, int(choice['step']), final_seeds, trajectory_seeds):
            row = fit_gcn_diagnostic(cx, cy, mass, graph, q, masks, student_seed, epochs, eval_every,
                                     hidden, dropout, student_lr, weight_decay,
                                     folder / 'gcn_diagnostics' / f'step_{step:06d}')
            records.append(dict(**common, architecture='GCN', **row))
        pd.DataFrame(records).to_csv(root / 'trajectory.csv', index=False)
    trajectory = pd.DataFrame(records)
    reference = full_data_reference(z, graph, q, masks, space['penalty'], final_seeds, epochs, eval_every,
                                    hidden, dropout, student_lr, weight_decay, root, solver) if full_baseline else None
    summaries = []
    for architecture, group in trajectory[trajectory.selected].groupby('architecture'):
        summary = dict(method=method, architecture=architecture, dataset=dataset, ratio=ratio,
                       nodes=cells, selected_step=int(choice['step']), selection_architecture='SGC',
                       loss_weighting='uniform' if method == 'C-prior' else 'mass',
                       **{key: choice[key] for key in space}, search_val=choice['val_acc'], n_runs=len(group))
        for metric in ('train_acc', 'val_acc', 'test_acc', 'train_ce', 'val_ce', 'test_ce',
                       'val_minus_train_ce', 'train_minus_val_acc', 'full_teacher_ce', 'condensed_ce'):
            summary[metric] = group[metric].mean()
            summary[f'{metric}_std'] = group[metric].std(ddof=0) if len(group) > 1 else np.nan
        initial_group = trajectory[(trajectory.architecture == architecture) & (trajectory.step == 0)]
        summary['initial_test_acc'] = initial_group.test_acc.mean()
        summary['test_gain_from_initial'] = summary['test_acc'] - summary['initial_test_acc']
        if reference is not None:
            original = reference[reference.architecture == architecture]
            summary['full_test_acc'] = original.test_acc.mean()
            summary['test_drop_from_full'] = original.test_acc.mean() - summary['test_acc']
        summaries.append(dict(**summary, output_dir=str(root), protocol_dir=str(protocol)))
    result = pd.DataFrame(summaries)
    result.to_csv(root / 'summary.csv', index=False)
    save_json(dict(method=method, complete=True, output_dir=str(root)), root / 'complete.json')
    return result


def collect_diagnostics(protocol_dir):
    protocol = Path(protocol_dir)
    methods = [name for name in ('A', 'C-mean', 'C-prior') if (protocol / name / 'complete.json').exists()]
    if not methods:
        raise ValueError('No completed methods in this protocol directory')
    summaries = pd.concat([pd.read_csv(protocol / name / 'summary.csv') for name in methods], ignore_index=True)
    trajectories = pd.concat([pd.read_csv(protocol / name / 'trajectory.csv') for name in methods], ignore_index=True)
    checks = []
    if {'A', 'C-mean'}.issubset(methods):
        configs = {name: json.loads((protocol / name / 'config.json').read_text()) for name in ('A', 'C-mean')}
        if {key: value for key, value in configs['A'].items() if key != 'method'} != \
                {key: value for key, value in configs['C-mean'].items() if key != 'method'}:
            raise ValueError('A and C-mean protocol settings differ')
        candidates = grid_rows(configs['A']['space'])
        seen = set()
        for index, params in enumerate(candidates):
            key = params['gamma'], params['T']
            if key in seen:
                continue
            seen.add(key)
            values = [torch.load(protocol / name / f'candidate_{index:04d}' / 'checkpoints' / 'step_000000.pt',
                                  map_location='cpu', weights_only=False)['moments'] for name in ('A', 'C-mean')]
            error = float((values[0] - values[1]).abs().max())
            matched = torch.allclose(values[0], values[1], rtol=1e-7, atol=1e-10)
            checks.append(dict(gamma=key[0], T=key[1], max_difference=error, matched=matched))
            if not matched:
                raise ValueError('A and C-mean initial representatives differ; inspect the saved runs')
    return summaries, trajectories, pd.DataFrame(checks)


def plot_diagnostics(trajectory, fixed_epoch=True):
    import matplotlib.pyplot as plt
    trajectory = trajectory.copy()
    if fixed_epoch:
        mask = trajectory.architecture == 'GCN'
        for key in ('val_acc', 'train_ce', 'val_ce', 'val_minus_train_ce', 'full_teacher_ce'):
            trajectory.loc[mask, key] = trajectory.loc[mask, f'last_{key}']
    fig, axes = plt.subplots(2, 4, figsize=(19, 8), squeeze=False)
    for row, architecture in enumerate(('SGC', 'GCN')):
        for method, group in trajectory[trajectory.architecture == architecture].groupby('method'):
            means = group.groupby('step').mean(numeric_only=True)
            axes[row, 0].plot(means.index, means.val_acc, marker='o', label=method)
            axes[row, 1].plot(means.index, means.train_ce, label=f'{method}: train')
            axes[row, 1].plot(means.index, means.val_ce, linestyle='--', label=f'{method}: val')
            axes[row, 2].plot(means.index, means.val_minus_train_ce, marker='o', label=method)
            axes[row, 3].plot(means.index, means.full_teacher_ce, marker='o', label=method)
        for ax, ylabel in zip(axes[row], ('Validation accuracy (%)', 'Ground-truth CE',
                                          'Validation CE - train CE', 'Whole-graph teacher CE')):
            ax.set(xlabel='Condensation step', ylabel=ylabel, title=architecture)
            ax.legend(fontsize=8)
        axes[row, 2].axhline(0, color='gray', linewidth=.8)
    fig.suptitle('SGC-selected configuration | GCN: ' + ('fixed final epoch' if fixed_epoch else 'best validation epoch'))
    fig.tight_layout()
    return fig


def plot_student_learning(protocol_dir):
    import matplotlib.pyplot as plt
    protocol = Path(protocol_dir)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for method in ('A', 'C-mean', 'C-prior'):
        root = protocol / method
        if not (root / 'complete.json').exists():
            continue
        choice = json.loads((root / 'selected.json').read_text())
        folder = root / f'candidate_{int(choice["candidate"]):04d}' / 'gcn_diagnostics' / f'step_{int(choice["step"]):06d}'
        history = pd.concat([pd.read_csv(path) for path in sorted(folder.glob('seed_*_epochs.csv'))])
        means = history.groupby('epoch').mean(numeric_only=True)
        for ax, metric, ylabel in zip(axes, ('ce', 'acc'), ('Ground-truth CE', 'Accuracy (%)')):
            line, = ax.plot(means.index, means[f'train_{metric}'], label=f'{method}: train')
            ax.plot(means.index, means[f'val_{metric}'], linestyle='--', color=line.get_color(), label=f'{method}: val')
            ax.set(xlabel='GCN training epoch', ylabel=ylabel)
            ax.legend(fontsize=8)
    fig.suptitle('GCN trained on each selected condensed dataset; evaluated on original graph')
    fig.tight_layout()
    return fig
