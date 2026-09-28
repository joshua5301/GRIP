import json
import math
import subprocess
import time
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm

from src.assignment_sweep import grid_rows
from src.condensation_diagnostics import feature_kmeans, fit_gcn_diagnostic, save_json
from src.node_distances import array_digest
from src.ntk_transforms import FeatureTransform
from src.risk_experiment import _fingerprint, _prepare_dataset
from src.soft_ce_partition import cpu_state
from src.soft_ridge_partition import augmented


def cell_targets(q, assignment, cells):
    counts = torch.bincount(assignment, minlength=cells)
    if bool((counts == 0).any()):
        raise ValueError('Every cell must contain a node')
    labels = q.new_zeros(cells, q.shape[1]).index_add_(0, assignment, q)
    return labels / counts[:, None], counts.to(q) / len(q)


def mean_centers(z, assignment, cells):
    counts = torch.bincount(assignment, minlength=cells)
    return z.new_zeros(cells, z.shape[1]).index_add_(0, assignment, z) / counts[:, None]


def head_log_probability(heads, features):
    return (augmented(features).unsqueeze(0) @ heads.transpose(1, 2)).log_softmax(-1)


def fit_ce_trajectory(centers, labels, mass, penalty, epochs, lr, checkpoints, seeds):
    states, records = [], []
    x = augmented(centers.detach())
    for seed in seeds:
        generator = torch.Generator(device=x.device).manual_seed(seed)
        theta = torch.randn(labels.shape[1], x.shape[1], dtype=x.dtype, device=x.device,
                            generator=generator) / x.shape[1] ** .5
        theta.requires_grad_()
        optimizer = torch.optim.Adam([theta], lr=lr, foreach=False)
        for epoch in range(epochs + 1):
            optimizer.zero_grad(set_to_none=True)
            log_probability = (x @ theta.T).log_softmax(1)
            ce = -(mass[:, None] * labels * log_probability).sum()
            loss = ce + .5 * penalty * theta.square().sum()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('Nonfinite trajectory training loss')
            loss.backward()
            if epoch in checkpoints:
                states.append(theta.detach().clone())
                records.append(dict(seed=seed, epoch=epoch, condensed_ce=float(ce.detach()),
                                    regularized_ce=float(loss.detach()),
                                    gradient_norm=float(theta.grad.norm())))
            if epoch < epochs:
                optimizer.step()
    return torch.stack(states), records


@torch.no_grad()
def original_losses(z, q, heads, chunk_size):
    values = z.new_empty(len(z), len(heads))
    for start in range(0, len(z), chunk_size):
        stop = start + chunk_size
        log_probability = head_log_probability(heads, z[start:stop])
        values[start:stop] = -(log_probability * q[None, start:stop]).sum(-1).T
    return values


class TrajectoryLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, log_probability, targets, q, assignment, chunk_size, loss):
        ctx.save_for_backward(log_probability, targets, q, assignment)
        ctx.chunk_size, ctx.loss = chunk_size, loss
        total = q.new_zeros(())
        for start in range(0, len(q), chunk_size):
            stop = start + chunk_size
            prediction = -(log_probability[:, assignment[start:stop]] * q[None, start:stop]).sum(-1).T
            gap = prediction - targets[start:stop]
            total += (gap.abs() if loss == 'absolute' else gap.square()).sum()
        return total / targets.numel()

    @staticmethod
    def backward(ctx, output_gradient):
        log_probability, targets, q, assignment = ctx.saved_tensors
        gradient = torch.zeros_like(log_probability)
        for start in range(0, len(q), ctx.chunk_size):
            stop = start + ctx.chunk_size
            current = assignment[start:stop]
            prediction = -(log_probability[:, current] * q[None, start:stop]).sum(-1).T
            gap = prediction - targets[start:stop]
            coefficient = gap.sign() if ctx.loss == 'absolute' else 2 * gap
            gradient.index_add_(1, current, -coefficient.T[:, :, None] * q[None, start:stop])
        gradient *= output_gradient / targets.numel()
        return gradient, None, None, None, None, None


def assignment_objective(centers, heads, targets, q, assignment, chunk_size, loss):
    return TrajectoryLoss.apply(head_log_probability(heads, centers), targets, q,
                                assignment, chunk_size, loss)


@torch.no_grad()
def assign_nodes(centers, heads, targets, q, previous, chunk_size, loss):
    cells = len(centers)
    log_probability = head_log_probability(heads, centers)
    assignment, previous_cost = torch.empty_like(previous), q.new_empty(len(q))
    for start in range(0, len(q), chunk_size):
        stop = min(start + chunk_size, len(q))
        cost = q.new_zeros(stop - start, cells)
        for t, logp in enumerate(log_probability):
            gap = -q[start:stop] @ logp.T - targets[start:stop, t, None]
            cost += gap.abs() if loss == 'absolute' else gap.square()
        if not bool(torch.isfinite(cost).all()):
            raise FloatingPointError('Nonfinite assignment cost')
        old = previous[start:stop]
        rows = torch.arange(stop - start, device=q.device)
        previous_cost[start:stop] = cost[rows, old]
        best = cost.argmin(1)
        assignment[start:stop] = torch.where(cost[rows, best] < cost[rows, old], best, old)
    minima = q.new_full((cells,), torch.inf).scatter_reduce_(0, previous, previous_cost, reduce='amin')
    ids = torch.arange(len(q), device=q.device)
    eligible = torch.where(previous_cost == minima[previous], ids, len(q))
    anchors = torch.full((cells,), len(q), device=q.device, dtype=torch.long)
    anchors.scatter_reduce_(0, previous, eligible, reduce='amin')
    assignment[anchors] = previous[anchors]
    return assignment


def update_centers(centers, heads, targets, q, assignment, steps, lr, chunk_size, loss):
    variable = centers.detach().clone().requires_grad_()
    optimizer = torch.optim.Adam([variable], lr=lr, foreach=False)
    initial = float(assignment_objective(variable, heads, targets, q, assignment, chunk_size, loss).detach())
    scale, best_value, best, best_step = max(initial, 1e-12), initial, variable.detach().clone(), 0
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        value = assignment_objective(variable, heads, targets, q, assignment, chunk_size, loss)
        (value / scale).backward()
        if not bool(torch.isfinite(variable.grad).all()):
            raise FloatingPointError('Nonfinite representative gradient')
        optimizer.step()
        with torch.no_grad():
            current = float(assignment_objective(variable, heads, targets, q, assignment, chunk_size, loss))
        if not math.isfinite(current):
            raise FloatingPointError('Nonfinite representative objective')
        if current < best_value:
            best_value, best, best_step = current, variable.detach().clone(), step
    return best, dict(center_initial=initial, center_final=best_value, center_best_step=best_step)


@torch.no_grad()
def risk_diagnostics(centers, heads, targets, q, assignment, chunk_size, loss):
    labels, mass = cell_targets(q, assignment, len(centers))
    log_probability = head_log_probability(heads, centers)
    full = targets.mean(0)
    condensed = -(log_probability * labels[None] * mass[None, :, None]).sum((1, 2))
    objective = float(assignment_objective(centers, heads, targets, q, assignment, chunk_size, loss))
    return dict(objective=objective, risk_gap=float((full - condensed).abs().mean()),
                risk_bound=objective if loss == 'absolute' else objective ** .5,
                full_ce=float(full.mean()), condensed_ce=float(condensed.mean()),
                mass_min=float(mass.min()), mass_max=float(mass.max()))


def save_state(state, path):
    path = Path(path)
    temporary = path.with_suffix('.tmp.pt')
    torch.save(cpu_state(state), temporary)
    temporary.replace(path)


def optimize_trajectory(z, q, assignment, centers, folder, rounds=5, lloyd_steps=3,
                        center_steps=30, center_lr=.01, loss='absolute', penalty=3e-5,
                        trajectory_epochs=500, trajectory_lr=.1,
                        trajectory_checkpoints=(10, 50, 100, 200, 500), trajectory_seeds=(0,),
                        retain_rounds=2, chunk_size=2048):
    if (loss not in ('absolute', 'squared') or min(rounds, lloyd_steps, center_steps,
            trajectory_epochs, retain_rounds, chunk_size) < 1 or min(center_lr, penalty, trajectory_lr) <= 0):
        raise ValueError('Invalid trajectory optimization settings')
    checkpoints = sorted(set(trajectory_checkpoints))
    if not trajectory_seeds or not checkpoints or checkpoints[0] < 0 or checkpoints[-1] > trajectory_epochs:
        raise ValueError('Invalid trajectory seeds or checkpoints')
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    config = dict(rounds=rounds, lloyd_steps=lloyd_steps, center_steps=center_steps, center_lr=center_lr,
                  loss=loss, penalty=penalty, trajectory_epochs=trajectory_epochs, trajectory_lr=trajectory_lr,
                  trajectory_checkpoints=checkpoints, trajectory_seeds=list(trajectory_seeds),
                  retain_rounds=retain_rounds, chunk_size=chunk_size,
                  data_digest=array_digest(*(value.detach().cpu().numpy() for value in (z, q, assignment, centers))))
    path = folder / 'resume.pt'
    if path.exists():
        state = torch.load(path, map_location=z.device, weights_only=False)
        if state['config'] != config:
            raise ValueError('Trajectory resume configuration or input changed')
    else:
        state = dict(config=config, round=0, assignment=assignment.detach().clone(),
                     centers=centers.detach().clone(), heads=[], history=[], training=[], seconds=0.)
        labels, mass = cell_targets(q, assignment, len(centers))
        save_state(dict(round=0, centers=centers, labels=labels, mass=mass, assignment=assignment),
                   folder / 'round_000.pt')
        save_state(state, path)
    for round_index in tqdm(range(state['round'] + 1, rounds + 1), desc='CE trajectory alternation'):
        started = time.perf_counter()
        centers, assignment = state['centers'], state['assignment']
        labels, mass = cell_targets(q, assignment, len(centers))
        heads, training = fit_ce_trajectory(centers, labels, mass, penalty, trajectory_epochs,
                                           trajectory_lr, checkpoints, trajectory_seeds)
        buffer = (state['heads'] + [heads])[-retain_rounds:]
        frozen = torch.cat(buffer).detach()
        targets = original_losses(z, q, frozen, chunk_size)
        before = risk_diagnostics(centers, frozen, targets, q, assignment, chunk_size, loss)
        history = [dict(round=round_index, lloyd_step=0, phase='start', models=len(frozen), **before)]
        for sweep in range(1, lloyd_steps + 1):
            updated = assign_nodes(centers, frozen, targets, q, assignment, chunk_size, loss)
            fraction = float((updated != assignment).double().mean())
            assignment = updated
            after_assignment = float(assignment_objective(centers, frozen, targets, q, assignment, chunk_size, loss))
            centers, optimization = update_centers(centers, frozen, targets, q, assignment,
                                                  center_steps, center_lr, chunk_size, loss)
            diagnostic = risk_diagnostics(centers, frozen, targets, q, assignment, chunk_size, loss)
            history.append(dict(round=round_index, lloyd_step=sweep, phase='updated', models=len(frozen),
                                moved_fraction=fraction, assignment_objective=after_assignment,
                                **optimization, **diagnostic))
        labels, mass = cell_targets(q, assignment, len(centers))
        seconds = time.perf_counter() - started
        state.update(round=round_index, centers=centers, assignment=assignment, heads=buffer,
                     history=state['history'] + history,
                     training=state['training'] + [dict(round=round_index, **row) for row in training],
                     seconds=state['seconds'] + seconds)
        save_state(dict(round=round_index, centers=centers, labels=labels, mass=mass,
                        assignment=assignment, diagnostic=diagnostic, seconds=state['seconds']),
                   folder / f'round_{round_index:03d}.pt')
        save_state(state, path)
        pd.DataFrame(state['history']).to_csv(folder / 'optimization.csv', index=False)
        pd.DataFrame(state['training']).to_csv(folder / 'trajectory_training.csv', index=False)
    pd.DataFrame(state['history']).to_csv(folder / 'optimization.csv', index=False)
    pd.DataFrame(state['training']).to_csv(folder / 'trajectory_training.csv', index=False)
    return pd.DataFrame(state['history'])


def select_validation(table):
    return table.sort_values(['val', 'round', 'candidate'], ascending=[False, True, True]).iloc[0].to_dict()


def run_trajectory_sweep(legacy_run, output_dir, space, initialization='kmeans', rounds=5,
                         lloyd_steps=3, center_steps=30, trajectory_epochs=500, trajectory_lr=.1,
                         trajectory_checkpoints=(10, 50, 100, 200, 500), trajectory_seeds=(0,),
                         retain_rounds=2, chunk_size=2048, search_seeds=(0, 1, 2),
                         final_seeds=tuple(range(100, 110)), seed=0, data_dir='/content/data/', device='cuda'):
    if set(space) != {'T', 'penalty', 'center_lr', 'loss'} or initialization not in ('kmeans', 'risk'):
        raise ValueError('Specify T/penalty/center_lr/loss grid and kmeans or risk initialization')
    candidates = grid_rows(space)
    if (not search_seeds or not final_seeds or set(search_seeds) & set(final_seeds)
            or any(row['loss'] not in ('absolute', 'squared') or
                   min(row['T'], row['penalty'], row['center_lr']) <= 0 for row in candidates)):
        raise ValueError('Invalid grid or evaluation seeds')
    legacy_run = Path(legacy_run)
    legacy = json.loads((legacy_run / 'config.json').read_text())
    prior = legacy['prior']
    original, source = prior['original'], Path(prior['source'])
    if original['dataset'] != 'arxiv':
        raise ValueError('Use the saved Arxiv 909-cell reference')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    graph, train_mask, validation, testing, h = _prepare_dataset('arxiv', data_dir, device)
    adjacency = graph['adj']
    digest = array_digest(*(t.cpu().numpy() for t in (
        graph['x'], graph['y'], adjacency.crow_indices(), adjacency.col_indices(), adjacency.values(),
        train_mask, validation[1], testing[1])))
    if digest != original['data_digest']:
        raise ValueError('Legacy graph or splits differ')
    teacher = torch.load(source / 'teacher_logits.pt', map_location=device, weights_only=True)
    baseline = torch.load(source / 'baseline_partition.pt', map_location=device, weights_only=False)
    if (len(baseline['counts']) != 909 or array_digest(teacher.cpu().numpy()) != prior['logits_digest']
            or array_digest(baseline['assignment'].cpu().numpy()) != prior['assignment_digest']):
        raise ValueError('Legacy teacher or partition differs')
    reference = torch.load(Path(legacy['previous_run']) / 'reference.pt', map_location=device, weights_only=False)
    transform = FeatureTransform(**reference['transform'])
    if transform.kind != 'rms' or transform.matrix is not None:
        raise ValueError('Require the invertible legacy RMS feature scaling')
    z = transform(h.double())
    config = dict(legacy_run=str(legacy_run), legacy=legacy, space=space, initialization=initialization,
                  rounds=rounds, lloyd_steps=lloyd_steps, center_steps=center_steps,
                  trajectory_epochs=trajectory_epochs, trajectory_lr=trajectory_lr,
                  trajectory_checkpoints=list(trajectory_checkpoints), trajectory_seeds=list(trajectory_seeds),
                  retain_rounds=retain_rounds, chunk_size=chunk_size, search_seeds=list(search_seeds),
                  final_seeds=list(final_seeds), seed=seed, z_digest=array_digest(z.cpu().numpy()),
                  algorithm_version=1)
    root = Path(output_dir) / _fingerprint(config)
    root.mkdir(parents=True, exist_ok=True)
    save_json(config, root / 'config.json')
    if not (root / 'revision.txt').exists():
        (root / 'revision.txt').write_text(subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
    init_path = root / 'initial_assignment.pt'
    if init_path.exists():
        assignment = torch.load(init_path, map_location=device, weights_only=True)
    else:
        assignment = (baseline['assignment'] if initialization == 'risk'
                      else feature_kmeans(h.cpu(), 909, seed).to(device))
        save_state(assignment, init_path)
    centers = mean_centers(z, assignment, 909)
    params = original['params']
    settings = {key: original[key] for key in ('epochs', 'eval_every', 'hidden')}
    settings.update(dropout=params['dropout'], lr=params['lr'], weight_decay=params['weight_decay'])
    masks = dict(train=train_mask, val=validation[1], test=testing[1])
    search_masks = {key: masks[key] for key in ('train', 'val')}

    def evaluate(saved, q, seeds, directory, evaluation_masks):
        cx = (saved['centers'].to(z) * transform.scale + transform.output_center + transform.center).float()
        cy, mass = saved['labels'].to(z).float(), saved['mass'].to(z)
        return [fit_gcn_diagnostic(cx, cy, mass, graph, q, evaluation_masks, student_seed,
                                   folder=directory, **settings) for student_seed in seeds]

    search = []
    for index, candidate in enumerate(tqdm(candidates, desc='Trajectory clustering grid')):
        folder = root / f'candidate_{index:04d}'
        folder.mkdir(exist_ok=True)
        save_json(candidate, folder / 'params.json')
        q = (teacher / candidate['T']).softmax(1).double()
        optimize_trajectory(z, q, assignment, centers, folder, rounds=rounds, lloyd_steps=lloyd_steps,
                            center_steps=center_steps, trajectory_epochs=trajectory_epochs,
                            trajectory_lr=trajectory_lr, trajectory_checkpoints=trajectory_checkpoints,
                            trajectory_seeds=trajectory_seeds, retain_rounds=retain_rounds,
                            chunk_size=chunk_size, **{key: candidate[key] for key in ('penalty', 'center_lr', 'loss')})
        students = []
        for round_index in range(rounds + 1):
            saved = torch.load(folder / f'round_{round_index:03d}.pt', map_location=device, weights_only=False)
            directory = (root / 'initial_search' / _fingerprint(dict(T=candidate['T'])) if round_index == 0
                         else folder / 'search' / f'round_{round_index:03d}')
            records = evaluate(saved, q, search_seeds, directory, search_masks)
            students.extend(dict(round=round_index, **row) for row in records)
            values = pd.Series([row['val_acc'] for row in records])
            search.append(dict(candidate=index, round=round_index, **candidate,
                               val=values.mean(), val_std=values.std(ddof=0)))
            pd.DataFrame(search).to_csv(root / 'search.csv', index=False)
            pd.DataFrame(students).to_csv(folder / 'search_students.csv', index=False)
    table = pd.DataFrame(search)
    selected = select_validation(table)
    save_json(selected, root / 'selected.json')
    index, selected_round = int(selected['candidate']), int(selected['round'])
    candidate = candidates[index]
    folder = root / f'candidate_{index:04d}'
    q = (teacher / candidate['T']).softmax(1).double()
    records = []
    for round_index in sorted({0, selected_round, rounds}):
        saved = torch.load(folder / f'round_{round_index:03d}.pt', map_location=device, weights_only=False)
        rows = evaluate(saved, q, final_seeds, folder / 'final' / f'round_{round_index:03d}', masks)
        records.extend(dict(round=round_index, **row) for row in rows)
        pd.DataFrame(records).to_csv(root / 'final_students.csv', index=False)
    final = pd.DataFrame(records)
    rows = []
    for phase, round_index in [('initial', 0), ('selected', selected_round), ('last', rounds)]:
        group = final[final['round'] == round_index]
        rows.append(dict(dataset='arxiv', ratio=.005, nodes=909, initialization=initialization,
                         phase=phase, round=round_index, **candidate, gamma=params.get('gamma'),
                         dropout=params['dropout'],
                         search_val=float(table[(table.candidate == index) & (table['round'] == round_index)].iloc[0]['val']),
                         final_val=group.val_acc.mean(), final_val_std=group.val_acc.std(ddof=0),
                         test_mean=group.test_acc.mean(), test_std=group.test_acc.std(ddof=0),
                         output_dir=str(root)))
    result = pd.DataFrame(rows)
    result.to_csv(root / 'summary.csv', index=False)
    return result, root
