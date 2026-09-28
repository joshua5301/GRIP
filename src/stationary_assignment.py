import json
import subprocess
import time
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import trange

from src.assignment_sweep import representative
from src.condensation_diagnostics import feature_kmeans, fit_gcn_diagnostic, save_json, split_metrics
from src.node_distances import array_digest
from src.ntk_transforms import FeatureTransform
from src.risk_experiment import _fingerprint, _prepare_dataset
from src.soft_ce_partition import cpu_state, head_gradient, solve_inner
from src.soft_ridge_partition import AssignmentMoments, augmented, decode_moments, initial_logits, make_material


def stationary_objective(moments, dimension, theta, penalty):
    centers, labels, mass = decode_moments(moments, dimension)
    gradient = head_gradient(augmented(centers), labels, mass, theta.detach(), penalty)
    return gradient.square().sum(), gradient


def optimize_stationary(z, q, assignment, theta, penalty, folder, steps=300, lr=.01,
                        mixing=.05, chunk_size=4096, checkpoint_steps=(0, 50, 100, 200, 300)):
    if steps < 1 or not 0 < mixing < 1 or min(lr, penalty, chunk_size) <= 0:
        raise ValueError('Invalid stationary optimization settings')
    checkpoints = sorted({0, steps, *checkpoint_steps})
    if any(not isinstance(step, int) or step < 0 or step > steps for step in checkpoints):
        raise ValueError('Invalid checkpoint steps')
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    config = dict(steps=steps, lr=lr, penalty=penalty, mixing=mixing, chunk_size=chunk_size,
                  checkpoint_steps=checkpoints, digest=array_digest(*(
                      t.detach().cpu().numpy() for t in (z, q, assignment, theta))))
    theta = theta.detach()
    material = make_material(z.detach(), q.detach())
    logits = initial_logits(assignment, int(assignment.max()) + 1, mixing).requires_grad_()
    optimizer = torch.optim.Adam([logits], lr=lr, eps=1e-12, foreach=False)
    path = folder / 'resume.pt'
    start, history, scale, elapsed = 0, [], None, 0.
    if path.exists():
        state = torch.load(path, map_location='cpu', weights_only=False)
        if state['config'] != config:
            raise ValueError('Stationary resume settings or data differ')
        with torch.no_grad():
            logits.copy_(state['logits'].to(logits))
        optimizer.load_state_dict(state['optimizer'])
        start, scale, elapsed = state['step'], state['scale'], state['elapsed']
        history = [row for row in state['history'] if row['step'] < start]
        del state
    started = time.perf_counter()
    for step in trange(start, steps + 1, desc='Fixed-head gradient residual'):
        optimizer.zero_grad(set_to_none=True)
        moments = AssignmentMoments.apply(logits, material, chunk_size)
        value, gradient = stationary_objective(moments, z.shape[1], theta, penalty)
        if not bool(torch.isfinite(value)):
            raise FloatingPointError('Nonfinite stationary objective')
        if scale is None:
            scale = max(float(value.detach()), 1e-30)
        norm = float(gradient.detach().norm())
        row = dict(step=step, objective=float(value.detach()), relative_objective=float(value.detach()) / scale,
                   gradient_norm=norm, gradient_max=float(gradient.detach().abs().max()),
                   head_distance_bound=norm / penalty,
                   seconds=elapsed + time.perf_counter() - started)
        history.append(row)
        if step in checkpoints:
            directory = folder / 'checkpoints'
            directory.mkdir(exist_ok=True)
            torch.save(dict(step=step, moments=moments.detach().cpu(), **{k: v for k, v in row.items() if k != 'step'}),
                       directory / f'step_{step:06d}.pt')
            pd.DataFrame(history).to_csv(folder / 'optimization.csv', index=False)
            state = cpu_state(dict(config=config, step=step, logits=logits, optimizer=optimizer.state_dict(),
                                   history=history, scale=scale, elapsed=row['seconds']))
            torch.save(state, folder / 'resume.tmp.pt')
            (folder / 'resume.tmp.pt').replace(path)
            del state
        if step == steps:
            break
        (value / scale).backward()
        if not bool(torch.isfinite(logits.grad).all()):
            raise FloatingPointError('Nonfinite assignment gradient')
        optimizer.step()
    return pd.DataFrame(history)


def run_stationary_assignment(legacy_run, output_dir, initializations=('kmeans', 'risk'),
                              steps=300, lr=.01, checkpoint_steps=(0, 10, 25, 50, 100, 150, 200, 250, 300),
                              search_seeds=(0, 1, 2), final_seeds=tuple(range(100, 110)), seed=0,
                              reference_tol=1e-8, data_dir='/content/data/', device='cuda'):
    if not initializations or set(initializations) - {'kmeans', 'risk'}:
        raise ValueError('Choose kmeans and/or risk initialization')
    if not search_seeds or not final_seeds or set(search_seeds) & set(final_seeds):
        raise ValueError('Search and final seeds must be nonempty and disjoint')
    checkpoints = sorted({0, steps, *checkpoint_steps})
    legacy_run = Path(legacy_run)
    legacy = json.loads((legacy_run / 'config.json').read_text())
    prior, options = legacy['prior'], legacy['options']
    original, source = prior['original'], Path(prior['source'])
    if original['dataset'] != 'arxiv':
        raise ValueError('Use the saved Arxiv CE bilevel reference')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    graph, train_mask, validation, testing, h = _prepare_dataset('arxiv', data_dir, device)
    adjacency = graph['adj']
    digest = array_digest(*(t.cpu().numpy() for t in (
        graph['x'], graph['y'], adjacency.crow_indices(), adjacency.col_indices(), adjacency.values(),
        train_mask, validation[1], testing[1])))
    if digest != original['data_digest']:
        raise ValueError('Legacy graph or split mismatch')
    teacher = torch.load(source / 'teacher_logits.pt', map_location=device, weights_only=True)
    baseline = torch.load(source / 'baseline_partition.pt', map_location=device, weights_only=False)
    assignment = baseline['assignment']
    if (len(baseline['counts']) != 909 or array_digest(teacher.cpu().numpy()) != prior['logits_digest']
            or array_digest(assignment.cpu().numpy()) != prior['assignment_digest']):
        raise ValueError('Legacy teacher or 909-cell partition mismatch')
    reference = torch.load(Path(legacy['previous_run']) / 'reference.pt', map_location=device, weights_only=False)
    transform = FeatureTransform(**reference['transform'])
    if transform.kind != 'rms':
        raise ValueError('Require the legacy RMS representation')
    z = transform(h.double())
    params = original['params']
    q = (teacher / params['T']).softmax(1).double()
    penalty, mixing = options['penalty'], options['mixing']
    config = dict(legacy_run=str(legacy_run), legacy=legacy, steps=steps, lr=lr, checkpoint_steps=checkpoints,
                  search_seeds=list(search_seeds), final_seeds=list(final_seeds), seed=seed,
                  reference_tol=reference_tol, z_digest=array_digest(z.cpu().numpy()),
                  revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
    root = Path(output_dir) / _fingerprint(config)
    root.mkdir(parents=True, exist_ok=True)
    masks = dict(train=train_mask, val=validation[1], test=testing[1])
    search_masks = {key: masks[key] for key in ('train', 'val')}
    settings = {key: original[key] for key in ('epochs', 'eval_every', 'hidden')}
    settings.update(dropout=params['dropout'], lr=params['lr'], weight_decay=params['weight_decay'])
    reference_fit = None
    for initialization in initializations:
        folder = root / initialization
        folder.mkdir(exist_ok=True)
        save_json(dict(**config, initialization=initialization, penalty=penalty, mixing=mixing, student=settings),
                  folder / 'config.json')
        if (folder / 'complete.json').exists():
            continue
        reference_path = folder / 'fixed_head.pt'
        if reference_fit is not None:
            fitted = reference_fit
            if not reference_path.exists():
                torch.save(cpu_state(fitted), reference_path)
        elif reference_path.exists():
            fitted = torch.load(reference_path, map_location=device, weights_only=False)
        else:
            fitted = solve_inner(z, q, z.new_full((len(z),), 1 / len(z)), penalty,
                                 max_iter=3000, grad_tol=reference_tol, cg_max_iter=2048)
            if not fitted['inner_converged']:
                raise RuntimeError('Original-data reference head did not converge')
            torch.save(cpu_state(fitted), reference_path)
        reference_fit = fitted
        theta = fitted['theta'].detach()
        full_gradient = head_gradient(augmented(z), q, z.new_full((len(z),), 1 / len(z)), theta, penalty)
        full_norm = float(full_gradient.norm())
        save_json(dict(gradient_norm=full_norm, gradient_max=float(full_gradient.abs().max()),
                       head_norm=float(theta.norm()), penalty=penalty), folder / 'reference_diagnostics.json')
        path = folder / 'initial_assignment.pt'
        if path.exists():
            current = torch.load(path, map_location=device, weights_only=True)
        else:
            current = assignment if initialization == 'risk' else feature_kmeans(h.cpu(), 909, seed).to(device)
            torch.save(current.cpu(), path)
        optimize_stationary(z, q, current, theta, penalty, folder, steps, lr, mixing,
                            options.get('chunk_size', 4096), checkpoints)
        records = []
        for step in checkpoints:
            saved = torch.load(folder / 'checkpoints' / f'step_{step:06d}.pt', map_location='cpu', weights_only=False)
            cx, cy, mass = representative(saved['moments'], transform, z.shape[1], device)
            for student_seed in search_seeds:
                record = fit_gcn_diagnostic(cx, cy, mass, graph, q, search_masks, student_seed,
                                            folder=folder / 'search' / f'step_{step:06d}', **settings)
                records.append(dict(step=step, **record))
            pd.DataFrame(records).to_csv(folder / 'search_students.csv', index=False)
        search = pd.DataFrame(records).groupby('step', as_index=False).agg(
            val=('val_acc', 'mean'), val_std=('val_acc', lambda values: values.std(ddof=0)))
        search.to_csv(folder / 'search.csv', index=False)
        selected = int(search.sort_values(['val', 'step'], ascending=[False, True]).iloc[0]['step'])
        save_json(dict(step=selected, selection='GCN validation'), folder / 'selected.json')
        records, diagnostics = [], []
        for step in sorted({0, selected, steps}):
            saved = torch.load(folder / 'checkpoints' / f'step_{step:06d}.pt', map_location='cpu', weights_only=False)
            centers, labels, mass = decode_moments(saved['moments'].to(z), z.shape[1])
            trained = solve_inner(centers, labels, mass, penalty, max_iter=3000,
                                  grad_tol=reference_tol, cg_max_iter=2048)
            if not trained['inner_converged']:
                raise RuntimeError('Diagnostic condensed head did not converge')
            diagnostic = dict(step=step, gradient_norm=saved['gradient_norm'],
                              head_distance=float((trained['theta'] - theta).norm()),
                              head_distance_bound=saved['head_distance_bound'],
                              optima_distance_bound=(saved['gradient_norm'] + full_norm) / penalty,
                              **split_metrics((augmented(z) @ trained['theta'].T).log_softmax(1), graph['y'], q, masks))
            diagnostics.append(diagnostic)
            cx, cy, mass = representative(saved['moments'], transform, z.shape[1], device)
            for student_seed in final_seeds:
                record = fit_gcn_diagnostic(cx, cy, mass, graph, q, masks, student_seed,
                                            folder=folder / 'final' / f'step_{step:06d}', **settings)
                records.append(dict(step=step, **record))
            pd.DataFrame(records).to_csv(folder / 'final_students.csv', index=False)
            pd.DataFrame(diagnostics).to_csv(folder / 'head_diagnostics.csv', index=False)
        save_json(split_metrics((augmented(z) @ theta.T).log_softmax(1), graph['y'], q, masks),
                  folder / 'reference_accuracy.json')
        final = pd.DataFrame(records)
        rows = []
        for phase, step in [('initial', 0), ('selected', selected), ('last', steps)]:
            group = final[final.step == step]
            rows.append(dict(initialization=initialization, phase=phase, step=step, nodes=909,
                             gamma=params.get('gamma'), T=params['T'], penalty=penalty, assignment_lr=lr,
                             val=group.val_acc.mean(), test=group.test_acc.mean(), test_std=group.test_acc.std(ddof=0),
                             output_dir=str(folder)))
        pd.DataFrame(rows).to_csv(folder / 'summary.csv', index=False)
        save_json(dict(complete=True), folder / 'complete.json')
    result = pd.concat([pd.read_csv(root / name / 'summary.csv') for name in ('kmeans', 'risk')
                        if (root / name / 'complete.json').exists()], ignore_index=True)
    return result, root
