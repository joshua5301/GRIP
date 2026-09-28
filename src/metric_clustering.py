import json
import math
import subprocess
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm, trange

from src.assignment_sweep import grid_rows, representative
from src.condensation_diagnostics import feature_kmeans, fit_gcn_diagnostic, save_json
from src.node_distances import array_digest
from src.ntk_transforms import FeatureTransform
from src.risk_experiment import _fingerprint, _prepare_dataset
from src.soft_ce_partition import (implicit_moment_gradient, outer_value_gradient,
                                   solve_head_system, solve_inner)
from src.soft_ridge_partition import augmented, decode_moments, make_material
from src.trajectory_clustering import mean_centers, save_state


def normalize_metric(weight):
    return weight * (weight.shape[1] ** .5 / weight.norm().clamp_min(1e-12))


def metric_probability(z, centers, weight, tau):
    nodes, representatives = z @ weight.T, centers @ weight.T
    distance = (nodes.square().sum(1, keepdim=True) + representatives.square().sum(1)[None]
                - 2 * nodes @ representatives.T).clamp_min(0)
    return (-distance / tau).softmax(1)


class MetricMoments(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, centers, z, material, tau, chunk_size):
        ctx.save_for_backward(weight, centers, z, material)
        ctx.tau, ctx.chunk_size = tau, chunk_size
        moments = z.new_zeros(len(centers), material.shape[1])
        for start in range(0, len(z), chunk_size):
            stop = start + chunk_size
            p = metric_probability(z[start:stop], centers, weight, tau)
            moments.add_(p.T @ material[start:stop] / len(z))
        return moments

    @staticmethod
    def backward(ctx, output_gradient):
        weight, centers, z, material = ctx.saved_tensors
        weight_gradient, center_gradient = torch.zeros_like(weight), torch.zeros_like(centers)
        with torch.enable_grad():
            w = weight.detach().requires_grad_()
            c = centers.detach().requires_grad_()
            for start in range(0, len(z), ctx.chunk_size):
                stop = start + ctx.chunk_size
                p = metric_probability(z[start:stop], c, w, ctx.tau)
                moments = p.T @ material[start:stop] / len(z)
                dw, dc = torch.autograd.grad(moments, (w, c), output_gradient)
                weight_gradient.add_(dw)
                center_gradient.add_(dc)
        return weight_gradient, center_gradient, None, None, None, None


def metric_partition(weight, z, material, initial_centers, tau, lloyd_steps, chunk_size):
    centers = initial_centers
    metric = normalize_metric(weight)
    for _ in range(lloyd_steps):
        moments = MetricMoments.apply(metric, centers, z, material, tau, chunk_size)
        if not bool(torch.isfinite(moments).all()) or bool((moments[:, 0] <= 0).any()):
            raise FloatingPointError('Nonfinite moments or empty metric cell; inspect tau')
        centers, _, _ = decode_moments(moments, z.shape[1])
    return moments


def optimize_metric(z, q, initial_centers, folder, tau=.01, metric_lr=.003, penalty=3e-5,
                    steps=100, lloyd_steps=3, checkpoint_steps=(0, 10, 25, 50, 100),
                    chunk_size=2048, inner_max_iter=2000, inner_tol=1e-7,
                    cg_max_iter=1024, cg_rtol=1e-6):
    if min(tau, metric_lr, penalty, inner_tol, cg_rtol) <= 0 or min(steps, lloyd_steps, chunk_size) < 1:
        raise ValueError('Invalid metric optimization settings')
    checkpoints = sorted({0, steps, *checkpoint_steps})
    if any(step < 0 or step > steps for step in checkpoints):
        raise ValueError('Checkpoint outside optimization budget')
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    config = dict(tau=tau, metric_lr=metric_lr, penalty=penalty, steps=steps, lloyd_steps=lloyd_steps,
                  checkpoint_steps=checkpoints, chunk_size=chunk_size, inner_max_iter=inner_max_iter,
                  inner_tol=inner_tol, cg_max_iter=cg_max_iter, cg_rtol=cg_rtol,
                  digest=array_digest(*(x.detach().cpu().numpy() for x in (z, q, initial_centers))))
    weight = torch.eye(z.shape[1], device=z.device, dtype=z.dtype).requires_grad_()
    optimizer = torch.optim.Adam([weight], lr=metric_lr, foreach=False)
    material = make_material(z, q)
    theta, history, start = None, [], 0
    resume = folder / 'resume.pt'
    if resume.exists():
        state = torch.load(resume, map_location=z.device, weights_only=False)
        if state['config'] != config:
            raise ValueError('Metric resume inputs or configuration differ')
        with torch.no_grad():
            weight.copy_(state['weight'])
        optimizer.load_state_dict(state['optimizer'])
        theta, history, start = state['theta'], state['history'], state['next_step']
    for step in trange(start, steps + 1, desc='Linear metric CE bilevel'):
        optimizer.zero_grad(set_to_none=True)
        moments = metric_partition(weight, z, material, initial_centers, tau, lloyd_steps, chunk_size)
        centers, labels, mass = decode_moments(moments.detach(), z.shape[1])
        fitted = solve_inner(centers, labels, mass, penalty, initial=theta,
                             max_iter=inner_max_iter, grad_tol=inner_tol, cg_max_iter=cg_max_iter)
        theta = fitted.pop('theta')
        value, outer_gradient = outer_value_gradient(z, q, theta, chunk_size)
        if not fitted['inner_converged'] or not math.isfinite(value):
            save_json(dict(step=step, **fitted), folder / 'failure.json')
            raise RuntimeError(f'Step {step}: inner CE did not converge')
        row = dict(step=step, outer_ce=value, **fitted,
                   mass_min=float(mass.min()), mass_max=float(mass.max()),
                   effective_cells=float(1 / mass.square().sum()))
        with torch.no_grad():
            singular = torch.linalg.svdvals(normalize_metric(weight))
            row.update(metric_norm=float(singular.norm()), singular_min=float(singular.min()),
                       singular_max=float(singular.max()))
        if step in checkpoints:
            save_state(dict(step=step, moments=moments, theta=theta,
                            metric=normalize_metric(weight), diagnostic=row),
                       folder / f'step_{step:06d}.pt')
        if step < steps:
            vector, diagnostic = solve_head_system(
                augmented(centers), labels, mass, theta, penalty, outer_gradient,
                rtol=cg_rtol, max_iter=cg_max_iter,
            )
            row.update(diagnostic)
            if not diagnostic['cg_converged']:
                save_json(row, folder / 'failure.json')
                raise RuntimeError(f'Step {step}: implicit solve failed, residual={diagnostic["cg_relative_residual"]:.3g}')
            derivative = implicit_moment_gradient(moments, z.shape[1], theta, vector, penalty)
            moments.backward(derivative)
            if not bool(torch.isfinite(weight.grad).all()):
                raise FloatingPointError('Nonfinite metric gradient')
            row['metric_gradient_norm'] = float(weight.grad.norm())
            optimizer.step()
        history.append(row)
        if step in checkpoints:
            save_state(dict(config=config, next_step=step + 1, weight=weight,
                            optimizer=optimizer.state_dict(), theta=theta, history=history), resume)
            pd.DataFrame(history).to_csv(folder / 'optimization.csv', index=False)
    pd.DataFrame(history).to_csv(folder / 'optimization.csv', index=False)
    return pd.DataFrame(history)


def run_metric_sweep(legacy_run, output_dir, space, initialization='kmeans', steps=100,
                     lloyd_steps=3, checkpoint_steps=(0, 10, 25, 50, 100),
                     search_seeds=(0, 1, 2), final_seeds=tuple(range(100, 110)),
                     seed=0, data_dir='/content/data/', device='cuda', solver=None):
    if set(space) != {'T', 'tau', 'metric_lr', 'penalty'} or initialization not in ('kmeans', 'risk'):
        raise ValueError('Specify T/tau/metric_lr/penalty and kmeans or risk initialization')
    candidates = grid_rows(space)
    if (not search_seeds or not final_seeds or set(search_seeds) & set(final_seeds)
            or any(not math.isfinite(v) or v <= 0 for row in candidates for v in row.values())):
        raise ValueError('Require positive grid values and disjoint evaluation seeds')
    checkpoints = sorted({0, steps, *checkpoint_steps})
    legacy = json.loads((Path(legacy_run) / 'config.json').read_text())
    prior, original = legacy['prior'], legacy['prior']['original']
    source = Path(prior['source'])
    if original['dataset'] != 'arxiv':
        raise ValueError('Use the saved Arxiv 909-cell reference')
    solver = dict(solver or {})
    if set(solver) - {'chunk_size', 'inner_max_iter', 'inner_tol', 'cg_max_iter', 'cg_rtol'}:
        raise ValueError('Unknown solver option')
    config = dict(legacy=legacy, space=space, initialization=initialization, steps=steps,
                  lloyd_steps=lloyd_steps, checkpoint_steps=checkpoints, search_seeds=list(search_seeds),
                  final_seeds=list(final_seeds), seed=seed, solver=solver, algorithm_version=1)
    root = Path(output_dir) / _fingerprint(config)
    root.mkdir(parents=True, exist_ok=True)
    save_json(config, root / 'config.json')
    if not (root / 'revision.txt').exists():
        (root / 'revision.txt').write_text(subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    graph, train_mask, validation, testing, h = _prepare_dataset('arxiv', data_dir, device)
    adj = graph['adj']
    digest = array_digest(*(t.cpu().numpy() for t in (
        graph['x'], graph['y'], adj.crow_indices(), adj.col_indices(), adj.values(),
        train_mask, validation[1], testing[1])))
    if digest != original['data_digest']:
        raise ValueError('Original graph or splits differ')
    teacher = torch.load(source / 'teacher_logits.pt', map_location=device, weights_only=True)
    baseline = torch.load(source / 'baseline_partition.pt', map_location=device, weights_only=False)
    if (len(baseline['counts']) != 909 or array_digest(teacher.cpu().numpy()) != prior['logits_digest']
            or array_digest(baseline['assignment'].cpu().numpy()) != prior['assignment_digest']):
        raise ValueError('Saved teacher or 909-cell initialization differs')
    inputs_path = root / 'inputs.pt'
    if inputs_path.exists():
        inputs = torch.load(inputs_path, map_location=device, weights_only=False)
        z, assignment = inputs['z'], inputs['assignment']
        transform = FeatureTransform(**inputs['transform'])
    else:
        reference = torch.load(Path(legacy['previous_run']) / 'reference.pt', map_location=device, weights_only=False)
        transform = FeatureTransform(**reference['transform'])
        if transform.kind != 'rms' or transform.matrix is not None:
            raise ValueError('Require the saved affine RMS transform')
        z = transform(h.double())
        assignment = (baseline['assignment'] if initialization == 'risk'
                      else feature_kmeans(h.cpu(), 909, seed).to(device))
        save_state(dict(z=z, assignment=assignment, transform=vars(transform)), inputs_path)
    initial_centers = mean_centers(z, assignment, 909)
    params = original['params']
    settings = {key: original[key] for key in ('epochs', 'eval_every', 'hidden')}
    settings.update(dropout=params['dropout'], lr=params['lr'], weight_decay=params['weight_decay'])
    masks = dict(train=train_mask, val=validation[1], test=testing[1])
    search_masks = {key: masks[key] for key in ('train', 'val')}

    def evaluate(saved, q, seeds, folder, evaluation_masks):
        cx, cy, mass = representative(saved['moments'], transform, z.shape[1], device)
        return [fit_gcn_diagnostic(cx, cy, mass, graph, q, evaluation_masks, student_seed,
                                   folder=folder, **settings) for student_seed in seeds]

    search = []
    for index, candidate in enumerate(tqdm(candidates, desc='Global metric grid')):
        folder = root / f'candidate_{index:04d}'
        folder.mkdir(exist_ok=True)
        save_json(candidate, folder / 'params.json')
        q = (teacher / candidate['T']).softmax(1).double()
        material = make_material(z, q)
        hard = z.new_zeros(909, material.shape[1]).index_add_(0, assignment, material) / len(z)
        save_state(dict(step=-1, moments=hard), folder / 'hard.pt')
        optimize_metric(z, q, initial_centers, folder, steps=steps, lloyd_steps=lloyd_steps,
                        checkpoint_steps=checkpoints, **solver,
                        **{key: candidate[key] for key in ('tau', 'metric_lr', 'penalty')})
        for step in [-1, *checkpoints]:
            path = folder / ('hard.pt' if step == -1 else f'step_{step:06d}.pt')
            saved = torch.load(path, map_location='cpu', weights_only=False)
            cache = folder / 'search' / f'step_{step}'
            if step <= 0:
                key = dict(T=candidate['T'], step=step)
                if step == 0:
                    key['tau'] = candidate['tau']
                cache = root / 'initial_search' / _fingerprint(key)
            rows = evaluate(saved, q, search_seeds, cache, search_masks)
            scores = pd.Series([row['val_acc'] for row in rows])
            search.append(dict(candidate=index, step=step, **candidate,
                               val=scores.mean(), val_std=scores.std(ddof=0)))
            pd.DataFrame(search).to_csv(root / 'search.csv', index=False)
    table = pd.DataFrame(search)
    selected = table.sort_values(['val', 'step', 'candidate'], ascending=[False, True, True]).iloc[0].to_dict()
    save_json(selected, root / 'selected.json')
    candidate = candidates[int(selected['candidate'])]
    folder = root / f'candidate_{int(selected["candidate"]):04d}'
    q = (teacher / candidate['T']).softmax(1).double()
    records = []
    for step in sorted({-1, 0, int(selected['step']), steps}):
        path = folder / ('hard.pt' if step == -1 else f'step_{step:06d}.pt')
        saved = torch.load(path, map_location='cpu', weights_only=False)
        rows = evaluate(saved, q, final_seeds, folder / 'final' / f'step_{step}', masks)
        records.extend(dict(step=step, **row) for row in rows)
        pd.DataFrame(records).to_csv(root / 'final_students.csv', index=False)
    final, rows = pd.DataFrame(records), []
    for phase, step in [('hard_initial', -1), ('metric_initial', 0), ('selected', int(selected['step'])), ('last', steps)]:
        group = final[final.step == step]
        score = table[(table.candidate == selected['candidate']) & (table.step == step)].iloc[0]['val']
        rows.append(dict(dataset='arxiv', ratio=.005, nodes=909, phase=phase, step=step,
                         initialization=initialization, metric_parameters=z.shape[1] ** 2,
                         gamma=params.get('gamma'), **candidate, search_val=score,
                         final_val=group.val_acc.mean(), final_val_std=group.val_acc.std(ddof=0),
                         test_mean=group.test_acc.mean(), test_std=group.test_acc.std(ddof=0), output_dir=str(root)))
    result = pd.DataFrame(rows)
    result.to_csv(root / 'summary.csv', index=False)
    return result, root
