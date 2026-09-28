import inspect
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from src.assignment_sweep import representative
from src.condensation_diagnostics import feature_kmeans, fit_gcn_diagnostic, save_json, sgc_metrics
from src.node_distances import array_digest
from src.ntk_transforms import FeatureTransform
from src.risk_experiment import _fingerprint, _prepare_dataset
from src.soft_ce_partition import optimize_ce_assignment
from src.soft_ridge_partition import AssignmentMoments, initial_logits, make_material


CASES = ('risk_dense', 'kmeans_dense', 'risk_mlp', 'kmeans_mlp')


def solver_options(saved, case, checkpoints, seed):
    if case not in CASES:
        raise ValueError('Unknown initialization/assignment case')
    options = {key: value for key, value in saved.items()
               if key in inspect.signature(optimize_ce_assignment).parameters}
    mlp = case.endswith('mlp')
    options.update(steps=300, checkpoint_steps=sorted({0, 300, *checkpoints}),
                   solver_mode='exact', mass_mode='free', feature_control='joint',
                   assignment_rank=16 if mlp else None, factor_seed=seed,
                   assignment_input='features' if mlp else 'node',
                   assignment_encoder='mlp' if mlp else 'linear', encoder_hidden=64,
                   save_assignment=False, save_resume=True)
    for key in ('folder', 'resume_state', 'initial_representatives', 'outer_indices'):
        options.pop(key, None)
    return options


def run_initialization_factorial(legacy_run, output_dir, cases=CASES,
                                 checkpoint_steps=(0, 10, 25, 50, 100, 150, 200, 250, 300),
                                 search_seeds=(0, 1, 2), final_seeds=tuple(range(100, 110)),
                                 seed=0, data_dir='/content/data/', device='cuda'):
    if not cases or any(case not in CASES for case in cases):
        raise ValueError('Specify nonempty supported cases')
    if not search_seeds or not final_seeds or set(search_seeds) & set(final_seeds):
        raise ValueError('Search and final seeds must be nonempty and disjoint')
    legacy_run = Path(legacy_run)
    legacy = json.loads((legacy_run / 'config.json').read_text())
    prior = legacy['prior']
    original, source = prior['original'], Path(prior['source'])
    if original['dataset'] != 'arxiv' or legacy['options']['steps'] != 300:
        raise ValueError('Use the original Arxiv 300-step CE bilevel experiment')
    if legacy['options'].get('assignment_rank') is not None or legacy['options'].get('mass_mode', 'free') != 'free':
        raise ValueError('Legacy reference must use dense assignments and free mass')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    graph, train_mask, validation, testing, h = _prepare_dataset('arxiv', data_dir, device)
    adjacency = graph['adj']
    digest = array_digest(*(t.cpu().numpy() for t in (
        graph['x'], graph['y'], adjacency.crow_indices(), adjacency.col_indices(), adjacency.values(),
        train_mask, validation[1], testing[1])))
    if digest != original['data_digest']:
        raise ValueError('Legacy graph or splits differ')
    logits = torch.load(source / 'teacher_logits.pt', map_location=device, weights_only=True)
    baseline = torch.load(source / 'baseline_partition.pt', map_location=device, weights_only=False)
    risk_assignment = baseline['assignment']
    if (array_digest(logits.cpu().numpy()) != prior['logits_digest'] or
            array_digest(risk_assignment.cpu().numpy()) != prior['assignment_digest']):
        raise ValueError('Legacy teacher or partition differs')
    cells = len(baseline['counts'])
    if cells != 909 or not torch.equal(torch.bincount(risk_assignment).cpu(), baseline['counts'].long().cpu()):
        raise ValueError('Require the original 909-cell partition')
    reference = torch.load(Path(legacy['previous_run']) / 'reference.pt', map_location=device, weights_only=False)
    transform = FeatureTransform(**reference['transform'])
    if transform.kind != 'rms':
        raise ValueError('Legacy transform must be RMS for inverse feature mapping')
    z = transform(h.double())
    params = original['params']
    q = (logits / params['T']).softmax(1).double()
    expected = torch.load(legacy_run / 'optimized.pt', map_location='cpu', weights_only=False)['initial_moments']
    options = solver_options(legacy['options'], 'risk_dense', checkpoint_steps, seed)
    initial = AssignmentMoments.apply(initial_logits(risk_assignment, cells, options['mixing']),
                                       make_material(z, q), options['chunk_size'])
    if not torch.allclose(initial.cpu(), expected, atol=1e-10, rtol=1e-8):
        raise ValueError('Risk initialization does not reproduce legacy initial moments')
    config = dict(legacy_run=str(legacy_run), legacy=legacy, checkpoint_steps=options['checkpoint_steps'],
                  search_seeds=list(search_seeds), final_seeds=list(final_seeds), seed=seed,
                  z_digest=array_digest(z.cpu().numpy()),
                  revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
    root = Path(output_dir) / _fingerprint(config)
    root.mkdir(parents=True, exist_ok=True)
    masks = dict(train=train_mask, val=validation[1], test=testing[1])
    search_masks = {key: masks[key] for key in ('train', 'val')}
    settings = {key: original[key] for key in ('epochs', 'eval_every', 'hidden')}
    settings.update(dropout=params['dropout'], lr=params['lr'], weight_decay=params['weight_decay'])
    kmeans = None
    for case in cases:
        folder = root / case
        folder.mkdir(exist_ok=True)
        save_json(dict(**config, case=case, student=settings), folder / 'config.json')
        if (folder / 'complete.json').exists():
            continue
        options = solver_options(legacy['options'], case, checkpoint_steps, seed)
        if case.startswith('risk'):
            assignment = risk_assignment
        else:
            if kmeans is None:
                path = folder / 'initial_assignment.pt'
                kmeans = torch.load(path, map_location=device, weights_only=True) if path.exists() else \
                    feature_kmeans(h.cpu(), cells, seed).to(device)
            assignment = kmeans
        torch.save(assignment.cpu(), folder / 'initial_assignment.pt')
        state_path = folder / 'resume.pt'
        state = torch.load(state_path, map_location='cpu', weights_only=False) if state_path.exists() else None
        if state is None or state['step'] < 300:
            optimize_ce_assignment(z, q, assignment, folder=folder, resume_state=state, **options)
        rows, source_rows = [], []
        for step in tqdm(options['checkpoint_steps'], desc=f'{case}: GCN validation'):
            snapshot = torch.load(folder / 'checkpoints' / f'step_{step:06d}.pt', map_location='cpu', weights_only=False)
            source_rows.append(dict(step=step, **sgc_metrics(snapshot, z, q, graph['y'], search_masks)))
            cx, cy, mass = representative(snapshot['moments'], transform, z.shape[1], device)
            for student_seed in search_seeds:
                row = fit_gcn_diagnostic(cx, cy, mass, graph, q, search_masks, student_seed,
                                         folder=folder / 'search' / f'step_{step:06d}', **settings)
                rows.append(dict(step=step, **row))
            pd.DataFrame(rows).to_csv(folder / 'search_students.csv', index=False)
        search = pd.DataFrame(rows).groupby('step', as_index=False).agg(
            val=('val_acc', 'mean'), val_std=('val_acc', lambda x: x.std(ddof=0)))
        search.to_csv(folder / 'search.csv', index=False)
        pd.DataFrame(source_rows).to_csv(folder / 'sgc_search.csv', index=False)
        selected = int(search.sort_values(['val', 'step'], ascending=[False, True]).iloc[0]['step'])
        save_json(dict(step=selected, selection='GCN validation'), folder / 'selected.json')
        records = []
        for step in sorted({0, selected, 300}):
            snapshot = torch.load(folder / 'checkpoints' / f'step_{step:06d}.pt', map_location='cpu', weights_only=False)
            cx, cy, mass = representative(snapshot['moments'], transform, z.shape[1], device)
            for student_seed in final_seeds:
                row = fit_gcn_diagnostic(cx, cy, mass, graph, q, masks, student_seed,
                                         folder=folder / 'final' / f'step_{step:06d}', **settings)
                records.append(dict(step=step, **row))
            pd.DataFrame(records).to_csv(folder / 'final_students.csv', index=False)
        final = pd.DataFrame(records)
        rows = []
        for name, step in [('initial', 0), ('selected', selected), ('step300', 300)]:
            group = final[final.step == step]
            rows.append(dict(case=case, phase=name, step=step, nodes=cells,
                             gamma=params.get('gamma'), T=params['T'], penalty=options['penalty'],
                             assignment_lr=options['lr'], val=group.val_acc.mean(), test=group.test_acc.mean(),
                             test_std=group.test_acc.std(ddof=0), output_dir=str(folder)))
        pd.DataFrame(rows).to_csv(folder / 'summary.csv', index=False)
        save_json(dict(complete=True), folder / 'complete.json')
    result = pd.concat([pd.read_csv(root / case / 'summary.csv') for case in CASES
                        if (root / case / 'complete.json').exists()], ignore_index=True)
    checks = []
    for initialization in ('risk', 'kmeans'):
        paths = [root / f'{initialization}_{encoder}' / 'checkpoints' / 'step_000000.pt'
                 for encoder in ('dense', 'mlp')]
        if all(path.exists() for path in paths):
            moments = [torch.load(path, map_location='cpu', weights_only=False)['moments'] for path in paths]
            if not torch.allclose(moments[0], moments[1], atol=1e-10, rtol=1e-8):
                raise ValueError('Dense and MLP initial moments differ')
            checks.append(dict(initialization=initialization, matched=True,
                               max_difference=float((moments[0] - moments[1]).abs().max())))
    pd.DataFrame(checks).to_csv(root / f'initial_checks_{cases[0]}.csv', index=False)
    return result, root
