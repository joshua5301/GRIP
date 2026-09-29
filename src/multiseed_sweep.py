import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from src.assignment_sweep import grid_rows, representative, teacher_logits, boundary_profile
from src.cmean_family_sweep import write_table
from src.condensation_diagnostics import feature_kmeans, fit_gcn_diagnostic, save_json
from src.node_distances import array_digest
from src.ntk_transforms import fit_transform, FeatureTransform
from src.risk_experiment import _fingerprint, _prepare_dataset
from src.soft_ce_partition import optimize_ce_assignment
from src.trajectory_clustering import save_state
from src.utils import BUDGET


def aggregate_search(records, condensation_seeds, student_seeds):
    frame = pd.DataFrame(records)
    keys = ['candidate', 'T', 'rank', 'penalty', 'step']
    expected = {(a, b) for a in condensation_seeds for b in student_seeds}
    rows = []
    for key, group in frame.groupby(keys, sort=False):
        observed = set(zip(group.condensation_seed, group.seed))
        if observed != expected or len(group) != len(expected):
            raise ValueError('Each checkpoint must contain exactly the configured seed pairs')
        means = group.groupby('condensation_seed').val_acc.mean()
        rows.append(dict(zip(keys, key), val=means.mean(),
                         condensation_val_std=means.std(ddof=1), evaluations=len(group)))
    return pd.DataFrame(rows)


def run_cora_multiseed(ratio, output_dir, space, gammas, steps=1000,
                       checkpoint_steps=(0, 100, 300, 500, 750, 1000),
                       condensation_seeds=(0, 1, 2), search_seeds=(0, 1, 2),
                       final_seeds=tuple(range(100, 110)), teacher_seed=0,
                       assignment_lr=.01, basis=3000, dropout=.9,
                       epochs=1000, eval_every=10, hidden=256,
                       student_lr=.01, weight_decay=.0005,
                       data_dir='/content/data/', device='cuda'):
    if ('cora', ratio) not in BUDGET or set(space) != {'T', 'rank', 'penalty'}:
        raise ValueError('Use a configured Cora ratio and T/rank/penalty grid')
    cells = BUDGET[('cora', ratio)]
    candidates = grid_rows(space)
    if any(not np.isfinite(c[k]) or c[k] <= 0 for c in candidates for k in ('T', 'penalty')):
        raise ValueError('T and penalty must be positive and finite')
    if any(not isinstance(c['rank'], int) or not 1 <= c['rank'] <= cells for c in candidates):
        raise ValueError('Rank must be an integer no greater than the cell budget')
    if (not gammas or any(not np.isfinite(g) or g <= 0 for g in gammas)
            or not search_seeds or not final_seeds or len(condensation_seeds) < 2
            or set(search_seeds) & set(final_seeds)
            or any(len(set(s)) != len(s) for s in (condensation_seeds, search_seeds, final_seeds))):
        raise ValueError('Invalid gamma grid or seed configuration')
    checkpoints = sorted({0, steps, *checkpoint_steps})
    if steps < 1 or any(not isinstance(s, int) or not 0 <= s <= steps for s in checkpoints):
        raise ValueError('Invalid checkpoint budget')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    graph, train, validation, testing, h = _prepare_dataset('cora', data_dir, device)
    adj = graph['adj']
    digest = array_digest(*(t.cpu().numpy() for t in (graph['x'], graph['y'], adj.crow_indices(),
        adj.col_indices(), adj.values(), train, validation[1], testing[1])))
    solver = dict(solver_mode='exact', inner_method='newton_first', implicit_warm_start=True,
                  inner_loss_weighting='mass', cg_check_interval=1, cache_assignment=False,
                  inner_max_iter=2000, inner_tol=1e-7, cg_max_iter=512, cg_rtol=1e-6)
    settings = dict(epochs=epochs, eval_every=eval_every, hidden=hidden, dropout=dropout,
                    lr=student_lr, weight_decay=weight_decay)
    config = dict(dataset='cora', ratio=ratio, cells=cells, space=space, gammas=list(gammas),
        steps=steps, checkpoints=checkpoints, condensation_seeds=list(condensation_seeds),
        search_seeds=list(search_seeds), final_seeds=list(final_seeds), teacher_seed=teacher_seed,
        assignment_lr=assignment_lr, basis=basis, teacher_kernel='relu', mixing=.05,
        student=settings, student_loss='uniform', solver=solver, data_digest=digest, version=1)
    root = Path(output_dir) / f'ratio_{ratio}' / _fingerprint(config)
    root.mkdir(parents=True, exist_ok=True)
    save_json(config, root / 'config.json')
    logits, gamma = teacher_logits(h, graph, train, validation, 'relu', list(gammas),
                                   basis, teacher_seed, root)
    save_json(dict(gamma=gamma, teacher_seed=teacher_seed,
                   logits_digest=array_digest(logits.cpu().numpy())), root / 'selected_teacher.json')
    inputs = root / 'inputs.pt'
    if inputs.exists():
        saved = torch.load(inputs, map_location=device, weights_only=False)
        z, assignments = saved['z'], saved['assignments']
        transform = FeatureTransform(**saved['transform'])
        del saved
    else:
        z, transform = fit_transform(h.double(), kind='rms')
        assignments = {s: feature_kmeans(h.cpu(), cells, s).to(device) for s in condensation_seeds}
        save_state(dict(z=z, assignments=assignments, transform=vars(transform)), inputs)
    masks = dict(train=train, val=validation[1], test=testing[1])

    def evaluate(index, seed, step, final=False):
        candidate = candidates[index]
        folder = root / f'candidate_{index:04d}' / f'condensation_{seed}'
        snapshot = torch.load(folder / 'checkpoints' / f'step_{step:06d}.pt',
                              map_location='cpu', weights_only=False)
        x, y, mass = representative(snapshot['moments'], transform, z.shape[1], device)
        weights = torch.full_like(mass, 1 / len(mass))
        cache = folder / ('final' if final else 'search') / f'step_{step:06d}'
        q = (logits / candidate['T']).softmax(1).double()
        return [dict(candidate=index, **candidate, condensation_seed=seed, step=step,
            **fit_gcn_diagnostic(x, y, weights, graph, q,
                masks if final else {k: masks[k] for k in ('train', 'val')}, student_seed,
                folder=cache, **settings)) for student_seed in (final_seeds if final else search_seeds)]

    search_records = []
    for index, candidate in enumerate(tqdm(candidates, desc=f'Cora {ratio}: joint seed grid')):
        q = (logits / candidate['T']).softmax(1).double()
        for seed in condensation_seeds:
            folder = root / f'candidate_{index:04d}' / f'condensation_{seed}'
            folder.mkdir(parents=True, exist_ok=True)
            if not (folder / 'complete.json').exists():
                resume = folder / 'resume.pt'
                state = torch.load(resume, map_location='cpu', weights_only=False) if resume.exists() else None
                result = optimize_ce_assignment(z, q, assignments[seed], penalty=candidate['penalty'],
                    lr=assignment_lr, steps=steps, assignment_rank=candidate['rank'], factor_seed=seed,
                    mixing=.05, folder=folder, checkpoint_steps=checkpoints,
                    resume_state=state, save_resume=True, save_assignment=False, **solver)
                save_json(dict(steps=steps), folder / 'complete.json')
                del result, state
            for step in checkpoints:
                search_records.extend(evaluate(index, seed, step))
        write_table(pd.DataFrame(search_records), root / 'search_students.csv')
        search = aggregate_search(search_records, condensation_seeds, search_seeds)
        write_table(search, root / 'search.csv')
    chosen = search.sort_values(['val', 'step', 'candidate'], ascending=[False, True, True]).iloc[0].to_dict()
    save_json(chosen, root / 'selected.json')
    index, selected_step = int(chosen['candidate']), int(chosen['step'])
    final = []
    for phase, step in [('initial', 0), ('selected', selected_step)]:
        for seed in condensation_seeds:
            final.extend(dict(phase=phase, **r) for r in evaluate(index, seed, step, final=True))
        write_table(pd.DataFrame(final), root / 'final_students.csv')
    final = pd.DataFrame(final)
    by_seed = final.groupby(['phase', 'condensation_seed'], sort=False).agg(
        val_mean=('val_acc', 'mean'), test_mean=('test_acc', 'mean'),
        test_student_std=('test_acc', lambda v: v.std(ddof=0))).reset_index()
    summary = by_seed.groupby('phase', sort=False).agg(
        final_val=('val_mean', 'mean'), test_mean=('test_mean', 'mean'),
        test_condensation_std=('test_mean', 'std'),
        mean_student_std=('test_student_std', 'mean')).reset_index()
    summary['dataset'], summary['ratio'], summary['nodes'] = 'cora', ratio, cells
    summary['gamma'] = gamma
    for key in ('T', 'rank', 'penalty'):
        summary[key] = candidates[index][key]
    summary['step'] = summary.phase.map({'initial': 0, 'selected': selected_step})
    summary['search_val'] = summary['step'].map(search[search.candidate == index].set_index('step').val)
    write_table(by_seed, root / 'by_condensation_seed.csv')
    write_table(summary, root / 'summary.csv')
    write_table(boundary_profile(search, space), root / 'boundaries.csv')
    return summary, by_seed, search, root
