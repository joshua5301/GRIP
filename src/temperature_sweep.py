import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from src.assignment_family_sweep import family_options
from src.assignment_sweep import representative
from src.cmean_family_sweep import write_table
from src.condensation_diagnostics import fit_gcn_diagnostic, save_json
from src.node_distances import array_digest
from src.ntk_transforms import FeatureTransform
from src.risk_experiment import _fingerprint, _prepare_dataset
from src.soft_ce_partition import optimize_ce_assignment


def temperature_grid(inner_temperatures, outer_temperatures):
    inner, outer = [list(dict.fromkeys(map(float, values))) for values in
                    (inner_temperatures, outer_temperatures)]
    if not inner or not outer or any(not np.isfinite(t) or t <= 0 for t in inner + outer):
        raise ValueError('Temperatures must be finite and positive')
    return [dict(inner_T=i, outer_T=o) for i, o in product(inner, outer)]


def select_temperature(table):
    return table.sort_values(['val', 'step', 'candidate'], ascending=[False, True, True]).iloc[0].to_dict()


def run_temperature_sweep(previous_run, output_dir, inner_temperatures, outer_temperatures,
                          steps=1000, checkpoint_steps=(0, 100, 300, 500, 750, 1000),
                          search_seeds=(0, 1, 2), final_seeds=tuple(range(100, 110)),
                          shard=0, shards=1, data_dir='/content/data/', device='cuda'):
    candidates = temperature_grid(inner_temperatures, outer_temperatures)
    checkpoints = sorted({0, steps, *checkpoint_steps})
    if (not 0 <= shard < shards or steps < 1 or any(s < 0 or s > steps for s in checkpoints)
            or not search_seeds or not final_seeds):
        raise ValueError('Invalid shard, budget or evaluation seeds')
    source = Path(previous_run)
    prior = json.loads((source / 'config.json').read_text())
    selected = json.loads((source / 'selected_low_rank.json').read_text())
    params = json.loads((source / f'candidate_{int(selected["candidate"]):04d}' / 'params.json').read_text())
    solver = dict(prior['solver'], solver_mode='exact', implicit_warm_start=True,
                  inner_method='newton_first', cg_check_interval=1, cache_assignment=False)
    config = dict(source=str(source), prior=prior, params=params, candidates=candidates, solver=solver,
                  steps=steps, checkpoints=checkpoints, search_seeds=list(search_seeds),
                  final_seeds=list(final_seeds), version=1)
    root = Path(output_dir) / _fingerprint(config)
    root.mkdir(parents=True, exist_ok=True)
    save_json(config, root / f'config_shard_{shard}.json')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    graph, train, validation, testing, unused = _prepare_dataset(prior['dataset'], data_dir, device)
    del unused
    adj = graph['adj']
    digest = array_digest(*(t.cpu().numpy() for t in (graph['x'], graph['y'], adj.crow_indices(),
        adj.col_indices(), adj.values(), train, validation[1], testing[1])))
    if digest != prior['data_digest']:
        raise ValueError('Dataset differs from the source experiment')
    saved = torch.load(source / 'inputs.pt', map_location=device, weights_only=False)
    z, assignment = saved['z'], saved['assignment']
    transform = FeatureTransform(**saved['transform'])
    del saved
    teacher = torch.load(source / 'teacher.pt', map_location=device, weights_only=False)
    logits = teacher['logits'].detach()
    del teacher
    masks = dict(train=train, val=validation[1], test=testing[1])
    settings = dict(epochs=prior['epochs'], eval_every=prior['eval_every'], hidden=prior['hidden'],
                    dropout=prior['dropout'], lr=prior['student_lr'], weight_decay=prior['weight_decay'])

    def evaluate(index, step, seeds, final=False):
        folder = root / f'candidate_{index:04d}'
        snapshot = torch.load(folder / 'checkpoints' / f'step_{step:06d}.pt',
                              map_location='cpu', weights_only=False)
        cx, cy, mass = representative(snapshot['moments'], transform, z.shape[1], device)
        q = (logits / candidates[index]['outer_T']).softmax(1).double()
        cache = folder / ('final' if final else 'search') / f'step_{step:06d}'
        scores = pd.DataFrame([fit_gcn_diagnostic(cx, cy, mass, graph, q,
            masks if final else {k: masks[k] for k in ('train', 'val')}, seed,
            folder=cache, **settings) for seed in seeds])
        return scores, snapshot['teacher_ce']

    for index in tqdm(list(range(shard, len(candidates), shards)), desc='Inner/outer temperature grid'):
        candidate = candidates[index]
        folder = root / f'candidate_{index:04d}'
        folder.mkdir(exist_ok=True)
        save_json(candidate, folder / 'params.json')
        if not (folder / 'complete.json').exists():
            resume = folder / 'resume.pt'
            state = torch.load(resume, map_location='cpu', weights_only=False) if resume.exists() else None
            q_inner = (logits / candidate['inner_T']).softmax(1).double()
            q_outer = (logits / candidate['outer_T']).softmax(1).double()
            result = optimize_ce_assignment(z, q_inner, assignment,
                outer_targets=q_outer, penalty=params['penalty'], lr=params['assignment_lr'],
                steps=steps, mixing=prior['mixing'], factor_seed=prior['seed'],
                folder=folder, checkpoint_steps=checkpoints, resume_state=state,
                save_resume=True, save_assignment=False, **family_options(params), **solver)
            save_json(dict(steps=steps), folder / 'complete.json')
            del result, state, q_inner, q_outer
        rows = []
        for step in checkpoints:
            scores, outer_ce = evaluate(index, step, search_seeds)
            rows.append(dict(candidate=index, step=step, **candidate, val=scores.val_acc.mean(),
                             val_std=scores.val_acc.std(ddof=0), outer_ce=outer_ce))
        write_table(pd.DataFrame(rows), folder / 'search.csv')
    paths = [root / f'candidate_{i:04d}' / 'search.csv' for i in range(len(candidates))]
    available = [pd.read_csv(path) for path in paths if path.exists()]
    search = pd.concat(available, ignore_index=True) if available else pd.DataFrame()
    if len(available) != len(candidates) or shard != shards - 1:
        return pd.DataFrame(), search, root
    choices = dict(global_best=select_temperature(search))
    diagonal = search[np.isclose(search.inner_T, search.outer_T)]
    if not diagonal.empty:
        choices['best_shared_T'] = select_temperature(diagonal)
    reference = search[np.isclose(search.inner_T, params['T']) & np.isclose(search.outer_T, params['T'])]
    if not reference.empty:
        choices['source_T'] = select_temperature(reference)
    summary, final_rows = [], []
    for name, choice in choices.items():
        index, step = int(choice['candidate']), int(choice['step'])
        scores, _ = evaluate(index, step, final_seeds, final=True)
        summary.append(dict(selection=name, candidate=index, step=step,
            inner_T=choice['inner_T'], outer_T=choice['outer_T'], search_val=choice['val'],
            final_val=scores.val_acc.mean(), final_val_std=scores.val_acc.std(ddof=0),
            test_mean=scores.test_acc.mean(), test_std=scores.test_acc.std(ddof=0)))
        final_rows.extend(scores.assign(selection=name, candidate=index, step=step).to_dict('records'))
    write_table(search, root / f'search_shard_{shard}.csv')
    write_table(pd.DataFrame(summary), root / f'summary_shard_{shard}.csv')
    write_table(pd.DataFrame(final_rows), root / f'final_students_shard_{shard}.csv')
    save_json(choices, root / f'selected_shard_{shard}.json')
    return pd.DataFrame(summary), search, root
