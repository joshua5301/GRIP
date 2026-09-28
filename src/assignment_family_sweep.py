import json
import math
import subprocess
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm

from src.assignment_sweep import grid_rows, representative, teacher_logits
from src.condensation_diagnostics import feature_kmeans, fit_gcn_diagnostic, save_json
from src.node_distances import array_digest
from src.ntk_transforms import FeatureTransform, fit_transform
from src.risk_experiment import _fingerprint, _prepare_dataset
from src.soft_ce_partition import optimize_ce_assignment
from src.trajectory_clustering import save_state
from src.utils import BUDGET


METHODS = ('dense', 'low_rank', 'mlp1', 'mlp2')


def family_variants(methods, ranks, widths, cells):
    if not methods or set(methods) - set(METHODS) or len(set(methods)) != len(methods):
        raise ValueError('Choose distinct dense, low_rank, mlp1 and/or mlp2 methods')
    if (not ranks or not widths or any(not isinstance(r, int) or not 1 <= r <= cells for r in ranks)
            or any(not isinstance(w, int) or w < 1 for w in widths)):
        raise ValueError('Require positive widths and ranks no greater than the cell budget')
    variants = []
    for method in methods:
        if method == 'dense':
            variants.append(dict(method=method, rank=None, width=None))
        else:
            for rank in dict.fromkeys(ranks):
                for width in dict.fromkeys(widths) if method == 'mlp2' else (None,):
                    variants.append(dict(method=method, rank=rank, width=width))
    return variants


def family_options(variant):
    method = variant['method']
    return dict(assignment_rank=variant['rank'],
                assignment_input='features' if method in ('mlp1', 'mlp2') else 'node',
                assignment_encoder='mlp' if method == 'mlp2' else 'linear',
                encoder_hidden=variant['width'] if method == 'mlp2' else 64)


def parameter_count(variant, nodes, dimension, cells):
    method, rank, width = variant['method'], variant['rank'], variant['width']
    if method == 'dense':
        return nodes * cells
    if method == 'low_rank':
        return rank * (nodes + cells)
    if method == 'mlp1':
        return rank * (dimension + cells)
    return dimension * width + width + width * rank + rank + cells * rank


def validation_choice(table):
    return table.sort_values(['val', 'step', 'candidate'], ascending=[False, True, True]).iloc[0].to_dict()


def run_family_sweep(output_dir, space, gammas, ratio=.0005, methods=METHODS,
                      ranks=(8, 16, 32, 64), widths=(32, 64, 128), steps=200,
                      checkpoint_steps=(0, 25, 50, 100, 200), teacher_kernel='relu', basis=3000,
                      seed=0, mixing=.05, search_seeds=(0, 1, 2), final_seeds=tuple(range(100, 110)),
                      dropout=.31881090213944857, epochs=1000, eval_every=10, hidden=256,
                      student_lr=.01, weight_decay=.0005, solver=None,
                      data_dir='/content/data/', device='cuda'):
    if ('arxiv', ratio) not in BUDGET or set(space) != {'T', 'penalty', 'assignment_lr'}:
        raise ValueError('Use a configured Arxiv ratio and a T/penalty/assignment_lr grid')
    cells = BUDGET[('arxiv', ratio)]
    variants = family_variants(methods, ranks, widths, cells)
    common = grid_rows(space)
    if (not gammas or any(not math.isfinite(v) or v <= 0 for v in gammas)
            or any(not math.isfinite(v) or v <= 0 for row in common for v in row.values())):
        raise ValueError('Require positive finite teacher gamma and grid values')
    if not search_seeds or not final_seeds or set(search_seeds) & set(final_seeds):
        raise ValueError('Search and final student seeds must be nonempty and disjoint')
    checkpoints = sorted({0, steps, *checkpoint_steps})
    if steps < 1 or any(not isinstance(s, int) or s < 0 or s > steps for s in checkpoints):
        raise ValueError('Invalid step budget or checkpoints')
    solver = dict(solver or {})
    allowed = {'inner_max_iter', 'inner_tol', 'cg_max_iter', 'cg_rtol', 'chunk_size',
               'outer_chunk_size', 'solver_mode', 'tracking_inner_steps', 'tracking_cg_steps', 'tracking_refresh'}
    if set(solver) - allowed:
        raise ValueError('Unknown CE solver option')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    graph, train_mask, validation, testing, h = _prepare_dataset('arxiv', data_dir, device)
    adj = graph['adj']
    digest = array_digest(*(t.cpu().numpy() for t in (
        graph['x'], graph['y'], adj.crow_indices(), adj.col_indices(), adj.values(),
        train_mask, validation[1], testing[1])))
    config = dict(dataset='arxiv', ratio=ratio, cells=cells, space=space, gammas=list(gammas),
                  methods=list(methods), ranks=list(ranks), widths=list(widths), steps=steps,
                  checkpoint_steps=checkpoints, teacher_kernel=teacher_kernel, basis=basis,
                  seed=seed, mixing=mixing, search_seeds=list(search_seeds), final_seeds=list(final_seeds),
                  dropout=dropout, epochs=epochs, eval_every=eval_every, hidden=hidden,
                  student_lr=student_lr, weight_decay=weight_decay, solver=solver,
                  data_digest=digest, algorithm_version=1)
    root = Path(output_dir) / _fingerprint(config)
    root.mkdir(parents=True, exist_ok=True)
    save_json(config, root / 'config.json')
    if not (root / 'revision.txt').exists():
        (root / 'revision.txt').write_text(subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
    inputs_path = root / 'inputs.pt'
    if inputs_path.exists():
        saved = torch.load(inputs_path, map_location=device, weights_only=False)
        z, h, assignment = saved['z'], saved['h'], saved['assignment']
        transform = FeatureTransform(**saved['transform'])
    else:
        z, transform = fit_transform(h.double(), kind='rms')
        assignment = feature_kmeans(h.cpu(), cells, seed).to(device)
        save_state(dict(z=z, h=h, assignment=assignment, transform=vars(transform)), inputs_path)
    logits, gamma = teacher_logits(h, graph, train_mask, validation, teacher_kernel,
                                   list(gammas), basis, seed, root)
    teacher_table = pd.read_csv(root / 'teacher_grid.csv')
    teacher_val = float(teacher_table.val.max())
    save_json(dict(gamma=gamma, validation=teacher_val, selection='teacher validation accuracy'),
              root / 'selected_teacher.json')
    candidates = [dict(**variant, **params) for variant in variants for params in common]
    pd.DataFrame([dict(candidate=i, parameters=parameter_count(c, len(z), z.shape[1], cells), **c)
                  for i, c in enumerate(candidates)]).to_csv(root / 'candidates.csv', index=False)
    masks = dict(train=train_mask, val=validation[1], test=testing[1])
    search_masks = {key: masks[key] for key in ('train', 'val')}
    settings = dict(epochs=epochs, eval_every=eval_every, hidden=hidden, dropout=dropout,
                    lr=student_lr, weight_decay=weight_decay)

    def evaluate(snapshot, q, seeds, folder, evaluation_masks):
        cx, cy, mass = representative(snapshot['moments'], transform, z.shape[1], device)
        return [fit_gcn_diagnostic(cx, cy, mass, graph, q, evaluation_masks, student_seed,
                                   folder=folder, **settings) for student_seed in seeds]

    search, initial_checks = [], []
    for index, candidate in enumerate(tqdm(candidates, desc='Assignment family grid')):
        folder = root / f'candidate_{index:04d}'
        folder.mkdir(exist_ok=True)
        save_json(candidate, folder / 'params.json')
        q = (logits / candidate['T']).softmax(1).double()
        resume_path = folder / 'resume.pt'
        state = torch.load(resume_path, map_location='cpu', weights_only=False) if resume_path.exists() else None
        complete = folder / 'optimization_complete.json'
        if not complete.exists():
            result = optimize_ce_assignment(
                z, q, assignment, penalty=candidate['penalty'], lr=candidate['assignment_lr'],
                steps=steps, mixing=mixing, factor_seed=seed, folder=folder,
                checkpoint_steps=checkpoints, resume_state=state, save_resume=True,
                save_assignment=False, mass_mode='free', feature_control='joint',
                **family_options(candidate), **solver,
            )
            save_json(dict(parameters=result['assignment_parameters']), complete)
            del result
        del state
        initial = torch.load(folder / 'checkpoints' / 'step_000000.pt', map_location='cpu', weights_only=False)
        shared_path = root / f'initial_{_fingerprint(dict(T=candidate["T"]))}.pt'
        if not shared_path.exists():
            save_state(initial['moments'], shared_path)
        expected = torch.load(shared_path, map_location='cpu', weights_only=True)
        difference = float((initial['moments'] - expected).abs().max())
        if not torch.allclose(initial['moments'], expected, atol=1e-10, rtol=1e-8):
            raise ValueError('Initial moments differ between assignment parameterizations')
        initial_checks.append(dict(candidate=index, method=candidate['method'], T=candidate['T'],
                                   max_difference=difference, matched=True))
        pd.DataFrame(initial_checks).to_csv(root / 'initial_checks.csv', index=False)
        for step in checkpoints:
            snapshot = torch.load(folder / 'checkpoints' / f'step_{step:06d}.pt', map_location='cpu', weights_only=False)
            cache = (root / 'initial_search' / _fingerprint(dict(T=candidate['T'])) if step == 0
                     else folder / 'search' / f'step_{step:06d}')
            records = evaluate(snapshot, q, search_seeds, cache, search_masks)
            scores = pd.Series([row['val_acc'] for row in records])
            search.append(dict(candidate=index, step=step, **candidate, val=scores.mean(),
                               val_std=scores.std(ddof=0), outer_ce=snapshot['teacher_ce']))
            pd.DataFrame(search).to_csv(root / 'search.csv', index=False)
    table = pd.DataFrame(search)
    summaries, all_final = [], []
    for method in methods:
        choice = validation_choice(table[table.method == method])
        save_json(choice, root / f'selected_{method}.json')
        index, step = int(choice['candidate']), int(choice['step'])
        candidate = candidates[index]
        folder = root / f'candidate_{index:04d}'
        q = (logits / candidate['T']).softmax(1).double()
        for phase, current_step in [('initial', 0), ('selected', step)]:
            snapshot = torch.load(folder / 'checkpoints' / f'step_{current_step:06d}.pt', map_location='cpu', weights_only=False)
            cache = (root / 'initial_final' / _fingerprint(dict(T=candidate['T'])) if current_step == 0
                     else folder / 'final' / f'step_{current_step:06d}')
            records = evaluate(snapshot, q, final_seeds, cache, masks)
            group = pd.DataFrame(records)
            all_final.extend(dict(method=method, phase=phase, step=current_step, **row) for row in records)
            score = table[(table.candidate == index) & (table.step == current_step)].iloc[0]['val']
            summaries.append(dict(dataset='arxiv', ratio=ratio, nodes=cells, phase=phase, step=current_step,
                                  **candidate, gamma=gamma, teacher_val=teacher_val,
                                  dropout=dropout, loss_weighting='mass',
                                  parameters=parameter_count(candidate, len(z), z.shape[1], cells),
                                  search_val=score, final_val=group.val_acc.mean(),
                                  final_val_std=group.val_acc.std(ddof=0), test_mean=group.test_acc.mean(),
                                  test_std=group.test_acc.std(ddof=0), output_dir=str(root)))
        pd.DataFrame(summaries).to_csv(root / 'summary.csv', index=False)
        pd.DataFrame(all_final).to_csv(root / 'final_students.csv', index=False)
    return pd.DataFrame(summaries), root
