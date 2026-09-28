import itertools
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from src.node_distances import array_digest
from src.ntk_transforms import fit_transform
from src.risk_experiment import _fingerprint, _prepare_dataset, _train_student
from src.risk_partition import risk_partition
from src.soft_ce_partition import optimize_ce_assignment
from src.soft_ridge_partition import decode_moments
from src.teacher import fit_logistic, get_kernel_values
from src.utils import BUDGET


def grid_rows(space):
    if not space or any(not isinstance(v, (list, tuple)) or not v for v in space.values()):
        raise ValueError('Require nonempty grid lists')
    return [dict(zip(space, values)) for values in itertools.product(*space.values())]


def promote(table, count):
    ranked = table.sort_values(['val', 'candidate', 'step', 'dropout'], ascending=[False, True, True, True])
    return ranked.drop_duplicates('candidate').head(count).candidate.tolist()


def boundary_profile(table, space):
    rows = []
    for key, values in space.items():
        eligible = table[table.step > 0] if key in ('penalty', 'assignment_lr') else table
        low, high = min(values), max(values)
        for value, group in eligible.groupby(key):
            rows.append(dict(parameter=key, value=value, best_val=group.val.max(),
                             boundary='fixed' if low == high else 'lower' if value == low else
                             'upper' if value == high else 'interior'))
    return pd.DataFrame(rows)


def representative(moments, transform, dimension, device):
    centers, labels, mass = decode_moments(moments.to(device), dimension)
    features = centers * transform.scale + transform.output_center + transform.center
    return features.float(), labels.float(), mass


def teacher_logits(h, train, mask, validation, kernel, gammas, basis, seed, folder, return_all=False):
    path = folder / ('teachers.pt' if return_all else 'teacher.pt')
    if path.exists():
        saved = torch.load(path, map_location=h.device, weights_only=False)
        return saved if return_all else (saved['logits'], saved['gamma'])
    generator = torch.Generator(device=h.device).manual_seed(seed)
    hd = h.double()
    anchors = hd if basis >= len(hd) else hd[torch.randperm(len(hd), device=h.device, generator=generator)[:basis]]
    gram = get_kernel_values(anchors, anchors, kernel)
    gram = (gram + gram.T) / 2
    eye = torch.eye(len(anchors), device=h.device, dtype=torch.double)
    chol = torch.linalg.cholesky(gram + 1e-8 * gram.diagonal().mean() * eye)
    mapping = torch.linalg.solve_triangular(chol, eye, upper=False).T
    def features(x):
        return torch.cat([get_kernel_values(block.double(), anchors, kernel) @ mapping for block in x.split(8192)])
    phi = features(h)
    graph, val_mask = validation
    if graph is train:
        val_phi, val_y = phi[val_mask], graph['y'][val_mask]
    else:
        val_h = torch.sparse.mm(graph['adj'], torch.sparse.mm(graph['adj'], graph['x']))
        val_phi, val_y = features(val_h), graph['y']
    targets = F.one_hot(train['y'][mask], int(train['y'].max()) + 1).double()
    rows, best, selected, logits = [], -float('inf'), None, None
    all_logits = {}
    for gamma in tqdm(gammas, desc='Teacher gamma validation'):
        weight = fit_logistic(phi[mask], targets, gamma)
        if return_all:
            all_logits[gamma] = (phi @ weight).detach().cpu()
        scores = val_phi @ weight
        val = float((scores.argmax(1) == val_y).double().mean())
        rows.append(dict(gamma=gamma, val=100 * val, val_ce=float(F.cross_entropy(scores, val_y))))
        if val > best:
            best, selected, logits = val, gamma, (phi @ weight).detach()
    pd.DataFrame(rows).to_csv(folder / 'teacher_grid.csv', index=False)
    if return_all:
        torch.save(all_logits, path)
        return all_logits
    torch.save(dict(logits=logits.cpu(), gamma=selected), path)
    return logits, selected


def run_assignment_sweep(dataset, ratios, output_dir, space, gammas, teacher_kernel='relu', basis=3000,
                         dropouts=(.5,), stage_steps=(300, 1000), keep=3,
                         search_seeds=(0, 1), refine_seeds=(0, 1, 2), final_seeds=tuple(range(100, 110)),
                         rank=16, encoder_hidden=64, seed=0, partition_B=1., partition_sweeps=30,
                         epochs=1000, eval_every=10, hidden=256, student_lr=.01, weight_decay=.0005,
                         data_dir='/content/data/', device='cuda', solver=None, full_grid=False,
                         assignment_steps=200, checkpoint_steps=(25, 50, 100, 150, 200),
                         feature_control='joint', initialization_source=None):
    if dataset not in ('cora', 'citeseer', 'flickr'):
        raise ValueError('Supported datasets: cora, citeseer, flickr')
    if set(space) != {'T', 'penalty', 'assignment_lr'}:
        raise ValueError('Sweep T, penalty and assignment_lr')
    if feature_control not in ('joint', 'assignment', 'direct'):
        raise ValueError('Unknown feature control')
    candidates = grid_rows(dict(gamma=list(gammas), **space) if full_grid else space)
    if any(not np.isfinite(v) or v <= 0 for row in candidates for v in row.values()):
        raise ValueError('Sweep values must be positive finite numbers')
    if (len(stage_steps) != 2 or not 0 < stage_steps[0] < stage_steps[1] or keep < 1
            or not gammas or not dropouts or not search_seeds or not refine_seeds or not final_seeds):
        raise ValueError('Invalid staged search settings')
    if set(final_seeds) & (set(search_seeds) | set(refine_seeds)):
        raise ValueError('Final student seeds must be separate from search seeds')
    solver = dict(solver or {})
    allowed = {'inner_max_iter', 'inner_tol', 'cg_max_iter', 'cg_rtol', 'chunk_size', 'outer_chunk_size',
               'solver_mode', 'tracking_inner_steps', 'tracking_cg_steps', 'tracking_refresh'}
    if set(solver) - allowed:
        raise ValueError('Unknown solver option')
    if full_grid and (assignment_steps < 1 or any(s < 1 or s > assignment_steps for s in checkpoint_steps)):
        raise ValueError('Checkpoints must lie within the assignment budget')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    train, mask, validation, testing, h = _prepare_dataset(dataset, data_dir, device)
    z, transform = fit_transform(h.double(), kind='rms')
    def graph_arrays(graph):
        adj = graph['adj'].to_sparse_csr()
        return [value.cpu().numpy() for value in (graph['x'], graph['y'], adj.crow_indices(),
                                                 adj.col_indices(), adj.values())]
    config = dict(full_grid=full_grid, feature_control=feature_control,
                  initialization_source=str(initialization_source) if initialization_source is not None else None,
                  assignment_steps=assignment_steps, checkpoint_steps=list(checkpoint_steps),
                  dataset=dataset, ratios=list(ratios), space=space, gammas=list(gammas),
                  teacher_kernel=teacher_kernel, basis=basis, dropouts=list(dropouts),
                  stage_steps=list(stage_steps), keep=keep, search_seeds=list(search_seeds),
                  refine_seeds=list(refine_seeds), final_seeds=list(final_seeds), rank=rank,
                  encoder_hidden=encoder_hidden, seed=seed, partition_B=partition_B,
                  partition_sweeps=partition_sweeps, epochs=epochs, eval_every=eval_every, hidden=hidden,
                  student_lr=student_lr, weight_decay=weight_decay, solver=solver,
                  data_digest=array_digest(h.cpu().numpy(), mask.cpu().numpy(), *graph_arrays(train),
                                           *graph_arrays(validation[0]), *graph_arrays(testing[0]),
                                           *(m.cpu().numpy() for _, m in (validation, testing) if m is not None)),
                  revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
    root = Path(output_dir) / _fingerprint(config)
    root.mkdir(parents=True, exist_ok=True)
    (root / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    source = Path(initialization_source) if initialization_source is not None else root
    if initialization_source is not None:
        source_config = json.loads((source / 'config.json').read_text())
        keys = ('dataset', 'ratios', 'gammas', 'teacher_kernel', 'basis', 'seed', 'partition_B',
                'partition_sweeps', 'data_digest', 'full_grid', 'revision')
        if (any(source_config[key] != config[key] for key in keys)
                or source_config['space']['T'] != list(space['T'])):
            raise ValueError('Shared initialization settings differ')
        if not (source / ('teachers.pt' if full_grid else 'teacher.pt')).exists():
            raise ValueError('Shared teacher cache is missing')
    teachers = teacher_logits(h, train, mask, validation, teacher_kernel, gammas, basis, seed, source,
                              return_all=full_grid)
    if not full_grid:
        logits, gamma = teachers
        teachers = {gamma: logits}
        candidates = [dict(gamma=gamma, **row) for row in candidates]
    torch.save(transform.state_dict(), root / 'transform.pt')
    settings = dict(epochs=epochs, eval_every=eval_every, hidden=hidden, loss_weighting='mass')
    summaries = []
    for ratio in ratios:
        cells = BUDGET[(dataset, ratio)]
        if rank > cells:
            raise ValueError('rank exceeds cell budget')
        folder = root / f'ratio_{ratio:g}'
        folder.mkdir(exist_ok=True)
        assignments = {}
        for gamma, temperature in itertools.product(teachers, dict.fromkeys(space['T'])):
            q = (teachers[gamma].to(device) / temperature).softmax(1).double()
            path = source / f'ratio_{ratio:g}' / f'initial_{_fingerprint([gamma, temperature])}.pt'
            if initialization_source is not None and not path.exists():
                raise ValueError('Shared initial partition is missing')
            if path.exists():
                assignment = torch.load(path, map_location=device, weights_only=True)
            else:
                partition = risk_partition(h, q, cells, partition_B, seed=seed,
                                           max_sweeps=partition_sweeps, return_assignment=True)
                assignment = partition['assignment'].to(device)
                torch.save(assignment.cpu(), path)
            assignments[gamma, temperature] = assignment

        def evaluate(moments, params, seeds, destination, test=False):
            destination.mkdir(parents=True, exist_ok=True)
            cx, cy, mass = representative(moments, transform, z.shape[1], device)
            records = []
            for student_seed in seeds:
                path = destination / f'seed_{student_seed}.json'
                if path.exists():
                    record = json.loads(path.read_text())
                else:
                    val, value, epoch = _train_student(cx, cy, validation, params, student_seed, settings,
                                                     testing=testing if test else None, counts=mass)
                    record = dict(seed=student_seed, val=100 * val,
                                  test=100 * value if value is not None else None, epoch=epoch)
                    path.write_text(json.dumps(record), encoding='utf-8')
                records.append(record)
            return pd.DataFrame(records)

        def stage(indices, budget, seeds, name):
            rows = []
            for index in tqdm(indices, desc=f'{dataset} {ratio:g} {name}'):
                candidate = candidates[index]
                path = folder / f'candidate_{index:03d}'
                path.mkdir(exist_ok=True)
                (path / 'params.json').write_text(json.dumps(candidate, indent=2))
                q = (teachers[candidate['gamma']].to(device) / candidate['T']).softmax(1).double()
                checkpoints = sorted({0, min(100, budget), min(300, budget), budget} |
                                     ({500, 750} if budget >= 1000 else set()))
                if full_grid:
                    checkpoints = sorted({0, budget, *checkpoint_steps})
                state_path = path / 'resume.pt'
                state = torch.load(state_path, map_location='cpu', weights_only=False) if state_path.exists() else None
                if state is None or state['step'] < budget:
                    optimized = optimize_ce_assignment(
                        z, q, assignments[candidate['gamma'], candidate['T']], penalty=candidate['penalty'], steps=budget,
                        lr=candidate['assignment_lr'], folder=path, checkpoint_steps=checkpoints,
                        assignment_rank=rank, factor_seed=seed, assignment_input='features',
                        assignment_encoder='mlp', encoder_hidden=encoder_hidden,
                        save_assignment=False, save_resume=True, resume_state=state,
                        feature_control=feature_control, **solver)
                    del optimized
                for step in checkpoints:
                    snapshot = torch.load(path / 'checkpoints' / f'step_{step:06d}.pt', map_location='cpu', weights_only=False)
                    if full_grid and step == 0 and any(
                            r['step'] == 0 and r['gamma'] == candidate['gamma'] and r['T'] == candidate['T']
                            for r in rows):
                        continue
                    for dropout in dropouts:
                        params = dict(dropout=dropout, lr=student_lr, weight_decay=weight_decay)
                        destination = path / f'eval_{step}_{dropout:g}'
                        if full_grid and step == 0:
                            destination = folder / f'baseline_{_fingerprint([candidate["gamma"], candidate["T"], dropout])}'
                        records = evaluate(snapshot['moments'], params, seeds, destination)
                        rows.append(dict(candidate=index, step=step, dropout=dropout, **candidate,
                                         val=records.val.mean(), val_std=records.val.std(ddof=0),
                                         teacher_ce=snapshot['teacher_ce'], stage=name))
                pd.DataFrame(rows).to_csv(folder / f'{name}.csv', index=False)
            return pd.DataFrame(rows)

        if full_grid:
            refined = stage(range(len(candidates)), assignment_steps, search_seeds, 'full_grid')
            baseline = refined[refined.step == 0].copy()
            baseline[['penalty', 'assignment_lr']] = np.nan
            baseline.to_csv(folder / 'baselines.csv', index=False)
            boundary_profile(refined, dict(gamma=list(gammas), **space)).to_csv(folder / 'boundaries.csv', index=False)
        else:
            screen = stage(range(len(candidates)), stage_steps[0], search_seeds, 'screen')
            promoted = promote(screen, keep)
            alternate = screen[~screen.candidate.isin(promoted)]
            chosen_lrs = {candidates[i]['assignment_lr'] for i in promoted}
            alternate = alternate[~alternate.assignment_lr.isin(chosen_lrs)]
            if not alternate.empty:
                promoted += promote(alternate, 1)
            (folder / 'promoted.json').write_text(json.dumps(promoted))
            refined = stage(promoted, stage_steps[1], refine_seeds, 'refine')
        choice = refined.sort_values(['val', 'candidate', 'step', 'dropout'],
                                     ascending=[False, True, True, True]).iloc[0].to_dict()
        if full_grid and choice['step'] == 0:
            choice['penalty'] = choice['assignment_lr'] = None
        (folder / 'selected.json').write_text(json.dumps(choice, indent=2))
        index, step = int(choice['candidate']), int(choice['step'])
        snapshot = torch.load(folder / f'candidate_{index:03d}' / 'checkpoints' / f'step_{step:06d}.pt',
                              map_location='cpu', weights_only=False)
        params = dict(dropout=choice['dropout'], lr=student_lr, weight_decay=weight_decay)
        final = evaluate(snapshot['moments'], params, final_seeds,
                         folder / f'final_{index}_{step}_{choice["dropout"]:g}', test=True)
        final.to_csv(folder / 'final_students.csv', index=False)
        cx, cy, mass = representative(snapshot['moments'], transform, z.shape[1], device)
        torch.save(dict(x=cx.cpu(), y=cy.cpu(), mass=mass.cpu(), choice=choice), folder / 'selected_partition.pt')
        summaries.append(dict(dataset=dataset, ratio=ratio, nodes=len(cx), gamma=choice['gamma'],
                              feature_control=feature_control, loss_weighting='mass',
                              **{key: choice[key] for key in ('T', 'penalty', 'assignment_lr', 'dropout')},
                              selected_step=step, search_val=choice['val'], final_val=final.val.mean(),
                              final_val_std=final.val.std(ddof=0), test_mean=final.test.mean(),
                              test_std=final.test.std(ddof=0), output_dir=str(root)))
        pd.DataFrame(summaries).to_csv(root / 'summary.csv', index=False)
    return pd.DataFrame(summaries)
