import json
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm

from src.assignment_family_sweep import validation_choice
from src.assignment_sweep import boundary_profile, grid_rows, representative
from src.condensation_diagnostics import fit_gcn_diagnostic, save_json
from src.node_distances import array_digest
from src.ntk_transforms import FeatureTransform
from src.risk_experiment import _fingerprint, _prepare_dataset
from src.soft_ce_partition import optimize_ce_assignment


def shard_candidates(space, shard, shards):
    if not isinstance(shards, int) or not isinstance(shard, int) or not 0 <= shard < shards:
        raise ValueError('Invalid zero-based shard index')
    if set(space) != {'T', 'penalty', 'assignment_lr'}:
        raise ValueError('Specify T, penalty and assignment_lr')
    candidates = grid_rows(space)
    if any(not 0 < float(v) < float('inf') for p in candidates for v in p.values()):
        raise ValueError('Grid values must be positive and finite')
    return [(i, p) for i, p in enumerate(candidates) if i % shards == shard]


def write_table(table, path):
    path = Path(path)
    temporary = path.with_suffix('.tmp.csv')
    table.to_csv(temporary, index=False)
    temporary.replace(path)


def run_cmean_shard(previous_run, output_dir, space, shard=0, shards=3, steps=1000,
                    checkpoint_steps=(0, 25, 50, 100, 200, 300, 500, 750, 1000),
                    data_dir='/content/data/', device='cuda'):
    assigned = shard_candidates(space, shard, shards)
    checkpoints = sorted({0, steps, *checkpoint_steps})
    if steps < 1 or any(not isinstance(s, int) or s < 0 or s > steps for s in checkpoints):
        raise ValueError('Invalid checkpoint budget')
    source = Path(previous_run)
    prior = json.loads((source / 'config.json').read_text())
    config = dict(source=str(source), prior=prior, space=space, shards=shards,
                  steps=steps, checkpoint_steps=checkpoints, algorithm_version=1)
    root = Path(output_dir) / _fingerprint(config)
    root.mkdir(parents=True, exist_ok=True)
    shard_dir = root / f'shard_{shard}'
    shard_dir.mkdir(exist_ok=True)
    save_json(config, shard_dir / 'config.json')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    graph, train, validation, testing, unused = _prepare_dataset(prior['dataset'], data_dir, device)
    del unused
    adj = graph['adj']
    digest = array_digest(*(t.cpu().numpy() for t in (
        graph['x'], graph['y'], adj.crow_indices(), adj.col_indices(), adj.values(),
        train, validation[1], testing[1])))
    if digest != prior['data_digest']:
        raise ValueError('Dataset differs from the source family sweep')
    saved = torch.load(source / 'inputs.pt', map_location=device, weights_only=False)
    z, assignment = saved['z'], saved['assignment']
    transform = FeatureTransform(**saved['transform'])
    del saved
    teacher = torch.load(source / 'teacher.pt', map_location=device, weights_only=False)
    logits, gamma = teacher['logits'], teacher['gamma']
    del teacher
    masks = dict(train=train, val=validation[1], test=testing[1])
    settings = dict(epochs=prior['epochs'], eval_every=prior['eval_every'], hidden=prior['hidden'],
                    dropout=prior['dropout'], lr=prior['student_lr'], weight_decay=prior['weight_decay'])

    def evaluate(snapshot, q, seeds, cache, final=False):
        cx, cy, mass = representative(snapshot['moments'], transform, z.shape[1], device)
        selected_masks = masks if final else {k: masks[k] for k in ('train', 'val')}
        return pd.DataFrame([fit_gcn_diagnostic(cx, cy, mass, graph, q, selected_masks, seed,
                            folder=cache, **settings) for seed in seeds])

    rows = []
    for index, params in tqdm(assigned, desc=f'C-mean shard {shard + 1}/{shards}'):
        folder = shard_dir / f'candidate_{index:04d}'
        folder.mkdir(exist_ok=True)
        save_json(params, folder / 'params.json')
        q = (logits / params['T']).softmax(1).double()
        resume = folder / 'resume.pt'
        state = torch.load(resume, map_location='cpu', weights_only=False) if resume.exists() else None
        if state is None or state['step'] < steps:
            result = optimize_ce_assignment(
                z, q, assignment, penalty=params['penalty'], lr=params['assignment_lr'], steps=steps,
                mixing=prior['mixing'], folder=folder, checkpoint_steps=checkpoints,
                assignment_rank=8, factor_seed=prior['seed'], feature_control='direct',
                save_assignment=False, save_resume=True, resume_state=state, **prior['solver'])
            del result
        del state
        initial = torch.load(folder / 'checkpoints' / 'step_000000.pt', map_location='cpu', weights_only=False)
        reference = source / f'initial_{_fingerprint(dict(T=params["T"]))}.pt'
        if reference.exists():
            expected = torch.load(reference, map_location='cpu', weights_only=True)
            if not torch.allclose(initial['moments'], expected, atol=1e-10, rtol=1e-8):
                raise ValueError('C-mean initialization differs from the source sweep')
        for step in checkpoints:
            snapshot = torch.load(folder / 'checkpoints' / f'step_{step:06d}.pt',
                                  map_location='cpu', weights_only=False)
            cache = folder / 'search' / f'step_{step:06d}'
            if step == 0:
                cache = shard_dir / 'initial_search' / _fingerprint(dict(T=params['T']))
            scores = evaluate(snapshot, q, prior['search_seeds'], cache)
            rows.append(dict(candidate=index, shard=shard, step=step, **params,
                             val=scores.val_acc.mean(), val_std=scores.val_acc.std(ddof=0),
                             outer_ce=snapshot['teacher_ce']))
            write_table(pd.DataFrame(rows), shard_dir / 'search.csv')
    save_json(dict(complete=True, candidates=len(assigned)), shard_dir / 'complete.json')
    complete = [root / f'shard_{s}' for s in range(shards)]
    if shard != shards - 1 or not all((p / 'complete.json').exists() for p in complete):
        return pd.DataFrame(), pd.DataFrame(rows), root
    search = pd.concat([pd.read_csv(p / 'search.csv') for p in complete], ignore_index=True)
    write_table(search, root / 'search.csv')
    write_table(boundary_profile(search, space), root / 'boundaries.csv')
    choice = validation_choice(search)
    save_json(choice, root / 'selected.json')
    folder = root / f'shard_{int(choice["shard"])}' / f'candidate_{int(choice["candidate"]):04d}'
    q = (logits / choice['T']).softmax(1).double()
    summaries = []
    for phase, step in [('initial', 0), ('selected', int(choice['step']))]:
        snapshot = torch.load(folder / 'checkpoints' / f'step_{step:06d}.pt', map_location='cpu', weights_only=False)
        scores = evaluate(snapshot, q, prior['final_seeds'], folder / 'final' / f'step_{step:06d}', final=True)
        summaries.append(dict(method='C-mean', phase=phase, step=step, dataset=prior['dataset'],
            ratio=prior['ratio'], nodes=prior['cells'], gamma=gamma, loss_weighting='mass',
            **{k: choice[k] for k in space}, final_val=scores.val_acc.mean(),
            final_val_std=scores.val_acc.std(ddof=0), test_mean=scores.test_acc.mean(),
            test_std=scores.test_acc.std(ddof=0), search_val=float(search.loc[
                (search.candidate == choice['candidate']) & (search.step == step), 'val'].iloc[0])))
    result = pd.DataFrame(summaries)
    write_table(result, root / 'summary.csv')
    return result, search, root
