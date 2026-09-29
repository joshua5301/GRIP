import json
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm

from src.assignment_family_sweep import validation_choice
from src.assignment_sweep import grid_rows, representative
from src.cmean_family_sweep import write_table
from src.condensation_diagnostics import fit_gcn_diagnostic, save_json
from src.node_distances import array_digest
from src.ntk_transforms import FeatureTransform
from src.risk_experiment import _fingerprint, _prepare_dataset
from src.soft_ce_partition import optimize_ce_assignment


def run_node_weight_sweep(previous_run, output_dir, space, shard=0, shards=3, steps=300,
                          checkpoint_steps=(0, 25, 50, 100, 200, 300),
                          data_dir='/content/data/', device='cuda'):
    if set(space) != {'node_weight_penalty', 'node_weight_lr'} or not 0 <= shard < shards:
        raise ValueError('Specify node_weight_penalty/node_weight_lr and a valid shard')
    candidates = grid_rows(space)
    if any(not 0 <= p['node_weight_penalty'] < float('inf') or
           not 0 < p['node_weight_lr'] < float('inf') for p in candidates):
        raise ValueError('Require nonnegative regularization and positive weight learning rate')
    source = Path(previous_run)
    prior = json.loads((source / 'config.json').read_text())
    selection = json.loads((source / 'selected_low_rank.json').read_text())
    original = source / f'candidate_{int(selection["candidate"]):04d}'
    fixed = json.loads((original / 'params.json').read_text())
    checkpoints = sorted({0, steps, *checkpoint_steps})
    if steps < 1 or any(not isinstance(s, int) or not 0 <= s <= steps for s in checkpoints):
        raise ValueError('Invalid step budget')
    config = dict(source=str(source), prior=prior, fixed=fixed, space=space, shards=shards,
                  steps=steps, checkpoint_steps=checkpoints, algorithm_version=1)
    root = Path(output_dir) / _fingerprint(config)
    part = root / f'shard_{shard}'
    part.mkdir(parents=True, exist_ok=True)
    save_json(config, part / 'config.json')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    graph, train, validation, testing, unused = _prepare_dataset(prior['dataset'], data_dir, device)
    del unused
    adj = graph['adj']
    digest = array_digest(*(t.cpu().numpy() for t in (
        graph['x'], graph['y'], adj.crow_indices(), adj.col_indices(), adj.values(),
        train, validation[1], testing[1])))
    if digest != prior['data_digest']:
        raise ValueError('Dataset differs from source experiment')
    saved = torch.load(source / 'inputs.pt', map_location=device, weights_only=False)
    z, assignment = saved['z'], saved['assignment']
    transform = FeatureTransform(**saved['transform'])
    del saved
    teacher = torch.load(source / 'teacher.pt', map_location=device, weights_only=False)
    q = (teacher['logits'] / fixed['T']).softmax(1).double()
    gamma = teacher['gamma']
    del teacher
    masks = dict(train=train, val=validation[1], test=testing[1])
    settings = dict(epochs=prior['epochs'], eval_every=prior['eval_every'], hidden=prior['hidden'],
                    dropout=prior['dropout'], lr=prior['student_lr'], weight_decay=prior['weight_decay'])
    expected = torch.load(original / 'checkpoints' / 'step_000000.pt', map_location='cpu', weights_only=False)['moments']
    baseline = pd.read_csv(source / 'search.csv')
    baseline = baseline[(baseline.candidate == selection['candidate']) & (baseline.step <= steps)].copy()
    extended = original / 'extended' / 'search.csv'
    if extended.exists():
        baseline = pd.concat([baseline, pd.read_csv(extended)], ignore_index=True)
        baseline = baseline[baseline.step <= steps].drop_duplicates('step', keep='first')
    baseline['method'] = 'low_rank'

    def evaluate(folder, step, final=False):
        snapshot = torch.load(folder / 'checkpoints' / f'step_{step:06d}.pt', map_location='cpu', weights_only=False)
        cx, cy, mass = representative(snapshot['moments'], transform, z.shape[1], device)
        cache = folder / ('final' if final else 'search') / f'step_{step:06d}'
        if step == 0:
            cache = part / ('initial_final' if final else 'initial_search')
        seeds = prior['final_seeds'] if final else prior['search_seeds']
        selected_masks = masks if final else {k: masks[k] for k in ('train', 'val')}
        scores = pd.DataFrame([fit_gcn_diagnostic(cx, cy, mass, graph, q, selected_masks, seed,
                               folder=cache, **settings) for seed in seeds])
        return scores, snapshot

    rows = []
    assigned = [(i, p) for i, p in enumerate(candidates) if i % shards == shard]
    for index, params in tqdm(assigned, desc=f'Node weights {shard + 1}/{shards}'):
        folder = part / f'candidate_{index:04d}'
        folder.mkdir(exist_ok=True)
        save_json(params, folder / 'params.json')
        resume = folder / 'resume.pt'
        state = torch.load(resume, map_location='cpu', weights_only=False) if resume.exists() else None
        if state is None or state['step'] < steps:
            result = optimize_ce_assignment(
                z, q, assignment, penalty=fixed['penalty'], lr=fixed['assignment_lr'], steps=steps,
                mixing=prior['mixing'], assignment_rank=fixed['rank'], factor_seed=prior['seed'],
                folder=folder, checkpoint_steps=checkpoints, save_resume=True, save_assignment=False,
                resume_state=state, node_weighting=True, **params, **prior['solver'])
            del result
        del state
        initial = torch.load(folder / 'checkpoints' / 'step_000000.pt', map_location='cpu', weights_only=False)
        if not torch.allclose(initial['moments'], expected, atol=1e-10, rtol=1e-8):
            raise ValueError('Weighted initialization differs from the unweighted baseline')
        for step in checkpoints:
            scores, snapshot = evaluate(folder, step)
            rows.append(dict(candidate=index, shard=shard, method='weighted_low_rank', step=step,
                **params, val=scores.val_acc.mean(), val_std=scores.val_acc.std(ddof=0),
                outer_ce=snapshot['teacher_ce'], **{k: snapshot[k] for k in
                ('weight_kl', 'node_ess_fraction', 'weight_min', 'weight_max')}))
            write_table(pd.DataFrame(rows), part / 'search.csv')
    save_json(dict(complete=True), part / 'complete.json')
    parts = [root / f'shard_{s}' for s in range(shards)]
    if shard != shards - 1 or not all((p / 'complete.json').exists() for p in parts):
        return pd.DataFrame(), pd.DataFrame(rows), baseline, root
    search = pd.concat([pd.read_csv(p / 'search.csv') for p in parts], ignore_index=True)
    write_table(search, root / 'search.csv')
    choice = validation_choice(search)
    save_json(choice, root / 'selected.json')
    baseline_choice = validation_choice(baseline)
    selected_folder = root / f'shard_{int(choice["shard"])}' / f'candidate_{int(choice["candidate"]):04d}'
    baseline_step = int(baseline_choice['step'])
    baseline_folder = original if baseline_step <= prior['steps'] else original / 'extended'
    summaries = []
    targets = [('initial', selected_folder, 0), ('low_rank', baseline_folder, baseline_step),
               ('weighted_low_rank', selected_folder, int(choice['step']))]
    for method, folder, step in targets:
        scores, snapshot = evaluate(folder, step, final=True)
        summaries.append(dict(method=method, step=step, gamma=gamma, T=fixed['T'], rank=fixed['rank'],
            dataset=prior['dataset'], ratio=prior['ratio'], nodes=prior['cells'], loss_weighting='mass',
            penalty=fixed['penalty'], assignment_lr=fixed['assignment_lr'],
            node_weight_penalty=choice['node_weight_penalty'] if method == 'weighted_low_rank' else 0.,
            node_weight_lr=choice['node_weight_lr'] if method == 'weighted_low_rank' else 0.,
            final_val=scores.val_acc.mean(), final_val_std=scores.val_acc.std(ddof=0),
            search_val=choice['val'] if method == 'weighted_low_rank' else float(
                baseline.loc[baseline.step == step, 'val'].iloc[0]),
            test_mean=scores.test_acc.mean(), test_std=scores.test_acc.std(ddof=0),
            node_ess_fraction=snapshot.get('node_ess_fraction', 1.),
            weight_max=snapshot.get('weight_max', 1.)))
    result = pd.DataFrame(summaries)
    write_table(result, root / 'summary.csv')
    write_table(baseline, root / 'baseline_curve.csv')
    return result, search, baseline, root
