import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric import seed_everything
from tqdm.auto import tqdm

from src.dataloader import get_dataset
from src.fsw import quotient_edges
from src.grid_search import GridStudy
from src.models import GCN
from src.node_distances import array_digest
from src.partition import cell_means, kmeans_init
from src.risk_experiment import _fingerprint, _forward, _prepare_dataset, _train_student
from src.teacher_metric_grip import prepare_metric_teacher
from src.tree_distance import _write_json
from src.utils import BUDGET


def readout_embedding(model, x, adjacency):
    first = model.layers[0]
    h = adjacency @ first.lin(x)
    if first.bias is not None:
        h = h + first.bias
    return adjacency @ F.relu(h)


def normalized_graph(edge_logits):
    weights = ((edge_logits + edge_logits.T) / 2).sigmoid()
    eye = torch.eye(len(weights), device=weights.device, dtype=weights.dtype)
    weights = weights * (1 - eye) + eye
    scale = weights.sum(1).rsqrt()
    return scale[:, None] * weights * scale[None, :]


def moments(z, q, mass):
    weighted = z * mass[:, None]
    return weighted.T @ q, weighted.T @ z


def moment_loss(z, q, mass, reference, second_weight):
    normalized = (z - reference['center']) / reference['scale']
    first, second = moments(normalized, q, mass)
    first_loss = (first - reference['first']).square().sum() / reference['first'].square().sum().clamp_min(.01)
    second_loss = (second - reference['second']).square().sum() / reference['second'].square().sum().clamp_min(.01)
    return first_loss + second_weight * second_loss, first_loss, second_loss


def _save(path, value):
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


@torch.no_grad()
def _references(model, graph, logits, temperatures):
    z = readout_embedding(model, graph['x'], graph['adj'])
    center = z.mean(0)
    scale = (z - center).square().sum(1).mean().sqrt().clamp_min(1e-6)
    normalized = (z - center) / scale
    second = normalized.T @ normalized / len(z)
    prediction = F.log_softmax(model.layers[-1].lin(z) + model.layers[-1].bias, dim=1)
    result = {}
    for temperature in temperatures:
        q = (logits / temperature).softmax(1)
        result[str(temperature)] = dict(center=center.cpu(), scale=scale.cpu(),
                                       first=(normalized.T @ q / len(z)).cpu(), second=second.cpu(),
                                       ce=float(-(q * prediction).sum(1).mean()))
    return result


def _probe_bank(graph, logits, nodes, temperatures, folder, config):
    path = folder / 'probes.pt'
    if path.exists():
        return torch.load(path, map_location='cpu', weights_only=True)
    records = []
    q = logits.softmax(1)
    for seed in tqdm(config['seeds'], desc='GCN moment probes'):
        seed_everything(seed)
        model = GCN(graph['x'].shape[1], config['hidden'], logits.shape[1], 2,
                    config['dropout']).to(logits.device)
        pool = torch.randperm(len(logits), device=logits.device)[:nodes]
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        for epoch in range(max(config['epochs']) + 1):
            if epoch in config['epochs']:
                model.eval()
                records.append(dict(seed=seed, epoch=epoch,
                                    state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                                    references=_references(model, graph, logits, temperatures)))
            if epoch < max(config['epochs']):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                loss = -(q[pool] * _forward(model, graph['x'], graph['adj'])[pool]).sum(1).mean()
                loss.backward()
                optimizer.step()
    _save(path, records)
    return records


def _load_probes(bank, temperature, inputs, hidden, classes, device):
    probes = []
    for entry in bank:
        model = GCN(inputs, hidden, classes, 2, 0.).to(device).eval()
        model.load_state_dict(entry['state'])
        model.requires_grad_(False)
        reference = {key: value.to(device) if torch.is_tensor(value) else value
                     for key, value in entry['references'][str(temperature)].items()}
        probes.append((model, reference))
    return probes


def fit_graph_moments(initial_x, edge_logits, labels, mass, probes, steps=1000,
                      feature_lr=.01, edge_lr=.01, second_weight=1., eval_every=25):
    x = torch.nn.Parameter(initial_x.clone())
    edges = torch.nn.Parameter(edge_logits.clone())
    optimizer = torch.optim.Adam([dict(params=[x], lr=feature_lr), dict(params=[edges], lr=edge_lr)])
    history, best = [], float('inf')
    best_x, best_edges = x.detach().clone(), edges.detach().clone()
    best_step = 0
    for step in range(steps + 1):
        if step % eval_every == 0 or step == steps:
            with torch.no_grad():
                adjacency = normalized_graph(edges)
                losses, gaps = [], []
                for model, reference in probes:
                    z = readout_embedding(model, x, adjacency)
                    values = moment_loss(z, labels, mass, reference, second_weight)
                    losses.append(torch.stack(values))
                    prediction = F.log_softmax(model.layers[-1].lin(z) + model.layers[-1].bias, dim=1)
                    ce = -(mass * (labels * prediction).sum(1)).sum()
                    gaps.append((ce - reference['ce']).abs())
                loss, first, second = torch.stack(losses).mean(0).tolist()
                if not np.isfinite(loss):
                    raise FloatingPointError('Nonfinite graph moment objective')
                history.append(dict(step=step, objective=loss, first=first, second=second,
                                    probe_ce_gap=float(torch.stack(gaps).mean())))
                if loss < best:
                    best, best_step = loss, step
                    best_x, best_edges = x.detach().clone(), edges.detach().clone()
        if step < steps:
            model, reference = probes[step % len(probes)]
            optimizer.zero_grad(set_to_none=True)
            z = readout_embedding(model, x, normalized_graph(edges))
            loss = moment_loss(z, labels, mass, reference, second_weight)[0]
            loss.backward()
            optimizer.step()
    selected = next(row for row in history if row['step'] == best_step)
    state = dict(x=best_x.cpu(), adjacency=normalized_graph(best_edges).detach().cpu(),
                 edge_logits=best_edges.cpu(), objective_initial=history[0]['objective'],
                 objective_final=best, best_step=best_step, first_final=selected['first'],
                 second_final=selected['second'], probe_ce_gap=selected['probe_ce_gap'])
    return state, pd.DataFrame(history)


def run_graph_moment_sweep(dataset, ratio, output_dir, space, teacher_run=None,
                           data_dir='/content/data/', device='cuda', variants=('initial', 'moments'),
                           search_seeds=(0, 1, 2), final_seeds=tuple(range(100, 110)),
                           probe_seeds=(5000, 5001), probe_epochs=(0, 50, 200),
                           hidden=256, init_seed=1234, neighbors=8, steps=1000,
                           moment_eval_every=25, epochs=1000, eval_every=10,
                           loss_weighting='mass'):
    keys = {'T', 'second_weight', 'feature_lr', 'edge_lr', 'dropout', 'lr', 'weight_decay'}
    if set(space) != keys or any(not values for values in space.values()):
        raise ValueError('Provide T, second_weight, feature_lr, edge_lr, dropout, lr, weight_decay grids')
    if dataset not in ('cora', 'citeseer', 'arxiv') or (dataset, ratio) not in BUDGET:
        raise ValueError('Require a supported transductive dataset and ratio')
    groups = [set(search_seeds), set(final_seeds), set(probe_seeds)]
    if any(not g for g in groups) or any(groups[i] & groups[j] for i in range(3) for j in range(i)):
        raise ValueError('Use nonempty disjoint probe, search and final seeds')
    if not variants or not set(variants) <= {'initial', 'moments'} or loss_weighting not in ('mass', 'uniform'):
        raise ValueError('Require initial/moments variants and mass/uniform CE')
    if min(steps, moment_eval_every, epochs, eval_every, neighbors) < 1 or not probe_epochs or min(probe_epochs) < 0:
        raise ValueError('Require positive steps and nonnegative probe epochs')
    if min(v for key in ('T', 'feature_lr', 'edge_lr', 'lr') for v in space[key]) <= 0 or min(space['second_weight']) < 0:
        raise ValueError('Require positive temperatures/rates and nonnegative second moment weight')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if teacher_run is None:
        teacher_run = prepare_metric_teacher(dataset, output / 'teacher', data_dir, device)['folder']
    teacher_run = Path(teacher_run)
    teacher_protocol = json.loads((teacher_run / 'protocol.json').read_text())
    graph = get_dataset(SimpleNamespace(dataset_name=dataset, raw_data_dir=str(data_dir).rstrip('/') + '/'))
    graph_hash = array_digest(graph.x.numpy(), graph.edge_index.numpy())
    supervision = array_digest(graph.train_mask.numpy(), graph.val_mask.numpy(),
                               graph.y[graph.train_mask].numpy(), graph.y[graph.val_mask].numpy())
    if (teacher_protocol.get('dataset') != dataset or teacher_protocol.get('graph_sha256') != graph_hash
            or teacher_protocol.get('supervision_sha256') != supervision):
        raise ValueError('Use a matching teacher from prepare_metric_teacher, or teacher_run=None')
    with np.load(teacher_run / 'teacher_predictions.npz') as saved:
        teacher_logits = torch.as_tensor(saved['logits'].copy(), device=device)
    train, _, validation, testing, propagated = _prepare_dataset(dataset, data_dir, device)
    nodes = BUDGET[dataset, ratio]
    temperatures = sorted(set(float(t) for t in space['T']))
    probe_config = dict(seeds=list(probe_seeds), epochs=sorted(set(probe_epochs)), hidden=hidden,
                        dropout=.5, lr=.01, weight_decay=5e-4, pool_size=nodes, target_temperature=1.)
    shared = dict(version=1, dataset=dataset, ratio=ratio, nodes=nodes, graph=graph_hash,
                  teacher_logits=array_digest(teacher_logits.cpu().numpy()), teacher=teacher_protocol,
                  probes=probe_config, temperatures=temperatures, init_seed=init_seed, neighbors=neighbors,
                  torch=str(torch.__version__), device=str(device))
    cache = output / 'cache' / _fingerprint(shared)
    cache.mkdir(parents=True, exist_ok=True)
    initialization = cache / 'initial.pt'
    if initialization.exists():
        initial = torch.load(initialization, map_location='cpu', weights_only=True)
    else:
        assignment = kmeans_init(propagated, nodes, init_seed)
        counts = torch.bincount(assignment, minlength=nodes)
        for empty in (counts == 0).nonzero().flatten().tolist():
            donor = int(counts.argmax())
            member = (assignment == donor).nonzero().flatten()[0]
            assignment[member] = empty
            counts[donor] -= 1
            counts[empty] += 1
        x = cell_means(train['x'], assignment, nodes)
        edges = quotient_edges(graph.edge_index.to(device), assignment, nodes, neighbors)
        probabilities = x.new_full((nodes, nodes), 1e-4)
        probabilities[edges[0], edges[1]] = .8
        initial = dict(x=x.cpu(), edge_logits=torch.logit(probabilities).cpu(),
                       assignment=assignment.cpu(), counts=counts.cpu())
        _save(initialization, initial)
    del propagated, graph
    bank = _probe_bank(train, teacher_logits, nodes, temperatures, cache, probe_config)
    settings = dict(hidden=hidden, epochs=epochs, eval_every=eval_every, loss_weighting=loss_weighting)
    config = dict(**shared, grid=list(space.items()), variants=list(variants), steps=steps,
                  moment_eval_every=moment_eval_every, student=settings,
                  search_seeds=list(search_seeds), final_seeds=list(final_seeds),
                  labels='fixed_cluster_mean', features='learned_raw_X', synthetic_graph='learned_symmetric_dense_weights',
                  objective='normalized_squared_first_and_second_readout_moments')
    folder = output / _fingerprint(config)
    folder.mkdir(exist_ok=True)
    _write_json(folder / 'protocol.json', config)
    counts = initial['counts'].to(device)
    mass = counts.float() / counts.sum()
    assignment = initial['assignment'].to(device)
    active = {}

    def condensed(variant, params):
        temperature = float(params['T'])
        identity = dict(variant=variant, T=temperature)
        if variant == 'moments':
            identity.update({key: params[key] for key in ('second_weight', 'feature_lr', 'edge_lr')})
        path = folder / f'condensed_{_fingerprint(identity)}.pt'
        if path.exists():
            return torch.load(path, map_location='cpu', weights_only=True), path
        started = time.perf_counter()
        if active.get('T') != temperature:
            active.clear()
            active['T'] = temperature
            active['probes'] = _load_probes(bank, temperature, train['x'].shape[1], hidden,
                                           teacher_logits.shape[1], device)
        labels = cell_means((teacher_logits / temperature).softmax(1), assignment, nodes)
        state, history = fit_graph_moments(
            initial['x'].to(device), initial['edge_logits'].to(device), labels, mass, active['probes'],
            steps=steps if variant == 'moments' else 0, feature_lr=params.get('feature_lr', .01),
            edge_lr=params.get('edge_lr', .01), second_weight=params.get('second_weight', 1.),
            eval_every=moment_eval_every)
        state.update(y=labels.cpu(), counts=counts.cpu(), nodes=nodes, seconds=time.perf_counter() - started)
        history.to_csv(path.with_suffix('.csv'), index=False)
        _save(path, state)
        return state, path

    selected = []
    for variant in variants:
        case = folder / variant
        case.mkdir(exist_ok=True)
        grid = {key: values for key, values in space.items()
                if variant == 'moments' or key in ('T', 'dropout', 'lr', 'weight_decay')}
        study = GridStudy(grid, case)

        def objective(trial):
            state, _ = condensed(variant, trial.params)
            values = [_train_student(state['x'].to(device), state['y'].to(device), validation,
                                     trial.params, seed, settings, counts=state['counts'],
                                     adjacency=state['adjacency'].to(device))[0] for seed in search_seeds]
            trial.set_user_attr('validation_per_seed', values)
            return np.mean(values)

        study.optimize(objective)
        study.trials_dataframe().to_csv(case / 'trials.csv', index=False)
        best = study.best_trial
        _write_json(case / 'best.json', dict(params=best.params, search_val=best.value))
        selected.append((variant, case, best))
    rows, repeats = [], []
    for variant, case, best in selected:
        state, path = condensed(variant, best.params)
        records = []
        for seed in tqdm(final_seeds, desc=f'{variant}: final students'):
            record_path = case / f'final_{seed}.json'
            if record_path.exists():
                record = json.loads(record_path.read_text())
            else:
                val, test, epoch = _train_student(state['x'].to(device), state['y'].to(device), validation,
                                                  best.params, seed, settings, testing=testing,
                                                  counts=state['counts'], adjacency=state['adjacency'].to(device))
                record = dict(variant=variant, seed=seed, validation=100 * val, test=100 * test, epoch=epoch)
                _write_json(record_path, record)
            records.append(record)
        frame = pd.DataFrame(records)
        repeats.extend(records)
        rows.append(dict(dataset=dataset, ratio=ratio, variant=variant, nodes=nodes,
                         loss_weighting=loss_weighting, **best.params, search_val=100 * best.value,
                         final_val=frame.validation.mean(), final_val_std=frame.validation.std(ddof=0),
                         test_mean=frame.test.mean(), test_std=frame.test.std(ddof=0),
                         **{key: state[key] for key in ('objective_initial', 'objective_final', 'best_step',
                                                       'first_final', 'second_final', 'probe_ce_gap', 'seconds')},
                         artifact=str(path)))
    summary, detail = pd.DataFrame(rows), pd.DataFrame(repeats)
    summary.to_csv(folder / 'summary.csv', index=False)
    detail.to_csv(folder / 'final_seeds.csv', index=False)
    return dict(summary=summary, repeats=detail, folder=str(folder))


def plot_graph_moments(report):
    import matplotlib.pyplot as plt

    summary = report['summary']
    figure, axes = plt.subplots(1, 3, figsize=(16, 4))
    for row in summary.itertuples():
        curve = pd.read_csv(Path(row.artifact).with_suffix('.csv'))
        if row.variant == 'moments':
            axes[0].plot(curve.step, curve['first'], label='Feature-label moment')
            axes[0].plot(curve.step, curve['second'], label='Feature second moment')
            axes[1].plot(curve.step, curve.probe_ce_gap, label='Matching probes')
            axes[1].scatter([row.best_step], [row.probe_ce_gap], color='black', label='Saved step')
    axes[0].set(title='Moment errors', xlabel='Adam step', ylabel='Normalized squared error')
    axes[0].set_yscale('symlog', linthresh=1e-6)
    axes[1].set(title='Mean absolute CE risk gap', xlabel='Adam step', ylabel='Original vs condensed CE')
    for axis in axes[:2]:
        if axis.lines:
            axis.legend()
    positions = np.arange(len(summary))
    axes[2].bar(positions - .18, summary.final_val, .36, yerr=summary.final_val_std, label='Validation')
    axes[2].bar(positions + .18, summary.test_mean, .36, yerr=summary.test_std, label='Test')
    axes[2].set(xticks=positions, xticklabels=summary.variant, ylabel='Accuracy (%)', title='Independent final student seeds')
    axes[2].legend()
    figure.tight_layout()
    figure.savefig(Path(report['folder']) / 'graph_moments.png', dpi=180)
    plt.show()
    return figure
