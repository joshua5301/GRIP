import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from src.dataloader import get_dataset
from src.graph_moment_condensation import _probe_bank, moment_loss, normalized_graph, readout_embedding
from src.models import GCN
from src.node_distances import array_digest
from src.risk_experiment import _fingerprint, _prepare_dataset, _train_student
from src.tree_distance import _write_json


SHARED_KEYS = ('version', 'dataset', 'ratio', 'nodes', 'graph', 'teacher_logits', 'teacher',
               'probes', 'temperatures', 'init_seed', 'neighbors', 'torch', 'device')


def _load_pair(source, config, params):
    cache = source.parent / 'cache' / _fingerprint({key: config[key] for key in SHARED_KEYS})
    identity = dict(variant='moments', T=float(params['T']),
                    **{key: params[key] for key in ('second_weight', 'feature_lr', 'edge_lr')})
    optimized = torch.load(source / f'condensed_{_fingerprint(identity)}.pt',
                           map_location='cpu', weights_only=True)
    original = torch.load(cache / 'initial.pt', map_location='cpu', weights_only=True)
    if not torch.equal(original['counts'], optimized['counts']):
        raise ValueError('Initial and optimized cell masses differ')
    initial = dict(x=original['x'], adjacency=normalized_graph(original['edge_logits']),
                   y=optimized['y'].clone(), counts=optimized['counts'].clone())
    saved_initial = source / f'condensed_{_fingerprint(dict(variant="initial", T=float(params["T"])))}.pt'
    if saved_initial.exists():
        saved = torch.load(saved_initial, map_location='cpu', weights_only=True)
        initial.update(x=saved['x'], adjacency=saved['adjacency'])
    return dict(initial=initial, moments=optimized), cache


def _teacher_logits(source, config, teacher_run, device):
    candidates = [Path(teacher_run)] if teacher_run is not None else sorted((source.parent / 'teacher').glob('*'))
    for candidate in candidates:
        path = candidate / 'protocol.json'
        if path.exists() and json.loads(path.read_text()) == config['teacher']:
            with np.load(candidate / 'teacher_predictions.npz') as saved:
                logits = saved['logits'].copy()
            if array_digest(logits) != config['teacher_logits']:
                raise ValueError('Teacher logits differ from the condensation run')
            return torch.as_tensor(logits, device=device)
    raise FileNotFoundError('Matching teacher not found; pass its prepare_metric_teacher folder as teacher_run')


@torch.no_grad()
def measure_probe_pair(model, reference, states, second_weight):
    rows = []
    for variant, state in states.items():
        mass = state['counts'].to(state['x'].dtype)
        mass = mass / mass.sum()
        z = readout_embedding(model, state['x'], state['adjacency'])
        objective, first, second = moment_loss(z, state['y'], mass, reference, second_weight)
        prediction = F.log_softmax(model.layers[-1].lin(z) + model.layers[-1].bias, dim=1)
        synthetic_ce = float(-(mass * (state['y'] * prediction).sum(1)).sum())
        signed = synthetic_ce - reference['ce']
        rows.append(dict(variant=variant, original_ce=reference['ce'], synthetic_ce=synthetic_ce,
                         signed_gap=signed, abs_gap=abs(signed), objective=float(objective),
                         first=float(first), second=float(second)))
    return rows


def summarize_risks(frame):
    rows = []
    for (cohort, epoch, variant), group in frame.groupby(['cohort', 'epoch', 'variant'], sort=False):
        rows.append(dict(cohort=cohort, epoch=epoch, variant=variant, seeds=group.seed.nunique(),
                         abs_gap_mean=group.abs_gap.mean(), abs_gap_std=group.abs_gap.std(ddof=0),
                         abs_gap_max=group.abs_gap.max(), signed_gap_mean=group.signed_gap.mean(),
                         first_mean=group['first'].mean(), second_mean=group['second'].mean()))
    pairs = frame.pivot(index=['cohort', 'seed', 'epoch'], columns='variant', values='abs_gap').reset_index()
    pairs['gap_change'] = pairs['moments'] - pairs['initial']
    return pd.DataFrame(rows), pairs


def run_graph_moment_diagnostics(source_dir, output_dir=None, teacher_run=None,
                                  student_seeds=tuple(range(100, 110)),
                                  heldout_seeds=tuple(range(6000, 6010)),
                                  data_dir='/content/data/', device='cuda'):
    source = Path(source_dir)
    config = json.loads((source / 'protocol.json').read_text())
    params = json.loads((source / 'moments' / 'best.json').read_text())['params']
    forbidden = set(config['probes']['seeds']) | set(config['search_seeds'])
    if not student_seeds or len(set(student_seeds)) != len(student_seeds) or set(student_seeds) & forbidden:
        raise ValueError('Use distinct student seeds outside search and matching-probe seeds')
    forbidden |= set(student_seeds) | set(config['final_seeds'])
    if not heldout_seeds or len(set(heldout_seeds)) != len(heldout_seeds) or set(heldout_seeds) & forbidden:
        raise ValueError('Held-out probes must have distinct seeds outside all existing seed groups')
    states, cache = _load_pair(source, config, params)
    teacher_logits = _teacher_logits(source, config, teacher_run, device)
    graph = get_dataset(SimpleNamespace(dataset_name=config['dataset'], raw_data_dir=str(data_dir).rstrip('/') + '/'))
    if array_digest(graph.x.numpy(), graph.edge_index.numpy()) != config['graph']:
        raise ValueError('Original graph/features differ from the source experiment')
    supervision = array_digest(graph.train_mask.numpy(), graph.val_mask.numpy(),
                               graph.y[graph.train_mask].numpy(), graph.y[graph.val_mask].numpy())
    if supervision != config['teacher']['supervision_sha256']:
        raise ValueError('Training/validation supervision differs from the source experiment')
    evaluation_digest = array_digest(graph.y.numpy(), graph.val_mask.numpy(), graph.test_mask.numpy())
    del graph
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    train, _, validation, testing, propagated = _prepare_dataset(config['dataset'], data_dir, device)
    del propagated
    settings = config['student']
    protocol = dict(version=1, source=config, params=params, student_seeds=list(student_seeds),
                    heldout_seeds=list(heldout_seeds), evaluation=evaluation_digest,
                    artifacts={name: array_digest(*(state[key].numpy() for key in ('x', 'adjacency', 'y', 'counts')))
                               for name, state in states.items()}, torch=str(torch.__version__), device=str(device))
    root = Path(output_dir) if output_dir is not None else source / 'diagnostics'
    folder = root / _fingerprint(protocol)
    folder.mkdir(parents=True, exist_ok=True)
    _write_json(folder / 'protocol.json', protocol)
    states = {name: {key: state[key].to(device) for key in ('x', 'adjacency', 'y', 'counts')}
              for name, state in states.items()}
    records = []
    initial_best_path = source / 'initial' / 'best.json'
    initial_params = json.loads(initial_best_path.read_text())['params'] if initial_best_path.exists() else {}
    for variant, state in states.items():
        for seed in tqdm(student_seeds, desc=f'{variant}: matched-T students'):
            path = folder / f'student_{variant}_{seed}.json'
            existing = source / variant / f'final_{seed}.json'
            reusable = (config['torch'] == str(torch.__version__) and config['device'] == str(device)
                        and (variant == 'moments' or all(initial_params.get(k) == params[k]
                                                         for k in ('T', 'dropout', 'lr', 'weight_decay'))))
            if path.exists():
                record = json.loads(path.read_text())
            elif reusable and existing.exists():
                record = dict(json.loads(existing.read_text()), reused=True)
                _write_json(path, record)
            else:
                val, test, epoch = _train_student(state['x'], state['y'], validation, params, seed, settings,
                                                  testing=testing, counts=state['counts'], adjacency=state['adjacency'])
                record = dict(variant=variant, seed=seed, validation=100 * val, test=100 * test,
                              epoch=epoch, reused=False)
                _write_json(path, record)
            records.append(record)
    students = pd.DataFrame(records)
    student_summary = pd.DataFrame([
        dict(variant=variant, seeds=len(group), T=params['T'], dropout=params['dropout'],
             loss_weighting=settings['loss_weighting'], validation=group.validation.mean(),
             validation_std=group.validation.std(ddof=0), test=group.test.mean(), test_std=group.test.std(ddof=0))
        for variant, group in students.groupby('variant', sort=False)])
    student_pairs = students.pivot(index='seed', columns='variant', values=['validation', 'test'])
    changes = pd.DataFrame({f'{metric}_change': student_pairs[metric]['moments'] - student_pairs[metric]['initial']
                            for metric in ('validation', 'test')}).reset_index()
    students.to_csv(folder / 'students.csv', index=False)
    student_summary.to_csv(folder / 'student_summary.csv', index=False)
    changes.to_csv(folder / 'student_changes.csv', index=False)
    probe_config = dict(config['probes'], seeds=list(heldout_seeds))
    heldout_folder = folder / 'heldout'
    heldout_folder.mkdir(exist_ok=True)
    banks = dict(matching=torch.load(cache / 'probes.pt', map_location='cpu', weights_only=True),
                 heldout=_probe_bank(train, teacher_logits, config['nodes'], [float(params['T'])],
                                      heldout_folder, probe_config))
    risks = []
    for cohort, bank in banks.items():
        for entry in tqdm(bank, desc=f'{cohort}: CE preservation'):
            path = folder / f'risk_{cohort}_{entry["seed"]}_{entry["epoch"]}.json'
            if path.exists():
                measured = json.loads(path.read_text())['rows']
            else:
                model = GCN(train['x'].shape[1], config['probes']['hidden'], teacher_logits.shape[1], 2, 0.).to(device).eval()
                model.load_state_dict(entry['state'])
                reference = {key: value.to(device) if torch.is_tensor(value) else value
                             for key, value in entry['references'][str(float(params['T']))].items()}
                measured = [dict(cohort=cohort, seed=entry['seed'], epoch=entry['epoch'], **row)
                            for row in measure_probe_pair(model, reference, states, params['second_weight'])]
                _write_json(path, dict(rows=measured))
            risks.extend(measured)
    risks = pd.DataFrame(risks)
    risk_summary, risk_pairs = summarize_risks(risks)
    risks.to_csv(folder / 'probe_risks.csv', index=False)
    risk_summary.to_csv(folder / 'risk_summary.csv', index=False)
    risk_pairs.to_csv(folder / 'risk_changes.csv', index=False)
    return dict(folder=str(folder), params=params, student_summary=student_summary, students=students,
                student_changes=changes, risks=risks, risk_summary=risk_summary, risk_changes=risk_pairs)


def plot_graph_moment_diagnostics(report):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    for axis, metric in zip(axes[0], ('validation', 'test')):
        pairs = report['students'].pivot(index='seed', columns='variant', values=metric)
        for _, row in pairs.iterrows():
            axis.plot([0, 1], [row['initial'], row['moments']], color='gray', alpha=.45)
        axis.errorbar([0, 1], pairs[['initial', 'moments']].mean(),
                      yerr=pairs[['initial', 'moments']].std(ddof=0), color='black', marker='o', capsize=4)
        axis.set(xticks=[0, 1], xticklabels=['Initial', 'Moments'], ylabel='Accuracy (%)',
                 title=f'Matched T={report["params"]["T"]}: {metric}')
    for cohort, color in [('matching', 'tab:blue'), ('heldout', 'tab:orange')]:
        for variant, style in [('initial', '--'), ('moments', '-')]:
            group = report['risk_summary']
            group = group[(group.cohort == cohort) & (group.variant == variant)].sort_values('epoch')
            axes[1, 0].errorbar(group.epoch, group.abs_gap_mean, yerr=group.abs_gap_std,
                                color=color, linestyle=style, marker='o', capsize=3, label=f'{cohort}: {variant}')
    axes[1, 0].set(xlabel='Probe training epoch', ylabel='Mean absolute CE gap', title='Same checkpoints, both graphs')
    axes[1, 0].legend()
    heldout = report['risk_changes'].query("cohort == 'heldout'")
    for epoch, group in heldout.groupby('epoch'):
        axes[1, 1].scatter(group['initial'], group['moments'], label=f'epoch {epoch}', alpha=.7)
    limit = max(float(heldout[['initial', 'moments']].max().max()) * 1.05, 1e-6)
    axes[1, 1].plot([0, limit], [0, limit], '--', color='gray')
    axes[1, 1].set(xlabel='Initial absolute CE gap', ylabel='Optimized absolute CE gap',
                   title='Held-out probes: below diagonal is better', xlim=(0, limit), ylim=(0, limit))
    axes[1, 1].legend()
    figure.tight_layout()
    figure.savefig(Path(report['folder']) / 'diagnostics.png', dpi=180)
    plt.show()
    return figure
