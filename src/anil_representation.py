import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric import seed_everything
from tqdm.auto import trange

from src.models import GCN
from src.node_distances import array_digest
from src.risk_experiment import _fingerprint, _prepare_dataset


def representation(model, x, adjacency):
    layer = model.layers[0]
    z = torch.sparse.mm(adjacency, layer.lin(x))
    if layer.bias is not None:
        z = z + layer.bias
    z = F.dropout(z.relu(), model.dropout, training=model.training)
    return torch.sparse.mm(adjacency, z)


def adapt_head(z, y, weight, bias, steps, lr, penalty=0., differentiable=True):
    for _ in range(steps):
        loss = F.cross_entropy(F.linear(z, weight, bias), y)
        loss = loss + penalty * weight.square().sum() / 2
        dw, db = torch.autograd.grad(loss, (weight, bias), create_graph=differentiable)
        weight, bias = weight - lr * dw, bias - lr * db
        if not differentiable:
            weight = weight.detach().requires_grad_()
            bias = bias.detach().requires_grad_()
    return weight, bias


def sample_support(ids, labels, size, seed, leave_query=False):
    generator = torch.Generator(device=ids.device).manual_seed(seed)
    groups = [ids[labels[ids] == c] for c in labels[ids].unique(sorted=True)]
    capacity = sum(len(g) - int(leave_query) for g in groups)
    if not len(groups) <= size <= capacity:
        raise ValueError('Support size must cover each class and leave query nodes when requested')
    groups = [g[torch.randperm(len(g), generator=generator, device=g.device)] for g in groups]
    counts = [0] * len(groups)
    order = torch.randperm(len(groups), generator=generator, device=ids.device).tolist()
    while sum(counts) < size:
        for c in order:
            if counts[c] < len(groups[c]) - int(leave_query) and sum(counts) < size:
                counts[c] += 1
    return torch.cat([g[:n] for g, n in zip(groups, counts)])


def fresh_probe(z, y, support, classes, seed, steps, lr, penalty):
    seed_everything(seed)
    head = torch.nn.Linear(z.shape[1], classes, device=z.device)
    return adapt_head(z[support].detach(), y[support], head.weight, head.bias,
                      steps, lr, penalty, differentiable=False)


@torch.no_grad()
def accuracy(z, y, ids, head):
    logits = F.linear(z[ids], *head)
    return float((logits.argmax(1) == y[ids]).float().mean()), float(F.cross_entropy(logits, y[ids]))


def run_anil_representation(output_dir, data_dir='/content/data/', device='cuda',
                            episodes=1000, inner_steps=5, inner_lr=.1,
                            outer_lr=.001, support_size=35, hidden=256, dropout=0.,
                            weight_decay=.0005, head_penalty=.0005, eval_every=50,
                            probe_steps=200, probe_lr=.1, selection_size=70,
                            probe_sizes=(35, 70, 140), seeds=tuple(range(100, 110)), seed=0):
    if min(episodes, inner_steps, eval_every, probe_steps) < 1 or not seeds:
        raise ValueError('Require positive iteration counts and evaluation seeds')
    options = {k: v for k, v in locals().copy().items() if k != 'output_dir'}
    train, mask, validation, testing, _ = _prepare_dataset('cora', data_dir, device)
    x, y, adjacency = train['x'], train['y'], train['adj'].to_sparse_coo().coalesce()
    ids = mask.nonzero().flatten()
    val_ids, test_ids = validation[1].nonzero().flatten(), testing[1].nonzero().flatten()
    classes = int(y.max()) + 1
    options['revision'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    options['data'] = array_digest(x.cpu().numpy(), y.cpu().numpy(), adjacency.indices().cpu().numpy(),
                                   adjacency.values().cpu().numpy(), ids.cpu().numpy(),
                                   val_ids.cpu().numpy(), test_ids.cpu().numpy())
    root = Path(output_dir) / _fingerprint(options)
    root.mkdir(parents=True, exist_ok=True)
    (root / 'config.json').write_text(json.dumps(options, indent=2), encoding='utf-8')
    selection = sample_support(ids, y, selection_size, 777)
    rows, all_runs = [], []
    for method in ('supervised', 'anil'):
        folder = root / method
        folder.mkdir(exist_ok=True)
        seed_everything(seed)
        model = GCN(x.shape[1], hidden, classes, 2, dropout).to(device)
        checkpoint = folder / 'encoder.pt'
        if not checkpoint.exists():
            optimizer = torch.optim.Adam(model.parameters(), lr=outer_lr, weight_decay=weight_decay)
            history, best, saved = [], -float('inf'), None
            if x.is_cuda:
                torch.cuda.synchronize()
            started = time.perf_counter()
            for episode in trange(1, episodes + 1, desc=method):
                seed_everything(seed + episode)
                model.train()
                z = representation(model, x, adjacency)
                layer = model.layers[-1]
                if method == 'anil':
                    support = sample_support(ids, y, support_size, seed + episode, leave_query=True)
                    query = ids[~torch.isin(ids, support)]
                    weight, bias = adapt_head(z[support], y[support], layer.lin.weight, layer.bias,
                                              inner_steps, inner_lr, head_penalty)
                    loss = F.cross_entropy(F.linear(z[query], weight, bias), y[query])
                else:
                    loss = F.cross_entropy(F.linear(z[ids], layer.lin.weight, layer.bias), y[ids])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                if episode % eval_every == 0 or episode == episodes:
                    model.eval()
                    with torch.no_grad():
                        features = representation(model, x, adjacency)
                    head = fresh_probe(features, y, selection, classes, 777, probe_steps, probe_lr, head_penalty)
                    val, ce = accuracy(features, y, val_ids, head)
                    if x.is_cuda:
                        torch.cuda.synchronize()
                    history.append(dict(episode=episode, loss=float(loss.detach()), val=100 * val,
                                        val_ce=ce, elapsed_seconds=time.perf_counter() - started))
                    if val > best:
                        best = val
                        saved = dict(state_dict={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                                     episode=episode, selection_val=100 * val)
                        torch.save(saved, checkpoint)
                    pd.DataFrame(history).to_csv(folder / 'training.csv', index=False)
            saved['complete'] = True
            torch.save(saved, checkpoint)
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if not saved.get('complete'):
            raise RuntimeError(f'Interrupted training: remove {checkpoint} and rerun to restart this method')
        model.load_state_dict(saved['state_dict'])
        model.eval()
        with torch.no_grad():
            features = representation(model, x, adjacency).detach()
        torch.save(dict(features=features.cpu(), state_dict=saved['state_dict'], config=options,
                        epoch=saved['episode']), folder / 'features.pt')
        final_path = folder / 'probes.csv'
        if final_path.exists():
            final = pd.read_csv(final_path)
        else:
            records = []
            for size in probe_sizes:
                for probe_seed in seeds:
                    support = sample_support(ids, y, size, probe_seed)
                    head = fresh_probe(features, y, support, classes, probe_seed,
                                       probe_steps, probe_lr, head_penalty)
                    val, val_ce = accuracy(features, y, val_ids, head)
                    test, test_ce = accuracy(features, y, test_ids, head)
                    records.append(dict(method=method, support_size=size, seed=probe_seed,
                                        val=100 * val, test=100 * test, val_ce=val_ce, test_ce=test_ce))
            final = pd.DataFrame(records)
            final.to_csv(final_path, index=False)
        all_runs.append(final)
        for size, group in final.groupby('support_size'):
            rows.append(dict(method=method, support_size=size, selected_episode=saved['episode'],
                             val=group.val.mean(), val_std=group.val.std(ddof=0),
                             test=group.test.mean(), test_std=group.test.std(ddof=0),
                             val_ce=group.val_ce.mean(), test_ce=group.test_ce.mean(), output_dir=str(root)))
        pd.DataFrame(rows).to_csv(root / 'summary.csv', index=False)
    pd.concat(all_runs).to_csv(root / 'probes.csv', index=False)
    return pd.DataFrame(rows)
