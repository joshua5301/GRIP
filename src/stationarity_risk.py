import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import trange

from src.node_distances import array_digest
from src.ntk_transforms import fit_transform
from src.representation_learning_audit import cell_means
from src.risk_experiment import _fingerprint, _prepare_dataset, _train_student


def augment(z):
    return torch.cat((z, z.new_ones(len(z), 1)), dim=1)


def head_objective(x, q, mass, theta, penalty):
    return -(mass[:, None] * q * (x @ theta.T).log_softmax(1)).sum() + penalty * theta.square().sum() / 2


def fit_head(z, q, mass, penalty, max_iter=2000, grad_tol=1e-7):
    if penalty <= 0:
        raise ValueError('Positive regularization is required, including the bias')
    x, q, mass = augment(z.double()), q.double(), mass.double()
    theta = x.new_zeros(q.shape[1], x.shape[1], requires_grad=True)
    optimizer = torch.optim.LBFGS([theta], max_iter=max_iter, tolerance_grad=grad_tol,
                                 tolerance_change=1e-15, line_search_fn='strong_wolfe')

    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss = head_objective(x, q, mass, theta, penalty)
        loss.backward()
        return loss

    optimizer.step(closure)
    loss = head_objective(x, q, mass, theta, penalty)
    gradient, = torch.autograd.grad(loss, theta)
    norm = float(gradient.norm())
    if not np.isfinite(float(loss.detach()) + norm):
        raise FloatingPointError('Nonfinite head fit')
    return dict(theta=theta.detach(), grad_norm=norm, grad_max=float(gradient.abs().max()),
                converged=float(gradient.abs().max()) <= grad_tol)


def statistics(x, q, assignment, clusters):
    n = torch.bincount(assignment, minlength=clusters).to(x.dtype)
    sums = x.new_zeros(clusters, x.shape[1]).index_add_(0, assignment, x)
    labels = q.new_zeros(clusters, q.shape[1]).index_add_(0, assignment, q)
    return n, sums, labels


def contributions(n, sums, labels, theta, total):
    centers = sums / n[:, None]
    errors = labels.sum(1, keepdim=True) * (centers @ theta.T).softmax(1) - labels
    return errors[:, :, None] * centers[:, None, :] / total


def move_deltas(x, q, theta, penalty, n, sums, labels, nodes, sources, targets, parts=None, residual=None):
    total = n.sum()
    if parts is None:
        parts = contributions(n, sums, labels, theta, total)
    if residual is None:
        residual = parts.sum(0) + penalty * theta
    old = parts[sources] + parts[targets]
    removed = contributions(n[sources] - 1, sums[sources] - x[nodes],
                            labels[sources] - q[nodes], theta, total)
    added = contributions(n[targets] + 1, sums[targets] + x[nodes],
                          labels[targets] + q[nodes], theta, total)
    change = removed + added - old
    return 2 * (change * residual).sum((1, 2)) + change.square().sum((1, 2))


@torch.no_grad()
def refine_partition(z, q, initial_assignment, theta, penalty, max_sweeps=20,
                     block_size=16, pair_batch=256, seed=0, rtol=1e-10, atol=1e-18,
                     candidate_k=None, random_candidates=0):
    x, q, theta = augment(z.double()), q.double(), theta.double()
    assignment = initial_assignment.to(z.device).clone()
    clusters = int(assignment.max()) + 1
    n, sums, labels = statistics(x, q, assignment, clusters)
    if bool((n <= 0).any()) or penalty <= 0 or min(block_size, pair_batch) < 1 or max_sweeps < 0:
        raise ValueError('Require nonempty cells, positive penalty and valid solver settings')
    if (candidate_k is not None and candidate_k < 1) or random_candidates < 0:
        raise ValueError('Invalid candidate settings')

    def score(nn, ss, ll):
        residual = contributions(nn, ss, ll, theta, len(x)).sum(0) + penalty * theta
        return residual.square().sum()

    parts = contributions(n, sums, labels, theta, len(x))
    residual = parts.sum(0) + penalty * theta
    value = residual.square().sum()
    history = [dict(sweep=0, J=float(value), gradient_norm=float(value.sqrt()), moves=0, seconds=0.)]
    generator = torch.Generator(device=z.device).manual_seed(seed)
    started = time.perf_counter()

    def accept(ids, destinations):
        nonlocal residual, value
        sources = assignment[ids]
        affected, inverse = torch.unique(torch.cat((sources, destinations)), return_inverse=True)
        src, dst = inverse[:len(ids)], inverse[len(ids):]
        nn, ss, ll = n[affected].clone(), sums[affected].clone(), labels[affected].clone()
        ones = nn.new_ones(len(ids))
        nn.index_add_(0, src, -ones).index_add_(0, dst, ones)
        ss.index_add_(0, src, -x[ids]).index_add_(0, dst, x[ids])
        ll.index_add_(0, src, -q[ids]).index_add_(0, dst, q[ids])
        if bool((nn > 0).all()):
            new_parts = contributions(nn, ss, ll, theta, len(x))
            next_residual = residual + (new_parts - parts[affected]).sum(0)
            proposed = next_residual.square().sum()
            if proposed < value - (atol + rtol * value):
                assignment[ids] = destinations
                n[affected], sums[affected], labels[affected] = nn, ss, ll
                parts[affected] = new_parts
                residual, value = next_residual, proposed
                return len(ids)
        if len(ids) <= 1:
            return 0
        middle = len(ids) // 2
        return accept(ids[:middle], destinations[:middle]) + accept(ids[middle:], destinations[middle:])

    converged = False
    for sweep in trange(1, max_sweeps + 1, desc='Stationarity partition'):
        moved = 0
        for block in torch.randperm(len(z), generator=generator, device=z.device).split(block_size):
            ids = block[n[assignment[block]] > 1]
            if not len(ids):
                continue
            if candidate_k is None or candidate_k >= clusters:
                choices = torch.arange(clusters, device=z.device).expand(len(ids), -1)
            else:
                centers = sums[:, :-1] / n[:, None]
                distances = z[ids].double().square().sum(1)[:, None] + centers.square().sum(1)[None] - 2 * z[ids].double() @ centers.T
                distances.scatter_(1, assignment[ids, None], torch.inf)
                choices = distances.topk(min(candidate_k, clusters - 1), largest=False).indices
                if random_candidates:
                    random = torch.randint(clusters, (len(ids), random_candidates), generator=generator, device=z.device)
                    choices = torch.cat((choices, random), dim=1)
            width = choices.shape[1]
            nodes = ids.repeat_interleave(width)
            sources = assignment[nodes]
            targets = choices.flatten()
            delta = z.new_full((len(nodes),), torch.inf, dtype=torch.double)
            valid = (sources != targets).nonzero().flatten()
            for pair in valid.split(pair_batch):
                delta[pair] = move_deltas(x, q, theta, penalty, n, sums, labels,
                                          nodes[pair], sources[pair], targets[pair], parts, residual)
            best, indices = delta.reshape(len(ids), width).min(1)
            destinations = choices.gather(1, indices[:, None]).flatten()
            take = best < -(atol + rtol * value)
            if bool(take.any()):
                moved += accept(ids[take], destinations[take])
        n, sums, labels = statistics(x, q, assignment, clusters)
        parts = contributions(n, sums, labels, theta, len(x))
        residual = parts.sum(0) + penalty * theta
        value = score(n, sums, labels)
        history.append(dict(sweep=sweep, J=float(value), gradient_norm=float(value.sqrt()),
                            moves=moved, seconds=time.perf_counter() - started))
        if not moved:
            converged = True
            break
    exhaustive = candidate_k is None or candidate_k >= clusters
    return dict(assignment=assignment.cpu(), counts=n.long().cpu(), y=(labels / n[:, None]).float().cpu(),
                history=history, J=float(value), converged=converged and exhaustive,
                status=('no_accepted_move' if candidate_k is None or candidate_k >= clusters
                        else 'candidate_stalled') if converged else 'iteration_limit',
                candidate_k=candidate_k, random_candidates=random_candidates)


@torch.no_grad()
def metrics(z, q, y, val_ids, theta, penalty, test_ids=None):
    x = augment(z)
    logits = x @ theta.T
    result = dict(val=100 * float((logits[val_ids].argmax(1) == y[val_ids]).float().mean()),
                  val_ce=float(F.cross_entropy(logits[val_ids], y[val_ids])),
                  full_objective=float(head_objective(x, q, z.new_full((len(z),), 1 / len(z)), theta, penalty)))
    if test_ids is not None:
        result['test'] = 100 * float((logits[test_ids].argmax(1) == y[test_ids]).float().mean())
    return result


def run_stationarity_study(learning_root, output_dir=None, ratio=.026,
                           penalties=(.0001, .001, .01, .1), max_sweeps=20,
                           block_size=16, pair_batch=256, max_iter=2000,
                           final_seeds=tuple(range(100, 110)), seed=0,
                           data_dir='/content/data/', device='cuda', evaluate_test=True):
    options = locals().copy()
    learning_root = Path(learning_root)
    audit_config = json.loads((learning_root / 'config.json').read_text())
    source_root = Path(audit_config['directional_root'])
    direction_config = json.loads((source_root / 'config.json').read_text())
    table = pd.read_csv(direction_config['source'])
    rows = table[(table['dataset'] == 'cora') & (table['transform'] == 'rms') & np.isclose(table['ratio'], ratio)]
    if len(rows) != 1 or not penalties or min(penalties) <= 0 or not final_seeds:
        raise ValueError('Require one Cora RMS source row, positive penalties and student seeds')
    row = rows.iloc[0]
    path = Path(row.partition_path)
    original_config = json.loads((Path(row.output_dir) / 'config.json').read_text())
    params = json.loads((path.parent / row['mode'] / 'best.json').read_text())['params']
    baseline = torch.load(learning_root / 's2x' / str(float(ratio)) / 'baseline_partition.pt',
                          map_location='cpu', weights_only=False)
    train, mask, validation, testing, h = _prepare_dataset('cora', data_dir, device)
    signatures = []
    for graph, split in ((train, mask), validation, testing):
        adjacency = graph['adj'].to_sparse_csr()
        signatures.append(array_digest(graph['x'].cpu().numpy(), graph['y'].cpu().numpy(),
                          adjacency.crow_indices().cpu().numpy(), adjacency.col_indices().cpu().numpy(),
                          adjacency.values().cpu().numpy(), split.cpu().numpy()))
    if path.parent.parent.name != 'cora_' + _fingerprint(signatures)[:12]:
        raise ValueError('Source graph or splits do not match current Cora')
    label_root = Path(row.output_dir).parent / ('labels_' + _fingerprint(dict(
        dataset='cora', data=signatures, seed=original_config['seed'], revision=original_config['revision'])))
    key = (params['teacher_kernel'], params['basis'], params['gamma'])
    logits = torch.load(label_root / f'{_fingerprint(key)}.pt', map_location=device, weights_only=True)
    q = (logits / params['T']).softmax(1).double()
    z, transform = fit_transform(h.double(), kind='rms')
    initial = baseline['assignment'].to(device)
    _, check_labels, mass = cell_means(z, q, initial)
    if not torch.allclose(check_labels.float().cpu(), baseline['y'], atol=1e-5, rtol=1e-5):
        raise ValueError('Saved baseline label means do not match teacher probabilities')
    if not torch.equal(torch.bincount(initial).cpu(), baseline['counts'].long().cpu()):
        raise ValueError('Saved cell counts do not match assignments')
    options.update(learning_root=str(learning_root), output_dir=str(output_dir) if output_dir else None,
                   data=signatures, params=params, source_config=original_config,
                   assignment_digest=array_digest(initial.cpu().numpy()),
                   revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
    root = Path(output_dir or learning_root / 'stationarity') / _fingerprint(options)
    root.mkdir(parents=True, exist_ok=True)
    (root / 'config.json').write_text(json.dumps(options, indent=2), encoding='utf-8')
    return evaluate_study(root, train, validation, testing, h, q, z, transform, baseline, params,
                          original_config['settings'], penalties, max_iter, max_sweeps,
                          block_size, pair_batch, seed, final_seeds, evaluate_test)


def evaluate_study(root, train, validation, testing, h, q, z, transform, baseline, params,
                   settings, penalties, max_iter, max_sweeps, block_size, pair_batch, seed,
                   final_seeds, evaluate_test=True, candidate_k=None, random_candidates=0):
    initial = baseline['assignment'].to(z.device)
    y = train['y']
    val_ids = validation[1].nonzero().flatten()
    test_ids = testing[1].nonzero().flatten() if evaluate_test else None
    full_mass = z.new_full((len(z),), 1 / len(z))
    fits, candidates = [], []
    for penalty in penalties:
        head = fit_head(z, q, full_mass, penalty, max_iter)
        fits.append(head)
        candidates.append(dict(penalty=penalty, grad_norm=head['grad_norm'], converged=head['converged'],
                               **metrics(z, q, y, val_ids, head['theta'], penalty)))
    pd.DataFrame(candidates).to_csv(root / 'reference_grid.csv', index=False)
    eligible = [i for i, head in enumerate(fits) if head['converged']]
    if not eligible:
        raise RuntimeError('No reference head converged; inspect reference_grid.csv and increase max_iter')
    choice = min(eligible, key=lambda i: candidates[i]['val_ce'])
    penalty, reference = penalties[choice], fits[choice]
    theta = reference['theta']
    torch.save({k: v.cpu() if torch.is_tensor(v) else v for k, v in reference.items()}, root / 'reference.pt')
    refined_path = root / 'stationarity_partition.pt'
    if refined_path.exists():
        refined = torch.load(refined_path, map_location='cpu', weights_only=False)
    else:
        refined = refine_partition(z, q, initial, theta, penalty, max_sweeps, block_size, pair_batch, seed,
                                   candidate_k=candidate_k, random_candidates=random_candidates)
        torch.save(refined, refined_path)
    pd.DataFrame(refined['history']).to_csv(root / 'partition_history.csv', index=False)
    full_metrics = metrics(z, q, y, val_ids, theta, penalty, test_ids)
    pd.DataFrame([dict(penalty=penalty, grad_norm=reference['grad_norm'], **full_metrics)]).to_csv(
        root / 'reference.csv', index=False)
    summaries, students = [], []
    for method, partition in (('baseline', baseline), ('stationarity', refined)):
        assignment = partition['assignment'].to(z.device)
        centers, labels, mass = cell_means(z, q, assignment)
        x = augment(centers)
        errors = labels.sum(1, keepdim=True) * (x @ theta.T).softmax(1) - labels
        residual = ((mass[:, None] * errors).T @ x + penalty * theta).norm()
        fitted = fit_head(centers, labels, mass, penalty, max_iter)
        values = metrics(z, q, y, val_ids, fitted['theta'], penalty, test_ids)
        gap = float((fitted['theta'] - theta).norm())
        bound = float(residual) / penalty
        optimization_allowance = fitted['grad_norm'] / penalty
        raw_centers, _, _ = cell_means(h.double(), q, assignment)
        cx = raw_centers.float()
        error = float((mass * (transform(cx.double()) - centers).square().sum(1)).sum())
        final_path = root / f'{method}_students.csv'
        if final_path.exists():
            final = pd.read_csv(final_path)
        else:
            records = []
            for student_seed in final_seeds:
                val, test, epoch = _train_student(cx, labels.float(), validation, params, student_seed,
                                                  {**settings, 'loss_weighting': 'mass'},
                                                  testing=testing if evaluate_test else None,
                                                  counts=partition['counts'])
                records.append(dict(seed=student_seed, val=100 * val, test=100 * test if test is not None else np.nan,
                                    epoch=epoch))
                pd.DataFrame(records).to_csv(root / f'{method}_students.partial.csv', index=False)
            final = pd.DataFrame(records)
            final.to_csv(final_path, index=False)
        students.append(final.assign(method=method))
        torch.save({k: v.cpu() if torch.is_tensor(v) else v for k, v in fitted.items()}, root / f'{method}_head.pt')
        summaries.append(dict(method=method, penalty=penalty, gradient_norm=float(residual),
                              head_distance=gap, distance_bound=bound, fit_allowance=optimization_allowance,
                              bound_violation=max(0., gap - bound - optimization_allowance),
                              excess_full_objective=values['full_objective'] - full_metrics['full_objective'],
                              linear_val=values['val'], linear_test=values.get('test', np.nan),
                              linear_converged=fitted['converged'], linear_grad=fitted['grad_norm'],
                              val=final.val.mean(), val_std=final.val.std(ddof=0),
                              test=final.test.mean(), test_std=final.test.std(ddof=0),
                              realization_error=error,
                              changed_fraction=float((assignment != initial).double().mean()),
                              partition_converged=partition['converged'],
                              partition_status=partition.get('status', 'converged' if partition['converged'] else 'iteration_limit'),
                              candidate_k=partition.get('candidate_k'), output_dir=str(root)))
        pd.DataFrame(summaries).to_csv(root / 'summary.csv', index=False)
        pd.concat(students).to_csv(root / 'students.csv', index=False)
    return pd.DataFrame(summaries)
