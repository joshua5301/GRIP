import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric import seed_everything
from tqdm.auto import tqdm

from src.models import GCN
from src.node_distances import array_digest
from src.ntk_transforms import fit_transform
from src.risk_experiment import _accuracy, _fingerprint, _forward, _prepare_dataset
from src.risk_partition import risk_partition
from src.stationarity_risk import augment, evaluate_study, fit_head, metrics
from src.teacher import fit_logistic, get_kernel_features


def full_graph_student(train, train_mask, q, validation, testing, params, settings, seed, target):
    seed_everything(seed)
    model = GCN(train['x'].shape[1], settings['hidden'], q.shape[1], 2, params['dropout']).to(q.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    best_val, best_epoch, best_state = -1., 0, None
    for epoch in range(1, settings['epochs'] + 1):
        if epoch == settings['epochs'] // 2:
            optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'] * .1, weight_decay=params['weight_decay'])
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = _forward(model, train['x'], train['adj'])
        loss = (F.nll_loss(prediction[train_mask], train['y'][train_mask]) if target == 'train_true'
                else -(q.float() * prediction).sum(1).mean())
        loss.backward()
        optimizer.step()
        if epoch % settings['eval_every'] == 0 or epoch == settings['epochs']:
            value = _accuracy(model, validation)
            if value > best_val:
                best_val, best_epoch = value, epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    test = _accuracy(model, testing) if testing is not None else np.nan
    return dict(target=target, seed=seed, val=100 * best_val, test=100 * test, epoch=best_epoch)


def run_arxiv_stationarity(output_dir, nodes=909, params=None,
                           penalties=(.0001, .001, .01, .1), baseline_sweeps=100,
                           max_sweeps=20, candidate_k=16, random_candidates=4,
                           block_size=128, pair_batch=1024, max_iter=2000,
                           final_seeds=tuple(range(100, 110)), full_seeds=(100, 101, 102),
                           epochs=1000, eval_every=10, hidden=256, seed=0,
                           data_dir='/content/data/', device='cuda', evaluate_test=True):
    params = dict(params or dict(teacher_kernel='relu', basis=3000, gamma=.0037495720752924867,
                                T=.11302760163097268, B=1.1248257310473129,
                                dropout=.31881090213944857, lr=.01, weight_decay=.0005))
    options = locals().copy()
    options['output_dir'] = str(output_dir)
    if not final_seeds or not penalties or min(penalties) <= 0 or params['T'] <= 0:
        raise ValueError('Require student seeds, positive penalties and temperature')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    train, mask, validation, testing, h = _prepare_dataset('arxiv', data_dir, device)
    if not isinstance(nodes, int) or not 1 <= nodes <= len(h):
        raise ValueError('nodes must be an integer between 1 and the original node count')
    adjacency = train['adj'].to_sparse_csr()
    options.update(data_digest=array_digest(
        train['x'].cpu().numpy(), train['y'].cpu().numpy(), adjacency.crow_indices().cpu().numpy(),
        adjacency.col_indices().cpu().numpy(), adjacency.values().cpu().numpy(), mask.cpu().numpy(),
        validation[1].cpu().numpy(), testing[1].cpu().numpy()),
        dataset='arxiv', original_nodes=len(h), actual_node_ratio=nodes / len(h),
        revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
    root = Path(output_dir) / _fingerprint(options)
    root.mkdir(parents=True, exist_ok=True)
    (root / 'config.json').write_text(json.dumps(options, indent=2), encoding='utf-8')
    logits_path = root / 'teacher_logits.pt'
    if logits_path.exists():
        logits = torch.load(logits_path, map_location=device, weights_only=True)
    else:
        seed_everything(seed)
        features = get_kernel_features(h, params['teacher_kernel'], params['basis'])
        labels = F.one_hot(train['y'][mask], int(train['y'].max()) + 1).double()
        weight = fit_logistic(features[mask], labels, params['gamma'])
        logits = (features @ weight).detach()
        torch.save(logits.cpu(), logits_path)
        del features, labels, weight
    q = (logits / params['T']).softmax(1).double()
    teacher = dict(val=100 * float((logits[validation[1]].argmax(1) == train['y'][validation[1]]).float().mean()))
    if evaluate_test:
        teacher['test'] = 100 * float((logits[testing[1]].argmax(1) == train['y'][testing[1]]).float().mean())
    pd.DataFrame([teacher]).to_csv(root / 'label_teacher.csv', index=False)
    del logits
    baseline_path = root / 'baseline_partition.pt'
    if baseline_path.exists():
        baseline = torch.load(baseline_path, map_location='cpu', weights_only=False)
    else:
        baseline = risk_partition(h, q, nodes, params['B'], seed=seed, max_sweeps=baseline_sweeps,
                                  block_size=256, return_assignment=True)
        torch.save(baseline, baseline_path)
    z, transform = fit_transform(h.double(), kind='rms')
    settings = dict(epochs=epochs, eval_every=eval_every, hidden=hidden, loss_weighting='mass')
    result = evaluate_study(root, train, validation, testing, h, q, z, transform, baseline, params,
                            settings, penalties, max_iter, max_sweeps, block_size, pair_batch,
                            seed, final_seeds, evaluate_test, candidate_k, random_candidates)
    result = result.assign(dataset='arxiv', nodes=nodes, original_nodes=len(h), actual_node_ratio=nodes / len(h))
    result.to_csv(root / 'summary.csv', index=False)
    records = []
    for target, student_seed in tqdm([(t, s) for t in ('train_true', 'teacher_all') for s in full_seeds],
                                     desc='Full-graph GCN references'):
        record_path = root / f'full_{target}_{student_seed}.json'
        if record_path.exists():
            record = json.loads(record_path.read_text())
        else:
            record = full_graph_student(train, mask, q, validation, testing if evaluate_test else None,
                                         params, settings, student_seed, target)
            record_path.write_text(json.dumps(record, indent=2), encoding='utf-8')
        records.append(record)
        pd.DataFrame(records).to_csv(root / 'full_students.csv', index=False)
    return result


def sweep_reference_penalty(source, output_dir=None,
                            penalties=(1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4),
                            max_iter=5000, grad_tol=1e-8, tolerance_change=1e-18,
                            data_dir='/content/data/', device='cuda', evaluate_test=True):
    source = Path(source)
    source_config = json.loads((source / 'config.json').read_text())
    penalties = sorted(set(float(p) for p in penalties), reverse=True)
    if source_config.get('dataset') != 'arxiv' or not penalties or any(not np.isfinite(p) or p <= 0 for p in penalties):
        raise ValueError('Require an Arxiv source and finite positive penalties')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    train, mask, validation, testing, h = _prepare_dataset('arxiv', data_dir, device)
    adjacency = train['adj'].to_sparse_csr()
    digest = array_digest(train['x'].cpu().numpy(), train['y'].cpu().numpy(),
                          adjacency.crow_indices().cpu().numpy(), adjacency.col_indices().cpu().numpy(),
                          adjacency.values().cpu().numpy(), mask.cpu().numpy(),
                          validation[1].cpu().numpy(), testing[1].cpu().numpy())
    if digest != source_config['data_digest']:
        raise ValueError('Graph or splits differ from saved Arxiv experiment')
    logits = torch.load(source / 'teacher_logits.pt', map_location=device, weights_only=True)
    config = dict(source=str(source), source_config=source_config, penalties=penalties,
                  logits_digest=array_digest(logits.cpu().numpy()), max_iter=max_iter, grad_tol=grad_tol,
                  tolerance_change=tolerance_change, evaluate_test=evaluate_test,
                  revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
    root = Path(output_dir or source / 'reference_penalty') / _fingerprint(config)
    root.mkdir(parents=True, exist_ok=True)
    (root / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    q = (logits / source_config['params']['T']).softmax(1).double()
    z, transform = fit_transform(h.double(), kind='rms')
    torch.save(transform.state_dict(), root / 'transform.pt')
    mass = z.new_full((len(z),), 1 / len(z))
    val_ids = validation[1].nonzero().flatten()
    rows, initial = [], None
    for penalty in tqdm(penalties, desc='Full-data linear penalty grid'):
        path = root / f'head_{penalty:.12g}.pt'
        if path.exists():
            head = torch.load(path, map_location=device, weights_only=False)
        else:
            started = time.perf_counter()
            head = fit_head(z, q, mass, penalty, max_iter, grad_tol,
                            initial_theta=initial, tolerance_change=tolerance_change)
            head['seconds'] = time.perf_counter() - started
            torch.save({k: v.cpu() if torch.is_tensor(v) else v for k, v in head.items()}, path)
        initial = head['theta']
        values = metrics(z, q, train['y'], val_ids, initial, penalty)
        with torch.no_grad():
            prediction = augment(z) @ initial.T
            logp = prediction.log_softmax(1)
            teacher_ce = float(-(q * logp).sum(1).mean())
            entropy = float(-(q * q.clamp_min(1e-300).log()).sum(1).mean())
            agreement = 100 * float((prediction.argmax(1) == q.argmax(1)).float().mean())
        rows.append(dict(penalty=penalty, **values, teacher_ce=teacher_ce,
                         teacher_kl=teacher_ce - entropy, teacher_agreement=agreement,
                         head_norm=float(initial.norm()), weight_norm=float(initial[:, :-1].norm()),
                         bias_norm=float(initial[:, -1].norm()), grad_norm=head['grad_norm'],
                         grad_max=head['grad_max'], parameter_error_bound=head['grad_norm'] / penalty,
                         converged=head['converged'], iterations=head['iterations'], seconds=head['seconds']))
        pd.DataFrame(rows).to_csv(root / 'grid.csv', index=False)
    table = pd.DataFrame(rows)
    eligible = table[table.converged]
    if eligible.empty:
        raise RuntimeError(f'No converged head; inspect {root / "grid.csv"} and increase max_iter')
    selected = eligible.loc[eligible.val_ce.idxmin()].to_dict()
    selected['at_lower_boundary'] = selected['penalty'] == min(penalties)
    selected['all_converged'] = bool(table.converged.all())
    selected['head_path'] = str(root / f'head_{selected["penalty"]:.12g}.pt')
    if evaluate_test:
        head = torch.load(selected['head_path'], map_location=device, weights_only=False)
        selected.update(metrics(z, q, train['y'], val_ids, head['theta'], selected['penalty'],
                                testing[1].nonzero().flatten()))
    pd.DataFrame([selected]).to_csv(root / 'selected.csv', index=False)
    return table.assign(output_dir=str(root))


def compare_reference_partitions(reference_root, output_dir=None, conditions=None,
                                 max_sweeps=20, candidate_k=16, random_candidates=4,
                                 block_size=128, pair_batch=1024, max_iter=5000,
                                 final_seeds=tuple(range(100, 110)), seed=0,
                                 data_dir='/content/data/', device='cuda', evaluate_test=True):
    reference_root = Path(reference_root)
    reference_config = json.loads((reference_root / 'config.json').read_text())
    source = Path(reference_config['source'])
    original = json.loads((source / 'config.json').read_text())
    conditions = dict(conditions or dict(previous=1e-4, ce_selected=3e-5, accuracy_selected=3e-6))
    if original != reference_config['source_config'] or original.get('dataset') != 'arxiv':
        raise ValueError('Reference sweep and source experiment differ')
    if not final_seeds or not conditions or any(not np.isfinite(p) or p <= 0 for p in conditions.values()):
        raise ValueError('Require seeds and finite positive penalties')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    train, mask, validation, testing, h = _prepare_dataset('arxiv', data_dir, device)
    adjacency = train['adj'].to_sparse_csr()
    digest = array_digest(train['x'].cpu().numpy(), train['y'].cpu().numpy(),
                          adjacency.crow_indices().cpu().numpy(), adjacency.col_indices().cpu().numpy(),
                          adjacency.values().cpu().numpy(), mask.cpu().numpy(),
                          validation[1].cpu().numpy(), testing[1].cpu().numpy())
    if digest != original['data_digest']:
        raise ValueError('Graph or split differs from the saved experiment')
    logits = torch.load(source / 'teacher_logits.pt', map_location=device, weights_only=True)
    if array_digest(logits.cpu().numpy()) != reference_config['logits_digest']:
        raise ValueError('Saved teacher logits changed after the reference sweep')
    params = original['params']
    q = (logits / params['T']).softmax(1).double()
    z, transform = fit_transform(h.double(), kind='rms')
    baseline = torch.load(source / 'baseline_partition.pt', map_location='cpu', weights_only=False)
    references = {name: torch.load(reference_root / f'head_{penalty:.12g}.pt', map_location=device, weights_only=False)
                  for name, penalty in conditions.items()}
    options = dict(reference_root=str(reference_root), reference_config=reference_config,
                   conditions=conditions, max_sweeps=max_sweeps, candidate_k=candidate_k,
                   random_candidates=random_candidates, block_size=block_size, pair_batch=pair_batch,
                   max_iter=max_iter, final_seeds=list(final_seeds), seed=seed, evaluate_test=evaluate_test,
                   assignment_digest=array_digest(baseline['assignment'].numpy()),
                   head_digests={name: array_digest(head['theta'].cpu().numpy()) for name, head in references.items()},
                   revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
    root = Path(output_dir or reference_root / 'partition_comparison') / _fingerprint(options)
    root.mkdir(parents=True, exist_ok=True)
    (root / 'config.json').write_text(json.dumps(options, indent=2), encoding='utf-8')
    settings = {k: original[k] for k in ('epochs', 'eval_every', 'hidden')}
    fit_options = {k: reference_config[k] for k in ('grad_tol', 'tolerance_change')}
    shared_baseline = root / 'baseline_students.csv'
    summaries, students, histories, full_references = [], [], [], []
    for name, penalty in tqdm(conditions.items(), desc='Reference-head conditions'):
        folder = root / _fingerprint(dict(name=name, penalty=penalty))
        folder.mkdir(exist_ok=True)
        if shared_baseline.exists():
            pd.read_csv(shared_baseline).to_csv(folder / 'baseline_students.csv', index=False)
        table = evaluate_study(folder, train, validation, testing, h, q, z, transform, baseline,
                               params, settings, [penalty], max_iter, max_sweeps, block_size,
                               pair_batch, seed, final_seeds, evaluate_test, candidate_k, random_candidates,
                               reference_override=references[name], head_fit_options=fit_options)
        if not shared_baseline.exists():
            pd.read_csv(folder / 'baseline_students.csv').to_csv(shared_baseline, index=False)
        summaries.append(table.assign(condition=name, case_dir=str(folder), output_dir=str(root)))
        students.append(pd.read_csv(folder / 'students.csv').assign(condition=name, penalty=penalty))
        histories.append(pd.read_csv(folder / 'partition_history.csv').assign(condition=name, penalty=penalty))
        full_references.append(pd.read_csv(folder / 'reference.csv').assign(condition=name))
        pd.concat(summaries).to_csv(root / 'summary.csv', index=False)
        pd.concat(students).to_csv(root / 'students.csv', index=False)
        pd.concat(histories).to_csv(root / 'partition_history.csv', index=False)
        pd.concat(full_references).to_csv(root / 'references.csv', index=False)
    results = pd.concat(summaries, ignore_index=True)
    best = results[results.method == 'stationarity'].sort_values(['val', 'condition'], ascending=[False, True]).iloc[0]
    (root / 'selected.json').write_text(json.dumps(dict(condition=best['condition'], penalty=float(best.penalty),
                                                      criterion='mean_gcn_validation', val=float(best.val)), indent=2),
                                       encoding='utf-8')
    return results
