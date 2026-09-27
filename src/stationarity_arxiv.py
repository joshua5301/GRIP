import json
import subprocess
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
from src.stationarity_risk import evaluate_study
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
