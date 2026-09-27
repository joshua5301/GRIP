import gc
import hashlib
import json
import subprocess
import time
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric import seed_everything
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from src.dataloader import get_dataset
from src.hyperparams import BEST_HYPERPARAMS_DICT
from src.models import GCN
from src.partition import partition as grip_partition
from src.partition import PAIRED_INITS
from src.risk_partition import risk_partition
from src.uniform_transport import uniform_transport
from src.gcn_aware import gcn_aware_partition
from src.teacher import fit_logistic, get_kernel_features
from src.utils import BUDGET, normalize_adj_sparse
from src.grid_search import GridStudy
from src.grip_distance import distance_weights, training_support_distance
from src.grip_candidates import candidate_initialization
from src.fsw import FSWEncoder, neighborhoods, realize_graph


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:12]


def _prepare_dataset(name, data_dir, device):
    args = SimpleNamespace(dataset_name=name, raw_data_dir=str(data_dir).rstrip('/') + '/')
    datasets = get_dataset(args)

    def pack(graph):
        edges, weights = gcn_norm(graph.edge_index, graph.edge_attr, graph.num_nodes,
                                  dtype=graph.x.dtype)
        adjacency = torch.sparse_coo_tensor(edges.flip(0), weights,
                                            (graph.num_nodes, graph.num_nodes))
        adjacency = adjacency.coalesce().to_sparse_csr().to(device)
        return dict(x=graph.x.to(device), y=graph.y.to(device), adj=adjacency)

    if isinstance(datasets, list):
        train, val, test = [pack(g) for g in datasets]
        train_mask = datasets[0].train_mask.to(device)
        validation, testing = (val, None), (test, None)
    else:
        train = pack(datasets)
        train_mask = datasets.train_mask.to(device)
        validation = (train, datasets.val_mask.to(device))
        testing = (train, datasets.test_mask.to(device))
    with torch.no_grad():
        source = datasets[0] if isinstance(datasets, list) else datasets
        propagation = normalize_adj_sparse(source).coalesce().to_sparse_csr().to(device)
        H = torch.sparse.mm(propagation, torch.sparse.mm(propagation, train['x']))
    return train, train_mask, validation, testing, H


def _forward(model, x, adjacency=None):
    for i, layer in enumerate(model.layers):
        x = layer.lin(x)
        if adjacency is not None:
            x = adjacency @ x if adjacency.layout == torch.strided else torch.sparse.mm(adjacency, x)
        if layer.bias is not None:
            x = x + layer.bias
        if i + 1 < len(model.layers):
            x = F.dropout(F.relu(x), p=model.dropout, training=model.training)
    return F.log_softmax(x, dim=1)


@torch.no_grad()
def _accuracy(model, evaluation):
    graph, mask = evaluation
    model.eval()
    prediction = _forward(model, graph['x'], graph['adj']).argmax(1)
    correct = prediction.eq(graph['y'])
    return float(correct.float().mean() if mask is None else correct[mask].float().mean())


def _train_student(cx, cy, validation, params, seed, settings, testing=None, counts=None, return_model=False,
                   adjacency=None):
    weighting = settings.get('loss_weighting', 'uniform')
    if weighting not in ('uniform', 'mass'):
        raise ValueError('Require uniform or mass loss weighting')
    weights = None
    if weighting == 'mass':
        if counts is None:
            raise ValueError('Mass-weighted CE requires condensed cell counts')
        weights = counts.to(device=cx.device, dtype=cy.dtype)
        if weights.shape != (len(cx),) or not bool(torch.isfinite(weights).all()) or bool((weights <= 0).any()):
            raise ValueError('Require positive finite mass for every representative')
        weights = weights / weights.sum()
    seed_everything(seed)
    model = GCN(cx.shape[1], settings['hidden'], cy.shape[1], 2, params['dropout']).to(cx.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'],
                                 weight_decay=params['weight_decay'])
    best_val, best_epoch, best_state = -1.0, 0, None
    for epoch in range(1, settings['epochs'] + 1):
        if epoch == settings['epochs'] // 2:
            optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'] * 0.1,
                                         weight_decay=params['weight_decay'])
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = -(cy * _forward(model, cx, adjacency)).sum(1)
        loss = losses.mean() if weights is None else (weights * losses).sum()
        loss.backward()
        optimizer.step()
        if epoch % settings['eval_every'] == 0 or epoch == settings['epochs']:
            value = _accuracy(model, validation)
            if value > best_val:
                best_val, best_epoch = value, epoch
                if testing is not None or return_model:
                    best_state = {k: v.detach().cpu().clone()
                                  for k, v in model.state_dict().items()}
    test_value = None
    if testing is not None or return_model:
        model.load_state_dict(best_state)
    if testing is not None:
        test_value = _accuracy(model, testing)
    if return_model:
        model.eval()
        return best_val, test_value, best_epoch, model
    return best_val, test_value, best_epoch


def run_experiments(datasets, output_dir, n_trials=20, space=None,
                    data_dir='/content/data/', search_seeds=(0, 1, 2),
                    final_seeds=tuple(range(100, 110)), partition=None, teacher=None,
                    seed=0, epochs=1000, eval_every=10, hidden=256,
                    lr=0.01, weight_decay=5e-4, device='cuda', method='risk',
                    grip_steps=300, evaluate_test=True, initial_configs=(), loss_weighting='uniform',
                    aware=None, grip_init='kmeans', grip_init_block_size=256, search='optuna',
                    grip_seed=1234, grip_candidates=None, grip_candidate_seed=0,
                    grip_candidate_refinement='kmeans', fsw=None):
    if grip_candidate_refinement not in ('kmeans', 'grip') or (grip_candidate_refinement == 'grip' and grip_candidates is None):
        raise ValueError('Direct GRIP initialization requires a candidate pool')
    if grip_candidates is not None and (grip_candidates not in ('train', 'random') or method != 'grip' or grip_init != 'kmeans'):
        raise ValueError('Candidate initialization requires GRIP with kmeans and train/random candidates')
    if search not in ('optuna', 'grid'):
        raise ValueError('Unknown search method')
    if search == 'grid' and (not space or any(not isinstance(v, (list, tuple)) or not v for v in space.values())):
        raise ValueError('Grid search requires nonempty lists for every parameter')
    if search == 'grid' and initial_configs:
        raise ValueError('Include initial configurations in the grid instead')
    if grip_init not in ('kmeans', 'kmeans++', 'greedy') + PAIRED_INITS or grip_init_block_size < 1:
        raise ValueError('Invalid GRIP initialization settings')
    if loss_weighting not in ('uniform', 'mass'):
        raise ValueError('Require uniform or mass loss weighting')
    if method not in ('risk', 'risk_local', 'grip', 'transport', 'corrected', 'gcn_aware', 'risk_fro', 'grip_distance', 'fsw_grip') or grip_steps < 1:
        raise ValueError('Unknown method or invalid grip_steps')
    if method == 'fsw_grip' and loss_weighting != 'mass':
        raise ValueError('FSW-GRIP requires mass-weighted CE')
    fsw_settings = dict(seed=0, neighbors=8, steps=200, lr=.01)
    fsw_settings.update(fsw or {})
    if set(fsw_settings) - {'seed', 'neighbors', 'steps', 'lr'}:
        raise ValueError('Unknown FSW graph realization setting')
    if method == 'grip_distance' and (grip_init == 'greedy' or loss_weighting != 'uniform'):
        raise ValueError('Distance GRIP requires feature-only initialization and uniform CE')
    aware = dict(aware or {})
    if method == 'gcn_aware':
        if loss_weighting != 'uniform':
            raise ValueError('GCN-aware matching requires uniform student CE')
        if set(aware.get('probe_seeds', (1000, 1001))) & (set(search_seeds) | set(final_seeds)):
            raise ValueError('Use separate probe and evaluation seeds')
    if n_trials < 1 or not search_seeds or not final_seeds or min(epochs, eval_every) < 1:
        raise ValueError('Require positive trial/epoch counts and nonempty evaluation seeds')
    if space is None:
        space = dict(B=dict(low=0.01, high=100.0, log=True),
                     teacher_kernel=['erf', 'relu'], gamma=[0.01, 0.1, 1.0],
                     T=[0.2, 0.5, 1.0, 2.0], basis=[3000], dropout=[0.1, 0.5, 0.9])
        if method in ('grip', 'grip_distance', 'fsw_grip'):
            space.pop('B')
            space['kl_weight'] = dict(low=0.01, high=10.0, log=True)
    teacher_keys = {'teacher_kernel', 'gamma', 'T', 'basis'}
    coefficient = 'kl_weight' if method in ('grip', 'grip_distance', 'fsw_grip') else 'B'
    distance_keys = {'distance_k', 'distance_power'} if method == 'grip_distance' else set()
    fsw_keys = {'fsw_depth', 'fsw_width', 'fsw_frequency'} if method == 'fsw_grip' else set()
    if set(space) - teacher_keys - distance_keys - fsw_keys - {coefficient, 'dropout', 'lr', 'weight_decay'}:
        raise ValueError(f'Unsupported search parameter for {method}')
    aliases = {'kernel': 'teacher_kernel', 'temperature': 'T'}
    teacher = {aliases.get(k, k): v for k, v in (teacher or {}).items()}
    if set(teacher) - teacher_keys:
        raise ValueError('Unknown teacher setting')
    solver = dict(max_sweeps=30, block_size=1024, atol=1e-12, rtol=1e-10)
    solver.update(partition or {})
    if method == 'corrected':
        solver['objective_mode'] = 'uniform'
    if method == 'risk_fro':
        solver['objective_mode'] = 'frobenius'
    if method == 'risk_local':
        solver['objective_mode'] = 'local'
    settings = dict(epochs=epochs, eval_every=eval_every, hidden=hidden, loss_weighting=loss_weighting)
    output_dir, device = Path(output_dir), torch.device(device)
    output_dir.mkdir(parents=True, exist_ok=True)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True,
                                       cwd=Path(__file__).resolve().parents[1]).strip()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if search == 'optuna':
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    summaries = []

    for name, ratios in datasets.items():
        train, train_mask, validation, testing, H = _prepare_dataset(name, data_dir, device)
        teacher_cache = OrderedDict()
        feature_cache = {}
        distance_cache = {}
        fsw_cache = {}
        if method == 'fsw_grip':
            original_edges = train['adj'].to_sparse_coo().coalesce().indices()
            original_layout = neighborhoods(original_edges, len(H))

        def get_fsw(params):
            key = tuple(params[k] for k in ('fsw_depth', 'fsw_width', 'fsw_frequency'))
            if fsw_cache.get('key') != key:
                fsw_cache.clear()
                encoder = FSWEncoder(train['x'].shape[1], *key, seed=fsw_settings['seed']).to(device)
                embedding = encoder.fit_transform(train['x'], original_layout)
                fsw_cache.update(key=key, encoder=encoder, embedding=embedding)
            return fsw_cache['encoder'], fsw_cache['embedding']

        def get_labels(params):
            kernel, gamma, temperature, basis = (params[k] for k in
                                                  ('teacher_kernel', 'gamma', 'T', 'basis'))
            if gamma < 0 or temperature <= 0 or basis < 1 or int(basis) != basis:
                raise ValueError('Require gamma >= 0, T > 0 and positive integer basis')
            basis = int(basis)
            key = (kernel, gamma, basis)
            if key not in teacher_cache:
                feature_key = (kernel, basis)
                if feature_cache.get('key') != feature_key:
                    feature_cache.clear()
                    gc.collect()
                    seed_everything(seed)
                    feature_cache['features'] = get_kernel_features(H, kernel, basis)
                    feature_cache['key'] = feature_key
                features = feature_cache['features']
                labels = F.one_hot(train['y'][train_mask], int(train['y'].max()) + 1)
                W = fit_logistic(features[train_mask], labels.to(features.dtype), gamma)
                teacher_cache[key] = (features @ W).detach().cpu()
                if len(teacher_cache) > 4:
                    teacher_cache.popitem(last=False)
            teacher_cache.move_to_end(key)
            return F.softmax(teacher_cache[key].to(device) / temperature, dim=1)

        for ratio in ratios:
            defaults = BEST_HYPERPARAMS_DICT[(name, ratio)]
            m = BUDGET[(name, ratio)]
            teacher_config = dict(teacher_kernel=defaults[0], gamma=defaults[1],
                                  T=defaults[2], basis=3000)
            teacher_config.update(teacher)
            protocol = dict(version=3, revision=revision, torch=str(torch.__version__),
                            method=method, grip_steps=grip_steps, evaluate_test=evaluate_test,
                            grip_init=grip_init, grip_init_block_size=grip_init_block_size,
                            grip_seed=grip_seed,
                            grip_candidates=grip_candidates, grip_candidate_seed=grip_candidate_seed,
                            grip_candidate_refinement=grip_candidate_refinement,
                            initial_configs=initial_configs,
                            aware=aware,
                            dataset=name, ratio=ratio, budget=m, data_dir=str(data_dir),
                            teacher=teacher_config, seed=seed, partition=solver,
                            space=space, search=search, grid_order=list(space) if search == 'grid' else None,
                            search_seeds=list(search_seeds),
                            final_seeds=list(final_seeds), student=settings,
                            lr=lr, weight_decay=weight_decay, layers=2, loss=f'{loss_weighting}_soft_ce')
            if method == 'fsw_grip':
                protocol['fsw'] = fsw_settings
            case = output_dir / f'{name}_{ratio:g}_{_fingerprint(protocol)}'
            case.mkdir(parents=True, exist_ok=True)
            (case / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
            candidate_state = None
            if grip_candidates is not None:
                candidate_path = case / 'candidate_initialization.pt'
                if candidate_path.exists():
                    candidate_state = torch.load(candidate_path, map_location='cpu', weights_only=True)
                else:
                    candidate_state = candidate_initialization(H, train_mask, m, grip_candidates,
                                                               grip_candidate_seed, grip_seed, grip_candidate_refinement)
                    temporary = candidate_path.with_suffix('.tmp')
                    torch.save(candidate_state, temporary)
                    temporary.replace(candidate_path)
            study = GridStudy(space, case) if search == 'grid' else optuna.create_study(
                study_name='validation', storage=f'sqlite:///{case / "study.db"}',
                direction='maximize', sampler=optuna.samplers.TPESampler(seed=seed),
                pruner=optuna.pruners.NopPruner(), load_if_exists=True)
            if not study.trials:
                for candidate in initial_configs:
                    if candidate['dataset'] == name and candidate['ratio'] == ratio:
                        for key, value in candidate['params'].items():
                            if key not in space:
                                raise ValueError(f'Initial parameter {key} is outside the search space')
                            specification = space[key]
                            valid = (value in specification if isinstance(specification, (list, tuple))
                                     else specification['low'] <= value <= specification['high'])
                            if not valid:
                                raise ValueError(f'Initial {key}={value} is outside the search space')
                        study.enqueue_trial(candidate['params'])
            def get_condensed(params):
                condensation_config = {k: params[k] for k in sorted(teacher_keys | {coefficient} | distance_keys | fsw_keys)}
                path = case / f'condensed_{_fingerprint(condensation_config)}.pt'
                if path.exists():
                    return torch.load(path, map_location='cpu', weights_only=True), path.name
                Q = get_labels(params)
                if method == 'fsw_grip':
                    if device.type == 'cuda':
                        torch.cuda.synchronize(device)
                    started = time.perf_counter()
                    encoder, embedding = get_fsw(params)
                    state = grip_partition(embedding, Q, m, kl_weight=params['kl_weight'],
                                           iters=grip_steps, seed=grip_seed, init=grip_init,
                                           init_block_size=grip_init_block_size, return_diagnostics=True)
                    assignment, centers = state['assignment'].to(device), state['x'].to(device)
                    graph = realize_graph(train['x'], original_edges, assignment, centers, encoder,
                                          **{k: v for k, v in fsw_settings.items() if k != 'seed'})
                    with torch.no_grad():
                        weights = state['counts'].to(device) / len(embedding)
                        fit = (embedding - centers[assignment]).norm(dim=1).mean()
                        actual = graph['realized_embedding'].to(device)
                        direct = (embedding - actual[assignment]).norm(dim=1).mean()
                        reconstruction = ((centers - actual).norm(dim=1) * weights).sum()
                    if device.type == 'cuda':
                        torch.cuda.synchronize(device)
                    condensed = dict(**graph, y=state['y'], counts=state['counts'],
                                     assignment=state['assignment'], embedding_centers=state['x'],
                                     encoder_state={k: v.cpu() for k, v in encoder.state_dict().items()},
                                     converged=state['converged'], J=state['final_J'],
                                     history=[state['initial_J']], sweeps=None,
                                     embedding_fit=float(fit), embedding_direct=float(direct),
                                     embedding_upper=float(fit + reconstruction),
                                     seconds=time.perf_counter() - started)
                elif method == 'gcn_aware':
                    feature_cache.clear()
                    gc.collect()
                    condensed = gcn_aware_partition(H, Q, m, params['B'], train, train_mask,
                                                    hidden=hidden, seed=seed, **solver, **aware)
                elif method not in ('grip', 'grip_distance'):
                    partitioner = uniform_transport if method == 'transport' else risk_partition
                    condensed = partitioner(H, Q, m, params['B'], seed=seed, **solver)
                else:
                    seed_everything(seed)
                    if H.is_cuda:
                        torch.cuda.synchronize(device)
                    started = time.perf_counter()
                    label_weights = None
                    if method == 'grip_distance' and params['distance_power'] != 0:
                        k = params['distance_k']
                        if k not in distance_cache:
                            distance_cache[k] = training_support_distance(H, train_mask, k)
                        label_weights = distance_weights(distance_cache[k], params['distance_power'])
                    x, y, assignment, converged = grip_partition(
                        H, Q, m, kl_weight=params['kl_weight'], iters=grip_steps,
                        return_state=True, seed=grip_seed, init=grip_init,
                        init_block_size=grip_init_block_size, label_weights=label_weights,
                        initial_assignment=None if candidate_state is None else candidate_state.get('assignment'),
                        initial_node_ids=candidate_state['initial_centroid_ids']
                        if candidate_state is not None and grip_candidate_refinement == 'grip' else None)
                    condensed = dict(x=x.float().cpu(), y=y.float().cpu(),
                                     counts=torch.bincount(assignment).cpu(),
                                     converged=converged, J=None, history=[None], sweeps=None)
                    condensed['seconds'] = time.perf_counter() - started
                condensed['config'] = condensation_config
                temporary = path.with_suffix('.tmp')
                torch.save(condensed, temporary)
                temporary.replace(path)
                gc.collect()
                return condensed, path.name

            def objective(trial):
                params = dict(dropout=float(defaults[4].split(',')[0]),
                              lr=lr, weight_decay=weight_decay, **teacher_config)
                params[coefficient] = 0.5 if method in ('grip', 'grip_distance', 'fsw_grip') else 1.0
                if method == 'fsw_grip':
                    params.update(fsw_depth=2, fsw_width=128, fsw_frequency=2.)
                if method == 'grip_distance':
                    params.update(distance_k=10, distance_power=1.)
                for key, specification in space.items():
                    if isinstance(specification, (list, tuple)):
                        params[key] = trial.suggest_categorical(key, list(specification))
                    elif key == 'basis':
                        params[key] = trial.suggest_int(key, **specification)
                    else:
                        params[key] = trial.suggest_float(key, **specification)
                condensed, artifact = get_condensed(params)
                cx, cy = condensed['x'].to(device), condensed['y'].to(device)
                adjacency = condensed.get('adjacency')
                adjacency = None if adjacency is None else adjacency.to(device)
                values = [_train_student(cx, cy, validation, params, s, settings,
                                         counts=condensed['counts'], adjacency=adjacency)[0]
                          for s in search_seeds]
                for key, value in dict(config=params, artifact=artifact, validation_runs=values,
                                       nodes=len(condensed['x']),
                                       J_initial=condensed['history'][0], J_final=condensed['J'],
                                       converged=condensed['converged'], sweeps=condensed['sweeps'],
                                       partition_seconds=condensed['seconds']).items():
                    trial.set_user_attr(key, value)
                for key in ('status', 'mass_tv', 'row_residual', 'column_residual',
                            'mass_correction', 'variance_term', 'correction_term', 'moment_term',
                            'objective_name', 'label_gap', 'label_converged', 'head_stationarity_max',
                            'covariance_fro', 'covariance_trace', 'trace_bound', 'feature_term', 'trace_to_fro',
                            'global_moment_error', 'local_moment_error', 'cancellation_ratio',
                            'realization_initial', 'realization_final', 'embedding_fit', 'embedding_direct', 'embedding_upper'):
                    if key in condensed:
                        trial.set_user_attr(key, condensed[key])
                return float(np.mean(values))

            if search == 'grid':
                study.optimize(objective)
            else:
                completed = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
                remaining = max(0, n_trials - completed)
                if remaining:
                    study.optimize(objective, n_trials=remaining, n_jobs=1,
                                   gc_after_trial=True, show_progress_bar=True)
            study.trials_dataframe().to_csv(case / 'trials.csv', index=False)
            best = study.best_trial
            params = best.user_attrs['config']
            (case / 'best.json').write_text(json.dumps(dict(
                trial=best.number, validation=best.value, params=params,
                artifact=best.user_attrs['artifact']), indent=2), encoding='utf-8')
            final_path = case / f'final_trial_{best.number}.csv'
            if final_path.exists():
                final = pd.read_csv(final_path)
            else:
                condensed = torch.load(case / best.user_attrs['artifact'],
                                       map_location='cpu', weights_only=True)
                cx, cy = condensed['x'].to(device), condensed['y'].to(device)
                adjacency = condensed.get('adjacency')
                adjacency = None if adjacency is None else adjacency.to(device)
                records = []
                for s in final_seeds:
                    val, test, epoch = _train_student(cx, cy, validation, params, s,
                                                     settings, testing if evaluate_test else None,
                                                     counts=condensed['counts'], adjacency=adjacency)
                    records.append(dict(seed=s, validation=val,
                                        test=test if test is not None else np.nan, best_epoch=epoch))
                final = pd.DataFrame(records)
                temporary = final_path.with_suffix('.tmp')
                final.to_csv(temporary, index=False)
                temporary.replace(final_path)
                del cx, cy
            summaries.append(dict(
                dataset=name, ratio=ratio, method=method, search=search, nodes=best.user_attrs['nodes'],
                requested_nodes=m, loss_weighting=loss_weighting, **params,
                search_val=100 * best.value, final_val=100 * final['validation'].mean(),
                final_val_std=100 * final['validation'].std(ddof=1) if len(final) > 1 else 0.0,
                test_mean=100 * final['test'].mean(),
                test_std=100 * final['test'].std(ddof=1) if len(final) > 1 else 0.0,
                J_initial=best.user_attrs['J_initial'], J_final=best.user_attrs['J_final'],
                converged=best.user_attrs['converged'], sweeps=best.user_attrs['sweeps'],
                partition_seconds=best.user_attrs['partition_seconds'], folder=str(case)))
            if method in ('grip', 'grip_distance', 'fsw_grip'):
                summaries[-1]['grip_init'] = grip_init
                summaries[-1]['grip_seed'] = grip_seed
                summaries[-1]['grip_candidates'] = grip_candidates
                summaries[-1]['grip_candidate_seed'] = grip_candidate_seed
                summaries[-1]['grip_candidate_refinement'] = grip_candidate_refinement
            if method == 'fsw_grip':
                summaries[-1].update({k: best.user_attrs[k] for k in
                    ('realization_initial', 'realization_final', 'embedding_fit', 'embedding_direct', 'embedding_upper')})
            if method == 'transport':
                summaries[-1].update({k: best.user_attrs[k] for k in
                                      ('status', 'mass_tv', 'row_residual', 'column_residual')})
            if method == 'corrected':
                summaries[-1].update({k: best.user_attrs[k] for k in
                    ('mass_tv', 'mass_correction', 'variance_term', 'correction_term', 'moment_term')})
            if method == 'gcn_aware':
                summaries[-1].update({k: best.user_attrs[k] for k in
                    ('status', 'objective_name', 'label_gap', 'label_converged', 'head_stationarity_max')})
            if method == 'risk_fro':
                summaries[-1].update({k: best.user_attrs[k] for k in
                    ('objective_name', 'covariance_fro', 'covariance_trace', 'trace_bound',
                     'feature_term', 'moment_term', 'trace_to_fro')})
            if method == 'risk_local':
                summaries[-1].update({k: best.user_attrs[k] for k in
                    ('objective_name', 'global_moment_error', 'local_moment_error', 'cancellation_ratio',
                     'variance_term', 'moment_term')})
            pd.DataFrame(summaries).to_csv(output_dir / 'summary.csv', index=False)
        teacher_cache.clear()
        feature_cache.clear()
        distance_cache.clear()
        fsw_cache.clear()
        del train, train_mask, validation, testing, H
        gc.collect()
        torch.cuda.empty_cache()
    return pd.DataFrame(summaries)
