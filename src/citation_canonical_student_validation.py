"""One shared serving cache and four fixed canonical readout student arms.

No assignment, critic, source-SGC, target decoding or moment export occurs here.
The secondary route supplies original cached source-SGC H; it does not assert
that H equals S²X for the newly packed gcn_norm graph.
"""
import csv
import hashlib
import json
import math
import os
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch
from src.io import array_digest

KIND = 'canonical_P_raw_H_Q_four_arm_validation_v1'
SCIENCE_SHA = 'b52830f294c42899ed52bd0340a696c25055d2588ba3e29d5e145e6cec8ed28e'
SECONDARY = 'Supplied original cached source-SGC H (data.normalize_adj_sparse), exact GCN-selected weights; descriptive. No equality to newly packed gcn_norm S²X is asserted.'


def _require(value, message):
    if not value:
        raise ValueError(message)


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _read(ref):
    _require(_sha(ref['path']) == ref['sha256'], 'Pinned file changed')
    return json.loads(Path(ref['path']).read_text())


def _refs(value):
    result = {}
    if isinstance(value, dict):
        if {'path', 'sha256'} <= value.keys():
            result[value['path']] = value['sha256']
        for item in value.values():
            result.update(_refs(item))
    elif isinstance(value, list):
        for item in value:
            result.update(_refs(item))
    return result


def _own(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _own(item) for key, item in value.items()}
    return value


def _identity(t):
    t = t.detach().cpu()
    if t.layout == torch.sparse_csr:
        return dict(shape=list(t.shape), dtype=str(t.dtype), layout=str(t.layout),
                    crow=_identity(t.crow_indices()), col=_identity(t.col_indices()), values=_identity(t.values()))
    return dict(shape=list(t.shape), dtype=str(t.dtype), tensor=array_digest(t.numpy()))


def _cache_identity(cache):
    return dict(graph={key: _identity(value) for key, value in cache['graph'].items()},
                masks={key: _identity(value) for key, value in cache['masks'].items()},
                H=_identity(cache['propagated']), Q=_identity(cache['q']),
                arms={key: {name: _identity(t) for name, t in row.items()} for key, row in cache['arms'].items()})


def _observed(value):
    if isinstance(value, float) and not math.isfinite(value):
        return dict(nonfinite=repr(value))
    if isinstance(value, dict):
        return {key: _observed(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_observed(item) for item in value]
    return value


def run(protocol_path, protocol_sha256, phase, arm=None, stop=lambda: False):
    from src.research_loop import implementation_provenance
    _require(_sha(protocol_path) == protocol_sha256, 'Protocol changed')
    packet = json.loads(Path(protocol_path).read_text())
    _require(phase in ('prepare', 'evaluate') and (arm is None if phase == 'prepare' else type(arm) is str), 'Invalid phase/arm')
    folder = Path(packet['prepare_folder'] if phase == 'prepare' else packet['evaluation_folders'][arm])
    folder.mkdir(parents=True, exist_ok=False)
    started = time.monotonic(); cache = {}; failure = None; cuda_ready = False
    counts = dict(dataset_get_attempts=0, dataset_gets=0, graph_pack_attempts=0, graph_packs=0,
                  cache_write_attempts=0, cache_writes=0, fit_attempts=0, student_fits=0,
                  student_epochs=0, route_call_attempts=0, route_calls=0, validation_routes=0,
                  critic_heads=0, adjoints=0, P_updates=0, assignment_Adam_steps=0,
                  moment_exports=0, Q_decodes=0, source_SGC_propagations=0, test_evaluations=0,
                  native_factor_factories=0, teacher_kernel_Phi_RMS_fits=0, old_numerical_proof_replays=0)
    report = dict(schema=1, kind=KIND, phase=phase, arm=arm, passed=False, source=packet.get('source'),
                  protocol=dict(path=str(protocol_path), sha256=protocol_sha256), records=[], counts=counts,
                  secondary_provenance=SECONDARY, test_enabled=False, raw_cache_write_complete=False)
    def guard():
        limits = packet['resource_limits']
        _require(not stop() and time.monotonic()-started <= limits['max_seconds'], 'Terminal stop/deadline')
        if cuda_ready:
            _require(torch.cuda.max_memory_allocated() <= limits['peak_allocated_bytes'] and
                     torch.cuda.max_memory_reserved() <= limits['peak_reserved_bytes'], 'CUDA resource limit')
        return False
    def pins():
        _require(implementation_provenance() == packet['source'], 'Current source changed')
        for path, expected in packet['readonly_files_sha256'].items():
            _require(_sha(path) == expected, 'Readonly input changed: '+path)
        guard()
    try:
        contract = _read(packet['scientific_contract'])
        _require(packet['scientific_contract']['sha256'] == SCIENCE_SHA and type(packet['schema']) is int and packet['schema'] == 1
                 and packet['kind'] == KIND and packet['test_enabled'] is False and contract['test_enabled'] is False
                 and packet['resource_limits'] == contract['resource_limits'], 'Frozen domain changed')
        required = _refs(contract); required.update(_refs(packet['scientific_contract']))
        if phase == 'evaluate':
            required.update({ref['path']: ref['sha256'] for ref in
                            (packet['shared_cache'], packet['shared_prepare_report'], packet['shared_prepare_acceptance'])})
        _require(all(packet['readonly_files_sha256'].get(p) == h for p, h in required.items()), 'Incomplete readonly pins')
        torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest'); torch.cuda.init(); cuda_ready=True; torch.cuda.reset_peak_memory_stats(); pins()
        admission = _read(contract['canonical_export_actual_admission']); exported = _read(contract['canonical_export_report'])
        _require(admission['passed'] is True and exported['passed'] is True and
                 admission['actual_report'] == contract['canonical_export_report'] and
                 admission['actual_arrays'] == contract['canonical_export_arrays'] and admission['source'] == exported['source'], 'Export admission mismatch')
        report.update(scientific_contract=packet['scientific_contract'], canonical_export_producer=exported['source'],
                      canonical_export_report=contract['canonical_export_report'], canonical_export_arrays=contract['canonical_export_arrays'])
        settings = dict(contract['student_recipe']['settings']); _require(_read(contract['student_recipe']['file']) == settings
                 and settings.pop('input_scale') == 1.0 and contract['student_seeds'] == [20100,20101,20102]
                 and all(type(s) is int for s in contract['student_seeds']), 'Original recipe/cohort changed')
        if phase == 'prepare':
            from src.data import get_dataset
            from torch_geometric.nn.conv.gcn_conv import gcn_norm
            counts['dataset_get_attempts'] += 1
            data = get_dataset(SimpleNamespace(dataset_name='cora', raw_data_dir=contract['dataset_source']['data_dir'].rstrip('/')+'/', citation_features='row'))
            counts['dataset_gets'] += 1
            train, val = _own(data.train_mask), _own(data.val_mask)
            _require(train.dtype == val.dtype == torch.bool and train.shape == val.shape == (2708,)
                     and int(train.sum()) == 140 and int(val.sum()) == 500 and not bool((train & val).any())
                     and tuple(data.x.shape) == (2708,1433) and data.x.dtype == torch.float32, 'Dataset/split mismatch')
            allowed = train | val; labels = torch.full((2708,), -1, dtype=torch.int64); labels[allowed] = data.y[allowed].cpu()
            _require(bool((labels[allowed]>=0).all()) and bool((labels[allowed]<7).all()), 'Invalid allowed labels')
            cache.update(schema=1, kind=KIND, source=packet['source'], scientific_contract=packet['scientific_contract'],
                         graph=dict(x=_own(data.x), y=labels), masks=dict(train=train,val=val))
            guard(); counts['graph_pack_attempts'] += 1
            edges, weights = gcn_norm(data.edge_index, data.edge_attr, data.num_nodes, dtype=data.x.dtype)
            cache['graph']['adj'] = _own(torch.sparse_coo_tensor(edges.flip(0),weights,(2708,2708)).coalesce().to_sparse_csr()); counts['graph_packs'] += 1
            guard(); raw = torch.load(contract['canonical_export_arrays']['path'],map_location='cpu',weights_only=False)
            h = torch.load(contract['cached_H']['path'],map_location='cpu',weights_only=False)
            _require(type(h['schema']) is int and h['schema'] == 1 and h['kind'] == 'shared_h', 'Cached source-H schema mismatch')
            cache.update(propagated=_own(h['h']), q=_own(raw['new_source_Q']), arms={item['id']:
                         {key:_own(raw[item['id']][key]) for key in ('X','Q','uniform_weights')} for item in contract['arms']})
            cache['identities'] = _cache_identity(cache)
        else:
            prior = _read(packet['shared_prepare_report']); accepted = _read(packet['shared_prepare_acceptance'])
            _require(prior['passed'] is True and prior['phase'] == 'prepare' and prior['source'] == packet['source']
                     and accepted['passed'] is True and accepted['source'] == packet['source']
                     and accepted['shared_cache'] == packet['shared_cache'] and accepted['shared_prepare_report'] == packet['shared_prepare_report']
                     and prior['shared_cache'] == packet['shared_cache'], 'Shared cache not ROOT-admitted')
            cache = torch.load(packet['shared_cache']['path'],map_location='cpu',weights_only=False)
            _require(type(cache['schema']) is int and cache['schema'] == 1 and cache['kind'] == KIND and cache['source'] == packet['source']
                     and cache['scientific_contract'] == packet['scientific_contract'] and
                     cache['identities'] == _cache_identity(cache) == prior['cache_identities'], 'Shared cache identity mismatch')
        _require(cache['identities']['H'] == contract['expected_H'] and cache['identities']['Q'] == contract['expected_Q']
                 and set(cache['masks']) == {'train','val'} and set(cache['arms']) == {i['id'] for i in contract['arms']}, 'Shared source/arms mismatch')
        for item in contract['arms']:
            _require(cache['identities']['arms'][item['id']] == item['expected_descriptors'] and
                     all(bool(torch.isfinite(t).all()) for t in cache['arms'][item['id']].values()), 'Canonical readout bytes/finite mismatch')
            observed = next(row for row in exported['exports'] if row['id'] == item['id'])
            _require(observed['factor_provider'] == item['factor_provider'] and observed['step'] == item['P_step'], 'Historical provider mismatch')
        _require(bool(torch.isfinite(cache['graph']['x']).all()) and bool(torch.isfinite(cache['graph']['adj'].values()).all())
                 and bool(torch.isfinite(cache['propagated']).all()) and bool(torch.isfinite(cache['q']).all()), 'Nonfinite shared inputs')
        report['cache_identities'] = cache['identities']; guard()
        if phase == 'evaluate':
            from src.evaluation import fit_gcn_diagnostic
            from src.student_routes import replay_routes
            selected_arm = next(i for i in contract['arms'] if i['id'] == arm)
            report.update(shared_cache=packet['shared_cache'], shared_prepare_report=packet['shared_prepare_report'],
                          shared_prepare_acceptance=packet['shared_prepare_acceptance'], arm_provider=selected_arm['factor_provider'])
            graph = {k:t.to('cuda') for k,t in cache['graph'].items()}; masks = {k:t.to('cuda') for k,t in cache['masks'].items()}
            q, h = cache['q'].to('cuda'), cache['propagated'].to('cuda'); inputs = cache['arms'][arm]
            x,y,mass = (inputs[k].to('cuda') for k in ('X','Q','uniform_weights'))
            fit_folder = folder/f"step_{selected_arm['P_step']}_{contract['student_recipe']['id']}"
            for seed in contract['student_seeds']:
                guard(); counts['fit_attempts'] += 1
                fit = fit_gcn_diagnostic(x,y,mass,graph,q,masks,seed,folder=fit_folder,training_adjacency=None,stop=guard,**settings)
                counts['student_fits'] += 1; record = dict(seed=seed,fit=fit); report['records'].append(record)
                paths = {name:fit_folder/f'seed_{seed}{suffix}' for name,suffix in
                         (('json','.json'),('csv','_epochs.csv'),('selected','_selected.pt'))}
                record['fit_files'] = {name:dict(path=str(p),sha256=_sha(p)) for name,p in paths.items()}
                rows = list(csv.DictReader(paths['csv'].open())); counts['student_epochs'] += len(rows)
                saved = json.loads(paths['json'].read_text()); best = max(float(row['val_acc']) for row in rows)
                first = next(row for row in rows if float(row['val_acc']) == best)
                _require([int(r['epoch']) for r in rows] == list(range(1,601)) and fit['epoch'] == int(first['epoch'])
                         and fit['val_acc'] == best and saved['result'] == fit and saved['recipe']['test_enabled'] is False
                         and saved['recipe']['settings'] == settings and not any('test' in key for key in fit)
                         and all(not isinstance(v,float) or math.isfinite(v) for v in fit.values()), 'Incomplete/changed selected fit')
                guard(); route_path = fit_folder/f'seed_{seed}_routes.json'; counts['route_call_attempts'] += 1
                route = replay_routes(paths['selected'],graph,h,{'val':masks['val']},settings,route_path,seed=seed,stop=guard)
                counts['route_calls'] += 1; counts['validation_routes'] += 2; record['routes'] = route
                record['route_file'] = dict(path=str(route_path),sha256=_sha(route_path))
                cached_route=json.loads(route_path.read_text())
                _require(route['gcn_val_acc'] == fit['val_acc'] and route['epoch'] == fit['epoch'] and
                         cached_route['recipe']['source_fingerprint'] == saved['fingerprint'] and
                         cached_route['recipe']['test_enabled'] is False and not any('test' in key for key in route)
                         and all(not isinstance(v,float) or math.isfinite(v) for v in route.values()), 'Selected-weight route mismatch')
                guard()
            _require(counts['student_fits']==3 and counts['student_epochs']==1800 and counts['validation_routes']==6,'Incomplete arm')
        pins()
    except BaseException as error:
        failure = error; report['error'] = dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
    finally:
        if phase == 'prepare' and cache:
            path = folder/'shared_serving_cache.pt'; report['shared_cache'] = dict(path=str(path),sha256=None)
            try:
                counts['cache_write_attempts'] += 1
                with path.open('xb') as f:
                    torch.save(_own(cache),f); f.flush(); os.fsync(f.fileno())
                counts['cache_writes'] += 1; report['raw_cache_write_complete'] = True
                report['shared_cache']['sha256'] = _sha(path)
            except BaseException as error:
                failure = failure or error; report['cache_preservation_error']=repr(error)
            report['cache_preservation'] = dict(exists=path.exists(),bytes=path.stat().st_size if path.exists() else 0)
        try:
            pins(); guard()
        except BaseException as error:
            failure = failure or error; report['final_boundary_error']=repr(error)
        report.update(seconds=time.monotonic()-started, peak_allocated_bytes=torch.cuda.max_memory_allocated() if cuda_ready else 0,
                      peak_reserved_bytes=torch.cuda.max_memory_reserved() if cuda_ready else 0, passed=failure is None)
        with (folder/'phase_report.json').open('x') as f:
            json.dump(_observed(report),f,indent=2,allow_nan=False); f.write('\n')
    if failure is not None:
        raise RuntimeError('Terminal canonical student phase failure; evidence retained') from failure
    return report
