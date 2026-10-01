"""Bounded, train-only CE teachers on the already frozen NTK feature cache.

This changes Q and its teacher-aware initializer jointly. Historical ReLU
teachers remain frozen; their solver has no post-fit convergence certificate.
"""
import hashlib
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from src.io import _fingerprint, save_json, save_state
from src.nystrom_ce import _content_digest, blocks, fit_streaming_teacher
from src.relu_ntk import KIND, cache_paths, prepare_features
from src.relu_ntk import source_digest as map_source_digest

GAMMAS = (1e-5, 1e-4, 1e-3, .01)


def source_digest():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def file_digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def recipe():
    solver = Path(__file__).with_name('nystrom_ce.py')
    return dict(function='src.nystrom_ce.fit_streaming_teacher', function_file_sha256=file_digest(solver),
                helper_sha256=source_digest(), torch_version=str(torch.__version__),
                dtype='float64', bias=False, chunk=2048, max_iter=1000,
                gradient_max_required=1e-5, resident_training=True, initial_weight='zeros',
                objective='mean_train_CE + gamma/n_train * sum(weight**2)',
                optimizer='torch.optim.LBFGS', line_search_fn='strong_wolfe',
                optimizer_controls='unchanged PyTorch defaults', selection='first_strict_validation_accuracy_max')


def validate_kernel(kernel):
    if not isinstance(kernel, str) or kernel not in ('relu', KIND):
        raise ValueError('teacher_kernel must be relu or the frozen NTK kind')
    return kernel


def _inputs(h, train_mask, train_labels, val_mask, val_labels):
    n = len(h)
    if (train_mask.shape != (n,) or val_mask.shape != (n,) or train_mask.dtype != torch.bool
            or val_mask.dtype != torch.bool or not bool(train_mask.any()) or not bool(val_mask.any())
            or bool((train_mask & val_mask).any()) or train_labels.shape != (n,)
            or train_labels.dtype != torch.long or val_labels.shape != (int(val_mask.sum()),)
            or val_labels.dtype != torch.long or bool((train_labels < 0).any())
            or bool((train_labels[~train_mask] != 0).any())):
        raise ValueError('Require training-only labels and disjoint nonempty train/validation masks')
    classes = int(train_labels[train_mask].max()) + 1
    if classes < 2 or bool((val_labels < 0).any()) or bool((val_labels >= classes).any()):
        raise ValueError('Class vocabulary must come exclusively from training labels')
    return dict(train_mask=_content_digest(train_mask), train_labels=_content_digest(train_labels),
                validation_mask=_content_digest(val_mask), validation_labels=_content_digest(val_labels),
                classes=classes, train_nodes=int(train_mask.sum()), validation_nodes=int(val_mask.sum()))


def teacher_inputs(graph, train_mask, val_mask):
    """No test labels reach this helper or the unchanged solver vocabulary."""
    labels = torch.zeros_like(graph['y'])
    labels[train_mask] = graph['y'][train_mask]
    return labels, graph['y'][val_mask].clone()


def _load(path):
    try:
        saved = torch.load(path, map_location='cpu', weights_only=True)
    except Exception as error:
        raise ValueError('Invalid teacher cache; preserve it and use a new namespace') from error
    if not isinstance(saved, dict):
        raise ValueError('Invalid teacher cache envelope')
    return saved


def _assets(h, base_config, source_root):
    from src.citation_search import fixed_propagated_features
    source_root = Path(source_root).resolve()
    config_path = source_root / 'config.json'
    if not config_path.exists() or json.loads(config_path.read_text()) != base_config:
        raise ValueError('Teacher source configuration differs')
    if not (source_root / 'propagated_H.pt').exists():
        raise ValueError('Teacher requires existing frozen source H')
    frozen = fixed_propagated_features(h, base_config, source_root)
    feature_map, phi = prepare_features(frozen, source_root, require_existing=True)
    if phi.dtype != np.dtype("float64"):
        raise ValueError("NTK teacher features must be float64, including streaming fallback")
    ntk_map, ntk_phi = cache_paths(source_root)
    paths = [source_root/'propagated_H.pt', source_root/'nystrom_map_schema3.pt', ntk_map, ntk_phi, ntk_phi.with_suffix('.meta.json')]
    assets = {p.name: file_digest(p) for p in paths}
    identity = dict(h_digest=_content_digest(frozen, canonical_double=True),
                    anchor_digest=_content_digest(feature_map.anchors),
                    mapping_digest=_content_digest(feature_map.mapping), phi_digest=_content_digest(phi),
                    shape=list(phi.shape), scale=feature_map.scale)
    return frozen, phi, source_root, assets, identity


def context(h, base_config, source_root, train_mask, train_labels, val_mask, val_labels):
    signatures = _inputs(h, train_mask, train_labels, val_mask, val_labels)
    frozen, phi, source_root, assets, identity = _assets(h, base_config, source_root)
    ctx = dict(schema=1, kernel=KIND, helper_sha256=source_digest(), ntk_helper_sha256=map_source_digest(),
               torch_version=str(torch.__version__), gamma_grid=list(GAMMAS), recipe=recipe(),
               source_root=str(source_root), source_config_sha256=file_digest(source_root/'config.json'),
               assets=assets, numerical_inputs=identity, **signatures)
    config = dict(base_config, teacher_kernel=KIND, teacher_context=ctx)
    return frozen, phi, config


def _verify_root(root, config, require_complete_assets=True):
    root = Path(root)
    if root.name != _fingerprint(config):
        raise ValueError('Teacher root identity differs from its effective configuration')
    if (root/'config.json').exists() and json.loads((root/'config.json').read_text()) != config:
        raise ValueError('Teacher root configuration differs')
    ctx = config['teacher_context']
    binding = dict(source_root=ctx['source_root'], assets=ctx['assets'])
    if (root/'source_binding.json').exists() and json.loads((root/'source_binding.json').read_text()) != binding:
        raise ValueError('Teacher source binding metadata differs')
    if require_complete_assets and not (root/'source_binding.json').exists():
        raise ValueError('Teacher source binding metadata is missing')
    for name, digest in ctx['assets'].items():
        source = Path(ctx['source_root'])/name
        if not source.exists() or file_digest(source) != digest:
            raise ValueError('Frozen source teacher asset differs')
        target = root/name
        if target.exists() and file_digest(target) != digest:
            raise ValueError('Copied teacher asset differs; refusing overwrite')
        if require_complete_assets and not target.exists():
            raise ValueError('Teacher root lacks a verified copied feature asset')


def _install(root, config, inputs, stop):
    root = Path(root)
    _verify_root(root, config, require_complete_assets=False)
    if (root/'teacher_inputs.pt').exists():
        saved = _load(root/'teacher_inputs.pt')
        for key, tensor in inputs.items():
            if not torch.is_tensor(saved.get(key)) or not torch.equal(saved[key], tensor.detach().cpu()):
                raise ValueError('Teacher training/validation input cache differs')
    elif (root/'teacher.pt').exists() or any((root/'teacher_gammas').glob('gamma_*.pt')):
        raise ValueError('Cached teacher lacks verifiable train/validation inputs')
    if stop():
        raise InterruptedError('Teacher setup interrupted')
    root.mkdir(parents=True, exist_ok=True)
    for name, digest in config['teacher_context']['assets'].items():
        target = root/name
        if not target.exists():
            if stop():
                raise InterruptedError('Teacher asset copy interrupted')
            temporary = target.with_suffix(target.suffix+'.copy.tmp')
            shutil.copyfile(Path(config['teacher_context']['source_root'])/name, temporary)
            if file_digest(temporary) != digest:
                raise ValueError('Teacher source changed during asset copy')
            temporary.replace(target)
    if not (root/'teacher_inputs.pt').exists():
        save_state(inputs, root/'teacher_inputs.pt')
    if not (root/'config.json').exists():
        save_json(config, root/'config.json')
    if not (root/'source_binding.json').exists():
        save_json(dict(source_root=config['teacher_context']['source_root'], assets=config['teacher_context']['assets']), root/'source_binding.json')
    (root/'teacher_gammas').mkdir(exist_ok=True)


def certificate(phi, labels, mask, weight, gamma, stop=lambda: False):
    """Independently check no-bias CE objective, gradient and all-source logits."""
    n = int(mask.sum()); gradient = torch.zeros_like(weight); objective = weight.new_zeros(())
    predictions = []
    for start, x in blocks(phi, weight.device, 2048):
        if stop():
            raise InterruptedError('Teacher certificate interrupted')
        scores = x@weight; predictions.append(scores.detach().cpu())
        selected = mask[start:start+len(x)]
        if bool(selected.any()):
            y = labels[start:start+len(x)][selected]; chosen = x[selected]
            log_probability = scores[selected].log_softmax(1)
            objective -= log_probability[torch.arange(len(y),device=y.device), y].sum()/n
            error = log_probability.exp(); error[torch.arange(len(y),device=y.device), y] -= 1
            gradient += chosen.T@error/n
    objective += gamma/n*weight.square().sum(); gradient += 2*gamma/n*weight
    return torch.cat(predictions), float(objective), float(gradient.abs().max())


def gamma_path(root, gamma):
    return Path(root)/'teacher_gammas'/f'gamma_{_fingerprint(gamma)}.pt'


def _validate_gamma(state, config, phi, labels, mask, val_mask, val_labels, gamma, stop):
    ctx = config['teacher_context']; shape = (phi.shape[1], ctx['classes'])
    if type(state.get('schema')) is not int or state.get('schema') != 1 or state.get('kind') != 'ntk_teacher_gamma' or state.get('context') != ctx or state.get('gamma') != gamma:
        raise ValueError('Teacher gamma source/input/grid/solver context differs')
    weight, logits = state.get('weight'), state.get('logits')
    if (not torch.is_tensor(weight) or tuple(weight.shape) != shape or weight.dtype != torch.double
            or not torch.is_tensor(logits) or tuple(logits.shape) != (len(phi),shape[1]) or logits.dtype != torch.double
            or not bool(torch.isfinite(weight).all()) or not bool(torch.isfinite(logits).all())
            or state.get('weight_digest') != _content_digest(weight) or state.get('logits_digest') != _content_digest(logits)):
        raise ValueError('Teacher gamma weight/logits content differs')
    for key in ('objective', 'gradient_max', 'val', 'val_ce', 'seconds'):
        value = state.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError('Teacher gamma diagnostics must be finite numerical scalars')
    predicted, objective, residual = certificate(phi, labels, mask, weight.to(labels.device), gamma, stop)
    if (not torch.allclose(predicted, logits, atol=1e-10, rtol=1e-10) or not math.isfinite(residual) or residual > 1e-5
            or not math.isclose(state.get('objective',float('nan')),objective,abs_tol=1e-10,rel_tol=1e-9)
            or not math.isclose(state.get('gradient_max',float('nan')),residual,abs_tol=1e-10,rel_tol=1e-6)):
        raise ValueError('Teacher gamma convergence/objective/prediction certificate failed')
    expected_val = float((logits[val_mask.cpu()].argmax(1)==val_labels.cpu()).double().mean())
    expected_ce = float(torch.nn.functional.cross_entropy(logits[val_mask.cpu()],val_labels.cpu()))
    route = state.get('route',{})
    if (state.get('val') != expected_val or state.get('val_ce') != expected_ce
            or not isinstance(route,dict) or route.get('requested') != 'resident' or route.get('actual') not in ('resident','streaming')):
        raise ValueError('Teacher validation/solver-route metadata differs')
    return state


def _cached_grid(root, config, phi, labels, mask, val_mask, val_labels, stop):
    states = {}
    folder = Path(root)/'teacher_gammas'
    allowed = {gamma_path(root,g).name for g in GAMMAS}
    if any(p.name not in allowed for p in folder.glob('gamma_*.pt') if not p.name.endswith('.tmp.pt')):
        raise ValueError('Teacher cache contains a gamma outside the fixed grid')
    for gamma in GAMMAS:
        path = gamma_path(root,gamma)
        if path.exists():
            states[gamma] = _validate_gamma(_load(path),config,phi,labels,mask,val_mask,val_labels,gamma,stop)
    return states


def _selected(root, config, states, *, create=False, stop=lambda: False):
    if set(states) != set(GAMMAS):
        if (Path(root)/'teacher.pt').exists():
            raise ValueError('Selected teacher cannot outlive an incomplete gamma grid')
        return None
    selected = GAMMAS[0]
    for gamma in GAMMAS[1:]:
        if states[gamma]['val'] > states[selected]['val']:
            selected = gamma
    chosen = states[selected]
    expected = dict(schema=1,kind='ntk_teacher_selected',context=config['teacher_context'],gamma=selected,
                    grid=[dict(gamma=g,cache_digest=file_digest(gamma_path(root,g)),val=states[g]['val'],val_ce=states[g]['val_ce']) for g in GAMMAS],
                    weight_digest=chosen['weight_digest'],logits_digest=chosen['logits_digest'])
    path = Path(root)/'teacher.pt'
    if path.exists():
        saved = _load(path)
        if (type(saved.get('schema')) is not int or any(saved.get(k)!=v for k,v in expected.items()) or not torch.is_tensor(saved.get('weight'))
                or not torch.is_tensor(saved.get('logits')) or not torch.equal(saved['weight'],chosen['weight'])
                or not torch.equal(saved['logits'],chosen['logits'])):
            raise ValueError('Selected teacher full-grid/context/content differs')
        return saved
    if create is None:
        return None
    if not create:
        raise ValueError('Complete gamma grid has no atomically selected teacher')
    if stop():
        raise InterruptedError('Teacher selection interrupted')
    saved = dict(expected,weight=chosen['weight'],logits=chosen['logits'])
    save_state(saved,path)
    save_json(dict(gamma=selected,val=100*chosen['val'],val_ce=chosen['val_ce'],selection='first_strict_validation_accuracy_max'),Path(root)/'teacher_selected.json')
    return saved


def teacher_root(output_dir, base_config, config):
    return Path(output_dir)/base_config['dataset']/f"ratio_{base_config['ratio']}"/_fingerprint(config)


def validate_root(root, *, h=None, train_mask=None, train_labels=None, val_mask=None, val_labels=None,
                  device='cpu', require_selected=True, stop=lambda: False):
    """Read-only cached check; callers repeat with current graph inputs before writes."""
    root = Path(root)
    if not (root/'config.json').exists():
        raise ValueError('NTK teacher gamma preparation is incomplete')
    config = json.loads((root/'config.json').read_text())
    if config.get('teacher_kernel') != KIND or not isinstance(config.get('teacher_context'),dict):
        raise ValueError('NTK teacher root context is missing')
    ctx = config['teacher_context']
    if (type(ctx.get('schema')) is not int or ctx.get('schema') != 1 or ctx.get('kernel') != KIND
            or ctx.get('helper_sha256') != source_digest() or ctx.get('ntk_helper_sha256') != map_source_digest()
            or ctx.get('torch_version') != str(torch.__version__) or ctx.get('gamma_grid') != list(GAMMAS)
            or ctx.get('recipe') != recipe() or not isinstance(ctx.get('source_root'),str) or not ctx['source_root']):
        raise ValueError('Teacher root helper/schema/grid/solver context differs')
    base = {k:v for k,v in config.items() if k not in ('teacher_kernel','teacher_context')}
    inputs = _load(root/'teacher_inputs.pt')
    if any(not torch.is_tensor(inputs.get(k)) for k in ('train_mask','train_labels','val_mask','val_labels')):
        raise ValueError('Teacher root lacks valid training/validation tensors')
    if h is None:
        h = _load(root/'propagated_H.pt')['h'].to(device)
    supplied = (train_mask,train_labels,val_mask,val_labels)
    if any(value is not None for value in supplied) and not all(value is not None for value in supplied):
        raise ValueError('Current teacher training/validation inputs must be supplied together')
    if train_mask is None:
        train_mask,train_labels,val_mask,val_labels = (inputs[k].to(device) for k in ('train_mask','train_labels','val_mask','val_labels'))
    _,phi,expected = context(h,base,config['teacher_context']['source_root'],train_mask,train_labels,val_mask,val_labels)
    if config != expected:
        raise ValueError('Teacher current source/training/validation/solver context differs')
    _verify_root(root,config)
    for key,value in zip(('train_mask','train_labels','val_mask','val_labels'),(train_mask,train_labels,val_mask,val_labels),strict=True):
        if not torch.is_tensor(inputs.get(key)) or not torch.equal(inputs[key],value.detach().cpu()):
            raise ValueError('Teacher saved inputs differ from current graph')
    states = _cached_grid(root,config,phi,train_labels,train_mask,val_mask,val_labels,stop)
    selected = _selected(root,config,states,stop=stop)
    if require_selected and selected is None:
        raise ValueError('NTK condensation requires all four converged teacher gammas')
    return selected,config,phi


def initialize(base_config, h, train_mask, train_labels, val_mask, val_labels, output_dir, source_root, stop=lambda: False):
    frozen,phi,config = context(h,base_config,source_root,train_mask,train_labels,val_mask,val_labels)
    root = teacher_root(output_dir,base_config,config)
    inputs = dict(train_mask=train_mask,train_labels=train_labels,val_mask=val_mask,val_labels=val_labels)
    # Reject every existing mismatched gamma/final cache before asset/setup writes.
    _verify_root(root,config,require_complete_assets=False)
    states = _cached_grid(root,config,phi,train_labels,train_mask,val_mask,val_labels,stop)
    _selected(root,config,states,create=None,stop=stop)
    _install(root,config,inputs,stop)
    return root,frozen,phi,config,states


def prepare_gamma(base_config, h, train_mask, train_labels, val_mask, val_labels, output_dir, source_root,
                  gamma, stop=lambda: False):
    if isinstance(gamma,bool) or gamma not in GAMMAS:
        raise ValueError('One gamma from the complete frozen grid is required')
    root,h,phi,config,states = initialize(base_config,h,train_mask,train_labels,val_mask,val_labels,output_dir,source_root,stop)
    cached = gamma in states
    if not cached:
        started = time.monotonic()
        logits,weight = fit_streaming_teacher(phi,train_labels,train_mask,gamma,chunk=2048,max_iter=1000,
                                              resident_training=True,stop=stop)
        predicted,objective,residual = certificate(phi,train_labels,train_mask,weight,gamma,stop)
        if (not math.isfinite(residual) or residual>1e-5 or not math.isfinite(objective)
                or not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(weight).all())
                or not torch.allclose(predicted,logits.detach().cpu(),atol=1e-10,rtol=1e-10)):
            raise RuntimeError('Teacher result is nonconverged or has invalid predictions')
        logits,weight = logits.detach().cpu(),weight.detach().cpu()
        state = dict(schema=1,kind='ntk_teacher_gamma',context=config['teacher_context'],gamma=gamma,
                     logits=logits,weight=weight,logits_digest=_content_digest(logits),weight_digest=_content_digest(weight),
                     objective=objective,gradient_max=residual,seconds=time.monotonic()-started,
                     route=dict(fit_streaming_teacher.last_route),
                     val=float((logits[val_mask.cpu()].argmax(1)==val_labels.cpu()).double().mean()),
                     val_ce=float(torch.nn.functional.cross_entropy(logits[val_mask.cpu()],val_labels.cpu())))
        _validate_gamma(state,config,phi,train_labels,train_mask,val_mask,val_labels,gamma,stop)
        if stop():
            raise InterruptedError('Teacher gamma interrupted before atomic commit')
        save_state(state,gamma_path(root,gamma))
        states[gamma]=state
    selected = _selected(root,config,states,create=True,stop=stop)
    return dict(root=str(root),gamma=gamma,cached=cached,grid_complete=selected is not None,
                selected_gamma=None if selected is None else selected['gamma'],validation_only=True,
                train_gradient_max=states[gamma]['gradient_max'],route=states[gamma]['route'])


def prepare_gamma_job(dataset,ratio,output_dir,source_root,teacher_gamma,teacher_kernel=KIND,
                      citation_features='default',data_dir='data',device='cuda',require_existing_source_features=True,
                      teacher_recipe=None,stop=lambda: False):
    from src.citation_search import _legacy_teacher_config
    from src.data import BUDGET, _prepare_dataset
    if validate_kernel(teacher_kernel) != KIND or require_existing_source_features is not True:
        raise ValueError('Gamma preparation requires the existing frozen NTK source')
    if dataset not in ('cora','citeseer') or (dataset,ratio) not in BUDGET:
        raise ValueError('Use configured citation budgets')
    if isinstance(teacher_gamma,bool) or teacher_gamma not in GAMMAS:
        raise ValueError('One fixed-grid gamma required')
    if teacher_recipe is not None and teacher_recipe != recipe():
        raise ValueError('Teacher solver recipe controls are fixed; freeze current effective recipe')
    if stop():
        raise InterruptedError('Teacher gamma preparation interrupted')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    graph,train,validation,testing,h = _prepare_dataset(dataset,data_dir,device,citation_features)
    base = _legacy_teacher_config(dataset,ratio,graph,train,validation,testing,citation_features)
    labels,val_labels = teacher_inputs(graph,train,validation[1])
    return prepare_gamma(base,h,train,labels,validation[1],val_labels,output_dir,source_root,teacher_gamma,stop)
