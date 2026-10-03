"""Fixed CE plus P-derived kernel commutation geometry validation pilots.

Existing raw H, map, source Phi, teacher and native factors are immutable.
Only the native assignment is learned; no kernel-regression target or free
feature is introduced. Resident source assets are qualified for citations only.
"""
import json
from pathlib import Path

import numpy as np
import torch

from src.citation_factor_geometry import _pins, _sha, _source
from src.io import array_digest, save_json
from src.nystrom_ce import _cache_identity, _metadata_path, _validate_phi
from src.shared_features import _load_state, _map_identity, _tensor_identity, _validate_map_state

MODE = 'normalized_kernel_commutation_v1'
OBJECTIVE = 'teacher_CE_over_CE0_plus_kernel_gap_over_G0'


def _inputs(packet, B, h, stop):
    from src.nystrom_ce import NystromMap
    if B['dataset'] not in ('cora', 'citeseer') or len(h) > 4000 or h.shape[1] > 4000:
        raise ValueError('This resident-source pilot is qualified only for small citation datasets')
    family = Path(B['family_root'])
    map_path, phi_path = family / 'nystrom_map_schema3.pt', family / 'nystrom_phi_schema3.npy'
    roles = dict(H=str(family / 'propagated_H.pt'), map=str(map_path),
                 Phi=str(phi_path), Phi_metadata=str(_metadata_path(phi_path)))
    if any(p not in packet['readonly_files_sha256'] or not Path(p).is_file() for p in roles.values()):
        raise ValueError('Existing pinned raw H/map/Phi/mandatory sidecar are required')
    anchors, mapping = _validate_map_state(_load_state(map_path, 'shared_nystrom_map'),
                                          _map_identity(h, 3000, 0, 'relu'))
    phi = np.load(phi_path, mmap_mode='r')
    asset_bytes = h.numel() * 8 + anchors.numel() * 8 + mapping.numel() * 8 + phi.nbytes
    if asset_bytes > 512 * 1024**2 or phi.dtype != np.float64:
        raise ValueError('Fixed resident source assets exceed the frozen 512MiB bound or FP64 schema')
    feature_map = NystromMap(anchors.to(h.device), mapping.to(h.device), 'relu')
    identity = _cache_identity(h, feature_map, tuple(phi.shape), 2048, stop)
    phi_digest = _validate_phi(h, feature_map, phi, identity, 2048, stop)
    inputs = dict(physical_H=h.detach().double(),
                  source_phi=torch.from_numpy(np.array(phi)).to(device=h.device, dtype=torch.float64),
                  anchors=feature_map.anchors.detach(), mapping=feature_map.mapping.detach())
    return inputs, roles, phi_digest, asset_bytes


def _prepare(packet, B, z, q, assignment, h, options, digest, stop):
    from src.kernel_commutation_moments import POLICY
    folder = Path(B['commutation_input_folder'])
    if folder.exists():
        raise ValueError('Kernel commutation preparation exists; preserve the first execution')
    inputs, roles, phi_digest, asset_bytes = _inputs(packet, B, h, stop)
    native_ref = B['native_P0_factors']
    _pins({native_ref['path']: native_ref['sha256']})
    native = torch.load(native_ref['path'], map_location='cpu', weights_only=False)
    if (native['data_digest'] != digest or native['factor_seed'] != B['condensation_seed']
            or native['mixing'] != .05 or bool(native['u'].ne(0).any())):
        raise ValueError('Saved native zeroU/GaussianV factors differ from the original source')
    context = dict(schema=1, policy=POLICY,
        source_refs=dict(data_digest=digest, factor_seed=options['factor_seed'], rank=options['assignment_rank'],
                         nodes=len(z), cells=int(assignment.max()) + 1, physical_dimensions=h.shape[1], classes=q.shape[1],
                         mixing=.05, chunk_size=options['chunk_size'], device=str(z.device),
                         current_source=packet['source'], files_sha256=packet['readonly_files_sha256'],
                         asset_paths=roles),
        input_descriptors={k: _tensor_identity(v) for k, v in inputs.items()},
        native_parameter_digests=[array_digest(native[k].numpy()) for k in ('u', 'v')])
    folder.mkdir(parents=True)
    path = folder / 'commutation_context.json'
    save_json(context, path)
    report = dict(schema=1, budget=B['budget'], source=packet['source'],
                  context=dict(path=str(path), sha256=_sha(path)), source_phi_digest=phi_digest,
                  fixed_source_asset_bytes=asset_bytes, native_factor_generation_calls=0,
                  kernel_Gram_refits=0, phi_rebuilds=0, new_condensations=0, P_updates=0,
                  head_fits=0, student_fits=0, validation_evaluations=0, test_evaluations=0)
    _pins(packet['readonly_files_sha256'])
    save_json(report, folder / 'prepare_report.json')
    return report


def _run(protocol_path, budget, phase, mode, student_seeds, stop):
    from src.soft_ce_partition import optimize_ce_assignment
    from src.evaluation import fit_gcn_diagnostic
    from src.student_routes import replay_routes
    from src.sweep_utils import representative
    packet, B, graph, train, validation, h, z, q, assignment, transform, options, digest = _source(protocol_path, budget, stop)
    if any(k.startswith(('graph_assignment_kl_', 'kernel_commutation_')) for k in options):
        raise ValueError('Original native baseline cannot carry another auxiliary objective')
    if packet['kind'] != 'fixed_normalized_kernel_commutation_pilot_v1' or B['budget'] != budget:
        raise ValueError('Wrong fixed kernel-commutation protocol')
    if phase == 'prepare':
        if mode is not None or student_seeds:
            raise ValueError('Preparation cannot optimize or fit students')
        return _prepare(packet, B, z, q, assignment, h, options, digest, stop)
    if phase not in ('qualify', 'science') or mode not in (MODE, 'native_gaussian'):
        raise ValueError('Unknown kernel-commutation phase or arm')
    seeds = list(student_seeds)
    if seeds != ([] if phase == 'qualify' else B['student_seeds']):
        raise ValueError('Student cohort differs from the prospectively frozen phase')
    folder = Path(B['arm_folders'][mode])
    context = dict(schema=1, source=packet['source'], budget=budget, mode=mode,
                   baseline=B['baseline'], condensation_seed=B['condensation_seed'], native_data_digest=digest,
                   objective='original_teacher_CE' if mode == 'native_gaussian' else OBJECTIVE,
                   commutation=None if mode == 'native_gaussian' else B['commutation_context'])
    path = folder / 'context.json'
    if path.exists():
        if phase == 'qualify' or mode == 'native_gaussian':
            raise ValueError('Fresh qualification/native-control invocation already exists; preserve it')
        if json.loads(path.read_text()) != context:
            raise ValueError('Trajectory namespace or source context changed')
    elif folder.exists():
        raise ValueError('Partial output folder lacks its immutable context')
    else:
        folder.mkdir(parents=True)
        save_json(context, path)
    if phase == 'science' and ((folder / 'validation').exists() or (folder / 'science_report.json').exists()):
        raise ValueError('Fresh science already has a student cache or terminal report; never refit or miscount it')
    new_updates, fit_calls = 0, 0
    if mode == MODE:
        reference = B['commutation_context']
        _pins({reference['path']: reference['sha256']})
        frozen_context = json.loads(Path(reference['path']).read_text())
        inputs, _, _, _ = _inputs(packet, B, h, stop)
        options.update(kernel_commutation_mode=MODE, kernel_commutation_context=frozen_context,
                       kernel_commutation_inputs=inputs)
        resume, manifest = folder / 'resume.pt', folder / 'checkpoint_manifest.json'
        state = None
        if resume.exists():
            _pins(json.loads(manifest.read_text()))
            state = torch.load(resume, map_location='cpu', weights_only=False)
        if phase == 'qualify' and state is not None:
            raise ValueError('Qualification requires one fresh native P update')
        if phase == 'science':
            qualification = json.loads((folder / 'qualification.json').read_text())
            if not qualification['passed'] or state is None or state['step'] != 1:
                raise ValueError('Science requires its passed one-update prefix')
        steps = 1 if phase == 'qualify' else 25
        initial_step = 0 if state is None else state['step']
        optimize_ce_assignment(z, q, assignment, steps=steps, folder=folder, checkpoint_steps=(0, steps),
                               resume_state=state, stop=stop, **options)
        state = torch.load(resume, map_location='cpu', weights_only=False)
        new_updates, fit_calls = steps - initial_step, steps - initial_step + 1
        paths = [resume, folder / 'optimization.csv', *sorted((folder / 'checkpoints').glob('step_*.pt'))]
        save_json({str(p): _sha(p) for p in paths}, manifest)
        if state['step'] != steps or state['kernel_commutation_context'] != frozen_context:
            raise ValueError('Executed trajectory differs from its frozen phase')
        if phase == 'qualify':
            p0 = state['snapshots'][0]
            original = torch.load(B['baseline_snapshot0'], map_location='cpu', weights_only=False)
            native = torch.load(B['native_P0_factors']['path'], map_location='cpu', weights_only=False)
            expected_G0 = B['diagnostic_G0']
            discrepancy = abs(p0['kernel_commutation_G0'] - expected_G0)
            passed = (torch.equal(p0['moments'], original['moments'])
                and torch.equal(p0['theta'], original['theta']) and p0['J_exact']
                and p0['teacher_ce'] == original['teacher_ce'] == p0['kernel_commutation_CE0']
                and all(torch.equal(a, native[k]) for a, k in zip(p0['kernel_commutation_parameters'], ('u', 'v'), strict=True))
                and p0['kernel_commutation_G0'] > 0 and discrepancy <= 1e-12
                and p0['objective'] == 2.0 and state['scale'] == p0['kernel_commutation_CE0'])
            report = dict(schema=1, passed=bool(passed), budget=budget, mode=mode, source=packet['source'],
                P0_moments_and_head_bit_exact_original=bool(passed), commutation_context=frozen_context,
                diagnostic_G0=expected_G0, G0_discrepancy=discrepancy, actual_new_P_updates=1,
                classifier_fit_calls=2, new_condensations=1, physical_student_fits=0, test_evaluations=0)
            save_json(report, folder / 'qualification.json')
            if not passed:
                raise ValueError('Original native P0/head/factors and fixed CE0/G0 qualification failed')
            return report
        checkpoints = folder / 'checkpoints'
        evaluation_steps = (25,)
    else:
        if phase != 'science':
            raise ValueError('Native anchor reuses existing P0/P25')
        checkpoints, evaluation_steps = Path(B['baseline_folder']) / 'checkpoints', (0, 25)
    records = []
    settings = {k: v for k, v in B['recipe'].items() if k != 'input_scale'}
    for step in evaluation_steps:
        saved = torch.load(checkpoints / f'step_{step:06d}.pt', map_location='cuda', weights_only=False)
        x, y, mass = representative(saved['moments'], transform, z.shape[1], 'cuda')
        for seed in seeds:
            evaluation = folder / 'validation' / f"step_{step}_{B['recipe_id']}"
            result = fit_gcn_diagnostic(x, y, torch.full_like(mass, 1 / len(mass)), graph, q,
                dict(train=train, val=validation[1]), seed, folder=evaluation, stop=stop, **settings)
            routes = replay_routes(evaluation / f'seed_{seed}_selected.pt', graph, h.float(), dict(val=validation[1]),
                settings, evaluation / f'seed_{seed}_validation_routes_v1.json', seed=seed, stop=stop)
            if (any('test_' in key for key in result | routes) or routes['gcn_val_acc'] != result['val_acc']
                    or routes['epoch'] != result['epoch']):
                raise ValueError('Validation-only selected-weight routes do not match')
            records.append(dict(step=step, **result, SGC_MLP_sameweights_val=routes['mlp_val_acc']))
    report = dict(schema=1, budget=budget, mode=mode, phase=phase, source=packet['source'], records=records,
        trajectory_P_updates=25, actual_new_P_updates=new_updates, classifier_fit_calls=fit_calls,
        new_physical_student_fits=len(records), same_selected_weight_validation_routes=2 * len(records),
        candidate_P0_student_fits_reused_from_bit_exact_native_control=mode == MODE,
        test_evaluations=0, free_features=False, objective=OBJECTIVE if mode == MODE else 'original_teacher_CE')
    save_json(report, folder / 'science_report.json')
    return report


def run(protocol_path, protocol_sha256, budget, phase, mode=None, student_seeds=(), stop=lambda: False):
    if _sha(protocol_path) != protocol_sha256:
        raise ValueError('Frozen kernel-commutation protocol bytes changed')
    try:
        return _run(protocol_path, budget, phase, mode, student_seeds, stop)
    except InterruptedError as error:
        raise RuntimeError('Terminal bounded kernel-commutation interruption; no automatic retry') from error
