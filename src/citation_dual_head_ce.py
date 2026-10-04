"""Original-source, fixed two-stationary-head CE citation validation pilots.

Both heads share one native NODE assignment and its original base moments.
Existing H, RMS coordinates, Q, literal ReLU map and Phi remain immutable.
"""
import json
import math
from pathlib import Path

import numpy as np
import torch

from src.citation_factor_geometry import _pins, _sha, _source
from src.io import save_json
from src.shared_features import _tensor_identity

MODE = 'normalized_dual_linear_Nystrom_CE_v1'
OBJECTIVE = 'CE_linear_over_CE_linear0_plus_CE_Nystrom_over_CE_Nystrom0'
KIND = 'fixed_normalized_dual_linear_Nystrom_CE_pilot_v1'


def _transform_descriptor(transform, dimension):
    if (transform.kind != 'rms' or transform.matrix is not None
            or not torch.is_tensor(transform.scale) or transform.scale.numel() != 1
            or transform.scale.dtype != torch.float64 or not bool(torch.isfinite(transform.scale))
            or float(transform.scale) <= 0):
        raise ValueError('Dual-head source requires its original positive scalar RMS transform')
    for value in (transform.center, transform.output_center):
        if (not torch.is_tensor(value) or value.shape != (dimension,)
                or value.dtype != torch.float64 or not bool(torch.isfinite(value).all())):
            raise ValueError('Original RMS transform shape/dtype/finite coordinates changed')
    return dict(kind='rms', matrix=None, center=_tensor_identity(transform.center),
                output_center=_tensor_identity(transform.output_center),
                scale=_tensor_identity(transform.scale), eps=transform.eps)


def _resident_inputs(packet, B, h, transform, stop):
    from src.nystrom_ce import NystromMap, _cache_identity, _metadata_path, _validate_phi
    from src.shared_features import _load_state, _map_identity, _validate_map_state

    family = Path(B['family_root'])
    map_path, phi_path = family / 'nystrom_map_schema3.pt', family / 'nystrom_phi_schema3.npy'
    sidecar = _metadata_path(phi_path)
    roles = dict(H=str(family / 'propagated_H.pt'), map=str(map_path),
                 Phi=str(phi_path), Phi_metadata=str(sidecar))
    if any(path not in packet['readonly_files_sha256'] or not Path(path).is_file()
           for path in roles.values()):
        raise ValueError('Require existing pinned H, literal ReLU map, Phi and mandatory sidecar')
    # Reject a missing sidecar before the validator's legacy verification/write path.
    identity = _map_identity(h, 3000, 0, 'relu')
    anchors, mapping = _validate_map_state(_load_state(map_path, 'shared_nystrom_map'), identity)
    phi = np.load(phi_path, mmap_mode='r')
    if phi.dtype != np.float64 or phi.ndim != 2 or phi.shape != (len(h), len(anchors)):
        raise ValueError('Original Phi has incompatible FP64 dimensions')
    asset_bytes = 8 * (h.numel() + anchors.numel() + mapping.numel()) + phi.nbytes
    if asset_bytes > 512 * 1024**2:
        raise ValueError('Frozen citation H/map/Phi residency exceeds 512MiB')
    feature_map = NystromMap(anchors.to(h.device), mapping.to(h.device), 'relu')
    cache_identity = _cache_identity(h, feature_map, tuple(phi.shape), 2048, stop)
    phi_digest = _validate_phi(h, feature_map, phi, cache_identity, 2048, stop)
    phi_identity = json.loads(sidecar.read_text())
    if phi_identity != dict(**cache_identity, phi_digest=phi_digest):
        raise ValueError('Original Phi sidecar identity differs')
    descriptors = dict(H=_tensor_identity(h), transform=_transform_descriptor(transform, h.shape[1]),
                       anchors=_tensor_identity(feature_map.anchors),
                       mapping=_tensor_identity(feature_map.mapping), Phi_identity=phi_identity)
    return feature_map, phi, roles, descriptors, asset_bytes


def _native_parameters(packet, B, z, assignment, options, digest):
    from src.io import array_digest

    ref = B['native_P0_factors']
    if packet['readonly_files_sha256'].get(ref['path']) != ref['sha256']:
        raise ValueError('Native P0 factors lack the frozen original-source pin')
    native = torch.load(ref['path'], map_location='cpu', weights_only=False)
    parameters = [native['u'], native['v']]
    cells, rank = int(assignment.max()) + 1, options['assignment_rank']
    if (native['data_digest'] != digest or native['factor_seed'] != B['condensation_seed']
            or native['mixing'] != .05 or any(v.dtype != torch.float32 for v in parameters)
            or parameters[0].shape != (len(z), rank) or parameters[1].shape != (cells, rank)
            or any(not bool(torch.isfinite(v).all()) for v in parameters)
            or bool(parameters[0].ne(0).any())):
        raise ValueError('Pinned native zero-U/Gaussian-V parameters differ from original source')
    return native, parameters, [array_digest(v.numpy()) for v in parameters]


def _work_delta(state, before):
    keys = {'moment_forward_calls', 'moment_backward_calls', 'P_updates',
            'linear_head_interfaces', 'Nystrom_head_interfaces',
            'linear_adjoint_solves', 'Nystrom_adjoint_solves', 'checkpoint_writes', 'resume_writes'}
    work = state['work']
    if (set(work) != keys or any(type(value) is not int or value < 0 for value in work.values())
            or (before and set(before) != keys)
            or any(type(value) is not int or value < 0 for value in before.values())):
        raise ValueError('Dual-head actual work counter schema differs')
    delta = {key: value - before.get(key, 0) for key, value in work.items()}
    if any(value < 0 for value in delta.values()):
        raise ValueError('Dual-head cumulative work counter moved backwards')
    return delta


def _source_geometry(original, native, h, z, assignment, transform, options, stop):
    from src.low_rank_assignment import logit_block

    u0, v0 = [native[key].to(device=z.device) for key in ('u', 'v')]
    numerator = z.new_zeros(len(v0), h.shape[1])
    with torch.no_grad():
        for begin in range(0, len(z), options['chunk_size']):
            if stop():
                raise InterruptedError('Dual-head P0 geometry qualification interrupted')
            end = begin + options['chunk_size']
            probability = logit_block(u0[begin:end], v0, assignment[begin:end], .05).double().softmax(1)
            numerator += probability.T @ h[begin:end].double() / len(z)
        moments = original['moments'].to(device=z.device)
        mass = moments[:, 0]
        if (moments.dtype != torch.float64 or moments.ndim != 2
                or moments.shape[0] != len(v0) or not bool(torch.isfinite(moments).all())
                or not bool((mass > 0).all())):
            raise ValueError('Pinned original P0 moments are outside the strict geometry domain')
        physical = transform.center + transform.output_center + transform.scale * (
            moments[:, 1:1 + z.shape[1]] / mass[:, None])
        error = float((physical - numerator / mass[:, None]).abs().max())
    return dict(affine_inverse_vs_direct_H_centroid_max_absolute=error,
                affine_inverse_vs_direct_H_centroid_passed=math.isfinite(error) and error <= 1e-12,
                source_geometry_checked_before_any_candidate_head_or_P_update=True)


def _P0_geometry(p0, original, native, z, q, transform, source_geometry):
    from src.sweep_utils import representative

    equal = lambda left, right: _tensor_identity(left) == _tensor_identity(right)
    initial = p0['initial_parameters']
    exact = (equal(p0['moments'], original['moments'])
             and equal(p0['theta_linear'], original['theta'])
             and p0['CE_linear'] == original['teacher_ce'] == p0['CE_linear0']
             and all(equal(value, native[key]) for value, key in zip(initial, ('u', 'v'), strict=True))
             and all(equal(value, native[key]) for value, key in zip(p0['parameters'], ('u', 'v'), strict=True)))
    actual = representative(p0['moments'].to(z.device), transform, z.shape[1], z.device)
    expected = representative(original['moments'].to(z.device), transform, z.shape[1], z.device)
    serving_exact = all(equal(a, b) for a, b in zip(actual[:2], expected[:2], strict=True))
    serving_exact = serving_exact and equal(torch.full_like(actual[2], 1 / len(actual[2])),
                                          torch.full_like(expected[2], 1 / len(expected[2])))
    work = p0['head_work']['Nystrom']
    stationary = (work['inner_converged'] is True and math.isfinite(work['inner_grad_max'])
                  and work['inner_grad_max'] <= 1e-7)
    scales = (all(math.isfinite(p0[key]) and p0[key] > 0 for key in ('CE_linear0', 'CE_Nystrom0'))
              and p0['CE_Nystrom'] == p0['CE_Nystrom0'] and p0['objective'] == 2.0)
    return dict(P0_moments_linear_head_factors_and_teacherCE_bit_exact_original=bool(exact),
                P0_student_X_Q_uniform_weights_bit_exact_original=bool(serving_exact),
                source_Q_descriptor=_tensor_identity(q),
                mapped_P0_head_stationary=bool(stationary), mapped_P0_head_work=work,
                both_CE0_positive_and_J0_equals2=bool(scales), **source_geometry)


def _run(protocol_path, budget, phase, mode, student_seeds, stop):
    from src.dual_head_ce import POLICY, optimize
    from src.evaluation import fit_gcn_diagnostic
    from src.research_loop import implementation_provenance
    from src.student_routes import replay_routes
    from src.sweep_utils import representative

    if phase not in ('qualify', 'science') or mode not in (MODE, 'native_gaussian'):
        raise ValueError('Dual-head pilot has only qualification/science and its original native control')
    # _source may use require-existing cache helpers; prove all original paths
    # exist and are pinned before it can enter an absent-file fallback branch.
    declaration_packet = json.loads(Path(protocol_path).read_text())
    declaration_budget = declaration_packet['budget_packets'][budget]
    family = Path(declaration_budget['family_root'])
    originals = [family / name for name in ('config.json', 'propagated_H.pt', 'teacher.pt')]
    originals += [family / f"inputs_{declaration_budget['condensation_seed']}.pt",
                  Path(declaration_budget['hard_assignment']),
                  Path(declaration_budget['baseline_folder']) / 'resume.pt',
                  Path(declaration_budget['baseline_snapshot0'])]
    if any(str(path) not in declaration_packet['readonly_files_sha256'] or not path.is_file()
           for path in originals):
        raise ValueError('Original source cache must exist and be pinned before loading')
    packet, B, graph, train, validation, h, z, q, assignment, transform, options, digest = _source(protocol_path, budget, stop)
    incompatible = ('node_factor_', 'assignment_kl_', 'uniform_cell_q_prior_', 'graph_assignment_kl_',
                    'kernel_commutation_', 'conditional_label_entropy_', 'soft_cell_mass_KL_', 'dual_head_')
    if (packet['kind'] != KIND or B['budget'] != budget or B['dataset'] not in ('cora', 'citeseer')
            or any(key.startswith(incompatible) for key in options)):
        raise ValueError('Wrong original-source dual-head protocol or modified native baseline')
    seeds = list(student_seeds)
    if (seeds != ([] if phase == 'qualify' else B['student_seeds'])
            or any(type(seed) is not int for seed in seeds)
            or (phase == 'science' and (len(seeds) != 3 or len(set(seeds)) != 3))):
        raise ValueError('Student cohort differs from its frozen three-seed phase')
    if mode == 'native_gaussian' and phase != 'science':
        raise ValueError('Native control reuses only original P0/P25 checkpoints')
    folder = Path(B['arm_folders'][mode])
    context_path = folder / 'context.json'
    frozen_context = None
    feature_map, phi, native, initial_parameters = None, None, None, None
    asset_bytes, original, source_geometry = None, None, None
    if mode == MODE:
        feature_map, phi, roles, descriptors, asset_bytes = _resident_inputs(packet, B, h, transform, stop)
        declaration = B['dual_head_context']
        refs = declaration['source_refs']
        expected = dict(current_source=packet['source'], data_digest=digest,
                        factor_seed=B['condensation_seed'], rank=options['assignment_rank'], nodes=len(z),
                        cells=int(assignment.max()) + 1, dimension=z.shape[1], classes=q.shape[1],
                        mixing=.05, chunk_size=options['chunk_size'], device=str(z.device), asset_paths=roles)
        if (set(declaration) != {'schema', 'policy', 'source_refs'}
                or type(declaration['schema']) is not int or declaration['schema'] != 1
                or declaration['policy'] != POLICY
                or set(refs) != set(expected) | {'files_sha256'}
                or any(type(refs[key]) is not int or refs[key] <= 0
                       for key in ('rank', 'nodes', 'cells', 'dimension', 'classes', 'chunk_size'))
                or type(refs['factor_seed']) is not int or refs['factor_seed'] < 0
                or any(refs.get(key) != value for key, value in expected.items())
                or any(packet['readonly_files_sha256'].get(path) != pin for path, pin in refs['files_sha256'].items())
                or not (set(roles.values()) | {B['native_P0_factors']['path']}) <= set(refs['files_sha256'])):
            raise ValueError('Dual-head declaration lost original source/map/RMS/Phi bindings')
        native, initial_parameters, parameter_digests = _native_parameters(packet, B, z, assignment, options, digest)
        observed_context = dict(schema=1, policy=POLICY,
            source_refs=dict(refs, original_options={key: value for key, value in options.items()
                                                   if key not in ('save_resume', 'data_digest')}),
            native_parameter_digests=parameter_digests, asset_descriptors=descriptors)
        if phase == 'qualify':
            frozen_context = observed_context
            original = torch.load(B['baseline_snapshot0'], map_location='cpu', weights_only=False)
            source_geometry = _source_geometry(original, native, h, z, assignment, transform, options, stop)
            if not source_geometry['affine_inverse_vs_direct_H_centroid_passed']:
                raise ValueError('Original RMS inverse/direct-H source gate failed before any candidate update')
        else:
            if not context_path.is_file() or str(context_path) not in packet['readonly_files_sha256']:
                raise ValueError('Science requires its accepted immutable canonical context file')
            frozen_context = json.loads(context_path.read_text())
            if frozen_context != observed_context:
                raise ValueError('Current original-source observations differ from saved canonical context')
            # Consume the saved context; observed descriptors are only lineage checks.
    context = frozen_context if mode == MODE else dict(
        schema=1, source=packet['source'], budget=budget, mode=mode,
        baseline=B['baseline'], condensation_seed=B['condensation_seed'], native_data_digest=digest,
        objective='original_teacher_CE', dual_head=None)
    if context_path.exists():
        if phase == 'qualify' or mode == 'native_gaussian' or json.loads(context_path.read_text()) != context:
            raise ValueError('Dual-head invocation is not a fresh qualification or exact accepted trajectory')
    elif folder.exists():
        raise ValueError('Partial dual-head output lacks its immutable context')
    else:
        if phase == 'science' and mode == MODE:
            raise ValueError('Science requires an existing qualified prefix')
        folder.mkdir(parents=True)
        save_json(context, context_path)
    if phase == 'science' and ((folder / 'validation').exists() or (folder / 'science_report.json').exists()):
        raise ValueError('Student validation output already exists; no retry or refit')
    new_updates, manifest_before, work_before, work_delta, state = 0, None, {}, None, None
    mutable_prefix = set()
    if mode == MODE:
        resume, manifest = folder / 'resume.pt', folder / 'checkpoint_manifest.json'
        if resume.exists():
            manifest_before = json.loads(manifest.read_text())
            _pins(manifest_before)
            state = torch.load(resume, map_location='cpu', weights_only=False)
            work_before = dict(state['work'])
        if phase == 'qualify' and state is not None:
            raise ValueError('Qualification owns exactly one fresh native update')
        if phase == 'science':
            qualification_path = folder / 'qualification.json'
            qualification = json.loads(qualification_path.read_text())
            if (qualification.get('passed') is not True or state is None or type(state.get('step')) is not int
                    or state['step'] != 1 or qualification.get('source') != packet['source']
                    or qualification.get('dual_head_context') != frozen_context
                    or qualification.get('source_geometry_checked_before_any_candidate_head_or_P_update') is not True
                    or qualification.get('affine_inverse_vs_direct_H_centroid_passed') is not True
                    or any(str(path) not in packet['readonly_files_sha256'] for path in (
                        context_path, qualification_path, folder / 'checkpoints/step_000000.pt',
                        folder / 'checkpoints/step_000001.pt'))):
                raise ValueError('Science requires its pinned accepted exact one-update prefix')
            mutable_prefix = {str(folder / name) for name in ('resume.pt', 'optimization.csv', 'checkpoint_manifest.json')}
            if (not mutable_prefix <= set(packet['readonly_files_sha256'])
                    or not packet.get('accepted_mutable_prefix_pin_policy', '').startswith('Sole pre-execution read')):
                raise ValueError('Science must bind evolving prefix bytes at its sole pre-execution read')
        steps, initial_step = (1 if phase == 'qualify' else 25), (0 if state is None else state['step'])
        optimize(z, q, assignment, initial_parameters, feature_map, phi, transform, folder, steps,
                 options, frozen_context, checkpoint_steps=(0, 1, 25), resume_state=state, stop=stop)
        state = torch.load(resume, map_location='cpu', weights_only=False)
        new_updates = steps - initial_step
        work_delta = _work_delta(state, work_before)
        if state['step'] != steps or state['context'] != frozen_context:
            raise ValueError('Executed dual-head trajectory differs from its frozen phase')
        if work_delta['P_updates'] != new_updates or work_delta['moment_backward_calls'] != new_updates:
            raise ValueError('Actual dual-head P updates/backwards differ from accepted phase frontier')
        if manifest_before is not None:
            _pins({name: pin for name, pin in manifest_before.items() if name not in mutable_prefix})
        paths = [context_path, resume, folder / 'optimization.csv', *sorted((folder / 'checkpoints').glob('step_*.pt'))]
        save_json({str(path): _sha(path) for path in paths}, manifest)
        if phase == 'qualify':
            geometry = _P0_geometry(state['snapshots'][0], original, native, z, q, transform, source_geometry)
            gates = [value for key, value in geometry.items() if key.endswith(('_original', '_stationary', '_equals2', '_passed'))]
            passed = all(value is True for value in gates)
            report = dict(schema=1, passed=passed, budget=budget, mode=mode, source=packet['source'],
                          dual_head_context=frozen_context, **geometry, actual_new_P_updates=new_updates,
                          optimizer_work_before=work_before, optimizer_work=state['work'],
                          actual_optimizer_work=work_delta,
                          classifier_fit_calls=work_delta['linear_head_interfaces'] + work_delta['Nystrom_head_interfaces'],
                          fixed_source_asset_bytes=asset_bytes, new_condensations=1, physical_student_fits=0,
                          existing_map_constructor_calls=1, existing_Phi_mmap_open_calls=1, native_factor_file_loads=1,
                          native_factor_generation_calls=0, kernel_factory_calls=0, Phi_rebuilds=0, test_evaluations=0)
            _pins(packet['readonly_files_sha256'])
            if implementation_provenance() != packet['source']:
                raise ValueError('Qualification changed current original-source bytes')
            save_json(report, folder / 'qualification.json')
            if not passed:
                raise ValueError('Dual-head original native P0/map/RMS geometry qualification failed')
            return report
        checkpoints, evaluation_steps = folder / 'checkpoints', (25,)
    else:
        checkpoints, evaluation_steps = Path(B['baseline_folder']) / 'checkpoints', (0, 25)
    records = []
    settings = {key: value for key, value in B['recipe'].items() if key != 'input_scale'}
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
                raise ValueError('Validation-only selected-weight route metadata differs')
            records.append(dict(step=step, **result, SGC_MLP_sameweights_val=routes['mlp_val_acc']))
    if mode == MODE:
        _pins(json.loads((folder / 'checkpoint_manifest.json').read_text()))
    _pins({name: pin for name, pin in packet['readonly_files_sha256'].items() if name not in mutable_prefix})
    if implementation_provenance() != packet['source']:
        raise ValueError('Science changed current original-source bytes')
    report = dict(schema=1, budget=budget, mode=mode, phase=phase, source=packet['source'], records=records,
                  trajectory_P_updates=25, actual_new_P_updates=new_updates,
                  new_condensations=0,
                  optimizer_work_before=work_before, optimizer_work=None if state is None else state['work'],
                  actual_optimizer_work=work_delta,
                  classifier_fit_calls=0 if work_delta is None else work_delta['linear_head_interfaces'] + work_delta['Nystrom_head_interfaces'],
                  new_physical_student_fits=len(records), same_selected_weight_validation_routes=2 * len(records),
                  candidate_P0_student_fits_reused_from_bit_exact_native_control=mode == MODE,
                  checkpoint_manifest_before=manifest_before, test_evaluations=0, free_features=False,
                  existing_map_constructor_calls=int(mode == MODE), existing_Phi_mmap_open_calls=int(mode == MODE),
                  native_factor_file_loads=int(mode == MODE), fixed_source_asset_bytes=asset_bytes,
                  native_factor_generation_calls=0, kernel_factory_calls=0, Phi_rebuilds=0,
                  objective=OBJECTIVE if mode == MODE else 'original_teacher_CE')
    save_json(report, folder / 'science_report.json')
    return report


def run(protocol_path, protocol_sha256, budget, phase, mode=None, student_seeds=(), stop=lambda: False):
    if _sha(protocol_path) != protocol_sha256:
        raise ValueError('Frozen dual-head CE protocol bytes changed')
    try:
        return _run(protocol_path, budget, phase, mode, student_seeds, stop)
    except InterruptedError as error:
        raise RuntimeError('Terminal bounded dual-head CE interruption; no automatic retry') from error
