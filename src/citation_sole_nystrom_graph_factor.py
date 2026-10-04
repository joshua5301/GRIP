"""Pinned original-cache sole-Nystrom critic, direct versus original-S NODE factors.

PREPARE owns a new canonical CSR/native-origin packet, not an old linear-head
packet. Qualification and condensation have no students; evaluation uses only
fixed original student recipes and validation routes. No cache factory is used.
"""
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from src.citation_factor_geometry import _pins, _sha, _source
from src.citation_dual_head_ce import _native_parameters, _resident_inputs
from src.citation_graph_factor import (
    _equal, _geometry_certificate, _graph_descriptor, _independent_reference,
    _json_write, _manifest, _native_gate_policy, _original_paths, _precision, _state_write,
)
from src.shared_features import _tensor_identity

KIND = 'fixed_sole_original_Nystrom_CE_graph_direct_interaction_v1'
CANDIDATE = 'onehop_original_S_NODE_factors_sole_original_Nystrom_CE_interaction_v1'
CONTROL_P0 = 'common_original_P0'
CONTROL_LINEAR = 'original_linear_CE25'
RESIDENT_MAXIMUM = 512 * 1024**2


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _asset_paths(B):
    family = Path(B['family_root'])
    phi = family / 'nystrom_phi_schema3.npy'
    return dict(H=str(family / 'propagated_H.pt'), map=str(family / 'nystrom_map_schema3.pt'),
                Phi=str(phi), Phi_metadata=str(phi.with_suffix('.meta.json')))


def _observed_source(packet, B, graph, h, z, q, assignment, transform, options, digest, stop):
    from src.moments import decode_moments
    from src.sole_nystrom_graph_factor_ce import _runtime, _native_options
    from src.sweep_utils import representative

    feature_map, phi, roles, assets, resident = _resident_inputs(packet, B, h, transform, stop)
    _require(resident <= RESIDENT_MAXIMUM and roles == _asset_paths(B), 'Original resident asset declaration differs')
    native, initial, factor_digests = _native_parameters(packet, B, z, assignment, options, digest)
    original = torch.load(B['baseline_snapshot0'], map_location='cpu', weights_only=False)
    M0 = original['moments']
    k, d, c, r = int(assignment.max()) + 1, z.shape[1], q.shape[1], options['assignment_rank']
    _require(M0.dtype == torch.float64 and M0.shape == (k, 1 + d + c)
             and bool(torch.isfinite(M0).all()) and bool((M0[:, 0] > 0).all()), 'Original base M0 is malformed')
    centers, labels, _ = decode_moments(M0, d)
    x0, y0, mass = representative(M0, transform, d, z.device)
    serving = dict(X=x0.detach().cpu().clone(), Q=y0.detach().cpu().clone(),
                   uniform_weights=torch.full_like(mass, 1 / len(mass)).detach().cpu().clone())
    geometry = _geometry_certificate(packet, B, digest)
    assets.update(z=_tensor_identity(z), Q=_tensor_identity(q), assignment=_tensor_identity(assignment))
    graph_descriptor = _graph_descriptor(graph['adj'], json.loads(
        (Path(B['family_root']) / 'config.json').read_text())['data_digest'])
    refs = dict(nodes=len(z), cells=k, rank=r, dimension=d, classes=c,
                chunk_size=options['chunk_size'], factor_seed=B['condensation_seed'], mixing=.05,
                original_options=_native_options(options), runtime=_runtime(z.device), device=str(z.device),
                data_digest=digest, asset_paths=roles, files_sha256=dict(packet['original_files_sha256']),
                current_source=packet['source'], optimizer_source_sha256=packet['optimizer_source_sha256'],
                source_graph=graph_descriptor, original_RMS_geometry=geometry,
                original_baseline_source=packet['original_baseline_source'], original_P0=B['baseline_snapshot0'],
                original_native_factors=B['native_P0_factors'], original_student_P0={key: _tensor_identity(value)
                    for key, value in serving.items()})
    _require(set(roles.values()) | {str(path) for path in _original_paths(B)} <= set(refs['files_sha256'])
             and all(packet['readonly_files_sha256'].get(path) == pin for path, pin in refs['files_sha256'].items()),
             'Original immutable source/cache references are not fully pinned')
    origin = dict(moments=_tensor_identity(M0), centers=_tensor_identity(centers), labels=_tensor_identity(labels))
    return feature_map, phi, initial, factor_digests, M0, serving, refs, assets, origin, resident


def _new_operator(arrays, rank, refs):
    from src.graph_factor import FrozenCPUCSR
    return FrozenCPUCSR(*(np.array(arrays[name], copy=True) for name in ('rowptr', 'col', 'values')),
                        rank=rank, source_refs=refs)


def _prepare(packet, B, graph, observed, progress):
    from src.graph_factor import FrozenCPUCSR
    from src.sole_nystrom_graph_factor_ce import SCHEMA, DIRECT, GRAPH, POLICY, _cpu, _seal

    _, _, initial, digests, M0, serving, refs, assets, origin, resident = observed
    folder = Path(B['input_folder'])
    _require(not folder.exists(), 'PREPARE requires a fresh owned namespace')
    folder.mkdir(parents=True)
    progress['receipt_path'] = str(folder / 'prepare_report.json')
    op_refs = dict(schema=1, domain=KIND, current_source=packet['source'], budget=B['budget'],
                   original_source_graph=refs['source_graph'], data_digest=refs['data_digest'],
                   native_parameter_digests=digests, native_origin=origin,
                   original_files_sha256=refs['files_sha256'])
    operator = FrozenCPUCSR.from_torch(graph['adj'], rank=refs['rank'], source_refs=op_refs)
    descriptor = operator.descriptor()
    arrays = dict(rowptr=graph['adj'].crow_indices().detach().cpu().numpy().copy(),
                  col=graph['adj'].col_indices().detach().cpu().numpy().copy(),
                  values=graph['adj'].values().detach().cpu().numpy().copy())
    contexts = {mode: dict(schema=SCHEMA, mode=mode, policy=POLICY, source_refs=refs,
        native_parameter_digests=digests, asset_descriptors=assets, native_origin=origin,
        operator=None if mode == DIRECT else descriptor) for mode in (DIRECT, GRAPH)}
    artifact = dict(schema=1, kind=KIND, source=packet['source'], budget=B['budget'], CSR=arrays,
        operator_source_refs=op_refs, operator_descriptor=descriptor, contexts=contexts,
        initial_parameters=_cpu(initial), original_moments=M0.detach().cpu().clone(), original_serving=serving)
    artifact['content_seal'] = _seal({key: value for key, value in artifact.items() if key != 'CSR'})
    path = folder / 'native_source_packet.pt'
    _state_write(artifact, path)
    for mode, context in contexts.items():
        _json_write(context, folder / f'{mode}_context.json')
    progress.update(native_source_packet=dict(path=str(path), sha256=_sha(path)), contexts=contexts,
        fixed_source_asset_bytes=resident, genuine_new_CSR_from_current_original_source=True,
        old_graph_packet_retargeted=False, old_linear_head_used_as_critic=False,
        new_native_factor_generations=0, actual_new_P_updates=0, physical_student_fits=0,
        actual_optimizer_work=None, actual_operator_work=operator.counts())


def _load_prepared(packet, B, graph, observed, mode):
    from src.sole_nystrom_graph_factor_ce import _seal, _cpu

    ref = B['native_source_packet']
    _require(packet['readonly_files_sha256'].get(ref['path']) == ref['sha256'], 'Own new prepared source packet unpinned')
    artifact = torch.load(ref['path'], map_location='cpu', weights_only=False)
    _, _, initial, digests, M0, serving, refs, assets, origin, _ = observed
    _require(artifact['schema'] == 1 and artifact['kind'] == KIND and artifact['source'] == packet['source']
             and artifact['budget'] == B['budget'] and artifact['content_seal'] ==
             _seal({key: value for key, value in artifact.items() if key not in ('CSR', 'content_seal')})
             and _seal(artifact['initial_parameters']) == _seal(_cpu(initial))
             and _equal(artifact['original_moments'], M0)
             and all(_equal(artifact['original_serving'][key], value) for key, value in serving.items()),
             'New prepared native-origin/source packet changed')
    operator = _new_operator(artifact['CSR'], refs['rank'], artifact['operator_source_refs'])
    _require(operator.descriptor() == artifact['operator_descriptor'] and operator.counts() ==
             dict(S_forward_products=0, T_transpose_products=0), 'Loading prepared CSR changed bits/runtime or performed products')
    actual = dict(rowptr=graph['adj'].crow_indices().detach().cpu().numpy(),
                  col=graph['adj'].col_indices().detach().cpu().numpy(), values=graph['adj'].values().detach().cpu().numpy())
    _require(all(value.dtype == artifact['CSR'][name].dtype and value.shape == artifact['CSR'][name].shape
                 and value.tobytes() == artifact['CSR'][name].tobytes() for name, value in actual.items()), 'Prepared CSR differs from original source coefficient/index bits')
    context = artifact['contexts'][mode]
    _require(context['source_refs'] == refs and context['asset_descriptors'] == assets
             and context['native_origin'] == origin and context['native_parameter_digests'] == digests,
             'Prepared cache/RMS/source/native origin no longer matches current frozen observations')
    return artifact, context, operator


def _work_delta(state, before):
    from src.sole_nystrom_graph_factor_ce import WORK_KEYS
    work = state['work']
    _require(set(work) == set(WORK_KEYS) and all(type(value) is int and value >= 0 for value in work.values())
             and (not before or set(before) == set(work)), 'Sole-head work schema changed')
    delta = {key: value - before.get(key, 0) for key, value in work.items()}
    _require(all(value >= 0 for value in delta.values()), 'Actual work moved backwards')
    return delta


def _P0(state, artifact, transform, dimension, device):
    from src.sweep_utils import representative

    p0 = state['snapshots'][0]
    x, q, mass = representative(p0['moments'], transform, dimension, device)
    serving = dict(X=x, Q=q, uniform_weights=torch.full_like(mass, 1 / len(mass)))
    return dict(P0_original_baseM_and_native_parameters_bit_exact=(
        _equal(p0['moments'], artifact['original_moments']) and bool(p0['effective_u'].eq(0).all())
        and all(_equal(a, b) for a, b in zip(p0['parameters'], artifact['initial_parameters'], strict=True))),
        P0_original_physical_student_X_Q_uniform_weights_bit_exact=all(_equal(value, artifact['original_serving'][key])
            for key, value in serving.items()), own_mapped_P0_head_stationary=p0['head_work']['inner_converged'] is True
            and math.isfinite(p0['head_work']['inner_grad_max']) and p0['head_work']['inner_grad_max'] <= 1e-7,
        own_positive_mapped_CE0_and_J0_equals1=math.isfinite(p0['CE0']) and p0['CE0'] > 0
            and p0['CE'] == p0['CE0'] and p0['objective'] == 1.0,
        own_mapped_P0_theta_descriptor=_tensor_identity(p0['theta']), own_mapped_CE0=p0['CE0'],
        original_linear_theta_or_CE0_used_as_critic=False,
        original_M0_enforced_by_core_before_any_head_or_Adam=True,
        original_student_bridge_descriptors={key: _tensor_identity(value) for key, value in serving.items()},
        source_geometry_qualification='transitive exact original M0/RMS/cache bridge and accepted old RMS certificate',
        source_geometry_new_numeric_measurements=0)


def _native_operator_gate(packet, state, artifact, graph, device, folder, progress, stop):
    gate = _native_gate_policy(packet)
    operator = _new_operator(artifact['CSR'], artifact['contexts'][state['context']['mode']]['source_refs']['rank'],
                             artifact['operator_source_refs'])
    descriptor = operator.descriptor()
    terminal, pb = state['snapshots'][1], state['current']['last_pullback']
    _require(pb['step'] == 0 and state['work']['S_forward_products'] == 2 and state['work']['T_transpose_products'] == 1,
             'Qualification lost its actual native graph pullback/counts')
    U, gW = terminal['parameters'][0].to(device), pb['gW'].to(device)
    forwards, transposes = [terminal['effective_u'].clone()], [pb['gU'].clone()]
    arrays = dict(U1=terminal['parameters'][0].clone(), gW0=pb['gW'].clone(),
                  forwards=forwards, transposes=transposes)
    path = folder / 'native_operator_gate_arrays.pt'
    complete, primary = False, None
    try:
        for _ in range(2):
            if stop():
                raise InterruptedError('Native original-S repeat stopped')
            forwards.append(operator.forward(U).cpu().clone()); transposes.append(operator.transpose(gW).cpu().clone())
        F, T = _independent_reference(graph['adj'], U, gW, stop)
        arrays.update(reference_forward_FP64=torch.from_numpy(F), reference_transpose_FP64=torch.from_numpy(T))
        complete = True
    except BaseException as error:
        primary = error
        raise
    finally:
        progress['additional_operator_work'] = operator.counts()
        try:
            _state_write(arrays, path)
            progress.update(native_operator_gate_arrays=dict(path=str(path), sha256=_sha(path),
                complete=complete, partial_unqualified=not complete))
        except BaseException as observation_error:
            if primary is None:
                raise
            primary.add_note('Actual native-array preservation also failed: ' + repr(observation_error))
    repeated = all(_equal(value, forwards[0]) for value in forwards) and all(_equal(value, transposes[0]) for value in transposes)
    precision = dict(forward=_precision(forwards[0], F, gate['precision']), transpose=_precision(transposes[0], T, gate['precision']))
    _require(operator.descriptor() == descriptor and operator.counts() == dict(S_forward_products=2, T_transpose_products=2), 'Native extra operator counts/runtime changed')
    progress.update(native_full_bytes_three_repeats_passed=repeated, native_independent_FP64_precision=precision,
        actual_operator_work=dict(S_forward_products=4, T_transpose_products=3),
        native_core_operator_work=dict(S_forward_products=2, T_transpose_products=1),
        native_repeat_run1='saved_actual_core_W1_and_gU0', independent_FP64_reference_calls=dict(S=1,T=1),
        native_operator_passed=repeated and all(value['passed'] for value in precision.values()))


def _evaluate(packet, B, graph, train, validation, h, z, q, transform, mode, seeds, progress, stop):
    from src.evaluation import fit_gcn_diagnostic
    from src.student_routes import replay_routes
    from src.sweep_utils import representative
    from src.sole_nystrom_graph_factor_ce import DIRECT, GRAPH, validate_core_resume

    folder = Path(B['evaluation_folders'][mode])
    _require(not folder.exists(), 'Physical student evaluation namespace exists; never refit/retry')
    folder.mkdir(parents=True); progress['receipt_path'] = str(folder / 'evaluation_report.json')
    if mode in (CONTROL_P0, CONTROL_LINEAR):
        step = 0 if mode == CONTROL_P0 else 25
        checkpoint = Path(B['baseline_folder']) / 'checkpoints' / f'step_{step:06d}.pt'
    else:
        _require(mode in (DIRECT, GRAPH), 'Unknown serving arm')
        trajectory = Path(B['arm_folders'][mode])
        accepted = json.loads((trajectory / 'condensation_report.json').read_text())
        _require(accepted['passed'] is True and accepted['source'] == packet['source'] and accepted['observed_final_step'] == 25,
                 'Evaluation requires accepted fixed25 condensation')
        _pins(json.loads((trajectory / 'checkpoint_manifest.json').read_text()))
        state = torch.load(trajectory / 'resume.pt', map_location='cpu', weights_only=False)
        validate_core_resume(state, state['config'], state['context'], trajectory)
        _require(state['step'] == 25, 'Candidate evaluation does not use exact fixed25 endpoint')
        checkpoint = trajectory / 'checkpoints/step_000025.pt'; step = 25
    _require(str(checkpoint) in packet['readonly_files_sha256'], 'Selected condensation checkpoint is not root-pinned')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    x, y, mass = representative(saved['moments'], transform, z.shape[1], z.device)
    settings = {key: value for key, value in B['recipe'].items() if key != 'input_scale'}
    records = []; progress.update(records=records, physical_student_fit_attempts=0, physical_student_fits=0,
                                 same_selected_weight_validation_routes=0, whole_epochs=0)
    for seed in seeds:
        if stop():
            raise InterruptedError('Fixed student cohort stopped')
        evaluation = folder / f'step_{step}_{B["recipe_id"]}'
        progress['physical_student_fit_attempts'] += 1
        result = fit_gcn_diagnostic(x, y, torch.full_like(mass, 1 / len(mass)), graph, q,
            dict(train=train, val=validation[1]), seed, folder=evaluation, stop=stop, **settings)
        progress['physical_student_fits'] += 1; progress['whole_epochs'] += B['recipe']['epochs']
        selected = evaluation / f'seed_{seed}_selected.pt'
        route_path = evaluation / f'seed_{seed}_validation_routes_v1.json'
        routes = replay_routes(selected, graph, h.float(), dict(val=validation[1]), settings, route_path, seed=seed, stop=stop)
        _require(not any('test_' in key for key in result | routes) and routes['gcn_val_acc'] == result['val_acc']
                 and routes['epoch'] == result['epoch'], 'Selected-weight validation route links differ')
        progress['same_selected_weight_validation_routes'] += 2
        records.append(dict(step=step, **result, SGC_MLP_sameweights_val=routes['mlp_val_acc'],
            selected_weights=dict(path=str(selected), sha256=_sha(selected)), routes=dict(path=str(route_path), sha256=_sha(route_path))))
    progress.update(actual_new_P_updates=0, actual_optimizer_work=None,
                    control_checkpoint=dict(path=str(checkpoint), sha256=_sha(checkpoint)))


def _run(protocol_path, budget, phase, mode, student_seeds, stop, progress):
    from src.research_loop import implementation_provenance
    from src.sole_nystrom_graph_factor_ce import DIRECT, GRAPH, optimize, validate_core_resume

    declaration = json.loads(Path(protocol_path).read_text()); B0 = declaration['budget_packets'][budget]
    _require(declaration['kind'] == KIND and B0['budget'] == budget and B0['dataset'] in ('cora', 'citeseer')
             and phase in ('prepare', 'qualify', 'condense', 'evaluate'), 'Wrong frozen sole-head interaction protocol')
    required = set(_asset_paths(B0).values()) | {str(path) for path in _original_paths(B0)}
    _require(all(Path(path).is_file() and path in declaration['readonly_files_sha256'] for path in required),
             'Original source/native/map/Phi/mandatory sidecar must exist before any loader fallback')
    packet, B, graph, train, validation, h, z, q, assignment, transform, options, digest = _source(protocol_path, budget, stop)
    seeds = list(student_seeds)
    _require(seeds == ([] if phase != 'evaluate' else B['student_seeds'])
             and all(type(seed) is int for seed in seeds) and (phase != 'evaluate' or len(seeds) == len(set(seeds)) == 3),
             'Frozen phase cohort differs')
    _require((phase == 'prepare' and mode is None) or (phase in ('qualify', 'condense') and mode in (DIRECT, GRAPH))
             or (phase == 'evaluate' and mode in (DIRECT, GRAPH, CONTROL_P0, CONTROL_LINEAR)), 'Unknown phase/arm')
    progress.update(schema=1, budget=budget, phase=phase, mode=mode, source=packet['source'],
        protocol_path=str(protocol_path), native_data_digest=digest, objective='own_sole_original_Nystrom_CE_over_own_CE0',
        test_evaluations=0, RMS_refits=0, H_rebuilds=0, kernel_factory_calls=0, Phi_rebuilds=0,
        native_factor_generation_calls=0, physical_student_fits=0, actual_new_P_updates=0, candidate=CANDIDATE,
        source_RMS_numeric_replays=0, basic_source_SGC_X_work='original source loader; uninstrumented, not claimed zero')
    mutable = set()
    if phase == 'evaluate':
        _evaluate(packet, B, graph, train, validation, h, z, q, transform, mode, seeds, progress, stop)
    else:
        observed = _observed_source(packet, B, graph, h, z, q, assignment, transform, options, digest, stop)
        if phase == 'prepare':
            _prepare(packet, B, graph, observed, progress)
        else:
            artifact, context, operator = _load_prepared(packet, B, graph, observed, mode)
            feature_map, phi = observed[:2]; initial = artifact['initial_parameters']
            folder = Path(B['arm_folders'][mode]); context_path = folder / 'context.json'
            _require(not (folder / 'failure.json').exists(), 'Failed trajectory cannot be rescued/retried')
            before, state = {}, None
            if phase == 'qualify':
                _require(not folder.exists(), 'Qualification requires a fresh one-update namespace')
                folder.mkdir(parents=True); _json_write(context, context_path)
            else:
                _require(context_path.is_file() and str(context_path) in packet['readonly_files_sha256']
                         and json.loads(context_path.read_text()) == context, 'Condensation lacks its immutable accepted context')
                accepted = json.loads((folder / 'qualification.json').read_text())
                _require(accepted['passed'] is True and accepted['source'] == packet['source']
                         and accepted['sole_head_context'] == context and accepted['P0_original_baseM_and_native_parameters_bit_exact']
                         and accepted['P0_original_physical_student_X_Q_uniform_weights_bit_exact']
                         and accepted['own_positive_mapped_CE0_and_J0_equals1']
                         and (mode == DIRECT or accepted['native_operator_passed']), 'Condensation qualification gates were not accepted')
                pair_ref = B['paired_native_acceptance']
                _require(packet['readonly_files_sha256'].get(pair_ref['path']) == pair_ref['sha256'], 'Root paired Ny-P0 acceptance is unpinned')
                pair = json.loads(Path(pair_ref['path']).read_text())
                _require(pair['passed'] is True and pair['source'] == packet['source']
                         and pair['budget_acceptances'][budget]['both_arms_original_P0_serving_bit_exact'] is True
                         and pair['budget_acceptances'][budget]['both_arms_new_Nystrom_theta0_CE0_bit_exact'] is True,
                         'Both matched critics must pass exact owned Ny-P0/CE0 equivalence before condensation')
                immutable = [context_path, folder / 'qualification.json', folder / 'checkpoints/step_000000.pt', folder / 'checkpoints/step_000001.pt']
                _require(all(str(path) in packet['readonly_files_sha256'] for path in immutable), 'Accepted0/1 prefix is not fully pinned')
                manifest = json.loads((folder / 'checkpoint_manifest.json').read_text()); _pins(manifest)
                state = torch.load(folder / 'resume.pt', map_location='cpu', weights_only=False)
                _require(type(state['step']) is int and state['step'] == 1, 'Only accepted1→25 continuation is authorized')
                before = dict(state['work'])
                mutable = {str(folder / name) for name in ('resume.pt', 'optimization.csv', 'checkpoint_manifest.json')}
                _require(mutable <= set(packet['readonly_files_sha256']) and packet.get('accepted_mutable_prefix_pin_policy', '').startswith('Sole pre-execution read'),
                         'Mutable accepted prefix must be pinned at its sole pre-execution read')
            progress['receipt_path'] = str(folder / ('qualification.json' if phase == 'qualify' else 'condensation_report.json'))
            steps = 1 if phase == 'qualify' else 25
            progress.update(optimizer_invocation_attempts=1, sole_head_context=context, observed_initial_step=0 if state is None else state['step'])
            state = optimize(z, q, assignment, initial, feature_map, phi, transform, folder, steps, options, context,
                operator=None if mode == DIRECT else operator, checkpoint_steps=(0, 1, steps), resume_state=state, stop=stop)
            progress['optimizer_invocation_completed'] = 1
            validate_core_resume(state, state['config'], context, folder)
            work = _work_delta(state, before); updates = 1 if phase == 'qualify' else 24
            _require(state['step'] == steps and work['P_updates'] == work['Adam_steps'] == work['moment_backward_calls'] == work['adjoint_solves'] == updates
                     and work['head_interfaces'] == work['moment_forward_calls'] == updates + 1,
                     'Actual one-head/adjoint/update counts differ from the fixed phase')
            progress.update(actual_optimizer_work=work, cumulative_optimizer_work=state['work'], actual_new_P_updates=updates,
                observed_final_step=steps, head_return_history=state['history'], fixed_source_asset_bytes=observed[-1],
                actual_operator_work={key: work[key] for key in ('S_forward_products', 'T_transpose_products')})
            if phase == 'qualify':
                gates = _P0(state, artifact, transform, z.shape[1], z.device); progress.update(gates)
                _require(all(gates[key] is True for key in ('P0_original_baseM_and_native_parameters_bit_exact',
                    'P0_original_physical_student_X_Q_uniform_weights_bit_exact', 'own_mapped_P0_head_stationary',
                    'own_positive_mapped_CE0_and_J0_equals1')), 'New sole-head original-P0 qualification failed')
                if mode == GRAPH:
                    _native_operator_gate(packet, state, artifact, graph, z.device, folder, progress, stop)
                    _require(progress['native_operator_passed'] is True, 'New actual factor repeat/precision gate failed')
                else:
                    progress['native_operator_passed'] = None
            paths = [context_path, folder / 'resume.pt', folder / 'optimization.csv', *sorted((folder / 'checkpoints').glob('step_*.pt'))]
            progress['checkpoint_manifest_after'] = _manifest(folder, paths)
            _pins({name: pin for name, pin in packet['readonly_files_sha256'].items() if name not in mutable})
    _pins({name: pin for name, pin in packet['readonly_files_sha256'].items() if name not in mutable})
    _require(implementation_provenance() == packet['source'], 'Full current implementation/Git changed')
    return progress


def run(protocol_path, protocol_sha256, budget, phase, mode=None, student_seeds=(), stop=lambda: False):
    """One root-owned bounded phase; no retry/resume from failed namespaces."""
    progress, started, primary = {}, time.monotonic(), None
    def bounded_stop():
        return stop() or time.monotonic() - started > 300
    try:
        _require(callable(stop) and _sha(protocol_path) == protocol_sha256, 'Frozen invocation protocol/callback changed')
        progress['protocol_sha256'] = protocol_sha256
        result = _run(protocol_path, budget, phase, mode, student_seeds, bounded_stop, progress)
        _require(not bounded_stop(), 'Sole-Nystrom phase exceeded its 300s body limit')
        result['passed'] = True
    except BaseException as error:
        primary = error; progress.update(passed=False, error_type=type(error).__name__, error=str(error))
        if isinstance(error, InterruptedError):
            raise RuntimeError('Terminal bounded sole-Nystrom interruption; no automatic retry') from error
        raise
    finally:
        progress['elapsed_seconds'] = time.monotonic() - started
        if torch.cuda.is_initialized():
            progress.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
        path = progress.pop('receipt_path', None)
        if path is not None:
            try:
                _json_write(progress, path)
            except BaseException as observation_error:
                if primary is None:
                    raise
                primary.add_note('Owned result preservation also failed: ' + repr(observation_error))
    return result
