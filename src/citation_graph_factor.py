"""Original-source one-hop NODE factor pilots with a frozen CPU CSR operator.

PREPARE copies the original graph, native factors and stationary P0 only.
Qualification owns one update; science continues its accepted prefix to 25.
Teachers, source caches, the linear head and student recipes remain original.
"""
import inspect
import json
import math
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch

from src.citation_factor_geometry import _pins, _sha, _source
from src.shared_features import _tensor_identity

MODE = 'cpu_scipy_FP32_CSR_graph_factor_v1'
KIND = 'fixed_onehop_original_S_NODE_factor_pilot_v1'
CANDIDATE = 'onehop_original_S_NODE_factors_linear_CE_v1'
OBJECTIVE = 'original_teacher_CE_over_original_positive_CE0'
GRAPH_KEYS = {'graph_factor_mode', 'graph_factor_artifact', 'graph_factor_sha256', 'graph_factor_context'}
CLOSED_PREFIXES = ('mlp_initial', 'mlp_output', 'mlp_source', 'source_linear', 'uniform_cell_q_prior',
                   'assignment_kl', 'node_factor', 'graph_assignment_kl', 'kernel_commutation',
                   'conditional_label_entropy', 'soft_cell_mass_KL', 'dual_head', 'graph_factor')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _json_write(value, path):
    """Create an owned finite-JSON output exclusively; no overwrite or repair."""
    raw = json.dumps(value, indent=2, allow_nan=False).encode()
    with Path(path).open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def _state_write(value, path):
    with Path(path).open('xb') as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())


def _equal(left, right):
    return _tensor_identity(left) == _tensor_identity(right)


def _original_paths(B):
    family, baseline = Path(B['family_root']), Path(B['baseline_folder'])
    return [family / name for name in ('config.json', 'propagated_H.pt', 'teacher.pt')] + [
        family / f"inputs_{B['condensation_seed']}.pt", Path(B['hard_assignment']),
        baseline / 'resume.pt', Path(B['baseline_snapshot0']),
        baseline / 'checkpoints/step_000025.pt', Path(B['native_P0_factors']['path'])]


def _canonical_config(options, digest):
    """Mirror only the unchanged inactive locals/default-key configuration.

    Omitted native defaults are merged for this declaration, never inserted in
    the actual options passed to the original optimizer. No optimizer executes.
    """
    from src.soft_ce_partition import optimize_ce_assignment

    excluded = {'z', 'q', 'assignment', 'steps', 'folder', 'checkpoint_steps', 'resume_state',
                'save_resume', 'log_every', 'initial_representatives', 'outer_indices',
                'implicit_solver', 'inner_solver', 'temperature_logits', 'outer_targets',
                'stop', 'mlp_initial_options'}
    signature = inspect.signature(optimize_ce_assignment)
    names = set(signature.parameters)
    _require(set(options) <= names and not any(key.startswith(CLOSED_PREFIXES) for key in options),
             'Original options contain an unsupported activated method')
    config = {name: parameter.default for name, parameter in signature.parameters.items()
              if name not in excluded and parameter.default is not inspect.Parameter.empty}
    config.update({key: value for key, value in options.items() if key not in excluded})
    _require(config['implicit_warm_start'] is True and config['inner_loss_weighting'] == 'uniform'
             and config['inner_method'] == 'newton_first' and config['cache_assignment'] is False
             and config['node_weighting'] is False, 'Original native default policy differs')
    config.pop('temperature_initial')
    config.pop('temperature_lr')
    if config['cg_check_interval'] == 1:
        config.pop('cg_check_interval')
    config.pop('cache_assignment')
    for key in ('node_weighting', 'node_weight_penalty', 'node_weight_lr'):
        config.pop(key)
    config['data_digest'] = digest
    # This copy is JSON metadata, separate from the unchanged native invocation.
    return json.loads(json.dumps(config, allow_nan=False))


def _transform(transform, dimension):
    _require(transform.kind == 'rms' and transform.matrix is None
             and torch.is_tensor(transform.scale) and transform.scale.dtype == torch.float64
             and transform.scale.numel() == 1 and bool(torch.isfinite(transform.scale))
             and float(transform.scale) > 0 and math.isfinite(transform.eps) and transform.eps > 0,
             'Require the original positive scalar RMS transform without a matrix')
    for value in (transform.center, transform.output_center):
        _require(value.dtype == torch.float64 and tuple(value.shape) == (dimension,)
                 and not value.requires_grad and bool(torch.isfinite(value).all()),
                 'Original RMS center shape/dtype/finite guard failed')
    return dict(kind=transform.kind, matrix=None, eps=transform.eps,
                center=_tensor_identity(transform.center), output_center=_tensor_identity(transform.output_center),
                scale=_tensor_identity(transform.scale))


def _native(packet, B, z, assignment, options, digest):
    ref = B['native_P0_factors']
    _require(packet['readonly_files_sha256'].get(ref['path']) == ref['sha256'],
             'Native P0 factor file is not a pinned original')
    native = torch.load(ref['path'], map_location='cpu', weights_only=False)
    rank, cells = options['assignment_rank'], int(assignment.max()) + 1
    parameters = [native[key] for key in ('u', 'v')]
    _require(native['data_digest'] == digest and native['factor_seed'] == B['condensation_seed']
             and native['mixing'] == .05 and type(rank) is int and rank > 0
             and all(torch.is_tensor(p) and p.dtype == torch.float32 and p.device.type == 'cpu'
                     and not p.requires_grad and bool(torch.isfinite(p).all()) for p in parameters)
             and tuple(parameters[0].shape) == (len(z), rank) and tuple(parameters[1].shape) == (cells, rank)
             and bool(parameters[0].eq(0).all()), 'Pinned original native factor origin differs')
    return parameters


def _geometry_certificate(packet, B, digest):
    """Reuse accepted original RMS geometry metadata, without replaying P0."""
    ref = B['source_geometry_acceptance']
    _require(packet['readonly_files_sha256'].get(ref['path']) == ref['sha256'],
             'Accepted original RMS geometry receipt is not pinned')
    accepted = json.loads(Path(ref['path']).read_text())
    _require(accepted['both_original_RMS_inverse_gates_passed'] is True
             and accepted['original_native_P0_and_serving_bitexact'] is True,
             'Original RMS source geometry was not accepted')
    report_ref, protocol_ref = accepted['reports'][B['budget']], accepted['protocol']
    _require(all(packet['readonly_files_sha256'].get(r['path']) == r['sha256']
                 for r in (report_ref, protocol_ref)), 'Original geometry report/protocol bytes are not pinned')
    old = json.loads(Path(protocol_ref['path']).read_text())
    _require(all(old['readonly_files_sha256'].get(str(path)) == packet['readonly_files_sha256'][str(path)]
                 for path in _original_paths(B)), 'Geometry acceptance uses different original source/cache bytes')
    report = json.loads(Path(report_ref['path']).read_text())
    residual = report['affine_inverse_vs_direct_H_centroid_max_absolute']
    _require(report['passed'] is True and report['budget'] == B['budget']
             and report['affine_inverse_vs_direct_H_centroid_passed'] is True
             and report['source_geometry_checked_before_any_candidate_head_or_P_update'] is True
             and report['dual_head_context']['source_refs']['data_digest'] == digest
             and math.isfinite(residual) and 0 <= residual <= 1e-12,
             'Accepted original RMS source residual/context differs')
    return dict(acceptance=ref, report=report_ref, original_residual=residual,
                original_RMS_geometry_metadata_reused=True, RMS_geometry_numeric_replays=0)


def _graph_descriptor(source_CSR, dataset_digest):
    _require(source_CSR.layout == torch.sparse_csr and source_CSR.dtype == torch.float32
             and source_CSR.crow_indices().dtype == source_CSR.col_indices().dtype == torch.int64,
             'Original normalized source graph is not canonical native FP32/int64 CSR')
    return dict(shape=list(source_CSR.shape), dtype=str(source_CSR.dtype), layout=str(source_CSR.layout),
                rowptr=_tensor_identity(source_CSR.crow_indices()), col=_tensor_identity(source_CSR.col_indices()),
                values=_tensor_identity(source_CSR.values()), dataset_digest=dataset_digest,
                normalization_calls=0, coalescing_calls=0)


def _prepare(packet, B, graph, z, q, assignment, transform, options, digest, progress):
    from src.graph_factor_context import build_packet, packet_context

    folder = Path(B['graph_input_folder'])
    _require(not folder.exists(), 'Graph PREPARE namespace exists; never overwrite/rebuild')
    folder.mkdir(parents=True)
    progress['receipt_path'] = str(folder / 'prepare_report.json')
    parameters = _native(packet, B, z, assignment, options, digest)
    original = torch.load(B['baseline_snapshot0'], map_location='cpu', weights_only=False)
    dims = dict(nodes=len(z), cells=int(assignment.max()) + 1, rank=options['assignment_rank'],
                dimension=z.shape[1], classes=q.shape[1])
    _require(all(type(v) is int and v > 0 for v in dims.values()) and dims['nodes'] <= 4000
             and dims['dimension'] <= 4000, 'Only the bounded small citation domain is authorized')
    original_config = _canonical_config(options, digest)
    refs = dict(**dims, data_digest=digest, factor_seed=B['condensation_seed'], mixing=.05,
                chunk_size=options.get('chunk_size', 4096), device=str(z.device), current_source=packet['source'],
                original_config=original_config,
                original_options={key: value for key, value in options.items() if key != 'save_resume'},
                files_sha256={str(path): packet['readonly_files_sha256'][str(path)] for path in _original_paths(B)},
                source_graph=_graph_descriptor(graph['adj'], json.loads(
                    (Path(B['family_root']) / 'config.json').read_text())['data_digest']),
                original_baseline_source=packet['original_baseline_source'],
                native_P0_factors=B['native_P0_factors'], original_P0=B['baseline_snapshot0'],
                RMS_transform=_transform(transform, dims['dimension']),
                source_Q=_tensor_identity(q), source_z=_tensor_identity(z), hard_assignment=_tensor_identity(assignment),
                original_RMS_geometry=_geometry_certificate(packet, B, digest))
    progress.update(native_factor_file_loads=1, native_factor_generation_calls=0, graph_packet_build_attempts=1)
    frozen = build_packet(graph['adj'], parameters, original, refs)
    context = packet_context(frozen)
    progress['graph_packet_build_completed'] = 1
    path, context_path = folder / 'graph_factor.pt', folder / 'context.json'
    _state_write(frozen, path)
    _json_write(context, context_path)
    progress.update(graph_factor=dict(path=str(path), sha256=_sha(path), context=context),
                    context_file=dict(path=str(context_path), sha256=_sha(context_path)),
                    actual_new_P_updates=0, classifier_fit_calls=0, physical_student_fits=0,
                    actual_operator_work=dict(S_forward_products=0, T_transpose_products=0),
                    native_origin_copied_without_head_or_moment_replay=True)
    return progress


def _native_gate_policy(packet):
    gate = packet['native_operator_gate']
    repeat, precision = gate['repeat'], gate['precision']
    _require(type(repeat['runs']) is int and repeat['runs'] == 3
             and repeat['additional_product_calls_each_direction'] == 2
             and repeat['dtype_shape_full_bytes_bit_exact'] is True
             and precision['absolute_Linf_maximum'] == 5e-7
             and precision['relative_Linf_maximum'] == 2e-5 and precision['both'] is True,
             'Native operator gate changed after prospective freeze')
    return gate


def _precision(actual, reference, limits):
    value = actual.detach().cpu().double().numpy()
    _require(value.shape == reference.shape and bool(np.isfinite(value).all())
             and bool(np.isfinite(reference).all()), 'Native graph reference or actual array is nonfinite')
    error = float(np.max(np.abs(value - reference)))
    norm = float(np.max(np.abs(reference)))
    relative = 0.0 if norm == 0 and error == 0 else (None if norm == 0 else error / norm)
    passed = (error == 0 if norm == 0 else error <= limits['absolute_Linf_maximum']
              and relative <= limits['relative_Linf_maximum'])
    return dict(absolute_Linf=error, reference_Linf=norm, relative_Linf=relative, passed=bool(passed))


def _independent_reference(source_CSR, U, gW, stop):
    """Independent FP64 per-row sums; never another production sparse backend."""
    rowptr = source_CSR.crow_indices().detach().cpu().numpy().copy()
    columns = source_CSR.col_indices().detach().cpu().numpy().copy()
    values = source_CSR.values().detach().cpu().numpy().copy()
    _require(rowptr.dtype == columns.dtype == np.int64 and values.dtype == np.float32,
             'Independent reference changed original coefficient/index bits')
    nodes = len(rowptr) - 1
    rows = np.repeat(np.arange(nodes, dtype=np.int64), np.diff(rowptr))
    permutation = np.lexsort((rows, columns))
    t_counts = np.bincount(columns, minlength=nodes)
    t_rowptr = np.empty(nodes + 1, dtype=np.int64)
    t_rowptr[0] = 0
    np.cumsum(t_counts, dtype=np.int64, out=t_rowptr[1:])
    t_columns, t_values = rows[permutation], values[permutation]
    _require(np.array_equal(t_values.view(np.uint32), values[permutation].view(np.uint32)),
             'Independent transpose changed original coefficient bits')
    operands = [value.detach().cpu().double().numpy().copy() for value in (U, gW)]
    references = []
    for rp, col, coeff, operand in ((rowptr, columns, values, operands[0]),
                                   (t_rowptr, t_columns, t_values, operands[1])):
        result = np.zeros_like(operand, dtype=np.float64)
        for row in range(nodes):
            if stop():
                raise InterruptedError('Native independent reference stopped')
            for index in range(int(rp[row]), int(rp[row + 1])):
                result[row] += np.float64(coeff[index]) * operand[col[index]]
        references.append(result)
    return references


def _operator_gate(operator, state, source_CSR, device, folder, gate, progress, stop):
    terminal, pullback = state['snapshots'][1], state['graph_factor_last_pullback']
    _require(pullback['step'] == 0 and operator.counts() == dict(S_forward_products=0, T_transpose_products=0),
             'Native diagnostic operator must be fresh and use the actual step0 pullback')
    U = terminal['graph_factor_parameters'][0].to(device=device)
    gW = pullback['gW'].to(device=device)
    before = operator.descriptor()
    forwards = [terminal['graph_factor_effective_u'].detach().cpu().clone()]
    transposes = [pullback['gU'].detach().cpu().clone()]
    arrays_path = folder / 'native_operator_gate_arrays.pt'
    arrays = dict(forwards=forwards, transposes=transposes,
                  U1=U.detach().cpu().clone(), gW0=gW.detach().cpu().clone())
    try:
        for _ in range(2):
            if stop():
                raise InterruptedError('Native graph repeat stopped')
            forwards.append(operator.forward(U).cpu().clone())
            transposes.append(operator.transpose(gW).cpu().clone())
        reference_F, reference_T = _independent_reference(source_CSR, U, gW, stop)
        arrays.update(reference_forward_FP64=torch.from_numpy(reference_F),
                      reference_transpose_FP64=torch.from_numpy(reference_T))
    except BaseException as error:
        # Completed extra products remain opaque evidence, never qualified work.
        try:
            _state_write(arrays, arrays_path)
            progress['native_operator_gate_arrays'] = dict(path=str(arrays_path), sha256=_sha(arrays_path),
                                                          complete=False, partial_unqualified=True)
        except BaseException as observation_error:
            error.add_note(f'Partial native-array preservation failed: {observation_error}')
        raise
    finally:
        progress['additional_native_operator_work'] = operator.counts()
    # Retain all completed actual arrays and independent references before gates.
    _state_write(arrays, arrays_path)
    progress['native_operator_gate_arrays'] = dict(path=str(arrays_path), sha256=_sha(arrays_path),
                                                  complete=True, partial_unqualified=False)
    repeated = (all(_equal(value, forwards[0]) for value in forwards)
                and all(_equal(value, transposes[0]) for value in transposes))
    precision = dict(forward=_precision(forwards[0], reference_F, gate['precision']),
                     transpose=_precision(transposes[0], reference_T, gate['precision']))
    _require(before == operator.descriptor(), 'Native operator descriptor changed during diagnostic products')
    core = state['graph_factor_operator_counts']
    extra = operator.counts()
    _require(core == dict(S_forward_products=2, T_transpose_products=1)
             and extra == dict(S_forward_products=2, T_transpose_products=2),
             'Native core/diagnostic graph product counts differ from the frozen gate')
    progress.update(native_full_bytes_three_repeats_passed=bool(repeated),
                    native_independent_FP64_precision=precision,
                    actual_operator_work={key: core[key] + extra[key] for key in core},
                    native_operator_passed=bool(repeated and all(v['passed'] for v in precision.values())),
                    native_core_operator_work=core,
                    native_operator_descriptor_before=before, native_operator_descriptor_after=operator.descriptor())


def _P0(state, original, native, transform, dimension, device):
    from src.sweep_utils import representative

    p0 = state['snapshots'][0]
    exact = (_equal(p0['moments'], original['moments']) and _equal(p0['theta'], original['theta'])
             and p0['teacher_ce'] == original['teacher_ce'] == state['scale']
             and p0['J_exact'] is True and bool(p0['graph_factor_effective_u'].eq(0).all())
             and all(_equal(a, b) for a, b in zip(p0['graph_factor_parameters'], native, strict=True)))
    left = representative(p0['moments'].to(device), transform, dimension, device)
    right = representative(original['moments'].to(device), transform, dimension, device)
    serving = all(_equal(a, b) for a, b in zip(left[:2], right[:2], strict=True))
    serving = serving and _equal(torch.full_like(left[2], 1 / len(left[2])),
                                torch.full_like(right[2], 1 / len(right[2])))
    return dict(P0_M_theta_CE_native_parameters_bit_exact_original=bool(exact),
                P0_student_X_Q_uniform_weights_bit_exact_original=bool(serving),
                P0_origin_enforced_in_core_snapshot_before_first_Adam=True)


def _manifest(folder, paths):
    value = {str(path): _sha(path) for path in paths}
    # A completed continuation replaces only this declared owned manifest.
    target = folder / 'checkpoint_manifest.json'
    temporary = folder / 'checkpoint_manifest.tmp.json'
    _json_write(value, temporary)
    temporary.replace(target)
    return value


def _run(protocol_path, budget, phase, mode, student_seeds, stop, progress):
    from src.graph_factor_context import load_frozen_graph, validate_resume
    from src.soft_ce_partition import optimize_ce_assignment
    from src.research_loop import implementation_provenance
    from src.evaluation import fit_gcn_diagnostic
    from src.student_routes import replay_routes
    from src.sweep_utils import representative

    _require(phase in ('prepare', 'qualify', 'science') and (mode is None if phase == 'prepare'
             else mode in (MODE, 'native_gaussian')), 'Unknown graph-factor phase or mode')
    declaration = json.loads(Path(protocol_path).read_text())
    declared = declaration['budget_packets'][budget]
    _require(declaration['kind'] == KIND and declaration['candidate'] == CANDIDATE
             and declared['budget'] == budget and declared['dataset'] in ('cora', 'citeseer'),
             'Wrong graph-factor candidate or bounded citation source')
    _require(all(path.is_file() and str(path) in declaration['readonly_files_sha256']
                 for path in _original_paths(declared)), 'Original source caches must exist and be pinned before load')
    packet, B, graph, train, validation, h, z, q, assignment, transform, options, digest = _source(protocol_path, budget, stop)
    _require(torch.get_default_dtype() == torch.float32 and z.dtype == q.dtype == torch.float64
             and not any(key.startswith(CLOSED_PREFIXES) for key in options), 'Original native source/default dtype differs')
    seeds = list(student_seeds)
    _require(seeds == ([] if phase != 'science' else B['student_seeds'])
             and all(type(seed) is int for seed in seeds)
             and (phase != 'science' or len(seeds) == len(set(seeds)) == 3), 'Frozen phase cohort differs')
    progress.update(schema=1, budget=budget, phase=phase, mode=mode, source=packet['source'],
                    protocol_path=str(protocol_path), native_data_digest=digest, candidate=CANDIDATE,
                    objective=OBJECTIVE, test_evaluations=0, validation_evaluations=0,
                    native_factor_generation_calls=0, RMS_refits=0, H_rebuilds=0, kernel_factory_calls=0,
                    Phi_rebuilds=0, mapped_head_interfaces=0,
                    native_environment=dict(device=str(z.device), Python=platform.python_version(),
                        Torch=str(torch.__version__), CPU_threads=torch.get_num_threads(),
                        default_dtype=str(torch.get_default_dtype()),
                        TF32_matmul=torch.backends.cuda.matmul.allow_tf32,
                        TF32_cudnn=torch.backends.cudnn.allow_tf32,
                        deterministic=torch.are_deterministic_algorithms_enabled()))
    mutable_prefix = set()
    if phase == 'prepare':
        _prepare(packet, B, graph, z, q, assignment, transform, options, digest, progress)
    else:
        _require(mode != 'native_gaussian' or phase == 'science', 'Native control has no new qualification/update')
        folder = Path(B['arm_folders'][mode])
        context_path = folder / 'context.json'
        _require(not (folder / 'validation').exists() and not (folder / 'science_report.json').exists(),
                 'Student outputs already exist; never retry/refit')
        state, manifest_before, before_counts = None, None, dict(S_forward_products=0, T_transpose_products=0)
        if mode == MODE:
            ref = B['graph_factor']
            gate = _native_gate_policy(packet)
            _require(packet['readonly_files_sha256'].get(ref['path']) == ref['sha256'], 'Prepared graph artifact is not pinned')
            context = ref['context']
            refs = context['source_refs']
            _require(refs['current_source'] == packet['source'] and refs['data_digest'] == digest
                     and refs['original_config'] == _canonical_config(options, digest)
                     and refs['original_options'] == {k: v for k, v in options.items() if k != 'save_resume'}
                     and refs['source_Q'] == _tensor_identity(q) and refs['source_z'] == _tensor_identity(z)
                     and refs['hard_assignment'] == _tensor_identity(assignment)
                     and refs['RMS_transform'] == _transform(transform, z.shape[1])
                     and refs['source_graph'] == _graph_descriptor(graph['adj'], json.loads(
                         (Path(B['family_root']) / 'config.json').read_text())['data_digest']),
                     'Prepared native graph/source/default/options context differs')
            loaded = load_frozen_graph(ref['path'], ref['sha256'],
                {key: refs[key] for key in ('nodes', 'cells', 'rank', 'dimension', 'classes')}, z.device, context)
            _require(loaded['operator'].counts() == before_counts, 'Loading a graph packet must perform no products')
            options.update(graph_factor_mode=MODE, graph_factor_artifact=ref['path'],
                           graph_factor_sha256=ref['sha256'], graph_factor_context=context)
            if phase == 'qualify':
                _require(not folder.exists(), 'Qualification requires a fresh namespace and one new update')
                folder.mkdir(parents=True)
                _json_write(context, context_path)
            else:
                _require(context_path.is_file() and str(context_path) in packet['readonly_files_sha256']
                         and json.loads(context_path.read_text()) == context, 'Science lacks its exact accepted context')
                manifest_before = json.loads((folder / 'checkpoint_manifest.json').read_text())
                _pins(manifest_before)
                state = torch.load(folder / 'resume.pt', map_location='cpu', weights_only=False)
                qualification_path = folder / 'qualification.json'
                qualification = json.loads(qualification_path.read_text())
                immutable = [context_path, qualification_path, folder / 'native_operator_gate_arrays.pt',
                             folder / 'checkpoints/step_000000.pt', folder / 'checkpoints/step_000001.pt']
                _require(qualification['passed'] is True and qualification['native_operator_passed'] is True
                         and qualification['graph_factor_context'] == context and qualification['source'] == packet['source']
                         and all(str(path) in packet['readonly_files_sha256'] for path in immutable)
                         and type(state['step']) is int and state['step'] == 1,
                         'Science requires its passed pinned exact one-update prefix')
                mutable_prefix = {str(folder / name) for name in ('resume.pt', 'optimization.csv', 'checkpoint_manifest.json')}
                _require(mutable_prefix <= set(packet['readonly_files_sha256'])
                         and packet.get('accepted_mutable_prefix_pin_policy', '').startswith('Sole pre-execution read'),
                         'Evolving accepted prefix must be pinned at its sole pre-execution read')
                before_counts = dict(state['graph_factor_operator_counts'])
            progress['receipt_path'] = str(folder / ('qualification.json' if phase == 'qualify' else 'science_report.json'))
            steps, start = (1 if phase == 'qualify' else 25), (0 if state is None else state['step'])
            progress.update(optimizer_invocation_attempts=1, observed_initial_step=start,
                            graph_factor_context=context, checkpoint_manifest_before=manifest_before)
            optimize_ce_assignment(z, q, assignment, steps=steps, folder=folder,
                                   checkpoint_steps=(0, steps), resume_state=state, stop=stop, **options)
            progress['optimizer_invocation_completed'] = 1
            state = torch.load(folder / 'resume.pt', map_location='cpu', weights_only=False)
            validate_resume(state, context, state['config'], steps)
            _require(state['step'] == steps, 'Core did not reach its exact authorized frontier')
            delta = {key: state['graph_factor_operator_counts'][key] - before_counts[key] for key in before_counts}
            _require(delta == dict(S_forward_products=steps - start + 1, T_transpose_products=steps - start),
                     'Actual original core graph products differ from inclusive endpoint/update counts')
            progress.update(actual_new_P_updates=steps - start, classifier_fit_calls=len(state['history']) - start,
                            actual_core_operator_work=delta, actual_operator_work=delta,
                            actual_head_interfaces=steps - start + 1, actual_adjoint_interfaces=steps - start,
                            actual_moment_forwards=steps - start + 1, actual_moment_backwards=steps - start,
                            actual_Adam_steps=steps - start,
                            original_head_return_history=state['history'],
                            uninstrumented_solver_interiors='unknown beyond original returned head/CG diagnostics')
            if manifest_before is not None:
                _pins({name: pin for name, pin in manifest_before.items() if name not in mutable_prefix})
            paths = [context_path, folder / 'resume.pt', folder / 'optimization.csv',
                     *sorted((folder / 'checkpoints').glob('step_*.pt'))]
            progress['checkpoint_manifest_after'] = _manifest(folder, paths)
            if phase == 'qualify':
                progress.update(_P0(state, loaded['native_snapshot'], loaded['initial_parameters'], transform, z.shape[1], z.device))
                _operator_gate(loaded['operator'], state, graph['adj'], z.device, folder,
                               gate, progress, stop)
                progress.update(physical_student_fits=0, graph_products_accuracy_scope='small native factor copy/operator only')
                _require(progress['P0_M_theta_CE_native_parameters_bit_exact_original'] is True
                         and progress['P0_student_X_Q_uniform_weights_bit_exact_original'] is True
                         and progress['native_operator_passed'] is True, 'Native P0/copy/repeat/precision gate failed')
            checkpoints, endpoints = folder / 'checkpoints', (25,)
        else:
            _require(not folder.exists(), 'Native fresh-student control namespace exists')
            folder.mkdir(parents=True)
            _json_write(dict(schema=1, source=packet['source'], budget=budget, mode=mode,
                             baseline=B['baseline'], native_data_digest=digest), context_path)
            progress.update(receipt_path=str(folder / 'science_report.json'), actual_new_P_updates=0,
                            classifier_fit_calls=0, actual_operator_work=before_counts)
            checkpoints, endpoints = Path(B['baseline_folder']) / 'checkpoints', (0, 25)
        if phase == 'science':
            records = []
            progress.update(records=records, physical_student_fits=0, same_selected_weight_validation_routes=0)
            settings = {key: value for key, value in B['recipe'].items() if key != 'input_scale'}
            for step in endpoints:
                saved = torch.load(checkpoints / f'step_{step:06d}.pt', map_location='cuda', weights_only=False)
                x, y, mass = representative(saved['moments'], transform, z.shape[1], z.device)
                for seed in seeds:
                    evaluation = folder / 'validation' / f"step_{step}_{B['recipe_id']}"
                    result = fit_gcn_diagnostic(x, y, torch.full_like(mass, 1 / len(mass)), graph, q,
                        dict(train=train, val=validation[1]), seed, folder=evaluation, stop=stop, **settings)
                    progress['physical_student_fits'] += 1
                    routes = replay_routes(evaluation / f'seed_{seed}_selected.pt', graph, h.float(), dict(val=validation[1]),
                        settings, evaluation / f'seed_{seed}_validation_routes_v1.json', seed=seed, stop=stop)
                    _require(not any('test_' in key for key in result | routes)
                             and routes['epoch'] == result['epoch'] and routes['gcn_val_acc'] == result['val_acc'],
                             'Validation-only selected-weight route links differ')
                    records.append(dict(step=step, **result, SGC_MLP_sameweights_val=routes['mlp_val_acc']))
                    progress['same_selected_weight_validation_routes'] += 2
                    progress['validation_evaluations'] += 2
            progress.update(trajectory_P_updates=25 if mode == MODE else 0,
                            candidate_P0_student_fits_reused_from_bit_exact_native_control=mode == MODE)
        if mode == MODE:
            _pins(json.loads((folder / 'checkpoint_manifest.json').read_text()))
    _pins({name: pin for name, pin in packet['readonly_files_sha256'].items() if name not in mutable_prefix})
    _require(implementation_provenance() == packet['source'], 'Invocation changed frozen implementation bytes/Git')
    return progress


def run(protocol_path, protocol_sha256, budget, phase, mode=None, student_seeds=(), stop=lambda: False):
    """One bounded invocation; root alone registers and advances its phases."""
    progress, started, primary_error = {}, time.monotonic(), None
    try:
        _require(callable(stop) and _sha(protocol_path) == protocol_sha256, 'Frozen protocol bytes/callback changed')
        progress['protocol_sha256'] = protocol_sha256
        result = _run(protocol_path, budget, phase, mode, student_seeds, stop, progress)
        _require(time.monotonic() - started <= 300, 'Native graph-factor invocation exceeded its frozen 300s body limit')
        result['passed'] = True
    except BaseException as error:
        primary_error = error
        progress.update(passed=False, error_type=type(error).__name__, error=str(error))
        if isinstance(error, InterruptedError):
            raise RuntimeError('Terminal bounded graph-factor interruption; no automatic retry') from error
        raise
    finally:
        progress['elapsed_seconds'] = time.monotonic() - started
        if torch.cuda.is_initialized():
            progress['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            progress['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        path = progress.pop('receipt_path', None)
        if path is not None:
            # Original history permits NaN for unevaluated interior diagnostics;
            # persist those explicitly as unknown, not as fabricated zero counts.
            rows = progress.get('original_head_return_history', [])
            progress['original_head_return_history'] = [{key: (None if isinstance(value, float)
                and not math.isfinite(value) else value) for key, value in row.items()} for row in rows]
            try:
                _json_write(progress, path)
            except BaseException as observation_error:
                if primary_error is None:
                    raise
                primary_error.add_note(f'Owned metadata preservation failed: {observation_error}')
    return result
