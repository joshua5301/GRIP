"""Frozen same-P0 NODE factor initialization and paired validation pilots.

Only the initial cell factor changes. All synthetic features and targets still
come from the existing assignment moments. Teachers, P0, losses and student
recipes are the pinned original ones; this adapter never evaluates test masks.
"""
import hashlib
import json
from pathlib import Path

import torch

from src.io import array_digest, save_json, save_state


MODES = ('prototype_gram_v1', 'centered_gaussian_v1')


def _sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _pins(pins):
    for path, pin in pins.items():
        if _sha(path) != pin:
            raise ValueError(f'Factor geometry pinned input changed: {path}')


def _source(protocol_path, budget, stop):
    from src.research_loop import implementation_provenance
    from src.citation_search import dataset_digest, fixed_propagated_features
    from src.data import _prepare_dataset
    from src.target_refinement import training_refined_targets
    from src.transforms import FeatureTransform

    packet = json.loads(Path(protocol_path).read_text())
    if packet['source'] != implementation_provenance():
        raise ValueError('Factor geometry implementation differs from its protocol')
    _pins(packet['readonly_files_sha256'])
    if stop():
        raise InterruptedError('Factor geometry stopped before loading source')
    B = packet['budget_packets'][budget]
    family = Path(B['family_root'])
    config = json.loads((family / 'config.json').read_text())
    if (config.get('teacher_backend') is not None or config.get('teacher_kernel') not in (None, 'relu')
            or config.get('citation_features') != 'row' or config.get('mixing') != .05
            or B['recipe']['input_scale'] != 1.0):
        raise ValueError('Require the original ROW ReLU source and student scale')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    graph, train, validation, testing, current_h = _prepare_dataset(B['dataset'], packet['data_dir'], 'cuda', 'row')
    if dataset_digest(graph, train, validation, testing) != config['data_digest']:
        raise ValueError('Factor geometry graph/splits differ from original source')
    h = fixed_propagated_features(current_h, config, family)
    seed = B['condensation_seed']
    inputs = torch.load(family / f'inputs_{seed}.pt', map_location='cuda', weights_only=False)
    z = inputs['z']
    transform = FeatureTransform(**inputs['transform'])
    assignment = torch.load(B['hard_assignment'], map_location='cuda', weights_only=False)
    logits = torch.load(family / 'teacher.pt', map_location='cuda', weights_only=False)['logits']
    q = training_refined_targets(logits, B['baseline']['T'], graph['y'], train, 0.0)
    original = torch.load(Path(B['baseline_folder']) / 'resume.pt', map_location='cpu', weights_only=False)
    data_digest = array_digest(z.cpu().numpy(), q.cpu().numpy(), assignment.cpu().numpy())
    options = dict(original['config'])
    expected = dict(data_digest=data_digest, assignment_rank=B['baseline']['rank'], factor_seed=seed,
        assignment_input='node', assignment_encoder='linear', mass_mode='free', inner_loss_weighting='uniform',
        penalty=B['baseline']['penalty'], lr=B['baseline']['lr'], mixing=.05, solver_mode='exact',
        inner_method='newton_first', implicit_warm_start=True, save_assignment=False)
    if any(options.get(key) != value for key, value in expected.items()):
        raise ValueError('Original baseline lost its native factor/data/solver binding')
    if any(key.startswith(('node_factor_', 'assignment_kl_', 'uniform_cell_q_prior_')) for key in options):
        raise ValueError('Original baseline must use its unmodified Gaussian recipe')
    options.pop('data_digest')
    options['save_resume'] = True
    return packet, B, graph, train, validation, h, z, q, assignment, transform, options, data_digest


def _prepare(protocol_path, budget, stop):
    from src.low_rank_assignment import initialize_factors
    from src.prototype_factor_initialization import build_factor_packet, packet_context

    packet, B, _, _, _, _, z, _, assignment, _, options, digest = _source(protocol_path, budget, stop)
    folder = Path(B['factor_input_folder'])
    if folder.exists():
        raise ValueError('Frozen factor PREPARE namespace already exists; never overwrite/recompute')
    folder.mkdir(parents=True)
    rank, cells = options['assignment_rank'], int(assignment.max()) + 1
    u, v = initialize_factors(assignment, cells, rank, options['factor_seed'])
    if not torch.equal(u.detach(), torch.zeros_like(u)):
        raise ValueError('Native factor initialization must have exact zero U0')
    native = folder / 'native_reference_factors.pt'
    save_state(dict(u=u, v=v, data_digest=digest, factor_seed=options['factor_seed'], mixing=.05), native)
    initial = torch.load(B['baseline_snapshot0'], map_location='cpu', weights_only=False)
    moments = initial['moments']
    if moments.shape != (cells, 1 + z.shape[1] + initial['theta'].shape[0]):
        raise ValueError('Pinned P0 moments have incompatible source dimensions')
    refs = dict(data_digest=digest, factor_seed=options['factor_seed'], mixing=.05,
        original_baseline_source=packet['original_baseline_source'],
        native_reference_factors=dict(path=str(native), sha256=_sha(native)),
        original_P0=dict(path=B['baseline_snapshot0'], sha256=_sha(B['baseline_snapshot0'])),
        source_inputs=dict(path=str(Path(B['family_root']) / f"inputs_{B['condensation_seed']}.pt"),
            sha256=_sha(Path(B['family_root']) / f"inputs_{B['condensation_seed']}.pt")),
        hard_assignment=dict(path=B['hard_assignment'], sha256=_sha(B['hard_assignment'])))
    factors = {}
    for mode in MODES:
        frozen = build_factor_packet(dict(mass=moments[:, 0], features=moments[:, 1:1 + z.shape[1]]),
                                     v.detach().cpu(), mode, rank, refs)
        path = folder / f'{mode}.pt'
        save_state(frozen, path)
        factors[mode] = dict(path=str(path), sha256=_sha(path), context=packet_context(frozen),
                             diagnostics=frozen['diagnostics'])
    report = dict(schema=1, budget=budget, source=packet['source'], factors=factors,
        native_reference_factors_sha256=_sha(native), native_reference_factors_path=str(native),
        native_factor_generation_calls=1, new_condensations=0, P_updates=0, student_fits=0, test_evaluations=0)
    save_json(report, folder / 'prepare_report.json')
    return report


def _run(protocol_path, budget, phase, mode, student_seeds, stop):
    from src.soft_ce_partition import optimize_ce_assignment
    from src.evaluation import fit_gcn_diagnostic
    from src.student_routes import replay_routes
    from src.sweep_utils import representative
    from src.prototype_factor_initialization import load_frozen_factor

    if phase == 'prepare':
        if mode is not None or student_seeds:
            raise ValueError('Factor PREPARE never selects an arm or fits students')
        return _prepare(protocol_path, budget, stop)
    if phase not in ('qualify', 'science') or mode not in (*MODES, 'native_gaussian'):
        raise ValueError('Unknown frozen factor geometry phase/arm')
    packet, B, graph, train, validation, h, z, q, assignment, transform, options, digest = _source(protocol_path, budget, stop)
    seeds = list(student_seeds)
    if seeds != ([] if phase == 'qualify' else B['student_seeds']):
        raise ValueError('Factor geometry cohort differs from its frozen phase')
    steps = 1 if phase == 'qualify' else 25
    folder = Path(B['arm_folders'][mode])
    context = dict(schema=1, source=packet['source'], budget=budget, mode=mode,
        baseline=B['baseline'], condensation_seed=B['condensation_seed'], native_data_digest=digest,
        fixed_loss='original_source_teacher_CE; uniform_inner_and_student_CE; free_P_mass',
        factor=None if mode == 'native_gaussian' else B['factors'][mode])
    context_path = folder / 'context.json'
    if context_path.exists():
        if json.loads(context_path.read_text()) != context:
            raise ValueError('Factor geometry trajectory context changed')
    elif folder.exists():
        raise ValueError('Factor geometry partial folder lacks its frozen context')
    else:
        folder.mkdir(parents=True)
        save_json(context, context_path)
    initial_step, new_updates = 25, 0
    if mode != 'native_gaussian':
        factor = B['factors'][mode]
        frozen_v = load_frozen_factor(factor['path'], factor['sha256'], mode,
            dict(cells=int(assignment.max()) + 1, feature_dimension=z.shape[1]), 'cuda', expected_context=factor['context'])
        options.update(node_factor_mode=mode, node_factor_artifact=factor['path'],
                       node_factor_sha256=factor['sha256'], node_factor_context=factor['context'])
        resume_path, manifest = folder / 'resume.pt', folder / 'checkpoint_manifest.json'
        if resume_path.exists():
            if not manifest.exists():
                raise ValueError('Factor resume lacks its completed checkpoint manifest')
            _pins(json.loads(manifest.read_text()))
            state = torch.load(resume_path, map_location='cpu', weights_only=False)
            if state['step'] > steps or state.get('node_factor_context') != factor['context']:
                raise ValueError('Factor geometry resume is outside its bound phase')
        else:
            state = None
        initial_step = 0 if state is None else state['step']
        if phase == 'qualify' and state is not None:
            raise ValueError('Qualification requires a fresh arm namespace and exactly one new P update')
        if phase == 'science':
            qualification = json.loads((folder / 'qualification.json').read_text())
            if not qualification['passed'] or state is None or state['step'] != 1:
                raise ValueError('Science requires its passed frozen one-update qualification prefix')
        if state is None or initial_step < steps:
            optimize_ce_assignment(z, q, assignment, steps=steps, folder=folder, checkpoint_steps=(0, steps),
                                   resume_state=state, stop=stop, **options)
            state = torch.load(resume_path, map_location='cpu', weights_only=False)
            new_updates = steps - initial_step
            paths = [resume_path, folder / 'optimization.csv', *sorted((folder / 'checkpoints').glob('step_*.pt'))]
            save_json({str(path): _sha(path) for path in paths}, manifest)
        if state['step'] != steps or state.get('node_factor_context') != factor['context']:
            raise ValueError('Factor geometry terminal state lacks its bound endpoint')
        if new_updates != (1 if phase == 'qualify' else 24):
            raise ValueError('Factor geometry work differs from its frozen one-update prefix or 24-update continuation')
        p0 = torch.load(folder / 'checkpoints/step_000000.pt', map_location='cuda', weights_only=False)
        original = torch.load(B['baseline_snapshot0'], map_location='cuda', weights_only=False)
        u0, v0 = p0['node_factor_parameters']
        passed = (u0.dtype == torch.float32 and v0.dtype == torch.float32
            and torch.equal(u0, torch.zeros_like(u0)) and torch.equal(v0, frozen_v)
            and torch.equal(p0['moments'], original['moments']) and torch.equal(p0['theta'], original['theta'])
            and p0['teacher_ce'] == original['teacher_ce'] and p0['J_exact']
            and p0['node_factor_context'] == factor['context']
            and all(s.get('node_factor_context') == factor['context'] for s in state['snapshots'].values()))
        if not passed:
            raise ValueError('Same-P0 native moments/head/factor qualification failed')
        if phase == 'qualify':
            qualification = dict(schema=1, passed=passed, budget=budget, mode=mode,
                source=packet['source'], factor_context=factor['context'],
                P0_moments_and_head_bit_exact_original=True, U0_exact_zero=True, V0_exact_frozen=True,
                actual_new_P_updates=new_updates, physical_student_fits=0, test_evaluations=0)
            save_json(qualification, folder / 'qualification.json')
            return qualification
        checkpoints = folder / 'checkpoints'
    else:
        if phase != 'science':
            raise ValueError('Original Gaussian anchor uses pinned existing P0/P25, no qualification rerun')
        checkpoints = Path(B['baseline_folder']) / 'checkpoints'
    records = []
    settings = {key: value for key, value in B['recipe'].items() if key != 'input_scale'}
    for step in (0, 25):
        saved = torch.load(checkpoints / f'step_{step:06d}.pt', map_location='cuda', weights_only=False)
        x, y, mass = representative(saved['moments'], transform, z.shape[1], 'cuda')
        for seed in seeds:
            evaluation = folder / 'validation' / f"step_{step}_{B['recipe_id']}"
            result = fit_gcn_diagnostic(x, y, torch.full_like(mass, 1 / len(mass)), graph, q,
                dict(train=train, val=validation[1]), seed, folder=evaluation, stop=stop, **settings)
            routes = replay_routes(evaluation / f'seed_{seed}_selected.pt', graph, h.float(), dict(val=validation[1]),
                settings, evaluation / f'seed_{seed}_validation_routes_v1.json', seed=seed, stop=stop)
            if (any('test_' in key for key in result | routes) or routes['epoch'] != result['epoch']
                    or routes['gcn_val_acc'] != result['val_acc']):
                raise ValueError('Factor geometry validation routes differ from the selected student')
            records.append(dict(step=step, **result, SGC_MLP_sameweights_val=routes['mlp_val_acc']))
    report = dict(schema=1, budget=budget, mode=mode, phase=phase, folder=str(folder), records=records,
        trajectory_P_updates=25, actual_new_P_updates=new_updates, new_physical_student_fits=len(records),
        same_selected_weight_validation_routes=2 * len(records), test_evaluations=0,
        interpretation='Factor coordinate/spectrum initialization under Adam; no capacity or kernel-only claim')
    save_json(report, folder / 'science_report.json')
    return report


def run(protocol_path, protocol_sha256, budget, phase, mode=None, student_seeds=(), stop=lambda: False):
    try:
        if _sha(protocol_path) != protocol_sha256:
            raise ValueError('Frozen factor geometry protocol bytes changed')
        return _run(protocol_path, budget, phase, mode, student_seeds, stop)
    except InterruptedError as error:
        raise RuntimeError('Terminal bounded factor-geometry interruption; no automatic partial retry') from error
