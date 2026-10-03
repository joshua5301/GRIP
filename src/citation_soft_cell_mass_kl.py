"""Fixed original-source CE plus soft cell occupancy KL pilots."""
import json
from pathlib import Path

import torch

from src.citation_factor_geometry import _pins, _sha, _source
from src.io import save_json
from src.shared_features import _tensor_identity

MODE = 'normalized_soft_cell_mass_KL_v1'
OBJECTIVE = 'teacher_CE_over_CE0_plus_soft_cell_mass_KL_over_Omega0'


def _run(protocol_path, budget, phase, mode, student_seeds, stop):
    from src.soft_ce_partition import optimize_ce_assignment
    from src.evaluation import fit_gcn_diagnostic
    from src.student_routes import replay_routes
    from src.sweep_utils import representative
    from src.research_loop import implementation_provenance
    from src.soft_cell_mass_kl import POLICY

    if phase not in ('qualify', 'science') or mode not in (MODE, 'native_gaussian'):
        raise ValueError('Soft cell mass KL has only qualification and science phases')
    packet, B, graph, train, validation, h, z, q, assignment, transform, options, digest = _source(protocol_path, budget, stop)
    if (packet['kind'] != 'fixed_normalized_soft_cell_mass_KL_pilot_v1'
            or B['budget'] != budget or B['dataset'] not in ('cora', 'citeseer')
            or any(key.startswith(('graph_assignment_kl_', 'kernel_commutation_',
                                   'conditional_label_entropy_', 'soft_cell_mass_KL_')) for key in options)):
        raise ValueError('Wrong original-source soft cell mass KL protocol or baseline')
    seeds = list(student_seeds)
    if seeds != ([] if phase == 'qualify' else B['student_seeds']):
        raise ValueError('Student cohort differs from the prospectively frozen phase')
    if mode == 'native_gaussian' and phase != 'science':
        raise ValueError('Native control uses only existing original P0/P25 checkpoints')
    folder = Path(B['arm_folders'][mode])
    frozen_context = B['soft_mass_context'] if mode == MODE else None
    if mode == MODE and (frozen_context['policy'] != POLICY
            or frozen_context['source_refs']['current_source'] != packet['source']
            or frozen_context['source_refs']['data_digest'] != digest):
        raise ValueError('Soft cell mass KL context does not bind current original source and policy')
    context = dict(schema=1, source=packet['source'], budget=budget, mode=mode,
                   baseline=B['baseline'], condensation_seed=B['condensation_seed'],
                   native_data_digest=digest,
                   objective=OBJECTIVE if mode == MODE else 'original_teacher_CE',
                   soft_mass=frozen_context)
    path = folder / 'context.json'
    if path.exists():
        if phase == 'qualify' or mode == 'native_gaussian':
            raise ValueError('Fresh qualification/native-control invocation already exists')
        if json.loads(path.read_text()) != context:
            raise ValueError('Trajectory namespace or original-source context changed')
    elif folder.exists():
        raise ValueError('Partial soft_mass output folder lacks its immutable context')
    else:
        folder.mkdir(parents=True)
        save_json(context, path)
    if phase == 'science' and ((folder / 'validation').exists() or (folder / 'science_report.json').exists()):
        raise ValueError('Science already has a student cache/report; no retry or refit')
    new_updates, fit_calls, manifest_before = 0, 0, None
    mutable_prefix = set()
    if mode == MODE:
        options.update(soft_cell_mass_KL_mode=MODE, soft_cell_mass_KL_context=frozen_context)
        resume, manifest = folder / 'resume.pt', folder / 'checkpoint_manifest.json'
        state = None
        if resume.exists():
            manifest_before = json.loads(manifest.read_text())
            _pins(manifest_before)
            state = torch.load(resume, map_location='cpu', weights_only=False)
        if phase == 'qualify' and state is not None:
            raise ValueError('Qualification must own exactly one fresh native P update')
        if phase == 'science':
            qualification = json.loads((folder / 'qualification.json').read_text())
            if (qualification.get('passed') is not True or state is None or type(state.get('step')) is not int
                    or state['step'] != 1 or qualification.get('source') != packet['source']
                    or qualification.get('soft_mass_context') != frozen_context):
                raise ValueError('Science requires its accepted exact one-update prefix')
            mutable_prefix = {str(folder / name) for name in ('resume.pt', 'optimization.csv', 'checkpoint_manifest.json')}
            if (not mutable_prefix <= set(packet['readonly_files_sha256'])
                    or not packet.get('accepted_mutable_prefix_pin_policy', '').startswith('Sole pre-execution read')):
                raise ValueError('Science protocol must bind the three accepted mutable prefix bytes before its sole invocation')
        steps = 1 if phase == 'qualify' else 25
        initial_step = 0 if state is None else state['step']
        optimize_ce_assignment(z, q, assignment, steps=steps, folder=folder, checkpoint_steps=(0, steps),
                               resume_state=state, stop=stop, **options)
        state = torch.load(resume, map_location='cpu', weights_only=False)
        new_updates, fit_calls = steps - initial_step, steps - initial_step + 1
        if state['step'] != steps or state['soft_cell_mass_KL_context'] != frozen_context:
            raise ValueError('Executed soft_mass trajectory differs from its frozen phase')
        if manifest_before is not None:
            _pins({name: pin for name, pin in manifest_before.items() if name not in mutable_prefix})
        paths = [resume, folder / 'optimization.csv', *sorted((folder / 'checkpoints').glob('step_*.pt'))]
        save_json({str(p): _sha(p) for p in paths}, manifest)
        if phase == 'qualify':
            p0 = state['snapshots'][0]
            original = torch.load(B['baseline_snapshot0'], map_location='cpu', weights_only=False)
            native_ref = B['native_P0_factors']
            _pins({native_ref['path']: native_ref['sha256']})
            native = torch.load(native_ref['path'], map_location='cpu', weights_only=False)
            original_mass = original['moments'][:, 0]
            expected_Omega0 = float((original_mass * (len(original_mass) * original_mass).log()).sum())
            if not torch.isfinite(torch.tensor(expected_Omega0)) or expected_Omega0 <= 0:
                raise ValueError('Original native P0 soft cell mass KL scale must be strictly positive')
            discrepancy = abs(p0['soft_cell_mass_KL_Omega0'] - expected_Omega0)
            equal = lambda a, b: _tensor_identity(a) == _tensor_identity(b)
            passed = (equal(p0['moments'], original['moments']) and equal(p0['theta'], original['theta'])
                and p0['J_exact'] is True
                and p0['teacher_ce'] == original['teacher_ce'] == p0['soft_cell_mass_KL_CE0']
                and native['data_digest'] == digest and native['factor_seed'] == B['condensation_seed']
                and native['mixing'] == .05
                and all(equal(a, native[k]) for a, k in zip(p0['soft_cell_mass_KL_parameters'], ('u', 'v'), strict=True))
                and p0['soft_cell_mass_KL_Omega0'] > 0 and discrepancy <= 1e-12
                and p0['objective'] == 2.0 and state['scale'] == p0['soft_cell_mass_KL_CE0'])
            report = dict(schema=1, passed=bool(passed), budget=budget, mode=mode, source=packet['source'],
                P0_moments_head_factors_and_teacherCE_bit_exact_original=bool(passed), soft_mass_context=frozen_context,
                diagnostic_Omega0=expected_Omega0, Omega0_discrepancy=discrepancy, actual_new_P_updates=new_updates,
                classifier_fit_calls=fit_calls, new_condensations=1, physical_student_fits=0,
                kernel_factory_calls=0, Phi_rebuilds=0, test_evaluations=0)
            _pins(packet['readonly_files_sha256'])
            if implementation_provenance() != packet['source']:
                raise ValueError('Qualification changed current original-source bytes')
            save_json(report, folder / 'qualification.json')
            if not passed:
                raise ValueError('Original native P0/head/factors and CE0/Omega0 qualification failed')
            return report
        checkpoints, evaluation_steps = folder / 'checkpoints', (25,)
    else:
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
                raise ValueError('Validation-only selected-weight route metadata differs')
            records.append(dict(step=step, **result, SGC_MLP_sameweights_val=routes['mlp_val_acc']))
    if mode == MODE:
        _pins(json.loads((folder / 'checkpoint_manifest.json').read_text()))
    _pins({name: pin for name, pin in packet['readonly_files_sha256'].items() if name not in mutable_prefix})
    if implementation_provenance() != packet['source']:
        raise ValueError('Science changed current original-source bytes')
    report = dict(schema=1, budget=budget, mode=mode, phase=phase, source=packet['source'], records=records,
        trajectory_P_updates=25, actual_new_P_updates=new_updates, classifier_fit_calls=fit_calls,
        new_physical_student_fits=len(records), same_selected_weight_validation_routes=2 * len(records),
        candidate_P0_student_fits_reused_from_bit_exact_native_control=mode == MODE,
        checkpoint_manifest_before=manifest_before,
        test_evaluations=0, free_features=False, kernel_factory_calls=0, Phi_rebuilds=0,
        objective=OBJECTIVE if mode == MODE else 'original_teacher_CE')
    save_json(report, folder / 'science_report.json')
    return report


def run(protocol_path, protocol_sha256, budget, phase, mode=None, student_seeds=(), stop=lambda: False):
    if _sha(protocol_path) != protocol_sha256:
        raise ValueError('Frozen soft cell mass KL protocol bytes changed')
    try:
        return _run(protocol_path, budget, phase, mode, student_seeds, stop)
    except InterruptedError as error:
        raise RuntimeError('Terminal bounded soft cell mass KL interruption; no automatic retry') from error
