"""Bounded validation pilots for a frozen source-graph assignment prior.

R is prepared once from the binary source random walk and native soft P0.
Only P is learned. Uniform inner/student CE and the original native Gaussian
factors, teacher, feature transform and serving recipes remain matched.
"""
import json
from pathlib import Path

import torch

from src.citation_factor_geometry import _pins, _sha, _source
from src.io import save_json, save_state

MODE = 'binary_rw_diffused_native_P0_v1'
OBJECTIVE = 'teacher_CE_plus_KL_current_to_frozen_graph_diffused_P0'


def _prepare(protocol_path, budget, stop):
    from src.frozen_graph_assignment_prior import build_prior_packet, packet_context
    from src.low_rank_assignment import initialize_factors, logit_block
    from src.moments import make_material

    packet, B, graph, _, _, _, z, q, assignment, _, options, digest = _source(protocol_path, budget, stop)
    folder = Path(B['prior_input_folder'])
    if folder.exists():
        raise ValueError('Graph prior PREPARE already exists; never overwrite or regenerate')
    folder.mkdir(parents=True)
    cells, rank = int(assignment.max()) + 1, options['assignment_rank']
    u, v = initialize_factors(assignment, cells, rank, options['factor_seed'])
    with torch.no_grad():
        chunks = []
        material = make_material(z, q)
        moments = material.new_zeros(cells, material.shape[1])
        for first in range(0, len(z), options['chunk_size']):
            if stop():
                raise InterruptedError('Graph prior preparation stopped')
            last = first + options['chunk_size']
            chunk = logit_block(u[first:last], v, assignment[first:last], .05).double().softmax(1)
            chunks.append(chunk)
            moments += chunk.T @ material[first:last] / len(z)
        p0 = torch.cat(chunks)
    original = torch.load(B['baseline_snapshot0'], map_location='cuda', weights_only=False)
    if not torch.equal(moments, original['moments']):
        raise ValueError('Native graph-prior P0 does not reproduce the original moments exactly')
    native = folder / 'native_reference_factors.pt'
    save_state(dict(u=u, v=v, data_digest=digest, factor_seed=options['factor_seed'], mixing=.05), native)
    refs = dict(data_digest=digest, factor_seed=options['factor_seed'], mixing=.05,
        chunk_size=options['chunk_size'], source=packet['source'],
        original_baseline_source=packet['original_baseline_source'],
        source_graph_scope='all original citation nodes in the original ROW dataset order; transductive unlabeled adjacency',
        teacher_T=B['baseline']['T'], baseline=B['baseline'],
        native_reference_factors=dict(path=str(native), sha256=_sha(native)),
        original_P0=dict(path=B['baseline_snapshot0'], sha256=_sha(B['baseline_snapshot0'])),
        source_inputs=dict(path=str(Path(B['family_root']) / f"inputs_{B['condensation_seed']}.pt"),
            sha256=_sha(Path(B['family_root']) / f"inputs_{B['condensation_seed']}.pt")),
        hard_assignment=dict(path=B['hard_assignment'], sha256=_sha(B['hard_assignment'])))
    frozen = build_prior_packet(graph['adj'].detach().cpu(), p0.detach().cpu(),
                                [u.detach().cpu(), v.detach().cpu()], refs)
    artifact = folder / 'frozen_graph_prior.pt'
    save_state(frozen, artifact)
    report = dict(schema=1, budget=budget, source=packet['source'],
        prior=dict(path=str(artifact), sha256=_sha(artifact), context=packet_context(frozen),
                   diagnostics=frozen['diagnostics']),
        native_reference_factors_path=str(native), native_reference_factors_sha256=_sha(native),
        native_factor_generation_calls=1, P0_moments_bit_exact_original=True,
        new_condensations=0, P_updates=0, student_fits=0, test_evaluations=0)
    save_json(report, folder / 'prepare_report.json')
    return report


def _run(protocol_path, budget, phase, mode, student_seeds, stop):
    from src.soft_ce_partition import optimize_ce_assignment
    from src.evaluation import fit_gcn_diagnostic
    from src.student_routes import replay_routes
    from src.sweep_utils import representative
    from src.frozen_graph_assignment_prior import load_frozen_prior
    from src.low_rank_assignment import logit_block

    if phase == 'prepare':
        if mode is not None or student_seeds:
            raise ValueError('Graph PREPARE never optimizes P or fits students')
        return _prepare(protocol_path, budget, stop)
    if phase not in ('qualify', 'science') or mode not in (MODE, 'native_gaussian'):
        raise ValueError('Unknown graph prior phase/arm')
    packet, B, graph, train, validation, h, z, q, assignment, transform, options, digest = _source(protocol_path, budget, stop)
    seeds = list(student_seeds)
    if seeds != ([] if phase == 'qualify' else B['student_seeds']):
        raise ValueError('Graph prior cohort differs from the frozen phase')
    steps = 1 if phase == 'qualify' else 25
    folder = Path(B['arm_folders'][mode])
    context = dict(schema=1, source=packet['source'], budget=budget, mode=mode,
        baseline=B['baseline'], condensation_seed=B['condensation_seed'], native_data_digest=digest,
        fixed_loss='uniform_inner_and_student_CE; free_P_mass',
        objective='original_teacher_CE' if mode == 'native_gaussian' else OBJECTIVE,
        prior=None if mode == 'native_gaussian' else B['prior'])
    context_path = folder / 'context.json'
    if context_path.exists():
        if json.loads(context_path.read_text()) != context:
            raise ValueError('Graph prior trajectory context changed')
    elif folder.exists():
        raise ValueError('Graph prior partial folder lacks its frozen context')
    else:
        folder.mkdir(parents=True)
        save_json(context, context_path)
    new_updates = 0
    if mode != 'native_gaussian':
        prior = B['prior']
        log_reference = load_frozen_prior(prior['path'], prior['sha256'],
            dict(nodes=len(assignment), cells=int(assignment.max()) + 1, rank=options['assignment_rank']),
            'cuda', expected_context=prior['context'])
        options.update(graph_assignment_kl_mode=MODE, graph_assignment_kl_artifact=prior['path'],
                       graph_assignment_kl_sha256=prior['sha256'], graph_assignment_kl_context=prior['context'])
        resume_path, manifest = folder / 'resume.pt', folder / 'checkpoint_manifest.json'
        if resume_path.exists():
            if not manifest.exists():
                raise ValueError('Graph prior resume lacks its checkpoint byte manifest')
            _pins(json.loads(manifest.read_text()))
            state = torch.load(resume_path, map_location='cpu', weights_only=False)
            if state['step'] > steps or state.get('graph_assignment_kl_context') != prior['context']:
                raise ValueError('Graph prior resume is outside its bound phase')
        else:
            state = None
        initial_step = 0 if state is None else state['step']
        if phase == 'qualify' and state is not None:
            raise ValueError('Graph qualification requires exactly one fresh P update')
        if phase == 'science':
            qualification = json.loads((folder / 'qualification.json').read_text())
            if not qualification['passed'] or state is None or state['step'] != 1:
                raise ValueError('Graph science requires its passed one-update prefix')
        optimize_ce_assignment(z, q, assignment, steps=steps, folder=folder, checkpoint_steps=(0, steps),
                               resume_state=state, stop=stop, **options)
        state = torch.load(resume_path, map_location='cpu', weights_only=False)
        new_updates = steps - initial_step
        paths = [resume_path, folder / 'optimization.csv', *sorted((folder / 'checkpoints').glob('step_*.pt'))]
        save_json({str(path): _sha(path) for path in paths}, manifest)
        if (state['step'] != steps or state.get('graph_assignment_kl_context') != prior['context']
                or new_updates != (1 if phase == 'qualify' else 24)):
            raise ValueError('Graph prior work/context differs from its frozen phase')
        p0 = torch.load(folder / 'checkpoints/step_000000.pt', map_location='cuda', weights_only=False)
        original = torch.load(B['baseline_snapshot0'], map_location='cuda', weights_only=False)
        native = torch.load(prior['context']['source_refs']['native_reference_factors']['path'],
                            map_location='cuda', weights_only=False)
        u0, v0 = p0['graph_assignment_kl_parameters']
        with torch.no_grad():
            logits0 = logit_block(u0, v0, assignment, .05).double()
            independently_evaluated_KL0 = float((logits0.softmax(1) * (logits0.log_softmax(1) - log_reference)).sum() / len(z))
        error = abs(independently_evaluated_KL0 - p0['graph_assignment_kl_KL'])
        passed = (u0.dtype == torch.float32 and v0.dtype == torch.float32
            and torch.equal(u0, native['u']) and torch.equal(v0, native['v'])
            and torch.equal(p0['moments'], original['moments']) and torch.equal(p0['theta'], original['theta'])
            and p0['teacher_ce'] == original['teacher_ce'] and p0['J_exact']
            and p0['graph_assignment_kl_scale'] == max(original['teacher_ce'], 1e-12)
            and independently_evaluated_KL0 > 0 and error <= 1e-12
            and p0['graph_assignment_kl_context'] == prior['context']
            and all(s.get('graph_assignment_kl_context') == prior['context'] for s in state['snapshots'].values()))
        if not passed:
            raise ValueError('Native Gaussian/P0/head/fixed graph-prior qualification failed')
        if phase == 'qualify':
            qualification = dict(schema=1, passed=True, budget=budget, mode=mode, source=packet['source'],
                prior_context=prior['context'], P0_moments_and_head_bit_exact_original=True,
                native_U0_V0_bit_exact=True, normalization='CE0_only', independently_evaluated_KL0=independently_evaluated_KL0,
                KL0_error=error, actual_new_P_updates=1, physical_student_fits=0, test_evaluations=0)
            save_json(qualification, folder / 'qualification.json')
            return qualification
        checkpoints = folder / 'checkpoints'
    else:
        if phase != 'science':
            raise ValueError('Original Gaussian anchor reuses pinned existing P0/P25')
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
                raise ValueError('Graph prior routes differ from the selected validation student')
            records.append(dict(step=step, **result, SGC_MLP_sameweights_val=routes['mlp_val_acc']))
    report = dict(schema=1, budget=budget, mode=mode, phase=phase, folder=str(folder), records=records,
        trajectory_P_updates=25, actual_new_P_updates=new_updates, new_physical_student_fits=len(records),
        same_selected_weight_validation_routes=2 * len(records), test_evaluations=0,
        interpretation='Current-to-frozen graph-diffused P0 KL; R is not graph TV and need not preserve cell masses')
    save_json(report, folder / 'science_report.json')
    return report


def run(protocol_path, protocol_sha256, budget, phase, mode=None, student_seeds=(), stop=lambda: False):
    try:
        if _sha(protocol_path) != protocol_sha256:
            raise ValueError('Frozen graph prior protocol bytes changed')
        return _run(protocol_path, budget, phase, mode, student_seeds, stop)
    except InterruptedError as error:
        raise RuntimeError('Terminal bounded graph-prior interruption; no automatic partial retry') from error
