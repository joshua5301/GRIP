"""Training-label outer CE with original teacher-Q clustering inner targets.

This adapter uses the existing exact implicit head and node factors. Its owned
folder is separate from ordinary teacher-CE candidates and it never fits a
teacher or changes cached source inputs. Evaluation remains validation only.
"""
import hashlib
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from src.io import array_digest, save_json, save_state


def hard_training_outer(q, labels, train):
    """Return a detached train mask and targets, indexing only train labels."""
    if (q.ndim != 2 or not q.is_floating_point() or min(q.shape) < 1
            or not bool(torch.isfinite(q).all()) or bool((q < 0).any())
            or not torch.allclose(q.sum(1), torch.ones_like(q[:, 0]), atol=1e-6, rtol=0)):
        raise ValueError("Require finite node-aligned teacher probabilities")
    if (train.dtype != torch.bool or train.shape != (len(q),) or not bool(train.any())
            or labels.shape != (len(q),) or labels.dtype not in (torch.int32, torch.int64)):
        raise ValueError("Require nonempty boolean train mask and integer labels")
    selected = labels[train.to(labels.device)].to(device=q.device, dtype=torch.long)
    if bool((selected < 0).any()) or bool((selected >= q.shape[1]).any()):
        raise ValueError("Training labels must index teacher-provided classes")
    mask = train.detach().to(q.device).clone()
    target = q.detach().clone()
    target[mask] = F.one_hot(selected, num_classes=q.shape[1]).to(q)
    return mask, target


def balanced_source_training_outer(q, labels, train):
    """Exactly coalesce teacher CE plus fixed repeated train hard-label CE.

    R=ceil(N/T). Expanding all N source rows plus R copies of each train row
    and blending only train targets yields (sum_source_CE + R*sum_train_CE)
    /(N+R*T). Inner targets and original source features remain unchanged.
    """
    mask, hard = hard_training_outer(q, labels, train)
    repeats = math.ceil(len(q) / int(mask.sum()))
    target = q.detach().clone()
    target[mask] = (q.detach()[mask] + repeats * hard[mask]) / (repeats + 1)
    indices = torch.cat((torch.arange(len(q), device=q.device), mask.nonzero().flatten().repeat(repeats)))
    return indices, target, repeats


def core_options(candidate, seed, train, target):
    allowed = {'method', 'width', 'lr', 'T', 'rank', 'penalty', 'initialization', 'alpha', 'inner_loss_weighting'}
    if (set(candidate) != allowed or candidate['method'] != 'low_rank' or candidate['width'] != 0
            or candidate['inner_loss_weighting'] != 'uniform' or candidate['initialization'] != 'teacher_balanced'
            or type(candidate['rank']) is not int or candidate['rank'] < 1 or type(seed) is not int
            or any(isinstance(candidate[key], bool) or not math.isfinite(candidate[key]) or candidate[key] <= 0
                   for key in ('lr', 'T', 'penalty'))):
        raise ValueError("Training outer pilot requires the original uniform/free balanced NODE controls")
    return dict(penalty=candidate['penalty'], lr=candidate['lr'], mixing=.05,
        assignment_rank=candidate['rank'], factor_seed=seed, assignment_input='node', assignment_encoder='linear',
        encoder_hidden=64, solver_mode='exact', inner_method='newton_first', implicit_warm_start=True,
        mass_mode='free', inner_loss_weighting='uniform', inner_max_iter=2000, inner_tol=1e-7,
        cg_max_iter=512, cg_rtol=1e-6, cache_assignment=False, save_resume=True, save_assignment=False,
        outer_indices=train, outer_targets=target)


def validate_resume(state, z, q, assignment, train, target, steps):
    expected = dict(data_digest=array_digest(z.cpu().numpy(), q.cpu().numpy(), assignment.cpu().numpy()),
                    outer_digest=array_digest(train.cpu().numpy()), outer_targets_digest=array_digest(target.cpu().numpy()))
    if (type(state.get('step')) is not int or state['step'] > steps
            or any(state.get('config', {}).get(key) != value for key, value in expected.items())):
        raise ValueError("Training outer resume lost its source/target/mask binding")


def _sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _run(protocol_path, budget, steps, student_seeds, stop):
    from src.research_loop import implementation_provenance
    from src.citation_search import dataset_digest, fixed_propagated_features
    from src.data import _prepare_dataset
    from src.evaluation import fit_gcn_diagnostic
    from src.soft_ce_partition import optimize_ce_assignment
    from src.student_routes import replay_routes
    from src.sweep_utils import representative
    from src.target_refinement import training_refined_targets
    from src.transforms import FeatureTransform

    packet = json.loads(Path(protocol_path).read_text())
    if packet['source'] != implementation_provenance() or steps not in (1, 25):
        raise ValueError("Training outer source/protocol changed")
    for path, pin in packet['readonly_files_sha256'].items():
        if _sha(path) != pin:
            raise ValueError(f"Training outer source input changed: {path}")
    B = packet['budget_packets'][budget]
    condensation_seed = B['condensation_seed']
    if type(condensation_seed) is not int or condensation_seed < 0:
        raise ValueError("Training outer condensation seed must be a nonnegative integer")
    if B['recipe']['input_scale'] != 1.0:
        raise ValueError("Training outer pilot requires the original input scale")
    if list(student_seeds) != ([] if steps == 1 else B['student_seeds']):
        raise ValueError("Training outer cohort differs from its frozen phase")
    if stop():
        raise InterruptedError("Training outer stopped before source loading")
    family, folder = Path(B['family_root']), Path(B['outer_folder'])
    device = 'cuda'
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    graph, train, validation, testing, current_h = _prepare_dataset(B['dataset'], packet['data_dir'], device, 'row')
    config = json.loads((family / 'config.json').read_text())
    if (config.get('teacher_backend') is not None or config.get('teacher_kernel') not in (None, 'relu')
            or config.get('citation_features') != 'row' or config.get('mixing') != .05):
        raise ValueError("Training outer requires the original ROW ReLU teacher context")
    if dataset_digest(graph, train, validation, testing) != config['data_digest']:
        raise ValueError("Training outer graph/splits differ from original source")
    h = fixed_propagated_features(current_h, config, family)
    inputs = torch.load(family / f'inputs_{condensation_seed}.pt', map_location=device, weights_only=False)
    z = inputs['z']
    transform = FeatureTransform(**inputs['transform'])
    assignment = torch.load(B['hard_assignment'], map_location=device, weights_only=False)
    logits = torch.load(family / 'teacher.pt', map_location=device, weights_only=False)['logits'].to(device)
    q = training_refined_targets(logits, B['candidate']['T'], graph['y'], train, 0.0)
    policy = B.get('outer_policy', 'train_hard_v1')
    if policy == 'balanced_source_train_v1':
        mask, target, repeats = balanced_source_training_outer(q, graph['y'], train)
        objective = 'mean_coalesced_source_teacher_CE_plus_fixed_repeated_train_hard_CE'
    elif policy == 'train_hard_v1':
        mask, target = hard_training_outer(q, graph['y'], train)
        repeats = None
        objective = 'mean_hard_CE_on_original_training_nodes_only'
    else:
        raise ValueError("Unknown training outer policy")
    options = core_options(B['candidate'], condensation_seed, mask, target)
    context = dict(schema=1, objective=objective,
        inner_targets='original_all_node_teacher_Q; labels_and_features_derived_from_P',
        source=packet['source'], candidate=B['candidate'], condensation_seed=condensation_seed, train_nodes=int(train.sum()),
        inputs=dict(data_digest=array_digest(z.cpu().numpy(), q.cpu().numpy(), assignment.cpu().numpy()),
                    outer_digest=array_digest(mask.cpu().numpy()), outer_targets_digest=array_digest(target.cpu().numpy())))
    if repeats is not None:
        context.update(outer_policy=policy, repeats=repeats, outer_rows=len(mask))
    context_path, manifest_path = folder / 'context.json', folder / 'checkpoint_manifest.json'
    if context_path.exists():
        if json.loads(context_path.read_text()) != context or not manifest_path.exists():
            raise ValueError("Training outer context changed or completion manifest is missing")
        for path, pin in json.loads(manifest_path.read_text()).items():
            if _sha(path) != pin:
                raise ValueError("Training outer saved trajectory changed")
    elif folder.exists():
        raise ValueError("Training outer partial folder lacks its original context")
    else:
        folder.mkdir(parents=True)
        save_json(context, context_path)
        if repeats is not None:
            # Persist native teacher Q bits for the independent mixed-CE audit.
            # Ground-truth storage contains training labels only.
            save_state(dict(q=q, indices=mask, targets=target, train=train,
                train_labels=graph['y'][train], repeats=repeats), folder / 'outer_inputs.pt')
    if repeats is not None:
        saved_outer = torch.load(folder / 'outer_inputs.pt', map_location=device, weights_only=False)
        if (not torch.equal(saved_outer['q'], q) or not torch.equal(saved_outer['indices'], mask)
                or not torch.equal(saved_outer['targets'], target) or not torch.equal(saved_outer['train'], train)
                or not torch.equal(saved_outer['train_labels'], graph['y'][train]) or saved_outer['repeats'] != repeats):
            raise ValueError("Balanced training outer persisted inputs changed")
    resume_path = folder / 'resume.pt'
    state = torch.load(resume_path, map_location='cpu', weights_only=False) if resume_path.exists() else None
    initial_step = 0 if state is None else state['step']
    if state is not None:
        validate_resume(state, z, q, assignment, mask, target, steps)
    if state is None or state['step'] < steps:
        optimize_ce_assignment(z, q, assignment, steps=steps, folder=folder, checkpoint_steps=(0, steps),
                               resume_state=state, stop=stop, **options)
        state = torch.load(resume_path, map_location='cpu', weights_only=False)
        validate_resume(state, z, q, assignment, mask, target, steps)
        paths = [resume_path, folder / 'optimization.csv', *sorted((folder / 'checkpoints').glob('step_*.pt'))]
        if repeats is not None:
            paths.append(folder / 'outer_inputs.pt')
        save_json({str(path): _sha(path) for path in paths}, manifest_path)
    if state['step'] != steps:
        raise ValueError("Training outer did not reach its frozen endpoint")
    records = []
    settings = {key: value for key, value in B['recipe'].items() if key != 'input_scale'}
    for step in (0, steps):
        snapshot = torch.load(folder / 'checkpoints' / f'step_{step:06d}.pt', map_location=device, weights_only=False)
        x, y, mass = representative(snapshot['moments'], transform, z.shape[1], device)
        for seed in student_seeds:
            evaluation = folder / 'validation' / f"step_{step}_{B['recipe_id']}"
            result = fit_gcn_diagnostic(x, y, torch.full_like(mass, 1 / len(mass)), graph, q,
                dict(train=train, val=validation[1]), seed, folder=evaluation, stop=stop, **settings)
            routes = replay_routes(evaluation / f'seed_{seed}_selected.pt', graph, h.float(),
                dict(val=validation[1]), settings, evaluation / f'seed_{seed}_validation_routes_v1.json', seed=seed, stop=stop)
            if (any('test_' in key for key in result | routes) or routes['epoch'] != result['epoch']
                    or routes['gcn_val_acc'] != result['val_acc']):
                raise ValueError("Training outer validation routes differ from the selected student")
            records.append(dict(step=step, **result, SGC_MLP_sameweights_val=routes['mlp_val_acc']))
    report = dict(schema=1, budget=budget, folder=str(folder), steps=steps,
        train_nodes=int(train.sum()), records=records, validation_only=True,
        actual_new_P_updates=steps - initial_step,
        objective=context['objective'], synthetic_targets_remain_teacher_Q=True, test_evaluations=0)
    report['trajectory_P_updates'] = steps
    save_json(report, folder / f'report_step_{steps}.json')
    return report


def run(protocol_path, budget, steps, student_seeds, stop=lambda: False):
    try:
        return _run(protocol_path, budget, steps, student_seeds, stop)
    except InterruptedError as error:
        raise RuntimeError("Terminal bounded training-outer interruption; no automatic partial retry") from error
