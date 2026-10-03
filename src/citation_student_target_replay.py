"""Validation-only target calibration on immutable P-derived condensates.

This student recipe comparison never optimizes P or refits the teacher. Squared
targets are normalized per cell and inherit all information from its saved P.
"""
import hashlib
import json
from pathlib import Path

import torch

from src.io import array_digest, save_json


def calibrated_targets(labels, policy):
    if (labels.ndim != 2 or not labels.is_floating_point() or min(labels.shape) < 1
            or not bool(torch.isfinite(labels).all()) or bool((labels < 0).any())
            or not torch.allclose(labels.sum(1), torch.ones_like(labels[:, 0]), atol=1e-6, rtol=0)):
        raise ValueError("Require normalized nonnegative P-derived cell targets")
    if policy == 'original':
        return labels
    if policy == 'square_normalized':
        squared = labels.square()
        return squared / squared.sum(1, keepdim=True)
    raise ValueError("Unknown frozen target policy")


def _sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _run(protocol_path, budget, policy, stop):
    from src.research_loop import implementation_provenance
    from src.citation_search import dataset_digest, fixed_propagated_features
    from src.data import _prepare_dataset
    from src.evaluation import fit_gcn_diagnostic
    from src.student_routes import replay_routes
    from src.sweep_utils import representative
    from src.target_refinement import training_refined_targets
    from src.transforms import FeatureTransform

    packet = json.loads(Path(protocol_path).read_text())
    if packet['source'] != implementation_provenance() or policy not in packet['policies']:
        raise ValueError("Target replay source or frozen policy changed")
    for path, pin in packet['readonly_files_sha256'].items():
        if _sha(path) != pin:
            raise ValueError(f"Fixed target replay input changed: {path}")
    B = packet['budget_packets'][budget]
    if B['recipe']['input_scale'] != 1.0:
        raise ValueError("Target replay requires original input scale")
    if stop():
        raise InterruptedError("Target replay stopped before loading")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    family, trial = Path(B['family_root']), Path(B['trial_folder'])
    graph, train, validation, testing, current_h = _prepare_dataset(B['dataset'], packet['data_dir'], 'cuda', 'row')
    config = json.loads((family / 'config.json').read_text())
    if (config.get('teacher_backend') is not None or config.get('teacher_kernel') not in (None, 'relu')
            or config.get('citation_features') != 'row'
            or dataset_digest(graph, train, validation, testing) != config['data_digest']):
        raise ValueError("Target replay original ROW ReLU source changed")
    h = fixed_propagated_features(current_h, config, family)
    inputs = torch.load(family / f"inputs_{B['condensation_seed']}.pt", map_location='cuda', weights_only=False)
    transform = FeatureTransform(**inputs['transform'])
    logits = torch.load(family / 'teacher.pt', map_location='cuda', weights_only=False)['logits']
    q = training_refined_targets(logits, B['candidate']['T'], graph['y'], train, 0.0)
    folder = Path(packet['output_root']) / budget / policy / f"condensation_{B['condensation_seed']}"
    folder.mkdir(parents=True, exist_ok=True)
    context = dict(protocol_sha256=_sha(protocol_path), budget=budget, policy=policy,
                   source=packet['source'], recipe=B['recipe'], checkpoints=B['checkpoints'])
    context_path = folder / 'context.json'
    if context_path.exists() and json.loads(context_path.read_text()) != context:
        raise ValueError("Target replay output context changed")
    save_json(context, context_path)
    settings = {key: value for key, value in B['recipe'].items() if key != 'input_scale'}
    records, new_fits, target_inputs = [], 0, {}
    for step in B['checkpoints']:
        snapshot = torch.load(trial / 'checkpoints' / f'step_{step:06d}.pt', map_location='cuda', weights_only=False)
        x, original, mass = representative(snapshot['moments'], transform, inputs['z'].shape[1], 'cuda')
        target = calibrated_targets(original, policy)
        target_inputs[str(step)] = dict(original=array_digest(original.cpu().numpy()),
            calibrated=array_digest(target.cpu().numpy()),
            entropy_original=float(-(original * original.clamp_min(1e-30).log()).sum(1).mean()),
            entropy_calibrated=float(-(target * target.clamp_min(1e-30).log()).sum(1).mean()))
        for seed in B['student_seeds']:
            evaluation = folder / 'validation' / f'step_{step}'
            cached = (evaluation / f'seed_{seed}.json').exists()
            result = fit_gcn_diagnostic(x, target, torch.full_like(mass, 1 / len(mass)), graph, q,
                dict(train=train, val=validation[1]), seed, folder=evaluation, stop=stop, **settings)
            routes = replay_routes(evaluation / f'seed_{seed}_selected.pt', graph, h.float(),
                dict(val=validation[1]), settings, evaluation / f'seed_{seed}_validation_routes_v1.json', seed=seed, stop=stop)
            if (any('test_' in key for key in result | routes) or routes['epoch'] != result['epoch']
                    or routes['gcn_val_acc'] != result['val_acc']):
                raise ValueError("Target replay selected validation routes disagree")
            new_fits += int(not cached)
            records.append(dict(step=step, **result, SGC_MLP_sameweights_val=routes['mlp_val_acc']))
    report = dict(schema=1, budget=budget, policy=policy, folder=str(folder), records=records,
        target_inputs=target_inputs, physical_new_GCN_fits=new_fits, actual_new_P_updates=0,
        validation_only=True, test_evaluations=0, independently_fit_raw_X_MLP=False,
        interpretation='Student target calibration only; no claim of improved learned clustering')
    save_json(report, folder / 'report.json')
    return report


def run(protocol_path, budget, policy, stop=lambda: False):
    try:
        return _run(protocol_path, budget, policy, stop)
    except InterruptedError as error:
        raise RuntimeError("Terminal bounded target replay interruption; no automatic partial retry") from error
