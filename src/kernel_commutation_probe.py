"""Read-only kernel feature/weighted-mean commutation diagnostics.

No head, assignment or student is fitted. Native P0/P25 factors and the
original shared H, map and phi caches are required and never recreated.
"""
import math
from pathlib import Path

import numpy as np
import torch

from src.citation_factor_geometry import _pins, _sha, _source
from src.io import save_json, save_state
from src.moments import initial_logits
from src.nystrom_ce import NystromMap, _cache_identity, _metadata_path, _validate_phi
from src.shared_features import _load_state, _map_identity, _validate_map_state


@torch.no_grad()
def summarize(probability, source_phi, mapped_centroids):
    """Uniform-cell squared gap, scaled by fixed original-node phi energy."""
    values = (probability, source_phi, mapped_centroids)
    if any(not torch.is_tensor(x) or x.ndim != 2 or min(x.shape) < 1
           or x.dtype != torch.float64 or x.requires_grad
           or not bool(torch.isfinite(x).all()) for x in values):
        raise ValueError('Require finite detached FP64 matrices')
    if len({x.device for x in values}) != 1:
        raise ValueError('Diagnostic matrices must share a device')
    if (len(probability) != len(source_phi)
            or mapped_centroids.shape != (probability.shape[1], source_phi.shape[1])):
        raise ValueError('Source/cell/feature dimensions differ')
    if bool((probability < 0).any()) or float((probability.sum(1) - 1).abs().max()) > 1e-12:
        raise ValueError('Require source-row-stochastic probabilities')
    mass = probability.sum(0)
    energy = source_phi.square().sum(1).mean()
    if bool((mass <= 0).any()) or not float(energy) > 0:
        raise ValueError('Require positive cell mass and source feature energy')
    mean_phi = probability.T @ source_phi / mass[:, None]
    gap2 = (mapped_centroids - mean_phi).square().sum(1)
    variance = (source_phi - source_phi.mean(0)).square().sum(1).mean()
    return dict(uniform_cell_gap_squared=float(gap2.mean()),
                source_mass_weighted_gap_squared=float((gap2 * mass / len(probability)).sum()),
                fixed_source_phi_energy=float(energy),
                fixed_source_phi_variance=float(variance),
                relative_gap_squared=float(gap2.mean() / energy),
                gap_over_source_phi_variance=float(gap2.mean() / variance) if float(variance) > 0 else None,
                cell_mass_min=float(mass.min() / len(probability)),
                cell_mass_max=float(mass.max() / len(probability)),
                per_cell_gap_squared=gap2.cpu().tolist()), mean_phi, mass


@torch.no_grad()
def _native_probability(parameters, assignment, cells, chunk, stop):
    u, v = parameters
    if (u.dtype != torch.float32 or v.dtype != torch.float32
            or u.shape != (len(assignment), v.shape[1]) or v.shape[0] != cells
            or not bool(torch.isfinite(u).all()) or not bool(torch.isfinite(v).all())):
        raise ValueError('Require the pinned native FP32 factor geometry')
    rows = []
    for first in range(0, len(u), chunk):
        if stop():
            raise InterruptedError('Commutation probe interrupted')
        last = first + chunk
        prior = initial_logits(assignment[first:last], cells, .05, dtype=u.dtype)
        logits = prior + u[first:last] @ v.T / math.sqrt(u.shape[1])
        rows.append(logits.double().softmax(1))
    return torch.cat(rows)


@torch.no_grad()
def run(protocol_path, protocol_sha256, budget, stop=lambda: False):
    path = Path(protocol_path)
    if _sha(path) != protocol_sha256:
        raise ValueError('Commutation protocol bytes differ')
    packet, B, _, _, _, h, z, q, assignment, _, options, digest = _source(path, budget, stop)
    if packet['kind'] != 'frozen_kernel_feature_commutation_gap_diagnostic_v1':
        raise ValueError('Wrong diagnostic protocol')
    folder = Path(B['diagnostic_folder'])
    if folder.exists():
        raise ValueError('Diagnostic output exists; preserve it, never rerun')
    native_ref = B['native_P0_factors']
    _pins({native_ref['path']: native_ref['sha256']})
    native = torch.load(native_ref['path'], map_location='cuda', weights_only=False)
    if (native['data_digest'] != digest or native['factor_seed'] != B['condensation_seed']
            or native['mixing'] != .05 or bool(native['u'].ne(0).any())):
        raise ValueError('Pinned P0 is not this original native zeroU/GaussianV source')
    terminal = torch.load(Path(B['baseline_folder']) / 'resume.pt', map_location='cuda', weights_only=False)
    if terminal['step'] != 25 or terminal['config']['data_digest'] != digest:
        raise ValueError('Exact native terminal P25 parameters are required')
    family = Path(B['family_root'])
    map_path = family / 'nystrom_map_schema3.pt'
    phi_path = family / 'nystrom_phi_schema3.npy'
    # Existing cache paths must be explicit immutable pins. Never call a cache
    # creation path or fall back to recomputing an untracked feature cache.
    for source_path in (map_path, phi_path, _metadata_path(phi_path)):
        if str(source_path) not in packet['readonly_files_sha256'] or not source_path.exists():
            raise ValueError('Frozen existing map/phi/sidecar pins are mandatory')
    map_state = _load_state(map_path, 'shared_nystrom_map')
    anchors, mapping = _validate_map_state(map_state, _map_identity(h, 3000, 0, 'relu'))
    feature_map = NystromMap(anchors.to('cuda'), mapping.to('cuda'), 'relu')
    phi = np.load(phi_path, mmap_mode='r')
    identity = _cache_identity(h, feature_map, tuple(phi.shape), 2048, stop)
    phi_digest = _validate_phi(h, feature_map, phi, identity, 2048, stop)
    source_phi = torch.from_numpy(np.array(phi)).to(device='cuda', dtype=torch.float64)
    h64 = h.double()
    cells = int(assignment.max()) + 1
    if source_phi.shape[0] != len(h64) or h64.shape != z.shape:
        raise ValueError('Source H/phi dimensions differ from the original node coordinates')
    folder.mkdir(parents=True)
    reports, artifact_pins = {}, {}
    for step, parameters in ((0, [native['u'], native['v']]), (25, terminal['parameters'])):
        probability = _native_probability(parameters, assignment, cells, options['chunk_size'], stop)
        mass = probability.sum(0)
        centroids = probability.T @ h64 / mass[:, None]
        mapped = feature_map(centroids).double()
        summary, mean_phi, mass = summarize(probability, source_phi, mapped)
        mean_q = probability.T @ q.double() / mass[:, None]
        entropy = -(mean_q * torch.where(mean_q > 0, mean_q.log(), torch.zeros_like(mean_q))).sum(1)
        summary['per_cell_teacher_Q_entropy'] = entropy.cpu().tolist()
        snapshot_path = Path(B['baseline_folder']) / 'checkpoints' / f'step_{step:06d}.pt'
        snapshot = torch.load(snapshot_path, map_location='cpu', weights_only=False)
        summary['saved_native_teacher_CE'] = snapshot['teacher_ce']
        reports[str(step)] = summary
        artifact = folder / f'step_{step:06d}_feature_gap.pt'
        save_state(dict(step=step, centroids=centroids.cpu(), mapped_centroids=mapped.cpu(),
                        mean_source_phi=mean_phi.cpu(), mean_source_q=mean_q.cpu(), source_cell_mass=mass.cpu(),
                        summary=summary, source_phi_digest=phi_digest), artifact)
        artifact_pins[str(artifact)] = _sha(artifact)
    # No old head/moment replay domain, validation score, label metric or test
    # is recomputed by this diagnostic. This is a new feature-moment domain.
    _pins(packet['readonly_files_sha256'])
    report = dict(schema=1, budget=budget, kind=packet['kind'], source=packet['source'],
                  original_baseline_source=packet['original_baseline_source'],
                  source_phi_digest=phi_digest, checkpoints=reports,
                  P25_minus_P0_relative_gap=reports['25']['relative_gap_squared'] - reports['0']['relative_gap_squared'],
                  artifact_files_sha256=artifact_pins, source_inputs_unchanged=True,
                  native_factor_generation_calls=0, kernel_Gram_refits=0, phi_rebuilds=0,
                  new_condensations=0, P_updates=0, head_fits=0, student_fits=0,
                  validation_evaluations=0, test_evaluations=0,
                  interpretation='Vector feature-map/averaging commutation gap; no signed Jensen inequality or performance benefit follows from a nonzero gap.')
    save_json(report, folder / 'report.json')
    return report
