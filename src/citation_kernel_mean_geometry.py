"""One new direct-H/RMS geometry admission, without heads or P updates."""
import json
import math
import time
from pathlib import Path

import torch

from src.citation_factor_geometry import _pins, _sha, _source
from src.citation_dual_head_ce import _native_parameters
from src.citation_graph_factor import _json_write, _state_write, _original_paths
from src.io import array_digest

KIND = 'original_native_NODE_P0_direct_H_RMS_no_head_geometry_admission_v1'
RAW_H = 'original_raw_H_centroid_moments_v1'
_BUDGET_CELLS = {'cora35': 35, 'cora140': 140}
LIMITS = dict(max_seconds=300, peak_allocated_bytes=4 * 1024**3, peak_reserved_bytes=6 * 1024**3)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _identity(value):
    # Exact historical BB descriptor grammar; only source-byte observations.
    return dict(shape=list(value.shape), dtype=str(value.dtype), tensor=array_digest(value.detach().cpu().numpy()))


def load_admission(packet, B, digest):
    ref = B['source_geometry_acceptance']
    _require(ref['kind'] == KIND and packet['readonly_files_sha256'].get(ref['path']) == ref['sha256'],
        'Typed new geometry admission is unpinned')
    report = json.loads(Path(ref['path']).read_text())
    error = report['affine_inverse_vs_direct_H_centroid_max_absolute']
    _require(report['kind'] == KIND and report['passed'] is True and report['budget'] == B['budget']
        and report['source'] == packet['source'] and report['native_data_digest'] == digest
        and type(error) in (int, float) and math.isfinite(error) and 0 <= error <= 1e-12
        and report['source_geometry_checked_before_any_candidate_head_or_P_update'] is True
        and report['original_BB_P0_and_serving_metadata_matched'] is True
        and report['raw_H_centroid_P0_serving_exact_original_physical_P0'] is True,
        'New no-head source/readout geometry admission failed')
    _require(report['original_files_sha256'] == packet['original_files_sha256']
        and {str(p) for p in _original_paths(B)} <= set(report['original_files_sha256'])
        and all(packet['readonly_files_sha256'].get(path) == pin
            for path, pin in report['original_files_sha256'].items())
        and set(report['budget_bindings']) == {'baseline_snapshot0','native_P0_factors',
            'centroid_Nystrom_checkpoint0','centroid_Nystrom_checkpoint25','centroid_control_representation'}
        and report['budget_bindings'] == {key:B[key] for key in report['budget_bindings']},
        'Geometry origin source/control pins changed')
    return dict(acceptance=ref, report=dict(path=ref['path'], sha256=ref['sha256']), original_residual=error,
        original_RMS_geometry_metadata_reused=True, RMS_geometry_numeric_replays=0)


def run(protocol_path, protocol_sha256, budget, stop=lambda: False):
    from src.low_rank_assignment import logit_block
    from src.moments import decode_moments
    from src.research_loop import implementation_provenance
    from src.sweep_utils import representative

    _require(_sha(protocol_path) == protocol_sha256, 'Geometry protocol bytes changed')
    declaration = json.loads(Path(protocol_path).read_text()); B = declaration['budget_packets'][budget]
    _require(declaration['kind'] == KIND and declaration['geometry_resource_limits'] == LIMITS
        and declaration['geometry_absolute_bound'] == 1e-12 and B['dataset'] == 'cora'
        and B['budget'] in _BUDGET_CELLS and B['budget'] == budget and B['condensation_seed'] == 0
        and B['centroid_control_representation'] == RAW_H, 'Unknown prospective geometry domain')
    folder = Path(B['geometry_folder']); _require(not folder.exists(), 'Fresh geometry namespace required')
    folder.mkdir(parents=True); started = time.monotonic(); arrays = {}; progress = dict(schema=1, kind=KIND,
        budget=budget, source=declaration['source'], protocol=dict(path=str(protocol_path), sha256=protocol_sha256),
        passed=False, old_head_or_FD_replays=0, head_solves=0, P_updates=0, optimizer_steps=0, student_fits=0,
        native_factor_generations=0, teacher_map_Phi_fits=0, original_M0_remomentizations=0,
        new_direct_H_probability_blocks=0, new_direct_H_numerator_GEMMs=0,
        source_SGC_loader_work='original _source/_prepare_dataset; uninstrumented, not claimed zero')
    error = None
    def guard():
        interrupted = stop() or time.monotonic() - started > LIMITS['max_seconds']
        if torch.cuda.is_initialized():
            interrupted |= torch.cuda.max_memory_allocated() > LIMITS['peak_allocated_bytes']
            interrupted |= torch.cuda.max_memory_reserved() > LIMITS['peak_reserved_bytes']
        if interrupted:
            raise RuntimeError('Terminal bounded source-geometry interruption')
        return False
    try:
        torch.cuda.init(); torch.cuda.reset_peak_memory_stats(); guard()
        packet, B, _, _, _, h, z, q, assignment, transform, options, digest = _source(protocol_path, budget, guard)
        certref = B['original_BB_certificate']; closeref = B['original_BB_closure']
        _require(all(packet['readonly_files_sha256'].get(r['path']) == r['sha256'] for r in (certref, closeref)),
            'Original accepted BB certificate/closure unpinned')
        cert = json.loads(Path(certref['path']).read_text()); old = cert['roots'][B['original_BB_root']]
        closure = json.loads(Path(closeref['path']).read_text())
        cases = [case for case in closure['per_case'] if case['root'] == B['original_BB_root']]
        _require(closure['passed'] is True and closure['classification'] ==
            'fixed_Cora_ROW35_70_140_original_source_material_linear_readout_certified'
            and any(parent.get('path') == certref['path'] and parent.get('sha256') == certref['sha256']
                for parent in closure['parents']) and len(cases) == 1
            and cases[0]['cells'] == _BUDGET_CELLS[budget] and cases[0]['reference_id'] == B['baseline_id']
            and cases[0]['recipe_id'] == B['recipe_id'] and cases[0]['source_P0_linear_certificate_passed'] is True
            and cases[0]['current_M0_bitwise_equal_cached_NODE0'] is True
            and cases[0]['actual_FP32_X_Q_F64_uniform_equal'] is True
            and cases[0]['original_source_context'] == old['source_context'], 'BB root acceptance closure differs')
        _require(cert['passed'] is True and old['passed'] is True and old['source_P0_linear_certificate_passed'] is True
            and old['reference']['current_native_M0_bitwise_equal_cached_NODE0'] is True
            and old['reference']['actual_FP32_P0_X_Q_uniform_equal_reference'] is True, 'BB source/P0 not accepted')
        _require(all(packet['readonly_files_sha256'].get(str(Path(B['family_root']) / name)) == pin
            for name, pin in old['source_context']['assets'].items()), 'BB source asset bytes differ')
        _, initial, _ = _native_parameters(packet, B, z, assignment, options, digest)
        original = torch.load(B['baseline_snapshot0'], map_location='cpu', weights_only=False); M0 = original['moments']
        _require([_identity(v) for v in initial] == old['reference']['P0_parameters']
            and _identity(M0) == old['reference']['P0_moments']
            and all(_identity(v) == old['source_context'][key] for key, v in
                (('h', h), ('z', z), ('q', q), ('hard_assignment', assignment))), 'BB source/native/M0 identities differ')
        for key in ('center', 'output_center', 'scale'):
            _require(_identity(getattr(transform, key)) == old['source_context']['transform'][key], 'RMS bytes changed')
        _require(transform.kind == 'rms' and transform.matrix is None and transform.eps == 1e-12
            and bool(torch.isfinite(M0).all()) and bool((M0[:, 0] > 0).all())
            and bool(torch.isfinite(transform.scale)) and bool(transform.scale > 0), 'Invalid original RMS/mass')
        x, y, mass = representative(M0, transform, z.shape[1], z.device); weights = torch.full_like(mass, 1/len(mass))
        arrays.update(original_physical_X=x.detach().cpu().clone(), original_physical_Q=y.detach().cpu().clone(),
            original_uniform_weights=weights.detach().cpu().clone())
        _require([_identity(v) for v in (x, y, weights)] == old['reference']['student_inputs'], 'Original P0 serving changed')
        progress['original_BB_P0_and_serving_metadata_matched'] = True
        cpu0 = B['centroid_Nystrom_checkpoint0']; _require(packet['readonly_files_sha256'].get(cpu0['path']) == cpu0['sha256'], 'Raw-H control P0 unpinned')
        raw = torch.load(cpu0['path'], map_location=z.device, weights_only=False)
        _require(type(raw['step']) is int and raw['step'] == 0, 'Raw-H control initial step differs')
        rx, ry, rm = decode_moments(raw['moments'], h.shape[1]); rx, ry = rx.float(), ry.float()
        arrays.update(raw_H_control_P0_X=rx.detach().cpu().clone(), raw_H_control_P0_Q=ry.detach().cpu().clone(),
            raw_H_control_P0_uniform_weights=torch.full_like(rm, 1/len(rm)).detach().cpu().clone())
        _require([_identity(v) for v in (rx, ry, torch.full_like(rm, 1/len(rm)))] ==
            old['reference']['student_inputs'], 'Raw-H centroid control P0 serving differs')
        progress['raw_H_centroid_P0_serving_exact_original_physical_P0'] = True
        u, v = [p.detach().to(z.device).clone() for p in initial]; m = M0[:, 0].to(z.device)
        numerator = z.new_zeros(len(v), h.shape[1]); guard()
        with torch.no_grad():
            for begin in range(0, len(z), options['chunk_size']):
                guard(); end = min(begin + options['chunk_size'], len(z))
                probability = logit_block(u[begin:end], v, assignment[begin:end], .05).double().softmax(1)
                progress['new_direct_H_probability_blocks'] += 1
                arrays['last_returned_probability_block'] = probability.detach().cpu().clone()
                progress['last_returned_probability_rows'] = [begin, end]; guard()
                numerator += probability.T @ h[begin:end].double() / len(z)
                progress['new_direct_H_numerator_GEMMs'] += 1
                arrays['new_direct_H_numerator'] = numerator.detach().cpu().clone()
                progress['completed_numerator_row_end'] = end
            physical = M0[:, 1:1 + z.shape[1]].to(z.device) / m[:, None]
            physical = physical * transform.scale + transform.output_center + transform.center
            direct = numerator / m[:, None]; residual = physical - direct
        arrays.update(original_M0=M0.detach().cpu().clone(), new_direct_H_numerator=numerator.detach().cpu().clone(),
            physical_RMS_centroids=physical.detach().cpu().clone(), new_direct_H_centroids=direct.detach().cpu().clone(),
            new_geometry_residual=residual.detach().cpu().clone())
        guard(); maximum = float(residual.abs().max())
        progress.update(native_data_digest=digest, affine_inverse_vs_direct_H_centroid_max_absolute=maximum if math.isfinite(maximum) else 'nonfinite',
            source_geometry_checked_before_any_candidate_head_or_P_update=True,
            original_files_sha256=dict(packet['original_files_sha256']),
            budget_bindings={key:B[key] for key in ('baseline_snapshot0','native_P0_factors',
                'centroid_Nystrom_checkpoint0','centroid_Nystrom_checkpoint25','centroid_control_representation')})
        _require(bool(torch.isfinite(residual).all()) and math.isfinite(maximum) and maximum <= 1e-12,
            'NEW direct-H/RMS geometry outside frozen bound')
        _pins(packet['readonly_files_sha256']); _require(implementation_provenance() == packet['source'], 'Source changed'); guard()
        progress['passed'] = True
    except Exception as exc:
        error = exc; progress['error'] = dict(type=type(exc).__name__, message=str(exc))
    finally:
        if arrays:
            path = folder/'new_geometry_arrays.pt'; _state_write(arrays, path)
            progress['new_geometry_arrays'] = dict(path=str(path), sha256=_sha(path), raw_write_complete=True,
                geometry_passed=progress['passed'])
        progress.update(seconds=time.monotonic()-started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0,
            peak_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0)
        _json_write(progress, folder/'geometry_report.json')
    if error is not None:
        raise RuntimeError('Terminal original source-geometry admission failure; preserve evidence') from error
    return progress
