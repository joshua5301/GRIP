"""Fresh seed-specific origin and cold matched controls; original solvers unchanged.

BB certifies common source coordinates only. This module never borrows its
condensation-zero factors, moments, head, readout or acceptance for a new seed.
"""
import json
import math
import time
from pathlib import Path

import torch

from src.citation_factor_geometry import _sha, _pins
from src.citation_graph_factor import _json_write, _state_write
from src.io import array_digest, cpu_state, _fingerprint
from src.moments import make_material, decode_moments
from src.shared_features import _load_state, _tensor_identity
from src.sweep_utils import representative

KIND = 'seed_specific_original_NODE_P0_direct_H_RMS_no_head_capture_v1'
RAW_H = 'original_raw_H_centroid_moments_v1'
LIMITS = dict(max_seconds=300, peak_allocated_bytes=4 * 1024**3, peak_reserved_bytes=6 * 1024**3)


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _identity(t):
    return dict(shape=list(t.shape), dtype=str(t.dtype), tensor=array_digest(t.detach().cpu().numpy()))


def _readout(M, transform, D, device, raw=False):
    if raw:
        x, y, mass = decode_moments(M.to(device), D)
        x, y = x.float(), y.float()
    else:
        x, y, mass = representative(M, transform, D, device)
    return [x, y, torch.full_like(mass, 1 / len(mass))]


def _source(packet, B, guard):
    """Read pinned caches only, with no graph propagation or cache factory."""
    from src.transforms import FeatureTransform

    family = Path(B['family_root']); seed = B['condensation_seed']
    _require(type(seed) is int and seed in (1, 2) and B['budget'] == f'cora140_s{seed}', 'Unknown seed domain')
    _require(B['baseline'] == dict(method='low_rank',width=0,lr=.1,T=.3,rank=32,penalty=1e-4,
        initialization='teacher_balanced',alpha=.3,inner_loss_weighting='uniform')
        and B['recipe'] == dict(epochs=600,eval_every=1,hidden=256,dropout=0.,lr=.01,weight_decay=.0005,
            lr_schedule='constant',initialization='geom_uniform',input_scale=1.), 'Frozen preset changed')
    names = ['config.json', 'propagated_H.pt', 'teacher.pt', f'inputs_{seed}.pt',
        'nystrom_map_schema3.pt', 'nystrom_phi_schema3.npy', 'nystrom_phi_schema3.meta.json']
    required = {str(family / name) for name in names} | {B['hard_assignment'], B['original_options_resume']}
    required |= {B[key]['path'] for key in ('original_BB_certificate', 'original_BB_closure')}
    _require(required <= set(packet['readonly_files_sha256']), 'Required original cache is not pinned')
    _require(all(packet['readonly_files_sha256'][B[k]['path']] == B[k]['sha256']
        for k in ('original_BB_certificate','original_BB_closure')), 'BB reference digest differs from readonly pin')
    config = json.loads((family / 'config.json').read_text())
    _require(config['citation_features'] == 'row' and config['mixing'] == .05
        and config.get('teacher_backend') is None and config.get('teacher_kernel') in (None, 'relu')
        and config.get('kernel') == 'relu' and config['teacher_basis'] == 3000, 'Original ROW source changed')
    cert = json.loads(Path(B['original_BB_certificate']['path']).read_text())
    closure = json.loads(Path(B['original_BB_closure']['path']).read_text())
    old = cert['roots'][B['original_BB_root']]['source_context']
    _require(cert['passed'] is True and cert['roots'][B['original_BB_root']]['passed'] is True
        and closure['passed'] is True and old['root'] == str(family) and old['config'] == config
        and any(p.get('path') == B['original_BB_certificate']['path']
            and p.get('sha256') == B['original_BB_certificate']['sha256'] for p in closure['parents']),
        'Accepted BB common source reference changed')
    for name in names[:3] + names[4:]:
        _require(packet['readonly_files_sha256'][str(family/name)] == old['assets'][name], 'Common source asset changed')
    key = _fingerprint(dict(mode='teacher_balanced', alpha=.3, T=.3, seed=seed))
    _require(Path(B['hard_assignment']) == family/f'assignment_{key}.pt', 'Seed-specific hard cache key differs')
    shared = _load_state(family/'propagated_H.pt', 'shared_h')
    h = shared['h'].detach().to('cuda').clone(); guard()
    _require(shared['identity'] == _tensor_identity(h), 'Saved original H descriptor changed')
    inputs = torch.load(family/f'inputs_{seed}.pt', map_location='cuda', weights_only=False)
    z = inputs['z']; transform = FeatureTransform(**inputs['transform'])
    hard = torch.load(B['hard_assignment'], map_location='cuda', weights_only=False)
    logits = torch.load(family/'teacher.pt', map_location='cuda', weights_only=False)['logits']
    q = (logits / .3).softmax(1).double()  # exact original mixing-zero teacher expression; no labels
    _require(all(_identity(t) == old[name] for name,t in [('h',h),('z',z),('q',q)]),
        'New seed source is not byte-identical to accepted common H/z/Q')
    _require(transform.kind == 'rms' and transform.matrix is None and transform.eps == 1e-12
        and all(_identity(getattr(transform,k)) == old['transform'][k] for k in ('center','output_center','scale')),
        'Original RMS coordinates changed')
    _require(h.shape == z.shape == (2708,1433) and q.shape == (2708,7) and hard.shape == (2708,)
        and hard.dtype == torch.int64 and int(hard.min()) == 0 and int(hard.max()) == 139
        and len(torch.unique(hard)) == 140 and z.dtype == q.dtype == torch.float64
        and all(bool(torch.isfinite(t).all()) for t in (h,z,q)) and bool((q >= 0).all())
        and bool((q.sum(1) > 0).all()), 'Invalid original source dimensions/support')
    original = torch.load(B['original_options_resume'], map_location='cpu', weights_only=False)['config']
    expected = dict(factor_seed=0, assignment_rank=32, assignment_input='node', assignment_encoder='linear',
        mass_mode='free', inner_loss_weighting='uniform', penalty=1e-4, lr=.1, mixing=.05,
        solver_mode='exact', inner_method='newton_first', implicit_warm_start=True, save_assignment=False, chunk_size=4096)
    _require(all(original.get(k) == v for k,v in expected.items()) and not any(k.startswith(
        ('node_factor_', 'assignment_kl_', 'uniform_cell_q_prior_', 'graph_assignment_', 'graph_factor_',
         'kernel_commutation_', 'conditional_label_entropy_', 'soft_cell_mass_')) for k in original),
        'Original cold control options lost their unmodified recipe')
    digest = array_digest(z.cpu().numpy(), q.cpu().numpy(), hard.cpu().numpy())
    options = dict(original, factor_seed=seed, data_digest=digest)
    guard()
    return h, z, q, hard, transform, options, digest


def _load_capture(packet, B, digest, options):
    ref, accepted = B['captured_origin'], B['passed_capture_report']
    _require(all(packet['readonly_files_sha256'].get(r['path']) == r['sha256'] for r in (ref,accepted)),
        'New seed capture/acceptance is not pinned')
    report = json.loads(Path(accepted['path']).read_text())
    origin = torch.load(ref['path'], map_location='cpu', weights_only=False)
    _require(report['passed'] is True and report['kind'] == KIND and report['budget'] == B['budget']
        and report['source'] == packet['source'] and report['captured_origin'] == ref
        and origin['source'] == packet['source'] and origin['budget'] == B['budget']
        and origin['data_digest'] == digest and origin['options'] == options
        and origin['source_files_sha256'] == report['source_files_sha256']
        and all(packet['readonly_files_sha256'].get(p) == s for p,s in origin['source_files_sha256'].items()),
        'Fresh capture native/source/options lineage changed')
    native = origin['native']; _require(native['data_digest'] == digest and native['factor_seed'] == B['condensation_seed']
        and native['mixing'] == .05 and native['u'].dtype == native['v'].dtype == torch.float32
        and native['u'].shape == (2708,32) and native['v'].shape == (140,32)
        and bool(native['u'].eq(0).all()) and all(bool(torch.isfinite(native[k]).all()) for k in ('u','v')),
        'Captured native factors changed')
    return origin


def run(protocol_path, protocol_sha256, budget, phase, stop=lambda: False):
    """Exclusive terminal phase, with full input pins and owning failure evidence."""
    from src.research_loop import implementation_provenance
    from src.low_rank_assignment import initialize_factors, LowRankMoments, logit_block
    from src.citation_dual_head_ce import _resident_inputs

    _require(_sha(protocol_path) == protocol_sha256, 'Protocol changed')
    packet = json.loads(Path(protocol_path).read_text()); B = packet['budget_packets'][budget]
    _require(packet['kind'] == KIND and packet['resource_limits'] == LIMITS and B['budget'] == budget
        and phase in ('capture','linear','controlNy') and packet['test_enabled'] is False,
        'Unknown frozen phase/protocol')
    _require(packet['source'] == implementation_provenance(), 'Current source differs')
    folder = Path(B[dict(capture='capture_folder', linear='linear_folder', controlNy='centroid_folder')[phase]])
    _require(not folder.exists(), 'Never overwrite or retry a partial phase namespace')
    folder.mkdir(parents=True); started = time.monotonic(); arrays = {}; restored = None
    counters = dict(native_factory_calls=0, frozen_factory_calls=0, capture_physical_forwards=0,
        capture_raw_H_forwards=0, direct_H_probability_blocks=0, direct_H_numerator_GEMMs=0,
        returned_progress_rows=0, head_solves_from_capture=0, student_fits=0, test_evaluations=0)
    report = dict(schema=1, kind=KIND, passed=False, source=packet['source'], budget=budget, phase=phase,
        condensation_seed=B['condensation_seed'], protocol=dict(path=str(protocol_path),sha256=protocol_sha256),
        counts=counters, BB_reuse='common H/z/Q/RMS source only; no cond0 native/head/moments/readout',
        source_SGC_propagations=0, cache_fits=0, geometry_bound=1e-12)
    error = None
    def guard():
        if stop() or time.monotonic()-started > LIMITS['max_seconds'] or (
            torch.cuda.is_initialized() and (torch.cuda.max_memory_allocated() > LIMITS['peak_allocated_bytes']
                or torch.cuda.max_memory_reserved() > LIMITS['peak_reserved_bytes'])):
            raise RuntimeError('Terminal bounded seed phase; preserve partial evidence')
        return False
    try:
        torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False; torch.cuda.init(); torch.cuda.reset_peak_memory_stats(); guard()
        _pins(packet['readonly_files_sha256']); h,z,q,hard,transform,options,digest = _source(packet,B,guard)
        if phase == 'capture':
            u,v = initialize_factors(hard,140,32,B['condensation_seed']); counters['native_factory_calls'] += 1
            _require(u.dtype == v.dtype == torch.float32 and bool(u.eq(0).all())
                and bool(torch.isfinite(v).all()), 'Native U0/V0 is not finite FP32 zero/Gaussian')
            native = cpu_state(dict(u=u,v=v,data_digest=digest,factor_seed=B['condensation_seed'],mixing=.05))
            arrays['native'] = native; guard()
            with torch.no_grad():
                M = LowRankMoments.apply(u,v,hard,make_material(z,q),.05,4096)
                counters['capture_physical_forwards'] += 1; arrays['physical_M0'] = cpu_state(M); guard()
                raw = LowRankMoments.apply(u,v,hard,make_material(h.double(),q),.05,2048)
                counters['capture_raw_H_forwards'] += 1; arrays['raw_H_M0'] = cpu_state(raw); guard()
                actual = _readout(M,transform,1433,z.device); rawout = _readout(raw,transform,1433,z.device,True)
                arrays.update(physical_readout=cpu_state(actual),raw_H_readout=cpu_state(rawout))
                _require([_identity(t) for t in actual] == [_identity(t) for t in rawout],
                    'NEW seed raw-H P0 FP32 readout differs from physical P0')
                numerator = z.new_zeros(140,1433)
                for begin in range(0,len(z),4096):
                    guard(); end = min(begin+4096,len(z))
                    P = logit_block(u[begin:end],v,hard[begin:end],.05).double().softmax(1)
                    counters['direct_H_probability_blocks'] += 1
                    arrays['last_probability'] = cpu_state(P); report['last_probability_rows'] = [begin,end]; guard()
                    numerator += P.T @ h[begin:end].double()/len(z)
                    counters['direct_H_numerator_GEMMs'] += 1; arrays['direct_H_numerator'] = cpu_state(numerator)
                    report['completed_numerator_row_end'] = end
                mass = M[:,0]; physical = M[:,1:1434]/mass[:,None]
                physical = physical*transform.scale+transform.output_center+transform.center
                residual = physical-numerator/mass[:,None]; arrays['geometry_residual'] = cpu_state(residual)
            bound = float(residual.abs().max()); _require(math.isfinite(bound) and bound <= 1e-12
                and bool(torch.isfinite(M).all()) and bool((mass > 0).all()), 'NEW seed direct-H geometry failed')
            report.update(native_data_digest=digest,affine_inverse_vs_direct_H_centroid_max_absolute=bound,
                raw_H_P0_physical_readout_fullbyte_equal=True, new_seed_own_origin=True,
                new_origin_descriptors=dict(native_parameters=[_identity(t) for t in (u,v)],
                    physical_M0=_identity(M),raw_H_M0=_identity(raw),physical_readout=[_identity(t) for t in actual]),
                new_seed_source_descriptors=dict(h=_identity(h),z=_identity(z),q=_identity(q),
                    hard_assignment=_identity(hard),transform={k:_identity(getattr(transform,k))
                        for k in ('center','output_center','scale')}))
            source_pins = dict(packet['readonly_files_sha256'])
            origin = dict(schema=1,source=packet['source'],budget=budget,data_digest=digest,options=options,
                source_files_sha256=source_pins,native=native,physical_M0=cpu_state(M),raw_H_M0=cpu_state(raw),
                physical_readout=cpu_state(actual),raw_H_readout=cpu_state(rawout),transform=cpu_state(inputs_transform(transform)))
            path = folder/'captured_origin.pt'; _state_write(origin,path)
            native_path = folder/'native_reference_factors.pt'; _state_write(native,native_path)
            report.update(captured_origin=dict(path=str(path),sha256=_sha(path)),source_files_sha256=source_pins,
                native_P0_factors=dict(path=str(native_path),sha256=_sha(native_path)), actual_P_updates=0)
        else:
            origin = _load_capture(packet,B,digest,options)
            initial = [origin['native'][k].detach().to(z.device).clone() for k in ('u','v')]
            if phase == 'linear':
                from src import soft_ce_partition as core
            else:
                from src import nystrom_ce as core
            factory = core.initialize_factors; restored = (core,factory)
            def frozen_factory(assignment,cells,rank,seed=0):
                _require(assignment is hard and type(cells) is int and cells == 140 and type(rank) is int
                    and rank == 32 and type(seed) is int and seed == B['condensation_seed']
                    and counters['frozen_factory_calls'] == 0, 'Unexpected native factor request')
                result = [p.detach().clone().requires_grad_() for p in initial]
                _require([_identity(t) for t in result] == [_identity(origin['native'][k]) for k in ('u','v')],
                    'Owning native factor copy differs')
                counters['frozen_factory_calls'] += 1
                return result
            core.initialize_factors = frozen_factory
            if phase == 'linear':
                callopts = dict(options); callopts.pop('data_digest')
                core.optimize_ce_assignment(z,q,hard,steps=25,folder=folder,checkpoint_steps=(0,25),
                    save_resume=True,stop=guard,**callopts)
                state = torch.load(folder/'resume.pt',map_location='cpu',weights_only=False); history = state['history']
                p0,p25 = [state['snapshots'][k] for k in (0,25)]
                _require(state['config'] == options and all(row['J_exact'] and row['inner_converged'] for row in history)
                    and all(row['cg_converged'] for row in history[:25]), 'Original linear stationary interfaces failed')
                _require(_identity(p0['moments']) == _identity(origin['physical_M0']), 'Cold linear M0 differs from fresh capture')
                checkpoints = [folder/f'checkpoints/step_{k:06d}.pt' for k in (0,25)]
                expected_cfg = options
            else:
                fmap,phi,_,_,_ = _resident_inputs(packet,B,h,transform,guard)
                def progress(row):
                    counters['returned_progress_rows'] += 1; guard()
                core.optimize(h,q,hard,fmap,phi,folder,25,penalty=1e-4,lr=.1,rank=32,
                    seed=B['condensation_seed'],chunk=2048,stop=guard,progress=progress,
                    checkpoint_every=25,mixing=.05,inner_loss_weighting='uniform')
                state = torch.load(folder/'resume.pt',map_location='cpu',weights_only=False); history = state['history']
                checkpoints = [folder/f'step_{k:06d}.pt' for k in (0,25)]
                p0,p25 = [torch.load(p,map_location='cpu',weights_only=False) for p in checkpoints]
                expected_cfg = dict(steps_schema=2,penalty=1e-4,lr=.1,rank=32,seed=B['condensation_seed'],
                    cells=140,chunk=2048,inner_loss_weighting='uniform')
                _require(state['config'] == expected_cfg and counters['returned_progress_rows'] == 26
                    and all(math.isfinite(row['inner_grad']) and row['inner_grad'] <= 1e-7 for row in history)
                    and _identity(p0['moments']) == _identity(origin['raw_H_M0']), 'Original Ny control endpoints differ')
                report['Ny_completion_evidence'] = '26 returned progress rows; original optimize refuses failed inner/CG before update; no solver alias or diagnostic replay'
                report['Ny_Adam_recipe'] = 'original torch.optim.Adam([u,v],lr=.1): default eps1e-8 and foreach default'
            _require(type(state['step']) is int and state['step'] == 25 and [r['step'] for r in history] == list(range(26))
                and counters['frozen_factory_calls'] == 1 and all(float(slot['step']) == 25
                    for slot in state['optimizer']['state'].values()), 'Cold control fixed25/Adam completion failed')
            _require([_identity(t) for t in _readout(p0['moments'],transform,1433,z.device,phase=='controlNy')]
                == [_identity(t) for t in origin['physical_readout']], 'Actual control P0 serving differs from fresh origin')
            report.update(actual_P_updates=25,completed_optimizer_steps=25,completed_history_head_interfaces=26,
                structural_update_adjoints=25,structural_moment_backwards=25,
                complete_history_rows=26,original_options=expected_cfg,
                endpoints={str(k):dict(path=str(p),sha256=_sha(p)) for k,p in zip((0,25),checkpoints)},
                resume=dict(path=str(folder/'resume.pt'),sha256=_sha(folder/'resume.pt')),
                source_origin=B['captured_origin'],physical_P0_readout_exact_new_capture=True,
                representation=RAW_H if phase=='controlNy' else 'original_RMS_physical_moments_v1')
        _pins(packet['readonly_files_sha256']); _require(implementation_provenance() == packet['source'], 'Source changed')
        guard(); report['passed'] = True
    except Exception as exc:
        error = exc; report['error'] = dict(type=type(exc).__name__,message=str(exc))
    finally:
        if restored is not None:
            restored[0].initialize_factors = restored[1]
        if arrays:
            rawpath = folder/'new_seed_arrays.pt'; _state_write(arrays,rawpath)
            report['raw_evidence'] = dict(path=str(rawpath),sha256=_sha(rawpath),write_complete=True,phase_passed=report['passed'])
        report.update(seconds=time.monotonic()-started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0,
            peak_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0)
        _json_write(report,folder/'phase_report.json')
    if error is not None:
        raise RuntimeError('Terminal seed capture/control failure; preserve namespace, no automatic retry') from error
    return report


def inputs_transform(transform):
    return {key:getattr(transform,key) for key in ('center','matrix','output_center','scale','kind','eps')}
