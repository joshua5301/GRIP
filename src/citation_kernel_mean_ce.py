"""Frozen original-Phi kernel-mean critic with original physical student serving.

PREPARE measures one new kernel P0 origin without heads. Qualification owns one
native update; condensation extends its accepted prefix to 25. Evaluation uses
only four fixed physical endpoints and fresh paired validation students.
"""
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from src.citation_factor_geometry import _pins, _sha, _source
from src.citation_dual_head_ce import _native_parameters, _resident_inputs
from src.citation_graph_factor import (
    _equal, _geometry_certificate, _json_write, _manifest, _original_paths,
    _precision, _state_write,
)
from src.shared_features import _tensor_identity

KIND = 'fixed_original_Phi_kernel_mean_uniform_CE_native_NODE_v1'
CONTROL_P0 = 'common_original_P0'
CONTROL_LINEAR = 'original_linear_CE25'
CONTROL_CENTROID = 'closed_original_centroid_Nystrom_CE25'
RESOURCE_LIMITS = dict(max_seconds=300, resident_source_bytes=512 * 1024**2,
    peak_allocated_bytes=4 * 1024**3, peak_reserved_bytes=6 * 1024**3)
ORIGIN_LIMITS = dict(absolute_Linf_maximum=5e-7, relative_Linf_maximum=2e-5,
    both=True, reference='CPU_FP64_dense_and_chunked_cached_Phi_Q_using_native_P0_probabilities')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _assets(B):
    family = Path(B['family_root']); phi = family / 'nystrom_phi_schema3.npy'
    return dict(H=str(family / 'propagated_H.pt'), map=str(family / 'nystrom_map_schema3.pt'),
        Phi=str(phi), Phi_metadata=str(phi.with_suffix('.meta.json')))


def _source_geometry_admission(packet, B, digest):
    ref = B['source_geometry_acceptance']
    if isinstance(ref, dict) and ref.get('kind') == 'original_native_NODE_P0_direct_H_RMS_no_head_geometry_admission_v1':
        from src.citation_kernel_mean_geometry import load_admission
        return load_admission(packet, B, digest)
    if isinstance(ref, dict) and ref.get('kind') == 'seed_specific_original_NODE_P0_direct_H_RMS_no_head_capture_v1':
        from src.citation_kernel_mean_geometry import load_seed_admission
        return load_seed_admission(packet, B, digest)
    return _geometry_certificate(packet, B, digest)


def _observed_source(packet, B, h, z, q, assignment, transform, options, digest, stop):
    from src.kernel_mean_ce import _native_options, _runtime
    from src.moments import decode_moments
    from src.sweep_utils import representative

    _, phi, roles, assets, resident = _resident_inputs(packet, B, h, transform, stop)
    _require(roles == _assets(B) and resident <= RESOURCE_LIMITS['resident_source_bytes'],
        'Existing original H/map/Phi/sidecar residency differs')
    _, initial, digests = _native_parameters(packet, B, z, assignment, options, digest)
    original = torch.load(B['baseline_snapshot0'], map_location='cpu', weights_only=False)
    M0 = original['moments']; k, d, c = int(assignment.max()) + 1, z.shape[1], q.shape[1]
    _require(M0.dtype == torch.float64 and M0.shape == (k, 1 + d + c)
        and bool(torch.isfinite(M0).all()) and bool((M0[:, 0] > 0).all()), 'Original physical P0 moments malformed')
    centers, labels, _ = decode_moments(M0, d)
    x0, y0, mass = representative(M0, transform, d, z.device)
    serving = dict(X=x0.detach().cpu().clone(), Q=y0.detach().cpu().clone(),
        uniform_weights=torch.full_like(mass, 1 / len(mass)).detach().cpu().clone())
    assets.update(z=_tensor_identity(z), Q=_tensor_identity(q), assignment=_tensor_identity(assignment))
    geometry = _source_geometry_admission(packet, B, digest)
    refs = dict(nodes=len(z), cells=k, rank=options['assignment_rank'], dimension=d, classes=c,
        basis=phi.shape[1], chunk_size=options['chunk_size'], factor_seed=B['condensation_seed'], mixing=.05,
        original_options=_native_options(options), runtime=_runtime(z.device), device=str(z.device),
        data_digest=digest, asset_paths=roles, files_sha256=dict(packet['original_files_sha256']),
        current_source=packet['source'], optimizer_source_sha256=packet['optimizer_source_sha256'],
        original_RMS_geometry=geometry, original_baseline_source=packet['original_baseline_source'],
        original_P0=B['baseline_snapshot0'], original_native_factors=B['native_P0_factors'],
        original_student_P0={key: _tensor_identity(value) for key, value in serving.items()})
    _require(set(roles.values()) | {str(p) for p in _original_paths(B)} <= set(refs['files_sha256'])
        and all(packet['readonly_files_sha256'].get(path) == pin for path, pin in refs['files_sha256'].items()),
        'Original source/native/cache files are not immutable protocol pins')
    origin = dict(moments=_tensor_identity(M0), centers=_tensor_identity(centers), labels=_tensor_identity(labels))
    return phi, initial, digests, M0, serving, refs, assets, origin, resident


def _new_kernel_origin(phi, q, assignment, initial, options, folder, progress, stop):
    from src.kernel_mean_ce import _resident_phi
    from src.low_rank_assignment import LowRankMoments, logit_block
    from src.moments import make_material, decode_moments

    device = q.device; n, basis = len(q), phi.shape[1]
    u, v = [value.detach().to(device=device).clone() for value in initial]
    material = make_material(_resident_phi(phi, device), q)
    arrays = {}; progress.update(preparation_critic_moment_forward_attempts=1,
        preparation_critic_moment_forward_calls=0, preparation_physical_moment_forward_calls=0,
        independent_dense_reference_calls=0, independent_chunked_reference_calls=0)
    with torch.no_grad():
        M = LowRankMoments.apply(u, v, assignment, material, .05, options['chunk_size'])
        progress['preparation_critic_moment_forward_calls'] = 1
        # Probabilities follow native FP32 logits then FP64 softmax. Only the
        # aggregation reference is independent; no source Phi/kernel is rebuilt.
        probability = torch.empty((n, len(v)), dtype=torch.float64, device='cpu')
        for begin in range(0, n, options['chunk_size']):
            if stop():
                raise InterruptedError('New kernel P0 reference interrupted')
            end = min(begin + options['chunk_size'], n)
            probability[begin:end] = logit_block(u[begin:end], v, assignment[begin:end], .05).double().softmax(1).cpu()
        cpu_material = material.detach().cpu().clone()
        dense = probability.T @ cpu_material / n
        progress['independent_dense_reference_calls'] = 1
        chunked = torch.zeros_like(dense)
        for begin in range(0, n, options['chunk_size']):
            if stop():
                raise InterruptedError('New kernel P0 chunk reference interrupted')
            end = min(begin + options['chunk_size'], n)
            chunked += probability[begin:end].T @ cpu_material[begin:end] / n
        progress['independent_chunked_reference_calls'] = 1
    actual = M.detach().cpu().clone(); mu, labels, mass = decode_moments(actual, basis)
    arrays.update(actual_moments=actual, dense_reference=dense, chunked_reference=chunked)
    raw_path = folder / 'new_kernel_origin_reference_arrays.pt'; _state_write(arrays, raw_path)
    progress['new_kernel_origin_reference_arrays'] = dict(path=str(raw_path), sha256=_sha(raw_path), complete=True)
    comparisons = {}
    for name, reference in (('dense', dense), ('chunked', chunked)):
        ref_mu, ref_labels, ref_mass = decode_moments(reference, basis)
        _require(bool(torch.isfinite(reference).all()) and bool((ref_mass > 0).all()), 'New reference has invalid mass')
        comparisons[name] = {key: _precision(value, target.numpy(), ORIGIN_LIMITS)
            for key, value, target in (('moments', actual, reference), ('centers', mu, ref_mu),
                ('labels', labels, ref_labels), ('mass', mass, ref_mass))}
    arrays.update(actual_moments=actual, dense_reference=dense, chunked_reference=chunked)
    progress['new_kernel_P0_reference_metrics'] = comparisons
    progress['new_kernel_P0_reference_passed'] = all(item['passed'] for group in comparisons.values() for item in group.values())
    _require(progress['new_kernel_P0_reference_passed'] and bool(torch.isfinite(actual).all())
        and bool((mass > 0).all()) and bool(torch.isfinite(mu).all())
        and bool((labels >= 0).all()) and bool((labels.sum(1) > 0).all()), 'FIRST new kernel P0 representation reference failed')
    origin = dict(moments=_tensor_identity(actual), centers=_tensor_identity(mu), labels=_tensor_identity(labels))
    return actual, origin, arrays


def _prepare(packet, B, observed, options, assignment, q, progress, stop):
    from src.kernel_mean_ce import SCHEMA, MODE, POLICY, _cpu, _seal

    phi, initial, digests, M0, serving, refs, assets, origin, resident = observed
    folder = Path(B['input_folder']); _require(not folder.exists(), 'Fresh PREPARE namespace required; no overwrite/retry')
    folder.mkdir(parents=True); progress['receipt_path'] = str(folder / 'prepare_report.json')
    kernel_M0, kernel_origin, arrays = _new_kernel_origin(phi, q, assignment, initial, options, folder, progress, stop)
    context = dict(schema=SCHEMA, mode=MODE, policy=POLICY, source_refs=refs,
        native_parameter_digests=digests, asset_descriptors=assets, native_origin=origin, kernel_origin=kernel_origin)
    artifact = dict(schema=1, kind=KIND, source=packet['source'], budget=B['budget'], context=context,
        initial_parameters=_cpu(initial), original_moments=M0.clone(), original_serving=serving,
        kernel_moments=kernel_M0, new_kernel_reference_arrays=arrays)
    artifact['content_seal'] = _seal(artifact)
    path = folder / 'native_kernel_mean_packet.pt'; _state_write(artifact, path)
    _json_write(context, folder / 'kernel_mean_context.json')
    progress.update(native_kernel_mean_packet=dict(path=str(path), sha256=_sha(path)), kernel_mean_context=context,
        fixed_source_asset_bytes=resident, original_physical_P0_source='readonly original checkpoint; no remomentization',
        source_RMS_numeric_replays=0, new_native_factor_generations=0, actual_new_P_updates=0,
        actual_optimizer_work=None, physical_student_fits=0)



def prepared_payload_seal(artifact):
    """Descriptor seal of every field except the two permitted source bindings."""
    from src.kernel_mean_ce import _seal
    payload={k:v for k,v in artifact.items() if k not in ('source','content_seal')}
    context=dict(payload['context']);refs=dict(context['source_refs'])
    refs.pop('current_source');context['source_refs']=refs;payload['context']=context
    return _seal(payload)


def _prepared_rebind(packet,B,ref,preparation,artifact):
    """Read-only, separately pinned ROOT admission of a metadata-only rebind."""
    declaration=B['prepared_source_rebind']
    _require(isinstance(declaration,dict) and set(declaration)=={'path','sha256'}
        and packet['readonly_files_sha256'].get(declaration['path'])==declaration['sha256'],
        'ROOT prepared-source rebind receipt is missing its exact pin')
    accepted=json.loads(Path(declaration['path']).read_text())
    _require(type(accepted['schema']) is int and accepted['schema']==1
        and accepted['kind']=='root_metadata_only_kernel_mean_prepared_source_rebind_v1'
        and accepted['passed'] is True and accepted['metadata_only'] is True
        and accepted['budget']==B['budget'] and accepted['old_source']==preparation['source']
        and accepted['new_source']==packet['source'] and accepted['rebound_packet']==ref,
        'ROOT rebind does not connect the actual OLD producer to this current source')
    _require(accepted['allowed_changes']==['source','context.source_refs.current_source','content_seal']
        and accepted['numerical_work']==dict(critic_moment_forwards=0,physical_moment_forwards=0,
            kernel_calls=0,source_Phi_references=0,heads=0,adjoints=0,P_updates=0,students=0)
        and all(type(value) is int for value in accepted['numerical_work'].values()),
        'ROOT rebind must be metadata-only with exactly the two source fields and derived seal')
    for name in ('original_packet','rebound_packet','original_context','rebound_context',
                 'original_prepare_report','original_reference_arrays'):
        pin=accepted[name]
        _require(isinstance(pin,dict) and set(pin)=={'path','sha256'}
            and packet['readonly_files_sha256'].get(pin['path'])==pin['sha256'],
            'Old/new ROOT rebind lineage or FIRST raw reference is not frozen')
    original_report=Path(B['input_folder'])/'prepare_report.json'
    _require(accepted['original_prepare_report']==dict(path=str(original_report),sha256=_sha(original_report))
        and accepted['original_packet']==preparation['native_kernel_mean_packet']
        and accepted['original_reference_arrays']=={k:preparation['new_kernel_origin_reference_arrays'][k]
            for k in ('path','sha256')} and preparation['new_kernel_origin_reference_arrays']['complete'] is True,
        'ROOT rebind retargets the OLD successful prepare producer or its numerical cache')
    _require(accepted['original_context']['path']==str(Path(accepted['original_packet']['path']).parent/'kernel_mean_context.json'),
        'Original context path does not belong to the actual FIRST prepared packet')
    old_context=json.loads(Path(accepted['original_context']['path']).read_text())
    _require(old_context==preparation['kernel_mean_context']
        and old_context['source_refs']['current_source']==accepted['old_source'],
        'Original successful context has been relabeled')
    expected=dict(old_context);expected['source_refs']=dict(old_context['source_refs'],current_source=packet['source'])
    _require(artifact['context']==expected and json.loads(Path(accepted['rebound_context']['path']).read_text())==expected
        and accepted['rebound_context']['path']==str(Path(ref['path']).parent/'kernel_mean_context.json')
        and type(accepted['unchanged_payload_seal']) is str and len(accepted['unchanged_payload_seal'])==64
        and all(c in '0123456789abcdef' for c in accepted['unchanged_payload_seal'])
        and prepared_payload_seal(artifact)==accepted['unchanged_payload_seal'],
        'Rebound packet changed numerical tensors/context or lacks the OLD immutable payload seal')
    return accepted


def _load_prepared(packet, B, observed):
    from src.kernel_mean_ce import _seal, _cpu

    ref = B['native_kernel_mean_packet']
    _require(packet['readonly_files_sha256'].get(ref['path']) == ref['sha256'], 'Prepared new kernel origin packet not pinned')
    prepare_path = Path(B['input_folder']) / 'prepare_report.json'
    _require(str(prepare_path) in packet['readonly_files_sha256'], 'Passed FIRST representation preparation receipt unpinned')
    preparation = json.loads(prepare_path.read_text())
    _require(preparation['passed'] is True and preparation['new_kernel_P0_reference_passed'] is True,
        'Original FIRST kernel representation preparation was not accepted')
    artifact = torch.load(ref['path'], map_location='cpu', weights_only=False)
    if 'prepared_source_rebind' in B:
        _prepared_rebind(packet,B,ref,preparation,artifact)
    else:
        _require(preparation['source']==packet['source'] and preparation['native_kernel_mean_packet']==ref,
            'Different-source prepare requires separate ROOT metadata-only rebind admission')
    _, initial, digests, M0, serving, refs, assets, origin, _ = observed
    context = artifact['context']
    _require(artifact['schema'] == 1 and artifact['kind'] == KIND and artifact['source'] == packet['source']
        and artifact['budget'] == B['budget'] and artifact['content_seal'] == _seal({k:v for k,v in artifact.items() if k != 'content_seal'})
        and _seal(artifact['initial_parameters']) == _seal(_cpu(initial)) and _equal(artifact['original_moments'], M0)
        and all(_equal(artifact['original_serving'][key], value) for key,value in serving.items())
        and context['source_refs'] == refs and context['asset_descriptors'] == assets
        and context['native_origin'] == origin and context['native_parameter_digests'] == digests,
        'Prepared original/native/kernel context changed')
    _require(str(Path(ref['path']).parent / 'kernel_mean_context.json') in packet['readonly_files_sha256']
        and json.loads((Path(ref['path']).parent / 'kernel_mean_context.json').read_text()) == context,
        'Prepared canonical context lacks its immutable exact pin')
    return artifact, context


def _work_delta(state, before):
    from src.kernel_mean_ce import WORK_KEYS
    _require(set(state['work']) == set(WORK_KEYS), 'Kernel-mean work keys differ')
    delta = {key:value - before.get(key,0) for key,value in state['work'].items()}
    _require(all(type(value) is int and value >= 0 for value in delta.values()), 'Actual work moved backwards')
    return delta


def _P0(state, artifact, transform, dimension, device):
    from src.sweep_utils import representative

    p0 = state['snapshots'][0]; x,q,mass = representative(p0['physical_moments'], transform, dimension, device)
    serving = dict(X=x,Q=q,uniform_weights=torch.full_like(mass,1/len(mass)))
    return dict(P0_original_physical_M_and_native_parameters_bit_exact=_equal(p0['physical_moments'],artifact['original_moments'])
        and all(_equal(a,b) for a,b in zip(p0['parameters'],artifact['initial_parameters'],strict=True)),
        P0_original_physical_X_Q_uniform_weights_bit_exact=all(_equal(v,artifact['original_serving'][k]) for k,v in serving.items()),
        own_kernel_P0_exact_prepared_origin=_equal(p0['moments'],artifact['kernel_moments']),
        own_kernel_P0_head_stationary=p0['head_work']['inner_converged'] is True
            and math.isfinite(p0['head_work']['inner_grad_max']) and p0['head_work']['inner_grad_max'] <= 1e-7,
        own_positive_kernel_CE0_and_J0_equals1=math.isfinite(p0['CE0']) and p0['CE0']>0
            and p0['CE']==p0['CE0'] and p0['objective']==1., own_kernel_CE0=p0['CE0'],
        own_kernel_theta0_descriptor=_tensor_identity(p0['theta']), original_linear_theta_or_CE_used_as_critic=False,
        original_physical_origin_enforced_before_first_head_and_Adam=True,
        kernel_vs_physical_mass_Q_bit_equality_assumed=False)



def _native_gradient_reference(initial, assignment, material, complete):
    """One independent CPU-FP64 analytic fixed-G reference; no autograd/head."""
    u,v=[x.detach().cpu().clone() for x in initial]; A=assignment.detach().cpu()
    data,G=material.detach().cpu(),complete.detach().cpu();n,k,r=len(u),len(v),u.shape[1]
    _require(u.dtype==v.dtype==torch.float32 and bool(u.eq(0).all()),'Native reference must use sealed zeroU origin')
    prior=torch.full((n,k),math.log(.05/k),dtype=torch.float32)
    prior.scatter_(1,A[:,None],float(math.log(.95+.05/k)))
    probability=(prior+u@v.T/math.sqrt(r)).double().softmax(1)
    M=probability.T@data/n;direction=data@G.T/n
    # Preserve the native whole-logit FP32 cast and rank division, then use
    # independent FP64 products to measure the actual FP32 factor GEMM error.
    block=(probability*(direction-(probability*direction).sum(1,keepdim=True))).float()/math.sqrt(r)
    return M,block.double()@v.double(),block.double().T@u.double()


def _native_gradient_gate(state,phi,q,assignment,options,folder,progress,stop):
    from src.kernel_mean_ce import _resident_phi,complete_moment_cotangent
    from src.low_rank_assignment import LowRankMoments
    from src.moments import make_material

    p0=state['snapshots'][0];pb=state['current']['last_pullback'];initial=p0['parameters']
    _require(pb['step']==0 and p0['vector_step']==0 and p0['vector'] is not None,
        'Actual native first pullback/adjoint lineage missing')
    progress.update(native_fixed_cotangent_API_calls=0,native_fixed_cotangent_API_attempts=0,
        native_additional_moment_forward_calls=0,native_additional_moment_forward_attempts=0,
        native_additional_moment_backward_calls=0,native_additional_moment_backward_attempts=0,
        independent_native_fixedG_CPU_reference_calls=0,independent_native_fixedG_CPU_reference_attempts=0)
    device=q.device;arrays={'actual_M0':p0['moments'].clone(),'actual_gU0':pb['gU'].clone(),'actual_gV0':pb['gV'].clone()}
    primary=None;complete_raw=False
    try:
        material=make_material(_resident_phi(phi,device),q)
        progress['native_fixed_cotangent_API_attempts']=1
        G=complete_moment_cotangent(p0['moments'].to(device),phi.shape[1],p0['theta'].to(device),
            p0['vector'].to(device),options['penalty'],p0['CE0'])
        progress['native_fixed_cotangent_API_calls']=1;arrays['complete_G0']=G.detach().cpu().clone()
        for index in range(2):
            if stop():raise InterruptedError('Native kernel fixed-G repeat interrupted')
            u,v=[x.detach().to(device).clone().requires_grad_() for x in initial]
            progress['native_additional_moment_forward_attempts']+=1
            M=LowRankMoments.apply(u,v,assignment,material,.05,options['chunk_size'])
            progress['native_additional_moment_forward_calls']+=1
            progress['native_additional_moment_backward_attempts']+=1;M.backward(G)
            progress['native_additional_moment_backward_calls']+=1
            for name,value in (('M0',M),('gU0',u.grad),('gV0',v.grad)):
                _require(value is not None and bool(torch.isfinite(value).all()),'Native repeat returned nonfinite array')
                arrays[f'repeat{index+2}_{name}']=value.detach().cpu().clone()
        if stop():raise InterruptedError('Native independent fixed-G reference interrupted')
        progress['independent_native_fixedG_CPU_reference_attempts']=1
        references=_native_gradient_reference(initial,assignment,material,G)
        progress['independent_native_fixedG_CPU_reference_calls']=1
        arrays.update({f'reference_{name}_FP64':value for name,value in zip(('M0','gU0','gV0'),references,strict=True)})
        complete_raw=True
    except BaseException as error:
        primary=error;raise
    finally:
        path=folder/'native_fixed_G_gradient_gate_arrays.pt'
        try:
            _state_write(arrays,path);progress['native_fixed_G_gradient_gate_arrays']=dict(path=str(path),sha256=_sha(path),
                complete=complete_raw,partial_unqualified=not complete_raw)
        except BaseException as observation_error:
            if primary is None:raise
            primary.add_note('Native raw-array preservation failed: '+repr(observation_error))
    repeated=all(_equal(arrays[f'repeat{index}_{name}'],arrays[f'actual_{name}'])
        for index in (2,3) for name in ('M0','gU0','gV0'))
    metrics={name:_precision(arrays[f'actual_{name}'],arrays[f'reference_{name}_FP64'].numpy(),ORIGIN_LIMITS)
        for name in ('M0','gU0','gV0')}
    zero=bool(arrays['actual_gV0'].eq(0).all()) and bool(arrays['reference_gV0_FP64'].eq(0).all())
    progress.update(native_three_full_byte_fixedG_repeats_passed=repeated,native_fixedG_precision_metrics=metrics,
        native_initial_gV_exact_zero=zero,native_repeat_run1='saved_actual_core_M0_gU0_gV0',
        native_kernel_gradient_gate_passed=repeated and zero and all(item['passed'] for item in metrics.values()),
        native_additional_heads=0,native_additional_adjoints=0,native_additional_P_updates=0)
    _require(progress['native_kernel_gradient_gate_passed'],'NEW native kernel-mean gradient precision/repeat gate failed')

def _evaluate(packet,B,graph,train,validation,h,z,q,transform,mode,seeds,progress,stop):
    from src.evaluation import fit_gcn_diagnostic
    from src.kernel_mean_ce import MODE,validate_core_resume
    from src.student_routes import replay_routes
    from src.sweep_utils import representative

    admission=packet['fixed25_acceptance']
    _require(packet['readonly_files_sha256'].get(admission['path'])==admission['sha256'],
        'Root completed-condensation student admission is unpinned')
    completed=json.loads(Path(admission['path']).read_text())
    _require(completed[admission['acceptance_field']]==admission['acceptance_value']
        and completed['source']==packet['source'],'Fresh students require accepted fixed25 candidate trajectories')
    folder=Path(B['evaluation_folders'][mode]);_require(not folder.exists(),'Fresh paired student namespace required')
    folder.mkdir(parents=True);progress['receipt_path']=str(folder/'evaluation_report.json')
    if mode==CONTROL_CENTROID:
        ref=B['centroid_Nystrom_checkpoint25'];checkpoint=Path(ref['path']);step=25
        _require(packet['readonly_files_sha256'].get(ref['path'])==ref['sha256'],'Closed centroid control is not readonly pinned')
    elif mode in (CONTROL_P0,CONTROL_LINEAR):
        step=0 if mode==CONTROL_P0 else 25;checkpoint=Path(B['baseline_folder'])/'checkpoints'/f'step_{step:06d}.pt'
    else:
        _require(mode==MODE,'Unknown candidate endpoint');trajectory=Path(B['candidate_folder'])
        accepted=json.loads((trajectory/'condensation_report.json').read_text())
        _require(accepted['passed'] is True and accepted['source']==packet['source'] and accepted['observed_final_step']==25,
            'Evaluation lacks accepted fixed25 candidate')
        _pins(json.loads((trajectory/'checkpoint_manifest.json').read_text()))
        state=torch.load(trajectory/'resume.pt',map_location='cpu',weights_only=False)
        validate_core_resume(state,state['config'],state['context'],trajectory)
        _require(state['step']==25,'Candidate endpoint is not25');step=25;checkpoint=trajectory/'checkpoints/step_000025.pt'
    _require(str(checkpoint) in packet['readonly_files_sha256'],'Selected physical endpoint is not root-pinned')
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    moments=saved['physical_moments'] if mode==MODE else saved['moments']
    if mode==CONTROL_CENTROID and 'centroid_control_representation' in B:
        _require(B['centroid_control_representation']=='original_raw_H_centroid_moments_v1',
            'Unknown explicit readonly centroid control coordinate representation')
        from src.moments import decode_moments
        x,y,mass=decode_moments(moments.to(z.device),h.shape[1]); x,y=x.float(),y.float()
        progress['control_readout_representation']='original_raw_H_centroid_moments_v1'
    else:
        x,y,mass=representative(moments,transform,z.shape[1],z.device)
    settings={key:value for key,value in B['recipe'].items() if key!='input_scale'}
    records=[];progress.update(records=records,physical_student_fit_attempts=0,physical_student_fits=0,
        same_selected_weight_validation_routes=0,whole_epochs=0)
    for seed in seeds:
        if stop():raise InterruptedError('Fixed paired cohort interrupted')
        evaluation=folder/f'step_{step}_{B["recipe_id"]}';progress['physical_student_fit_attempts']+=1
        result=fit_gcn_diagnostic(x,y,torch.full_like(mass,1/len(mass)),graph,q,dict(train=train,val=validation[1]),
            seed,folder=evaluation,stop=stop,**settings)
        progress['physical_student_fits']+=1;progress['whole_epochs']+=B['recipe']['epochs']
        selected=evaluation/f'seed_{seed}_selected.pt';route_path=evaluation/f'seed_{seed}_validation_routes_v1.json'
        routes=replay_routes(selected,graph,h.float(),dict(val=validation[1]),settings,route_path,seed=seed,stop=stop)
        _require(not any('test_' in key for key in result|routes) and routes['gcn_val_acc']==result['val_acc']
            and routes['epoch']==result['epoch'],'Validation selected-weight route link differs')
        progress['same_selected_weight_validation_routes']+=2
        records.append(dict(step=step,**result,SGC_MLP_sameweights_val=routes['mlp_val_acc'],
            selected_weights=dict(path=str(selected),sha256=_sha(selected)),routes=dict(path=str(route_path),sha256=_sha(route_path))))
    progress.update(actual_new_P_updates=0,actual_optimizer_work=None,
        control_checkpoint=dict(path=str(checkpoint),sha256=_sha(checkpoint)))


def _run(protocol_path,budget,phase,mode,student_seeds,stop,progress):
    from src.kernel_mean_ce import MODE,optimize,validate_core_resume
    from src.research_loop import implementation_provenance

    declaration=json.loads(Path(protocol_path).read_text());B0=declaration['budget_packets'][budget]
    _require(declaration['kind']==KIND and B0['budget']==budget and B0['dataset'] in ('cora','citeseer')
        and phase in ('prepare','qualify','condense','evaluate'),'Wrong frozen kernel-mean protocol')
    _require(declaration['resource_limits']==RESOURCE_LIMITS and declaration['kernel_origin_gate']==ORIGIN_LIMITS,
        'Prospective resource or new-representation bounds changed')
    required=set(_assets(B0).values())|{str(p) for p in _original_paths(B0)}
    _require(all(Path(p).is_file() and p in declaration['readonly_files_sha256'] for p in required),
        'Original existing caches/mandatory sidecar required before any fallback')
    for name in ('complete_math','production_QA','metadata_lineage'):
        ref=declaration['prerequisite_acceptances'][name]
        _require(declaration['readonly_files_sha256'].get(ref['path'])==ref['sha256'],'New prerequisite receipt unpinned')
        accepted=json.loads(Path(ref['path']).read_text())
        _require(accepted[ref['acceptance_field']] == ref['acceptance_value'],'New proof/production/lineage prerequisite not accepted')
    packet,B,graph,train,validation,h,z,q,assignment,transform,options,digest=_source(protocol_path,budget,stop)
    seeds=list(student_seeds)
    _require(seeds==([] if phase!='evaluate' else B['student_seeds']) and all(type(s) is int for s in seeds)
        and (phase!='evaluate' or len(seeds)==len(set(seeds))==3),'Frozen phase cohort differs')
    _require((phase=='prepare' and mode is None) or (phase in ('qualify','condense') and mode==MODE)
        or (phase=='evaluate' and mode in (MODE,CONTROL_P0,CONTROL_LINEAR,CONTROL_CENTROID)),'Unknown phase/endpoint')
    progress.update(schema=1,budget=budget,phase=phase,mode=mode,source=packet['source'],protocol_path=str(protocol_path),
        native_data_digest=digest,objective='own_kernel_mean_outer_CE_over_own_positive_CE0',
        test_evaluations=0,RMS_refits=0,H_rebuilds=0,kernel_factory_calls=0,Phi_rebuilds=0,
        native_factor_generation_calls=0,graph_factor_products=0,physical_student_fits=0,actual_new_P_updates=0,
        source_RMS_numeric_replays=0,basic_source_SGC_X_work='original loader; uninstrumented, not claimed zero')
    mutable=set()
    if phase=='evaluate':
        _evaluate(packet,B,graph,train,validation,h,z,q,transform,mode,seeds,progress,stop)
    else:
        observed=_observed_source(packet,B,h,z,q,assignment,transform,options,digest,stop)
        if phase=='prepare':_prepare(packet,B,observed,options,assignment,q,progress,stop)
        else:
            artifact,context=_load_prepared(packet,B,observed);phi=observed[0]
            folder=Path(B['candidate_folder']);context_path=folder/'context.json';before={};state=None
            _require(not (folder/'failure.json').exists(),'Failed candidate cannot be retried/rescued')
            if phase=='qualify':
                _require(not folder.exists(),'Fresh qualification namespace required');folder.mkdir(parents=True)
                _json_write(context,context_path)
            else:
                accepted=json.loads((folder/'qualification.json').read_text())
                _require(accepted['passed'] is True and accepted['source']==packet['source']
                    and accepted['kernel_mean_context']==context and all(accepted[key] is True for key in
                        ('P0_original_physical_M_and_native_parameters_bit_exact','P0_original_physical_X_Q_uniform_weights_bit_exact',
                         'own_kernel_P0_exact_prepared_origin','own_kernel_P0_head_stationary','own_positive_kernel_CE0_and_J0_equals1',
                         'native_kernel_gradient_gate_passed')),
                    'Candidate does not have accepted native0/1 qualification')
                immutable=[context_path,folder/'qualification.json',folder/'checkpoints/step_000000.pt',folder/'checkpoints/step_000001.pt']
                _require(all(str(p) in packet['readonly_files_sha256'] for p in immutable)
                    and json.loads(context_path.read_text())==context,'Accepted immutable0/1 prefix context unpinned')
                _pins(json.loads((folder/'checkpoint_manifest.json').read_text()))
                state=torch.load(folder/'resume.pt',map_location='cpu',weights_only=False)
                _require(type(state['step']) is int and state['step']==1,'Only accepted1→25 authorized')
                before=dict(state['work']);mutable={str(folder/name) for name in ('resume.pt','optimization.csv','checkpoint_manifest.json')}
                _require(mutable<=set(packet['readonly_files_sha256'])
                    and packet.get('accepted_mutable_prefix_pin_policy','').startswith('Sole pre-execution read'),
                    'Evolving prefix must be pinned only at this sole pre-execution read')
            progress['receipt_path']=str(folder/('qualification.json' if phase=='qualify' else 'condensation_report.json'))
            steps=1 if phase=='qualify' else 25
            progress.update(optimizer_invocation_attempts=0,kernel_mean_context=context,observed_initial_step=0 if state is None else state['step'])
            initial_device=[p.detach().to(device=z.device).clone() for p in artifact['initial_parameters']]
            _require(all(_equal(current,stored) for current,stored in zip(initial_device,artifact['initial_parameters'],strict=True)),
                'Native device-only copy changed initial factor dtype/shape/full bytes')
            progress['native_initial_device_copy_dtype_shape_bytes_preserved']=True
            progress['optimizer_invocation_attempts']=1
            state=optimize(z,q,assignment,initial_device,phi,folder,steps,options,context,
                checkpoint_steps=(0,1,steps),resume_state=state,stop=stop)
            progress['optimizer_invocation_completed']=1;validate_core_resume(state,state['config'],context,folder)
            work=_work_delta(state,before);updates=1 if phase=='qualify' else 24
            _require(state['step']==steps and all(work[k]==updates for k in ('P_updates','Adam_steps','moment_backward_calls','adjoint_solves'))
                and all(work[k]==updates+1 for k in ('head_interfaces','moment_forward_calls','physical_moment_forward_calls')),
                'Actual critic/physical/head/adjoint/P work differs from fixed phase')
            progress.update(actual_optimizer_work=work,cumulative_optimizer_work=state['work'],actual_new_P_updates=updates,
                observed_final_step=steps,head_return_history=state['history'],fixed_source_asset_bytes=observed[-1])
            if phase=='qualify':
                gates=_P0(state,artifact,transform,z.shape[1],z.device);progress.update(gates)
                _require(all(gates[key] is True for key in ('P0_original_physical_M_and_native_parameters_bit_exact',
                    'P0_original_physical_X_Q_uniform_weights_bit_exact','own_kernel_P0_exact_prepared_origin',
                    'own_kernel_P0_head_stationary','own_positive_kernel_CE0_and_J0_equals1')),'Native kernel P0 qualification failed')
                _native_gradient_gate(state,phi,q,assignment,options,folder,progress,stop)
            paths=[context_path,folder/'resume.pt',folder/'optimization.csv',*sorted((folder/'checkpoints').glob('step_*.pt'))]
            progress['checkpoint_manifest_after']=_manifest(folder,paths)
    _pins({path:pin for path,pin in packet['readonly_files_sha256'].items() if path not in mutable})
    _require(implementation_provenance()==packet['source'],'Current full implementation/Git changed')
    return progress


def run(protocol_path,protocol_sha256,budget,phase,mode=None,student_seeds=(),stop=lambda:False):
    """One fresh bounded phase; InterruptedError becomes terminal, never retry."""
    progress,started,primary={},time.monotonic(),None
    def bounded_stop():return stop() or time.monotonic()-started>RESOURCE_LIMITS['max_seconds']
    try:
        _require(callable(stop) and _sha(protocol_path)==protocol_sha256,'Frozen invocation protocol/callback differs')
        progress['protocol_sha256']=protocol_sha256
        result=_run(protocol_path,budget,phase,mode,student_seeds,bounded_stop,progress)
        _require(not bounded_stop(),'Kernel-mean phase exceeded300s')
        allocated=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0
        reserved=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0
        _require(allocated<=RESOURCE_LIMITS['peak_allocated_bytes'] and reserved<=RESOURCE_LIMITS['peak_reserved_bytes'],
            'Observed peak GPU memory exceeded prospective ceilings')
        result.update(passed=True,peak_allocated_bytes=allocated,peak_reserved_bytes=reserved)
    except BaseException as error:
        primary=error;progress.update(passed=False,error_type=type(error).__name__,error=str(error),
            interrupted_operation_interiors='unknown; completed call counters preserved separately')
        if isinstance(error,InterruptedError):raise RuntimeError('Terminal bounded kernel-mean interruption; no retry') from error
        raise
    finally:
        progress['elapsed_seconds']=time.monotonic()-started
        if torch.cuda.is_initialized():
            progress.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
        path=progress.pop('receipt_path',None)
        if path is not None:
            try:
                progress['observed_owned_files_sha256']={str(p):_sha(p) for p in Path(path).parent.rglob('*') if p.is_file()}
                _json_write(progress,path)
            except BaseException as observation_error:
                if primary is None:raise
                primary.add_note('Owned result preservation also failed: '+repr(observation_error))
    return result
