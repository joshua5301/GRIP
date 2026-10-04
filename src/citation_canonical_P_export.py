"""New canonical serving-only exports; historical critics and failures stay intact."""
import hashlib
import json
import math
import os
import time
from pathlib import Path

import torch

from src.io import array_digest
from src.low_rank_assignment import LowRankMoments
from src.moments import make_material, decode_moments

KIND = 'canonical_saved_NODE_P_raw_H_Q_six_export_preflight_v1'
CONTRACT_SHA = '3f93bb99376f945a5b782a7003fad0d6055b4303fcb9e113dea98872e5ef788b'
LIMITS = dict(max_seconds=300, peak_allocated_bytes=4*1024**3, peak_reserved_bytes=6*1024**3,
              exclusive_fresh_output_namespace=True, terminal_failures_no_auto_retry=True)
LINEAR = dict(factor_seed=0, assignment_rank=32, assignment_input='node', assignment_encoder='linear',
    mass_mode='free', inner_loss_weighting='uniform', penalty=1e-4, lr=.1, mixing=.05,
    solver_mode='exact', inner_method='newton_first', implicit_warm_start=True, chunk_size=4096)
NY = dict(steps_schema=2, penalty=1e-4, lr=.1, rank=32, seed=0, cells=140, chunk=2048,
          inner_loss_weighting='uniform')


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def _own(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k:_own(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_own(v) for v in value]
    return value


def _identity(t):
    return dict(shape=list(t.shape), dtype=str(t.dtype), tensor=array_digest(t.detach().cpu().numpy()))


def _content(t, canonical_double=False):
    dtype = 'float64' if canonical_double else str(t.dtype)
    digest = hashlib.sha256(json.dumps(dict(shape=tuple(t.shape),dtype=dtype),sort_keys=True).encode())
    for block in t.detach().split(2048):
        array = (block.double() if canonical_double else block).cpu().contiguous().numpy()
        digest.update(memoryview(array).cast('B'))
    return digest.hexdigest()


def run(protocol_path, protocol_sha256, stop=lambda:False):
    from src.research_loop import implementation_provenance

    _require(_sha(protocol_path)==protocol_sha256 and callable(stop), 'Frozen invocation differs')
    packet=json.loads(Path(protocol_path).read_text()); ref=packet['scientific_contract']
    _require(ref['sha256']==CONTRACT_SHA and _sha(ref['path'])==CONTRACT_SHA, 'Prospective contract differs')
    contract=json.loads(Path(ref['path']).read_text()); cases=contract['six_exports']
    _require(packet['schema']==1 and type(packet['schema']) is int and packet['kind']==KIND
        and packet['source']==implementation_provenance() and packet['test_enabled'] is False
        and packet['resource_limits']==LIMITS and packet['six_exports']==cases, 'New export domain differs')
    def collect(value):
        if isinstance(value,dict):
            if 'path' in value and 'sha256' in value:
                _require(packet['readonly_files_sha256'].get(value['path'])==value['sha256'], 'Required source pin absent')
            for child in value.values():collect(child)
        elif isinstance(value,list):
            for child in value:collect(child)
    collect(contract)
    folder=Path(packet['output_folder']); _require(not folder.exists(), 'Exclusive namespace required')
    folder.mkdir(parents=True); started=time.monotonic(); arrays={}; cache={}; failure=None
    counts=dict(target_decode_attempts=0,target_decodes=0,moment_forward_attempts=0,moment_forwards=0,
        readouts=0,seed1_geometry_attempts=0,seed1_geometry_comparisons=0,heads=0,adjoints=0,
        P_updates=0,Adam_steps=0,students=0,test_evaluations=0,native_factories=0,source_SGC_propagations=0)
    report=dict(schema=1,kind=KIND,passed=False,source=packet['source'],protocol=dict(path=str(protocol_path),
        sha256=protocol_sha256),scientific_contract=ref,counts=counts,exports=[],old_failure_requalified=False)
    def guard():
        _require(not stop() and time.monotonic()-started<=300 and torch.cuda.max_memory_allocated()<=LIMITS[
            'peak_allocated_bytes'] and torch.cuda.max_memory_reserved()<=LIMITS['peak_reserved_bytes'], 'Terminal resource/stop boundary')
    def pins():
        for path,digest in packet['readonly_files_sha256'].items():
            _require(_sha(path)==digest,'Readonly source bytes differ');guard()
    def load(ref):
        if ref['path'] not in cache:cache[ref['path']]=torch.load(ref['path'],map_location='cpu',weights_only=False)
        return cache[ref['path']]
    try:
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest');torch.cuda.init();torch.cuda.reset_peak_memory_stats();pins()
        common=contract['common_source']; bb=json.loads(Path(common['BB_common_source_metadata']['path']).read_text())
        old=bb['roots']['b23d5978e22a']['source_context'];_require(bb['passed'] is True,'Common BB source not accepted')
        shared=load(common['H']);_require(shared['schema']==1 and shared['kind']=='shared_h','H cache schema differs')
        h=shared['h'].detach().to('cuda').clone();_require(_identity(h)==old['h']==common['expected_array_descriptors']['h'],'H source differs')
        logits=load(common['teacher_logits'])['logits'].to('cuda');counts['target_decode_attempts']=1
        q=(logits/.3).softmax(1).double();counts['target_decodes']=1;arrays['new_source_Q']=_own(q)
        _require(_identity(q)==old['q']==contract['raw_Q_admission']['exact_descriptor'],'Original Q differs');guard()
        material=make_material(h.double(),q); native0=load(cases[0]['factors'])
        for item in cases:
            guard();saved=load(item['factors']);inputs=load(item['original_inputs']);z=inputs['z'].to('cuda')
            hard=load(item['hard_assignment']).to('cuda');seed=item['seed'];step=item['step']
            _require(type(seed) is int and type(step) is int and _identity(z)==old['z']
                and hard.dtype==torch.int64 and tuple(hard.shape)==(2708,) and int(hard.min())==0
                and int(hard.max())==139 and len(torch.unique(hard))==140,'Original seed source differs')
            digest=array_digest(z.cpu().numpy(),q.cpu().numpy(),hard.cpu().numpy());config=None
            if step==0:
                native=saved['native'] if seed else saved
                _require(type(native['factor_seed']) is int and native['factor_seed']==seed
                    and native['data_digest']==digest and native['mixing']==.05,'Native capture origin differs')
                params=[native['u'],native['v']]
                if seed:
                    original=json.loads(Path(item['original_report']['path']).read_text())
                    _require(original['passed'] is (seed==2) and original['condensation_seed']==seed
                        and original['budget']==f'cora140_s{seed}' and original['phase']=='capture'
                        and original['counts']['native_factory_calls']==1
                        and (original['raw_evidence']['path']==item['factors']['path'] if seed==1
                            else original['captured_origin']=={k:item['factors'][k] for k in ('path','sha256')}
                                and saved['source']==original['source'] and saved['data_digest']==digest),
                        'Historical capture status differs')
            else:
                _require(type(saved['step']) is int and saved['step']==25,'Resume is not exact endpoint25')
                config=saved['config'];params=[saved['u'],saved['v']] if item['selector']==['u','v'] else saved['parameters']
                if item['selector']==['u','v']:
                    checkpoint=load(item['checkpoint25']);fp=saved['input_fingerprint']
                    _require(config==NY and type(config['seed']) is int and checkpoint['step']==25
                        and fp==checkpoint['input_fingerprint'] and fp['h_digest']==_content(h,True)
                        and fp['q_digest']==_content(q,True) and fp['assignment_digest']==_content(hard),'Ny source/config differs')
                else:
                    _require(all(config.get(k)==v for k,v in LINEAR.items()) and type(config['factor_seed']) is int
                        and config['data_digest']==digest,'Original linear options/source differ')
                    if 'context' in item:
                        context=json.loads(Path(item['context']['path']).read_text())
                        accepted=json.loads(Path(item['acceptance']['path']).read_text())
                        _require(saved['context']==context and context['source_refs']['data_digest']==digest
                            and accepted['passed'] is True and accepted['total_P_updates']==25
                            and accepted['source']==context['source_refs']['current_source']
                            and accepted['immutable_resume']=={k:item['factors'][k] for k in ('path','sha256')}
                            and context['native_parameter_digests']==[array_digest(native0[k].numpy()) for k in ('u','v')]
                            and [_identity(t) for t in saved['initial_parameters']]==[_identity(native0[k]) for k in ('u','v')]
                            and type(saved['current']['step']) is int and saved['current']['step']==25
                            and [_identity(t) for t in saved['current']['parameters']]==[_identity(t) for t in params],
                            'Own mean endpoint lost captured native/source context')
            _require(len(params)==2 and all(torch.is_tensor(t) and t.dtype==torch.float32 and bool(torch.isfinite(t).all())
                for t in params) and tuple(params[0].shape)==(2708,32) and tuple(params[1].shape)==(140,32)
                and (step!=0 or bool(params[0].eq(0).all())),'Invalid owning factor endpoint')
            u,v=[t.detach().to('cuda').clone() for t in params]
            _require([_identity(t) for t in (u,v)]==[_identity(t) for t in params],'Owning factor copy changed bytes')
            returned=dict(parameters=_own([u,v]));arrays[item['id']]=returned;counts['moment_forward_attempts']+=1
            with torch.no_grad():M=LowRankMoments.apply(u,v,hard,material,.05,4096)
            counts['moment_forwards']+=1;returned['moments']=_own(M)
            x,y,mass=decode_moments(M,1433);returned.update(X=_own(x.float()),Q=_own(y.float()),
                uniform_weights=_own(torch.full_like(mass,1/140)));counts['readouts']+=1
            report['exports'].append(dict(id=item['id'],factor_provider=item['factors'],seed=seed,step=step,
                historical_provider_status=item.get('old_provider_status',item['status']),observed_config=config,
                native_data_digest=digest,descriptors={k:_identity(t) for k,t in returned.items() if torch.is_tensor(t)}))
            _require(bool(torch.isfinite(M).all()) and bool((mass>0).all()) and bool((y>=0).all())
                and bool((y.sum(1)>0).all()) and all(bool(torch.isfinite(returned[k]).all())
                    for k in ('X','Q','uniform_weights')),'New canonical readout invalid');guard()
            if seed==1:
                counts['seed1_geometry_attempts']=1;physical=saved['physical_M0'].to('cuda');transform=inputs['transform']
                _require(transform['kind']=='rms' and transform['matrix'] is None and transform['eps']==1e-12,'Original RMS metadata differs')
                center=decode_moments(physical,1433)[0];inverse=center*transform['scale'].to('cuda')+transform['output_center'].to('cuda')+transform['center'].to('cuda')
                residual=inverse-x;returned.update(first_seed1_inverse=_own(inverse),first_seed1_residual=_own(residual))
                counts['seed1_geometry_comparisons']=1;value=float(residual.abs().max())
                report['first_seed1_geometry']=dict(Linf=value if math.isfinite(value) else dict(nonfinite=repr(value)),
                    bound=1e-12,passed=math.isfinite(value) and value<=1e-12,old_FP32_parity_requalified=False)
                _require(bool(torch.isfinite(residual).all()) and value<=1e-12,'First seed1 geometry fails');guard()
        pins();_require(implementation_provenance()==packet['source'] and counts['moment_forwards']==6
            and counts['target_decodes']==counts['seed1_geometry_comparisons']==1,'Source/work differs');guard();report['passed']=True
    except BaseException as error:
        failure=error;report['error']=dict(type=type(error).__name__,message=str(error))
    finally:
        raw=folder/'returned_canonical_export_arrays.pt';report['raw_evidence']=dict(path=str(raw),exists=False,complete=False)
        try:
            with raw.open('xb') as stream:torch.save(arrays,stream);stream.flush();os.fsync(stream.fileno())
            report['raw_evidence'].update(exists=True,bytes=raw.stat().st_size,sha256=_sha(raw),complete=True)
        except BaseException as error:
            report['passed']=False;report['raw_evidence'].update(exists=raw.exists(),bytes=raw.stat().st_size if raw.exists() else 0,error=str(error));failure=failure or error
        report.update(seconds=time.monotonic()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
        if report['seconds']>300 or report['peak_allocated_bytes']>LIMITS['peak_allocated_bytes'] or report['peak_reserved_bytes']>LIMITS['peak_reserved_bytes']:
            report['passed']=False;failure=failure or RuntimeError('Final output resource boundary exceeded')
        with (folder/'canonical_export_report.json').open('x') as stream:json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
    if failure is not None:raise RuntimeError('Terminal canonical preflight; original failures unchanged') from failure
    return report
