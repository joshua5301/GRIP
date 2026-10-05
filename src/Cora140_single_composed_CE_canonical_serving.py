"""Unexecuted FB: clone four admitted canonical-H readouts; one new headless P25 export."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import time

KIND = 'Cora140_single_composed_CE_five_arm_common_canonical_H_validation_stageFB_v1'
SCIENCE_SHA = 'b0f3b74a775e927642691fccb107fa0942c533d3b6888b257925437df46b0863'
ARMS = ('P0','linear25','centroidNy25','meanPhi25','composed25')
SEEDS = [620500,620501,620502]
SECONDARY = 'Original cached source-SGC H and exact GCN-selected weights, descriptive; no packed gcn_norm S²X equality or raw-X MLP claim.'


def _require(value, message):
    if not value: raise ValueError(message)


def _sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()


def _refs(value, found=None):
    found={} if found is None else found
    if isinstance(value,dict):
        if set(value)=={'path','sha256'}:
            p,h=value['path'],value['sha256']
            _require(type(p) is str and Path(p).is_absolute() and str(Path(p).resolve())==p
                and type(h) is str and len(h)==64 and all(c in '0123456789abcdef' for c in h),'Invalid normalized ABS ref')
            _require(p not in found or found[p]==h,'Conflicting ref');found[p]=h
        for v in value.values():_refs(v,found)
    elif isinstance(value,list):
        for v in value:_refs(v,found)
    return found


def _observed(value):
    if isinstance(value,dict):return {k:_observed(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [_observed(v) for v in value]
    if type(value) is float and not math.isfinite(value):return dict(nonfinite_observation=repr(value))
    return value


def run(protocol_path, protocol_sha256, budget, phase, arm=None, stop=lambda:False):
    started=time.monotonic(); pp=Path(protocol_path)
    _require(pp.is_absolute() and str(pp.resolve())==protocol_path and _sha(pp)==protocol_sha256,'Protocol differs')
    packet=json.loads(pp.read_text()); science_ref=packet['scientific_contract']
    _require(science_ref['sha256']==SCIENCE_SHA and _sha(science_ref['path'])==SCIENCE_SHA,'Science differs')
    contract=json.loads(Path(science_ref['path']).read_text());B=contract['budgets'][budget]
    _require(budget=='cora140' and packet['kind']==contract['kind']==KIND and phase in ('prepare','evaluate')
        and (arm is None if phase=='prepare' else arm in ARMS),'Budget/phase/arm differs')
    folder=Path(packet['prepare_folders'][budget] if phase=='prepare' else packet['evaluation_folders'][budget][arm])
    _require(folder.is_absolute() and str(folder.resolve())==str(folder),'Output must be normalized ABS');folder.mkdir(exist_ok=False,parents=True)
    raw=folder/('shared_serving_cache.pt' if phase=='prepare' else 'returned_evaluation_evidence.pt')
    counts=dict(own_PT_load_attempts=0,own_PT_loads=0,material_attempts=0,materials=0,moment_forward_attempts=0,moment_forwards=0,
        readout_attempts=0,readouts=0,cache_write_attempts=0,cache_writes=0,dataset_gets=0,graph_packs=0,source_SGC=0,Q_decodes=0,map_fits=0,RMS_refits=0,
        factor_factories=0,critic_heads=0,adjoints=0,P_updates=0,assignment_Adam_steps=0,test_evaluations=0,
        fit_attempts=0,student_fits=0,student_epochs=0,route_call_attempts=0,route_calls=0,validation_routes=0)
    report=dict(schema=1,kind=KIND,budget=budget,phase=phase,arm=arm,source=packet['source'],scientific_contract=science_ref,
        protocol=dict(path=protocol_path,sha256=protocol_sha256),passed=False,completed=False,failure=None,records=[],counts=counts,
        secondary_provenance=SECONDARY,test_enabled=False,raw_evidence=dict(path=str(raw),exists=False,bytes=None,sha256=None,
            hash_unknown=True,hash_error=None,write_completed=False))
    cache={};saved={};torch=None;old_handler=signal.getsignal(signal.SIGALRM);old_timer=signal.getitimer(signal.ITIMER_REAL)
    def peaks():
        v=dict(seconds=time.monotonic()-started,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            peak_allocated_bytes=0,peak_reserved_bytes=0)
        if torch is not None and torch.cuda.is_initialized():v.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(0),peak_reserved_bytes=torch.cuda.max_memory_reserved(0))
        return v
    def guard():
        _require(not stop() and all(v<=contract['resources'][k] for k,v in peaks().items()),'Terminal frozen stop/resource boundary');return False
    def gate(name,ok,**detail):
        report.setdefault('gates',[]).append(dict(name=name,passed=bool(ok),**detail));_require(ok,name)
    def alarm(signum,frame):raise TimeoutError('Frozen300s serving boundary')
    def read(ref):_require(_sha(ref['path'])==ref['sha256'],'Pinned JSON differs');return json.loads(Path(ref['path']).read_text())
    def pins():
        expected=packet['readonly_files_sha256'];_require(all(Path(p).is_absolute() and str(Path(p).resolve())==p for p in expected),'Pin path differs')
        _require(all(expected.get(p)==h for p,h in _refs(dict(packet=packet,science=contract)).items()),'Incomplete direct pin closure')
        values={p:_sha(p) for p in expected};_require(values==expected,'Readonly bytes changed');guard();return values
    try:
        signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,max(1e-6,300-(time.monotonic()-started)))
        report['readonly_entry']=pins()
        import torch
        from src.research_loop import implementation_provenance
        from src.kernel_mean_ce import _cpu,_plain,_seal,_runtime
        from src.shared_features import _tensor_identity
        from src.citation_canonical_student_validation import _identity,_cache_identity
        from src.low_rank_assignment import LowRankMoments
        from src.moments import make_material,decode_moments
        report['source_entry']=implementation_provenance();bridge=read(packet['prerequisite_admission'])
        gate('current_source_production_entry_bridge',report['source_entry']==packet['source'] and bridge['passed'] is True
            and bridge['source']==packet['source'] and bridge['entrypoint']==packet['entrypoint'] and bridge['scientific_contract']==science_ref
            and Path(packet['entrypoint']['path']).resolve()==Path(__file__).resolve())
        torch.set_num_threads(4)
        if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
        torch.cuda.init();torch.cuda.reset_peak_memory_stats(0);device=torch.device('cuda:0')
        report['runtime']=_runtime(device);gate('original_runtime_and_four_threads',report['runtime']==B['original_runtime']
            and all(os.environ.get(k)=='4' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')))
        def load(ref,role):
            guard();counts['own_PT_load_attempts']+=1;v=torch.load(ref['path'],map_location='cpu',weights_only=False)
            counts['own_PT_loads']+=1;saved['loaded_'+role]=v;saved['loaded_'+role]=_cpu(v);return v
        def metadata(v):
            if torch.is_tensor(v):return _identity(v)
            if isinstance(v,dict):return {k:metadata(x) for k,x in v.items()}
            if isinstance(v,(list,tuple)):return [metadata(x) for x in v]
            return v
        settings=dict(contract['student_recipe']['settings']);gate('literal_original_recipe_and_fixed_fresh3',read(contract['student_recipe']['file'])==settings
            and settings.pop('input_scale')==1.0 and contract['student_seeds']==SEEDS)
        if phase=='prepare':
            legacy=B['historical121'];prior,root,peer=(read(legacy[x]) for x in ('report','ROOT','independent'))
            gate('old121_whole_cache_admitted',prior['passed'] is True and root['passed'] is True and peer['passed'] is True
                and root['shared_prepare_report']==peer['evidence']['prepare_report']==legacy['report']
                and prior['shared_cache']==root['shared_cache']==legacy['cache'] and prior['source']==root['source']==B['historical121_producer']
                and prior['scientific_contract']==B['historical121_science'] and prior['cache_identities']==root['cache_identities']==B['historical121_identities'])
            historical=load(legacy['cache'],'historical121_cache')
            header={k:historical[k] for k in ('schema','kind','source','scientific_contract','identities')}
            gate('old121_whole_payload_header_identities',set(historical)=={'schema','kind','source','scientific_contract','graph','masks','propagated','q','arms','identities'}
                and header==B['historical121_header_metadata'] and _seal(header)==B['historical121_header_seal']
                and historical['identities']==_cache_identity(historical)==B['historical121_identities']
                and set(historical['masks'])=={'train','val'} and set(historical['arms'])==set(B['old_arm_key_mapping'].values()))
            FA=B['original_FA'];fa=read(FA['report']);candidate=packet['composed_candidate'][budget];ep=candidate['endpoint25'];native=read(ep['report'])
            gate('packet_actual_native25_refs_exact',candidate==B['actual_composed_candidate'])
            for role in ('ROOT','independent'):
                fa_adm=read(FA[role]);native_adm=read(ep[role])
                gate(role+'_FA_and_complete25_admitted',fa_adm['passed'] is True and native_adm['passed'] is True
                    and fa_adm['budgets'][budget]['report' if role=='ROOT' else 'actual_report']==FA['report']
                    and fa_adm['budgets'][budget]['arrays']==FA['arrays'] and native_adm['budgets'][budget]['report']==ep['report']
                    and native_adm['budgets'][budget]['state']==ep['state'])
            gate('FA153_original_owning_producer',fa['passed'] is True and fa['budget']==budget
                and fa['source']==fa['source_entry']==fa['source_exit']==B['FA_producer']
                and fa['scientific_contract']==contract['required_refs']['FA_science'] and fa['raw_evidence']['sha256']==FA['arrays']['sha256'])
            gate('actual_FB_native_complete25_metadata',native['passed'] is True and native['completed'] is True and native['mode']=='continue'
                and native['kind']=='Cora140_single_composed_CE_native_qualification_and_continuation_stageFB_v1' and native['budget']==budget
                and native['scientific_contract']==contract['required_refs']['FB_native_science']
                and native['source']==native['source_entry']==native['source_exit']==candidate['native_producer'] and native['actual_FA_source']['report']==FA['report']
                and native['state']==ep['state'] and native['core_artifacts']['checkpoints']['step_000025.pt']==ep['checkpoint']
                and native['context']['asset_descriptors']==B['FA_assets'] and native['context']['native_parameter_digests']==B['native_parameter_digests']
                and native['context']['source_refs']['data_digest']==B['original_data_digest']
                and native['context']['source_refs']['original_options']==B['original_options']
                and native['core_work']['physical_forward']==27 and native['core_work']['head']==native['core_work']['endpoint_map']==26
                and native['core_work']['P_updates']==native['core_work']['G_map']==25 and native['core_work']['reattachment']==1)
            payload=load(FA['arrays'],'own_FA153_source');captured=payload['captured']
            gate('FA153_payload_header_H_Q_two_hash_domains',payload['schema']==1 and payload['kind']==fa['kind'] and payload['budget']==budget
                and payload['source']==B['FA_producer'] and payload['scientific_contract']==fa['scientific_contract']
                and _tensor_identity(captured['loaded_H']['h'])==B['FA_assets']['H'] and _tensor_identity(captured['raw_Q'])==B['FA_assets']['Q']
                and _identity(captured['loaded_H']['h'])==historical['identities']['H'] and _identity(captured['raw_Q'])==historical['identities']['Q']
                and _tensor_identity(captured['loaded_hard'])==B['FA_assets']['assignment'])
            exports=load(B['old120_arrays'],'old120_canonical_arrays');p0=exports[B['old120_P0_id']]
            export_report=read(contract['required_refs']['old120_report'])
            observed=next(row for row in export_report['exports'] if row['id']==B['old120_P0_id'])
            gate('old120_shared_native_P0_owning_parameters_and_readout',export_report['passed'] is True and export_report['source']==B['old120_export_producer']
                and observed==B['old120_P0_export'] and [_tensor_identity(v) for v in p0['parameters']]==[_tensor_identity(captured['loaded_native'][x]) for x in ('u','v')]
                and {k:_identity(p0[k]) for k in ('X','Q','uniform_weights')}==historical['identities']['arms'][B['old120_P0_id']]
                and _identity(p0['moments'])==observed['descriptors']['moments'] and _identity(exports['new_source_Q'])==historical['identities']['Q'])
            linear0=load(B['original_linear0'],'original_linear0_physical')
            gate('historical_original_linear0_physical_M0_exact',type(linear0['step']) is int and linear0['step']==0
                and _tensor_identity(linear0['moments'])==_tensor_identity(captured['physical_M0']))
            state=load(ep['state'],'own_FB_complete25_state');E=state['current_evaluation'];E0=state['snapshots'][0]
            gate('new_own_native25_state_E_context_pure_seals',state['step']==25 and state['context']==native['context']
                and state['state_digest']==_seal({k:v for k,v in state.items() if k!='state_digest'})
                and E['step']==25 and E['kind']=='composed_joint_head_evaluation_v1' and E['record_digest']==_seal({k:v for k,v in E.items() if k!='record_digest'})
                and E['context_digest']==_seal(native['context']) and E['source_admission_digest']==_seal(native['context']['source_refs']['source_admission'])
                and all(_plain(E[k])==v for k,v in native['evaluation_summaries']['25']['scalars'].items())
                and all(_tensor_identity(E[k])==v for k,v in native['evaluation_summaries']['25']['identities'].items())
                and _seal(E['parameters'])==native['evaluation_summaries']['25']['parameters_digest']
                and _seal(state['parameters'])==_seal(E['parameters']) and state['snapshots'][25]['record_digest']==E['record_digest'])
            gate('newE0_shared_native_factors_and_original_physical_frame',[_tensor_identity(v) for v in E0['parameters']]==[_tensor_identity(v) for v in p0['parameters']]
                and _tensor_identity(E0['physical_moments'])==_tensor_identity(linear0['moments'])==_tensor_identity(captured['physical_M0']))
            report['initialization_observation']=dict(shared_native_factor_bits_equal=True,original_linear_physical_M0_equal=True,
                historical_linear_V0_bits_observed=False,historical_linear_V0_reconstruction=False,scope=contract['initialization_observation'])
            h=captured['loaded_H']['h'].detach().to(device).clone();q=captured['raw_Q'].detach().to(device).clone();hard=captured['loaded_hard'].to(device)
            u,v=[t.detach().to(device).clone() for t in E['parameters']]
            saved['canonical_export_inputs']=_cpu(dict(u=u,v=v,hard=hard,H=h,Q=q))
            gate('own_factor_and_canonical_source_domain',u.dtype==v.dtype==torch.float32 and list(u.shape)==[2708,32] and list(v.shape)==[140,32]
                and [_identity(t) for t in (u,v)]==[_identity(t) for t in E['parameters']] and h.dtype==torch.float32 and q.dtype==torch.float64
                and hard.dtype==torch.int64 and list(hard.shape)==[2708] and int(hard.min())==0 and int(hard.max())==139 and len(torch.unique(hard))==140
                and all(bool(torch.isfinite(t).all()) for t in (u,v,h,q)))
            counts['material_attempts']+=1;material=make_material(h.double(),q);counts['materials']+=1;saved['canonical_material']=_cpu(material)
            counts['moment_forward_attempts']+=1
            with torch.no_grad():M=LowRankMoments.apply(u,v,hard,material,.05,4096)
            counts['moment_forwards']+=1;saved['returned_canonical_M25']=_cpu(M)
            counts['readout_attempts']+=1;x,y,mass=decode_moments(M,1433);counts['readouts']+=1
            saved['returned_native_canonical_quotients']=_cpu(dict(x=x,y=y,mass=mass))
            returned=_cpu(dict(X=x.float(),Q=y.float(),uniform_weights=torch.full_like(mass,1/140)))
            saved['returned_canonical_readout']=returned
            gate('canonical_original_H_Q_domain_and_FP64_uniform',M.dtype==torch.float64 and list(M.shape)==[140,1441]
                and bool(torch.isfinite(M).all()) and bool((mass>0).all()) and bool((y>=0).all()) and bool((y.sum(1)>0).all())
                and list(returned['X'].shape)==[140,1433] and list(returned['Q'].shape)==[140,7]
                and all(bool(torch.isfinite(t).all()) for t in returned.values())
                and returned['X'].dtype==returned['Q'].dtype==torch.float32 and returned['uniform_weights'].dtype==torch.float64
                and _identity(returned['uniform_weights'])==historical['identities']['arms'][B['old120_P0_id']]['uniform_weights'])
            rows={name:_cpu(historical['arms'][old]) for name,old in B['old_arm_key_mapping'].items()};rows['composed25']=returned;saved['all_five_readouts']=_cpu(rows)
            gate('old_four_readouts_unchanged',all({k:_identity(t) for k,t in rows[a].items()}==historical['identities']['arms'][old] for a,old in B['old_arm_key_mapping'].items()))
            cache=dict(schema=1,kind=KIND,budget=budget,source=packet['source'],scientific_contract=science_ref,graph=_cpu(historical['graph']),
                masks=_cpu(historical['masks']),propagated=_cpu(historical['propagated']),q=_cpu(historical['q']),arms=rows,
                upstream_providers=dict(historical121=legacy,header=header,FA153=FA,FA153_producer=fa['source'],old120_P0=B['old120_P0_export'],
                    old120_producer=export_report['source'],original_linear0=B['original_linear0'],native25=ep,native_producer=native['source'],arm_key_mapping=B['old_arm_key_mapping']))
            cache['owning_canonical_export']=_cpu(dict(parameters=[u,v],hard=hard,moments=M,H_Q_source='cache.propagated/cache.q',mixing=.05,chunk_size=4096))
            report['owning_canonical_export_identities']=metadata(cache['owning_canonical_export'])
            cache['identities']=_cache_identity(cache);report['cache_identities']=cache['identities'];report['upstream_providers']=cache['upstream_providers']
            report['canonical_export']=dict(moment=_identity(M),readout={k:_identity(t) for k,t in returned.items()},parameters_digest=_seal(E['parameters']),step=25)
            gate('shared_graph_H_Q_masks_unchanged',all(cache['identities'][k]==historical['identities'][k] for k in ('graph','masks','H','Q')))
            gate('all_five_own_inputs_unchanged',all(_seal(metadata(v))==_seal(metadata(saved['loaded_'+role])) for v,role in
                ((historical,'historical121_cache'),(payload,'own_FA153_source'),(exports,'old120_canonical_arrays'),(linear0,'original_linear0_physical'),(state,'own_FB_complete25_state'))))
            gate('prepare_exact_five_load_one_export_counts',counts['own_PT_load_attempts']==counts['own_PT_loads']==5
                and counts['material_attempts']==counts['materials']==counts['moment_forward_attempts']==counts['moment_forwards']==counts['readout_attempts']==counts['readouts']==1)
        else:
            adm=packet['shared'][budget];prior,root,peer=(read(adm[x]) for x in ('report','ROOT','independent'))
            gate('new_cache_ROOT_independent_admitted',prior['passed'] is True and root['passed'] is True and peer['passed'] is True
                and root['actual_report']==peer['actual_report']==adm['report'] and prior['shared_cache']==root['shared_cache']==peer['shared_cache']==adm['cache'])
            cache=load(adm['cache'],'new_shared_cache')
            gate('new_whole_shared_cache_identity',cache['schema']==1 and cache['kind']==KIND and cache['budget']==budget
                and cache['source']==prior['source']==packet['source'] and cache['scientific_contract']==science_ref
                and cache['identities']==_cache_identity(cache)==prior['cache_identities'] and set(cache['arms'])==set(ARMS) and set(cache['masks'])=={'train','val'}
                and cache['upstream_providers']==prior['upstream_providers']
                and metadata(cache['owning_canonical_export'])==prior['owning_canonical_export_identities']
                and _identity(cache['owning_canonical_export']['moments'])==prior['canonical_export']['moment']
                and all(cache['identities'][k]==B['historical121_identities'][k] for k in ('graph','masks','H','Q'))
                and all(cache['identities']['arms'][a]==B['historical121_identities']['arms'][old] for a,old in B['old_arm_key_mapping'].items()))
            gate('valonly_labels_and_five_canonical_domains',int(cache['masks']['val'].sum())==500 and int(cache['masks']['train'].sum())==140
                and not bool((cache['masks']['train']&cache['masks']['val']).any())
                and bool((cache['graph']['y'][~(cache['masks']['train']|cache['masks']['val'])]==-1).all())
                and all(list(row['X'].shape)==[140,1433] and list(row['Q'].shape)==[140,7] and row['X'].dtype==row['Q'].dtype==torch.float32
                    and _identity(row['uniform_weights'])==B['historical121_identities']['arms'][B['old120_P0_id']]['uniform_weights'] for row in cache['arms'].values()))
        if phase=='evaluate':
            packet.update(shared_cache=adm['cache'],shared_prepare_report=adm['report'],shared_prepare_acceptance=adm['ROOT'])
        if phase == 'evaluate':
            from src.evaluation import fit_gcn_diagnostic
            from src.student_routes import replay_routes
            selected_arm = next(i for i in contract['arms'] if i['id'] == arm)
            report.update(shared_cache=packet['shared_cache'], shared_prepare_report=packet['shared_prepare_report'],
                          shared_prepare_acceptance=packet['shared_prepare_acceptance'], arm_provider=selected_arm['factor_provider'])
            graph = {k:t.to('cuda') for k,t in cache['graph'].items()}; masks = {k:t.to('cuda') for k,t in cache['masks'].items()}
            q, h = cache['q'].to('cuda'), cache['propagated'].to('cuda'); inputs = cache['arms'][arm]
            x,y,mass = (inputs[k].to('cuda') for k in ('X','Q','uniform_weights'))
            fit_folder = folder/f"step_{selected_arm['P_step']}_{contract['student_recipe']['id']}"
            for seed in contract['student_seeds']:
                guard(); counts['fit_attempts'] += 1
                fit = fit_gcn_diagnostic(x,y,mass,graph,q,masks,seed,folder=fit_folder,training_adjacency=None,stop=guard,**settings)
                counts['student_fits'] += 1; record = dict(seed=seed,fit=fit); report['records'].append(record)
                paths = {name:fit_folder/f'seed_{seed}{suffix}' for name,suffix in
                         (('json','.json'),('csv','_epochs.csv'),('selected','_selected.pt'))}
                record['fit_files'] = {name:dict(path=str(p),sha256=_sha(p)) for name,p in paths.items()}
                rows = list(csv.DictReader(paths['csv'].open())); counts['student_epochs'] += len(rows)
                saved = json.loads(paths['json'].read_text()); best = max(float(row['val_acc']) for row in rows)
                first = next(row for row in rows if float(row['val_acc']) == best)
                _require([int(r['epoch']) for r in rows] == list(range(1,601)) and fit['epoch'] == int(first['epoch'])
                         and fit['val_acc'] == best and saved['result'] == fit and saved['recipe']['test_enabled'] is False
                         and saved['recipe']['settings'] == settings and not any('test' in key for key in fit)
                         and all(not isinstance(v,float) or math.isfinite(v) for v in fit.values()), 'Incomplete/changed selected fit')
                guard(); route_path = fit_folder/f'seed_{seed}_routes.json'; counts['route_call_attempts'] += 1
                route = replay_routes(paths['selected'],graph,h,{'val':masks['val']},settings,route_path,seed=seed,stop=guard)
                counts['route_calls'] += 1; counts['validation_routes'] += 2; record['routes'] = route
                record['route_file'] = dict(path=str(route_path),sha256=_sha(route_path))
                cached_route=json.loads(route_path.read_text())
                _require(route['gcn_val_acc'] == fit['val_acc'] and route['epoch'] == fit['epoch'] and
                         cached_route['recipe']['source_fingerprint'] == saved['fingerprint'] and
                         cached_route['recipe']['test_enabled'] is False and not any('test' in key for key in route)
                         and all(not isinstance(v,float) or math.isfinite(v) for v in route.values()), 'Selected-weight route mismatch')
                guard()
            _require(counts['student_fits']==3 and counts['student_epochs']==1800 and counts['validation_routes']==6,'Incomplete arm')
        gate('no_new_head_P_or_source_work',all(counts[k]==0 for k in ('critic_heads','adjoints','P_updates','assignment_Adam_steps','dataset_gets','graph_packs',
            'source_SGC','Q_decodes','map_fits','RMS_refits','factor_factories','test_evaluations')))
        if phase=='evaluate':
            gate('evaluate_exact_cache_and_fit_scope',counts['own_PT_load_attempts']==counts['own_PT_loads']==1
                and counts['materials']==counts['moment_forwards']==counts['readouts']==0)
            report['internal_route_checkpoint_reads']=counts['route_calls'];report['cache_identities']=cache['identities']
        torch.cuda.synchronize(device);guard();report['completed']=True
    except BaseException as error:
        report['failure']=dict(type=type(error).__name__,message=str(error));report['inflight_interiors']='Unknown until return; owning outputs and attempts preserved separately'
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        try:
            if torch is not None:
                evidence=cache if phase=='prepare' and report['completed'] else dict(partial_unqualified=not report['completed'],returned=saved,cache=cache,records=report['records'])
                raw_info=report['raw_evidence']
                try:
                    if phase=='prepare':counts['cache_write_attempts']+=1
                    with raw.open('xb') as f:torch.save(evidence,f);f.flush();os.fsync(f.fileno())
                    raw_info['write_completed']=True
                    if phase=='prepare':counts['cache_writes']+=1
                except BaseException as error:report['raw_write_error']=repr(error)
                try:
                    raw_info['exists']=raw.exists()
                    if raw.exists():
                        raw_info['bytes']=raw.stat().st_size
                        try:raw_info['sha256']=_sha(raw);raw_info['hash_unknown']=False
                        except BaseException as error:raw_info['hash_error']=repr(error)
                except BaseException as error:report['raw_metadata_error']=repr(error)
                if phase=='prepare' and report['completed']:report['shared_cache']=dict(path=str(raw),sha256=raw_info['sha256'])
            try:
                report['readonly_exit']=pins();report['source_exit']=implementation_provenance();_require(report['source_exit']==packet['source'],'Source changed at exit')
            except BaseException as error:report['exit_verification_error']=repr(error)
            report['resources']=peaks();report['passed']=bool(report['completed'] and report['failure'] is None
                and not any(k in report for k in ('raw_write_error','raw_metadata_error','exit_verification_error'))
                and report['raw_evidence']['write_completed'] and not report['raw_evidence']['hash_unknown']
                and all(v<=contract['resources'][k] for k,v in report['resources'].items()))
            rp=folder/('prepare_report.json' if phase=='prepare' else 'evaluation_report.json')
            with rp.open('x') as f:json.dump(_observed(report),f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
            final=peaks()
            if report['passed'] and any(v>contract['resources'][k] for k,v in final.items()):
                report.update(passed=False,resources=final,failure=dict(type='FinalSerializationResourceBoundary',message='Preserve failure; no retry'))
                with rp.open('w') as f:json.dump(_observed(report),f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
        finally:signal.signal(signal.SIGALRM,old_handler);signal.setitimer(signal.ITIMER_REAL,*old_timer)
    _require(report['passed'],'Terminal FB canonical serving failure; preserve namespace, no seed/recipe/secondary rescue')
    return str(rp)
