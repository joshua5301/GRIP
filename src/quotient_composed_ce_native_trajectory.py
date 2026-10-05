"""FS accepted own FR E1 -> fixed25; unchanged original native/CSR/head math.

Historical185 is the scientific owner. Invocation186 is separately authenticated.
Only current operator returns persist; original FIRST artifacts remain immutable.
"""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import sys
import time

KIND='original_ROW_CSR_quotient_SGC_original_Nystrom_uniform_CE_native_acceptedE1_to_fixed25_v1'
SCIENCE_SHA='cbd03aa96f529891b7d279afbc6016af155aa7bd28743347f63b3cc8b539a65a'
POLICY=dict(schema=1,kind=KIND,first_step=1,endpoint=25,cached_R1=True,own_CE0_once=True,original_Adam=True)


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def refs(value,found=None):
    found={} if found is None else found
    if isinstance(value,dict):
        if set(value)=={'path','sha256'}:
            p,h=value['path'],value['sha256'];require(Path(p).is_absolute() and str(Path(p).resolve())==p and len(h)==64,'Normalized ABS ref')
            require(p not in found or found[p]==h,'Conflicting ref');found[p]=h
        for child in value.values():refs(child,found)
    elif isinstance(value,(list,tuple)):
        for child in value:refs(child,found)
    return found


def observed(value):
    if isinstance(value,dict):return {str(k):observed(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [observed(v) for v in value]
    if type(value) is float and not math.isfinite(value):return dict(nonfinite=repr(value))
    return value


def validate_trajectory_state(state,expected_owner,expected_context,expected_source,expected_config):
    """Pure endpoint/Adam/metadata seals, no historical current-source _files."""
    import torch
    from src import quotient_composed_ce_native_source as supplier
    from src.dual_head_ce import _validate_optimizer
    from src.joint_mean_ce import _seal
    supplier.validate_context(expected_context)
    require(state['schema']==1 and state['kind']==KIND and state['step']==state['scientific_endpoint']==25
        and state['source_owner']==expected_owner and state['context']==expected_context
        and state['invocation_source']==expected_source and state['owning_source']==expected_owner['current_source_full_manifest']
        and state['config']==expected_config and state['config_digest']==_seal(expected_config),'Typed fixed25 state/context/config')
    E=state['current_endpoint'];parameters=state['native_parameters'];d=expected_config['dimensions']
    require(E['step']==25 and E['CE0']==expected_context['new_positive_CE0'] and math.isfinite(E['raw_CE']) and E['raw_CE']>0
        and supplier.metadata([parameters['U'],parameters['V']])==E['native_parameter_identities']
        and E['head']['inner_converged'] is True and math.isfinite(E['head']['inner_grad_max'])
        and 0<=E['head']['inner_grad_max']<=expected_config['original29_options']['inner_tol']
        and E['adjoint']['cg_converged'] is True and all(bool(torch.isfinite(t).all()) for t in supplier.tensors((E,parameters))), 'Fixed25 owned endpoint/parameter domain')
    require(_validate_optimizer(parameters['optimizer'],expected_config['original29_options'],dict(nodes=d['N'],cells=d['K'],rank=d['rank']))==25
        and all(slot['step'].dtype==torch.float32 for slot in parameters['optimizer']['state'].values()),'Original native Adam step25')
    require([row['step'] for row in state['history']]==list(range(26)) and all(row['CE0']==E['CE0'] for row in state['history']),'Compact full26 endpoint history')
    before_write={key:(0 if key in ('owning_terminal_arrays_write','final_JSON_report_write') else value) for key,value in expected_config['expected_counts'].items()}
    require(state['counts']==before_write,'Exact35 state counts sealed before terminal writes')
    body={key:value for key,value in state.items() if key!='state_metadata_seal'}
    require(state['state_metadata_seal']==_seal(observed(supplier.metadata(body))),'Pure fixed25 state metadata seal')
    return state


def run(protocol_path,protocol_sha256,budget='cora35',mode='accepted1to25',stop=lambda:False):
    started=time.monotonic();path=Path(protocol_path)
    require(path.is_absolute() and str(path.resolve())==str(path) and sha(path)==protocol_sha256,'Protocol bytes')
    packet=json.loads(path.read_text());sr=packet['scientific_contract']
    require(sr['sha256']==SCIENCE_SHA and sha(sr['path'])==SCIENCE_SHA,'Exact selected science')
    science=json.loads(Path(sr['path']).read_text());C=science['case']
    require(packet['kind']==science['kind']==KIND and budget=='cora35' and mode=='accepted1to25','Accepted E1 fixed25 only')
    folder=Path(packet['output_folder']);require(folder.is_absolute() and str(folder.resolve())==str(folder),'ABS output');folder.mkdir(parents=True,exist_ok=False)
    raw_path=folder/science['output_contract']['arrays'];report_path=folder/science['output_contract']['report']
    work={key:0 for key in science['expected_success_counts']};attempts={};saved={};live={};torch=None;storage=None
    report=dict(schema=1,kind=KIND,budget=budget,mode=mode,source=packet['source'],owning_source=science['owning_source'],scientific_contract=sr,
        protocol=dict(path=str(path),sha256=protocol_sha256),passed=False,completed=False,failure=None,gates=[],counts=work,attempts=attempts,
        returned_value_policy='Actual detached aliases registered before guards; CPU owning copies after charged overlap. Fixed current keys replace genuinely unreferenced prior returns; no endpoint-tape list.',
        actual_trajectory25_admitted=False,student_fits=0,test_evaluations=0,all15_open=True,goal_complete=False)
    raw_evidence=dict(path=str(raw_path),exists=False,bytes=None,sha256=None,hash_unknown=True,write_completed=False)
    old_handler,old_timer=signal.getsignal(signal.SIGALRM),signal.getitimer(signal.ITIMER_REAL);old_profile=sys.getprofile()
    def attempt(key):attempts[key]=attempts.get(key,0)+1
    def count(key):work[key]+=1
    def peaks():
        result=dict(seconds=time.monotonic()-started,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,peak_CUDA_allocated_bytes=0,peak_CUDA_reserved_bytes=0)
        if torch is not None and torch.cuda.is_initialized():result.update(peak_CUDA_allocated_bytes=torch.cuda.max_memory_allocated(0),peak_CUDA_reserved_bytes=torch.cuda.max_memory_reserved(0))
        return result
    def guard():
        p=peaks();report['resources']=p;limits=science['resource_protocol']
        require(not stop() and all(p[key]<=limits[key+'_max'] for key in p),'Frozen stop/time/RSS/CUDA cap')
    def gate(name,ok,**detail):
        report['gates'].append(dict(name=name,passed=bool(ok),**observed(detail)));require(ok,name);guard()
    def read(reference):
        require(sha(reference['path'])==reference['sha256'],'Pinned JSON');return json.loads(Path(reference['path']).read_text())
    def pins():
        expected=packet['readonly_files_sha256'];require(all(expected.get(p)==h for p,h in refs(dict(packet=packet,science=science)).items()),'Direct ref closure')
        for pinmap in (science['all24_original_input_pins'],science['all12_original_H_map_Phi_meta_pins']):require(all(expected.get(p)==h for p,h in pinmap.items()),'All24+12 invocation pins')
        root=Path(packet['repository']);require(all(expected.get(str(root/relative))==h for relative,h in packet['source']['files'].items()),'Full invocation source pin closure')
        actual={}
        for p,h in expected.items():guard();actual[p]=sha(p)
        require(actual==expected,'Readonly invocation bytes');return actual
    def alarm(signum,frame):raise TimeoutError('FS300s fixed25 deadline')
    try:
        signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,max(1e-6,300-(time.monotonic()-started)))
        report['readonly_entry']=pins();bridge=read(packet['prerequisite_admission'])
        gate('current_source_static_bridge',bridge['passed'] is True and bridge['source']==packet['source'] and bridge['scientific_contract']==sr
            and all(bridge[key]==packet[key] for key in ('entrypoint','helper_entrypoint','source_owner_entrypoint')))
        freeze=read(packet['ROOT_science_freeze']);gate('ROOT_selected_science',freeze['passed'] is True and freeze['kind']=='ROOT_FS_fixed25_continuation_selection_BEFORE_CODE_v1'
            and freeze['action']=='SELECT_CORA35_ACCEPTED_E1_TO25_ONLY_AND_PRESPECIFY_PAIRED_GCN_RAW_X_MLP_VALIDATION' and freeze['scientific_contract']==sr)
        first=read(science['FIRST_actual_report']);combined=read(science['FIRST_actual_admission']);root=read(science['FIRST_actual_ROOT']);peer=read(science['FIRST_actual_independent'])
        gate('actual_FIRST_ROOT_and_independent',all(row['passed'] is True for row in (first,combined,root,peer))
            and combined['actual_report']==root['actual_report']==peer['actual_report']==science['FIRST_actual_report']
            and combined['actual_arrays']==root['actual_arrays']==peer['actual_arrays']==science['FIRST_actual_arrays']
            and first['source']==combined['source']==root['source']==peer['source']==science['owning_source']
            and first['source_owner']==science['owning_source_owner'] and first['context']==science['owning_context'])
        FR=read(science['FIRST_scientific_contract']);gate('unchanged_original_science',C==FR['case'] and science['original_29_options']==FR['original_29_options'] and science['runtime']==FR['runtime'])
        old_files=science['owning_source']['files'];current=packet['source']['files'];compat=science['invocation_source_compatibility']
        gate('historical185_and_current186_separate',set(current)==set(old_files)|set(compat['new_files']) and len(current)==186
            and all(current[key]==h for key,h in old_files.items() if key not in compat['allowed_diff'])
            and set(compat['allowed_diff'])=={'src/research_loop.py'} and compat['new_files']==['src/quotient_composed_ce_native_trajectory.py'])
        import torch as torch_module
        torch=torch_module
        from src import quotient_composed_ce_native as public
        from src import quotient_composed_ce_native_source as supplier
        from src import joint_mean_ce as joint
        from src.coarsening import feature_centroids,quotient_adjacency
        from src.dual_head_ce import _validate_optimizer
        from src.kernel_mean_ce import _cpu
        from src.low_rank_assignment import LowRankLogits
        from src.moments import augmented
        from src.nystrom_ce import NystromMap,outer_gradient
        from src.research_loop import implementation_provenance
        from src.soft_ce_partition import solve_inner_newton_first,solve_head_system
        from src.transforms import FeatureTransform
        gate('exact_entrypoint_helper_owner',all(sha(module.__file__)==packet[key]['sha256'] and str(Path(module.__file__).resolve())==packet[key]['path']
            for module,key in ((sys.modules[__name__],'entrypoint'),(public,'helper_entrypoint'),(supplier,'source_owner_entrypoint'))))
        report['source_entry']=implementation_provenance();gate('full_current_source_entry',report['source_entry']==packet['source'])
        torch.set_num_threads(4)
        if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(False);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
        torch.cuda.init();device=torch.device('cuda:0');report['runtime']=joint._runtime(device)
        gate('original_native_runtime',report['runtime']==science['runtime'] and torch.get_num_interop_threads()==1 and not torch.is_autocast_enabled('cuda')
            and all(os.environ.get(key)=='4' for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')))
        storage=supplier.ActiveStorageBudget(saved,live,guard,science['resource_protocol'])
        def own(key,value):return storage.own(key,value)
        def load(reference,key,counter):
            attempt(counter);storage.charge('before_mmap_'+key);value=torch.load(reference['path'],mmap=True,map_location='cpu',weights_only=False)
            saved[key]=supplier.detached(value);live[key]=value;count(counter);count('own_PT_supplier_loads');storage.charge('after_mmap_'+key)
            report.setdefault('mapped_payloads',{})[key]=dict(path=reference['path'],file_bytes=Path(reference['path']).stat().st_size,unique_tensor_storage_bytes=supplier.storage_bytes(value),mmap=True,whole_clone=False,virtual_file_bytes_not_RSS=True)
            return value
        accepted=load(science['FIRST_actual_arrays'],'accepted_mapped','accepted_qualification_PT_loads')
        raw_selected=accepted['qualification_state'];attempt('accepted_qualification_state_validations')
        supplier.validate_qualification_state(raw_selected,science['owning_source_owner'],science['owning_context']);count('accepted_qualification_state_validations')
        selected=own('accepted_state',raw_selected);live['accepted_state']=selected;selected_identity=supplier.metadata(selected);report['accepted_state_metadata_seal']=selected['state_metadata_seal']
        # Actual independent owning selection precedes removal of all full435MB aliases.
        del raw_selected,accepted;saved.pop('accepted_mapped');live.pop('accepted_mapped');storage.charge('after_release_nonselected_FIRST_payload')
        report['FIRST_mapped_lifetime']='Only own copied qualification_state retained; all other FIRST payload references removed before supplier reload. Original file immutable/virtual length disclosed.'
        FA=C['FA153'];L=C['graph_supplier'];cap=read(FA['report']);fr,fp=(read(FA[key]) for key in ('ROOT','independent'))
        gate('literal_FA153_admission',cap['passed'] is True and cap['capture_completed'] is True and cap['budget']==budget and cap['source']==cap['source_entry']==cap['source_exit']==fr['source']==fp['source']
            and cap['scientific_contract']==FA['scientific_contract'] and fr['passed'] is True and fp['passed'] is True
            and fr['budgets'][budget]['report']==fp['budgets'][budget]['actual_report']==FA['report'] and fr['budgets'][budget]['arrays']==fp['budgets'][budget]['arrays']==FA['arrays']
            and cap['raw_evidence']['sha256']==FA['arrays']['sha256'] and Path(FA['arrays']['path']).stat().st_size==FA['expected_array_bytes'] and cap['original_data_digest']==C['original_data_digest'])
        prior,cache_root,cache_peer=(read(L[key]) for key in ('actual_report','ROOT','independent'));fp_science=read(C['WHOLE_cache_authority']['scientific_contract']);W=fp_science['cases'][budget]['WHOLE_old_cache']
        gate('literal_WHOLE_FD159_admission',prior['passed'] is True and cache_root['passed'] is True and cache_peer['passed'] is True
            and cache_root['actual_report']==cache_peer['actual_report']==L['actual_report'] and prior['shared_cache']==cache_root['shared_cache']==cache_peer['shared_cache']==L['shared_cache']
            and prior['source']==prior['source_entry']==prior['source_exit']==cache_root['source']==cache_peer['source']==W['literal_header']['source']
            and prior['cache_identities']==cache_root['cache_identities']==cache_peer['cache_identities']==W['ALL5_identities'] and Path(L['shared_cache']['path']).stat().st_size==L['actual_cache_bytes'])
        old_science=read(C['WHOLE_cache_authority']['old_scientific_contract']);OB=old_science['budgets'][budget];providers=dict(FA153=OB['source_FA'],old_controls=OB['control_suppliers'],native25=OB['native25'])
        gate('reconstructed_WHOLE_private_provider',providers==prior['upstream_providers']==cache_root['upstream_providers']==cache_peer['upstream_providers']==W['literal_upstream_providers']
            and joint._seal(providers)==W['upstream_providers_seal'] and prior['owning_content_metadata']==cache_root['owning_content_metadata']==cache_peer['owning_content_metadata']==W['private_owning_metadata']['expected_metadata'])
        family=read(OB['family_config']);gate('original_graph_family_dataset_digest',family['data_digest']==FR['sourceowner_API']['graph_family_dataset_digest']==prior['original_graph_dataset_digest'])
        captured=load(FA['arrays'],'FA153_mapped','FA_source_PT_loads');historical=load(L['shared_cache'],'FD159_mapped','WHOLE_graph_cache_PT_loads')
        stable=dict(case=C,WHOLE_old_cache=W,FA_literal_producer=cap['source'],current_source=science['owning_source'],stable_pins=science['owning_source_owner']['all_source_pin_map'],
            graph_family_dataset_digest=family['data_digest'],selected_graph_family_digest=FR['sourceowner_API']['graph_family_dataset_digest'])
        storage.charge('before_original_identity_hashing_constructor');owner=supplier.OwnOriginalROWQuotientSource(captured,historical,stable,science['original_29_options']);live['source_owner']=vars(owner)
        gate('exact_source185_owner_reconstructed',owner.descriptor==science['owning_source_owner']==selected['source_owner']);context=selected['context'];CE0=context['new_positive_CE0']
        inputs=owner.device_inputs(device,storage);live['inputs']=inputs;options=science['original_29_options'];d=C['dimensions'];n,k,r,c=(d[x] for x in ('N','K','rank','C'))
        transform=FeatureTransform(**inputs['transform']);feature_map=NystromMap(inputs['anchors'],inputs['mapping'],'relu');live['map_RMS']=dict(transform=vars(transform),map=vars(feature_map))
        npair=selected['native_parameters'];storage.charge('before_native_parameter_restore',2*supplier.copy_bytes([npair['U'],npair['V']]))
        u=npair['U'].to(device).detach().clone().requires_grad_(True);v=npair['V'].to(device).detach().clone().requires_grad_(True)
        live['parameters_optimizer']=dict(U=u,V=v);storage.charge('restored_native_parameters')
        adam=torch.optim.Adam([u,v],lr=options['lr'],eps=1e-12,foreach=False);own('Adam_restore',npair['optimizer']);storage.charge('before_deepcopy_CPU_Adam_slots',supplier.copy_bytes(saved['Adam_restore']))
        adam_restore=copy.deepcopy(saved['Adam_restore']);live['Adam_restore']=adam_restore
        storage.charge('own_deepcopy_Adam_restore');require(_validate_optimizer(adam_restore,options,dict(nodes=n,cells=k,rank=r))==1,'Original Adam step1');adam.load_state_dict(adam_restore)
        live['parameters_optimizer']=dict(U=u,V=v,optimizer=adam.state_dict());storage.charge('restored_Adam');E=selected['endpoint1'];live['current_endpoint']=E
        history=[dict(row) for row in first['endpoint_summaries']];report['history']=history
        counted={id(feature_centroids.__code__):'CUDA_original_feature_centroids_calls',id(quotient_adjacency.__code__):'CUDA_original_quotient_adjacency_calls',id(NystromMap.__call__.__code__):'CUDA_original_map_calls'}
        def profile(frame,event,arg):
            name=counted.get(id(frame.f_code))
            if name and event in ('call','return'):
                value=frame.f_locals.get('probability',frame.f_locals.get('h'));require(value.device.type=='cuda','No unexpected CPU/source original calls')
                if event=='call':attempt(name)
                elif arg is not None:count(name)
            if old_profile is not None:old_profile(frame,event,arg)
        sys.setprofile(profile)
        def observer(key,value):
            if key=='clamp_observation':count('CUDA_clamp_observation_records')
            own('current_'+key,value)
        attempt('core_LowRankLogits_forward');logits=LowRankLogits.apply(u,v,inputs['hard'],options['mixing'],options['chunk_size']);count('core_LowRankLogits_forward');own('current_logits',logits)
        P=logits.double().softmax(1);own('current_P',P)
        gate('accepted_P1_exact_reattachment',supplier.identity(P)==supplier.identity(E['P']) and P.dtype==E['P'].dtype and tuple(P.shape)==tuple(E['P'].shape)
            and P.detach().cpu().contiguous().numpy().tobytes()==E['P'].contiguous().numpy().tobytes());count('accepted_P1_reattachments')
        for step in range(2,26):
            live['current_endpoint']=E;live['current_tape']=dict(logits=logits,P=P)
            adam.zero_grad(set_to_none=True);R=E['complete_R'].to(device);live['pending_R']=R;storage.charge('before_native_backward')
            attempt('core_LowRankLogits_backward');P.backward(R);count('core_LowRankLogits_backward');own('current_native_gradient',dict(U=u.grad,V=v.grad))
            gate('finite_native_gradient',all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in (u,v)),update_to=step)
            attempt('Adam_updates');adam.step();count('Adam_updates');count('P_updates');live['parameters_optimizer']=dict(U=u,V=v,optimizer=adam.state_dict());own('current_native_parameters',dict(U=u,V=v,optimizer=adam.state_dict()))
            # Cached E1 and preceding endpoints survive until successful replacement.
            previous=E;live['warm_previous_endpoint']=previous
            attempt('core_LowRankLogits_forward');logits=LowRankLogits.apply(u,v,inputs['hard'],options['mixing'],options['chunk_size']);count('core_LowRankLogits_forward');own('current_logits',logits)
            P=logits.double().softmax(1);own('current_P',P);live['current_tape']=dict(logits=logits,P=P)
            attempt('CUDA_endpoint_builds');chain=public.build_quotient_features(P.detach(),inputs['X'],inputs['Q'],inputs['S'],transform,feature_map,observer);count('CUDA_endpoint_builds');live['current_chain']=chain
            weights=P.new_full((k,),1/k);live['uniform_weights']=weights;theta_initial=previous['theta'].to(device);vector_initial=previous['vector'].to(device);live['warm_inputs']=dict(theta=theta_initial,vector=vector_initial)
            with storage.solver('E'+str(step)+'_head',2*c*(d['D']+d['B']+1)*8) as phase:
                attempt('stationary_head_solves');head=solve_inner_newton_first(chain['features'].detach(),chain['Qc'].detach(),weights,options['penalty'],initial=theta_initial,
                    max_iter=options['inner_max_iter'],grad_tol=options['inner_tol'],cg_max_iter=options['cg_max_iter'],cg_check_interval=options.get('cg_check_interval',1))
                live['last_head']=head;saved['last_head_raw']=supplier.detached(head);phase['returned']=True;count('stationary_head_solves');own('current_head',head)
                theta=head['theta'];gate('stationary_head',head['inner_converged'] is True and type(head['inner_grad_max']) in (int,float) and math.isfinite(head['inner_grad_max'])
                    and 0<=head['inner_grad_max']<=options['inner_tol'] and bool(torch.isfinite(theta).all()) and tuple(theta.shape)==tuple(d['theta_RHS_vector']),step=step)
            known_outer_copy=2*(n*(d['D']+d['B']+1)*8+n*c*8);storage.charge('before_original_outer_CE_RHS',known_outer_copy)
            attempt('raw_fixedsource_outer_CE_RHS');raw_ce,rhs=outer_gradient(owner.outer,inputs['Q'],theta,options['outer_chunk_size']);count('raw_fixedsource_outer_CE_RHS');own('current_outer',dict(raw_CE=raw_ce,rhs=rhs))
            gate('raw_fixedsource_outer',math.isfinite(raw_ce) and raw_ce>0 and bool(torch.isfinite(rhs).all()),step=step)
            with storage.solver('E'+str(step)+'_raw_adjoint',2*c*(d['D']+d['B']+1)*8) as phase:
                attempt('full_implicit_adjoints');vector,diag=solve_head_system(augmented(chain['features'].detach()),chain['Qc'].detach(),weights,theta,options['penalty'],rhs,
                    rtol=options['cg_rtol'],max_iter=options['cg_max_iter'],initial=vector_initial,cg_check_interval=options.get('cg_check_interval',1))
                actual=dict(vector=vector,diagnostic=diag);live['last_adjoint']=actual;saved['last_adjoint_raw']=supplier.detached(actual);phase['returned']=True;count('full_implicit_adjoints');own('current_raw_adjoint',actual)
                gate('original_raw_adjoint',diag['cg_converged'] is True and bool(torch.isfinite(vector).all()) and tuple(vector.shape)==tuple(d['theta_RHS_vector']),step=step)
            attempt('CUDA_completeR_builds');result=public.complete_row_cotangent(P.detach(),inputs['X'],inputs['Q'],inputs['S'],transform,feature_map,theta,vector,options['penalty'],CE0,observer);count('CUDA_completeR_builds')
            endpoint=dict(step=step,P=P.detach(),chain=chain,theta=theta,vector=vector,head=head,adjoint=diag,raw_CE=float(raw_ce),CE0=CE0,complete_R=result['complete_R'],cotangent=result,
                native_parameter_identities=supplier.metadata([u,v]))
            E=own('current_endpoint',endpoint);live['current_endpoint']=E
            history.append(dict(step=step,raw_CE=E['raw_CE'],CE0=CE0,ratio=E['raw_CE']/CE0,head={key:value for key,value in E['head'].items() if key!='theta'},adjoint=E['adjoint'],endpoint_metadata_seal=joint._seal(observed(supplier.metadata(E)))))
            # Remove genuinely unused preceding endpoint references, not retained bookkeeping alone.
            previous=None;live.pop('warm_previous_endpoint');theta_initial=vector_initial=None;live.pop('warm_inputs');storage.charge('current_endpoint_replaced')
        gate('accepted_FIRST_state_unchanged',supplier.metadata(selected)==selected_identity)
        gate('exact_selected_counts_before_terminal',all(work[key]==value for key,value in science['expected_success_counts'].items() if key not in ('owning_terminal_arrays_write','final_JSON_report_write')))
        gate('exact48_original_solver_scopes',[row['role'] for row in storage.phases]==science['resource_protocol']['solver_scopes'] and all(row['completed'] and row['restored64'] for row in storage.phases))
        config=dict(policy=POLICY,original29_options=options,dimensions=d,scientific_endpoint=25,own_origin_digest=context['own_origin_digest'],expected_counts=science['expected_success_counts'])
        state=dict(schema=1,kind=KIND,step=25,scientific_endpoint=25,invocation_source=packet['source'],owning_source=science['owning_source'],source_owner=owner.descriptor,context=context,
            config=config,config_digest=joint._seal(config),current_endpoint=E,native_parameters=saved['current_native_parameters'],history=history,counts=dict(work),FIRST_actual_arrays=science['FIRST_actual_arrays'])
        state['state_metadata_seal']=joint._seal(observed(supplier.metadata(state)));own('trajectory_state',state);validate_trajectory_state(saved['trajectory_state'],owner.descriptor,context,packet['source'],config)
        report.update(context=context,source_owner=owner.descriptor,CE0=CE0,E25_raw_CE=E['raw_CE'],E25_ratio=E['raw_CE']/CE0,state_metadata_seal=state['state_metadata_seal'],config=config,endpoint25_metadata=supplier.metadata(E),completed=True)
    except BaseException as error:
        report['failure']=dict(type=type(error).__name__,message=str(error));report['completed']=False
    finally:
        sys.setprofile(old_profile);signal.setitimer(signal.ITIMER_REAL,0)
        try:
            if torch is not None:
                terminal={'trajectory_state':saved['trajectory_state']} if report['completed'] else {key:value for key,value in saved.items() if key not in ('FA153_mapped','FD159_mapped')}
                live['terminal_output']=terminal
                try:
                    attempt('owning_terminal_arrays_write')
                    try:
                        if storage is not None:storage.charge('before_terminal_serialization',supplier.storage_bytes(terminal))
                        guard()
                    except BaseException as error:
                        report['serialization_guard_failure']=dict(type=type(error).__name__,message=str(error));report['completed']=False
                        if report['failure'] is None:report['failure']=report['serialization_guard_failure']
                    with raw_path.open('xb') as stream:torch.save(terminal,stream);stream.flush();os.fsync(stream.fileno())
                    raw_evidence['write_completed']=True;count('owning_terminal_arrays_write')
                finally:
                    raw_evidence['exists']=raw_path.exists()
                    if raw_path.exists():raw_evidence['bytes']=raw_path.stat().st_size
                raw_evidence['sha256']=sha(raw_path);raw_evidence['hash_unknown']=False
                if storage is not None:storage.charge('after_terminal_serialization')
        except BaseException as error:
            report['raw_serialization_error']=dict(type=type(error).__name__,message=str(error));report['completed']=False
            if report['failure'] is None:report['failure']=report['raw_serialization_error']
        report['raw_evidence']=raw_evidence
        try:
            report['readonly_exit']=pins()
            if torch is not None:report['source_exit']=implementation_provenance();require(report['source_exit']==packet['source'],'Current fullsource exit')
            guard();require(work['owning_terminal_arrays_write']==1,'Owning output count')
        except BaseException as error:
            report['exit_guard_error']=dict(type=type(error).__name__,message=str(error));report['completed']=False
            if report['failure'] is None:report['failure']=report['exit_guard_error']
        count('final_JSON_report_write');report['resources']=peaks()
        report['passed']=report['completed'] and report['failure'] is None and work==science['expected_success_counts'] and all(row['passed'] for row in report['gates']);report['actual_trajectory25_admitted']=report['passed']
        if not report['passed']:report['completed']=False
        if storage is not None:report['active_storage']=dict(peak_charged_bytes=storage.peak,observations=storage.observations,solver_phases=storage.phases)
        try:
            with report_path.open('x') as stream:json.dump(observed(report),stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
            guard()
        except BaseException as error:
            report.update(passed=False,completed=False,actual_trajectory25_admitted=False);report['final_serialization_error']=dict(type=type(error).__name__,message=str(error))
            if report['failure'] is None:report['failure']=report['final_serialization_error']
            report['resources']=peaks()
            if storage is not None:report['active_storage']=dict(peak_charged_bytes=storage.peak,observations=storage.observations,solver_phases=storage.phases)
            with report_path.open('w') as stream:json.dump(observed(report),stream,indent=2,allow_nan=False);stream.write('\n')
        signal.signal(signal.SIGALRM,old_handler);signal.setitimer(signal.ITIMER_REAL,*old_timer)
    if not report['passed']:raise RuntimeError('Terminal FS continuation nonadmission: '+str(report_path))
    return str(report_path)
