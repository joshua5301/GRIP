"""FIRST FR Cora35 native0->1 qualifier; no continuation/student authorization.

ROOT supplies an exclusive current full-source protocol after exact static review.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import sys
import time

KIND='original_ROW_CSR_quotient_SGC_original_Nystrom_composed_uniform_CE_native_FIRST0to1_v1'
SCIENCE_SHA='5a3fd81488762db5f462829e37a065bfa4e93e54b944131469420d4dd9be3a05'


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


def run(protocol_path,protocol_sha256,budget='cora35',mode='qualify0to1',stop=lambda:False):
    started=time.monotonic();path=Path(protocol_path)
    require(path.is_absolute() and str(path.resolve())==str(path) and sha(path)==protocol_sha256,'Protocol bytes')
    packet=json.loads(path.read_text());sr=packet['scientific_contract']
    require(sr['sha256']==SCIENCE_SHA and sha(sr['path'])==SCIENCE_SHA,'Exact selected science')
    science=json.loads(Path(sr['path']).read_text());C=science['case']
    require(packet['kind']==science['kind']==KIND and budget=='cora35' and mode=='qualify0to1','FIRST Cora35-only scope')
    folder=Path(packet['output_folder']);require(folder.is_absolute() and str(folder.resolve())==str(folder),'ABS output');folder.mkdir(parents=True,exist_ok=False)
    raw_path=folder/'qualification_arrays.pt';report_path=folder/'qualification_report.json'
    work={key:0 for key in science['expected_success_counts']};attempts={};saved={};live={};torch=None;storage=None
    report=dict(schema=1,kind=KIND,budget=budget,mode=mode,source=packet['source'],scientific_contract=sr,
        protocol=dict(path=str(path),sha256=protocol_sha256),passed=False,completed=False,failure=None,gates=[],counts=work,attempts=attempts,
        returned_value_policy='Detached actual aliases registered before precharge; owning CPU copies replace only after accepted copy charge. Raw failure aliases retain actual storage/device; no whole supplier clone.',
        actual_real_native_admission=False,trajectory25=False,student_cohort=None,all15_open=True,goal_complete=False)
    raw_evidence=dict(path=str(raw_path),exists=False,bytes=None,sha256=None,hash_unknown=True,write_completed=False)
    old_handler,old_timer=signal.getsignal(signal.SIGALRM),signal.getitimer(signal.ITIMER_REAL);old_profile=sys.getprofile()
    def attempt(key):attempts[key]=attempts.get(key,0)+1
    def count(key):work[key]+=1
    def peaks():
        result=dict(seconds=time.monotonic()-started,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            peak_CUDA_allocated_bytes=0,peak_CUDA_reserved_bytes=0)
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
        expected=packet['readonly_files_sha256'];require(all(expected.get(p)==h for p,h in refs(dict(packet=packet,science=science)).items()),'Direct reference closure')
        for pinmap in (science['all24_original_input_pins'],science['all12_original_H_map_Phi_meta_pins']):
            require(all(expected.get(p)==h for p,h in pinmap.items()),'All24+12 invocation source pins')
        actual={}
        for p,h in expected.items():guard();actual[p]=sha(p)
        require(actual==expected,'Readonly source bytes');return actual
    def alarm(signum,frame):raise TimeoutError('FR300s native deadline')
    try:
        signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,max(1e-6,300-(time.monotonic()-started)))
        report['readonly_entry']=pins()
        bridge=read(packet['prerequisite_admission'])
        gate('current_source_static_bridge',bridge['passed'] is True and bridge['source']==packet['source'] and bridge['scientific_contract']==sr
            and all(bridge[key]==packet[key] for key in ('entrypoint','helper_entrypoint','source_owner_entrypoint')))
        freeze=read(packet['ROOT_science_freeze']);gate('ROOT_selected_science',freeze['passed'] is True and freeze['kind']=='ROOT_FR_Cora35_REAL_native_science_selection_BEFORE_CODE_v1' and freeze['action']=='SELECT_FIRST_CORa35_NATIVE_0_TO_1_ONLY' and freeze['scientific_contract']==sr)
        CPU=read(science['admitted_tiny_CPU_proof']);gate('actual_tiny_CPU_only_prerequisite',CPU['passed'] is True)
        import numpy as np
        import torch as torch_module
        torch=torch_module
        from src import quotient_composed_ce_native as public
        from src import quotient_composed_ce_native_source as supplier
        from src import joint_mean_ce as joint
        from src.coarsening import feature_centroids,quotient_adjacency
        from src.dual_head_ce import _factor_digests
        from src.kernel_mean_ce import _cpu
        from src.low_rank_assignment import LowRankLogits,logit_block
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
        gate('original_native_runtime',report['runtime']==science['runtime'] and torch.get_num_interop_threads()==1
            and not torch.is_autocast_enabled('cuda') and all(os.environ.get(key)=='4' for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')))
        storage=supplier.ActiveStorageBudget(saved,live,guard,science['resource_protocol'])
        def own(key,value):return storage.own(key,value)
        FA=C['FA153'];cap=read(FA['report']);fr,fp=(read(FA[key]) for key in ('ROOT','independent'))
        gate('literal_FA153_admission',cap['passed'] is True and cap['capture_completed'] is True and cap['budget']==budget
            and cap['source']==cap['source_entry']==cap['source_exit']==fr['source']==fp['source'] and cap['scientific_contract']==FA['scientific_contract']
            and fr['passed'] is True and fp['passed'] is True and fr['budgets'][budget]['report']==fp['budgets'][budget]['actual_report']==FA['report']
            and fr['budgets'][budget]['arrays']==fp['budgets'][budget]['arrays']==FA['arrays'] and cap['raw_evidence']['sha256']==FA['arrays']['sha256']
            and Path(FA['arrays']['path']).stat().st_size==FA['expected_array_bytes'] and cap['original_data_digest']==C['original_data_digest'])
        L=C['graph_supplier'];prior,root,peer=(read(L[key]) for key in ('actual_report','ROOT','independent'))
        fp_science=read(C['WHOLE_cache_authority']['scientific_contract']);W=fp_science['cases'][budget]['WHOLE_old_cache']
        gate('literal_WHOLE_FD159_admission',prior['passed'] is True and root['passed'] is True and peer['passed'] is True
            and root['actual_report']==peer['actual_report']==L['actual_report'] and prior['shared_cache']==root['shared_cache']==peer['shared_cache']==L['shared_cache']
            and prior['source']==prior['source_entry']==prior['source_exit']==root['source']==peer['source']==W['literal_header']['source']
            and prior['cache_identities']==root['cache_identities']==peer['cache_identities']==W['ALL5_identities']
            and Path(L['shared_cache']['path']).stat().st_size==L['actual_cache_bytes'])
        old_science=read(C['WHOLE_cache_authority']['old_scientific_contract']);OB=old_science['budgets'][budget]
        providers=dict(FA153=OB['source_FA'],old_controls=OB['control_suppliers'],native25=OB['native25'])
        gate('reconstructed_WHOLE_private_provider',providers==prior['upstream_providers']==root['upstream_providers']==peer['upstream_providers']==W['literal_upstream_providers']
            and joint._seal(providers)==W['upstream_providers_seal'] and prior['owning_content_metadata']==root['owning_content_metadata']==peer['owning_content_metadata']==W['private_owning_metadata']['expected_metadata'])
        family=read(OB['family_config']);gate('original_graph_family_dataset_digest',family['data_digest']==science['sourceowner_API']['graph_family_dataset_digest']
            and prior['original_graph_dataset_digest']==family['data_digest'])
        def load(reference,key,counter):
            attempt(counter);storage.charge('before_mmap_'+key);value=torch.load(reference['path'],mmap=True,map_location='cpu',weights_only=False)
            saved[key]=supplier.detached(value);live[key]=value;count(counter);count('own_PT_supplier_loads');storage.charge('after_mmap_'+key)
            report.setdefault('mapped_payloads',{})[key]=dict(path=reference['path'],file_bytes=Path(reference['path']).stat().st_size,
                unique_tensor_storage_bytes=supplier.storage_bytes(value),mmap=True,whole_clone=False,virtual_file_bytes_not_RSS=True)
            return value
        captured=load(FA['arrays'],'FA153_mapped','FA_source_PT_loads');historical=load(L['shared_cache'],'FD159_mapped','WHOLE_graph_cache_PT_loads')
        stable_pins=dict(science['all24_original_input_pins']);stable_pins.update(science['all12_original_H_map_Phi_meta_pins'])
        repository=Path(packet['entrypoint']['path']).parent.parent
        stable_pins.update({str((repository/relative).resolve()):h for relative,h in packet['source']['files'].items()})
        for p,h in refs(dict(FA153=FA,WHOLE_graph=L,authority=C['WHOLE_cache_authority'],CPU=science['admitted_tiny_CPU_proof'])).items():stable_pins[p]=h
        stable=dict(case=C,WHOLE_old_cache=W,FA_literal_producer=cap['source'],current_source=packet['source'],stable_pins=stable_pins,
            graph_family_dataset_digest=family['data_digest'],selected_graph_family_digest=science['sourceowner_API']['graph_family_dataset_digest'])
        storage.charge('before_original_identity_hashing_constructor')
        report['identity_hash_storage_scope']='Original array_digest z.tobytes is31,044,512B and original mapping/row identity hash temporaries fit the separately charged ordinary64MiB reserve;4MiB scratch covers wrapper file hashing only.'
        owner=supplier.OwnOriginalROWQuotientSource(captured,historical,stable,science['original_29_options']);live['source_owner']=vars(owner)
        report['source_owner']=owner.descriptor;gate('typed_own_source_validated',owner.descriptor['original_data_digest']==C['original_data_digest'])
        inputs=owner.device_inputs(device,storage);live['inputs']=inputs;options=science['original_29_options'];d=C['dimensions'];n,k,r,c=(d[x] for x in ('N','K','rank','C'))
        transform=FeatureTransform(**inputs['transform']);feature_map=NystromMap(inputs['anchors'],inputs['mapping'],'relu')
        live['map_RMS']=dict(transform=vars(transform),map=vars(feature_map));u=inputs['U0'].detach().clone().requires_grad_(True);v=inputs['V0'].detach().clone().requires_grad_(True)
        optimizer=torch.optim.Adam([u,v],lr=options['lr'],eps=1e-12,foreach=False);live['parameters_optimizer']=dict(u=u,v=v,optimizer=optimizer.state_dict())
        counted={id(feature_centroids.__code__):'original_feature_centroids_calls',id(quotient_adjacency.__code__):'original_quotient_adjacency_calls',id(NystromMap.__call__.__code__):'original_map_calls'}
        def profile(frame,event,arg):
            name=counted.get(id(frame.f_code))
            if name and event in ('call','return'):
                value=frame.f_locals.get('probability',frame.f_locals.get('h'));domain='CPU' if value.device.type=='cpu' else 'CUDA';key=domain+'_'+name
                if event=='call':attempt(key)
                elif arg is not None:count(key)
            if old_profile is not None:old_profile(frame,event,arg)
        sys.setprofile(profile)
        def observer(role,domain):
            def returned(key,value):
                if key in ('clamp_observation','CPU_clamp_observation'):count(domain+'_clamp_observation_records')
                own(role+'_'+key,value)
            return returned
        endpoints=[];live['endpoints']=endpoints;CE0=None;context=None
        for step in (0,1):
            attempt('core_LowRankLogits_forward');logits=LowRankLogits.apply(u,v,inputs['hard'],options['mixing'],options['chunk_size']);count('core_LowRankLogits_forward');own('E'+str(step)+'_logits',logits)
            P=logits.double().softmax(1);own('E'+str(step)+'_P',P)
            attempt('CUDA_endpoint_builds');chain=public.build_quotient_features(P.detach(),inputs['X'],inputs['Q'],inputs['S'],transform,feature_map,observer('E'+str(step),'CUDA'));count('CUDA_endpoint_builds');live['current_chain']=chain
            weights=P.new_full((k,),1/k);live['uniform_weights']=weights
            with storage.solver('baseline_head0' if step==0 else 'current_head1',2*c*(d['D']+d['B']+1)*8) as phase:
                attempt('stationary_head_solves');head=solve_inner_newton_first(chain['features'].detach(),chain['Qc'].detach(),weights,options['penalty'],
                    initial=None if step==0 else endpoints[0]['theta'].detach(),max_iter=options['inner_max_iter'],grad_tol=options['inner_tol'],
                    cg_max_iter=options['cg_max_iter'],cg_check_interval=options.get('cg_check_interval',1))
                live['last_head']=head;saved['last_head_raw']=supplier.detached(head);phase['returned']=True;count('stationary_head_solves');own('E'+str(step)+'_head',head)
                theta=head['theta'];gate('stationary_head'+str(step),head['inner_converged'] is True and type(head['inner_grad_max']) in (int,float)
                    and math.isfinite(head['inner_grad_max']) and 0<=head['inner_grad_max']<=options['inner_tol'] and bool(torch.isfinite(theta).all()) and tuple(theta.shape)==tuple(d['theta_RHS_vector']))
            known_outer_copy=2*(n*(d['D']+d['B']+1)*8+n*c*8)
            report.setdefault('original_outer_copy_observations',[]).append(dict(step=step,original_numpy_rows_copy_bytes=n*(d['D']+d['B'])*8,conservative_simultaneous_future_copy_bytes=known_outer_copy))
            storage.charge('before_original_outer_CE_RHS',known_outer_copy)
            attempt('raw_fixedsource_outer_CE_RHS');raw_ce,rhs=outer_gradient(owner.outer,inputs['Q'],theta,options['outer_chunk_size'])
            count('raw_fixedsource_outer_CE_RHS');own('E'+str(step)+'_outer',dict(raw_CE=raw_ce,rhs=rhs))
            gate('raw_fixedsource_outer'+str(step),type(raw_ce) in (int,float) and math.isfinite(raw_ce) and raw_ce>0 and bool(torch.isfinite(rhs).all()))
            if step==0:CE0=float(raw_ce)
            with storage.solver('baseline_raw_adjoint0' if step==0 else 'current_raw_adjoint1',2*c*(d['D']+d['B']+1)*8) as phase:
                attempt('full_implicit_adjoints');vector,diag=solve_head_system(augmented(chain['features'].detach()),chain['Qc'].detach(),weights,theta,options['penalty'],rhs,
                    rtol=options['cg_rtol'],max_iter=options['cg_max_iter'],initial=None if step==0 else endpoints[0]['vector'].detach(),cg_check_interval=options.get('cg_check_interval',1))
                actual=dict(vector=vector,diagnostic=diag);live['last_adjoint']=actual;saved['last_adjoint_raw']=supplier.detached(actual);phase['returned']=True
                count('full_implicit_adjoints');own('E'+str(step)+'_raw_adjoint',actual)
                gate('original_raw_adjoint'+str(step),diag['cg_converged'] is True and bool(torch.isfinite(vector).all()) and tuple(vector.shape)==tuple(d['theta_RHS_vector']))
            attempt('CUDA_completeR_builds');result=public.complete_row_cotangent(P.detach(),inputs['X'],inputs['Q'],inputs['S'],transform,feature_map,theta,vector,options['penalty'],CE0,observer('A'+str(step),'CUDA'));count('CUDA_completeR_builds')
            endpoint=dict(step=step,P=P.detach(),chain=chain,theta=theta,vector=vector,head=head,adjoint=diag,raw_CE=float(raw_ce),CE0=CE0,complete_R=result['complete_R'],cotangent=result)
            endpoints.append(endpoint);own('E'+str(step)+'_complete',endpoint)
            if step==0:
                context=owner.origin_context(endpoint,CE0);supplier.validate_context(context);report['context']=context;own('new_context',context)
                optimizer.zero_grad(set_to_none=True);attempt('core_LowRankLogits_backward');P.backward(result['complete_R']);count('core_LowRankLogits_backward')
                own('baseline_native_gradient',dict(U=u.grad,V=v.grad));gate('baseline_native_finite',all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in (u,v)))
                attempt('Adam_updates');optimizer.step();count('Adam_updates');count('P_updates');live['parameters_optimizer']=dict(u=u,v=v,optimizer=optimizer.state_dict());own('Adam_update1',dict(U=u,V=v,optimizer=optimizer.state_dict()))
        E1=endpoints[1];report['CE0']=CE0;report['E1_raw_CE']=E1['raw_CE'];report['E1_ratio']=E1['raw_CE']/CE0
        cpu_inputs=dict(P=E1['P'],theta=E1['theta'],vector=E1['vector']);cpu_inputs=own('held_CPU_inputs',cpu_inputs)
        cpu_transform=owner.transform;cpu_map=owner.feature_map;X=owner.X.double();Q=owner.Q;S=owner.S.double();live['CPU_sources']=dict(X=X,Q=Q,S=S,transform=vars(cpu_transform),map=vars(cpu_map))
        attempt('CPU_held_completeR_builds');CPU_R=public.independent_CPU_complete_row(cpu_inputs['P'],X,Q,S,cpu_transform,cpu_map,cpu_inputs['theta'],cpu_inputs['vector'],options['penalty'],CE0,observer('held_CPU','CPU'));count('CPU_held_completeR_builds');own('CPU_full_cotangent',CPU_R)
        def compare(name,actual,reference,limits):
            a=actual.detach().cpu().double();b=reference.detach().cpu().double();delta=a-b;live['compare_'+name]=dict(actual=a,reference=b,delta=delta);own('compare_'+name,live['compare_'+name])
            absolute=float(delta.abs().max());den=float(b.norm());num=float(delta.norm());relative=num/den if den>0 else None
            gate(name,bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all()) and den>0 and math.isfinite(absolute) and relative is not None
                and math.isfinite(relative) and absolute<=limits['absolute_Linf_max'] and relative<=limits['relative_L2_max'],absolute_Linf=absolute,relative_L2=relative,actual_reference_norm=den)
        compare('complete_R_GPU_CPU',E1['complete_R'],CPU_R['complete_R'],science['completeR_comparator']['gates'])
        gpu_clamps=saved['A1_clamp_observation'];cpu_clamps=saved['held_CPU_CPU_clamp_observation']
        def equal_tree(a,b):
            if torch.is_tensor(a):return a.dtype==b.dtype and tuple(a.shape)==tuple(b.shape) and torch.equal(a,b)
            return set(a)==set(b) and all(equal_tree(a[key],b[key]) for key in a)
        gate('held_CPU_GPU_original_clamp_branch_masks',equal_tree(gpu_clamps['masks'],cpu_clamps['masks']))
        repeats=[];live['native_repeats']=repeats;held=E1['complete_R'].detach()
        for repeat in range(3):
            uu=u.detach().clone().requires_grad_(True);vv=v.detach().clone().requires_grad_(True);live['repeat_parameters']=dict(U=uu,V=vv)
            attempt('diagnostic_sameR_native_forward');ll=LowRankLogits.apply(uu,vv,inputs['hard'],options['mixing'],options['chunk_size']);count('diagnostic_sameR_native_forward');own('repeat_logits'+str(repeat),ll)
            pp=ll.double().softmax(1);own('repeat_P'+str(repeat),pp);attempt('diagnostic_sameR_native_backward');du,dv=torch.autograd.grad((pp*held).sum(),(uu,vv));count('diagnostic_sameR_native_backward')
            row=own('repeat'+str(repeat),dict(logits=ll,P=pp,U=du,V=dv));repeats.append(row)
            gate('nonzero_held_native_gradient'+str(repeat),all(bool(torch.isfinite(t).all()) and float(t.norm())>0 for t in (du,dv)))
        def bytes_equal(a,b):return a.dtype==b.dtype and tuple(a.shape)==tuple(b.shape) and bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all()) and a.contiguous().numpy().tobytes()==b.contiguous().numpy().tobytes()
        gate('three_same_R_native_byte_equal',all(bytes_equal(repeats[0][key],row[key]) for row in repeats[1:] for key in repeats[0]))
        U,V=own('CPU_native_parameters',dict(U=u,V=v)).values();hard=owner.hard
        attempt('CPU_original_logit_block_reference');cl=logit_block(U,V,hard,options['mixing']);count('CPU_original_logit_block_reference');own('CPU_native_logits',cl)
        cp=cl.double().softmax(1);rr=saved['E1_complete']['complete_R'];W64=cp*(rr-(cp*rr).sum(1,keepdim=True));block=W64.float()/math.sqrt(r)
        du=block@V;dv=block.T@U;reference=dict(logits=cl,P=cp,W64=W64,wholeK_FP32_scaled_block=block,U=du,V=dv);count('CPU_wholeK_FP32_factor_reference');own('CPU_wholeK_reference',reference)
        for name,a,b in [('native_U',repeats[0]['U'],du),('native_V',repeats[0]['V'],dv),('native_full',torch.cat((repeats[0]['U'].flatten(),repeats[0]['V'].flatten())),torch.cat((du.flatten(),dv.flatten())))]:compare(name,a,b,science['native_operator']['gates'])
        report['endpoint_summaries']=[dict(step=E['step'],raw_CE=E['raw_CE'],CE0=CE0,ratio=E['raw_CE']/CE0,head=observed({k:v for k,v in E['head'].items() if k!='theta'}),adjoint=observed(E['adjoint'])) for E in endpoints]
        report['native_repeats']=[supplier.metadata(row) for row in repeats];report['context_digest']=joint._seal(context)
        state=dict(schema=1,mode=KIND,current_source=packet['source'],source_owner=owner.descriptor,context=context,endpoint1=saved['E1_complete'],native_parameters=saved['Adam_update1'],counts=dict(work))
        state['state_metadata_seal']=joint._seal(observed(dict(context=context,endpoint=supplier.metadata(state['endpoint1']),parameters=supplier.metadata(state['native_parameters']),counts=work)));own('qualification_state',state);supplier.validate_qualification_state(state,owner.descriptor,context)
        gate('exact_selected_success_counts_before_write',all(work[key]==expected for key,expected in science['expected_success_counts'].items() if key not in ('owning_terminal_arrays_write','final_JSON_report_write')))
        gate('four_fixed_original_solver_scopes',len(storage.phases)==4 and all(row['completed'] and row['restored64'] for row in storage.phases))
        report['completed']=True
    except BaseException as error:report['failure']=dict(type=type(error).__name__,message=str(error));report['unreturned_interiors']='Unknown; actual aliases and owning returned copies retained'
    finally:
        sys.setprofile(old_profile);signal.setitimer(signal.ITIMER_REAL,0)
        if storage is not None:
            report['active_storage']=dict(peak_charged_bytes=storage.peak,observations=storage.observations,solver_phases=storage.phases)
        try:
            if torch is not None:
                try:
                    attempt('owning_terminal_arrays_write')
                    try:
                        if storage is not None:storage.charge('before_terminal_serialization',supplier.storage_bytes(saved))
                        guard()
                    except BaseException as error:
                        report['serialization_guard_failure']=dict(type=type(error).__name__,message=str(error));report['completed']=False
                        if report['failure'] is None:report['failure']=report['serialization_guard_failure']
                    # Failure evidence writes only already registered storage; no math/clone.
                    with raw_path.open('xb') as stream:torch.save(saved,stream);stream.flush();os.fsync(stream.fileno())
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
        report['passed']=report['completed'] and report['failure'] is None and all(row['passed'] for row in report['gates']);report['actual_real_native_admission']=report['passed']
        count('final_JSON_report_write');report['resources']=peaks()
        if storage is not None:report['active_storage']=dict(peak_charged_bytes=storage.peak,observations=storage.observations,solver_phases=storage.phases)
        try:
            with report_path.open('x') as stream:json.dump(observed(report),stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
            guard()
        except BaseException as error:
            report.update(passed=False,completed=False,actual_real_native_admission=False);report['final_serialization_error']=dict(type=type(error).__name__,message=str(error))
            if report['failure'] is None:report['failure']=report['final_serialization_error']
            report['resources']=peaks()
            if storage is not None:report['active_storage']=dict(peak_charged_bytes=storage.peak,observations=storage.observations,solver_phases=storage.phases)
            with report_path.open('w') as stream:json.dump(observed(report),stream,indent=2,allow_nan=False);stream.write('\n')
        signal.signal(signal.SIGALRM,old_handler);signal.setitimer(signal.ITIMER_REAL,*old_timer)
    if not report['passed']:raise RuntimeError('Terminal FR FIRST nonadmission: '+str(report_path))
    return str(report_path)
