"""FT selected Cora35 three-arm independent GCN and unpropagated ROW-X MLP validation.

Historical source185 owns the origin; pure admitted source186 owns fixed25 state.
The current source187 invocation is separately pinned. Original student core is literal.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import time
import sys

KIND='Cora35_original_quotient_fixed25_three_arm_paired_GCN_RAW_X_MLP_validation_v1'
SCIENCE_SHA='2603293425e2a565313ecc43f8829ab26c8bfcd1b68d641dd1562ce2ccc3477e'

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


def prepare_arms(torch, supplier, storage, source, state25, count, gate):
    from src.low_rank_assignment import LowRankLogits
    from src.coarsening import feature_centroids, quotient_adjacency
    dev=torch.device('cuda:0')
    inputs={};storage.live['serving_source']=inputs
    for key,value,dtype in [('X',source.X,torch.float64),('Q',source.Q,torch.float64),('S',source.S,torch.float64),('hard',source.hard,torch.int64)]:
        storage.charge('before_serving_convert_'+key,2*supplier.copy_bytes(value))
        inputs[key]=value.to(dev,dtype=dtype);storage.charge('after_serving_convert_'+key)
    X,Q,S,hard=(inputs[key] for key in ('X','Q','S','hard'))
    native=source.native
    gate('controls_share_original_native_initialization',
         all(torch.equal(a,b) for a,b in zip(source.cache['owning_native_source']['parameters'],(native['u'],native['v'])))
         and torch.equal(source.cache['owning_native_source']['hard'],source.hard))
    pairs=dict(P0=(native['u'],native['v']),singleComposed25=source.cache['own_E25_parameters'],
               quotient25=(state25['native_parameters']['U'],state25['native_parameters']['V']))
    arms={};storage.live['serving_arms']=arms
    for name,(U,V) in pairs.items():
        storage.charge('before_serving_parameters',supplier.copy_bytes((U,V)))
        U,V=U.to(dev),V.to(dev);storage.live['serving_parameters']=dict(U=U,V=V)
        storage.charge('after_serving_parameters')
        logits=LowRankLogits.apply(U,V,hard,.05,4096);count('serving_native_forward')
        storage.own('serving_current_logits',logits)
        P=logits.double().softmax(1);storage.own('serving_current_P',P)
        Xc,n=feature_centroids(P,X);count('original_feature_centroids_calls')
        storage.own('serving_current_Xc_n',(Xc,n))
        Qc,nq=feature_centroids(P,Q);count('original_feature_centroids_calls')
        storage.own('serving_current_Qc_n',(Qc,nq))
        Sc=quotient_adjacency(P,S,normalization='symmetric',mass_scaling=False,self_loops='retain',chunk_size=35)
        count('original_quotient_adjacency_calls');storage.own('serving_current_Sc',Sc)
        gate(name+'_positive_finite',torch.equal(n,nq) and bool((n>0).all())
             and bool(torch.isfinite(Xc).all()) and bool(torch.isfinite(Qc).all()) and bool(torch.isfinite(Sc).all()))
        if name=='quotient25':
            gate('native_quotient25_P_exact',supplier.identity(P)==supplier.identity(state25['current_endpoint']['P']))
        if name=='P0':gate('native_P0_exact',supplier.identity(P)==state25['context']['new_P0_identity'])
        storage.charge('before_student_FP32_casts',supplier.copy_bytes((Xc,Qc,Sc))+35*8)
        arms[name]=dict(X=Xc.float(),Q=Qc.float(),Sc=Sc.float(),P=P.detach(),n=n.detach(),uniform_weights=torch.full((35,),1/35,dtype=torch.float64,device=dev))
        storage.own('serving_arm_'+name,arms[name])
    graph={};masks={};storage.live['validation_inputs']=dict(graph=graph,masks=masks)
    for key,value in source.cache['graph'].items():
        storage.charge('before_validation_graph_'+key,supplier.copy_bytes(value));graph[key]=value.to(dev);storage.charge('after_validation_graph_'+key)
    for key,value in source.cache['masks'].items():
        storage.charge('before_validation_mask_'+key,supplier.copy_bytes(value));masks[key]=value.to(dev);storage.charge('after_validation_mask_'+key)
    storage.charge('before_validation_rawQ',supplier.copy_bytes(source.Q));q=source.Q.to(dev);storage.live['validation_inputs']['q']=q;storage.charge('after_validation_rawQ')
    return arms,graph,masks,q


def fit_arms(torch, supplier, storage, arms, graph, masks, q, folder, settings, seeds, count, attempt, gate, guard, report):
    import csv,json,math,hashlib
    from pathlib import Path
    from src.evaluation import fit_gcn_diagnostic, _forward, _input_digest
    from src.io import _fingerprint
    from src.models import GCN
    rows=[];report['students']=rows;report['student_resource_observations']=[]
    for architecture in ('GCN','RAW_X_MLP'):
        validation_graph=dict(graph,adj=graph['adj'] if architecture=='GCN' else None)
        for name,arm in arms.items():
            for seed in seeds:
                destination=Path(folder)/architecture/name
                cpu_before=supplier.storage_bytes((storage.saved,storage.live),True)
                attempt('student_fits')
                result=fit_gcn_diagnostic(arm['X'],arm['Q'],arm['uniform_weights'],
                    validation_graph,q,masks,seed=seed,folder=destination,
                    training_adjacency=arm['Sc'] if architecture=='GCN' else None,
                    stop=lambda:(guard() or False),**settings)
                count('student_fits');count('GCN_student_fits' if architecture=='GCN' else 'RAW_X_MLP_student_fits')
                row=dict(architecture=architecture,arm=name,seed=seed,result=result);rows.append(row)
                cpu_after=supplier.storage_bytes((storage.saved,storage.live),True)
                upper=max(cpu_before,cpu_after)+torch.cuda.max_memory_allocated(0)+storage.policy['ordinary_reserve_bytes']+storage.policy['hash_scratch_bytes']
                report['student_resource_observations'].append(dict(architecture=architecture,arm=name,seed=seed,
                    external_CPU_owned_before=cpu_before,external_CPU_owned_after=cpu_after,cumulative_CUDA_peak=torch.cuda.max_memory_allocated(0),
                    original_fitter_CPU_opaque_reserve=storage.policy['ordinary_reserve_bytes'],observed_active_upper=upper))
                gate('student_observed_storage_'+architecture+'_'+name+'_'+str(seed),upper<=storage.policy['active_source_material_copy_storage_bytes_max'])
                gate('finite_fit_'+architecture+'_'+name+'_'+str(seed),all(math.isfinite(v) for v in result.values() if type(v) is float))
                storage.charge('student_return_'+architecture+'_'+name+'_'+str(seed))
                path=destination/f'seed_{seed}_epochs.csv'
                with path.open() as stream: history=list(csv.DictReader(stream))
                count('student_epoch_rows',len(history))
                best=max(float(x['val_acc']) for x in history);first=next(x for x in history if float(x['val_acc'])==best)
                gate('first_max_'+architecture+'_'+name+'_'+str(seed),[int(x['epoch']) for x in history]==list(range(1,601)) and int(first['epoch'])==result['epoch'] and all(math.isfinite(float(x['val_acc'])) for x in history))
                ckpt=destination/f'seed_{seed}_selected.pt'
                storage.charge('before_selected_checkpoint_load',4*ckpt.stat().st_size)
                attempt('selected_checkpoint_PT_loads');selected=torch.load(ckpt,map_location='cpu',weights_only=False);count('selected_checkpoint_PT_loads')
                storage.live['selected_checkpoint']=selected
                fitted=json.loads((destination/f'seed_{seed}.json').read_text())
                input_digest=_input_digest(arm['X'],arm['Q'],arm['uniform_weights'],validation_graph,q,masks,arm['Sc'] if architecture=='GCN' else None,lambda:(guard() or False))
                count('selected_recipe_fingerprint_checks')
                gate('selected_fingerprint_'+architecture+'_'+name+'_'+str(seed),selected['epoch']==result['epoch']
                     and selected['fingerprint']==fitted['fingerprint']==_fingerprint(fitted['recipe']) and fitted['result']==result
                     and fitted['recipe']['input_digest']==input_digest and fitted['recipe']['test_enabled'] is False)
                row['artifacts']={}
                for label,artifact in [('history',path),('selected',ckpt),('result',destination/f'seed_{seed}.json')]:
                    with artifact.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
                    row['artifacts'][label]=dict(path=str(artifact.resolve()),bytes=artifact.stat().st_size,sha256=digest)
                storage.charge('before_replay_model',4*supplier.copy_bytes(selected['model_state']))
                model=GCN(arm['X'].shape[1],256,arm['Q'].shape[1],2,0.0).to(arm['X'].device)
                model.load_state_dict(selected['model_state']);model.eval();storage.live['replay_model']=model.state_dict()
                with torch.no_grad():logprob=_forward(model,validation_graph['x'],validation_graph['adj'])
                count('selected_validation_replays');storage.own('selected_replay_logits',logprob)
                gate('finite_replay_'+architecture+'_'+name+'_'+str(seed),bool(torch.isfinite(logprob).all()))
                correct=int((logprob[masks['val']].argmax(1)==graph['y'][masks['val']]).sum())
                val_nodes=int(masks['val'].sum());acc=100*float((logprob[masks['val']].argmax(1)==graph['y'][masks['val']]).double().mean())
                row.update(val_correct=correct,val_nodes=val_nodes,replayed_val_acc=acc)
                gate('selected_actual_accuracy_'+architecture+'_'+name+'_'+str(seed),val_nodes==500 and acc==result['val_acc'])
                row.update(history_path=str(path),selected_checkpoint_path=str(ckpt),result_path=str(destination/f'seed_{seed}.json'))
                del model,selected,logprob;storage.live.pop('selected_checkpoint',None);storage.live.pop('replay_model',None)
                guard()
    comparisons=[]
    for architecture in ('GCN','RAW_X_MLP'):
        lookup={(x['arm'],x['seed']):x['val_correct'] for x in rows if x['architecture']==architecture}
        for control in ('P0','singleComposed25'):
            differences=[lookup['quotient25',seed]-lookup[control,seed] for seed in seeds]
            passed=(sum(differences)>=9 and sum(x>0 for x in differences)>=2) if architecture=='GCN' else sum(differences)>=0
            comparisons.append(dict(architecture=architecture,control=control,paired_correct_differences=differences,
                sum_gain=sum(differences),strict_wins=sum(x>0 for x in differences),preset_passed=passed))
    report['comparisons']=comparisons;report['preset_passed']=all(x['preset_passed'] for x in comparisons)
    report['cohort_means']=[dict(architecture=a,arm=b,val_acc_mean=sum(x['replayed_val_acc'] for x in rows if x['architecture']==a and x['arm']==b)/3)
        for a in ('GCN','RAW_X_MLP') for b in arms]
    return rows



def run(protocol_path,protocol_sha256,budget='cora35',stop=lambda:False):
    started=time.monotonic();path=Path(protocol_path)
    require(path.is_absolute() and str(path.resolve())==str(path) and sha(path)==protocol_sha256,'Protocol bytes')
    packet=json.loads(path.read_text());sr=packet['scientific_contract']
    require(sr['sha256']==SCIENCE_SHA and sha(sr['path'])==SCIENCE_SHA,'Exact selected FT science')
    science=json.loads(Path(sr['path']).read_text());C=science['case']
    require(packet['kind']==science['kind']==KIND and budget==science['budget']=='cora35','Selected FT validation only')
    folder=Path(packet['output_folder']);require(folder.is_absolute() and str(folder.resolve())==str(folder),'ABS output');folder.mkdir(parents=True,exist_ok=False)
    raw_path=folder/science['output_contract']['arrays'];report_path=folder/science['output_contract']['report']
    work={key:0 for key in science['expected_success_counts']};attempts={};saved={};live={};torch=None;storage=None
    report=dict(schema=1,kind=KIND,budget=budget,source=packet['source'],owning_source=science['owning_source'],FS_invocation_source=science['FS_invocation_source'],scientific_contract=sr,
        protocol=dict(path=str(path),sha256=protocol_sha256),passed=False,completed=False,failure=None,gates=[],counts=work,attempts=attempts,
        returned_value_policy='Actual detached aliases enter live and failure serialization trees before guards; successful owning CPU copies charge both existing aliases and copies. GPU raw aliases remain honest on failed precharge.',
        original_core=science['unexecuted_prescribed_core'],settings=science['settings'],student_seeds=science['student_seeds'],condensation_updates=0,
        actual_student_efficacy=False,SOTA_admitted=False,all15_open=True,goal_complete=False)
    raw_evidence=dict(path=str(raw_path),exists=False,bytes=None,sha256=None,hash_unknown=True,write_completed=False)
    old_handler,old_timer=signal.getsignal(signal.SIGALRM),signal.getitimer(signal.ITIMER_REAL)
    def attempt(key):attempts[key]=attempts.get(key,0)+1
    def count(key,amount=1):work[key]+=amount
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
    def alarm(signum,frame):raise TimeoutError('FT300s selected validation deadline')
    try:
        signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,max(1e-6,300-(time.monotonic()-started)))
        report['readonly_entry']=pins();bridge=read(packet['prerequisite_admission'])
        gate('current_source_static_bridge',bridge['passed'] is True and bridge['source']==packet['source'] and bridge['scientific_contract']==sr
            and all(bridge[key]==packet[key] for key in ('entrypoint','helper_entrypoint','source_owner_entrypoint')))
        freeze=read(packet['ROOT_science_freeze']);gate('ROOT_selected_science',freeze['passed'] is True and freeze['kind']=='ROOT_FT_Cora35_paired_GCN_RAW_X_MLP_selection_BEFORE_CODE_v1' and freeze['scientific_contract']==sr)
        prior25=read(science['FS_actual_report']);combined=read(science['FS_actual_admission']);root=read(science['FS_actual_ROOT']);peer=read(science['FS_actual_independent'])
        gate('actual_FS_ROOT_and_independent',all(row['passed'] is True for row in (prior25,combined,root,peer))
            and combined['actual_report']==root['actual_report']==peer['actual_report']==science['FS_actual_report']
            and combined['actual_arrays']==root['actual_arrays']==peer['actual_arrays']==science['FS_actual_arrays']
            and prior25['source']==prior25['source_entry']==prior25['source_exit']==combined['source']==root['source']==peer['source']==science['FS_invocation_source']
            and all(row['scientific_contract']==science['FS_scientific_contract'] and row['context']==science['owning_context'] and row['config']==science['FS_state_config']
                and row['state_metadata_seal']==science['FS_state_metadata_seal'] for row in (prior25,combined,root,peer))
            and prior25['source_owner']==root['source_owner']==peer['source_owner']==science['owning_source_owner'] and prior25['owning_source']==science['owning_source'])
        FR=read(science['FIRST_scientific_contract']);gate('unchanged_original_source_options',C==FR['case'] and science['original_29_options']==FR['original_29_options'] and science['runtime']==FR['runtime'])
        compat=science['invocation_source_compatibility'];old_files=compat['old']['files'];current=packet['source']['files']
        gate('historical186_and_current187_separate',compat['old']==science['FS_invocation_source']==science['source_before'] and len(current)==187
            and set(current)==set(old_files)|{compat['new_file']} and compat['new_file']=='src/quotient_composed_ce_native_student_validation.py'
            and compat['allowed_diff']==['src/research_loop.py'] and all(current[key]==h for key,h in old_files.items() if key not in compat['allowed_diff']))
        import torch as torch_module
        torch=torch_module
        from src import quotient_composed_ce_native as public
        from src import quotient_composed_ce_native_source as supplier
        from src import quotient_composed_ce_native_trajectory as trajectory
        from src import evaluation,joint_mean_ce as joint
        from src.research_loop import implementation_provenance
        gate('exact_entrypoint_helpers',all(sha(module.__file__)==packet[key]['sha256'] and str(Path(module.__file__).resolve())==packet[key]['path']
            for module,key in ((sys.modules[__name__],'entrypoint'),(public,'helper_entrypoint'),(supplier,'source_owner_entrypoint')))
            and sha(trajectory.__file__)==science['pure_FS_state_validator']['sha256'] and sha(evaluation.__file__)==science['original_evaluator_entrypoint']['sha256'])
        report['source_entry']=implementation_provenance();gate('full_current_source_entry',report['source_entry']==packet['source'])
        torch.set_num_threads(4)
        if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(False);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
        torch.cuda.init();device=torch.device('cuda:0');report['runtime']=joint._runtime(device)
        gate('original_native_runtime',report['runtime']==science['runtime'] and torch.get_num_interop_threads()==1 and not torch.is_autocast_enabled('cuda')
            and all(os.environ.get(key)=='4' for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')))
        storage=supplier.ActiveStorageBudget(saved,live,guard,science['resource_protocol'])
        def load(reference,key,counter):
            attempt(counter);storage.charge('before_mmap_'+key);value=torch.load(reference['path'],mmap=True,map_location='cpu',weights_only=False)
            saved[key]=supplier.detached(value);live[key]=value;count(counter);count('own_PT_supplier_loads');storage.charge('after_mmap_'+key)
            report.setdefault('mapped_payloads',{})[key]=dict(path=reference['path'],file_bytes=Path(reference['path']).stat().st_size,unique_tensor_storage_bytes=supplier.storage_bytes(value),mmap=True,whole_clone=False,virtual_file_bytes_not_RSS=True)
            return value
        accepted=load(science['FS_actual_arrays'],'accepted_FS_mapped','accepted_trajectory_PT_loads');state=accepted['trajectory_state']
        saved['accepted_trajectory_state']=supplier.detached(state);live['accepted_trajectory_state']=state
        attempt('accepted_trajectory_state_validations');trajectory.validate_trajectory_state(state,science['owning_source_owner'],science['owning_context'],science['FS_invocation_source'],science['FS_state_config']);count('accepted_trajectory_state_validations')
        state_before=supplier.metadata(state);report['accepted_state_metadata_seal']=state['state_metadata_seal']
        gate('pure_actual_FS25_state',state['state_metadata_seal']==science['FS_state_metadata_seal'] and state['context']==science['owning_context']);storage.charge('after_pure_FS_state_validation')
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
        gate('exact_source185_owner_reconstructed',owner.descriptor==science['owning_source_owner']==state['source_owner']);context=state['context']

        masks0=owner.cache['masks'];graph0=owner.cache['graph'];n=C['dimensions']['N']
        gate('train_val_only_before_any_student',set(masks0)=={'train','val'} and all(mask.dtype==torch.bool and tuple(mask.shape)==(n,) for mask in masks0.values())
            and int(masks0['val'].sum())==500 and not bool((masks0['train']&masks0['val']).any())
            and bool((graph0['y'][~(masks0['train']|masks0['val'])]==-1).all()))
        returned=prepare_arms(torch,supplier,storage,owner,state,count,gate)
        live['prepared_return']=returned;saved['prepared_return']=supplier.detached(returned);storage.charge('prepared_return_owned_aliases')
        arms,graph,masks,q=returned
        report['serving_arm_metadata']={name:supplier.metadata(saved['serving_arm_'+name]) for name in arms}
        report['source_owner']=owner.descriptor;report['context']=context
        fit_arms(torch,supplier,storage,arms,graph,masks,q,folder,science['settings'],science['student_seeds'],count,attempt,gate,guard,report)
        gate('accepted_state_and_supplier_unchanged',supplier.metadata(state)==state_before and owner.descriptor==science['owning_source_owner']
            and supplier.cache_ids(historical)==W['ALL5_identities'] and joint._seal(supplier.metadata({key:historical[key] for key in ('owning_physical_endpoints','owning_native_source','own_E25_parameters','owning_raw_H_decode')}))==W['private_owning_metadata']['canonical_JSON_seal'])
        expected_before=dict(science['expected_success_counts'],owning_terminal_arrays_write=0,final_JSON_report_write=0)
        gate('exact_selected_28_counts_before_terminal_writes',work==expected_before)
        report.update(completed=True,actual_student_efficacy=False,validation_scope='Only selected18fit screen; preset_passed is separate from execution passed.')
    except BaseException as error:
        report['failure']=dict(type=type(error).__name__,message=str(error));report['completed']=False
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        try:
            if torch is not None:
                terminal=dict(serving_arms={name:saved['serving_arm_'+name] for name in science['scope']['arms']},source_owner=report['source_owner'],context=report['context'],FS_state_metadata_seal=report['accepted_state_metadata_seal']) if report['completed'] else {key:value for key,value in saved.items() if key not in ('FA153_mapped','FD159_mapped','accepted_FS_mapped')}
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
        report['passed']=report['completed'] and report['failure'] is None and work==science['expected_success_counts'] and all(row['passed'] for row in report['gates'])
        if not report['passed']:report['completed']=False
        if storage is not None:report['active_storage']=dict(peak_charged_bytes=storage.peak,observations=storage.observations,solver_phases=storage.phases)
        try:
            with report_path.open('x') as stream:json.dump(observed(report),stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
            guard()
        except BaseException as error:
            report.update(passed=False,completed=False);report['final_serialization_error']=dict(type=type(error).__name__,message=str(error))
            if report['failure'] is None:report['failure']=report['final_serialization_error']
            report['resources']=peaks()
            if storage is not None:report['active_storage']=dict(peak_charged_bytes=storage.peak,observations=storage.observations,solver_phases=storage.phases)
            with report_path.open('w') as stream:json.dump(observed(report),stream,indent=2,allow_nan=False);stream.write('\n')
        signal.signal(signal.SIGALRM,old_handler);signal.setitimer(signal.ITIMER_REAL,*old_timer)
    if not report['passed']:raise RuntimeError('Terminal FT validation execution failure: '+str(report_path))
    return str(report_path)
