"""FU CPU-only descriptive action of three immutable FT quotient graphs.

Spectrum uses the symmetric part; signal contractions use the original saved A.
Raw teacher targets stay immutable. No fits, source replay or causal conclusion.
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

KIND='Cora35_stored_three_arm_quotient_graph_action_zero_fit_diagnostic_v1'
SCIENCE_SHA='a9824586bb9d12c9d487f3ee8c8fa71302a355d577714551684651a39a4804d6'


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def refs(value,out=None):
    out={} if out is None else out
    if isinstance(value,dict):
        if set(value)=={'path','sha256'}:
            p,h=value['path'],value['sha256'];require(Path(p).is_absolute() and str(Path(p).resolve())==p and len(h)==64,'Normalized ABS ref')
            require(p not in out or out[p]==h,'Conflicting ref');out[p]=h
        for child in value.values():refs(child,out)
    elif isinstance(value,(list,tuple)):
        for child in value:refs(child,out)
    return out


def observed(value):
    if isinstance(value,dict):return {str(k):observed(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [observed(v) for v in value]
    if type(value) is float and not math.isfinite(value):return dict(nonfinite=repr(value))
    return value


def run(protocol_path,protocol_sha256,budget='cora35',stop=lambda:False):
    require(Path(protocol_path).is_absolute() and str(Path(protocol_path).resolve())==protocol_path,'Normalized ABS protocol path')
    started=time.monotonic();require(sha(protocol_path)==protocol_sha256,'Frozen protocol SHA')
    packet=json.loads(Path(protocol_path).read_text());sr=packet['scientific_contract']
    require(sr['sha256']==SCIENCE_SHA and sha(sr['path'])==SCIENCE_SHA,'Selected SCI bytes')
    science=json.loads(Path(sr['path']).read_text());policy=science['resource_protocol']
    require(packet['kind']==science['kind']==KIND and budget==science['budget']=='cora35','Selected kind/budget')
    folder=Path(packet['output_folder']);require(folder.is_absolute() and str(folder.resolve())==packet['output_folder'],'Normalized ABS output folder');folder.mkdir(parents=True,exist_ok=False)
    raw_path=folder/science['output_contract']['arrays'];report_path=folder/science['output_contract']['report']
    work={key:0 for key in science['expected_success_counts']};attempts={};saved={};live={};storage=None;torch=None
    report=dict(schema=1,kind=KIND,budget=budget,scientific_contract=sr,protocol=dict(path=str(Path(protocol_path).resolve()),sha256=protocol_sha256),
        source=packet['source'],FT_invocation_source=science['FT_invocation_source'],FS_invocation_source=science['FS_invocation_source'],owning_source=science['owning_source'],
        passed=False,completed=False,failure=None,gates=[],counts=work,attempts=attempts,metrics={},scientific_efficacy_admission=False,SOTA_admitted=False,all15_open=True,goal_complete=False,
        returned_value_policy='Actual CPU NumPy buffers enter detached Torch-view live/saved trees before guards; successful owns clone with overlap charged. Failure serializes registered aliases without new math/copy.',
        diagnostic_scope=science['mathematical_protocol']['inference'])
    raw=dict(path=str(raw_path),exists=False,bytes=None,sha256=None,hash_unknown=True,write_completed=False)
    def count(key,amount=1):work[key]+=amount
    def attempt(key):attempts[key]=attempts.get(key,0)+1
    def peaks():return dict(seconds=time.monotonic()-started,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,peak_CUDA_allocated_bytes=0,peak_CUDA_reserved_bytes=0)
    def guard():
        p=peaks();require(not stop(),'Stopped');require(p['seconds']<=policy['seconds_max'] and p['peak_RSS_bytes']<=policy['peak_RSS_bytes_max'],'Time/RSS cap')
        require(torch is None or not torch.cuda.is_initialized(),'CPU-only CUDA must remain uninitialized')
    def gate(name,ok,detail=None):
        report['gates'].append(dict(name=name,passed=bool(ok),detail=observed(detail)));require(ok,name);guard()
    def read(reference):require(sha(reference['path'])==reference['sha256'],'JSON ref SHA');return json.loads(Path(reference['path']).read_text())
    def pins():
        p=packet['readonly_files_sha256'];require(all(p.get(path)==h for path,h in refs(dict(packet=packet,science=science)).items()),'Full packet and SCI refs pinned')
        require(all(p.get(str(Path(packet['repository'])/path))==h for path,h in packet['source']['files'].items()),'Full current source map pinned')
        require(all(sha(path)==h for path,h in p.items()),'Readonly bytes');return dict(files=len(p),files_sha256=dict(p),passed=True)
    old_handler=signal.getsignal(signal.SIGALRM);old_timer=signal.getitimer(signal.ITIMER_REAL)
    def timeout(signum,frame):raise TimeoutError('FU frozen300s bound')
    signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,policy['seconds_max'])
    try:
        report['readonly_entry']=pins();bridge=read(packet['prerequisite_admission']);freeze=read(packet['ROOT_science_freeze'])
        gate('exact_static_current_bridge',bridge['passed'] is True and bridge['source']==packet['source'] and bridge['scientific_contract']==sr and all(bridge[k]==packet[k] for k in ('entrypoint','helper_entrypoint')))
        gate('ROOT_selected_before_metrics',freeze['passed'] is True and freeze['kind']=='ROOT_FU_Cora35_stored_graph_action_BEFORE_CODE_AND_METRICS_v1' and freeze['scientific_contract']==sr and freeze['negative_closure']==science['FT_negative_closure'])
        compat=science['source_compatibility'];old=science['source_before']['files'];current=packet['source']['files']
        gate('current188_separate_historical187_186_185',science['source_before']==science['FT_invocation_source'] and len(current)==compat['currentfiles']
            and set(current)==set(old)|{compat['new_file']} and all(current[k]==h for k,h in old.items() if k not in compat['allowed_diff']))
        prior=read(science['FT_actual_report']);root=read(science['FT_ROOT']);peer=read(science['FT_independent']);closure=read(science['FT_negative_closure'])
        gate('actual_FT_execution_and_closed_negative',all(v['passed'] is True for v in (prior,root,peer,closure)) and closure['negative_closed'] is True and closure['scientific_preset_passed'] is False
            and all(v['preset_passed'] is False for v in (prior,root,peer)) and root['actual_report']==peer['actual_report']==closure['actual_report']==science['FT_actual_report']
            and root['actual_arrays']==peer['actual_arrays']==closure['actual_arrays']==science['FT_actual_arrays'] and closure['ROOT']==science['FT_ROOT'] and closure['independent']==science['FT_independent']
            and all(v['source']==science['FT_invocation_source'] and v['source_owner']==science['source_owner'] and v['context']==science['context'] and v['accepted_state_metadata_seal']==science['FS_state_metadata_seal'] for v in (prior,root,peer,closure))
            and prior['serving_arm_metadata']==peer['serving_arm_metadata']==science['serving_arm_metadata'] and prior['raw_evidence']['sha256']==science['FT_actual_arrays']['sha256'])
        import torch as torch_module
        import numpy as np
        torch=torch_module
        from src import quotient_composed_ce_native_source as supplier
        from src.research_loop import implementation_provenance
        gate('exact_source_entry_helpers',all(str(Path(module.__file__).resolve())==packet[key]['path'] and sha(module.__file__)==packet[key]['sha256'] for module,key in ((sys.modules[__name__],'entrypoint'),(supplier,'helper_entrypoint')))
            and packet['helper_entrypoint']==science['source_metadata_helper'])
        report['source_entry']=implementation_provenance();gate('full_current_source_entry',report['source_entry']==packet['source'])
        torch.set_num_threads(4)
        if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
        report['runtime']=dict(device='cpu',torch_threads=torch.get_num_threads(),torch_interop_threads=torch.get_num_interop_threads(),**{key:os.environ.get(key) for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')})
        gate('selected_CPU_runtime',report['runtime']==science['runtime'] and not torch.cuda.is_initialized())
        class NumPyBudget(supplier.ActiveStorageBudget):
            def charge(self,role,future=0):return super().charge(role,future+policy['NumPy_transient_reserve_bytes'])
        storage=NumPyBudget(saved,live,guard,policy)
        attempt('FT_terminal_PT_loads');storage.charge('before_FT_mmap');payload=torch.load(science['FT_actual_arrays']['path'],mmap=True,map_location='cpu',weights_only=False)
        live['FT_terminal']=payload;saved['FT_terminal']=supplier.detached(payload);count('FT_terminal_PT_loads');storage.charge('after_FT_mmap')
        report['mapped_input']=dict(path=science['FT_actual_arrays']['path'],file_bytes=Path(science['FT_actual_arrays']['path']).stat().st_size,unique_tensor_storage_bytes=supplier.storage_bytes(payload),mmap=True,whole_clone=False,virtual_file_length_is_not_resident_exemption=True)
        gate('exact_terminal_keys_context',set(payload)=={'serving_arms','source_owner','context','FS_state_metadata_seal'} and payload['source_owner']==science['source_owner'] and payload['context']==science['context'] and payload['FS_state_metadata_seal']==science['FS_state_metadata_seal'])
        report.update(source_owner=payload['source_owner'],context=payload['context'],FS_state_metadata_seal=payload['FS_state_metadata_seal'])
        supplier.validate_context(payload['context']);gate('three_saved_arms',set(payload['serving_arms'])==set(science['arms']))
        before=supplier.metadata(payload);live['numpy_actual_returns']={}
        def views(value):
            if isinstance(value,np.ndarray):return torch.from_numpy(value)
            if isinstance(value,dict):return {k:views(v) for k,v in value.items()}
            if isinstance(value,(list,tuple)):return [views(v) for v in value]
            return value
        def compute(key,fn,future=0,counter=None):
            storage.charge('before_numpy_'+key,future);value=fn()
            view=views(value);live['numpy_actual_returns'][key]=view;saved[key]=supplier.detached(view)
            if counter is not None:count(counter)
            storage.own(key,view);return value
        def entropy(value):
            logs=np.zeros_like(value);positive=value>0;logs[positive]=np.log(value[positive]);return -(value*logs).sum(axis=1)
        def ratio(numerator,denominator):
            require(math.isfinite(numerator) and math.isfinite(denominator),'Finite ratio scalars')
            return dict(value=numerator/denominator if denominator!=0 else None,status='defined' if denominator!=0 else 'zero_reference',numerator=numerator,denominator=denominator)
        for name in science['arms']:
            arm=payload['serving_arms'][name];attempt('serving_arm_metadata_checks')
            actual=supplier.metadata(arm);count('serving_arm_metadata_checks');gate(name+'_literal_metadata',actual==science['serving_arm_metadata'][name] and all(t.device.type=='cpu' and not t.requires_grad and bool(torch.isfinite(t).all()) for t in supplier.tensors(arm)))
            arrays=compute(name+'_FP64_inputs',lambda:{k:np.array(v.numpy(),dtype=np.float64,order='C',copy=True) for k,v in arm.items()},sum(v.numel()*8 for v in arm.values()))
            X,Q,A,P,n=(arrays[k] for k in ('X','Q','Sc','P','n'));m={};report['metrics'][name]=m
            geom=compute(name+'_P_geometry',lambda:dict(rowmass=P.sum(1),nmass=P.sum(0),rowentropy=entropy(P),rowmax=P.max(1)),(P.shape[0]*3+len(n))*8,counter='P_geometry_metrics')
            gate(name+'_assignment',bool((P>=0).all()) and float(np.max(np.abs(geom['rowmass']-1)))<=1e-9 and float(np.max(np.abs(geom['nmass']-n)))<=1e-9 and bool((n>0).all()))
            m['assignment_geometry']=dict(mean_row_entropy=float(geom['rowentropy'].mean()),mean_row_max=float(geom['rowmax'].mean()),mass_min=float(n.min()),mass_mean=float(n.mean()),mass_max=float(n.max()),mass_CV=float(n.std()/n.mean()),largest_mass_fraction=float(n.max()/n.sum()))
            qmass=compute(name+'_Q_rowmass',lambda:Q.sum(1),len(n)*8);gate(name+'_positive_Q_A',bool((Q>=0).all()) and bool((qmass>0).all()) and bool((A>=0).all()) and float(A.sum())>0)
            T=compute(name+'_diagnostic_T',lambda:Q/qmass[:,None],Q.nbytes,counter='Q_row_normalizations_diagnostic_only')
            teacher=compute(name+'_teacher_geometry',lambda:dict(purity=T.max(1),entropy=entropy(T)),len(n)*16)
            m['teacher_geometry']=dict(raw_Q_rowmass_min=float(qmass.min()),raw_Q_rowmass_mean=float(qmass.mean()),raw_Q_rowmass_max=float(qmass.max()),uniform_purity=float(teacher['purity'].mean()),mass_weighted_purity=float(n@teacher['purity']/n.sum()),uniform_entropy=float(teacher['entropy'].mean()),mass_weighted_entropy=float(n@teacher['entropy']/n.sum()))
            sym=compute(name+'_symmetric_diagnostic',lambda:(A+A.T)/2,A.nbytes)
            eigenvalues,eigenvectors=compute(name+'_eigh',lambda:np.linalg.eigh(sym),A.nbytes+len(n)*8,counter='Sc_eigendecompositions')
            u=compute(name+'_principal_u',lambda:np.array(eigenvectors[:,-1]*(-1 if eigenvectors[:,-1].sum()<0 else 1),copy=True),len(n)*8)
            gap=float(eigenvalues[-1]-eigenvalues[-2]);gate(name+'_eigensystem',bool(np.isfinite(eigenvalues).all()) and bool(np.isfinite(eigenvectors).all()) and float(eigenvalues[-1])>0 and gap>1e-10 and float(u.sum())>0)
            m['spectrum']=dict(eigenvalues=eigenvalues.tolist(),top=float(eigenvalues[-1]),nonprincipal_max_abs=float(np.max(np.abs(eigenvalues[:-1]))),principal_gap=gap,trace=float(np.trace(sym)),offdiagonal_weight_fraction=float((A.sum()-np.trace(A))/A.sum()),symmetry_Linf=float(np.max(np.abs(A-A.T))))
            m['contractions']={}
            for signal_name,Y in (('X',X),('T',T)):
                Yc=compute(name+'_'+signal_name+'_centered',lambda:Y-u[:,None]*(u@Y)[None,:],Y.nbytes);den=float(np.linalg.norm(Yc));acted=Yc
                for power in (1,2):
                    acted=compute(name+'_'+signal_name+'_A'+str(power),lambda:A@acted,Y.nbytes)
                    Z=compute(name+'_'+signal_name+'_BcA'+str(power),lambda:acted-u[:,None]*(u@acted)[None,:],Y.nbytes,counter='stationary_centered_signal_contractions')
                    m['contractions'][signal_name+str(power)]=ratio(float(np.linalg.norm(Z)),den);gate(name+'_'+signal_name+str(power)+'_finite',bool(np.isfinite(Z).all()))
            E=compute(name+'_teacher_agreement',lambda:T.T@(A/A.sum())@T,Q.shape[1]**2*8,counter='soft_teacher_agreement_matrices')
            a,b=compute(name+'_agreement_marginals',lambda:(E.sum(1),E.sum(0)),Q.shape[1]*16)
            h=float(np.trace(E));baseline=float(a@b);excess=h-baseline;den=1-baseline
            m['soft_teacher_agreement']=dict(E=E.tolist(),row_marginal=a.tolist(),column_marginal=b.tolist(),agreement=h,baseline=baseline,excess=excess,normalized_excess=ratio(excess,den))
            gate(name+'_finite_metrics',all(math.isfinite(v) for v in (h,baseline,excess)) and bool(np.isfinite(T).all()));storage.charge('after_arm_'+name)
        gate('immutable_FT_inputs',supplier.metadata(payload)==before)
        gate('exact_21_counts_before_writes',work==dict(science['expected_success_counts'],owning_terminal_arrays_write=0,final_JSON_report_write=0));report['completed']=True
    except BaseException as error:report['failure']=dict(type=type(error).__name__,message=str(error));report['completed']=False
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        try:
            if torch is not None:
                attempt('owning_terminal_arrays_write')
                try:
                    if storage is not None:storage.charge('before_terminal_serialization',supplier.storage_bytes(saved))
                    guard()
                except BaseException as error:
                    report['serialization_guard_failure']=dict(type=type(error).__name__,message=str(error));report['completed']=False
                    if report['failure'] is None:report['failure']=report['serialization_guard_failure']
                try:
                    with raw_path.open('xb') as stream:torch.save(saved,stream);stream.flush();os.fsync(stream.fileno())
                    raw['write_completed']=True;count('owning_terminal_arrays_write')
                finally:
                    raw['exists']=raw_path.exists()
                    if raw['exists']:raw['bytes']=raw_path.stat().st_size
                raw['sha256']=sha(raw_path);raw['hash_unknown']=False
                if storage is not None:storage.charge('after_terminal_serialization')
        except BaseException as error:
            report['raw_serialization_error']=dict(type=type(error).__name__,message=str(error));report['completed']=False
            if report['failure'] is None:report['failure']=report['raw_serialization_error']
        report['raw_evidence']=raw
        try:
            report['readonly_exit']=pins()
            if torch is not None:report['source_exit']=implementation_provenance();require(report['source_exit']==packet['source'],'Full current source exit')
            guard();require(work['owning_terminal_arrays_write']==1,'Owning arrays write')
        except BaseException as error:
            report['exit_guard_error']=dict(type=type(error).__name__,message=str(error));report['completed']=False
            if report['failure'] is None:report['failure']=report['exit_guard_error']
        count('final_JSON_report_write');report['resources']=peaks();report['passed']=report['completed'] and report['failure'] is None and work==science['expected_success_counts'] and all(row['passed'] for row in report['gates'])
        if not report['passed']:report['completed']=False
        if storage is not None:report['active_storage']=dict(peak_charged_bytes=storage.peak,observations=storage.observations,NumPy_transient_reserve_bytes=policy['NumPy_transient_reserve_bytes'])
        try:
            with report_path.open('x') as stream:json.dump(observed(report),stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
            guard()
        except BaseException as error:
            report.update(passed=False,completed=False);report['final_serialization_error']=dict(type=type(error).__name__,message=str(error))
            if report['failure'] is None:report['failure']=report['final_serialization_error']
            report['resources']=peaks()
            with report_path.open('w') as stream:json.dump(observed(report),stream,indent=2,allow_nan=False);stream.write('\n')
        signal.signal(signal.SIGALRM,old_handler);signal.setitimer(signal.ITIMER_REAL,*old_timer)
    if not report['passed']:raise RuntimeError('Terminal FU diagnostic failure: '+str(report_path))
    return str(report_path)
