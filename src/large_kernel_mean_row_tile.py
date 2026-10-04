"""UNEXECUTED native complete-critic/update integration gate; ROOT binding required."""
import argparse
import hashlib
import json
import math
import os
import platform
import resource
import signal
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

N, B, C, K, R, CHUNK = 169343, 512, 40, 90, 16, 2048
ABS, REL, SECONDS = 5e-7, 2e-5, 300
STAGE = Path(__file__).resolve().parents[1] / "results/research_loop/large_kernel_mean_mmap_row_tile_backend_stageEF_v1"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def tagged(value):
    if isinstance(value,float) and not math.isfinite(value):
        return dict(nonfinite_observation=repr(value))
    if isinstance(value,dict):
        return {str(k):tagged(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [tagged(v) for v in value]
    return value


def main(binding_path, binding_sha256, external_stop=lambda: False):
    started = time.monotonic()
    torch = None
    rows = provider = None
    errors, counts, metrics, evidence, raw = [], {}, {}, {}, {}
    observed = dict(schema=1, passed=False, goal_complete=False, efficacy_qualified=False,
        test_enabled=False, admission='ONE native update and noninitial complete-cotangent arithmetic only')

    def memory():
        out = dict(process_peak_RSS_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024)
        if torch is not None and torch.cuda.is_initialized():
            out.update(GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
                GPU_peak_reserved_bytes=torch.cuda.max_memory_reserved(0))
        return out

    def stop():
        if torch is not None and torch.cuda.is_initialized():
            torch.cuda.synchronize(0)
        m = memory()
        require(m['process_peak_RSS_bytes'] <= 16*1024**3, 'RSS16GiB exceeded')
        require(m.get('GPU_peak_allocated_bytes',0) <= 4*1024**3
            and m.get('GPU_peak_reserved_bytes',0) <= 6*1024**3, 'GPU4/6GiB exceeded')
        return external_stop() or time.monotonic()-started > SECONDS

    def guard():
        if stop():
            raise TimeoutError('Whole native integration exceeded300seconds')

    def sha(path):
        h = hashlib.sha256()
        with Path(path).open('rb') as f:
            while True:
                guard()
                block = f.read(8*1024**2)
                if not block:
                    break
                h.update(block)
                counts['source_hash_bytes'] = counts.get('source_hash_bytes',0)+len(block)
        return h.hexdigest()

    def pin(ref):
        require(sha(ref['path']) == ref['sha256'], 'Pinned bytes changed: '+ref['path'])

    require(sha(binding_path) == binding_sha256, 'Root binding changed')
    binding = json.loads(Path(binding_path).read_text())
    require(binding['schema'] == 1 and binding['root_FIRST_execution'] is True, 'Root FIRST binding required')
    for name in ('script','spec','source_manifest','ledger'):
        pin(binding[name])
    spec = json.loads(Path(binding['spec']['path']).read_text())
    source = json.loads(Path(binding['source_manifest']['path']).read_text())
    repo = Path(binding['repo'])
    require(source == spec['current_source'] == binding['source'], 'Actual current source mismatch')
    require(Path(binding['script']['path']).resolve() == Path(__file__).resolve(), 'Wrong qualifier script')
    require(spec['fixed'] == dict(nodes=N,basis=B,classes=C,cells=K,rank=R,chunk=CHUNK,
        mixing=.05,steps=1,diagnostic_noninitial_step=1), 'Frozen native geometry changed')
    outputs = {k:Path(v) for k,v in binding['outputs'].items()}
    require(set(outputs) == {'result','arrays','trajectory'} and len(set(outputs.values())) == 3
        and all(p.resolve().parent == STAGE and not p.exists() for p in outputs.values()), 'Exclusive outputs required')
    for ref in spec['references'].values():
        pin(ref)
    for p,h in spec['original_assets_sha256'].items():
        pin(dict(path=p,sha256=h))
    for p,h in source['files'].items():
        pin(dict(path=str(repo/p),sha256=h))
    require(subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip() == source['git_head'], 'Current Git changed')
    require(all(os.environ.get(k) == v for k,v in spec['environment'].items()), 'Native environment changed')
    for name in ('CPU_integration_result','CPU_integration_root_acceptance'):
        require(json.loads(Path(spec['references'][name]['path']).read_text())['passed'] is True, 'New complete-critic integration prerequisite missing')
    import numpy as np
    import torch as imported_torch
    torch = imported_torch
    require(binding['runtime'] == spec['runtime'] == dict(Python=platform.python_version(),
        NumPy=np.__version__,Torch=str(torch.__version__),CUDA=torch.version.cuda), 'Independently frozen runtime changed')
    torch.set_num_threads(4)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    require(torch.get_default_dtype() == torch.float32 and not torch.is_autocast_enabled('cuda'), 'Precision changed')
    require(torch.cuda.get_device_name(0) == spec['GPU'], 'GPU changed')
    device = torch.device('cuda:0'); torch.cuda.reset_peak_memory_stats(0)
    from src import kernel_mean_ce as core
    from src.kernel_mean_row_tile import FrozenPhiRows, TiledKernelMeanMoments, bounded_outer_gradient
    from src.kernel_mean_row_tile_execution import RowTileExecution, EXECUTION_MODE, BOUNDS
    from src.io import array_digest
    from src.moments import decode_moments, augmented
    from src.shared_features import _tensor_identity
    from src.soft_ce_partition import solve_head_system
    from src.sweep_utils import representative

    def descriptor(x):
        if torch.is_tensor(x):
            if x.layout == torch.sparse_csr:
                return dict(shape=list(x.shape),dtype=str(x.dtype),layout=str(x.layout),
                    parts=[descriptor(t) for t in (x.crow_indices(),x.col_indices(),x.values())])
            return _tensor_identity(x)
        if isinstance(x,dict):
            return {str(k):descriptor(v) for k,v in sorted(x.items())}
        if isinstance(x,(list,tuple)):
            return [descriptor(v) for v in x]
        return x

    def gate(name, actual, reference):
        a,b = actual.detach().cpu().double(),reference.detach().cpu().double()
        require(a.shape == b.shape and bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all()), name+' domain')
        error,scale = float((a-b).abs().max()),float(b.abs().max())
        relative = 0. if error == scale == 0 else (math.inf if scale == 0 else error/scale)
        metrics[name] = dict(absolute_Linf=error,relative_Linf=relative,reference_Linf=scale,
            passed=error <= ABS and relative <= REL)
        require(metrics[name]['passed'], name+' BOTH precision limits failed')

    def alarm(signum, frame):
        raise TimeoutError('Whole invocation hard alarm; CG interiors unobserved on interruption')

    old_handler = signal.signal(signal.SIGALRM,alarm)
    signal.setitimer(signal.ITIMER_REAL,max(1,270-(time.monotonic()-started)))
    primary = None
    try:
        counts['original_BW_payload_load_attempts'] = 1
        packet = torch.load(spec['references']['origin']['path'],map_location='cpu',weights_only=True)
        counts['original_BW_payload_load_completed'] = 1
        require(set(packet) == {'schema','context','arrays','descriptors','content_sha256'}, 'Origin packet schema changed')
        arrays = packet['arrays']; before = descriptor(arrays)
        capture = json.loads(Path(spec['references']['BW_capture']['path']).read_text())
        require(packet['context'] == capture['origin_context'] and packet['descriptors'] == before
            and core._seal(dict(context=packet['context'],descriptors=before)) == packet['content_sha256'], 'BW source packet seal changed')
        u0,v0,z0,q0,a0,m_phys0 = (arrays[k] for k in ('U0','V0','z','Q','hard','M0'))
        counts['new_native_M0_payload_load_attempts'] = 1
        prior = torch.load(spec['references']['new_native_arithmetic_arrays']['path'],map_location='cpu',weights_only=True)
        m_kernel0 = prior['GPU_repeat0']['M'].detach().clone(); del prior
        counts['new_native_M0_payload_load_completed'] = 1
        require(tuple(m_kernel0.shape) == (K,1+B+C), 'Accepted new mean-Phi M0 shape changed')
        z,q,assignment,u,v = (x.detach().to(device).clone() for x in (z0,q0,a0,u0,v0))
        side = json.loads(Path(spec['references']['Phi_sidecar']['path']).read_text())
        rows = FrozenPhiRows.from_npy(spec['references']['Phi']['path'],expected_shape=(N,B),
            expected_content_digest=side['phi_digest'],expected_file_sha256=spec['references']['Phi']['sha256'],
            source_refs=dict(original_BW_origin=spec['references']['origin'],Phi_sidecar=spec['references']['Phi_sidecar']),
            validation_chunk=CHUNK,evidence=evidence.setdefault('Phi_admission',{}),stop=stop)
        options = dict(capture['original_config']); options.pop('data_digest',None)
        options['save_resume'] = True
        data_digest = array_digest(z0.numpy(),q0.numpy(),a0.numpy())
        require(data_digest == capture['original_config']['data_digest'], 'Original z/Q/hard data identity changed')
        assets = dict(H=capture['origin_context']['native_source_buffers']['H'],
            transform=before['transform'],anchors=capture['origin_context']['native_source_buffers']['map']['anchors'],
            mapping=capture['origin_context']['native_source_buffers']['map']['mapping'],Phi_identity=side,
            z=_tensor_identity(z),Q=_tensor_identity(q),assignment=_tensor_identity(assignment))
        def origin(m,d):
            # Accepted immutable M0 is reused; only the native readout quotient
            # is decoded on its owning arithmetic device, as the core does.
            x,y,mass = decode_moments(m.to(device),d)
            return dict(moments=_tensor_identity(m),centers=_tensor_identity(x),labels=_tensor_identity(y))
        refs = dict(nodes=N,cells=K,rank=R,dimension=128,classes=C,basis=B,chunk_size=CHUNK,
            factor_seed=0,mixing=.05,original_options=core._native_options(options),
            device=str(device),runtime=core._runtime(device),data_digest=data_digest,
            asset_paths=spec['asset_paths'],files_sha256=spec['original_assets_sha256'],
            current_source=source,optimizer_source_sha256=source['files']['src/kernel_mean_ce.py'])
        refs['execution_backend'] = dict(schema=1,mode=EXECUTION_MODE,
            provider_source=spec['references']['provider'],operator_source=spec['references']['operator'],
            rows=rows.descriptor(),arithmetic_acceptance=spec['references']['native_arithmetic_acceptance'],bounds=BOUNDS)
        context = dict(schema=core.SCHEMA,mode=core.MODE,policy=core.POLICY,source_refs=refs,
            native_parameter_digests=core._factor_digests([u,v],N,K,R),asset_descriptors=assets,
            native_origin=origin(m_phys0,128),kernel_origin=origin(m_kernel0,B))
        provider = RowTileExecution(rows,context,stop=stop)
        state = core.optimize(z,q,assignment,[u,v],rows,outputs['trajectory'],1,options,context,
            checkpoint_steps=(0,1),execution=provider,stop=stop)
        raw['actual_core_state'] = state
        core.validate_core_resume(state,state['config'],context,outputs['trajectory'])
        require(state['step'] == 1 and state['work']['P_updates'] == state['work']['Adam_steps'] == 1,
            'Exactly ONE actual update required')
        require(state['work']['head_interfaces'] == state['work']['moment_forward_calls']
            == state['work']['physical_moment_forward_calls'] == 2, 'Exactly2 own critic/head/physical evaluations required')
        require(state['work']['adjoint_solves'] == state['work']['moment_backward_calls'] == 1, 'Exactly1 original update pullback required')
        require(state['CE0'] > 0 and state['snapshots'][0]['CE'] == state['CE0']
            and state['current']['CE0'] == state['CE0'], 'Own immutable positive CE0 missing')
        require(not torch.equal(state['parameters'][0],u0) and bool((state['parameters'][0] != 0).any()), 'U did not update')
        require(torch.equal(state['parameters'][1],v0), 'U0zero must leave first V unchanged')
        # New candidate physical moment readout; reuse the frozen RMS buffers.
        # The reference is the accepted original readout, never a replayed fit.
        transform = SimpleNamespace(**{k:x.to(device) if torch.is_tensor(x) else x
            for k,x in arrays['transform'].items()})
        counts['new_candidate_initial_physical_readout_attempts'] = 1
        x_initial,y_initial,mass_initial = representative(state['snapshots'][0]['physical_moments'],transform,128,device)
        actual_readout = [x_initial,y_initial,torch.full_like(mass_initial,1/K)]
        raw['new_candidate_initial_physical_readout'] = [x.detach().cpu() for x in actual_readout]
        counts['new_candidate_initial_physical_readout_completed'] = 1
        require(all(_tensor_identity(x.detach().cpu()) == before['readout'][i]
            for i,x in enumerate(actual_readout)), 'Initial physical FP32 X/Q/uniform readout differs from original')
        uu,vv = (x.to(device).clone().requires_grad_() for x in state['parameters'])
        diag = evidence.setdefault('noninitial_endpoint1_diagnostic',{})
        dm = TiledKernelMeanMoments.apply(uu,vv,assignment,rows,q,.05,CHUNK,diag,stop)
        diagnostic_raw = raw.setdefault('noninitial_diagnostic',{})
        diagnostic_raw.update(U=uu.detach().cpu(),V=vv.detach().cpu(),M=dm.detach().cpu())
        require(_tensor_identity(dm.detach().cpu()) == _tensor_identity(state['current']['moments']), 'Endpoint1 diagnostic moment bytes changed')
        theta = state['current']['theta'].to(device)
        ce,rhs = bounded_outer_gradient(rows,q,theta,CHUNK,diag,stop)
        diagnostic_raw.update(theta=theta.detach().cpu(),rhs=rhs.detach().cpu(),CE=ce)
        require(ce == state['current']['CE'], 'Endpoint1 fixed theta/raw CE bytes changed')
        features,labels,mass = decode_moments(dm.detach(),B)
        counts['endpoint1_diagnostic_adjoint_attempts'] = 1
        vector,adj = solve_head_system(augmented(features),labels,torch.full_like(mass,1/K),theta,
            options['penalty'],rhs,rtol=options['cg_rtol'],max_iter=options['cg_max_iter'],
            initial=state['current']['vector'].to(device),cg_check_interval=options.get('cg_check_interval',1))
        counts['endpoint1_diagnostic_adjoint_completed'] = 1
        diagnostic_raw.update(vector=vector.detach().cpu(),adjoint=adj)
        require(adj['cg_converged'] is True, 'Endpoint1 diagnostic adjoint failed')
        guard(); cot = core.complete_moment_cotangent(dm.detach(),B,theta,vector,options['penalty'],state['CE0'])
        diagnostic_raw['G'] = cot.detach().cpu()
        dm.backward(cot)
        diagnostic_raw.update(grad_U=uu.grad.detach().cpu(),grad_V=vv.grad.detach().cpu())
        guard()
        require(bool((uu.grad != 0).any()) and bool((vv.grad != 0).any()), 'Noninitial U/V pullback must both be nonzero')
        # NEW nonzero-U endpoint reference; reuse immutable input Phi, never replay an old proof.
        phi_cpu = np.load(spec['references']['Phi']['path'],mmap_mode='r',allow_pickle=False)
        uc,vc,gc = (raw['noninitial_diagnostic'][k] for k in ('U','V','G'))
        ref_u = torch.empty(N,R,dtype=torch.float64); ref_v = torch.zeros(K,R,dtype=torch.float64)
        completed = 0; returned_u = 0; returned_v = 0
        counts['NEW_noninitial_independent_CPU_reference_attempts'] = 1
        try:
            for first in range(0,N,CHUNK):
                guard(); last=min(first+CHUNK,N)
                phi = torch.from_numpy(np.array(phi_cpu[first:last],copy=True,order='C'))
                material = torch.cat((phi.new_ones(last-first,1),phi,q0[first:last]),1)
                prior_logits = torch.full((last-first,K),float(np.log(.05/K)),dtype=torch.float32)
                prior_logits.scatter_(1,a0[first:last,None],float(np.log(.95+.05/K)))
                probability = (prior_logits+uc[first:last]@vc.T/math.sqrt(R)).double().softmax(1)
                gp = material@gc.T/N
                block = (probability*(gp-(probability*gp).sum(1,keepdim=True))).float()/math.sqrt(R)
                ref_u[first:last] = block.double()@vc.double(); returned_u=last; guard()
                ref_v += block.double().T@uc[first:last].double(); returned_v=last; guard()
                completed=last; counts['NEW_noninitial_independent_CPU_reference_tiles_completed'] = counts.get('NEW_noninitial_independent_CPU_reference_tiles_completed',0)+1
            counts['NEW_noninitial_independent_CPU_reference_completed'] = 1
        finally:
            raw['NEW_noninitial_independent_CPU_reference'] = dict(grad_U_prefix=ref_u[:returned_u].clone(),
                grad_V=ref_v.clone(),completed_row_end=completed,returned_U_row_end=returned_u,
                returned_V_row_end=returned_v,complete=completed == N)
        gate('noninitial_native_grad_U_vs_CPU',uu.grad,ref_u)
        gate('noninitial_native_grad_V_vs_CPU',vv.grad,ref_v)
        require(descriptor(arrays) == before, 'Original source packet mutated')
        observed.update(passed=True,own_CE0=state['CE0'],endpoint1_CE=state['current']['CE'],
            objective1=state['current']['objective'],core_work=state['work'],core_attempts=state['attempts'],
            head_work=[state['snapshots'][i]['head_work'] for i in (0,1)],
            actual_endpoint1_noninitial_factor_gradients_qualified=True,
            initial_physical_readout_preserved=True,context=context)
    except BaseException as error:
        primary = error
        observed.update(passed=False,error_type=type(error).__name__,error=str(error))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0); signal.signal(signal.SIGALRM,old_handler)
        try:
            if provider is not None and not observed['passed']:
                raw['incomplete_provider_evidence'] = provider.failure_evidence()
        except BaseException as error:
            errors.append(dict(stage='partial_provider_recovery',error=repr(error)))
        try:
            if not observed['passed'] and outputs['trajectory'].exists():
                failed = outputs['trajectory']/'failure.json'
                if failed.exists():
                    observed['actual_core_failure'] = json.loads(failed.read_text())
                prefix = outputs['trajectory']/'resume.pt'
                if prefix.exists():
                    # Preserve the already-written accepted prefix, no math replay.
                    raw['last_accepted_core_prefix'] = torch.load(prefix,map_location='cpu',weights_only=False)
        except BaseException as error:
            errors.append(dict(stage='accepted_prefix_failure_recovery',error=repr(error)))
        raw['operator_partials'] = {k:v.get('partial_arrays',[]) for k,v in evidence.items() if v.get('partial_arrays')}
        observed['raw_arrays'] = dict(path=str(outputs['arrays']),exists=False,bytes=0,
            complete_write=False,sha256=None,checksum_status='not_attempted')
        try:
            with outputs['arrays'].open('xb') as f:
                torch.save(raw,f);f.flush();os.fsync(f.fileno())
            observed['raw_arrays'].update(exists=True,bytes=outputs['arrays'].stat().st_size,complete_write=True)
            observed['raw_arrays'].update(sha256=sha(outputs['arrays']),checksum_status='complete')
        except BaseException as error:
            errors.append(dict(stage='raw_preservation',error=repr(error)))
            observed['raw_arrays'].update(exists=outputs['arrays'].exists(),
                bytes=outputs['arrays'].stat().st_size if outputs['arrays'].exists() else 0,
                checksum_status='unavailable_after_write_or_guard_error',preservation_error=repr(error))
        try:
            if rows is not None:
                rows.assert_immutable(full=True,evidence=evidence.setdefault('final_Phi',{}),stop=stop)
            for ref in spec['references'].values():pin(ref)
            for p,h in source['files'].items():pin(dict(path=str(repo/p),sha256=h))
            for p,h in spec['original_assets_sha256'].items():pin(dict(path=p,sha256=h))
            for name in ('script','spec','source_manifest','ledger'):pin(binding[name])
            require(subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip() == source['git_head'], 'Final Git changed')
            guard(); observed['full_source_exit_unchanged'] = True
        except BaseException as error:
            errors.append(dict(stage='full_source_exit',error=repr(error)))
        observed.update(passed=observed['passed'] and not errors,finalization_errors=errors,
            elapsed_seconds=time.monotonic()-started,memory=memory(),metrics=metrics,counts=counts,
            operation_evidence={k:{a:b for a,b in v.items() if a != 'partial_arrays'} for k,v in evidence.items()},
            source=source,binding=dict(path=str(binding_path),sha256=binding_sha256),
            excluded_work=dict(students=0,test_evaluation=0,new_teachers=0,new_kernels=0,new_RMS=0,
                new_initializers=0,old_math_or_origin_replays=0),
            output_trajectory=str(outputs['trajectory']))
        with outputs['result'].open('x') as f:
            json.dump(tagged(observed),f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
        print(json.dumps(dict(passed=observed['passed'],result=str(outputs['result']),
            elapsed_seconds=observed['elapsed_seconds'],memory=observed['memory'])))
    if primary is not None:
        raise primary
    require(observed['passed'],'Native FIRST complete-critic gate failed; preserve evidence')
    return dict(passed=True, report=str(outputs['result']), elapsed_seconds=observed['elapsed_seconds'],
        memory=observed['memory'], work=observed['core_work'], trajectory=str(outputs['trajectory']))


def continue_fixed_prefix(binding_path, binding_sha256, stop=lambda: False):
    """Resume an immutable accepted prefix in its original owning namespace.

    This is input reuse and new endpoint computation. It never rewrites an old
    checkpoint, context or teacher, and never runs students or selects by test.
    """
    import torch
    from src import kernel_mean_ce as core
    from src.kernel_mean_row_tile import FrozenPhiRows
    from src.kernel_mean_row_tile_execution import RowTileExecution
    started = time.monotonic()
    def check():
        if stop() or time.monotonic()-started > 300:
            raise InterruptedError('Bounded fixed continuation stopped; retain accepted prefix')
    def sha(path):
        h=hashlib.sha256()
        with Path(path).open('rb') as stream:
            while block:=stream.read(8*1024**2):
                check();h.update(block)
        return h.hexdigest()
    require(sha(binding_path)==binding_sha256,'Continuation binding changed')
    binding=json.loads(Path(binding_path).read_text())
    require(binding['schema']==1 and binding['accepted_native_oneupdate'] is True
        and type(binding['steps']) is int and binding['steps']==25,
        'Only a frozen accepted 1-to25 continuation is admitted')
    for path,expected in binding['immutable_files_sha256'].items():
        require(sha(path)==expected,'Immutable continuation input changed: '+path)
    result_path=Path(binding['result'])
    require(not result_path.exists(),'Exclusive fixed continuation result required')
    state=torch.load(binding['accepted_prefix']['path'],map_location='cpu',weights_only=False)
    require(sha(binding['accepted_prefix']['path'])==binding['accepted_prefix']['sha256']
        and state['step']==1,'Immutable accepted native1P prefix changed')
    torch.set_num_threads(4)
    if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    context=state['context'];refs=context['source_refs'];device=torch.device(refs['device'])
    require(core._runtime(device)==refs['runtime'],'Continuation native arithmetic changed')
    if device.type=='cuda':
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(device)
    core.validate_core_resume(state,state['config'],context,binding['trajectory'])
    packet=torch.load(binding['origin']['path'],map_location='cpu',weights_only=True)
    require(sha(binding['origin']['path'])==binding['origin']['sha256'],'Original frozen BW packet changed')
    z,q,assignment=(packet['arrays'][k].detach().to(device).clone() for k in ('z','Q','hard'))
    ev={};row_descriptor=refs['execution_backend']['rows']
    rows=FrozenPhiRows.from_npy(refs['asset_paths']['Phi'],expected_shape=tuple(row_descriptor['shape']),
        expected_content_digest=row_descriptor['content_digest'],expected_file_sha256=row_descriptor['file']['sha256'],
        source_refs=row_descriptor['source_refs'],validation_chunk=refs['chunk_size'],evidence=ev,stop=stop)
    require(rows.descriptor()==row_descriptor,'Original row source changed at continuation')
    execution=RowTileExecution(rows,context,stop=stop)
    options=dict(refs['original_options'],save_resume=True)
    initial=[p.detach().to(device).clone() for p in state['initial_parameters']]
    require(core._factor_digests(initial,refs['nodes'],refs['cells'],refs['rank'],device=device)
        ==context['native_parameter_digests'],'Owning native initial copies changed exact bytes')
    old_handler=signal.signal(signal.SIGALRM,lambda signum,frame:(_ for _ in ()).throw(TimeoutError('Continuation hard alarm')))
    signal.setitimer(signal.ITIMER_REAL,max(1,270-(time.monotonic()-started)))
    report=dict(schema=1,passed=False,test_enabled=False,student_fits=0,condensations=1,
        prior_P_updates=1,new_P_updates_planned=24,fixed_steps=25,goal_complete=False)
    try:
        final=core.optimize(z,q,assignment,initial,rows,binding['trajectory'],25,
            options,context,checkpoint_steps=(0,1,25),resume_state=state,execution=execution,stop=stop)
        core.validate_core_resume(final,final['config'],context,binding['trajectory'])
        require(final['step']==25 and final['work']['P_updates']==25,'Exact fixed25 endpoint missing')
        for path,expected in binding['immutable_files_sha256'].items():
            require(sha(path)==expected,'Accepted immutable prefix changed: '+path)
        report.update(passed=True,actual_P_updates=final['work']['P_updates'],new_P_updates=24,
            work=final['work'],attempts=final['attempts'],CE0=final['CE0'],endpoint_CE=final['current']['CE'],
            endpoint_objective=final['current']['objective'],elapsed_seconds=time.monotonic()-started,
            resume_sha256=sha(Path(binding['trajectory'])/'resume.pt'))
    except BaseException as error:
        report.update(error_type=type(error).__name__,error=str(error),elapsed_seconds=time.monotonic()-started)
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old_handler)
        with result_path.open('x') as f:json.dump(tagged(report),f,indent=2,allow_nan=False)
    return report


def run(operation, binding_path, binding_sha256, stop=lambda: False):
    """Terminal bounded jobs; incomplete qualification never auto-retries."""
    try:
        if operation=='qualify':return main(binding_path,binding_sha256,external_stop=stop)
        if operation=='condense25':return continue_fixed_prefix(binding_path,binding_sha256,stop)
        raise ValueError('Unknown fixed kernel-mean row-tile operation')
    except InterruptedError as error:
        raise RuntimeError('Terminal incomplete row-tile job; preserve evidence and use a new reviewed job ID') from error


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--binding',required=True)
    parser.add_argument('--binding-sha256',required=True);args=parser.parse_args()
    main(args.binding,args.binding_sha256)
