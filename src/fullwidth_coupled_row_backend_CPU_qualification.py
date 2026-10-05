"""Unexecuted FC fresh literal full-width file-backed row algebra qualification."""
import hashlib
import json
import math
import resource
import signal
import sys
import time
from decimal import Decimal, Context, localcontext, ROUND_HALF_EVEN
from pathlib import Path

KIND = "fullwidth_coupled_row_backend_literal_CPU_qualification_stageFC_v1"
SCIENCE_SHA = "9e706f997f1a2e65bfc530a102269274f8449fdc536c8f4079fd86ad7173e216"


def require(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4194304), b""): value.update(block)
    return value.hexdigest()


def refs(value, found=None):
    found = {} if found is None else found
    if isinstance(value, dict):
        if set(value) == {"path", "sha256"}:
            p, h = value["path"], value["sha256"]
            require(type(p) is str and Path(p).is_absolute() and str(Path(p).resolve()) == p, "Ref path")
            require(type(h) is str and len(h) == 64 and all(c in "0123456789abcdef" for c in h), "Ref digest")
            require(p not in found or found[p] == h, "Conflicting ref"); found[p] = h
        for child in value.values(): refs(child, found)
    elif isinstance(value, list):
        for child in value: refs(child, found)
    return found


def run(protocol_path, protocol_sha256, stop=lambda: False):
    started = time.monotonic()
    require(str(Path(protocol_path).resolve()) == protocol_path and Path(protocol_path).is_absolute()
        and sha(protocol_path) == protocol_sha256, "Protocol path/SHA")
    packet = json.loads(Path(protocol_path).read_text()); sr = packet["scientific_contract"]
    require(sr["sha256"] == SCIENCE_SHA and sha(sr["path"]) == SCIENCE_SHA, "Science SHA")
    science = json.loads(Path(sr["path"]).read_text())
    require(packet["kind"] == science["kind"] == KIND, "Kind differs")
    out = Path(packet["output_folder"]); require(out.is_absolute() and str(out.resolve()) == str(out), "Output ABS")
    out.mkdir(parents=True, exist_ok=False)
    arrays, report_path = out / "qualification_arrays.pt", out / "qualification_report.json"
    report = dict(schema=1, kind=KIND, scientific_contract=sr, source=packet["source"],
        protocol=dict(path=protocol_path, sha256=protocol_sha256), passed=False, completed=False, failure=None,
        gates=[], counts={k:0 for k in science["counts"]},
        raw_evidence=dict(path=str(arrays), exists=False, bytes=None, sha256=None, hash_unknown=True,
            hash_error=None, write_completed=False), head_summaries=[], FD_summaries=[])
    report["attempts"] = {key:0 for key in ("stationary_heads","adjoints","generic_moment_gradient_calls","component_writes","block_witness_writes","smooth_forwards","smooth_factor_adjoints","native_forwards","native_factor_adjoints","source_CE_RHS")}
    report["own_component_files"] = {}; report["block_witness_files"] = []
    report["arithmetic_work"] = dict(dense_Hessians=0, Newton_steps=0, Armijo_trials=0, Decimal_scalar_objective_evaluations=0)
    report["scalar_Armijo_arithmetic"] = dict(precision=80, rounding="ROUND_HALF_EVEN", all_heads=True, fallback=False, precision_search=False)
    saved, torch = {}, None
    old_handler, old_timer = signal.getsignal(signal.SIGALRM), signal.getitimer(signal.ITIMER_REAL)

    def peaks():
        return dict(seconds=time.monotonic()-started, peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            peak_allocated_bytes=0, peak_reserved_bytes=0)

    def check(): require(not stop() and all(v <= science["resources"][k] for k,v in peaks().items()), "Frozen resource/stop")
    def alarm(signum, frame): raise TimeoutError("Frozen300s CPU boundary")
    def gate(name, ok, **detail):
        report["gates"].append(dict(name=name, passed=bool(ok), **detail)); require(ok, name)
    def report_json():
        nonfinite=[]
        def safe(value,path="report"):
            if type(value) is float and not math.isfinite(value):
                nonfinite.append(dict(path=path,scalar_tag=repr(value)));return None
            if isinstance(value,dict): return {key:safe(child,path+"."+str(key)) for key,child in value.items()}
            if isinstance(value,list): return [safe(child,path+"["+str(i)+"]") for i,child in enumerate(value)]
            return value
        value=safe(report)
        if nonfinite: value["nonfinite_scalar_diagnostics"]=nonfinite
        return json.dumps(value,indent=2,allow_nan=False)+"\n"
    def pin_values():
        expected = packet["readonly_files_sha256"]
        require(all(Path(p).is_absolute() and str(Path(p).resolve()) == p for p in expected), "Readonly paths")
        require(all(expected.get(p) == h for p,h in refs(dict(packet=packet, science=science)).items()), "Missing ref pin")
        result = {}
        for p in expected: check(); result[p] = sha(p)
        return result

    try:
        signal.signal(signal.SIGALRM, alarm); signal.setitimer(signal.ITIMER_REAL, max(1e-6,300-(time.monotonic()-started)))
        report["readonly_entry"] = pin_values(); gate("entry_pins", report["readonly_entry"] == packet["readonly_files_sha256"])
        gate("own_entrypoint", Path(packet["entrypoint"]["path"]).resolve() == Path(__file__).resolve())
        freeze=json.loads(Path(packet["prerequisite_admission"]["path"]).read_text())
        gate("ROOT_current_static_prerequisites",freeze["passed"] is True and freeze["scientific_contract"]==sr and freeze["source"]==packet["source"] and freeze["entrypoint"]==packet["entrypoint"] and freeze["helper_entrypoint"]==packet["helper_entrypoint"])
        import torch
        from src import fullwidth_coupled_row_backend as backend
        from src.nystrom_ce import moment_gradient
        from src.research_loop import implementation_provenance
        import numpy as np
        import os
        report['source_entry']=implementation_provenance(); gate('source_entry',report['source_entry']==packet['source'])
        gate('helper_source',str(Path(backend.__file__).resolve())==packet['helper_entrypoint']['path'] and sha(backend.__file__)==packet['helper_entrypoint']['sha256'])
        torch.set_num_threads(4)
        if torch.get_num_interop_threads()!=1: torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False; torch.set_float32_matmul_precision('highest')
        runtime=dict(python=sys.version.split()[0],torch=torch.__version__,numpy=np.__version__,torch_threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),
            **{name:os.environ.get(name) for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')},
            device='cpu',default_dtype=str(torch.get_default_dtype()),AMP=torch.is_autocast_enabled(),TF32=torch.backends.cuda.matmul.allow_tf32,matmul_precision=torch.get_float32_matmul_precision())
        report['runtime']=runtime; gate('frozen_CPU_runtime',runtime==science['runtime'] and not torch.cuda.is_initialized())
        fixture=science['fixture']; n,d,b,c,k,r=(fixture[name] for name in ('N','D','B','C','K','rank')); F=d+b
        def tensor(value): return torch.tensor(value,dtype=torch.float64,device='cpu')
        def own(value):
            if torch.is_tensor(value): return value.detach().cpu().clone()
            if isinstance(value,dict): return {name:own(child) for name,child in value.items()}
            if isinstance(value,(list,tuple)): return [own(child) for child in value]
            return value
        z,q,phi,U,V,L=(tensor(fixture[name]) for name in ('z','Q','Phi','U_current','V','L_rows_B_by_D'))
        hard=torch.tensor(fixture['hard'],dtype=torch.long); U0=torch.zeros_like(U)
        mixing=fixture['mixing']; penalty=fixture['penalty_all_theta_entries_including_bias']
        S=torch.cat((torch.ones((n,1),dtype=torch.float64),z,q),1)
        source_X=torch.cat((z,phi,torch.ones((n,1),dtype=torch.float64)),1)
        saved['literal_inputs']=own(dict(z=z,Q=q,Phi=phi,U=U,V=V,U0=U0,L=L,hard=hard,material=S,source_X=source_X))
        gate('independent_general_Q_full_width',bool((q>0).all()) and bool((q.sum(1)-1).abs().max()>1e-2) and not torch.equal(phi,z@L.T))
        components={}
        for name,value in (('z',z),('Q',q),('Phi',phi)):
            path=out/(name+'.npy'); row=dict(path=str(path),exists=False,write_completed=False,bytes=None,sha256=None)
            report['own_component_files'][name]=row; report['attempts']['component_writes']+=1
            try:
                np.save(path,value.numpy(),allow_pickle=False); row['write_completed']=True; report['counts']['component_writes']+=1
            finally:
                row['exists']=path.exists()
                if row['exists']: row['bytes']=path.stat().st_size; row['sha256']=sha(path)
            arr=value.numpy(); h=hashlib.sha256(json.dumps(dict(shape=tuple(arr.shape),dtype=str(arr.dtype)),sort_keys=True).encode());h.update(memoryview(arr).cast('B'))
            row['content_digest']=h.hexdigest()
            components[name]=backend.ImmutableRowComponent(str(path),tuple(arr.shape),str(arr.dtype),h.hexdigest(),dict(kind='own_FC_literal_fixture',fixture=fixture['name']),n)
            report['counts']['component_constructions']+=1
        layout=backend.CoupledTileLayout(n,d,b,c,k,r)
        owners=[]; witness_index=0
        def witness(role,start,end,values):
            nonlocal witness_index
            witness_index+=1; report['attempts']['block_witness_writes']+=1
            path=out/('block_witness_%04d.pt'%witness_index)
            row=dict(role=role,start=start,end=end,path=str(path),exists=False,write_completed=False,bytes=None,sha256=None)
            report['block_witness_files'].append(row)
            frozen=own(values)  # One active block, written/discarded; never gather full P in backend.
            try:
                torch.save(dict(schema=1,kind='own_FC_bounded_block_witness',role=role,start=start,end=end,values=frozen),path)
                row['write_completed']=True;report['counts']['block_witness_writes']+=1
            finally:
                row['exists']=path.exists()
                if row['exists']:
                    row['bytes']=path.stat().st_size
                    try: row['sha256']=sha(path)
                    except BaseException as exc: row['hash_unknown']=True;row['hash_error']=str(exc);raise
            del frozen
        for partition in fixture['partitions']:
            owner=backend.CoupledOriginalSourceTiles(components['z'],components['Q'],components['Phi'],layout,partition,dict(fixture=fixture['name']))
            owner.witness_callback=witness; owners.append(owner);report['counts']['source_owner_constructions']+=1
        saved['source_descriptors']=[owner.descriptor() for owner in owners]
        gate('current_nonzero_factors',float(U.norm())>0 and float(V.norm())>0)
        def dense_probability(u,v):
            prior=torch.full((n,k),float(np.log(mixing/k)),dtype=u.dtype)
            prior.scatter_(1,hard[:,None],float(np.log(1-mixing+mixing/k)))
            return (prior+u@v.T/math.sqrt(r)).double().softmax(1)
        def dense_moments(u,v):
            report['counts']['dense_reference_moments']+=1
            return dense_probability(u,v).T@S/n
        def inner_features(cen): return torch.cat((cen,cen@L.T),1)
        def inner(x,t,theta): return -(t*(x@theta.T).log_softmax(1)).sum()/k+penalty*theta.square().sum()/2
        def decimal_inner(x_values,t_values,theta):
            report['arithmetic_work']['Decimal_scalar_objective_evaluations']+=1
            with localcontext(Context(prec=80,rounding=ROUND_HALF_EVEN)):
                th=[[Decimal.from_float(float(v)) for v in row] for row in theta.tolist()]; total=Decimal(0)
                for xi,ti in zip(x_values,t_values):
                    logits=[sum((a*b for a,b in zip(xi,tr)),Decimal(0)) for tr in th];shift=max(logits)
                    lse=shift+sum(((v-shift).exp() for v in logits),Decimal(0)).ln()
                    total+=sum((target*(lse-v) for target,v in zip(ti,logits)),Decimal(0))
                ridge=sum((v*v for row in th for v in row),Decimal(0))
                return total/Decimal(k)+Decimal.from_float(float(penalty))*ridge/Decimal(2)
        def dense_terms(x,t,theta):
            p=(x@theta.T).softmax(1);ts=t.sum(1)
            gradient=((ts[:,None]*p-t).T@x)/k+penalty*theta
            cov=(torch.diag_embed(p)-p[:,:,None]*p[:,None,:])*(ts/k)[:,None,None]
            H=torch.einsum('iab,ijl->ajbl',cov,x[:,:,None]*x[:,None,:]).reshape(c*(F+1),c*(F+1))
            H+=penalty*torch.eye(c*(F+1),dtype=torch.float64);report['arithmetic_work']['dense_Hessians']+=1
            return gradient,H
        heads=[]
        def head(mp,role):
            check();report['attempts']['stationary_heads']+=1
            mu=mp[:,0:1];cen=mp[:,1:1+d]/mu;t=mp[:,1+d:]/mu;theta=torch.zeros((c,F+1),dtype=torch.float64)
            row=own(dict(role=role,physical_M=mp,mass=mu,centers=cen,targets=t,theta=theta));heads.append(row);saved['heads']=heads
            summary=dict(role=role,completed=False,head_gradient_abs_max=None,Newton_steps=0,Armijo_trials=0,last_alpha=None)
            report['head_summaries'].append(summary)
            gate('positive_head_domain_'+role,bool((mu>0).all()) and bool((t>0).all()))
            x=torch.cat((inner_features(cen),torch.ones((k,1),dtype=torch.float64)),1);row['head_features_augmented']=own(x)
            xv=[[Decimal.from_float(float(v)) for v in ri] for ri in x.tolist()];tv=[[Decimal.from_float(float(v)) for v in ri] for ri in t.tolist()]
            iterations=trials=0
            for iteration in range(101):
                check();gradient,H=dense_terms(x,t,theta);norm=float(gradient.abs().max())
                row.update(own(dict(theta=theta,gradient=gradient,Hessian=H)));summary.update(head_gradient_abs_max=norm,Newton_steps=iterations,Armijo_trials=trials,last_theta_FP64=theta.tolist())
                require(math.isfinite(norm),'Nonfinite stationary gradient')
                if norm<=1e-12: break
                require(iteration<100,'Frozen independent Newton100 exhausted')
                direction=torch.linalg.solve(H,gradient.reshape(-1)).reshape_as(theta);descent=(gradient*direction).sum()
                row['last_Newton_direction']=own(direction)
                require(bool(torch.isfinite(direction).all()) and math.isfinite(float(descent)) and float(descent)>0,'Independent descent')
                value=decimal_inner(xv,tv,theta);alpha=1.;summary['last_Decimal_current']=str(value)
                for trial in range(60):
                    check();trials+=1;report['arithmetic_work']['Armijo_trials']+=1
                    candidate=theta-alpha*direction; candidate_value=decimal_inner(xv,tv,candidate)
                    with localcontext(Context(prec=80,rounding=ROUND_HALF_EVEN)):
                        rhs=value-Decimal.from_float(1e-4)*Decimal.from_float(alpha)*Decimal.from_float(float(descent))
                    row['last_candidate_theta']=own(candidate);summary.update(Armijo_trials=trials,last_alpha=alpha,last_Decimal_candidate=str(candidate_value),last_Decimal_RHS=str(rhs))
                    if candidate_value<=rhs: break
                    alpha*=.5
                else: raise RuntimeError('Frozen Armijo60 exhausted')
                theta=candidate.detach();iterations+=1;report['arithmetic_work']['Newton_steps']+=1
            row.update(own(dict(theta=theta,gradient=gradient,Hessian=H)));summary.update(completed=True,head_gradient_abs_max=norm,Newton_steps=iterations,Armijo_trials=trials,inner_loss=float(inner(x,t,theta)))
            report['counts']['stationary_heads']+=1;gate('stationary_head_'+role,norm<=1e-12 and bool(torch.isfinite(theta).all()))
            return theta,x,t,H
        M0,M=dense_moments(U0,V).detach(),dense_moments(U,V).detach();saved.update(M0=own(M0),current_M=own(M))
        theta0,x0,t0,H0=head(M0,'baseline')
        CE0=(-(q*(source_X@theta0.T).log_softmax(1)).sum()/n).detach().clone();saved['CE0']=own(CE0);report['counts']['CE0_captures']+=1
        gate('own_positive_CE0',bool(torch.isfinite(CE0)) and float(CE0)>0)
        theta,x,t,H=head(M,'current');p=(source_X@theta.T).softmax(1)
        denseCE=-(q*(source_X@theta.T).log_softmax(1)).sum()/n;denseRHS=(q.sum(1,keepdim=True)*p-q).T@source_X/n
        saved.update(theta=own(theta),dense_source_CE=own(denseCE),dense_source_RHS=own(denseRHS))
        report['attempts']['adjoints']+=1
        vector=torch.linalg.solve(H,denseRHS.reshape(-1)).reshape_as(theta).detach();report['counts']['adjoints']+=1;saved.update(H=own(H),raw_adjoint=own(vector))
        gate('full_coupled_adjoint_residual',float((H@vector.flatten()-denseRHS.flatten()).abs().max())<=1e-12)
        p=(x@theta.T).softmax(1);ts=t.sum(1,keepdim=True);err=ts*p-t;vx=x@vector.T;pa=(p*vx).sum(1,keepdim=True)
        RX=-(err@vector+(ts*p*(vx-pa))@theta)/k
        gc=RX[:,:d]+RX[:,d:F]@L;gt=(vx-pa)/k;mu=M[:,0:1];cen=M[:,1:1+d]/mu
        rawG=torch.cat((-((gc*cen).sum(1,keepdim=True)+(gt*t).sum(1,keepdim=True))/mu,gc/mu,gt/mu),1).detach()
        completeG=(rawG/CE0).detach();saved.update(raw_feature_G=own(RX),raw_center_G=own(gc),raw_target_G=own(gt),manual_raw_G=own(rawG),manual_complete_G=own(completeG))
        report['counts']['manual_complete_chains']+=1;report['attempts']['generic_moment_gradient_calls']+=1
        actualraw=moment_gradient(M,d,inner_features,theta.detach(),vector,penalty,inner_loss_weighting='uniform').detach()
        report['counts']['generic_moment_gradient_calls']+=1;saved['public_raw_G']=own(actualraw)
        actualcomplete=(actualraw/CE0).detach();saved['public_complete_G']=own(actualcomplete);report['counts']['complete_normalizations']+=1
        def compare(name,a,ref,absolute=1e-10,relative=1e-8):
            diff=(a-ref).detach();ab=float(diff.abs().max());rel=float((diff.abs()/torch.maximum(torch.maximum(a.abs(),ref.abs()),a.new_tensor(1e-12))).max())
            saved.setdefault('differences',{})[name]=own(diff)
            gate(name,bool(torch.isfinite(a).all()) and bool(torch.isfinite(ref).all()) and ab<=absolute and rel<=relative,abs_max=ab,max_rel=rel)
        for label,a,ref in (('raw',actualraw,rawG),('complete',actualcomplete,completeG)):
            for block,sl in (('mass',slice(0,1)),('z',slice(1,1+d)),('Q',slice(1+d,None)),('full',slice(None))): compare(label+'_'+block,a[:,sl],ref[:,sl])
        gate('G_FP64_detached_units',all(val.dtype==torch.float64 and not val.requires_grad and val.grad_fn is None for val in (actualraw,actualcomplete,vector)))
        denseP=dense_probability(U,V);R=S@completeG.T/n;W=denseP*(R-(denseP*R).sum(1,keepdim=True))
        denseDU,denseDV=W@V/math.sqrt(r),W.T@U/math.sqrt(r);saved.update(dense_probability=own(denseP),dense_R=own(R),dense_W=own(W),dense_dU=own(denseDU),dense_dV=own(denseDV))
        uf,vf=U.float(),V.float();nativeP=dense_probability(uf,vf);nativeR=S@completeG.T/n;nativeW=nativeP*(nativeR-(nativeP*nativeR).sum(1,keepdim=True))
        nativeDU,nativeDV=(nativeW.float()/math.sqrt(r))@vf,(nativeW.float()/math.sqrt(r)).T@uf
        saved.update(native_dense_probability=own(nativeP),native_dense_R=own(nativeR),native_dense_W=own(nativeW),native_dense_dU=own(nativeDU),native_dense_dV=own(nativeDV))
        def native_compare(name,a,ref):
            diff=(a-ref).detach();ab=float(diff.abs().max());den=float(ref.norm());an=float(a.norm());rel=float(diff.norm())/den if den>0 else None
            saved.setdefault('differences',{})[name]=own(diff)
            gate(name,bool(torch.isfinite(a).all()) and bool(torch.isfinite(ref).all()) and den>0 and an>0 and ab<=5e-7 and rel is not None and rel<=2e-5,abs_max=ab,relative_L2=rel,reference_L2=den,candidate_L2=an)
        results=[];saved['partition_results']=results
        def invoke(owner,domain,repeat):
            check();row=dict(domain=domain,partition=list(owner.partition),repeat=repeat);results.append(row)
            cu,cv=(U,V) if domain=='smooth' else (uf,vf)
            key='smooth' if domain=='smooth' else 'native'
            report['attempts'][key+'_forwards']+=1
            try: mm,fe=backend.physical_moments(cu,cv,hard,owner,mixing,owner.partition)
            finally: row['last_forward_evidence']=own(owner.last_evidence)
            report['counts'][key+'_forwards']+=1;row.update(M=own(mm),forward=own(fe))
            quotient=(mm[:,0:1],mm[:,1:1+d]/mm[:,0:1],mm[:,1+d:]/mm[:,0:1]);row['quotients']=own(quotient)
            report['attempts'][key+'_factor_adjoints']+=1
            try: du,dv,ae=backend.physical_factor_adjoint(cu,cv,hard,owner,actualcomplete,mixing,owner.partition,domain=='native')
            finally: row['last_adjoint_evidence']=own(owner.last_evidence)
            report['counts'][key+'_factor_adjoints']+=1;row.update(dU=own(du),dV=own(dv),adjoint=own(ae))
            if domain=='smooth':
                compare('M_'+str(owner.partition),mm,M)
                for label,a,ref in zip(('mass','centers','targets'),quotient,(M[:,0:1],M[:,1:1+d]/M[:,0:1],M[:,1+d:]/M[:,0:1])): compare(label+'_'+str(owner.partition),a,ref)
                gx,hx=dense_terms(torch.cat((inner_features(quotient[1]),torch.ones((k,1),dtype=torch.float64)),1),quotient[2],theta)
                row['held_theta_gradient']=own(gx);gate('held_theta_'+str(owner.partition),float(gx.abs().max())<=1e-10)
                compare('smooth_dU_'+str(owner.partition),du,denseDU);compare('smooth_dV_'+str(owner.partition),dv,denseDV)
                report['attempts']['source_CE_RHS']+=1
                try: ce,rhs,se=backend.source_ce_rhs(theta,owner,owner.partition)
                finally: row['last_source_evidence']=own(owner.last_evidence)
                report['counts']['source_CE_RHS']+=1;row.update(source_CE=own(ce),source_RHS=own(rhs),source=own(se))
                compare('source_CE_'+str(owner.partition),ce,denseCE);compare('source_RHS_'+str(owner.partition),rhs,denseRHS)
            else:
                # Independent literal dense-source reference, SAME original block order for native M.
                nm=torch.zeros_like(M)
                for start,end in owner.blocks(owner.partition):
                    prior=torch.full((end-start,k),float(np.log(mixing/k)),dtype=torch.float32)
                    prior.scatter_(1,hard[start:end,None],float(np.log(1-mixing+mixing/k)))
                    pp=(prior+uf[start:end]@vf.T/math.sqrt(r)).double().softmax(1)
                    nm+=pp.T@S[start:end]/n
                row['native_original_order_M_reference']=own(nm)
                compare('native_M_original_order_'+str(owner.partition)+'_'+str(repeat),mm,nm)
                for label,a,ref in zip(('mass','centers','targets'),quotient,(nm[:,0:1],nm[:,1:1+d]/nm[:,0:1],nm[:,1+d:]/nm[:,0:1])):
                    compare('native_'+label+'_'+str(owner.partition)+'_'+str(repeat),a,ref)
                native_compare('native_dU_'+str(owner.partition)+'_'+str(repeat),du,nativeDU)
                native_compare('native_dV_'+str(owner.partition)+'_'+str(repeat),dv,nativeDV)
            return row
        for owner in owners: invoke(owner,'smooth',0)
        native_rows=[invoke(owner,'native',0) for owner in owners]
        repeats=[native_rows[1],invoke(owners[1],'native',1),invoke(owners[1],'native',2)]
        for idx,row in enumerate(repeats[1:],1):
            gate('same_native_repeat_'+str(idx),all(backend.tensor_bytes(row[key])==backend.tensor_bytes(repeats[0][key]) for key in ('M','dU','dV'))
                and [item['probability_bytes_sha256'] for item in row['forward']['blocks']]==[item['probability_bytes_sha256'] for item in repeats[0]['forward']['blocks']])
        eps=fixture['FD_epsilon']
        for name,index in (('mass',(1,0)),('z',(2,1)),('Q',(0,1+d+1)),('U',(4,1)),('V',(3,0))):
            direction=torch.zeros_like(U if name=='U' else V if name=='V' else M);direction[index]=1
            analytic=float(((denseDU if name=='U' else denseDV if name=='V' else completeG)*direction).sum());endpoints=[]
            record=dict(name=name,epsilon=eps,direction=index,analytic=analytic,endpoints=endpoints);report['FD_summaries'].append(record);saved['FD_summaries']=report['FD_summaries']
            for sign in (1,-1):
                mp=dense_moments(U+sign*eps*direction,V) if name=='U' else dense_moments(U,V+sign*eps*direction) if name=='V' else M+sign*eps*direction
                saved.setdefault('FD_inputs',[]).append(own(dict(name=name,sign=sign,M=mp)))
                gate('FD_positive_'+name+'_'+str(sign),bool((mp[:,0]>0).all()) and bool((mp[:,1+d:]>0).all()))
                th,fx,ft,fh=head(mp,name+('_plus' if sign==1 else '_minus'))
                raw=-(q*(source_X@th.T).log_softmax(1)).sum()/n;objective=float(raw/CE0)
                endpoints.append(dict(sign=sign,raw_source_CE=float(raw),objective=objective));report['counts']['FD_endpoints']+=1
            numerical=(endpoints[0]['objective']-endpoints[1]['objective'])/(2*eps);ab=abs(analytic-numerical);rel=ab/max(abs(analytic),abs(numerical),1e-12)
            record.update(numerical=numerical,abs_error=ab,relative_error=rel);report['counts']['FD_records']+=1
            gate('FD_BOTH_'+name,math.isfinite(numerical) and ab<=1e-6 and rel<=.005,abs_error=ab,max_rel=rel)
        gate('immutable_CE0',torch.equal(CE0,saved['CE0']))
        for component in components.values(): component.verify()
        report['component_work']={name:dict(work=value.work,staging=value.staging) for name,value in components.items()}
        report['backend_ownership']=[dict(descriptor=owner.descriptor(),work=owner.work,memory=owner.memory) for owner in owners]
        saved['backend_ownership']=report['backend_ownership']
        gate('selected_discrete_counts',report['counts']==science['counts'])
        report['readonly_exit']=pin_values();gate('exit_pins',report['readonly_exit']==report['readonly_entry'])
        report['source_exit']=implementation_provenance();gate('source_exit',report['source_exit']==packet['source'])
        gate('CPU_no_GPU',not torch.cuda.is_initialized());check();report['completed']=True;report['passed']=True
    except BaseException as exc:
        if "owners" in locals():
            report['backend_ownership']=[dict(descriptor=o.descriptor(),work=o.work,memory=o.memory) for o in owners]
            saved['partial_last_backend_evidence']=[own(o.last_evidence) for o in owners]
        if "components" in locals(): report['component_work']={name:dict(work=v.work,staging=v.staging) for name,v in components.items()}
        report["failure"]=dict(type=type(exc).__name__,error=str(exc))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old_handler)
        if old_timer[0]>0: signal.setitimer(signal.ITIMER_REAL,*old_timer)
        if torch is not None:
            try:
                torch.save(dict(schema=1,kind=KIND,source=packet["source"],scientific_contract=sr,saved=saved),arrays)
                report["raw_evidence"]["write_completed"]=True
            except BaseException as exc:
                report["passed"]=False;report["failure"]=report["failure"] or dict(type=type(exc).__name__,error=str(exc))
            try:
                report["raw_evidence"]["exists"]=arrays.exists()
                if report["raw_evidence"]["exists"]:
                    report["raw_evidence"]["bytes"]=arrays.stat().st_size
                    try:
                        report["raw_evidence"]["sha256"]=sha(arrays);report["raw_evidence"]["hash_unknown"]=False
                    except BaseException as exc:
                        report["raw_evidence"]["hash_error"]=dict(type=type(exc).__name__,error=str(exc));raise
            except BaseException as exc:
                report["passed"]=False;report["failure"]=report["failure"] or dict(type=type(exc).__name__,error=str(exc))
        if not report["completed"]:
            try:
                report["readonly_exit"]=pin_values()
                require(report["readonly_exit"]==packet["readonly_files_sha256"],"Failure exit pins changed")
                report["source_exit"]=implementation_provenance()
                require(report["source_exit"]==packet["source"],"Failure exit source changed")
            except BaseException as exc: report["failure_exit_verification_unknown_or_failed"]=dict(type=type(exc).__name__,error=str(exc))
        report["resources"]=peaks()
        if not all(v<=science["resources"][k] for k,v in report["resources"].items()):
            report["passed"]=False;report["failure"]=report["failure"] or dict(type="ResourceBoundary",error="Frozen final evidence boundary")
        report_path.write_text(report_json())
        report["resources"]=peaks()
        if not all(v<=science["resources"][k] for k,v in report["resources"].items()):
            report["passed"]=False;report["failure"]=report["failure"] or dict(type="ResourceBoundary",error="Frozen post-report boundary")
            report_path.write_text(report_json())
    require(report["passed"],"Terminal FC CPU algebra failure; preserve owning report/arrays, no retry or fixture search")
    return dict(passed=report["passed"],report=dict(path=str(report_path),sha256=sha(report_path)),
        arrays=report["raw_evidence"],counts=report["counts"])
