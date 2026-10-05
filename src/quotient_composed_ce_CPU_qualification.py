"""Selected tiny CPU quotient/composed proof; never a real native admission."""
import hashlib
import json
import math
import os
import platform
import resource
import signal
import sys
import time
from decimal import Context, Decimal, ROUND_HALF_EVEN, localcontext
from pathlib import Path

KIND = 'quotient_propagated_original_Nystrom_composed_uniform_CE_tiny_CPU_complete_chain_FIRST_v1'
SCIENCE_SHA = 'a2c93fa31ef2e6b07222f2d3f5180fb67eaec94e52694be8aac7b557ea3b9b41'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def refs(value, found=None):
    found = {} if found is None else found
    if isinstance(value, dict):
        if set(value) == {'path', 'sha256'}:
            p, h = value['path'], value['sha256']
            require(Path(p).is_absolute() and str(Path(p).resolve()) == p, 'Ref path')
            require(p not in found or found[p] == h, 'Conflicting refs'); found[p] = h
        for v in value.values(): refs(v, found)
    elif isinstance(value, list):
        for v in value: refs(v, found)
    return found


def observed(value):
    if isinstance(value, float) and not math.isfinite(value):
        return {'nonfinite': repr(value)}
    if isinstance(value, dict): return {k:observed(v) for k,v in value.items()}
    if isinstance(value, (tuple, list)): return [observed(v) for v in value]
    return value


def run(protocol_path, protocol_sha256, stop=lambda: False):
    started = time.monotonic(); require(str(Path(protocol_path).resolve()) == protocol_path
        and sha(protocol_path) == protocol_sha256, 'Protocol path/SHA')
    packet = json.loads(Path(protocol_path).read_text()); sr = packet['scientific_contract']
    require(sr['sha256'] == SCIENCE_SHA and sha(sr['path']) == SCIENCE_SHA, 'Science SHA')
    science = json.loads(Path(sr['path']).read_text()); require(packet['kind'] == science['kind'] == KIND, 'Kind')
    folder = Path(packet['output_folder']); require(folder.is_absolute() and str(folder.resolve()) == str(folder), 'Output path')
    folder.mkdir(parents=True, exist_ok=False); arrays = folder/'qualification_arrays.pt'; rp = folder/'qualification_report.json'
    work = {k:0 for k in science['exact_counts']}; attempts = {}; arithmetic = {}; saved = {}; live = {}
    report = dict(schema=1, kind=KIND, source=packet['source'], scientific_contract=sr,
        protocol=dict(path=protocol_path,sha256=protocol_sha256), passed=False, completed=False,
        failure=None, gates=[], work=work, attempts=attempts, arithmetic=arithmetic,
        head_summaries=[], FD_summaries=[], native_domain='CPU32 heldR only; not GPU or real native qualification',
        raw_evidence=dict(path=str(arrays),exists=False,bytes=None,sha256=None,hash_unknown=True,write_completed=False),
        ownership_policy='Detached CPU aliases registered before any guard; cloned owning copy replaces aliases only on success; raw live storage retained and charged')
    torch = None; old_profile = sys.getprofile(); old_signal = signal.getsignal(signal.SIGALRM); old_timer = signal.getitimer(signal.ITIMER_REAL)
    def count(name): work[name] += 1
    def attempt(name): attempts[name] = attempts.get(name,0)+1
    def arith(name): arithmetic[name] = arithmetic.get(name,0)+1
    def storage_bytes(value, seen=None):
        seen = set() if seen is None else seen
        if torch is not None and torch.is_tensor(value):
            st=value.untyped_storage(); key=(str(value.device),st.data_ptr(),st.nbytes())
            if key in seen: return 0
            seen.add(key); return st.nbytes()
        if isinstance(value,dict): return sum(storage_bytes(v,seen) for v in value.values())
        if isinstance(value,(tuple,list)): return sum(storage_bytes(v,seen) for v in value)
        return 0
    def peaks():
        return dict(seconds=time.monotonic()-started,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            active_owned_tensor_storage_bytes=storage_bytes((saved,live)),peak_CUDA_allocated_bytes=0,peak_CUDA_reserved_bytes=0)
    def guard(future=0):
        v=peaks(); report['resources']=v
        require(not stop() and v['seconds']<=300 and v['peak_RSS_bytes']<=science['resources']['peak_RSS_bytes_max']
            and v['active_owned_tensor_storage_bytes']+future<=science['resources']['active_owned_tensor_storage_bytes_max']
            and (torch is None or not torch.cuda.is_initialized()), 'Frozen CPU stop/resource boundary')
    def cpu(value):
        if torch.is_tensor(value): return value.detach().cpu().clone()
        if isinstance(value,dict): return {k:cpu(v) for k,v in value.items()}
        if isinstance(value,(tuple,list)): return [cpu(v) for v in value]
        return value
    def detached_alias(value):
        if torch.is_tensor(value): return value.detach()
        if isinstance(value,dict): return {k:detached_alias(v) for k,v in value.items()}
        if isinstance(value,(tuple,list)): return [detached_alias(v) for v in value]
        return value
    def own(key,value):
        live[key]=value; saved[key]=detached_alias(value)
        guard(storage_bytes(value)); saved[key]=cpu(value); guard(); return saved[key]
    def gate(name,ok,**detail):
        report['gates'].append(dict(name=name,passed=bool(ok),**observed(detail))); require(ok,name); guard()
    def pins():
        expected=packet['readonly_files_sha256']; require(all(expected.get(p)==h for p,h in refs(dict(packet=packet,science=science)).items()),'Pin closure')
        actual={p:sha(p) for p in expected}; require(actual==expected,'Readonly bytes'); return actual
    def alarm(signum,frame): raise TimeoutError('Frozen300s CPU deadline')
    try:
        signal.signal(signal.SIGALRM,alarm); signal.setitimer(signal.ITIMER_REAL,max(1e-6,300-(time.monotonic()-started)))
        report['readonly_entry']=pins()
        bridge_ref=packet['prerequisite_admission']; require(sha(bridge_ref['path'])==bridge_ref['sha256'],'Static bridge bytes')
        bridge=json.loads(Path(bridge_ref['path']).read_text())
        gate('source_science_static_bridge',bridge['passed'] is True and bridge['source']==packet['source']
             and bridge['scientific_contract']==sr and bridge['entrypoint']==packet['entrypoint']
             and bridge['helper_entrypoint']==packet['helper_entrypoint'])
        freeze=json.loads(Path(packet['ROOT_science_freeze']['path']).read_text())
        gate('selected_science_before_code',freeze['passed'] is True and freeze['selected'] is True and freeze['scientific_contract']==sr)
        import numpy as np
        import torch as torch_module
        torch=torch_module
        from src import quotient_composed_ce as public
        from src.coarsening import feature_centroids, quotient_adjacency
        from src.low_rank_assignment import LowRankLogits, logit_block
        from src.moments import augmented
        from src.nystrom_ce import NystromMap, outer_gradient
        from src.transforms import FeatureTransform
        from src.soft_ce_partition import head_gradient, hessian_operator
        from src.research_loop import implementation_provenance
        require(sha(__file__)==packet['entrypoint']['sha256'] and str(Path(__file__).resolve())==packet['entrypoint']['path']
            and sha(public.__file__)==packet['helper_entrypoint']['sha256'] and str(Path(public.__file__).resolve())==packet['helper_entrypoint']['path'],'Typed source paths')
        report['source_entry']=implementation_provenance(); gate('source_entry',report['source_entry']==packet['source'])
        torch.set_num_threads(4)
        if torch.get_num_interop_threads()!=1: torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest')
        runtime=dict(Python=platform.python_version(),Torch=str(torch.__version__),NumPy=np.__version__,threads=torch.get_num_threads(),
            interop_threads=torch.get_num_interop_threads(),default_dtype=str(torch.get_default_dtype()),deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            AMP=torch.is_autocast_enabled('cpu'),TF32=torch.backends.cuda.matmul.allow_tf32,float32_matmul_precision=torch.get_float32_matmul_precision())
        report['runtime']=runtime; expected=science['resources']['runtime']
        gate('exact_CPU_runtime',all(runtime[k]==expected[k] for k in runtime) and not torch.cuda.is_initialized())
        f=science['fixture']; dims=f['dimensions']; n,d,b,c,k,r=(dims[x] for x in ('N','D','B','C','K','rank')); width=d+b
        def tensor(v): return torch.tensor(v,dtype=torch.float64,device='cpu')
        X,Q,A,anchors,mapping=(tensor(f['X']),tensor(f['Q']),tensor(f['source_graph']['weighted_A']),tensor(f['anchors']),tensor(f['mapping']))
        hard=torch.tensor(f['hard'],dtype=torch.int64); U0,V0=tensor(f['U0']),tensor(f['V0'])
        U=U0+tensor(f['U1_delta']); V=V0+tensor(f['V1_delta']); penalty=f['penalty']; mixing=f['mixing']
        t=f['RMS']; center,oc,scale=tensor(t['center']),tensor(t['output_center']),tensor(t['scale'])
        transform=FeatureTransform(center,None,oc,scale,'rms',1e-12); feature_map=NystromMap(anchors,mapping,'relu')
        own('literal_inputs',dict(X=X,Q=Q,A=A,hard=hard,U0=U0,V0=V0,U=U,V=V,anchors=anchors,mapping=mapping,center=center,output_center=oc,scale=scale))
        gate('literal_general_Q_nonzero_factors',bool((Q>0).all()) and float((Q.sum(1)-1).abs().max())>1e-2
            and all(float(v.norm())>0 for v in (U0,V0,U,V)) and torch.equal(A,A.T) and bool((A.diagonal()>0).all()))
        degree=A.sum(1); inv=degree.rsqrt(); S=inv[:,None]*A*inv[None,:]; count('source_graph_normalizations'); own('source_S_degree',dict(S=S,degree=degree))
        Hsource=S@(S@X); count('source_S_squared_X_builds'); zsource=(Hsource-center-oc)/scale
        hsource=zsource*scale+oc+center; own('source_H_z_inverse',dict(H=Hsource,z=zsource,map_input=hsource))
        prior=torch.full((n,k),math.log(mixing/k),dtype=torch.float64); prior[torch.arange(n),hard]=math.log(1-mixing+mixing/k)
        P0=(prior+U0@V0.T/math.sqrt(r)).softmax(1); P=(prior+U@V.T/math.sqrt(r)).softmax(1); own('smooth_prior_P',dict(prior=prior,P0=P0,P=P))
        bandwidth=anchors.square().sum(1).mean()/d; an=anchors.norm(dim=1)
        def smooth(h,role):
            hn=h.norm(dim=1); co=(h@anchors.T)/(hn[:,None]*an[None,:]); angle=co.acos()
            relu=(angle.sin()+(math.pi-angle)*angle.cos())/math.pi
            own('smooth_'+role,dict(input=h,norms=hn,cosine=co,relu=relu))
            gate('strict_map_interior_'+role,bool((hn>1e-8).all()) and bool((an>1e-8).all()) and float(bandwidth)>1e-8
                and bool((co.abs()<1-1e-4).all()) and bool((relu>-1+1e-4).all()) and bool((relu<1-1e-4).all()))
            return hn,co,angle,relu
        def analytic_map(h,role):
            hn,co,angle,relu=smooth(h,role)
            value=hn[:,None]*an[None,:]/(d*bandwidth)*relu@mapping
            own('analytic_map_'+role,value); count('independent_source_analytic_map' if role=='source' else 'independent_endpoint_analytic_maps')
            return value
        paths=dict(coarsening=str(Path(feature_centroids.__code__.co_filename).resolve()),nystrom=str(Path(NystromMap.__call__.__code__.co_filename).resolve()))
        def profile(frame,event,arg):
            name=frame.f_code.co_name; path=str(Path(frame.f_code.co_filename).resolve())
            key=None
            if path==paths['coarsening'] and name=='feature_centroids': key='original_feature_centroids_calls'
            elif path==paths['coarsening'] and name=='quotient_adjacency': key='original_coarsening_quotient_adjacency_calls'
            elif path==paths['nystrom'] and name=='__call__': key='original_map_calls_total'
            if key and event=='call': attempt(key)
            if key and event=='return' and arg is not None: count(key)
            if old_profile is not None: old_profile(frame,event,arg)
        sys.setprofile(profile)
        def observe(name,value):
            own('public_'+name,value)
            if name=='map_input': smooth(value,'public_'+str(attempts.get('original_map_calls_total',0)))
        def compare(name,actual,reference,limits):
            own('comparison_'+name,dict(actual=actual,reference=reference)); delta=actual-reference; live['comparison_delta']=delta
            absolute=float(delta.abs().max()); denom=float(reference.norm()) if 'relative_L2_max' in limits else float(reference.abs().max())
            numerator=float(delta.norm()) if 'relative_L2_max' in limits else absolute
            rel=numerator/denom if denom>0 else None
            gate(name, bool(torch.isfinite(actual).all()) and bool(torch.isfinite(reference).all()) and denom>0
                and math.isfinite(absolute) and math.isfinite(rel) and absolute<=limits['absolute_Linf_max']
                and rel<=limits.get('relative_L2_max',limits.get('relative_Linf_max')),absolute_Linf=absolute,relative=rel,reference_denominator=denom)
        limits=science['gates']; source_manual=torch.cat((zsource,analytic_map(hsource,'source')),1)
        smooth(hsource,'source_original'); source_actual=torch.cat((zsource,feature_map(hsource)),1); count('original_source_map_calls')
        compare('fixed_source_features',source_actual,source_manual,limits['endpoint_fields'])
        source_aug=augmented(source_manual).detach(); source_rows=source_actual.detach().numpy().copy(); source_rows.setflags(write=False)
        own('fixed_source_feature_rows',dict(manual=source_manual,actual=source_actual,augmented=source_aug))
        def independent_build(prob,role):
            mu=prob.sum(0); xc=(prob.T@X)/mu[:,None]; qc=(prob.T@Q)/mu[:,None]
            coarse=prob.T@(S@prob); deg=coarse.sum(1); a=deg.rsqrt(); sc=a[:,None]*coarse*a[None,:]
            hc=sc@(sc@xc); zc=(hc-center-oc)/scale; h=zc*scale+oc+center
            value=dict(n=mu,Xc=xc,Qc=qc,C=coarse,degree=deg,Sc=sc,Hc=hc,zc=zc,map_input=h,features=torch.cat((zc,analytic_map(h,role)),1))
            own('independent_'+role,value); count('independent_endpoint_quotient_feature_builds')
            gate('positive_endpoint_'+role,bool((prob>0).all()) and bool((mu>0).all()) and bool((deg>0).all()))
            attempt('public_endpoint_build_calls'); actual=public.build_quotient_features(prob,X,Q,S,transform,feature_map,observe); count('public_endpoint_build_calls')
            own('public_endpoint_'+role,actual)
            for key in value: compare(role+'_'+key,actual[key],value[key],limits['endpoint_fields'])
            return value
        def terms(x,t,theta):
            p=(x@theta.T).softmax(1); sums=t.sum(1)
            grad=((sums[:,None]*p-t).T@x)/k+penalty*theta
            cov=(torch.diag_embed(p)-p[:,:,None]*p[:,None,:])*(sums/k)[:,None,None]
            xx=x[:,:,None]*x[:,None,:]; hess=torch.einsum('iab,ijl->ajbl',cov,xx).reshape(c*(width+1),c*(width+1))
            hess=hess+penalty*torch.eye(c*(width+1),dtype=torch.float64); arith('dense_Hessians')
            return grad,hess
        def decimal_loss(xv,tv,theta):
            arith('Decimal_scalar_objectives')
            with localcontext(Context(prec=80,rounding=ROUND_HALF_EVEN)):
                th=[[Decimal.from_float(float(v)) for v in row] for row in theta.tolist()]; loss=Decimal(0)
                for xi,ti in zip(xv,tv):
                    logits=[sum((a*b for a,b in zip(xi,row)),Decimal(0)) for row in th]; shift=max(logits)
                    lse=shift+sum(((v-shift).exp() for v in logits),Decimal(0)).ln()
                    loss+=sum((q*(lse-v) for q,v in zip(ti,logits)),Decimal(0))
                return loss/Decimal(k)+Decimal.from_float(float(penalty))*sum((v*v for row in th for v in row),Decimal(0))/Decimal(2)
        def endpoint(prob,role):
            attempt('independent_stationary_heads'); count('independent_stationary_head_attempts')
            value=independent_build(prob,role); x=augmented(value['features']); targets=value['Qc']; theta=torch.zeros((c,width+1),dtype=torch.float64)
            xv=[[Decimal.from_float(float(v)) for v in row] for row in x.tolist()]; tv=[[Decimal.from_float(float(v)) for v in row] for row in targets.tolist()]
            summary=dict(role=role,completed=False,Newton_steps=0,Armijo_trials=0); report['head_summaries'].append(summary)
            for step in range(101):
                guard(); grad,hess=terms(x,targets,theta); norm=float(grad.abs().max())
                own('head_'+role,dict(theta=theta,gradient=grad,Hessian=hess,features=x,targets=targets,chain=value))
                summary.update(last_theta=theta.detach().tolist(),gradient_abs_Linf=norm)
                if norm<=1e-12: break
                require(step<100,'Frozen Newton100 exhausted')
                direction=torch.linalg.solve(hess,grad.flatten()).reshape_as(theta); arith('Newton_direction_solves')
                own('head_direction_'+role,direction); descent=float((grad*direction).sum()); gate('descent_'+role+'_'+str(step),math.isfinite(descent) and descent>0)
                current=decimal_loss(xv,tv,theta); alpha=1.
                for trial in range(60):
                    guard(); candidate=theta-alpha*direction; own('candidate_'+role,candidate); summary['Armijo_trials']+=1; arith('Armijo_trials')
                    proposed=decimal_loss(xv,tv,candidate)
                    with localcontext(Context(prec=80,rounding=ROUND_HALF_EVEN)):
                        threshold=current-Decimal.from_float(1e-4)*Decimal.from_float(alpha)*Decimal.from_float(descent)
                    summary.update(last_alpha=alpha,last_Decimal_current=str(current),last_Decimal_candidate=str(proposed),last_Decimal_rhs=str(threshold))
                    if proposed<=threshold: break
                    alpha*=.5
                else: raise RuntimeError('Frozen Armijo60 exhausted')
                theta=candidate.detach(); summary['Newton_steps']+=1; arith('Newton_steps')
            raw=-(Q*(source_aug@theta.T).log_softmax(1)).sum()/n
            result=dict(theta=theta,Hessian=hess,gradient=grad,chain=value,x=x,targets=targets,raw_CE=raw)
            own('endpoint_'+role,result); summary.update(completed=True,raw_outer_CE=float(raw)); count('independent_stationary_heads')
            count('independent_source_outer_CE_evaluations'); count('baseline_heads' if role=='baseline' else 'current_heads' if role=='current' else 'reoptimized_FD_heads')
            gate('head_stationarity_'+role,norm<=1e-12 and bool(torch.isfinite(theta).all())); return result
        baseline=endpoint(P0,'baseline')
        ce0,rhs0=outer_gradient(source_rows,Q,baseline['theta'],chunk=n); count('public_original_outer_gradient_calls'); own('baseline_public_outer',dict(CE=ce0,RHS=rhs0))
        compare('baseline_public_CE',tensor(ce0),baseline['raw_CE'],limits['endpoint_fields'])
        baseline_source_p=(source_aug@baseline['theta'].T).softmax(1)
        baseline_manual_rhs=((Q.sum(1,keepdim=True)*baseline_source_p-Q).T@source_aug)/n
        compare('baseline_public_RHS',rhs0,baseline_manual_rhs,limits['analytic_components_and_complete_R_and_smooth_factor'])
        CE0=float(ce0); gate('own_positive_frozen_CE0',math.isfinite(CE0) and CE0>0); own('CE0',tensor(CE0)); report['CE0']=CE0
        current=endpoint(P,'current'); theta=current['theta']; hess=current['Hessian']; chain=current['chain']; xc,qc,sc=(chain[x] for x in ('Xc','Qc','Sc'))
        raw,rhs=outer_gradient(source_rows,Q,theta,chunk=n); count('public_original_outer_gradient_calls'); own('current_public_outer',dict(CE=raw,RHS=rhs))
        sourcep=(source_aug@theta.T).softmax(1); manual_rhs=((Q.sum(1,keepdim=True)*sourcep-Q).T@source_aug)/n
        compare('current_public_CE',tensor(raw),current['raw_CE'],limits['endpoint_fields']); compare('source_full_RHS',rhs,manual_rhs,limits['analytic_components_and_complete_R_and_smooth_factor'])
        attempt('full_dense_current_adjoints'); vector=torch.linalg.solve(hess,rhs.flatten()).reshape_as(theta); count('full_dense_current_adjoints'); own('adjoint',dict(vector=vector,Hessian=hess,RHS=rhs))
        compare('full_adjoint_residual',hess@vector.flatten(),rhs.flatten(),limits['full_adjoint_residual'])
        grad=head_gradient(current['x'],qc,torch.full((k,),1/k,dtype=torch.float64),theta,penalty); count('public_original_head_gradient_fixed_theta_checks'); own('original_fixed_head_gradient',grad)
        gate('original_head_gradient_absolute_only',float(grad.abs().max())<=1e-12 and float((grad-current['gradient']).abs().max())<=1e-12)
        multiply,diag=hessian_operator(current['x'],qc,torch.full((k,),1/k,dtype=torch.float64),theta,penalty)
        own('original_Hessian_diagonal',diag)
        basis=torch.eye(theta.numel(),dtype=torch.float64); columns=[]
        for j in range(theta.numel()):
            returned=multiply(basis[j].reshape_as(theta)); columns.append(own('original_Hessian_column_'+str(j),returned).flatten())
        original_Hessian=torch.stack(columns,1); count('public_original_Hessian_operator_fixed_theta_checks')
        own('original_full_Hessian',dict(value=original_Hessian,diagonal=diag,basis=basis))
        compare('original_full_Hessian',original_Hessian,hess,limits['analytic_components_and_complete_R_and_smooth_factor'])
        p=(current['x']@theta.T).softmax(1); sums=qc.sum(1,keepdim=True); error=sums*p-qc; a=current['x']@vector.T; pa=(p*a).sum(1,keepdim=True)
        gf=-(error@vector+(sums*p*(a-pa))@theta)/k; gq=(a-pa)/k
        hn,co,angle,relu=smooth(chain['map_input'],'analytic_VJP'); cp=(math.pi-angle)/math.pi
        kernel_grad=an[None,:,None]/(d*bandwidth)*(chain['map_input'][:,None,:]/hn[:,None,None]*relu[:,:,None]
            +cp[:,:,None]*(anchors[None,:,:]/an[None,:,None]-co[:,:,None]*chain['map_input'][:,None,:]/hn[:,None,None]))
        mapped_rhs=gf[:,d:width]@mapping.T; map_vjp=(mapped_rhs[:,:,None]*kernel_grad).sum(1)
        gh=(gf[:,:d]+scale*map_vjp)/scale; count('analytic_current_map_VJP')
        gy=sc.T@gh; gx=sc.T@gy; gsc=gh@(sc@xc).T+gy@xc.T
        aa=chain['degree'].rsqrt(); C=chain['C']; ga=(gsc*C*aa[None,:]).sum(1)+(gsc*aa[:,None]*C).sum(0)
        gd=-.5*chain['degree'].pow(-1.5)*ga; gc=aa[:,None]*gsc*aa[None,:]+gd[:,None]
        rc=((X[:,None,:]-xc[None,:,:])*gx[None,:,:]).sum(2)/chain['n'][None,:]+((Q[:,None,:]-qc[None,:,:])*gq[None,:,:]).sum(2)/chain['n'][None,:]
        rg=(S@P)@gc.T+(S.T@P)@gc; rawR=rc+rg; completeR=rawR/CE0; count('manual_complete_row_cotangent_evaluations')
        own('manual_complete_chain',dict(gf=gf,gq=gq,map_vjp=map_vjp,gh=gh,gx=gx,gsc=gsc,ga=ga,gd=gd,gc=gc,Rcent=rc,Rgraph=rg,raw_R=rawR,complete_R=completeR))
        gate('nonzero_coupled_components',all(float(v.norm())>0 and bool(torch.isfinite(v).all()) for v in (rc,rg,gd,map_vjp,gf[:,:d],gq)))
        attempt('public_complete_row_cotangent_calls'); actual=public.complete_row_cotangent(P,X,Q,S,transform,feature_map,theta,vector,penalty,CE0,observe); count('public_complete_row_cotangent_calls'); own('public_complete_row',actual)
        compare('complete_raw_R',actual['raw_R'],rawR,limits['analytic_components_and_complete_R_and_smooth_factor']); compare('complete_R',actual['complete_R'],completeR,limits['analytic_components_and_complete_R_and_smooth_factor'])
        Cleaf=C.detach().clone().requires_grad_(True); Xleaf=xc.detach().clone().requires_grad_(True); Qleaf=qc.detach().clone().requires_grad_(True)
        dd=Cleaf.sum(1); ll=dd.rsqrt(); ss=ll[:,None]*Cleaf*ll[None,:]
        local=((ss@(ss@Xleaf))*gh.detach()).sum()+(Qleaf*gq.detach()).sum()
        gcl,gxl,gql=torch.autograd.grad(local,(Cleaf,Xleaf,Qleaf)); count('local_C_Xc_Qc_held_chain_autograd_VJP'); own('local_chain_VJP',dict(C=gcl,Xc=gxl,Qc=gql))
        for name,a0,b0 in (('degree_C',gcl,gc),('twoSc_Xc',gxl,gx),('general_Q_target',gql,gq)):compare(name,a0,b0,limits['analytic_components_and_complete_R_and_smooth_factor'])
        W=P*(completeR-(P*completeR).sum(1,keepdim=True)); du=W@V/math.sqrt(r); dv=W.T@U/math.sqrt(r); count('smooth_FP64_manual_factor_pullbacks'); own('smooth_manual_factor',dict(W=W,U=du,V=dv))
        ul=U.detach().clone().requires_grad_(True); vl=V.detach().clone().requires_grad_(True); pp=(prior+ul@vl.T/math.sqrt(r)).softmax(1)
        auto=public.build_quotient_features(pp,X,Q,S,transform,feature_map,observe)
        gg=head_gradient(augmented(auto['features']),auto['Qc'],torch.full((k,),1/k,dtype=torch.float64),theta.detach(),penalty)
        dua,dva=torch.autograd.grad(-(gg*vector.detach()).sum()/CE0,(ul,vl)); count('smooth_factor_full_chain_autograd_VJP'); own('smooth_factor_autograd',dict(U=dua,V=dva))
        for name,a0,b0 in (('smooth_U',dua,du),('smooth_V',dva,dv),('smooth_full',torch.cat((dua.flatten(),dva.flatten())),torch.cat((du.flatten(),dv.flatten())))):compare(name,a0,b0,limits['analytic_components_and_complete_R_and_smooth_factor'])
        fd=science['finite_difference']; eps=fd['epsilon']
        for name,literal in fd['directions'].items():
            direction=tensor(literal); own('FD_direction_'+name,direction)
            if name.startswith('P_'):
                plus,minus=P+eps*direction,P-eps*direction; slope=(completeR*direction).sum()
                gate('FD_row_sum_'+name,bool(direction.sum(1).eq(0).all()))
                if name=='P_mass_changing': gate('mass_changing',bool(direction.sum(0).ne(0).any()))
                if name=='P_mass_neutral_cycle': gate('mass_neutral',bool(direction.sum(0).eq(0).all()))
                if name=='P_neighbor_degree_sensitive':gate('degree_sensitive',float((direction.T@(S@P)+P.T@(S@direction)).sum(1).norm())>0)
            else:
                pu,pv=(U+eps*direction,V) if name=='U' else (U,V+eps*direction)
                mu,mv=(U-eps*direction,V) if name=='U' else (U,V-eps*direction)
                plus=(prior+pu@pv.T/math.sqrt(r)).softmax(1); minus=(prior+mu@mv.T/math.sqrt(r)).softmax(1)
                slope=((du if name=='U' else dv)*direction).sum()
            own('FD_probabilities_'+name,dict(plus=plus,minus=minus,slope=slope))
            ep=endpoint(plus,name+'_plus'); count('reoptimized_FD_endpoints'); em=endpoint(minus,name+'_minus'); count('reoptimized_FD_endpoints')
            numeric=(ep['raw_CE']/CE0-em['raw_CE']/CE0)/(2*eps); own('FD_return_'+name,dict(plus=ep['raw_CE'],minus=em['raw_CE'],numeric=numeric,reference=slope))
            abs_error=float((numeric-slope).abs()); denominator=float(slope.abs()); relative=abs_error/denominator if denominator>0 else None
            row=dict(direction=name,reference=float(slope),numerical=float(numeric),absolute_error=abs_error,relative_error=relative);report['FD_summaries'].append(observed(row))
            gate('FD_'+name,denominator>0 and bool(torch.isfinite(numeric)) and float(numeric*slope)>0
                and math.isfinite(relative) and abs_error<=limits['reoptimized_FD']['absolute_error_max'] and relative<=limits['reoptimized_FD']['relative_error_max'],**row)
            count('reoptimized_FD_directions')
        U32,V32=U.float(),V.float(); work['native32_input_conversions']+=2; own('native32_inputs',dict(U=U32,V=V32,held_R=actual['complete_R']))
        repeats=[]
        for i in range(3):
            uu=U32.detach().clone().requires_grad_(True); vv=V32.detach().clone().requires_grad_(True)
            attempt('native32_LowRankLogits_forward'); logits=LowRankLogits.apply(uu,vv,hard,mixing,n); count('native32_LowRankLogits_forward'); own('native_logits_'+str(i),logits)
            prob=logits.double().softmax(1); own('native_P_'+str(i),prob)
            attempt('native32_LowRankLogits_backward'); gu,gv=torch.autograd.grad((prob*actual['complete_R']).sum(),(uu,vv)); count('native32_LowRankLogits_backward')
            row=dict(logits=logits,P=prob,U=gu,V=gv); repeats.append(own('native_repeat_'+str(i),row)); gate('nonzero_native_gradients_'+str(i),float(gu.norm())>0 and float(gv.norm())>0)
        def same_bytes(a,b):
            return a.dtype==b.dtype and tuple(a.shape)==tuple(b.shape) and bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all()) and a.contiguous().numpy().tobytes(order='C')==b.contiguous().numpy().tobytes(order='C')
        report['native_repeat_descriptors']=[{key:dict(shape=list(value.shape),dtype=str(value.dtype),sha256=hashlib.sha256(value.contiguous().numpy().tobytes(order='C')).hexdigest()) for key,value in row.items()} for row in repeats]
        gate('native3_same_R_byteequal',all(same_bytes(repeats[0][key],row[key]) for row in repeats[1:] for key in repeats[0]))
        native_logits=logit_block(U32,V32,hard,mixing); count('native32_public_logit_block_reference'); pn=native_logits.double().softmax(1)
        wn=pn*(actual['complete_R']-(pn*actual['complete_R']).sum(1,keepdim=True)); block=wn.float()/math.sqrt(r)
        gu,gv=block@V32,block.T@U32; count('native32_independent_wholeK_reference'); own('native_wholeK_reference',dict(logits=native_logits,P=pn,W64=wn,FP32_scaled_block=block,U=gu,V=gv))
        for name,a0,b0 in (('native_U',repeats[0]['U'],gu),('native_V',repeats[0]['V'],gv),('native_full',torch.cat((repeats[0]['U'].flatten(),repeats[0]['V'].flatten())),torch.cat((gu.flatten(),gv.flatten())))):
            compare(name,a0.double(),b0.double(),limits['native32_reference'])
        gate('all_selected_numerical_counts',all(work[key]==expected for key,expected in science['exact_counts'].items() if key not in ('own_raw_evidence_writes','final_report_writes')))
        report['completed']=True
    except BaseException as error:
        report['failure']=dict(type=type(error).__name__,message=str(error));report['unreturned_interiors']='Unknown; attempts and actual returned/owned values retained'
    finally:
        sys.setprofile(old_profile); signal.setitimer(signal.ITIMER_REAL,0)
        try:
            if torch is not None:
                try:
                    attempt('own_raw_evidence_writes'); guard()
                    with arrays.open('xb') as f:torch.save(saved,f);f.flush();os.fsync(f.fileno())
                    report['raw_evidence']['write_completed']=True; count('own_raw_evidence_writes')
                except BaseException as error: report['raw_write_error']=repr(error)
            info=report['raw_evidence'];info.update(exists=arrays.exists(),bytes=arrays.stat().st_size if arrays.exists() else None)
            if arrays.exists():
                try:info.update(sha256=sha(arrays),hash_unknown=False)
                except BaseException as error:info['hash_error']=repr(error)
            try:
                report['readonly_exit']=pins()
                if torch is not None:report['source_exit']=implementation_provenance();require(report['source_exit']==packet['source'],'Source exit')
                guard()
            except BaseException as error:report['exit_guard_error']=repr(error)
            count('final_report_writes'); report['resources']=peaks()
            report['passed']=bool(report['completed'] and report['failure'] is None and info['write_completed'] and not info['hash_unknown']
                and not any(key in report for key in ('raw_write_error','exit_guard_error'))
                and all(work[key]==expected for key,expected in science['exact_counts'].items()))
            if not report['passed']:report['completed']=False
            with rp.open('x') as f:json.dump(observed(report),f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
            try:guard()
            except BaseException as error:
                report.update(passed=False,completed=False,resources=peaks(),failure=dict(type='FinalSerializationResourceBoundary',message=str(error)))
                with rp.open('w') as f:json.dump(observed(report),f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
        finally:signal.signal(signal.SIGALRM,old_signal);signal.setitimer(signal.ITIMER_REAL,*old_timer)
    require(report['passed'],'Terminal FQ CPU proof failure; preserve namespace and no rescue')
    return str(rp)
