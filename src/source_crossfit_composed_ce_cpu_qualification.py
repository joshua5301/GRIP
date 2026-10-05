"""Unexecuted FL two-fold complete general-Q CPU proof; fixed Decimal80 Armijo."""
import hashlib
import json
import math
import resource
import signal
import sys
import time
from decimal import Decimal, Context, localcontext, ROUND_HALF_EVEN
from pathlib import Path

KIND = "two_fixed_source_fold_crossfit_composed_uniform_CE_literal_CPU_complete_chain_stageFL_v1"
SCIENCE_SHA = "8e4bba1c01ed91c279105edaf7523382d259cfaadc29eb1f75e292fa9cb862e3"


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
        gates=[], work={k:0 for k in science["counts"]},
        raw_evidence=dict(path=str(arrays), exists=False, bytes=None, sha256=None, hash_unknown=True,
            hash_error=None, write_completed=False), head_summaries=[], FD_summaries=[])
    report["attempts"] = dict(stationary_heads=0, adjoints=0, public_raw_cotangent_calls=0,
        public_packed_cotangent_calls=0, public_held_outer_calls=0, generic_moment_gradient_calls=0)
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
        freeze=json.loads(Path(packet["ROOT_science_freeze"]["path"]).read_text())
        gate("ROOT_science_frozen_before_calls",freeze["passed"] is True and freeze["scientific_contract"]==sr)
        bridge=json.loads(Path(packet["prerequisite_admission"]["path"]).read_text())
        gate("ROOT_peer_exact_static_prerequisites",bridge["passed"] is True
            and bridge["source"]==packet["source"] and bridge["scientific_contract"]==sr
            and bridge["entrypoint"]==packet["entrypoint"] and bridge["helper_entrypoint"]==packet["helper_entrypoint"])
        require(Path(packet["repository"]).is_absolute() and str(Path(packet["repository"]).resolve())==packet["repository"],"Repository ABS")
        import torch
        from src import composed_centroid_joint_ce as joint
        from src import source_crossfit_composed_ce as crossfit
        from src.low_rank_assignment import LowRankMoments
        from src.nystrom_ce import NystromMap
        from src.transforms import FeatureTransform
        from src.research_loop import implementation_provenance
        from src.shared_features import _tensor_identity
        from src.kernel_mean_ce import _cpu
        gate("helper_source", sha(crossfit.__file__) == packet["helper_entrypoint"]["sha256"]
            and Path(crossfit.__file__).resolve() == Path(packet["helper_entrypoint"]["path"]).resolve())
        original_helper=science["required_refs"]["original_composed_helper"]
        gate("original_composed_helper",sha(joint.__file__)==original_helper["sha256"]
            and Path(joint.__file__).resolve()==Path(original_helper["path"]).resolve())
        report["source_entry"] = implementation_provenance(); gate("source_entry", report["source_entry"] == packet["source"])
        torch.set_num_threads(4)
        if torch.get_num_interop_threads() != 1: torch.set_num_interop_threads(1)
        gate("CPU_only", not torch.cuda.is_initialized())
        fixture = science["fixture"]; dims = fixture["dimensions"]
        n,d,b,c,k,r = (dims[x] for x in ("N","D","B","C","K","rank")); F = d+b
        def tensor(v): return torch.tensor(v, dtype=torch.float64, device="cpu")
        z,q,anchors,mapping = (tensor(fixture[x]) for x in ("z","Q","anchors","mapping"))
        hard = torch.tensor(fixture["hard"], dtype=torch.long)
        U0,V0=tensor(fixture["U0"]),tensor(fixture["V0"])
        U,V=U0+tensor(fixture["U1_delta"]),V0+tensor(fixture["V1_delta"])
        W=1+d+c
        physical=torch.cat((torch.ones((n,1),dtype=torch.float64),z,q),1)
        even=(torch.arange(n)%2==0).to(torch.float64)[:,None]
        material=torch.cat((even*physical,(1-even)*physical),1)
        fold_indices={"A":torch.arange(0,n,2),"B":torch.arange(1,n,2)}
        scale=fixture["RMS"]["scale"]; oc=tensor(fixture["RMS"]["output_center"]); center=tensor(fixture["RMS"]["center"])
        penalty=fixture["penalty"]; mixing=fixture["mixing"]
        prior=torch.full((n,k), math.log(mixing/k), dtype=torch.float64)
        prior[torch.arange(n),hard]=math.log(1-mixing+mixing/k)
        saved["literal_inputs"]=_cpu(dict(z=z,Q=q,hard=hard,U0=U0,V0=V0,U1=U,V1=V,
            anchors=anchors,mapping=mapping,physical_material=physical,packed_material=material,prior=prior,fold_indices=fold_indices))
        gate("general_Q_nonunit",bool((q>0).all()) and bool((q.sum(1)-1).abs().max()>1e-2))
        gate("nonzero_factors",all(float(v.norm())>0 for v in (U0,V0,U,V)))
        def moments(u,v): return (prior+u@v.T/math.sqrt(r)).softmax(1).T@material/n
        M0,M=moments(U0,V0).detach(),moments(U,V).detach(); saved.update(M0=M0.clone(),M1=M.clone())
        gate("positive_masses_general_targets",all(bool((mp[:,0]>0).all()) and bool((mp[:,1+d:]>0).all()) for mp in (M[:,:W],M[:,W:])))
        bandwidth=anchors.square().sum(1).mean()/d; an=anchors.norm(dim=1)
        independent_maps=[]
        def independent_features(cen, role):
            check(); h=cen*scale+oc+center; hn=h.norm(dim=1)
            cosine=(h@anchors.T)/(hn[:,None]*an[None,:]); angle=cosine.acos()
            relu=(angle.sin()+(math.pi-angle)*angle.cos())/math.pi
            kernel=hn[:,None]*an[None,:]/(d*bandwidth)*relu
            features=torch.cat((cen,kernel@mapping),1)
            independent_maps.append(dict(role=role,input=h.detach().clone(),cosine=cosine.detach().clone(),relu=relu.detach().clone(),features=features.detach().clone()))
            saved["independent_maps"]=independent_maps
            report["work"]["independent_source_feature_evaluations" if role=="source" else "independent_endpoint_feature_evaluations"]+=1
            gate("smooth_map_"+role,bool((hn>1e-8).all()) and bool((an>1e-8).all())
                and bool((cosine.abs()<1-1e-4).all()) and bool((relu>-1+1e-4).all()) and bool((relu<1-1e-4).all()))
            return features
        source_features=independent_features(z,"source").detach(); source_X=torch.cat((source_features,torch.ones((n,1),dtype=torch.float64)),1)
        saved["source_features"]=source_features.clone()
        def inner(x,t,theta):
            lp=(x@theta.T).log_softmax(1)
            return -(t*lp).sum()/k + penalty*theta.square().sum()/2
        def decimal_inner(x_values,t_values,theta):
            # Only the scalar line-search objective uses fixed high precision.
            report["arithmetic_work"]["Decimal_scalar_objective_evaluations"]+=1
            with localcontext(Context(prec=80,rounding=ROUND_HALF_EVEN)):
                th=[[Decimal.from_float(float(v)) for v in row] for row in theta.tolist()]
                total=Decimal(0)
                for xi,ti in zip(x_values,t_values):
                    logits=[sum((a*b for a,b in zip(xi,tr)),Decimal(0)) for tr in th]
                    shift=max(logits)
                    lse=shift+sum(((value-shift).exp() for value in logits),Decimal(0)).ln()
                    total+=sum((target*(lse-value) for target,value in zip(ti,logits)),Decimal(0))
                ridge=sum((v*v for row in th for v in row),Decimal(0))
                return total/Decimal(k)+Decimal.from_float(float(penalty))*ridge/Decimal(2)
        def dense_terms(x,t,theta):
            p=(x@theta.T).softmax(1); s=t.sum(1)
            gradient=((s[:,None]*p-t).T@x)/k+penalty*theta
            cov=(torch.diag_embed(p)-p[:,:,None]*p[:,None,:])*(s/k)[:,None,None]
            xx=x[:,:,None]*x[:,None,:]
            H=torch.einsum("iab,ijl->ajbl",cov,xx).reshape(c*(F+1),c*(F+1))
            H=H+penalty*torch.eye(c*(F+1),dtype=torch.float64)
            report["arithmetic_work"]["dense_Hessians"]+=1
            return gradient,H
        heads=[]
        def head(mp,role):
            check(); report["attempts"]["stationary_heads"]+=1; report["work"]["stationary_head_attempts"]+=1
            mu=mp[:,0:1]; cen=mp[:,1:1+d]/mu; t=mp[:,1+d:]/mu
            theta=torch.zeros((c,F+1),dtype=torch.float64); iterations=0; trials=0
            row=dict(role=role,theta=theta.clone(),physical_moments=mp.detach().clone(),centers=cen.detach().clone(),targets=t.detach().clone(),features=None)
            heads.append(row);saved["heads"]=heads
            summary=dict(role=role,completed=False,head_gradient_abs_max=None,Newton_steps=0,Armijo_trials=0,
                inner_loss=None,last_theta_FP64=theta.tolist(),last_alpha=None,
                last_Decimal_current_value=None,last_Decimal_candidate_value=None,last_Decimal_Armijo_rhs=None)
            report["head_summaries"].append(summary)
            f=independent_features(cen,role);row["features"]=f.detach().clone()
            x=torch.cat((f,torch.ones((k,1),dtype=torch.float64)),1)
            x_values=[[Decimal.from_float(float(v)) for v in ri] for ri in x.tolist()]
            t_values=[[Decimal.from_float(float(v)) for v in ri] for ri in t.tolist()]
            for iteration in range(101):
                check(); gradient,H=dense_terms(x,t,theta); norm=float(gradient.abs().max())
                row.update(theta=theta.detach().clone(),gradient=gradient.detach().clone(),Hessian=H.detach().clone(),head_gradient_abs_max=norm,Newton_steps=iterations,Armijo_trials=trials)
                summary.update(head_gradient_abs_max=norm,Newton_steps=iterations,Armijo_trials=trials,last_theta_FP64=theta.tolist())
                if norm<=1e-12: break
                require(iteration<100,"Frozen independent Newton100 exhausted")
                direction=torch.linalg.solve(H,gradient.reshape(-1)).reshape_as(theta); descent=(gradient*direction).sum()
                require(bool(torch.isfinite(direction).all()) and float(descent)>0,"Independent Newton descent")
                value=decimal_inner(x_values,t_values,theta); alpha=1.
                summary["last_Decimal_current_value"]=str(value)
                for trial in range(60):
                    check(); trials+=1;report["arithmetic_work"]["Armijo_trials"]+=1
                    candidate=theta-alpha*direction
                    candidate_value=decimal_inner(x_values,t_values,candidate)
                    with localcontext(Context(prec=80,rounding=ROUND_HALF_EVEN)):
                        armijo_rhs=value-Decimal.from_float(1e-4)*Decimal.from_float(alpha)*Decimal.from_float(float(descent))
                    summary.update(Armijo_trials=trials,last_alpha=alpha,
                        last_Decimal_candidate_value=str(candidate_value),last_Decimal_Armijo_rhs=str(armijo_rhs))
                    row.update(last_candidate_theta=candidate.detach().clone(),last_Decimal_candidate_value=str(candidate_value),last_Decimal_Armijo_rhs=str(armijo_rhs))
                    if candidate_value<=armijo_rhs: break
                    alpha*=.5
                else: raise RuntimeError("Frozen Armijo60 exhausted")
                theta=candidate.detach();iterations+=1;report["arithmetic_work"]["Newton_steps"]+=1
            row.update(theta=theta.clone(),head_gradient_abs_max=norm,Newton_steps=iterations,Armijo_trials=trials,
                inner_loss=float(inner(x,t,theta)))
            summary.update({name:row[name] for name in ("role","head_gradient_abs_max","Newton_steps","Armijo_trials","inner_loss")})
            summary.update(completed=True,last_theta_FP64=theta.tolist())
            report["work"]["stationary_heads"]+=1
            gate("stationary_head_"+role,norm<=1e-12 and bool(torch.isfinite(theta).all()))
            return theta,x,t,H
        outer_records=[]
        def independent_outer(theta, held, role):
            indices=fold_indices[held]; sx=source_X.index_select(0,indices); qt=q.index_select(0,indices)
            lp=(sx@theta.T).log_softmax(1); raw=-(qt*lp).sum()/len(indices)
            rhs=((qt.sum(1,keepdim=True)*lp.exp()-qt).T@sx)/len(indices)
            outer_records.append(dict(role=role,held_fold=held,held_count=len(indices),raw_CE=raw.detach().clone(),raw_RHS=rhs.detach().clone()))
            saved["independent_held_outer"]=outer_records
            report["work"]["independent_held_outer_evaluations"]+=1
            return raw.detach(),rhs.detach()
        def pair(mp,role):
            records={}
            for fit,held,block in (("A","B",mp[:,:W]),("B","A",mp[:,W:])):
                th,x,t,H=head(block,role+"_"+fit)
                raw,rhs=independent_outer(th,held,role+"_"+fit)
                records[fit]=dict(theta=th,x=x,targets=t,H=H,raw_CE=raw,raw_RHS=rhs,held_fold=held)
            saved.setdefault("evaluated_pairs",{})[role]=records
            return records,.5*(records["A"]["raw_CE"]+records["B"]["raw_CE"])
        baseline,raw_CE0=pair(M0,"baseline")
        CE0=raw_CE0.detach().clone();saved["CE0"]=CE0.clone()
        gate("own_two_head_immutable_CE0",float(CE0)>0 and bool(torch.isfinite(CE0)))
        current,raw_CE=pair(M,"current")
        gate("odd_fixed_parity_means",tuple(fold_indices["A"].tolist())==(0,2,4,6)
            and tuple(fold_indices["B"].tolist())==(1,3,5) and len(fold_indices["A"])==4 and len(fold_indices["B"])==3)
        def compare(name,actual,reference):
            difference=(actual-reference).detach();absolute=float(difference.abs().max());den=float(reference.norm())
            relative=float(difference.norm())/den if den>0 else None
            detail=dict(abs_max=absolute,relative_l2=relative,reference_l2=den)
            gate(name,bool(torch.isfinite(actual).all()) and den>0 and absolute<=science["bounds"]["comparison_abs_max"]
                and relative is not None and relative<=science["bounds"]["comparison_relative_l2"],**detail)
        for role,records in (("baseline",baseline),("current",current)):
            for fit in ("A","B"):
                row=records[fit];report["attempts"]["public_held_outer_calls"]+=1
                value,rhs=crossfit.held_outer(source_features,q,row["theta"],row["held_fold"],stop=stop)
                saved.setdefault("public_held_outer",[]).append(dict(role=role,fit_fold=fit,raw_CE=value,raw_RHS=rhs.detach().clone()))
                report["work"]["public_held_outer_calls"]+=1
                compare("held_CE_"+role+"_"+fit,tensor(value),row["raw_CE"])
                compare("held_RHS_"+role+"_"+fit,rhs,row["raw_RHS"])
        rawGs=[];vectors={}
        for fit,mp in (("A",M[:,:W]),("B",M[:,W:])):
            row=current[fit];theta,x,t,H,rhs=(row[name] for name in ("theta","x","targets","H","raw_RHS"))
            report["attempts"]["adjoints"]+=1;report["work"]["adjoint_attempts"]+=1
            v=torch.linalg.solve(H,rhs.reshape(-1)).reshape_as(theta).detach()
            saved.setdefault("raw_adjoints",{})[fit]=dict(vector=v.clone(),raw_RHS=rhs.clone(),Hessian=H.clone())
            report["work"]["adjoints"]+=1;vectors[fit]=v
            gate("full_adjoint_residual_"+fit,float((H@v.reshape(-1)-rhs.reshape(-1)).abs().max())<=science["reference"]["adjoint_residual_abs_max"])
            p=(x@theta.T).softmax(1);s=t.sum(1,keepdim=True);err=s*p-t;a=x@v.T;pa=(p*a).sum(1,keepdim=True)
            RX=-(err@v+(s*p*(a-pa))@theta)/k;gamma=(a-pa)/k
            cen=mp[:,1:1+d]/mp[:,0:1];h=cen*scale+oc+center;hn=h.norm(dim=1)
            co=(h@anchors.T)/(hn[:,None]*an[None,:]);angle=co.acos()
            kernel_value=(angle.sin()+(math.pi-angle)*angle.cos())/math.pi;Cp=(math.pi-angle)/math.pi
            kernel_grad=an[None,:,None]/(d*bandwidth)*(h[:,None,:]/hn[:,None,None]*kernel_value[:,:,None]
                +Cp[:,:,None]*(anchors[None,:,:]/an[None,:,None]-co[:,:,None]*h[:,None,:]/hn[:,None,None]))
            mapped_rhs=RX[:,d:F]@mapping.T
            beta=RX[:,:d]+scale*(mapped_rhs[:,:,None]*kernel_grad).sum(1)
            mu=mp[:,0:1];rawG=torch.cat((-((beta*cen).sum(1,keepdim=True)+(gamma*t).sum(1,keepdim=True))/mu,beta/mu,gamma/mu),1).detach()
            rawGs.append(rawG);report["work"]["manual_composed_VJP_evaluations"]+=1
            saved.setdefault("manual_fold_VJP",{})[fit]=dict(R_X=RX.clone(),beta=beta.clone(),gamma=gamma.clone(),raw_G=rawG.clone())
        completeG=(.5*torch.cat(rawGs,1)/CE0).detach();saved["manual_complete_packed_G"]=completeG.clone()
        feature_map=NystromMap(anchors.clone(),mapping.clone(),"relu")
        transform=FeatureTransform(kind="rms",center=center.clone(),matrix=None,output_center=oc.clone(),scale=tensor(scale),eps=1e-12)
        layout=joint.ComposedJointLayout(d,b,c);composed=joint.OriginalRMSComposedCentroidFeatures(transform,feature_map,layout)
        def observe_map(callable_map,centers):
            value=callable_map(centers)
            saved.setdefault("production_map_returns",[]).append(composed.last_returned())
            report["work"]["production_G_map_calls"]+=1
            return value
        def observe_return(name,value): saved.setdefault("public_cotangent_returns",{})[name]=value.detach().clone()
        previous_profile=sys.getprofile()
        generic_path=str(Path(science["required_refs"]["original_nystrom_ce_source"]["path"]).resolve())
        raw_path=str(Path(original_helper["path"]).resolve())
        def profile(frame,event,arg):
            name,filename=frame.f_code.co_name,str(Path(frame.f_code.co_filename).resolve())
            for key,func,path in (("generic_moment_gradient_calls","moment_gradient",generic_path),
                                  ("public_raw_cotangent_calls","raw_moment_cotangent",raw_path)):
                if name==func and filename==path:
                    if event=="call":report["attempts"][key]+=1
                    elif event=="return" and arg is not None:report["work"][key]+=1
            if previous_profile is not None:previous_profile(frame,event,arg)
        try:
            sys.setprofile(profile);report["attempts"]["public_packed_cotangent_calls"]+=1
            actual=crossfit.complete_packed_cotangent(M,layout,composed,current["A"]["theta"],current["B"]["theta"],
                vectors["A"],vectors["B"],penalty,float(CE0),map_call=observe_map,observe_return=observe_return)
            saved["production_complete_packed_G"]=actual.detach().clone();report["work"]["public_packed_cotangent_calls"]+=1
        finally:sys.setprofile(previous_profile)
        gate("observation_hook_restored",sys.getprofile() is previous_profile)
        for j,fit in enumerate(("A","B")):
            returned=saved["public_cotangent_returns"]["raw_"+fit]
            for name,sl in (("mass",slice(0,1)),("z",slice(1,1+d)),("Q",slice(1+d,W)),("full",slice(None))):
                compare("raw_"+fit+"_"+name,returned[:,sl],rawGs[j][:,sl])
            for name,sl in (("mass",slice(j*W,j*W+1)),("z",slice(j*W+1,j*W+1+d)),("Q",slice(j*W+1+d,(j+1)*W))):
                compare("complete_"+fit+"_"+name,actual[:,sl],completeG[:,sl])
        compare("complete_full",actual,completeG)
        gate("returned_G_detached_FP64_packed",not actual.requires_grad and actual.grad_fn is None
            and actual.dtype==torch.float64 and tuple(actual.shape)==(k,2*W))
        probability=(prior+U@V.T/math.sqrt(r)).softmax(1);direction=material@completeG.T/n
        GP=probability*(direction-(probability*direction).sum(1,keepdim=True))
        manualU,manualV=(GP/math.sqrt(r))@V,(GP/math.sqrt(r)).T@U
        ug,vg=U.detach().requires_grad_(),V.detach().requires_grad_()
        factorM=moments(ug,vg);report["work"]["factor_autograd_moment_forward"]+=1
        saved["factor_autograd_moments"]=factorM.detach().clone()
        autoU,autoV=torch.autograd.grad((factorM*completeG).sum(),(ug,vg))
        saved.update(manual_U=manualU.detach().clone(),manual_V=manualV.detach().clone(),autograd_U=autoU.clone(),autograd_V=autoV.clone(),full_K_FP64_W=GP.detach().clone())
        native_material=crossfit.pack_physical_material(z,q)
        saved["native_packed_material"]=native_material.detach().clone();report["work"]["native_packed_material_builds"]+=1
        compare("packed_material",native_material,material)
        un,vn=U.detach().requires_grad_(),V.detach().requires_grad_()
        nativeM=LowRankMoments.apply(un,vn,hard,native_material,mixing,n)
        saved["LowRank_packed_moments"]=nativeM.detach().clone();report["work"]["LowRank_forward"]+=1
        (nativeM*completeG).sum().backward()
        nativeU,nativeV=un.grad.detach().clone(),vn.grad.detach().clone()
        saved.update(LowRank_U=nativeU,LowRank_V=nativeV);report["work"]["LowRank_backward"]+=1
        for name,sl in (("A",slice(0,W)),("B",slice(W,2*W)),("full",slice(None))):
            compare("autograd_moments_"+name,factorM.detach()[:,sl],M[:,sl])
            compare("LowRank_moments_"+name,nativeM.detach()[:,sl],M[:,sl])
        for method,du,dv in (("autograd",autoU,autoV),("LowRank",nativeU,nativeV)):
            compare(method+"_factor_U",du,manualU);compare(method+"_factor_V",dv,manualV)
            compare(method+"_factor_full",torch.cat((du.flatten(),dv.flatten())),torch.cat((manualU.flatten(),manualV.flatten())))
        eps=science["bounds"]["FD_epsilon"]
        for name in ("mass","z","Q","U","V"):
            endpoints=[]
            if name in ("mass","z","Q"):
                dm=torch.zeros_like(M);offset={"mass":(0,1),"z":(1,1+d),"Q":(1+d,W)}[name]
                for j,fit in enumerate(("A","B")):dm[:,j*W+offset[0]:j*W+offset[1]]=tensor(science["directions"][name][fit])
                analytic=float((completeG*dm).sum())
            else:
                fdirection=tensor(science["directions"][name]);analytic=float(((manualU if name=="U" else manualV)*fdirection).sum())
            for sign in (1,-1):
                mp=(M+sign*eps*dm) if name in ("mass","z","Q") else moments(U+sign*eps*fdirection,V) if name=="U" else moments(U,V+sign*eps*fdirection)
                saved.setdefault("FD_endpoint_inputs",[]).append(dict(name=name,sign=sign,moments=mp.detach().clone()))
                require(all(bool((part[:,0]>0).all()) and bool((part[:,1+d:]>0).all()) for part in (mp[:,:W],mp[:,W:])),"FD positive fold domain")
                records,raw=pair(mp,name+("_plus" if sign==1 else "_minus"));value=float(raw/CE0)
                endpoints.append(dict(sign=sign,raw_symmetric_CE=float(raw),objective=value));report["work"]["FD_endpoints"]+=1
            numerical=(endpoints[0]["objective"]-endpoints[1]["objective"])/(2*eps)
            absolute=abs(analytic-numerical);den=max(abs(analytic),abs(numerical));relative=absolute/den if den>0 else None
            row=dict(name=name,epsilon=eps,analytic=analytic,numerical=numerical,abs_error=absolute,relative_error=relative,endpoints=endpoints)
            report["FD_summaries"].append(row);report["work"]["FD_records"]+=1;saved["FD_summaries"]=list(report["FD_summaries"])
            gate("FD_BOTH_"+name,den>0 and absolute<=science["bounds"]["FD_abs_error"]
                and relative is not None and relative<=science["bounds"]["FD_relative_error"])
        gate("all_discrete_counts",report["work"]==science["counts"])
        gate("CE0_unchanged",torch.equal(CE0,saved["CE0"]))
        report["readonly_exit"]=pin_values();gate("exit_pins",report["readonly_exit"]==report["readonly_entry"])
        report["source_exit"]=implementation_provenance();gate("source_exit",report["source_exit"]==packet["source"])
        gate("still_CPU_only",not torch.cuda.is_initialized());check();report["completed"]=True;report["passed"]=True
    except BaseException as exc:
        report["failure"]=dict(type=type(exc).__name__,error=str(exc))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old_handler)
        if old_timer[0]>0: signal.setitimer(signal.ITIMER_REAL,*old_timer)
        if torch is not None:
            try:
                torch.save(dict(schema=1,kind=KIND,source=packet["source"],scientific_contract=sr,saved=saved),arrays)
                report["raw_evidence"]["write_completed"]=True
            except BaseException as exc:
                report["passed"]=False;report["completed"]=False;report["failure"]=report["failure"] or dict(type=type(exc).__name__,error=str(exc))
            try:
                report["raw_evidence"]["exists"]=arrays.exists()
                if report["raw_evidence"]["exists"]:
                    report["raw_evidence"]["bytes"]=arrays.stat().st_size
                    try:
                        report["raw_evidence"]["sha256"]=sha(arrays);report["raw_evidence"]["hash_unknown"]=False
                    except BaseException as exc:
                        report["raw_evidence"]["hash_error"]=dict(type=type(exc).__name__,error=str(exc));raise
            except BaseException as exc:
                report["passed"]=False;report["completed"]=False;report["failure"]=report["failure"] or dict(type=type(exc).__name__,error=str(exc))
        if not report["completed"]:
            try:
                report["readonly_exit"]=pin_values()
                require(report["readonly_exit"]==packet["readonly_files_sha256"],"Failure exit pins changed")
                report["source_exit"]=implementation_provenance()
                require(report["source_exit"]==packet["source"],"Failure exit source changed")
            except BaseException as exc: report["failure_exit_verification_unknown_or_failed"]=dict(type=type(exc).__name__,error=str(exc))
        report["resources"]=peaks()
        if not all(v<=science["resources"][k] for k,v in report["resources"].items()):
            report["passed"]=False;report["completed"]=False;report["failure"]=report["failure"] or dict(type="ResourceBoundary",error="Frozen final evidence boundary")
        report_path.write_text(report_json())
        report["resources"]=peaks()
        if not all(v<=science["resources"][k] for k,v in report["resources"].items()):
            report["passed"]=False;report["completed"]=False;report["failure"]=report["failure"] or dict(type="ResourceBoundary",error="Frozen post-report boundary")
            report_path.write_text(report_json())
    require(report["passed"],"Terminal FL CPU proof failure; preserve owning report/arrays, no retry or fixture search")
    return dict(passed=report["passed"],report=dict(path=str(report_path),sha256=sha(report_path)),
        arrays=report["raw_evidence"],work=report["work"])
