"""Unexecuted EZ literal general-Q CPU complete-chain proof; no old oracle/fixture."""
import hashlib
import json
import math
import resource
import signal
import sys
import time
from pathlib import Path

KIND = "single_composed_centroid_joint_CE_fresh_generalQ_CPU_complete_chain_qualification_v1"
SCIENCE_SHA = "df4c8715ee1d5245bc7716676c27904d642468e79d678a09a05b92f113e4215f"


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
        public_complete_cotangent_calls=0, generic_moment_gradient_calls=0)
    report["arithmetic_work"] = dict(dense_Hessians=0, Newton_steps=0, Armijo_trials=0)
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
        import torch
        from src import composed_centroid_joint_ce as joint
        from src.nystrom_ce import NystromMap
        from src.transforms import FeatureTransform
        from src.research_loop import implementation_provenance
        from src.shared_features import _tensor_identity
        from src.kernel_mean_ce import _cpu
        gate("helper_source", sha(joint.__file__) == packet["helper_entrypoint"]["sha256"]
            and Path(joint.__file__).resolve() == Path(packet["helper_entrypoint"]["path"]).resolve())
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
        material=torch.cat((torch.ones((n,1),dtype=torch.float64),z,q),1)
        scale=fixture["RMS"]["scale"]; oc=tensor(fixture["RMS"]["output_center"]); center=tensor(fixture["RMS"]["center"])
        penalty=fixture["penalty"]; mixing=fixture["mixing"]
        prior=torch.full((n,k), math.log(mixing/k), dtype=torch.float64)
        prior[torch.arange(n),hard]=math.log(1-mixing+mixing/k)
        saved["literal_inputs"]=_cpu(dict(z=z,Q=q,hard=hard,U0=U0,V0=V0,U1=U,V1=V,
            anchors=anchors,mapping=mapping,material=material,prior=prior))
        gate("general_Q_nonunit",bool((q>0).all()) and bool((q.sum(1)-1).abs().max()>1e-2))
        gate("nonzero_factors",all(float(v.norm())>0 for v in (U0,V0,U,V)))
        def moments(u,v): return (prior+u@v.T/math.sqrt(r)).softmax(1).T@material/n
        M0,M=moments(U0,V0).detach(),moments(U,V).detach(); saved.update(M0=M0.clone(),M1=M.clone())
        gate("positive_masses_general_targets",bool((M[:,0]>0).all()) and bool((M[:,1+d:]>0).all()))
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
            f=independent_features(cen,role);row["features"]=f.detach().clone()
            x=torch.cat((f,torch.ones((k,1),dtype=torch.float64)),1)
            for iteration in range(101):
                check(); gradient,H=dense_terms(x,t,theta); norm=float(gradient.abs().max())
                row.update(theta=theta.detach().clone(),gradient=gradient.detach().clone(),Hessian=H.detach().clone(),head_gradient_abs_max=norm,Newton_steps=iterations,Armijo_trials=trials)
                if norm<=1e-12: break
                require(iteration<100,"Frozen independent Newton100 exhausted")
                direction=torch.linalg.solve(H,gradient.reshape(-1)).reshape_as(theta); descent=(gradient*direction).sum()
                require(bool(torch.isfinite(direction).all()) and float(descent)>0,"Independent Newton descent")
                value=inner(x,t,theta); alpha=1.
                for trial in range(60):
                    check(); trials+=1;report["arithmetic_work"]["Armijo_trials"]+=1
                    candidate=theta-alpha*direction
                    if float(inner(x,t,candidate))<=float(value-1e-4*alpha*descent): break
                    alpha*=.5
                else: raise RuntimeError("Frozen Armijo60 exhausted")
                theta=candidate.detach();iterations+=1;report["arithmetic_work"]["Newton_steps"]+=1
            row.update(theta=theta.clone(),head_gradient_abs_max=norm,Newton_steps=iterations,Armijo_trials=trials,
                inner_loss=float(inner(x,t,theta)))
            summary={name:row[name] for name in ("role","head_gradient_abs_max","Newton_steps","Armijo_trials","inner_loss")}
            report["head_summaries"].append(summary);report["work"]["stationary_heads"]+=1
            gate("stationary_head_"+role,norm<=1e-12 and bool(torch.isfinite(theta).all()))
            return theta,x,t,H
        theta0,x0,t0,H0=head(M0,"baseline")
        raw_CE0=-(q*(source_X@theta0.T).log_softmax(1)).sum()/n
        CE0=raw_CE0.detach().clone();saved["CE0"]=CE0.clone();gate("own_immutable_CE0",float(CE0)>0 and bool(torch.isfinite(CE0)))
        theta,x,t,H=head(M,"current")
        srcp=(source_X@theta.T).softmax(1); rawrhs=((q.sum(1,keepdim=True)*srcp-q).T@source_X)/n
        report["attempts"]["adjoints"]+=1;report["work"]["adjoint_attempts"]+=1
        v=torch.linalg.solve(H,rawrhs.reshape(-1)).reshape_as(theta).detach();report["work"]["adjoints"]+=1
        saved.update(theta=theta.clone(),raw_rhs=rawrhs.clone(),raw_adjoint=v.clone(),H=H.clone())
        gate("full_adjoint_residual",float((H@v.reshape(-1)-rawrhs.reshape(-1)).abs().max())<=1e-12)
        p=(x@theta.T).softmax(1);s=t.sum(1,keepdim=True);err=s*p-t;a=x@v.T;pa=(p*a).sum(1,keepdim=True)
        RX=-(err@v+(s*p*(a-pa))@theta)/k;gamma=(a-pa)/k
        cen=M[:,1:1+d]/M[:,0:1];h=cen*scale+oc+center;hn=h.norm(dim=1);co=(h@anchors.T)/(hn[:,None]*an[None,:]);angle=co.acos()
        C=(angle.sin()+(math.pi-angle)*angle.cos())/math.pi;Cp=(math.pi-angle)/math.pi
        kernel_grad=an[None,:,None]/(d*bandwidth)*(h[:,None,:]/hn[:,None,None]*C[:,:,None]
            +Cp[:,:,None]*(anchors[None,:,:]/an[None,:,None]-co[:,:,None]*h[:,None,:]/hn[:,None,None]))
        mapped_rhs=RX[:,d:F]@mapping.T
        beta=RX[:,:d]+scale*(mapped_rhs[:,:,None]*kernel_grad).sum(1)
        mu=M[:,0:1];rawG=torch.cat((-((beta*cen).sum(1,keepdim=True)+(gamma*t).sum(1,keepdim=True))/mu,beta/mu,gamma/mu),1)
        completeG=(rawG/CE0).detach();report["work"]["manual_composed_VJP_evaluations"]+=1
        saved.update(R_X=RX.clone(),beta=beta.clone(),gamma=gamma.clone(),manual_raw_G=rawG.detach().clone(),manual_complete_G=completeG.clone())
        feature_map=NystromMap(anchors.clone(),mapping.clone(),"relu")
        transform=FeatureTransform(kind="rms",center=center.clone(),matrix=None,output_center=oc.clone(),scale=tensor(scale),eps=1e-12)
        layout=joint.ComposedJointLayout(d,b,c);composed=joint.OriginalRMSComposedCentroidFeatures(transform,feature_map,layout)
        def observe_map(callable_map, centers):
            report["work"]["production_G_map_calls"]+=1
            value=callable_map(centers);saved["last_production_map_return"]=composed.last_returned();return value
        previous_profile=sys.getprofile()
        original_generic_path=str(Path(science["required_refs"]["original_nystrom_ce_source"]["path"]).resolve())
        def profile(frame,event,arg):
            if frame.f_code.co_name=="moment_gradient" and str(Path(frame.f_code.co_filename).resolve())==original_generic_path:
                if event=="call": report["attempts"]["generic_moment_gradient_calls"]+=1
                elif event=="return" and arg is not None: report["work"]["generic_moment_gradient_calls"]+=1
            if previous_profile is not None: previous_profile(frame,event,arg)
        try:
            sys.setprofile(profile)
            report["attempts"]["public_raw_cotangent_calls"]+=1
            actual_raw=joint.raw_moment_cotangent(M,layout,composed,theta.detach(),v,penalty,map_call=observe_map)
            report["work"]["public_raw_cotangent_calls"]+=1;saved["production_raw_G"]=actual_raw.detach().clone()
            report["attempts"]["public_complete_cotangent_calls"]+=1
            actual_complete=joint.complete_moment_cotangent(M,layout,composed,theta.detach(),v,penalty,float(CE0),map_call=observe_map)
            report["work"]["public_complete_cotangent_calls"]+=1;saved["production_complete_G"]=actual_complete.detach().clone()
        finally: sys.setprofile(previous_profile)
        gate("observation_hook_restored",sys.getprofile() is previous_profile)
        def compare(name,actual,reference):
            difference=(actual-reference).detach();absolute=float(difference.abs().max());den=float(reference.norm())
            relative=float(difference.norm())/den if den else (0. if float(difference.norm())==0 else None)
            detail=dict(abs_max=absolute,relative_l2=relative,reference_l2=den)
            gate(name,bool(torch.isfinite(actual).all()) and absolute<=1e-10 and relative is not None and relative<=1e-8,**detail)
        for kind,actual,reference in (("raw",actual_raw,rawG),("complete",actual_complete,completeG)):
            for name,sl in (("mass",slice(0,1)),("z",slice(1,1+d)),("Q",slice(1+d,None)),("full",slice(None))): compare(kind+"_"+name,actual[:,sl],reference[:,sl])
        gate("returned_G_detached_FP64_physical",all(not val.requires_grad and val.grad_fn is None and val.dtype==torch.float64
            and tuple(val.shape)==(k,1+d+c) for val in (actual_raw,actual_complete)))
        probability=(prior+U@V.T/math.sqrt(r)).softmax(1);direction=material@completeG.T/n
        GP=probability*(direction-(probability*direction).sum(1,keepdim=True))
        manualU,manualV=GP@V/math.sqrt(r),GP.T@U/math.sqrt(r)
        ug,vg=U.detach().requires_grad_(),V.detach().requires_grad_()
        factorM=moments(ug,vg);report["work"]["factor_autograd_moment_forward"]+=1
        autoU,autoV=torch.autograd.grad((factorM*completeG.detach()).sum(),(ug,vg))
        saved.update(manual_U=manualU.detach().clone(),manual_V=manualV.detach().clone(),autograd_U=autoU.clone(),autograd_V=autoV.clone())
        compare("factor_U",autoU,manualU);compare("factor_V",autoV,manualV)
        compare("factor_full",torch.cat((autoU.flatten(),autoV.flatten())),torch.cat((manualU.flatten(),manualV.flatten())))
        eps=1e-4
        for name in ("mass","z","Q","U","V"):
            direction=tensor(science["directions"][name]); endpoints=[]
            if name in ("mass","z","Q"):
                dm=torch.zeros_like(M);sl={"mass":slice(0,1),"z":slice(1,1+d),"Q":slice(1+d,None)}[name];dm[:,sl]=direction
                analytic=float((completeG*dm).sum())
            else: analytic=float(((manualU if name=="U" else manualV)*direction).sum())
            for sign in (1,-1):
                mp=(M+sign*eps*dm) if name in ("mass","z","Q") else moments(U+sign*eps*direction,V) if name=="U" else moments(U,V+sign*eps*direction)
                saved.setdefault("FD_endpoint_inputs",[]).append(dict(name=name,sign=sign,moments=mp.detach().clone()))
                require(bool((mp[:,0]>0).all()) and bool((mp[:,1+d:]>0).all()),"FD positive domain")
                th,fx,ft,fh=head(mp,name+("_plus" if sign==1 else "_minus"))
                raw=-(q*(source_X@th.T).log_softmax(1)).sum()/n;value=float(raw/CE0)
                endpoint=dict(sign=sign,raw_source_CE=float(raw),objective=value);endpoints.append(endpoint)
                report["work"]["FD_endpoints"]+=1
            numerical=(endpoints[0]["objective"]-endpoints[1]["objective"])/(2*eps)
            absolute=abs(analytic-numerical);den=max(abs(analytic),abs(numerical));relative=absolute/den if den else (0. if absolute==0 else None)
            row=dict(name=name,epsilon=eps,analytic=analytic,numerical=numerical,abs_error=absolute,relative_error=relative,endpoints=endpoints)
            report["FD_summaries"].append(row);report["work"]["FD_records"]+=1;saved["FD_summaries"]=list(report["FD_summaries"])
            gate("FD_BOTH_"+name,absolute<=1e-6 and relative is not None and relative<=.005)
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
    require(report["passed"],"Terminal EZ CPU proof failure; preserve owning report/arrays, no retry or fixture search")
    return dict(passed=report["passed"],report=dict(path=str(report_path),sha256=sha(report_path)),
        arrays=report["raw_evidence"],work=report["work"])
