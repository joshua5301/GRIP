"""Fresh CPU-only single joint-mean CE algebra; no production numerical imports.

Authoring does not execute this file. ROOT binds and runs it once after review.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import signal
import time

KIND = "single_joint_mean_CE_fresh_CPU_qualification_v1"
CONTRACT_SHA = "5fc96bde3db7f929866af17027183d7151e5b7639bf56d6cfaf0105c65b5d099"


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _sha(path, check=lambda: None):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
            check()
    return h.hexdigest()


def _refs(value):
    found = {}
    def visit(x):
        if isinstance(x, dict):
            if "path" in x and "sha256" in x:
                p = Path(x["path"])
                _require(p.is_absolute() and str(p) == str(p.resolve()), "References must be normalized ABS paths")
                _require(type(x["sha256"]) is str and len(x["sha256"]) == 64, "Invalid reference SHA")
                _require(str(p) not in found or found[str(p)] == x["sha256"], "Conflicting reference")
                found[str(p)] = x["sha256"]
            for v in x.values():
                visit(v)
        elif isinstance(x, list):
            for v in x:
                visit(v)
    visit(value)
    return found


def _observed(x):
    if isinstance(x, float) and not math.isfinite(x):
        return {"nonfinite_observation": repr(x)}
    if isinstance(x, dict):
        return {str(k): _observed(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_observed(v) for v in x]
    return x


def run(protocol_path, protocol_sha256, stop=lambda: False):
    started = time.monotonic()
    _require(callable(stop), "Stop must be callable")
    path = Path(protocol_path)
    _require(path.is_absolute() and str(path) == str(path.resolve()), "Protocol must be normalized ABS")
    _require(_sha(path) == protocol_sha256, "Protocol SHA differs")
    packet = json.loads(path.read_text())
    _require(packet["schema"] == 1 and type(packet["schema"]) is int and packet["kind"] == KIND, "Protocol schema/kind differs")
    contract_ref = packet["scientific_contract"]
    _require(contract_ref["sha256"] == CONTRACT_SHA and _sha(contract_ref["path"]) == CONTRACT_SHA, "Frozen science differs")
    contract = json.loads(Path(contract_ref["path"]).read_text())
    folder = Path(packet["output_folder"])
    _require(folder.is_absolute() and str(folder) == str(folder.resolve()) and not folder.exists(), "Fresh normalized ABS namespace required")
    folder.mkdir(parents=True)
    arrays_path, report_path = folder / "qualification_arrays.pt", folder / "qualification_report.json"
    work = dict(randn_calls=0, head_attempts=0, stationary_heads=0, CE0_heads=0, current_heads=0,
        FD_endpoint_head_attempts=0, FD_endpoint_heads=0, adjoint_attempts=0, adjoint_solves=0,
        FD_record_attempts=0, FD_records=0, FD_endpoint_attempts=0, FD_endpoints=0,
        moment_attempts=0, moment_forwards=0, dense_Hessian_calls=0, Newton_linear_solves=0,
        line_search_trials=0, autograd_moment_cotangents=0, autograd_factor_cotangents=0,
        Adam_steps=0, P_optimizer_updates=0, production_calls=0, original_cache_payload_loads=0,
        source_SGC_products=0, GPU_calls=0, students=0, tests_on_real_data=0)
    arrays = {"fixture": {}, "oracle": {}, "head_states": {}, "FD_records": []}
    report = dict(schema=1, kind=KIND, passed=False, source=packet["source"],
        scientific_contract=contract_ref, protocol=dict(path=str(path), sha256=protocol_sha256),
        seed=510051, work=work, gates=[], scope="NEW synthetic CPU FP64 algebra only; not source/native/student qualification",
        raw_evidence=dict(path=str(arrays_path), exists=False, bytes=None, sha256=None, write_completed=False),
        failure=None)
    torch = None
    old_handler = signal.getsignal(signal.SIGALRM)
    old_timer = signal.getitimer(signal.ITIMER_REAL)
    def peaks():
        return dict(seconds=time.monotonic()-started, peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    def check():
        p = peaks()
        if stop() or p["seconds"] > 300 or p["peak_RSS_bytes"] > 16*1024**3:
            raise InterruptedError("Frozen CPU time/stop/RSS boundary reached")
    def alarm(signum, frame):
        raise TimeoutError("Frozen 300s CPU deadline reached")
    def pins():
        _require(all(Path(p).is_absolute() and str(Path(p)) == str(Path(p).resolve())
            for p in packet["readonly_files_sha256"]), "Readonly keys must be normalized ABS")
        refs = _refs(packet)
        _require(all(packet["readonly_files_sha256"].get(p) == h for p, h in refs.items()), "Required readonly admission omitted")
        return {p: _sha(p, check) for p in packet["readonly_files_sha256"]}
    def gate(name, passed, **details):
        report["gates"].append(dict(name=name, passed=bool(passed), **details))
    def metric(name, actual, expected, bounds):
        diff = float((actual-expected).abs().max())
        rel = diff / max(float(expected.abs().max()), 1e-12)
        gate(name, math.isfinite(diff) and math.isfinite(rel) and diff <= bounds["abs"]
            and rel <= bounds["rel"], max_abs=diff, relative_Linf=rel, bounds=bounds)
    try:
        signal.signal(signal.SIGALRM, alarm)
        signal.setitimer(signal.ITIMER_REAL, max(1e-6, 300-(time.monotonic()-started)))
        report["readonly_entry"] = pins()
        _require(report["readonly_entry"] == packet["readonly_files_sha256"], "Readonly bytes differ")
        _require(Path(packet["entrypoint"]["path"]).resolve() == Path(__file__).resolve(), "Entrypoint path differs")
        from src.research_loop import implementation_provenance  # orchestration only
        report["source_entry"] = implementation_provenance()
        _require(report["source_entry"] == packet["source"], "Current source differs")
        _require(platform.python_version() == "3.12.7" and all(os.environ.get(k) == "4"
            for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")), "Frozen CPU environment differs")
        import torch
        _require(str(torch.__version__) == "2.9.1+cu128", "Torch version differs")
        torch.set_num_threads(4)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        _require(torch.get_num_threads() == 4 and torch.get_num_interop_threads() == 1, "Thread counts differ")
        report["runtime"] = dict(python=platform.python_version(), torch=str(torch.__version__),
            threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(), device="cpu", dtype="torch.float64")
        def own(x):
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().clone()
            if isinstance(x, dict):
                return {k: own(v) for k, v in x.items()}
            if isinstance(x, (list, tuple)):
                return [own(v) for v in x]
            return x
        def keep(group, key, value):
            arrays[group][key] = own(value)
        generator = torch.Generator(device="cpu").manual_seed(510051)
        def draw(name, shape):
            value = torch.randn(shape, generator=generator, device="cpu", dtype=torch.float64)
            work["randn_calls"] += 1
            keep("fixture", name, value)
            check()
            return value
        z, phi = draw("z", (11,3)), draw("frozen_Phi", (11,5))
        logits_Q, V0 = draw("Q_logits", (11,3)), draw("V0", (4,2))
        U, V = .2*draw("current_U_draw", (11,2)), V0+.15*draw("current_V_draw", (4,2))
        directions = {factor: [] for factor in ("U", "V")}
        for factor, shape in (("U", (11,2)), ("V", (4,2))):
            for i in range(2):
                value = draw(f"{factor}_direction_{i}_draw", shape)
                directions[factor].append(value/value.norm())
        Q = logits_Q.softmax(1)*torch.linspace(.7,1.3,11,device="cpu",dtype=torch.float64)[:,None]
        U0 = torch.zeros((11,2),device="cpu",dtype=torch.float64)
        hard = torch.tensor([0,1,2,3,0,1,2,3,0,1,2],device="cpu",dtype=torch.int64)
        prior = torch.full((11,4),math.log(.05/4),device="cpu",dtype=torch.float64)
        prior.scatter_(1,hard[:,None],math.log(1-.05+.05/4))
        F = torch.cat((z,phi),1)
        material = torch.cat((torch.ones((11,1),device="cpu",dtype=torch.float64),F,Q),1)
        full = torch.cat((F,torch.ones((11,1),device="cpu",dtype=torch.float64)),1)
        arrays["fixture"].update(own(dict(Q=Q,U0=U0,U=U,V=V,hard=hard,prior=prior,F=F,
            material=material,full_features_bias_last=full,directions=directions)))
        _require(bool(torch.isfinite(Q).all()) and bool((Q>0).all()) and bool((Q.sum(1)>0).all())
            and float(Q.sum(1).min()) < 1 < float(Q.sum(1).max()), "General non-unit Q fixture invalid before heads")
        _require(work["randn_calls"] == 10, "Frozen RNG count differs")
        report["direction_norms"] = {k: [float(v.norm()) for v in vs] for k,vs in directions.items()}
        def moments(u,v):
            work["moment_attempts"] += 1
            p = (prior+u@v.T/math.sqrt(2)).softmax(1)
            m = p.T@material/11
            work["moment_forwards"] += 1
            return p,m
        def decode(m):
            return m[:,1:9]/m[:,0:1],m[:,9:12]/m[:,0:1]
        def inner(c,t,theta):
            a = torch.cat((c,torch.ones((4,1),device="cpu",dtype=torch.float64)),1)
            return -(t*(a@theta.T).log_softmax(1)).sum(1).mean()+.07/2*theta.square().sum()
        def outer(theta):
            return -(Q*(full@theta.T).log_softmax(1)).sum(1).mean()
        def solve(m,label,category):
            work["head_attempts"] += 1
            if category == "FD_endpoint_heads":
                work["FD_endpoint_head_attempts"] += 1
            c,t = decode(m)
            state = dict(label=label,category=category,converged=False,moments=own(m),
                centroids=own(c),targets=own(t),target_row_mass=own(t.sum(1)),iterations=[])
            arrays["head_states"][label] = state
            theta = torch.zeros((3,9),device="cpu",dtype=torch.float64)
            for iteration in range(100):
                check()
                theta = theta.detach().requires_grad_()
                loss = inner(c,t,theta)
                grad, = torch.autograd.grad(loss,theta)
                state.update(theta=own(theta),gradient=own(grad),inner_loss=float(loss),grad_Linf=float(grad.abs().max()))
                state["iterations"].append(dict(iteration=iteration,loss=float(loss),grad_Linf=state["grad_Linf"]))
                check()
                _require(bool(torch.isfinite(theta).all()) and bool(torch.isfinite(grad).all())
                    and math.isfinite(float(loss)), "Nonfinite independent head state")
                if state["grad_Linf"] <= 1e-12:
                    state["converged"] = True
                    work["stationary_heads"] += 1
                    work[category] += 1
                    return theta.detach()
                H = torch.autograd.functional.hessian(lambda f: inner(c,t,f.reshape(3,9)),theta.detach().reshape(-1))
                work["dense_Hessian_calls"] += 1
                state["Hessian"] = own(H)
                check()
                delta = torch.linalg.solve(H,grad.reshape(-1)).reshape(3,9)
                work["Newton_linear_solves"] += 1
                state["Newton_direction"] = own(delta)
                check()
                slope = float((grad*delta).sum())
                _require(math.isfinite(slope) and slope > 0, "Invalid dense Newton descent")
                for backtrack in range(60):
                    alpha = .5**backtrack
                    trial = theta.detach()-alpha*delta
                    value = inner(c,t,trial)
                    work["line_search_trials"] += 1
                    state.update(last_trial_theta=own(trial),last_trial_loss=float(value),last_trial_step=alpha)
                    check()
                    if math.isfinite(float(value)) and float(value) <= float(loss)-1e-4*alpha*slope:
                        theta = trial
                        break
                else:
                    raise ValueError("Frozen Armijo search exhausted")
            raise ValueError("Frozen Newton iteration budget exhausted")
        P0,M0 = moments(U0,V0)
        keep("oracle","P0",P0); keep("oracle","M0",M0); check()
        theta0 = solve(M0,"CE0_native_zero","CE0_heads")
        CE0 = float(outer(theta0)); keep("oracle","theta0",theta0); keep("oracle","CE0",CE0); check()
        _require(math.isfinite(CE0) and CE0 > 1e-12, "Own CE0 must be positive before normalization")
        P,M = moments(U,V); keep("oracle","current_P",P); keep("oracle","current_M",M); check()
        theta = solve(M,"current","current_heads")
        c,t = decode(M); th = theta.detach().requires_grad_()
        rhs, = torch.autograd.grad(outer(th),th)
        keep("oracle","theta",theta); keep("oracle","raw_outer_theta_cotangent",rhs)
        keep("oracle","CE_current",float(outer(theta))); check()
        work["adjoint_attempts"] += 1
        H = torch.autograd.functional.hessian(lambda f: inner(c,t,f.reshape(3,9)),theta.reshape(-1))
        work["dense_Hessian_calls"] += 1
        keep("oracle","current_Hessian",H); check()
        vector = torch.linalg.solve(H,rhs.reshape(-1)).reshape(3,9)
        work["adjoint_solves"] += 1
        keep("oracle","raw_CE_adjoint",vector); check()
        metric("raw_adjoint_residual",(H@vector.reshape(-1)).reshape(3,9),rhs,contract["gates"]["adjoint_residual"])
        a = torch.cat((c,torch.ones((4,1),device="cpu",dtype=torch.float64)),1)
        p = (a@theta.T).softmax(1); sigma = t.sum(1,keepdim=True)
        error = sigma*p-t; av = a@vector.T; centered_av = av-(p*av).sum(1,keepdim=True)
        gc = -(error@vector[:,:8]+(sigma*p*centered_av)@theta[:,:8])/4
        gt = centered_av/4
        gm = -(gc*c+gt*t).sum(1,keepdim=True)/M[:,0:1]
        raw_G = torch.cat((gm,gc/M[:,0:1],gt/M[:,0:1]),1)
        G = raw_G/CE0
        arrays["oracle"].update(own(dict(gc_raw_CE=gc,gt_raw_CE=gt,raw_moment_cotangent=raw_G,
            complete_moment_cotangent=G,centroids=c,targets=t,target_row_mass=sigma)))
        check()
        auto_M = M.detach().requires_grad_(); ac,at = decode(auto_M); ath = theta.detach().requires_grad_()
        hg, = torch.autograd.grad(inner(ac,at,ath),ath,create_graph=True)
        cross, = torch.autograd.grad((hg*vector.detach()).sum(),auto_M)
        auto_G = -cross/CE0; work["autograd_moment_cotangents"] += 1
        keep("oracle","autograd_complete_moment_cotangent",auto_G); check()
        for name,lo,hi in (("mass",0,1),("physical_features",1,4),("Phi_features",4,9),("targets",9,12)):
            metric("complete_M_"+name,G[:,lo:hi],auto_G[:,lo:hi],contract["gates"]["analytic_vs_autograd"])
        direction = material@G.T/11
        logit_cotangent = P*(direction-(P*direction).sum(1,keepdim=True))
        gU,gV = logit_cotangent@V/math.sqrt(2),logit_cotangent.T@U/math.sqrt(2)
        arrays["oracle"].update(own(dict(P_cotangent=direction,logit_cotangent=logit_cotangent,grad_U=gU,grad_V=gV)))
        check()
        au,avfactor = U.detach().requires_grad_(),V.detach().requires_grad_()
        _,am = moments(au,avfactor); ac,at = decode(am); ath = theta.detach().requires_grad_()
        hg, = torch.autograd.grad(inner(ac,at,ath),ath,create_graph=True)
        gu_auto,gv_auto = torch.autograd.grad((hg*vector.detach()).sum(),(au,avfactor))
        gu_auto,gv_auto = -gu_auto/CE0,-gv_auto/CE0
        work["autograd_factor_cotangents"] += 1
        keep("oracle","autograd_grad_U",gu_auto); keep("oracle","autograd_grad_V",gv_auto); check()
        for name,g,ag in (("U",gU,gu_auto),("V",gV,gv_auto)):
            metric("factor_"+name,g,ag,contract["gates"]["analytic_vs_autograd"])
            norm = float(g.norm())
            gate("current_"+name+"_gradient_nonzero",math.isfinite(norm) and norm > 1e-10,norm=norm,bound_gt=1e-10)
        for factor,grad in (("U",gU),("V",gV)):
            for index,d in enumerate(directions[factor]):
                for eps in (1e-4,5e-5):
                    work["FD_record_attempts"] += 1
                    item = dict(factor=factor,direction=index,epsilon=eps,direction_norm=float(d.norm()),endpoint_values={})
                    arrays["FD_records"].append(item)
                    for sign in (-1,1):
                        work["FD_endpoint_attempts"] += 1
                        eu,ev = (U+sign*eps*d,V) if factor == "U" else (U,V+sign*eps*d)
                        ep,em = moments(eu,ev)
                        label = f"FD_{factor}_{index}_{eps}_{sign}"
                        item.setdefault("endpoint_arrays",{})[str(sign)] = own(dict(U=eu,V=ev,P=ep,M=em))
                        check()
                        et = solve(em,label,"FD_endpoint_heads")
                        value = float(outer(et))/CE0
                        item["endpoint_values"][str(sign)] = value
                        item.setdefault("endpoint_theta",{})[str(sign)] = own(et)
                        work["FD_endpoints"] += 1
                        check()
                    fd = (item["endpoint_values"]["1"]-item["endpoint_values"]["-1"])/(2*eps)
                    analytic = float((grad*d).sum()); absolute = abs(fd-analytic)
                    relative = absolute/max(abs(fd),abs(analytic),1e-12)
                    item.update(analytic=analytic,finite_difference=fd,max_abs=absolute,relative=relative,complete=True)
                    work["FD_records"] += 1
                    b = contract["gates"]["central_FD"]
                    gate(f"FD_{factor}_{index}_{eps}",math.isfinite(absolute) and math.isfinite(relative)
                        and absolute <= b["abs"] and relative <= b["rel"],max_abs=absolute,relative=relative,bounds=b)
                    check()
        report["CE0"] = CE0
        report["CE_current"] = arrays["oracle"]["CE_current"]
        report["objective_current"] = report["CE_current"]/CE0
        for k,v in contract["exact_counts"].items():
            gate("exact_count_"+k,work[k] == v,actual=work[k],expected=v)
        for k,v in dict(head_attempts=18,FD_endpoint_head_attempts=16,adjoint_attempts=1,
                FD_record_attempts=8,FD_endpoint_attempts=16,moment_attempts=19,moment_forwards=19).items():
            gate("exact_attempt_or_moment_count_"+k,work[k] == v,actual=work[k],expected=v)
        gate("RNG_count",work["randn_calls"] == 10,actual=work["randn_calls"],expected=10)
        gate("all18_heads_stationary",len(arrays["head_states"]) == 18 and all(
            s["converged"] and s["grad_Linf"] <= 1e-12 for s in arrays["head_states"].values()))
        def finite(x):
            if isinstance(x,torch.Tensor):
                return x.device.type == "cpu" and bool(torch.isfinite(x).all())
            if isinstance(x,float):
                return math.isfinite(x)
            if isinstance(x,dict):
                return all(finite(v) for v in x.values())
            if isinstance(x,(list,tuple)):
                return all(finite(v) for v in x)
            return True
        gate("all_owning_fixture_oracle_head_FD_arrays_finite",finite(arrays))
        report["math_completed"] = True
    except BaseException as error:
        report["failure"] = dict(type=type(error).__name__,message=str(error))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        signal.signal(signal.SIGALRM,old_handler)
        try:
            if torch is not None:
                with arrays_path.open("xb") as f:
                    torch.save(arrays,f); f.flush(); os.fsync(f.fileno())
                report["raw_evidence"]["write_completed"] = True
            if arrays_path.exists():
                report["raw_evidence"].update(exists=True,bytes=arrays_path.stat().st_size,sha256=_sha(arrays_path))
            report["readonly_exit"] = pins()
            from src.research_loop import implementation_provenance
            report["source_exit"] = implementation_provenance()
            _require(report["readonly_exit"] == packet["readonly_files_sha256"]
                and report["source_exit"] == packet["source"], "Source/readonly exit bytes differ")
            check()
        except BaseException as error:
            report["preservation_or_exit_error"] = dict(type=type(error).__name__,message=str(error))
            if report["failure"] is None:
                report["failure"] = report["preservation_or_exit_error"]
        if arrays_path.exists():
            report["raw_evidence"].update(exists=True,bytes=arrays_path.stat().st_size)
        report["resources"] = peaks()
        if report["failure"] is None and any(not g["passed"] for g in report["gates"]):
            report["failure"] = dict(type="GateFailure",message="Frozen gates failed",
                gates=[g["name"] for g in report["gates"] if not g["passed"]])
        if report["failure"] is None and (report["resources"]["seconds"] > 300 or report["resources"]["peak_RSS_bytes"] > 16*1024**3):
            report["failure"] = dict(type="ResourceBoundary",message="Resource overrun during output preservation")
        report["passed"] = bool(report.get("math_completed") and report["failure"] is None
            and report["raw_evidence"]["write_completed"] and all(g["passed"] for g in report["gates"])
            and report["resources"]["seconds"] <= 300 and report["resources"]["peak_RSS_bytes"] <= 16*1024**3)
        with report_path.open("x") as f:
            json.dump(_observed(report),f,indent=2,allow_nan=False); f.write("\n"); f.flush(); os.fsync(f.fileno())
        final_resources = peaks()
        if report["passed"] and (final_resources["seconds"] > 300 or final_resources["peak_RSS_bytes"] > 16*1024**3):
            report.update(passed=False,resources=final_resources,failure=dict(type="ResourceBoundary",message="Resource overrun during final report preservation"))
            with report_path.open("w") as f:
                json.dump(_observed(report),f,indent=2,allow_nan=False); f.write("\n"); f.flush(); os.fsync(f.fileno())
        if old_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL,*old_timer)
    if not report["passed"]:
        raise RuntimeError("Fresh CPU algebra failed terminally; inspect owning qualification_report.json and partial arrays")
    return report
