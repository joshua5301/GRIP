"""Unexecuted owning centered-metric calibration and derived native M0 gate."""
import hashlib
import json
import math
import os
import platform
import resource
import signal
import sys
import time
from pathlib import Path

KIND = "centered_trace_joint_CE_source_calibration_capture_v1"
SCIENCE_SHA = "52b519f9a380d26ef6cf030eefa54c41acb051afef3ccb30da015fa01536f544"
HELPER_SHA = "5e813f187d5c8720da6a0a40176d2fb16e07825d9bf478ed239e6d721750eebe"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path, check=lambda: None):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4194304), b""):
            value.update(block); check()
    return value.hexdigest()


def refs(value, found=None):
    found = {} if found is None else found
    if isinstance(value, dict):
        if set(value) == {"path", "sha256"}:
            path, digest = value["path"], value["sha256"]
            require(type(path) is str and Path(path).is_absolute() and str(Path(path).resolve()) == path, "Ref must be normalized ABS")
            require(type(digest) is str and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest), "Ref SHA differs")
            require(path not in found or found[path] == digest, "Conflicting ref")
            found[path] = digest
        for child in value.values(): refs(child, found)
    elif isinstance(value, list):
        for child in value: refs(child, found)
    return found


def run(protocol_path, protocol_sha256, budget, stop=lambda: False):
    started = time.monotonic()
    require(callable(stop) and Path(protocol_path).is_absolute() and str(Path(protocol_path).resolve()) == protocol_path
        and sha(protocol_path) == protocol_sha256, "Protocol path/SHA/stop differs")
    packet = json.loads(Path(protocol_path).read_text()); science_ref = packet["scientific_contract"]
    require(science_ref["sha256"] == SCIENCE_SHA and sha(science_ref["path"]) == SCIENCE_SHA, "Science differs")
    science = json.loads(Path(science_ref["path"]).read_text())
    require(type(packet["schema"]) is int and packet["schema"] == 1 and packet["kind"] == science["kind"] == KIND
        and budget in science["cases"], "Schema/kind/budget differs")
    case = science["cases"][budget]; out = Path(packet["output_folders"][budget])
    require(out.is_absolute() and str(out.resolve()) == str(out), "Output must be normalized ABS")
    out.mkdir(parents=True, exist_ok=False)
    arrays_path, report_path = out/"calibration_capture_arrays.pt", out/"calibration_capture_report.json"
    work = dict.fromkeys(science["exact_counts"], 0)
    saved, torch = {}, None
    report = dict(schema=1, kind=KIND, budget=budget, passed=False, completed=False, source=packet["source"],
        source135=case["source135"], scientific_contract=science_ref, protocol=dict(path=protocol_path,sha256=protocol_sha256),
        work=work, gates=[], failure=None, calibrated_origin_math_qualified=False, head_update_student_qualified=False,
        raw_evidence=dict(path=str(arrays_path),exists=False,bytes=None,sha256=None,hash_unknown=True,hash_error=None,write_completed=False))
    old_profile = sys.getprofile()
    old_handler, old_timer = signal.getsignal(signal.SIGALRM), signal.getitimer(signal.ITIMER_REAL)
    profile_work = dict(LowRank_forward=0,LowRank_backward=0,initialize_factors=0,make_material=0,decode_moments=0)
    def peaks():
        p = dict(seconds=time.monotonic()-started,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            peak_allocated_bytes=0,peak_reserved_bytes=0)
        if torch is not None and torch.cuda.is_initialized():
            p.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(0),peak_reserved_bytes=torch.cuda.max_memory_reserved(0))
        return p
    def check():
        require(not stop() and all(v <= science["resources"][k] for k,v in peaks().items()), "Frozen stop/resource boundary")
    def alarm(signum, frame): raise TimeoutError("Frozen300s calibration boundary")
    def gate(name, ok, **details):
        report["gates"].append(dict(name=name,passed=bool(ok),**details)); require(ok,name)
    def pins():
        expected = packet["readonly_files_sha256"]
        require(all(Path(p).is_absolute() and str(Path(p).resolve()) == p for p in expected), "Pins must be normalized ABS")
        require(all(expected.get(p) == h for p,h in refs(dict(packet=packet,science=science)).items()), "Required immutable ref omitted")
        return {p:sha(p,check) for p in expected}
    try:
        signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,max(1e-6,300-(time.monotonic()-started)))
        report["readonly_entry"] = pins();gate("readonly_entry_exact",report["readonly_entry"]==packet["readonly_files_sha256"])
        gate("entrypoint_exact",Path(packet["entrypoint"]["path"]).resolve()==Path(__file__).resolve())
        from src.research_loop import implementation_provenance
        report["source_entry"] = implementation_provenance()
        gate("current_source_Git_exact",report["source_entry"]==packet["source"])
        required = science["required_refs"]
        EW = json.loads(Path(required["EW_report"]["path"]).read_text())
        for name in ("EW_ROOT","EW_independent"):
            admission=json.loads(Path(required[name]["path"]).read_text())
            gate(name+"_actual_admitted",admission["passed"] is True and admission["actual_report"]==required["EW_report"])
        gate("EW_actual_inputs_exact",EW["passed"] is True and EW["completed"] is True
            and EW["budgets"][budget]["source135"]==case["source135"])
        original = json.loads(Path(case["source135"]["report"]["path"]).read_text())
        root = json.loads(Path(case["source135"]["ROOT"]["path"]).read_text())
        peer = json.loads(Path(case["source135"]["independent"]["path"]).read_text())
        gate("actual_own135_admitted",original["passed"] is True and root["passed"] is True and peer["passed"] is True
            and root["budgets"][budget]["report"]==peer["budgets"][budget]["actual_report"]==case["source135"]["report"]
            and root["budgets"][budget]["arrays"]==peer["budgets"][budget]["arrays"]==case["source135"]["arrays"]
            and original["source"]==original["source_entry"]==original["source_exit"]==case["source135_producer"]
            and original["scientific_contract"]==required["source135_science"]
            and Path(case["source135"]["arrays"]["path"]).stat().st_size==case["source135_array_bytes"])
        import numpy as np
        import torch
        from src import centered_trace_joint_mean_ce as metric_helper
        from src.dual_head_ce import _factor_digests
        from src.io import array_digest
        from src.joint_mean_ce import _runtime
        from src.kernel_mean_ce import _cpu, _seal
        from src.low_rank_assignment import LowRankMoments, initialize_factors
        from src.moments import decode_moments, make_material
        from src.nystrom_ce import _content_digest
        from src.shared_features import _tensor_identity
        gate("derived_helper_exact",Path(metric_helper.__file__).resolve()==Path(packet["helper_entrypoint"]["path"]).resolve()
            and sha(metric_helper.__file__)==packet["helper_entrypoint"]["sha256"]==HELPER_SHA)
        torch.set_num_threads(4)
        if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision("highest");torch.cuda.init();torch.cuda.reset_peak_memory_stats(0)
        device=torch.device("cuda:0");report["runtime"]=_runtime(device)
        gate("original_native_runtime",report["runtime"]==case["original_runtime"] and platform.python_version()=="3.12.7"
            and np.__version__=="1.26.4" and torch.get_num_interop_threads()==1
            and all(os.environ.get(k)=="4" for k in science["runtime"]["environment_threads"]))
        work["PT_load_attempts"]+=1
        payload=torch.load(case["source135"]["arrays"]["path"],map_location="cpu",weights_only=False);work["PT_loads"]+=1
        saved["source135_captured"]=_cpu(payload["captured"])
        gate("own135_header",payload["schema"]==1 and payload["kind"]==original["kind"] and payload["budget"]==budget
            and payload["source"]==case["source135_producer"] and payload["scientific_contract"]==required["source135_science"])
        x=payload["captured"];z=x["z_transform"]["z"];phi=x["original_Phi"];q=x["Q"];hard=x["hard"]
        dims=case["dimensions"];n,d,b,c,k,rank=(dims[name] for name in ("nodes","physical_dimension","original_Phi_basis","classes","cells","rank"))
        gate("raw135_tensor_domains",all(torch.is_tensor(v) and v.dtype==torch.float64 and v.device.type=="cpu"
            and not v.requires_grad and v.grad_fn is None and bool(torch.isfinite(v).all()) for v in (z,phi,q))
            and list(z.shape)==[n,d] and list(phi.shape)==[n,b] and list(q.shape)==[n,c]
            and hard.dtype==torch.int64 and hard.device.type=="cpu" and list(hard.shape)==[n]
            and int(hard.min())==0 and int(hard.max())==k-1 and len(torch.unique(hard))==k
            and bool((q>=0).all()) and bool((q.sum(1)>0).all()) and bool((q.sum(0)>0).all()))
        report["raw_identities"]={name:_tensor_identity(v) for name,v in (("z",z),("Phi",phi),("Q",q),("hard",hard))}
        gate("original_raw_identity_domains_exact",report["raw_identities"]["z"]==case["original_z_identity"]
            and report["raw_identities"]["Phi"]==case["original_Phi_tensor_identity"]
            and _content_digest(phi,canonical_double=True)==case["original_Phi_sidecar_identity"]["phi_digest"]
            and report["raw_identities"]["Q"]==case["original_assets"]["Q"] and report["raw_identities"]["hard"]==case["original_assets"]["assignment"]
            and array_digest(z.numpy(),q.numpy(),hard.numpy())==case["original_data_digest"])
        parameters=[x["native_factors"][name].detach().clone().to(device) for name in ("u","v")]
        gate("own_native_origin_factors",_factor_digests(parameters,n,k,rank,device)==case["native_parameter_digests"]
            and bool(parameters[0].eq(0).all()) and x["native_factors"]["factor_seed"]==0
            and x["native_factors"]["mixing"]==.05 and x["native_factors"]["data_digest"]==case["original_data_digest"])
        physical=dict(moments=x["physical_M0"],**x["native_physical"])
        gate("reused_original_physical_own_descriptors",all(_tensor_identity(v)=={field:case["native_physical_descriptors"][name][field]
            for field in ("shape","dtype","digest")} for name,v in physical.items()))
        functions={LowRankMoments.forward.__code__:"LowRank_forward",LowRankMoments.backward.__code__:"LowRank_backward",
            initialize_factors.__code__:"initialize_factors",make_material.__code__:"make_material",decode_moments.__code__:"decode_moments"}
        def profile(frame,event,arg):
            if event=="call" and frame.f_code in functions:profile_work[functions[frame.f_code]]+=1
        sys.setprofile(profile)
        with torch.no_grad():
            mean_phi=phi.mean(0);work["Phi_mean"]+=1;saved["mean_phi"]=_cpu(mean_phi)
            tau_z=float(z.square().sum(1).mean());tau_raw=float(phi.square().sum(1).mean());centered=phi-mean_phi
            tau_centered=float(centered.square().sum(1).mean());work["source_trace_reductions"]+=3
            saved["source_trace_scalars"]=dict(tau_z=tau_z,tau_Phi_raw=tau_raw,tau_Phi_centered=tau_centered)
            gate("finite_positive_source_traces",all(math.isfinite(v) and v>0 for v in saved["source_trace_scalars"].values()))
            total=tau_z+tau_raw;az=math.sqrt(total/(2*tau_z));aphi=math.sqrt(total/(2*tau_centered));work["coefficient_pair"]+=1
            scalars=dict(tau_z=tau_z,tau_Phi_raw=tau_raw,tau_Phi_centered=tau_centered,total=total,a_z=az,a_phi=aphi)
            saved["calibration_scalars"]=dict(values=scalars,scalar_hex={key:value.hex() for key,value in scalars.items()})
            report["calibration_scalars"]=dict(values={key:value if math.isfinite(value) else None for key,value in scalars.items()},scalar_hex=saved["calibration_scalars"]["scalar_hex"])
            gate("source_metric_inputs_equal_EW",dict(tau_z=tau_z,tau_Phi_raw=tau_raw,tau_Phi_centered=tau_centered)==case["observed_metric_inputs"])
            gate("finite_positive_calibration_domain",bool(torch.isfinite(mean_phi).all()) and all(math.isfinite(v) and v>0 for v in scalars.values()))
            layout=metric_helper.CenteredTraceJointLayout(d,b,c)
            metric=metric_helper.FrozenCenteredTraceMetric(mean_phi,tau_z,tau_raw,tau_centered,az,aphi)
            provider=metric_helper.ResidentCenteredTraceFeatures(z,phi,layout,metric,dict(z=case["original_z_identity"],Phi_identity=case["original_Phi_sidecar_identity"]))
            rows=torch.from_numpy(provider.outer_rows().copy());work["derived_rows"]+=1;saved["derived_rows"]=_cpu(rows)
            material_CPU=provider.material_on(q,"cpu");work["derived_material"]+=1;saved["derived_material"]=_cpu(material_CPU)
            report["metric_descriptor"],report["provider_descriptor"],report["layout"]=metric.descriptor(),provider.descriptor(),layout.descriptor()
            report["derived_identities"]=dict(rows=_tensor_identity(rows),material=_tensor_identity(material_CPU),mean_phi=_tensor_identity(mean_phi))
            gate("derived_material_shape_and_logical_cap",list(rows.shape)==[n,d+b] and list(material_CPU.shape)==[n,1+d+b+c]
                and n*(d+b+c)*8<=science["resources"]["logical_source_bytes"] and bool(torch.isfinite(material_CPU).all()))
            branch=dict(z=float(rows[:,:d].square().sum(1).mean()),Phi=float(rows[:,d:].square().sum(1).mean()))
            work["derived_trace_reductions"]+=2;report["derived_traces"]=branch
            target=total/2;report["trace_invariant_errors"]={name:dict(absolute=abs(value-target),relative=abs(value-target)/target) for name,value in branch.items()}
            report["trace_invariant_errors"]["total"]=dict(absolute=abs(sum(branch.values())-total),relative=abs(sum(branch.values())-total)/total)
            gate("balanced_branches_and_original_raw_total_BOTH",all(v["absolute"]<=1e-12 and v["relative"]<=1e-10 for v in report["trace_invariant_errors"].values()))
            material=material_CPU.to(device);work["joint_moment_forward_attempts"]+=1
            M=LowRankMoments.apply(*parameters,hard.to(device),material,.05,4096);work["joint_moment_forwards"]+=1
            saved["joint_M0"]=_cpu(M)
            centers,targets,mass=decode_moments(M,d+b);work["native_joint_decode"]+=1
            saved["native_joint"]=_cpu(dict(centers=centers,labels=targets,mass=mass))
            report["native_joint_descriptors"]={name:dict(_tensor_identity(v),device=str(v.device),requires_grad=v.requires_grad) for name,v in dict(moments=M,centers=centers,labels=targets,mass=mass).items()}
            sys.setprofile(old_profile)
            work["CPU_origin_reference_attempts"]+=1
            prior=torch.full((n,k),float(np.log(.05/k)),dtype=torch.float32)
            prior.scatter_(1,hard[:,None],float(np.log(1-.05+.05/k)))
            U,V=(p.detach().cpu() for p in parameters);logits=prior+U@V.T/math.sqrt(rank);probability=logits.double().softmax(1)
            reference=material_CPU.new_zeros(k,material_CPU.shape[1])
            for start in range(0,n,4096):reference+=probability[start:start+4096].T@material_CPU[start:start+4096]/n
            work["CPU_origin_reference"]+=1
            difference=M.detach().cpu()-reference
            saved["CPU_origin_reference"]=_cpu(dict(logits=logits,probability=probability,moments=reference,difference=difference))
            errors={};slices=dict(mass=slice(0,1),z=slice(1,1+d),Phi=slice(1+d,1+d+b),Q=slice(1+d+b,None),full=slice(None))
            for name,columns in slices.items():
                delta=difference[:,columns];ref=reference[:,columns];absolute=float(delta.abs().max());err=float(torch.linalg.vector_norm(delta));norm=float(torch.linalg.vector_norm(ref))
                relative=err/norm if norm>0 else (0.0 if err==0 else None)
                errors[name]=dict(absolute_max=absolute if math.isfinite(absolute) else None,difference_L2=err if math.isfinite(err) else None,reference_L2=norm if math.isfinite(norm) else None,relative_L2=relative if relative is not None and math.isfinite(relative) else None,
                    both_passed=math.isfinite(absolute) and absolute<=5e-7 and relative is not None and math.isfinite(relative) and relative<=2e-5)
            report["origin_errors"]=errors
            gate("finite_positive_native_origin",all(not v.requires_grad and v.grad_fn is None and bool(torch.isfinite(v).all()) for v in (M,centers,targets,mass,reference))
                and bool((mass>0).all()) and bool((targets>=0).all()) and bool((targets.sum(1)>0).all()) and bool((targets.sum(0)>0).all()))
            for name,error in errors.items():gate(name+"_origin_BOTH",error["both_passed"],**error)
        report["observed_profile_calls"]=profile_work
        gate("exact_original_profile",profile_work==dict(LowRank_forward=1,LowRank_backward=0,initialize_factors=0,make_material=1,decode_moments=1))
        gate("own_raw135_payload_unchanged",_seal(payload["captured"])==_seal(saved["source135_captured"]))
        gate("exact_capture_work",work==science["exact_counts"])
        torch.cuda.synchronize(device);check();report["completed"]=report["calibrated_origin_math_qualified"]=True
    except BaseException as error:
        report["failure"]=dict(type=type(error).__name__,message=str(error))
    finally:
        sys.setprofile(old_profile);signal.setitimer(signal.ITIMER_REAL,0)
        report["observed_profile_calls"]=profile_work
        try:
            if torch is not None:
                raw=report["raw_evidence"]
                try:
                    with arrays_path.open("xb") as stream:
                        torch.save(dict(schema=1,kind=KIND,budget=budget,source=packet["source"],scientific_contract=science_ref,saved=saved),stream);stream.flush();os.fsync(stream.fileno())
                    raw["write_completed"]=True
                except BaseException as error:report["raw_write_error"]=repr(error)
                try:
                    raw["exists"]=arrays_path.exists()
                    if raw["exists"]:
                        raw["bytes"]=arrays_path.stat().st_size
                        try:raw["sha256"]=sha(arrays_path);raw["hash_unknown"]=False
                        except BaseException as error:raw["hash_error"]=repr(error)
                except BaseException as error:report["raw_metadata_error"]=repr(error)
            try:
                report["readonly_exit"],report["source_exit"]=pins(),implementation_provenance()
                require(report["readonly_exit"]==packet["readonly_files_sha256"] and report["source_exit"]==packet["source"],"Exit source/pins changed")
            except BaseException as error:report["exit_verification_error"]=repr(error)
            report["resources"]=peaks();report["passed"]=bool(report["completed"] and report["failure"] is None
                and not any(key in report for key in ("raw_write_error","raw_metadata_error","exit_verification_error"))
                and report["raw_evidence"]["write_completed"] and not report["raw_evidence"]["hash_unknown"]
                and all(v<=science["resources"][key] for key,v in report["resources"].items()))
            with report_path.open("x") as stream:json.dump(report,stream,indent=2,allow_nan=False);stream.write("\n");stream.flush();os.fsync(stream.fileno())
            final=peaks()
            if report["passed"] and any(v>science["resources"][key] for key,v in final.items()):
                report.update(passed=False,resources=final,failure=dict(type="FinalSerializationResourceBoundary",message="Preserve namespace; no retry"))
                with report_path.open("w") as stream:json.dump(report,stream,indent=2,allow_nan=False);stream.write("\n");stream.flush();os.fsync(stream.fileno())
        finally:
            signal.signal(signal.SIGALRM,old_handler);signal.setitimer(signal.ITIMER_REAL,*old_timer)
    require(report["passed"],"Terminal calibration/origin failure; inspect owning evidence, no retry")
    return str(report_path)
