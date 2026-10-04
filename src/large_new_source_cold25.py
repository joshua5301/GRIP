"""Cold fixed25 critics on one accepted owning Flickr source; no students.

Only source IO, frozen factor provision and canonical endpoint export are new.
All stationary heads, moment derivatives, factor arithmetic and Adam are the
unchanged original implementations. A failed namespace is never resumed here.
"""
import hashlib
import json
import math
import os
import platform
import resource
import signal
import time
from pathlib import Path

import numpy as np
import torch

from src import kernel_mean_ce as mean, nystrom_ce as ny, soft_ce_partition as linear
from src.io import array_digest, cpu_state
from src.large_kernel_mean_source_capture import _matrix, _observed, _require, _sha
from src.low_rank_assignment import LowRankMoments
from src.moments import decode_moments, make_material
from src.shared_features import _tensor_identity

KIND = "large_new_source_cold25_three_critic_v1"
CONTRACT_SHA = "54863c90d936c734a258f6195c29e9da3806deca7faa7f325378abbc4c1edc71"
ARMS = ("meanPhi25", "uniform_linear25", "uniform_centroid_Ny25")
DIMS = dict(nodes=44625, dimension=500, classes=7, cells=44, rank=16, basis=512)


def _write(value, path):
    with Path(path).open("xb") as stream:
        torch.save(value, stream); stream.flush(); os.fsync(stream.fileno())


def run(protocol_path, protocol_sha256, budget, arm, stop=lambda: False):
    from src.research_loop import implementation_provenance

    _require(callable(stop) and _sha(protocol_path)==protocol_sha256, "Frozen cold invocation differs")
    packet=json.loads(Path(protocol_path).read_text()); ref=packet["scientific_contract"]
    _require(ref["sha256"]==CONTRACT_SHA and _sha(ref["path"])==CONTRACT_SHA, "Cold25 contract differs")
    contract=json.loads(Path(ref["path"]).read_text())
    _require(type(packet["schema"]) is int and packet["schema"]==1 and packet["kind"]==KIND
        and budget==packet["budget"]==contract["budget"]=="flickr44" and arm in ARMS
        and contract["arms"]==list(ARMS) and type(packet["steps"]) is int and packet["steps"]==25
        and packet["test_enabled"] is False and packet["dimensions"]==contract["dimensions"]==DIMS
        and packet["native_options"]==contract["native_options"]
        and packet["runtime"]==contract["runtime"] and packet["resource_limits"]==contract["resource_limits"]
        and packet["source"]==implementation_provenance(), "Frozen cold source/options/domain differs")
    def collect(value):
        if isinstance(value,dict):
            if "path" in value and "sha256" in value:
                _require(Path(value["path"]).is_absolute()
                    and packet["readonly_files_sha256"].get(value["path"])==value["sha256"], "Required admission/cache pin omitted")
            for child in value.values():collect(child)
        elif isinstance(value,list):
            for child in value:collect(child)
    collect(contract);collect(ref)
    _require(all(packet["readonly_files_sha256"].get(p)==s for p,s in contract["source_pins_sha256"].items()),
        "Original math source pins omitted")
    folder=Path(packet["output_folders"][arm]); trajectory=folder/"trajectory"
    _require(folder.is_absolute() and not folder.exists(), "Exclusive fresh cold namespace required")
    folder.mkdir(parents=True); started=time.monotonic(); raw={}; error=None; restored=None; gpu=False
    counts=dict(optimizer_call_attempts=0,optimizer_calls_completed=0,frozen_factor_copy_attempts=0,
        frozen_factor_copies_completed=0,canonical_export_forward_attempts=0,canonical_export_forwards=0,
        returned_Ny_progress_rows=0,native_RNG_factories=0,RMS_fits=0,Q_decodes=0,teacher_fits=0,
        map_fits=0,Phi_rebuilds=0,source_SGC_products=0,students=0,test_evaluations=0,old_proof_replays=0)
    report=dict(schema=1,kind=KIND,passed=False,budget=budget,arm=arm,source=packet["source"],
        scientific_contract=ref,protocol=dict(path=str(Path(protocol_path).resolve()),sha256=protocol_sha256),
        counts=counts,goal_complete=False,efficacy_qualified=False,test_enabled=False)
    limits=contract["resource_limits"]
    def guard():
        _require(not stop() and time.monotonic()-started<=300
            and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=limits["peak_RSS_bytes"], "Terminal stop/time/RSS boundary")
        if gpu:
            _require(torch.cuda.max_memory_allocated()<=limits["peak_allocated_bytes"]
                and torch.cuda.max_memory_reserved()<=limits["peak_reserved_bytes"], "Terminal GPU boundary")
        return False
    def pins(label):
        observed={};report[label]=observed
        for path,digest in {**packet["readonly_files_sha256"],**packet["source"]["files"]}.items():
            observed[path]=_sha(path,guard);_require(observed[path]==digest,"Readonly producer/input bytes changed: "+path)
        _require(implementation_provenance()==packet["source"], "Whole cold producer changed")
    def expire(signum,frame):raise InterruptedError("Frozen cold25 wall deadline reached")
    previous_handler=signal.getsignal(signal.SIGALRM);previous_timer=signal.getitimer(signal.ITIMER_REAL)
    try:
        signal.signal(signal.SIGALRM,expire);signal.setitimer(signal.ITIMER_REAL,
            min(300,previous_timer[0]) if previous_timer[0] else 300)
        runtime=contract["runtime"]
        _require(platform.python_version()==runtime["python"] and str(torch.__version__)==runtime["torch"]
            and np.__version__==runtime["numpy"] and str(torch.get_default_dtype())==runtime["default_dtype"]
            and not torch.is_autocast_enabled("cuda")
            and all(os.environ.get(k)==runtime[k] for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS","CUBLAS_WORKSPACE_CONFIG","CUDA_VISIBLE_DEVICES")), "Frozen native runtime differs")
        torch.set_num_threads(4)
        if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision("highest");torch.use_deterministic_algorithms(True)
        torch.cuda.init();_require(torch.cuda.get_device_name(0)==runtime["GPU"],"Native GPU differs")
        torch.cuda.reset_peak_memory_stats(0);gpu=True;device=torch.device("cuda:0");pins("entry_files_sha256")
        accepted=json.loads(Path(contract["capture_acceptance"]["path"]).read_text())
        capture=json.loads(Path(contract["capture_report"]["path"]).read_text())
        cp=json.loads(Path(contract["capture_protocol"]["path"]).read_text())
        _require(accepted["passed"] is True and capture["passed"] is True
            and accepted["source"]==capture["source"]==cp["source"]==contract["source_before"]
            and accepted["actual_report"]==contract["capture_report"]
            and accepted["actual_arrays"]==contract["capture_arrays"]
            and accepted["protocol"]==contract["capture_protocol"]
            and capture["new_data_digest"]==accepted["new_data_digest"]==contract["new_source_data_digest"]
            and capture["raw_evidence"]["path"]==contract["capture_arrays"]["path"]
            and capture["raw_evidence"]["sha256"]==contract["capture_arrays"]["sha256"], "Accepted source122 origin linkage differs")
        arrays=torch.load(contract["capture_arrays"]["path"],map_location="cpu",weights_only=False)
        _require(all(_tensor_identity(arrays[k])==d for k,d in capture["array_descriptors"].items())
            and capture["array_descriptors"]==accepted["asset_descriptors"], "Owning source/native arrays changed")
        _require({k:_tensor_identity(t) if torch.is_tensor(t) else t for k,t in arrays["transform"].items()}
            ==capture["transform_descriptors"]==accepted["transform_descriptors"], "Owning RMS metadata changed")
        z,q,hard=[arrays[k].detach().to(device).clone() for k in ("z","Q","hard")]
        h=arrays["saved_H_FP32"].detach().to(device).double()
        initial=[arrays[k].detach().to(device).clone() for k in ("U0","V0")]
        _require(all(_tensor_identity(t)==capture["array_descriptors"][k] for k,t in
            zip(("z","Q","hard","U0","V0"),[z,q,hard,*initial],strict=True)), "Owning CUDA source/native copy differs")
        digest=array_digest(z.cpu().numpy(),q.cpu().numpy(),hard.cpu().numpy())
        _require(digest==contract["new_source_data_digest"],"NEW coordinate data digest differs")
        raw["native_parameter_digests"]=[array_digest(arrays[k].numpy()) for k in ("U0","V0")]
        report.update(capture_source=accepted["source"],capture_arrays=contract["capture_arrays"],
            capture_acceptance=contract["capture_acceptance"],new_data_digest=digest,
            historical_dataset_digest=accepted["historical_cached_dataset_digest"])
        cache=contract["required_cache_refs"];sidecar=json.loads(Path(cache["Phi_sidecar"]["path"]).read_text())
        _require(sidecar==capture["original_Phi_identity"]==accepted["original_Phi_identity"],"Mandatory old Phi identity differs")
        phi=np.load(cache["Phi"]["path"],mmap_mode="r",allow_pickle=False)
        _require(phi.shape==(44625,512) and phi.dtype==np.float64 and phi.dtype.isnative
            and phi.flags.c_contiguous and not phi.flags.writeable,"Original read-only Phi header differs")
        options=dict(contract["native_options"]);context=None
        if arm=="meanPhi25":
            def origin(M,width):
                native_M=M.detach().to(device).clone()
                centers,labels,_=decode_moments(native_M,width)
                return dict(moments=_tensor_identity(M),centers=_tensor_identity(centers),labels=_tensor_identity(labels))
            assets=dict(H=capture["array_descriptors"]["saved_H_FP32"],transform=capture["transform_descriptors"],
                anchors=_tensor_identity(arrays["map_state"]["anchors"]),mapping=_tensor_identity(arrays["map_state"]["mapping"]),
                Phi_identity=sidecar,z=capture["array_descriptors"]["z"],Q=capture["array_descriptors"]["Q"],
                assignment=capture["array_descriptors"]["hard"])
            refs=dict(nodes=44625,cells=44,rank=16,dimension=500,classes=7,basis=512,chunk_size=2048,
                factor_seed=0,mixing=.05,data_digest=digest,device=str(device),runtime=mean._runtime(device),
                original_options=mean._native_options(options),current_source=packet["source"],
                optimizer_source_sha256=_sha(mean.__file__),files_sha256=packet["readonly_files_sha256"],
                asset_paths={k:cache[n]["path"] for k,n in (("H","H"),("map","map"),("Phi","Phi"),("Phi_metadata","Phi_sidecar"))},
                captured_origin=contract["capture_arrays"],capture_acceptance=contract["capture_acceptance"])
            context=dict(schema=1,mode=mean.MODE,policy=mean.POLICY,source_refs=refs,
                native_parameter_digests=raw["native_parameter_digests"],asset_descriptors=assets,
                native_origin=origin(arrays["RMS_M0"],500),kernel_origin=origin(arrays["Phi_M0"],512))
            report["context"]=context
        else:
            module=linear if arm=="uniform_linear25" else ny;factory=module.initialize_factors
            restored=(module,factory)
            def frozen_factory(assignment,cells,rank,seed=0):
                counts["frozen_factor_copy_attempts"]+=1
                _require(assignment is hard and type(cells) is int and cells==44 and type(rank) is int
                    and rank==16 and type(seed) is int and seed==0 and counts["frozen_factor_copy_attempts"]==1,
                    "Unexpected captured factor request")
                result=[p.detach().clone().requires_grad_() for p in initial]
                _require([array_digest(p.detach().cpu().numpy()) for p in result]==raw["native_parameter_digests"],
                    "Cold factory copy changed native bytes")
                counts["frozen_factor_copies_completed"]+=1;return result
            module.initialize_factors=frozen_factory
        guard();counts["optimizer_call_attempts"]+=1
        if arm=="meanPhi25":
            state=mean.optimize(z,q,hard,initial,phi,trajectory,25,options,context,
                checkpoint_steps=(0,1,25),resume_state=None,stop=guard)
            counts["optimizer_calls_completed"]+=1;raw["returned_terminal_parameters"]=cpu_state(state["parameters"])
            _require(type(state["step"]) is int and state["step"]==25 and state["context"]==context
                and all(state["work"][k]==v for k,v in contract["expected_mean_counts"].items())
                and [_tensor_identity(p) for p in state["parameters"]]
                    ==[_tensor_identity(p) for p in state["current"]["parameters"]], "Mean fixed25 work/state differs")
            history=state["history"];params=state["parameters"];p0=state["snapshots"][0]
            _require(p0["objective"]==1.0 and p0["CE0"]>0,"Mean own initial scale differs")
            report.update(actual_core_work=state["work"],actual_core_attempts=state["attempts"],
                CE0=state["CE0"],CE25=state["current"]["CE"],objective25=state["current"]["objective"],
                critic_convention=contract["critic_conventions"]["meanPhi25"])
        elif arm=="uniform_linear25":
            callopts=dict(options);callopts.pop("save_resume")
            returned=linear.optimize_ce_assignment(z,q,hard,steps=25,folder=trajectory,
                checkpoint_steps=(0,25),save_resume=True,stop=guard,**callopts)
            counts["optimizer_calls_completed"]+=1;raw["returned_linear_history"]=cpu_state(returned["history"])
            state=torch.load(trajectory/"resume.pt",map_location="cpu",weights_only=False)
            params=state["parameters"];raw["returned_terminal_parameters"]=cpu_state(params);history=state["history"]
            for k,v in options.items():
                if k=="save_resume":continue
                _require(state["config"].get(k,False if k in ("node_weighting","cache_assignment") else 1
                    if k=="cg_check_interval" else None)==v,"Original linear config differs: "+k)
            _require(state["config"]["data_digest"]==digest and all(row["J_exact"] and row["inner_converged"]
                for row in history) and all(row["cg_converged"] for row in history[:25])
                and _tensor_identity(state["snapshots"][0]["moments"])==capture["array_descriptors"]["RMS_M0"],
                "Original cold linear stationary/native origin differs")
            report.update(original_config=state["config"],CE0_scale=state["scale"],
                critic_convention=contract["critic_conventions"]["linear25"])
        else:
            m=arrays["map_state"]
            fmap=ny.NystromMap(m["anchors"].to(device),m["mapping"].to(device),m["kernel"])
            def progress(row):
                counts["returned_Ny_progress_rows"]+=1;raw["last_returned_Ny_progress"]=dict(row);guard()
            returned=ny.optimize(h,q,hard,fmap,phi,trajectory,25,penalty=1e-4,lr=.01,rank=16,
                seed=0,chunk=2048,stop=guard,progress=progress,checkpoint_every=25,mixing=.05,
                inner_loss_weighting="uniform",mass_mode="free")
            counts["optimizer_calls_completed"]+=1;report["returned_endpoint_path"]=str(returned)
            state=torch.load(trajectory/"resume.pt",map_location="cpu",weights_only=False)
            params=[state["u"],state["v"]];raw["returned_terminal_parameters"]=cpu_state(params);history=state["history"]
            expected=dict(steps_schema=2,penalty=1e-4,lr=.01,rank=16,seed=0,cells=44,chunk=2048,inner_loss_weighting="uniform")
            p0=torch.load(trajectory/"step_000000.pt",map_location="cpu",weights_only=False)
            _require(state["config"]==expected and counts["returned_Ny_progress_rows"]==26
                and Path(returned)==trajectory/"step_000025.pt" and all(math.isfinite(row["inner_grad"])
                    and row["inner_grad"]<=1e-7 for row in history)
                and _tensor_identity(p0["moments"])==capture["array_descriptors"]["raw_H_M0"], "Original uniform Ny completion/native origin differs")
            report.update(original_config=state["config"],input_fingerprint=state["input_fingerprint"],
                critic_convention=contract["critic_conventions"]["centroidNy25"],
                kernel_work_scope="Original unchanged API one source-row width probe plus current-centroid critic/Jacobian evaluations; no map fit/Phi rebuild. Kernel call interiors not instrumented.")
        guard();_require(type(state["step"]) is int and state["step"]==25 and [r["step"] for r in history]==list(range(26))
            and len(state["optimizer"]["state"])==2
            and all(float(s["step"])==25 for s in state["optimizer"]["state"].values()),"Cold25 history/Adam completion differs")
        if arm!="meanPhi25":
            _require(counts["frozen_factor_copies_completed"]==counts["frozen_factor_copy_attempts"]==1,
                "Cold captured factor provision differs")
            report["structural_completion_counts"]=dict(head_interfaces=26,moment_forwards=26,
                update_adjoints=25,moment_backwards=25,Adam_steps=25,P_updates=25,
                scope="Successful unchanged producer history/progress/slots; no solver spies or invented interior iteration counts")
        _matrix(params[0],(44625,16),torch.float32);_matrix(params[1],(44,16),torch.float32)
        u,v=[t.detach().to(device).clone() for t in params];guard()
        counts["canonical_export_forward_attempts"]+=1
        with torch.no_grad():M=LowRankMoments.apply(u,v,hard,make_material(h,q),.05,2048)
        counts["canonical_export_forwards"]+=1;raw["canonical_moments"]=cpu_state(M)
        x,y,mass=decode_moments(M,500)
        raw.update(X=cpu_state(x.float()),Qbar=cpu_state(y.float()),uniform_weights=cpu_state(mass.new_full((44,),1/44)))
        _matrix(raw["X"],(44,500),torch.float32);_matrix(raw["Qbar"],(44,7),torch.float32)
        _matrix(raw["uniform_weights"],(44,),torch.float64)
        _require(bool(torch.isfinite(M).all()) and bool((mass>0).all()) and bool((raw["Qbar"]>=0).all())
            and bool((raw["Qbar"].sum(1)>0).all()) and bool((raw["uniform_weights"]>0).all()),"Canonical endpoint invalid")
        report.update(complete_history_rows=26,actual_P_updates=25,owning_endpoint_factor_descriptors=[_tensor_identity(t) for t in params],
            canonical_descriptors={k:_tensor_identity(raw[k]) for k in ("canonical_moments","X","Qbar","uniform_weights")})
        raw.update(source=packet["source"],budget=budget,arm=arm,step=25,capture_arrays=contract["capture_arrays"],
            new_data_digest=digest);guard();report["passed"]=True
    except BaseException as exc:
        error=exc;report["error"]=dict(type=type(exc).__name__,message=str(exc))
    finally:
        if restored is not None:restored[0].initialize_factors=restored[1]
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous_handler)
        if previous_timer[0]>0:signal.setitimer(signal.ITIMER_REAL,max(1e-6,previous_timer[0]-(time.monotonic()-started)),previous_timer[1])
        path=folder/"canonical_endpoint25.pt";report["raw_evidence"]=dict(path=str(path),exists=False,complete=False,sha256=None)
        try:
            _write(raw,path);report["raw_evidence"].update(exists=True,bytes=path.stat().st_size,complete=True)
            report["raw_evidence"]["sha256"]=_sha(path,guard)
        except BaseException as exc:
            error=error or exc;report["passed"]=False
            report["raw_evidence"].update(exists=path.exists(),bytes=path.stat().st_size if path.exists() else 0,error=str(exc))
        report["trajectory_files_sha256"]={}
        try:
            for p in sorted(trajectory.rglob("*")):
                if p.is_file():report["trajectory_files_sha256"][str(p)]=_sha(p,guard)
            pins("exit_files_sha256");guard()
        except BaseException as exc:
            error=error or exc;report["passed"]=False;report["preservation_exit_error"]=dict(type=type(exc).__name__,message=str(exc))
        report.update(seconds=time.monotonic()-started,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            peak_allocated_bytes=torch.cuda.max_memory_allocated() if gpu else 0,
            peak_reserved_bytes=torch.cuda.max_memory_reserved() if gpu else 0)
        if report["seconds"]>300 or report["peak_RSS_bytes"]>limits["peak_RSS_bytes"] or report["peak_allocated_bytes"]>limits["peak_allocated_bytes"] or report["peak_reserved_bytes"]>limits["peak_reserved_bytes"]:
            error=error or RuntimeError("Final cold25 preservation resource bound failed");report["passed"]=False
        if error is not None and "error" not in report:report["error"]=dict(type=type(error).__name__,message=str(error))
        with (folder/"cold25_report.json").open("x") as stream:
            json.dump(_observed(report),stream,indent=2,allow_nan=False);stream.write("\n")
    if error is not None:raise RuntimeError("Terminal cold25 arm; preserve actual incomplete evidence, no retry") from error
    return report
