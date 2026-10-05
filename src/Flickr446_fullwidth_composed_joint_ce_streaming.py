"""Separate full-width file-backed source/origin owner; no resident-source escape."""
import hashlib
import json
from pathlib import Path
from contextlib import contextmanager

import torch
from src import composed_centroid_joint_ce as joint
from src import fullwidth_coupled_row_backend as row
from src.dual_head_ce import _native_options
from src.kernel_mean_ce import _cpu, _plain, _seal
from src.shared_features import _tensor_identity

MODE = "Flickr446_original_fullwidth_filebacked_single_composed_uniform_CE_v1"
CONTEXT = "Flickr446_own_streaming_physical_origin_composed_context_v1"
LAYOUT = dict(N=44625,D=500,B=512,C=7,K=446,rank=16)
PARTITION = [2048]*21+[1617]


def require(ok, message):
    if not ok: raise ValueError(message)


def tensors(value):
    if torch.is_tensor(value): yield value
    elif isinstance(value, dict):
        for child in value.values(): yield from tensors(child)
    elif isinstance(value, (list,tuple)):
        for child in value: yield from tensors(child)


def storage_bytes(value, cpu_only=False):
    unique={}
    for v in tensors(value):
        if not cpu_only or v.device.type == "cpu":
            s=v.untyped_storage();unique[(str(v.device),s.data_ptr())]=s.nbytes()
    return sum(unique.values())


def copy_bytes(value):
    # _cpu clones every occurrence, including repeated aliases in owning records.
    return sum(v.numel()*v.element_size() for v in tensors(value))


class ActiveStorageBudget:
    """Conservative simultaneous charge before copies and FC's later account()."""
    def __init__(self, limit=1536*1024**2, check=lambda:None):
        self.limit,self.check=limit,check;self.retained={};self.peak=0;self.observations=[]
        self.reserve=64*1024**2;self.scratch=4*1024**2;self.solver_phases=[];self.active_phase=None

    def charge(self, role, live=None, future_cpu_bytes=0):
        cuda=torch.cuda.memory_allocated(0) if torch.cuda.is_initialized() else 0
        cpu=storage_bytes((self.retained,live),cpu_only=True)
        total=cuda+cpu+self.reserve+self.scratch+future_cpu_bytes
        self.peak=max(self.peak,total)
        if self.active_phase is not None:
            self.active_phase["CPU_copy_upper_bytes"]=max(self.active_phase["CPU_copy_upper_bytes"],cpu+future_cpu_bytes)
        self.observations.append(dict(role=role,CUDA_live_bytes=cuda,CPU_live_unique_bytes=cpu,
            future_CPU_copy_bytes=future_cpu_bytes,bounded_transient_reserve_bytes=self.reserve,
            hash_scratch_bytes=self.scratch,charged_bytes=total))
        require(total <= self.limit,"1536MiB simultaneous active-storage charge");self.check()
        return total

    @contextmanager
    def phase_reservation(self, role, future_cpu_bytes=2*7*1013*8):
        """Fixed selected allowance, not a private-library workspace theorem."""
        require(self.active_phase is None and self.reserve==64*1024**2,"No nested/adaptive solver reserve")
        event=dict(role=role,entry_attempted=True,entry_passed=False,returned=False,completed=False,
            reserve_bytes=768*1024**2,CPU_copy_upper_bytes=0,error=None,restored64=False)
        self.solver_phases.append(event);self.active_phase=event;failure=None
        try:
            event["preceding64_charge_bytes"]=self.charge("solver_phase_before64_"+role,future_cpu_bytes=future_cpu_bytes)
            event.update(allocated_before=torch.cuda.memory_allocated(0),max_allocated_before=torch.cuda.max_memory_allocated(0),
                max_reserved_before=torch.cuda.max_memory_reserved(0))
            self.reserve=768*1024**2
            event["precall_charge_bytes"]=self.charge("solver_phase_entry_"+role,future_cpu_bytes=future_cpu_bytes)
            event["entry_passed"]=True
            yield event
            event.update(max_allocated_after=torch.cuda.max_memory_allocated(0),max_reserved_after=torch.cuda.max_memory_reserved(0))
            event["managed_increment_upper_bytes"]=event["max_allocated_after"]-event["allocated_before"]
            event["managed_plus_opaque_bytes"]=event["managed_increment_upper_bytes"]+64*1024**2
            event["post_own_charge_bytes"]=self.charge("solver_phase_post_own_"+role)
            event["observed_active_upper_bytes"]=event["max_allocated_after"]+event["CPU_copy_upper_bytes"]+64*1024**2+self.scratch
            require(event["managed_plus_opaque_bytes"]<=768*1024**2,"Fixed768MiB managed solver peak+64MiB opaque")
            require(event["observed_active_upper_bytes"]<=self.limit,"1536MiB cumulative solver peak+liveCPU/copy+opaque/scratch")
            event["completed"]=True
        except BaseException as error:
            failure=error;event["error"]=dict(type=type(error).__name__,message=str(error));raise
        finally:
            if torch.cuda.is_initialized():
                event.update(max_allocated_after=torch.cuda.max_memory_allocated(0),max_reserved_after=torch.cuda.max_memory_reserved(0),
                    allocated_after=torch.cuda.memory_allocated(0))
                if "allocated_before" in event:
                    event["managed_increment_upper_bytes"]=event["max_allocated_after"]-event["allocated_before"]
                    event["managed_plus_opaque_bytes"]=event["managed_increment_upper_bytes"]+64*1024**2
                    event["observed_active_upper_bytes"]=event["max_allocated_after"]+event["CPU_copy_upper_bytes"]+64*1024**2+self.scratch
            self.reserve=64*1024**2;self.active_phase=None;event["restored64"]=True
            try:event["restore_charge_bytes"]=self.charge("solver_phase_restore64_"+role)
            except BaseException as error:
                event["restore_error"]=dict(type=type(error).__name__,message=str(error))
                if failure is None:raise

    def own(self, key, value):
        self.charge("before_own_"+key,value,2*copy_bytes(value))
        result=_cpu(value);self.retained[key]=result
        self.charge("after_own_"+key,result);return result

    def write(self, value, path, exclusive=False):
        path=Path(path);temporary=path if exclusive else path.with_suffix(path.suffix+".tmp")
        evidence=dict(path=str(path),exists=False,bytes=None,sha256=None,hash_unknown=True,write_completed=False,
            temporary_path=str(temporary),temporary_exists=False,temporary_bytes=None,temporary_hash_unknown=True)
        self.last_write=evidence
        self.charge("before_PT_write",value,storage_bytes(value))
        try:
            with temporary.open("xb") as stream:
                torch.save(value,stream);stream.flush()
                import os
                os.fsync(stream.fileno())
            if not exclusive: temporary.replace(path)
            evidence["write_completed"]=True
        finally:
            evidence["exists"]=path.exists()
            if path.exists(): evidence["bytes"]=path.stat().st_size
            evidence["temporary_exists"]=temporary.exists()
            if temporary.exists():evidence["temporary_bytes"]=temporary.stat().st_size
            self.last_write=evidence
        try:
            evidence["sha256"]=row.file_sha(path);evidence["hash_unknown"]=False
        except BaseException as error:
            evidence["hash_error"]=repr(error);raise
        self.charge("after_PT_write",value);return evidence


class ObservedComponent(row.ImmutableRowComponent):
    def __init__(self,*args,budget,**kwargs):
        self.budget=budget;budget.charge("component_constructor")
        super().__init__(*args,**kwargs)
        budget.charge("component_verified")

    def read(self,start,end,device):
        self.budget.charge("before_component_read",future_cpu_bytes=(end-start)*self.shape[1]*self.dtype.itemsize)
        value=super().read(start,end,device)
        if self.shape[1]==7:
            require(bool((value>=0).all()) and bool((value.sum(1)>0).all()),"Raw general-Q row domain")
        self.budget.charge("after_component_read",value);return value


class ObservedSource(row.CoupledOriginalSourceTiles):
    def __init__(self,*args,budget,witness_folder,**kwargs):
        super().__init__(*args,**kwargs);self.budget=budget;self.phase="unassigned"
        self.witness_folder=Path(witness_folder);self.witness_folder.mkdir(exist_ok=False)
        self.witness_records=[];self.witness_attempts=0;self.witness_returns=0

    def account(self,role,*values):
        self.budget.charge("FC_"+role,values);super().account(role,*values)

    def witness(self,role,start,end,**values):
        # Original operators call this before account(): observe/copy/write here.
        self.witness_attempts+=1;label=f"{self.witness_attempts:06d}_{self.phase}_{role}_{start:06d}_{end:06d}"
        record=dict(index=self.witness_attempts,phase=self.phase,role=role,start=start,end=end,
            path=str(self.witness_folder/(label+".pt")),exists=False,bytes=None,sha256=None,hash_unknown=True)
        self.witness_records.append(record)
        self.budget.charge("before_witness_copy_"+role,values,2*copy_bytes(values))
        own=_cpu(values)
        self.budget.charge("witness_copy_live_"+role,(values,own),storage_bytes(own))
        try:
            record.update(self.budget.write(own,record["path"],exclusive=True));self.witness_returns+=1
        finally:
            p=Path(record["path"]);record["exists"]=p.exists()
            if p.exists():record["bytes"]=p.stat().st_size
            if hasattr(self.budget,"last_write") and self.budget.last_write["path"]==record["path"]:
                record.update(self.budget.last_write)
            del own
        self.budget.charge("after_witness_"+role,values)


def build_streaming_inputs(component_manifest,actual_admissions,captured_small_native,RMS_map,layout,partition,current_source,options,
        *,budget,witness_folder,source_refs):
    require(type(layout) is row.CoupledTileLayout and vars(layout)==LAYOUT and list(partition)==PARTITION,"Typed full original layout/partition")
    _native_options(options)
    require(options["assignment_rank"]==16 and options["factor_seed"]==0 and options["penalty"]==1e-4
        and options["lr"]==.01 and options["chunk_size"]==options["outer_chunk_size"]==2048,"Original selected native settings")
    require(set(captured_small_native)=={"hard","U0","V0"},"Only selected small native buffers")
    n,k,r=layout.N,layout.K,layout.rank
    for key,shape,dtype in (("hard",(n,),torch.int64),("U0",(n,r),torch.float32),("V0",(k,r),torch.float32)):
        v=captured_small_native[key]
        require(v.shape==shape and v.dtype==dtype and not v.requires_grad and bool(torch.isfinite(v).all()),"Small native domain")
    hard=captured_small_native["hard"]
    require(int(hard.min())==0 and int(hard.max())==445 and len(hard.unique())==446
        and bool(captured_small_native["U0"].eq(0).all()),"Exact original hard coverage/zero U0")
    entries={key:component_manifest["components"][key] for key in ("z","Q")}
    old=component_manifest["original_Phi"]
    entries["Phi"]=dict(file=old["file"],NumPy_descriptor=dict(shape=[n,512],dtype="float64",digest=old["identity"]["phi_digest"]))
    parts={}
    for key,e in entries.items():
        d=e["NumPy_descriptor"]
        parts[key]=ObservedComponent(e["file"]["path"],d["shape"],d["dtype"],d["digest"],
            component_manifest["exporter_source"] if key!="Phi" else component_manifest["source122"]["source"],2048,budget=budget)
        require(parts[key].file_digest==e["file"]["sha256"],"Exact admitted component whole bytes")
    owner=ObservedSource(parts["z"],parts["Q"],parts["Phi"],layout,partition,source_refs,
        budget=budget,witness_folder=witness_folder)
    transform,feature_map=RMS_map
    composed=joint.OriginalRMSComposedCentroidFeatures(transform,feature_map,joint.ComposedJointLayout(500,512,7))
    budget.retained["composed_owned_assets"]=(composed._transform.state_dict(),composed._map.anchors,composed._map.mapping)
    contract=dict(schema=1,kind=MODE,layout=layout.descriptor(),partition=list(partition),source_owner=owner.descriptor(),
        composed_owner=composed.descriptor(),native_descriptors={key:_tensor_identity(v) for key,v in captured_small_native.items()},
        actual_admissions=actual_admissions,current_source=current_source,original_options=options,
        original_Torch_components={key:component_manifest["components"][key]["original_Torch_descriptor"] for key in ("z","Q")},
        original_Phi_identity=old["identity"],source122_data_digest=source_refs["source122_data_digest"],
        own_native_data_digest=source_refs["own_native_data_digest"],
        runtime=dict(joint._runtime(hard.device),interop_threads=torch.get_num_interop_threads(),GPU=torch.cuda.get_device_name(0)))
    contract["input_digest"]=_seal(contract);validate_input_contract(contract)
    return contract,owner,composed


def validate_input_contract(contract):
    require(contract["schema"]==1 and contract["kind"]==MODE and contract["input_digest"]==_seal({k:v for k,v in contract.items() if k!="input_digest"})
        and contract["layout"]==row.CoupledTileLayout(**LAYOUT).descriptor() and contract["partition"]==PARTITION,"Streaming input seal/domain")
    _native_options(contract["original_options"]);return True


def build_streaming_context(source_contract,new_M0,new_native_quotients):
    validate_input_contract(source_contract);centers,targets,mass=new_native_quotients
    require(new_M0.shape==(446,508) and new_M0.dtype==torch.float64
        and centers.shape==(446,500) and targets.shape==(446,7) and mass.shape==(446,),"New physical origin dimensions")
    require(all(bool(torch.isfinite(v).all()) for v in (new_M0,centers,targets,mass)) and bool((mass>0).all())
        and bool((targets>=0).all()) and bool((targets.sum(1)>0).all()) and bool((targets.sum(0)>0).all()),"New native general-Q origin")
    context=dict(schema=1,kind=CONTEXT,input_contract=source_contract,
        native_origin=dict(moments=_tensor_identity(new_M0),centers=_tensor_identity(centers),targets=_tensor_identity(targets),mass=_tensor_identity(mass)))
    context["context_digest"]=_seal(context);return context


def validate_context(context,input_contract=None):
    require(set(context)=={"schema","kind","input_contract","native_origin","context_digest"} and context["schema"]==1
        and context["kind"]==CONTEXT and context["context_digest"]==_seal({k:v for k,v in context.items() if k!="context_digest"}),"New streaming context seal")
    validate_input_contract(context["input_contract"])
    if input_contract is not None:require(_seal(input_contract)==_seal(context["input_contract"]),"Streaming source/options changed")
    require(set(context["native_origin"])=={"moments","centers","targets","mass"},"Own streaming origin required")
    return joint.ComposedJointLayout(500,512,7)
