"""Bounded immutable row operators; no full-source/P array conversion API."""
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

KIND = "fullwidth_coupled_immutable_row_backend_v1"


def require(ok, message):
    if not ok: raise ValueError(message)


def seal(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4194304), b""): h.update(block)
    return h.hexdigest()


def tensor_bytes(value):
    return hashlib.sha256(memoryview(value.detach().contiguous().cpu().numpy()).cast("B")).hexdigest()


@dataclass(frozen=True)
class CoupledTileLayout:
    N: int
    D: int
    B: int
    C: int
    K: int
    rank: int

    def __post_init__(self):
        require(all(type(v) is int and v > 0 for v in vars(self).values()) and self.K >= 2, "Typed positive layout")

    @property
    def outer_width(self): return self.D + self.B

    @property
    def material_width(self): return 1 + self.D + self.C

    def descriptor(self): return dict(**vars(self), outer_width=self.outer_width, material_width=self.material_width)


class ImmutableRowComponent:
    def __init__(self, path, shape, dtype, content_digest, producer, active_row_limit):
        p = Path(path)
        require(p.is_absolute() and str(p.resolve()) == str(p), "Normalized absolute component path")
        require(len(shape) == 2 and all(type(v) is int and v > 0 for v in shape), "Component shape")
        require(type(active_row_limit) is int and 0 < active_row_limit <= shape[0], "Active row limit")
        require(type(content_digest) is str and len(content_digest) == 64, "Content digest")
        self.path, self.shape, self.dtype = str(p), tuple(shape), np.dtype(dtype)
        require(self.dtype in (np.dtype("float32"), np.dtype("float64")), "Original floating component dtype")
        self.content_digest, self.producer, self.active_row_limit = content_digest, dict(producer), active_row_limit
        self._mapped_rows = np.load(p, mmap_mode="r", allow_pickle=False)
        require(isinstance(self._mapped_rows, np.memmap) and self._mapped_rows.shape == self.shape and self._mapped_rows.dtype == self.dtype
                and not self._mapped_rows.flags.writeable and self._mapped_rows.flags.c_contiguous, "Read-only C-order component header")
        self.work = dict(read_attempts=0, reads=0, rows_read=0, verification_attempts=0, verifications=0, hash_rows=0)
        self.staging = dict(peak_read_staging_bytes=0, peak_hash_boolean_scratch_bytes=0, file_hash_buffer_bytes=4194304,
            scope="Per-read owning NumPy source block plus nonalias converted tensor; source block freed on return. Hash finite boolean scratch is bounded; mapping is virtual, process RSS measured separately.")
        self.file_digest = file_sha(p)
        self.verify()

    def __array__(self, *args, **kwargs):
        raise TypeError("Whole component conversion is prohibited; request a bounded row block")

    def verify(self):
        self.work["verification_attempts"] += 1
        # Exact original nystrom_ce._content_digest NumPy domain/header/order.
        h = hashlib.sha256(json.dumps(dict(shape=self.shape, dtype=str(self.dtype)), sort_keys=True).encode())
        for start in range(0, self.shape[0], self.active_row_limit):
            block = np.ascontiguousarray(self._mapped_rows[start:start+self.active_row_limit])
            self.staging["peak_hash_boolean_scratch_bytes"] = max(self.staging["peak_hash_boolean_scratch_bytes"], int(block.size))
            require(np.isfinite(block).all(), "Finite immutable component rows")
            h.update(memoryview(block).cast("B")); self.work["hash_rows"] += len(block)
        require(h.hexdigest() == self.content_digest and file_sha(self.path) == self.file_digest, "Immutable file/content identity")
        self.work["verifications"] += 1
        return self.descriptor()

    def descriptor(self):
        return dict(kind="immutable_filebacked_component_v1", path=self.path, shape=list(self.shape), dtype=str(self.dtype),
                    content_digest=self.content_digest, file_sha256=self.file_digest, producer=self.producer,
                    active_row_limit=self.active_row_limit, mapped_logical_bytes=int(self._mapped_rows.nbytes), writeable=False)

    def read(self, start, end, device):
        self.work["read_attempts"] += 1
        require(type(start) is int and type(end) is int and 0 <= start < end <= self.shape[0]
                and end-start <= self.active_row_limit, "Bounded component slice")
        # Own only this block; no np.asarray(component), rows.copy(), torch.load, or full-source clone.
        block = np.array(self._mapped_rows[start:end], copy=True, order="C")
        result = torch.from_numpy(block).to(device=device, dtype=torch.float64).contiguous()
        aliases = result.device.type == "cpu" and self.dtype == np.dtype("float64")
        simultaneous = int(block.nbytes) + (0 if aliases else result.untyped_storage().nbytes())
        self.staging["peak_read_staging_bytes"] = max(self.staging["peak_read_staging_bytes"], simultaneous)
        self.work["reads"] += 1; self.work["rows_read"] += end-start
        return result


class CoupledOriginalSourceTiles:
    def __init__(self, z_component, Q_component, Phi_component, layout, partition, source_refs):
        require(isinstance(layout, CoupledTileLayout), "Typed layout")
        require(all(type(n) is int and n > 0 for n in partition) and sum(partition) == layout.N, "Fixed ordered partition")
        self.z, self.Q, self.Phi, self.layout = z_component, Q_component, Phi_component, layout
        require(all(isinstance(x, ImmutableRowComponent) for x in (self.z,self.Q,self.Phi)), "File-backed components only")
        require((self.z.shape,self.Q.shape,self.Phi.shape) == ((layout.N,layout.D),(layout.N,layout.C),(layout.N,layout.B)), "Independent full source widths")
        require(max(partition) <= min(x.active_row_limit for x in (self.z,self.Q,self.Phi)), "Partition active cap")
        self.partition, self.source_refs = tuple(partition), dict(source_refs)
        self.work = dict(outer_blocks=0, physical_blocks=0, moment_attempts=0, moments=0,
                         adjoint_attempts=0, adjoints=0, source_attempts=0, source_returns=0)
        self.memory = dict(tracked_peak_active_storage_bytes=0, observations=[],
                           scope="Explicit live tensor storages at operator checkpoints; BLAS/internal allocations use measured process/device peaks")
        self.last_evidence = None
        self.witness_callback = None

    def descriptor(self):
        value = dict(schema=1, kind=KIND, layout=self.layout.descriptor(), partition=list(self.partition),
                     components={k:v.descriptor() for k,v in (("z",self.z),("Q",self.Q),("Phi",self.Phi))},
                     source_refs=self.source_refs, component_domains="Raw z/Q/independent original Phi; no transforms/fits")
        return dict(**value, descriptor_digest=seal(value))

    def __array__(self, *args, **kwargs): raise TypeError("Whole source conversion is prohibited")

    def blocks(self, partition):
        require(tuple(partition) == self.partition, "No silent partition substitution")
        start = 0
        for size in partition:
            end = start+size; yield start,end; start=end

    def account(self, role, *values):
        storages = {v.untyped_storage().data_ptr(): v.untyped_storage().nbytes() for v in values if torch.is_tensor(v)}
        total = sum(storages.values())
        self.memory["tracked_peak_active_storage_bytes"] = max(self.memory["tracked_peak_active_storage_bytes"], total)
        self.memory["observations"].append(dict(role=role, explicit_storage_bytes=total))

    def witness(self, role, start, end, **values):
        # Qualification observer may persist one owning block; no full-P return/gather here.
        if self.witness_callback is not None:
            self.witness_callback(role, start, end, values)

    def outer_block(self, start, end, device="cpu"):
        z, phi = self.z.read(start,end,device), self.Phi.read(start,end,device)
        result = torch.cat((z,phi),1).contiguous(); self.account("outer_concat",z,phi,result)
        self.work["outer_blocks"] += 1; return result

    def augmented(self, start, end, device="cpu"):
        rows = self.outer_block(start,end,device)
        result = torch.cat((rows, rows.new_ones(len(rows),1)),1).contiguous()
        self.account("outer_augmented",rows,result); return result

    def physical_block(self, start, end, device="cpu"):
        z,q = self.z.read(start,end,device), self.Q.read(start,end,device)
        result = torch.cat((z.new_ones(len(z),1),z,q),1).contiguous(); self.account("physical_concat",z,q,result)
        self.work["physical_blocks"] += 1; return result


def _factor_inputs(u,v,hard,owner,mixing,native_cast):
    l=owner.layout; dtype=torch.float32 if native_cast else torch.float64
    require(tuple(u.shape)==(l.N,l.rank) and tuple(v.shape)==(l.K,l.rank) and tuple(hard.shape)==(l.N,), "Factor shapes")
    require(u.dtype==v.dtype==dtype and hard.dtype==torch.long and u.device==v.device==hard.device, "Factor domains")
    require(bool(torch.isfinite(u).all()) and bool(torch.isfinite(v).all()) and bool(((hard>=0)&(hard<l.K)).all())
            and 0<mixing<1, "Finite factors/hard/mixing")


def _probability(u,v,hard,mixing):
    prior=torch.full((len(u),len(v)),float(np.log(mixing/len(v))),dtype=u.dtype,device=u.device)
    prior.scatter_(1,hard[:,None],float(np.log(1-mixing+mixing/len(v))))
    logits=prior+u@v.T/math.sqrt(u.shape[1])
    return logits.to(torch.float64).softmax(1)


@torch.no_grad()
def physical_moments(u,v,hard,owner,mixing,partition):
    owner.work["moment_attempts"]+=1; _factor_inputs(u,v,hard,owner,mixing,u.dtype==torch.float32)
    l=owner.layout; result=torch.zeros((l.K,l.material_width),dtype=torch.float64,device=u.device)
    evidence=dict(partition=list(partition), blocks=[], moment=result, complete=False); owner.last_evidence=evidence
    for start,end in owner.blocks(partition):
        s=owner.physical_block(start,end,u.device); p=_probability(u[start:end],v,hard[start:end],mixing)
        contribution=p.T@s/l.N; result+=contribution
        owner.witness("forward",start,end,material=s,probability=p,contribution=contribution)
        evidence["blocks"].append(dict(start=start,end=end,probability_bytes_sha256=tensor_bytes(p),
                                      contribution_bytes_sha256=tensor_bytes(contribution), material_bytes_sha256=tensor_bytes(s)))
        owner.account("forward_live",s,p,contribution,result,u,v,hard)
        del s,p,contribution
    evidence["moment"]=result.detach().clone(); evidence["complete"]=True; owner.work["moments"]+=1
    return result.detach(), evidence


@torch.no_grad()
def physical_factor_adjoint(u,v,hard,owner,G_complete,mixing,partition,native_cast):
    owner.work["adjoint_attempts"]+=1; _factor_inputs(u,v,hard,owner,mixing,native_cast)
    l=owner.layout; require(G_complete.dtype==torch.float64 and tuple(G_complete.shape)==(l.K,l.material_width)
        and G_complete.device==u.device and bool(torch.isfinite(G_complete).all()), "Complete physical G domain")
    du,dv=torch.zeros_like(u),torch.zeros_like(v)
    evidence=dict(partition=list(partition),native_cast=bool(native_cast),blocks=[],du=du,dv=dv,complete=False); owner.last_evidence=evidence
    for start,end in owner.blocks(partition):
        s=owner.physical_block(start,end,u.device); p=_probability(u[start:end],v,hard[start:end],mixing)
        direction=s@G_complete.T/l.N
        whole=p*(direction-(p*direction).sum(1,keepdim=True))
        block=whole.to(u.dtype)/math.sqrt(l.rank)
        blockdu=block@v; blockdv=block.T@u[start:end]; du[start:end]=blockdu; dv+=blockdv
        owner.witness("adjoint",start,end,material=s,probability=p,R=direction,whole_W=whole,cast_scaled_W=block,du=blockdu,dv_contribution=blockdv)
        evidence["blocks"].append(dict(start=start,end=end,probability_bytes_sha256=tensor_bytes(p),
            whole_W_bytes_sha256=tensor_bytes(whole),du_bytes_sha256=tensor_bytes(blockdu),dv_contribution_bytes_sha256=tensor_bytes(blockdv)))
        owner.account("adjoint_live",s,p,direction,whole,block,blockdu,blockdv,du,dv,u,v,hard,G_complete)
        del s,p,direction,whole,block,blockdu,blockdv
    evidence.update(du=du.detach().clone(),dv=dv.detach().clone(),complete=True); owner.work["adjoints"]+=1
    return du.detach(),dv.detach(),evidence


@torch.no_grad()
def source_ce_rhs(theta,owner,partition):
    owner.work["source_attempts"]+=1; l=owner.layout
    require(theta.dtype==torch.float64 and tuple(theta.shape)==(l.C,l.outer_width+1) and bool(torch.isfinite(theta).all()), "Full held theta domain")
    ce=theta.new_zeros(()); rhs=torch.zeros_like(theta)
    evidence=dict(partition=list(partition),blocks=[],CE=ce,RHS=rhs,complete=False); owner.last_evidence=evidence
    for start,end in owner.blocks(partition):
        a=owner.augmented(start,end,theta.device); q=owner.Q.read(start,end,theta.device)
        logits=a@theta.T; lp=logits.log_softmax(1); p=logits.softmax(1)
        blockce=-(q*lp).sum()/l.N; blockrhs=(q.sum(1,keepdim=True)*p-q).T@a/l.N
        ce+=blockce; rhs+=blockrhs
        owner.witness("source",start,end,outer_augmented=a,Q=q,probability=p,log_probability=lp,CE=blockce,RHS=blockrhs)
        evidence["blocks"].append(dict(start=start,end=end,CE_bytes_sha256=tensor_bytes(blockce),RHS_bytes_sha256=tensor_bytes(blockrhs),
            outer_augmented_bytes_sha256=tensor_bytes(a),Q_bytes_sha256=tensor_bytes(q)))
        owner.account("source_live",a,q,logits,lp,p,blockce,blockrhs,ce,rhs,theta)
        del a,q,logits,lp,p,blockce,blockrhs
    evidence.update(CE=ce.detach().clone(),RHS=rhs.detach().clone(),complete=True); owner.work["source_returns"]+=1
    return ce.detach(),rhs.detach(),evidence
