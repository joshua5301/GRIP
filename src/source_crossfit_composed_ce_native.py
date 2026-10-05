"""Typed original-source two-fold native ownership; no head/source fitting.

Original composed/map/generalQ/LowRank math is supplied unchanged. A new packed
origin is bound only AFTER the first live native forward and its owning decode.
"""
import torch

from src import composed_centroid_joint_ce as joint
from src import source_crossfit_composed_ce as math_supplier
from src.dual_head_ce import _factor_digests, _files, _native_options
from src.kernel_mean_ce import _cpu, _plain, _seal
from src.shared_features import _tensor_identity
from src.io import array_digest
from src.nystrom_ce import _content_digest
from pathlib import Path
import hashlib
import json

SCHEMA = 1
MODE = "normalized_two_fixed_source_fold_original_composed_uniform_CE_v1"
PARTITION = "row_parity_even_A_odd_B_v1"
POLICY = dict(schema=1,mode=MODE,folds=["A","B"],head_count=2,head_weighting="uniform",
    branch_weights=[.5,.5],bias_regularized=True,material="masked_physical_1_z_rawQ",
    moment_normalization="all_source_N",outer="opposite_fixed_original_z_Phi_rawQ_mean",
    source_scales=[1,1],RMS_map="original_composed",target_renormalization=False,free_features=False)


def require(ok,message):
    if not ok: raise ValueError(message)


def partition_descriptor(n):
    joint._int(n,2)
    indices={fold:torch.arange(start,n,2,dtype=torch.long) for fold,start in (("A",0),("B",1))}
    return dict(schema=1,kind=PARTITION,nodes=n,indices={f:_tensor_identity(v) for f,v in indices.items()},
        counts={f:len(v) for f,v in indices.items()},branch_weights=[.5,.5])


class TwoFoldOriginalSourceOwner:
    """Own immutable full original source plus explicit held NumPy row copies."""
    def __init__(self,z,phi,q,layout,source_admissions,partition=PARTITION,original_components=None):
        require(type(layout) is joint.ComposedJointLayout and partition==PARTITION,"Original layout/fixed partition differs")
        self.layout=layout; self._source=joint.ResidentOriginalJointSource(z,phi,layout,original_components)
        joint._matrix(q,(len(z),layout.classes));require(not q.requires_grad,"Frozen rawQ required")
        self._q=q.detach().cpu().clone();self._q_identity=_tensor_identity(self._q)
        self._z_identity=_tensor_identity(z);self._phi_digest=_content_digest(phi,canonical_double=True)
        require(bool((self._q>=0).all()) and bool((self._q.sum(1)>0).all())
            and bool((self._q.sum(0)>0).all()),"Raw generalQ domain differs")
        self._partition=partition_descriptor(len(z));self._admissions=joint._copy(source_admissions)
        rows=self._source.outer_rows()
        self._held={f:rows[start::2].copy(order="C") for f,start in (("A",0),("B",1))}
        for rows in self._held.values():rows.setflags(write=False)
        self._held_id={f:dict(shape=list(rows.shape),dtype=str(rows.dtype),digest=array_digest(rows)) for f,rows in self._held.items()}
        self._descriptor=self._describe();self._last_returned={}

    def _describe(self):
        body=dict(schema=1,kind="own_original135_two_fold_fixed_source_v1",layout=self.layout.descriptor(),
            source_outer=self._source.descriptor(),raw_Q=self._q_identity,partition=self._partition,
            held_rows=self._held_id,source_admissions=self._admissions)
        return dict(body,descriptor_digest=_seal(body))

    def descriptor(self):
        self._source.descriptor()
        require(_tensor_identity(self._q)==self._q_identity,"Owned rawQ changed")
        for f,rows in self._held.items():
            require(not rows.flags.writeable and list(rows.shape)==self._held_id[f]["shape"]
                and array_digest(rows)==self._held_id[f]["digest"],"Owned fixed held rows changed")
        require(self._describe()==self._descriptor,"Source owner changed")
        return joint._copy(self._descriptor)

    def last_returned(self):return _cpu(self._last_returned)

    def held_numpy_rows(self,fold):
        self.descriptor();require(fold in ("A","B"),"Unknown held fold");return self._held[fold]

    def held_Q(self,fold,device):
        self.descriptor();require(fold in ("A","B"),"Unknown held fold")
        return self._q[0 if fold=="A" else 1::2].clone().to(device)

    def material_on(self,device):
        self.descriptor()
        z=torch.from_numpy(self._source._source._z.copy()).to(device)
        q=self._q.to(device)
        value=math_supplier.pack_physical_material(z,q)
        self._last_returned["packed_material"]=value;self._last_returned["packed_material"]=_cpu(value)
        require(value.numel()*value.element_size()+self._source._source._rows.nbytes+self._q.numel()*8
            <=536870912,"Intrinsic source+packed logical512 differs")
        return value


def build_pre_origin_contract(source_refs,assets,initial_digests,source_descriptor,composed_descriptor):
    body=dict(schema=1,mode=MODE,policy=POLICY,source_refs=joint._copy(source_refs),
        asset_descriptors=joint._copy(assets),native_parameter_digests=list(initial_digests),
        feature_contract=dict(source_outer=joint._copy(source_descriptor),composed=joint._copy(composed_descriptor)),
        partition=partition_descriptor(source_refs["nodes"]))
    value=dict(body,contract_digest=_seal(body));validate_pre_origin_contract(value);return value


def validate_pre_origin_contract(value):
    require(isinstance(value,dict) and set(value)=={"schema","mode","policy","source_refs","asset_descriptors",
        "native_parameter_digests","feature_contract","partition","contract_digest"}
        and value["schema"]==SCHEMA and value["mode"]==MODE and value["policy"]==POLICY
        and value["contract_digest"]==_seal({k:v for k,v in value.items() if k!="contract_digest"}),"New pre-origin schema/seal differs")
    _plain(value);refs=value["source_refs"];assets=value["asset_descriptors"]
    n,k,d,b,c,r=(refs[x] for x in ("nodes","cells","dimension","basis","classes","rank"))
    for x in (n,k,d,b,c,r):joint._int(x)
    require(r<=min(n,k) and refs["factor_seed"]==0 and refs["mixing"]==.05
        and value["partition"]==partition_descriptor(n),"Original native/fixed parity differs")
    layout=joint.ComposedJointLayout(d,b,c)
    for name,shape,dtype in (("z",(n,d),"torch.float64"),("Q",(n,c),"torch.float64"),("assignment",(n,),"torch.int64"),
                            ("anchors",(b,d),"torch.float64"),("mapping",(b,b),"torch.float64")):
        joint._identity(assets[name],shape,dtype)
    require(set(assets)=={"H","transform","anchors","mapping","Phi_identity","z","Q","assignment"},"Original eight descriptors differ")
    joint._identity(assets["H"],(n,d),assets["H"]["dtype"]);require(assets["H"]["dtype"] in ("torch.float32","torch.float64"),"Original H dtype differs")
    td=assets["transform"];require(set(td)=={"kind","matrix","center","output_center","scale","eps"} and td["kind"]=="rms"
        and td["matrix"] is None and td["eps"]==1e-12,"Original six-field RMS differs")
    for name,shape in (("center",(d,)),("output_center",(d,)),("scale",())):joint._identity(td[name],shape)
    joint._phi_identity(assets["Phi_identity"],n,b)
    for descriptor in value["feature_contract"].values():require(descriptor["descriptor_digest"]==_seal({k:v for k,v in descriptor.items() if k!="descriptor_digest"}),"Owning descriptor seal differs")
    require(value["feature_contract"]["source_outer"]["layout"]==layout.descriptor()
        and value["feature_contract"]["source_outer"]["source_outer"]["components"]==dict(z=assets["z"],original_Phi_identity=assets["Phi_identity"])
        and value["feature_contract"]["source_outer"]["raw_Q"]==assets["Q"]
        and value["feature_contract"]["source_outer"]["partition"]==value["partition"]
        and value["feature_contract"]["composed"]["components"]=={x:assets[x] for x in ("transform","anchors","mapping")},"Original source/composed owner differs")
    require(len(value["native_parameter_digests"])==2,"Initial native pair absent")
    for x in value["native_parameter_digests"]:joint._hex(x)
    for x in ("data_digest","helper_source_sha256","inner_helper_source_sha256","CPU_math_source_sha256"):joint._hex(refs[x])
    require(refs["inner_helper_source_sha256"]!=refs["helper_source_sha256"] and refs["chunk_size"]==4096
        and isinstance(refs["source_admission"],dict) and isinstance(refs["CPU_admission"],dict),"Typed native/current/math supplier refs differ")
    return refs,assets,layout


def bind_origin_context(contract,packed_M0,quotients):
    refs,_,layout=validate_pre_origin_contract(contract);k=refs["cells"]
    joint._matrix(packed_M0,(k,2*layout.material_width));require(set(quotients)=={"A","B"},"Two native quotient returns absent")
    origin=dict(packed_moments=_tensor_identity(packed_M0),folds={})
    for f,(centers,targets,mass) in quotients.items():
        origin["folds"][f]=dict(centers=_tensor_identity(centers),targets=_tensor_identity(targets),mass=_tensor_identity(mass))
    body=dict(schema=1,mode=MODE,pre_origin_contract=joint._copy(contract),native_origin=origin)
    value=dict(body,context_digest=_seal(body));validate_context(value);return value


def validate_context(value):
    require(set(value)=={"schema","mode","pre_origin_contract","native_origin","context_digest"}
        and value["schema"]==SCHEMA and value["mode"]==MODE
        and value["context_digest"]==_seal({k:v for k,v in value.items() if k!="context_digest"}),"New origin context/seal differs")
    refs,assets,layout=validate_pre_origin_contract(value["pre_origin_contract"]);org=value["native_origin"];k=refs["cells"]
    require(set(org)=={"packed_moments","folds"} and set(org["folds"])=={"A","B"},"Native packed origin absent")
    joint._identity(org["packed_moments"],(k,2*layout.material_width))
    for fold in org["folds"].values():
        for name,shape in (("centers",(k,layout.physical_dimension)),("targets",(k,layout.classes)),("mass",(k,))):joint._identity(fold[name],shape)
    return refs,assets,layout


def check_current_contract(contract):
    refs,_,_=validate_pre_origin_contract(contract)
    _files(refs["files_sha256"]);_files(refs["current_source"]["files"])


def complete_packed_cotangent(M,layout,composed,theta_A,theta_B,vector_A,vector_B,penalty,CE0,
        map_call=None,observe_return=None,raw_call=None):
    # Same admitted two raw calls and normalization; raw_call only observes entry/return.
    joint._num(CE0,True);joint._matrix(M,(len(M),2*layout.material_width))
    invoke=(lambda f,*a,**kw:f(*a,**kw)) if raw_call is None else raw_call
    w=layout.material_width
    raw_A=invoke(joint.raw_moment_cotangent,M[:,:w],layout,composed,theta_A,vector_A,penalty,map_call=map_call)
    if observe_return is not None:observe_return("raw_A",raw_A)
    raw_B=invoke(joint.raw_moment_cotangent,M[:,w:],layout,composed,theta_B,vector_B,penalty,map_call=map_call)
    if observe_return is not None:observe_return("raw_B",raw_B)
    value=(.5*torch.cat((raw_A,raw_B),1)/CE0).detach()
    if observe_return is not None:observe_return("complete",value)
    return joint._matrix(value,M.shape,M.device)

def validate_inputs(initial,hard,options,contract,owner,composed,material):
    refs,assets,layout=validate_pre_origin_contract(contract);n,k,r=(refs[x] for x in ("nodes","cells","rank"))
    require(type(owner) is TwoFoldOriginalSourceOwner and type(composed) is joint.OriginalRMSComposedCentroidFeatures
        and owner.descriptor()==contract["feature_contract"]["source_outer"]
        and composed.descriptor()==contract["feature_contract"]["composed"],"Exact source/composed owners differ")
    require(_native_options(options)==refs["original_options"] and refs["runtime"]==joint._runtime(initial[0].device)
        and refs["device"]==str(initial[0].device) and torch.get_default_dtype()==torch.float32
        and not torch.is_autocast_enabled(initial[0].device.type),"Original options/runtime differ")
    require(_factor_digests(initial,n,k,r,initial[0].device)==contract["native_parameter_digests"]
        and bool(initial[0].eq(0).all()),"Original native initial pair differs")
    require(torch.is_tensor(hard) and hard.dtype==torch.int64 and hard.shape==(n,) and hard.device==initial[0].device
        and not hard.requires_grad and int(hard.min())>=0 and int(hard.max())==k-1
        and _tensor_identity(hard)==assets["assignment"],"Original hard owner differs")
    joint._matrix(material,(n,2*layout.material_width),initial[0].device);require(not material.requires_grad,"Material must be frozen")
    require("packed_material" in owner._last_returned and _tensor_identity(material)==_tensor_identity(owner._last_returned["packed_material"]),"Supplied tape differs from owning single material return")
    require(owner._z_identity==assets["z"] and owner._phi_digest==assets["Phi_identity"]["phi_digest"]
        and array_digest(owner._source._source._z,owner._q.numpy(),hard.cpu().numpy())==refs["data_digest"],"Original source digest differs")
    require(refs["helper_source_sha256"]==hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"New helper bytes differ")
    for p in refs["asset_paths"].values():require(p in refs["files_sha256"],"Original asset pin absent")
    require(json.loads(Path(refs["asset_paths"]["Phi_metadata"]).read_text())==assets["Phi_identity"],"Original Phi sidecar differs")
    check_current_contract(contract)
    return dict(_native_options(options),data_digest=refs["data_digest"],two_fold_mode=MODE,
        pre_origin_contract_digest=contract["contract_digest"])


OriginalRMSComposedCentroidFeatures=joint.OriginalRMSComposedCentroidFeatures
