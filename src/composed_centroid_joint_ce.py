"""Unqualified EZ helpers: one composed centroid head, physical moments only.

Source rows remain literal [z,Phi]. Original Phi has its own [N,B] identity.
Source/admission report semantics are authenticated by the bound caller.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import torch

from src import joint_mean_ce as base
from src.dual_head_ce import _factor_digests, _files, _native_options, _transform, _map
from src.io import array_digest
from src.kernel_mean_ce import _cpu
from src.moments import make_material
from src.nystrom_ce import NystromMap, _content_digest, moment_gradient
from src.shared_features import _tensor_identity
from src.transforms import FeatureTransform

_require, _int, _num, _hex = base._require, base._int, base._num, base._hex
_plain, _copy, _seal = base._plain, base._copy, base._seal
_matrix, _identity, _phi_identity, _runtime = base._matrix, base._identity, base._phi_identity, base._runtime
SCHEMA = 1
MODE = "normalized_single_unscaled_composed_centroid_joint_uniform_CE_v1"
BACKEND = "resident_literal_original_z_Phi_outer_physical_moment_v1"
RESIDENT_SOURCE_MAX_BYTES = base.RESIDENT_SOURCE_MAX_BYTES
POLICY = dict(schema=1, head_count=1, head_weighting="uniform", mass_mode="free",
    objective="own_raw_source_joint_CE_over_own_positive_CE0", source_outer="literal_unscaled_z_originalPhi",
    inner="concat_physical_z_centroid_original_map_of_original_RMS_inverse",
    material="physical_only_1_z_rawQ", block_scales=[1,1], bias_regularized=True,
    original_relu_clamps=True, target_renormalization=False, scale_floor=False,
    original_native_cast=True, free_features=False, new_source_fits=False)


@dataclass(frozen=True)
class ComposedJointLayout:
    physical_dimension: int
    original_phi_basis: int
    classes: int

    def __post_init__(self):
        _int(self.physical_dimension); _int(self.original_phi_basis); _int(self.classes,2)

    @property
    def critic_dimension(self): return self.physical_dimension + self.original_phi_basis

    @property
    def material_width(self): return 1 + self.physical_dimension + self.classes

    def descriptor(self):
        d,b,c=self.physical_dimension,self.original_phi_basis,self.classes
        return dict(schema=1,kind="physical_moment_composed_joint_head_layout_v1",
            physical_dimension=d,original_phi_basis=b,classes=c,critic_dimension=d+b,
            material_width=1+d+c,material_blocks=["mass","physical_z","raw_Q"],
            material_offsets=dict(mass=[0,1],physical_z=[1,1+d],raw_Q=[1+d,1+d+c]),
            head_blocks=["physical_centroid","original_map_RMS_inverse_centroid"],head_bias="last",
            source_outer_blocks=["original_z","original_Phi"],block_scales=[1,1])


def _layout(value):
    _require(type(value) is ComposedJointLayout,"EZ needs its exact physical/composed layout")
    return value


class ResidentOriginalJointSource:
    """Reuse owning literal concatenation arithmetic solely as source-outer supplier."""
    def __init__(self,z,phi,layout,source_refs=None):
        self._layout=_layout(layout)
        self._source=base.ResidentJointFeatures(z,phi,
            base.JointMeanLayout(layout.physical_dimension,layout.original_phi_basis,layout.classes),source_refs)

    def descriptor(self):
        old=self._source.descriptor()
        body=dict(schema=1,kind=BACKEND,shape=old["shape"],dtype=old["dtype"],
            layout=self._layout.descriptor(),components=old["components"],joint_digest=old["joint_digest"])
        return dict(body,descriptor_digest=_seal(body))

    def outer_rows(self): return self._source.outer_rows()

    def material_on(self,q,device):
        # The legacy source provider's wider material method is deliberately unused.
        self._source._verify()
        _require(str(q.device)==str(torch.device(device)),"Physical material device differs")
        z=torch.from_numpy(self._source._z.copy()).to(device)
        _matrix(q,(len(z),self._layout.classes),z.device)
        _require(not q.requires_grad and bool((q>=0).all()) and bool((q.sum(1)>0).all())
            and bool((q.sum(0)>0).all()),"Raw general-Q source domain differs")
        return make_material(z,q).detach()


class OriginalRMSComposedCentroidFeatures:
    """Own original RMS/map bytes; no fit, cache loader, normalization or map probe."""
    def __init__(self,transform,feature_map,layout,source_refs=None):
        self._layout=_layout(layout); d=layout.physical_dimension
        device=transform.scale.device
        td=_transform(transform,d,device); b,md=_map(feature_map,d,device)
        _require(b==layout.original_phi_basis and transform.eps==1e-12
            and not feature_map.anchors.requires_grad and not feature_map.mapping.requires_grad,
            "Original RMS/map domain differs")
        expected=dict(transform=td,**md)
        if source_refs is not None: _require(source_refs==expected,"Composed original supplier bytes differ")
        self._transform=FeatureTransform(center=transform.center.detach().clone(),matrix=None,
            output_center=transform.output_center.detach().clone(),scale=transform.scale.detach().clone(),
            kind="rms",eps=transform.eps)
        self._map=NystromMap(feature_map.anchors.detach().clone(),feature_map.mapping.detach().clone(),kernel="relu")
        self._components=_copy(expected); self._last_returned={}

    def descriptor(self):
        d=self._layout.physical_dimension;dev=self._transform.scale.device
        _,md=_map(self._map,d,dev)
        _require(dict(transform=_transform(self._transform,d,dev),**md)==self._components,
            "Owned composed source bytes changed")
        body=dict(schema=1,kind="original_RMS_inverse_original_Nystrom_composed_centroid_v1",
            layout=self._layout.descriptor(),components=_copy(self._components),
            operation_order="concat(c,map(c*scale+output_center+center))",kernel="relu",head_bias="last")
        return dict(body,descriptor_digest=_seal(body))

    def __call__(self,centers):
        _matrix(centers,(len(centers),self._layout.physical_dimension),self._transform.scale.device)
        _require(not torch.is_autocast_enabled(centers.device.type),"Composed map autocast forbidden")
        raw=centers*self._transform.scale+self._transform.output_center+self._transform.center
        self._last_returned=_cpu(dict(physical_centers=centers,raw_H_centroids=raw))
        mapped=self._map(raw)
        self._last_returned["mapped_centroids"]=_cpu(mapped)
        result=torch.cat((centers,mapped),dim=1)
        self._last_returned["head_features"]=_cpu(result)
        _matrix(result,(len(centers),self._layout.critic_dimension),centers.device)
        return result

    def last_returned(self): return _cpu(self._last_returned)


def _binding(context):
    body={k:v for k,v in context.items() if k!="source_refs"}
    body["source_refs"]={k:v for k,v in context["source_refs"].items() if k!="origin_binding_digest"}
    return _seal(body)


def validate_context(context):
    """Consistency seals only; caller authenticates actual ROOT/source admission."""
    _require(isinstance(context,dict) and set(context)=={"schema","mode","policy","source_refs",
        "native_parameter_digests","asset_descriptors","native_origin","feature_contract"}
        and type(context["schema"]) is int and context["schema"]==1 and context["mode"]==MODE
        and _seal(context["policy"])==_seal(POLICY),"EZ context/mode differs")
    _plain(context);refs=context["source_refs"];assets=context["asset_descriptors"]
    for k in ("nodes","cells","rank","dimension","basis","classes","chunk_size"): _int(refs.get(k))
    _int(refs.get("factor_seed"),0);layout=ComposedJointLayout(refs["dimension"],refs["basis"],refs["classes"])
    n,k,d,b,c=(refs[x] for x in ("nodes","cells","dimension","basis","classes"))
    _require(k>=2 and refs["rank"]<=min(n,k) and type(refs.get("mixing")) in (int,float)
        and refs["mixing"]==.05 and _seal(refs.get("layout"))==_seal(layout.descriptor())
        and type(refs.get("critic_dimension")) is int and refs["critic_dimension"]==d+b
        and refs.get("critic_backend")==BACKEND and isinstance(refs.get("original_options"),dict)
        and isinstance(refs.get("runtime"),dict) and type(refs.get("device")) is str,"EZ source/options/layout differs")
    _hex(refs.get("data_digest"));_hex(refs.get("helper_source_sha256"))
    pair=context["native_parameter_digests"];_require(isinstance(pair,list) and len(pair)==2,"Native pair absent")
    for value in pair:_hex(value)
    _require(isinstance(assets,dict) and set(assets)=={"H","transform","anchors","mapping","Phi_identity","z","Q","assignment"},"Original source components differ")
    for name,shape,dtype in (("z",(n,d),"torch.float64"),("Q",(n,c),"torch.float64"),
        ("assignment",(n,),"torch.int64"),("anchors",(b,d),"torch.float64"),("mapping",(b,b),"torch.float64")):
        _identity(assets[name],shape,dtype)
    _require(assets["H"].get("dtype") in ("torch.float32","torch.float64"),"H dtype differs")
    _identity(assets["H"],(n,d),assets["H"]["dtype"]);_phi_identity(assets["Phi_identity"],n,b)
    t=assets["transform"]
    _require(isinstance(t,dict) and set(t)=={"kind","matrix","center","output_center","scale","eps"}
        and t["kind"]=="rms" and t["matrix"] is None and type(t["eps"]) is float and t["eps"]==1e-12,"Original sixfield RMS differs")
    for name,shape in (("center",(d,)),("output_center",(d,)),("scale",())):_identity(t[name],shape)
    org=context["native_origin"];_require(set(org)=={"moments","centers","labels"},"Physical origin fields differ")
    for name,shape in (("moments",(k,1+d+c)),("centers",(k,d)),("labels",(k,c))):_identity(org[name],shape)
    f=context["feature_contract"];_require(set(f)=={"source_outer","composed"},"Composed/source descriptors absent")
    for desc in f.values():
        _require(type(desc.get("schema")) is int and desc["schema"]==1 and _seal(desc["layout"])==_seal(layout.descriptor())
            and desc["descriptor_digest"]==_seal({x:v for x,v in desc.items() if x!="descriptor_digest"}),"Feature descriptor seal differs")
    outer=f["source_outer"]
    _require(set(outer)=={"schema","kind","shape","dtype","layout","components","joint_digest","descriptor_digest"}
        and outer["kind"]==BACKEND and outer["shape"]==[n,d+b] and all(type(v) is int for v in outer["shape"])
        and outer["dtype"]=="float64" and outer["components"]==dict(z=assets["z"],original_Phi_identity=assets["Phi_identity"]),"Literal source outer differs")
    _hex(outer["joint_digest"])
    comp=f["composed"]
    _require(set(comp)=={"schema","kind","layout","components","operation_order","kernel","head_bias","descriptor_digest"}
        and comp["kind"]=="original_RMS_inverse_original_Nystrom_composed_centroid_v1"
        and comp["components"]=={x:assets[x] for x in ("transform","anchors","mapping")}
        and comp["operation_order"]=="concat(c,map(c*scale+output_center+center))"
        and comp["kernel"]=="relu" and comp["head_bias"]=="last","Composed feature supplier differs")
    a=refs.get("source_admission");_require(isinstance(a,dict) and set(a)=={"report","arrays","root_acceptance"},"Actual raw-source admission absent")
    for v in a.values():
        _require(set(v)=={"path","sha256"} and type(v["path"]) is str and Path(v["path"]).is_absolute()
            and str(Path(v["path"]).resolve())==v["path"],"Admission path not normalized ABS");_hex(v["sha256"])
    _hex(refs.get("origin_binding_digest"));_require(refs["origin_binding_digest"]==_binding(context),"Source/origin binding differs")
    return _copy(refs),_copy(assets),layout


def build_context(source_refs,assets,native_parameter_digests,native_origin,layout,source_descriptor,composed_descriptor):
    _layout(layout);refs=_copy(source_refs)
    refs.update(layout=layout.descriptor(),critic_dimension=layout.critic_dimension,critic_backend=BACKEND)
    value=dict(schema=1,mode=MODE,policy=_copy(POLICY),source_refs=refs,
        native_parameter_digests=_copy(native_parameter_digests),asset_descriptors=_copy(assets),native_origin=_copy(native_origin),
        feature_contract=dict(source_outer=_copy(source_descriptor),composed=_copy(composed_descriptor)))
    refs["origin_binding_digest"]=_binding(value);validate_context(value);return _copy(value)


def validate_input_metadata(z,q,assignment,initial_parameters,phi,options,context,features,composed):
    refs,assets,layout=validate_context(context);n,k,r=(refs[x] for x in ("nodes","cells","rank"))
    _require(type(features) is ResidentOriginalJointSource and type(composed) is OriginalRMSComposedCentroidFeatures
        and features.descriptor()==context["feature_contract"]["source_outer"]
        and composed.descriptor()==context["feature_contract"]["composed"],"Actual owning EZ components differ")
    _matrix(z,(n,layout.physical_dimension));_matrix(q,(n,layout.classes),z.device)
    _require(not z.requires_grad and not q.requires_grad and bool((q>=0).all())
        and bool((q.sum(1)>0).all()) and bool((q.sum(0)>0).all()),"Frozen general-Q domain differs")
    _require(torch.is_tensor(assignment) and assignment.dtype==torch.int64 and assignment.shape==(n,)
        and assignment.device==z.device and not assignment.requires_grad and int(assignment.min())>=0
        and int(assignment.max())==k-1,"Original hard order differs")
    data=array_digest(z.detach().cpu().numpy(),q.detach().cpu().numpy(),assignment.cpu().numpy());old=_native_options(options)
    _require(data==refs["data_digest"] and old==refs["original_options"] and options["factor_seed"]==refs["factor_seed"]
        and options["assignment_rank"]==r and options["chunk_size"]==refs["chunk_size"]
        and ("data_digest" not in options or options["data_digest"]==data) and refs["device"]==str(z.device)
        and _runtime(z.device)==refs["runtime"] and torch.get_default_dtype()==torch.float32
        and not torch.is_autocast_enabled(z.device.type),"Source/native options/runtime differs")
    for name,value in (("z",z),("Q",q),("assignment",assignment)):
        _require(_tensor_identity(value)==assets[name],"Source array bytes differ")
    _require(_factor_digests(initial_parameters,n,k,r,device=z.device)==context["native_parameter_digests"]
        and bool(initial_parameters[0].eq(0).all()),"Initial native origin differs")
    _require(_content_digest(phi,canonical_double=True)==assets["Phi_identity"]["phi_digest"],"Original Phi bytes differ")
    paths,pins=refs.get("asset_paths"),refs.get("files_sha256")
    _require(isinstance(paths,dict) and set(paths)=={"H","map","Phi","Phi_metadata"} and isinstance(pins,dict) and pins,"Full asset pins absent")
    for p in pins:_require(type(p) is str and Path(p).is_absolute() and str(Path(p).resolve())==p,"Readonly path not ABS")
    for p in paths.values():_require(p in pins,"Original cache omitted")
    for v in refs["source_admission"].values():_require(pins.get(v["path"])==v["sha256"],"Source admission omitted")
    _files(pins);_require(json.loads(Path(paths["Phi_metadata"]).read_text())==assets["Phi_identity"],"Mandatory original sidecar differs")
    src=refs.get("current_source");_require(isinstance(src,dict) and set(src)=={"git_head","source_digest","files"},"Current source absent")
    _hex(src["git_head"],40);_hex(src["source_digest"],12);_files(src["files"])
    _require(refs["helper_source_sha256"]==hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"EZ helper bytes differ")
    return dict(old,data_digest=data,composed_joint_mode=MODE,composed_joint_context=_copy(context))


def raw_moment_cotangent(moments,layout,composed,theta,vector,penalty,map_call=None):
    _layout(layout);_matrix(moments,(len(moments),layout.material_width));_num(penalty,True)
    _require(len(moments)>=2,"Composed uniform head requires at least two cells")
    _require(type(composed) is OriginalRMSComposedCentroidFeatures and composed._layout==layout
        and bool((moments[:,0]>0).all()) and bool((moments[:,1+layout.physical_dimension:]>=0).all())
        and bool((moments[:,1+layout.physical_dimension:].sum(1)>0).all())
        and bool((moments[:,1+layout.physical_dimension:].sum(0)>0).all()),"Physical general-Q cotangent domain differs")
    _matrix(theta,(layout.classes,layout.critic_dimension+1),moments.device);_matrix(vector,theta.shape,moments.device)
    feature=composed if map_call is None else lambda centers:map_call(composed,centers)
    value=moment_gradient(moments,layout.physical_dimension,feature,theta.detach(),vector.detach(),penalty,inner_loss_weighting="uniform")
    return _matrix(value.detach(),moments.shape,moments.device)


def complete_moment_cotangent(moments,layout,composed,theta,vector,penalty,CE0,map_call=None):
    _num(CE0,True)
    value=raw_moment_cotangent(moments,layout,composed,theta,vector,penalty,map_call)/CE0
    return _matrix(value.detach(),moments.shape,moments.device)
