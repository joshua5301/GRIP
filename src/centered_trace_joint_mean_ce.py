"""Separate typed helpers for centered original-Phi trace-balanced joint CE.

This limited module has no optimizer, initializer, head solver, trajectory,
resume validator, source loader, or native-origin qualification entry point.
Original Phi remains RAW [N,B]. The derived critic has width F=D+B;
its frozen metric and new origin are distinct from the closed unscaled mode.
"""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import platform

import numpy as np
import torch

from src.dual_head_ce import _factor_digests, _files, _native_options
from src.io import array_digest
from src.moments import decode_moments, make_material
from src.nystrom_ce import _content_digest, moment_gradient
from src.shared_features import _tensor_identity

SCHEMA = 1
MODE = "single_joint_centered_originalPhi_trace_balanced_preserve_raw_total_uniform_CE_v1"
OBJECTIVE = "centered_trace_joint_outer_raw_teacher_CE_over_frozen_own_positive_CE0"
BACKEND = "resident_centered_trace_original_z_Phi_FP64_v1"
RESIDENT_SOURCE_MAX_BYTES = 512 * 1024**2
POLICY = dict(schema=SCHEMA, objective=OBJECTIVE, coefficient=1, mass_mode="free",
    head_weighting="uniform", teacher="original_frozen_raw_Q", head_count=1,
    critic="P_weighted_frozen_centered_trace_original_z_and_Phi_means",
    metric="all_source_Phi_centered_equal_branch_preserve_raw_total",
    kernel="original_relu_cache", bias_regularized=True, scale_floor=False,
    target_renormalization=False, original_moment_operator=True,
    original_native_cast=True, physical_material_forward="separate_original_BLAS_shape",
    free_features=False, graph_factor=False)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _int(value, minimum=1):
    _require(type(value) is int and value >= minimum, "Expected typed integer dimension")
    return value


def _num(value, positive=False):
    _require(type(value) in (int, float) and math.isfinite(value)
        and (not positive or value > 0), "Expected non-bool finite scalar")
    return value


def _hex(value, length=64):
    _require(type(value) is str and len(value) == length
        and all(c in "0123456789abcdef" for c in value), "Malformed content/source digest")
    return value


def _plain(value):
    if isinstance(value, dict):
        _require(all(type(k) is str for k in value), "Metadata keys must be strings")
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    _require(value is None or type(value) in (str, bool, int)
        or type(value) is float and math.isfinite(value), "Only finite JSON metadata is allowed")
    return value


def _seal(value):
    return hashlib.sha256(json.dumps(_plain(value), sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _copy(value):
    return json.loads(json.dumps(_plain(value), allow_nan=False))


def _matrix(value, shape, device=None):
    _require(torch.is_tensor(value) and value.layout == torch.strided
        and tuple(value.shape) == tuple(shape) and value.dtype == torch.float64
        and (device is None or value.device == device) and bool(torch.isfinite(value).all()),
        "FP64 matrix shape/dtype/device/finiteness differs")
    return value


def _identity(value, shape, dtype="torch.float64"):
    _require(isinstance(value, dict) and set(value) == {"shape", "dtype", "digest"}
        and isinstance(value["shape"], list) and all(type(v) is int for v in value["shape"])
        and value["shape"] == list(shape) and value["dtype"] == dtype,
        "Array identity shape/dtype differs")
    _hex(value["digest"])


def _phi_identity(value, nodes, basis):
    _require(isinstance(value, dict) and set(value) == {"schema", "h_digest", "map_digest",
        "shape", "dtype", "phi_digest"} and type(value["schema"]) is int
        and value["schema"] == SCHEMA and value["dtype"] == "float64"
        and isinstance(value["shape"], list) and all(type(v) is int for v in value["shape"])
        and value["shape"] == [nodes, basis], "Original Phi must retain its actual [N,B] identity")
    for key in ("h_digest", "map_digest", "phi_digest"):
        _hex(value[key])


@dataclass(frozen=True)
class CenteredTraceJointLayout:
    physical_dimension: int
    original_phi_basis: int
    classes: int

    def __post_init__(self):
        _int(self.physical_dimension); _int(self.original_phi_basis); _int(self.classes, 2)

    @property
    def critic_dimension(self):
        return self.physical_dimension + self.original_phi_basis

    @property
    def material_width(self):
        return 1 + self.critic_dimension + self.classes

    def descriptor(self):
        d, b, c = self.physical_dimension, self.original_phi_basis, self.classes
        return dict(schema=SCHEMA, kind="centered_trace_original_z_originalPhi_joint_layout_v1",
            physical_dimension=d, original_phi_basis=b, classes=c, critic_dimension=d+b,
            material_width=1+d+b+c, block_order=["mass", "scaled_original_z", "centered_scaled_original_Phi", "raw_Q"],
            material_offsets=dict(mass=[0, 1], z=[1, 1+d], original_Phi=[1+d, 1+d+b],
                raw_Q=[1+d+b, 1+d+b+c]), block_scales="frozen_metric", head_bias="last")


def _layout(value):
    _require(type(value) is CenteredTraceJointLayout, "Only the fixed typed joint layout is supported")
    return value


METRIC_KIND = "centered_originalPhi_equal_branch_preserve_raw_total_v1"
OPERATION_ORDER = ["CPU_FP64_mean_phi_axis0", "tau_z_square_sum1_mean",
    "tau_Phi_raw_square_sum1_mean", "centered_phi=phi-mean_phi",
    "tau_Phi_centered_square_sum1_mean", "total=tau_z+tau_Phi_raw",
    "a_z=math.sqrt(total/(2*tau_z))", "a_phi=math.sqrt(total/(2*tau_Phi_centered))",
    "scaled_z=z*a_z", "scaled_phi=centered_phi*a_phi", "torch.cat_scaled_z_scaled_phi_dim1"]


def _refs(value, keys):
    _require(isinstance(value, dict) and set(value) == set(keys), "Actual immutable admissions missing")
    for ref in value.values():
        _require(isinstance(ref, dict) and set(ref) == {"path", "sha256"}
            and type(ref["path"]) is str and Path(ref["path"]).is_absolute()
            and str(Path(ref["path"]).resolve()) == ref["path"], "Admission ref not normalized ABS")
        _hex(ref["sha256"])


def _metric_descriptor(value, basis, require_admission=False):
    keys={"schema", "kind", "original_phi_basis", "mean_phi", "scalar_hex",
        "operation_order", "calibration_refs", "content_digest", "descriptor_digest"}
    _require(isinstance(value, dict) and set(value) == keys and type(value["schema"]) is int
        and value["schema"] == SCHEMA and value["kind"] == METRIC_KIND
        and type(value["original_phi_basis"]) is int and value["original_phi_basis"] == basis
        and value["operation_order"] == OPERATION_ORDER, "Frozen metric schema/operation order differs")
    _identity(value["mean_phi"], (basis,))
    scalars=value["scalar_hex"]
    names={"tau_z", "tau_Phi_raw", "tau_Phi_centered", "total", "a_z", "a_phi"}
    _require(isinstance(scalars, dict) and set(scalars) == names
        and all(type(x) is str for x in scalars.values()), "Metric scalar hex fields differ")
    numbers={}
    for name,text in scalars.items():
        try: number=float.fromhex(text)
        except (ValueError, OverflowError) as error: raise ValueError("Invalid metric scalar hex") from error
        _num(number, positive=True)
        _require(number.hex() == text, "Metric scalar hex is not canonical")
        numbers[name]=number
    _require(numbers["total"] == numbers["tau_z"]+numbers["tau_Phi_raw"]
        and numbers["a_z"] == math.sqrt(numbers["total"]/(2*numbers["tau_z"]))
        and numbers["a_phi"] == math.sqrt(numbers["total"]/(2*numbers["tau_Phi_centered"])),
        "Frozen trace-preserving coefficient formula differs")
    if require_admission or value["calibration_refs"] is not None:
        _refs(value["calibration_refs"], {"report", "arrays", "ROOT", "independent"})
    _hex(value["content_digest"]); _hex(value["descriptor_digest"])
    _require(value["content_digest"] == _seal({k:v for k,v in value.items()
        if k not in ("calibration_refs", "content_digest", "descriptor_digest")}),
        "Intrinsic metric content seal differs")
    _require(value["descriptor_digest"] == _seal({k:v for k,v in value.items()
        if k != "descriptor_digest"}), "Metric descriptor seal differs")
    return _copy(value)


class FrozenCenteredTraceMetric:
    """Own frozen calibration values; no trace/mean recalculation from source rows."""
    def __init__(self, mean_phi, tau_z, tau_phi_raw, tau_phi_centered,
            a_z, a_phi, calibration_refs=None):
        _require(torch.is_tensor(mean_phi) and mean_phi.ndim == 1 and len(mean_phi) > 0,
            "Mean Phi must be a nonempty CPU FP64 vector")
        basis=len(mean_phi); _matrix(mean_phi, (basis,), torch.device("cpu"))
        _require(not mean_phi.requires_grad, "Mean Phi must be frozen")
        _require(all(type(v) is float for v in (tau_z,tau_phi_raw,tau_phi_centered,a_z,a_phi)),
            "Calibration scalars must be non-bool floats")
        values=dict(tau_z=tau_z, tau_Phi_raw=tau_phi_raw, tau_Phi_centered=tau_phi_centered,
            total=tau_z+tau_phi_raw, a_z=a_z, a_phi=a_phi)
        _require(all(type(v) is float for v in values.values()), "Calibration scalars must be non-bool floats")
        self._mean=mean_phi.detach().clone().contiguous()
        value=dict(schema=SCHEMA,kind=METRIC_KIND,original_phi_basis=basis,
            mean_phi=_tensor_identity(self._mean),scalar_hex={k:v.hex() for k,v in values.items()},
            operation_order=_copy(OPERATION_ORDER),calibration_refs=_copy(calibration_refs))
        value["content_digest"]=_seal({k:v for k,v in value.items() if k != "calibration_refs"})
        self._descriptor=dict(value,descriptor_digest=_seal(value))
        _metric_descriptor(self._descriptor,basis)

    def _verify(self):
        _matrix(self._mean,(self._descriptor["original_phi_basis"],),torch.device("cpu"))
        _require(not self._mean.requires_grad and _tensor_identity(self._mean) == self._descriptor["mean_phi"],
            "Owning mean Phi bytes changed")
        _metric_descriptor(self._descriptor,len(self._mean))

    def descriptor(self):
        self._verify(); return _copy(self._descriptor)

    def mean_phi(self):
        self._verify(); return self._mean.detach().clone()

    def scalar(self, name):
        self._verify(); _require(name in self._descriptor["scalar_hex"], "Metric scalar name differs")
        return float.fromhex(self._descriptor["scalar_hex"][name])


class ResidentCenteredTraceFeatures:
    """Own raw source components and DIRECT derived rows; never a fake Phi cache."""
    def __init__(self, z, phi, layout, metric, source_refs=None):
        self._layout=_layout(layout)
        _require(type(metric) is FrozenCenteredTraceMetric, "Typed frozen calibration is required")
        self._metric=metric; _metric_descriptor(metric.descriptor(),layout.original_phi_basis)
        _require(torch.is_tensor(z) and z.ndim == 2 and len(z) > 0, "Original z must be FP64")
        n=len(z); _require(n*(layout.critic_dimension+layout.classes)*8 <= RESIDENT_SOURCE_MAX_BYTES,
            "Logical resident source payload exceeds512MiB")
        _matrix(z,(n,layout.physical_dimension))
        _require(not z.requires_grad and z.device.type in ("cpu","cuda"), "z must be frozen")
        if torch.is_tensor(phi):
            _matrix(phi,(n,layout.original_phi_basis),torch.device("cpu"))
            _require(not phi.requires_grad,"Original Phi must be frozen"); phi=phi.detach().numpy()
        _require(isinstance(phi,np.ndarray) and phi.shape == (n,layout.original_phi_basis)
            and phi.dtype == np.float64 and bool(np.isfinite(phi).all()), "Original Phi must remain CPU FP64[N,B]")
        refs={} if source_refs is None else _copy(source_refs)
        _require(isinstance(refs,dict) and set(refs) <= {"z","Phi_identity"}, "Only raw component refs allowed")
        zid=_tensor_identity(z.detach()); phi_digest=_content_digest(phi,canonical_double=True)
        if "z" in refs: _require(refs["z"] == zid,"Original z identity differs")
        if "Phi_identity" in refs:
            _phi_identity(refs["Phi_identity"],n,layout.original_phi_basis)
            _require(refs["Phi_identity"]["phi_digest"] == phi_digest,"Original Phi content differs")
        z_cpu=z.detach().cpu().clone().contiguous()
        phi_cpu=torch.from_numpy(np.array(phi,copy=True,order="C"))
        centered=phi_cpu-metric.mean_phi()
        scaled_z=z_cpu*metric.scalar("a_z"); scaled_phi=centered*metric.scalar("a_phi")
        derived=torch.cat((scaled_z,scaled_phi),1)
        _matrix(derived,(n,layout.critic_dimension),torch.device("cpu"))
        self._z=np.array(z_cpu.numpy(),copy=True,order="C")
        self._phi=np.array(phi_cpu.numpy(),copy=True,order="C")
        self._rows=np.array(derived.numpy(),copy=True,order="C")
        for value in (self._z,self._phi,self._rows): value.flags.writeable=False
        self._component_refs=dict(z=zid,original_Phi_identity=refs.get("Phi_identity"))
        self._component_digests=[_content_digest(v,canonical_double=True) for v in (self._z,self._phi,self._rows)]
        self._joint_identity=_tensor_identity(derived)
        self._metric_digest=metric.descriptor()["content_digest"]

    def _verify(self):
        _require(self._metric.descriptor()["content_digest"] == self._metric_digest,"Metric bytes changed")
        _require(all(v.flags.c_contiguous and not v.flags.writeable and v.flags.owndata
            and v.dtype == np.float64 for v in (self._z,self._phi,self._rows)),"Owning source/derived buffers changed")
        _require([_content_digest(v,canonical_double=True) for v in (self._z,self._phi,self._rows)]
            == self._component_digests,"Raw or derived component bytes changed")

    def descriptor(self):
        self._verify()
        value=dict(schema=SCHEMA,kind=BACKEND,shape=list(self._rows.shape),dtype="float64",
            layout=self._layout.descriptor(),components=_copy(self._component_refs),
            metric_content_digest=self._metric_digest,joint_digest=self._component_digests[2],
            joint_tensor_identity=_copy(self._joint_identity))
        return dict(value,descriptor_digest=_seal(value))

    def metric_descriptor(self):
        self._verify(); return self._metric.descriptor()

    def outer_rows(self):
        self._verify(); value=self._rows.copy(); value.flags.writeable=False; return value

    def material_on(self, q, device):
        self._verify(); device=torch.device(device)
        _matrix(q,(len(self._rows),self._layout.classes),device)
        _require(not q.requires_grad and bool((q >= 0).all()) and bool((q.sum(1) > 0).all())
            and bool((q.sum(0) > 0).all()),"Raw Q general row/class support differs")
        rows=torch.from_numpy(self._rows.copy()).to(device=device)
        return make_material(rows,q).detach()


def _features_descriptor(value, refs, assets, layout, metric):
    _require(isinstance(value,dict) and set(value) == {"schema","kind","shape","dtype",
        "layout","components","metric_content_digest","joint_digest","joint_tensor_identity","descriptor_digest"}
        and type(value["schema"]) is int and value["schema"] == SCHEMA and value["kind"] == BACKEND
        and value["dtype"] == "float64" and isinstance(value["shape"],list)
        and all(type(v) is int for v in value["shape"])
        and value["shape"] == [refs["nodes"],layout.critic_dimension]
        and _seal(value["layout"]) == _seal(layout.descriptor()),"Derived provider layout differs")
    _require(value["components"] == dict(z=assets["z"],original_Phi_identity=assets["Phi_identity"]),
        "Raw z/Phi supplier identities changed")
    _require(value["metric_content_digest"] == metric["content_digest"],"Provider metric content seal differs")
    _identity(value["joint_tensor_identity"],(refs["nodes"],layout.critic_dimension))
    for name in ("joint_digest","metric_content_digest","descriptor_digest"): _hex(value[name])
    _require(value["descriptor_digest"] == _seal({k:v for k,v in value.items()
        if k != "descriptor_digest"}),"Derived provider descriptor seal differs")


def _binding(context):
    value = {k: v for k, v in context.items() if k != "source_refs"}
    value["source_refs"] = {k: v for k, v in context["source_refs"].items()
        if k != "origin_binding_digest"}
    return _seal(value)


def validate_context(context):
    """Typed metadata consistency only; does not qualify source files or origins."""
    _require(isinstance(context, dict) and set(context) == {"schema", "mode", "policy", "source_refs",
        "native_parameter_digests", "asset_descriptors", "metric", "native_origin", "joint_origin"}
        and type(context["schema"]) is int and context["schema"] == SCHEMA
        and context["mode"] == MODE and _seal(context["policy"]) == _seal(POLICY),
        "Joint context schema/mode/policy differs")
    _plain(context)
    refs, assets = context["source_refs"], context["asset_descriptors"]
    _require(isinstance(refs, dict), "Source references must be an object")
    for key in ("nodes", "cells", "rank", "dimension", "basis", "classes", "chunk_size"):
        _int(refs.get(key))
    _int(refs.get("factor_seed"), 0)
    layout = CenteredTraceJointLayout(refs["dimension"], refs["basis"], refs["classes"])
    _require(refs["cells"] >= 2 and refs["rank"] <= min(refs["nodes"], refs["cells"])
        and type(refs.get("mixing")) in (int, float) and refs["mixing"] == .05
        and type(refs.get("critic_dimension")) is int
        and refs["critic_dimension"] == layout.critic_dimension
        and refs.get("critic_backend") == BACKEND and "execution_backend" not in refs
        and _seal(refs.get("joint_layout")) == _seal(layout.descriptor())
        and isinstance(refs.get("original_options"), dict) and isinstance(refs.get("runtime"), dict)
        and type(refs.get("device")) is str, "Original dimensions/options or joint route differs")
    _hex(refs.get("data_digest")); _hex(refs.get("helper_source_sha256"))
    native = context["native_parameter_digests"]
    _require(isinstance(native, list) and len(native) == 2, "Native U0/V0 identity pair is required")
    for value in native:
        _hex(value)
    _require(isinstance(assets, dict) and set(assets) == {"H", "transform", "anchors", "mapping",
        "Phi_identity", "z", "Q", "assignment"}, "Original assets cannot be replaced by joint assets")
    n, k, d, b, c = (refs[x] for x in ("nodes", "cells", "dimension", "basis", "classes"))
    _identity(assets["z"], (n, d)); _identity(assets["Q"], (n, c))
    _identity(assets["assignment"], (n,), "torch.int64")
    _require(isinstance(assets["H"], dict) and assets["H"].get("dtype")
        in ("torch.float32", "torch.float64"), "Original H identity dtype differs")
    _identity(assets["H"], (n, d), assets["H"]["dtype"])
    _identity(assets["anchors"], (b, d)); _identity(assets["mapping"], (b, b))
    transform = assets["transform"]
    _require(isinstance(transform, dict) and set(transform) == {"kind", "matrix", "center",
        "output_center", "scale", "eps"} and transform["kind"] == "rms"
        and transform["matrix"] is None and type(transform["eps"]) in (int, float)
        and transform["eps"] == 1e-12, "Original RMS descriptor differs")
    _identity(transform["center"], (d,)); _identity(transform["output_center"], (d,))
    _identity(transform["scale"], ())
    _phi_identity(assets["Phi_identity"], n, b)
    metric = _metric_descriptor(context["metric"], b, require_admission=True)
    _features_descriptor(refs.get("joint_features"), refs, assets, layout, metric)
    for name, width in (("native_origin", d), ("joint_origin", layout.critic_dimension)):
        origin = context[name]
        _require(isinstance(origin, dict) and set(origin) == {"moments", "centers", "labels"},
            "Physical and newly measured joint origins are separate")
        _identity(origin["moments"], (k, 1+width+c))
        _identity(origin["centers"], (k, width)); _identity(origin["labels"], (k, c))
    admission = refs.get("source_admission")
    _require(isinstance(admission, dict) and set(admission) == {"report", "arrays", "root_acceptance"},
        "Future actual source admission requires all three immutable refs")
    for value in admission.values():
        _require(isinstance(value, dict) and set(value) == {"path", "sha256"}
            and type(value["path"]) is str and Path(value["path"]).is_absolute()
            and str(Path(value["path"]).resolve()) == value["path"], "Admission refs must be normalized ABS")
        _hex(value["sha256"])
    _hex(refs.get("origin_binding_digest"))
    _require(refs["origin_binding_digest"] == _binding(context), "Component/origin/source seal differs")
    return _copy(refs), _copy(assets), layout


def build_context(source_refs, asset_descriptors, native_parameter_digests, native_origin,
                  joint_origin, layout, joint_feature_descriptor, metric_descriptor):
    _layout(layout)
    refs = _copy(source_refs)
    _require(isinstance(refs, dict) and [refs.get(k) for k in ("dimension", "basis", "classes")]
        == [layout.physical_dimension, layout.original_phi_basis, layout.classes],
        "Original source D/B/C must match the fixed layout")
    refs.update(critic_dimension=layout.critic_dimension, joint_layout=layout.descriptor(),
        joint_features=_copy(joint_feature_descriptor), critic_backend=BACKEND)
    value = dict(schema=SCHEMA, mode=MODE, policy=_copy(POLICY), source_refs=refs,
        native_parameter_digests=_copy(native_parameter_digests),
        asset_descriptors=_copy(asset_descriptors), metric=_copy(metric_descriptor),
        native_origin=_copy(native_origin),
        joint_origin=_copy(joint_origin))
    refs["origin_binding_digest"] = _binding(value)
    validate_context(value)
    return _copy(value)


def _runtime(device):
    return dict(Python=platform.python_version(), Torch=str(torch.__version__), NumPy=np.__version__,
        threads=torch.get_num_threads(), device=str(device), default_dtype=str(torch.get_default_dtype()),
        AMP=torch.is_autocast_enabled(device.type), TF32_matmul=torch.backends.cuda.matmul.allow_tf32,
        TF32_cudnn=torch.backends.cudnn.allow_tf32, matmul_precision=torch.get_float32_matmul_precision(),
        deterministic=torch.are_deterministic_algorithms_enabled())


def validate_input_metadata(z, q, assignment, initial_parameters, phi, options, context, features):
    """Check actual input bytes/files; future adapter must validate admission report semantics."""
    refs, assets, layout = validate_context(context)
    _require(type(features) is ResidentCenteredTraceFeatures and features.descriptor() == refs["joint_features"],
        "Only the exact owning centered-trace provider is admitted")
    _require(features.metric_descriptor() == context["metric"], "Frozen metric descriptor differs")
    n, k, r = (refs[x] for x in ("nodes", "cells", "rank"))
    _matrix(z, (n, layout.physical_dimension)); _matrix(q, (n, layout.classes), z.device)
    _require(not z.requires_grad and not q.requires_grad and z.device.type in ("cpu", "cuda")
        and bool((q >= 0).all()) and bool((q.sum(1) > 0).all()) and bool((q.sum(0) > 0).all()),
        "Original z/raw Q frozen general-target domain differs")
    _require(torch.is_tensor(assignment) and assignment.shape == (n,) and assignment.dtype == torch.int64
        and assignment.device == z.device and not assignment.requires_grad
        and int(assignment.min()) >= 0 and int(assignment.max()) == k-1,
        "Original hard-cell order differs")
    data = array_digest(z.detach().cpu().numpy(), q.detach().cpu().numpy(), assignment.cpu().numpy())
    old = _native_options(options)
    _require(data == refs["data_digest"] and old == refs["original_options"]
        and ("data_digest" not in options or options["data_digest"] == data)
        and options["assignment_rank"] == r and options["factor_seed"] == refs["factor_seed"]
        and options["chunk_size"] == refs["chunk_size"] and refs["device"] == str(z.device)
        and _runtime(z.device) == refs["runtime"] and torch.get_default_dtype() == torch.float32
        and not torch.is_autocast_enabled(z.device.type), "Source options/runtime/data changed")
    for name, value in (("z", z), ("Q", q), ("assignment", assignment)):
        _require(_tensor_identity(value) == assets[name], "Original input array bytes changed")
    _require(_factor_digests(initial_parameters, n, k, r, device=z.device)
        == context["native_parameter_digests"] and bool(initial_parameters[0].eq(0).all()),
        "Original native zeroU/GaussianV identity changed")
    if torch.is_tensor(phi):
        _matrix(phi, (n, layout.original_phi_basis), torch.device("cpu"))
        _require(not phi.requires_grad, "Original Phi must be detached")
    else:
        _require(isinstance(phi, np.ndarray) and phi.shape == (n, layout.original_phi_basis)
            and phi.dtype == np.float64 and not phi.flags.writeable and bool(np.isfinite(phi).all()),
            "Original Phi must be read-only CPU FP64 [N,B]")
    _require(_content_digest(phi, canonical_double=True) == assets["Phi_identity"]["phi_digest"],
        "Original Phi content changed")
    _require(n*(layout.critic_dimension+layout.classes)*8 <= RESIDENT_SOURCE_MAX_BYTES,
        "Logical resident source payload exceeds 512MiB; whole-process peaks remain unqualified")
    paths, pins = refs.get("asset_paths"), refs.get("files_sha256")
    _require(isinstance(paths, dict) and set(paths) == {"H", "map", "Phi", "Phi_metadata"}
        and isinstance(pins, dict) and pins, "Original cache paths/full pins are required")
    for name in pins:
        _require(type(name) is str and Path(name).is_absolute()
            and str(Path(name).resolve()) == name, "Readonly pins must be normalized ABS")
    for path in paths.values():
        _require(type(path) is str and path in pins, "Original cache/sidecar path omitted")
    for value in [*refs["source_admission"].values(), *context["metric"]["calibration_refs"].values()]:
        _require(pins.get(value["path"]) == value["sha256"], "Source admission ref omitted or changed")
    _files(pins)
    _require(json.loads(Path(paths["Phi_metadata"]).read_text()) == assets["Phi_identity"],
        "Mandatory original Phi sidecar differs; no fallback is permitted")
    source = refs.get("current_source")
    _require(isinstance(source, dict) and set(source) == {"git_head", "source_digest", "files"},
        "Full current source manifest is required")
    _hex(source["git_head"], 40); _hex(source["source_digest"], 12); _files(source["files"])
    _require(refs["helper_source_sha256"] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "Joint helper source bytes changed")
    return dict(old, data_digest=data, centered_trace_joint_mean_mode=MODE,
        centered_trace_joint_mean_context=_copy(context))


def _decoded(moments, layout):
    _layout(layout)
    _require(torch.is_tensor(moments) and moments.ndim == 2 and len(moments) >= 2,
        "Joint moments must be a typed FP64 cell matrix")
    _matrix(moments, (len(moments), layout.material_width))
    centers, labels, mass = decode_moments(moments, layout.critic_dimension)
    _require(bool((mass > 0).all()) and bool(torch.isfinite(centers).all())
        and bool(torch.isfinite(labels).all()) and bool((labels >= 0).all())
        and bool((labels.sum(1) > 0).all()) and bool((labels.sum(0) > 0).all()),
        "Joint positive-mass/general-target domain differs")
    return centers, labels, mass


def joint_centroids(moments, layout):
    """Decode metric-derived joint means; no additional centering or scaling."""
    return _decoded(moments, layout)[0]


def raw_moment_cotangent(moments, layout, theta, vector, penalty):
    """Raw-CE G for DIRECT derived moments using original generic math at F."""
    _decoded(moments, layout)
    _matrix(theta, (layout.classes, layout.critic_dimension+1), moments.device)
    _matrix(vector, theta.shape, moments.device); _num(penalty, positive=True)
    value = moment_gradient(moments, layout.critic_dimension, lambda centers: centers,
        theta.detach(), vector.detach(), penalty, inner_loss_weighting="uniform")
    return _matrix(value.detach(), moments.shape, moments.device)


def complete_moment_cotangent(moments, layout, theta, vector, penalty, CE0):
    """Whole raw G divided by this single head's immutable positive CE0 once."""
    _num(CE0, positive=True)
    value = raw_moment_cotangent(moments, layout, theta, vector, penalty) / CE0
    return _matrix(value.detach(), moments.shape, moments.device)
