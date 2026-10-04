"""EY source-view ownership and metadata; full centered inner math is imported.

This adapter has no head solve, optimizer, source load, or numerical admission.
An inner_contract is a labelled centered INNER data/math supplier only. It is
never an EX trajectory, accepted EX head, or EX CE0 in the new EY envelope.
"""
import hashlib
from pathlib import Path

import numpy as np
import torch

from src import centered_trace_joint_mean_ce as inner
from src.dual_head_ce import _files
from src.nystrom_ce import _content_digest
from src.shared_features import _tensor_identity

SCHEMA = 1
MODE = "centered_joint_inner_physical_only_source_outer_uniform_CE_v1"
BACKEND = "owning_fullwidth_physical_prefix_positive_zero_Phi_source_outer_v1"
POLICY = dict(schema=SCHEMA, objective="physical_only_source_raw_CE_over_own_positive_CE0",
    inner="unchanged_full_centered_trace_joint_uniform_CE_all_bias_ridge",
    outer="literal_fullwidth_scaled_z_positive_zero_Phi_trailing_bias",
    full_joint_adjoint_and_G=True, forced_nonzero_Phi_coupling=False,
    physical_material_forward="separate_original_BLAS_shape", original_native_cast=True)

# Unchanged numerical helpers and typed inner owners are explicit suppliers.
CenteredTraceJointLayout = inner.CenteredTraceJointLayout
FrozenCenteredTraceMetric = inner.FrozenCenteredTraceMetric
ResidentCenteredTraceFeatures = inner.ResidentCenteredTraceFeatures
raw_moment_cotangent = inner.raw_moment_cotangent
complete_moment_cotangent = inner.complete_moment_cotangent
_matrix, _num, _hex, _runtime = inner._matrix, inner._num, inner._hex, inner._runtime


class FrozenPhysicalOnlySourceOuterView:
    """Own [a_z*z, positive-zero Phi] separately from the full inner provider."""
    def __init__(self, inner_provider, layout):
        inner._layout(layout)
        inner._require(type(inner_provider) is ResidentCenteredTraceFeatures,
            "Exact full centered inner provider is required")
        supplied = inner_provider.descriptor()
        inner._require(supplied["layout"] == layout.descriptor(), "Inner layout differs")
        rows = inner_provider.outer_rows()
        self._rows = np.array(rows, copy=True, order="C")
        d, f = layout.physical_dimension, layout.critic_dimension
        self._rows[:, d:f] = 0.0
        self._rows.flags.writeable = False
        self._inner = inner_provider
        self._layout = layout
        self._inner_descriptor = inner._copy(supplied)
        self._prefix = _tensor_identity(torch.from_numpy(np.array(rows[:, :d], copy=True, order="C")))
        self._identity = _tensor_identity(torch.from_numpy(self._rows.copy()))
        self._digest = _content_digest(self._rows, canonical_double=True)
        self._verify()

    def _verify(self):
        rows, layout = self._rows, self._layout
        supplied = self._inner.descriptor()
        inner._require(supplied == self._inner_descriptor, "Full inner supplier changed")
        inner._require(rows.dtype == np.float64 and rows.flags.c_contiguous
            and rows.flags.owndata and not rows.flags.writeable and bool(np.isfinite(rows).all())
            and rows.shape == tuple(supplied["shape"]), "Owning masked source domain differs")
        d, f = layout.physical_dimension, layout.critic_dimension
        inner._require(bool((rows[:, d:f] == 0).all()) and not bool(np.signbit(rows[:, d:f]).any()),
            "Source outer Phi block must be literal positive zero")
        prefix = _tensor_identity(torch.from_numpy(np.array(rows[:, :d], copy=True, order="C")))
        source_prefix = _tensor_identity(torch.from_numpy(np.array(self._inner.outer_rows()[:, :d], copy=True, order="C")))
        inner._require(prefix == source_prefix == self._prefix
            and _content_digest(rows, canonical_double=True) == self._digest
            and _tensor_identity(torch.from_numpy(rows.copy())) == self._identity,
            "Masked source content or physical prefix changed")

    def descriptor(self):
        self._verify()
        value = dict(schema=SCHEMA, kind=BACKEND, shape=list(self._rows.shape), dtype="float64",
            layout=self._layout.descriptor(), inner_supplier_kind="centered_INNER_metadata_and_material_only",
            inner_provider_descriptor_digest=self._inner_descriptor["descriptor_digest"],
            inner_joint_digest=self._inner_descriptor["joint_digest"],
            metric_content_digest=self._inner_descriptor["metric_content_digest"],
            physical_prefix=inner._copy(self._prefix), positive_zero_Phi_range=[self._layout.physical_dimension,
                self._layout.critic_dimension], masked_tensor_identity=inner._copy(self._identity),
            masked_content_digest=self._digest, owning_readonly_CPU=True)
        return dict(value, descriptor_digest=inner._seal(value))

    def outer_rows(self):
        self._verify()
        rows = self._rows.copy(); rows.flags.writeable = False
        return rows


def _view_descriptor(value, refs, layout):
    keys = {"schema", "kind", "shape", "dtype", "layout", "inner_supplier_kind",
        "inner_provider_descriptor_digest", "inner_joint_digest", "metric_content_digest",
        "physical_prefix", "positive_zero_Phi_range", "masked_tensor_identity",
        "masked_content_digest", "owning_readonly_CPU", "descriptor_digest"}
    inner._require(isinstance(value, dict) and set(value) == keys and type(value["schema"]) is int
        and value["schema"] == SCHEMA and value["kind"] == BACKEND and value["dtype"] == "float64"
        and value["shape"] == [refs["nodes"], layout.critic_dimension]
        and all(type(v) is int for v in value["shape"])
        and value["layout"] == layout.descriptor()
        and value["inner_supplier_kind"] == "centered_INNER_metadata_and_material_only"
        and value["owning_readonly_CPU"] is True
        and value["positive_zero_Phi_range"] == [layout.physical_dimension, layout.critic_dimension]
        and all(type(v) is int for v in value["positive_zero_Phi_range"]), "EY outer descriptor policy/layout differs")
    supplied = refs["joint_features"]
    inner._require(value["inner_provider_descriptor_digest"] == supplied["descriptor_digest"]
        and value["inner_joint_digest"] == supplied["joint_digest"]
        and value["metric_content_digest"] == supplied["metric_content_digest"], "Inner supplier seal differs")
    inner._identity(value["physical_prefix"], (refs["nodes"], layout.physical_dimension))
    inner._identity(value["masked_tensor_identity"], (refs["nodes"], layout.critic_dimension))
    for name in ("masked_content_digest", "descriptor_digest"):
        inner._hex(value[name])
    inner._require(value["descriptor_digest"] == inner._seal({k:v for k,v in value.items()
        if k != "descriptor_digest"}), "EY outer descriptor seal differs")


def _binding(context):
    value = inner._copy(context)
    value["outer_contract"].pop("binding_digest", None)
    return inner._seal(value)


def validate_context(context):
    """Metadata consistency; actual source/calibration admission belongs to driver."""
    inner._require(isinstance(context, dict) and set(context) == {"schema", "mode", "policy",
        "inner_contract", "outer_contract"} and type(context["schema"]) is int
        and context["schema"] == SCHEMA and context["mode"] == MODE
        and inner._seal(context["policy"]) == inner._seal(POLICY),
        "EY context schema/mode differs")
    inner._plain(context)
    refs, assets, layout = inner.validate_context(context["inner_contract"])
    outer = context["outer_contract"]
    inner._require(isinstance(outer, dict) and set(outer) == {"descriptor", "helper_source_sha256", "binding_digest"},
        "EY outer contract fields differ")
    _view_descriptor(outer["descriptor"], refs, layout)
    inner._hex(outer["helper_source_sha256"]); inner._hex(outer["binding_digest"])
    inner._require(outer["binding_digest"] == _binding(context), "Full EY inner/outer binding differs")
    return refs, assets, layout


def build_context(inner_contract, outer_view_descriptor, outer_helper_source_sha256):
    value = dict(schema=SCHEMA, mode=MODE, policy=inner._copy(POLICY),
        inner_contract=inner._copy(inner_contract), outer_contract=dict(descriptor=inner._copy(outer_view_descriptor),
            helper_source_sha256=outer_helper_source_sha256))
    value["outer_contract"]["binding_digest"] = _binding(value)
    validate_context(value)
    return inner._copy(value)


def validate_input_metadata(z, q, assignment, initial_parameters, phi, options, context, features, outer_view):
    refs, _, layout = validate_context(context)
    config = inner.validate_input_metadata(z, q, assignment, initial_parameters, phi, options,
        context["inner_contract"], features)
    inner._require(type(outer_view) is FrozenPhysicalOnlySourceOuterView
        and outer_view.descriptor() == context["outer_contract"]["descriptor"], "Actual masked owner differs")
    inner._require(context["outer_contract"]["helper_source_sha256"]
        == hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "EY helper bytes differ")
    inner._require(refs["nodes"]*(2*layout.critic_dimension+layout.classes)*8
        <= inner.RESIDENT_SOURCE_MAX_BYTES, "Combined inner source and outer view exceed512MiB")
    _files(refs["files_sha256"])
    # No EX mode/context is persisted as this trajectory's config identity.
    config.pop("centered_trace_joint_mean_mode")
    config.pop("centered_trace_joint_mean_context")
    return dict(config, joint_inner_physical_only_source_outer_mode=MODE,
        joint_inner_physical_only_source_outer_context=inner._copy(context))


def validate_direct_Phi_rhs(rhs, layout):
    """Only DIRECT source RHS is zero; full coupled adjoint/G are never masked."""
    inner._layout(layout)
    inner._matrix(rhs, (layout.classes, layout.critic_dimension+1))
    inner._require(bool(rhs[:, layout.physical_dimension:layout.critic_dimension].eq(0).all()),
        "Direct source Phi RHS must be exactly zero")
    return True
