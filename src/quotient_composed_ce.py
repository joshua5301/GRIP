"""CPU FP64 P-derived quotient/composed features and complete row cotangent.

No head fit, adjoint solve, source fit or native/GPU admission is performed here.
The fixed original source and caller's own CE0 remain caller responsibilities.
"""
import math

import torch

from src.coarsening import feature_centroids, quotient_adjacency
from src.moments import augmented
from src.soft_ce_partition import head_gradient
from src.transforms import FeatureTransform
from src.nystrom_ce import NystromMap

MODE = 'P_quotient_original_RMS_Nystrom_composed_uniform_CE_CPU_v1'


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _matrix(value, shape=None):
    _require(torch.is_tensor(value) and value.dtype == torch.float64 and value.device.type == 'cpu'
             and value.layout == torch.strided and value.ndim == 2
             and (shape is None or tuple(value.shape) == tuple(shape))
             and bool(torch.isfinite(value).all()), 'Require finite dense CPU FP64 matrix')
    return value


def _observe(callback, name, value):
    if callback is not None:
        callback(name, value)
    return value


def build_quotient_features(P, X, Q, S, transform, feature_map, observe_return=None):
    """One original quotient, two centroid calls and one original frozen map.

    C/degree are explicit diagnostic expressions, since the unchanged original
    quotient API returns Sc only. They introduce no second quotient API call.
    All newly returned tensors are observable before this helper's next guard.
    """
    _matrix(P); n, k = P.shape; _matrix(X); _matrix(Q)
    _require(n >= 2 and k >= 2 and X.shape[0] == Q.shape[0] == n, 'Dimensions differ')
    _matrix(S, (n, n))
    _require(bool((P > 0).all()) and torch.allclose(P.sum(1), P.new_ones(n), atol=1e-12, rtol=1e-12)
             and bool((Q > 0).all()) and bool((S >= 0).all()), 'Positive fixed general-Q/graph domain differs')
    _require(type(transform) is FeatureTransform and transform.kind == 'rms' and transform.matrix is None
             and type(feature_map) is NystromMap and feature_map.kernel == 'relu', 'Original RMS/map types differ')
    _require(all(torch.is_tensor(t) and t.device.type == 'cpu' and t.dtype == torch.float64
                 and not t.requires_grad and bool(torch.isfinite(t).all())
                 for t in (transform.center, transform.output_center, transform.scale,
                           feature_map.anchors, feature_map.mapping)), 'Frozen original map/RMS domain differs')
    _require(tuple(transform.center.shape) == tuple(transform.output_center.shape) == (X.shape[1],)
             and transform.scale.numel() == 1 and float(transform.scale) > 0
             and math.isfinite(transform.eps) and transform.eps == 1e-12, 'RMS fields differ')
    Xc, mass = feature_centroids(P, X); _observe(observe_return, 'Xc_mass', (Xc, mass))
    Qc, target_mass = feature_centroids(P, Q); _observe(observe_return, 'Qc_mass', (Qc, target_mass))
    Sc = quotient_adjacency(P, S, normalization='symmetric', mass_scaling=False,
                            self_loops='retain', chunk_size=k)
    _observe(observe_return, 'Sc', Sc)
    C = P.T @ (S @ P); degree = C.sum(1)
    _observe(observe_return, 'C_degree', (C, degree))
    Y = Sc @ Xc; Hc = Sc @ Y; zc = transform(Hc)
    _observe(observe_return, 'propagated', (Y, Hc, zc))
    map_input = zc * transform.scale + transform.output_center + transform.center
    _observe(observe_return, 'map_input', map_input)
    mapped = feature_map(map_input); _observe(observe_return, 'map_return', mapped)
    features = torch.cat((zc, mapped), 1)
    value = dict(n=mass, Xc=Xc, Qc=Qc, C=C, degree=degree, Sc=Sc,
                 Hc=Hc, zc=zc, map_input=map_input, features=features)
    _observe(observe_return, 'quotient_features', value)
    _require(bool((mass > 0).all()) and bool((degree > 0).all()) and torch.equal(mass, target_mass)
             and all(bool(torch.isfinite(t).all()) for t in value.values()), 'Returned quotient domain failed')
    return value


def complete_row_cotangent(P, X, Q, S, transform, feature_map, theta, vector,
                           penalty, CE0, observe_return=None):
    """Full shared-P derivative; one map/quotient, no solver or separate casts."""
    _matrix(P); _matrix(theta); _matrix(vector, theta.shape)
    _require(math.isfinite(penalty) and penalty > 0 and math.isfinite(CE0) and CE0 > 0,
             'Require original positive ridge and own CE0')
    variable = P.detach().clone().requires_grad_(True)
    _observe(observe_return, 'P_leaf', variable)
    value = build_quotient_features(variable, X, Q, S, transform, feature_map, observe_return)
    weights = variable.new_full((len(variable.T),), 1 / len(variable.T))
    gradient = head_gradient(augmented(value['features']), value['Qc'], weights,
                             theta.detach(), penalty)
    _observe(observe_return, 'fixed_head_gradient', gradient)
    raw = torch.autograd.grad(-(gradient * vector.detach()).sum(), variable)[0]
    _observe(observe_return, 'raw_R', raw)
    complete = raw / CE0; _observe(observe_return, 'complete_R', complete)
    result = dict(raw_R=raw.detach(), complete_R=complete.detach(), owned_chain=value)
    _observe(observe_return, 'complete_row_result', result)
    _matrix(raw, P.shape); _matrix(complete, P.shape)
    return result
