"""Full-row stored-CSR segment SUM in fixed rank2 feature bands of 16.

The product is no-grad; dense isolated CE/VJP order matches the protected
whole-width segment helper. No row splitting, fallback, or dtype conversion.
"""
import torch
import torch.nn.functional as F

from src.csr_explicit_adjoint import (
    _begin,
    _csr,
    _end,
    _inputs,
    _invoke,
    _require,
    transpose_csr,
)
from src.csr_segment_adjoint import _product as _segment_product
from src.csr_segment_adjoint import target_packet as target_packet

BAND_WIDTH = 16


@torch.no_grad()
def _product(source, value, *, evidence=None, key=None):
    """Copy left-to-right disjoint full-row output bands without arithmetic."""
    key = "feature_band16_logical_products" if key is None else key
    _begin(evidence, key)
    _csr(source)
    _require(isinstance(value, torch.Tensor) and value.device == source.device
             and value.layout == torch.strided and value.ndim == 2
             and value.shape[0] == source.shape[1] and value.shape[1] > 0,
             "Require same-device rank2 input with matching rows")
    _require(value.dtype == source.dtype and not value.requires_grad
             and bool(torch.isfinite(value).all()), "Require frozen finite input with identical source dtype")
    result = _invoke(evidence, "feature_band16_output_allocations", torch.empty,
                     (source.shape[0], value.shape[1]), dtype=value.dtype, device=value.device)
    for start in range(0, value.shape[1], BAND_WIDTH):
        end = min(start + BAND_WIDTH, value.shape[1])
        band = _segment_product(source, value[:, start:end], evidence=evidence,
                                key="feature_band16_physical_products")
        _invoke(evidence, "feature_band16_output_copy_calls", result[:, start:end].copy_, band)
    _require(result.shape == (source.shape[0], value.shape[1]) and result.dtype == value.dtype
             and result.device == value.device and not result.requires_grad
             and bool(torch.isfinite(result).all()), "Nonfinite/malformed tiled result")
    _end(evidence, key)
    return result


def gradient_boundaries(parameters, x, source, q, evidence=None):
    """Compute all four parameter cotangents without sparse autograd backward."""
    _inputs(parameters, x, source, q)
    w1, b1, w2, b2 = parameters
    transposed = _invoke(evidence, "transpose_CSR_constructions", transpose_csr, source, evidence)
    _begin(evidence, "source_GCN_full_forward_passes")
    with torch.no_grad():
        u1 = x @ w1.T
        v1 = _product(source, u1, evidence=evidence, key="feature_band16_original_source_products")
        a = v1 + b1
        h = F.relu(a)
        u2 = h @ w2.T
        v2 = _product(source, u2, evidence=evidence, key="feature_band16_original_source_products")
        logits = v2 + b2
        full_logp = F.log_softmax(logits, dim=1)
        full_loss = _invoke(evidence, "source_CE_forward_evaluations", lambda: -(q * full_logp).sum(1).mean())
    _end(evidence, "source_GCN_full_forward_passes")
    leaf_logits = logits.detach().clone().requires_grad_(True)
    logp = F.log_softmax(leaf_logits, dim=1)
    loss = _invoke(evidence, "source_CE_forward_evaluations", lambda: -(q * logp).sum(1).mean())
    _require(torch.equal(full_logp, logp) and torch.equal(full_loss, loss), "No-grad/dense-leaf forward differs")
    d = _invoke(evidence, "dense_logits_CE_VJP_calls", torch.autograd.grad, loss, leaf_logits)[0].detach()
    b = _product(transposed, d, evidence=evidence, key="feature_band16_explicit_transpose_products")
    leaf_h, leaf_w2 = (value.detach().clone().requires_grad_(True) for value in (h, w2))
    dh, gw2 = _invoke(evidence, "dense_second_linear_VJP_calls", torch.autograd.grad, leaf_h @ leaf_w2.T, (leaf_h, leaf_w2), grad_outputs=b)
    leaf_v2, leaf_b2 = (value.detach().clone().requires_grad_(True) for value in (v2, b2))
    gb2 = _invoke(evidence, "dense_second_bias_VJP_calls", torch.autograd.grad, leaf_v2 + leaf_b2, leaf_b2, grad_outputs=d)[0]
    leaf_v1, leaf_b1 = (value.detach().clone().requires_grad_(True) for value in (v1, b1))
    e, gb1 = _invoke(evidence, "dense_first_ReLU_bias_VJP_calls", torch.autograd.grad, F.relu(leaf_v1 + leaf_b1), (leaf_v1, leaf_b1), grad_outputs=dh.detach())
    c = _product(transposed, e.detach(), evidence=evidence, key="feature_band16_explicit_transpose_products")
    leaf_w1 = w1.detach().clone().requires_grad_(True)
    gw1 = _invoke(evidence, "dense_first_linear_VJP_calls", torch.autograd.grad, x @ leaf_w1.T, leaf_w1, grad_outputs=c)[0]
    gradients = tuple(value.detach().clone() for value in (gw1, gb1, gw2, gb2))
    _require(bool(torch.isfinite(loss)) and all(bool(torch.isfinite(value).all()) for value in gradients),
             "Nonfinite CE or gradient")
    _begin(evidence, "explicit_gradient_block_outputs", 4)
    _end(evidence, "explicit_gradient_block_outputs", 4)
    boundaries = dict(U1=u1, V1=v1, A=a, mask=a > 0, H=h, U2=u2, V2=v2, logits=logits,
                      logp=logp, full_logp=full_logp, full_CE=full_loss, D=d, B=b, DH=dh, E=e, C=c)
    return dict(loss=loss.detach().clone(), gradients=gradients,
                boundaries={key: value.detach().clone() for key, value in boundaries.items()},
                parameters=tuple(value.detach().clone() for value in parameters), transpose=transposed,
                sparse_forward_products=0, coefficient_feature_band16_logical_products=4, sparse_backward_calls=0, parameter_updates=0)
