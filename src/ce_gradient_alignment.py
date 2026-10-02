"""Pure frozen-parameter CE gradient-direction moment partials.

No model updates, source loading or assignment backward occurs here. Source
GCN gradient targets are detached. Only the identity-adjacency synthetic CE
parameter gradients retain a graph through a fresh moment leaf.
"""
from collections.abc import Mapping

import torch

from src.finite_student_outer_v2 import (
    _adjacency,
    _float_tensor,
    _parameters,
    _precision,
    _probabilities,
    _snapshot,
    _stop,
    decode_rms_moments,
    functional_forward,
)

RHO = 1e-3
BLOCKS = ("W1", "b1", "W2", "b2")
POLICY = dict(
    rho=RHO, blocks=BLOCKS, block_weighting="equal1/4", ensemble_weighting="equal1/E",
    smoothing="symmetric sqrt(norm_syn²+delta²)*sqrt(norm_target²+delta²)",
    delta="rho*positive_target_norm; no clamp or block removal",
    source_route="original GCN linear→S→bias; dropout0; uniform teacher CE",
    synthetic_route="P-derived inverse-RMS Hbar/Qbar; identity; uniform1/K CE",
    source_Q="retain frozen FP64", synthetic_conversion="Hbar/Qbar to model dtype",
    direction_accumulation="FP64", classifier_updates=0,
    custom_moment_double_backward=False,
)


def _frozen_parameters(parameters):
    dimensions = _parameters(parameters)
    if any(value.requires_grad for value in parameters):
        raise ValueError("Frozen initial parameters must not require gradients")
    return dimensions


def _native_precision(device):
    _precision()
    if device.type == "cuda" and torch.backends.cuda.matmul.allow_tf32:
        raise ValueError("Gradient alignment requires TF32 disabled")


def _norm64(value):
    # Norm/dot arithmetic is explicitly independent of model precision.
    return torch.linalg.vector_norm(value.to(torch.float64).reshape(-1))


def _margin(parameters, x, adjacency=None):
    preactivation = x @ parameters[0].T
    if adjacency is not None:
        preactivation = (adjacency @ preactivation if adjacency.layout == torch.strided
                         else torch.sparse.mm(adjacency, preactivation))
    preactivation = preactivation + parameters[1]
    return dict(minimum_absolute_preactivation=preactivation.detach().abs().min().clone(),
                exact_zero_preactivations=int((preactivation.detach() == 0).sum()))


@torch.enable_grad()
def source_gradient_targets(ensemble, source_x, source_adjacency, source_q, *, stop=lambda: False):
    """Cache all four positive-norm gradient blocks at immutable model states.

    The caller supplies the ensemble; this mathematical helper selects no
    seeds or ensemble size. Sparse source S is accepted by the protected
    functional forward, but native sparse/deterministic support is unproved.
    """
    _stop(stop)
    if not isinstance(ensemble, (tuple, list)) or not ensemble:
        raise ValueError("Require a nonempty frozen parameter ensemble")
    nin, classes, device, dtype = _frozen_parameters(ensemble[0])
    _native_precision(device)
    _float_tensor(source_x, "frozen original X", device=device, dtype=dtype)
    if source_x.ndim != 2 or source_x.shape[1] != nin or not len(source_x):
        raise ValueError("Original X shape differs")
    _float_tensor(source_q, "frozen source Q", shape=(len(source_x), classes),
                  device=device, dtype=torch.float64)
    _probabilities(source_q, "frozen source Q")
    _adjacency(source_adjacency, len(source_x), device, dtype)
    if source_x.requires_grad or source_q.requires_grad:
        raise ValueError("Source X/Q must be immutable frozen inputs")
    anchors = []
    for initial in ensemble:
        _stop(stop)
        if _frozen_parameters(initial) != (nin, classes, device, dtype):
            raise ValueError("Ensemble dimensions/precision differ")
        parameters = tuple(value.detach().clone().requires_grad_(True) for value in initial)
        logp = functional_forward(parameters, source_x, source_adjacency)
        ce = -(source_q * logp).sum(1).mean()
        gradients = torch.autograd.grad(ce, parameters)
        norms = tuple(_norm64(value) for value in gradients)
        if any(not bool(torch.isfinite(norm)) or float(norm) <= 0 for norm in norms):
            raise ValueError("Every source gradient block must have finite positive norm")
        anchors.append(dict(parameters=_snapshot(initial), gradients=_snapshot(gradients),
                            source_ce=ce.detach().clone(), source_norms=_snapshot(norms),
                            delta=_snapshot(tuple(RHO * norm for norm in norms)),
                            source_relu=_margin(initial, source_x, source_adjacency)))
    _stop(stop)
    return dict(policy=dict(POLICY), anchors=tuple(anchors))


def smoothed_block_alignment(synthetic, target):
    """One fixed four-block symmetric smoothed cosine loss and diagnostics.

    Targets must be detached. Zero synthetic blocks are allowed and have a
    finite derivative; zero source blocks are rejected, never discarded.
    """
    if (not isinstance(synthetic, (tuple, list)) or not isinstance(target, (tuple, list))
            or len(synthetic) != 4 or len(target) != 4):
        raise ValueError("Require exactly four synthetic/target blocks")
    scores, diagnostics = [], []
    for name, value, fixed in zip(BLOCKS, synthetic, target, strict=True):
        _float_tensor(value, f"synthetic {name}")
        _float_tensor(fixed, f"target {name}", shape=value.shape, device=value.device, dtype=value.dtype)
        if fixed.requires_grad:
            raise ValueError("Target gradient blocks must be detached")
        s, t = value.to(torch.float64).reshape(-1), fixed.to(torch.float64).reshape(-1)
        target_norm, synthetic_norm = _norm64(fixed), _norm64(value)
        if not bool(torch.isfinite(target_norm)) or float(target_norm) <= 0:
            raise ValueError("Every source gradient block must have finite positive norm")
        if not bool(torch.isfinite(synthetic_norm)):
            raise ValueError("Synthetic gradient norm is nonfinite")
        delta = RHO * target_norm
        if not bool(torch.isfinite(delta)) or float(delta) <= 0:
            raise ValueError("Smoothing delta must be finite and positive")
        numerator = torch.dot(s, t)
        denominator = (s.square().sum() + delta.square()).sqrt() * (t.square().sum() + delta.square()).sqrt()
        if not bool(torch.isfinite(denominator)) or float(denominator.detach()) <= 0:
            raise ValueError("Smoothed direction denominator must be finite and positive")
        score = numerator / denominator
        _float_tensor(score, f"smoothed {name} cosine", shape=())
        scores.append(score)
        raw = numerator / (synthetic_norm * target_norm) if float(synthetic_norm.detach()) > 0 else None
        diagnostics.append(dict(block=name, synthetic_norm=synthetic_norm.detach().clone(),
                                target_norm=target_norm.detach().clone(), delta=delta.detach().clone(),
                                smoothed_cosine=score.detach().clone(),
                                raw_cosine=None if raw is None else raw.detach().clone()))
    return 1 - torch.stack(scores).mean(), tuple(diagnostics)


@torch.enable_grad()
def gradient_alignment_partials(moments, transform, targets, *, stop=lambda: False):
    """Return detached J and dJ/dM; call original moment backward once outside.

    No gradients connect to the supplied moments, frozen parameters or cached
    source gradients. Feature AND target mass denominators remain connected
    through the protected decode on a new differentiable leaf.
    """
    _stop(stop)
    if (not isinstance(targets, Mapping) or set(targets) != {"policy", "anchors"}
            or targets["policy"] != POLICY or not isinstance(targets["anchors"], (tuple, list))
            or not targets["anchors"]):
        raise ValueError("Malformed fixed gradient target cache/policy")
    first = targets["anchors"][0]
    if not isinstance(first, Mapping) or "parameters" not in first:
        raise ValueError("Malformed gradient target anchor")
    nin, classes, device, dtype = _frozen_parameters(first["parameters"])
    _native_precision(device)
    _float_tensor(moments, "moments", device=device)
    leaf = moments.detach().clone().requires_grad_(True)
    hbar, qbar, mass = decode_rms_moments(leaf, nin, transform)
    if qbar.shape[1] != classes:
        raise ValueError("Moment target classes differ")
    inner_x, inner_q = hbar.to(dtype), qbar.to(dtype)
    losses, diagnostics = [], []
    for anchor in targets["anchors"]:
        _stop(stop)
        if (not isinstance(anchor, Mapping) or set(anchor) != {
                "parameters", "gradients", "source_ce", "source_norms", "delta", "source_relu"}
                or _frozen_parameters(anchor["parameters"]) != (nin, classes, device, dtype)):
            raise ValueError("Malformed gradient target anchor/dimensions")
        parameters = tuple(value.detach().clone().requires_grad_(True) for value in anchor["parameters"])
        ce = -(inner_q * functional_forward(parameters, inner_x)).sum(1).mean()
        gradients = torch.autograd.grad(ce, parameters, create_graph=True)
        loss, blocks = smoothed_block_alignment(gradients, anchor["gradients"])
        norms = tuple(_norm64(value).detach() for value in anchor["gradients"])
        if (not isinstance(anchor["source_norms"], (tuple, list)) or len(anchor["source_norms"]) != 4
                or not isinstance(anchor["delta"], (tuple, list)) or len(anchor["delta"]) != 4):
            raise ValueError("Malformed cached source norms/delta")
        for field in (anchor["source_norms"], anchor["delta"]):
            for scalar in field:
                _float_tensor(scalar, "cached norm/delta", shape=(), device=device, dtype=torch.float64)
                if scalar.requires_grad:
                    raise ValueError("Cached norm/delta must be detached")
        if (any(not torch.equal(a, b) for a, b in zip(norms, anchor["source_norms"], strict=True))
                or any(not torch.equal(RHO*a, b) for a, b in zip(norms, anchor["delta"], strict=True))):
            raise ValueError("Cached source norms/delta differ from actual gradient blocks")
        _float_tensor(anchor["source_ce"], "cached source CE", shape=(), device=device)
        losses.append(loss)
        diagnostics.append(dict(uniform_synthetic_ce=ce.detach().clone(), blocks=blocks,
                                synthetic_relu=_margin(parameters, inner_x)))
    loss = torch.stack(losses).mean()
    partial, = torch.autograd.grad(loss, leaf)
    _float_tensor(partial, "moment partial", shape=leaf.shape)
    _stop(stop)
    return dict(loss=loss.detach().clone(), moment_gradient=partial.detach().clone(),
                Hc=hbar.detach().clone(), Qc=qbar.detach().clone(), mass=mass.detach().clone(),
                anchors=tuple(diagnostics), policy=dict(POLICY))
