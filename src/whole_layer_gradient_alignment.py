"""Isolated two-whole-layer direction objective; no source loading or P update.

Preserve CE/decode/P-chain arithmetic; change only the gradient grouping.
Future native cache/backend/optimizer acceptance is outside this CPU proof.
"""
import torch

from src.finite_student_outer_v2 import (
    _float_tensor,
    _parameters,
    _precision,
    decode_rms_moments,
    functional_forward,
)

RHO = .001
LAYERS = ((0, 1), (2, 3))
POLICY = dict(rho=RHO, layer_blocks=['concat(vec(W1),b1)', 'concat(vec(W2),b2)'],
              layer_weighting='equal1/2', anchor_weighting='equal1/3', anchors=3,
              accumulation='FP64', smoothing='symmetric', target_gradients='detached',
              clipping=False, renormalization=False, dropped_blocks=False)


def whole_layer_alignment(synthetic, target):
    """One anchor; concatenate a layer weight and bias before normalization."""
    if _parameters(synthetic) != _parameters(target):
        raise ValueError('Synthetic/target shapes, dtype or device differ')
    if any(value.requires_grad for value in target):
        raise ValueError('Source target gradients must be detached')
    scores, diagnostics = [], []
    for layer, pair in enumerate(LAYERS, 1):
        s = torch.cat([synthetic[i].to(torch.float64).reshape(-1) for i in pair])
        t = torch.cat([target[i].to(torch.float64).reshape(-1) for i in pair])
        sn, tn = torch.linalg.vector_norm(s), torch.linalg.vector_norm(t)
        if not bool(torch.isfinite(tn)) or float(tn) <= 0:
            raise ValueError('Every whole-layer source norm must be finite and positive')
        if not bool(torch.isfinite(sn)):
            raise ValueError('Whole-layer synthetic norm must be finite')
        delta = RHO * tn
        if not bool(torch.isfinite(delta)) or float(delta) <= 0:
            raise ValueError('Layer smoothing must be finite and positive')
        denominator = (s.square().sum()+delta.square()).sqrt() * (t.square().sum()+delta.square()).sqrt()
        if not bool(torch.isfinite(denominator)) or float(denominator.detach()) <= 0:
            raise ValueError('Layer denominator must be finite and positive')
        score = torch.dot(s, t) / denominator
        _float_tensor(score, 'whole-layer cosine', shape=())
        scores.append(score)
        diagnostics.append(dict(layer=layer, synthetic_norm=sn.detach().clone(), target_norm=tn.detach().clone(),
            delta=delta.detach().clone(), smoothed_cosine=score.detach().clone(),
            raw_cosine=None if float(sn.detach()) == 0 else (torch.dot(s,t)/(sn*tn)).detach().clone()))
    return 1-torch.stack(scores).mean(), tuple(diagnostics)


def whole_layer_ensemble_alignment(synthetic, targets):
    if (not isinstance(synthetic, (tuple,list)) or not isinstance(targets, (tuple,list))
            or len(synthetic) != 3 or len(targets) != 3):
        raise ValueError('Exactly three anchors required')
    values = [whole_layer_alignment(s,t) for s,t in zip(synthetic, targets, strict=True)]
    return torch.stack([v[0] for v in values]).mean(), tuple(v[1] for v in values)


@torch.enable_grad()
def moment_partials(moments, transform, ensemble, target_gradients):
    """Detach M; differentiate synthetic CE gradients and return only dJ/dM.

    The caller invokes the original first LowRankMoments backward outside.
    This toy API accepts caller-supplied frozen anchors/targets; it does not
    qualify a native source target cache or choose initialization seeds.
    """
    _precision()
    if not isinstance(ensemble, (tuple,list)) or len(ensemble) != 3:
        raise ValueError('Exactly three frozen anchors required')
    nin, classes, device, dtype = _parameters(ensemble[0])
    _float_tensor(moments, 'FP64 moments', device=device, dtype=torch.float64)
    leaf = moments.detach().clone().requires_grad_(True)
    h, q, mass = decode_rms_moments(leaf, nin, transform)
    if q.shape[1] != classes:
        raise ValueError('Moment target classes differ')
    synthetic, masks, CE = [], [], []
    for initial in ensemble:
        if _parameters(initial) != (nin, classes, device, dtype) or any(p.requires_grad for p in initial):
            raise ValueError('Frozen anchor dimensions/precision differ')
        parameters = tuple(p.detach().clone().requires_grad_(True) for p in initial)
        value = -(q.to(dtype)*functional_forward(parameters,h.to(dtype))).sum(1).mean()
        synthetic.append(torch.autograd.grad(value,parameters,create_graph=True))
        masks.append((h.to(dtype)@parameters[0].T+parameters[1] > 0).detach())
        CE.append(value.detach().clone())
    loss, diagnostics = whole_layer_ensemble_alignment(tuple(synthetic),target_gradients)
    gradient, = torch.autograd.grad(loss,leaf)
    _float_tensor(gradient,'whole-layer moment partial',shape=leaf.shape)
    return dict(loss=loss.detach().clone(), moment_gradient=gradient.detach().clone(),
                masks=tuple(masks), synthetic_CE=tuple(CE), mass=mass.detach().clone(), layers=diagnostics,
                policy=dict(POLICY))
