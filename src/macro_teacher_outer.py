"""Pure fixed-Q macro teacher CE and converged linear-head outer RHS.

Only the outer source risk changes. Q is the original frozen FP64 probability
matrix; class masses are positive, unclamped sums of its columns. Inner uniform
CE, P-derived centroids/targets and ridge/head stationarity remain the caller's
unchanged problem. This helper loads no source, fits no head and updates no P.
"""
import torch

POLICY = "equal_soft_teacher_class_outer_CE_v1"


def _matrix(value, name):
    if (not torch.is_tensor(value) or value.layout != torch.strided or value.dtype != torch.float64
            or value.ndim != 2 or min(value.shape) <= 0 or not bool(torch.isfinite(value).all())):
        raise ValueError(f"Require finite nonempty FP64 matrix: {name}")
    return value


def _stop(stop):
    if not callable(stop):
        raise ValueError("stop must be callable")
    if stop():
        raise InterruptedError("Macro teacher outer evaluation interrupted")


@torch.no_grad()
def macro_teacher_weights(q):
    """Return W=Q/(C*sum_nodes Q), class masses; never alter/renormalize Q."""
    _matrix(q, "original frozen teacher Q")
    if (q.requires_grad or bool((q < 0).any())
            or float((q.sum(1)-1).abs().max()) > 32*torch.finfo(q.dtype).eps):
        raise ValueError("Original teacher Q must be frozen full-class probabilities")
    class_mass = q.sum(0)
    if not bool(torch.isfinite(class_mass).all()) or not bool((class_mass > 0).all()):
        raise ValueError("Every soft teacher class requires a finite strictly positive mass; no clamp")
    weights = q / (q.shape[1]*class_mass)
    if not bool(torch.isfinite(weights).all()):
        raise ValueError("Nonfinite macro teacher weights")
    return weights, class_mass


@torch.no_grad()
def macro_teacher_outer(z, q, theta, *, chunk=4096, stop=lambda: False):
    """Return J=-sum W*logp and dJ/dtheta, class-major C×(D+1).

    Original RMS z has no appended bias column. This function appends ones and
    uses the unchanged linear logits X_aug@theta.T. W has row masses that need
    not be1/N: the analytic residual is p*W.sum(1)-W, NOT p-Q.
    """
    _stop(stop)
    if type(chunk) is not int or chunk <= 0:
        raise ValueError("chunk must be a positive native integer")
    _matrix(z, "original frozen source z")
    _matrix(theta, "class-major head theta")
    if z.requires_grad:
        raise ValueError("Original source z must be frozen")
    weights, _ = macro_teacher_weights(q)
    if (len(z) != len(q) or theta.shape != (q.shape[1], z.shape[1]+1)
            or z.device != q.device or theta.device != z.device):
        raise ValueError("Source/Q/head dimensions or devices differ")
    value, rhs = theta.new_zeros(()), torch.zeros_like(theta)
    for start in range(0, len(z), chunk):
        _stop(stop)
        features = z[start:start+chunk]
        x = torch.cat((features, features.new_ones(len(features), 1)), dim=1)
        target = weights[start:start+chunk]
        lp = (x @ theta.T).log_softmax(1)
        if not bool(torch.isfinite(lp).all()):
            raise ValueError("Nonfinite macro linear-head log probabilities")
        value -= (target*lp).sum()
        error = lp.exp()*target.sum(1, keepdim=True)-target
        rhs += error.T @ x
        _stop(stop)
    if not bool(torch.isfinite(value)) or not bool(torch.isfinite(rhs).all()):
        raise ValueError("Nonfinite macro teacher CE/RHS")
    return float(value), rhs
