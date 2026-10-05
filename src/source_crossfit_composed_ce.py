"""Fixed-parity two-fold source-cross-fit composed uniform-CE math only.

One shared all-N physical moment tape, two original full composed heads, one
averaged FP64 cotangent/CE0 before the original assignment pullback.
"""
import math
import torch

from src import composed_centroid_joint_ce as joint
from src.nystrom_ce import outer_gradient

MODE = "two_fixed_source_fold_crossfit_composed_uniform_CE_v1"


def _matrix(value, name):
    if (not torch.is_tensor(value) or value.layout != torch.strided or value.dtype != torch.float64
            or value.ndim != 2 or min(value.shape) <= 0 or not bool(torch.isfinite(value).all())):
        raise ValueError("Finite nonempty FP64 matrix required: " + name)
    return value


def pack_physical_material(z, q):
    """Return [even*(1,z,Q), odd*(1,z,Q)]; original forward divides by ALL N."""
    _matrix(z, "z"); _matrix(q, "raw Q")
    if (len(z) != len(q) or len(z) < 2 or z.device != q.device or z.requires_grad or q.requires_grad
            or bool((q < 0).any()) or not bool((q.sum(1) > 0).all()) or not bool((q.sum(0) > 0).all())):
        raise ValueError("Fixed-source general-Q physical material domain differs")
    physical = torch.cat((z.new_ones(len(z), 1), z, q), 1)
    even = (torch.arange(len(z), device=z.device) % 2 == 0).to(z.dtype)[:, None]
    return torch.cat((even * physical, (1-even) * physical), 1).detach()


@torch.no_grad()
def held_outer(source_features, q, theta, held_fold, *, chunk=65536, stop=lambda: False):
    """Fixed held source raw CE/RHS; A even, B odd; mean by ACTUAL held count."""
    _matrix(source_features, "fixed source features"); _matrix(q, "raw source Q"); _matrix(theta, "theta")
    if (held_fold not in ("A", "B") or type(chunk) is not int or chunk < 1 or not callable(stop)
            or len(source_features) != len(q) or len(q) < 2 or source_features.requires_grad or q.requires_grad
            or source_features.device.type != "cpu" or source_features.device != q.device or theta.device != q.device
            or theta.shape != (q.shape[1], source_features.shape[1]+1)
            or bool((q < 0).any()) or not bool((q.sum(1) > 0).all())):
        raise ValueError("Held source/head/fold domain differs")
    if stop(): raise InterruptedError("Held source stopped")
    index = torch.arange(0 if held_fold == "A" else 1, len(q), 2, device=q.device)
    features, targets = source_features.index_select(0, index), q.index_select(0, index)
    # This selected helper is CPU-only; future native source-row staging is a separate admission.
    rows = features.detach().numpy(); rows.setflags(write=False)
    value, rhs = outer_gradient(rows, targets, theta, chunk=min(chunk, len(index)))
    return value, rhs.detach()


def complete_packed_cotangent(moments, layout, composed, theta_A, theta_B, vector_A, vector_B,
                              penalty, CE0, *, map_call=None, observe_return=None):
    """Exactly two original general-Q raw cotangents, .5 concat / own CE0 ONCE."""
    if type(layout) is not joint.ComposedJointLayout:
        raise ValueError("Exact original composed physical layout required")
    _matrix(moments, "packed physical moments")
    width = layout.material_width
    if (moments.shape[1] != 2*width or not isinstance(CE0, (int, float)) or isinstance(CE0, bool)
            or not math.isfinite(CE0) or CE0 <= 0 or observe_return is not None and not callable(observe_return)):
        raise ValueError("Packed layout/own CE0/observation domain differs")
    raw_A = joint.raw_moment_cotangent(moments[:, :width], layout, composed, theta_A,
                                      vector_A, penalty, map_call=map_call)
    if observe_return is not None: observe_return("raw_A", raw_A)
    raw_B = joint.raw_moment_cotangent(moments[:, width:], layout, composed, theta_B,
                                      vector_B, penalty, map_call=map_call)
    if observe_return is not None: observe_return("raw_B", raw_B)
    value = (.5 * torch.cat((raw_A, raw_B), 1) / CE0).detach()
    if observe_return is not None: observe_return("complete", value)
    return _matrix(value, "complete packed physical cotangent")
