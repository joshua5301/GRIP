"""Differentiable graph quotients for the existing assignment models.

Rows of ``probability`` are nodes and columns are condensed cells.  For a
row-stochastic P and an adjacency A, the unnormalized quotient is C=P.T @ A @ P.
Every original edge contributes its weight fractionally to cell pairs.  Dense
soft assignments generally produce a dense quotient; no threshold is applied.

Use original X for ``feature_centroids`` when training a GCN on the quotient.
Using the existing S^2 X centroids and then propagating through C introduces
additional graph propagation and changes the experiment.
"""

import torch

from src.low_rank_assignment import logit_block, saved_encoder_nodes


def _check_probability(probability):
    if probability.ndim != 2 or min(probability.shape) == 0 or not probability.is_floating_point():
        raise ValueError("probability must be a nonempty floating-point nodes-by-cells matrix")
    if not bool(torch.isfinite(probability).all()) or bool((probability < 0).any()):
        raise ValueError("probability must have finite nonnegative entries")
    tolerance = max(1e-10, 50 * torch.finfo(probability.dtype).eps)
    if not torch.allclose(
        probability.sum(1), probability.new_ones(len(probability)), atol=tolerance, rtol=tolerance
    ):
        raise ValueError("assignment rows must sum to one")


def assignment_probability(saved, *, z=None, q=None, column_dual=None, chunk_size=1024, dtype=torch.float64):
    """Reconstruct P from a saved logits tensor, low-rank dict, or encoder dict.

    Pass the contents of best_assignment_logits/factors/encoder.pt.  Encoder
    snapshots additionally require the same normalized z and teacher q used during
    assignment fitting.  For balanced assignments, pass the saved column dual.
    The computation is differentiable; callers can use torch.no_grad for replay.
    The output uses z's device when supplied and otherwise the snapshot's device.
    Node-weight logits and learned label temperatures are separate from P.
    """
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    if isinstance(saved, torch.Tensor):
        logits = saved.to(device=z.device if z is not None else saved.device, dtype=dtype)
    elif isinstance(saved, dict) and "v" in saved:
        device = z.device if z is not None else saved["v"].device
        v = saved["v"].to(device=device)
        assignment = saved["assignment"].to(device=device)
        if "u" in saved:
            u = saved["u"].to(device=device)
        else:
            if z is None or q is None:
                raise ValueError("encoder replay requires the original assignment inputs z and q")
            u = saved_encoder_nodes(z, q, saved)
        logits = torch.cat(
            [
                logit_block(
                    u[start : start + chunk_size], v, assignment[start : start + chunk_size], saved["mixing"]
                ).to(dtype)
                for start in range(0, len(u), chunk_size)
            ],
            dim=0,
        )
    else:
        raise ValueError("saved must be a logits tensor or an assignment factors/encoder dictionary")
    if column_dual is not None:
        logits = logits + column_dual.to(logits)
    probability = logits.softmax(1)
    _check_probability(probability)
    return probability


def feature_centroids(probability, features):
    """Return (P.T @ X / mass, mass), with mass=P.sum(0) in node units.

    The same function can form soft label means.  Empty cells are rejected rather
    than silently replacing their feature means.  Both arguments retain gradients.
    """
    _check_probability(probability)
    if features.ndim != 2 or len(features) != len(probability):
        raise ValueError("features must have one row per original node")
    mass = probability.sum(0)
    if bool((mass <= 0).any()):
        raise ValueError("feature centroids require nonempty cells")
    features = features.to(probability)
    return (probability.T @ features) / mass[:, None], mass


def quotient_adjacency(
    probability,
    adjacency,
    *,
    normalization="symmetric",
    mass_scaling=False,
    self_loops="retain",
    chunk_size=1024,
):
    """Return a dense cell adjacency computed from dense or sparse original A.

    Convention, in order:
          1. C=P.T @ A @ P.  A may be raw or already normalized; this function does
             not recover raw edges from a normalized input.
          2. If mass_scaling, replace C by M^-1/2 @ C @ M^-1/2, M=diag(P.sum(0)).
             This is spectral mass scaling, not division by pairwise mean masses.
          3. self_loops='retain' keeps C's diagonal, including within-cell edges;
             'remove' sets it to zero; 'add' retains it and adds the identity;
             'unit' replaces it with ones.  Policies act AFTER mass scaling.
          4. normalization='symmetric' applies D^-1/2 C D^-1/2; 'row' applies
             D^-1 C; 'none' returns C.  Zero-degree cells remain zero.

    For raw loop-free A, 'add' plus symmetric normalization is a GCN convention
    that preserves within-cell edges.  For the existing normalized graph['adj'],
    use 'retain' to avoid adding a second identity.  With singleton hard cells,
    normalization='none' and 'retain' reproduce A (up to the node permutation).
    Symmetric normalization is intended for symmetric nonnegative adjacency.

    Column chunks bound the temporary A @ P allocation, but P itself is dense;
    this helper targets Cora/Citeseer and fixed-assignment replay first.
    """
    _check_probability(probability)
    if adjacency.ndim != 2 or adjacency.shape != (len(probability), len(probability)):
        raise ValueError("adjacency must be a square matrix with one row per original node")
    if normalization not in ("symmetric", "row", "none"):
        raise ValueError("normalization must be symmetric, row, or none")
    if self_loops not in ("retain", "remove", "add", "unit"):
        raise ValueError("self_loops must be retain, remove, add, or unit")
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    adjacency = adjacency.to(probability)
    if adjacency.layout == torch.sparse_coo:
        adjacency = adjacency.coalesce()
    values = adjacency if adjacency.layout == torch.strided else adjacency.values()
    if not bool(torch.isfinite(values).all()) or bool((values < 0).any()):
        raise ValueError("adjacency must have finite nonnegative weights")
    blocks = []
    for start in range(0, probability.shape[1], chunk_size):
        block = probability[:, start : start + chunk_size]
        propagated = (
            adjacency @ block if adjacency.layout == torch.strided else torch.sparse.mm(adjacency, block)
        )
        blocks.append(probability.T @ propagated)
    quotient = torch.cat(blocks, dim=1)
    if mass_scaling:
        mass = probability.sum(0)
        if bool((mass <= 0).any()):
            raise ValueError("mass scaling requires nonempty cells")
        inverse = mass.rsqrt()
        quotient = inverse[:, None] * quotient * inverse[None, :]
    if self_loops in ("remove", "unit"):
        quotient = quotient - torch.diag_embed(quotient.diagonal())
    if self_loops in ("add", "unit"):
        quotient = quotient + torch.eye(len(quotient), dtype=quotient.dtype, device=quotient.device)
    if normalization != "none":
        degree = quotient.sum(1)
        # Clamp before inversion so isolated cells have finite backward values.
        positive_degree = degree.clamp_min(torch.finfo(degree.dtype).tiny)
        inverse = positive_degree.rsqrt() if normalization == "symmetric" else positive_degree.reciprocal()
        inverse = torch.where(degree > 0, inverse, torch.zeros_like(inverse))
        if normalization == "symmetric":
            quotient = inverse[:, None] * quotient * inverse[None, :]
        else:
            quotient = inverse[:, None] * quotient
    return quotient
