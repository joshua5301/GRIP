"""Exact CE bilevel optimization of edge-preserving low-rank clustering.

P alone determines every condensed quantity: Xc=P.T X/m, Qc=P.T Q/m,
Sc=normalize(P.T A P), and Hc=Sc^2 Xc.  The inner linear softmax head sees
Hc in the fixed RMS coordinates fitted to original H=A^2 X.  The outer loss
is uniform teacher CE on original H in those same coordinates.  Assignment
gradients differentiate both the centroids and Sc through an exact implicit
adjoint.  No independent synthetic features, labels, or edges are optimized.

This implementation materializes dense P and targets small citation graphs.
Snapshots retain original-X centroids and Sc for existing two-layer GCN replay.
"""

import time
from pathlib import Path

import torch

from src.coarsening import feature_centroids, quotient_adjacency
from src.io import array_digest, cpu_state, save_json, save_state
from src.low_rank_assignment import initialize_factors, logit_block
from src.moments import augmented
from src.soft_ce_partition import (
    head_gradient,
    outer_value_gradient,
    solve_head_system,
    solve_inner_newton_first,
)
from src.transforms import FeatureTransform, fit_transform


def coarsened_problem(
    probability, x, q, adjacency, transform, *, mass_scaling=False, self_loops="retain", chunk_size=1024
):
    """Build differentiable condensed inputs, using original X rather than H.

    ``adjacency`` is the original propagation matrix.  ``self_loops='retain'`` is
    appropriate for graph['adj'], which already contains normalized self-loops.
    ``mass`` sums to one; ``node_mass`` counts fractional original nodes.
    """
    centers, node_mass = feature_centroids(probability, x)
    labels, _ = feature_centroids(probability, q)
    quotient = quotient_adjacency(
        probability, adjacency, mass_scaling=mass_scaling, self_loops=self_loops, chunk_size=chunk_size
    )
    propagated = quotient @ (quotient @ centers)
    return dict(
        x=centers,
        labels=labels,
        mass=node_mass / len(probability),
        node_mass=node_mass,
        adj=quotient,
        head_features=transform(propagated),
        probability=probability,
    )


def gcn_inputs(snapshot, device=None):
    """Return float32 (Xc, Qc, mass, Sc) for the existing GCN diagnostic fitter.

    The optimizer retains double precision in snapshots for exact head fitting and
    resume.  Existing GCN models use float32, so their features and adjacency must
    be converted together.  Returned mass reflects cluster size; callers requesting
    uniform student CE should replace it by ones or a uniform probability vector.
    """
    return tuple(
        snapshot[key].to(device=device, dtype=torch.float32) for key in ("x", "labels", "mass", "adj")
    )


def implicit_low_rank_gradient(
    u,
    v,
    assignment,
    x,
    q,
    adjacency,
    outer_z,
    transform,
    penalty,
    *,
    mixing=0.05,
    mass_scaling=False,
    self_loops="retain",
    chunk_size=1024,
    inner_loss_weighting="mass",
    theta=None,
    vector=None,
    inner_tol=1e-7,
    cg_rtol=1e-6,
    inner_max_iter=2000,
    cg_max_iter=512,
    compute_gradient=True,
):
    """Solve one exact bilevel objective and return its U,V implicit gradients.

    The returned dict contains the differentiable condensed problem, detached head
    and adjoint, solver diagnostics, outer CE, and (du,dv).  Set compute_gradient
    false for endpoint evaluation.  ``outer_z`` and the transform must remain fixed
    throughout optimization; they are formed from original propagated H, not Hc.
    """
    if inner_loss_weighting not in ("mass", "uniform"):
        raise ValueError("inner_loss_weighting must be mass or uniform")
    probability = logit_block(u, v, assignment, mixing).double().softmax(1)
    problem = coarsened_problem(
        probability,
        x,
        q,
        adjacency,
        transform,
        mass_scaling=mass_scaling,
        self_loops=self_loops,
        chunk_size=chunk_size,
    )
    mass = (
        problem["mass"]
        if inner_loss_weighting == "mass"
        else torch.full_like(problem["mass"], 1 / len(problem["mass"]))
    )
    features, labels = problem["head_features"], problem["labels"]
    inner = solve_inner_newton_first(
        features.detach(),
        labels.detach(),
        mass.detach(),
        penalty,
        initial=theta,
        max_iter=inner_max_iter,
        grad_tol=inner_tol,
        cg_max_iter=cg_max_iter,
    )
    if not inner["inner_converged"]:
        raise RuntimeError("Coarsening inner CE did not converge; refusing an inexact update")
    theta = inner["theta"]
    value, rhs = outer_value_gradient(outer_z, q, theta, chunk_size=chunk_size)
    result = dict(**problem, theta=theta, outer_ce=value, inner=inner, vector=vector)
    if compute_gradient:
        vector, diagnostic = solve_head_system(
            augmented(features.detach()),
            labels.detach(),
            mass.detach(),
            theta,
            penalty,
            rhs,
            rtol=cg_rtol,
            max_iter=cg_max_iter,
            initial=vector,
        )
        if not diagnostic["cg_converged"]:
            raise RuntimeError("Coarsening implicit Hessian solve did not converge")
        gradient = head_gradient(augmented(features), labels, mass, theta.detach(), penalty)
        gradients = torch.autograd.grad(-(gradient * vector.detach()).sum(), (u, v))
        if not all(bool(torch.isfinite(direction).all()) for direction in gradients):
            raise FloatingPointError("Nonfinite coarsening assignment gradient")
        result.update(vector=vector, implicit=diagnostic, gradients=gradients)
    return result


def optimize_coarsening_ce(
    x,
    q,
    assignment,
    adjacency,
    *,
    h=None,
    transform=None,
    penalty=1e-4,
    steps=50,
    lr=0.01,
    rank=16,
    seed=0,
    mixing=0.05,
    mass_scaling=False,
    self_loops="retain",
    chunk_size=1024,
    inner_loss_weighting="mass",
    inner_tol=1e-7,
    cg_rtol=1e-6,
    inner_max_iter=2000,
    cg_max_iter=512,
    checkpoint_steps=None,
    folder=None,
    resume_state=None,
    save_resume=True,
    stop=lambda: False,
    progress=None,
):
    """Optimize low-rank P with Adam and return history, checkpoints and state.

    ``steps`` counts Adam updates; step zero and the endpoint are both evaluated.
    Pass a loaded resume.pt as resume_state to extend a prior run.  The step budget
    is deliberately excluded from the configuration fingerprint.  Data, teacher
    targets, graph and fixed transform must match; resume checks their digest.

    Each checkpoint contains x/raw Xc, labels/Qc, normalized mass, adj/Sc,
    head_features/fixed-RMS Hc, theta, probability, u,v, assignment and mixing.
    Use gcn_inputs to cast x, labels and adj for fit_gcn_diagnostic.  For uniform
    student CE pass uniform masses to that fitter.  Endpoint selection is left to
    validation outside this optimizer; outer CE is not a substitute for validation.
    """
    if not isinstance(steps, int) or steps < 0 or penalty <= 0 or lr <= 0:
        raise ValueError("steps must be nonnegative and penalty/lr must be positive")
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    if x.ndim != 2 or q.ndim != 2 or len(q) != len(x) or len(assignment) != len(x):
        raise ValueError("x, q and assignment must have matching node rows")
    x, q = x.detach().double(), q.detach().to(device=x.device, dtype=torch.double)
    assignment = assignment.detach().to(device=x.device, dtype=torch.long)
    adjacency = adjacency.detach().to(x)
    if adjacency.layout == torch.sparse_coo:
        adjacency = adjacency.coalesce()
    if h is None:
        propagate = lambda value: (
            adjacency @ value if adjacency.layout == torch.strided else torch.sparse.mm(adjacency, value)
        )
        h = propagate(propagate(x))
    h = h.detach().to(x)
    if h.shape != x.shape:
        raise ValueError("original H must match the original feature shape")
    if transform is None:
        outer_z, transform = fit_transform(h, kind="rms")
    else:
        if transform.kind != "rms" or transform.matrix is not None:
            raise ValueError("Use a fixed original-H RMS transform")
        transform = FeatureTransform(
            **{
                key: value.to(x) if torch.is_tensor(value) else value
                for key, value in vars(transform).items()
            }
        )
        outer_z = transform(h)
    clusters = int(assignment.max()) + 1
    u, v = [
        value.detach().double().requires_grad_()
        for value in initialize_factors(assignment, clusters, rank, seed)
    ]
    optimizer = torch.optim.Adam([u, v], lr=lr)
    if adjacency.layout == torch.strided:
        graph_arrays = [adjacency.cpu().numpy()]
    elif adjacency.layout == torch.sparse_coo:
        graph_arrays = [adjacency.indices().cpu().numpy(), adjacency.values().cpu().numpy()]
    else:
        graph_arrays = [
            adjacency.crow_indices().cpu().numpy(),
            adjacency.col_indices().cpu().numpy(),
            adjacency.values().cpu().numpy(),
        ]
    digest_arrays = [
        value.detach().cpu().numpy()
        for value in (x, q, assignment, h, transform.center, transform.output_center, transform.scale)
    ]
    config = dict(
        version=1,
        penalty=penalty,
        lr=lr,
        rank=rank,
        seed=seed,
        mixing=mixing,
        mass_scaling=mass_scaling,
        self_loops=self_loops,
        chunk_size=chunk_size,
        inner_loss_weighting=inner_loss_weighting,
        inner_tol=inner_tol,
        cg_rtol=cg_rtol,
        inner_max_iter=inner_max_iter,
        cg_max_iter=cg_max_iter,
        data_digest=array_digest(*(digest_arrays + graph_arrays)),
    )
    if folder is not None:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
    checks = {0, steps, *(checkpoint_steps or [])}
    if any(not isinstance(step, int) or step < 0 or step > steps for step in checks):
        raise ValueError("checkpoint_steps must lie within the update budget")
    start, theta, vector, history, snapshots = 0, None, None, [], {}
    if resume_state is not None:
        if resume_state["config"] != config or resume_state["step"] > steps:
            raise ValueError("Resume state differs from data, settings or step budget")
        with torch.no_grad():
            u.copy_(resume_state["u"].to(u))
            v.copy_(resume_state["v"].to(v))
        optimizer.load_state_dict(resume_state["optimizer"])
        start = resume_state["step"]
        theta = resume_state["theta"].to(x) if resume_state["theta"] is not None else None
        vector = resume_state["vector"].to(x) if resume_state["vector"] is not None else None
        history = [dict(row) for row in resume_state["history"] if row["step"] < start]
        snapshots = dict(resume_state["checkpoints"])
    if folder is not None:
        save_json(config, folder / "config.json")

    def state_at(step):
        return cpu_state(
            dict(
                config=config,
                step=step,
                u=u,
                v=v,
                optimizer=optimizer.state_dict(),
                theta=theta,
                vector=vector,
                history=history,
                checkpoints=snapshots,
            )
        )

    def persist(step):
        state = state_at(step)
        if folder is not None and save_resume:
            save_state(state, folder / "resume.pt")
        return state

    for step in range(start, steps + 1):
        if stop():
            persist(step)
            raise InterruptedError("Coarsening stopped with resumable state")
        started = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        try:
            evaluated = implicit_low_rank_gradient(
                u,
                v,
                assignment,
                x,
                q,
                adjacency,
                outer_z,
                transform,
                penalty,
                mixing=mixing,
                mass_scaling=mass_scaling,
                self_loops=self_loops,
                chunk_size=chunk_size,
                inner_loss_weighting=inner_loss_weighting,
                theta=theta,
                vector=vector,
                inner_tol=inner_tol,
                cg_rtol=cg_rtol,
                inner_max_iter=inner_max_iter,
                cg_max_iter=cg_max_iter,
                compute_gradient=step < steps,
            )
        except (RuntimeError, FloatingPointError):
            persist(step)
            raise
        theta, vector = evaluated["theta"], evaluated["vector"]
        row = dict(
            step=step,
            outer_ce=evaluated["outer_ce"],
            inner_grad=evaluated["inner"]["inner_grad_max"],
            inner_converged=True,
            seconds=time.monotonic() - started,
        )
        if step < steps:
            row.update(
                implicit_residual=evaluated["implicit"]["cg_relative_residual"],
                gradient_norm=float(torch.cat([g.flatten() for g in evaluated["gradients"]]).norm()),
            )
        history.append(row)
        if step in checks:
            snapshot = {
                key: evaluated[key]
                for key in (
                    "x",
                    "labels",
                    "mass",
                    "node_mass",
                    "adj",
                    "head_features",
                    "probability",
                    "theta",
                    "outer_ce",
                )
            }
            snapshot.update(step=step, u=u, v=v, assignment=assignment, mixing=mixing)
            snapshots[step] = cpu_state(snapshot)
            if folder is not None:
                (folder / "checkpoints").mkdir(exist_ok=True)
                save_state(snapshot, folder / "checkpoints" / f"step_{step:06d}.pt")
                save_json(history, folder / "history.json")
            persist(step)
        if progress is not None:
            progress(dict(row))
        if step == steps:
            return dict(
                history=history,
                checkpoints=snapshots,
                resume_state=persist(step),
                transform=transform.state_dict(),
                config=config,
            )
        for parameter, direction in zip((u, v), evaluated["gradients"], strict=True):
            parameter.grad = direction
        optimizer.step()
    raise AssertionError("Unreachable")
