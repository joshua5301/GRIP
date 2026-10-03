import torch


def bound_features(x, q, weight):
    xc, qc = x - x.mean(0), q - q.mean(0)
    vx, vq = xc.square().sum(1).mean(), qc.square().sum(1).mean()
    if weight == 0 or float(vq) == 0:
        return xc, dict(seeding_eta=None, feature_weight=1.0, label_weight=0.0)
    if float(vx) == 0:
        return qc, dict(seeding_eta=None, feature_weight=0.0, label_weight=1.0)
    eta = (vq / vx).sqrt()
    a, b = 1 + weight * eta / 2, weight / (2 * eta)
    return torch.cat((a.sqrt() * xc, b.sqrt() * qc), 1), dict(
        seeding_eta=float(eta), feature_weight=float(a), label_weight=float(b)
    )


def minimum_scatter_split(centered, projection):
    order = torch.argsort(projection, stable=True)
    values = projection[order]
    valid = values[:-1] < values[1:]
    if not bool(valid.any()):
        cut = len(order) // 2
    else:
        prefix = centered[order].cumsum(0)
        count = torch.arange(1, len(order), device=centered.device, dtype=centered.dtype)
        gain = prefix[:-1].square().sum(1) / count
        gain += (prefix[-1] - prefix[:-1]).square().sum(1) / (len(order) - count)
        cut = int(gain.masked_fill(~valid, -torch.inf).argmax()) + 1
    left = torch.zeros(len(order), device=centered.device, dtype=torch.bool)
    left[order[:cut]] = True
    return left


def pca_partition(z, cells, power_steps=100, tolerance=1e-8, method="pca"):
    if method not in ("pca", "var", "pca_sse"):
        raise ValueError("Unknown hierarchical split method")
    if not 1 <= cells <= len(z):
        raise ValueError("Invalid cell count")
    groups = [torch.arange(len(z), device=z.device)]

    def scatter(ids):
        if len(ids) < 2:
            return -1.0
        block = z[ids]
        return float((block - block.mean(0)).square().sum())

    energies = [scatter(groups[0])]
    diagnostics = []
    while len(groups) < cells:
        index = max(range(len(groups)), key=lambda j: energies[j])
        ids = groups[index]
        centered = z[ids] - z[ids].mean(0)
        variance = centered.square().sum(0)
        axis = variance.argmax()
        direction = torch.zeros_like(variance)
        direction[axis] = 1
        converged, used = False, 0
        for step in range(0 if method == "var" else power_steps):
            updated = centered.T @ (centered @ direction)
            norm = updated.norm()
            used = step + 1
            if float(norm) == 0:
                break
            updated = updated / norm
            change = torch.minimum((updated - direction).norm(), (updated + direction).norm())
            direction = updated
            if float(change) <= tolerance:
                converged = True
                break
        if direction[direction.abs().argmax()] < 0:
            direction = -direction
        projection = centered @ direction
        left = minimum_scatter_split(centered, projection) if method == "pca_sse" else projection <= 0
        if bool(left.all()) or not bool(left.any()):
            order = torch.argsort(projection, stable=True)
            left = torch.zeros_like(left)
            left[order[: len(ids) // 2]] = True
        first, second = ids[left], ids[~left]
        groups[index] = first
        groups.append(second)
        energies[index] = scatter(first)
        energies.append(scatter(second))
        diagnostics.append(dict(power_steps=used, power_converged=converged, method=method))
    assignment = torch.empty(len(z), device=z.device, dtype=torch.long)
    for j, ids in enumerate(groups):
        assignment[ids] = j
    return assignment, dict(
        initialization_steps=0,
        initialization_converged=False,
        initial_center_indices=[],
        pca_splits=diagnostics,
    )
