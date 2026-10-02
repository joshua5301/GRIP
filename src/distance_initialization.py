import torch


def project_features(x, rank):
    dimension = min(rank - 1, x.shape[1], len(x) - 1)
    if dimension < 1:
        raise ValueError("Distance initialization requires rank >= 2")
    devices = [x.device.index] if x.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(0)
        _, _, vectors = torch.pca_lowrank(x, q=min(dimension + 8, *x.shape), center=False, niter=4)
    return x @ vectors[:, :dimension]


def distance_factors(projected, cells, rank, seed, temperature=1.0, lloyd_steps=20, block_size=None):
    x = projected
    generator = torch.Generator(device=x.device).manual_seed(seed)
    first = int(torch.randint(len(x), (1,), device=x.device, generator=generator))
    centers = [x[first]]
    nearest = (x - centers[0]).square().sum(1)
    for _ in range(1, cells):
        index = (
            int(torch.multinomial(nearest, 1, generator=generator))
            if float(nearest.sum()) > 0
            else int(torch.randint(len(x), (1,), device=x.device, generator=generator))
        )
        centers.append(x[index])
        nearest = torch.minimum(nearest, (x - centers[-1]).square().sum(1))
    centers = torch.stack(centers)
    for _ in range(lloyd_steps):
        counts = torch.zeros(cells, device=x.device, dtype=torch.long)
        sums = torch.zeros_like(centers)
        for chunk in x.split(block_size or len(x)):
            assignment = torch.cdist(chunk, centers).argmin(1)
            counts.add_(torch.bincount(assignment, minlength=cells))
            sums.index_add_(0, assignment, chunk)
        centers = torch.where(counts[:, None] > 0, sums / counts.clamp_min(1)[:, None], centers)
    left = torch.cat((2 * x, x.new_ones(len(x), 1)), 1)
    right = torch.cat((centers, -centers.square().sum(1, keepdim=True)), 1)
    right = right - right.mean(0)
    rms = ((left.T @ left) * (right.T @ right)).sum().div(len(x) * cells).clamp_min(1e-24).sqrt()
    left = left / (temperature * rms)
    a, ra = torch.linalg.qr(left, mode="reduced")
    b, rb = torch.linalg.qr(right, mode="reduced")
    u, s, vh = torch.linalg.svd(ra @ rb.T, full_matrices=False)
    scale = s.sqrt() * rank**0.25
    u, v = (a @ u) * scale, (b @ vh.T) * scale
    padding = rank - u.shape[1]
    return torch.nn.functional.pad(u, (0, padding)), torch.nn.functional.pad(v, (0, padding))
