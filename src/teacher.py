import math

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


def teacher_logits(h, train, mask, validation, kernel, gammas, basis, seed, folder, return_all=False):
    path = folder / ("teachers.pt" if return_all else "teacher.pt")
    if path.exists():
        saved = torch.load(path, map_location=h.device, weights_only=False)
        return saved if return_all else (saved["logits"], saved["gamma"])
    generator = torch.Generator(device=h.device).manual_seed(seed)
    hd = h.double()
    anchors = (
        hd if basis >= len(hd) else hd[torch.randperm(len(hd), device=h.device, generator=generator)[:basis]]
    )
    gram = get_kernel_values(anchors, anchors, kernel)
    gram = (gram + gram.T) / 2
    eye = torch.eye(len(anchors), device=h.device, dtype=torch.double)
    chol = torch.linalg.cholesky(gram + 1e-8 * gram.diagonal().mean() * eye)
    mapping = torch.linalg.solve_triangular(chol, eye, upper=False).T

    def features(x):
        return torch.cat(
            [get_kernel_values(block.double(), anchors, kernel) @ mapping for block in x.split(8192)]
        )

    phi = features(h)
    graph, val_mask = validation
    if graph is train:
        val_phi, val_y = phi[val_mask], graph["y"][val_mask]
    else:
        val_h = torch.sparse.mm(graph["adj"], torch.sparse.mm(graph["adj"], graph["x"]))
        val_phi, val_y = features(val_h), graph["y"]
    targets = F.one_hot(train["y"][mask], int(train["y"].max()) + 1).double()
    rows, best, selected, logits = [], -float("inf"), None, None
    all_logits = {}
    for gamma in tqdm(gammas, desc="Teacher gamma validation"):
        weight = fit_logistic(phi[mask], targets, gamma)
        if return_all:
            all_logits[gamma] = (phi @ weight).detach().cpu()
        scores = val_phi @ weight
        val = float((scores.argmax(1) == val_y).double().mean())
        rows.append(dict(gamma=gamma, val=100 * val, val_ce=float(F.cross_entropy(scores, val_y))))
        if val > best:
            best, selected, logits = val, gamma, (phi @ weight).detach()
    pd.DataFrame(rows).to_csv(folder / "teacher_grid.csv", index=False)
    if return_all:
        torch.save(all_logits, path)
        return all_logits
    torch.save(dict(logits=logits.cpu(), gamma=selected), path)
    return logits, selected


EPS = 1e-12


def get_kernel_values(A: torch.Tensor, B: torch.Tensor, kernel_kind: str):
    d = B.shape[1]
    if kernel_kind == "erf":
        bandwidth = (B * B).sum(1).mean() / d
        S = (A @ B.T) / (d * bandwidth)
        a = (A * A).sum(1, keepdim=True) / (d * bandwidth)
        b = (B * B).sum(1).unsqueeze(0) / (d * bandwidth)
        r = 2 * S / torch.sqrt((1 + 2 * a) * (1 + 2 * b))
        return (2 / math.pi) * torch.asin(r.clamp(-1 + EPS, 1 - EPS))
    if kernel_kind.startswith("relu"):
        layers = int(kernel_kind[4:] or 1)
        bandwidth = (B * B).sum(1).mean() / d
        na = A.norm(dim=1, keepdim=True).clamp(min=EPS)
        nb = B.norm(dim=1).unsqueeze(0).clamp(min=EPS)
        cos = ((A @ B.T) / (na * nb)).clamp(-1 + EPS, 1 - EPS)
        for _ in range(layers):
            th = torch.acos(cos)
            cos = ((torch.sin(th) + (math.pi - th) * torch.cos(th)) / math.pi).clamp(-1 + EPS, 1 - EPS)
        return (na * nb) / (d * bandwidth) * cos
    if kernel_kind == "rbf":
        d2 = (A * A).sum(1, keepdim=True) + (B * B).sum(1).unsqueeze(0) - 2 * (A @ B.T)
        bandwidth = (B * B).sum(1).mean()
        return torch.exp(-d2.clamp(min=0) / bandwidth)
    if kernel_kind == "linear":
        return A @ B.T
    raise ValueError(f"unknown teacher kernel {kernel_kind}")


def fit_logistic(X_kernel_train: torch.Tensor, y_train: torch.Tensor, gamma: float, steps=1000):
    X_kernel_train, y_train = X_kernel_train.double(), y_train.double()
    n, dim = X_kernel_train.shape
    W = torch.zeros(
        dim, y_train.shape[1], dtype=X_kernel_train.dtype, device=X_kernel_train.device
    ).requires_grad_(True)
    opt = torch.optim.LBFGS([W], max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(X_kernel_train @ W, y_train) + gamma / n * (W**2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return W.detach()
