import hashlib
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from src.data import BUDGET, _prepare_dataset
from src.distance_finetune import evaluation_splits
from src.evaluation import fit_gcn_diagnostic
from src.io import _fingerprint, save_json, save_state
from src.moment_lloyd import moment_lloyd_partition
from src.variance_moment_sweep import _data_digest


def push_average_operator(adjacency):
    coo = adjacency.to_sparse_coo().coalesce()
    receiver, sender = coo.indices()
    n = adjacency.shape[0]
    degree = torch.bincount(sender, minlength=n).to(coo.values())
    weights = degree[sender].reciprocal()
    incoming = weights.new_zeros(n).index_add_(0, receiver, weights)
    weights /= incoming[receiver]
    return torch.sparse_coo_tensor(coo.indices(), weights, coo.shape).coalesce().to_sparse_csr()


@torch.no_grad()
def push_labels(operator, source, alpha, tolerance=1e-6, max_steps=2000):
    if not 0 <= alpha < 1:
        raise ValueError("Alpha must be in [0, 1)")
    q = source.clone()
    for step in range(1, max_steps + 1):
        updated = (1 - alpha) * source + alpha * torch.sparse.mm(operator, q)
        delta = float((updated - q).abs().max())
        q = updated
        if delta <= tolerance * (1 - alpha):
            return q, dict(iterations=step, residual=delta, converged=True)
    raise RuntimeError(f"Push did not converge: alpha={alpha}, residual={delta}")


def nearest_train(z, train, k=64):
    import faiss
    train_ids = torch.where(train)[0].cpu().numpy()
    x = np.ascontiguousarray(z.float().cpu().numpy())
    index = faiss.IndexFlatL2(x.shape[1])
    resources = None
    if z.is_cuda and hasattr(faiss, "StandardGpuResources"):
        resources = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(resources, z.device.index or 0, index)
    index.add(np.ascontiguousarray(x[train_ids]))
    distances, neighbors = [], []
    for start in tqdm(range(0, len(x), 4096), desc="Exact train-neighbor search"):
        d, ids = index.search(x[start:start + 4096], min(k, len(train_ids)))
        distances.append(torch.from_numpy(d.copy()))
        neighbors.append(torch.from_numpy(train_ids[ids].copy()))
    return torch.cat(distances), torch.cat(neighbors)


@torch.no_grad()
def kernel_labels(distances, neighbor_labels, classes, sigma, beta):
    weights = (-distances / (2 * sigma**2)).exp()
    evidence = weights.new_zeros(len(weights), classes).scatter_add_(1, neighbor_labels, weights)
    q = (evidence + beta / classes) / (weights.sum(1, keepdim=True) + beta)
    return q


def run_propagation_kmeans(dataset, ratio, output_dir, alphas=(0.2, 0.5, 0.8, 0.95),
                           bandwidths=(0.5, 1., 2.), priors=(0.1, 1., 10.), neighbors=64,
                           search_seeds=(0, 1, 2, 3, 4), final_seeds=tuple(range(100, 110)),
                           data_dir="/content/data/", device="cuda"):
    if (dataset, ratio) not in BUDGET or not alphas or not bandwidths or not priors:
        raise ValueError("Invalid density or empty grid")
    if any(not 0 <= a < 1 for a in alphas) or any(v <= 0 for v in (*bandwidths, *priors)):
        raise ValueError("Invalid propagation parameters")
    if neighbors < 1 or set(search_seeds) & set(final_seeds):
        raise ValueError("Invalid neighbors or overlapping evaluation seeds")
    graph, train, validation, testing, h = _prepare_dataset(dataset, data_dir, device)
    masks = evaluation_splits(graph, train, validation, testing)
    classes = int(graph["y"][train].max()) + 1
    config = dict(dataset=dataset, ratio=ratio, nodes=BUDGET[dataset, ratio], alphas=list(alphas),
                  bandwidths=list(bandwidths), priors=list(priors), neighbors=neighbors,
                  search_seeds=list(search_seeds), final_seeds=list(final_seeds),
                  train_nodes=int(train.sum()), total_nodes=len(h),
                  data_digest=_data_digest(dict(train=(graph, train), val=validation, test=testing)),
                  source_digest=hashlib.sha256(b"".join((Path(__file__).parent / name).read_bytes()
                      for name in ("propagation_kmeans.py", "moment_lloyd.py", "moment_seeding.py",
                                   "data.py", "evaluation.py", "models.py"))).hexdigest(),
                  kernel_self_evidence=True, graph_weights="unweighted support with self loops")
    root = Path(output_dir) / _fingerprint(config)
    root.mkdir(parents=True, exist_ok=True)
    save_json(config, root / "config.json")
    partition_path = root / "partition.pt"
    if partition_path.exists():
        partition = torch.load(partition_path, weights_only=False)
    else:
        partition = moment_lloyd_partition(h, h.new_full((len(h), classes), 1 / classes),
            config["nodes"], mode="variance_sum", moment_weight=0., seeding="feature_var", max_sweeps=1000)
        if not partition["converged"]:
            raise RuntimeError("K-means did not converge")
        save_state(partition, partition_path)
    assignment = partition["assignment"].to(device)
    counts = partition["counts"].to(device)
    x = partition["x"].to(device)
    source = h.new_full((len(h), classes), 1 / classes)
    source[train] = F.one_hot(graph["y"][train], classes).to(source)
    operator = push_average_operator(graph["adj"])
    grid = [dict(method="push", alpha=float(a)) for a in alphas]
    grid += [dict(method="kernel", bandwidth=float(s), prior=float(b))
             for s, b in itertools.product(bandwidths, priors)]
    settings = dict(epochs=1000, eval_every=10, hidden=256, dropout=0.9 if dataset == "cora" else 0.5,
                    lr=0.01, weight_decay=0.0005)
    distances = neighbor_labels = None
    rows = []
    for candidate, params in enumerate(tqdm(grid, desc=f"{dataset}: propagation sweep")):
        folder = root / f"candidate_{candidate:04d}"
        folder.mkdir(exist_ok=True)
        path = folder / "labels.pt"
        if path.exists():
            artifact = torch.load(path, weights_only=False)
        else:
            if params["method"] == "push":
                q, diagnostic = push_labels(operator, source, params["alpha"])
            else:
                if distances is None:
                    neighbor_path = root / "neighbors.pt"
                    if neighbor_path.exists():
                        cached = torch.load(neighbor_path, weights_only=True)
                        d, ids = cached["distances"], cached["ids"]
                    else:
                        z = h - h.mean(0)
                        z /= z.square().sum(1).mean().sqrt().clamp_min(1e-30)
                        d, ids = nearest_train(z, train, neighbors)
                        save_state(dict(distances=d, ids=ids), neighbor_path)
                    distances = d.to(device).clamp_min(0)
                    neighbor_labels = graph["y"][ids.to(device)]
                    positive = distances[:, -1][distances[:, -1] > 0]
                    sigma_base = float(positive.median().sqrt()) if len(positive) else 1.
                sigma = sigma_base * params["bandwidth"]
                q = kernel_labels(distances, neighbor_labels, classes, sigma, params["prior"])
                diagnostic = dict(sigma=sigma, sigma_base=sigma_base)
            labels = q.new_zeros(config["nodes"], classes).index_add_(0, assignment, q) / counts[:, None]
            artifact = dict(labels=labels.cpu(), **diagnostic)
            save_state(artifact, path)
            del q
        labels = artifact["labels"].to(device)
        scores = [fit_gcn_diagnostic(x, labels, torch.ones(len(x), device=device), graph, None,
                  {key: masks[key] for key in ("train", "val")}, seed, folder=folder / "search", **settings)
                  for seed in search_seeds]
        frame = pd.DataFrame(scores)
        frame.assign(student_seed=list(search_seeds)).to_csv(folder / "search_students.csv", index=False)
        rows.append(dict(candidate=candidate, **params, search_val=frame.val_acc.mean(),
                         search_val_std=frame.val_acc.std()))
        pd.DataFrame(rows).to_csv(root / "search.csv", index=False)
    search = pd.DataFrame(rows)
    summaries = []
    for method in ("push", "kernel"):
        winner = search[search.method == method].sort_values(["search_val", "candidate"],
                     ascending=[False, True]).iloc[0].dropna().to_dict()
        folder = root / f"candidate_{int(winner['candidate']):04d}"
        labels = torch.load(folder / "labels.pt", weights_only=False)["labels"].to(device)
        scores = pd.DataFrame([fit_gcn_diagnostic(x, labels, torch.ones(len(x), device=device), graph,
                   None, masks, seed, folder=folder / "final", **settings) for seed in final_seeds])
        scores.assign(student_seed=list(final_seeds)).to_csv(root / f"{method}_final_students.csv", index=False)
        summaries.append(dict(dataset=dataset, ratio=ratio, nodes=config["nodes"], **winner,
                              final_val=scores.val_acc.mean(), test_mean=scores.test_acc.mean(),
                              test_std=scores.test_acc.std(), output_dir=str(root)))
        pd.DataFrame(summaries).to_csv(root / "summary.csv", index=False)
    return pd.DataFrame(summaries), search, root
