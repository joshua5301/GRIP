import faiss
import torch
from sklearn.cluster import kmeans_plusplus


def feature_kmeans(h, cells, seed):
    threads = faiss.omp_get_max_threads()
    faiss.omp_set_num_threads(1)
    try:
        assignment = kmeans_init(h.cpu(), cells, seed=seed, plus_plus=True)
    finally:
        faiss.omp_set_num_threads(threads)
    counts = torch.bincount(assignment, minlength=cells)
    residual = (h.cpu() - cell_means(h.cpu(), assignment, cells)[assignment]).square().sum(1)
    for empty in (counts == 0).nonzero().flatten().tolist():
        node = int(residual.masked_fill(counts[assignment] <= 1, -torch.inf).argmax())
        counts[assignment[node]] -= 1
        assignment[node] = empty
        counts[empty] += 1
        residual[node] = 0
    return assignment


def kmeans_init(X: torch.Tensor, cluster_num: int, seed=1234, plus_plus=False, initial_centers=None):
    X_np = X.detach().cpu().numpy().astype("float32")
    kmeans = faiss.Kmeans(X_np.shape[1], cluster_num, gpu=False)
    kmeans.cp.min_points_per_centroid = 1
    kmeans.cp.seed = seed
    if initial_centers is not None:
        kmeans.train(X_np, init_centroids=initial_centers.detach().cpu().numpy().astype("float32"))
    elif plus_plus:
        centers, _ = kmeans_plusplus(X_np, cluster_num, random_state=seed, n_local_trials=1)
        kmeans.train(X_np, init_centroids=centers)
    else:
        kmeans.train(X_np)
    _, assign = kmeans.index.search(X_np, 1)
    return torch.from_numpy(assign.flatten()).long().to(X.device)


def cell_means(y_pred: torch.Tensor, assign: torch.Tensor, cluster_num: int):
    counts = torch.bincount(assign, minlength=cluster_num).clamp(min=1).to(y_pred.dtype)
    return torch.zeros(cluster_num, y_pred.shape[1], dtype=y_pred.dtype, device=y_pred.device).index_add_(
        0, assign, y_pred
    ) / counts.unsqueeze(1)
