import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp
import torch
import torch_geometric.transforms as T
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import subgraph

BUDGET = {
    ("cora", 0.013): 35,
    ("cora", 0.026): 70,
    ("cora", 0.052): 140,
    ("citeseer", 0.009): 30,
    ("citeseer", 0.018): 60,
    ("citeseer", 0.036): 120,
    ("arxiv", 0.0005): 90,
    ("arxiv", 0.0025): 454,
    ("arxiv", 0.005): 909,
    ("flickr", 0.001): 44,
    ("flickr", 0.005): 223,
    ("flickr", 0.01): 446,
    ("reddit", 0.0005): 77,
    ("reddit", 0.001): 153,
    ("reddit", 0.002): 307,
}


def normalize_adj_sparse(data):
    mx = sp.csr_matrix(
        (np.ones(data.edge_index.shape[1]), data.edge_index.cpu().numpy()),
        shape=(data.x.shape[0], data.x.shape[0]),
    )
    mx = mx.tolil()
    if mx[0, 0] == 0:
        mx = mx + sp.eye(mx.shape[0])
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1 / 2).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    mx = mx.dot(r_mat_inv)

    sparse_mx = mx.tocoo().astype(np.float32)
    sparserow = torch.LongTensor(sparse_mx.row).unsqueeze(1)
    sparsecol = torch.LongTensor(sparse_mx.col).unsqueeze(1)
    sparseconcat = torch.cat((sparserow, sparsecol), 1)
    sparsedata = torch.FloatTensor(sparse_mx.data)
    adj = torch.sparse_coo_tensor(sparseconcat.t(), sparsedata, sparse_mx.shape)
    return adj


def _prepare_dataset(name, data_dir, device, citation_features="default"):
    if citation_features not in ("default", "raw", "row"):
        raise ValueError("Unknown citation feature preprocessing")
    if name not in ("cora", "citeseer") and citation_features != "default":
        raise ValueError("Citation preprocessing only applies to Cora/Citeseer")
    args = SimpleNamespace(dataset_name=name, raw_data_dir=str(data_dir).rstrip("/") + "/",
                           citation_features=citation_features)
    datasets = get_dataset(args)

    def pack(graph):
        edges, weights = gcn_norm(graph.edge_index, graph.edge_attr, graph.num_nodes, dtype=graph.x.dtype)
        adjacency = torch.sparse_coo_tensor(edges.flip(0), weights, (graph.num_nodes, graph.num_nodes))
        adjacency = adjacency.coalesce().to_sparse_csr().to(device)
        return dict(x=graph.x.to(device), y=graph.y.to(device), adj=adjacency)

    if isinstance(datasets, list):
        train, val, test = [pack(g) for g in datasets]
        train_mask = datasets[0].train_mask.to(device)
        validation, testing = (val, None), (test, None)
    else:
        train = pack(datasets)
        train_mask = datasets.train_mask.to(device)
        validation = (train, datasets.val_mask.to(device))
        testing = (train, datasets.test_mask.to(device))
    with torch.no_grad():
        source = datasets[0] if isinstance(datasets, list) else datasets
        propagation = normalize_adj_sparse(source).coalesce().to_sparse_csr().to(device)
        H = torch.sparse.mm(propagation, torch.sparse.mm(propagation, train["x"]))
    return train, train_mask, validation, testing, H


def get_dataset(args):
    name = args.dataset_name
    if name in ("cora", "citeseer"):
        feature_mode = getattr(args, "citation_features", "default")
        transform = T.NormalizeFeatures() if (feature_mode == "row" or
                    (feature_mode == "default" and name == "citeseer")) else None
        return Planetoid(args.raw_data_dir, name, transform=transform)[0]
    if name not in ("arxiv", "flickr", "reddit"):
        raise ValueError(f"Unknown dataset: {name}")
    folder = Path(args.raw_data_dir) / ("ogbn-arxiv" if name == "arxiv" else name) / "raw"
    adjacency = sp.load_npz(folder / "adj_full.npz")
    if name == "arxiv":
        adjacency = adjacency + adjacency.T
        adjacency[adjacency > 1] = 1
    nodes = adjacency.shape[0]
    role = json.loads((folder / "role.json").read_text())
    masks = {}
    for split, key in (("train", "tr"), ("val", "va"), ("test", "te")):
        mask = torch.zeros(nodes, dtype=torch.bool)
        mask[role[key]] = True
        masks[f"{split}_mask"] = mask
    labels = process_labels(json.loads((folder / "class_map.json").read_text()), nodes)
    features = np.load(folder / "feats.npy")
    scaler = StandardScaler().fit(features[role["tr"]])
    graph = Data(
        x=torch.FloatTensor(scaler.transform(features)),
        edge_index=torch.LongTensor(np.array(adjacency.nonzero())),
        y=torch.LongTensor(labels),
        **masks,
    )
    graph = T.ToUndirected()(graph)
    return graph if name == "arxiv" else inductive_processing(graph)


def inductive_processing(data):
    graphs = []
    for split in ("train", "val", "test"):
        mask = getattr(data, f"{split}_mask")
        edges, _ = subgraph(mask, data.edge_index, relabel_nodes=True)
        graph = Data(x=data.x[mask], y=data.y[mask], edge_index=edges)
        setattr(graph, f"{split}_mask", torch.ones(len(graph.x), dtype=torch.bool))
        graphs.append(graph)
    return graphs


def process_labels(class_map, nodes):
    first = next(iter(class_map.values()))
    labels = np.zeros((nodes, len(first))) if isinstance(first, list) else np.zeros(nodes, dtype=int)
    for index, value in class_map.items():
        labels[int(index)] = value
    return labels if isinstance(first, list) else labels - labels.min()
