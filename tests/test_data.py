import json
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp
import torch
from sklearn.preprocessing import StandardScaler

from src.data import get_dataset


@pytest.mark.parametrize("name", ["arxiv", "flickr", "reddit"])
def test_raw_loading_preserves_training_scaling_and_splits(tmp_path, name):
    folder = tmp_path / ("ogbn-arxiv" if name == "arxiv" else name) / "raw"
    folder.mkdir(parents=True)
    features = np.arange(18, dtype=np.float64).reshape(6, 3)
    role = {"tr": [0, 1], "va": [2, 3], "te": [4, 5]}
    labels = {str(i): i % 2 + 4 for i in range(6)}
    adjacency = sp.csr_matrix((np.ones(3), ([0, 2, 4], [1, 3, 5])), shape=(6, 6))
    sp.save_npz(folder / "adj_full.npz", adjacency)
    np.save(folder / "feats.npy", features)
    (folder / "role.json").write_text(json.dumps(role))
    (folder / "class_map.json").write_text(json.dumps(labels))
    graph = get_dataset(SimpleNamespace(dataset_name=name, raw_data_dir=str(tmp_path)))
    expected = torch.tensor(StandardScaler().fit(features[:2]).transform(features), dtype=torch.float32)
    if name == "arxiv":
        torch.testing.assert_close(graph.x, expected)
        assert graph.y.tolist() == [0, 1, 0, 1, 0, 1]
        assert graph.train_mask.nonzero().flatten().tolist() == [0, 1]
        assert graph.val_mask.nonzero().flatten().tolist() == [2, 3]
        assert graph.test_mask.nonzero().flatten().tolist() == [4, 5]
        assert graph.edge_index.shape[1] == 6
    else:
        for part, split, ids in zip(graph, ("train", "val", "test"), role.values()):
            torch.testing.assert_close(part.x, expected[ids])
            assert part.y.tolist() == [0, 1]
            assert getattr(part, f"{split}_mask").all()
            assert sorted(map(tuple, part.edge_index.T.tolist())) == [(0, 1), (1, 0)]
