import json

import pytest
import torch

from src.gcn_moment_sweep import shared_source_identity
from src.sgc_teacher import rms_features, select_sgc_teacher


def test_normalization_reuses_reference_statistics():
    h = torch.tensor([[1., 2.], [3., 6.], [5., 4.]])
    z, offset, scale = rms_features(h)
    torch.testing.assert_close(z.mean(0), torch.zeros(2, dtype=torch.double))
    torch.testing.assert_close(z.square().sum(1).mean(), torch.tensor(1., dtype=torch.double))
    other, _, _ = rms_features(h + 10, offset, scale)
    torch.testing.assert_close(other, z + 10 / scale)


def test_sgc_teacher_selection_and_cache(tmp_path):
    h = torch.tensor([[-2., 0.], [-1., 0.], [1., 0.], [2., 0.]])
    graph = dict(y=torch.tensor([0, 0, 1, 1]))
    train = torch.tensor([0, 3])
    validation = (graph, torch.tensor([1, 2]))
    result = select_sgc_teacher(h, graph, train, validation, validation, [0.1, 1.], tmp_path)
    assert result["teacher_type"] == "sgc"
    assert result["val_acc"] == 100.
    assert result["penalty"] == 0.1
    cached = select_sgc_teacher(h, graph, train, validation, validation, [0.1, 1.], tmp_path)
    torch.testing.assert_close(result["logits"].cpu(), cached["logits"])


def test_shared_teacher_type_mismatch(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"teacher_type": "gcn"}))
    with pytest.raises(ValueError, match="teacher_type"):
        shared_source_identity(tmp_path, {"teacher_type": "sgc"})
