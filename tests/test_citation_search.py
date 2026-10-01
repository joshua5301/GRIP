import pytest
import torch

from src.citation_search import dataset_digest, run_screen, selected_test
from src.io import save_json


def problem():
    graph = dict(x=torch.randn(8, 3), y=torch.arange(8) % 2,
                 adj=torch.eye(8).to_sparse_csr())
    train = torch.arange(8) < 2
    val, test = torch.arange(8).eq(2) | torch.arange(8).eq(3), torch.arange(8) >= 4
    return graph, train, (graph, val), (graph, test), graph["x"]


def test_selected_test_rejects_changed_same_shape_data_before_loading_teacher(tmp_path, monkeypatch):
    prepared = problem()
    graph, train, val, test, _ = prepared
    root = tmp_path / "screen"
    root.mkdir()
    save_json(dict(dataset="cora", ratio=.026, data_digest=dataset_digest(graph, train, val, test)), root / "config.json")
    graph["x"][0, 0] += 1
    monkeypatch.setattr("src.citation_search._prepare_dataset", lambda *args: prepared)
    choice = dict(method="distance", width=0, lr=.03, T=1., rank=2, penalty=.1, step=1)
    with pytest.raises(ValueError, match="Graph or splits differ"):
        selected_test(root, choice, device="cpu")
    assert not (root / "selected.json").exists()


def test_distance_screen_uses_validation_only_and_replays_correct_feature_coordinates(tmp_path, monkeypatch):
    torch.manual_seed(0)
    prepared = problem()
    monkeypatch.setattr("src.citation_search._prepare_dataset", lambda *args: prepared)
    monkeypatch.setattr("src.citation_search.teacher_logits", lambda *args: (torch.randn(8, 2), .01))
    monkeypatch.setattr("src.citation_search.feature_kmeans", lambda *args: torch.arange(8) % 2)
    seen = []

    def evaluate(cx, cy, mass, graph, q, masks, seed, **kwargs):
        assert set(masks) == {"train", "val"}
        assert cx.shape == (2, 3)
        assert torch.allclose(cy.sum(1), torch.ones(2), atol=1e-6)
        seen.append(cx.clone())
        return dict(seed=seed, epoch=1, val_acc=50.)

    monkeypatch.setattr("src.citation_search.fit_gcn_diagnostic", evaluate)
    ranking, root = run_screen("cora", .026, tmp_path,
        [dict(method="distance", T=1., rank=2, penalty=.1, lr=.03)],
        steps=1, student_seeds=(0,), epochs=2, device="cpu")
    assert set(ranking.step) == {0, 1}
    assert len(seen) == 2
    # RMS inverse must recover the initial P-weighted means in original H space.
    from src.moments import initial_logits
    probability = initial_logits(torch.arange(8) % 2, 2).double().softmax(1)
    expected = probability.T @ prepared[-1].double() / probability.sum(0)[:, None]
    torch.testing.assert_close(seen[0], expected.float())
