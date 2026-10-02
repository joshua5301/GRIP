import json

import pandas as pd
import pytest
import torch

from src import variance_moment_sweep as experiment
from src.io import save_json


def test_teacher_preselection_uses_ce_not_accuracy():
    rows = [dict(gamma=0.1, val_ce=0.4, val_acc=90), dict(gamma=0.01, val_ce=0.3, val_acc=80)]
    assert experiment._select_teacher_ce(rows)["gamma"] == 0.01
    rows.append(dict(gamma=0.001, val_ce=0.3, val_acc=75))
    assert experiment._select_teacher_ce(rows)["gamma"] == 0.001
    with pytest.raises(ValueError):
        experiment._select_teacher_ce([dict(gamma=0.1, val_ce=float("nan"))])


def test_selection_averages_all_seed_pairs_and_ignores_test():
    rows = []
    for candidate, values in enumerate(([95, 60, 60], [80, 80, 80])):
        for condensation_seed, value in enumerate(values):
            for student_seed in (0, 1):
                rows.append(
                    dict(
                        candidate=candidate,
                        condensation_seed=condensation_seed,
                        student_seed=student_seed,
                        gamma=0.01,
                        T=1.0,
                        B=candidate + 1,
                        val_acc=value,
                        test_acc=100 - 30 * candidate,
                    )
                )
    scores = experiment._search_scores(rows, (0, 1, 2), (0, 1))
    assert experiment._select(scores)["candidate"] == 1
    with pytest.raises(ValueError):
        experiment._search_scores(rows[:-1], (0, 1, 2), (0, 1))
    with pytest.raises(ValueError):
        experiment._search_scores(rows + rows[:1], (0, 1, 2), (0, 1))
    ranked = [dict(row, rank=4 if row["candidate"] == 0 else 8) for row in rows]
    assert experiment._select(experiment._search_scores(ranked, (0, 1, 2), (0, 1)))["rank"] == 8
    weighted = [{**{k: v for k, v in row.items() if k != "B"}, "lambda": 8 / row["B"]} for row in ranked]
    selected = experiment._select(experiment._search_scores(weighted, (0, 1, 2), (0, 1)))
    assert selected["lambda"] == 4 and "B" not in selected


@pytest.mark.parametrize(
    "dataset,ratio,kernel,dropout,cells,inductive",
    [
        ("cora", 0.013, "relu", 0.9, 35, False),
        ("citeseer", 0.009, "erf", 0.5, 30, False),
        ("arxiv", 0.0005, "relu", 0.5, 90, False),
        ("flickr", 0.001, "relu", 0.5, 44, True),
        ("reddit", 0.0005, "erf", 0.5, 77, True),
    ],
)
@pytest.mark.parametrize("method", ["variance_moment", "variance_kl", "variance_moment_low_rank"])
def test_sweep_uniform_evaluation_selection_and_restart(
    tmp_path, monkeypatch, dataset, ratio, kernel, dropout, cells, inductive, method
):
    h = torch.arange(320, dtype=torch.float32).reshape(160, 2)
    graph = dict(x=h, y=torch.arange(160) % 2, adj=torch.eye(160).to_sparse_csr())
    train, val, test = (torch.arange(160) < 10), (torch.arange(160) == 10), (torch.arange(160) > 10)
    val_graph, test_graph = graph, graph
    if inductive:
        train = torch.ones(160, dtype=torch.bool)
        val_graph = dict(x=h[:6] + 1000, y=graph["y"][:6], adj=torch.eye(6).to_sparse_csr())
        test_graph = dict(x=h[:8] + 2000, y=graph["y"][:8], adj=torch.eye(8).to_sparse_csr())
        val, test = None, None

    def prepare(name, *args):
        assert name == dataset
        return graph, train, (val_graph, val), (test_graph, test), h

    monkeypatch.setattr(experiment, "_prepare_dataset", prepare)
    solves, fits = [], []

    def partition(features, labels, cells, B, seed, **kwargs):
        solves.append((B, seed))
        counts = torch.ones(cells, dtype=torch.long)
        counts[-1] += len(features) - cells
        return dict(
            x=torch.full((cells, 2), B),
            y=labels[:cells].float(),
            counts=counts,
            J=0.5,
            history=[1.0, 0.5],
            V=0.1,
            moment_error=0.1,
            sweeps=1,
            converged=True,
            seconds=0.01,
        )

    def features(x, kind, basis):
        assert kind == kernel
        return x.double()

    def kernel_values(a, b, kind):
        assert kind == kernel
        assert b.max() <= h.max()
        return torch.exp(-torch.cdist(a, b).square())

    teacher = dict(
        get_kernel_features=features,
        get_kernel_values=kernel_values,
        fit_logistic=lambda x, y, gamma: torch.zeros(x.shape[1], y.shape[1], dtype=x.dtype),
    )
    monkeypatch.setattr(
        experiment,
        "_load_reference",
        lambda: (
            {"risk_partition": "fake", "teacher": "fake"},
            {"risk_partition": {"risk_partition": partition, "seed_partition": None}, "teacher": teacher},
        ),
    )
    monkeypatch.setattr(experiment, "variance_kl_partition", partition)
    monkeypatch.setattr(experiment, "low_rank_partition", partition)

    def evaluate(x, y, mass, graph, q, masks, seed, folder, **kwargs):
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"seed_{seed}.json"
        if path.exists():
            return json.loads(path.read_text())
        assert kwargs["dropout"] == dropout
        if inductive:
            assert masks["train"][0] is graph and masks["val"][0] is val_graph
            if "test" in masks:
                assert masks["test"][0] is test_graph
        else:
            assert masks["val"] is val
        torch.testing.assert_close(mass / mass.sum(), torch.full_like(mass, 1 / len(mass)))
        B = round(float(x[0, 0]), 1)
        condensation_seed = int(folder.parent.name.split("_")[1])
        fits.append((B, condensation_seed, seed, "test" in masks))
        value = (95 if condensation_seed == 0 else 60) if B == 0.3 else 80
        score = dict(seed=seed, val_acc=value, train_acc=90, epoch=10)
        if "test" in masks:
            score["test_acc"] = 82 + condensation_seed + (seed - 100) * 0.1
        save_json(score, path)
        return score

    monkeypatch.setattr(experiment, "fit_gcn_diagnostic", evaluate)
    options = dict(
        ratio=ratio,
        dataset=dataset,
        method=method,
        output_dir=tmp_path,
        space=dict(gamma=[0.01], T=[1.0], B=[0.3, 1.0]),
        search_seeds=(0, 1),
        final_seeds=(100, 101),
        device="cpu",
        epochs=10,
    )
    run = experiment.run_cora_risk_sweep if dataset == "cora" else experiment.run_risk_sweep
    if method == "variance_moment_low_rank":
        options["space"]["rank"] = [8]
        options["assignment_initialization"] = "random"
    summary, by_seed, search, root = run(**options)
    if method == "variance_moment_low_rank":
        assert summary.iloc[0]["rank"] == 8
    assert summary.iloc[0].dataset == dataset and summary.iloc[0].nodes == cells
    config = json.loads((root / "config.json").read_text())
    assert config["teacher_kernel"] == kernel
    assert config["protocol"] == ("inductive" if inductive else "transductive")
    assert summary.iloc[0].B == 1
    assert len(solves) == 6 and len(fits) == 18
    assert all(B == 1 and seed >= 100 for B, _, seed, testing in fits if testing)
    assert summary.iloc[0].test_mean == pytest.approx(83.05)
    assert summary.iloc[0].test_std == pytest.approx(1.0)
    assert summary.iloc[0].mean_student_test_std == pytest.approx(0.1 / 2**0.5)
    assert set(by_seed.condensation_seed) == {0, 1, 2}
    assert len(pd.read_csv(root / "search_students.csv")) == 12
    before = len(fits), len(solves)
    repeated = run(**options)
    assert repeated[-1] == root and before == (len(fits), len(solves))
    pd.testing.assert_frame_equal(summary, repeated[0])
    with pytest.raises(ValueError, match="basis"):
        run(**dict(options, shared_run=root, basis=7))
    shared = run(**dict(options, shared_run=root, output_dir=tmp_path / "comparison"))
    torch.testing.assert_close(
        torch.load(root / "features.pt", weights_only=True),
        torch.load(shared[-1] / "features.pt", weights_only=True),
    )
    assert shared[0].iloc[0].test_mean == summary.iloc[0].test_mean


def test_inductive_teacher_uses_training_anchors_and_shared_mapping():
    h = torch.tensor([[0.0], [1.0], [2.0]], dtype=torch.double)
    graph = dict(x=h)
    val_graph = dict(
        x=torch.tensor([[4.0], [5.0]], dtype=torch.double), adj=torch.eye(2).to_sparse_csr().double()
    )
    calls = []

    def kernel(a, b, kind):
        calls.append((a.clone(), b.clone()))
        return torch.exp(-torch.cdist(a, b).square())

    phi, val_phi = experiment._teacher_features(
        h, graph, (val_graph, None), {"get_kernel_values": kernel}, "relu", 3
    )
    gram = torch.exp(-torch.cdist(h, h).square())
    chol = torch.linalg.cholesky(gram + 1e-8 * torch.eye(3, dtype=h.dtype))
    expected = torch.linalg.solve_triangular(
        chol, torch.exp(-torch.cdist(val_graph["x"], h).square()).T, upper=False
    ).T
    torch.testing.assert_close(val_phi, expected)
    torch.testing.assert_close(phi @ phi.T, gram, atol=1e-7, rtol=1e-7)
    assert all(torch.equal(anchors, h) for _, anchors in calls)


def test_digest_includes_inductive_validation_graph():
    graph = dict(x=torch.ones(3, 2), y=torch.zeros(3, dtype=torch.long), adj=torch.eye(3).to_sparse_csr())
    validation = {key: value.clone() for key, value in graph.items()}
    splits = dict(train=(graph, None), val=(validation, None))
    before = experiment._data_digest(splits)
    validation["x"][0, 0] += 1
    assert before != experiment._data_digest(splits)
