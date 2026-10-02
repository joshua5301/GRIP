import json

import pandas as pd
import pytest
import torch

from src import variance_moment_sweep as experiment
from src.io import save_json


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


@pytest.mark.parametrize(
    "dataset,ratio,kernel,dropout,cells",
    [("cora", 0.013, "relu", 0.9, 35), ("citeseer", 0.009, "erf", 0.5, 30)],
)
def test_sweep_uniform_evaluation_selection_and_restart(
    tmp_path, monkeypatch, dataset, ratio, kernel, dropout, cells
):
    h = torch.arange(80, dtype=torch.float32).reshape(40, 2)
    graph = dict(x=h, y=torch.arange(40) % 2, adj=torch.eye(40).to_sparse_csr())
    train, val, test = (torch.arange(40) < 10), (torch.arange(40) == 10), (torch.arange(40) > 10)

    def prepare(name, *args):
        assert name == dataset
        return graph, train, (graph, val), (graph, test), h

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

    teacher = dict(
        get_kernel_features=features,
        fit_logistic=lambda x, y, gamma: torch.zeros(x.shape[1], y.shape[1], dtype=x.dtype),
    )
    monkeypatch.setattr(
        experiment,
        "_load_reference",
        lambda: (
            {"risk_partition": "fake", "teacher": "fake"},
            {"risk_partition": {"risk_partition": partition}, "teacher": teacher},
        ),
    )

    def evaluate(x, y, mass, graph, q, masks, seed, folder, **kwargs):
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"seed_{seed}.json"
        if path.exists():
            return json.loads(path.read_text())
        assert kwargs["dropout"] == dropout
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
        output_dir=tmp_path,
        space=dict(gamma=[0.01], T=[1.0], B=[0.3, 1.0]),
        search_seeds=(0, 1),
        final_seeds=(100, 101),
        device="cpu",
        epochs=10,
    )
    run = experiment.run_cora_risk_sweep if dataset == "cora" else experiment.run_risk_sweep
    summary, by_seed, search, root = run(**options)
    assert summary.iloc[0].dataset == dataset and summary.iloc[0].nodes == cells
    config = json.loads((root / "config.json").read_text())
    assert config["teacher_kernel"] == kernel
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
