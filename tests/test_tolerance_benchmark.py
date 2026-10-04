import json

import torch

import src.tolerance_benchmark as benchmark
from src.io import _fingerprint


def test_tolerances_preserve_initialization_and_strict_validation(tmp_path, monkeypatch):
    source, reference = tmp_path / "source", tmp_path / "reference"
    (source / "teachers").mkdir(parents=True)
    reference.mkdir()
    folder = source / "candidate_0000" / "seed_0"
    folder.mkdir(parents=True)
    h = torch.tensor([[0.0], [1.0], [2.0], [3.0]], dtype=torch.float64)
    torch.save(h, source / "features.pt")
    torch.save({"logits": torch.zeros(4, 2)}, source / "teachers" / f"{_fingerprint(dict(gamma=0.1))}.pt")
    torch.save({"assignment": torch.tensor([0, 0, 1, 1])}, folder / "partition.pt")
    config = dict(dataset="cora", data_digest="same", search_seeds=[0], student={})
    (reference / "config.json").write_text(json.dumps(dict(
        source=str(source), source_config=config, selected=dict(candidate=0, gamma=0.1, T=1, **{"lambda": 0.1}),
    )))
    graph = {}
    mask = torch.ones(4, dtype=torch.bool)
    monkeypatch.setattr(benchmark, "_prepare_dataset", lambda *args: (graph, mask, (graph, mask), (graph, mask), h))
    monkeypatch.setattr(benchmark, "_data_digest", lambda splits: "same")
    calls, verifications = [], []

    def optimize(z, q, assignment, **options):
        calls.append(options)
        material = torch.cat((z.new_ones(2, 1), z[:2], q[:2]), 1) / 2
        snapshot = dict(moments=material, theta=z.new_zeros(2, 2), teacher_ce=1.0)
        columns = ("assignment_seconds", "inner_seconds", "outer_seconds", "implicit_seconds",
                   "backward_seconds", "cg_iterations", "inner_newton_cg_iterations", "inner_lbfgs_fallback")
        return dict(checkpoints={options["steps"]: snapshot}, history=[dict(step=1, **dict.fromkeys(columns, 1.0))])

    def verify(c, y, mass, penalty, **options):
        verifications.append(options)
        assert torch.allclose(mass, torch.full_like(mass, 0.5))
        return dict(theta=options["initial"], inner_converged=True, inner_grad_max=0.0)

    def evaluate(x, y, mass, graph, q, masks, seed, **options):
        assert "test" not in masks
        assert torch.equal(mass, torch.ones_like(mass))
        return dict(val_acc=80.0)

    monkeypatch.setattr(benchmark, "optimize_ce_assignment", optimize)
    monkeypatch.setattr(benchmark, "solve_inner_newton_first", verify)
    monkeypatch.setattr(benchmark, "outer_value_gradient", lambda *args, **kwargs: (1.0, None))
    monkeypatch.setattr(benchmark, "fit_gcn_diagnostic", evaluate)
    options = dict(rank=1, steps=2, repeats=2, device="cpu")
    results, root = benchmark.run_tolerance_benchmark(reference, tmp_path / "out", **options)
    assert len(results) == 6
    assert [(c["inner_tol"], c["cg_rtol"]) for c in calls] == list(benchmark.TOLERANCES.values()) + list(
        benchmark.TOLERANCES.values())[::-1]
    assert all(c["factor_seed"] == 0 and c["implicit_warm_start"] for c in calls)
    assert all(torch.equal(c["base_factors"][0], calls[0]["base_factors"][0]) for c in calls)
    assert all(v["grad_tol"] == 1e-7 for v in verifications)
    assert (root / "summary.csv").exists()
    benchmark.run_tolerance_benchmark(reference, tmp_path / "out", **options)
    assert len(calls) == 6
