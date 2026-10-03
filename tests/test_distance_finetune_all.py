import json

import pandas as pd
import torch

import src.distance_finetune_all as sweep
from src.distance_finetune_all import FIRST_RATIOS, source_matches


def test_source_reuse_requires_same_density_and_uniform_var_partition(tmp_path):
    config = dict(
        dataset="arxiv", ratio=0.005, method="moment_lloyd_normalized_variance",
        condensation_seeds=[0], loss_weighting="uniform", lloyd_options=dict(seeding="bound_var"),
    )
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "selected.json").write_text("{}")
    assert source_matches(tmp_path, "arxiv", 0.005)
    assert not source_matches(tmp_path, "arxiv", FIRST_RATIOS["arxiv"])
    config["loss_weighting"] = "mass"
    (tmp_path / "config.json").write_text(json.dumps(config))
    assert not source_matches(tmp_path, "arxiv", 0.005)


def test_first_density_uses_smallest_configured_budget():
    from src.data import BUDGET

    for dataset, ratio in FIRST_RATIOS.items():
        assert ratio == min(r for name, r in BUDGET if name == dataset)


def test_sequential_sweep_continues_after_one_method_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(sweep, "FIRST_RATIOS", {"cora": 0.013, "citeseer": 0.009})
    monkeypatch.setattr(sweep, "source_matches", lambda *args: True)
    calls = []

    def run(**options):
        method = options["methods"][0]
        calls.append((options["source"].name, method))
        assert options["inner_loss_weighting"] == "mass"
        assert options["continue_on_error"]
        if len(calls) == 1:
            raise RuntimeError("solver failure")
        root = options["output_dir"]
        root.mkdir(parents=True)
        return pd.DataFrame([dict(method=method, phase="selected")]), pd.DataFrame(), root

    monkeypatch.setattr(sweep, "run_distance_finetune", run)
    summary, status = sweep.run_all_distance_finetune(
        tmp_path, sources={name: tmp_path / name for name in ("cora", "citeseer")}, device="cpu",
    )
    assert calls == [("cora", "fixed_D"), ("cora", "svd_UV"), ("citeseer", "fixed_D"), ("citeseer", "svd_UV")]
    assert len(summary) == 3
    assert status["cora"]["fixed_D"]["state"] == "failed"
    assert status["citeseer"]["svd_UV"]["state"] == "complete"


def test_new_stage_one_routes_valid_var_part_option_to_real_solver(tmp_path, monkeypatch):
    from src.moment_lloyd import moment_lloyd_partition

    monkeypatch.setattr(sweep, "FIRST_RATIOS", {"cora": 0.013})
    calls = []

    def stage_one(**options):
        x = torch.tensor([[-2.0], [-1.0], [1.0], [2.0]], dtype=torch.float64)
        q = torch.tensor([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]], dtype=x.dtype)
        partition = moment_lloyd_partition(
            x, q, 2, mode="normalized_variance", seeding=options["seeding"],
            moment_weight=0.1, max_sweeps=10,
        )
        assert partition["assignment"].unique().numel() == 2
        calls.append(options["seeding"])
        return None, None, None, tmp_path / "source"

    def finetune(**options):
        root = options["output_dir"]
        root.mkdir(parents=True)
        return pd.DataFrame([dict(method=options["methods"][0])]), pd.DataFrame(), root

    monkeypatch.setattr(sweep, "run_risk_sweep", stage_one)
    monkeypatch.setattr(sweep, "run_distance_finetune", finetune)
    summary, status = sweep.run_all_distance_finetune(tmp_path, device="cpu")
    assert calls == ["bound_var"]
    assert len(summary) == 2
    assert status["cora"]["fixed_D"]["state"] == "complete"


def test_inner_weighting_grid_has_isolated_runs_and_selects_validation_not_test(tmp_path, monkeypatch):
    monkeypatch.setattr(sweep, "FIRST_RATIOS", {"cora": 0.013})
    monkeypatch.setattr(sweep, "source_matches", lambda *args: True)
    calls = []

    def run(**options):
        weighting = options["inner_loss_weighting"]
        method = options["methods"][0]
        root = options["output_dir"]
        root.mkdir(parents=True)
        calls.append((method, weighting, root))
        result = dict(method=method, phase="selected", step=25,
                      search_val=81 if weighting == "uniform" else 80,
                      test_mean=60 if weighting == "uniform" else 90)
        return pd.DataFrame([result]), pd.DataFrame(), root

    monkeypatch.setattr(sweep, "run_distance_finetune", run)
    summary, status = sweep.run_all_distance_finetune(
        tmp_path, sources={"cora": tmp_path / "source"}, device="cpu",
        inner_loss_weightings=["mass", "uniform"],
    )
    assert [(method, weighting) for method, weighting, _ in calls] == [
        ("fixed_D", "mass"), ("fixed_D", "uniform"), ("svd_UV", "mass"), ("svd_UV", "uniform"),
    ]
    assert len({root for _, _, root in calls}) == 4
    assert set(sweep.select_inner_weighting(summary).inner_loss_weighting) == {"uniform"}
    assert status["cora"]["svd_UV_uniform"]["state"] == "complete"
