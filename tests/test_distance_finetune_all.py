import json

import pandas as pd

import src.distance_finetune_all as sweep
from src.distance_finetune_all import FIRST_RATIOS, source_matches


def test_source_reuse_requires_same_density_and_uniform_var_partition(tmp_path):
    config = dict(
        dataset="arxiv", ratio=0.005, method="moment_lloyd_normalized_variance",
        condensation_seeds=[0], loss_weighting="uniform", lloyd_options=dict(seeding="var"),
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
