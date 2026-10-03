import gc
import json
import traceback
from pathlib import Path
from time import perf_counter

import pandas as pd
import torch

from src.data import BUDGET
from src.distance_finetune import run_distance_finetune
from src.io import save_json, write_table
from src.variance_moment_sweep import run_risk_sweep

FIRST_RATIOS = dict(cora=0.013, citeseer=0.009, flickr=0.001, reddit=0.0005, arxiv=0.0005)


def source_matches(source, dataset, ratio):
    source = Path(source)
    if not (source / "selected.json").is_file() or not (source / "config.json").is_file():
        return False
    config = json.loads((source / "config.json").read_text())
    return (
        config.get("dataset") == dataset
        and config.get("ratio") == ratio
        and config.get("method") == "moment_lloyd_normalized_variance"
        and config.get("condensation_seeds") == [0]
        and config.get("loss_weighting") == "uniform"
        and config.get("lloyd_options", {}).get("seeding") == "var"
    )


def run_all_distance_finetune(output_dir, sources=None, ranks=(8, 16, 32, 64),
                              taus=(0.1, 0.3, 1.0, 3.0), penalties=(1e-6, 1e-5, 1e-4),
                              steps=300, checkpoints=(0, 10, 25, 50, 100, 200, 300),
                              lr=0.01, data_dir="/content/data/", device="cuda"):
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    sources = {} if sources is None else sources
    status_path = root / "status.json"
    status, tables = {}, []
    for dataset, ratio in FIRST_RATIOS.items():
        cells = BUDGET[(dataset, ratio)]
        grid_ranks = [rank for rank in ranks if rank <= cells]
        if not grid_ranks:
            raise ValueError(f"No valid ranks for {dataset}: {cells} cells")
        print(f"\n{dataset} | ratio={ratio} | nodes={cells} | ranks={grid_ranks}", flush=True)
        started = perf_counter()
        try:
            source = sources.get(dataset)
            if source is None or not source_matches(source, dataset, ratio):
                _, _, _, source = run_risk_sweep(
                    dataset=dataset, ratio=ratio, output_dir=root / "stage_one" / dataset,
                    method="moment_lloyd_normalized_variance", seeding="var",
                    teacher_selection="accuracy_only",
                    space={
                        "gamma": [0.0001, 0.001, 0.003, 0.01, 0.03, 0.1] if dataset == "cora"
                        else [0.001, 0.003, 0.01, 0.03, 0.1, 0.3],
                        "T": [0.1, 0.3, 0.5, 1.0, 2.0],
                        "lambda": [0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0],
                    },
                    condensation_seeds=[0], search_seeds=list(range(5)),
                    final_seeds=list(range(100, 110)), max_sweeps=300,
                    basis=3000, epochs=1000, eval_every=10, hidden=256,
                    data_dir=data_dir, device=device,
                )
            source = Path(source)
            status[dataset] = dict(source=str(source), stage_one_seconds=perf_counter() - started)
            save_json(status, status_path)
        except Exception:
            status[dataset] = dict(stage="stage_one", error=traceback.format_exc())
            save_json(status, status_path)
            print(status[dataset]["error"], flush=True)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        for method in ("fixed_D", "svd_UV"):
            started = perf_counter()
            try:
                summary, _, run = run_distance_finetune(
                    source=source, output_dir=root / dataset / method,
                    methods=[method], ranks=grid_ranks, taus=taus, penalties=penalties,
                    steps=steps, checkpoints=checkpoints, lr=lr,
                    inner_loss_weighting="mass", data_dir=data_dir, device=device,
                    continue_on_error=True,
                )
                table = summary.assign(dataset=dataset, ratio=ratio, nodes=cells, run_dir=str(run))
                tables.append(table)
                write_table(pd.concat(tables, ignore_index=True), root / "summary.csv")
                status[dataset][method] = dict(
                    state="complete", output_dir=str(run), seconds=perf_counter() - started,
                    failed_candidates=len(pd.read_csv(run / "failures.csv")) if (run / "failures.csv").exists() else 0,
                )
                print(table.to_string(index=False), flush=True)
            except Exception:
                status[dataset][method] = dict(
                    state="failed", error=traceback.format_exc(), seconds=perf_counter() - started,
                )
                print(status[dataset][method]["error"], flush=True)
            finally:
                save_json(status, status_path)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame(), status
