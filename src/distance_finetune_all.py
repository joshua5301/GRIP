import gc
import itertools
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
        and config.get("lloyd_options", {}).get("seeding") == "bound_var"
    )


def select_inner_weighting(summary):
    selected = summary.loc[summary.phase == "selected"]
    return selected.sort_values(
        ["search_val", "step", "inner_loss_weighting"], ascending=[False, True, True],
    ).drop_duplicates(["dataset", "method"]).reset_index(drop=True)


def run_all_distance_finetune(output_dir, sources=None, ranks=(8, 16, 32, 64),
                              taus=(0.1, 0.3, 1.0, 3.0), penalties=(1e-6, 1e-5, 1e-4),
                              steps=300, checkpoints=(0, 10, 25, 50, 100, 200, 300),
                              lr=0.01, data_dir="/content/data/", device="cuda",
                              inner_loss_weightings=("mass",)):
    if (not inner_loss_weightings or len(set(inner_loss_weightings)) != len(inner_loss_weightings)
            or any(value not in ("mass", "uniform") for value in inner_loss_weightings)):
        raise ValueError("Choose unique mass and/or uniform inner CE weightings")
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
                    method="moment_lloyd_normalized_variance", seeding="bound_var",
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

        for method, inner_weighting in itertools.product(("fixed_D", "svd_UV"), inner_loss_weightings):
            key = method if len(inner_loss_weightings) == 1 else f"{method}_{inner_weighting}"
            output = root / dataset / method
            if len(inner_loss_weightings) > 1:
                output = output / f"inner_{inner_weighting}"
            started = perf_counter()
            try:
                summary, _, run = run_distance_finetune(
                    source=source, output_dir=output,
                    methods=[method], ranks=grid_ranks, taus=taus, penalties=penalties,
                    steps=steps, checkpoints=checkpoints, lr=lr,
                    inner_loss_weighting=inner_weighting, data_dir=data_dir, device=device,
                    continue_on_error=True,
                )
                table = summary.assign(dataset=dataset, ratio=ratio, nodes=cells, run_dir=str(run),
                                       inner_loss_weighting=inner_weighting)
                tables.append(table)
                combined = pd.concat(tables, ignore_index=True)
                write_table(combined, root / "summary.csv")
                if "step" in combined:
                    write_table(select_inner_weighting(combined), root / "selected_by_validation.csv")
                status[dataset][key] = dict(
                    state="complete", output_dir=str(run), seconds=perf_counter() - started,
                    method=method, inner_loss_weighting=inner_weighting,
                    failed_candidates=len(pd.read_csv(run / "failures.csv")) if (run / "failures.csv").exists() else 0,
                )
                print(table.to_string(index=False), flush=True)
            except Exception:
                status[dataset][key] = dict(
                    state="failed", error=traceback.format_exc(), seconds=perf_counter() - started,
                    method=method, inner_loss_weighting=inner_weighting,
                )
                print(status[dataset][key]["error"], flush=True)
            finally:
                save_json(status, status_path)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame(), status
