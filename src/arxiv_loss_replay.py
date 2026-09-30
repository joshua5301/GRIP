import json
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm

from src.data import _prepare_dataset
from src.evaluation import fit_gcn_diagnostic
from src.io import _fingerprint, array_digest, save_json, write_table
from src.sweep_utils import representative
from src.transforms import FeatureTransform


def selected_uniform_results(frame, selected, condensation_seeds, student_seeds):
    rows = []
    expected = {(a, b) for a in condensation_seeds for b in student_seeds}
    for choice in selected.itertuples(index=False):
        group = frame[
            (frame.phase == "selected") & (frame.method == choice.method)
            & (frame.candidate == choice.candidate) & (frame.step == choice.step)
        ]
        if set(zip(group.condensation_seed, group.seed)) != expected or len(group) != len(expected):
            raise ValueError("Uniform reference must contain exactly the selected seed pairs")
        rows.append(group)
    return pd.concat(rows, ignore_index=True)


def run_arxiv_loss_replay(source_root, data_dir="/content/data/", device="cuda"):
    source = Path(source_root)
    config = json.loads((source / "config.json").read_text())
    if config["dataset"] != "arxiv" or config["student_loss"] != "uniform":
        raise ValueError("Expected an Arxiv uniform-CE width sweep")
    if not str(device).startswith("cuda"):
        raise ValueError("Run student training on a Colab CUDA GPU")
    selected = pd.read_csv(source / "selected.csv")
    if selected.empty or selected.method.duplicated().any():
        raise ValueError("Require one fixed selection per assignment family")
    cond_seeds, seeds = config["condensation_seeds"], config["final_seeds"]
    uniform = selected_uniform_results(
        pd.read_csv(source / "final_students.csv"), selected, cond_seeds, seeds
    )
    replay_config = dict(
        version=1, source=str(source.resolve()), data_digest=config["data_digest"],
        selections=selected.to_dict("records"), condensation_seeds=cond_seeds,
        final_seeds=seeds, student=config["full_student"], student_loss="mass",
        checkpoint_selection="fixed from uniform-CE validation search",
    )
    root = source / "mass_replay" / _fingerprint(replay_config)
    root.mkdir(parents=True, exist_ok=True)
    save_json(replay_config, root / "config.json")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    graph, train, validation, testing, h = _prepare_dataset("arxiv", data_dir, device)
    del h
    adjacency = graph["adj"]
    digest = array_digest(*(
        t.cpu().numpy() for t in (
            graph["x"], graph["y"], adjacency.crow_indices(), adjacency.col_indices(),
            adjacency.values(), train, validation[1], testing[1],
        )
    ))
    if digest != config["data_digest"]:
        raise ValueError("Graph or splits differ from the source sweep")
    saved = torch.load(source / "inputs.pt", map_location="cpu", weights_only=False)
    dimension = saved["z"].shape[1]
    transform = FeatureTransform(**{
        k: v.to(device) if torch.is_tensor(v) else v for k, v in saved["transform"].items()
    })
    del saved
    teacher = torch.load(source / "teacher.pt", map_location=device, weights_only=False)
    masks = dict(train=train, val=validation[1], test=testing[1])
    records = []
    progress = tqdm(total=len(selected) * len(cond_seeds) * len(seeds), desc="Mass-CE GCN replay")
    try:
        for choice in selected.itertuples(index=False):
            q = (teacher["logits"] / choice.T).softmax(1).double()
            for cond_seed in cond_seeds:
                folder = source / f"candidate_{int(choice.candidate):03d}" / f"condensation_{cond_seed}"
                snapshot = torch.load(
                    folder / "checkpoints" / f"step_{int(choice.step):06d}.pt",
                    map_location="cpu", weights_only=False,
                )
                x, y, mass = representative(snapshot["moments"], transform, dimension, device)
                cache = root / f"candidate_{int(choice.candidate):03d}" / f"condensation_{cond_seed}"
                for seed in seeds:
                    score = fit_gcn_diagnostic(
                        x, y, mass, graph, q, masks, seed, folder=cache, **config["full_student"]
                    )
                    records.append(dict(
                        method=choice.method, candidate=int(choice.candidate), step=int(choice.step),
                        condensation_seed=cond_seed, **score,
                    ))
                    write_table(pd.DataFrame(records), root / "mass_students.csv")
                    progress.update(1)
    finally:
        progress.close()
    keys = ["method", "candidate", "step", "condensation_seed", "seed"]
    paired = pd.DataFrame(records).merge(
        uniform[keys + ["val_acc", "test_acc"]].rename(columns={
            "val_acc": "uniform_val", "test_acc": "uniform_test",
        }), on=keys, validate="one_to_one",
    ).rename(columns={"val_acc": "mass_val", "test_acc": "mass_test"})
    paired["delta_val"] = paired.mass_val - paired.uniform_val
    paired["delta_test"] = paired.mass_test - paired.uniform_test
    by_seed = paired.groupby(["method", "candidate", "step", "condensation_seed"], sort=False).agg(
        uniform_val=("uniform_val", "mean"), mass_val=("mass_val", "mean"),
        uniform_test=("uniform_test", "mean"), mass_test=("mass_test", "mean"),
        delta_val=("delta_val", "mean"), delta_test=("delta_test", "mean"),
        mass_student_std=("mass_test", lambda v: v.std(ddof=0)),
    ).reset_index()
    summary = by_seed.groupby(["method", "candidate", "step"], sort=False).agg(
        uniform_val=("uniform_val", "mean"), mass_val=("mass_val", "mean"),
        uniform_test=("uniform_test", "mean"), mass_test=("mass_test", "mean"),
        delta_val=("delta_val", "mean"), delta_test=("delta_test", "mean"),
        mass_condensation_std=("mass_test", "std"),
        mean_mass_student_std=("mass_student_std", "mean"),
    ).reset_index()
    for name, table in (("paired_students", paired), ("by_seed", by_seed), ("summary", summary)):
        write_table(table, root / f"{name}.csv")
    return summary, by_seed, paired, root
