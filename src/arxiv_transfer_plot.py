import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

from src.data import _prepare_dataset
from src.io import array_digest, save_json, write_table
from src.soft_ce_partition import classification


@torch.no_grad()
def plot_arxiv_transfer(source_root, data_dir="/content/data/", device="cuda"):
    source = Path(source_root)
    config = json.loads((source / "config.json").read_text())
    if config["dataset"] != "arxiv" or config["student_loss"] != "uniform":
        raise ValueError("Expected the uniform-CE Arxiv width sweep")
    search = pd.read_csv(source / "search_students.csv")
    candidates = pd.read_csv(source / "candidates.csv").set_index("candidate")
    expected = {(a, b) for a in config["condensation_seeds"] for b in config["search_seeds"]}
    for _, group in search.groupby(["candidate", "step"]):
        if set(zip(group.condensation_seed, group.seed)) != expected or len(group) != len(expected):
            raise ValueError("Incomplete or duplicate GCN search seed pairs")
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
    z = saved["z"].to(device)
    del saved
    root = source / "transfer_diagnostic"
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for (index, step, cond_seed), group in search.groupby(["candidate", "step", "condensation_seed"]):
        snapshot = torch.load(
            source / f"candidate_{int(index):03d}" / f"condensation_{int(cond_seed)}"
            / "checkpoints" / f"step_{int(step):06d}.pt",
            map_location="cpu", weights_only=False,
        )
        if snapshot["step"] != step or not snapshot["J_exact"]:
            raise ValueError("Expected an exact inner-student checkpoint at the requested step")
        inner_val, inner_val_ce = classification(z, graph["y"], validation[1], snapshot["theta"].to(z))
        candidate = candidates.loc[index]
        rows.append(dict(
            candidate=int(index), method=candidate.method, width=int(candidate.width),
            lr=float(candidate.lr), step=int(step), condensation_seed=int(cond_seed),
            inner_val=inner_val, inner_val_ce=inner_val_ce,
            outer_ce=snapshot["teacher_ce"], gcn_val=group.val_acc.mean(),
            gcn_student_std=group.val_acc.std(ddof=0),
        ))
    by_seed = pd.DataFrame(rows)
    curves = by_seed.groupby(["candidate", "method", "width", "lr", "step"], sort=True).agg(
        inner_val=("inner_val", "mean"), inner_condensation_std=("inner_val", "std"),
        gcn_val=("gcn_val", "mean"), gcn_condensation_std=("gcn_val", "std"),
        outer_ce=("outer_ce", "mean"), inner_val_ce=("inner_val_ce", "mean"),
    ).reset_index()
    write_table(by_seed, root / "by_seed.csv")
    write_table(curves, root / "curves.csv")
    save_json(dict(
        inner="Saved mass-weighted linear student, evaluated on normalized S^2X",
        gcn="Cached uniform-CE 2-layer GCN, evaluated on original X and adjacency",
        aggregation="GCN student seeds averaged within condensation seed, then across condensation seeds",
        bands="Sample standard deviation across condensation seeds; not confidence intervals",
        splits="Validation only; no test metrics or new training",
    ), root / "protocol.json")
    groups = list(curves.groupby("candidate", sort=True))
    colors = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True, constrained_layout=True)
    relation, relation_axis = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
    for order, (index, group) in enumerate(groups):
        group = group.sort_values("step")
        first = group.iloc[0]
        label = "Low-rank" if first.method == "low_rank" else f"MLP width {int(first.width)}"
        label += f" (lr={first.lr:g})"
        color = colors(order % 10)
        for axis, metric, deviation in (
            (axes[0], "inner_val", "inner_condensation_std"),
            (axes[1], "gcn_val", "gcn_condensation_std"),
        ):
            axis.plot(group.step, group[metric], marker="o", markersize=4, color=color, label=label)
            axis.fill_between(
                group.step.to_numpy(), (group[metric] - group[deviation]).to_numpy(),
                (group[metric] + group[deviation]).to_numpy(), color=color, alpha=0.12,
            )
        relation_axis.plot(group.outer_ce, group.gcn_val, marker="o", color=color, label=label)
        for row in (group.iloc[0], group.iloc[-1]):
            relation_axis.annotate(
                f"step {int(row.step)}", (row.outer_ce, row.gcn_val), xytext=(5, 5),
                textcoords="offset points", fontsize=8, color=color,
            )
    for axis, title in zip(axes, ("Inner linear student on propagated features", "GCN on original graph")):
        axis.set_title(title)
        axis.set_xlabel("Condensation step")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Validation accuracy (%)")
    axes[1].legend(fontsize=8, loc="best")
    fig.suptitle("Arxiv 0.05%: mean and condensation-seed SD")
    relation_axis.set_xlabel("Outer teacher CE (lower is better)")
    relation_axis.set_ylabel("GCN validation accuracy (%)")
    relation_axis.set_title("Outer objective and GCN transfer")
    relation_axis.grid(alpha=0.2)
    relation_axis.legend(fontsize=8, loc="best")
    for name, figure in (("validation_trajectories", fig), ("outer_ce_vs_gcn", relation)):
        figure.savefig(root / f"{name}.png", dpi=200)
        figure.savefig(root / f"{name}.pdf")
    return curves, by_seed, (fig, relation), root
