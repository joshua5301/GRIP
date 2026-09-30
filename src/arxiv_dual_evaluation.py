import json
from pathlib import Path

import pandas as pd
import torch
from torch_geometric import seed_everything
from tqdm.auto import tqdm

from src.arxiv_loss_replay import selected_uniform_results
from src.data import _prepare_dataset
from src.evaluation import _forward
from src.io import _fingerprint, array_digest, save_json, write_table
from src.models import GCN
from src.sweep_utils import representative
from src.transforms import FeatureTransform


def select_route_epochs(history):
    if not history:
        raise ValueError("No evaluated student epochs")
    return {route: dict(max(history, key=lambda row: row[f"{route}_val"])) for route in ("gcn", "mlp")}


@torch.no_grad()
def route_metrics(probability, labels, val_mask, test_mask):
    prediction = probability.argmax(1)
    return {
        name: 100 * float((prediction[mask] == labels[mask]).double().mean())
        for name, mask in (("val", val_mask), ("test", test_mask))
    }


def fit_dual_student(x, y, graph, propagated, val_mask, test_mask, seed, settings, folder):
    folder.mkdir(parents=True, exist_ok=True)
    cache = folder / f"seed_{seed}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    seed_everything(seed)
    model = GCN(x.shape[1], settings["hidden"], y.shape[1], 2, settings["dropout"]).to(x.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"])
    weights = x.new_full((len(x),), 1 / len(x))
    history = []
    for epoch in range(1, settings["epochs"] + 1):
        if epoch == settings["epochs"] // 2:
            optimizer = torch.optim.Adam(
                model.parameters(), lr=settings["lr"] * 0.1, weight_decay=settings["weight_decay"]
            )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = -(weights[:, None] * y * _forward(model, x)).sum()
        loss.backward()
        optimizer.step()
        if epoch % settings["eval_every"] != 0 and epoch != settings["epochs"]:
            continue
        model.eval()
        with torch.no_grad():
            gcn = route_metrics(_forward(model, graph["x"], graph["adj"]), graph["y"], val_mask, test_mask)
            mlp = route_metrics(_forward(model, propagated), graph["y"], val_mask, test_mask)
        history.append(dict(epoch=epoch, **{f"gcn_{k}": v for k, v in gcn.items()},
                            **{f"mlp_{k}": v for k, v in mlp.items()}))
    result = dict(seed=seed, **select_route_epochs(history))
    write_table(pd.DataFrame(history), folder / f"seed_{seed}_epochs.csv")
    save_json(result, cache)
    return result


def run_arxiv_dual_evaluation(source_root, data_dir="/content/data/", device="cuda"):
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
    reference = selected_uniform_results(
        pd.read_csv(source / "final_students.csv"), selected, cond_seeds, seeds
    )
    protocol = dict(
        version=1, source=str(source.resolve()), data_digest=config["data_digest"],
        selections=selected.to_dict("records"), condensation_seeds=cond_seeds, student_seeds=seeds,
        student=config["full_student"], student_loss="uniform", synthetic_adjacency="identity",
        mlp_input="Saved inverse-transformed propagated features S^2X; no graph propagation",
        gcn_input="Original X and normalized adjacency",
        epoch_selection=["GCN validation epoch, identical weights for both routes",
                         "Each route's validation epoch on the same training trajectory"],
    )
    root = source / "dual_evaluation" / _fingerprint(protocol)
    root.mkdir(parents=True, exist_ok=True)
    save_json(protocol, root / "config.json")
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
    inputs = torch.load(source / "inputs.pt", map_location="cpu", weights_only=False)
    transform = FeatureTransform(**{
        k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs["transform"].items()
    })
    if transform.kind != "rms" or transform.matrix is not None:
        raise ValueError("Exact inverse propagation features require the source RMS transform")
    z = inputs["z"].to(device)
    propagated = (z * transform.scale + transform.output_center + transform.center).float()
    dimension = z.shape[1]
    del inputs, z
    rows = []
    progress = tqdm(total=len(selected) * len(cond_seeds) * len(seeds), desc="Paired MLP / GCN student fits")
    try:
        for choice in selected.itertuples(index=False):
            for cond_seed in cond_seeds:
                checkpoint = source / f"candidate_{int(choice.candidate):03d}" / f"condensation_{cond_seed}"
                snapshot = torch.load(
                    checkpoint / "checkpoints" / f"step_{int(choice.step):06d}.pt",
                    map_location="cpu", weights_only=False,
                )
                if snapshot["step"] != int(choice.step):
                    raise ValueError("Checkpoint step differs from the fixed selection")
                x, y, _ = representative(snapshot["moments"], transform, dimension, device)
                folder = root / f"candidate_{int(choice.candidate):03d}" / f"condensation_{cond_seed}"
                for seed in seeds:
                    result = fit_dual_student(
                        x, y, graph, propagated, validation[1], testing[1], seed,
                        config["full_student"], folder,
                    )
                    common, mlp_best = result["gcn"], result["mlp"]
                    for selection in ("gcn_epoch", "route_epochs"):
                        mlp = common if selection == "gcn_epoch" else mlp_best
                        rows.append(dict(
                            method=choice.method, candidate=int(choice.candidate), step=int(choice.step),
                            condensation_seed=cond_seed, seed=seed, epoch_selection=selection,
                            gcn_epoch=common["epoch"], mlp_epoch=mlp["epoch"],
                            gcn_val=common["gcn_val"], gcn_test=common["gcn_test"],
                            mlp_val=mlp["mlp_val"], mlp_test=mlp["mlp_test"],
                        ))
                    write_table(pd.DataFrame(rows), root / "students.csv")
                    progress.update(1)
    finally:
        progress.close()
    students = pd.DataFrame(rows)
    keys = ["method", "candidate", "step", "condensation_seed", "seed"]
    students = students.merge(
        reference[keys + ["val_acc", "test_acc"]].rename(columns={
            "val_acc": "reference_gcn_val", "test_acc": "reference_gcn_test",
        }), on=keys, validate="many_to_one",
    )
    students["gcn_replay_val_delta"] = students.gcn_val - students.reference_gcn_val
    students["gcn_replay_test_delta"] = students.gcn_test - students.reference_gcn_test
    students["gcn_minus_mlp_val"] = students.gcn_val - students.mlp_val
    students["gcn_minus_mlp_test"] = students.gcn_test - students.mlp_test
    by_seed = students.groupby(["method", "epoch_selection", "condensation_seed"], sort=False).agg(
        mlp_val=("mlp_val", "mean"), gcn_val=("gcn_val", "mean"),
        mlp_test=("mlp_test", "mean"), gcn_test=("gcn_test", "mean"),
        gcn_minus_mlp_val=("gcn_minus_mlp_val", "mean"),
        gcn_minus_mlp_test=("gcn_minus_mlp_test", "mean"),
        mlp_student_std=("mlp_test", lambda v: v.std(ddof=0)),
        gcn_student_std=("gcn_test", lambda v: v.std(ddof=0)),
    ).reset_index()
    summary = by_seed.groupby(["method", "epoch_selection"], sort=False).agg(
        mlp_val=("mlp_val", "mean"), gcn_val=("gcn_val", "mean"),
        mlp_test=("mlp_test", "mean"), gcn_test=("gcn_test", "mean"),
        gcn_minus_mlp_val=("gcn_minus_mlp_val", "mean"),
        gcn_minus_mlp_test=("gcn_minus_mlp_test", "mean"),
        mlp_condensation_std=("mlp_test", "std"), gcn_condensation_std=("gcn_test", "std"),
        mean_mlp_student_std=("mlp_student_std", "mean"),
        mean_gcn_student_std=("gcn_student_std", "mean"),
    ).reset_index()
    replay_check = students[students.epoch_selection == "gcn_epoch"].groupby("method", sort=False).agg(
        mean_val_difference=("gcn_replay_val_delta", "mean"),
        max_abs_val_difference=("gcn_replay_val_delta", lambda v: v.abs().max()),
        mean_test_difference=("gcn_replay_test_delta", "mean"),
        max_abs_test_difference=("gcn_replay_test_delta", lambda v: v.abs().max()),
    ).reset_index()
    for name, table in (("students", students), ("by_seed", by_seed), ("summary", summary),
                        ("gcn_replay_check", replay_check)):
        write_table(table, root / f"{name}.csv")
    return summary, by_seed, students, replay_check, root
