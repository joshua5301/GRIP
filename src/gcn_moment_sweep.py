import hashlib
from itertools import product
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric import seed_everything
from tqdm.auto import tqdm

from src.data import BUDGET, _prepare_dataset
from src.evaluation import _forward, fit_gcn_diagnostic
from src.io import _fingerprint, save_json, save_state, write_table
from src.models import GCN
from src.moment_lloyd import moment_lloyd_partition
from src.variance_moment_sweep import _data_digest


def _metrics(model, pair):
    graph, mask = pair
    prediction = _forward(model, graph["x"], graph["adj"])
    labels = graph["y"]
    if mask is not None:
        prediction, labels = prediction[mask], labels[mask]
    return dict(acc=100 * float((prediction.argmax(1) == labels).double().mean()),
                ce=float(F.nll_loss(prediction, labels)))


def fit_teacher(graph, train, validation, testing, settings, seed, folder):
    path = folder / "teacher.pt"
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=False)
    seed_everything(seed)
    model = GCN(graph["x"].shape[1], settings["hidden"], int(graph["y"].max()) + 1,
                2, settings["dropout"]).to(graph["x"].device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"])
    best, best_acc, best_epoch, history = None, -float("inf"), 0, []
    for epoch in tqdm(range(1, settings["epochs"] + 1), desc="Full GCN teacher"):
        if epoch == settings["epochs"] // 2:
            optimizer = torch.optim.Adam(model.parameters(), lr=settings["lr"] * 0.1,
                                         weight_decay=settings["weight_decay"])
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = _forward(model, graph["x"], graph["adj"])
        loss = F.nll_loss(prediction[train], graph["y"][train])
        loss.backward()
        optimizer.step()
        if epoch % settings["eval_every"] and epoch != settings["epochs"]:
            continue
        model.eval()
        with torch.no_grad():
            metrics = _metrics(model, validation)
        history.append(dict(epoch=epoch, val_acc=metrics["acc"], val_ce=metrics["ce"]))
        if metrics["acc"] > best_acc:
            best_acc, best_epoch = metrics["acc"], epoch
            best = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best)
    model.eval()
    with torch.no_grad():
        scores = _forward(model, graph["x"], graph["adj"]).cpu()
        metrics = {f"{name}_{key}": value for name, pair in
                   (("train", (graph, train)), ("val", validation), ("test", testing))
                   for key, value in _metrics(model, pair).items()}
    result = dict(logits=scores, state=best, epoch=best_epoch, seed=seed, **metrics)
    save_state(result, path)
    save_json(dict(epoch=best_epoch, seed=seed, **metrics), folder / "teacher_metrics.json")
    write_table(pd.DataFrame(history), folder / "teacher_history.csv")
    return result


def run_gcn_moment_sweep(dataset, ratio, output_dir, temperatures, lambdas,
                         search_seeds=(0, 1, 2, 3, 4), final_seeds=tuple(range(100, 110)),
                         teacher_seed=0, max_sweeps=100, block_size=1024,
                         epochs=1000, eval_every=10, hidden=256, dropout=None,
                         lr=0.01, weight_decay=0.0005, data_dir="/content/data/", device="cuda"):
    if (dataset, ratio) not in BUDGET or not temperatures or not lambdas:
        raise ValueError("Use a configured density and nonempty grids")
    if any(not 0 < float(t) < float("inf") for t in temperatures):
        raise ValueError("Temperatures must be finite and positive")
    if any(not 0 <= float(w) < float("inf") for w in lambdas):
        raise ValueError("Lambdas must be finite and nonnegative")
    if not search_seeds or not final_seeds or set(search_seeds) & set(final_seeds):
        raise ValueError("Use nonempty disjoint search and final student seeds")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    graph, train, validation, testing, h = _prepare_dataset(dataset, data_dir, device)
    splits = dict(train=(graph, train), val=validation, test=testing)
    masks = splits if validation[0] is not graph else {key: pair[1] for key, pair in splits.items()}
    settings = dict(epochs=epochs, eval_every=eval_every, hidden=hidden,
                    dropout=(0.9 if dataset == "cora" else 0.5) if dropout is None else dropout,
                    lr=lr, weight_decay=weight_decay)
    files = ("gcn_moment_sweep.py", "moment_lloyd.py", "moment_seeding.py", "evaluation.py",
             "models.py", "data.py", "io.py")
    code = hashlib.sha256(b"".join((Path(__file__).parent / name).read_bytes() for name in files)).hexdigest()
    config = dict(dataset=dataset, ratio=ratio, nodes=BUDGET[(dataset, ratio)],
                  temperatures=list(temperatures), lambdas=list(lambdas), teacher_seed=teacher_seed,
                  search_seeds=list(search_seeds), final_seeds=list(final_seeds), settings=settings,
                  max_sweeps=max_sweeps, block_size=block_size, data_digest=_data_digest(splits),
                  source_digest=code, torch=str(torch.__version__),
                  objective="RMS feature variance + lambda * global cross-moment Frobenius norm",
                  seeding="feature_var", mode="filtered_batch", student_loss="uniform")
    root = Path(output_dir) / _fingerprint(config)
    root.mkdir(parents=True, exist_ok=True)
    save_json(config, root / "config.json")
    if (root / "features.pt").exists():
        h = torch.load(root / "features.pt", map_location=device, weights_only=True)
    else:
        save_state(h, root / "features.pt")
    teacher = fit_teacher(graph, train, validation, testing, settings, teacher_seed, root)
    logits = teacher["logits"].to(device)
    students, rows = [], []
    for index, (temperature, weight) in enumerate(tqdm(list(product(temperatures, lambdas)), desc=f"{dataset}: moment sweep")):
        folder = root / f"candidate_{index:04d}"
        folder.mkdir(exist_ok=True)
        path = folder / "partition.pt"
        if path.exists():
            partition = torch.load(path, map_location="cpu", weights_only=False)
        else:
            q = (logits / temperature).softmax(1)
            partition = moment_lloyd_partition(h, q, config["nodes"], mode="filtered_batch",
                                              seeding="feature_var", moment_weight=weight,
                                              max_sweeps=max_sweeps, block_size=block_size)
            save_state(partition, path)
        x, y = partition["x"].to(device), partition["y"].to(device)
        scores = []
        for seed in search_seeds:
            score = fit_gcn_diagnostic(x, y, torch.ones(len(x), device=device), graph, None,
                                       {key: masks[key] for key in ("train", "val")}, seed,
                                       folder=folder / "search", **settings)
            scores.append(score["val_acc"])
            students.append(dict(candidate=index, T=temperature, **{"lambda": weight}, **score))
        rows.append(dict(candidate=index, T=temperature, **{"lambda": weight},
                         search_val=sum(scores) / len(scores), J_initial=partition["history"][0],
                         J_final=partition["J"], sweeps=partition["sweeps"],
                         status=partition["status"], converged=partition["converged"],
                         partition_seconds=partition["seconds"]))
        write_table(pd.DataFrame(rows), root / "search.csv")
        write_table(pd.DataFrame(students), root / "search_students.csv")
    search = pd.DataFrame(rows)
    selected = search.sort_values(["search_val", "candidate"], ascending=[False, True]).iloc[0].to_dict()
    selected["candidate"] = int(selected["candidate"])
    save_json(selected, root / "selected.json")
    folder = root / f"candidate_{selected['candidate']:04d}"
    partition = torch.load(folder / "partition.pt", map_location=device, weights_only=False)
    x, y = partition["x"], partition["y"]
    final = pd.DataFrame([
        fit_gcn_diagnostic(x, y, torch.ones(len(x), device=device), graph, None, masks, seed,
                           folder=folder / "final", **settings)
        for seed in tqdm(final_seeds, desc="Final GCN evaluation")
    ])
    summary = pd.DataFrame([dict(dataset=dataset, ratio=ratio, nodes=config["nodes"], **selected,
                                 final_val=final.val_acc.mean(), test_mean=final.test_acc.mean(),
                                 test_std=final.test_acc.std(), teacher_val=teacher["val_acc"],
                                 teacher_test=teacher["test_acc"], teacher_epoch=teacher["epoch"],
                                 output_dir=str(root))])
    write_table(final, root / "final_students.csv")
    write_table(summary, root / "summary.csv")
    return summary, search, root
