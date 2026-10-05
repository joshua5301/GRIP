import hashlib
import json
import shutil
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
from src.sgc_teacher import select_sgc_teacher
from src.variance_moment_sweep import _data_digest


def shared_source_identity(source, config):
    source = Path(source)
    previous = json.loads((source / "config.json").read_text())
    if previous.get("teacher_type", "gcn") != config.get("teacher_type", "gcn"):
        raise ValueError("Shared teacher source differs: teacher_type")
    for key in ("dataset", "data_digest", "teacher_seed", "settings", "teacher_grid"):
        if previous.get(key) != config.get(key):
            raise ValueError(f"Shared teacher source differs: {key}")
    return dict(path=str(source.resolve()), **{
        name.replace(".pt", "_sha256"): hashlib.sha256((source / name).read_bytes()).hexdigest()
        for name in ("teacher.pt", "features.pt")
    })


def width_source_identity(source, candidate, config):
    source = Path(source)
    previous = json.loads((source / "config.json").read_text())
    for key in ("dataset", "data_digest"):
        if previous.get(key) != config[key]:
            raise ValueError(f"Width teacher source differs: {key}")
    if previous.get("seed") != config["teacher_seed"]:
        raise ValueError("Width teacher source differs: seed")
    grid = pd.read_csv(source / "teacher_grid.csv")
    rows = grid.loc[grid.candidate == candidate]
    if len(rows) != 1:
        raise ValueError("Width teacher candidate must identify exactly one saved teacher")
    row = rows.iloc[0]
    width = int(row["width"] if "width" in row else row["hidden"])
    path = source / f"candidate_{int(candidate):03d}" / "teacher.pt"
    return dict(path=str(source.resolve()), candidate=int(candidate), hidden=width,
                dropout=float(row.dropout), training=previous["training"],
                teacher_path=str(path.resolve()), **{
                    name: hashlib.sha256(file.read_bytes()).hexdigest()
                    for name, file in (("teacher_sha256", path),
                                       ("features_sha256", source / "features.pt"))
                })


def load_width_teacher(identity, root, graph, testing):
    if (root / "teacher.pt").exists():
        return torch.load(root / "teacher.pt", map_location="cpu", weights_only=False)
    result = torch.load(identity["teacher_path"], map_location="cpu", weights_only=False)
    model = GCN(graph["x"].shape[1], identity["hidden"], int(graph["y"].max()) + 1,
                2, identity["dropout"]).to(graph["x"].device)
    model.load_state_dict(result["state"])
    if result["logits"].shape != (len(graph["x"]), int(graph["y"].max()) + 1):
        raise ValueError("Width teacher logits differ from the condensation graph")
    model.eval()
    with torch.no_grad():
        test = _metrics(model, testing)
    result.update(hidden=identity["hidden"], dropout=identity["dropout"],
                  test_acc=test["acc"], test_ce=test["ce"])
    save_state(result, root / "teacher.pt")
    save_json(identity, root / "selected_teacher.json")
    save_json({k: v for k, v in result.items() if k not in ("state", "logits")},
              root / "teacher_metrics.json")
    return result


def prepare_dropout_teachers(source, output_dir, dropouts=(0., .1, .5), hidden=256,
                              data_dir="/content/data/", device="cuda"):
    source = Path(source)
    previous = json.loads((source / "config.json").read_text())
    grid = pd.read_csv(source / "teacher_grid.csv")
    width_column = "hidden" if "hidden" in grid else "width"
    sources, missing = {}, []
    for dropout in dropouts:
        if not 0 <= dropout < 1:
            raise ValueError("Dropout must be in [0, 1)")
        matches = grid[(grid[width_column] == hidden) & ((grid.dropout - dropout).abs() < 1e-12)]
        if len(matches) > 1:
            raise ValueError("Ambiguous saved teacher configuration")
        if len(matches):
            candidate = int(matches.iloc[0].candidate)
            path = source / f"candidate_{candidate:03d}" / "teacher.pt"
            if not path.exists():
                raise FileNotFoundError(path)
            sources[dropout] = dict(source=str(source), candidate=candidate)
        else:
            missing.append(dropout)
    if not missing:
        return sources
    config = dict(dataset=previous["dataset"], data_digest=previous["data_digest"],
                  seed=previous["seed"], training=previous["training"], hidden=hidden,
                  dropouts=missing, source=str(source.resolve()),
                  features_sha256=hashlib.sha256((source / "features.pt").read_bytes()).hexdigest(),
                  code=hashlib.sha256(b"".join((Path(__file__).parent / name).read_bytes()
                                             for name in ("gcn_moment_sweep.py", "models.py", "data.py", "evaluation.py"))).hexdigest(),
                  torch=str(torch.__version__))
    root = Path(output_dir) / _fingerprint(config)
    root.mkdir(parents=True, exist_ok=True)
    save_json(config, root / "config.json")
    shutil.copy2(source / "features.pt", root / "features.pt")
    graph = train = validation = testing = None
    rows = []
    for candidate, dropout in enumerate(missing):
        folder = root / f"candidate_{candidate:03d}"
        if (folder / "teacher.pt").exists():
            teacher = torch.load(folder / "teacher.pt", map_location="cpu", weights_only=False)
        else:
            if graph is None:
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
                graph, train, validation, testing, computed = _prepare_dataset(config["dataset"], data_dir, device)
                del computed
                if _data_digest(dict(train=(graph, train), val=validation, test=testing)) != config["data_digest"]:
                    raise ValueError("Data differ from the saved teacher experiment")
            teacher = fit_teacher(graph, train, validation, None,
                                  dict(config["training"], hidden=hidden, dropout=dropout),
                                  config["seed"], folder)
        rows.append(dict(candidate=candidate, hidden=hidden, dropout=dropout,
                         val_acc=teacher["val_acc"], val_ce=teacher["val_ce"], epoch=teacher["epoch"]))
        write_table(pd.DataFrame(rows), root / "teacher_grid.csv")
        sources[dropout] = dict(source=str(root), candidate=candidate)
    return sources


def _metrics(model, pair):
    graph, mask = pair
    prediction = _forward(model, graph["x"], graph["adj"])
    labels = graph["y"]
    if mask is not None:
        prediction, labels = prediction[mask], labels[mask]
    return dict(acc=100 * float((prediction.argmax(1) == labels).double().mean()),
                ce=float(F.nll_loss(prediction, labels)))


def fit_teacher(graph, train, validation, testing, settings, seed, folder):
    folder.mkdir(parents=True, exist_ok=True)
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
                   if pair is not None for key, value in _metrics(model, pair).items()}
    result = dict(logits=scores, state=best, epoch=best_epoch, seed=seed, **metrics)
    save_state(result, path)
    save_json(dict(epoch=best_epoch, seed=seed, **metrics), folder / "teacher_metrics.json")
    write_table(pd.DataFrame(history), folder / "teacher_history.csv")
    return result


def select_teacher(graph, train, validation, testing, settings, seed, folder, dropouts, penalties):
    if (folder / "teacher.pt").exists():
        return torch.load(folder / "teacher.pt", map_location="cpu", weights_only=False)
    rows = []
    for index, (dropout, penalty) in enumerate(product(dropouts, penalties)):
        options = dict(settings, dropout=dropout, weight_decay=penalty)
        candidate = folder / "teacher_search" / f"candidate_{index:03d}"
        result = fit_teacher(graph, train, validation, None, options, seed, candidate)
        rows.append(dict(candidate=index, dropout=dropout, weight_decay=penalty,
                         val_acc=result["val_acc"], val_ce=result["val_ce"], epoch=result["epoch"]))
        write_table(pd.DataFrame(rows), folder / "teacher_grid.csv")
    selected = pd.DataFrame(rows).sort_values(
        ["val_acc", "val_ce", "candidate"], ascending=[False, True, True],
    ).iloc[0].to_dict()
    selected["candidate"] = int(selected["candidate"])
    path = folder / "teacher_search" / f"candidate_{selected['candidate']:03d}" / "teacher.pt"
    result = torch.load(path, map_location="cpu", weights_only=False)
    model = GCN(graph["x"].shape[1], settings["hidden"], result["logits"].shape[1],
                2, selected["dropout"]).to(graph["x"].device)
    model.load_state_dict(result["state"])
    model.eval()
    with torch.no_grad():
        test = _metrics(model, testing)
    result.update(test_acc=test["acc"], test_ce=test["ce"],
                  dropout=selected["dropout"], weight_decay=selected["weight_decay"])
    save_json(selected, folder / "selected_teacher.json")
    save_json({key: value for key, value in result.items() if key not in ("state", "logits")},
              folder / "teacher_metrics.json")
    save_state(result, folder / "teacher.pt")
    return result


def run_gcn_moment_sweep(dataset, ratio, output_dir, temperatures, lambdas,
                         search_seeds=(0, 1, 2, 3, 4), final_seeds=tuple(range(100, 110)),
                         teacher_seed=0, max_sweeps=100, block_size=1024,
                         epochs=1000, eval_every=10, hidden=256, dropout=None,
                         lr=0.01, weight_decay=0.0005, data_dir="/content/data/", device="cuda",
                         teacher_dropouts=None, teacher_weight_decays=None, partition_method="moment",
                         shared_teacher_source=None, initialization_space="features",
                         teacher_type="gcn", teacher_penalties=None,
                         width_teacher_source=None, width_teacher_candidate=None):
    if (width_teacher_source is None) != (width_teacher_candidate is None):
        raise ValueError("Supply both width_teacher_source and width_teacher_candidate")
    if width_teacher_source is not None and (
        teacher_type != "gcn" or shared_teacher_source is not None
        or teacher_dropouts is not None or teacher_weight_decays is not None
    ):
        raise ValueError("Width teacher reuse requires a GCN source without other teacher sources or grids")
    if teacher_type not in ("gcn", "sgc"):
        raise ValueError("Choose gcn or sgc teacher")
    if teacher_type == "sgc":
        if teacher_dropouts is not None or teacher_weight_decays is not None or not teacher_penalties:
            raise ValueError("SGC teacher requires only teacher_penalties")
        if any(not 0 < float(p) < float("inf") for p in teacher_penalties):
            raise ValueError("SGC penalties must be finite and positive")
    elif teacher_penalties is not None:
        raise ValueError("teacher_penalties requires an SGC teacher")
    if partition_method not in ("moment", "kmeans", "kl", "variance_sum"):
        raise ValueError("Choose moment, kmeans, kl, or variance_sum")
    if initialization_space not in ("features", "joint"):
        raise ValueError("Choose features or joint initialization")
    if initialization_space == "joint" and partition_method != "variance_sum":
        raise ValueError("Joint initialization requires variance_sum")
    if partition_method == "kmeans" and list(lambdas) != [0.0]:
        raise ValueError("K-means requires lambdas=[0.0]")
    if (dataset, ratio) not in BUDGET or not temperatures or not lambdas:
        raise ValueError("Use a configured density and nonempty grids")
    if any(not 0 < float(t) < float("inf") for t in temperatures):
        raise ValueError("Temperatures must be finite and positive")
    if any(not 0 <= float(w) < float("inf") for w in lambdas):
        raise ValueError("Lambdas must be finite and nonnegative")
    if not search_seeds or not final_seeds or set(search_seeds) & set(final_seeds):
        raise ValueError("Use nonempty disjoint search and final student seeds")
    teacher_grid = teacher_dropouts is not None or teacher_weight_decays is not None
    if teacher_grid:
        if not teacher_dropouts or not teacher_weight_decays:
            raise ValueError("Supply both nonempty teacher grids")
        if any(not 0 <= float(p) < 1 for p in teacher_dropouts):
            raise ValueError("Teacher dropout must be in [0, 1)")
        if any(not 0 <= float(p) < float("inf") for p in teacher_weight_decays):
            raise ValueError("Teacher weight decay must be finite and nonnegative")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    graph, train, validation, testing, h = (
        _prepare_dataset(dataset, data_dir, device, include_split_features=True)
        if teacher_type == "sgc" else _prepare_dataset(dataset, data_dir, device)
    )
    splits = dict(train=(graph, train), val=validation, test=testing)
    masks = splits if validation[0] is not graph else {key: pair[1] for key, pair in splits.items()}
    settings = dict(epochs=epochs, eval_every=eval_every, hidden=hidden,
                    dropout=(0.9 if dataset == "cora" else 0.5) if dropout is None else dropout,
                    lr=lr, weight_decay=weight_decay)
    files = ("gcn_moment_sweep.py", "moment_lloyd.py", "moment_seeding.py", "evaluation.py",
             "models.py", "data.py", "io.py")
    if teacher_type == "sgc":
        files += ("sgc_teacher.py", "head.py")
    code = hashlib.sha256(b"".join((Path(__file__).parent / name).read_bytes() for name in files)).hexdigest()
    config = dict(dataset=dataset, ratio=ratio, nodes=BUDGET[(dataset, ratio)],
                  temperatures=list(temperatures), lambdas=list(lambdas), teacher_seed=teacher_seed,
                  search_seeds=list(search_seeds), final_seeds=list(final_seeds), settings=settings,
                  max_sweeps=max_sweeps, block_size=block_size, data_digest=_data_digest(splits),
                  source_digest=code, torch=str(torch.__version__),
                  objective="RMS feature variance + lambda * global cross-moment Frobenius norm",
                  seeding="feature_var", mode="filtered_batch", student_loss="uniform")
    if partition_method == "kmeans":
        config.update(mode="variance_sum", objective="RMS feature variance", partition_method="kmeans")
    elif partition_method in ("kl", "variance_sum"):
        label_term = "label KL" if partition_method == "kl" else "label variance"
        config.update(mode=partition_method, partition_method=partition_method,
                      objective=f"RMS feature variance + lambda * {label_term}")
    if initialization_space == "joint":
        config.update(seeding="bound_var", initialization_space="joint")
    if teacher_grid:
        config["teacher_grid"] = dict(dropouts=list(teacher_dropouts), weight_decays=list(teacher_weight_decays))
    if teacher_type == "sgc":
        config.update(teacher_type="sgc", teacher_grid=dict(penalties=list(teacher_penalties)),
                      teacher_normalization="pool-centered global RMS", teacher_bias_regularized=True)
    if shared_teacher_source is not None:
        config["shared_source"] = shared_source_identity(shared_teacher_source, config)
    if width_teacher_source is not None:
        config["width_teacher_source"] = width_source_identity(
            width_teacher_source, width_teacher_candidate, config)
    root = Path(output_dir) / _fingerprint(config)
    root.mkdir(parents=True, exist_ok=True)
    save_json(config, root / "config.json")
    if width_teacher_source is not None:
        shutil.copy2(Path(width_teacher_source) / "features.pt", root / "features.pt")
    if shared_teacher_source is not None:
        for name in ("features.pt", "teacher.pt", "teacher_metrics.json", "teacher_grid.csv",
                     "selected_teacher.json", "teacher_history.csv"):
            source_path = Path(shared_teacher_source) / name
            if source_path.exists():
                shutil.copy2(source_path, root / name)
    if (root / "features.pt").exists():
        h = torch.load(root / "features.pt", map_location=device, weights_only=True)
    else:
        save_state(h, root / "features.pt")
    if width_teacher_source is not None:
        teacher = load_width_teacher(config["width_teacher_source"], root, graph, testing)
    elif teacher_type == "sgc":
        teacher = select_sgc_teacher(h, graph, train, validation, testing, teacher_penalties, root)
    elif teacher_grid:
        teacher = select_teacher(graph, train, validation, testing, settings, teacher_seed, root,
                                 teacher_dropouts, teacher_weight_decays)
    else:
        teacher = fit_teacher(graph, train, validation, testing, settings, teacher_seed, root)
    logits = teacher["logits"].to(device)
    kmeans_partition = None
    students, rows = [], []
    for index, (temperature, weight) in enumerate(tqdm(list(product(temperatures, lambdas)), desc=f"{dataset}: moment sweep")):
        folder = root / f"candidate_{index:04d}"
        folder.mkdir(exist_ok=True)
        path = folder / "partition.pt"
        if path.exists():
            partition = torch.load(path, map_location="cpu", weights_only=False)
        else:
            q = (logits / temperature).softmax(1)
            partition = moment_lloyd_partition(h, q, config["nodes"], mode=config["mode"],
                                              seeding=config["seeding"], moment_weight=weight,
                                              max_sweeps=max_sweeps, block_size=block_size) if kmeans_partition is None else dict(kmeans_partition)
            if partition_method == "kmeans":
                if not partition["converged"]:
                    raise RuntimeError("K-means did not converge; increase max_sweeps")
                assignment = partition["assignment"].to(q.device)
                counts = torch.bincount(assignment, minlength=config["nodes"]).to(q)
                partition["y"] = (q.new_zeros(config["nodes"], q.shape[1]).index_add_(
                    0, assignment, q) / counts[:, None]).float().cpu()
            save_state(partition, path)
        if partition_method == "kmeans":
            kmeans_partition = partition
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
