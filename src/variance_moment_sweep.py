import hashlib
import math
import subprocess
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric import seed_everything
from tqdm.auto import tqdm

from src.data import BUDGET, _prepare_dataset
from src.evaluation import fit_gcn_diagnostic
from src.io import _fingerprint, array_digest, save_json, save_state, write_table
from src.sweep_utils import grid_rows

REFERENCE = "12ceec5810e2c709f98f8aa518d401cd04390168"
DEFAULT_SPACE = dict(
    gamma=[0.001, 0.01, 0.1, 1.0],
    T=[0.2, 0.5, 1.0, 2.0],
    B=[0.3, 1.0, 3.0, 10.0, 30.0],
)


def _load_reference():
    sources, modules = {}, {}
    for name in ("risk_partition", "teacher"):
        source = subprocess.check_output(
            ["git", "show", f"{REFERENCE}:src/{name}.py"],
            cwd=Path(__file__).resolve().parents[1],
            encoding="utf-8",
        )
        sources[name] = source
        modules[name] = {}
        exec(compile(source, f"{REFERENCE}/{name}.py", "exec"), modules[name])
    return sources, modules


def _search_scores(rows, condensation_seeds, search_seeds):
    frame = pd.DataFrame(rows)
    expected = {(c, s) for c in condensation_seeds for s in search_seeds}
    records = []
    for candidate, group in frame.groupby("candidate", sort=True):
        pairs = list(zip(group.condensation_seed, group.student_seed))
        if (
            len(pairs) != len(expected)
            or set(pairs) != expected
            or not group.val_acc.map(math.isfinite).all()
        ):
            raise ValueError("Candidate evaluation requires every condensation/student seed pair")
        means = group.groupby("condensation_seed").val_acc.mean()
        records.append(
            dict(
                candidate=int(candidate),
                **{key: float(group.iloc[0][key]) for key in ("gamma", "T", "B")},
                search_val=float(means.mean()),
                condensation_val_std=float(means.std()),
                student_val_std=float(group.groupby("condensation_seed").val_acc.std().mean()),
            )
        )
    return pd.DataFrame(records)


def _select(search):
    return search.sort_values(["search_val", "candidate"], ascending=[False, True]).iloc[0].to_dict()


def run_cora_risk_sweep(
    ratio,
    output_dir,
    space=None,
    condensation_seeds=(0, 1, 2),
    search_seeds=(0, 1, 2),
    final_seeds=tuple(range(100, 110)),
    max_sweeps=100,
    block_size=1024,
    basis=3000,
    teacher_seed=0,
    epochs=1000,
    eval_every=10,
    hidden=256,
    dropout=0.9,
    student_lr=0.01,
    weight_decay=0.0005,
    data_dir="/content/data/",
    device="cuda",
):
    space = DEFAULT_SPACE if space is None else space
    if set(space) != {"gamma", "T", "B"} or ("cora", ratio) not in BUDGET:
        raise ValueError("Use a Cora ratio and grid keys gamma, T, B")
    candidates = grid_rows(space)
    if any(not math.isfinite(value) or value <= 0 for row in candidates for value in row.values()):
        raise ValueError("Grid values must be finite and positive")
    if any(
        not seeds or len(seeds) != len(set(seeds))
        for seeds in (condensation_seeds, search_seeds, final_seeds)
    ):
        raise ValueError("Seed lists must be nonempty and unique")
    if set(search_seeds) & set(final_seeds):
        raise ValueError("Search and final student seeds must be disjoint")
    if min(max_sweeps, block_size, basis, epochs, eval_every, hidden) < 1:
        raise ValueError("Solver and student budgets must be positive")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sources, modules = _load_reference()
    solver, teacher = modules["risk_partition"]["risk_partition"], modules["teacher"]
    graph, train, validation, testing, h = _prepare_dataset("cora", data_dir, device)
    masks = dict(train=train, val=validation[1], test=testing[1])
    adjacency = graph["adj"]
    data_digest = array_digest(
        *[
            value.cpu().numpy()
            for value in (
                graph["x"],
                graph["y"],
                adjacency.crow_indices(),
                adjacency.col_indices(),
                adjacency.values(),
                *masks.values(),
            )
        ]
    )
    settings = dict(
        epochs=epochs,
        eval_every=eval_every,
        hidden=hidden,
        dropout=dropout,
        lr=student_lr,
        weight_decay=weight_decay,
    )
    code = b"".join(
        (Path(__file__).parent / name).read_bytes()
        for name in (
            "variance_moment_sweep.py",
            "evaluation.py",
            "models.py",
            "data.py",
            "io.py",
            "sweep_utils.py",
        )
    )
    config = dict(
        version=1,
        dataset="cora",
        ratio=ratio,
        nodes=BUDGET[("cora", ratio)],
        reference=REFERENCE,
        source_digest=hashlib.sha256(code).hexdigest(),
        reference_digest=_fingerprint(sources),
        data_digest=data_digest,
        space=space,
        condensation_seeds=list(condensation_seeds),
        search_seeds=list(search_seeds),
        final_seeds=list(final_seeds),
        max_sweeps=max_sweeps,
        block_size=block_size,
        basis=basis,
        teacher_seed=teacher_seed,
        teacher_kernel="relu",
        student=settings,
        loss_weighting="uniform",
        layers=2,
        objective="B**2/4 * variance + 2*B * global_moment_norm",
        initialization="historical_risk_surrogate",
        torch=str(torch.__version__),
    )
    root = Path(output_dir) / f"ratio_{ratio:g}" / _fingerprint(config)
    root.mkdir(parents=True, exist_ok=True)
    save_json(config, root / "config.json")
    for name, source in sources.items():
        (root / f"reference_{name}.py").write_text(source, encoding="utf-8")
    feature_path = root / "features.pt"
    if feature_path.exists():
        h = torch.load(feature_path, map_location=device, weights_only=True)
    else:
        save_state(h, feature_path)
    save_json(dict(h_digest=array_digest(h.cpu().numpy())), root / "features.json")

    teacher_dir = root / "teachers"
    teacher_dir.mkdir(exist_ok=True)
    logits, teacher_rows, phi = {}, [], None
    for gamma in tqdm(list(dict.fromkeys(space["gamma"])), desc="Teacher gamma cache"):
        path = teacher_dir / f"{_fingerprint(dict(gamma=gamma))}.pt"
        if path.exists():
            scores = torch.load(path, map_location=device, weights_only=True)
        else:
            if phi is None:
                seed_everything(teacher_seed)
                phi = teacher["get_kernel_features"](h, "relu", basis)
            targets = F.one_hot(graph["y"][train], int(graph["y"].max()) + 1).to(phi)
            weights = teacher["fit_logistic"](phi[train], targets, gamma)
            scores = (phi @ weights).detach()
            save_state(scores, path)
        logits[gamma] = scores
        teacher_rows.append(
            dict(
                gamma=gamma,
                val_acc=100
                * float((scores[masks["val"]].argmax(1) == graph["y"][masks["val"]]).double().mean()),
                val_ce=float(F.cross_entropy(scores[masks["val"]], graph["y"][masks["val"]])),
            )
        )
    del phi
    write_table(pd.DataFrame(teacher_rows), root / "teacher_grid.csv")

    rows, diagnostics = [], []
    progress = tqdm(enumerate(candidates), total=len(candidates), desc=f"Cora {ratio:g}: risk grid")
    for index, candidate in progress:
        q = (logits[candidate["gamma"]] / candidate["T"]).softmax(1)
        for seed in condensation_seeds:
            folder = root / f"candidate_{index:04d}" / f"seed_{seed}"
            folder.mkdir(parents=True, exist_ok=True)
            artifact = folder / "partition.pt"
            if artifact.exists():
                partition = torch.load(artifact, map_location="cpu", weights_only=False)
            else:
                partition = solver(
                    h,
                    q,
                    config["nodes"],
                    candidate["B"],
                    seed=seed,
                    max_sweeps=max_sweeps,
                    block_size=block_size,
                )
                save_state(partition, artifact)
            if not math.isfinite(partition["J"]) or partition["J"] > partition["history"][0] + 1e-8:
                raise RuntimeError("Partition objective is invalid")
            diagnostics.append(
                dict(
                    candidate=index,
                    condensation_seed=seed,
                    **candidate,
                    nodes=len(partition["x"]),
                    J_initial=partition["history"][0],
                    J_final=partition["J"],
                    variance=partition["V"],
                    moment_error=partition["moment_error"],
                    sweeps=partition["sweeps"],
                    converged=partition["converged"],
                    partition_seconds=partition["seconds"],
                    partition_path=str(artifact),
                )
            )
            x, y = partition["x"].to(device), partition["y"].to(device)
            uniform = torch.ones(len(x), device=device)
            for student_seed in search_seeds:
                score = fit_gcn_diagnostic(
                    x,
                    y,
                    uniform,
                    graph,
                    None,
                    {key: masks[key] for key in ("train", "val")},
                    student_seed,
                    folder=folder / "search",
                    **settings,
                )
                rows.append(
                    dict(
                        **score,
                        candidate=index,
                        condensation_seed=seed,
                        student_seed=student_seed,
                        **candidate,
                    )
                )
        search = _search_scores(rows, condensation_seeds, search_seeds)
        progress.set_postfix(best_val=f"{search.search_val.max():.3f}%")
        write_table(pd.DataFrame(rows), root / "search_students.csv")
        write_table(search, root / "search.csv")
        write_table(pd.DataFrame(diagnostics), root / "partitions.csv")

    selected = _select(search)
    index = int(selected["candidate"])
    selected["candidate"] = index
    save_json(selected, root / "selected.json")
    final = []
    for seed in tqdm(condensation_seeds, desc="Selected setting: final GCNs"):
        folder = root / f"candidate_{index:04d}" / f"seed_{seed}"
        partition = torch.load(folder / "partition.pt", map_location="cpu", weights_only=False)
        x, y = partition["x"].to(device), partition["y"].to(device)
        for student_seed in final_seeds:
            score = fit_gcn_diagnostic(
                x,
                y,
                torch.ones(len(x), device=device),
                graph,
                None,
                masks,
                student_seed,
                folder=folder / "final",
                **settings,
            )
            final.append(dict(**score, condensation_seed=seed, student_seed=student_seed))
        write_table(pd.DataFrame(final), root / "final_students.csv")
    by_seed = (
        pd.DataFrame(final)
        .groupby("condensation_seed", as_index=False)
        .agg(
            final_val=("val_acc", "mean"),
            final_val_std=("val_acc", "std"),
            test_mean=("test_acc", "mean"),
            test_std=("test_acc", "std"),
        )
    )
    partitions = pd.DataFrame(diagnostics)
    by_seed = by_seed.merge(
        partitions[partitions.candidate == index], on="condensation_seed", validate="one_to_one"
    )
    summary = pd.DataFrame(
        [
            dict(
                dataset="cora",
                ratio=ratio,
                nodes=config["nodes"],
                **selected,
                final_val=float(by_seed.final_val.mean()),
                final_val_std=float(by_seed.final_val.std()),
                test_mean=float(by_seed.test_mean.mean()),
                test_std=float(by_seed.test_mean.std()),
                mean_student_test_std=float(by_seed.test_std.mean()),
                J_initial=float(by_seed.J_initial.mean()),
                J_final=float(by_seed.J_final.mean()),
                converged=bool(by_seed.converged.all()),
                max_sweeps_used=int(by_seed.sweeps.max()),
                partition_seconds=float(by_seed.partition_seconds.mean()),
                output_dir=str(root),
            )
        ]
    )
    write_table(by_seed, root / "by_seed.csv")
    write_table(summary, root / "summary.csv")
    return summary, by_seed, search, root
