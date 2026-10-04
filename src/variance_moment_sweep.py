import gc
import hashlib
import json
import math
import shutil
import subprocess
from functools import partial
from pathlib import Path
from time import perf_counter

import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric import seed_everything
from tqdm.auto import tqdm

from src.data import BUDGET, _prepare_dataset
from src.evaluation import fit_gcn_diagnostic
from src.io import _fingerprint, array_digest, save_json, save_state, write_table
from src.moment_lloyd import moment_lloyd_partition
from src.sweep_utils import grid_rows
from src.teacher_calibration import calibrate_temperature, select_accuracy
from src.variance_kl import variance_kl_partition
from src.variance_moment_low_rank import low_rank_partition

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
                **{
                    key: float(group.iloc[0][key])
                    for key in ("gamma", "T", "B", "lambda", "assignment_lr")
                    if key in group
                },
                **({"rank": int(group.iloc[0]["rank"])} if "rank" in group else {}),
                search_val=float(means.mean()),
                condensation_val_std=float(means.std()),
                student_val_std=float(group.groupby("condensation_seed").val_acc.std().mean()),
            )
        )
    return pd.DataFrame(records)


def _select(search):
    return search.sort_values(["search_val", "candidate"], ascending=[False, True]).iloc[0].to_dict()


def _select_teacher_ce(rows):
    frame = pd.DataFrame(rows)
    if frame.empty or not frame.val_ce.map(math.isfinite).all():
        raise ValueError("Teacher validation CE must be finite")
    return frame.sort_values(["val_ce", "gamma"]).iloc[0].to_dict()


def _data_digest(splits):
    graphs, result = {}, {}
    for name, (graph, mask) in splits.items():
        key = id(graph)
        if key not in graphs:
            adjacency = graph["adj"]
            graphs[key] = array_digest(
                *[
                    value.cpu().numpy()
                    for value in (
                        graph["x"],
                        graph["y"],
                        adjacency.crow_indices(),
                        adjacency.col_indices(),
                        adjacency.values(),
                    )
                ]
            )
        result[name] = dict(
            graph=graphs[key], mask=None if mask is None else array_digest(mask.cpu().numpy())
        )
    return _fingerprint(result)


def _teacher_features(h, graph, validation, teacher, kernel, basis):
    val_graph, mask = validation
    if val_graph is graph:
        phi = teacher["get_kernel_features"](h, kernel, basis)
        return phi, phi[mask]
    x = h.double()
    anchors = x if basis >= len(x) else x[torch.randperm(len(x))[:basis]]
    values = teacher["get_kernel_values"]
    gram = values(anchors, anchors, kernel)
    gram = (gram + gram.T) / 2
    eye = torch.eye(len(anchors), dtype=x.dtype, device=x.device)
    chol = torch.linalg.cholesky(gram + 1e-8 * gram.diagonal().mean() * eye)
    mapping = torch.linalg.solve_triangular(chol, eye, upper=False).T

    def project(features):
        return torch.cat(
            [values(block.double(), anchors, kernel) @ mapping for block in features.split(8192)]
        )

    val_h = torch.sparse.mm(val_graph["adj"], torch.sparse.mm(val_graph["adj"], val_graph["x"]))
    return project(x), project(val_h if mask is None else val_h[mask])


def run_risk_sweep(
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
    dropout=None,
    student_lr=0.01,
    weight_decay=0.0005,
    data_dir="/content/data/",
    device="cuda",
    dataset="cora",
    method="variance_moment",
    shared_run=None,
    assignment_rank=8,
    assignment_steps=1000,
    assignment_lr=0.01,
    assignment_mixing=0.05,
    assignment_initialization="historical",
    assignment_backend="auto",
    teacher_selection="grid",
    candidate_subset=None,
    evaluate_test=True,
    initialization_steps=20,
    require_initialization_convergence=False,
    backtrack_steps=8,
    temperature_bounds=(0.05, 20.0),
    seeding="feature",
    greedy_trials=4,
):
    if teacher_selection not in ("grid", "validation_ce", "accuracy_only", "accuracy_then_ce", "calibrated_ce"):
        raise ValueError("Use grid or validation_ce teacher selection")
    lloyd = method in ("moment_lloyd_hybrid", "moment_lloyd_full_only", "moment_lloyd_filtered_batch", "moment_lloyd_variance", "moment_lloyd_normalized_variance", "moment_lloyd_raw_variance", "moment_lloyd_kl")
    if method not in ("variance_moment", "variance_kl", "variance_moment_low_rank") and not lloyd:
        raise ValueError("Unknown partition method")
    if (dataset, ratio) not in BUDGET:
        raise ValueError("Use a configured dataset and ratio")
    kernel = "erf" if dataset in ("citeseer", "reddit") else "relu"
    if dropout is None:
        dropout = 0.9 if dataset == "cora" else 0.5
    if space is None:
        bounds = {"cora": DEFAULT_SPACE["B"], "citeseer": [0.03, 0.1, 0.3, 1.0, 3.0]}
        space = dict(DEFAULT_SPACE, B=bounds.get(dataset, [0.1, 0.3, 1.0, 3.0, 10.0]))
    keys = set(space)
    if teacher_selection in ("accuracy_then_ce", "calibrated_ce") and (space.get("T") != [1.0] or candidate_subset is not None):
        raise ValueError("Calibrated teachers require placeholder T=[1.0] and no candidate subset")
    if teacher_selection == "validation_ce" and space.get("T") != [1.0]:
        raise ValueError("Teacher CE preselection requires T=[1.0]")
    required = {"gamma", "T", "lambda" if "lambda" in keys else "B"}
    optional = {"rank", "assignment_lr"} if method == "variance_moment_low_rank" else set()
    if not required <= keys or not keys <= required | optional:
        raise ValueError("Use gamma, T and B or lambda; low-rank also supports rank and assignment_lr")
    if lloyd and "lambda" not in keys:
        raise ValueError("Moment Lloyd requires a lambda grid")
    if "lambda" in keys and not lloyd and (
        method != "variance_moment_low_rank" or assignment_initialization not in ("random", "distance")
    ):
        raise ValueError("Lambda grid requires random or distance low-rank optimization")
    candidates = grid_rows(space)
    if candidate_subset is not None:
        if not candidate_subset or any(candidate not in candidates for candidate in candidate_subset):
            raise ValueError("Candidate subset must be nonempty and belong to the supplied grid")
        if len({_fingerprint(candidate) for candidate in candidate_subset}) != len(candidate_subset):
            raise ValueError("Candidate subset must not contain duplicates")
    if "rank" in space and any(isinstance(r, bool) or not isinstance(r, int) or r < 1 for r in space["rank"]):
        raise ValueError("Ranks must be positive integers")
    if any(
        not math.isfinite(value) or (value < 0 if lloyd and key == "lambda" else value <= 0)
        for row in candidates for key, value in row.items()
    ):
        raise ValueError("Grid values must be finite and positive (Lloyd lambda may be zero)")
    if any(
        not seeds or len(seeds) != len(set(seeds))
        for seeds in (condensation_seeds, search_seeds, final_seeds)
    ):
        raise ValueError("Seed lists must be nonempty and unique")
    if set(search_seeds) & set(final_seeds):
        raise ValueError("Search and final student seeds must be disjoint")
    if max_sweeps < (0 if lloyd else 1) or min(block_size, basis, epochs, eval_every, hidden) < 1:
        raise ValueError("Solver and student budgets must be positive")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sources, modules = _load_reference()
    solver, teacher = modules["risk_partition"]["risk_partition"], modules["teacher"]
    if lloyd:
        solver = partial(
            moment_lloyd_partition, mode=method.removeprefix("moment_lloyd_"),
            initialization_steps=initialization_steps,
            require_initialization_convergence=require_initialization_convergence,
            backtrack_steps=backtrack_steps,
            seeding=seeding, greedy_trials=greedy_trials,
        )
    if method == "variance_kl":
        solver = partial(variance_kl_partition, seed_partition=modules["risk_partition"]["seed_partition"])
    if method == "variance_moment_low_rank":
        solver = partial(
            low_rank_partition,
            seed_partition=modules["risk_partition"]["seed_partition"],
            rank=assignment_rank,
            steps=assignment_steps,
            lr=assignment_lr,
            mixing=assignment_mixing,
            initialization=assignment_initialization,
            backend=assignment_backend,
        )
    graph, train, validation, testing, h = _prepare_dataset(dataset, data_dir, device)
    splits = dict(train=(graph, train), val=validation, test=testing)
    inductive = validation[0] is not graph
    masks = splits if inductive else {name: split[1] for name, split in splits.items()}
    data_digest = _data_digest(splits)
    val_graph, val_mask = validation
    val_labels = val_graph["y"] if val_mask is None else val_graph["y"][val_mask]
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
            "variance_kl.py",
            "variance_moment_low_rank.py",
            "moment_lloyd.py",
            "moment_seeding.py",
            "teacher_calibration.py",
            "distance_initialization.py",
            "evaluation.py",
            "models.py",
            "data.py",
            "io.py",
            "sweep_utils.py",
        )
    )
    config = dict(
        version=1,
        dataset=dataset,
        ratio=ratio,
        nodes=BUDGET[(dataset, ratio)],
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
        teacher_kernel=kernel,
        teacher_selection=teacher_selection,
        protocol="inductive" if inductive else "transductive",
        student=settings,
        loss_weighting="uniform",
        layers=2,
        objective="B**2/4 * variance + 2*B * global_moment_norm",
        initialization="historical_risk_surrogate",
        torch=str(torch.__version__),
    )
    config["method"] = method
    if teacher_selection in ("accuracy_then_ce", "calibrated_ce"):
        config["temperature_bounds"] = list(temperature_bounds)
    if lloyd:
        config["initialization"] = f"moment_seeding_{seeding}"
        config["lloyd_options"] = dict(
            seeding=seeding, greedy_trials=greedy_trials,
            initialization_steps=initialization_steps,
            require_initialization_convergence=require_initialization_convergence,
            backtrack_steps=backtrack_steps,
        )
    if candidate_subset is not None:
        config["candidate_subset"] = candidate_subset
    if "lambda" in space:
        config["objective"] = "variance + lambda * global_moment_norm"
    if method == "moment_lloyd_variance":
        config["objective"] = "a * feature_variance + b * label_variance; bound-derived a,b"
    if method == "moment_lloyd_kl":
        config["objective"] = "feature_variance + lambda * mean KL(q_i || cell_mean)"
    if method == "moment_lloyd_normalized_variance":
        config["objective"] = "feature_variance / global_feature_variance + alpha * label_variance / global_label_variance"
        config["coefficient"] = "lambda grid stores alpha; global variances fixed before clustering; zero-variance terms omitted"
    if method == "moment_lloyd_raw_variance":
        config["objective"] = "raw propagated feature variance + lambda * label variance"
        config["coefficient"] = "lambda is absolute; no feature RMS or label variance normalization"
    if method == "variance_moment_low_rank":
        config["assignment"] = dict(
            rank=space.get("rank", assignment_rank),
            steps=assignment_steps,
            lr=space.get("assignment_lr", assignment_lr),
            mixing=assignment_mixing if assignment_initialization == "historical" else None,
            initialization=assignment_initialization,
            backend=assignment_backend,
        )
        config["initialization"] = assignment_initialization
    if method == "variance_kl":
        config["objective"] = "B**2/2 * variance + 8 * mean_forward_label_KL"
    if shared_run is not None:
        shared_run = Path(shared_run)
        previous = json.loads((shared_run / "config.json").read_text(encoding="utf-8"))
        keys = (
            "dataset",
            "ratio",
            "nodes",
            "reference_digest",
            "data_digest",
            "basis",
            "teacher_seed",
            "teacher_kernel",
            "protocol",
            "torch",
            "student",
            "loss_weighting",
            "search_seeds",
            "final_seeds",
            "block_size",
        )
        changed = [key for key in keys if previous.get(key) != config[key]]
        if changed:
            raise ValueError(f"Shared comparison run differs: {changed}")
        config["shared_run"] = str(shared_run.resolve())
    root = Path(output_dir) / f"ratio_{ratio:g}" / _fingerprint(config)
    root.mkdir(parents=True, exist_ok=True)
    save_json(config, root / "config.json")
    for name, source in sources.items():
        (root / f"reference_{name}.py").write_text(source, encoding="utf-8")
    feature_path = root / "features.pt"
    if not feature_path.exists() and shared_run is not None:
        shutil.copy2(shared_run / "features.pt", feature_path)
    if feature_path.exists():
        h = torch.load(feature_path, map_location=device, weights_only=True)
    else:
        save_state(h, feature_path)
    save_json(dict(h_digest=array_digest(h.cpu().numpy())), root / "features.json")

    teacher_dir = root / "teachers"
    teacher_dir.mkdir(exist_ok=True)
    if shared_run is not None:
        for source in (shared_run / "teachers").glob("*.pt"):
            target = teacher_dir / source.name
            if not source.stem.endswith(".tmp") and not target.exists():
                shutil.copy2(source, target)
    logits, teacher_rows, phi, val_phi, weights = {}, [], None, None, None
    for gamma in tqdm(list(dict.fromkeys(space["gamma"])), desc="Teacher gamma cache"):
        path = teacher_dir / f"{_fingerprint(dict(gamma=gamma))}.pt"
        if path.exists():
            saved = torch.load(path, map_location=device, weights_only=True)
            scores, val_scores = saved["logits"], saved["validation_logits"]
        else:
            if phi is None:
                seed_everything(teacher_seed)
                phi, val_phi = _teacher_features(h, graph, validation, teacher, kernel, basis)
            targets = F.one_hot(graph["y"][train], int(graph["y"].max()) + 1).to(phi)
            weights = teacher["fit_logistic"](phi[train], targets, gamma)
            scores = (phi @ weights).detach()
            val_scores = (val_phi @ weights).detach()
            save_state(dict(logits=scores, validation_logits=val_scores), path)
        logits[gamma] = scores
        teacher_rows.append(
            dict(
                gamma=gamma,
                val_acc=100 * float((val_scores.argmax(1) == val_labels).double().mean()),
                val_ce=float(F.cross_entropy(val_scores, val_labels)),
            )
        )
    del phi, val_phi, weights
    write_table(pd.DataFrame(teacher_rows), root / "teacher_grid.csv")
    if teacher_selection == "calibrated_ce":
        calibrated, curves = [], []
        for row in teacher_rows:
            gamma = row["gamma"]
            saved = torch.load(teacher_dir / f"{_fingerprint(dict(gamma=gamma))}.pt", map_location="cpu", weights_only=True)
            calibration, curve = calibrate_temperature(saved["validation_logits"], val_labels, temperature_bounds)
            calibrated.append(dict(row, **calibration))
            curves.extend(dict(point, gamma=gamma) for point in curve)
        table = pd.DataFrame(calibrated).sort_values(["calibrated_val_ce", "gamma"])
        selected_teacher = table.iloc[0].to_dict()
        gamma = selected_teacher["gamma"]
        save_json(selected_teacher, root / "selected_teacher.json")
        write_table(table, root / "calibrated_teacher_grid.csv")
        write_table(pd.DataFrame(curves), root / "all_temperature_curves.csv")
        write_table(pd.DataFrame(curves).query("gamma == @gamma").drop(columns="gamma"), root / "temperature_grid.csv")
        candidates = [dict(candidate, T=selected_teacher["T"]) for candidate in candidates if candidate["gamma"] == gamma]
        logits = {gamma: logits[gamma]}
    if teacher_selection == "accuracy_only":
        selected_teacher = select_accuracy(teacher_rows)
        gamma = selected_teacher["gamma"]
        save_json(selected_teacher, root / "selected_teacher.json")
        candidates = [candidate for candidate in candidates if candidate["gamma"] == gamma]
        logits = {gamma: logits[gamma]}
    if teacher_selection == "accuracy_then_ce":
        selected_teacher = select_accuracy(teacher_rows)
        gamma = selected_teacher["gamma"]
        saved = torch.load(teacher_dir / f"{_fingerprint(dict(gamma=gamma))}.pt", map_location="cpu", weights_only=True)
        calibration, curve = calibrate_temperature(saved["validation_logits"], val_labels, temperature_bounds)
        selected_teacher.update(calibration)
        save_json(selected_teacher, root / "selected_teacher.json")
        write_table(pd.DataFrame(curve), root / "temperature_grid.csv")
        candidates = [dict(candidate, T=calibration["T"]) for candidate in candidates if candidate["gamma"] == gamma]
        logits = {gamma: logits[gamma]}
    if teacher_selection == "validation_ce":
        selected_teacher = _select_teacher_ce(teacher_rows)
        save_json(selected_teacher, root / "selected_teacher.json")
        candidates = [
            candidate for candidate in candidates if candidate["gamma"] == selected_teacher["gamma"]
        ]
        logits = {selected_teacher["gamma"]: logits[selected_teacher["gamma"]]}
    if candidate_subset is not None:
        candidates = [candidate for candidate in candidates if candidate in candidate_subset]
        if not candidates:
            raise ValueError("No requested candidates remain after teacher preselection")

    rows, diagnostics = [], []
    progress = tqdm(enumerate(candidates), total=len(candidates), desc=f"{dataset} {ratio:g}: {method}")
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
                    candidate.get("B"),
                    seed=seed,
                    max_sweeps=max_sweeps,
                    block_size=block_size,
                    **({"rank": candidate["rank"]} if "rank" in candidate else {}),
                    **({"lr": candidate["assignment_lr"]} if "assignment_lr" in candidate else {}),
                    **({"moment_weight": candidate["lambda"]} if "lambda" in candidate else {}),
                    **(
                        {"initialization_cache": root / "initializations"}
                        if method == "variance_moment_low_rank" and assignment_initialization == "distance"
                        else {}
                    ),
                )
                save_state(partition, artifact)
                gc.collect()
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
                    label_variance=partition.get("label_variance", float("nan")),
                    moment_objective=partition.get("moment_objective", float("nan")),
                    variance_objective=partition.get("variance_objective", float("nan")),
                    beta=partition.get("beta"),
                    alpha=partition.get("alpha", float("nan")),
                    global_feature_variance=partition.get("global_feature_variance", float("nan")),
                    global_label_variance=partition.get("global_label_variance", float("nan")),
                    label_kl=partition.get("label_kl", float("nan")),
                    best_step=partition.get("best_step", float("nan")),
                    sweeps=partition["sweeps"],
                    converged=partition["converged"],
                    status=partition.get("status", ""),
                    initialization_steps=partition.get("initialization_steps", float("nan")),
                    initialization_converged=partition.get("initialization_converged", None),
                    partition_seconds=partition["seconds"],
                    partition_path=str(artifact),
                )
            )
            x, y = partition["x"].to(device), partition["y"].to(device)
            uniform = torch.ones(len(x), device=device)
            for student_seed in search_seeds:
                cached = (folder / "search" / f"seed_{student_seed}.json").exists()
                if x.is_cuda:
                    torch.cuda.synchronize(x.device)
                started = perf_counter()
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
                if x.is_cuda:
                    torch.cuda.synchronize(x.device)
                elapsed = perf_counter() - started
                rows.append(
                    dict(
                        **score,
                        candidate=index,
                        condensation_seed=seed,
                        student_seed=student_seed,
                        evaluation_seconds=elapsed if not cached else float("nan"),
                        evaluation_cached=cached,
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
    if not evaluate_test:
        return pd.DataFrame(), pd.DataFrame(), search, root
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
                dataset=dataset,
                method=method,
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


run_cora_risk_sweep = run_risk_sweep
