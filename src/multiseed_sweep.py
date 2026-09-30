from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from src.data import BUDGET, _prepare_dataset
from src.evaluation import fit_gcn_diagnostic
from src.initialization import cell_means, feature_kmeans
from src.io import _fingerprint, array_digest, save_json, save_state, write_table
from src.moments import initial_logits
from src.prototype_assignment import prototype_logits
from src.soft_ce_partition import optimize_ce_assignment
from src.sweep_utils import boundary_profile, grid_rows, representative
from src.teacher import teacher_logits
from src.transforms import FeatureTransform, fit_transform


@torch.no_grad()
def calibrate_teacher_temperature(logits, labels, bounds=(0.01, 100.0)):
    from scipy.optimize import minimize_scalar

    lower, upper = bounds
    if not 0 < lower <= 1 <= upper or lower == upper:
        raise ValueError("Calibration bounds must bracket T=1")
    logits = logits.detach().double()

    def objective(log_t):
        return float(torch.nn.functional.cross_entropy(logits / np.exp(log_t), labels))

    result = minimize_scalar(objective, bounds=np.log(bounds), method="bounded", options={"xatol": 1e-7})
    if not result.success:
        raise RuntimeError("Teacher temperature calibration failed")
    candidates = [0.0, float(result.x), float(np.log(lower)), float(np.log(upper))]
    log_t = min(candidates, key=objective)
    temperature = float(np.exp(log_t))
    return dict(
        T_cal=temperature,
        validation_ce_before=objective(0.0),
        validation_ce_after=objective(log_t),
        validation_accuracy=100 * float((logits.argmax(1) == labels).double().mean()),
        lower_bound=lower,
        upper_bound=upper,
        boundary="lower"
        if np.isclose(temperature, lower)
        else "upper"
        if np.isclose(temperature, upper)
        else "interior",
        applied_to_sweep=False,
    )


def aggregate_search(records, condensation_seeds, student_seeds, parameter_keys=("T", "rank", "penalty")):
    frame = pd.DataFrame(records)
    keys = ["candidate", *parameter_keys, "step"]
    expected = {(a, b) for a in condensation_seeds for b in student_seeds}
    rows = []
    for key, group in frame.groupby(keys, sort=False):
        observed = set(zip(group.condensation_seed, group.seed))
        if observed != expected or len(group) != len(expected):
            raise ValueError("Each checkpoint must contain exactly the configured seed pairs")
        means = group.groupby("condensation_seed").val_acc.mean()
        rows.append(
            dict(
                zip(keys, key),
                val=means.mean(),
                condensation_val_std=means.std(ddof=1),
                evaluations=len(group),
            )
        )
    return pd.DataFrame(rows)


def run_cora_multiseed(
    ratio,
    output_dir,
    space,
    gammas,
    steps=1000,
    checkpoint_steps=(0, 100, 300, 500, 750, 1000),
    condensation_seeds=(0, 1, 2),
    search_seeds=(0, 1, 2),
    final_seeds=tuple(range(100, 110)),
    teacher_seed=0,
    assignment_lr=0.01,
    basis=3000,
    dropout=0.9,
    epochs=1000,
    eval_every=10,
    hidden=256,
    student_lr=0.01,
    weight_decay=0.0005,
    data_dir="/content/data/",
    device="cuda",
    method="low_rank",
    label_source="teacher",
):
    required = {
        "low_rank": {"T", "rank", "penalty"},
        "mlp": {"T", "rank", "width", "penalty"},
        "prototype": {"T", "tau", "penalty"},
        "sparse": {"T", "k", "penalty"},
    }
    if label_source not in ("teacher", "train"):
        raise ValueError("Label source must be teacher or train")
    if label_source == "train":
        required = {key: value - {"T"} for key, value in required.items()}
        if gammas:
            raise ValueError("Train-only condensation requires an empty gamma list")
    if ("cora", ratio) not in BUDGET or method not in required or set(space) != required[method]:
        raise ValueError("Use a configured Cora ratio and the selected method's grid keys")
    cells = BUDGET[("cora", ratio)]
    candidates = grid_rows(space)
    if any(not np.isfinite(c[k]) or c[k] <= 0 for c in candidates for k in ({"T", "penalty"} & set(space))):
        raise ValueError("T and penalty must be positive and finite")
    if method not in ("prototype", "sparse") and any(
        not isinstance(c["rank"], int) or not 1 <= c["rank"] <= cells for c in candidates
    ):
        raise ValueError("Rank must be an integer no greater than the cell budget")
    if method == "sparse" and any(not isinstance(c["k"], int) or not 2 <= c["k"] <= cells for c in candidates):
        raise ValueError("Sparse candidate count must satisfy 2 <= k <= cells")
    if method == "prototype" and any(not np.isfinite(c["tau"]) or c["tau"] <= 0 for c in candidates):
        raise ValueError("Assignment temperature must be positive and finite")
    if method == "mlp" and any(not isinstance(c["width"], int) or c["width"] < 1 for c in candidates):
        raise ValueError("MLP width must be a positive integer")
    if (
        (label_source == "teacher" and not gammas)
        or any(not np.isfinite(g) or g <= 0 for g in gammas)
        or not search_seeds
        or not final_seeds
        or len(condensation_seeds) < 2
        or set(search_seeds) & set(final_seeds)
        or any(len(set(s)) != len(s) for s in (condensation_seeds, search_seeds, final_seeds))
    ):
        raise ValueError("Invalid gamma grid or seed configuration")
    checkpoints = sorted({0, steps, *checkpoint_steps})
    if steps < 1 or any(not isinstance(s, int) or not 0 <= s <= steps for s in checkpoints):
        raise ValueError("Invalid checkpoint budget")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    graph, train, validation, testing, h = _prepare_dataset("cora", data_dir, device)
    if label_source == "train" and cells > int(train.sum()):
        raise ValueError("Cell budget exceeds the number of training nodes")
    adj = graph["adj"]
    digest = array_digest(
        *(
            t.cpu().numpy()
            for t in (
                graph["x"],
                graph["y"],
                adj.crow_indices(),
                adj.col_indices(),
                adj.values(),
                train,
                validation[1],
                testing[1],
            )
        )
    )
    solver = dict(
        solver_mode="exact",
        inner_method="newton_first",
        implicit_warm_start=True,
        inner_loss_weighting="mass",
        cg_check_interval=1,
        cache_assignment=False,
        inner_max_iter=2000,
        inner_tol=1e-7,
        cg_max_iter=512,
        cg_rtol=1e-6,
    )
    settings = dict(
        epochs=epochs,
        eval_every=eval_every,
        hidden=hidden,
        dropout=dropout,
        lr=student_lr,
        weight_decay=weight_decay,
    )
    config = dict(
        dataset="cora",
        ratio=ratio,
        cells=cells,
        space=space,
        gammas=list(gammas),
        steps=steps,
        checkpoints=checkpoints,
        condensation_seeds=list(condensation_seeds),
        search_seeds=list(search_seeds),
        final_seeds=list(final_seeds),
        teacher_seed=teacher_seed,
        assignment_lr=assignment_lr,
        basis=basis,
        teacher_kernel="relu",
        mixing=0.05,
        student=settings,
        student_loss="uniform",
        solver=solver,
        data_digest=digest,
        version=1,
    )
    if method != "low_rank":
        config["method"] = method
    if label_source == "train":
        config["label_source"] = "train"
        config["feature_normalization"] = "full_graph_rms"
    root = Path(output_dir) / f"ratio_{ratio}" / _fingerprint(config)
    root.mkdir(parents=True, exist_ok=True)
    save_json(config, root / "config.json")
    gamma = None
    if label_source == "teacher":
        logits, gamma = teacher_logits(
            h, graph, train, validation, "relu", list(gammas), basis, teacher_seed, root
        )
        save_json(
            dict(gamma=gamma, teacher_seed=teacher_seed, logits_digest=array_digest(logits.cpu().numpy())),
            root / "selected_teacher.json",
        )
        calibration = calibrate_teacher_temperature(logits[validation[1]], graph["y"][validation[1]])
        save_json(dict(gamma=gamma, **calibration), root / "teacher_calibration.json")
    else:
        train_q = torch.nn.functional.one_hot(graph["y"][train]).double()

    def targets(candidate):
        return train_q if label_source == "train" else (logits / candidate["T"]).softmax(1).double()
    inputs = root / "inputs.pt"
    if inputs.exists():
        saved = torch.load(inputs, map_location=device, weights_only=False)
        z, assignments = saved["z"], saved["assignments"]
        transform = FeatureTransform(**saved["transform"])
        del saved
    else:
        z, transform = fit_transform(h.double(), kind="rms")
        material_h = h
        if label_source == "train":
            z, material_h = z[train], h[train]
        assignments = {s: feature_kmeans(material_h.cpu(), cells, s).to(device) for s in condensation_seeds}
        save_state(dict(z=z, assignments=assignments, transform=vars(transform)), inputs)
    masks = dict(train=train, val=validation[1], test=testing[1])

    def evaluate(index, seed, step, final=False):
        candidate = candidates[index]
        folder = root / f"candidate_{index:04d}" / f"condensation_{seed}"
        snapshot = torch.load(
            folder / "checkpoints" / f"step_{step:06d}.pt", map_location="cpu", weights_only=False
        )
        x, y, mass = representative(snapshot["moments"], transform, z.shape[1], device)
        weights = torch.full_like(mass, 1 / len(mass))
        cache = folder / ("final" if final else "search") / f"step_{step:06d}"
        q = None if label_source == "train" else targets(candidate)
        return [
            dict(
                candidate=index,
                **candidate,
                condensation_seed=seed,
                step=step,
                **fit_gcn_diagnostic(
                    x,
                    y,
                    weights,
                    graph,
                    q,
                    masks if final else {k: masks[k] for k in ("train", "val")},
                    student_seed,
                    folder=cache,
                    **settings,
                ),
            )
            for student_seed in (final_seeds if final else search_seeds)
        ]

    search_records = []
    for index, candidate in enumerate(tqdm(candidates, desc=f"Cora {ratio}: joint seed grid")):
        q = targets(candidate)
        for seed in condensation_seeds:
            folder = root / f"candidate_{index:04d}" / f"condensation_{seed}"
            folder.mkdir(parents=True, exist_ok=True)
            if method in ("mlp", "prototype") and not (folder / "initial_assignment.json").exists():
                assignment = assignments[seed]
                centers = cell_means(z, assignment, cells)
                with torch.no_grad():
                    initial = (
                        prototype_logits(z, centers, candidate["tau"])
                        if method == "prototype"
                        else initial_logits(assignment, cells).double()
                    )
                    probabilities = initial.softmax(1)
                    mass = probabilities.mean(0)
                    realized = probabilities.T @ z / probabilities.sum(0)[:, None]
                    save_json(
                        dict(
                            agreement_with_kmeans=float(
                                (probabilities.argmax(1) == assignment).double().mean()
                            ),
                            mean_entropy=float(
                                -(probabilities * probabilities.clamp_min(1e-300).log()).sum(1).mean()
                            ),
                            mass_tv_from_hard=float(
                                (mass - torch.bincount(assignment, minlength=cells) / len(z)).abs().sum() / 2
                            ),
                            feature_rms_from_hard=float((realized - centers).square().sum(1).mean().sqrt()),
                        ),
                        folder / "initial_assignment.json",
                    )
            options = dict(assignment_rank=candidate.get("rank"))
            if method == "sparse":
                options["sparse_k"] = candidate["k"]
            elif method == "mlp":
                options.update(
                    assignment_input="features", assignment_encoder="mlp", encoder_hidden=candidate["width"]
                )
            elif method == "prototype":
                options.update(
                    assignment_input="features",
                    assignment_encoder="prototype",
                    prototype_temperature=candidate["tau"],
                )
            if not (folder / "complete.json").exists():
                resume = folder / "resume.pt"
                state = (
                    torch.load(resume, map_location="cpu", weights_only=False) if resume.exists() else None
                )
                result = optimize_ce_assignment(
                    z,
                    q,
                    assignments[seed],
                    penalty=candidate["penalty"],
                    lr=assignment_lr,
                    steps=steps,
                    factor_seed=seed,
                    mixing=0.05,
                    folder=folder,
                    checkpoint_steps=checkpoints,
                    resume_state=state,
                    save_resume=True,
                    save_assignment=False,
                    **options,
                    **solver,
                )
                save_json(dict(steps=steps), folder / "complete.json")
                del result, state
            for step in checkpoints:
                search_records.extend(evaluate(index, seed, step))
        write_table(pd.DataFrame(search_records), root / "search_students.csv")
        search = aggregate_search(search_records, condensation_seeds, search_seeds, tuple(space))
        write_table(search, root / "search.csv")
    chosen = search.sort_values(["val", "step", "candidate"], ascending=[False, True, True]).iloc[0].to_dict()
    save_json(chosen, root / "selected.json")
    index, selected_step = int(chosen["candidate"]), int(chosen["step"])
    final = []
    for phase, step in [("initial", 0), ("selected", selected_step)]:
        for seed in condensation_seeds:
            final.extend(dict(phase=phase, **r) for r in evaluate(index, seed, step, final=True))
        write_table(pd.DataFrame(final), root / "final_students.csv")
    final = pd.DataFrame(final)
    by_seed = (
        final.groupby(["phase", "condensation_seed"], sort=False)
        .agg(
            val_mean=("val_acc", "mean"),
            test_mean=("test_acc", "mean"),
            test_student_std=("test_acc", lambda v: v.std(ddof=0)),
        )
        .reset_index()
    )
    summary = (
        by_seed.groupby("phase", sort=False)
        .agg(
            final_val=("val_mean", "mean"),
            test_mean=("test_mean", "mean"),
            test_condensation_std=("test_mean", "std"),
            mean_student_std=("test_student_std", "mean"),
        )
        .reset_index()
    )
    summary["dataset"], summary["ratio"], summary["nodes"] = "cora", ratio, cells
    summary["gamma"] = gamma
    if label_source == "train":
        summary["label_source"] = "train"
        summary["source_nodes"] = int(train.sum())
    if method != "low_rank":
        summary["method"] = method
    for key in space:
        summary[key] = candidates[index][key]
    summary["step"] = summary.phase.map({"initial": 0, "selected": selected_step})
    summary["search_val"] = summary["step"].map(search[search.candidate == index].set_index("step").val)
    write_table(by_seed, root / "by_condensation_seed.csv")
    write_table(summary, root / "summary.csv")
    write_table(boundary_profile(search, space), root / "boundaries.csv")
    return summary, by_seed, search, root
