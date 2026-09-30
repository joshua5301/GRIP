from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data import BUDGET, _prepare_dataset
from src.evaluation import fit_gcn_diagnostic
from src.initialization import feature_kmeans
from src.io import _fingerprint, array_digest, save_json, save_state, write_table
from src.soft_ce_partition import optimize_ce_assignment
from src.sweep_utils import grid_rows, representative
from src.teacher import teacher_logits
from src.transforms import FeatureTransform, fit_transform


def promote_widths(screen, baseline_index, keep):
    ranked = screen[(screen.method == "mlp") & (screen.step > 0)].sort_values(
        ["val", "step", "candidate"], ascending=[False, True, True]
    )
    return [baseline_index, *ranked.drop_duplicates("width").head(keep).candidate.astype(int).tolist()]


def run_arxiv_width_sweep(
    output_dir,
    baseline=None,
    baseline_space=None,
    gammas=(1e-5, 1e-4, 1e-3, 0.01),
    widths=(128, 512, 1024),
    mlp_lrs=(0.003, 0.01),
    screen_steps=200,
    steps=1000,
    keep=2,
    condensation_seeds=(0, 1, 2),
    screen_seeds=(0, 1),
    search_seeds=(0, 1, 2),
    final_seeds=tuple(range(100, 110)),
    student_loss="uniform",
    dropout=0.31881090213944857,
    basis=3000,
    teacher_seed=0,
    data_dir="/content/data/",
    device="cuda",
):
    if baseline_space is None:
        baseline_space = dict(T=[0.3, 1.0, 3.0], penalty=[1e-4, 1e-3], rank=[16], lr=[0.01])
    keys = {"T", "penalty", "rank", "lr"}
    baselines = [dict(baseline)] if baseline is not None else grid_rows(baseline_space)
    cells = BUDGET[("arxiv", 0.0005)]
    if any(set(c) != keys for c in baselines):
        raise ValueError("Baseline requires exactly T, penalty, rank and lr")
    if any(
        not isinstance(c["rank"], int) or not 1 <= c["rank"] <= cells
        or any(not np.isfinite(c[k]) or c[k] <= 0 for k in ("T", "penalty", "lr"))
        for c in baselines
    ):
        raise ValueError("Invalid baseline grid")
    seed_sets = (condensation_seeds, screen_seeds, search_seeds, final_seeds)
    if (
        not isinstance(screen_steps, int) or not isinstance(steps, int) or not 0 < screen_steps < steps
        or len(condensation_seeds) < 2 or condensation_seeds[0] != 0
        or any(not s or len(set(s)) != len(s) for s in seed_sets)
        or set(final_seeds) & (set(screen_seeds) | set(search_seeds))
        or not widths or len(set(widths)) != len(widths)
        or any(not isinstance(w, int) or w < 1 for w in widths)
        or not mlp_lrs or len(set(mlp_lrs)) != len(mlp_lrs)
        or any(not np.isfinite(v) or v <= 0 for v in mlp_lrs)
        or not isinstance(keep, int) or not 1 <= keep <= len(widths)
        or student_loss not in ("mass", "uniform")
        or not gammas or any(not np.isfinite(v) or v <= 0 for v in gammas)
        or not str(device).startswith("cuda")
    ):
        raise ValueError("Invalid budget, seeds, widths, learning rates or GPU device")
    if "A100" not in torch.cuda.get_device_name(torch.device(device)):
        raise RuntimeError("Run this experiment on a Colab A100")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    graph, train, validation, testing, h = _prepare_dataset("arxiv", data_dir, device)
    adjacency = graph["adj"]
    digest = array_digest(*(
        t.cpu().numpy() for t in (
            graph["x"], graph["y"], adjacency.crow_indices(), adjacency.col_indices(),
            adjacency.values(), train, validation[1], testing[1],
        )
    ))
    solver = dict(
        solver_mode="exact", inner_method="newton_first", implicit_warm_start=True,
        inner_loss_weighting="mass", inner_max_iter=2000, inner_tol=1e-7,
        cg_max_iter=512, cg_rtol=1e-6, cache_assignment=False, cg_check_interval=1,
    )
    student = dict(hidden=256, dropout=dropout, lr=0.01, weight_decay=0.0005)
    screen_student = dict(**student, epochs=500, eval_every=25)
    full_student = dict(**student, epochs=1000, eval_every=10)
    checkpoints = sorted({0, screen_steps, steps, *(s for s in (100, 300, 500, 750) if s < steps)})
    config = dict(
        version=1, dataset="arxiv", ratio=0.0005, cells=cells, baselines=baselines,
        gammas=list(gammas), widths=list(widths), mlp_lrs=list(mlp_lrs), keep=keep,
        screen_steps=screen_steps, steps=steps, checkpoints=checkpoints,
        condensation_seeds=list(condensation_seeds), screen_seeds=list(screen_seeds),
        search_seeds=list(search_seeds), final_seeds=list(final_seeds),
        teacher_seed=teacher_seed, basis=basis, kernel="relu", mixing=0.05,
        student_loss=student_loss, screen_student=screen_student, full_student=full_student,
        solver=solver, data_digest=digest,
    )
    root = Path(output_dir) / _fingerprint(config)
    root.mkdir(parents=True, exist_ok=True)
    save_json(config, root / "config.json")
    save_json(dict(torch=str(torch.__version__), cuda=torch.version.cuda,
                   device=torch.cuda.get_device_name(torch.device(device))), root / "environment.json")
    logits, gamma = teacher_logits(h, graph, train, validation, "relu", list(gammas), basis, teacher_seed, root)
    save_json(dict(gamma=gamma, logits_digest=array_digest(logits.cpu().numpy())), root / "selected_teacher.json")
    inputs = root / "inputs.pt"
    if inputs.exists():
        saved = torch.load(inputs, map_location=device, weights_only=False)
        z, assignments = saved["z"], saved["assignments"]
        transform = FeatureTransform(**saved["transform"])
        del saved
    else:
        z, transform = fit_transform(h.double())
        assignments = {s: feature_kmeans(h.cpu(), cells, s).to(device) for s in condensation_seeds}
        save_state(dict(z=z, assignments=assignments, transform=vars(transform)), inputs)
    del h
    masks = dict(train=train, val=validation[1], test=testing[1])
    candidates = [dict(method="low_rank", width=0, **c) for c in baselines]
    initial_checks = []

    def folder_for(index, seed):
        return root / f"candidate_{index:03d}" / f"condensation_{seed}"

    def optimize(index, seed, budget):
        c = candidates[index]
        folder = folder_for(index, seed)
        folder.mkdir(parents=True, exist_ok=True)
        resume = folder / "resume.pt"
        state = torch.load(resume, map_location="cpu", weights_only=False) if resume.exists() else None
        if state is None or state["step"] < budget:
            q = (logits / c["T"]).softmax(1).double()
            result = optimize_ce_assignment(
                z, q, assignments[seed], penalty=c["penalty"], lr=c["lr"], steps=budget,
                assignment_rank=c["rank"], factor_seed=seed,
                assignment_input="features" if c["method"] == "mlp" else "node",
                assignment_encoder="mlp" if c["method"] == "mlp" else "linear",
                encoder_hidden=c["width"] if c["method"] == "mlp" else 64,
                folder=folder, checkpoint_steps=[s for s in checkpoints if s <= budget],
                resume_state=state, save_resume=True, save_assignment=False, **solver,
            )
            del result
        del state
        initial = torch.load(folder / "checkpoints" / "step_000000.pt", weights_only=False)["moments"]
        shared = root / f"initial_{seed}_{_fingerprint(c['T'])}.pt"
        if not shared.exists():
            save_state(initial, shared)
        expected = torch.load(shared, map_location="cpu", weights_only=True)
        difference = float((initial - expected).abs().max())
        if not torch.allclose(initial, expected, atol=1e-10, rtol=1e-8):
            raise ValueError("Assignment families do not share initial moments")
        initial_checks.append(dict(candidate=index, condensation_seed=seed, max_difference=difference))
        write_table(pd.DataFrame(initial_checks), root / "initial_checks.csv")

    def evaluate(index, seed, step, seeds, phase):
        c = candidates[index]
        snapshot = torch.load(folder_for(index, seed) / "checkpoints" / f"step_{step:06d}.pt",
                              map_location="cpu", weights_only=False)
        x, y, mass = representative(snapshot["moments"], transform, z.shape[1], device)
        weights = mass if student_loss == "mass" else torch.full_like(mass, 1 / len(mass))
        q = (logits / c["T"]).softmax(1).double()
        cache = (root / f"initial_eval_{seed}_{_fingerprint(c['T'])}" if step == 0 else folder_for(index, seed))
        cache = cache / phase / f"step_{step:06d}"
        settings = screen_student if phase == "screen" else full_student
        return [dict(candidate=index, **c, condensation_seed=seed, step=step,
                     outer_ce=snapshot["teacher_ce"], **fit_gcn_diagnostic(
                         x, y, weights, graph, q,
                         masks if phase == "final" else {k: masks[k] for k in ("train", "val")},
                         s, folder=cache, **settings)) for s in seeds]

    def aggregate(records, cond_seeds, students):
        frame = pd.DataFrame(records)
        rows = []
        expected = {(a, b) for a in cond_seeds for b in students}
        for (index, step), group in frame.groupby(["candidate", "step"], sort=False):
            if set(zip(group.condensation_seed, group.seed)) != expected or len(group) != len(expected):
                raise ValueError("Incomplete or duplicate seed pairs")
            rows.append(dict(candidate=index, **candidates[index], step=step, val=group.val_acc.mean(),
                             outer_ce=group.outer_ce.mean()))
        return pd.DataFrame(rows)

    screen_records = []
    screen_checkpoints = [s for s in checkpoints if s <= screen_steps]
    for index in range(len(baselines)):
        optimize(index, 0, screen_steps)
        for step in screen_checkpoints:
            screen_records.extend(evaluate(index, 0, step, screen_seeds, "screen"))
        write_table(pd.DataFrame(screen_records), root / "screen_students.csv")
    screen = aggregate(screen_records, (0,), screen_seeds)
    chosen = screen.sort_values(["val", "step", "candidate"], ascending=[False, True, True]).iloc[0]
    baseline_index = int(chosen.candidate)
    common = {k: candidates[baseline_index][k] for k in keys}
    save_json(dict(candidate=baseline_index, **common, screen_step=int(chosen.step),
                   provisional_selection=True), root / "baseline_selected.json")
    for width in widths:
        for lr in mlp_lrs:
            candidates.append(dict(method="mlp", width=width, **{**common, "lr": lr}))
    write_table(pd.DataFrame([dict(candidate=i, **c) for i, c in enumerate(candidates)]), root / "candidates.csv")
    for index in range(len(baselines), len(candidates)):
        optimize(index, 0, screen_steps)
        for step in screen_checkpoints:
            screen_records.extend(evaluate(index, 0, step, screen_seeds, "screen"))
        write_table(pd.DataFrame(screen_records), root / "screen_students.csv")
        write_table(aggregate(screen_records, (0,), screen_seeds), root / "screen.csv")
    screen = aggregate(screen_records, (0,), screen_seeds)
    promoted = promote_widths(screen, baseline_index, keep)
    save_json(dict(candidates=promoted, selection="screen validation; distinct MLP widths"), root / "promoted.json")
    records = []
    for index in promoted:
        for seed in condensation_seeds:
            optimize(index, seed, steps)
            for step in checkpoints:
                records.extend(evaluate(index, seed, step, search_seeds, "search"))
            write_table(pd.DataFrame(records), root / "search_students.csv")
    search = aggregate(records, condensation_seeds, search_seeds)
    write_table(search, root / "search.csv")
    selected = search.sort_values(["val", "step", "candidate"], ascending=[False, True, True]).drop_duplicates("method")
    write_table(selected, root / "selected.csv")
    final_records = []
    for choice in selected.itertuples(index=False):
        for phase, step in (("initial", 0), ("selected", int(choice.step))):
            for seed in condensation_seeds:
                final_records.extend(dict(phase=phase, **r) for r in evaluate(
                    int(choice.candidate), seed, step, final_seeds, "final"))
                write_table(pd.DataFrame(final_records), root / "final_students.csv")
    final = pd.DataFrame(final_records)
    by_seed = final.groupby(["method", "phase", "condensation_seed"], sort=False).agg(
        val_mean=("val_acc", "mean"), test_mean=("test_acc", "mean"),
        student_std=("test_acc", lambda v: v.std(ddof=0)),
    ).reset_index()
    summary = by_seed.groupby(["method", "phase"], sort=False).agg(
        final_val=("val_mean", "mean"), test_mean=("test_mean", "mean"),
        condensation_std=("test_mean", "std"), mean_student_std=("student_std", "mean"),
    ).reset_index()
    summary = summary.merge(selected.drop(columns=["outer_ce", "step"]).rename(columns={"val": "selected_search_val"}),
                            on="method", how="left")
    selected_steps = selected.set_index("method").step.to_dict()
    summary["step"] = [0 if p == "initial" else selected_steps[m] for m, p in zip(summary.method, summary.phase)]
    summary["gamma"], summary["nodes"], summary["student_loss"] = gamma, cells, student_loss
    write_table(by_seed, root / "by_seed.csv")
    write_table(summary, root / "summary.csv")
    return summary, by_seed, search, root
