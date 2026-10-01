import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from src.data import BUDGET, _prepare_dataset
from src.evaluation import fit_gcn_diagnostic
from src.initialization import cell_means, feature_kmeans
from src.io import _fingerprint, array_digest, save_json, save_state, write_table
from src.low_rank_assignment import LowRankMoments, initialize_factors
from src.moments import make_material
from src.multiseed_sweep import aggregate_search
from src.soft_ce_partition import optimize_ce_assignment
from src.sweep_utils import grid_rows, representative
from src.teacher import teacher_logits
from src.transforms import FeatureTransform, fit_transform


def distance_base(z, assignment, tau, normalized=False):
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError("Positive finite assignment temperature required")
    centers = cell_means(z, assignment, int(assignment.max()) + 1)
    distance = (z.square().sum(1, keepdim=True) + centers.square().sum(1) - 2 * z @ centers.T).clamp_min(0)
    if normalized:
        distance = distance - distance.mean(1, keepdim=True)
        return -distance / distance.square().mean().sqrt().clamp_min(1e-12) / tau
    return -(distance - distance.min(1, keepdim=True).values) / tau


def run_distance_cost_sweep(output_dir, gammas, temperatures, assignment_temperatures, penalties, **options):
    return run_soft_init_sweep(
        output_dir, gammas, temperatures, [1.0], penalties,
        finetune_temperatures=assignment_temperatures, **options,
    )


def random_cost_base(nodes, cells, seed, temperature, device, dtype):
    generator = torch.Generator().manual_seed(seed)
    cost = torch.randn(nodes, cells, generator=generator, dtype=dtype).to(device)
    cost = cost - cost.mean(1, keepdim=True)
    return -cost / cost.square().mean().sqrt().clamp_min(1e-12) / temperature


def run_soft_init_sweep(
    output_dir, gammas, temperatures, taus, penalties, ratio=0.013, rank=8,
    steps=1000, checkpoint_steps=(0, 25, 100, 300, 500, 750, 1000),
    condensation_seeds=(0, 1, 2), search_seeds=(0, 1, 2), final_seeds=tuple(range(100, 110)),
    assignment_lr=0.01, basis=3000, teacher_seed=0, epochs=1000, eval_every=10,
    hidden=256, dropout=0.9, student_lr=0.01, weight_decay=0.0005,
    data_dir="/content/data/", device="cuda",
    finetune_temperatures=None,
    initialization="kmeans",
    resume_from=None,
):
    if initialization not in ("kmeans", "random"):
        raise ValueError("Unknown initialization")
    if initialization == "random" and finetune_temperatures is None:
        raise ValueError("Random costs require the normalized cost sweep")
    values = [*gammas, *temperatures, *taus, *penalties, *(finetune_temperatures or ())]
    if any(not v for v in (gammas, temperatures, taus, penalties)) or any(
        not np.isfinite(v) or v <= 0 for v in values
    ):
        raise ValueError("All grids must be nonempty with positive finite values")
    if finetune_temperatures is not None and not finetune_temperatures:
        raise ValueError("Fine-tuning temperature grid must be nonempty")
    cells = BUDGET[("cora", ratio)]
    if not isinstance(rank, int) or not 1 <= rank <= cells or steps < 1:
        raise ValueError("Invalid rank or optimization budget")
    if (len(condensation_seeds) < 2 or not search_seeds or not final_seeds
            or set(search_seeds) & set(final_seeds)
            or any(len(set(s)) != len(s) for s in (condensation_seeds, search_seeds, final_seeds))):
        raise ValueError("Use distinct condensation seeds and disjoint search/final student seeds")
    checkpoints = sorted({0, steps, *checkpoint_steps})
    if any(not isinstance(s, int) or not 0 <= s <= steps for s in checkpoints):
        raise ValueError("Checkpoint outside optimization budget")
    config = {k: v for k, v in locals().copy().items() if k not in ("output_dir", "device", "values", "resume_from")}
    if finetune_temperatures is None:
        config.pop("finetune_temperatures")
    if initialization == "kmeans":
        config.pop("initialization")
    normalized = finetune_temperatures is not None
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    graph, train, validation, testing, h = _prepare_dataset("cora", data_dir, device)
    config.update(version=1, data_digest=array_digest(
        h.cpu().numpy(), graph["y"].cpu().numpy(), train.cpu().numpy(),
        validation[1].cpu().numpy(), testing[1].cpu().numpy(),
    ))
    root = Path(output_dir) / _fingerprint(config)
    root.mkdir(parents=True, exist_ok=True)
    if resume_from is not None:
        previous = Path(resume_from)
        old = json.loads((previous / "config.json").read_text(encoding="utf-8"))
        ignored = {"steps", "checkpoint_steps", "checkpoints"}
        changed = {
            key: (old.get(key), config.get(key))
            for key in old.keys() | config.keys()
            if key not in ignored and old.get(key) != config.get(key)
        }
        if changed or steps <= old["steps"]:
            raise ValueError(f"Continuation settings differ: {changed}; steps {old['steps']} -> {steps}")
        if not set(old["checkpoints"]).issubset(checkpoints):
            raise ValueError("Keep the original checkpoints when extending a run")
        if any(s < old["steps"] and s not in old["checkpoints"] for s in checkpoints):
            raise ValueError("Cannot add unsaved checkpoints before the previous endpoint")
        candidates = previous / "finetune"
        folders = list(candidates.glob("candidate_*/condensation_*"))
        expected = len(penalties) * len(finetune_temperatures or taus) * len(condensation_seeds)
        if len(folders) != expected or any(not (p / "resume.pt").exists() for p in folders):
            raise ValueError("Continuation requires every candidate's saved optimizer state")
        if not (root / "continued_from.json").exists():
            shutil.copytree(previous, root, dirs_exist_ok=True)
            save_json(dict(path=str(previous), steps=old["steps"]), root / "continued_from.json")
    save_json(config, root / "config.json")
    teachers = teacher_logits(
        h, graph, train, validation, "relu", list(gammas), basis, teacher_seed, root, return_all=True
    )
    if (root / "inputs.pt").exists():
        saved = torch.load(root / "inputs.pt", map_location=device, weights_only=False)
        z, assignments = saved["z"], saved["assignments"]
        transform = FeatureTransform(**saved["transform"])
    else:
        z, transform = fit_transform(h.double(), kind="rms")
        assignments = {
            s: feature_kmeans(h.cpu(), cells, s).to(device) if initialization == "kmeans"
            else torch.arange(len(z), device=device) % cells
            for s in condensation_seeds
        }
        save_state(dict(z=z, assignments=assignments, transform=vars(transform)), root / "inputs.pt")
    settings = dict(epochs=epochs, eval_every=eval_every, hidden=hidden, dropout=dropout,
                    lr=student_lr, weight_decay=weight_decay)
    masks = dict(train=train, val=validation[1], test=testing[1])

    def evaluate(moments, q, folder, final=False):
        x, y, mass = representative(moments, transform, z.shape[1], device)
        return [fit_gcn_diagnostic(
            x, y, torch.full_like(mass, 1 / cells), graph, q,
            masks if final else {k: masks[k] for k in ("train", "val")},
            seed, folder=folder, **settings,
        ) for seed in (final_seeds if final else search_seeds)]

    def initial_state(seed, tau, q):
        base = (distance_base(z, assignments[seed], tau, normalized=normalized)
                if initialization == "kmeans"
                else random_cost_base(len(z), cells, seed, tau, device, z.dtype))
        u, v = initialize_factors(assignments[seed], cells, rank, seed)
        with torch.no_grad():
            moments = LowRankMoments.apply(u, v, base, make_material(z, q), 0.05, 8192)
        return base, moments

    grid = grid_rows(dict(gamma=list(gammas), T=list(temperatures), tau=list(taus)))
    records = []
    for index, params in enumerate(tqdm(grid, desc="Soft initialization validation")):
        q = (teachers[params["gamma"]].to(device) / params["T"]).softmax(1).double()
        for seed in condensation_seeds:
            base, moments = initial_state(seed, params["tau"], q)
            folder = root / "initial_grid" / f"candidate_{index:04d}" / f"condensation_{seed}"
            folder.mkdir(parents=True, exist_ok=True)
            probability = base.softmax(1)
            save_json(dict(
                entropy=float(-(probability * probability.clamp_min(1e-300).log()).sum(1).mean()),
                agreement=(float((probability.argmax(1) == assignments[seed]).double().mean())
                           if initialization == "kmeans" else None),
                min_mass=float(moments[:, 0].min()),
            ), folder / "assignment.json")
            records.extend(dict(candidate=index, **params, step=0, condensation_seed=seed, **row)
                           for row in evaluate(moments, q, folder / "search"))
        write_table(pd.DataFrame(records), root / "initial_students.csv")
        initial_grid = aggregate_search(records, condensation_seeds, search_seeds, ("gamma", "T", "tau"))
        write_table(initial_grid, root / "initial_grid.csv")
    chosen = initial_grid.sort_values(["val", "candidate"], ascending=[False, True]).iloc[0].to_dict()
    save_json(chosen, root / "selected_initialization.json")
    q = (teachers[chosen["gamma"]].to(device) / chosen["T"]).softmax(1).double()
    fine_grid = grid_rows(dict(t=list(finetune_temperatures or [chosen["tau"]]), penalty=list(penalties)))
    records = []
    for index, params in enumerate(tqdm(fine_grid, desc="Low-rank fine-tuning")):
        t, penalty = params["t"], params["penalty"]
        for seed in condensation_seeds:
            base, initial = initial_state(seed, t, q)
            folder = root / "finetune" / f"candidate_{index:04d}" / f"condensation_{seed}"
            folder.mkdir(parents=True, exist_ok=True)
            completed = folder / "complete.json"
            if not completed.exists() or json.loads(completed.read_text())["steps"] < steps:
                resume = folder / "resume.pt"
                state = torch.load(resume, map_location="cpu", weights_only=False) if resume.exists() else None
                result = optimize_ce_assignment(
                    z, q, assignments[seed], penalty=penalty, steps=steps, lr=assignment_lr,
                    assignment_rank=rank, factor_seed=seed, base_logits=base,
                    correction_scale=-1 / t if normalized else 1.0,
                    checkpoint_steps=checkpoints, folder=folder, resume_state=state,
                    save_resume=True, save_assignment=False, inner_method="newton_first",
                    implicit_warm_start=True, inner_loss_weighting="mass", cg_max_iter=512,
                )
                del result, state
                save_json(dict(steps=steps), folder / "complete.json")
            for step in checkpoints:
                snapshot = torch.load(folder / "checkpoints" / f"step_{step:06d}.pt",
                                      map_location=device, weights_only=False)
                if step == 0 and not torch.allclose(snapshot["moments"], initial, atol=1e-12, rtol=1e-10):
                    raise RuntimeError("Fine-tuning initialization differs from selection")
                if step == 0:
                    cache = ((root / "initial_grid" / f"candidate_{int(chosen['candidate']):04d}")
                             if t == chosen["tau"] else root / "fine_initial" / f"t_{t:.12g}")
                    cache = cache / f"condensation_{seed}" / "search"
                    if normalized:
                        cache.parent.mkdir(parents=True, exist_ok=True)
                        probability = base.softmax(1)
                        save_json(dict(
                            t=t,
                            entropy=float(-(probability * probability.clamp_min(1e-300).log()).sum(1).mean()),
                            agreement=(float((probability.argmax(1) == assignments[seed]).double().mean())
                                       if initialization == "kmeans" else None),
                            min_mass=float(initial[:, 0].min()),
                        ), cache.parent / "assignment.json")
                else:
                    cache = folder / "search" / f"step_{step:06d}"
                records.extend(dict(candidate=index, t=t, penalty=penalty, step=step, condensation_seed=seed, **row)
                               for row in evaluate(snapshot["moments"], q, cache))
        write_table(pd.DataFrame(records), root / "finetune_students.csv")
        search = aggregate_search(records, condensation_seeds, search_seeds, ("t", "penalty"))
        write_table(search, root / "finetune_grid.csv")
    selected = search.sort_values(["val", "step", "candidate"], ascending=[False, True, True]).iloc[0].to_dict()
    save_json(selected, root / "selected_finetune.json")
    rows = []
    phases = [("initial", 0), ("selected", int(selected["step"]))]
    if normalized:
        phases.insert(0, ("selection_initial", 0))
    for phase, step in phases:
        for seed in condensation_seeds:
            folder = root / "finetune" / f"candidate_{int(selected['candidate']):04d}" / f"condensation_{seed}"
            if phase == "selection_initial":
                _, moments = initial_state(seed, chosen["tau"], q)
            else:
                moments = torch.load(folder / "checkpoints" / f"step_{step:06d}.pt",
                                     map_location=device, weights_only=False)["moments"]
            cache = root / "final" / f"condensation_{seed}" / f"step_{step:06d}"
            if step > 0:
                cache = cache / f"candidate_{int(selected['candidate']):04d}"
            if normalized:
                temperature = chosen["tau"] if phase == "selection_initial" else selected["t"]
                cache = cache / f"t_{temperature:.12g}"
            rows.extend(dict(phase=phase, condensation_seed=seed, **r)
                        for r in evaluate(moments, q, cache, final=True))
    final = pd.DataFrame(rows)
    by_seed = final.groupby(["phase", "condensation_seed"], sort=False).agg(
        val_mean=("val_acc", "mean"), test_mean=("test_acc", "mean"),
        test_student_std=("test_acc", lambda x: x.std(ddof=0)),
    ).reset_index()
    summary = by_seed.groupby("phase", sort=False).agg(
        final_val=("val_mean", "mean"), test_mean=("test_mean", "mean"),
        test_condensation_std=("test_mean", "std"), mean_student_std=("test_student_std", "mean"),
    ).reset_index()
    for key in ("gamma", "T", "tau"):
        summary[key] = chosen[key]
    summary["rank"], summary["nodes"], summary["penalty"] = rank, cells, selected["penalty"]
    summary["step"] = summary.phase.map(dict(initial=0, selected=int(selected["step"])))
    summary["search_val"] = summary.phase.map(dict(initial=chosen["val"], selected=selected["val"]))
    if normalized:
        initial_score = search[(search.candidate == selected["candidate"]) & (search.step == 0)].iloc[0]["val"]
        summary = summary.drop(columns="tau")
        summary["t"] = summary.phase.map(dict(selection_initial=chosen["tau"], initial=selected["t"], selected=selected["t"]))
        summary["step"] = summary.phase.map(dict(selection_initial=0, initial=0, selected=int(selected["step"])))
        summary["search_val"] = summary.phase.map(dict(selection_initial=chosen["val"], initial=initial_score, selected=selected["val"]))
    summary["initialization"] = initialization
    for name, table in (("summary", summary), ("by_seed", by_seed), ("final_students", final)):
        write_table(table, root / f"{name}.csv")
    return summary, by_seed, initial_grid, search, root
