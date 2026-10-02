import hashlib
import json
import math
from pathlib import Path
from time import perf_counter

import pandas as pd
import torch
from tqdm.auto import tqdm

from src.data import _prepare_dataset
from src.evaluation import fit_gcn_diagnostic
from src.io import _fingerprint, array_digest, save_json, save_state, write_table
from src.variance_moment_low_rank import moment_objective
from src.variance_moment_sweep import _data_digest


def project_features(x, rank):
    dimension = min(rank - 1, x.shape[1], len(x) - 1)
    if dimension < 1:
        raise ValueError("Distance initialization requires rank >= 2")
    devices = [x.device.index] if x.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(0)
        _, _, vectors = torch.pca_lowrank(x, q=min(dimension + 8, *x.shape), center=False, niter=4)
    return x @ vectors[:, :dimension]


def distance_factors(projected, cells, rank, seed, temperature=1.0, lloyd_steps=20):
    x = projected
    generator = torch.Generator(device=x.device).manual_seed(seed)
    first = int(torch.randint(len(x), (1,), device=x.device, generator=generator))
    centers = [x[first]]
    nearest = (x - centers[0]).square().sum(1)
    for _ in range(1, cells):
        index = (
            int(torch.multinomial(nearest, 1, generator=generator))
            if float(nearest.sum()) > 0
            else int(torch.randint(len(x), (1,), device=x.device, generator=generator))
        )
        centers.append(x[index])
        nearest = torch.minimum(nearest, (x - centers[-1]).square().sum(1))
    centers = torch.stack(centers)
    for _ in range(lloyd_steps):
        assignment = torch.cdist(x, centers).argmin(1)
        counts = torch.bincount(assignment, minlength=cells)
        sums = torch.zeros_like(centers).index_add_(0, assignment, x)
        centers = torch.where(counts[:, None] > 0, sums / counts.clamp_min(1)[:, None], centers)
    left = torch.cat((2 * x, x.new_ones(len(x), 1)), 1)
    right = torch.cat((centers, -centers.square().sum(1, keepdim=True)), 1)
    right = right - right.mean(0)
    rms = ((left.T @ left) * (right.T @ right)).sum().div(len(x) * cells).clamp_min(1e-24).sqrt()
    left = left / (temperature * rms)
    a, ra = torch.linalg.qr(left, mode="reduced")
    b, rb = torch.linalg.qr(right, mode="reduced")
    u, s, vh = torch.linalg.svd(ra @ rb.T, full_matrices=False)
    scale = s.sqrt() * rank**0.25
    u, v = (a @ u) * scale, (b @ vh.T) * scale
    padding = rank - u.shape[1]
    return torch.nn.functional.pad(u, (0, padding)), torch.nn.functional.pad(v, (0, padding))


def optimize_initialization(
    H,
    Q,
    cells,
    weight,
    rank,
    seed,
    steps=3000,
    lr=0.01,
    initialization="random",
    projected=None,
    temperature=1.0,
    entropy_fraction=0.0,
    anneal_fraction=0.8,
):
    if initialization not in ("random", "distance") or not 0 < anneal_fraction < 1:
        raise ValueError("Invalid initialization or annealing schedule")
    if (
        min(steps, rank, cells) < 1
        or cells > len(H)
        or min(weight, lr, temperature) <= 0
        or entropy_fraction < 0
    ):
        raise ValueError("Invalid optimization settings")
    if H.is_cuda:
        torch.cuda.synchronize(H.device)
    started = perf_counter()
    x, q = H.detach().double(), Q.detach().double()
    offset = x.mean(0)
    x = x - offset
    scale = x.square().sum(1).mean().sqrt().clamp_min(1e-12)
    x = x / scale
    n = len(x)
    generator = torch.Generator(device=x.device).manual_seed(seed)
    if initialization == "random":
        u = torch.randn(n, rank, generator=generator, device=x.device, dtype=x.dtype)
        v = torch.randn(cells, rank, generator=generator, device=x.device, dtype=x.dtype)
    else:
        projected = project_features(x, rank) if projected is None else projected
        u, v = distance_factors(projected, cells, rank, seed, temperature)
    u, v = torch.nn.Parameter(u), torch.nn.Parameter(v)
    optimizer = torch.optim.Adam([u, v], lr=lr)
    energy, original = x.square().sum(1).mean(), x.T @ q / n
    history, best, best_stats, beta0 = [], math.inf, None, None
    for step in range(steps + 1):
        optimizer.zero_grad(set_to_none=True)
        logp = (u @ v.T / rank**0.5).log_softmax(1)
        p = logp.exp()
        stats = p.mean(0), p.T @ x / n, p.T @ q / n
        if not bool((stats[0] > 0).all()):
            raise FloatingPointError(f"Empty numerical cell at step {step}")
        smooth, exact, variance, moment = moment_objective(
            stats, energy, original, None, moment_weight=weight
        )
        entropy = -(p * logp).sum() / n
        value = float(exact.detach())
        if not math.isfinite(value):
            raise FloatingPointError(f"Nonfinite objective at step {step}")
        if beta0 is None:
            beta0 = entropy_fraction * value / max(math.log(cells), 1e-12)
        beta = beta0 * max(0.0, 1 - step / (steps * anneal_fraction))
        if value < best:
            best, best_step = value, step
            best_stats = [s.detach().clone() for s in stats]
        if x.is_cuda:
            torch.cuda.synchronize(x.device)
        history.append(
            dict(
                step=step,
                J=value,
                best_J=best,
                entropy=float(entropy.detach()),
                beta=beta,
                seconds=perf_counter() - started,
            )
        )
        if step == steps:
            break
        (smooth - beta * entropy).backward()
        if not all(bool(torch.isfinite(t.grad).all()) for t in (u, v)):
            raise FloatingPointError(f"Nonfinite gradient at step {step}")
        optimizer.step()
    mass, sx, sq = best_stats
    _, exact, variance, moment = moment_objective(best_stats, energy, original, None, moment_weight=weight)
    return dict(
        x=(sx / mass[:, None] * scale + offset).float().cpu(),
        y=(sq / mass[:, None]).float().cpu(),
        counts=(mass * n).cpu(),
        J_initial=history[0]["J"],
        J_final=float(exact),
        variance=float(variance),
        moment=float(moment),
        best_step=best_step,
        seconds=history[-1]["seconds"],
        history=history,
        seed=seed,
        initialization=initialization,
    )


def run_initialization_comparison(
    previous_run,
    output_dir,
    steps=3000,
    lr=0.01,
    starts=5,
    condensation_seeds=(0, 1, 2),
    student_seeds=tuple(range(100, 110)),
    temperature=1.0,
    entropy_fraction=0.1,
    anneal_fraction=0.8,
    data_dir="/content/data/",
    device="cuda",
):
    if starts < 2 or len(set(condensation_seeds)) != len(condensation_seeds):
        raise ValueError("Require multiple starts and distinct condensation seeds")
    previous = Path(previous_run)
    old = json.loads((previous / "config.json").read_text())
    selected = json.loads((previous / "selected.json").read_text())
    if old["dataset"] not in ("cora", "citeseer"):
        raise ValueError("This full-matrix comparison is intended for Cora or Citeseer")
    graph, train, validation, test, _ = _prepare_dataset(old["dataset"], data_dir, device)
    splits = dict(train=(graph, train), val=validation, test=test)
    if _data_digest(splits) != old["data_digest"]:
        raise ValueError("Dataset differs from the source experiment")
    h = torch.load(previous / "features.pt", map_location=device, weights_only=True)
    feature_config = json.loads((previous / "features.json").read_text())
    if array_digest(h.cpu().numpy()) != feature_config["h_digest"]:
        raise ValueError("Cached source features changed")
    gamma, T = selected["gamma"], selected["T"]
    teacher = torch.load(
        previous / "teachers" / f"{_fingerprint(dict(gamma=gamma))}.pt",
        map_location=device,
        weights_only=True,
    )
    q = (teacher["logits"] / T).softmax(1).double()
    rank, weight = int(selected["rank"]), selected["lambda"]
    config = dict(
        previous_run=str(previous),
        source_config=old,
        selected=selected,
        steps=steps,
        lr=lr,
        starts=starts,
        condensation_seeds=list(condensation_seeds),
        student_seeds=list(student_seeds),
        temperature=temperature,
        entropy_fraction=entropy_fraction,
        anneal_fraction=anneal_fraction,
        h_digest=feature_config["h_digest"],
        q_digest=array_digest(q.cpu().numpy()),
        torch=str(torch.__version__),
        source_digest=hashlib.sha256(
            b"".join(
                Path(__file__).with_name(name).read_bytes()
                for name in (
                    "moment_initialization.py",
                    "variance_moment_low_rank.py",
                    "evaluation.py",
                    "models.py",
                )
            )
        ).hexdigest(),
    )
    root = Path(output_dir) / f"ratio_{old['ratio']:g}" / _fingerprint(config)
    root.mkdir(parents=True, exist_ok=True)
    save_json(config, root / "config.json")
    projection_path = root / "projection.pt"
    if projection_path.exists():
        projected = torch.load(projection_path, map_location=device, weights_only=True)
    else:
        x = h.double() - h.double().mean(0)
        projected = project_features(x / x.square().sum(1).mean().sqrt().clamp_min(1e-12), rank)
        save_state(projected, projection_path)
    options = dict(
        cells=old["nodes"],
        weight=weight,
        rank=rank,
        steps=steps,
        lr=lr,
        projected=projected,
        temperature=temperature,
        anneal_fraction=anneal_fraction,
    )
    records, evaluations, histories = [], [], []
    settings = old["student"]
    masks = dict(train=train, val=validation[1], test=test[1])
    seed_set = set(condensation_seeds)
    for seed in tqdm(condensation_seeds, desc="Initialization comparison"):
        trials = []
        for restart in range(starts):
            trial_seed = seed if restart == 0 else 1_000_000 + seed * starts + restart
            if restart and trial_seed in seed_set:
                raise ValueError("Restart seed overlaps a baseline seed")
            trials.append((f"random_{restart}", trial_seed, "random", 0.0))
        trials += [("distance", seed, "distance", 0.0), ("annealed", seed, "distance", entropy_fraction)]
        outcomes = {}
        for name, trial_seed, initialization, entropy in trials:
            folder = root / f"seed_{seed}" / name
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / "condensed.pt"
            if path.exists():
                result = torch.load(path, map_location="cpu", weights_only=True)
            else:
                result = optimize_initialization(
                    h, q, seed=trial_seed, initialization=initialization, entropy_fraction=entropy, **options
                )
                save_state(result, path)
            outcomes[name] = result
            records.append(
                dict(
                    condensation_seed=seed,
                    trial=name,
                    factor_seed=trial_seed,
                    **{key: result[key] for key in ("J_initial", "J_final", "best_step", "seconds")},
                )
            )
            histories.extend(dict(condensation_seed=seed, trial=name, **row) for row in result["history"])
        winner = min(range(starts), key=lambda k: outcomes[f"random_{k}"]["J_final"])
        methods = dict(
            random=outcomes["random_0"],
            distance=outcomes["distance"],
            annealed=outcomes["annealed"],
            multistart=outcomes[f"random_{winner}"],
        )
        for method, result in methods.items():
            seconds = (
                sum(outcomes[f"random_{k}"]["seconds"] for k in range(starts))
                if method == "multistart"
                else result["seconds"]
            )
            name = (
                f"random_{winner}"
                if method == "multistart"
                else ("random_0" if method == "random" else method)
            )
            for student_seed in student_seeds:
                metrics = fit_gcn_diagnostic(
                    result["x"].to(device),
                    result["y"].to(device),
                    torch.ones(old["nodes"], device=device),
                    graph,
                    q,
                    masks,
                    student_seed,
                    **settings,
                    folder=root / f"seed_{seed}" / name / "students",
                )
                evaluations.append(
                    dict(
                        method=method,
                        condensation_seed=seed,
                        student_seed=student_seed,
                        selected_restart=winner if method == "multistart" else 0,
                        seconds=seconds,
                        J_initial=result["J_initial"],
                        J_final=result["J_final"],
                        best_step=result["best_step"],
                        val_acc=metrics["val_acc"],
                        test_acc=metrics["test_acc"],
                    )
                )
        write_table(pd.DataFrame(records), root / "trials.csv")
        write_table(pd.DataFrame(histories), root / "history.csv")
        write_table(pd.DataFrame(evaluations), root / "students.csv")
    students = pd.DataFrame(evaluations)
    by_seed = students.groupby(["method", "condensation_seed"], as_index=False).agg(
        J_initial=("J_initial", "first"),
        J_final=("J_final", "first"),
        seconds=("seconds", "first"),
        val_mean=("val_acc", "mean"),
        val_std=("val_acc", "std"),
        test_mean=("test_acc", "mean"),
        test_std=("test_acc", "std"),
    )
    summary = by_seed.groupby("method", as_index=False).agg(
        J_final=("J_final", "mean"),
        J_std=("J_final", "std"),
        seconds=("seconds", "mean"),
        val_mean=("val_mean", "mean"),
        val_seed_std=("val_mean", "std"),
        test_mean=("test_mean", "mean"),
        test_seed_std=("test_mean", "std"),
    )
    write_table(by_seed, root / "by_seed.csv")
    write_table(summary, root / "summary.csv")
    return summary, by_seed, pd.DataFrame(histories), root
