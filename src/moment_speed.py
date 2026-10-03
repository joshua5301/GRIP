import gc
import hashlib
import json
import random
from pathlib import Path
from time import perf_counter

import pandas as pd
import torch
from tqdm.auto import tqdm

from src.distance_initialization import distance_factors, project_features
from src.io import _fingerprint, array_digest, save_json, save_state, write_table
from src.variance_moment_low_rank import moment_objective

BACKENDS = ("baseline", "direct", "cached", "cached_full", "full_autograd")


class AssignmentStatistics(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, v, material, block_size, cache):
        n, k = len(u), len(v)
        probability = u.new_empty(n, k) if cache else u.new_empty(0)
        statistics = material.new_zeros(k, material.shape[1])
        scale = u.shape[1] ** -0.5
        for start in range(0, n, block_size):
            end = min(start + block_size, n)
            p = (u[start:end] @ v.T * scale).softmax(1)
            statistics.addmm_(p.T, material[start:end])
            if cache:
                probability[start:end].copy_(p)
        ctx.save_for_backward(u, v, material, probability)
        ctx.block_size, ctx.cache = block_size, cache
        return statistics / n

    @staticmethod
    def backward(ctx, derivative):
        u, v, material, probability = ctx.saved_tensors
        du, dv = torch.empty_like(u), torch.zeros_like(v)
        scale, n = u.shape[1] ** -0.5, len(u)
        for start in range(0, n, ctx.block_size):
            end = min(start + ctx.block_size, n)
            p = probability[start:end] if ctx.cache else (u[start:end] @ v.T * scale).softmax(1)
            direction = material[start:end] @ derivative.T / n
            direction.sub_((p * direction).sum(1, keepdim=True)).mul_(p).mul_(scale)
            du[start:end] = direction @ v
            dv.addmm_(direction.T, u[start:end])
        return du, dv, None, None, None


def _chunk(u, v, x, q, start, end):
    p = (u[start:end] @ v.T / u.shape[1] ** 0.5).softmax(1) / len(u)
    return p.sum(0), p.T @ x[start:end], p.T @ q[start:end]


def forward_objective(u, v, x, q, material, energy, original, weight, backend, block_size):
    if backend not in BACKENDS:
        raise ValueError(f"Unknown backend: {backend}")
    if backend == "baseline":
        with torch.no_grad():
            stats = [x.new_zeros(len(v)), x.new_zeros(len(v), x.shape[1]), q.new_zeros(len(v), q.shape[1])]
            for start in range(0, len(u), block_size):
                for total, part in zip(stats, _chunk(u, v, x, q, start, start + block_size)):
                    total.add_(part)
        stats = [part.requires_grad_() for part in stats]
    else:
        if backend == "full_autograd":
            p = (u @ v.T / u.shape[1] ** 0.5).softmax(1)
            combined = p.T @ material / len(u)
        else:
            size = len(u) if backend == "cached_full" else block_size
            combined = AssignmentStatistics.apply(u, v, material, size, backend != "direct")
        stats = combined[:, 0], combined[:, 1 : 1 + x.shape[1]], combined[:, 1 + x.shape[1] :]
    smooth, exact, _, _ = moment_objective(stats, energy, original, None, moment_weight=weight)
    return smooth, exact, stats


def backward_objective(smooth, stats, u, v, x, q, backend, block_size):
    if backend == "baseline":
        derivatives = torch.autograd.grad(smooth, stats)
        for start in range(0, len(u), block_size):
            parts = _chunk(u, v, x, q, start, start + block_size)
            sum((part * gradient).sum() for part, gradient in zip(parts, derivatives)).backward()
    else:
        smooth.backward()


def _relative(actual, expected):
    return float((actual - expected).norm() / expected.norm().clamp_min(1e-12))


def gradient_snapshot(factors, data, weight, backend, block_size):
    u, v = [tensor.detach().clone().requires_grad_() for tensor in factors]
    x, q, material, energy, original = data
    smooth, exact, stats = forward_objective(u, v, *data, weight, backend, block_size)
    backward_objective(smooth, stats, u, v, x, q, backend, block_size)
    return dict(J=float(exact.detach()), du=u.grad.detach().cpu(), dv=v.grad.detach().cpu())


def _trajectory(factors, data, weight, backend, block_size, lr, steps, timed=False):
    u, v = [torch.nn.Parameter(tensor.detach().clone()) for tensor in factors]
    optimizer = torch.optim.Adam([u, v], lr=lr, foreach=False)
    x, q, material, energy, original = data
    rows, events, best, best_stats = [], [], float("inf"), None
    if timed:
        torch.cuda.synchronize(x.device)
        torch.cuda.reset_peak_memory_stats(x.device)
    started = perf_counter()
    for step in range(steps):
        marks = [torch.cuda.Event(enable_timing=True) for _ in range(4)] if timed else None
        if marks:
            marks[0].record()
        optimizer.zero_grad(set_to_none=True)
        smooth, exact, stats = forward_objective(u, v, *data, weight, backend, block_size)
        if not bool(torch.stack([torch.isfinite(s).all() for s in stats] + [(stats[0] > 0).all()]).all()):
            raise FloatingPointError(f"Invalid statistics at step {step}")
        value = float(exact.detach())
        if value < best:
            best, best_stats = value, [s.detach().clone() for s in stats]
        if marks:
            marks[1].record()
        backward_objective(smooth, stats, u, v, x, q, backend, block_size)
        if not bool(torch.stack([torch.isfinite(t.grad).all() for t in (u, v)]).all()):
            raise FloatingPointError(f"Nonfinite gradient at step {step}")
        if marks:
            marks[2].record()
        optimizer.step()
        if marks:
            marks[3].record()
            events.append(marks)
        rows.append(dict(step=step, J=value, best_J=best))
    if timed:
        torch.cuda.synchronize(x.device)
    seconds = perf_counter() - started
    peak = torch.cuda.max_memory_allocated(x.device) / 2**30 if timed else 0.0
    reserved = torch.cuda.max_memory_reserved(x.device) / 2**30 if timed else 0.0
    with torch.no_grad():
        mass = x.new_zeros(len(v))
        sx, sq = x.new_zeros(len(v), x.shape[1]), q.new_zeros(len(v), q.shape[1])
        for start in range(0, len(u), block_size):
            for total, part in zip((mass, sx, sq), _chunk(u, v, x, q, start, start + block_size)):
                total.add_(part)
        _, last, _, _ = moment_objective((mass, sx, sq), energy, original, None, moment_weight=weight)
        if not bool(torch.isfinite(last)):
            raise FloatingPointError("Nonfinite objective after the last update")
        if float(last) < best:
            best, best_stats = float(last), [mass.clone(), sx.clone(), sq.clone()]
    rows.append(dict(step=steps, J=float(last), best_J=best))
    phase = dict(forward_ms=0.0, backward_ms=0.0, adam_ms=0.0)
    if timed:
        for name, index in (("forward_ms", 0), ("backward_ms", 1), ("adam_ms", 2)):
            phase[name] = sum(e[index].elapsed_time(e[index + 1]) for e in events) / steps
    return (
        dict(
            seconds=seconds,
            step_ms=seconds * 1000 / steps,
            peak_allocated_gib=peak,
            peak_reserved_gib=reserved,
            J_final=float(last),
            J_best=best,
            **phase,
        ),
        rows,
        (u.detach(), v.detach()),
        best_stats,
    )


def benchmark_moments(
    H,
    Q,
    factors,
    weight,
    output_dir,
    steps=100,
    warmup=5,
    repeats=3,
    block_sizes=(8192, 32768),
    lr=0.01,
    metadata=None,
):
    if not H.is_cuda or not block_sizes or min(steps, warmup, repeats, *block_sizes) < 1:
        raise ValueError("Use CUDA with positive timing budgets and block sizes")
    device = H.device
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    x, q = H.detach().double(), Q.detach().double()
    x = x - x.mean(0)
    x = x / x.square().sum(1).mean().sqrt().clamp_min(1e-12)
    material = torch.cat((x.new_ones(len(x), 1), x, q), 1)
    data = x, q, material, x.square().sum(1).mean(), x.T @ q / len(x)
    factors = tuple(t.detach().to(device=device, dtype=torch.double) for t in factors)
    variants = [("baseline", block_sizes[0])]
    variants += [(backend, size) for backend in ("direct", "cached") for size in block_sizes]
    variants += [(backend, len(x)) for backend in ("cached_full", "full_autograd")]
    config = dict(
        version=1,
        steps=steps,
        warmup=warmup,
        repeats=repeats,
        block_sizes=list(block_sizes),
        lr=lr,
        weight=weight,
        shape=[len(x), len(factors[1]), x.shape[1], factors[0].shape[1]],
        data_digest=array_digest(H.cpu().numpy(), Q.cpu().numpy()),
        factors_digest=array_digest(*[t.cpu().numpy() for t in factors]),
        gpu=torch.cuda.get_device_name(device),
        torch=str(torch.__version__),
        metadata=metadata,
        source_digest=hashlib.sha256(
            b"".join(
                Path(__file__).with_name(name).read_bytes()
                for name in ("moment_speed.py", "variance_moment_low_rank.py", "distance_initialization.py")
            )
        ).hexdigest(),
    )
    root = Path(output_dir) / _fingerprint(config)
    root.mkdir(parents=True, exist_ok=True)
    save_json(config, root / "config.json")
    _, _, trained, _ = _trajectory(factors, data, weight, "baseline", block_sizes[0], lr, warmup)
    snapshots = [factors, trained]
    reference = [gradient_snapshot(state, data, weight, "baseline", block_sizes[0]) for state in snapshots]
    verification, timings, history, failures, accepted = [], [], [], [], []
    for backend, size in tqdm(variants, desc="Exact gradient verification"):
        label = f"{backend}_{size}"
        try:
            checks = []
            for index, state in enumerate(snapshots):
                actual = gradient_snapshot(state, data, weight, backend, size)
                expected = reference[index]
                passed = all(
                    torch.allclose(actual[key], expected[key], atol=1e-10, rtol=1e-7) for key in ("du", "dv")
                )
                passed = passed and abs(actual["J"] - expected["J"]) <= 1e-9 * max(1.0, abs(expected["J"]))
                checks.append(
                    dict(
                        variant=label,
                        state=index,
                        passed=passed,
                        J_abs_error=abs(actual["J"] - expected["J"]),
                        U_grad_relative_error=_relative(actual["du"], expected["du"]),
                        V_grad_relative_error=_relative(actual["dv"], expected["dv"]),
                    )
                )
            verification.extend(checks)
            if all(row["passed"] for row in checks):
                accepted.append((backend, size))
        except (torch.OutOfMemoryError, FloatingPointError) as error:
            failures.append(dict(variant=label, stage="verification", reason=str(error)))
        gc.collect()
        torch.cuda.empty_cache()
    del snapshots, trained, reference, state
    write_table(pd.DataFrame(verification), root / "gradient_checks.csv")
    for repeat in range(repeats):
        order = list(accepted)
        random.Random(repeat + 37).shuffle(order)
        for backend, size in tqdm(order, desc=f"Speed repeat {repeat + 1}/{repeats}"):
            label = f"{backend}_{size}"
            path = root / f"{label}_repeat_{repeat}.pt"
            try:
                if path.exists():
                    saved = torch.load(path, map_location="cpu", weights_only=True)
                else:
                    warm = _trajectory(factors, data, weight, backend, size, lr, warmup)
                    del warm
                    gc.collect()
                    torch.cuda.empty_cache()
                    metrics, rows, final_factors, best_stats = _trajectory(
                        factors, data, weight, backend, size, lr, steps, timed=True
                    )
                    saved = dict(
                        metrics=metrics,
                        history=rows,
                        factors=[t.cpu() for t in final_factors],
                        statistics=[t.cpu() for t in best_stats],
                    )
                    save_state(saved, path)
                    del final_factors, best_stats
                timings.append(
                    dict(variant=label, backend=backend, block_size=size, repeat=repeat, **saved["metrics"])
                )
                history.extend(dict(variant=label, repeat=repeat, **row) for row in saved["history"])
                del saved
            except (torch.OutOfMemoryError, FloatingPointError) as error:
                failures.append(dict(variant=label, repeat=repeat, stage="timing", reason=str(error)))
            gc.collect()
            torch.cuda.empty_cache()
            write_table(pd.DataFrame(timings), root / "timings.csv")
            save_json(failures, root / "failures.json")
    timing = pd.DataFrame(timings)
    if timing.empty:
        raise RuntimeError(f"No successful timing runs; inspect {root}")
    summary = timing.groupby("variant", as_index=False).agg(
        repeats=("repeat", "count"),
        step_ms=("step_ms", "median"),
        step_ms_std=("step_ms", "std"),
        forward_ms=("forward_ms", "median"),
        backward_ms=("backward_ms", "median"),
        adam_ms=("adam_ms", "median"),
        peak_allocated_gib=("peak_allocated_gib", "max"),
        peak_reserved_gib=("peak_reserved_gib", "max"),
        J_final=("J_final", "mean"),
        J_best=("J_best", "mean"),
    )
    baseline = summary[summary.variant == f"baseline_{block_sizes[0]}"]
    summary["speedup"] = float(baseline.step_ms.iloc[0]) / summary.step_ms if len(baseline) else float("nan")
    summary["projected_3000_minutes"] = summary.step_ms * 3000 / 60000
    comparisons = []
    for repeat in range(repeats):
        ref_path = root / f"baseline_{block_sizes[0]}_repeat_{repeat}.pt"
        if not ref_path.exists():
            continue
        reference = torch.load(ref_path, weights_only=True)
        for row in timing[timing.repeat == repeat].itertuples():
            actual = torch.load(root / f"{row.variant}_repeat_{repeat}.pt", weights_only=True)
            comparisons.append(
                dict(
                    variant=row.variant,
                    repeat=repeat,
                    J_final_abs_error=abs(actual["metrics"]["J_final"] - reference["metrics"]["J_final"]),
                    U_relative_error=_relative(actual["factors"][0], reference["factors"][0]),
                    V_relative_error=_relative(actual["factors"][1], reference["factors"][1]),
                    statistics_relative_error=_relative(
                        torch.cat([t.flatten() for t in actual["statistics"]]),
                        torch.cat([t.flatten() for t in reference["statistics"]]),
                    ),
                )
            )
    write_table(summary, root / "summary.csv")
    write_table(pd.DataFrame(history), root / "history.csv")
    write_table(pd.DataFrame(comparisons), root / "trajectory_checks.csv")
    return summary, pd.DataFrame(verification), pd.DataFrame(comparisons), root


def benchmark_saved_run(previous_run, output_dir, rank=32, weight=1.0, seed=0, **options):
    previous = Path(previous_run)
    config = json.loads((previous / "config.json").read_text())
    if config.get("initialization") != "distance" or config["space"]["T"] != [1.0]:
        raise ValueError("Use a distance-initialized sweep with T=1")
    h = torch.load(previous / "features.pt", map_location="cuda", weights_only=True)
    expected = json.loads((previous / "features.json").read_text())["h_digest"]
    if array_digest(h.cpu().numpy()) != expected:
        raise ValueError("Cached features differ from source digest")
    teacher = json.loads((previous / "selected_teacher.json").read_text())
    logits = torch.load(
        previous / "teachers" / f"{_fingerprint(dict(gamma=teacher['gamma']))}.pt",
        map_location="cuda",
        weights_only=True,
    )["logits"]
    q = logits.softmax(1)
    cells = config["nodes"]
    factors_path = previous / "initializations" / f"rank_{rank}_cells_{cells}_seed_{seed}.pt"
    if factors_path.exists():
        factors = torch.load(factors_path, map_location="cuda", weights_only=True)
    else:
        x = h.double() - h.double().mean(0)
        x = x / x.square().sum(1).mean().sqrt().clamp_min(1e-12)
        projection_path = previous / "initializations" / f"projection_{rank}.pt"
        projected = (
            torch.load(projection_path, map_location="cuda", weights_only=True)
            if projection_path.exists()
            else project_features(x, rank)
        )
        factors = distance_factors(projected, cells, rank, seed, block_size=8192)
        del x, projected
    del logits
    return benchmark_moments(
        h,
        q,
        factors,
        weight,
        output_dir,
        metadata=dict(
            previous_run=str(previous),
            dataset=config["dataset"],
            ratio=config["ratio"],
            gamma=teacher["gamma"],
            T=1.0,
            seed=seed,
        ),
        **options,
    )
