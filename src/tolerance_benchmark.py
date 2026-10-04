import hashlib
import json
from pathlib import Path
from time import perf_counter

import pandas as pd
import torch

from src.data import _prepare_dataset
from src.distance_finetune import evaluation_splits, factorized_distance, factorized_svd
from src.evaluation import fit_gcn_diagnostic
from src.io import _fingerprint, save_json, save_state, write_table
from src.moments import decode_moments
from src.soft_ce_partition import optimize_ce_assignment, outer_value_gradient, solve_inner_newton_first
from src.variance_moment_sweep import _data_digest

TOLERANCES = dict(strict=(1e-7, 1e-6), relaxed=(1e-6, 1e-4), loose=(1e-5, 1e-3))


def run_tolerance_benchmark(reference_run, output_dir, rank=64, tau=0.3, penalty=1e-6,
                            method="fixed_D", inner_loss_weighting="uniform", steps=100,
                            repeats=2, lr=0.01, data_dir="/content/data/", device="cuda"):
    if method not in ("fixed_D", "svd_UV") or repeats < 1 or steps < 1:
        raise ValueError("Require a supported method and positive budgets")
    reference_run = Path(reference_run)
    reference = json.loads((reference_run / "config.json").read_text())
    config, selected = reference["source_config"], reference["selected"]
    source = Path(reference["source"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    graph, train, validation, testing, computed = _prepare_dataset(config["dataset"], data_dir, device)
    del computed
    if _data_digest(dict(train=(graph, train), val=validation, test=testing)) != config["data_digest"]:
        raise ValueError("Graph or splits differ from the reference")
    masks = evaluation_splits(graph, train, validation, testing)
    masks = {name: masks[name] for name in ("train", "val")}
    h = torch.load(source / "features.pt", map_location=device, weights_only=True).double()
    teacher = torch.load(source / "teachers" / f"{_fingerprint(dict(gamma=selected['gamma']))}.pt",
                         map_location=device, weights_only=True)
    q = (teacher["logits"] / selected["T"]).softmax(1).double()
    del teacher
    partition = torch.load(source / f"candidate_{int(selected['candidate']):04d}/seed_0/partition.pt",
                           map_location="cpu", weights_only=False)
    assignment = partition["assignment"].to(device)
    del partition
    offset, scale = h.mean(0), (h - h.mean(0)).square().sum(1).mean().sqrt().clamp_min(1e-30)
    z = (h - offset) / scale
    left, right, _ = factorized_distance(h, q, assignment, selected["lambda"])
    factors = None
    if method == "svd_UV":
        a, s, b = factorized_svd(left, right)
        if rank > len(s):
            raise ValueError("Rank exceeds distance factorization dimension")
        values = (s[:rank] / tau).sqrt() * rank**0.25
        factors = ((a[:, :rank] * values).float(), (b[:, :rank] * values).float())
        base = (z.new_zeros(len(z), 1), z.new_zeros(int(assignment.max()) + 1, 1))
    else:
        base = (left / tau, right)
    code = hashlib.sha256(b"".join(
        (Path(__file__).parent / file).read_bytes()
        for file in ("tolerance_benchmark.py", "soft_ce_partition.py", "low_rank_assignment.py", "evaluation.py")
    )).hexdigest()
    settings = dict(reference=str(reference_run), reference_config=reference, rank=rank, tau=tau,
                    penalty=penalty, method=method, inner_loss_weighting=inner_loss_weighting,
                    steps=steps, repeats=repeats, lr=lr, tolerances=TOLERANCES, code=code)
    root = Path(output_dir) / _fingerprint(settings)
    root.mkdir(parents=True, exist_ok=True)
    save_json(settings, root / "config.json")

    def sync():
        if z.is_cuda:
            torch.cuda.synchronize(z.device)

    # Reverse alternate repeats to reduce a systematic timing-order advantage.
    trials = [(repeat, name) for repeat in range(repeats)
              for name in (list(TOLERANCES) if repeat % 2 == 0 else list(TOLERANCES)[::-1])]
    timings = []
    for repeat, name in trials:
        folder = root / f"repeat_{repeat}" / name
        folder.mkdir(parents=True, exist_ok=True)
        artifact, timing_path = folder / "optimized.pt", folder / "timing.json"
        if artifact.exists() and timing_path.exists():
            timings.append(json.loads(timing_path.read_text()))
            continue
        inner_tol, cg_rtol = TOLERANCES[name]
        print(f"{name} | 반복={repeat + 1} | inner_tol={inner_tol} | cg_rtol={cg_rtol}", flush=True)
        sync()
        started = perf_counter()
        optimized = optimize_ce_assignment(
            z, q, assignment, penalty=penalty, steps=steps, lr=lr, assignment_rank=rank,
            factor_seed=0, initial_factors=factors, base_factors=base,
            correction_scale=1 / tau if method == "fixed_D" else 1.0,
            inner_loss_weighting=inner_loss_weighting, inner_method="newton_first",
            implicit_warm_start=True, inner_tol=inner_tol, cg_rtol=cg_rtol,
            cg_max_iter=2048, solver_mode="exact", checkpoint_steps=[0, steps],
            folder=folder, save_assignment=False, save_resume=False,
        )
        sync()
        seconds = perf_counter() - started
        save_state(optimized, artifact)
        history = pd.DataFrame(optimized["history"])
        updates = history[(history.step > 0) & (history.step < steps)]
        timing = dict(repeat=repeat, tolerance=name, inner_tol=inner_tol, cg_rtol=cg_rtol,
                      optimization_seconds=seconds)
        for column in ("assignment_seconds", "inner_seconds", "outer_seconds", "implicit_seconds",
                       "backward_seconds", "cg_iterations", "inner_newton_cg_iterations",
                       "inner_lbfgs_fallback"):
            timing[column] = float(updates[column].mean())
        save_json(timing, timing_path)
        timings.append(timing)
        del optimized
    write_table(pd.DataFrame(timings), root / "timings.csv")

    records = []
    for timing in timings:
        repeat, name = timing["repeat"], timing["tolerance"]
        folder = root / f"repeat_{repeat}" / name
        optimized = torch.load(folder / "optimized.pt", map_location="cpu", weights_only=False)
        checkpoint = optimized["checkpoints"][steps]
        c, y, mass = decode_moments(checkpoint["moments"].to(z), z.shape[1])
        student_mass = torch.full_like(mass, 1 / len(mass)) if inner_loss_weighting == "uniform" else mass
        fitted = solve_inner_newton_first(c, y, student_mass, penalty,
                                         initial=checkpoint["theta"].to(z), grad_tol=1e-7,
                                         cg_max_iter=2048)
        if not fitted["inner_converged"]:
            raise RuntimeError(f"Strict verification did not converge: {folder}")
        ce, _ = outer_value_gradient(z, q, fitted["theta"], chunk_size=65536)
        x = (c * scale + offset).float()
        scores = pd.DataFrame([
            fit_gcn_diagnostic(x, y.float(), torch.ones_like(mass), graph, None, masks, seed,
                               folder=folder / "gcn_validation", **config["student"])
            for seed in config["search_seeds"]
        ])
        records.append(dict(timing, reported_outer_ce=checkpoint["teacher_ce"],
                            verified_outer_ce=ce, verification_gradient=fitted["inner_grad_max"],
                            validation=float(scores.val_acc.mean()), validation_std=float(scores.val_acc.std())))
        write_table(pd.DataFrame(records), root / "summary.csv")
        del optimized
    return pd.DataFrame(records), root
