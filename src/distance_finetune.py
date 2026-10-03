import gc
import hashlib
import itertools
import json
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm

from src.data import _prepare_dataset
from src.evaluation import fit_gcn_diagnostic
from src.io import _fingerprint, save_json, save_state
from src.moment_seeding import normalized_variance_features
from src.moments import AssignmentMoments, decode_moments, make_material
from src.soft_ce_partition import optimize_ce_assignment
from src.variance_moment_sweep import _data_digest


def svd_factors(logits, rank):
    if not 1 <= rank <= min(logits.shape):
        raise ValueError("Invalid SVD rank")
    a, s, bt = torch.linalg.svd(logits.double(), full_matrices=False)
    scale = s[:rank].sqrt() * rank**0.25
    return (a[:, :rank] * scale).float(), (bt[:rank].T * scale).float()


@torch.no_grad()
def distance_logits(h, q, assignment, alpha):
    x = h.double() - h.double().mean(0)
    x /= x.square().sum(1).mean().sqrt().clamp_min(1e-30)
    z, _ = normalized_variance_features(x, q.double(), alpha)
    cells = int(assignment.max()) + 1
    counts = torch.bincount(assignment, minlength=cells)
    centers = z.new_zeros(cells, z.shape[1]).index_add_(0, assignment, z) / counts[:, None]
    d = (z.square().sum(1, keepdim=True) + centers.square().sum(1) - 2 * z @ centers.T).clamp_min(0)
    gaps = d.topk(2, largest=False).values.diff(dim=1).flatten()
    positive = gaps[gaps > 0]
    scale = positive.median() if len(positive) else d.new_tensor(1.0)
    return -(d - d.mean(1, keepdim=True)) / scale, float(scale)


def run_distance_finetune(source, output_dir, ranks=(4, 8, 16), taus=(0.1, 0.3, 1.0),
                          penalties=(1e-5, 1e-4, 1e-3), steps=300,
                          checkpoints=(0, 25, 50, 100, 200, 300), lr=0.01,
                          inner_loss_weighting="mass", data_dir="/content/data/", device="cuda"):
    source = Path(source)
    config = json.loads((source / "config.json").read_text())
    selected = json.loads((source / "selected.json").read_text())
    if config["dataset"] != "citeseer" or config["condensation_seeds"] != [0]:
        raise ValueError("Use a Citeseer single deterministic partition source")
    if not all(0 <= step <= steps for step in checkpoints):
        raise ValueError("Checkpoints must fit the step budget")
    checkpoints = sorted(set(checkpoints) | {0, steps})
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    graph, train, validation, testing, computed = _prepare_dataset("citeseer", data_dir, device)
    del computed
    splits = dict(train=(graph, train), val=validation, test=testing)
    if _data_digest(splits) != config["data_digest"]:
        raise ValueError("Data differ from the stage-one run")
    masks = {name: split[1] for name, split in splits.items()}
    h = torch.load(source / "features.pt", map_location=device, weights_only=True).double()
    teacher = torch.load(source / "teachers" / f"{_fingerprint(dict(gamma=selected['gamma']))}.pt",
                         map_location=device, weights_only=True)
    q = (teacher["logits"] / selected["T"]).softmax(1).double()
    del teacher
    original = torch.load(source / f"candidate_{int(selected['candidate']):04d}" / "seed_0" / "partition.pt",
                          map_location="cpu", weights_only=False)
    assignment = original["assignment"].to(device)
    offset = h.mean(0)
    scale = (h - offset).square().sum(1).mean().sqrt().clamp_min(1e-30)
    z = (h - offset) / scale
    distance, distance_scale = distance_logits(h, q, assignment, selected["lambda"])
    files = ("distance_finetune.py", "soft_ce_partition.py", "low_rank_assignment.py", "moments.py",
             "head.py", "evaluation.py", "models.py", "data.py", "moment_seeding.py")
    code = hashlib.sha256(b"".join((Path(__file__).parent / name).read_bytes() for name in files)).hexdigest()
    settings = dict(source=str(source.resolve()), source_config=config, selected=selected, ranks=list(ranks),
                    taus=list(taus), penalties=list(penalties), steps=steps, checkpoints=checkpoints, lr=lr,
                    inner_loss_weighting=inner_loss_weighting, distance_scale=distance_scale, code=code,
                    cg_max_iter=2048, cg_rtol=1e-6, inner_method="newton_first")
    root = Path(output_dir) / _fingerprint(settings)
    root.mkdir(parents=True, exist_ok=True)
    save_json(settings, root / "config.json")
    save_state(dict(offset=offset.cpu(), scale=scale.cpu()), root / "transform.pt")
    search_seeds, final_seeds = config["search_seeds"], config["final_seeds"]

    def representative(moments):
        c, y, mass = decode_moments(moments.to(z), z.shape[1])
        return (c * scale + offset).float(), y.float(), mass

    def evaluate(x, y, folder, seeds, final=False):
        chosen_masks = masks if final else {key: masks[key] for key in ("train", "val")}
        return [dict(student_seed=seed, **fit_gcn_diagnostic(
            x, y, torch.ones(len(x), device=device), graph, None, chosen_masks, seed,
            folder=folder, **config["student"],
        )) for seed in seeds]

    rows, final_rows = [], []
    baseline = evaluate(original["x"].to(device), original["y"].to(device), root / "hard" / "search", search_seeds)
    baseline_val = float(pd.DataFrame(baseline).val_acc.mean())
    init_cache = {}
    grid = list(itertools.product(("fixed_D", "svd_UV"), ranks, taus, penalties))
    for candidate, (method, rank, tau, penalty) in enumerate(tqdm(grid, desc="Distance-initialized bilevel sweep")):
        key = (rank, tau)
        if key not in init_cache:
            target = distance / tau
            factors = svd_factors(target, rank)
            reconstructed = factors[0].double() @ factors[1].double().T / rank**0.5
            init_cache[key] = (factors, float((reconstructed - target).norm() / target.norm().clamp_min(1e-30)))
        factors, error = init_cache[key]
        base = distance / tau if method == "fixed_D" else torch.zeros_like(distance)
        folder = root / f"candidate_{candidate:04d}"
        folder.mkdir(exist_ok=True)
        artifact = folder / "optimized.pt"
        if artifact.exists():
            optimized = torch.load(artifact, map_location="cpu", weights_only=False)
        else:
            resume = folder / "resume.pt"
            state = torch.load(resume, map_location="cpu", weights_only=False) if resume.exists() else None
            optimized = optimize_ce_assignment(
                z, q, assignment, penalty=penalty, steps=steps, lr=lr, assignment_rank=rank,
                factor_seed=0, base_logits=base, correction_scale=1 / tau if method == "fixed_D" else 1.0,
                initial_factors=factors if method == "svd_UV" else None,
                checkpoint_steps=checkpoints, folder=folder, resume_state=state, save_resume=True,
                save_assignment=False, inner_method="newton_first", implicit_warm_start=True,
                inner_loss_weighting=inner_loss_weighting, cg_max_iter=2048, cg_rtol=1e-6,
            )
            save_state(optimized, artifact)
        for step in checkpoints:
            moments = optimized["checkpoints"][step]["moments"]
            x, y, _ = representative(moments)
            scores = pd.DataFrame(evaluate(x, y, folder / f"search_step_{step}", search_seeds))
            rows.append(dict(candidate=candidate, method=method, rank=rank, tau=tau, penalty=penalty,
                             step=step, search_val=float(scores.val_acc.mean()),
                             search_val_std=float(scores.val_acc.std()),
                             outer_ce=optimized["checkpoints"][step]["teacher_ce"],
                             svd_relative_error=error if method == "svd_UV" else 0.0))
        pd.DataFrame(rows).to_csv(root / "search.csv", index=False)
        del optimized
        gc.collect()
    search = pd.DataFrame(rows)
    hard_scores = pd.DataFrame(evaluate(original["x"].to(device), original["y"].to(device),
                                        root / "hard" / "final", final_seeds, final=True))

    def report(method, phase, scores, params, search_val):
        final_rows.append(dict(method=method, phase=phase, **params, search_val=search_val,
                               final_val=float(scores.val_acc.mean()), final_val_std=float(scores.val_acc.std()),
                               test_mean=float(scores.test_acc.mean()), test_std=float(scores.test_acc.std())))
        pd.DataFrame(final_rows).to_csv(root / "summary.csv", index=False)

    report("hard", "stage_one", hard_scores, dict(rank=None, tau=None, penalty=None, step=0), baseline_val)
    for method in ("fixed_D", "svd_UV"):
        winner = search[search.method == method].sort_values(
            ["search_val", "step", "candidate"], ascending=[False, True, True]).iloc[0]
        candidate, step = int(winner.candidate), int(winner.step)
        folder = root / f"candidate_{candidate:04d}"
        optimized = torch.load(folder / "optimized.pt", map_location="cpu", weights_only=False)
        for phase, checkpoint in (("initial", 0), ("selected", step)):
            x, y, _ = representative(optimized["checkpoints"][checkpoint]["moments"])
            scores = pd.DataFrame(evaluate(x, y, folder / f"final_step_{checkpoint}", final_seeds, final=True))
            params = dict(rank=int(winner["rank"]), tau=float(winner.tau), penalty=float(winner.penalty), step=checkpoint)
            val = float(search[(search.candidate == candidate) & (search.step == checkpoint)].iloc[0].search_val)
            report(method, phase, scores, params, val)
            scores.assign(method=method, phase=phase).to_csv(root / f"{method}_{phase}_students.csv", index=False)
        if method == "svd_UV":
            logits = distance / float(winner.tau)
            moments = AssignmentMoments.apply(logits, make_material(z, q), 4096)
            x, y, _ = representative(moments)
            scores = pd.DataFrame(evaluate(x, y, root / f"soft_reference_tau_{winner.tau}" / "final", final_seeds, final=True))
            report(method, "full_distance_soft_reference", scores,
                   dict(rank=None, tau=float(winner.tau), penalty=None, step=0), float("nan"))
    return pd.DataFrame(final_rows), search, root
