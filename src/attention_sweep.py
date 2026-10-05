import hashlib
import itertools
import json
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm

from src.attention_assignment import AttentionAssignment
from src.data import _prepare_dataset
from src.distance_finetune import evaluation_splits
from src.evaluation import fit_gcn_diagnostic
from src.io import _fingerprint, save_json, save_state
from src.moments import decode_moments
from src.soft_ce_partition import optimize_ce_assignment
from src.variance_moment_sweep import _data_digest


def run_attention_sweep(source, output_dir, ranks=(8, 16, 32), taus=(0.1, 0.3, 1.0),
                        penalties=(1e-5, 1e-4, 1e-3), steps=300,
                        checkpoints=(0, 25, 50, 100, 200, 300), lr=0.01,
                        methods=("metric", "attention"), data_dir="/content/data/", device="cuda"):
    source = Path(source)
    config = json.loads((source / "config.json").read_text())
    selected = json.loads((source / "selected.json").read_text())
    if config.get("partition_method") != "variance_sum" or config.get("initialization_space") != "joint":
        raise ValueError("Use a joint Var-Part label-variance source")
    if not methods or any(m not in ("metric", "attention") for m in methods):
        raise ValueError("Invalid methods")
    checkpoints = sorted(set(checkpoints) | {0, steps})
    if any(s < 0 or s > steps for s in checkpoints):
        raise ValueError("Invalid checkpoints")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    graph, train, validation, testing, computed = _prepare_dataset(config["dataset"], data_dir, device)
    del computed
    if _data_digest(dict(train=(graph, train), val=validation, test=testing)) != config["data_digest"]:
        raise ValueError("Data differ from source")
    masks = evaluation_splits(graph, train, validation, testing)
    h = torch.load(source / "features.pt", map_location=device, weights_only=True).double()
    teacher = torch.load(source / "teacher.pt", map_location=device, weights_only=False)
    q = (teacher["logits"] / selected["T"]).softmax(1).double()
    del teacher
    original = torch.load(source / f"candidate_{int(selected['candidate']):04d}" / "partition.pt",
                          map_location=device, weights_only=False)
    assignment = original["assignment"]
    offset = h.mean(0)
    scale = (h - offset).square().sum(1).mean().sqrt().clamp_min(1e-30)
    z = (h - offset) / scale
    inputs = torch.cat((z, selected["lambda"]**0.5 * (q - q.mean(0))), 1)
    counts = torch.bincount(assignment, minlength=config["nodes"]).to(z)
    centers = inputs.new_zeros(config["nodes"], inputs.shape[1]).index_add_(0, assignment, inputs) / counts[:, None]
    files = ("attention_sweep.py", "attention_assignment.py", "soft_ce_partition.py",
             "low_rank_assignment.py", "moments.py", "head.py", "evaluation.py", "models.py", "data.py")
    code = hashlib.sha256(b"".join((Path(__file__).parent / f).read_bytes() for f in files)).hexdigest()
    identity = dict(source=str(source.resolve()), source_config=config, selected=selected,
                    ranks=list(ranks), taus=list(taus), penalties=list(penalties), steps=steps,
                    checkpoints=checkpoints, lr=lr, methods=list(methods), code=code,
                    inner_tol=1e-5, cg_rtol=1e-3, inner_loss="uniform", student_loss="uniform",
                    artifacts={f: hashlib.sha256((source / f).read_bytes()).hexdigest()
                               for f in ("teacher.pt", "features.pt")})
    root = Path(output_dir) / _fingerprint(identity)
    root.mkdir(parents=True, exist_ok=True)
    save_json(identity, root / "config.json")

    def evaluate(x, y, folder, final=False):
        seeds = config["final_seeds"] if final else config["search_seeds"]
        split = masks if final else {k: masks[k] for k in ("train", "val")}
        return pd.DataFrame([dict(student_seed=seed, **fit_gcn_diagnostic(
            x, y, torch.ones(len(x), device=device), graph, None, split, seed,
            folder=folder, **config["settings"])) for seed in seeds])

    def decode(moments):
        x, y, _ = decode_moments(moments.to(z), z.shape[1])
        return (x * scale + offset).float(), y.float()

    rows, reports = [], []
    grid = list(itertools.product(methods, ranks, taus, penalties))
    for index, (method, rank, tau, penalty) in enumerate(tqdm(grid, desc="Assignment sweep")):
        folder = root / f"candidate_{index:04d}"
        folder.mkdir(exist_ok=True)
        path = folder / "optimized.pt"
        if path.exists():
            optimized = torch.load(path, map_location="cpu", weights_only=False)
        else:
            model = AttentionAssignment(inputs, centers, rank, tau, method)
            resume = folder / "resume.pt"
            state = torch.load(resume, map_location="cpu", weights_only=False) if resume.exists() else None
            optimized = optimize_ce_assignment(
                z, q, assignment, assignment_model=model, penalty=penalty, steps=steps, lr=lr,
                folder=folder, checkpoint_steps=checkpoints, resume_state=state, save_resume=True,
                save_assignment=False, inner_method="newton_first", implicit_warm_start=True,
                inner_loss_weighting="uniform", inner_tol=1e-5, cg_rtol=1e-3, cg_max_iter=2048)
            save_state(optimized, path)
            del model, state
        for step in checkpoints:
            scores = evaluate(*decode(optimized["checkpoints"][step]["moments"]),
                              folder / f"search_{step}")
            rows.append(dict(candidate=index, method=method, rank=rank, tau=tau, penalty=penalty,
                             step=step, search_val=scores.val_acc.mean()))
        pd.DataFrame(rows).to_csv(root / "search.csv", index=False)
        del optimized
    search = pd.DataFrame(rows)

    def report(method, phase, scores, params):
        scores.to_csv(root / f"{method}_{phase}_students.csv", index=False)
        reports.append(dict(method=method, phase=phase, **params, final_val=scores.val_acc.mean(),
                            final_val_std=scores.val_acc.std(), test_mean=scores.test_acc.mean(),
                            test_std=scores.test_acc.std()))
        pd.DataFrame(reports).to_csv(root / "summary.csv", index=False)

    report("hard", "stage_one", evaluate(original["x"], original["y"], root / "hard", True), dict(step=0))
    for method in methods:
        winner = search[search.method == method].sort_values(
            ["search_val", "step", "candidate"], ascending=[False, True, True]).iloc[0]
        folder = root / f"candidate_{int(winner.candidate):04d}"
        optimized = torch.load(folder / "optimized.pt", map_location="cpu", weights_only=False)
        for phase, step in (("initial", 0), ("selected", int(winner.step))):
            scores = evaluate(*decode(optimized["checkpoints"][step]["moments"]),
                              folder / f"final_{step}", True)
            report(method, phase, scores, dict(rank=int(winner["rank"]), tau=float(winner.tau),
                                               penalty=float(winner.penalty), step=step))
    return pd.DataFrame(reports), search, root
