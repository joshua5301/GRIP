import ast
import json
import subprocess
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric import seed_everything

from src.data import _prepare_dataset
from src.evaluation import fit_gcn_diagnostic
from src.initialization import cell_means, feature_kmeans
from src.io import _fingerprint, array_digest, save_json, save_state, write_table
from src.models import GCN
from src.soft_ce_partition import optimize_ce_assignment
from src.sweep_utils import representative
from src.teacher import teacher_logits
from src.transforms import fit_transform

REFERENCE = "12ceec5810e2c709f98f8aa518d401cd04390168"


def reference_source(name):
    return subprocess.check_output(
        ["git", "show", f"{REFERENCE}:src/{name}.py"],
        cwd=Path(__file__).resolve().parents[1],
        encoding="utf-8",
    )


def reference_functions(source, names, namespace):
    tree = ast.parse(source)
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in tree.body} != set(names):
        raise ValueError("Reference functions are missing")
    exec(compile(tree, REFERENCE, "exec"), namespace)
    return namespace


def hard_representatives(h, q, assignment, cells):
    return (
        cell_means(h.double(), assignment, cells).float(),
        cell_means(q.double(), assignment, cells).float(),
        torch.bincount(assignment, minlength=cells).double() / len(h),
    )


def run_risk_reproduction(
    output_dir,
    condensation_seeds=(0, 1, 2),
    search_seeds=(0, 1, 2),
    final_seeds=tuple(range(100, 110)),
    B=0.6053863613840811,
    gamma=0.01,
    temperature=1.0,
    rank=8,
    penalty=0.001,
    steps=1000,
    checkpoints=(0, 100, 300, 500, 750, 1000),
    data_dir="/content/data/",
    device="cuda",
):
    if 0 not in condensation_seeds or set(search_seeds) & set(final_seeds):
        raise ValueError("Include condensation seed zero and disjoint student seed sets")
    if any(
        not values or len(values) != len(set(values))
        for values in (condensation_seeds, search_seeds, final_seeds)
    ):
        raise ValueError("Seed lists must be nonempty and unique")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    graph, train, validation, testing, h = _prepare_dataset("cora", data_dir, device)
    settings = dict(epochs=1000, eval_every=10, hidden=256, dropout=0.9, lr=0.01, weight_decay=0.0005)
    config = dict(
        reference=REFERENCE,
        dataset="cora",
        ratio=0.013,
        cells=35,
        B=B,
        gamma=gamma,
        T=temperature,
        rank=rank,
        penalty=penalty,
        steps=steps,
        checkpoints=sorted({0, steps, *checkpoints}),
        condensation_seeds=list(condensation_seeds),
        search_seeds=list(search_seeds),
        final_seeds=list(final_seeds),
        student=settings,
        version=1,
        data_digest=array_digest(
            *(v.cpu().numpy() for v in (h, graph["y"], train, validation[1], testing[1]))
        ),
    )
    root = Path(output_dir) / _fingerprint(config)
    root.mkdir(parents=True, exist_ok=True)
    save_json(config, root / "config.json")
    save_json(
        dict(
            torch=str(torch.__version__),
            cuda=torch.version.cuda,
            device=torch.cuda.get_device_name() if str(device).startswith("cuda") else str(device),
        ),
        root / "environment.json",
    )
    sources = {name: reference_source(name) for name in ("risk_partition", "teacher", "risk_experiment")}
    for name, source in sources.items():
        (root / f"reference_{name}.py").write_text(source, encoding="utf-8")
    legacy_teacher = {}
    exec(compile(sources["teacher"], REFERENCE, "exec"), legacy_teacher)
    legacy_student = reference_functions(
        sources["risk_experiment"],
        {"_forward", "_accuracy", "_train_student"},
        dict(torch=torch, F=F, GCN=GCN, seed_everything=seed_everything),
    )
    teacher_path = root / "reference_teacher.pt"
    if teacher_path.exists():
        logits = torch.load(teacher_path, map_location=device, weights_only=True)
    else:
        seed_everything(0)
        phi = legacy_teacher["get_kernel_features"](h, "relu", 3000)
        targets = F.one_hot(graph["y"][train], int(graph["y"].max()) + 1).to(phi)
        weight = legacy_teacher["fit_logistic"](phi[train], targets, gamma)
        logits = (phi @ weight).detach()
        save_state(logits, teacher_path)
    current, _ = teacher_logits(h, graph, train, validation, "relu", [gamma], 3000, 0, root)
    q = (logits / temperature).softmax(1)
    save_json(
        dict(
            max_logit_difference=float((logits - current).abs().max()),
            mean_label_difference=float((q - (current / temperature).softmax(1)).abs().mean()),
            teacher_validation=100
            * float((logits[validation[1]].argmax(1) == graph["y"][validation[1]]).double().mean()),
        ),
        root / "teacher_comparison.json",
    )
    z, transform = fit_transform(h.double(), kind="rms")
    masks = dict(train=train, val=validation[1], test=testing[1])
    records, replay, search_rows = [], [], []
    for seed in condensation_seeds:
        folder = root / f"seed_{seed}"
        folder.mkdir(exist_ok=True)
        risk = {}
        exec(compile(sources["risk_partition"], REFERENCE, "exec"), risk)
        native_seed = risk["seed_partition"]
        artifact = folder / "risk_native.pt"
        if artifact.exists():
            original = torch.load(artifact, map_location="cpu", weights_only=False)
        else:
            captured = {}

            def capture(*args, **kwargs):
                assignment = native_seed(*args, **kwargs)
                captured["initial_assignment"] = assignment.cpu().clone()
                return assignment

            risk["seed_partition"] = capture
            original = risk["risk_partition"](h, q, 35, B, seed=seed)
            original.update(captured)
            save_state(original, artifact)
        initializations = dict(
            risk=original["initial_assignment"].to(device),
            kmeans=feature_kmeans(h.cpu(), 35, seed).to(device),
        )
        for init, assignment in initializations.items():
            base = folder / init
            base.mkdir(exist_ok=True)
            if init == "risk":
                fitted = original
            else:
                path = base / "risk.pt"
                if path.exists():
                    fitted = torch.load(path, map_location="cpu", weights_only=False)
                else:

                    def replace_initialization(*args, **kwargs):
                        native_seed(*args, **kwargs)
                        return assignment.clone()

                    risk["seed_partition"] = replace_initialization
                    fitted = risk["risk_partition"](h, q, 35, B, seed=seed)
                    save_state(fitted, path)
            hard = hard_representatives(h, q, assignment, 35)
            final_risk = (
                fitted["x"].to(device),
                fitted["y"].to(device),
                fitted["counts"].to(device).double() / len(h),
            )
            low = base / "low_rank"
            low.mkdir(exist_ok=True)
            if not (low / "complete.json").exists():
                resume_path = low / "resume.pt"
                resume = (
                    torch.load(resume_path, map_location="cpu", weights_only=False)
                    if resume_path.exists()
                    else None
                )
                optimize_ce_assignment(
                    z,
                    q,
                    assignment,
                    penalty=penalty,
                    steps=steps,
                    lr=0.01,
                    assignment_rank=rank,
                    factor_seed=seed,
                    folder=low,
                    checkpoint_steps=config["checkpoints"],
                    resume_state=resume,
                    save_resume=True,
                    save_assignment=False,
                    inner_method="newton_first",
                    implicit_warm_start=True,
                    inner_loss_weighting="mass",
                    inner_max_iter=2000,
                    inner_tol=1e-7,
                    cg_max_iter=512,
                    cg_rtol=1e-6,
                )
                save_json(dict(steps=steps), low / "complete.json")
            cases = [("hard_initial", 0, hard), ("risk", fitted["sweeps"], final_risk)]
            for step in config["checkpoints"]:
                snapshot = torch.load(
                    low / "checkpoints" / f"step_{step:06d}.pt", map_location="cpu", weights_only=False
                )
                cases.append(
                    ("low_rank", step, representative(snapshot["moments"], transform, z.shape[1], device))
                )
            for method, step, (x, y, mass) in cases:
                cache = base / method / f"eval_{step}"
                for student_seed in search_seeds:
                    score = fit_gcn_diagnostic(
                        x,
                        y,
                        torch.full_like(mass, 1 / len(mass)),
                        graph,
                        q,
                        {k: masks[k] for k in ("train", "val")},
                        student_seed,
                        folder=cache / "search",
                        **settings,
                    )
                    search_rows.append(
                        dict(method=method, initialization=init, condensation_seed=seed, step=step, **score)
                    )
            if seed == 0 and init == "risk":
                params = dict(dropout=0.9, lr=0.01, weight_decay=0.0005)
                for student_seed in final_seeds:
                    path = base / f"reference_student_{student_seed}.json"
                    if path.exists():
                        result = json.loads(path.read_text())
                    else:
                        val, test, epoch = legacy_student["_train_student"](
                            *final_risk[:2], validation, params, student_seed, settings, testing
                        )
                        result = dict(seed=student_seed, val_acc=100 * val, test_acc=100 * test, epoch=epoch)
                        save_json(result, path)
                    replay.append(result)
                save_json(
                    dict(
                        J_initial=original["history"][0],
                        J_final=original["J"],
                        sweeps=original["sweeps"],
                        reported_J_initial=0.10858818925302646,
                        reported_J_final=0.07925720612237395,
                        reported_sweeps=19,
                        reported_test_mean=85.05000472068787,
                        reported_test_std=0.4527697905075321,
                    ),
                    root / "historical_reference.json",
                )
        write_table(pd.DataFrame(search_rows), root / "search_students.csv")
    search = pd.DataFrame(search_rows)
    curve = search.groupby(["method", "initialization", "step"], as_index=False).val_acc.mean()
    selected = (
        curve[curve.method == "low_rank"]
        .sort_values(["val_acc", "step"], ascending=[False, True])
        .drop_duplicates("initialization")
    )
    write_table(selected, root / "selected.csv")
    for seed in condensation_seeds:
        for init in ("risk", "kmeans"):
            base = root / f"seed_{seed}" / init
            native = torch.load(root / f"seed_{seed}" / "risk_native.pt", weights_only=False)
            assignment = (
                native["initial_assignment"].to(device)
                if init == "risk"
                else feature_kmeans(h.cpu(), 35, seed).to(device)
            )
            fitted = native if init == "risk" else torch.load(base / "risk.pt", weights_only=False)
            chosen = int(selected[selected.initialization == init].iloc[0].step)
            cases = [
                ("hard_initial", 0, hard_representatives(h, q, assignment, 35)),
                (
                    "risk",
                    fitted["sweeps"],
                    (fitted["x"].to(device), fitted["y"].to(device), fitted["counts"].to(device).double()),
                ),
            ]
            for step in sorted({0, chosen}):
                snapshot = torch.load(
                    base / "low_rank" / "checkpoints" / f"step_{step:06d}.pt",
                    map_location="cpu",
                    weights_only=False,
                )
                cases.append(
                    (
                        "low_rank_initial" if step == 0 else "low_rank",
                        step,
                        representative(snapshot["moments"], transform, z.shape[1], device),
                    )
                )
            if chosen == 0:
                cases.append(("low_rank", 0, cases[-1][2]))
            for method, step, (x, y, mass) in cases:
                for student_seed in final_seeds:
                    score = fit_gcn_diagnostic(
                        x,
                        y,
                        torch.full_like(mass, 1 / len(mass)),
                        graph,
                        q,
                        masks,
                        student_seed,
                        folder=base / method / f"final_{step}",
                        **settings,
                    )
                    records.append(
                        dict(method=method, initialization=init, condensation_seed=seed, step=step, **score)
                    )
    final = pd.DataFrame(records)
    by_seed = final.groupby(["method", "initialization", "condensation_seed"], as_index=False).agg(
        val_mean=("val_acc", "mean"), test_mean=("test_acc", "mean"), student_std=("test_acc", "std")
    )
    summary = by_seed.groupby(["method", "initialization"], as_index=False).agg(
        val_mean=("val_mean", "mean"), test_mean=("test_mean", "mean"), condensation_std=("test_mean", "std")
    )
    for name, table in (
        ("final_students", final),
        ("by_seed", by_seed),
        ("summary", summary),
        ("reference_students", pd.DataFrame(replay)),
        ("validation_curve", curve),
    ):
        write_table(table, root / f"{name}.csv")
    return summary, by_seed, curve, root
