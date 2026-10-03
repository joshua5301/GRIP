import gc
import hashlib
import json
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm

from src.data import _prepare_dataset
from src.evaluation import fit_gcn_diagnostic
from src.io import _fingerprint, save_json, save_state, write_table
from src.moment_lloyd import moment_lloyd_partition, statistics
from src.moment_seeding import normalized_variance_features
from src.variance_moment_sweep import _data_digest


@torch.no_grad()
def common_objective(h, q, assignment, cells, alpha):
    x = h.double() - h.double().mean(0)
    x /= x.square().sum(1).mean().sqrt().clamp_min(1e-30)
    q = q.double()
    _, scaling = normalized_variance_features(x, q, alpha)
    state = statistics(x, q, assignment.to(x.device), cells, x.square().sum(1).mean(), x.T @ q / len(x))
    vq = q.square().sum(1).mean() - (state[0] * state[2].square().sum(1)).sum() / len(x)
    rx, rq = float(state[3]) * scaling["feature_weight"], float(vq) * scaling["label_weight"]
    return dict(common_J=rx + rq, normalized_feature_variance=rx, weighted_normalized_label_variance=rq)


def run_ablation(previous, output_dir, datasets=None, seeds=tuple(range(100, 110)),
                 data_dir="/content/data/", device="cuda"):
    previous, output_dir = Path(previous), Path(output_dir)
    runs = json.loads((previous / "runs.json").read_text())
    datasets = list(runs) if datasets is None else list(datasets)
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Student seeds must be nonempty and unique")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    files = ("normalized_variance_ablation.py", "moment_lloyd.py", "moment_seeding.py", "evaluation.py", "models.py", "data.py")
    code = hashlib.sha256(b"".join((Path(__file__).parent / name).read_bytes() for name in files)).hexdigest()
    all_students, diagnostics = [], []
    for dataset in datasets:
        source = Path(runs[dataset])
        config = json.loads((source / "config.json").read_text())
        chosen = json.loads((source / "selected.json").read_text())
        graph, train, validation, testing, computed = _prepare_dataset(dataset, data_dir, device)
        del computed
        splits = dict(train=(graph, train), val=validation, test=testing)
        if _data_digest(splits) != config["data_digest"]:
            raise ValueError(f"{dataset}: original data differ from the source run")
        masks = splits if validation[0] is not graph else {name: pair[1] for name, pair in splits.items()}
        h = torch.load(source / "features.pt", map_location=device, weights_only=True)
        teacher = torch.load(source / "teachers" / f"{_fingerprint(dict(gamma=chosen['gamma']))}.pt",
                             map_location=device, weights_only=True)
        q = (teacher["logits"] / chosen["T"]).softmax(1)
        del teacher
        seed = config["condensation_seeds"][0]
        if config["condensation_seeds"] != [seed]:
            raise ValueError("This ablation expects one deterministic source partition")
        original = torch.load(source / f"candidate_{int(chosen['candidate']):04d}" / f"seed_{seed}" / "partition.pt",
                              map_location="cpu", weights_only=False)
        variants = {
            "main": {},
            "no_label_distance": dict(moment_weight=0.0),
            "initialization_only": dict(max_sweeps=0),
            "feature_only_initialization": dict(seeding="feature_var"),
            "without_label_normalization": dict(mode="variance_sum"),
            "kmeans_plus_plus": dict(seeding="bound", initialization_steps=300,
                                     require_initialization_convergence=True),
            "mass_CE": {},
        }
        manifest = dict(source=str(source.resolve()), source_config=config, selected=chosen,
                        seeds=list(seeds), source_digest=code, variants=variants)
        root = output_dir / dataset / _fingerprint(manifest)
        root.mkdir(parents=True, exist_ok=True)
        save_json(manifest, root / "config.json")
        for name, changes in tqdm(variants.items(), desc=f"{dataset}: ablations"):
            folder = root / name
            folder.mkdir(exist_ok=True)
            path = folder / "partition.pt"
            if path.exists():
                partition = torch.load(path, map_location="cpu", weights_only=False)
            elif name in ("main", "mass_CE"):
                partition = original
                save_state(partition, path)
            else:
                options = dict(mode="normalized_variance", seeding="bound_var", seed=seed,
                               moment_weight=chosen["lambda"], max_sweeps=config["max_sweeps"],
                               block_size=config["block_size"])
                options.update(changes)
                partition = moment_lloyd_partition(h, q, config["nodes"], **options)
                save_state(partition, path)
            diagnostic = dict(dataset=dataset, variant=name, gamma=chosen["gamma"], T=chosen["T"],
                              alpha=chosen["lambda"], optimized_J=partition["J"],
                              sweeps=partition["sweeps"], converged=partition["converged"],
                              status=partition["status"], partition_seconds=partition["seconds"],
                              **common_objective(h, q, partition["assignment"], config["nodes"], chosen["lambda"]))
            diagnostics.append(diagnostic)
            cx, cy = partition["x"].to(device), partition["y"].to(device)
            mass = partition["counts"].to(device) if name == "mass_CE" else torch.ones(len(cx), device=device)
            for student_seed in seeds:
                score = fit_gcn_diagnostic(cx, cy, mass, graph, None, masks, student_seed,
                                           folder=folder / "students", **config["student"])
                all_students.append(dict(dataset=dataset, variant=name, student_seed=student_seed, **score))
            write_table(pd.DataFrame(all_students), output_dir / "students.csv")
            write_table(pd.DataFrame(diagnostics), output_dir / "partitions.csv")
        del graph, train, validation, testing, splits, masks, h, q, original, partition, cx, cy, mass
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    students = pd.DataFrame(all_students)
    baseline = students[students.variant == "main"][["dataset", "student_seed", "val_acc", "test_acc"]]
    paired = students.merge(baseline, on=["dataset", "student_seed"], suffixes=("", "_main"), validate="many_to_one")
    paired["val_delta"] = paired.val_acc - paired.val_acc_main
    paired["test_delta"] = paired.test_acc - paired.test_acc_main
    summary = paired.groupby(["dataset", "variant"], as_index=False).agg(
        val_mean=("val_acc", "mean"), val_std=("val_acc", "std"),
        test_mean=("test_acc", "mean"), test_std=("test_acc", "std"),
        val_delta=("val_delta", "mean"), test_delta=("test_delta", "mean"),
        paired_test_delta_std=("test_delta", "std"),
    ).merge(pd.DataFrame(diagnostics), on=["dataset", "variant"], validate="one_to_one")
    write_table(paired, output_dir / "paired_students.csv")
    write_table(summary, output_dir / "summary.csv")
    return summary, paired
