import hashlib
import json
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm

from src.data import _prepare_dataset
from src.distance_finetune import evaluation_splits
from src.evaluation import fit_gcn_diagnostic
from src.io import _fingerprint, save_json, save_state
from src.propagation_kmeans import nearest_train
from src.variance_moment_sweep import _data_digest


def smoothing_position(distance, train):
    distance = distance.clamp_min(0).clone()
    distance[train] = 0
    positive = distance[(~train) & (distance > 0)]
    scale = positive.median() if len(positive) else distance.new_tensor(1.)
    position = distance / (distance + scale)
    return position, float(scale)


def temperature_labels(logits, position, base, eta):
    if base <= 0 or eta < 0:
        raise ValueError("Base temperature must be positive and eta nonnegative")
    temperature = base * (1 + eta * position)
    return (logits / temperature[:, None]).softmax(1), temperature


def run_adaptive_temperature(source, output_dir, etas=(0., .25, .5, 1., 2., 4.),
                              data_dir="/content/data/", device="cuda"):
    if not etas or any(e < 0 for e in etas):
        raise ValueError("Use a nonempty nonnegative eta grid")
    etas = sorted(set([0.] + list(etas)))
    source = Path(source)
    config = json.loads((source / "config.json").read_text())
    selected = json.loads((source / "selected.json").read_text())
    if config.get("partition_method") != "kmeans":
        raise ValueError("Use the feature-only Var-Part k-means stage-one run")
    graph, train, validation, testing, computed = _prepare_dataset(config["dataset"], data_dir, device)
    del computed
    if _data_digest(dict(train=(graph, train), val=validation, test=testing)) != config["data_digest"]:
        raise ValueError("Data differ from the saved stage-one run")
    masks = evaluation_splits(graph, train, validation, testing)
    h = torch.load(source / "features.pt", map_location=device, weights_only=True).double()
    logits = torch.load(source / "teacher.pt", map_location=device, weights_only=False)["logits"].double()
    partition = torch.load(source / f"candidate_{int(selected['candidate']):04d}" / "partition.pt",
                           map_location=device, weights_only=False)
    assignment, counts, x = partition["assignment"], partition["counts"], partition["x"]
    bases = sorted(set([1., float(selected["T"])]))
    fingerprint = dict(source=str(source.resolve()), source_config=config, selected=selected,
                       etas=etas, bases=bases, code=hashlib.sha256(b"".join(
                           (Path(__file__).parent / name).read_bytes() for name in
                           ("adaptive_temperature.py", "propagation_kmeans.py", "evaluation.py", "models.py"))).hexdigest())
    root = Path(output_dir) / _fingerprint(fingerprint)
    root.mkdir(parents=True, exist_ok=True)
    save_json(fingerprint, root / "config.json")
    distance_path = root / "distance.pt"
    if distance_path.exists():
        distance = torch.load(distance_path, map_location=device, weights_only=True)
    else:
        z = h - h.mean(0)
        z /= z.square().sum(1).mean().sqrt().clamp_min(1e-30)
        squared, _ = nearest_train(z, train, k=1)
        distance = squared[:, 0].to(device).double().clamp_min(0).sqrt()
        save_state(distance.cpu(), distance_path)
    position, scale = smoothing_position(distance, train)
    save_json(dict(distance_scale=scale, train_nodes=int(train.sum()), nodes=len(train),
                   feature="Centered global RMS S2X", fixed_teacher=True), root / "distance_info.json")
    rows = []
    for candidate, (base, eta) in enumerate(tqdm([(b, e) for b in bases for e in etas], desc="Adaptive temperature")):
        folder = root / f"candidate_{candidate:04d}"
        folder.mkdir(exist_ok=True)
        q, temperature = temperature_labels(logits, position, base, eta)
        labels = q.new_zeros(len(x), q.shape[1]).index_add_(0, assignment, q) / counts[:, None]
        labels = labels.float()
        save_state(dict(labels=labels.cpu()), folder / "labels.pt")
        scores = pd.DataFrame([fit_gcn_diagnostic(x, labels, torch.ones(len(x), device=device), graph, None,
                  {k: masks[k] for k in ("train", "val")}, seed, folder=folder / "search", **config["settings"])
                  for seed in config["search_seeds"]])
        scores.assign(student_seed=config["search_seeds"]).to_csv(folder / "search_students.csv", index=False)
        rows.append(dict(candidate=candidate, base_T=base, eta=eta, search_val=scores.val_acc.mean(),
                         search_val_std=scores.val_acc.std(), mean_T=float(temperature.mean()),
                         max_T=float(temperature.max()), node_entropy=float(-(q * q.clamp_min(1e-30).log()).sum(1).mean())))
        pd.DataFrame(rows).to_csv(root / "search.csv", index=False)
    search = pd.DataFrame(rows)
    summaries = []
    for base in bases:
        group = search[search.base_T == base]
        baseline = group[group.eta == 0].iloc[0]
        winner = group.sort_values(["search_val", "eta"], ascending=[False, True]).iloc[0]
        for phase, row in (("global", baseline), ("selected", winner)):
            folder = root / f"candidate_{int(row.candidate):04d}"
            labels = torch.load(folder / "labels.pt", map_location=device, weights_only=True)["labels"]
            scores = pd.DataFrame([fit_gcn_diagnostic(x, labels, torch.ones(len(x), device=device), graph, None,
                      masks, seed, folder=folder / "final", **config["settings"]) for seed in config["final_seeds"]])
            scores.assign(student_seed=config["final_seeds"]).to_csv(root / f"T_{base}_{phase}_students.csv", index=False)
            summaries.append(dict(dataset=config["dataset"], ratio=config["ratio"], phase=phase,
                                  **row.to_dict(), final_val=scores.val_acc.mean(), test_mean=scores.test_acc.mean(),
                                  test_std=scores.test_acc.std(), output_dir=str(root)))
            pd.DataFrame(summaries).to_csv(root / "summary.csv", index=False)
    return pd.DataFrame(summaries), search, root
