from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from src.data import _prepare_dataset
from src.head import augment, fit_head
from src.io import _fingerprint, array_digest, save_json, write_table
from src.teacher import teacher_logits
from src.transforms import fit_transform


def select_cora_teacher(
    output_dir,
    gammas=(1e-5, 1e-4, 1e-3, 1e-2, 1e-1),
    temperatures=(0.1, 0.3, 0.5, 1.0, 2.0, 3.0),
    penalty=0.001,
    basis=3000,
    seed=0,
    data_dir="/content/data/",
    device="cuda",
):
    if not gammas or not temperatures or any(
        not np.isfinite(v) or v <= 0 for v in (*gammas, *temperatures, penalty)
    ):
        raise ValueError("Gamma, temperature and penalty must be positive and finite")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    graph, train, validation, _, h = _prepare_dataset("cora", data_dir, device)
    mask = validation[1]
    config = dict(
        gammas=list(gammas), temperatures=list(temperatures), penalty=penalty,
        basis=basis, seed=seed, version=1,
        data_digest=array_digest(
            h.cpu().numpy(), train.cpu().numpy(), mask.cpu().numpy(),
            graph["y"][train].cpu().numpy(), graph["y"][mask].cpu().numpy(),
        ),
    )
    root = Path(output_dir) / _fingerprint(config)
    root.mkdir(parents=True, exist_ok=True)
    save_json(config, root / "config.json")
    teachers = teacher_logits(
        h, graph, train, validation, "relu", list(gammas), basis, seed, root, return_all=True
    )
    z, _ = fit_transform(h.double(), kind="rms")
    mass = z.new_full((len(z),), 1 / len(z))
    x_val, y_val = augment(z[mask]), graph["y"][mask]
    path = root / "selection_grid.csv"
    rows = pd.read_csv(path).to_dict("records") if path.exists() else []
    for gamma in tqdm(gammas, desc="Student-based teacher selection"):
        logits = teachers[gamma].to(device)
        for temperature in temperatures:
            if any(np.isclose(r["gamma"], gamma, rtol=1e-12, atol=0)
                   and np.isclose(r["T"], temperature, rtol=1e-12, atol=0) for r in rows):
                continue
            q = (logits / temperature).softmax(1)
            fitted = fit_head(z, q, mass, penalty)
            if not fitted["converged"]:
                fitted = fit_head(z, q, mass, penalty, max_iter=5000, initial_theta=fitted["theta"])
            if not fitted["converged"]:
                raise RuntimeError(f"Selection head did not converge: gamma={gamma}, T={temperature}")
            scores = x_val @ fitted["theta"].T
            rows.append(dict(
                gamma=gamma, T=temperature,
                student_val_ce=float(F.cross_entropy(scores, y_val)),
                student_val=100 * float((scores.argmax(1) == y_val).double().mean()),
                teacher_val_ce=float(F.cross_entropy(logits[mask] / temperature, y_val)),
                teacher_val=100 * float((logits[mask].argmax(1) == y_val).double().mean()),
                head_grad_max=fitted["grad_max"],
            ))
            write_table(pd.DataFrame(rows), path)
    table = pd.DataFrame(rows).sort_values(["student_val_ce", "gamma", "T"])
    selected = table.iloc[0].to_dict()
    save_json(selected, root / "selected.json")
    return selected, table, root
