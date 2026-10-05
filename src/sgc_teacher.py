import math

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from src.head import augment, fit_head
from src.io import save_json, save_state, write_table


def rms_features(h, offset=None, scale=None):
    h = h.double()
    offset = h.mean(0) if offset is None else offset
    scale = (h - offset).square().sum(1).mean().sqrt().clamp_min(1e-30) if scale is None else scale
    return (h - offset) / scale, offset, scale


def select_sgc_teacher(h, graph, train, validation, testing, penalties, folder):
    if not penalties or any(not math.isfinite(float(p)) or p <= 0 for p in penalties):
        raise ValueError("SGC penalties must be finite and positive")
    path = folder / "teacher.pt"
    if path.exists():
        result = torch.load(path, map_location="cpu", weights_only=False)
        if result.get("teacher_type") != "sgc":
            raise ValueError("Expected an SGC teacher")
        return result
    z, offset, scale = rms_features(h)
    classes = int(graph["y"].max()) + 1
    targets = F.one_hot(graph["y"][train], classes).to(z)
    x = z[train]
    mass = z.new_full((len(x),), 1 / len(x))

    def metrics(theta, pair):
        other, mask = pair
        features = z if other is graph else rms_features(other["sgc_features"], offset, scale)[0]
        labels = other["y"]
        if mask is not None:
            features, labels = features[mask], labels[mask]
        prediction = augment(features) @ theta.T
        return dict(acc=100 * float((prediction.argmax(1) == labels).double().mean()),
                    ce=float(F.cross_entropy(prediction, labels)))

    rows = []
    for index, penalty in enumerate(tqdm(penalties, desc="SGC teacher penalty validation")):
        candidate = folder / "teacher_search" / f"candidate_{index:03d}"
        candidate.mkdir(parents=True, exist_ok=True)
        fitted_path = candidate / "head.pt"
        if fitted_path.exists():
            fitted = torch.load(fitted_path, map_location=z.device, weights_only=False)
        else:
            fitted = fit_head(x, targets, mass, penalty, max_iter=2000, grad_tol=1e-6)
            if not fitted["converged"]:
                fitted = fit_head(x, targets, mass, penalty, max_iter=2000, grad_tol=1e-6,
                                  initial_theta=fitted["theta"], tolerance_change=1e-18)
            if not fitted["converged"]:
                raise RuntimeError(f"SGC penalty {penalty}: gradient tolerance not reached ({fitted['grad_max']})")
            save_state(fitted, fitted_path)
        with torch.no_grad():
            val = metrics(fitted["theta"], validation)
        rows.append(dict(candidate=index, penalty=penalty, val_acc=val["acc"], val_ce=val["ce"],
                         grad_max=fitted["grad_max"], iterations=fitted["iterations"]))
        write_table(pd.DataFrame(rows), folder / "teacher_grid.csv")
    selected = pd.DataFrame(rows).sort_values(
        ["val_acc", "val_ce", "candidate"], ascending=[False, True, True]).iloc[0].to_dict()
    selected["candidate"] = int(selected["candidate"])
    fitted = torch.load(folder / "teacher_search" / f"candidate_{selected['candidate']:03d}" / "head.pt",
                        map_location=z.device, weights_only=False)
    with torch.no_grad():
        theta = fitted["theta"]
        logits = augment(z) @ theta.T
        test = metrics(theta, testing)
        training = metrics(theta, (graph, train))
    result = dict(teacher_type="sgc", logits=logits, theta=theta, offset=offset, scale=scale,
                  penalty=selected["penalty"], val_acc=selected["val_acc"], val_ce=selected["val_ce"],
                  test_acc=test["acc"], test_ce=test["ce"], train_acc=training["acc"],
                  train_ce=training["ce"], epoch=None)
    save_state(result, path)
    save_json(selected, folder / "selected_teacher.json")
    save_json({k: v for k, v in result.items() if not torch.is_tensor(v)}, folder / "teacher_metrics.json")
    return result
