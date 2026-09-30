"""Short validation-only Cora/Citeseer screens, resumable in small rounds."""
import json
import time
from pathlib import Path

import pandas as pd
import torch

from src.data import BUDGET, _prepare_dataset
from src.evaluation import fit_gcn_diagnostic
from src.initialization import feature_kmeans
from src.io import _fingerprint, array_digest, save_json, save_state, write_table
from src.soft_ce_partition import optimize_ce_assignment
from src.sweep_utils import representative
from src.teacher import teacher_logits
from src.transforms import FeatureTransform, fit_transform


def run_screen(dataset, ratio, output_dir, candidates, steps=50,
               student_seeds=(0, 1), condensation_seed=0, dropout=0.9,
               epochs=500, data_dir="data", device="cuda", checkpoints=None, input_scale=1.0,
               citation_features="default"):
    if dataset not in ("cora", "citeseer") or (dataset, ratio) not in BUDGET:
        raise ValueError("Use a configured Cora/Citeseer budget")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    graph, train, validation, testing, h = _prepare_dataset(dataset, data_dir, device, citation_features)
    config = dict(dataset=dataset, ratio=ratio, data_digest=array_digest(
        graph["x"].cpu().numpy(), graph["y"].cpu().numpy(),
        graph["adj"].crow_indices().cpu().numpy(), graph["adj"].col_indices().cpu().numpy(),
        graph["adj"].values().cpu().numpy(), train.cpu().numpy(), validation[1].cpu().numpy(),
        testing[1].cpu().numpy()), teacher_basis=3000, teacher_seed=0,
        teacher_gammas=[1e-5, 1e-4, 1e-3, 0.01], kernel="relu", mixing=0.05,
        inner_loss="mass_ce", student_loss="uniform_ce", version=1)
    if citation_features != "default":
        config["citation_features"] = citation_features
    root = Path(output_dir) / dataset / f"ratio_{ratio}" / _fingerprint(config)
    root.mkdir(parents=True, exist_ok=True)
    save_json(config, root / "config.json")
    logits, gamma = teacher_logits(h, graph, train, validation, "relu", config["teacher_gammas"],
                                  3000, 0, root)
    save_json(dict(gamma=gamma, val=100 * float((logits[validation[1]].argmax(1)
                   == graph["y"][validation[1]]).double().mean())), root / "teacher_selected.json")
    inputs_path = root / f"inputs_{condensation_seed}.pt"
    if inputs_path.exists():
        saved = torch.load(inputs_path, map_location=device, weights_only=False)
        z, assignment = saved["z"], saved["assignment"]
        transform = FeatureTransform(**saved["transform"])
    else:
        z, transform = fit_transform(h.double())
        assignment = feature_kmeans(h.cpu(), BUDGET[(dataset, ratio)], condensation_seed).to(device)
        save_state(dict(z=z, assignment=assignment, transform=vars(transform)), inputs_path)
    settings = dict(epochs=epochs, eval_every=10, hidden=256, dropout=dropout,
                    lr=0.01, weight_decay=0.0005)
    recipe = dict(**settings, input_scale=input_scale)
    recipe_key = _fingerprint(recipe)
    save_json(recipe, root / f"student_recipe_{recipe_key}.json")
    screen_protocol = dict(candidates=candidates, steps=steps, checkpoints=checkpoints,
                           condensation_seed=condensation_seed, student_seeds=list(student_seeds), recipe=recipe)
    screen_key = _fingerprint(screen_protocol)
    save_json(screen_protocol, root / f"screen_protocol_{screen_key}.json")
    evaluation_graph = dict(graph, x=graph["x"] * input_scale)
    checks = sorted({0, steps, *(checkpoints or [])})
    records = []
    feature_assignment = assignment
    for candidate in candidates:
        identity = {"method": "low_rank", "width": 0, "lr": 0.01, **candidate}
        folder = root / _fingerprint(identity) / f"condensation_{condensation_seed}"
        folder.mkdir(parents=True, exist_ok=True)
        save_json(identity, folder.parent / "candidate.json")
        resume = folder / "resume.pt"
        state = torch.load(resume, map_location="cpu", weights_only=False) if resume.exists() else None
        q = (logits / identity["T"]).softmax(1).double()
        assignment = feature_assignment
        if identity.get("initialization", "feature") != "feature":
            from src.partition_initialization import teacher_aware_kmeans, teacher_balanced_kmeans
            init_config = dict(mode=identity["initialization"], alpha=identity.get("alpha", 1.0),
                               T=identity["T"], seed=condensation_seed)
            init_path = root / f"assignment_{_fingerprint(init_config)}.pt"
            if not init_path.exists():
                initializer = (teacher_aware_kmeans if init_config["mode"] == "teacher_joint"
                               else teacher_balanced_kmeans if init_config["mode"] == "teacher_balanced"
                               else None)
                if initializer is None:
                    raise ValueError("Unknown initializer")
                save_state(initializer(h, q, BUDGET[(dataset, ratio)], condensation_seed,
                                       alpha=init_config["alpha"]), init_path)
            assignment = torch.load(init_path, map_location=device, weights_only=False)
        started = time.monotonic()
        if state is None or state["step"] < steps:
            print("CANDIDATE", dataset, ratio, identity, "budget", steps, flush=True)
            if identity["method"] == "nystrom":
                from src.nystrom_ce import NystromMap, cache_features, optimize
                if identity.get("inner_loss_weighting", "mass") != "mass":
                    raise ValueError("Nyström screen currently uses mass inner CE")
                feature_map = NystromMap.fit(h.double(), basis=3000, seed=0)
                phi = cache_features(h, feature_map, root / "nystrom_phi.npy")
                optimize(h.double(), q, assignment, feature_map, phi, folder, steps,
                         penalty=identity["penalty"], lr=identity["lr"], rank=identity["rank"],
                         seed=condensation_seed, checkpoint_every=25)
                del feature_map, phi
            elif identity["method"] == "coarsening":
                from src.coarsening_ce import optimize_coarsening_ce
                optimize_coarsening_ce(graph["x"], q, assignment, graph["adj"], h=h, transform=transform,
                    penalty=identity["penalty"], steps=steps, lr=identity["lr"], rank=identity["rank"],
                    seed=condensation_seed, mass_scaling=identity.get("mass_scaling", False),
                    inner_loss_weighting=identity.get("inner_loss_weighting", "mass"),
                    folder=folder, checkpoint_steps=checks, resume_state=state)
            else:
                optimize_ce_assignment(z, q, assignment, penalty=identity["penalty"], steps=steps,
                    lr=identity["lr"], assignment_rank=identity["rank"], factor_seed=condensation_seed,
                    assignment_input="features" if identity["method"] == "mlp" else "node",
                    assignment_encoder="mlp" if identity["method"] == "mlp" else "linear",
                    encoder_hidden=identity["width"] if identity["method"] == "mlp" else 64,
                    solver_mode="exact", inner_method="newton_first", implicit_warm_start=True,
                    inner_loss_weighting=identity.get("inner_loss_weighting", "mass"), inner_max_iter=2000, inner_tol=1e-7,
                    cg_max_iter=512, cg_rtol=1e-6, cache_assignment=False,
                    folder=folder, checkpoint_steps=checks, resume_state=state,
                    save_resume=True, save_assignment=False)
        for step in checks:
            snapshot_path = folder / "checkpoints" / f"step_{step:06d}.pt"
            if identity["method"] == "nystrom":
                snapshot_path = folder / f"step_{step:06d}.pt"
            if not snapshot_path.exists():
                continue
            snapshot = torch.load(snapshot_path, map_location=device, weights_only=False)
            if identity["method"] == "coarsening":
                from src.coarsening_ce import gcn_inputs
                x, y, mass, training_adj = gcn_inputs(snapshot, device)
            elif identity["method"] == "nystrom":
                from src.moments import decode_moments
                x, y, mass = decode_moments(snapshot["moments"], h.shape[1])
                x, y, training_adj = x.float(), y.float(), None
            else:
                x, y, mass = representative(snapshot["moments"], transform, z.shape[1], device)
                training_adj = None
            for seed in student_seeds:
                result = fit_gcn_diagnostic(x * input_scale, y, torch.full_like(mass, 1 / len(mass)), evaluation_graph, q,
                    dict(train=train, val=validation[1]), seed, folder=folder / "validation"
                    / f"step_{step}_{recipe_key}", training_adjacency=training_adj, **settings)
                records.append(dict(**identity, condensation_seed=condensation_seed, step=step,
                                    **result, dropout=dropout, input_scale=input_scale, epochs=epochs,
                                    candidate_path=str(folder.parent.resolve())))
        print("CANDIDATE_DONE", dataset, identity, "seconds", round(time.monotonic() - started, 2), flush=True)
        write_table(pd.DataFrame(records), root / f"screen_{condensation_seed}_{steps}_{recipe_key}_{screen_key}.csv")
    frame = pd.DataFrame(records)
    keys = ["method", "width", "lr", "T", "rank", "penalty", "step", "candidate_path"]
    keys += [key for key in ("initialization", "alpha", "inner_loss_weighting", "mass_scaling") if key in frame.columns]
    keys += ["dropout", "input_scale", "epochs"]
    summary = frame.groupby(keys, dropna=False).val_acc.agg(["mean", "std", "count"]).reset_index()
    summary = summary.sort_values("mean", ascending=False)
    write_table(summary, root / f"ranking_{condensation_seed}_{steps}_{recipe_key}_{screen_key}.csv")
    print("VALIDATION_RANKING", dataset, "\n", summary.head(8).to_string(index=False), flush=True)
    return summary, root


def selected_test(root, choice, condensation_seeds=(0, 1, 2), student_seeds=(100, 101, 102, 103, 104),
                  dropout=0.9, epochs=1000, data_dir="data", device="cuda", input_scale=1.0):
    """Test only a fixed configuration and checkpoint, with fresh student seeds."""
    root = Path(root)
    config = json.loads((root / "config.json").read_text())
    candidate = {key: choice[key] for key in ("method", "width", "lr", "T", "rank", "penalty")}
    candidate["rank"], candidate["width"] = int(candidate["rank"]), int(candidate["width"])
    for key in ("lr", "T", "penalty"):
        candidate[key] = float(candidate[key])
    candidate.update({key: choice[key] for key in ("initialization", "alpha", "inner_loss_weighting", "mass_scaling") if key in choice
                      and pd.notna(choice[key])})
    step = int(choice["step"])
    selection = dict(candidate=candidate, step=step, selection="validation_only",
                     student=dict(dropout=dropout, input_scale=input_scale, epochs=epochs),
                     condensation_seeds=list(condensation_seeds), student_seeds=list(student_seeds))
    selection_key = _fingerprint(selection)
    save_json(selection, root / "selected.json")
    save_json(selection, root / f"selected_{selection_key}.json")
    graph, train, validation, testing, h = _prepare_dataset(config["dataset"], data_dir, device,
                                                          config.get("citation_features", "default"))
    logits = torch.load(root / "teacher.pt", map_location=device, weights_only=False)["logits"]
    q = (logits / candidate["T"]).softmax(1).double()
    records = []
    settings = dict(epochs=epochs, eval_every=10, hidden=256, dropout=dropout, lr=0.01, weight_decay=0.0005)
    recipe_key = _fingerprint(dict(**settings, input_scale=input_scale))
    evaluation_graph = dict(graph, x=graph["x"] * input_scale)
    for cond_seed in condensation_seeds:
        # This also produces the needed condensate if it is absent.
        run_screen(config["dataset"], config["ratio"], root.parents[2], [candidate], steps=step,
                   student_seeds=(0,), condensation_seed=cond_seed, dropout=dropout, epochs=epochs,
                   data_dir=data_dir, device=device, input_scale=input_scale,
                   citation_features=config.get("citation_features", "default"))
        folder = root / _fingerprint(candidate) / f"condensation_{cond_seed}"
        inputs = torch.load(root / f"inputs_{cond_seed}.pt", map_location=device, weights_only=False)
        transform = FeatureTransform(**inputs["transform"])
        snapshot_path = (folder / f"step_{step:06d}.pt" if candidate["method"] == "nystrom"
                         else folder / "checkpoints" / f"step_{step:06d}.pt")
        snapshot = torch.load(snapshot_path, map_location=device, weights_only=False)
        if candidate["method"] == "coarsening":
            from src.coarsening_ce import gcn_inputs
            x, y, mass, training_adj = gcn_inputs(snapshot, device)
        elif candidate["method"] == "nystrom":
            from src.moments import decode_moments
            x, y, mass = decode_moments(snapshot["moments"], h.shape[1])
            x, y, training_adj = x.float(), y.float(), None
        else:
            x, y, mass = representative(snapshot["moments"], transform, inputs["z"].shape[1], device)
            training_adj = None
        for seed in student_seeds:
            result = fit_gcn_diagnostic(x * input_scale, y, torch.full_like(mass, 1 / len(mass)), evaluation_graph, q,
                dict(train=train, val=validation[1], test=testing[1]), seed,
                folder=folder / "final" / f"step_{step}_{recipe_key}", training_adjacency=training_adj, **settings)
            records.append(dict(condensation_seed=cond_seed, **result))
            write_table(pd.DataFrame(records), root / "final.csv")
            write_table(pd.DataFrame(records), root / f"final_{selection_key}.csv")
    return pd.DataFrame(records)
