"""Short validation-only Cora/Citeseer screens, resumable in small rounds."""
import hashlib
import inspect
import json
import math
import time
from collections.abc import Mapping
from numbers import Real
from pathlib import Path

import pandas as pd
import torch

from src.data import BUDGET, _prepare_dataset, normalize_adj_sparse
from src.evaluation import fit_gcn_diagnostic
from src.initialization import feature_kmeans
from src.io import _fingerprint, array_digest, save_json, save_state, write_table
from src.shared_features import get_shared_h, get_shared_map
from src.soft_ce_partition import optimize_ce_assignment
from src.sweep_utils import representative
from src.target_refinement import training_refined_targets
from src.teacher import teacher_logits
from src.transforms import FeatureTransform, fit_transform


def dataset_digest(graph, train, validation, testing):
    return array_digest(graph["x"].cpu().numpy(), graph["y"].cpu().numpy(),
                        graph["adj"].crow_indices().cpu().numpy(), graph["adj"].col_indices().cpu().numpy(),
                        graph["adj"].values().cpu().numpy(), train.cpu().numpy(),
                        validation[1].cpu().numpy(), testing[1].cpu().numpy())


def fixed_propagated_features(h, config, root):
    source = _fingerprint(dict(data_digest=config["data_digest"], steps=2,
        implementation=hashlib.sha256((inspect.getsource(normalize_adj_sparse) +
                                       inspect.getsource(_prepare_dataset)).encode()).hexdigest(),
        dtype=str(h.dtype), torch=str(torch.__version__)))
    return get_shared_h(h, root / "propagated_H.pt", source)


def _assignment_mass_mode(identity):
    if "mass_mode" in identity and identity["method"] not in ("low_rank", "mlp"):
        raise ValueError("mass_mode is supported only by low_rank and mlp assignments")
    mode = identity.get("mass_mode", "free")
    if mode not in ("free", "uniform"):
        raise ValueError("mass_mode must be free or uniform")
    return mode


def _nystrom_inner_weighting(identity):
    """Validate the surrogate loss without changing candidate identities."""
    weighting = identity.get("inner_loss_weighting", "mass")
    if weighting not in ("mass", "uniform"):
        raise ValueError("Nyström inner_loss_weighting must be mass or uniform")
    return weighting


def _candidate_background_mixing(candidate):
    """Canonicalize a per-candidate prior without changing legacy defaults."""
    if not isinstance(candidate, Mapping):
        raise ValueError("Candidates must be mappings")
    candidate = dict(candidate)
    mixing = candidate.get("mixing", 0.05)
    if (isinstance(mixing, bool) or not isinstance(mixing, Real)
            or not math.isfinite(mixing) or not 0 < mixing < 1):
        raise ValueError("mixing must be finite and lie strictly in (0, 1)")
    mixing = float(mixing)
    if mixing == 0.05:
        candidate.pop("mixing", None)
    else:
        candidate["mixing"] = mixing
    return candidate, mixing


def _candidate_temperature(candidate):
    """Keep fixed-temperature identities unchanged and validate the new mode."""
    candidate = dict(candidate)
    for key in ("T", "lr"):
        if key in candidate:
            value = candidate[key]
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not math.isfinite(value) or value <= 0):
                raise ValueError(f"{key} must be a positive finite number")
    enabled = candidate.get("learn_temperature", False)
    if not pd.api.types.is_bool(enabled):
        raise ValueError("learn_temperature must be a boolean")
    enabled = bool(enabled)
    learning_rate = candidate.get("temperature_lr", 0.003)
    if (isinstance(learning_rate, bool) or not isinstance(learning_rate, Real)
            or not math.isfinite(learning_rate) or learning_rate <= 0):
        raise ValueError("temperature_lr must be a positive finite number")
    if not enabled:
        if learning_rate != 0.003:
            raise ValueError("temperature_lr requires learn_temperature=True")
        candidate.pop("learn_temperature", None)
        candidate.pop("temperature_lr", None)
        return candidate
    if "T" not in candidate:
        raise ValueError("Learnable temperature requires an initial T")
    if (candidate.get("method", "low_rank") != "low_rank"
            or candidate.get("mass_mode", "free") != "free"
            or candidate.get("node_weighting", False) is not False
            or candidate.get("solver_mode", "exact") != "exact"
            or candidate.get("assignment_input", "node") != "node"
            or candidate.get("assignment_encoder", "linear") != "linear"
            or candidate.get("feature_control", "joint") != "joint"):
        raise ValueError("Learnable temperature requires exact, unweighted, free-mass low_rank assignments")
    target_mix = candidate.get("train_target_mix", 0.0)
    if (isinstance(target_mix, bool) or not isinstance(target_mix, Real)
            or not math.isfinite(target_mix) or target_mix != 0):
        raise ValueError("Learnable temperature requires train_target_mix=0")
    candidate["learn_temperature"] = True
    candidate["temperature_lr"] = float(learning_rate)
    return candidate


def _temperature_diagnostics(snapshot, state, step):
    temperature, value = snapshot.get("temperature"), snapshot.get("teacher_ce")
    if (snapshot.get("J_exact") is not True
            or isinstance(temperature, bool) or not isinstance(temperature, Real)
            or not math.isfinite(temperature) or temperature <= 0
            or isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value)):
        raise ValueError("Invalid learnable-temperature checkpoint diagnostics")
    gradients = []
    for row in state.get("history", []):
        if row["step"] <= step and "log_temperature_gradient" in row:
            gradient = row["log_temperature_gradient"]
            if (isinstance(gradient, bool) or not isinstance(gradient, Real)
                    or not math.isfinite(gradient)):
                raise ValueError("Nonfinite temperature gradient in checkpoint history")
            gradients.append(row)
    latest = max(gradients, key=lambda row: row["step"]) if gradients else None
    return dict(learned_temperature=float(temperature), surrogate_ce=float(value), surrogate_J_exact=True,
                log_temperature_gradient=None if latest is None else float(latest["log_temperature_gradient"]),
                temperature_gradient_step=None if latest is None else int(latest["step"]))


def _student_settings(dropout, epochs, overrides):
    settings = dict(epochs=epochs, eval_every=10, hidden=256, dropout=dropout,
                    lr=0.01, weight_decay=0.0005)
    if overrides is None:
        return settings
    allowed = {"eval_every", "hidden", "lr", "weight_decay", "lr_schedule", "initialization"}
    if not isinstance(overrides, dict) or set(overrides) - allowed:
        raise ValueError("Unknown student setting; use epochs/dropout arguments for those fields")
    settings.update(overrides)
    # Explicit defaults also reuse the historical recipe and evaluation folders.
    for key, default in (("lr_schedule", "half_reset"), ("initialization", "pyg")):
        if settings.get(key) == default:
            del settings[key]
    return settings


def run_screen(dataset, ratio, output_dir, candidates, steps=50,
               student_seeds=(0, 1), condensation_seed=0, dropout=0.9,
               epochs=500, data_dir="data", device="cuda", checkpoints=None, input_scale=1.0,
               citation_features="default", stop=lambda: False, student_settings=None,
               report_routes=False):
    if dataset not in ("cora", "citeseer") or (dataset, ratio) not in BUDGET:
        raise ValueError("Use a configured Cora/Citeseer budget")
    # Check every requested prior before loading a dataset or fitting a teacher.
    # Explicit .05 removes the field so legacy identities/protocols are retained.
    candidates = [_candidate_temperature(_candidate_background_mixing(candidate)[0])
                  for candidate in candidates]
    for candidate in candidates:
        if candidate.get("method", "low_rank") == "nystrom":
            _nystrom_inner_weighting(candidate)
    if not isinstance(report_routes, bool):
        raise ValueError("report_routes must be a boolean")
    settings = _student_settings(dropout, epochs, student_settings)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    graph, train, validation, testing, h = _prepare_dataset(dataset, data_dir, device, citation_features)
    config = dict(dataset=dataset, ratio=ratio, data_digest=dataset_digest(graph, train, validation, testing),
        teacher_basis=3000, teacher_seed=0,
        teacher_gammas=[1e-5, 1e-4, 1e-3, 0.01], kernel="relu", mixing=0.05,
        inner_loss="mass_ce", student_loss="uniform_ce", version=1)
    if citation_features != "default":
        config["citation_features"] = citation_features
    root = Path(output_dir) / dataset / f"ratio_{ratio}" / _fingerprint(config)
    root.mkdir(parents=True, exist_ok=True)
    save_json(config, root / "config.json")
    h = fixed_propagated_features(h, config, root)
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
    recipe = dict(**settings, input_scale=input_scale)
    recipe_key = _fingerprint(recipe)
    save_json(recipe, root / f"student_recipe_{recipe_key}.json")
    screen_protocol = dict(candidates=candidates, steps=steps, checkpoints=checkpoints,
                           condensation_seed=condensation_seed, student_seeds=list(student_seeds), recipe=recipe)
    if report_routes:
        screen_protocol["report_routes"] = True
    screen_key = _fingerprint(screen_protocol)
    save_json(screen_protocol, root / f"screen_protocol_{screen_key}.json")
    evaluation_graph = dict(graph, x=graph["x"] * input_scale)
    checks = sorted({0, steps, *(checkpoints or [])})
    records = []
    feature_assignment = assignment
    for candidate in candidates:
        if stop():
            raise InterruptedError("Citation screen stopped between candidates")
        identity = {"method": "low_rank", "width": 0, "lr": 0.01, **candidate}
        mixing = identity.get("mixing", config["mixing"])
        mass_mode = _assignment_mass_mode(identity)
        if identity["method"] == "nystrom":
            # Keep legacy, unverified resume files separate from hardened inputs.
            identity.setdefault("nystrom_schema", 3)
        folder = root / _fingerprint(identity) / f"condensation_{condensation_seed}"
        folder.mkdir(parents=True, exist_ok=True)
        save_json(identity, folder.parent / "candidate.json")
        resume = folder / "resume.pt"
        state = torch.load(resume, map_location="cpu", weights_only=False) if resume.exists() else None
        q = training_refined_targets(logits, identity["T"], graph["y"], train,
                                     identity.get("train_target_mix", 0.0))
        assignment = feature_assignment
        if identity.get("initialization", "feature") != "feature":
            from src.partition_initialization import teacher_aware_kmeans, teacher_balanced_kmeans
            init_config = dict(mode=identity["initialization"], alpha=identity.get("alpha", 1.0),
                               T=identity["T"], seed=condensation_seed)
            if identity.get("train_target_mix", 0.0) != 0:
                init_config["train_target_mix"] = identity["train_target_mix"]
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
            if identity["method"] == "distance":
                from src.distance_partition import optimize_distance_ce
                metric_mode = identity.get("metric_input", "features")
                if metric_mode not in ("features", "teacher_joint"):
                    raise ValueError("Unknown distance metric input")
                metric_inputs = (torch.cat((z, identity.get("metric_alpha", 0.3) * (q - q.mean(0))), 1)
                                 if metric_mode == "teacher_joint" else None)
                optimize_distance_ce(z, q, assignment, metric_inputs=metric_inputs,
                    penalty=identity["penalty"], steps=steps, lr=identity["lr"],
                    mixing=mixing,
                    strength=identity.get("strength", 3.0),
                    inner_loss_weighting=identity.get("inner_loss_weighting", "mass"),
                    folder=folder, checkpoint_steps=checks, resume_state=state, stop=stop)
            elif identity["method"] == "nystrom":
                from src.nystrom_ce import cache_features, optimize
                nystrom_options = (dict(inner_loss_weighting="uniform")
                                   if _nystrom_inner_weighting(identity) == "uniform" else {})
                feature_map = get_shared_map(h, root / "nystrom_map_schema3.pt", basis=3000, seed=0)
                phi = cache_features(h, feature_map, root / "nystrom_phi_schema3.npy", stop=stop)
                optimize(h.double(), q, assignment, feature_map, phi, folder, steps,
                         penalty=identity["penalty"], lr=identity["lr"], rank=identity["rank"],
                         seed=condensation_seed, checkpoint_every=25, mixing=mixing, stop=stop,
                         **nystrom_options)
                del feature_map, phi
            elif identity["method"] == "coarsening":
                from src.coarsening_ce import optimize_coarsening_ce
                optimize_coarsening_ce(graph["x"], q, assignment, graph["adj"], h=h, transform=transform,
                    penalty=identity["penalty"], steps=steps, lr=identity["lr"], rank=identity["rank"],
                    mixing=mixing,
                    seed=condensation_seed, mass_scaling=identity.get("mass_scaling", False),
                    inner_loss_weighting=identity.get("inner_loss_weighting", "mass"),
                    folder=folder, checkpoint_steps=checks, resume_state=state, stop=stop)
            else:
                if identity["method"] not in ("low_rank", "mlp"):
                    raise ValueError("Unknown assignment method")
                temperature_options = {}
                if identity.get("learn_temperature", False):
                    temperature_options = dict(temperature_logits=logits.double(),
                        temperature_initial=identity["T"], temperature_lr=identity["temperature_lr"],
                        outer_targets=q.detach())
                optimize_ce_assignment(z, q, assignment, penalty=identity["penalty"], steps=steps,
                    mixing=mixing,
                    lr=identity["lr"], assignment_rank=identity["rank"], factor_seed=condensation_seed,
                    assignment_input="features" if identity["method"] == "mlp" else "node",
                    assignment_encoder="mlp" if identity["method"] == "mlp" else "linear",
                    encoder_hidden=identity["width"] if identity["method"] == "mlp" else 64,
                    solver_mode="exact", inner_method="newton_first", implicit_warm_start=True,
                    mass_mode=mass_mode,
                    inner_loss_weighting=identity.get("inner_loss_weighting", "mass"), inner_max_iter=2000, inner_tol=1e-7,
                    cg_max_iter=512, cg_rtol=1e-6, cache_assignment=False,
                    folder=folder, checkpoint_steps=checks, resume_state=state,
                    save_resume=True, save_assignment=False, stop=stop, **temperature_options)
        if identity.get("learn_temperature", False):
            if not resume.exists():
                raise ValueError("Learnable-temperature resume state is missing")
            state = torch.load(resume, map_location="cpu", weights_only=False)
        for step in checks:
            snapshot_path = folder / "checkpoints" / f"step_{step:06d}.pt"
            if identity["method"] == "nystrom":
                snapshot_path = folder / f"step_{step:06d}.pt"
            if not snapshot_path.exists():
                continue
            snapshot = torch.load(snapshot_path, map_location=device, weights_only=False)
            if (identity["method"] == "nystrom"
                    and snapshot.get("inner_loss_weighting", "mass") != _nystrom_inner_weighting(identity)):
                raise ValueError("Nyström checkpoint inner weighting differs from the candidate")
            temperature_diagnostic = (_temperature_diagnostics(snapshot, state, step)
                                      if identity.get("learn_temperature", False) else {})
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
                if stop():
                    raise InterruptedError("Citation student evaluation stopped")
                result = fit_gcn_diagnostic(x * input_scale, y, torch.full_like(mass, 1 / len(mass)), evaluation_graph, q,
                    dict(train=train, val=validation[1]), seed, folder=folder / "validation"
                    / f"step_{step}_{recipe_key}", training_adjacency=training_adj, stop=stop, **settings)
                if report_routes:
                    from src.student_routes import replay_routes
                    evaluation_folder = folder / "validation" / f"step_{step}_{recipe_key}"
                    if any("test_" in key for key in result):
                        raise ValueError("Validation-only student unexpectedly returned test metrics")
                    routes = replay_routes(
                        evaluation_folder / f"seed_{seed}_selected.pt", evaluation_graph,
                        h.float() * input_scale, dict(val=validation[1]), settings,
                        evaluation_folder / f"seed_{seed}_validation_routes_v1.json", seed=seed, stop=stop)
                    if (any("test_" in key for key in routes) or routes["epoch"] != result["epoch"]
                            or not math.isclose(routes["gcn_val_acc"], result["val_acc"], abs_tol=1e-8, rel_tol=0)
                            or not math.isclose(routes["gcn_val_ce"], result["val_ce"], abs_tol=1e-6, rel_tol=1e-6)):
                        raise ValueError("Validation route replay differs from the selected student")
                    result = dict(result, **{key: value for key, value in routes.items()
                                            if key.startswith(("gcn_", "mlp_"))})
                records.append(dict(**identity, condensation_seed=condensation_seed, step=step,
                                    **result, **temperature_diagnostic,
                                    dropout=dropout, input_scale=input_scale, epochs=epochs,
                                    student_recipe=recipe_key,
                                    candidate_path=str(folder.parent.resolve())))
        print("CANDIDATE_DONE", dataset, identity, "seconds", round(time.monotonic() - started, 2), flush=True)
        write_table(pd.DataFrame(records), root / f"screen_{condensation_seed}_{steps}_{recipe_key}_{screen_key}.csv")
    frame = pd.DataFrame(records)
    if "mixing" in frame.columns:
        # A selected default row must carry .05, not an ambiguous NaN value.
        frame["mixing"] = frame["mixing"].fillna(config["mixing"])
    if "learn_temperature" in frame.columns:
        frame["learn_temperature"] = frame["learn_temperature"].eq(True)
        frame["temperature_lr"] = frame["temperature_lr"].fillna(0.003)
    keys = ["method", "width", "lr", "T", "rank", "penalty", "step", "candidate_path"]
    keys += [key for key in ("initialization", "alpha", "inner_loss_weighting", "mass_mode", "mass_scaling",
                            "metric_input", "metric_alpha", "strength", "train_target_mix", "mixing",
                            "learn_temperature", "temperature_lr") if key in frame.columns]
    keys += ["dropout", "input_scale", "epochs", "student_recipe"]
    grouped = frame.groupby(keys, dropna=False)
    if report_routes or "learn_temperature" in frame.columns:
        aggregates = dict(mean=("val_acc", "mean"), std=("val_acc", "std"), count=("val_acc", "count"))
        if report_routes:
            aggregates.update({f"{metric}_{stat}": (metric, stat)
                               for metric in ("gcn_val_acc", "mlp_val_acc", "gcn_val_ce", "mlp_val_ce")
                               for stat in ("mean", "std")})
        if "learn_temperature" in frame.columns:
            aggregates.update({f"{metric}_{stat}": (metric, stat)
                               for metric in ("learned_temperature", "log_temperature_gradient", "surrogate_ce")
                               for stat in ("mean", "std")})
            aggregates["surrogate_J_exact"] = ("surrogate_J_exact", "min")
        summary = grouped.agg(**aggregates).reset_index()
    else:
        summary = grouped.val_acc.agg(["mean", "std", "count"]).reset_index()
    summary = summary.sort_values("mean", ascending=False)
    write_table(summary, root / f"ranking_{condensation_seed}_{steps}_{recipe_key}_{screen_key}.csv")
    print("VALIDATION_RANKING", dataset, "\n", summary.head(8).to_string(index=False), flush=True)
    return summary, root


def selected_test(root, choice, condensation_seeds=(0, 1, 2), student_seeds=(100, 101, 102, 103, 104),
                  dropout=0.9, epochs=1000, data_dir="data", device="cuda", input_scale=1.0,
                  stop=lambda: False, report_routes=True, student_settings=None):
    """Test only a fixed configuration and checkpoint, with fresh student seeds."""
    root = Path(root)
    candidate = {key: choice[key] for key in ("method", "width", "lr", "T", "rank", "penalty")}
    candidate["rank"], candidate["width"] = int(candidate["rank"]), int(candidate["width"])
    if candidate["method"] == "nystrom":
        candidate["nystrom_schema"] = 3
    candidate.update({key: choice[key] for key in ("initialization", "alpha", "inner_loss_weighting", "mass_mode", "mass_scaling",
                                                  "metric_input", "metric_alpha", "strength", "train_target_mix") if key in choice
                      and pd.notna(choice[key])})
    if "mixing" in choice:
        candidate["mixing"] = choice["mixing"]
    for key in ("learn_temperature", "temperature_lr"):
        if key in choice:
            candidate[key] = choice[key]
    candidate, _ = _candidate_background_mixing(candidate)
    candidate = _candidate_temperature(candidate)
    for key in ("lr", "T", "penalty"):
        candidate[key] = float(candidate[key])
    _assignment_mass_mode(candidate)
    if candidate["method"] == "nystrom":
        _nystrom_inner_weighting(candidate)
    config = json.loads((root / "config.json").read_text())
    step = int(choice["step"])
    selection = dict(candidate=candidate, step=step, selection="validation_only",
                     student=dict(dropout=dropout, input_scale=input_scale, epochs=epochs),
                     condensation_seeds=list(condensation_seeds), student_seeds=list(student_seeds))
    settings = _student_settings(dropout, epochs, student_settings)
    if student_settings:
        selection["student"]["overrides"] = student_settings
    if report_routes:
        selection["serving_comparison"] = "MLP/GCN, same GCN validation-selected weights"
    selection_key = _fingerprint(selection)
    graph, train, validation, testing, h = _prepare_dataset(config["dataset"], data_dir, device,
                                                          config.get("citation_features", "default"))
    if dataset_digest(graph, train, validation, testing) != config["data_digest"]:
        raise ValueError("Graph or splits differ from the selected source screen")
    h = fixed_propagated_features(h, config, root)
    save_json(selection, root / "selected.json")
    save_json(selection, root / f"selected_{selection_key}.json")
    logits = torch.load(root / "teacher.pt", map_location=device, weights_only=False)["logits"]
    q = training_refined_targets(logits, candidate["T"], graph["y"], train,
                                 candidate.get("train_target_mix", 0.0))
    records = []
    recipe_key = _fingerprint(dict(**settings, input_scale=input_scale))
    evaluation_graph = dict(graph, x=graph["x"] * input_scale)
    for cond_seed in condensation_seeds:
        # This also produces the needed condensate if it is absent.
        _, generated_root = run_screen(config["dataset"], config["ratio"], root.parents[2], [candidate], steps=step,
                   student_seeds=(0,), condensation_seed=cond_seed, dropout=dropout, epochs=epochs,
                   data_dir=data_dir, device=device, input_scale=input_scale,
                   citation_features=config.get("citation_features", "default"), stop=stop,
                   student_settings=student_settings)
        if generated_root.resolve() != root.resolve():
            raise ValueError("Generated condensate belongs to a different source screen")
        folder = root / _fingerprint(candidate) / f"condensation_{cond_seed}"
        inputs = torch.load(root / f"inputs_{cond_seed}.pt", map_location=device, weights_only=False)
        transform = FeatureTransform(**inputs["transform"])
        snapshot_path = (folder / f"step_{step:06d}.pt" if candidate["method"] == "nystrom"
                         else folder / "checkpoints" / f"step_{step:06d}.pt")
        snapshot = torch.load(snapshot_path, map_location=device, weights_only=False)
        if (candidate["method"] == "nystrom"
                and snapshot.get("inner_loss_weighting", "mass") != _nystrom_inner_weighting(candidate)):
            raise ValueError("Nyström checkpoint inner weighting differs from the selection")
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
            evaluation_folder = folder / "final" / f"step_{step}_{recipe_key}"
            result = fit_gcn_diagnostic(x * input_scale, y, torch.full_like(mass, 1 / len(mass)), evaluation_graph, q,
                dict(train=train, val=validation[1], test=testing[1]), seed,
                folder=evaluation_folder, training_adjacency=training_adj, stop=stop, **settings)
            if report_routes:
                from src.student_routes import replay_routes
                routes = replay_routes(evaluation_folder / f"seed_{seed}_selected.pt",
                    evaluation_graph, h.float() * input_scale,
                    dict(val=validation[1], test=testing[1]), settings,
                    evaluation_folder / f"seed_{seed}_routes.json", seed=seed, stop=stop)
                result = dict(result, **{key: value for key, value in routes.items()
                                        if key.startswith(("mlp_", "gcn_"))})
            records.append(dict(condensation_seed=cond_seed, **result))
            write_table(pd.DataFrame(records), root / "final.csv")
            write_table(pd.DataFrame(records), root / f"final_{selection_key}.csv")
    return pd.DataFrame(records)
