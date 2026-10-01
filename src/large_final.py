"""Fixed-configuration large-data test replay after validation-only student fits."""
import csv
import hashlib
import io
import json
import math
import time
from pathlib import Path

import torch

from src.data import _prepare_dataset
from src.evaluation import _update_tensor_digest
from src.io import _fingerprint, save_json
from src.large_pilot import _wall_deadline, run_pilot
from src.shared_features import get_shared_h
from src.student_routes import replay_routes

_FIXED_OPTIONS = {
    "dataset", "ratio", "basis", "steps", "rank", "penalty", "lr", "initialization",
    "alpha", "chunk", "teacher_seed", "teacher_max_iter", "dropout", "hidden", "epochs",
    "temperature", "train_target_mix", "inner_loss_weighting", "surrogate", "mixing",
    "teacher_gamma", "teacher_type", "gcn_teacher_epochs",
}


def _read_selection(path, expected_sha256):
    payload = Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("Frozen large selection changed; use a new selection and job ID")
    selection = json.loads(payload)
    required = {"version", "selection_basis", "options", "source_data_digest", "required_source_digest",
                "validation_student_seeds", "validation_job_ids"}
    if (not required <= selection.keys() or selection["version"] != 1
            or selection["selection_basis"] != "validation_only"
            or not selection["source_data_digest"] or not selection["required_source_digest"]
            or not selection["validation_job_ids"]):
        raise ValueError("Require a frozen validation-only selection with source provenance")
    options = selection["options"]
    if not {"dataset", "ratio", "steps"} <= options.keys() or set(options) - _FIXED_OPTIONS:
        raise ValueError("Frozen options contain missing or unsupported experiment fields")
    return selection


def _check_source(selection):
    from src.research_loop import implementation_provenance

    if implementation_provenance()["source_digest"] != selection["required_source_digest"]:
        raise ValueError("Final source differs from the frozen selection's implementation")


def _selected_student(folder, seed, settings, student):
    """Read one completed validation selection and verify its three artifacts."""
    source = folder / f"seed_{seed}_selected.pt"
    payload = source.read_bytes()
    selected = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
    cached = json.loads((folder / f"seed_{seed}.json").read_text())
    recipe, result = cached["recipe"], cached["result"]
    if (cached["fingerprint"] != _fingerprint(recipe)
            or selected["fingerprint"] != cached["fingerprint"]
            or recipe.get("seed") != seed or recipe.get("settings") != settings
            or recipe.get("weighting") != "uniform" or recipe.get("layers") != 2
            or recipe.get("selection") != "first maximum validation accuracy"
            or recipe.get("test_enabled") is not False
            or any("test_" in key for key in result)
            or result["seed"] != seed or result["weighting"] != "uniform"
            or selected["epoch"] != result["epoch"] or selected["epoch"] != student["epoch"]
            or result["val_acc"] != student["val_acc"]):
        raise ValueError("Selected student checkpoint differs from its completed validation fit")
    history_bytes = (folder / f"seed_{seed}_epochs.csv").read_bytes()
    history = list(csv.DictReader(io.StringIO(history_bytes.decode())))
    if (not history or any("test_" in key for key in history[0])
            or any(not math.isfinite(float(row["val_acc"])) for row in history)):
        raise ValueError("Require a complete validation-only student history")
    epochs = [int(row["epoch"]) for row in history]
    best = max(history, key=lambda row: float(row["val_acc"]))
    if (epochs != sorted(set(epochs)) or epochs[0] < 1
            or epochs[-1] != settings["epochs"] or result["last_epoch"] != settings["epochs"]
            or int(best["epoch"]) != selected["epoch"]
            or float(best["val_acc"]) != result["val_acc"]):
        raise ValueError("Selected epoch differs from the first maximum of completed validation history")
    return selected, payload, hashlib.sha256(history_bytes).hexdigest()


def _test_inputs(dataset, data_dir, device, folder, stopped, guard):
    graph, train_mask, validation, testing, h = _prepare_dataset(dataset, data_dir, device)
    del graph, train_mask, validation, h
    graph, mask = testing
    x, adjacency = graph["x"], graph["adj"]
    digest = hashlib.sha256()
    for key, tensor in (("test_x", x), ("test_adj", adjacency)):
        _update_tensor_digest(digest, key, tensor, stopped)
    with torch.no_grad():
        propagated = (adjacency @ (adjacency @ x) if adjacency.layout == torch.strided
                      else torch.sparse.mm(adjacency, torch.sparse.mm(adjacency, x)))
    guard()
    source = _fingerprint(dict(inputs=digest.hexdigest(), propagation="test-own-packed-S2X-v1",
                               dtype=str(x.dtype), torch_version=str(torch.__version__)))
    propagated = get_shared_h(propagated, folder / "test_S2X.pt", source)
    if mask is None:
        mask = torch.ones(len(x), dtype=torch.bool, device=x.device)
    return graph, propagated, {"test": mask}


def run_final(selection_path, selection_sha256, output_dir, *, condensation_seed,
              student_seeds=(910, 911, 912, 913, 914), pilot_output_dir="results/large_pilot_v2",
              data_dir="data", device="cuda", deadline_seconds=300, stop=lambda: False):
    """Fit fresh students on a frozen condensation, then replay its fixed weights.

    One condensation seed per bounded job; aggregate several jobs as independent
    condensations, and report student variability separately. The selection file
    is hashed in the immutable job arguments, and is copied into the result root.
    Student seeds must be disjoint from the selection's validation search seeds.
    run_pilot receives no testing graph. Only after that full validation-only fit
    completes do we load each split's testing graph and replay selected weights.
    No optimizer, target fitting or epoch selection occurs in the test replay.
    """
    selection = _read_selection(selection_path, selection_sha256)
    _check_source(selection)
    if (isinstance(deadline_seconds, bool) or not isinstance(deadline_seconds, (float, int))
            or not math.isfinite(deadline_seconds) or deadline_seconds <= 0):
        raise ValueError("Final deadline must be positive and finite")
    if (isinstance(condensation_seed, bool) or not isinstance(condensation_seed, int)
            or condensation_seed < 0 or not student_seeds
            or len(set(student_seeds)) != len(student_seeds)
            or any(isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in student_seeds)
            or set(student_seeds) & set(selection["validation_student_seeds"])):
        raise ValueError("Require one valid condensation seed and distinct fresh student seeds")
    options = dict(selection["options"])
    root = Path(output_dir) / options["dataset"] / f"selection_{selection_sha256[:16]}"
    root.mkdir(parents=True, exist_ok=True)
    frozen_path = root / "frozen_selection.json"
    if frozen_path.exists() and json.loads(frozen_path.read_text()) != selection:
        raise ValueError("Result root contains a different frozen selection")
    save_json(selection, frozen_path)
    invocation = dict(selection_sha256=selection_sha256, condensation_seed=condensation_seed,
                      student_seeds=list(student_seeds), device=str(device),
                      data_dir=str(Path(data_dir).resolve()), selection_basis="validation_only")
    report = dict(status="starting", stage="validation_only_student_fit", rows=[], invocation=invocation,
                  dataset=options["dataset"], ratio=options["ratio"], selection_sha256=selection_sha256,
                  serving_selection="same weights at GCN validation-selected epoch")
    report_path = root / f"final_{_fingerprint(invocation)}.json"
    deadline = time.monotonic() + deadline_seconds
    started = time.monotonic()

    def stopped():
        return bool(stop() or time.monotonic() >= deadline or (root / "STOP").exists())

    def guard():
        if stopped():
            raise InterruptedError("Fixed large final stopped or deadline reached")
        _read_selection(selection_path, selection_sha256)

    try:
        with _wall_deadline(deadline_seconds):
            guard()
            pilot, candidate_root = run_pilot(
                **options, output_dir=pilot_output_dir, condensation_seed=condensation_seed,
                student_seeds=student_seeds, data_dir=data_dir, device=device,
                deadline_seconds=max(1e-3, deadline - time.monotonic()), report_routes=True, stop=stopped)
            if pilot["status"] != "complete":
                report.update(status=pilot["status"], reason=pilot.get("reason", "Validation-only fit did not complete"))
            else:
                guard()
                candidate_root = Path(candidate_root)
                protocol = json.loads((candidate_root / "protocol.json").read_text())
                if protocol["data_digest"] != selection["source_data_digest"]:
                    raise ValueError("Final inputs differ from the validation-selected source")
                settings = pilot["student_recipe"]
                students = {r["seed"]: r for r in pilot["students"]}
                if set(students) != set(student_seeds) or any("test_" in k for r in students.values() for k in r):
                    raise ValueError("Validation fit must complete every fresh student without test metrics")
                folder = candidate_root / "validation" / f"step_{options['steps']}_{_fingerprint(settings)}"
                frozen_models = root / f"condensation_{condensation_seed}" / "selected_weights"
                frozen_models.mkdir(parents=True, exist_ok=True)
                manifest = {}
                for seed in student_seeds:
                    source = folder / f"seed_{seed}_selected.pt"
                    selected, payload, history_digest = _selected_student(folder, seed, settings, students[seed])
                    digest = hashlib.sha256(payload).hexdigest()
                    target = frozen_models / source.name
                    if target.exists():
                        if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                            raise ValueError("Selected weights differ from the frozen final checkpoint")
                    else:
                        temporary = target.with_suffix(".tmp.pt")
                        temporary.write_bytes(payload)
                        temporary.replace(target)
                    manifest[str(seed)] = dict(epoch=selected["epoch"], fingerprint=selected["fingerprint"],
                                               checkpoint_sha256=digest, validation_history_sha256=history_digest)
                manifest_path = frozen_models / f"manifest_{_fingerprint(invocation)}.json"
                if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
                    raise ValueError("Final selected-weight manifest changed")
                save_json(manifest, manifest_path)
                report.update(stage="fixed_weights_test_replay", candidate_root=str(candidate_root.resolve()),
                              student_recipe=settings, teacher=pilot["teacher"], selected_weights=manifest)
                _check_source(selection)
                for seed in student_seeds:
                    frozen = frozen_models / f"seed_{seed}_selected.pt"
                    if hashlib.sha256(frozen.read_bytes()).hexdigest() != manifest[str(seed)]["checkpoint_sha256"]:
                        raise ValueError("Frozen student weights changed before loading test inputs")
                guard()
                graph, propagated, masks = _test_inputs(options["dataset"], data_dir, device, root, stopped, guard)
                for seed in student_seeds:
                    guard()
                    frozen = frozen_models / f"seed_{seed}_selected.pt"
                    if hashlib.sha256(frozen.read_bytes()).hexdigest() != manifest[str(seed)]["checkpoint_sha256"]:
                        raise ValueError("Frozen student weights changed before test replay")
                    scores = replay_routes(
                        frozen, graph, propagated, masks, settings,
                        root / f"condensation_{condensation_seed}" / f"student_{seed}_test_routes.json",
                        seed=seed, stop=stopped, test_only=True)
                    report["rows"].append(dict(condensation_seed=condensation_seed, **scores,
                                               gcn_val_acc=students[seed]["val_acc"],
                                               mlp_val_acc=students[seed]["mlp_val_acc"]))
                    save_json(report, report_path)
                report.update(status="complete")
    except InterruptedError as exc:
        report.update(status="stopped", reason=str(exc))
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
        report.update(status="failed", reason=str(exc), error_type=type(exc).__name__)
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        save_json(report, report_path)
    return report, root
