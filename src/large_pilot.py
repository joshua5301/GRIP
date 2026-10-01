"""Bounded full-data Arxiv/Flickr/Reddit validation pilots on one GPU.

Teacher features are shared CPU memory maps; only training labels fit the
Nyström logistic teacher.  Representatives remain P-weighted means of S^2 X
and teacher probabilities.  The pilot uses the existing exact linear-CE
low-rank surrogate in fixed original-H RMS coordinates and uniform GCN CE.
Flickr/Reddit validation uses its own induced graph.  No test split is scored.

Run in a dedicated main-thread process: POSIX alarms bound long existing
solver calls, and cooperative guards bound streaming chunks and epochs.  An
in-flight native CPU/CUDA operation finishes before Python handles an alarm.
The global advisory GPU lock serializes pilots; the caller must also serialize
other experiment families sharing that GPU.  Interrupted runs retain atomic
teacher/feature caches and the latest completed condensation checkpoint.
"""

import fcntl
import hashlib
import json
import math
import signal
import tempfile
import threading
import time
from contextlib import contextmanager
from numbers import Real
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.data import BUDGET, _prepare_dataset
from src.inductive_evaluation import _update_tensor_digest, fit_inductive_gcn
from src.initialization import feature_kmeans
from src.io import _fingerprint, save_json, save_state
from src.moments import decode_moments
from src.nystrom_ce import NystromMap, cache_features, fit_streaming_teacher
from src.nystrom_ce import optimize as optimize_nystrom
from src.partition_initialization import teacher_aware_kmeans, teacher_balanced_kmeans
from src.shared_features import get_shared_h
from src.soft_ce_partition import optimize_ce_assignment, solve_head_system, solve_inner_newton_first
from src.student_routes import replay_routes
from src.sweep_utils import representative
from src.target_refinement import training_refined_targets
from src.transforms import fit_transform


@contextmanager
def _wall_deadline(seconds):
    if threading.current_thread() is not threading.main_thread() or not hasattr(signal, "setitimer"):
        raise ValueError("Run bounded pilots in a POSIX main-thread worker process")
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    started = time.monotonic()

    def expired(signum, frame):
        raise InterruptedError("Pilot wall-clock deadline reached")

    signal.signal(signal.SIGALRM, expired)
    duration = min(seconds, previous_timer[0]) if previous_timer[0] > 0 else seconds
    signal.setitimer(signal.ITIMER_REAL, duration)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            signal.setitimer(
                signal.ITIMER_REAL,
                max(1e-6, previous_timer[0] - (time.monotonic() - started)),
                previous_timer[1],
            )


class _BoundedFeatures:
    def __init__(self, phi, guard):
        self.phi, self.guard = phi, guard
        self.shape, self.dtype = phi.shape, phi.dtype
        self.filename = getattr(phi, "filename", None)

    def __len__(self):
        return len(self.phi)

    def __getitem__(self, key):
        self.guard()
        return self.phi[key]


def _data_digest(graph, train, validation, h, guard):
    digest = hashlib.sha256()
    values = [
        ("x", graph["x"]),
        ("adj", graph["adj"]),
        ("h", h),
        ("train_mask", train),
        ("train_labels", graph["y"][train]),
    ]
    val_graph, val_mask = validation
    values.extend(
        [
            ("val_x", val_graph["x"]),
            ("val_adj", val_graph["adj"]),
            ("val_mask", val_mask),
            ("val_labels", val_graph["y"] if val_mask is None else val_graph["y"][val_mask]),
        ]
    )
    for name, value in values:
        guard()
        _update_tensor_digest(digest, name, value)
    return digest.hexdigest()


@torch.no_grad()
def _teacher_validation(feature_map, weight, graph, mask, chunk, guard):
    x, adjacency = graph["x"], graph["adj"]
    h = (
        adjacency @ (adjacency @ x)
        if adjacency.layout == torch.strided
        else torch.sparse.mm(adjacency, torch.sparse.mm(adjacency, x))
    )
    correct, count, ce = 0, 0, 0.0
    for start in range(0, len(h), chunk):
        guard()
        end = min(start + chunk, len(h))
        chosen = slice(None) if mask is None else mask[start:end]
        scores = feature_map(h[start:end])[chosen] @ weight
        labels = graph["y"][start:end][chosen]
        if len(labels):
            correct += int((scores.argmax(1) == labels).sum())
            count += len(labels)
            ce += float(F.cross_entropy(scores, labels, reduction="sum"))
    if not count:
        raise ValueError("Empty teacher validation split")
    return dict(val_acc=100 * correct / count, val_ce=ce / count, val_nodes=count)


@torch.no_grad()
def _validation_route_inputs(validation, source_digest, teacher_root, guard):
    """Freeze the validation graph's own packed-adjacency S²X for paired serving."""
    graph, mask = validation
    guard()
    x, adjacency = graph["x"], graph["adj"]
    if adjacency.layout == torch.strided:
        propagated = adjacency @ (adjacency @ x)
    else:
        propagated = torch.sparse.mm(adjacency, torch.sparse.mm(adjacency, x))
    guard()
    source = _fingerprint(
        dict(
            data_digest=source_digest,
            propagation="validation-packed-adjacency-S2X-v1",
            dtype=str(x.dtype),
            torch_version=str(torch.__version__),
        )
    )
    propagated = get_shared_h(propagated, teacher_root / "validation_routes_S2X_v1.pt", source)
    guard()
    # replay_routes requires explicit masks. None means every node in this
    # induced validation graph, never the separate training/testing graph.
    if mask is None:
        mask = torch.ones(len(x), dtype=torch.bool, device=x.device)
    return graph, propagated, dict(val=mask)


def run_pilot(
    dataset,
    ratio,
    output_dir,
    *,
    basis=512,
    steps=20,
    condensation_seed=0,
    student_seeds=(0,),
    epochs=300,
    deadline_seconds=300,
    data_dir="data",
    device="cuda",
    rank=16,
    penalty=1e-4,
    lr=0.01,
    initialization="feature",
    alpha=1.0,
    chunk=2048,
    teacher_seed=0,
    teacher_max_iter=1000,
    dropout=0.31881090213944857,
    hidden=256,
    stop=lambda: False,
    temperature=0.3,
    train_target_mix=0.0,
    inner_loss_weighting="mass",
    surrogate="linear",
    report_routes=False,
    mixing=0.05,
    teacher_gamma=0.01,
):
    """Return (report, root) for a bounded, resumable full-data screen.

    Minimal configured budgets are Arxiv .0005/90, Flickr .001/44, Reddit .0005/77.
    initialization may be feature, teacher_joint, or teacher_balanced; the latter
    two use the selected target Q. train_target_mix optionally blends training
    rows with their hard labels; held-out Q stays softmax(logits/temperature).
    mixing controls the fixed hard-partition logit prior used by both surrogates;
    it does not change k-means or teacher targets. The default preserves caches.
    inner_loss_weighting controls only the convex surrogate; final students
    always use uniform CE. Extend steps (e.g. 20 -> 50) to resume the same candidate.
    surrogate='nystrom' applies the shared teacher map after P-weighted raw-H
    means and evaluates outer CE on verified CPU phi; it requires mass inner CE.
    teacher_gamma defaults to .01 with its exact legacy cache identity. Other
    values have separate teacher/condensation/student folders, while original H,
    map, phi and validation-route H reuse the legacy geometry folder unchanged.
    Teacher basis defaults to 512 for a pilot and is explicit in protocol/results;
    such a pilot is not a claim of parity with a basis-3000 research benchmark.

    The report records complete/stopped/failed, stages, elapsed seconds, GPU peak,
    teacher validation and completed student validation fits. report_routes=True
    also replays paired MLP/GCN serving with the same GCN validation-selected
    weights on the validation graph's own S²X/X+adjacency. No route-specific
    training or epoch selection occurs. The default preserves existing cache
    identities and report names. A nonconverged
    teacher or exact inner/adjoint is refused and reported.  No test metrics exist.
    """
    if dataset not in ("arxiv", "flickr", "reddit") or (dataset, ratio) not in BUDGET:
        raise ValueError("Use a configured Arxiv, Flickr or Reddit node budget")
    if any(not isinstance(value, int) or value < 1 for value in (basis, epochs, chunk, teacher_max_iter)):
        raise ValueError("basis, epochs, chunk and teacher_max_iter must be positive integers")
    if (
        not isinstance(steps, int)
        or steps < 0
        or not math.isfinite(deadline_seconds)
        or deadline_seconds <= 0
    ):
        raise ValueError("Require nonnegative steps and a positive finite deadline")
    if initialization not in ("feature", "teacher_joint", "teacher_balanced") or not student_seeds:
        raise ValueError("Use a supported initialization and nonempty student seeds")
    if not math.isfinite(alpha) or alpha < 0 or not 1 <= rank <= BUDGET[(dataset, ratio)]:
        raise ValueError("Invalid teacher-feature weight or assignment rank")
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, Real)
        or not math.isfinite(temperature)
        or temperature <= 0
    ):
        raise ValueError("temperature must be positive and finite")
    if (
        isinstance(train_target_mix, bool)
        or not isinstance(train_target_mix, Real)
        or not math.isfinite(train_target_mix)
        or not 0 <= train_target_mix <= 1
    ):
        raise ValueError("train_target_mix must be finite and lie in [0, 1]")
    if inner_loss_weighting not in ("mass", "uniform"):
        raise ValueError("inner_loss_weighting must be mass or uniform")
    if surrogate not in ("linear", "nystrom"):
        raise ValueError("surrogate must be linear or nystrom")
    if surrogate == "nystrom" and inner_loss_weighting != "mass":
        raise ValueError("The Nyström surrogate supports mass inner CE only")
    if (
        isinstance(mixing, bool)
        or not isinstance(mixing, Real)
        or not math.isfinite(mixing)
        or not 0 < mixing < 1
    ):
        raise ValueError("mixing must be finite and lie strictly in (0, 1)")
    mixing = float(mixing)
    if not isinstance(report_routes, bool):
        raise ValueError("report_routes must be boolean")
    if (
        isinstance(teacher_gamma, bool)
        or not isinstance(teacher_gamma, Real)
        or not math.isfinite(teacher_gamma)
        or teacher_gamma <= 0
    ):
        raise ValueError("teacher_gamma must be positive and finite")
    teacher_gamma = float(teacher_gamma)
    started, deadline = time.monotonic(), time.monotonic() + deadline_seconds
    # The legacy gamma=.01 identity anchors shared, gamma-independent geometry.
    geometry_protocol = dict(
        version=2,
        dataset=dataset,
        data_dir=str(Path(data_dir).resolve()),
        basis=basis,
        teacher_seed=teacher_seed,
        kernel="relu",
        gamma=0.01,
        teacher_feature_route="resident training rows with memory-checked streaming fallback",
        chunk=chunk,
        torch_version=str(torch.__version__),
    )
    geometry_root = Path(output_dir) / dataset / f"teacher_{_fingerprint(geometry_protocol)}"
    teacher_protocol = dict(geometry_protocol, gamma=teacher_gamma)
    teacher_root = (
        geometry_root if teacher_gamma == 0.01
        else geometry_root / f"gamma_{_fingerprint(teacher_protocol)}"
    )
    candidate = dict(
        version=1,
        ratio=ratio,
        cells=BUDGET[(dataset, ratio)],
        temperature=temperature,
        rank=rank,
        penalty=penalty,
        lr=lr,
        condensation_seed=condensation_seed,
        initialization=initialization,
        alpha=alpha,
        assignment="low_rank",
        inner_loss=f"exact_{inner_loss_weighting}_ce",
        student_loss="uniform_ce",
        synthetic_adjacency="identity",
    )
    # Preserve the existing mass/zero-mix candidate identity and saved endpoints.
    if train_target_mix != 0:
        candidate["train_target_mix"] = train_target_mix
    if surrogate != "linear":
        candidate["surrogate"] = surrogate
    if mixing != 0.05:
        candidate["mixing"] = mixing
    root = teacher_root / f"candidate_{_fingerprint(candidate)}"
    root.mkdir(parents=True, exist_ok=True)
    settings = dict(
        epochs=epochs, eval_every=10, hidden=hidden, dropout=dropout, lr=0.01, weight_decay=0.0005
    )
    report = dict(
        dataset=dataset,
        ratio=ratio,
        cells=candidate["cells"],
        basis=basis,
        requested_steps=steps,
        deadline_seconds=deadline_seconds,
        status="starting",
        stage="starting",
        students=[],
        device=str(device),
        selection="validation_only",
        candidate=candidate,
        student_recipe=settings,
    )
    if teacher_gamma != 0.01:
        report["teacher_gamma"] = teacher_gamma
        report["geometry_cache_root"] = str(geometry_root.resolve())
    if report_routes:
        report["serving_comparison"] = dict(
            selection="same weights at GCN validation-selected epoch",
            graph="validation split graph; Arxiv full graph with validation mask",
            gcn="validation X and its own normalized adjacency",
            mlp="validation graph own S²X; no adjacency",
        )
    lock, gpu_started = None, False

    def guard():
        if stop() or time.monotonic() >= deadline or (root / "STOP").exists():
            raise InterruptedError("Pilot stopped or deadline reached")

    def stopped():
        return bool(stop() or time.monotonic() >= deadline or (root / "STOP").exists())

    def stage(name, **detail):
        guard()
        report.update(stage=name, elapsed_seconds=time.monotonic() - started, **detail)
        print("LARGE_PILOT_STAGE", json.dumps(report), flush=True)
        save_json(report, root / "report.json")

    def bounded_inner(*args, **kwargs):
        guard()
        fitted = solve_inner_newton_first(*args, **kwargs)
        guard()
        return fitted

    def bounded_adjoint(*args, **kwargs):
        guard()
        solved = solve_head_system(*args, **kwargs)
        guard()
        return solved

    try:
        with _wall_deadline(deadline_seconds):
            guard()
            torch.set_num_threads(4)
            if str(device).startswith("cuda"):
                index = torch.device(device).index or 0
                lock = (Path(tempfile.gettempdir()) / f"grip_large_pilot_gpu_{index}.lock").open("a")
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as exc:
                    raise RuntimeError(
                        "Another large pilot holds this GPU; use the single-GPU queue"
                    ) from exc
                torch.cuda.reset_peak_memory_stats(device)
                gpu_started = True
                report["gpu"] = torch.cuda.get_device_name(device)
            stage("loading_full_graph")
            graph, train, validation, unused_testing, h = _prepare_dataset(dataset, data_dir, device)
            del unused_testing
            source_digest = _data_digest(graph, train, validation, None, guard)
            h = get_shared_h(
                h,
                geometry_root / "propagated_H.pt",
                _fingerprint(
                    dict(
                        data=source_digest,
                        propagation="SGC2-normalize-adj-v1",
                        torch=str(torch.__version__),
                        dtype=str(h.dtype),
                    )
                ),
            )
            stage("fingerprinting_data", train_nodes=len(h), feature_dimension=h.shape[1])
            digest = _data_digest(graph, train, validation, h, guard)
            protocol = dict(teacher_protocol, data_digest=digest)
            protocol_path = teacher_root / "protocol.json"
            if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
                raise ValueError("Dataset or teacher protocol changed under the existing cache")
            save_json(protocol, protocol_path)
            save_json(dict(candidate=candidate, data_digest=digest), root / "protocol.json")
            h = h.double()
            map_path = geometry_root / "feature_map.pt"
            stage("preparing_teacher_map")
            if map_path.exists():
                feature_map = NystromMap(**torch.load(map_path, map_location=device, weights_only=False))
            else:
                feature_map = NystromMap.fit(h, basis=basis, seed=teacher_seed)
                save_state(vars(feature_map), map_path)
            stage("caching_teacher_features")
            phi = cache_features(h, feature_map, geometry_root / "phi.npy", chunk=chunk, stop=stopped)
            bounded_phi = _BoundedFeatures(phi, guard)
            teacher_path = teacher_root / "teacher.pt"
            stage("fitting_teacher")
            if teacher_path.exists():
                saved = torch.load(teacher_path, map_location=device, weights_only=False)
                if (
                    saved["data_digest"] != digest or not saved["converged"]
                    or saved.get("gamma") != teacher_gamma
                ):
                    raise ValueError("Teacher cache does not match the converged training-only fit")
                logits, weight = saved["logits"], saved["weight"]
            else:
                train_labels = graph["y"].new_zeros(len(h))
                train_labels[train] = graph["y"][train]
                logits, weight = fit_streaming_teacher(
                    bounded_phi,
                    train_labels,
                    train,
                    gamma=teacher_gamma,
                    chunk=chunk,
                    max_iter=teacher_max_iter,
                    stop=stopped,
                    resident_training=True,
                )
                guard()
                save_state(
                    dict(logits=logits, weight=weight, gamma=teacher_gamma, data_digest=digest, converged=True),
                    teacher_path,
                )
            stage("teacher_validation")
            report["teacher"] = _teacher_validation(feature_map, weight, *validation, chunk, guard)
            q = training_refined_targets(logits, temperature, graph["y"], train, mixing=train_target_mix)
            del logits, weight
            if surrogate == "linear":
                del phi, bounded_phi, feature_map
                z, transform = fit_transform(h, kind="rms")
            assignment_path = root / "initial_assignment.pt"
            stage("initializing_assignment")
            if assignment_path.exists():
                assignment = torch.load(assignment_path, map_location=device, weights_only=False)
            else:
                if initialization == "feature":
                    assignment = feature_kmeans(h.float().cpu(), candidate["cells"], condensation_seed).to(
                        device
                    )
                else:
                    initializer = (
                        teacher_aware_kmeans if initialization == "teacher_joint" else teacher_balanced_kmeans
                    )
                    assignment = initializer(h, q, candidate["cells"], condensation_seed, alpha=alpha)
                guard()
                save_state(assignment, assignment_path)
            condensation = root / "condensation"
            resume_path = condensation / "resume.pt"
            state = (
                torch.load(resume_path, map_location="cpu", weights_only=False)
                if resume_path.exists()
                else None
            )
            endpoint = (
                condensation / f"step_{steps:06d}.pt"
                if surrogate == "nystrom"
                else condensation / "checkpoints" / f"step_{steps:06d}.pt"
            )
            stage("condensing", resume_step=state["step"] if state is not None else 0)
            if not endpoint.exists():
                if surrogate == "nystrom":
                    optimize_nystrom(
                        h,
                        q,
                        assignment,
                        feature_map,
                        bounded_phi,
                        condensation,
                        steps,
                        penalty=penalty,
                        lr=lr,
                        rank=rank,
                        seed=condensation_seed,
                        chunk=chunk,
                        stop=stopped,
                        checkpoint_every=5,
                        mixing=mixing,
                    )
                else:
                    optimize_ce_assignment(
                        z,
                        q,
                        assignment,
                        penalty=penalty,
                        steps=steps,
                        lr=lr,
                        assignment_rank=rank,
                        mixing=mixing,
                        factor_seed=condensation_seed,
                        chunk_size=chunk,
                        inner_method="newton_first",
                        inner_solver=bounded_inner,
                        implicit_solver=bounded_adjoint,
                        solver_mode="exact",
                        inner_loss_weighting=inner_loss_weighting,
                        folder=condensation,
                        resume_state=state,
                        save_resume=True,
                        save_assignment=False,
                        checkpoint_steps=tuple(range(0, steps + 1, 5)) + (steps,),
                        outer_chunk_size=chunk,
                        stop=stopped,
                    )
            guard()
            snapshot = torch.load(endpoint, map_location=device, weights_only=False)
            if surrogate == "nystrom":
                report.update(completed_steps=steps, outer_ce=float(snapshot["outer_ce"]))
                cx, cy, mass = decode_moments(snapshot["moments"], h.shape[1])
                cx, cy = cx.float(), cy.float()
                del phi, bounded_phi, feature_map
            else:
                if not snapshot["J_exact"]:
                    raise RuntimeError("Refusing a nonconverged condensation endpoint")
                report.update(completed_steps=steps, outer_ce=float(snapshot["teacher_ce"]))
                cx, cy, mass = representative(snapshot["moments"], transform, z.shape[1], device)
            evaluation_folder = root / "validation" / f"step_{steps}_{_fingerprint(settings)}"
            if report_routes:
                stage("preparing_validation_serving_inputs")
                route_graph, route_h, route_masks = _validation_route_inputs(
                    validation, source_digest, geometry_root, guard
                )
            for seed in student_seeds:
                stage("validation_student", student_seed=seed)
                result = fit_inductive_gcn(
                    cx,
                    cy,
                    mass,
                    graph,
                    validation,
                    seed=seed,
                    settings=settings,
                    folder=evaluation_folder,
                    stop=stopped,
                    weighting="uniform",
                    train_mask=train,
                )
                if any(key.startswith("test_") for key in result):
                    raise RuntimeError("Screening evaluator unexpectedly returned test metrics")
                if report_routes:
                    stage("validation_serving_routes", student_seed=seed)
                    routes = replay_routes(
                        evaluation_folder / f"seed_{seed}_selected.pt",
                        route_graph,
                        route_h,
                        route_masks,
                        settings,
                        evaluation_folder / f"seed_{seed}_validation_routes_v1.json",
                        seed=seed,
                        stop=stopped,
                    )
                    if any("test_" in key for key in routes):
                        raise RuntimeError("Validation route replay unexpectedly returned test metrics")
                    if not math.isclose(routes["gcn_val_acc"], result["val_acc"], rel_tol=0, abs_tol=1e-8):
                        raise RuntimeError("GCN route replay differs from the selected student's validation score")
                    if not math.isclose(routes["gcn_val_ce"], result["val_ce"], rel_tol=1e-6, abs_tol=1e-6):
                        raise RuntimeError("GCN route replay differs from the selected student's validation CE")
                    result = dict(
                        result,
                        **{key: value for key, value in routes.items() if key.startswith(("mlp_", "gcn_"))},
                        serving_selection=routes["selection"],
                        gcn_minus_mlp_val_acc=routes["gcn_val_acc"] - routes["mlp_val_acc"],
                        gcn_minus_mlp_val_ce=routes["gcn_val_ce"] - routes["mlp_val_ce"],
                    )
                report["students"].append(result)
                save_json(report, root / "report.json")
            report.update(
                status="complete", val_acc_mean=float(np.mean([r["val_acc"] for r in report["students"]]))
            )
            if report_routes:
                report.update(
                    mlp_val_acc_mean=float(np.mean([r["mlp_val_acc"] for r in report["students"]])),
                    gcn_val_acc_mean=float(np.mean([r["gcn_val_acc"] for r in report["students"]])),
                    gcn_minus_mlp_val_acc_mean=float(
                        np.mean([r["gcn_minus_mlp_val_acc"] for r in report["students"]])
                    ),
                    gcn_minus_mlp_val_ce_mean=float(
                        np.mean([r["gcn_minus_mlp_val_ce"] for r in report["students"]])
                    ),
                )
    except InterruptedError as exc:
        report.update(status="stopped", reason=str(exc))
    except (RuntimeError, ValueError, OSError) as exc:
        report.update(status="failed", error_type=type(exc).__name__, reason=str(exc))
    finally:
        if lock is not None:
            lock.close()
        report["elapsed_seconds"] = time.monotonic() - started
        report["peak_gpu_bytes"] = int(torch.cuda.max_memory_allocated(device)) if gpu_started else 0
        save_json(report, root / "report.json")
        report_key = (
            _fingerprint(dict(student=settings, report_routes=True, version=1))
            if report_routes else _fingerprint(settings)
        )
        save_json(report, root / f"report_step_{steps}_{report_key}.json")
        print("LARGE_PILOT_RESULT", json.dumps(report), flush=True)
    return report, root
