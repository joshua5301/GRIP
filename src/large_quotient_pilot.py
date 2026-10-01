"""Bounded, validation-only structural transfer of one immutable assignment.

No assignment, teacher, target or independent synthetic feature is optimized.
The two students receive the same saved Qc/mass and fresh seed: H means with
identity propagation, or original-X means with a raw-support graph quotient.
The caller serializes GPU jobs. Native tensor operations can finish before a
POSIX deadline is delivered; column chunks bound the quotient work unit.
"""

import hashlib
import io
import json
import math
import time
from numbers import Real
from pathlib import Path

import torch

from src.coarsening import assignment_probability, feature_centroids, quotient_adjacency
from src.data import _prepare_dataset
from src.inductive_evaluation import fit_inductive_gcn
from src.io import _fingerprint, save_json
from src.large_pilot import _data_digest, _validation_route_inputs, _wall_deadline
from src.moments import make_material
from src.student_routes import replay_routes
from src.sweep_utils import representative
from src.target_refinement import training_refined_targets
from src.transforms import FeatureTransform, fit_transform

_ROLES = {"resume", "initial_assignment", "endpoint", "teacher", "propagated_h", "candidate_protocol"}
_ARMS = ("h_identity", "raw_quotient")


def _guard(stop):
    if stop():
        raise InterruptedError("Frozen quotient pilot stopped or deadline reached")


def _integer(value, minimum=0):
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def _matrix(value, name):
    if (not torch.is_tensor(value) or value.ndim != 2 or min(value.shape) < 1
            or not value.is_floating_point() or not bool(torch.isfinite(value).all())):
        raise ValueError(f"{name} must be a finite nonempty floating matrix")


@torch.no_grad()
def raw_looped_support(packed_adj, *, atol=1e-7, rtol=1e-5, stop=lambda: False):
    """Return (B, checks), where B is binary support with original-node loops.

    This inference is valid only for this repository's unweighted, undirected
    data protocol. It verifies complete diagonal support and that normalizing B
    once recovers packed S. With singleton P, normalize(P.T B P) thus recovers S.
    B stays sparse; no original nodes-by-nodes dense tensor is allocated.
    """
    _guard(stop)
    if (not torch.is_tensor(packed_adj) or packed_adj.ndim != 2 or len(packed_adj) < 1
            or packed_adj.shape[0] != packed_adj.shape[1] or not packed_adj.is_floating_point()
            or not math.isfinite(atol) or not math.isfinite(rtol) or min(atol, rtol) < 0):
        raise ValueError("Require a square floating packed adjacency and finite tolerances")
    packed = packed_adj.to_sparse_coo().coalesce()
    indices, values = packed.indices(), packed.values()
    if not len(values) or not bool(torch.isfinite(values).all()) or bool((values <= 0).any()):
        raise ValueError("Packed adjacency requires finite positive support weights")
    row, col = indices
    nodes = packed.shape[0]
    if int((row == col).sum()) != nodes:
        raise ValueError("Packed support must contain exactly one self-loop per original node")
    support = torch.sparse_coo_tensor(indices, torch.ones_like(values), packed.shape).coalesce()
    if not torch.equal(indices, support.transpose(0, 1).coalesce().indices()):
        raise ValueError("Raw-support quotient requires an undirected symmetric support")
    _guard(stop)
    degree = torch.bincount(row, minlength=nodes).to(values)
    inverse = degree.rsqrt()
    normalized = inverse[row] * inverse[col]
    difference = (normalized - values).abs()
    if not torch.allclose(normalized, values, atol=atol, rtol=rtol):
        raise ValueError("Binary raw-looped support does not reproduce packed normalization")
    _guard(stop)
    return support.to_sparse_csr(), dict(
        nodes=nodes, nnz=len(values), normalization_max_abs=float(difference.max()),
        normalization_atol=atol, normalization_rtol=rtol,
        convention="B=undirected off-diagonal binary support+I_N; normalized exactly once",
    )


@torch.no_grad()
def load_frozen_assignment(resume, initial_assignment, endpoint, *, expected_step,
                           nodes=None, cells=None, device="cpu", chunk_size=None,
                           stop=lambda: False):
    """Decode already SHA-verified objects into the exact saved node-factor P.

    Only free-mass, unweighted, fixed-target linear node factors are supported.
    U/V retain their saved dtype during logit_block; logits then become float64
    before full-column softmax, matching LowRankMoments. resume.scale is the
    initial outer-CE gradient normalization, and never scales P or its logits.
    """
    _guard(stop)
    if (not isinstance(resume, dict) or not isinstance(endpoint, dict)
            or not _integer(expected_step) or resume.get("step") != expected_step
            or endpoint.get("step") != expected_step or endpoint.get("J_exact") is not True):
        raise ValueError("Require an exact endpoint and matching frozen resume step")
    config = resume.get("config", {})
    if (not isinstance(config, dict) or config.get("assignment_input", "node") != "node"
            or config.get("assignment_encoder", "linear") != "linear"
            or config.get("mass_mode", "free") != "free" or config.get("node_weighting", False)
            or config.get("feature_control", "joint") != "joint"
            or "temperature_logits_digest" in config or resume.get("dual") is not None
            or config.get("solver_mode", "exact") != "exact"):
        raise ValueError("Only exact, free-mass, fixed-target linear node-factor assignments are supported")
    parameters = resume.get("parameters")
    if not isinstance(parameters, (list, tuple)) or len(parameters) != 2:
        raise ValueError("Frozen node assignment must have exactly U and V parameters")
    u, v = parameters
    _matrix(u, "U")
    _matrix(v, "V")
    if (u.dtype != v.dtype or u.shape[1] != v.shape[1] or u.shape[1] > min(len(u), len(v))
            or (nodes is not None and len(u) != nodes) or (cells is not None and len(v) != cells)
            or not _integer(config.get("assignment_rank"), 1)
            or config["assignment_rank"] != u.shape[1]):
        raise ValueError("Frozen U/V shapes or rank differ from the source protocol")
    if (not torch.is_tensor(initial_assignment) or initial_assignment.shape != (len(u),)
            or initial_assignment.dtype not in (torch.int32, torch.int64)
            or bool((initial_assignment < 0).any()) or bool((initial_assignment >= len(v)).any())):
        raise ValueError("Initial assignment must contain one valid original-node cell index")
    mixing = config.get("mixing")
    scale = resume.get("scale")
    if (isinstance(mixing, bool) or not isinstance(mixing, Real) or not math.isfinite(mixing)
            or not 0 < mixing < 1 or isinstance(scale, bool) or not isinstance(scale, Real)
            or not math.isfinite(scale) or scale <= 0):
        raise ValueError("Frozen assignment mixing and objective scale must be finite and positive")
    snapshot = resume.get("snapshots", {}).get(expected_step)
    moments = endpoint.get("moments")
    _matrix(moments, "Endpoint moments")
    if (not isinstance(snapshot, dict) or snapshot.get("step") != expected_step
            or snapshot.get("J_exact") is not True
            or not torch.is_tensor(snapshot.get("moments"))
            or not torch.equal(snapshot["moments"], moments) or len(moments) != len(v)):
        raise ValueError("Frozen parameters must match the exact saved endpoint moments")
    chunk_size = config.get("chunk_size", 2048) if chunk_size is None else chunk_size
    if not _integer(chunk_size, 1):
        raise ValueError("Assignment replay chunk size must be a positive integer")
    saved = dict(u=u.detach().to(device), v=v.detach().to(device),
                 assignment=initial_assignment.to(device=device, dtype=torch.long), mixing=mixing)
    _guard(stop)
    probability = torch.empty((len(u), len(v)), device=device, dtype=torch.float64)
    for start in range(0, len(u), chunk_size):
        _guard(stop)
        block = dict(saved, u=saved["u"][start:start + chunk_size],
                     assignment=saved["assignment"][start:start + chunk_size])
        probability[start:start + chunk_size] = assignment_probability(
            block, chunk_size=chunk_size, dtype=torch.float64)
        _guard(stop)
    _guard(stop)
    return dict(probability=probability, node_mass=probability.sum(0), config=dict(config),
                step=expected_step, scale=float(scale), diagnostics=dict(
                    nodes=len(u), cells=len(v), rank=u.shape[1], parameter_dtype=str(u.dtype),
                    probability_dtype=str(probability.dtype), mixing=float(mixing),
                    objective_scale=float(scale), objective_scale_applied_to_probability=False))


def _close(actual, expected, name, *, atol=1e-8, rtol=1e-5):
    if actual.shape != expected.shape or not torch.allclose(actual, expected, atol=atol, rtol=rtol):
        raise ValueError(f"Replayed {name} differ from the frozen endpoint")
    return dict(max_abs=float((actual - expected).abs().max()), atol=atol, rtol=rtol)


def _quotient_with_stop(probability, support, column_chunk, stop):
    """Poll between sparse column products, then reuse the quotient normalizer."""
    support = support.to(probability)
    cells = probability.shape[1]
    coarse = probability.new_empty(cells, cells)
    for start in range(0, cells, column_chunk):
        _guard(stop)
        block = probability[:, start:start + column_chunk]
        propagated = (support @ block if support.layout == torch.strided else
                      torch.sparse.mm(support, block))
        coarse[:, start:start + column_chunk] = probability.T @ propagated
        _guard(stop)
    # Singleton identity leaves the already constructed raw C unchanged while
    # applying the shared retain-diagonal symmetric normalization convention.
    identity = torch.eye(cells, device=probability.device, dtype=probability.dtype)
    normalized = quotient_adjacency(identity, coarse, normalization="symmetric",
                                     mass_scaling=False, self_loops="retain", chunk_size=cells)
    _guard(stop)
    return normalized


@torch.no_grad()
def frozen_pair_inputs(probability, x, h, q, endpoint, *, transform=None, column_chunk=64,
                       packed_adjacency=None, stop=lambda: False):
    """Verify original-H/Q moments, then build the two fixed-P student inputs."""
    _guard(stop)
    if not _integer(column_chunk, 1) or packed_adjacency is None:
        raise ValueError("Require a positive column chunk and the original packed adjacency")
    _matrix(probability, "P")
    probability = probability.detach().double()
    for value, name in ((x, "Original X"), (h, "Frozen H"), (q, "Frozen teacher Q")):
        _matrix(value, name)
    if x.shape != h.shape or len(q) != len(x) or len(probability) != len(x):
        raise ValueError("P, original X/H and teacher Q must share the original-node ordering")
    h = h.detach().to(device=probability.device, dtype=torch.float64)
    q = q.detach().to(h)
    if bool((q < 0).any()) or not torch.allclose(q.sum(1), torch.ones(len(q), device=q.device,
                                                                        dtype=q.dtype), atol=1e-8, rtol=1e-6):
        raise ValueError("Teacher targets must be row-stochastic probabilities")
    if transform is None:
        z, transform = fit_transform(h, kind="rms")
    else:
        if transform.kind != "rms" or transform.matrix is not None:
            raise ValueError("Replay requires the fixed original-H RMS transform")
        transform = FeatureTransform(**{
            key: value.to(h) if torch.is_tensor(value) else value
            for key, value in vars(transform).items()
        })
        z = transform(h)
    saved = endpoint["moments"].to(h)
    if saved.shape != (probability.shape[1], 1 + x.shape[1] + q.shape[1]):
        raise ValueError("Frozen endpoint moment dimensions differ from original H/Q")
    recomputed = torch.zeros_like(saved)
    for start in range(0, len(h), 2048):
        _guard(stop)
        end = start + 2048
        recomputed.add_(probability[start:end].T @ make_material(z[start:end], q[start:end]),
                        alpha=1 / len(h))
    moment_check = _close(recomputed, saved, "H/Q moments")
    baseline_x, labels, mass = representative(saved, transform, x.shape[1], h.device)
    h_mean, node_mass = feature_centroids(probability, h)
    q_mean, _ = feature_centroids(probability, q)
    diagnostics = dict(moments=moment_check,
                       normalized_mass=_close(node_mass / len(h), mass, "normalized mass"),
                       H_centroids=_close(h_mean, baseline_x.double(), "H centroids", atol=1e-6),
                       Q_centroids=_close(q_mean, labels.double(), "Q centroids", atol=1e-6))
    _guard(stop)
    raw_x, _ = feature_centroids(probability, x)
    support, support_check = raw_looped_support(packed_adjacency, stop=stop)
    _guard(stop)
    coarse = _quotient_with_stop(probability, support, column_chunk, stop)
    _guard(stop)
    asymmetry = float((coarse - coarse.T).norm() / coarse.norm().clamp_min(1e-30))
    if not bool(torch.isfinite(coarse).all()) or bool((coarse < 0).any()) or asymmetry > 1e-8:
        raise ValueError("Coarse adjacency must be finite, nonnegative and symmetric")
    coarse_h = coarse @ (coarse @ raw_x)
    difference = coarse_h - h_mean
    diagnostics.update(raw_support=support_check, coarse_asymmetry_relative=asymmetry,
                       coarse_nnz=int((coarse != 0).sum()),
                       coarse_vs_Hmean_relative=float(difference.norm() / h_mean.norm().clamp_min(1e-30)),
                       coarse_vs_Hmean_mass_weighted_relative=float(
                           (mass[:, None] * difference.square()).sum().sqrt()
                           / (mass[:, None] * h_mean.square()).sum().sqrt().clamp_min(1e-30)))
    return dict(h_identity=dict(x=baseline_x, labels=labels, mass=mass, adj=None),
                raw_quotient=dict(x=raw_x.float(), labels=labels, mass=mass, adj=coarse.float()),
                diagnostics=diagnostics)


def _read_manifest(path, expected_sha256):
    payload = Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("Frozen quotient source manifest changed")
    manifest = json.loads(payload)
    required = {"version", "dataset", "ratio", "step", "source_data_digest", "student_recipe", "files"}
    if (not isinstance(manifest, dict) or not required <= manifest.keys() or manifest["version"] != 1
            or manifest["dataset"] not in ("arxiv", "flickr", "reddit")
            or not _integer(manifest["step"]) or not manifest["source_data_digest"]
            or isinstance(manifest["ratio"], bool) or not isinstance(manifest["ratio"], Real)
            or not math.isfinite(manifest["ratio"]) or not 0 < manifest["ratio"] <= 1
            or not isinstance(manifest["files"], dict)
            or not _ROLES <= manifest["files"].keys()
            or set(manifest["files"]) - (_ROLES | {"teacher_protocol"})):
        raise ValueError("Require a complete version1 frozen quotient source manifest")
    settings = manifest["student_recipe"]
    if (not isinstance(settings, dict)
            or set(settings) != {"epochs", "eval_every", "hidden", "dropout", "lr", "weight_decay"}
            or any(not _integer(settings[k], 1) for k in ("epochs", "eval_every", "hidden"))
            or any(isinstance(settings[k], bool) or not isinstance(settings[k], Real)
                   or not math.isfinite(settings[k]) for k in ("dropout", "lr", "weight_decay"))
            or not 0 <= settings["dropout"] < 1 or settings["lr"] <= 0 or settings["weight_decay"] < 0):
        raise ValueError("Require a valid complete frozen student recipe")
    return manifest


def _load_files(manifest, manifest_path, guard):
    loaded = {}
    for role, record in sorted(manifest["files"].items()):
        guard()
        if not isinstance(record, dict) or not {"path", "sha256"} <= record.keys():
            raise ValueError(f"Missing frozen {role} file provenance")
        path = Path(record["path"])
        if not path.is_absolute():
            path = Path(manifest_path).parent / path
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != record["sha256"]:
            raise ValueError(f"Frozen {role} file SHA256 differs")
        loaded[role] = (json.loads(payload) if role.endswith("protocol") else
                        torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False))
        guard()
    return loaded


def _implementation():
    from src.research_loop import implementation_provenance

    return implementation_provenance()


def run_quotient_pilot(source_manifest_path, source_manifest_sha256, output_dir, *,
                       student_seeds=(0, 1, 2), data_dir="data", device="cuda", column_chunk=64,
                       deadline_seconds=300, stop=lambda: False):
    """Replay frozen P, then compare paired fixed-input students on validation.

    Completed seed fits/replays resume from fingerprinted caches. An unfinished
    fit restarts from its fixed seed. No testing graph is passed to fitting or
    serving. The existing dataset loader eagerly packs its unused test split,
    which is discarded immediately and never consulted by this diagnostic.
    """
    manifest = _read_manifest(source_manifest_path, source_manifest_sha256)
    if (not isinstance(student_seeds, (list, tuple)) or not student_seeds
            or any(not _integer(seed) for seed in student_seeds)
            or len(set(student_seeds)) != len(student_seeds) or not _integer(column_chunk, 1)
            or isinstance(deadline_seconds, bool) or not isinstance(deadline_seconds, Real)
            or not math.isfinite(deadline_seconds) or deadline_seconds <= 0):
        raise ValueError("Require distinct valid student seeds, a positive chunk and finite deadline")
    implementation = _implementation()
    protocol = dict(version=1, source_manifest_sha256=source_manifest_sha256,
                    source_data_digest=manifest["source_data_digest"], source_files=manifest["files"],
                    source_digest=implementation["source_digest"], student_recipe=manifest["student_recipe"],
                    arms=list(_ARMS), feature_modes=["frozen_Pmean_H", "frozen_Pmean_original_X"],
                    adjacency="normalize_symmetric(P.T raw_looped_binary_support P);retain;no_mass_scaling",
                    probability="saved-dtype logits including prior + UV/sqrt(rank); float64 softmax",
                    column_chunk=column_chunk, device=str(device), torch_version=str(torch.__version__),
                    tf32=False, cudnn_tf32=False, torch_threads=4,
                    selection="validation_only; first maximum GCN validation epoch; uniform student CE")
    root = (Path(output_dir) / manifest["dataset"] / f"source_{source_manifest_sha256[:16]}"
            / f"protocol_{_fingerprint(protocol)}")
    root.mkdir(parents=True, exist_ok=True)
    protocol_path = root / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Cached quotient pilot protocol changed")
    save_json(protocol, protocol_path)
    save_json(manifest, root / "source_manifest.json")
    report = dict(status="starting", stage="verifying_frozen_files", dataset=manifest["dataset"],
                  ratio=manifest["ratio"], step=manifest["step"], rows=[], paired_deltas=[],
                  selection="validation_only", source_manifest_sha256=source_manifest_sha256,
                  source_code_provenance=implementation, student_recipe=manifest["student_recipe"],
                  student_seeds=list(student_seeds), protocol_fingerprint=_fingerprint(protocol))
    started = time.monotonic()
    deadline = started + deadline_seconds
    gpu_started = False
    invocation = dict(student_seeds=list(student_seeds), device=str(device),
                      data_dir=str(Path(data_dir).resolve()), deadline_seconds=deadline_seconds)
    invocation_path = root / f"invocation_{_fingerprint(invocation)}_{time.time_ns()}.json"
    report["invocation"] = invocation
    report["invocation_report_path"] = str(invocation_path.resolve())
    previous_threads = torch.get_num_threads()
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    previous_cudnn_tf32 = torch.backends.cudnn.allow_tf32

    def stopped():
        return bool(stop() or time.monotonic() >= deadline or (root / "STOP").exists())

    def guard():
        _guard(stopped)
        _read_manifest(source_manifest_path, source_manifest_sha256)

    def stage(name, **details):
        guard()
        report.update(stage=name, elapsed_seconds=time.monotonic() - started, **details)
        print("QUOTIENT_PILOT_STAGE", json.dumps(dict(stage=name, **details)), flush=True)
        save_json(report, root / "report.json")

    try:
        with _wall_deadline(deadline_seconds):
            guard()
            torch.set_num_threads(4)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            if str(device).startswith("cuda"):
                torch.cuda.reset_peak_memory_stats(device)
                gpu_started = True
                report["gpu"] = torch.cuda.get_device_name(device)
            files = _load_files(manifest, source_manifest_path, guard)
            candidate_protocol = files["candidate_protocol"]
            candidate = candidate_protocol["candidate"]
            if (candidate_protocol["data_digest"] != manifest["source_data_digest"]
                    or candidate.get("assignment") != "low_rank" or candidate.get("surrogate", "linear") != "linear"
                    or candidate.get("ratio") != manifest["ratio"] or not _integer(candidate.get("cells"), 1)):
                raise ValueError("Frozen candidate protocol is not the selected linear low-rank source")
            teacher = files["teacher"]
            if (not isinstance(teacher, dict) or "logits" not in teacher
                    or teacher.get("converged", True) is not True
                    or teacher.get("training_complete", True) is not True
                    or teacher.get("data_digest", manifest["source_data_digest"]) != manifest["source_data_digest"]):
                raise ValueError("Frozen teacher differs from the selected converged source")
            teacher_protocol = files.get("teacher_protocol")
            if (teacher_protocol is not None and
                    teacher_protocol.get("data_digest") != manifest["source_data_digest"]):
                raise ValueError("Frozen teacher protocol input digest differs")
            stage("loading_source_graph")
            graph, train, validation, unused_testing, unused_h = _prepare_dataset(
                manifest["dataset"], data_dir, device)
            del unused_testing, unused_h
            h = files["propagated_h"]["h"].to(device)
            _matrix(h, "Frozen propagated H")
            if h.shape != graph["x"].shape or h.dtype != graph["x"].dtype:
                raise ValueError("Frozen propagated H shape/dtype differs from original X")
            stage("verifying_source_data")
            if _data_digest(graph, train, validation, h, guard) != manifest["source_data_digest"]:
                raise ValueError("Original source graph/H/validation digest differs from the frozen manifest")
            logits = teacher["logits"].to(device)
            q = training_refined_targets(logits, candidate["temperature"], graph["y"], train,
                                         mixing=candidate.get("train_target_mix", 0.0))
            del logits
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()
                free, total = torch.cuda.mem_get_info(device)
                # Two P-sized buffers plus original-H/X double working copies.
                nnz = (int((graph["adj"] != 0).sum()) if graph["adj"].layout == torch.strided
                       else graph["adj"]._nnz())
                needed = (16 * len(h) * candidate["cells"] + 24 * h.numel()
                          + 8 * len(h) * min(column_chunk, candidate["cells"])
                          + 64 * nnz + 8 * len(h) + 8 * candidate["cells"] ** 2)
                report["construction_headroom_estimate_bytes"] = needed
                if needed + max(2 * 2**30, total // 4) > free:
                    raise RuntimeError("Insufficient GPU headroom for frozen dense-P quotient construction")
            stage("replaying_frozen_assignment")
            decoded = load_frozen_assignment(
                files["resume"], files["initial_assignment"], files["endpoint"],
                expected_step=manifest["step"], nodes=len(h), cells=candidate["cells"],
                device=device, stop=stopped)
            if decoded["config"]["mixing"] != candidate.get("mixing", 0.05):
                raise ValueError("Frozen prior mixing differs from the candidate protocol")
            for candidate_key, config_key in (("rank", "assignment_rank"),
                                               ("condensation_seed", "factor_seed"),
                                               ("penalty", "penalty"), ("lr", "lr")):
                if candidate_key in candidate and candidate[candidate_key] != decoded["config"].get(config_key):
                    raise ValueError("Frozen assignment configuration differs from the candidate protocol")
            stage("constructing_fixed_pair")
            pair = frozen_pair_inputs(decoded["probability"], graph["x"], h.double(), q,
                                      files["endpoint"], column_chunk=column_chunk,
                                      packed_adjacency=graph["adj"], stop=stopped)
            report["construction"] = dict(assignment=decoded["diagnostics"], **pair["diagnostics"])
            del decoded, q, h, files, teacher
            val_graph, val_h, masks = _validation_route_inputs(
                validation, manifest["source_data_digest"], root / "validation_geometry", guard)
            for seed in student_seeds:
                completed = {}
                for arm in _ARMS:
                    stage("validation_student", arm=arm, student_seed=seed)
                    inputs = pair[arm]
                    folder = root / arm
                    result = fit_inductive_gcn(
                        inputs["x"], inputs["labels"], inputs["mass"], graph, validation,
                        seed=seed, settings=manifest["student_recipe"], folder=folder,
                        training_adjacency=inputs["adj"], weighting="uniform", train_mask=train,
                        stop=stopped)
                    if any("test_" in key for key in result):
                        raise ValueError("Frozen quotient validation fit unexpectedly returned test metrics")
                    guard()
                    routes = replay_routes(
                        folder / f"seed_{seed}_selected.pt", val_graph, val_h, masks,
                        manifest["student_recipe"], folder / f"seed_{seed}_validation_routes.json",
                        seed=seed, stop=stopped)
                    if (any("test_" in key for key in routes)
                            or routes["epoch"] != result["epoch"]
                            or not math.isclose(routes["gcn_val_acc"], result["val_acc"], abs_tol=1e-8, rel_tol=0)
                            or not math.isclose(routes["gcn_val_ce"], result["val_ce"], abs_tol=1e-6, rel_tol=1e-6)):
                        raise ValueError("Validation route replay differs from its selected student")
                    row = {"arm": arm, **result, **routes}
                    completed[arm] = row
                    report["rows"].append(row)
                    save_json(report, root / "report.json")
                report["paired_deltas"].append(dict(
                    seed=seed,
                    gcn_val_acc_pp=completed["raw_quotient"]["gcn_val_acc"] - completed["h_identity"]["gcn_val_acc"],
                    mlp_val_acc_pp=completed["raw_quotient"]["mlp_val_acc"] - completed["h_identity"]["mlp_val_acc"],
                    gcn_val_ce=completed["raw_quotient"]["gcn_val_ce"] - completed["h_identity"]["gcn_val_ce"],
                    mlp_val_ce=completed["raw_quotient"]["mlp_val_ce"] - completed["h_identity"]["mlp_val_ce"]))
            guard()
            report.update(status="complete", stage="complete")
    except InterruptedError as exc:
        report.update(status="stopped", reason=str(exc))
    except (ValueError, RuntimeError, OSError, KeyError, TypeError, AttributeError) as exc:
        report.update(status="failed", reason=str(exc), error_type=type(exc).__name__)
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        report["peak_gpu_bytes"] = int(torch.cuda.max_memory_allocated(device)) if gpu_started else 0
        save_json(report, root / "report.json")
        save_json(report, invocation_path)
        torch.set_num_threads(previous_threads)
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
        torch.backends.cudnn.allow_tf32 = previous_cudnn_tf32
    return report, root
