"""Source-only fixed assignment coordinates; representatives/outer keep z/Q.

Linear output bias is trainable. There is no mass target, mean centering or
map refit. Native cache reuse requires the original device; CPU inspection of
CUDA snapshots is an audit, not a supported production replay.
"""
import hashlib
import inspect
import json
import math
from numbers import Integral, Real
from pathlib import Path

import numpy as np
import torch

from src.io import array_digest
from src.low_rank_assignment import LowRankMoments, logit_block
from src.mlp_initial_mass import _step, _tensor, digest, seal
from src.moments import augmented, decode_moments, make_material

SCHEMA = 1
MODES = ("raw_rms", "nystrom_relu")
_KEYS = {"assignment_coordinates", "source_linear_schema", "source_linear_source_digest"}
_ALLOWED = _KEYS | {"method", "width", "rank", "lr", "T", "penalty", "initialization", "alpha",
                    "inner_loss_weighting", "mixing", "mass_mode"}
_FORBIDDEN = {"mass_target", "target_masses", "initial_mass_target", "initial_mass_context", "column_dual",
              "external_coordinates", "source_coordinates", "outer_targets", "outer_indices", "initial_representatives",
              "center_mean", "raw_output_mean", "source_centering_context", "mlp_output_centering"}
_DEPENDENCIES = ("source_linear_assignment.py", "soft_ce_partition.py", "citation_search.py", "low_rank_assignment.py",
                 "moments.py", "head.py", "mlp_initial_mass.py", "target_refinement.py", "transforms.py",
                 "partition_initialization.py", "data.py", "shared_features.py", "teacher.py", "io.py", "nystrom_ce.py")


def source_digest():
    return {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest() for name in _DEPENDENCIES}


def active(candidate):
    return isinstance(candidate, dict) and (candidate.get("method") == "source_linear" or any(
        isinstance(key, str) and key.startswith(("source_linear", "assignment_coordinate")) for key in candidate))


def candidate_controls(candidate):
    if not isinstance(candidate, dict):
        raise ValueError("Source linear candidate must be a mapping")
    candidate = dict(candidate)
    if not active(candidate):
        return candidate
    if candidate.get("method") != "source_linear" or set(candidate) - _ALLOWED:
        raise ValueError("Source linear controls do not permit external coordinates/targets/modes or unknown options")
    if candidate.get("assignment_coordinates") not in MODES:
        raise ValueError("assignment_coordinates must be raw_rms or nystrom_relu")
    if (isinstance(candidate.get("source_linear_schema"), bool)
            or not isinstance(candidate.get("source_linear_schema"), Integral)
            or candidate["source_linear_schema"] != SCHEMA):
        raise ValueError("Source linear requires explicit source_linear_schema=1")
    candidate["source_linear_schema"] = SCHEMA
    for key in ("penalty", "lr", "T"):
        x = candidate.get(key)
        if isinstance(x, bool) or not isinstance(x, Real) or not math.isfinite(x) or x <= 0:
            raise ValueError("Source linear requires positive finite penalty/lr/T")
        candidate[key] = float(x)
    for key in ("rank", "width"):
        x = candidate.get(key)
        if isinstance(x, bool) or not isinstance(x, Integral) or (x < 1 if key == "rank" else x != 0):
            raise ValueError("Source linear requires positive integer rank and width=0")
        candidate[key] = int(x)
    if (candidate.get("mass_mode", "free") != "free" or candidate.get("inner_loss_weighting") != "uniform"
            or candidate.get("mixing", .05) != .05 or isinstance(candidate.get("mixing", .05), bool)):
        raise ValueError("Source linear supports only free mass, mixing .05 and uniform CE")
    if candidate.get("initialization") not in ("feature", "teacher_joint", "teacher_balanced"):
        raise ValueError("Source linear requires an existing supported hard initializer")
    if candidate["initialization"] != "feature":
        x = candidate.get("alpha")
        if isinstance(x, bool) or not isinstance(x, Real) or not math.isfinite(x) or x <= 0:
            raise ValueError("Source linear requires finite positive initializer alpha")
    candidate.pop("mass_mode", None)
    candidate.pop("mixing", None)
    token = seal(source_digest())
    if candidate.get("source_linear_source_digest", token) != token:
        raise ValueError("Source linear numerical helper source changed; preserve its old namespace")
    candidate["source_linear_source_digest"] = token
    return candidate


def native_environment():
    if (torch.get_default_dtype() != torch.float32 or torch.is_autocast_enabled()
            or torch.is_autocast_enabled("cpu") or torch.backends.cuda.matmul.allow_tf32):
        raise ValueError("Source linear requires native FP32 parameters without autocast or TF32")


def initialize_linear(inputs, cells, rank, seed=0):
    from src.low_rank_assignment import initialize_encoder
    a, v = initialize_encoder(inputs, cells, rank, seed)
    return [a, inputs.new_zeros(rank).requires_grad_()], v


def linear_nodes(inputs, parameters):
    """Keep F frozen; only A and b receive encoder gradients."""
    a, b = parameters
    result = inputs.detach() @ a + b
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError("Nonfinite source-linear node encoding")
    return result


def _coordinate_tensor(inputs, z, dimension):
    _tensor(inputs, (len(z), dimension), torch.float32, "source assignment coordinates")
    if inputs.device != z.device or inputs.requires_grad:
        raise ValueError("Source assignment coordinates must be detached on the original source device")


@torch.no_grad()
def original_context(z, q, assignment, inputs, rank, seed, mixing, chunk_size, config, source=None):
    native_environment()
    if (not all(torch.is_tensor(v) for v in (z, q, assignment, inputs))
            or z.ndim != 2 or q.ndim != 2 or inputs.ndim != 2
            or z.dtype != torch.float64 or q.dtype != torch.float64 or z.ndim != 2 or q.ndim != 2
            or len(z) != len(q) or assignment.shape != (len(z),) or assignment.dtype != torch.int64
            or min(z.shape) < 1 or min(q.shape) < 1 or not bool(torch.isfinite(z).all())
            or not bool(torch.isfinite(q).all()) or bool((q < 0).any())
            or float((q.sum(1) - 1).abs().max()) > 1e-12 or z.device != q.device
            or z.device != assignment.device or bool((assignment < 0).any())
            or type(chunk_size) is not int or chunk_size < 1 or torch.get_default_dtype() != torch.float32
            or not isinstance(source, dict) or source.get("coordinate_mode") not in MODES):
        raise ValueError("Source linear requires aligned original float64 z/Q and verified source coordinates")
    _coordinate_tensor(inputs, z, inputs.shape[1])
    if digest(inputs) != source.get("coordinates_fp32"):
        raise ValueError("Source coordinates differ from their verified original cache")
    if source["coordinate_mode"] == "raw_rms" and not torch.equal(inputs, z.detach().float()):
        raise ValueError("Raw source coordinates must be the original RMS z.float()")
    # A caller-provided F digest alone is not source evidence: replay the
    # require-existing cache and compare all material, partition and F values.
    # This does not fit/recompute a map, Phi, teacher, partition or head.
    try:
        root = Path(source["root"])
        source_h = torch.load(root / "propagated_H.pt", map_location=z.device, weights_only=False)["h"]
        _, original_z, _, original_hard, original_inputs, replayed_source = cached_source(
            root, source["candidate"], seed, source_h, q, source["config"])
    except (KeyError, TypeError, OSError, RuntimeError) as error:
        raise ValueError("Source linear original cache evidence is missing or malformed") from error
    if (seal(source) != seal(replayed_source) or digest(z) != source.get("z") or digest(q) != source.get("q")
            or digest(assignment) != source.get("hard_assignment") or not torch.equal(z, original_z)
            or not torch.equal(assignment, original_hard) or not torch.equal(inputs, original_inputs)):
        raise ValueError("Source linear material/partition/coordinates differ from the original source cache")
    cells = int(assignment.max()) + 1
    if len(torch.unique(assignment)) != cells:
        raise ValueError("Original hard partition must contain every cell")
    encoder, v = initialize_linear(inputs, cells, rank, seed)
    u = linear_nodes(inputs, encoder)
    context = dict(schema=SCHEMA, coordinate_mode=source["coordinate_mode"], device=str(z.device),
                   nodes=len(z), cells=cells, dimension=z.shape[1], coordinate_dimension=inputs.shape[1],
                   classes=q.shape[1], rank=rank, seed=seed, mixing=mixing, chunk_size=chunk_size,
                   z=digest(z), q=digest(q), assignment=digest(assignment), coordinates=digest(inputs),
                   initializer=[digest(p) for p in [*encoder, v]], encoded_u0=digest(u), source=source,
                   controls=config, helper_source=source_digest(), torch_version=str(torch.__version__),
                   default_dtype=str(torch.get_default_dtype()),
                   native_precision=dict(parameter_dtype="torch.float32", autocast=False, matmul_tf32=False))
    return context, [*encoder, v]


def _optimizer(saved, initial_parameters, step, lr):
    optimizer = saved.get("optimizer")
    if not isinstance(optimizer, dict) or set(optimizer) != {"state", "param_groups"}:
        raise ValueError("Malformed source-linear Adam state")
    groups, states = optimizer["param_groups"], optimizer["state"]
    if not isinstance(groups, list) or len(groups) != 1 or not isinstance(states, dict):
        raise ValueError("Malformed source-linear Adam groups")
    group = groups[0]
    if not isinstance(group, dict):
        raise ValueError("Malformed source-linear Adam group")
    expected = dict(lr=lr, betas=[.9, .999], eps=1e-12, weight_decay=0, amsgrad=False,
                    maximize=False, foreach=False, capturable=False, differentiable=False, fused=None,
                    decoupled_weight_decay=False)
    # Torch versions differ only in whether the unchanged default field exists.
    if "decoupled_weight_decay" not in group:
        expected.pop("decoupled_weight_decay")
    if set(group) != set(expected) | {"params"} or any(group.get(key) != value for key, value in expected.items()):
        raise ValueError("Cached Adam controls differ from the fixed recipe")
    for key in ("lr", "eps", "weight_decay"):
        if isinstance(group[key], bool) or not isinstance(group[key], Real) or not math.isfinite(group[key]):
            raise ValueError("Malformed numeric Adam control")
    if (not isinstance(group["betas"], list) or len(group["betas"]) != 2
            or any(isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value)
                   for value in group["betas"])):
        raise ValueError("Malformed Adam beta controls")
    for key in ("amsgrad", "maximize", "foreach", "capturable", "differentiable", "decoupled_weight_decay"):
        if key in group and type(group[key]) is not bool:
            raise ValueError("Malformed boolean Adam control")
    if (not isinstance(group["params"], list) or any(type(value) is not int for value in group["params"])
            or group["params"] != list(range(3)) or any(type(key) is not int for key in states)):
        raise ValueError("Malformed Adam parameter identifiers")
    if group["params"] != list(range(3)) or set(states) != (set(range(3)) if step else set()):
        raise ValueError("Cached Adam parameter IDs or update counts differ")
    for index, value in states.items():
        if not isinstance(value, dict) or set(value) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError("Malformed Adam moments")
        counter = value["step"]
        if (not torch.is_tensor(counter) or counter.ndim != 0 or counter.dtype != torch.float32
                or not bool(torch.isfinite(counter))
                or float(counter) != step):
            raise ValueError("Adam update counter differs from saved step")
        for key in ("exp_avg", "exp_avg_sq"):
            _tensor(value[key], initial_parameters[index].shape, initial_parameters[index].dtype, key)
        if bool((value["exp_avg_sq"] < 0).any()):
            raise ValueError("Negative Adam second moment")


def attach(saved, context, parameters=None, u=None):
    saved.update(mass_mode="free", assignment_coordinates=context["coordinate_mode"], source_linear_schema=SCHEMA,
                 source_linear_context=context)
    if parameters is not None:
        saved["parameters"] = [p.detach().cpu().clone() for p in parameters]
    if u is not None:
        saved["encoded_u_digest"] = digest(u)
    saved["source_linear_content_sha256"] = seal({key: value for key, value in saved.items()
                                                   if key != "source_linear_content_sha256"})
    return saved


def _identity(saved, context):
    if not isinstance(saved, dict):
        raise ValueError("Source linear cache must be a mapping")
    if (saved.get("mass_mode") != "free" or saved.get("assignment_coordinates") != context["coordinate_mode"]
            or type(saved.get("source_linear_schema")) is not int or saved["source_linear_schema"] != SCHEMA
            or set(saved) & (_FORBIDDEN | {"column_dual", "initial_mass_context"})):
        raise ValueError("Invalid source linear mode/schema/external mean or target")
    if seal(saved.get("source_linear_context")) != seal(context):
        raise ValueError("Source linear source/initializer/device/controls changed")
    if saved.get("source_linear_content_sha256") != seal({key: value for key, value in saved.items()
                                                           if key != "source_linear_content_sha256"}):
        raise ValueError("Source linear content checksum mismatch")


def _parameters(saved, original):
    parameters = saved.get("parameters")
    if not isinstance(parameters, (list, tuple)) or len(parameters) != 3:
        raise ValueError("Source linear must retain all three native parameters")
    for parameter, initial in zip(parameters, original, strict=True):
        _tensor(parameter, initial.shape, initial.dtype, "linear parameter")
    return parameters


@torch.no_grad()
def validate_snapshot(saved, z, q, assignment, inputs, context, initial_parameters):
    _identity(saved, context)
    step = _step(saved.get("step"))
    parameters = _parameters(saved, initial_parameters)
    if step == 0 and [digest(p) for p in parameters] != context["initializer"]:
        raise ValueError("Linear checkpoint zero differs from its original initializer")
    k, d, c = context["cells"], context["dimension"], context["classes"]
    _tensor(saved.get("moments"), (k, 1 + d + c), torch.float64, "linear moments")
    _tensor(saved.get("theta"), (c, 1 + d), torch.float64, "linear head")
    _coordinate_tensor(inputs, z, context["coordinate_dimension"])
    if digest(inputs) != context["coordinates"]:
        raise ValueError("Linear assignment-only coordinates changed")
    u = linear_nodes(inputs, [p.to(z.device) for p in parameters[:-1]])
    if saved.get("encoded_u_digest") != digest(u):
        raise ValueError("Linear snapshot U differs from its current A/b/source coordinates")
    replay = LowRankMoments.apply(u, parameters[-1].to(z.device), assignment, make_material(z, q),
                                 context["mixing"], context["chunk_size"])
    if not bool(torch.isfinite(replay).all()) or bool((replay[:, 0] <= 0).any()):
        raise ValueError("Invalid free-mass source-derived linear moments")
    for start in range(0, len(z), context["chunk_size"]):
        probability = logit_block(u[start:start + context["chunk_size"]], parameters[-1].to(z.device),
                                  assignment[start:start + context["chunk_size"]], context["mixing"]).double().softmax(1)
        if float((probability.sum(1) - 1).abs().max()) > 1e-12:
            raise ValueError("Linear probability row stochasticity failed")
    if float((replay.sum(0) - make_material(z, q).mean(0)).abs().max()) > 1e-12:
        raise ValueError("Linear source material conservation failed")
    if not torch.equal(replay.cpu(), saved["moments"].cpu()):
        raise ValueError("Linear snapshot moments do not replay from its actual parameters")
    from src.soft_ce_partition import head_gradient, outer_value_gradient
    centers, labels, mass = decode_moments(replay, d)
    maximum = float(head_gradient(augmented(centers), labels, torch.full_like(mass, 1 / k),
                                  saved["theta"].to(z), context["controls"]["penalty"]).abs().max())
    if (saved.get("J_exact") is not True or not math.isfinite(maximum)
            or maximum > context["controls"]["inner_tol"] * (1 + 1e-6)):
        raise ValueError("Linear checkpoint CE head is uncertified")
    # Certificates replay the actual fixed source/head, rather than trusting a
    # resealed pair of history J0 and normalization scale. Scalar replay allows
    # only double arithmetic roundoff; it never refits a head or changes P.
    actual_outer, _ = outer_value_gradient(z, q, saved["theta"].to(z),
                                            context["controls"]["outer_chunk_size"], augmented(z))
    for key, actual in (("teacher_ce", actual_outer), ("inner_grad_max", maximum)):
        scalar = saved.get(key)
        if (type(scalar) is not float or not math.isfinite(scalar) or scalar < 0
                or not math.isclose(scalar, actual, rel_tol=1e-12, abs_tol=1e-12)):
            raise ValueError("Linear saved outer CE/stationarity certificate differs from its actual head/source")
    return replay


def validate_resume(saved, z, q, assignment, inputs, context, initial_parameters, steps, folder=None):
    _identity(saved, context)
    step = _step(saved.get("step"))
    if step > steps or saved.get("config") != context["controls"]:
        raise ValueError("Linear resume configuration/budget differs")
    _optimizer(saved, initial_parameters, step, context["controls"]["lr"])
    for state in saved["optimizer"]["state"].values():
        if state["step"].device.type != "cpu":
            raise ValueError("Native noncapturable Adam counter must be CPU")
    snapshots = saved.get("snapshots")
    if not isinstance(snapshots, dict) or 0 not in snapshots or step not in snapshots:
        raise ValueError("Linear resume requires original/current verifiable snapshots")
    for key, snapshot in snapshots.items():
        if not isinstance(snapshot, dict) or type(key) is not int or key > step or snapshot.get("step") != key:
            raise ValueError("Malformed linear snapshot history")
        validate_snapshot(snapshot, z, q, assignment, inputs, context, initial_parameters)
    parameters = _parameters(saved, initial_parameters)
    if any(not torch.equal(p, current) for p, current in zip(parameters, snapshots[step]["parameters"], strict=True)):
        raise ValueError("Linear resume parameters differ from its current snapshot")
    if saved.get("dual") is not None:
        raise ValueError("Free linear assignments have no column dual")
    _tensor(saved.get("theta"), snapshots[step]["theta"].shape, torch.float64, "linear resume head")
    if not torch.equal(saved["theta"], snapshots[step]["theta"]):
        raise ValueError("Linear resume head differs from its snapshot")
    for key in ("tracking_vector_before", "tracking_theta_before"):
        if saved.get(key) is not None:
            _tensor(saved[key], saved["theta"].shape, torch.float64, key)
    history = saved.get("history")
    if (not isinstance(history, list) or len(history) != step + 1
            or any(not isinstance(row, dict) or type(row.get("step")) is not int or row["step"] != index
                   for index, row in enumerate(history))):
        raise ValueError("Incomplete linear resume history")
    for row in history:
        for key in ("J", "inner_grad_max", "seconds"):
            value = row.get(key)
            if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
                raise ValueError("Invalid linear history certificate")
        if (row.get("inner_converged") is not True or row.get("J_exact") is not True
                or row["inner_grad_max"] > context["controls"]["inner_tol"] * (1 + 1e-6)):
            raise ValueError("Linear history contains an uncertified endpoint")
        if row["step"] < step and (row.get("cg_converged") is not True
                or type(row.get("cg_relative_residual")) not in (float, int)
                or not math.isfinite(row["cg_relative_residual"])
                or row["cg_relative_residual"] > context["controls"]["cg_rtol"] * (1 + 1e-6)):
            raise ValueError("Linear history contains an uncertified implicit update")
    for checkpoint, snapshot in snapshots.items():
        row = history[checkpoint]
        if (type(row.get("J")) is not float or type(row.get("inner_grad_max")) is not float
                or row["J"] != snapshot["teacher_ce"] or row["inner_grad_max"] != snapshot["inner_grad_max"]):
            raise ValueError("Linear history differs from its verified snapshot outer CE/stationarity certificate")
    _tensor(saved.get("initial_moments"), snapshots[0]["moments"].shape, torch.float64, "linear original moments")
    if not torch.equal(saved["initial_moments"], snapshots[0]["moments"]):
        raise ValueError("Linear original moments differ")
    if (type(saved.get("scale")) not in (float, int) or saved["scale"] != max(snapshots[0]["teacher_ce"], 1e-12)
            or type(saved.get("elapsed")) not in (float, int) or not math.isfinite(saved["elapsed"])
            or saved["elapsed"] != history[-1]["seconds"] or type(saved.get("best_step")) is not int
            or not 0 <= saved["best_step"] <= step or type(saved.get("best")) is not float or not math.isfinite(saved["best"])
            or saved.get("best") != min(row["J"] for row in history)
            or saved["best_step"] != min(range(len(history)), key=lambda index: history[index]["J"])):
        raise ValueError("Invalid linear objective/scale/history controls")
    _tensor(saved.get("best_moments"), snapshots[0]["moments"].shape, torch.float64, "linear best moments")
    _tensor(saved.get("best_theta"), snapshots[0]["theta"].shape, torch.float64, "linear best head")
    if saved["best_step"] in snapshots and (not torch.equal(saved["best_moments"], snapshots[saved["best_step"]]["moments"])
            or not torch.equal(saved["best_theta"], snapshots[saved["best_step"]]["theta"])):
        raise ValueError("Linear best checkpoint differs from its recorded snapshot")
    if folder is not None:
        for path in sorted((Path(folder) / "checkpoints").glob("step_*.pt")):
            disk = torch.load(path, map_location="cpu", weights_only=False)
            if not isinstance(disk, dict):
                raise ValueError("Malformed linear disk checkpoint")
            key = _step(disk.get("step"))
            if key not in snapshots or seal(disk) != seal(snapshots[key]):
                raise ValueError("Orphan or changed linear disk checkpoint")
    return saved




@torch.no_grad()
def cached_source(root, candidate, seed, h, q, config):
    """Only existing-file loads/content validation; no fit or creation fallback."""
    from src import citation_search
    from src.data import BUDGET
    from src.io import _fingerprint
    from src.nystrom_ce import NystromMap, _cache_identity, _content_digest
    from src.shared_features import (
        _load_state,
        _map_identity,
        _tensor_identity,
        _validate_map_state,
        _validate_matrix,
    )
    from src.transforms import FeatureTransform
    candidate = candidate_controls(candidate)
    native_environment()
    root = Path(root)
    if (not active(candidate) or config.get("teacher_kernel") is not None or config.get("teacher_backend") is not None
            or config.get("dataset") not in ("cora", "citeseer")):
        raise ValueError("Source linear requires its original ReLU citation source")
    origin = dict(mode=candidate["initialization"], alpha=candidate.get("alpha", 1.), T=candidate["T"], seed=seed)
    hard_path = root / f"assignment_{_fingerprint(origin)}.pt"
    names = ["config.json", "propagated_H.pt", "teacher.pt", f"inputs_{seed}.pt",
             "nystrom_map_schema3.pt", "nystrom_phi_schema3.npy", "nystrom_phi_schema3.meta.json"]
    if candidate["initialization"] != "feature":
        names.append(hard_path.name)
    if not all((root / name).is_file() for name in names):
        raise ValueError("Source linear requires existing immutable H/RMS/teacher/hard/map/Phi/sidecar; no creation or refit")
    if json.loads((root / "config.json").read_text()) != config:
        raise ValueError("Current graph/preprocessing/splits differ from the original source")
    _validate_matrix(h, "Current H")
    state = _load_state(root / "propagated_H.pt", "shared_h")
    saved_h = state.get("h")
    _validate_matrix(saved_h, "Frozen H")
    expected_source = _fingerprint(dict(data_digest=config["data_digest"], steps=2,
        implementation=hashlib.sha256((inspect.getsource(citation_search.normalize_adj_sparse) +
                                       inspect.getsource(citation_search._prepare_dataset)).encode()).hexdigest(),
        dtype=str(h.dtype), torch=str(torch.__version__)))
    if (type(state.get("schema")) is not int or state.get("source_digest") != expected_source
            or state.get("identity") != _tensor_identity(saved_h) or h.shape != saved_h.shape or h.dtype != saved_h.dtype):
        raise ValueError("Frozen H source, dtype, shape or content changed")
    h = saved_h.to(h.device)
    saved = torch.load(root / f"inputs_{seed}.pt", map_location=h.device, weights_only=False)
    if not isinstance(saved, dict) or not isinstance(saved.get("transform"), dict):
        raise ValueError("Malformed original RMS input cache")
    try:
        transform = FeatureTransform(**saved["transform"])
    except (TypeError, ValueError) as error:
        raise ValueError("Malformed original transform") from error
    if transform.kind != "rms" or transform.matrix is not None:
        raise ValueError("Source linear requires the original RMS transform")
    _tensor(transform.center, (h.shape[1],), torch.float64, "source RMS center")
    _tensor(transform.output_center, (h.shape[1],), torch.float64, "source RMS output center")
    _tensor(transform.scale, (), torch.float64, "source RMS scale")
    if float(transform.scale) <= 0 or type(transform.eps) is not float or not math.isfinite(transform.eps) or transform.eps <= 0:
        raise ValueError("Malformed source RMS controls")
    z = saved.get("z")
    _tensor(z, h.shape, torch.float64, "source RMS z")
    if not torch.allclose(z, transform(h.double()), atol=1e-12, rtol=1e-12):
        raise ValueError("Original RMS z differs from frozen H/transform")
    teacher = torch.load(root / "teacher.pt", map_location=h.device, weights_only=False)
    if not isinstance(teacher, dict):
        raise ValueError("Malformed source teacher")
    _tensor(teacher.get("logits"), q.shape, torch.float64, "source ReLU logits")
    if not torch.equal((teacher["logits"] / candidate["T"]).softmax(1), q):
        raise ValueError("Source Q differs from original ReLU logits/T")
    assignment = (torch.load(hard_path, map_location=h.device, weights_only=False)
                  if candidate["initialization"] != "feature" else saved.get("assignment"))
    _tensor(assignment, (len(h),), torch.int64, "source hard assignment")
    budget = BUDGET.get((config["dataset"], config["ratio"]))
    if budget is None or bool((assignment < 0).any()) or len(torch.unique(assignment)) != budget or int(assignment.max()) + 1 != budget:
        raise ValueError("Source hard partition differs from its node budget")
    map_state = _load_state(root / "nystrom_map_schema3.pt", "shared_nystrom_map")
    if type(map_state.get("schema")) is not int:
        raise ValueError("Malformed source map schema")
    anchors, mapping = _validate_map_state(map_state, _map_identity(h, 3000, 0, "relu"))
    feature_map = NystromMap(anchors.to(h.device), mapping.to(h.device), "relu")
    try:
        phi = np.load(root / "nystrom_phi_schema3.npy", mmap_mode="r", allow_pickle=False)
        metadata = json.loads((root / "nystrom_phi_schema3.meta.json").read_text())
    except (OSError, ValueError) as error:
        raise ValueError("Malformed source Phi/metadata") from error
    expected_shape = (len(h), min(len(h), 3000))
    if tuple(phi.shape) != expected_shape or np.dtype(phi.dtype) != np.float64:
        raise ValueError("Source Phi shape/dtype changed")
    identity = _cache_identity(h, feature_map, expected_shape, 2048, lambda: False)
    phi_digest = _content_digest(phi)
    if (not isinstance(metadata, dict) or type(metadata.get("schema")) is not int
            or metadata != dict(**identity, phi_digest=phi_digest)):
        raise ValueError("Source Phi H/map/content metadata fingerprint differs")
    if any(not np.isfinite(phi[start:start + 2048]).all() for start in range(0, len(phi), 2048)):
        raise ValueError("Source Phi must contain finite coordinates")
    inputs = (z.detach().float() if candidate["assignment_coordinates"] == "raw_rms" else
              torch.from_numpy(np.array(phi, dtype=np.float32, copy=True)).to(z.device).detach())
    _coordinate_tensor(inputs, z, inputs.shape[1])
    source = dict(root=str(root.resolve()), config=config, initializer_origin=origin, candidate=candidate,
                  assets={name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names},
                  h=digest(h), z=digest(z), q=digest(q), hard_assignment=digest(assignment), transform=digest(saved["transform"]),
                  coordinate_mode=candidate["assignment_coordinates"], coordinates_fp32=digest(inputs),
                  phi_identity=dict(**identity, phi_digest=phi_digest),
                  map_identity=dict(anchors=digest(feature_map.anchors), mapping=digest(feature_map.mapping)),
                  coordinate_policy="existing_F64_source_to_detached_F32; no_normalization_or_refit",
                  teacher_feature_identity="same_family_shared_ReLU_map; historical_teacher_exact_Phi_unproved")
    return h, z, transform, assignment, inputs, source


def citation_core_kwargs(candidate, seed, inputs, source):
    return dict(penalty=candidate["penalty"], lr=candidate["lr"], mixing=candidate.get("mixing", .05),
                assignment_rank=candidate["rank"], factor_seed=seed, assignment_input="features",
                assignment_encoder="linear", solver_mode="exact", inner_method="newton_first", implicit_warm_start=True,
                mass_mode="free", inner_loss_weighting="uniform", inner_max_iter=2000, inner_tol=1e-7,
                cg_max_iter=512, cg_rtol=1e-6, cache_assignment=False, save_assignment=False, save_resume=True,
                source_linear_coordinates=candidate["assignment_coordinates"], source_linear_schema=SCHEMA,
                source_linear_source=source, source_linear_inputs=inputs)


def expected_citation_config(candidate, seed, z, q, assignment, inputs, source):
    from src.soft_ce_partition import optimize_ce_assignment
    values = {key: value.default for key, value in inspect.signature(optimize_ce_assignment).parameters.items()
              if value.default is not inspect.Parameter.empty}
    values.update(citation_core_kwargs(candidate, seed, inputs, source))
    for key in ("steps", "folder", "checkpoint_steps", "resume_state", "save_resume", "log_every", "initial_representatives",
                "outer_indices", "implicit_solver", "inner_solver", "temperature_logits", "outer_targets", "stop",
                "temperature_initial", "temperature_lr", "cg_check_interval", "cache_assignment", "node_weighting",
                "node_weight_penalty", "node_weight_lr", "source_linear_inputs"):
        values.pop(key, None)
    values["data_digest"] = array_digest(z.detach().cpu().numpy(), q.detach().cpu().numpy(), assignment.cpu().numpy())
    return values


def validate_folder(root, candidate, seed, z, q, assignment, inputs, source, steps, snapshot=None):
    from src.io import _fingerprint
    folder = Path(root) / _fingerprint(candidate) / f"condensation_{seed}"
    recorded = folder.parent / "candidate.json"
    try:
        if recorded.exists() and json.loads(recorded.read_text()) != candidate:
            raise ValueError("Recorded source-linear candidate differs")
    except (OSError, ValueError) as error:
        raise ValueError("Malformed or changed source-linear candidate identity") from error
    config = expected_citation_config(candidate, seed, z, q, assignment, inputs, source)
    context, parameters = original_context(z, q, assignment, inputs, candidate["rank"], seed,
                                           candidate.get("mixing", .05), 4096, config, source)
    resume = folder / "resume.pt"
    if resume.exists():
        saved = torch.load(resume, map_location="cpu", weights_only=False)
        if not isinstance(saved, dict):
            raise ValueError("Malformed source-linear resume")
        validate_resume(saved, z, q, assignment, inputs, context, parameters, max(steps, _step(saved.get("step"))), folder)
    elif snapshot is not None or any((folder / "checkpoints").glob("step_*.pt")):
        raise ValueError("Source-linear cached endpoints require a verifiable resume")
    if snapshot is not None:
        validate_snapshot(snapshot, z, q, assignment, inputs, context, parameters)
    return context, parameters


def prepare_probe(dataset, ratio, output_dir, candidate, condensation_seed=0, data_dir="data",
                  device="cuda", citation_features="default", stop=lambda: False):
    """One update, zero students, identical source/core config to screen25."""
    from src import citation_search
    from src.data import BUDGET
    from src.io import _fingerprint, save_json
    from src.soft_ce_partition import optimize_ce_assignment
    from src.target_refinement import training_refined_targets
    candidate = candidate_controls(candidate)
    native_environment()
    if not active(candidate) or dataset not in ("cora", "citeseer") or (dataset, ratio) not in BUDGET:
        raise ValueError("Source-linear probe requires a supported citation candidate")
    if not callable(stop):
        raise ValueError("stop must be callable")
    if stop():
        raise InterruptedError("Source-linear probe stopped")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    graph, train, validation, testing, h = citation_search._prepare_dataset(dataset, data_dir, device, citation_features)
    config = citation_search._legacy_teacher_config(dataset, ratio, graph, train, validation, testing, citation_features)
    root = Path(output_dir) / dataset / f"ratio_{ratio}" / _fingerprint(config)
    if not (root / "teacher.pt").is_file():
        raise ValueError("Source-linear probe requires original cached ReLU teacher")
    teacher = torch.load(root / "teacher.pt", map_location=device, weights_only=False)
    if not isinstance(teacher, dict) or not torch.is_tensor(teacher.get("logits")):
        raise ValueError("Malformed original teacher")
    q = training_refined_targets(teacher["logits"], candidate["T"], graph["y"], train, 0)
    h, z, _, assignment, inputs, source = cached_source(root, candidate, condensation_seed, h, q, config)
    folder = root / _fingerprint(candidate) / f"condensation_{condensation_seed}"
    validate_folder(root, candidate, condensation_seed, z, q, assignment, inputs, source, 1)
    resume = folder / "resume.pt"
    state = torch.load(resume, map_location="cpu", weights_only=False) if resume.exists() else None
    cached = state is not None and state["step"] >= 1
    if not cached:
        if not (folder.parent / "candidate.json").exists():
            folder.parent.mkdir(parents=True, exist_ok=True)
            save_json(candidate, folder.parent / "candidate.json")
        optimize_ce_assignment(z, q, assignment, steps=1, folder=folder, checkpoint_steps=(0, 1),
                               resume_state=state, stop=stop, **citation_core_kwargs(candidate, condensation_seed, inputs, source))
    validate_folder(root, candidate, condensation_seed, z, q, assignment, inputs, source, 1)
    state = torch.load(resume, map_location="cpu", weights_only=False)
    row = next(row for row in state["history"] if row["step"] == 1)
    return dict(root=str(root.resolve()), candidate_path=str(folder.parent.resolve()), step=1, cached=cached,
                validation_only=True, student_fits=0, inner_grad_max=row["inner_grad_max"], inner_converged=row["inner_converged"],
                min_mass=row["min_mass"], max_mass=row["max_mass"], effective_cells=row["effective_cells"], seconds=row["seconds"])
