"""MLP-only original-P0 marginal provenance and strict cached-state validation.

The target is derived on the native consumer device from the original encoder
and factors. Completed or interrupted initial-mass trajectories are reusable
only on that same device. CPU inspection of CUDA artifacts is a numeric audit,
not a supported cross-device resume. Legacy free/uniform paths never call here.
"""
import hashlib
import inspect
import json
import math
from numbers import Real
from pathlib import Path

import torch

from src.fixed_mass_assignment import validate_mass_target
from src.io import array_digest
from src.low_rank_assignment import assignment_inputs, encode_nodes, initialize_mlp, logit_block
from src.moments import augmented, decode_moments, make_material

SCHEMA = 1
_EXTERNAL = {"mass_target", "target_masses", "initial_mass_target", "initial_mass_provenance", "initial_context"}
_DEPENDENCIES = ("mlp_initial_mass.py", "soft_ce_partition.py", "fixed_mass_assignment.py",
                 "balanced_assignment.py", "low_rank_assignment.py", "moments.py", "head.py",
                 "citation_search.py", "target_refinement.py", "transforms.py", "partition_initialization.py",
                 "data.py", "shared_features.py", "teacher.py", "io.py")


def candidate_controls(candidate):
    candidate = dict(candidate)
    active = candidate.get("mass_mode") == "initial" and candidate.get("method") == "mlp"
    if set(candidate) & {"mlp_initial_mass_schema", "mlp_initial_mass_source_digest"} and not active:
        raise ValueError("mlp_initial_mass_schema requires MLP initial mass")
    if not active:
        return candidate
    if set(candidate) & (_EXTERNAL | {"initial_mass_schema", "outer_targets", "outer_indices", "initial_representatives", "temperature_logits", "implicit_solver", "inner_solver", "feature_control", "assignment_rank", "factor_seed", "encoder_hidden"}):
        raise ValueError("MLP target masses must be derived internally from original P0")
    schema = candidate.get("mlp_initial_mass_schema")
    if type(schema) is not int or schema != SCHEMA:
        raise ValueError("MLP mass_mode=initial requires explicit mlp_initial_mass_schema=1")
    for key in ("penalty", "lr", "T"):
        value = candidate.get(key)
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
            raise ValueError("MLP initial mass requires positive finite penalty/lr/T")
    if (type(candidate.get("rank")) is not int or candidate["rank"] < 1
            or type(candidate.get("width")) is not int or candidate["width"] < 1):
        raise ValueError("MLP initial mass requires positive integer rank and width")
    if (candidate.get("inner_loss_weighting") != "uniform"
            or candidate.get("assignment_input", "features") != "features"
            or candidate.get("assignment_encoder", "mlp") != "mlp"
            or candidate.get("solver_mode", "exact") != "exact"
            or candidate.get("learn_temperature", False)
            or candidate.get("train_target_mix", 0) != 0
            or candidate.get("node_weighting", False)
            or candidate.get("surrogate_kernel") is not None
            or candidate.get("mixing", .05) != .05):
        raise ValueError("MLP initial mass supports only fixed features, uniform CE and original P0")
    token = seal(source_digest())
    if candidate.get("mlp_initial_mass_source_digest", token) != token:
        raise ValueError("MLP initial numerical source changed; preserve the old candidate namespace")
    candidate["mlp_initial_mass_source_digest"] = token
    if any(key in candidate for key in ("balance_cg_steps", "balance_cg_rtol")):
        raise ValueError("MLP initial mass uses only the fixed chunked projection")
    for key, expected in (("balance_backend", "chunked"), ("balance_steps", 5000), ("balance_tol", 1e-8)):
        value = candidate.get(key, expected)
        if isinstance(value, bool) or value != expected:
            raise ValueError("First MLP initial-mass pilot pins chunked/5000/1e-8")
        candidate[key] = expected
    return candidate


def digest(value):
    if torch.is_tensor(value):
        return dict(tensor=array_digest(value.detach().cpu().numpy()), shape=list(value.shape), dtype=str(value.dtype))
    if isinstance(value, dict):
        return {str(key): digest(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [digest(item) for item in value]
    return value


def seal(value):
    return hashlib.sha256(json.dumps(digest(value), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def source_digest():
    base = Path(__file__).parent
    return {name: hashlib.sha256((base / name).read_bytes()).hexdigest() for name in _DEPENDENCIES}


def _tensor(value, shape, dtype, description):
    if (not torch.is_tensor(value) or tuple(value.shape) != tuple(shape) or value.dtype != dtype
            or not bool(torch.isfinite(value).all())):
        raise ValueError(f"Invalid MLP initial-mass {description}")


def _step(value):
    if type(value) is not int or value < 0:
        raise ValueError("MLP initial-mass step must be a nonnegative integer")
    return value


@torch.no_grad()
def original_context(z, q, assignment, rank, hidden, seed, mixing, chunk_size, config, source=None):
    if (z.dtype != torch.float64 or q.dtype != torch.float64 or z.ndim != 2 or q.ndim != 2
            or len(z) != len(q) or assignment.shape != (len(z),) or assignment.dtype != torch.int64
            or min(z.shape) < 1 or min(q.shape) < 1 or not bool(torch.isfinite(z).all())
            or not bool(torch.isfinite(q).all()) or bool((q < 0).any())
            or float((q.sum(1) - 1).abs().max()) > 1e-12
            or z.device != q.device or z.device != assignment.device
            or bool((assignment < 0).any()) or type(chunk_size) is not int or chunk_size < 1):
        raise ValueError("MLP initial mass requires finite, aligned native float64 source inputs")
    cells = int(assignment.max()) + 1
    if len(torch.unique(assignment)) != cells:
        raise ValueError("Original hard partition must contain every cell")
    inputs = assignment_inputs(z, q, "features")
    encoder, v = initialize_mlp(inputs, cells, rank, hidden, seed)
    u = encode_nodes(inputs, encoder)
    target = z.new_zeros(cells)
    for start in range(0, len(z), chunk_size):
        p = logit_block(u[start:start + chunk_size], v, assignment[start:start + chunk_size], mixing).double().softmax(1)
        target += p.sum(0) / len(z)
    validate_mass_target(target, cells)
    context = dict(schema=SCHEMA, device=str(z.device), nodes=len(z), cells=cells,
                   dimension=z.shape[1], classes=q.shape[1], rank=rank, hidden=hidden,
                   seed=seed, mixing=mixing, chunk_size=chunk_size,
                   z=digest(z), q=digest(q), assignment=digest(assignment),
                   initializer=[digest(p) for p in [*encoder, v]], encoded_u0=digest(u),
                   target=digest(target), source=source, controls=config,
                   helper_source=source_digest(), torch_version=str(torch.__version__),
                   default_dtype=str(torch.get_default_dtype()))
    return target.detach(), context, [*encoder, v]


def attach(saved, target, context, parameters=None):
    saved.update(mass_mode="initial", mlp_initial_mass_schema=SCHEMA,
                 mass_target=target.detach().cpu().clone(), initial_mass_context=context)
    if parameters is not None:
        saved["parameters"] = [p.detach().cpu().clone() for p in parameters]
    saved["initial_mass_content_sha256"] = seal({key: value for key, value in saved.items()
                                               if key != "initial_mass_content_sha256"})
    return saved


def _identity(saved, target, context):
    if not isinstance(saved, dict):
        raise ValueError("MLP initial-mass cached payload must be a mapping")
    if (saved.get("mass_mode") != "initial" or type(saved.get("mlp_initial_mass_schema")) is not int
            or saved.get("mlp_initial_mass_schema") != SCHEMA):
        raise ValueError("Invalid MLP initial-mass cache mode/schema")
    if saved.get("initial_mass_context") != context:
        raise ValueError("MLP initial-mass original source/initializer/device/controls changed")
    validate_mass_target(saved.get("mass_target"), len(target))
    if not torch.equal(saved["mass_target"].cpu(), target.cpu()):
        raise ValueError("Cached target is not the current native original P0 mass")
    if saved.get("initial_mass_content_sha256") != seal({key: value for key, value in saved.items()
                                                       if key != "initial_mass_content_sha256"}):
        raise ValueError("MLP initial-mass cache content checksum mismatch")


@torch.no_grad()
def validate_snapshot(saved, z, q, assignment, target, context, initial_parameters):
    _identity(saved, target, context)
    step = _step(saved.get("step"))
    parameters = saved.get("parameters")
    if not isinstance(parameters, (list, tuple)) or len(parameters) != 5:
        raise ValueError("MLP initial-mass checkpoint must retain all five native parameters")
    for parameter, original in zip(parameters, initial_parameters, strict=True):
        _tensor(parameter, original.shape, original.dtype, "checkpoint parameter")
    if step == 0 and [digest(p) for p in parameters] != context["initializer"]:
        raise ValueError("Checkpoint zero differs from the original MLP initializer")
    k, d, c = context["cells"], context["dimension"], context["classes"]
    _tensor(saved.get("moments"), (k, 1 + d + c), torch.float64, "checkpoint moments")
    _tensor(saved.get("theta"), (c, 1 + d), torch.float64, "checkpoint CE head")
    dual = saved.get("column_dual")
    _tensor(dual, (k,), torch.float64, "checkpoint dual")
    if abs(float(dual.mean())) > 1e-10:
        raise ValueError("Checkpoint dual is outside the zero-mean gauge")
    u = encode_nodes(assignment_inputs(z, q, "features"), [p.to(z.device) for p in parameters[:-1]])
    v, material = parameters[-1].to(z.device), make_material(z, q)
    replay = z.new_zeros(k, 1 + d + c)
    for start in range(0, len(z), context["chunk_size"]):
        p = (logit_block(u[start:start + context["chunk_size"]], v,
                        assignment[start:start + context["chunk_size"]], context["mixing"]).double()
             + dual.to(z.device)).softmax(1)
        if float((p.sum(1) - 1).abs().max()) > 1e-12:
            raise ValueError("Checkpoint row stochasticity failed")
        replay += p.T @ material[start:start + context["chunk_size"]] / len(z)
    if float((replay[:, 0] / target - 1).abs().max()) > context["controls"]["balance_tol"]:
        raise ValueError("Checkpoint violates original P0 column masses")
    if not torch.equal(replay.cpu(), saved["moments"].cpu()):
        raise ValueError("Checkpoint moments do not replay from saved native MLP parameters")
    from src.soft_ce_partition import head_gradient
    centers, labels, mass = decode_moments(replay, d)
    maximum = float(head_gradient(augmented(centers), labels, torch.full_like(mass, 1 / k),
                                  saved["theta"].to(z), context["controls"]["penalty"]).abs().max())
    if (saved.get("J_exact") is not True or not math.isfinite(maximum)
            or maximum > context["controls"]["inner_tol"] * (1 + 1e-6)):
        raise ValueError("Cached uniform CE head lacks a valid stationarity certificate")
    return replay


def _optimizer(saved, initial_parameters, step, lr):
    optimizer = saved.get("optimizer")
    if not isinstance(optimizer, dict) or set(optimizer) != {"state", "param_groups"}:
        raise ValueError("Malformed initial-mass Adam state")
    groups, states = optimizer["param_groups"], optimizer["state"]
    if not isinstance(groups, list) or len(groups) != 1 or not isinstance(states, dict):
        raise ValueError("Malformed initial-mass Adam groups")
    group = groups[0]
    if not isinstance(group, dict):
        raise ValueError("Malformed initial-mass Adam group")
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
            or group["params"] != list(range(5)) or any(type(key) is not int for key in states)):
        raise ValueError("Malformed Adam parameter identifiers")
    if group["params"] != list(range(5)) or set(states) != (set(range(5)) if step else set()):
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


def validate_resume(saved, z, q, assignment, target, context, initial_parameters, steps, folder=None):
    _identity(saved, target, context)
    step = _step(saved.get("step"))
    if step > steps or saved.get("config") != context["controls"]:
        raise ValueError("Resume configuration or budget differs from the current MLP protocol")
    _optimizer(saved, initial_parameters, step, context["controls"]["lr"])
    snapshots = saved.get("snapshots")
    if not isinstance(snapshots, dict) or 0 not in snapshots or step not in snapshots:
        raise ValueError("Initial-mass resume requires its zero and current verifiable checkpoints")
    for key, snapshot in snapshots.items():
        if not isinstance(snapshot, dict):
            raise ValueError("Initial-mass checkpoint history must contain mappings")
        if type(key) is not int or key > step or snapshot.get("step") != key:
            raise ValueError("Malformed initial-mass checkpoint history")
        validate_snapshot(snapshot, z, q, assignment, target, context, initial_parameters)
    parameters = saved.get("parameters")
    if not isinstance(parameters, (list, tuple)) or len(parameters) != 5:
        raise ValueError("Initial-mass resume requires all five native parameters")
    for parameter, original in zip(parameters, initial_parameters, strict=True):
        _tensor(parameter, original.shape, original.dtype, "resume parameter")
    for parameter, current in zip(parameters, snapshots[step]["parameters"], strict=True):
        if not torch.equal(parameter, current):
            raise ValueError("Resume parameters differ from the current checkpoint")
    _tensor(saved.get("dual"), target.shape, torch.float64, "resume dual")
    _tensor(saved.get("theta"), snapshots[step]["theta"].shape, torch.float64, "resume head")
    if not torch.equal(saved.get("dual"), snapshots[step]["column_dual"]):
        raise ValueError("Resume dual differs from the current checkpoint")
    if not torch.equal(saved.get("theta"), snapshots[step]["theta"]):
        raise ValueError("Resume head differs from the current checkpoint")
    history = saved.get("history")
    if (not isinstance(history, list) or len(history) != step + 1
            or any(not isinstance(row, dict) or type(row.get("step")) is not int
                   or row["step"] != index for index, row in enumerate(history))):
        raise ValueError("Initial-mass resume history is incomplete or unordered")
    for row in history:
        for key in ("J", "inner_grad_max", "row_residual", "column_residual", "seconds"):
            value = row.get(key)
            if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
                raise ValueError("Invalid initial-mass history certificate")
        if (row.get("inner_converged") is not True or row.get("J_exact") is not True
                or row["inner_grad_max"] > context["controls"]["inner_tol"] * (1 + 1e-6)
                or max(row["row_residual"], row["column_residual"]) > context["controls"]["balance_tol"]):
            raise ValueError("Initial-mass history contains an uncertified endpoint")
        if row["step"] < step and (row.get("cg_converged") is not True
                or not math.isfinite(row.get("cg_relative_residual", float("nan")))
                or row["cg_relative_residual"] > context["controls"]["cg_rtol"] * (1 + 1e-6)):
            raise ValueError("Initial-mass history contains an uncertified implicit update")
    _tensor(saved.get("initial_moments"), snapshots[0]["moments"].shape, torch.float64, "original moments")
    if not torch.equal(saved.get("initial_moments"), snapshots[0]["moments"]):
        raise ValueError("Resume initial moments differ from source P0")
    if (type(saved.get("scale")) not in (float, int) or saved["scale"] != max(history[0]["J"], 1e-12)
            or type(saved.get("elapsed")) not in (float, int) or not math.isfinite(saved["elapsed"])
            or saved["elapsed"] != history[-1]["seconds"]
            or type(saved.get("best_step")) is not int or not 0 <= saved["best_step"] <= step
            or saved.get("best") != min(row["J"] for row in history)):
        raise ValueError("Invalid initial-mass resume objective/scale/history controls")
    if folder is not None:
        paths = sorted((Path(folder) / "checkpoints").glob("step_*.pt"))
        for path in paths:
            disk = torch.load(path, map_location="cpu", weights_only=False)
            key = _step(disk.get("step"))
            if key not in snapshots or seal(disk) != seal(snapshots[key]):
                raise ValueError("Orphan or changed initial-mass checkpoint on disk")
    return saved


def expected_citation_config(candidate, seed, z, q, assignment, source):
    """Mirror the unchanged core's argument-only resume configuration."""
    from src.soft_ce_partition import optimize_ce_assignment
    values = {key: value.default for key, value in inspect.signature(optimize_ce_assignment).parameters.items()
              if value.default is not inspect.Parameter.empty}
    values.update(penalty=candidate["penalty"], lr=candidate["lr"], mixing=candidate.get("mixing", .05),
                  assignment_rank=candidate["rank"], factor_seed=seed, assignment_input="features",
                  assignment_encoder="mlp", encoder_hidden=candidate["width"], solver_mode="exact",
                  inner_method="newton_first", implicit_warm_start=True, mass_mode="initial",
                  inner_loss_weighting="uniform", inner_max_iter=2000, inner_tol=1e-7,
                  cg_max_iter=512, cg_rtol=1e-6, cache_assignment=False, save_assignment=False,
                  balance_backend="chunked", balance_steps=5000, balance_tol=1e-8)
    for key in ("steps", "folder", "checkpoint_steps", "resume_state", "save_resume", "log_every",
                "initial_representatives", "outer_indices", "implicit_solver", "inner_solver", "temperature_logits",
                "outer_targets", "stop", "mlp_initial_mass_schema", "mlp_initial_mass_source",
                "temperature_initial", "temperature_lr", "cg_check_interval", "cache_assignment",
                "node_weighting", "node_weight_penalty", "node_weight_lr"):
        values.pop(key, None)
    values["data_digest"] = array_digest(z.detach().cpu().numpy(), q.detach().cpu().numpy(), assignment.cpu().numpy())
    values["mlp_initial_mass_schema"] = SCHEMA
    values["mlp_initial_mass_source"] = source
    return values


@torch.no_grad()
def cached_source(root, candidate, seed, h, q, config):
    """Require existing ReLU assets; never create a teacher, RMS fit or partition."""
    from src.citation_search import _cached_initial_assignment, fixed_propagated_features
    from src.transforms import FeatureTransform
    root = Path(root)
    if config.get("teacher_kernel") is not None or config.get("teacher_backend") is not None:
        raise ValueError("MLP initial-mass pilot requires its original ReLU source root")
    paths = [root / "config.json", root / "propagated_H.pt", root / "teacher.pt", root / f"inputs_{seed}.pt"]
    if not all(path.is_file() for path in paths):
        raise ValueError("Original MLP source assets must exist before the initial-mass pilot")
    if json.loads(paths[0].read_text()) != config:
        raise ValueError("Current graph/splits differ from the original source configuration")
    h = fixed_propagated_features(h, config, root)
    saved = torch.load(paths[-1], map_location=h.device, weights_only=False)
    if not isinstance(saved, dict) or not isinstance(saved.get("transform"), dict):
        raise ValueError("Malformed original RMS input cache")
    transform = FeatureTransform(**saved["transform"])
    if transform.kind != "rms" or transform.matrix is not None:
        raise ValueError("MLP initial-mass pilot requires the original RMS source transform")
    z = saved.get("z")
    _tensor(z, h.shape, torch.float64, "original RMS features")
    replay = transform(h.double())
    if not torch.allclose(z, replay, atol=1e-12, rtol=1e-12):
        raise ValueError("Original RMS features do not match the frozen source H/transform")
    teacher = torch.load(paths[2], map_location=h.device, weights_only=False)
    if not isinstance(teacher, dict):
        raise ValueError("Malformed original ReLU teacher cache")
    logits = teacher.get("logits")
    _tensor(logits, q.shape, torch.float64, "original ReLU logits")
    if not torch.equal((logits / candidate["T"]).softmax(1).double(), q):
        raise ValueError("MLP target Q differs from the original frozen ReLU teacher")
    assignment = _cached_initial_assignment(root, candidate, seed, h.device)
    from src.data import BUDGET
    if assignment.ndim != 1 or assignment.dtype != torch.int64 or assignment.shape != (len(h),) or (
            int(assignment.max()) + 1 != BUDGET[(config["dataset"], config["ratio"])]):
        raise ValueError("Original hard partition does not match the current node budget")
    if candidate.get("initialization", "feature") != "feature":
        from src.io import _fingerprint
        origin = dict(mode=candidate["initialization"], alpha=candidate.get("alpha", 1.), T=candidate["T"], seed=seed)
        paths.append(root / f"assignment_{_fingerprint(origin)}.pt")
    else:
        origin = dict(mode="feature", seed=seed)
    source = dict(root=str(root.resolve()), config=config, initializer_origin=origin, candidate=candidate,
                  assets={path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
                  h=digest(h), z=digest(z), q=digest(q), hard_assignment=digest(assignment),
                  transform=digest(saved["transform"]))
    return h, z, transform, assignment, source


def citation_core_kwargs(candidate, seed, source):
    return dict(penalty=candidate["penalty"], lr=candidate["lr"], mixing=candidate.get("mixing", .05),
                assignment_rank=candidate["rank"], factor_seed=seed, assignment_input="features",
                assignment_encoder="mlp", encoder_hidden=candidate["width"], solver_mode="exact",
                inner_method="newton_first", implicit_warm_start=True, mass_mode="initial",
                inner_loss_weighting="uniform", inner_max_iter=2000, inner_tol=1e-7,
                cg_max_iter=512, cg_rtol=1e-6, cache_assignment=False, save_assignment=False,
                save_resume=True, balance_backend="chunked", balance_steps=5000, balance_tol=1e-8,
                mlp_initial_mass_schema=SCHEMA, mlp_initial_mass_source=source)


def validate_folder(root, candidate, seed, z, q, assignment, source, steps, snapshot=None):
    from src.io import _fingerprint
    folder = Path(root) / _fingerprint(candidate) / f"condensation_{seed}"
    recorded = folder.parent / "candidate.json"
    if recorded.exists():
        try:
            identity = json.loads(recorded.read_text())
        except (ValueError, OSError) as error:
            raise ValueError("Malformed recorded MLP initial candidate") from error
        if identity != candidate:
            raise ValueError("Recorded MLP initial candidate differs from the current identity")
    config = expected_citation_config(candidate, seed, z, q, assignment, source)
    target, context, parameters = original_context(z, q, assignment, candidate["rank"], candidate["width"],
                                                  seed, candidate.get("mixing", .05), 4096, config, source)
    resume_path = folder / "resume.pt"
    if resume_path.exists():
        saved = torch.load(resume_path, map_location="cpu", weights_only=False)
        validate_resume(saved, z, q, assignment, target, context, parameters, max(steps, _step(saved.get("step"))), folder)
    elif snapshot is not None or any((folder / "checkpoints").glob("step_*.pt")):
        raise ValueError("Initial-mass cached endpoints require a verifiable resume state")
    if snapshot is not None:
        validate_snapshot(snapshot, z, q, assignment, target, context, parameters)
    return target, context, parameters


def prepare_probe(dataset, ratio, output_dir, candidate, condensation_seed=0, data_dir="data",
                  device="cuda", citation_features="default", stop=lambda: False):
    """One native update, zero students, same source/config as the later screen."""
    from src import citation_search
    from src.data import BUDGET
    from src.io import _fingerprint
    from src.soft_ce_partition import optimize_ce_assignment
    from src.target_refinement import training_refined_targets
    candidate = citation_search._candidate_surrogate(citation_search._candidate_nystrom_mass(
        citation_search._candidate_temperature(citation_search._candidate_background_mixing(candidate)[0])))
    if candidate.get("mass_mode") != "initial" or (dataset, ratio) not in BUDGET or dataset not in ("cora", "citeseer"):
        raise ValueError("Native initial-mass probe requires a supported MLP citation candidate")
    if not callable(stop):
        raise ValueError("stop must be callable")
    if stop():
        raise InterruptedError("MLP initial probe stopped")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    graph, train, validation, testing, h = citation_search._prepare_dataset(dataset, data_dir, device, citation_features)
    config = citation_search._legacy_teacher_config(dataset, ratio, graph, train, validation, testing, citation_features)
    root = Path(output_dir) / dataset / f"ratio_{ratio}" / _fingerprint(config)
    if not (root / "teacher.pt").is_file():
        raise ValueError("Probe requires a prepared original ReLU teacher")
    logits = torch.load(root / "teacher.pt", map_location=device, weights_only=False)["logits"]
    q = training_refined_targets(logits, candidate["T"], graph["y"], train, 0)
    h, z, _, assignment, source = cached_source(root, candidate, condensation_seed, h, q, config)
    folder = root / _fingerprint(candidate) / f"condensation_{condensation_seed}"
    validate_folder(root, candidate, condensation_seed, z, q, assignment, source, 1)
    resume = folder / "resume.pt"
    state = torch.load(resume, map_location="cpu", weights_only=False) if resume.exists() else None
    cached = state is not None and state["step"] >= 1
    if not cached:
        from src.io import save_json
        recorded = folder.parent / "candidate.json"
        if not recorded.exists():
            folder.parent.mkdir(parents=True, exist_ok=True)
            save_json(candidate, recorded)
        optimize_ce_assignment(z, q, assignment, steps=1, folder=folder, checkpoint_steps=(0, 1),
                               resume_state=state, stop=stop, **citation_core_kwargs(candidate, condensation_seed, source))
    validate_folder(root, candidate, condensation_seed, z, q, assignment, source, 1)
    state = torch.load(resume, map_location="cpu", weights_only=False)
    last = next(row for row in state["history"] if row["step"] == 1)
    return dict(root=str(root.resolve()), candidate_path=str(folder.parent.resolve()), step=1,
                cached=cached, validation_only=True, student_fits=0,
                inner_grad_max=last["inner_grad_max"], inner_converged=last["inner_converged"],
                row_residual=last["row_residual"], column_residual=last["column_residual"],
                balance_iterations=last["balance_iterations"], seconds=last["seconds"])
