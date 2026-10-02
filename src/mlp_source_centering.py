"""MLP-only differentiable all-source-node output centering with free masses.

No stored/external mean is an input to the forward. Native completed/interrupted
states are replayable only on their original device. CPU inspection of CUDA
artifacts is a numerical audit, not supported cross-device cache reuse.
"""
import inspect
import json
import math
from numbers import Real
from pathlib import Path

import torch

from src.io import array_digest
from src.low_rank_assignment import LowRankMoments, assignment_inputs, initialize_mlp, logit_block
from src.mlp_initial_mass import _optimizer, _step, _tensor, cached_source, digest, seal
from src.moments import augmented, decode_moments, make_material

SCHEMA = 1
MODE = "source_mean_v1"
POLICY = "native_mean_subtract; measured_FP64_mean; bound=4*gamma(N+2)*max(1,maxabs(raw)); gamma(n)=n*eps/(1-n*eps)"
_KEYS = {"mlp_output_centering", "mlp_source_centering_schema", "mlp_source_centering_source_digest"}
_FORBIDDEN = {"mass_target", "target_masses", "initial_mass_target", "initial_mass_provenance", "initial_context",
              "center_mean", "source_mean", "external_mean", "centering_mean", "raw_output_mean", "source_centering_context",
              "mlp_source_centering_source", "initial_mass_schema",
              "mlp_initial_mass_schema", "mlp_initial_mass_source_digest", "outer_targets", "outer_indices",
              "initial_representatives", "temperature_logits", "implicit_solver", "inner_solver", "feature_control",
              "assignment_rank", "factor_seed", "encoder_hidden", "balance_steps", "balance_tol", "balance_backend",
              "balance_cg_steps", "balance_cg_rtol"}
_DEPENDENCIES = ("mlp_source_centering.py", "soft_ce_partition.py", "low_rank_assignment.py", "moments.py",
                 "head.py", "citation_search.py", "mlp_initial_mass.py", "target_refinement.py", "transforms.py",
                 "partition_initialization.py", "data.py", "shared_features.py", "teacher.py", "io.py")


def source_digest():
    import hashlib
    return {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest() for name in _DEPENDENCIES}


def active(candidate):
    return candidate.get("mlp_output_centering") == MODE


def candidate_controls(candidate):
    candidate = dict(candidate)
    if not set(candidate) & _KEYS:
        return candidate
    if not active(candidate) or candidate.get("method") != "mlp":
        raise ValueError("mlp_output_centering requires its MLP source_mean_v1 mode")
    if set(candidate) & _FORBIDDEN or any(key.startswith("mlp_source_centering") and key not in _KEYS for key in candidate):
        raise ValueError("Source centering has no external mean, target, projection or alternate material")
    if type(candidate.get("mlp_source_centering_schema")) is not int or candidate["mlp_source_centering_schema"] != SCHEMA:
        raise ValueError("Source centering requires explicit mlp_source_centering_schema=1")
    for key in ("penalty", "lr", "T"):
        value = candidate.get(key)
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
            raise ValueError("Source centering requires positive finite penalty/lr/T")
    if any(type(candidate.get(key)) is not int or candidate[key] < 1 for key in ("rank", "width")):
        raise ValueError("Source centering requires positive integer rank and width")
    if (candidate.get("mass_mode", "free") != "free" or candidate.get("inner_loss_weighting") != "uniform"
            or candidate.get("assignment_input", "features") != "features"
            or candidate.get("assignment_encoder", "mlp") != "mlp" or candidate.get("solver_mode", "exact") != "exact"
            or candidate.get("learn_temperature", False) or candidate.get("train_target_mix", 0) != 0
            or candidate.get("node_weighting", False) or candidate.get("surrogate_kernel") is not None
            or candidate.get("mixing", .05) != .05):
        raise ValueError("Source centering supports only original MLP/free-mass/uniform-CE inputs")
    if candidate.get("initialization", "feature") not in ("feature", "teacher_joint", "teacher_balanced"):
        raise ValueError("Source centering requires an original supported hard initializer")
    if "alpha" in candidate and (isinstance(candidate["alpha"], bool) or not isinstance(candidate["alpha"], Real)
                                  or not math.isfinite(candidate["alpha"]) or candidate["alpha"] <= 0):
        raise ValueError("Source centering requires a finite positive initializer alpha")
    token = seal(source_digest())
    if candidate.get("mlp_source_centering_source_digest", token) != token:
        raise ValueError("Source centering numerical source changed; preserve its old namespace")
    candidate["mlp_source_centering_source_digest"] = token
    return candidate


def centered_nodes(inputs, parameters):
    """Return differentiable centered U, actual native mean, detached diagnostics."""
    first, bias, last, output_bias = parameters
    raw = (inputs @ first + bias).relu() @ last
    mean = raw.mean(0, keepdim=True)
    u = raw - mean + 0 * output_bias
    n = len(raw) + 2
    epsilon = torch.finfo(raw.dtype).eps
    if n * epsilon >= .1 or not bool(torch.isfinite(u).all()):
        raise FloatingPointError("Nonfinite or unsupported source centering reduction")
    bound = 4 * n * epsilon / (1 - n * epsilon) * max(1., float(raw.detach().abs().max()))
    residual = float(u.detach().double().mean(0).abs().max())
    if residual > bound:
        raise FloatingPointError("Native source mean is outside its declared roundoff bound")
    diagnostic = dict(center_mean_residual=residual, center_mean_bound=bound,
                      raw_output_mean_rms=float(mean.detach().double().square().mean().sqrt()),
                      raw_output_rms=float(raw.detach().double().square().mean().sqrt()),
                      centered_output_rms=float(u.detach().double().square().mean().sqrt()))
    return u, mean, diagnostic


@torch.no_grad()
def original_context(z, q, assignment, rank, hidden, seed, mixing, chunk_size, config, source=None):
    if (z.dtype != torch.float64 or q.dtype != torch.float64 or z.ndim != 2 or q.ndim != 2
            or len(z) != len(q) or assignment.shape != (len(z),) or assignment.dtype != torch.int64
            or min(z.shape) < 1 or min(q.shape) < 1 or not bool(torch.isfinite(z).all())
            or not bool(torch.isfinite(q).all()) or bool((q < 0).any())
            or float((q.sum(1) - 1).abs().max()) > 1e-12
            or z.device != q.device or z.device != assignment.device or bool((assignment < 0).any())
            or type(chunk_size) is not int or chunk_size < 1 or torch.get_default_dtype() != torch.float32):
        raise ValueError("Source centering requires aligned native float64 source inputs/float32 parameters")
    cells = int(assignment.max()) + 1
    if len(torch.unique(assignment)) != cells:
        raise ValueError("Original hard partition must contain every cell")
    inputs = assignment_inputs(z, q, "features")
    encoder, v = initialize_mlp(inputs, cells, rank, hidden, seed)
    u, mean, diagnostic = centered_nodes(inputs, encoder)
    context = dict(schema=SCHEMA, mode=MODE, policy=POLICY, device=str(z.device), nodes=len(z), cells=cells,
                   dimension=z.shape[1], classes=q.shape[1], rank=rank, hidden=hidden, seed=seed,
                   mixing=mixing, chunk_size=chunk_size, z=digest(z), q=digest(q), assignment=digest(assignment),
                   initializer=[digest(p) for p in [*encoder, v]], encoded_u0=digest(u), original_mean=digest(mean),
                   original_diagnostic=diagnostic, source=source, controls=config, helper_source=source_digest(),
                   torch_version=str(torch.__version__), default_dtype=str(torch.get_default_dtype()))
    return context, [*encoder, v]


def attach(saved, context, parameters=None, u=None, mean=None, diagnostic=None):
    saved.update(mass_mode="free", mlp_output_centering=MODE, mlp_source_centering_schema=SCHEMA,
                 source_centering_context=context)
    if parameters is not None:
        saved["parameters"] = [p.detach().cpu().clone() for p in parameters]
    if u is not None:
        saved.update(centered_u_digest=digest(u), raw_output_mean=mean.detach().cpu().clone(),
                     center_diagnostic=diagnostic)
    saved["source_centering_content_sha256"] = seal({key: value for key, value in saved.items()
                                                   if key != "source_centering_content_sha256"})
    return saved


def _identity(saved, context):
    if not isinstance(saved, dict):
        raise ValueError("Source centering cache must be a mapping")
    if (saved.get("mass_mode") != "free" or saved.get("mlp_output_centering") != MODE
            or type(saved.get("mlp_source_centering_schema")) is not int or saved["mlp_source_centering_schema"] != SCHEMA
            or set(saved) & ((_FORBIDDEN - {"raw_output_mean", "source_centering_context"}) | {"column_dual", "initial_mass_context"})):
        raise ValueError("Invalid source centering mode/schema/external mean or target")
    if seal(saved.get("source_centering_context")) != seal(context):
        raise ValueError("Source centering source/initializer/device/controls changed")
    if saved.get("source_centering_content_sha256") != seal({key: value for key, value in saved.items()
                                                           if key != "source_centering_content_sha256"}):
        raise ValueError("Source centering content checksum mismatch")


def _parameters(saved, original):
    parameters = saved.get("parameters")
    if not isinstance(parameters, (list, tuple)) or len(parameters) != 5:
        raise ValueError("Source centering must retain all five native parameters")
    for parameter, initial in zip(parameters, original, strict=True):
        _tensor(parameter, initial.shape, initial.dtype, "centered parameter")
    if not torch.equal(parameters[3], torch.zeros_like(parameters[3])):
        raise ValueError("Source centering output bias must remain exactly zero")
    return parameters


@torch.no_grad()
def validate_snapshot(saved, z, q, assignment, context, initial_parameters):
    _identity(saved, context)
    step = _step(saved.get("step"))
    parameters = _parameters(saved, initial_parameters)
    if step == 0 and [digest(p) for p in parameters] != context["initializer"]:
        raise ValueError("Centered checkpoint zero differs from its original initializer")
    k, d, c = context["cells"], context["dimension"], context["classes"]
    _tensor(saved.get("moments"), (k, 1 + d + c), torch.float64, "centered moments")
    _tensor(saved.get("theta"), (c, 1 + d), torch.float64, "centered head")
    u, mean, diagnostic = centered_nodes(assignment_inputs(z, q, "features"), [p.to(z.device) for p in parameters[:-1]])
    _tensor(saved.get("raw_output_mean"), mean.shape, mean.dtype, "current raw output mean")
    if (not torch.equal(saved["raw_output_mean"].cpu(), mean.cpu()) or saved.get("centered_u_digest") != digest(u)
            or seal(saved.get("center_diagnostic")) != seal(diagnostic)):
        raise ValueError("Centered snapshot does not retain its recomputed current all-source mean/U")
    replay = LowRankMoments.apply(u, parameters[-1].to(z.device), assignment, make_material(z, q),
                                 context["mixing"], context["chunk_size"])
    if not bool(torch.isfinite(replay).all()) or bool((replay[:, 0] <= 0).any()):
        raise ValueError("Invalid free-mass source-derived centered moments")
    for start in range(0, len(z), context["chunk_size"]):
        probability = logit_block(u[start:start + context["chunk_size"]], parameters[-1].to(z.device),
                                  assignment[start:start + context["chunk_size"]], context["mixing"]).double().softmax(1)
        if float((probability.sum(1) - 1).abs().max()) > 1e-12:
            raise ValueError("Centered probability row stochasticity failed")
    if float((replay.sum(0) - make_material(z, q).mean(0)).abs().max()) > 1e-12:
        raise ValueError("Centered source material conservation failed")
    if not torch.equal(replay.cpu(), saved["moments"].cpu()):
        raise ValueError("Centered snapshot moments do not replay from its actual parameters")
    from src.soft_ce_partition import head_gradient, outer_value_gradient
    centers, labels, mass = decode_moments(replay, d)
    maximum = float(head_gradient(augmented(centers), labels, torch.full_like(mass, 1 / k),
                                  saved["theta"].to(z), context["controls"]["penalty"]).abs().max())
    if (saved.get("J_exact") is not True or not math.isfinite(maximum)
            or maximum > context["controls"]["inner_tol"] * (1 + 1e-6)):
        raise ValueError("Centered checkpoint CE head is uncertified")
    # Certificates replay the actual fixed source/head, rather than trusting a
    # resealed pair of history J0 and normalization scale. Scalar replay allows
    # only double arithmetic roundoff; it never refits a head or changes P.
    actual_outer, _ = outer_value_gradient(z, q, saved["theta"].to(z),
                                            context["controls"]["outer_chunk_size"], augmented(z))
    for key, actual in (("teacher_ce", actual_outer), ("inner_grad_max", maximum)):
        scalar = saved.get(key)
        if (type(scalar) is not float or not math.isfinite(scalar) or scalar < 0
                or not math.isclose(scalar, actual, rel_tol=1e-12, abs_tol=1e-12)):
            raise ValueError("Centered saved outer CE/stationarity certificate differs from its actual head/source")
    return replay


def validate_resume(saved, z, q, assignment, context, initial_parameters, steps, folder=None):
    _identity(saved, context)
    step = _step(saved.get("step"))
    if step > steps or saved.get("config") != context["controls"]:
        raise ValueError("Centered resume configuration/budget differs")
    _optimizer(saved, initial_parameters, step, context["controls"]["lr"])
    for state in saved["optimizer"]["state"].values():
        if state["step"].device.type != "cpu":
            raise ValueError("Native noncapturable Adam counter must be CPU")
    if step and any(bool(saved["optimizer"]["state"][3][key].ne(0).any()) for key in ("exp_avg", "exp_avg_sq")):
        raise ValueError("Centered output-bias Adam moments must remain exactly zero")
    snapshots = saved.get("snapshots")
    if not isinstance(snapshots, dict) or 0 not in snapshots or step not in snapshots:
        raise ValueError("Centered resume requires original/current verifiable snapshots")
    for key, snapshot in snapshots.items():
        if not isinstance(snapshot, dict) or type(key) is not int or key > step or snapshot.get("step") != key:
            raise ValueError("Malformed centered snapshot history")
        validate_snapshot(snapshot, z, q, assignment, context, initial_parameters)
    parameters = _parameters(saved, initial_parameters)
    if any(not torch.equal(p, current) for p, current in zip(parameters, snapshots[step]["parameters"], strict=True)):
        raise ValueError("Centered resume parameters differ from its current snapshot")
    if saved.get("dual") is not None:
        raise ValueError("Free centered assignments have no column dual")
    _tensor(saved.get("theta"), snapshots[step]["theta"].shape, torch.float64, "centered resume head")
    if not torch.equal(saved["theta"], snapshots[step]["theta"]):
        raise ValueError("Centered resume head differs from its snapshot")
    history = saved.get("history")
    if (not isinstance(history, list) or len(history) != step + 1
            or any(not isinstance(row, dict) or type(row.get("step")) is not int or row["step"] != index
                   for index, row in enumerate(history))):
        raise ValueError("Incomplete centered resume history")
    for row in history:
        for key in ("J", "inner_grad_max", "center_mean_residual", "center_mean_bound", "seconds"):
            value = row.get(key)
            if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
                raise ValueError("Invalid centered history certificate")
        if (row.get("inner_converged") is not True or row.get("J_exact") is not True
                or row["inner_grad_max"] > context["controls"]["inner_tol"] * (1 + 1e-6)
                or row["center_mean_residual"] > row["center_mean_bound"]):
            raise ValueError("Centered history contains an uncertified endpoint")
        if row["step"] < step and (row.get("cg_converged") is not True
                or type(row.get("cg_relative_residual")) not in (float, int)
                or not math.isfinite(row["cg_relative_residual"])
                or row["cg_relative_residual"] > context["controls"]["cg_rtol"] * (1 + 1e-6)):
            raise ValueError("Centered history contains an uncertified implicit update")
    for checkpoint, snapshot in snapshots.items():
        row = history[checkpoint]
        if (type(row.get("J")) is not float or type(row.get("inner_grad_max")) is not float
                or row["J"] != snapshot["teacher_ce"] or row["inner_grad_max"] != snapshot["inner_grad_max"]):
            raise ValueError("Centered history differs from its verified snapshot outer CE/stationarity certificate")
    _tensor(saved.get("initial_moments"), snapshots[0]["moments"].shape, torch.float64, "centered original moments")
    if not torch.equal(saved["initial_moments"], snapshots[0]["moments"]):
        raise ValueError("Centered original moments differ")
    if (type(saved.get("scale")) not in (float, int) or saved["scale"] != max(snapshots[0]["teacher_ce"], 1e-12)
            or type(saved.get("elapsed")) not in (float, int) or not math.isfinite(saved["elapsed"])
            or saved["elapsed"] != history[-1]["seconds"] or type(saved.get("best_step")) is not int
            or not 0 <= saved["best_step"] <= step or saved.get("best") != min(row["J"] for row in history)):
        raise ValueError("Invalid centered objective/scale/history controls")
    _tensor(saved.get("best_moments"), snapshots[0]["moments"].shape, torch.float64, "centered best moments")
    _tensor(saved.get("best_theta"), snapshots[0]["theta"].shape, torch.float64, "centered best head")
    if saved["best_step"] in snapshots and (not torch.equal(saved["best_moments"], snapshots[saved["best_step"]]["moments"])
            or not torch.equal(saved["best_theta"], snapshots[saved["best_step"]]["theta"])):
        raise ValueError("Centered best checkpoint differs from its recorded snapshot")
    if folder is not None:
        for path in sorted((Path(folder) / "checkpoints").glob("step_*.pt")):
            disk = torch.load(path, map_location="cpu", weights_only=False)
            if not isinstance(disk, dict):
                raise ValueError("Malformed centered disk checkpoint")
            key = _step(disk.get("step"))
            if key not in snapshots or seal(disk) != seal(snapshots[key]):
                raise ValueError("Orphan or changed centered disk checkpoint")
    return saved


def citation_core_kwargs(candidate, seed, source):
    return dict(penalty=candidate["penalty"], lr=candidate["lr"], mixing=candidate.get("mixing", .05),
                assignment_rank=candidate["rank"], factor_seed=seed, assignment_input="features",
                assignment_encoder="mlp", encoder_hidden=candidate["width"], solver_mode="exact",
                inner_method="newton_first", implicit_warm_start=True, mass_mode="free",
                inner_loss_weighting="uniform", inner_max_iter=2000, inner_tol=1e-7,
                cg_max_iter=512, cg_rtol=1e-6, cache_assignment=False, save_assignment=False, save_resume=True,
                mlp_output_centering=MODE, mlp_source_centering_schema=SCHEMA, mlp_source_centering_source=source)


def expected_citation_config(candidate, seed, z, q, assignment, source):
    from src.soft_ce_partition import optimize_ce_assignment
    values = {key: value.default for key, value in inspect.signature(optimize_ce_assignment).parameters.items()
              if value.default is not inspect.Parameter.empty}
    values.update(citation_core_kwargs(candidate, seed, source))
    for key in ("steps", "folder", "checkpoint_steps", "resume_state", "save_resume", "log_every", "initial_representatives",
                "outer_indices", "implicit_solver", "inner_solver", "temperature_logits", "outer_targets", "stop",
                "temperature_initial", "temperature_lr", "cg_check_interval", "cache_assignment", "node_weighting",
                "node_weight_penalty", "node_weight_lr"):
        values.pop(key, None)
    values["data_digest"] = array_digest(z.detach().cpu().numpy(), q.detach().cpu().numpy(), assignment.cpu().numpy())
    return values


def validate_folder(root, candidate, seed, z, q, assignment, source, steps, snapshot=None):
    from src.io import _fingerprint
    folder = Path(root) / _fingerprint(candidate) / f"condensation_{seed}"
    recorded = folder.parent / "candidate.json"
    if recorded.exists() and json.loads(recorded.read_text()) != candidate:
        raise ValueError("Recorded centered candidate differs")
    config = expected_citation_config(candidate, seed, z, q, assignment, source)
    context, parameters = original_context(z, q, assignment, candidate["rank"], candidate["width"], seed,
                                           candidate.get("mixing", .05), 4096, config, source)
    resume = folder / "resume.pt"
    if resume.exists():
        saved = torch.load(resume, map_location="cpu", weights_only=False)
        if not isinstance(saved, dict):
            raise ValueError("Malformed centered resume")
        validate_resume(saved, z, q, assignment, context, parameters, max(steps, _step(saved.get("step"))), folder)
    elif snapshot is not None or any((folder / "checkpoints").glob("step_*.pt")):
        raise ValueError("Centered cached endpoints require a verifiable resume")
    if snapshot is not None:
        validate_snapshot(snapshot, z, q, assignment, context, parameters)
    return context, parameters


def prepare_probe(dataset, ratio, output_dir, candidate, condensation_seed=0, data_dir="data",
                  device="cuda", citation_features="default", stop=lambda: False):
    from src import citation_search
    from src.data import BUDGET
    from src.io import _fingerprint, save_json
    from src.soft_ce_partition import optimize_ce_assignment
    from src.target_refinement import training_refined_targets
    candidate = candidate_controls(citation_search._candidate_surrogate(citation_search._candidate_nystrom_mass(
        citation_search._candidate_temperature(citation_search._candidate_background_mixing(candidate)[0]))))
    if not active(candidate) or (dataset, ratio) not in BUDGET or dataset not in ("cora", "citeseer"):
        raise ValueError("Centered probe requires a supported MLP citation candidate")
    if not callable(stop):
        raise ValueError("stop must be callable")
    if stop():
        raise InterruptedError("Centered probe stopped")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    graph, train, validation, testing, h = citation_search._prepare_dataset(dataset, data_dir, device, citation_features)
    config = citation_search._legacy_teacher_config(dataset, ratio, graph, train, validation, testing, citation_features)
    root = Path(output_dir) / dataset / f"ratio_{ratio}" / _fingerprint(config)
    if not (root / "teacher.pt").is_file():
        raise ValueError("Centered probe requires original cached ReLU teacher")
    logits = torch.load(root / "teacher.pt", map_location=device, weights_only=False)["logits"]
    q = training_refined_targets(logits, candidate["T"], graph["y"], train, 0)
    h, z, _, assignment, source = cached_source(root, candidate, condensation_seed, h, q, config)
    folder = root / _fingerprint(candidate) / f"condensation_{condensation_seed}"
    validate_folder(root, candidate, condensation_seed, z, q, assignment, source, 1)
    resume = folder / "resume.pt"
    state = torch.load(resume, map_location="cpu", weights_only=False) if resume.exists() else None
    cached = state is not None and state["step"] >= 1
    if not cached:
        if not (folder.parent / "candidate.json").exists():
            folder.parent.mkdir(parents=True, exist_ok=True)
            save_json(candidate, folder.parent / "candidate.json")
        optimize_ce_assignment(z, q, assignment, steps=1, folder=folder, checkpoint_steps=(0, 1),
                               resume_state=state, stop=stop, **citation_core_kwargs(candidate, condensation_seed, source))
    validate_folder(root, candidate, condensation_seed, z, q, assignment, source, 1)
    state = torch.load(resume, map_location="cpu", weights_only=False)
    row = next(row for row in state["history"] if row["step"] == 1)
    return dict(root=str(root.resolve()), candidate_path=str(folder.parent.resolve()), step=1, cached=cached,
                validation_only=True, student_fits=0, inner_grad_max=row["inner_grad_max"], inner_converged=row["inner_converged"],
                center_mean_residual=row["center_mean_residual"], center_mean_bound=row["center_mean_bound"],
                min_mass=row["min_mass"], max_mass=row["max_mass"], effective_cells=row["effective_cells"], seconds=row["seconds"])
