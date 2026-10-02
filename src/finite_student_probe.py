"""Bounded source-derived finite-student probe; one NODE-P Adam update, no students.

Production-candidate draft only. It never creates missing source assets or
accepts legacy linear/CG/stationarity artifacts as finite-model certificates.
An atomic sealed resume0/1 bundle is authoritative; checkpoints/CSV are mirrors.
"""
import csv
import hashlib
import io
import json
import math
import os
import uuid
from pathlib import Path

import numpy as np
import scipy
import torch
import torch_geometric

from src import citation_search
from src import source_linear_assignment as source_helper
from src.evaluation import _input_digest
from src.io import _fingerprint, array_digest, cpu_state
from src.low_rank_assignment import LowRankLogits, LowRankMoments, initialize_factors
from src.moments import augmented, decode_moments, make_material
from src.soft_ce_partition import head_gradient, outer_value_gradient
from src.sweep_utils import representative
from src.target_refinement import training_refined_targets

from .finite_student_outer import (
    INITIAL_SEED,
    LEARNING_RATE,
    STEPS,
    WEIGHT_DECAY,
    finite_student_outer_partials,
    geom_uniform_initial,
)

SCHEMA = 2
ENGINE_SHA = "35c3457e4e1bc25ab140b7f634b956cd4d7202d7bcc40095fbdf6aa20e1eee00"
ROOT_NAME = "19fccc37cc2f"
REFERENCE_ID = "c24c1ffc2b76"
_FIXED = dict(method="finite_student", width=0, lr=.05, T=.3, rank=32, penalty=.001,
              initialization="teacher_balanced", alpha=.3, inner_loss_weighting="uniform",
              mass_mode="free", mixing=.05, finite_student_schema=2,
              finite_model_steps=5, finite_model_lr=.1, finite_model_hidden=256,
              finite_model_seed=0, finite_model_weight_decay=.001,
              finite_graph_backend="dense_original_S", finite_native_policy="deterministic_cuda_v1")
_SOURCE_FIELD = "finite_student_source_digest"
_ROUTES = ("sgc_mlp", "gcn")


def _require(value, message):
    if not value:
        raise ValueError(message)


def _stop(stop):
    _require(callable(stop), "stop must be callable")
    if stop():
        raise InterruptedError("Finite student probe interrupted")


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value):
    if torch.is_tensor(value):
        if value.layout != torch.strided:
            _require(value.layout in (torch.sparse_csr, torch.sparse_coo), "Unsupported sparse tensor")
            if value.layout == torch.sparse_csr:
                arrays = [value.crow_indices(), value.col_indices(), value.values()]
            else:
                value = value.coalesce()
                arrays = [value.indices(), value.values()]
            return dict(shape=list(value.shape), dtype=str(value.dtype), layout=str(value.layout),
                        parts=[_digest(part) for part in arrays])
        return dict(shape=list(value.shape), dtype=str(value.dtype), tensor=array_digest(value.detach().cpu().numpy()))
    if isinstance(value, dict):
        return {str(key): _digest(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_digest(item) for item in value]
    return value


def _seal(value):
    return hashlib.sha256(json.dumps(_digest(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def numerical_source():
    """Full src manifest, including obsolete prototype bytes as inventory ONLY."""
    base = Path(__file__).resolve().parent
    manifest = {path.name: _sha(path) for path in sorted(base.glob("*.py"))}
    _require(manifest.get("finite_student_outer.py") == ENGINE_SHA
             and "finite_student_probe.py" in manifest and "research_loop.py" in manifest,
             "Actual production engine/fullsrc manifest is missing or changed")
    return dict(files=manifest, versions=dict(torch=str(torch.__version__), pyg=str(torch_geometric.__version__),
                                             numpy=str(np.__version__), scipy=str(scipy.__version__)))


def canonical_candidate(candidate):
    _require(isinstance(candidate, dict), "Finite student candidate must be a mapping")
    allowed = set(_FIXED) | {_SOURCE_FIELD, "outer_route"}
    _require(set(candidate) <= allowed and set(_FIXED) | {"outer_route"} <= set(candidate),
             "Explicit finite student controls required; unknown/legacy/source/target overrides forbidden")
    result = dict(candidate)
    for key, expected in _FIXED.items():
        actual = candidate[key]
        if type(expected) is int:
            _require(type(actual) is int and actual == expected, f"Changed fixed finite control: {key}")
        elif type(expected) is float:
            _require(type(actual) in (int, float) and math.isfinite(actual) and actual == expected,
                     f"Changed fixed finite control: {key}")
        else:
            _require(type(actual) is str and actual == expected, f"Changed fixed finite control: {key}")
        result[key] = expected
    _require(result["outer_route"] in _ROUTES and type(result["outer_route"]) is str, "Unsupported finite outer route")
    _require((STEPS, LEARNING_RATE, WEIGHT_DECAY, INITIAL_SEED) == (5, .1, .001, 0), "Frozen engine policy changed")
    token = _seal(numerical_source())
    _require(result.get(_SOURCE_FIELD, token) == token, "Finite source changed; preserve old namespace and refreeze")
    result[_SOURCE_FIELD] = token
    return result


def _runtime_precision_guard():
    """Assert the frozen native policy without repairing changed runtime flags."""
    _require(os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
             and torch.are_deterministic_algorithms_enabled()
             and not torch.is_deterministic_algorithms_warn_only_enabled()
             and torch.get_float32_matmul_precision() == "highest"
             and torch.get_default_dtype() == torch.float32
             and not torch.is_autocast_enabled() and not torch.is_autocast_enabled("cpu")
             and not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32,
             "Finite deterministic CUDA precision/environment changed")


def _native(device):
    # Pin this in the fresh child environment before Python/CUDA initialization.
    # Never attempt to repair a missing value after CUDA initialization.
    _require(os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8",
             "Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before CUDA initialization")
    _require(device == "cuda" and torch.get_default_dtype() == torch.float32,
             "Finite probe requires CUDA/nativeFP32/defaultFP32")
    _require(not torch.is_autocast_enabled() and not torch.is_autocast_enabled("cpu"), "Finite probe forbids external AMP")
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    _runtime_precision_guard()
    _require(torch.cuda.is_available(), "Finite probe requires available CUDA")
    return dict(device=str(torch.device("cuda", torch.cuda.current_device())), GPU=torch.cuda.get_device_name(),
                torch=str(torch.__version__), cuda_runtime=str(torch.version.cuda), default_dtype=str(torch.get_default_dtype()),
                AMP=False, TF32=False, deterministic_algorithms=True, deterministic_warn_only=False,
                float32_matmul_precision="highest", CUBLAS_WORKSPACE_CONFIG=":4096:8",
                finite_graph_backend="dense_original_S", finite_native_policy="deterministic_cuda_v1")


def _scalar(value, message):
    _require(type(value) in (int, float) and math.isfinite(value) and value >= 0, message)
    return value


def _tensor(value, shape, dtype, message):
    _require(torch.is_tensor(value) and value.layout == torch.strided and tuple(value.shape) == tuple(shape)
             and value.dtype == dtype and bool(torch.isfinite(value).all()), message)
    return value


def _checked_files(mapping):
    _require(isinstance(mapping, dict) and bool(mapping), "Missing frozen source assets")
    for path, digest in mapping.items():
        _require(_sha(path) == digest, f"Frozen source file changed: {path}")


def _reference_candidate():
    return dict(method="low_rank", width=0, lr=.05, T=.3, rank=32, penalty=1e-4,
                initialization="teacher_balanced", alpha=.3, inner_loss_weighting="uniform")


def _public_options(dataset, ratio, output_dir, condensation_seed, data_dir, device, citation_features):
    repo = Path(__file__).resolve().parents[1]
    _require(dataset == "cora" and type(ratio) in (int, float) and ratio == .026
             and citation_features == "row" and type(condensation_seed) is int and condensation_seed == 0
             and device == "cuda", "First finite probe pins Cora70ROW/seed0/CUDA")
    _require(Path(output_dir).resolve() == repo / "results/citation_search_v1" and Path(data_dir).resolve() == repo / "data",
             "First finite probe forbids external source/output overrides")


def _frozen_transform(transform):
    """Clone original runtime RMS buffers without serialization/device transfer."""
    return {key: value.detach().clone() if torch.is_tensor(value) else value
            for key, value in vars(transform).items()}


def _frozen_original_dense_S(original):
    """Materialize existing CSR coefficients only; preserve the original graph."""
    _require(torch.is_tensor(original) and original.layout == torch.sparse_csr
             and original.dtype == torch.float32 and original.ndim == 2
             and original.shape[0] == original.shape[1] and original.shape[0] > 0
             and not original.requires_grad and bool(torch.isfinite(original.values()).all())
             and bool((original.values() != 0).all()),
             "Require frozen finite original FP32 CSR graph without explicit zeros")
    dense = original.detach().to_dense().clone().contiguous()
    _require(dense.device == original.device and dense.dtype == torch.float32 and not dense.requires_grad
             and _seal(dense.to_sparse_csr()) == _seal(original),
             "Dense graph differs from exact original CSR entries or zero edges")
    return dense


def _load_source(dataset, ratio, output_dir, candidate, condensation_seed, data_dir, device, citation_features, stop):
    repo = Path(__file__).resolve().parents[1]
    _require(dataset == "cora" and type(ratio) in (int, float) and ratio == .026
             and citation_features == "row" and type(condensation_seed) is int and condensation_seed == 0,
             "First finite probe pins Cora70ROW/seed0")
    _require(Path(output_dir).resolve() == repo / "results/citation_search_v1" and Path(data_dir).resolve() == repo / "data",
             "First finite probe forbids external source/output overrides")
    root = Path(output_dir) / dataset / "ratio_0.026" / ROOT_NAME
    origin = dict(mode="teacher_balanced", alpha=.3, T=.3, seed=0)
    assets = [root / name for name in ("config.json", "propagated_H.pt", "teacher.pt", "inputs_0.pt",
                                      "nystrom_map_schema3.pt", "nystrom_phi_schema3.npy", "nystrom_phi_schema3.meta.json")]
    assets += [Path(data_dir) / "cora/raw" / f"ind.cora.{name}" for name in
               ("x", "tx", "allx", "y", "ty", "ally", "graph", "test.index")]
    assets += [path for path in (Path(data_dir) / "cora/processed").glob("*.pt")]
    assets += [root / f"assignment_{_fingerprint(origin)}.pt", root / REFERENCE_ID / "candidate.json",
               root / REFERENCE_ID / "condensation_0/resume.pt", root / REFERENCE_ID / "condensation_0/checkpoints/step_000000.pt"]
    _require(all(path.is_file() for path in assets) and (Path(data_dir) / "cora/processed/data.pt").is_file(),
             "Require existing original teacher/H/RMS/hard/map/Phi/reference/processeddata; no creation fallback")
    pins = {str(path.resolve()): _sha(path) for path in assets}
    _stop(stop)
    graph, train, val, test, fresh_h = citation_search._prepare_dataset(dataset, data_dir, device, citation_features)
    config = citation_search._legacy_teacher_config(dataset, ratio, graph, train, val, test, citation_features)
    _require(config == json.loads((root / "config.json").read_text()) and _fingerprint(config) == ROOT_NAME,
             "Current graph/nodeorder/preprocessing/splits differ from frozen source")
    logits = torch.load(root / "teacher.pt", map_location=device, weights_only=False)["logits"]
    q = training_refined_targets(logits, .3, graph["y"], train, 0)
    # Ghost is only an internal require-existing shared-source loader. Its
    # ORIGINAL linear-reference penalty is1e-4, never the active finite .001.
    ghost = source_helper.candidate_controls(dict(_reference_candidate(), method="source_linear",
                 assignment_coordinates="raw_rms", source_linear_schema=1))
    h, z, transform, hard, _, source = source_helper.cached_source(root, ghost, 0, fresh_h, q, config)
    _require(int(hard.max())+1 == 70 and len(torch.unique(hard)) == 70
             and z.shape == (2708, 1433) and q.shape == (2708, 7), "Original Cora70 source dimensions/partition changed")
    model_initial = geom_uniform_initial(1433, 7, hidden=256, dtype=torch.float32, device=device)
    u, v = initialize_factors(hard, 70, 32, 0)
    original_S = graph["adj"]
    dense_S = _frozen_original_dense_S(original_S)
    buffers = dict(z=z.detach(), q=q.detach(), h=h.float().detach(), hard=hard.detach(), transform=_frozen_transform(transform),
                   x=graph["x"].detach(), S=dense_S, original_S=original_S.detach(), model_initial=model_initial, initial=[u.detach(), v.detach()],
                   graph=graph, train=train, val=val[1], config=config, ghost=ghost, source=source, root=root, pins=pins)
    _checked_files(pins)
    _stop(stop)
    return buffers


def _moments(buffers, parameters):
    return LowRankMoments.apply(parameters[0], parameters[1], buffers["hard"],
                                make_material(buffers["z"], buffers["q"]), .05, 4096)


def _p0_reference(buffers):
    root = buffers["root"] / REFERENCE_ID
    _require(json.loads((root / "candidate.json").read_text()) == _reference_candidate(), "Cached original NODE candidate changed")
    resume = torch.load(root / "condensation_0/resume.pt", map_location="cpu", weights_only=False)
    snapshot = torch.load(root / "condensation_0/checkpoints/step_000000.pt", map_location="cpu", weights_only=False)
    _require(isinstance(resume, dict) and isinstance(snapshot, dict) and type(snapshot.get("step")) is int
             and snapshot["step"] == 0 and 0 in resume.get("snapshots", {})
             and _seal(snapshot) == _seal(resume["snapshots"][0]), "Original NODE P0 schema/resume agreement changed")
    expected = source_helper.expected_citation_config(buffers["ghost"], 0, buffers["z"], buffers["q"],
                 buffers["hard"], buffers["z"].detach().float(), buffers["source"])
    for key in ("source_linear_coordinates", "source_linear_schema", "source_linear_source"):
        expected.pop(key)
    expected["assignment_input"] = "node"
    # Historical save_assignment affects only artifact retention; bind its
    # explicit native bool rather than silently assuming moderndefaultFalse.
    retained = resume.get("config", {}).get("save_assignment")
    _require(type(retained) is bool, "Original NODE artifact retention flag malformed")
    expected["save_assignment"] = retained
    _require(resume["config"] == expected, "Original NODE source/config/schema is incompatible")
    # Legacy checkpoint0 has no factor parameters. Verify TWO current native
    # initializers from the original paired hard/config, without a retrospective
    # historical-V0 assertion; its saved artifacts certify moments/FP32inputs.
    baseline_u, baseline_v = initialize_factors(buffers["hard"], 70, expected["assignment_rank"], expected["factor_seed"])
    _require(torch.equal(baseline_u, buffers["initial"][0]) and torch.equal(baseline_v, buffers["initial"][1]),
             "Paired current native baseline U0/V0 differ")
    historical_best0_verified = False
    if resume.get("best_step") == 0 and resume.get("best_parameters") is not None:
        retained_parameters = resume["best_parameters"]
        _require(isinstance(retained_parameters, (list, tuple)) and len(retained_parameters) == 2,
                 "Available historical best0 native parameters malformed")
        for value, initial in zip(retained_parameters, buffers["initial"], strict=True):
            _tensor(value, initial.shape, torch.float32, "Available historical best0 native parameters malformed")
            _require(torch.equal(value.to(initial), initial), "Available historical best0 native parameters differ")
        historical_best0_verified = True
    actual_moments = _moments(buffers, buffers["initial"])
    old = _tensor(snapshot["moments"], actual_moments.shape, torch.float64, "Original NODE moments malformed")
    _require(torch.allclose(old.to(actual_moments), actual_moments, atol=1e-12, rtol=1e-12), "Original source-derived P0 moments differ")
    actual_x, actual_q, mass = representative(actual_moments, _transform(buffers), 1433, "cuda")
    old_x, old_q, old_mass = representative(old, _transform(buffers), 1433, "cuda")
    weights, old_weights = torch.full_like(mass, 1/70), torch.full_like(old_mass, 1/70)
    _require(all(torch.equal(a, b) for a, b in zip((actual_x, actual_q, weights), (old_x, old_q, old_weights), strict=True)),
             "Actual FP32 P0 X/Q/supplieduniformweights differ from cached NODE")
    centers, labels, mass = decode_moments(old.to(buffers["z"]), 1433)
    theta = _tensor(snapshot["theta"], (7, 1434), torch.float64, "Original linear reference head malformed").to(buffers["z"])
    gradient = float(head_gradient(augmented(centers), labels, torch.full_like(mass, 1/70), theta, 1e-4).abs().max())
    value, _ = outer_value_gradient(buffers["z"], buffers["q"], theta, 65536, augmented(buffers["z"]))
    _require(snapshot["J_exact"] is True and math.isfinite(gradient) and gradient <= 1e-7*(1+1e-6)
             and math.isclose(gradient, snapshot["inner_grad_max"], abs_tol=1e-12, rel_tol=1e-12)
             and math.isclose(value, snapshot["teacher_ce"], abs_tol=1e-12, rel_tol=1e-12), "Cached original linear P0 certificate differs")
    actual_digest = _input_digest(actual_x, actual_q, weights, buffers["graph"], buffers["q"],
                                 dict(train=buffers["train"], val=buffers["val"]), None, lambda: False)
    return dict(reference_candidate=_reference_candidate(), reference_id=REFERENCE_ID,
                old_retention_flag=retained, old_linear_penalty=1e-4,
                manual_native_baseline_U0_V0_exact=True, historical_snapshot0_UV_available=False,
                optional_historical_best0_parameters_verified=historical_best0_verified,
                paired_current_native_initializer=_digest([baseline_u, baseline_v]),
                native_initial_moments=_digest(actual_moments), actual_FP32_X_Q_uniform_weights_exact=True,
                input_digest=actual_digest, double_moment_max_abs=float((old.to(actual_moments)-actual_moments).abs().max()))


def _transform(buffers):
    from src.transforms import FeatureTransform
    return FeatureTransform(**buffers["transform"])


def _context(candidate, buffers, environment, reference):
    from src.research_loop import implementation_provenance
    return dict(schema=SCHEMA, candidate=candidate, numerical_source=numerical_source(), native_environment=environment,
                implementation=implementation_provenance(),
                dataset_config=buffers["config"], ghost_source=buffers["source"], source_pins=buffers["pins"],
                source_buffers=_digest({key: buffers[key] for key in ("x", "S", "original_S", "h", "q", "z", "hard", "transform")}),
                source_initial=_digest(buffers["initial"]), model_initial=_digest(buffers["model_initial"]), reference=reference,
                factory_seed=0, outer_optimizer=dict(name="Adam", lr=.05, betas=[.9, .999], eps=1e-12, weight_decay=0, foreach=False, fused=False),
                inner_policy=dict(name="functionalfullbatchSGD", steps=5, lr=.1, weight_decay=.001, all_parameters=True,
                                  hidden=256, dropout=0, identity_adjacency=True, uniform_CE=True, reset_every_P=True),
                graph_backend=dict(kind="dense_original_S", original_CSR=_digest(buffers["original_S"]),
                                   runtime_dense_S=_digest(buffers["S"]), exact_CSR_roundtrip=True,
                                   dtype=str(buffers["S"].dtype), device=str(buffers["S"].device),
                                   normalization_or_synthesized_edges=False),
                source_H_graph_S="independent original SciPyH /exactdenseoriginalPyGgraphS; no H equality/recompute claim",
                own_J0_scale="positive frozen initial selected-route teacherCE", allowed_steps=[0, 1], student_fits=0)


def _source_unchanged(context):
    _runtime_precision_guard()
    from src.research_loop import implementation_provenance
    _require(numerical_source() == context["numerical_source"] and implementation_provenance() == context["implementation"],
             "Finite numerical source or recorded Git lineage changed")


def _evaluate(buffers, candidate, context, parameters, step, scale, stop):
    _stop(stop)
    _runtime_precision_guard()
    current = [parameter.detach().clone().requires_grad_(True) for parameter in parameters]
    moments = _moments(buffers, current)
    result = finite_student_outer_partials(moments, buffers["transform"], buffers["model_initial"],
                 buffers["x"], buffers["S"], buffers["h"], buffers["q"], stop=stop)
    route = candidate["outer_route"]
    loss = float(result["outer_losses"][route])
    _scalar(loss, "Nonfinite finite-model source CE")
    if scale is None:
        _require(step == 0 and loss > 0, "Original finite J0 must be positive")
        scale = max(loss, 1e-12)
    moments.backward(result["moment_gradients"][route] / scale)
    gradients = [parameter.grad.detach().clone() for parameter in current]
    _require(all(bool(torch.isfinite(value).all()) for value in gradients), "Nonfinite finite outer factor gradient")
    logits = LowRankLogits.apply(current[0], current[1], buffers["hard"], .05, 4096)
    probability = logits.double().softmax(1)
    _, labels, mass = decode_moments(moments, buffers["z"].shape[1])
    conservation = dict(row=float((probability.sum(1)-1).abs().max()), mass_sum=float((mass.sum()-1).abs()),
        material=float((moments.sum(0)-make_material(buffers["z"], buffers["q"]).mean(0)).abs().max()),
        Qc_simplex=float((labels.sum(1)-1).abs().max()))
    _require(bool((mass > 0).all()) and all(math.isfinite(value) and value <= 1e-12 for value in conservation.values()),
             "Native source P/material conservation failed")
    snapshot = dict(schema=SCHEMA, context=context, step=step, parameters=current, moments=moments,
        teacher_ce=loss, scale=scale, finite_model_steps=5, selected_route=route,
        moment_partial=result["moment_gradients"][route], scaled_factor_gradients=gradients,
        adapted_model=result["adapted_parameters"], inner_parameter_states=result["inner_parameter_states"], inner_trace=result["inner_trace"],
        paired_outer_losses={name: float(value) for name, value in result["outer_losses"].items()}, conservation=conservation,
        encoded_U=_digest(current[0]), raw_logits=_digest(logits), min_mass=float(mass.min()), max_mass=float(mass.max()),
        effective_cells=float(1/mass.square().sum()))
    return cpu_state(snapshot), scale


def _first_adam(initial, gradient):
    """Readonly first-step native Adam certificate, not an optimizer invocation."""
    first = torch.zeros_like(gradient).lerp_(gradient, 1-.9)
    second = torch.zeros_like(gradient).addcmul_(gradient, gradient, value=1-.999)
    denominator = second.sqrt().div_(math.sqrt(1-.999)).add_(1e-12)
    updated = initial.detach().clone().addcdiv_(first, denominator, value=-.05/(1-.9))
    return updated, first, second


def _optimizer(saved, parameters, gradients, step):
    _require(isinstance(saved, dict) and set(saved) == {"state", "param_groups"}
             and isinstance(saved["param_groups"], list) and len(saved["param_groups"]) == 1, "Malformed finite Adam")
    group = saved["param_groups"][0]
    expected = dict(lr=.05, betas=[.9, .999], eps=1e-12, weight_decay=0, amsgrad=False, maximize=False,
                    foreach=False, capturable=False, differentiable=False, fused=False)
    _require(isinstance(group, dict) and set(group) in (set(expected)|{"params"}, set(expected)|{"params", "decoupled_weight_decay"})
             and group.get("params") == [0, 1] and all(type(value) is int for value in group["params"]), "Finite Adam controls/IDs changed")
    for key, value in expected.items():
        actual = group[key]
        if key == "betas":
            _require(isinstance(actual, (list, tuple)) and len(actual) == 2 and all(type(x) in (int, float) for x in actual)
                     and list(actual) == value, "Finite Adam betas changed")
        elif value is None or type(value) is bool:
            _require(actual is value, "Finite Adam flags changed")
        else:
            _require(type(actual) in (int, float) and math.isfinite(actual) and actual == value, "Finite Adam scalar changed")
    if "decoupled_weight_decay" in group:
        _require(group["decoupled_weight_decay"] is False, "Finite Adam decay changed")
    slots = saved["state"]
    _require(isinstance(slots, dict) and all(type(key) is int for key in slots)
             and set(slots) == (set() if step == 0 else {0, 1}), "Finite Adam slots/counters changed")
    if step == 0:
        return
    for index, (initial, gradient) in enumerate(zip(parameters, gradients, strict=True)):
        slot = slots[index]
        _require(isinstance(slot, dict) and set(slot) == {"step", "exp_avg", "exp_avg_sq"}, "Malformed finite Adam slot")
        counter = _tensor(slot["step"], (), torch.float32, "Malformed finite Adam CPU counter")
        _require(counter.device.type == "cpu" and float(counter) == 1, "Finite Adam counter/device changed")
        _, first, second = _first_adam(initial, gradient)
        _require(torch.equal(_tensor(slot["exp_avg"], initial.shape, torch.float32, "Malformed finite Adam first moment").to(first), first)
                 and torch.equal(_tensor(slot["exp_avg_sq"], initial.shape, torch.float32, "Malformed finite Adam second moment").to(second), second),
                 "Finite Adam moments differ from original finite gradient0")


def _attach(bundle):
    bundle = cpu_state(bundle)
    bundle["content_sha256"] = _seal(bundle)
    return bundle


def _validate_bundle(bundle, buffers, candidate, context, stop):
    _require(isinstance(bundle, dict) and set(bundle) == {"schema", "context", "step", "parameters", "optimizer", "snapshots", "scale", "history", "content_sha256"}
             and bundle.get("context") == context and type(bundle.get("schema")) is int
             and bundle["schema"] == SCHEMA and _seal({key: value for key, value in bundle.items() if key != "content_sha256"}) == bundle.get("content_sha256"),
             "Finite cache source/schema/content changed")
    step = bundle.get("step")
    _require(type(step) is int and step in (0, 1) and isinstance(bundle.get("snapshots"), dict)
             and all(type(key) is int for key in bundle["snapshots"])
             and set(bundle["snapshots"]) == set(range(step+1)), "Finite cache frontier0/1 changed")
    original = [value.detach().clone() for value in buffers["initial"]]
    for key, snapshot in bundle["snapshots"].items():
        _require(isinstance(snapshot, dict) and {"schema", "context", "step", "parameters"} <= set(snapshot)
                 and type(snapshot["schema"]) is int and snapshot["schema"] == SCHEMA and snapshot["context"] == context
                 and type(snapshot["step"]) is int and snapshot["step"] == key
                 and isinstance(snapshot["parameters"], (list, tuple)) and len(snapshot["parameters"]) == 2,
                 "Malformed finite snapshot container/step")
        for value, initial in zip(snapshot["parameters"], original, strict=True):
            _tensor(value, initial.shape, torch.float32, "Malformed finite snapshot parameters")
    _require(_seal(bundle["snapshots"][0]["parameters"]) == _seal(original), "Finite original U0/V0 changed")
    actual0, scale = _evaluate(buffers, candidate, context, original, 0, None, stop)
    _require(type(bundle.get("scale")) is float and math.isfinite(bundle["scale"]) and bundle["scale"] > 0
             and _seal(actual0) == _seal(bundle["snapshots"][0]) and bundle["scale"] == scale, "Finite coupled J0/gradient/trace cache changed")
    gradients = [value.to(initial) for value, initial in zip(actual0["scaled_factor_gradients"], original, strict=True)]
    expected = original if step == 0 else [_first_adam(initial, gradient)[0] for initial, gradient in zip(original, gradients, strict=True)]
    _require(_seal(bundle["parameters"]) == _seal(expected), "Finite native current Adam parameters changed")
    _optimizer(bundle["optimizer"], original, gradients, step)
    if step == 1:
        actual1, _ = _evaluate(buffers, candidate, context, expected, 1, scale, stop)
        _require(_seal(actual1) == _seal(bundle["snapshots"][1]), "Finite current model/moments/partial cache changed")
    expected_history = [_history(bundle["snapshots"][key]) for key in range(step+1)]
    _require(bundle.get("history") == expected_history, "Finite history differs from actual source certificates")
    return bundle


def _history(snapshot):
    return dict(step=snapshot["step"], teacher_ce=snapshot["teacher_ce"], J0_scale=snapshot["scale"],
                finite_model_steps=5, min_mass=snapshot["min_mass"], max_mass=snapshot["max_mass"],
                effective_cells=snapshot["effective_cells"], gradient_U_norm=float(snapshot["scaled_factor_gradients"][0].norm()),
                gradient_V_norm=float(snapshot["scaled_factor_gradients"][1].norm()))


def _atomic(path, value, binary):
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("xb" if binary else "x") as stream:
            if binary:
                torch.save(value, stream)
            else:
                stream.write(value)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _mirrors(folder, bundle, *, export=False):
    """Validate ALL present mirrors before any repair/write; resume is authority."""
    snapshots = bundle["snapshots"]
    directory = folder / "checkpoints"
    _require(not directory.exists() or directory.is_dir(), "Malformed finite checkpoint directory preserved")
    for path in directory.iterdir() if directory.exists() else ():
        _require(path.is_file() and path.name.startswith("step_") and path.suffix == ".pt", "Malformed finite mirror path preserved")
        saved = torch.load(path, map_location="cpu", weights_only=False)
        key = saved.get("step") if isinstance(saved, dict) else None
        _require(type(key) is int and key in snapshots and path.name == f"step_{key:06d}.pt"
                 and _seal(saved) == _seal(snapshots[key]), "Malformed/orphan finite checkpoint preserved")
    csv_path = folder / "optimization.csv"
    if csv_path.exists():
        with csv_path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        _require(len(rows) in (1, len(bundle["history"])) and all(int(row["step"]) == index for index, row in enumerate(rows)), "Malformed finite CSV preserved")
        for row, expected in zip(rows, bundle["history"], strict=False):
            _require(set(row) == set(expected) and all(float(row[key]) == value for key, value in expected.items()), "Finite CSV certificate differs")
    if export:
        directory.mkdir(exist_ok=True)
        for key, snapshot in snapshots.items():
            path = directory / f"step_{key:06d}.pt"
            if not path.exists():
                _atomic(path, snapshot, True)
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=list(bundle["history"][0]))
        writer.writeheader()
        writer.writerows(bundle["history"])
        _atomic(csv_path, stream.getvalue(), False)


def prepare_probe(dataset, ratio, output_dir, candidate, condensation_seed=0, data_dir="data", device="cuda",
                  citation_features="row", stop=lambda: False):
    """One committed P update/zero finalstudents. Only native pilot0/1 is supported."""
    candidate = canonical_candidate(candidate)  # Beforedata/device/cachemutation.
    _stop(stop)
    _public_options(dataset, ratio, output_dir, condensation_seed, data_dir, device, citation_features)
    environment = _native(device)
    # Include source/model loading and both objective unrolls in the job peak.
    torch.cuda.reset_peak_memory_stats()
    total_cuda_bytes = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
    buffers = _load_source(dataset, ratio, output_dir, candidate, condensation_seed, data_dir, device, citation_features, stop)
    reference = _p0_reference(buffers)
    context = _context(candidate, buffers, environment, reference)
    cid = _fingerprint(candidate)
    folder = buffers["root"] / cid / "condensation_0"
    candidate_path = folder.parent / "candidate.json"
    if candidate_path.exists():
        _require(json.loads(candidate_path.read_text()) == candidate, "Finite candidate identity changed")
    resume_path = folder / "resume.pt"
    bundle = torch.load(resume_path, map_location="cpu", weights_only=False) if resume_path.exists() else None
    if bundle is None:
        _require(not any(folder.glob("**/*")), "Finite orphan artifacts preserved; no silent overwrite")
    else:
        _require(candidate_path.is_file(), "Finite cached bundle requires its canonical candidate record")
        _validate_bundle(bundle, buffers, candidate, context, stop)
        _mirrors(folder, bundle)
    initial_step = -1 if bundle is None else bundle["step"]
    cached = initial_step == 1
    parameters = [value.detach().clone().requires_grad_(True) for value in buffers["initial"]]
    optimizer = torch.optim.Adam(parameters, lr=.05, eps=1e-12, foreach=False, fused=False)
    if bundle is not None:
        for target, value in zip(parameters, bundle["parameters"], strict=True):
            with torch.no_grad():
                target.copy_(value.to(target))
        optimizer.load_state_dict(bundle["optimizer"])
    if not cached:
        if bundle is None:
            snapshot, scale = _evaluate(buffers, candidate, context, parameters, 0, None, stop)
            bundle = _attach(dict(schema=SCHEMA, context=context, step=0, parameters=parameters,
                                 optimizer=optimizer.state_dict(), snapshots={0: snapshot}, scale=scale, history=[_history(snapshot)]))
            _checked_files(buffers["pins"])
            _source_unchanged(context)
            _stop(stop)
            folder.mkdir(parents=True, exist_ok=True)
            if not candidate_path.exists():
                _atomic(candidate_path, json.dumps(candidate, indent=2), False)
            _atomic(resume_path, bundle, True)
            _mirrors(folder, bundle, export=True)
        _stop(stop)
        for parameter, gradient in zip(parameters, bundle["snapshots"][0]["scaled_factor_gradients"], strict=True):
            parameter.grad = gradient.to(parameter).clone()
        optimizer.step()
        snapshot, scale = _evaluate(buffers, candidate, context, parameters, 1, bundle["scale"], stop)
        bundle = _attach(dict(schema=SCHEMA, context=context, step=1, parameters=parameters, optimizer=optimizer.state_dict(),
                     snapshots={0: bundle["snapshots"][0], 1: snapshot}, scale=scale,
                     history=[bundle["history"][0], _history(snapshot)]))
        _checked_files(buffers["pins"])
        _source_unchanged(context)
        _stop(stop)
        _mirrors(folder, bundle)  # Failclosed onpresentcorruption beforebundleoverwrite.
        _atomic(resume_path, bundle, True)
    _checked_files(buffers["pins"])
    _source_unchanged(context)
    _stop(stop)
    _mirrors(folder, bundle, export=True)
    torch.cuda.synchronize()
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    return dict(CUDA_peak_allocated_bytes=peak_allocated, CUDA_peak_reserved_bytes=peak_reserved,
                CUDA_total_bytes=total_cuda_bytes, root=str(buffers["root"].resolve()), candidate_path=str(folder.parent.resolve()), candidate_id=cid,
                outer_route=candidate["outer_route"], step=1, cached=cached, validation_only=True, student_fits=0,
                P_updates=0 if cached else 1, finite_model_steps=5, finite_source_context_digest=_seal(context),
                own_J0=bundle["scale"], teacher_ce=bundle["snapshots"][1]["teacher_ce"],
                actual_FP32_P0_X_Q_supplieduniform_equal_reference=True,
                counts=dict(committed_P_updates=0 if cached else 1, final_student_fits=0, teacher_map_Phi_hardinit_fits=0,
                            finite_inner_model_steps_per_objective=5, finite_inner_objective_unrolls_this_invocation=2,
                            finite_inner_functional_SGD_steps_this_invocation=10, linear_head_or_CG_solves=0,
                            cached_source_reference_linear_certificate_gradient_checks=1),
                caveat="Finite5SGD is not finalAdam/convergedhead; .001 active allparameterinnerWD vsreferenceλ1e-4; sourceH/S distinct; explicitouterAdamforeachFalse/fusedFalse differsfromlegacyreferencebackend; schema2strictdeterministicdenseoriginalS; oldschema1incompatible; no full25 integration orpromotion.")
