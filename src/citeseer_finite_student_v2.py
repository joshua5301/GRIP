"""Fixed Citeseer30/120 finite-GCN assignment, native1→25 and gated students.

The two certified original sources are immutable. Original AP/AQ and the pure
five-SGD engine are reused without retargeting their public Cora interfaces.
Every persisted optimizer frame and objective is replayed; final Adam students
are separate from the five-step functional model used to differentiate P.
"""
import csv
import json
import math
import platform
import time
from pathlib import Path

import torch

from src import citation_search
from src import citation_source_certificate as original
from src import finite_student_experiment as legacy
from src import finite_student_probe as probe
from src import source_linear_assignment as source_helper
from src.citation_source_preflight import _exact, _write_new
from src.evaluation import _input_digest, fit_gcn_diagnostic
from src.finite_student_outer_v2 import finite_student_outer_partials
from src.io import _fingerprint
from src.low_rank_assignment import initialize_factors
from src.research_loop import implementation_provenance
from src.student_routes import replay_routes
from src.sweep_utils import representative
from src.target_refinement import training_refined_targets

SCHEMA = 2
HORIZON = 25
SCIENCE = "Citeseer30_120_finite_GCN_Qbar_contract_fixed25_scientific_stageAV_v1.json"
SCIENCE_SHA = "3a9c3131d6d2e2f007485cb873875269b7c1c94a146bf5b368cd0a98cec74e53"
SOURCE_FIELD = "finite_student_source_digest"
_FIXED = dict(method="finite_student_citeseer_qbar_contract", width=0, lr=.01, T=1., rank=8, penalty=.001,
              initialization="teacher_joint", alpha=.3, inner_loss_weighting="uniform", mass_mode="free", mixing=.05,
              finite_student_schema=2, finite_assignment_steps=25, finite_model_steps=5, finite_model_lr=.1,
              finite_model_hidden=256, finite_model_seed=0, finite_model_weight_decay=.001,
              finite_graph_backend="dense_original_S", finite_native_policy="deterministic_cuda_v1", outer_route="gcn")
_SEEDS = (3700, 3701, 3702)
_SETTINGS = {
    30: dict(epochs=500, eval_every=1, hidden=256, dropout=.3, lr=.003, weight_decay=5e-5,
             lr_schedule="constant", initialization="geom_uniform"),
    120: dict(epochs=1000, eval_every=1, hidden=256, dropout=0., lr=.001, weight_decay=.0005,
              lr_schedule="constant", initialization="geom_uniform"),
}
_require, _stop, _tensor, _scalar = probe._require, probe._stop, probe._tensor, probe._scalar
_seal, _sha, _attach, _atomic = probe._seal, probe._sha, probe._attach, probe._atomic
_mirrors, _bundle_paths, _student_paths = legacy._mirrors, legacy._bundle_paths, legacy._student_paths


def numerical_source():
    result = probe.numerical_source()
    _require(result["files"].get("citeseer_finite_student_v2.py") == _sha(__file__),
             "Actual Citeseer finite module/full source manifest missing or changed")
    _require(result["files"].get("finite_student_outer_v2.py") == "431d0687f5d1844df417d80b885339074ce397c2fdc32d2870acac3fa87c5fa6",
             "Frozen v2 derived-Qbar engine changed")
    return result


def canonical_candidate(candidate):
    _require(isinstance(candidate, dict) and set(_FIXED) <= set(candidate)
             and set(candidate) <= set(_FIXED) | {SOURCE_FIELD},
             "Explicit fixed Citeseer candidate required; unknown/external controls forbidden")
    result = dict(_FIXED)
    for key, expected in _FIXED.items():
        actual = candidate[key]
        valid = (type(actual) is int and actual == expected) if type(expected) is int else (
            type(actual) in (int, float) and math.isfinite(actual) and actual == expected) if type(expected) is float else (
            type(actual) is str and actual == expected)
        _require(valid, f"Changed fixed Citeseer finite control: {key}")
    token = _seal(numerical_source())
    _require(candidate.get(SOURCE_FIELD, token) == token, "Source changed; preserve old finite namespace")
    result[SOURCE_FIELD] = token
    return result


def _budget(cells):
    _require(type(cells) is int and cells in _SETTINGS, "Only fixed Citeseer30/120 ROW sources are supported")
    return dict(next(row for row in original._roots(Path(__file__).resolve().parents[1]) if row["cells"] == cells))


def _phase(value):
    _require(type(value) is int and value in (1, HORIZON), "Only native frontier1 or25 is supported")
    return value


def _settings(cells):
    _budget(cells)
    return dict(_SETTINGS[cells])


def _load_spec(path, checksum, candidate):
    repo = Path(__file__).resolve().parents[1]
    path = Path(path).resolve()
    _require(type(checksum) is str and len(checksum) == 64 and path.is_file()
             and path.is_relative_to(repo / "results/proposals") and _sha(path) == checksum,
             "Require immutable prospective execution spec path/SHA")
    spec = json.loads(path.read_text())
    fields = {"schema", "scientific_preregistration", "source", "numerical_source", "python_version", "files_sha256",
              "roots", "candidate", "candidate_id", "artifacts_sha256", "output_root", "certificate_outputs"}
    _require(isinstance(spec, dict) and set(spec) == fields and type(spec["schema"]) is int and spec["schema"] == SCHEMA
             and spec["candidate"] == candidate and spec["candidate_id"] == _fingerprint(candidate)
             and spec["roots"] == original._roots(repo), "Malformed or changed fixed execution spec")
    reference = dict(path=str(repo / "results/proposals" / SCIENCE), sha256=SCIENCE_SHA)
    _require(spec["scientific_preregistration"] == reference, "Wrong frozen scientific policy")
    probe._checked_files({reference["path"]: SCIENCE_SHA})
    science = json.loads(Path(reference["path"]).read_text())
    _require(_exact(science["candidate"], _FIXED) and science["student_seeds"] == list(_SEEDS)
             and spec["files_sha256"] == science["original_files_sha256"], "Changed science or original source inventory")
    _require(isinstance(spec["artifacts_sha256"], dict) and spec["artifacts_sha256"]
             and all(Path(p).resolve().is_relative_to(repo / "results") for p in spec["artifacts_sha256"]),
             "Require frozen implementation and review artifact pins")
    probe._checked_files(spec["files_sha256"])
    probe._checked_files(spec["artifacts_sha256"])
    probe._checked_files({row["path"]: row["sha256"] for row in science["parents"]})
    output_root = Path(spec["output_root"]).resolve()
    _require(output_root.is_relative_to(repo / "results/research_loop") and str(output_root) == spec["output_root"],
             "Evidence must use frozen research output root")
    outputs = spec["certificate_outputs"]
    _require(isinstance(outputs, dict) and set(outputs) == {"30", "120"}
             and all(isinstance(rows, dict) and set(rows) == {"1", "25"} for rows in outputs.values()),
             "Require declared native1/25 certificate paths for both budgets")
    for rows in outputs.values():
        for value in rows.values():
            _require(type(value) is str and Path(value).resolve().is_relative_to(output_root)
                     and str(Path(value).resolve()) == value and Path(value).suffix == ".json",
                     "Invalid fixed native certificate output")
    _require(len({value for rows in outputs.values() for value in rows.values()}) == 4,
             "Native certificate outputs must be distinct")
    _source_unchanged(spec)
    return spec, science


def _source_unchanged(context):
    expected = context.get("spec", context)
    _require(implementation_provenance() == expected["source"] and numerical_source() == expected["numerical_source"]
             and platform.python_version() == expected["python_version"], "Source/Git/Python/versions changed")


def _stable_files(buffers):
    for path, expected in buffers["file_stats"].items():
        stat = Path(path).stat()
        _require((stat.st_size, stat.st_mtime_ns, stat.st_ino) == expected, "Immutable source file changed during trusted trajectory")


def _count(evidence, key, complete=False):
    suffix = "_completed" if complete else "_attempts"
    evidence["counts"][key + suffix] = evidence["counts"].get(key + suffix, 0) + 1


def _load_source(cells, spec, science, evidence, stop):
    row = _budget(cells)
    root = Path(row["source_root"])
    _stop(stop)
    evidence["stage"] = "original_dataset_and_source"
    _count(evidence, "source_load")
    repo = Path(__file__).resolve().parents[1]
    graph, train, val, test, fresh_h = citation_search._prepare_dataset("citeseer", str(repo / "data"), "cuda", "row")
    config = citation_search._legacy_teacher_config("citeseer", row["ratio"], graph, train, val, test, "row")
    _require(config == original._config(row) and _fingerprint(config) == row["root"], "Current graph/splits/preprocessing differ")
    teacher = torch.load(root / "teacher.pt", map_location="cuda", weights_only=False)
    _require(isinstance(teacher, dict), "Malformed original teacher")
    logits = _tensor(teacher.get("logits"), (3327, 6), torch.float64, "Malformed original teacher logits")
    q = training_refined_targets(logits, 1., graph["y"], train, 0)
    ghost = source_helper.candidate_controls(dict(original.CANDIDATE, method="source_linear",
                  assignment_coordinates="raw_rms", source_linear_schema=1))
    h, z, transform, hard, _, source = source_helper.cached_source(root, ghost, 0, fresh_h, q, config)
    _require(z.shape == (3327, 3703) and q.shape == (3327, 6) and int(hard.max())+1 == cells
             and len(torch.unique(hard)) == cells, "Fixed native source dimensions/partition differ")
    _require(torch.is_tensor(val[1]) and val[1].dtype == torch.bool and val[1].shape == (3327,)
             and int(val[1].sum()) == 500, "Require original500-node validation mask")
    _count(evidence, "model_GEOM_factory")
    model_initial = probe.geom_uniform_initial(3703, 6, hidden=256, dtype=torch.float32, device="cuda")
    _count(evidence, "model_GEOM_factory", True)
    _count(evidence, "native_factor_factory")
    u, v = initialize_factors(hard, cells, 8, 0)
    _count(evidence, "native_factor_factory", True)
    dense = probe._frozen_original_dense_S(graph["adj"])
    buffers = dict(root=root, cells=cells, graph=graph, train=train, val=val[1], x=graph["x"].detach(),
                   S=dense, original_S=graph["adj"].detach(), h=h.detach().float(), z=z.detach(), q=q.detach(), hard=hard.detach(),
                   transform=probe._frozen_transform(transform), model_initial=model_initial, initial=[u.detach(), v.detach()],
                   ghost=ghost, config=config, source=source, pins=spec["files_sha256"], counts=evidence["counts"])
    # Reuse AT's unchanged full original config,0/25 material and linear certificates.
    reference_counts = dict(native_initializer_calls=0, native_initializer_completed_calls=0,
                            native_P0_probability_material_evaluations=0, cached_head_gradient_evaluations=0,
                            cached_outer_CE_evaluations=0)
    try:
        reference = original._reference(dict(buffers, transform=transform), ghost, cells, reference_counts)
    finally:
        for key, value in reference_counts.items():
            if key != "stage":
                evidence["counts"][key] = value
    _require(reference["P0_parameters"] == probe._digest(buffers["initial"]), "Current paired native U0/V0 differ")
    certified = json.loads(Path(science["parents"][0]["path"]).read_text())["roots"][row["root"]]
    native_buffers = probe._digest(dict(H=h, z=z, Q=q, hard=hard, transform=probe._frozen_transform(transform),
                                    X=graph["x"], original_CSR=graph["adj"], dense_original_S=dense))
    _require(certified["native_buffers"] == native_buffers, "Actual native X/S/H/RMS/Q/partition differ from AT")
    _require(certified["source_context"] == source and certified["reference"] == reference,
             "Actual source/reference buffers differ from immutable AT proof")
    recipe_row = next(item for item in science["roots"] if item["cells"] == cells)["student_recipe"]
    _require(json.loads(Path(recipe_row["path"]).read_text()) == dict(_settings(cells), input_scale=1.),
             "Original student recipe or identity input scale changed")
    buffers["recipe_origin"] = recipe_row
    buffers["file_stats"] = {path: (Path(path).stat().st_size, Path(path).stat().st_mtime_ns, Path(path).stat().st_ino)
                             for path in buffers["pins"]}
    _count(evidence, "source_load", True)
    return buffers, reference


def _context(candidate, buffers, environment, reference, spec):
    result = probe._context(candidate, buffers, environment, reference)
    result.update(schema=SCHEMA, numerical_source=numerical_source(), assignment_steps=HORIZON,
                  allowed_steps=list(range(HORIZON+1)), spec=spec, cells=buffers["cells"], student_recipe_origin=buffers["recipe_origin"],
                  scientific_preregistration_sha256=SCIENCE_SHA,
                  checkpoint_policy="atomic complete prefix0..25; same source probe1 then continuation24",
                  derived_Qbar_contract=dict(policy="derived_Qbar_original_contract_v1",
                      source_Q="32*finfo(source_Q.dtype).eps", FP64_absolute_row_sum_tolerance=1e-12,
                      other_dtypes="unchanged32*finfo(dtype).eps", renormalization=False),
                  outer_optimizer=dict(name="Adam", lr=.01, betas=[.9, .999], eps=1e-12, weight_decay=0, foreach=False, fused=False))
    return result


def _evaluate(buffers, candidate, context, parameters, step, scale, stop):
    _require(type(step) is int and 0 <= step <= HORIZON, "Invalid Citeseer snapshot step")
    _count({"counts": buffers["counts"]}, "finite_objective")
    _stop(stop)
    probe._runtime_precision_guard()
    current = [parameter.detach().clone().requires_grad_(True) for parameter in parameters]
    moments = probe._moments(buffers, current)
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
    logits = probe.LowRankLogits.apply(current[0], current[1], buffers["hard"], .05, 4096)
    probability = logits.double().softmax(1)
    _, labels, mass = probe.decode_moments(moments, buffers["z"].shape[1])
    conservation = dict(row=float((probability.sum(1)-1).abs().max()), mass_sum=float((mass.sum()-1).abs()),
        material=float((moments.sum(0)-probe.make_material(buffers["z"], buffers["q"]).mean(0)).abs().max()),
        Qc_simplex=float((labels.sum(1)-1).abs().max()))
    _require(bool((mass > 0).all()) and all(math.isfinite(value) and value <= 1e-12 for value in conservation.values()),
             "Native source P/material conservation failed")
    snapshot = dict(schema=SCHEMA, context=context, step=step, parameters=current, moments=moments,
        teacher_ce=loss, scale=scale, finite_model_steps=5, selected_route=route,
        moment_partial=result["moment_gradients"][route], scaled_factor_gradients=gradients,
        adapted_model=result["adapted_parameters"], inner_parameter_states=result["inner_parameter_states"], inner_trace=result["inner_trace"],
        paired_outer_losses={name: float(value) for name, value in result["outer_losses"].items()}, conservation=conservation,
        encoded_U=probe._digest(current[0]), raw_logits=probe._digest(logits), min_mass=float(mass.min()), max_mass=float(mass.max()),
        effective_cells=float(1/mass.square().sum()))
    _count({"counts": buffers["counts"]}, "finite_objective", True)
    return probe.cpu_state(snapshot), scale


def _adam_step(parameters, first, second, gradients, step):
    """Read-only recurrence matching fixed native noncapturable Adam ordering."""
    _require(type(step) is int and 1 <= step <= HORIZON and all(isinstance(x, (list, tuple)) and len(x) == 2
             for x in (parameters, first, second, gradients)), "Malformed Citeseer finite Adam recurrence")
    updated, next_first, next_second = [], [], []
    for parameter, old_first, old_second, gradient in zip(parameters, first, second, gradients, strict=True):
        _require(torch.is_tensor(parameter), "Malformed native Adam parameter")
        _tensor(parameter, parameter.shape, torch.float32, "Malformed native Adam parameter")
        for value in (old_first, old_second, gradient):
            _tensor(value, parameter.shape, torch.float32, "Malformed native Adam moment/gradient")
            _require(value.device == parameter.device, "Native Adam recurrence device changed")
        _require(bool((old_second >= 0).all()), "Negative native Adam second moment")
        m = old_first.detach().clone().lerp_(gradient, 1-.9)
        v = old_second.detach().clone().mul_(.999).addcmul_(gradient, gradient, value=1-.999)
        denominator = (v.sqrt() / ((1-.999**step)**.5)).add_(1e-12)
        p = parameter.detach().clone().addcdiv_(m, denominator, value=-.01/(1-.9**step))
        _require(all(bool(torch.isfinite(value).all()) for value in (p, m, v)), "Nonfinite native Adam recurrence")
        updated.append(p)
        next_first.append(m)
        next_second.append(v)
    return updated, next_first, next_second


def _optimizer(saved, parameters, first, second, step):
    _require(type(step) is int and 0 <= step <= HORIZON and isinstance(saved, dict)
             and set(saved) == {"state", "param_groups"} and isinstance(saved["param_groups"], list)
             and len(saved["param_groups"]) == 1, "Malformed Citeseer finite Adam")
    group = saved["param_groups"][0]
    expected = dict(lr=.01, betas=[.9, .999], eps=1e-12, weight_decay=0, amsgrad=False, maximize=False,
                    foreach=False, capturable=False, differentiable=False, fused=False)
    _require(isinstance(group, dict) and set(group) in (set(expected)|{"params"},
             set(expected)|{"params", "decoupled_weight_decay"}) and group.get("params") == [0, 1]
             and all(type(x) is int for x in group["params"]), "Citeseer finite Adam controls/IDs changed")
    for key, value in expected.items():
        actual = group[key]
        if key == "betas":
            valid = isinstance(actual, (list, tuple)) and len(actual) == 2 and all(type(x) in (int, float) for x in actual) and list(actual) == value
        elif type(value) is bool:
            valid = actual is value
        else:
            valid = type(actual) in (int, float) and math.isfinite(actual) and actual == value
        _require(valid, "Citeseer finite Adam fixed controls changed")
    if "decoupled_weight_decay" in group:
        _require(group["decoupled_weight_decay"] is False, "Citeseer finite Adam decay changed")
    slots = saved["state"]
    _require(isinstance(slots, dict) and all(type(key) is int for key in slots)
             and set(slots) == (set() if step == 0 else {0, 1}), "Citeseer finite Adam slots/counters changed")
    if step == 0:
        return
    for index, parameter in enumerate(parameters):
        slot = slots[index]
        _require(isinstance(slot, dict) and set(slot) == {"step", "exp_avg", "exp_avg_sq"}, "Malformed Citeseer finite Adam slot")
        counter = _tensor(slot["step"], (), torch.float32, "Malformed scalar FP32 Adam counter")
        _require(counter.device.type == "cpu" and float(counter) == step, "Citeseer finite Adam counter/device changed")
        for key, expected_value in (("exp_avg", first[index]), ("exp_avg_sq", second[index])):
            actual = _tensor(slot[key], parameter.shape, torch.float32, "Malformed Citeseer finite Adam moments")
            _require(torch.equal(actual.to(expected_value), expected_value), "Citeseer finite Adam moments differ from exact native recurrence")


def _validate_bundle(bundle, buffers, candidate, context, stop):
    _require(isinstance(bundle, dict) and set(bundle) == {"schema", "context", "step", "parameters", "optimizer", "snapshots", "scale", "history", "content_sha256"}
             and type(bundle.get("schema")) is int and bundle["schema"] == SCHEMA and bundle.get("context") == context
             and _seal({key: value for key, value in bundle.items() if key != "content_sha256"}) == bundle.get("content_sha256"),
             "Citeseer finite source/schema/content changed")
    frontier = bundle.get("step")
    _require(type(frontier) is int and 0 <= frontier <= HORIZON and isinstance(bundle.get("snapshots"), dict)
             and all(type(key) is int for key in bundle["snapshots"]) and set(bundle["snapshots"]) == set(range(frontier+1)),
             "Malformed Citeseer finite complete-prefix frontier")
    initial = [value.detach().clone() for value in buffers["initial"]]
    for step, snapshot in bundle["snapshots"].items():
        _require(isinstance(snapshot, dict) and {"schema", "context", "step", "parameters"} <= set(snapshot)
                 and type(snapshot["schema"]) is int and snapshot["schema"] == SCHEMA and snapshot["context"] == context
                 and type(snapshot["step"]) is int and snapshot["step"] == step
                 and isinstance(snapshot["parameters"], (list, tuple)) and len(snapshot["parameters"]) == 2
                 and isinstance(snapshot.get("optimizer"), dict),
                 "Malformed Citeseer finite snapshot container")
        for value, origin_parameter in zip(snapshot["parameters"], initial, strict=True):
            _tensor(value, origin_parameter.shape, torch.float32, "Malformed Citeseer finite snapshot parameters")
    _require(_seal(bundle["snapshots"][0]["parameters"]) == _seal(initial), "Citeseer finite original native U0/V0 changed")
    actual0, scale = _evaluate(buffers, candidate, context, initial, 0, None, stop)
    _require(type(bundle.get("scale")) is float and math.isfinite(bundle["scale"]) and bundle["scale"] > 0
             and bundle["scale"] == scale
             and _seal(actual0) == _seal({key: value for key, value in bundle["snapshots"][0].items() if key != "optimizer"}),
             "Citeseer finite coupled J0/model/partial certificate changed")
    parameters = initial
    first, second = [torch.zeros_like(x) for x in initial], [torch.zeros_like(x) for x in initial]
    _optimizer(bundle["snapshots"][0]["optimizer"], parameters, first, second, 0)
    previous = actual0
    for step in range(1, frontier+1):
        _stop(stop)
        gradients = [value.to(parameter) for value, parameter in zip(previous["scaled_factor_gradients"], parameters, strict=True)]
        parameters, first, second = _adam_step(parameters, first, second, gradients, step)
        _require(_seal(bundle["snapshots"][step]["parameters"]) == _seal(parameters), "Citeseer finite checkpoint Adam transition changed")
        actual, _ = _evaluate(buffers, candidate, context, parameters, step, scale, stop)
        _optimizer(bundle["snapshots"][step]["optimizer"], parameters, first, second, step)
        _require(_seal(actual) == _seal({key: value for key, value in bundle["snapshots"][step].items() if key != "optimizer"}),
                 "Citeseer finite native model/moment/partial/trace replay changed")
        previous = actual
    _require(_seal(bundle.get("parameters")) == _seal(parameters), "Citeseer finite current native parameters changed")
    _optimizer(bundle["optimizer"], parameters, first, second, frontier)
    _require(_seal(bundle["optimizer"]) == _seal(bundle["snapshots"][frontier]["optimizer"]),
             "Citeseer finite latest optimizer differs from actual terminal snapshot optimizer")
    expected_history = [_history(bundle["snapshots"][step]) for step in range(frontier+1)]
    _require(_seal(bundle.get("history")) == _seal(expected_history), "Citeseer finite history differs from actual source certificates")
    return frontier+1


def _history(snapshot):
    row = probe._history(snapshot)
    slots = snapshot["optimizer"]["state"]
    row.update(Adam_U_step=0. if snapshot["step"] == 0 else float(slots[0]["step"]),
               Adam_V_step=0. if snapshot["step"] == 0 else float(slots[1]["step"]))
    return row


def _load_bundle(folder, candidate_path, resume_path, candidate, buffers, context, stop, *, required=False):
    if candidate_path.exists():
        _require(json.loads(candidate_path.read_text()) == candidate, "Citeseer finite candidate identity changed")
    if not resume_path.exists():
        _require(not required and not any(folder.glob("**/*")), "Citeseer finite missing/orphan resume preserved; no fallback")
        return None, 0
    _require(candidate_path.is_file(), "Citeseer finite cached resume requires canonical candidate")
    bundle = torch.load(resume_path, map_location="cpu", weights_only=False)
    unrolls = _validate_bundle(bundle, buffers, candidate, context, stop)
    _mirrors(folder, bundle)
    return bundle, unrolls


def _commit(folder, candidate_path, resume_path, candidate, buffers, context, bundle, stop):
    _stable_files(buffers)
    _source_unchanged(context)
    _stop(stop)
    _mirrors(folder, bundle)  # Validate present artifacts before replacing authority.
    folder.mkdir(parents=True, exist_ok=True)
    if not candidate_path.exists():
        _atomic(candidate_path, json.dumps(candidate, indent=2), False)
    _atomic(resume_path, bundle, True)
    _mirrors(folder, bundle, export=True)


def _gate(path, checksum, cells, candidate, spec, frontier, *, continuation=False):
    _phase(frontier)
    expected_path = spec["certificate_outputs"][str(cells)][str(frontier)]
    _require(type(checksum) is str and len(checksum) == 64 and str(Path(path).resolve()) == expected_path
             and Path(path).is_file() and _sha(path) == checksum, "Require declared immutable native gate")
    gate = json.loads(Path(path).read_text())
    _require(isinstance(gate, dict) and gate.get("passed") is True and gate.get("test_enabled") is False
             and type(gate.get("cells")) is int and gate["cells"] == cells
             and type(gate.get("finite_student_schema")) is int and gate["finite_student_schema"] == SCHEMA
             and type(gate.get("frontier")) is int and gate["frontier"] == frontier
             and gate.get("source") == spec["source"] and gate.get("candidate") == candidate
             and gate.get("candidate_id") == _fingerprint(candidate)
             and gate.get("scientific_preregistration") == spec["scientific_preregistration"]
             and gate.get("full_prefix_cache_replay_passed") is True
             and gate.get("actual_FP32_P0_X_Q_uniform_equal_reference") is True,
             "Unaccepted or incompatible native prefix gate")
    _require(isinstance(gate.get("files_sha256"), dict) and set(spec["files_sha256"]) <= set(gate["files_sha256"]),
             "Native gate lacks original numerical source pins")
    pins = gate["files_sha256"]
    if continuation:
        # Resume/CSV advance. Immutable0/1 mirrors and the sealed prefix below
        # bind the old proof; no historical bytes are silently rebound.
        pins = {p: digest for p, digest in pins.items() if p in spec["files_sha256"]
                or Path(p).name in ("candidate.json", "step_000000.pt", "step_000001.pt")}
    probe._checked_files(pins)
    return gate


def _prefix(bundle, frontier):
    snapshot = bundle["snapshots"][frontier]
    return _attach(dict(schema=SCHEMA, context=bundle["context"], step=frontier,
                        parameters=snapshot["parameters"], optimizer=snapshot["optimizer"],
                        snapshots={step: bundle["snapshots"][step] for step in range(frontier+1)},
                        scale=bundle["scale"], history=bundle["history"][:frontier+1]))


def _cache_files(folder, frontier):
    _require(type(frontier) is int and 0 <= frontier <= HORIZON, "Invalid exact mirror frontier")
    expected = [folder / f"checkpoints/step_{step:06d}.pt" for step in range(frontier+1)]
    _require(sorted((folder / "checkpoints").glob("step_*.pt")) == expected, "Certificate requires all canonical prefix mirrors; no repair")
    paths = [folder.parent / "candidate.json", folder / "resume.pt", folder / "optimization.csv"]
    paths += expected
    _require(all(path.is_file() for path in paths), "Incomplete numerical frontier files")
    return {str(path.resolve()): _sha(path) for path in paths}


def _prepare_target(candidate, target, buffers, context, evidence, stop, gate=None):
    folder, candidate_path, resume_path = _bundle_paths(buffers, candidate)
    bundle, replayed = _load_bundle(folder, candidate_path, resume_path, candidate, buffers, context, stop)
    initial_step = 0 if bundle is None else bundle["step"]
    _require(initial_step <= target, "Requested frontier precedes existing native prefix")
    if target == HORIZON:
        _require(bundle is not None and bundle["step"] >= 1 and gate is not None
                 and _seal(_prefix(bundle, 1)) == gate["prefix_seal"],
                 "Continuation requires exact accepted own1 prefix; no fresh25 fallback")
    parameters = [value.detach().clone().requires_grad_(True) for value in buffers["initial"]]
    optimizer = torch.optim.Adam(parameters, lr=.01, eps=1e-12, foreach=False, fused=False)
    if bundle is not None:
        for current, saved in zip(parameters, bundle["parameters"], strict=True):
            with torch.no_grad():
                current.copy_(saved.to(current))
        optimizer.load_state_dict(bundle["optimizer"])
    scientific_unrolls = 0
    if bundle is None:
        snapshot, scale = _evaluate(buffers, candidate, context, parameters, 0, None, stop)
        scientific_unrolls += 1
        snapshot["optimizer"] = probe.cpu_state(optimizer.state_dict())
        bundle = _attach(dict(schema=SCHEMA, context=context, step=0, parameters=parameters, optimizer=optimizer.state_dict(),
                             snapshots={0: snapshot}, scale=scale, history=[_history(snapshot)]))
        _commit(folder, candidate_path, resume_path, candidate, buffers, context, bundle, stop)
    for step in range(bundle["step"]+1, target+1):
        _stop(stop)
        for parameter, gradient in zip(parameters, bundle["snapshots"][step-1]["scaled_factor_gradients"], strict=True):
            parameter.grad = gradient.to(parameter).clone()
        _count(evidence, "P_update")
        optimizer.step()
        _count(evidence, "P_update", True)
        snapshot, scale = _evaluate(buffers, candidate, context, parameters, step, bundle["scale"], stop)
        scientific_unrolls += 1
        snapshot["optimizer"] = probe.cpu_state(optimizer.state_dict())
        bundle = _attach(dict(schema=SCHEMA, context=context, step=step, parameters=parameters, optimizer=optimizer.state_dict(),
                             snapshots={**bundle["snapshots"], step: snapshot}, scale=scale,
                             history=[*bundle["history"], _history(snapshot)]))
        _commit(folder, candidate_path, resume_path, candidate, buffers, context, bundle, stop)
    _mirrors(folder, bundle, export=True)
    evidence["counts"].update(scientific_objective_unrolls=scientific_unrolls, diagnostic_cache_replay_unrolls=replayed,
                              functional_inner_SGD_steps=5*(scientific_unrolls+replayed), committed_P_updates=target-initial_step)
    return dict(candidate_path=str(folder.parent.resolve()), step=target, cached=initial_step == target,
                own_J0=bundle["scale"], teacher_ce=bundle["snapshots"][target]["teacher_ce"], files_sha256=_cache_files(folder, target))


def _manual_stock(snapshot, buffers, evidence, stop):
    """Five diagnostic stock SGD steps; never a final research student fit."""
    from src.evaluation import _forward, _initialize_geom_uniform
    from src.models import GCN

    _count(evidence, "manual_stock_unroll")
    cx, cy, _ = representative(snapshot["moments"], probe._transform(buffers), 3703, "cuda")
    with torch.random.fork_rng(devices=[]):
        model = GCN(3703, 256, 6, 2, dropout=0.)
        _initialize_geom_uniform(model, 0)
    model = model.to(buffers["x"])
    parameters = [model.layers[0].lin.weight, model.layers[0].bias, model.layers[1].lin.weight, model.layers[1].bias]
    _require(all(torch.equal(a, b) for a, b in zip(parameters, buffers["model_initial"], strict=True)),
             "Diagnostic stock GEOM initial state differs")
    errors = []
    for step in range(5):
        _stop(stop)
        loss = -(cy * _forward(model, cx)).sum(1).mean()
        gradients = torch.autograd.grad(loss, parameters)
        _count(evidence, "manual_stock_SGD_step")
        with torch.no_grad():
            for parameter, gradient in zip(parameters, gradients, strict=True):
                parameter.copy_(parameter - .1 * (gradient + .001 * parameter))
        _count(evidence, "manual_stock_SGD_step", True)
        saved = snapshot["inner_parameter_states"][step+1]
        _require(all(torch.allclose(a, b.to(a), atol=2e-6, rtol=2e-5) for a, b in zip(parameters, saved, strict=True)),
                 "Diagnostic stock5SGD differs from frozen functional model")
        errors.append(max(float((a-b.to(a)).abs().max()) for a, b in zip(parameters, saved, strict=True)))
    values = {}
    with torch.no_grad():
        for route, inputs, adjacency in (("gcn", buffers["x"], buffers["original_S"]), ("sgc_mlp", buffers["h"], None)):
            _stop(stop)
            actual = float(-(buffers["q"] * _forward(model, inputs, adjacency)).sum(1).mean())
            saved = snapshot["paired_outer_losses"][route]
            _require(math.isfinite(actual) and abs(actual-saved) <= 2e-6+2e-5*abs(saved),
                     "Original CSR/H diagnostic outer CE differs")
            values[route] = dict(actual=actual, saved=saved)
    _count(evidence, "manual_stock_unroll", True)
    return dict(all_parameter_max_abs_by_step=errors, source_teacher_CE=values, tolerances=dict(atol=2e-6, rtol=2e-5))


def _certify_prefix(candidate, frontier, buffers, context, evidence, stop):
    folder, candidate_path, resume_path = _bundle_paths(buffers, candidate)
    before = _cache_files(folder, frontier)
    bundle, replayed = _load_bundle(folder, candidate_path, resume_path, candidate, buffers, context, stop, required=True)
    _require(bundle["step"] == frontier, "Certificate requires exact requested frontier")
    # Endpoint1 is retained even at25. All native states replay exactly; only
    # the independent stock diagnostic uses the already preregistered bounds.
    parity = {str(step): _manual_stock(bundle["snapshots"][step], buffers, evidence, stop)
              for step in sorted({0, 1, frontier})}
    _require(_cache_files(folder, frontier) == before, "Readonly native certificate changed numerical cache")
    evidence["counts"].update(diagnostic_cache_replay_unrolls=replayed, functional_inner_SGD_steps=5*replayed)
    return dict(frontier=frontier, finite_student_schema=SCHEMA, finite_assignment_steps=HORIZON,
                full_prefix_cache_replay_passed=True, full25_cache_replay_passed=frontier == HORIZON,
                actual_FP32_P0_X_Q_uniform_equal_reference=True, test_enabled=False,
                source_context_digest=_seal(context), prefix_seal=_seal(bundle), manual_stock_parity=parity,
                files_sha256={**buffers["pins"], **before}, candidate_path=str(folder.parent.resolve()),
                own_J0=bundle["scale"], shared_P0_input_digest=context["reference"]["input_digest"])


def _selected_route_counts(selected, buffers, stop):
    """Two diagnostic forwards only, no training or independent epoch selection."""
    import torch.nn.functional as F

    from src.evaluation import _forward
    from src.models import GCN
    state = selected["model_state"]
    expected = {"layers.0.lin.weight": (256, 3703), "layers.0.bias": (256,),
                "layers.1.lin.weight": (6, 256), "layers.1.bias": (6,)}
    _require(isinstance(state, dict) and set(state) == set(expected), "Malformed selected student state")
    for key, shape in expected.items():
        _tensor(state[key], shape, torch.float32, "Malformed selected student weights")
    with torch.random.fork_rng(devices=[]):
        model = GCN(3703, 256, 6, 2, 0.)
    model = model.to(buffers["x"])
    model.load_state_dict(state)
    model.eval()
    counts, values = {}, {}
    mask = buffers["val"]
    target = buffers["graph"]["y"][mask]
    with torch.no_grad():
        for route, features, adjacency in (("gcn", buffers["graph"]["x"], buffers["graph"]["adj"]),
                                          ("mlp", buffers["h"], None)):
            _stop(stop)
            probability = _forward(model, features, adjacency)[mask]
            correct = probability.argmax(1) == target
            counts[route] = int(correct.sum())
            values[f"{route}_val_acc"] = 100*float(correct.double().mean())
            values[f"{route}_val_ce"] = float(F.nll_loss(probability, target))
    return counts, values


def _student_header(context, gate_sha, input_digest, seed):
    return dict(schema=1, source=context["implementation"], numerical_source=context["numerical_source"],
                native_gate_sha256=gate_sha, input_digest=input_digest, seed=seed,
                settings=_settings(context["cells"]), cells=context["cells"], recipe_origin=context["student_recipe_origin"],
                uniform_supplied_weights=True, synthetic_adjacency=None,
                shared_P0_origin_reference_id=original.REFERENCE, physical_source_root=context["ghost_source"]["root"],
                selection="first strict maximum GCN validation accuracy", test_enabled=False)


def _student_certificate(paths, header, buffers, stop):
    settings = _settings(header["cells"])
    _require(all(paths[key].is_file() for key in ("metadata", "epochs", "selected", "routes")),
             "Partial final student cache preserved; no retraining fallback")
    metadata = json.loads(paths["metadata"].read_text())
    routes = json.loads(paths["routes"].read_text())
    selected = torch.load(paths["selected"], map_location="cpu", weights_only=False)
    _require(isinstance(metadata, dict) and isinstance(metadata.get("recipe"), dict)
             and isinstance(metadata.get("result"), dict) and isinstance(selected, dict)
             and isinstance(routes, dict) and isinstance(routes.get("recipe"), dict)
             and isinstance(routes.get("result"), dict), "Malformed final student cache")
    recipe = metadata["recipe"]
    expected_recipe = dict(version=2, seed=header["seed"], settings=settings, layers=2,
                           optimizer="Adam; constant lr", weighting="normalized supplied mass",
                           selection="first maximum validation accuracy", test_enabled=False,
                           test_evaluation="selected weights once", torch_version=torch.__version__,
                           input_digest=header["input_digest"])
    _require(_seal(recipe) == _seal(expected_recipe) and metadata.get("fingerprint") == _fingerprint(expected_recipe)
             and selected.get("fingerprint") == metadata["fingerprint"],
             "Final student full recipe/input/selected fingerprint changed")
    with paths["epochs"].open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    _require(len(rows) == settings["epochs"], "Incomplete final student epoch history")
    best, winner = -math.inf, None
    for epoch, row in enumerate(rows, 1):
        try:
            valid = int(row["epoch"]) == epoch and str(epoch) == row["epoch"] and all(math.isfinite(float(value)) for value in row.values())
            accuracy = float(row["val_acc"])
            valid = valid and 0 <= accuracy <= 100
        except (KeyError, TypeError, ValueError):
            valid = False
        _require(valid, "Malformed final student epoch scalar/history")
        if accuracy > best:
            best, winner = accuracy, epoch
    _require(type(selected.get("epoch")) is int and selected["epoch"] == winner
             and metadata["result"].get("epoch") == winner and metadata["result"].get("val_acc") == best,
             "Selected student is not first strict maximum validation epoch")
    route_recipe = routes["recipe"]
    _require(routes.get("fingerprint") == _fingerprint(route_recipe) and route_recipe.get("source_fingerprint") == metadata["fingerprint"]
             and route_recipe.get("epoch") == winner and route_recipe.get("seed") == header["seed"]
             and route_recipe.get("settings") == settings and route_recipe.get("test_enabled") is False,
             "Selected sameweights route cache provenance changed")
    # This call checks actual graph/H/state/mask content digests even for a hot cache.
    replay_routes(paths["selected"], buffers["graph"], buffers["h"], {"val": buffers["val"]}, settings,
                  paths["routes"], seed=header["seed"], stop=stop)
    counts, actual = _selected_route_counts(selected, buffers, stop)
    _require(actual["gcn_val_acc"] == best == metadata["result"]["val_acc"],
             "Actual selected-state GCN accuracy differs from recorded first maximum")
    for route in ("gcn", "mlp"):
        _require(routes["result"].get(f"{route}_val_acc") == actual[f"{route}_val_acc"],
                 "Native selected-state validation correct count/scalar changed")
        _require(type(routes["result"].get(f"{route}_val_ce")) in (int, float)
                 and math.isclose(routes["result"][f"{route}_val_ce"], actual[f"{route}_val_ce"], abs_tol=2e-6, rel_tol=2e-5),
                 "Native selected-state route CE changed")
    return dict(schema=1, header=header, files_sha256={key: _sha(paths[key]) for key in ("metadata", "epochs", "selected", "routes")},
                selected_epoch=winner, route_correct_counts=counts, validation_nodes=500)


def _evaluate_student(folder, inputs, buffers, context, gate_sha, seed, stop):
    settings = _settings(context["cells"])
    cx, cy, weights = inputs
    digest = _input_digest(cx, cy, weights, buffers["graph"], buffers["q"],
                           {"train": buffers["train"], "val": buffers["val"]}, None, stop)
    header = _student_header(context, gate_sha, digest, seed)
    paths = _student_paths(folder, seed)
    prefix_files = list(folder.glob(f"seed_{seed}*"))
    cached = paths["certificate"].is_file()
    if prefix_files:
        _require(cached, "Partial/score-only final student cache preserved; no fallback")
        saved = json.loads(paths["certificate"].read_text())
        _require(isinstance(saved, dict) and type(saved.get("schema")) is int and saved["schema"] == 1
                 and saved.get("header") == header and isinstance(saved.get("files_sha256"), dict),
                 "Final student completion/source header changed")
        _require(set(prefix_files) == set(paths.values()) and all(paths[key].is_file() and _sha(paths[key]) == saved["files_sha256"].get(key)
                 for key in ("metadata", "epochs", "selected", "routes")), "Final student numerical files changed")
        _require(_student_certificate(paths, header, buffers, stop) == saved, "Final student completion certificate changed")
    else:
        _stop(stop)
        _count({"counts": buffers["counts"]}, "final_student_fit")
        fit_gcn_diagnostic(cx, cy, weights, buffers["graph"], buffers["q"],
                           {"train": buffers["train"], "val": buffers["val"]}, seed=seed,
                           folder=folder, training_adjacency=None, stop=stop, **settings)
        _count({"counts": buffers["counts"]}, "final_student_fit", True)
        replay_routes(paths["selected"], buffers["graph"], buffers["h"], {"val": buffers["val"]}, settings,
                      paths["routes"], seed=seed, stop=stop)
        saved = _student_certificate(paths, header, buffers, stop)
        _stop(stop)
        _source_unchanged(context)
        _atomic(paths["certificate"], json.dumps(saved, indent=2), False)
    result = json.loads(paths["routes"].read_text())["result"]
    return dict(seed=seed, selected_epoch=saved["selected_epoch"], input_digest=digest,
                gcn_correct=saved["route_correct_counts"]["gcn"], mlp_correct=saved["route_correct_counts"]["mlp"],
                result=result, cached=cached, actual_student_fits=0 if cached else 1,
                physical_route_outputs=0 if cached else 2, diagnostic_route_forwards=2,
                student_completion_path=str(paths["certificate"].resolve()), student_completion_sha256=_sha(paths["certificate"]))


def _validate_students(candidate, arm, buffers, context, spec, gate_sha, evidence, stop):
    _require(type(arm) is str and arm in ("node_reference", "gcn"), "Only fixed NODE/GCN validation arms")
    gate = _gate(spec["certificate_outputs"][str(buffers["cells"])]["25"], gate_sha,
                 buffers["cells"], candidate, spec, HORIZON)
    # Both logical arms require the accepted current finite prefix. Never use
    # NODE selection as a bypass for an incomplete or changed finite source.
    folder, candidate_path, resume_path = _bundle_paths(buffers, candidate)
    bundle, replayed = _load_bundle(folder, candidate_path, resume_path, candidate, buffers, context, stop, required=True)
    _require(bundle["step"] == HORIZON and _seal(bundle) == gate["prefix_seal"], "Final students require accepted exact25 prefix")
    reference_dir = buffers["root"] / original.REFERENCE / "condensation_0/checkpoints"
    reference = {step: torch.load(reference_dir / f"step_{step:06d}.pt", map_location="cpu", weights_only=False)
                 for step in (0, HORIZON)}
    snapshots = reference if arm == "node_reference" else bundle["snapshots"]
    cid = original.REFERENCE if arm == "node_reference" else _fingerprint(candidate)
    rows = []
    for step in (0, HORIZON):
        _stop(stop)
        cx, cy, mass = representative(snapshots[step]["moments"], probe._transform(buffers), 3703, "cuda")
        weights = torch.full_like(mass, 1/buffers["cells"])
        if step == 0:
            original_x, original_q, original_mass = representative(reference[0]["moments"], probe._transform(buffers), 3703, "cuda")
            _require(all(torch.equal(a, b) for a, b in zip((cx, cy, weights),
                         (original_x, original_q, torch.full_like(original_mass, 1/buffers["cells"])), strict=True)),
                     "Shared actual FP32 X/Q and supplied FP64 uniformweights equality failed")
            directory = buffers["root"] / "citeseer_finite_schema2_validation" / candidate[SOURCE_FIELD] / "shared_P0"
        else:
            directory = buffers["root"] / cid / "condensation_0/finite_validation" / f"step25_{_fingerprint(_settings(buffers['cells']))}"
        _stable_files(buffers)
        _source_unchanged(context)
        for seed in _SEEDS:
            row = _evaluate_student(directory, (cx, cy, weights), buffers, context, gate_sha, seed, stop)
            rows.append(dict(arm=arm, candidate_id=cid, step=step,
                             physical_condition=str(directory.resolve())+f":{seed}",
                             shared_P0=step == 0, **row))
    evidence["counts"].update(diagnostic_cache_replay_unrolls=replayed, functional_inner_SGD_steps=5*replayed,
                              logical_student_conditions=6, logical_serving_conditions=12,
                              physical_final_GCN_fits=sum(row["actual_student_fits"] for row in rows),
                              physical_sameweights_serving_outputs=sum(row["physical_route_outputs"] for row in rows),
                              selected_state_diagnostic_route_forwards=sum(row["diagnostic_route_forwards"] for row in rows),
                              final_optimizer_epochs=_settings(buffers["cells"])["epochs"]*sum(row["actual_student_fits"] for row in rows))
    return dict(arm=arm, rows=rows, test_enabled=False, steps=[0, HORIZON], student_seeds=list(_SEEDS),
                caveat="One physical sharedP0 cohort per budget; finite5SGD states never initialize final Adam students. Same GCN-selected state/epoch serves originalCSR and originalcachedH.")


def _run(operation, cells, candidate, spec_path, spec_sha256, output_path, phase, gate_sha256, stop):
    _budget(cells)
    candidate = canonical_candidate(candidate)
    _require(operation in ("prepare", "certify", "validate"), "Unknown fixed finite operation")
    if operation in ("prepare", "certify"):
        _phase(phase)
        _require((phase == 1 and gate_sha256 is None) or (phase == HORIZON and type(gate_sha256) is str),
                 "Frontier1 forbids gate override; frontier25 requires frozen own1 gate SHA")
    else:
        _require(type(phase) is str and phase in ("node_reference", "gcn") and type(gate_sha256) is str,
                 "Validation requires accepted25 gate and fixed arm")
    spec, science = _load_spec(spec_path, spec_sha256, candidate)
    output = Path(output_path).resolve()
    _require(output.is_relative_to(Path(spec["output_root"])) and output.suffix == ".json" and not output.exists(),
             "Require absent frozen output path")
    if operation == "certify":
        _require(str(output) == spec["certificate_outputs"][str(cells)][str(phase)], "Certificate output differs from frozen spec")
    gate = None
    if gate_sha256 is not None:
        frontier = HORIZON if operation == "validate" else 1
        gate = _gate(spec["certificate_outputs"][str(cells)][str(frontier)], gate_sha256, cells,
                     candidate, spec, frontier, continuation=frontier == 1)
    started = time.monotonic()
    bounded_stop = lambda: stop() or time.monotonic()-started >= 300
    evidence = dict(passed=False, operation=operation, cells=cells, candidate=candidate, candidate_id=_fingerprint(candidate),
                    source=spec["source"], numerical_source=spec["numerical_source"], python_version=spec["python_version"],
                    spec_path=str(Path(spec_path).resolve()), spec_sha256=spec_sha256,
                    scientific_preregistration=spec["scientific_preregistration"], finite_student_schema=SCHEMA,
                    finite_assignment_steps=HORIZON, test_enabled=False, gate_sha256=gate_sha256,
                    files_sha256=spec["files_sha256"], counts=dict(P_update_attempts=0, P_update_completed=0,
                    final_student_fit_attempts=0, final_student_fit_completed=0, teacher_map_Phi_hardinit_fits=0,
                    linear_head_or_CG_solves=0, extra_MLP_fits=0), stage="fresh_native_policy")
    native, primary = False, None
    try:
        _stop(bounded_stop)
        _require(torch.get_num_threads() == 4 and not torch.cuda.is_initialized(), "Require threads4/fresh CUDA worker")
        environment = dict(**probe._native("cuda"), python_version=platform.python_version(), threads=4)
        native = True
        torch.cuda.reset_peak_memory_stats()
        _require(torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory == original.CUDA_CAPACITY,
                 "Frozen physical GPU capacity changed")
        evidence["native_environment"] = environment
        buffers, reference = _load_source(cells, spec, science, evidence, bounded_stop)
        context = _context(candidate, buffers, environment, reference, spec)
        evidence["stage"] = operation
        if operation == "prepare":
            result = _prepare_target(candidate, phase, buffers, context, evidence, bounded_stop, gate)
        elif operation == "certify":
            result = _certify_prefix(candidate, phase, buffers, context, evidence, bounded_stop)
            if phase == HORIZON:
                bundle = torch.load(_bundle_paths(buffers, candidate)[2], map_location="cpu", weights_only=False)
                _require(_seal(_prefix(bundle, 1)) == gate["prefix_seal"], "Full25 proof differs from accepted own1 origin")
                result["accepted_frontier1_sha256"] = gate_sha256
        else:
            result = _validate_students(candidate, phase, buffers, context, spec, gate_sha256, evidence, bounded_stop)
        evidence.update(result)
        _stop(bounded_stop)
        evidence["passed"] = True
    except BaseException as error:
        primary = error
        evidence.update(error_type=type(error).__name__, error=str(error), failed_stage=evidence["stage"])
    finally:
        try:
            probe._checked_files(spec["files_sha256"])
            probe._checked_files(spec["artifacts_sha256"])
            probe._checked_files({spec["scientific_preregistration"]["path"]: SCIENCE_SHA})
            _source_unchanged(spec)
            evidence["source_unchanged"] = True
        except BaseException as error:
            evidence.update(passed=False, source_unchanged=False, preservation_error=repr(error))
            if primary is None:
                primary = error
        if native:
            try:
                torch.cuda.synchronize()
                capacity = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
                allocated, reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
                _require(capacity == original.CUDA_CAPACITY and allocated <= capacity and reserved <= capacity,
                         "Native memory/capacity certificate failed")
                evidence.update(CUDA_peak_allocated_bytes=allocated, CUDA_peak_reserved_bytes=reserved, CUDA_total_bytes=capacity)
                probe._runtime_precision_guard()
            except BaseException as error:
                evidence.update(passed=False, native_finalization_error=repr(error))
                if primary is None:
                    primary = error
        evidence["seconds"] = time.monotonic()-started
        evidence["counts"]["observed_functional_inner_SGD_steps"] = 5*evidence["counts"].get("finite_objective_completed", 0)
        evidence["counts"]["partial_finite_objective_uncommitted"] = (
            evidence["counts"].get("finite_objective_attempts", 0) != evidence["counts"].get("finite_objective_completed", 0))
        evidence["counts"]["completed_unroll_SGD_steps_are_lower_bound_on_interrupted_objective"] = True
        try:
            _write_new(output, evidence)
        except BaseException:
            if primary is not None:
                raise primary
            raise
    if primary is not None:
        raise primary
    return dict(evidence, evidence_path=str(output), evidence_sha256=_sha(output))


def prepare(cells, candidate, target, spec_path, spec_sha256, output_path, gate_sha256=None, stop=lambda: False):
    """One native update, or accepted same-source continuation24; zero students."""
    return _run("prepare", cells, candidate, spec_path, spec_sha256, output_path, target, gate_sha256, stop)


def certify(cells, candidate, frontier, spec_path, spec_sha256, output_path, gate_sha256=None, stop=lambda: False):
    """Readonly exact prefix replay and independent diagnostic stock5SGD."""
    return _run("certify", cells, candidate, spec_path, spec_sha256, output_path, frontier, gate_sha256, stop)


def validate(cells, candidate, arm, spec_path, spec_sha256, output_path, gate_sha256, stop=lambda: False):
    """Accepted full25 only; fixed0/25 students3700..2, shared physicalP0."""
    return _run("validate", cells, candidate, spec_path, spec_sha256, output_path, arm, gate_sha256, stop)
