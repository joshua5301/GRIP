"""Fixed schema3 finite-five-SGD assignment experiment; native horizon25.

AP2 and the pure engine remain unchanged. Atomic complete-prefix resumes are
validated by exact native objective and single-tensor Adam replay. Final student
fits are a separate, gate-required, cached-input-only phase with shared P0.
"""
import csv
import io
import json
import math
from pathlib import Path

import torch

from src import finite_student_probe as probe
from src.evaluation import _input_digest, fit_gcn_diagnostic
from src.io import _fingerprint
from src.moments import augmented, decode_moments, make_material
from src.soft_ce_partition import head_gradient, outer_value_gradient
from src.student_routes import replay_routes
from src.sweep_utils import representative

SCHEMA = 3
HORIZON = 25
SCIENCE_SHA = "23b8a1eee48040808af8c2b9311f3f02a9b08b2482817857067064af0026b6a2"
_FIXED = dict(probe._FIXED, finite_student_schema=SCHEMA, finite_assignment_steps=HORIZON)
_SEEDS = (3600, 3601, 3602)
_SETTINGS = dict(epochs=600, eval_every=1, hidden=256, dropout=0., lr=.001,
                 weight_decay=.001, lr_schedule="constant", initialization="geom_uniform")
_require, _stop, _tensor, _scalar = probe._require, probe._stop, probe._tensor, probe._scalar
_seal, _sha, _attach, _atomic = probe._seal, probe._sha, probe._attach, probe._atomic


def numerical_source():
    result = probe.numerical_source()
    _require(result["files"].get("finite_student_experiment.py") == _sha(__file__),
             "Actual schema3 experiment source/full manifest is missing or changed")
    return result


def canonical_candidate(candidate):
    _require(isinstance(candidate, dict), "Finite25 candidate must be a mapping")
    allowed = set(_FIXED) | {probe._SOURCE_FIELD, "outer_route"}
    _require(set(candidate) <= allowed and set(_FIXED) | {"outer_route"} <= set(candidate),
             "Explicit finite25 controls required; external/unknown/legacy controls forbidden")
    result = dict(candidate)
    for key, expected in _FIXED.items():
        actual = candidate[key]
        if type(expected) is int:
            valid = type(actual) is int and actual == expected
        elif type(expected) is float:
            valid = type(actual) in (int, float) and math.isfinite(actual) and actual == expected
        else:
            valid = type(actual) is str and actual == expected
        _require(valid, f"Changed fixed finite25 control: {key}")
        result[key] = expected
    _require(type(result["outer_route"]) is str and result["outer_route"] in probe._ROUTES,
             "Unsupported finite25 outer route")
    token = _seal(numerical_source())
    _require(result.get(probe._SOURCE_FIELD, token) == token, "Finite25 source changed; preserve old namespace")
    result[probe._SOURCE_FIELD] = token
    return result


def _context(candidate, buffers, environment, reference):
    result = probe._context(candidate, buffers, environment, reference)
    result.update(schema=SCHEMA, numerical_source=numerical_source(), allowed_steps=list(range(HORIZON+1)),
                  assignment_steps=HORIZON, scientific_preregistration_sha256=SCIENCE_SHA,
                  checkpoint_policy="complete native0..frontier; atomic resume authority; no AP continuation")
    return result


def _source_unchanged(context):
    probe._source_unchanged(context)
    _require(numerical_source() == context["numerical_source"], "Finite25 numerical source changed")


def _evaluate(buffers, candidate, context, parameters, step, scale, stop):
    _require(type(step) is int and 0 <= step <= HORIZON, "Invalid finite25 snapshot step")
    snapshot, scale = probe._evaluate(buffers, candidate, context, parameters, step, scale, stop)
    snapshot["schema"] = SCHEMA
    return snapshot, scale


def _adam_step(parameters, first, second, gradients, step):
    """Read-only recurrence matching fixed native noncapturable Adam ordering."""
    _require(type(step) is int and 1 <= step <= HORIZON and all(isinstance(x, (list, tuple)) and len(x) == 2
             for x in (parameters, first, second, gradients)), "Malformed finite25 Adam recurrence")
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
        p = parameter.detach().clone().addcdiv_(m, denominator, value=-.05/(1-.9**step))
        _require(all(bool(torch.isfinite(value).all()) for value in (p, m, v)), "Nonfinite native Adam recurrence")
        updated.append(p)
        next_first.append(m)
        next_second.append(v)
    return updated, next_first, next_second


def _optimizer(saved, parameters, first, second, step):
    _require(type(step) is int and 0 <= step <= HORIZON and isinstance(saved, dict)
             and set(saved) == {"state", "param_groups"} and isinstance(saved["param_groups"], list)
             and len(saved["param_groups"]) == 1, "Malformed finite25 Adam")
    group = saved["param_groups"][0]
    expected = dict(lr=.05, betas=[.9, .999], eps=1e-12, weight_decay=0, amsgrad=False, maximize=False,
                    foreach=False, capturable=False, differentiable=False, fused=False)
    _require(isinstance(group, dict) and set(group) in (set(expected)|{"params"},
             set(expected)|{"params", "decoupled_weight_decay"}) and group.get("params") == [0, 1]
             and all(type(x) is int for x in group["params"]), "Finite25 Adam controls/IDs changed")
    for key, value in expected.items():
        actual = group[key]
        if key == "betas":
            valid = isinstance(actual, (list, tuple)) and len(actual) == 2 and all(type(x) in (int, float) for x in actual) and list(actual) == value
        elif type(value) is bool:
            valid = actual is value
        else:
            valid = type(actual) in (int, float) and math.isfinite(actual) and actual == value
        _require(valid, "Finite25 Adam fixed controls changed")
    if "decoupled_weight_decay" in group:
        _require(group["decoupled_weight_decay"] is False, "Finite25 Adam decay changed")
    slots = saved["state"]
    _require(isinstance(slots, dict) and all(type(key) is int for key in slots)
             and set(slots) == (set() if step == 0 else {0, 1}), "Finite25 Adam slots/counters changed")
    if step == 0:
        return
    for index, parameter in enumerate(parameters):
        slot = slots[index]
        _require(isinstance(slot, dict) and set(slot) == {"step", "exp_avg", "exp_avg_sq"}, "Malformed finite25 Adam slot")
        counter = _tensor(slot["step"], (), torch.float32, "Malformed scalar FP32 Adam counter")
        _require(counter.device.type == "cpu" and float(counter) == step, "Finite25 Adam counter/device changed")
        for key, expected_value in (("exp_avg", first[index]), ("exp_avg_sq", second[index])):
            actual = _tensor(slot[key], parameter.shape, torch.float32, "Malformed finite25 Adam moments")
            _require(torch.equal(actual.to(expected_value), expected_value), "Finite25 Adam moments differ from exact native recurrence")


def _validate_bundle(bundle, buffers, candidate, context, stop):
    _require(isinstance(bundle, dict) and set(bundle) == {"schema", "context", "step", "parameters", "optimizer", "snapshots", "scale", "history", "content_sha256"}
             and type(bundle.get("schema")) is int and bundle["schema"] == SCHEMA and bundle.get("context") == context
             and _seal({key: value for key, value in bundle.items() if key != "content_sha256"}) == bundle.get("content_sha256"),
             "Finite25 source/schema/content changed")
    frontier = bundle.get("step")
    _require(type(frontier) is int and 0 <= frontier <= HORIZON and isinstance(bundle.get("snapshots"), dict)
             and all(type(key) is int for key in bundle["snapshots"]) and set(bundle["snapshots"]) == set(range(frontier+1)),
             "Malformed finite25 complete-prefix frontier")
    initial = [value.detach().clone() for value in buffers["initial"]]
    for step, snapshot in bundle["snapshots"].items():
        _require(isinstance(snapshot, dict) and {"schema", "context", "step", "parameters"} <= set(snapshot)
                 and type(snapshot["schema"]) is int and snapshot["schema"] == SCHEMA and snapshot["context"] == context
                 and type(snapshot["step"]) is int and snapshot["step"] == step
                 and isinstance(snapshot["parameters"], (list, tuple)) and len(snapshot["parameters"]) == 2
                 and isinstance(snapshot.get("optimizer"), dict),
                 "Malformed finite25 snapshot container")
        for value, original in zip(snapshot["parameters"], initial, strict=True):
            _tensor(value, original.shape, torch.float32, "Malformed finite25 snapshot parameters")
    _require(_seal(bundle["snapshots"][0]["parameters"]) == _seal(initial), "Finite25 original native U0/V0 changed")
    actual0, scale = _evaluate(buffers, candidate, context, initial, 0, None, stop)
    _require(type(bundle.get("scale")) is float and math.isfinite(bundle["scale"]) and bundle["scale"] > 0
             and bundle["scale"] == scale
             and _seal(actual0) == _seal({key: value for key, value in bundle["snapshots"][0].items() if key != "optimizer"}),
             "Finite25 coupled J0/model/partial certificate changed")
    parameters = initial
    first, second = [torch.zeros_like(x) for x in initial], [torch.zeros_like(x) for x in initial]
    _optimizer(bundle["snapshots"][0]["optimizer"], parameters, first, second, 0)
    previous = actual0
    for step in range(1, frontier+1):
        _stop(stop)
        gradients = [value.to(parameter) for value, parameter in zip(previous["scaled_factor_gradients"], parameters, strict=True)]
        parameters, first, second = _adam_step(parameters, first, second, gradients, step)
        _require(_seal(bundle["snapshots"][step]["parameters"]) == _seal(parameters), "Finite25 checkpoint Adam transition changed")
        actual, _ = _evaluate(buffers, candidate, context, parameters, step, scale, stop)
        _optimizer(bundle["snapshots"][step]["optimizer"], parameters, first, second, step)
        _require(_seal(actual) == _seal({key: value for key, value in bundle["snapshots"][step].items() if key != "optimizer"}),
                 "Finite25 native model/moment/partial/trace replay changed")
        previous = actual
    _require(_seal(bundle.get("parameters")) == _seal(parameters), "Finite25 current native parameters changed")
    _optimizer(bundle["optimizer"], parameters, first, second, frontier)
    _require(_seal(bundle["optimizer"]) == _seal(bundle["snapshots"][frontier]["optimizer"]),
             "Finite25 latest optimizer differs from actual terminal snapshot optimizer")
    expected_history = [_history(bundle["snapshots"][step]) for step in range(frontier+1)]
    _require(_seal(bundle.get("history")) == _seal(expected_history), "Finite25 history differs from actual source certificates")
    return frontier+1


def _history(snapshot):
    row = probe._history(snapshot)
    slots = snapshot["optimizer"]["state"]
    row.update(Adam_U_step=0. if snapshot["step"] == 0 else float(slots[0]["step"]),
               Adam_V_step=0. if snapshot["step"] == 0 else float(slots[1]["step"]))
    return row


def _mirrors(folder, bundle, *, export=False):
    """Only an exact canonical prefix can lag a committed atomic resume."""
    directory = folder / "checkpoints"
    snapshots = bundle["snapshots"]
    _require(not directory.exists() or directory.is_dir(), "Malformed finite25 checkpoint directory preserved")
    for path in directory.iterdir() if directory.exists() else ():
        _require(path.is_file() and path.name.startswith("step_") and path.suffix == ".pt", "Malformed finite25 mirror filename preserved")
        saved = torch.load(path, map_location="cpu", weights_only=False)
        step = saved.get("step") if isinstance(saved, dict) else None
        _require(type(step) is int and step in snapshots and path.name == f"step_{step:06d}.pt"
                 and _seal(saved) == _seal(snapshots[step]), "Malformed/orphan finite25 mirror preserved")
    csv_path = folder / "optimization.csv"
    if csv_path.exists():
        with csv_path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            rows, fields = list(reader), reader.fieldnames
        expected_fields = list(bundle["history"][0])
        _require(fields == expected_fields and 1 <= len(rows) <= len(bundle["history"]), "Malformed finite25 CSV prefix preserved")
        for row, expected in zip(rows, bundle["history"], strict=False):
            _require(set(row) == set(expected), "Malformed finite25 CSV columns preserved")
            try:
                valid = all(math.isfinite(float(row[key])) and float(row[key]) == value for key, value in expected.items())
            except (TypeError, ValueError):
                valid = False
            _require(valid, "Finite25 CSV differs from exact native history prefix")
    if export:
        directory.mkdir(exist_ok=True)
        for step, snapshot in snapshots.items():
            path = directory / f"step_{step:06d}.pt"
            if not path.exists():
                _atomic(path, snapshot, True)
        text = io.StringIO()
        writer = csv.DictWriter(text, fieldnames=list(bundle["history"][0]))
        writer.writeheader()
        writer.writerows(bundle["history"])
        _atomic(csv_path, text.getvalue(), False)


def _prepare(dataset, ratio, output_dir, candidate, condensation_seed, data_dir, device, citation_features, stop):
    candidate = canonical_candidate(candidate)
    _stop(stop)
    probe._public_options(dataset, ratio, output_dir, condensation_seed, data_dir, device, citation_features)
    environment = probe._native(device)
    buffers = probe._load_source(dataset, ratio, output_dir, candidate, condensation_seed, data_dir, device, citation_features, stop)
    reference = probe._p0_reference(buffers)
    context = _context(candidate, buffers, environment, reference)
    return candidate, buffers, context


def _bundle_paths(buffers, candidate):
    folder = buffers["root"] / _fingerprint(candidate) / "condensation_0"
    return folder, folder.parent / "candidate.json", folder / "resume.pt"


def _load_bundle(folder, candidate_path, resume_path, candidate, buffers, context, stop, *, required=False):
    if candidate_path.exists():
        _require(json.loads(candidate_path.read_text()) == candidate, "Finite25 candidate identity changed")
    if not resume_path.exists():
        _require(not required and not any(folder.glob("**/*")), "Finite25 missing/orphan resume preserved; no fallback")
        return None, 0
    _require(candidate_path.is_file(), "Finite25 cached resume requires canonical candidate")
    bundle = torch.load(resume_path, map_location="cpu", weights_only=False)
    unrolls = _validate_bundle(bundle, buffers, candidate, context, stop)
    _mirrors(folder, bundle)
    return bundle, unrolls


def _commit(folder, candidate_path, resume_path, candidate, buffers, context, bundle, stop):
    probe._checked_files(buffers["pins"])
    _source_unchanged(context)
    _stop(stop)
    _mirrors(folder, bundle)  # Validate present artifacts before replacing authority.
    folder.mkdir(parents=True, exist_ok=True)
    if not candidate_path.exists():
        _atomic(candidate_path, json.dumps(candidate, indent=2), False)
    _atomic(resume_path, bundle, True)
    _mirrors(folder, bundle, export=True)


def prepare_full25(dataset, ratio, output_dir, candidate, condensation_seed=0, data_dir="data", device="cuda",
                   citation_features="row", stop=lambda: False):
    """Commit fresh own25 trajectory or a validated own prefix; zero students."""
    candidate = canonical_candidate(candidate)
    _stop(stop)
    probe._public_options(dataset, ratio, output_dir, condensation_seed, data_dir, device, citation_features)
    probe._native(device)
    torch.cuda.reset_peak_memory_stats()
    capacity = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
    candidate, buffers, context = _prepare(dataset, ratio, output_dir, candidate, condensation_seed, data_dir, device, citation_features, stop)
    folder, candidate_path, resume_path = _bundle_paths(buffers, candidate)
    bundle, diagnostic_unrolls = _load_bundle(folder, candidate_path, resume_path, candidate, buffers, context, stop)
    starting_step = 0 if bundle is None else bundle["step"]
    parameters = [value.detach().clone().requires_grad_(True) for value in buffers["initial"]]
    optimizer = torch.optim.Adam(parameters, lr=.05, eps=1e-12, foreach=False, fused=False)
    if bundle is not None:
        for target, value in zip(parameters, bundle["parameters"], strict=True):
            with torch.no_grad():
                target.copy_(value.to(target))
        optimizer.load_state_dict(bundle["optimizer"])
    scientific_unrolls = 0
    if bundle is None:
        snapshot, scale = _evaluate(buffers, candidate, context, parameters, 0, None, stop)
        scientific_unrolls += 1
        snapshot["optimizer"] = probe.cpu_state(optimizer.state_dict())
        bundle = _attach(dict(schema=SCHEMA, context=context, step=0, parameters=parameters, optimizer=optimizer.state_dict(),
                             snapshots={0: snapshot}, scale=scale, history=[_history(snapshot)]))
        _commit(folder, candidate_path, resume_path, candidate, buffers, context, bundle, stop)
    for step in range(bundle["step"]+1, HORIZON+1):
        _stop(stop)
        for parameter, gradient in zip(parameters, bundle["snapshots"][step-1]["scaled_factor_gradients"], strict=True):
            parameter.grad = gradient.to(parameter).clone()
        optimizer.step()
        snapshot, scale = _evaluate(buffers, candidate, context, parameters, step, bundle["scale"], stop)
        scientific_unrolls += 1
        snapshot["optimizer"] = probe.cpu_state(optimizer.state_dict())
        bundle = _attach(dict(schema=SCHEMA, context=context, step=step, parameters=parameters, optimizer=optimizer.state_dict(),
                             snapshots={**bundle["snapshots"], step: snapshot}, scale=scale,
                             history=[*bundle["history"], _history(snapshot)]))
        _commit(folder, candidate_path, resume_path, candidate, buffers, context, bundle, stop)
    probe._checked_files(buffers["pins"])
    _source_unchanged(context)
    _stop(stop)
    _mirrors(folder, bundle, export=True)
    torch.cuda.synchronize()
    return dict(root=str(buffers["root"].resolve()), candidate_path=str(folder.parent.resolve()), candidate_id=_fingerprint(candidate),
                outer_route=candidate["outer_route"], step=HORIZON, cached=starting_step == HORIZON, validation_only=True,
                P_updates=HORIZON-starting_step, student_fits=0, finite_student_schema=SCHEMA, finite_assignment_steps=HORIZON,
                finite_source_context_digest=_seal(context), own_J0=bundle["scale"], teacher_ce=bundle["snapshots"][HORIZON]["teacher_ce"],
                CUDA_peak_allocated_bytes=torch.cuda.max_memory_allocated(), CUDA_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                CUDA_total_bytes=capacity, counts=dict(committed_P_updates=HORIZON-starting_step, final_student_fits=0,
                teacher_map_Phi_hardinit_fits=0, scientific_objective_unrolls=scientific_unrolls,
                diagnostic_cache_replay_unrolls=diagnostic_unrolls, functional_inner_SGD_steps=5*(scientific_unrolls+diagnostic_unrolls)),
                caveat="Fresh schema3 horizon25; no AP continuation; functional5SGD differs from finalAdam; linear λ1e-4 reference differs from active .001 allparameter innerWD.")


def _gate(path, checksum):
    """Fail before native/data/fit calls; a frozen external audit is required."""
    from src.research_loop import implementation_provenance
    _require(type(checksum) is str and len(checksum) == 64 and isinstance(path, (str, Path)), "Explicit frozen native gate path/SHA required")
    path = Path(path)
    _require(path.is_file() and _sha(path) == checksum, "Native full25 gate missing or changed")
    gate = json.loads(path.read_text())
    _require(isinstance(gate, dict) and gate.get("passed") is True and type(gate.get("finite_student_schema")) is int
             and gate["finite_student_schema"] == SCHEMA and type(gate.get("finite_assignment_steps")) is int
             and gate["finite_assignment_steps"] == HORIZON and gate.get("full25_cache_replay_passed") is True
             and gate.get("actual_FP32_P0_X_Q_uniform_equal_reference") is True
             and gate.get("source") == implementation_provenance(), "Unaccepted or incompatible native full25 gate")
    science = gate.get("scientific_preregistration")
    _require(isinstance(science, dict) and set(science) == {"path", "sha256"}
             and science["sha256"] == SCIENCE_SHA, "Native gate scientific policy changed")
    _require(isinstance(gate.get("files_sha256"), dict) and gate["files_sha256"], "Native gate lacks immutable numerical file pins")
    probe._checked_files({**gate["files_sha256"], science["path"]: SCIENCE_SHA})
    expected = {route: _fingerprint(canonical_candidate(dict(_FIXED, outer_route=route))) for route in probe._ROUTES}
    _require(gate.get("candidate_ids") == expected, "Native gate candidate/source token changed")
    return gate


def _gate_files(gate, paths):
    _require(all(str(path.resolve()) in gate["files_sha256"] and _sha(path) == gate["files_sha256"][str(path.resolve())]
                 for path in paths), "Used numerical cache file is absent from the frozen native gate")


def _reference_snapshots(buffers, gate):
    root = buffers["root"] / probe.REFERENCE_ID
    files = [root / "candidate.json", root / "condensation_0/resume.pt"]
    files += [root / f"condensation_0/checkpoints/step_{step:06d}.pt" for step in (0, HORIZON)]
    _gate_files(gate, files)
    resume = torch.load(files[1], map_location="cpu", weights_only=False)
    _require(isinstance(resume, dict) and isinstance(resume.get("snapshots"), dict), "Malformed reference resume preserved")
    result = {}
    for step, path in zip((0, HORIZON), files[2:], strict=True):
        snapshot = torch.load(path, map_location="cpu", weights_only=False)
        _require(isinstance(snapshot, dict) and type(snapshot.get("step")) is int and snapshot["step"] == step
                 and step in resume["snapshots"] and _seal(snapshot) == _seal(resume["snapshots"][step]),
                 "Cached original NODE endpoint/resume changed")
        moments = _tensor(snapshot.get("moments"), (70, 1441), torch.float64, "Malformed reference moments").to(buffers["z"])
        centers, labels, mass = decode_moments(moments, 1433)
        _require(bool((mass > 0).all()) and float((mass.sum()-1).abs()) <= 1e-12
                 and float((labels.sum(1)-1).abs().max()) <= 1e-12
                 and float((moments.sum(0)-make_material(buffers["z"], buffers["q"]).mean(0)).abs().max()) <= 1e-12,
                 "Cached reference material conservation changed")
        theta = _tensor(snapshot.get("theta"), (7, 1434), torch.float64, "Malformed reference head").to(buffers["z"])
        gradient = float(head_gradient(augmented(centers), labels, torch.full_like(mass, 1/70), theta, 1e-4).abs().max())
        value, _ = outer_value_gradient(buffers["z"], buffers["q"], theta, 65536, augmented(buffers["z"]))
        _scalar(snapshot.get("inner_grad_max"), "Malformed reference head certificate")
        _scalar(snapshot.get("teacher_ce"), "Malformed reference CE certificate")
        _require(snapshot.get("J_exact") is True and gradient <= 1e-7*(1+1e-6)
                 and math.isclose(gradient, snapshot["inner_grad_max"], abs_tol=1e-12, rel_tol=1e-12)
                 and math.isclose(value, snapshot["teacher_ce"], abs_tol=1e-12, rel_tol=1e-12),
                 "Cached original NODE linear certificate changed")
        result[step] = snapshot
    return result


def _student_paths(folder, seed):
    return dict(metadata=folder/f"seed_{seed}.json", epochs=folder/f"seed_{seed}_epochs.csv",
                selected=folder/f"seed_{seed}_selected.pt", routes=folder/f"seed_{seed}_validation_routes_v1.json",
                certificate=folder/f"seed_{seed}_completion_v1.json")


def _selected_route_counts(selected, buffers, stop):
    """Two diagnostic forwards only, no training or independent epoch selection."""
    import torch.nn.functional as F

    from src.evaluation import _forward
    from src.models import GCN
    state = selected["model_state"]
    expected = {"layers.0.lin.weight": (256, 1433), "layers.0.bias": (256,),
                "layers.1.lin.weight": (7, 256), "layers.1.bias": (7,)}
    _require(isinstance(state, dict) and set(state) == set(expected), "Malformed selected student state")
    for key, shape in expected.items():
        _tensor(state[key], shape, torch.float32, "Malformed selected student weights")
    with torch.random.fork_rng(devices=[]):
        model = GCN(1433, 256, 7, 2, 0.)
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
                settings=_SETTINGS, uniform_supplied_weights=True, synthetic_adjacency=None,
                selection="first strict maximum GCN validation accuracy", test_enabled=False)


def _student_certificate(paths, header, buffers, stop):
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
    expected_recipe = dict(version=2, seed=header["seed"], settings=_SETTINGS, layers=2,
                           optimizer="Adam; constant lr", weighting="normalized supplied mass",
                           selection="first maximum validation accuracy", test_enabled=False,
                           test_evaluation="selected weights once", torch_version=torch.__version__,
                           input_digest=header["input_digest"])
    _require(_seal(recipe) == _seal(expected_recipe) and metadata.get("fingerprint") == _fingerprint(expected_recipe)
             and selected.get("fingerprint") == metadata["fingerprint"],
             "Final student full recipe/input/selected fingerprint changed")
    with paths["epochs"].open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    _require(len(rows) == 600, "Incomplete final student epoch history")
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
             and route_recipe.get("settings") == _SETTINGS and route_recipe.get("test_enabled") is False,
             "Selected sameweights route cache provenance changed")
    # This call checks actual graph/H/state/mask content digests even for a hot cache.
    replay_routes(paths["selected"], buffers["graph"], buffers["h"], {"val": buffers["val"]}, _SETTINGS,
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
        fit_gcn_diagnostic(cx, cy, weights, buffers["graph"], buffers["q"],
                           {"train": buffers["train"], "val": buffers["val"]}, seed=seed,
                           folder=folder, training_adjacency=None, stop=stop, **_SETTINGS)
        replay_routes(paths["selected"], buffers["graph"], buffers["h"], {"val": buffers["val"]}, _SETTINGS,
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


def evaluate_cached(dataset, ratio, output_dir, arm, native_gate_path, native_gate_sha256, candidate=None,
                    condensation_seed=0, data_dir="data", device="cuda", citation_features="row", stop=lambda: False):
    """Gate-required0/25 final fits only, no source/teacher/head/P fallback."""
    _require(type(arm) is str and arm in ("node_reference", "sgc_mlp", "gcn"), "Unsupported fixed validation arm")
    if arm == "node_reference":
        _require(candidate is None, "Reference arm forbids external candidate overrides")
        loader_candidate = canonical_candidate(dict(_FIXED, outer_route="sgc_mlp"))
    else:
        loader_candidate = canonical_candidate(candidate)
        _require(loader_candidate["outer_route"] == arm, "Validation arm/candidate route changed")
    gate = _gate(native_gate_path, native_gate_sha256)  # Before data/native/student calls.
    loader_candidate, buffers, context = _prepare(dataset, ratio, output_dir, loader_candidate, condensation_seed,
                                                  data_dir, device, citation_features, stop)
    _gate_files(gate, [Path(path) for path in buffers["pins"]])
    reference = _reference_snapshots(buffers, gate)
    diagnostic_unrolls = 0
    if arm == "node_reference":
        cid, snapshots = probe.REFERENCE_ID, reference
    else:
        folder, candidate_path, resume_path = _bundle_paths(buffers, loader_candidate)
        required = [candidate_path, resume_path, folder/"optimization.csv"]
        required += [folder/f"checkpoints/step_{step:06d}.pt" for step in range(HORIZON+1)]
        _gate_files(gate, required)
        bundle, diagnostic_unrolls = _load_bundle(folder, candidate_path, resume_path, loader_candidate, buffers, context, stop, required=True)
        _require(bundle["step"] == HORIZON, "Final validation requires complete25; no P continuation")
        cid, snapshots = _fingerprint(loader_candidate), bundle["snapshots"]
    rows = []
    for step in (0, HORIZON):
        _stop(stop)
        cx, cy, mass = representative(snapshots[step]["moments"], probe._transform(buffers), 1433, device)
        weights = torch.full_like(mass, 1/70)
        if step == 0:
            rx, rq, rm = representative(reference[0]["moments"], probe._transform(buffers), 1433, device)
            _require(all(torch.equal(a, b) for a, b in zip((cx, cy, weights), (rx, rq, torch.full_like(rm, 1/70)), strict=True)),
                     "Shared P0 actual FP32 X/Q/uniform input equality failed; stop")
            folder = buffers["root"] / "finite_schema3_validation" / loader_candidate[probe._SOURCE_FIELD] / "shared_P0"
        else:
            folder = buffers["root"] / cid / "condensation_0/validation" / f"step25_{_fingerprint(_SETTINGS)}"
        probe._checked_files(buffers["pins"])
        _source_unchanged(context)
        for seed in _SEEDS:
            row = _evaluate_student(folder, (cx, cy, weights), buffers, context, native_gate_sha256, seed, stop)
            rows.append(dict(arm=arm, candidate_id=cid, step=step, **row))
    return dict(root=str(buffers["root"].resolve()), arm=arm, candidate_id=cid, validation_only=True,
                finite_student_schema=SCHEMA, native_gate_sha256=native_gate_sha256, rows=rows,
                counts=dict(P_updates=0, logical_student_conditions=6, logical_serving_conditions=12,
                            physical_final_GCN_fits=sum(row["actual_student_fits"] for row in rows),
                            physical_serving_route_outputs=sum(row["physical_route_outputs"] for row in rows),
                            diagnostic_route_forwards=sum(row["diagnostic_route_forwards"] for row in rows),
                            diagnostic_cache_replay_unrolls=diagnostic_unrolls,
                            diagnostic_functional_inner_SGD_steps=5*diagnostic_unrolls, extra_MLP_fits=0,
                            reference_linear_certificate_gradient_checks=3),
                caveat="Shared P0 supplies one3seedbaseline; finite5SGD states are not finalAdam student states. GCN-selected weights/epoch serve both originalCSR and originalcachedH routes; no test metrics.")
