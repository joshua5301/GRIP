"""Require-existing Arxiv90 source/new-RMS/legacy-P0 compatibility; no fits.

Numerical imports occur only after immutable metadata and every file prerequisite.
A newly reconstructed RMS is never described as a recovered historical origin.
"""
import ast
import hashlib
import importlib
import json
import math
import os
import platform
import signal
import subprocess
import time
from importlib.metadata import version
from pathlib import Path

SCIENCE = "Arxiv90_require_existing_RMS_P0_readonly_scientific_stageBT_v1.json"
SCIENCE_SHA = "d5b8a0b87cc038c75d77e3a7bf483246d56c8206a30ad1aeb241c22a27d71ba1"
REVIEW = "independent_scientific_review_stageBT_v1.json"
REVIEW_SHA = "b5b56b9e6e0b3b5eb52e2a1896946cd86d1c3b35535cd9e547f1d9bbc4f2b1ae"
FIELDS = {"schema", "fixed", "source", "numerical_source", "python_version", "files_sha256",
          "artifacts_sha256", "protocol", "output_path", "family_protocol", "candidate_protocol"}
FIXED = dict(dataset="arxiv", cells=90, nodes=169343, features=128, classes=40, basis=512,
             temperature=.3, rank=16, factor_seed=0, mixing=.05, chunk=2048,
             RMS_kind="rms", RMS_eps=1e-12, max_seconds=300, legacy_endpoint=20,
             historical_RMS_recovered=False, test_enabled=False)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


def _seal(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:12]


def _source(repo):
    files = {str(p.relative_to(repo)): _sha(p) for p in sorted((repo / "src").glob("*.py"))}
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True)
    return dict(git_head=head.stdout.strip(), source_digest=_fingerprint(files), files=files)


def numerical_source(repo=None):
    """Metadata-only descriptor; runtime versions are checked after lazy import."""
    repo = Path(__file__).resolve().parents[1] if repo is None else Path(repo)
    torch_root = Path(importlib.util.find_spec("torch").origin).parent
    module = ast.parse((torch_root / "version.py").read_text())
    literal = next(ast.literal_eval(n.value) for n in module.body if isinstance(n, (ast.Assign, ast.AnnAssign))
                   and any(isinstance(t, ast.Name) and t.id == "__version__"
                           for t in (n.targets if isinstance(n, ast.Assign) else [n.target])))
    return dict(files={p.name: _sha(p) for p in sorted((repo / "src").glob("*.py"))},
                versions=dict(torch=literal, pyg=version("torch-geometric"), numpy=version("numpy"), scipy=version("scipy"), sklearn=version("scikit-learn")))


def _checked(pins):
    _require(isinstance(pins, dict) and pins, "Empty file inventory")
    for path, checksum in pins.items():
        _require(type(path) is str and Path(path).is_absolute() and type(checksum) is str
                 and len(checksum) == 64 and Path(path).is_file() and _sha(path) == checksum,
                 "Missing or changed existing file: " + str(path))


def _extra_paths(science):
    family = science["family"]
    return [Path(family["geometry_root"]) / "phi.npy",
            Path(family["candidate_root"]) / "condensation/checkpoints/step_000020.pt"]


def _preserve(spec, path, checksum, science, repo):
    _require(_sha(path) == checksum and _source(repo) == spec["source"]
             and numerical_source(repo) == spec["numerical_source"]
             and platform.python_version() == spec["python_version"], "Spec/source/Git/numerical/Python changed")
    for pins in (spec["files_sha256"], spec["artifacts_sha256"],
                 {str(repo / "results/proposals" / SCIENCE): SCIENCE_SHA,
                  str(repo / "results/proposals" / REVIEW): REVIEW_SHA,
                  spec["protocol"]["path"]: spec["protocol"]["sha256"]},
                 {str(repo / p): h for p, h in science["source_before"]["files"].items() if p != "src/research_loop.py"},
                 {str(repo / p): h for p, h in science["old_tests_sha256"].items()}):
        _checked(pins)


def _controls(spec, science, repo):
    _require(isinstance(spec, dict) and set(spec) == FIELDS and type(spec["schema"]) is int and spec["schema"] == 1,
             "Require exact eleven-field spec")
    for key, expected in (("fixed", FIXED), ("protocol", science["design"]),
                          ("files_sha256", science["original_files_sha256"]),
                          ("output_path", science["implementation_contract"]["output_path"]),
                          ("family_protocol", science["family"]["teacher_protocol"]),
                          ("candidate_protocol", science["family"]["candidate_protocol"])):
        _require(_seal(spec[key]) == _seal(expected), "Changed fixed field: " + key)
    owned = {repo / "src/large_source_preflight.py", repo / "src/research_loop.py"}
    extras = set(_extra_paths(science))
    pins = spec["artifacts_sha256"]
    _require(isinstance(pins, dict) and all(type(p) is str and Path(p).is_absolute()
             and (Path(p).resolve().is_relative_to(repo / "results") or Path(p).resolve() in owned) for p in pins)
             and extras <= {Path(p) for p in pins}, "Require results/exact owned artifacts and two preservation-only extra assets")
    _require(isinstance(spec["source"], dict) and set(spec["source"].get("files", {})) ==
             set(science["source_before"]["files"]) | {"src/large_source_preflight.py"}, "Require promoted source81")
    _require(spec["numerical_source"].get("versions", {}).get("torch") == science["native_policy"]["torch"],
             "Require frozen native Torch literal")


def _load_spec(path, checksum):
    repo = Path(__file__).resolve().parents[1]
    path = Path(path).resolve()
    _require(type(checksum) is str and len(checksum) == 64 and path.is_file() and _sha(path) == checksum,
             "Require exact existing spec")
    science_path = repo / "results/proposals" / SCIENCE
    _checked({str(science_path): SCIENCE_SHA})
    science = json.loads(science_path.read_text())
    _require(str(path) == science["implementation_contract"]["spec_path"], "Require fixed execution spec path")
    spec = json.loads(path.read_text())
    _controls(spec, science, repo)
    _preserve(spec, path, checksum, science, repo)
    _require(not Path(spec["output_path"]).exists(), "Preserve existing receipt; no retry")
    for extra in _extra_paths(science):
        _require(extra.is_file(), "Require existing Phi and legacy endpoint20")
    return spec, science, repo


def _libraries():
    # No obsolete prototype/public experiment or creation getter is invoked.
    names = ("torch", "torch_geometric", "numpy", "scipy", "sklearn", "src.data", "src.shared_features",
             "src.nystrom_ce", "src.large_pilot", "src.target_refinement", "src.transforms",
             "src.low_rank_assignment", "src.moments", "src.sweep_utils", "src.soft_ce_partition",
             "src.finite_student_outer_v2", "src.io")
    return {name: importlib.import_module(name) for name in names}


def _count(evidence, name, completed=False):
    key = name + ("_completed" if completed else "_attempts")
    evidence["counts"][key] = evidence["counts"].get(key, 0) + 1


def _call(evidence, name, function, *args, **kwargs):
    evidence["_guard"]()
    _count(evidence, name)
    result = function(*args, **kwargs)
    _count(evidence, name, True)
    evidence["_guard"]()
    return result


def _precision(torch):
    _require(os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8" and "CUDA_VISIBLE_DEVICES" not in os.environ
             and torch.get_num_threads() == 4 and torch.get_default_dtype() == torch.float32
             and torch.are_deterministic_algorithms_enabled() and not torch.is_deterministic_algorithms_warn_only_enabled()
             and torch.get_float32_matmul_precision() == "highest" and not torch.is_autocast_enabled()
             and not torch.is_autocast_enabled("cpu") and not torch.backends.cuda.matmul.allow_tf32
             and not torch.backends.cudnn.allow_tf32, "Native precision/threads/environment changed")


def _native(libs, science):
    t = libs["torch"]
    _require("CUDA_VISIBLE_DEVICES" not in os.environ and os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
             and not t.cuda.is_initialized() and t.get_default_dtype() == t.float32
             and not t.is_autocast_enabled() and not t.is_autocast_enabled("cpu"), "Require fresh native CUDA without AMP")
    t.set_num_threads(4)
    t.use_deterministic_algorithms(True, warn_only=False)
    t.set_float32_matmul_precision("highest")
    t.backends.cuda.matmul.allow_tf32 = False
    t.backends.cudnn.allow_tf32 = False
    _precision(t)
    p = science["native_policy"]
    versions = dict(torch=str(t.__version__), pyg=str(libs["torch_geometric"].__version__),
                    numpy=str(libs["numpy"].__version__), scipy=str(libs["scipy"].__version__),
                    sklearn=str(libs["sklearn"].__version__))
    _require(str(t.__version__) == p["torch"] and str(t.version.cuda) == p["cuda_runtime"]
             and t.cuda.is_available() and t.cuda.device_count() == 1 and t.cuda.get_device_name(0) == p["GPU"]
             and t.cuda.get_device_properties(0).total_memory == p["CUDA_total_bytes"], "Native CUDA/version/capacity changed")
    t.cuda.set_device(0)
    t.cuda.reset_peak_memory_stats(0)
    return dict(device="cuda:0", GPU=t.cuda.get_device_name(0), CUDA_total_bytes=p["CUDA_total_bytes"],
                cuda_runtime=str(t.version.cuda), versions=versions, python_version=platform.python_version(), threads=t.get_num_threads(),
                deterministic=True, warn_only=False, AMP=False, TF32=False, default_dtype=str(t.get_default_dtype()),
                matmul_precision=t.get_float32_matmul_precision(), CUBLAS_WORKSPACE_CONFIG=os.environ["CUBLAS_WORKSPACE_CONFIG"],
                source_backend="original_Arxiv_CSR_source_readonly_compatibility_only")


def _descriptor(libs, value):
    t = libs["torch"]
    if t.is_tensor(value):
        if value.layout == t.sparse_csr:
            return dict(shape=list(value.shape), dtype=str(value.dtype), layout=str(value.layout),
                        parts=[_descriptor(libs, v) for v in (value.crow_indices(), value.col_indices(), value.values())])
        _require(value.layout == t.strided, "Unsupported source layout")
        return libs["src.shared_features"]._tensor_identity(value)
    if isinstance(value, dict):
        return {str(k): _descriptor(libs, v) for k, v in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_descriptor(libs, v) for v in value]
    return value


def _tensor(t, value, shape, dtype, name):
    _require(t.is_tensor(value) and tuple(value.shape) == tuple(shape) and value.dtype == dtype
             and bool(t.isfinite(value).all()), "Malformed " + name)
    return value


def _expected_core_config(repo, data_digest):
    """Reproduce only original resume metadata; optimizer is never called."""
    tree = ast.parse((repo / "src/soft_ce_partition.py").read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "optimize_ce_assignment")
    defaults = dict(zip([a.arg for a in fn.args.args][-len(fn.args.defaults):], fn.args.defaults))
    excluded = {"steps", "folder", "checkpoint_steps", "resume_state", "save_resume", "log_every",
                "initial_representatives", "outer_indices", "implicit_solver", "inner_solver", "temperature_logits", "outer_targets", "stop"}
    config = {key: ast.literal_eval(value) for key, value in defaults.items() if key not in excluded}
    config.update(penalty=.0001, lr=.01, mixing=.05, chunk_size=2048, assignment_rank=16, factor_seed=0,
                  inner_method="newton_first", solver_mode="exact", inner_loss_weighting="uniform",
                  outer_chunk_size=2048, save_assignment=False, implicit_warm_start=True, data_digest=data_digest)
    for key in ("temperature_initial", "temperature_lr", "cg_check_interval", "cache_assignment",
                "node_weighting", "node_weight_penalty", "node_weight_lr"):
        config.pop(key)
    return config


def _execute(spec, science, repo, libs, evidence):
    t, shared, ny = libs["torch"], libs["src.shared_features"], libs["src.nystrom_ce"]
    guard = evidence["_guard"]
    family = Path(science["family"]["geometry_root"])
    candidate = Path(science["family"]["candidate_root"])
    load = lambda path: _call(evidence, "existing_tensor_cache_loads", t.load, path, map_location="cuda", weights_only=False)
    graph, train, validation, unused_testing, fresh_h = _call(evidence, "dataset_loader_calls", libs["src.data"]._prepare_dataset,
                                                            "arxiv", repo / "data", "cuda", citation_features="default")
    del unused_testing
    evidence["counts"]["integrated_loader_H_products_completed"] = 2
    evidence["integrated_loader_H_partial_work"] = "Two products only after loader returned; partial internal work unknown on failure."
    _tensor(t, graph["x"], (169343, 128), t.float32, "raw X")
    _require(graph["adj"].layout == t.sparse_csr and graph["adj"].dtype == t.float32
             and tuple(graph["adj"].shape) == (169343, 169343) and bool(t.isfinite(graph["adj"].values()).all())
             and train.dtype == t.bool and tuple(train.shape) == (169343,) and bool(train.any())
             and validation[0] is graph and validation[1].dtype == t.bool and tuple(validation[1].shape) == (169343,),
             "Malformed original full graph/masks/CSR")
    shared._validate_matrix(fresh_h, "Fresh source H")
    source_digest = libs["src.large_pilot"]._data_digest(graph, train, validation, None, guard)
    token = _fingerprint(dict(data=source_digest, propagation="SGC2-normalize-adj-v1", torch=str(t.__version__), dtype=str(fresh_h.dtype)))
    state = _call(evidence, "existing_tensor_cache_loads", shared._load_state, family / "propagated_H.pt", "shared_h")
    h = _tensor(t, state.get("h"), (169343, 128), fresh_h.dtype, "saved H")
    _require(state.get("source_digest") == token and state.get("identity") == shared._tensor_identity(h)
             and h.shape == fresh_h.shape and h.dtype == fresh_h.dtype, "Saved H source/content/shape differs")
    h = h.to(device="cuda")
    del fresh_h
    digest = libs["src.large_pilot"]._data_digest(graph, train, validation, h, guard)
    _require(digest == spec["family_protocol"]["data_digest"] == spec["candidate_protocol"]["data_digest"], "Original family/source digest differs")
    evidence.update(exact_saved_H_source_passed=True, source_data_digest=digest, shared_H_source_digest=token,
                    saved_H_identity=state["identity"])
    h = h.double()  # exact original large_pilot line440, preceding fit_transform.
    map_state = load(family / "feature_map.pt")
    _require(isinstance(map_state, dict) and set(map_state) == {"anchors", "mapping", "kernel"}
             and map_state["kernel"] == "relu", "Original raw map schema differs")
    _tensor(t, map_state["anchors"], (512, 128), t.float64, "map anchors")
    _tensor(t, map_state["mapping"], (512, 512), t.float64, "map mapping")
    feature_map = _call(evidence, "map_constructor_calls", ny.NystromMap, **map_state)
    identity = ny._cache_identity(h, feature_map, (169343, 512), 2048, guard)
    sidecar = json.loads((family / "phi.meta.json").read_text())
    _require({key: sidecar.get(key) for key in identity} == identity and type(sidecar.get("phi_digest")) is str
             and len(sidecar["phi_digest"]) == 64, "Original H/map/Phi sidecar identity differs")
    teacher = load(family / "teacher.pt")
    _require(isinstance(teacher, dict) and teacher.get("converged") is True and teacher.get("gamma") == .01
             and teacher.get("data_digest") == digest, "Require original converged kernel teacher lineage")
    logits = teacher["logits"]
    _require(t.is_tensor(logits) and tuple(logits.shape) == (169343, 40) and logits.is_floating_point()
             and bool(t.isfinite(logits).all()), "Malformed teacher logits")
    shared._validate_matrix(teacher["weight"], "Teacher weight")
    _require(tuple(teacher["weight"].shape) == (512, 40), "Teacher weight shape differs")
    q = _call(evidence, "teacher_Q_constructions", libs["src.target_refinement"].training_refined_targets,
              logits, .3, graph["y"], train, mixing=0)
    _require(q.dtype == t.float64, "Original softmax result must be FP64 source Q")
    libs["src.finite_student_outer_v2"]._probabilities(q, "Original source Q")
    evidence.update(existing_teacher_map_Q_passed=True, map_identity=identity, teacher_saved_converged=True,
                    fresh_H_numerical_equality_assessed=False)
    z, transform = _call(evidence, "RMS_diagnostic_fit_transform_calls", libs["src.transforms"].fit_transform,
                         h, kind="rms", eps=1e-12)
    hard = _tensor(t, load(candidate / "initial_assignment.pt"), (169343,), t.int64, "hard assignment")
    _require(int(hard.min()) >= 0 and int(hard.max()) < 90 and len(t.unique(hard)) == 90, "Hard assignment classes differ")
    resume = load(candidate / "condensation/resume.pt")
    snapshot = load(candidate / "condensation/checkpoints/step_000000.pt")
    config = _expected_core_config(repo, libs["src.io"].array_digest(z.detach().cpu().numpy(), q.detach().cpu().numpy(), hard.cpu().numpy()))
    _require(isinstance(resume, dict) and type(resume.get("step")) is int and resume["step"] == 20
             and _seal(resume.get("config")) == _seal(config) and isinstance(snapshot, dict)
             and type(snapshot.get("step")) is int and snapshot["step"] == 0
             and _descriptor(libs, snapshot) == _descriptor(libs, resume.get("snapshots", {}).get(0)), "Original config/resume/endpoint0 differs")
    retained = dict(graph=graph, train=train, validation_mask=validation[1], H=h, z=z, Q=q, hard=hard,
                    transform=vars(transform), map=map_state, teacher=teacher,
                    initial_moments=resume["initial_moments"], legacy0=snapshot)
    evidence["_retained"] = retained
    evidence["native_buffers_before"] = _descriptor(libs, retained)
    low, moments = libs["src.low_rank_assignment"], libs["src.moments"]
    u, v = _call(evidence, "native_factor_factory_calls", low.initialize_factors, hard, 90, 16, 0)
    u2, v2 = _call(evidence, "native_factor_factory_calls", low.initialize_factors, hard, 90, 16, 0)
    _require(u.dtype == v.dtype == t.float32 and t.equal(u, u2) and t.equal(v, v2), "Current native factor repeat differs")
    evidence["current_native_factory_repeat_passed"] = True
    material = moments.make_material(z, q)
    with t.no_grad():
        current = _call(evidence, "P0_materializations", low.LowRankMoments.apply, u, v, hard, material, .05, 2048)
    legacy = _tensor(t, snapshot.get("moments"), (90, 169), t.float64, "legacy M0").to(current)
    _require(t.equal(current, legacy) and t.equal(resume["initial_moments"].to(current), legacy), "Current M0 is not bitwise legacy0")
    _require(bool((current[:, 0] > 0).all()) and t.allclose(current.sum(0), material.mean(0), atol=1e-12, rtol=1e-12), "P0 material conservation differs")
    centers, labels, mass = moments.decode_moments(current, 128)
    _require(bool((labels >= 0).all()) and float((labels.sum(1) - 1).abs().max()) <= 1e-12
             and abs(float(mass.sum()) - 1) <= 1e-12, "P0 mass/simplex differs")
    libs["src.finite_student_outer_v2"].decode_rms_moments(current, 128, vars(transform))
    def readout(value):
        x, y, weight = libs["src.sweep_utils"].representative(value, transform, 128, "cuda")
        return x, y, t.full_like(weight, 1 / 90)
    actual = _call(evidence, "current_and_legacy_readout_calls", readout, current)
    previous = _call(evidence, "current_and_legacy_readout_calls", readout, legacy)
    _require(all(t.equal(a, b) for a, b in zip(actual, previous)) and actual[0].dtype == actual[1].dtype == t.float32
             and actual[2].dtype == t.float64, "Actual representative/uniform readout differs")
    evidence.update(actual_M0_bitwise_equal_legacy0=True, actual_FP32_X_Q_and_FP64_uniform_bitwise_equal_legacy0=True,
                    original_config=config, RMS_diagnostic=_descriptor(libs, vars(transform)), RMS_input_dtype=str(h.dtype),
                    origin=dict(U=_descriptor(libs, u), V=_descriptor(libs, v), M0=_descriptor(libs, current),
                                readout=_descriptor(libs, actual), historical_factor_bytes_available=False))
    theta = _tensor(t, snapshot.get("theta"), (40, 129), t.float64, "legacy head").to(z)
    ce = libs["src.soft_ce_partition"]
    gradient = _call(evidence, "cached_head_gradient_diagnostics", ce.head_gradient,
                     moments.augmented(centers), labels, t.full_like(mass, 1 / 90), theta, .0001)
    maximum = float(gradient.abs().max())
    value, unused_rhs = _call(evidence, "cached_teacher_CE_diagnostics", ce.outer_value_gradient, z, q, theta, 2048, moments.augmented(z))
    del unused_rhs
    _require(snapshot.get("J_exact") is True and math.isfinite(maximum) and maximum <= 1e-7
             and math.isclose(maximum, snapshot["inner_grad_max"], abs_tol=1e-12, rel_tol=1e-12)
             and math.isfinite(value) and math.isclose(value, snapshot["teacher_ce"], abs_tol=1e-12, rel_tol=1e-12),
             "Original cached head/teacher CE diagnostic differs")
    evidence.update(cached_head_and_teacher_CE_passed=True, head_gradient_max=maximum, teacher_CE=value)
    return retained


def run(spec_path, spec_sha256, output_path, stop=lambda: False):
    started = time.monotonic()
    evidence = dict(passed=False, operation="run", spec_path=str(spec_path), spec_sha256=spec_sha256,
                    output_path=str(output_path), counts={}, success_counts=None, test_enabled=False,
                    historical_RMS_recovered=False, optimization_origin_persisted=False, Phi_numerical_consistency_verified=False,
                    legacy_endpoint=20, legacy_endpoint20_evaluated=False, large_gradient_backend_qualified=False,
                    old_BI_or_BK_qualified=False, Cora_BN_qualifies_Arxiv=False, efficacy_measured=False,
                    integrated_loader_H_partial_work="Internal partial work unknown until loader returns.", finalization_errors=[])
    repo = Path(__file__).resolve().parents[1]
    fixed_output = repo / "results/research_loop/Arxiv90_require_existing_RMS_P0_readonly_stageBT_v1/native_source_RMS_P0_preflight_v1.json"
    _require(str(output_path) == str(fixed_output) and not fixed_output.exists(), "Require absent fixed receipt path; no retry")
    libs = spec = science = buffers = None
    handler = timer = None
    primary = None
    try:
        _require(callable(stop), "stop must be callable")
        spec, science, repo = _load_spec(spec_path, spec_sha256)
        _require(str(output_path) == spec["output_path"], "Require fixed receipt path")
        evidence["counts"] = {key + suffix: 0 for key in science["counts_success"]
                              for suffix in ("_attempts", "_completed") if not (key == "integrated_loader_H_products" and suffix == "_attempts")}
        evidence.update(source=spec["source"], numerical_source=spec["numerical_source"], protocol=spec["protocol"],
                        scientific_preregistration=dict(path=str(repo / "results/proposals" / SCIENCE), sha256=SCIENCE_SHA),
                        family_protocol=spec["family_protocol"], candidate_protocol=spec["candidate_protocol"])
        handler, timer = signal.getsignal(signal.SIGALRM), signal.getitimer(signal.ITIMER_REAL)
        def expired(signum, frame):
            raise InterruptedError("Readonly preflight deadline reached")
        signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, max(1e-6, 300 - (time.monotonic() - started)))
        libs = _libraries()
        evidence["native_environment"] = _native(libs, science)
        _require(evidence["native_environment"]["versions"] == spec["numerical_source"]["versions"], "Actual runtime versions differ")
        def guard():
            if stop() or time.monotonic() - started >= 300:
                raise InterruptedError("Readonly preflight stopped/deadline")
            _precision(libs["torch"])
        evidence["_guard"] = guard
        buffers = _execute(spec, science, repo, libs, evidence)
        guard()
        _require(buffers is evidence.get("_retained"), "Retained source packet differs")
        for key, expected in science["counts_success"].items():
            observed = evidence["counts"].get(key + "_completed", 0)
            _require(observed == expected, "Successful interface count differs: " + key)
        evidence["success_counts"] = dict(science["counts_success"])
        evidence["passed"] = True
    except BaseException as exc:
        primary = exc
        evidence["primary_error"] = dict(type=type(exc).__name__, message=str(exc))
        evidence["partial_internal_work_qualified"] = False
    finally:
        evidence.pop("_guard", None)
        retained = evidence.pop("_retained", None)
        if retained is not None and "native_buffers_before" in evidence:
            try:
                evidence["native_buffers_after"] = _descriptor(libs, retained)
                evidence["source_buffers_unchanged"] = evidence["native_buffers_before"] == evidence["native_buffers_after"]
                evidence["source_buffer_preservation_scope"] = "Retained arrays compared even on later failure."
                _require(evidence["source_buffers_unchanged"], "Retained source/origin buffers changed")
            except BaseException as exc:
                evidence["passed"] = False
                evidence["finalization_errors"].append(dict(stage="retained_buffers", type=type(exc).__name__, message=str(exc)))
                if primary is None:
                    primary = exc
        else:
            evidence["source_buffers_unchanged"] = None
            evidence["source_buffer_preservation_scope"] = "Unknown: failure before retained-array baseline capture."
        if handler is not None:
            try:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, handler)
                if timer[0] > 0:
                    signal.setitimer(signal.ITIMER_REAL, max(1e-6, timer[0] - (time.monotonic() - started)), timer[1])
            except BaseException as exc:
                evidence["passed"] = False
                evidence["finalization_errors"].append(dict(stage="timer", type=type(exc).__name__, message=str(exc)))
                if primary is None:
                    primary = exc
        for name, action in (("preservation", lambda: _preserve(spec, Path(spec_path), spec_sha256, science, repo)),
                             ("precision", lambda: _precision(libs["torch"]))):
            if spec is None or (name == "precision" and libs is None):
                continue
            try:
                action()
                if name == "preservation":
                    evidence["source_assets_spec_protocol_unchanged"] = True
            except BaseException as exc:
                evidence["passed"] = False
                evidence["finalization_errors"].append(dict(stage=name, type=type(exc).__name__, message=str(exc)))
                if primary is None:
                    primary = exc
        if libs is not None and libs["torch"].cuda.is_initialized():
            try:
                libs["torch"].cuda.synchronize()
                evidence.update(peak_allocated_bytes=libs["torch"].cuda.max_memory_allocated(),
                                peak_reserved_bytes=libs["torch"].cuda.max_memory_reserved())
            except BaseException as exc:
                evidence["passed"] = False
                evidence["finalization_errors"].append(dict(stage="memory", type=type(exc).__name__, message=str(exc)))
                if primary is None:
                    primary = exc
        evidence["elapsed_seconds"] = time.monotonic() - started
        if evidence["elapsed_seconds"] > 300:
            evidence["passed"] = False
            if primary is None:
                primary = TimeoutError("Readonly preflight exceeded300s")
        if not evidence["passed"]:
            evidence["success_counts"] = None
        if primary is not None and "primary_error" not in evidence:
            evidence["primary_error"] = dict(type=type(primary).__name__, message=str(primary))
        path = fixed_output
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as stream:
            json.dump(evidence, stream, indent=2, allow_nan=False)
    if primary is not None:
        raise primary
    return dict(evidence, evidence_path=str(path), evidence_sha256=_sha(path), validation_only=True)
