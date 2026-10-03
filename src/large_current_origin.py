"""Capture a genuine current Arxiv90 origin and replay its persisted RMS readonly.

No optimization, historical-origin recovery, source-gradient qualification or fits.
Scientific/review pins bind the separately frozen current-origin root protocol.
"""
import json
import math
import os
import platform
import signal
import time
from pathlib import Path

from src import large_source_preflight as original

SCIENCE = "Arxiv90_genuine_current_origin_scientific_stageBW_v1.json"
SCIENCE_SHA = "e05e896a560a3e9dbcd62eb58ae3fa6f5bf352cb652df736528fb9207bdf7ecc"
REVIEW = "independent_scientific_review_stageBW_v1.json"
REVIEW_SHA = "ac59b66490a320c62ec810cc5b2648f744ac6923ae515ad7c19626cac0ea1f4e"
STAGE = "Arxiv90_genuine_current_origin_stageBW_v1"
FIELDS = {"schema", "fixed", "source", "numerical_source", "python_version", "files_sha256",
          "artifacts_sha256", "scientific_preregistration", "family_protocol", "candidate_protocol",
          "output_root", "operation_outputs", "origin_path"}
FIXED = dict(nodes=169343, features=128, classes=40, basis=512, dataset="arxiv", cells=90,
             temperature=.3, train_target_mix=0, rank=16, factor_seed=0, mixing=.05,
             chunk=2048, RMS_kind="rms", RMS_eps=1e-12, legacy_endpoint=20)
ARRAY_KEYS = {"X", "original_CSR", "graph_y_identity", "train_mask", "validation_mask", "saved_H",
              "H_double", "Q", "hard", "z", "transform", "U0", "V0", "M0", "readout", "theta0"}
TRANSFORM_KEYS = {"center", "matrix", "output_center", "scale", "kind", "eps"}
_require, _sha, _seal = original._require, original._sha, original._seal
_fingerprint, _source, numerical_source = original._fingerprint, original._source, original.numerical_source
_checked, _libraries = original._checked, original._libraries
_count, _call, _precision, _native = original._count, original._call, original._precision, original._native
_descriptor, _tensor, _expected_core_config = original._descriptor, original._tensor, original._expected_core_config


def _references(value):
    if isinstance(value, dict):
        if set(value) >= {"path", "sha256"}:
            _checked({value["path"]: value["sha256"]})
        for child in value.values():
            _references(child)
    elif isinstance(value, list):
        for child in value:
            _references(child)


def _required_paths(science, repo):
    family, candidate = Path(science["family"]["geometry_root"]), Path(science["family"]["candidate_root"])
    return {str(family / name) for name in ("protocol.json", "propagated_H.pt", "feature_map.pt", "teacher.pt", "phi.meta.json", "phi.npy")} | {
        str(candidate / name) for name in ("protocol.json", "initial_assignment.pt", "condensation/resume.pt",
                                         "condensation/checkpoints/step_000000.pt", "condensation/checkpoints/step_000020.pt")} | {
        str(repo / "data/ogbn-arxiv/raw" / name) for name in ("adj_full.npz", "feats.npy", "role.json", "class_map.json")}


def _preserve(spec, path, checksum, science, repo):
    _require(_sha(path) == checksum and _source(repo) == spec["source"]
             and numerical_source(repo) == spec["numerical_source"]
             and platform.python_version() == spec["python_version"], "Spec/source/Git/numerical/Python changed")
    for pins in (spec["files_sha256"], spec["artifacts_sha256"],
                 {str(repo / "results/proposals" / SCIENCE): SCIENCE_SHA,
                  str(repo / "results/proposals" / REVIEW): REVIEW_SHA},
                 {str(repo / p): h for p, h in science["source_before"]["files"].items() if p != "src/research_loop.py"},
                 {str(repo / p): h for p, h in science["old_tests_sha256"].items()}):
        _checked(pins)
    _references(science.get("parents", {}))


def _controls(spec, science, repo):
    _require(isinstance(spec, dict) and set(spec) == FIELDS and type(spec["schema"]) is int and spec["schema"] == 1,
             "Require exact thirteen-field spec")
    contract = science["implementation_contract"]
    for key, expected in (("fixed", science["fixed"]), ("files_sha256", science["original_files_sha256"]),
                          ("scientific_preregistration", dict(path=str(repo / "results/proposals" / SCIENCE), sha256=SCIENCE_SHA)),
                          ("family_protocol", science["family"]["teacher_protocol"]),
                          ("candidate_protocol", science["family"]["candidate_protocol"]),
                          ("output_root", contract["output_root"]), ("operation_outputs", contract["operation_outputs"]),
                          ("origin_path", contract["origin_path"])):
        _require(_seal(spec[key]) == _seal(expected), "Changed fixed field: " + key)
    _require(all(_seal(science["fixed"].get(k)) == _seal(v) for k, v in FIXED.items()), "Changed source controls")
    owned = {repo / "src/large_current_origin.py", repo / "src/research_loop.py", repo / "tests/test_large_current_origin.py"}
    pins = spec["artifacts_sha256"]
    _require(isinstance(pins, dict) and all(type(p) is str and Path(p).is_absolute()
             and (Path(p).resolve().is_relative_to(repo / "results") or Path(p).resolve() in owned) for p in pins),
             "Require results/exact three owned artifacts")
    _require(_required_paths(science, repo) <= set(spec["files_sha256"]) | set(pins),
             "Require every original dataset/source/reference asset before numerical imports")
    _require(isinstance(spec["source"], dict) and set(spec["source"].get("files", {})) ==
             set(science["source_before"]["files"]) | {"src/large_current_origin.py"}, "Require promoted source85")
    _require(spec["numerical_source"].get("versions", {}).get("torch") == science["native_policy"]["torch"],
             "Require frozen native Torch literal")
    root = Path(spec["output_root"]).resolve()
    _require(root.is_relative_to(repo / "results/research_loop") and set(spec["operation_outputs"]) == {"capture", "replay"}
             and all(Path(p).is_absolute() and Path(p).resolve().parent == root for p in spec["operation_outputs"].values())
             and Path(spec["origin_path"]).is_absolute() and Path(spec["origin_path"]).resolve().parent == root,
             "Changed own output namespace")


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
    return spec, science, repo


def _capture_binding(spec, spec_path, spec_sha256, checksum):
    path = Path(spec["operation_outputs"]["capture"])
    _checked({str(path): checksum})
    receipt = json.loads(path.read_text())
    _require(receipt.get("passed") is True and receipt.get("operation") == "capture"
             and receipt.get("current_origin_capture_passed") is True
             and receipt.get("source_assets_spec_science_unchanged") is True
             and receipt.get("source_buffers_unchanged") is True and receipt.get("origin_buffers_unchanged") is True
             and receipt.get("spec_path") == str(spec_path) and receipt.get("spec_sha256") == spec_sha256
             and receipt.get("source") == spec["source"] and receipt.get("numerical_source") == spec["numerical_source"]
             and receipt.get("scientific_preregistration") == spec["scientific_preregistration"]
             and receipt.get("origin_cache_path") == spec["origin_path"], "Require passed exact own capture receipt")
    for key in ("origin_cache_sha256", "origin_content_sha256", "origin_context_sha256"):
        _require(type(receipt.get(key)) is str and len(receipt[key]) == 64, "Missing capture seal: " + key)
    _checked({spec["origin_path"]: receipt["origin_cache_sha256"]})
    return receipt


def _validate_transform_state(libs, state):
    t = libs["torch"]
    _require(isinstance(state, dict) and set(state) == TRANSFORM_KEYS and state["matrix"] is None
             and type(state["kind"]) is str and state["kind"] == "rms"
             and type(state["eps"]) is float and state["eps"] == 1e-12, "Malformed persisted RMS state")
    for name, shape in (("center", (128,)), ("output_center", (128,)), ("scale", ())):
        _tensor(t, state[name], shape, t.float64, "persisted RMS " + name)
    _require(float(state["scale"]) > 0, "Persisted RMS scale must be positive")


def _persisted_transform(libs, state, evidence):
    _validate_transform_state(libs, state)
    t = libs["torch"]
    copied = {key: value.to(device="cuda") if t.is_tensor(value) else value for key, value in state.items()}
    return _call(evidence, "persisted_transform_constructors", libs["src.transforms"].FeatureTransform, **copied)


def _validate_arrays(libs, arrays, cpu_only=False):
    t = libs["torch"]
    _require(isinstance(arrays, dict) and set(arrays) == ARRAY_KEYS, "Malformed current-origin arrays")
    shapes = {"X": ((169343, 128), t.float32), "graph_y_identity": ((169343,), t.int64),
              "train_mask": ((169343,), t.bool), "validation_mask": ((169343,), t.bool),
              "saved_H": ((169343, 128), t.float32), "H_double": ((169343, 128), t.float64),
              "Q": ((169343, 40), t.float64), "hard": ((169343,), t.int64), "z": ((169343, 128), t.float64),
              "U0": ((169343, 16), t.float32), "V0": ((90, 16), t.float32),
              "M0": ((90, 169), t.float64), "theta0": ((40, 129), t.float64)}
    for name, (shape, dtype) in shapes.items():
        _tensor(t, arrays[name], shape, dtype, "current-origin " + name)
        _require(arrays[name].layout == t.strided, "Malformed dense origin layout")
    csr = arrays["original_CSR"]
    _require(t.is_tensor(csr) and csr.layout == t.sparse_csr and csr.dtype == t.float32
             and tuple(csr.shape) == (169343, 169343) and bool(t.isfinite(csr.values()).all()), "Malformed persisted original CSR")
    crow, col, values = csr.crow_indices(), csr.col_indices(), csr.values()
    _require(crow.dtype in (t.int32, t.int64) and col.dtype in (t.int32, t.int64)
             and tuple(crow.shape) == (169344,) and col.ndim == values.ndim == 1
             and len(col) == len(values), "Malformed original CSR index parts")
    _validate_transform_state(libs, arrays["transform"])
    readout = arrays["readout"]
    _require(isinstance(readout, (tuple, list)) and len(readout) == 3, "Malformed current-origin readout")
    for value, shape, dtype in zip(readout, ((90, 128), (90, 40), (90,)), (t.float32, t.float32, t.float64)):
        _tensor(t, value, shape, dtype, "current-origin readout")
    if cpu_only:
        def check_cpu(value):
            if t.is_tensor(value):
                _require(value.device.type == "cpu", "Persisted origin arrays must be CPU")
            elif isinstance(value, dict):
                for child in value.values():
                    check_cpu(child)
            elif isinstance(value, (tuple, list)):
                for child in value:
                    check_cpu(child)
        check_cpu(arrays)


def _context(spec, path, checksum, evidence):
    return dict(schema=1, spec=dict(path=str(path), sha256=checksum), scientific_preregistration=spec["scientific_preregistration"],
                source=spec["source"], numerical_source=spec["numerical_source"], python_version=spec["python_version"],
                fixed=spec["fixed"], family_protocol=spec["family_protocol"], candidate_protocol=spec["candidate_protocol"],
                original_files_sha256=spec["files_sha256"], original_config=evidence["original_config"],
                source_data_digest=evidence["source_data_digest"], shared_H_source_digest=evidence["shared_H_source_digest"],
                saved_H_identity=evidence["saved_H_identity"], map_identity=evidence["map_identity"],
                teacher_saved_converged=evidence["teacher_saved_converged"], RMS=evidence["RMS_diagnostic"],
                native_environment=evidence["native_environment"], native_source_buffers=evidence["native_buffers_before"],
                historical_RMS_recovered=False, historical_factor_bytes_recovered=False,
                heldout_labels="Identity hashes only; no heldout loss/accuracy/selection.")


def _copy_cpu(libs, value):
    t = libs["torch"]
    if t.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _copy_cpu(libs, child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_copy_cpu(libs, child) for child in value]
    return value


def _packet(libs, arrays, context, evidence):
    copied = _call(evidence, "origin_CPU_copy_calls", _copy_cpu, libs, arrays)
    _validate_arrays(libs, copied, cpu_only=True)
    _require(_descriptor(libs, copied) == _descriptor(libs, arrays), "Origin CPU copy changed bytes")
    result = dict(schema=1, context=context, arrays=copied, descriptors=_descriptor(libs, copied))
    result["content_sha256"] = _seal(dict(context=context, descriptors=result["descriptors"]))
    return result


def _validate_packet(libs, packet, binding):
    _require(isinstance(packet, dict) and set(packet) == {"schema", "context", "arrays", "descriptors", "content_sha256"}
             and type(packet["schema"]) is int and packet["schema"] == 1, "Malformed current-origin packet")
    _validate_arrays(libs, packet["arrays"], cpu_only=True)
    actual = _descriptor(libs, packet["arrays"])
    _require(actual == packet["descriptors"] and packet["content_sha256"] ==
             _seal(dict(context=packet["context"], descriptors=actual)) == binding["origin_content_sha256"]
             and _seal(packet["context"]) == binding["origin_context_sha256"], "Current-origin packet seals differ")


def _write_origin(libs, packet, path, evidence):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        libs["torch"].save(packet, stream)
        stream.flush()
        os.fsync(stream.fileno())
    evidence.update(origin_cache_path=str(path), origin_cache_sha256=_sha(path),
                    origin_content_sha256=packet["content_sha256"], origin_context_sha256=_seal(packet["context"]))

def _execute(operation, spec, science, repo, libs, evidence, packet=None):
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
    saved_h = h
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
    if operation == "capture":
        z, transform = _call(evidence, "RMS_fit_transform_calls", libs["src.transforms"].fit_transform,
                             h, kind="rms", eps=1e-12)
    else:
        transform = _persisted_transform(libs, packet["arrays"]["transform"], evidence)
        z = _call(evidence, "persisted_transform_apply_calls", transform, h)
        _require(_descriptor(libs, z) == packet["descriptors"]["z"], "Persisted transform replay z differs")
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
    material = _call(evidence, "material_constructor_calls", moments.make_material, z, q)
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
    arrays = dict(X=graph["x"], original_CSR=graph["adj"], graph_y_identity=graph["y"],
                  train_mask=train, validation_mask=validation[1], saved_H=saved_h, H_double=h,
                  Q=q, hard=hard, z=z, transform=vars(transform), U0=u, V0=v, M0=current,
                  readout=actual, theta0=theta)
    _validate_arrays(libs, arrays)
    evidence["_origin_arrays"] = arrays
    evidence["origin_buffers_before"] = _descriptor(libs, arrays)
    return arrays

def run(operation, spec_path, spec_sha256, output_path, capture_evidence_sha256=None, stop=lambda: False):
    _require(type(operation) is str and operation in {"capture", "replay"}, "Require capture or replay")
    repo = Path(__file__).resolve().parents[1]
    fixed_output = repo / "results/research_loop" / STAGE / (operation + "_receipt_v1.json")
    _require(str(output_path) == str(fixed_output) and not fixed_output.exists(), "Require absent own receipt; no retry")
    started = time.monotonic()
    evidence = dict(passed=False, operation=operation, spec_path=str(spec_path), spec_sha256=spec_sha256,
                    output_path=str(output_path), counts={}, success_counts=None, test_enabled=False,
                    current_origin_capture_passed=False, current_origin_readonly_replay_passed=False,
                    current_origin_independent_replay_qualified=False, historical_RMS_recovered=False,
                    historical_factor_bytes_recovered=False, optimization_origin_persisted=False,
                    Phi_numerical_consistency_verified=False, legacy_endpoint=20, legacy_endpoint20_evaluated=False,
                    large_gradient_backend_qualified=False, old_BI_or_BK_qualified=False,
                    Cora_BN_qualifies_Arxiv=False, efficacy_measured=False, finalization_errors=[],
                    integrated_loader_H_partial_work="Internal partial work unknown until loader returns.")
    libs = spec = science = arrays = binding = packet = None
    handler = timer = None
    primary = None
    try:
        _require(callable(stop), "stop must be callable")
        spec, science, repo = _load_spec(spec_path, spec_sha256)
        _require(str(output_path) == spec["operation_outputs"][operation], "Require fixed operation output")
        origin_path = Path(spec["origin_path"])
        if operation == "capture":
            _require(capture_evidence_sha256 is None and not origin_path.exists()
                     and not Path(spec["operation_outputs"]["replay"]).exists(), "Fresh capture only; preserve any partial namespace")
        else:
            _require(type(capture_evidence_sha256) is str and len(capture_evidence_sha256) == 64,
                     "Replay requires exact passed capture receipt SHA")
            binding = _capture_binding(spec, spec_path, spec_sha256, capture_evidence_sha256)
            evidence.update(capture_evidence_path=spec["operation_outputs"]["capture"],
                            capture_evidence_sha256=capture_evidence_sha256,
                            origin_cache_path=spec["origin_path"], origin_cache_sha256=binding["origin_cache_sha256"],
                            origin_content_sha256=binding["origin_content_sha256"],
                            origin_context_sha256=binding["origin_context_sha256"])
        expected = science["counts_contract"]["per_" + operation + "_success"]
        evidence["counts"] = {key + suffix: 0 for key in expected for suffix in ("_attempts", "_completed")
                              if not (key == "integrated_loader_H_products" and suffix == "_attempts")}
        evidence.update(source=spec["source"], numerical_source=spec["numerical_source"], python_version=spec["python_version"],
                        scientific_preregistration=spec["scientific_preregistration"],
                        family_protocol=spec["family_protocol"], candidate_protocol=spec["candidate_protocol"],
                        fixed=spec["fixed"])
        handler, timer = signal.getsignal(signal.SIGALRM), signal.getitimer(signal.ITIMER_REAL)
        def expired(signum, frame):
            raise InterruptedError("Current-origin deadline reached")
        signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, max(1e-6, 300 - (time.monotonic() - started)))
        libs = _libraries()
        evidence["native_environment"] = _native(libs, science)
        evidence["native_environment"]["source_backend"] = "original_Arxiv_CSR_current_origin_readonly_compatibility_only"
        _require(evidence["native_environment"]["versions"] == spec["numerical_source"]["versions"], "Actual runtime versions differ")
        def guard():
            if stop() or time.monotonic() - started >= 300:
                raise InterruptedError("Current-origin stopped/deadline")
            _precision(libs["torch"])
        evidence["_guard"] = guard
        if operation == "replay":
            packet = _call(evidence, "origin_checkpoint_reads", libs["torch"].load, origin_path,
                           map_location="cpu", weights_only=False)
            _validate_packet(libs, packet, binding)
            evidence["_packet_arrays"] = packet["arrays"]
            evidence["persisted_packet_arrays_before"] = _descriptor(libs, packet["arrays"])
        arrays = _execute(operation, spec, science, repo, libs, evidence, packet)
        context = _context(spec, spec_path, spec_sha256, evidence)
        evidence["origin_context"] = context
        if operation == "capture":
            packet = _packet(libs, arrays, context, evidence)
            _call(evidence, "origin_checkpoint_writes", _write_origin, libs, packet, origin_path, evidence)
            evidence["current_origin_persistence_bytes_written"] = True
        else:
            _require(_seal(context) == binding["origin_context_sha256"]
                     and _descriptor(libs, arrays) == packet["descriptors"], "Actual current-source origin replay differs")
            evidence["persisted_transform_and_full_origin_exact_replay_passed"] = True
        guard()
        for key, value in expected.items():
            _require(evidence["counts"].get(key + "_completed", 0) == value, "Successful interface count differs: " + key)
        evidence["success_counts"] = dict(expected)
        evidence["passed"] = True
    except BaseException as exc:
        primary = exc
        evidence["primary_error"] = dict(type=type(exc).__name__, message=str(exc))
        evidence["partial_internal_work_qualified"] = False
    finally:
        evidence.pop("_guard", None)
        for transient, before, after, flag in (
                ("_retained", "native_buffers_before", "native_buffers_after", "source_buffers_unchanged"),
                ("_origin_arrays", "origin_buffers_before", "origin_buffers_after", "origin_buffers_unchanged"),
                ("_packet_arrays", "persisted_packet_arrays_before", "persisted_packet_arrays_after", "persisted_packet_arrays_unchanged")):
            retained = evidence.pop(transient, None)
            if retained is None or before not in evidence:
                evidence[flag] = None
                continue
            try:
                evidence[after] = _descriptor(libs, retained)
                evidence[flag] = evidence[before] == evidence[after]
                _require(evidence[flag], "Retained arrays changed: " + flag)
            except BaseException as exc:
                evidence["passed"] = False
                evidence["finalization_errors"].append(dict(stage=flag, type=type(exc).__name__, message=str(exc)))
                if primary is None:
                    primary = exc
        evidence["buffer_preservation_scope"] = "Retained baselines checked on later failure; pre-capture scope unknown."
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
                    evidence["source_assets_spec_science_unchanged"] = True
            except BaseException as exc:
                evidence["passed"] = False
                evidence["finalization_errors"].append(dict(stage=name, type=type(exc).__name__, message=str(exc)))
                if primary is None:
                    primary = exc
        if spec is not None:
            try:
                path = Path(spec["origin_path"])
                if path.exists():
                    actual_sha = _sha(path)
                    evidence["origin_cache_observed_sha256_finally"] = actual_sha
                    if "origin_cache_sha256" in evidence:
                        _require(actual_sha == evidence["origin_cache_sha256"], "New origin file changed")
                        evidence["origin_cache_bytes_unchanged"] = True
                    else:
                        evidence["unqualified_partial_origin_path"] = str(path)
                        evidence["unqualified_partial_origin_sha256"] = actual_sha
                if binding is not None:
                    _checked({spec["operation_outputs"]["capture"]: capture_evidence_sha256})
            except BaseException as exc:
                evidence["passed"] = False
                evidence["finalization_errors"].append(dict(stage="origin_preservation", type=type(exc).__name__, message=str(exc)))
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
                primary = TimeoutError("Current-origin exceeded300s")
        if not evidence["passed"]:
            evidence["success_counts"] = None
        evidence["current_origin_capture_passed"] = evidence["passed"] and operation == "capture"
        evidence["current_origin_readonly_replay_passed"] = evidence["passed"] and operation == "replay"
        evidence["current_origin_independent_replay_qualified"] = evidence["current_origin_readonly_replay_passed"]
        evidence["optimization_origin_persisted"] = evidence["passed"]
        if primary is not None and "primary_error" not in evidence:
            evidence["primary_error"] = dict(type=type(primary).__name__, message=str(primary))
        fixed_output.parent.mkdir(parents=True, exist_ok=True)
        with fixed_output.open("x") as stream:
            json.dump(evidence, stream, indent=2, allow_nan=False)
    if primary is not None:
        raise primary
    return dict(evidence, evidence_path=str(fixed_output), evidence_sha256=_sha(fixed_output), validation_only=True)
