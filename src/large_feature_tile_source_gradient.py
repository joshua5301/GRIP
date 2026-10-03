"""Quarantined Arxiv anchor0 tiled-gradient resource/repeat qualification.

Only an accepted CPU origin is read. Historical BW provenance is kept separate
from this additive execution manifest. Resource/repeat is not accuracy proof.
"""
import importlib
import json
import math
import os
import platform
import time
import traceback
from pathlib import Path

from src import large_current_origin as bw
from src import large_source_preflight as original

SCIENCE = "Arxiv90_feature_band16_native_anchor0_resource_repeat_scientific_stageBY_v1.json"
SCIENCE_SHA = "fee4524fdd7a417ff7da376c35039005c1bd6362204f385bb83614533eb2d587"
REVIEW = "independent_scientific_review_stageBY_v1.json"
REVIEW_SHA = "c5e0a6faea6256efad4614e699f370870733343e917941924c997025d099a669"
FIELDS = {"schema", "fixed", "source", "numerical_source", "python_version", "files_sha256",
          "scientific_preregistration", "artifacts_sha256", "output_root", "pass_outputs", "target_outputs", "comparison_policy"}
BACKEND = "original_Arxiv_CSR_feature_band16_stored_order_segment_SUM"
_require, _sha, _seal = original._require, original._sha, original._seal
_source, numerical_source, _checked = original._source, original.numerical_source, original._checked
_descriptor, _native, _precision = bw._descriptor, original._native, original._precision
_validate_packet = bw._validate_packet


def _controls(spec, science, repo):
    _require(isinstance(spec, dict) and set(spec) == FIELDS and type(spec["schema"]) is int and spec["schema"] == 1,
             "Require exact twelve-field spec")
    contract = science["implementation_contract"]
    for key, expected in (("fixed", science["fixed"]), ("files_sha256", science["original_files_sha256"]),
                          ("output_root", contract["output_root"]), ("pass_outputs", contract["pass_outputs"]),
                          ("target_outputs", contract["target_outputs"]), ("comparison_policy", science["comparison_policy"]),
                          ("scientific_preregistration", dict(path=str(repo / "results/proposals" / SCIENCE), sha256=SCIENCE_SHA))):
        _require(_seal(spec[key]) == _seal(expected), "Changed fixed spec field: " + key)
    owned = {(repo / p).resolve() for p in contract["owned_paths"]}
    pins = spec["artifacts_sha256"]
    _require(isinstance(pins, dict) and pins and all(type(p) is str and Path(p).is_absolute()
             and (Path(p).resolve().is_relative_to(repo / "results") or Path(p).resolve() in owned) for p in pins),
             "Require results or exact three owned artifacts")
    _require(isinstance(spec["source"], dict) and set(spec["source"].get("files", {})) ==
             set(science["source_before"]["files"]) | {"src/large_feature_tile_source_gradient.py"}, "Require promoted source87")
    _require(spec["numerical_source"].get("versions") == science["native_policy"]["versions"]
             and spec["python_version"] == science["native_policy"]["python_version"], "Changed fixed versions/Python")


def _preserve(spec, path, checksum, science, repo):
    _require(_sha(path) == checksum and _source(repo) == spec["source"]
             and numerical_source(repo) == spec["numerical_source"]
             and platform.python_version() == spec["python_version"], "Spec/source/Git/numerical/Python changed")
    for pins in (spec["files_sha256"], spec["artifacts_sha256"],
                 {str(repo / "results/proposals" / SCIENCE): SCIENCE_SHA, str(repo / "results/proposals" / REVIEW): REVIEW_SHA},
                 {ref["path"]: ref["sha256"] for ref in science["parents"].values()},
                 {str(repo / p): h for p, h in science["source_before"]["files"].items() if p != "src/research_loop.py"},
                 {str(repo / p): h for p, h in science["protected_tests_before"].items()}):
        _checked(pins)


def _load_spec(path, checksum):
    repo, path = Path(__file__).resolve().parents[1], Path(path).resolve()
    _require(type(checksum) is str and len(checksum) == 64 and path.is_file() and _sha(path) == checksum,
             "Require pinned existing spec")
    science_path = repo / "results/proposals" / SCIENCE
    _checked({str(science_path): SCIENCE_SHA})
    science = json.loads(science_path.read_text())
    _require(str(path) == science["implementation_contract"]["spec_path"], "Wrong prospective spec path")
    spec = json.loads(path.read_text())
    _controls(spec, science, repo)
    _preserve(spec, path, checksum, science, repo)
    return spec, science, repo


def _request(pass_index, previous, output, spec):
    _require(type(pass_index) is int and pass_index in (1, 2), "Require pass1 or pass2")
    _require((pass_index == 1 and previous is None) or (pass_index == 2 and type(previous) is str and len(previous) == 64),
             "Wrong previous receipt protocol")
    output, target = Path(output).resolve(), Path(spec["target_outputs"][str(pass_index)]).resolve()
    _require(str(output) == spec["pass_outputs"][str(pass_index)] and output.parent == target.parent
             and not output.exists() and not output.parent.exists(), "Require absent exclusive pass namespace")
    return output, target


def _libraries():
    # Only immutable helpers are imported; no dataset loader or fit is invoked.
    return {name: importlib.import_module(name) for name in
            ("torch", "torch_geometric", "numpy", "scipy", "sklearn", "src.shared_features",
             "src.csr_feature_tile_adjoint", "src.csr_explicit_adjoint", "src.citation_gradient_probe")}


def _guard(libs, started, stop):
    _require(not stop() and time.monotonic() - started < 300, "Stopped/deadline reached")
    _precision(libs["torch"])


def _finite_tree(libs, value, cpu_only=False):
    t = libs["torch"]
    if t.is_tensor(value):
        _require(value.layout in (t.strided, t.sparse_csr), "Unsupported full-cache layout")
        _require(not cpu_only or value.device.type == "cpu", "Raw cache must be CPU only")
        array = value if value.layout == t.strided else value.values()
        _require(not value.requires_grad and bool(t.isfinite(array).all()), "Nonfinite/unfrozen complete raw tensor")
    elif isinstance(value, dict):
        for child in value.values():
            _finite_tree(libs, child, cpu_only)
    elif isinstance(value, (tuple, list)):
        for child in value:
            _finite_tree(libs, child, cpu_only)


def _copy_cpu(libs, value):
    t = libs["torch"]
    if t.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _copy_cpu(libs, child) for key, child in value.items()}
    if isinstance(value, tuple):
        return tuple(_copy_cpu(libs, child) for child in value)
    if isinstance(value, list):
        return [_copy_cpu(libs, child) for child in value]
    return value


def _origin(libs, science, evidence, retained):
    invoke, csr = libs["src.csr_explicit_adjoint"]._invoke, libs["src.csr_explicit_adjoint"]._csr
    binding = science["origin_binding"]
    packet = invoke(evidence, "accepted_BW_origin_CPU_loads", libs["torch"].load,
                    binding["origin_path"], map_location="cpu", weights_only=False)
    retained["origin"] = packet
    evidence["origin_buffers_before"] = _descriptor(libs, packet["arrays"])
    invoke(evidence, "BW_full_packet_validation_calls", _validate_packet, libs, packet,
           dict(origin_content_sha256=binding["origin_content_sha256"], origin_context_sha256=binding["origin_context_sha256"]))
    _require(packet["descriptors"] == binding["full16_descriptors"], "All16 signed BW descriptors differ")
    context = packet["context"]
    capture = json.loads(Path(binding["capture_receipt"]["path"]).read_text())
    _require(context == capture["origin_context"] and context["source"] == binding["historical_source"]
             and context["numerical_source"] == binding["historical_numerical_source"]
             and context["python_version"] == binding["historical_python_version"]
             and context["spec"] == binding["historical_spec"], "Historical BW origin context differs")
    csr(packet["arrays"]["original_CSR"])
    evidence.update(BW_origin=dict(path=binding["origin_path"], file_sha256=binding["origin_file_sha256"],
                    content_sha256=binding["origin_content_sha256"], context_sha256=binding["origin_context_sha256"],
                    historical_source=context["source"], historical_numerical_source=context["numerical_source"],
                    historical_python_version=context["python_version"], historical_spec=context["spec"],
                    descriptors=packet["descriptors"]), accepted_BW_origin_binding_passed=True)
    return packet["arrays"]


def _collect(libs, arrays, science, evidence, retained):
    t, explicit, helper = libs["torch"], libs["src.csr_explicit_adjoint"], libs["src.csr_feature_tile_adjoint"]
    inputs = {key: explicit._invoke(evidence, "source_X_S_Q_CUDA_transfer_calls", arrays[key].to, device="cuda")
              for key in ("X", "original_CSR", "Q")}
    retained["inputs"] = inputs
    evidence["native_source_buffers_before"] = _descriptor(libs, inputs)
    _require(evidence["native_source_buffers_before"] == {k: science["origin_binding"]["full16_descriptors"][k] for k in inputs},
             "Native original X/CSR/Q bits differ")
    anchor = explicit._invoke(evidence, "private_GEOM_anchor_factory_calls", libs["src.citation_gradient_probe"].anchor_initial,
                              128, 40, 256, 0, dtype=t.float32, device="cuda")
    retained["anchor"] = anchor
    evidence["anchor_digest_before"] = _descriptor(libs, anchor)
    explicit._begin(evidence, "source_GCN_gradient_targets_completed")
    result = explicit._invoke(evidence, "explicit_source_parameter_gradient_assemblies", helper.gradient_boundaries,
                              anchor, inputs["X"], inputs["original_CSR"], inputs["Q"], evidence)
    packet = explicit._invoke(evidence, "source_target_packet_assemblies", helper.target_packet, result, evidence)
    explicit._end(evidence, "source_GCN_gradient_targets_completed")
    common = dict(source_buffers={k: arrays[k] for k in inputs}, S=arrays["original_CSR"], T=result["transpose"],
                  anchor=anchor, boundaries=result["boundaries"], CE=result["loss"], target=packet)
    retained["common"] = common
    _common_contract(libs, common, science)
    _finite_tree(libs, common)
    evidence["common_numerical_digest"] = _descriptor(libs, common)
    evidence["common_numerical_seal"] = _seal(evidence["common_numerical_digest"])
    _require(_descriptor(libs, result["boundaries"]["logp"]) == _descriptor(libs, result["boundaries"]["full_logp"])
             and _descriptor(libs, result["loss"]) == _descriptor(libs, result["boundaries"]["full_CE"]),
             "Within-worker logp/CE descriptor bits differ")
    evidence.update(source_gradient_norm_values=[float(v) for v in packet["source_norms"]],
                    source_gradient_delta_values=[float(v) for v in packet["delta"]],
                    all_four_source_norms_deltas_finite_positive=True,
                    within_worker_full_logp_CE_bitwise_equal=True, complete_common_finite_passed=True)
    return common


def _common_contract(libs, common, science):
    t, shape = libs["torch"], science["full_boundary_gate"]["shapes"]
    _require(isinstance(common, dict) and set(common) == {"source_buffers", "S", "T", "anchor", "boundaries", "CE", "target"},
             "Incomplete common fields")
    boundaries = common["boundaries"]
    _require(set(boundaries) == set(shape["hidden_FP32"] + shape["class_FP32"]) | {"mask", "full_CE"}, "Incomplete full boundaries")
    for keys, dimensions in ((shape["hidden_FP32"], shape["hidden_shape"]), (shape["class_FP32"], shape["class_shape"])):
        for key in keys:
            original._tensor(t, boundaries[key], dimensions, t.float32, key)
    original._tensor(t, boundaries["mask"], shape["mask_bool_shape"], t.bool, "mask")
    for value in (common["CE"], boundaries["full_CE"]):
        original._tensor(t, value, (), t.float64, "CE")
    params = ((256, 128), (256,), (40, 256), (40,))
    _require(set(common["target"]) == {"blocks", "rho", "gradients", "source_norms", "delta"}
             and all(len(common["target"][key]) == 4 for key in ("gradients", "source_norms", "delta"))
             and len(common["anchor"]) == 4 and common["target"]["rho"] == .001
             and tuple(common["target"]["blocks"]) == ("W1", "b1", "W2", "b2"), "Wrong target/anchor blocks")
    for parameter, gradient, dimensions, norm, delta in zip(common["anchor"], common["target"]["gradients"], params,
            common["target"]["source_norms"], common["target"]["delta"]):
        original._tensor(t, parameter, dimensions, t.float32, "anchor")
        original._tensor(t, gradient, dimensions, t.float32, "gradient")
        for scalar in (norm, delta):
            original._tensor(t, scalar, (), t.float64, "norm/delta")
            _require(float(scalar) > 0, "Nonpositive gradient norm/delta")


def _context(spec, evidence):
    return dict(schema=1, spec=dict(path=evidence["spec_path"], sha256=evidence["spec_sha256"]),
                scientific_preregistration=spec["scientific_preregistration"], source=spec["source"],
                numerical_source=spec["numerical_source"], python_version=spec["python_version"], fixed=spec["fixed"],
                BW_origin=evidence["BW_origin"], source_backend=BACKEND)


def _capture(libs, target, context, pass_index, common, evidence):
    explicit = libs["src.csr_explicit_adjoint"]
    cpu = explicit._invoke(evidence, "completed_result_CPU_copy_calls", _copy_cpu, libs, common)
    _finite_tree(libs, cpu, cpu_only=True)
    digest = _descriptor(libs, cpu)
    _require(digest == evidence["common_numerical_digest"], "CPU snapshot changed native bits")
    payload = dict(schema=1, pass_index=pass_index, context=context, common=cpu, common_numerical_digest=digest,
                   common_numerical_seal=_seal(digest))
    payload["content_sha256"] = _seal(dict(schema=1, pass_index=pass_index, context=context, common_numerical_digest=digest))
    def save():
        target.parent.mkdir(parents=True, exist_ok=False)
        with target.open("xb") as stream:
            libs["torch"].save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
    explicit._invoke(evidence, "raw_cache_writes", save)
    evidence.update(raw_cache_path=str(target), raw_cache_sha256=_sha(target), raw_cache_content_sha256=payload["content_sha256"],
                    raw_cache_context_sha256=_seal(context), raw_cache_is_qualification=False,
                    current_finite_CPU_raw_capture_completed=True)


def _success_counts(science, pass_index):
    contract = science["counts_contract"]
    return dict(contract["per_complete_worker_expected_helper_and_wrapper_interfaces"], **contract["zero_work_per_worker"],
                previous_raw_cache_CPU_loads=contract["pass_specific"]["previous_raw_cache_CPU_loads"][str(pass_index)])


def _previous(spec, science, evidence, checksum):
    path = Path(spec["pass_outputs"]["1"])
    _checked({str(path): checksum})
    first = json.loads(path.read_text())
    for key in ("source", "numerical_source", "python_version", "scientific_preregistration", "spec_path", "spec_sha256", "native_environment"):
        _require(first.get(key) == evidence[key], "Previous provenance/environment differs: " + key)
    _require(first.get("passed") is True and type(first.get("pass_index")) is int and first["pass_index"] == 1
             and first.get("operation") == "prepare_source_gradient" and first.get("source_backend") == BACKEND
             and first.get("source_assets_spec_science_unchanged") is True
             and first.get("local_resource_finite_passed") is True
             and first.get("within_anchor0_feature_band16_exact_repeat_passed") is False
             and first.get("anchor0_resource_reproducibility_qualified") is False
             and first.get("actual_source_gradient_accuracy_qualified") is False
             and first.get("production_alignment_target_consumption_allowed") is False
             and first.get("success_counts") == _success_counts(science, 1), "First local receipt not accepted")
    target = Path(spec["target_outputs"]["1"])
    _require(first.get("raw_cache_path") == str(target), "Wrong previous cache path")
    _checked({str(target): first["raw_cache_sha256"]})
    return first, path, target


def _repeat(libs, spec, science, evidence, previous):
    first, path, target = _previous(spec, science, evidence, previous)
    payload = libs["src.csr_explicit_adjoint"]._invoke(evidence, "previous_raw_cache_CPU_loads", libs["torch"].load,
                                                    target, map_location="cpu", weights_only=False)
    _require(isinstance(payload, dict) and set(payload) == {"schema", "pass_index", "context", "common", "common_numerical_digest", "common_numerical_seal", "content_sha256"}
             and type(payload["schema"]) is int and payload["schema"] == 1
             and type(payload["pass_index"]) is int and payload["pass_index"] == 1, "Malformed previous raw cache")
    _finite_tree(libs, payload["common"], cpu_only=True)
    _common_contract(libs, payload["common"], science)
    digest = _descriptor(libs, payload["common"])
    _require(payload["context"] == _context(spec, evidence) and _seal(payload["context"]) == first["raw_cache_context_sha256"]
             and digest == payload["common_numerical_digest"] == first["common_numerical_digest"]
             and _seal(digest) == payload["common_numerical_seal"] == first["common_numerical_seal"]
             and _seal(dict(schema=1, pass_index=1, context=payload["context"], common_numerical_digest=digest))
             == payload["content_sha256"] == first["raw_cache_content_sha256"], "Previous full CPU cache seals differ")
    evidence.update(previous_receipt_path=str(path), previous_receipt_sha256=previous, previous_raw_cache_path=str(target),
                    previous_raw_cache_sha256=first["raw_cache_sha256"], previous_common_numerical_digest=digest)
    _require(evidence["common_numerical_digest"] == digest and evidence["common_numerical_seal"] == _seal(digest),
             "Full anchor0 feature-band16 exact repeat differs")
    evidence["within_anchor0_feature_band16_exact_repeat_passed"] = True


def _retained(libs, retained, evidence):
    for name, before, after in (("origin", "origin_buffers_before", "origin_buffers_after"),
                               ("inputs", "native_source_buffers_before", "native_source_buffers_after"),
                               ("anchor", "anchor_digest_before", "anchor_digest_after"),
                               ("common", "common_numerical_digest", "common_numerical_digest_after")):
        if name in retained and before in evidence:
            value = retained[name]["arrays"] if name == "origin" else retained[name]
            evidence[after] = _descriptor(libs, value)
            _require(evidence[after] == evidence[before], "Retained buffers changed: " + name)
            evidence[name + "_buffers_unchanged"] = True
    evidence["retained_failure_scope"] = "Only captured retained buffers are certified; missing earlier/failed ephemeral arrays are unqualified."


def _observe_raw(target, evidence):
    evidence["observed_raw_cache_path"] = str(target)
    evidence["observed_raw_cache_exists"] = target.is_file()
    if target.is_file():
        evidence.update(observed_raw_cache_sha256=_sha(target), observed_raw_cache_bytes=target.stat().st_size,
                        observed_raw_cache_complete=evidence.get("current_finite_CPU_raw_capture_completed") is True,
                        observed_raw_cache_partial_unqualified=evidence.get("current_finite_CPU_raw_capture_completed") is not True)


def _resources(libs, started, evidence, science):
    t = libs["torch"]
    t.cuda.synchronize()
    evidence.update(elapsed_seconds=time.monotonic() - started, peak_allocated_bytes=t.cuda.max_memory_allocated(0),
                    peak_reserved_bytes=t.cuda.max_memory_reserved(0), CUDA_total_bytes=t.cuda.get_device_properties(0).total_memory)
    _require(math.isfinite(evidence["elapsed_seconds"]) and 0 <= evidence["elapsed_seconds"] <= 300
             and 0 <= evidence["peak_allocated_bytes"] <= evidence["peak_reserved_bytes"] <= evidence["CUDA_total_bytes"]
             == science["native_policy"]["CUDA_total_bytes"], "Runtime/memory bound failed")


def prepare_source_gradient(pass_index, spec_path, spec_sha256, output_path, previous_sha256=None, stop=lambda: False):
    started = time.monotonic()
    _require(callable(stop), "Require stop callback")
    spec, science, repo = _load_spec(spec_path, spec_sha256)
    output, target = _request(pass_index, previous_sha256, output_path, spec)
    evidence = dict(passed=False, operation="prepare_source_gradient", pass_index=pass_index, validation_only=True,
                    spec_path=str(Path(spec_path).resolve()), spec_sha256=spec_sha256,
                    source=spec["source"], numerical_source=spec["numerical_source"], python_version=spec["python_version"],
                    scientific_preregistration=spec["scientific_preregistration"], files_sha256=spec["files_sha256"],
                    fixed=spec["fixed"], source_backend=BACKEND, counts=dict(science["counts_contract"]["zero_work_per_worker"]),
                    operation_attempts={}, success_counts=None, test_enabled=False,
                    count_scope=science["counts_contract"]["scope"], local_resource_finite_passed=False,
                    within_anchor0_feature_band16_exact_repeat_passed=False, anchor0_resource_reproducibility_qualified=False,
                    actual_source_gradient_accuracy_qualified=False, large_gradient_backend_qualified=False,
                    production_alignment_target_consumption_allowed=False, original_BI_or_BK_rescued=False,
                    three_anchor_Arxiv_qualified=False, all15_budgets_open=True, goal_complete=False,
                    stage="before_lazy_numerical_imports")
    libs, retained, primary, rng = None, {}, None, None
    try:
        _require(not stop() and time.monotonic() - started < 300, "Stopped before libraries")
        libs = _libraries()
        evidence["native_environment"] = _native(libs, science)
        evidence["native_environment"].update(source_backend=BACKEND, precision_initializer_label_scope="precision_only")
        _require(evidence["native_environment"]["versions"] == spec["numerical_source"]["versions"], "Runtime versions changed")
        evidence["_guard"] = lambda: _guard(libs, started, stop)
        t = libs["torch"]
        rng = (t.random.get_rng_state().clone(), t.cuda.get_rng_state().clone())
        _preserve(spec, Path(spec_path), spec_sha256, science, repo)
        evidence["stage"] = "accepted_BW_CPU_origin"
        arrays = _origin(libs, science, evidence, retained)
        evidence["stage"] = "one_tiled_source_gradient_target"
        common = _collect(libs, arrays, science, evidence, retained)
        _preserve(spec, Path(spec_path), spec_sha256, science, repo)
        evidence["stage"] = "finite_CPU_capture_before_repeat"
        _capture(libs, target, _context(spec, evidence), pass_index, common, evidence)
        if pass_index == 2:
            evidence["stage"] = "previous_CPU_full_common_repeat"
            _repeat(libs, spec, science, evidence, previous_sha256)
        evidence["passed"] = True
    except BaseException as error:
        primary = error
        evidence.update(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    finally:
        try:
            if libs is not None:
                _retained(libs, retained, evidence)
                _precision(libs["torch"])
                if rng is not None:
                    t = libs["torch"]
                    _require(t.equal(rng[0], t.random.get_rng_state()) and t.equal(rng[1], t.cuda.get_rng_state()), "Global RNG changed")
                    evidence["global_CPU_CUDA_RNG_unchanged"] = True
                if "native_environment" in evidence:
                    _resources(libs, started, evidence, science)
            _preserve(spec, Path(spec_path), spec_sha256, science, repo)
            evidence["source_assets_spec_science_unchanged"] = True
            if evidence["passed"]:
                expected = _success_counts(science, pass_index)
                _require({key: evidence["counts"].get(key, 0) for key in expected} == expected, "Observed interface counts differ")
                evidence.update(success_counts=expected, local_resource_finite_passed=True,
                                anchor0_resource_reproducibility_qualified=pass_index == 2)
        except BaseException as error:
            evidence.update(passed=False, finally_error_type=type(error).__name__, finally_error=str(error), finally_traceback=traceback.format_exc())
            if primary is None:
                primary = error
        try:
            _observe_raw(target, evidence)
        except BaseException as error:
            evidence.update(passed=False, raw_observation_error_type=type(error).__name__, raw_observation_error=str(error))
            if primary is None:
                primary = error
        if not evidence["passed"]:
            evidence.update(success_counts=None, local_resource_finite_passed=False, anchor0_resource_reproducibility_qualified=False)
        evidence.pop("_guard", None)
        evidence.setdefault("elapsed_seconds", time.monotonic() - started)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x") as stream:
            json.dump(evidence, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    if primary is not None:
        raise primary
    return dict(evidence, evidence_path=str(output), evidence_sha256=_sha(output), validation_only=True)
