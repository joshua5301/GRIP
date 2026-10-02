"""Two independent, readonly original-CSR source-gradient passes; no P or fits."""
import json
import math
import platform
import time
import traceback
from importlib.metadata import version as distribution_version
from pathlib import Path

import torch

from src import citation_gradient_probe as ay
from src import citation_search
from src import finite_student_probe as probe
from src import source_linear_assignment as source_helper
from src.ce_gradient_alignment import POLICY, source_gradient_targets
from src.citation_source_preflight import _exact, _write_new
from src.io import _fingerprint
from src.research_loop import implementation_provenance

SCIENCE = "Cora70_original_CSR_source_gradient_readonly_scientific_stageBI_v1.json"
SCIENCE_SHA = "60a132bcdaa6a7af9880a0e6804a5cc060f7d5329e6a833d85e591b156f7d920"
SCIENCE_REVIEW = "Cora70_original_CSR_source_gradient_scientific_independent_metadata_review_stageBI_v1.json"
SCIENCE_REVIEW_SHA = "67b90ae4c40498f380a24e8fd86926e8a9a7da47a8774cf6bd22077b2b290b59"
HELPER_SHA = "75f6ef10cd1f1e1e15aa2b10ec4e05a924f8bb3634df7b0d3f8f28020cf98bec"
anchor_initial = ay.anchor_initial
_require, _sha, _seal, _stop = probe._require, probe._sha, probe._seal, probe._stop
_POLICY = json.loads(json.dumps(POLICY))
_FIELDS = {"schema", "fixed", "source", "numerical_source", "python_version", "files_sha256",
           "scientific_preregistration", "artifacts_sha256", "output_root", "pass_outputs", "target_outputs", "comparison_policy"}


def numerical_source():
    value = ay.numerical_source()
    _require(value["files"].get("ce_gradient_alignment.py") == HELPER_SHA
             and value["files"].get("cora_csr_source_gradient.py") == _sha(__file__), "CSR/helper implementation changed")
    return value


def _science(repo):
    path = repo / "results/proposals" / SCIENCE
    probe._checked_files({str(path): SCIENCE_SHA})
    return json.loads(path.read_text())


def _paths(repo):
    return [Path(p) for p in _science(repo)["original_files_sha256"]]


def _count(evidence, key, function, *args, **kwargs):
    _stop(evidence["_stop"])
    evidence["operation_attempts"][key] = evidence["operation_attempts"].get(key, 0) + 1
    result = function(*args, **kwargs)
    evidence["counts"][key] = evidence["counts"].get(key, 0) + 1
    return result


def _preserve(spec, path, checksum, science, repo):
    _require(_sha(path) == checksum, "Execution spec changed")
    probe._checked_files(spec["files_sha256"])
    probe._checked_files(spec["artifacts_sha256"])
    probe._checked_files({spec["scientific_preregistration"]["path"]: SCIENCE_SHA,
                         str(repo / "results/proposals" / SCIENCE_REVIEW): SCIENCE_REVIEW_SHA})
    probe._checked_files({p["path"]: p["sha256"] for p in science["parents"]})
    probe._checked_files({str(repo / p): s for p, s in science["old_tests_sha256"].items()})
    probe._checked_files({str(repo / p): s for p, s in science["source_before"]["files"].items()
                         if p != "src/research_loop.py"})
    comparator = science["dense_comparison"]["target_cache"]
    probe._checked_files({comparator["path"]: comparator["sha256"]})
    version = science["version_binding"]["torch_version_file"]
    probe._checked_files({version["path"]: version["sha256"]})
    _require({k: distribution_version(k) for k in science["version_binding"]["installed_distribution_versions"]}
             == science["version_binding"]["installed_distribution_versions"], "Installed distribution versions changed")
    _require(implementation_provenance() == spec["source"] and numerical_source() == spec["numerical_source"]
             and platform.python_version() == spec["python_version"] and torch.get_num_threads() == 4,
             "Source/Git/versions/Python/threads changed")


def _artifact_paths(pins, science, repo):
    owned = {(repo / science["helper_boundary"][key]).resolve()
             for key in ("next_owned_source", "next_owned_test")}
    owned.add((repo / "src/research_loop.py").resolve())
    _require(isinstance(pins, dict) and pins
             and all(type(p) is str and Path(p).is_absolute()
                     and (Path(p).resolve().is_relative_to(repo / "results") or Path(p).resolve() in owned)
                     for p in pins), "Require immutable reviewed artifact pins")


def _load_spec(path, checksum):
    repo = Path(__file__).resolve().parents[1]
    path = Path(path).resolve()
    _require(type(checksum) is str and len(checksum) == 64 and path.is_file()
             and path.is_relative_to(repo / "results/proposals") and _sha(path) == checksum, "Require pinned prospective spec")
    science, spec = _science(repo), json.loads(path.read_text())
    _require(isinstance(spec, dict) and set(spec) == _FIELDS and type(spec["schema"]) is int and spec["schema"] == 1,
             "Unknown/malformed CSR spec")
    for key in ("fixed", "files_sha256", "output_root", "pass_outputs", "target_outputs"):
        expected = science["original_files_sha256"] if key == "files_sha256" else science[key]
        _require(_exact(spec[key], expected), "Changed fixed CSR spec: " + key)
    _require(_exact(spec["comparison_policy"], science["dense_comparison"])
             and _exact(spec["scientific_preregistration"], dict(path=str(repo / "results/proposals" / SCIENCE), sha256=SCIENCE_SHA))
             and _exact(science["gradient_policy"], _POLICY), "Changed scientific/comparison policy")
    _artifact_paths(spec["artifacts_sha256"], science, repo)
    _require(isinstance(spec["source"], dict) and isinstance(spec["source"].get("files"), dict)
             and set(spec["source"]["files"]) == set(science["source_before"]["files"]) | {"src/cora_csr_source_gradient.py"}
             and spec["python_version"] == science["version_binding"]["python_version"]
             and spec["numerical_source"].get("versions") == science["version_binding"]["runtime_versions_from_pinned_native_metadata"],
             "Require exact promoted69-file source and frozen versions")
    _preserve(spec, path, checksum, science, repo)
    return spec, science


def _request(pass_index, previous, output, spec):
    _require(type(pass_index) is int and pass_index in (1, 2), "Require fixed pass1 or2")
    _require((pass_index == 1 and previous is None) or
             (pass_index == 2 and type(previous) is str and len(previous) == 64), "Wrong prior-pass SHA protocol")
    output, target = Path(output).resolve(), Path(spec["target_outputs"][str(pass_index)])
    _require(str(output) == spec["pass_outputs"][str(pass_index)] and output.parent == target.parent
             and not output.exists() and not output.parent.exists(), "Require exclusive declared new pass namespace")
    return output, target


def _source_digest(buffers):
    return probe._digest(dict(H=buffers["h"], z=buffers["z"], Q=buffers["q"], hard=buffers["hard"],
        transform=buffers["transform"], X=buffers["x"], original_CSR=buffers["S"]))


def _bind_source(source, native, ghost, science):
    expected = dict(science["original_source_binding"]["source_context_descriptor"], candidate=ghost)
    _require(_exact(source, expected) and _exact(native, science["original_source_binding"]["native_buffers"]),
             "Original source/CSR/X/Q/H/RMS/hard/map/Phi provenance differs")


def _load_source(spec, science, evidence, repo):
    evidence["stage"] = "require_existing_dataset_source"
    root = Path(spec["fixed"]["source_root"])
    config = science["original_source_binding"]["config"]
    dataset_files = [repo / "data/cora/raw" / ("ind.cora." + name) for name in
                     ("x", "tx", "allx", "y", "ty", "ally", "graph", "test.index")]
    dataset_files += [repo / "data/cora/processed" / name for name in ("data.pt", "pre_filter.pt", "pre_transform.pt")]
    _require(all(p.is_file() and str(p) in spec["files_sha256"] for p in dataset_files), "No dataset creation/download fallback")
    graph, train, val, test, fresh_h = _count(evidence, "dataset_prepare_calls", citation_search._prepare_dataset,
                                           "cora", str(repo / "data"), "cuda", "row")
    evidence["counts"]["dataset_H_validation_sparse_products"] = 2
    actual_config = citation_search._legacy_teacher_config("cora", .026, graph, train, val, test, "row")
    _require(_exact(actual_config, config) and _fingerprint(config) == "19fccc37cc2f", "Graph/config/preprocessing changed")
    _require(graph["adj"].layout == torch.sparse_csr and graph["adj"].dtype == torch.float32
             and tuple(graph["adj"].shape) == (2708, 2708) and not graph["adj"].requires_grad,
             "Source backend must be original native FP32 CSR")
    teacher = torch.load(root / "teacher.pt", map_location="cuda", weights_only=False)
    _require(isinstance(teacher, dict), "Malformed original teacher")
    logits = probe._tensor(teacher.get("logits"), (2708, 7), torch.float64, "Malformed source logits")
    q = _count(evidence, "source_Q_constructions", lambda: (logits / .3).softmax(1).detach())
    controls = dict(science["original_source_binding"]["source_context_descriptor"]["candidate"])
    controls.pop("source_linear_source_digest")
    ghost = source_helper.candidate_controls(controls)
    h, z, transform, hard, _, source = _count(evidence, "cached_source_validation_calls", source_helper.cached_source,
                                            root, ghost, 0, fresh_h, q, config)
    # This additional BI consistency bound was fixed by SCIENCE before native outcomes.
    # It is separate from protected RMS1e-12 and from dense/CSR gradient equivalence.
    _require(torch.allclose(fresh_h, h, atol=2e-7, rtol=1e-6), "Fresh propagated H differs from original cached H")
    buffers = dict(h=h, z=z, q=q, hard=hard, transform=probe._frozen_transform(transform), x=graph["x"], S=graph["adj"])
    _bind_source(source, _source_digest(buffers), ghost, science)
    evidence.update(source_context=source, native_source_buffers_before=_source_digest(buffers),
                    source_reference_provenance_passed=True, heldout_labels_scope="immutable graph identity hashes only")
    return buffers


def _sealed_payload(value):
    _require(isinstance(value, dict) and type(value.get("content_sha256")) is str
             and _seal({k: v for k, v in value.items() if k != "content_sha256"}) == value["content_sha256"],
             "Malformed/changed target cache seal")
    return value


def _historical_targets(science, evidence):
    comparator = science["dense_comparison"]["target_cache"]
    payload = _count(evidence, "historical_dense_target_cache_loads", torch.load,
                     comparator["path"], map_location="cuda", weights_only=False)
    targets = _sealed_payload(payload).get("targets")
    _require(probe._digest(targets) == science["dense_comparison"]["complete_target_digest_from_historical_receipt"],
             "Historical dense target/model numerical digest differs")
    return targets


def _target_valid(targets):
    _require(isinstance(targets, dict) and _exact(json.loads(json.dumps(targets.get("policy"))), _POLICY)
             and isinstance(targets.get("anchors"), (tuple, list)) and len(targets["anchors"]) == 3, "Malformed3-anchor target policy")
    for target in targets["anchors"]:
        _require(isinstance(target, dict) and set(target) == {"parameters", "gradients", "source_ce", "source_norms", "delta", "source_relu"},
                 "Malformed target fields")
        for key in ("parameters", "gradients"):
            values = target[key]
            _require(isinstance(values, (list, tuple)) and len(values) == 4, "Malformed target blocks")
            for value, shape in zip(values, ((256, 1433), (256,), (7, 256), (7,)), strict=True):
                probe._tensor(value, shape, torch.float32, "Malformed target/anchor tensor")
                _require(not value.requires_grad and value.device.type == "cuda", "Target/anchor must be frozen native CUDA")
        ce = probe._tensor(target["source_ce"], (), torch.float64, "Nonfinite source CE")
        _require(float(ce) >= 0, "Negative source CE")
        for key in ("source_norms", "delta"):
            _require(isinstance(target[key], (list, tuple)) and len(target[key]) == 4, "Malformed norm/delta blocks")
        for norm, delta in zip(target["source_norms"], target["delta"], strict=True):
            probe._tensor(norm, (), torch.float64, "Nonfinite target norm")
            probe._tensor(delta, (), torch.float64, "Nonfinite target delta")
            _require(float(norm) > 0 and float(delta) > 0 and torch.equal(delta, .001 * norm), "Invalid positive norm/rho smoothing")
        relu = target["source_relu"]
        _require(isinstance(relu, dict) and set(relu) == {"minimum_absolute_preactivation", "exact_zero_preactivations"}, "Malformed margin")
        margin = probe._tensor(relu["minimum_absolute_preactivation"], (), torch.float32, "Nonfinite margin")
        _require(float(margin) >= 0 and type(relu["exact_zero_preactivations"]) is int
                 and 0 <= relu["exact_zero_preactivations"] <= 2708 * 256, "Malformed ReLU diagnostics")


def _retained(buffers, evidence):
    evidence["native_source_buffers_after"] = _source_digest(buffers)
    _require(_exact(evidence["native_source_buffers_after"], evidence["native_source_buffers_before"]), "Retained source buffers changed")
    if "anchors" in buffers:
        evidence["anchor_digests_after"] = probe._digest(buffers["anchors"])
        _require(evidence["anchor_digests_after"] == evidence["anchor_digests_before"], "Frozen anchors changed")
    if "targets" in buffers:
        _require(probe._digest(buffers["targets"]) == evidence["target_numerical_digest"], "Completed target cache changed")


def _collect(buffers, dense, evidence):
    anchors = []
    for seed, old in zip((0, 1, 2), dense["anchors"], strict=True):
        anchor = _count(evidence, "private_anchor_factory_calls", anchor_initial, 1433, 7, 256, seed,
                        dtype=torch.float32, device="cuda")
        _require(probe._digest(anchor) == probe._digest(old["parameters"]), "Anchor differs from historical exact GEOM state")
        anchors.append(anchor)
    buffers["anchors"] = anchors
    evidence["anchor_digests_before"] = probe._digest(anchors)
    cached = []
    for anchor in anchors:
        evidence["_verify"]()
        _retained(buffers, evidence)
        probe._runtime_precision_guard()
        result = _count(evidence, "source_gradient_target_helper_calls", source_gradient_targets,
                        (anchor,), buffers["x"], buffers["S"], buffers["q"], stop=evidence["_stop"])
        _require(isinstance(result, dict) and _exact(json.loads(json.dumps(result.get("policy"))), _POLICY)
                 and isinstance(result.get("anchors"), (tuple, list)) and len(result["anchors"]) == 1, "Singleton helper policy/shape differs")
        cached.append(result["anchors"][0])
        evidence["counts"]["source_CE_parameter_gradient_calls"] += 1
        evidence["counts"]["source_ReLU_margin_calls"] += 1
    targets = dict(policy=dict(POLICY), anchors=tuple(cached))
    _target_valid(targets)
    buffers["targets"] = targets
    evidence.update(target_numerical_digest=probe._digest(targets), target_numerical_seal=_seal(targets))
    return targets


def _descriptive_comparison(targets, dense):
    rows = []
    for seed, left, right in zip((0, 1, 2), targets["anchors"], dense["anchors"], strict=True):
        blocks = []
        for name, s, t, delta in zip(("W1", "b1", "W2", "b2"), left["gradients"], right["gradients"], right["delta"], strict=True):
            s, t = s.double().reshape(-1), t.double().reshape(-1)
            ns, nt = torch.linalg.vector_norm(s), torch.linalg.vector_norm(t)
            dot = torch.dot(s, t)
            row = dict(block=name, max_abs=float((s - t).abs().max()), relative_L2=float(torch.linalg.vector_norm(s - t) / nt),
                raw_cosine=float(dot / (ns * nt)), smoothed_cosine=float(dot / ((ns.square() + delta.square()).sqrt() *
                                                                         (nt.square() + delta.square()).sqrt())),
                CSR_norm=float(ns), dense_norm=float(nt), CSR_delta=float(.001 * ns), dense_delta=float(delta))
            _require(all(math.isfinite(v) for k, v in row.items() if k != "block"), "Nonfinite descriptive metric")
            blocks.append(row)
        rows.append(dict(seed=seed, blocks=blocks, CSR_source_CE=float(left["source_ce"]), dense_source_CE=float(right["source_ce"]),
            source_CE_difference=float(left["source_ce"] - right["source_ce"]),
            CSR_source_relu={k: float(v) if torch.is_tensor(v) else v for k, v in left["source_relu"].items()},
            dense_source_relu={k: float(v) if torch.is_tensor(v) else v for k, v in right["source_relu"].items()}))
    return dict(status="DESCRIPTIVE_ONLY_NO_EQUIVALENCE_BOUND_OR_ACCEPTANCE_CLAIM", anchors=rows,
                thresholds=None, dense_CSR_equivalent=False, fresh_dense_gradient_calls=0)


def _previous_receipt(spec, science, evidence, previous_sha256):
    first_path = Path(spec["pass_outputs"]["1"])
    _require(_sha(first_path) == previous_sha256, "First receipt SHA differs")
    first = json.loads(first_path.read_text())
    for key in ("source", "numerical_source", "python_version", "scientific_preregistration", "spec_path", "spec_sha256"):
        _require(_exact(first.get(key), evidence[key]), "First-pass provenance differs: " + key)
    _require(first.get("passed") is True and type(first.get("pass_index")) is int and first["pass_index"] == 1
             and first.get("operation") == "prepare_targets" and first.get("source_backend") == "existing_original_CSR"
             and first.get("source_assets_spec_science_unchanged") is True and first.get("strict_CSR_source_gradient_backend_qualified") is False
             and first.get("within_CSR_exact_repeat_passed") is False and first.get("test_enabled") is False
             and first.get("validation_only") is True and first.get("source_reference_provenance_passed") is True
             and _exact(first.get("success_counts"), science["counts_contract"]["per_successful_worker"]), "First local pass was not qualified")
    path = Path(spec["target_outputs"]["1"])
    _require(first.get("target_cache_path") == str(path) and _sha(path) == first.get("target_cache_sha256"), "First target bytes differ")
    return first, first_path, path


def _repeat(spec, science, evidence, targets, previous_sha256):
    first, first_path, path = _previous_receipt(spec, science, evidence, previous_sha256)
    _require(_exact(first.get("native_environment"), evidence["native_environment"]), "First native environment differs")
    payload = _count(evidence, "previous_CSR_target_cache_loads", torch.load, path, map_location="cuda", weights_only=False)
    _sealed_payload(payload)
    context = dict(source=spec["source"], numerical_source=spec["numerical_source"], scientific_preregistration=spec["scientific_preregistration"],
                   spec_sha256=evidence["spec_sha256"], source_context=evidence["source_context"], native_source_buffers=evidence["native_source_buffers_before"], source_backend="existing_original_CSR")
    _require(_exact(payload.get("context"), context) and type(payload.get("pass_index")) is int and payload["pass_index"] == 1
             and probe._digest(payload.get("targets")) == first.get("target_numerical_digest")
             and _seal(payload["targets"]) == first.get("target_numerical_seal") == _seal(targets), "Within-CSR exact target repeat differs")
    evidence.update(previous_receipt_path=str(first_path), previous_receipt_sha256=previous_sha256,
                    previous_target_cache_path=str(path), previous_target_cache_sha256=_sha(path),
                    within_CSR_exact_repeat_passed=True, strict_CSR_source_gradient_backend_qualified=True)


def prepare_targets(pass_index, spec_path, spec_sha256, output_path, previous_sha256=None, stop=lambda: False):
    _require(callable(stop), "Require stop callback")
    repo, spec_path = Path(__file__).resolve().parents[1], Path(spec_path).resolve()
    spec, science = _load_spec(spec_path, spec_sha256)
    output, target = _request(pass_index, previous_sha256, output_path, spec)
    started = time.monotonic()
    bounded = lambda: stop() or time.monotonic() - started >= 300
    evidence = dict(passed=False, operation="prepare_targets", pass_index=pass_index, source=spec["source"],
        numerical_source=spec["numerical_source"], python_version=spec["python_version"], scientific_preregistration=spec["scientific_preregistration"],
        spec_path=str(spec_path), spec_sha256=spec_sha256, files_sha256=spec["files_sha256"], source_backend="existing_original_CSR",
        validation_only=True, test_enabled=False, count_scope="High-level attempts/completions; integrated source-gradient/margin and dataset H products count only after a full helper return. Failed internal work is unknown; recorded completions are lower bounds.", counts={k: 0 for k in science["counts_contract"]["per_successful_worker"]},
        operation_attempts={}, within_CSR_exact_repeat_passed=False, strict_CSR_source_gradient_backend_qualified=False,
        _stop=bounded, _verify=lambda: _preserve(spec, spec_path, spec_sha256, science, repo), stage="fresh_native_precision")
    native, buffers, primary = False, None, None
    try:
        _stop(bounded)
        if pass_index == 2:
            _previous_receipt(spec, science, evidence, previous_sha256)
        _require(not torch.cuda.is_initialized(), "Require fresh CUDA worker")
        initializer = probe._native("cuda")
        native = True
        torch.cuda.reset_peak_memory_stats()
        capacity = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
        expected = science["native_policy"]
        _require(capacity == expected["CUDA_total_bytes"] and initializer["GPU"] == expected["GPU"]
                 and initializer["cuda_runtime"] == expected["cuda_runtime"] and torch.get_num_threads() == 4,
                 "Native GPU/capacity/runtime/threads differ")
        evidence["native_environment"] = dict(source_backend="existing_original_CSR", actual_source_layout="torch.sparse_csr", threads=torch.get_num_threads(),
            python_version=platform.python_version(), inherited_precision_initializer=initializer, inherited_graph_label_scope="precision initializer metadata only; adjacency remains originalCSR")
        buffers = _load_source(spec, science, evidence, repo)
        dense = _historical_targets(science, evidence)
        _target_valid(dense)
        evidence["stage"] = "original_CSR_source_gradients"
        targets = _collect(buffers, dense, evidence)
        evidence["dense_comparison"] = _descriptive_comparison(targets, dense)
        if pass_index == 2:
            _repeat(spec, science, evidence, targets, previous_sha256)
        _stop(bounded)
        _preserve(spec, spec_path, spec_sha256, science, repo)
        probe._runtime_precision_guard()
        context = dict(source=spec["source"], numerical_source=spec["numerical_source"], scientific_preregistration=spec["scientific_preregistration"],
            spec_sha256=spec_sha256, source_context=evidence["source_context"], native_source_buffers=evidence["native_source_buffers_before"], source_backend="existing_original_CSR")
        _retained(buffers, evidence)
        output.parent.mkdir(parents=True, exist_ok=False)
        with target.open("xb") as stream:
            torch.save(probe._attach(dict(context=context, pass_index=pass_index, targets=targets)), stream)
        evidence.update(target_cache_path=str(target), target_cache_sha256=_sha(target), passed=True)
    except BaseException as error:
        primary = error
        evidence.update(passed=False, error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc(),
            partial_helper_failure_scope="Returned helper/data completions and integrated gradient/margin/H-product counts are lower bounds; partial internal work on failure is unknown. Failed ephemeral targets remain unqualified.")
    finally:
        try:
            _preserve(spec, spec_path, spec_sha256, science, repo)
            if buffers is not None:
                _retained(buffers, evidence)
            evidence["source_assets_spec_science_unchanged"] = True
        except BaseException as error:
            evidence.update(passed=False, source_assets_spec_science_unchanged=False, preservation_error=repr(error))
            primary = primary or error
        if native:
            try:
                torch.cuda.synchronize()
                capacity = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
                allocated, reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
                evidence.update(CUDA_total_bytes=capacity, CUDA_peak_allocated_bytes=allocated, CUDA_peak_reserved_bytes=reserved)
                _require(capacity == 8316977152 and 0 <= allocated <= capacity and 0 <= reserved <= capacity, "Native memory/capacity changed")
                probe._runtime_precision_guard()
            except BaseException as error:
                evidence.update(passed=False, native_finalization_error=repr(error))
                primary = primary or error
        evidence["seconds"] = time.monotonic() - started
        if evidence["seconds"] > 300:
            error = ValueError("CSR source qualification exceeded300seconds")
            evidence.update(passed=False, deadline_error=str(error))
            primary = primary or error
        if evidence["passed"]:
            expected = science["counts_contract"]["per_successful_worker"]
            actual = {k: evidence["counts"][k] for k in expected}
            if actual != expected:
                primary = ValueError("Unexpected actual CSR-only operation counts")
                evidence.update(passed=False, counter_error=str(primary))
            else:
                evidence["success_counts"] = actual
        if not evidence["passed"]:
            evidence["strict_CSR_source_gradient_backend_qualified"] = False
        evidence.pop("_stop")
        evidence.pop("_verify")
        try:
            _write_new(output, evidence)
        except BaseException:
            if primary is not None:
                raise primary
            raise
    if primary is not None:
        raise primary
    return dict(evidence, evidence_path=str(output), evidence_sha256=_sha(output), validation_only=True)
