"""Readonly H/RMS ancestry observations; never qualifies or repairs a source.

Inverse affine coordinates are floating-point proxies, not certified historical
H. Cancellation near zeros and a successful proxy reencode do not waive the
unchanged original 1e-12 source guard. Only H and RMS inputs are tensor-loaded.
"""
import hashlib
import inspect
import json
import math
import platform
import time
from pathlib import Path

import torch

from src import citation_search
from src import finite_student_probe as probe
from src.citation_source_preflight import _exact, _write_new
from src.io import _fingerprint
from src.research_loop import implementation_provenance
from src.shared_features import _tensor_identity
from src.transforms import FeatureTransform

SCIENCE = "Citeseer_allbudgets_readonly_RMS_provenance_scientific_stageAS_v1.json"
SCIENCE_SHA = "3dbe3acbef6f4f7b0dc9da399b49d0042e41a773bb15ce9accfba428a8a51292"
CUDA_CAPACITY = 8316977152
FIXED = dict(dataset="citeseer", condensation_seed=0, device="cuda", nodes=3327, dimension=3703,
             classes=6, native_policy="deterministic_cuda_v1", P_updates=0, student_fits=0)
ROOTS = ((30, .009, "row", "ca115f4751a1"), (60, .018, "default", "e6669760f3b3"),
         (120, .036, "row", "2e2127c2bc3f"))


def _roots(repo):
    return [dict(cells=cells, ratio=ratio, citation_features=features, root=key,
                 source_root=str(repo / "results/citation_search_v1/citeseer" / f"ratio_{ratio}" / key))
            for cells, ratio, features, key in ROOTS]


def _paths(repo):
    names = ("config.json", "propagated_H.pt", "inputs_0.pt", "teacher.pt", "assignment_20199efc7d26.pt",
             "nystrom_map_schema3.pt", "nystrom_phi_schema3.npy", "nystrom_phi_schema3.meta.json")
    files = [Path(root["source_root"]) / name for root in _roots(repo) for name in names]
    files += [repo / "data/citeseer/raw" / f"ind.citeseer.{name}" for name in
              ("x", "tx", "allx", "y", "ty", "ally", "graph", "test.index")]
    files += sorted((repo / "data/citeseer/processed").glob("*.pt"))
    return files


def _config(root):
    config = json.loads((Path(root["source_root"]) / "config.json").read_text())
    probe._require(isinstance(config, dict) and _fingerprint(config) == root["root"]
                   and config.get("dataset") == "citeseer" and type(config.get("ratio")) is float
                   and config["ratio"] == root["ratio"]
                   and config.get("citation_features", "default") == root["citation_features"],
                   "Original config/root fingerprint or preprocessing differs")
    return config


def _load_spec(path, digest, output, repo):
    path, output = Path(path).resolve(), Path(output).resolve()
    probe._require(path.is_relative_to(repo / "results/proposals") and path.is_file()
                   and probe._sha(path) == digest, "Require frozen prospective spec path/SHA")
    spec = json.loads(path.read_text())
    fields = {"schema", "fixed", "roots", "source", "numerical_source", "python_version", "files_sha256",
              "scientific_preregistration", "output_path"}
    roots = _roots(repo)
    probe._require(isinstance(spec, dict) and set(spec) == fields and type(spec["schema"]) is int
                   and spec["schema"] == 1 and _exact(spec["fixed"], FIXED)
                   and isinstance(spec["roots"], list) and len(spec["roots"]) == len(roots)
                   and all(_exact(actual, expected) for actual, expected in zip(spec["roots"], roots)),
                   "Unknown or changed fixed provenance spec/roots")
    probe._require(output == Path(spec["output_path"]) and output.is_relative_to(repo / "results/research_loop")
                   and output.suffix == ".json" and not output.exists(), "Require absent new frozen evidence path")
    science = spec["scientific_preregistration"]
    probe._require(science == dict(path=str(repo / "results/proposals" / SCIENCE), sha256=SCIENCE_SHA),
                   "Changed fixed scientific reference")
    pins = spec["files_sha256"]
    probe._require(isinstance(pins, dict) and set(pins) == {str(p) for p in _paths(repo)}
                   and (repo / "data/citeseer/processed/data.pt").is_file(), "Require exact existing asset inventory")
    probe._checked_files(pins)
    probe._checked_files({science["path"]: science["sha256"]})
    frozen_science = json.loads(Path(science["path"]).read_text())
    probe._require(_exact(frozen_science.get("fixed"), FIXED) and frozen_science.get("roots") == roots
                   and isinstance(frozen_science.get("parents"), list) and len(frozen_science["parents"]) == 3,
                   "Malformed fixed scientific lineage")
    probe._checked_files({ref["path"]: ref["sha256"] for ref in frozen_science["parents"]})
    _source_unchanged(spec)
    for root in roots:
        _config(root)
    return spec, frozen_science["parents"], output


def _source_unchanged(spec):
    probe._require(spec["source"] == implementation_provenance()
                   and spec["numerical_source"] == probe.numerical_source()
                   and spec["python_version"] == platform.python_version(), "Current source/Git/Python/versions differ")


def _transform(state, h):
    fields = {"center", "matrix", "output_center", "scale", "kind", "eps"}
    probe._require(isinstance(state, dict) and set(state) == fields and state["kind"] == "rms"
                   and state["matrix"] is None, "Require original affine RMS transform")
    for name in ("center", "output_center"):
        probe._tensor(state[name], (h.shape[1],), torch.float64, f"Malformed RMS {name}")
    probe._tensor(state["scale"], (), torch.float64, "Malformed RMS scalar scale")
    probe._require(float(state["scale"]) > 0 and type(state["eps"]) is float
                   and math.isfinite(state["eps"]) and state["eps"] > 0
                   and all(state[name].device == h.device for name in ("center", "output_center", "scale")),
                   "Malformed RMS positive controls/device")
    return FeatureTransform(**state)


def diagnose_rms(h, z, state):
    """Pure generic observations; no mutation, refit, source recovery or acceptance."""
    probe._require(torch.is_tensor(h) and h.ndim == 2 and min(h.shape) > 0, "Require nonempty H matrix")
    probe._tensor(h, h.shape, torch.float32, "Malformed finite FP32 H")
    probe._tensor(z, h.shape, torch.float64, "Malformed finite FP64 z")
    probe._require(z.device == h.device, "H/z devices differ")
    h, z = h.detach(), z.detach()
    transform = _transform(state, h)
    actual = transform(h.double())
    inverse = ((z * transform.scale) + transform.output_center) + transform.center
    rounded = inverse.float()
    probe._require(bool(torch.isfinite(actual).all()) and bool(torch.isfinite(inverse).all())
                   and bool(torch.isfinite(rounded).all()), "Nonfinite RMS replay/inverse/FP32 cast")
    reencoded = transform(rounded.double())
    difference, cast_error = inverse-h.double(), inverse-rounded.double()
    fp32_difference = rounded.double()-h.double()
    positive = (h > 0) & (rounded > 0)
    ulp = (rounded.contiguous().view(torch.int32).to(torch.int64)
           - h.contiguous().view(torch.int32).to(torch.int64)).abs()[positive]
    def replay(value):
        delta = (z-value).abs()
        return dict(max_abs=float(delta.max()), max_coordinate_relative=float((delta/z.abs().clamp_min(1e-300)).max()),
                    relative_denominator_floor=1e-300, strict_allclose=bool(torch.allclose(z, value, atol=1e-12, rtol=1e-12)),
                    atol=1e-12, rtol=1e-12)
    return dict(strict_original_H_RMS=replay(actual), inverse_FP32_reencode=replay(reencoded),
                inverse_order="((z*scale)+output_center)+center", inverse_proxy_max_abs=float(difference.abs().max()),
                inverse_proxy_signed_min=float(difference.min()), inverse_proxy_signed_max=float(difference.max()),
                nearest_FP32_max_abs=float(cast_error.abs().max()), nearest_FP32_nonrepresentable_coordinates=int((cast_error != 0).sum()),
                cast_FP32_equal_coordinates=int((rounded == h).sum()), cast_FP32_changed_coordinates=int((rounded != h).sum()),
                cast_FP32_max_abs=float(fp32_difference.abs().max()), cast_FP32_signed_min=float(fp32_difference.min()),
                cast_FP32_signed_max=float(fp32_difference.max()), positive_finite_ULP_coordinates=int(positive.sum()),
                positive_finite_ULP_max=int(ulp.max()) if len(ulp) else None,
                positive_finite_ULP_changed_coordinates=int((ulp != 0).sum()),
                saved_H_negative_coordinates=int((h < 0).sum()), inverse_proxy_negative_coordinates=int((inverse < 0).sum()),
                cast_negative_coordinates=int((rounded < 0).sum()), saved_H_zero_coordinates=int((h == 0).sum()),
                inverse_nonzero_at_saved_zero=int(((h == 0) & (inverse != 0)).sum()),
                cast_nonzero_at_saved_zero=int(((h == 0) & (rounded != 0)).sum()),
                cast_zero_at_saved_positive=int(((h > 0) & (rounded == 0)).sum()),
                source_accepted=False, historical_H_recovered=False, observational_only=True,
                zero_cancellation_caveat="Inverse cancellation can create nonzero proxies at saved zeros; ULP excludes zero/negative pairs.")


def _load_pair(root, evidence):
    path = Path(root["source_root"])
    evidence["existing_H_RMS_tensor_loads"] += 1
    h_state = torch.load(path / "propagated_H.pt", map_location="cuda", weights_only=False)
    evidence["existing_H_RMS_tensor_loads"] += 1
    saved = torch.load(path / "inputs_0.pt", map_location="cuda", weights_only=False)
    probe._require(isinstance(h_state, dict) and type(h_state.get("schema")) is int and h_state["schema"] == 1
                   and h_state.get("kind") == "shared_h" and isinstance(saved, dict), "Malformed shared H/RMS cache schema")
    h = probe._tensor(h_state.get("h"), (3327, 3703), torch.float32, "Malformed original saved H")
    z = probe._tensor(saved.get("z"), h.shape, torch.float64, "Malformed original saved z")
    config = _config(root)
    expected = _fingerprint(dict(data_digest=config["data_digest"], steps=2,
        implementation=hashlib.sha256((inspect.getsource(citation_search.normalize_adj_sparse)
                         + inspect.getsource(citation_search._prepare_dataset)).encode()).hexdigest(),
        dtype=str(h.dtype), torch=str(torch.__version__)))
    probe._require(h_state.get("source_digest") == expected and h_state.get("identity") == _tensor_identity(h),
                   "Shared H original source/header/content fingerprint differs")
    diagnostic = diagnose_rms(h, z, saved.get("transform"))
    return dict(**root, **diagnostic, H_identity=h_state["identity"], H_source_digest=h_state["source_digest"],
                saved_z_digest=probe._digest(z), transform_digest=probe._digest(saved["transform"]))


def prepare_provenance(spec_path, spec_sha256, output_path, stop=lambda: False):
    """Diagnostic completion only; six H/RMS loads, zero research/model operations."""
    repo = Path(__file__).resolve().parents[1]
    spec, parents, output = _load_spec(spec_path, spec_sha256, output_path, repo)
    evidence = dict(passed=False, source_accepted=False, historical_H_recovered=False, source=spec["source"],
                    numerical_source=spec["numerical_source"], python_version=spec["python_version"],
                    spec_path=str(Path(spec_path).resolve()), spec_sha256=spec_sha256,
                    scientific_preregistration=spec["scientific_preregistration"], parents=parents,
                    files_sha256=spec["files_sha256"], roots={}, existing_H_RMS_tensor_loads=0,
                    P_updates=0, optimizer_steps=0, student_fits=0, head_solves=0, teacher_map_Phi_fits=0,
                    model_forwards=0, finite_unrolls=0, native_initializers=0, strict_source_guard_unchanged=True)
    started, native = time.monotonic(), False
    try:
        probe._stop(stop)
        evidence["stage"] = "fresh_native_policy"
        probe._require(torch.get_num_threads() == 4 and not torch.cuda.is_initialized(), "Require threads4/fresh CUDA worker")
        evidence["native_environment"] = dict(**probe._native("cuda"), python_version=platform.python_version(), threads=4)
        native = True
        torch.cuda.reset_peak_memory_stats()
        for root in spec["roots"]:
            probe._stop(stop)
            probe._runtime_precision_guard()
            evidence["stage"] = "readonly_H_RMS_" + root["root"]
            evidence["roots"][root["root"]] = _load_pair(root, evidence)
        probe._require(evidence["existing_H_RMS_tensor_loads"] == 6, "Unexpected tensor load count")
        evidence["passed"], evidence["stage"] = True, "diagnostic_complete_not_source_acceptance"
    except Exception as error:
        evidence["source_error"] = dict(type=type(error).__name__, message=str(error))
        evidence["failed_stage"] = evidence.get("stage", "pre_native_stop")
        raise
    finally:
        if native:
            try:
                torch.cuda.synchronize()
                evidence.update(CUDA_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                CUDA_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                                CUDA_total_bytes=torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory)
            except Exception as error:
                evidence["passed"] = False
                evidence["memory_error"] = dict(type=type(error).__name__, message=str(error))
        evidence["seconds"] = time.monotonic()-started
        try:
            probe._checked_files(spec["files_sha256"])
            probe._checked_files({ref["path"]: ref["sha256"] for ref in parents})
            science = spec["scientific_preregistration"]
            probe._checked_files({science["path"]: science["sha256"]})
            probe._require(probe._sha(Path(spec_path).resolve()) == spec_sha256, "Spec changed during diagnostic")
            _source_unchanged(spec)
            if native:
                probe._runtime_precision_guard()
                probe._require("memory_error" not in evidence and torch.get_num_threads() == 4
                               and evidence["CUDA_total_bytes"] == CUDA_CAPACITY
                               and 0 <= evidence["CUDA_peak_allocated_bytes"] <= CUDA_CAPACITY
                               and 0 <= evidence["CUDA_peak_reserved_bytes"] <= CUDA_CAPACITY,
                               "Native threads/capacity/peak policy changed")
            probe._require(evidence["seconds"] <= 300, "Diagnostic exceeded declared300seconds")
            evidence["source_assets_spec_science_unchanged"] = True
        except Exception as error:
            evidence["passed"] = False
            evidence["source_assets_spec_science_unchanged"] = False
            evidence["preservation_error"] = dict(type=type(error).__name__, message=str(error))
        _write_new(output, evidence)
    probe._require(evidence["passed"], "Diagnostic integrity failed; inspect new evidence")
    return dict(passed=True, source_accepted=False, validation_only=True, evidence_path=str(output),
                evidence_sha256=probe._sha(output), source=spec["source"], P_updates=0, student_fits=0,
                head_solves=0, existing_H_RMS_tensor_loads=6)
