"""Require-existing Citeseer60 native source evidence only; no research fits.

The strict original RMS loader is unchanged. Its error is recorded and re-raised;
an unsuccessful preflight is never a usable source or permission to change it.
"""
import json
import math
import os
import time
import uuid
from pathlib import Path

import torch

from src import citation_search
from src import finite_student_probe as probe
from src import source_linear_assignment as source_helper
from src.evaluation import _input_digest
from src.io import _fingerprint
from src.low_rank_assignment import LowRankLogits, LowRankMoments, initialize_factors
from src.moments import augmented, decode_moments, make_material
from src.research_loop import implementation_provenance
from src.soft_ce_partition import head_gradient, outer_value_gradient
from src.sweep_utils import representative
from src.target_refinement import training_refined_targets
from src.transforms import FeatureTransform

ROOT = "e6669760f3b3"
REFERENCE = "a2d47970c967"
SCIENCE = "Citeseer60_native_source_preflight_scientific_stageAR_v1.json"
SCIENCE_SHA = "c9ef3adba3057cc484bdd32f4c7b85e204d5e164af6ad723454a3ebc6fed263b"
CUDA_CAPACITY = 8316977152
FIXED = dict(dataset="citeseer", ratio=.018, citation_features="default", condensation_seed=0,
             device="cuda", nodes=3327, dimension=3703, classes=6, cells=60,
             native_policy="deterministic_cuda_v1", P_updates=0, student_fits=0)
CANDIDATE = dict(method="low_rank", width=0, lr=.01, T=1., rank=8, penalty=.001,
                 initialization="teacher_joint", alpha=.3, inner_loss_weighting="uniform")


def _exact(actual, expected):
    """Reject bool/numeric coercion and unknown nested controls."""
    if isinstance(expected, dict):
        return isinstance(actual, dict) and set(actual) == set(expected) and all(
            _exact(actual[key], value) for key, value in expected.items())
    return type(actual) is type(expected) and actual == expected


def _paths(repo):
    root = repo / "results/citation_search_v1/citeseer/ratio_0.018" / ROOT
    origin = dict(mode="teacher_joint", alpha=.3, T=1., seed=0)
    files = [root / name for name in ("config.json", "propagated_H.pt", "teacher.pt", "inputs_0.pt",
             "nystrom_map_schema3.pt", "nystrom_phi_schema3.npy", "nystrom_phi_schema3.meta.json",
             f"assignment_{_fingerprint(origin)}.pt")]
    files += [root / REFERENCE / "candidate.json", root / REFERENCE / "condensation_0/resume.pt"]
    files += [root / REFERENCE / "condensation_0/checkpoints" / f"step_{step:06d}.pt" for step in (0, 25)]
    files += [repo / "data/citeseer/raw" / f"ind.citeseer.{name}" for name in
              ("x", "tx", "allx", "y", "ty", "ally", "graph", "test.index")]
    files += sorted((repo / "data/citeseer/processed").glob("*.pt"))
    return root, files


def _load_spec(path, digest, output, repo):
    path, output = Path(path).resolve(), Path(output).resolve()
    probe._require(path.is_relative_to(repo / "results/proposals") and path.is_file()
                   and probe._sha(path) == digest, "Require frozen prospective spec path/SHA")
    spec = json.loads(path.read_text())
    fields = {"schema", "fixed", "candidate", "source_root", "reference_candidate_id", "source",
              "numerical_source", "files_sha256", "scientific_preregistration", "output_path"}
    probe._require(isinstance(spec, dict) and set(spec) == fields and type(spec["schema"]) is int
                   and spec["schema"] == 1 and _exact(spec["fixed"], FIXED)
                   and _exact(spec["candidate"], CANDIDATE), "Unknown or changed fixed preflight spec")
    root, files = _paths(repo)
    probe._require(spec["source_root"] == str(root) and spec["reference_candidate_id"] == REFERENCE,
                   "Original source/reference override forbidden")
    probe._require(output == Path(spec["output_path"]) and output.is_relative_to(repo / "results/research_loop")
                   and output.suffix == ".json" and not output.exists(), "Require absent new frozen evidence path")
    science = spec["scientific_preregistration"]
    probe._require(science == dict(path=str(repo / "results/proposals" / SCIENCE), sha256=SCIENCE_SHA),
                   "Changed fixed scientific reference")
    pins = spec["files_sha256"]
    probe._require(isinstance(pins, dict) and set(pins) == {str(p) for p in files}
                   and (repo / "data/citeseer/processed/data.pt").is_file(),
                   "Require exact existing source/raw/processed/reference inventory")
    probe._checked_files(pins)
    probe._checked_files({science["path"]: science["sha256"]})
    probe._require(spec["source"] == implementation_provenance()
                   and spec["numerical_source"] == probe.numerical_source(), "Current source/Git/versions differ from spec")
    return spec, root, output


def _write_new(path, value):
    """Atomic exclusive report publication; never overwrite an earlier result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _rms_diagnostic(root, device):
    """Observe the residual before the unchanged1e-12 loader; do not retarget z."""
    h_state = torch.load(root / "propagated_H.pt", map_location=device, weights_only=False)
    saved = torch.load(root / "inputs_0.pt", map_location=device, weights_only=False)
    probe._require(isinstance(h_state, dict) and isinstance(saved, dict)
                   and isinstance(saved.get("transform"), dict), "Malformed source RMS containers")
    h = probe._tensor(h_state.get("h"), (3327, 3703), torch.float32, "Malformed frozen H")
    z = probe._tensor(saved.get("z"), h.shape, torch.float64, "Malformed frozen RMS z")
    transform = FeatureTransform(**saved["transform"])
    actual = transform(h.double())
    probe._require(bool(torch.isfinite(actual).all()), "RMS replay is nonfinite")
    return dict(max_abs=float((z-actual).abs().max()),
                max_coordinate_relative=float(((z-actual).abs()/z.abs().clamp_min(1e-300)).max()),
                atol=1e-12, rtol=1e-12, strict_allclose=bool(torch.allclose(z, actual, atol=1e-12, rtol=1e-12)),
                observational_only=True, transform=probe._digest(probe._frozen_transform(transform)))


def _reference(buffers, ghost, evidence):
    evidence["stage"] = "cached_NODE_config_and_native_P0"
    root = buffers["root"] / REFERENCE
    probe._require(json.loads((root / "candidate.json").read_text()) == CANDIDATE, "Original NODE candidate changed")
    resume = torch.load(root / "condensation_0/resume.pt", map_location="cpu", weights_only=False)
    probe._require(isinstance(resume, dict) and isinstance(resume.get("config"), dict)
                   and type(resume.get("step")) is int and resume["step"] >= 25
                   and isinstance(resume.get("snapshots"), dict), "Malformed original NODE resume")
    expected = source_helper.expected_citation_config(ghost, 0, buffers["z"], buffers["q"], buffers["hard"],
                         buffers["z"].detach().float(), buffers["source"])
    for key in ("source_linear_coordinates", "source_linear_schema", "source_linear_source"):
        expected.pop(key)
    expected["assignment_input"] = "node"
    retained = resume.get("config", {}).get("save_assignment")
    probe._require(type(retained) is bool, "Malformed original artifact retention flag")
    expected["save_assignment"] = retained
    probe._require(resume["config"] == expected, "Original NODE config/current source differs")
    penalty = probe._scalar(resume["config"]["penalty"], "Malformed original head penalty")
    tolerance = probe._scalar(resume["config"]["inner_tol"], "Malformed original inner tolerance")
    probe._require(tolerance > 0, "Original head tolerance must be positive")
    material = make_material(buffers["z"], buffers["q"])
    u, v = initialize_factors(buffers["hard"], 60, 8, 0)
    baseline_u, baseline_v = initialize_factors(buffers["hard"], 60, 8, 0)
    evidence["native_initializer_calls"] = 2
    probe._require(torch.equal(u, baseline_u) and torch.equal(v, baseline_v), "Current native initializers differ")
    logits = LowRankLogits.apply(u, v, buffers["hard"], .05, 4096)
    probability = logits.double().softmax(1).detach()
    moments = LowRankMoments.apply(u, v, buffers["hard"], material, .05, 4096).detach()
    evidence["native_P0_probability_material_evaluations"] = 1
    probe._require(float((probability.sum(1)-1).abs().max()) <= 1e-12
                   and torch.allclose(moments, probability.T @ material / len(u), atol=1e-12, rtol=1e-12),
                   "Original native P0 row/material conservation differs")
    certificates, original = {}, {}
    for step in (0, 25):
        evidence["stage"] = f"cached_NODE_{step}_material_and_linear_certificate"
        snapshot = torch.load(root / "condensation_0/checkpoints" / f"step_{step:06d}.pt", map_location="cpu", weights_only=False)
        probe._require(isinstance(snapshot, dict) and type(snapshot.get("step")) is int and snapshot["step"] == step
                       and step in resume["snapshots"] and probe._seal(snapshot) == probe._seal(resume["snapshots"][step]),
                       "Original checkpoint/resume disagrees")
        saved = probe._tensor(snapshot.get("moments"), (60, 3710), torch.float64, "Malformed NODE moments").to(moments)
        probe._require(bool((saved[:, 0] > 0).all()) and torch.allclose(saved.sum(0), material.mean(0), atol=1e-12, rtol=1e-12),
                       "Original NODE material conservation differs")
        centers, labels, mass = decode_moments(saved, 3703)
        probe._require(bool((mass > 0).all()) and abs(float(mass.sum())-1) <= 1e-12
                       and bool(torch.isfinite(labels).all()) and bool((labels >= 0).all())
                       and float((labels.sum(1)-1).abs().max()) <= 1e-12,
                       "Cached material mass/simplex certificate differs")
        theta = probe._tensor(snapshot.get("theta"), (6, 3704), torch.float64, "Malformed NODE head").to(moments)
        saved_grad = probe._scalar(snapshot.get("inner_grad_max"), "Malformed cached gradient scalar")
        saved_value = probe._scalar(snapshot.get("teacher_ce"), "Malformed cached teacher CE scalar")
        grad = float(head_gradient(augmented(centers), labels, torch.full_like(mass, 1/60), theta, penalty).abs().max())
        evidence["cached_head_gradient_evaluations"] += 1
        value, _ = outer_value_gradient(buffers["z"], buffers["q"], theta, 65536, augmented(buffers["z"]))
        evidence["cached_outer_CE_evaluations"] += 1
        probe._require(snapshot.get("J_exact") is True and math.isfinite(grad) and grad <= tolerance*(1+1e-6)
                       and math.isclose(grad, saved_grad, abs_tol=1e-12, rel_tol=1e-12)
                       and math.isclose(value, saved_value, abs_tol=1e-12, rel_tol=1e-12),
                       "Cached NODE linear certificate differs")
        certificates[str(step)] = dict(head_gradient_max=grad, teacher_ce=value, snapshot=probe._digest(snapshot))
        original[step] = saved
    probe._require(torch.allclose(original[0], moments, atol=1e-12, rtol=1e-12), "Original P0 moments differ")
    actual_x, actual_q, mass = representative(moments, buffers["transform"], 3703, "cuda")
    old_x, old_q, old_mass = representative(original[0], buffers["transform"], 3703, "cuda")
    weights, old_weights = torch.full_like(mass, 1/60), torch.full_like(old_mass, 1/60)
    probe._require(all(torch.equal(a, b) for a, b in zip((actual_x, actual_q, weights), (old_x, old_q, old_weights), strict=True)),
                   "Actual FP32 P0 X/Q/uniformweights differ from cached NODE")
    return dict(manual_current_native_baseline_U0_V0_exact=True, historical_checkpoint_UV_available=False,
                P0_probability=probe._digest(probability), P0_parameters=probe._digest([u, v]), P0_moments=probe._digest(moments),
                actual_FP32_P0_X_Q_uniform_equal_reference=True, student_inputs=probe._digest([actual_x, actual_q, weights]),
                input_digest=_input_digest(actual_x, actual_q, weights, buffers["graph"], buffers["q"],
                             dict(train=buffers["train"], val=buffers["val"]), None, lambda: False),
                original_linear_certificates=certificates, artifact_retention_flag=retained,
                original_head_penalty_from_validated_resume=penalty,
                original_head_tolerance_from_validated_resume=tolerance)


def _execute(spec, root, evidence, stop):
    probe._stop(stop)
    evidence["stage"] = "current_existing_dataset_context"
    data_dir = Path(__file__).resolve().parents[1] / "data"
    graph, train, val, test, fresh_h = citation_search._prepare_dataset("citeseer", str(data_dir), "cuda", "default")
    config = citation_search._legacy_teacher_config("citeseer", .018, graph, train, val, test, "default")
    probe._require(config == json.loads((root / "config.json").read_text()) and _fingerprint(config) == ROOT,
                   "Current dataset/nodeorder/preprocessing/masks differ from original source")
    teacher = torch.load(root / "teacher.pt", map_location="cuda", weights_only=False)
    probe._require(isinstance(teacher, dict), "Malformed original ReLU teacher")
    logits = teacher.get("logits")
    probe._tensor(logits, (3327, 6), torch.float64, "Malformed original ReLU teacher logits")
    q = training_refined_targets(logits, 1., graph["y"], train, 0)
    evidence["stage"] = "observational_native_RMS_residual"
    evidence["RMS_diagnostic_before_unchanged_guard"] = _rms_diagnostic(root, "cuda")
    ghost = source_helper.candidate_controls(dict(CANDIDATE, method="source_linear", assignment_coordinates="raw_rms", source_linear_schema=1))
    # Preserve this loader's exact1e-12 contract and exact raised source error.
    evidence["stage"] = "unchanged_cached_source_strict_guard"
    h, z, transform, hard, _, source = source_helper.cached_source(root, ghost, 0, fresh_h, q, config)
    probe._require(z.shape == (3327, 3703) and q.shape == (3327, 6) and int(hard.max())+1 == 60,
                   "Original Citeseer60 source dimensions changed")
    probe._stop(stop)
    evidence["stage"] = "exact_original_CSR_dense_roundtrip"
    dense = probe._frozen_original_dense_S(graph["adj"])
    evidence["readonly_native_dense_graph_materializations"] = 1
    evidence["source_context"] = source
    evidence["native_buffers"] = probe._digest(dict(H=h, z=z, Q=q, hard=hard, transform=probe._frozen_transform(transform),
                                                     X=graph["x"], original_CSR=graph["adj"], dense_original_S=dense))
    evidence["exact_original_CSR_dense_roundtrip"] = True
    evidence["reference"] = _reference(dict(root=root, graph=graph, train=train, val=val[1], h=h, z=z,
                                     q=q, hard=hard, transform=transform, source=source), ghost, evidence)
    probe._stop(stop)
    probe._runtime_precision_guard()
    probe._checked_files(spec["files_sha256"])
    probe._require(spec["source"] == implementation_provenance() and spec["numerical_source"] == probe.numerical_source(),
                   "Source/Git/versions changed during preflight")


def prepare_preflight(spec_path, spec_sha256, output_path, stop=lambda: False):
    """New evidence only. No P updates, fits, finite unrolls or accuracy forwards."""
    repo = Path(__file__).resolve().parents[1]
    spec, root, output = _load_spec(spec_path, spec_sha256, output_path, repo)
    evidence = dict(passed=False, source=spec["source"], numerical_source=spec["numerical_source"],
                    spec_path=str(Path(spec_path).resolve()), spec_sha256=spec_sha256,
                    scientific_preregistration=spec["scientific_preregistration"], files_sha256=spec["files_sha256"],
                    fixed=FIXED, P_updates=0, head_solves=0, teacher_map_Phi_fits=0, student_fits=0,
                    optimizer_steps=0, finite_engine_unrolls=0, accuracy_forwards=0,
                    strict_source_guard_unchanged=True, cached_linear_certificate_evaluations=0)
    evidence.update(native_initializer_calls=0, native_P0_probability_material_evaluations=0,
                    cached_head_gradient_evaluations=0, cached_outer_CE_evaluations=0,
                    readonly_native_dense_graph_materializations=0)
    started, native = time.monotonic(), False
    try:
        probe._stop(stop)
        evidence["stage"] = "fresh_native_precision_policy"
        probe._require(not torch.cuda.is_initialized(), "Preflight requires fresh worker before CUDA initialization")
        evidence["native_environment"] = probe._native("cuda")
        native = True
        torch.cuda.reset_peak_memory_stats()
        _execute(spec, root, evidence, stop)
        evidence["cached_linear_certificate_evaluations"] = 2
        evidence["passed"] = True
        evidence["stage"] = "complete_native_source_preflight"
    except Exception as error:
        evidence["source_error"] = dict(type=type(error).__name__, message=str(error))
        evidence["failed_stage"] = evidence.get("stage", "pre_native_stop")
        raise
    finally:
        if native:
            torch.cuda.synchronize()
            evidence.update(CUDA_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                            CUDA_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                            CUDA_total_bytes=torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory)
        evidence["seconds"] = time.monotonic()-started
        if native:
            evidence["native_memory_capacity_passed"] = (
                evidence["CUDA_total_bytes"] == CUDA_CAPACITY
                and 0 <= evidence["CUDA_peak_allocated_bytes"] <= CUDA_CAPACITY
                and 0 <= evidence["CUDA_peak_reserved_bytes"] <= CUDA_CAPACITY)
            if not evidence["native_memory_capacity_passed"]:
                evidence["passed"] = False
                evidence["memory_error"] = "Declared physical CUDA capacity/peak bound differs"
        if evidence["seconds"] > 300:
            evidence["passed"] = False
            evidence["deadline_error"] = "Preflight exceeded declared300second runtime"
        try:
            probe._checked_files(spec["files_sha256"])
            evidence["original_assets_unchanged"] = True
        except Exception as error:
            evidence["original_assets_unchanged"] = False
            evidence["preservation_error"] = dict(type=type(error).__name__, message=str(error))
            evidence["passed"] = False
        try:
            probe._require(probe._sha(Path(spec_path).resolve()) == spec_sha256,
                           "Frozen spec changed during preflight")
            science = spec["scientific_preregistration"]
            probe._checked_files({science["path"]: science["sha256"]})
            probe._require(spec["source"] == implementation_provenance()
                           and spec["numerical_source"] == probe.numerical_source(),
                           "Source/Git/versions changed during preflight")
            evidence["source_spec_science_unchanged"] = True
            evidence["source_unchanged"] = True
        except Exception as error:
            evidence["source_spec_science_unchanged"] = False
            evidence["source_unchanged"] = False
            evidence["source_preservation_error"] = dict(type=type(error).__name__, message=str(error))
            evidence["passed"] = False
        _write_new(output, evidence)
    probe._require(evidence["passed"], "Source preservation failed; inspect new evidence")
    return dict(passed=True, validation_only=True, evidence_path=str(output), evidence_sha256=probe._sha(output),
                source=spec["source"], P_updates=0, student_fits=0, head_solves=0, finite_engine_unrolls=0)
