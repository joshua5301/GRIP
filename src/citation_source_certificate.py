"""Fixed Citeseer30/120 original-source and cached NODE algebraic certificates.

No P optimization, fits or accuracy models. The original strict loader and
all cached tolerances are preserved; failures are recorded and re-raised.
This separate module never retargets the failed Citeseer60 preflight.
"""
import json
import math
import platform
import time
from pathlib import Path

import torch

from src import citation_search
from src import finite_student_probe as probe
from src import source_linear_assignment as source_helper
from src.citation_source_preflight import _exact, _write_new
from src.evaluation import _input_digest
from src.io import _fingerprint
from src.low_rank_assignment import LowRankLogits, LowRankMoments, initialize_factors
from src.moments import augmented, decode_moments, make_material
from src.research_loop import implementation_provenance
from src.soft_ce_partition import head_gradient, outer_value_gradient
from src.sweep_utils import representative
from src.target_refinement import training_refined_targets

SCIENCE = "Citeseer30_120_original_source_certificate_scientific_stageAT_v1.json"
SCIENCE_SHA = "83ab2499facf97b9ab43f180a34969fba107cce9a4a46fb0cdb6dda683d507c7"
CUDA_CAPACITY = 8316977152
FIXED = dict(dataset="citeseer", citation_features="row", condensation_seed=0, device="cuda", nodes=3327, dimension=3703,
             classes=6, native_policy="deterministic_cuda_v1", P_updates=0, student_fits=0)
REFERENCE = "a2d47970c967"
CANDIDATE = dict(method="low_rank", width=0, lr=.01, T=1., rank=8, penalty=.001,
                 initialization="teacher_joint", alpha=.3, inner_loss_weighting="uniform")
ROOTS = ((30, .009, "ca115f4751a1"), (120, .036, "2e2127c2bc3f"))


def _roots(repo):
    return [dict(cells=cells, ratio=ratio, root=key,
                 source_root=str(repo / "results/citation_search_v1/citeseer" / f"ratio_{ratio}" / key))
            for cells, ratio, key in ROOTS]


def _paths(repo):
    names = ("config.json", "propagated_H.pt", "inputs_0.pt", "teacher.pt", "assignment_20199efc7d26.pt",
             "nystrom_map_schema3.pt", "nystrom_phi_schema3.npy", "nystrom_phi_schema3.meta.json")
    files = [Path(root["source_root"]) / name for root in _roots(repo) for name in names]
    for root in _roots(repo):
        base = Path(root["source_root"]) / REFERENCE
        files += [base / name for name in ("candidate.json", "condensation_0/resume.pt",
                  "condensation_0/checkpoints/step_000000.pt", "condensation_0/checkpoints/step_000025.pt")]
    files += [repo / "data/citeseer/raw" / f"ind.citeseer.{name}" for name in
              ("x", "tx", "allx", "y", "ty", "ally", "graph", "test.index")]
    files += sorted((repo / "data/citeseer/processed").glob("*.pt"))
    return files


def _config(root):
    config = json.loads((Path(root["source_root"]) / "config.json").read_text())
    probe._require(isinstance(config, dict) and _fingerprint(config) == root["root"]
                   and config.get("dataset") == "citeseer" and type(config.get("ratio")) is float
                   and config["ratio"] == root["ratio"]
                   and config.get("citation_features", "default") == FIXED["citation_features"],
                   "Original config/root fingerprint or preprocessing differs")
    return config


def _load_spec(path, digest, output, repo):
    path, output = Path(path).resolve(), Path(output).resolve()
    probe._require(path.is_relative_to(repo / "results/proposals") and path.is_file()
                   and probe._sha(path) == digest, "Require frozen prospective spec path/SHA")
    spec = json.loads(path.read_text())
    fields = {"schema", "fixed", "roots", "candidate", "reference_candidate_id", "source", "numerical_source", "python_version", "files_sha256",
              "scientific_preregistration", "output_path"}
    roots = _roots(repo)
    probe._require(isinstance(spec, dict) and set(spec) == fields and type(spec["schema"]) is int
                   and spec["schema"] == 1 and _exact(spec["fixed"], FIXED)
                   and _exact(spec["candidate"], CANDIDATE) and spec["reference_candidate_id"] == REFERENCE
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
                   and _exact(frozen_science.get("candidate"), CANDIDATE)
                   and frozen_science.get("reference_candidate_id") == REFERENCE
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


def _reference(buffers, ghost, cells, evidence):
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
    evidence["native_initializer_calls"] += 1
    u, v = initialize_factors(buffers["hard"], cells, 8, 0)
    evidence["native_initializer_completed_calls"] += 1
    evidence["native_initializer_calls"] += 1
    baseline_u, baseline_v = initialize_factors(buffers["hard"], cells, 8, 0)
    evidence["native_initializer_completed_calls"] += 1
    probe._require(torch.equal(u, baseline_u) and torch.equal(v, baseline_v), "Current native initializers differ")
    logits = LowRankLogits.apply(u, v, buffers["hard"], .05, 4096)
    probability = logits.double().softmax(1).detach()
    moments = LowRankMoments.apply(u, v, buffers["hard"], material, .05, 4096).detach()
    evidence["native_P0_probability_material_evaluations"] += 1
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
        saved = probe._tensor(snapshot.get("moments"), (cells, 3710), torch.float64, "Malformed NODE moments").to(moments)
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
        grad = float(head_gradient(augmented(centers), labels, torch.full_like(mass, 1/cells), theta, penalty).abs().max())
        evidence["cached_head_gradient_evaluations"] += 1
        value, _ = outer_value_gradient(buffers["z"], buffers["q"], theta, 65536, augmented(buffers["z"]))
        evidence["cached_outer_CE_evaluations"] += 1
        probe._require(snapshot.get("J_exact") is True and math.isfinite(grad) and grad <= tolerance
                       and math.isclose(grad, saved_grad, abs_tol=1e-12, rel_tol=1e-12)
                       and math.isclose(value, saved_value, abs_tol=1e-12, rel_tol=1e-12),
                       "Cached NODE linear certificate differs")
        certificates[str(step)] = dict(head_gradient_max=grad, teacher_ce=value, snapshot=probe._digest(snapshot))
        original[step] = saved
    probe._require(torch.allclose(original[0], moments, atol=1e-12, rtol=1e-12), "Original P0 moments differ")
    actual_x, actual_q, mass = representative(moments, buffers["transform"], 3703, "cuda")
    old_x, old_q, old_mass = representative(original[0], buffers["transform"], 3703, "cuda")
    weights, old_weights = torch.full_like(mass, 1/cells), torch.full_like(old_mass, 1/cells)
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


def _execute_root(row, evidence, stop):
    probe._stop(stop)
    root, cells = Path(row["source_root"]), row["cells"]
    evidence["stage"] = "current_existing_dataset_" + row["root"]
    data_dir = Path(__file__).resolve().parents[1] / "data"
    graph, train, val, test, fresh_h = citation_search._prepare_dataset("citeseer", str(data_dir), "cuda", "row")
    config = citation_search._legacy_teacher_config("citeseer", row["ratio"], graph, train, val, test, "row")
    probe._require(config == _config(row) and _fingerprint(config) == row["root"],
                   "Current dataset/nodeorder/preprocessing/masks differ from original source")
    teacher = torch.load(root / "teacher.pt", map_location="cuda", weights_only=False)
    probe._require(isinstance(teacher, dict), "Malformed original ReLU teacher")
    logits = probe._tensor(teacher.get("logits"), (3327, 6), torch.float64, "Malformed original ReLU logits")
    q = training_refined_targets(logits, 1., graph["y"], train, 0)
    ghost = source_helper.candidate_controls(dict(CANDIDATE, method="source_linear", assignment_coordinates="raw_rms", source_linear_schema=1))
    evidence["stage"] = "unchanged_cached_source_guard_" + row["root"]
    h, z, transform, hard, _, source = source_helper.cached_source(root, ghost, 0, fresh_h, q, config)
    probe._require(z.shape == (3327, 3703) and q.shape == (3327, 6) and int(hard.max())+1 == cells,
                   "Original fixed Citeseer source dimensions changed")
    probe._stop(stop)
    evidence["stage"] = "exact_original_CSR_dense_roundtrip_" + row["root"]
    dense = probe._frozen_original_dense_S(graph["adj"])
    evidence["readonly_native_dense_graph_materializations"] += 1
    result = dict(**row, source_context=source, exact_original_CSR_dense_roundtrip=True,
                  native_buffers=probe._digest(dict(H=h, z=z, Q=q, hard=hard, transform=probe._frozen_transform(transform),
                                      X=graph["x"], original_CSR=graph["adj"], dense_original_S=dense)))
    result["reference"] = _reference(dict(root=root, graph=graph, train=train, val=val[1], h=h, z=z,
                                        q=q, hard=hard, transform=transform, source=source), ghost, cells, evidence)
    probe._stop(stop)
    result["source_P0_linear_certificate_passed"] = True
    return result


def prepare_certificate(spec_path, spec_sha256, output_path, stop=lambda: False):
    """Fixed original source/P0/linear certificates only; zero fits or P updates."""
    repo = Path(__file__).resolve().parents[1]
    spec, parents, output = _load_spec(spec_path, spec_sha256, output_path, repo)
    evidence = dict(passed=False, source=spec["source"],
                    numerical_source=spec["numerical_source"], python_version=spec["python_version"],
                    spec_path=str(Path(spec_path).resolve()), spec_sha256=spec_sha256,
                    scientific_preregistration=spec["scientific_preregistration"], parents=parents,
                    files_sha256=spec["files_sha256"], roots={}, native_initializer_calls=0, native_initializer_completed_calls=0,
                    native_P0_probability_material_evaluations=0, cached_head_gradient_evaluations=0,
                    cached_outer_CE_evaluations=0, readonly_native_dense_graph_materializations=0,
                    P_updates=0, optimizer_steps=0, student_fits=0, head_solves=0, teacher_map_Phi_fits=0,
                    model_forwards=0, finite_unrolls=0, strict_source_guard_unchanged=True)
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
            evidence["active_root"] = dict(root)
            evidence["stage"] = "original_source_certificate_" + root["root"]
            evidence["roots"][root["root"]] = _execute_root(root, evidence, stop)
        probe._require((evidence["native_initializer_calls"], evidence["native_initializer_completed_calls"],
                        evidence["native_P0_probability_material_evaluations"],
                        evidence["cached_head_gradient_evaluations"], evidence["cached_outer_CE_evaluations"],
                        evidence["readonly_native_dense_graph_materializations"]) == (4, 4, 2, 4, 4, 2), "Unexpected certificate counts")
        evidence["passed"], evidence["stage"] = True, "both_fixed_original_source_certificates_passed"
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
            probe._require(probe._sha(Path(spec_path).resolve()) == spec_sha256, "Spec changed during certificates")
            _source_unchanged(spec)
            if native:
                probe._runtime_precision_guard()
                probe._require("memory_error" not in evidence and torch.get_num_threads() == 4
                               and evidence["CUDA_total_bytes"] == CUDA_CAPACITY
                               and 0 <= evidence["CUDA_peak_allocated_bytes"] <= CUDA_CAPACITY
                               and 0 <= evidence["CUDA_peak_reserved_bytes"] <= CUDA_CAPACITY,
                               "Native threads/capacity/peak policy changed")
            probe._require(evidence["seconds"] <= 300, "Certificates exceeded declared300seconds")
            evidence["source_assets_spec_science_unchanged"] = True
        except Exception as error:
            evidence["passed"] = False
            evidence["source_assets_spec_science_unchanged"] = False
            evidence["preservation_error"] = dict(type=type(error).__name__, message=str(error))
        _write_new(output, evidence)
    probe._require(evidence["passed"], "Original source certificate integrity failed; inspect new evidence")
    return dict(passed=True, validation_only=True, evidence_path=str(output),
                evidence_sha256=probe._sha(output), source=spec["source"], P_updates=0, student_fits=0,
                head_solves=0, native_initializer_calls=4, native_initializer_completed_calls=4, cached_linear_certificate_evaluations=4)
