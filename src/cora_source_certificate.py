"""Three fixed Cora ROW original-source/P0/cached-linear certificates only.

No optimization candidate, fits, model forwards or cache repair. Current native
M0 bitwise equality is descriptive; strict1e-12 material and exact FP32 student
inputs plus supplied FP64 uniform weights are mandatory.
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

SCIENCE = "Cora35_70_140_original_source_certificate_scientific_stageBB_v2.json"
SCIENCE_SHA = "d3bd9a6915544728b454ed3559d610968ab46fa8a6984f95052b7c013aed9211"
CUDA_CAPACITY = 8316977152
FIXED = dict(dataset="cora", citation_features="row", nodes=2708, dimension=1433, classes=7,
             validation_nodes=500, condensation_seed=0, device="cuda")
_CASES = ((35,.013,"a3fddd7d4e0f","d7bf98a19228",.1,"fb15faa95499"),
          (70,.026,"19fccc37cc2f","c24c1ffc2b76",.05,"1c11249d043f"),
          (140,.052,"b23d5978e22a","d7bf98a19228",.1,"2953d51f450a"))
_require, _sha = probe._require, probe._sha


def numerical_source():
    return probe.numerical_source()


def _science(repo):
    path = repo / "results/proposals" / SCIENCE
    probe._checked_files({str(path): SCIENCE_SHA})
    return json.loads(path.read_text())


def _cases(repo):
    return _science(repo)["cases"]


def _paths(repo):
    return [Path(p) for p in _science(repo)["original_files_sha256"]]


def _validate_cases(cases, repo):
    _require(isinstance(cases,list) and len(cases)==3,"Require all three fixed Cora cases")
    for row,(cells,ratio,key,cid,lr,recipe) in zip(cases,_CASES,strict=True):
        expected = dict(cells=cells,ratio=ratio,root=key,source_root=str(repo/"results/citation_search_v1/cora"/f"ratio_{ratio}"/key),
                        condensation_seed=0,reference_id=cid,recipe_id=recipe,hard_path="assignment_35b4ed94ab71.pt")
        candidate = dict(method="low_rank",width=0,lr=lr,T=.3,rank=32,penalty=.0001,
                         initialization="teacher_balanced",alpha=.3,inner_loss_weighting="uniform")
        _require(isinstance(row,dict) and _exact({k:row.get(k) for k in expected},expected)
                 and _exact(row.get("reference_candidate"),candidate),"Changed fixed Cora source/reference/recipe controls")


def _source_unchanged(spec):
    _require(implementation_provenance()==spec["source"] and numerical_source()==spec["numerical_source"]
             and platform.python_version()==spec["python_version"],"Current source/Git/Python/versions changed")


def _preserve(spec,path,checksum,science):
    _require(_sha(path)==checksum,"Frozen spec changed")
    probe._checked_files(spec["files_sha256"])
    probe._checked_files(spec["artifacts_sha256"])
    probe._checked_files({spec["scientific_preregistration"]["path"]:SCIENCE_SHA})
    probe._checked_files({p["path"]:p["sha256"] for p in science["parents"]})
    previous=science["preregistration_revision"]["previous_unexecuted"]
    probe._checked_files({previous["path"]:previous["sha256"]})
    probe._checked_files({p:h for p,h in science["readonly_helper_source_pins"].items() if Path(p).name!="research_loop.py"})
    _source_unchanged(spec)


def _load_spec(path,checksum,output,repo):
    path,output=Path(path).resolve(),Path(output).resolve()
    _require(type(checksum) is str and len(checksum)==64 and path.is_relative_to(repo/"results/proposals")
             and path.is_file() and _sha(path)==checksum,"Require exact frozen Cora spec path/SHA")
    science=_science(repo)
    _require(_exact(science.get("fixed"),FIXED),"Frozen Cora fixed controls changed")
    _validate_cases(science["cases"],repo)
    spec=json.loads(path.read_text())
    fields={"schema","fixed","cases","source","numerical_source","python_version","files_sha256",
            "scientific_preregistration","artifacts_sha256","output_path"}
    _require(isinstance(spec,dict) and set(spec)==fields and type(spec["schema"]) is int and spec["schema"]==1
             and _exact(spec["fixed"],FIXED) and _exact(spec["cases"],science["cases"]),"Unknown or changed fixed Cora spec/cases")
    _require(spec["files_sha256"]==science["original_files_sha256"] and
             spec["scientific_preregistration"]==dict(path=str(repo/"results/proposals"/SCIENCE),sha256=SCIENCE_SHA),
             "Original inventory/scientific lineage changed")
    _require(isinstance(spec["artifacts_sha256"],dict) and spec["artifacts_sha256"] and
             all(Path(p).is_absolute() and Path(p).resolve().is_relative_to(repo/"results") for p in spec["artifacts_sha256"]),
             "Require reviewed immutable artifact pins")
    expected=repo/"results/research_loop/Cora35_70_140_original_source_certificate_stageBB_v1/native_source_certificate_v1.json"
    _require(type(spec["output_path"]) is str and output==expected and output==Path(spec["output_path"])
             and not output.exists(),"Require absent declared new Cora evidence")
    _require((repo/"data/cora/processed/data.pt").is_file(),"Require existing processed dataset")
    _preserve(spec,path,checksum,science)
    for case in spec["cases"]:
        _config(case)
        _recipe(case)
    return spec,science,output


def _config(case):
    config=json.loads((Path(case["source_root"])/"config.json").read_text())
    _require(_exact(config,case["source_config"]) and _fingerprint(config)==case["root"],"Original Cora config/root differs")
    return config


def _recipe(case):
    path=Path(case["source_root"])/f"student_recipe_{case['recipe_id']}.json"
    value=json.loads(path.read_text())
    _require(_exact(value,case["recipe"]) and _fingerprint(value)==case["recipe_id"]
             and type(value.get("input_scale")) is float and value["input_scale"]==1.,"Own original recipe/identity input scale differs")
    return dict(path=str(path),sha256=_sha(path),recipe_id=case["recipe_id"],contents=value,
                scope="Recipe explicitly bound separately; input_digest does not hash recipe")


def _attempt(evidence,key):
    evidence["operation_attempts"][key]=evidence["operation_attempts"].get(key,0)+1


def _complete(evidence,key):
    evidence["counts"][key]+=1


def _shared_digest(shared):
    return probe._digest(dict(X=shared["graph"]["x"], original_CSR=shared["graph"]["adj"],
                              dense_original_S=shared["dense"], fresh_H=shared["fresh_h"],
                              train=shared["train"], validation=shared["val"], testing_mask=shared["test"][1]))


def _reference(buffers, ghost, case, evidence):
    cells=case["cells"]
    evidence["stage"] = "cached_NODE_config_and_native_P0"
    root = buffers["root"] / case["reference_id"]
    probe._require(json.loads((root / "candidate.json").read_text()) == case["reference_candidate"], "Original NODE candidate changed")
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
    evidence["counts"]["factor_factory_attempts"] += 1
    u, v = initialize_factors(buffers["hard"], cells, 32, 0)
    evidence["counts"]["factor_factory_completed"] += 1
    evidence["counts"]["factor_factory_attempts"] += 1
    baseline_u, baseline_v = initialize_factors(buffers["hard"], cells, 32, 0)
    evidence["counts"]["factor_factory_completed"] += 1
    probe._require(torch.equal(u, baseline_u) and torch.equal(v, baseline_v), "Current native initializers differ")
    _attempt(evidence,"P0_logit_evaluations")
    logits = LowRankLogits.apply(u, v, buffers["hard"], .05, 4096)
    _complete(evidence,"P0_logit_evaluations")
    probability = logits.double().softmax(1).detach()
    _attempt(evidence,"P0_material_moment_evaluations")
    moments = LowRankMoments.apply(u, v, buffers["hard"], material, .05, 4096).detach()
    _complete(evidence,"P0_material_moment_evaluations")
    probe._require(float((probability.sum(1)-1).abs().max()) <= 1e-12
                   and torch.allclose(moments, probability.T @ material / len(u), atol=1e-12, rtol=1e-12),
                   "Original native P0 row/material conservation differs")
    certificates, saved_moments = {}, {}
    for step in (0, 25):
        evidence["stage"] = f"cached_NODE_{step}_material_and_linear_certificate"
        snapshot = torch.load(root / "condensation_0/checkpoints" / f"step_{step:06d}.pt", map_location="cpu", weights_only=False)
        probe._require(isinstance(snapshot, dict) and type(snapshot.get("step")) is int and snapshot["step"] == step
                       and step in resume["snapshots"] and probe._seal(snapshot) == probe._seal(resume["snapshots"][step]),
                       "Original checkpoint/resume disagrees")
        saved = probe._tensor(snapshot.get("moments"), (cells, 1441), torch.float64, "Malformed NODE moments").to(moments)
        probe._require(bool((saved[:, 0] > 0).all()) and torch.allclose(saved.sum(0), material.mean(0), atol=1e-12, rtol=1e-12),
                       "Original NODE material conservation differs")
        centers, labels, mass = decode_moments(saved, 1433)
        probe._require(bool((mass > 0).all()) and abs(float(mass.sum())-1) <= 1e-12
                       and bool(torch.isfinite(labels).all()) and bool((labels >= 0).all())
                       and float((labels.sum(1)-1).abs().max()) <= 1e-12,
                       "Cached material mass/simplex certificate differs")
        theta = probe._tensor(snapshot.get("theta"), (7, 1434), torch.float64, "Malformed NODE head").to(moments)
        saved_grad = probe._scalar(snapshot.get("inner_grad_max"), "Malformed cached gradient scalar")
        saved_value = probe._scalar(snapshot.get("teacher_ce"), "Malformed cached teacher CE scalar")
        _attempt(evidence,"cached_head_gradients")
        grad = float(head_gradient(augmented(centers), labels, torch.full_like(mass, 1/cells), theta, penalty).abs().max())
        _complete(evidence,"cached_head_gradients")
        _attempt(evidence,"cached_outer_teacher_CE")
        value, _ = outer_value_gradient(buffers["z"], buffers["q"], theta, 65536, augmented(buffers["z"]))
        _complete(evidence,"cached_outer_teacher_CE")
        probe._require(snapshot.get("J_exact") is True and math.isfinite(grad) and grad <= tolerance
                       and math.isclose(grad, saved_grad, abs_tol=1e-12, rel_tol=1e-12)
                       and math.isclose(value, saved_value, abs_tol=1e-12, rel_tol=1e-12),
                       "Cached NODE linear certificate differs")
        certificates[str(step)] = dict(head_gradient_max=grad, teacher_ce=value, snapshot=probe._digest(snapshot))
        saved_moments[step] = saved
    probe._require(torch.allclose(saved_moments[0], moments, atol=1e-12, rtol=1e-12), "Original P0 moments differ")
    actual_x, actual_q, mass = representative(moments, buffers["transform"], 1433, "cuda")
    old_x, old_q, old_mass = representative(saved_moments[0], buffers["transform"], 1433, "cuda")
    weights, old_weights = torch.full_like(mass, 1/cells), torch.full_like(old_mass, 1/cells)
    probe._require(all(torch.equal(a, b) for a, b in zip((actual_x, actual_q, weights), (old_x, old_q, old_weights), strict=True)),
                   "Actual FP32 P0 X/Q/uniformweights differ from cached NODE")
    return dict(manual_current_native_baseline_U0_V0_exact=True,
                current_native_M0_bitwise_equal_cached_NODE0=bool(torch.equal(saved_moments[0],moments)),
                current_native_M0_vs_cached_NODE0_max_abs=float((saved_moments[0]-moments).abs().max()),
                current_native_M0_allclose_1e12_cached_NODE0=True, historical_checkpoint_UV_available=False,
                P0_probability=probe._digest(probability), P0_parameters=probe._digest([u, v]), P0_moments=probe._digest(moments),
                actual_FP32_P0_X_Q_uniform_equal_reference=True, student_inputs=probe._digest([actual_x, actual_q, weights]),
                input_digest=_input_digest(actual_x, actual_q, weights, buffers["graph"], buffers["q"],
                             dict(train=buffers["train"], val=buffers["val"]), None, lambda: False),
                original_linear_certificates=certificates, artifact_retention_flag=retained,
                original_head_penalty_from_validated_resume=penalty,
                original_head_tolerance_from_validated_resume=tolerance)


def _prepare_graph(repo,evidence,stop):
    probe._stop(stop)
    evidence["stage"]="original_existing_dataset_and_CSR"
    _attempt(evidence,"dataset_preparations")
    graph,train,val,test,fresh_h=citation_search._prepare_dataset("cora",str(repo/"data"),"cuda","row")
    _complete(evidence,"dataset_preparations")
    _require(torch.is_tensor(val[1]) and val[1].dtype==torch.bool and val[1].shape==(2708,) and int(val[1].sum())==500,
             "Require exact original500-node validation mask")
    _attempt(evidence,"dense_S_materializations")
    dense=probe._frozen_original_dense_S(graph["adj"])
    _complete(evidence,"dense_S_materializations")
    return dict(graph=graph,train=train,val=val[1],test=test,fresh_h=fresh_h,dense=dense)


def _execute_case(case,shared,evidence,stop):
    probe._stop(stop)
    root,cells=Path(case["source_root"]),case["cells"]
    graph=shared["graph"]
    config=citation_search._legacy_teacher_config("cora",case["ratio"],graph,shared["train"],
                                                  (None,shared["val"]),shared["test"],"row")
    _require(_exact(config,_config(case)) and _fingerprint(config)==case["root"],"Current Cora graph/nodeorder/masks/preprocessing changed")
    recipe=_recipe(case)
    evidence["stage"]="unchanged_cached_source_guard_"+case["root"]
    teacher=torch.load(root/"teacher.pt",map_location="cuda",weights_only=False)
    _require(isinstance(teacher,dict),"Malformed original teacher")
    logits=probe._tensor(teacher.get("logits"),(2708,7),torch.float64,"Malformed original Cora teacher logits")
    q=training_refined_targets(logits,case["reference_candidate"]["T"],graph["y"],shared["train"],0)
    ghost=source_helper.candidate_controls(dict(case["reference_candidate"],method="source_linear",
                        assignment_coordinates="raw_rms",source_linear_schema=1))
    _attempt(evidence,"per_root_cached_source_calls")
    h,z,transform,hard,_,source=source_helper.cached_source(root,ghost,0,shared["fresh_h"],q,config)
    _complete(evidence,"per_root_cached_source_calls")
    _require(z.shape==(2708,1433) and q.shape==(2708,7) and int(hard.max())+1==cells,"Original Cora shape/budget changed")
    _attempt(evidence,"per_root_CSR_dense_binding_checks")
    _require(probe._seal(graph["adj"])==probe._seal(shared["dense"].to_sparse_csr()),"Original CSR/dense support or values changed")
    _complete(evidence,"per_root_CSR_dense_binding_checks")
    buffers=dict(root=root,graph=graph,train=shared["train"],val=shared["val"],h=h,z=z,q=q,hard=hard,
                 transform=transform,source=source)
    native=probe._digest(dict(H=h,z=z,Q=q,hard=hard,transform=probe._frozen_transform(transform),X=graph["x"],
                              original_CSR=graph["adj"],dense_original_S=shared["dense"]))
    reference=_reference(buffers,ghost,case,evidence)
    _require(native==probe._digest(dict(H=h,z=z,Q=q,hard=hard,transform=probe._frozen_transform(transform),X=graph["x"],
                              original_CSR=graph["adj"],dense_original_S=shared["dense"])),"Native source buffers changed")
    return dict(passed=True,cells=cells,root=case["root"],source_context=source,native_buffers=native,reference=reference,
                student_recipe_origin=recipe,source_P0_linear_certificate_passed=True,
                exact_original_CSR_dense_roundtrip=True,source_buffer_unchanged=True,
                future_gradient_target_or_objective_qualified=False)


def prepare_certificate(spec_path,spec_sha256,output_path,stop=lambda:False):
    """One fresh process visits35/70/140, no fitting or scientific P updates."""
    _require(callable(stop),"Require stop callback")
    repo=Path(__file__).resolve().parents[1]
    spec,science,output=_load_spec(spec_path,spec_sha256,output_path,repo)
    expected=science["expected_success_counts"]
    evidence=dict(passed=False,source=spec["source"],numerical_source=spec["numerical_source"],python_version=spec["python_version"],
                  spec_path=str(Path(spec_path).resolve()),spec_sha256=spec_sha256,scientific_preregistration=spec["scientific_preregistration"],
                  files_sha256=spec["files_sha256"],parents=science["parents"],counts={k:0 for k in expected},operation_attempts={},roots={},case_visit_order=[],
                  strict_source_guard_unchanged=True,test_enabled=False,no_optimization_candidate=True,
                  heldout_labels_scope="Immutable original graph identity digest only; never loss, accuracy or selection")
    native,primary,shared=False,None,None
    started=time.monotonic()
    bounded=lambda:stop() or time.monotonic()-started>=300
    try:
        probe._stop(bounded)
        evidence["stage"]="fresh_native_policy"
        _require(torch.get_num_threads()==4 and not torch.cuda.is_initialized(),"Require threads4/fresh CUDA worker")
        evidence["native_environment"]=dict(**probe._native("cuda"),python_version=platform.python_version(),threads=4)
        native=True
        torch.cuda.reset_peak_memory_stats()
        _require(torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory==CUDA_CAPACITY,"GPU capacity changed")
        shared=_prepare_graph(repo,evidence,bounded)
        evidence["shared_graph_buffers_before"]=_shared_digest(shared)
        for case in spec["cases"]:
            probe._stop(bounded);probe._runtime_precision_guard()
            evidence["active_root"]=case["root"]
            evidence["case_visit_order"].append(case["cells"])
            evidence["roots"][case["root"]]=_execute_case(case,shared,evidence,bounded)
        _require({k:evidence["counts"][k] for k in expected}==expected,"Unexpected source certificate counts")
        evidence.update(passed=True,stage="all_three_original_source_certificates_passed")
    except BaseException as error:
        primary=error
        evidence.update(source_error=dict(type=type(error).__name__,message=str(error)),failed_stage=evidence.get("stage","pre_native"))
    finally:
        if shared is not None:
            try:
                evidence["shared_graph_buffers_after"]=_shared_digest(shared)
                _require(evidence["shared_graph_buffers_before"]==evidence["shared_graph_buffers_after"],"Shared graph/masks changed")
            except BaseException as error:
                evidence.update(passed=False,native_buffer_preservation_error=dict(type=type(error).__name__,message=str(error)))
                if primary is None:primary=error
        if native:
            try:
                torch.cuda.synchronize()
                total=torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
                allocated,reserved=torch.cuda.max_memory_allocated(),torch.cuda.max_memory_reserved()
                evidence.update(CUDA_peak_allocated_bytes=allocated,CUDA_peak_reserved_bytes=reserved,CUDA_total_bytes=total)
                probe._runtime_precision_guard()
                _require(torch.get_num_threads()==4 and total==CUDA_CAPACITY and 0<=allocated<=total and 0<=reserved<=total,
                         "Native threads/capacity/memory policy changed")
            except BaseException as error:
                evidence.update(passed=False,memory_error=dict(type=type(error).__name__,message=str(error)))
                if primary is None:primary=error
        evidence["seconds"]=time.monotonic()-started
        try:
            _preserve(spec,Path(spec_path).resolve(),spec_sha256,science)
            _require(evidence["seconds"]<=300,"Certificates exceeded300seconds")
            evidence["source_assets_spec_science_unchanged"]=True
        except BaseException as error:
            evidence.update(passed=False,source_assets_spec_science_unchanged=False,
                            preservation_error=dict(type=type(error).__name__,message=str(error)))
            if primary is None:primary=error
        try:
            _write_new(output,evidence)
        except BaseException:
            if primary is not None:raise primary
            raise
    if primary is not None:raise primary
    return dict(evidence,evidence_path=str(output),evidence_sha256=_sha(output),validation_only=True)
