"""Independent Citeseer120 cond1/2 replication of frozen gradient alignment.

Protected AZ mathematics, Adam validation and final student engines are reused.
Each new condensation first certifies its original source/NODE0/25, then starts
fresh. No old source, cache or numerical module is retargeted or monkeypatched.
"""
import json
import math
import platform
import time
from pathlib import Path

import torch

from src import citation_gradient_fixed25 as az
from src import citation_gradient_probe as ay
from src import citation_search
from src import citation_source_certificate as original
from src import citeseer_finite_student_v2 as inherited
from src import finite_student_probe as probe
from src import source_linear_assignment as source_helper
from src.ce_gradient_alignment import POLICY
from src.citation_source_preflight import _exact, _write_new
from src.evaluation import _input_digest
from src.io import _fingerprint
from src.low_rank_assignment import LowRankLogits, LowRankMoments, initialize_factors
from src.moments import augmented, decode_moments, make_material
from src.research_loop import implementation_provenance
from src.soft_ce_partition import head_gradient, outer_value_gradient
from src.sweep_utils import representative
from src.target_refinement import training_refined_targets

SCIENCE = "Citeseer120_CE_gradient_alignment_fixed25_replication_scientific_stageBA_v2.json"
SCIENCE_SHA = "765b0b29f7675e76f3930a9e355481a78b9f5e79660f47a202b70e3f9d8ee3a1"
HORIZON = 25
_SEEDS = (4000, 4001, 4002)
_COMMON = ("H", "z", "Q", "transform", "X", "original_CSR", "dense_original_S")
_require, _stop, _tensor, _sha, _seal = probe._require, probe._stop, probe._tensor, probe._sha, probe._seal
_count = inherited._count
_state, _check_progress, _verify_targets, _store = az._state, az._check_progress, az._verify_targets, az._store
_cache_files = az._cache_files


def _cond(value):
    _require(type(value) is int and value in (1, 2), "Only independent condensation1/2 is supported")
    return value


def _roots(repo):
    return [row for row in original._roots(repo) if row["cells"] == 120]


def numerical_source():
    result = az.numerical_source()
    _require(result["files"].get("citation_gradient_replication.py") == _sha(__file__), "BA module source changed")
    return result


def canonical_candidate(condensation_seed):
    c = _cond(condensation_seed)
    fixed = dict(az._FIXED, method="frozen_GCN_CE_gradient_alignment_fixed25_replication",
                 replication_schema=1, condensation_seed=c)
    return dict(fixed, gradient_source_digest=_seal(numerical_source()))



def _preserve(spec, path, checksum, science):
    _require(_sha(path) == checksum, "Frozen BA spec changed")
    probe._checked_files(spec["files_sha256"])
    probe._checked_files(spec["artifacts_sha256"])
    probe._checked_files({spec["scientific_preregistration"]["path"]: SCIENCE_SHA})
    probe._checked_files({row["path"]: row["sha256"] for row in science["parents"]})
    _require(implementation_provenance() == spec["source"] and numerical_source() == spec["numerical_source"]
             and platform.python_version() == spec["python_version"], "Source/Git/Python/versions changed")
    _require(torch.get_num_threads() == 4, "Require unchanged threads4")


def _load_spec(path, checksum):
    repo = Path(__file__).resolve().parents[1]
    path = Path(path).resolve()
    _require(type(checksum) is str and len(checksum) == 64 and path.is_file()
             and path.is_relative_to(repo / "results/proposals") and _sha(path) == checksum, "Require frozen BA spec path/SHA")
    spec = json.loads(path.read_text())
    fields = {"schema", "scientific_preregistration", "source", "numerical_source", "python_version", "files_sha256",
              "roots", "candidates", "candidate_ids", "artifacts_sha256", "output_root", "certificate_outputs", "gradient_policy", "condensation_seeds"}
    candidates = {str(c): canonical_candidate(c) for c in (1, 2)}
    _require(isinstance(spec, dict) and set(spec) == fields and type(spec["schema"]) is int and spec["schema"] == 1
             and _exact(spec["candidates"], candidates) and spec["candidate_ids"] == {c:_fingerprint(v) for c,v in candidates.items()}
             and _exact(spec["gradient_policy"], json.loads(json.dumps(POLICY)))
             and _exact(spec["condensation_seeds"], [1, 2]) and spec["roots"] == _roots(repo),
             "Changed BA candidates/condensations/roots/schema or unknown controls")
    reference = dict(path=str(repo / "results/proposals" / SCIENCE), sha256=SCIENCE_SHA)
    _require(spec["scientific_preregistration"] == reference, "Wrong fixed BA science")
    probe._checked_files({reference["path"]: SCIENCE_SHA})
    science = json.loads(Path(reference["path"]).read_text())
    fixed = {c:{k:v for k,v in row.items() if k != "gradient_source_digest"} for c,row in candidates.items()}
    _require(_exact(science["candidates"], fixed) and spec["files_sha256"] == science["original_files_sha256"]
             and science["evaluation"]["student_seeds"] == list(_SEEDS), "Changed BA science or original assets")
    _require(isinstance(spec["artifacts_sha256"], dict) and spec["artifacts_sha256"]
             and all(Path(p).resolve().is_relative_to(repo / "results") for p in spec["artifacts_sha256"]), "Require reviewed artifact pins")
    root = Path(spec["output_root"]).resolve()
    _require(str(root) == spec["output_root"] and root.is_relative_to(repo / "results/research_loop"), "Invalid BA output root")
    _require(spec["certificate_outputs"] == {str(c):str(root/f"condensation_{c}/native_certificate25_v1.json") for c in (1,2)},
             "Wrong declared condensation-specific native25 outputs")
    _preserve(spec, path, checksum, science)
    return spec, science


def _folder(spec, condensation_seed):
    c = _cond(condensation_seed)
    return Path(spec["output_root"]) / f"condensation_{c}" / spec["candidate_ids"][str(c)]


def _common_certificate(buffers, certified):
    actual = probe._digest(dict(H=buffers["h"], z=buffers["z"], Q=buffers["q"], transform=buffers["transform"],
                      X=buffers["x"], original_CSR=buffers["original_S"], dense_original_S=buffers["S"]))
    _require(isinstance(certified, dict) and set(certified) == set(_COMMON)|{"hard"}
             and _exact(actual, {key:certified[key] for key in _COMMON}),
             "Actual common source differs from original AT120 (exclude only hard)")
    return actual



def _reference(buffers, ghost, condensation_seed, evidence):
    c = _cond(condensation_seed)
    cells = 120
    evidence["stage"] = "cached_NODE_config_and_native_P0"
    root = buffers["root"] / original.REFERENCE
    probe._require(json.loads((root / "candidate.json").read_text()) == original.CANDIDATE, "Original NODE candidate changed")
    resume = torch.load(root / f"condensation_{c}/resume.pt", map_location="cpu", weights_only=False)
    probe._require(isinstance(resume, dict) and isinstance(resume.get("config"), dict)
                   and type(resume.get("step")) is int and resume["step"] >= 25
                   and isinstance(resume.get("snapshots"), dict), "Malformed original NODE resume")
    expected = source_helper.expected_citation_config(ghost, c, buffers["z"], buffers["q"], buffers["hard"],
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
    u, v = initialize_factors(buffers["hard"], cells, 8, c)
    evidence["native_initializer_completed_calls"] += 1
    evidence["native_initializer_calls"] += 1
    baseline_u, baseline_v = initialize_factors(buffers["hard"], cells, 8, c)
    evidence["native_initializer_completed_calls"] += 1
    probe._require(torch.equal(u, baseline_u) and torch.equal(v, baseline_v), "Current native initializers differ")
    logits = LowRankLogits.apply(u, v, buffers["hard"], .05, 4096)
    probability = logits.double().softmax(1).detach()
    moments = LowRankMoments.apply(u, v, buffers["hard"], material, .05, 4096).detach()
    evidence["native_P0_probability_material_evaluations"] += 1
    probe._require(float((probability.sum(1)-1).abs().max()) <= 1e-12
                   and torch.allclose(moments, probability.T @ material / len(u), atol=1e-12, rtol=1e-12),
                   "Original native P0 row/material conservation differs")
    certificates, saved_moments = {}, {}
    for step in (0, 25):
        evidence["stage"] = f"cached_NODE_{step}_material_and_linear_certificate"
        snapshot = torch.load(root / f"condensation_{c}/checkpoints" / f"step_{step:06d}.pt", map_location="cpu", weights_only=False)
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
        saved_moments[step] = saved
    probe._require(torch.equal(saved_moments[0], moments), "Original P0 moments differ")
    actual_x, actual_q, mass = representative(moments, buffers["transform"], 3703, "cuda")
    old_x, old_q, old_mass = representative(saved_moments[0], buffers["transform"], 3703, "cuda")
    weights, old_weights = torch.full_like(mass, 1/cells), torch.full_like(old_mass, 1/cells)
    probe._require(all(torch.equal(a, b) for a, b in zip((actual_x, actual_q, weights), (old_x, old_q, old_weights), strict=True)),
                   "Actual FP32 P0 X/Q/uniformweights differ from cached NODE")
    return dict(condensation_seed=c, source_reference_certificate_passed=True, current_native_M0_exactly_equal_own_cached_NODE0=True,
                manual_current_native_baseline_U0_V0_exact=True, historical_checkpoint_UV_available=False,
                P0_probability=probe._digest(probability), P0_parameters=probe._digest([u, v]), P0_moments=probe._digest(moments),
                actual_FP32_P0_X_Q_uniform_equal_reference=True, student_inputs=probe._digest([actual_x, actual_q, weights]),
                input_digest=_input_digest(actual_x, actual_q, weights, buffers["graph"], buffers["q"],
                             dict(train=buffers["train"], val=buffers["val"]), None, lambda: False),
                original_linear_certificates=certificates, artifact_retention_flag=retained,
                original_head_penalty_from_validated_resume=penalty,
                original_head_tolerance_from_validated_resume=tolerance)


def _load_source(condensation_seed, spec, science, evidence, stop):
    c = _cond(condensation_seed)
    row = _roots(Path(__file__).resolve().parents[1])[0]
    root = Path(row["source_root"])
    _stop(stop)
    evidence["stage"] = "original_cond_indexed_source"
    _count(evidence, "source_load")
    repo = Path(__file__).resolve().parents[1]
    graph, train, val, test, fresh_h = citation_search._prepare_dataset("citeseer", str(repo/"data"), "cuda", "row")
    config = citation_search._legacy_teacher_config("citeseer", row["ratio"], graph, train, val, test, "row")
    _require(config == original._config(row) and _fingerprint(config) == row["root"], "Original graph/splits/preprocessing differ")
    teacher = torch.load(root/"teacher.pt", map_location="cuda", weights_only=False)
    _require(isinstance(teacher, dict), "Malformed original teacher")
    logits = _tensor(teacher.get("logits"), (3327,6), torch.float64, "Malformed original teacher logits")
    q = training_refined_targets(logits, 1., graph["y"], train, 0)
    ghost = source_helper.candidate_controls(dict(original.CANDIDATE, method="source_linear",
                  assignment_coordinates="raw_rms", source_linear_schema=1))
    h,z,transform,hard,_,source = source_helper.cached_source(root,ghost,c,fresh_h,q,config)
    _require(z.shape == (3327,3703) and q.shape == (3327,6) and int(hard.max())+1 == 120
             and len(torch.unique(hard)) == 120, "Fixed native source dimensions/hard partition differ")
    _require(torch.is_tensor(val[1]) and val[1].dtype == torch.bool and val[1].shape == (3327,)
             and int(val[1].sum()) == 500, "Require original500-node validation mask")
    dense = probe._frozen_original_dense_S(graph["adj"])
    buffers = dict(root=root,cells=120,condensation_seed=c,graph=graph,train=train,val=val[1],x=graph["x"].detach(),
         S=dense,original_S=graph["adj"].detach(),h=h.detach().float(),z=z.detach(),q=q.detach(),hard=hard.detach(),
         transform=probe._frozen_transform(transform),ghost=ghost,config=config,source=source,pins=spec["files_sha256"],counts=evidence["counts"])
    certified = json.loads(Path(science["parents"][0]["path"]).read_text())["roots"][row["root"]]
    common = _common_certificate(buffers, certified["native_buffers"])
    _count(evidence,"model_GEOM_factory")
    buffers["model_initial"] = probe.geom_uniform_initial(3703,6,hidden=256,dtype=torch.float32,device="cuda")
    _count(evidence,"model_GEOM_factory",True)
    _count(evidence,"native_factor_factory")
    u,v = initialize_factors(hard,120,8,c)
    _count(evidence,"native_factor_factory",True)
    buffers["initial"] = [u.detach(),v.detach()]
    counters = dict(native_initializer_calls=0,native_initializer_completed_calls=0,
       native_P0_probability_material_evaluations=0,cached_head_gradient_evaluations=0,cached_outer_CE_evaluations=0)
    try:
        reference = _reference(dict(buffers,transform=transform),ghost,c,counters)
    finally:
        evidence["counts"].update({k:v for k,v in counters.items() if k != "stage"})
    _require(reference["P0_parameters"] == probe._digest(buffers["initial"]), "Current own native factors differ from reference")
    recipe_row = science["roots"][0]["student_recipe"]
    _require(json.loads(Path(recipe_row["path"]).read_text()) == dict(inherited._settings(120),input_scale=1.),
             "Original120 student recipe/identity scaling changed")
    buffers["recipe_origin"] = recipe_row
    buffers["file_stats"] = {p:(Path(p).stat().st_size,Path(p).stat().st_mtime_ns,Path(p).stat().st_ino) for p in buffers["pins"]}
    evidence.update(common_AT_native_buffers_exact_excluding_hard=True,source_reference_certificate_passed=True,
                    source_reference=reference,common_source_buffers=common)
    _count(evidence,"source_load",True)
    return buffers,reference


def _context(spec, condensation_seed, buffers, reference, environment):
    c = _cond(condensation_seed)
    _require(reference.get("source_reference_certificate_passed") is True and reference.get("condensation_seed") == c,
             "Own condensation source/reference certificate required")
    return dict(schema=1,candidate=spec["candidates"][str(c)],cells=120,condensation_seed=c,implementation=spec["source"],
        numerical_source=spec["numerical_source"],spec=spec,scientific_preregistration=spec["scientific_preregistration"],
        native_environment=environment,ghost_source=buffers["source"],reference=reference,student_recipe_origin=buffers["recipe_origin"],
        policy=json.loads(json.dumps(POLICY)),assignment_steps=25,
        source_buffers=ay._source_buffer_digest(buffers),no_AZ_AY_state_reuse=True)



def _origin_check(moments, buffers, reference, evidence):
    _require(reference.get("source_reference_certificate_passed") is True and
             reference.get("condensation_seed") == buffers["condensation_seed"], "Own source/reference certificate required")
    origin=torch.load(buffers["root"]/original.REFERENCE/f"condensation_{buffers['condensation_seed']}/checkpoints/step_000000.pt",
                      map_location="cpu",weights_only=False)
    _require(isinstance(origin,dict),"Malformed original NODE0")
    cached=_tensor(origin.get("moments"),moments.shape,torch.float64,"Malformed original moments")
    _require(torch.equal(moments.detach(),cached.to(moments)),"Current native P0 differs from exact cached NODE0")
    _count(evidence,"readonly_P0_representative")
    x,q,mass=representative(moments.detach(),probe._transform(buffers),buffers["z"].shape[1],buffers["z"].device)
    _count(evidence,"readonly_P0_representative",True)
    _require(reference["actual_FP32_P0_X_Q_uniform_equal_reference"] is True and
        probe._digest([x,q,torch.full_like(mass,1/buffers["cells"])])==reference["student_inputs"],
        "Actual native own P0 inputs differ from original own NODE0")


def _load_progress(folder, buffers, context, evidence, stop):
    _require(folder.is_dir(),"Require completed BA candidate cache")
    pins=_cache_files(folder,complete=True)
    _require(json.loads((folder/"candidate.json").read_text())==context["candidate"],"Changed BA candidate")
    targets,target_sha=_verify_targets(folder,buffers,context,evidence,stop)
    bundle=torch.load(folder/"progress.pt",map_location="cpu",weights_only=False)
    _check_progress(bundle,buffers,context,targets,target_sha,evidence,stop)
    _require(bundle["frontier"]==HORIZON,"Partial trajectory preserved; no resume/fallback")
    _origin_check(bundle["states"][0]["moments"].to(buffers["z"]),buffers,context["reference"],evidence)
    _require(json.loads((folder/"history.json").read_text())==bundle["history"],"Changed history mirror")
    for step in (0,HORIZON):
        mirror=torch.load(folder/f"step_{step:06d}.pt",map_location="cpu",weights_only=False)
        _require(_seal(mirror)==_seal(bundle["states"][step]),"Endpoint mirror differs from authority")
    _require(_cache_files(folder,complete=True)==pins,"Readonly cache replay changed files")
    return bundle,pins


def _prepare(spec, buffers, context, evidence, stop):
    folder=_folder(spec,buffers["condensation_seed"])
    if folder.exists():
        bundle,pins=_load_progress(folder,buffers,context,evidence,stop)
        return dict(cached=True,frontier=HORIZON,prefix_sha256=_seal(bundle),cache_files_sha256=pins)
    _require(context["reference"]["source_reference_certificate_passed"] is True
             and context["condensation_seed"] == buffers["condensation_seed"], "Own source certificate required before any P update")
    parameters=[p.detach().clone().requires_grad_() for p in buffers["initial"]]
    _count(evidence,"origin_connected_moment")
    current=probe._moments(buffers,parameters)
    _count(evidence,"origin_connected_moment",True)
    _origin_check(current,buffers,context["reference"],evidence)
    targets=ay._targets(buffers,evidence,stop)
    target_digest=probe._digest(targets)
    _count(evidence,"P_optimizer_constructor")
    optimizer=torch.optim.Adam(parameters,lr=.01,betas=(.9,.999),eps=1e-12,weight_decay=0,foreach=False,fused=False)
    _count(evidence,"P_optimizer_constructor",True)
    first,second=[torch.zeros_like(p) for p in parameters],[torch.zeros_like(p) for p in parameters]
    folder.mkdir(parents=True,exist_ok=False)
    probe._atomic(folder/"candidate.json",json.dumps(context["candidate"],indent=2),False)
    probe._atomic(folder/"source_gradient_targets.pt",probe._attach(dict(schema=1,context=context,
                  targets=probe.cpu_state(targets))),True)
    target_sha=_sha(folder/"source_gradient_targets.pt")
    states,scale={},None
    for step in range(HORIZON+1):
        _stop(stop);inherited._stable_files(buffers)
        state,scale=_state(buffers,parameters,optimizer.state_dict(),step,scale,targets,evidence,stop)
        states[step]=state
        _require(probe._digest(targets)==target_digest,"Immutable source targets/models changed")
        bundle=_store(folder,context,states,scale,target_sha,targets)
        if step==HORIZON:break
        expected,first,second=inherited._adam_step(parameters,first,second,
            [g.to(p) for g,p in zip(state["scaled_factor_gradients"],parameters,strict=True)],step+1)
        _stop(stop);probe._runtime_precision_guard()
        _count(evidence,"P_update");optimizer.step();_count(evidence,"P_update",True)
        _require(all(torch.equal(p.detach(),t) for p,t in zip(parameters,expected,strict=True)),"Native Adam recurrence differs")
        inherited._optimizer(optimizer.state_dict(),parameters,first,second,step+1)
    _require(_sha(folder/"source_gradient_targets.pt")==target_sha,"Frozen targetcache file changed")
    return dict(cached=False,frontier=HORIZON,own_J0=scale,prefix_sha256=_seal(bundle),
        cache_files_sha256=_cache_files(folder,complete=True),target_model_digest_before=target_digest,
        target_model_digest_after=probe._digest(targets))


def _gate(spec, condensation_seed, checksum):
    c = _cond(condensation_seed)
    path = Path(spec["certificate_outputs"][str(c)])
    _require(type(checksum) is str and len(checksum) == 64 and path.is_file() and _sha(path) == checksum, "Require exact accepted native25 gate SHA")
    gate = json.loads(path.read_text())
    expected = dict(passed=True, schema=1, assignment_steps=HORIZON, source=spec["source"], numerical_source=spec["numerical_source"],
        candidate=spec["candidates"][str(c)], candidate_id=spec["candidate_ids"][str(c)], cells=120, condensation_seed=c, full25_cache_replay_passed=True,
        actual_FP32_P0_X_Q_uniform_equal_reference=True, source_reference_certificate_passed=True, common_AT_native_buffers_exact_excluding_hard=True, source_target_cache_replay_passed=True,
        gradient_policy=spec["gradient_policy"], scientific_preregistration=spec["scientific_preregistration"])
    _require(isinstance(gate, dict) and all(_exact(gate.get(k), v) for k, v in expected.items()), "Changed/partial BA native25 gate")
    _require(all(type(gate.get(k)) is str and len(gate[k]) == 64 for k in ("prefix_sha256", "spec_sha256")),
             "Gate lacks exact source-prefix/spec hashes")
    _require(isinstance(gate.get("files_sha256"), dict) and gate["files_sha256"], "Gate lacks exact input/cache pins")
    probe._checked_files(gate["files_sha256"])
    return gate


def _validate(spec, buffers, context, arm, gate_sha, evidence, stop):
    gate = _gate(spec, buffers["condensation_seed"], gate_sha)
    bundle, _ = _load_progress(_folder(spec, buffers["condensation_seed"]), buffers, context, evidence, stop)
    _require(_seal(bundle) == gate["prefix_sha256"], "Students require accepted actual BA25 prefix")
    reference = {s: torch.load(buffers["root"] / original.REFERENCE / f"condensation_{buffers['condensation_seed']}/checkpoints/step_{s:06d}.pt",
                              map_location="cpu", weights_only=False) for s in (0, HORIZON)}
    snapshots = reference if arm == "node_reference" else bundle["states"]
    rows = []
    for step in (0, HORIZON):
        cx, cq, mass = representative(snapshots[step]["moments"], probe._transform(buffers), 3703, "cuda")
        supplied = torch.full_like(mass, 1/buffers["cells"])
        if step == 0:
            rx, rq, rm = representative(reference[0]["moments"], probe._transform(buffers), 3703, "cuda")
            _require(all(torch.equal(a, b) for a, b in zip((cx, cq, supplied), (rx, rq, torch.full_like(rm, 1/buffers["cells"])), strict=True)),
                     "Actual FP32 shared P0 X/Q/uniform differs")
            directory = _folder(spec, buffers["condensation_seed"])/"shared_P0_validation"
        else:
            directory = _folder(spec, buffers["condensation_seed"])/f"{arm}25_validation"
        for seed in _SEEDS:
            row = inherited._evaluate_student(directory, (cx, cq, supplied), buffers, context, gate_sha, seed, stop)
            rows.append(dict(arm=arm, step=step, condensation_seed=buffers["condensation_seed"], candidate_id=original.REFERENCE if arm == "node_reference" else spec["candidate_ids"][str(buffers["condensation_seed"])],
                             shared_P0=step == 0, physical_condition=str(directory.resolve())+f":{seed}", **row))
    evidence["counts"].update(logical_student_conditions=6, logical_serving_conditions=12,
        physical_final_GCN_fits=sum(r["actual_student_fits"] for r in rows),
        physical_sameweights_serving_outputs=sum(r["physical_route_outputs"] for r in rows),
        selected_state_diagnostic_route_forwards=sum(r["diagnostic_route_forwards"] for r in rows),
        final_optimizer_epochs=inherited._settings(buffers["cells"])["epochs"]*sum(r["actual_student_fits"] for r in rows))
    return dict(rows=rows, arm=arm, condensation_seed=buffers["condensation_seed"], checkpoints=[0, HORIZON], student_seeds=list(_SEEDS), test_enabled=False,
                shared_P0_physical_reuse=True, secondary_same_selected_GCN_weights_and_epoch=True)


def _run(operation, condensation_seed, spec_path, spec_sha256, output_path, arm, gate_sha256, stop):
    c = _cond(condensation_seed)
    _require(operation in ("prepare", "certify", "validate") and callable(stop), "Unsupported BA operation")
    _require((operation != "validate" and arm is None and gate_sha256 is None) or
             (operation == "validate" and type(arm) is str and arm in ("node_reference", "gradient") and type(gate_sha256) is str), "Invalid fixed BA phase controls")
    spec_path = Path(spec_path).resolve()
    spec, science = _load_spec(spec_path, spec_sha256)
    output = Path(output_path).resolve()
    expected = Path(spec["certificate_outputs"][str(c)]) if operation == "certify" else Path(spec["output_root"])/f"condensation_{c}"/(
        "native_prepare25_v1.json" if operation == "prepare" else f"validation_{arm}_v1.json")
    _require(output == expected and not output.exists(), "Require new declared BA output")
    if operation == "validate":
        gate = _gate(spec, c, gate_sha256)
        _require(gate["spec_sha256"] == spec_sha256, "Gate refers to another spec")
    started = time.monotonic()
    bounded = lambda: stop() or time.monotonic()-started >= 300
    evidence = dict(passed=False, operation=operation, arm=arm, cells=120, condensation_seed=c, schema=1, assignment_steps=HORIZON,
        source=spec["source"], numerical_source=spec["numerical_source"], candidate=spec["candidates"][str(c)], candidate_id=spec["candidate_ids"][str(c)],
        scientific_preregistration=spec["scientific_preregistration"], spec_path=str(spec_path), spec_sha256=spec_sha256,
        counts=dict(P_update_attempts=0, P_update_completed=0, final_student_fit_attempts=0, final_student_fit_completed=0,
                    teacher_map_Phi_hardinit_fits=0, functional_inner_SGD_steps=0, extra_MLP_fits=0,
                    fresh_head_fits=0, adjoint_or_Hessian_solves=0), test_enabled=False,
        gradient_policy=json.loads(json.dumps(POLICY)),
        stage="fresh_native_policy", gate_sha256=gate_sha256)
    native, primary, buffers = False, None, None
    try:
        _stop(bounded)
        _require(torch.get_num_threads() == 4 and not torch.cuda.is_initialized(), "Require threads4/fresh CUDA process")
        environment = dict(**probe._native("cuda"), python_version=platform.python_version(), threads=4)
        native = True
        torch.cuda.reset_peak_memory_stats()
        _require(torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory == original.CUDA_CAPACITY, "GPU capacity changed")
        evidence["native_environment"] = environment
        buffers, reference = _load_source(c, spec, science, evidence, bounded)
        evidence["source_buffer_digest_before"] = ay._source_buffer_digest(buffers)
        evidence["seed0_model_digest_before"] = probe._digest(buffers["model_initial"])
        context = _context(spec, c, buffers, reference, environment)
        evidence["stage"] = operation
        if operation == "prepare":
            result = _prepare(spec, buffers, context, evidence, bounded)
        elif operation == "certify":
            bundle, pins = _load_progress(_folder(spec, c), buffers, context, evidence, bounded)
            repo = Path(__file__).resolve().parents[1]
            inputs = {str(repo/p): h for p, h in spec["source"]["files"].items()}
            inputs.update(spec["files_sha256"])
            inputs.update(spec["artifacts_sha256"])
            inputs.update({str(spec_path): spec_sha256, spec["scientific_preregistration"]["path"]: SCIENCE_SHA})
            inputs.update({row["path"]: row["sha256"] for row in science["parents"]})
            inputs.update(pins)
            result = dict(full25_cache_replay_passed=True, actual_FP32_P0_X_Q_uniform_equal_reference=True,
                prefix_sha256=_seal(bundle), own_J0=bundle["J0"], files_sha256=inputs, frontier=HORIZON,
                source_target_cache_replay_passed=True, target_model_digest=bundle["target_digest"],
                target_cache_sha256=bundle["target_sha256"])
        else:
            result = _validate(spec, buffers, context, arm, gate_sha256, evidence, bounded)
        evidence.update(result)
        _stop(bounded)
        evidence["passed"] = True
    except BaseException as error:
        primary = error
        evidence.update(error_type=type(error).__name__, error=str(error), failed_stage=evidence["stage"])
    finally:
        if buffers is not None:
            try:
                evidence["source_buffer_digest_after"] = ay._source_buffer_digest(buffers)
                evidence["seed0_model_digest_after"] = probe._digest(buffers["model_initial"])
                _require(evidence["source_buffer_digest_after"] == evidence["source_buffer_digest_before"]
                         and evidence["seed0_model_digest_after"] == evidence["seed0_model_digest_before"],
                         "Immutable native source/model buffers changed")
            except BaseException as error:
                evidence.update(passed=False,native_source_preservation_error=repr(error))
                if primary is None: primary = error
        try:
            _preserve(spec, spec_path, spec_sha256, science)
            evidence["source_unchanged"] = True
        except BaseException as error:
            evidence.update(passed=False, source_unchanged=False, preservation_error=repr(error))
            if primary is None: primary = error
        if native:
            try:
                torch.cuda.synchronize()
                capacity = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
                allocated, reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
                _require(capacity == original.CUDA_CAPACITY and allocated <= capacity and reserved <= capacity, "Invalid native memory/capacity")
                evidence.update(CUDA_peak_allocated_bytes=allocated, CUDA_peak_reserved_bytes=reserved, CUDA_total_bytes=capacity)
                probe._runtime_precision_guard()
            except BaseException as error:
                evidence.update(passed=False, native_finalization_error=repr(error))
                if primary is None: primary = error
        evidence["seconds"] = time.monotonic()-started
        try:
            _write_new(output, evidence)
        except BaseException:
            if primary is not None: raise primary
            raise
    if primary is not None: raise primary
    return dict(evidence, evidence_path=str(output), evidence_sha256=_sha(output))


def prepare(condensation_seed, spec_path, spec_sha256, output_path, stop=lambda: False):
    return _run("prepare", condensation_seed, spec_path, spec_sha256, output_path, None, None, stop)


def certify(condensation_seed, spec_path, spec_sha256, output_path, stop=lambda: False):
    return _run("certify", condensation_seed, spec_path, spec_sha256, output_path, None, None, stop)


def validate(condensation_seed, arm, spec_path, spec_sha256, output_path, gate_sha256, stop=lambda: False):
    return _run("validate", condensation_seed, spec_path, spec_sha256, output_path, arm, gate_sha256, stop)
