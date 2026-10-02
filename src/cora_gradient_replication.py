"""Own Cora70 cond1/2 gradient replication; original baselines stay readonly.

AZ math/cache and BG endpoint checks are direct private aliases. The new source,
origin/context/namespace/student contracts own only this fixed replication.
"""
import json
import platform
import time
from pathlib import Path

import torch

from src import citation_gradient_fixed25 as az
from src import citation_gradient_probe as ay
from src import citeseer_finite_student_v2 as inherited
from src import cora_node_reference as baseline
from src import cora_node_reference_certificate_v2 as bg
from src import cora_source_certificate as original
from src import finite_student_probe as probe
from src.ce_gradient_alignment import POLICY
from src.citation_source_preflight import _exact, _write_new
from src.io import _fingerprint
from src.research_loop import implementation_provenance
from src.sweep_utils import representative

SCIENCE = "Cora70_CE_gradient_alignment_fixed25_replication_scientific_stageBH_v1.json"
SCIENCE_SHA = "6d84f4f498866aa40f244b3fdc7277e72d75d843e1b83a18774d6c36e85bb5ec"
HORIZON = 25
_SEEDS = (4200, 4201, 4202)
_POLICY_SPEC = json.loads(json.dumps(POLICY))
_require, _stop, _tensor, _sha, _seal = probe._require, probe._stop, probe._tensor, probe._sha, probe._seal
_count = inherited._count
_own_source, _native_buffers, _baseline_endpoints = baseline._load_source, baseline._native_buffers, bg._endpoints
_state, _check_progress = az._state, az._check_progress
_verify_targets, _store, _cache_files = az._verify_targets, az._store, az._cache_files


def _seed(value):
    _require(type(value) is int and value in (1, 2), "Only own condensation1/2")
    return value


def numerical_source():
    result = ay.numerical_source()
    repo = Path(__file__).resolve().parents[1]
    for name in ("cora_gradient_replication.py", "cora_gradient_replication_students.py"):
        _require(result["files"].get(name) == _sha(repo / "src" / name), "BH source missing or changed")
    return result


def _science(repo):
    path = repo / "results/proposals" / SCIENCE
    probe._checked_files({str(path): SCIENCE_SHA})
    return json.loads(path.read_text())


def canonical_candidate(condensation_seed):
    seed = _seed(condensation_seed)
    science = _science(Path(__file__).resolve().parents[1])
    return dict(science["candidates"][str(seed)], gradient_source_digest=_seal(numerical_source()))


def _pins(spec, path, checksum, science):
    repo = Path(__file__).resolve().parents[1]
    pins = {str(repo / p): h for p, h in spec["source"]["files"].items()}
    pins.update(spec["files_sha256"])
    pins.update(spec["artifacts_sha256"])
    pins.update({str(path): checksum, spec["scientific_preregistration"]["path"]: SCIENCE_SHA})
    for row in [*science["parents"], *science["BG_readonly_certificate_preservation_parents"],
                science["original_BB_certificate"], science["original_BB_closure"], science["source_before_artifact"]]:
        pins[row["path"]] = row["sha256"]
    return pins


def _preserve(spec, path, checksum, science):
    probe._checked_files(_pins(spec, path, checksum, science))
    _require(implementation_provenance() == spec["source"] and numerical_source() == spec["numerical_source"]
             and platform.python_version() == spec["python_version"] and torch.get_num_threads() == 4,
             "Source/Git/Python/versions/threads changed")


def _load_spec(path, checksum):
    repo = Path(__file__).resolve().parents[1]
    path = Path(path).resolve()
    _require(type(checksum) is str and len(checksum) == 64 and path.is_file()
             and path.is_relative_to(repo / "results/proposals") and _sha(path) == checksum, "Require frozen BH spec path/SHA")
    science, spec = _science(repo), json.loads(path.read_text())
    fields = set(science["implementation_contract"]["spec_fields"])
    _require(isinstance(spec, dict) and set(spec) == fields and type(spec["schema"]) is int and spec["schema"] == 1
             and _exact(spec["fixed"], science["fixed"]) and _exact(spec["case"], science["case"])
             and isinstance(spec["candidates"], dict) and set(spec["candidates"]) == {"1", "2"}
             and isinstance(spec["candidate_ids"], dict) and set(spec["candidate_ids"]) == {"1", "2"}
             and all(_exact(spec["candidates"][str(c)], canonical_candidate(c)) and
                     spec["candidate_ids"][str(c)] == _fingerprint(spec["candidates"][str(c)]) for c in (1, 2))
             and _exact(spec["gradient_policy"], _POLICY_SPEC) and _exact(spec["student_seeds"], list(_SEEDS)),
             "Unknown/changed BH owncond candidate/source/seeds/policy")
    _require(spec["files_sha256"] == science["files_sha256"] and
             spec["scientific_preregistration"] == dict(path=str(repo / "results/proposals" / SCIENCE), sha256=SCIENCE_SHA),
             "Original69bytes/science changed")
    _require(isinstance(spec["artifacts_sha256"], dict) and spec["artifacts_sha256"] and
             all(Path(p).is_absolute() and Path(p).resolve().is_relative_to(repo / "results") for p in spec["artifacts_sha256"]),
             "Require reviewed immutable artifacts")
    contract = science["implementation_contract"]
    _require(spec["output_root"] == contract["output_root"] and _exact(spec["certificate_outputs"], contract["certificate_outputs"]),
             "Wrong owncond output namespace")
    _preserve(spec, path, checksum, science)
    original._config(spec["case"])
    original._recipe(spec["case"])
    return spec, science


def _case(science, seed):
    seed = _seed(seed)
    return dict(science["case"], condensation_seed=seed,
                hard_path=Path(science["hard_assignments"][str(seed)]["path"]).name)


def _folder(spec, condensation_seed):
    seed = _seed(condensation_seed)
    return Path(spec["output_root"]) / f"condensation_{seed}" / spec["candidate_ids"][str(seed)]


def _load_source(spec, science, seed, evidence, stop):
    seed, case = _seed(seed), _case(science, seed)
    source_spec = dict(spec, case=case, baseline_candidate=science["baseline_candidate"])
    buffers, origin, expected = _own_source(source_spec, science, seed, evidence, stop)
    _stop(stop)
    folder = Path(science["existing_only_reference_dirs"][str(seed)])
    _baseline_endpoints(buffers, origin, expected, folder, evidence)
    certified = science["BG_certified_own_origins"][str(seed)]
    fields = ("source_context", "native_buffers_before", "pre_optimizer_P0", "student_recipe_origin",
              "endpoint_certificates", "certified_files_sha256", "history_metadata")
    _require(all(_exact(evidence.get(k), certified[k]) for k in fields), "Own source/reference differs from accepted BG certificate")
    _require(evidence.get("complete25_native_certificate_passed") is True and
             evidence.get("actual_P0_moments_bitwise_equal_core_step0") is True and
             evidence.get("actual_FP32_P0_X_Q_F64_uniform_equal_core_step0") is True, "Own baseline/P0 unqualified")
    saved = torch.load(folder / "checkpoints/step_000000.pt", map_location="cpu", weights_only=False)
    buffers.update(cells=70, condensation_seed=seed, case=case, initial=origin["parameters"],
                   transform=probe._frozen_transform(buffers["transform"]), x=buffers["graph"]["x"],
                   original_S=buffers["graph"]["adj"], S=buffers["dense"])
    _count(evidence, "source_anchor0_factory")
    buffers["model_initial"] = ay.anchor_initial(1433, 7, 256, 0, dtype=torch.float32, device=buffers["z"].device)
    _count(evidence, "source_anchor0_factory", True)
    reference = dict(actual_current_M0_bitwise_equal_own_NODE0=True,
                     actual_FP32_P0_X_Q_uniform_equal_reference=True,
                     student_inputs=probe._digest(list(origin["inputs"])), input_digest=origin["input_digest"],
                     own_BG_certificate=science["BG_certificates"][str(seed)], own_condensation_seed=seed,
                     reference_id=case["reference_id"], source_context=evidence["source_context"],
                     baseline_files_sha256=evidence["certified_files_sha256"],
                     endpoint_certificates=evidence["endpoint_certificates"], historical_UV_available=False)
    evidence.update(source_reference_certificate_passed=True, BG_source_reference_certificate=science["BG_certificates"][str(seed)],
                    original_recipe_origin=evidence["student_recipe_origin"])
    return buffers, saved, reference


def _attach_source(buffers, spec, path, checksum, science, case, evidence):
    pins = _pins(spec, path, checksum, science)
    probe._checked_files(pins)
    buffers.update(case=case, recipe_origin=evidence["original_recipe_origin"], counts=evidence["counts"], files_sha256=pins,
                   file_stats={p: (Path(p).stat().st_size, Path(p).stat().st_mtime_ns, Path(p).stat().st_ino) for p in pins})


def _context(spec, seed, buffers, reference, environment):
    return dict(schema=1, candidate=spec["candidates"][str(seed)], cells=70, condensation_seed=seed, case=buffers["case"],
        implementation=spec["source"], numerical_source=spec["numerical_source"], spec=spec,
        scientific_preregistration=spec["scientific_preregistration"], native_environment=environment,
        ghost_source=buffers["source"], reference=reference, student_recipe_origin=buffers["recipe_origin"],
        policy=_POLICY_SPEC, assignment_steps=HORIZON, source_buffers=ay._source_buffer_digest(buffers),
        no_cond0_or_prior_failed_state_reuse=True)


def _origin_check(moments, buffers, reference, evidence, saved=None):
    case = buffers["case"]
    if saved is None:
        saved = torch.load(buffers["root"] / case["reference_id"] / f"condensation_{buffers['condensation_seed']}/checkpoints/step_000000.pt",
                           map_location="cpu", weights_only=False)
    _require(isinstance(saved, dict), "Malformed own original NODE0")
    cached = _tensor(saved.get("moments"), (buffers["cells"], 1441), torch.float64, "Malformed own original moments")
    _require(torch.equal(moments.detach(), cached.to(moments)), "Current native P0 differs from exact own cached NODE0")
    _count(evidence, "readonly_P0_representative")
    x, q, mass = representative(moments.detach(), probe._transform(buffers), 1433, buffers["z"].device)
    _count(evidence, "readonly_P0_representative", True)
    _require(reference["actual_FP32_P0_X_Q_uniform_equal_reference"] is True and
        probe._digest([x, q, torch.full_like(mass, 1/buffers["cells"])]) == reference["student_inputs"],
        "Actual native shared P0 inputs differ from own BB reference")


def _prepare(spec, buffers, context, saved, evidence, stop):
    folder = _folder(spec, buffers["condensation_seed"])
    _require(not folder.exists(), "Require fresh absent BH namespace; incomplete and old caches preserved")
    parameters = [p.detach().clone().requires_grad_() for p in buffers["initial"]]
    _count(evidence, "origin_connected_moment")
    current = probe._moments(buffers, parameters)
    _count(evidence, "origin_connected_moment", True)
    _origin_check(current, buffers, context["reference"], evidence, saved)
    targets = ay._targets(buffers, evidence, stop)
    target_digest = probe._digest(targets)
    _count(evidence, "P_optimizer_constructor")
    optimizer = torch.optim.Adam(parameters, lr=.01, betas=(.9, .999), eps=1e-12, weight_decay=0, foreach=False, fused=False)
    _count(evidence, "P_optimizer_constructor", True)
    first, second = [torch.zeros_like(p) for p in parameters], [torch.zeros_like(p) for p in parameters]
    _stop(stop)
    inherited._stable_files(buffers)
    folder.mkdir(parents=True, exist_ok=False)
    probe._atomic(folder / "candidate.json", json.dumps(context["candidate"], indent=2), False)
    probe._atomic(folder / "source_gradient_targets.pt", probe._attach(dict(schema=1, context=context,
                  targets=probe.cpu_state(targets))), True)
    target_sha = _sha(folder / "source_gradient_targets.pt")
    states, scale = {}, None
    for step in range(HORIZON + 1):
        _stop(stop)
        inherited._stable_files(buffers)
        state, scale = _state(buffers, parameters, optimizer.state_dict(), step, scale, targets, evidence, stop)
        states[step] = state
        _require(probe._digest(targets) == target_digest, "Immutable source targets/models changed")
        bundle = _store(folder, context, states, scale, target_sha, targets)
        if step == HORIZON:
            break
        expected, first, second = inherited._adam_step(parameters, first, second,
            [g.to(p) for g, p in zip(state["scaled_factor_gradients"], parameters, strict=True)], step + 1)
        _stop(stop)
        probe._runtime_precision_guard()
        _count(evidence, "P_update")
        optimizer.step()
        _count(evidence, "P_update", True)
        _require(all(torch.equal(p.detach(), t) for p, t in zip(parameters, expected, strict=True)), "Native Adam recurrence differs")
        inherited._optimizer(optimizer.state_dict(), parameters, first, second, step + 1)
    _require(_sha(folder / "source_gradient_targets.pt") == target_sha, "Frozen targetcache changed")
    return dict(cached=False, frontier=HORIZON, own_J0=scale, prefix_sha256=_seal(bundle),
        cache_files_sha256=_cache_files(folder, complete=True), target_model_digest_before=target_digest,
        target_model_digest_after=probe._digest(targets))


def _load_progress(folder, buffers, context, evidence, stop):
    _require(folder.is_dir(), "Require complete BH candidate cache")
    pins = _cache_files(folder, complete=True)
    _require(json.loads((folder / "candidate.json").read_text()) == context["candidate"], "Changed BH candidate")
    targets, target_sha = _verify_targets(folder, buffers, context, evidence, stop)
    bundle = torch.load(folder / "progress.pt", map_location="cpu", weights_only=False)
    _check_progress(bundle, buffers, context, targets, target_sha, evidence, stop)
    _require(bundle["frontier"] == HORIZON, "Partial trajectory preserved; no resume/fallback")
    _origin_check(bundle["states"][0]["moments"].to(buffers["z"]), buffers, context["reference"], evidence)
    _require(json.loads((folder / "history.json").read_text()) == bundle["history"], "Changed history mirror")
    for step in (0, HORIZON):
        mirror = torch.load(folder / f"step_{step:06d}.pt", map_location="cpu", weights_only=False)
        _require(_seal(mirror) == _seal(bundle["states"][step]), "Endpoint mirror differs from authority")
    _require(_cache_files(folder, complete=True) == pins, "Readonly cache replay changed files")
    return bundle, pins


def _gate(spec, condensation_seed, checksum, spec_checksum):
    seed = _seed(condensation_seed)
    path = Path(spec["certificate_outputs"][str(seed)])
    _require(type(checksum) is str and len(checksum) == 64 and path.is_file() and _sha(path) == checksum,
             "Require accepted owncond native25 gate SHA")
    gate = json.loads(path.read_text())
    expected = dict(passed=True, schema=1, assignment_steps=HORIZON, source=spec["source"], numerical_source=spec["numerical_source"],
        candidate=spec["candidates"][str(seed)], candidate_id=spec["candidate_ids"][str(seed)], cells=70, condensation_seed=seed,
        full25_cache_replay_passed=True, actual_FP32_P0_X_Q_uniform_equal_reference=True, source_target_cache_replay_passed=True,
        source_reference_certificate_passed=True, gradient_policy=spec["gradient_policy"], scientific_preregistration=spec["scientific_preregistration"],
        spec_sha256=spec_checksum, source_assets_spec_science_unchanged=True, test_enabled=False)
    _require(isinstance(gate, dict) and all(_exact(gate.get(k), v) for k, v in expected.items()), "Changed/partial/crosscond native25 gate")
    _require(type(gate.get("prefix_sha256")) is str and len(gate["prefix_sha256"]) == 64 and
             isinstance(gate.get("files_sha256"), dict) and gate["files_sha256"], "Gate lacks exact prefix/file pins")
    probe._checked_files(gate["files_sha256"])
    return gate


def _validate(spec, buffers, context, arm, gate_sha, spec_sha, evidence, stop):
    from src.cora_gradient_replication_students import evaluate_student
    gate = _gate(spec, buffers["condensation_seed"], gate_sha, spec_sha)
    bundle, _ = _load_progress(_folder(spec, buffers["condensation_seed"]), buffers, context, evidence, stop)
    _require(_seal(bundle) == gate["prefix_sha256"], "Students require accepted actual BD25 prefix")
    reference = {s: torch.load(buffers["root"] / buffers["case"]["reference_id"] / f"condensation_{buffers['condensation_seed']}/checkpoints/step_{s:06d}.pt",
                              map_location="cpu", weights_only=False) for s in (0, HORIZON)}
    snapshots = reference if arm == "node_reference" else bundle["states"]
    rows = []
    for step in (0, HORIZON):
        cx, cq, mass = representative(snapshots[step]["moments"], probe._transform(buffers), 1433, buffers["z"].device)
        supplied = torch.full_like(mass, 1/buffers["cells"])
        if step == 0:
            rx, rq, rm = representative(reference[0]["moments"], probe._transform(buffers), 1433, buffers["z"].device)
            _require(all(torch.equal(a, b) for a, b in zip((cx, cq, supplied), (rx, rq, torch.full_like(rm, 1/buffers["cells"])), strict=True)),
                     "Actual FP32 shared P0 X/Q/uniform differs")
            directory = _folder(spec, buffers["condensation_seed"]) / "shared_P0_validation"
        else:
            directory = _folder(spec, buffers["condensation_seed"]) / f"{arm}25_validation"
        for seed in _SEEDS:
            row = evaluate_student(directory, (cx, cq, supplied), buffers, context, gate_sha, seed, stop)
            rows.append(dict(arm=arm, step=step, condensation_seed=buffers["condensation_seed"], candidate_id=buffers["case"]["reference_id"] if arm == "node_reference" else spec["candidate_ids"][str(buffers["condensation_seed"])],
                             shared_P0=step == 0, physical_condition=str(directory.resolve()) + f":{seed}", **row))
    actual = sum(r["actual_student_fits"] for r in rows)
    evidence["counts"].update(logical_student_conditions=6, logical_serving_conditions=12, physical_final_GCN_fits=actual,
        physical_sameweights_serving_outputs=sum(r["physical_route_outputs"] for r in rows),
        selected_state_diagnostic_route_forwards=sum(r["diagnostic_route_forwards"] for r in rows), final_optimizer_epochs=600 * actual)
    return dict(rows=rows, arm=arm, checkpoints=[0, HORIZON], student_seeds=list(_SEEDS), test_enabled=False,
                shared_P0_physical_reuse=True, secondary_same_selected_GCN_weights_and_epoch=True)


def _run(operation, condensation_seed, spec_path, spec_sha256, output_path, arm=None, gate_sha256=None, stop=lambda: False):
    """One prepare/certify/validation operation in one fresh bounded worker."""
    _require(type(operation) is str and operation in ("prepare", "certify", "validate") and callable(stop), "Unsupported BH operation")
    _require((operation != "validate" and arm is None and gate_sha256 is None) or
             (operation == "validate" and type(arm) is str and arm in ("node_reference", "gradient") and type(gate_sha256) is str),
             "Invalid fixed BH phase controls")
    _seed(condensation_seed)
    spec_path = Path(spec_path).resolve()
    spec, science = _load_spec(spec_path, spec_sha256)
    seed = _seed(condensation_seed)
    case = _case(science, seed)
    contract = science["implementation_contract"]
    expected = (contract["validation_outputs"][str(seed)][arm] if operation == "validate" else
                contract["certificate_outputs" if operation == "certify" else "prepare_outputs"][str(seed)])
    output = Path(output_path).resolve()
    _require(str(output) == expected and not output.exists(), "Require new declared BH output")
    if operation == "prepare":
        _require(not _folder(spec, seed).exists(), "Require fresh absent BH candidate; preserve partial cache")
    if operation == "validate":
        _gate(spec, seed, gate_sha256, spec_sha256)
    counts = {k: 0 for k in original._science(Path(__file__).resolve().parents[1])["expected_success_counts"] if k != "P_updates"}
    counts.update(P_update_attempts=0, P_update_completed=0, final_student_fit_attempted=0, final_student_fit_completed=0,
                  teacher_map_Phi_hardinit_fits=0, inner_SGD_steps=0, fresh_head_fits=0, adjoint_or_Hessian_solves=0)
    evidence = dict(passed=False, operation=operation, arm=arm, cells=70, condensation_seed=seed, schema=1, assignment_steps=HORIZON,
        source=spec["source"], numerical_source=spec["numerical_source"], candidate=spec["candidates"][str(seed)], candidate_id=spec["candidate_ids"][str(seed)],
        scientific_preregistration=spec["scientific_preregistration"], spec_path=str(spec_path), spec_sha256=spec_sha256,
        counts=counts, operation_attempts={}, test_enabled=False, validation_only=True, gradient_policy=_POLICY_SPEC,
        count_scope="BB source/reference diagnostics plus actual scientific prepare or readonly replay attempts/completions; no cond0 state reuse",
        stage="fresh_native_policy", gate_sha256=gate_sha256, source_reference_certificate_passed=False,
        heldout_labels_scope="Protected dataset/input identity hashes only; never loss/accuracy/selection")
    started = time.monotonic()
    bounded = lambda: stop() or time.monotonic() - started >= 300
    native, primary, buffers = False, None, None
    try:
        _stop(bounded)
        _require(torch.get_num_threads() == 4 and not torch.cuda.is_initialized(), "Require threads4/fresh CUDA process")
        environment = dict(**probe._native("cuda"), python_version=platform.python_version(), threads=4)
        native = True
        torch.cuda.reset_peak_memory_stats()
        _require(torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory == original.CUDA_CAPACITY, "GPU capacity changed")
        evidence["native_environment"] = environment
        buffers, saved, reference = _load_source(spec, science, seed, evidence, bounded)
        _attach_source(buffers, spec, spec_path, spec_sha256, science, case, evidence)
        evidence["source_buffer_digest_before"] = ay._source_buffer_digest(buffers)
        evidence["seed0_model_digest_before"] = probe._digest(buffers["model_initial"])
        context = _context(spec, seed, buffers, reference, environment)
        evidence["stage"] = operation
        if operation == "prepare":
            result = _prepare(spec, buffers, context, saved, evidence, bounded)
        elif operation == "certify":
            bundle, pins = _load_progress(_folder(spec, seed), buffers, context, evidence, bounded)
            result = dict(full25_cache_replay_passed=True, actual_FP32_P0_X_Q_uniform_equal_reference=True,
                prefix_sha256=_seal(bundle), own_J0=bundle["J0"], files_sha256=dict(buffers["files_sha256"], **pins), frontier=HORIZON,
                source_target_cache_replay_passed=True, target_model_digest=bundle["target_digest"], target_cache_sha256=bundle["target_sha256"])
        else:
            result = _validate(spec, buffers, context, arm, gate_sha256, spec_sha256, evidence, bounded)
        evidence.update(result, reference=reference, source_context=buffers["source"], original_recipe_origin=buffers["recipe_origin"],
                        context_digest=_seal(context))
        _stop(bounded)
        evidence["passed"] = True
    except BaseException as error:
        primary = error
        evidence.update(error_type=type(error).__name__, error=str(error), failed_stage=evidence["stage"])
    finally:
        if buffers is not None and "source_buffer_digest_before" in evidence:
            try:
                evidence["source_buffer_digest_after"] = ay._source_buffer_digest(buffers)
                evidence["seed0_model_digest_after"] = probe._digest(buffers["model_initial"])
                _require(evidence["source_buffer_digest_after"] == evidence["source_buffer_digest_before"]
                         and evidence["seed0_model_digest_after"] == evidence["seed0_model_digest_before"], "Immutable native source/model changed")
            except BaseException as error:
                evidence.update(passed=False, native_source_preservation_error=repr(error))
                if primary is None:
                    primary = error
        try:
            _preserve(spec, spec_path, spec_sha256, science)
            evidence.update(source_unchanged=True, source_assets_spec_science_unchanged=True)
        except BaseException as error:
            evidence.update(passed=False, source_unchanged=False, source_assets_spec_science_unchanged=False, preservation_error=repr(error))
            if primary is None:
                primary = error
        if native:
            try:
                torch.cuda.synchronize()
                capacity = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
                allocated, reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
                _require(capacity == original.CUDA_CAPACITY and 0 <= allocated <= capacity and 0 <= reserved <= capacity,
                         "Native memory/capacity changed")
                evidence.update(CUDA_peak_allocated_bytes=allocated, CUDA_peak_reserved_bytes=reserved, CUDA_total_bytes=capacity)
                probe._runtime_precision_guard()
            except BaseException as error:
                evidence.update(passed=False, native_finalization_error=repr(error))
                if primary is None:
                    primary = error
        evidence["seconds"] = time.monotonic() - started
        if evidence["seconds"] > 300:
            error = ValueError("Runtime exceeded300seconds")
            evidence.update(passed=False, deadline_error=str(error))
            if primary is None:
                primary = error
        try:
            _write_new(output, evidence)
        except BaseException:
            if primary is not None:
                raise primary
            raise
    if primary is not None:
        raise primary
    return dict(evidence, evidence_path=str(output), evidence_sha256=_sha(output), validation_only=True)


def prepare(condensation_seed, spec_path, spec_sha256, output_path, stop=lambda: False):
    return _run("prepare", condensation_seed, spec_path, spec_sha256, output_path, stop=stop)


def certify(condensation_seed, spec_path, spec_sha256, output_path, stop=lambda: False):
    return _run("certify", condensation_seed, spec_path, spec_sha256, output_path, stop=stop)


def validate(condensation_seed, spec_path, spec_sha256, output_path, arm, gate_sha256, stop=lambda: False):
    return _run("validate", condensation_seed, spec_path, spec_sha256, output_path, arm, gate_sha256, stop)
