"""New Cora namespace/phase/origin/receipt guards using generated CPU fixtures.

No source-cache tensors, objectives, old math suites or native calls are run.
"""
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from src import cora_gradient_fixed25 as exp


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    actual = Path(__file__).resolve().parents[1]
    repo = tmp_path / "repo"
    science = json.loads((actual / "results/proposals" / exp.SCIENCE).read_text().replace(str(actual), str(repo)))
    for mapping in (science["original_files_sha256"], science["protected_BC_checkpoints_sha256"]):
        for name in mapping:
            path = Path(name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("opaque generated fixture, never deserialized")
    for case in science["cases"]:
        root = Path(case["source_root"])
        (root / "config.json").write_text(json.dumps(case["source_config"]))
        (root / f"student_recipe_{case['recipe_id']}.json").write_text(json.dumps(case["recipe"]))
    for key in ("original_files_sha256", "protected_BC_checkpoints_sha256"):
        science[key] = {p: exp._sha(p) for p in science[key]}
    for row in [*science["parents"], science["original_source_certificate"], science["original_source_closure"], science["source_before_artifact"]]:
        path = Path(row["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("immutable generated lineage")
        row["sha256"] = exp._sha(path)
    helper = repo / "src/protected.py"
    helper.parent.mkdir()
    helper.write_text("protected metadata fixture")
    monkeypatch.setattr(exp, "__file__", str(repo / "src/cora_gradient_fixed25.py"))
    source = dict(files={"src/protected.py": exp._sha(helper)}, source_digest="fake", git_head="frozen")
    monkeypatch.setattr(exp, "implementation_provenance", lambda: copy.deepcopy(source))
    monkeypatch.setattr(exp, "numerical_source", lambda: dict(policy="frozen fake versions"))
    monkeypatch.setattr(exp.torch, "get_num_threads", lambda: 4)
    science_path = repo / "results/proposals" / exp.SCIENCE
    science_path.write_text(json.dumps(science))
    monkeypatch.setattr(exp, "SCIENCE_SHA", exp._sha(science_path))
    review = repo / "results/proposals/review.json"
    review.write_text("review fixture")
    spec = dict(schema=1, fixed=copy.deepcopy(exp.FIXED), cases=copy.deepcopy(science["cases"]), candidate=exp.canonical_candidate(),
        source=copy.deepcopy(source), numerical_source=exp.numerical_source(), python_version=exp.platform.python_version(),
        files_sha256=science["original_files_sha256"], scientific_preregistration=dict(path=str(science_path), sha256=exp.SCIENCE_SHA),
        artifacts_sha256={str(review): exp._sha(review)}, output_root=science["implementation_contract"]["output_root"],
        certificate_outputs=copy.deepcopy(science["implementation_contract"]["certificate_outputs"]), gradient_policy=copy.deepcopy(exp._POLICY_SPEC),
        student_seeds=list(exp._SEEDS))
    spec["candidate_id"] = exp._fingerprint(spec["candidate"])
    path = repo / "results/proposals/spec.json"
    def save():
        path.write_text(json.dumps(spec))
        return path, exp._sha(path)
    return SimpleNamespace(repo=repo, science=science, spec=spec, path=path, save=save)


def test_direct_math_aliases_and_own_context(frozen):
    path, checksum = frozen.save()
    assert exp._load_spec(path, checksum)[0] == frozen.spec
    assert exp._load_source is exp.bc._load_source
    for name in ("_state", "_check_progress", "_verify_targets", "_store", "_cache_files"):
        assert getattr(exp, name) is getattr(exp.az, name)
    assert exp._folder(frozen.spec, 70) == Path(frozen.spec["output_root"]) / "cora70" / frozen.spec["candidate_id"]
    assert exp._SEEDS == (4100, 4101, 4102) and exp.HORIZON == 25


@pytest.mark.parametrize("mutation", ["extra", "boolschema", "caseorder", "reference", "recipe", "candidate", "id", "seed", "policy", "science", "output", "source", "python", "asset"])
def test_resealed_spec_overrides_reject_before_source(frozen, mutation):
    spec = frozen.spec
    if mutation == "extra": spec["external_source"] = "override"
    elif mutation == "boolschema": spec["schema"] = True
    elif mutation == "caseorder": spec["cases"].reverse()
    elif mutation == "reference": spec["cases"][0]["reference_id"] = spec["cases"][1]["reference_id"]
    elif mutation == "recipe": spec["cases"][0]["recipe"]["lr"] = .01
    elif mutation == "candidate": spec["candidate"]["lr"] = .05
    elif mutation == "id": spec["candidate_id"] = "old_BC_id"
    elif mutation == "seed": spec["student_seeds"] = [0, 1, 2]
    elif mutation == "policy": spec["gradient_policy"]["rho"] = .01
    elif mutation == "science": spec["scientific_preregistration"]["sha256"] = "0" * 64
    elif mutation == "output": spec["certificate_outputs"]["35"] += ".elsewhere"
    elif mutation == "source": spec["source"]["git_head"] = "changed"
    elif mutation == "python": spec["python_version"] = "changed"
    else: Path(next(iter(spec["files_sha256"]))).write_text("changed opaque source")
    with pytest.raises(ValueError): exp._load_spec(*frozen.save())


def native_gate(tmp_path):
    asset = tmp_path / "asset"
    asset.write_text("bound")
    path = tmp_path / "gate.json"
    spec = dict(certificate_outputs={"35": str(path)}, source={}, numerical_source={}, candidate={}, candidate_id="own",
                gradient_policy={}, scientific_preregistration={})
    gate = dict(passed=True, schema=1, assignment_steps=25, source={}, numerical_source={}, candidate={}, candidate_id="own", cells=35,
        full25_cache_replay_passed=True, actual_FP32_P0_X_Q_uniform_equal_reference=True, source_target_cache_replay_passed=True,
        source_reference_certificate_passed=True, gradient_policy={}, scientific_preregistration={}, spec_sha256="s" * 64,
        source_assets_spec_science_unchanged=True, test_enabled=False, prefix_sha256="p" * 64, files_sha256={str(asset): exp._sha(asset)})
    return spec, gate, path, asset


@pytest.mark.parametrize("mutation", ["origin", "frontier", "case", "spec", "prefix", "assets", "pass"])
def test_gate_is_complete_exact_and_before_student_access(tmp_path, mutation):
    spec, gate, path, asset = native_gate(tmp_path)
    if mutation == "origin": gate["actual_FP32_P0_X_Q_uniform_equal_reference"] = False
    elif mutation == "frontier": gate["assignment_steps"] = 1
    elif mutation == "case": gate["cells"] = 70
    elif mutation == "spec": gate["spec_sha256"] = "another" * 8
    elif mutation == "prefix": gate.pop("prefix_sha256")
    elif mutation == "assets": asset.write_text("tampered after certification")
    else: gate["passed"] = False
    path.write_text(json.dumps(gate))
    with pytest.raises(ValueError): exp._gate(spec, 35, exp._sha(path), "s" * 64)


def test_valid_gate_and_same_receipt_context(tmp_path):
    spec, gate, path, _ = native_gate(tmp_path)
    path.write_text(json.dumps(gate))
    assert exp._gate(spec, 35, exp._sha(path), "s" * 64) == gate


@pytest.mark.parametrize("contents", ["empty", "old_BC", "partial"])
def test_prepare_never_resumes_any_existing_namespace(tmp_path, contents):
    spec = dict(output_root=str(tmp_path), candidate_id="new")
    folder = exp._folder(spec, 35)
    folder.mkdir(parents=True)
    if contents != "empty": (folder / contents).write_text("preserve")
    before = {str(p): p.read_bytes() for p in folder.iterdir()}
    evidence = dict(counts={})
    with pytest.raises(ValueError): exp._prepare(spec, dict(cells=35), {}, {}, evidence, lambda: False)
    assert evidence["counts"] == {} and before == {str(p): p.read_bytes() for p in folder.iterdir()}


def test_complete_cache_origin_and_mirrors_use_own_reference(tmp_path, monkeypatch):
    folder = tmp_path / "cache"
    folder.mkdir()
    state = dict(step=0, moments=SimpleNamespace(to=lambda device: "nativeM"))
    states = {0: state, 25: dict(step=25)}
    bundle = dict(frontier=25, states=states, history=[{"step": 0}, {"step": 25}])
    (folder / "candidate.json").write_text("{}")
    (folder / "history.json").write_text(json.dumps(bundle["history"]))
    seen = []
    monkeypatch.setattr(exp, "_cache_files", lambda *a, **kw: {"fixed": "pins"})
    monkeypatch.setattr(exp, "_verify_targets", lambda *a: ({}, "targetsha"))
    monkeypatch.setattr(exp, "_check_progress", lambda *a: seen.append("full26 replay"))
    monkeypatch.setattr(exp, "_origin_check", lambda m, b, r, e: seen.append((m, b["case"]["reference_id"], r)))
    monkeypatch.setattr(exp, "_seal", lambda value: repr(value))
    monkeypatch.setattr(exp.torch, "load", lambda p, **kw: bundle if Path(p).name == "progress.pt" else states[int(Path(p).stem[5:])])
    context = dict(candidate={}, reference="ownBB")
    assert exp._load_progress(folder, dict(z="native", case=dict(reference_id="ownCora")), context, {}, lambda: False)[0] is bundle
    assert seen == ["full26 replay", ("nativeM", "ownCora", "ownBB")]
    (folder / "history.json").write_text("[]")
    with pytest.raises(ValueError): exp._load_progress(folder, dict(z="native", case=dict(reference_id="ownCora")), context, {}, lambda: False)


@pytest.mark.parametrize("operation", ["prepare", "certify", "validate"])
def test_one_real_lazy_dispatch_preserves_options_and_stop(operation, monkeypatch):
    from src.research_loop import dispatch
    seen = []
    stop = lambda: False
    monkeypatch.setitem(sys.modules, "src.cora_gradient_fixed25", SimpleNamespace(run=lambda **kw: seen.append(kw) or "result"))
    options = dict(operation=operation, cells=140, spec_path="frozen", spec_sha256="s" * 64, output_path="out")
    assert dispatch(dict(kind="citation_cora_gradient_fixed25", options=options), stop) == "result"
    assert seen == [dict(options, stop=stop)]


def test_failure_receipt_keeps_primary_and_preserves_sources(frozen, monkeypatch):
    path, checksum = frozen.save()
    output = Path(frozen.science["implementation_contract"]["prepare_outputs"]["35"])
    monkeypatch.setattr(exp.original, "_science", lambda repo: dict(expected_success_counts={"students": 0}))
    monkeypatch.setattr(exp.torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(exp.probe, "_native", lambda device: {})
    monkeypatch.setattr(exp.torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(exp.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(exp.torch.cuda, "get_device_properties", lambda d: SimpleNamespace(total_memory=exp.original.CUDA_CAPACITY))
    primary = ValueError("strict original source mismatch")
    def fail(*args): raise primary
    monkeypatch.setattr(exp, "_load_source", fail)
    monkeypatch.setattr(exp.torch.cuda, "synchronize", lambda: (_ for _ in ()).throw(RuntimeError("secondary finalization")))
    with pytest.raises(ValueError, match="strict original source mismatch") as caught:
        exp.run("prepare", 35, path, checksum, output)
    assert caught.value is primary
    receipt = json.loads(output.read_text())
    assert not receipt["passed"] and receipt["source_assets_spec_science_unchanged"] is True
    assert receipt["counts"]["P_update_completed"] == 0 and "secondary finalization" in receipt["native_finalization_error"]
