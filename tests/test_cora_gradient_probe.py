"""Focused generated metadata/fake-source tests; no old math or native calls."""
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from src import citation_gradient_probe as ay
from src import cora_gradient_probe as exp


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    actual = Path(__file__).resolve().parents[1]
    repo = tmp_path / "repo"
    data = json.loads((actual / "results/proposals" / exp.SCIENCE).read_text())
    for case in data["cases"]:
        case["source_root"] = case["source_root"].replace(str(actual), str(repo))
    data["original_files_sha256"] = {p.replace(str(actual), str(repo)): h
                                     for p, h in data["original_files_sha256"].items()}
    for p in data["original_files_sha256"]:
        path = Path(p)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("opaque fixture; not a real cache tensor")
    for case in data["cases"]:
        root = Path(case["source_root"])
        (root / "config.json").write_text(json.dumps(case["source_config"]))
        (root / f"student_recipe_{case['recipe_id']}.json").write_text(json.dumps(case["recipe"]))
    data["original_files_sha256"] = {p: exp._sha(p) for p in data["original_files_sha256"]}
    for item in data["parents"]:
        item["path"] = item["path"].replace(str(actual), str(repo))
        path = Path(item["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("immutable parent")
        item["sha256"] = exp._sha(path)
    for key in ("original_source_certificate", "original_source_closure"):
        old = data[key]["path"].replace(str(actual), str(repo))
        data[key] = dict(path=old, sha256=exp._sha(old))
    previous = data["preregistration_revision"]["previous_unexecuted"]
    previous["path"] = previous["path"].replace(str(actual), str(repo))
    Path(previous["path"]).write_text("unexecuted old science")
    previous["sha256"] = exp._sha(previous["path"])
    helper = repo / "src/unchanged_helper.py"
    helper.parent.mkdir(parents=True)
    helper.write_text("unchanged protected source")
    data["readonly_helper_source_pins"] = {str(helper): exp._sha(helper)}
    contract = data["implementation_contract"]
    contract["output_root"] = contract["output_root"].replace(str(actual), str(repo))
    contract["probe_outputs"] = {k: p.replace(str(actual), str(repo))
                                 for k, p in contract["probe_outputs"].items()}
    science_path = repo / "results/proposals" / exp.SCIENCE
    science_path.write_text(json.dumps(data))
    monkeypatch.setattr(exp, "SCIENCE_SHA", exp._sha(science_path))
    monkeypatch.setattr(exp, "implementation_provenance", lambda: {"actual": "source"})
    monkeypatch.setattr(exp, "numerical_source", lambda: {"actual": "versions"})
    monkeypatch.setattr(exp.torch, "get_num_threads", lambda: 4)
    monkeypatch.setattr(exp, "__file__", str(repo / "src/cora_gradient_probe.py"))
    artifact = repo / "results/proposals/review.json"
    artifact.write_text("review")
    spec = dict(schema=1, fixed=exp.FIXED, cases=copy.deepcopy(data["cases"]),
        candidate=copy.deepcopy(data["candidate"]), source={"actual": "source"},
        numerical_source={"actual": "versions"}, python_version=exp.platform.python_version(),
        files_sha256=data["original_files_sha256"],
        scientific_preregistration=dict(path=str(science_path), sha256=exp.SCIENCE_SHA),
        artifacts_sha256={str(artifact): exp._sha(artifact)}, output_root=contract["output_root"],
        probe_outputs=copy.deepcopy(contract["probe_outputs"]), gradient_policy=exp._POLICY_SPEC)
    path = repo / "results/proposals/spec.json"

    def save(cells=35):
        path.write_text(json.dumps(spec))
        return cells, path, exp._sha(path), Path(spec["probe_outputs"][str(cells)]), repo

    return SimpleNamespace(repo=repo, science=data, spec=spec, path=path, save=save)


def test_direct_numerical_aliases_are_unchanged():
    assert exp._one_update is ay._one_update and exp._targets is ay._targets
    assert exp.anchor_initial is ay.anchor_initial and exp._source_buffer_digest is ay._source_buffer_digest


@pytest.mark.parametrize("cells", [35, 70, 140])
def test_fixed_per_case_spec_preserves_original_recipe_and_reference(frozen, cells):
    spec, _, case, output, folder = exp._load_spec(*frozen.save(cells))
    assert case["cells"] == cells and case["reference_candidate"]["rank"] == 32
    assert spec["candidate"]["lr"] == .01 and case["recipe"]["epochs"] == 600
    assert output == Path(spec["probe_outputs"][str(cells)]) and folder == output.with_suffix("")


@pytest.mark.parametrize("mutation", ["unknown", "bool_schema", "candidate", "case_order", "source",
    "python", "policy", "duplicate", "assets", "previous", "recipe", "existing", "partial"])
def test_resealed_spec_rejects_before_native_or_source(frozen, mutation):
    if mutation == "unknown":
        frozen.spec["external_targets"] = []
    elif mutation == "bool_schema":
        frozen.spec["schema"] = True
    elif mutation == "candidate":
        frozen.spec["candidate"]["lr"] = .05
    elif mutation == "case_order":
        frozen.spec["cases"].reverse()
    elif mutation == "source":
        frozen.spec["source"] = {}
    elif mutation == "python":
        frozen.spec["python_version"] = "different"
    elif mutation == "policy":
        frozen.spec["gradient_policy"] = {"rho": 1.}
    elif mutation == "duplicate":
        frozen.spec["probe_outputs"]["70"] = frozen.spec["probe_outputs"]["35"]
    elif mutation == "assets":
        Path(next(iter(frozen.spec["files_sha256"]))).write_text("changed asset")
    elif mutation == "previous":
        Path(frozen.science["preregistration_revision"]["previous_unexecuted"]["path"]).write_text("changed")
    elif mutation == "recipe":
        case = frozen.spec["cases"][0]
        Path(case["source_root"], f"student_recipe_{case['recipe_id']}.json").write_text("wrong")
    else:
        output = Path(frozen.spec["probe_outputs"]["35"])
        if mutation == "partial":
            output.with_suffix("").mkdir(parents=True)
        else:
            output.parent.mkdir(parents=True)
            output.write_text("existing evidence retained")
    with pytest.raises((ValueError, FileNotFoundError)):
        exp._load_spec(*frozen.save())


@pytest.mark.parametrize("budget", [True, 0, 60, 35.0])
def test_budget_types_and_values_are_fixed(frozen, budget):
    with pytest.raises(ValueError):
        exp._budget(budget, frozen.spec["cases"])


@pytest.mark.parametrize("mutation", ["source", "native", "reference", "recipe", "nonexact_M0", "readout"])
def test_BB_binding_and_new_exact_optimizer_origin_do_not_fall_back(mutation):
    source, native, recipe = {"root": "own"}, {"H": "own"}, {"recipe": "own"}
    reference = dict(current_native_M0_bitwise_equal_cached_NODE0=True,
                     current_native_M0_allclose_1e12_cached_NODE0=True,
                     actual_FP32_P0_X_Q_uniform_equal_reference=True)
    certified = dict(source_context=copy.deepcopy(source), native_buffers=copy.deepcopy(native),
                     reference=copy.deepcopy(reference), student_recipe_origin=copy.deepcopy(recipe))
    if mutation == "nonexact_M0":
        reference["current_native_M0_bitwise_equal_cached_NODE0"] = False
        certified["reference"] = copy.deepcopy(reference)
    elif mutation == "readout":
        reference["actual_FP32_P0_X_Q_uniform_equal_reference"] = False
        certified["reference"] = copy.deepcopy(reference)
    else:
        {"source": source, "native": native, "reference": reference, "recipe": recipe}[mutation]["changed"] = True
    with pytest.raises(ValueError):
        exp._bind_reference(source, native, reference, recipe, certified)


def test_success_counts_are_observed_not_planned(frozen):
    raw = dict(P_update_completed=1, source_CE_gradient_target_completed=3,
               synthetic_alignment_partial_completed=2, original_moment_backward_completed=1,
               anchor0_GEOM_factory_completed=1, additional_anchor_GEOM_factory_completed=2,
               cached_head_gradients=2, cached_outer_teacher_CE=2)
    assert exp._success_counts(dict(counts=raw), frozen.science) == frozen.science["counts_contract"]["per_successful_case"]
    raw["source_CE_gradient_target_completed"] = 2
    with pytest.raises(ValueError):
        exp._success_counts(dict(counts=raw), frozen.science)


@pytest.fixture
def runtime(frozen, monkeypatch):
    case = frozen.spec["cases"][0]
    bb_counts = {k: 0 for k in ("cached_head_gradients", "cached_outer_teacher_CE")}
    monkeypatch.setattr(exp.original, "_science", lambda repo: {"expected_success_counts": bb_counts})
    cuda = SimpleNamespace(is_initialized=lambda: False, reset_peak_memory_stats=lambda: None,
        get_device_properties=lambda _: SimpleNamespace(total_memory=exp.original.CUDA_CAPACITY),
        current_device=lambda: 0, synchronize=lambda: None, max_memory_allocated=lambda: 10,
        max_memory_reserved=lambda: 20)
    monkeypatch.setattr(exp.torch, "cuda", cuda)
    monkeypatch.setattr(exp.probe, "_native", lambda _: {"strict": True})
    monkeypatch.setattr(exp.probe, "_runtime_precision_guard", lambda: None)
    monkeypatch.setattr(exp, "_source_buffer_digest", lambda b: copy.deepcopy(b["source_buffers"]))
    monkeypatch.setattr(exp.probe, "_digest", lambda value: copy.deepcopy(value))
    buffers = dict(model_initial=["frozen seed0"], source={"root": case["root"]}, source_buffers={"H": "own"})
    reference = {"actual_FP32_P0_X_Q_uniform_equal_reference": True}

    def load(case, spec, science, evidence, stop):
        evidence.update(source_reference_certificate_passed=True, original_recipe_origin={"recipe_id": case["recipe_id"]})
        return buffers, {"old snapshot": True}, reference

    monkeypatch.setattr(exp, "_load_source", load)
    written = {}
    monkeypatch.setattr(exp.probe, "_atomic", lambda p, value, binary: (p.write_text("sealed fake payload"), written.update({p.name: value})))
    monkeypatch.setattr(exp.probe, "cpu_state", copy.deepcopy)
    monkeypatch.setattr(exp.probe, "_attach", lambda value: dict(value, content_sha256="sealed"))

    def one_update(b, saved, ref, e, stop):
        assert b is buffers and saved == {"old snapshot": True} and ref is reference
        e["counts"].update(P_update_attempts=1, P_update_completed=1, source_CE_gradient_target_completed=3,
            synthetic_alignment_partial_completed=2, original_moment_backward_completed=1,
            anchor0_GEOM_factory_completed=1, additional_anchor_GEOM_factory_completed=2,
            cached_head_gradients=2, cached_outer_teacher_CE=2)
        e.update(target_model_digest_before="frozen targets", target_model_digest_after="frozen targets",
            actual_FP32_P0_X_Q_uniform_equal_reference=True, native_P0_moments_exactly_equal_cached_NODE0=True)
        return {0: dict(alignment={"loss": .6}, scale=.6, conservation={"mass_sum": 0}),
                1: dict(alignment={"loss": .7}, conservation={"mass_sum": 0})}, {"immutable": "targets"}

    monkeypatch.setattr(exp, "_one_update", one_update)
    return SimpleNamespace(frozen=frozen, buffers=buffers, cuda=cuda, written=written)


def call_runtime(runtime):
    cells, path, checksum, output, _ = runtime.frozen.save()
    return exp.prepare_probe(cells, path, checksum, output)


def test_runtime_control_flow_uses_declared_alias_inputs_and_three_sealed_outputs(runtime):
    result = call_runtime(runtime)
    assert result["passed"] and result["validation_only"] and result["source_unchanged"]
    assert result["success_counts"]["P_updates"] == 1 and result["success_counts"]["student_fits"] == 0
    assert set(runtime.written) == {"source_gradient_targets.pt", "step_000000.pt", "step_000001.pt"}
    assert result["alignment_J1"] > result["alignment_J0"]  # Descent is not a runtime gate.
    assert result["source_buffer_digest_before"] == result["source_buffer_digest_after"]
    assert result["anchor0_model_digest_before"] == result["anchor0_model_digest_after"]
    assert all(value["context"]["no_resume_or_continuation"] for value in runtime.written.values())


@pytest.mark.parametrize("mutation", ["fresh", "capacity", "stop", "source_failure", "helper_failure", "buffer", "model0", "dual_failure"])
def test_failure_records_observed_counts_preserves_primary_and_never_qualifies(runtime, monkeypatch, mutation):
    if mutation == "fresh":
        runtime.cuda.is_initialized = lambda: True
    elif mutation == "capacity":
        runtime.cuda.get_device_properties = lambda _: SimpleNamespace(total_memory=1)
    elif mutation == "stop":
        monkeypatch.setattr(exp.probe, "_stop", lambda _: (_ for _ in ()).throw(ValueError("stop primary")))
    elif mutation == "source_failure":
        monkeypatch.setattr(exp, "_load_source", lambda *a: (_ for _ in ()).throw(ValueError("source primary")))
    else:
        def fail(b, saved, ref, e, stop):
            e["counts"]["P_update_attempts"] = 1
            if mutation == "buffer":
                b["source_buffers"]["H"] = "mutation"
            elif mutation == "model0":
                b["model_initial"].append("mutation")
            raise ValueError("helper primary")
        monkeypatch.setattr(exp, "_one_update", fail)
        if mutation == "dual_failure":
            runtime.cuda.synchronize = lambda: (_ for _ in ()).throw(RuntimeError("sync secondary"))
    with pytest.raises(ValueError) as caught:
        call_runtime(runtime)
    output = Path(runtime.frozen.spec["probe_outputs"]["35"])
    saved = json.loads(output.read_text())
    assert saved["passed"] is False and "success_counts" not in saved and runtime.written == {}
    assert saved["counts"].get("P_update_completed", 0) == 0
    if mutation == "dual_failure":
        assert str(caught.value) == "helper primary" and "sync secondary" in saved["native_finalization_error"]
    if mutation in ("buffer", "model0"):
        assert "native_source_preservation_error" in saved


def test_lazy_worker_preserves_options_and_callback(monkeypatch):
    from src.research_loop import dispatch
    calls, expected = [], object()
    stop = lambda: False
    monkeypatch.setitem(sys.modules, "src.cora_gradient_probe", SimpleNamespace(
        prepare_probe=lambda **kwargs: calls.append(kwargs) or expected))
    options = dict(cells=35, spec_path="frozen", spec_sha256="sha", output_path="new")
    assert dispatch(dict(kind="citation_cora_gradient_probe", options=options), stop) is expected
    assert calls == [dict(options, stop=stop)] and "stop" not in options
