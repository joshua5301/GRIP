"""AST-only generated guards: no production/Torch imports or scientific data."""
import ast
import copy
import csv
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

BASE = Path(__file__).resolve().parents[1]
MODULE = BASE / "src/cora_node_reference.py"
REPO = next(p for p in Path(__file__).resolve().parents if (p / "src/transforms.py").is_file())


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exact(actual, expected):
    if isinstance(expected, dict):
        return isinstance(actual, dict) and set(actual) == set(expected) and all(exact(actual[k], v) for k, v in expected.items())
    return type(actual) is type(expected) and actual == expected


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:12]


@pytest.fixture
def exp():
    tree = ast.parse(MODULE.read_text())
    code = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)
                          or isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                          and n.targets[0].id in {"SCIENCE", "SCIENCE_SHA", "CUDA_CAPACITY", "_COMMON"}], type_ignores=[])
    checked = []

    def check_files(pins):
        checked.append(dict(pins))
        require(all(Path(p).is_file() and sha(p) == h for p, h in pins.items()), "Changed immutable input")

    ns = dict(Path=Path, json=json, math=math, csv=csv, platform=platform, time=time, __file__=str(MODULE),
              _require=require, _sha=sha, _exact=exact, _fingerprint=fingerprint,
              _seal=lambda v: json.dumps(v, sort_keys=True),
              probe=SimpleNamespace(_checked_files=check_files, _runtime_precision_guard=lambda: None,
                                    _scalar=lambda v, label: v if type(v) in (float, int) and math.isfinite(v) else require(False, label),
                                    _stop=lambda stop: require(not stop(), "Stopped")),
              torch=SimpleNamespace(get_num_threads=lambda: 4),
              implementation_provenance=lambda: {"git": "current"},
              original=SimpleNamespace(_config=lambda row: row["source_config"], _recipe=lambda row: row["recipe"]))
    exec(compile(code, str(MODULE), "exec"), ns)
    ns["numerical_source"] = lambda: {"numerical": "current"}
    return SimpleNamespace(ns=ns, checked=checked)


@pytest.fixture
def frozen(tmp_path, exp):
    ns = exp.ns
    science = json.loads((REPO / "results/proposals" / ns["SCIENCE"]).read_text())
    # Rebase opaque assets and metadata only. Never deserialize any tensor file.
    science = json.loads(json.dumps(science).replace(str(REPO), str(tmp_path)))
    for p in science["original_files_sha256"]:
        path = Path(p)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("opaque generated asset")
    science["original_files_sha256"] = {p: sha(p) for p in science["original_files_sha256"]}
    refs = [*science["parents"], science["original_BB_certificate"], science["preregistration_revision"]["previous_unexecuted"]]
    for ref in refs:
        path = Path(ref["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("immutable generated parent")
        ref["sha256"] = sha(path)
    science_path = tmp_path / "results/proposals" / ns["SCIENCE"]
    science_path.write_text(json.dumps(science))
    ns["SCIENCE_SHA"] = sha(science_path)
    ns["_science"] = lambda repo: science
    artifact = tmp_path / "results/proposals/review.json"
    artifact.write_text("review")
    spec = dict(schema=1, fixed=copy.deepcopy(science["fixed"]), case=copy.deepcopy(science["case"]),
                baseline_candidate=copy.deepcopy(science["baseline_candidate"]), source={"git": "current"},
                numerical_source={"numerical": "current"}, python_version=platform.python_version(),
                files_sha256=dict(science["original_files_sha256"]),
                scientific_preregistration=dict(path=str(science_path), sha256=ns["SCIENCE_SHA"]),
                artifacts_sha256={str(artifact): sha(artifact)}, outputs=ns["_outputs"](science))
    path = tmp_path / "results/proposals/spec.json"

    def save(seed=1):
        path.write_text(json.dumps(spec))
        return seed, path, sha(path), Path(spec["outputs"][str(seed)]), tmp_path

    return SimpleNamespace(ns=ns, science=science, spec=spec, path=path, save=save, exp=exp)


@pytest.mark.parametrize("seed", [1, 2])
def test_own_condensation_namespace_and_inputs_are_exact(frozen, seed):
    spec, science, output, folder = frozen.ns["_load_spec"](*frozen.save(seed))
    assert str(folder) == science["new_only_reference_dirs"][str(seed)]
    assert output == Path(spec["outputs"][str(seed)])
    assert len(spec["files_sha256"]) == 61
    assert any(p.endswith(f"inputs_{seed}.pt") for p in spec["files_sha256"])
    assert science["preregistration_revision"]["previous_unexecuted"]["path"] in frozen.exp.checked[-1]


@pytest.mark.parametrize("bad", [0, 3, True, 1.0, "1", None])
def test_seed_types_cannot_alias_original_cond0(exp, bad):
    with pytest.raises(ValueError):
        exp.ns["_seed"](bad)


@pytest.mark.parametrize("mutation", ["unknown", "lr", "penalty", "case", "input_missing", "asset_hash", "source", "numerical", "python", "output", "partial"])
def test_resealed_spec_rejects_before_any_scientific_access(frozen, mutation):
    s = frozen.spec
    if mutation == "unknown": s["fallback"] = True
    elif mutation == "lr": s["baseline_candidate"]["lr"] = .01
    elif mutation == "penalty": s["baseline_candidate"]["penalty"] = .001
    elif mutation == "case": s["case"]["root"] = "other"
    elif mutation == "input_missing": s["files_sha256"].pop(next(p for p in s["files_sha256"] if p.endswith("inputs_1.pt")))
    elif mutation == "asset_hash": s["files_sha256"][next(iter(s["files_sha256"]))] = "0" * 64
    elif mutation == "source": s["source"] = {"git": "historical"}
    elif mutation == "numerical": s["numerical_source"] = {"numerical": "old"}
    elif mutation == "python": s["python_version"] = "old"
    elif mutation == "output": s["outputs"]["1"] = str(frozen.path)
    elif mutation == "partial": Path(frozen.science["new_only_reference_dirs"]["1"]).mkdir(parents=True)
    with pytest.raises(ValueError): frozen.ns["_load_spec"](*frozen.save())


@pytest.mark.parametrize("key", ["H", "z", "Q", "transform", "X", "original_CSR", "dense_original_S", "recipe"])
def test_common_BB_provenance_has_no_source_substitution(exp, tmp_path, key):
    ns = exp.ns
    native = {k: f"original:{k}" for k in ns["_COMMON"]}
    native["hard"] = "ownseed1"
    recipe = {"path": "own original recipe"}
    path = tmp_path / "bb.json"
    path.write_text(json.dumps(dict(passed=True, source_assets_spec_science_unchanged=True,
        roots={"own": dict(passed=True, source_P0_linear_certificate_passed=True,
                           native_buffers=dict(native, hard="historical0"), student_recipe_origin=recipe)})))
    science = dict(original_BB_certificate=dict(path=str(path), sha256=sha(path)))
    ns["_bind_common"](science, {"root": "own"}, native, recipe)
    if key == "recipe": recipe = {"path": "wrong"}
    else: native[key] = "substituted"
    with pytest.raises(ValueError): ns["_bind_common"](science, {"root": "own"}, native, recipe)


@pytest.mark.parametrize("seed", [1, 2])
def test_exact_legacy_core_call_without_students_or_resume(exp, tmp_path, seed):
    calls = []
    sentinel = object()
    exp.ns["optimize_ce_assignment"] = lambda *args, **kw: calls.append((args, kw)) or sentinel
    buffers = dict(z="originalz", q="originalQ", hard="ownhard")
    folder = tmp_path / f"condensation_{seed}"
    evidence = dict(counts={})
    stop = lambda: False
    assert exp.ns["_run_core"](buffers, seed, folder, evidence, stop) is sentinel
    args, kw = calls[0]
    assert args == ("originalz", "originalQ", "ownhard")
    assert kw == dict(penalty=.0001, steps=25, mixing=.05, lr=.05, assignment_rank=32, factor_seed=seed,
        assignment_input="node", assignment_encoder="linear", encoder_hidden=64, solver_mode="exact",
        inner_method="newton_first", implicit_warm_start=True, mass_mode="free", inner_loss_weighting="uniform",
        inner_max_iter=2000, inner_tol=1e-7, cg_max_iter=512, cg_rtol=1e-6, cache_assignment=False,
        folder=folder, checkpoint_steps=(0, 25), resume_state=None, save_resume=True, save_assignment=False, stop=stop)
    assert evidence["counts"]["P_update_completed"] == 25
    folder.mkdir()
    with pytest.raises(ValueError): exp.ns["_run_core"](buffers, seed, folder, evidence, stop)
    assert len(calls) == 1


def rows():
    return [dict(step=i, J=.8, inner_grad_max=1e-8, seconds=float(i), inner_converged=True,
                 J_exact=True, status="evaluated" if i == 25 else "update", cg_converged=i < 25,
                 cg_relative_residual=1e-8 if i < 25 else float("nan")) for i in range(26)]


@pytest.mark.parametrize("mutation", [None, "gap", "cg", "head", "csv"])
def test_contiguous_certified_history_distinguishes_terminal_no_CG(exp, tmp_path, mutation):
    history = rows()
    path = tmp_path / "history.csv"
    if mutation == "gap": history[12]["step"] = 13
    elif mutation == "cg": history[12]["cg_converged"] = False
    elif mutation == "head": history[12]["inner_grad_max"] = 1e-3
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history[:-1] if mutation == "csv" else history)
    if mutation is None:
        assert exp.ns["_history"](dict(history=history), path) == history
    else:
        with pytest.raises(ValueError): exp.ns["_history"](dict(history=history), path)


@pytest.mark.parametrize("failure", ["memory", "buffers"])
def test_cleanup_checks_all_pins_even_if_native_finalization_fails(exp, tmp_path, failure):
    ns = exp.ns
    original = RuntimeError("primary optimizer error")
    called = []
    ns["_preserve"] = lambda *args: called.append("files preserved")
    ns["_native_buffers"] = lambda value: {"changed": True}
    ns["torch"].cuda = SimpleNamespace(synchronize=lambda: (_ for _ in ()).throw(RuntimeError("sync failed")))
    evidence = dict(passed=False, native_buffers_before={"original": True})
    got = ns["_finish"]({}, {}, tmp_path / "spec", "sha", {} if failure == "buffers" else None,
                         True, time.monotonic(), evidence, original)
    assert got is original and called == ["files preserved"]
    assert evidence["source_assets_spec_science_unchanged"] is True and evidence["passed"] is False
    assert "memory_error" in evidence
    if failure == "buffers": assert "native_buffer_preservation_error" in evidence


def test_lazy_worker_preserves_callback_and_options(monkeypatch):
    tree = ast.parse((BASE / "src/research_loop.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "dispatch")
    calls = []
    fake = SimpleNamespace(prepare=lambda **kw: calls.append(kw) or "receipt")
    monkeypatch.setitem(sys.modules, "src.cora_node_reference", fake)
    ns = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "worker-dispatch", "exec"), ns)
    stop = lambda: False
    options = dict(condensation_seed=1, spec_path="spec", spec_sha256="sha", output_path="evidence")
    assert ns["dispatch"](dict(kind="citation_cora_node_reference_prepare", options=options), stop) == "receipt"
    assert calls == [dict(options, stop=stop)] and "stop" not in options
