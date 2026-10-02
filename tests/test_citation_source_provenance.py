"""Tiny CPU/AST mocks only; no actual source/cache/data/native module imports."""
import ast
import copy
import hashlib
import inspect
import json
import math
import platform
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def tensor(value, shape, dtype, message):
    require(torch.is_tensor(value) and value.layout == torch.strided and tuple(value.shape) == tuple(shape)
            and value.dtype == dtype and bool(torch.isfinite(value).all()), message)
    return value


@pytest.fixture
def scope():
    base = Path(__file__).resolve().parents[1]
    module = base / "citation_source_provenance.py"
    if not module.is_file():
        module = base / "src/citation_source_provenance.py"
    namespace = dict(Path=Path, hashlib=hashlib, inspect=inspect, json=json, math=math, platform=platform,
                     time=time, torch=torch, __file__=str(module))
    repo = next(parent for parent in (base, *Path(__file__).resolve().parents)
                if (parent / "src/transforms.py").is_file())
    transform = next(node for node in ast.parse((repo / "src/transforms.py").read_text()).body
                     if isinstance(node, ast.ClassDef) and node.name == "FeatureTransform")
    exact = next(node for node in ast.parse((repo / "src/citation_source_preflight.py").read_text()).body
                 if isinstance(node, ast.FunctionDef) and node.name == "_exact")
    exec(compile(ast.Module(body=[transform, exact], type_ignores=[]), str(module), "exec"), namespace)
    def checked_files(pins):
        for path, digest in pins.items():
            require(Path(path).is_file() and sha(path) == digest, "Frozen file changed")
    namespace.update(probe=SimpleNamespace(_require=require, _tensor=tensor, _sha=sha, _checked_files=checked_files,
                                          numerical_source=lambda: {"mock_versions_and_source": True},
                                          _stop=lambda callback: require(not callback(), "Stopped")),
                     implementation_provenance=lambda: {"mock_fullsource_Git": True},
                     _fingerprint=lambda value: hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:12])
    tree = ast.parse(module.read_text())
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.Assign))]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(module), "exec"), namespace)
    namespace["module"] = module
    return namespace


@pytest.fixture
def toy():
    h = torch.tensor([[0., 1., 2.], [3., 0., 4.]], dtype=torch.float32)
    state = dict(center=torch.tensor([.5, .25, .125], dtype=torch.float64), matrix=None,
                 output_center=torch.tensor([.125, .25, .5], dtype=torch.float64),
                 scale=torch.tensor(2., dtype=torch.float64), kind="rms", eps=1e-12)
    z = ((h.double()-state["center"])-state["output_center"])/state["scale"]
    return h, z, state


def test_coherent_toy_inverse_and_strict_replay_do_not_accept_source(scope, toy):
    h, z, state = toy
    before = [x.clone() for x in (h, z, state["center"], state["output_center"], state["scale"])]
    report = scope["diagnose_rms"](h.requires_grad_(), z.requires_grad_(), state)
    assert report["strict_original_H_RMS"]["strict_allclose"]
    assert report["inverse_FP32_reencode"]["strict_allclose"]
    assert report["cast_FP32_changed_coordinates"] == 0
    assert report["positive_finite_ULP_max"] == 0
    assert not report["source_accepted"] and not report["historical_H_recovered"]
    for current, original in zip((h, z, state["center"], state["output_center"], state["scale"]), before):
        assert torch.equal(current, original)
    assert h.grad is None and z.grad is None


def test_inverse_FP32_reencode_pass_cannot_waive_strict_original_source(scope):
    h = torch.tensor([[1., 0.]], dtype=torch.float32)
    proxy = h.clone()
    proxy[0, 0] = torch.nextafter(proxy[0, 0], torch.tensor(float("inf")))
    state = dict(center=torch.zeros(2, dtype=torch.float64), output_center=torch.zeros(2, dtype=torch.float64),
                 scale=torch.tensor(1., dtype=torch.float64), matrix=None, kind="rms", eps=1e-12)
    report = scope["diagnose_rms"](h, proxy.double(), state)
    assert not report["strict_original_H_RMS"]["strict_allclose"]
    assert report["inverse_FP32_reencode"]["strict_allclose"]
    assert report["positive_finite_ULP_max"] == 1 and report["cast_FP32_changed_coordinates"] == 1
    assert not report["source_accepted"]


def test_cancellation_zero_proxy_is_descriptive_ulp_excludes_it(scope):
    h = torch.tensor([[0., 1.]], dtype=torch.float32)
    state = dict(center=torch.tensor([.1, .2], dtype=torch.float64), output_center=torch.tensor([.2, .1], dtype=torch.float64),
                 scale=torch.tensor(1., dtype=torch.float64), matrix=None, kind="rms", eps=1e-12)
    z = ((h.double()-state["center"])-state["output_center"])/state["scale"]
    report = scope["diagnose_rms"](h, z, state)
    assert report["inverse_nonzero_at_saved_zero"] == 1
    assert report["cast_nonzero_at_saved_zero"] == 1
    assert report["positive_finite_ULP_coordinates"] == 1
    assert "cancellation" in report["zero_cancellation_caveat"]


@pytest.mark.parametrize("kind", ["H64", "z32", "z_shape", "H_nan", "z_inf", "scale_zero", "scale_nan",
                                  "scale_vector", "eps_bool", "eps_zero", "matrix", "kind", "extra", "center32"])
def test_toy_invalid_transform_or_input_rejects_cleanly(scope, toy, kind):
    h, z, state = toy
    if kind == "H64":
        h = h.double()
    elif kind == "z32":
        z = z.float()
    elif kind == "z_shape":
        z = z[:1]
    elif kind in ("H_nan", "z_inf"):
        (h if kind == "H_nan" else z)[0, 0] = float("nan" if kind == "H_nan" else "inf")
    elif kind in ("scale_zero", "scale_nan", "scale_vector"):
        state["scale"] = torch.tensor([1.] if kind == "scale_vector" else 0. if kind == "scale_zero" else float("nan"), dtype=torch.float64)
    elif kind.startswith("eps"):
        state["eps"] = True if kind == "eps_bool" else 0.
    elif kind == "matrix":
        state["matrix"] = torch.eye(3)
    elif kind == "kind":
        state["kind"] = "l2"
    elif kind == "extra":
        state["external_mean"] = 0.
    else:
        state["center"] = state["center"].float()
    with pytest.raises(ValueError):
        scope["diagnose_rms"](h, z, state)


@pytest.fixture
def frozen(scope, tmp_path):
    repo = tmp_path / "repo"
    (repo / "data/citeseer/processed").mkdir(parents=True)
    (repo / "data/citeseer/processed/data.pt").write_text("mock processed")
    files = scope["_paths"](repo)
    for path in files:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("mock immutable " + str(path))
    parents = []
    for i in range(3):
        path = repo / "results/proposals" / f"parent_{i}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("mock frozen lineage")
        parents.append(dict(path=str(path), sha256=sha(path)))
    science = repo / "results/proposals" / scope["SCIENCE"]
    science.write_text(json.dumps(dict(fixed=scope["FIXED"], roots=scope["_roots"](repo), parents=parents)))
    scope["SCIENCE_SHA"] = sha(science)
    output = repo / "results/research_loop/new_stage/evidence.json"
    spec = dict(schema=1, fixed=copy.deepcopy(scope["FIXED"]), roots=scope["_roots"](repo),
                source=scope["implementation_provenance"](), numerical_source=scope["probe"].numerical_source(),
                python_version=platform.python_version(), files_sha256={str(path): sha(path) for path in files},
                scientific_preregistration=dict(path=str(science), sha256=scope["SCIENCE_SHA"]), output_path=str(output))
    path = repo / "results/proposals/frozen_spec.json"
    scope["_config"] = lambda root: {"mock_config_guard": root["root"]}
    def save():
        path.write_text(json.dumps(spec))
        return str(path), sha(path), str(output), repo
    return SimpleNamespace(repo=repo, spec=spec, path=path, output=output, parents=parents, files=files, save=save)


def test_exact_path_inventory_spec_no_native_or_output_creation(scope, frozen):
    spec, parents, output = scope["_load_spec"](*frozen.save())
    assert spec == frozen.spec and parents == frozen.parents and output == frozen.output
    assert len(frozen.files) == 33 and len(set(frozen.files)) == 33
    assert not output.parent.exists()


@pytest.mark.parametrize("kind", ["schema_bool", "fixed_bool", "seed", "root", "root_bool", "extra", "source",
                                  "versions", "python", "science", "missing", "tamper", "output_exists"])
def test_resealed_spec_or_asset_mutations_reject_before_native(scope, frozen, kind):
    spec = frozen.spec
    if kind == "schema_bool":
        spec["schema"] = True
    elif kind in ("fixed_bool", "seed"):
        spec["fixed"]["condensation_seed"] = False if kind == "fixed_bool" else 1
    elif kind == "root":
        spec["roots"][0]["source_root"] = "/arbitrary"
    elif kind == "root_bool":
        spec["roots"][0]["cells"] = True
    elif kind == "extra":
        spec["external_H"] = "forbidden"
    elif kind == "source":
        spec["source"]["mock_fullsource_Git"] = False
    elif kind == "versions":
        spec["numerical_source"]["mock_versions_and_source"] = False
    elif kind == "python":
        spec["python_version"] = "other"
    elif kind == "science":
        spec["scientific_preregistration"]["sha256"] = "0"*64
    elif kind == "missing":
        frozen.files[0].unlink()
    elif kind == "tamper":
        frozen.files[0].write_text("tampered bytes")
    else:
        frozen.output.parent.mkdir(parents=True)
        frozen.output.write_text("prior report")
    with pytest.raises(ValueError):
        scope["_load_spec"](*frozen.save())


def test_later_lineage_and_source_drift_reject(scope, frozen):
    scope["_load_spec"](*frozen.save())
    Path(frozen.parents[0]["path"]).write_text("later drift")
    with pytest.raises(ValueError):
        scope["_load_spec"](*frozen.save())
    scope["implementation_provenance"] = lambda: {"changed": True}
    with pytest.raises(ValueError):
        scope["_source_unchanged"](frozen.spec)


@pytest.mark.parametrize("failure", [None, "load", "load_memory", "source", "threads", "fresh", "capacity"])
def test_fake_future_execution_counts_failure_preservation_and_no_source_acceptance(scope, frozen, failure):
    scope["__file__"] = str(frozen.repo / "src/citation_source_provenance.py")
    frozen.save()
    calls, reports = [], []
    def synchronize():
        calls.append("sync")
        if failure == "load_memory":
            raise RuntimeError("secondary memory collection error")
    cuda = SimpleNamespace(is_initialized=lambda: failure == "fresh", reset_peak_memory_stats=lambda: calls.append("reset"),
                           synchronize=synchronize, max_memory_allocated=lambda: 10,
                           max_memory_reserved=lambda: 20, current_device=lambda: 0,
                           get_device_properties=lambda device: SimpleNamespace(total_memory=1 if failure == "capacity" else scope["CUDA_CAPACITY"]))
    scope["torch"] = SimpleNamespace(cuda=cuda, get_num_threads=lambda: 1 if failure == "threads" else 4)
    scope["probe"]._native = lambda device: calls.append("native") or {"mock_only": True}
    scope["probe"]._runtime_precision_guard = lambda: calls.append("precision")
    def load_pair(root, evidence):
        calls.append(root["root"])
        evidence["existing_H_RMS_tensor_loads"] += 2
        if failure in ("load", "load_memory"):
            raise ValueError("original malformed RMS error")
        if failure == "source":
            scope["implementation_provenance"] = lambda: {"later_drift": True}
        return {"mock_strict_RMS_failure_is_observation": True, "source_accepted": False}
    scope["_load_pair"] = load_pair
    def write_report(path, value):
        reports.append(copy.deepcopy(value))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    scope["_write_new"] = write_report
    if failure is None:
        result = scope["prepare_provenance"](*frozen.save()[:3])
        assert result["passed"] and not result["source_accepted"]
    else:
        with pytest.raises(ValueError, match="original malformed RMS error" if failure in ("load", "load_memory") else ".*"):
            scope["prepare_provenance"](*frozen.save()[:3])
    report = reports[0]
    assert not report["source_accepted"] and not report["historical_H_recovered"]
    assert all(report[key] == 0 for key in ("P_updates", "optimizer_steps", "student_fits", "head_solves",
                                           "teacher_map_Phi_fits", "model_forwards", "finite_unrolls", "native_initializers"))
    assert report["existing_H_RMS_tensor_loads"] == (6 if failure in (None, "source", "capacity") else 2 if failure in ("load", "load_memory") else 0)
    assert report["passed"] is (failure is None)
    if failure in ("load", "load_memory"):
        assert report["failed_stage"].startswith("readonly_H_RMS_")
        assert report["source_error"]["message"] == "original malformed RMS error"
        if failure == "load":
            assert report["source_assets_spec_science_unchanged"]
        else:
            assert report["memory_error"]["message"] == "secondary memory collection error"
            assert not report["source_assets_spec_science_unchanged"]
    if failure == "source":
        assert not report["source_assets_spec_science_unchanged"]


def test_actual_worker_dispatch_lazy_preserves_options_stop_return(scope, monkeypatch):
    worker = scope["module"].parent / "research_loop.py"
    function = next(node for node in ast.parse(worker.read_text()).body if isinstance(node, ast.FunctionDef) and node.name == "dispatch")
    calls = []
    module = ModuleType("src.citation_source_provenance")
    module.prepare_provenance = lambda **options: calls.append(options) or {"mock": True}
    monkeypatch.setitem(sys.modules, "src.citation_source_provenance", module)
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(worker), "exec"), namespace)
    stop = lambda: False
    result = namespace["dispatch"](dict(kind="citation_source_provenance", options={"spec_path": "fixed", "spec_sha256": "digest", "output_path": "new"}), stop)
    assert result == {"mock": True} and calls == [dict(spec_path="fixed", spec_sha256="digest", output_path="new", stop=stop)]


@pytest.mark.parametrize("kind", [None, "fingerprint", "dataset", "ratio_bool", "features"])
def test_actual_config_digest_and_preprocessing_guard(scope, tmp_path, kind):
    config = dict(dataset="citeseer", ratio=.009, citation_features="row", data_digest="mock")
    root = dict(source_root=str(tmp_path), root=scope["_fingerprint"](config), ratio=.009, citation_features="row")
    if kind == "fingerprint":
        root["root"] = "other"
    elif kind == "dataset":
        config["dataset"] = "cora"
    elif kind == "ratio_bool":
        config["ratio"] = True
    elif kind == "features":
        config["citation_features"] = "default"
    (tmp_path / "config.json").write_text(json.dumps(config))
    if kind is None:
        assert scope["_config"](root) == config
    else:
        with pytest.raises(ValueError):
            scope["_config"](root)
