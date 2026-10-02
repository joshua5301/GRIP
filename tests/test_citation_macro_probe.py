"""CPU synthetic/AST QA only: no native source, datasets or CUDA invocation."""
import ast
import hashlib
import json
import math
import platform
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src.low_rank_assignment import LowRankMoments, initialize_factors
from src.macro_teacher_outer import macro_teacher_outer
from src.moments import augmented, decode_moments, make_material
from src.soft_ce_partition import (
    head_gradient,
    hessian_operator,
    implicit_moment_gradient,
    solve_head_system,
    solve_inner_newton_first,
)

REPO = Path(__file__).resolve().parents[1]
PATH = REPO / "src/citation_macro_probe.py"


def require(value, message):
    if not value:
        raise ValueError(message)


def tensor(value, shape, dtype, message):
    require(torch.is_tensor(value) and value.layout == torch.strided and tuple(value.shape) == tuple(shape)
            and value.dtype == dtype and bool(torch.isfinite(value).all()), message)
    return value


def cpu_state(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_state(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [cpu_state(item) for item in value]
    return value


def extract(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def math_namespace():
    def count(evidence, key, complete=False):
        key += "_completed" if complete else "_attempts"
        evidence["counts"][key] = evidence["counts"].get(key, 0)+1

    def stop(callback):
        if callback():
            raise InterruptedError("stopped")

    inherited = extract(REPO / "src/citeseer_finite_student_v2.py", {"_adam_step", "_optimizer"},
                        dict(torch=torch, math=math, _require=require, _tensor=tensor, HORIZON=25))
    probe = SimpleNamespace(_runtime_precision_guard=lambda: None, cpu_state=cpu_state,
        _moments=lambda b, p: LowRankMoments.apply(*p, b["hard"], make_material(b["z"], b["q"]), .05, 4096))
    from src.low_rank_assignment import logit_block
    names = {"_material", "_head_certificate", "_one_update"}
    return extract(PATH, names, dict(torch=torch, math=math, _tensor=tensor, _require=require, _stop=stop,
        _count=count, probe=probe, inherited=SimpleNamespace(**inherited), logit_block=logit_block,
        decode_moments=decode_moments, make_material=make_material, augmented=augmented,
        head_gradient=head_gradient, hessian_operator=hessian_operator, macro_teacher_outer=macro_teacher_outer,
        implicit_moment_gradient=implicit_moment_gradient, solve_head_system=solve_head_system,
        solve_inner_newton_first=solve_inner_newton_first))


def toy(dtype=torch.float64, interior=False):
    g = torch.Generator().manual_seed(4102)
    z = torch.randn(15, 4, generator=g, dtype=torch.float64)*.3
    q = (torch.randn(15, 3, generator=g, dtype=torch.float64)+torch.tensor([1.5, .0, -.7])).softmax(1)
    hard = torch.arange(15) % 3
    if dtype == torch.float32:
        u, v = initialize_factors(hard, 3, 2, 0)
    else:
        u = torch.zeros(15, 2, dtype=dtype)
        v = torch.randn(3, 2, generator=g, dtype=dtype)
    if interior:
        u = torch.randn(15, 2, generator=g, dtype=dtype)*.02
    return dict(z=z, q=q, hard=hard, initial=[u.detach(), v.detach()])


def fit(moments, tol=1e-11):
    x, q, m = decode_moments(moments, 4)
    fitted = solve_inner_newton_first(x, q, torch.full_like(m, 1/len(m)), .001,
        initial=torch.zeros(3, 5, dtype=torch.float64), grad_tol=tol, newton_steps=20)
    assert fitted["inner_converged"]
    return fitted["theta"]


def test_actual_connected_implicit_Adam_and_P1_head(record_property):
    ns = math_namespace()
    buffers = toy(torch.float32)
    before = cpu_state(buffers)
    m0 = ns["probe"]._moments(buffers, buffers["initial"])
    theta0 = fit(m0)
    evidence = dict(counts={})
    states = ns["_one_update"](buffers, dict(moments=m0.clone(), theta=theta0), evidence, lambda: False)
    assert torch.equal(states[0]["moments"], m0) and torch.equal(states[0]["theta"], theta0)
    assert states[0]["optimizer"]["state"] == {} and set(states[1]["optimizer"]["state"]) == {0, 1}
    assert torch.equal(states[0]["parameters"][1], states[1]["parameters"][1])  # V gradient is zero at U0.
    assert not torch.equal(states[0]["parameters"][0], states[1]["parameters"][0])
    assert all(evidence["counts"][key+"_completed"] == 1 for key in
        ("P_update", "adjoint_solve", "implicit_moment_partial", "original_moment_backward", "endpoint_head_solve"))
    assert evidence["counts"]["macro_outer_completed"] == 2
    assert all(torch.equal(a, b) for a, b in zip(buffers["initial"], before["initial"], strict=True))
    assert states[1]["uniform_inner_grad_max"] <= 1e-7
    assert states[0]["scale"] == states[0]["macro_ce"]
    record_property("toy_top_level_head_fits", 2)  # reference P0 and actual P1, no dataset fits.
    record_property("adjoint_relative_residual", states[0]["adjoint"]["cg_relative_residual"])
    record_property("P1_grad_max", states[1]["uniform_inner_grad_max"])


@pytest.mark.parametrize("epsilon", [1e-4, 3e-5])
@pytest.mark.parametrize("direction", ["u", "both"])
def test_full_converged_head_macro_factor_directional_FD(epsilon, direction, record_property):
    b = toy(interior=True)
    u, v = [p.clone().requires_grad_() for p in b["initial"]]
    m = LowRankMoments.apply(u, v, b["hard"], make_material(b["z"], b["q"]), .05, 4096)
    theta = fit(m.detach())
    value, rhs = macro_teacher_outer(b["z"], b["q"], theta)
    x, q, mass = decode_moments(m.detach(), 4)
    vec, cert = solve_head_system(augmented(x), q, torch.full_like(mass, 1/3), theta, .001, rhs,
                                rtol=1e-10, atol=1e-13, max_iter=512, initial=None, cg_check_interval=1)
    assert cert["cg_converged"]
    partial = implicit_moment_gradient(m, 4, theta, vec, .001, loss_weighting="uniform")
    m.backward(partial/value)
    du = torch.linspace(-.7, .4, u.numel(), dtype=u.dtype).reshape_as(u)
    dv = torch.linspace(.1, -.3, v.numel(), dtype=v.dtype).reshape_as(v) if direction == "both" else torch.zeros_like(v)
    observed = []
    for sign in (-1, 1):
        candidate = LowRankMoments.apply(u.detach()+sign*epsilon*du, v.detach()+sign*epsilon*dv,
                                        b["hard"], make_material(b["z"], b["q"]), .05, 4096)
        newtheta = fit(candidate)
        observed.append(macro_teacher_outer(b["z"], b["q"], newtheta)[0]/value)
    fd = (observed[1]-observed[0])/(2*epsilon)
    analytic = float((u.grad*du).sum()+(v.grad*dv).sum())
    record_property("FD_abs", abs(fd-analytic))
    record_property("reference_head_fits", 3)
    assert abs(fd-analytic) <= 3e-7+2e-5*abs(fd)


@pytest.mark.parametrize("mutation", ["container", "moment", "head", "stop"])
def test_reject_before_update(mutation):
    ns = math_namespace()
    b = toy(torch.float32)
    m = ns["probe"]._moments(b, b["initial"])
    saved = dict(moments=m.clone(), theta=fit(m))
    if mutation == "container":
        saved = None
    elif mutation == "moment":
        saved["moments"][0, 0] += 1e-9
    elif mutation == "head":
        saved["theta"][0, 0] += .1
    e = dict(counts={})
    with pytest.raises((ValueError, InterruptedError)):
        ns["_one_update"](b, saved, e, lambda: mutation == "stop")
    assert e["counts"].get("P_update_attempts", 0) == 0


def spec_fixture(tmp_path):
    repo = tmp_path / "repo"
    folder = repo / "results/proposals"
    folder.mkdir(parents=True)
    tree = ast.parse(PATH.read_text())
    assignment = next(n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "_FIXED" for t in n.targets))
    fixed = {kw.arg: ast.literal_eval(kw.value) for kw in assignment.value.keywords}
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    asset = repo / "results/asset"; asset.write_text("immutable")
    pins = {str(asset): sha(asset)}
    science = dict(candidate=fixed, original_files_sha256=pins, parents=[])
    science_path = folder / "science.json"; science_path.write_text(json.dumps(science))
    root = str(repo / "results/research_loop/AW")
    spec = dict(schema=1, scientific_preregistration=dict(path=str(science_path), sha256=sha(science_path)),
        source={"actual": True}, numerical_source={"versions": "fixed"}, python_version=platform.python_version(),
        files_sha256=pins, roots=[{"cells": 30}, {"cells": 120}], candidate=fixed,
        artifacts_sha256=pins, output_root=root, probe_outputs={"30": root+"/c30.json", "120": root+"/c120.json"})
    path = folder / "spec.json"
    def checked(files):
        require(all(Path(p).is_file() and sha(p) == h for p, h in files.items()), "changed file")
    ns = extract(PATH, {"_preserve", "_load_spec"}, dict(__file__=str(repo / "src/citation_macro_probe.py"),
        Path=Path, json=json, platform=platform, torch=SimpleNamespace(get_num_threads=lambda: 4), _FIXED=fixed,
        SCIENCE="science.json", SCIENCE_SHA=sha(science_path), _require=require, _sha=sha,
        _exact=lambda a, e: type(a) is type(e) and a == e and not any(type(a[k]) is bool for k in a if type(e[k]) is int),
        implementation_provenance=lambda: {"actual": True}, numerical_source=lambda: {"versions": "fixed"},
        probe=SimpleNamespace(_checked_files=checked), original=SimpleNamespace(_roots=lambda r: spec["roots"])))
    return spec, path, ns, sha, asset


@pytest.mark.parametrize("mutation", [None, "unknown", "bool_schema", "candidate", "source", "duplicate", "asset"])
def test_spec_fail_closed_before_native(tmp_path, mutation):
    spec, path, ns, sha, asset = spec_fixture(tmp_path)
    if mutation == "unknown": spec["external_Q"] = "override"
    elif mutation == "bool_schema": spec["schema"] = True
    elif mutation == "candidate": spec["candidate"] = dict(spec["candidate"], cg_rtol=.1)
    elif mutation == "source": spec["source"] = {"actual": False}
    elif mutation == "duplicate": spec["probe_outputs"]["120"] = spec["probe_outputs"]["30"]
    elif mutation == "asset": asset.write_text("tampered")
    path.write_text(json.dumps(spec))
    if mutation is None:
        assert ns["_load_spec"](path, sha(path))[0] == spec
    else:
        with pytest.raises(ValueError): ns["_load_spec"](path, sha(path))


@pytest.mark.parametrize("cells", [True, 60, 30.0, "30"])
def test_public_cells_preflight_before_spec_or_native(tmp_path, cells):
    calls = []
    ns = extract(PATH, {"prepare_probe"}, dict(inherited=SimpleNamespace(_budget=lambda c: require(type(c) is int and c in (30, 120), "cells")),
        _require=require, _load_spec=lambda *x: calls.append("spec"), Path=Path))
    with pytest.raises(ValueError): ns["prepare_probe"](cells, "unused", "unused", "unused")
    assert calls == []


def test_primary_source_error_survives_finalization_error(tmp_path):
    spec = dict(probe_outputs={"30": str(tmp_path / "out.json")}, numerical_source={}, source={}, python_version="test",
                scientific_preregistration={}, files_sha256={})
    primary = ValueError("strict cached_source failed")
    written = []
    class CUDA:
        is_initialized = staticmethod(lambda: False)
        reset_peak_memory_stats = staticmethod(lambda: None)
        current_device = staticmethod(lambda: 0)
        get_device_properties = staticmethod(lambda _: SimpleNamespace(total_memory=8))
        synchronize = staticmethod(lambda: (_ for _ in ()).throw(RuntimeError("sync failed")))
    ns = extract(PATH, {"prepare_probe"}, dict(Path=Path, platform=platform, time=__import__("time"),
        torch=SimpleNamespace(get_num_threads=lambda: 4, cuda=CUDA), original=SimpleNamespace(CUDA_CAPACITY=8),
        inherited=SimpleNamespace(_budget=lambda c: None, _load_source=lambda *a: (_ for _ in ()).throw(primary)),
        probe=SimpleNamespace(_native=lambda d: {}, _runtime_precision_guard=lambda: None),
        _load_spec=lambda *a: (spec, {}), _require=require, _stop=lambda cb: require(not cb(), "stop"),
        _FIXED={}, _seal=lambda v: "token", _fingerprint=lambda v: "candidate", POLICY="fixed",
        _preserve=lambda *a: None, _write_new=lambda p, e: written.append(dict(e)), _sha=lambda p: "sha"))
    with pytest.raises(ValueError, match="strict cached_source failed") as observed:
        ns["prepare_probe"](30, tmp_path / "spec", "sha", tmp_path / "out.json")
    assert observed.value is primary and written[0]["passed"] is False
    assert written[0]["source_unchanged"] is True and "sync failed" in written[0]["native_finalization_error"]
    assert written[0]["counts"]["P_update_completed"] == 0


def test_real_dispatch_lazy_options_callback_and_result(monkeypatch):
    expected = object()
    calls = []
    stop = lambda: False
    module = SimpleNamespace(prepare_probe=lambda **kwargs: calls.append(kwargs) or expected)
    monkeypatch.setitem(sys.modules, "src.citation_macro_probe", module)
    ns = extract(REPO / "src/research_loop.py", {"dispatch"}, {})
    options = dict(cells=30, spec_path="fixed", spec_sha256="sha", output_path="new")
    assert ns["dispatch"](dict(kind="citation_macro_teacher_probe", options=options), stop) is expected
    assert calls == [dict(options, stop=stop)] and "stop" not in options
