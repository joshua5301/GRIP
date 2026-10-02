"""Actual CPU Adam recurrence plus AST/mock persistence/phase integration QA.

No head/adjoint mathematics is re-run here; AW already covers that connection.
The replay objective is a declared synthetic stub, never a real-data certificate.
"""
import ast
import hashlib
import json
import math
import os
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src.citation_source_preflight import _exact
from src.io import array_digest, cpu_state

REPO = Path(__file__).resolve().parents[1]
PATH = REPO / "src/citation_macro_fixed25.py"


def extract(path, names, namespace):
    nodes = [n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def namespace():
    pure = extract(REPO / "src/finite_student_probe.py",
        {"_require", "_tensor", "_digest", "_seal", "_attach", "_atomic", "_sha"},
        dict(torch=torch, json=json, hashlib=hashlib, array_digest=array_digest, cpu_state=cpu_state, Path=Path, os=os, uuid=uuid))
    require, tensor = pure["_require"], pure["_tensor"]
    adam = extract(REPO / "src/citeseer_finite_student_v2.py", {"_adam_step", "_optimizer"},
                   dict(torch=torch, math=math, _require=require, _tensor=tensor, HORIZON=25))
    def stop(callback):
        if callback(): raise InterruptedError("stopped")
    def count(evidence, key, complete=False):
        key += "_completed" if complete else "_attempts"
        evidence["counts"][key] = evidence["counts"].get(key, 0)+1
    def fake_state(buffers, parameters, theta, optimizer, step, scale, work, evidence, stop, replay=None):
        loss = 2+sum(float(p.detach().double().square().mean()) for p in parameters)
        scale = loss if scale is None else scale
        state = dict(step=step, parameters=parameters, optimizer=optimizer, theta=theta, macro_ce=loss,
            J0=scale, uniform_inner_grad_max=0., head_work=work, terminal_no_update=step == 25)
        if step < 25:
            state.update(scaled_factor_gradients=[(p.detach().sin()+.05)/scale for p in parameters], explicit_adjoint_residual=0.)
        if step in (0, 25):
            # Synthetic material only; this fixture does not claim source/P proof.
            state["moments"] = torch.ones(3, 8, dtype=torch.float64)*loss
        return cpu_state(state), scale
    def rep(moments, transform, dimension, device):
        return moments[:, 1:3].float(), moments[:, -3:].float(), torch.full((3,), 1/3, dtype=torch.float64)
    probe = SimpleNamespace(**pure, _transform=lambda b: None)
    ns = dict(Path=Path, json=json, torch=torch, math=math, HORIZON=25, _require=require, _tensor=tensor,
        _sha=pure["_sha"], _seal=pure["_seal"], _exact=_exact, _stop=stop, _count=count,
        inherited=SimpleNamespace(**adam), probe=probe, _state=fake_state, representative=rep,
        original=SimpleNamespace(REFERENCE="original_NODE"))
    extract(PATH, {"_history", "_check_progress", "_cache_files", "_load_progress", "_store", "_gate", "_prepare"}, ns)
    return ns


@pytest.fixture(scope="module")
def prefix():
    ns = namespace()
    generator = torch.Generator().manual_seed(1103)
    initial = [torch.zeros(5, 2), torch.randn(3, 2, generator=generator)*.1]
    parameters = [p.clone().requires_grad_() for p in initial]
    optimizer = torch.optim.Adam(parameters, lr=.01, betas=(.9, .999), eps=1e-12, weight_decay=0, foreach=False, fused=False)
    buffers = dict(initial=initial, z=torch.zeros(5, 4, dtype=torch.float64), q=torch.full((5, 3), 1/3, dtype=torch.float64), cells=3)
    states, scale = {}, None
    first, second = [torch.zeros_like(p) for p in initial], [torch.zeros_like(p) for p in initial]
    for step in range(26):
        state, scale = ns["_state"](buffers, parameters, torch.zeros(3, 5, dtype=torch.float64),
            optimizer.state_dict(), step, scale, {}, dict(counts={}), lambda: False)
        states[step] = state
        ns["inherited"]._optimizer(state["optimizer"], parameters, first, second, step)
        if step == 25: break
        gradients = state["scaled_factor_gradients"]
        expected, first, second = ns["inherited"]._adam_step(parameters, first, second, gradients, step+1)
        for parameter, gradient in zip(parameters, gradients, strict=True): parameter.grad = gradient.clone()
        optimizer.step()
        assert all(torch.equal(p, q) for p, q in zip(parameters, expected, strict=True))
    bundle = ns["probe"]._attach(dict(schema=1, context={"candidate": {"method": "fixture"}}, frontier=25,
        states=states, J0=scale, history=ns["_history"](states)))
    return ns, buffers, bundle


def reseal(ns, bundle):
    return ns["probe"]._attach({k: v for k, v in bundle.items() if k != "content_sha256"})


def test_actual_all25_Adam_and_valid_complete_prefix(prefix, record_property):
    ns, buffers, bundle = prefix
    assert ns["_check_progress"](bundle, buffers, bundle["context"], dict(counts={}), lambda: False) is bundle
    assert all(float(bundle["states"][s]["optimizer"]["state"][0]["step"]) == s for s in range(1, 26))
    assert bundle["states"][0]["optimizer"]["state"] == {}
    assert bundle["states"][25]["terminal_no_update"] is True and "scaled_factor_gradients" not in bundle["states"][25]
    record_property("actual_tiny_CPU_Adam_updates", 25)
    record_property("manual_native_recurrences_equal", 25)
    record_property("source_or_head_fits", 0)


@pytest.mark.parametrize("mutation", ["counter", "bool_counter", "first", "second", "group", "factor", "gradient", "theta_shape",
                                     "snapshot_container", "missing_step", "frontier_bool", "history", "coupled_J0", "context"])
def test_resealed_midprefix_corruption_rejects(prefix, mutation):
    ns, buffers, original = prefix
    bundle = cpu_state(original)
    state = bundle["states"][7]
    if mutation == "counter": state["optimizer"]["state"][0]["step"] = torch.tensor(8.)
    elif mutation == "bool_counter": state["optimizer"]["state"][0]["step"] = torch.tensor(True)
    elif mutation == "first": state["optimizer"]["state"][0]["exp_avg"][0, 0] += .01
    elif mutation == "second": state["optimizer"]["state"][1]["exp_avg_sq"][0, 0] += .01
    elif mutation == "group": state["optimizer"]["param_groups"][0]["lr"] = .05
    elif mutation == "factor": state["parameters"][0][0, 0] += .01
    elif mutation == "gradient": state["scaled_factor_gradients"][1][0, 0] += .01
    elif mutation == "theta_shape": state["theta"] = state["theta"].T
    elif mutation == "snapshot_container": bundle["states"][7] = None
    elif mutation == "missing_step": del bundle["states"][7]
    elif mutation == "frontier_bool": bundle["frontier"] = True
    elif mutation == "history": bundle["history"][7]["step"] = True
    elif mutation == "context": bundle["context"]["candidate"]["method"] = "AW"
    elif mutation == "coupled_J0":
        for saved in bundle["states"].values(): saved["J0"] += .1
        bundle["states"][0]["macro_ce"] += .1
        bundle["J0"] += .1
        bundle["history"] = ns["_history"](bundle["states"])
    bundle = reseal(ns, bundle)
    with pytest.raises(ValueError): ns["_check_progress"](bundle, buffers, original["context"], dict(counts={}), lambda: False)


def install_fixture(tmp_path, prefix):
    ns, source, original = prefix
    buffers = dict(source, root=tmp_path / "source")
    folder = tmp_path / "AX"
    folder.mkdir()
    bundle = cpu_state(original)
    zero = bundle["states"][0]
    rep = ns["representative"](zero["moments"], None, 3703, "cuda")
    context = dict(bundle["context"], reference={"student_inputs": ns["probe"]._digest(list(rep))})
    bundle["context"] = context
    bundle = reseal(ns, bundle)
    path = buffers["root"] / "original_NODE/condensation_0/checkpoints/step_000000.pt"
    path.parent.mkdir(parents=True)
    torch.save(dict(moments=zero["moments"], theta=zero["theta"]), path)
    (folder/"candidate.json").write_text(json.dumps(context["candidate"]))
    ns["_store"](folder, context, {0: zero}, bundle["J0"])
    ns["_store"](folder, context, bundle["states"], bundle["J0"])
    return ns, buffers, bundle, folder, path


@pytest.mark.parametrize("mutation", [None, "missing25", "orphan", "history_mirror", "endpoint_mirror", "candidate", "alternate_origin_theta"])
def test_endpoint_files_origin_and_readonly_guard(tmp_path, prefix, mutation):
    ns, buffers, bundle, folder, origin_path = install_fixture(tmp_path, prefix)
    if mutation == "missing25": (folder/"step_000025.pt").unlink()
    elif mutation == "orphan": torch.save(bundle["states"][7], folder/"step_000007.pt")
    elif mutation == "history_mirror": (folder/"history.json").write_text("[]")
    elif mutation == "endpoint_mirror":
        mirror = cpu_state(bundle["states"][25]); mirror["macro_ce"] += .1; torch.save(mirror, folder/"step_000025.pt")
    elif mutation == "candidate": (folder/"candidate.json").write_text('{"method":"AW"}')
    elif mutation == "alternate_origin_theta":
        bundle["states"][0]["theta"][0, 0] = .01
        bundle = reseal(ns, bundle)
        torch.save(bundle, folder/"progress.pt")
        torch.save(bundle["states"][0], folder/"step_000000.pt")
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir()}
    if mutation is None:
        saved, _ = ns["_load_progress"](folder, buffers, bundle["context"], dict(counts={}), lambda: False)
        assert saved["frontier"] == 25
    else:
        with pytest.raises(ValueError): ns["_load_progress"](folder, buffers, bundle["context"], dict(counts={}), lambda: False)
    assert before == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir()}
    assert origin_path.is_file()


def test_partial_cache_prepare_never_updates_or_repairs(tmp_path, prefix):
    ns, buffers, bundle, folder, _ = install_fixture(tmp_path, prefix)
    (folder/"step_000025.pt").unlink()
    ns["_folder"] = lambda *a: folder
    with pytest.raises(ValueError, match="Partial/orphan"):
        ns["_prepare"]({}, buffers, bundle["context"], dict(counts={}), lambda: False)
    assert not (folder/"step_000025.pt").exists()


@pytest.mark.parametrize("kind,function", [("citation_macro_fixed25_prepare", "prepare"),
                                         ("citation_macro_fixed25_certify", "certify"),
                                         ("citation_macro_fixed25_validate", "validate")])
def test_three_real_lazy_dispatch_branches(kind, function, monkeypatch):
    expected, calls, stop = object(), [], lambda: False
    fake = SimpleNamespace(**{function: lambda **options: calls.append(options) or expected})
    monkeypatch.setitem(sys.modules, "src.citation_macro_fixed25", fake)
    ns = extract(REPO / "src/research_loop.py", {"dispatch"}, {})
    options = dict(cells=30, spec_path="fixed", spec_sha256="sha", output_path="new")
    if function == "validate": options.update(arm="macro", gate_sha256="gate")
    assert ns["dispatch"](dict(kind=kind, options=options), stop) is expected
    assert calls == [dict(options, stop=stop)] and "stop" not in options


@pytest.mark.parametrize("mutation", [None, "failed", "schema", "partial", "source", "pins", "prefix", "unknown_candidate"])
def test_native_gate_before_student_access(tmp_path, mutation):
    ns = namespace()
    path = tmp_path/"native_certificate25.json"
    asset = tmp_path/"cache"; asset.write_text("immutable")
    spec = dict(source={"git": "fixed"}, numerical_source={}, candidate={"fixed": True}, candidate_id="id",
                scientific_preregistration={"fixed": True}, certificate_outputs={"30": str(path)})
    gate = dict(passed=True, schema=1, assignment_steps=25, cells=30, full25_cache_replay_passed=True,
        actual_FP32_P0_X_Q_uniform_equal_reference=True, prefix_sha256="a"*64, spec_sha256="b"*64,
        files_sha256={str(asset): ns["_sha"](asset)}, **json.loads(json.dumps({k: spec[k] for k in
            ("source", "numerical_source", "candidate", "candidate_id", "scientific_preregistration")})))
    if mutation == "failed": gate["passed"] = False
    elif mutation == "schema": gate["schema"] = True
    elif mutation == "partial": gate["full25_cache_replay_passed"] = False
    elif mutation == "source": gate["source"] = {"git": "changed"}
    elif mutation == "pins": asset.write_text("tampered")
    elif mutation == "prefix": del gate["prefix_sha256"]
    elif mutation == "unknown_candidate": gate["candidate"]["external_mean"] = 0
    def checked(files):
        ns["_require"](all(ns["_sha"](p) == h for p, h in files.items()), "changed cache")
    ns["probe"]._checked_files = checked
    path.write_text(json.dumps(gate))
    if mutation is None:
        assert ns["_gate"](spec, 30, ns["_sha"](path))["passed"] is True
    else:
        with pytest.raises(ValueError): ns["_gate"](spec, 30, ns["_sha"](path))
