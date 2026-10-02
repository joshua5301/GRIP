"""Tiny CPU AST guards/native Adam only; no production data or finite engine run."""
import ast
import copy
import csv
import hashlib
import json
import math
import os
import sys
import uuid
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from src.io import _fingerprint, array_digest, cpu_state

_COUNTS = dict(native_toy_Adam_steps=0, mock_objective_evaluations=0)


@pytest.fixture(scope="session", autouse=True)
def counts(record_testsuite_property):
    yield
    for key, value in _COUNTS.items():
        record_testsuite_property(key, value)


@pytest.fixture
def namespace(monkeypatch):
    original_step = torch.optim.Adam.step
    def counted_step(*args, **kwargs):
        _COUNTS["native_toy_Adam_steps"] += 1
        return original_step(*args, **kwargs)
    monkeypatch.setattr(torch.optim.Adam, "step", counted_step)
    root = Path(__file__).resolve().parents[1]
    candidate_path = root / "finite_student_experiment.py"
    if not candidate_path.is_file():
        candidate_path = root / "src/finite_student_experiment.py"
    probe_path = Path(__file__).resolve().parents[1] / "finite_student_probe.py"
    if not probe_path.is_file():
        repo = Path(__file__).resolve()
        while not (repo / "src/finite_student_probe.py").is_file():
            repo = repo.parent
        probe_path = repo / "src/finite_student_probe.py"
    names = {"_require", "_stop", "_sha", "_digest", "_seal", "_tensor", "_scalar", "_attach", "_atomic", "_history"}
    parsed = ast.parse(probe_path.read_text())
    old = dict(torch=torch, os=os, uuid=uuid, math=math, json=json, hashlib=hashlib, Path=Path,
               array_digest=array_digest, cpu_state=cpu_state)
    exec(compile(ast.Module(body=[x for x in parsed.body if isinstance(x, ast.FunctionDef) and x.name in names],
                            type_ignores=[]), str(probe_path), "exec"), old)
    fixed = next(x for x in parsed.body if isinstance(x, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "_FIXED" for t in x.targets))
    exec(compile(ast.Module(body=[fixed], type_ignores=[]), str(probe_path), "exec"), old)
    probe = SimpleNamespace(**{x: old[x] for x in names}, _FIXED=old["_FIXED"], _ROUTES=("sgc_mlp", "gcn"),
                            _SOURCE_FIELD="finite_student_source_digest", cpu_state=cpu_state,
                            _public_options=lambda *a: None, _native=lambda *a: {"mock_only": True},
                            _checked_files=lambda pins: None, REFERENCE_ID="c24c1ffc2b76")
    new = dict(old, probe=probe, csv=csv, io=__import__("io"), _fingerprint=_fingerprint,
               numerical_source=lambda: {"toy_only": "immutable"})
    parsed = ast.parse(candidate_path.read_text())
    excluded = {"numerical_source", "_context", "_source_unchanged", "_evaluate", "_prepare", "_selected_route_counts"}
    nodes = [x for x in parsed.body if isinstance(x, ast.Assign) or isinstance(x, ast.FunctionDef) and x.name not in excluded]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(candidate_path), "exec"), new)
    new["_source_unchanged"] = lambda context: None
    new["_evaluate"] = toy_evaluate
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(total_memory=123456))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 0)
    return new


def toy_evaluate(buffers, candidate, context, parameters, step, scale, stop):
    _COUNTS["mock_objective_evaluations"] += 1
    if stop():
        raise InterruptedError("toy stop")
    parameters = [p.detach().clone() for p in parameters]
    loss = float(((parameters[0]-.3).square() + .7*(parameters[1]+.2).square()).mean()) + .7
    if scale is None:
        scale = loss
    gradients = [2*(parameters[0]-.3)/(parameters[0].numel()*scale),
                 1.4*(parameters[1]+.2)/(parameters[1].numel()*scale)]
    snapshot = dict(schema=3, context=context, step=step, parameters=parameters, teacher_ce=loss, scale=scale,
                    finite_model_steps=5, scaled_factor_gradients=gradients,
                    min_mass=.2, max_mass=.8, effective_cells=1.4, selected_route=candidate["outer_route"],
                    toy_only=True)
    return snapshot, scale


def initial():
    return [torch.zeros(2, 3), torch.tensor([[.2, -.1, .3], [-.4, .7, .1]])]


def candidate(namespace):
    return namespace["canonical_candidate"](dict(namespace["_FIXED"], outer_route="sgc_mlp"))


def build_bundle(namespace, frontier=25):
    context = {"mock_original_source": True}
    parameters = [p.requires_grad_() for p in initial()]
    optimizer = torch.optim.Adam(parameters, lr=.05, eps=1e-12, foreach=False, fused=False)
    snapshots, history, scale = {}, [], None
    for step in range(frontier+1):
        if step:
            for p, gradient in zip(parameters, snapshots[step-1]["scaled_factor_gradients"], strict=True):
                p.grad = gradient.clone()
            optimizer.step()
        snapshot, scale = toy_evaluate({}, candidate(namespace), context, parameters, step, scale, lambda: False)
        snapshot["optimizer"] = cpu_state(optimizer.state_dict())
        snapshots[step] = snapshot
        history.append(namespace["_history"](snapshot))
    return namespace["_attach"](dict(schema=3, context=context, step=frontier, parameters=parameters,
                 optimizer=optimizer.state_dict(), snapshots=snapshots, scale=scale, history=history))


def reseal(namespace, bundle):
    bundle.pop("content_sha256", None)
    return namespace["_attach"](bundle)


@pytest.mark.parametrize("seed", [0, 19, 45])
def test_native_single_tensor_Adam_all25_matches_exact_manual_recurrence(namespace, seed):
    generator = torch.Generator().manual_seed(seed)
    native = [torch.randn(2, 3, generator=generator, requires_grad=True), torch.randn(3, generator=generator, requires_grad=True)]
    manual = [p.detach().clone() for p in native]
    first, second = [torch.zeros_like(p) for p in native], [torch.zeros_like(p) for p in native]
    optimizer = torch.optim.Adam(native, lr=.05, eps=1e-12, foreach=False, fused=False)
    for step in range(1, 26):
        gradients = [torch.randn(p.shape, generator=generator)*(.00001 if step % 4 == 0 else 1) for p in native]
        if step % 5 == 0:
            gradients = [torch.zeros_like(p) for p in native]
        for p, gradient in zip(native, gradients, strict=True):
            p.grad = gradient
        optimizer.step()
        manual, first, second = namespace["_adam_step"](manual, first, second, gradients, step)
        assert all(torch.equal(p, q) for p, q in zip(native, manual, strict=True))
        namespace["_optimizer"](cpu_state(optimizer.state_dict()), manual, first, second, step)
        assert all(float(s["step"]) == step and s["step"].device.type == "cpu" and s["step"].dtype == torch.float32
                   for s in optimizer.state.values())


@pytest.mark.parametrize("frontier", [0, 1, 7, 25])
def test_complete_prefix_replays_every_snapshot_and_actual_optimizer(namespace, frontier):
    bundle = build_bundle(namespace, frontier)
    calls = []
    def evaluate(*args):
        calls.append(args[4])
        return toy_evaluate(*args)
    namespace["_evaluate"] = evaluate
    assert namespace["_validate_bundle"](bundle, {"initial": initial()}, candidate(namespace), bundle["context"], lambda: False) == frontier+1
    assert calls == list(range(frontier+1))


@pytest.mark.parametrize("kind", ["counter", "boolcounter", "counterdtype", "counterNaN", "first", "second", "lr", "paramID", "snapshotNone", "current", "history", "J0coupled", "snapshotgradient", "frontierbool"])
def test_resealed_midpoint_or_coupled_corruption_rejected(namespace, kind):
    bundle = copy.deepcopy(build_bundle(namespace))
    slot = bundle["snapshots"][7]["optimizer"]["state"][0]
    if kind == "counter":
        slot["step"] += 1
    elif kind == "boolcounter":
        slot["step"] = torch.tensor(True)
    elif kind == "counterdtype":
        slot["step"] = slot["step"].double()
    elif kind == "counterNaN":
        slot["step"] = torch.tensor(float("nan"))
    elif kind == "first":
        slot["exp_avg"][0, 0] += .1
    elif kind == "second":
        slot["exp_avg_sq"][0, 0] += .1
    elif kind == "lr":
        bundle["snapshots"][7]["optimizer"]["param_groups"][0]["lr"] = .1
    elif kind == "paramID":
        bundle["snapshots"][7]["optimizer"]["param_groups"][0]["params"][0] = False
    elif kind == "snapshotNone":
        bundle["snapshots"][7] = None
    elif kind == "current":
        bundle["parameters"][0][0, 0] += .1
    elif kind == "history":
        bundle["history"][7]["teacher_ce"] += .1
    elif kind == "J0coupled":
        bundle["scale"] += .1
        bundle["snapshots"][0]["teacher_ce"] += .1
        for step in range(26):
            bundle["snapshots"][step]["scale"] = bundle["scale"]
            bundle["history"][step] = namespace["_history"](bundle["snapshots"][step])
    elif kind == "snapshotgradient":
        bundle["snapshots"][7]["scaled_factor_gradients"][0][0, 0] += .1
        bundle["history"][7] = namespace["_history"](bundle["snapshots"][7])
    else:
        bundle["step"] = True
    bundle = reseal(namespace, bundle)
    before = namespace["_seal"](bundle)
    with pytest.raises(ValueError):
        namespace["_validate_bundle"](bundle, {"initial": initial()}, candidate(namespace), bundle["context"], lambda: False)
    assert namespace["_seal"](bundle) == before


def install_toy_prepare(namespace, root):
    root.mkdir()
    buffers = {"initial": initial(), "root": root, "pins": {}}
    context = {"mock_original_source": True}
    namespace["_prepare"] = lambda *args: (args[3], buffers, context)
    return buffers


def run_prepare(namespace):
    return namespace["prepare_full25"]("cora", .026, "unused_mock", candidate(namespace))


@pytest.mark.parametrize("when", ["before_snapshot_evaluation", "after_snapshot_evaluation", "after_resume_commit"])
def test_real_CPU_toy_25_stop_resume_and_mirror_prefix_recovery(namespace, tmp_path, when):
    original_evaluate, original_atomic = namespace["_evaluate"], namespace["_atomic"]
    direct_buffers = install_toy_prepare(namespace, tmp_path/"direct")
    direct = run_prepare(namespace)
    direct_folder = direct_buffers["root"]/direct["candidate_id"]/"condensation_0"
    direct_bundle = torch.load(direct_folder/"resume.pt", weights_only=False)
    buffers = install_toy_prepare(namespace, tmp_path/"resumed")
    if when in ("before_snapshot_evaluation", "after_snapshot_evaluation"):
        def interrupt(*args):
            if args[4] == 7:
                if when == "after_snapshot_evaluation":
                    original_evaluate(*args)
                raise InterruptedError("controlled toy interruption")
            return original_evaluate(*args)
        namespace["_evaluate"] = interrupt
    else:
        def interrupt_atomic(path, value, binary):
            original_atomic(path, value, binary)
            if path.name == "resume.pt" and value["step"] == 7:
                raise InterruptedError("committed before mirror completion")
        namespace["_atomic"] = interrupt_atomic
    with pytest.raises(InterruptedError):
        run_prepare(namespace)
    folder = buffers["root"]/direct["candidate_id"]/"condensation_0"
    interrupted = torch.load(folder/"resume.pt", weights_only=False)
    assert interrupted["step"] == (7 if when == "after_resume_commit" else 6)
    namespace["_evaluate"], namespace["_atomic"] = original_evaluate, original_atomic
    resumed = run_prepare(namespace)
    final = torch.load(folder/"resume.pt", weights_only=False)
    assert namespace["_seal"](final) == namespace["_seal"](direct_bundle)
    assert resumed["P_updates"] == 25-interrupted["step"]
    assert len(list((folder/"checkpoints").glob("*.pt"))) == 26
    assert len(list(csv.DictReader((folder/"optimization.csv").open()))) == 26
    again = run_prepare(namespace)
    assert again["cached"] is True and again["P_updates"] == 0
    assert again["counts"]["diagnostic_cache_replay_unrolls"] == 26


@pytest.mark.parametrize("kind", ["alias", "orphan", "badCSV", "extraCSV"])
def test_present_corrupt_mirrors_are_preserved(namespace, tmp_path, kind):
    bundle = build_bundle(namespace, 7)
    folder = tmp_path/"mirrors"
    folder.mkdir()
    namespace["_mirrors"](folder, bundle, export=True)
    if kind == "alias":
        torch.save(bundle["snapshots"][0], folder/"checkpoints/step_wrong.pt")
    elif kind == "orphan":
        value = copy.deepcopy(bundle["snapshots"][0]); value["step"] = 99
        torch.save(value, folder/"checkpoints/step_000099.pt")
    else:
        text = (folder/"optimization.csv").read_text()
        if kind == "badCSV":
            text = text.replace("teacher_ce", "unknown_header")
        else:
            text += text.splitlines()[1]+"\n"
        (folder/"optimization.csv").write_text(text)
    hashes = {p: namespace["_sha"](p) for p in folder.rglob('*') if p.is_file()}
    with pytest.raises(ValueError):
        namespace["_mirrors"](folder, bundle, export=True)
    assert hashes == {p: namespace["_sha"](p) for p in folder.rglob('*') if p.is_file()}


@pytest.mark.parametrize("change", ["schema", "horizon", "missing", "external", "source", "route"])
def test_unknown_or_changed_fixed_control_rejects_without_data(namespace, change):
    value = dict(namespace["_FIXED"], outer_route="sgc_mlp")
    if change == "schema":
        value["finite_student_schema"] = 2
    elif change == "horizon":
        value["finite_assignment_steps"] = 1
    elif change == "missing":
        del value["finite_assignment_steps"]
    elif change == "external":
        value["source_Q"] = "external"
    elif change == "source":
        value["finite_student_source_digest"] = "stale"
    else:
        value["outer_route"] = "raw_X_MLP"
    with pytest.raises(ValueError):
        namespace["canonical_candidate"](value)


@pytest.mark.parametrize("changed", ["missing", "passed", "schemaBool", "horizon", "source", "replay", "P0", "candidate", "science", "filehash"])
def test_gate_fails_before_prepare_or_student_call(namespace, monkeypatch, tmp_path, changed):
    import src.research_loop as worker
    monkeypatch.setattr(worker, "implementation_provenance", lambda: {"toy_source": True})
    science_path = Path(__file__).resolve()
    while not (science_path/"results/proposals/finite_student_Cora70_fixed25_validation_scientific_stageAQ_v1.json").is_file():
        science_path = science_path.parent
    science_path /= "results/proposals/finite_student_Cora70_fixed25_validation_scientific_stageAQ_v1.json"
    asset = tmp_path/"asset"; asset.write_text("immutable")
    gate = dict(passed=True, finite_student_schema=3, finite_assignment_steps=25, source={"toy_source": True},
                full25_cache_replay_passed=True, actual_FP32_P0_X_Q_uniform_equal_reference=True,
                scientific_preregistration={"path": str(science_path), "sha256": namespace["SCIENCE_SHA"]},
                files_sha256={str(asset): namespace["_sha"](asset)},
                candidate_ids={route: _fingerprint(namespace["canonical_candidate"](dict(namespace["_FIXED"], outer_route=route))) for route in ("sgc_mlp", "gcn")})
    if changed == "passed":gate["passed"] = False
    elif changed == "schemaBool":gate["finite_student_schema"] = True
    elif changed == "horizon":gate["finite_assignment_steps"] = 1
    elif changed == "source":gate["source"] = {}
    elif changed == "replay":gate["full25_cache_replay_passed"] = False
    elif changed == "P0":gate["actual_FP32_P0_X_Q_uniform_equal_reference"] = False
    elif changed == "candidate":gate["candidate_ids"]["gcn"] = "other"
    elif changed == "science":gate["scientific_preregistration"]["sha256"] = "bad"
    elif changed == "filehash":gate["files_sha256"][str(asset)] = "bad"
    path = tmp_path/"gate.json"; path.write_text(json.dumps(gate))
    checksum = namespace["_sha"](path)
    if changed == "missing":path.unlink()
    def checked_files(pins):
        if any(not Path(p).is_file() or namespace["_sha"](Path(p)) != value for p, value in pins.items()):
            raise ValueError("changed toy asset")
    namespace["probe"]._checked_files = checked_files
    namespace["_prepare"] = lambda *args: pytest.fail("No source/data prepare allowed")
    with pytest.raises(ValueError):
        namespace["evaluate_cached"]("cora", .026, "mock", "node_reference", str(path), checksum)


def test_shared_P0_header_is_route_independent_but_input_bound(namespace):
    context = dict(implementation={"source": 1}, numerical_source={"numerical": 2})
    first = namespace["_student_header"](context, "gate", "exact_P0", 3600)
    second = namespace["_student_header"](context, "gate", "exact_P0", 3600)
    assert first == second
    assert namespace["_student_header"](context, "gate", "changed_X", 3600) != first
    assert namespace["_student_header"](context, "gate", "exact_P0", 3601) != first


def student_cache(namespace, folder):
    folder.mkdir()
    paths = namespace["_student_paths"](folder, 3600)
    header = namespace["_student_header"]({"implementation": {}, "numerical_source": {}}, "gate", "input", 3600)
    recipe = dict(version=2, seed=3600, settings=namespace["_SETTINGS"], layers=2, optimizer="Adam; constant lr",
                  weighting="normalized supplied mass", selection="first maximum validation accuracy", test_enabled=False,
                  test_evaluation="selected weights once", torch_version=torch.__version__, input_digest="input")
    fingerprint = _fingerprint(recipe)
    metadata = dict(recipe=recipe, fingerprint=fingerprint, result=dict(epoch=2, val_acc=80.))
    selected = dict(epoch=2, fingerprint=fingerprint, model_state={"mock_only": True})
    route_recipe = dict(source_fingerprint=fingerprint, epoch=2, seed=3600, settings=namespace["_SETTINGS"], test_enabled=False)
    routes = dict(recipe=route_recipe, fingerprint=_fingerprint(route_recipe),
                  result=dict(gcn_val_acc=80., gcn_val_ce=.7, mlp_val_acc=78., mlp_val_ce=.8))
    paths["metadata"].write_text(json.dumps(metadata))
    paths["routes"].write_text(json.dumps(routes))
    torch.save(selected, paths["selected"])
    with paths["epochs"].open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['epoch', 'val_acc', 'val_ce'])
        writer.writeheader()
        writer.writerows(dict(epoch=epoch, val_acc=80. if epoch == 2 else 79., val_ce=.7) for epoch in range(1, 601))
    namespace["replay_routes"] = lambda *args, **kwargs: routes["result"]
    namespace["_selected_route_counts"] = lambda *args: ({"gcn": 400, "mlp": 390}, routes["result"])
    buffers = {"graph": {}, "h": None, "val": None}
    return paths, header, buffers, metadata, selected, routes


def test_valid_student_certificate_is_actual_state_and_complete_recipe_bound(namespace, tmp_path):
    paths, header, buffers, *_ = student_cache(namespace, tmp_path/'student')
    result = namespace["_student_certificate"](paths, header, buffers, lambda: False)
    assert result["selected_epoch"] == 2 and result["route_correct_counts"] == {"gcn": 400, "mlp": 390}


@pytest.mark.parametrize("field,value", [('optimizer', 'SGD'), ('weighting', 'mass'), ('layers', 3),
                                         ('selection', 'last'), ('test_evaluation', 'all'), ('torch_version', 'other')])
def test_consistent_student_recipe_refingerprinting_cannot_change_active_policy(namespace, tmp_path, field, value):
    paths, header, buffers, metadata, selected, routes = student_cache(namespace, tmp_path/'student')
    metadata['recipe'][field] = value
    fingerprint = _fingerprint(metadata['recipe'])
    metadata['fingerprint'] = selected['fingerprint'] = routes['recipe']['source_fingerprint'] = fingerprint
    routes['fingerprint'] = _fingerprint(routes['recipe'])
    paths['metadata'].write_text(json.dumps(metadata)); paths['routes'].write_text(json.dumps(routes)); torch.save(selected, paths['selected'])
    with pytest.raises(ValueError, match='full recipe'):
        namespace['_student_certificate'](paths, header, buffers, lambda: False)


def test_resealed_CSV_best_and_result_must_match_actual_selected_GCN_accuracy(namespace, tmp_path):
    paths, header, buffers, metadata, _, _ = student_cache(namespace, tmp_path/'student')
    metadata['result']['val_acc'] = 81.
    paths['metadata'].write_text(json.dumps(metadata))
    text = paths['epochs'].read_text().replace('2,80.0,0.7', '2,81.0,0.7')
    paths['epochs'].write_text(text)
    with pytest.raises(ValueError, match='Actual selected-state'):
        namespace['_student_certificate'](paths, header, buffers, lambda: False)


@pytest.mark.parametrize("kind,entrypoint", [
    ("citation_finite_student_full25", "prepare_full25"),
    ("citation_finite_student_validation", "evaluate_cached"),
])
def test_actual_worker_dispatch_preserves_options_stop_and_return(monkeypatch, kind, entrypoint):
    root = Path(__file__).resolve().parents[1]
    worker = root / "research_loop.py"
    if not worker.is_file():
        worker = root / "src/research_loop.py"
    parsed = ast.parse(worker.read_text())
    dispatch = next(node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name == "dispatch")
    scope = {}
    exec(compile(ast.Module(body=[dispatch], type_ignores=[]), str(worker), "exec"), scope)
    fake_module = ModuleType("src.finite_student_experiment")
    calls = []
    expected_result = {"mock_only": True, "kind": kind, "token": object()}
    def callback(**kwargs):
        calls.append(kwargs)
        return expected_result
    def forbidden(**kwargs):
        raise AssertionError("Wrong finite experiment entrypoint")
    fake_module.prepare_full25 = callback if entrypoint == "prepare_full25" else forbidden
    fake_module.evaluate_cached = callback if entrypoint == "evaluate_cached" else forbidden
    monkeypatch.setitem(sys.modules, "src.finite_student_experiment", fake_module)
    stop = lambda: False
    candidate = {"fixture_only": True}
    options = {"dataset": "cora", "ratio": .026, "candidate": candidate, "output_dir": "never-created"}
    job = {"kind": kind, "options": options}
    before = copy.deepcopy(job)
    assert scope["dispatch"](job, stop) is expected_result
    assert len(calls) == 1
    assert calls[0] == dict(options, stop=stop)
    assert calls[0]["candidate"] is candidate and calls[0]["stop"] is stop
    assert job == before
