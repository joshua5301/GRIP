"""Metadata/fake orchestration only; no actual tensors, fits or cache loads."""
import copy
import csv
import json
import platform
from pathlib import Path
from types import SimpleNamespace

import pytest

from src import cora_gradient_students as students
from src import finite_student_experiment as old_cora


class FakeTensor:
    def __init__(self, shape, dtype, value=None):
        self.shape, self.dtype, self.value = shape, dtype, value


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    original = Path(students.__file__).resolve().parents[1]
    science = json.loads((original / "results/proposals" / students.SCIENCE).read_text())
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "results/proposals").mkdir(parents=True)
    for case in science["cases"]:
        case["source_root"] = str(repo / "original" / case["root"])
        root = Path(case["source_root"])
        root.mkdir(parents=True)
        (root / f"student_recipe_{case['recipe_id']}.json").write_text(json.dumps(case["recipe"]))
    science_path = repo / "results/proposals" / students.SCIENCE
    science_path.write_text(json.dumps(science))
    monkeypatch.setattr(students, "__file__", str(repo / "src/cora_gradient_students.py"))
    monkeypatch.setattr(students, "SCIENCE_SHA", students._sha(science_path))
    monkeypatch.setattr(students, "implementation_provenance", lambda: {"source": "fixed"})
    monkeypatch.setattr(students.ay, "numerical_source", lambda: {"versions": "fixed"})
    monkeypatch.setattr(students.torch, "get_num_threads", lambda: 4)
    selected = {}
    calls = {"fit": 0, "replay": 0, "counts": 0}
    monkeypatch.setattr(students.torch, "load", lambda *a, **k: copy.deepcopy(selected))
    monkeypatch.setattr(students, "_input_digest", lambda *a, **k: "own-input")

    def tensor(value, shape, dtype, message):
        if not isinstance(value, FakeTensor) or value.shape != shape or value.dtype != dtype:
            raise ValueError(message)

    monkeypatch.setattr(students, "_tensor", tensor)
    monkeypatch.setattr(students.torch, "full_like", lambda x, fill: FakeTensor(x.shape, x.dtype, fill))
    monkeypatch.setattr(students.torch, "equal", lambda a, b: (a.shape, a.dtype, a.value) == (b.shape, b.dtype, b.value))

    def build(cells=35, seed=4100, folder_name="shared_P0_validation"):
        case = copy.deepcopy(next(c for c in science["cases"] if c["cells"] == cells))
        recipe_path = Path(case["source_root"]) / f"student_recipe_{case['recipe_id']}.json"
        origin = dict(path=str(recipe_path), sha256=students._sha(recipe_path), recipe_id=case["recipe_id"],
                      contents=copy.deepcopy(case["recipe"]), scope="original recipe separately pinned")
        output_root = repo / "results/research_loop/StageBD"
        base = output_root / f"cora{cells}" / "candidate"
        gate = base.parent / "native_certificate25_v1.json"
        gate.parent.mkdir(parents=True, exist_ok=True)
        gate.write_text("generated passed-gate fixture; not native evidence")
        context = dict(implementation={"source": "fixed"}, numerical_source={"versions": "fixed"},
            assignment_steps=25, case=copy.deepcopy(case), cells=cells, student_recipe_origin=copy.deepcopy(origin),
            scientific_preregistration=dict(path=str(science_path), sha256=students.SCIENCE_SHA),
            spec=dict(output_root=str(output_root), candidate_id="candidate", python_version=platform.python_version(),
                      certificate_outputs={str(cells): str(gate)}))
        buffers = dict(case=case, recipe_origin=origin, counts={}, graph={"fixture": "original graph"},
                       q="teacher Q", h="own cached H", train="train-mask", val="val-mask")
        inputs = (FakeTensor((cells, 1433), students.torch.float32),
                  FakeTensor((cells, 7), students.torch.float32),
                  FakeTensor((cells,), students.torch.float64, 1 / cells))
        folder = base / folder_name
        settings = {k: v for k, v in case["recipe"].items() if k != "input_scale"}
        return SimpleNamespace(folder=folder, inputs=inputs, buffers=buffers, context=context,
            gate=gate, gate_sha=students._sha(gate), seed=seed, settings=settings)

    def write_cache(f):
        paths = students._student_paths(f.folder, f.seed)
        f.folder.mkdir(parents=True, exist_ok=True)
        recipe = dict(version=2, seed=f.seed, settings=f.settings, layers=2, optimizer="Adam; constant lr",
            weighting="normalized supplied mass", selection="first maximum validation accuracy", test_enabled=False,
            test_evaluation="selected weights once", torch_version=students.torch.__version__, input_digest="own-input")
        fingerprint = students._fingerprint(recipe)
        selected.clear()
        selected.update(epoch=10, fingerprint=fingerprint, model_state="opaque fake state; no tensors")
        paths["selected"].write_text("opaque selected fixture; torch.load is replaced")
        paths["metadata"].write_text(json.dumps(dict(fingerprint=fingerprint, recipe=recipe,
                                                     result=dict(epoch=10, val_acc=80.))))
        with paths["epochs"].open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["epoch", "val_acc", "train_ce"])
            writer.writeheader()
            for epoch in range(1, 601):
                writer.writerow(dict(epoch=epoch, val_acc=80. if epoch in (10, 20) else 79., train_ce=.1))
        route_recipe = dict(epoch=10, seed=f.seed, source_fingerprint=fingerprint, settings=f.settings,
            test_enabled=False, input_digest="mock-input-state-hash", selection="same weights at GCN validation-selected epoch")
        paths["routes"].write_text(json.dumps(dict(fingerprint=students._fingerprint(route_recipe), recipe=route_recipe,
            result=dict(seed=f.seed, epoch=10, gcn_val_acc=80., mlp_val_acc=78., gcn_val_ce=.2, mlp_val_ce=.3))))
        return paths

    active = [None]

    def fit(*args, **kwargs):
        calls["fit"] += 1
        assert kwargs["training_adjacency"] is None
        assert kwargs["seed"] == active[0].seed and set(args[5]) == {"train", "val"}
        assert {k: kwargs[k] for k in active[0].settings} == active[0].settings
        assert "input_scale" not in kwargs
        write_cache(active[0])

    def replay(selected_path, graph, h, masks, settings, output_path, **kwargs):
        calls["replay"] += 1
        assert graph is active[0].buffers["graph"] and h == "own cached H" and masks == {"val": "val-mask"}
        assert settings == active[0].settings and kwargs["seed"] == active[0].seed
        return json.loads(Path(output_path).read_text())["result"]

    def counts(checkpoint, buffers, stop):
        calls["counts"] += 1
        assert checkpoint == selected and buffers is active[0].buffers
        return {"gcn": 400, "mlp": 390}, dict(gcn_val_acc=80., mlp_val_acc=78., gcn_val_ce=.2, mlp_val_ce=.3)

    monkeypatch.setattr(students, "fit_gcn_diagnostic", fit)
    monkeypatch.setattr(students, "replay_routes", replay)
    monkeypatch.setattr(students, "_selected_route_counts", counts)

    def evaluate(f):
        active[0] = f
        return students.evaluate_student(f.folder, f.inputs, f.buffers, f.context, f.gate_sha, f.seed, lambda: False)

    return SimpleNamespace(build=build, write_cache=write_cache, evaluate=evaluate, calls=calls,
                           selected=selected, active=active, science_path=science_path)


def test_direct_unchanged_cora_route_helpers():
    assert students._student_paths is old_cora._student_paths
    assert students._selected_route_counts is old_cora._selected_route_counts


@pytest.mark.parametrize("cells", [35, 70, 140])
def test_own_recipe_single_fit_then_shared_cache_reuse(fixture, cells):
    f = fixture.build(cells)
    first = fixture.evaluate(f)
    second = fixture.evaluate(f)
    assert first["actual_student_fits"] == 1 and first["physical_route_outputs"] == 2
    assert second["cached"] and second["actual_student_fits"] == second["physical_route_outputs"] == 0
    assert first["gcn_correct"] == 400 and first["mlp_correct"] == 390 and first["selected_epoch"] == 10
    assert fixture.calls == {"fit": 1, "replay": 3, "counts": 2}
    assert f.buffers["counts"] == dict(final_student_fit_attempted=1, final_student_fit_completed=1)
    saved = json.loads(Path(first["student_completion_path"]).read_text())
    assert saved["header"]["cells"] == cells and saved["header"]["recipe_origin"] == f.buffers["recipe_origin"]
    assert "arm" not in saved["header"] and "step" not in saved["header"]
    assert saved["validation_nodes"] == 500 and set(saved["files_sha256"]) == {"metadata", "epochs", "selected", "routes"}


@pytest.mark.parametrize("seed", [True, 4099, 4103, "4100"])
def test_unknown_or_boolean_seed_rejected_before_fit(fixture, seed):
    f = fixture.build(seed=seed)
    with pytest.raises(ValueError):
        fixture.evaluate(f)
    assert fixture.calls["fit"] == 0


@pytest.mark.parametrize("mutation", ["case", "recipe", "recipe_file", "science", "gate", "folder", "source", "frontier"])
def test_resealed_origin_and_control_mismatches_rejected_before_fit(fixture, monkeypatch, mutation):
    f = fixture.build()
    if mutation == "case":
        f.buffers["case"]["root"] = "otherroot"
    elif mutation == "recipe":
        f.buffers["recipe_origin"]["contents"]["lr"] = .02
    elif mutation == "recipe_file":
        Path(f.buffers["recipe_origin"]["path"]).write_text("changed recipe")
    elif mutation == "science":
        fixture.science_path.write_text("changed science")
    elif mutation == "gate":
        f.gate.write_text("changed accepted gate")
    elif mutation == "folder":
        f.folder = f.folder.parent / "other_student_namespace"
    elif mutation == "source":
        monkeypatch.setattr(students, "implementation_provenance", lambda: {})
    elif mutation == "frontier":
        f.context["assignment_steps"] = 1
    with pytest.raises(ValueError):
        fixture.evaluate(f)
    assert fixture.calls["fit"] == 0 and not f.folder.exists()


@pytest.mark.parametrize("mutation", ["x", "q", "weights", "nonuniform"])
def test_exact_native_student_shapes_and_uniform_weight_guard(fixture, mutation):
    f = fixture.build()
    i = {"x": 0, "q": 1, "weights": 2, "nonuniform": 2}[mutation]
    if mutation == "nonuniform":
        f.inputs[i].value = .9
    else:
        f.inputs[i].shape = (1,)
    with pytest.raises(ValueError):
        fixture.evaluate(f)
    assert fixture.calls["fit"] == 0


@pytest.mark.parametrize("mutation", ["partial", "orphan", "header", "bytes"])
def test_partial_or_changed_complete_cache_never_retrains(fixture, mutation):
    f = fixture.build()
    result = fixture.evaluate(f)
    p = students._student_paths(f.folder, f.seed)
    if mutation == "partial":
        p["certificate"].unlink()
    elif mutation == "orphan":
        (f.folder / f"seed_{f.seed}_orphan.pt").write_text("unqualified")
    elif mutation == "header":
        v = json.loads(p["certificate"].read_text());v["header"]["cells"] = 70
        p["certificate"].write_text(json.dumps(v))
    else:
        p["selected"].write_text("changed weights")
    assert result["actual_student_fits"] == 1
    with pytest.raises(ValueError):
        fixture.evaluate(f)
    assert fixture.calls["fit"] == 1


@pytest.mark.parametrize("mutation", ["short", "order", "nonfinite", "epoch", "fingerprint", "route_epoch", "route_scalar", "route_ce"])
def test_history_selected_and_route_proof_failures_preserved(fixture, mutation):
    f = fixture.build();fixture.active[0] = f
    paths = fixture.write_cache(f)
    header = students._student_header(f.folder, f.buffers, f.context, f.gate_sha, "own-input", f.seed, f.settings)
    if mutation in ("short", "order", "nonfinite"):
        lines = paths["epochs"].read_text().splitlines()
        if mutation == "short": lines.pop()
        elif mutation == "order": lines[2] = lines[1]
        else: lines[2] = lines[2].replace("79.0", "nan")
        paths["epochs"].write_text("\n".join(lines)+"\n")
    elif mutation == "epoch": fixture.selected["epoch"] = 20
    elif mutation == "fingerprint": fixture.selected["fingerprint"] = "other-inputs"
    else:
        v = json.loads(paths["routes"].read_text())
        if mutation == "route_epoch":
            v["recipe"]["epoch"] = 20;v["fingerprint"] = students._fingerprint(v["recipe"])
        elif mutation == "route_scalar": v["result"]["gcn_val_acc"] = 80.2
        else: v["result"]["mlp_val_ce"] = 1.
        paths["routes"].write_text(json.dumps(v))
    with pytest.raises(ValueError):
        students._student_certificate(paths, header, f.buffers, lambda: False)
    assert fixture.calls["fit"] == 0 and not paths["certificate"].exists()


@pytest.mark.parametrize("point", ["fit", "replay", "source_after_fit"])
def test_attempt_completion_counters_and_failed_files_are_not_qualified(fixture, monkeypatch, point):
    f = fixture.build()
    def fail(*a, **k): raise RuntimeError("mock failure")
    if point == "fit": monkeypatch.setattr(students, "fit_gcn_diagnostic", fail)
    elif point == "replay": monkeypatch.setattr(students, "replay_routes", fail)
    else:
        calls = [0]
        def source(context):
            calls[0] += 1
            if calls[0] > 1: raise RuntimeError("changed source after fit")
        monkeypatch.setattr(students, "_source_unchanged", source)
    with pytest.raises(RuntimeError): fixture.evaluate(f)
    assert f.buffers["counts"]["final_student_fit_attempted"] == 1
    assert f.buffers["counts"].get("final_student_fit_completed", 0) == (0 if point == "fit" else 1)
    assert not students._student_paths(f.folder, f.seed)["certificate"].exists()


def test_stop_before_any_read_or_fit(fixture):
    f = fixture.build()
    with pytest.raises(InterruptedError):
        students.evaluate_student(f.folder, f.inputs, f.buffers, f.context, f.gate_sha, f.seed, lambda: True)
    assert fixture.calls["fit"] == 0 and not f.folder.exists()
