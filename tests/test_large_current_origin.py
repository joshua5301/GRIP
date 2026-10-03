"""New capture/replay contracts only: stdlib AST and mocked interfaces; no imports/data."""
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src/large_source_preflight.py").is_file())
DRAFT = (ROOT / "src/large_current_origin.py" if Path(__file__).resolve().parent == ROOT / "tests" else
         ROOT / "results/implementation_drafts/large_current_origin_v1/proposed/src/large_current_origin.py")
WORKER = DRAFT.with_name("research_loop.py")
TREE = ast.parse(DRAFT.read_text())
BT_TREE = ast.parse((ROOT / "src/large_source_preflight.py").read_text())
PROTECTED_WORKER_AST_SHA = 'a0f5a2b7a0cafedb590dd8daeb77f8b876cbd9654dc9735e31312d7c903d1dab'


def function(tree, name):
    return copy.deepcopy(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name))


def load(*names, **extra):
    namespace = dict(Path=Path, json=json, math=math, _require=lambda c, m: c or (_ for _ in ()).throw(ValueError(m)),
                     _sha=lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest(),
                     _seal=lambda v: hashlib.sha256(json.dumps(v, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest())
    namespace.update(extra)
    module = ast.Module(body=[function(TREE, n) for n in names], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(DRAFT), "exec"), namespace)
    return namespace


def test_exact_thirteen_spec_and_public_api():
    fields = next(ast.literal_eval(n.value) for n in TREE.body if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == "FIELDS" for t in n.targets))
    assert fields == {"schema", "fixed", "source", "numerical_source", "python_version", "files_sha256",
                      "artifacts_sha256", "scientific_preregistration", "family_protocol", "candidate_protocol",
                      "output_root", "operation_outputs", "origin_path"}
    assert [a.arg for a in function(TREE, "run").args.args] == [
        "operation", "spec_path", "spec_sha256", "output_path", "capture_evidence_sha256", "stop"]


def test_BT_numerical_body_preserved_with_only_owned_persistence_branch():
    old, new = function(BT_TREE, "_execute"), function(TREE, "_execute")
    # Reconstruct exactly the old observed RMS call, never normalize numerical expressions.
    old_rms = next(n for n in old.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Tuple)
                   and ast.unparse(n.targets[0]) == "(z, transform)")
    kept = []
    for node in new.body:
        text = ast.unparse(node)
        if text == "saved_h = h" or text.startswith("arrays = dict(") or text == "_validate_arrays(libs, arrays)":
            continue
        if text.startswith("evidence['_origin_arrays'] =") or text.startswith("evidence['origin_buffers_before'] ="):
            continue
        if isinstance(node, ast.If) and ast.unparse(node.test) == "operation == 'capture'":
            kept.append(copy.deepcopy(old_rms))
            continue
        if text.startswith("material = _call(evidence, 'material_constructor_calls'"):
            kept.append(ast.parse("material = moments.make_material(z, q)").body[0])
            continue
        if isinstance(node, ast.Return):
            node.value = ast.Name(id="retained", ctx=ast.Load())
        kept.append(node)
    new.args, new.body = old.args, kept
    assert ast.dump(ast.fix_missing_locations(new), include_attributes=False) == ast.dump(old, include_attributes=False)


def test_replay_has_no_RMS_refit_or_forbidden_work():
    branch = next(n for n in function(TREE, "_execute").body if isinstance(n, ast.If)
                  and ast.unparse(n.test) == "operation == 'capture'")
    assert "fit_transform" in ast.unparse(ast.Module(body=branch.body, type_ignores=[]))
    replay = ast.unparse(ast.Module(body=branch.orelse, type_ignores=[]))
    assert "fit_transform" not in replay and "_persisted_transform" in replay and "persisted_transform_apply_calls" in replay
    forbidden = {"optimize_ce_assignment", "backward", "source_gradient_targets", "fit_gcn_diagnostic",
                 "solve_inner", "solve_head_system", "get_shared_h", "get_shared_map", "KMeans"}
    calls = {n.func.attr for n in ast.walk(TREE) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    assert not calls & forbidden


class Tensor:
    def __init__(self, shape, dtype="FP64", value=1.0):
        self.shape, self.dtype, self.value = shape, dtype, value
        self.layout, self.device = "strided", SimpleNamespace(type="cpu")

    @property
    def ndim(self):
        return len(self.shape)

    def __len__(self):
        return self.shape[0]

    def __float__(self):
        return float(self.value)


def transform_namespace():
    def tensor(t, v, shape, dtype, name):
        if not isinstance(v, Tensor) or v.shape != shape or v.dtype != dtype or not math.isfinite(v.value):
            raise ValueError(name)
        return v
    ns = load("_validate_transform_state", TRANSFORM_KEYS={"center", "matrix", "output_center", "scale", "kind", "eps"}, _tensor=tensor)
    return ns, {"torch": SimpleNamespace(float64="FP64")}


def state():
    return dict(center=Tensor((128,)), matrix=None, output_center=Tensor((128,)), scale=Tensor(()), kind="rms", eps=1e-12)


@pytest.mark.parametrize("mutation", ["extra", "integer_eps", "nonpositive_scale", "shape", "nonfinite"])
def test_persisted_transform_rejects_new_malformed_states(mutation):
    ns, libs = transform_namespace()
    candidate = state()
    if mutation == "extra":
        candidate["recovered_historical"] = True
    elif mutation == "integer_eps":
        candidate["eps"] = 0
    elif mutation == "nonpositive_scale":
        candidate["scale"] = Tensor((), value=0)
    elif mutation == "shape":
        candidate["center"] = Tensor((127,))
    else:
        candidate["scale"] = Tensor((), value=float("nan"))
    with pytest.raises(ValueError):
        ns["_validate_transform_state"](libs, candidate)


def binding_fixture(tmp_path):
    cache = tmp_path / "origin_cache.pt"
    cache.write_bytes(b"opaque mocked origin")
    capture = tmp_path / "capture.json"
    spec = dict(operation_outputs={"capture": str(capture)}, origin_path=str(cache), source={"full": 85},
                numerical_source={"exact": "versions"}, scientific_preregistration={"path": "science", "sha256": "a" * 64})
    receipt = dict(passed=True, operation="capture", current_origin_capture_passed=True,
                   source_assets_spec_science_unchanged=True, source_buffers_unchanged=True, origin_buffers_unchanged=True,
                   spec_path="spec", spec_sha256="b" * 64, source=spec["source"], numerical_source=spec["numerical_source"],
                   scientific_preregistration=spec["scientific_preregistration"], origin_cache_path=str(cache),
                   origin_cache_sha256=hashlib.sha256(cache.read_bytes()).hexdigest(), origin_content_sha256="c" * 64,
                   origin_context_sha256="d" * 64)
    return spec, receipt, capture, cache


@pytest.mark.parametrize("mutation", ["none", "failed_capture", "other_spec", "changed_cache"])
def test_dynamic_replay_binding_requires_exact_passed_own_capture(tmp_path, mutation):
    spec, receipt, capture, cache = binding_fixture(tmp_path)
    if mutation == "failed_capture":
        receipt["passed"] = False
    elif mutation == "other_spec":
        receipt["spec_sha256"] = "e" * 64
    elif mutation == "changed_cache":
        cache.write_bytes(b"changed")
    capture.write_text(json.dumps(receipt))
    def checked(pins):
        for p, h in pins.items():
            if not Path(p).is_file() or hashlib.sha256(Path(p).read_bytes()).hexdigest() != h:
                raise ValueError("changed pin")
    ns = load("_capture_binding", _checked=checked)
    invoke = lambda: ns["_capture_binding"](spec, "spec", "b" * 64, hashlib.sha256(capture.read_bytes()).hexdigest())
    if mutation == "none":
        assert invoke() == receipt
    else:
        with pytest.raises(ValueError):
            invoke()


def test_exact_content_and_context_binding_without_tensor_replay():
    # Exercise actual typed packet guards with shape/dtype-only objects, no numeric tensor.
    class CSR(Tensor):
        def __init__(self):
            super().__init__((169343, 169343), "FP32")
            self.layout = "CSR"
            self.crow, self.col, self.entries = Tensor((169344,), "INT64"), Tensor((3,), "INT64"), Tensor((3,), "FP32")

        def crow_indices(self):
            return self.crow

        def col_indices(self):
            return self.col

        def values(self):
            return self.entries
    t = SimpleNamespace(float32="FP32", float64="FP64", int32="INT32", int64="INT64", bool="BOOL",
                        strided="strided", sparse_csr="CSR", is_tensor=lambda v: isinstance(v, Tensor),
                        isfinite=lambda v: SimpleNamespace(all=lambda: math.isfinite(v.value)))
    layouts = {"X": ((169343, 128), "FP32"), "graph_y_identity": ((169343,), "INT64"),
               "train_mask": ((169343,), "BOOL"), "validation_mask": ((169343,), "BOOL"),
               "saved_H": ((169343, 128), "FP32"), "H_double": ((169343, 128), "FP64"),
               "Q": ((169343, 40), "FP64"), "hard": ((169343,), "INT64"), "z": ((169343, 128), "FP64"),
               "U0": ((169343, 16), "FP32"), "V0": ((90, 16), "FP32"), "M0": ((90, 169), "FP64"),
               "theta0": ((40, 129), "FP64")}
    arrays = {k: Tensor(shape, dtype) for k, (shape, dtype) in layouts.items()}
    arrays.update(original_CSR=CSR(), transform=state(), readout=[Tensor((90, 128), "FP32"),
                  Tensor((90, 40), "FP32"), Tensor((90,), "FP64")])
    schema, _ = transform_namespace()
    validate = load("_validate_arrays", ARRAY_KEYS=set(arrays), _tensor=schema["_tensor"],
                    _validate_transform_state=schema["_validate_transform_state"])["_validate_arrays"]
    validate({"torch": t}, arrays, cpu_only=True)
    for mutation in ("CPU", "crow_dtype", "crow_shape", "col_values", "CSR_finite", "dense_layout"):
        bad = copy.deepcopy(arrays)
        if mutation == "CPU":
            bad["X"].device.type = "cuda"
        elif mutation == "crow_dtype":
            bad["original_CSR"].crow.dtype = "FP64"
        elif mutation == "crow_shape":
            bad["original_CSR"].crow.shape = (169343,)
        elif mutation == "col_values":
            bad["original_CSR"].entries.shape = (4,)
        elif mutation == "CSR_finite":
            bad["original_CSR"].entries.value = float("inf")
        else:
            bad["X"].layout = "CSR"
        with pytest.raises(ValueError):
            validate({"torch": t}, bad, cpu_only=True)
    ns = load("_validate_packet", _validate_arrays=lambda libs, arrays, cpu_only=False: None, _descriptor=lambda libs, arrays: arrays)
    packet = dict(schema=1, context={"source": "current85"}, arrays={"M0": "exact"}, descriptors={"M0": "exact"})
    packet["content_sha256"] = ns["_seal"](dict(context=packet["context"], descriptors=packet["descriptors"]))
    binding = dict(origin_content_sha256=packet["content_sha256"], origin_context_sha256=ns["_seal"](packet["context"]))
    ns["_validate_packet"]({}, packet, binding)
    packet["context"] = {"source": "other"}
    with pytest.raises(ValueError):
        ns["_validate_packet"]({}, packet, binding)


def test_origin_writer_never_overwrites_partial_bytes(tmp_path):
    ns = load("_write_origin", os=SimpleNamespace(fsync=lambda fd: None))
    path = tmp_path / "origin.pt"
    path.write_bytes(b"partial")
    with pytest.raises(FileExistsError):
        ns["_write_origin"]({"torch": SimpleNamespace(save=lambda *args: pytest.fail("overwrite"))}, {}, path, {})
    assert path.read_bytes() == b"partial"


def test_additive_lazy_worker_preserves_whole_protected_AST():
    new = ast.parse(WORKER.read_text())
    dispatch = next(n for n in new.body if isinstance(n, ast.FunctionDef) and n.name == "dispatch")
    branch = next(n for n in dispatch.body if isinstance(n, ast.If)
                  and ast.unparse(n.test) == "job['kind'] == 'large_current_origin'")
    assert len(branch.body) == 2 and isinstance(branch.body[0], ast.ImportFrom)
    assert branch.body[0].module == "src.large_current_origin"
    assert ast.unparse(branch.body[1]) == "return run(**options, stop=stop)"
    dispatch.body.remove(branch)
    assert hashlib.sha256(ast.dump(new, include_attributes=False).encode()).hexdigest() == PROTECTED_WORKER_AST_SHA


@pytest.mark.parametrize("failure", ["after_source_retention", "late_preservation"])
def test_failure_finally_preserves_retained_scope_and_clears_qualification(tmp_path, failure):
    stage = "mock_stage"
    output = tmp_path / "results/research_loop" / stage / "capture_receipt_v1.json"
    origin = output.with_name("origin_cache.pt")
    expected = {"origin_checkpoint_writes": 1, "origin_CPU_copy_calls": 1}
    spec = dict(operation_outputs={"capture": str(output), "replay": str(output.with_name("replay_receipt_v1.json"))},
                origin_path=str(origin), source={"full": 85}, numerical_source={"versions": {"mock": "only"}},
                scientific_preregistration={"path": "frozen", "sha256": "a" * 64}, python_version="mock",
                family_protocol={}, candidate_protocol={}, fixed={})
    t = SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: False))
    clock = SimpleNamespace(monotonic=lambda: 10.0)
    signal = SimpleNamespace(SIGALRM=1, ITIMER_REAL=2, getsignal=lambda n: None, getitimer=lambda n: (0, 0),
                             signal=lambda *a: None, setitimer=lambda *a: None)
    def execute(operation, spec, science, repo, libs, evidence, packet):
        evidence["_retained"] = {"source": "immutable"}
        evidence["native_buffers_before"] = {"source": "immutable"}
        if failure == "after_source_retention":
            raise RuntimeError("new mock failure")
        evidence["_origin_arrays"] = {"origin": "immutable"}
        evidence["origin_buffers_before"] = {"origin": "immutable"}
        return evidence["_origin_arrays"]
    def packet(libs, arrays, context, evidence):
        evidence["counts"]["origin_CPU_copy_calls_attempts"] += 1
        evidence["counts"]["origin_CPU_copy_calls_completed"] += 1
        return {"context": context, "arrays": arrays, "content_sha256": "c" * 64}
    def write(libs, packet, path, evidence):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(b"mock durable bytes")
        evidence.update(origin_cache_path=str(path), origin_cache_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        origin_content_sha256="c" * 64, origin_context_sha256="d" * 64)
    def count(evidence, name, completed=False):
        key = name + ("_completed" if completed else "_attempts")
        evidence["counts"][key] = evidence["counts"].get(key, 0) + 1
    def call(evidence, name, fn, *args, **kwargs):
        evidence["_guard"]()
        count(evidence, name)
        result = fn(*args, **kwargs)
        count(evidence, name, True)
        evidence["_guard"]()
        return result
    def preserve(*args):
        if failure == "late_preservation":
            raise RuntimeError("late mock preservation")
    ns = load("run", __file__=str(tmp_path / "src/owned.py"), STAGE=stage, time=clock, signal=signal,
              _load_spec=lambda *a: (spec, {"counts_contract": {"per_capture_success": expected}}, tmp_path),
              _libraries=lambda: {"torch": t}, _native=lambda *a: {"versions": {"mock": "only"}},
              _precision=lambda *a: None, _execute=execute, _context=lambda *a: {"current": "origin"},
              _packet=packet, _write_origin=write, _call=call, _preserve=preserve,
              _descriptor=lambda libs, arrays: copy.deepcopy(arrays))
    with pytest.raises(RuntimeError):
        ns["run"]("capture", "spec", "b" * 64, str(output))
    receipt = json.loads(output.read_text())
    assert receipt["passed"] is False and receipt["success_counts"] is None
    assert receipt["source_buffers_unchanged"] is True and receipt["current_origin_capture_passed"] is False
    assert receipt["current_origin_independent_replay_qualified"] is False
    if failure == "late_preservation":
        assert origin.read_bytes() == b"mock durable bytes"
        assert receipt["origin_cache_bytes_unchanged"] is True and receipt["counts"]["origin_checkpoint_writes_completed"] == 1
    else:
        assert receipt["origin_buffers_unchanged"] is None and not origin.exists()
