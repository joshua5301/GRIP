"""CPU-only AST guards: no adapter/data/native GPU import or execution."""
import ast
import hashlib
import math
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src.io import array_digest


@pytest.fixture
def namespace():
    root = Path(__file__).resolve().parents[1]
    path = root / "finite_student_probe.py"
    if not path.is_file():
        path = root / "src/finite_student_probe.py"
    parsed = ast.parse(path.read_text())
    names = {"_require", "_digest", "_seal", "_tensor", "canonical_candidate", "_native",
             "_runtime_precision_guard", "_frozen_original_dense_S", "_evaluate", "_source_unchanged"}
    constants = {"SCHEMA", "_FIXED", "_SOURCE_FIELD", "_ROUTES"}
    nodes = [node for node in parsed.body if
             isinstance(node, ast.FunctionDef) and node.name in names or
             isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id in constants
                                                  for target in node.targets)]
    result = dict(torch=torch, os=os, math=math, hashlib=hashlib, array_digest=array_digest,
                  json=__import__("json"), STEPS=5, LEARNING_RATE=.1, WEIGHT_DECAY=.001, INITIAL_SEED=0,
                  numerical_source=lambda: {"test_only_frozen_source": 1})
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), result)
    result["_stop"] = lambda stop: None
    return result


def original_graph():
    return torch.tensor([[1., 0., .2, 0.], [0., 2., 0., 0.],
                         [.3, 0., 4., .5], [0., .6, 0., 3.]], dtype=torch.float32).to_sparse_csr()


def test_exact_original_entries_zero_edges_and_independent_buffers(namespace):
    original = original_graph()
    before = namespace["_seal"](original)
    dense = namespace["_frozen_original_dense_S"](original)
    expected = torch.tensor([[1., 0., .2, 0.], [0., 2., 0., 0.],
                             [.3, 0., 4., .5], [0., .6, 0., 3.]], dtype=torch.float32)
    assert torch.equal(dense, expected)
    assert dense.device == original.device and dense.dtype == torch.float32
    assert dense.layout == torch.strided and dense.is_contiguous() and not dense.requires_grad
    assert namespace["_seal"](dense.to_sparse_csr()) == before
    dense[0, 0] = 12
    assert namespace["_seal"](original) == before


@pytest.mark.parametrize("corruption", ["dense", "dtype", "nonsquare", "grad", "nan", "zero", "duplicate"])
def test_original_graph_rejections(namespace, corruption):
    original = original_graph()
    if corruption == "dense":
        original = original.to_dense()
    elif corruption == "dtype":
        original = original.double()
    elif corruption == "nonsquare":
        original = torch.ones(2, 3).to_sparse_csr()
    elif corruption == "grad":
        original.requires_grad_()
    elif corruption in ("nan", "zero"):
        original.values()[0] = float("nan") if corruption == "nan" else 0
    else:
        original = torch.sparse_csr_tensor(torch.tensor([0, 2, 2]), torch.tensor([0, 0]),
                                          torch.tensor([1., 2.]), size=(2, 2))
    with pytest.raises(ValueError):
        namespace["_frozen_original_dense_S"](original)


def candidate(namespace):
    return dict(namespace["_FIXED"], outer_route="sgc_mlp")


def test_new_namespace_explicit_schema_backend_policy(namespace):
    value = candidate(namespace)
    result = namespace["canonical_candidate"](value)
    assert result["finite_student_schema"] == 2
    assert result["finite_graph_backend"] == "dense_original_S"
    assert result["finite_native_policy"] == "deterministic_cuda_v1"
    assert namespace["_SOURCE_FIELD"] in result
    assert namespace["_SOURCE_FIELD"] not in value


@pytest.mark.parametrize("change", ["schema1", "backend", "policy", "missing", "external"])
def test_old_unknown_or_external_controls_rejected(namespace, change):
    value = candidate(namespace)
    if change == "schema1":
        value["finite_student_schema"] = 1
    elif change == "backend":
        value["finite_graph_backend"] = "coo"
    elif change == "policy":
        value["finite_native_policy"] = "warn_only"
    elif change == "missing":
        del value["finite_graph_backend"]
    else:
        value["source_adjacency"] = "external"
    with pytest.raises(ValueError):
        namespace["canonical_candidate"](value)


class FakeTorch:
    float32 = torch.float32

    def __init__(self):
        self.deterministic, self.warn = False, True
        self.precision, self.dtype = "high", torch.float32
        self.amp_cuda, self.amp_cpu = False, False
        self.calls = []
        self.backends = SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
                                        cudnn=SimpleNamespace(allow_tf32=True))
        self.cuda = SimpleNamespace(is_available=self.available, current_device=lambda: 0,
                                    get_device_name=lambda: "MOCK_ONLY_NO_GPU")
        self.version = SimpleNamespace(cuda="12.8")
        self.__version__ = "MOCK_ONLY"

    def available(self):
        self.calls.append("CUDA_available")
        return True

    def get_default_dtype(self):
        return self.dtype

    def is_autocast_enabled(self, device="cuda"):
        return self.amp_cpu if device == "cpu" else self.amp_cuda

    def use_deterministic_algorithms(self, mode, *, warn_only):
        self.calls.append("deterministic_set")
        self.deterministic, self.warn = mode, warn_only

    def set_float32_matmul_precision(self, value):
        self.calls.append("precision_set")
        self.precision = value

    def are_deterministic_algorithms_enabled(self):
        return self.deterministic

    def is_deterministic_algorithms_warn_only_enabled(self):
        return self.warn

    def get_float32_matmul_precision(self):
        return self.precision

    @staticmethod
    def device(name, index):
        return f"{name}:{index}"


@pytest.mark.parametrize("workspace", [None, "", ":16:8", ":4096:2"])
def test_workspace_guard_precedes_cuda_availability(namespace, monkeypatch, workspace):
    fake = FakeTorch()
    namespace["torch"] = fake
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    if workspace is not None:
        monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", workspace)
    with pytest.raises(ValueError, match="CUBLAS_WORKSPACE_CONFIG"):
        namespace["_native"]("cuda")
    assert fake.calls == []


def test_native_initializes_and_records_strict_policy_without_real_cuda(namespace, monkeypatch):
    fake = FakeTorch()
    namespace["torch"] = fake
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    environment = namespace["_native"]("cuda")
    assert fake.calls == ["deterministic_set", "precision_set", "CUDA_available"]
    assert environment["deterministic_algorithms"] is True
    assert environment["deterministic_warn_only"] is False
    assert environment["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert environment["float32_matmul_precision"] == "highest"
    assert environment["finite_graph_backend"] == "dense_original_S"
    assert environment["cuda_runtime"] == "12.8"
    assert fake.backends.cuda.matmul.allow_tf32 is False and fake.backends.cudnn.allow_tf32 is False


@pytest.mark.parametrize("changed", ["deterministic", "warn", "precision", "tf32", "cudnn", "dtype", "amp_cuda", "amp_cpu", "workspace"])
def test_changed_runtime_rejected_before_evaluation_without_repair(namespace, monkeypatch, changed):
    fake = FakeTorch()
    namespace["torch"] = fake
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    namespace["_native"]("cuda")
    fake.calls.clear()
    if changed == "deterministic":
        fake.deterministic = False
    elif changed == "warn":
        fake.warn = True
    elif changed == "precision":
        fake.precision = "high"
    elif changed == "tf32":
        fake.backends.cuda.matmul.allow_tf32 = True
    elif changed == "cudnn":
        fake.backends.cudnn.allow_tf32 = True
    elif changed == "dtype":
        fake.dtype = torch.float64
    elif changed == "amp_cuda":
        fake.amp_cuda = True
    elif changed == "amp_cpu":
        fake.amp_cpu = True
    else:
        monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    namespace["_moments"] = lambda *args: pytest.fail("No moments/math call permitted")
    with pytest.raises(ValueError, match="precision/environment"):
        namespace["_evaluate"]({}, {}, {}, [], 0, None, lambda: False)
    with pytest.raises(ValueError, match="precision/environment"):
        namespace["_source_unchanged"]({})
    assert fake.calls == []
