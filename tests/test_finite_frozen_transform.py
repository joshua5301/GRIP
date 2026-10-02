"""Tiny pure-helper regression; AST extraction never imports the data adapter."""

import ast
from pathlib import Path

import pytest
import torch


@pytest.fixture
def frozen_transform():
    path = Path(__file__).resolve().parents[1] / "src/finite_student_probe.py"
    tree = ast.parse(path.read_text())
    function = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name == "_frozen_transform"]
    assert len(function) == 1
    space = {"torch": torch}
    exec(compile(ast.Module(body=function, type_ignores=[]), str(path), "exec"), space)
    return space["_frozen_transform"]


class RuntimeRMS:
    def __init__(self, center, output_center, scale):
        self.center = center
        self.matrix = None
        self.output_center = output_center
        self.scale = scale
        self.kind = "rms"
        self.eps = 1e-12

    def state_dict(self):
        raise AssertionError("Runtime RMS buffers must not use the CPU serialization path")


def test_fp64_runtime_values_precision_device_and_metadata_are_preserved(frozen_transform, monkeypatch):
    transform = RuntimeRMS(torch.tensor([1. + 2.**-40, -2. - 2.**-39], dtype=torch.float64),
                           torch.tensor([.25 + 2.**-41, -.5], dtype=torch.float64),
                           torch.tensor(1.3 + 2.**-40, dtype=torch.float64))
    calls = []
    def forbidden_state_dict(self):
        calls.append(self)
        raise AssertionError("state_dict would move original native buffers to CPU")
    monkeypatch.setattr(RuntimeRMS, "state_dict", forbidden_state_dict)
    actual = frozen_transform(transform)
    assert calls == []
    assert set(actual) == set(vars(transform))
    for name in ("center", "output_center", "scale"):
        original = getattr(transform, name)
        assert actual[name].dtype == original.dtype
        assert actual[name].device == original.device
        assert actual[name].shape == original.shape
        assert torch.equal(actual[name], original)
    assert actual["scale"].ndim == 0
    assert actual["matrix"] is None and actual["kind"] == "rms" and actual["eps"] == 1e-12


def test_returned_buffers_are_detached_and_cannot_mutate_original(frozen_transform):
    transform = RuntimeRMS(torch.tensor([.1, .2], dtype=torch.float64, requires_grad=True),
                           torch.tensor([.3, .4], dtype=torch.float64, requires_grad=True),
                           torch.tensor(1.7, dtype=torch.float64, requires_grad=True))
    old = {name: getattr(transform, name).detach().clone() for name in ("center", "output_center", "scale")}
    actual = frozen_transform(transform)
    for name in old:
        assert not actual[name].requires_grad and actual[name].grad_fn is None
        assert actual[name].data_ptr() != getattr(transform, name).data_ptr()
        actual[name].add_(5.)
        assert torch.equal(getattr(transform, name), old[name])
        assert getattr(transform, name).requires_grad


def test_each_snapshot_captures_current_buffers_without_aliasing_prior_snapshot(frozen_transform):
    transform = RuntimeRMS(torch.tensor([.1, .2], dtype=torch.float32),
                           torch.tensor([.3, .4], dtype=torch.float64),
                           torch.tensor(1.7, dtype=torch.float64))
    first = frozen_transform(transform)
    original_first = {name: first[name].clone() for name in ("center", "output_center", "scale")}
    transform.center.add_(.5)
    transform.output_center.sub_(.25)
    transform.scale.mul_(2.)
    second = frozen_transform(transform)
    for name in original_first:
        assert torch.equal(first[name], original_first[name])
        assert torch.equal(second[name], getattr(transform, name))
        assert first[name].data_ptr() != second[name].data_ptr()
    assert second["center"].dtype == torch.float32
    assert second["output_center"].dtype == second["scale"].dtype == torch.float64
