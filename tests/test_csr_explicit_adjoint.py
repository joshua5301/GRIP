"""New representation/validator controls only; no old gradient/FD suite replay."""
import importlib.util
from pathlib import Path

import pytest
import torch

PATH = Path(__file__).resolve().parents[1] / "src/csr_explicit_adjoint.py"
spec = importlib.util.spec_from_file_location("owned_explicit", PATH)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


def csr(crow=None, columns=None, values=None, dtype=torch.float32):
    return torch.sparse_csr_tensor(torch.tensor(crow if crow is not None else [0, 2, 2, 4], dtype=torch.int64),
        torch.tensor(columns if columns is not None else [0, 2, 0, 1], dtype=torch.int64),
        torch.tensor(values if values is not None else [-0., 2., 0., -3.], dtype=dtype), size=(3, 3))


def test_FP32_coordinate_bijection_stored_signed_zero_and_empty_row():
    source = csr()
    before = (source.crow_indices().clone(), source.col_indices().clone(), source.values().view(torch.int32).clone())
    target = helper.transpose_csr(source)
    assert target.crow_indices().tolist() == [0, 2, 3, 4]
    assert target.col_indices().tolist() == [0, 2, 2, 0]
    order = torch.tensor([0, 2, 3, 1])
    assert torch.equal(target.values().view(torch.int32), before[2].index_select(0, order))
    assert target.values().view(torch.int32)[0].item() == -2147483648
    assert target.values().view(torch.int32)[1].item() == 0
    assert all(torch.equal(v, b) for v, b in zip((source.crow_indices(), source.col_indices(), source.values().view(torch.int32)), before))
    assert not target.requires_grad and target._nnz() == source._nnz() == 4


def test_constructor_interface_counts_without_GPU_claim():
    evidence = dict(_guard=lambda: None, operation_attempts={}, counts={})
    helper.transpose_csr(csr(), evidence)
    expected = dict(transpose_bulk_index_host_copies=2, transpose_GPU_integer_uploads=3, transpose_GPU_value_gathers=1)
    assert evidence["counts"] == evidence["operation_attempts"] == expected
    # Logical interface labels on a CPU fixture do not certify a native CUDA call.


@pytest.mark.parametrize("kind", ["duplicate", "unsorted", "range", "monotonic", "nonfinite", "integer", "grad", "nonsquare"])
def test_vectorized_CSR_validator_rejects_invalid_representation(kind):
    if kind == "duplicate": source = csr(columns=[0, 0, 0, 1])
    elif kind == "unsorted": source = csr(columns=[2, 0, 0, 1])
    elif kind == "range": source = csr(columns=[0, 3, 0, 1])
    elif kind == "monotonic": source = csr(crow=[0, 3, 2, 4])
    elif kind == "nonfinite": source = csr(values=[0., float("inf"), 0., 1.])
    elif kind == "integer": source = csr(dtype=torch.int64)
    elif kind == "grad": source = csr().requires_grad_(True)
    else: source = torch.sparse_csr_tensor(torch.tensor([0, 1, 1]), torch.tensor([0]), torch.tensor([1.]), size=(2, 3))
    with pytest.raises(ValueError): helper._csr(source)


def test_invoke_records_attempt_only_on_failure():
    evidence = dict(_guard=lambda: None, operation_attempts={}, counts={})
    def fail(): raise RuntimeError("primitive failed")
    with pytest.raises(RuntimeError, match="primitive failed"):
        helper._invoke(evidence, "new_primitive", fail)
    assert evidence["operation_attempts"] == {"new_primitive": 1} and evidence["counts"] == {}
