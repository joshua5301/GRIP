"""Frozen CPU SciPy FP32 CSR graph map and explicit transpose pullback.

Compose GraphProduct.apply(baseU, operator) with the unchanged LowRankMoments.
This module maps only U: it contains no moment, head, optimizer or graph-fit code.
The caller separately pins source/origin/versions and qualifies its native device
copies.  There is no sparse backend, precision or normalization fallback.
"""
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import scipy
import scipy.sparse as sparse
import torch
from torch.autograd.function import once_differentiable


SCHEMA = 1
MODE = "cpu_scipy_FP32_CSR_graph_factor_v1"
THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "CUDA_VISIBLE_DEVICES",
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _json_copy(value):
    try:
        return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError("Graph source references must be finite JSON") from error


def _array_descriptor(value):
    return dict(shape=list(value.shape), dtype=value.dtype.str,
                content_sha256=hashlib.sha256(value.tobytes(order="C")).hexdigest())


def _runtime_descriptor():
    path = Path(__file__).resolve()
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=path.parents[1],
                              capture_output=True, text=True, check=False)
    return dict(
        versions=dict(Python=platform.python_version(), NumPy=np.__version__,
                      SciPy=scipy.__version__, Torch=str(torch.__version__)),
        library_paths=dict(NumPy=str(Path(np.__file__).resolve()),
                           SciPy=str(Path(scipy.__file__).resolve()),
                           SciPy_sparse=str(Path(sparse.__file__).resolve()),
                           Torch=str(Path(torch.__file__).resolve())),
        executable=str(Path(sys.executable).resolve()), machine=platform.machine(),
        CPU_threads=torch.get_num_threads(), CPU_interop_threads=torch.get_num_interop_threads(),
        thread_environment={key: os.environ.get(key) for key in THREAD_ENVIRONMENT},
        implementation=dict(module_path=str(path),
            module_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            git_head=revision.stdout.strip() if revision.returncode == 0 else None))


def _canonical_arrays(rowptr, columns, values):
    for value, dtype, name in ((rowptr, np.int64, "rowptr"),
                               (columns, np.int64, "col"),
                               (values, np.float32, "values")):
        _require(isinstance(value, np.ndarray) and value.ndim == 1 and value.dtype == np.dtype(dtype),
                 "Graph " + name + " must be a rank1 array with exact native dtype")
    _require(len(rowptr) >= 2, "Graph CSR needs positive square size")
    nodes, entries = len(rowptr) - 1, len(columns)
    _require(len(values) == entries and int(rowptr[0]) == 0 and int(rowptr[-1]) == entries
             and bool(((rowptr >= 0) & (rowptr <= entries)).all())
             and bool((rowptr[1:] >= rowptr[:-1]).all()), "Malformed graph CSR row envelope")
    _require(bool(((columns >= 0) & (columns < nodes)).all())
             and bool(np.isfinite(values).all()) and bool((values >= 0).all()),
             "Graph CSR needs finite nonnegative values and bounded columns")
    rows = np.repeat(np.arange(nodes, dtype=np.int64), rowptr[1:] - rowptr[:-1])
    _require(len(rows) == entries and bool(((rows[1:] != rows[:-1])
             | (columns[1:] > columns[:-1])).all()),
             "Graph CSR rows must be strictly ascending and unique; no coalescing")
    arrays = tuple(np.array(value, dtype=value.dtype, order="C", copy=True)
                   for value in (rowptr, columns, values))
    return arrays, rows


def _readonly_csr(arrays, nodes):
    """Attach exact int64/FP32 buffers without constructor downcasting or arithmetic."""
    rowptr, columns, values = arrays
    matrix = sparse.csr_matrix((nodes, nodes), dtype=np.float32)
    matrix.indptr, matrix.indices, matrix.data = rowptr, columns, values
    # Canonicality was independently checked; these flags avoid library repair.
    matrix.has_sorted_indices = True
    matrix.has_canonical_format = True
    for value in arrays:
        value.setflags(write=False)
    _require(matrix.shape == (nodes, nodes) and matrix.dtype == np.dtype(np.float32)
             and matrix.indptr.dtype == matrix.indices.dtype == np.dtype(np.int64)
             and matrix.indptr is rowptr and matrix.indices is columns and matrix.data is values,
             "SciPy did not retain exact canonical int64/FP32 buffers")
    return matrix


class FrozenCPUCSR:
    """Owning immutable canonical S and its bit-preserving sorted CSR transpose.

    forward/transpose return detached FP32 tensors on the operand's exact device.
    Differentiation belongs to GraphProduct; the descriptor never includes mutable
    work counters.  Explicit zero coefficients and empty rows are retained.
    """

    def __init__(self, rowptr, col, values, rank, source_refs=None):
        arrays, rows = _canonical_arrays(rowptr, col, values)
        self.__nodes = len(arrays[0]) - 1
        _require(type(rank) is int and 1 <= rank <= self.__nodes, "Graph factor rank must be a positive integer")
        _require(source_refs is None or isinstance(source_refs, dict), "Graph source_refs must be a JSON object")
        self.__rank = rank
        self.__source_refs = _json_copy({} if source_refs is None else source_refs)
        self.__arrays = arrays
        columns, coefficients = arrays[1], arrays[2]
        permutation = np.array(np.lexsort((rows, columns)), dtype=np.int64, order="C", copy=True)
        counts = np.bincount(columns, minlength=self.__nodes).astype(np.int64, copy=False)
        t_rowptr = np.empty(self.__nodes + 1, dtype=np.int64)
        t_rowptr[0] = 0
        np.cumsum(counts, dtype=np.int64, out=t_rowptr[1:])
        t_columns = np.array(rows[permutation], dtype=np.int64, order="C", copy=True)
        t_values = np.array(coefficients[permutation], dtype=np.float32, order="C", copy=True)
        t_arrays, _ = _canonical_arrays(t_rowptr, t_columns, t_values)
        _require(np.array_equal(t_arrays[2].view(np.uint32), coefficients[permutation].view(np.uint32)),
                 "Graph transpose changed coefficient bits")
        self.__transpose_arrays = t_arrays
        permutation.setflags(write=False)
        self.__permutation = permutation
        self.__S = _readonly_csr(self.__arrays, self.__nodes)
        self.__T = _readonly_csr(self.__transpose_arrays, self.__nodes)
        self.__runtime = _runtime_descriptor()
        self.__counts = dict(S_forward_products=0, T_transpose_products=0)
        self.__expected = self.__current_descriptor()
        self.__check_frozen()

    @classmethod
    def from_torch(cls, source_CSR, rank, source_refs=None):
        _require(isinstance(source_CSR, torch.Tensor) and source_CSR.layout == torch.sparse_csr
                 and source_CSR.dtype == torch.float32 and source_CSR.ndim == 2
                 and source_CSR.shape[0] == source_CSR.shape[1] and source_CSR.shape[0] > 0
                 and source_CSR.device.type in ("cpu", "cuda") and not source_CSR.requires_grad,
                 "Require the frozen square original FP32 source CSR")
        _require(source_CSR.crow_indices().dtype == source_CSR.col_indices().dtype == torch.int64,
                 "Original source CSR index buffers must be int64")
        buffers = [value.detach().to(device="cpu").contiguous().numpy().copy(order="C")
                   for value in (source_CSR.crow_indices(), source_CSR.col_indices(), source_CSR.values())]
        return cls(*buffers, rank=rank, source_refs=source_refs)

    def __current_descriptor(self):
        envelope = lambda arrays: {key: _array_descriptor(value)
            for key, value in zip(("rowptr", "col", "values"), arrays, strict=True)}
        return dict(schema=SCHEMA, mode=MODE, nodes=self.__nodes, rank=self.__rank,
            shape=[self.__nodes, self.__nodes], S=envelope(self.__arrays),
            T=envelope(self.__transpose_arrays),
            transpose_permutation=_array_descriptor(self.__permutation),
            transpose_policy="lexsort(original_destination,original_source); exact FP32 coefficient-bit permutation",
            source_refs=_json_copy(self.__source_refs), runtime=_json_copy(self.__runtime),
            factor_policy=dict(dtype="torch.float32", layout="torch.strided",
                shape=[self.__nodes, self.__rank], device="same exact original CPU-or-CUDA operand device",
                CPU_operand="owning contiguous native FP32 copy", output="owning contiguous FP32",
                adjoint="fixed explicit FP32 T product after original LowRank native cast/rank/GEMMs",
                source_gradient=False, normalization=False, fallback=False))

    def __check_frozen(self):
        for matrix, arrays in ((self.__S, self.__arrays), (self.__T, self.__transpose_arrays)):
            rowptr, columns, values = arrays
            _require(matrix.shape == (self.__nodes, self.__nodes)
                     and matrix.dtype == np.dtype(np.float32)
                     and matrix.indptr is rowptr and matrix.indices is columns and matrix.data is values
                     and rowptr.dtype == columns.dtype == np.dtype(np.int64)
                     and values.dtype == np.dtype(np.float32)
                     and matrix.has_sorted_indices and matrix.has_canonical_format,
                     "Frozen SciPy graph structure, dtype or identity changed")
            _require(all(value.flags.c_contiguous and value.flags.owndata and not value.flags.writeable
                         for value in arrays), "Frozen graph buffers lost owning readonly storage")
        _require(self.__permutation.flags.c_contiguous and self.__permutation.flags.owndata
                 and not self.__permutation.flags.writeable, "Frozen transpose permutation changed ownership")
        _require(self.__current_descriptor() == self.__expected
                 and _runtime_descriptor() == self.__runtime,
                 "Frozen graph buffers/source refs/operator/runtime descriptor changed")

    def descriptor(self):
        self.__check_frozen()
        return _json_copy(self.__expected)

    def counts(self):
        """Number of actual returned SciPy dot calls, including later rejected output."""
        return dict(self.__counts)

    def __operand(self, value):
        _require(isinstance(value, torch.Tensor) and value.layout == torch.strided
                 and value.dtype == torch.float32 and tuple(value.shape) == (self.__nodes, self.__rank)
                 and value.device.type in ("cpu", "cuda") and bool(torch.isfinite(value).all()),
                 "Graph factor operand must be finite native FP32 (nodes,rank) on CPU or CUDA")
        return value.detach().to(device="cpu").contiguous().numpy().copy(order="C")

    def __product(self, value, *, transpose):
        self.__check_frozen()
        operand = self.__operand(value)
        operand_descriptor = _array_descriptor(operand)
        matrix = self.__T if transpose else self.__S
        result = matrix.dot(operand)
        counter = "T_transpose_products" if transpose else "S_forward_products"
        self.__counts[counter] += 1
        _require(isinstance(result, np.ndarray) and result.dtype == np.dtype(np.float32)
                 and result.shape == (self.__nodes, self.__rank) and result.flags.c_contiguous
                 and bool(np.isfinite(result).all()), "SciPy returned malformed/nonfinite/non-FP32 factor product")
        _require(_array_descriptor(operand) == operand_descriptor
                 and _array_descriptor(self.__operand(value)) == operand_descriptor,
                 "Graph product changed the copied or original factor operand")
        output = torch.from_numpy(np.array(result, dtype=np.float32, order="C", copy=True)).to(device=value.device)
        _require(output.dtype == value.dtype and output.device == value.device
                 and tuple(output.shape) == tuple(value.shape) and output.is_contiguous()
                 and not output.requires_grad and bool(torch.isfinite(output).all()),
                 "Graph product copy changed native factor device/dtype/shape")
        self.__check_frozen()
        return output

    def forward(self, U):
        return self.__product(U, transpose=False)

    def transpose(self, gW):
        return self.__product(gW, transpose=True)


class GraphProduct(torch.autograd.Function):
    """Only the fixed graph map U→W; compose with the original moment Function."""

    @staticmethod
    def forward(ctx, U, operator):
        _require(isinstance(operator, FrozenCPUCSR), "GraphProduct requires FrozenCPUCSR")
        _require(isinstance(U, torch.Tensor), "GraphProduct requires a native factor tensor")
        ctx.operator = operator
        ctx.shape, ctx.dtype, ctx.device = tuple(U.shape), U.dtype, U.device
        return operator.forward(U)

    @staticmethod
    @once_differentiable
    def backward(ctx, gW):
        _require(isinstance(gW, torch.Tensor) and tuple(gW.shape) == ctx.shape
                 and gW.dtype == ctx.dtype and gW.device == ctx.device,
                 "GraphProduct received a changed native cotangent boundary")
        gU = ctx.operator.transpose(gW)
        return gU, None
