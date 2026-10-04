"""UNQUALIFIED draft: frozen Phi row tiles, native NODE moments, streamed CE.

No optimizer, head fit, feature factory, graph product or source dispatch lives
here. Large-source admission and fresh mathematical/native gates are external.
Evidence records attempted and returned operations separately. On failure its
partial_arrays owns CPU copies; a partial result is never a completed result.
"""
import hashlib
import json
import math
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch

from src.low_rank_assignment import logit_block
from src.moments import augmented, make_material


SCHEMA = 1
BACKEND = "original_Phi_FP64_complete_row_tile_native_NODE_v1"
MAX_SECONDS = 300
MAX_TILE_PAYLOAD_BYTES = 512 * 1024**2


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, minimum=0):
    _require(type(value) is int and value >= minimum, "Typed integer required")
    return value


def _digest(value):
    _require(type(value) is str and len(value) == 64
             and all(c in "0123456789abcdef" for c in value), "SHA256 required")
    return value


def _json_copy(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _evidence(value):
    _require(isinstance(value, dict), "Caller-owned evidence dictionary required")
    value.setdefault("backend", BACKEND)
    _require(value["backend"] == BACKEND, "Mixed evidence backend")
    value.setdefault("counts", {})
    value.setdefault("bytes", {})
    value.setdefault("failures", [])
    _require(isinstance(value["counts"], dict) and isinstance(value["bytes"], dict)
             and isinstance(value["failures"], list), "Malformed evidence")
    value["completion_scope"] = "Python operation returned; CUDA synchronization is owned by the bounded caller"
    return value


def _count(evidence, key, amount=1):
    values = _evidence(evidence)["counts"]
    _integer(amount); _integer(values.get(key, 0))
    values[key] = values.get(key, 0) + amount


def _bytes(evidence, key, size):
    _integer(size)
    values = _evidence(evidence)["bytes"]
    _integer(values.get(key, 0))
    values[key] = values.get(key, 0) + size


def _call(evidence, key, function, *args):
    _count(evidence, key + "_attempts")
    evidence["last_started_operation"] = key
    result = function(*args)
    _count(evidence, key + "_completed")
    evidence["last_completed_operation"] = key
    return result


def _guard(stop, started=None):
    _require(callable(stop), "Callable stop required")
    if stop() or started is not None and time.monotonic() - started > MAX_SECONDS:
        raise InterruptedError("Bounded row-tile operation stopped; preserve incomplete evidence")


def _failure(evidence, operation, error, row_end, arrays):
    evidence["failures"].append(dict(operation=operation,
        error_type=type(error).__name__, error=str(error), completed_row_end=row_end,
        last_returned_progress=_json_copy(evidence.get("last_returned_progress")),
        operation_interiors="unknown; only returned operation counts are completed"))
    evidence["partial_result_is_complete"] = False
    try:
        captured = {}
        for key, value in arrays.items():
            captured[key] = _call(evidence, "failure_raw_CPU_copy", lambda: value.detach().cpu().clone())
            _bytes(evidence, "failure_raw_CPU_copy", captured[key].numel() * captured[key].element_size())
        evidence.setdefault("partial_arrays", []).append(dict(
            operation=operation, completed_row_end=row_end, arrays=captured))
    except BaseException as preservation_error:
        evidence["failures"].append(dict(operation="partial_evidence_preservation",
            error_type=type(preservation_error).__name__, error=str(preservation_error)))
        error.add_note("Partial-array preservation also failed: " + repr(preservation_error))


def runtime_descriptor(device):
    """Observed arithmetic backend; callers bind it before native admission."""
    return dict(Python=platform.python_version(), NumPy=str(np.__version__),
        Torch=str(torch.__version__), CUDA=torch.version.cuda, device=str(device),
        default_dtype=str(torch.get_default_dtype()),
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        TF32=bool(torch.backends.cuda.matmul.allow_tf32),
        deterministic=bool(torch.are_deterministic_algorithms_enabled()),
        threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
        thread_environment={name: os.environ.get(name) for name in
            ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "CUBLAS_WORKSPACE_CONFIG")})


class FrozenPhiRows:
    """Read-only C-order FP64 mmap or an owning copy of a fresh CPU toy array.

    Construction validates the complete content once in rows, without a full
    boolean/copy allocation. Metadata checks guard each pass and tile. External
    native callers must call assert_immutable(full=True) at final admission or
    failure cleanup to verify all file/content bytes again. Stat equality alone
    is explicitly not a complete preservation proof.
    """
    def __init__(self, array, *, expected_shape, expected_content_digest,
                 source_refs, expected_file_sha256=None, validation_chunk=4096,
                 tile_payload_limit=MAX_TILE_PAYLOAD_BYTES, evidence, stop=lambda: False):
        _evidence(evidence); _integer(validation_chunk, 1)
        _integer(tile_payload_limit, 1)
        _require(tile_payload_limit <= MAX_TILE_PAYLOAD_BYTES, "Tile ceiling exceeds frozen bound")
        _require(isinstance(expected_shape, (list, tuple)) and len(expected_shape) == 2
                 and all(type(x) is int and x > 0 for x in expected_shape), "Positive matrix shape required")
        _require(isinstance(array, np.ndarray) and array.ndim == 2
                 and tuple(array.shape) == tuple(expected_shape) and array.dtype == np.dtype(np.float64)
                 and array.flags.c_contiguous, "Phi must be exact native C-order FP64")
        _require(isinstance(source_refs, dict) and source_refs, "Typed nonempty source refs required")
        self._source_refs = _json_copy(source_refs)
        self._shape, self._digest = tuple(expected_shape), _digest(expected_content_digest)
        self._limit, self._validation_chunk = tile_payload_limit, validation_chunk
        self._file = None; self._file_sha256 = None; self._stat = None
        if isinstance(array, np.memmap):
            _require(array.mode == "r" and not array.flags.writeable
                     and expected_file_sha256 is not None, "Readonly file mapping requires complete file pin")
            self._file = str(Path(array.filename).resolve())
            self._file_sha256 = _digest(expected_file_sha256)
            self._array = array
            self._stat = self._file_stat()
        else:
            _require(expected_file_sha256 is None and array.nbytes < self._limit,
                     "Only a bounded owning CPU toy may omit a file pin")
            self._array = _call(evidence, "toy_owned_CPU_copy", lambda: np.array(array, copy=True, order="C"))
            self._array.setflags(write=False)
            _require(self._array.flags.owndata, "CPU toy array must own its storage")
        self._array_id = id(self._array)
        self.assert_immutable(full=True, evidence=evidence, stop=stop)

    @classmethod
    def from_npy(cls, path, **kwargs):
        """Load-only; mandatory original sidecar/source identity is caller-pinned."""
        array = np.load(Path(path), mmap_mode="r", allow_pickle=False)
        _require(isinstance(array, np.memmap), "Existing uncompressed npy mapping required")
        return cls(array, **kwargs)

    @property
    def shape(self):
        return self._shape

    @property
    def owned_CPU_toy(self):
        return self._file is None

    def _file_stat(self):
        s = Path(self._file).stat()
        return dict(device=s.st_dev, inode=s.st_ino, size=s.st_size,
                    mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns)

    def _metadata(self):
        _require(id(self._array) == self._array_id and self._array.shape == self._shape
                 and self._array.dtype == np.dtype(np.float64) and self._array.flags.c_contiguous
                 and not self._array.flags.writeable, "Phi mapping/storage metadata changed")
        if self._file is not None:
            _require(isinstance(self._array, np.memmap) and self._array.mode == "r"
                     and str(Path(self._array.filename).resolve()) == self._file
                     and self._file_stat() == self._stat, "Original Phi file identity changed")

    def _file_digest(self, evidence, stop, started):
        digest = hashlib.sha256()
        with Path(self._file).open("rb") as stream:
            while True:
                _guard(stop, started)
                block = stream.read(8 * 1024**2)
                if not block:
                    break
                digest.update(block); _bytes(evidence, "source_file_hash", len(block))
        return digest.hexdigest()

    def _content_digest(self, evidence, stop, started):
        # Same shape/dtype header as original canonical-FP64 cached-Phi digest.
        digest = hashlib.sha256(json.dumps(dict(shape=self._shape, dtype="float64"), sort_keys=True).encode())
        for start in range(0, len(self._array), self._validation_chunk):
            _guard(stop, started); self._metadata()
            block = self._array[start:start + self._validation_chunk]
            _require(block.nbytes < self._limit and bool(np.isfinite(block).all()), "Invalid/oversized Phi validation tile")
            digest.update(memoryview(block).cast("B"))
            _bytes(evidence, "Phi_content_validation", block.nbytes)
            _count(evidence, "Phi_validation_tiles_completed")
        return digest.hexdigest()

    def assert_immutable(self, *, full=False, evidence, stop=lambda: False):
        _require(type(full) is bool, "Typed full-validation flag required")
        started = time.monotonic(); _guard(stop, started); self._metadata()
        if full:
            if self._file is not None:
                actual = _call(evidence, "source_file_full_hash", self._file_digest, evidence, stop, started)
                _require(actual == self._file_sha256, "Original complete Phi file bytes changed")
            actual = _call(evidence, "Phi_full_content_validation", self._content_digest, evidence, stop, started)
            _require(actual == self._digest, "Original Phi content changed")
            self._metadata()

    def descriptor(self):
        self._metadata()
        return dict(schema=SCHEMA, backend=BACKEND, shape=list(self._shape), dtype="float64",
            storage="readonly_npy_memmap" if self._file else "owned_frozen_CPU_toy",
            file=None if self._file is None else dict(path=self._file, sha256=self._file_sha256, stat=dict(self._stat)),
            content_digest=self._digest, tile_payload_limit=self._limit,
            source_refs=_json_copy(self._source_refs))

    def tile(self, start, end, device, *, evidence, stop=lambda: False):
        _integer(start); _integer(end, start + 1)
        _require(end <= self._shape[0], "Out-of-range Phi row slice")
        _guard(stop); self._metadata()
        cpu = _call(evidence, "Phi_owned_row_tile", lambda: np.array(self._array[start:end], copy=True, order="C"))
        _require(cpu.dtype == np.float64 and cpu.flags.owndata and cpu.flags.c_contiguous
                 and cpu.nbytes < self._limit and bool(np.isfinite(cpu).all()), "Invalid/oversized copied Phi rows")
        _bytes(evidence, "Phi_owned_row_tile", cpu.nbytes)
        _guard(stop)
        result = _call(evidence, "Phi_row_device_copy", lambda: torch.from_numpy(cpu).to(device=device))
        _require(result.dtype == torch.float64 and result.device == device and result.is_contiguous()
                 and not result.requires_grad, "Phi row placement/dtype changed")
        _bytes(evidence, "Phi_row_device_copy", result.numel() * result.element_size())
        return result


def _tensor(value, shape, dtype, device):
    _require(torch.is_tensor(value) and value.layout == torch.strided
             and tuple(value.shape) == tuple(shape) and value.dtype == dtype
             and value.device == device and value.is_contiguous()
             and bool(torch.isfinite(value).all()), "Tensor shape/dtype/device/finiteness changed")


def _arithmetic(device, evidence):
    _require(device.type in ("cpu", "cuda") and torch.get_default_dtype() == torch.float32
             and not torch.is_autocast_enabled(device.type)
             and not torch.backends.cuda.matmul.allow_tf32, "Original native FP32 arithmetic policy required")
    actual = runtime_descriptor(device)
    if "runtime" in evidence:
        _require(evidence["runtime"] == actual, "Row-tile arithmetic backend changed")
    else:
        evidence["runtime"] = actual


def _inputs(u, v, assignment, rows, q, mixing, chunk, evidence):
    _evidence(evidence); _integer(chunk, 1)
    _require(isinstance(rows, FrozenPhiRows) and torch.is_tensor(u) and u.ndim == 2
             and torch.is_tensor(v) and v.ndim == 2, "Frozen Phi rows/native U,V required")
    n, basis = rows.shape; k, rank = v.shape
    _require(n > 0 and k > 0 and rank > 0 and u.shape == (n, rank)
             and torch.is_tensor(q) and q.ndim == 2 and q.shape[1] > 0,
             "Source/factor dimensions differ")
    _arithmetic(u.device, evidence)
    factor_dtype = u.dtype
    _require(factor_dtype == torch.float32 or factor_dtype == torch.float64
             and u.device.type == "cpu" and rows.owned_CPU_toy,
             "Native factors are FP32; FP64 factors are allowed ONLY for owning CPU toy mathematics")
    policy = "owned_CPU_toy_FP64_math_only" if factor_dtype == torch.float64 else "original_native_FP32_factors"
    if "factor_policy" in evidence:
        _require(evidence["factor_policy"] == policy, "Factor dtype domain changed within evidence")
    else:
        evidence["factor_policy"] = policy
    _tensor(u, (n, rank), factor_dtype, u.device); _tensor(v, (k, rank), factor_dtype, u.device)
    _tensor(assignment, (n,), torch.int64, u.device)
    _tensor(q, (n, q.shape[1]), torch.float64, u.device)
    _require(not assignment.requires_grad and not q.requires_grad and bool((q >= 0).all())
             and bool((q.sum(1) > 0).all()) and bool((q.sum(0) > 0).all())
             and int(assignment.min()) >= 0 and int(assignment.max()) == k - 1,
             "Raw frozen Q support or native cell ordering changed")
    _require(type(mixing) in (float, int) and not isinstance(mixing, bool) and mixing == .05,
             "Original positive mixing .05 required")
    # A conservative known-array tile envelope, not a claim of allocator peak.
    t = min(chunk, n); classes = q.shape[1]
    item = u.element_size()
    bound = t * (basis * 8 + (1 + basis + classes) * 8 + k * (item + 8 + 8 + 8 + item) + rank * item)
    _require(bound < rows._limit, "Complete row material/probability/cotangent tile payload exceeds ceiling")
    evidence["known_tile_array_upper_bound_bytes"] = bound
    evidence["tile_bound_excludes"] = "persistent source Q/U/V/assignment, K moments/cotangent/heads and allocator/BLAS workspace"
    return n, k, rank, basis


class TiledKernelMeanMoments(torch.autograd.Function):
    """First-order original native LowRank arithmetic with complete Phi rows."""
    @staticmethod
    def forward(ctx, u, v, assignment, rows, q, mixing, row_chunk, evidence, stop):
        started = time.monotonic(); _evidence(evidence); _count(evidence, "moment_forward_attempts")
        evidence["last_returned_progress"] = dict(operation="moment_forward", moment_rows=0)
        result = None; completed = 0
        try:
            _guard(stop, started)
            n, k, _, basis = _inputs(u, v, assignment, rows, q, mixing, row_chunk, evidence)
            rows.assert_immutable(evidence=evidence, stop=stop)
            result = q.new_zeros(k, 1 + basis + q.shape[1])
            for start in range(0, n, row_chunk):
                _guard(stop, started); end = min(start + row_chunk, n)
                phi = rows.tile(start, end, u.device, evidence=evidence, stop=stop)
                material = _call(evidence, "forward_complete_material", make_material, phi, q[start:end])
                _guard(stop, started)
                probability = _call(evidence, "forward_native_probability", lambda:
                    logit_block(u[start:end], v, assignment[start:end], mixing).to(material.dtype).softmax(1))
                _guard(stop, started)
                _call(evidence, "forward_original_row_accumulation", lambda:
                    result.add_(probability.T @ material / n))
                evidence["last_returned_progress"]["moment_rows"] = end
                _guard(stop, started)
                _require(bool(torch.isfinite(result).all()), "Nonfinite partial moment accumulation")
                completed = end; _count(evidence, "forward_tiles_completed")
            _guard(stop, started); rows.assert_immutable(evidence=evidence, stop=stop)
            ctx.save_for_backward(u, v, assignment, q)
            ctx.rows, ctx.mixing, ctx.chunk, ctx.evidence, ctx.stop = rows, mixing, row_chunk, evidence, stop
            _count(evidence, "moment_forward_completed")
            return result
        except BaseException as error:
            _failure(evidence, "moment_forward", error, completed, {} if result is None else {"incomplete_moments": result})
            raise

    @staticmethod
    def backward(ctx, gradient):
        started = time.monotonic(); evidence = ctx.evidence; _count(evidence, "moment_backward_attempts")
        evidence["last_returned_progress"] = dict(operation="moment_backward", U_gradient_rows=0, V_gradient_rows=0)
        du, dv, completed = None, None, 0
        try:
            u, v, assignment, q = ctx.saved_tensors
            _require(not torch.is_grad_enabled(), "Only the declared first-order fixed cotangent VJP is supported")
            _guard(ctx.stop, started)
            n, k, rank, basis = _inputs(u, v, assignment, ctx.rows, q, ctx.mixing, ctx.chunk, evidence)
            _tensor(gradient, (k, 1 + basis + q.shape[1]), torch.float64, u.device)
            ctx.rows.assert_immutable(evidence=evidence, stop=ctx.stop)
            du, dv = torch.empty_like(u), torch.zeros_like(v); scale = math.sqrt(rank)
            for start in range(0, n, ctx.chunk):
                _guard(ctx.stop, started); end = min(start + ctx.chunk, n)
                phi = ctx.rows.tile(start, end, u.device, evidence=evidence, stop=ctx.stop)
                material = _call(evidence, "backward_complete_material", make_material, phi, q[start:end])
                _guard(ctx.stop, started)
                probability = _call(evidence, "backward_native_probability", lambda:
                    logit_block(u[start:end], v, assignment[start:end], ctx.mixing).to(material.dtype).softmax(1))
                _guard(ctx.stop, started)
                direction = _call(evidence, "backward_whole_FP64_direction", lambda: material @ gradient.T / n)
                _guard(ctx.stop, started)
                # One WHOLE row-logit factor-dtype cast before rank/GEMMs.
                # Native mode is FP32; owning CPU mathematical mode is FP64.
                block = _call(evidence, "backward_whole_native_cotangent_cast", lambda:
                    (probability * (direction - (probability * direction).sum(1, keepdim=True))).to(u.dtype) / scale)
                _guard(ctx.stop, started)
                _call(evidence, "backward_original_U_GEMM", lambda: du[start:end].copy_(block @ v))
                evidence["last_returned_progress"]["U_gradient_rows"] = end
                _guard(ctx.stop, started)
                _call(evidence, "backward_original_V_GEMM_accumulation", lambda: dv.add_(block.T @ u[start:end]))
                evidence["last_returned_progress"]["V_gradient_rows"] = end
                _guard(ctx.stop, started)
                _require(bool(torch.isfinite(du[start:end]).all()) and bool(torch.isfinite(dv).all()),
                         "Nonfinite native factor pullback")
                completed = end; _count(evidence, "backward_tiles_completed")
            _guard(ctx.stop, started); ctx.rows.assert_immutable(evidence=evidence, stop=ctx.stop)
            _count(evidence, "moment_backward_completed")
            return du, dv, None, None, None, None, None, None, None
        except BaseException as error:
            returned_u = evidence["last_returned_progress"]["U_gradient_rows"]
            arrays = {} if du is None else {"incomplete_grad_U_prefix": du[:returned_u], "incomplete_grad_V": dv}
            _failure(evidence, "moment_backward", error, completed, arrays)
            raise


@torch.no_grad()
def bounded_outer_gradient(rows, q, theta, row_chunk, evidence, stop=lambda: False):
    """Original streamed raw source CE/rhs; bias-last and general-Q row masses."""
    started = time.monotonic(); _evidence(evidence); _count(evidence, "outer_CE_attempts")
    evidence["last_returned_progress"] = dict(operation="outer_CE", CE_rows=0, rhs_rows=0)
    value, gradient, completed = None, None, 0
    try:
        _guard(stop, started); _integer(row_chunk, 1)
        _require(isinstance(rows, FrozenPhiRows) and torch.is_tensor(theta) and theta.ndim == 2,
                 "Frozen source rows and FP64 bias-inclusive theta required")
        n, basis = rows.shape; classes = theta.shape[0]
        _arithmetic(theta.device, evidence)
        _tensor(theta, (classes, basis + 1), torch.float64, theta.device)
        _tensor(q, (n, classes), torch.float64, theta.device)
        t = min(row_chunk, n)
        bound = t * (basis * 8 + (basis + 1) * 8 + classes * 32)
        _require(bound < rows._limit, "Outer source tile payload exceeds the declared ceiling")
        evidence["known_outer_tile_array_upper_bound_bytes"] = bound
        _require(not q.requires_grad and bool((q >= 0).all()) and bool((q.sum(1) > 0).all()),
                 "Raw source Q support changed")
        rows.assert_immutable(evidence=evidence, stop=stop)
        value, gradient = theta.new_zeros(()), torch.zeros_like(theta)
        for start in range(0, n, row_chunk):
            _guard(stop, started); end = min(start + row_chunk, n)
            phi = rows.tile(start, end, theta.device, evidence=evidence, stop=stop)
            x = _call(evidence, "outer_bias_last_augmentation", augmented, phi)
            _guard(stop, started)
            target = q[start:end]
            lp = _call(evidence, "outer_log_softmax", lambda: (x @ theta.T).log_softmax(1))
            _guard(stop, started)
            _call(evidence, "outer_original_CE_accumulation", lambda: value.sub_((target * lp).sum() / n))
            evidence["last_returned_progress"]["CE_rows"] = end
            _guard(stop, started)
            error = target.sum(1, keepdim=True) * lp.exp() - target
            _call(evidence, "outer_original_rhs_accumulation", lambda: gradient.add_(error.T @ x / n))
            evidence["last_returned_progress"]["rhs_rows"] = end
            _guard(stop, started)
            _require(bool(torch.isfinite(value)) and bool(torch.isfinite(gradient).all()), "Nonfinite source CE/rhs")
            completed = end; _count(evidence, "outer_tiles_completed")
        _guard(stop, started); rows.assert_immutable(evidence=evidence, stop=stop)
        result = float(value)
        _require(math.isfinite(result), "Nonfinite completed source CE")
        _count(evidence, "outer_CE_completed")
        return result, gradient
    except BaseException as error:
        arrays = {} if value is None else {"incomplete_raw_CE": value, "incomplete_outer_rhs": gradient}
        _failure(evidence, "outer_CE", error, completed, arrays)
        raise
