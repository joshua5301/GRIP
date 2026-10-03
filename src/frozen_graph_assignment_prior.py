"""Immutable binary random-walk diffusion of the actual native soft P0.

Only preparation consumes a graph. Production loads its pinned FP64 log-prior
without recreating graph normalization, P0, source features, or an RNG. The
existing streamed KL operator consumes this fixed target as a constant.
"""
import hashlib
import json
from pathlib import Path

import torch

from src.io import array_digest

SCHEMA = 1
MODE = "binary_rw_diffused_native_P0_v1"
ROW_ATOL = 1e-12
POLICY = dict(coefficient=1.0, mixing=.05, reduction="mean_source_rows_sum_cells",
              normalization="max_teacher_CE_native_P0_1e-12",
              objective="teacher_CE_plus_KL_current_to_frozen_binary_rw_diffused_native_P0",
              graph="binary_undirected_source_support_with_one_unit_loop_per_node",
              diffusion="one_pass_D_inverse_B_times_detached_actual_native_FP64_P0",
              packed_reconstruction_atol=1e-7, packed_reconstruction_rtol=1e-5,
              target_row_atol=ROW_ATOL, probability_floor=False, target_recomputed=False,
              target_gradients=False, inner_loss="uniform_CE", columns="free")
_CONTEXT_KEYS = ("schema", "mode", "dimensions", "source_refs", "native_parameter_digests",
                 "log_probability_digest", "context_digest")
_PACKET_KEYS = set(_CONTEXT_KEYS) | {"log_probability", "policy", "input_descriptors", "diagnostics", "implementation"}


def _json(value):
    try:
        return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError("Graph prior metadata must be finite JSON") from error


def _seal(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _dense(value, shape, dtype, name):
    if (not torch.is_tensor(value) or value.layout != torch.strided or value.shape != shape
            or value.dtype != dtype or value.device.type != "cpu" or value.requires_grad
            or not bool(torch.isfinite(value).all())):
        raise ValueError(f"{name} must be detached finite CPU {dtype} with shape {tuple(shape)}")


def _refs(source_refs):
    if not isinstance(source_refs, dict) or not source_refs:
        raise ValueError("Graph prior requires root-owned pinned source references")
    refs = _json(source_refs)
    if (not _sha(refs.get("data_digest")) or type(refs.get("factor_seed")) is not int
            or not 0 <= refs["factor_seed"] < 2**31
            or isinstance(refs.get("mixing"), bool) or refs.get("mixing") != .05):
        raise ValueError("Graph prior source references require native data_digest/factor_seed/mixing.05")
    return refs


def _dimensions(dimensions):
    if (not isinstance(dimensions, dict) or set(dimensions) != {"nodes", "cells", "rank"}
            or any(type(value) is not int or value < 1 for value in dimensions.values())
            or dimensions["rank"] > min(dimensions["nodes"], dimensions["cells"])):
        raise ValueError("Graph prior requires positive integer nodes/cells/native factor rank")
    return dimensions


def _canonical_csr(adjacency, nodes):
    if (not torch.is_tensor(adjacency) or adjacency.layout != torch.sparse_csr
            or adjacency.ndim != 2 or adjacency.shape != (nodes, nodes)
            or adjacency.device.type != "cpu" or adjacency.requires_grad
            or adjacency.dtype not in (torch.float32, torch.float64)):
        raise ValueError("Require detached CPU FP32/FP64 native packed CSR adjacency")
    crow, col, values = adjacency.crow_indices(), adjacency.col_indices(), adjacency.values()
    nnz = len(values)
    if (crow.shape != (nodes + 1,) or col.shape != (nnz,) or values.shape != (nnz,)
            or crow.dtype not in (torch.int32, torch.int64) or col.dtype != crow.dtype
            or int(crow[0]) != 0 or int(crow[-1]) != nnz or bool((crow[1:] <= crow[:-1]).any())
            or bool((col < 0).any()) or bool((col >= nodes).any())
            or not bool(torch.isfinite(values).all()) or bool((values <= 0).any())):
        raise ValueError("Packed CSR has malformed pointers/indices or nonpositive/nonfinite weights")
    rows = torch.repeat_interleave(torch.arange(nodes), (crow[1:] - crow[:-1]).long())
    if bool(((rows[1:] == rows[:-1]) & (col[1:] <= col[:-1])).any()):
        raise ValueError("Packed CSR must have strictly sorted unique columns within each row")


def _tensor_descriptor(value):
    return dict(shape=list(value.shape), dtype=str(value.dtype), tensor=array_digest(value.numpy()))


def _csr_descriptor(value):
    return dict(shape=list(value.shape), layout=str(value.layout), dtype=str(value.dtype),
                crow=_tensor_descriptor(value.crow_indices()), col=_tensor_descriptor(value.col_indices()),
                values=_tensor_descriptor(value.values()))


def _implementation():
    local = Path(__file__).resolve().parent
    names = ("frozen_graph_assignment_prior.py", "initial_assignment_row_kl.py", "large_quotient_pilot.py", "io.py")
    return dict(torch_version=str(torch.__version__), source_files_sha256={
        name: hashlib.sha256((local / name).read_bytes()).hexdigest() for name in names})


@torch.no_grad()
def build_prior_packet(packed_adj, p0, initial_parameters, source_refs):
    """Prepare one positive frozen logR from pinned native P0 and Gaussian U0/V0."""
    from src.large_quotient_pilot import raw_looped_support

    if not torch.is_tensor(p0) or p0.ndim != 2 or min(p0.shape) < 1:
        raise ValueError("Require nonempty node-by-cell native P0")
    nodes, cells = p0.shape
    _dense(p0, torch.Size((nodes, cells)), torch.float64, "Native P0")
    if bool((p0 <= 0).any()) or float((p0.sum(1) - 1).abs().max()) > ROW_ATOL:
        raise ValueError("Native P0 must be strictly positive with normalized rows")
    if (not isinstance(initial_parameters, (tuple, list)) or len(initial_parameters) != 2
            or not torch.is_tensor(initial_parameters[0]) or initial_parameters[0].ndim != 2):
        raise ValueError("Require the actual two native initial factor tensors")
    u0, v0 = initial_parameters
    rank = u0.shape[1]
    dimensions = _dimensions(dict(nodes=nodes, cells=cells, rank=rank))
    _dense(u0, torch.Size((nodes, rank)), torch.float32, "Native U0")
    _dense(v0, torch.Size((cells, rank)), torch.float32, "Native V0")
    if bool(u0.ne(0).any()):
        raise ValueError("Graph prior requires exactly zero native U0")
    refs = _refs(source_refs)
    _canonical_csr(packed_adj, nodes)
    binary, reconstruction = raw_looped_support(packed_adj, atol=1e-7, rtol=1e-5)
    crow, col = binary.crow_indices().clone(), binary.col_indices().clone()
    degree = (crow[1:] - crow[:-1]).double()
    rows = torch.repeat_interleave(torch.arange(nodes), degree.long())
    rw = torch.sparse_csr_tensor(crow, col, degree.reciprocal()[rows], size=(nodes, nodes), dtype=torch.float64)
    target = torch.sparse.mm(rw, p0)
    if (not bool(torch.isfinite(target).all()) or bool((target <= 0).any())
            or float((target.sum(1) - 1).abs().max()) > ROW_ATOL):
        raise ValueError("Diffused prior must be finite strictly positive with normalized rows")
    log_probability = target.log().detach().contiguous()
    packet = dict(schema=SCHEMA, mode=MODE, dimensions=dimensions, source_refs=refs, policy=dict(POLICY),
        native_parameter_digests=[array_digest(parameter.numpy()) for parameter in (u0, v0)],
        log_probability=log_probability, log_probability_digest=array_digest(log_probability.numpy()),
        input_descriptors=dict(packed_CSR=_csr_descriptor(packed_adj), binary_looped_support=_csr_descriptor(binary),
                               random_walk_CSR=_csr_descriptor(rw), native_P0=_tensor_descriptor(p0)),
        diagnostics=dict(packed_reconstruction=reconstruction,
            source_P0_row_error=float((p0.sum(1) - 1).abs().max()),
            diffused_R_row_error=float((target.sum(1) - 1).abs().max()), R_min=float(target.min()),
            R_minus_P0_max_abs=float((target - p0).abs().max()),
            rows_changed_at_1e12=int(((target - p0).abs().max(1).values > 1e-12).sum()),
            initial_KL_P0_to_R=float((p0 * (p0.log() - log_probability)).sum() / nodes),
            original_cell_mass=p0.sum(0).tolist(), diffused_cell_mass=target.sum(0).tolist()),
        implementation=_implementation())
    packet["context_digest"] = _seal({key: value for key, value in packet.items() if key != "log_probability"})
    return packet


def packet_context(packet):
    """Return exactly the seven JSON identity fields without numerical replay."""
    if not isinstance(packet, dict) or not set(_CONTEXT_KEYS).issubset(packet):
        raise ValueError("Graph prior packet lacks its frozen context")
    return _json({key: packet[key] for key in _CONTEXT_KEYS})


def load_frozen_prior(path, sha256, dimensions, device, expected_context):
    """Validate pinned file/metadata/tensor bytes and clone FP64 logR to device."""
    if not _sha(sha256):
        raise ValueError("Require a pinned lowercase SHA256 graph prior artifact digest")
    dimensions = _dimensions(dimensions)
    with Path(path).open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != sha256:
            raise ValueError("Frozen graph prior artifact bytes changed")
    packet = torch.load(path, map_location="cpu", weights_only=True)
    if (not isinstance(packet, dict) or set(packet) != _PACKET_KEYS or type(packet["schema"]) is not int
            or packet["schema"] != SCHEMA or packet["mode"] != MODE
            or not isinstance(packet["native_parameter_digests"], list)
            or len(packet["native_parameter_digests"]) != 2
            or any(not _sha(pin) for pin in packet["native_parameter_digests"])
            or not _sha(packet["log_probability_digest"])
            or _seal(_json(packet["policy"])) != _seal(POLICY)
            or not isinstance(packet["input_descriptors"], dict)
            or not isinstance(packet["diagnostics"], dict) or not isinstance(packet["implementation"], dict)):
        raise ValueError("Frozen graph prior has an unsupported schema or policy")
    _dimensions(packet["dimensions"])
    _refs(packet["source_refs"])
    metadata = {key: value for key, value in packet.items() if key not in ("log_probability", "context_digest")}
    if packet["context_digest"] != _seal(metadata):
        raise ValueError("Frozen graph prior metadata changed")
    if (_seal(dimensions) != _seal(packet["dimensions"])
            or _seal(_json(expected_context)) != _seal(packet_context(packet))):
        raise ValueError("Frozen graph prior differs from the consumer dimensions or native source context")
    log_probability = packet["log_probability"]
    _dense(log_probability, torch.Size((dimensions["nodes"], dimensions["cells"])), torch.float64, "Frozen logR")
    if array_digest(log_probability.numpy()) != packet["log_probability_digest"]:
        raise ValueError("Frozen graph prior logR tensor changed")
    if float(torch.logsumexp(log_probability, 1).abs().max()) > ROW_ATOL:
        raise ValueError("Frozen graph prior logR rows are not normalized")
    return log_probability.detach().to(device=device).clone()
