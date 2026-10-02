"""Original CSR forwards and explicitly transposed CSR adjoints; no fallback.

Dense isolated autograd leaves preserve the model-dtype CE/VJP boundaries.
No sparse product belongs to any autograd graph. No source or model is updated.
"""
import torch
import torch.nn.functional as F

BLOCKS = ("W1", "b1", "W2", "b2")
RHO = 1e-3


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _dense(value, name, *, dtype=None, shape=None):
    _require(isinstance(value, torch.Tensor) and value.device.type in ("cpu", "cuda")
             and value.layout == torch.strided and value.dtype in (torch.float32, torch.float64)
             and not value.requires_grad and bool(torch.isfinite(value).all()),
             name + " must be frozen finite float data")
    _require(dtype is None or value.dtype == dtype, name + " dtype differs")
    _require(shape is None or tuple(value.shape) == tuple(shape), name + " shape differs")


def _csr(source):
    _require(isinstance(source, torch.Tensor) and source.device.type in ("cpu", "cuda")
             and source.layout == torch.sparse_csr and source.ndim == 2
             and source.shape[0] == source.shape[1] and source.shape[0] > 0
             and source.dtype in (torch.float32, torch.float64) and not source.requires_grad,
             "Require frozen square CSR")
    crow, columns, values = source.crow_indices(), source.col_indices(), source.values()
    _require(crow.dtype == columns.dtype == torch.int64 and len(crow) == source.shape[0] + 1
             and int(crow[0]) == 0 and int(crow[-1]) == len(columns) == len(values)
             and bool(torch.isfinite(values).all()) and bool((crow[1:] >= crow[:-1]).all())
             and bool(((columns >= 0) & (columns < source.shape[1])).all()), "Malformed CSR buffers")
    rows = torch.repeat_interleave(torch.arange(source.shape[0], device=source.device), crow[1:] - crow[:-1])
    _require(len(rows) == len(columns)
             and bool(((rows[1:] != rows[:-1]) | (columns[1:] > columns[:-1])).all()),
             "CSR coordinates must be sorted and unique")


def _invoke(evidence, key, function, *args, **kwargs):
    if evidence is None:
        return function(*args, **kwargs)
    evidence["_guard"]()
    evidence["operation_attempts"][key] = evidence["operation_attempts"].get(key, 0) + 1
    result = function(*args, **kwargs)
    evidence["counts"][key] = evidence["counts"].get(key, 0) + 1
    return result


def _begin(evidence, key, amount=1):
    if evidence is not None:
        evidence["_guard"]()
        evidence["operation_attempts"][key] = evidence["operation_attempts"].get(key, 0) + amount


def _end(evidence, key, amount=1):
    if evidence is not None:
        evidence["counts"][key] = evidence["counts"].get(key, 0) + amount


@torch.no_grad()
def transpose_csr(source, evidence=None):
    """Two host integer copies, three index uploads and one bit-value gather."""
    _csr(source)
    crow = _invoke(evidence, "transpose_bulk_index_host_copies", lambda: source.crow_indices().detach().cpu()).tolist()
    columns = _invoke(evidence, "transpose_bulk_index_host_copies", lambda: source.col_indices().detach().cpu()).tolist()
    original_rows = [row for row in range(source.shape[0]) for _ in range(crow[row], crow[row + 1])]
    entries = [(columns[i], row, i) for row in range(source.shape[0])
               for i in range(crow[row], crow[row + 1])]
    entries.sort(key=lambda entry: (entry[0], entry[1]))
    transposed_crow = [0]
    cursor = 0
    for row in range(source.shape[0]):
        while cursor < len(entries) and entries[cursor][0] == row:
            cursor += 1
        transposed_crow.append(cursor)
    _require(sorted(entry[2] for entry in entries) == list(range(len(columns)))
             and all((entry[0], entry[1]) == (columns[entry[2]], original_rows[entry[2]]) for entry in entries),
             "Transpose coordinate bijection differs")
    upload = lambda value: _invoke(evidence, "transpose_GPU_integer_uploads", torch.tensor, value, dtype=torch.int64, device=source.device)
    target_crow = upload(transposed_crow)
    target_columns = upload([entry[1] for entry in entries])
    order = upload([entry[2] for entry in entries])
    gathered = _invoke(evidence, "transpose_GPU_value_gathers", source.values().index_select, 0, order)
    target = torch.sparse_csr_tensor(target_crow, target_columns, gathered, size=source.shape, device=source.device)
    _csr(target)
    bit_dtype = torch.int32 if source.dtype == torch.float32 else torch.int64
    _require(len(target.values()) == len(source.values()) and target.shape == source.shape
             and torch.equal(target.values().view(bit_dtype), gathered.view(bit_dtype)), "Transpose value bits changed")
    return target


@torch.no_grad()
def _product(source, value, *, evidence=None, key=None):
    result = _invoke(evidence, key, torch.sparse.mm, source, value)
    _require(bool(torch.isfinite(result).all()), "Nonfinite CSR product")
    return result


def _inputs(parameters, x, source, q):
    _require(isinstance(parameters, (tuple, list)) and len(parameters) == 4,
             "Require W1,b1,W2,b2")
    _dense(x, "X")
    _require(x.ndim == 2 and min(x.shape) > 0, "Empty/malformed X")
    w1, b1, w2, b2 = parameters
    _require(all(value.device == x.device for value in parameters) and q.device == x.device
             and source.device == x.device, "Input devices differ")
    _dense(w1, "W1", dtype=x.dtype)
    _require(w1.ndim == 2 and w1.shape[1] == x.shape[1] and w1.shape[0] > 0, "W1 shape differs")
    _dense(b1, "b1", dtype=x.dtype, shape=(w1.shape[0],))
    _dense(w2, "W2", dtype=x.dtype)
    _require(w2.ndim == 2 and w2.shape[1] == w1.shape[0] and w2.shape[0] > 0, "W2 shape differs")
    _dense(b2, "b2", dtype=x.dtype, shape=(w2.shape[0],))
    _csr(source)
    _require(source.shape == (len(x), len(x)) and source.dtype == x.dtype, "Source CSR shape/dtype differs")
    _dense(q, "Q", dtype=torch.float64, shape=(len(x), w2.shape[0]))
    _require(bool((q >= 0).all()) and bool((q.sum(1) - 1).abs().max() <= 32 * torch.finfo(q.dtype).eps),
             "Q must satisfy the unchanged strict source probability contract")


def gradient_boundaries(parameters, x, source, q, evidence=None):
    """Compute all four parameter cotangents without sparse autograd backward."""
    _inputs(parameters, x, source, q)
    w1, b1, w2, b2 = parameters
    transposed = _invoke(evidence, "transpose_CSR_constructions", transpose_csr, source, evidence)
    _begin(evidence, "source_GCN_full_forward_passes")
    with torch.no_grad():
        u1 = x @ w1.T
        v1 = _product(source, u1, evidence=evidence, key="original_CSR_full_forward_SpMM_calls")
        a = v1 + b1
        h = F.relu(a)
        u2 = h @ w2.T
        v2 = _product(source, u2, evidence=evidence, key="original_CSR_full_forward_SpMM_calls")
        logits = v2 + b2
        full_logp = F.log_softmax(logits, dim=1)
        full_loss = _invoke(evidence, "source_CE_forward_evaluations", lambda: -(q * full_logp).sum(1).mean())
    _end(evidence, "source_GCN_full_forward_passes")
    leaf_logits = logits.detach().clone().requires_grad_(True)
    logp = F.log_softmax(leaf_logits, dim=1)
    loss = _invoke(evidence, "source_CE_forward_evaluations", lambda: -(q * logp).sum(1).mean())
    _require(torch.equal(full_logp, logp) and torch.equal(full_loss, loss), "No-grad/dense-leaf forward differs")
    d = _invoke(evidence, "dense_logits_CE_VJP_calls", torch.autograd.grad, loss, leaf_logits)[0].detach()
    b = _product(transposed, d, evidence=evidence, key="explicit_transpose_CSR_SpMM_calls")
    leaf_h, leaf_w2 = (value.detach().clone().requires_grad_(True) for value in (h, w2))
    dh, gw2 = _invoke(evidence, "dense_second_linear_VJP_calls", torch.autograd.grad, leaf_h @ leaf_w2.T, (leaf_h, leaf_w2), grad_outputs=b)
    leaf_v2, leaf_b2 = (value.detach().clone().requires_grad_(True) for value in (v2, b2))
    gb2 = _invoke(evidence, "dense_second_bias_VJP_calls", torch.autograd.grad, leaf_v2 + leaf_b2, leaf_b2, grad_outputs=d)[0]
    leaf_v1, leaf_b1 = (value.detach().clone().requires_grad_(True) for value in (v1, b1))
    e, gb1 = _invoke(evidence, "dense_first_ReLU_bias_VJP_calls", torch.autograd.grad, F.relu(leaf_v1 + leaf_b1), (leaf_v1, leaf_b1), grad_outputs=dh.detach())
    c = _product(transposed, e.detach(), evidence=evidence, key="explicit_transpose_CSR_SpMM_calls")
    leaf_w1 = w1.detach().clone().requires_grad_(True)
    gw1 = _invoke(evidence, "dense_first_linear_VJP_calls", torch.autograd.grad, x @ leaf_w1.T, leaf_w1, grad_outputs=c)[0]
    gradients = tuple(value.detach().clone() for value in (gw1, gb1, gw2, gb2))
    _require(bool(torch.isfinite(loss)) and all(bool(torch.isfinite(value).all()) for value in gradients),
             "Nonfinite CE or gradient")
    _begin(evidence, "explicit_gradient_block_outputs", 4)
    _end(evidence, "explicit_gradient_block_outputs", 4)
    boundaries = dict(U1=u1, V1=v1, A=a, mask=a > 0, H=h, U2=u2, V2=v2, logits=logits,
                      logp=logp, full_logp=full_logp, full_CE=full_loss, D=d, B=b, DH=dh, E=e, C=c)
    return dict(loss=loss.detach().clone(), gradients=gradients,
                boundaries={key: value.detach().clone() for key, value in boundaries.items()},
                parameters=tuple(value.detach().clone() for value in parameters), transpose=transposed,
                sparse_forward_products=4, sparse_backward_calls=0, parameter_updates=0)


def target_packet(result, evidence=None):
    """Reject any nonpositive source block norm, without clamp or block removal."""
    gradients = result["gradients"]
    _require(isinstance(gradients, (tuple, list)) and len(gradients) == 4, "Require four gradient blocks")
    for value, parameter in zip(gradients, result["parameters"]):
        _dense(value, "source gradient", dtype=parameter.dtype, shape=parameter.shape)
    norms = tuple(_invoke(evidence, "source_gradient_block_FP64_norms", torch.linalg.vector_norm, value.double().reshape(-1)) for value in gradients)
    _require(all(bool(torch.isfinite(norm)) and float(norm) > 0 for norm in norms),
             "Every source block norm must be finite positive")
    delta = tuple(_invoke(evidence, "source_gradient_block_FP64_delta_values", lambda: RHO * norm) for norm in norms)
    _require(all(bool(torch.isfinite(value)) and float(value) > 0 for value in delta),
             "Source smoothing must be finite positive")
    return dict(blocks=BLOCKS, rho=RHO, gradients=tuple(value.clone() for value in gradients),
                source_norms=norms, delta=delta)
