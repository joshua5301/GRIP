"""Pure mathematical prototype: fixed five-step student-SGD moment partials.

No production API, cache, data loading or assignment update. The caller supplies
original source moments/H/X/S/Q and immutable initial parameters. Moment output
is detached to a new leaf: the returned dJ/dM can then be supplied to the
original first-order LowRankMoments backward, without custom double backward.
"""
import math
from collections.abc import Mapping

import torch
import torch.nn.functional as F

STEPS = 5
LEARNING_RATE = .1
WEIGHT_DECAY = .001
INITIAL_SEED = 0


def _stop(stop):
    if not callable(stop):
        raise ValueError("stop must be callable")
    if stop():
        raise InterruptedError("Finite student outer interrupted")


def _float_tensor(value, name, *, shape=None, device=None, dtype=None):
    if (not torch.is_tensor(value) or value.layout != torch.strided
            or value.dtype not in (torch.float32, torch.float64)
            or not bool(torch.isfinite(value).all())
            or (shape is not None and tuple(value.shape) != tuple(shape))
            or (device is not None and value.device != device)
            or (dtype is not None and value.dtype != dtype)):
        raise ValueError(f"Malformed finite tensor: {name}")
    return value


def _precision():
    if torch.is_autocast_enabled() or torch.is_autocast_enabled("cpu"):
        raise ValueError("Finite student prototype requires explicit FP32/FP64; autocast is unsupported")


def geom_uniform_initial(nin, classes, hidden=256, *, dtype=torch.float64, device="cpu"):
    """Frozen local-seed0 CPU fan-in uniform weights[in,out] then bias draws.

    Hidden size is generic for mathematical toy QA; the prospective scientific
    policy fixes hidden256. Exact native evaluator parity is a later QA gate.
    This helper does not mutate the global Torch RNG or construct a model.
    """
    _precision()
    if (any(type(value) is not int or value <= 0 for value in (nin, classes, hidden))
            or dtype not in (torch.float32, torch.float64)):
        raise ValueError("Require positive native integer dimensions and FP32/FP64")
    generator = torch.Generator(device="cpu").manual_seed(INITIAL_SEED)
    result = []
    for inputs, outputs in ((nin, hidden), (hidden, classes)):
        bound = 1 / math.sqrt(inputs)
        weight = torch.empty(inputs, outputs, dtype=dtype, device="cpu")
        weight.uniform_(-bound, bound, generator=generator)
        bias = torch.empty(outputs, dtype=dtype, device="cpu")
        bias.uniform_(-bound, bound, generator=generator)
        result.extend((weight.T.contiguous().to(device).detach(), bias.to(device).detach()))
    return tuple(result)


def _parameters(parameters):
    if not isinstance(parameters, (tuple, list)) or len(parameters) != 4:
        raise ValueError("Require [W1,b1,W2,b2]")
    weight1 = _float_tensor(parameters[0], "W1")
    if weight1.ndim != 2 or min(weight1.shape) <= 0:
        raise ValueError("W1 must be class-major hidden×input")
    hidden, nin = weight1.shape
    weight2 = _float_tensor(parameters[2], "W2", device=weight1.device, dtype=weight1.dtype)
    if weight2.ndim != 2 or weight2.shape[1] != hidden or weight2.shape[0] <= 0:
        raise ValueError("W2 must be class-major classes×hidden")
    _float_tensor(parameters[1], "b1", shape=(hidden,), device=weight1.device, dtype=weight1.dtype)
    _float_tensor(parameters[3], "b2", shape=(weight2.shape[0],), device=weight1.device, dtype=weight1.dtype)
    return nin, weight2.shape[0], weight1.device, weight1.dtype


def _adjacency(adjacency, nodes, device, dtype):
    if (not torch.is_tensor(adjacency) or adjacency.shape != (nodes, nodes)
            or adjacency.device != device or adjacency.dtype != dtype
            or adjacency.layout not in (torch.strided, torch.sparse_coo, torch.sparse_csr)
            or adjacency.requires_grad):
        raise ValueError("Source S must be a frozen dense/COO/CSR square tensor in model precision")
    values = adjacency if adjacency.layout == torch.strided else (adjacency.coalesce().values()
             if adjacency.layout == torch.sparse_coo else adjacency.values())
    if not bool(torch.isfinite(values).all()):
        raise ValueError("Source S contains nonfinite values")


def _propagate(adjacency, value):
    return adjacency @ value if adjacency.layout == torch.strided else torch.sparse.mm(adjacency, value)


def functional_forward(parameters, x, adjacency=None):
    """Two biased ReLU layers, dropout0; actual evaluator linear→S→bias order.

    AdjacencyNone represents identity synthetic training or SGC+MLP serving.
    S is already normalized/frozen: no self-loop addition or normalization.
    """
    _precision()
    nin, _, device, dtype = _parameters(parameters)
    _float_tensor(x, "model input", device=device, dtype=dtype)
    if x.ndim != 2 or x.shape[1] != nin or len(x) == 0:
        raise ValueError("Model input shape differs")
    if adjacency is not None:
        _adjacency(adjacency, len(x), device, dtype)
    weight1, bias1, weight2, bias2 = parameters
    hidden = x @ weight1.T
    if adjacency is not None:
        hidden = _propagate(adjacency, hidden)
    hidden = F.relu(hidden + bias1)
    logits = hidden @ weight2.T
    if adjacency is not None:
        logits = _propagate(adjacency, logits)
    logits = logits + bias2
    _float_tensor(logits, "student logits")
    return F.log_softmax(logits, dim=1)


def _probabilities(value, name, *, tolerance=None):
    _float_tensor(value, name)
    if value.ndim != 2 or min(value.shape) <= 0 or bool((value.detach() < 0).any()):
        raise ValueError(f"{name} must be a nonnegative full-class probability matrix")
    if tolerance is None:
        tolerance = 32 * torch.finfo(value.dtype).eps
    if not bool((value.detach().sum(1)-1).abs().max() <= tolerance):
        raise ValueError(f"{name} rows must sum to1; no renormalization is performed")


def decode_rms_moments(moments, dimension, transform):
    """Original [mass,z-moment,Q-moment] → Hbar/Qbar with differentiable m.

    No feature/target detachment, mass weighting or renormalization. Ambient
    moment partials require only positive denominators, not a mass-sum guard;
    the caller's original LowRankMoments supplies the normalized material.
    """
    _float_tensor(moments, "moments")
    if (type(dimension) is not int or dimension <= 0 or moments.ndim != 2
            or len(moments) == 0 or moments.shape[1] <= dimension+1):
        raise ValueError("Moment shape/dimension differs")
    if (not isinstance(transform, Mapping) or transform.get("kind") != "rms"
            or transform.get("matrix") is not None):
        raise ValueError("Require the frozen original RMS inverse transform")
    center = _float_tensor(transform.get("center"), "RMS center", shape=(dimension,), device=moments.device)
    output_center = _float_tensor(transform.get("output_center"), "RMS output center", shape=(dimension,), device=moments.device)
    scale = _float_tensor(transform.get("scale"), "RMS scale", shape=(), device=moments.device)
    if any(value.requires_grad for value in (center, output_center, scale)) or float(scale) <= 0:
        raise ValueError("RMS inverse buffers must be frozen, with positive scale")
    mass = moments[:, 0]
    if not bool((mass.detach() > 0).all()):
        raise ValueError("Every moment mass must be strictly positive")
    zbar = moments[:, 1:dimension+1] / mass[:, None]
    qbar = moments[:, dimension+1:] / mass[:, None]
    hbar = zbar * scale.to(moments) + output_center.to(moments) + center.to(moments)
    _float_tensor(hbar, "source-derived Hbar")
    _probabilities(qbar, "source-derived Qbar", tolerance=1e-12 if qbar.dtype == torch.float64 else None)
    return hbar, qbar, mass


def _snapshot(parameters):
    return tuple(parameter.detach().clone() for parameter in parameters)


@torch.enable_grad()
def finite_student_outer_partials(moments, transform, initial_parameters, source_x,
                                  source_adjacency, source_h, source_q, *, stop=lambda: False):
    """Shared5-step plainSGD trace, two teacher-CE outer losses and dJ/dM.

    LR.1, decay.001 on ALL4parameters, uniform softCE1/K, no warm start,
    dropout, Adam state, source GT or validation labels. Internal learning has
    higher derivatives only through functional Torch SGD. The original custom
    moment output is detached; neither its backward nor graph is unrolled.
    """
    _stop(stop)
    _precision()
    nin, classes, device, dtype = _parameters(initial_parameters)
    if any(parameter.requires_grad for parameter in initial_parameters):
        raise ValueError("Initial parameters must be immutable frozen tensors")
    _float_tensor(moments, "moments", device=device)
    _float_tensor(source_x, "original source X", device=device, dtype=dtype)
    _float_tensor(source_h, "frozen source H", shape=source_x.shape, device=device, dtype=dtype)
    if source_x.ndim != 2 or source_x.shape[1] != nin or len(source_x) == 0:
        raise ValueError("Source X/H must share N×input shape")
    _float_tensor(source_q, "frozen teacher Q", shape=(len(source_x), classes), device=device)
    _probabilities(source_q, "frozen teacher Q")
    _adjacency(source_adjacency, len(source_x), device, dtype)
    if device.type == "cuda" and torch.backends.cuda.matmul.allow_tf32:
        raise ValueError("Native source/model matrix products require TF32disabled")
    leaf = moments.detach().clone().requires_grad_(True)
    hbar, qbar, mass = decode_rms_moments(leaf, nin, transform)
    if qbar.shape[1] != classes:
        raise ValueError("Moment Q classes differ from frozen student/source Q")
    # Exactly the future actual FP32 student input conversion, when dtypeFP32.
    # The original leaf/material precision is retained in the returned partial.
    inner_x, inner_q = hbar.to(dtype), qbar.to(dtype)
    parameters = tuple(parameter.detach().clone().requires_grad_(True) for parameter in initial_parameters)
    original = _snapshot(initial_parameters)
    states, trace = [_snapshot(parameters)], []
    for step in range(STEPS):
        _stop(stop)
        log_probability = functional_forward(parameters, inner_x)
        ce = -(inner_q * log_probability).sum(1).mean()
        _float_tensor(ce, "inner uniform softCE", shape=())
        gradient = torch.autograd.grad(ce, parameters, create_graph=True)
        for value in gradient:
            _float_tensor(value, "inner student gradient")
        decay = .5 * WEIGHT_DECAY * sum(parameter.square().sum() for parameter in parameters)
        trace.append(dict(step=step, uniform_soft_ce=ce.detach().clone(), weight_decay_objective=decay.detach().clone(),
                          regularized_objective=(ce+decay).detach().clone()))
        parameters = tuple(parameter - LEARNING_RATE * (direction + WEIGHT_DECAY * parameter)
                           for parameter, direction in zip(parameters, gradient, strict=True))
        for value in parameters:
            _float_tensor(value, "updated functional parameter")
        states.append(_snapshot(parameters))
        _stop(stop)
    # H and normalized S are independent original frozen buffers. Historical
    # H uses SciPy propagation and S uses PyG normalization; no equality/refit.
    outer_inputs = (("sgc_mlp", source_h.detach(), None), ("gcn", source_x.detach(), source_adjacency.detach()))
    losses, cotangents = {}, {}
    frozen_q = source_q.detach()  # Preserve the original teacher-Q precision.
    for index, (name, x, adjacency) in enumerate(outer_inputs):
        _stop(stop)
        log_probability = functional_forward(parameters, x, adjacency)
        loss = -(frozen_q * log_probability).sum(1).mean()
        _float_tensor(loss, "source teacher outer CE", shape=())
        cotangent, = torch.autograd.grad(loss, leaf, retain_graph=index == 0)
        _float_tensor(cotangent, "moment partial", shape=leaf.shape)
        losses[name], cotangents[name] = loss.detach().clone(), cotangent.detach().clone()
        _stop(stop)
    if any(not torch.equal(before, after) for before, after in zip(original, initial_parameters, strict=True)):
        raise RuntimeError("Immutable initial student state changed")
    return dict(outer_losses=losses, moment_gradients=cotangents, inner_trace=tuple(trace),
                inner_parameter_states=tuple(states), adapted_parameters=_snapshot(parameters),
                Hc=hbar.detach().clone(), Qc=qbar.detach().clone(), mass=mass.detach().clone(),
                policy=dict(steps=STEPS, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, decay_all_parameters=True,
                            optimizer="functional full-batch plain SGD", inner_weighting="uniform1/K",
                            reset="identical provided immutable state every invocation", factory_initial_seed=INITIAL_SEED,
                            seed_provenance="caller must bind factory state; mathematical engine accepts frozen toy states",
                            routes=("sgc_mlp", "gcn"), gcn_order="linear→S→bias", dropout=0.,
                            warm_state=False, source_GT_or_validation_labels=False,
                            outer_Q_conversion="retain original frozen teacher dtype",
                            inner_conversion="Hc/Qc to model dtype; dM keeps original moment dtype",
                            probability_validation=dict(policy="derived_Qbar_original_contract_v1",
                                source_Q="32*finfo(source_Q.dtype).eps", derived_FP64_Qbar_absolute_row_sum_tolerance=1e-12,
                                other_derived_dtypes="unchanged32*finfo(dtype).eps",
                                renormalization=False, arithmetic_change=False)))
