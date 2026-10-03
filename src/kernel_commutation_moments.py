"""Activated-only raw-H kernel commutation moments with immutable native origin."""
import hashlib
import json
import math
from pathlib import Path

import torch

from src.io import array_digest, cpu_state
from src.low_rank_assignment import logit_block
from src.shared_features import _tensor_identity
from src.teacher import EPS, get_kernel_values

SCHEMA = 1
MODE = "normalized_kernel_commutation_v1"
POLICY = dict(schema=SCHEMA, mode=MODE,
              objective="teacher_CE_over_CE0_plus_kernel_gap_over_G0",
              kernel="relu", cell_weighting="uniform", coefficient=1,
              both_P_paths=True, raw_saved_H=True, scale_floor=False,
              probability="original_native_logits_to_FP64_softmax",
              separate_base_H_Phi_products=True, single_native_backward_cast=True,
              relu_domain="original_EPS_norm_and_first_and_next_cosine_strict_interior")


def _seal(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _digest(value):
    return array_digest(value.detach().cpu().numpy())


def _matrix(value, shape, *, dtype=torch.float64, device=None, detached=True):
    if (not torch.is_tensor(value) or value.ndim != 2 or min(value.shape) < 1
            or value.shape != tuple(shape) or value.dtype != dtype
            or (device is not None and value.device != device)
            or (detached and value.requires_grad) or not bool(torch.isfinite(value).all())):
        raise ValueError("Kernel commutation tensor shape/dtype/device/finiteness differs")
    return value


def _factors(values, nodes, cells, rank, *, detached=True):
    if not isinstance(values, (list, tuple)) or len(values) != 2:
        raise ValueError("Kernel commutation needs exactly original U/V")
    for value, shape in zip(values, ((nodes, rank), (cells, rank)), strict=True):
        _matrix(value, shape, dtype=torch.float32, detached=detached)
    return [_digest(value) for value in values]


def _positive(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError("Kernel commutation CE0/G0 must be finite positive scalars, without floors")
    return value


def _files(mapping):
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("Kernel commutation immutable file pins are missing")
    for name, expected in mapping.items():
        if not isinstance(name, str) or not isinstance(expected, str) or len(expected) != 64:
            raise ValueError("Kernel commutation file pin is malformed")
        digest = hashlib.sha256()
        with Path(name).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError("Kernel commutation immutable source/cache bytes changed")


class JointKernelMoments(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, v, assignment, material, mixing, chunk_size, physical_H, source_phi):
        values = (u, v, material, physical_H, source_phi)
        if (any(not torch.is_tensor(x) or x.ndim != 2 or min(x.shape) < 1 for x in values)
                or u.dtype not in (torch.float32, torch.float64) or v.dtype != u.dtype
                or u.shape[1] != v.shape[1] or u.shape[1] > min(len(u), len(v))
                or any(x.device != u.device for x in values)
                or any(x.dtype != torch.float64 or x.requires_grad for x in values[2:])
                or any(len(x) != len(u) for x in values[2:])
                or not torch.is_tensor(assignment) or assignment.dtype != torch.int64
                or assignment.shape != (len(u),) or assignment.device != u.device
                or type(chunk_size) is not int or chunk_size < 1 or not 0 < mixing < 1):
            raise ValueError("Joint kernel moments require matching native factors and detached FP64 inputs")
        ctx.save_for_backward(u, v, assignment, material, physical_H, source_phi)
        ctx.mixing, ctx.chunk_size = mixing, chunk_size
        base = material.new_zeros(len(v), material.shape[1])
        hsum = physical_H.new_zeros(len(v), physical_H.shape[1])
        phisum = source_phi.new_zeros(len(v), source_phi.shape[1])
        for start in range(0, len(u), chunk_size):
            end = start + chunk_size
            p = logit_block(u[start:end], v, assignment[start:end], mixing).to(material.dtype).softmax(1)
            base += p.T @ material[start:end] / len(u)
            hsum += p.T @ physical_H[start:end] / len(u)
            phisum += p.T @ source_phi[start:end] / len(u)
        return base, hsum, phisum

    @staticmethod
    def backward(ctx, gb, gh, gp):
        u, v, assignment, material, physical_H, source_phi = ctx.saved_tensors
        gb, gh, gp = [x.new_zeros(len(v), x.shape[1]) if g is None else g
                      for x, g in zip((material, physical_H, source_phi), (gb, gh, gp), strict=True)]
        for x, g in zip((material, physical_H, source_phi), (gb, gh, gp), strict=True):
            _matrix(g, (len(v), x.shape[1]), device=x.device)
        du, dv = torch.empty_like(u), torch.zeros_like(v)
        dm = torch.empty_like(material) if ctx.needs_input_grad[3] else None
        dh = torch.empty_like(physical_H) if ctx.needs_input_grad[6] else None
        dp = torch.empty_like(source_phi) if ctx.needs_input_grad[7] else None
        for start in range(0, len(u), ctx.chunk_size):
            end = start + ctx.chunk_size
            p = logit_block(u[start:end], v, assignment[start:end], ctx.mixing).to(material.dtype).softmax(1)
            direction = material[start:end] @ gb.T / len(u)
            direction = direction + physical_H[start:end] @ gh.T / len(u)
            direction = direction + source_phi[start:end] @ gp.T / len(u)
            block = (p * (direction - (p * direction).sum(1, keepdim=True))).to(u.dtype) / math.sqrt(u.shape[1])
            if not bool(torch.isfinite(block).all()):
                raise FloatingPointError("Joint kernel native cotangent cast overflowed")
            du[start:end] = block @ v
            dv += block.T @ u[start:end]
            if dm is not None:
                dm[start:end] = p @ gb / len(u)
            if dh is not None:
                dh[start:end] = p @ gh / len(u)
            if dp is not None:
                dp[start:end] = p @ gp / len(u)
        if not bool(torch.isfinite(du).all()) or not bool(torch.isfinite(dv).all()):
            raise FloatingPointError("Joint kernel factor gradients are nonfinite")
        return du, dv, None, dm, None, None, dh, dp


def gap_partials(base, hsum, phisum, inputs):
    """Independent local moment graph; no source/factor/head graph is retained."""
    with torch.enable_grad():
        if any(not torch.is_tensor(x) or x.ndim != 2 or min(x.shape) < 1 for x in (base, hsum, phisum)):
            raise ValueError("Kernel commutation requires three nonempty moment matrices")
        b, h, p = [value.detach().clone().requires_grad_() for value in (base, hsum, phisum)]
        if b.dtype != torch.float64 or h.dtype != torch.float64 or p.dtype != torch.float64:
            raise ValueError("Kernel commutation moment cotangents must be FP64")
        if any(not bool(torch.isfinite(value).all()) for value in (b, h, p)) or not bool((b[:, 0] > 0).all()):
            raise FloatingPointError("Kernel commutation moments/mass are outside the finite positive domain")
        centers, means = h / b[:, :1], p / b[:, :1]
        anchors, mapping = inputs["anchors"], inputs["mapping"]
        na, nb = centers.norm(dim=1, keepdim=True), anchors.norm(dim=1).unsqueeze(0)
        if (not bool(torch.isfinite(na).all()) or not bool(torch.isfinite(nb).all())
                or not bool((na > EPS).all()) or not bool((nb > EPS).all())):
            raise FloatingPointError("Kernel commutation ReLU norms reach the clamp boundary")
        cosine = (centers @ anchors.T) / (na * nb)
        if not bool((cosine.abs() < 1 - EPS).all()):
            raise FloatingPointError("Kernel commutation ReLU cosine is outside the differentiable interior")
        angle = torch.acos(cosine)
        next_cosine = (torch.sin(angle) + (math.pi - angle) * torch.cos(angle)) / math.pi
        if not bool((next_cosine.abs() < 1 - EPS).all()):
            raise FloatingPointError("Kernel commutation ReLU next cosine enters the original clamp")
        mapped = get_kernel_values(centers, anchors, "relu") @ mapping
        value = (mapped - means).square().sum(1).mean()
        gradients = torch.autograd.grad(value, (b, h, p))
    if not bool(torch.isfinite(value)) or any(not bool(torch.isfinite(g).all()) for g in gradients):
        raise FloatingPointError("Kernel commutation value/cotangent is nonfinite")
    return dict(value=value.detach(), moment_gradient=gradients[0].detach(),
                physical_gradient=gradients[1].detach(), phi_gradient=gradients[2].detach())


def validate_context(z, q, assignment, parameters, mixing, chunk_size, context, inputs,
                     config, steps, resume_state=None, folder=None):
    """Validate immutable inputs/native origin once, before restoring updated U/V."""
    if (not isinstance(context, dict) or set(context) != {"schema", "policy", "source_refs", "input_descriptors", "native_parameter_digests"}
            or type(context["schema"]) is not int or context["schema"] != SCHEMA or _seal(context["policy"]) != _seal(POLICY)
            or not isinstance(inputs, dict) or set(inputs) != {"physical_H", "source_phi", "anchors", "mapping"}
            or type(steps) is not int or steps < 0 or type(chunk_size) is not int or chunk_size < 1 or mixing != .05):
        raise ValueError("Kernel commutation policy/schema/control differs")
    refs = context["source_refs"]
    if (not isinstance(parameters, (list, tuple)) or len(parameters) != 2
            or any(not torch.is_tensor(x) or x.ndim != 2 or min(x.shape) < 1 for x in (z, q, *parameters))
            or any(not torch.is_tensor(x) or x.ndim != 2 or min(x.shape) < 1 for x in inputs.values())):
        raise ValueError("Kernel commutation source/native matrices must be nonempty")
    nodes, cells, rank = len(z), len(parameters[1]), parameters[0].shape[1]
    _matrix(z, (nodes, z.shape[1]), device=z.device)
    _matrix(q, (nodes, q.shape[1]), device=z.device)
    if (not torch.is_tensor(assignment) or assignment.dtype != torch.int64 or assignment.requires_grad
            or assignment.shape != (nodes,) or assignment.device != z.device
            or not bool(((assignment >= 0) & (assignment < cells)).all())):
        raise ValueError("Kernel commutation original hard cell order differs")
    dimensions, basis = inputs["physical_H"].shape[1], len(inputs["anchors"])
    for key, shape in {"physical_H": (nodes, dimensions), "source_phi": (nodes, basis),
                       "anchors": (basis, dimensions), "mapping": (basis, basis)}.items():
        _matrix(inputs[key], shape, device=z.device)
    descriptors = {key: _tensor_identity(value) for key, value in inputs.items()}
    data = array_digest(z.detach().cpu().numpy(), q.detach().cpu().numpy(), assignment.cpu().numpy())
    if (_seal(context["input_descriptors"]) != _seal(descriptors) or not isinstance(refs, dict)
            or refs.get("data_digest") != data or config.get("data_digest") != data
            or refs.get("factor_seed") != config.get("factor_seed") or type(refs.get("factor_seed")) is not int
            or type(config.get("factor_seed")) is not int or type(config.get("assignment_rank")) is not int
            or type(refs.get("rank")) is not int or refs.get("rank") != rank or config.get("assignment_rank") != rank
            or refs.get("mixing") != mixing or refs.get("chunk_size") != chunk_size
            or type(refs.get("chunk_size")) is not int or type(config.get("chunk_size")) is not int
            or config.get("mixing") != mixing or config.get("chunk_size") != chunk_size
            or any(type(refs.get(key)) is not int or refs[key] != expected for key, expected in
                   {"nodes": nodes, "cells": cells, "physical_dimensions": dimensions, "classes": q.shape[1]}.items())
            or z.shape[1] != dimensions
            or refs.get("device") != str(z.device) or rank < 1 or rank > min(nodes, cells)):
        raise ValueError("Kernel commutation source/native shape/seed/config binding differs")
    assets = refs.get("asset_paths")
    pins = refs.get("files_sha256")
    if (not isinstance(assets, dict) or set(assets) != {"H", "map", "Phi", "Phi_metadata"}
            or not isinstance(pins, dict) or any(path not in pins for path in assets.values())):
        raise ValueError("Kernel commutation requires pinned existing H/map/Phi/sidecar")
    _files(pins)
    source = refs.get("current_source")
    if (not isinstance(source, dict) or not isinstance(source.get("git_head"), str) or not source["git_head"]
            or not isinstance(source.get("source_digest"), str) or not source["source_digest"]
            or not isinstance(source.get("files"), dict)):
        raise ValueError("Kernel commutation current source binding is missing")
    _files(source["files"])
    actual = _factors(parameters, nodes, cells, rank, detached=False)
    if (actual != context["native_parameter_digests"] or bool(parameters[0].ne(0).any())
            or any(value.device != z.device for value in parameters)):
        raise ValueError("Kernel commutation original zeroU/GaussianV origin differs")
    initial = cpu_state(parameters)
    if resume_state is not None:
        validate_resume(resume_state, context, config, steps, nodes, cells, rank, z.shape[1], q.shape[1])
    elif folder is not None and any((Path(folder) / "checkpoints").glob("step_*.pt")):
        raise ValueError("Kernel commutation cached checkpoints require a verified active resume")
    return initial


def attach_snapshot(snapshot, context, config, parameters, hsum, phisum, CE0, G0, G, objective):
    _positive(CE0); _positive(G0)
    if (type(G) not in (int, float) or not math.isfinite(G) or G < 0 or type(objective) not in (int, float)
            or type(snapshot.get("teacher_ce")) not in (int, float) or not math.isfinite(snapshot["teacher_ce"])
            or objective != snapshot["teacher_ce"] / CE0 + G / G0):
        raise ValueError("Kernel commutation checkpoint normalized objective differs")
    snapshot.update(kernel_commutation_context=context, kernel_commutation_config=config,
                    kernel_commutation_parameters=cpu_state(parameters),
                    kernel_commutation_H_moments=cpu_state(hsum), kernel_commutation_phi_moments=cpu_state(phisum),
                    kernel_commutation_CE0=CE0, kernel_commutation_G0=G0,
                    kernel_commutation_G=G, objective=objective)
    return snapshot


def attach_resume(state, context, initial_parameters, CE0, G0):
    _positive(CE0); _positive(G0)
    zero = state["snapshots"][0]
    state.update(kernel_commutation_context=context, kernel_commutation_initial_parameters=cpu_state(initial_parameters),
                 kernel_commutation_CE0=CE0, kernel_commutation_G0=G0,
                 kernel_commutation_initial_H_moments=cpu_state(zero["kernel_commutation_H_moments"]),
                 kernel_commutation_initial_phi_moments=cpu_state(zero["kernel_commutation_phi_moments"]))
    return state


def validate_resume(saved, context, config, steps, nodes, cells, rank, dimension, classes):
    if (not isinstance(saved, dict) or type(saved.get("step")) is not int or not 0 <= saved["step"] <= steps
            or _seal(saved.get("config")) != _seal(config) or _seal(saved.get("kernel_commutation_context")) != _seal(context)
            or _factors(saved.get("kernel_commutation_initial_parameters"), nodes, cells, rank) != context["native_parameter_digests"]):
        raise ValueError("Kernel commutation resume lost its native origin/config/context")
    CE0, G0 = _positive(saved.get("kernel_commutation_CE0")), _positive(saved.get("kernel_commutation_G0"))
    snapshots, end = saved.get("snapshots"), saved["step"]
    if not isinstance(snapshots, dict) or 0 not in snapshots or end not in snapshots:
        raise ValueError("Kernel commutation resume lacks initial/terminal moments")
    hdim = context["input_descriptors"]["physical_H"]["shape"][1]
    phidim = context["input_descriptors"]["source_phi"]["shape"][1]
    for step, snapshot in snapshots.items():
        if (type(step) is not int or not 0 <= step <= end or snapshot.get("step") != step
                or _seal(snapshot.get("kernel_commutation_context")) != _seal(context) or _seal(snapshot.get("kernel_commutation_config")) != _seal(config)
                or snapshot.get("kernel_commutation_CE0") != CE0 or snapshot.get("kernel_commutation_G0") != G0):
            raise ValueError("Kernel commutation checkpoint binding differs")
        _positive(snapshot.get("kernel_commutation_CE0")); _positive(snapshot.get("kernel_commutation_G0"))
        _factors(snapshot.get("kernel_commutation_parameters"), nodes, cells, rank)
        for key, shape in {"moments": (cells, 1 + dimension + classes),
                           "kernel_commutation_H_moments": (cells, hdim),
                           "kernel_commutation_phi_moments": (cells, phidim),
                           "theta": (classes, dimension + 1)}.items():
            _matrix(snapshot.get(key), shape)
        if not bool((snapshot["moments"][:, 0] > 0).all()):
            raise ValueError("Kernel commutation checkpoint mass is not positive")
        G, CE = snapshot.get("kernel_commutation_G"), snapshot.get("teacher_ce")
        if (type(G) not in (int, float) or not math.isfinite(G) or G < 0
                or type(CE) not in (int, float) or not math.isfinite(CE)
                or type(snapshot.get("objective")) not in (int, float)
                or snapshot.get("objective") != CE / CE0 + G / G0):
            raise ValueError("Kernel commutation checkpoint objective differs")
    zero, terminal = snapshots[0], snapshots[end]
    if (_factors(zero["kernel_commutation_parameters"], nodes, cells, rank) != context["native_parameter_digests"]
            or zero["teacher_ce"] != CE0 or zero["kernel_commutation_G"] != G0
            or saved.get("scale") != CE0
            or _tensor_identity(saved["initial_moments"]) != _tensor_identity(zero["moments"])
            or _tensor_identity(saved["kernel_commutation_initial_H_moments"]) != _tensor_identity(zero["kernel_commutation_H_moments"])
            or _tensor_identity(saved["kernel_commutation_initial_phi_moments"]) != _tensor_identity(zero["kernel_commutation_phi_moments"])
            or _tensor_identity(saved["theta"]) != _tensor_identity(terminal["theta"])
            or _factors(saved["parameters"], nodes, cells, rank) != _factors(terminal["kernel_commutation_parameters"], nodes, cells, rank)):
        raise ValueError("Kernel commutation resume initial/terminal links differ")
    history = saved.get("history")
    if not isinstance(history, list) or [row.get("step") for row in history] != list(range(end + 1)):
        raise ValueError("Kernel commutation resume history is not a complete typed prefix")
    for row in history:
        G, CE = row.get("kernel_commutation_G"), row.get("teacher_ce")
        if (type(row.get("step")) is not int or type(row.get("J")) not in (int, float) or row.get("J") != CE
                or type(CE) not in (int, float) or not math.isfinite(CE)
                or type(G) not in (int, float) or not math.isfinite(G) or G < 0
                or type(row.get("objective")) not in (int, float) or type(row.get("normalized_objective")) not in (int, float)
                or row.get("kernel_commutation_CE0") != CE0 or row.get("kernel_commutation_G0") != G0
                or row.get("objective") != CE / CE0 + G / G0
                or row.get("normalized_objective") != row["objective"]):
            raise ValueError("Kernel commutation history scale/objective binding differs")
        _positive(row.get("kernel_commutation_CE0")); _positive(row.get("kernel_commutation_G0"))
    for step, snapshot in snapshots.items():
        row = history[step]
        if (row["teacher_ce"] != snapshot["teacher_ce"] or row["kernel_commutation_G"] != snapshot["kernel_commutation_G"]
                or row["objective"] != snapshot["objective"]):
            raise ValueError("Kernel commutation checkpoint/history objective links differ")
    best_row = min(history, key=lambda row: row["objective"])
    if (type(saved.get("best_step")) is not int or type(saved.get("best")) not in (int, float) or saved["best_step"] != best_row["step"]
            or saved.get("best") != best_row["objective"]):
        raise ValueError("Kernel commutation saved best does not use the normalized objective")
    _matrix(saved.get("best_moments"), (cells, 1 + dimension + classes))
    _matrix(saved.get("best_theta"), (classes, dimension + 1))
    if best_row["step"] in snapshots:
        best_snapshot = snapshots[best_row["step"]]
        if (_tensor_identity(saved["best_moments"]) != _tensor_identity(best_snapshot["moments"])
                or _tensor_identity(saved["best_theta"]) != _tensor_identity(best_snapshot["theta"])):
            raise ValueError("Kernel commutation saved best moment/head link differs")
