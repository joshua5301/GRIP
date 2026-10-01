"""Shared teacher Nyström map and exact implicit CE condensation.

Representatives are averages in SGC input space; the nonlinear map is applied
AFTER averaging. Original-node feature blocks live in a CPU memory map.
"""
import hashlib
import inspect
import json
import marshal
import time
import types
from pathlib import Path

import numpy as np
import torch

from src.io import save_json, save_state
from src.low_rank_assignment import LowRankMoments, initialize_factors
from src.moments import augmented, decode_moments, make_material
from src.soft_ce_partition import head_gradient, solve_head_system, solve_inner_newton_first
from src.teacher import get_kernel_values


class NystromMap:
    def __init__(self, anchors, mapping, kernel="relu"):
        self.anchors, self.mapping, self.kernel = anchors, mapping, kernel

    @classmethod
    def fit(cls, h, basis=3000, seed=0, kernel="relu"):
        generator = torch.Generator(device=h.device).manual_seed(seed)
        indices = torch.randperm(len(h), generator=generator, device=h.device)[:basis] if basis < len(h) else None
        anchors = h[indices].double() if indices is not None else h.double()
        gram = get_kernel_values(anchors, anchors, kernel)
        gram = (gram + gram.T) / 2
        eye = torch.eye(len(anchors), dtype=gram.dtype, device=gram.device)
        chol = torch.linalg.cholesky(gram + 1e-8 * gram.diagonal().mean() * eye)
        return cls(anchors, torch.linalg.solve_triangular(chol, eye, upper=False).T, kernel)

    def __call__(self, h):
        return get_kernel_values(h.double(), self.anchors, self.kernel) @ self.mapping


def _check_stop(stop):
    if stop():
        raise InterruptedError("Nyström input validation interrupted")


def _content_digest(value, chunk=2048, stop=lambda: False, canonical_double=False):
    """Hash rows without materializing a potentially multi-gigabyte memmap."""
    shape = tuple(value.shape)
    dtype = "float64" if canonical_double else str(value.dtype)
    digest = hashlib.sha256(json.dumps(dict(shape=shape, dtype=dtype), sort_keys=True).encode())
    matrix = value if shape else value.reshape(1)
    for start in range(0, len(matrix), chunk):
        _check_stop(stop)
        block = matrix[start:start + chunk]
        if torch.is_tensor(block):
            block = block.detach().cpu()
            # NumPy cannot directly expose bfloat16; retain original dtype in header.
            if canonical_double or block.dtype == torch.bfloat16:
                block = block.double()
            block = block.numpy()
        block = np.ascontiguousarray(block, dtype=np.float64 if canonical_double else None)
        digest.update(memoryview(block).cast("B"))
    return digest.hexdigest()


def _map_token(value, chunk, stop, active=None):
    """Deterministic callable state, including closures such as FeatureTransform."""
    _check_stop(stop)
    active = set() if active is None else active
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value) or isinstance(value, np.ndarray):
        return dict(array=_content_digest(value, chunk, stop))
    if isinstance(value, (torch.dtype, torch.device, Path)):
        return str(value)
    if isinstance(value, types.ModuleType):
        return dict(module=value.__name__)
    if id(value) in active:
        return dict(recursive_type=f"{type(value).__module__}.{type(value).__qualname__}")
    active = active | {id(value)}
    convert = lambda item: _map_token(item, chunk, stop, active)
    if isinstance(value, dict):
        return {str(key): convert(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [convert(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((convert(item) for item in value), key=lambda item: json.dumps(item, sort_keys=True))
    if inspect.ismethod(value):
        return dict(function=convert(value.__func__), owner=convert(value.__self__))
    if inspect.isfunction(value):
        closure = inspect.getclosurevars(value)
        return dict(
            function=f"{value.__module__}.{value.__qualname__}",
            code=hashlib.sha256(marshal.dumps(value.__code__)).hexdigest(),
            defaults=convert(value.__defaults__), kwdefaults=convert(value.__kwdefaults__),
            nonlocals=convert(closure.nonlocals), globals=convert(closure.globals),
        )
    if inspect.isbuiltin(value):
        return dict(builtin=f"{value.__module__}.{value.__qualname__}")
    if isinstance(value, torch.nn.Module):
        return dict(module_type=f"{type(value).__module__}.{type(value).__qualname__}",
                    state=convert(value.state_dict()), attributes=convert(vars(value)),
                    forward=convert(type(value).forward))
    if hasattr(value, "__dict__"):
        result = dict(type=f"{type(value).__module__}.{type(value).__qualname__}", state=convert(vars(value)))
        if callable(value):
            result["call"] = convert(type(value).__call__)
        return result
    raise ValueError(f"Cannot safely fingerprint feature-map state of type {type(value).__qualname__}")


def _cache_identity(h, feature_map, shape, chunk, stop):
    token = _map_token(feature_map, chunk, stop)
    return dict(
        schema=1, h_digest=_content_digest(h, chunk, stop, canonical_double=True),
        map_digest=hashlib.sha256(json.dumps(token, sort_keys=True).encode()).hexdigest(),
        shape=list(shape), dtype="float64",
    )


def _metadata_path(path):
    return Path(path).with_suffix(".meta.json")


def _validate_phi(h, feature_map, phi, identity, chunk, stop):
    """A valid sidecar permits reuse; legacy/untracked arrays need full verification."""
    if tuple(phi.shape) != tuple(identity["shape"]) or np.dtype(phi.dtype) != np.float64:
        raise ValueError("Incompatible feature cache shape or dtype")
    filename = getattr(phi, "filename", None)
    sidecar = _metadata_path(filename) if filename is not None else None
    phi_digest = _content_digest(phi, chunk, stop)
    if sidecar is not None and sidecar.exists():
        try:
            saved = json.loads(sidecar.read_text())
        except (OSError, ValueError) as exc:
            raise ValueError("Invalid feature cache metadata") from exc
        if saved != dict(**identity, phi_digest=phi_digest):
            raise ValueError("Feature cache fingerprint differs in H, map, coordinates or cached content")
        return phi_digest
    # Shape agreement alone cannot identify the input/map of a legacy feature file.
    for start in range(0, len(h), chunk):
        _check_stop(stop)
        expected = feature_map(h[start:start + chunk]).detach().double().cpu().numpy()
        actual = np.asarray(phi[start:start + chunk])
        if not np.isfinite(expected).all() or not np.isfinite(actual).all() or not np.allclose(
            expected, actual, atol=1e-10, rtol=1e-8
        ):
            raise ValueError("Feature cache does not match current H/map; preserve it and use a new cache path")
    if sidecar is not None:
        save_json(dict(**identity, phi_digest=phi_digest), sidecar)
    return phi_digest


@torch.no_grad()
def cache_features(h, feature_map, path, chunk=2048, stop=lambda: False):
    path = Path(path)
    if chunk < 1 or not isinstance(chunk, int):
        raise ValueError("Feature cache chunk must be a positive integer")
    _check_stop(stop)
    width = feature_map(h[:1]).shape[1]
    identity = _cache_identity(h, feature_map, (len(h), width), chunk, stop)
    if path.exists():
        phi = np.load(path, mmap_mode="r")
        _validate_phi(h, feature_map, phi, identity, chunk, stop)
        return phi
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial.npy")
    phi = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float64,
                                  shape=(len(h), width))
    for start in range(0, len(h), chunk):
        if stop():
            raise InterruptedError("Feature preparation interrupted")
        block = feature_map(h[start:start + chunk]).double().cpu().numpy()
        if not np.isfinite(block).all():
            raise ValueError("Nonfinite mapped features cannot be cached")
        phi[start:start + chunk] = block
    phi.flush()
    phi_digest = _content_digest(phi, chunk, stop)
    del phi
    temporary.replace(path)
    save_json(dict(**identity, phi_digest=phi_digest), _metadata_path(path))
    return np.load(path, mmap_mode="r")


def blocks(phi, device, chunk=2048):
    for start in range(0, len(phi), chunk):
        # Copy avoids writing through a read-only NumPy mapping.
        yield start, torch.from_numpy(np.array(phi[start:start + chunk])).to(device)


@torch.no_grad()
def outer_gradient(phi, q, theta, chunk=2048):
    value, gradient = theta.new_zeros(()), torch.zeros_like(theta)
    for start, block in blocks(phi, theta.device, chunk):
        x = augmented(block)
        target = q[start:start + len(block)]
        lp = (x @ theta.T).log_softmax(1)
        value -= (target * lp).sum() / len(phi)
        error = target.sum(1, keepdim=True) * lp.exp() - target
        gradient += error.T @ x / len(phi)
    return float(value), gradient


def fit_streaming_teacher(phi, labels, mask, gamma=0.01, chunk=2048,
                          max_iter=1000, stop=lambda: False, resident_training=False):
    """Same no-bias teacher CE, optionally keeping only training phi resident.

    resident_training=True transfers selected training rows once, in chunks.
    On CUDA it first releases unused allocator cache and permits this route
    only when training bytes plus working allowance fit inside 75% of current
    free device memory. Insufficient memory or allocation OOM falls back to the
    original streamed closure. All-node phi stays on CPU; prediction is streamed.
    Callers still supply training-only labels (zero outside mask) when they need
    a class vocabulary inferred exclusively from the training split.
    """
    if stop():
        raise InterruptedError("Teacher fitting interrupted")
    if chunk < 1 or not isinstance(chunk, int) or mask.shape != (len(phi),) or mask.dtype != torch.bool:
        raise ValueError("Require a positive chunk and a node-aligned boolean training mask")
    n = int(mask.sum())
    if n < 1:
        raise ValueError("Teacher fitting requires at least one training node")
    classes = int(labels.max()) + 1
    weight = torch.zeros(phi.shape[1], classes, dtype=torch.double,
                         device=labels.device, requires_grad=True)
    resident_x, resident_y = None, None
    required = n * phi.shape[1] * weight.element_size()
    # Covers full training logits/errors, LBFGS history, matmul and transfer work.
    allowance = max(256 * 1024**2, 3 * n * classes * weight.element_size()
                    + 232 * weight.numel() * weight.element_size()
                    + 2 * chunk * phi.shape[1] * weight.element_size())
    route = dict(requested="resident" if resident_training else "streaming", actual="streaming",
                 train_rows=n, feature_width=phi.shape[1], resident_bytes=required,
                 working_allowance_bytes=allowance)
    if resident_training:
        allowed = True
        if weight.device.type == "cuda":
            torch.cuda.empty_cache()
            free, _ = torch.cuda.mem_get_info(weight.device)
            route["free_gpu_bytes"] = int(free)
            allowed = required + allowance <= 0.75 * free
            if not allowed:
                route["fallback"] = "free_memory_budget"
        if allowed:
            try:
                if stop():
                    raise InterruptedError("Resident teacher preparation interrupted")
                resident_x = torch.empty(n, phi.shape[1], dtype=weight.dtype, device=weight.device)
                cpu_mask = mask.detach().cpu().numpy()
                cursor = 0
                for start in range(0, len(phi), chunk):
                    if stop():
                        raise InterruptedError("Resident teacher transfer interrupted")
                    chosen = cpu_mask[start:start + chunk]
                    count = int(chosen.sum())
                    if not count:
                        continue
                    # Select on CPU before transferring; excluded rows never go resident.
                    block = np.array(phi[start:start + chunk][chosen], copy=True)
                    resident_x[cursor:cursor + count].copy_(torch.from_numpy(block).to(weight))
                    cursor += count
                resident_y = labels[mask]
                route["actual"] = "resident"
            except torch.cuda.OutOfMemoryError:
                resident_x, resident_y = None, None
                torch.cuda.empty_cache()
                route["fallback"] = "allocation_oom"
    fit_streaming_teacher.last_route = dict(route)
    print("TEACHER_FEATURE_ROUTE", json.dumps(route), flush=True)
    optimizer = torch.optim.LBFGS([weight], max_iter=max_iter, line_search_fn="strong_wolfe")

    def training_blocks():
        if resident_x is not None:
            yield resident_x, resident_y
        else:
            for start, block in blocks(phi, weight.device, chunk):
                if stop():
                    raise InterruptedError("Teacher fitting interrupted during streaming")
                chosen = mask[start:start + len(block)]
                if bool(chosen.any()):
                    yield block[chosen], labels[start:start + len(block)][chosen]

    def closure():
        if stop():
            raise InterruptedError("Teacher fitting interrupted")
        value = weight.new_zeros(())
        gradient = torch.zeros_like(weight)
        with torch.no_grad():
            for x, y in training_blocks():
                lp = (x @ weight).log_softmax(1)
                value -= lp[torch.arange(len(y), device=y.device), y].sum() / n
                error = lp.exp()
                error[torch.arange(len(y), device=y.device), y] -= 1
                gradient += x.T @ error / n
            value += gamma / n * weight.square().sum()
            gradient += 2 * gamma / n * weight
        weight.grad = gradient
        return value

    optimizer.step(closure)
    closure()
    if not torch.isfinite(weight).all() or float(weight.grad.abs().max()) > 1e-5:
        raise RuntimeError("Teacher logistic fit did not converge")
    with torch.no_grad():
        prediction = []
        for _, x in blocks(phi, weight.device, chunk):
            if stop():
                raise InterruptedError("Teacher prediction interrupted")
            prediction.append(x @ weight)
        logits = torch.cat(prediction)
    return logits, weight.detach()


def _inner_weights(mass, inner_loss_weighting):
    """Head-loss weights, separate from P-derived centroid denominators.

    Uniform weights are constant in the moments. Decoding centers and targets
    still divides their P-weighted sums by the actual cluster masses.
    """
    if not isinstance(inner_loss_weighting, str) or inner_loss_weighting not in ("mass", "uniform"):
        raise ValueError("Inner loss weighting must be mass or uniform")
    return mass if inner_loss_weighting == "mass" else torch.full_like(mass, 1 / len(mass))


def moment_gradient(moments, dimension, feature_map, theta, vector, penalty,
                    inner_loss_weighting="mass"):
    variable = moments.detach().requires_grad_()
    centers, labels, mass = decode_moments(variable, dimension)
    weights = _inner_weights(mass, inner_loss_weighting)
    gradient = head_gradient(augmented(feature_map(centers)), labels, weights,
                             theta.detach(), penalty)
    return torch.autograd.grad(-(gradient * vector.detach()).sum(), variable)[0]


def _mass_controls(mass_mode, balance_steps, balance_tol, balance_backend="chunked"):
    from numbers import Integral, Real

    if not isinstance(mass_mode, str) or mass_mode not in ("free", "uniform"):
        raise ValueError("mass_mode must be free or uniform")
    if isinstance(balance_steps, bool) or not isinstance(balance_steps, Integral) or balance_steps < 1:
        raise ValueError("balance_steps must be a positive integer")
    if (isinstance(balance_tol, bool) or not isinstance(balance_tol, Real)
            or not np.isfinite(balance_tol) or not 0 < balance_tol < 1):
        raise ValueError("balance_tol must be finite and lie strictly in (0, 1)")
    if mass_mode == "free" and (balance_steps != 300 or balance_tol != 1e-8):
        raise ValueError("Nondefault balancing controls require mass_mode='uniform'")
    if balance_backend != "chunked":
        raise ValueError("Nyström balancing currently supports only balance_backend=chunked")
    return int(balance_steps), float(balance_tol)


def validate_assignment_resume(saved, mass_mode="free", balance_steps=300, balance_tol=1e-8, balance_backend="chunked"):
    """Validate assignment mode before parameter copying or any cache mutation."""
    balance_steps, balance_tol = _mass_controls(mass_mode, balance_steps, balance_tol, balance_backend)
    config = saved.get("config", {})
    if config.get("mass_mode", "free") != mass_mode:
        raise ValueError("Resume mass_mode differs")
    if mass_mode == "uniform":
        if config.get("balance_steps") != balance_steps or config.get("balance_tol") != balance_tol or config.get("balance_backend") != balance_backend:
            raise ValueError("Resume balancing controls differ")
        dual = saved.get("dual")
        if dual is None and (saved.get("step", 0) > 0 or saved.get("theta") is not None):
            raise ValueError("Resume balancing dual is missing")
        if dual is not None and (not torch.is_tensor(dual) or dual.ndim != 1
                                 or not bool(torch.isfinite(dual).all())):
            raise ValueError("Resume balancing dual is invalid")
        if dual is not None and abs(float(dual.mean())) > 1e-10:
            raise ValueError("Resume balancing dual gauge is invalid")


def validate_assignment_snapshot(saved, mass_mode="free", balance_steps=300, balance_tol=1e-8, balance_backend="chunked"):
    """Protect cached endpoint bypasses, including selected_test recreation."""
    balance_steps, balance_tol = _mass_controls(mass_mode, balance_steps, balance_tol, balance_backend)
    if saved.get("mass_mode", "free") != mass_mode:
        raise ValueError("Nyström checkpoint mass_mode differs")
    if mass_mode == "free":
        return
    if saved.get("balance_steps") != balance_steps or saved.get("balance_tol") != balance_tol or saved.get("balance_backend") != balance_backend:
        raise ValueError("Nyström checkpoint balancing controls differ")
    moments, dual, diagnostic = saved.get("moments"), saved.get("dual"), saved.get("balance")
    if (not torch.is_tensor(moments) or moments.ndim != 2 or len(moments) < 1
            or not bool(torch.isfinite(moments).all()) or bool((moments[:, 0] <= 0).any())
            or not torch.is_tensor(dual) or dual.shape != (len(moments),)
            or not bool(torch.isfinite(dual).all()) or not isinstance(diagnostic, dict)):
        raise ValueError("Nyström balanced checkpoint lacks valid moments/dual/diagnostics")
    if abs(float(dual.mean())) > 1e-10 or float((len(moments) * moments.detach()[:, 0] - 1).abs().max()) > balance_tol:
        raise ValueError("Nyström checkpoint mass/gauge constraint failed")
    for key in ("row_residual", "column_residual"):
        value = diagnostic.get(key)
        if not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 <= value <= balance_tol:
            raise ValueError("Nyström checkpoint marginal diagnostic failed")
    iterations = diagnostic.get("balance_iterations")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or not 1 <= iterations <= balance_steps:
        raise ValueError("Nyström checkpoint balance iteration diagnostic failed")
    for key in ("backward_tangent_residual", "cg_residual", "cg_relative_residual"):
        value = diagnostic.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
            raise ValueError("Nyström checkpoint derivative diagnostic failed")
    if (diagnostic.get("balance_backward_checked") is not True
            or diagnostic.get("cg_converged") is not True):
        raise ValueError("Nyström checkpoint lacks a converged constrained derivative")


def optimize(h, q, assignment, feature_map, phi, folder, steps, penalty=1e-4,
             lr=0.01, rank=16, seed=0, chunk=2048, stop=lambda: False,
             progress=lambda row: None, checkpoint_every=25, mixing=0.05,
             inner_loss_weighting="mass", mass_mode="free", balance_steps=300, balance_tol=1e-8, balance_backend="chunked"):
    from numbers import Real

    if (isinstance(mixing, bool) or not isinstance(mixing, Real)
            or not np.isfinite(mixing) or not 0 < mixing < 1):
        raise ValueError("mixing must be finite and lie strictly in (0, 1)")
    if not isinstance(inner_loss_weighting, str) or inner_loss_weighting not in ("mass", "uniform"):
        raise ValueError("Inner loss weighting must be mass or uniform")
    balance_steps, balance_tol = _mass_controls(mass_mode, balance_steps, balance_tol, balance_backend)
    mixing = float(mixing)
    folder = Path(folder)
    # Cross-mode resumes must reject before even creating a phi sidecar.
    if (folder / "resume.pt").exists():
        preflight = torch.load(folder / "resume.pt", map_location="cpu", weights_only=False)
        validate_assignment_resume(preflight, mass_mode, balance_steps, balance_tol, balance_backend)
        del preflight
    folder.mkdir(parents=True, exist_ok=True)
    if chunk < 1 or not isinstance(chunk, int):
        raise ValueError("Optimization chunk must be a positive integer")
    _check_stop(stop)
    width = feature_map(h[:1]).shape[1]
    identity = _cache_identity(h, feature_map, (len(h), width), chunk, stop)
    phi_digest = _validate_phi(h, feature_map, phi, identity, chunk, stop)
    input_fingerprint = dict(
        **identity, phi_digest=phi_digest,
        q_digest=_content_digest(q, chunk, stop, canonical_double=True),
        assignment_digest=_content_digest(assignment, chunk, stop),
    )
    config = dict(steps_schema=2, penalty=penalty, lr=lr, rank=rank, seed=seed,
                  cells=int(assignment.max()) + 1, chunk=chunk)
    # Default .05 remains implicit for compatibility with verified old resumes.
    if mixing != 0.05:
        config["mixing"] = mixing
    if inner_loss_weighting != "mass":
        config["inner_loss_weighting"] = inner_loss_weighting
    if mass_mode == "uniform":
        config.update(mass_mode=mass_mode, balance_steps=balance_steps, balance_tol=balance_tol,
                      balance_backend=balance_backend)
    u, v = initialize_factors(assignment, config["cells"], rank, seed)
    optimizer = torch.optim.Adam([u, v], lr=lr)
    material = make_material(h.double(), q.double())
    start, theta, vector, history, dual = 0, None, None, [], None
    resume = folder / "resume.pt"
    if resume.exists():
        saved = torch.load(resume, map_location=h.device, weights_only=False)
        if "input_fingerprint" not in saved:
            raise ValueError(
                "Legacy resume lacks verifiable H/Q/assignment/map/phi fingerprints; "
                "preserve existing results and restart in a new folder"
            )
        if saved["input_fingerprint"] != input_fingerprint:
            raise ValueError("Resume input fingerprint differs in H/Q/assignment/map/phi")
        if saved["config"] != config:
            raise ValueError("Resume configuration differs")
        if mass_mode == "uniform":
            dual = saved.get("dual")
            if dual is not None and dual.shape != (config["cells"],):
                raise ValueError("Resume balancing dual has wrong cell count")
        with torch.no_grad():
            u.copy_(saved["u"])
            v.copy_(saved["v"])
        optimizer.load_state_dict(saved["optimizer"])
        start, theta, vector, history = (saved[k] for k in ("step", "theta", "vector", "history"))
    if start > steps:
        raise ValueError("Cannot resume to an earlier step")

    def persist(step):
        state = dict(config=config, step=step, u=u, v=v, optimizer=optimizer.state_dict(),
                     theta=theta, vector=vector, history=history,
                     input_fingerprint=input_fingerprint)
        if mass_mode == "uniform":
            state["dual"] = dual
        save_state(state, resume)

    def check_balanced_stop(step):
        if stop():
            persist(step)
            raise InterruptedError("Balanced condensation stopped with resumable state")

    for step in range(start, steps + 1):
        if stop():
            persist(step)
            raise InterruptedError("Condensation stopped with resumable state")
        started = time.monotonic()
        if mass_mode == "uniform":
            # The custom backward differentiates the solved column dual implicitly;
            # initial_dual is only a solver warm start, never a detached free-softmax path.
            from src.balanced_assignment import BalancedMoments
            from src.low_rank_assignment import LowRankLogits
            check_balanced_stop(step)
            logits = LowRankLogits.apply(u, v, assignment, mixing, chunk)
            logits.retain_grad()
            moments, dual, balancing = BalancedMoments.apply(
                logits, material, chunk, balance_steps, balance_tol, dual)
            if (not bool(torch.isfinite(moments).all()) or bool((moments[:, 0] <= 0).any())
                    or float((config["cells"] * moments.detach()[:, 0] - 1).abs().max()) > balance_tol
                    or float(balancing[1:3].max()) > balance_tol):
                raise FloatingPointError("Balanced assignment marginal validation failed")
            check_balanced_stop(step)
        else:
            moments = LowRankMoments.apply(u, v, assignment, material, mixing, chunk)
        centers, labels, mass = decode_moments(moments.detach(), h.shape[1])
        weights = _inner_weights(mass, inner_loss_weighting)
        mapped = feature_map(centers).detach()
        inner = solve_inner_newton_first(mapped, labels, weights, penalty, initial=theta)
        theta = inner["theta"]
        if not inner["inner_converged"]:
            if mass_mode == "free":
                persist(step)
            raise RuntimeError("Inner CE fit did not converge; refusing inexact update")
        value, rhs = outer_gradient(phi, q, theta, chunk)
        row = dict(step=step, outer_ce=value, inner_grad=inner["inner_grad_max"])
        if mass_mode == "uniform":
            # Validate the complete constrained derivative even at the final endpoint.
            check_balanced_stop(step)
            vector, diagnostic = solve_head_system(augmented(mapped), labels, weights, theta,
                                                   penalty, rhs, initial=vector)
            if not diagnostic["cg_converged"]:
                raise RuntimeError("Implicit Hessian solve did not converge")
            check_balanced_stop(step)
            derivative = moment_gradient(moments, h.shape[1], feature_map, theta, vector, penalty,
                                         inner_loss_weighting)
            optimizer.zero_grad(set_to_none=True)
            moments.backward(derivative)
            if not all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in (u, v)):
                raise RuntimeError("Nonfinite balanced assignment gradient")
            gauge_error = max(float(logits.grad.sum(0).abs().max()),
                              float(logits.grad.sum(1).abs().max()))
            if gauge_error > max(1e-10, 1e-5 * float(logits.grad.norm())):
                raise RuntimeError("Balancing backward row/column tangent validation failed")
            balance = dict(balance_iterations=int(balancing[0]), row_residual=float(balancing[1]),
                           column_residual=float(balancing[2]), balance_backward_checked=True,
                           backward_tangent_residual=gauge_error, **diagnostic)
            row.update(balance)
        history = [r for r in history if r["step"] < step] + [row]
        if step % checkpoint_every == 0 or step == steps:
            snapshot = dict(step=step, moments=moments, theta=theta, outer_ce=value,
                            input_fingerprint=input_fingerprint)
            if inner_loss_weighting != "mass":
                snapshot["inner_loss_weighting"] = inner_loss_weighting
            if mass_mode == "uniform":
                snapshot.update(mass_mode=mass_mode, balance_steps=balance_steps, balance_tol=balance_tol,
                                balance_backend=balance_backend,
                                dual=dual, balance=balance)
                validate_assignment_snapshot(snapshot, mass_mode, balance_steps, balance_tol, balance_backend)
            save_state(snapshot, folder / f"step_{step:06d}.pt")
            persist(step)
            save_json(history, folder / "history.json")
        if step == steps:
            progress(dict(**row, seconds=time.monotonic() - started))
            return folder / f"step_{step:06d}.pt"
        if mass_mode == "uniform":
            optimizer.step()
            progress(dict(**row, seconds=time.monotonic() - started))
            continue
        vector, diagnostic = solve_head_system(augmented(mapped), labels, weights, theta,
                                               penalty, rhs, initial=vector)
        if not diagnostic["cg_converged"]:
            persist(step)
            raise RuntimeError("Implicit Hessian solve did not converge")
        derivative = moment_gradient(moments, h.shape[1], feature_map, theta, vector, penalty,
                                     inner_loss_weighting)
        optimizer.zero_grad(set_to_none=True)
        moments.backward(derivative)
        if not all(torch.isfinite(p.grad).all() for p in (u, v)):
            persist(step)
            raise RuntimeError("Nonfinite assignment gradient")
        optimizer.step()
        progress(dict(**row, seconds=time.monotonic() - started))
    raise AssertionError("Unreachable")
