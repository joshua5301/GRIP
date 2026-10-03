"""Unexecuted independent CPU FP64 whole-CE precision-reference draft.

Stored FP32 source/model values are explicitly promoted. No tiled/segment or
partial-gradient helper is imported. The CPU adapter reads only signed existing
BW and BY caches; no dataset loader is called.
"""
import importlib
import json
import os
import platform
import resource
import time
import traceback
from pathlib import Path

from src import large_current_origin as bw
from src import large_source_preflight as metadata

np = sp = torch = F = None
SCIENCE = "Arxiv90_independent_FP64_SciPy_full_CE_reference_scientific_stageBZ_v1.json"
SCIENCE_SHA = "57f5285a35fb4f7f660451700709ab277de4d48d38e639054c0b5d5a270ccddb"
REVIEW = "independent_scientific_review_stageBZ_v1.json"
REVIEW_SHA = "8ccd4aca7d880871d8dce933113c7b1089b97035e01ca5e8a56ca82a20513f29"
FIELDS = {"schema", "fixed", "source", "numerical_source", "python_version", "files_sha256",
          "scientific_preregistration", "artifacts_sha256", "output_root", "output_path", "reference_cache_path", "comparison_details_path"}
_sha, _seal, _source_metadata, numerical_source = metadata._sha, metadata._seal, metadata._source, metadata.numerical_source
_descriptor, _checked = bw._descriptor, metadata._checked


def _libraries():
    global np, sp, torch, F
    libs = {name: importlib.import_module(name) for name in
            ("torch", "torch_geometric", "numpy", "scipy", "sklearn", "src.shared_features")}
    np, torch = libs["numpy"], libs["torch"]
    sp, F = importlib.import_module("scipy.sparse"), importlib.import_module("torch.nn.functional")
    return libs

PARAMETER_BLOCKS = ("W1", "b1", "W2", "b2")
RHO = .001
FORWARD_KEYS = ("U1", "V1", "A", "H", "U2", "V2", "logits", "logp", "full_logp")
COTANGENT_KEYS = ("D", "B", "DH", "E", "C")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _call(evidence, name, function, *args):
    if evidence is not None:
        evidence["_guard"]()
        evidence["operation_attempts"][name] = evidence["operation_attempts"].get(name, 0) + 1
    value = function(*args)
    if evidence is not None:
        evidence["counts"][name] = evidence["counts"].get(name, 0) + 1
    return value


def _cpu(value, dtype, shape, name):
    _require(isinstance(value, torch.Tensor) and value.device.type == "cpu" and value.layout == torch.strided
             and value.dtype == dtype and tuple(value.shape) == tuple(shape) and not value.requires_grad
             and bool(torch.isfinite(value).all()), "Malformed frozen CPU " + name)


def _source(source, evidence):
    _require(isinstance(source, torch.Tensor) and source.device.type == "cpu" and source.layout == torch.sparse_csr
             and source.dtype == torch.float32 and not source.requires_grad and source.ndim == 2
             and source.shape[0] == source.shape[1] > 0, "Require stored CPU FP32 square CSR")
    crow_tensor, col_tensor, values_tensor = source.crow_indices(), source.col_indices(), source.values()
    _require(crow_tensor.dtype == col_tensor.dtype == torch.int64 and bool(torch.isfinite(values_tensor).all()), "CSR type/value mismatch")
    crow, col, values32 = (value.detach().numpy().copy() for value in (crow_tensor, col_tensor, values_tensor))
    n = source.shape[0]
    _require(crow.shape == (n + 1,) and col.ndim == values32.ndim == 1 and crow[0] == 0
             and crow[-1] == len(col) == len(values32) and np.all(crow[1:] >= crow[:-1])
             and np.all((col >= 0) & (col < n)), "Malformed stored CSR coordinates")
    rows = np.repeat(np.arange(n, dtype=np.int64), np.diff(crow))
    _require(np.all((rows[1:] != rows[:-1]) | (col[1:] > col[:-1])), "Stored CSR must be canonical sorted unique")
    values64 = _call(evidence, "stored_FP32_coefficient_casts", lambda: values32.astype(np.float64, copy=True))
    _require(np.array_equal(values64.astype(np.float32).view(np.uint32), values32.view(np.uint32)), "FP32-to-FP64 value lineage differs")
    matrix = _call(evidence, "independent_SciPy_CSR_constructions", lambda: sp.csr_matrix((values64.copy(), col.copy(), crow.copy()), shape=(n, n)))
    # SciPy may choose smaller integer dtypes in its constructor; restore exact
    # original int64 coordinates without reordering, renormalizing or merging.
    matrix.indptr, matrix.indices = crow.copy(), col.copy()
    _require(np.array_equal(matrix.data.view(np.uint64), values64.view(np.uint64)), "SciPy coefficient bits differ")
    transposed = _call(evidence, "independent_actual_transpose_constructions", lambda: matrix.transpose(copy=True).tocsr(copy=True))
    order = np.lexsort((rows, col))
    trows = np.repeat(np.arange(n, dtype=np.int64), np.diff(transposed.indptr))
    _require(np.array_equal(trows, col[order]) and np.array_equal(transposed.indices, rows[order])
             and np.array_equal(transposed.data.view(np.uint64), values64[order].view(np.uint64))
             and len(order) == len(values64), "Actual SciPy transpose coordinate/value bijection differs")
    return matrix, transposed, dict(nodes=n, edges=len(col), original_index_dtype="int64", original_value_dtype="float32",
                                   reference_value_dtype="float64", exact_coefficient_cast_lineage=True,
                                   actual_transpose_coordinate_value_bijection=True, signed_zero_value_bits_preserved=True)


def _sparse_apply(value, source, transposed, evidence):
    class _FixedCSR(torch.autograd.Function):
        """One independent whole-width S forward / actual S.T reverse product."""
        @staticmethod
        def forward(ctx, value, source, transposed, evidence):
            _require(value.device.type == "cpu" and value.dtype == torch.float64 and value.ndim == 2,
                     "Independent product requires CPU FP64 rank2")
            ctx.transposed, ctx.evidence = transposed, evidence
            result = _call(evidence, "independent_SciPy_source_products", source.dot, value.detach().numpy())
            _require(result.dtype == np.float64 and np.isfinite(result).all(), "Nonfinite full SciPy forward")
            return torch.from_numpy(np.array(result, dtype=np.float64, order="C", copy=True))
    
        @staticmethod
        def backward(ctx, cotangent):
            _require(cotangent.device.type == "cpu" and cotangent.dtype == torch.float64 and cotangent.ndim == 2,
                     "Independent cotangent requires CPU FP64 rank2")
            result = _call(ctx.evidence, "independent_SciPy_transpose_products", ctx.transposed.dot, cotangent.detach().numpy())
            _require(result.dtype == np.float64 and np.isfinite(result).all(), "Nonfinite full SciPy reverse")
            return torch.from_numpy(np.array(result, dtype=np.float64, order="C", copy=True)), None, None, None
    return _FixedCSR.apply(value, source, transposed, evidence)

def _forward64(parameters, x, source, transposed, q, evidence=None):
    w1, b1, w2, b2 = parameters
    _attempt(evidence, "whole_FP64_source_forward_passes")
    u1 = x @ w1.T
    v1 = _sparse_apply(u1, source, transposed, evidence)
    a = v1 + b1
    h = F.relu(a)
    u2 = h @ w2.T
    v2 = _sparse_apply(u2, source, transposed, evidence)
    logits = v2 + b2
    logp = F.log_softmax(logits, dim=1)
    loss = _call(evidence, "whole_FP64_source_CE_evaluations", lambda: -(q * logp).sum(1).mean())
    _complete(evidence, "whole_FP64_source_forward_passes")
    return loss, dict(U1=u1, V1=v1, A=a, mask=a > 0, H=h, U2=u2, V2=v2, logits=logits,
                      logp=logp, full_logp=logp, full_CE=loss)


def full_ce_reference(parameters_fp32, x_fp32, source_fp32, q_fp64, evidence=None):
    """One whole gradient traversal returns params plus D/B/DH/E/C.

    A new science/source acceptance and constructive toy proof are prerequisites
    to executing this draft. No normalizer, source transform or head is fitted.
    """
    _require(isinstance(parameters_fp32, (tuple, list)) and len(parameters_fp32) == 4, "Require four frozen parameter blocks")
    w1_32, b1_32, w2_32, b2_32 = parameters_fp32
    _require(w1_32.ndim == w2_32.ndim == 2, "Require rank2 weights")
    hidden, features = w1_32.shape
    classes = w2_32.shape[0]
    n = source_fp32.shape[0]
    for value, shape, name in zip(parameters_fp32, ((hidden, features), (hidden,), (classes, hidden), (classes,)), PARAMETER_BLOCKS):
        _cpu(value, torch.float32, shape, name)
    _cpu(x_fp32, torch.float32, (n, features), "X")
    _cpu(q_fp64, torch.float64, (n, classes), "Q")
    _require(bool((q_fp64 >= 0).all()) and float((q_fp64.sum(1) - 1).abs().max()) <= 32 * torch.finfo(torch.float64).eps,
             "Frozen Q probability rows differ")
    source, transposed, source_certificate = _source(source_fp32, evidence)
    parameters = tuple(_call(evidence, "stored_FP32_anchor_block_casts", lambda v=value: v.detach().double().clone()).requires_grad_(True) for value in parameters_fp32)
    x, q = _call(evidence, "stored_FP32_X_casts", lambda: x_fp32.detach().double()), q_fp64.detach()
    for old, cast in zip((x_fp32,) + tuple(parameters_fp32), (x,) + parameters):
        _require(torch.equal(old.view(torch.int32), cast.detach().float().view(torch.int32)), "Stored FP32 cast lineage differs")
    loss, forward = _forward64(parameters, x, source, transposed, q, evidence)
    logits, u2, h, a, u1 = (forward[key] for key in ("logits", "U2", "H", "A", "U1"))
    derivatives = _call(evidence, "whole_torch_autograd_grad_calls", torch.autograd.grad,
                        loss, parameters + (logits, u2, h, a, u1))
    gradients, cotangents = derivatives[:4], derivatives[4:]
    boundaries = dict(forward, **dict(zip(COTANGENT_KEYS, cotangents)))
    _require(all(bool(torch.isfinite(v).all()) for v in tuple(boundaries.values()) + gradients + (loss,)),
             "Nonfinite independent complete reference")
    detached = lambda value: value.detach()  # Complete CPU arrays retained, no duplicate GPU snapshot.
    _attempt(evidence, "independent_gradient_block_outputs", 4)
    _complete(evidence, "independent_gradient_block_outputs", 4)
    norms = tuple(_call(evidence, "independent_source_gradient_norms", torch.linalg.vector_norm, value.reshape(-1)) for value in gradients)
    _require(all(bool(torch.isfinite(v)) and float(v) > 0 for v in norms), "Independent four source norms must be finite positive")
    delta = tuple(_call(evidence, "independent_source_gradient_deltas", lambda v=value: RHO * v) for value in norms)
    _require(all(bool(torch.isfinite(v)) and float(v) > 0 for v in delta), "Independent source smoothing must be finite positive")
    return dict(parameters=tuple(detached(v) for v in parameters), CE=detached(loss),
                gradients=tuple(detached(v) for v in gradients), boundaries={key: detached(value) for key, value in boundaries.items()},
                source_norms=tuple(detached(v) for v in norms), delta=tuple(detached(v) for v in delta),
                source_certificate=source_certificate, partial_VJP_helper_calls=0, target_accuracy_qualified=False,
                input_lineage=dict(original=dict(X=x_fp32, S=source_fp32, Q=q_fp64, anchor=tuple(parameters_fp32)),
                    cast=dict(X=x, Q=q, anchor=tuple(v.detach() for v in parameters),
                        S=_csr_arrays(source), T=_csr_arrays(transposed))))


def error_metrics(native, reference):
    """Descriptive full-array distributions; no finite filtering or pruning."""
    _require(native.device.type == reference.device.type == "cpu" and native.shape == reference.shape
             and bool(torch.isfinite(native).all()) and bool(torch.isfinite(reference).all()), "Invalid comparison arrays")
    actual, expected = native.detach().double(), reference.detach().double()
    difference = actual - expected
    absolute = difference.abs().reshape(-1)
    norm = torch.linalg.vector_norm(expected.reshape(-1))
    absolute_l2 = torch.linalg.vector_norm(difference.reshape(-1))
    quantiles = np.quantile(absolute.numpy(), np.array([0., .5, .9, .99, 1.]))
    return dict(max_abs=float(absolute.max()), mean_abs=float(absolute.mean()), RMS=float(difference.square().mean().sqrt()),
                absolute_L2=float(absolute_l2), reference_L2=float(norm), relative_L2=None if float(norm) == 0 else float(absolute_l2 / norm),
                absolute_error_quantiles={str(p): float(v) for p, v in zip((0., .5, .9, .99, 1.), quantiles)},
                reference_abs_max=float(expected.abs().max()))


def compare_complete(native, reference):
    """Prospective pre-BY ff31 limits, subject to final root science freeze.

    All near-zero mask cells remain in every complete derivative comparison.
    The caller must persist complete independent raw arrays before this call.
    """
    _validate_comparison(native, reference)
    comparisons, passed = {}, True
    for key in FORWARD_KEYS + COTANGENT_KEYS + ("full_CE",):
        metrics = error_metrics(native["boundaries"][key], reference["boundaries"][key])
        bound = (1e-5 * (1 + metrics["reference_abs_max"]) if key in FORWARD_KEYS else (5e-6 * (1 + metrics["reference_abs_max"]) if key == "full_CE" else 1e-12 + 5e-5 * metrics["reference_abs_max"]))
        metrics.update(bound=bound, passed=metrics["max_abs"] <= bound)
        comparisons[key], passed = metrics, passed and metrics["passed"]
    ce = error_metrics(native["CE"], reference["CE"])
    ce.update(bound=5e-6 * (1 + abs(float(reference["CE"]))), passed=abs(float(native["CE"]) - float(reference["CE"])) <= 5e-6 * (1 + abs(float(reference["CE"]))))
    comparisons["CE"], passed = ce, passed and ce["passed"]
    for block, actual, expected, norm, reference_norm, delta, reference_delta in zip(PARAMETER_BLOCKS, native["target"]["gradients"],
            reference["gradients"], native["target"]["source_norms"], reference["source_norms"], native["target"]["delta"], reference["delta"]):
        metrics = error_metrics(actual, expected)
        bound = 1e-10 + 5e-5 * metrics["reference_abs_max"]
        norm_bound = 1e-10 + 1e-4 * float(reference_norm)
        valid = bool(torch.isfinite(norm)) and bool(torch.isfinite(reference_norm)) and float(norm) > 0 and float(reference_norm) > 0 and bool(torch.isfinite(delta)) and bool(torch.isfinite(reference_delta)) and float(delta) > 0 and float(reference_delta) > 0
        record = dict(gradient=metrics, gradient_bound=bound, norm_absolute_error=abs(float(norm) - float(reference_norm)),
                      norm_bound=norm_bound, delta_absolute_error=abs(float(delta) - float(reference_delta)), delta_bound=RHO * norm_bound)
        record["passed"] = valid and metrics["max_abs"] <= bound and record["norm_absolute_error"] <= norm_bound and record["delta_absolute_error"] <= RHO * norm_bound
        comparisons[block], passed = record, passed and record["passed"]
    reference_a = reference["boundaries"]["A"]
    threshold = 1e-5 * (1 + float(reference_a.abs().max()))
    mismatch = native["boundaries"]["mask"] != reference["boundaries"]["mask"]
    far = reference_a.abs() > threshold
    far_count, near_count = int((mismatch & far).sum()), int((mismatch & ~far).sum())
    return dict(passed=passed and far_count == 0, comparisons=comparisons, mask_threshold=threshold,
                mask_mismatches_full=far_count + near_count, mask_mismatches_far=far_count, mask_mismatches_near=near_count,
                raw_mask_mismatch_flatcoordinates=mismatch.reshape(-1).nonzero().reshape(-1),
                native_mask=native["boundaries"]["mask"], reference_mask=reference["boundaries"]["mask"],
                near_zero_cells_filtered_from_derivatives=False, native_accuracy_qualified=False)


def _attempt(evidence, name, amount=1):
    if evidence is not None:
        evidence["_guard"]()
        evidence["operation_attempts"][name] = evidence["operation_attempts"].get(name, 0) + amount


def _complete(evidence, name, amount=1):
    if evidence is not None:
        evidence["counts"][name] = evidence["counts"].get(name, 0) + amount


def _csr_arrays(matrix):
    return dict(shape=list(matrix.shape), crow=torch.from_numpy(matrix.indptr.astype(np.int64, copy=True)),
                col=torch.from_numpy(matrix.indices.astype(np.int64, copy=True)), values=torch.from_numpy(matrix.data.copy()))


def _validate_comparison(native, reference):
    _require(isinstance(native, dict) and isinstance(reference, dict), "Malformed complete comparison")
    nb, rb = native["boundaries"], reference["boundaries"]
    expected = set(FORWARD_KEYS + COTANGENT_KEYS + ("mask", "full_CE"))
    _require(set(nb) == set(rb) == expected, "Require every full boundary")
    n, hidden = rb["A"].shape
    classes, features = reference["parameters"][2].shape[0], reference["parameters"][0].shape[1]
    for key in expected:
        shape = (() if key == "full_CE" else (n, hidden) if key in ("U1", "V1", "A", "H", "DH", "E", "C", "mask") else (n, classes))
        _cpu(nb[key], torch.bool if key == "mask" else torch.float64 if key == "full_CE" else torch.float32, shape, "native " + key)
        _cpu(rb[key], torch.bool if key == "mask" else torch.float64, shape, "reference " + key)
    for value in (native["CE"], reference["CE"]):
        _cpu(value, torch.float64, (), "CE")
    _require(torch.equal(native["CE"].view(torch.int64), nb["full_CE"].view(torch.int64))
             and torch.equal(reference["CE"].view(torch.int64), rb["full_CE"].view(torch.int64)), "CE aliases differ")
    target = native["target"]
    _require(set(target) == {"blocks", "rho", "gradients", "source_norms", "delta"}
             and tuple(target["blocks"]) == PARAMETER_BLOCKS and type(target["rho"]) is float and target["rho"] == RHO
             and all(isinstance(target[k], (list, tuple)) and len(target[k]) == 4 for k in ("gradients", "source_norms", "delta"))
             and all(isinstance(reference[k], (list, tuple)) and len(reference[k]) == 4 for k in ("parameters", "gradients", "source_norms", "delta")), "Truncated/extra/mislabeled target blocks")
    for index, shape in enumerate(((hidden, features), (hidden,), (classes, hidden), (classes,))):
        _cpu(reference["parameters"][index], torch.float64, shape, "reference parameter")
        _cpu(target["gradients"][index], torch.float32, shape, "native gradient")
        _cpu(reference["gradients"][index], torch.float64, shape, "reference gradient")
        for values in (target["source_norms"], target["delta"], reference["source_norms"], reference["delta"]):
            _cpu(values[index], torch.float64, (), "norm/delta")
            _require(float(values[index]) > 0, "Each norm/delta must be strictly positive")


def _finite_cpu(value):
    if torch.is_tensor(value):
        _require(value.device.type == "cpu" and value.layout in (torch.strided, torch.sparse_csr) and not value.requires_grad,
                 "Require frozen CPU cache arrays")
        array = value if value.layout == torch.strided else value.values()
        _require(bool(torch.isfinite(array).all()), "Nonfinite complete CPU cache")
    elif isinstance(value, dict):
        for child in value.values():
            _finite_cpu(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            _finite_cpu(child)


def _controls(spec, science, repo):
    _require(isinstance(spec, dict) and set(spec) == FIELDS and type(spec["schema"]) is int and spec["schema"] == 1, "Require exact twelve-field reference spec")
    c = science["implementation_contract"]
    for key, expected in (("fixed", science["fixed"]), ("files_sha256", science["original_files_sha256"]), ("output_root", c["output_root"]),
                          ("output_path", c["receipt_path"]), ("reference_cache_path", c["reference_cache_path"]), ("comparison_details_path", c["comparison_details_path"]),
                          ("scientific_preregistration", dict(path=str(repo / "results/proposals" / SCIENCE), sha256=SCIENCE_SHA))):
        _require(_seal(spec[key]) == _seal(expected), "Changed fixed reference spec: " + key)
    owned = {(repo / p).resolve() for p in c["owned_paths"]}
    _require(isinstance(spec["artifacts_sha256"], dict) and spec["artifacts_sha256"] and all(Path(p).is_absolute()
             and (Path(p).resolve().is_relative_to(repo / "results") or Path(p).resolve() in owned) for p in spec["artifacts_sha256"]), "Wrong exact owned/results artifact allowlist")
    _require(set(spec["source"].get("files", {})) == set(science["source_before"]["files"]) | {"src/scipy_full_ce_reference.py"}
             and spec["numerical_source"].get("versions") == science["cpu_policy"]["versions"]
             and spec["python_version"] == science["cpu_policy"]["python_version"], "Require new88 current source/versions")


def _preserve(spec, path, checksum, science, repo):
    _require(_sha(path) == checksum and _source_metadata(repo) == spec["source"] and numerical_source(repo) == spec["numerical_source"]
             and platform.python_version() == spec["python_version"], "Spec/current source/Git/versions changed")
    for pins in (spec["files_sha256"], spec["artifacts_sha256"],
                 {str(repo / "results/proposals" / SCIENCE): SCIENCE_SHA, str(repo / "results/proposals" / REVIEW): REVIEW_SHA},
                 {r["path"]: r["sha256"] for r in science["parents"].values()},
                 {str(repo / p): h for p, h in science["source_before"]["files"].items() if p != "src/research_loop.py"},
                 {str(repo / p): h for p, h in science["protected_tests_before"].items()},
                 {science["native_binding"]["raw_cache"]["path"]: science["native_binding"]["raw_cache"]["sha256"],
                  science["native_binding"]["receipt"]["path"]: science["native_binding"]["receipt"]["sha256"]}):
        _checked(pins)


def _load_spec(path, checksum):
    repo, path = Path(__file__).resolve().parents[1], Path(path).resolve()
    _checked({str(path): checksum, str(repo / "results/proposals" / SCIENCE): SCIENCE_SHA})
    science = json.loads((repo / "results/proposals" / SCIENCE).read_text())
    _require(str(path) == science["implementation_contract"]["spec_path"], "Wrong prospective reference spec path")
    spec = json.loads(path.read_text())
    _controls(spec, science, repo)
    _preserve(spec, path, checksum, science, repo)
    return spec, science, repo


def _cpu_precision(policy=None):
    _require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not torch.cuda.is_initialized() and torch.get_num_threads() == 4
             and torch.get_default_dtype() == torch.float32 and torch.are_deterministic_algorithms_enabled()
             and not torch.is_deterministic_algorithms_warn_only_enabled() and torch.get_float32_matmul_precision() == "highest"
             and not torch.is_autocast_enabled() and not torch.is_autocast_enabled("cpu")
             and not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32,
             "CPU-only precision/environment changed")
    if policy is not None:
        _require(all(os.environ.get(name) == policy[name] for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")), "CPU threadpool environment changed")


def _cpu_native(libs, science):
    policy = science["cpu_policy"]
    _require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not torch.cuda.is_initialized(), "Require fresh CPU-only worker")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    _cpu_precision(policy)
    versions = {k: str(libs[name].__version__) for k, name in (("torch", "torch"), ("pyg", "torch_geometric"), ("numpy", "numpy"), ("scipy", "scipy"), ("sklearn", "sklearn"))}
    _require(versions == policy["versions"], "Actual CPU library versions differ")
    return dict(device="cpu", CUDA_initialized=False, versions=versions, python_version=platform.python_version(), threads=torch.get_num_threads(),
                deterministic=True, warn_only=False, AMP=False, TF32=False, default_dtype=str(torch.get_default_dtype()), matmul_precision="highest")


def _validate_native(libs, payload, science):
    binding = science["native_binding"]
    _require(isinstance(payload, dict) and set(payload) == {"schema", "pass_index", "context", "common", "common_numerical_digest", "common_numerical_seal", "content_sha256"}
             and type(payload["schema"]) is int and payload["schema"] == 1 and type(payload["pass_index"]) is int and payload["pass_index"] == 1, "Malformed accepted BY raw payload")
    _finite_cpu(payload["common"])
    digest = _descriptor(libs, payload["common"])
    _require(payload["context"] == binding["context"] and _seal(payload["context"]) == binding["raw_cache_context_sha256"]
             and digest == payload["common_numerical_digest"] == binding["common_numerical_digest"]
             and _seal(digest) == payload["common_numerical_seal"] == binding["common_numerical_seal"]
             and _seal(dict(schema=1, pass_index=1, context=payload["context"], common_numerical_digest=digest)) == payload["content_sha256"] == binding["raw_cache_content_sha256"], "Accepted BY full seals differ")
    common = payload["common"]
    _require(set(common) == {"source_buffers", "S", "T", "anchor", "boundaries", "CE", "target"}
             and set(common["source_buffers"]) == {"X", "original_CSR", "Q"}, "Incomplete source common")
    for name in common["source_buffers"]:
        _require(_descriptor(libs, common["source_buffers"][name]) == science["origin_binding"]["full16_descriptors"][name], "Original source descriptor differs")
    _require(_descriptor(libs, common["S"]) == _descriptor(libs, common["source_buffers"]["original_CSR"]), "S/original source differs")
    return common


def _context(spec, path, checksum, science):
    return dict(schema=1, spec=dict(path=str(path), sha256=checksum), scientific_preregistration=spec["scientific_preregistration"],
                source=spec["source"], numerical_source=spec["numerical_source"], python_version=spec["python_version"], fixed=spec["fixed"],
                BW_origin=dict(file_sha256=science["origin_binding"]["origin_file_sha256"], context_sha256=science["origin_binding"]["origin_context_sha256"], content_sha256=science["origin_binding"]["origin_content_sha256"]),
                BY_native=dict(file_sha256=science["native_binding"]["raw_cache"]["sha256"], context_sha256=science["native_binding"]["raw_cache_context_sha256"], common_numerical_seal=science["native_binding"]["common_numerical_seal"]))


def _write_raw(libs, path, context, value, evidence, name):
    _finite_cpu(value)
    digest = _descriptor(libs, value)
    payload = dict(schema=1, context=context, value=value, descriptors=digest)
    payload["content_sha256"] = _seal(dict(context=context, descriptors=digest))
    def save():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
    _call(evidence, name, save)
    return dict(path=str(path), sha256=_sha(path), content_sha256=payload["content_sha256"], context_sha256=_seal(context), descriptors=digest)


def _observe_raw(path, evidence, prefix):
    evidence[prefix + "_observed_path"] = str(path)
    evidence[prefix + "_observed_exists"] = path.is_file()
    if path.is_file():
        evidence[prefix + "_observed_sha256"] = _sha(path)
        evidence[prefix + "_observed_bytes"] = path.stat().st_size


def _guard(started, stop, science):
    _require(not stop() and time.monotonic() - started < 300, "Stopped/CPU reference deadline")
    _cpu_precision(science["cpu_policy"])
    _require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 <= science["cpu_policy"]["peak_process_RSS_budget_bytes"], "CPU RSS cap exceeded")


def prepare_independent_reference(spec_path, spec_sha256, output_path, stop=lambda: False):
    started = time.monotonic()
    _require(callable(stop), "Require stop callback")
    try:
        spec, science, repo = _load_spec(spec_path, spec_sha256)
    except InterruptedError as error:
        raise RuntimeError("Terminal independent CPU reference interruption before spec binding") from error
    output, raw, details = (Path(spec[k]) for k in ("output_path", "reference_cache_path", "comparison_details_path"))
    _require(Path(output_path).resolve() == output and not any(p.exists() for p in (output, raw, details)), "Require absent own reference paths; no resume")
    evidence = dict(passed=False, operation="prepare_independent_reference", validation_only=True, test_enabled=False,
                    source=spec["source"], numerical_source=spec["numerical_source"], python_version=spec["python_version"],
                    scientific_preregistration=spec["scientific_preregistration"], spec_path=str(Path(spec_path).resolve()), spec_sha256=spec_sha256,
                    fixed=spec["fixed"], files_sha256=spec["files_sha256"], counts=dict(science["counts_contract"]["zero_work"]), operation_attempts={}, success_counts=None,
                    count_scope=science["counts_contract"]["scope"], anchor0_independent_FP64_precision_agreement_passed=False,
                    production_alignment_target_consumption_allowed=False, three_anchor_backend_qualified=False, efficacy_qualified=False,
                    all15_budgets_open=True, goal_complete=False, stage="before_lazy_CPU_libraries")
    libs, retained, primary, rng = None, {}, None, None
    try:
        _require(not stop(), "Stopped before CPU libraries")
        libs = _libraries()
        evidence["native_environment"] = _cpu_native(libs, science)
        evidence["_guard"] = lambda: _guard(started, stop, science)
        rng = torch.random.get_rng_state().clone()
        _preserve(spec, Path(spec_path), spec_sha256, science, repo)
        b = science["origin_binding"]
        origin = _call(evidence, "accepted_BW_origin_CPU_loads", lambda: torch.load(b["origin_path"], map_location="cpu", weights_only=False))
        retained["origin"] = origin
        evidence["origin_arrays_before"] = _descriptor(libs, origin["arrays"])
        _call(evidence, "BW_full_packet_validation_calls", bw._validate_packet, libs, origin,
              dict(origin_content_sha256=b["origin_content_sha256"], origin_context_sha256=b["origin_context_sha256"]))
        _require(origin["descriptors"] == b["full16_descriptors"] and origin["context"]["source"] == b["historical_source"], "Signed16 BW lineage differs")
        payload = _call(evidence, "accepted_BY_raw_cache_CPU_loads", lambda: torch.load(science["native_binding"]["raw_cache"]["path"], map_location="cpu", weights_only=False))
        retained["native"] = payload
        evidence["native_arrays_before"] = _descriptor(libs, payload["common"])
        native = _call(evidence, "BY_full_cache_validation_calls", _validate_native, libs, payload, science)
        arrays = origin["arrays"]
        _preserve(spec, Path(spec_path), spec_sha256, science, repo)
        evidence["stage"] = "independent_whole_FP64_CE"
        _attempt(evidence, "independent_source_target_packets")
        reference = _call(evidence, "independent_source_parameter_gradient_assemblies", full_ce_reference,
                          native["anchor"], arrays["X"], arrays["original_CSR"], arrays["Q"], evidence)
        _complete(evidence, "independent_source_target_packets")
        retained["reference"] = reference
        evidence["reference_numerical_digest"] = _descriptor(libs, reference)
        evidence["reference_numerical_seal"] = _seal(evidence["reference_numerical_digest"])
        context = _context(spec, Path(spec_path).resolve(), spec_sha256, science)
        evidence["stage"] = "exclusive_reference_before_comparison"
        capture = _write_raw(libs, raw, context, reference, evidence, "complete_reference_cache_writes")
        evidence.update(reference_cache_path=capture["path"], reference_cache_sha256=capture["sha256"],
                        reference_cache_content_sha256=capture["content_sha256"], reference_cache_context_sha256=capture["context_sha256"], complete_reference_raw_captured_before_comparison=True)
        _preserve(spec, Path(spec_path), spec_sha256, science, repo)
        evidence["stage"] = "full_unfiltered_native_reference_comparison"
        comparison = _call(evidence, "full_native_reference_comparison_calls", compare_complete, native, reference)
        arrays_detail = {key: comparison.pop(key) for key in ("raw_mask_mismatch_flatcoordinates", "native_mask", "reference_mask")}
        capture = _write_raw(libs, details, context, arrays_detail, evidence, "comparison_details_writes")
        evidence.update(comparison=comparison, comparison_details_path=capture["path"], comparison_details_sha256=capture["sha256"],
                        comparison_details_content_sha256=capture["content_sha256"])
        _require(comparison["passed"] is True, "Independent FP64 precision budgets failed")
        evidence["passed"] = True
    except BaseException as error:
        primary = error
        evidence.update(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    finally:
        try:
            if libs is not None:
                for name, value, before in (("origin", retained.get("origin", {}).get("arrays"), "origin_arrays_before"),
                                            ("native", retained.get("native", {}).get("common"), "native_arrays_before"),
                                            ("reference", retained.get("reference"), "reference_numerical_digest")):
                    if value is not None and before in evidence:
                        after = _descriptor(libs, value)
                        evidence[name + "_arrays_after"] = after
                        _require(after == evidence[before], "Retained CPU arrays changed: " + name)
                        evidence[name + "_arrays_unchanged"] = True
                _cpu_precision(science["cpu_policy"])
                if rng is not None:
                    _require(torch.equal(rng, torch.random.get_rng_state()), "Global CPU RNG changed")
                    evidence["global_CPU_RNG_unchanged"] = True
            _preserve(spec, Path(spec_path), spec_sha256, science, repo)
            evidence["source_assets_spec_science_unchanged"] = True
            evidence.update(elapsed_seconds=time.monotonic() - started, peak_process_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
            _require(evidence["elapsed_seconds"] <= 300 and evidence["peak_process_RSS_bytes"] <= science["cpu_policy"]["peak_process_RSS_budget_bytes"], "Reference resource gate failed")
            if evidence["passed"]:
                expected = dict(science["counts_contract"]["per_complete_large_reference"], **science["counts_contract"]["zero_work"])
                _require({key: evidence["counts"].get(key) for key in expected} == expected, "Actual interface counts differ")
                evidence.update(success_counts=expected, anchor0_independent_FP64_precision_agreement_passed=True)
        except BaseException as error:
            evidence.update(passed=False, finally_error_type=type(error).__name__, finally_error=str(error))
            if primary is None:
                primary = error
        for path, prefix in ((raw, "reference_cache"), (details, "comparison_details")):
            try:
                _observe_raw(path, evidence, prefix)
            except BaseException as error:
                evidence.update(passed=False, raw_observation_error_type=type(error).__name__, raw_observation_error=str(error))
                if primary is None:
                    primary = error
        if not evidence["passed"]:
            evidence.update(success_counts=None, anchor0_independent_FP64_precision_agreement_passed=False)
        evidence.pop("_guard", None)
        try:
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x") as stream:
                json.dump(evidence, stream, indent=2, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException as error:
            # Preserve the original failure and any partial exclusive receipt.
            if primary is None:
                primary = error
    if primary is not None:
        # Keep the observed original type in the receipt but never trigger the
        # generic loop's InterruptedError retry semantics for this exact job.
        if isinstance(primary, InterruptedError):
            raise RuntimeError("Terminal independent CPU reference interruption") from primary
        raise primary
    return dict(evidence, evidence_path=str(output), evidence_sha256=_sha(output), validation_only=True)
