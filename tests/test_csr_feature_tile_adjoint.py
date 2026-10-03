"""Eight new CPU groups for fixed feature16 tiling; no real-data/native replay."""
import hashlib
import importlib.util
import json
import os
import time
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from src import csr_segment_adjoint as full

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src/csr_segment_adjoint.py").is_file())
MODULE_PATH = (ROOT / "src/csr_feature_tile_adjoint.py" if Path(__file__).resolve().parent == ROOT / "tests"
               else Path(__file__).resolve().parents[1] / "src/csr_feature_tile_adjoint.py")
SPEC = importlib.util.spec_from_file_location("bx_feature_tile", MODULE_PATH)
tile = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tile)

SCIENCE_SHA256 = "44af15ea25cb8a462a8afbba0336770eb45aea5991f3a36c0d4e3aa77d17805f"
CROW = (0, 2, 2, 4, 7, 8, 10, 11)
COLUMNS = (0, 3, 1, 4, 0, 2, 6, 5, 1, 3, 6)
VALUES = (.75, -0.0, .25, .5, 0.0, .375, .125, .25, .5, .25, 1.0)
WIDTHS = (1, 16, 17, 31, 37)
EPSILONS = (1e-6, 5e-7)
REPORT = {"schema": 1, "scientific_preregistration_sha256": SCIENCE_SHA256, "metrics": {}}
EVIDENCES = []
STARTED = None
ISOLATED = os.environ.get("BX_ISOLATED_PROOF") == "1"


def _bits(value):
    value = value.detach().contiguous()
    if value.dtype == torch.float32:
        return value.view(torch.int32)
    if value.dtype == torch.float64:
        return value.view(torch.int64)
    return value


def _descriptor(value):
    if value.layout == torch.sparse_csr:
        return {"shape": list(value.shape), "dtype": str(value.dtype), "layout": str(value.layout),
                "parts": [_descriptor(v) for v in (value.crow_indices(), value.col_indices(), value.values())]}
    value = value.detach().contiguous()
    return {"shape": list(value.shape), "dtype": str(value.dtype), "layout": str(value.layout),
            "digest": hashlib.sha256(bytes(value.untyped_storage())).hexdigest()}


def _bit_equal(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert torch.equal(_bits(actual), _bits(expected))


def _csr(dtype):
    return torch.sparse_csr_tensor(torch.tensor(CROW, dtype=torch.int64),
                                  torch.tensor(COLUMNS, dtype=torch.int64),
                                  torch.tensor(VALUES, dtype=dtype), size=(7, 7))


def _guard():
    if ISOLATED:
        assert not torch.cuda.is_initialized()
    if time.monotonic() - STARTED > 60:
        raise RuntimeError("BX fixed CPU60s bound")


def _evidence(name, guard=_guard):
    result = {"_guard": guard, "operation_attempts": {}, "counts": {}}
    EVIDENCES.append((name, result))
    return result


def _fixture(dtype=torch.float64):
    gen = torch.Generator(device="cpu").manual_seed(2020)
    x = torch.randn((7, 5), generator=gen, dtype=torch.float64) * .2
    w1 = torch.randn((19, 5), generator=gen, dtype=torch.float64) * .03
    b1 = torch.tensor([.4 if j % 2 == 0 else -.4 for j in range(19)], dtype=torch.float64)
    w1[0, :] = 0
    b1[0] = 0
    w2 = torch.randn((17, 19), generator=gen, dtype=torch.float64) * .25
    b2 = torch.randn((17,), generator=gen, dtype=torch.float64) * .05
    qgen = torch.Generator(device="cpu").manual_seed(2021)
    q = (torch.randn((7, 17), generator=qgen, dtype=torch.float64) * .25).softmax(1)
    return tuple(v.to(dtype=dtype) for v in (w1, b1, w2, b2)), x.to(dtype=dtype), _csr(dtype), q


def _directions(parameters):
    directions = [torch.sin(torch.arange(v.numel(), dtype=torch.float64) + 1).reshape(v.shape) * .1
                  for v in parameters]
    directions[0][0, :] = 0
    directions[1][0] = 0
    return directions


def _close(actual, expected):
    error = float((actual - expected).abs().max())
    bound = 2e-12 * (1 + float(expected.abs().max()))
    assert error <= bound
    return {"max_abs": error, "bound": bound}


def _snapshot(parameters, x, source, q):
    return {"parameters": [_descriptor(v) for v in parameters], "X": _descriptor(x),
            "S": _descriptor(source), "Q": _descriptor(q)}


@pytest.fixture(scope="module", autouse=True)
def _CPU_policy_and_report():
    global STARTED
    previous = (torch.get_num_threads(), torch.are_deterministic_algorithms_enabled(),
                torch.is_deterministic_algorithms_warn_only_enabled())
    if ISOLATED:
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
        assert str(torch.__version__) == "2.9.1+cu128"
        assert torch.get_default_dtype() == torch.float32
        assert not torch.cuda.is_initialized()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True, warn_only=False)
    STARTED = time.monotonic()
    rng = torch.get_rng_state().clone()
    params, x, source, q = _fixture()
    before = _snapshot(params, x, source, q)
    precision = {"threads": torch.get_num_threads(), "default_dtype": str(torch.get_default_dtype()),
                 "deterministic": torch.are_deterministic_algorithms_enabled(),
                 "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
                 "CUDA_initialized": torch.cuda.is_initialized(), "torch": str(torch.__version__)}
    REPORT["CPU_policy_before"] = precision
    REPORT["frozen_fixture_descriptors"] = before
    yield
    after_precision = {"threads": torch.get_num_threads(), "default_dtype": str(torch.get_default_dtype()),
                       "deterministic": torch.are_deterministic_algorithms_enabled(),
                       "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
                       "CUDA_initialized": torch.cuda.is_initialized(), "torch": str(torch.__version__)}
    REPORT["CPU_policy_after"] = after_precision
    REPORT["elapsed_seconds"] = time.monotonic() - STARTED
    REPORT["global_RNG_unchanged"] = torch.equal(rng, torch.get_rng_state())
    REPORT["fixture_unchanged"] = before == _snapshot(params, x, source, q)
    REPORT["precision_unchanged"] = precision == after_precision
    REPORT["interfaces"] = {name: {k: evidence[k] for k in ("operation_attempts", "counts")}
                            for name, evidence in EVIDENCES}
    REPORT["scope"] = {"real_dataset_or_PT_loads": 0, "GPU_calls": 0, "P_updates": 0,
                       "head_solves": 0, "student_fits": 0, "old_suites_rerun": 0,
                       "native_Arxiv_backend_qualified": False}
    path = os.environ.get("BX_REPORT_PATH")
    if path:
        with Path(path).open("x") as stream:
            json.dump(REPORT, stream, indent=2, allow_nan=False)
            stream.write("\n")
    assert REPORT["global_RNG_unchanged"] and REPORT["fixture_unchanged"] and REPORT["precision_unchanged"]
    assert REPORT["elapsed_seconds"] <= 60
    torch.set_num_threads(previous[0])
    torch.use_deterministic_algorithms(previous[1], warn_only=previous[2])


def test_operator_bitwise_full_vs_band_two_dtypes_five_widths_two_layouts():
    gen = torch.Generator(device="cpu").manual_seed(104)
    details = []
    for width in WIDTHS:
        base64 = torch.randn((7, 2 * width), generator=gen, dtype=torch.float64)
        for dtype in (torch.float64, torch.float32):
            base = base64.to(dtype=dtype)
            source = _csr(dtype)
            for name, value in (("contiguous", base[:, :width].contiguous()), ("stride2", base[:, ::2])):
                if name == "stride2":
                    assert value.stride(1) == 2 and not value.is_contiguous()
                before = (_descriptor(source), _descriptor(value))
                evidence = _evidence(f"operator_{width}_{dtype}_{name}")
                result = tile._product(source, value, evidence=evidence)
                expected = full._product(source, value, evidence=_evidence(f"reference_{width}_{dtype}_{name}"))
                _bit_equal(result, expected)
                assert before == (_descriptor(source), _descriptor(value))
                assert result.shape == (7, width) and not result.requires_grad
                assert evidence["counts"]["feature_band16_physical_products"] == (width + 15) // 16
                assert bool((result[1] == 0).all())
                details.append({"width": width, "dtype": str(dtype), "layout": name,
                                "input_stride": list(value.stride()), "bitwise": True,
                                "output": _descriptor(result)})
    REPORT["metrics"]["operator_bitwise"] = details


def test_nonsymmetric_transpose_coordinate_signedzero_bijection_and_empty_row():
    details = []
    for dtype in (torch.float64, torch.float32):
        source = _csr(dtype)
        before = _descriptor(source)
        transposed = tile.transpose_csr(source, _evidence(f"transpose_{dtype}"))
        expected = sorted((COLUMNS[i], row, i) for row in range(7) for i in range(CROW[row], CROW[row + 1]))
        tc = transposed.crow_indices().tolist()
        columns = transposed.col_indices().tolist()
        actual = [(row, columns[i]) for row in range(7) for i in range(tc[row], tc[row + 1])]
        assert actual == [(row, column) for row, column, _ in expected]
        _bit_equal(transposed.values(), source.values()[torch.tensor([index for _, _, index in expected])])
        assert not torch.equal(source.to_dense(), transposed.to_dense())
        assert before == _descriptor(source)
        assert CROW[1] == CROW[2]
        assert int(torch.signbit(source.values()).sum()) == 1
        details.append({"dtype": str(dtype), "transpose": _descriptor(transposed),
                        "coordinate_bijection": True, "signedzero_bitcopy": True})
    REPORT["metrics"]["transpose"] = details


def test_F64_operator_two_epsilon_FD_and_adjoint_dot_identity():
    source = _csr(torch.float64)
    gen = torch.Generator(device="cpu").manual_seed(104)
    value = None
    for width in WIDTHS:
        base = torch.randn((7, 2 * width), generator=gen, dtype=torch.float64)
        if width == 37:
            value = base[:, :width].contiguous()
    direction = torch.randn((7, 37), generator=torch.Generator(device="cpu").manual_seed(105), dtype=torch.float64)
    evidence = _evidence("operator_FD_adjoint")
    exact = tile._product(source, direction, evidence=evidence)
    fd = []
    for eps in EPSILONS:
        observed = (tile._product(source, value + eps * direction, evidence=evidence)
                    - tile._product(source, value - eps * direction, evidence=evidence)) / (2 * eps)
        error = float((observed - exact).abs().max())
        assert error <= 2e-8
        fd.append({"epsilon": eps, "max_abs": error, "bound": 2e-8})
    transposed = tile.transpose_csr(source, evidence)
    left = (tile._product(source, value, evidence=evidence) * direction).sum()
    right = (value * tile._product(transposed, direction, evidence=evidence)).sum()
    error = float((left - right).abs())
    bound = 2e-12 * (1 + max(abs(float(left)), abs(float(right))))
    assert error <= bound
    REPORT["metrics"]["operator_FD_adjoint"] = {"FD": fd, "left": float(left), "right": float(right),
                                                "dot_abs": error, "dot_bound": bound}


def test_complete_chain_all_boundaries_and_gradients_bitwise_two_dtypes():
    details = []
    for dtype in (torch.float64, torch.float32):
        params, x, source, q = _fixture(dtype)
        before = _snapshot(params, x, source, q)
        evidence = _evidence(f"chain_{dtype}")
        actual = tile.gradient_boundaries(params, x, source, q, evidence)
        expected = full.gradient_boundaries(params, x, source, q, _evidence(f"whole_chain_reference_{dtype}"))
        _bit_equal(actual["loss"], expected["loss"])
        assert actual["boundaries"].keys() == expected["boundaries"].keys()
        for key in actual["boundaries"]:
            _bit_equal(actual["boundaries"][key], expected["boundaries"][key])
        for a, b in zip(actual["gradients"], expected["gradients"]):
            _bit_equal(a, b)
        assert before == _snapshot(params, x, source, q)
        assert evidence["counts"]["feature_band16_physical_products"] == 8
        margin = float(actual["boundaries"]["A"][:, 1:].abs().min())
        assert margin >= .1
        details.append({"dtype": str(dtype), "loss": float(actual["loss"]), "margin": margin,
                        "all_boundaries_bitwise": True,
                        "boundaries": {k: _descriptor(v) for k, v in actual["boundaries"].items()},
                        "gradients": [_descriptor(v) for v in actual["gradients"]]})
    REPORT["metrics"]["complete_chain_bitwise"] = details


def test_F64_complete_chain_against_independent_dense_oracle_autograd():
    params, x, source, q = _fixture()
    actual = tile.gradient_boundaries(params, x, source, q, _evidence("dense_oracle_tile"))
    w1, b1, w2, b2 = [value.clone().requires_grad_(True) for value in params]
    dense = source.to_dense()
    u1 = x @ w1.T
    v1 = dense @ u1
    a = v1 + b1
    h = F.relu(a)
    u2 = h @ w2.T
    v2 = dense @ u2
    logits = v2 + b2
    logp = F.log_softmax(logits, dim=1)
    loss = -(q * logp).sum(1).mean()
    d, b, dh, e, c = torch.autograd.grad(loss, (logits, u2, h, v1, u1), retain_graph=True)
    gradients = torch.autograd.grad(loss, (w1, b1, w2, b2))
    expected = dict(U1=u1, V1=v1, A=a, mask=a > 0, H=h, U2=u2, V2=v2, logits=logits,
                    logp=logp, full_logp=logp, full_CE=loss, D=d, B=b, DH=dh, E=e, C=c)
    errors = {"loss": _close(actual["loss"], loss)}
    for key, value in expected.items():
        if key == "mask":
            assert torch.equal(actual["boundaries"][key], value)
        else:
            errors[key] = _close(actual["boundaries"][key], value)
    errors["gradients"] = [_close(a, b) for a, b in zip(actual["gradients"], gradients)]
    REPORT["metrics"]["independent_dense_oracle"] = errors


def test_F64_complete_CE_two_epsilon_parameter_FD_exact_masks():
    params, x, source, q = _fixture()
    direction = _directions(params)
    evidence = _evidence("complete_CE_FD")
    base = tile.gradient_boundaries(params, x, source, q, evidence)
    exact = sum((grad * change).sum() for grad, change in zip(base["gradients"], direction))
    details = []
    for eps in EPSILONS:
        plus = tile.gradient_boundaries(tuple(v + eps * d for v, d in zip(params, direction)), x, source, q, evidence)
        minus = tile.gradient_boundaries(tuple(v - eps * d for v, d in zip(params, direction)), x, source, q, evidence)
        assert torch.equal(base["boundaries"]["mask"], plus["boundaries"]["mask"])
        assert torch.equal(base["boundaries"]["mask"], minus["boundaries"]["mask"])
        observed = (plus["loss"] - minus["loss"]) / (2 * eps)
        error = abs(float(observed - exact))
        assert error <= 2e-8
        details.append({"epsilon": eps, "analytic": float(exact), "FD": float(observed),
                        "abs_error": error, "bound": 2e-8, "plus_minus_masks_exact": True})
    REPORT["metrics"]["complete_CE_FD"] = details


def test_exact_zero_ReLU_subgradient_and_four_positive_target_blocks():
    details = []
    for dtype in (torch.float64, torch.float32):
        params, x, source, q = _fixture(dtype)
        evidence = _evidence(f"zero_ReLU_target_{dtype}")
        result = tile.gradient_boundaries(params, x, source, q, evidence)
        for key in ("A", "H", "E", "C"):
            assert bool((result["boundaries"][key][:, 0] == 0).all())
        assert bool((result["gradients"][0][0] == 0).all()) and float(result["gradients"][1][0]) == 0
        packet = tile.target_packet(result, evidence)
        assert all(torch.isfinite(v) and float(v) > 0 for v in packet["source_norms"] + packet["delta"])
        rejected = dict(result, gradients=(torch.zeros_like(result["gradients"][0]), *result["gradients"][1:]))
        with pytest.raises(ValueError, match="finite positive"):
            tile.target_packet(rejected, evidence)
        details.append({"dtype": str(dtype), "zero_subgradient": True,
                        "norms": [float(v) for v in packet["source_norms"]],
                        "deltas": [float(v) for v in packet["delta"]], "zero_block_rejected": True})
    REPORT["metrics"]["zero_ReLU_positive_targets"] = details


def test_actual_logical_physical_counters_bounded_stop_and_immutability():
    params, x, source, q = _fixture()
    before = _snapshot(params, x, source, q)
    evidence = _evidence("counter_complete_chain")
    result = tile.gradient_boundaries(params, x, source, q, evidence)
    tile.target_packet(result, evidence)
    expected = {"feature_band16_original_source_products": 2, "feature_band16_explicit_transpose_products": 2,
                "feature_band16_physical_products": 8, "feature_band16_output_allocations": 4,
                "feature_band16_output_copy_calls": 8, "rank2_weighted_segment_edge_gather_calls": 8,
                "rank2_FP32_edge_multiplication_calls": 8, "rank2_segment_SUM_calls": 8,
                "transpose_CSR_constructions": 1, "transpose_bulk_index_host_copies": 2,
                "transpose_GPU_integer_uploads": 3, "transpose_GPU_value_gathers": 1,
                "source_GCN_full_forward_passes": 1, "source_CE_forward_evaluations": 2,
                "dense_logits_CE_VJP_calls": 1, "dense_second_linear_VJP_calls": 1,
                "dense_second_bias_VJP_calls": 1, "dense_first_ReLU_bias_VJP_calls": 1,
                "dense_first_linear_VJP_calls": 1, "explicit_gradient_block_outputs": 4,
                "source_gradient_block_FP64_norms": 4, "source_gradient_block_FP64_delta_values": 4}
    assert evidence["counts"] == evidence["operation_attempts"] == expected
    assert not any(value.requires_grad for value in (result["loss"], *result["gradients"], *result["boundaries"].values()))
    stopped = _evidence("stop_after_one_physical_band")
    def stop():
        _guard()
        if stopped["counts"].get("feature_band16_output_copy_calls", 0) == 1:
            raise RuntimeError("fixed stop after first band")
    stopped["_guard"] = stop
    value = torch.arange(7 * 37, dtype=torch.float64).reshape(7, 37) * .01
    value_before = _descriptor(value)
    with pytest.raises(RuntimeError, match="fixed stop"):
        tile._product(source, value, evidence=stopped)
    assert stopped["operation_attempts"]["feature_band16_logical_products"] == 1
    assert stopped["counts"].get("feature_band16_logical_products", 0) == 0
    for key in ("feature_band16_physical_products", "feature_band16_output_copy_calls",
                "rank2_weighted_segment_edge_gather_calls", "rank2_FP32_edge_multiplication_calls", "rank2_segment_SUM_calls"):
        assert stopped["counts"][key] == stopped["operation_attempts"][key] == 1
    assert before == _snapshot(params, x, source, q) and value_before == _descriptor(value)
    REPORT["metrics"]["counters_and_stop"] = {"full_chain_exact_counts": expected,
                                              "stop_logical_attempted": 1, "stop_logical_completed": 0,
                                              "completed_bands_before_stop": 1, "inputs_immutable": True}
