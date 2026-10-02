"""CPU-only prospective Qbar contract QA; no source cache or native calls."""
import ast
import copy
import hashlib
import json
import math
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from src import finite_student_outer as old
from src import finite_student_outer_v2 as new
from src.low_rank_assignment import LowRankLogits, LowRankMoments, initialize_factors
from src.moments import decode_moments, make_material

REPO = Path(__file__).resolve().parents[1]
ADAPTER = REPO / "src/citeseer_finite_student_v2.py"
PROBE = REPO / "src/finite_student_probe.py"
SCIENCE = REPO / "results/proposals/Citeseer30_120_finite_GCN_Qbar_contract_fixed25_scientific_stageAV_v1.json"


def _arguments(dtype=torch.float64, residual=0.):
    generator = torch.Generator().manual_seed(20261002)
    x = torch.rand(7, 3, generator=generator, dtype=dtype) * .3 - .15
    s = torch.eye(7, dtype=dtype) * .8 + torch.ones(7, 7, dtype=dtype) * (.2 / 7)
    h = s @ (s @ x)  # A toy identity only; real frozen H and S remain independent.
    q = torch.randn(7, 2, generator=generator, dtype=torch.float64).softmax(1)
    p = torch.randn(7, 3, generator=generator, dtype=torch.float64).softmax(1)
    transform = dict(kind="rms", matrix=None, center=torch.zeros(3, dtype=torch.float64),
                     output_center=torch.zeros(3, dtype=torch.float64), scale=torch.tensor(1.3, dtype=torch.float64))
    z = h.double() / transform["scale"]
    moments = p.T @ make_material(z, q) / 7
    moments[:, 4] += moments[:, 0] * residual
    # Positive hidden preactivations avoid ReLU boundary ambiguity in FD.
    initial = (torch.randn(5, 3, generator=generator, dtype=dtype) * .03, torch.ones(5, dtype=dtype),
               torch.randn(2, 5, generator=generator, dtype=dtype) * .08, torch.tensor([.07, -.04], dtype=dtype))
    return dict(moments=moments, transform=transform, initial_parameters=initial, source_x=x,
                source_adjacency=s, source_h=h, source_q=q)


def _equal(a, b):
    if torch.is_tensor(a):
        assert torch.is_tensor(b) and a.dtype == b.dtype and torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            _equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert type(a) is type(b) and len(a) == len(b)
        for left, right in zip(a, b, strict=True):
            _equal(left, right)
    else:
        assert a == b


def _raw_arithmetic(arguments):
    """No-validator oracle: use the original, unnormalized Qbar in every CE.

    This is an explicit arithmetic oracle, not independent proof of the model
    equations. Central differences below independently check its new dJ/dM.
    """
    leaf = arguments["moments"].detach().clone().requires_grad_(True)
    mass = leaf[:, 0]
    z = leaf[:, 1:4] / mass[:, None]
    q = leaf[:, 4:] / mass[:, None]
    transform = arguments["transform"]
    h = z * transform["scale"] + transform["output_center"] + transform["center"]
    parameters = tuple(p.detach().clone().requires_grad_(True) for p in arguments["initial_parameters"])
    states, trace = [tuple(p.detach().clone() for p in parameters)], []

    def forward(parameters, x, s=None):
        w1, b1, w2, b2 = parameters
        hidden = x @ w1.T
        if s is not None:
            hidden = s @ hidden
        hidden = F.relu(hidden + b1)
        logits = hidden @ w2.T
        if s is not None:
            logits = s @ logits
        return F.log_softmax(logits + b2, dim=1)

    for step in range(5):
        ce = -(q.to(parameters[0].dtype) * forward(parameters, h.to(parameters[0].dtype))).sum(1).mean()
        gradients = torch.autograd.grad(ce, parameters, create_graph=True)
        decay = .5 * .001 * sum(p.square().sum() for p in parameters)
        trace.append(dict(step=step, uniform_soft_ce=ce.detach().clone(), weight_decay_objective=decay.detach().clone(),
                          regularized_objective=(ce+decay).detach().clone()))
        parameters = tuple(p - .1 * (g + .001*p) for p, g in zip(parameters, gradients, strict=True))
        states.append(tuple(p.detach().clone() for p in parameters))
    losses, partials = {}, {}
    for index, (route, x, s) in enumerate((("sgc_mlp", arguments["source_h"], None),
                                          ("gcn", arguments["source_x"], arguments["source_adjacency"]))):
        loss = -(arguments["source_q"] * forward(parameters, x, s)).sum(1).mean()
        partial, = torch.autograd.grad(loss, leaf, retain_graph=index == 0)
        losses[route], partials[route] = loss.detach(), partial.detach()
    return dict(outer_losses=losses, moment_gradients=partials, inner_trace=tuple(trace),
                inner_parameter_states=tuple(states), adapted_parameters=tuple(p.detach().clone() for p in parameters),
                Hc=h.detach(), Qc=q.detach(), mass=mass.detach())


@pytest.mark.parametrize("residual", [6.56e-14, 1.54e-13])
def test_measured_scale_derived_contract_accepts_original_values(residual):
    arguments = _arguments(residual=residual)
    expected = arguments["moments"][:, 4:] / arguments["moments"][:, :1]
    assert 32*torch.finfo(torch.float64).eps < float((expected.sum(1)-1).abs().max()) < 1e-12
    with pytest.raises(ValueError, match="source-derived Qbar"):
        old.decode_rms_moments(arguments["moments"], 3, arguments["transform"])
    _, actual, _ = new.decode_rms_moments(arguments["moments"], 3, arguments["transform"])
    assert torch.equal(actual, expected) and not torch.equal(actual, actual/actual.sum(1, keepdim=True))
    with pytest.raises(ValueError, match="source-derived Qbar"):
        old.finite_student_outer_partials(**arguments)
    result = new.finite_student_outer_partials(**arguments)
    oracle = _raw_arithmetic(arguments)
    for key in oracle:
        _equal(result[key], oracle[key])
    assert result["policy"]["probability_validation"]["renormalization"] is False


@pytest.mark.parametrize("malformed", ["sum", "negative", "nan", "inf"])
def test_derived_outside_contract_rejects_without_mutation(malformed):
    arguments = _arguments()
    moments = arguments["moments"]
    if malformed == "sum":
        moments[:, 4] += moments[:, 0]*2e-12
    else:
        moments[0, 4] = dict(negative=-.1, nan=float("nan"), inf=float("inf"))[malformed]
    before = moments.clone()
    with pytest.raises(ValueError):
        new.decode_rms_moments(moments, 3, arguments["transform"])
    torch.testing.assert_close(moments, before, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_original_source_probability_guard_remains_32eps(dtype):
    value = torch.tensor([[.4, .6]], dtype=dtype)
    value[0, 0] += 64*torch.finfo(dtype).eps
    for engine in (old, new):
        with pytest.raises(ValueError, match="rows must sum"):
            engine._probabilities(value, "frozen teacher Q")


@pytest.mark.parametrize("malformed", ["measured_residual", "negative", "nan", "inf"])
def test_engine_source_Q_strict_even_when_derived_contract_passes(malformed):
    arguments = _arguments(residual=1.54e-13)
    q = arguments["source_q"]
    if malformed == "measured_residual":
        q[:, 0] += 1.54e-13
    else:
        q[0, 0] = dict(negative=-.1, nan=float("nan"), inf=float("inf"))[malformed]
    with pytest.raises(ValueError, match="frozen teacher Q"):
        new.finite_student_outer_partials(**arguments)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_nominal_accepted_outputs_states_and_cotangents_bitwise_same(dtype):
    arguments = _arguments(dtype=dtype)
    before = copy.deepcopy(arguments)
    first = old.finite_student_outer_partials(**arguments)
    second = new.finite_student_outer_partials(**arguments)
    added_policy = second["policy"].pop("probability_validation")
    assert added_policy["derived_FP64_Qbar_absolute_row_sum_tolerance"] == 1e-12
    _equal(first, second)
    _equal(arguments, before)


@pytest.mark.parametrize("route", ["sgc_mlp", "gcn"])
@pytest.mark.parametrize("epsilon", [1e-5, 2e-6])
def test_derived_contract_full_unroll_directional_FD(route, epsilon, record_property):
    arguments = _arguments(residual=1.54e-13)
    direction = torch.zeros_like(arguments["moments"])
    direction[:, 1:4] = torch.tensor([[.08, -.04, .03], [-.03, .07, .01], [.02, -.03, -.06]], dtype=torch.float64)
    direction[:, 4:] = torch.tensor([[.04, -.04], [-.03, .03], [.02, -.02]], dtype=torch.float64)
    result = new.finite_student_outer_partials(**arguments)
    predicted = float((result["moment_gradients"][route]*direction).sum())
    plus = new.finite_student_outer_partials(**dict(arguments, moments=arguments["moments"]+epsilon*direction))
    minus = new.finite_student_outer_partials(**dict(arguments, moments=arguments["moments"]-epsilon*direction))
    measured = float((plus["outer_losses"][route]-minus["outer_losses"][route])/(2*epsilon))
    record_property("absolute_error", abs(predicted-measured))
    record_property("predicted", predicted)
    record_property("central_FD", measured)
    assert abs(predicted-measured) <= 2e-8 + 2e-5*abs(measured)


def _function(path, name):
    return next(node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef) and node.name == name)


def _dump(node):
    return ast.dump(node, include_attributes=False)


def test_only_engine_guard_and_policy_changed_AST():
    original = ast.parse(Path(old.__file__).read_text())
    changed = ast.parse(Path(new.__file__).read_text())
    for before, after in zip(original.body, changed.body, strict=True):
        if isinstance(before, ast.FunctionDef) and before.name == "_probabilities":
            after.args = copy.deepcopy(before.args)
            index = next(i for i, node in enumerate(after.body) if isinstance(node, ast.If)
                         and _dump(node.test) == _dump(ast.parse("tolerance is None", mode="eval").body))
            after.body[index] = after.body[index].body[0]
        if isinstance(before, ast.FunctionDef) and before.name == "decode_rms_moments":
            for node in ast.walk(after):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_probabilities":
                    expected = ast.parse("1e-12 if qbar.dtype == torch.float64 else None", mode="eval").body
                    assert len(node.keywords) == 1 and node.keywords[0].arg == "tolerance"
                    assert _dump(node.keywords[0].value) == _dump(expected)
                    node.keywords = []
        if isinstance(before, ast.FunctionDef) and before.name == "finite_student_outer_partials":
            removed = 0
            for node in ast.walk(after):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "dict":
                    removed += sum(key.arg == "probability_validation" for key in node.keywords)
                    node.keywords = [key for key in node.keywords if key.arg != "probability_validation"]
            assert removed == 1
        assert _dump(before) == _dump(after)


def test_FP32_decoder_retains_original_32eps_guard():
    arguments = _arguments()
    moments = arguments["moments"].float()
    moments[:, 4] += moments[:, 0]*1e-6
    transform = {key: value.float() if torch.is_tensor(value) else value for key, value in arguments["transform"].items()}
    residual = float((moments[:, 4:].sum(1)/moments[:, 0]-1).abs().max())
    assert 1e-12 < residual < 32*torch.finfo(torch.float32).eps
    _equal(old.decode_rms_moments(moments, 3, transform), new.decode_rms_moments(moments, 3, transform))


def test_local_evaluator_copies_original_probe_arithmetic_AST():
    reference, actual = _function(PROBE, "_evaluate"), _function(ADAPTER, "_evaluate")
    actual.body = actual.body[3:-2] + actual.body[-1:]  # Remove step/attempt/stop prefix; restore original stop below.
    actual.body.insert(0, copy.deepcopy(reference.body[0]))
    # The common body uses explicit probe-qualified helpers instead of copied globals.
    class Unqualify(ast.NodeTransformer):
        def visit_Attribute(self, node):
            if isinstance(node.value, ast.Name) and node.value.id == "probe":
                return ast.Name(id=node.attr, ctx=node.ctx)
            return self.generic_visit(node)
    actual = Unqualify().visit(actual)
    assert _dump(reference) == _dump(actual)
    assert "probe._evaluate" not in ast.unparse(_function(ADAPTER, "_evaluate"))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _controls():
    tree = ast.parse(ADAPTER.read_text())
    names = {"SCHEMA", "HORIZON", "SCIENCE", "SCIENCE_SHA", "SOURCE_FIELD", "_FIXED", "_SEEDS", "_SETTINGS"}
    body = [node for node in tree.body if isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id in names for target in node.targets)]
    body.append(_function(ADAPTER, "canonical_candidate"))
    namespace = dict(__file__=str(ADAPTER), _require=_require, math=math,
                     numerical_source=lambda: {"frozen_v2": True}, _seal=lambda value: hashlib.sha256(json.dumps(value).encode()).hexdigest())
    exec(compile(ast.Module(body=body, type_ignores=[]), str(ADAPTER), "exec"), namespace)
    return namespace


@pytest.mark.parametrize("change", [dict(method="finite_student_citeseer"), dict(finite_student_schema=1),
                                  dict(finite_student_schema=True), dict(external_Qbar_tolerance=.01),
                                  dict(renormalize=True), dict(finite_student_source_digest="old_namespace")])
def test_v2_explicit_namespace_rejects_old_or_external_controls(change):
    namespace = _controls()
    with pytest.raises(ValueError):
        namespace["canonical_candidate"](dict(namespace["_FIXED"], **change))


def test_candidate_science_and_shared_P0_filesystem_namespace():
    namespace = _controls()
    assert namespace["_FIXED"] == json.loads(SCIENCE.read_text())["candidate"]
    assert namespace["SCHEMA"] == 2 and namespace["SCIENCE_SHA"] == hashlib.sha256(SCIENCE.read_bytes()).hexdigest()
    canonical = namespace["canonical_candidate"](namespace["_FIXED"])
    assert namespace["canonical_candidate"](canonical) == canonical
    assert '"citeseer_finite_schema2_validation"' in ADAPTER.read_text()
    assert '"citeseer_finite_schema1_validation"' not in ADAPTER.read_text()
    manifest = _function(ADAPTER, "numerical_source")
    assert ast.unparse(manifest.body[0]) == "result = probe.numerical_source()"
    assert "finite_student_outer_v2.py" in ast.unparse(manifest)
    assert "citeseer_finite_student_v2.py" in ast.unparse(manifest)


@pytest.mark.parametrize("action", ["prepare", "certify", "validate"])
def test_actual_v2_worker_dispatch_forwards_options_callback_and_result(action, monkeypatch):
    fake = ModuleType("src.citeseer_finite_student_v2")
    calls, returned = [], object()
    def callback(**options):
        calls.append(options)
        return returned
    setattr(fake, action, callback)
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    function = _function(REPO / "src/research_loop.py", "dispatch")
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "<actual_worker_AST>", "exec"), namespace)
    stop = lambda: False
    options = dict(cells=30, candidate={"fixed": True}, spec_path="frozen", gate_sha256=None)
    assert namespace["dispatch"]({"kind": f"citeseer_finite_student_v2_{action}", "options": options}, stop) is returned
    assert calls == [dict(options, stop=stop)]
    assert "stop" not in options


def test_toy_adapter_local_evaluator_matches_original_snapshot_arithmetic():
    arguments = _arguments(dtype=torch.float32)
    hard = torch.tensor([0, 1, 2, 0, 1, 2, 0])
    u, v = initialize_factors(hard, 3, 2, 0)
    buffers = dict(z=arguments["source_h"].double()/1.3, q=arguments["source_q"], hard=hard,
                   transform=arguments["transform"], model_initial=arguments["initial_parameters"],
                   x=arguments["source_x"], S=arguments["source_adjacency"], h=arguments["source_h"], counts={})
    def digest(value):
        return hashlib.sha256(value.detach().contiguous().numpy().tobytes()).hexdigest()
    def cpu_state(value):
        if torch.is_tensor(value):
            return value.detach().clone()
        if isinstance(value, dict):
            return {key: cpu_state(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return type(value)(cpu_state(item) for item in value)
        return value
    helpers = dict(_runtime_precision_guard=lambda: None,
                   _moments=lambda b, p: LowRankMoments.apply(p[0], p[1], b["hard"], make_material(b["z"], b["q"]), .05, 4096),
                   LowRankLogits=LowRankLogits, decode_moments=decode_moments, make_material=make_material,
                   _digest=digest, cpu_state=cpu_state)
    common = dict(torch=torch, math=math, SCHEMA=2, HORIZON=25, _require=_require,
                  _stop=lambda stop: _require(not stop(), "stopped"), _scalar=lambda value, message: _require(math.isfinite(value), message))
    original = dict(common, **helpers, finite_student_outer_partials=old.finite_student_outer_partials)
    current = dict(common, probe=SimpleNamespace(**helpers), finite_student_outer_partials=new.finite_student_outer_partials,
                   _count=lambda e, key, complete=False: e["counts"].__setitem__(key+str(complete), e["counts"].get(key+str(complete), 0)+1))
    for path, namespace in ((PROBE, original), (ADAPTER, current)):
        exec(compile(ast.Module(body=[_function(path, "_evaluate")], type_ignores=[]), str(path), "exec"), namespace)
    expected, expected_scale = original["_evaluate"](buffers, {"outer_route": "gcn"}, {}, [u, v], 0, None, lambda: False)
    actual, actual_scale = current["_evaluate"](buffers, {"outer_route": "gcn"}, {}, [u, v], 0, None, lambda: False)
    _equal(expected, actual)
    assert actual_scale == expected_scale
    assert buffers["counts"] == {"finite_objectiveFalse": 1, "finite_objectiveTrue": 1}
