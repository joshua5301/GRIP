"""Eight CPU-only row-KL proof domains for the activated production implementation.

No dataset, saved tensor, CUDA, student, source-gradient or old-suite execution.
The root freezes source/test bytes externally before the sole new proof run.
"""
import ast
import builtins
import copy
import hashlib
import math
from numbers import Real
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src/soft_ce_partition.py").is_file())
SOURCE = ROOT / "src"
LEGACY_HEAD_AST_SHA = {'head_gradient': 'ca243f7e53a7e62d5af6f381716c7cd5103ee1c3dcb85716a6387d46f30dddfa', 'solve_head_system': '62693c266ecf4368f6bcdf23b0775ddf03b119a2152ba1fa8e1a11780055773d', 'solve_inner_newton_first': '381507dcd96076f390ca7fa2de5459e570f552996c6805db57ae440aa9477cb0', 'outer_value_gradient': '11a9e84d7c07457344b467c22f7f4d0b1dcc4395eb39190e707600cd8cdb34f6', 'implicit_moment_gradient': '69724073decdd557cafdb6e1d60f217915ccab80f03af49a062aeb591b0a824f'}
LEGACY_RESUME_CONFIG_AST_SHA = 'f17230780e9cf5f969ecbbbf70899ffbee1382b22e4bf8d084aa74b75e073dd4'
LEGACY_OPERATION_ALIAS_AST_SHA = 'c9d23da3a5284efc2a863aec0d97421d2b8035d59b344e81987b7317f9b94011'
LEGACY_OPERATION_CALL_AST_SHA = '53fb5ba20c3adfb029ff00abaa0cd626756849f9546891e77496972b82d49824'
SIGNATURE_SHA = "b109a20373f1dc0a94362ff0ec9ffd254b76cceac2ebba7f9dc7b24f393e336d"
EPS = (1e-5, 3e-6)
VALUE_ATOL = 2e-14
GRAD_ATOL, GRAD_RTOL = 2e-12, 2e-10
FD_ATOL, FD_RTOL = 2e-9, 2e-6
PENALTY, NEWTON_STEPS, STATIONARITY = .2, 16, 1e-11


def _observed_descriptor(value):
    # Independent byte observation using the existing documented array grammar.
    array = value.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256(str((array.shape, array.dtype.str)).encode() + array.tobytes()).hexdigest()
    return dict(shape=list(array.shape), dtype=str(value.dtype), tensor=digest)


import pytest
import torch

from src import initial_assignment_row_kl as helper
from src import soft_ce_partition as legacy
from src.low_rank_assignment import LowRankMoments, initialize_factors, logit_block
from src.moments import augmented, decode_moments, initial_logits, make_material


def fixture():
    z = torch.tensor([[.25, -.5], [-.125, .25], [.5, .125], [-.25, .5],
                      [.75, -.125], [.125, .75], [-.5, -.25]], dtype=torch.float64)
    q = torch.tensor([[.75, .125, .125], [.625, .25, .125], [.75, .125, .125],
                      [.125, .75, .125], [.125, .625, .25], [.125, .125, .75],
                      [.25, .125, .625]], dtype=torch.float64)
    hard = torch.tensor([0, 0, 0, 1, 1, 2, 2], dtype=torch.int64)
    u = torch.tensor([[.1, -.2], [-.15, .05], [.2, .1], [-.1, .25],
                      [.05, -.15], [.125, .2], [-.2, -.1]], dtype=torch.float64)
    v = torch.tensor([[.2, -.3], [-.25, .15], [.1, .25]], dtype=torch.float64)
    return z, q, hard, u, v


def _base(hard, cells, dtype):
    # Independent literal original fixed logit-prior formula, no active helper.
    base = torch.full((len(hard), cells), math.log(.05 / cells), dtype=dtype)
    base.scatter_(1, hard[:, None], math.log(.95 + .05 / cells))
    return base


def _dense(u, v, hard, material, logp0):
    logits = (_base(hard, len(v), u.dtype) + u @ v.T / math.sqrt(u.shape[1])).to(material.dtype)
    p = logits.softmax(1)
    logp = logits.log_softmax(1)
    moments = p.T @ material / len(p)
    row_kl = (p * (logp - logp0)).sum() / len(p)
    return moments, row_kl


def _reference_logp(hard, v):
    return (_base(hard, len(v), torch.float64)
            + torch.zeros(len(hard), v.shape[1], dtype=torch.float64) @ v.T / math.sqrt(v.shape[1])).log_softmax(1).detach()


def _direction(value):
    d = torch.arange(1, value.numel() + 1, dtype=value.dtype).reshape_as(value)
    d = d - d.mean()
    return d / d.norm()


def _assert_fd(prediction, values, eps, record_property, label):
    measured = float((values[0] - values[1]) / (2 * eps))
    error = abs(float(prediction) - measured)
    record_property(label, error)
    assert error <= FD_ATOL + FD_RTOL * abs(measured)


def _newton(moments, differentiable=False):
    # Independent dense convex head oracle. No original head/adjoint solver.
    centers = moments[:, 1:3] / moments[:, :1]
    labels = moments[:, 3:] / moments[:, :1]
    x = torch.cat((centers, torch.ones(len(centers), 1, dtype=centers.dtype)), 1)
    theta = moments.new_zeros(3, 3).requires_grad_()

    def inner(t):
        return -(labels * (x @ t.T).log_softmax(1)).sum() / len(labels) + PENALTY * t.square().sum() / 2

    for _ in range(NEWTON_STEPS):
        g, = torch.autograd.grad(inner(theta), theta, create_graph=differentiable)
        hessian = torch.autograd.functional.hessian(inner, theta, create_graph=differentiable).reshape(9, 9)
        theta = theta - torch.linalg.solve(hessian, g.flatten()).reshape_as(theta)
        if not differentiable:
            theta = theta.detach().requires_grad_()
    residual, = torch.autograd.grad(inner(theta), theta, retain_graph=True)
    assert float(residual.abs().max()) <= STATIONARITY
    return theta


def _teacher_value(moments, z, q, differentiable=False):
    theta = _newton(moments, differentiable)
    x = torch.cat((z, torch.ones(len(z), 1, dtype=z.dtype)), 1)
    return -(q * (x @ theta.T).log_softmax(1)).sum() / len(q)


def _ce_partial(moments, z, q):
    theta = _newton(moments.detach()).detach()
    value, rhs = legacy.outer_value_gradient(z, q, theta)
    centers, labels, mass = decode_moments(moments.detach(), 2)
    vector, diagnostic = legacy.solve_head_system(
        augmented(centers), labels, torch.full_like(mass, 1 / len(mass)), theta,
        PENALTY, rhs, rtol=1e-11, atol=1e-14,
    )
    assert diagnostic["cg_converged"]
    partial = legacy.implicit_moment_gradient(moments, 2, theta, vector, PENALTY, "uniform")
    return value, partial


def _native_reference():
    z, q, hard, _, _ = fixture()
    u0, v0 = initialize_factors(hard, 3, 2, seed=17)
    assert u0.dtype == v0.dtype == torch.float32
    reference, context = helper.original_reference(
        z, q, hard, 2, 17, .05, 3, {}, helper.source_digest(), parameters=(u0, v0)
    )
    return z, q, hard, u0, v0, reference, context


def test_domain1_exact_logit_KL_cotangent_and_invariances(record_property):
    z, q, hard, u, v = fixture()
    material = make_material(z, q)
    logp0 = _reference_logp(hard, v)
    a = (_base(hard, 3, u.dtype) + u @ v.T / math.sqrt(2)).detach().requires_grad_()
    p, logp = a.softmax(1), a.log_softmax(1)
    literal = (p * (logp - logp0)).sum() / len(p)
    dense, = torch.autograd.grad(literal, a)
    ell = logp.detach() - logp0
    formula = p.detach() * (ell - (p.detach() * ell).sum(1, keepdim=True)) / len(p)
    torch.testing.assert_close(formula, dense, atol=GRAD_ATOL, rtol=GRAD_RTOL)
    # U spans arbitrary logit perturbations when V=sqrt(K)*I; thus actual fused
    # dR/dU is independently checked against the exact row-logit formula.
    delta = (a.detach() - _base(hard, 3, a.dtype)).requires_grad_()
    identity = torch.eye(3, dtype=a.dtype) * math.sqrt(3)
    _, r = helper.InitialAssignmentKLMoments.apply(delta, identity, hard, material, .05, 3, logp0)
    actual, = torch.autograd.grad(r, delta)
    torch.testing.assert_close(actual, formula, atol=GRAD_ATOL, rtol=GRAD_RTOL)
    assert float(r) == pytest.approx(float(literal), abs=VALUE_ATOL)
    torch.testing.assert_close(actual.sum(1), torch.zeros(7, dtype=a.dtype), atol=GRAD_ATOL, rtol=0)
    shifted = delta.detach() + torch.tensor([.3, -.2, .7, -.4, .1, -.6, .2], dtype=a.dtype)[:, None]
    _, shifted_r = helper.InitialAssignmentKLMoments.apply(shifted, identity, hard, material, .05, 3, logp0)
    assert float(shifted_r) == pytest.approx(float(r), abs=VALUE_ATOL)
    order = torch.tensor([2, 0, 1])
    inverse = torch.argsort(order)
    _, permuted_r = helper.InitialAssignmentKLMoments.apply(
        delta.detach()[:, order], identity, inverse[hard], material, .05, 3, logp0[:, order]
    )
    assert float(permuted_r) == pytest.approx(float(r), abs=VALUE_ATOL)
    record_property("actual_logit_cotangent_max_abs", float((actual - dense).abs().max()))
    record_property("one_over_N_not_NK", True)


def test_domain2_streamed_moments_KL_UV_VJP_and_two_epsilon_FD(record_property):
    z, q, hard, u, v = fixture()
    material = make_material(z, q)
    logp0 = _reference_logp(hard, v)
    before = [x.clone() for x in (material, hard, logp0)]
    d = torch.tensor([[.2, -.3, .1, .125, -.15, .05],
                      [-.1, .25, -.2, -.075, .1, .15],
                      [.15, -.05, .3, .2, -.125, -.05]], dtype=torch.float64)
    s0 = .7  # Fixed cotangent-scale fixture, no fitted/search parameter.
    du, dv = _direction(u), _direction(v)
    for chunk in (1, 3, 7):
        au, av = u.clone().requires_grad_(), v.clone().requires_grad_()
        amaterial = material.clone().requires_grad_()
        m, r = helper.InitialAssignmentKLMoments.apply(au, av, hard, amaterial, .05, chunk, logp0)
        actual = torch.autograd.grad((m, r), (au, av, amaterial), grad_outputs=(d / s0, r.new_tensor(1 / s0)))
        ou, ov = u.clone().requires_grad_(), v.clone().requires_grad_()
        omaterial = material.clone().requires_grad_()
        dm, dr = _dense(ou, ov, hard, omaterial, logp0)
        expected = torch.autograd.grad(((dm * d).sum() + dr) / s0, (ou, ov, omaterial))
        torch.testing.assert_close(m, dm, atol=VALUE_ATOL, rtol=GRAD_RTOL)
        assert float(r) == pytest.approx(float(dr), abs=VALUE_ATOL)
        for a, b in zip(actual, expected, strict=True):
            torch.testing.assert_close(a, b, atol=GRAD_ATOL, rtol=GRAD_RTOL)
        prediction = (actual[0] * du).sum() + (actual[1] * dv).sum()
        for eps in EPS:
            values = []
            for sign in (1, -1):
                fm, fr = _dense(u + sign * eps * du, v + sign * eps * dv, hard, material, logp0)
                values.append(((fm * d).sum() + fr) / s0)
            _assert_fd(prediction, values, eps, record_property, f"joint_MR_UV_chunk{chunk}_FD_{eps}")
        dmaterial = _direction(material)
        for eps in EPS:
            values = []
            for sign in (1, -1):
                fm, fr = _dense(u, v, hard, material + sign * eps * dmaterial, logp0)
                values.append(((fm * d).sum() + fr) / s0)
            _assert_fd((actual[2] * dmaterial).sum(), values, eps, record_property,
                       f"material_cotangent_chunk{chunk}_FD_{eps}")
    assert all(torch.equal(x, y) for x, y in zip(before, (material, hard, logp0), strict=True))


def test_domain3_actual_initial_reference_zero_R_and_native_M_order(record_property):
    z, q, hard, u0, v0, reference, context = _native_reference()
    material = make_material(z, q)
    logp0 = reference["log_probability"]
    assert reference["schema"] == 1 and not logp0.requires_grad and logp0.dtype == torch.float64
    assert all(torch.equal(a, b.detach()) for a, b in zip(reference["initial_parameters"], (u0, v0), strict=True))
    before = [x.clone() for x in (material, hard, u0, v0, logp0)]
    exact_a0 = logit_block(u0.detach(), v0.detach(), hard, .05)
    assert torch.equal(logp0, exact_a0.double().log_softmax(1))
    observed = dict(initial_U0=u0, initial_V0=v0, hard_assignment=hard,
                    hard_logit_prior=initial_logits(hard, 3, .05, u0.dtype),
                    initial_residual=u0 @ v0.T / math.sqrt(2), actual_A0_FP32=exact_a0,
                    log_P0=logp0, actual_P0=exact_a0.double().softmax(1))
    assert all(context[key] == _observed_descriptor(value) for key, value in observed.items())
    m, r = helper.InitialAssignmentKLMoments.apply(u0, v0, hard, material, .05, 3, logp0)
    original = LowRankMoments.apply(u0, v0, hard, material, .05, 3)
    assert torch.equal(m, original), "Active moments changed original native reduction/cast order"
    assert float(r) == 0.0
    kl_grad = torch.autograd.grad(r, (u0, v0))
    assert all(torch.equal(x, torch.zeros_like(x)) for x in kl_grad)
    assert all(torch.equal(x, y) for x, y in zip(before, (material, hard, u0, v0, logp0), strict=True))
    assert isinstance(context, dict) and context
    record_property("native_initial_M_exact", True)
    record_property("native_initial_R_exact", float(r))
    record_property("native_initial_KL_UV_grad_exact_zero", True)


def test_domain4_actual_first_uniform_CE_Adam_step_parity(record_property):
    z, q, hard, u0, v0, reference, _ = _native_reference()
    material = make_material(z, q)
    bu, bv = u0.detach().clone().requires_grad_(), v0.detach().clone().requires_grad_()
    au, av = u0.detach().clone().requires_grad_(), v0.detach().clone().requires_grad_()
    bm = LowRankMoments.apply(bu, bv, hard, material, .05, 3)
    am, ar = helper.InitialAssignmentKLMoments.apply(au, av, hard, material, .05, 3, reference["log_probability"])
    assert torch.equal(bm, am) and float(ar) == 0.0
    f0, ce_partial = _ce_partial(bm, z, q)
    s0 = max(f0, 1e-12)
    baseline = torch.optim.Adam((bu, bv), lr=.01, eps=1e-12, foreach=False)
    activated = torch.optim.Adam((au, av), lr=.01, eps=1e-12, foreach=False)
    bm.backward(ce_partial / s0)
    torch.autograd.backward((am, ar), (ce_partial / s0, ar.new_tensor(1 / s0)))
    assert torch.equal(bu.grad, au.grad) and torch.equal(bv.grad, av.grad)
    baseline.step()
    activated.step()
    assert torch.equal(bu, au) and torch.equal(bv, av), "First nativecast update parity must be observed"
    for old, new in ((bu, au), (bv, av)):
        for key in ("step", "exp_avg", "exp_avg_sq"):
            assert torch.equal(baseline.state[old][key], activated.state[new][key])
    record_property("baseline_active_first_Adam_exact", True)
    record_property("one_outside_backward_per_path_one_update", True)
    record_property("fixed_teacherF0_scale", s0)


def test_domain5_noninitial_implicit_plus_direct_joint_Newton_oracle_FD(record_property):
    z, q, hard, u, v = fixture()
    material = make_material(z, q)
    logp0 = _reference_logp(hard, v)
    initial_m, initial_r = _dense(torch.zeros_like(u), v, hard, material, logp0)
    assert float(initial_r) == 0.0
    s0 = max(float(_teacher_value(initial_m, z, q)), 1e-12)
    au, av = u.clone().requires_grad_(), v.clone().requires_grad_()
    m, r = helper.InitialAssignmentKLMoments.apply(au, av, hard, material, .05, 3, logp0)
    _, ce_partial = _ce_partial(m, z, q)
    actual = torch.autograd.grad((m, r), (au, av), grad_outputs=(ce_partial / s0, r.new_tensor(1 / s0)))
    ou, ov = u.clone().requires_grad_(), v.clone().requires_grad_()
    oracle_m, oracle_r = _dense(ou, ov, hard, material, logp0)
    oracle = (_teacher_value(oracle_m, z, q, True) + oracle_r) / s0
    expected = torch.autograd.grad(oracle, (ou, ov))
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, atol=2e-10, rtol=2e-8)
    du, dv = _direction(u), _direction(v)
    prediction = (actual[0] * du).sum() + (actual[1] * dv).sum()
    for eps in EPS:
        values = []
        for sign in (1, -1):
            fm, fr = _dense(u + sign * eps * du, v + sign * eps * dv, hard, material, logp0)
            values.append((_teacher_value(fm, z, q) + fr) / s0)
        _assert_fd(prediction, values, eps, record_property, f"new_implicit_plus_direct_FD_{eps}")
    assert all(float(x.norm()) > 0 for x in actual)
    record_property("new_joint_F0_scale_frozen_not_R0", s0)


def _active_candidate():
    return helper.candidate_controls(dict(method="low_rank", width=0, lr=.01, T=1., rank=2,
                                         penalty=.2, initialization="teacher_balanced", alpha=.3,
                                         inner_loss_weighting="uniform", assignment_kl_weight=1.,
                                         assignment_kl_schema=1))


def test_domain6_reference_schema_support_lineage_and_retention_guards():
    z, q, hard, u0, v0, reference, context = _native_reference()
    assert not reference["log_probability"].requires_grad
    assert all(not p.requires_grad for p in reference["initial_parameters"])
    assert bool(reference["log_probability"].isfinite().all())
    assert bool(reference["log_probability"].exp().gt(0).all())
    material = make_material(z, q)
    before = [x.clone() for x in (u0, v0, hard, material, reference["log_probability"])]
    for bad_logp in (reference["log_probability"].float(), reference["log_probability"] * float("nan"),
                     reference["log_probability"][:, :2], reference["log_probability"].clone().requires_grad_()):
        with pytest.raises((ValueError, FloatingPointError)):
            helper.InitialAssignmentKLMoments.apply(u0, v0, hard, material, .05, 3, bad_logp)
    for bad_u, bad_v in ((u0 * float("inf"), v0), (u0, v0 * float("nan"))):
        with pytest.raises((ValueError, FloatingPointError)):
            helper.InitialAssignmentKLMoments.apply(bad_u, bad_v, hard, material, .05, 3, reference["log_probability"])
    assert all(torch.equal(x, y) for x, y in zip(before, (u0, v0, hard, material, reference["log_probability"]), strict=True))
    assert isinstance(context, dict)


def test_domain7_active_resume_checkpoint_context_and_forwarding_guards():
    z, q, hard, _, _ = fixture()
    candidate = _active_candidate()
    config = helper.expected_citation_config(candidate, 17, z, q, hard)
    reference, context = helper.original_reference(
        z, q, hard, 2, 17, .05, config["chunk_size"], config, helper.source_digest()
    )
    checkpoint = dict(step=1, teacher_ce=.5, assignment_kl_KL=.025, objective=.525)
    helper.attach(checkpoint, reference, context, config)
    resume = dict(step=1, config=config, snapshots={1: checkpoint})
    helper.attach(resume, reference, context, config)
    assert helper.validate_cached(checkpoint, candidate, z, q, hard, 17, 25, step=1) is checkpoint
    assert helper.validate_cached(resume, candidate, z, q, hard, 17, 25, resume=True) is resume
    for mutation in ({"assignment_kl_weight": .5}, {"assignment_kl_weight": True},
                     {"assignment_kl_schema": True}, {"assignment_kl_source_digest": "stale"},
                     {"assignment_kl_typo": 1}, {"train_target_mix": .25},
                     {"method": "nystrom"}, {"T": .3}, {"inner_loss_weighting": "mass"}):
        with pytest.raises(ValueError):
            helper.candidate_controls(dict(candidate, **mutation))
    for changed_z, changed_q, changed_hard in ((z + .125, q, hard), (z, q[:, [1, 0, 2]], hard),
                                              (z, q, (hard + 1) % 3)):
        with pytest.raises(ValueError):
            helper.validate_cached(resume, candidate, changed_z, changed_q, changed_hard, 17, 25, resume=True)
    for key in ("assignment_kl_reference", "assignment_kl_context", "assignment_kl_config"):
        changed = copy.deepcopy(checkpoint)
        changed.pop(key)
        with pytest.raises(ValueError):
            helper.validate_cached(changed, candidate, z, q, hard, 17, 25, step=1)
    changed = copy.deepcopy(resume)
    changed["assignment_kl_reference"]["log_probability"][:, [0, 1]] = changed["assignment_kl_reference"]["log_probability"][:, [1, 0]]
    with pytest.raises(ValueError):
        helper.validate_cached(changed, candidate, z, q, hard, 17, 25, resume=True)
    with pytest.raises(ValueError):
        helper.validate_cached(dict(checkpoint, step=True), candidate, z, q, hard, 17, 25, step=1)
    core = ast.parse((SOURCE / "soft_ce_partition.py").read_text())
    optimize = next(n for n in core.body if isinstance(n, ast.FunctionDef) and n.name == "optimize_ce_assignment")
    validation = next(n for n in ast.walk(optimize) if isinstance(n, ast.Call)
                      and isinstance(n.func, ast.Name) and n.func.id == "validate_row_kl_resume")
    saved = next(n for n in ast.walk(optimize) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "saved_reference" for t in n.targets))
    assert ast.unparse(saved.value) == "resume_state['assignment_kl_reference']"
    consume = next(n for n in ast.walk(optimize) if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "row_kl_reference" for t in n.targets)
                   and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name)
                   and n.value.func.id == "dict")
    assert validation.lineno < saved.lineno < consume.lineno
    logp = next(k.value for k in consume.value.keywords if k.arg == "log_probability")
    assert ast.unparse(logp) == "saved_reference['log_probability'].detach().to(device=z.device).clone()"
    copies = [n for n in ast.walk(consume) if isinstance(n, ast.Call)
              and isinstance(n.func, ast.Attribute) and n.func.attr == "to"]
    assert len(copies) == 2
    assert all(not n.args and len(n.keywords) == 1 and n.keywords[0].arg == "device"
               and ast.unparse(n.keywords[0].value) == "z.device" for n in copies)
    parameter_load = next(n for n in ast.walk(optimize) if isinstance(n, ast.Call)
                          and isinstance(n.func, ast.Attribute) and n.func.attr == "copy_"
                          and ast.unparse(n.func.value) == "parameter")
    assert consume.lineno < parameter_load.lineno
    operator = next(n for n in ast.walk(optimize) if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name)
                    and n.func.value.id == "InitialAssignmentKLMoments" and n.func.attr == "apply")
    assert ast.unparse(operator.args[-1]) == "row_kl_reference['log_probability']"
    public = ast.parse((SOURCE / "citation_search.py").read_text())
    aliases = {a.asname or a.name for n in ast.walk(public) if isinstance(n, ast.ImportFrom)
               and n.module == "src.initial_assignment_row_kl" for a in n.names if a.name == "validate_cached"}
    run = next(n for n in public.body if isinstance(n, ast.FunctionDef) and n.name == "run_screen")
    guard = next(n for n in ast.walk(run) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                 and n.func.id in aliases and any(k.arg == "resume" for k in n.keywords))
    skip = next(n for n in ast.walk(run) if isinstance(n, ast.If) and "state['step'] < steps" in ast.unparse(n.test))
    assert guard.lineno < skip.lineno
    for name in ("run_screen", "selected_test"):
        fn = next(n for n in public.body if isinstance(n, ast.FunctionDef) and n.name == name)
        assert any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in aliases
                   and any(k.arg == "step" for k in n.keywords) for n in ast.walk(fn))


def test_domain8_default_zero_signature_and_legacy_path_AST(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Disabled default evaluated new reference/source algebra")

    monkeypatch.setattr(helper, "source_digest", forbidden)
    monkeypatch.setattr(helper, "original_reference", forbidden)
    base = dict(method="low_rank", width=0, lr=.01, T=1., rank=2, penalty=.2,
                initialization="teacher_balanced", alpha=.3, inner_loss_weighting="uniform")
    assert helper.candidate_controls(base) == base and helper.core_options(base) == {}
    assert helper.candidate_controls(dict(base, assignment_kl_weight=0.)) == base
    assert helper.core_options(dict(base, assignment_kl_weight=0.)) == {}
    sentinel = object()
    assert helper.validate_cached(sentinel, base, None, None, None, None, None) is sentinel
    core = ast.parse((SOURCE / "soft_ce_partition.py").read_text())
    fn = next(n for n in core.body if isinstance(n, ast.FunctionDef) and n.name == "optimize_ce_assignment")
    assert hashlib.sha256(ast.dump(fn.args, include_attributes=False).encode()).hexdigest() == SIGNATURE_SHA
    for name in ("head_gradient", "solve_head_system", "solve_inner_newton_first", "outer_value_gradient", "implicit_moment_gradient"):
        a = next(n for n in core.body if isinstance(n, ast.FunctionDef) and n.name == name)
        assert hashlib.sha256(ast.dump(a, include_attributes=False).encode()).hexdigest() == LEGACY_HEAD_AST_SHA[name]
    config = next(n for n in ast.walk(fn) if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == "resume_config" for t in n.targets))
    assert hashlib.sha256(ast.dump(config, include_attributes=False).encode()).hexdigest() == LEGACY_RESUME_CONFIG_AST_SHA
    weight_parse = next(n for n in ast.walk(fn) if isinstance(n, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == "row_kl_weight" for t in n.targets))
    assert config.lineno < weight_parse.lineno
    inactive = next(n for n in ast.walk(fn) if isinstance(n, ast.If)
                    and any(isinstance(x, ast.Assign) and hashlib.sha256(ast.dump(x, include_attributes=False).encode()).hexdigest()
                            == LEGACY_OPERATION_ALIAS_AST_SHA
                            for statement in n.orelse for x in ast.walk(statement))
                    and any(isinstance(x, ast.Call) and isinstance(x.func, ast.Attribute)
                            and isinstance(x.func.value, ast.Name) and x.func.value.id == "InitialAssignmentKLMoments"
                            and x.func.attr == "apply" for statement in n.body for x in ast.walk(statement)))
    assert any(isinstance(x, ast.Call) and hashlib.sha256(ast.dump(x, include_attributes=False).encode()).hexdigest()
               == LEGACY_OPERATION_CALL_AST_SHA
               for statement in inactive.orelse for x in ast.walk(statement))
    public = ast.parse((SOURCE / "citation_search.py").read_text())
    canonicalizer = next(n for n in public.body if isinstance(n, ast.FunctionDef) and n.name == "_candidate_nystrom_mass")
    namespace = {"__builtins__": dict(vars(builtins), __import__=forbidden),
                 "np": legacy.np, "Real": Real, "math": math}
    exec(compile(ast.Module(body=[canonicalizer], type_ignores=[]), "public_zero_candidate_AST", "exec"), namespace)
    assert namespace["_candidate_nystrom_mass"](base) == base
    assert namespace["_candidate_nystrom_mass"](dict(base, assignment_kl_weight=0.)) == base
    selected = next(n for n in public.body if isinstance(n, ast.FunctionDef) and n.name == "selected_test")
    parents = {child: parent for parent in ast.walk(selected) for child in ast.iter_child_nodes(parent)}
    for node in ast.walk(selected):
        if isinstance(node, ast.ImportFrom) and node.module == "src.initial_assignment_row_kl":
            ancestors, ancestor = [], parents.get(node)
            while ancestor is not None:
                ancestors.append(ancestor)
                ancestor = parents.get(ancestor)
            assert any(isinstance(a, ast.If) and isinstance(a.test, ast.Compare)
                       and ast.unparse(a.test) == "candidate.get('assignment_kl_weight') == 1" for a in ancestors)
    assert not torch.cuda.is_initialized()
