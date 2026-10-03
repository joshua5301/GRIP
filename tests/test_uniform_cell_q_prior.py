"""Eight proposed new CPU domains; unexecuted and pending protocol approval.

No datasets, saved tensors, GPU, P optimization or students. The sole new
mathematical term is KL(source teacher prior || uniform-cell decoded Q prior).
"""
import ast
import copy
import hashlib
import importlib.util
import math
from pathlib import Path

import pytest
import torch

from src import soft_ce_partition as legacy
from src.low_rank_assignment import LowRankMoments
from src.moments import augmented, decode_moments, make_material

ROOT = next(p for p in Path(__file__).resolve().parents if (p/"src/soft_ce_partition.py").is_file())
PROPOSED = ROOT/"results/implementation_drafts/uniform_cell_Q_prior_stageCB_v1/proposed"
if Path(__file__).resolve().parent == ROOT/"tests":
    from src import uniform_cell_q_prior as helper
else:
    spec = importlib.util.spec_from_file_location("uniform_cell_q_prior_CB_draft", PROPOSED/"src/uniform_cell_q_prior.py")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)

EPS = (1e-5, 3e-6)
VALUE_ATOL = 2e-14
GRAD_ATOL, GRAD_RTOL = 2e-12, 2e-10
FD_ATOL, FD_RTOL = 2e-9, 2e-6
TOY_PENALTY = .2
NEWTON_STEPS = 16
STATIONARITY = 1e-11


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


def probability(u, v, hard):
    # Independent dense original NODE logits, not the protected custom VJP.
    base = torch.full((len(u), len(v)), math.log(.05/len(v)), dtype=u.dtype)
    base.scatter_(1, hard[:, None], math.log(.95+.05/len(v)))
    return (base+u@v.T/math.sqrt(u.shape[1])).softmax(1)


def dense_moments(p, z, q):
    return p.T@torch.cat((torch.ones(len(z), 1, dtype=z.dtype), z, q), 1)/len(z)


def literal_kl(p, q):
    cell_q = (p.T@q)/p.sum(0)[:, None]
    pi, cell_pi = q.mean(0), cell_q.mean(0)
    return (pi*(pi.log()-cell_pi.log())).sum()


def direction(value):
    result = torch.arange(1, value.numel()+1, dtype=value.dtype).reshape_as(value)
    result = result-result.mean()
    return result/result.norm()


def assert_fd(prediction, values, epsilon, record_property, name):
    measured = float((values[0]-values[1])/(2*epsilon))
    error = abs(float(prediction)-measured)
    record_property(name, error)
    assert error <= FD_ATOL+FD_RTOL*abs(measured)


def test_domain1_analytic_moment_quotient_and_mass_denominator(record_property):
    z, q, hard, u, v = fixture()
    m = dense_moments(probability(u, v, hard), z, q).requires_grad_()
    pi = helper.source_prior(q)
    actual = helper.prior_partials(m, 2, pi)
    labels = m[:, 3:]/m[:, :1]
    reference = (pi*(pi.log()-labels.mean(0).log())).sum()
    expected, = torch.autograd.grad(reference, m)
    record_property("moment_gradient_max_abs", float((actual["moment_gradient"]-expected).abs().max()))
    torch.testing.assert_close(actual["moment_gradient"], expected, atol=GRAD_ATOL, rtol=GRAD_RTOL)
    assert float(actual["value"]) == pytest.approx(float(reference.detach()), abs=VALUE_ATOL)
    assert bool(actual["moment_gradient"][:, 1:3].eq(0).all())
    assert bool(actual["moment_gradient"][:, 0].ne(0).all())
    dm = direction(m)
    for epsilon in EPS:
        values = [helper.prior_value(m.detach()+s*epsilon*dm, 2, pi) for s in (1, -1)]
        assert_fd((expected*dm).sum(), values, epsilon, record_property, f"moment_FD_{epsilon}")


def test_domain2_dense_P_chain_autograd_and_two_epsilon_FD(record_property):
    z, q, hard, u, v = fixture()
    p = probability(u, v, hard).requires_grad_()
    before_q = q.clone()
    m = dense_moments(p, z, q)
    partial = helper.prior_partials(m, 2, helper.source_prior(q))
    actual, = torch.autograd.grad(m, p, grad_outputs=partial["moment_gradient"], retain_graph=True)
    expected, = torch.autograd.grad(literal_kl(p, q), p)
    torch.testing.assert_close(actual, expected, atol=GRAD_ATOL, rtol=GRAD_RTOL)
    dp = direction(p)
    dp = dp-dp.mean(1, keepdim=True)  # Row-stochastic tangent, strictly positive +/-P.
    dp = dp/dp.norm()
    for epsilon in EPS:
        variants = [p.detach()+s*epsilon*dp for s in (1, -1)]
        assert all(bool((x > 0).all()) for x in variants)
        values = [literal_kl(x, q) for x in variants]
        assert_fd((actual*dp).sum(), values, epsilon, record_property, f"P_FD_{epsilon}")
    assert torch.equal(q, before_q)


def test_domain3_streamed_lowrank_A_chain_vs_dense_autograd_and_FD(record_property):
    z, q, hard, u, v = fixture()
    u, v = u.requires_grad_(), v.requires_grad_()
    rng = torch.get_rng_state().clone()
    original = [x.detach().clone() for x in (z, q, hard, u, v)]
    m = LowRankMoments.apply(u, v, hard, make_material(z, q), .05, 3)
    partial = helper.prior_partials(m, 2, helper.source_prior(q))
    actual = torch.autograd.grad(m, (u, v), grad_outputs=partial["moment_gradient"])
    expected = torch.autograd.grad(literal_kl(probability(u, v, hard), q), (u, v))
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, atol=GRAD_ATOL, rtol=GRAD_RTOL)
        assert float(a.norm()) > 0
    du, dv = direction(u), direction(v)
    prediction = (actual[0]*du).sum()+(actual[1]*dv).sum()
    for epsilon in EPS:
        values = [literal_kl(probability(u.detach()+s*epsilon*du, v.detach()+s*epsilon*dv, hard), q) for s in (1, -1)]
        assert_fd(prediction, values, epsilon, record_property, f"lowrank_FD_{epsilon}")
    assert torch.equal(torch.get_rng_state(), rng)
    assert all(torch.equal(a, b) for a, b in zip(original, (z, q, hard, u, v), strict=True))


def _newton_theta(m, differentiable):
    centers, labels, _ = decode_moments(m, 2)
    x = augmented(centers)
    theta = m.new_zeros(3, 3).requires_grad_()
    def inner(t):
        return -(labels*(x@t.T).log_softmax(1)).sum()/len(labels)+TOY_PENALTY*t.square().sum()/2
    for _ in range(NEWTON_STEPS):
        g, = torch.autograd.grad(inner(theta), theta, create_graph=differentiable)
        h = torch.autograd.functional.hessian(inner, theta, create_graph=differentiable).reshape(9, 9)
        theta = theta-torch.linalg.solve(h, g.flatten()).reshape_as(theta)
        if not differentiable:
            theta = theta.detach().requires_grad_()
    g, = torch.autograd.grad(inner(theta), theta, retain_graph=True)
    assert float(g.abs().max()) <= STATIONARITY
    return theta


def _joint_oracle(u, v, z, q, hard, differentiable):
    p = probability(u, v, hard)
    theta = _newton_theta(dense_moments(p, z, q), differentiable)
    f = -(q*(augmented(z)@theta.T).log_softmax(1)).sum()/len(q)
    return f+literal_kl(p, q)


def test_domain4_new_joint_objective_implicit_plus_direct_vs_whole_oracle_FD(record_property):
    z, q, hard, u, v = fixture()
    u, v = u.requires_grad_(), v.requires_grad_()
    m = LowRankMoments.apply(u, v, hard, make_material(z, q), .05, 3)
    theta = _newton_theta(m.detach(), False).detach()
    f, rhs = legacy.outer_value_gradient(z, q, theta)
    centers, labels, mass = decode_moments(m.detach(), 2)
    vector, diagnostic = legacy.solve_head_system(augmented(centers), labels, torch.full_like(mass, 1/len(mass)),
                                                 theta, TOY_PENALTY, rhs, rtol=1e-11, atol=1e-14)
    assert diagnostic["cg_converged"]
    ce_partial = legacy.implicit_moment_gradient(m, 2, theta, vector, TOY_PENALTY, "uniform")
    combined = helper.combine_direction(ce_partial, m, 2, helper.source_prior(q))
    actual = torch.autograd.grad(m, (u, v), grad_outputs=combined)
    expected = torch.autograd.grad(_joint_oracle(u, v, z, q, hard, True), (u, v))
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, atol=2e-10, rtol=2e-8)
    assert math.isfinite(f)
    du, dv = direction(u), direction(v)
    prediction = (actual[0]*du).sum()+(actual[1]*dv).sum()
    for epsilon in EPS:
        values = [_joint_oracle(u.detach()+s*epsilon*du, v.detach()+s*epsilon*dv, z, q, hard, False) for s in (1, -1)]
        assert_fd(prediction, values, epsilon, record_property, f"joint_FD_{epsilon}")
    record_property("new_joint_only_no_P_optimizer", True)


def test_domain5_conservation_is_not_uniform_prior_and_cell_invariances():
    z, q, hard, u, v = fixture()
    pi = helper.source_prior(q)
    m = dense_moments(probability(u, v, hard), z, q)
    _, labels, mass = decode_moments(m, 2)
    torch.testing.assert_close((mass[:, None]*labels).sum(0), pi, atol=2e-15, rtol=0)
    assert float((labels.mean(0)-pi).abs().max()) > .01
    original = helper.prior_partials(m, 2, pi)
    order = torch.tensor([2, 0, 1])
    permuted = helper.prior_partials(m[order], 2, pi)
    torch.testing.assert_close(permuted["value"], original["value"], atol=VALUE_ATOL, rtol=0)
    torch.testing.assert_close(permuted["moment_gradient"], original["moment_gradient"][order], atol=GRAD_ATOL, rtol=GRAD_RTOL)
    factors = torch.tensor([.5, 2., 1.5], dtype=m.dtype)[:, None]
    scaled = helper.prior_partials(m*factors, 2, pi)
    torch.testing.assert_close(scaled["value"], original["value"], atol=VALUE_ATOL, rtol=0)
    torch.testing.assert_close(scaled["moment_gradient"]*factors, original["moment_gradient"], atol=GRAD_ATOL, rtol=GRAD_RTOL)
    # This radial identity fails if the positive denominator derivative is lost.
    torch.testing.assert_close((original["moment_gradient"]*m).sum(1), torch.zeros(3, dtype=m.dtype), atol=2e-15, rtol=0)


def test_domain6_positive_support_finite_typed_domain_no_pseudocount():
    z, q, hard, u, v = fixture()
    pi = helper.source_prior(q)
    m = dense_moments(probability(u, v, hard), z, q)
    no_support = q.clone()
    no_support[:, 2] = 0
    no_support = no_support/no_support.sum(1, keepdim=True)
    for bad_q in (no_support, q.float(), q.clone().requires_grad_(), q*2, q*float("nan")):
        with pytest.raises(ValueError):
            helper.source_prior(bad_q)
    bad_mass = m.clone(); bad_mass[0, 0] = 0
    bad_label = m.clone(); bad_label[0, 3] = -.1
    no_cell_support = m.clone(); no_cell_support[:, -1] = 0
    for bad_m, bad_pi in ((bad_mass, pi), (bad_label, pi), (no_cell_support, pi),
                          (m*float("inf"), pi), (m, pi.clone().requires_grad_()), (m.float(), pi)):
        with pytest.raises((ValueError, FloatingPointError)):
            helper.prior_partials(bad_m, 2, bad_pi)


def test_domain7_zero_exact_legacy_candidate_cache_and_no_prior_algebra(monkeypatch):
    base = dict(method="low_rank", width=0, lr=.05, T=.3, rank=32, penalty=.0001,
                initialization="teacher_balanced", alpha=.3, inner_loss_weighting="uniform")
    assert helper.candidate_controls(base) == base and helper.core_options(base) == {}
    disabled = dict(base, uniform_cell_q_prior_weight=0.0)
    assert helper.candidate_controls(disabled) == base and helper.core_options(disabled) == {}
    active = helper.candidate_controls(dict(base, uniform_cell_q_prior_weight=1.0, uniform_cell_q_prior_schema=1))
    assert active != base and active["uniform_cell_q_prior_source_digest"] == helper.source_digest()
    for mutation in ({"uniform_cell_q_prior_schema": True}, {"uniform_cell_q_prior_source_digest": "stale"},
                     {"uniform_cell_q_prior_weight": .5}, {"uniform_cell_q_prior_weight": True},
                     {"uniform_cell_q_prior_typo": 1}, {"train_target_mix": .2}):
        with pytest.raises(ValueError):
            helper.candidate_controls(dict(active, **mutation))
    z, q, hard, _, _ = fixture()
    config = helper.expected_citation_config(active, 0, z, q, hard)
    checkpoint = dict(step=1, uniform_cell_q_prior_config=config,
                      uniform_cell_q_prior_context=config["uniform_cell_q_prior_context"])
    resume = dict(step=1, config=config, snapshots={1: checkpoint})
    assert helper.validate_cached(checkpoint, active, z, q, hard, 0, 25, step=1) is checkpoint
    assert helper.validate_cached(resume, active, z, q, hard, 0, 25, resume=True) is resume
    # Cached completed runs must reject changed source material and all core controls.
    for source_z, source_q, source_hard in ((z+.125, q, hard), (z, q[:, [1, 0, 2]], hard),
                                            (z, q, (hard+1)%3)):
        with pytest.raises(ValueError):
            helper.validate_cached(resume, active, source_z, source_q, source_hard, 0, 25, resume=True)
    for key, value in (("lr", .01), ("penalty", .2), ("factor_seed", 1),
                       ("uniform_cell_q_prior_schema", True), ("data_digest", "stale")):
        changed = copy.deepcopy(resume)
        changed["config"][key] = value
        with pytest.raises(ValueError):
            helper.validate_cached(changed, active, z, q, hard, 0, 25, resume=True)
    for key in ("uniform_cell_q_prior_context", "uniform_cell_q_prior_config"):
        changed = dict(checkpoint)
        changed.pop(key)
        with pytest.raises(ValueError):
            helper.validate_cached(changed, active, z, q, hard, 0, 25, step=1)
    with pytest.raises(ValueError):
        helper.validate_cached(dict(checkpoint, step=True), active, z, q, hard, 0, 25, step=1)
    sentinel = object()
    assert helper.validate_cached(sentinel, base, None, None, None, None, None) is sentinel
    def forbidden(*args, **kwargs):
        raise AssertionError("Disabled path evaluated prior algebra")
    monkeypatch.setattr(helper, "prior_partials", forbidden)
    old_direction = object()
    assert helper.combine_direction(old_direction, None, None, None, 0) is old_direction


def test_domain8_signature_default_config_and_public_field_forwarding_AST():
    source_root = ROOT/"src" if Path(__file__).resolve().parent == ROOT/"tests" else PROPOSED/"src"
    core = ast.parse((source_root/"soft_ce_partition.py").read_text())
    optimizer = next(n for n in core.body if isinstance(n, ast.FunctionDef) and n.name == "optimize_ce_assignment")
    # Literal frozen old signature, computed by stdlib before any proof run.
    signature = hashlib.sha256(ast.dump(optimizer.args, include_attributes=False).encode()).hexdigest()
    assert signature == "b109a20373f1dc0a94362ff0ec9ffd254b76cceac2ebba7f9dc7b24f393e336d"
    names = [n for n in ast.walk(optimizer) if isinstance(n, ast.Assign)]
    capture = next(n.lineno for n in names if any(isinstance(t, ast.Name) and t.id == "resume_config" for t in n.targets))
    parse = next(n.lineno for n in names if any(isinstance(t, ast.Name) and t.id == "prior_weight" for t in n.targets))
    assert capture < parse
    imports = [n for n in ast.walk(optimizer) if isinstance(n, ast.ImportFrom) and n.module == "src.uniform_cell_q_prior"]
    assert len(imports) == 1
    active_branch = next(n for n in ast.walk(optimizer) if isinstance(n, ast.If) and isinstance(n.test, ast.Name)
                         and n.test.id == "prior_active" and any(isinstance(x, ast.ImportFrom) for x in n.body))
    assert imports[0] in active_branch.body
    combined = next(n for n in names if any(isinstance(t, ast.Name) and t.id == "direction" for t in n.targets)
                    and isinstance(n.value, ast.BinOp))
    back = [n for n in ast.walk(optimizer) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and isinstance(n.func.value, ast.Name) and n.func.value.id == "moments" and n.func.attr == "backward"]
    assert len(back) == 1 and combined.lineno < back[0].lineno
    text = (source_root/"citation_search.py").read_text()
    for key in helper.KEYS:
        assert text.count('"'+key+'"') >= 2  # Grouped rows and selected candidate reconstruction.
    assert "**prior_options" in text and "Selected row lost its recorded prior identity" in text
    public = ast.parse(text)
    run = next(n for n in public.body if isinstance(n, ast.FunctionDef) and n.name == "run_screen")
    resume_guard = next(n for n in ast.walk(run) if isinstance(n, ast.Call)
                        and isinstance(n.func, ast.Name) and n.func.id == "validate_prior_cached"
                        and any(k.arg == "resume" for k in n.keywords))
    skip = next(n for n in ast.walk(run) if isinstance(n, ast.If)
                and isinstance(n.test, ast.BoolOp) and "state['step'] < steps" in ast.unparse(n.test))
    assert resume_guard.lineno < skip.lineno
    for name in ("run_screen", "selected_test"):
        function = next(n for n in public.body if isinstance(n, ast.FunctionDef) and n.name == name)
        assert any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                   and n.func.id == "validate_prior_cached" and any(k.arg == "step" for k in n.keywords)
                   for n in ast.walk(function))
