"""Fixed original-P0 marginals preserve prototypes and exact nonlinear CE gradients."""
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from test_nystrom_balanced_mass import citation_mock as citation_mock
from test_nystrom_balanced_mass import cpu_threads as cpu_threads
from test_nystrom_balanced_mass import load, problem

import src.citation_search as search
import src.nystrom_ce as nys
from src.fixed_mass_assignment import FixedMassMoments, validate_mass_target
from src.io import _fingerprint, save_state
from src.low_rank_assignment import LowRankLogits, LowRankMoments, initialize_factors
from src.moments import augmented, decode_moments, make_material
from src.shared_features import get_shared_map
from src.soft_ce_partition import solve_head_system, solve_inner_newton_first


def fit_objective(value, h, q, feature_map, phi, weighting):
    x, y, mass = decode_moments(value, h.shape[1])
    weights = nys._inner_weights(mass, weighting)
    mapped = feature_map(x).detach()
    inner = solve_inner_newton_first(mapped, y, weights, .1, grad_tol=1e-9)
    assert inner["inner_converged"] and inner["inner_grad_max"] <= 1e-9
    loss, rhs = nys.outer_gradient(phi, q, inner["theta"], 7)
    return loss, rhs, inner["theta"], mapped, y, weights

def controls():
    return dict(penalty=0.1, rank=3, chunk=7, checkpoint_every=1, mass_mode="initial",
                balance_steps=5000, balance_tol=1e-8, inner_loss_weighting="uniform")


def context_config(a):
    context = nys.initial_mass_context(a, 4, 3, 0, 0.05, 7)
    config = nys.initial_mass_config(0.1, 0.01, 3, 0, 4, 7, 0.05, "uniform", 5000, 1e-8, "chunked", context)
    return context, config


def fixed(u, v, a, material, target, tol=1e-12):
    logits = LowRankLogits.apply(u, v, a, 0.05, 7)
    return FixedMassMoments.apply(logits, material, target, 7, 5000, tol, None), logits


def files(folder):
    return {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in folder.rglob("*") if p.is_file()}


@pytest.mark.parametrize("factor", ["u", "v", "both"])
@pytest.mark.parametrize("weighting", ["uniform", "mass"])
def test_full_nonuniform_ce_hypergradient_fd(factor, weighting, record_property):
    h, q, a, feature_map = problem()
    target = context_config(a)[0]["target"]
    generator = torch.Generator().manual_seed(19)
    u = (0.2 * torch.randn(24, 3, dtype=torch.double, generator=generator)).requires_grad_()
    v = torch.randn(4, 3, dtype=torch.double, generator=generator).requires_grad_()
    du = 0.2 * torch.randn(u.shape, dtype=u.dtype, generator=generator)
    dv = 0.2 * torch.randn(v.shape, dtype=v.dtype, generator=generator)
    if factor == "u":
        dv.zero_()
    elif factor == "v":
        du.zero_()
    material, phi = make_material(h, q), feature_map(h).detach().numpy()
    (moments, dual, diagnostic), logits = fixed(u, v, a, material, target)
    logits.retain_grad()
    _, rhs, theta, mapped, y, weights = fit_objective(moments.detach(), h, q, feature_map, phi, weighting)
    vector, result = solve_head_system(augmented(mapped), y, weights, theta, 0.1, rhs, rtol=1e-10)
    assert result["cg_converged"]
    moments.backward(nys.moment_gradient(moments, 3, feature_map, theta, vector, 0.1, weighting))
    analytic = float((u.grad * du).sum() + (v.grad * dv).sum())
    epsilon = 0.001
    plus = fixed(u.detach() + epsilon * du, v.detach() + epsilon * dv, a, material, target)[0][0]
    minus = fixed(u.detach() - epsilon * du, v.detach() - epsilon * dv, a, material, target)[0][0]
    finite = (fit_objective(plus, h, q, feature_map, phi, weighting)[0]
              - fit_objective(minus, h, q, feature_map, phi, weighting)[0]) / (2 * epsilon)
    error = abs(analytic - finite)
    record_property("fd_absolute_error", error)
    record_property("fd_relative_error", error / max(abs(analytic), abs(finite), 1e-15))
    assert error < 2e-8 + 2e-4 * abs(finite)
    assert float((moments.detach()[:, 0] / target - 1).abs().max()) <= 1e-12
    assert float(diagnostic[1:3].max()) <= 1e-12
    assert float(logits.grad.sum(0).abs().max()) < 1e-12
    assert float(logits.grad.sum(1).abs().max()) < 1e-12
    assert abs(float(dual.mean())) < 1e-14


def test_arbitrary_fixed_marginal_moment_fd_and_gauge(record_property):
    h, q, _, _ = problem()
    g = torch.Generator().manual_seed(123)
    logits = torch.randn(24, 4, dtype=torch.double, generator=g, requires_grad=True)
    direction = torch.randn(24, 4, dtype=torch.double, generator=g)
    material = make_material(h, q)
    target = torch.tensor([0.55, 0.25, 0.15, 0.05], dtype=torch.double)
    cotangent = torch.randn(4, 7, dtype=torch.double, generator=g)
    moments, dual, _ = FixedMassMoments.apply(logits, material, target, 7, 5000, 1e-12, None)
    moments.backward(cotangent)
    objective = lambda z: float((FixedMassMoments.apply(z, material, target, 7, 5000, 1e-12, None)[0] * cotangent).sum())
    epsilon = 1e-4
    finite = (objective(logits.detach() + epsilon * direction) - objective(logits.detach() - epsilon * direction)) / (2 * epsilon)
    error = abs(float((logits.grad * direction).sum()) - finite)
    record_property("moment_fd_absolute_error", error)
    assert error < 1e-8
    probability = (logits.detach() + dual).softmax(1)
    torch.testing.assert_close(probability.mean(0), target, atol=1e-12, rtol=0)
    torch.testing.assert_close(moments, probability.T @ material / len(h), atol=1e-14, rtol=1e-13)
    shifted = logits.detach() + torch.linspace(-0.7, 0.5, 24, dtype=torch.double)[:, None] + torch.tensor([.5, -.2, .1, -.4])
    shifted_moments, shifted_dual, _ = FixedMassMoments.apply(shifted, material, target, 7, 5000, 1e-12, None)
    torch.testing.assert_close(shifted_moments, moments, atol=2e-12, rtol=1e-11)
    torch.testing.assert_close((shifted + shifted_dual).softmax(1), probability, atol=3e-12, rtol=1e-11)


def test_original_p0_moments_head_and_inputs_bitwise_free(tmp_path, record_property):
    h, q, a, feature_map = problem()
    u, v = initialize_factors(a, 4, 3, 0)
    reference = LowRankMoments.apply(u, v, a, make_material(h, q), .05, 7)
    context, config = context_config(a)
    torch.testing.assert_close(context["target"], reference[:, 0], atol=1e-15, rtol=0)
    assert not torch.allclose(context["target"], torch.full((4,), .25, dtype=torch.double))
    (initial, dual, diagnostic), _ = fixed(u, v, a, make_material(h, q), context["target"])
    assert torch.equal(initial, reference) and torch.equal(dual, torch.zeros_like(dual))
    assert int(diagnostic[0]) == 1
    for mode in ("free", "initial"):
        option = controls() if mode == "initial" else dict(penalty=.1, rank=3, chunk=7, inner_loss_weighting="uniform")
        nys.optimize(h, q, a, feature_map, feature_map(h).detach().numpy(), tmp_path / mode, 0, **option)
    free, initial = [load(tmp_path / mode / "step_000000.pt") for mode in ("free", "initial")]
    for key in ("moments", "theta"):
        assert torch.equal(free[key], initial[key])
    for key in ("u", "v"):
        assert torch.equal(load(tmp_path / "free/resume.pt")[key], load(tmp_path / "initial/resume.pt")[key])
    assert free["input_fingerprint"] == initial["input_fingerprint"]
    record_property("p0_moments_max_abs", float((free["moments"] - initial["moments"]).abs().max()))
    record_property("p0_head_max_abs", float((free["theta"] - initial["theta"]).abs().max()))
    nys.validate_assignment_snapshot(initial, "initial", 5000, initial_context=context, initial_config=config)


def test_mass_and_uniform_inner_heads_and_gradients_differ(record_property):
    h, q, a, feature_map = problem()
    g = torch.Generator().manual_seed(16)
    base_u = .2 * torch.randn(24, 3, dtype=torch.double, generator=g)
    base_v = torch.randn(4, 3, dtype=torch.double, generator=g)
    target = context_config(a)[0]["target"]
    heads, gradients = [], []
    for weighting in ("uniform", "mass"):
        u, v = base_u.clone().requires_grad_(), base_v.clone().requires_grad_()
        (moments, _, _), _ = fixed(u, v, a, make_material(h, q), target)
        _, rhs, theta, mapped, y, weights = fit_objective(moments.detach(), h, q, feature_map, feature_map(h).detach().numpy(), weighting)
        vector, diagnostic = solve_head_system(augmented(mapped), y, weights, theta, .1, rhs, rtol=1e-10)
        assert diagnostic["cg_converged"]
        moments.backward(nys.moment_gradient(moments, 3, feature_map, theta, vector, .1, weighting))
        heads.append(theta)
        gradients.append(torch.cat([u.grad.flatten(), v.grad.flatten()]))
    head_error, grad_error = float((heads[0] - heads[1]).abs().max()), float((gradients[0] - gradients[1]).abs().max())
    record_property("mass_uniform_head_max_abs", head_error)
    record_property("mass_uniform_gradient_max_abs", grad_error)
    assert head_error > 1e-3 and grad_error > 1e-6


def test_initial_short_interrupted_and_extended_resume_parity(tmp_path):
    h, q, a, feature_map = problem()
    phi = feature_map(h).detach().numpy()
    nys.optimize(h, q, a, feature_map, phi, tmp_path / "full", 2, **controls())
    nys.optimize(h, q, a, feature_map, phi, tmp_path / "extend", 1, **controls())
    nys.optimize(h, q, a, feature_map, phi, tmp_path / "extend", 2, **controls())
    stopped = {"yes": False}
    def progress(row):
        stopped["yes"] = row["step"] == 0
    with pytest.raises(InterruptedError):
        nys.optimize(h, q, a, feature_map, phi, tmp_path / "stop", 2, stop=lambda: stopped["yes"], progress=progress, **controls())
    assert load(tmp_path / "stop/resume.pt")["step"] == 1
    nys.optimize(h, q, a, feature_map, phi, tmp_path / "stop", 2, **controls())
    reference = load(tmp_path / "full/step_000002.pt")
    context, config = context_config(a)
    for folder in ("extend", "stop"):
        actual, state = load(tmp_path / folder / "step_000002.pt"), load(tmp_path / folder / "resume.pt")
        nys.validate_assignment_snapshot(actual, "initial", 5000, initial_context=context, initial_config=config)
        nys.validate_assignment_resume(state, "initial", 5000, initial_context=context, initial_config=config)
        for key in ("moments", "theta", "mass_target"):
            torch.testing.assert_close(actual[key], reference[key], atol=1e-9, rtol=1e-7)
        assert actual["initial_mass_provenance"] == reference["initial_mass_provenance"]


@pytest.mark.parametrize("target", [True, [0.4, .3, .2, .1], torch.ones(4), torch.tensor([.5, .3, .2, 0.], dtype=torch.double),
                                   torch.tensor([.5, .3, .3, -.1], dtype=torch.double), torch.tensor([.5, .3, .2, float("nan")], dtype=torch.double),
                                   torch.full((4,), .2, dtype=torch.double), torch.ones(2, 2, dtype=torch.double)/4,
                                   torch.full((4,), .25, dtype=torch.double, requires_grad=True)])
def test_invalid_target_rejected(target):
    with pytest.raises(ValueError, match="target"):
        validate_mass_target(target, 4)


@pytest.mark.parametrize("change", ["target", "forged_target", "source", "assignment", "seed", "rank", "mixing", "input", "config", "mode", "dual", "diagnostic",
                                   "h", "q", "map", "phi", "penalty", "lr", "inner", "balance_steps", "balance_tol"])
def test_invalid_resume_snapshot_current_input_rejects_without_mutation(tmp_path, monkeypatch, change):
    h, q, a, feature_map = problem()
    path = tmp_path / "phi.npy"
    phi = nys.cache_features(h, feature_map, path, 7)
    folder, option = tmp_path / "cond", controls()
    nys.optimize(h, q, a, feature_map, phi, folder, 0, **option)
    state = load(folder / "resume.pt")
    if change in ("target", "forged_target"):
        state["mass_target"] = state["mass_target"].roll(1)
        if change == "forged_target":
            digest = nys._content_digest(state["mass_target"], 7, canonical_double=True)
            state["initial_mass_provenance"]["target_digest"] = digest
            state["config"]["initial_mass_provenance"]["target_digest"] = digest
    elif change == "source":
        monkeypatch.setattr(nys, "_initial_mass_source_digest", lambda: "a"*64)
    elif change == "input":
        state["input_fingerprint"]["q_digest"] = "a"*64
    elif change == "config":
        state["config"]["lr"] *= 2
    elif change == "mode":
        state["config"]["mass_mode"] = "free"
    elif change == "dual":
        state["dual"] = torch.full((4,), float("nan"))
    elif change == "diagnostic":
        snap = load(folder / "step_000000.pt")
        snap["balance"]["balance_backward_checked"] = False
        save_state(snap, folder / "step_000000.pt")
    elif change == "assignment":
        a = a.roll(1)
    elif change == "seed":
        option["seed"] = 1
    elif change == "rank":
        option["rank"] = 2
    elif change == "mixing":
        option["mixing"] = .1
    elif change == "h":
        h = h + .01
    elif change == "q":
        q = q.roll(1, 0)
    elif change == "map":
        feature_map.mapping += .01
    elif change == "phi":
        phi = np.array(phi) + .01
    elif change == "penalty":
        option["penalty"] = .2
    elif change == "lr":
        option["lr"] = .02
    elif change == "inner":
        option["inner_loss_weighting"] = "mass"
    elif change == "balance_steps":
        option["balance_steps"] = 4999
    elif change == "balance_tol":
        option["balance_tol"] = 1e-7
    save_state(state, folder / "resume.pt")
    path.with_suffix(".meta.json").unlink()
    before = files(tmp_path)
    with pytest.raises(ValueError):
        nys.optimize(h, q, a, feature_map, phi, folder, 1, **option)
    assert files(tmp_path) == before


@pytest.mark.parametrize("failure", ["marginals", "adjoint", "backward", "tangent", "head"])
def test_failed_initial_derivative_never_persists_endpoint(tmp_path, monkeypatch, failure):
    h, q, a, feature_map = problem()
    if failure == "marginals":
        original = FixedMassMoments.apply
        def corrupt(*args):
            moments, dual, diagnostic = original(*args)
            moments = moments.clone()
            moments[0, 0] *= 1.01
            return moments, dual, diagnostic
        monkeypatch.setattr(FixedMassMoments, "apply", corrupt)
    elif failure == "adjoint":
        monkeypatch.setattr(nys, "solve_head_system", lambda *args, **kwargs: (torch.zeros_like(args[3]), {"cg_converged": False}))
    elif failure == "backward":
        def fail(*args):
            raise RuntimeError("Synthetic backward failure")
        monkeypatch.setattr(FixedMassMoments, "backward", fail)
    elif failure == "tangent":
        original = FixedMassMoments.backward
        def corrupt(*args):
            result = original(*args)
            return (result[0]+.01, *result[1:])
        monkeypatch.setattr(FixedMassMoments, "backward", corrupt)
    else:
        original = nys.solve_inner_newton_first
        def corrupt(*args, **kwargs):
            result = original(*args, **kwargs)
            result["inner_converged"] = False
            return result
        monkeypatch.setattr(nys, "solve_inner_newton_first", corrupt)
    with pytest.raises((RuntimeError, FloatingPointError)):
        nys.optimize(h, q, a, feature_map, feature_map(h).detach().numpy(), tmp_path / "failed", 0, **controls())
    assert not list((tmp_path / "failed").glob("*.pt"))


def test_insufficient_budget_preserves_valid_p0_only(tmp_path):
    h, q, a, feature_map = problem()
    option = controls()
    option["balance_steps"] = 1
    with pytest.raises(RuntimeError, match="did not converge"):
        nys.optimize(h, q, a, feature_map, feature_map(h).detach().numpy(), tmp_path, 2, **option)
    assert (tmp_path / "step_000000.pt").exists() and not (tmp_path / "step_000001.pt").exists()


def test_map_code_cache_identity_firstlineno39_exact():
    code = nys.NystromMap.__call__.__code__
    assert code.co_firstlineno == 39
    expected = "    def __call__(self, h):\n        return get_kernel_values(h.double(), self.anchors, self.kernel) @ self.mapping\n"
    assert inspect.getsource(nys.NystromMap.__call__) == expected


def candidate(**extra):
    return dict(method="nystrom", width=0, lr=.01, T=1., rank=3, penalty=.1, inner_loss_weighting="uniform", **extra)


def run(tmp_path, candidates):
    return search.run_screen("cora", .013, tmp_path, candidates, steps=1, student_seeds=(1800,),
                             condensation_seed=0, epochs=1, dropout=0, device="cpu")


@pytest.mark.parametrize("method", ["low_rank", "mlp", "distance", "coarsening"])
def test_initial_mode_not_allowed_for_other_methods(method):
    with pytest.raises(ValueError, match="mass_mode"):
        search._assignment_mass_mode(dict(method=method, mass_mode="initial"))


@pytest.mark.parametrize("extra", [dict(mass_target=[.5,.2,.2,.1]), dict(initial_mass_schema=2), dict(initial_mass_schema=True),
                                    dict(balance_backend="cached"), dict(balance_steps=0), dict(balance_tol=float("nan"))])
def test_initial_invalid_candidate_rejects_before_dataset(tmp_path, citation_mock, extra):
    with pytest.raises(ValueError):
        run(tmp_path, [candidate(mass_mode="initial", **extra)])
    assert citation_mock["data_calls"] == 0 and not list(tmp_path.iterdir())


def test_initial_citation_forward_identity_and_forged_cached_target(tmp_path, citation_mock, monkeypatch):
    monkeypatch.setattr(search, "get_shared_map", get_shared_map)
    options = candidate(mass_mode="initial", balance_steps=5000)
    ranking, root = run(tmp_path, [candidate(), options])
    assert len(ranking) == 4 and len(citation_mock["fits"]) == 4
    default, initial = citation_mock["optimizer_kwargs"]
    assert "mass_mode" not in default and initial["mass_mode"] == "initial"
    assert initial["balance_steps"] == 5000 and initial["balance_tol"] == 1e-8
    choice = ranking[(ranking.mass_mode == "initial") & (ranking.step == 1)].iloc[0].to_dict()
    folder = Path(choice["candidate_path"]) / "condensation_0"
    identity = json.loads((folder.parent / "candidate.json").read_text())
    assert identity["initial_mass_schema"] == 1 and _fingerprint(identity) == folder.parent.name
    before = len(citation_mock["optimizer_kwargs"])
    run(tmp_path, [options])
    assert len(citation_mock["optimizer_kwargs"]) == before
    snapshot = load(folder / "step_000001.pt")
    snapshot["mass_target"] = snapshot["mass_target"].roll(1)
    digest = nys._content_digest(snapshot["mass_target"], canonical_double=True)
    snapshot["initial_mass_provenance"]["target_digest"] = digest
    snapshot["initial_assignment_config"]["initial_mass_provenance"]["target_digest"] = digest
    save_state(snapshot, folder / "step_000001.pt")
    before = len(citation_mock["fits"])
    with pytest.raises(ValueError, match="Initial-mass"):
        run(tmp_path, [options])
    assert len(citation_mock["fits"]) == before
    before = citation_mock["data_calls"]
    with pytest.raises(ValueError, match="Initial-mass"):
        search.selected_test(root, choice, condensation_seeds=(0,), student_seeds=(1900,), epochs=1, device="cpu", report_routes=False)
    assert citation_mock["data_calls"] == before and not (root / "selected.json").exists()


@pytest.mark.parametrize("field", ["balance_steps", "balance_tol", "balance_backend", "initial_mass_schema"])
def test_selected_initial_requires_frozen_controls_preload(tmp_path, citation_mock, monkeypatch, field):
    monkeypatch.setattr(search, "get_shared_map", get_shared_map)
    ranking, root = run(tmp_path, [candidate(mass_mode="initial", balance_steps=5000)])
    choice = ranking[ranking.step == 1].iloc[0].to_dict()
    choice.pop(field)
    before = citation_mock["data_calls"]
    with pytest.raises(ValueError):
        search.selected_test(root, choice, condensation_seeds=(0,), student_seeds=(1900,), device="cpu", report_routes=False)
    assert citation_mock["data_calls"] == before and not (root / "selected.json").exists()


@pytest.mark.parametrize("weighting", ["uniform", "mass"])
def test_tiny_positive_target_exact_head_and_hypergradient_fd(weighting, record_property):
    h, q, a, feature_map = problem()
    g = torch.Generator().manual_seed(120)
    u = (.1 * torch.randn(24, 3, dtype=torch.double, generator=g)).requires_grad_()
    v = torch.randn(4, 3, dtype=torch.double, generator=g).requires_grad_()
    du = .1 * torch.randn(u.shape, dtype=u.dtype, generator=g)
    dv = .1 * torch.randn(v.shape, dtype=v.dtype, generator=g)
    target = torch.tensor([.97, .028, .0011, .0009], dtype=torch.double)
    material, phi = make_material(h, q), feature_map(h).detach().numpy()
    (moments, _, diagnostic), logits = fixed(u, v, a, material, target)
    logits.retain_grad()
    _, rhs, theta, mapped, y, weights = fit_objective(moments.detach(), h, q, feature_map, phi, weighting)
    vector, solved = solve_head_system(augmented(mapped), y, weights, theta, .1, rhs, rtol=1e-10)
    assert solved["cg_converged"]
    moments.backward(nys.moment_gradient(moments, 3, feature_map, theta, vector, .1, weighting))
    analytic = float((u.grad * du).sum() + (v.grad * dv).sum())
    epsilon = .001
    plus = fixed(u.detach()+epsilon*du, v.detach()+epsilon*dv, a, material, target)[0][0]
    minus = fixed(u.detach()-epsilon*du, v.detach()-epsilon*dv, a, material, target)[0][0]
    finite = (fit_objective(plus, h, q, feature_map, phi, weighting)[0]-fit_objective(minus, h, q, feature_map, phi, weighting)[0])/(2*epsilon)
    error = abs(analytic-finite)
    record_property("tiny_target_fd_absolute_error", error)
    record_property("tiny_target_relative_column_residual", float((moments.detach()[:,0]/target-1).abs().max()))
    assert error < 2e-8 + 2e-4*abs(finite)
    assert float(diagnostic[1:3].max()) <= 1e-12
    assert float(logits.grad.sum(0).abs().max()) < 1e-12
    assert float(logits.grad.sum(1).abs().max()) < 1e-12


@pytest.mark.parametrize("method", ["low_rank", "mlp", "distance", "coarsening"])
@pytest.mark.parametrize("extra", [dict(mass_mode="initial"), dict(initial_mass_schema=1), dict(mass_target=[.4,.3,.2,.1])])
def test_unsupported_initial_controls_reject_pre_dataset(tmp_path, citation_mock, method, extra):
    with pytest.raises(ValueError):
        run(tmp_path, [dict(method=method, width=4, lr=.01, T=1., rank=3, penalty=.1, **extra)])
    assert citation_mock["data_calls"] == 0 and not list(tmp_path.iterdir())


def test_cross_device_replay_and_wrong_dual_shape_rejected_pre_mutation(tmp_path):
    h, q, a, feature_map = problem()
    nys.optimize(h, q, a, feature_map, feature_map(h).detach().numpy(), tmp_path, 0, **controls())
    snapshot, state = load(tmp_path / "step_000000.pt"), load(tmp_path / "resume.pt")
    context, config = context_config(a)
    altered = dict(target=context["target"], provenance=dict(context["provenance"], factor_device="cuda:0"))
    with pytest.raises(ValueError, match="Cross-device"):
        nys.validate_assignment_snapshot(snapshot, "initial", 5000, initial_context=altered, initial_config=config)
    with pytest.raises(ValueError, match="Cross-device"):
        nys.validate_assignment_resume(state, "initial", 5000, initial_context=altered, initial_config=config)
    state["dual"] = torch.zeros(3, dtype=torch.double)
    with pytest.raises(ValueError, match="cell count"):
        nys.validate_assignment_resume(state, "initial", 5000, initial_context=context, initial_config=config)


def test_orphan_initial_snapshot_rejects_before_selection_write(tmp_path, citation_mock, monkeypatch):
    monkeypatch.setattr(search, "get_shared_map", get_shared_map)
    ranking, root = run(tmp_path, [candidate(mass_mode="initial", balance_steps=5000)])
    choice = ranking[ranking.step == 1].iloc[0].to_dict()
    folder = Path(choice["candidate_path"])/"condensation_0"
    (folder/"resume.pt").unlink()
    before = files(tmp_path)
    calls = citation_mock["data_calls"]
    with pytest.raises(ValueError, match="resume"):
        search.selected_test(root, choice, condensation_seeds=(0,), student_seeds=(1900,), device="cpu", report_routes=False)
    assert citation_mock["data_calls"] == calls and files(tmp_path) == before


@pytest.mark.parametrize("filename", ["nystrom_ce.py", "fixed_mass_assignment.py", "balanced_assignment.py", "low_rank_assignment.py",
                                       "moments.py", "soft_ce_partition.py", "head.py", "teacher.py", "shared_features.py", "partition_initialization.py"])
def test_initial_source_guard_covers_numerical_dependencies(monkeypatch, filename):
    original = Path.read_bytes
    before = nys._initial_mass_source_digest()
    def altered(path):
        content = original(path)
        return content+b"\n# altered numerical source\n" if path.name == filename else content
    monkeypatch.setattr(Path, "read_bytes", altered)
    assert nys._initial_mass_source_digest() != before


def test_malformed_cached_initial_cell_count_is_value_error(tmp_path):
    h, q, a, feature_map = problem()
    nys.optimize(h, q, a, feature_map, feature_map(h).detach().numpy(), tmp_path, 0, **controls())
    snapshot = load(tmp_path / "step_000000.pt")
    snapshot["moments"] = snapshot["moments"][:1]
    snapshot["dual"] = torch.zeros(1, dtype=torch.double)
    context, config = context_config(a)
    with pytest.raises(ValueError, match="cell count"):
        nys.validate_assignment_snapshot(snapshot, "initial", 5000, initial_context=context, initial_config=config)
