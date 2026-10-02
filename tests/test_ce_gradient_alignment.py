"""CPU toy proof; no data/source caches, solvers, model updates or GPU."""
import copy
import math

import pytest
import torch

from src import ce_gradient_alignment as engine
from src.finite_student_outer_v2 import decode_rms_moments, functional_forward
from src.low_rank_assignment import LowRankMoments, logit_block


def fixture(dtype=torch.float64, anchors=2):
    generator = torch.Generator().manual_seed(731)
    x = torch.randn(6, 3, generator=generator, dtype=dtype) * .25
    z = x.double() * 1.2 + .15
    q = torch.tensor([[.1, .3, .6], [.2, .2, .6], [.15, .35, .5],
                      [.3, .15, .55], [.1, .4, .5], [.25, .2, .55]], dtype=torch.float64)
    s = torch.eye(6, dtype=dtype) * .55 + torch.ones(6, 6, dtype=dtype) * .045
    ensemble = []
    for _ in range(anchors):
        ensemble.append((torch.randn(4, 3, generator=generator, dtype=dtype)*.17,
                         torch.full((4,), 1.5, dtype=dtype),
                         torch.randn(3, 4, generator=generator, dtype=dtype)*.22,
                         torch.tensor([.2, -.1, .05], dtype=dtype)))
    u = torch.randn(6, 2, generator=generator, dtype=torch.float64) * .3
    v = torch.randn(3, 2, generator=generator, dtype=torch.float64) * .5
    assignment = torch.tensor([0, 1, 2, 0, 1, 2])
    material = torch.cat((torch.ones(6, 1, dtype=torch.float64), z, q), dim=1)
    p = logit_block(u, v, assignment, .2).softmax(1)
    moments = p.T @ material / len(x)
    transform = dict(kind="rms", matrix=None, center=torch.tensor([.04, -.03, .02], dtype=torch.float64),
                     output_center=torch.tensor([-.1, .05, .03], dtype=torch.float64),
                     scale=torch.tensor(.8, dtype=torch.float64))
    return dict(x=x, z=z, q=q, s=s, ensemble=tuple(ensemble), moments=moments,
                transform=transform, u=u, v=v, assignment=assignment, material=material)


def cached(f):
    return engine.source_gradient_targets(f["ensemble"], f["x"], f["s"], f["q"])


def tree_equal(a, b):
    if torch.is_tensor(a):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert set(a) == set(b)
        for key in a:
            tree_equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for aa, bb in zip(a, b, strict=True):
            tree_equal(aa, bb)
    else:
        assert a == b


def manual_source_gradient(parameters, x, s, q):
    # Independent dense chain rule, including original linear→S→bias order.
    w1, b1, w2, b2 = parameters
    pre = s @ (x @ w1.T) + b1
    hidden = pre.relu()
    logits = s @ (hidden @ w2.T) + b2
    residual = (logits.softmax(1) * q.sum(1, keepdim=True) - q) / len(x)
    routed = s.T @ residual
    gpre = (routed @ w2) * (pre > 0)
    return ((s.T @ gpre).T @ x, gpre.sum(0), routed.T @ hidden, residual.sum(0))


def direct_objective(moments, transform, targets):
    # Independent scalar expression; derivatives obtained separately in tests.
    nin = targets["anchors"][0]["parameters"][0].shape[1]
    h, q, _ = decode_rms_moments(moments, nin, transform)
    values = []
    for anchor in targets["anchors"]:
        parameters = tuple(p.clone().requires_grad_() for p in anchor["parameters"])
        dtype = parameters[0].dtype
        ce = -(q.to(dtype) * functional_forward(parameters, h.to(dtype))).sum(1).mean()
        grads = torch.autograd.grad(ce, parameters, create_graph=True)
        for g, t in zip(grads, anchor["gradients"], strict=True):
            g, t = g.double().reshape(-1), t.double().reshape(-1)
            delta2 = 1e-6 * t.square().sum()
            values.append(1 - (g*t).sum()/((g.square().sum()+delta2).sqrt()
                                          * (t.square().sum()+delta2).sqrt()))
    return torch.stack(values).mean()


def test_source_gcn_gradient_manual_and_frozen():
    f = fixture()
    before = copy.deepcopy(f)
    targets = cached(f)
    for anchor, parameters in zip(targets["anchors"], f["ensemble"], strict=True):
        manual = manual_source_gradient(parameters, f["x"], f["s"], f["q"])
        for value, reference in zip(anchor["gradients"], manual, strict=True):
            torch.testing.assert_close(value, reference, atol=2e-16, rtol=2e-14)
            assert not value.requires_grad and value.grad_fn is None
        assert anchor["source_relu"]["minimum_absolute_preactivation"] > 1
    tree_equal(f, before)


@pytest.mark.parametrize("zero", [False, True])
def test_symmetric_smoothed_block_analytic_autograd_and_fd(zero):
    generator = torch.Generator().manual_seed(23)
    targets = tuple(torch.randn(size, generator=generator, dtype=torch.float64) for size in (5, 2, 7, 3))
    syn = tuple((torch.zeros_like(t) if zero else torch.randn(t.shape, generator=generator, dtype=torch.float64))
                .requires_grad_() for t in targets)
    loss, diagnostics = engine.smoothed_block_alignment(syn, targets)
    gradients = torch.autograd.grad(loss, syn)
    for value, target, gradient, diag in zip(syn, targets, gradients, diagnostics, strict=True):
        delta2 = 1e-6 * target.square().sum()
        a, b = (value.square().sum()+delta2).sqrt(), (target.square().sum()+delta2).sqrt()
        expected = (-target/(a*b)+(value*target).sum()*value/(a.pow(3)*b))/4
        torch.testing.assert_close(gradient, expected, atol=2e-12, rtol=2e-14)
        assert diag["raw_cosine"] is None if zero else diag["raw_cosine"] is not None
    if zero:
        assert loss.item() == 1
        assert all(torch.isfinite(g).all() for g in gradients)
    else:
        directions = tuple(torch.randn(t.shape, generator=generator, dtype=torch.float64) for t in syn)
        expected = sum((a*b).sum() for a, b in zip(gradients, directions, strict=True))
        for eps in (1e-5, 3e-6):
            plus = tuple(a+eps*b for a, b in zip(syn, directions, strict=True))
            minus = tuple(a-eps*b for a, b in zip(syn, directions, strict=True))
            fd = (engine.smoothed_block_alignment(plus, targets)[0]
                  - engine.smoothed_block_alignment(minus, targets)[0])/(2*eps)
            torch.testing.assert_close(fd, expected, atol=2e-10, rtol=2e-7)


def direction(f, mode):
    g = torch.Generator().manual_seed(612)
    m = f["moments"]
    d = torch.zeros_like(m)
    if mode in ("feature", "combined"):
        d[:, 1:4] = torch.randn(3, 3, generator=g, dtype=m.dtype) * .12
    if mode in ("target", "combined"):
        dq = torch.randn(3, 3, generator=g, dtype=m.dtype) * .03
        d[:, 4:] = dq - dq.mean(1, keepdim=True)
    if mode in ("mass", "combined"):
        dm = torch.tensor([.015, -.012, -.003], dtype=m.dtype)
        d[:, 0] = dm
        d[:, 4:] += dm[:, None] * (m[:, 4:]/m[:, :1])
    return d


@pytest.mark.parametrize("mode", ["feature", "target", "mass", "combined"])
@pytest.mark.parametrize("eps", [1e-4, 3e-5])
def test_full_moment_fd_fixed_relu_branches(mode, eps, record_property):
    f = fixture()
    targets = cached(f)
    out = engine.gradient_alignment_partials(f["moments"], f["transform"], targets)
    d = direction(f, mode)
    expected = (out["moment_gradient"] * d).sum()
    plus = direct_objective(f["moments"]+eps*d, f["transform"], targets)
    minus = direct_objective(f["moments"]-eps*d, f["transform"], targets)
    fd = (plus-minus)/(2*eps)
    err = float((fd-expected).detach().abs())
    record_property("absolute_error", err)
    record_property("relative_error", err/max(float(expected.abs()), 1e-15))
    assert out["anchors"][0]["synthetic_relu"]["minimum_absolute_preactivation"] > 1
    # The independent oracle averages 1-score over eight blocks; the helper
    # averages 1-mean(score) per anchor. FP64 grouping can differ by an ULP.
    torch.testing.assert_close(out["loss"], direct_objective(f["moments"], f["transform"], targets), atol=2e-15, rtol=0)
    torch.testing.assert_close(fd, expected, atol=3e-9, rtol=2e-6)


def test_moment_autograd_oracle_ensemble_average_and_nonmutation():
    f = fixture()
    targets = cached(f)
    before = copy.deepcopy((f, targets))
    leaf = f["moments"].clone().requires_grad_()
    expected = direct_objective(leaf, f["transform"], targets)
    gradient, = torch.autograd.grad(expected, leaf)
    out = engine.gradient_alignment_partials(leaf, f["transform"], targets)
    torch.testing.assert_close(out["moment_gradient"], gradient, atol=2e-15, rtol=2e-14)
    single = [engine.gradient_alignment_partials(leaf, f["transform"],
              dict(policy=targets["policy"], anchors=(anchor,))) for anchor in targets["anchors"]]
    torch.testing.assert_close(out["loss"], sum(v["loss"] for v in single)/2, atol=0, rtol=0)
    torch.testing.assert_close(out["moment_gradient"], sum(v["moment_gradient"] for v in single)/2,
                               atol=2e-15, rtol=2e-14)
    assert leaf.grad is None
    assert not out["moment_gradient"].requires_grad
    tree_equal((f, targets), before)


@pytest.mark.parametrize("eps", [1e-4, 3e-5])
def test_original_lowrank_first_backward_interior_u_v_chain(eps, record_property):
    f = fixture()
    targets = cached(f)
    u, v = f["u"].clone().requires_grad_(), f["v"].clone().requires_grad_()
    m = LowRankMoments.apply(u, v, f["assignment"], f["material"], .2, 2)
    out = engine.gradient_alignment_partials(m, f["transform"], targets)
    m.backward(out["moment_gradient"])  # Exactly one original custom backward.
    generator = torch.Generator().manual_seed(405)
    du, dv = torch.randn(u.shape, generator=generator, dtype=u.dtype)*.2, torch.randn(v.shape, generator=generator, dtype=v.dtype)*.2
    expected = (u.grad*du).sum()+(v.grad*dv).sum()
    values = []
    for sign in (1, -1):
        p = logit_block(u.detach()+sign*eps*du, v.detach()+sign*eps*dv, f["assignment"], .2).softmax(1)
        values.append(direct_objective(p.T@f["material"]/len(u), f["transform"], targets))
    fd = (values[0]-values[1])/(2*eps)
    record_property("absolute_error", float((fd-expected).detach().abs()))
    record_property("relative_error", float((fd-expected).detach().abs())/max(float(expected.abs()),1e-15))
    torch.testing.assert_close(fd, expected, atol=3e-9, rtol=2e-6)


def test_row_rescaling_null_direction_and_original_reconstruction():
    f = fixture()
    targets = cached(f)
    out = engine.gradient_alignment_partials(f["moments"], f["transform"], targets)
    d = f["moments"] * torch.tensor([.2, -.1, .3])[:, None]
    assert abs(float((out["moment_gradient"]*d).sum())) < 1e-14
    scaled = f["moments"] * torch.tensor([1.2, .9, 1.3])[:, None]
    other = engine.gradient_alignment_partials(scaled, f["transform"], targets)
    torch.testing.assert_close(other["loss"], out["loss"], atol=3e-16, rtol=2e-14)


def test_fp32_model_fp64_moment_native_ready_not_native_certified():
    f = fixture(torch.float32)
    targets = cached(f)
    out = engine.gradient_alignment_partials(f["moments"], f["transform"], targets)
    assert out["loss"].dtype == out["moment_gradient"].dtype == torch.float64
    assert out["Hc"].dtype == out["Qc"].dtype == torch.float64
    assert targets["anchors"][0]["gradients"][0].dtype == torch.float32
    assert torch.isfinite(out["moment_gradient"]).all()


@pytest.mark.parametrize("bad", ["source_zero", "source_nan", "source_shape", "source_qnegative", "source_qsum",
                                 "source_qdtype", "source_trainable", "source_Sshape", "source_Strainable",
                                 "parameters_shape", "empty_ensemble"])
def test_malformed_source_and_zero_blocks_rejected(bad):
    f = fixture()
    if bad == "source_zero":
        f["ensemble"] = ((torch.zeros(4,3,dtype=torch.float64), torch.zeros(4,dtype=torch.float64),
                         torch.zeros(3,4,dtype=torch.float64), torch.zeros(3,dtype=torch.float64)),)
    elif bad == "source_nan":
        f["x"][0,0] = math.nan
    elif bad == "source_shape":
        f["x"] = f["x"][:, :2]
    elif bad == "source_qnegative":
        f["q"][0,0] = -.1
    elif bad == "source_qsum":
        f["q"][0,0] += .01
    elif bad == "source_qdtype":
        f["q"] = f["q"].float()
    elif bad == "source_trainable":
        f["x"].requires_grad_()
    elif bad == "source_Sshape":
        f["s"] = f["s"][:5]
    elif bad == "source_Strainable":
        f["s"].requires_grad_()
    elif bad == "parameters_shape":
        f["ensemble"][0][1].resize_(3)
    else:
        f["ensemble"] = ()
    with pytest.raises(ValueError):
        cached(f)


@pytest.mark.parametrize("bad", ["target_zero", "target_nan", "target_shape", "target_graph", "norm", "delta",
                                 "policy", "moments_nan", "mass_zero", "qbar_sum", "transform_scale"])
def test_malformed_moment_and_target_cache_rejected(bad):
    f = fixture()
    targets = cached(f)
    a = targets["anchors"][0]
    if bad == "target_zero":
        a["gradients"][0].zero_()
    elif bad == "target_nan":
        a["gradients"][0][0,0] = math.nan
    elif bad == "target_shape":
        a["gradients"][0].resize_(1)
    elif bad == "target_graph":
        a["gradients"][0].requires_grad_()
    elif bad == "norm":
        a["source_norms"][0].add_(.01)
    elif bad == "delta":
        a["delta"][0].add_(.01)
    elif bad == "policy":
        targets["policy"]["rho"] = .01
    elif bad == "moments_nan":
        f["moments"][0,1] = math.nan
    elif bad == "mass_zero":
        f["moments"][0,0] = 0
    elif bad == "qbar_sum":
        f["moments"][0,4] += 1e-8
    else:
        f["transform"]["scale"].zero_()
    with pytest.raises(ValueError):
        engine.gradient_alignment_partials(f["moments"], f["transform"], targets)


def test_stop_autocast_and_no_source_graph_retained():
    f = fixture()
    with pytest.raises(InterruptedError):
        engine.source_gradient_targets(f["ensemble"], f["x"], f["s"], f["q"], stop=lambda: True)
    targets = cached(f)
    with pytest.raises(InterruptedError):
        engine.gradient_alignment_partials(f["moments"], f["transform"], targets, stop=lambda: True)
    with torch.autocast("cpu", dtype=torch.bfloat16), pytest.raises(ValueError, match="autocast"):
        cached(f)
    assert all(p.grad_fn is None and not p.requires_grad for anchor in targets["anchors"]
               for p in (*anchor["parameters"], *anchor["gradients"], *anchor["source_norms"], *anchor["delta"]))


def test_finite_entries_with_overflowed_synthetic_norm_rejected():
    synthetic = (torch.tensor([1e200, -1e200], dtype=torch.float64),) * 4
    target = (torch.ones(2, dtype=torch.float64),) * 4
    with pytest.raises(ValueError, match="norm"):
        engine.smoothed_block_alignment(synthetic, target)


def test_equal_directions_have_declared_smoothing_floor_not_exact_zero():
    targets = (torch.tensor([.3, -.7], dtype=torch.float64),) * 4
    loss, _ = engine.smoothed_block_alignment(targets, targets)
    assert abs(float(loss) - 1e-6/(1+1e-6)) < 2e-16
