import pytest
import torch

from src.attention_assignment import AttentionAssignment
from src.moments import make_material
from src.soft_ce_partition import optimize_ce_assignment


@pytest.mark.parametrize("method", ["metric", "attention", "low_rank", "svd_UV"])
def test_chunked_moments_and_gradients_match_dense(method):
    torch.manual_seed(4)
    x, c = torch.randn(12, 5, dtype=torch.double), torch.randn(3, 5, dtype=torch.double)
    model = AttentionAssignment(x, c, 2, 0.3, method)
    material = torch.randn(12, 7, dtype=torch.double)
    if method == "svd_UV":
        logits = model.u @ model.v.T / (2**0.5 * model.tau)
    elif method == "low_rank":
        logits = (-torch.cdist(x, c).square() + model.u @ model.v.T / 2**0.5) / model.tau
    elif method == "metric":
        a, b = x @ model.query, c @ model.query
        logits = -(torch.cdist(x, c).square() + torch.cdist(a, b).square()) / model.tau
    else:
        a, b = x @ model.query, c @ model.key
        logits = a @ b.T / (2**0.5 * model.tau)
    expected = logits.softmax(1).T @ material / len(x)
    actual = model(material, 5)
    torch.testing.assert_close(actual, expected)
    direction = torch.randn_like(actual)
    params = tuple(model.parameters())
    exact = torch.autograd.grad((expected * direction).sum(), params)
    chunked = torch.autograd.grad((actual * direction).sum(), params)
    for first, second in zip(exact, chunked):
        torch.testing.assert_close(first, second)


@pytest.mark.parametrize("method", ["metric", "attention", "low_rank", "svd_UV"])
def test_custom_assignment_bilevel_smoke(method, tmp_path):
    torch.manual_seed(5)
    z = torch.randn(12, 2, dtype=torch.double)
    q = torch.randn(12, 2, dtype=torch.double).softmax(1)
    assignment = torch.arange(12) % 3
    model = AttentionAssignment(z, z[:3], 2, 1., method)
    expected = model(make_material(z, q), 4).detach()
    result = optimize_ce_assignment(
        z, q, assignment, assignment_model=model, steps=1, penalty=0.1,
        save_assignment=False, folder=tmp_path, checkpoint_steps=[0, 1],
        inner_loss_weighting="uniform", inner_tol=1e-5, cg_rtol=1e-3,
        inner_method="newton_first", cg_max_iter=512)
    torch.testing.assert_close(result["checkpoints"][0]["moments"].to(expected), expected)
    assert torch.isfinite(result["checkpoints"][1]["moments"]).all()


def test_distance_svd_reconstructs_centered_logits():
    torch.manual_seed(7)
    x, c = torch.randn(13, 4, dtype=torch.double), torch.randn(5, 4, dtype=torch.double)
    model = AttentionAssignment(x, c, 5, 0.3, "svd_UV")
    target = -torch.cdist(x, c).square()
    target -= target.mean(1, keepdim=True)
    torch.testing.assert_close(model.u @ model.v.T / 5**0.5, target)
    assert not bool(model.left.any()) and not bool(model.right.any())
    u, v = model.factors()
    torch.testing.assert_close((u @ v.T / 5**0.5).softmax(1), (target / 0.3).softmax(1))
