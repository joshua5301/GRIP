import pytest
import torch

from src.attention_assignment import AttentionAssignment
from src.moments import make_material
from src.soft_ce_partition import optimize_ce_assignment


@pytest.mark.parametrize("method", ["metric", "attention"])
def test_chunked_moments_and_gradients_match_dense(method):
    torch.manual_seed(4)
    x, c = torch.randn(12, 5, dtype=torch.double), torch.randn(3, 5, dtype=torch.double)
    model = AttentionAssignment(x, c, 2, 0.3, method)
    material = torch.randn(12, 7, dtype=torch.double)
    a = x @ model.query
    b = c @ (model.query if method == "metric" else model.key)
    if method == "metric":
        logits = -(torch.cdist(x, c).square() + torch.cdist(a, b).square()) / model.tau
    else:
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


@pytest.mark.parametrize("method", ["metric", "attention"])
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
