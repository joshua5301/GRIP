import pytest
import torch

from src.distance_finetune import distance_logits, svd_factors
from src.low_rank_assignment import logit_block
from src.moment_seeding import normalized_variance_features
from src.moments import make_material
from src.soft_ce_partition import optimize_ce_assignment


def test_svd_initialization_matches_low_rank_logit_scaling():
    generator = torch.Generator().manual_seed(4)
    logits = torch.randn(17, 5, generator=generator, dtype=torch.float64)
    logits -= logits.mean(1, keepdim=True)
    factors = svd_factors(logits, 5)
    recovered = logit_block(*factors, torch.zeros_like(logits), 0.05)
    torch.testing.assert_close(recovered, logits, atol=1e-6, rtol=1e-6)
    errors = []
    for rank in (1, 2, 4):
        u, v = svd_factors(logits, rank)
        errors.append(float((u.double() @ v.double().T / rank**0.5 - logits).norm()))
    assert errors[0] >= errors[1] >= errors[2]


def test_distance_initialization_matches_normalized_variance_metric():
    generator = torch.Generator().manual_seed(5)
    h = torch.randn(15, 4, generator=generator, dtype=torch.float64)
    q = torch.randn(15, 3, generator=generator, dtype=torch.float64).softmax(1)
    assignment = torch.arange(15) % 3
    logits, scale = distance_logits(h, q, assignment, 0.2)
    x = h - h.mean(0)
    x /= x.square().sum(1).mean().sqrt()
    z, _ = normalized_variance_features(x, q, 0.2)
    centers = torch.stack([z[assignment == j].mean(0) for j in range(3)])
    distance = torch.cdist(z, centers).square()
    expected = -(distance - distance.mean(1, keepdim=True)) / scale
    torch.testing.assert_close(logits, expected)
    torch.testing.assert_close(logits.softmax(1), (-distance / scale).softmax(1))
    assert scale > 0
    with pytest.raises(ValueError):
        svd_factors(logits, 4)


def test_optimizer_uses_supplied_factors_at_step_zero(tmp_path):
    z = torch.tensor([[-1.0], [-0.5], [0.5], [1.0]], dtype=torch.float64)
    q = torch.tensor([[0.8, 0.2], [0.7, 0.3], [0.3, 0.7], [0.2, 0.8]], dtype=z.dtype)
    assignment = torch.tensor([0, 0, 1, 1])
    logits = torch.tensor([[1., -1.], [.5, -.5], [-.5, .5], [-1., 1.]], dtype=z.dtype)
    u, v = svd_factors(logits, 2)
    result = optimize_ce_assignment(
        z, q, assignment, assignment_rank=2, initial_factors=(u, v),
        base_logits=torch.zeros_like(logits), steps=0, penalty=0.1,
        checkpoint_steps=[0], folder=tmp_path, save_assignment=False,
    )
    expected = logits.softmax(1).T @ make_material(z, q) / len(z)
    torch.testing.assert_close(result["checkpoints"][0]["moments"], expected, atol=1e-7, rtol=1e-7)
