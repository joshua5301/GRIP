import pytest
import torch

from src.balanced_assignment import BalancedMoments


def test_marginals_and_global_moments():
    generator = torch.Generator().manual_seed(19)
    logits = torch.randn(11, 4, generator=generator, dtype=torch.double)
    features = torch.randn(11, 3, generator=generator, dtype=torch.double)
    material = torch.cat((torch.ones(11, 1, dtype=torch.double), features), 1)
    moments, dual, diagnostics = BalancedMoments.apply(logits, material, 3, 1000, 1e-12, None)
    probability = (logits + dual).softmax(1)
    torch.testing.assert_close(probability.sum(1), torch.ones(11, dtype=torch.double))
    torch.testing.assert_close(probability.sum(0), torch.full((4,), 11 / 4, dtype=torch.double), atol=1e-11, rtol=0)
    torch.testing.assert_close(moments, probability.T @ material / 11)
    torch.testing.assert_close(moments.sum(0), material.mean(0))
    assert diagnostics[2] <= 1e-12
    warmed, _, _ = BalancedMoments.apply(logits, material, 3, 1000, 1e-12, dual)
    torch.testing.assert_close(warmed, moments)


def test_balanced_derivative_and_gauge_invariance():
    generator = torch.Generator().manual_seed(21)
    logits = torch.randn(6, 3, generator=generator, dtype=torch.double).requires_grad_()
    material = torch.randn(6, 4, generator=generator, dtype=torch.double)
    def mapping(value):
        return BalancedMoments.apply(value, material, 2, 1000, 1e-13, None)[0]
    assert torch.autograd.gradcheck(mapping, (logits,), eps=1e-5, atol=1e-7, rtol=1e-4)
    output = mapping(logits)
    gradient, = torch.autograd.grad(output.square().sum(), logits)
    torch.testing.assert_close(gradient.sum(0), torch.zeros(3, dtype=torch.double), atol=1e-11, rtol=0)
    torch.testing.assert_close(gradient.sum(1), torch.zeros(6, dtype=torch.double), atol=1e-11, rtol=0)
    shifted = logits.detach() + torch.arange(6, dtype=torch.double)[:, None] + torch.arange(3, dtype=torch.double)
    torch.testing.assert_close(mapping(shifted), output, atol=1e-11, rtol=1e-11)


def test_unconverged_balancing_is_rejected():
    logits = torch.tensor([[5., 0., 0.]] * 7, dtype=torch.double)
    with pytest.raises(RuntimeError, match='Uniform assignment did not converge'):
        BalancedMoments.apply(logits, torch.ones(7, 1, dtype=torch.double), 3, 1, 1e-12, None)
