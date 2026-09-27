import torch

from src.anil_probe import fit_probe, normalize_features


def test_normalization_uses_train_features_only():
    z = torch.tensor([[1., 2.], [3., 4.], [100., 200.]], dtype=torch.double)
    ids = torch.tensor([0, 1])
    normalized, center, scale = normalize_features(z, ids)
    torch.testing.assert_close(center, torch.tensor([2., 3.], dtype=torch.double))
    torch.testing.assert_close(normalized[ids].mean(0), torch.zeros(2, dtype=torch.double))
    torch.testing.assert_close(normalized[ids].square().sum(1).mean(), torch.ones((), dtype=torch.double))
    changed = z.clone()
    changed[2] *= 100
    _, other_center, other_scale = normalize_features(changed, ids)
    torch.testing.assert_close(center, other_center)
    torch.testing.assert_close(scale, other_scale)


def test_polishing_reaches_same_convex_solution_from_different_starts():
    generator = torch.Generator().manual_seed(1)
    z = torch.randn(24, 4, generator=generator, dtype=torch.double)
    y = torch.arange(24) % 3
    first = fit_probe(z, y, 3, 0, .1, .2, steps=5, polish_steps=300, grad_tol=1e-6)
    second = fit_probe(z, y, 3, 4, .3, .2, steps=5, polish_steps=300, grad_tol=1e-6)
    assert first['objective_after'] <= first['objective_before']
    assert first['grad_after'] < 1e-5
    assert second['grad_after'] < 1e-5
    assert abs(first['objective_after'] - second['objective_after']) < 1e-8
