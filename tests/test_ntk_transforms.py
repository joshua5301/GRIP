import torch

from src.ntk_transforms import fit_transform, TransformedMap
from src.ntk_risk import fit_representatives


def test_rms_removes_global_rescaling_and_power_zero_matches_baseline():
    torch.manual_seed(4)
    z = torch.randn(15, 5, dtype=torch.double)
    baseline, _ = fit_transform(z)
    scaled, _ = fit_transform(z * 7)
    power_zero, _ = fit_transform(z, kind='whiten', power=0)
    torch.testing.assert_close(baseline, scaled)
    torch.testing.assert_close(baseline, power_zero)


def test_all_maps_use_frozen_statistics_for_new_points():
    torch.manual_seed(4)
    z = torch.randn(15, 5, dtype=torch.double)
    for kind in ('rms', 'l2', 'whiten'):
        features, transform = fit_transform(z, kind=kind)
        torch.testing.assert_close(features.mean(0), torch.zeros(5, dtype=torch.double), atol=1e-12, rtol=0)
        torch.testing.assert_close(features.square().sum(1).mean(), torch.ones((), dtype=torch.double))
        query = z[:3].clone().requires_grad_()
        actual = transform(query)
        torch.testing.assert_close(actual, features[:3])
        gradient, = torch.autograd.grad(actual.square().sum(), query)
        assert bool(torch.isfinite(gradient).all())


def test_transformed_reconstruction_uses_same_mapping():
    torch.manual_seed(3)
    x = torch.randn(8, 3, dtype=torch.double)
    features, transform = fit_transform(x, kind='whiten', power=.5)
    mapping = TransformedMap(lambda value: value, transform)
    assignment = torch.arange(8) // 2
    fitted = fit_representatives(x, torch.eye(8, dtype=torch.double).to_sparse(), None,
                                 assignment, steps=2, feature_map=mapping, mapped_features=features)
    assert fitted['reconstruction_initial'] < 1e-20
    assert bool(torch.isfinite(fitted['x']).all())
