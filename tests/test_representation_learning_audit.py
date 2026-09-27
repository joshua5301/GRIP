import torch

from src.ntk_transforms import fit_transform
from src.representation_learning_audit import cell_means, fit_soft_head


def test_weighted_probe_equals_duplicated_samples():
    torch.manual_seed(3)
    z = torch.randn(4, 3, dtype=torch.double)
    q = torch.randn(4, 2, dtype=torch.double).softmax(1)
    counts = torch.tensor([1, 2, 3, 2])
    first = fit_soft_head(z, q, counts.double() / counts.sum(), .1)
    second = fit_soft_head(z.repeat_interleave(counts, dim=0), q.repeat_interleave(counts, dim=0),
                           torch.full((8,), 1 / 8, dtype=torch.double), .1)
    torch.testing.assert_close(first['weight'], second['weight'], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(first['bias'], second['bias'], atol=1e-6, rtol=1e-5)
    assert first['grad'] < 1e-6


def test_s2x_identity_realization_commutes_with_centering_scaling():
    torch.manual_seed(5)
    x = torch.randn(9, 4, dtype=torch.double)
    s = torch.eye(9, dtype=torch.double) * .7 + torch.ones(9, 9, dtype=torch.double) / 30
    h = s @ s @ x
    q = torch.randn(9, 3, dtype=torch.double).softmax(1)
    assignment = torch.arange(9) % 3
    z, mapping = fit_transform(h)
    target, labels, mass = cell_means(z, q, assignment)
    raw, _, _ = cell_means(h, q, assignment)
    torch.testing.assert_close(mapping(raw), target)
    torch.testing.assert_close(mass.sum(), torch.tensor(1., dtype=torch.double))
    torch.testing.assert_close(labels.sum(1), torch.ones(3, dtype=torch.double))
