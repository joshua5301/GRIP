import unittest

import torch

from src.kernel_commutation_probe import summarize


class KernelCommutationMath(unittest.TestCase):
    def test_linear_zero_and_nonlinear_exact_counterexample(self):
        p = torch.tensor([[.75, .25], [.25, .75]], dtype=torch.float64)
        h = torch.tensor([[-1.], [1.]], dtype=torch.float64)
        c = p.T @ h / p.sum(0)[:, None]
        linear, mean_h, _ = summarize(p, h, c)
        self.assertEqual(linear['relative_gap_squared'], 0.0)
        self.assertTrue(torch.equal(mean_h, c))
        nonlinear, mean_phi, mass = summarize(p, h.square(), c.square())
        # Each mean feature is exactly1 but phi(mean input) is exactly1/4.
        self.assertTrue(torch.equal(mean_phi, torch.ones_like(mean_phi)))
        self.assertTrue(torch.equal(mass, torch.ones_like(mass)))
        self.assertEqual(nonlinear['fixed_source_phi_energy'], 1.0)
        self.assertEqual(nonlinear['uniform_cell_gap_squared'], .5625)
        self.assertEqual(nonlinear['relative_gap_squared'], .5625)
        for bad in (p * .5, -p, torch.tensor([[1., 0.], [1., 0.]], dtype=torch.float64)):
            with self.assertRaises(ValueError):
                summarize(bad, h.square(), c.square())


if __name__ == '__main__':
    unittest.main()
