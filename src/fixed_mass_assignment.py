"""Fixed positive source-derived assignment marginals, with an implicit dual.

This helper deliberately leaves the existing equal-mass implementation alone.
The target is a constant constraint, not a differentiable or learned parameter.
"""
import math
from numbers import Integral, Real

import torch

from src.balanced_assignment import BalancedMoments


def validate_mass_target(target, cells=None):
    if (not torch.is_tensor(target) or target.ndim != 1 or len(target) < 1
            or target.dtype != torch.float64 or target.requires_grad
            or not bool(torch.isfinite(target).all()) or bool((target <= 0).any())
            or abs(float(target.sum()) - 1) > 1e-12
            or (cells is not None and len(target) != cells)):
        raise ValueError("Fixed mass target must be a detached positive float64 probability vector")


class FixedMassMoments(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, material, target, chunk_size, max_iter, tolerance, initial_dual):
        if (isinstance(chunk_size, bool) or not isinstance(chunk_size, Integral) or chunk_size < 1
                or isinstance(max_iter, bool) or not isinstance(max_iter, Integral) or max_iter < 1
                or isinstance(tolerance, bool) or not isinstance(tolerance, Real)
                or not math.isfinite(tolerance) or not 0 < tolerance < 1):
            raise ValueError("Invalid fixed-mass solver controls")
        if not torch.is_tensor(logits) or logits.ndim != 2 or min(logits.shape) < 1:
            raise ValueError("Fixed mass logits must be a nonempty matrix")
        n, m = logits.shape
        validate_mass_target(target, m)
        if target.device != material.device or material.dtype != torch.float64:
            raise ValueError("Fixed mass target and float64 material must share a device")
        if n < 1 or material.ndim != 2 or len(material) != n:
            raise ValueError("Fixed mass logits and material must have aligned nonempty rows")
        if initial_dual is not None and (
            not torch.is_tensor(initial_dual) or initial_dual.shape != (m,)
            or not bool(torch.isfinite(initial_dual).all())
            or abs(float(initial_dual.mean())) > 1e-10
        ):
            raise ValueError("Fixed mass dual must be finite and in the zero-mean gauge")
        dual = material.new_zeros(m) if initial_dual is None else initial_dual.detach().to(material).clone()
        for iteration in range(max_iter):
            columns = material.new_zeros(m)
            row_error = material.new_zeros(())
            for start in range(0, n, chunk_size):
                probability = (logits[start:start + chunk_size].to(material.dtype) + dual).softmax(1)
                columns += probability.sum(0) / n
                row_error = torch.maximum(row_error, (probability.sum(1) - 1).abs().max())
            if not bool(torch.isfinite(columns).all()) or bool((columns <= 0).any()):
                raise FloatingPointError("Nonfinite or empty fixed-mass assignment column")
            column_error = float((columns / target - 1).abs().max())
            if column_error <= tolerance and float(row_error) <= tolerance:
                break
            dual -= (columns / target).log()
            dual -= dual.mean()
        else:
            raise RuntimeError(f"Initial-mass assignment did not converge: column residual={column_error:.3g}")
        moments = material.new_zeros(m, material.shape[1])
        for start in range(0, n, chunk_size):
            probability = (logits[start:start + chunk_size].to(material.dtype) + dual).softmax(1)
            moments += probability.T @ material[start:start + chunk_size] / n
        ctx.save_for_backward(logits, material, dual)
        ctx.chunk_size = chunk_size
        diagnostic = material.new_tensor([iteration + 1, float(row_error), column_error])
        ctx.mark_non_differentiable(dual, diagnostic)
        return moments, dual, diagnostic

    @staticmethod
    def backward(ctx, gradient, dual_gradient, diagnostic_gradient):
        # BalancedMoments' correction uses the actual columns diag(c)-P'P/N;
        # its derivation holds for any fixed positive marginal, not only 1/K.
        result = BalancedMoments.backward(ctx, gradient, dual_gradient, diagnostic_gradient)[0]
        return result, None, None, None, None, None, None
