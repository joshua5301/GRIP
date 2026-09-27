import torch


class BalancedMoments(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, material, chunk_size, max_iter, tolerance, initial_dual):
        n, m = logits.shape
        dual = material.new_zeros(m) if initial_dual is None else initial_dual.detach().to(material).clone()
        for iteration in range(max_iter):
            columns = material.new_zeros(m)
            row_error = material.new_zeros(())
            for start in range(0, n, chunk_size):
                probability = (logits[start:start + chunk_size].to(material.dtype) + dual).softmax(1)
                columns += probability.sum(0) / n
                row_error = torch.maximum(row_error, (probability.sum(1) - 1).abs().max())
            column_error = float((m * columns - 1).abs().max())
            if column_error <= tolerance:
                break
            if not bool(torch.isfinite(columns).all()) or bool((columns <= 0).any()):
                raise FloatingPointError('Nonfinite or empty balanced assignment column')
            dual -= (m * columns).log()
            dual -= dual.mean()
        else:
            raise RuntimeError(f'Uniform assignment did not converge: column residual={column_error:.3g}')
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
        logits, material, dual = ctx.saved_tensors
        n, m = logits.shape
        gram = material.new_zeros(m, m)
        columns, rhs = material.new_zeros(m), material.new_zeros(m)
        for start in range(0, n, ctx.chunk_size):
            probability = (logits[start:start + ctx.chunk_size].to(material.dtype) + dual).softmax(1)
            direction = material[start:start + ctx.chunk_size] @ gradient.T
            centered = direction - (probability * direction).sum(1, keepdim=True)
            gram += probability.T @ probability / n
            columns += probability.sum(0) / n
            rhs += (probability * centered).sum(0) / n
        system = torch.diag(columns) - gram
        rhs -= rhs.mean()
        system += torch.ones_like(system) / m
        correction = torch.linalg.solve(system, rhs)
        residual = float((system @ correction - rhs).norm())
        if not bool(torch.isfinite(correction).all()) or residual > max(1e-12, 1e-7 * float(rhs.norm())):
            raise RuntimeError('Balanced assignment implicit derivative solve failed')
        result = torch.empty_like(logits)
        for start in range(0, n, ctx.chunk_size):
            probability = (logits[start:start + ctx.chunk_size].to(material.dtype) + dual).softmax(1)
            direction = material[start:start + ctx.chunk_size] @ gradient.T - correction
            direction -= (probability * direction).sum(1, keepdim=True)
            result[start:start + ctx.chunk_size] = (probability * direction / n).to(logits.dtype)
        return result, None, None, None, None, None
