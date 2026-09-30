import torch


@torch.no_grad()
def matrix_free_correction(probability, columns, rhs, chunk_size, max_iter, rtol):
    n, m = probability.shape
    diagonal = columns.clone() + 1 / m
    for block in probability.split(chunk_size):
        diagonal -= block.square().sum(0) / n

    def multiply(value):
        return columns * value - probability.T @ (probability @ value) / n + value.mean()

    rhs = rhs - rhs.mean()
    solution = torch.zeros_like(rhs)
    residual = rhs.clone()
    direction = residual / diagonal
    product = (residual * direction).sum()
    target = max(1e-12, rtol * float(rhs.norm()))
    for iteration in range(max_iter):
        if float(residual.norm()) <= target:
            break
        image = multiply(direction)
        curvature = (direction * image).sum()
        if not torch.isfinite(curvature) or float(curvature) <= 0:
            break
        alpha = product / curvature
        solution += alpha * direction
        residual -= alpha * image
        restart = (iteration + 1) % 50 == 0 or float(residual.norm()) <= target
        if restart:
            residual = rhs - multiply(solution)
        preconditioned = residual / diagonal
        next_product = (residual * preconditioned).sum()
        direction = preconditioned if restart else preconditioned + (next_product / product) * direction
        product = next_product
    if not bool(torch.isfinite(solution).all()) or float((rhs - multiply(solution)).norm()) > target:
        raise RuntimeError("Matrix-free balancing derivative did not converge; increase balance_cg_steps")
    return solution


class CachedBalancedMoments(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, material, chunk_size, max_iter, tolerance, initial_dual, cg_steps, cg_rtol):
        n, m = logits.shape
        probability = torch.empty((n, m), dtype=material.dtype, device=material.device)
        for start in range(0, n, chunk_size):
            block = logits[start : start + chunk_size].to(material.dtype)
            probability[start : start + chunk_size] = (block - block.max(1, keepdim=True).values).exp()
        dual = material.new_zeros(m) if initial_dual is None else initial_dual.to(material)
        column_scale = (dual - dual.max()).exp()
        for iteration in range(4 * max_iter):
            row_scale = (probability @ column_scale).reciprocal()
            columns = column_scale * (probability.T @ row_scale) / n
            error = float((m * columns - 1).abs().max())
            if error <= 0.95 * tolerance:
                break
            if not bool(torch.isfinite(columns).all()) or bool((columns <= 0).any()):
                raise FloatingPointError('Cached balancing underflow/overflow; use balance_backend="chunked"')
            column_scale /= m * columns
            column_scale /= column_scale.max()
        else:
            raise RuntimeError(
                f"Uniform assignment did not converge after {4 * max_iter} iterations: column residual={error:.3g}"
            )
        for start in range(0, n, chunk_size):
            probability[start : start + chunk_size] *= (
                row_scale[start : start + chunk_size, None] * column_scale
            )
        row_error = float((probability.sum(1) - 1).abs().max())
        columns = probability.sum(0) / n
        column_error = float((m * columns - 1).abs().max())
        if max(row_error, column_error) > tolerance or not bool((column_scale > 0).all()):
            raise FloatingPointError("Cached balancing marginal check failed")
        dual = column_scale.log()
        dual -= dual.mean()
        moments = probability.T @ material / n
        ctx.save_for_backward(probability, material, columns)
        ctx.chunk_size, ctx.cg_steps, ctx.cg_rtol = chunk_size, cg_steps, cg_rtol
        ctx.logit_dtype = logits.dtype
        diagnostic = material.new_tensor([iteration + 1, row_error, column_error])
        ctx.mark_non_differentiable(dual, diagnostic)
        return moments, dual, diagnostic

    @staticmethod
    def backward(ctx, gradient, dual_gradient, diagnostic_gradient):
        probability, material, columns = ctx.saved_tensors
        n, m = probability.shape
        rhs = material.new_zeros(m)
        for start in range(0, n, ctx.chunk_size):
            block = probability[start : start + ctx.chunk_size]
            direction = material[start : start + ctx.chunk_size] @ gradient.T
            direction -= (block * direction).sum(1, keepdim=True)
            rhs += (block * direction).sum(0) / n
        correction = matrix_free_correction(
            probability, columns, rhs, ctx.chunk_size, ctx.cg_steps, ctx.cg_rtol
        )
        result = torch.empty(probability.shape, dtype=ctx.logit_dtype, device=probability.device)
        for start in range(0, n, ctx.chunk_size):
            block = probability[start : start + ctx.chunk_size]
            direction = material[start : start + ctx.chunk_size] @ gradient.T - correction
            direction -= (block * direction).sum(1, keepdim=True)
            result[start : start + ctx.chunk_size] = (block * direction / n).to(ctx.logit_dtype)
        return result, None, None, None, None, None, None, None


class BalancedMoments(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, material, chunk_size, max_iter, tolerance, initial_dual):
        n, m = logits.shape
        dual = material.new_zeros(m) if initial_dual is None else initial_dual.detach().to(material).clone()
        for iteration in range(max_iter):
            columns = material.new_zeros(m)
            row_error = material.new_zeros(())
            for start in range(0, n, chunk_size):
                probability = (logits[start : start + chunk_size].to(material.dtype) + dual).softmax(1)
                columns += probability.sum(0) / n
                row_error = torch.maximum(row_error, (probability.sum(1) - 1).abs().max())
            column_error = float((m * columns - 1).abs().max())
            if column_error <= tolerance:
                break
            if not bool(torch.isfinite(columns).all()) or bool((columns <= 0).any()):
                raise FloatingPointError("Nonfinite or empty balanced assignment column")
            dual -= (m * columns).log()
            dual -= dual.mean()
        else:
            raise RuntimeError(f"Uniform assignment did not converge: column residual={column_error:.3g}")
        moments = material.new_zeros(m, material.shape[1])
        for start in range(0, n, chunk_size):
            probability = (logits[start : start + chunk_size].to(material.dtype) + dual).softmax(1)
            moments += probability.T @ material[start : start + chunk_size] / n
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
            probability = (logits[start : start + ctx.chunk_size].to(material.dtype) + dual).softmax(1)
            direction = material[start : start + ctx.chunk_size] @ gradient.T
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
            raise RuntimeError("Balanced assignment implicit derivative solve failed")
        result = torch.empty_like(logits)
        for start in range(0, n, ctx.chunk_size):
            probability = (logits[start : start + ctx.chunk_size].to(material.dtype) + dual).softmax(1)
            direction = material[start : start + ctx.chunk_size] @ gradient.T - correction
            direction -= (probability * direction).sum(1, keepdim=True)
            result[start : start + ctx.chunk_size] = (probability * direction / n).to(logits.dtype)
        return result, None, None, None, None, None
