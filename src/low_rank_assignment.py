import math

import torch

from src.soft_ridge_partition import initial_logits


def initialize_factors(assignment, clusters, rank, seed=0):
    if not isinstance(rank, int) or not 1 <= rank <= min(len(assignment), clusters):
        raise ValueError('rank must be a positive integer no larger than min(nodes, cells)')
    generator = torch.Generator(device=assignment.device).manual_seed(seed)
    u = torch.zeros(len(assignment), rank, device=assignment.device)
    v = torch.randn(clusters, rank, generator=generator, device=assignment.device)
    return u.requires_grad_(), v.requires_grad_()


def logit_block(u, v, assignment, mixing):
    return initial_logits(assignment, len(v), mixing, u.dtype) + u @ v.T / math.sqrt(u.shape[1])


class LowRankLogits(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, v, assignment, mixing, chunk_size):
        ctx.save_for_backward(u, v)
        ctx.chunk_size = chunk_size
        result = u.new_empty(len(u), len(v))
        for start in range(0, len(u), chunk_size):
            end = start + chunk_size
            result[start:end] = logit_block(u[start:end], v, assignment[start:end], mixing)
        return result

    @staticmethod
    def backward(ctx, gradient):
        u, v = ctx.saved_tensors
        du, dv = torch.empty_like(u), torch.zeros_like(v)
        scale = math.sqrt(u.shape[1])
        for start in range(0, len(u), ctx.chunk_size):
            end = start + ctx.chunk_size
            block = gradient[start:end] / scale
            du[start:end] = block @ v
            dv += block.T @ u[start:end]
        return du, dv, None, None, None


class LowRankMoments(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, v, assignment, material, mixing, chunk_size):
        ctx.save_for_backward(u, v, assignment, material)
        ctx.mixing, ctx.chunk_size = mixing, chunk_size
        result = material.new_zeros(len(v), material.shape[1])
        for start in range(0, len(u), chunk_size):
            end = start + chunk_size
            probability = logit_block(u[start:end], v, assignment[start:end], mixing).to(material.dtype).softmax(1)
            result += probability.T @ material[start:end] / len(u)
        return result

    @staticmethod
    def backward(ctx, gradient):
        u, v, assignment, material = ctx.saved_tensors
        du, dv = torch.empty_like(u), torch.zeros_like(v)
        scale = math.sqrt(u.shape[1])
        for start in range(0, len(u), ctx.chunk_size):
            end = start + ctx.chunk_size
            probability = logit_block(u[start:end], v, assignment[start:end], ctx.mixing).to(material.dtype).softmax(1)
            direction = material[start:end] @ gradient.T / len(u)
            block = (probability * (direction - (probability * direction).sum(1, keepdim=True))).to(u.dtype) / scale
            du[start:end] = block @ v
            dv += block.T @ u[start:end]
        return du, dv, None, None, None, None
