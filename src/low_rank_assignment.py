import math

import torch

from src.moments import initial_logits


def assignment_inputs(z, q, mode):
    if mode == "features":
        return z.detach().float()
    if mode == "features_labels":
        return torch.cat((z.detach(), q.detach()), dim=1).float()
    raise ValueError("Encoder input must be features or features_labels")


def initialize_encoder(inputs, clusters, rank, seed=0):
    if not isinstance(rank, int) or not 1 <= rank <= min(len(inputs), clusters):
        raise ValueError("rank must be a positive integer no larger than min(nodes, cells)")
    generator = torch.Generator(device=inputs.device).manual_seed(seed)
    weight = inputs.new_zeros(inputs.shape[1], rank)
    v = torch.randn(clusters, rank, generator=generator, device=inputs.device, dtype=inputs.dtype)
    return weight.requires_grad_(), v.requires_grad_()


def initialize_factors(assignment, clusters, rank, seed=0, u_std=0.0):
    if not isinstance(rank, int) or not 1 <= rank <= min(len(assignment), clusters):
        raise ValueError("rank must be a positive integer no larger than min(nodes, cells)")
    generator = torch.Generator(device=assignment.device).manual_seed(seed)
    if not math.isfinite(u_std) or u_std < 0:
        raise ValueError("Invalid factor initialization scale")
    v = torch.randn(clusters, rank, generator=generator, device=assignment.device)
    u = (torch.randn(len(assignment), rank, generator=generator, device=assignment.device) * u_std
         if u_std else torch.zeros(len(assignment), rank, device=assignment.device))
    return u.requires_grad_(), v.requires_grad_()


def initialize_mlp(inputs, clusters, rank, hidden=64, seed=0):
    if not isinstance(hidden, int) or hidden < 1:
        raise ValueError("encoder_hidden must be a positive integer")
    _, v = initialize_encoder(inputs, clusters, rank, seed)
    generator = torch.Generator(device=inputs.device).manual_seed(seed + 1)
    first = inputs.new_empty(inputs.shape[1], hidden)
    first.uniform_(-math.sqrt(6 / inputs.shape[1]), math.sqrt(6 / inputs.shape[1]), generator=generator)
    parameters = [first, inputs.new_zeros(hidden), inputs.new_zeros(hidden, rank), inputs.new_zeros(rank)]
    return [p.requires_grad_() for p in parameters], v


def encode_nodes(inputs, parameters):
    if len(parameters) == 1:
        return inputs @ parameters[0]
    first, bias, last, output_bias = parameters
    return (inputs @ first + bias).relu() @ last + output_bias


def saved_encoder_nodes(z, q, saved):
    inputs = assignment_inputs(z, q, saved["assignment_input"])
    parameters = saved["encoder_parameters"] if "encoder_parameters" in saved else [saved["weight"]]
    return encode_nodes(inputs, [p.to(inputs) for p in parameters])


def logit_block(u, v, assignment, mixing):
    base = assignment if assignment.ndim == 2 else initial_logits(assignment, len(v), mixing, u.dtype)
    return base + u @ v.T / math.sqrt(u.shape[1])


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
            probability = (
                logit_block(u[start:end], v, assignment[start:end], mixing).to(material.dtype).softmax(1)
            )
            result += probability.T @ material[start:end] / len(u)
        return result

    @staticmethod
    def backward(ctx, gradient):
        u, v, assignment, material = ctx.saved_tensors
        du, dv = torch.empty_like(u), torch.zeros_like(v)
        dm = torch.empty_like(material) if ctx.needs_input_grad[3] else None
        scale = math.sqrt(u.shape[1])
        for start in range(0, len(u), ctx.chunk_size):
            end = start + ctx.chunk_size
            probability = (
                logit_block(u[start:end], v, assignment[start:end], ctx.mixing).to(material.dtype).softmax(1)
            )
            if dm is not None:
                dm[start:end] = probability @ gradient / len(u)
            direction = material[start:end] @ gradient.T / len(u)
            block = (probability * (direction - (probability * direction).sum(1, keepdim=True))).to(
                u.dtype
            ) / scale
            du[start:end] = block @ v
            dv += block.T @ u[start:end]
        return du, dv, None, dm, None, None


class CachedLowRankMoments(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, v, assignment, material, mixing, chunk_size):
        probability = material.new_empty(len(u), len(v))
        result = material.new_zeros(len(v), material.shape[1])
        for start in range(0, len(u), chunk_size):
            end = start + chunk_size
            block = logit_block(u[start:end], v, assignment[start:end], mixing).to(material.dtype).softmax(1)
            probability[start:end] = block
            result += block.T @ material[start:end] / len(u)
        ctx.save_for_backward(u, v, material, probability)
        ctx.chunk_size = chunk_size
        return result

    @staticmethod
    def backward(ctx, gradient):
        u, v, material, probability = ctx.saved_tensors
        du, dv = torch.empty_like(u), torch.zeros_like(v)
        dm = torch.empty_like(material) if ctx.needs_input_grad[3] else None
        scale = math.sqrt(u.shape[1])
        for start in range(0, len(u), ctx.chunk_size):
            end = start + ctx.chunk_size
            p = probability[start:end]
            if dm is not None:
                dm[start:end] = p @ gradient / len(u)
            direction = material[start:end] @ gradient.T / len(u)
            block = (p * (direction - (p * direction).sum(1, keepdim=True))).to(u.dtype) / scale
            du[start:end] = block @ v
            dv += block.T @ u[start:end]
        return du, dv, None, dm, None, None


class WeightedLowRankMoments(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, v, weights, assignment, material, mixing, chunk_size):
        ctx.save_for_backward(u, v, weights, assignment, material)
        ctx.mixing, ctx.chunk_size = mixing, chunk_size
        result = material.new_zeros(len(v), material.shape[1])
        for start in range(0, len(u), chunk_size):
            end = start + chunk_size
            probability = (
                logit_block(u[start:end], v, assignment[start:end], mixing).to(material.dtype).softmax(1)
            )
            result += probability.T @ (weights[start:end, None] * material[start:end]) / len(u)
        return result

    @staticmethod
    def backward(ctx, gradient):
        u, v, weights, assignment, material = ctx.saved_tensors
        du, dv, dw = torch.empty_like(u), torch.zeros_like(v), torch.empty_like(weights)
        scale = math.sqrt(u.shape[1])
        for start in range(0, len(u), ctx.chunk_size):
            end = start + ctx.chunk_size
            probability = (
                logit_block(u[start:end], v, assignment[start:end], ctx.mixing).to(material.dtype).softmax(1)
            )
            direction = material[start:end] @ gradient.T / len(u)
            expectation = (probability * direction).sum(1, keepdim=True)
            dw[start:end] = expectation[:, 0]
            block = (weights[start:end, None] * probability * (direction - expectation)).to(u.dtype) / scale
            du[start:end] = block @ v
            dv += block.T @ u[start:end]
        return du, dv, dw, None, None, None, None


def normalized_node_weights(logits):
    log_weights = logits.double().log_softmax(0) + math.log(len(logits))
    weights = log_weights.exp()
    return weights, (weights * log_weights).mean()
