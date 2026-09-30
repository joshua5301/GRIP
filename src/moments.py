import numpy as np
import torch


class AssignmentMoments(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, material, chunk_size):
        ctx.save_for_backward(logits, material)
        ctx.chunk_size = chunk_size
        result = material.new_zeros(logits.shape[1], material.shape[1])
        for start in range(0, len(logits), chunk_size):
            end = start + chunk_size
            probability = logits[start:end].to(material.dtype).softmax(1)
            result.add_(probability.T @ material[start:end], alpha=1 / len(logits))
        return result

    @staticmethod
    def backward(ctx, gradient):
        logits, material = ctx.saved_tensors
        result = torch.empty_like(logits)
        for start in range(0, len(logits), ctx.chunk_size):
            end = start + ctx.chunk_size
            probability = logits[start:end].to(material.dtype).softmax(1)
            direction = material[start:end] @ gradient.T / len(logits)
            value = probability * (direction - (direction * probability).sum(1, keepdim=True))
            result[start:end] = value.to(logits.dtype)
        return result, None, None


def make_material(z, q):
    return torch.cat((z.new_ones(len(z), 1), z, q), dim=1)


def initial_logits(assignment, clusters, mixing=0.05, dtype=torch.float32):
    if not 0 < mixing < 1:
        raise ValueError("mixing must lie strictly between zero and one")
    result = torch.full(
        (len(assignment), clusters), np.log(mixing / clusters), dtype=dtype, device=assignment.device
    )
    result.scatter_(1, assignment[:, None], float(np.log(1 - mixing + mixing / clusters)))
    return result


def decode_moments(moments, dimension):
    mass = moments[:, 0]
    centers = moments[:, 1 : dimension + 1] / mass[:, None]
    labels = moments[:, dimension + 1 :] / mass[:, None]
    return centers, labels, mass


def augmented(z):
    return torch.cat((z, z.new_ones(len(z), 1)), dim=1)
