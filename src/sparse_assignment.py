import torch

from src.initialization import cell_means


def initialize_sparse(z, assignment, cells, k, mixing=0.05):
    if not isinstance(k, int) or not 2 <= k <= cells or not 0 < mixing < 1:
        raise ValueError("Sparse assignments require 2 <= k <= cells and 0 < mixing < 1")
    centers = cell_means(z, assignment, cells)
    indices = []
    for start in range(0, len(z), 1024):
        block = z[start:start + 1024]
        own = assignment[start:start + len(block)]
        distance = block.square().sum(1, keepdim=True) + centers.square().sum(1) - 2 * block @ centers.T
        distance.scatter_(1, own[:, None], torch.inf)
        nearest = distance.argsort(dim=1, stable=True)[:, :k - 1]
        indices.append(torch.cat((own[:, None], nearest), dim=1))
    indices = torch.cat(indices)
    probabilities = z.new_full((len(z), k), mixing / k)
    probabilities[:, 0] += 1 - mixing
    return indices, probabilities.log().detach().requires_grad_()


class SparseMoments(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, indices, material, cells, chunk_size):
        probability = logits.softmax(1)
        moments = material.new_zeros(cells, material.shape[1])
        for start in range(0, len(logits), chunk_size):
            stop = start + chunk_size
            for slot in range(indices.shape[1]):
                moments.index_add_(
                    0, indices[start:stop, slot],
                    probability[start:stop, slot, None] * material[start:stop],
                )
        ctx.save_for_backward(probability, indices, material)
        ctx.chunk_size = chunk_size
        return moments / len(logits)

    @staticmethod
    def backward(ctx, gradient):
        probability, indices, material = ctx.saved_tensors
        scores = torch.empty_like(probability)
        material_gradient = torch.zeros_like(material) if ctx.needs_input_grad[2] else None
        for start in range(0, len(probability), ctx.chunk_size):
            stop = start + ctx.chunk_size
            for slot in range(indices.shape[1]):
                selected = gradient[indices[start:stop, slot]]
                scores[start:stop, slot] = (selected * material[start:stop]).sum(1) / len(probability)
                if material_gradient is not None:
                    material_gradient[start:stop] += probability[start:stop, slot, None] * selected / len(probability)
        logits_gradient = probability * (scores - (scores * probability).sum(1, keepdim=True))
        return logits_gradient, None, material_gradient, None, None
