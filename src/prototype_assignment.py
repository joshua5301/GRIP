import torch


def initialize_prototypes(features, assignment, cells):
    counts = torch.bincount(assignment, minlength=cells).to(features)
    if bool((counts == 0).any()):
        raise ValueError("Prototype initialization requires nonempty cells")
    centers = features.new_zeros(cells, features.shape[1]).index_add_(0, assignment, features)
    return (centers / counts[:, None]).detach().requires_grad_()


def prototype_logits(features, prototypes, temperature):
    return (2 * features @ prototypes.T - prototypes.square().sum(1)[None, :]) / temperature
