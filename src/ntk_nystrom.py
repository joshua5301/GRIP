import numpy as np
import torch

from src.ntk_risk import tangent_kernel


class NystromGCN:
    def __init__(self, landmarks, transform, input_dim):
        self.landmarks = landmarks
        self.transform = transform
        self.input_dim = input_dim

    def __call__(self, raw):
        return tangent_kernel(raw / np.sqrt(self.input_dim), self.landmarks) @ self.transform

    def state_dict(self):
        return dict(landmarks=self.landmarks.cpu(), transform=self.transform.cpu(), input_dim=self.input_dim)

    @classmethod
    def from_state(cls, state, device):
        return cls(state['landmarks'].to(device), state['transform'].to(device), state['input_dim'])


@torch.no_grad()
def nystrom_graph_features(x, propagation, landmarks=512, block_size=2048, seed=0, eigen_rtol=1e-10):
    if landmarks < 1 or block_size < 1 or not 0 < eigen_rtol < 1:
        raise ValueError('Invalid Nyström settings')
    x, propagation = x.double(), propagation.double()
    propagated = torch.sparse.mm(propagation, x) / np.sqrt(x.shape[1])
    generator = torch.Generator(device=x.device).manual_seed(seed)
    ids = torch.randperm(len(x), generator=generator, device=x.device)[:min(landmarks, len(x))]
    support = propagated[ids].clone()
    gram = tangent_kernel(support, support)
    values, vectors = torch.linalg.eigh((gram + gram.T) / 2)
    keep = values > values.max().clamp_min(1e-30) * eigen_rtol
    if not bool(keep.any()):
        raise ValueError('Nyström support has no positive kernel eigenvalues')
    transform = vectors[:, keep] / values[keep].sqrt()
    mapping = NystromGCN(support, transform, x.shape[1])
    features = x.new_empty(len(x), int(keep.sum()))
    for start in range(0, len(x), block_size):
        block = propagated[start:start + block_size]
        features[start:start + len(block)] = tangent_kernel(block, support) @ transform
    probe_ids = torch.randperm(len(x), generator=generator, device=x.device)[:min(256, len(x))]
    exact = tangent_kernel(propagated[probe_ids], propagated[probe_ids])
    approximation = features[probe_ids] @ features[probe_ids].T
    details = dict(landmarks=len(ids), rank=int(keep.sum()), seed=seed,
                   base_probe_relative_fro=float((exact - approximation).norm() / exact.norm().clamp_min(1e-30)),
                   base_probe_trace_residual=float((exact - approximation).trace() / exact.trace().clamp_min(1e-30)))
    features = torch.sparse.mm(propagation, features)
    return features, mapping, details
