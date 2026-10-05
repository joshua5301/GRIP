import math

import torch

from src.io import array_digest
from src.low_rank_assignment import FactorizedBaseMoments


@torch.no_grad()
def distance_svd(inputs, centers, rank):
    left = torch.cat((2 * inputs, torch.ones_like(inputs[:, :1])), 1)
    right = torch.cat((centers, -centers.square().sum(1, keepdim=True)), 1)
    right = right - right.mean(0)
    ql, rl = torch.linalg.qr(left, mode="reduced")
    qr, rr = torch.linalg.qr(right, mode="reduced")
    a, s, bt = torch.linalg.svd(rl @ rr.T, full_matrices=False)
    if not 1 <= rank <= len(s):
        raise ValueError("SVD rank exceeds the factorized distance dimensions")
    scale = s[:rank].sqrt() * rank**0.25
    return ((ql @ a[:, :rank]) * scale, (qr @ bt[:rank].T) * scale,
            float(s[rank:].norm() / s.norm().clamp_min(1e-30)))


class AttentionAssignment(torch.nn.Module):
    def __init__(self, inputs, centers, rank, tau, method, seed=0):
        super().__init__()
        if method not in ("metric", "attention", "low_rank", "svd_UV", "random_UV") or rank < 1 or not math.isfinite(tau) or tau <= 0:
            raise ValueError("Invalid attention configuration")
        self.method, self.rank, self.tau, self.seed = method, rank, tau, seed
        self.register_buffer("inputs", inputs.detach())
        self.register_buffer("centers", centers.detach())
        generator = torch.Generator(device=inputs.device).manual_seed(seed)
        weight = torch.randn(inputs.shape[1], rank, device=inputs.device,
                             dtype=inputs.dtype, generator=generator) / math.sqrt(inputs.shape[1])
        self.svd_relative_error = 0.0
        if method in ("svd_UV", "random_UV"):
            u, v, self.svd_relative_error = distance_svd(inputs, centers, rank)
            if method == "random_UV":
                factors = []
                for index, reference in enumerate((u, v)):
                    random = torch.randn(reference.shape, device=inputs.device,
                                         dtype=inputs.dtype, generator=generator)
                    if index == 1:
                        random -= random.mean(0)
                    random *= reference.norm(dim=0) / random.norm(dim=0).clamp_min(1e-30)
                    factors.append(random)
                u, v = factors
            self.u, self.v = torch.nn.Parameter(u), torch.nn.Parameter(v)
        elif method == "low_rank":
            self.u = torch.nn.Parameter(inputs.new_zeros(len(inputs), rank))
            self.v = torch.nn.Parameter(torch.randn(len(centers), rank, device=inputs.device,
                                                    dtype=inputs.dtype, generator=generator))
        else:
            self.query = torch.nn.Parameter(weight * (0.1 if method == "metric" else 1.0))
        if method == "attention":
            self.key = torch.nn.Parameter(weight.clone())
        left = torch.cat((2 * inputs, torch.ones_like(inputs[:, :1])), 1)
        right = torch.cat((centers, -centers.square().sum(1, keepdim=True)), 1)
        fixed = method in ("metric", "low_rank")
        self.register_buffer("left", left / tau if fixed else inputs.new_zeros(len(inputs), 1))
        self.register_buffer("right", right if fixed else inputs.new_zeros(len(centers), 1))

    def identity(self):
        return dict(method=self.method, rank=self.rank, tau=self.tau, seed=self.seed,
                    digest=array_digest(*[v.detach().cpu().numpy() for v in self.state_dict().values()]))

    def factors(self):
        if self.method in ("low_rank", "svd_UV", "random_UV"):
            return self.u / self.tau, self.v
        a = self.inputs @ self.query
        b = self.centers @ (self.query if self.method == "metric" else self.key)
        if self.method == "attention":
            return a / self.tau, b
        u = torch.cat((2 * a, torch.ones_like(a[:, :1])), 1)
        v = torch.cat((b, -b.square().sum(1, keepdim=True)), 1)
        return u * math.sqrt(u.shape[1]) / self.tau, v

    def forward(self, material, chunk_size):
        u, v = self.factors()
        return FactorizedBaseMoments.apply(u, v, self.left, self.right, material, chunk_size)
