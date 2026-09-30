import torch


class FeatureTransform:
    def __init__(self, center, matrix, output_center, scale, kind, eps):
        self.center, self.matrix = center, matrix
        self.output_center, self.scale, self.kind, self.eps = output_center, scale, kind, eps

    def __call__(self, z):
        z = z - self.center
        if self.matrix is not None:
            z = z @ self.matrix
        if self.kind == "l2":
            z = z / z.norm(dim=1, keepdim=True).clamp_min(self.eps)
        return (z - self.output_center) / self.scale

    def state_dict(self):
        return {k: v.cpu() if torch.is_tensor(v) else v for k, v in vars(self).items()}


@torch.no_grad()
def fit_transform(features, kind="rms", power=0.5, ridge=0.01, eps=1e-12):
    if kind not in ("rms", "l2", "whiten") or not 0 <= power <= 1 or ridge <= 0 or eps <= 0:
        raise ValueError("Invalid NTK transform settings")
    center = features.mean(0)
    z = features - center
    matrix = None
    if kind == "whiten":
        covariance = z.T @ z / len(z)
        values, vectors = torch.linalg.eigh(covariance)
        stabilizer = ridge * values.max().clamp_min(eps)
        factors = (values.clamp_min(0) + stabilizer).pow(-power / 2)
        matrix = (vectors * factors) @ vectors.T
        z = z @ matrix
    if kind == "l2":
        z = z / z.norm(dim=1, keepdim=True).clamp_min(eps)
    output_center = z.mean(0)
    scale = (z - output_center).square().sum(1).mean().sqrt().clamp_min(eps)
    transform = FeatureTransform(center, matrix, output_center, scale, kind, eps)
    return transform(features), transform
