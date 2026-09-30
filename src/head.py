import numpy as np
import torch


def augment(z):
    return torch.cat((z, z.new_ones(len(z), 1)), dim=1)


def head_objective(x, q, mass, theta, penalty):
    return -(mass[:, None] * q * (x @ theta.T).log_softmax(1)).sum() + penalty * theta.square().sum() / 2


def fit_head(z, q, mass, penalty, max_iter=2000, grad_tol=1e-7, initial_theta=None, tolerance_change=1e-15):
    if penalty <= 0:
        raise ValueError("Positive regularization is required, including the bias")
    x, q, mass = augment(z.double()), q.double(), mass.double()
    theta = (
        x.new_zeros(q.shape[1], x.shape[1]) if initial_theta is None else initial_theta.to(x).detach().clone()
    ).requires_grad_()
    if theta.shape != (q.shape[1], x.shape[1]):
        raise ValueError("Initial head shape does not match features and classes")
    optimizer = torch.optim.LBFGS(
        [theta],
        max_iter=max_iter,
        tolerance_grad=grad_tol,
        tolerance_change=tolerance_change,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss = head_objective(x, q, mass, theta, penalty)
        loss.backward()
        return loss

    optimizer.step(closure)
    loss = head_objective(x, q, mass, theta, penalty)
    (gradient,) = torch.autograd.grad(loss, theta)
    norm = float(gradient.norm())
    if not np.isfinite(float(loss.detach()) + norm):
        raise FloatingPointError("Nonfinite head fit")
    return dict(
        theta=theta.detach(),
        grad_norm=norm,
        grad_max=float(gradient.abs().max()),
        converged=float(gradient.abs().max()) <= grad_tol,
        iterations=optimizer.state[theta].get("n_iter", 0),
    )
