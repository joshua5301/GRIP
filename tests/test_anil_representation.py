import torch
import torch.nn.functional as F

from src.anil_representation import adapt_head, representation, sample_support
from src.models import GCN
from src.risk_experiment import _forward


def test_unrolled_encoder_gradient_matches_finite_difference():
    torch.manual_seed(9)
    x = torch.randn(8, 3, dtype=torch.double)
    y = torch.tensor([0, 1, 0, 1, 1, 0, 1, 0])
    weight = torch.randn(2, 3, dtype=torch.double)
    bias = torch.zeros(2, dtype=torch.double)

    def objective(theta):
        z = x * theta
        w, b = adapt_head(z[:4], y[:4], weight.clone().requires_grad_(),
                          bias.clone().requires_grad_(), 3, .2, .01)
        return F.cross_entropy(F.linear(z[4:], w, b), y[4:])

    theta = torch.tensor(.7, dtype=torch.double, requires_grad=True)
    gradient, = torch.autograd.grad(objective(theta), theta)
    numerical = (objective(theta.detach() + 1e-5) - objective(theta.detach() - 1e-5)) / 2e-5
    torch.testing.assert_close(gradient, numerical, rtol=1e-5, atol=1e-7)


def test_representation_includes_last_propagation():
    torch.manual_seed(3)
    x = torch.randn(5, 3)
    adjacency = (torch.eye(5) * .5 + torch.ones(5, 5) * .1).to_sparse()
    model = GCN(3, 7, 2, 2, .5).eval()
    z = representation(model, x, adjacency)
    logits = F.linear(z, model.layers[-1].lin.weight, model.layers[-1].bias)
    torch.testing.assert_close(logits.log_softmax(1), _forward(model, x, adjacency))


def test_support_is_balanced_reproducible_and_leaves_queries():
    ids = torch.arange(21)
    labels = ids // 7
    selected = sample_support(ids, labels, 8, 9, leave_query=True)
    torch.testing.assert_close(selected, sample_support(ids, labels, 8, 9, leave_query=True))
    counts = torch.bincount(labels[selected], minlength=3)
    assert len(selected.unique()) == 8
    assert counts.max() - counts.min() <= 1
    assert bool((counts < 7).all())
