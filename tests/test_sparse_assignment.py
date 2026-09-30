import pytest
import torch

from src.soft_ce_partition import optimize_ce_assignment
from src.sparse_assignment import SparseMoments, initialize_sparse


def problem():
    generator = torch.Generator().manual_seed(41)
    z = torch.randn(12, 3, generator=generator, dtype=torch.double)
    q = torch.randn(12, 2, generator=generator, dtype=torch.double).softmax(1)
    return z, q, torch.arange(12) % 3


def test_candidates_and_full_support_initialization():
    z, _, assignment = problem()
    indices, logits = initialize_sparse(z, assignment, 3, 3)
    p = torch.zeros(12, 3, dtype=z.dtype).scatter(1, indices, logits.softmax(1))
    expected = torch.full_like(p, 0.05 / 3)
    expected[torch.arange(12), assignment] += 0.95
    torch.testing.assert_close(p, expected)
    assert torch.equal(indices[:, 0], assignment)
    with pytest.raises(ValueError):
        initialize_sparse(z, assignment, 3, 1)


def test_sparse_moments_and_backward_match_dense():
    z, q, assignment = problem()
    indices, logits = initialize_sparse(z, assignment, 3, 2)
    material = torch.cat((torch.ones(12, 1, dtype=z.dtype), z, q), dim=1).requires_grad_()
    actual = SparseMoments.apply(logits, indices, material, 3, 5)
    probability = logits.softmax(1)
    dense = torch.zeros(12, 3, dtype=z.dtype).scatter(1, indices, probability)
    expected = dense.T @ material / 12
    torch.testing.assert_close(actual, expected)
    weight = torch.arange(actual.numel(), dtype=z.dtype).reshape_as(actual) / 10
    ga = torch.autograd.grad((actual * weight).sum(), (logits, material), retain_graph=True)
    ge = torch.autograd.grad((expected * weight).sum(), (logits, material))
    for a, e in zip(ga, ge):
        torch.testing.assert_close(a, e)


def test_sparse_optimizer_resume(tmp_path):
    z, q, assignment = problem()
    options = dict(sparse_k=2, penalty=0.2, inner_method="newton_first",
                   checkpoint_steps=(0, 1), save_resume=True, inner_tol=1e-8)
    optimize_ce_assignment(z, q, assignment, steps=1, folder=tmp_path, **options)
    state = torch.load(tmp_path / "resume.pt", weights_only=False)
    resumed = optimize_ce_assignment(z, q, assignment, steps=2, resume_state=state, **options)
    full = optimize_ce_assignment(z, q, assignment, steps=2, **options)
    torch.testing.assert_close(resumed["checkpoints"][2]["moments"], full["checkpoints"][2]["moments"])
    saved = torch.load(tmp_path / "best_sparse_assignment.pt", weights_only=False)
    assert saved["indices"].shape == saved["logits"].shape == (12, 2)
