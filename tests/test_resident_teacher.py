import pytest
import torch
import torch.nn.functional as F

from src.nystrom_ce import fit_streaming_teacher


def problem():
    generator = torch.Generator().manual_seed(881)
    phi = torch.randn(36, 5, generator=generator, dtype=torch.double).numpy()
    mask = torch.arange(36) % 3 == 0
    labels = torch.zeros(36, dtype=torch.long)
    labels[mask] = torch.arange(int(mask.sum())) % 3
    return phi, labels, mask


def test_resident_and_streamed_fits_have_same_objective_gradient_and_logits(monkeypatch):
    phi, labels, mask = problem()
    original = torch.optim.LBFGS
    initial = []

    class CapturingLBFGS(original):
        def step(self, closure):
            value = closure()
            initial.append((float(value), self.param_groups[0]["params"][0].grad.clone()))
            return super().step(closure)

    monkeypatch.setattr(torch.optim, "LBFGS", CapturingLBFGS)
    streamed_logits, streamed_weight = fit_streaming_teacher(phi, labels, mask, gamma=0.2, chunk=7)
    resident_logits, resident_weight = fit_streaming_teacher(
        phi, labels, mask, gamma=0.2, chunk=7, resident_training=True
    )
    assert fit_streaming_teacher.last_route["actual"] == "resident"
    torch.testing.assert_close(initial[0][1], initial[1][1], atol=1e-14, rtol=1e-12)
    assert initial[0][0] == pytest.approx(initial[1][0], abs=1e-14)
    torch.testing.assert_close(resident_weight, streamed_weight, atol=2e-7, rtol=1e-6)
    torch.testing.assert_close(resident_logits, streamed_logits, atol=1e-6, rtol=1e-6)
    x = torch.from_numpy(phi[mask.numpy()])

    def objective_and_gradient(weight):
        value = weight.clone().requires_grad_()
        loss = F.cross_entropy(x @ value, labels[mask]) + 0.2 / len(x) * value.square().sum()
        (gradient,) = torch.autograd.grad(loss, value)
        return loss.detach(), gradient

    resident_objective, resident_gradient = objective_and_gradient(resident_weight)
    streamed_objective, streamed_gradient = objective_and_gradient(streamed_weight)
    torch.testing.assert_close(resident_objective, streamed_objective, atol=1e-12, rtol=1e-10)
    torch.testing.assert_close(resident_gradient, streamed_gradient, atol=1e-7, rtol=1e-5)
    assert float(resident_gradient.abs().max()) <= 1e-5


def test_resident_allocation_contains_only_selected_rows_and_transfers_them_once(monkeypatch):
    phi, labels, mask = problem()
    original_empty = torch.empty
    allocated = []

    def capture(*shape, **kwargs):
        tensor = original_empty(*shape, **kwargs)
        if tuple(tensor.shape) == (int(mask.sum()), phi.shape[1]):
            allocated.append(tensor)
        return tensor

    monkeypatch.setattr(torch, "empty", capture)
    logits, _ = fit_streaming_teacher(phi, labels, mask, gamma=0.2, chunk=7, resident_training=True)
    assert logits.shape == (len(phi), 3)
    assert len(allocated) == 1
    torch.testing.assert_close(allocated[0], torch.from_numpy(phi[mask.numpy()]), atol=0, rtol=0)
    assert allocated[0].numel() < phi.size
    assert fit_streaming_teacher.last_route["resident_bytes"] == int(mask.sum()) * phi.shape[1] * 8


def test_resident_transfer_stop_guard_prevents_optimization(monkeypatch):
    phi, labels, mask = problem()
    checks = 0

    def stop():
        nonlocal checks
        checks += 1
        return checks >= 4

    monkeypatch.setattr(
        torch.optim, "LBFGS", lambda *args, **kwargs: pytest.fail("Optimizer started after stop")
    )
    with pytest.raises(InterruptedError, match="transfer interrupted"):
        fit_streaming_teacher(phi, labels, mask, resident_training=True, chunk=7, stop=stop)


@pytest.mark.parametrize("resident", [False, True])
def test_stop_guard_prevents_all_training_and_empty_mask_is_rejected(resident):
    phi, labels, mask = problem()
    with pytest.raises(InterruptedError):
        fit_streaming_teacher(phi, labels, mask, resident_training=resident, stop=lambda: True)
    with pytest.raises(ValueError, match="at least one"):
        fit_streaming_teacher(phi, labels, torch.zeros_like(mask), resident_training=resident)
