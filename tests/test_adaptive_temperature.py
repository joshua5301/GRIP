import torch

from src.adaptive_temperature import smoothing_position, temperature_labels


def test_train_temperature_fixed_and_zero_eta_recovers_global():
    train = torch.tensor([True, False, False, False])
    position, scale = smoothing_position(torch.tensor([1e-7, 0., 2., 4.]), train)
    assert scale == 2. and position[0] == 0 and position[1] == 0
    logits = torch.tensor([[3., 0.]]).repeat(4, 1)
    global_q, _ = temperature_labels(logits, position, .2, 0.)
    torch.testing.assert_close(global_q, (logits / .2).softmax(1))
    q, t = temperature_labels(logits, position, .2, 4.)
    assert t[0] == .2 and bool((t >= .2).all())
    torch.testing.assert_close(q[0], global_q[0])
    assert q[3, 0] < q[2, 0] < q[0, 0]
