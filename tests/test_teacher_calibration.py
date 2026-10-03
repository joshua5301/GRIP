import pytest
import torch

from src.teacher_calibration import calibrate_temperature, select_accuracy


def test_accuracy_has_priority_over_ce():
    result = select_accuracy(
        [
            dict(gamma=0.1, val_acc=80.0, val_ce=0.2),
            dict(gamma=0.3, val_acc=90.0, val_ce=0.8),
            dict(gamma=0.2, val_acc=90.0, val_ce=0.6),
        ]
    )
    assert result["gamma"] == 0.2


def test_calibration_preserves_predictions_and_reduces_ce():
    z = torch.tensor([[8.0, 0.0], [8.0, 0.0], [0.0, 8.0], [0.0, 8.0]], dtype=torch.float64)
    y = torch.tensor([0, 0, 1, 0])
    info, curve = calibrate_temperature(z, y)
    assert info["T"] == pytest.approx(8 / torch.log(torch.tensor(3.0)).item(), rel=1e-5)
    assert info["calibrated_val_ce"] < info["uncalibrated_val_ce"]
    assert torch.equal(z.argmax(1), (z / info["T"]).argmax(1))
    assert info["calibrated_val_ce"] <= min(row["val_ce"] for row in curve) + 1e-10


def test_temperature_boundary():
    info, _ = calibrate_temperature(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), torch.tensor([0, 1]))
    assert info["T"] == 0.05
    assert info["temperature_boundary"] == "lower"
