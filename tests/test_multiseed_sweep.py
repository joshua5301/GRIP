import pandas as pd
import pytest
import torch

from src.multiseed_sweep import aggregate_search, calibrate_teacher_temperature


def records():
    return [dict(candidate=candidate, T=.3, rank=8, penalty=3e-5, step=100,
                 condensation_seed=a, seed=b, val_acc=70 + candidate + a + b)
            for candidate in (0, 1) for a in (0, 1, 2) for b in (0, 1, 2)]


def test_joint_seed_average():
    table = aggregate_search(records(), [0, 1, 2], [0, 1, 2])
    assert table.val.tolist() == [72., 73.]
    assert table.evaluations.tolist() == [9, 9]
    assert table.condensation_val_std.tolist() == [1., 1.]


def test_missing_or_duplicate_pair_is_rejected():
    data = records()
    with pytest.raises(ValueError):
        aggregate_search(data[:-1], [0, 1, 2], [0, 1, 2])
    with pytest.raises(ValueError):
        aggregate_search(data + [data[0]], [0, 1, 2], [0, 1, 2])


def test_selection_uses_common_step():
    data = records()
    alternative = pd.DataFrame(records())
    alternative['step'] = 200
    alternative['val_acc'] += alternative.condensation_seed.map({0: 5., 1: -4., 2: -4.})
    table = aggregate_search(data + alternative.to_dict('records'), [0, 1, 2], [0, 1, 2])
    selected = table.sort_values(['val', 'step', 'candidate'], ascending=[False, True, True]).iloc[0]
    assert selected['candidate'] == 1
    assert selected['step'] == 100


def test_temperature_calibration_preserves_accuracy():
    logits = torch.tensor([[10., 0.]] * 8 + [[0., 10.]] * 2, dtype=torch.double)
    labels = torch.zeros(10, dtype=torch.long)
    result = calibrate_teacher_temperature(logits, labels)
    assert result['validation_ce_after'] < result['validation_ce_before']
    assert result['T_cal'] == pytest.approx(10 / 1.38629436112, rel=1e-4)
    assert result['validation_accuracy'] == 80.
    assert result['applied_to_sweep'] is False
    assert torch.equal(logits.argmax(1), (logits / result['T_cal']).argmax(1))
