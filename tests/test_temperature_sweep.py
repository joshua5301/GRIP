import pytest
import torch
import pandas as pd

from src.temperature_sweep import temperature_grid, select_temperature
from src.soft_ce_partition import optimize_ce_assignment, outer_value_gradient


def test_grid_and_validation_selection():
    grid = temperature_grid([.1, .3, .1], [.3, 1.])
    assert len(grid) == 4
    assert grid[0] == dict(inner_T=.1, outer_T=.3)
    table = pd.DataFrame([dict(candidate=0, step=10, val=70., outer_ce=9.),
                          dict(candidate=1, step=20, val=69., outer_ce=.01),
                          dict(candidate=2, step=5, val=70., outer_ce=10.)])
    assert select_temperature(table)['candidate'] == 2
    with pytest.raises(ValueError):
        temperature_grid([0], [.3])


def test_separate_outer_labels_and_resume(tmp_path):
    g = torch.Generator().manual_seed(18)
    z = torch.randn(12, 3, generator=g, dtype=torch.double)
    logits = torch.randn(12, 3, generator=g, dtype=torch.double)
    inner, outer = (logits / .1).softmax(1), (logits / .7).softmax(1)
    assignment = torch.arange(12) % 3
    options = dict(steps=0, penalty=.2, assignment_rank=2, checkpoint_steps=(0,),
                   save_resume=True, save_assignment=False, inner_method='newton_first')
    shared = optimize_ce_assignment(z, inner, assignment, **options)
    explicit = optimize_ce_assignment(z, inner, assignment, outer_targets=inner, **options)
    assert shared['history'][0]['J'] == explicit['history'][0]['J']
    separate = optimize_ce_assignment(z, inner, assignment, outer_targets=outer, folder=tmp_path, **options)
    torch.testing.assert_close(shared['initial_moments'], separate['initial_moments'], atol=0, rtol=0)
    snapshot = separate['checkpoints'][0]
    expected, _ = outer_value_gradient(z, outer, snapshot['theta'])
    assert abs(snapshot['teacher_ce'] - expected) < 1e-12
    state = torch.load(tmp_path / 'resume.pt', weights_only=False)
    with pytest.raises(ValueError, match='Resume state'):
        optimize_ce_assignment(z, inner, assignment, outer_targets=inner, resume_state=state, **options)
