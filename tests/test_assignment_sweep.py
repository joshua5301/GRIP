import pandas as pd
import pytest
import torch

from src.assignment_sweep import grid_rows, promote, representative
from src.ntk_transforms import fit_transform
from src.soft_ce_partition import optimize_ce_assignment


def test_promotion_uses_validation_not_test_and_distinct_candidates():
    table = pd.DataFrame([
        dict(candidate=0, step=100, dropout=.1, val=70., test=99.),
        dict(candidate=1, step=300, dropout=.1, val=72., test=60.),
        dict(candidate=1, step=100, dropout=.5, val=71., test=50.),
        dict(candidate=2, step=100, dropout=.1, val=71.5, test=40.),
    ])
    assert promote(table, 2) == [1, 2]
    assert grid_rows({'T': [1., 2.], 'penalty': [.1]}) == [
        {'T': 1., 'penalty': .1}, {'T': 2., 'penalty': .1}]


def test_representative_inverse_transform_preserves_raw_means():
    h = torch.arange(24, dtype=torch.double).reshape(8, 3)
    z, transform = fit_transform(h)
    q = torch.tensor([[.2, .8]], dtype=torch.double).repeat(8, 1)
    material = torch.cat((torch.ones(8, 1), z, q), dim=1)
    moments = material.reshape(2, 4, 6).sum(1) / 8
    x, y, mass = representative(moments, transform, 3, 'cpu')
    torch.testing.assert_close(x, h.reshape(2, 4, 3).mean(1).float())
    torch.testing.assert_close(y, q[:2].float())
    torch.testing.assert_close(mass, torch.full((2,), .5, dtype=torch.double))


def test_resume_preserves_optimizer_trajectory_and_checks_data(tmp_path):
    generator = torch.Generator().manual_seed(13)
    z = torch.randn(12, 3, generator=generator, dtype=torch.double)
    q = torch.randn(12, 2, generator=generator, dtype=torch.double).softmax(1)
    assignment = torch.arange(12) % 4
    options = dict(penalty=.2, lr=.01, chunk_size=5, inner_tol=1e-9, cg_rtol=1e-9,
                   assignment_rank=2, assignment_input='features', assignment_encoder='mlp',
                   encoder_hidden=8, save_assignment=False, save_resume=True)
    full = optimize_ce_assignment(z, q, assignment, steps=4, checkpoint_steps=[2], **options)
    optimize_ce_assignment(z, q, assignment, steps=2, folder=tmp_path, checkpoint_steps=[2], **options)
    state = torch.load(tmp_path / 'resume.pt', weights_only=False)
    assert state['step'] == 2
    assert state['optimizer']['state']
    resumed = optimize_ce_assignment(z, q, assignment, steps=4, folder=tmp_path,
                                     checkpoint_steps=[2, 4], resume_state=state, **options)
    torch.testing.assert_close(full['checkpoints'][4]['moments'], resumed['checkpoints'][4]['moments'],
                               atol=2e-6, rtol=2e-5)
    assert [row['step'] for row in resumed['history']] == list(range(5))
    with pytest.raises(ValueError, match='Resume state'):
        optimize_ce_assignment(z + .1, q, assignment, steps=4, resume_state=state, **options)
