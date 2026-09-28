import json

import pandas as pd
import pytest
import torch
import torch.nn.functional as F

import src.condensation_diagnostics as experiment
from src.soft_ce_partition import optimize_ce_assignment
from src.soft_ridge_partition import augmented, decode_moments


def problem():
    generator = torch.Generator().manual_seed(34)
    z = torch.randn(18, 3, generator=generator, dtype=torch.double)
    q = torch.randn(18, 3, generator=generator, dtype=torch.double).softmax(1)
    y = torch.arange(18) % 3
    mask = torch.arange(18) < 12
    return z, q, y, mask


def test_prior_uses_true_training_classes_and_preserves_requested_counts():
    z, _, y, mask = problem()
    initial, ids, counts = experiment.train_prior_initial(z, y, mask, 6, 7, 3)
    assert counts.tolist() == [2, 2, 2]
    assert mask[ids].all() and len(ids.unique()) == 6
    torch.testing.assert_close(initial['centers'], z[ids])
    torch.testing.assert_close(initial['labels'], F.one_hot(y[ids], 3).to(z))
    torch.testing.assert_close(initial['mass'], z.new_full((6,), 1 / 6))
    _, repeated, _ = experiment.train_prior_initial(z, y, mask, 6, 7, 3)
    assert torch.equal(ids, repeated)
    _, all_ids, _ = experiment.train_prior_initial(z, y, mask, 12, 7, 3)
    assert torch.equal(all_ids.sort().values, mask.nonzero().flatten())
    skewed = torch.tensor([0] * 7 + [1] * 4 + [2] + [2] * 6)
    _, _, counts = experiment.train_prior_initial(z, skewed, mask, 7, 7, 3)
    assert counts.tolist() == [4, 2, 1]


def test_custom_direct_targets_and_train_only_outer_survive_resume(tmp_path):
    z, q, y, mask = problem()
    initial, _, _ = experiment.train_prior_initial(z, y, mask, 3, 0, 3)
    args = dict(penalty=.2, lr=.01, inner_tol=1e-9, cg_rtol=1e-9,
                feature_control='direct', initial_representatives=initial,
                outer_indices=mask.nonzero().flatten(), save_assignment=False,
                save_resume=True, checkpoint_steps=[0, 1, 2])
    assignment = torch.arange(len(z)) % 3
    result = optimize_ce_assignment(z, q, assignment, steps=2, folder=tmp_path, **args)
    first = decode_moments(result['initial_moments'], z.shape[1])
    for actual, key in zip(first, ('centers', 'labels', 'mass')):
        torch.testing.assert_close(actual, initial[key])
    for snapshot in result['checkpoints'].values():
        _, labels, mass = decode_moments(snapshot['moments'], z.shape[1])
        torch.testing.assert_close(labels, initial['labels'])
        torch.testing.assert_close(mass, initial['mass'])
        log_probability = (augmented(z[mask]) @ snapshot['theta'].T).log_softmax(1)
        expected = float(-(q[mask] * log_probability).sum(1).mean())
        assert snapshot['teacher_ce'] == pytest.approx(expected, abs=1e-12)
    final = decode_moments(result['checkpoints'][2]['moments'], z.shape[1])[0]
    assert float((first[0] - final).norm()) > 1e-6
    state = torch.load(tmp_path / 'resume.pt', weights_only=False)
    resumed = optimize_ce_assignment(z, q, assignment, steps=3, resume_state=state, **args)
    full = optimize_ce_assignment(z, q, assignment, steps=3, **args)
    torch.testing.assert_close(resumed['checkpoints'][3]['moments'], full['checkpoints'][3]['moments'])
    with pytest.raises(ValueError, match='Resume state'):
        optimize_ce_assignment(z, q, assignment, steps=3, resume_state=state,
                               **{**args, 'outer_indices': None})
    with pytest.raises(ValueError, match='Resume state'):
        optimize_ce_assignment(z, q, assignment, steps=3, resume_state=state,
                               **{**args, 'initial_representatives': {**initial, 'centers': initial['centers'] + .1}})


def test_source_selection_ignores_test_and_target_architecture():
    table = pd.DataFrame([
        dict(candidate=0, step=20, val_acc=81., test_acc=100., gcn_val=100.),
        dict(candidate=1, step=20, val_acc=82., test_acc=20., gcn_val=20.),
        dict(candidate=1, step=10, val_acc=82., test_acc=0., gcn_val=0.),
    ])
    choice = experiment.select_source(table)
    assert choice['candidate'] == 1 and choice['step'] == 10
    assert 'test_acc' not in experiment.split_metrics(
        torch.tensor([[2., 1.], [1., 2.]]).log_softmax(1), torch.tensor([0, 0]),
        torch.tensor([[1., 0.], [1., 0.]]),
        dict(train=torch.tensor([True, False]), val=torch.tensor([False, True])))


def test_trajectory_budget_keeps_initial_and_selected_seed_pairs():
    final, trajectory = list(range(100, 110)), [100, 101, 102]
    for selected in (0, 50, 200):
        assert experiment.checkpoint_seeds(0, selected, final, trajectory) == final
        assert experiment.checkpoint_seeds(selected, selected, final, trajectory) == final
        assert experiment.checkpoint_seeds(25, selected, final, trajectory) == trajectory


def test_arxiv_budget_accepted_before_data_loading(tmp_path, monkeypatch):
    def prepare(dataset, *args):
        assert dataset == 'arxiv'
        raise FileNotFoundError('Arxiv data loading reached')
    monkeypatch.setattr(experiment, '_prepare_dataset', prepare)
    grid = dict(gamma=[.01], T=[.5], penalty=[3e-5], assignment_lr=[.003])
    assert experiment.BUDGET[('arxiv', .005)] == 909
    with pytest.raises(FileNotFoundError, match='Arxiv data loading reached'):
        experiment.run_condensation_diagnostics('A', tmp_path, grid, dataset='arxiv', ratio=.005)
    with pytest.raises(ValueError, match='Unsupported dataset ratio'):
        experiment.run_condensation_diagnostics('A', tmp_path, grid, dataset='arxiv', ratio=.05)


def test_gcn_retains_selected_and_fixed_epoch_metrics(tmp_path, monkeypatch):
    x = torch.randn(6, 3, generator=torch.Generator().manual_seed(3))
    y = torch.arange(6) % 2
    labels = F.one_hot(y, 2).float()
    graph = dict(x=x, y=y, adj=torch.eye(6))
    masks = dict(train=torch.arange(6) < 2, val=(torch.arange(6) >= 2) & (torch.arange(6) < 4),
                 test=torch.arange(6) >= 4)
    observed = iter([dict(val_acc=80., train_ce=.5, val_ce=.6, test_acc=20.),
                     dict(val_acc=70., train_ce=.2, val_ce=.9, test_acc=99.)])
    monkeypatch.setattr(experiment, 'split_metrics', lambda *args: next(observed))
    result = experiment.fit_gcn_diagnostic(x, labels, torch.ones(6), graph, labels, masks,
                                          0, 2, 1, 4, 0., .01, .0005, tmp_path)
    assert result['epoch'] == 1 and result['test_acc'] == 20.
    assert result['last_epoch'] == 2 and result['last_test_acc'] == 99.
    assert result['last_train_ce'] < result['train_ce']
    assert result['last_val_ce'] > result['val_ce']
    assert json.loads((tmp_path / 'seed_0.json').read_text()) == result
    assert len(pd.read_csv(tmp_path / 'seed_0_epochs.csv')) == 2


def test_gcn_reselection_reuses_checkpoints_and_keeps_test_out_of_search(tmp_path, monkeypatch):
    from pathlib import Path
    x = torch.arange(18, dtype=torch.float).reshape(6, 3) / 10
    graph = dict(x=x, y=torch.arange(6) % 2, adj=torch.eye(6).to_sparse_csr())
    masks = [torch.arange(6) // 2 == k for k in range(3)]
    monkeypatch.setattr(experiment, '_prepare_dataset', lambda *args: (
        graph, masks[0], (graph, masks[1]), (graph, masks[2]), x))
    monkeypatch.setitem(experiment.BUDGET, ('cora', .052), 2)
    monkeypatch.setattr(experiment, 'feature_kmeans', lambda *args: torch.arange(6) % 2)
    monkeypatch.setattr(experiment.subprocess, 'check_output', lambda *args, **kwargs: 'test-revision')
    def teachers(*args, **kwargs):
        value = {.01: torch.zeros(6, 2, dtype=torch.double)}
        torch.save(value, args[8] / 'teachers.pt')
        return value
    monkeypatch.setattr(experiment, 'teacher_logits', teachers)
    def optimize(*args, **kwargs):
        folder = kwargs['folder'] / 'checkpoints'
        folder.mkdir()
        for step in kwargs['checkpoint_steps']:
            torch.save(dict(step=step, moments=torch.ones(2, 6, dtype=torch.double)),
                       folder / f'step_{step:06d}.pt')
    monkeypatch.setattr(experiment, 'optimize_ce_assignment', optimize)
    def metrics(value):
        return dict(train_acc=90., val_acc=value, test_acc=75., train_ce=.2, val_ce=.4,
                    test_ce=.5, val_minus_train_ce=.2, train_minus_val_acc=90 - value,
                    full_teacher_ce=.3, condensed_ce=.1)
    monkeypatch.setattr(experiment, 'sgc_metrics', lambda snapshot, *args:
                        metrics(60. if snapshot['step'] == 0 else 90.))
    seen = []
    def evaluate(cx, cy, mass, graph, q, masks, seed, *args, **kwargs):
        folder = args[-1]
        search = 'gcn_search' in folder.parts
        seen.append((search, set(masks), seed))
        step = int(folder.name.split('_')[-1])
        row = metrics(85. if step == 0 else 80.)
        if search:
            row.pop('test_acc')
            row.pop('test_ce')
        return dict(seed=seed, **row)
    monkeypatch.setattr(experiment, 'fit_gcn_diagnostic', evaluate)
    args = dict(method='A', output_dir=tmp_path, space=dict(
        gamma=[.01], T=[1.], penalty=[.01], assignment_lr=[.003]),
        steps=1, checkpoint_steps=[0, 1], device='cpu', full_baseline=False, final_seeds=[100])
    original = experiment.run_condensation_diagnostics(**args)
    assert set(original.selected_step) == {1}
    def forbidden(*args, **kwargs):
        raise AssertionError('Saved condensation must not be optimized again')
    monkeypatch.setattr(experiment, 'optimize_ce_assignment', forbidden)
    selected = experiment.run_condensation_diagnostics(
        **args, selection_architecture='GCN', search_seeds=[0, 1],
        checkpoint_source=original.output_dir.iloc[0])
    assert set(selected.selected_step) == {0}
    assert set(selected.selection_architecture) == {'GCN'}
    assert all('test' not in masks and seed in (0, 1) for search, masks, seed in seen if search)
    assert all(seed == 100 for search, _, seed in seen if not search)
    search = pd.read_csv(Path(selected.output_dir.iloc[0]) / 'search_gcn.csv')
    assert 'test_acc' not in search and search.val_acc.max() == 85.
