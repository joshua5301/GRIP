import math

import pandas as pd
import pytest
import torch
import torch.nn.functional as F

from src.graph_moment_condensation import _references, normalized_graph, readout_embedding
from src.graph_moment_diagnostics import SHARED_KEYS, _load_pair, measure_probe_pair, summarize_risks
from src.models import GCN
from src.risk_experiment import _fingerprint


def test_probe_risk_and_moments_are_zero_for_identical_permuted_graph():
    torch.manual_seed(9)
    model = GCN(3, 5, 2, 2, .5).eval()
    x = torch.randn(4, 3)
    adjacency = normalized_graph(torch.randn(4, 4))
    logits = torch.randn(4, 2)
    reference = _references(model, dict(x=x, adj=adjacency), logits, [.5])['0.5']
    labels = (logits / .5).softmax(1)
    ids = torch.tensor([2, 0, 3, 1])
    states = dict(initial=dict(x=x, adjacency=adjacency, y=labels, counts=torch.ones(4)),
                  moments=dict(x=x[ids], adjacency=adjacency[ids][:, ids], y=labels[ids], counts=torch.ones(4)))
    for row in measure_probe_pair(model, reference, states, .1):
        assert row['abs_gap'] < 1e-6
        assert row['first'] < 1e-8 and row['second'] < 1e-8


def test_probe_ce_uses_cell_mass_and_preserves_signed_gap():
    model = GCN(1, 1, 2, 2, 0.).eval()
    with torch.no_grad():
        model.layers[0].lin.weight.fill_(1.)
        model.layers[0].bias.zero_()
        model.layers[1].lin.weight.copy_(torch.tensor([[1.], [-1.]]))
        model.layers[1].bias.zero_()
    state = dict(x=torch.tensor([[0.], [2.]]), adjacency=torch.eye(2), y=torch.eye(2), counts=torch.tensor([3, 1]))
    reference = dict(center=torch.zeros(1), scale=torch.tensor(1.), first=torch.zeros(1, 2),
                     second=torch.zeros(1, 1), ce=2.)
    row = measure_probe_pair(model, reference, {'initial': state}, 1.)[0]
    expected = .75 * math.log(2) + .25 * float(F.softplus(torch.tensor(4.)))
    assert row['synthetic_ce'] == pytest.approx(expected)
    assert row['signed_gap'] == pytest.approx(expected - 2.)
    assert row['abs_gap'] == pytest.approx(abs(expected - 2.))


def test_pair_uses_optimized_labels_and_original_graph_at_same_temperature(tmp_path):
    config = {key: key for key in SHARED_KEYS}
    source = tmp_path / 'source'
    source.mkdir()
    cache = tmp_path / 'cache' / _fingerprint(config)
    cache.mkdir(parents=True)
    original = dict(x=torch.tensor([[1.], [2.]]), edge_logits=torch.zeros(2, 2), counts=torch.tensor([3, 1]))
    torch.save(original, cache / 'initial.pt')
    params = dict(T=.5, second_weight=.1, feature_lr=.01, edge_lr=.01)
    optimized = dict(x=torch.tensor([[4.], [5.]]), adjacency=torch.eye(2),
                     counts=original['counts'], y=torch.tensor([[.9, .1], [.2, .8]]))
    torch.save(optimized, source / f'condensed_{_fingerprint(dict(variant="moments", **params))}.pt')
    pair, found_cache = _load_pair(source, config, params)
    assert found_cache == cache
    torch.testing.assert_close(pair['initial']['x'], original['x'])
    torch.testing.assert_close(pair['initial']['y'], optimized['y'])
    torch.testing.assert_close(pair['initial']['counts'], optimized['counts'])
    torch.testing.assert_close(pair['initial']['adjacency'], normalized_graph(original['edge_logits']))
    optimized['counts'] = torch.tensor([2, 2])
    torch.save(optimized, source / f'condensed_{_fingerprint(dict(variant="moments", **params))}.pt')
    with pytest.raises(ValueError, match='masses differ'):
        _load_pair(source, config, params)


def test_risk_summary_keeps_epochs_separate_and_pairs_by_seed():
    rows = []
    for seed in (6000, 6001):
        for epoch in (0, 50):
            for variant, gap in [('initial', .2), ('moments', .1)]:
                rows.append(dict(cohort='heldout', seed=seed, epoch=epoch, variant=variant,
                                 abs_gap=gap, signed_gap=-gap, first=1., second=2.))
    summary, pairs = summarize_risks(pd.DataFrame(rows))
    assert len(summary) == 4 and len(pairs) == 4
    assert summary.seeds.eq(2).all()
    assert pairs.gap_change.to_numpy() == pytest.approx([-.1] * 4)
