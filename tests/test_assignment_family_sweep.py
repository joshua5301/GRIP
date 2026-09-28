import pandas as pd
import pytest
import torch

from src.assignment_family_sweep import family_options, family_variants, parameter_count, validation_choice
from src.low_rank_assignment import (LowRankMoments, assignment_inputs, encode_nodes,
                                     initialize_encoder, initialize_factors, initialize_mlp)
from src.soft_ridge_partition import AssignmentMoments, initial_logits, make_material


def test_variants_cover_requested_ranks_and_widths_without_duplicate_dense_runs():
    variants = family_variants(['dense', 'low_rank', 'mlp1', 'mlp2'], [8, 16], [32, 64], 90)
    assert len(variants) == 9
    assert sum(v['method'] == 'dense' for v in variants) == 1
    assert {(v['rank'], v['width']) for v in variants if v['method'] == 'mlp2'} == {
        (8, 32), (8, 64), (16, 32), (16, 64),
    }
    assert all(v['width'] is None for v in variants if v['method'] == 'mlp1')
    with pytest.raises(ValueError):
        family_variants(['low_rank'], [128], [64], 90)


def test_all_parameterizations_preserve_initial_moments_and_parameter_counts():
    generator = torch.Generator().manual_seed(27)
    z = torch.randn(13, 3, generator=generator, dtype=torch.double)
    q = torch.randn(13, 2, generator=generator, dtype=torch.double).softmax(1)
    assignment = torch.arange(13) % 4
    material = make_material(z, q)
    inputs = assignment_inputs(z, q, 'features')
    expected = AssignmentMoments.apply(initial_logits(assignment, 4, .05), material, 5)
    for variant in family_variants(['dense', 'low_rank', 'mlp1', 'mlp2'], [2, 3], [5], 4):
        options = family_options(variant)
        if variant['method'] == 'dense':
            parameters = [initial_logits(assignment, 4, .05)]
            actual = AssignmentMoments.apply(parameters[0], material, 5)
            assert options['assignment_rank'] is None
        else:
            if variant['method'] == 'low_rank':
                u, v = initialize_factors(assignment, 4, variant['rank'], 0)
                parameters = [u, v]
            elif variant['method'] == 'mlp1':
                weight, v = initialize_encoder(inputs, 4, variant['rank'], 0)
                u, parameters = encode_nodes(inputs, [weight]), [weight, v]
            else:
                encoder, v = initialize_mlp(inputs, 4, variant['rank'], variant['width'], 0)
                u, parameters = encode_nodes(inputs, encoder), [*encoder, v]
            actual = LowRankMoments.apply(u, v, assignment, material, .05, 5)
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-10)
        assert sum(p.numel() for p in parameters) == parameter_count(variant, len(z), 3, 4)


def test_selection_ignores_test_and_prefers_initial_state_on_ties():
    table = pd.DataFrame([
        dict(candidate=0, step=100, val=70., test=90.),
        dict(candidate=1, step=0, val=71., test=60.),
        dict(candidate=2, step=100, val=71., test=95.),
    ])
    selected = validation_choice(table)
    assert selected['candidate'] == 1
    assert selected['step'] == 0
