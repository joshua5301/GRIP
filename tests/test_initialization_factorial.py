import pytest

from src.initialization_factorial import CASES, solver_options


def test_factorial_changes_only_assignment_family_and_retains_legacy_settings():
    saved = dict(penalty=3e-5, lr=.01, mixing=.05, chunk_size=4096,
                 steps=300, inner_tol=1e-7, cg_rtol=1e-6, save_assignment=True)
    options = {case: solver_options(saved, case, [25, 100, 200], 7) for case in CASES}
    assert options['risk_dense'] == options['kmeans_dense']
    assert options['risk_mlp'] == options['kmeans_mlp']
    for settings in options.values():
        for key in ('penalty', 'lr', 'mixing', 'chunk_size', 'inner_tol', 'cg_rtol'):
            assert settings[key] == saved[key]
        assert settings['checkpoint_steps'] == [0, 25, 100, 200, 300]
        assert settings['solver_mode'] == 'exact'
        assert settings['mass_mode'] == 'free'
        assert settings['feature_control'] == 'joint'
        assert settings['steps'] == 300
    changed = {key for key in options['risk_dense']
               if options['risk_dense'][key] != options['risk_mlp'][key]}
    assert changed == {'assignment_rank', 'assignment_input', 'assignment_encoder'}
    assert options['risk_dense']['assignment_rank'] is None
    assert options['risk_mlp']['assignment_rank'] == 16
    with pytest.raises(ValueError):
        solver_options(saved, 'unknown', [0, 300], 0)
