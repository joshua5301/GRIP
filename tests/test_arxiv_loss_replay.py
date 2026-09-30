import pandas as pd
import pytest

from src.arxiv_loss_replay import selected_uniform_results


def test_reference_uses_fixed_candidate_checkpoint_and_final_phase():
    selected = pd.DataFrame([dict(method="mlp", candidate=10, step=1000)])
    frame = pd.DataFrame([
        dict(method="mlp", candidate=c, step=step, phase=phase, condensation_seed=0, seed=100)
        for c, step, phase in [(10, 1000, "selected"), (10, 0, "initial"), (8, 1000, "selected")]
    ])
    result = selected_uniform_results(frame, selected, [0], [100])
    assert len(result) == 1
    assert result.iloc[0].candidate == 10
    assert result.iloc[0].step == 1000


@pytest.mark.parametrize("seeds", [[100], [100, 100]])
def test_incomplete_or_duplicate_reference_pairs_are_rejected(seeds):
    selected = pd.DataFrame([dict(method="mlp", candidate=10, step=1000)])
    frame = pd.DataFrame([
        dict(method="mlp", candidate=10, step=1000, phase="selected", condensation_seed=0, seed=s)
        for s in seeds
    ])
    with pytest.raises(ValueError, match="exactly the selected seed pairs"):
        selected_uniform_results(frame, selected, [0], [100, 101])
