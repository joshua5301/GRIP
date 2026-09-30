import pandas as pd

from src.arxiv_width_sweep import promote_widths


def test_promotion_excludes_shared_initial_score_and_retains_distinct_widths():
    screen = pd.DataFrame([
        dict(candidate=0, method="low_rank", width=0, step=100, val=70),
        dict(candidate=1, method="mlp", width=128, step=0, val=99),
        dict(candidate=1, method="mlp", width=128, step=200, val=68),
        dict(candidate=2, method="mlp", width=128, step=100, val=69),
        dict(candidate=3, method="mlp", width=512, step=100, val=68.5),
        dict(candidate=4, method="mlp", width=1024, step=100, val=67),
    ])
    assert promote_widths(screen, 0, 2) == [0, 2, 3]


def test_promotion_ties_prefer_earlier_step_then_candidate():
    screen = pd.DataFrame([
        dict(candidate=3, method="mlp", width=512, step=200, val=68),
        dict(candidate=2, method="mlp", width=512, step=100, val=68),
        dict(candidate=1, method="mlp", width=512, step=100, val=68),
    ])
    assert promote_widths(screen, 0, 1) == [0, 1]
