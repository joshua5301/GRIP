import pytest

from src.arxiv_dual_evaluation import select_route_epochs


def test_validation_selects_each_route_without_using_test_scores():
    history = [
        dict(epoch=10, gcn_val=70, mlp_val=68, gcn_test=60, mlp_test=99),
        dict(epoch=20, gcn_val=69, mlp_val=72, gcn_test=99, mlp_test=60),
    ]
    selected = select_route_epochs(history)
    assert selected["gcn"]["epoch"] == 10
    assert selected["gcn"]["mlp_val"] == 68
    assert selected["mlp"]["epoch"] == 20


def test_validation_ties_preserve_the_first_evaluated_epoch():
    history = [dict(epoch=e, gcn_val=70, mlp_val=70) for e in (10, 20)]
    selected = select_route_epochs(history)
    assert selected["gcn"]["epoch"] == selected["mlp"]["epoch"] == 10


def test_empty_history_is_rejected():
    with pytest.raises(ValueError, match="No evaluated"):
        select_route_epochs([])
