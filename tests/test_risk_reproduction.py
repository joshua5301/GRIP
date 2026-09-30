import ast

import pytest
import torch

from src.risk_reproduction import hard_representatives, reference_functions


def test_reference_extraction_does_not_execute_legacy_runner():
    namespace = reference_functions(
        "raise RuntimeError('legacy runner')\ndef double(x):\n    return 2*x\n", {"double"}, {}
    )
    assert namespace["double"](3) == 6
    with pytest.raises(ValueError):
        reference_functions("def unrelated(): pass", {"double"}, {})


def test_hard_representatives_preserve_feature_and_label_moments():
    h = torch.tensor([[1.0, 2.0], [3.0, 6.0], [8.0, 4.0]], dtype=torch.double)
    q = torch.tensor([[1.0, 0.0], [0.6, 0.4], [0.0, 1.0]], dtype=torch.double)
    x, y, mass = hard_representatives(h, q, torch.tensor([0, 0, 1]), 2)
    torch.testing.assert_close((mass[:, None] * x.double()).sum(0), h.mean(0))
    torch.testing.assert_close((mass[:, None] * y.double()).sum(0), q.mean(0), atol=1e-7, rtol=1e-7)


def test_original_student_functions_can_be_extracted_without_optuna():
    from src.risk_reproduction import reference_source

    tree = ast.parse(reference_source("risk_experiment"))
    names = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert {"_forward", "_accuracy", "_train_student"} <= names
