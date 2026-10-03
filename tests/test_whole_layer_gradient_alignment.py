"""Static promotion controls; accepted twelve-case grouping math is not rerun."""
import ast
from pathlib import Path

REPO = next(p for p in Path(__file__).resolve().parents if (p/'src/finite_student_outer_v2.py').is_file())
BASE = Path(__file__).resolve().parents[1]
HELPER = BASE/'src/whole_layer_gradient_alignment.py'
if not HELPER.is_file():
    HELPER = REPO/'src/whole_layer_gradient_alignment.py'
ORIGINAL = REPO/'results/implementation_drafts/cora_whole_layer_grouping_v1/whole_layer_grouping.py'


def test_grouping_helper_byte_exact_accepted_proof():
    assert HELPER.read_bytes() == ORIGINAL.read_bytes()


def test_moment_leaf_boundary_and_no_optimizer_or_source_work():
    tree = ast.parse(HELPER.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'moment_partials')
    original = next(n for n in ast.parse(ORIGINAL.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'moment_partials')
    assert ast.dump(function, include_attributes=False) == ast.dump(original, include_attributes=False)
    text = ast.unparse(tree)
    assert 'source_gradient_targets' not in text and 'torch.optim' not in text and 'torch.load' not in text
