"""Prospective CPU pre-load guard tests; no real data/models/source caches."""

import pytest

import src.citation_search as search


def candidate(**changes):
    return dict(method="low_rank", width=0, lr=.05, T=.3, rank=32, penalty=1e-4, **changes)


FORBIDDEN = [
    {"method": "finite_student"},
    {"finite_student_schema": 1},
    {"finite_model_steps": 5},
    {"finite_typo_control": None},
]


def changed_candidate(changes):
    value = candidate()
    value.update(changes)
    return value


@pytest.mark.parametrize("changes", FORBIDDEN)
def test_direct_candidate_guard_rejects_probe_only_controls(changes):
    with pytest.raises(ValueError, match="runtime-probe-only"):
        search._candidate_nystrom_mass(changed_candidate(changes))


@pytest.mark.parametrize("changes", FORBIDDEN)
def test_screen_rejects_before_dataset_or_teacher_write(tmp_path, monkeypatch, changes):
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError("Unsupported finite controls reached data or teacher")

    monkeypatch.setattr(search, "_prepare_dataset", forbidden)
    monkeypatch.setattr(search, "teacher_logits", forbidden)
    output = tmp_path / "uncreated_screen"
    with pytest.raises(ValueError, match="runtime-probe-only"):
        search.run_screen("cora", .026, output, [changed_candidate(changes)],
                          steps=1, device="cpu", student_seeds=())
    assert not calls
    assert not output.exists()


@pytest.mark.parametrize("changes", FORBIDDEN)
def test_selected_rejects_before_root_config_read_or_row_whitelist(tmp_path, monkeypatch, changes):
    root = tmp_path / "uncreated_selected_root"
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError("Unsupported finite controls reached data")

    monkeypatch.setattr(search, "_prepare_dataset", forbidden)
    with pytest.raises(ValueError, match="runtime-probe-only"):
        search.selected_test(root, changed_candidate(changes), device="cpu")
    assert not calls
    assert not root.exists()


@pytest.mark.parametrize("method", ["low_rank", "mlp"])
def test_legacy_candidate_is_unchanged_by_narrow_guard(method):
    old = candidate()
    old["method"] = method
    assert search._candidate_nystrom_mass(old) == old
