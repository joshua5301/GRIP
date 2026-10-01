import pytest
import torch

from src.nystrom_ce import NystromMap, _content_digest, cache_features
from src.shared_features import get_shared_h, get_shared_map


def inputs():
    generator = torch.Generator().manual_seed(73)
    return torch.randn(24, 4, generator=generator) + 0.2


def test_same_source_freezes_first_coordinates_despite_recomputed_rounding(tmp_path):
    h = inputs()
    original = h.clone()
    path = tmp_path / "nested" / "h.pt"
    first = get_shared_h(h, path, "raw-graph-and-preprocessing-digest")
    changed = h + 2e-7
    second = get_shared_h(changed, path, "raw-graph-and-preprocessing-digest")
    assert second.device == changed.device and second.dtype == changed.dtype
    assert torch.equal(first, original) and torch.equal(second, original)
    assert torch.equal(h, original)
    assert not torch.equal(second, changed)
    first[0, 0] = 100  # Returned tensors cannot change the saved coordinates.
    assert torch.equal(get_shared_h(h, path, "raw-graph-and-preprocessing-digest"), original)


@pytest.mark.parametrize("change", ["source", "shape", "dtype"])
def test_shared_h_rejects_changed_provenance_shape_or_dtype_without_overwriting(tmp_path, change):
    h, path = inputs(), tmp_path / "h.pt"
    get_shared_h(h, path, "source1")
    content = path.read_bytes()
    changed = h[:-1] if change == "shape" else h.double() if change == "dtype" else h
    with pytest.raises(ValueError, match="source digest differs|shape or dtype differs"):
        get_shared_h(changed, path, "source2" if change == "source" else "source1")
    assert path.read_bytes() == content


def test_shared_h_rejects_nonfinite_current_or_corrupt_saved_coordinates(tmp_path):
    h, path = inputs(), tmp_path / "h.pt"
    get_shared_h(h, path, "source1")
    content = path.read_bytes()
    bad = h.clone()
    bad[-1, -1] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        get_shared_h(bad, path, "source1")
    assert path.read_bytes() == content
    state = torch.load(path, weights_only=True)
    state["h"][-1, -1] += 0.001
    torch.save(state, path)
    corrupt_content = path.read_bytes()
    with pytest.raises(ValueError, match="content or metadata fingerprint differs"):
        get_shared_h(h, path, "source1")
    assert path.read_bytes() == corrupt_content
    state["h"][-1, -1] = torch.inf
    torch.save(state, path)
    with pytest.raises(ValueError, match="finite"):
        get_shared_h(h, path, "source1")


def test_shared_map_reuses_exact_mapping_without_refitting_or_mutating_rng(tmp_path, monkeypatch):
    h, path = inputs(), tmp_path / "nested" / "map.pt"
    rng = torch.random.get_rng_state().clone()
    first = get_shared_map(h, path, basis=8, seed=5)
    assert torch.equal(rng, torch.random.get_rng_state())
    anchors, mapping = first.anchors.clone(), first.mapping.clone()
    content = path.read_bytes()

    def forbidden_fit(*args, **kwargs):
        raise AssertionError("A validated map must not refit or recompute Cholesky")

    monkeypatch.setattr(NystromMap, "fit", forbidden_fit)
    second = get_shared_map(h.double(), path, basis=8, seed=5)
    assert torch.equal(anchors, second.anchors) and torch.equal(mapping, second.mapping)
    assert second.anchors.device == h.device and second.mapping.device == h.device
    first.mapping[0, 0] += 1
    third = get_shared_map(h, path, basis=8, seed=5)
    assert torch.equal(mapping, third.mapping)
    assert path.read_bytes() == content


def test_frozen_h_and_map_keep_phi_identity_identical_across_condensation_seeds(tmp_path):
    h = inputs()
    h_path, map_path, phi_path = (tmp_path / name for name in ("h.pt", "map.pt", "phi.npy"))
    shared_h = get_shared_h(h, h_path, "raw-source1")
    first_map = get_shared_map(shared_h, map_path, basis=8)
    first_phi = cache_features(shared_h, first_map, phi_path, chunk=7)
    original_digest = _content_digest(first_phi)
    for epsilon in (1e-7, -3e-7):
        reloaded_h = get_shared_h(h + epsilon, h_path, "raw-source1")
        reloaded_map = get_shared_map(reloaded_h, map_path, basis=8)
        phi = cache_features(reloaded_h, reloaded_map, phi_path, chunk=5)
        assert _content_digest(phi) == original_digest


@pytest.mark.parametrize("change", ["h", "basis", "seed", "kernel"])
def test_shared_map_rejects_different_requested_identity_without_overwriting(tmp_path, change):
    h, path = inputs(), tmp_path / "map.pt"
    get_shared_map(h, path, basis=8)
    content = path.read_bytes()
    arguments = dict(basis=8, seed=0, kernel="relu")
    if change == "h":
        h = h + 1e-7
    else:
        arguments[change] = {"basis": 9, "seed": 1, "kernel": "erf"}[change]
    with pytest.raises(ValueError, match="H/basis/seed/kernel fingerprint differs"):
        get_shared_map(h, path, **arguments)
    assert path.read_bytes() == content


@pytest.mark.parametrize("kind", ["h", "map"])
def test_unknown_legacy_caches_are_preserved_and_rejected(tmp_path, kind):
    path, h = tmp_path / "legacy.pt", inputs()
    torch.save({"h": h} if kind == "h" else {"anchors": h[:8], "mapping": torch.eye(8)}, path)
    content = path.read_bytes()
    with pytest.raises(ValueError, match="unknown legacy"):
        if kind == "h":
            get_shared_h(h, path, "source1")
        else:
            get_shared_map(h, path, basis=8)
    assert path.read_bytes() == content


@pytest.mark.parametrize("change", ["anchors", "mapping", "shape", "dtype", "finite"])
def test_shared_map_checks_saved_coordinate_content_and_structure(tmp_path, change):
    path, h = tmp_path / "map.pt", inputs()
    get_shared_map(h, path, basis=8)
    state = torch.load(path, weights_only=True)
    if change in {"anchors", "mapping"}:
        state[change][0, 0] += 0.01
    elif change == "shape":
        state["mapping"] = state["mapping"][:-1]
    elif change == "dtype":
        state["anchors"] = state["anchors"].float()
    else:
        state["mapping"][-1, -1] = torch.nan
    torch.save(state, path)
    content = path.read_bytes()
    with pytest.raises(ValueError, match="fingerprint differs|shape or dtype differs|finite"):
        get_shared_map(h, path, basis=8)
    assert path.read_bytes() == content


@pytest.mark.parametrize("kwargs", [{"basis": 0}, {"basis": 2.0}, {"basis": True},
                                   {"seed": -1}, {"seed": True}, {"kernel": "unknown"}])
def test_invalid_map_settings_never_create_cache(tmp_path, kwargs):
    path = tmp_path / "map.pt"
    with pytest.raises(ValueError):
        get_shared_map(inputs(), path, **kwargs)
    assert not path.exists()
