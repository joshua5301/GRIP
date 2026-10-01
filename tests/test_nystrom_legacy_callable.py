"""Preserve verified legacy map identities while extending the optimizer API.

The current serialized code fingerprint includes source line positions. A new
import above NystromMap can reject physically identical verified feature caches.
Keep this targeted regression until a controlled semantic-fingerprint migration.
"""
import json

import pytest
import torch

import src.nystrom_ce as nystrom


def legacy_map_class():
    # Canonical old callable text and line placement, without a Git or results
    # directory dependency. Same filename/module is part of the old fingerprint.
    lines = [""] * 41
    lines[24] = "class NystromMap:"
    lines[25] = '    def __init__(self, anchors, mapping, kernel="relu"):'
    lines[26] = "        self.anchors, self.mapping, self.kernel = anchors, mapping, kernel"
    lines[38] = "    def __call__(self, h):"
    lines[39] = "        return get_kernel_values(h.double(), self.anchors, self.kernel) @ self.mapping"
    namespace = dict(__name__="src.nystrom_ce", get_kernel_values=nystrom.get_kernel_values)
    filename = nystrom.NystromMap.__call__.__code__.co_filename
    exec(compile("\n".join(lines), filename, "exec"), namespace)
    return namespace["NystromMap"]


def maps():
    h = torch.randn(8, 3, generator=torch.Generator().manual_seed(8), dtype=torch.double)
    current = nystrom.NystromMap.fit(h, basis=4, seed=3)
    legacy = legacy_map_class()(current.anchors.clone(), current.mapping.clone(), current.kernel)
    return h, legacy, current


def test_optimizer_api_extension_preserves_legacy_map_callable_identity():
    _, legacy, current = maps()
    assert legacy.__call__.__code__.co_firstlineno == 39
    assert current.__call__.__code__.co_firstlineno == 39
    assert nystrom._map_token(legacy, 3, lambda: False) == nystrom._map_token(current, 3, lambda: False)


def test_verified_legacy_feature_cache_reuses_without_metadata_rewrite(tmp_path):
    h, legacy, current = maps()
    path = tmp_path / "phi.npy"
    old = nystrom.cache_features(h, legacy, path, chunk=3)
    metadata = path.with_suffix(".meta.json")
    before = dict(phi=path.read_bytes(), metadata=metadata.read_bytes())
    reused = nystrom.cache_features(h, current, path, chunk=3)
    assert torch.equal(torch.from_numpy(old.copy()), torch.from_numpy(reused.copy()))
    assert path.read_bytes() == before["phi"]
    assert metadata.read_bytes() == before["metadata"]
    saved = json.loads(metadata.read_text())
    identity = nystrom._cache_identity(h, current, old.shape, 3, lambda: False)
    assert saved == dict(identity, phi_digest=nystrom._content_digest(reused, chunk=3))


def test_real_map_change_remains_rejected_and_preserves_legacy_cache(tmp_path):
    h, legacy, current = maps()
    path = tmp_path / "phi.npy"
    nystrom.cache_features(h, legacy, path, chunk=3)
    metadata = path.with_suffix(".meta.json")
    before = (path.read_bytes(), metadata.read_bytes())
    current.mapping = current.mapping + 0.01
    with pytest.raises(ValueError, match="Feature cache fingerprint differs"):
        nystrom.cache_features(h, current, path, chunk=3)
    assert (path.read_bytes(), metadata.read_bytes()) == before
