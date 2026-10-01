"""Cached NTK endpoint checks align moments with the current map's device."""
import os
from types import SimpleNamespace

import pytest
import torch

import src.citation_search as search
import src.relu_ntk as ntk
from src.io import _fingerprint


def fixture(tmp_path):
    candidate = dict(method="nystrom", lr=.01, penalty=.1, rank=2,
                     inner_loss_weighting="uniform", surrogate_kernel=ntk.KIND)
    folder = tmp_path / _fingerprint(candidate) / "condensation_0"
    folder.mkdir(parents=True)
    (folder / "resume.pt").touch()
    h = torch.tensor([[1., .2], [.3, 1.], [-.4, .7], [.8, -.1]], dtype=torch.double)
    q = torch.tensor([[.7, .3], [.3, .7], [.2, .8], [.8, .2]], dtype=torch.double)
    assignment = torch.tensor([0, 1, 1, 0])
    mass = torch.tensor([.4, .6], dtype=torch.double)
    centers = torch.tensor([[.8, .2], [-.4, .7]], dtype=torch.double)
    targets = torch.tensor([[.7, .3], [.2, .8]], dtype=torch.double)
    moments = mass[:, None] * torch.cat((torch.ones(2, 1), centers, targets), 1)
    saved = dict(moments=moments, theta=torch.zeros(2, 4, dtype=torch.double),
                 inner_loss_weighting="uniform", input_fingerprint={"frozen": "same"},
                 extra={"keep": "shared"})
    feature_map = SimpleNamespace(anchors=torch.tensor([[1., 0.], [0., 1.], [-1., -1.]],
                                                      dtype=torch.double), scale=2.)
    return candidate, saved, h, q, assignment, feature_map


@pytest.mark.parametrize("load_inputs", [False, True])
@pytest.mark.parametrize("target_device", ["cpu", "meta"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_snapshot_device_view_preserves_original_payload(tmp_path, monkeypatch, load_inputs, target_device, dtype):
    """Meta is a CPU-only transfer spy, not evidence about CUDA numerics."""
    candidate, saved, h, q, assignment, _ = fixture(tmp_path)
    saved["moments"] = saved["moments"].to(dtype)
    original = saved["moments"]
    before = original.clone()
    h = h.to(target_device)
    observed = []
    if load_inputs:
        monkeypatch.setattr(search, "_cached_ntk_inputs", lambda *args: (h, q, assignment))

    def validate(view, actual_candidate, actual_h, actual_q, actual_assignment, root, seed, *, resume):
        observed.append(view)
        assert not resume and view is not saved
        assert view["moments"].device == actual_h.device == torch.device(target_device)
        assert view["moments"].dtype == original.dtype and view["moments"].shape == original.shape
        assert view["theta"] is saved["theta"] and view["extra"] is saved["extra"]
        assert actual_candidate is candidate and actual_q is q and actual_assignment is assignment
        assert root == tmp_path and seed == 0
        if target_device == "cpu":
            assert torch.equal(view["moments"], before)

    monkeypatch.setattr(ntk, "validate_cached", validate)
    inputs = {} if load_inputs else dict(h=h, q=q, assignment=assignment)
    search._check_nystrom_assignment(candidate, saved, root=tmp_path, condensation_seed=0,
                                     device="cpu", **inputs)
    assert len(observed) == 1 and saved["moments"] is original
    assert original.device.type == "cpu" and torch.equal(original, before)


def test_resume_does_not_copy_or_transfer_payload(tmp_path, monkeypatch):
    candidate, saved, h, q, assignment, _ = fixture(tmp_path)
    original = saved["moments"]
    observed = []

    def validate(view, *args, resume):
        observed.append(view)
        assert resume and view is saved and view["moments"] is original
        assert view["moments"].device.type == "cpu"

    monkeypatch.setattr(ntk, "validate_cached", validate)
    search._check_nystrom_assignment(candidate, saved, resume=True, root=tmp_path,
                                     condensation_seed=0, h=h.to("meta"), q=q, assignment=assignment)
    assert observed == [saved]


def test_legacy_default_does_not_enter_ntk_device_path(tmp_path, monkeypatch):
    candidate, saved, h, q, assignment, _ = fixture(tmp_path)
    candidate.pop("surrogate_kernel")
    original = saved["moments"]

    def forbidden(*args, **kwargs):
        raise AssertionError("Legacy cache must not enter the NTK validator")

    monkeypatch.setattr(ntk, "validate_cached", forbidden)
    search._check_nystrom_assignment(candidate, saved, root=tmp_path, condensation_seed=0,
                                     h=h.to("meta"), q=q, assignment=assignment)
    assert saved["moments"] is original and original.device.type == "cpu"


@pytest.mark.parametrize("bad", [None, [], "malformed"])
def test_non_tensor_moments_keep_existing_clean_rejection(tmp_path, monkeypatch, bad):
    candidate, saved, h, q, assignment, feature_map = fixture(tmp_path)
    saved["moments"] = bad
    monkeypatch.setattr(ntk, "expected_fingerprint", lambda *args: ({"frozen": "same"}, feature_map))
    with pytest.raises(ValueError, match="moments are invalid"):
        search._check_nystrom_assignment(candidate, saved, root=tmp_path, condensation_seed=0,
                                         h=h, q=q, assignment=assignment)
    assert saved["moments"] is bad


def test_actual_cpu_domain_guard_accepts_same_device_snapshot(tmp_path, monkeypatch):
    candidate, saved, h, q, assignment, feature_map = fixture(tmp_path)
    before = saved["moments"].clone()
    monkeypatch.setattr(ntk, "expected_fingerprint", lambda *args: ({"frozen": "same"}, feature_map))
    search._check_nystrom_assignment(candidate, saved, root=tmp_path, condensation_seed=0,
                                     h=h, q=q, assignment=assignment)
    assert torch.equal(saved["moments"], before)


@pytest.mark.parametrize("change", ["fingerprint", "shape", "nonfinite", "head", "cusp"])
def test_transfer_does_not_relax_actual_ntk_cache_guards(tmp_path, monkeypatch, change):
    candidate, saved, h, q, assignment, feature_map = fixture(tmp_path)
    if change == "fingerprint":
        saved["input_fingerprint"] = {"tampered": True}
    elif change == "shape":
        saved["moments"] = saved["moments"][:, :-1]
    elif change == "nonfinite":
        saved["moments"][0, 1] = float("nan")
    elif change == "head":
        saved["theta"] = saved["theta"][:, :-1]
    else:
        saved["moments"][0, 1:3] = saved["moments"][0, 0] * feature_map.anchors[0]
    original = saved["moments"]
    before = original.clone()
    monkeypatch.setattr(ntk, "expected_fingerprint", lambda *args: ({"frozen": "same"}, feature_map))
    with pytest.raises((ValueError, FloatingPointError)):
        search._check_nystrom_assignment(candidate, saved, root=tmp_path, condensation_seed=0,
                                         h=h, q=q, assignment=assignment)
    assert saved["moments"] is original
    torch.testing.assert_close(original, before, equal_nan=True, atol=0, rtol=0)


@pytest.mark.skipif(os.environ.get("GRIP_RUN_CUDA_SNAPSHOT_TEST") != "1",
                    reason="Explicit CUDA opt-in required; author regression run is CPU-only")
def test_cpu_loaded_snapshot_validates_against_cuda_map(tmp_path, monkeypatch):
    """Optional synthetic same-device domain check; no real cached endpoint claim."""
    candidate, saved, h, q, assignment, feature_map = fixture(tmp_path)
    original = saved["moments"]
    before = original.clone()
    h, q, assignment = h.cuda(), q.cuda(), assignment.cuda()
    feature_map.anchors = feature_map.anchors.cuda()
    monkeypatch.setattr(ntk, "expected_fingerprint", lambda *args: ({"frozen": "same"}, feature_map))
    search._check_nystrom_assignment(candidate, saved, root=tmp_path, condensation_seed=0,
                                     device="cuda", h=h, q=q, assignment=assignment)
    assert original.device.type == "cpu" and saved["moments"] is original and torch.equal(original, before)
