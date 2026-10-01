"""Proposed CPU regressions to apply alongside minimal.patch."""

import numpy as np
import pytest
import test_large_pilot as pipeline_fixtures
import torch

import src.large_pilot as pilot
import src.nystrom_ce as nystrom
from src.low_rank_assignment import logit_block
from src.moments import decode_moments, initial_logits, make_material
from src.soft_ce_partition import optimize_ce_assignment

mocked_pipeline = pipeline_fixtures.mocked_pipeline


@pytest.mark.parametrize("surrogate", ["linear", "nystrom"])
def test_pilot_forwards_nondefault_mixing_and_separates_candidate_only(
    tmp_path, mocked_pipeline, monkeypatch, surrogate
):
    name = "optimize_ce_assignment" if surrogate == "linear" else "optimize_nystrom"
    previous = getattr(pilot, name)
    supplied = []

    def capture(*args, **kwargs):
        supplied.append(kwargs["mixing"])
        return previous(*args, **kwargs)

    monkeypatch.setattr(pilot, name, capture)
    kwargs = dict(device="cpu", basis=2, rank=2, steps=20, surrogate=surrogate)
    baseline, old_root = pilot.run_pilot("flickr", 0.001, tmp_path, **kwargs)
    changed, new_root = pilot.run_pilot("flickr", 0.001, tmp_path, mixing=0.005, **kwargs)
    assert baseline["status"] == changed["status"] == "complete"
    assert "mixing" not in baseline["candidate"]
    assert changed["candidate"]["mixing"] == 0.005
    assert old_root != new_root and old_root.parent == new_root.parent
    assert supplied == [0.05, 0.005]
    assert mocked_pipeline["teacher"] == 1
    old_assignment = torch.load(old_root / "initial_assignment.pt", weights_only=False)
    new_assignment = torch.load(new_root / "initial_assignment.pt", weights_only=False)
    assert torch.equal(old_assignment, new_assignment)


@pytest.mark.parametrize("surrogate", ["linear", "nystrom"])
def test_implicit_and_explicit_default_reuse_existing_candidate(
    tmp_path, mocked_pipeline, surrogate
):
    kwargs = dict(device="cpu", basis=2, rank=2, steps=20, surrogate=surrogate)
    first, root = pilot.run_pilot("flickr", 0.001, tmp_path, **kwargs)
    second, same = pilot.run_pilot("flickr", 0.001, tmp_path, mixing=0.05, **kwargs)
    assert first["candidate"] == second["candidate"]
    assert same == root
    assert mocked_pipeline["teacher"] == 1
    assert mocked_pipeline["optimizer" if surrogate == "linear" else "nystrom"] == 1


@pytest.mark.parametrize("mixing", [0, 1, -0.005, float("nan"), float("inf"), True, "0.005"])
def test_invalid_mixing_is_rejected_before_data_loading(tmp_path, mocked_pipeline, mixing):
    with pytest.raises(ValueError, match="mixing"):
        pilot.run_pilot("flickr", 0.001, tmp_path, device="cpu", basis=2, rank=2, mixing=mixing)
    assert mocked_pipeline["data"] == mocked_pipeline["teacher"] == 0


def _problem():
    generator = torch.Generator().manual_seed(17)
    h = torch.randn(18, 3, generator=generator, dtype=torch.double) + 0.2
    q = torch.randn(18, 3, generator=generator, dtype=torch.double).softmax(1)
    assignment = torch.arange(18) % 4
    feature_map = nystrom.NystromMap.fit(h, basis=7, seed=0)
    return h, q, assignment, feature_map, feature_map(h).numpy()


def test_nystrom_default_accepts_verified_prepatch_resume(tmp_path):
    h, q, assignment, feature_map, phi = _problem()
    kwargs = dict(penalty=0.1, rank=3, chunk=5, checkpoint_every=1)
    nystrom.optimize(h, q, assignment, feature_map, phi, tmp_path, 0, **kwargs)
    saved = torch.load(tmp_path / "resume.pt", weights_only=False)
    # This is exactly the prepatch steps_schema=2 config: default mixing was implicit.
    assert saved["config"] == dict(steps_schema=2, penalty=0.1, lr=0.01, rank=3, seed=0,
                                   cells=4, chunk=5)
    nystrom.optimize(h, q, assignment, feature_map, phi, tmp_path, 1, mixing=0.05, **kwargs)
    assert torch.load(tmp_path / "resume.pt", weights_only=False)["step"] == 1


def test_nondefault_nystrom_resume_is_exact_and_cross_mixing_is_rejected(tmp_path):
    h, q, assignment, feature_map, phi = _problem()
    kwargs = dict(penalty=0.1, rank=3, chunk=5, checkpoint_every=1, mixing=0.005)
    nystrom.optimize(h, q, assignment, feature_map, phi, tmp_path / "full", 2, **kwargs)
    nystrom.optimize(h, q, assignment, feature_map, phi, tmp_path / "resumed", 1, **kwargs)
    nystrom.optimize(h, q, assignment, feature_map, phi, tmp_path / "resumed", 2, **kwargs)
    full = torch.load(tmp_path / "full" / "step_000002.pt", weights_only=False)
    resumed = torch.load(tmp_path / "resumed" / "step_000002.pt", weights_only=False)
    torch.testing.assert_close(full["moments"], resumed["moments"], atol=1e-10, rtol=1e-8)
    torch.testing.assert_close(full["theta"], resumed["theta"], atol=1e-7, rtol=1e-5)
    path = tmp_path / "resumed" / "resume.pt"
    saved = torch.load(path, weights_only=False)
    assert saved["config"]["mixing"] == 0.005
    original = path.read_bytes()
    with pytest.raises(ValueError, match="configuration differs"):
        nystrom.optimize(h, q, assignment, feature_map, phi, tmp_path / "resumed", 3,
                         **dict(kwargs, mixing=0.05))
    assert path.read_bytes() == original


def test_linear_core_already_rejects_cross_mixing_resume(tmp_path):
    h, q, assignment, _, _ = _problem()
    kwargs = dict(penalty=0.1, assignment_rank=3, chunk_size=5, inner_method="newton_first",
                  save_assignment=False, save_resume=True, checkpoint_steps=(0,), steps=0,
                  folder=tmp_path, mixing=0.005)
    optimize_ce_assignment(h, q, assignment, **kwargs)
    path = tmp_path / "resume.pt"
    saved = torch.load(path, weights_only=False)
    assert saved["config"]["mixing"] == 0.005
    original = path.read_bytes()
    with pytest.raises(ValueError, match="Resume state does not match"):
        optimize_ce_assignment(h, q, assignment, **dict(kwargs, mixing=0.05, resume_state=saved))
    assert path.read_bytes() == original


def test_initial_mixing_changes_centroids_but_is_not_a_residual_probability_floor():
    # A tiny cell beside a large cell exposes contamination by the global background.
    assignment = torch.tensor([0] + [1] * 99)
    h = torch.tensor([[10.0]] + [[0.0]] * 99, dtype=torch.double)
    q = torch.nn.functional.one_hot(assignment, 2).double()
    centers = []
    for mixing in (0.05, 0.005):
        probability = initial_logits(assignment, 2, mixing, dtype=torch.double).softmax(1)
        expected = ((1 - mixing) * torch.nn.functional.one_hot(assignment, 2).double()
                    + mixing / 2)
        torch.testing.assert_close(probability, expected)
        moments = probability.T @ make_material(h, q) / len(h)
        cx, _, _ = decode_moments(moments, 1)
        centers.append(float(cx[0, 0]))
    assert abs(centers[1] - 10) < abs(centers[0] - 10)
    u = torch.ones(1, 1, dtype=torch.double)
    v = torch.tensor([[-50.0], [50.0]], dtype=torch.double)
    probability = logit_block(u, v, torch.tensor([0]), 0.005).softmax(1)
    assert float(probability[0, 0]) < 0.005 / 2
    assert bool(torch.isfinite(probability).all())


@pytest.mark.parametrize("mixing", [0, 1, np.nan, np.inf, True, "0.005"])
def test_nystrom_invalid_mixing_does_not_create_a_result_folder(tmp_path, mixing):
    h, q, assignment, feature_map, phi = _problem()
    folder = tmp_path / "invalid"
    with pytest.raises(ValueError, match="mixing"):
        nystrom.optimize(h, q, assignment, feature_map, phi, folder, 0, mixing=mixing)
    assert not folder.exists()
