import json

import numpy as np
import pytest
import torch

from src.moments import augmented, decode_moments, initial_logits, make_material
from src.nystrom_ce import (
    NystromMap,
    _cache_identity,
    _content_digest,
    _validate_phi,
    cache_features,
    moment_gradient,
    optimize,
    outer_gradient,
)
from src.soft_ce_partition import solve_head_system, solve_inner_newton_first
from src.transforms import fit_transform


def problem():
    torch.manual_seed(7)
    h = torch.randn(24, 3, dtype=torch.double) + 0.2
    q = torch.randn(24, 3, dtype=torch.double).softmax(1)
    assignment = torch.arange(24) % 4
    feature_map = NystromMap.fit(h, basis=8)
    return h, q, assignment, feature_map


def test_nonlinear_implicit_gradient_matches_refitted_finite_difference():
    h, q, assignment, feature_map = problem()
    moments = initial_logits(assignment, 4).double().softmax(1).T @ make_material(h, q) / len(h)
    phi = feature_map(h).detach().numpy()

    def objective(value):
        centers, labels, mass = decode_moments(value, h.shape[1])
        mapped = feature_map(centers).detach()
        fitted = solve_inner_newton_first(mapped, labels, mass, 0.1, grad_tol=1e-10)
        assert fitted["inner_converged"]
        loss, rhs = outer_gradient(phi, q, fitted["theta"], chunk=7)
        return loss, rhs, fitted["theta"], mapped, labels, mass

    _, rhs, theta, mapped, labels, mass = objective(moments)
    vector, info = solve_head_system(augmented(mapped), labels, mass, theta, 0.1, rhs, rtol=1e-10)
    assert info["cg_converged"]
    derivative = moment_gradient(moments, 3, feature_map, theta, vector, 0.1)
    direction = torch.randn_like(moments) * 0.01
    epsilon = 1e-3
    numerical = (objective(moments + epsilon * direction)[0] -
                 objective(moments - epsilon * direction)[0]) / (2 * epsilon)
    np.testing.assert_allclose(float((derivative * direction).sum()), numerical, rtol=2e-4, atol=1e-7)


def test_checkpoint_resume_matches_uninterrupted_updates(tmp_path):
    h, q, assignment, feature_map = problem()
    phi = feature_map(h).detach().numpy()
    kwargs = dict(penalty=0.1, rank=3, chunk=7, checkpoint_every=1)
    optimize(h, q, assignment, feature_map, phi, tmp_path / "full", 2, **kwargs)
    optimize(h, q, assignment, feature_map, phi, tmp_path / "resumed", 1, **kwargs)
    optimize(h, q, assignment, feature_map, phi, tmp_path / "resumed", 2, **kwargs)
    a = torch.load(tmp_path / "full" / "step_000002.pt", weights_only=False)
    b = torch.load(tmp_path / "resumed" / "step_000002.pt", weights_only=False)
    torch.testing.assert_close(a["moments"], b["moments"], rtol=1e-8, atol=1e-10)
    torch.testing.assert_close(a["theta"], b["theta"], rtol=1e-5, atol=1e-7)


def test_feature_cache_reuses_float_and_double_h_with_matching_coordinates(tmp_path):
    h, _, _, _ = problem()
    h = h.float()
    feature_map = NystromMap.fit(h.double(), basis=8)
    path = tmp_path / "phi.npy"
    actual = cache_features(h, feature_map, path, chunk=5)
    assert isinstance(actual, np.memmap)
    metadata = json.loads(path.with_suffix(".meta.json").read_text())
    reused = cache_features(h.double(), feature_map, path, chunk=7)
    np.testing.assert_array_equal(actual, reused)
    np.testing.assert_allclose(actual, feature_map(h).numpy())
    assert metadata == json.loads(path.with_suffix(".meta.json").read_text())


@pytest.mark.parametrize("changed", ["h", "anchors", "mapping", "kernel", "content"])
def test_shape_matching_stale_feature_cache_is_rejected_without_overwriting(tmp_path, changed):
    h, _, _, feature_map = problem()
    path = tmp_path / "phi.npy"
    cache_features(h, feature_map, path, chunk=5)
    original_metadata = path.with_suffix(".meta.json").read_bytes()
    if changed == "h":
        h = h + 0.03
    elif changed == "anchors":
        feature_map.anchors = feature_map.anchors + 0.03
    elif changed == "mapping":
        feature_map.mapping = feature_map.mapping * 1.01
    elif changed == "kernel":
        feature_map.kernel = "linear"
    else:
        phi = np.load(path, mmap_mode="r+")
        phi[2, 3] += 0.01
        phi.flush()
        del phi
    original_cache = path.read_bytes()
    with pytest.raises(ValueError, match="fingerprint differs"):
        cache_features(h, feature_map, path, chunk=5)
    assert path.read_bytes() == original_cache
    assert path.with_suffix(".meta.json").read_bytes() == original_metadata


def test_legacy_cache_requires_full_semantic_verification_before_migration(tmp_path):
    h, _, _, feature_map = problem()
    path = tmp_path / "legacy.npy"
    np.save(path, feature_map(h).numpy())
    cache = cache_features(h, feature_map, path, chunk=5)
    assert path.with_suffix(".meta.json").exists()
    np.testing.assert_array_equal(cache, np.load(path))
    stale_path = tmp_path / "stale.npy"
    stale = feature_map(h).numpy()
    stale[-1, -1] += 0.01  # Last block must be checked, not just sampled rows.
    np.save(stale_path, stale)
    with pytest.raises(ValueError, match="does not match current H/map"):
        cache_features(h, feature_map, stale_path, chunk=5)
    assert not stale_path.with_suffix(".meta.json").exists()


def test_validated_memmap_metadata_avoids_full_map_recomputation(tmp_path):
    h, _, _, feature_map = problem()
    path = tmp_path / "phi.npy"
    phi = cache_features(h, feature_map, path, chunk=5)
    identity = _cache_identity(h, feature_map, phi.shape, 5, lambda: False)

    def forbidden_full_evaluation(block):
        raise AssertionError("Validated shared cache must not recompute kernel features")

    digest = _validate_phi(h, forbidden_full_evaluation, phi, identity, 5, lambda: False)
    assert digest == json.loads(path.with_suffix(".meta.json").read_text())["phi_digest"]


@pytest.mark.parametrize("changed", ["h", "q", "assignment", "map", "phi"])
def test_optimizer_rejects_same_shape_stale_inputs_and_preserves_resume(tmp_path, changed):
    h, q, assignment, feature_map = problem()
    phi = feature_map(h).numpy()
    folder = tmp_path / "trial"
    kwargs = dict(penalty=0.1, rank=3, chunk=7, checkpoint_every=1)
    optimize(h, q, assignment, feature_map, phi, folder, 1, **kwargs)
    original_resume = (folder / "resume.pt").read_bytes()
    if changed == "h":
        h = h + 0.03
        phi = feature_map(h).numpy()
    elif changed == "q":
        q = q.roll(1, dims=0)
    elif changed == "assignment":
        assignment = assignment.roll(1)
    elif changed == "map":
        feature_map.mapping = feature_map.mapping * 1.01
        phi = feature_map(h).numpy()
    else:
        phi = phi.copy()
        phi[-1, -1] += 0.01
    with pytest.raises(ValueError, match="fingerprint differs|does not match current H/map"):
        optimize(h, q, assignment, feature_map, phi, folder, 2, **kwargs)
    assert (folder / "resume.pt").read_bytes() == original_resume


def test_legacy_resume_with_no_input_proof_is_preserved_and_rejected(tmp_path):
    h, q, assignment, feature_map = problem()
    phi = feature_map(h).numpy()
    kwargs = dict(penalty=0.1, rank=3, chunk=7, checkpoint_every=1)
    optimize(h, q, assignment, feature_map, phi, tmp_path, 1, **kwargs)
    path = tmp_path / "resume.pt"
    saved = torch.load(path, weights_only=False)
    saved.pop("input_fingerprint")
    saved["config"]["steps_schema"] = 1
    torch.save(saved, path)
    original = path.read_bytes()
    with pytest.raises(ValueError, match="Legacy resume lacks verifiable"):
        optimize(h, q, assignment, feature_map, phi, tmp_path, 2, **kwargs)
    assert path.read_bytes() == original


def test_linear_callable_transform_and_deadline_proxy_remain_supported(tmp_path):
    h, q, assignment, _ = problem()
    _, transform = fit_transform(h)

    def linear_map(centers):
        return transform(centers)

    class PhiProxy:
        def __init__(self, data):
            self.data = data
            self.shape, self.dtype, self.filename = data.shape, data.dtype, data.filename

        def __len__(self):
            return len(self.data)

        def __getitem__(self, index):
            return self.data[index]

    phi = cache_features(h, linear_map, tmp_path / "phi.npy", chunk=5)
    proxy = PhiProxy(phi)
    kwargs = dict(penalty=0.1, rank=3, chunk=7, checkpoint_every=1)
    optimize(h, q, assignment, linear_map, proxy, tmp_path / "condensation", 1, **kwargs)
    optimize(h, q, assignment, linear_map, proxy, tmp_path / "condensation", 2, **kwargs)
    saved = torch.load(tmp_path / "condensation" / "resume.pt", weights_only=False)
    assert saved["step"] == 2
    assert "map_digest" in saved["input_fingerprint"]


def test_callable_map_changes_between_nodes_reject_resume_even_when_phi_is_identical(tmp_path):
    h, q, assignment, _ = problem()
    reference = h[:, 0].clone()

    def make_map(scale):
        def feature_map(values):
            # Zero at every original node, nonzero at most cluster barycenters.
            extra = (values[:, 0, None] - reference[None]).prod(1, keepdim=True)
            return values + scale * extra

        return feature_map

    original, changed = make_map(0.0), make_map(0.2)
    phi = original(h).numpy()
    np.testing.assert_array_equal(phi, changed(h).numpy())
    kwargs = dict(penalty=0.1, rank=3, chunk=7, checkpoint_every=1)
    optimize(h, q, assignment, original, phi, tmp_path, 1, **kwargs)
    with pytest.raises(ValueError, match="Resume input fingerprint differs"):
        optimize(h, q, assignment, changed, phi, tmp_path, 2, **kwargs)


def test_module_map_hook_state_cannot_reuse_stale_cache(tmp_path):
    h, _, _, _ = problem()
    feature_map = torch.nn.Linear(3, 4).double()
    path = tmp_path / "phi.npy"
    cache_features(h, feature_map, path, chunk=5)
    handle = feature_map.register_forward_hook(lambda module, arguments, output: output + 0.01)
    with pytest.raises(ValueError, match="fingerprint differs"):
        cache_features(h, feature_map, path, chunk=5)
    handle.remove()


def test_streaming_digest_checks_stop_between_blocks():
    counter = 0

    def stop():
        nonlocal counter
        counter += 1
        return counter == 3

    with pytest.raises(InterruptedError, match="input validation interrupted"):
        _content_digest(np.ones((12, 4)), chunk=2, stop=stop)
    assert counter == 3
