"""CPU checks for fixed-assignment structural transfer and paired validation."""

import copy
import hashlib
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import src.large_pilot as original_pilot
import src.large_quotient_pilot as pilot
from src.evaluation import _forward
from src.inductive_evaluation import fit_inductive_gcn
from src.io import save_json, save_state
from src.large_pilot import _data_digest
from src.models import GCN
from src.moments import initial_logits, make_material
from src.student_routes import replay_routes
from src.transforms import fit_transform


@pytest.fixture(scope="module", autouse=True)
def small_cpu_jobs():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize(matrix):
    inverse = matrix.sum(1).rsqrt()
    return inverse[:, None] * matrix * inverse[None, :]


@pytest.fixture
def frozen():
    # Nonregular degrees make double normalization and an extra identity visible.
    support = torch.tensor([
        [1., 1., 1., 0., 0.], [1., 1., 0., 0., 0.],
        [1., 0., 1., 1., 0.], [0., 0., 1., 1., 1.], [0., 0., 0., 1., 1.],
    ], dtype=torch.double)
    adjacency = normalize(support).float().to_sparse_csr()
    x = torch.tensor([[2., .1], [.2, 2.], [1.5, .3], [.3, 1.4], [.8, .5]])
    h = torch.sparse.mm(adjacency, torch.sparse.mm(adjacency, x))
    logits = torch.tensor([[1.1, -.2], [-.3, 1.2], [.9, .1], [.1, .7], [.7, .4]],
                          dtype=torch.double)
    q = (logits / .3).softmax(1)
    assignment = torch.tensor([0, 1, 0, 1, 0])
    u = torch.tensor([[.6, -.2], [-.4, .3], [.1, .5], [-.3, -.4], [.8, .1]])
    v = torch.tensor([[.4, .7], [-.8, .2]])
    # Saved dtype arithmetic comes before double softmax; scale is unrelated.
    probability = (initial_logits(assignment, 2, .05) + u @ v.T / math.sqrt(2)).double().softmax(1)
    z, transform = fit_transform(h.double(), kind="rms")
    endpoint = dict(step=20, J_exact=True, moments=probability.T @ make_material(z, q) / len(x))
    config = dict(assignment_input="node", assignment_encoder="linear", assignment_rank=2,
                  mass_mode="free", solver_mode="exact", feature_control="joint", mixing=.05,
                  chunk_size=2)
    resume = dict(config=config, parameters=[u, v], step=20, scale=.712, dual=None,
                  snapshots={20: copy.deepcopy(endpoint)})
    return SimpleNamespace(support=support, adjacency=adjacency, x=x, h=h, q=q, logits=logits,
                           assignment=assignment, probability=probability, transform=transform,
                           endpoint=endpoint, resume=resume)


@pytest.mark.parametrize("layout", ["dense", "coo", "csr"])
def test_binary_support_recovery_preserves_original_normalization(frozen, layout):
    packed = frozen.adjacency.to_dense()
    packed = {"dense": lambda a: a, "coo": torch.Tensor.to_sparse,
              "csr": torch.Tensor.to_sparse_csr}[layout](packed)
    support, diagnostics = pilot.raw_looped_support(packed)
    assert support.layout != torch.strided
    assert torch.equal(support.to_dense().double(), frozen.support)
    assert diagnostics["nodes"] == 5
    assert diagnostics["normalization_max_abs"] < 1e-7
    torch.testing.assert_close(normalize(support.to_dense()), packed.to_dense())


@pytest.mark.parametrize("permutation", [[0, 1, 2, 3, 4], [3, 0, 4, 1, 2]])
def test_singleton_cells_recover_source_s_up_to_permutation(frozen, permutation):
    indices = torch.tensor(permutation)
    probability = torch.eye(5, dtype=torch.double)[:, indices]
    z = frozen.transform(frozen.h.double())
    endpoint = dict(moments=probability.T @ make_material(z, frozen.q) / 5)
    pair = pilot.frozen_pair_inputs(probability, frozen.x, frozen.h, frozen.q, endpoint,
                                    packed_adjacency=frozen.adjacency, column_chunk=1)
    torch.testing.assert_close(pair["raw_quotient"]["adj"],
                               frozen.adjacency.to_dense()[indices][:, indices], atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(pair["raw_quotient"]["x"], frozen.x[indices])
    torch.testing.assert_close(pair["h_identity"]["x"], frozen.h[indices])


@pytest.mark.parametrize("column_chunk", [1, 2, 64])
def test_soft_quotient_retains_fractional_diagonal_without_mass_scaling_or_extra_i(frozen, column_chunk):
    p = frozen.probability
    pair = pilot.frozen_pair_inputs(p, frozen.x, frozen.h, frozen.q, frozen.endpoint,
                                    packed_adjacency=frozen.adjacency, column_chunk=column_chunk)
    quotient = p.T @ frozen.support @ p
    expected = normalize(quotient).float()
    torch.testing.assert_close(pair["raw_quotient"]["adj"], expected, atol=1e-7, rtol=1e-6)
    assert bool((quotient.diagonal() > 1).all())
    assert not torch.allclose(expected, normalize(quotient + torch.eye(2)).float())
    inverse_mass = p.sum(0).rsqrt()
    scaled = inverse_mass[:, None] * quotient * inverse_mass[None, :]
    assert not torch.allclose(expected, normalize(scaled).float())
    # Coarsening packed S followed by another normalization is a distinct graph.
    assert not torch.allclose(expected, normalize(p.T @ frozen.adjacency.to_dense().double() @ p).float())
    mass = p.sum(0)
    for arm in ("h_identity", "raw_quotient"):
        torch.testing.assert_close(pair[arm]["labels"].double(), p.T @ frozen.q / mass[:, None])
        torch.testing.assert_close(pair[arm]["mass"].double(), mass / len(p))
    assert pair["h_identity"]["adj"] is None
    assert torch.equal(pair["h_identity"]["labels"], pair["raw_quotient"]["labels"])
    assert torch.equal(pair["h_identity"]["mass"], pair["raw_quotient"]["mass"])
    torch.testing.assert_close(pair["h_identity"]["x"].double(), p.T @ frozen.h.double() / mass[:, None])
    torch.testing.assert_close(pair["raw_quotient"]["x"].double(), p.T @ frozen.x.double() / mass[:, None])
    assert not torch.allclose(pair["h_identity"]["x"], pair["raw_quotient"]["x"])


@pytest.mark.parametrize("change", ["weighted", "asymmetric", "missing_loop", "negative", "nan"])
def test_raw_support_refuses_other_graph_conventions(frozen, change):
    packed = frozen.adjacency.to_dense().double()
    if change == "weighted":
        weighted = frozen.support.clone()
        weighted[0, 1] = weighted[1, 0] = 1.7
        packed = normalize(weighted)
    elif change == "asymmetric":
        packed[0, 1] = 0
    elif change == "missing_loop":
        packed[1, 1] = 0
    else:
        packed[0, 1] = -1 if change == "negative" else float("nan")
    with pytest.raises(ValueError):
        pilot.raw_looped_support(packed.to_sparse_csr())


@pytest.mark.parametrize("scale", [1e-10, .712, 71_200.])
@pytest.mark.parametrize("chunk", [1, 5])
def test_frozen_factors_replay_saved_dtype_logits_and_ignore_outer_loss_scale(frozen, scale, chunk):
    frozen.resume["scale"] = scale
    decoded = pilot.load_frozen_assignment(frozen.resume, frozen.assignment, frozen.endpoint,
                                           expected_step=20, nodes=5, cells=2, chunk_size=chunk)
    assert torch.equal(decoded["probability"], frozen.probability)
    assert decoded["probability"].dtype == torch.double
    assert not decoded["probability"].requires_grad
    torch.testing.assert_close(decoded["node_mass"], frozen.probability.sum(0))
    assert decoded["scale"] == scale
    assert decoded["diagnostics"]["objective_scale_applied_to_probability"] is False


@pytest.mark.parametrize("key,value", [
    ("assignment_input", "features"), ("assignment_encoder", "mlp"), ("mass_mode", "uniform"),
    ("solver_mode", "tracking"), ("node_weighting", True), ("assignment_rank", 1),
    ("feature_control", "independent"),
    ("mixing", 0.), ("mixing", float("nan")), ("mixing", True),
    ("temperature_logits_digest", "learned-temperature"),
])
def test_frozen_factor_loader_refuses_unsupported_or_malformed_recipes(frozen, key, value):
    frozen.resume["config"][key] = value
    with pytest.raises(ValueError):
        pilot.load_frozen_assignment(frozen.resume, frozen.assignment, frozen.endpoint, expected_step=20)


@pytest.mark.parametrize("change", ["step", "endpoint_step", "snapshot_step", "inexact",
                                  "snapshot_inexact", "snapshot_moments", "parameters", "nonfinite_u",
                                  "dtype", "invalid_index", "dual", "scale"])
def test_frozen_factor_loader_checks_parameter_snapshot_endpoint_integrity(frozen, change):
    if change == "step":
        frozen.resume["step"] = 19
    elif change == "endpoint_step":
        frozen.endpoint["step"] = 19
    elif change == "snapshot_step":
        frozen.resume["snapshots"][20]["step"] = 19
    elif change == "inexact":
        frozen.endpoint["J_exact"] = False
    elif change == "snapshot_inexact":
        frozen.resume["snapshots"][20]["J_exact"] = False
    elif change == "snapshot_moments":
        frozen.resume["snapshots"][20]["moments"][0, 1] += .1
    elif change == "parameters":
        frozen.resume["parameters"].append(torch.zeros(5))
    elif change == "nonfinite_u":
        frozen.resume["parameters"][0][0, 0] = float("nan")
    elif change == "dtype":
        frozen.resume["parameters"][1] = frozen.resume["parameters"][1].double()
    elif change == "invalid_index":
        frozen.assignment[0] = 2
    elif change == "dual":
        frozen.resume["dual"] = torch.zeros(2)
    else:
        frozen.resume["scale"] = 0.
    with pytest.raises(ValueError):
        pilot.load_frozen_assignment(frozen.resume, frozen.assignment, frozen.endpoint, expected_step=20)


@pytest.mark.parametrize("change", ["missing_snapshot", "missing_snapshot_moments", "rank_too_large"])
def test_incomplete_snapshots_or_impossible_optimizer_rank_are_refused_clearly(frozen, change):
    if change == "missing_snapshot":
        frozen.resume["snapshots"].pop(20)
    elif change == "missing_snapshot_moments":
        frozen.resume["snapshots"][20].pop("moments")
    else:
        frozen.resume["config"]["assignment_rank"] = 3
        frozen.resume["parameters"] = [torch.zeros(5, 3), torch.zeros(2, 3)]
    with pytest.raises(ValueError):
        pilot.load_frozen_assignment(frozen.resume, frozen.assignment, frozen.endpoint, expected_step=20)


@pytest.mark.parametrize("changed", ["probability", "h", "q", "moments"])
def test_reconstructed_p_must_reproduce_the_saved_h_q_moments_before_graph_construction(frozen, changed,
                                                                                      monkeypatch):
    if changed == "probability":
        frozen.probability[0] = frozen.probability[0].flip(0)
    elif changed == "h":
        frozen.h[0, 0] += .2
    elif changed == "q":
        frozen.q[0] = frozen.q[0].flip(0)
    else:
        frozen.endpoint["moments"][0, 1] += .1

    def forbid_construction(*args, **kwargs):
        pytest.fail("Inconsistent frozen moments must fail before quotient construction")

    monkeypatch.setattr(pilot, "quotient_adjacency", forbid_construction)
    with pytest.raises(ValueError, match="moments"):
        pilot.frozen_pair_inputs(frozen.probability, frozen.x, frozen.h, frozen.q, frozen.endpoint,
                                 packed_adjacency=frozen.adjacency)


def test_explicit_fixed_rms_transform_matches_implicit_reconstruction(frozen):
    arguments = (frozen.probability, frozen.x, frozen.h, frozen.q, frozen.endpoint)
    implicit = pilot.frozen_pair_inputs(*arguments, packed_adjacency=frozen.adjacency)
    explicit = pilot.frozen_pair_inputs(*arguments, transform=frozen.transform,
                                       packed_adjacency=frozen.adjacency)
    for arm in ("h_identity", "raw_quotient"):
        for key in ("x", "labels", "mass"):
            assert torch.equal(implicit[arm][key], explicit[arm][key])


def test_construction_cooperative_stop_prevents_quotient_work(frozen, monkeypatch):
    def forbid_construction(*args, **kwargs):
        pytest.fail("Stopped construction must not build a quotient")

    monkeypatch.setattr(pilot, "quotient_adjacency", forbid_construction)
    with pytest.raises(InterruptedError):
        pilot.frozen_pair_inputs(frozen.probability, frozen.x, frozen.h, frozen.q, frozen.endpoint,
                                 packed_adjacency=frozen.adjacency, stop=lambda: True)
    with pytest.raises(InterruptedError):
        pilot.load_frozen_assignment(frozen.resume, frozen.assignment, frozen.endpoint,
                                     expected_step=20, stop=lambda: True)


def test_assignment_stop_is_polled_after_each_row_chunk(frozen, monkeypatch):
    original = pilot.assignment_probability
    rows = []

    def replay_chunk(saved, **kwargs):
        rows.append(len(saved["u"]))
        return original(saved, **kwargs)

    monkeypatch.setattr(pilot, "assignment_probability", replay_chunk)
    with pytest.raises(InterruptedError):
        pilot.load_frozen_assignment(frozen.resume, frozen.assignment, frozen.endpoint,
                                     expected_step=20, chunk_size=1, stop=lambda: bool(rows))
    assert rows == [1]


def test_quotient_stop_is_polled_between_sparse_column_products(frozen, monkeypatch):
    original = torch.sparse.mm
    columns = []

    def multiply(adjacency, block):
        columns.append(block.shape[1])
        return original(adjacency, block)

    monkeypatch.setattr(torch.sparse, "mm", multiply)
    with pytest.raises(InterruptedError):
        pilot.frozen_pair_inputs(frozen.probability, frozen.x, frozen.h, frozen.q, frozen.endpoint,
                                 packed_adjacency=frozen.adjacency, column_chunk=1,
                                 stop=lambda: bool(columns))
    assert columns == [1]


@pytest.fixture
def experiment(tmp_path, monkeypatch, frozen):
    train_graph = dict(x=frozen.x, y=torch.tensor([0, 1, 0, 1, 0]), adj=frozen.adjacency)
    train_mask = torch.ones(5, dtype=torch.bool)
    validation = dict(x=torch.tensor([[.1, 1.8], [1.7, .2], [1.1, .6]]), y=torch.tensor([1, 0, 0]),
                      adj=normalize(torch.tensor([[1., 1., 0.], [1., 1., 1.], [0., 1., 1.]])).to_sparse_csr())

    class TestLabelsForbidden:
        def __getitem__(self, item):
            pytest.fail("Validation-only quotient pilot accessed test labels")

        def __len__(self):
            pytest.fail("Validation-only quotient pilot inspected test labels")

    testing = dict(x=torch.full((7, 2), 99.), y=TestLabelsForbidden(), adj=torch.eye(7).to_sparse_csr())
    source_digest = _data_digest(train_graph, train_mask, (validation, None), frozen.h, lambda: None)
    settings = dict(epochs=4, eval_every=1, hidden=3, dropout=0., lr=.01, weight_decay=.0005)
    objects = dict(
        resume=frozen.resume, initial_assignment=frozen.assignment, endpoint=frozen.endpoint,
        teacher=dict(logits=frozen.logits, data_digest=source_digest, converged=True),
        propagated_h=dict(h=frozen.h),
        candidate_protocol=dict(data_digest=source_digest, candidate=dict(
            ratio=.001, cells=2, assignment="low_rank", temperature=.3, mixing=.05)),
        teacher_protocol=dict(data_digest=source_digest),
    )
    files = {}
    for role, value in objects.items():
        path = tmp_path / (role + (".json" if role.endswith("protocol") else ".pt"))
        (save_json if role.endswith("protocol") else save_state)(value, path)
        files[role] = dict(path=str(path), sha256=digest(path))
    manifest = dict(version=1, dataset="flickr", ratio=.001, step=20,
                    source_data_digest=source_digest, student_recipe=settings, files=files,
                    validation_only=True)
    manifest_path = tmp_path / "manifest.json"
    save_json(manifest, manifest_path)
    state = SimpleNamespace(frozen=frozen, manifest=manifest, manifest_path=manifest_path,
                            manifest_sha=digest(manifest_path), train=train_graph, mask=train_mask,
                            validation=validation, testing=testing, calls=[], route_calls=[],
                            optimizer_calls=[], data_calls=0, implementation="fixed-source-for-test")
    state.validation_h = torch.sparse.mm(validation["adj"], torch.sparse.mm(validation["adj"], validation["x"]))

    def prepare(dataset, data_dir, device):
        assert dataset == "flickr" and device == "cpu"
        state.data_calls += 1
        return train_graph, train_mask, (validation, None), (testing, None), frozen.h

    def fitted(cx, cy, mass, graph, supplied_validation, **kwargs):
        assert graph is train_graph
        assert supplied_validation == (validation, None)
        assert kwargs["train_mask"] is train_mask
        assert kwargs["weighting"] == "uniform"
        assert "testing" not in kwargs
        state.calls.append(dict(x=cx.clone(), labels=cy.clone(), mass=mass.clone(),
                                adjacency=kwargs["training_adjacency"], seed=kwargs["seed"],
                                settings=copy.deepcopy(kwargs["settings"]), folder=kwargs["folder"]))
        return fit_inductive_gcn(cx, cy, mass, graph, supplied_validation, **kwargs)

    def replay(path, graph, propagated, masks, settings, output_path, **kwargs):
        assert graph is validation
        assert set(masks) == {"val"}
        assert torch.equal(propagated, state.validation_h)
        state.route_calls.append(dict(path=path, graph=graph, propagated=propagated.clone(), masks=masks))
        return replay_routes(path, graph, propagated, masks, settings, output_path, **kwargs)

    original_adam = torch.optim.Adam

    def adam(parameters, *args, **kwargs):
        parameters = list(parameters)
        state.optimizer_calls.append([value.detach().clone() for value in parameters])
        return original_adam(parameters, *args, **kwargs)

    monkeypatch.setattr(torch.optim, "Adam", adam)
    monkeypatch.setattr(pilot, "fit_inductive_gcn", fitted)
    monkeypatch.setattr(pilot, "replay_routes", replay)
    monkeypatch.setattr(pilot, "_prepare_dataset", prepare)
    monkeypatch.setattr(pilot, "_implementation", lambda: dict(source_digest=state.implementation))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *args: pytest.fail("GPU used by a CPU test"))

    def call(**kwargs):
        options = dict(source_manifest_path=state.manifest_path, source_manifest_sha256=state.manifest_sha,
                       output_dir=tmp_path / "output", student_seeds=(7, 8), device="cpu",
                       deadline_seconds=10., column_chunk=1)
        options.update(kwargs)
        return pilot.run_quotient_pilot(**options)

    state.call = call
    state.object_hashes = {role: digest(record["path"]) for role, record in files.items()}
    return state


def test_frozen_pair_trains_only_students_with_same_seeds_targets_mass_and_validation_weights(experiment):
    state = experiment
    report, root = state.call()
    assert report["status"] == "complete"
    assert report["selection"] == "validation_only"
    assert report["peak_gpu_bytes"] == 0
    assert len(state.calls) == len(state.route_calls) == len(report["rows"]) == 4
    assert len(report["paired_deltas"]) == 2
    assert len(state.optimizer_calls) == 8  # Two scheduled Adam instances per four student fits.
    for offset in (0, 2):
        identity, quotient = state.calls[offset:offset + 2]
        assert identity["seed"] == quotient["seed"]
        assert identity["settings"] == quotient["settings"] == state.manifest["student_recipe"]
        assert torch.equal(identity["labels"], quotient["labels"])
        assert torch.equal(identity["mass"], quotient["mass"])
        assert identity["adjacency"] is None and quotient["adjacency"] is not None
        # Initial weights are paired even though the graph inputs differ.
        for a, b in zip(state.optimizer_calls[offset * 2], state.optimizer_calls[offset * 2 + 2], strict=True):
            assert torch.equal(a, b)
    for row in report["rows"]:
        assert not any("test_" in key for key in row)
        folder = root / row["arm"]
        selected = torch.load(folder / f"seed_{row['seed']}_selected.pt", weights_only=False)
        cache = json.loads((folder / f"seed_{row['seed']}_validation_routes.json").read_text())
        assert selected["epoch"] == row["epoch"] == cache["recipe"]["epoch"]
        assert cache["recipe"]["source_fingerprint"] == selected["fingerprint"]
        assert cache["recipe"]["test_enabled"] is False
        assert row["gcn_val_acc"] == row["val_acc"]
        # Independently score S²X with the saved GCN weights and no adjacency.
        with torch.random.fork_rng(devices=[]):
            model = GCN(2, 3, 2, 2, 0.)
        model.load_state_dict(selected["model_state"])
        model.eval()
        own_h = torch.sparse.mm(state.validation["adj"],
                                torch.sparse.mm(state.validation["adj"], state.validation["x"]))
        with torch.no_grad():
            prediction = _forward(model, own_h, None)
        assert row["mlp_val_ce"] == pytest.approx(float(F.nll_loss(prediction, state.validation["y"])))
    assert state.object_hashes == {role: digest(record["path"]) for role, record in state.manifest["files"].items()}


def test_completed_pair_reuses_native_student_and_route_caches_without_optimizer(experiment, monkeypatch):
    state = experiment
    first, root = state.call()
    assert first["status"] == "complete"
    files = {str(path.relative_to(root)): digest(path) for path in root.rglob("seed_*") if path.is_file()}

    def forbid_optimizer(*args, **kwargs):
        pytest.fail("A completed frozen pair must reuse its student caches")

    monkeypatch.setattr(torch.optim, "Adam", forbid_optimizer)
    second, same = state.call()
    assert second["status"] == "complete" and same == root
    assert first["rows"] == second["rows"]
    assert first["paired_deltas"] == second["paired_deltas"]
    assert files == {str(path.relative_to(root)): digest(path) for path in root.rglob("seed_*") if path.is_file()}


def test_frozen_quotient_run_never_refits_teacher_or_assignment(experiment, monkeypatch):
    def forbid_refit(*args, **kwargs):
        pytest.fail("Frozen structural transfer must not refit a teacher or assignment")

    for name in ("fit_streaming_teacher", "fit_gcn_teacher", "optimize_ce_assignment", "optimize_nystrom"):
        monkeypatch.setattr(original_pilot, name, forbid_refit)
    report, _ = experiment.call(student_seeds=(7,))
    assert report["status"] == "complete"
    assert len(experiment.calls) == 2


def test_cached_validation_h_preserves_first_rounding_during_resume(experiment, monkeypatch):
    state = experiment
    first, root = state.call(student_seeds=(7,))
    assert first["status"] == "complete"
    original = torch.sparse.mm

    def rounded(adjacency, features):
        value = original(adjacency, features)
        if adjacency.shape == (3, 3) and features.shape == (3, 2):
            return value + 3e-8
        return value

    monkeypatch.setattr(torch.sparse, "mm", rounded)
    second, same = state.call(student_seeds=(7,))
    assert second["status"] == "complete" and same == root
    assert first["rows"] == second["rows"]
    assert torch.equal(state.route_calls[-1]["propagated"], state.validation_h)


@pytest.mark.parametrize("change", ["teacher_incomplete", "teacher_inexact", "teacher_digest", "protocol_digest"])
def test_frozen_teacher_and_protocol_metadata_are_verified_before_loading_data(experiment, change):
    state = experiment
    role = "teacher_protocol" if change == "protocol_digest" else "teacher"
    path = Path(state.manifest["files"][role]["path"])
    value = json.loads(path.read_text()) if role.endswith("protocol") else torch.load(path, weights_only=False)
    if change == "teacher_incomplete":
        value["training_complete"] = False
    elif change == "teacher_inexact":
        value["converged"] = False
    else:
        value["data_digest"] = "different-source"
    (save_json if role.endswith("protocol") else save_state)(value, path)
    state.manifest["files"][role]["sha256"] = digest(path)
    save_json(state.manifest, state.manifest_path)
    state.manifest_sha = digest(state.manifest_path)
    report, _ = state.call()
    assert report["status"] == "failed"
    assert "teacher" in report["reason"].lower()
    assert state.data_calls == 0 and state.calls == []


@pytest.mark.parametrize("role", ["resume", "endpoint", "initial_assignment", "teacher", "propagated_h",
                                  "candidate_protocol", "teacher_protocol"])
def test_changed_frozen_file_fails_before_data_load_or_any_student(experiment, role):
    state = experiment
    path = Path(state.manifest["files"][role]["path"])
    path.write_bytes(path.read_bytes() + b"changed")
    report, _ = state.call()
    assert report["status"] == "failed" and "SHA256" in report["reason"]
    assert state.data_calls == 0 and state.calls == state.route_calls == []


def test_changed_manifest_fails_before_data_or_student(experiment):
    state = experiment
    state.manifest["step"] += 1
    save_json(state.manifest, state.manifest_path)
    with pytest.raises(ValueError, match="manifest changed"):
        state.call()
    assert state.data_calls == 0 and state.calls == []


def test_manifest_relative_paths_resolve_from_manifest_directory(experiment):
    state = experiment
    for record in state.manifest["files"].values():
        record["path"] = Path(record["path"]).name
    save_json(state.manifest, state.manifest_path)
    state.manifest_sha = digest(state.manifest_path)
    report, _ = state.call(student_seeds=(7,))
    assert report["status"] == "complete" and len(state.calls) == 2


@pytest.mark.parametrize("change", ["missing_endpoint", "nonfinite_lr", "test_recipe", "invalid_ratio"])
def test_malformed_manifest_or_student_recipe_fails_before_fit(experiment, change):
    state = experiment
    if change == "missing_endpoint":
        state.manifest["files"].pop("endpoint")
    elif change == "nonfinite_lr":
        state.manifest["student_recipe"]["lr"] = float("nan")
    elif change == "test_recipe":
        state.manifest["student_recipe"]["test_enabled"] = True
    else:
        state.manifest["ratio"] = 0.
    save_json(state.manifest, state.manifest_path)
    state.manifest_sha = digest(state.manifest_path)
    with pytest.raises(ValueError):
        state.call()
    assert state.data_calls == 0 and state.calls == []


@pytest.mark.parametrize("changed", ["features", "validation_labels", "adjacency"])
def test_source_graph_or_validation_mutation_is_refused_before_student(experiment, changed):
    state = experiment
    if changed == "features":
        state.train["x"][0, 0] += .1
    elif changed == "validation_labels":
        state.validation["y"][0] = 0
    else:
        state.train["adj"] = torch.eye(5).to_sparse_csr()
    report, _ = state.call()
    assert report["status"] == "failed" and "digest differs" in report["reason"]
    assert state.calls == []


def test_invalid_replayed_moments_fail_before_student_even_with_valid_file_hashes(experiment):
    state = experiment
    role = "endpoint"
    path = Path(state.manifest["files"][role]["path"])
    endpoint = torch.load(path, weights_only=False)
    endpoint["moments"][0, 1] += .1
    save_state(endpoint, path)
    state.manifest["files"][role]["sha256"] = digest(path)
    save_json(state.manifest, state.manifest_path)
    state.manifest_sha = digest(state.manifest_path)
    report, _ = state.call()
    assert report["status"] == "failed" and "endpoint moments" in report["reason"]
    assert state.calls == []


def test_stop_before_loading_keeps_resumable_report_and_never_touches_data(experiment):
    state = experiment
    report, root = state.call(stop=lambda: True)
    assert report["status"] == "stopped"
    assert state.data_calls == 0 and state.calls == []
    assert json.loads((root / "report.json").read_text())["status"] == "stopped"
    resumed, same = state.call()
    assert resumed["status"] == "complete" and same == root


def test_stop_between_arms_preserves_completed_student_for_resume(experiment, monkeypatch):
    state = experiment
    report, root = state.call(stop=lambda: len(state.route_calls) >= 1)
    assert report["status"] == "stopped"
    assert (root / "h_identity" / "seed_7_selected.pt").exists()
    assert not (root / "raw_quotient" / "seed_7_selected.pt").exists()
    before = digest(root / "h_identity" / "seed_7_selected.pt")
    state.calls.clear()
    state.route_calls.clear()
    state.optimizer_calls.clear()
    resumed, same = state.call()
    assert resumed["status"] == "complete" and same == root
    assert digest(root / "h_identity" / "seed_7_selected.pt") == before
    assert len(state.optimizer_calls) == 6  # Three remaining fits; first arm was already complete.


def test_deadline_bounds_a_blocking_student_and_retains_frozen_sources(experiment, monkeypatch):
    state = experiment

    def blocking_student(*args, **kwargs):
        time.sleep(2)
        pytest.fail("The deadline did not interrupt a blocking student")

    monkeypatch.setattr(pilot, "fit_inductive_gcn", blocking_student)
    started = time.monotonic()
    report, root = state.call(deadline_seconds=.15)
    assert time.monotonic() - started < 1.
    assert report["status"] == "stopped"
    assert json.loads((root / "report.json").read_text())["status"] == "stopped"
    assert state.object_hashes == {role: digest(record["path"]) for role, record in state.manifest["files"].items()}


@pytest.mark.parametrize("contaminated", ["fit", "replay"])
def test_test_metric_contamination_is_refused(experiment, monkeypatch, contaminated):
    state = experiment
    name = "fit_inductive_gcn" if contaminated == "fit" else "replay_routes"
    original = getattr(pilot, name)

    def contaminated_result(*args, **kwargs):
        return dict(original(*args, **kwargs), test_acc=100.)

    monkeypatch.setattr(pilot, name, contaminated_result)
    report, _ = state.call()
    assert report["status"] == "failed"
    assert "test metrics" in report["reason"] or "route replay differs" in report["reason"]


def test_changed_implementation_gets_separate_cache_instead_of_reusing_old_student(experiment):
    state = experiment
    first, root = state.call(student_seeds=(7,))
    state.optimizer_calls.clear()
    state.implementation = "changed-source-for-test"
    second, different = state.call(student_seeds=(7,))
    assert first["status"] == second["status"] == "complete"
    assert different != root and len(state.optimizer_calls) == 4
