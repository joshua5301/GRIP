"""Training-label isolation, selected logits and exact resumability of GCN teachers."""
import pytest
import torch
import torch.nn.functional as F

import src.gcn_teacher as teacher
from src.evaluation import _forward
from src.models import GCN


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def graphs():
    generator = torch.Generator().manual_seed(814)
    x = torch.randn(12, 4, generator=generator)
    source = dict(x=x, y=torch.tensor([0, 1] * 4 + [999] * 4),
                  adj=(torch.eye(12) * 0.7 + torch.ones(12, 12) * 0.025).to_sparse_csr())
    train = torch.arange(12) < 8
    validation = dict(x=torch.randn(7, 4, generator=generator),
                      y=torch.tensor([0, 1, 1, 0, 999, 999, 999]),
                      adj=(torch.eye(7) * 0.6 + torch.ones(7, 7) * 0.05).to_sparse_csr())
    val_mask = torch.arange(7) < 4
    return source, train, (validation, val_mask)


def fit(graphs, folder, **kwargs):
    return teacher.fit_gcn_teacher(*graphs, folder, epochs=kwargs.pop("epochs", 4), seed=17, **kwargs)


def equal_states(left, right):
    assert left.keys() == right.keys()
    for key in left:
        torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)


def clone_inputs(value):
    # Sparse CSR tensors do not support Python deepcopy in this Torch version.
    if torch.is_tensor(value):
        return value.clone()
    if isinstance(value, dict):
        return {key: clone_inputs(item) for key, item in value.items()}
    return tuple(clone_inputs(item) for item in value)


@pytest.mark.parametrize("layout", ["dense", "coo", "csr", "none"])
@pytest.mark.parametrize("training", [False, True])
def test_raw_logits_match_evaluator_packed_forward(layout, training):
    torch.manual_seed(73)
    model = GCN(4, 9, 3, 2, dropout=0.5).double()
    model.train(training)
    x = torch.randn(6, 4, dtype=torch.double)
    adjacency = torch.eye(6, dtype=torch.double) * 0.8 + torch.ones(6, 6, dtype=torch.double) / 30
    if layout == "coo":
        adjacency = adjacency.to_sparse()
    elif layout == "csr":
        adjacency = adjacency.to_sparse_csr()
    elif layout == "none":
        adjacency = None
    torch.manual_seed(44)
    actual = F.log_softmax(teacher._raw_forward(model, x, adjacency), dim=1)
    torch.manual_seed(44)
    expected = _forward(model, x, adjacency)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_only_known_labels_determine_classes_training_and_cache(tmp_path, graphs, monkeypatch):
    first = fit(graphs, tmp_path / "first")
    changed = clone_inputs(graphs)
    changed[0]["y"][~changed[1]] = -123456
    changed[2][0]["y"][~changed[2][1]] = 987654
    second = fit(changed, tmp_path / "second")
    assert first["recipe"]["class_vocabulary"] == [0, 1]
    assert first["logits"].shape == (12, 2)
    assert first["fingerprint"] == second["fingerprint"]
    assert first["history"] == second["history"]
    equal_states(first["selected_state"], second["selected_state"])
    torch.testing.assert_close(first["logits"], second["logits"], rtol=0, atol=0)
    before = (tmp_path / "first" / "teacher.pt").read_bytes()
    monkeypatch.setattr(torch.optim, "Adam", lambda *a, **k: pytest.fail("Cache hit created optimizer"))
    cached = fit(changed, tmp_path / "first")
    assert cached["fingerprint"] == first["fingerprint"]
    torch.testing.assert_close(cached["logits"], first["logits"], rtol=0, atol=0)
    assert (tmp_path / "first" / "teacher.pt").read_bytes() == before


def test_validation_uses_own_graph_mask_and_selected_raw_source_logits(tmp_path, graphs, monkeypatch):
    source, train, (validation, mask) = graphs
    original = teacher._raw_forward
    eval_inputs = []

    def recorded(model, x, adjacency):
        if not model.training:
            eval_inputs.append((x, adjacency))
        return original(model, x, adjacency)

    monkeypatch.setattr(teacher, "_raw_forward", recorded)
    result = fit(graphs, tmp_path)
    assert all(x is validation["x"] and adj is validation["adj"] for x, adj in eval_inputs[:-1])
    assert eval_inputs[-1][0] is source["x"] and eval_inputs[-1][1] is source["adj"]
    assert result["selected_validation"]["val_nodes"] == int(mask.sum())
    model = GCN(4, 256, 2, 2, dropout=0.5)
    model.load_state_dict(result["selected_state"])
    model.eval()
    with torch.no_grad():
        source_logits = original(model, source["x"], source["adj"])
        val_logits = original(model, validation["x"], validation["adj"])[mask]
    torch.testing.assert_close(result["logits"], source_logits, rtol=0, atol=0)
    assert result["selected_validation"]["val_acc"] == 100 * float(
        (val_logits.argmax(1) == validation["y"][mask]).double().mean())
    assert result["selected_validation"]["val_ce"] == float(F.cross_entropy(val_logits, validation["y"][mask]))
    assert "test" not in str(result.keys())
    assert not any("test" in key for row in result["history"] for key in row)


def test_first_maximum_epoch_keeps_its_weights_and_source_logits(tmp_path, graphs, monkeypatch):
    source, train, (validation, mask) = graphs
    validation["y"][mask] = 0
    original = teacher._raw_forward
    states = []

    def scores(model, x, adjacency):
        if not model.training and x is validation["x"]:
            states.append({key: value.detach().clone() for key, value in model.state_dict().items()})
            score = x.new_zeros(len(x), 2)
            score[:, 1 if len(states) == 1 else 0] = 1
            return score
        return original(model, x, adjacency)

    monkeypatch.setattr(teacher, "_raw_forward", scores)
    result = fit(graphs, tmp_path, epochs=3)
    assert [row["val_acc"] for row in result["history"]] == [0, 100, 100]
    assert result["selected_validation"]["epoch"] == 2
    equal_states(result["selected_state"], states[1])
    assert any(not torch.equal(states[1][key], states[2][key]) for key in states[1])
    model = GCN(4, 256, 2, 2, dropout=0.5)
    model.load_state_dict(states[1])
    model.eval()
    torch.testing.assert_close(result["logits"], original(model, source["x"], source["adj"]),
                               rtol=0, atol=0)


def test_epoch_checkpoint_resume_matches_uninterrupted_dropout_fit(tmp_path, graphs):
    reference = fit(graphs, tmp_path / "reference", epochs=5)
    interrupted = tmp_path / "interrupted"
    resume = interrupted / "teacher_resume.pt"

    def stop_after_saved_epoch():
        if resume.exists():
            saved = torch.load(resume, weights_only=False)
            if saved["epoch"] >= 1:
                raise InterruptedError("Test stopped after atomic first epoch")

    with pytest.raises(InterruptedError, match="atomic first epoch"):
        fit(graphs, interrupted, epochs=5, guard=stop_after_saved_epoch)
    assert not (interrupted / "teacher.pt").exists()
    assert torch.load(resume, weights_only=False)["epoch"] == 1
    resumed = fit(graphs, interrupted, epochs=5)
    assert resumed["history"] == reference["history"]
    assert resumed["selected_validation"] == reference["selected_validation"]
    assert resumed["fingerprint"] == reference["fingerprint"]
    equal_states(resumed["selected_state"], reference["selected_state"])
    torch.testing.assert_close(resumed["logits"], reference["logits"], rtol=0, atol=0)
    resumed_state = torch.load(resume, weights_only=False)
    reference_state = torch.load(tmp_path / "reference" / "teacher_resume.pt", weights_only=False)
    equal_states(resumed_state["model_state"], reference_state["model_state"])
    torch.testing.assert_close(resumed_state["rng"]["torch"], reference_state["rng"]["torch"],
                               rtol=0, atol=0)


@pytest.mark.parametrize("change", ["epochs", "source_x", "source_adj", "train_label", "train_mask",
                                   "val_x", "val_adj", "val_label", "val_mask", "recipe"])
def test_changed_inputs_or_requested_recipe_refuse_completed_cache(tmp_path, graphs, monkeypatch, change):
    fit(graphs, tmp_path)
    before = (tmp_path / "teacher.pt").read_bytes()
    source, train, (validation, mask) = clone_inputs(graphs)
    epochs = 4
    if change == "epochs":
        epochs = 5
    elif change == "source_x":
        source["x"][0, 0] += 0.25
    elif change == "source_adj":
        source["adj"].values()[0] += 0.25
    elif change == "train_label":
        source["y"][0] = 1
    elif change == "train_mask":
        train[0] = False
    elif change == "val_x":
        validation["x"][0, 0] += 0.25
    elif change == "val_adj":
        validation["adj"].values()[0] += 0.25
    elif change == "val_label":
        validation["y"][0] = 1
    elif change == "val_mask":
        mask[0] = False
    elif change == "recipe":
        settings = teacher.teacher_settings
        monkeypatch.setattr(teacher, "teacher_settings", lambda epochs, seed: dict(settings(epochs, seed), lr=0.02))
    with pytest.raises(ValueError, match="cache differs"):
        fit((source, train, (validation, mask)), tmp_path, epochs=epochs)
    assert (tmp_path / "teacher.pt").read_bytes() == before


@pytest.mark.parametrize("change", ["epochs", "source_x", "recipe"])
def test_changed_request_refuses_interrupted_resume(tmp_path, graphs, monkeypatch, change):
    resume = tmp_path / "teacher_resume.pt"

    def stop_after_saved_epoch():
        if resume.exists():
            raise InterruptedError("Saved epoch")

    with pytest.raises(InterruptedError):
        fit(graphs, tmp_path, guard=stop_after_saved_epoch)
    before = resume.read_bytes()
    changed = clone_inputs(graphs)
    epochs = 5 if change == "epochs" else 4
    if change == "source_x":
        changed[0]["x"][0, 0] += 0.25
    elif change == "recipe":
        settings = teacher.teacher_settings
        monkeypatch.setattr(teacher, "teacher_settings", lambda epochs, seed: dict(settings(epochs, seed), lr=0.02))
    with pytest.raises(ValueError, match="resume differs"):
        fit(changed, tmp_path, epochs=epochs)
    assert resume.read_bytes() == before


@pytest.mark.parametrize("epochs", [0, -1, True, 1.5])
def test_invalid_epochs_rejected_before_cache_creation(tmp_path, graphs, epochs):
    with pytest.raises(ValueError, match="epochs must be a positive integer"):
        fit(graphs, tmp_path, epochs=epochs)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("seed", [-1, True, 0.5])
def test_invalid_seed_rejected_before_cache_creation(tmp_path, graphs, seed):
    with pytest.raises(ValueError, match="seed must be a nonnegative integer"):
        teacher.fit_gcn_teacher(*graphs, tmp_path, epochs=1, seed=seed)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("invalid", ["missing_class", "negative_train", "empty_train", "nonboolean_train",
                                    "empty_validation", "nonboolean_validation", "unknown_validation"])
def test_invalid_known_training_and_validation_vocabulary_refused(tmp_path, graphs, invalid):
    source, train, (validation, mask) = clone_inputs(graphs)
    if invalid == "missing_class":
        source["y"][train] *= 2
    elif invalid == "negative_train":
        source["y"][0] = -1
    elif invalid == "empty_train":
        train[:] = False
    elif invalid == "nonboolean_train":
        train = train.long()
    elif invalid == "empty_validation":
        mask[:] = False
    elif invalid == "nonboolean_validation":
        mask = mask.long()
    elif invalid == "unknown_validation":
        validation["y"][0] = 2
    with pytest.raises(ValueError):
        fit((source, train, (validation, mask)), tmp_path)
    assert not list(tmp_path.iterdir())
