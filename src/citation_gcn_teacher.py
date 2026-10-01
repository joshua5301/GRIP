"""Citation GCN-Q with an explicitly pinned, original ReLU hard partition.

The teacher retains native float32 logits. Only this backend converts them to
float64 *before* the temperature softmax. No synthetic features/targets/edges,
new initializer, or new feature maps are introduced.
"""
import hashlib
import inspect
import json
import math
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.gcn_teacher import _inputs, _raw_forward, fit_gcn_teacher, teacher_settings
from src.inductive_evaluation import _update_tensor_digest
from src.io import _fingerprint, array_digest, cpu_state, save_json, save_state
from src.low_rank_assignment import LowRankMoments, initialize_factors
from src.models import GCN
from src.moments import make_material
from src.transforms import FeatureTransform

BACKEND = "gcn_fixed_relu_partition_v1"
LOGIT_ATOL, LOGIT_RTOL = 2e-5, 1e-5
CE_ATOL, CE_RTOL = 2e-5, 1e-5
# Historical coordinates came from native FP32 SGC. Asset hashes remain exact.
RMS_ATOL, RMS_RTOL = 1e-7, 1e-7
Q_FORMATION = "float64_softmax_from_native_fp32_GCN_raw_logits"
PIN_KEYS = {"policy", "source_root", "source_config_sha256", "source_H_sha256", "source_inputs_path",
            "source_inputs_sha256", "source_ReLU_teacher_sha256", "source_hard_assignment_path",
            "source_hard_assignment_sha256", "origin_initialization", "origin_alpha", "origin_T",
            "origin_condensation_seed", "origin_candidate_id", "origin_candidate_sha256"}
DEPENDENCIES = ("gcn_teacher.py", "models.py", "data.py", "transforms.py", "shared_features.py",
                "soft_ce_partition.py", "low_rank_assignment.py", "moments.py", "head.py", "io.py",
                "inductive_evaluation.py", "citation_search.py")


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def recipe():
    return dict(backend=BACKEND, epochs=200, seed=0, settings=teacher_settings(200, 0),
                helper_sha256=file_digest(__file__),
                dependencies={name: file_digest(Path(__file__).with_name(name)) for name in DEPENDENCIES},
                torch_version=str(torch.__version__), q_formation=Q_FORMATION,
                logit_replay=dict(atol=LOGIT_ATOL, rtol=LOGIT_RTOL),
                val_ce_replay=dict(atol=CE_ATOL, rtol=CE_RTOL),
                source_rms_replay=dict(atol=RMS_ATOL, rtol=RMS_RTOL),
                history_selection="first_strict_maximum_validation_accuracy_no_CE_tie")


def controls(backend, source, candidates, teacher_kernel="relu", seed=None):
    """Reject new/unsupported controls before data or cache mutations."""
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError("GCN candidates must be mappings")
        if any(k in candidate for k in ("teacher_backend", "initialization_source", "teacher_type",
                                       "initialization_policy", "gcn_teacher_epochs", "gcn_teacher_seed")):
            raise ValueError("GCN controls are top-level only")
    if backend is None:
        if source is not None:
            raise ValueError("initialization_source requires the GCN fixed-ReLU backend")
        return
    if backend != BACKEND or teacher_kernel != "relu":
        raise ValueError("Unsupported or conflicting citation teacher backend")
    if not isinstance(source, dict) or set(source) != PIN_KEYS or source.get("policy") != "frozen_relu":
        raise ValueError("GCN backend requires a complete frozen-ReLU source pin")
    if (type(source["origin_condensation_seed"]) is not int or source["origin_condensation_seed"] < 0
            or seed is not None and seed != source["origin_condensation_seed"]):
        raise ValueError("Condensation seed differs from frozen-ReLU source pin")
    for key in ("origin_alpha", "origin_T"):
        value = source[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("Frozen initializer controls must be finite positive scalars")
    if source["origin_initialization"] not in ("teacher_joint", "teacher_balanced"):
        raise ValueError("Only existing teacher-aware ReLU partitions are supported")
    for key in ("source_root", "source_inputs_path", "source_hard_assignment_path"):
        if not isinstance(source[key], str) or not source[key].strip():
            raise ValueError("Frozen source paths must be nonempty strings")
    if (not isinstance(source["origin_candidate_id"], str) or len(source["origin_candidate_id"]) != 12
            or any(v not in "0123456789abcdef" for v in source["origin_candidate_id"])):
        raise ValueError("Frozen source candidate identity must be canonical")
    for key in PIN_KEYS:
        if key.endswith("sha256") and (not isinstance(source[key], str) or len(source[key]) != 64
                                      or any(v not in "0123456789abcdef" for v in source[key])):
            raise ValueError("Frozen ReLU source digests must be canonical SHA256 strings")
    for candidate in candidates:
        if (candidate.get("method", "low_rank") != "low_rank"
                or candidate.get("inner_loss_weighting") != "uniform"
                or candidate.get("mass_mode", "free") != "free"
                or candidate.get("mixing", .05) != .05
                or candidate.get("train_target_mix", 0) != 0
                or candidate.get("learn_temperature", False) is not False
                or candidate.get("surrogate_kernel") is not None
                or candidate.get("node_weighting", False) is not False
                or candidate.get("assignment_input", "node") != "node"
                or candidate.get("assignment_encoder", "linear") != "linear"
                or candidate.get("solver_mode", "exact") != "exact"):
            raise ValueError("GCN pilot supports fixed-T, unweighted, free low_rank uniform CE only")
        if (candidate.get("initialization") != source["origin_initialization"]
                or candidate.get("alpha", 1) != source["origin_alpha"]
                or candidate.get("T") != source["origin_T"]):
            raise ValueError("GCN candidate differs from pinned ReLU initializer")


def _load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise ValueError("Invalid GCN cache; preserve it rather than overwriting") from error


def _finite(tensor, shape=None, dtype=None):
    return (torch.is_tensor(tensor) and (shape is None or tuple(tensor.shape) == tuple(shape))
            and (dtype is None or tensor.dtype == dtype) and bool(torch.isfinite(tensor).all()))


def _digest(value):
    digest = hashlib.sha256()
    def visit(name, item):
        if torch.is_tensor(item):
            _update_tensor_digest(digest, name, item)
        elif isinstance(item, dict):
            for key in sorted(item):
                visit(name + "." + str(key), item[key])
        elif isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                visit(name + "." + str(index), child)
        else:
            digest.update((name + ":" + repr(item)).encode())
    visit("value", value)
    return digest.hexdigest()


def teacher_inputs(graph, train, validation, guard=lambda: None):
    if (not _finite(graph.get("x")) or graph["x"].ndim != 2 or graph["x"].dtype != torch.float32
            or not torch.is_tensor(graph.get("adj")) or graph["adj"].layout != torch.sparse_csr
            or graph["adj"].dtype != torch.float32 or graph["adj"].shape != (len(graph["x"]), len(graph["x"]))
            or not torch.is_tensor(graph.get("y")) or graph["y"].dtype != torch.long
            or graph["y"].shape != (len(graph["x"]),)
            or not bool(torch.isfinite(graph["adj"].values()).all())):
        raise ValueError("GCN citation teacher requires finite native float32 X and packed adjacency")
    if (not torch.is_tensor(train) or train.dtype != torch.bool or train.shape != (len(graph["x"]),)
            or not isinstance(validation, tuple) or len(validation) != 2 or validation[0] is not graph
            or not torch.is_tensor(validation[1]) or validation[1].dtype != torch.bool
            or validation[1].shape != train.shape or bool((train & validation[1]).any())):
        raise ValueError("GCN citation teacher requires disjoint same-graph train/validation masks")
    return _inputs(graph, train, validation, guard)


def _source(h, base_config, source_root, pin, candidate=None):
    from src.citation_search import fixed_propagated_features
    source_root = Path(source_root).resolve()
    if Path(pin["source_root"]).resolve() != source_root:
        raise ValueError("ReLU pin source root differs from current graph geometry/configuration")
    if (source_root.name != _fingerprint(base_config)
            or json.loads((source_root / "config.json").read_text()) != base_config
            or base_config.get("teacher_backend") is not None or base_config.get("teacher_kernel") is not None):
        raise ValueError("Frozen initializer must originate in the expected legacy ReLU root")
    seed = pin["origin_condensation_seed"]
    init = dict(mode=pin["origin_initialization"], alpha=pin["origin_alpha"], T=pin["origin_T"], seed=seed)
    paths = dict(source_config_sha256=source_root / "config.json", source_H_sha256=source_root / "propagated_H.pt",
                 source_inputs_sha256=source_root / f"inputs_{seed}.pt", source_ReLU_teacher_sha256=source_root / "teacher.pt",
                 source_hard_assignment_sha256=source_root / f"assignment_{_fingerprint(init)}.pt",
                 origin_candidate_sha256=source_root / pin["origin_candidate_id"] / "candidate.json")
    for key, path in paths.items():
        if not path.exists() or file_digest(path) != pin[key]:
            raise ValueError("Frozen ReLU source asset digest differs: " + key)
    for field, key in (("source_inputs_path", "source_inputs_sha256"),
                       ("source_hard_assignment_path", "source_hard_assignment_sha256")):
        if Path(pin[field]).resolve() != paths[key]:
            raise ValueError("Arbitrary external initializer paths are not supported")
    origin = json.loads(paths["origin_candidate_sha256"].read_text())
    if not isinstance(origin, dict) or _fingerprint(origin) != pin["origin_candidate_id"]:
        raise ValueError("Origin candidate identity differs")
    controls(BACKEND, pin, [origin], seed=seed)
    if candidate is not None and origin != candidate:
        raise ValueError("GCN pilot candidate must exactly match its original ReLU candidate")
    frozen = fixed_propagated_features(h, base_config, source_root)
    inputs, assignment = _load(paths["source_inputs_sha256"]), _load(paths["source_hard_assignment_sha256"])
    from src.data import BUDGET
    cells = BUDGET[(base_config["dataset"], base_config["ratio"])]
    if (not isinstance(inputs, dict) or not _finite(inputs.get("z"), frozen.shape, torch.double) or not isinstance(inputs.get("transform"), dict)
            or not torch.is_tensor(assignment) or assignment.dtype != torch.long or assignment.shape != (len(frozen),)
            or not torch.equal(torch.unique(assignment), torch.arange(cells))):
        raise ValueError("Frozen ReLU source coordinates/partition shape or vocabulary differs")
    try:
        transform = FeatureTransform(**inputs["transform"])
        replayed = transform(frozen.detach().cpu().double())
    except Exception as error:
        raise ValueError("Frozen ReLU RMS transform is malformed") from error
    if not torch.allclose(replayed, inputs["z"], atol=RMS_ATOL, rtol=RMS_RTOL):
        raise ValueError("Source RMS coordinates do not match frozen H")
    return frozen, inputs, assignment, origin, paths


def context(graph, train, validation, h, base_config, source_root, pin, candidate=None, stop=lambda: False):
    guard = lambda: _stop(stop)
    labels, _, val_mask, val_labels, classes, input_digest = teacher_inputs(graph, train, validation, guard)
    frozen, inputs, assignment, origin, paths = _source(h, base_config, source_root, pin, candidate)
    ctx = dict(schema=1, backend=BACKEND, recipe=recipe(), teacher_input_digest=input_digest,
               classes=len(classes), class_vocabulary=classes.cpu().tolist(), source_pin=pin,
               source_candidate=origin, assets={str(p.resolve()): file_digest(p) for p in paths.values()},
               h_digest=_digest(frozen), z_digest=_digest(inputs["z"]), transform_digest=_digest(inputs["transform"]),
               assignment_digest=_digest(assignment), q_formation=Q_FORMATION, temperature=origin["T"], train_target_mix=0)
    config = dict(base_config, teacher_backend=BACKEND, gcn_context=ctx)
    return frozen, inputs, assignment.to(h.device), config


def teacher_root(output_dir, config):
    return Path(output_dir) / config["dataset"] / f"ratio_{config['ratio']}" / _fingerprint(config)


def _stop(stop):
    if stop():
        raise InterruptedError("GCN teacher/citation validation stopped")


def _history(state, epochs, complete):
    history = state.get("history")
    if not isinstance(history, list) or len(history) != epochs:
        raise ValueError("GCN teacher history is incomplete")
    best = None
    for index, row in enumerate(history, 1):
        if (not isinstance(row, dict) or type(row.get("epoch")) is not int or row["epoch"] != index
                or type(row.get("val_nodes")) is not int or row["val_nodes"] < 1
                or any(isinstance(row.get(k), bool) or not isinstance(row.get(k), (int, float))
                       or not math.isfinite(row[k]) for k in ("val_acc", "val_ce"))
                or not 0 <= row["val_acc"] <= 100 or row["val_ce"] < 0):
            raise ValueError("Malformed GCN teacher epoch history")
        if best is None or row["val_acc"] > best["val_acc"]:
            best = row
    selected = state.get("selected_validation" if complete else "best")
    if selected != best:
        raise ValueError("GCN teacher did not select first strict validation accuracy maximum")
    return best


def _teacher_recipe(graph, train, validation):
    _, _, _, _, classes, digest = teacher_inputs(graph, train, validation)
    return dict(version=1, settings=teacher_settings(200, 0), input_digest=digest,
                class_vocabulary=classes.cpu().tolist(), torch_version=str(torch.__version__))


def _model_state(state, graph, classes):
    if not isinstance(state, dict) or any(not _finite(value) for value in state.values()):
        raise ValueError("GCN selected/model weights must be finite tensors")
    with torch.random.fork_rng(devices=[]):
        model = GCN(graph["x"].shape[1], 256, classes, 2, .5).to(graph["x"].device)
    if set(state) != set(model.state_dict()) or any(
            value.dtype != model.state_dict()[key].dtype for key, value in state.items()):
        raise ValueError("GCN model state keys or native dtype differ")
    try:
        model.load_state_dict(state, strict=True)
    except Exception as error:
        raise ValueError("GCN selected/model state shape differs") from error
    model.eval()
    return model


def _raw_validate(state, graph, train, validation, ctx, stop=lambda: False):
    expected = _teacher_recipe(graph, train, validation)
    if not isinstance(state, dict):
        raise ValueError("Malformed GCN native teacher envelope")
    logits = state.get("logits")
    if (state.get("training_complete") is not True or state.get("recipe") != expected
            or state.get("fingerprint") != _fingerprint(expected)
            or not _finite(logits, (len(graph["x"]), ctx["classes"]), torch.float32)):
        raise ValueError("GCN teacher complete recipe/logit cache differs")
    best = _history(state, 200, True)
    timings = state.get("timings")
    if (not isinstance(timings, dict) or set(timings) != {"training_seconds", "validation_seconds", "source_logits_seconds"}
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0
                   for v in timings.values())):
        raise ValueError("GCN teacher timings must be finite nonnegative scalars")
    if any(row["val_nodes"] != int(validation[1].sum()) for row in state["history"]):
        raise ValueError("GCN history validation size differs")
    model = _model_state(state.get("selected_state"), graph, ctx["classes"])
    _stop(stop)
    with torch.no_grad():
        scores = _raw_forward(model, graph["x"], graph["adj"])
        validation_scores = _raw_forward(model, validation[0]["x"], validation[0]["adj"])[validation[1]]
    if not _finite(scores) or not torch.allclose(scores.cpu(), logits.cpu(), atol=LOGIT_ATOL, rtol=LOGIT_RTOL):
        raise ValueError("GCN selected weights do not replay native source logits")
    labels = validation[0]["y"][validation[1]]
    accuracy = 100 * float((validation_scores.argmax(1) == labels).double().mean())
    ce = float(F.cross_entropy(validation_scores, labels))
    if (accuracy != best["val_acc"] or best["val_nodes"] != len(labels)
            or not math.isclose(ce, best["val_ce"], abs_tol=CE_ATOL, rel_tol=CE_RTOL)):
        raise ValueError("GCN selected own-graph validation replay differs")
    _stop(stop)
    return dict(state_digest=_digest(state["selected_state"]), logits_digest=_digest(logits),
                history_digest=_digest(state["history"]), selected_validation=best,
                double_logits_digest=_digest(logits.double()),
                q_digest=_digest((logits.double() / ctx["temperature"]).softmax(1)))


def _resume_validate(state, graph, train, validation, ctx):
    expected = _teacher_recipe(graph, train, validation)
    if not isinstance(state, dict):
        raise ValueError("Malformed GCN partial teacher envelope")
    epoch = state.get("epoch")
    if (type(epoch) is not int or not 1 <= epoch <= 200 or state.get("recipe") != expected
            or state.get("fingerprint") != _fingerprint(expected)):
        raise ValueError("GCN teacher resume input/recipe/epoch differs")
    _history(state, epoch, False)
    model = _model_state(state.get("model_state"), graph, ctx["classes"])
    best_model = _model_state(state.get("best_state"), graph, ctx["classes"])
    with torch.no_grad():
        best_scores = _raw_forward(best_model, validation[0]["x"], validation[0]["adj"])[validation[1]]
    labels = validation[0]["y"][validation[1]]
    if (not _finite(best_scores) or 100 * float((best_scores.argmax(1) == labels).double().mean()) != state["best"]["val_acc"]
            or not math.isclose(float(F.cross_entropy(best_scores, labels)), state["best"]["val_ce"],
                                abs_tol=CE_ATOL, rel_tol=CE_RTOL)):
        raise ValueError("GCN resume best weights do not replay historical selected validation")
    if any(row["val_nodes"] != int(validation[1].sum()) for row in state["history"]):
        raise ValueError("GCN resume validation size differs")
    optimizer = state.get("optimizer")
    reference = torch.optim.Adam(model.parameters(), lr=.01, weight_decay=.0005).state_dict()
    if (not isinstance(optimizer, dict) or set(optimizer) != {"state", "param_groups"}
            or _digest(optimizer["param_groups"]) != _digest(cpu_state(reference["param_groups"]))
            or not isinstance(optimizer["state"], dict)
            or any(type(key) is not int for key in optimizer["state"])
            or set(optimizer["state"]) != set(reference["param_groups"][0]["params"])):
        raise ValueError("GCN resume Adam controls/parameter coverage differs")
    for index, parameter in enumerate(model.parameters()):
        value = optimizer["state"][index]
        if (not isinstance(value, dict) or set(value) != {"step", "exp_avg", "exp_avg_sq"}
                or not _finite(value.get("step"), (), torch.float32) or float(value["step"]) != epoch
                or not _finite(value.get("exp_avg"), parameter.shape, parameter.dtype)
                or not _finite(value.get("exp_avg_sq"), parameter.shape, parameter.dtype)
                or bool((value["exp_avg_sq"] < 0).any())):
            raise ValueError("GCN resume Adam state shape/step/finite values differ")
    rng = state.get("rng")
    if (not isinstance(rng, dict) or set(rng) != {"torch", "python", "numpy", "cuda"}
            or not torch.is_tensor(rng.get("torch")) or rng["torch"].dtype != torch.uint8
            or rng["torch"].shape != torch.get_rng_state().shape or not isinstance(rng.get("cuda"), list)
            or any(not torch.is_tensor(v) or v.dtype != torch.uint8 or v.ndim != 1 or len(v) < 1 for v in rng["cuda"])
            or graph["x"].device.type == "cuda" and len(rng["cuda"]) != torch.cuda.device_count()
            or graph["x"].device.type == "cpu" and rng["cuda"]):
        raise ValueError("GCN resume RNG state/device differs")
    try:
        torch.Generator().set_state(rng["torch"].cpu())
        python = rng["python"]
        random.Random().setstate((python[0], tuple(python[1]), python[2]))
        np.random.RandomState().set_state(tuple(rng["numpy"]))
    except Exception as error:
        raise ValueError("GCN resume Python/NumPy RNG state is malformed") from error
    timings = state.get("timings")
    if (not isinstance(timings, dict) or set(timings) != {"training_seconds", "validation_seconds"}
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0
                   for v in timings.values())):
        raise ValueError("GCN resume timings must be finite nonnegative scalars")


def _device_class(graph):
    return graph["x"].device.type


def _execution(root, graph, *, complete, create=False):
    path = Path(root) / "teacher_execution.json"
    if path.exists():
        state = json.loads(path.read_text())
        if (not isinstance(state, dict) or set(state) != {"schema", "fit_device_class", "dtype", "autocast", "tf32"}
                or type(state.get("schema")) is not int or state["schema"] != 1
                or state.get("fit_device_class") not in ("cpu", "cuda") or state.get("dtype") != "torch.float32"
                or state.get("autocast") is not False or state.get("tf32") is not False):
            raise ValueError("GCN native-fit execution metadata differs")
        if not complete and state["fit_device_class"] != _device_class(graph):
            raise ValueError("GCN partial teacher cross-device resume is unsupported")
        return state
    if (Path(root) / "gcn_teacher" / "teacher.pt").exists() or (Path(root) / "gcn_teacher" / "teacher_resume.pt").exists():
        raise ValueError("GCN teacher lacks native-fit device provenance")
    state = dict(schema=1, fit_device_class=_device_class(graph), dtype="torch.float32", autocast=False, tf32=False)
    if create:
        save_json(state, path)
    return state


def _binding(root, config, complete=True):
    root = Path(root)
    if root.name != _fingerprint(config):
        raise ValueError("GCN root identity differs")
    for name, value in (("config.json", config), ("source_binding.json", config["gcn_context"])):
        path = root / name
        if path.exists():
            if json.loads(path.read_text()) != value:
                raise ValueError("GCN root/source binding differs")
        elif complete:
            raise ValueError("GCN root binding is incomplete")
    for source, digest in config["gcn_context"]["assets"].items():
        if not Path(source).exists() or file_digest(source) != digest:
            raise ValueError("GCN frozen source asset changed")
    for name, source in (("propagated_H.pt", Path(config["gcn_context"]["source_pin"]["source_root"]) / "propagated_H.pt"),
                         (f"inputs_{config['gcn_context']['source_pin']['origin_condensation_seed']}.pt",
                          Path(config["gcn_context"]["source_pin"]["source_inputs_path"]))):
        path = root / name
        if path.exists() and file_digest(path) != file_digest(source) or complete and not path.exists():
            raise ValueError("GCN copied source-only coordinates differ or are absent")


def validate_root(root, graph, train, validation, h, *, stop=lambda: False, require_selected=True):
    """Current actual graph/pin replay is mandatory, even for cached endpoints."""
    root = Path(root)
    if torch.is_autocast_enabled() or torch.is_autocast_enabled("cpu"):
        raise ValueError("GCN native FP32 backend does not support external autocast")
    config = json.loads((root / "config.json").read_text())
    if not isinstance(config, dict) or config.get("teacher_backend") != BACKEND or not isinstance(config.get("gcn_context"), dict):
        raise ValueError("Not a fixed-ReLU-partition GCN citation root")
    base = {k: v for k, v in config.items() if k not in ("teacher_backend", "gcn_context")}
    frozen, inputs, assignment, expected = context(graph, train, validation, h, base,
        config["gcn_context"]["source_pin"]["source_root"], config["gcn_context"]["source_pin"], stop=stop)
    if config != expected:
        raise ValueError("GCN source graph, labels, geometry or helper context differs")
    _binding(root, config)
    raw_path = root / "gcn_teacher" / "teacher.pt"
    if not raw_path.exists():
        raise ValueError("GCN teacher must complete all 200 epochs before condensation")
    execution = _execution(root, graph, complete=True)
    raw = _load(raw_path)
    proof = _raw_validate(raw, graph, train, validation, config["gcn_context"], stop)
    path = root / "teacher.pt"
    if not path.exists():
        if require_selected:
            raise ValueError("GCN teacher completion certificate is absent")
        return raw, config, (frozen, inputs, assignment), proof
    selected = _load(path)
    expected_metadata = dict(schema=1, backend=BACKEND, context=config["gcn_context"],
                             native_teacher_sha256=file_digest(raw_path), native_fit_execution=execution, **proof)
    if (not isinstance(selected, dict) or type(selected.get("schema")) is not int
            or any(selected.get(k) != v for k, v in expected_metadata.items())
            or not torch.is_tensor(selected.get("logits")) or not torch.equal(selected["logits"], raw["logits"])):
        raise ValueError("GCN selected teacher completion/content certificate differs")
    return selected, config, (frozen, inputs, assignment), proof


def prepare_job(dataset, ratio, output_dir, initialization_source, data_dir="data", device="cuda",
                citation_features="default", teacher_recipe=None, stop=lambda: False):
    from src.citation_search import _legacy_teacher_config
    from src.data import BUDGET, _prepare_dataset
    controls(BACKEND, initialization_source, [])
    if torch.is_autocast_enabled() or torch.is_autocast_enabled("cpu"):
        raise ValueError("GCN native FP32 teacher does not support external autocast")
    if dataset not in ("cora", "citeseer") or (dataset, ratio) not in BUDGET:
        raise ValueError("Use a configured citation GCN budget")
    if teacher_recipe is not None and teacher_recipe != recipe():
        raise ValueError("Frozen GCN teacher recipe/source/Torch differs")
    _stop(stop)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    graph, train, validation, testing, h = _prepare_dataset(dataset, data_dir, device, citation_features)
    base = _legacy_teacher_config(dataset, ratio, graph, train, validation, testing, citation_features)
    source = Path(output_dir) / dataset / f"ratio_{ratio}" / _fingerprint(base)
    _, _, _, config = context(graph, train, validation, h, base, source, initialization_source, stop=stop)
    root = teacher_root(output_dir, config)
    _binding(root, config, complete=False)
    native = root / "gcn_teacher"
    _execution(root, graph, complete=(native / "teacher.pt").exists())
    if (native / "teacher.pt").exists():
        _raw_validate(_load(native / "teacher.pt"), graph, train, validation, config["gcn_context"], stop)
    elif (native / "teacher_resume.pt").exists():
        _resume_validate(_load(native / "teacher_resume.pt"), graph, train, validation, config["gcn_context"])
    elif (root / "teacher.pt").exists():
        raise ValueError("GCN selected teacher lacks complete native teacher")
    if (root / "teacher.pt").exists():
        validate_root(root, graph, train, validation, h, stop=stop)
    _stop(stop)
    root.mkdir(parents=True, exist_ok=True)
    for name, original in (("propagated_H.pt", source / "propagated_H.pt"),
                           (f"inputs_{initialization_source['origin_condensation_seed']}.pt", Path(initialization_source["source_inputs_path"]))):
        if not (root / name).exists():
            temporary = root / (name + ".copy.tmp")
            shutil.copyfile(original, temporary)
            if file_digest(temporary) != file_digest(original):
                raise ValueError("Source changed while copying GCN coordinates")
            temporary.replace(root / name)
    if not (root / "config.json").exists():
        save_json(config, root / "config.json")
    if not (root / "source_binding.json").exists():
        save_json(config["gcn_context"], root / "source_binding.json")
    execution = _execution(root, graph, complete=(native / "teacher.pt").exists(), create=True)
    cached = (native / "teacher.pt").exists()
    raw = fit_gcn_teacher(graph, train, validation, native, epochs=200, seed=0, guard=lambda: _stop(stop))
    proof = _raw_validate(raw, graph, train, validation, config["gcn_context"], stop)
    if not (root / "teacher.pt").exists():
        _stop(stop)
        save_state(dict(schema=1, backend=BACKEND, context=config["gcn_context"],
                        native_teacher_sha256=file_digest(native / "teacher.pt"), native_fit_execution=execution, logits=raw["logits"], **proof), root / "teacher.pt")
    validate_root(root, graph, train, validation, h, stop=stop)
    return dict(root=str(root.resolve()), teacher_complete=True, cached=cached,
                selected_epoch=raw["selected_validation"]["epoch"], validation_only=True)


def expected_resume_config(candidate, z, q, assignment, seed):
    """Reconstruct unchanged core defaults without calling an optimizer/head."""
    from src.soft_ce_partition import optimize_ce_assignment
    signature = inspect.signature(optimize_ce_assignment)
    values = {k: p.default for k, p in signature.parameters.items() if p.default is not inspect.Parameter.empty}
    values.update(penalty=candidate["penalty"], lr=candidate["lr"], mixing=.05,
                  assignment_rank=candidate["rank"], factor_seed=seed, assignment_input="node",
                  assignment_encoder="linear", encoder_hidden=64, solver_mode="exact", inner_method="newton_first",
                  implicit_warm_start=True, mass_mode="free", inner_loss_weighting="uniform", inner_max_iter=2000,
                  inner_tol=1e-7, cg_max_iter=512, cg_rtol=1e-6, cache_assignment=False)
    excluded = {"steps", "folder", "checkpoint_steps", "resume_state", "save_resume", "log_every",
                "initial_representatives", "outer_indices", "implicit_solver", "inner_solver", "temperature_logits",
                "outer_targets", "stop", "temperature_initial", "temperature_lr", "cg_check_interval", "cache_assignment",
                "node_weighting", "node_weight_penalty", "node_weight_lr"}
    result = {k: v for k, v in values.items() if k not in excluded}
    result["data_digest"] = array_digest(z.cpu().numpy(), q.cpu().numpy(), assignment.cpu().numpy())
    return result


def bind_condensation_inputs(root, candidate, seed, z, q, assignment, *, create=False):
    ctx = json.loads((Path(root) / "config.json").read_text())["gcn_context"]
    controls(BACKEND, ctx["source_pin"], [candidate], seed=seed)
    if candidate != ctx["source_candidate"]:
        raise ValueError("GCN candidate differs from source pin")
    proof = dict(schema=1, backend=BACKEND, context_digest=_digest(ctx), candidate=candidate, seed=seed,
                 z_digest=_digest(z), q_digest=_digest(q), assignment_digest=_digest(assignment),
                 factor_device_class=z.device.type, factor_dtype="torch.float32", q_dtype=str(q.dtype))
    path = Path(root) / _fingerprint(candidate) / f"condensation_{seed}" / "gcn_input_binding.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if not isinstance(saved, dict) or type(saved.get("schema")) is not int or saved != proof:
            raise ValueError("GCN cached input/source/teacher/factor-device binding differs")
    elif create:
        save_json(proof, path)
    return path.exists()


def validate_condensation(root, candidate, seed, z, q, assignment, steps, *, create=False):
    """Reject cached input/context/control/content mismatches before bypass."""
    ctx = json.loads((Path(root) / "config.json").read_text())["gcn_context"]
    if q.dtype != torch.double or not _finite(q) or not torch.allclose(q.sum(1), torch.ones_like(q[:, 0]), atol=1e-14, rtol=0):
        raise ValueError("GCN derived targets must be double probabilities")
    controls(BACKEND, ctx["source_pin"], [candidate], seed=seed)
    if candidate != ctx["source_candidate"]:
        raise ValueError("GCN candidate differs from frozen source controls")
    folder = Path(root) / _fingerprint(candidate) / f"condensation_{seed}"
    candidate_path = folder.parent / "candidate.json"
    if candidate_path.exists() and json.loads(candidate_path.read_text()) != candidate:
        raise ValueError("GCN recorded candidate identity differs")
    paths = list(sorted((folder / "checkpoints").glob("step_*.pt")))
    resume_path = folder / "resume.pt"
    binding_path = folder / "gcn_binding.json"
    input_bound = bind_condensation_inputs(root, candidate, seed, z, q, assignment)
    if not resume_path.exists():
        if paths or binding_path.exists():
            raise ValueError("GCN cached endpoints require a verifiable resume state")
        return
    if not input_bound:
        raise ValueError("GCN cached resume lacks its pre-optimization source input binding")
    state = _load(resume_path)
    if (not isinstance(state, dict) or type(state.get("step")) is not int or not 0 <= state["step"] <= 25
            or state.get("config") != expected_resume_config(candidate, z, q, assignment, seed)):
        raise ValueError("GCN condensation cached input/control/step context differs")
    cells = int(assignment.max()) + 1
    parameters = state.get("parameters")
    if (not isinstance(parameters, list) or len(parameters) != 2
            or not _finite(parameters[0], (len(z), candidate["rank"]), torch.float32)
            or not _finite(parameters[1], (cells, candidate["rank"]), torch.float32)):
        raise ValueError("GCN cached assignment factor shape/dtype/content differs")
    material = make_material(z, q)
    with torch.no_grad():
        u0, v0 = initialize_factors(assignment, cells, candidate["rank"], seed)
        initial = LowRankMoments.apply(u0, v0, assignment, material, .05, 4096)
        current = LowRankMoments.apply(parameters[0].to(z.device), parameters[1].to(z.device), assignment, material, .05, 4096)
    if not _finite(state.get("initial_moments"), initial.shape, torch.double) or not torch.allclose(
            state["initial_moments"].to(initial), initial, atol=1e-10, rtol=1e-10):
        raise ValueError("GCN cached P0 does not derive from pinned original partition")
    snapshots = state.get("snapshots")
    if not isinstance(snapshots, dict):
        raise ValueError("GCN resume lacks exact cached snapshots")
    if set(snapshots) != {int(path.stem.split("_")[1]) for path in paths} or not {0, state["step"]} <= set(snapshots):
        raise ValueError("GCN cached checkpoint coverage differs from resume")
    for path in paths:
        snapshot = _load(path)
        if not isinstance(snapshot, dict):
            raise ValueError("Malformed GCN snapshot envelope")
        step = snapshot.get("step")
        if (type(step) is not int or path.name != f"step_{step:06d}.pt" or step not in snapshots
                or _digest(snapshot) != _digest(snapshots[step])
                or not _finite(snapshot.get("moments"), initial.shape, torch.double)
                or not _finite(snapshot.get("theta"), (q.shape[1], z.shape[1] + 1), torch.double)
                or snapshot.get("J_exact") is not True
                or isinstance(snapshot.get("inner_grad_max"), bool) or not isinstance(snapshot.get("inner_grad_max"), (int, float))
                or not math.isfinite(snapshot["inner_grad_max"]) or not 0 <= snapshot["inner_grad_max"] <= 1e-7):
            raise ValueError("GCN cached snapshot content/head certificate differs")
        if (isinstance(snapshot.get("teacher_ce"), bool) or not isinstance(snapshot.get("teacher_ce"), (int, float))
                or not math.isfinite(snapshot["teacher_ce"])):
            raise ValueError("GCN cached outer loss must be finite")
        from src.moments import augmented, decode_moments
        from src.soft_ce_partition import head_gradient
        centers, targets, mass = decode_moments(snapshot["moments"].to(z), z.shape[1])
        residual = float(head_gradient(augmented(centers), targets, torch.full_like(mass, 1 / len(mass)),
                                      snapshot["theta"].to(z), candidate["penalty"]).abs().max())
        if not math.isfinite(residual) or residual > 1.01e-7:
            raise ValueError("GCN cached head stationarity replay failed")
        reference = initial if step == 0 else current if step == state["step"] else None
        if reference is not None and not torch.allclose(snapshot["moments"].to(reference), reference, atol=1e-10, rtol=1e-10):
            raise ValueError("GCN cached snapshot does not replay assignment factors")
    proof = dict(schema=1, backend=BACKEND, context_digest=_digest(ctx), candidate=candidate, seed=seed,
                 z_digest=_digest(z), q_digest=_digest(q), assignment_digest=_digest(assignment),
                 resume_sha256=file_digest(resume_path), snapshots={p.name: file_digest(p) for p in paths})
    if binding_path.exists() and json.loads(binding_path.read_text()) != proof:
        if not create:
            raise ValueError("GCN cached condensation provenance binding differs")
        # A validated resumed update may advance the existing binding; no cached
        # bypass reaches create=True before the previous binding was checked.
    if create:
        save_json(proof, binding_path)
