"""Completion proofs survive native CUDA tensors being serialized onto CPU."""
import copy

import pytest
import torch
import torch.nn.functional as F

import src.citation_gcn_teacher as helper
from src.gcn_teacher import _raw_forward
from src.io import _fingerprint, cpu_state
from src.models import GCN


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_native_to_saved_certificate_proof_is_identical(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.set_num_threads(4)
    x = torch.arange(1, 33, dtype=torch.float32).reshape(8, 4) / 30
    graph = dict(x=x, adj=torch.eye(8).to_sparse_csr(), y=torch.tensor([0, 1] * 4))
    train = torch.arange(8) < 4
    val = ~train
    with torch.random.fork_rng(devices=[]):
        model = GCN(4, 256, 2, 2, .5).eval()
    with torch.no_grad():
        for value in model.parameters():
            value.zero_()
        model.layers[0].lin.weight.fill_(.02)
        model.layers[1].lin.weight[0].fill_(.03)
        model.layers[1].lin.weight[1].fill_(-.04)
        model.layers[1].bias.copy_(torch.tensor([.1, -.1]))
        logits = _raw_forward(model, x, graph["adj"])
    metric = dict(epoch=1, val_acc=50., val_ce=float(F.cross_entropy(logits[val], graph["y"][val])), val_nodes=4)
    recipe = helper._teacher_recipe(graph, train, (graph, val))
    raw = dict(training_complete=True, recipe=recipe, fingerprint=_fingerprint(recipe), logits=logits,
               selected_state=cpu_state(model.state_dict()), selected_validation=metric,
               history=[dict(metric, epoch=i) for i in range(1, 201)],
               timings=dict(training_seconds=0., validation_seconds=0., source_logits_seconds=0.))
    context = dict(classes=2, temperature=.3)
    saved_proof = helper._raw_validate(raw, graph, train, (graph, val), context)
    native_graph = {k: v.to(device) for k, v in graph.items()}
    native_raw = copy.deepcopy(raw)
    native_raw["logits"] = raw["logits"].to(device)
    native_proof = helper._raw_validate(native_raw, native_graph, train.to(device),
                                        (native_graph, val.to(device)), context)
    assert native_proof == saved_proof
    assert raw["logits"].dtype == torch.float32 and raw["logits"].device.type == "cpu"
    assert helper.recipe()["derived_probability_digest_device"] == "cpu"
