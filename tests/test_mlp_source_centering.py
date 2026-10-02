import copy
import json

import pytest
import torch

from src import citation_search as citation
from src import mlp_source_centering as helper
from src import soft_ce_partition as core
from src.io import _fingerprint, save_json, save_state
from src.low_rank_assignment import LowRankMoments, assignment_inputs, encode_nodes
from src.moments import make_material
from src.transforms import fit_transform


def problem():
    generator = torch.Generator().manual_seed(42)
    z = torch.randn(12, 3, generator=generator, dtype=torch.float64)
    q = torch.randn(12, 2, generator=generator, dtype=torch.float64).softmax(1)
    return z, q, torch.tensor([0] * 6 + [1] * 3 + [2] * 3)


def options():
    return dict(penalty=.1, lr=.005, assignment_rank=2, assignment_input="features",
                assignment_encoder="mlp", encoder_hidden=5, mass_mode="free",
                inner_loss_weighting="uniform", save_resume=True, save_assignment=False,
                mlp_output_centering=helper.MODE, mlp_source_centering_schema=1,
                inner_tol=1e-9, inner_method="newton_first")


def load(folder):
    return torch.load(folder / "resume.pt", map_location="cpu", weights_only=False)


def reseal(state):
    state["source_centering_content_sha256"] = helper.seal({key: value for key, value in state.items()
                                                          if key != "source_centering_content_sha256"})
    return state


def expected(state, values):
    return helper.original_context(*values, 2, 5, 0, .05, 4096, state["config"],
                                   state["config"]["mlp_source_centering_source"])


def test_actual_native_core_original_p0_matches_free_encoder_and_float32_material(tmp_path):
    values = problem()
    core.optimize_ce_assignment(*values, steps=0, folder=tmp_path / "center", **options())
    state = load(tmp_path / "center")
    context, parameters = expected(state, values)
    z, q, hard = values
    u = encode_nodes(assignment_inputs(z, q, "features"), parameters[:-1])
    moments = LowRankMoments.apply(u, parameters[-1], hard, make_material(z, q), .05, 4096)
    snapshot = state["snapshots"][0]
    assert torch.equal(moments, snapshot["moments"])
    assert snapshot["raw_output_mean"].eq(0).all()
    helper.validate_resume(state, *values, context, parameters, 0, tmp_path / "center")
    free = {key: value for key, value in options().items() if not key.startswith("mlp_")}
    core.optimize_ce_assignment(*values, steps=0, checkpoint_steps=(0,), folder=tmp_path / "free", **free)
    actual_free = load(tmp_path / "free")["snapshots"][0]
    assert torch.equal(actual_free["moments"], snapshot["moments"])
    assert torch.equal(actual_free["theta"], snapshot["theta"])
    assert not any(key.startswith("source_center") for key in actual_free)


def test_actual_one_update_then_24_resume_matches_25_and_preserves_rng_and_zero_bias(tmp_path):
    values = problem()
    torch.manual_seed(900)
    rng = torch.get_rng_state().clone()
    full, split = tmp_path / "full", tmp_path / "split"
    core.optimize_ce_assignment(*values, steps=25, checkpoint_steps=(0, 1, 25), folder=full, **options())
    core.optimize_ce_assignment(*values, steps=1, checkpoint_steps=(0, 1), folder=split, **options())
    one = load(split)
    zero = (split / "checkpoints/step_000000.pt").read_bytes()
    calls = 0
    def stop():
        nonlocal calls
        calls += 1
        return calls >= 4
    before = (split / "resume.pt").read_bytes()
    with pytest.raises(InterruptedError):
        core.optimize_ce_assignment(*values, steps=25, folder=split, resume_state=one, stop=stop, **options())
    assert (split / "resume.pt").read_bytes() == before
    core.optimize_ce_assignment(*values, steps=25, checkpoint_steps=(0, 25), folder=split, resume_state=one, **options())
    assert torch.equal(torch.get_rng_state(), rng)
    assert (split / "checkpoints/step_000000.pt").read_bytes() == zero
    a, b = load(full), load(split)
    assert set(b["snapshots"]) == {0, 1, 25}
    for key in ("moments", "theta", "raw_output_mean"):
        assert torch.equal(a["snapshots"][25][key], b["snapshots"][25][key])
    assert all(torch.equal(x, y) for x, y in zip(a["parameters"], b["parameters"], strict=True))
    assert b["parameters"][3].eq(0).all() and b["dual"] is None
    for key in ("exp_avg", "exp_avg_sq"):
        assert b["optimizer"]["state"][3][key].eq(0).all()
    for value in b["optimizer"]["state"].values():
        assert value["step"].ndim == 0 and value["step"].dtype == torch.float32
        assert value["step"].device.type == "cpu" and float(value["step"]) == 25
    context, parameters = expected(b, values)
    helper.validate_resume(b, *values, context, parameters, 25, split)


@pytest.mark.parametrize("mutation", ["external_target", "external_mean", "source", "initializer", "device", "schema", "mode", "dual", "b2", "parameters_none", "snapshot_none", "snapshots_list", "mean", "u_digest", "moments", "head", "checksum", "history", "scale", "adam_lr", "bool_wd", "bool_id", "bool_counter", "int_counter", "double_counter", "vector_counter", "negative_counter", "nan_counter", "adam_missing", "adam_nan", "adam_negative", "b2_avg", "b2_sq"])
def test_resealed_cache_corruption_rejects_before_parameter_copy_fit_or_output(tmp_path, monkeypatch, mutation):
    values = problem()
    folder = tmp_path / "valid"
    core.optimize_ce_assignment(*values, steps=1, folder=folder, **options())
    state = copy.deepcopy(load(folder))
    if mutation in ("external_target", "external_mean"):
        state["mass_target" if mutation == "external_target" else "source_mean"] = torch.ones(3)
    elif mutation in ("source", "initializer", "device"):
        state["source_centering_context"][mutation] = "forged"
    elif mutation == "schema":
        state["mlp_source_centering_schema"] = True
    elif mutation == "mode":
        state["mlp_output_centering"] = "wrong"
    elif mutation == "dual":
        state["dual"] = torch.zeros(3, dtype=torch.float64)
    elif mutation == "b2":
        state["parameters"][3].fill_(1e-8)
    elif mutation == "parameters_none":
        state["parameters"] = None
    elif mutation == "snapshot_none":
        state["snapshots"][1] = None
    elif mutation == "snapshots_list":
        state["snapshots"] = []
    elif mutation in ("mean", "u_digest", "moments", "head"):
        s = state["snapshots"][1]
        if mutation == "mean":s["raw_output_mean"].add_(.01)
        elif mutation == "u_digest":s["centered_u_digest"] = {}
        elif mutation == "moments":s["moments"][0, 1] += .01
        else:s["theta"].add_(1)
        reseal(s)
    elif mutation in ("checksum", "scale"):
        state["scale"] += .1
    elif mutation == "history":
        state["history"].pop(0)
    else:
        group = state["optimizer"]["param_groups"][0]
        adam = state["optimizer"]["state"][0]
        if mutation == "adam_lr":group["lr"] = .7
        elif mutation == "bool_wd":group["weight_decay"] = False
        elif mutation == "bool_id":group["params"][0] = False
        elif mutation.endswith("counter"):
            adam["step"] = {"bool_counter":torch.tensor(True), "int_counter":torch.tensor(1),
                            "double_counter":torch.tensor(1., dtype=torch.float64), "vector_counter":torch.tensor([1.]),
                            "negative_counter":torch.tensor(-1.), "nan_counter":torch.tensor(float("nan"))}[mutation]
        elif mutation == "adam_missing":adam.pop("exp_avg")
        elif mutation == "adam_nan":adam["exp_avg"].fill_(float("nan"))
        elif mutation == "adam_negative":adam["exp_avg_sq"].fill_(-1)
        else:state["optimizer"]["state"][3]["exp_avg" if mutation == "b2_avg" else "exp_avg_sq"].fill_(1e-8)
    if mutation != "checksum":reseal(state)
    monkeypatch.setattr(core, "solve_inner_newton_first", lambda *a, **kw:pytest.fail("inner fit ran"))
    with pytest.raises(ValueError):
        core.optimize_ce_assignment(*values, steps=2, folder=tmp_path / "absent", resume_state=state, **options())
    assert not (tmp_path / "absent").exists()


@pytest.mark.parametrize("field", ["z", "q", "hard"])
def test_changed_actual_source_rejects_even_resealed_config(tmp_path, field):
    values = list(problem())
    core.optimize_ce_assignment(*values, steps=0, folder=tmp_path / "original", **options())
    state = load(tmp_path / "original")
    i = {"z":0,"q":1,"hard":2}[field]
    values[i] = values[i].clone()
    if field == "z":values[0][0, 0] += .1
    else:values[i] = values[i].roll(1, 0)
    with pytest.raises(ValueError):
        core.optimize_ce_assignment(*values, steps=1, folder=tmp_path / "absent", resume_state=state, **options())
    assert not (tmp_path / "absent").exists()


def candidate():
    return helper.candidate_controls(dict(method="mlp", width=5, rank=2, penalty=.1, lr=.005, T=.3,
                                          inner_loss_weighting="uniform", mlp_output_centering=helper.MODE,
                                          mlp_source_centering_schema=1))


@pytest.mark.parametrize("change", [dict(method="low_rank"), dict(mlp_output_centering="wrong"), dict(mlp_source_centering_schema=True), dict(mass_mode="initial"), dict(mass_mode="uniform"), dict(inner_loss_weighting="mass"), dict(train_target_mix=.5), dict(learn_temperature=True), dict(mixing=.1), dict(mass_target=[.5,.5]), dict(source_mean=[1]), dict(rank=True), dict(width=0), dict(penalty=float("nan")), dict(balance_steps=5000), dict(mlp_source_centering_source_digest="forged")])
def test_preload_rejects_invalid_unforwarded_controls_before_dataset(tmp_path, monkeypatch, change):
    monkeypatch.setattr(citation, "_prepare_dataset", lambda *a, **kw:pytest.fail("dataset loaded"))
    with pytest.raises(ValueError):citation.run_screen("cora", .013, tmp_path, [{**candidate(), **change}], device="cpu")


def source_fixture(tmp_path, monkeypatch):
    monkeypatch.setitem(citation.BUDGET, ("cora", .013), 3)
    generator = torch.Generator().manual_seed(88)
    x = torch.randn(12, 3, generator=generator)
    graph = dict(x=x, y=torch.arange(12) % 2, adj=torch.eye(12).to_sparse_csr())
    train = torch.arange(12) < 4
    val = (None, (torch.arange(12) >= 4) & (torch.arange(12) < 8))
    test = (None, torch.arange(12) >= 8)
    h = x.clone()
    monkeypatch.setattr(citation, "_prepare_dataset", lambda *a, **kw:(graph,train,val,test,h))
    config = citation._legacy_teacher_config("cora", .013, graph, train, val, test, "default")
    root = tmp_path / "cora/ratio_0.013" / _fingerprint(config)
    root.mkdir(parents=True)
    save_json(config, root / "config.json")
    citation.fixed_propagated_features(h, config, root)
    save_state(dict(logits=torch.randn(12,2,generator=generator,dtype=torch.float64),gamma=.01),root/"teacher.pt")
    z, transform = fit_transform(h.double())
    save_state(dict(z=z,assignment=problem()[2],transform=vars(transform)),root/"inputs_0.pt")
    return root


def test_actual_toy_probe_then_screen25_and_hot_probe_same_config_source_unchanged(tmp_path, monkeypatch):
    root = source_fixture(tmp_path, monkeypatch)
    source_bytes = {p:p.read_bytes() for p in root.iterdir() if p.is_file()}
    c = candidate()
    probe = helper.prepare_probe("cora", .013, tmp_path, c, device="cpu")
    assert probe["step"] == 1 and probe["student_fits"] == 0 and not probe["cached"]
    folder = root / _fingerprint(c) / "condensation_0"
    one = load(folder)
    monkeypatch.setattr(citation,"teacher_logits",lambda *a,**kw:pytest.fail("teacher fit"))
    monkeypatch.setattr(citation,"feature_kmeans",lambda *a,**kw:pytest.fail("kmeans fit"))
    calls = []
    monkeypatch.setattr(citation,"fit_gcn_diagnostic",lambda *a,**kw:calls.append(a[0]) or dict(val_acc=75.,val_ce=.5))
    ranking, actual = citation.run_screen("cora", .013, tmp_path, [c], steps=25, checkpoints=(0,25),
                                         student_seeds=(3200,),dropout=0,epochs=1,device="cpu")
    assert actual == root and len(ranking) == len(calls) == 2
    assert load(folder)["config"] == one["config"] and load(folder)["step"] == 25
    assert set(load(folder)["snapshots"]) == {0,1,25}
    assert all(p.read_bytes() == b for p,b in source_bytes.items())
    monkeypatch.setattr(core,"solve_inner_newton_first",lambda *a,**kw:pytest.fail("inner fit ran"))
    assert helper.prepare_probe("cora", .013, tmp_path, c, device="cpu")["cached"] is True


@pytest.mark.parametrize("tamper", ["teacher", "hard", "z", "config", "snapshot", "mean", "b2", "optimizer", "orphan", "candidate"])
def test_hot_screen_and_selected_guards_before_student_optimizer_or_selection_writes(tmp_path, monkeypatch, tamper):
    root = source_fixture(tmp_path, monkeypatch)
    c = candidate()
    helper.prepare_probe("cora", .013, tmp_path, c, device="cpu")
    folder = root / _fingerprint(c) / "condensation_0"
    if tamper == "candidate":save_json({**c,"lr":.9},folder.parent/"candidate.json")
    elif tamper == "teacher":
        state = torch.load(root/"teacher.pt",weights_only=False);state["logits"][0,0]+=.1;save_state(state,root/"teacher.pt")
    elif tamper in ("hard","z"):
        state = torch.load(root/"inputs_0.pt",weights_only=False)
        if tamper == "hard":state["assignment"] = state["assignment"].roll(1,0)
        else:state["z"][0,0]+=.1
        save_state(state,root/"inputs_0.pt")
    elif tamper == "config":
        state=json.loads((root/"config.json").read_text());state["data_digest"]="forged";save_json(state,root/"config.json")
    elif tamper == "orphan":(folder/"resume.pt").unlink()
    elif tamper == "snapshot":
        path=folder/"checkpoints/step_000001.pt";s=torch.load(path,weights_only=False);s["moments"][0,1]+=.01;save_state(reseal(s),path)
    else:
        state=load(folder)
        if tamper == "mean":state["snapshots"][1]["raw_output_mean"].add_(.1);reseal(state["snapshots"][1])
        elif tamper == "b2":state["parameters"][3].fill_(1e-8)
        else:state["optimizer"]["param_groups"][0]["lr"] = .9
        save_state(reseal(state),folder/"resume.pt")
    monkeypatch.setattr(citation,"fit_gcn_diagnostic",lambda *a,**kw:pytest.fail("student fit ran"))
    monkeypatch.setattr(citation,"optimize_ce_assignment",lambda *a,**kw:pytest.fail("optimizer ran"))
    before={p:p.read_bytes() for p in root.rglob('*') if p.is_file()}
    with pytest.raises(ValueError):citation.run_screen("cora",.013,tmp_path,[c],steps=1,student_seeds=(0,),device="cpu")
    assert before=={p:p.read_bytes() for p in root.rglob('*') if p.is_file()}
    with pytest.raises(ValueError):citation.selected_test(root,dict(c,step=1,candidate_path=str(folder.parent)),condensation_seeds=(0,),student_seeds=(0,),device="cpu")
    assert before=={p:p.read_bytes() for p in root.rglob('*') if p.is_file()}


def test_missing_mode_selected_row_and_stale_token_reject_before_data(tmp_path, monkeypatch):
    root=source_fixture(tmp_path,monkeypatch);c=candidate();p=root/_fingerprint(c);p.mkdir();save_json(c,p/'candidate.json')
    monkeypatch.setattr(citation,"_prepare_dataset",lambda *a,**kw:pytest.fail("dataset loaded"))
    raw={k:v for k,v in c.items() if k!='mlp_output_centering'}
    with pytest.raises(ValueError):citation.selected_test(root,dict(raw,step=1,candidate_path=str(p)),device='cpu')
    monkeypatch.setattr(helper,'source_digest',lambda:{'changed':'source'})
    with pytest.raises(ValueError,match='source changed'):citation._candidate_nystrom_mass(c)
    assert _fingerprint(helper.candidate_controls({k:v for k,v in c.items() if k!='mlp_source_centering_source_digest'}))!=_fingerprint(c)


def test_centered_dispatch_and_legacy_ids_unchanged(monkeypatch):
    from src import research_loop
    seen=[];stop=lambda:False
    monkeypatch.setattr(helper,'prepare_probe',lambda **kw:seen.append(kw) or {'step':1})
    assert research_loop.dispatch(dict(kind='citation_mlp_centered_probe',options=dict(dataset='cora')),stop)=={'step':1}
    assert seen==[dict(dataset='cora',stop=stop)]
    free=dict(method='mlp',width=256,lr=.05,T=.3,rank=32,penalty=1e-4,initialization='teacher_balanced',alpha=.3,inner_loss_weighting='uniform')
    assert citation._candidate_nystrom_mass(free)==free and _fingerprint(free)=='2755e0bc56ec'


def test_orphan_stop_and_non_native_dtype_fail_before_outputs(tmp_path):
    core.optimize_ce_assignment(*problem(),steps=0,folder=tmp_path/'old',**options())
    with pytest.raises(ValueError,match='verifiable resume'):core.optimize_ce_assignment(*problem(),steps=1,folder=tmp_path/'old',**options())
    with pytest.raises(InterruptedError):core.optimize_ce_assignment(*problem(),steps=0,folder=tmp_path/'absent',stop=lambda:True,**options())
    previous=torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        with pytest.raises(ValueError,match='Source centering'):core.optimize_ce_assignment(*problem(),steps=0,folder=tmp_path/'absent',**options())
    finally:torch.set_default_dtype(previous)
    assert not (tmp_path/'absent').exists()


@pytest.mark.parametrize('field', ['raw_output_mean', 'mlp_source_centering_source', 'mlp_source_centering_unknown', 'initialization', 'alpha'])
def test_extra_or_malformed_source_control_rejected_before_dataset(tmp_path, monkeypatch, field):
    values = {'raw_output_mean':[0], 'mlp_source_centering_source':{}, 'mlp_source_centering_unknown':1,
              'initialization':'forged', 'alpha':True}
    monkeypatch.setattr(citation,'_prepare_dataset',lambda *a,**kw:pytest.fail('dataset loaded'))
    with pytest.raises(ValueError):citation.run_screen('cora',.013,tmp_path,[{**candidate(),field:values[field]}],device='cpu')


@pytest.mark.parametrize('tamper', ['context_scalar_bool', 'diagnostic_scalar_bool', 'best_moments', 'best_head', 'adam_groups_none'])
def test_resealed_nested_native_schema_invalid_before_mutation(tmp_path, monkeypatch, tamper):
    values=problem();folder=tmp_path/'valid'
    core.optimize_ce_assignment(*values,steps=0,folder=folder,**options())
    state=load(folder)
    if tamper=='context_scalar_bool':state['source_centering_context']['original_diagnostic']['center_mean_residual']=False
    elif tamper=='diagnostic_scalar_bool':
        state['snapshots'][0]['center_diagnostic']['center_mean_residual']=False;reseal(state['snapshots'][0])
    elif tamper=='best_moments':state['best_moments'][0,1]+=.1
    elif tamper=='best_head':state['best_theta'].fill_(float('nan'))
    else:state['optimizer']['param_groups']=None
    reseal(state)
    monkeypatch.setattr(core,'solve_inner_newton_first',lambda *a,**kw:pytest.fail('fit ran'))
    with pytest.raises(ValueError):core.optimize_ce_assignment(*values,steps=1,folder=tmp_path/'absent',resume_state=state,**options())
    assert not (tmp_path/'absent').exists()


def test_valid_selected_preflight_reaches_generation_after_guard(tmp_path, monkeypatch):
    root=source_fixture(tmp_path,monkeypatch);c=candidate()
    helper.prepare_probe('cora',.013,tmp_path,c,device='cpu')
    folder=root/_fingerprint(c)
    def generated(*args,**kwargs):
        assert (root/'selected.json').exists()
        raise InterruptedError('completed guarded toy selection')
    monkeypatch.setattr(citation,'run_screen',generated)
    with pytest.raises(InterruptedError,match='guarded toy'):
        citation.selected_test(root,dict(c,step=1,candidate_path=str(folder)),condensation_seeds=(0,),student_seeds=(0,),device='cpu')


@pytest.mark.parametrize('tamper', ['coupled_J0_scale', 'coupled_J0_scale_snapshot', 'snapshot_outer_ce', 'snapshot_gradient', 'history_gradient', 'coupled_snapshot_history_gradient', 'bool_outer_ce', 'bool_gradient'])
def test_v2_actual_head_certificates_reject_coupled_resealed_resume_forging_before_fit_or_write(tmp_path, monkeypatch, tamper):
    values=problem();folder=tmp_path/'valid'
    core.optimize_ce_assignment(*values,steps=1,folder=folder,**options())
    state=load(folder)
    if tamper.startswith('coupled_J0'):
        state['history'][0]['J']+=.1
        state['scale']=max(state['history'][0]['J'],1e-12)
        if tamper.endswith('snapshot'):
            state['snapshots'][0]['teacher_ce']=state['history'][0]['J']
            reseal(state['snapshots'][0])
    elif tamper=='snapshot_outer_ce':
        state['snapshots'][1]['teacher_ce']+=.1;reseal(state['snapshots'][1])
    elif tamper=='snapshot_gradient':
        state['snapshots'][0]['inner_grad_max']+=.01;reseal(state['snapshots'][0])
    elif tamper=='history_gradient':
        state['history'][0]['inner_grad_max']*=.5
    elif tamper=='coupled_snapshot_history_gradient':
        state['snapshots'][0]['inner_grad_max']+=.01
        state['history'][0]['inner_grad_max']=state['snapshots'][0]['inner_grad_max']
        reseal(state['snapshots'][0])
    else:
        state['snapshots'][0]['teacher_ce' if tamper=='bool_outer_ce' else 'inner_grad_max']=False
        reseal(state['snapshots'][0])
    reseal(state)
    existing={p:p.read_bytes() for p in folder.rglob('*') if p.is_file()}
    monkeypatch.setattr(core,'solve_inner_newton_first',lambda *a,**kw:pytest.fail('head fit ran'))
    with pytest.raises(ValueError):
        core.optimize_ce_assignment(*values,steps=2,folder=tmp_path/'absent',resume_state=state,**options())
    assert not (tmp_path/'absent').exists()
    assert existing=={p:p.read_bytes() for p in folder.rglob('*') if p.is_file()}
