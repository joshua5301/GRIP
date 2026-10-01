"""NTK geometry changes the surrogate while preserving teacher Q/source prototypes."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from test_nystrom_balanced_mass import citation_mock as citation_mock
from test_nystrom_balanced_mass import cpu_threads as cpu_threads
from test_nystrom_balanced_mass import load, problem

import src.citation_search as search
import src.nystrom_ce as nys
import src.relu_ntk as ntk
from src.io import _fingerprint, save_state
from src.low_rank_assignment import LowRankMoments
from src.moments import augmented, decode_moments, make_material
from src.shared_features import get_shared_map
from src.soft_ce_partition import solve_head_system, solve_inner_newton_first


def files(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file()}


def fixture_map(h, root):
    get_shared_map(h, root / 'nystrom_map_schema3.pt', basis=3000, seed=0)
    return ntk.get_shared_ntk_map(h, root)


def test_exact_formula_symmetric_fixed_scale_psd_zero_and_diagonal(record_property):
    g = torch.Generator().manual_seed(78)
    x = torch.randn(9, 4, dtype=torch.double, generator=g)
    x = torch.cat([x, torch.zeros(1, 4, dtype=torch.double)])
    scale = 2.7
    kernel = ntk.kernel_values(x, x, scale)
    assert bool(torch.isfinite(kernel).all())
    torch.testing.assert_close(kernel, kernel.T, atol=1e-14, rtol=1e-14)
    eigenvalues = torch.linalg.eigvalsh((kernel + kernel.T)/2)
    assert float(eigenvalues.min()) >= -1e-12
    diagonal_error = float((kernel.diagonal() - x.square().sum(1)/scale).abs().max())
    record_property('theoretical_diagonal_max_abs', diagonal_error)
    # Norm/dot rounding can change acos near1 by O(sqrt(machine epsilon)).
    assert diagonal_error <= 3e-8 * float(x.square().sum(1).max()/scale)
    assert torch.equal(kernel[-1], torch.zeros_like(kernel[-1]))
    a, b = x[:3], x[3:8]
    torch.testing.assert_close(ntk.kernel_values(a, b, scale), ntk.kernel_values(b, a, scale).T,
                               atol=1e-14, rtol=1e-14)
    orthogonal = ntk.kernel_values(torch.tensor([[1.,0.]],dtype=torch.double), torch.tensor([[0.,2.]],dtype=torch.double),scale)
    torch.testing.assert_close(orthogonal, torch.tensor([[1/(math.pi*scale)]], dtype=torch.double),atol=1e-14,rtol=0)


@pytest.mark.parametrize('delta', [0., 1e-7, 1e-4])
def test_collinear_near_collinear_antipodal_forward_and_explicit_gradient_policy(delta):
    a = torch.tensor([[1., delta, 0.]],dtype=torch.double)
    b = torch.tensor([[1.,0.,0.],[-1.,0.,0.],[0.,0.,0.]],dtype=torch.double)
    values = ntk.kernel_values(a,b,1.)
    assert bool(torch.isfinite(values).all()) and float(values[0,2]) == 0
    if delta < 1e-6:
        with pytest.raises(FloatingPointError,match='angular'):
            ntk.kernel_values(a.requires_grad_(),b,1.)
    else:
        ntk.kernel_values(a.requires_grad_(),b,1.).sum().backward()
        assert bool(torch.isfinite(a.grad).all())


def test_zero_tiny_query_and_all_zero_anchor_failure_contract(tmp_path):
    zero = torch.zeros(2,3,dtype=torch.double)
    anchors = torch.ones(2,3,dtype=torch.double)
    assert torch.equal(ntk.kernel_values(zero,anchors,1.),torch.zeros(2,2,dtype=torch.double))
    with pytest.raises(FloatingPointError,match='tiny'):
        ntk.kernel_values(zero.requires_grad_(),anchors,1.)
    with pytest.raises(ValueError,match='scale'):
        ntk.kernel_values(anchors,anchors,0.)
    h = torch.zeros(3,3,dtype=torch.double)
    # A valid reference-map envelope can contain all-zero anchors; NTK refuses
    # its zero bandwidth before writing its own map cache.
    get_shared_map(torch.eye(3,dtype=torch.double),tmp_path/'nystrom_map_schema3.pt',basis=3000,seed=0)
    state = load(tmp_path/'nystrom_map_schema3.pt')
    from src.shared_features import _map_identity, _tensor_identity
    state.update(identity=_map_identity(h,3000,0,'relu'),anchors=h,anchors_identity=_tensor_identity(h))
    save_state(state,tmp_path/'nystrom_map_schema3.pt')
    before=files(tmp_path)
    with pytest.raises(ValueError,match='energy'):
        ntk.get_shared_ntk_map(h,tmp_path)
    assert files(tmp_path)==before


def test_finite_width_actual_autograd_jacobian_and_large_mc_agree(record_property):
    g=torch.Generator().manual_seed(198)
    x=torch.randn(6,4,dtype=torch.double,generator=g)
    scale=2.3
    neurons=32
    w=torch.randn(neurons,4,dtype=torch.double,generator=g,requires_grad=True)
    a=torch.randn(neurons,dtype=torch.double,generator=g,requires_grad=True)
    outputs=(x@w.T/math.sqrt(scale)).relu()@a/math.sqrt(neurons)
    jac=[]
    for i in range(len(x)):
        gradients=torch.autograd.grad(outputs[i],(a,w),retain_graph=True)
        jac.append(torch.cat([v.flatten() for v in gradients]))
    actual=torch.stack(jac)@torch.stack(jac).T
    hidden=(x@w.T/math.sqrt(scale)).relu().detach()
    mask=(hidden>0).double()
    analytic_finite=(hidden@hidden.T+(mask*a.detach().square())@mask.T*(x@x.T/scale))/neurons
    torch.testing.assert_close(actual,analytic_finite,atol=2e-14,rtol=2e-14)
    width=200000
    ww=torch.randn(width,4,dtype=torch.double,generator=g)
    aa=torch.randn(width,dtype=torch.double,generator=g)
    activation=(x@ww.T/math.sqrt(scale)).relu()
    mask=(activation>0).double()
    empirical=(activation@activation.T+(mask*aa.square())@mask.T*(x@x.T/scale))/width
    theory=ntk.kernel_values(x,x,scale)
    error=float((empirical-theory).abs().max()/theory.diagonal().max())
    record_property('wide_full_jacobian_normalized_max_abs',error)
    assert error<.015


def fit_objective(moments,h,q,feature_map,phi):
    x,y,mass=decode_moments(moments,h.shape[1])
    weights=torch.full_like(mass,1/len(mass))
    mapped=feature_map(x).detach()
    inner=solve_inner_newton_first(mapped,y,weights,.1,grad_tol=1e-9)
    assert inner['inner_converged'] and inner['inner_grad_max']<=1e-9
    value,rhs=nys.outer_gradient(phi,q,inner['theta'],7)
    return value,rhs,inner['theta'],mapped,y,weights


@pytest.mark.parametrize('factor',['u','v','both'])
def test_full_nonlinear_ntk_centroid_ce_hypergradient_fd(tmp_path,factor,record_property):
    h,q,assignment,_=problem()
    feature_map=fixture_map(h,tmp_path)
    phi=feature_map(h).detach().numpy()
    g=torch.Generator().manual_seed(19)
    u=(.2*torch.randn(24,3,dtype=torch.double,generator=g)).requires_grad_()
    v=torch.randn(4,3,dtype=torch.double,generator=g).requires_grad_()
    du=.2*torch.randn(u.shape,dtype=u.dtype,generator=g)
    dv=.2*torch.randn(v.shape,dtype=v.dtype,generator=g)
    if factor=='u':dv.zero_()
    if factor=='v':du.zero_()
    material=make_material(h,q)
    moment=lambda uu,vv:LowRankMoments.apply(uu,vv,assignment,material,.05,7)
    moments=moment(u,v)
    _,rhs,theta,mapped,y,weights=fit_objective(moments.detach(),h,q,feature_map,phi)
    vector,diagnostic=solve_head_system(augmented(mapped),y,weights,theta,.1,rhs,rtol=1e-10)
    assert diagnostic['cg_converged']
    moments.backward(nys.moment_gradient(moments,3,feature_map,theta,vector,.1,'uniform'))
    analytic=float((u.grad*du).sum()+(v.grad*dv).sum())
    epsilon=.001
    plus=moment(u.detach()+epsilon*du,v.detach()+epsilon*dv)
    minus=moment(u.detach()-epsilon*du,v.detach()-epsilon*dv)
    finite=(fit_objective(plus,h,q,feature_map,phi)[0]-fit_objective(minus,h,q,feature_map,phi)[0])/(2*epsilon)
    error=abs(analytic-finite)
    record_property('ntk_head_fd_absolute_error',error)
    record_property('ntk_head_fd_relative_error',error/max(abs(analytic),abs(finite),1e-15))
    assert error<2e-8+2e-4*abs(finite)


def test_ntk_same_p0_source_prototypes_and_interrupted_resume(tmp_path):
    h,q,a,old=problem()
    root=tmp_path/'cache';root.mkdir()
    feature_map=fixture_map(h,root)
    old_phi=old(h).detach().numpy()
    ntk_phi=feature_map(h).detach().numpy()
    controls=dict(penalty=.1,rank=3,chunk=7,checkpoint_every=1,inner_loss_weighting='uniform')
    nys.optimize(h,q,a,old,old_phi,tmp_path/'old',0,**controls)
    nys.optimize(h,q,a,feature_map,ntk_phi,tmp_path/'ntk',2,**controls)
    assert torch.equal(load(tmp_path/'old/step_000000.pt')['moments'],load(tmp_path/'ntk/step_000000.pt')['moments'])
    stopped={'yes':False}
    def progress(row):stopped['yes']=row['step']==0
    with pytest.raises(InterruptedError):
        nys.optimize(h,q,a,feature_map,ntk_phi,tmp_path/'stopped',2,stop=lambda:stopped['yes'],progress=progress,**controls)
    nys.optimize(h,q,a,feature_map,ntk_phi,tmp_path/'stopped',2,**controls)
    reference=load(tmp_path/'ntk/step_000002.pt');actual=load(tmp_path/'stopped/step_000002.pt')
    torch.testing.assert_close(actual['moments'],reference['moments'],atol=1e-9,rtol=1e-7)
    torch.testing.assert_close(actual['theta'],reference['theta'],atol=1e-7,rtol=1e-5)


@pytest.mark.parametrize('change',['h','anchor_order','mapping','map_identity','source','phi','phi_meta'])
def test_map_phi_identity_content_mismatch_rejects_without_overwrite(tmp_path,monkeypatch,change):
    h,q,a,_=problem();fixture_map(h,tmp_path)
    feature_map,phi=ntk.prepare_features(h,tmp_path)
    map_path,phi_path=ntk.cache_paths(tmp_path)
    if change=='h':h=h+.01
    elif change=='anchor_order':
        p=tmp_path/'nystrom_map_schema3.pt';state=load(p)
        state['anchors']=state['anchors'].roll(1,0)
        from src.shared_features import _tensor_identity
        state['anchors_identity']=_tensor_identity(state['anchors']);save_state(state,p)
    elif change in ('mapping','map_identity'):
        state=load(map_path)
        if change=='mapping':state['mapping']+=.01
        else:state['identity']['scale']*=2
        save_state(state,map_path)
    elif change=='source':
        # Keep the old namespace pinned while simulating a changed helper source.
        monkeypatch.setattr(ntk,'source_digest',lambda:'a'*64)
        monkeypatch.setattr(ntk,'cache_paths',lambda root:(map_path,phi_path))
    elif change=='phi':
        data=np.load(phi_path).copy();data[0,0]+=.01;np.save(phi_path,data)
    else:phi_path.with_suffix('.meta.json').unlink()
    before=files(tmp_path)
    with pytest.raises(ValueError):ntk.prepare_features(h,tmp_path,require_existing=True)
    assert files(tmp_path)==before


def candidate(**extra):
    return dict(method='nystrom',width=0,lr=.01,T=1.,rank=3,penalty=.1,inner_loss_weighting='uniform',**extra)


def run(root,candidates):
    return search.run_screen('cora',.013,root,candidates,steps=1,student_seeds=(2100,),condensation_seed=0,epochs=1,dropout=0,device='cpu')


@pytest.mark.parametrize('extra',[dict(surrogate_kernel='bad'),dict(surrogate_kernel=ntk.KIND,mass_mode='initial'),
    dict(surrogate_kernel=ntk.KIND,mass_mode='uniform'),dict(surrogate_kernel=ntk.KIND,train_target_mix=.25),
    dict(surrogate_kernel=ntk.KIND,ntk_angle_guard=1e-8),dict(surrogate_kernel=ntk.KIND,surrogate_schema=2),
    dict(surrogate_kernel=ntk.KIND,surrogate_source_digest='a'*64),dict(surrogate_kernel=ntk.KIND,ntk_layers=2),
    dict(ntk_unknown=1),dict(kernel='relu2')])
def test_invalid_ntk_candidate_controls_reject_before_dataset(tmp_path,citation_mock,extra):
    with pytest.raises(ValueError):run(tmp_path,[candidate(**extra)])
    assert citation_mock['data_calls']==0 and not list(tmp_path.iterdir())


@pytest.mark.parametrize('method',['low_rank','mlp','distance','coarsening'])
def test_surrogate_control_unsupported_method_rejects_preload(tmp_path,citation_mock,method):
    option=candidate(surrogate_kernel=ntk.KIND);option['method']=method
    with pytest.raises(ValueError):run(tmp_path,[option])
    assert citation_mock['data_calls']==0 and not list(tmp_path.iterdir())


def paired_screen(tmp_path,monkeypatch):
    monkeypatch.setattr(search,'get_shared_map',get_shared_map)
    return run(tmp_path,[candidate(),candidate(surrogate_kernel=ntk.KIND)])


def test_conditional_citation_preserves_default_caches_identity_and_p0(tmp_path,citation_mock,monkeypatch):
    monkeypatch.setattr(search,'get_shared_map',get_shared_map)
    default,root=run(tmp_path,[candidate()])
    legacy={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'nystrom_map_schema3.pt',root/'nystrom_phi_schema3.npy',root/'nystrom_phi_schema3.meta.json')}
    default_identity=json.loads((Path(default.iloc[0].candidate_path)/'candidate.json').read_text())
    explicit,same=run(tmp_path,[candidate(surrogate_kernel='relu')])
    assert root==same and explicit.candidate_path.tolist()==default.candidate_path.tolist()
    ranking,_=run(tmp_path,[candidate(surrogate_kernel=ntk.KIND)])
    choice=ranking[ranking.step==1].iloc[0].to_dict()
    folder=Path(choice['candidate_path'])/'condensation_0'
    new_identity=json.loads((folder.parent/'candidate.json').read_text())
    assert _fingerprint(new_identity)==folder.parent.name and new_identity['surrogate_source_digest']==ntk.source_digest()
    assert 'surrogate_kernel' not in default_identity
    assert legacy=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'nystrom_map_schema3.pt',root/'nystrom_phi_schema3.npy',root/'nystrom_phi_schema3.meta.json')}
    original=load(Path(default.iloc[0].candidate_path)/'condensation_0/step_000000.pt')
    assert torch.equal(original['moments'],load(folder/'step_000000.pt')['moments'])
    before=len(citation_mock['optimizer_kwargs']);run(tmp_path,[candidate(surrogate_kernel=ntk.KIND)])
    assert len(citation_mock['optimizer_kwargs'])==before


@pytest.mark.parametrize('change',['q','h','assignment','theta_fingerprint','resume_config','phi','anchor_order'])
def test_ntk_cached_optimizer_bypass_guard_before_new_fit(tmp_path,citation_mock,monkeypatch,change):
    ranking,root=paired_screen(tmp_path,monkeypatch)
    choice=ranking[(ranking.surrogate_kernel==ntk.KIND)&(ranking.step==1)].iloc[0].to_dict()
    folder=Path(choice['candidate_path'])/'condensation_0'
    if change=='q':
        path=root/'teacher.pt';state=load(path);state['logits']+=torch.linspace(-.1,.1,3);save_state(state,path)
    elif change=='h':
        path=root/'propagated_H.pt';state=load(path);state['h']+=.01;save_state(state,path)
    elif change=='assignment':
        path=root/'inputs_0.pt';state=load(path);state['assignment']=state['assignment'].roll(1);save_state(state,path)
    elif change=='theta_fingerprint':
        path=folder/'step_000001.pt';state=load(path);state['input_fingerprint']['map_digest']='a'*64;save_state(state,path)
    elif change=='resume_config':
        path=folder/'resume.pt';state=load(path);state['config']['lr']*=2;save_state(state,path)
    elif change=='phi':
        _,path=ntk.cache_paths(root);data=np.load(path).copy();data[0,0]+=.1;np.save(path,data)
    else:
        path=root/'nystrom_map_schema3.pt';state=load(path);state['anchors']=state['anchors'].roll(1,0)
        from src.shared_features import _tensor_identity
        state['anchors_identity']=_tensor_identity(state['anchors']);save_state(state,path)
    before=len(citation_mock['fits'])
    # For q, the mock teacher would overwrite tampered logits on run_screen;
    # selected_test preflight still reads the actual frozen source first.
    with pytest.raises(ValueError):
        search.selected_test(root,choice,condensation_seeds=(0,),student_seeds=(2200,),device='cpu',report_routes=False)
    assert len(citation_mock['fits'])==before and not (root/'selected.json').exists()


@pytest.mark.parametrize('field',['surrogate_schema','surrogate_source_digest','ntk_angle_guard','ntk_norm_guard','ntk_jitter'])
def test_selected_ntk_requires_all_frozen_controls_before_mutation(tmp_path,citation_mock,monkeypatch,field):
    ranking,root=paired_screen(tmp_path,monkeypatch)
    choice=ranking[(ranking.surrogate_kernel==ntk.KIND)&(ranking.step==1)].iloc[0].to_dict();choice.pop(field)
    before=citation_mock['data_calls'];snapshot=files(tmp_path)
    with pytest.raises(ValueError):search.selected_test(root,choice,condensation_seeds=(0,),student_seeds=(2200,),device='cpu',report_routes=False)
    assert citation_mock['data_calls']==before and files(tmp_path)==snapshot


def test_ntk_scale_is_calibrated_from_original_saved_cpu_anchor_bytes(tmp_path):
    h,_,_,_=problem();mapped=fixture_map(h,tmp_path)
    original=load(tmp_path/'nystrom_map_schema3.pt')['anchors']
    assert original.device.type=='cpu' and mapped.scale==float(original.square().sum(1).mean())
    path,_=ntk.cache_paths(tmp_path);saved=load(path)
    assert saved['identity']['scale_calibration']=='original_saved_anchors_cpu_float64_mean_squared_norm_v1'
    assert saved['identity']['scale']==mapped.scale
    assert ntk.get_shared_ntk_map(h,tmp_path,require_existing=True).scale==mapped.scale


@pytest.mark.parametrize('extra',[dict(kernel='relu2'),dict(teacher_kernel='relu_ntk'),dict(ntk_layers=2),
    dict(ntk_scale=2),dict(ntk_smoothing=1e-8),dict(ntk_unknown=0)])
def test_selected_raw_unsupported_kernel_controls_reject_before_whitelist(tmp_path,citation_mock,monkeypatch,extra):
    ranking,root=paired_screen(tmp_path,monkeypatch)
    choice=ranking[(ranking.surrogate_kernel==ntk.KIND)&(ranking.step==1)].iloc[0].to_dict();choice.update(extra)
    before=citation_mock['data_calls'];snapshot=files(tmp_path)
    with pytest.raises(ValueError):search.selected_test(root,choice,condensation_seeds=(0,),student_seeds=(2200,),device='cpu',report_routes=False)
    assert citation_mock['data_calls']==before and files(tmp_path)==snapshot


@pytest.mark.parametrize('change',['missing_kernel','nan_kernel','all_ntk_nan','wrong_path','recorded_tamper'])
def test_selected_recorded_ntk_identity_cannot_become_legacy_row(tmp_path,citation_mock,monkeypatch,change):
    ranking,root=paired_screen(tmp_path,monkeypatch)
    choice=ranking[(ranking.surrogate_kernel==ntk.KIND)&(ranking.step==1)].iloc[0].to_dict()
    if change=='missing_kernel':choice.pop('surrogate_kernel')
    elif change=='nan_kernel':choice['surrogate_kernel']=float('nan')
    elif change=='all_ntk_nan':
        for key in ('surrogate_kernel','surrogate_schema','surrogate_source_digest','ntk_angle_guard','ntk_norm_guard','ntk_jitter'):choice[key]=float('nan')
    elif change=='wrong_path':choice['candidate_path']=ranking[ranking.surrogate_kernel.isna()].iloc[0].candidate_path
    else:
        path=Path(choice['candidate_path'])/'candidate.json';identity=json.loads(path.read_text());identity['surrogate_kernel']='relu';path.write_text(json.dumps(identity))
    before=citation_mock['data_calls'];snapshot=files(tmp_path)
    with pytest.raises(ValueError):search.selected_test(root,choice,condensation_seeds=(0,),student_seeds=(2200,),device='cpu',report_routes=False)
    assert citation_mock['data_calls']==before and files(tmp_path)==snapshot


@pytest.mark.parametrize('change',['moments_width','theta_shape','inner_weight','resume_u','resume_v','resume_theta','orphan'])
def test_selected_ntk_malformed_cached_content_rejects_before_mutation(tmp_path,citation_mock,monkeypatch,change):
    ranking,root=paired_screen(tmp_path,monkeypatch)
    choice=ranking[(ranking.surrogate_kernel==ntk.KIND)&(ranking.step==1)].iloc[0].to_dict();folder=Path(choice['candidate_path'])/'condensation_0'
    if change=='orphan':(folder/'resume.pt').unlink()
    else:
        resume=change.startswith('resume_');path=folder/('resume.pt' if resume else 'step_000001.pt');state=load(path)
        if change=='moments_width':state['moments']=state['moments'][:,:-1]
        elif change=='theta_shape':state['theta']=state['theta'][:-1]
        elif change=='inner_weight':state['inner_loss_weighting']='mass'
        else:state[change.removeprefix('resume_')]=state[change.removeprefix('resume_')][:-1]
        save_state(state,path)
    before=citation_mock['data_calls'];snapshot=files(tmp_path)
    with pytest.raises(ValueError):search.selected_test(root,choice,condensation_seeds=(0,),student_seeds=(2200,),device='cpu',report_routes=False)
    assert citation_mock['data_calls']==before and files(tmp_path)==snapshot


def test_legitimate_legacy_selection_from_mixed_ntk_ranking_keeps_identity(tmp_path,citation_mock,monkeypatch):
    ranking,root=paired_screen(tmp_path,monkeypatch)
    choice=ranking[ranking.surrogate_kernel.isna()&(ranking.step==1)].iloc[0].to_dict()
    search.selected_test(root,choice,condensation_seeds=(0,),student_seeds=(2200,),device='cpu',report_routes=False)
    selected=json.loads((root/'selected.json').read_text())
    assert 'surrogate_kernel' not in selected['candidate']
    assert _fingerprint(selected['candidate'])==Path(choice['candidate_path']).name


def test_cusp_failure_retains_value_head_but_wrapper_refuses_endpoint_acceptance(tmp_path):
    h=torch.tensor([[1.,0.],[2.,0.],[3.,0.],[4.,0.]],dtype=torch.double)
    q=torch.tensor([[.6,.4],[.5,.5],[.4,.6],[.3,.7]],dtype=torch.double)
    assignment=torch.zeros(4,dtype=torch.long)
    feature_map=fixture_map(h,tmp_path);feature_map,phi=ntk.prepare_features(h,tmp_path)
    option=ntk.candidate_controls(dict(method='nystrom',width=0,lr=.01,T=1.,rank=1,penalty=.1,
                                     nystrom_schema=3,inner_loss_weighting='uniform',surrogate_kernel=ntk.KIND))
    folder=tmp_path/_fingerprint(option)/'condensation_0'
    with pytest.raises(FloatingPointError,match='angular'):
        nys.optimize(h,q,assignment,feature_map,phi,folder,1,penalty=.1,lr=.01,rank=1,inner_loss_weighting='uniform')
    # Legacy free numerical core persists a converged value/head before differentiating.
    snapshot=load(folder/'step_000000.pt');resume=load(folder/'resume.pt')
    assert bool(torch.isfinite(snapshot['theta']).all()) and resume['step']==0
    assert torch.equal(resume['u'],torch.zeros_like(resume['u']))
    assert not (folder/'step_000001.pt').exists()
    before=files(tmp_path)
    with pytest.raises(FloatingPointError,match='angular'):
        search._check_nystrom_assignment(option,snapshot,root=tmp_path,condensation_seed=0,
                                        h=h,q=q,assignment=assignment,device='cpu')
    assert files(tmp_path)==before
