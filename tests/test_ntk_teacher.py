"""Tiny synthetic cache/workflow checks: no dataset teacher/student fits."""
import hashlib
import json
from pathlib import Path

import pytest
import torch
from test_nystrom_balanced_mass import citation_mock as citation_mock
from test_nystrom_balanced_mass import load

import src.citation_search as search
import src.ntk_teacher as teacher
from src.io import _fingerprint, save_state
from src.nystrom_ce import _content_digest
from src.relu_ntk import KIND
from src.shared_features import get_shared_map


def files(root):
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()}


def option():
    return dict(method='nystrom',width=0,lr=.01,T=1.,rank=3,penalty=.1,inner_loss_weighting='uniform',surrogate_kernel=KIND)


@pytest.fixture
def source(tmp_path,citation_mock,monkeypatch):
    monkeypatch.setattr(search,'get_shared_map',get_shared_map)
    plain=option();plain.pop('surrogate_kernel')
    _,root=search.run_screen('cora',.013,tmp_path,[plain,option()],steps=1,student_seeds=(2100,),epochs=1,dropout=0,device='cpu')
    graph,train,val,testing,h=search._prepare_dataset('cora','data','cpu','default')
    labels,val_labels=teacher.teacher_inputs(graph,train,val[1])
    return dict(base_config=json.loads((root/'config.json').read_text()),h=h,train_mask=train,train_labels=labels,
                val_mask=val[1],val_labels=val_labels,output_dir=tmp_path,source_root=root)


def prepare(source,gamma,**extra):
    return teacher.prepare_gamma(**source,gamma=gamma,**extra)


def complete(source):
    result=None
    for gamma in teacher.GAMMAS:result=prepare(source,gamma)
    return Path(result['root'])


@pytest.mark.parametrize('value',[None,True,'bad','relu2',1])
def test_top_level_kernel_rejects_before_data(tmp_path,citation_mock,value):
    with pytest.raises(ValueError):search.run_screen('cora',.013,tmp_path,[option()],teacher_kernel=value,device='cpu')
    assert citation_mock['data_calls']==0 and not list(tmp_path.iterdir())


def test_ntk_teacher_requires_matching_ntk_surrogate_preload(tmp_path,citation_mock):
    plain=option();plain.pop('surrogate_kernel')
    with pytest.raises(ValueError):search.run_screen('cora',.013,tmp_path,[plain],teacher_kernel=KIND,device='cpu')
    assert citation_mock['data_calls']==0 and not list(tmp_path.iterdir())


def test_gamma_only_copy_source_assets_and_partial_grid_blocks_condensation(source,citation_mock):
    old=files(source['source_root']);result=prepare(source,teacher.GAMMAS[0]);root=Path(result['root'])
    assert not result['grid_complete'] and not (root/'teacher.pt').exists()
    assert files(source['source_root'])==old and root!=source['source_root']
    context=json.loads((root/'config.json').read_text())['teacher_context']
    for name,digest in context['assets'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest
    assert not list(root.glob('assignment_*.pt')) and not list(root.glob('inputs_*.pt'))
    fits=len(citation_mock['fits']);before=files(source['output_dir'])
    with pytest.raises(ValueError,match='all four'):
        search.run_screen('cora',.013,source['output_dir'],[option()],teacher_kernel=KIND,device='cpu',steps=1,epochs=1)
    assert len(citation_mock['fits'])==fits and files(source['output_dir'])==before


def test_complete_grid_strict_first_max_selected_content_and_cached_bypass(source,monkeypatch):
    root=complete(source);selected,config,_=teacher.validate_root(root)
    rows=[load(teacher.gamma_path(root,g)) for g in teacher.GAMMAS]
    best=max(row['val'] for row in rows);expected=next(row['gamma'] for row in rows if row['val']==best)
    assert selected['gamma']==expected
    assert torch.equal(selected['logits'],rows[list(teacher.GAMMAS).index(expected)]['logits'])
    assert config['teacher_context']['recipe']['initial_weight']=='zeros'
    before=files(root)
    monkeypatch.setattr(teacher,'fit_streaming_teacher',lambda *a,**k:pytest.fail('cached fit bypassed'))
    assert prepare(source,teacher.GAMMAS[0])['cached'] and files(root)==before


def test_interruption_after_fourth_cache_before_selection_recovers_without_fit(source,monkeypatch):
    for gamma in teacher.GAMMAS[:-1]:prepare(source,gamma)
    original=teacher._selected
    def interrupted(*args,**kwargs):
        if kwargs.get('create') is True:raise InterruptedError('after fourth commit')
        return original(*args,**kwargs)
    monkeypatch.setattr(teacher,'_selected',interrupted)
    with pytest.raises(InterruptedError):prepare(source,teacher.GAMMAS[-1])
    _,_,config=teacher.context(source['h'],source['base_config'],source['source_root'],source['train_mask'],source['train_labels'],source['val_mask'],source['val_labels'])
    root=teacher.teacher_root(source['output_dir'],source['base_config'],config)
    assert all(teacher.gamma_path(root,g).exists() for g in teacher.GAMMAS) and not (root/'teacher.pt').exists()
    with pytest.raises(ValueError,match='atomically selected'):teacher.validate_root(root)
    monkeypatch.setattr(teacher,'_selected',original)
    monkeypatch.setattr(teacher,'fit_streaming_teacher',lambda *a,**k:pytest.fail('recovery must not fit'))
    assert prepare(source,teacher.GAMMAS[-1])['grid_complete']
    assert (root/'teacher.pt').exists()


@pytest.mark.parametrize('change',['weight','logits','objective','gradient','val','val_ce','context','nan_weight','nan_logits','route','schema'])
def test_gamma_tampering_rejected_before_new_fit_or_existing_cache_mutation(source,monkeypatch,change):
    result=prepare(source,teacher.GAMMAS[0]);root=Path(result['root']);path=teacher.gamma_path(root,teacher.GAMMAS[0]);state=load(path)
    if change=='weight':state['weight'][0,0]+=.3;state['weight_digest']=_content_digest(state['weight'])
    elif change=='logits':state['logits'][0,0]+=.3;state['logits_digest']=_content_digest(state['logits'])
    elif change=='context':state['context']['recipe']['max_iter']=999
    elif change=='route':state['route']['actual']='invented'
    elif change=='gradient':state['gradient_max']+=1
    elif change=='schema':state['schema']=True
    elif change.startswith('nan_'):
        key=change[4:];state[key][0,0]=float('nan');state[key+'_digest']=_content_digest(state[key])
    else:state[change]+=1
    save_state(state,path);before=files(root)
    monkeypatch.setattr(teacher,'fit_streaming_teacher',lambda *a,**k:pytest.fail('tampered cache must reject before fit'))
    with pytest.raises(ValueError):prepare(source,teacher.GAMMAS[1])
    assert files(root)==before


@pytest.mark.parametrize('change',['train_mask','train_labels','val_mask','val_labels','helper_source','binding','copied_phi','partial_grid','final_logits'])
def test_selected_teacher_replays_current_source_context_and_grid_readonly(source,monkeypatch,change):
    root=complete(source);kwargs={}
    if change in ('train_mask','train_labels','val_mask','val_labels'):
        kwargs={k:source[k].clone() for k in ('h','train_mask','train_labels','val_mask','val_labels')}
        if change=='train_mask':kwargs['train_mask'][0]=False;kwargs['train_labels'][0]=0
        elif change=='train_labels':kwargs['train_labels'][0]=1
        elif change=='val_mask':kwargs['val_mask']=kwargs['val_mask'].roll(1)
        else:kwargs['val_labels'][0]=(kwargs['val_labels'][0]+1)%3
    elif change=='helper_source':monkeypatch.setattr(teacher,'source_digest',lambda:'a'*64)
    elif change=='binding':p=root/'source_binding.json';data=json.loads(p.read_text());data['assets']['propagated_H.pt']='a'*64;p.write_text(json.dumps(data))
    elif change=='copied_phi':
        import numpy as np
        p=next(root.glob('nystrom_phi_*.npy'));data=np.load(p);data[0,0]+=.1;np.save(p,data)
    elif change=='partial_grid':teacher.gamma_path(root,teacher.GAMMAS[0]).unlink()
    else:p=root/'teacher.pt';state=load(p);state['logits']+=.1;save_state(state,p)
    before=files(root)
    with pytest.raises(ValueError):teacher.validate_root(root,**kwargs)
    assert files(root)==before


def test_stop_or_nonfinite_solver_result_never_commits_gamma(source,monkeypatch):
    with pytest.raises(InterruptedError):prepare(source,teacher.GAMMAS[0],stop=lambda:True)
    original=teacher.fit_streaming_teacher
    def broken(*args,**kwargs):
        logits,weight=original(*args,**kwargs);weight[0,0]=float('nan');return logits,weight
    monkeypatch.setattr(teacher,'fit_streaming_teacher',broken)
    with pytest.raises(RuntimeError):prepare(source,teacher.GAMMAS[0])
    for root in source['output_dir'].glob('cora/ratio_0.013/*'):
        if root!=source['source_root']:assert not list((root/'teacher_gammas').glob('gamma_*.pt')) and not (root/'teacher.pt').exists()


def test_explicit_relu_and_default_legacy_root_candidate_and_cache_identity(source,citation_mock):
    root=source['source_root'];before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('nystrom*') if p.is_file()}
    a,x=search.run_screen('cora',.013,source['output_dir'],[option()],steps=1,student_seeds=(2100,),epochs=1,dropout=0,device='cpu')
    b,y=search.run_screen('cora',.013,source['output_dir'],[option()],steps=1,student_seeds=(2100,),epochs=1,dropout=0,device='cpu',teacher_kernel='relu')
    assert x==y==root and a.candidate_path.tolist()==b.candidate_path.tolist()
    assert 'teacher_kernel' not in json.loads((root/'config.json').read_text())
    assert before=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('nystrom*') if p.is_file()}


def test_gamma_dispatch_is_one_preparation_only(monkeypatch):
    import src.research_loop as loop
    calls=[]
    monkeypatch.setattr(teacher,'prepare_gamma_job',lambda **kwargs:calls.append(kwargs) or {'validation_only':True})
    stop=lambda:False
    assert loop.dispatch(dict(kind='citation_teacher_gamma',options=dict(teacher_gamma=.001)),stop)['validation_only']
    assert calls==[dict(teacher_gamma=.001,stop=stop)]


@pytest.mark.parametrize('field',['objective','gradient_max','val','val_ce','seconds'])
@pytest.mark.parametrize('value',[None,'invalid',[],True,float('nan')])
def test_malformed_gamma_scalar_diagnostics_fail_cleanly_before_fit(source,monkeypatch,field,value):
    root=Path(prepare(source,teacher.GAMMAS[0])['root']);path=teacher.gamma_path(root,teacher.GAMMAS[0]);state=load(path)
    state[field]=value;save_state(state,path);before=files(root)
    monkeypatch.setattr(teacher,'fit_streaming_teacher',lambda *a,**k:pytest.fail('no fit on malformed diagnostic'))
    with pytest.raises(ValueError):prepare(source,teacher.GAMMAS[1])
    assert files(root)==before


@pytest.mark.parametrize('value',[None,'invalid',[],1])
def test_malformed_teacher_inputs_fail_cleanly_before_cache_mutation(source,monkeypatch,value):
    root=Path(prepare(source,teacher.GAMMAS[0])['root']);path=root/'teacher_inputs.pt';state=load(path)
    state['train_mask']=value;save_state(state,path);before=files(root)
    monkeypatch.setattr(teacher,'fit_streaming_teacher',lambda *a,**k:pytest.fail('no fit on malformed input'))
    with pytest.raises(ValueError):prepare(source,teacher.GAMMAS[1])
    assert files(root)==before


def test_new_teacher_condensation_and_selected_context_roundtrip(source,citation_mock):
    root=complete(source);selected,_,_=teacher.validate_root(root)
    before_teacher_calls=citation_mock['teacher_calls']
    ranking,actual=search.run_screen('cora',.013,source['output_dir'],[option()],teacher_kernel=KIND,
        steps=1,student_seeds=(2200,),epochs=1,dropout=0,device='cpu')
    assert actual==root and citation_mock['teacher_calls']==before_teacher_calls
    choice=ranking[ranking.step==1].iloc[0].to_dict()
    search.selected_test(root,choice,condensation_seeds=(0,),student_seeds=(2201,),epochs=1,dropout=0,device='cpu',report_routes=False)
    record=json.loads((root/'selected.json').read_text())
    assert record['teacher_kernel']==KIND
    assert record['teacher_context_digest']==_fingerprint(json.loads((root/'config.json').read_text())['teacher_context'])
    expected=(selected['logits']/option()['T']).softmax(1).double()
    torch.testing.assert_close(citation_mock['fits'][-1]['q'],expected,atol=0,rtol=0)


def test_selected_teacher_validates_actual_training_labels_before_selection_write(source,citation_mock,monkeypatch):
    root=complete(source)
    ranking,_=search.run_screen('cora',.013,source['output_dir'],[option()],teacher_kernel=KIND,
        steps=1,student_seeds=(2200,),epochs=1,dropout=0,device='cpu')
    choice=ranking[ranking.step==1].iloc[0].to_dict()
    graph,*_=search._prepare_dataset('cora','data','cpu','default');graph['y'][0]=1
    # Hold the outer dataset guard fixed to check independent teacher-input binding.
    monkeypatch.setattr(search,'dataset_digest',lambda *args:source['base_config']['data_digest'])
    before=len(citation_mock['fits']);snapshot=files(root)
    with pytest.raises(ValueError,match='teacher.*context|Teacher.*context'):
        search.selected_test(root,choice,condensation_seeds=(0,),student_seeds=(2201,),epochs=1,device='cpu',report_routes=False)
    assert len(citation_mock['fits'])==before and files(root)==snapshot and not (root/'selected.json').exists()


def test_teacher_aware_initializer_is_regenerated_in_new_teacher_root(source,citation_mock):
    root=complete(source);candidate=option();candidate.update(initialization='teacher_joint',alpha=.3)
    ranking,actual=search.run_screen('cora',.013,source['output_dir'],[candidate],teacher_kernel=KIND,
        steps=0,student_seeds=(2200,),epochs=1,dropout=0,device='cpu')
    assert actual==root and list(root.glob('assignment_*.pt'))
    path=Path(ranking.iloc[0].candidate_path)/'condensation_0/step_000000.pt';saved=load(path)
    assert saved['input_fingerprint']['q_digest']==_content_digest((load(root/'teacher.pt')['logits']/candidate['T']).softmax(1),canonical_double=True)


def test_selected_schema_boolean_rejected_readonly(source):
    root=complete(source);path=root/'teacher.pt';state=load(path);state['schema']=True;save_state(state,path)
    before=files(root)
    with pytest.raises(ValueError):teacher.validate_root(root)
    assert files(root)==before


@pytest.mark.parametrize('value',[None,'invalid',[],1])
def test_selected_missing_tensor_input_rejected_cleanly_readonly(source,value):
    root=complete(source);path=root/'teacher_inputs.pt';state=load(path);state['train_mask']=value;save_state(state,path)
    before=files(root)
    with pytest.raises(ValueError):teacher.validate_root(root)
    assert files(root)==before


def test_selection_tie_uses_first_accuracy_max_without_ce_tiebreak(tmp_path):
    config={'teacher_context':{'scope':'tiny selector fixture'}}
    states={}
    for index,gamma in enumerate(teacher.GAMMAS):
        state=dict(val=.5,val_ce=4-index,logits=torch.zeros(2,2,dtype=torch.double),weight=torch.zeros(3,2,dtype=torch.double),weight_digest='weight',logits_digest='logits')
        states[gamma]=state;path=teacher.gamma_path(tmp_path,gamma);path.parent.mkdir(exist_ok=True);save_state(state,path)
    selected=teacher._selected(tmp_path,config,states,create=True)
    assert selected['gamma']==teacher.GAMMAS[0]


@pytest.mark.parametrize('change',[dict(teacher_kernel='bad'),dict(teacher_gamma=.1),dict(teacher_gamma=True),
    dict(require_existing_source_features=False),dict(teacher_recipe={'max_iter':1})])
def test_per_gamma_unsupported_controls_reject_before_dataset(tmp_path,citation_mock,change):
    controls=dict(dataset='cora',ratio=.013,output_dir=tmp_path,source_root=tmp_path,teacher_gamma=.001,device='cpu');controls.update(change)
    with pytest.raises(ValueError):teacher.prepare_gamma_job(**controls)
    assert citation_mock['data_calls']==0 and not list(tmp_path.iterdir())


@pytest.mark.parametrize('change',['helper','torch'])
def test_explicit_frozen_job_recipe_rejects_source_runtime_drift_preload(tmp_path,citation_mock,monkeypatch,change):
    frozen=teacher.recipe()
    if change=='helper':monkeypatch.setattr(teacher,'source_digest',lambda:'a'*64)
    else:monkeypatch.setattr(torch,'__version__','unsupported_future_runtime')
    with pytest.raises(ValueError,match='recipe'):
        teacher.prepare_gamma_job(dataset='cora',ratio=.013,output_dir=tmp_path,source_root=tmp_path,
                                  teacher_gamma=.001,device='cpu',teacher_recipe=frozen)
    assert citation_mock['data_calls']==0 and not list(tmp_path.iterdir())
