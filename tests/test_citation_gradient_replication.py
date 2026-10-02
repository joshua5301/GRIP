"""CPU-only cond-indexed source/reference/cache/dispatch checks, no realdata."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src import citation_gradient_replication as exp
from src.low_rank_assignment import initialize_factors
from src.moments import make_material
from src.sweep_utils import representative
from src.transforms import FeatureTransform


@pytest.mark.parametrize('c',[0,3,-1,True,1.,'1'])
def test_only_native_independent_cond_indices(c):
    with pytest.raises(ValueError):exp.canonical_candidate(c)


@pytest.mark.parametrize('c',[1,2])
def test_namespace_and_math_are_fixed_and_cond_specific(c):
    candidate=exp.canonical_candidate(c)
    assert candidate['condensation_seed']==c and candidate['replication_schema']==1
    assert candidate['anchor_seeds']==[0,1,2] and candidate['lr']==.01 and candidate['assignment_steps']==25
    assert candidate!=exp.canonical_candidate(3-c)
    assert exp._state is exp.az._state and exp._check_progress is exp.az._check_progress
    assert exp._verify_targets is exp.az._verify_targets and exp._store is exp.az._store
    assert exp._folder(dict(output_root='/toy',candidate_ids={'1':'first','2':'second'}),c)==Path(f'/toy/condensation_{c}/'+('first' if c==1 else 'second'))


@pytest.fixture
def common():
    h=torch.ones(4,3,dtype=torch.float32);z=h.double();q=torch.ones(4,2,dtype=torch.float64)/2
    t=dict(kind='rms',matrix=None,center=torch.zeros(3,dtype=torch.float64),output_center=torch.zeros(3,dtype=torch.float64),scale=torch.tensor(1.,dtype=torch.float64),eps=1e-12)
    b=dict(h=h,z=z,q=q,transform=t,x=h,S=torch.eye(4),original_S=torch.eye(4).to_sparse_csr())
    digest=exp.probe._digest(dict(H=h,z=z,Q=q,transform=t,X=h,original_CSR=b['original_S'],dense_original_S=b['S']))
    return b,dict(digest,hard={'different':'permitted'})


@pytest.mark.parametrize('key',list(exp._COMMON)+['extra'])
def test_AT_common_exact_excludes_only_hard(common,key):
    b,saved=common
    assert exp._common_certificate(b,saved)
    saved['hard']={'another':'independent partition'}
    assert exp._common_certificate(b,saved)
    if key=='extra':saved['new_unknown']=1
    else:saved[key]={'changed':'same source is mandatory'}
    with pytest.raises(ValueError):exp._common_certificate(b,saved)


@pytest.fixture(scope='module')
def references(tmp_path_factory):
    """Actual toy native factors/material/cached gradient algebra, never a fit."""
    patch=pytest.MonkeyPatch()
    patch.setattr(exp,'representative',lambda m,t,d,dev:representative(m,t,d,'cpu'))
    patch.setattr(exp,'_input_digest',lambda *a,**k:'toy input fingerprint')
    root=tmp_path_factory.mktemp('ownref');folder=root/exp.original.REFERENCE;folder.mkdir()
    (folder/'candidate.json').write_text(json.dumps(exp.original.CANDIDATE))
    z=torch.zeros(240,3703,dtype=torch.float64);q=torch.full((240,6),1/6,dtype=torch.float64);hard=torch.arange(240)%120
    transform=FeatureTransform(kind='rms',matrix=None,center=torch.zeros(3703,dtype=torch.float64),output_center=torch.zeros(3703,dtype=torch.float64),scale=torch.tensor(1.,dtype=torch.float64),eps=1e-12)
    ghost=exp.source_helper.candidate_controls(dict(exp.original.CANDIDATE,method='source_linear',assignment_coordinates='raw_rms',source_linear_schema=1))
    buffers=dict(root=root,z=z,q=q,hard=hard,transform=transform,source={},graph={},train=None,val=None)
    for c in (1,2):
        config=exp.source_helper.expected_citation_config(ghost,c,z,q,hard,z.float(),{})
        for key in ('source_linear_coordinates','source_linear_schema','source_linear_source'):config.pop(key)
        config.update(assignment_input='node',save_assignment=False)
        u,v=initialize_factors(hard,120,8,c)
        m=exp.LowRankMoments.apply(u,v,hard,make_material(z,q),.05,4096).detach()
        theta=torch.zeros(6,3704,dtype=torch.float64)
        centers,labels,mass=exp.decode_moments(m,3703)
        gradient=float(exp.head_gradient(exp.augmented(centers),labels,torch.full_like(mass,1/120),theta,.001).abs().max())
        value,_=exp.outer_value_gradient(z,q,theta,65536,exp.augmented(z))
        snapshots={s:dict(step=s,moments=m,theta=theta,inner_grad_max=gradient,teacher_ce=value,J_exact=True) for s in (0,25)}
        d=folder/f'condensation_{c}';(d/'checkpoints').mkdir(parents=True)
        torch.save(dict(step=25,config=config,snapshots=snapshots),d/'resume.pt')
        for s,snapshot in snapshots.items():torch.save(snapshot,d/f'checkpoints/step_{s:06d}.pt')
    yield SimpleNamespace(buffers=buffers,ghost=ghost,root=root)
    patch.undo()


def counters():return dict(native_initializer_calls=0,native_initializer_completed_calls=0,native_P0_probability_material_evaluations=0,cached_head_gradient_evaluations=0,cached_outer_CE_evaluations=0)


@pytest.mark.parametrize('c',[1,2])
def test_actual_own_reference_config_factory_material_and_head_certificates(references,c):
    e=counters();result=exp._reference(references.buffers,references.ghost,c,e)
    assert result['condensation_seed']==c and result['source_reference_certificate_passed'] is True
    assert result['current_native_M0_exactly_equal_own_cached_NODE0'] is True
    assert result['actual_FP32_P0_X_Q_uniform_equal_reference'] is True
    assert result['historical_checkpoint_UV_available'] is False
    assert e['native_initializer_calls']==e['native_initializer_completed_calls']==2
    assert e['cached_head_gradient_evaluations']==e['cached_outer_CE_evaluations']==2
    u,v=initialize_factors(references.buffers['hard'],120,8,c)
    assert result['P0_parameters']==exp.probe._digest([u,v])


@pytest.mark.parametrize('mutation',['factor_seed','retention','head_penalty','theta','moment','checkpoint','missing_own'])
def test_original_percond_reference_tamper_preserved_before_P(references,tmp_path,mutation):
    import shutil
    shutil.copytree(references.root,tmp_path/'source');b=dict(references.buffers,root=tmp_path/'source')
    d=b['root']/exp.original.REFERENCE/'condensation_1'
    resume=torch.load(d/'resume.pt',weights_only=False)
    if mutation=='factor_seed':resume['config']['factor_seed']=0
    elif mutation=='retention':resume['config']['save_assignment']=0
    elif mutation=='head_penalty':resume['config']['penalty']=.0001
    elif mutation in ('theta','moment'):
        key='theta' if mutation=='theta' else 'moments';resume['snapshots'][0][key]=resume['snapshots'][0][key].clone();resume['snapshots'][0][key].flatten()[0]+=.01
        torch.save(resume['snapshots'][0],d/'checkpoints/step_000000.pt')
    elif mutation=='checkpoint':resume['snapshots'][25]['step']=24
    else:(d/'checkpoints/step_000000.pt').unlink()
    torch.save(resume,d/'resume.pt')
    before={str(p):exp._sha(p) for p in b['root'].rglob('*') if p.is_file()}
    with pytest.raises((ValueError,FileNotFoundError)):exp._reference(b,references.ghost,1,counters())
    assert before=={str(p):exp._sha(p) for p in b['root'].rglob('*') if p.is_file()}


@pytest.mark.parametrize('case',['false','cross_cond'])
def test_no_scientific_update_or_target_access_before_own_source_certificate(tmp_path,case):
    b=dict(condensation_seed=1)
    reference=dict(source_reference_certificate_passed=case!='false',condensation_seed=2 if case=='cross_cond' else 1)
    context=dict(reference=reference,condensation_seed=2 if case=='cross_cond' else 1)
    spec=dict(output_root=str(tmp_path),candidate_ids={'1':'one','2':'two'});e=dict(counts={})
    with pytest.raises(ValueError):exp._prepare(spec,b,context,e,lambda:False)
    assert e['counts']=={} and not list(tmp_path.iterdir())


@pytest.mark.parametrize('mutation',['cond','source_certificate','common_source','prefix'])
def test_cond_specific_native_gate_before_student_access(tmp_path,monkeypatch,mutation):
    evidence=tmp_path/'gate.json';asset=tmp_path/'asset';asset.write_text('pin')
    s=dict(certificate_outputs={'1':str(evidence)},source={},numerical_source={},candidates={'1':{}},candidate_ids={'1':'one'},gradient_policy={},scientific_preregistration={})
    g=dict(passed=True,schema=1,assignment_steps=25,source={},numerical_source={},candidate={},candidate_id='one',cells=120,condensation_seed=1,
       full25_cache_replay_passed=True,actual_FP32_P0_X_Q_uniform_equal_reference=True,source_reference_certificate_passed=True,
       common_AT_native_buffers_exact_excluding_hard=True,source_target_cache_replay_passed=True,gradient_policy={},scientific_preregistration={},
       prefix_sha256='a'*64,spec_sha256='b'*64,files_sha256={str(asset):exp._sha(asset)})
    key={'cond':'condensation_seed','source_certificate':'source_reference_certificate_passed','common_source':'common_AT_native_buffers_exact_excluding_hard','prefix':'prefix_sha256'}[mutation]
    g[key]=2 if mutation=='cond' else False if mutation!='prefix' else ''
    evidence.write_text(json.dumps(g));monkeypatch.setattr(exp,'_load_progress',lambda *a,**k:pytest.fail('student/cache access beforegate'))
    with pytest.raises(ValueError):exp._validate(s,dict(condensation_seed=1),{},'gradient',exp._sha(evidence),dict(counts={}),lambda:False)


@pytest.mark.parametrize('operation',['prepare','certify','validate'])
def test_three_real_lazy_dispatches(operation,monkeypatch):
    from src.research_loop import dispatch
    seen=[];callback=lambda:False
    fake=SimpleNamespace(**{operation:lambda **kw:seen.append(kw) or 'returned'})
    monkeypatch.setitem(sys.modules,'src.citation_gradient_replication',fake)
    assert dispatch(dict(kind='citation_gradient_replication_'+operation,options={'condensation_seed':2}),callback)=='returned'
    assert seen==[dict(condensation_seed=2,stop=callback)]
