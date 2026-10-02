"""Focused new Cora certificate guards using generated CPU fixtures only."""
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src import cora_source_certificate as exp
from src.low_rank_assignment import initialize_factors
from src.moments import make_material
from src.sweep_utils import representative
from src.transforms import FeatureTransform


@pytest.fixture
def science():
    return json.loads((Path(__file__).resolve().parents[1]/"results/proposals"/exp.SCIENCE).read_text())


def evidence(science):
    return dict(counts={k:0 for k in science['expected_success_counts']},operation_attempts={})


@pytest.fixture
def frozen(tmp_path,monkeypatch,science):
    repo=tmp_path/'repo';repo.mkdir()
    original=str(Path(__file__).resolve().parents[1])
    data=copy.deepcopy(science)
    for case in data['cases']:case['source_root']=case['source_root'].replace(original,str(repo))
    data['original_files_sha256']={p.replace(original,str(repo)):h for p,h in data['original_files_sha256'].items()}
    for p in data['original_files_sha256']:
        path=Path(p);path.parent.mkdir(parents=True,exist_ok=True);path.write_text('opaque fixture')
    for case in data['cases']:
        root=Path(case['source_root'])
        (root/'config.json').write_text(json.dumps(case['source_config']))
        (root/f"student_recipe_{case['recipe_id']}.json").write_text(json.dumps(case['recipe']))
    data['original_files_sha256']={p:exp._sha(p) for p in data['original_files_sha256']}
    for item in data['parents']:
        item['path']=item['path'].replace(original,str(repo));p=Path(item['path']);p.parent.mkdir(parents=True,exist_ok=True);p.write_text('parent');item['sha256']=exp._sha(p)
    previous=data['preregistration_revision']['previous_unexecuted'];previous['path']=previous['path'].replace(original,str(repo))
    Path(previous['path']).write_text('unexecuted old science');previous['sha256']=exp._sha(previous['path'])
    data['original_files_sha256']={p:exp._sha(p) for p in data['original_files_sha256']}
    helper=repo/'src/unchanged_helper.py';helper.parent.mkdir(parents=True);helper.write_text('opaque protected source')
    data['readonly_helper_source_pins']={str(helper):exp._sha(helper)}
    sp=repo/'results/proposals'/exp.SCIENCE;sp.write_text(json.dumps(data));monkeypatch.setattr(exp,'SCIENCE_SHA',exp._sha(sp))
    monkeypatch.setattr(exp,'implementation_provenance',lambda:{'frozen':'source'})
    monkeypatch.setattr(exp,'numerical_source',lambda:{'frozen':'versions'})
    artifact=repo/'results/proposals/review.json';artifact.write_text('review')
    output=repo/'results/research_loop/Cora35_70_140_original_source_certificate_stageBB_v1/native_source_certificate_v1.json'
    spec=dict(schema=1,fixed=exp.FIXED,cases=copy.deepcopy(data['cases']),source={'frozen':'source'},numerical_source={'frozen':'versions'},
        python_version=exp.platform.python_version(),files_sha256=data['original_files_sha256'],
        scientific_preregistration={'path':str(sp),'sha256':exp.SCIENCE_SHA},artifacts_sha256={str(artifact):exp._sha(artifact)},output_path=str(output))
    path=repo/'results/proposals/spec.json'
    def save():path.write_text(json.dumps(spec));return path,exp._sha(path),output,repo
    return SimpleNamespace(repo=repo,data=data,spec=spec,path=path,output=output,save=save)


def test_fixed_three_case_spec_and_recipes_need_no_native(frozen):
    spec,_,_=exp._load_spec(*frozen.save())
    assert [c['cells'] for c in spec['cases']]==[35,70,140]
    assert [c['reference_candidate']['lr'] for c in spec['cases']]==[.1,.05,.1]
    assert len(exp._paths(frozen.repo))==57
    assert [exp._recipe(c)['contents']['weight_decay'] for c in spec['cases']]==[.0005,.001,.0005]


@pytest.mark.parametrize('mutation',['schema','caseorder','LR','rank','T','lambda','recipe','extra','source','python','missing','assets','science','output'])
def test_resealed_controls_and_source_reject_before_data(frozen,monkeypatch,mutation):
    if mutation=='schema':frozen.spec['schema']=True
    elif mutation=='caseorder':frozen.spec['cases'].reverse()
    elif mutation in ('LR','rank','T','lambda'):
        key={'LR':'lr','rank':'rank','T':'T','lambda':'penalty'}[mutation];frozen.spec['cases'][0]['reference_candidate'][key]=9
    elif mutation=='recipe':frozen.spec['cases'][0]['recipe']['weight_decay']=.001
    elif mutation=='extra':frozen.spec['candidate']={}
    elif mutation=='source':frozen.spec['source']={}
    elif mutation=='python':frozen.spec['python_version']='other'
    elif mutation=='missing':Path(next(iter(frozen.spec['files_sha256']))).unlink()
    elif mutation=='assets':Path(next(iter(frozen.spec['files_sha256']))).write_text('changed')
    elif mutation=='science':frozen.spec['scientific_preregistration']['sha256']='f'*64
    else:frozen.output.parent.mkdir(parents=True);frozen.output.write_text('preserve')
    monkeypatch.setattr(exp,'_prepare_graph',lambda *a:pytest.fail('dataset called'))
    with pytest.raises((ValueError,FileNotFoundError)):exp._load_spec(*frozen.save())


@pytest.mark.parametrize('kind',['input_scale','contents','id'])
def test_recipe_bound_separately_not_assumed_input_digest(frozen,kind):
    case=copy.deepcopy(frozen.spec['cases'][0]);path=Path(case['source_root'])/f"student_recipe_{case['recipe_id']}.json"
    if kind=='id':case['recipe_id']='unknown'
    else:
        value=json.loads(path.read_text());value['input_scale' if kind=='input_scale' else 'weight_decay']=2.;path.write_text(json.dumps(value))
    with pytest.raises((ValueError,FileNotFoundError)):exp._recipe(case)


@pytest.fixture(scope='module')
def reference_fixture(tmp_path_factory):
    """Generated280-node zero-feature toy, no data/head optimization/model."""
    patch=pytest.MonkeyPatch();patch.setattr(exp,'representative',lambda m,t,d,device:representative(m,t,d,'cpu'))
    patch.setattr(exp,'_input_digest',lambda *a:'mock graph input digest')
    science=json.loads((Path(__file__).resolve().parents[1]/'results/proposals'/exp.SCIENCE).read_text())
    root=tmp_path_factory.mktemp('Cora_reference')
    z=torch.zeros(280,1433,dtype=torch.float64);q=torch.full((280,7),1/7,dtype=torch.float64)
    transform=FeatureTransform(kind='rms',matrix=None,center=torch.zeros(1433,dtype=torch.float64),output_center=torch.zeros(1433,dtype=torch.float64),scale=torch.tensor(1.,dtype=torch.float64),eps=1e-12)
    result={}
    for case in science['cases']:
        cells=case['cells'];base=root/str(cells);folder=base/case['reference_id'];(folder/'condensation_0/checkpoints').mkdir(parents=True)
        hard=torch.arange(280)%cells
        b=dict(root=base,z=z,q=q,hard=hard,transform=transform,source={},graph={},train=None,val=None)
        ghost=exp.source_helper.candidate_controls(dict(case['reference_candidate'],method='source_linear',assignment_coordinates='raw_rms',source_linear_schema=1))
        cfg=exp.source_helper.expected_citation_config(ghost,0,z,q,hard,z.float(),{})
        for k in ('source_linear_coordinates','source_linear_schema','source_linear_source'):cfg.pop(k)
        cfg.update(assignment_input='node',save_assignment=False)
        u,v=initialize_factors(hard,cells,32,0);m=exp.LowRankMoments.apply(u,v,hard,make_material(z,q),.05,4096).detach()
        theta=torch.zeros(7,1434,dtype=torch.float64);center,label,mass=exp.decode_moments(m,1433)
        g=float(exp.head_gradient(exp.augmented(center),label,torch.full_like(mass,1/cells),theta,.0001).abs().max());J,_=exp.outer_value_gradient(z,q,theta,65536,exp.augmented(z))
        snapshots={s:dict(step=s,moments=m,theta=theta,inner_grad_max=g,teacher_ce=J,J_exact=True) for s in (0,25)}
        (folder/'candidate.json').write_text(json.dumps(case['reference_candidate']))
        torch.save(dict(step=25,config=cfg,snapshots=snapshots),folder/'condensation_0/resume.pt')
        for s,value in snapshots.items():torch.save(value,folder/f'condensation_0/checkpoints/step_{s:06d}.pt')
        result[cells]=(b,ghost,case,science)
    yield result
    patch.undo()


@pytest.mark.parametrize('cells',[35,70,140])
def test_actual_per_case_source_config_material_head_and_inputs(reference_fixture,cells):
    b,ghost,case,s=reference_fixture[cells];e=evidence(s);cert=exp._reference(b,ghost,case,e)
    assert cert['current_native_M0_allclose_1e12_cached_NODE0'] and cert['actual_FP32_P0_X_Q_uniform_equal_reference']
    assert cert['historical_checkpoint_UV_available'] is False and cert['original_head_penalty_from_validated_resume']==.0001
    assert e['counts']['factor_factory_attempts']==e['counts']['factor_factory_completed']==2
    assert e['counts']['cached_head_gradients']==e['counts']['cached_outer_teacher_CE']==2


def test_bitwise_M0_is_descriptive_with_same_exact_native_inputs(reference_fixture,tmp_path):
    import shutil
    b,g,c,s=reference_fixture[70];copyroot=tmp_path/'source';shutil.copytree(b['root'],copyroot);b=dict(b,root=copyroot)
    d=copyroot/c['reference_id']/'condensation_0';resume=torch.load(d/'resume.pt',weights_only=False)
    m=resume['snapshots'][0]['moments'].clone();m[0,-1]+=1e-16;m[0,-2]-=1e-16;m[1,-1]-=1e-16;m[1,-2]+=1e-16
    resume['snapshots'][0]['moments']=m;torch.save(resume,d/'resume.pt');torch.save(resume['snapshots'][0],d/'checkpoints/step_000000.pt')
    cert=exp._reference(b,g,c,evidence(s))
    assert cert['current_native_M0_bitwise_equal_cached_NODE0'] is False
    assert cert['current_native_M0_vs_cached_NODE0_max_abs']>0 and cert['actual_FP32_P0_X_Q_uniform_equal_reference']


@pytest.mark.parametrize('mutation',['config','theta','moment','snapshot','missing','retention'])
def test_reference_corruption_preserved_no_fit_fallback(reference_fixture,tmp_path,mutation):
    import shutil
    b,g,c,s=reference_fixture[35];root=tmp_path/'source';shutil.copytree(b['root'],root);b=dict(b,root=root);d=root/c['reference_id']/'condensation_0'
    resume=torch.load(d/'resume.pt',weights_only=False)
    if mutation=='config':resume['config']['penalty']=.001
    elif mutation=='retention':resume['config']['save_assignment']=0
    elif mutation=='theta':resume['snapshots'][0]['theta']=resume['snapshots'][0]['theta'].clone();resume['snapshots'][0]['theta'][0,0]=.01;torch.save(resume['snapshots'][0],d/'checkpoints/step_000000.pt')
    elif mutation=='moment':resume['snapshots'][0]['moments']=resume['snapshots'][0]['moments'].clone();resume['snapshots'][0]['moments'][0,0]+=1e-4;torch.save(resume['snapshots'][0],d/'checkpoints/step_000000.pt')
    elif mutation=='snapshot':resume['snapshots'][25]['step']=0
    else:(d/'checkpoints/step_000000.pt').unlink()
    torch.save(resume,d/'resume.pt');before={str(p):exp._sha(p) for p in root.rglob('*') if p.is_file()}
    with pytest.raises((ValueError,FileNotFoundError)):exp._reference(b,g,c,evidence(s))
    assert before=={str(p):exp._sha(p) for p in root.rglob('*') if p.is_file()}


@pytest.mark.parametrize('failure',[None,'second','memory','source','fresh'])
def test_one_fresh_job_all_cases_failure_and_primary_error(frozen,monkeypatch,failure):
    monkeypatch.setattr(exp,'__file__',str(frozen.repo/'src/cora_source_certificate.py'))
    native=[];cuda=SimpleNamespace(is_initialized=lambda:failure=='fresh',reset_peak_memory_stats=lambda:None,
        current_device=lambda:0,get_device_properties=lambda _:SimpleNamespace(total_memory=exp.CUDA_CAPACITY),
        synchronize=lambda:(_ for _ in ()).throw(RuntimeError('secondary memory')) if failure=='memory' else None,
        max_memory_allocated=lambda:1,max_memory_reserved=lambda:2)
    monkeypatch.setattr(exp,'torch',SimpleNamespace(cuda=cuda,get_num_threads=lambda:4))
    monkeypatch.setattr(exp.probe,'_native',lambda _:native.append(1) or {'fake':'environment'})
    monkeypatch.setattr(exp.probe,'_runtime_precision_guard',lambda:None)
    def graph(*args):args[1]['counts']['dataset_preparations']=1;args[1]['counts']['dense_S_materializations']=1;return 'shared'
    monkeypatch.setattr(exp,'_prepare_graph',graph);monkeypatch.setattr(exp,'_shared_digest',lambda _: 'exact')
    def case(c,shared,e,stop):
        if (failure=='second' and c['cells']==70) or failure=='memory':e['stage']='original_source_guard';raise ValueError('primary source error')
        for k,v in dict(per_root_cached_source_calls=1,factor_factory_attempts=2,factor_factory_completed=2,P0_logit_evaluations=1,
                        P0_material_moment_evaluations=1,cached_head_gradients=2,cached_outer_teacher_CE=2,per_root_CSR_dense_binding_checks=1).items():e['counts'][k]+=v
        if failure=='source':monkeypatch.setattr(exp,'implementation_provenance',lambda:{'changed':True})
        return {'passed':True}
    monkeypatch.setattr(exp,'_execute_case',case)
    path,checksum,output,_=frozen.save()
    if failure is None:
        result=exp.prepare_certificate(path,checksum,output);assert result['case_visit_order']==[35,70,140] and native==[1]
        assert result['counts']==frozen.data['expected_success_counts']
    else:
        with pytest.raises((ValueError,RuntimeError),match='primary source error' if failure in ('second','memory') else '.*'):exp.prepare_certificate(path,checksum,output)
    result=json.loads(output.read_text());assert result['passed'] is (failure is None)
    if failure=='memory':assert result['memory_error']['message']=='secondary memory' and result['source_error']['message']=='primary source error'
    if failure=='second':assert result['case_visit_order']==[35,70] and result['counts']['factor_factory_completed']==2
    assert all(result['counts'][k]==0 for k in ('head_solves','P_updates','students','SGD_steps','accuracy_forwards'))


def test_actual_lazy_worker_dispatch_preserves_options_and_stop(monkeypatch):
    from src.research_loop import dispatch
    seen=[];stop=lambda:False
    monkeypatch.setitem(sys.modules,'src.cora_source_certificate',SimpleNamespace(prepare_certificate=lambda **kw:seen.append(kw) or 'returned'))
    assert dispatch(dict(kind='citation_cora_source_certificate',options={'spec_path':'frozen'}),stop)=='returned'
    assert seen==[dict(spec_path='frozen',stop=stop)]
