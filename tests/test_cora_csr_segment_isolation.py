"""BM-only new science/source73/artifact/dispatch contracts; AST and metadata."""
import ast
import copy
import hashlib
import json
import types
from pathlib import Path

import pytest

BASE=Path(__file__).resolve().parents[1]
PATH=BASE/'src/cora_csr_segment_isolation.py'
WORKER=BASE/'src/research_loop.py'
REPO=next(p for p in Path(__file__).resolve().parents if (p/'src/transforms.py').is_file())
SCIENCE=json.loads((REPO/'results/proposals/Cora70_original_CSR_weighted_segment_fixedanchor0_scientific_stageBM_v1.json').read_text())
SHA='37c5581f3ee9220d8217050c05936f265da26c4931e6922bc00b8660a6d0522c'


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def require(v,m):
    if not v: raise ValueError(m)


def api(tmp):
    tree=ast.parse(PATH.read_text());names={'_load_spec','_artifact_paths'}
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    nodes += [n for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('_FIELDS','SCIENCE','SCIENCE_SHA','_BACKEND')]
    ns=dict(Path=Path,json=json,__file__=str(tmp/'src/module.py'),_require=require,_sha=sha,_exact=lambda a,b:type(a) is type(b) and a==b,
            _science=lambda r:SCIENCE,_preserve=lambda *a:None)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(PATH),'exec'),ns)
    spec={k:copy.deepcopy(SCIENCE['original_files_sha256'] if k=='files_sha256' else SCIENCE[k]) for k in
          ('fixed','files_sha256','output_root','pass_outputs','target_outputs','comparison_policy')}
    spec.update(schema=1,source={'files':dict(SCIENCE['source_before']['files'])|{n:'future' for n in SCIENCE['helper_boundary']['next_owned_sources']}},
                numerical_source={'versions':SCIENCE['version_binding']['runtime_versions_from_pinned_native_metadata']},
                python_version=SCIENCE['version_binding']['python_version'],artifacts_sha256={str(tmp/'src/csr_segment_adjoint.py'):'future'},
                scientific_preregistration={'path':str(tmp/'results/proposals'/ns['SCIENCE']),'sha256':SHA})
    path=tmp/'results/proposals/spec.json';path.parent.mkdir(parents=True)
    return ns,spec,path


def test_new_science_source73_and_backend_contract_accepts(tmp_path):
    ns,spec,path=api(tmp_path);path.write_text(json.dumps(spec))
    accepted,science=ns['_load_spec'](path,sha(path))
    assert accepted==spec and len(accepted['source']['files'])==73
    assert ns['_BACKEND']==science['fixed']['source_backend']=='original_CSR_coefficients_rank2_weighted_segment_SUM'


@pytest.mark.parametrize('mutation',['old_science','missing_owned_source','extra_source'])
def test_old_namespace_or_wrong_promoted_source_boundary_rejected(tmp_path,mutation):
    ns,spec,path=api(tmp_path)
    if mutation=='old_science':spec['scientific_preregistration']['sha256']='oldBK'
    elif mutation=='missing_owned_source':spec['source']['files'].pop('src/csr_segment_adjoint.py')
    else:spec['source']['files']['src/undeclared.py']='unreviewed'
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError):ns['_load_spec'](path,sha(path))


def test_exact_five_new_owned_artifacts_allowed_but_undeclared_src_rejected(tmp_path):
    ns,_,_=api(tmp_path)
    allowed=SCIENCE['helper_boundary']['next_owned_sources']+SCIENCE['helper_boundary']['next_owned_tests']+['src/research_loop.py']
    ns['_artifact_paths']({str(tmp_path/p):'pin' for p in allowed},SCIENCE,tmp_path)
    with pytest.raises(ValueError):ns['_artifact_paths']({str(tmp_path/'src/undeclared.py'):'pin'},SCIENCE,tmp_path)


def test_actual_worker_new_lazy_branch_preserves_callback_and_result(monkeypatch):
    import sys
    tree=ast.parse(WORKER.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='dispatch')
    branch=next(n for n in fn.body if isinstance(n,ast.If) and 'citation_cora_csr_segment_isolation' in ast.unparse(n.test))
    call=ast.FunctionDef(name='call',args=ast.arguments(posonlyargs=[],args=[ast.arg(arg='job'),ast.arg(arg='stop')],kwonlyargs=[],kw_defaults=[],defaults=[]),
                        body=[fn.body[0],branch],decorator_list=[])
    ns={};exec(compile(ast.fix_missing_locations(ast.Module(body=[call],type_ignores=[])),str(WORKER),'exec'),ns)
    marker=object();stop=lambda:False;calls=[]
    def prepare(**kwargs):calls.append(kwargs);return marker
    module=types.ModuleType('src.cora_csr_segment_isolation');module.prepare_isolation=prepare
    package=types.ModuleType('src');package.__path__=[]
    monkeypatch.setitem(sys.modules,'src',package);monkeypatch.setitem(sys.modules,'src.cora_csr_segment_isolation',module)
    assert ns['call']({'kind':'citation_cora_csr_segment_isolation','options':{'pass_index':1}},stop) is marker
    assert calls==[{'pass_index':1,'stop':stop}]
