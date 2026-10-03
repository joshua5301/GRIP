"""New independent SciPy constructive domains only; literal seven-node toy.

Durable tests use explicit CPU tensors. The isolated proof launcher establishes
CUDA-hidden/precision/thread policy; no real source/cache is deserialized here.
"""
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src/large_current_origin.py").is_file())
HELPER = (ROOT / "src/scipy_full_ce_reference.py" if Path(__file__).resolve().parent == ROOT / "tests" else
          ROOT / "results/implementation_drafts/independent_scipy_full_CE_stageBZ_v1/proposed/src/scipy_full_ce_reference.py")
SPEC = importlib.util.spec_from_file_location("owned_independent_scipy_reference", HELPER)
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)
m._libraries()
OLD_WORKER_AST_SHA256 = "04fb462ee7f68881eecb3091aa5571b5ce22f8f3731cddb00d48facaf44aa6cf"
METRICS = dict(operator=[], full=[], FD=[], masks=[], counters={})
CALL_CONTEXTS = []
ROWS = [[(0,.5),(2,-.25),(4,0.)],[],[(0,-.125),(1,.75),(3,.25)],[(2,.5),(5,-0.)],
        [(1,.25),(4,.5),(6,-.375)],[(0,-.5),(3,.25)],[(2,.125),(5,.25),(6,.625)]]
X = [[.25,-.5,.75],[-.125,.25,.5],[.5,.125,-.25],[-.25,.5,.125],[.75,-.125,.25],[.125,.75,-.5],[-.5,-.25,.375]]
W1 = [[.125,-.25,.375],[-.25,.125,.25],[.25,.125,-.125],[0.,0.,0.]]
B1 = [2.,-2.,1.5,0.]
W2 = [[.25,-.125,.5,.125],[-.375,.25,-.125,.25],[.125,.375,.25,-.125]]
B2 = [.125,-.25,.375]
Q = [[.6,.3,.1],[.1,.7,.2],[.2,.3,.5],[.4,.4,.2],[.1,.2,.7],[.5,.2,.3],[.25,.5,.25]]


def fixture():
    crow, col, values = [0], [], []
    for row in ROWS:
        col.extend(i for i,v in row)
        values.extend(v for i,v in row)
        crow.append(len(col))
    source = torch.sparse_csr_tensor(torch.tensor(crow,dtype=torch.int64),torch.tensor(col,dtype=torch.int64),
                                    torch.tensor(values,dtype=torch.float32),size=(7,7),device="cpu")
    return tuple(torch.tensor(v,dtype=torch.float32,device="cpu") for v in (W1,B1,W2,B2)), torch.tensor(X,dtype=torch.float32), source, torch.tensor(Q,dtype=torch.float64)


def evidence():
    value = dict(_guard=lambda: None, counts={}, operation_attempts={})
    CALL_CONTEXTS.append(value)
    return value


def dense_coefficient(source):
    # Independent small-toy oracle only; production reference never densifies.
    result = torch.zeros((7,7),dtype=torch.float64)
    crow,col,values=source.crow_indices(),source.col_indices(),source.values()
    for row in range(7):
        for index in range(int(crow[row]),int(crow[row+1])):
            result[row,int(col[index])]=values[index].double()
    return result


def dense_full(parameters,x,source,q):
    p=tuple(v.detach().double().clone().requires_grad_(True) for v in parameters)
    w1,b1,w2,b2=p
    u1=x.double()@w1.T;v1=source@u1;a=v1+b1;h=F.relu(a)
    u2=h@w2.T;v2=source@u2;logits=v2+b2;logp=F.log_softmax(logits,dim=1)
    loss=-(q*logp).sum(1).mean()
    all_grad=torch.autograd.grad(loss,p+(logits,u2,h,a,u1))
    return loss,dict(U1=u1,V1=v1,A=a,H=h,U2=u2,V2=v2,logits=logits,logp=logp,full_logp=logp,full_CE=loss,mask=a>0,
                     **dict(zip(m.COTANGENT_KEYS,all_grad[4:]))),all_grad[:4]


def close_record(actual,expected,section,label):
    maximum=float((actual.detach()-expected.detach()).abs().max())
    bound=2e-12+1e-12*float(expected.detach().abs().max())
    METRICS[section].append(dict(label=label,max_abs=maximum,bound=bound))
    assert maximum<=bound


def native_from(reference):
    boundaries={k:v.clone() if k in ("mask","full_CE") else v.float() for k,v in reference["boundaries"].items()}
    return dict(CE=reference["CE"].clone(),boundaries=boundaries,target=dict(blocks=m.PARAMETER_BLOCKS,rho=.001,
                gradients=tuple(v.float() for v in reference["gradients"]),source_norms=reference["source_norms"],delta=reference["delta"]))


def test_domain1_canonical_nonsymmetric_signed_zero_cast_and_actual_transpose():
    params,x,source,q=fixture();e=evidence()
    matrix,transposed,certificate=m._source(source,e)
    assert certificate["actual_transpose_coordinate_value_bijection"] and certificate["exact_coefficient_cast_lineage"]
    original=source.values().numpy();zeros=original==0
    assert np.array_equal(np.signbit(matrix.data[zeros]),np.signbit(original[zeros])) and np.signbit(original[zeros]).tolist()==[False,True]
    assert (matrix!=transposed).nnz>0 and matrix.indptr[1]==matrix.indptr[2]
    assert e["counts"]=={"stored_FP32_coefficient_casts":1,"independent_SciPy_CSR_constructions":1,"independent_actual_transpose_constructions":1}


def test_domain2_complete_operator_contiguous_strided_dense_and_duality():
    params,x,source,q=fixture();matrix,transposed,_=m._source(source,evidence());dense=dense_coefficient(source)
    for width in (3,5):
        for strided in (False,True):
            base=torch.arange(1,7*width*(2 if strided else 1)+1,dtype=torch.float64).reshape(7,-1)/100
            rhs=base[:,::2] if strided else base
            assert not strided or rhs.stride(1)==2 and not rhs.is_contiguous()
            leaf=rhs.detach().requires_grad_(True);e=evidence()
            output=m._sparse_apply(leaf,matrix,transposed,e)
            close_record(output,dense@rhs,"operator",str((width,strided)))
            cotangent=torch.arange(1,7*width+1,dtype=torch.float64).reshape(7,width)/70
            reverse=torch.autograd.grad(output,leaf,grad_outputs=cotangent)[0]
            close_record(reverse,dense.T@cotangent,"operator","T"+str((width,strided)))
            dot_error=abs(float((output*cotangent).sum()-(rhs*reverse).sum()))
            assert dot_error<=2e-12
            assert e["counts"]=={"independent_SciPy_source_products":1,"independent_SciPy_transpose_products":1}


def test_domain3_whole_CE_all_gradients_and_cotangents_vs_independent_dense():
    params,x,source,q=fixture();e=evidence();ref=m.full_ce_reference(params,x,source,q,e)
    loss,boundaries,gradients=dense_full(params,x,dense_coefficient(source),q)
    for key in set(ref["boundaries"])-{"mask"}:
        close_record(ref["boundaries"][key],boundaries[key],"full",key)
    close_record(ref["CE"],loss,"full","CE")
    for key,actual,expected in zip(m.PARAMETER_BLOCKS,ref["gradients"],gradients):
        close_record(actual,expected,"full",key)
    assert torch.equal(ref["boundaries"]["mask"],boundaries["mask"])
    assert e["counts"]["independent_SciPy_source_products"]==2 and e["counts"]["independent_SciPy_transpose_products"]==2
    assert e["counts"]["whole_torch_autograd_grad_calls"]==1
    METRICS["counters"]["base_whole_reference"]=e
    METRICS["counters"]["base_whole_reference"].pop("_guard")


def test_domain4_two_epsilon_FD_on_already_cast_FP64_smooth_subspace():
    params,x,source,q=fixture();ref=m.full_ce_reference(params,x,source,q,evidence())
    matrix,transposed,_=m._source(source,evidence());p0=tuple(v.detach().double() for v in params)
    for index,block in enumerate(p0):
        direction=torch.arange(1,block.numel()+1,dtype=torch.float64).reshape(block.shape)
        if index==0:direction[3]=0
        if index==1:direction[3]=0
        direction=direction/torch.linalg.vector_norm(direction.reshape(-1))
        analytical=float((ref["gradients"][index]*direction).sum())
        for eps in (1e-6,5e-7):
            pair=[]
            for sign in (1,-1):
                changed=list(p0);changed[index]=block+sign*eps*direction
                value,boundaries=m._forward64(tuple(changed),x.double(),matrix,transposed,q,evidence())
                assert torch.equal(boundaries["mask"],ref["boundaries"]["mask"])
                pair.append(float(value))
            numerical=(pair[0]-pair[1])/(2*eps);error=abs(numerical-analytical)
            METRICS["FD"].append(dict(block=m.PARAMETER_BLOCKS[index],eps=eps,analytical=analytical,numerical=numerical,abs_error=error))
            METRICS["masks"].append(dict(block=m.PARAMETER_BLOCKS[index],eps=eps,base_plus_minus_exact=True))
            assert error<=2e-8


def test_domain5_exact_zero_channel_positive_targets_and_nonpositive_rejection():
    params,x,source,q=fixture();ref=m.full_ce_reference(params,x,source,q,evidence())
    for key in ("A","H","E"):assert torch.count_nonzero(ref["boundaries"][key][:,3])==0
    assert torch.count_nonzero(ref["gradients"][0][3])==torch.count_nonzero(ref["gradients"][1][3])==torch.count_nonzero(ref["gradients"][2][:,3])==0
    assert all(float(v)>0 and torch.isfinite(v) for v in ref["source_norms"]+ref["delta"])
    dead=list(params);dead[0]=torch.zeros_like(params[0]);dead[1]=torch.full_like(params[1],-2)
    with pytest.raises(ValueError,match="norms"):
        m.full_ce_reference(tuple(dead),x,source,q,evidence())


def test_domain6_full_comparator_pass_fail_types_lengths_and_positive_deltas():
    params,x,source,q=fixture();ref=m.full_ce_reference(params,x,source,q,evidence());native=native_from(ref)
    result=m.compare_complete(native,ref);assert result["passed"]
    for edit in ("truncated","extra","dtype","nan","negative_delta","alias"):
        bad=copy.deepcopy(native)
        if edit=="truncated":bad["target"]["gradients"]=bad["target"]["gradients"][:3]
        elif edit=="extra":bad["target"]["delta"]=bad["target"]["delta"]+(torch.tensor(.1,dtype=torch.float64),)
        elif edit=="dtype":bad["boundaries"]["D"]=bad["boundaries"]["D"].double()
        elif edit=="nan":bad["boundaries"]["A"][0,0]=float("nan")
        elif edit=="negative_delta":bad["target"]["delta"]=(-torch.tensor(1e-30,dtype=torch.float64),)+bad["target"]["delta"][1:]
        else:bad["boundaries"]["full_CE"]=bad["boundaries"]["full_CE"]+1e-12
        with pytest.raises(ValueError):m.compare_complete(bad,ref)
    failed=copy.deepcopy(native);failed["target"]["gradients"]=failed["target"]["gradients"][:3]+(failed["target"]["gradients"][3]+.1,)
    assert m.compare_complete(failed,ref)["passed"] is False
    metrics=m.error_metrics(torch.ones(3,dtype=torch.float32),torch.zeros(3,dtype=torch.float64))
    assert metrics["relative_L2"] is None and metrics["absolute_L2"]>0 and len(metrics["absolute_error_quantiles"])==5
    near=copy.deepcopy(native);near["boundaries"]["mask"][0,3]=True
    result=m.compare_complete(near,ref);assert result["mask_mismatches_near"]==1 and result["raw_mask_mismatch_flatcoordinates"].tolist()==[3]
    assert result["near_zero_cells_filtered_from_derivatives"] is False


def test_domain7_source_independence_cast_raw_lineage_and_no_global_mutation():
    params,x,source,q=fixture();before=torch.random.get_rng_state().clone()
    identities=[v.detach().clone() for v in params+(x,q)]
    indices=(source.crow_indices().clone(),source.col_indices().clone(),source.values().clone())
    ref=m.full_ce_reference(params,x,source,q,evidence())
    for old,actual in zip(identities,params+(x,q)):assert torch.equal(old.view(torch.int32 if old.dtype==torch.float32 else torch.int64),actual.view(torch.int32 if actual.dtype==torch.float32 else torch.int64))
    for old,actual in zip(indices,(source.crow_indices(),source.col_indices(),source.values())):assert torch.equal(old.view(torch.int32) if old.dtype==torch.float32 else old,actual.view(torch.int32) if actual.dtype==torch.float32 else actual)
    assert torch.equal(before,torch.random.get_rng_state())
    assert set(ref["input_lineage"]["original"])=={"X","S","Q","anchor"} and set(ref["input_lineage"]["cast"])=={"X","S","T","Q","anchor"}
    tree=ast.parse(HELPER.read_text());imports=[ast.unparse(v) for v in ast.walk(tree) if isinstance(v,(ast.Import,ast.ImportFrom))]
    assert not any(any(word in value for word in ("csr_segment","csr_feature_tile","csr_explicit","finite_student_outer")) for value in imports)
    graph=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="full_ce_reference")
    assert ast.unparse(graph).count("torch.autograd.grad")==1 and "parameters + (logits, u2, h, a, u1)" in ast.unparse(graph)


def test_domain8_lifecycle_primary_error_partial_cache_stop_and_worker_AST(tmp_path,monkeypatch):
    e=evidence()
    with pytest.raises(InterruptedError):m._call(e,"partial",lambda:(_ for _ in ()).throw(InterruptedError("original")))
    assert e["operation_attempts"]=={"partial":1} and e["counts"]=={}
    path=tmp_path/"partial.pt";path.write_bytes(b"rawpartial");observed={};m._observe_raw(path,observed,"reference_cache")
    assert observed["reference_cache_observed_exists"] and observed["reference_cache_observed_bytes"]==10
    output=tmp_path/"receipt.json";raw=tmp_path/"reference.pt";details=tmp_path/"details.pt"
    spec=dict(output_path=str(output),reference_cache_path=str(raw),comparison_details_path=str(details),source={},numerical_source={},python_version="literal",scientific_preregistration={},fixed={},files_sha256={})
    science=dict(counts_contract={"zero_work":{},"scope":"actual"},cpu_policy={"peak_process_RSS_budget_bytes":17179869184})
    monkeypatch.setattr(m,"_load_spec",lambda *a:(spec,science,tmp_path))
    monkeypatch.setattr(m,"_preserve",lambda *a:None)
    monkeypatch.setattr(m,"_libraries",lambda:(_ for _ in ()).throw(InterruptedError("prelibs")))
    with pytest.raises(RuntimeError,match="Terminal"):
        m.prepare_independent_reference(tmp_path/"spec.json","h"*64,output)
    receipt=json.loads(output.read_text());assert receipt["error_type"]=="InterruptedError" and receipt["success_counts"] is None and not receipt["passed"]
    assert "counts" in receipt and "reference_cache_observed_exists" in receipt
    stop_output=tmp_path/"stop_receipt.json"
    stop_spec=dict(spec,output_path=str(stop_output),reference_cache_path=str(tmp_path/"stop_reference.pt"),comparison_details_path=str(tmp_path/"stop_details.pt"))
    library_calls=[]
    monkeypatch.setattr(m,"_load_spec",lambda *a:(stop_spec,science,tmp_path))
    monkeypatch.setattr(m,"_libraries",lambda:library_calls.append(True))
    with pytest.raises(ValueError,match="Stopped before CPU libraries"):
        m.prepare_independent_reference(tmp_path/"spec.json","h"*64,stop_output,stop=lambda:True)
    stopped=json.loads(stop_output.read_text())
    assert library_calls==[] and stopped["counts"]=={} and stopped["operation_attempts"]=={}
    assert stopped["error_type"]=="ValueError" and stopped["error"]=="Stopped before CPU libraries" and stopped["success_counts"] is None and not stopped["passed"]
    assert stopped["reference_cache_observed_exists"] is False and stopped["comparison_details_observed_exists"] is False
    monkeypatch.setattr(m,"_load_spec",lambda *a:(_ for _ in ()).throw(InterruptedError("prebind")))
    with pytest.raises(RuntimeError,match="before spec binding"):
        m.prepare_independent_reference(tmp_path/"spec.json","h"*64,tmp_path/"never.json")
    worker=HELPER.with_name("research_loop.py");tree=ast.parse(worker.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="dispatch")
    selected=[n for n in fn.body if isinstance(n,ast.If) and "'large_scipy_full_ce_reference'" in ast.unparse(n.test)]
    assert len(selected)==1 and ast.unparse(selected[0].body[-1])=="return prepare_independent_reference(**options, stop=stop)"
    fn.body.remove(selected[0])
    assert hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest()==OLD_WORKER_AST_SHA256


@pytest.fixture(scope="module",autouse=True)
def report_metrics():
    yield
    totals = {name: {} for name in ("counts", "operation_attempts")}
    for context in CALL_CONTEXTS:
        for name in totals:
            for key, value in context[name].items():
                totals[name][key] = totals[name].get(key, 0) + value
    METRICS["actual_interface_totals"] = totals
    METRICS["instrumentation_scope"] = "Every independent SciPy core/FD/operator call plus deliberate dead-target and partial-interface controls; separate dense toy oracle arithmetic is not a real source operation."
    if os.environ.get("BZ_PROOF_METRICS"):
        Path(os.environ["BZ_PROOF_METRICS"]).write_text(json.dumps(METRICS,indent=2,allow_nan=False)+"\n")
