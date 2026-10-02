"""New native operator interface/counter AST mocks; no numeric suite replay."""
import ast
import copy
import types
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "src/csr_segment_adjoint.py"


class Tensor:
    def __init__(self, shape, finite=True):
        self.shape, self.finite = shape, finite
        self.device, self.dtype, self.layout = types.SimpleNamespace(type="cuda", index=0), "float32", "strided"
        self.ndim, self.requires_grad = len(shape), False

    def index_select(self, axis, columns):
        assert axis == 0 and columns == [0, 1, 2, 0]
        return Tensor((4, self.shape[1]))

    def unsqueeze(self, axis):
        assert axis == 1
        return Tensor((4, 1))


class RowPtr(list):
    def __getitem__(self, key):
        value = super().__getitem__(key)
        return RowPtr(value) if isinstance(key, slice) else value

    def __sub__(self, other):
        return [a-b for a,b in zip(self,other)]


class CSR(Tensor):
    def __init__(self): super().__init__((3,3))
    def col_indices(self): return [0,1,2,0]
    def crow_indices(self): return RowPtr([0,2,3,4])
    def values(self): return Tensor((4,))


def require(value, message):
    if not value: raise ValueError(message)


def product_api(nonfinite_result=False):
    tree = ast.parse(PATH.read_text())
    fn = copy.deepcopy(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == "_product"))
    fn.decorator_list=[]
    def begin(e,key): e['attempts'][key]=e['attempts'].get(key,0)+1
    def end(e,key): e['counts'][key]=e['counts'].get(key,0)+1
    def invoke(e,key,f,*args,**kwargs):
        begin(e,key)
        out=f(*args,**kwargs)
        end(e,key)
        return out
    def multiply(value,weights):
        assert value.shape[0]==weights.shape[0]==4 and weights.shape[1]==1
        return Tensor(value.shape)
    def segment(value,reduce,**kw):
        assert value.ndim==2 and reduce=='sum' and kw==dict(axis=0,lengths=[2,1,1],initial=0,unsafe=False)
        return Tensor((3,value.shape[1]),finite=not nonfinite_result)
    torch=types.SimpleNamespace(Tensor=Tensor,strided="strided",mul=multiply,segment_reduce=segment,
                                isfinite=lambda v:types.SimpleNamespace(all=lambda:v.finite))
    ns=dict(torch=torch,_begin=begin,_end=end,_invoke=invoke,_require=require,_csr=lambda s:None)
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(PATH),'exec'),ns)
    return ns['_product']


def test_four_declared_segment_products_use_rank2_safe_operator_and_counts():
    api=product_api();e=dict(attempts={},counts={})
    for width,key in ((256,'segment_original_source_products'),(7,'segment_original_source_products'),
                      (7,'segment_explicit_transpose_products'),(256,'segment_explicit_transpose_products')):
        result=api(CSR(),Tensor((3,width)),evidence=e,key=key)
        assert result.shape==(3,width)
    expected={'segment_original_source_products':2,'segment_explicit_transpose_products':2,
              'rank2_weighted_segment_edge_gather_calls':4,'rank2_FP32_edge_multiplication_calls':4,'rank2_segment_SUM_calls':4}
    assert e['counts']==e['attempts']==expected


def test_returned_nonfinite_segment_keeps_primitive_completion_before_reject():
    e=dict(attempts={},counts={})
    with pytest.raises(ValueError,match='row-segment result'):
        product_api(nonfinite_result=True)(CSR(),Tensor((3,1)),evidence=e,key='segment_original_source_products')
    assert e['counts']['rank2_segment_SUM_calls']==1 and e['counts']['rank2_weighted_segment_edge_gather_calls']==1
    assert e['attempts']['segment_original_source_products']==1 and 'segment_original_source_products' not in e['counts']


def test_exact_four_route_calls_and_no_original_model_sparse_mm():
    tree=ast.parse(PATH.read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='gradient_boundaries')
    calls=[n for n in ast.walk(fn) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='_product']
    assert len(calls)==4
    assert [ast.unparse(n.args[0]) for n in calls]==['transposed','transposed','source','source'] or sorted(ast.unparse(n.args[0]) for n in calls)==['source','source','transposed','transposed']
    labels=[next(k.value.value for k in n.keywords if k.arg=='key') for n in calls]
    assert labels.count('segment_original_source_products')==labels.count('segment_explicit_transpose_products')==2
    text=ast.unparse(tree)
    assert 'torch.sparse.mm' not in text and '.squeeze(' not in text and '.to_dense(' not in text and '.mul_(' not in text


def test_target_packet_transpose_and_validators_direct_protected_aliases():
    tree=ast.parse(PATH.read_text())
    imported=[n for n in tree.body if isinstance(n,ast.ImportFrom) and n.module=='src.csr_explicit_adjoint']
    assert {'_csr','_inputs','transpose_csr','target_packet'} <= {a.name for n in imported for a in n.names}
    assert not any(isinstance(n,ast.FunctionDef) and n.name in ('target_packet','transpose_csr','_csr') for n in tree.body)
