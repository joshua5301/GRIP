"""FR typed owning original ROW/CSR source and strict simultaneous storage budget.

The two signed mmap payloads remain retained aliases, never whole-tree clones.
Historical FA/FD producer validation belongs to the qualifier before construction.
"""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

import torch
from src.dual_head_ce import _native_options, _transform, _map, _factor_digests
from src.io import array_digest
from src.joint_mean_ce import _seal
from src.kernel_mean_ce import _cpu
from src.nystrom_ce import NystromMap, _content_digest
from src.shared_features import _tensor_identity
from src.transforms import FeatureTransform

MODE = 'original_ROW_CSR_quotient_SGC_original_Nystrom_composed_uniform_CE_native_FIRST0to1_v1'


def require(ok, message):
    if not ok: raise ValueError(message)


def tensors(value):
    if torch.is_tensor(value): yield value
    elif isinstance(value, dict):
        for child in value.values(): yield from tensors(child)
    elif isinstance(value, (list, tuple)):
        for child in value: yield from tensors(child)


def detached(value):
    if torch.is_tensor(value): return value.detach()
    if isinstance(value, dict): return {k:detached(v) for k,v in value.items()}
    if isinstance(value, (tuple,list)): return [detached(v) for v in value]
    return value


def storage_bytes(value, cpu_only=False):
    seen={}
    for tensor in tensors(value):
        fields=(tensor.crow_indices(),tensor.col_indices(),tensor.values()) if tensor.layout==torch.sparse_csr else (tensor,)
        for field in fields:
            if not cpu_only or field.device.type=='cpu':
                storage=field.untyped_storage();seen[(str(field.device),storage.data_ptr())]=storage.nbytes()
    return sum(seen.values())


def copy_bytes(value):
    return sum((v.crow_indices().numel()*8+v.col_indices().numel()*8+v.values().numel()*v.element_size())
               if v.layout==torch.sparse_csr else v.numel()*v.element_size() for v in tensors(value))


def identity(value):
    if torch.is_tensor(value) and value.layout==torch.sparse_csr:
        return dict(shape=list(value.shape),dtype=str(value.dtype),layout=str(value.layout),
            crow=_tensor_identity(value.crow_indices()),col=_tensor_identity(value.col_indices()),values=_tensor_identity(value.values()))
    return _tensor_identity(value)


def metadata(value):
    if torch.is_tensor(value): return identity(value)
    if isinstance(value,dict): return {k:metadata(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [metadata(v) for v in value]
    return value


def cache_ids(value):
    return dict(graph=metadata(value['graph']),masks=metadata(value['masks']),H=identity(value['H']),Q=identity(value['Q']),arms=metadata(value['arms']))


class ActiveStorageBudget:
    def __init__(self, saved, live, check, resource_policy):
        self.saved,self.live,self.check=saved,live,check;self.policy=resource_policy
        self.reserve=resource_policy['ordinary_reserve_bytes'];self.peak=0;self.observations=[];self.phases=[];self.phase=None

    def charge(self, role, future=0):
        cuda=torch.cuda.memory_allocated(0) if torch.cuda.is_initialized() else 0
        cpu=storage_bytes((self.saved,self.live),True);total=cpu+cuda+self.reserve+self.policy['hash_scratch_bytes']+future
        row=dict(role=role,CPU_live_unique_bytes=cpu,CUDA_live_bytes=cuda,future_CPU_copy_bytes=future,
            reserve_bytes=self.reserve,hash_scratch_bytes=self.policy['hash_scratch_bytes'],charged_bytes=total)
        self.observations.append(row);self.peak=max(self.peak,total)
        if self.phase is not None:self.phase['CPU_copy_upper_bytes']=max(self.phase['CPU_copy_upper_bytes'],cpu+future)
        require(total<=self.policy['active_source_material_copy_storage_bytes_max'],'FR simultaneous active storage cap');self.check();return total

    def own(self, key, value):
        # Raw detached aliases enter serialization/live trees before any assertion.
        self.live[key]=value;self.saved[key]=detached(value)
        self.charge('before_own_'+key,2*copy_bytes(value))
        result=_cpu(value);self.saved[key]=result;self.charge('after_own_'+key);return result

    @contextmanager
    def solver(self, role, returned_copy_bytes):
        require(self.phase is None and self.reserve==self.policy['ordinary_reserve_bytes'],'No nested/adaptive solver reservation')
        event=dict(role=role,returned=False,completed=False,CPU_copy_upper_bytes=0,error=None,restored64=False)
        self.phases.append(event);self.phase=event;failure=None
        try:
            event['preceding64_charge_bytes']=self.charge('solver_before64_'+role,returned_copy_bytes)
            event.update(allocated_before=torch.cuda.memory_allocated(0),max_allocated_before=torch.cuda.max_memory_allocated(0),max_reserved_before=torch.cuda.max_memory_reserved(0))
            self.reserve=self.policy['solver_reserve_bytes'];event['precall_charge_bytes']=self.charge('solver_entry_'+role,returned_copy_bytes)
            yield event
            event.update(max_allocated_after=torch.cuda.max_memory_allocated(0),max_reserved_after=torch.cuda.max_memory_reserved(0))
            event['managed_increment_upper_bytes']=event['max_allocated_after']-event['allocated_before']
            event['managed_plus_opaque_bytes']=event['managed_increment_upper_bytes']+self.policy['ordinary_reserve_bytes']
            event['post_own_charge_bytes']=self.charge('solver_post_own_'+role)
            event['observed_active_upper_bytes']=event['max_allocated_after']+event['CPU_copy_upper_bytes']+self.policy['ordinary_reserve_bytes']+self.policy['hash_scratch_bytes']
            require(event['managed_plus_opaque_bytes']<=self.policy['solver_reserve_bytes'],'Fixed solver managed peak plus opaque cap')
            require(event['observed_active_upper_bytes']<=self.policy['active_source_material_copy_storage_bytes_max'],'Solver cumulative peak plus all CPU/copy cap');event['completed']=True
        except BaseException as error:failure=error;event['error']=dict(type=type(error).__name__,message=str(error));raise
        finally:
            if torch.cuda.is_initialized():event.update(max_allocated_after=torch.cuda.max_memory_allocated(0),max_reserved_after=torch.cuda.max_memory_reserved(0),allocated_after=torch.cuda.memory_allocated(0))
            self.reserve=self.policy['ordinary_reserve_bytes'];self.phase=None;event['restored64']=True
            try:event['restore_charge_bytes']=self.charge('solver_restore64_'+role)
            except BaseException as error:
                event['restore_error']=dict(type=type(error).__name__,message=str(error))
                if failure is None:raise


class OwnOriginalROWQuotientSource:
    def __init__(self, FA153_capture, WHOLE_FD159_cache, selected_source_contract, original_options):
        self.capture,self.cache=FA153_capture,WHOLE_FD159_cache;self.contract=selected_source_contract
        self.case=selected_source_contract['case'];self.options=dict(original_options)
        C=self.case;W=selected_source_contract['WHOLE_old_cache'];c=FA153_capture['captured'];d=C['dimensions'];n,k,r=(d[x] for x in ('N','K','rank'))
        header={key:WHOLE_FD159_cache[key] for key in W['literal_header']}
        require(set(WHOLE_FD159_cache)==set(W['expected_payload_keys']) and header==W['literal_header'],'Literal whole FD159 cache header/keys')
        require(cache_ids(WHOLE_FD159_cache)==WHOLE_FD159_cache['identities']==W['ALL5_identities'] and _seal(cache_ids(WHOLE_FD159_cache))==W['ALL5_identities_seal'],'Whole five-arm/shared identity domain')
        require(WHOLE_FD159_cache['upstream_providers']==W['literal_upstream_providers'] and _seal(WHOLE_FD159_cache['upstream_providers'])==W['upstream_providers_seal'],'Whole private provider tree')
        private=W['private_owning_metadata'];actual_private=metadata({key:WHOLE_FD159_cache[key] for key in ('owning_physical_endpoints','owning_native_source','own_E25_parameters','owning_raw_H_decode')})
        require(actual_private==WHOLE_FD159_cache[private['field']]==private['expected_metadata'] and _seal(actual_private)==private['canonical_JSON_seal'],'Whole private owning export identity')
        expected=C['FA_expected_header'];require(all(FA153_capture[key]==expected[key] for key in ('schema','kind','budget','scientific_contract'))
            and FA153_capture['source']==selected_source_contract['FA_literal_producer'],'FA153 literal payload header')
        self.z,self.Q,self.hard=c['loaded_inputs']['z'],c['raw_Q'],c['loaded_hard'];self.native=c['loaded_native']
        self.H,self.Phi=c['loaded_H']['h'],c['original_Phi'];self.X,self.S=WHOLE_FD159_cache['graph']['x'],WHOLE_FD159_cache['graph']['adj']
        td=c['loaded_inputs']['transform'];self.transform=FeatureTransform(**td)
        self.feature_map=NystromMap(c['loaded_map']['anchors'],c['loaded_map']['mapping'],'relu')
        assets=dict(H=identity(self.H),z=identity(self.z),Q=identity(self.Q),assignment=identity(self.hard),transform=_transform(self.transform,d['D'],self.z.device),
            anchors=identity(self.feature_map.anchors),mapping=identity(self.feature_map.mapping),Phi_identity=C['asset_descriptors']['Phi_identity'])
        require(assets==C['asset_descriptors'],'Exact original FA153 source descriptors')
        require(_content_digest(self.Phi,canonical_double=True)==assets['Phi_identity']['phi_digest'],'Exact owning original Phi bytes')
        require(metadata(WHOLE_FD159_cache['graph'])==C['graph_X_S_descriptors'] and identity(WHOLE_FD159_cache['H'])==assets['H']
            and identity(WHOLE_FD159_cache['Q'])==assets['Q'],'Original ROW CSR and FA H/Q linkage')
        require(self.S.layout==torch.sparse_csr and self.S.dtype==torch.float32 and self.X.dtype==torch.float32,'Original ROW FP32 CSR domain')
        require(self.hard.dtype==torch.int64 and self.hard.shape==(n,) and int(self.hard.min())>=0 and int(self.hard.max())==k-1,'Original hard coverage')
        require(_native_options(dict(self.options,save_resume=True))==C['original_options'] and self.native['data_digest']==C['original_data_digest']
            and self.native['factor_seed']==self.options['factor_seed'] and self.native['mixing']==self.options['mixing'],'Original full native options/header')
        pair=_factor_digests((self.native['u'],self.native['v']),n,k,r,device=torch.device('cpu'))
        require(pair==C['native_initial_parameter_digests'] and bool(self.native['u'].eq(0).all()),'Original native U0/V0')
        require(array_digest(self.z.numpy(),self.Q.numpy(),self.hard.numpy())==C['original_data_digest'],'Original normalized input digest')
        require(selected_source_contract['graph_family_dataset_digest']==selected_source_contract['selected_graph_family_digest'],'Authenticated original graph family digest')
        self.assets=assets;self.parameter_digests=pair
        self.descriptor=dict(schema=1,kind=MODE,current_source_full_manifest=selected_source_contract['current_source'],
            FA_literal_producer=selected_source_contract['FA_literal_producer'],graph_literal_producer=W['literal_header']['source'],
            original_data_digest=C['original_data_digest'],graph_family_dataset_digest=selected_source_contract['graph_family_dataset_digest'],
            all_source_pin_map=selected_source_contract['stable_pins'],graph_X_CSR_indices_values_identities=C['graph_X_S_descriptors'],
            FA_z_Q_H_Phi_map_RMS_native_identities=assets,original29_options=self.options,native_initial_parameter_digests=pair)
        self.descriptor['source_owner_digest']=_seal(self.descriptor);self.on_device={};self.outer=None

    def device_inputs(self, device, budget):
        require(not self.on_device,'One owning source device conversion')
        for key,value,dtype in [('X',self.X,torch.float64),('Q',self.Q,torch.float64),('z',self.z,torch.float64),('hard',self.hard,torch.int64),
            ('U0',self.native['u'],torch.float32),('V0',self.native['v'],torch.float32),('anchors',self.feature_map.anchors,torch.float64),('mapping',self.feature_map.mapping,torch.float64)]:
            budget.charge('source_conversion_'+key,copy_bytes(value));self.on_device[key]=value.to(device=device,dtype=dtype);budget.live['source_device']=self.on_device;budget.charge('source_converted_'+key)
        self.on_device['S']=self.S.to(device=device,dtype=torch.float64);budget.live['source_device']=self.on_device;budget.charge('source_CSR_converted')
        fields={key:(value.to(device) if torch.is_tensor(value) else value) for key,value in vars(self.transform).items()}
        self.on_device['transform']=fields;budget.live['source_device']=self.on_device;budget.charge('source_RMS_converted')
        # Original source outer rows are fixed [z,Phi], with no source-map invocation.
        budget.charge('before_source_outer_concat',self.z.numel()*8+self.Phi.numel()*8)
        self.outer=torch.cat((self.z,self.Phi),1).contiguous().numpy();self.outer.setflags(write=False)
        budget.live['source_outer']=torch.from_numpy(self.outer);budget.charge('source_outer_owned')
        return self.on_device

    def origin_context(self, endpoint, CE0):
        require(type(CE0) is float and CE0>0,'Positive own CE0')
        context=dict(schema=1,mode=MODE,source_owner=self.descriptor,new_quotient_P0_chain_identity=metadata(endpoint['chain']),
            new_P0_identity=identity(endpoint['P']),new_head0_identity=identity(endpoint['theta']),new_positive_CE0=CE0)
        context['own_origin_digest']=_seal(context);return context


def validate_context(context):
    require(context['mode']==MODE and context['own_origin_digest']==_seal({k:v for k,v in context.items() if k!='own_origin_digest'}),'New quotient context seal')
    owner=context['source_owner'];require(owner['source_owner_digest']==_seal({k:v for k,v in owner.items() if k!='source_owner_digest'}),'Stable source owner seal')
    return context


def validate_qualification_state(state, expected_owner, expected_context):
    """Pure seals and actual tensor identities; never current legacy _files checks."""
    validate_context(expected_context)
    require(state['schema']==1 and state['mode']==MODE and state['source_owner']==expected_owner
        and state['context']==expected_context and state['current_source']==expected_owner['current_source_full_manifest']
        and state['endpoint1']['step']==1 and state['endpoint1']['CE0']==expected_context['new_positive_CE0'],'Typed own E1 state/context')
    body=dict(context=state['context'],endpoint=metadata(state['endpoint1']),parameters=metadata(state['native_parameters']),counts=state['counts'])
    # Original Adam slot keys are integers in PT and strings in JSON projection.
    body=json.loads(json.dumps(body,allow_nan=False))
    require(state['state_metadata_seal']==_seal(body),'Typed own E1 complete state seal')
    return state
