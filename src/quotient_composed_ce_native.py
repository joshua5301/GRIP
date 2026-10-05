"""FR original CSR coupled quotient features and complete FP64 row cotangent.

Original public quotient/map operators remain unchanged. The CPU held comparator
is a separate fresh-P-leaf implementation using original sparse products/map.
"""
import math
import torch
from src.coarsening import feature_centroids, quotient_adjacency
from src.moments import augmented
from src.nystrom_ce import NystromMap
from src.soft_ce_partition import head_gradient
from src.teacher import EPS
from src.transforms import FeatureTransform
from src.quotient_composed_ce_native_source import require

MODE = 'original_ROW_CSR_quotient_SGC_original_Nystrom_composed_uniform_CE_native_FIRST0to1_v1'


def observe(callback, key, value):
    if callback is not None: callback(key,value)
    return value


def matrix(value, shape=None):
    require(torch.is_tensor(value) and value.layout==torch.strided and value.dtype==torch.float64
        and value.ndim==2 and (shape is None or tuple(value.shape)==tuple(shape)) and bool(torch.isfinite(value).all()),'Finite FP64 matrix domain')
    return value


def clamp_observation(h, anchors):
    # Literal arithmetic and comparisons of the unchanged original relu teacher.
    # This is branch evidence, not another NystromMap/kernel API invocation.
    na_raw=h.norm(dim=1,keepdim=True);nb_raw=anchors.norm(dim=1).unsqueeze(0)
    na=na_raw.clamp(min=EPS);nb=nb_raw.clamp(min=EPS)
    pre=(h@anchors.T)/(na*nb);cos=pre.clamp(-1+EPS,1-EPS);th=torch.acos(cos)
    post=(torch.sin(th)+(math.pi-th)*torch.cos(th))/math.pi
    def branches(v,low,high):
        return dict(low=v<low,equal_low=v==low,interior=(v>low)&(v<high),equal_high=v==high,high=v>high)
    return dict(raw_norms_A=na_raw,raw_norms_B=nb_raw,raw_pre_cosine=pre,raw_post_relu=post,
        masks=dict(norm_A=dict(low=na_raw<EPS,equal=na_raw==EPS,above=na_raw>EPS),
            norm_B=dict(low=nb_raw<EPS,equal=nb_raw==EPS,above=nb_raw>EPS),
            pre_cosine=branches(pre,-1+EPS,1-EPS),post_relu=branches(post,-1+EPS,1-EPS)))


def build_quotient_features(P,X,Q,S,transform,feature_map,observe_return=None,map_leaf=False):
    matrix(P);matrix(X);matrix(Q);n,k=P.shape
    require(S.layout==torch.sparse_csr and S.dtype==torch.float64 and tuple(S.shape)==(n,n)
        and all(v.device==P.device for v in (X,Q,S)) and bool(torch.isfinite(S.values()).all()) and bool((S.values()>=0).all()),'Original CSR domain')
    require(bool((P>0).all()) and torch.allclose(P.sum(1),P.new_ones(n),atol=1e-12,rtol=1e-12)
        and bool((Q>=0).all()) and bool((Q.sum(1)>0).all()) and bool((Q.sum(0)>0).all()),'Probability/general-Q domain')
    require(type(transform) is FeatureTransform and transform.kind=='rms' and transform.matrix is None and transform.eps==EPS
        and type(feature_map) is NystromMap and feature_map.kernel=='relu','Original frozen RMS/map')
    Xc,mass=feature_centroids(P,X);observe(observe_return,'Xc_mass',(Xc,mass))
    Qc,target_mass=feature_centroids(P,Q);observe(observe_return,'Qc_mass',(Qc,target_mass))
    Sc=quotient_adjacency(P,S,normalization='symmetric',mass_scaling=False,self_loops='retain',chunk_size=k);observe(observe_return,'Sc',Sc)
    SP=torch.sparse.mm(S,P);C=P.T@SP;degree=C.sum(1);observe(observe_return,'C_degree_SP',(C,degree,SP))
    Y=Sc@Xc;Hc=Sc@Y;zc=transform(Hc);observe(observe_return,'two_Sc_RMS',(Y,Hc,zc))
    h=zc*transform.scale+transform.output_center+transform.center
    if map_leaf:h=h.detach().clone().requires_grad_(True)
    observe(observe_return,'map_input',h)
    clamps=clamp_observation(h,feature_map.anchors);observe(observe_return,'clamp_observation',clamps)
    mapped=feature_map(h);observe(observe_return,'map_return',mapped)
    features=torch.cat((zc,mapped),1)
    result=dict(n=mass,Xc=Xc,Qc=Qc,C=C,degree=degree,Sc=Sc,SP=SP,Y=Y,Hc=Hc,zc=zc,map_input=h,mapped=mapped,features=features,clamps=clamps)
    observe(observe_return,'quotient_features',result)
    require(torch.equal(mass,target_mass) and bool((mass>0).all()) and bool((degree>0).all())
        and all(bool(torch.isfinite(v).all()) for v in result.values() if torch.is_tensor(v)),'Returned quotient/map domain')
    return result


def complete_row_cotangent(P,X,Q,S,transform,feature_map,theta,vector,penalty,CE0,observe_return=None):
    """Complete target/mass/map/degree/two-Sc/both-P chain, then CE0 once."""
    require(math.isfinite(CE0) and CE0>0 and math.isfinite(penalty) and penalty>0,'Own positive CE0/ridge')
    value=build_quotient_features(P.detach(),X,Q,S,transform,feature_map,observe_return,map_leaf=True)
    f=value['features'].detach();q=value['Qc'];k=len(q);d=X.shape[1]
    matrix(theta,(Q.shape[1],f.shape[1]+1));matrix(vector,theta.shape)
    x=augmented(f);p=(x@theta.T).softmax(1);sums=q.sum(1,keepdim=True);err=sums*p-q
    a=x@vector.T;pa=(p*a).sum(1,keepdim=True)
    gf=-(err@vector[:,:-1]+(sums*p*(a-pa))@theta[:,:-1])/k;gq=(a-pa)/k
    observe(observe_return,'head_feature_target_cotangents',dict(gf=gf,gq=gq))
    map_vjp=torch.autograd.grad(value['mapped'],value['map_input'],grad_outputs=gf[:,d:],retain_graph=False)[0]
    observe(observe_return,'original_clamped_map_VJP',map_vjp)
    gh=(gf[:,:d]+transform.scale*map_vjp)/transform.scale
    Sc,Xc,C,degree=value['Sc'],value['Xc'],value['C'],value['degree'];gy=Sc.T@gh;gx=Sc.T@gy
    gsc=gh@value['Y'].T+gy@Xc.T;aa=degree.rsqrt()
    ga=(gsc*C*aa[None,:]).sum(1)+(gsc*aa[:,None]*C).sum(0);gd=-.5*degree.pow(-1.5)*ga
    gc=aa[:,None]*gsc*aa[None,:]+gd[:,None]
    rc=(X@gx.T-(Xc*gx).sum(1)[None,:])/value['n'][None,:]
    rc=rc+(Q@gq.T-(value['Qc']*gq).sum(1)[None,:])/value['n'][None,:]
    STP=torch.sparse.mm(S.transpose(0,1),P);rg=value['SP']@gc.T+STP@gc
    raw=rc+rg;complete=raw/CE0
    result=dict(raw_R=raw,complete_R=complete,owned_chain=value,components=dict(gf=gf,gq=gq,map_vjp=map_vjp,gh=gh,gy=gy,gx=gx,gsc=gsc,ga=ga,gd=gd,gc=gc,Rcent=rc,Rgraph=rg,STP=STP))
    observe(observe_return,'complete_row_result',result);matrix(raw,P.shape);matrix(complete,P.shape);return result


def independent_CPU_complete_row(P,X,Q,S,transform,feature_map,theta,vector,penalty,CE0,observe_return=None):
    """Fresh CPU original-CSR/original-clamped-map autograd, no head/adjoint fit."""
    require(all(v.device.type=='cpu' for v in (P,X,Q,S,theta,vector)) and S.layout==torch.sparse_csr,'CPU original CSR reference domain')
    variable=P.detach().clone().requires_grad_(True);observe(observe_return,'CPU_P_leaf',variable)
    Xc,mass=feature_centroids(variable,X);observe(observe_return,'CPU_Xc_mass',(Xc,mass))
    Qc,target_mass=feature_centroids(variable,Q);observe(observe_return,'CPU_Qc_mass',(Qc,target_mass))
    Sc=quotient_adjacency(variable,S,normalization='symmetric',mass_scaling=False,self_loops='retain',chunk_size=variable.shape[1]);observe(observe_return,'CPU_Sc',Sc)
    Hc=Sc@(Sc@Xc);zc=transform(Hc);h=zc*transform.scale+transform.output_center+transform.center
    observe(observe_return,'CPU_Hc_zc_map_input',dict(Hc=Hc,zc=zc,map_input=h))
    clamps=clamp_observation(h,feature_map.anchors);observe(observe_return,'CPU_clamp_observation',clamps)
    mapped=feature_map(h);observe(observe_return,'CPU_map_return',mapped);features=torch.cat((zc,mapped),1)
    gradient=head_gradient(augmented(features),Qc,features.new_full((len(Qc),),1/len(Qc)),theta.detach(),penalty)
    observe(observe_return,'CPU_fixed_head_gradient',gradient)
    raw=torch.autograd.grad(-(gradient*vector.detach()).sum(),variable)[0];complete=raw/CE0
    result=dict(raw_R=raw,complete_R=complete,clamps=clamps,owned_chain=dict(P=variable,Xc=Xc,Qc=Qc,mass=mass,target_mass=target_mass,Sc=Sc,Hc=Hc,zc=zc,map_input=h,mapped=mapped,features=features,gradient=gradient))
    observe(observe_return,'CPU_complete_row_result',result);matrix(raw,P.shape);matrix(complete,P.shape);return result
