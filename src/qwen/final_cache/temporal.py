"""Temporal ownership inputs with explicit observation and causal contracts.

Inputs are predictions, never reference masks. Hidden removes the entire center
prediction. Corrupted exchanges actors in a reproducible, label-blind rectangle.
"""
import hashlib
import numpy as np
import torch
from torch.nn import functional as F


def source_indices(rows,index,causal=False,max_gap=4.):
    r=rows[index]
    keep=[i for i,x in enumerate(rows) if x['clip']==r['clip'] and x['shot']==r['shot']
          and 0<abs(x['time']-r['time'])<=max_gap]
    past=sorted([i for i in keep if rows[i]['time']<r['time']],key=lambda i:rows[i]['time'],reverse=True)
    future=sorted([i for i in keep if rows[i]['time']>r['time']],key=lambda i:rows[i]['time'])
    if causal:return (past+[None,None])[:2]
    return [past[0] if past else None,future[0] if future else None]


def corrupt(x,seed):
    rng=np.random.default_rng(seed);h,w=x.shape[-2:];y=int(rng.integers(h//4,max(h//4+1,h//2)));z=int(rng.integers(w//4,max(w//4+1,w//2)))
    out=x.clone();hh=max(1,h//3);ww=max(1,w//3)
    out[1,y:y+hh,z:z+ww]=x[2,y:y+hh,z:z+ww]
    out[2,y:y+hh,z:z+ww]=x[1,y:y+hh,z:z+ww]
    return out


def temporal_input(rows,predictions,index,condition='observed',causal=False,order='ordered',seed=7):
    r=rows[index];center=predictions[r['id']]
    if condition=='hidden':center=torch.zeros_like(center)
    elif condition=='corrupted':center=corrupt(center,seed+r['id'])
    elif condition!='observed':raise ValueError('Invalid observation condition')
    ids=source_indices(rows,index,causal)
    original_ids=list(ids)
    if order=='reversed':ids=list(reversed(ids))
    elif order=='shuffled':
        # Permute the exact same bounded neighbor set; do not confound order with displacement.
        rng=np.random.default_rng(seed+r['id']);rng.shuffle(ids)
    elif order!='ordered':raise ValueError('Invalid temporal order')
    source=[torch.zeros_like(center) if i is None else predictions[rows[i]['id']] for i in ids]
    dt=[0. if i is None else (rows[i]['time']-r['time'])/4. for i in ids]
    flags=torch.tensor([float(condition!='hidden'),*dt],device=center.device,dtype=center.dtype)[:,None,None].expand_as(center)
    return torch.cat([center,*source,flags]),{'source_ids':[None if i is None else rows[i]['id'] for i in ids],
          'source_times':[None if i is None else rows[i]['time'] for i in ids],
          'order_changed':ids!=original_ids,'time_deltas_scaled':dt,
          'condition':condition,'causal':causal,'order':order,'center_features_used':condition!='hidden'}


def baseline(rows,predictions,index,method,causal=False):
    r=rows[index];ids=source_indices(rows,index,causal);past,future=ids
    if method=='static':return predictions[r['id']]
    if method=='persistence':
        return predictions[rows[past]['id']] if past is not None else torch.zeros_like(predictions[r['id']])
    if method!='interpolation':raise ValueError('Invalid temporal baseline')
    if past is None and future is None:return torch.zeros_like(predictions[r['id']])
    if future is None:return predictions[rows[past]['id']]
    if past is None:return predictions[rows[future]['id']]
    if causal:return predictions[rows[past]['id']]  # Do not relabel extrapolation as interpolation.
    w=(r['time']-rows[past]['time'])/(rows[future]['time']-rows[past]['time'])
    return predictions[rows[past]['id']]*(1-w)+predictions[rows[future]['id']]*w


def correspondence(target_feature,source_feature,source_prediction,temperature=.05,chunk=128):
    """Chunked content correspondence; only valid when the target image is observed."""
    hw=target_feature.shape[-2:];a=F.normalize(target_feature.flatten(1).T,dim=-1)
    b=F.normalize(source_feature.flatten(1).T,dim=-1)
    p=F.interpolate(source_prediction[None],size=source_feature.shape[-2:],mode='bilinear',align_corners=False)[0].flatten(1).T
    pieces=[]
    for i in range(0,len(a),chunk):pieces.append(torch.softmax(a[i:i+chunk]@b.T/temperature,dim=-1)@p)
    return torch.cat(pieces).T.reshape(3,*hw)
