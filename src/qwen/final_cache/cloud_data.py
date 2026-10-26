"""Bounded cache access. This module runs on real data only inside the rented cloud GPU."""
from __future__ import annotations
import gc
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from safetensors import safe_open
from PIL import Image
from core import read_json,restore_grid,sha256


def actor_order(clip,condition,context,alignment):
    if clip=='armbar':
        # Frozen source descriptions: only these action answers swapped physical identities.
        return ['A2','A1'] if condition=='action_relational' and context in ('2fps','five_frame') else ['A1','A2']
    matches=[x for x in alignment if x['clip_id']==clip and x['condition']==condition]
    if len(matches)!=1 or matches[0]['decision']!='include' or not matches[0]['grounding_contract_valid']:return None
    return matches[0]['canonical_actor_order']


def read_map(path,kind,corrected=True):
    with safe_open(str(path),framework='pt',device='cpu') as f:
        h=f.get_tensor('hidden').float();t,gh,gw=map(int,f.get_tensor('grid_thw')[0])
    if t!=1 or not torch.isfinite(h).all():raise ValueError('Nonfinite or non-image spatial cache')
    if kind=='merged':gh//=2;gw//=2
    if len(h)!=gh*gw:raise ValueError('Shape/grid mismatch')
    if kind=='full' and corrected:
        if gh%2 or gw%2:raise ValueError('Invalid Qwen grouping')
        h=h.reshape(gh//2,gw//2,2,2,-1).permute(0,2,1,3,4).reshape(gh,gw,-1)
    else:h=h.reshape(gh,gw,-1)
    return h.permute(2,0,1).contiguous()


def ridge_fit(x,y,regularization):
    if x.ndim!=2 or len(x)==0 or len(y)!=len(x):raise ValueError('Empty or invalid fitting data')
    x=x.float();y=y.float();mean=x.mean(0);scale=x.std(0,unbiased=False).clamp_min(1e-4)
    z=(x-mean)/scale;z=torch.cat([z,torch.ones((len(z),1),device=z.device)],1)
    gram=z.T@z/len(z);rhs=z.T@y/len(z)
    reg=torch.eye(gram.shape[0],device=z.device)*regularization;reg[-1,-1]=1e-7
    w=torch.linalg.solve(gram+reg,rhs)
    return {'mean':mean,'scale':scale,'weight':w}


def ridge_predict(x,model):
    z=(x.float()-model['mean'])/model['scale']
    return z@model['weight'][:-1]+model['weight'][-1]


class CacheBank:
    def __init__(self,packet,volume_root='/workspace',device='cuda'):
        self.packet=Path(packet);self.volume=Path(volume_root);self.device=torch.device(device)
        self.manifest=read_json(self.packet/'manifest.json');self.rows=self.manifest['rows']
        for i,r in enumerate(self.rows):r['id']=i
        self.root=self.volume/'qwen38-campaign/runs/20260827T074622Z'
        self.labels={};self.contacts={};self.cache={};self.projections={};self.checked=set();self.expected={}
        for line in (self.packet/'provenance/original-checksums.sha256').read_text().splitlines():
            if line.strip():digest,name=line.split(maxsplit=1);self.expected[name.lstrip('* ')]=digest
        self.align=read_json(self.packet/'provenance/actor-alignment.json')['records']
        self.refs={}
        for r in self.rows:
            if r['label']:
                y=np.array(Image.open(self.packet/r['label']),copy=True).astype(np.int64)
                c=np.array(Image.open(self.packet/r['contact']),copy=True)
                if r['contact_encoding']=='binary_use_owner_label':c=np.where(c>0,y,0)
                if np.any((c>0)&(c!=y)):raise ValueError('Contact/label disagreement')
                self.labels[r['id']]=y;self.contacts[r['id']]=c
        self.clips=sorted({r['clip'] for r in self.rows})

    def validate_packet(self):
        for name,r in self.manifest['files'].items():
            if sha256(self.packet/name)!=r['sha256']:raise ValueError(f'Packet hash mismatch: {name}')
        return {'files':len(self.manifest['files']),'rows':len(self.rows),'labelled':len(self.labels)}

    def check(self,path):
        path=Path(path);name=str(path.relative_to(self.root))
        if name not in self.checked:
            expected=self.expected.get(name)
            if expected is None or sha256(path)!=expected:raise ValueError(f'Original cache hash mismatch: {name}')
            self.checked.add(name)

    def project(self,grid,dimension=128):
        channels=grid.shape[0];key=(channels,dimension)
        if key not in self.projections:
            gen=torch.Generator().manual_seed(20261013+channels)
            self.projections[key]=(torch.randn(channels,dimension,generator=gen)/dimension**.5).to(self.device)
        z=grid.to(self.device).flatten(1).T@self.projections[key]
        return z.T.reshape(dimension,*grid.shape[1:])

    def native_grid(self,r):
        path=self.root/r['cache_prefix']/'spatial/full/layer_11'/f"frame_{r['frame']:06d}.safetensors"
        self.check(path)
        with safe_open(str(path),framework='pt',device='cpu') as f:t,h,w=map(int,f.get_tensor('grid_thw')[0])
        if t!=1 or h<=0 or w<=0:raise ValueError('Invalid native reference grid')
        return h,w

    def feature(self,r,representation):
        key=(r['id'],representation)
        if key in self.cache:return self.cache[key].to(self.device).float()
        if '+' in representation:
            parts=[self.feature(r,p) for p in representation.split('+')]
            hw=parts[0].shape[-2:]
            out=torch.cat([F.interpolate(p[None],size=hw,mode='bilinear',align_corners=False)[0] for p in parts])
        else:
            from audit import representation_contract
            rep,dimension=representation_contract(representation);legacy=rep.startswith('legacy_');rep=rep.removeprefix('legacy_')
            repool=rep.startswith('repool');rep=rep.replace('repool','full')
            if rep=='geometry':
                from jobs import geometry_features
                if r['clip']=='armbar':
                    ref=next(x for x in self.rows if x['clip']==r['clip'] and x['reference'])
                    y=self.labels[ref['id']];centers=[]
                    for a in (1,2):
                        yy,xx=np.where(y==a);centers.append([(xx.mean()+.5)/y.shape[1],(yy.mean()+.5)/y.shape[0]])
                else:centers=[[(b[0]+b[2])/2,(b[1]+b[3])/2] for b in self.manifest['reference_boxes'][r['clip']]]
                h,w=self.native_grid(r)
                out=geometry_features(torch.empty(1,h,w,device=self.device),torch.tensor(centers,device=self.device))
            elif rep=='rgb':
                h,w=self.native_grid(r)
                im=Image.open(self.volume/r['image']).convert('RGB').resize((w,h))
                out=torch.tensor(np.array(im).copy(),device=self.device).permute(2,0,1).float()/255
            else:
                if rep=='merged':kind='merged';rel='merged-vision'
                else:kind='full' if rep.startswith('full') else 'pooled';layer=int(rep.replace('full','').replace('pooled',''));rel=f'spatial/{kind}/layer_{layer:02d}'
                path=self.root/r['cache_prefix']/rel/f"frame_{r['frame']:06d}.safetensors"
                self.check(path);grid=read_map(path,kind,corrected=not legacy)
                out=grid.to(self.device) if dimension is None else self.project(grid,dimension)
                if repool:out=F.adaptive_avg_pool2d(out[None],(30,30))[0]
                del grid
        self.cache[key]=out.detach().half().cpu()
        # Bound CPU residency even if a future job combines many layers.
        while sum(x.numel()*x.element_size() for x in self.cache.values())>3*1024**3:
            self.cache.pop(next(iter(self.cache)))
        return out

    def reference(self,r,representation):
        key=(r['clip'],representation)
        if key in self.refs:return self.refs[key].to(self.device)
        seed=next(x for x in self.rows if x['clip']==r['clip'] and x['reference'])
        f=self.feature(seed,representation);h,w=f.shape[1:];yy,xx=torch.meshgrid((torch.arange(h,device=self.device)+.5)/h,(torch.arange(w,device=self.device)+.5)/w,indexing='ij')
        values=[]
        for actor in (1,2):
            if r['clip']=='armbar':
                y=torch.tensor(self.labels[seed['id']],device=self.device).float()[None,None]
                m=F.interpolate(y,size=(h,w),mode='nearest')[0,0]==actor
            else:
                x0,y0,x1,y1=self.manifest['reference_boxes'][r['clip']][actor-1]
                m=(xx>=x0)&(xx<x1)&(yy>=y0)&(yy<y1)
            if not m.any():raise ValueError('Appearance reference has no tokens')
            values.append(f[:,m].mean(1))
        q=torch.stack(values);self.refs[key]=q.detach().cpu();return q

    def semantic(self,r,condition='action_relational',layer=25,context='4fps',thinking='off',dimension=64):
        order=actor_order(r['clip'],condition,context,self.align)
        if order is None:return None
        states=[]
        for actor in order:
            path=self.root/r['cache_prefix']/f'semantic/video/{context}/{condition}/{thinking}/{actor}.safetensors'
            if not path.exists():return None
            self.check(path)
            with safe_open(str(path),framework='pt',device='cpu') as f:s=f.get_tensor('marker_states')[layer].float()
            if not torch.isfinite(s).all():raise ValueError('Invalid semantic state')
            states.append(s)
        values=torch.stack(states).to(self.device)
        # Fixed data-independent projection. No test-fitted PCA or channel scaler.
        key=(values.shape[-1],dimension)
        if key not in self.projections:
            gen=torch.Generator().manual_seed(20261013+key[0]);self.projections[key]=(torch.randn(*key,generator=gen)/dimension**.5).to(self.device)
        return values@self.projections[key]

    def clear(self):
        self.cache.clear();self.refs.clear();gc.collect()
        if self.device.type=='cuda':torch.cuda.empty_cache()

    def labelled(self,clip=None):
        return [r for r in self.rows if r['id'] in self.labels and not r['reference'] and (clip is None or r['clip']==clip)]

    def image(self,r):return Image.open(self.volume/r['image']).convert('RGB')
