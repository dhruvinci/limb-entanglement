"""Separate matched-crop encoder experiment; never conflated with native caches."""
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from core import read_json,sha256


def combine(a,b,condition,axis=1):
    if condition=='dino':return a
    if condition=='qwen':return b
    if condition not in ('combined','gated'):raise ValueError('Invalid paired condition')
    sa=[slice(None)]*a.ndim;sa[axis]=slice(0,32)
    return torch.cat([a[tuple(sa)],b[tuple(sa)]],axis)


class PairedBank:
    def __init__(self,native,heldout,condition):
        self.native=native;self.packet=native.packet;self.device=native.device;self.condition=condition
        root=self.packet/'paired';freeze=read_json(root/'freeze.json');meta=read_json(root/f'features/{heldout}.json')
        feature_path=root/f'features/{heldout}.npz'
        if sha256(feature_path)!=meta['features_sha256']:raise ValueError('Paired features checksum mismatch')
        if heldout in meta['train_sources']:raise ValueError('Paired PCA used test source')
        self.provenance=meta;self.rows=[];self.labels={};self.contacts={};self.mapping={}
        native_lookup={(r['clip'],r['frame']):r for r in native.rows}
        for i,r in enumerate(freeze['rows']):
            n=native_lookup[r['clip'],r['frame']];row={**n,'id':i,'label':'paired' if r['annotated'] else None,'paired_crop':r}
            self.rows.append(row);self.mapping[i]=n
            if r['annotated']:
                y=np.array(Image.open(root/f"data/{r['clip']}/{r['frame']:03d}-labels.png")).astype(np.int64)
                c=np.array(Image.open(root/f"data/{r['clip']}/{r['frame']:03d}-contact.png"))
                self.labels[i]=y;self.contacts[i]=np.where((c>0)&np.isin(y,[1,2]),y,0)
        with np.load(feature_path) as data:
            self.x=combine(torch.from_numpy(data['dino'].copy()),torch.from_numpy(data['qwen'].copy()),condition)
            self.q=combine(torch.from_numpy(data['dino_queries'].copy()),torch.from_numpy(data['qwen_queries'].copy()),condition,axis=2)
        self.clips=sorted({r['clip'] for r in self.rows})

    def feature(self,r,representation):return self.x[r['id']].to(self.device).float()
    def reference(self,r,representation):return self.q[r['id']].to(self.device).float()
    def semantic(self,r,*a,**kw):return self.native.semantic(self.mapping[r['id']],*a,**kw)
    def clear(self):
        if self.device.type=='cuda':torch.cuda.empty_cache()
    def image(self,r):
        c=r['paired_crop'];im=self.native.image(self.mapping[r['id']]).crop(c['crop_original_xyxy']).resize(c['resized_wh'])
        out=Image.new('RGB',(448,448));out.paste(im,tuple(c['offset_xy']));return out
