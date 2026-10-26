"""Artifact integrity and small, interpretation-critical analysis primitives."""
from collections import deque
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
from core import sha256,atomic_json,read_json


def representation_contract(name):
    parts=name.split('@')
    if len(parts)>2:raise ValueError('Invalid representation dimension')
    dimension=128 if len(parts)==1 else (None if parts[1]=='native' else int(parts[1]))
    if dimension is not None and (dimension<1 or dimension>1024):raise ValueError('Invalid projection dimension')
    return parts[0],dimension


def verify_artifact(path,expected):
    path=Path(path);digest=sha256(path)
    if digest!=expected:raise ValueError(f'Checksum mismatch: {path.name}')
    result={'sha256':digest,'bytes':path.stat().st_size,'tensors':{}}
    if path.suffix=='.safetensors':
        with safe_open(str(path),framework='pt',device='cpu') as f:
            for key in f.keys():
                x=f.get_tensor(key)
                if x.is_floating_point() and not torch.isfinite(x).all():raise ValueError(f'Nonfinite tensor: {path.name}/{key}')
                result['tensors'][key]={'shape':list(x.shape),'dtype':str(x.dtype),'finite':True}
                if x.is_floating_point() and x.numel():result['tensors'][key]['rms']=float(x.float().square().mean().sqrt())
                del x
    return result


def inventory(bank,out):
    out=Path(out);cache=out/'integrity-artifacts';cache.mkdir(parents=True,exist_ok=True);records=[]
    for i,(name,digest) in enumerate(sorted(bank.expected.items())):
        receipt=cache/f'{i:06d}.json';path=bank.root/name
        # Original file metadata must be unchanged to reuse an integrity receipt.
        stamp={'size':path.stat().st_size,'mtime_ns':path.stat().st_mtime_ns}
        saved=read_json(receipt) if receipt.exists() else None
        if saved is not None and saved.get('name')==name and saved.get('sha256')==digest and saved.get('stat')==stamp:record=saved
        else:
            record={'name':name,**verify_artifact(path,digest),'stat':stamp};atomic_json(receipt,record)
        bank.checked.add(name);records.append(record)
        if i%16==0:atomic_json(out/'integrity-progress.json',{'verified':i+1,'expected':len(bank.expected),'name':name})
    for r in bank.rows:
        if sha256(bank.volume/r['image'])!=r['image_sha256']:raise ValueError(f"Source frame checksum mismatch: {r['clip']}/{r['frame']}")
    result={'files':len(records),'bytes':sum(r['bytes'] for r in records),'source_frames':len(bank.rows),'artifacts':records}
    atomic_json(out/'integrity.json',result);return result


def select_layers(records,count=3):
    rows=[r for r in records if r.get('validation_iou') is not None and np.isfinite(r['validation_iou'])]
    return [r['layer'] for r in sorted(rows,key=lambda r:(-r['validation_iou'],r['layer']))[:count]]


def contact_regions(prediction,contact):
    p=np.asarray(prediction);c=np.asarray(contact);h,w=c.shape;records=[]
    for actor in (1,2):
        unseen=set(map(tuple,np.argwhere(c==actor)))
        while unseen:
            seed=min(unseen);unseen.remove(seed);queue=deque([seed]);component=[]
            while queue:
                y,x=queue.popleft();component.append((y,x))
                for z in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)):
                    if z in unseen:unseen.remove(z);queue.append(z)
            yy,xx=np.array(component).T
            records.append({'owner':actor,'pixels':len(component),'accuracy':float((p[yy,xx]==actor).mean()),'box_xyxy':[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)]})
    return records
