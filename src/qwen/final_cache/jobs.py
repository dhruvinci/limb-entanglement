"""Job identities, leak-free split rules and durable completion evidence."""
import hashlib
import json
from pathlib import Path
import torch
from core import atomic_json,read_json,sha256


def split(rows,heldout,evaluation='test'):
    eligible=[r for r in rows if r.get('label') and not r['reference']]
    if heldout=='armbar':
        train=[r for r in eligible if r['clip']=='armbar' and r['subset']=='train']
        accepted={'val','validation'} if evaluation in ('val','validation') else {evaluation}
        test=[r for r in eligible if r['clip']=='armbar' and r['subset'] in accepted]
    else:
        train=[r for r in eligible if r['clip'] not in ('armbar',heldout)]
        test=[r for r in eligible if r['clip']==heldout]
    if set(r['id'] for r in train)&set(r['id'] for r in test):raise ValueError('Split leakage')
    return train,test


def job_digest(job):return hashlib.sha256(json.dumps(job,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def seal_job(root,job):
    root=Path(root);files=[]
    for p in sorted(root.rglob('*')):
        if p.is_file() and p.name!='COMPLETE.json' and not p.name.endswith('.tmp'):
            files.append({'path':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':sha256(p)})
    if not files:raise ValueError('Cannot complete an empty job')
    atomic_json(root/'COMPLETE.json',{'job':job,'job_sha256':job_digest(job),'files':files})


def completed(root,job):
    root=Path(root);p=root/'COMPLETE.json'
    if not p.exists():return False
    try:
        receipt=read_json(p)
        return receipt['job_sha256']==job_digest(job) and bool(receipt['files']) and all(
            (root/f['path']).stat().st_size==f['bytes'] and sha256(root/f['path'])==f['sha256'] for f in receipt['files'])
    except (OSError,ValueError,KeyError):return False


def geometry_features(shape_template,centers):
    _,h,w=shape_template.shape;device=shape_template.device
    yy,xx=torch.meshgrid((torch.arange(h,device=device)+.5)/h,(torch.arange(w,device=device)+.5)/w,indexing='ij')
    channels=[xx,yy,xx*yy,xx.square()+yy.square()]
    for x,y in centers:channels.extend([xx-x,yy-y])
    return torch.stack(channels)
