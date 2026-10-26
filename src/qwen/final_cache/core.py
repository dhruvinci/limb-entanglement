"""Small, audited numerical and lifecycle primitives for the final cache campaign."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import numpy as np


def restore_grid(tokens, height, width, merge=2):
    a=np.asarray(tokens)
    if a.ndim!=2 or height<=0 or width<=0 or merge<1 or height%merge or width%merge or len(a)!=height*width:
        raise ValueError("Invalid token/grid/grouping contract")
    return a.reshape(height//merge,width//merge,merge,merge,a.shape[-1]).transpose(0,2,1,3,4).reshape(height,width,-1)


def metrics(prediction, target, contact=None):
    p=np.asarray(prediction).ravel();y=np.asarray(target).ravel()
    if p.shape!=y.shape or not np.isin(y,[0,1,2,255]).all() or not np.isin(p,[0,1,2]).all():
        raise ValueError("Invalid ownership labels")
    keep=y!=255
    cm=np.bincount(y[keep].astype(int)*3+p[keep],minlength=9).reshape(3,3)
    tp=cm.diagonal();den=cm.sum(0)+cm.sum(1)-tp
    iou=np.divide(tp,den,out=np.full(3,np.nan),where=den>0)
    con=np.zeros_like(y) if contact is None else np.asarray(contact).ravel()
    if con.shape!=y.shape or not np.isin(con,[0,1,2]).all() or np.any((con>0)&(con!=y)):
        raise ValueError("Contact ownership must agree with known target ownership")
    ck=con>0
    return {'confusion':cm.tolist(),'actor_iou':float(np.nanmean(iou[1:])) if np.isfinite(iou[1:]).any() else None,
            'known_count':int(keep.sum()),'contact_count':int(ck.sum()),
            'contact_accuracy':float((p[ck]==con[ck]).mean()) if ck.any() else None,
            'background_false_positive':float((p[y==0]!=0).mean()) if (y==0).any() else None}


def temporal_neighbors(rows,index,causal=False,max_gap=2.0):
    r=rows[index]
    candidates=[i for i,x in enumerate(rows) if i!=index and x['clip']==r['clip'] and x['shot']==r['shot']
                and 0<abs(x['time']-r['time'])<=max_gap and (not causal or x['time']<r['time'])]
    return sorted(candidates,key=lambda i:(abs(rows[i]['time']-r['time']),rows[i]['time']))


def must_stop(elapsed_hours,compute_rate,spent_before=0,storage_hours=0):
    vals=[elapsed_hours,compute_rate,spent_before,storage_hours]
    if not all(np.isfinite(vals)) or any(v<0 for v in vals):return True
    return elapsed_hours>=10 or elapsed_hours*compute_rate+spent_before>=17.5 or elapsed_hours*compute_rate+spent_before+storage_hours*21/720>=19


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def atomic_json(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.tmp')
    with tmp.open('w') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)


def read_json(path):
    with Path(path).open() as f:return json.load(f)


def safe_member(root,name):
    p=Path(name)
    if p.is_absolute() or '..' in p.parts or any(x.startswith('.env') or x.startswith('private-') for x in p.parts):
        raise ValueError('Unsafe packet member')
    result=Path(root)/p
    if not result.resolve().is_relative_to(Path(root).resolve()):raise ValueError('Symlink escapes packet')
    return result
