"""Fixed-budget fitting and complete per-example ownership evaluation."""
from __future__ import annotations
import hashlib
import os
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from cloud_data import ridge_fit,ridge_predict
from core import atomic_json,metrics,sha256
from models import OwnershipHead,ownership_loss


def sample_indices(labels,per_class,seed):
    y=np.asarray(labels).ravel();rng=np.random.default_rng(seed);indices=[]
    for k in (0,1,2):
        at=np.flatnonzero(y==k)
        if len(at):indices.extend(rng.choice(at,min(per_class,len(at)),replace=False).tolist())
    return np.array(indices,dtype=np.int64)


def sample_features(feature,indices,label_hw):
    h,w=label_hw;indices=torch.as_tensor(indices,device=feature.device)
    yy=(indices//w+.5)/h*2-1;xx=(indices%w+.5)/w*2-1
    grid=torch.stack([xx,yy],-1).reshape(1,-1,1,2)
    return F.grid_sample(feature[None],grid,align_corners=False,padding_mode='border')[0,:,:,0].T


def relation_features(features,query):
    f=F.normalize(features,dim=-1);q=F.normalize(query,dim=-1)
    return torch.cat([f*q,(f-q).abs()],-1)


def summarize(records):
    keys=('actor_iou','contact_accuracy','background_false_positive','contact_margin')
    clips={}
    for clip in sorted({r['clip'] for r in records}):
        rows=[r['metrics'] for r in records if r['clip']==clip]
        clips[clip]={k:float(np.mean([x[k] for x in rows if x.get(k) is not None])) if any(x.get(k) is not None for x in rows) else None for k in keys}
    return {'per_clip':clips,'equal_clip':{k:float(np.mean([x[k] for x in clips.values() if x[k] is not None])) if any(x[k] is not None for x in clips.values()) else None for k in keys},'scored_frames':len(records)}


def normalize_feature(x):return F.layer_norm(x.permute(1,2,0),(x.shape[0],)).permute(2,0,1)


def evaluate(bank,rows,predict,out,job):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);records=[];files=[];exclusions=[]
    with torch.inference_mode():
        for r in rows:
            logits=predict(r)
            if logits is None:
                exclusions.append({'clip':r['clip'],'frame':r['frame'],'reason':'Missing or excluded semantic state under the frozen actor audit'});continue
            if logits.ndim==4:logits=logits[0]
            if not torch.isfinite(logits).all():raise ValueError('Nonfinite output')
            fn=f"{r['clip']}-{r['frame']:06d}.npz";p=out/fn;tmp=out/(fn+'.tmp')
            with tmp.open('wb') as h:np.savez_compressed(h,logits=logits.detach().half().cpu().numpy());h.flush();os.fsync(h.fileno())
            os.replace(tmp,p);files.append({'path':fn,'sha256':sha256(p),'bytes':p.stat().st_size})
            if r['id'] not in bank.labels or r['reference']:continue
            y=bank.labels[r['id']];c=bank.contacts[r['id']]
            probs=F.interpolate(logits[None].float(),size=y.shape,mode='bilinear',align_corners=False).softmax(1)[0].cpu().numpy()
            pred=probs.argmax(0);m=metrics(pred,y,c)
            from audit import contact_regions
            m['contact_regions']=contact_regions(pred,c)
            cm=np.asarray(m['confusion']);den=cm.sum(0)+cm.sum(1)-cm.diagonal()
            m['actor_iou_separate']={f'A{k}':float(cm[k,k]/den[k]) if den[k]>0 else None for k in (1,2)}
            if (c>0).any():
                yy,xx=np.where(c>0);own=c[yy,xx].astype(int);m['contact_margin']=float((probs[own,yy,xx]-probs[3-own,yy,xx]).mean())
            else:m['contact_margin']=None
            # Risk/coverage measured only where there are references; raw confidence is not calibrated.
            known=y!=255;conf=probs.max(0)[known];errors=(pred[known]!=y[known]);order=np.argsort(-conf,kind='stable')
            m['risk_coverage']={str(cov):float(errors[order[:max(1,int(cov*len(order)))]].mean()) if len(order) else None for cov in (.25,.5,.75,1.)}
            swapped=np.choose(pred,[0,2,1]);m['swapped_actor_iou']=metrics(swapped,y,c)['actor_iou']
            records.append({'clip':r['clip'],'frame':r['frame'],'time':r['time'],'shot':r['shot'],'prediction':fn,'metrics':m})
    result={'job':job,'records':records,'files':files,'exclusions':exclusions,**summarize(records)}
    atomic_json(out/'result.json',result);return result


def fit_linear(bank,train,representation,seed=7,regularization=.01,mode='visual',fraction=1.):
    rows=list(train);rng=np.random.default_rng(seed)
    if fraction<1:rows=[rows[i] for i in sorted(rng.choice(len(rows),max(1,int(len(rows)*fraction)),replace=False))]
    xf=[];yf=[];xr=[];yr=[]
    for r in rows:
        y=bank.labels[r['id']];ids=sample_indices(y,128,seed+r['id']);f=bank.feature(r,representation)
        v=sample_features(f,ids,y.shape);target=torch.tensor(y.ravel()[ids],device=bank.device)
        if mode=='random_labels':target=target[torch.randperm(len(target),generator=torch.Generator(device=bank.device).manual_seed(seed+r['id']),device=bank.device)]
        if mode=='vision_only':
            xf.append(v);yf.append(F.one_hot(target.long(),3).float()*2-1);continue
        q=bank.reference(r,representation)
        xf.append(v);yf.append(torch.where(target==0,1.,-1.)[:,None])
        for actor in (1,2):
            xr.append(relation_features(v,q[actor-1]));yr.append(torch.where(target==actor,1.,-1.)[:,None])
    fg=ridge_fit(torch.cat(xf),torch.cat(yf),regularization)
    actor=None if mode=='vision_only' else ridge_fit(torch.cat(xr),torch.cat(yr),regularization)
    def predict(r,override=None):
        f=bank.feature(r,representation);h,w=f.shape[-2:];v=f.flatten(1).T
        intervention=override or mode
        if intervention=='spatial_mean':v=v.mean(0,keepdim=True).expand_as(v)
        if intervention=='displaced':v=f.roll((h//2,w//2),(1,2)).flatten(1).T
        if mode=='vision_only':return ridge_predict(v,fg).T.reshape(3,h,w)
        q=reference_control(bank,r,representation,intervention)
        logits=[ridge_predict(v,fg)]
        for a in range(2):logits.append(ridge_predict(relation_features(v,q[a]),actor))
        return torch.cat(logits,-1).T.reshape(3,h,w)
    return predict,{'foreground':fg,'actor':actor,'train_rows':[r['id'] for r in rows],'representation':representation,'seed':seed}


def reference_control(bank,r,representation,control):
    q=bank.reference(r,representation)
    if control=='swap_reference':return q.flip(0)
    if control=='wrong_reference':
        other=next(x for x in bank.rows if x['clip']!=r['clip'] and x['clip']!='armbar' and x['reference'])
        return bank.reference(other,representation)
    return q


def save_checkpoint(path,payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('wb') as f:torch.save(payload,f);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def transformed_semantic(bank,r,condition,layer,control='real',seed=7):
    if control.startswith('context:'):return bank.semantic(r,condition,layer,context=control.split(':',1)[1])
    if control.startswith('thinking:'):return bank.semantic(r,condition,layer,thinking=control.split(':',1)[1])
    s=bank.semantic(r,condition,layer)
    if s is None:return None
    if control=='zero':return torch.zeros_like(s)
    if control=='mean':return s.mean(0,keepdim=True).expand_as(s)
    if control=='swap':return s.flip(0)
    if control=='wrong':
        clips=sorted({x['clip'] for x in bank.rows if x['clip']!='armbar'})
        others=[c for c in clips if c!=r['clip']]
        other=next(x for x in bank.rows if x['clip']==others[0] and x['reference'])
        return bank.semantic(other,condition,layer)
    if control=='random':
        offset=int(hashlib.sha256(r['clip'].encode()).hexdigest()[:8],16)
        g=torch.Generator(device=bank.device).manual_seed(seed+offset)
        z=torch.randn(s.shape,generator=g,device=bank.device)
        return F.normalize(z,dim=-1)*s.norm(dim=-1,keepdim=True)
    return s


def fit_neural(bank,train,representation,out,seed=7,mode='appearance',condition='action_relational',layer=25,control='real',epochs=80,width=24,contact_weight=1.,fraction=1.,gated=False,eligibility_condition=None):
    torch.manual_seed(seed);rng=np.random.default_rng(seed);rows=list(train)
    if contact_weight==0:rows=[r for r in rows if np.any((bank.labels[r['id']]!=255)&(bank.contacts[r['id']]==0))]
    if mode!='appearance':rows=[r for r in rows if transformed_semantic(bank,r,condition,layer,control,seed) is not None]
    elif eligibility_condition:rows=[r for r in rows if transformed_semantic(bank,r,eligibility_condition,layer,'real',seed) is not None]
    if fraction<1:rows=[rows[i] for i in sorted(rng.choice(len(rows),max(1,int(len(rows)*fraction)),replace=False))]
    if not rows:raise ValueError('No eligible training rows')
    channels=bank.feature(rows[0],representation).shape[0]
    model=OwnershipHead(channels=channels,width=width,mode=mode,gated=gated).to(bank.device)
    model.training_rows=[r['id'] for r in rows]
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    history=[];start=time.time()
    for epoch in range(epochs):
        model.train();losses=[]
        for j in rng.permutation(len(rows)):
            r=rows[j];f=normalize_feature(bank.feature(r,representation));q=F.layer_norm(bank.reference(r,representation),(channels,))
            s=None if mode=='appearance' else transformed_semantic(bank,r,condition,layer,control,seed)
            y=torch.tensor(bank.labels[r['id']],device=bank.device)[None];c=torch.tensor(bank.contacts[r['id']],device=bank.device)[None]
            opt.zero_grad(set_to_none=True);logits=model(f[None],q[None],None if s is None else s[None])
            loss=ownership_loss(logits,y,c,contact_weight)
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();losses.append(float(loss.detach()))
        history.append({'epoch':epoch+1,'loss':float(np.mean(losses))})
        if (epoch+1)%10==0:atomic_json(Path(out)/'training.json',{'epoch':epoch+1,'history':history,'seconds':time.time()-start})
    model.eval();save_checkpoint(Path(out)/'decoder.pt',{'state_dict':model.state_dict(),'channels':channels,'width':width,'mode':mode,'gated':gated,'parameters':sum(p.numel() for p in model.parameters()),'representation':representation,'seed':seed,'epochs':epochs,'condition':condition,'layer':layer,'control':control,'train_rows':[r['id'] for r in rows]})
    atomic_json(Path(out)/'training.json',{'history':history,'seconds':time.time()-start})
    def predict(r,override=None,semantic_override=None):
        s=None if mode=='appearance' else (semantic_override if semantic_override is not None else transformed_semantic(bank,r,condition,layer,override or control,seed))
        if mode!='appearance' and s is None:return None
        f=normalize_feature(bank.feature(r,representation));q=F.layer_norm(reference_control(bank,r,representation,override),(channels,))
        if override=='spatial_mean':f=f.mean((1,2),keepdim=True).expand_as(f)
        if override=='displaced':f=f.roll((f.shape[1]//2,f.shape[2]//2),(1,2))
        logits=model(f[None],q[None],None if s is None else s[None])[0]
        if override=='connectivity':
            from models import connectivity
            logits=connectivity(logits,f)
        return logits
    return predict,model
