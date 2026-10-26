"""Fixed scientific job matrix. Heavy jobs are executed only on the cloud worker."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from core import atomic_json,read_json
from jobs import split,seal_job,completed
from experiment import fit_linear,fit_neural,evaluate,save_checkpoint
from models import TemporalHead,ownership_loss
from temporal import temporal_input,baseline,corrupt,correspondence,source_indices


def build_jobs(clips):
    jobs=[];breadth=[c for c in clips if c!='armbar']
    def add(kind,heldout,rep,seed=7,**kw):
        name='-'.join([kind,heldout,rep.replace('+','_'),str(seed)]+[f'{k}_{v}' for k,v in sorted(kw.items()) if k not in ('controls',)])
        jobs.append({'name':name,'kind':kind,'heldout':heldout,'representation':rep,'seed':seed,**kw})
    # All-layer screens remain fixed; do not choose a layer on breadth test scores.
    for layer in range(27):
        if 'armbar' in clips:
            for prefix in ('full','legacy_full','repool','pooled'):
                add('linear','armbar',f'{prefix}{layer}',evaluation='val_and_test')
            add('linear','armbar',f'full{layer}',mode='vision_only',evaluation='val_and_test')
        for clip in breadth:add('linear',clip,f'pooled{layer}',layout_caveat=True)
    for clip in clips:
        for seed in (7,71,701):
            for rep in ('rgb','geometry','full5','full11','full18','full26','merged','full11+merged'):
                add('linear',clip,rep,seed,controls=['spatial_mean','displaced','wrong_reference','swap_reference'])
            add('linear',clip,'full11+merged',seed,mode='vision_only')
            add('linear',clip,'full11+merged',seed,mode='random_labels')
            if seed==7:
                for dimension in (64,256,'native'):add('linear',clip,f'full11@{dimension}',seed)
            for mode,control in (('appearance','real'),('global','real'),('local','real'),('local','random')):
                add('neural',clip,'full11+merged',seed,mode=mode,control=control,epochs=80,
                    controls=['swap','zero','mean','wrong','random','connectivity']+(['context:'+x for x in ('single_frame','five_frame','2fps','8fps','4fps_reversed','4fps_shuffled')]+['thinking:xhigh'] if clip=='armbar' else []) if mode!='appearance' else ['swap_reference','wrong_reference','spatial_mean','displaced','connectivity'])
            add('temporal',clip,'full11+merged',seed,epochs=40)
            if clip!='armbar':
                for enc in ('dino','qwen','combined','gated'):add('paired',clip,enc,seed,epochs=80,mode='appearance',gated=enc=='gated',controls=['swap_reference','wrong_reference','spatial_mean','connectivity'])
    # Fixed depth/capacity/supervision changes, scored independently rather than picked post hoc.
    for clip in breadth:
        for seed in (7,71,701):
            for width in (12,48):add('neural',clip,'full11+merged',seed,mode='appearance',width=width,epochs=80)
            for fraction in (.25,.5):add('neural',clip,'full11+merged',seed,mode='appearance',fraction=fraction,epochs=80)
            for contact_weight in (0.,4.):add('neural',clip,'full11+merged',seed,mode='appearance',contact_weight=contact_weight,epochs=80)
        for layer in (12,45,60):add('neural',clip,'full11+merged',mode='local',layer=layer,epochs=80)
        for condition in ('identity_only','contact_ownership'):add('neural',clip,'full11+merged',mode='local',condition=condition,epochs=80)
        add('neural',clip,'full11+merged',mode='appearance',eligibility_condition='contact_ownership',epochs=80)
    return jobs


def run_temporal(bank,job,out):
    heldout=job['heldout'];seed=job['seed'];rep=job['representation'];train,test=split(bank.rows,heldout)
    if heldout=='armbar':train=[r for r in bank.rows if r.get('label') and not r['reference'] and r['clip']!='armbar']
    trainclips=sorted({r['clip'] for r in train});preds={};provenance=[]
    # Inner source holdout prevents the temporal model learning from in-sample base predictions.
    for clip in trainclips+[heldout]:
        base_train=[r for r in train if r['clip']!=clip]
        predictor,state=fit_linear(bank,base_train,rep,seed)
        provenance.append({'predicted_clip':clip,'training_clips':sorted({r['clip'] for r in base_train}),'train_rows':[r['id'] for r in base_train]})
        save_checkpoint(out/f'base-{clip}.pt',state)
        with torch.inference_mode():
            for r in bank.rows:
                if r['clip']==clip:preds[r['id']]=predictor(r).softmax(0).detach()
    rows=[r for r in bank.rows if r['id'] in preds];lookup={r['id']:i for i,r in enumerate(rows)}
    records={};rng=np.random.default_rng(seed);training_seconds=0.;training_steps=0
    for causal in (True,False):
        tag='causal' if causal else 'offline'
        for condition in ('observed','hidden','corrupted'):
            key=f'{tag}-{condition}';torch.manual_seed(seed);model=TemporalHead().to(bank.device)
            opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4);history=[]
            training_started=time.perf_counter()
            for epoch in range(job.get('epochs',40)):
                losses=[];model.train()
                for j in rng.permutation(len(train)):
                    r=train[j];x,_=temporal_input(rows,preds,lookup[r['id']],condition,causal,seed=seed)
                    y=torch.tensor(bank.labels[r['id']],device=bank.device)[None]
                    opt.zero_grad(set_to_none=True);loss=ownership_loss(model(x[None]),y)
                    if not torch.isfinite(loss):raise ValueError('Nonfinite temporal loss')
                    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();losses.append(float(loss.detach()))
                history.append(float(np.mean(losses)))
                if (epoch+1)%10==0:atomic_json(out/'training.json',{'stage':key,'epoch':epoch+1,'losses':history})
            training_seconds+=time.perf_counter()-training_started;training_steps+=job.get('epochs',40)*len(train)
            model.eval();save_checkpoint(out/f'{key}.pt',{'state_dict':model.state_dict(),'history':history,'training_rows':[r['id'] for r in train],'base_provenance':provenance,'condition':condition,'causal':causal})
            for order in ('ordered','reversed','shuffled'):
                sources=[]
                def predict(r,order=order):
                    x,info=temporal_input(rows,preds,lookup[r['id']],condition,causal,order,seed)
                    sources.append({'target_id':r['id'],'target_time':r['time'],**info})
                    return model(x[None])[0]
                ev=evaluate(bank,test,predict,out/key/order,{**job,'condition':condition,'causal':causal,'order':order})
                atomic_json(out/key/order/'sources.json',sources)
                records[f'{key}-{order}']=ev['equal_clip']
            for method in ('static','persistence','interpolation'):
                if causal and method=='interpolation':continue
                def predict(r,method=method):
                    if method=='static':
                        p=preds[r['id']]
                        if condition=='hidden':p=torch.zeros_like(p)
                        elif condition=='corrupted':p=corrupt(p,seed+r['id'])
                    else:p=baseline(rows,preds,lookup[r['id']],method,causal)
                    return p.clamp_min(1e-6).log()
                ev=evaluate(bank,test,predict,out/key/method,{**job,'condition':condition,'causal':causal,'baseline':method})
                records[f'{key}-{method}']=ev['equal_clip']
            if condition!='hidden':
                def predict(r):
                    idx=lookup[r['id']];ids=[i for i in source_indices(rows,idx,causal) if i is not None]
                    if not ids:return preds[r['id']].clamp_min(1e-6).log()
                    f=bank.feature(r,rep);f=F.adaptive_avg_pool2d(f,(28,28))
                    transported=[]
                    for i in ids:
                        other=rows[i];g=F.adaptive_avg_pool2d(bank.feature(other,rep),(28,28))
                        transported.append(correspondence(f,g,preds[other['id']]))
                    return torch.stack(transported).mean(0).clamp_min(1e-6).log()
                ev=evaluate(bank,test,predict,out/key/'correspondence',{**job,'condition':condition,'causal':causal,'baseline':'feature-correspondence-target-image-observed'})
                records[f'{key}-correspondence']=ev['equal_clip']
    result={'job':job,'inner_source_holdout':True,'provenance':provenance,'comparisons':records,
            'scope':'Cache-only ownership prediction recovery. Hidden means no center logits or features; corruption is label-blind. No semantic/video states are used in causal or offline temporal training. No true occluded-pixel ground truth.'}
    atomic_json(out/'training-cost.json',{'seconds':training_seconds,'steps':training_steps,'train_rows':len(train)})
    atomic_json(out/'result.json',result);return result


def run_job(bank,job,root):
    out=Path(root)/job['name']
    if completed(out,job):return read_json(out/'result.json')
    if out.exists():
        # Preserve failed/corrupted attempts; never overwrite their evidence.
        out.rename(out.with_name(out.name+f'.invalid-{time.time_ns()}'))
    out.mkdir(parents=True,exist_ok=True)
    atomic_json(out/'job.json',job);start=time.time()
    if job['kind']=='paired':
        from paired import PairedBank
        bank=PairedBank(bank,job['heldout'],job['representation'])
        atomic_json(out/'paired-provenance.json',bank.provenance)
    train,test=split(bank.rows,job['heldout'])
    if job['kind']=='temporal':result=run_temporal(bank,job,out)
    elif job['kind']=='linear':
        predict,state=fit_linear(bank,train,job['representation'],job['seed'],mode=job.get('mode','visual'))
        save_checkpoint(out/'linear.pt',state)
        result=evaluate(bank,test,predict,out,job)
        if job.get('evaluation')=='val_and_test':
            _,validation=split(bank.rows,job['heldout'],evaluation='val')
            evaluate(bank,validation,predict,out/'validation',job)
        for control in job.get('controls',[]):
            evaluate(bank,test,lambda r:predict(r,override=control),out/'controls'/control,{**job,'evaluation_control':control})
    elif job['kind'] in ('neural','paired'):
        options={k:job[k] for k in ('seed','mode','condition','layer','control','epochs','width','contact_weight','fraction','gated','eligibility_condition') if k in job}
        predict,model=fit_neural(bank,train,job['representation'],out,**options)
        result=evaluate(bank,test,predict,out,job)
        result['train_rows']=model.training_rows
        atomic_json(out/'result.json',result)
        for control in job.get('controls',[]):
            evaluate(bank,test,lambda r:predict(r,override=control),out/'controls'/control,{**job,'evaluation_control':control})
        del model
    else:raise ValueError('Unknown job type')
    atomic_json(out/'timing.json',{'seconds':time.time()-start})
    seal_job(out,job);bank.clear();return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--packet',required=True);ap.add_argument('--out',required=True);ap.add_argument('--volume',default='/workspace');ap.add_argument('--job',type=int);a=ap.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('Real campaign requires the approved cloud GPU')
    from cloud_data import CacheBank
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.cuda.set_per_process_memory_fraction(.75)
    bank=CacheBank(a.packet,a.volume);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    bank.validate_packet();jobs=build_jobs(bank.clips);atomic_json(out/'jobs.json',jobs)
    for i,job in enumerate(jobs):
        if a.job is not None and i!=a.job:continue
        atomic_json(out/'current.json',{'index':i,'job':job,'time':time.time(),'total':len(jobs)})
        run_job(bank,job,out/'jobs')
        print(json.dumps({'index':i,'name':job['name'],'complete':True}),flush=True)


if __name__=='__main__':main()
