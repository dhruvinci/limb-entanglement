"""Matched comparisons, localized harms and readable scientific evidence."""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image,ImageDraw,ImageOps
from core import atomic_json,read_json
from archive import matched_comparison,correction_counts


def contact_gate(delta):
    return delta.get('contact_accuracy') is not None and delta.get('actor_iou') is not None and delta['contact_accuracy']>=.10 and delta['actor_iou']>=-.02


def semantic_comparison(candidate,baseline):
    if not candidate.get('train_rows') or sorted(candidate['train_rows'])!=sorted(baseline.get('train_rows',[])):
        raise ValueError('Unmatched semantic training rows')
    comp=matched_comparison(candidate,baseline)
    return {**comp,'training_rows_matched':True,'contact_screen_pass':contact_gate(comp['delta'])}


def temporal_improvement(learned,baselines):
    valid=[x for x in baselines if x is not None]
    if learned is None or not valid:return {'relative_error_reduction':None,'passes':False}
    best=max(valid);reduction=(learned-best)/(1-best) if best<1 else None
    return {'strongest_simple_iou':best,'relative_error_reduction':reduction,'passes':reduction is not None and reduction>=.1}


def probabilities(logits,shape):
    x=torch.as_tensor(np.asarray(logits).copy()).float()
    return F.interpolate(x[None],size=shape,mode='bilinear',align_corners=False).softmax(1)[0].numpy()


def render_overlay(image,logits,label,contact,path,title):
    im=ImageOps.contain(image.convert('RGB'),(320,256));w,h=im.size
    p=probabilities(logits,label.shape);pred=p.argmax(0)
    palette=np.array([[20,20,20],[255,135,30],[20,195,255]],np.uint8)
    mask=Image.fromarray(palette[pred]).resize((w,h),Image.Resampling.NEAREST)
    overlay=Image.blend(im,mask,.45)
    truth=np.full((*label.shape,3),110,np.uint8);known=label!=255;truth[known]=palette[label[known]]
    truth=Image.fromarray(truth).resize((w,h),Image.Resampling.NEAREST)
    margin=p[2]-p[1];heat=np.full((*margin.shape,3),128.,np.float32)
    heat+=np.maximum(margin,0)[...,None]*np.array([-108,67,127])+np.maximum(-margin,0)[...,None]*np.array([127,7,-98])
    heat=Image.fromarray(heat.clip(0,255).astype(np.uint8)).resize((w,h),Image.Resampling.BILINEAR)
    heat=Image.blend(im,heat,.65)
    canvas=Image.new('RGB',(960,300),(245,245,245));draw=ImageDraw.Draw(canvas)
    for i,(panel,text) in enumerate(((overlay,'Prediction over source'),(truth,'Reference: gray = ignored'),(heat,'A2 blue / A1 orange margin'))):
        canvas.paste(panel,(320*i+(320-w)//2,44+(256-h)//2));draw.text((320*i+5,25),text,fill=(0,0,0))
    draw.text((5,5),title[:145],fill=(0,0,0))
    # Contact references are outlined on the ownership overlay, not invented labels.
    from audit import contact_regions
    xoff=(320-w)//2;yoff=44+(256-h)//2
    for r in contact_regions(pred,contact):
        x0,y0,x1,y1=r['box_xyxy'];xy=(xoff+x0/label.shape[1]*w,yoff+y0/label.shape[0]*h,xoff+x1/label.shape[1]*w,yoff+y1/label.shape[0]*h)
        draw.rectangle(xy,outline=(0,255,0) if r['accuracy']==1 else (255,0,180),width=2)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);canvas.save(path)


def group_rows(results):
    groups=defaultdict(list)
    for result in results:
        if 'equal_clip' not in result:continue
        job=result['job'];cfg={k:v for k,v in job.items() if k not in ('name','heldout','seed','binding','controls')}
        cfg['cohort']='armbar' if job['heldout']=='armbar' else 'breadth'
        groups[json.dumps(cfg,sort_keys=True)].append(result)
    output=[]
    for group,values in sorted(groups.items()):
        metrics={}
        for metric in ('actor_iou','contact_accuracy','contact_margin','background_false_positive'):
            vals=[r['equal_clip'].get(metric) for r in values if r['equal_clip'].get(metric) is not None]
            metrics[metric]={'mean':float(np.mean(vals)),'std_across_source_seed_cells':float(np.std(vals)),'cells':len(vals)} if vals else None
        output.append({'configuration':json.loads(group),'metrics':metrics,'source_seed_cells':len(values),'sources':sorted({r['job']['heldout'] for r in values})})
    return output


def make_report(bank,jobs,root):
    root=Path(root);results=[];by_name={};comparisons=[];temporal=[];images=[];localized=[]
    lookup={(r['clip'],r['frame']):r for r in bank.rows}
    for job in jobs:
        r=read_json(root/'jobs'/job['name']/'result.json');results.append(r);by_name[job['name']]=r
    def is_base(j):return j['kind']=='neural' and j.get('mode')=='appearance' and j.get('width',24)==24 and j.get('fraction',1)==1 and j.get('contact_weight',1)==1 and j['representation']=='full11+merged'
    bases={(j['heldout'],j['seed'],j.get('eligibility_condition')):j for j in jobs if is_base(j)}
    for n,job in enumerate(jobs):
        out=root/'jobs'/job['name'];result=by_name[job['name']]
        if job['kind']=='temporal':
            for direction in ('causal','offline'):
                for condition in ('observed','hidden','corrupted'):
                    key=f'{direction}-{condition}';m=result['comparisons'];learned=m[key+'-ordered']['actor_iou']
                    simple=[m[key+'-'+b]['actor_iou'] for b in ('static','persistence','interpolation','correspondence') if key+'-'+b in m]
                    temporal.append({'job':job['name'],'heldout':job['heldout'],'seed':job['seed'],'condition':key,
                        'ordered':m[key+'-ordered'],'reversed':m[key+'-reversed'],'shuffled':m[key+'-shuffled'],**temporal_improvement(learned,simple)})
        if job['kind']=='neural' and job.get('mode') in ('global','local'):
            eligibility='contact_ownership' if job.get('condition')=='contact_ownership' else None
            basejob=bases[job['heldout'],job['seed'],eligibility];base=by_name[basejob['name']]
            comp=semantic_comparison(result,base);comparisons.append({'candidate':job['name'],'baseline':basejob['name'],**comp})
            bmap={(r['clip'],r['frame']):r for r in base['records']}
            for row in result['records']:
                key=(row['clip'],row['frame'])
                if key not in bmap:continue
                native=lookup[key];target=bank.labels[native['id']];contact=bank.contacts[native['id']]
                with np.load(out/row['prediction']) as z:a=probabilities(z['logits'],target.shape).argmax(0)
                with np.load(root/'jobs'/basejob['name']/bmap[key]['prediction']) as z:b=probabilities(z['logits'],target.shape).argmax(0)
                localized.append({'candidate':job['name'],'baseline':basejob['name'],'clip':key[0],'frame':key[1],**correction_counts(b,a,target,contact)})
        if result.get('records'):
            for control in job.get('controls',[]):
                p=out/'controls'/control/'result.json'
                if p.exists():comparisons.append({'candidate':job['name'],'baseline':job['name']+'/controls/'+control,**matched_comparison(result,read_json(p))})
        if job['seed']==7 and job['kind']=='neural' and job.get('mode') in ('appearance','local') and job.get('control','real')=='real' and not any(k in job for k in ('width','fraction','contact_weight','condition','layer')):
            for r in result['records']:
                native=lookup[r['clip'],r['frame']]
                with np.load(out/r['prediction']) as z:logits=z['logits'].copy()
                path=root/'visuals'/job['name']/f"{r['clip']}-{r['frame']:06d}.png"
                render_overlay(bank.image(native),logits,bank.labels[native['id']],bank.contacts[native['id']],path,f"{r['clip']} frame {r['frame']} | {job.get('mode')} | exploratory sparse labels")
                images.append(str(path.relative_to(root)))
        if n%16==0:atomic_json(root/'archive-progress.json',{'phase':'report','jobs':n+1,'total':len(jobs)})
    findings={'groups':group_rows(results),'comparisons':comparisons,'temporal':temporal,'localized_fixes_and_harms':localized,'visuals':images,
        'limitations':['Exploratory reused source clips and sparse pseudo/AI-reviewed labels, not independent ground truth.',
        'Reference-frame appearance supervision is assisted, and reference frames are not scored.',
        'Old pooled tensors retain the original spatial grouping error; only raw full tensors can be correctly repooled.',
        'Most readouts use a fixed 128-channel projection; dedicated 64/256/native L11 checks test compression sensitivity.',
        'Causal temporal models never consume full-video semantic states. Hidden-center recovery is not validation of physically invisible anatomy.',
        'Raw scores and risk/coverage are not calibrated probabilities. Do not infer accuracy over ignored pixels.',
        'Four held-out breadth sources share venue/context and were studied previously. Seeds are not new independent videos.',
        'Per-cell gates are screening diagnostics, not campaign-level success claims; inspect matched controls, all sources and all seeds.']}
    atomic_json(root/'findings.json',findings);return findings
