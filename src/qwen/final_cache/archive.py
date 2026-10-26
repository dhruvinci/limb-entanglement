"""Compact scientific evidence archive and checksum-verified local restoration."""
import os
import tarfile
from pathlib import Path
import numpy as np
from core import atomic_json,sha256,safe_member


LIVE={'heartbeat.json','telemetry.jsonl','current.json','archive-progress.json'}


def build_export(root,max_bytes=4*1024**3):
    root=Path(root);dest=root/'export';dest.mkdir(exist_ok=True);files=[]
    for p in sorted(root.rglob('*')):
        rel=str(p.relative_to(root));safe_member(root,rel)
        if p.is_symlink():raise ValueError('Symlink cannot enter findings archive')
        if not p.is_file() or rel.startswith(('export/','integrity-artifacts/')) or p.name.endswith('.tmp'):continue
        if rel in LIVE or (p.parent==root and p.name.startswith('worker-') and p.suffix=='.log'):continue
        files.append({'path':rel,'bytes':p.stat().st_size,'sha256':sha256(p)})
    total=sum(x['bytes'] for x in files)
    if not files or total>max_bytes:raise ValueError('Empty or over-limit export; investigate before transfer')
    archive=dest/'findings.tar.gz';tmp=dest/'findings.tar.gz.tmp'
    with tarfile.open(tmp,'w:gz',compresslevel=1) as tar:
        for i,f in enumerate(files):
            p=root/f['path']
            if p.stat().st_size!=f['bytes'] or sha256(p)!=f['sha256']:raise ValueError('Artifact changed during packaging')
            tar.add(p,arcname=f['path'],recursive=False)
            if i%64==0:atomic_json(root/'archive-progress.json',{'packed':i+1,'files':len(files),'bytes':total})
    with tmp.open('rb') as f:os.fsync(f.fileno())
    os.replace(tmp,archive)
    receipt={'archive':'findings.tar.gz','archive_bytes':archive.stat().st_size,'archive_sha256':sha256(archive),'unpacked_bytes':total,'files':files}
    atomic_json(dest/'receipt.json',receipt);return receipt


def verify_extract(archive,receipt,destination):
    archive=Path(archive);destination=Path(destination)
    if archive.stat().st_size!=receipt['archive_bytes'] or sha256(archive)!=receipt['archive_sha256']:raise ValueError('Archive checksum mismatch')
    allowed={r['path']:r for r in receipt['files']};seen=set()
    if len(allowed)!=len(receipt['files']):raise ValueError('Duplicate manifest paths')
    destination.mkdir(parents=True,exist_ok=True)
    with tarfile.open(archive,'r:gz') as tar:
        for member in tar:
            if not member.isfile() or member.name not in allowed or member.name in seen:raise ValueError('Unexpected archive member')
            record=allowed[member.name]
            if member.size!=record['bytes']:raise ValueError('Archive member size mismatch')
            target=safe_member(destination,member.name);target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():
                if sha256(target)!=record['sha256']:raise ValueError('Refusing to overwrite different local artifact')
            else:
                tmp=target.with_name(target.name+'.tmp');source=tar.extractfile(member)
                with tmp.open('wb') as f:
                    for block in iter(lambda:source.read(1024*1024),b''):f.write(block)
                    f.flush();os.fsync(f.fileno())
                if sha256(tmp)!=record['sha256']:raise ValueError('Extracted artifact checksum mismatch')
                os.replace(tmp,target)
            seen.add(member.name)
    if seen!=set(allowed):raise ValueError('Archive is missing expected artifacts')
    return {'verified_files':len(seen),'bytes':receipt['unpacked_bytes'],'archive_sha256':receipt['archive_sha256']}


def matched_comparison(candidate,baseline):
    a={(r['clip'],r['frame']):r['metrics'] for r in candidate['records']}
    b={(r['clip'],r['frame']):r['metrics'] for r in baseline['records']};common=sorted(set(a)&set(b));perclip={}
    keys=('actor_iou','contact_accuracy','background_false_positive','contact_margin')
    for clip in sorted({c for c,_ in common}):
        perclip[clip]={}
        for metric in keys:
            vals=[a[k][metric]-b[k][metric] for k in common if k[0]==clip and a[k].get(metric) is not None and b[k].get(metric) is not None]
            perclip[clip][metric]=float(np.mean(vals)) if vals else None
    delta={k:float(np.mean([v[k] for v in perclip.values() if v[k] is not None])) if any(v[k] is not None for v in perclip.values()) else None for k in keys}
    return {'matched_frames':len(common),'candidate_frames':len(a),'baseline_frames':len(b),'per_clip_delta':perclip,'delta':delta}


def correction_counts(before,after,target,contact):
    before=np.asarray(before);after=np.asarray(after);target=np.asarray(target);contact=np.asarray(contact)
    fixed=(before!=target)&(after==target);harm=(before==target)&(after!=target)
    return {'contact_fixed':int((fixed&(contact>0)).sum()),'contact_harmed':int((harm&(contact>0)).sum()),
            'background_fixed':int((fixed&(target==0)).sum()),'background_harmed':int((harm&(target==0)).sum()),
            'known_changes':int(((before!=after)&(target!=255)).sum()),'unknown_changes':int(((before!=after)&(target==255)).sum())}
