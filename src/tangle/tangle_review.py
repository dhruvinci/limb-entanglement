"""Review pack for the tangle tracker (CPU): blind A/B frames (current labels vs tangle) + side-by-side tangle videos.

  ../.venv-pose/bin/python tangle_review.py --variant v1 <trial/interval ...>   -> reviews/tangle-ab/
  ../.venv/bin/python maskfix_ab_serve.py --pack reviews/tangle-ab --port 8770
Frames: every gold-faulted / gold-praised mask frame in the clips, the maskfix A/B frames the reviewer judged not fixed,
then random contact frames and a few apart frames (do-no-harm), capped per clip. key.json (not served) holds the
side of the tangle mask, the category and the tracker's own agreement for each item.
"""
import argparse
import json
import random
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from autolabel_common import ROOT, box_iou, rle_decode

OUT = ROOT/'reviews/tangle-ab'
WORK = Path('/tmp/tangle-work')
COL = [(75, 25, 230), (200, 130, 0)]  # BGR red, blue


def overlay(img, masks):
    a = img.astype(np.float32)
    for c, m in zip(COL, masks):
        if m is not None and m.any():
            a[m] = 0.45*a[m] + 0.55*np.array(c)
    return a.astype(np.uint8)


def crop_of(masks, shape, pad=0.12):
    ys, xs = np.nonzero(np.logical_or.reduce([m for m in masks if m is not None]))
    if not len(xs):
        return 0, 0, shape[1], shape[0]
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
    p = pad*max(x2 - x1, y2 - y1)
    return int(max(0, x1 - p)), int(max(0, y1 - p)), int(min(shape[1], x2 + p)), int(min(shape[0], y2 + p))


def gold_frames():
    rows = json.loads((ROOT/'reviews/gold-v1/mask_postmortem/rows.json').read_text())
    out = {}
    for r in rows:
        if r['origin'] == 'mask_item':
            out[(r['iv'], r['fi'])] = 'gold_praised' if r.get('praised') else ('gold_good' if r['label'] == 'good' else 'gold_faulted')
    key = json.loads((ROOT/'reviews/maskfix-ab/key.json').read_text())
    verdicts = {}
    for line in (ROOT/'reviews/maskfix-ab/verdicts.jsonl').read_text().splitlines():
        v = json.loads(line)
        verdicts[v['id']] = v
    for i, k in key.items():
        v = verdicts.get(int(i), {})
        if v.get('verdict') != ('A' if k['fixed'] == 'A' else 'B'):  # fix not judged better
            iv, fi = k['src'].split('#')
            out.setdefault((iv, int(fi)), 'maskfix_not_fixed')
    return out


def video(iv, variant, T, L, prim):
    """Side-by-side mp4 over the window: current dense masks (left) vs tangle (right), 640x360 each."""
    base = ROOT/'autolabel/v1'/iv
    lm = np.load(base/'tangle'/variant/'labelmap.npz')['labelmap']
    times = T['times']
    D = json.loads((base/'dense/dense.json').read_text()) if (base/'dense/dense.json').exists() else None
    M = np.load(base/'dense/masks.npz') if D else None
    dt = np.array([f['time_s'] for f in D['frames']]) if D else None
    s, e = L['frames'][0]['time_s'], L['frames'][-1]['time_s']
    out = OUT/'videos'/f"{iv.replace('/', '_')}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        n = 0
        for j, t in enumerate(times):
            img = cv2.imread(str(WORK/iv.replace('/', '_')/'frames'/f'{j:05d}.jpg'))
            if img is None:
                continue
            h, w = img.shape[:2]
            lab = cv2.resize(lm[j], (w, h), interpolation=cv2.INTER_NEAREST)
            right = overlay(img, [lab == 1, lab == 2])
            left = img.copy()
            if D is not None and s - 0.05 <= t <= e + 0.05:
                k = int(np.argmin(np.abs(dt - t)))
                if abs(dt[k] - t) < 0.06:
                    H, W = D['shape']
                    ms = [np.unpackbits(M[f'{k}_{a}'])[:H*W].reshape(H, W).astype(bool) if f'{k}_{a}' in M.files else None for a in prim]
                    left = overlay(img, ms)
            tag = 'in window' if s - 0.05 <= t <= e + 0.05 else 'outside window (tangle tracker only)'
            frame = np.hstack([cv2.resize(left, (640, 360)), cv2.resize(right, (640, 360))])
            cv2.putText(frame, f'current  t={t:.1f}s', (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.putText(frame, f'tangle tracker  {tag}', (648, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.imwrite(f'{td}/{n:05d}.jpg', frame)
            n += 1
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', '12.5', '-i', f'{td}/%05d.jpg', '-c:v', 'libx264',
                        '-pix_fmt', 'yuv420p', '-crf', '26', str(out)], check=True)
    return out.name


VIDEO_PAGE = '''<!doctype html><html><head><meta charset="utf-8"><title>Tangle Videos</title>
<style>:root{--bg:#111;--fg:#eee;--mut:#999}body{margin:0;background:var(--bg);color:var(--fg);font:14px system-ui,sans-serif}
main{max-width:1320px;margin:auto;padding:12px 16px}video{width:100%;border-radius:4px;background:#000}h3{margin:18px 0 6px}
p{color:var(--mut)}textarea{width:100%;min-height:44px;background:#1b1b1b;color:var(--fg);border:1px solid #444;border-radius:4px;box-sizing:border-box}</style></head>
<body><main><h2>Tangle tracker: videos</h2><p>Left: current masks (12.5 fps video track, only inside the labelled window).
Right: tangle tracker (forward from the last separated moment before the tangle, backward from the first one after, fused).
Red = first fighter, blue = second. Notes autosave.</p><div id="v"></div><p><a href="/" style="color:#8cf">Back to the blind A/B frames</a></p></main>
<script>
fetch('videos.json').then(r=>r.json()).then(vs=>fetch('/verdicts').then(r=>r.json()).then(V=>{const d=document.getElementById('v');
vs.forEach((x,i)=>{const id='video:'+x.iv;const h=document.createElement('div');
h.innerHTML=`<h3>${i+1}. ${x.iv}</h3><video controls preload="metadata" src="${x.file}"></video><textarea placeholder="Notes on this tangle (autosaved)"></textarea>`;
const t=h.querySelector('textarea');t.value=(V[id]||{}).note||'';let tm;const save=()=>fetch('/verdict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id:id,note:t.value})});
t.oninput=()=>{clearTimeout(tm);tm=setTimeout(save,800)};t.onblur=save;d.appendChild(h)})}));
</script></body></html>'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', default='v1')
    ap.add_argument('--per-clip', type=int, default=5)
    ap.add_argument('--seed', type=int, default=3)
    ap.add_argument('targets', nargs='+')
    a = ap.parse_args()
    rng = random.Random(a.seed)
    special = gold_frames()
    (OUT/'assets').mkdir(parents=True, exist_ok=True)
    picks, videos = [], []
    for iv in a.targets:
        base = ROOT/'autolabel/v1'/iv
        tj = base/'tangle'/a.variant/'tangle.json'
        if not tj.exists():
            continue
        T = json.loads(tj.read_text())
        L = json.loads((base/'labels.json').read_text())
        prim = T['prim']
        videos.append({'iv': iv, 'file': 'videos/' + video(iv, a.variant, T, L, prim)})
        cand = []
        for k, fr in enumerate(L['frames']):
            if str(k) not in T['label_masks']:
                continue
            ins = {i['actor_id']: i for i in fr['instances'] if i['actor_id'] in prim and i['area']}
            contact = len(ins) == 2 and box_iou(ins[prim[0]]['box_xyxy'], ins[prim[1]]['box_xyxy']) > 0.1
            cat = special.get((iv, k)) or ('contact' if contact else 'apart')
            cand.append((k, cat))
        chosen = [c for c in cand if c[1].startswith(('gold', 'maskfix'))]
        rest = [c for c in cand if c not in chosen]
        contact = [c for c in rest if c[1] == 'contact']
        apart = [c for c in rest if c[1] == 'apart']
        n_left = max(0, a.per_clip - len(chosen))
        chosen += rng.sample(contact, min(len(contact), max(0, n_left - 1)))
        if apart and len(chosen) < a.per_clip:
            chosen += rng.sample(apart, 1)
        picks += [(iv, k, cat) for k, cat in chosen]
    rng.shuffle(picks)
    items, key = [], {}
    for n, (iv, k, cat) in enumerate(picks, 1):
        base = ROOT/'autolabel/v1'/iv
        T = json.loads((base/'tangle'/a.variant/'tangle.json').read_text())
        L = json.loads((base/'labels.json').read_text())
        prim = T['prim']
        fr = L['frames'][k]
        img = cv2.imread(fr['path'])
        cur = [rle_decode(i['mask_rle']).astype(bool) if i else None
               for i in (next((x for x in fr['instances'] if x['actor_id'] == p and x['area']), None) for p in prim)]
        new = [rle_decode(T['label_masks'][str(k)][p]).astype(bool) for p in prim]
        box = crop_of(cur + new, img.shape)
        tangle_is_a = rng.random() < 0.5
        sides = {'A': new if tangle_is_a else cur, 'B': cur if tangle_is_a else new}
        names = {}
        for tag, im in (('raw', img), ('A', overlay(img, sides['A'])), ('B', overlay(img, sides['B']))):
            names[tag] = f'assets/{n:03d}_{tag}.jpg'
            cv2.imwrite(str(OUT/names[tag]), im[box[1]:box[3], box[0]:box[2]], [cv2.IMWRITE_JPEG_QUALITY, 88])
        j = T['label_to_span'][str(k)] if str(k) in T['label_to_span'] else T['label_to_span'].get(k)
        ag = T['agreement'][j] if j is not None and j < len(T['agreement']) else None
        items.append({'id': n, 'raw': names['raw'], 'A': names['A'], 'B': names['B'], 'red': prim[0], 'blue': prim[1]})
        key[n] = {'src': f'{iv}#{k}', 'category': cat, 'tangle': 'A' if tangle_is_a else 'B', 'agreement': ag}
    (OUT/'items.json').write_text(json.dumps(items))
    (OUT/'key.json').write_text(json.dumps(key, indent=1))
    (OUT/'videos.json').write_text(json.dumps(videos, indent=1))
    (OUT/'videos.html').write_text(VIDEO_PAGE)
    print(len(items), 'items,', len(videos), 'videos')


if __name__ == '__main__':
    main()
