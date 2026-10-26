"""Review pack on the held-out TEST clips (picked at random before any experiment; never tuned on), CPU.

Two blind A/B questions, mixed and shuffled (key.json, not served, records which side is which):
  current_vs_tangle  production masks (labels.json) vs tangle tracker v4, on random contact frames: replicates the
                     dev-set result (13-2 for the tracker) on unbiased clips
  tangle_vs_zoom     tangle v4 vs v4 + zoom (each segment tracked in a crop around the pair, ~1.8x the pixels), on the
                     frames where the two differ most, plus one random contact frame per clip
Videos: three panels per clip, current | tangle | tangle+zoom (labelled, not blind).

  ../.venv-pose/bin/python tangle_review2.py --a v4 --b v5z <list file>      -> reviews/tangle-ab-test/
  ../.venv/bin/python maskfix_ab_serve.py --pack reviews/tangle-ab-test --port 8771 --host 0.0.0.0
"""
import argparse
import json
import random
import subprocess
import tempfile

import cv2
import numpy as np

from autolabel_common import ROOT, box_iou, rle_decode
from tangle_review import VIDEO_PAGE, WORK, crop_of, overlay

OUT = ROOT/'reviews/tangle-ab-test'


def iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum()/u) if u else 1.0


def masks_of(T, k, prim):
    return [rle_decode(T['label_masks'][str(k)][p]).astype(bool) for p in prim]


def video(iv, va, vb, L, prim):
    base = ROOT/'autolabel/v1'/iv
    T = json.loads((base/'tangle'/va/'tangle.json').read_text())
    la = np.load(base/'tangle'/va/'labelmap.npz')['labelmap']
    lb = np.load(base/'tangle'/vb/'labelmap.npz')['labelmap']
    D = json.loads((base/'dense/dense.json').read_text())
    M = np.load(base/'dense/masks.npz')
    dt = np.array([f['time_s'] for f in D['frames']])
    s, e = L['frames'][0]['time_s'], L['frames'][-1]['time_s']
    out = OUT/'videos'/f"{iv.replace('/', '_')}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        n = 0
        for j, t in enumerate(T['times']):
            if not (s - 0.05 <= t <= e + 0.05):
                continue
            img = cv2.imread(str(WORK/iv.replace('/', '_')/'frames'/f'{j:05d}.jpg'))
            if img is None:
                continue
            h, w = img.shape[:2]
            panels = []
            k = int(np.argmin(np.abs(dt - t)))
            cur = img.copy()
            if abs(dt[k] - t) < 0.06:
                H, W = D['shape']
                cur = overlay(img, [np.unpackbits(M[f'{k}_{a}'])[:H*W].reshape(H, W).astype(bool) if f'{k}_{a}' in M.files else None
                                    for a in prim])
            panels.append(cur)
            for lm in (la, lb):
                lab = cv2.resize(lm[j], (w, h), interpolation=cv2.INTER_NEAREST)
                panels.append(overlay(img, [lab == 1, lab == 2]))
            frame = np.hstack([cv2.resize(p, (480, 270)) for p in panels])
            for x, name in enumerate(('current', 'tangle', 'tangle + zoom')):
                cv2.putText(frame, name, (8 + 480*x, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.imwrite(f'{td}/{n:05d}.jpg', frame)
            n += 1
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', '12.5', '-i', f'{td}/%05d.jpg', '-c:v', 'libx264',
                        '-pix_fmt', 'yuv420p', '-crf', '26', str(out)], check=True)
    return out.name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', default='v4')
    ap.add_argument('--b', default='v5z')
    ap.add_argument('--n-current', type=int, default=3)
    ap.add_argument('--n-zoom', type=int, default=3)
    ap.add_argument('--seed', type=int, default=11)
    ap.add_argument('list')
    a = ap.parse_args()
    rng = random.Random(a.seed)
    (OUT/'assets').mkdir(parents=True, exist_ok=True)
    picks, videos = [], []
    for iv in open(a.list).read().split():
        base = ROOT/'autolabel/v1'/iv
        if not all((base/'tangle'/v/'tangle.json').exists() for v in (a.a, a.b)):
            print('missing', iv)
            continue
        TA = json.loads((base/'tangle'/a.a/'tangle.json').read_text())
        TB = json.loads((base/'tangle'/a.b/'tangle.json').read_text())
        L = json.loads((base/'labels.json').read_text())
        prim = TA['prim']
        videos.append({'iv': iv, 'file': 'videos/' + video(iv, a.a, a.b, L, prim)})
        contact, diff = [], []
        for k, fr in enumerate(L['frames']):
            if str(k) not in TA['label_masks'] or str(k) not in TB['label_masks']:
                continue
            ins = {i['actor_id']: i for i in fr['instances'] if i['actor_id'] in prim and i['area']}
            if len(ins) == 2 and box_iou(ins[prim[0]]['box_xyxy'], ins[prim[1]]['box_xyxy']) > 0.1:
                contact.append(k)
                ma, mb = masks_of(TA, k, prim), masks_of(TB, k, prim)
                diff.append((min(iou(ma[0], mb[0]), iou(ma[1], mb[1])), k))
        cur_k = rng.sample(contact, min(a.n_current, len(contact)))
        picks += [(iv, k, 'current_vs_tangle') for k in cur_k]
        diff.sort()
        zk = [k for d, k in diff if d < 0.97][:a.n_zoom]
        left = [k for k in contact if k not in zk]
        if left:
            zk.append(rng.choice(left))
        picks += [(iv, k, 'tangle_vs_zoom') for k in zk]
    rng.shuffle(picks)
    items, key = [], {}
    for n, (iv, k, q) in enumerate(picks, 1):
        base = ROOT/'autolabel/v1'/iv
        TA = json.loads((base/'tangle'/a.a/'tangle.json').read_text())
        TB = json.loads((base/'tangle'/a.b/'tangle.json').read_text())
        L = json.loads((base/'labels.json').read_text())
        prim = TA['prim']
        fr = L['frames'][k]
        img = cv2.imread(fr['path'])
        if q == 'current_vs_tangle':
            first = [rle_decode(i['mask_rle']).astype(bool) if i else None
                     for i in (next((x for x in fr['instances'] if x['actor_id'] == p and x['area']), None) for p in prim)]
            second, names = masks_of(TA, k, prim), ('current', a.a)
        else:
            first, second, names = masks_of(TA, k, prim), masks_of(TB, k, prim), (a.a, a.b)
        box = crop_of(first + second, img.shape)
        swap = rng.random() < 0.5
        sides = {'A': second if swap else first, 'B': first if swap else second}
        files = {}
        for tag, im in (('raw', img), ('A', overlay(img, sides['A'])), ('B', overlay(img, sides['B']))):
            files[tag] = f'assets/{n:03d}_{tag}.jpg'
            cv2.imwrite(str(OUT/files[tag]), im[box[1]:box[3], box[0]:box[2]], [cv2.IMWRITE_JPEG_QUALITY, 88])
        items.append({'id': n, 'raw': files['raw'], 'A': files['A'], 'B': files['B'], 'red': prim[0], 'blue': prim[1]})
        key[n] = {'src': f'{iv}#{k}', 'question': q, 'A': names[1] if swap else names[0], 'B': names[0] if swap else names[1]}
    (OUT/'items.json').write_text(json.dumps(items))
    (OUT/'key.json').write_text(json.dumps(key, indent=1))
    (OUT/'videos.json').write_text(json.dumps(videos, indent=1))
    (OUT/'videos.html').write_text(VIDEO_PAGE.replace('Left: current masks (12.5 fps video track, only inside the labelled window).\nRight: tangle tracker (forward from the last separated moment before the tangle, backward from the first one after, fused).',
                                                      'Panels: current masks | tangle tracker | tangle tracker + zoom (each segment tracked in a crop around the pair).'))
    print(len(items), 'items,', len(videos), 'videos', {q: sum(1 for p in picks if p[2] == q) for q in ('current_vs_tangle', 'tangle_vs_zoom')})


if __name__ == '__main__':
    main()
