"""Per-clip colour model test: Lab histograms of each fighter from the clip's agreement-anchor frames (eroded masks),
then classify each dot (7x7 patch) red vs blue. How much of the leg ownership is decidable by appearance alone?"""
import json, sys
from collections import defaultdict
import cv2, numpy as np
sys.path.insert(0, '.')
from autolabel_common import ROOT, rle_decode
from paths import SCRATCH
rows = json.load(open(SCRATCH/'dot_rows.json'))
B = 12
def hist(lab, m):
    h = cv2.calcHist([lab], [0, 1, 2], m.astype(np.uint8), [B]*3, [0, 256]*3) + 1e-3
    return h/h.sum()
res = defaultdict(lambda: [0, 0])
for iv in sorted({r['iv'] for r in rows}):
    base = ROOT/'autolabel/v1'/iv
    L = json.loads((base/'labels.json').read_text()); T = json.loads((base/'tangle/v4/tangle.json').read_text()); prim = T['prim']
    anc = [a['label_index'] for a in T['agreement_anchors']][::2][:10]
    H = [np.zeros([B]*3, np.float32) for _ in range(2)]
    for k in anc:
        fr = L['frames'][k]; img = cv2.cvtColor(cv2.imread(fr['path']), cv2.COLOR_BGR2LAB)
        ins = {i['actor_id']: i for i in fr['instances']}
        for x, p in enumerate(prim):
            m = cv2.erode(rle_decode(ins[p]['mask_rle']).astype(np.uint8), np.ones((9, 9), np.uint8))
            H[x] += hist(img, m)
    H = [h/h.sum() for h in H]
    imgs = {}
    for r in rows:
        if r['iv'] != iv or r['c'] == 'none': continue
        if r['k'] not in imgs: imgs[r['k']] = cv2.cvtColor(cv2.imread(L['frames'][r['k']]['path']), cv2.COLOR_BGR2LAB)
        im = imgs[r['k']]; y, x = r['y'], r['x']
        patch = (im[max(0, y-3):y+4, max(0, x-3):x+4].reshape(-1, 3).astype(int)*B)//256
        ll = [np.log(H[i][patch[:, 0], patch[:, 1], patch[:, 2]]).sum() for i in range(2)]
        pred = 'red' if ll[0] > ll[1] else 'blue'
        margin = abs(ll[0] - ll[1])/len(patch)
        v4ok = r['got']['v4'] == r['c']
        for g in ('all', 'v4 wrong' if not v4ok else 'v4 right', f"part:{r['part']}"):
            res[g][0] += pred == r['c']; res[g][1] += 1
        if margin > 1.0:
            res['confident (margin>1)'][0] += pred == r['c']; res['confident (margin>1)'][1] += 1
            if not v4ok: res['confident & v4 wrong'][0] += pred == r['c']; res['confident & v4 wrong'][1] += 1
for g, (a, b) in sorted(res.items()): print(f'{g:24s} {a}/{b} ({a/b:.0%})')
