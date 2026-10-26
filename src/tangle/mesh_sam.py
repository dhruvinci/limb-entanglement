"""3D bodies as SAM prompts (runs in .venv-bmp). Experimental.

s3d_own.py renders the two fitted SAM 3D Body meshes with a depth buffer: the structure is often right (whose leg is
where) but the outlines are coarse, spill onto the mat and some depth orders are wrong. Here the rendered ownership map
only places clicks: N points per fighter spread over the eroded visible part of its body (farthest-point sampling,
so limbs get points, not just the torso), as positives for that fighter and negatives for the other; SAM (as in
dotfix.py) then refines each fighter's source mask to the image edges.

  ../.venv-bmp/bin/python mesh_sam.py --own s3dm --src v4 --out msam --n 8
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from autolabel_common import ROOT, rle_encode, write_json
from paths import SCRATCH
from dotfix import correct, load_sam
from tangle_dot_eval import load_dots
from tangle_review3 import masks as variant_masks

SCR = SCRATCH/'tangle'


def fps(ys, xs, n):
    if not len(xs):
        return []
    pts = np.stack([xs, ys], 1).astype(np.float32)
    pick = [len(pts)//2]
    d = np.linalg.norm(pts - pts[pick[0]], axis=1)
    for _ in range(n - 1):
        pick.append(int(d.argmax()))
        d = np.minimum(d, np.linalg.norm(pts - pts[pick[-1]], axis=1))
    return pts[pick]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pack', default='reviews/tangle-ab-legs')
    ap.add_argument('--own', default='s3dm')
    ap.add_argument('--src', default='v4')
    ap.add_argument('--out', default='msam')
    ap.add_argument('--n', type=int, default=8)
    ap.add_argument('--erode', type=int, default=9)
    a = ap.parse_args()
    frames = sorted({(d['iv'], d['k']) for d in load_dots(ROOT/a.pack)})
    sam = load_sam()
    cache, L, P, res = defaultdict(dict), {}, {}, defaultdict(dict)
    for iv, k in frames:
        base = ROOT/'autolabel/v1'/iv
        if iv not in L:
            L[iv] = json.loads((base/'labels.json').read_text())
            P[iv] = json.loads((base/'tangle/v4/tangle.json').read_text())['prim']
        ms = variant_masks(a.src, base, L[iv], k, P[iv], cache[(a.src, iv)])
        own = np.load(SCR/f's3d_{a.own}_{iv.replace("/", "_")}_{k}.npy')
        img = cv2.imread(L[iv]['frames'][k]['path'])
        dots = []
        for i, c in ((1, 'red'), (2, 'blue')):
            m = cv2.erode((own == i).astype(np.uint8), np.ones((a.erode, a.erode), np.uint8))
            ys, xs = np.nonzero(m)
            dots += [{'x': int(x), 'y': int(y), 'c': c} for x, y in fps(ys, xs, a.n)]
        new = correct(sam, img, ms, dots)
        res[iv][str(k)] = {P[iv][i]: rle_encode(new[i]) for i in range(2)}
        print(iv, k, len(dots), 'mesh clicks', flush=True)
    for iv, lm in res.items():
        d = ROOT/'autolabel/v1'/iv/'tangle'/a.out
        d.mkdir(parents=True, exist_ok=True)
        write_json(d/'tangle.json', {'variant': a.out, 'src': a.src, 'own': a.own, 'n': a.n, 'prim': P[iv], 'label_masks': lm,
                                     'note': 'dotted frames only (mesh_sam.py experiment)'})


if __name__ == '__main__':
    main()
