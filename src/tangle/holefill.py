"""Hole fill for the tangle tracker's masks (CPU). Experimental.

Why: reviewer dots on frames near a correction show the corrected tracker (v4w) mostly trades swaps for holes: limb
pixels nobody owns. BMP (bmp_tangle.py: PMPose + SAM-pose2seg full-body masks) covers more of each body. Here the
tracker's owners are kept, and pixels it leaves unowned are given to the fighter BMP assigns them to.
Optional --despeckle: drop connected islands under 0.5% of a fighter's area and fill holes inside it (reviewer:
"speckling").

  ../.venv-pose/bin/python holefill.py --src v4w --fill bmpw --out v4wf <trial/interval ...>
"""
import argparse
import json

import cv2
import numpy as np

from autolabel_common import ROOT, rle_decode, rle_encode, write_json


def despeckle(m, other, frac=0.005):
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    if n <= 1:
        return m
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= frac*m.sum()
    keep[1 + np.argmax(st[1:, cv2.CC_STAT_AREA])] = True
    m = keep[lab]
    inv = (~m & ~other).astype(np.uint8)   # only unowned pixels can be filled as holes
    n, lab, st, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    for j in range(1, n):   # holes: background components not touching the border and small
        x, y, w, h, a = st[j]
        if x > 0 and y > 0 and x + w < m.shape[1] and y + h < m.shape[0] and a < 0.02*m.sum():
            m[lab == j] = True
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='v4w')
    ap.add_argument('--fill', default='bmpw')
    ap.add_argument('--out', default='v4wf')
    ap.add_argument('--despeckle', type=int, default=0)
    ap.add_argument('targets', nargs='+')
    a = ap.parse_args()
    for iv in a.targets:
        base = ROOT/'autolabel/v1'/iv
        T = json.loads((base/'tangle'/a.src/'tangle.json').read_text())
        F = json.loads((base/'tangle'/a.fill/'tangle.json').read_text())['label_masks']
        prim, out, added = T['prim'], {}, 0
        for k, lm in T['label_masks'].items():
            m = [rle_decode(lm[p]).astype(bool) for p in prim]
            if k in F:
                f = [rle_decode(F[k][p]).astype(bool) for p in prim]
                free = ~(m[0] | m[1])
                g = [free & f[0] & ~f[1], free & f[1] & ~f[0]]
                added += int(g[0].sum() + g[1].sum())
                m = [m[0] | g[0], m[1] | g[1]]
            if a.despeckle:
                m = [despeckle(m[0], m[1]), despeckle(m[1], m[0])]
                m[1] &= ~m[0]
            out[k] = {p: rle_encode(m[i]) for i, p in enumerate(prim)}
        d = base/'tangle'/a.out
        d.mkdir(parents=True, exist_ok=True)
        write_json(d/'tangle.json', {**{x: y for x, y in T.items() if x != 'label_masks'}, 'variant': a.out, 'src': a.src,
                                     'fill': a.fill, 'despeckle': a.despeckle, 'label_masks': out})
        print(iv, len(out), 'frames, filled px', added, flush=True)


if __name__ == '__main__':
    main()
