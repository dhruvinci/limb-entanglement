"""Make reviewer dots hold in an existing corrections file (runs in .venv-bmp). Experimental.

Correction files made before dotfix.py --enforce (corr_legs_near, the final-pack split) contradict about 1 dot in 10
they were made from (a foot or hand SAM's refinement stopped short of). This applies dotfix.enforce to every frame in
the file, with every red/blue dot any of the packs holds for that frame, and writes a new file (inputs untouched).
Unowned pixels inside the BMP pair (bmpw) may be filled by a dot's whole SAM part (see enforce).

  ../.venv-bmp/bin/python scripts/dotenforce.py IN.json OUT.json reviews/tangle-ab-legs,reviews/tangle-ab-near,...
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from autolabel_common import ROOT, rle_decode, rle_encode
from dotfix import enforce, load_sam, score
from tangle_dot_eval import load_dots
from tangle_review3 import masks as variant_masks


def main():
    src, dst, packs = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), sys.argv[3].split(',')
    corr = json.loads(src.read_text())
    dots = defaultdict(list)
    for p in packs:
        for d in load_dots(ROOT/p):
            dots[(d['iv'], d['k'])].append(d)
    sam = load_sam()
    out, tot, cache = {}, np.zeros(3, int), defaultdict(dict)
    for iv, fr in sorted(corr.items()):
        base = ROOT/'autolabel/v1'/iv
        L = json.loads((base/'labels.json').read_text())
        prim = json.loads((base/'tangle/v4/tangle.json').read_text())['prim']
        out[iv] = {}
        for k, c in sorted(fr.items(), key=lambda x: int(x[0])):
            ds = dots.get((iv, int(k)), [])
            ms = [rle_decode(c[p]).astype(bool) for p in prim]
            bm = [x for x in variant_masks('bmpw', base, L, int(k), prim, cache[iv]) if x is not None]
            body = np.logical_or.reduce(bm) if bm else None
            new, nf = enforce(sam, cv2.imread(L['frames'][int(k)]['path']), ms, ds, body=body) if ds else (ms, 0)
            out[iv][k] = {p: (rle_encode(new[i]) if nf else c[p]) for i, p in enumerate(prim)}
            n = sum(d['c'] in ('red', 'blue') for d in ds)
            r0, r1 = score(ms, [d for d in ds if d['c'] != 'none']), score(new, [d for d in ds if d['c'] != 'none'])
            tot += (r0, r1, n)
            if nf:
                moved = int(((ms[0] != new[0]) | (ms[1] != new[1])).sum())
                print(f'{iv}#{k}: red/blue dots right {r0} -> {r1} / {n} (enforced {nf}, {moved} px changed)', flush=True)
    dst.write_text(json.dumps(out))
    print(f'TOTAL red/blue dots right {tot[0]} -> {tot[1]} / {tot[2]}; wrote {dst}')


if __name__ == '__main__':
    main()
