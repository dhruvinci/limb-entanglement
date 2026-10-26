"""Compare tangle variants on withheld agreement anchors (run with --holdout 1): per fighter IoU of the variant's
mask vs the withheld label mask. Usage: tangle_holdout.py <list file> <variant>..."""
import json
import sys

import numpy as np

from autolabel_common import ROOT

ivs, variants = open(sys.argv[1]).read().split(), sys.argv[2:]
res = {v: {} for v in variants}
for iv in ivs:
    for v in variants:
        f = ROOT/'autolabel/v1'/iv/'tangle'/v/'tangle.json'
        if f.exists():
            res[v][iv] = json.loads(f.read_text()).get('heldout_iou', {})
common = [iv for iv in ivs if all(iv in res[v] for v in variants)]
keys = {iv: set.intersection(*[set(res[v][iv]) for v in variants]) for iv in common}
print('clips', len(common), 'held-out frames', sum(len(k) for k in keys.values()))
for v in variants:
    vals = np.array([res[v][iv][k] for iv in common for k in sorted(keys[iv])])
    if len(vals):
        mn = vals.min(1)
        print(f'{v:12s} mean IoU {vals.mean():.4f}  worst-fighter mean {mn.mean():.4f}  frames worst<0.7: {(mn < 0.7).mean():.3f}  <0.5: {(mn < 0.5).mean():.3f}')
if len(variants) > 1:
    base = variants[0]
    for v in variants[1:]:
        d = np.array([np.min(res[v][iv][k]) - np.min(res[base][iv][k]) for iv in common for k in keys[iv]])
        print(f'{v} vs {base}: worst-fighter IoU better on {(d > 0.02).sum()}, worse on {(d < -0.02).sum()}, same {(abs(d) <= 0.02).sum()}')
