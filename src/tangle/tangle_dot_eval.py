"""Score mask variants against the reviewer's dots in a tangle A/B pack (CPU).

Dots are placed on the review page (maskfix_ab.html) and mark the truth at a pixel, independent of A/B: 'red' = belongs
to the red fighter, 'blue' = the blue fighter, 'none' = neither (mat, background, a third person). Stored as fractions
of the crop; items.json 'box' maps them back to full-frame pixels. Because they are method-independent, any variant
with label masks on the same frames can be scored, including ones built after the review.

  ../.venv-pose/bin/python tangle_dot_eval.py reviews/tangle-ab-legs current v4 v4lf bmp6c
Prints per variant: dots whose pixel the variant labels correctly, overall and by dot colour.
"""
import json
import sys
from collections import defaultdict

from autolabel_common import ROOT
from tangle_review3 import masks


def load_dots(pack):
    items = {it['id']: it for it in json.loads((pack/'items.json').read_text())}
    key = json.loads((pack/'key.json').read_text())
    last = {}
    for line in (pack/'verdicts.jsonl').read_text().splitlines():
        if line.strip():
            v = json.loads(line)
            last[v['id']] = v
    out = []
    for i, v in last.items():
        if not v.get('dots') or i not in items or 'box' not in items[i]:
            continue
        x1, y1, x2, y2 = items[i]['box']
        iv, k = key[str(i)]['src'].split('#')
        for d in v['dots']:
            out.append({'iv': iv, 'k': int(k), 'c': d['c'], 'x': int(x1 + d['x']*(x2 - x1)), 'y': int(y1 + d['y']*(y2 - y1))})
    return out


def main():
    pack, variants = ROOT/sys.argv[1], sys.argv[2:]
    dots = load_dots(pack)
    print(len(dots), 'dots on', len({(d['iv'], d['k']) for d in dots}), 'frames')
    labels, prims, cache = {}, {}, defaultdict(dict)
    for name in variants:
        hit, tot = defaultdict(int), defaultdict(int)
        for d in dots:
            base = ROOT/'autolabel/v1'/d['iv']
            if d['iv'] not in labels:
                labels[d['iv']] = json.loads((base/'labels.json').read_text())
                prims[d['iv']] = json.loads((base/'tangle/v4/tangle.json').read_text())['prim']
            try:
                m = masks(name, base, labels[d['iv']], d['k'], prims[d['iv']], cache[(name, d['iv'])])
            except FileNotFoundError:
                m = None
            if m is None:
                continue
            red, blue = (bool(x[d['y'], d['x']]) if x is not None else False for x in m)
            got = 'red' if red else 'blue' if blue else 'none'
            for g in ('all', d['c']):
                tot[g] += 1
                hit[g] += got == d['c']
        print(f'{name:10s}', '  '.join(f'{g} {hit[g]}/{tot[g]} ({hit[g]/tot[g]:.0%})' for g in ('all', 'red', 'blue', 'none') if tot[g]))


if __name__ == '__main__':
    main()
