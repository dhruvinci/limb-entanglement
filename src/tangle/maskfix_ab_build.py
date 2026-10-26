"""Blind A/B review pack for mask_fix: original (pre-maskfix) vs fixed masks on random frames whose primary mask changed.

  ../.venv-pose/bin/python maskfix_ab_build.py      -> reviews/maskfix-ab/{items.json, key.json, assets/}
A/B order is random per item; key.json (which side is the fix) is not served to the page.
"""
import glob
import gzip
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

from autolabel_common import ROOT, rle_decode

OUT = ROOT/'reviews/maskfix-ab'
COL = [(230, 25, 75), (0, 130, 200)]
N_DENSE, N_CLEANED = 32, 8


def records():
    run1 = json.load(open(ROOT/'reviews/autolabel-v1/maskfix/records.json'))
    run2 = [r for p in sorted(glob.glob(str(ROOT/'reviews/autolabel-v1/maskfix/batches/b*/records.json'))) for r in json.load(open(p))]
    redone = {r['iv'] for r in run2}
    return [r for r in run1 if r['iv'] not in redone] + run2  # run-2 intervals were re-fixed from the original labels


def overlay(img, f, prim):
    a = np.asarray(img).astype(float)
    for c, act in zip(COL, prim):
        i = next((i for i in f['instances'] if i['actor_id'] == act), None)
        if i and i['area']:
            m = rle_decode(i['mask_rle']).astype(bool)
            a[m] = 0.45*a[m] + 0.55*np.array(c)
    return Image.fromarray(a.astype(np.uint8))


def main():
    rng = random.Random(11)
    recs = [r for r in records() if r['primary']]
    by_frame = {}
    for r in recs:
        by_frame.setdefault((r['iv'], r['fi']), set()).add(r['action'])
    dense = sorted(k for k, v in by_frame.items() if 'dense' in v)
    cleaned = sorted(k for k, v in by_frame.items() if v == {'cleaned'})
    pick = [(k, 'dense') for k in rng.sample(dense, N_DENSE)] + [(k, 'cleaned') for k in rng.sample(cleaned, N_CLEANED)]
    rng.shuffle(pick)
    items, key = [], {}
    for n, ((iv, fi), kind) in enumerate(pick, 1):
        base = ROOT/'autolabel/v1'/iv
        with gzip.open(base/'labels.pre-maskfix.json.gz', 'rt') as fh:
            old = json.load(fh)
        new = json.loads((base/'labels.json').read_text())
        prim = [a['actor_id'] for a in new['actors'] if a.get('primary')][:2]
        fo, fn = old['frames'][fi], new['frames'][fi]
        img = Image.open(ROOT/fn['path']).convert('RGB')
        bs = [i['box_xyxy'] for f in (fo, fn) for i in f['instances'] if i['actor_id'] in prim and i['area']]
        x1, y1, x2, y2 = min(b[0] for b in bs), min(b[1] for b in bs), max(b[2] for b in bs), max(b[3] for b in bs)
        p = 0.12*max(x2 - x1, y2 - y1)
        box = (int(max(0, x1 - p)), int(max(0, y1 - p)), int(min(img.width, x2 + p)), int(min(img.height, y2 + p)))
        fixed_is_a = rng.random() < 0.5
        sides = {'A': fn if fixed_is_a else fo, 'B': fo if fixed_is_a else fn}
        names = {}
        for tag, im in (('raw', img), ('A', overlay(img, sides['A'], prim)), ('B', overlay(img, sides['B'], prim))):
            names[tag] = f'assets/{n:03d}_{tag}.jpg'
            im.crop(box).save(OUT/names[tag], quality=88)
        items.append({'id': n, 'raw': names['raw'], 'A': names['A'], 'B': names['B'],
                      'red': prim[0], 'blue': prim[1] if len(prim) > 1 else None})
        key[n] = {'src': f'{iv}#{fi}', 'kind': kind, 'fixed': 'A' if fixed_is_a else 'B'}
    (OUT/'items.json').write_text(json.dumps(items))
    (OUT/'key.json').write_text(json.dumps(key, indent=1))
    print(len(items), 'items')


if __name__ == '__main__':
    main()
