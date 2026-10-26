"""Blind A/B pack for leg-entanglement clips with several candidate pairs (CPU).

  ../.venv-pose/bin/python tangle_review3.py --out reviews/tangle-ab-legs --pair current:v4:leg:3 --pair v4:v4lf:diff:3 \
      --pair v4:bmp6c:diff:3 <list file>
Each --pair is first:second:picker:n_per_clip; picker 'leg' = random frames Astra labelled leg_entanglement /
half_guard / closed_guard, 'diff' = frames where the two differ most (worst-fighter IoU < 0.97), plus one 'leg' frame.
'current' means the production labels.json masks. key.json (not served) records sides and the pair.
Videos: current | tangle v4 over the labelled window, as in tangle_review2 (two panels).
"""
import argparse
import json
import random

import cv2
import numpy as np

from autolabel_common import ROOT, rle_decode
from tangle_review import VIDEO_PAGE, crop_of, overlay
from tangle_review2 import iou

LEG = {'leg_entanglement', 'half_guard', 'closed_guard'}


def masks(name, base, L, k, prim, cache):
    if name == 'current':
        fr = L['frames'][k]
        return [rle_decode(i['mask_rle']).astype(bool) if i else None
                for i in (next((x for x in fr['instances'] if x['actor_id'] == p and x['area']), None) for p in prim)]
    if name not in cache:
        cache[name] = json.loads((base/'tangle'/name/'tangle.json').read_text())
    lm = cache[name]['label_masks'].get(str(k))
    return [rle_decode(lm[p]).astype(bool) for p in prim] if lm else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='reviews/tangle-ab-legs')
    ap.add_argument('--pair', action='append', required=True)
    ap.add_argument('--seed', type=int, default=17)
    ap.add_argument('--scores', default=None, help="picker 'unsure': JSON {interval: {label_index: score}}, highest first")
    ap.add_argument('--skip-dotted', default='', help='comma-separated packs whose dotted frames are not picked again (any picker)')
    ap.add_argument('list')
    a = ap.parse_args()
    OUT = ROOT/a.out
    rng = random.Random(a.seed)
    (OUT/'assets').mkdir(parents=True, exist_ok=True)
    picks, videos = [], []
    for iv in open(a.list).read().split():
        base = ROOT/'autolabel/v1'/iv
        if not (base/'tangle/v4/tangle.json').exists():
            continue
        L = json.loads((base/'labels.json').read_text())
        prim = json.loads((base/'tangle/v4/tangle.json').read_text())['prim']
        cache = {}
        leg = [k for k, f in enumerate(L['frames']) if f.get('pair_configuration') in LEG]
        # frames the reviewer already dotted are corrections or test points of some variant: never pick them again
        from tangle_dot_eval import load_dots   # here: tangle_dot_eval imports this module
        done = {d['k'] for p_ in a.skip_dotted.split(',') if p_ for d in load_dots(ROOT/p_) if d['iv'] == iv}
        leg = [k for k in leg if k not in done]
        taken = set()
        for spec in a.pair:
            first, second, picker, n = spec.split(':')
            rnd = set()
            n = int(n)
            ok = [k for k in leg if masks(first, base, L, k, prim, cache) is not None and masks(second, base, L, k, prim, cache) is not None
                  and all(m is not None and m.any() for m in masks(first, base, L, k, prim, cache) + masks(second, base, L, k, prim, cache))]
            ok = [k for k in ok if k not in taken]
            if picker == 'unsure':
                # frames where the ownership cues disagree most (unsure_scores.py), skipping frames already dotted:
                # the reviewer's next dots go where they are worth the most
                sc = json.loads(open(a.scores).read()).get(iv, {})
                ch = [int(k) for k, _ in sorted(sc.items(), key=lambda x: -x[1]) if int(k) in ok][:n]
            elif picker == 'near':
                # frames 1-4 label frames (~0.4-1.6 s) from a reviewer-corrected frame of the second variant, never
                # a corrected frame itself: does a correction carry to its neighbours? Most-differing first.
                masks(second, base, L, 0, prim, cache)   # loads the variant's tangle.json into the cache
                cor = [int(c) for c in cache[second].get('corrections', [])]
                near = [k for k in range(len(L['frames'])) if cor and 1 <= min(abs(k - c) for c in cor) <= 4 and k not in taken and k not in done]
                near = [k for k in near if masks(first, base, L, k, prim, cache) is not None and masks(second, base, L, k, prim, cache) is not None]
                d = sorted((min(iou(m1, m2) for m1, m2 in zip(masks(first, base, L, k, prim, cache), masks(second, base, L, k, prim, cache))), k)
                           for k in near)
                ch = [k for x, k in d if x < 0.97][:n]
            elif picker == 'leg':
                ch = rng.sample(ok, min(n, len(ok)))
            else:
                d = []
                for k in ok:
                    m1, m2 = masks(first, base, L, k, prim, cache), masks(second, base, L, k, prim, cache)
                    d.append((min(iou(m1[0], m2[0]), iou(m1[1], m2[1])), k))
                d.sort()
                ch = [k for s, k in d if s < 0.97][:n]
                rest = [k for k in ok if k not in ch]
                if rest:
                    ch.append(rng.choice(rest))
                    rnd = {ch[-1]}
            taken |= set(ch)
            # which picker chose the frame ('diff' appends one random frame): lets verdicts be split by picker
            why = {k: 'random' if k in rnd or picker == 'leg' else picker for k in ch}
            picks += [(iv, k, first, second, why[k]) for k in ch]
    rng.shuffle(picks)
    items, key = [], {}
    for n, (iv, k, first, second, why) in enumerate(picks, 1):
        base = ROOT/'autolabel/v1'/iv
        L = json.loads((base/'labels.json').read_text())
        prim = json.loads((base/'tangle/v4/tangle.json').read_text())['prim']
        cache = {}
        m1, m2 = masks(first, base, L, k, prim, cache), masks(second, base, L, k, prim, cache)
        img = cv2.imread(L['frames'][k]['path'])
        box = crop_of(m1 + m2, img.shape)
        swap = rng.random() < 0.5
        sides = {'A': m2 if swap else m1, 'B': m1 if swap else m2}
        files = {}
        for tag, im in (('raw', img), ('A', overlay(img, sides['A'])), ('B', overlay(img, sides['B']))):
            files[tag] = f'assets/{n:03d}_{tag}.jpg'
            cv2.imwrite(str(OUT/files[tag]), im[box[1]:box[3], box[0]:box[2]], [cv2.IMWRITE_JPEG_QUALITY, 88])
        # box: the crop in full-frame pixels, so review dots (stored as crop fractions) map back onto the frame
        items.append({'id': n, 'raw': files['raw'], 'A': files['A'], 'B': files['B'], 'red': prim[0], 'blue': prim[1],
                      'box': [int(x) for x in box]})
        key[n] = {'src': f'{iv}#{k}', 'question': f'{first}_vs_{second}', 'A': second if swap else first, 'B': first if swap else second,
                  'picker': why}
    (OUT/'items.json').write_text(json.dumps(items))
    (OUT/'key.json').write_text(json.dumps(key, indent=1))
    (OUT/'videos.json').write_text(json.dumps(videos))
    (OUT/'videos.html').write_text(VIDEO_PAGE)
    from collections import Counter
    print(len(items), 'items', dict(Counter(v['question'] for v in key.values())))


if __name__ == '__main__':
    main()
