"""Reviewer dots -> corrected masks on the dotted frames (runs in .venv-bmp). Experimental.

Why: on the leg pack, the tracker's errors are already in its seeds (dotted frames that are anchors: 66% of dots
right, the same as the production masks), so a correction on one frame should be propagated by the tracker as a new
anchor. This is the correction step: SAM (SAM-pose2seg's SAM 2.1 b+ image predictor, the only interactive SAM 2 we
have locally) refines each fighter's current mask (mask input, as confident logits) with the reviewer's dots as
clicks: own dots positive, the partner's and 'neither' dots negative. Run on the pair's crop for resolution. Pixels
both fighters claim go to the higher logit; a fighter without dots keeps the current mask where unclaimed.

  ../.venv-bmp/bin/python dotfix.py --pack reviews/tangle-ab-legs --src v4 --out dotfix.json [--split] [--local 40 --body bmpw] [--enforce]
--split: score the correction step itself, using half of each frame's dots as clicks and the other half as test
(both ways), against the source masks on the same test dots.
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from autolabel_common import ROOT, rle_encode
from tangle_dot_eval import load_dots
from tangle_review3 import masks as variant_masks

from paths import BMP  # noqa: E402
CKPT = Path.home()/'.cache/torch/hub/checkpoints'
LOGIT = 10.0
LOCAL = 0          # px; >0: a pixel may change owner only this close to a dot of its new owner (see localize)


def load_sam():
    sys.path.insert(0, str(BMP))
    os.chdir(BMP)
    from hydra import initialize_config_module
    from hydra.core.global_hydra import GlobalHydra
    if not GlobalHydra.instance().is_initialized():
        initialize_config_module('bboxmaskpose.sam2', version_base='1.2')
    from bboxmaskpose.sam2_utils import prepare_model
    return prepare_model(str(BMP/'bboxmaskpose/sam2/configs/sam-pose2seg/sam-pose2seg_hiera_b+.yaml'),
                         str(CKPT/'SAM-pose2seg_hiera_b%2B.pt'))


def crop_box(ms, shape, pad=0.15):
    ys, xs = np.nonzero(ms[0] | ms[1])
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
    p = pad*max(x2 - x1, y2 - y1)
    return int(max(0, x1 - p)), int(max(0, y1 - p)), int(min(shape[1], x2 + p)), int(min(shape[0], y2 + p))


def correct(sam, img, ms, dots, body=None):
    """ms: [red, blue] bool masks; dots: [{'x','y','c'}] in frame pixels -> corrected [red, blue]."""
    x1, y1, x2, y2 = crop_box(ms, img.shape)
    sam.set_image(np.ascontiguousarray(img[y1:y2, x1:x2, ::-1]))
    logits = []
    for i, col in enumerate(('red', 'blue')):
        own = [d for d in dots if d['c'] == col]
        if not own:
            logits.append(None)
            continue
        neg = [d for d in dots if d['c'] != col]
        pts = np.array([[d['x'] - x1, d['y'] - y1] for d in own + neg], np.float32)
        lab = np.array([1]*len(own) + [0]*len(neg))
        lo = cv2.resize(ms[i][y1:y2, x1:x2].astype(np.float32), (256, 256), interpolation=cv2.INTER_AREA)
        out, _, _ = sam.predict(point_coords=pts, point_labels=lab, mask_input=((lo*2 - 1)*LOGIT)[None] if LOGIT else None,
                                multimask_output=False, return_logits=True)
        logits.append(out[0])
    new = [m.copy() for m in ms]
    crop = [m[y1:y2, x1:x2] for m in new]
    have = [l is not None for l in logits]
    if all(have):
        a, b = logits
        crop[0][:] = (a > 0) & (a >= b)
        crop[1][:] = (b > 0) & (b > a)
    elif any(have):
        i = have.index(True)
        crop[i][:] = logits[i] > 0
        crop[1 - i][:] &= ~crop[i]
    return localize(ms, new, dots, body) if LOCAL else new


def localize(ms, new, dots, body=None):
    """Undo ownership changes far from the dots that justify them. A reviewer dots the spots they see wrong (or traces
    a limb), so a change is kept only within LOCAL px of a dot of the pixel's new owner ('none' and the other colour's
    dots justify a pixel becoming unowned). Without this, SAM's click-refined mask spreads well past the dots: on
    a2s474/02 #12 red dots tracing the bottom fighter's legs handed the top fighter's whole back to red, and on
    a2s270 #7 red reached spectators in red shirts. body (bool mask, e.g. the BMP pair's union): pixels nobody owned may
    be filled anywhere inside it, so a dotted leg the tracker missed is filled whole, not as discs around the dots."""
    H, W = ms[0].shape
    lab_src = np.where(ms[0], 0, np.where(ms[1], 1, 2))
    lab_new = np.where(new[0], 0, np.where(new[1], 1, 2))
    near = []
    for cols in (('red',), ('blue',), ('red', 'blue', 'none')):
        dm = np.ones((H, W), np.uint8)
        for d in dots:
            if d['c'] in cols:
                dm[d['y'], d['x']] = 0
        near.append(cv2.distanceTransform(dm, cv2.DIST_L2, 5) <= LOCAL if (dm == 0).any() else np.zeros((H, W), bool))
    keep = np.choose(lab_new, near)
    if body is not None:
        keep |= body & (lab_src == 2) & (lab_new != 2)
    lab = np.where((lab_new != lab_src) & ~keep, lab_src, lab_new)
    return [lab == 0, lab == 1]


def enforce(sam, img, ms, dots, radius=20, disc=6, body=None, fill=120):
    """Make every red/blue dot hold. SAM's click refinement often stops short of a dotted part (a trapped foot, a
    hand): 58 of 600 dots on the legs/near/final correction frames ended up a hole or the other fighter's in the
    correction made from them, and the tracker carries the wrong anchor forward (trial018/02 #24 -> #21-#27). Each such
    dot gets a part mask from one click on it, every other-colour dot negative: the smallest of SAM's three masks
    that holds no negative dot (the best-scored one is often the whole limb, which the clip cut into a disc on
    a3m007 #13/#29), kept connected to the dot; else a `disc` px disc on the dot's side of its nearest negative dot.
    The part takes the other fighter's pixels only within `radius` px of the dot (at 40 it took a half-disc of the top
    fighter's shirt on a2s270 #29), or 2*radius when it holds another dot of the same colour (a traced part, like the
    heel and toes of the trial018/02 #24 foot); unowned pixels it fills inside body
    (as in localize) or within `fill` px (a leg clipped at `radius` is a disc, and BMP missed the dotted legs of
    a2s054/02 #6/#22 too).
    Returns new [red, blue] and the number of dots it had to fix."""
    H, W = ms[0].shape
    lab = np.where(ms[0], 0, np.where(ms[1], 1, 2))
    want = {'red': 0, 'blue': 1}
    bad = [d for d in dots if d['c'] in want and lab[d['y'], d['x']] != want[d['c']]]
    if not bad:
        return ms, 0
    ys, xs = np.nonzero(ms[0] | ms[1])
    ys = np.r_[ys, [d['y'] for d in dots]]; xs = np.r_[xs, [d['x'] for d in dots]]
    p = 0.15*max(np.ptp(xs), np.ptp(ys))
    x1, y1 = int(max(0, xs.min() - p)), int(max(0, ys.min() - p))
    x2, y2 = int(min(W, xs.max() + p + 1)), int(min(H, ys.max() + p + 1))
    sam.set_image(np.ascontiguousarray(img[y1:y2, x1:x2, ::-1]))
    yy, xx = np.mgrid[0:H, 0:W]
    for d in bad:
        neg = [e for e in dots if e['c'] != d['c']]
        pts = np.array([[e['x'] - x1, e['y'] - y1] for e in [d] + neg], np.float32)
        out, _, _ = sam.predict(point_coords=pts, point_labels=np.array([1] + [0]*len(neg)), multimask_output=True)
        near = (yy - d['y'])**2 + (xx - d['x'])**2
        part = None
        for j in np.argsort((out > 0).reshape(len(out), -1).sum(1)):
            m = np.zeros((H, W), bool)
            m[y1:y2, x1:x2] = out[j] > 0
            if m[d['y'], d['x']] and not any(m[e['y'], e['x']] for e in neg):
                hole = (lab == 2) & ((near <= fill*fill) | (body if body is not None else False))
                r = 2*radius if any(m[e['y'], e['x']] for e in dots if e['c'] == d['c'] and e is not d) else radius
                part = m & ((near <= r*r) | hole)
                part = cv2.connectedComponents(part.astype(np.uint8))[1]
                part = part == part[d['y'], d['x']]
                break
        if part is None:
            part = near <= disc*disc
            for e in neg:
                part &= near < (yy - e['y'])**2 + (xx - e['x'])**2
        lab[part] = want[d['c']]
    return [lab == 0, lab == 1], len(bad)


def score(ms, dots):
    ok = 0
    for d in dots:
        r, b = ms[0][d['y'], d['x']], ms[1][d['y'], d['x']]
        ok += ('red' if r else 'blue' if b else 'none') == d['c']
    return ok


def main():
    global LOGIT, LOCAL
    ap = argparse.ArgumentParser()
    ap.add_argument('--pack', default='reviews/tangle-ab-legs')
    ap.add_argument('--src', default='v4')
    ap.add_argument('--out', default='')
    ap.add_argument('--split', action='store_true')
    ap.add_argument('--logit', type=float, default=10.0, help='mask prior strength; 0 = clicks only')
    ap.add_argument('--local', type=int, default=0, help='px: keep ownership changes only near a dot of the new owner')
    ap.add_argument('--body', default='', help='with --local: variant whose pair union may take hole fills anywhere (bmpw)')
    ap.add_argument('--enforce', action='store_true', help='then make every red/blue dot hold (see enforce)')
    a = ap.parse_args()
    LOGIT, LOCAL = a.logit, a.local
    dots = load_dots(ROOT/a.pack)
    frames = defaultdict(list)
    for d in dots:
        frames[(d['iv'], d['k'])].append(d)
    sam = load_sam()
    cache, L, P = defaultdict(dict), {}, {}
    res = defaultdict(lambda: [0, 0, 0])   # fold -> [src right, corrected right, n]
    out = defaultdict(dict)
    for (iv, k), ds in sorted(frames.items()):
        base = ROOT/'autolabel/v1'/iv
        if iv not in L:
            L[iv] = json.loads((base/'labels.json').read_text())
            P[iv] = json.loads((base/'tangle/v4/tangle.json').read_text())['prim']
        ms = variant_masks(a.src, base, L[iv], k, P[iv], cache[(a.src, iv)])
        img = cv2.imread(L[iv]['frames'][k]['path'])
        if a.split:
            for f in (0, 1):
                clicks = [d for n, d in enumerate(ds) if n % 2 == f]
                test = [d for n, d in enumerate(ds) if n % 2 != f]
                new = correct(sam, img, ms, clicks)
                r = res[f'{iv}#{k}']
                r[0] += score(ms, test); r[1] += score(new, test); r[2] += len(test)
            r = res[f'{iv}#{k}']
            print(f'{iv}#{k}: src {r[0]}/{r[2]} -> corrected {r[1]}/{r[2]}', flush=True)
        if a.out:
            body = None
            if a.body:
                bm = variant_masks(a.body, base, L[iv], k, P[iv], cache[(a.body, iv)])
                body = np.logical_or.reduce([x for x in bm if x is not None]) if any(x is not None for x in bm) else None
            new = correct(sam, img, ms, ds, body)
            nf = 0
            if a.enforce:
                new, nf = enforce(sam, img, new, ds, body=body)
            out[iv][str(k)] = {P[iv][i]: rle_encode(new[i]) for i in range(2)}
            print(f'{iv}#{k}: all dots as clicks; dots right {score(ms, ds)} -> {score(new, ds)} / {len(ds)}'
                  + (f' (enforced {nf})' if a.enforce else ''), flush=True)
    if a.split:
        t = np.array(list(res.values())).sum(0)
        print(f'SPLIT TOTAL src {t[0]}/{t[2]} ({t[0]/t[2]:.0%}) -> corrected {t[1]}/{t[2]} ({t[1]/t[2]:.0%})')
    if a.out:
        (ROOT/a.out).write_text(json.dumps(out))
        print('wrote', ROOT/a.out)


if __name__ == '__main__':
    main()
