"""Hole fill by limb connectivity (CPU). Experimental.

Reviewer: in leg tangles a trapped foot usually sits between the other athlete's limbs, so "nearest mask" or "inside
whose outline" (BMP fill, cv2 watershed) gives it to the wrong person about half the time. A foot belongs to whoever's
ankle -> knee -> hip chain leads to it. Here every pixel the tracker leaves unowned, and that BMP calls body (union of
the two BMP masks), goes to the fighter whose bones it lies on: per-fighter capsules along the COCO skeleton (PMPose
keypoints from bmp_tangle.py; feet and hands as the shin/forearm extended past ankle/wrist), distance normalised by
each capsule's radius, nearest bone wins. Pixels on nobody's bones stay unowned (--fallback 1: BMP's owner instead).

  ../.venv-pose/bin/python bonefill.py --src v4w --bmp bmpw --out v4wk <trial/interval ...>
"""
import argparse
import json

import cv2
import numpy as np

from autolabel_common import ROOT, rle_decode, rle_encode, write_json

THR = 0.3
# (joint a, joint b, radius as a share of torso length); feet/hands: extension past b along a->b
BONES = [(11, 13, .16), (13, 15, .13), (12, 14, .16), (14, 16, .13), (5, 7, .11), (7, 9, .10), (6, 8, .11), (8, 10, .10),
         (5, 11, .22), (6, 12, .22), (5, 6, .18), (11, 12, .2), (0, 5, .14), (0, 6, .14)]
EXT = [(13, 15, .45, .12), (14, 16, .45, .12), (7, 9, .35, .09), (8, 10, .35, .09)]   # foot, hand
FOOT_R = .10   # capsule radius of the RTMW foot bones (--feet)


def seg_dist(P, a, b):
    ab = b - a
    t = np.clip(((P - a) @ ab)/max(ab @ ab, 1e-6), 0, 1)
    return np.linalg.norm(P - (a + t[:, None]*ab), axis=1)


LEGS = {(11, 13), (13, 15), (12, 14), (14, 16)}   # thighs and shins (+ the foot extensions past the ankles)


def bone_score(kp, P, feet=None, only=None, thr=THR):
    """Normalised distance (1 = capsule edge) from pixels P [N,2] to the nearest bone of one fighter.
    feet: {ankle index: [[x, y, s] big toe, small toe, heel]} (rtmw_feet.py) replaces the shin-extension guess."""
    sh, hp = (kp[5, :2] + kp[6, :2])/2, (kp[11, :2] + kp[12, :2])/2
    torso = np.linalg.norm(sh - hp) + 1.0
    best = np.full(len(P), np.inf)
    segs = [(kp[a, :2], kp[b, :2], r) for a, b, r in BONES if kp[a, 2] >= thr and kp[b, 2] >= thr and (only is None or (a, b) in only)]
    feet = {int(k): np.asarray(v, np.float32)[:, :2] for k, v in (feet or {}).items()}
    segs += [(kp[b, :2], kp[b, :2] + e*(kp[b, :2] - kp[a, :2]), r) for a, b, e, r in EXT
             if kp[a, 2] >= thr and kp[b, 2] >= thr and b not in feet and (only is None or (a, b) in only)]
    for ank, (big, small, heel) in feet.items():
        segs += [(kp[ank, :2], big, FOOT_R), (kp[ank, :2], small, FOOT_R), (kp[ank, :2], heel, FOOT_R), (big, small, FOOT_R)]
    for a, b, r in segs:
        best = np.minimum(best, seg_dist(P, a, b)/(r*torso))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='v4w')
    ap.add_argument('--bmp', default='bmpw')
    ap.add_argument('--out', default='v4wk')
    ap.add_argument('--fallback', type=int, default=0)
    ap.add_argument('--near', type=int, default=0, help='only fill within this many px of the pair (0 = anywhere; a BMP '
                    'mask can reach a spectator, a2s474/02 #19)')
    ap.add_argument('--feet', type=int, default=0, help='1 = RTMW heel/toe bones from tangle/rtmw/feet.json (rtmw_feet.py)')
    ap.add_argument('--margin', type=float, default=0, help='fill a hole only when the other fighter\'s bones are at least '
                    'this much further (capsule radii): an ambiguous hole stays unowned (the reviewer: a wrong fill looks '
                    'worse than a hole, final pack #2/#28/#33)')
    ap.add_argument('--override', default='', help='IN,OUT: a pixel the tracker gave fighter i moves to fighter j when it '
                    'lies within IN of one of j\'s confident leg bones (keypoints >= 0.5) and beyond OUT of all of i\'s '
                    'bones: legs traced through occlusion (final pack #23, closed guard)')
    ap.add_argument('targets', nargs='+')
    a = ap.parse_args()
    for iv in a.targets:
        base = ROOT/'autolabel/v1'/iv
        T = json.loads((base/'tangle'/a.src/'tangle.json').read_text())
        B = json.loads((base/'tangle'/a.bmp/'tangle.json').read_text())
        prim, out, n_bone, n_fb, n_ov = T['prim'], {}, 0, 0, 0
        FT = json.loads((base/'tangle/rtmw/feet.json').read_text()) if a.feet else {}
        for k, lm in T['label_masks'].items():
            m = [rle_decode(lm[p]).astype(bool) for p in prim]
            if k in B['label_masks'] and k in B['frames']:
                f = [rle_decode(B['label_masks'][k][p]).astype(bool) for p in prim]
                kp = np.asarray(B['frames'][k]['keypoints'], np.float32)
                hole = (f[0] | f[1]) & ~(m[0] | m[1])
                if a.near:
                    hole &= cv2.dilate((m[0] | m[1]).astype(np.uint8), np.ones((2*a.near + 1,)*2, np.uint8)).astype(bool)
                ys, xs = np.nonzero(hole)
                if len(ys):
                    P = np.stack([xs, ys], 1).astype(np.float32)
                    ft = FT.get(k, [{}, {}])
                    s = np.stack([bone_score(kp[i], P, ft[i]) for i in range(2)])
                    own = np.where((s.min(0) <= 1.0) & (s.max(0) - s.min(0) >= a.margin), s.argmin(0), -1)
                    n_bone += int((own >= 0).sum())
                    if a.fallback:
                        fb = np.where(f[0][ys, xs] & ~f[1][ys, xs], 0, np.where(f[1][ys, xs], 1, -1))
                        n_fb += int(((own < 0) & (fb >= 0)).sum())
                        own = np.where(own >= 0, own, fb)
                    for i in range(2):
                        m[i][ys[own == i], xs[own == i]] = True
                if a.override:
                    t_in, t_out = (float(x) for x in a.override.split(','))
                    moved = []
                    for i in range(2):
                        j = 1 - i
                        ys, xs = np.nonzero(m[i])
                        if not len(ys):
                            moved.append(None)
                            continue
                        P = np.stack([xs, ys], 1).astype(np.float32)
                        leg_j = bone_score(kp[j], P, only=LEGS, thr=0.5)
                        sel = leg_j <= t_in
                        if sel.any():
                            sel[sel] = bone_score(kp[i], P[sel]) > t_out
                        moved.append((ys[sel], xs[sel]))
                    for i in range(2):
                        if moved[i] is not None and len(moved[i][0]):
                            m[i][moved[i]] = False
                            m[1 - i][moved[i]] = True
                            n_ov += len(moved[i][0])
            out[k] = {p: rle_encode(m[i]) for i, p in enumerate(prim)}
        d = base/'tangle'/a.out
        d.mkdir(parents=True, exist_ok=True)
        write_json(d/'tangle.json', {**{x: y for x, y in T.items() if x != 'label_masks'}, 'variant': a.out, 'src': a.src,
                                     'bmp': a.bmp, 'fallback': a.fallback, 'feet': a.feet, 'margin': a.margin,
                                     'override': a.override, 'label_masks': out})
        print(iv, 'hole px by bones', n_bone, 'by BMP fallback', n_fb, 'overridden', n_ov, flush=True)


if __name__ == '__main__':
    main()
