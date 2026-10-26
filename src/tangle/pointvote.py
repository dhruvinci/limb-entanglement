"""Ownership by motion: CoTracker3 point votes for the tangle tracker's holes (runs in .venv-pose). Experimental.

The cue nothing else uses: where a pixel came from. A trapped foot looks like the partner's skin and sits inside the
partner's outline, but it moves with its own shin. Points are seeded on each athlete where ownership is certain and
followed through time with CoTracker3 (Karaev et al. 2024, offline model); at every label frame, the visible tracked
points near a pixel vote for their athlete, weighted by how close in time their seed is.

Trusted seed frames (no reviewer dots from the test packs):
  - correction frames (--corrections, the reviewer's leg-pack dots turned into masks: what v4w was built from)
  - every other label frame, but only the parts of each athlete at least LOCAL px from the other athlete's mask
    (in leg locks the pair is never fully apart: the first version seeded only whole separated frames and found
    almost none; a foot is unambiguous while it is away from the partner, and that is what the vote should carry)
Points: SEED_N per athlete per seed frame, inside the eroded mask. Each seed is tracked over +-WIN span frames.

Outputs per interval, tangle/cot/:
  cot_votes.json  {label_index: {"x,y": vote}} at the reviewer's dots of --packs (vote in [-1, 1], + = first athlete)
  and variant <out> (tangle/<out>/tangle.json): v4w + holes next to the pair filled by confident votes

  ../.venv-pose/bin/python pointvote.py --src v4w --corrections dotfix_l10.json --out v4wc <trial/interval ...>
CoTracker3 joint tracking was tried before on Harmony4D (cot_h4d.py: 1/3 of joints stayed visible through tangles,
CPU out of memory); here it runs on the GPU and only has to vote inside holes.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

from autolabel_common import ROOT, rle_decode, rle_encode, write_json
from tangle_dot_eval import load_dots

WORK = Path('/tmp/tangle-work')
SCALE = 0.5        # frames are tracked at half resolution (640x360)
SEED_N = 120       # points per athlete per seed frame
WIN = 25           # span frames tracked either side of a seed (~2.5 s; 50 ran out of memory on the 6 GB card)
LOCAL = 30         # px from the other athlete for a pixel to be a trusted seed
RADIUS = 9         # px (full resolution): points this close to a pixel vote on it
MIN_W = 1.5        # total vote weight needed to fill a hole
MIN_MARGIN = 0.4   # |vote| needed to fill a hole


def sample_points(m, n, rng):
    e = cv2.erode(m.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    ys, xs = np.nonzero(e if e.sum() > 50 else m)
    if not len(ys):
        return np.zeros((0, 2), np.float32)
    idx = rng.choice(len(ys), size=min(n, len(ys)), replace=False)
    return np.stack([xs[idx], ys[idx]], 1).astype(np.float32)


def far_parts(m):
    """each athlete's pixels at least LOCAL px from the other athlete's mask."""
    out = []
    for i in range(2):
        other = m[1 - i]
        if not other.any():
            out.append(m[i].copy())
            continue
        d = cv2.distanceTransform((~other).astype(np.uint8), cv2.DIST_L2, 3)
        out.append(m[i] & (d >= LOCAL))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='v4w')
    ap.add_argument('--corrections', required=True)
    ap.add_argument('--out', default='v4wc')
    ap.add_argument('--packs', default='reviews/tangle-ab-near,reviews/tangle-ab-wide')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--smoke', type=int, default=0, help='tiny settings (a few points, short windows, 3 seeds) for a CPU test')
    ap.add_argument('targets', nargs='+')
    a = ap.parse_args()
    global SEED_N, WIN
    if a.smoke:
        SEED_N, WIN = 12, 3
    corr = json.loads(Path(a.corrections).read_text())
    dots = [d for p in a.packs.split(',') for d in load_dots(ROOT/p)]
    model = torch.hub.load('facebookresearch/co-tracker', 'cotracker3_offline').to(a.device).eval()
    rng = np.random.default_rng(0)
    for iv in a.targets:
        base = ROOT/'autolabel/v1'/iv
        T = json.loads((base/'tangle'/a.src/'tangle.json').read_text())
        prim, l2s = T['prim'], {int(k): v for k, v in T['label_to_span'].items()}
        s2l = {v: k for k, v in l2s.items()}
        paths = sorted((WORK/iv.replace('/', '_')/'frames').glob('*.jpg'))
        if not paths:
            print(iv, 'no span frames in', WORK, flush=True)
            continue
        LM = {int(k): [rle_decode(v[p]).astype(bool) for p in prim] for k, v in T['label_masks'].items()}
        seeds = {}
        for k, fr in corr.get(iv, {}).items():
            seeds[int(k)] = ('correction', [rle_decode(fr[p]).astype(bool) for p in prim])
        for k, m in LM.items():
            if k not in seeds:
                seeds[k] = ('local', far_parts(m))
        H, W = cv2.imread(str(paths[0])).shape[:2]
        h, w = int(H*SCALE), int(W*SCALE)
        frames = np.stack([cv2.resize(cv2.imread(str(p))[:, :, ::-1], (w, h), interpolation=cv2.INTER_AREA) for p in paths])
        # votes[label_k] = list of (x, y, athlete, weight) at full resolution
        votes = defaultdict(list)
        for k, (kind, m) in sorted(seeds.items())[:3 if a.smoke else None]:
            s = l2s[k]
            lo, hi = max(0, s - WIN), min(len(paths) - 1, s + WIN)
            q, who = [], []
            for i in range(2):
                pts = sample_points(m[i], SEED_N, rng)
                q += [(s - lo, x*SCALE, y*SCALE) for x, y in pts]
                who += [i]*len(pts)
            if not q:
                continue
            v = torch.from_numpy(frames[lo:hi + 1]).permute(0, 3, 1, 2)[None].float().to(a.device)
            qq = torch.tensor(q, dtype=torch.float32)[None].to(a.device)
            parts = 1
            while True:   # out of memory: track the queries in more, smaller groups
                try:
                    with torch.no_grad():
                        res = [model(v, queries=c, backward_tracking=True) for c in torch.tensor_split(qq, parts, dim=1)]
                    break
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    parts *= 2
                    if parts > 8:
                        raise
            tr = torch.cat([r[0] for r in res], 2)[0].cpu().numpy()/SCALE
            vis = torch.cat([r[1] for r in res], 2)[0].cpu().numpy()
            del v
            torch.cuda.empty_cache()
            who = np.array(who)
            for t in range(hi - lo + 1):
                kk = s2l.get(lo + t)
                if kk is None or kk == k:
                    continue
                wt = 1/(1 + abs(lo + t - s)/10)
                ok = vis[t] > 0.5
                for (x, y), i in zip(tr[t][ok], who[ok]):
                    votes[kk].append((x, y, i, wt))
        n_seed = {kind: sum(1 for x in seeds.values() if x[0] == kind) for kind in ('correction', 'local')}

        def vote_at(kk, xs, ys):
            """per query pixel: (vote in [-1, 1] for athlete 0 vs 1, total weight)."""
            V = votes.get(kk)
            if not V:
                return np.zeros(len(xs)), np.zeros(len(xs))
            arr = np.array(V, np.float32)
            from scipy.spatial import cKDTree
            tree = cKDTree(arr[:, :2])
            vote, tot = np.zeros(len(xs)), np.zeros(len(xs))
            for j, nb in enumerate(tree.query_ball_point(np.stack([xs, ys], 1), RADIUS)):
                if nb:
                    ww = arr[nb, 3]
                    w0 = ww[arr[nb, 2] == 0].sum(); w1 = ww[arr[nb, 2] == 1].sum()
                    vote[j], tot[j] = (w0 - w1)/(w0 + w1), w0 + w1
            return vote, tot
        # votes at the reviewer's dots (for clue_judge.py) and a hole-filled variant
        dv = defaultdict(dict)
        for d in dots:
            if d['iv'] == iv and d['k'] in l2s:
                vt, tot = vote_at(d['k'], np.array([d['x']]), np.array([d['y']]))
                dv[str(d['k'])][f"{d['x']},{d['y']}"] = float(vt[0]) if tot[0] >= MIN_W else 0.0
        (base/'tangle/cot').mkdir(parents=True, exist_ok=True)
        (base/'tangle/cot/cot_votes.json').write_text(json.dumps(dv))
        out, filled = {}, 0
        for k, m in LM.items():
            near = cv2.dilate((m[0] | m[1]).astype(np.uint8), np.ones((81, 81), np.uint8)).astype(bool)
            ys, xs = np.nonzero(near & ~(m[0] | m[1]))
            if len(ys) and k in votes:
                sub = slice(None, None, 2)   # every 2nd hole pixel, then dilated back: votes are smooth at 9 px
                vt, tot = vote_at(k, xs[sub].astype(np.float32), ys[sub].astype(np.float32))
                conf = (tot >= MIN_W) & (np.abs(vt) >= MIN_MARGIN)
                for i, sign in ((0, 1), (1, -1)):
                    g = np.zeros_like(m[i], np.uint8)
                    sel = conf & (np.sign(vt) == sign)
                    g[ys[sub][sel], xs[sub][sel]] = 1
                    g = cv2.dilate(g, np.ones((3, 3), np.uint8)).astype(bool) & ~(m[0] | m[1])
                    m[i] = m[i] | g
                    filled += int(g.sum())
                m[1] &= ~m[0]
            out[str(k)] = {p: rle_encode(m[i]) for i, p in enumerate(prim)}
        d = base/'tangle'/a.out
        d.mkdir(parents=True, exist_ok=True)
        write_json(d/'tangle.json', {**{x: y for x, y in T.items() if x != 'label_masks'}, 'variant': a.out, 'src': a.src,
                                     'seeds': {str(k): v[0] for k, v in seeds.items()}, 'label_masks': out})
        print(iv, 'seeds', n_seed, 'dot votes', sum(len(v) for v in dv.values()), 'filled px', filled, flush=True)


if __name__ == '__main__':
    main()
