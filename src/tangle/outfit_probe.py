"""Outfit cue probe (.venv-pose, CPU). Per clip, learn each fighter's look (colour + texture at 3 scales) from pixels
the tracker owns far from the other fighter (pointvote.far_parts, eroded) and from correction masks, then ask at
each reviewer dot: does a confident appearance vote fix or break the base variant's answer there?
Frames are split into 4 contiguous blocks; a dot is scored by the model trained without its own block.

usage: outfit_probe.py --train v4w2m --corrections corr_legs_near.json --base v4w2mk,v4w2mkm --sets final,wide
"""
import argparse, json, os, sys
from collections import defaultdict
import cv2, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from autolabel_common import ROOT, rle_decode
from paths import SCRATCH
from tangle_dot_eval import load_dots
from tangle_review3 import masks
from pointvote import far_parts

torch.set_num_threads(4)
S = str(SCRATCH/'tangle')
SC = 0.5          # feature resolution
NPF = 1500        # training pixels per fighter per frame
FOLDS = 4


def feats(img):
    """BGR full-res image -> [h, w, 33] float32 at SC."""
    im = cv2.resize(img, None, fx=SC, fy=SC, interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0]
    gx, gy = cv2.Sobel(L, cv2.CV_32F, 1, 0), cv2.Sobel(L, cv2.CV_32F, 0, 1)
    g = [np.abs(gx), np.abs(gy), np.abs(gx + gy)/1.414, np.abs(gx - gy)/1.414]
    out = [lab]
    for s in (5, 11, 21):
        mu = cv2.blur(lab, (s, s))
        sd = np.sqrt(np.maximum(cv2.blur(lab*lab, (s, s)) - mu*mu, 0))
        out += [mu, sd] + [cv2.blur(x, (s, s))[:, :, None] for x in g]
    return np.concatenate(out, 2)


def train(X, y):
    torch.manual_seed(0)
    X, y = torch.from_numpy(X), torch.from_numpy(y)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 64), torch.nn.ReLU(), torch.nn.Linear(64, 64), torch.nn.ReLU(),
                              torch.nn.Linear(64, 2))
    opt = torch.optim.Adam(net.parameters(), 3e-3)
    g = torch.Generator().manual_seed(0)
    for _ in range(4):
        for idx in torch.randperm(len(X), generator=g).split(4096):
            loss = torch.nn.functional.cross_entropy(net(X[idx]), y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    return net.eval()


def prob(net, X):
    with torch.no_grad():
        return torch.softmax(net(torch.from_numpy(X)), 1)[:, 0].numpy()   # p(red)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', default='v4w2m')
    ap.add_argument('--corrections', default='corr_legs_near.json')
    ap.add_argument('--base', default='v4w2mk,v4w2mkm')
    ap.add_argument('--sets', default='final,wide')
    ap.add_argument('--out', default='')
    ap.add_argument('--only-corr', action='store_true', help='train on correction masks only (one model, no folds); '
                    'dots on correction frames are skipped')
    a = ap.parse_args()
    corr = json.load(open(f'{S}/{a.corrections}'))
    near = {(d['iv'], d['k']) for d in load_dots(ROOT/'reviews/tangle-ab-near')}
    split = json.load(open(f'{S}/split_final.json'))
    Bf = {(iv, k) for iv, ks in split['B'].items() for k in ks}
    pool = {'final': [d for d in load_dots(ROOT/'reviews/tangle-ab-final')],
            'finalB': [d for d in load_dots(ROOT/'reviews/tangle-ab-final') if (d['iv'], d['k']) in Bf],
            'wide': [d for d in load_dots(ROOT/'reviews/tangle-ab-wide') if d['c'] != 'none' and (d['iv'], d['k']) not in near]}
    dots = []
    for s in a.sets.split(','):
        for d in pool[s]:
            dots.append({**d, 'set': s})
    bases = a.base.split(',')
    rows, mcache = [], defaultdict(dict)
    for iv in sorted({d['iv'] for d in dots}):
        base = ROOT/'autolabel/v1'/iv
        T = json.loads((base/'tangle'/a.train/'tangle.json').read_text())
        L = json.loads((base/'labels.json').read_text())
        prim = T['prim']
        ks = sorted(int(k) for k in T['label_masks'])
        block = {k: i*FOLDS//len(ks) for i, k in enumerate(ks)}
        cks = {int(k) for k in corr.get(iv, {})}
        if a.only_corr:
            if not cks:
                print(iv, 'no corrections: skipped')
                continue
            # one model from the correction frames; it scores every non-correction frame (fold 0 trains on blocks != 0,
            # so give corrections block 1 and dot frames block 0)
            block = {k: (1 if k in cks else 0) for k in ks}
        rng = np.random.default_rng(0)
        Xs, ys, bs = [], [], []
        dk = defaultdict(list)
        for d in dots:
            if d['iv'] == iv:
                dk[d['k']].append(d)
        F = {}
        for k in ks:
            img = cv2.imread(L['frames'][k]['path'])
            f = feats(img)
            if k in dk:
                F[k] = f
            if a.only_corr and k not in cks:
                continue
            if str(k) in corr.get(iv, {}):
                tm = [rle_decode(corr[iv][str(k)][p]).astype(bool) for p in prim]
            else:
                tm = far_parts([rle_decode(T['label_masks'][str(k)][p]).astype(bool) for p in prim])
            h, w = f.shape[:2]
            for i in range(2):
                m = cv2.resize(cv2.erode(tm[i].astype(np.uint8), np.ones((7, 7), np.uint8)), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                yy, xx = np.nonzero(m)
                if len(yy) == 0:
                    continue
                sel = rng.choice(len(yy), min(NPF, len(yy)), replace=False)
                Xs.append(f[yy[sel], xx[sel]]); ys.append(np.full(len(sel), i)); bs.append(np.full(len(sel), block[k]))
        X, y, b = np.concatenate(Xs), np.concatenate(ys), np.concatenate(bs)
        mu, sd = X.mean(0), X.std(0) + 1e-6
        X = ((X - mu)/sd).astype(np.float32)
        sep = []
        for fo in (range(1) if a.only_corr else range(FOLDS)):
            tr = b != fo
            net = train(X[tr], y[tr].astype(np.int64))
            if (~tr).any():
                p = prob(net, X[~tr])
                sep.append(((p > 0.5) == (y[~tr] == 0)).mean())
            for k, ds in dk.items():
                if block.get(k) != fo:
                    continue
                f = F[k]
                for d in ds:
                    yy, xx = min(int(d['y']*SC), f.shape[0] - 1), min(int(d['x']*SC), f.shape[1] - 1)
                    pr = float(prob(net, ((f[yy, xx] - mu)/sd)[None].astype(np.float32))[0])
                    r = {'iv': iv, 'k': k, 'x': d['x'], 'y': d['y'], 'c': d['c'], 'set': d['set'], 'p_red': pr}
                    if (k, 'dt') not in mcache[iv]:
                        m0 = masks(bases[0], base, L, k, prim, mcache[(bases[0], iv)])
                        mcache[iv][(k, 'dt')] = [cv2.distanceTransform((~x).astype(np.uint8), cv2.DIST_L2, 5) if x is not None
                                                 else np.full(m0[0].shape, 1e4, np.float32) for x in m0]
                    r['d_red'], r['d_blue'] = (float(x[d['y'], d['x']]) for x in mcache[iv][(k, 'dt')])
                    r['fold'] = fo
                    for bv in bases:
                        m = masks(bv, base, L, k, prim, mcache[(bv, iv)])
                        rr, bb = (bool(x[d['y'], d['x']]) if x is not None else False for x in m)
                        r[bv] = 'red' if rr else 'blue' if bb else 'none'
                    rows.append(r)
        for r in rows:
            if r['iv'] == iv:
                r['sep'] = float(np.mean(sep))
        print(f'{iv:20s} separability (held-out trusted px) {np.mean(sep):.1%}  per fold {[round(s, 3) for s in sep]}', flush=True)
        for k in dk:
            if k not in block:
                print('  dot frame', k, 'not a label frame of', a.train)
    if a.out:
        json.dump(rows, open(a.out, 'w'))
    # report
    for s in a.sets.split(','):
        R = [r for r in rows if r['set'] == s]
        print(f'=== {s}: {len(R)} dots')
        acc = np.mean([(r['p_red'] > 0.5) == (r['c'] == 'red') for r in R])
        print(f'  appearance alone (argmax) {acc:.0%}')
        for t in (0.8, 0.9, 0.95, 0.98):
            conf = [r for r in R if max(r['p_red'], 1 - r['p_red']) >= t]
            ok = sum((r['p_red'] > 0.5) == (r['c'] == 'red') for r in conf)
            line = f'  conf>={t}: {len(conf):3d} dots, {ok/max(len(conf), 1):.0%} right |'
            for bv in bases:
                fix = brk = 0
                for r in conf:
                    app = 'red' if r['p_red'] > 0.5 else 'blue'
                    if app != r[bv]:
                        fix += app == r['c']; brk += r[bv] == r['c']
                base_ok = sum(r[bv] == r['c'] for r in R)
                line += f'  {bv} {base_ok} -> {base_ok + fix - brk} (+{fix} -{brk})'
            print(line)


if __name__ == '__main__':
    main()
