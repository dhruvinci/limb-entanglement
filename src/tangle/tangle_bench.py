"""Tangle-tracking benchmark on Harmony4D grappling (ground-truth ownership rendered from SMPL fits, tangle_h4d.py).

Each (sequence, camera) is a video. Frames are cropped to the pair (fixed crop per video: union of both people's
boxes over the sequence, padded to 16:9) and resized to 1280x720, like our footage.
Seeds imitate our anchors: SAM 3.1 'person' detections on the seed frame, their foreground split between the two
people by the ground-truth ownership (nearest GT body; pixels > 30 px from both bodies dropped), i.e. a detector-
quality mask with the right owner. Seeds sit on the first frame (forward) and the last frame (backward).
Variants (tracking only; seeds identical):
  points_fwd      densify today: box corners + interior points, no pixel competition, forward only
  mask_fwd        full-mask prompts, no competition, forward
  mask_nonov_fwd  full-mask prompts, competition before memory encoding, forward
  mask_nonov_bidir  as above, forward + backward, fused by distance to each seed
Metrics on ground-truth body pixels (SMPL has no clothing, so background predictions on clothing are not penalised):
  correct = share of GT body pixels predicted as the right person; wrong_owner = share predicted as the other person;
  missed = predicted background. Reported separately for frames where the bodies touch (tangle) and do not.
  phases:  seeds (image model) -> track (video model) -> score (CPU)
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from tangle_h4d import H4D, Sequence, touching

ROOT = H4D/'03_grappling2'
WORK = Path('/tmp/tangle-bench')
CAMS = ('cam01', 'cam06', 'cam11', 'cam16')
STRIDE = 2   # Harmony4D exo video is 20 fps; tracked at 10 fps, close to our 12.5
VARIANTS = {'points_fwd': ('points', False, 'fwd'), 'mask_fwd': ('mask', False, 'fwd'),
            'mask_nonov_fwd': ('mask', True, 'fwd'), 'mask_nonov_bidir': ('mask', True, 'bidir'),
            'points_bidir': ('points', False, 'bidir'), 'mask_nonov_bidir_flip': ('mask', True, 'bidir_flip'),
            'mask_nonov_bidir_pres': ('mask', True, 'bidir_pres')}


def videos(cams=CAMS, min_frames=40):
    expected = json.loads((H4D/'03_grappling2_smpl_counts.json').read_text())  # frames per sequence in the archive
    for seq in sorted(ROOT.glob('*_grappling2')):
        if not (seq/'colmap/workplace/scale.npy').exists():
            continue
        s = Sequence(seq)
        if len(s.frames) < min_frames or len(s.frames) != expected.get(seq.name):
            continue
        for cam in cams:
            if cam in s.poses and all(s.image_path(cam, f).exists() for f in s.frames) and \
                    all((seq/'processed_data/bbox'/cam/f'{f:05d}.npy').exists() for f in s.frames):
                yield seq.name, cam


def crop_box(s, cam):
    """Fixed 16:9 crop covering both bodies over the whole sequence (in full-resolution pixels)."""
    W, H = s.cams[s.poses[cam][2]][:2]
    boxes = []
    for f in s.frames:
        b = np.load(s.root/'processed_data/bbox'/cam/f'{f:05d}.npy', allow_pickle=True).item()
        boxes += [np.array(v, float) for v in b.values()]
    b = np.array(boxes)
    x1, y1, x2, y2 = b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()
    cx, cy = (x1 + x2)/2, (y1 + y2)/2
    w = max((x2 - x1)*1.15, (y2 - y1)*1.15*16/9, 640)
    h = w*9/16
    x0, y0 = min(max(0, cx - w/2), W - w), min(max(0, cy - h/2), H - h)
    return [float(max(0, x0)), float(max(0, y0)), float(min(W, w)), float(min(H, h))]


def prepare(name, cam):
    """Crop frames + render GT label maps. Returns the video work dir."""
    wd = WORK/f'{name}_{cam}'
    if (wd/'gt.npz').exists():
        return wd
    s = Sequence(ROOT/name)
    x0, y0, w, h = crop_box(s, cam)
    (wd/'frames').mkdir(parents=True, exist_ok=True)
    gts = []
    for k, f in enumerate(s.frames):
        img = cv2.imread(str(s.image_path(cam, f)))
        crop = img[int(y0):int(y0 + h), int(x0):int(x0 + w)]
        cv2.imwrite(str(wd/'frames'/f'{k:05d}.jpg'), cv2.resize(crop, (1280, 720), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 92])
        gts.append(render_crop(s, cam, f, (x0, y0, w, h)))
    np.savez_compressed(wd/'gt.npz', gt=np.stack(gts))
    (wd/'meta.json').write_text(json.dumps({'seq': name, 'cam': cam, 'crop': [x0, y0, w, h], 'frames': s.frames}))
    return wd


def render_crop(s, cam, frame, crop, size=(1280, 720)):
    from tangle_h4d import FACES, PEOPLE
    x0, y0, w, h = crop
    d = np.load(s.root/'processed_data/smpl'/f'{frame:05d}.npy', allow_pickle=True).item()
    sx, sy = size[0]/w, size[1]/h
    tris = []
    for n, who in enumerate(PEOPLE, start=1):
        uv, z = s.project(d[who]['vertices'], cam)
        uv = (uv - [x0, y0])*[sx, sy]
        ok = (z[FACES] > 0).all(1)
        dep = z[FACES].mean(1)
        for tri, dp in zip(uv[FACES][ok], dep[ok]):
            tris.append((dp, n, tri))
    tris.sort(key=lambda x: -x[0])
    out = np.zeros((size[1], size[0]), np.uint8)
    for _, n, tri in tris:
        cv2.fillConvexPoly(out, np.round(tri*4).astype(np.int32), n, shift=2)
    return out


def seed_masks(processor, wd, k, gt):
    """Detector foreground near the pair, split by nearest GT body -> {1: mask, 2: mask}."""
    from PIL import Image
    import autolabel_detect as det
    from autolabel_common import rle_decode
    dets, _ = det.detect_frame(processor, Image.open(wd/'frames'/f'{k:05d}.jpg').convert('RGB'))
    body = gt > 0
    fg = np.zeros_like(body)
    for d in dets:
        m = rle_decode(d['mask_rle']).astype(bool)
        if (m & body).sum() > 0.2*m.sum():
            fg |= m
    out = {}
    dist = []
    for n in (1, 2):
        dist.append(cv2.distanceTransform((gt != n).astype(np.uint8), cv2.DIST_L2, 5))
    dist = np.stack(dist)
    owner = dist.argmin(0) + 1
    near = dist.min(0) <= 30
    for n in (1, 2):
        out[n] = (fg & near & (owner == n)) | (gt == n)
    return out


def score(wd, labelmaps, gt):
    rows = []
    for k in range(len(gt)):
        g, p = gt[k], labelmaps[k]
        r = {'k': k, 'touch': touching(g)}
        for n in (1, 2):
            body = g == n
            tot = max(1, body.sum())
            r[f'correct{n}'] = float((p[body] == n).sum()/tot)
            r[f'wrong{n}'] = float((p[body] == 3 - n).sum()/tot)
            r[f'area{n}'] = int(body.sum())
        rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('phase', choices=['prepare', 'seeds', 'track', 'score'])
    ap.add_argument('--variants', nargs='*', default=list(VARIANTS))
    ap.add_argument('--cams', nargs='*', default=list(CAMS))
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    vids = list(videos(a.cams))
    if a.limit:
        vids = vids[:a.limit]
    if a.phase == 'prepare':
        for name, cam in vids:
            prepare(name, cam)
            print('prepared', name, cam, flush=True)
    elif a.phase == 'seeds':
        import autolabel_detect as det
        model, proc = det.load()
        for name, cam in vids:
            wd = prepare(name, cam)
            if (wd/'seeds.npz').exists():
                continue
            gt = np.load(wd/'gt.npz')['gt']
            first, last = seed_masks(proc, wd, 0, gt[0]), seed_masks(proc, wd, len(gt) - 1, gt[-1])
            np.savez_compressed(wd/'seeds.npz', f1=first[1], f2=first[2], l1=last[1], l2=last[2])
            print('seeds', name, cam, flush=True)
    elif a.phase == 'track':
        import tangle_core
        for v in a.variants:
            prompt, nonov, mode = VARIANTS[v]
            tracker = tangle_core.load(non_overlap=nonov)
            for name, cam in vids:
                wd = WORK/f'{name}_{cam}'
                out = wd/f'pred_{v}.npz'
                if out.exists() or not (wd/'seeds.npz').exists():
                    continue
                sd = np.load(wd/'seeds.npz')
                frames = tangle_core.load_frames(wd/'frames', tracker, STRIDE)
                n = len(frames[0])
                pf, _, sf = tangle_core.track(tracker, frames, {0: {1: sd['f1'], 2: sd['f2']}}, start=0, reverse=False, prompt=prompt)
                pb, sb = None, {}
                if mode.startswith('bidir'):
                    pb, _, sb = tangle_core.track(tracker, frames, {n - 1: {1: sd['l1'], 2: sd['l2']}}, start=n - 1, reverse=True, prompt=prompt)
                if mode == 'bidir_flip':  # average with the mirrored-video passes
                    qf, _, _ = tangle_core.track(tracker, frames, {0: {1: sd['f1'], 2: sd['f2']}}, start=0, reverse=False, prompt=prompt, flip=True)
                    qb, _, _ = tangle_core.track(tracker, frames, {n - 1: {1: sd['l1'], 2: sd['l2']}}, start=n - 1, reverse=True, prompt=prompt, flip=True)
                    pf = {k: ((pf[k].astype(np.uint16) + qf[k])//2).astype(np.uint8) for k in pf}
                    pb = {k: ((pb[k].astype(np.uint16) + qb[k])//2).astype(np.uint8) for k in pb}
                labs = []
                for k in range(n):
                    a_, b_ = pf.get(k), (pb.get(k) if pb else None)
                    wf, wb = (n - 1 - k, k) if pb else (1, 0)
                    if mode == 'bidir_pres':  # times the tracker's presence belief per object
                        wf = wf/(1 + np.exp(-sf[k])) if k in sf else wf
                        wb = wb/(1 + np.exp(-sb[k])) if k in sb else wb
                    lab, _ = tangle_core.fuse(a_, b_, wf, wb)
                    lab = np.zeros((360, 640), np.uint8) if lab is None else lab
                    labs.append(cv2.resize(lab.astype(np.uint8), (1280, 720), interpolation=cv2.INTER_NEAREST))
                np.savez_compressed(out, labels=np.stack(labs))
                print('tracked', v, name, cam, n, flush=True)
    else:
        res = {}
        for v in a.variants:
            rows = []
            for name, cam in vids:
                wd = WORK/f'{name}_{cam}'
                if not (wd/f'pred_{v}.npz').exists():
                    continue
                gt = np.load(wd/'gt.npz')['gt']
                gt = np.concatenate([gt[::STRIDE], gt[-1:]]) if (len(gt) - 1) % STRIDE else gt[::STRIDE]
                rows += [dict(r, video=f'{name}/{cam}') for r in score(wd, np.load(wd/f'pred_{v}.npz')['labels'], gt)]
            if not rows:
                continue
            def agg(rs):
                c = np.mean([(r['correct1'] + r['correct2'])/2 for r in rs])
                w = np.mean([(r['wrong1'] + r['wrong2'])/2 for r in rs])
                sw = np.mean([max(r['wrong1'], r['wrong2']) > 0.3 for r in rs])
                return {'frames': len(rs), 'correct': round(float(c), 3), 'wrong_owner': round(float(w), 3),
                        'swallow_rate': round(float(sw), 3)}
            res[v] = {'all': agg(rows), 'tangle': agg([r for r in rows if r['touch']]),
                      'apart': agg([r for r in rows if not r['touch']]), 'videos': len({r['video'] for r in rows})}
            (WORK/f'rows_{v}.json').write_text(json.dumps(rows))
        print(json.dumps(res, indent=1))
        (WORK/'summary.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
