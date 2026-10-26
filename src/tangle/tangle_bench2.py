"""Tangle benchmark, round 2 (Harmony4D, same videos/crops/GT as tangle_bench.py): where does the remaining ownership
error come from, and can anatomy fix it?

Oracle experiments (ground truth used to build the prompts; they bound what a real component could achieve):
  ancN_mask   tracker restarted from a detector-quality mask with the right owner every N tracked frames (as
              tangle_bench seeds), forward + backward passes, fused by distance to the anchor each pass came from.
              If the error falls as N shrinks, it is drift between anchors (fixable by more/better anchors);
              if not, it is SAM's segmentation in contact (anchors cannot fix it).
  ancN_pose   masks only at the first/last frame; every N frames in between the anchor is a skeleton: positive
              point prompts on the person's visible ground-truth joints (no negatives: SAM-pose2seg, arXiv 2601.08982,
              found partner negatives unreliable). Bounds "a perfect skeleton tracker as the anchor source".
  paint_*     CPU post-process of an existing prediction: foreground pixels re-owned by the nearest ground-truth limb
              capsule (the 2026-10-11 limb_ownership experiment with perfect skeletons).
Estimated-pose experiments reuse ancN_pose with joints from a pose model instead of SMPL (phase 'pose').

  phases: joints (CPU) -> aseeds (image model) -> track (video model) -> paint (CPU) -> score (CPU)
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from tangle_bench import ROOT, STRIDE, WORK, render_crop, seed_masks, videos  # noqa: F401
from tangle_h4d import PEOPLE, Sequence, touching

CAMS = ('cam01', 'cam11')
ANCHOR_EVERY = 10   # tracked frames between anchor seeds (1 s at 10 fps); ancN variants use multiples of this
# SMPL joints used as skeleton points: pelvis, knees, ankles, spine3, neck, head, shoulders, elbows, wrists, hips
POSE_J = [0, 1, 2, 4, 5, 7, 8, 9, 12, 15, 16, 17, 18, 19, 20, 21]
BONES = [(0, 1), (0, 2), (1, 4), (2, 5), (4, 7), (5, 8), (0, 9), (9, 12), (12, 15), (12, 16), (12, 17), (16, 18),
         (17, 19), (18, 20), (19, 21), (16, 1), (17, 2)]   # last two: torso sides


def tracked_index(n_orig):
    idx = list(range(0, n_orig, STRIDE))
    if (n_orig - 1) % STRIDE:
        idx.append(n_orig - 1)
    return idx


def joints(wd):
    """GT SMPL joints projected into the crop at 1280x720: [T_orig, 2, 24, 3] (x, y, visible)."""
    out = wd/'joints.npz'
    if out.exists():
        return np.load(out)['kp']
    meta = json.loads((wd/'meta.json').read_text())
    s = Sequence(ROOT/meta['seq'])
    x0, y0, w, h = meta['crop']
    gt = np.load(wd/'gt.npz')['gt']
    kp = np.zeros((len(meta['frames']), 2, 24, 3), np.float32)
    for k, f in enumerate(meta['frames']):
        d = np.load(s.root/'processed_data/smpl'/f'{f:05d}.npy', allow_pickle=True).item()
        for n, who in enumerate(PEOPLE):
            uv, z = s.project(d[who]['joints'][:24], meta['cam'])
            uv = (uv - [x0, y0])*[1280/w, 720/h]
            for j in range(24):
                x, y = uv[j]
                xi, yi = int(round(x)), int(round(y))
                vis = 0 <= xi < 1280 and 0 <= yi < 720 and z[j] > 0 and gt[k][yi, xi] == n + 1
                kp[k, n, j] = (x, y, float(vis))
    np.savez_compressed(out, kp=kp)
    return kp


def anchor_ks(n, every):
    ks = list(range(0, n, every))
    if ks[-1] != n - 1:
        ks.append(n - 1)
    return ks


def fuse_anchored(pf, pb, ks, n, ious=None):
    """Per tracked frame: forward pass weighted by 1/(1 + distance to the previous anchor), backward by the next;
    with ious = (forward, backward) {frame: per-object predicted IoU}, each object's weight is also multiplied by its
    predicted IoU (clipped at 0.05; raw IoU head, not a probability) - SAM's own per-frame quality estimate."""
    import tangle_core
    labs = []
    for k in range(n):
        a = max(x for x in ks if x <= k)
        b = min(x for x in ks if x >= k)
        wf, wb = 1/(1 + k - a), 1/(1 + b - k)
        if ious is not None:
            if ious[0].get(k) is not None:
                wf = wf*np.clip(ious[0][k], 0.05, None)
            if ious[1].get(k) is not None:
                wb = wb*np.clip(ious[1][k], 0.05, None)
        lab, _ = tangle_core.fuse(pf.get(k), pb.get(k), wf, wb)
        lab = np.zeros((360, 640), np.uint8) if lab is None else lab
        labs.append(cv2.resize(lab.astype(np.uint8), (1280, 720), interpolation=cv2.INTER_NEAREST))
    return np.stack(labs)


def lowres_frames(wd, z):
    """Frames down-sampled by z and back to 1280x720: the pair at z times fewer pixels per side, as in a wide shot."""
    out = wd/f'frames_z{z}'
    src = sorted((wd/'frames').glob('*.jpg'))
    if len(list(out.glob('*.jpg'))) != len(src):
        out.mkdir(exist_ok=True)
        for f in src:
            im = cv2.imread(str(f))
            small = cv2.resize(im, (int(1280*z), int(720*z)), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(out/f.name), cv2.resize(small, (1280, 720), interpolation=cv2.INTER_LINEAR), [cv2.IMWRITE_JPEG_QUALITY, 92])
    return out


def pose_prompt(kp, n_person):
    """Positive points on the person's visible skeleton joints (pixel coords)."""
    pts = [kp[n_person, j, :2] for j in POSE_J if kp[n_person, j, 2] > 0]
    return np.array(pts, np.float32).reshape(-1, 2)


def capsule_dist(kp_person, shape=(720, 1280), radius=None):
    """Distance from every pixel to the person's skeleton bones (minus a body-size radius)."""
    canvas = np.full(shape, 255, np.uint8)
    p = kp_person[:, :2]
    scale = np.linalg.norm(p[12] - p[0]) + 1e-6   # neck-pelvis length
    for a, b in BONES:
        cv2.line(canvas, tuple(np.round(p[a]).astype(int)), tuple(np.round(p[b]).astype(int)), 0, 1)
    cv2.circle(canvas, tuple(np.round(p[15]).astype(int)), 1, 0, -1)
    d = cv2.distanceTransform(canvas, cv2.DIST_L2, 5)
    return d - (radius if radius is not None else 0.18*scale)


def paint(pred, kp, margin=10):
    """Foreground pixels move to the person whose skeleton is nearer by more than `margin` px."""
    out = pred.copy()
    fg = pred > 0
    d1, d2 = capsule_dist(kp[0]), capsule_dist(kp[1])
    to1 = fg & (d1 + margin < d2)
    to2 = fg & (d2 + margin < d1)
    out[to1], out[to2] = 1, 2
    return out


def score_rows(labelmaps, gt):
    rows = []
    for k in range(len(gt)):
        g, p = gt[k], labelmaps[k]
        r = {'k': k, 'touch': touching(g)}
        for n in (1, 2):
            body = g == n
            tot = max(1, body.sum())
            w = (body & (p == 3 - n)).astype(np.uint8)
            thick = cv2.dilate(cv2.erode(w, np.ones((21, 21), np.uint8)), np.ones((21, 21), np.uint8)) & w
            r[f'correct{n}'] = float((p[body] == n).sum()/tot)
            r[f'wrong{n}'] = float(w.sum()/tot)
            r[f'thick{n}'] = float(thick.sum()/tot)
        rows.append(r)
    return rows


def agg(rs):
    if not rs:
        return {}
    m = lambda key: float(np.mean([(r[key + '1'] + r[key + '2'])/2 for r in rs]))
    return {'frames': len(rs), 'correct': round(m('correct'), 4), 'wrong_owner': round(m('wrong'), 4),
            'limb_swap_px': round(m('thick'), 4),
            'limb_swap_frames': round(float(np.mean([max(r['thick1'], r['thick2']) > 0.05 for r in rs])), 4),
            'swallow_rate': round(float(np.mean([max(r['wrong1'], r['wrong2']) > 0.3 for r in rs])), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('phase', choices=['joints', 'aseeds', 'track', 'paint', 'score'])
    ap.add_argument('--variants', nargs='*', default=[])
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--cams', nargs='*', default=list(CAMS))
    a = ap.parse_args()
    vids = list(videos(tuple(a.cams)))
    if a.limit:
        vids = vids[:a.limit]
    if a.phase == 'joints':
        for name, cam in vids:
            joints(WORK/f'{name}_{cam}')
            print('joints', name, cam, flush=True)
    elif a.phase == 'aseeds':
        import autolabel_detect as det
        model, proc = det.load()
        for name, cam in vids:
            wd = WORK/f'{name}_{cam}'
            if (wd/'aseeds.npz').exists():
                continue
            gt = np.load(wd/'gt.npz')['gt']
            idx = tracked_index(len(gt))
            ks = anchor_ks(len(idx), ANCHOR_EVERY)
            out = {}
            for k in ks:
                m = seed_masks(proc, wd, idx[k], gt[idx[k]])
                out[f'{k}_1'], out[f'{k}_2'] = m[1], m[2]
            np.savez_compressed(wd/'aseeds.npz', **out)
            print('aseeds', name, cam, len(ks), flush=True)
    elif a.phase == 'track':
        import tangle_core
        tracker = tangle_core.load(non_overlap=True)
        for v in a.variants:
            # anc<N|inf>_<mask|pose|rtmo|cot>[_ms<thr>|_msmin<thr>][_iouw][_z<f>]: anchors every N tracked frames;
            # ms: SAM 3.1 memory selection (frames enter memory only if object score x predicted IoU > thr, mean or
            # min over the fighters); iouw: passes fused with per-fighter predicted-IoU weights; z: low-res frames
            parts = v.split('_')
            every = 10**9 if parts[0] == 'ancinf' else int(parts[0][3:])
            kind = parts[1]
            ms = next((x for x in parts[2:] if x.startswith('ms')), None)
            iouw = 'iouw' in parts[2:]
            zoom = next((float(x[1:]) for x in parts[2:] if x.startswith('z')), None)
            if ms or iouw:
                thr = float(ms.replace('msmin', '').replace('ms', '')) if ms else None
                tangle_core.memory_selection(tracker, thr, worst=bool(ms and ms.startswith('msmin')))
            else:
                tracker.use_memory_selection = False
            for name, cam in vids:
                wd = WORK/f'{name}_{cam}'
                out = wd/f'pred_{v}.npz'
                if out.exists() or not (wd/'aseeds.npz').exists():
                    continue
                if kind not in ('mask', 'pose') and not (wd/f'{kind}_kp.npz').exists():
                    continue
                S = np.load(wd/'aseeds.npz')
                fdir = wd/'frames' if zoom is None else lowres_frames(wd, zoom)
                frames = tangle_core.load_frames(fdir, tracker, STRIDE)
                n = len(frames[0])
                ks = anchor_ks(n, min(every, n - 1))
                if kind == 'mask':
                    seeds = {k: {1: S[f'{k}_1'], 2: S[f'{k}_2']} for k in ks}
                else:
                    kp = np.load(wd/f'{kind}_kp.npz')['kp'] if kind != 'pose' else joints(wd)
                    idx = tracked_index(len(kp))
                    seeds = {}
                    for k in ks:
                        if k in (0, n - 1):
                            seeds[k] = {1: S[f'{k}_1'], 2: S[f'{k}_2']}
                        else:
                            pts = {o: pose_prompt(kp[idx[k]], o - 1) for o in (1, 2)}
                            if all(len(p) for p in pts.values()):
                                seeds[k] = {o: {'points': pts[o]} for o in (1, 2)}
                    ks = sorted(seeds)
                # one fresh session per segment and direction, seeded only at its own anchor (as the production
                # segments mode; the multiplex demo cannot take a second prompt frame for existing objects)
                pf, pb, qf, qb = {}, {}, {}, {}
                for a_, b_ in zip(ks[:-1], ks[1:]):
                    f_, _, _ = tangle_core.track(tracker, frames, {a_: seeds[a_]}, start=a_, stop=b_, reverse=False, prompt='mixed')
                    fi_ = dict(tangle_core.IOU)
                    b2, _, _ = tangle_core.track(tracker, frames, {b_: seeds[b_]}, start=b_, stop=a_, reverse=True, prompt='mixed')
                    bi_ = dict(tangle_core.IOU)
                    for k in range(a_, b_ + 1):
                        if k < b_ or k not in pf:
                            pf[k], qf[k] = f_[k], fi_.get(k)
                        if k > a_ or k not in pb:
                            pb[k], qb[k] = b2[k], bi_.get(k)
                np.savez_compressed(out, labels=fuse_anchored(pf, pb, ks, n, (qf, qb) if iouw else None))
                print('tracked', v, name, cam, n, len(ks), flush=True)
    elif a.phase == 'paint':
        for v in a.variants:   # paint_<base variant>
            base = v[len('paint_'):]
            for name, cam in vids:
                wd = WORK/f'{name}_{cam}'
                if (wd/f'pred_{v}.npz').exists() or not (wd/f'pred_{base}.npz').exists():
                    continue
                pred = np.load(wd/f'pred_{base}.npz')['labels']
                kp = joints(wd)
                idx = tracked_index(len(kp))
                np.savez_compressed(wd/f'pred_{v}.npz', labels=np.stack([paint(pred[t], kp[idx[t]]) for t in range(len(pred))]))
                print('painted', v, name, cam, flush=True)
    else:
        res = {}
        common = None
        for v in a.variants:
            have = {(n, c) for n, c in vids if (WORK/f'{n}_{c}'/f'pred_{v}.npz').exists()}
            common = have if common is None else common & have
        for v in a.variants:
            rows = []
            for name, cam in sorted(common):
                wd = WORK/f'{name}_{cam}'
                gt = np.load(wd/'gt.npz')['gt']
                gt = gt[tracked_index(len(gt))]
                rows += [dict(r, video=f'{name}/{cam}') for r in score_rows(np.load(wd/f'pred_{v}.npz')['labels'], gt)]
            res[v] = {'videos': len(common), 'all': agg(rows), 'tangle': agg([r for r in rows if r['touch']]),
                      'apart': agg([r for r in rows if not r['touch']])}
            (WORK/f'rows2_{v}.json').write_text(json.dumps(rows))
            print(v, json.dumps(res[v]['tangle']), flush=True)
        (WORK/'summary2.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
