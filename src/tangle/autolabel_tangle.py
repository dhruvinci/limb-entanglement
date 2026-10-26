"""Stage 10: tangle tracking of the primary pair (runs in .venv-sam31). Experimental; writes only <interval>/tangle/.

Single frames cannot settle who owns which pixels when two grapplers interlock: the SAM 3.1 detector, ViTPose, RTMO
and SAM-3D-Body all collapse the pair into one person there. Ownership is certain when the fighters are apart and is
conserved through the tangle, so the pair is tracked from separated moments on both sides of the tangle:

1. plan   (tangle_plan.py, needs a 2 fps detection scan, tangle_scan.py): nearest frames before and after the window,
          in the same shot, where the two fighters are distinct, confident and not touching (entry / exit anchors).
2. decode the span [entry, exit] at TRACK_FPS, always including the label frames and the anchor frames.
3. anchor masks: SAM 3.1 detections at the anchor frames, matched to the planned boxes (image model, phase 'anchors').
   Without an entry/exit anchor the pass is seeded from the best agreement anchor inside the window (label mask
   that matches the dense video mask, IoU >= 0.9: 96% precision for a good mask on the gold set).
4. track  (tangle_core.py, phase 'track'): forward from the entry anchor and backward from the exit anchor, both
   fighters in one session, full-mask prompts, non-overlap constraint before memory encoding, sigmoid outputs kept.
5. identity: each pass's two tracks are assigned to P1/P2 by IoU votes against the original label masks over the
   window (text detections only); the vote margin is kept.
6. fuse: per frame, the passes' probabilities weighted by proximity to their own anchor; per-fighter agreement
   between the passes is the uncertainty signal.
Outputs: tangle/{tangle.json, labelmap.npz}: half-resolution label maps (0 bg, 1 P1, 2 P2) for every span frame,
full-resolution RLE masks at the label frames, anchors, identity votes, per-frame agreement.
"""
import argparse
import gzip
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from autolabel_common import ROOT, box_iou, rle_decode, rle_encode, write_json
from autolabel_densify import decode_span

TRACK_FPS = 12.5
MAX_EXT_S = 45.0
MIN_ANCHOR_GAP = 12   # span frames (~1 s) between re-anchoring points
BAD_PASS_W = 0.05     # weight of a pass that reached its next anchor lost or swapped
CORR_RADIUS = 6      # with --corrections: agreement anchors this close to a corrected frame are dropped
MAX_GAP = 60          # with --fill-gaps, longer stretches without an anchor get a looser one     # longest stretch tracked outside the window to reach a separated anchor
WORK = Path('/tmp/tangle-work')


def original_labels(base):
    g = base/'labels.pre-maskfix.json.gz'
    if g.exists():
        with gzip.open(g, 'rt') as fh:
            return json.load(fh)
    return json.loads((base/'labels.json').read_text())


def agreement_anchors(base, L, prim, thr=0.9):
    """Label frames where both fighters' text-detection masks agree with the dense video masks (IoU >= thr)."""
    if not (base/'dense/dense.json').exists():
        return []
    D = json.loads((base/'dense/dense.json').read_text())
    M = np.load(base/'dense/masks.npz')
    h, w = D['shape']
    dt = np.array([f['time_s'] for f in D['frames']])
    out = []
    for k, fr in enumerate(L['frames']):
        j = int(np.argmin(np.abs(dt - fr['time_s'])))
        ins = {i['actor_id']: i for i in fr['instances'] if i['actor_id'] in prim}
        if abs(dt[j] - fr['time_s']) > 0.06 or len(ins) < 2:
            continue
        if any(i['source'] != 'sam31_text_detection' or not i['area'] for i in ins.values()):
            continue
        ious = []
        for a in prim:
            if f'{j}_{a}' not in M.files:
                break
            dm = np.unpackbits(M[f'{j}_{a}'])[:h*w].reshape(h, w).astype(bool)
            lm = rle_decode(ins[a]['mask_rle']).astype(bool)
            ious.append(float((dm & lm).sum()/max(1, (dm | lm).sum())))
        if len(ious) == 2 and min(ious) >= thr:
            out.append({'label_index': k, 'time_s': fr['time_s'], 'iou': round(min(ious), 3),
                        'prompt_anchor': bool(D['frames'][j].get('is_prompt_anchor'))})
    return out


def span_frames(iv, plan):
    """Decode the tracked span; returns (frame paths, times, {label_index: span index}, entry/exit span index)."""
    base = ROOT/'autolabel/v1'/iv
    L = original_labels(base)
    D = json.loads((base/'dense/dense.json').read_text()) if (base/'dense/dense.json').exists() else None
    master = Path(D['master_path']) if D else Path(json.loads((Path(L['frames'][0]['path']).parents[2]/'index.json').read_text())['master_path'])
    s_lab, e_lab = L['frames'][0]['time_s'], L['frames'][-1]['time_s']
    entry = plan['entry'] if plan.get('entry') and s_lab - plan['entry']['t'] <= MAX_EXT_S else None
    exit_ = plan['exit'] if plan.get('exit') and plan['exit']['t'] - e_lab <= MAX_EXT_S else None
    t0 = entry['t'] if entry else s_lab
    t1 = exit_['t'] if exit_ else e_lab
    work = WORK/iv.replace('/', '_')
    if (work/'span.json').exists():
        sp = json.loads((work/'span.json').read_text())
        if sp['t0'] == t0 and sp['t1'] == t1:
            return work, sp, L, entry, exit_
        shutil.rmtree(work)
    decoded = decode_span(master, t0 - 0.05, t1 + 0.05, work/'native')
    times = np.array([t for _, t in decoded])
    must = {int(np.argmin(np.abs(times - t))) for t in [f['time_s'] for f in L['frames']] + [t0, t1]}
    keep, last = [], -1e9
    for i, t in enumerate(times):
        if i in must or t - last >= 1/TRACK_FPS - 1e-6:
            keep.append(i)
            last = t
    (work/'frames').mkdir(parents=True, exist_ok=True)
    for k, i in enumerate(keep):
        shutil.move(str(decoded[i][0]), work/'frames'/f'{k:05d}.jpg')
    shutil.rmtree(work/'native')
    ts = [float(times[i]) for i in keep]
    near = lambda t: int(np.argmin(np.abs(np.array(ts) - t)))
    sp = {'t0': t0, 't1': t1, 'times': ts, 'label_to_span': {k: near(f['time_s']) for k, f in enumerate(L['frames'])},
          'entry_k': near(t0) if entry else None, 'exit_k': near(t1) if exit_ else None,
          'master': str(master)}
    (work/'span.json').write_text(json.dumps(sp))
    return work, sp, L, entry, exit_


def frame_cuts(work):
    """Hard cuts between consecutive span frames: mean absolute difference of 64x36 thumbnails far above the span's
    median (ffmpeg's scene score at 0.35 missed same-arena camera switches, e.g. a2s025)."""
    import cv2
    fs = sorted((work/'frames').glob('*.jpg'))
    th = [cv2.resize(cv2.imread(str(f)), (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32) for f in fs]
    d = np.array([np.abs(th[i] - th[i - 1]).mean() for i in range(1, len(th))])
    if not len(d):
        return []
    med = float(np.median(d))
    return [int(i) + 1 for i in np.nonzero(d > max(35.0, 3*med))[0]]


def label_fallback(L, prim, lo, hi, l2s):
    """Last-resort seed in a shot without anchors: the label frame where both fighters are confident text detections
    with the least mask overlap."""
    best = None
    for k, fr in enumerate(L['frames']):
        if not lo <= l2s[k] <= hi:
            continue
        ins = {i['actor_id']: i for i in fr['instances'] if i['actor_id'] in prim and i['area']}
        if len(ins) < 2 or any(i['source'] != 'sam31_text_detection' for i in ins.values()):
            continue
        sc = min(i.get('score', 0) for i in ins.values())
        if best is None or sc > best[0]:
            best = (sc, k)
    return best[1] if best and best[0] >= 0.6 else None


def anchor_masks(processor, work, sp, plan_anchor, k):
    """SAM 3.1 detections at a separated anchor frame matched to the planned fighter boxes -> {1: mask, 2: mask}."""
    from PIL import Image
    import autolabel_detect as det
    dets, _ = det.detect_frame(processor, Image.open(work/'frames'/f'{k:05d}.jpg').convert('RGB'))
    out = {}
    for n, (actor, box) in enumerate(sorted(plan_anchor['boxes'].items()), start=1):
        best = max(dets, key=lambda d: box_iou(d['box_xyxy'], box), default=None)
        if best is None or box_iou(best['box_xyxy'], box) < 0.5:
            return None
        out[n] = rle_decode(best['mask_rle']).astype(bool)
    if (out[1] & out[2]).sum() > 0.05*min(out[1].sum(), out[2].sum()):
        return None
    return out


POSE_GAP = 12      # with --pose-anchors: span frames between a mask anchor and the next skeleton anchor
COCO_TOPK, COCO_THR = 5, 0.5


def pose_gap_anchors(identity_variant, iv, a, b, mask_ks):
    """Skeleton anchors in the gaps between mask anchors (and the shot edges): every POSE_GAP frames, the RTMO
    skeletons (mask-free, whole frame; rtmo_spans.py) of the frame are assigned to P1/P2 by their confident joints'
    votes on the labelmap of an earlier tangle run (identity only; the joints themselves never come from masks);
    each fighter's prompt is its TOPK most confident joints (SAM-pose2seg prompts with the 3 most visible), positives
    only. On Harmony4D these points sit on the right fighter's joint 75% of the time and on the partner's 0.5%."""
    import itertools
    wd = WORK/iv.replace('/', '_')
    lm = ROOT/'autolabel/v1'/iv/'tangle'/identity_variant/'labelmap.npz'
    if not (wd/'rtmo.npz').exists() or not lm.exists():
        return []
    R = np.load(wd/'rtmo.npz')
    labs = np.load(lm)['labelmap']
    sy, sx = labs.shape[1]/720, labs.shape[2]/1280
    out, ks = [], sorted(set(mask_ks))
    bounds = [a - 1] + ks + [b + 1]
    for g0, g1 in zip(bounds[:-1], bounds[1:]):
        for k0 in range(g0 + POSE_GAP, g1 - POSE_GAP//2, POSE_GAP):
            k = next((x for x in (k0, k0 + 1, k0 - 1) if f'kp_{x}' in R), None)
            if k is None or not (a <= k <= b):
                continue
            P, S = R[f'kp_{k}'], R[f'sc_{k}']
            if len(P) < 2:
                continue
            votes = np.zeros((2, len(P)))
            for p in range(len(P)):
                for c in range(17):
                    if S[p, c] > COCO_THR:
                        y, x = int(P[p, c, 1]*sy), int(P[p, c, 0]*sx)
                        if 0 <= y < labs.shape[1] and 0 <= x < labs.shape[2] and labs[k][y, x] in (1, 2):
                            votes[labs[k][y, x] - 1, p] += S[p, c]
            # best one-to-one assignment of two skeletons to P1/P2 (no scipy in .venv-sam31)
            p1, p2 = max(itertools.permutations(range(len(P)), 2), key=lambda q: votes[0, q[0]] + votes[1, q[1]])
            pts = {}
            for n_, p in ((0, p1), (1, p2)):
                top = [c for _, c in sorted(((S[p, c], c) for c in range(17) if S[p, c] > COCO_THR), reverse=True)[:COCO_TOPK]]
                # a skeleton must win its fighter clearly (>= 2/3 of its vote mass) and have >= 3 confident joints
                if votes[n_, p] > 0 and votes[n_, p] >= 2*votes[1 - n_, p] and len(top) >= 3:
                    pts[n_ + 1] = {'points': P[p, top].astype(np.float32)}
            if len(pts) == 2:
                out.append((k, pts, 'pose', True))
    return out


ZOOM_MARGIN = 0.15  # crop margin around the pair's union box, per side, as a share of the box size


def shot_crop(L, prim, l2s, a, b, anc):
    """Zoom: a crop around the pair for one shot, so SAM sees the fighters at more pixels (the tracker input is a
    1008x1008 resize of whatever it is given; our pair spans ~45% of the width and ~60% of the height). The box is
    the union of both fighters' label masks over the shot's label frames and anchor masks, plus ZOOM_MARGIN, widened
    to at least 4:5 / at most 16:9; None when it would cover most of the frame anyway."""
    H, W = L['frames'][0]['height'], L['frames'][0]['width']
    boxes = []
    for li, j in l2s.items():
        if a <= j <= b:
            for i in L['frames'][li]['instances']:
                if i['actor_id'] in prim and i.get('area'):
                    boxes.append(i['box_xyxy'])
    for k, m, kind, known in anc:
        for v in m.values():
            if isinstance(v, np.ndarray) and v.any():
                ys, xs = np.nonzero(v)
                boxes.append([xs.min(), ys.min(), xs.max(), ys.max()])
    if not boxes:
        return None
    bx = np.array(boxes, float)
    x1, y1, x2, y2 = bx[:, 0].min(), bx[:, 1].min(), bx[:, 2].max(), bx[:, 3].max()
    w, h = (x2 - x1)*(1 + 2*ZOOM_MARGIN), (y2 - y1)*(1 + 2*ZOOM_MARGIN)
    w, h = max(w, 0.8*h, 320), max(h, w*9/16, 240)
    w, h = min(w, W), min(h, H)
    if w*h > 0.8*W*H:
        return None
    cx, cy = (x1 + x2)/2, (y1 + y2)/2
    x0, y0 = min(max(0, cx - w/2), W - w), min(max(0, cy - h/2), H - h)
    return [int(x0), int(y0), int(x0 + w), int(y0 + h)]


def iou(a, b):
    return float((a & b).sum()/max(1, (a | b).sum()))


def cv2_resize_bool(m, shape):
    import cv2
    return cv2.resize(m.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)


def votes(probs, L, sp, prim, obj_ids, scale):
    """2x2 IoU sums between a pass's tracks and the original P1/P2 label masks over the window (text detections
    count fully, tracked/fallback masks half: every label frame is evidence, since a pass seeded at an unlabelled
    separated anchor has no other source of identity)."""
    import cv2
    m = np.zeros((2, 2))
    n = 0
    for k, fr in enumerate(L['frames']):
        j = sp['label_to_span'][str(k)] if isinstance(next(iter(sp['label_to_span'])), str) else sp['label_to_span'][k]
        if j not in probs:
            continue
        ins = {i['actor_id']: i for i in fr['instances'] if i['actor_id'] in prim and i['area']}
        if len(ins) < 2:
            continue
        wt = 1.0 if all(i['source'] == 'sam31_text_detection' for i in ins.values()) else 0.5
        p = probs[j]
        lab = np.where(p.max(0) > 127, p.argmax(0) + 1, 0)
        for x, a in enumerate(prim):
            lm = cv2.resize(rle_decode(ins[a]['mask_rle']).astype(np.uint8), (p.shape[2], p.shape[1]), interpolation=cv2.INTER_NEAREST).astype(bool)
            for y in range(2):
                tm = lab == y + 1
                m[x, y] += wt*(tm & lm).sum()/max(1, (tm | lm).sum())
        n += wt
    keep = m[0, 0] + m[1, 1]
    swap = m[0, 1] + m[1, 0]
    order = [0, 1] if keep >= swap else [1, 0]  # order[x] = track index for prim[x]
    return {'order': order, 'keep': round(float(keep), 3), 'swap': round(float(swap), 3), 'frames': n,
            'margin': round(float(abs(keep - swap)/max(1, n)), 3)}


def run(targets, scan_dir, variant, non_overlap, prompt, mode, redo=False, presence=False, fill_gaps=False, ensemble=False,
        iou_weight=False, pose_anchors=None, holdout=False, zoom=False, corrections=None, corr_memory=False):
    import torch
    import tangle_plan
    import tangle_core
    jobs = []
    for iv in targets:
        if not redo and (ROOT/'autolabel/v1'/iv/'tangle'/variant/'tangle.json').exists():
            continue
        scan = json.loads((Path(scan_dir)/(iv.replace('/', '_') + '.json')).read_text())
        plan = tangle_plan.plan(iv, scan)
        work, sp, L, entry, exit_ = span_frames(iv, plan)
        jobs.append((iv, plan, work, sp, L, entry, exit_))
        print(iv, 'span', round(sp['t1'] - sp['t0'], 1), 's', len(sp['times']), 'frames; entry', bool(entry), 'exit', bool(exit_), flush=True)
    # phase 1: anchor masks with the image model
    import autolabel_detect as det
    model, processor = det.load()
    seeds = {}
    for iv, plan, work, sp, L, entry, exit_ in jobs:
        s = {}
        if entry:
            s['entry'] = anchor_masks(processor, work, sp, entry, sp['entry_k'])
        if exit_:
            s['exit'] = anchor_masks(processor, work, sp, exit_, sp['exit_k'])
        seeds[iv] = s
    del model, processor
    torch.cuda.empty_cache()
    # phase 2: tracking
    tracker = tangle_core.load(non_overlap=non_overlap)
    if iou_weight:
        tangle_core.memory_selection(tracker, None)   # selection on with no threshold: only logs the IoU predictions
    for iv, plan, work, sp, L, entry, exit_ in jobs:
        started = time.time()
        base = ROOT/'autolabel/v1'/iv
        prim = plan['prim']
        sp['label_to_span'] = {int(k): v for k, v in sp['label_to_span'].items()}
        frames = tangle_core.load_frames(work/'frames', tracker)
        n = len(sp['times'])
        times = np.array(sp['times'])
        agree_anchors = agreement_anchors(base, L, prim)
        s = seeds[iv]
        # shots: never track across a scene cut (scan cuts + hard cuts between span frames)
        if 'frame_cuts' not in sp:
            sp['frame_cuts'] = frame_cuts(work)
        cut_k = sorted({int(np.searchsorted(times, c)) for c in plan['window_cuts'] if times[0] < c < times[-1]} |
                       set(sp['frame_cuts']))
        bounds = [0] + cut_k + [n]
        shots = [(bounds[x], bounds[x + 1] - 1) for x in range(len(bounds) - 1) if bounds[x + 1] > bounds[x]]
        contrib = {j: [] for j in range(n)}   # j -> [(prob uint8 [2,h,w] in prim order, weight, pass)]
        passes, ident, handoffs, scores = [], {}, [], {}
        held, zooms = [], []
        l2s = sp['label_to_span']

        def label_seed(k):
            ins = {i['actor_id']: i for i in L['frames'][k]['instances']}
            return {1: rle_decode(ins[prim[0]]['mask_rle']).astype(bool), 2: rle_decode(ins[prim[1]]['mask_rle']).astype(bool)}

        def labels_of(p):
            return np.where(p.max(0) > 127, p.argmax(0) + 1, 0)

        def add(prob, k, name, order, sc=None, quality=1.0, ious=None):
            # weight: proximity to the pass's anchor, times (presence) the tracker's own belief that each fighter is
            # visible, so a pass that has lost a fighter does not pull the fused estimate to background, times
            # (iou_weight) the decoder's predicted IoU per fighter: on Harmony4D its worst-fighter value ranks the
            # frames by ownership error (Spearman -0.81; bottom decile 34% wrong-owner vs 7.5%)
            for j, pj in prob.items():
                w = np.full(2, quality/(1 + abs(j - k)))
                if presence and sc is not None and j in sc:
                    w = w*(1/(1 + np.exp(-sc[j][order])))
                if ious is not None and j in ious:
                    w = w*np.clip(ious[j][order], 0.05, None)
                contrib[j].append((pj[order], w, name))

        for a, b in shots:
            # anchors in this shot: (span index, masks, kind, identity known)
            anc = []
            if s.get('entry') and a <= sp['entry_k'] <= b:
                anc.append((sp['entry_k'], s['entry'], 'entry', False))
            if s.get('exit') and a <= sp['exit_k'] <= b:
                anc.append((sp['exit_k'], s['exit'], 'exit', False))
            in_shot = [x for x in agree_anchors if a <= l2s[x['label_index']] <= b]
            if mode == 'segments':
                picked = []
                cands = []
                for x in sorted(in_shot, key=lambda x: -x['iou']):
                    j = l2s[x['label_index']]
                    if all(abs(j - y) >= MIN_ANCHOR_GAP for y in picked + [c[0] for c in anc]):
                        picked.append(j)
                        cands.append((j, x['label_index']))
                # holdout: every second agreement anchor (time order) is withheld from tracking and used as a
                # pseudo ground truth (agreement anchors are good masks 96% of the time on the gold set)
                for r_, (j, li) in enumerate(sorted(cands)):
                    if holdout and r_ % 2 == 1:
                        held.append(li)
                    else:
                        anc.append((j, label_seed(li), 'window_agreement', True))
                if fill_gaps and not holdout:
                    # gaps longer than MAX_GAP frames: add the best looser (IoU >= 0.8) agreement anchor inside the gap
                    loose = [x for x in agreement_anchors(base, L, prim, 0.8) if a <= l2s[x['label_index']] <= b]
                    while True:
                        ks = sorted([a - 1] + [c[0] for c in anc] + [b + 1])
                        gaps = [(ks[i], ks[i + 1]) for i in range(len(ks) - 1) if ks[i + 1] - ks[i] > MAX_GAP]
                        cand = [x for x in loose if any(g0 + MIN_ANCHOR_GAP <= l2s[x['label_index']] <= g1 - MIN_ANCHOR_GAP for g0, g1 in gaps)]
                        if not cand:
                            break
                        x = max(cand, key=lambda x: x['iou'])
                        loose.remove(x)
                        anc.append((l2s[x['label_index']], label_seed(x['label_index']), 'window_agreement_0.8', True))
            else:
                for x in sorted(in_shot, key=lambda x: -x['iou'])[:2]:
                    anc.append((l2s[x['label_index']], label_seed(x['label_index']), 'window_agreement', True))
            # corrections: reviewer-corrected masks on label frames (dotfix.py) become anchors; agreement anchors
            # within CORR_RADIUS frames are dropped (on the leg set the seeds themselves carry the leg errors)
            fix = {int(k): v for k, v in (corrections or {}).get(iv, {}).items() if a <= l2s[int(k)] <= b}
            if fix and mode == 'segments':
                cj = [l2s[k] for k in fix]
                anc = [c for c in anc if c[2] in ('entry', 'exit') or all(abs(c[0] - j) > CORR_RADIUS for j in cj)]
                for k, v in fix.items():
                    anc.append((l2s[k], {1: rle_decode(v[prim[0]]).astype(bool), 2: rle_decode(v[prim[1]]).astype(bool)},
                                'correction', True))
            if not anc:
                loose = [x for x in agreement_anchors(base, L, prim, 0.8) if a <= l2s[x['label_index']] <= b]
                if loose:
                    x = max(loose, key=lambda x: x['iou'])
                    anc.append((l2s[x['label_index']], label_seed(x['label_index']), 'window_agreement_0.8', True))
                else:
                    k = label_fallback(L, prim, a, b, l2s)
                    if k is not None:
                        anc.append((l2s[k], label_seed(k), 'label_fallback', True))
            if not anc:
                continue
            if pose_anchors is not None and mode == 'segments':
                anc += pose_gap_anchors(pose_anchors, iv, a, b, [c[0] for c in anc])
            anc.sort(key=lambda c: (c[0], c[2] != 'entry'))
            if mode != 'segments':
                first, last = anc[0], anc[-1]
                chosen = [first] + ([last] if last[0] != first[0] and mode == 'bidir' else [])
                for k, m, kind, known in chosen:
                    prob, sc = {}, {}
                    if k < b:
                        r = tangle_core.track(tracker, frames, {k: m}, start=k, reverse=False, stop=b, prompt=prompt)
                        prob.update(r[0]); sc.update(r[2])
                    if k > a:
                        r = tangle_core.track(tracker, frames, {k: m}, start=k, reverse=True, stop=a, prompt=prompt)
                        prob.update(r[0]); sc.update(r[2])
                    name = f'{kind}@{k}'
                    v = votes(prob, L, sp, prim, [1, 2], 0.5)
                    if known:
                        v['order'] = [0, 1]  # seeded from the labelled P1/P2 masks
                    ident[name] = v
                    add(prob, k, name, v['order'], sc)
                    passes.append({'name': name, 'k': k, 'kind': kind, 'shot': [a, b]})
                continue
            # segments: track both ways between consecutive anchors, backward from the first to the shot start and
            # forward from the last to the shot end. Identity of an entry/exit anchor (unknown) comes from its
            # hand-off to the nearest labelled anchor; a labelled anchor whose hand-off disagrees is flagged.
            runs = []   # (k, masks, kind, known, direction, stop)
            if anc[0][0] > a:
                runs.append(anc[0] + (True, a))
            for x in range(len(anc) - 1):
                runs.append(anc[x] + (False, anc[x + 1][0]))
                runs.append(anc[x + 1] + (True, anc[x][0]))
            if anc[-1][0] < b:
                runs.append(anc[-1] + (False, b))
            out = []
            for k, m, kind, known, rev, stop in runs:
                # zoom: crop around the pair over this segment only (the pair moves less in 1-3 s than in a shot)
                crop = None
                if zoom:
                    crop = shot_crop(L, prim, l2s, min(k, stop), max(k, stop), [c for c in anc if c[0] in (k, stop)])
                    if crop is not None:
                        zooms.append({'from': k, 'to': stop, 'crop': crop})
                # corr_memory: every corrected frame of the shot also enters this pass as a conditioning frame, so the
                # tracker attends to all of the reviewer's corrections at once (not only the anchor it starts from)
                seeds = {k: m}
                if corr_memory and not isinstance(m[1], dict):
                    seeds.update({c[0]: c[1] for c in anc if c[2] == 'correction' and c[0] != k})
                for fl in ((False, True) if ensemble else (False,)):
                    is_pose = isinstance(m[1], dict)
                    prob, _, sc = tangle_core.track(tracker, frames, seeds, start=k, reverse=rev, stop=stop,
                                                    prompt='mixed' if is_pose else prompt, flip=fl and not is_pose, crop=crop)
                    out.append(dict(k=k, m=m, kind=kind, known=known, rev=rev, stop=stop, flip=fl, prob=prob, sc=sc,
                                    ious=dict(tangle_core.IOU) if iou_weight else None))
            # identity: labelled anchors are [0,1]; an unknown anchor takes the order that best matches the labelled
            # anchor at the other end of its segment (summed over its samples), else the window votes
            order = {r['k']: [0, 1] for r in out if r['known']}
            for r in out:
                if r['k'] in order:
                    continue
                far = next((c for c in anc if c[0] == r['stop'] and c[3] and not isinstance(c[1][1], dict)), None)
                same = [x for x in out if x['k'] == r['k'] and x['stop'] == r['stop'] and x['stop'] in x['prob']]
                if far is not None and same:
                    keep = swap = 0.0
                    for x in same:
                        lab = labels_of(x['prob'][x['stop']])
                        g = [cv2_resize_bool(far[1][i], lab.shape) for i in (1, 2)]
                        keep += iou(lab == 1, g[0]) + iou(lab == 2, g[1])
                        swap += iou(lab == 1, g[1]) + iou(lab == 2, g[0])
                    order[r['k']] = [0, 1] if keep >= swap else [1, 0]
                    ident[f"{r['kind']}@{r['k']}"] = {'order': order[r['k']], 'keep': round(keep, 3), 'swap': round(swap, 3),
                                                      'by': f"handoff@{r['stop']}"}
                else:
                    v = votes(r['prob'], L, sp, prim, [1, 2], 0.5)
                    order[r['k']] = v['order']
                    ident[f"{r['kind']}@{r['k']}"] = dict(v, by='votes')
            # each pass is a noisy sample (fp16 noise, mirroring or the competition flag can flip a tangle outcome):
            # a pass that reaches the next anchor lost or swapped is down-weighted over its whole segment
            for r in out:
                k, rev, stop = r['k'], r['rev'], r['stop']
                name = f"{r['kind']}@{k}{'<' if rev else '>'}{stop}{'f' if r['flip'] else ''}"
                quality, state = 1.0, None
                far = next((c for c in anc if c[0] == stop and not isinstance(c[1][1], dict)), None)
                if far is not None and stop in r['prob'] and stop != k:
                    lab = labels_of(r['prob'][stop][order[k]])
                    g = [cv2_resize_bool(far[1][i], lab.shape) for i in (1, 2)]
                    fo = order.get(stop, [0, 1])
                    g = [g[fo[0]], g[fo[1]]]
                    ious = [iou(lab == 1, g[0]), iou(lab == 2, g[1])]
                    sw = [iou(lab == 1, g[1]), iou(lab == 2, g[0])]
                    state = 'lost' if max(ious + sw) < 0.3 else ('swap' if sum(sw) > sum(ious) else ('ok' if min(ious) >= 0.5 else 'weak'))
                    quality = {'ok': 1.0, 'weak': 0.5, 'swap': BAD_PASS_W, 'lost': BAD_PASS_W}[state]
                    handoffs.append({'from': k, 'to': stop, 'flip': r['flip'], 'state': state, 'iou': [round(x, 3) for x in ious],
                                     'swapped_iou': [round(x, 3) for x in sw]})
                add({j: v for j, v in r['prob'].items()}, k, name, order[k], r['sc'], quality, r['ious'])
                passes.append({'name': name, 'k': k, 'kind': r['kind'], 'shot': [a, b], 'reverse': rev, 'stop': stop,
                               'flip': r['flip'], 'handoff': state})
        if not passes:
            print(iv, 'no seed in any shot: skipped', flush=True)
            continue
        import cv2
        h2, w2 = next(c for c in contrib.values() if c)[0][0].shape[1:]
        labelmaps = np.zeros((n, h2, w2), np.uint8)
        agree, fused = [], {}
        label_js = set(sp['label_to_span'].values())
        for j in range(n):
            c = contrib[j]
            if not c:
                agree.append(None)
                continue
            prob = sum(p.astype(np.float32)*w[:, None, None] for p, w, _ in c)/np.maximum(sum(w for _, w, _ in c), 1e-6)[:, None, None]/255
            if j in label_js:
                fused[j] = prob
            labelmaps[j] = cv2.medianBlur(np.where(prob.max(0) > 0.5, prob.argmax(0) + 1, 0).astype(np.uint8), 5)
            if len(c) >= 2:
                la = [np.where(p.max(0) > 127, p.argmax(0) + 1, 0) for p, _, _ in c[:2]]
                agree.append([float(((la[0] == i) & (la[1] == i)).sum()/max(1, ((la[0] == i) | (la[1] == i)).sum())) for i in (1, 2)])
            else:
                agree.append(None)
        H, W = L['frames'][0]['height'], L['frames'][0]['width']
        label_masks = {}
        for k, j in sp['label_to_span'].items():
            if j in fused:
                up = np.stack([cv2.resize(ch, (W, H), interpolation=cv2.INTER_LINEAR) for ch in fused[j]])
                lab = cv2.medianBlur(np.where(up.max(0) > 0.5, up.argmax(0) + 1, 0).astype(np.uint8), 7)
                label_masks[k] = {prim[x]: rle_encode(lab == x + 1) for x in range(2)}
        heldout = {}
        for li in held:
            if li in label_masks:
                gtm = label_seed(li)
                heldout[str(li)] = [round(iou(rle_decode(label_masks[li][prim[x]]).astype(bool), gtm[x + 1]), 4) for x in range(2)]
        out = base/'tangle'/variant
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out/'labelmap.npz', labelmap=labelmaps)
        write_json(out/'tangle.json', {
            'version': 1, 'variant': variant, 'iou_weight': iou_weight, 'pose_anchors': pose_anchors, 'presence': presence, 'fill_gaps': fill_gaps, 'ensemble': ensemble, 'non_overlap_mem': non_overlap, 'prompt': prompt, 'mode': mode,
            'prim': prim, 'pair_fixed': plan.get('pair_fixed'), 'span': [sp['t0'], sp['t1']], 'times': sp['times'],
            'label_to_span': sp['label_to_span'],
            'entry': {'t': entry['t'], 'k': sp['entry_k'], 'mask_ok': bool(s.get('entry'))} if entry else None,
            'exit': {'t': exit_['t'], 'k': sp['exit_k'], 'mask_ok': bool(s.get('exit'))} if exit_ else None,
            'plan_reasons': [plan['entry_reason'], plan['exit_reason']], 'window_cuts': plan['window_cuts'], 'shots': shots,
            'frame_cuts': sp['frame_cuts'], 'passes': passes, 'identity': ident, 'handoffs': handoffs, 'agreement': agree, 'agreement_anchors': agree_anchors,
            'label_masks': {str(k): v for k, v in label_masks.items()}, 'heldout_iou': heldout, 'zoom': zooms, 'corrections': sorted((corrections or {}).get(iv, {}), key=int), 'corr_memory': corr_memory, 'elapsed_s': round(time.time() - started, 1)})
        print(iv, variant, 'shots', len(shots), 'passes', [p['name'] for p in passes],
              'identity margins', {k: v.get('margin', v.get('by')) for k, v in ident.items()}, f'{time.time() - started:.0f}s', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--scan-dir', required=True)
    ap.add_argument('--variant', default='v1')
    ap.add_argument('--non-overlap', type=int, default=1)
    ap.add_argument('--prompt', choices=['mask', 'points'], default='mask')
    ap.add_argument('--mode', choices=['bidir', 'fwd', 'bwd', 'segments'], default='bidir')
    ap.add_argument('--redo', action='store_true')
    ap.add_argument('--presence', type=int, default=0, help='weight passes by the tracker object score')
    ap.add_argument('--fill-gaps', type=int, default=0, help='segments: add IoU>=0.8 anchors in gaps > MAX_GAP')
    ap.add_argument('--ensemble', type=int, default=0, help='segments: also track the mirrored video')
    ap.add_argument('--iou-weight', type=int, default=0, help='weight passes by the per-fighter predicted IoU')
    ap.add_argument('--pose-anchors', default=None, help='segments: skeleton anchors in gaps; identity from this variant')
    ap.add_argument('--holdout', type=int, default=0, help='withhold every 2nd agreement anchor and score against it')
    ap.add_argument('--zoom', type=int, default=0, help='segments: track each shot inside a crop around the pair')
    ap.add_argument('--corrections', default=None, help='segments: JSON {interval: {label_index: {actor: rle}}} of corrected masks used as anchors')
    ap.add_argument('--corr-memory', type=int, default=0, help='segments: all corrected frames of a shot are conditioning frames of every pass')
    ap.add_argument('--corr-radius', type=int, default=CORR_RADIUS, help='span frames around a correction cleared of agreement anchors')
    ap.add_argument('targets', nargs='+')
    a = ap.parse_args()
    CORR_RADIUS = a.corr_radius
    run(a.targets, a.scan_dir, a.variant, bool(a.non_overlap), a.prompt, a.mode, a.redo, bool(a.presence), bool(a.fill_gaps), bool(a.ensemble),
        bool(a.iou_weight), a.pose_anchors, bool(a.holdout), bool(a.zoom),
        json.loads(Path(a.corrections).read_text()) if a.corrections else None, bool(a.corr_memory))
