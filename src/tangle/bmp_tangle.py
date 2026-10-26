"""BBoxMaskPose-v2 refinement of the tangle tracker's masks (runs in .venv-bmp; external/BBoxMaskPose, GPL-3.0).

What BMPv2 does well in crowds, and what broke it on BJJ (movement-research MASK_LOOP): its detector misses the second
grappler on most contact frames and pose-NMS then deletes the hidden athlete. Here the tracker supplies who and where,
and BMP's two strong parts do the rest, per label frame and per fighter:
  1. PMPose (mask-conditioned top-down pose): box + the fighter's current mask -> 17 keypoints, presence, visibility
  2. SAM-pose2seg (SAM 2.1 b+ fine-tuned to turn pose keypoints into a FULL-body mask): the 3 most visible keypoints
     of the fighter, positives only, no box (BMPv2 defaults); exclusive masks between the two fighters
  3. repeat once with the new masks (BMP's loop, 2 iterations)
Outputs <interval>/tangle/<out>/bmp.json: per label frame, BMP masks and keypoints for both fighters, plus the
pose-mask consistency (0.5 * own keypoints inside + 0.5 * partner keypoints outside; BMP's own check) of the tracker
mask and of the BMP mask. A second variant <out>c picks, per fighter and frame, the more consistent of the two
(ties keep the tracker), then makes the pair exclusive.

  ../.venv-bmp/bin/python bmp_tangle.py --src v4 --out bmp <trial/interval ...>
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from paths import BMP  # noqa: E402
sys.path.insert(0, str(BMP))
from autolabel_common import ROOT, rle_decode, rle_encode, write_json  # noqa: E402

CKPT = Path.home()/'.cache/torch/hub/checkpoints'
SAM_ARGS = dict(batch=False, use_bbox=False, num_pos_keypoints=3, num_pos_keypoints_if_crowd=3, num_neg_keypoints=0,
                confidence_thr=0.5, visibility_thr=0.5, selection_method='k_most_visible', extend_bbox=False,
                pose_mask_consistency=False, crowd_by_max_iou=None, crop=False, exclusive_masks=True,
                ignore_small_bboxes=False)
ITERS = 2
KPT_THR = 0.5


def box_of(m):
    ys, xs = np.nonzero(m)
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())] if len(xs) else None


def clean(m):
    """SAM-pose2seg masks come back speckled on dark clothing (the SAM 2 hole-filling post-process needs its CUDA
    extension, not built here): close small gaps, fill holes, drop specks under 2% of the mask."""
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(m)
    area = sum(cv2.contourArea(c) for c in cnts)
    keep = [c for c in cnts if cv2.contourArea(c) >= 0.02*area]
    cv2.drawContours(out, keep, -1, 1, -1)
    return out.astype(bool)


def consistency(mask, own, other):
    """BMP's pose-mask consistency: 0.5 * share of own confident keypoints inside + 0.5 * share of the partner's outside."""
    def inside(k):
        k = k[k[:, 2] > KPT_THR]
        if not len(k):
            return None
        x = np.clip(k[:, 0].astype(int), 0, mask.shape[1] - 1)
        y = np.clip(k[:, 1].astype(int), 0, mask.shape[0] - 1)
        return mask[y, x].mean()
    a, b = inside(own), inside(other)
    return 0.5*(a if a is not None else 0.0) + 0.5*(1 - b if b is not None else 1.0)


def load(device):
    from pmpose import PMPose
    from bboxmaskpose.sam2_utils import prepare_model
    os.chdir(BMP/'mmpose')   # PMPose configs name their dataset metainfo relative to mmpose/
    pm = PMPose(device=device, variant='PMPose-b', from_pretrained=False)
    pm.load_from_file(str(CKPT/'PMPose-b-1.0.0.pth'))
    os.chdir(BMP)
    from hydra import initialize_config_module
    from hydra.core.global_hydra import GlobalHydra
    if not GlobalHydra.instance().is_initialized():
        initialize_config_module('bboxmaskpose.sam2', version_base='1.2')
    sam = prepare_model(str(BMP/'bboxmaskpose/sam2/configs/sam-pose2seg/sam-pose2seg_hiera_b+.yaml'), str(CKPT/'SAM-pose2seg_hiera_b%2B.pt'))
    return pm, sam


def refine(pm, sam, img, masks):
    """BMP loop seeded with the tracker's two masks -> (masks [2,H,W] bool, keypoints [2,17,3])."""
    from mmengine.structures import InstanceData
    from bboxmaskpose.demo_utils import DotDict
    from bboxmaskpose.sam2_utils import process_image_with_SAM
    kps = None
    for _ in range(ITERS):
        boxes = [box_of(m) for m in masks]
        if any(b is None for b in boxes) or min(m.sum() for m in masks) < 200:
            break
        try:
            kp, pres, vis, _ = pm.predict(img, np.array(boxes), masks=np.stack(masks).astype(np.uint8))
        except ValueError:   # PMPose rejects a mask its crop does not contain: keep the last good result
            break
        kp, pres, vis = kp[:, :17], pres[:, :17], vis[:, :17]
        kps = kp
        inst = InstanceData(bboxes=np.array(boxes), bbox_scores=np.ones(2), keypoints=kp, keypoint_scores=kp[:, :, 2],
                            keypoint_vis=vis, keypoint_prob=pres, pred_masks=np.stack(masks).astype(np.uint8))
        out = process_image_with_SAM(DotDict(SAM_ARGS), img[:, :, ::-1].copy(), sam, inst, None)   # SAM 2 expects RGB
        masks = [clean(out.refined_masks[i]) for i in range(2)]
        masks[1] &= ~masks[0] if out.sam_scores[0] >= out.sam_scores[1] else masks[1]
        masks[0] &= ~masks[1]
    return masks, kps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='v4')
    ap.add_argument('--out', default='bmp')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--limit', type=int, default=0, help='label frames per clip (0 = all)')
    ap.add_argument('--kpts', type=int, default=3, help='positive keypoints per fighter for SAM-pose2seg (BMPv2: 3)')
    ap.add_argument('targets', nargs='+')
    a = ap.parse_args()
    pm, sam = load(a.device)
    SAM_ARGS.update(num_pos_keypoints=a.kpts, num_pos_keypoints_if_crowd=a.kpts)
    for iv in a.targets:
        base = ROOT/'autolabel/v1'/iv
        src = base/'tangle'/a.src/'tangle.json'
        if not src.exists():
            print('no', a.src, iv, flush=True)
            continue
        if (base/'tangle'/(a.out + 'c')/'tangle.json').exists():
            continue
        T = json.loads(src.read_text())
        L = json.loads((base/'labels.json').read_text())
        prim = T['prim']
        started = time.time()
        frames, picked = {}, {}
        ks = sorted(T['label_masks'], key=int)
        if a.limit:
            ks = ks[:a.limit]
        for k in ks:
            img = cv2.imread(L['frames'][int(k)]['path'])
            tm = [rle_decode(T['label_masks'][k][p]).astype(bool) for p in prim]
            if not all(m.any() for m in tm):
                continue
            bm, kp = refine(pm, sam, img, tm)
            if kp is None:
                continue
            kp = np.asarray(kp, np.float32)
            cons = {'tracker': [consistency(tm[i], kp[i], kp[1 - i]) for i in range(2)],
                    'bmp': [consistency(bm[i], kp[i], kp[1 - i]) for i in range(2)]}
            frames[k] = {'masks': {p: rle_encode(bm[i]) for i, p in enumerate(prim)}, 'keypoints': kp.round(2).tolist(),
                         'consistency': {s: [round(float(x), 3) for x in v] for s, v in cons.items()}}
            # per fighter: the more pose-consistent mask (ties keep the tracker)
            ch = [bm[i] if cons['bmp'][i] > cons['tracker'][i] + 1e-6 else tm[i] for i in range(2)]
            ov = ch[0] & ch[1]   # mixed sources can overlap: the tracker's (exclusive) owner keeps those pixels
            ch = [np.where(ov, tm[i], ch[i]) for i in range(2)]
            picked[k] = {'masks': {p: rle_encode(ch[i]) for i, p in enumerate(prim)},
                         'source': ['bmp' if cons['bmp'][i] > cons['tracker'][i] + 1e-6 else 'tracker' for i in range(2)]}
        for name, data in ((a.out, frames), (a.out + 'c', picked)):
            out = base/'tangle'/name
            out.mkdir(parents=True, exist_ok=True)
            write_json(out/'tangle.json', {'variant': name, 'src': a.src, 'prim': prim, 'label_to_span': T['label_to_span'],
                                           'times': T['times'], 'agreement': T.get('agreement', []),
                                           'label_masks': {k: v['masks'] for k, v in data.items()},
                                           'frames': {k: {x: y for x, y in v.items() if x != 'masks'} for k, v in data.items()}})
        src_counts = [s for v in picked.values() for s in v['source']]
        print(iv, len(frames), 'frames', f'{time.time() - started:.0f}s', 'chosen bmp', src_counts.count('bmp'), 'tracker',
              src_counts.count('tracker'), flush=True)


if __name__ == '__main__':
    main()
