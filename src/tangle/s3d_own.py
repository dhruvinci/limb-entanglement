"""Pixel ownership from two fitted 3D bodies (SAM 3D Body), on the dotted frames of a review pack (runs in .venv-s3d).
Experimental.

Idea (the reviewer's "human shapes as primitives in 3D"): fit one full-body mesh per fighter, render both into the
frame with a depth buffer, and let the nearer body own each pixel. A mesh has volume, hip-attached legs and a front-to
-back order, which a skeleton or a 2D mask does not; on Harmony4D, skeleton painting fixed swallows but made limb blobs.
Variants written as tangle/<name>/tangle.json (label_masks on the dotted frames only, for tangle_dot_eval.py):
  <out>o  ownership only: inside the source's union of the two masks, owner = the rendered nearer body (else source)
  <out>f  full: union of the source masks and the rendered bodies, owner = rendered body where rendered (adds
          dropped limbs, may spill onto the mat where the fit is loose)
Prompt: each fighter's box and source mask (--prompt mask) or the box alone (--prompt box).

  ../.venv-s3d/bin/python s3d_own.py --pack reviews/tangle-ab-legs --src v4 --out s3dm --prompt mask
"""
import argparse
import json
from collections import defaultdict

import cv2
import numpy as np

from autolabel_common import ROOT, mask_box, rle_encode, write_json
from paths import SCRATCH
from autolabel_body3d import load_estimator
from tangle_dot_eval import load_dots
from tangle_review3 import masks as variant_masks


def project(v, cam_t, f, w, h):
    p = v + cam_t
    return np.stack([f*p[:, 0]/p[:, 2] + w/2, f*p[:, 1]/p[:, 2] + h/2], 1), p[:, 2]


def render(people, faces, w, h):
    """people: [(verts [V,3], cam_t, focal)] -> owner map (0 none, i+1) and depth, painter's algorithm per triangle."""
    tris = []
    for i, (v, t, f) in enumerate(people):
        uv, z = project(v, t, f, w, h)
        tri = uv[faces]
        tz = z[faces].mean(1)
        keep = (z[faces] > 0.05).all(1)
        tris += [(tz[n], i + 1, tri[n]) for n in np.nonzero(keep)[0]]
    tris.sort(key=lambda x: -x[0])
    own = np.zeros((h, w), np.uint8)
    for _, i, tri in tris:
        cv2.fillConvexPoly(own, np.round(tri*4).astype(np.int32), i, lineType=cv2.LINE_8, shift=2)
    return own


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument('--pack', default='reviews/tangle-ab-legs')
    ap.add_argument('--src', default='v4')
    ap.add_argument('--out', default='s3dm')
    ap.add_argument('--prompt', choices=['mask', 'box'], default='mask')
    a = ap.parse_args()
    dots = load_dots(ROOT/a.pack)
    frames = sorted({(d['iv'], d['k']) for d in dots})
    est = load_estimator()
    faces = est.faces
    cache, L, P = defaultdict(dict), {}, {}
    res = {a.out + 'o': defaultdict(dict), a.out + 'f': defaultdict(dict)}
    checked = False
    for iv, k in frames:
        base = ROOT/'autolabel/v1'/iv
        if iv not in L:
            L[iv] = json.loads((base/'labels.json').read_text())
            P[iv] = json.loads((base/'tangle/v4/tangle.json').read_text())['prim']
        ms = variant_masks(a.src, base, L[iv], k, P[iv], cache[(a.src, iv)])
        img = cv2.cvtColor(cv2.imread(L[iv]['frames'][k]['path']), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        people = []
        for m in ms:
            x1, y1, x2, y2 = mask_box(m)
            pad = 0.1*max(x2 - x1, y2 - y1)
            box = np.array([[max(0, x1 - pad), max(0, y1 - pad), min(w, x2 + pad), min(h, y2 + pad)]], np.float32)
            with torch.inference_mode():
                out = est.process_one_image(img, bboxes=box, masks=m[None].astype(np.uint8) if a.prompt == 'mask' else None,
                                            use_mask=a.prompt == 'mask', inference_type='body')
            r = out[0]
            people.append((np.asarray(r['pred_vertices'], np.float32), np.asarray(r['pred_cam_t'], np.float32),
                           float(np.asarray(r['focal_length']).reshape(-1)[0])))
            if not checked:   # camera convention: projected 3D keypoints must land on the predicted 2D keypoints
                uv, _ = project(np.asarray(r['pred_keypoints_3d'], np.float32), people[-1][1], people[-1][2], w, h)
                err = np.median(np.linalg.norm(uv - np.asarray(r['pred_keypoints_2d'])[:, :2], axis=1))
                print('projection check: median px error', round(float(err), 2), flush=True)
                checked = True
            est.batch = est.image_embeddings = est.output = None
            torch.cuda.empty_cache()
        own = render(people, faces, w, h)
        union = ms[0] | ms[1]
        src_lab = np.where(ms[0], 1, np.where(ms[1], 2, 0))
        o = np.where(union & (own > 0), own, src_lab)
        full = np.where(own > 0, own, src_lab)
        for name, lab in ((a.out + 'o', o), (a.out + 'f', full)):
            res[name][iv][str(k)] = {P[iv][i]: rle_encode(lab == i + 1) for i in range(2)}
        print(iv, k, 'rendered px', int((own > 0).sum()), 'union px', int(union.sum()), flush=True)
        np.save(f'{SCRATCH}/tangle/s3d_{a.out}_{iv.replace("/", "_")}_{k}.npy', own)
    for name, per in res.items():
        for iv, lm in per.items():
            d = ROOT/'autolabel/v1'/iv/'tangle'/name
            d.mkdir(parents=True, exist_ok=True)
            write_json(d/'tangle.json', {'variant': name, 'src': a.src, 'prim': P[iv], 'prompt': a.prompt, 'label_masks': lm,
                                         'note': 'dotted frames only (s3d_own.py experiment)'})


if __name__ == '__main__':
    main()
