"""Body-part segmentation (Sapiens 0.6B, Goliath 28 classes) on the dotted frames of a review pack, and two ways of
using it to repair the tracker's masks (runs in .venv-s3d). Experimental.

Sapiens labels every human pixel with a body part (thigh, shin, foot, ...) but not with a person. Half of v4's leg
errors on the leg pack are limb pixels nobody owns, so the first use is coverage:
  <out>w  watershed fill: pixels Sapiens calls human and the source leaves unowned are grown from the two fighters'
          masks along image edges (cv2.watershed, markers = source owners + Sapiens background); owned pixels kept
Also writes the part map per frame (npy) for the connectivity experiment.

  ../.venv-s3d/bin/python sapiens_parts.py --pack reviews/tangle-ab-legs --src v4 --out sap
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from autolabel_common import ROOT, rle_encode, write_json
from paths import SCRATCH
from tangle_dot_eval import load_dots
from tangle_review3 import masks as variant_masks

CKPT = next((Path.home()/'.cache/huggingface/hub/models--facebook--sapiens-seg-0.6b-torchscript/snapshots').glob('*/*.pt2'))
DEV, DTYPE = 'cuda', None
OUT_DIR = SCRATCH/'tangle/sapiens'
CLASSES = ['Background', 'Apparel', 'Face_Neck', 'Hair', 'Left_Foot', 'Left_Hand', 'Left_Lower_Arm', 'Left_Lower_Leg',
           'Left_Shoe', 'Left_Sock', 'Left_Upper_Arm', 'Left_Upper_Leg', 'Lower_Clothing', 'Right_Foot', 'Right_Hand',
           'Right_Lower_Arm', 'Right_Lower_Leg', 'Right_Shoe', 'Right_Sock', 'Right_Upper_Arm', 'Right_Upper_Leg', 'Torso',
           'Upper_Clothing', 'Lower_Lip', 'Upper_Lip', 'Lower_Teeth', 'Tongue', 'Upper_Teeth']
# the torchscript export normalises internally: feed raw 0-255 RGB (mean/std here gave ~0 human pixels)


def crop_box(ms, shape, pad=0.15):
    ys, xs = np.nonzero(ms[0] | ms[1])
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
    p = pad*max(x2 - x1, y2 - y1)
    return int(max(0, x1 - p)), int(max(0, y1 - p)), int(min(shape[1], x2 + p)), int(min(shape[0], y2 + p))


def parts(model, img_bgr, box):
    """Part labels [H, W] uint8 for the crop, pasted into a full-frame map (0 = background outside the crop)."""
    import torch
    x1, y1, x2, y2 = box
    crop = np.ascontiguousarray(img_bgr[y1:y2, x1:x2, ::-1]).astype(np.float32)
    inp = cv2.resize(crop, (768, 1024), interpolation=cv2.INTER_LINEAR)
    t = torch.from_numpy(inp.transpose(2, 0, 1)).unsqueeze(0).to(DEV, DTYPE)
    with torch.inference_mode():
        logits = model(t)
    logits = torch.nn.functional.interpolate(logits.float(), size=(y2 - y1, x2 - x1), mode='bilinear', align_corners=False)
    lab = logits[0].argmax(0).byte().cpu().numpy()
    full = np.zeros(img_bgr.shape[:2], np.uint8)
    full[y1:y2, x1:x2] = lab
    return full


def watershed_fill(img, ms, human):
    markers = np.zeros(img.shape[:2], np.int32)
    markers[~human & ~(ms[0] | ms[1])] = 3
    markers[ms[0]] = 1
    markers[ms[1]] = 2
    ws = cv2.watershed(img.copy(), markers)
    out = [ms[0] | ((ws == 1) & human), ms[1] | ((ws == 2) & human)]
    out[1] &= ~out[0]
    return out


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument('--pack', default='reviews/tangle-ab-legs')
    ap.add_argument('--src', default='v4')
    ap.add_argument('--out', default='sap')
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dots = load_dots(ROOT/a.pack)
    frames = sorted({(d['iv'], d['k']) for d in dots})
    global DEV, DTYPE
    DEV, DTYPE = a.device, torch.float16 if a.device == 'cuda' else torch.float32
    model = torch.jit.load(str(CKPT), map_location=DEV).to(DTYPE).eval()
    cache, L, P = defaultdict(dict), {}, {}
    res = defaultdict(dict)
    cover = defaultdict(lambda: [0, 0])
    for iv, k in frames:
        base = ROOT/'autolabel/v1'/iv
        if iv not in L:
            L[iv] = json.loads((base/'labels.json').read_text())
            P[iv] = json.loads((base/'tangle/v4/tangle.json').read_text())['prim']
        ms = variant_masks(a.src, base, L[iv], k, P[iv], cache[(a.src, iv)])
        img = cv2.imread(L[iv]['frames'][k]['path'])
        pm = parts(model, img, crop_box(ms, img.shape))
        np.save(OUT_DIR/f'{iv.replace("/", "_")}_{k}.npy', pm)
        human = pm > 0
        for d in dots:
            if d['iv'] == iv and d['k'] == k and d['c'] != 'none':
                miss = not (ms[0][d['y'], d['x']] or ms[1][d['y'], d['x']])
                cover['missed' if miss else 'owned'][0] += bool(human[d['y'], d['x']])
                cover['missed' if miss else 'owned'][1] += 1
        new = watershed_fill(img, ms, human)
        res[iv][str(k)] = {P[iv][i]: rle_encode(new[i]) for i in range(2)}
        print(iv, k, 'human px', int(human.sum()), 'src union', int((ms[0] | ms[1]).sum()), flush=True)
    print('Sapiens calls the dot human:', {g: f'{a_}/{b}' for g, (a_, b) in cover.items()})
    for iv, lm in res.items():
        d = ROOT/'autolabel/v1'/iv/'tangle'/(a.out + 'w')
        d.mkdir(parents=True, exist_ok=True)
        write_json(d/'tangle.json', {'variant': a.out + 'w', 'src': a.src, 'prim': P[iv], 'label_masks': lm,
                                     'note': 'dotted frames only (sapiens_parts.py experiment)'})


if __name__ == '__main__':
    main()
