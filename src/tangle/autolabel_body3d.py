"""Stage 5b: SAM-3D-Body (ViT-H, local) per primary actor on label frames at ~1 fps (runs in .venv-s3d).

Prompted with each actor's own box AND mask (use_mask=True), so the body model reconstructs the right person in
entangled frames. It predicts the full body, including occluded limbs. Stores 3D/2D keypoints, camera translation
and focal length per actor-frame (no meshes, to keep the dataset small).

Loader recipe from vlm-experiments/qwen3-vl-seg-decoder/labels-work/sam3d_canary.py: meta-device backbone, mmap
checkpoint, materialise straight onto the GPU. About 3 GB peak, so it cannot share the GPU with densify.
Output: autolabel/<run>/<trial>/<packet>/body3d.npz + body3d.json
"""
import argparse
import gc
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from autolabel_common import interval_dir, mask_box, rle_decode, write_json

from paths import S3D  # noqa: E402
STRIDE = 2  # every 2nd label frame: about 1 fps


def load_estimator():
    import torch
    sys.path.insert(0, str(S3D))
    from sam_3d_body.models.meta_arch import sam3d_body as arch
    from sam_3d_body.utils.config import get_config
    from sam_3d_body.utils.checkpoint import load_state_dict
    from sam_3d_body import SAM3DBodyEstimator
    original_backbone, original_linspace = arch.create_backbone, torch.linspace

    def meta_backbone(*args, **kwargs):
        def cpu_linspace(*a, **k):
            k['device'] = 'cpu'
            return original_linspace(*a, **k)
        torch.linspace = cpu_linspace
        try:
            with torch.device('meta'):
                return original_backbone(*args, **kwargs)
        finally:
            torch.linspace = original_linspace

    arch.create_backbone = meta_backbone
    cfg = get_config(str(S3D/'checkpoints/vith/model_config.yaml'))
    cfg.defrost()
    cfg.MODEL.MHR_HEAD.MHR_MODEL_PATH = str(S3D/'checkpoints/vith/assets/mhr_model.pt')
    cfg.freeze()
    model = arch.SAM3DBody(cfg)
    arch.create_backbone = original_backbone
    checkpoint = torch.load(S3D/'checkpoints/vith/model.ckpt', map_location='cpu', mmap=True, weights_only=False)
    state = checkpoint.get('state_dict', checkpoint)
    for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        if not tensor.is_meta:
            continue
        parent_name, leaf = name.rsplit('.', 1)
        parent = model.get_submodule(parent_name)
        value = state[name].to(device='cuda', dtype=tensor.dtype)
        if leaf in parent._parameters:
            parent._parameters[leaf] = torch.nn.Parameter(value, requires_grad=False)
        else:
            parent._buffers[leaf] = value
    load_state_dict(model, state, strict=False)
    del state, checkpoint
    gc.collect()
    model = model.to('cuda').eval()
    return SAM3DBodyEstimator(model, cfg, human_detector=None, human_segmentor=None, fov_estimator=None)


KEYS = ('pred_keypoints_3d', 'pred_keypoints_2d', 'pred_cam_t', 'focal_length')


def run(run_name, targets, force=False):
    import cv2
    import torch
    est = None
    for trial_id, packet_id in targets:
        base = interval_dir(run_name, trial_id, packet_id)
        if (base/'body3d.json').exists() and not force:
            continue
        if est is None:
            est = load_estimator()
        labels = json.loads((base/'labels.json').read_text())
        primary = {a['actor_id'] for a in labels['actors'] if a.get('primary')}
        started = time.time()
        rows, arrays = [], {k: [] for k in KEYS}
        for f in labels['frames'][::STRIDE]:
            insts = [i for i in f['instances'] if i['actor_id'] in primary and i['area'] > 1500]
            if not insts:
                continue
            img = cv2.cvtColor(cv2.imread(f['path']), cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            for inst in insts:
                mask = rle_decode(inst['mask_rle'])
                x1, y1, x2, y2 = mask_box(mask)
                pad = 0.1*max(x2 - x1, y2 - y1)
                box = np.array([[max(0, x1 - pad), max(0, y1 - pad), min(w, x2 + pad), min(h, y2 + pad)]], np.float32)
                try:
                    with torch.inference_mode():
                        out = est.process_one_image(img, bboxes=box, masks=mask[None].astype(np.uint8),
                                                    use_mask=True, inference_type='body')
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    continue
                if not out:
                    continue
                r = out[0]
                for k in KEYS:
                    arrays[k].append(np.asarray(r[k], np.float32))
                rows.append({'frame_index': f['frame_index'], 'time_s': f['time_s'], 'actor_id': inst['actor_id'],
                             'box_xyxy': box[0].round(1).tolist(), 'mask_source': inst['source']})
                est.batch = est.image_embeddings = est.output = None
            torch.cuda.empty_cache()
        if rows:
            np.savez_compressed(base/'body3d.npz', **{k: np.stack(v) for k, v in arrays.items()})
        write_json(base/'body3d.json', {'version': 1, 'stage': 'sam_3d_body_vith_mask_prompted', 'stride_label_frames': STRIDE,
                                        'keys': list(KEYS), 'records': rows, 'elapsed_s': round(time.time() - started, 1),
                                        'created_at': datetime.now(timezone.utc).isoformat()})
        print(f'{trial_id}/{packet_id}: {len(rows)} actor-frames {time.time() - started:.0f}s', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='v1')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('targets', nargs='+')
    a = ap.parse_args()
    run(a.run, [tuple(t.split('/')) for t in a.targets], a.force)
