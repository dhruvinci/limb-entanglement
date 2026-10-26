"""Stage 1: exhaustive local SAM3.1 'person' detection + instance masks on every packet frame.

Runs in .venv-sam31. No network or model-API calls. Output: autolabel/<run>/<trial>/<packet>/detections.json
"""
import argparse
import sys
import time
from datetime import datetime, timezone

from autolabel_common import SAM31_CKPT, interval_dir, mask_box, packet, rle_encode, sha256, write_json

PROMPT = 'person'
THRESHOLD = 0.3
THRESHOLD_DEFAULT = THRESHOLD


def load(threshold=THRESHOLD):
    global THRESHOLD_DEFAULT
    THRESHOLD_DEFAULT = threshold
    import torch
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    # Box->mask (inst interactivity) weights are absent from sam3.1_multiplex.pt; box prompts go through
    # the SAM3.1 tracker in autolabel_fuse.py instead.
    model = build_sam3_image_model(checkpoint_path=str(SAM31_CKPT), load_from_HF=False,
                                   enable_inst_interactivity=False)
    return model, Sam3Processor(model, confidence_threshold=threshold)


def detect_frame(processor, image):
    """Returns (detections, threshold used). Crowded frames that exhaust the 6 GiB GPU retry at threshold 0.5."""
    import torch
    for threshold in (processor.confidence_threshold, 0.5, 0.7):
        processor.confidence_threshold = threshold
        try:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                state = processor.set_image(image)
                out = processor.set_text_prompt(PROMPT, state)
            break
        except torch.OutOfMemoryError:
            state = out = None
            torch.cuda.empty_cache()
    else:
        raise RuntimeError('SAM3.1 detection out of GPU memory even at threshold 0.7')
    masks = out['masks'][:, 0].cpu().numpy()
    scores = out['scores'].float().cpu().numpy()
    dets = []
    for i in scores.argsort()[::-1]:
        box = mask_box(masks[i])
        if box is None:
            continue
        dets.append({'score': round(float(scores[i]), 4), 'box_xyxy': box,
                     'detector_box_xyxy': [round(float(v), 1) for v in out['boxes'][i].float().cpu().tolist()],
                     'area': int(masks[i].sum()), 'mask_rle': rle_encode(masks[i])})
    used, processor.confidence_threshold = processor.confidence_threshold, THRESHOLD_DEFAULT
    del state, out
    return dets, used


def run(run_name, targets, threshold=THRESHOLD, force=False):
    from PIL import Image
    model = processor = None
    for trial_id, packet_id in targets:
        out_path = interval_dir(run_name, trial_id, packet_id)/'detections.json'
        if out_path.exists() and not force:
            print('skip', out_path)
            continue
        if processor is None:
            model, processor = load(threshold)
        pkt, pkt_path = packet(trial_id, packet_id)
        started, frames = time.time(), []
        for frame in pkt['frames']:
            with Image.open(frame['path']) as im:
                image = im.convert('RGB')
            dets, used = detect_frame(processor, image)
            frames.append({'frame_id': frame['frame_id'], 'time_s': frame['time_s'],
                           'width': frame['width'], 'height': frame['height'], 'path': frame['path'],
                           'confidence_threshold': used, 'detections': dets})
        write_json(out_path, {
            'version': 1, 'stage': 'sam31_person_detection', 'trial_id': trial_id, 'packet_id': packet_id,
            'source_key': pkt['source_key'], 'time_range_s': pkt['time_range_s'],
            'packet_path': str(pkt_path), 'packet_sha256': sha256(pkt_path),
            'model': {'name': 'sam3.1-image (detector of sam3.1_multiplex.pt)', 'checkpoint': str(SAM31_CKPT),
                      'checkpoint_sha256_prefix': '0567debeec80ba4a', 'text_prompt': PROMPT,
                      'confidence_threshold': threshold, 'autocast': 'bfloat16'},
            'created_at': datetime.now(timezone.utc).isoformat(),
            'elapsed_s': round(time.time() - started, 1), 'frames': frames})
        n = sum(len(f['detections']) for f in frames)
        print(f'{trial_id}/{packet_id}: {len(frames)} frames, {n} detections, {time.time()-started:.0f}s', flush=True)


def parse_targets(values):
    return [tuple(v.split('/')) for v in values]


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='v1')
    ap.add_argument('--threshold', type=float, default=THRESHOLD)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('targets', nargs='+', help='trialNNN/intervalNN')
    a = ap.parse_args()
    run(a.run, parse_targets(a.targets), a.threshold, a.force)
    sys.exit(0)
