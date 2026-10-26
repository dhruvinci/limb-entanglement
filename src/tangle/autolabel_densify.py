"""Stage 4: densify labels from ~2 fps to ~12.5 fps with the SAM 3.1 video tracker (runs in .venv-sam31).

For each interval:
1. Decode native frames over the labelled span (no seeking, decoded PTS, like the evidence frames). Keep a ~12.5 fps
   subset that always includes the frame nearest each label frame.
2. Anchors are label frames where every tracked primary actor has a mask and the frame has no review flags
   (risk-flagged frames are allowed). Every third clean anchor is held out for validation; the rest become point prompts (interior points of
   the actor's own mask, with the other actors' interior points as negatives).
3. Track in chunks of about 6 s. Each chunk is its own session (bounded memory) and starts on a prompt anchor.
4. Report held-out IoU per actor (tracked vs Astra-anchored label), the gap to the nearest anchor, and camera cuts
   per dense frame.

Output: autolabel/<run>/<trial>/<packet>/dense/{frames/*.jpg, masks.npz, dense.json}
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from autolabel_common import interval_dir, mask_box, rle_decode, write_json

TARGET_FPS = 12.5
CHUNK_S = 3.0
TRACK_ROLES = {'athlete', 'instructor', 'referee', 'coach_or_corner'}


def decode_span(master, t0, t1, out_dir):
    """Decode every native frame with t0 <= t < t1; returns [(path, pts_time)]. Mirrors the evidence decode."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = ['ffmpeg', '-hide_banner', '-nostdin', '-threads', '2', '-copyts', '-i', str(master), '-map', '0:v:0',
           '-vf', f"select='gte(t,{t0:.6f})*lt(t,{t1:.6f})',showinfo", '-fps_mode', 'vfr', '-q:v', '3',
           str(out_dir/'%06d.jpg')]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    times = [float(m) for m in re.findall(r'pts_time:([0-9.]+)', proc.stderr)]
    files = sorted(out_dir.glob('*.jpg'))
    assert len(files) == len(times), f'decoded {len(files)} frames but {len(times)} showinfo times'
    return list(zip(files, times))


def select_dense(decoded, label_times, fps=TARGET_FPS):
    """~fps subset that contains the native frame nearest each label time. Returns (indices, label->dense index)."""
    times = np.array([t for _, t in decoded])
    must = {int(np.argmin(np.abs(times - lt))): lt for lt in label_times}
    keep, last = [], -1e9
    for i, t in enumerate(times):
        if i in must or t - last >= 1/fps - 1e-6:
            keep.append(i)
            last = t
    pos = {i: k for k, i in enumerate(keep)}
    return keep, {lt: pos[i] for i, lt in must.items()}


def interior_points(mask, k=3):
    from autolabel_astra import anchor
    ys, xs = np.nonzero(mask[::4, ::4])
    ys, xs = ys*4, xs*4
    c = anchor(mask)
    pts = [c]
    for _ in range(k - 1):
        d = np.min([(xs - p[0])**2 + (ys - p[1])**2 for p in pts], axis=0)
        j = int(np.argmax(d))
        pts.append((int(xs[j]), int(ys[j])))
    inner = [(int(0.7*x + 0.3*c[0]), int(0.7*y + 0.3*c[1])) for x, y in pts[1:]]
    return [p for p in [c] + inner if mask[p[1], p[0]]]


def shot_cuts(paths, threshold=0.5):
    """Indices i where frame i starts a new shot (grey-histogram distance to frame i-1)."""
    from PIL import Image
    hists, cuts = [], []
    for p in paths:
        with Image.open(p) as im:
            h = np.asarray(im.convert('L').resize((160, 90)).histogram(), dtype=float)
        hists.append(h/h.sum())
    for i in range(1, len(hists)):
        if 0.5*np.abs(hists[i] - hists[i - 1]).sum() > threshold:
            cuts.append(i)
    return cuts


def iou(a, b):
    return float((a & b).sum()/max(1, (a | b).sum()))


def track_chunk(predictor, paths, a, b, seeds, obj_ids, h, w, work, direction, start_k):
    """One bounded session over dense frames a..b (inclusive), seeded by {dense_index: {actor: mask}}.
    Returns {dense_index: {obj_id: mask}}."""
    from autolabel_sam31_video import start, add_points, propagate, close
    out = {}
    with tempfile.TemporaryDirectory(dir=work) as wd:
        for j, k in enumerate(range(a, b + 1)):
            os.symlink(paths[k], Path(wd)/f'{j:05d}.jpg')
        sid = start(predictor, wd)
        try:
            for k, masks in seeds.items():
                for actor, m in masks.items():
                    own = interior_points(m)
                    neg = [interior_points(o, 1)[0] for o_actor, o in masks.items()
                           if o_actor != actor and interior_points(o, 1)]
                    x1, y1, x2, y2 = mask_box(m)
                    # box corners (labels 2/3) pin the whole-body extent; points alone pick garments/parts
                    add_points(predictor, sid, k - a, obj_ids[actor],
                               [[x1/w, y1/h], [x2/w, y2/h]] + [[x/w, y/h] for x, y in own + neg],
                               [2, 3] + [1]*len(own) + [0]*len(neg))
            for j, masks in propagate(predictor, sid, direction=direction, start_frame_idx=start_k - a):
                out[a + j] = masks
        finally:
            close(predictor, sid)
    return out


def densify(predictor, labels, master, work):
    import torch
    frames = labels['frames']
    primary = {a['actor_id'] for a in labels['actors'] if a.get('primary')}
    tracked = sorted(primary | {a['actor_id'] for a in labels['actors'] if a['role'] == 'referee'})[:4]
    obj_ids = {a: i + 1 for i, a in enumerate(tracked)}
    label_times = [f['time_s'] for f in frames]
    decoded = decode_span(master, label_times[0] - 0.02, label_times[-1] + 0.02, work/'native')
    keep, label_pos = select_dense(decoded, label_times)
    dense_dir = work/'frames'
    dense_dir.mkdir(parents=True, exist_ok=True)
    dense_times = []
    for k, i in enumerate(keep):
        shutil.move(str(decoded[i][0]), dense_dir/f'{k:05d}.jpg')
        dense_times.append(decoded[i][1])
    shutil.rmtree(work/'native')
    paths = sorted(dense_dir.glob('*.jpg'))
    h, w = frames[0]['height'], frames[0]['width']

    anchors = {}
    for f in frames:
        masks = {i['actor_id']: rle_decode(i['mask_rle']) for i in f['instances'] if i['actor_id'] in tracked and i['area'] > 400}
        # risk-flagged frames (merged detection split by Astra boxes) are allowed: in grappling they are most frames,
        # and with clean-only anchors the tracker loses the bottom player. Review-flagged frames are excluded.
        clean = not f['review_flags'] and primary <= set(masks)
        if clean:
            anchors[label_pos[f['time_s']]] = masks
    order = sorted(anchors)
    held = set(order[2::3])
    prompts = [k for k in order if k not in held]
    cuts = shot_cuts(paths)

    out, failed, carried = {}, [], set()
    inv_ids = {v: k for k, v in obj_ids.items()}
    chunk = int(CHUNK_S*TARGET_FPS)
    last = len(paths) - 1

    def as_actor_masks(tracked):
        return {inv_ids[o]: m for o, m in tracked.items() if o in inv_ids and m.sum() >= 400}

    def run_segment(a, b, direction, start_k):
        seeds = {p: anchors[p] for p in prompts if a <= p <= b}
        if start_k not in seeds:  # chained: seed from our own tracked masks at the segment start
            seeds[start_k] = as_actor_masks(out.get(start_k, {}))
            if not seeds[start_k]:
                return False
            carried.add(start_k)
        result = track_chunk(predictor, paths, a, b, seeds, obj_ids, h, w, work, direction, start_k)
        out.update(result)
        return True

    if prompts:
        # forward from the first real anchor to the end, in chunks; each chunk ends at a real anchor when one is in reach
        cur = prompts[0]
        while cur < last:
            end = min(last, cur + chunk)
            reach = [p for p in prompts if cur < p <= end]
            if reach and end < last:
                end = reach[-1]
            while True:
                try:
                    ok = run_segment(cur, end, 'forward', cur)
                    break
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if end - cur <= 8:
                        ok = False
                        break
                    end = cur + (end - cur)//2
            torch.cuda.empty_cache()
            if not ok:
                failed.append([cur, end])
            cur = end
        # backward from the first real anchor to frame 0, chained
        cur = prompts[0]
        while cur > 0:
            begin = max(0, cur - chunk)
            try:
                ok = run_segment(begin, cur, 'backward', cur)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                ok = False
            torch.cuda.empty_cache()
            if not ok:
                failed.append([begin, cur])
                break
            cur = begin

    inv = {v: k for k, v in obj_ids.items()}
    validation = []
    for k in sorted(held):
        got = out.get(k, {})
        validation.append({'dense_index': k, 'time_s': dense_times[k],
                           'iou': {a: round(iou(got.get(obj_ids[a], np.zeros((h, w), bool)), m), 3)
                                   for a, m in anchors[k].items()}})
    records, arrays = [], {}
    for k in range(len(paths)):
        nearest = min(prompts, key=lambda p: abs(p - k)) if prompts else None
        gap = None if nearest is None else abs(dense_times[nearest] - dense_times[k])
        cut_since_anchor = nearest is not None and any(min(k, nearest) < c <= max(k, nearest) for c in cuts)
        actors = {}
        for oid, m in out.get(k, {}).items():
            if oid in inv and m.any():
                arrays[f'{k}_{inv[oid]}'] = np.packbits(m)
                actors[inv[oid]] = int(m.sum())
        records.append({'dense_index': k, 'time_s': round(dense_times[k], 4), 'is_label_frame': k in label_pos.values(),
                        'is_prompt_anchor': k in prompts, 'is_heldout_anchor': k in held,
                        'gap_to_prompt_s': None if gap is None else round(gap, 3),
                        'shot_cut_since_anchor': cut_since_anchor, 'actor_areas': actors})
    np.savez_compressed(work/'masks.npz', **arrays)
    return {'tracked_actors': tracked, 'obj_ids': obj_ids, 'shape': [h, w], 'target_fps': TARGET_FPS,
            'dense_frames': len(paths), 'label_frames': len(frames), 'clean_anchors': len(order),
            'prompt_anchors': len(prompts), 'heldout_anchors': len(held), 'shot_cuts': cuts,
            'failed_chunks': failed, 'carried_seed_frames': sorted(carried),
            'validation': validation, 'frames': records}


def run(run_name, targets, force=False):
    from autolabel_sam31_video import build_predictor, interactive_mode
    import torch
    predictor = None
    for trial_id, packet_id in targets:
        base = interval_dir(run_name, trial_id, packet_id)
        work = base/'dense'
        if (work/'dense.json').exists() and not force:
            continue
        if work.exists():
            shutil.rmtree(work)
        work.mkdir(parents=True)
        if predictor is None:
            predictor = interactive_mode(build_predictor())
        labels = json.loads((base/'labels.json').read_text())
        det = json.loads((base/'detections.json').read_text())
        master = Path(json.loads((Path(det['packet_path']).parents[1]/'index.json').read_text())['master_path'])
        started = time.time()
        torch.cuda.reset_peak_memory_stats()
        result = densify(predictor, labels, master, work)
        result |= {'version': 1, 'stage': 'sam31_dense_tracking', 'trial_id': trial_id, 'packet_id': packet_id,
                   'master_path': str(master), 'elapsed_s': round(time.time() - started, 1),
                   'peak_vram_gb': round(torch.cuda.max_memory_allocated()/1e9, 2),
                   'created_at': datetime.now(timezone.utc).isoformat()}
        write_json(work/'dense.json', result)
        v = [x for r in result['validation'] for x in r['iou'].values()]
        print(f"{trial_id}/{packet_id}: {result['dense_frames']} dense frames, {result['prompt_anchors']} prompts, "
              f"held-out IoU median {np.median(v) if v else None}, {result['elapsed_s']}s, peak {result['peak_vram_gb']} GB", flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='v1')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('targets', nargs='+', help='trialNNN/intervalNN')
    a = ap.parse_args()
    run(a.run, [tuple(t.split('/')) for t in a.targets], a.force)
