"""Core of the tangle tracker: the SAM 3.1 multiplex tracker driven directly (runs in .venv-sam31).

Why not autolabel_sam31_video / densify: those prompt with box corners + 3 interior points per anchor, track each
object without competition (non_overlap_masks_for_mem_enc=False) and return binary masks only. In a tangle the
prompts are ambiguous, a leak onto the partner is written into that object's memory and reinforced, and overlaps are
later settled by a fixed order. Here:
  - anchors are given as full masks (add_new_masks), all tracked objects in one session;
  - the pixel-level non-overlap constraint is applied before memory encoding (objects compete for pixels), optional;
  - per-object sigmoid probabilities are returned, so forward and backward passes can be fused by evidence.
Image features come from the detector backbone (the multiplex tracker has no backbone of its own), as in
autolabel_sam31_video.interactive_mode.
"""
import numpy as np
import torch

_STATE = {}
IOU = {}   # frame -> per-object predicted IoU of the last track() call (needs tracker.use_memory_selection = True)


def memory_selection(tracker, threshold=None, worst=False):
    """SAM 3.1's built-in memory selection (off in the multiplex build): non-conditioning frames enter the memory
    bank only when (object score x predicted IoU) > threshold, scored by the mean over objects or, with worst=True,
    by the worst object. threshold=None keeps every frame (selection on only to log the IoU predictions)."""
    import torch as _t
    tracker.use_memory_selection = True
    tracker.mf_threshold = -1e9 if threshold is None else threshold
    if worst:
        def fn(osl, iou):
            norm = _t.where(osl > 0, osl.sigmoid()*2 - 1, _t.zeros_like(osl))
            return (norm.flatten()*iou.flatten()).min()
        _STATE['mem_score_fn'] = fn
    else:
        _STATE.pop('mem_score_fn', None)


def load(non_overlap=True):
    """Build the SAM 3.1 multiplex predictor once; returns the inner tracker (Sam3VideoTrackingMultiplexDemo)."""
    if 'tracker' in _STATE:
        _STATE['tracker'].non_overlap_masks_for_mem_enc = non_overlap
        return _STATE['tracker']
    from sam3.model.data_misc import NestedTensor
    from autolabel_sam31_video import build_predictor
    predictor = build_predictor()
    model = predictor.model
    tracker = model.tracker.model  # the wrapper delegates attribute reads, but patches must go on the inner model

    def image_feature(self, inference_state, frame_idx, batch_size):
        image = inference_state['images'][frame_idx].cuda().float().unsqueeze(0)
        feats = model.detector.backbone.forward_image(image, need_sam3_out=False, need_interactive_out=True,
                                                      need_propagation_out=True)
        inter, prop = feats['interactive'], feats['sam2_backbone_out']
        idec, sdec = self.interactive_sam_mask_decoder, self.sam_mask_decoder
        ifpn = [idec.conv_s0(inter['backbone_fpn'][0].tensors), idec.conv_s1(inter['backbone_fpn'][1].tensors),
                inter['backbone_fpn'][2].tensors]
        sfpn = [sdec.conv_s0(prop['backbone_fpn'][0].tensors), sdec.conv_s1(prop['backbone_fpn'][1].tensors),
                prop['backbone_fpn'][2].tensors]
        cache = {'interactive': {'vision_features': ifpn[-1], 'vision_mask': None, 'vision_pos_enc': inter['vision_pos_enc'],
                                 'backbone_fpn': [NestedTensor(x, None) for x in ifpn]},
                 'sam2_backbone_out': {'vision_features': sfpn[-1], 'vision_mask': None,
                                       'vision_pos_enc': prop['vision_pos_enc'],
                                       'backbone_fpn': [NestedTensor(x, None) for x in sfpn]}}
        return image, self._prepare_backbone_features(cache)

    type(tracker)._get_image_feature = image_feature
    default_mem_score = type(tracker).cal_mem_score

    def mem_score(self, object_score_logits, iou_score):
        # log the decoder's per-object IoU prediction of every tracked frame (only computed when memory selection is
        # on); MEM_SCORE['fn'] may replace the frame's memory-selection score (default: SAM 3.1's mean over objects)
        _STATE['iou_log'].append(iou_score.flatten().float().cpu().numpy())
        fn = _STATE.get('mem_score_fn')
        return fn(object_score_logits, iou_score) if fn else default_mem_score(self, object_score_logits, iou_score)

    type(tracker).cal_mem_score = mem_score
    _STATE['iou_log'] = []
    tracker.non_overlap_masks_for_mem_enc = non_overlap
    _STATE.update(tracker=tracker, predictor=predictor)
    return tracker


class LazyFrames:
    """Span frames decoded on demand (normalised like sam3.model.io_utils), with a small cache. Holding a whole span
    at 1008x1008 fp16 costs 6 MB a frame (a 500-frame span filled the 14 GB of RAM)."""

    def __init__(self, paths, size, cache=48, flip=False, crop=None):
        from collections import OrderedDict
        self.paths, self.size, self.cache, self.n, self.flip, self.crop = paths, size, OrderedDict(), cache, flip, crop

    def flipped(self):
        return LazyFrames(self.paths, self.size, self.n, not self.flip, self.crop)

    def cropped(self, box):
        """The same frames cut to box = (x0, y0, x1, y1) in pixels before resizing to the model input (zoom)."""
        return LazyFrames(self.paths, self.size, self.n, self.flip, tuple(int(v) for v in box))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        if i in self.cache:
            self.cache.move_to_end(i)
            return self.cache[i]
        import torchvision.transforms.functional as TF
        from PIL import Image
        img = Image.open(self.paths[i]).convert('RGB')
        if self.crop:
            img = img.crop(self.crop)
        img = TF.resize(img, size=(self.size, self.size))  # as io_utils
        t = TF.to_tensor(img).half()  # same op order and precision as io_utils (fp16 normalisation)
        t -= 0.5
        t /= 0.5
        if self.flip:
            t = t.flip(-1)
        self.cache[i] = t
        if len(self.cache) > self.n:
            self.cache.popitem(last=False)
        return t


def load_frames(frame_dir, tracker, stride=1):
    """(LazyFrames, video height, video width) for a directory of %05d.jpg frames (every stride-th, plus the last)."""
    import cv2
    from pathlib import Path
    allp = sorted(Path(frame_dir).glob('*.jpg'))
    paths = allp[::stride] + ([allp[-1]] if (len(allp) - 1) % stride else [])
    h, w = cv2.imread(str(paths[0])).shape[:2]
    return LazyFrames(paths, tracker.image_size), h, w


KEEP_BACK = 32  # tracked frames kept in the state behind the current one (memory bank: 7 masks, 16 object pointers)


def prune(st, fi, reverse):
    """Drop non-conditioning frame outputs further than KEEP_BACK behind the current frame: each holds ~14 MB of GPU
    tensors (image features + memory), a 6 GB card ran out at ~300 frames."""
    dicts = [st['output_dict']] + list(st['output_dict_per_obj'].values())
    for d in dicts:
        nc = d['non_cond_frame_outputs']
        for j in [j for j in nc if (j > fi + KEEP_BACK if reverse else j < fi - KEEP_BACK)]:
            del nc[j]


@torch.inference_mode()
def track(tracker, frames, seeds, start, reverse, stop=None, out_scale=0.5, prompt='mask', flip=False, crop=None):
    """One pass. frames = load_frames(...) result; seeds = {frame_index: {obj_id: bool mask (H, W)}} (all frames of
    a pass must carry the same obj_ids); propagation starts at `start` and runs forward (or backward when reverse)
    to `stop` (inclusive; default: the end of the video in that direction).
    prompt='mask' uses add_new_masks; prompt='points' mimics densify (box corners + 3 interior points + negatives).
    flip=True tracks the mirrored video (seeds mirrored, outputs mirrored back): a second, nearly independent sample of
    the tracker, whose decisions in a tangle swing on input noise as small as fp16 rounding.
    Returns ({frame_index: uint8 prob [n_obj, h*s, w*s]}, obj_ids, {frame_index: object score logits})."""
    images, vh, vw = frames
    if crop is not None:   # zoom: track inside crop = (x0, y0, x1, y1), paste the outputs back into the full frame
        x0, y0, x1, y1 = (int(v) for v in crop)
        cs = {k: {o: (m[y0:y1, x0:x1] if isinstance(m, np.ndarray) else {'points': m['points'] - [x0, y0]}) for o, m in v.items()}
              for k, v in seeds.items()}
        probs, ids, scores = track(tracker, (images.cropped(crop), y1 - y0, x1 - x0), cs, start, reverse, stop, 1.0, prompt, flip)
        import cv2
        H, W = int(round(vh*out_scale)), int(round(vw*out_scale))
        bx = [int(round(v*out_scale)) for v in (x0, y0, x1, y1)]
        full = {}
        for fi, p in probs.items():
            canvas = np.zeros((p.shape[0], H, W), np.uint8)
            for c in range(p.shape[0]):
                canvas[c, bx[1]:bx[3], bx[0]:bx[2]] = cv2.resize(p[c], (bx[2] - bx[0], bx[3] - bx[1]), interpolation=cv2.INTER_AREA)
            full[fi] = canvas
        return full, ids, scores
    if flip:
        images = images.flipped()
        seeds = {k: {o: np.ascontiguousarray(m[:, ::-1]) for o, m in v.items()} for k, v in seeds.items()}
    st = tracker.init_state(video_height=vh, video_width=vw, num_frames=len(images), offload_video_to_cpu=True,
                            )
    st['images'] = images
    obj_ids = sorted(next(iter(seeds.values())))
    for k, masks in seeds.items():
        if prompt == 'mixed' and isinstance(next(iter(masks.values())), dict):   # skeleton anchor: positive points
            assert not flip
            for o in obj_ids:
                pts = masks[o]['points']/np.array([vw, vh], np.float32)
                tracker.add_new_points(inference_state=st, frame_idx=k, obj_id=o, points=torch.tensor(pts, dtype=torch.float32),
                                       labels=torch.ones(len(pts), dtype=torch.int32), clear_old_points=True, rel_coordinates=True)
        elif prompt in ('mask', 'mixed'):
            # a second seed frame would take the multiplex "add to existing state" path, which needs a tracked output
            # on that frame and raises; with random=False the state is a pure function of obj_ids, so rebuilding it
            # makes every seed frame an independent initial conditioning frame
            st['multiplex_state'] = None
            tracker.add_new_masks(st, k, obj_ids, torch.stack([torch.from_numpy(masks[o].astype(np.float32)) for o in obj_ids]))
        else:
            for o in obj_ids:
                pts, labels = point_prompt(masks, o, vh, vw)
                tracker.add_new_points(inference_state=st, frame_idx=k, obj_id=o, points=torch.tensor(pts, dtype=torch.float32),
                                       labels=torch.tensor(labels, dtype=torch.int32), clear_old_points=True, rel_coordinates=True)
    # consolidate the prompts and run the memory encoder on the seed frame (add_new_masks/add_new_points skip it;
    # the predictor API calls this before propagating, without it the tracker runs with no memory of the seeds)
    tracker.propagate_in_video_preflight(st, run_mem_encoder=True)
    n = len(images) - 1
    stop = (0 if reverse else n) if stop is None else stop
    probs, scores = {}, {}
    IOU.clear()
    for fi, ids, lo, hi, sc in tracker.propagate_in_video(st, start_frame_idx=start, max_frame_num_to_track=abs(stop - start),
                                                          reverse=reverse, tqdm_disable=True):
        p = torch.sigmoid(hi.float())
        if out_scale != 1:
            p = torch.nn.functional.interpolate(p, scale_factor=out_scale, mode='bilinear', align_corners=False)
        order = [list(ids).index(o) for o in obj_ids]
        p = p[order, 0]
        if flip:
            p = p.flip(-1)
        probs[fi] = (p.clamp(0, 1)*255).round().to(torch.uint8).cpu().numpy()
        scores[fi] = sc[order].float().flatten().cpu().numpy()
        if tracker.use_memory_selection and _STATE['iou_log']:
            IOU[fi] = _STATE['iou_log'][-1][order]
        _STATE['iou_log'].clear()
        prune(st, fi, reverse)
    del st
    torch.cuda.empty_cache()
    return probs, obj_ids, scores


def point_prompt(masks, o, h, w):
    """densify-style prompt: box corners (labels 2/3), interior points of own mask (1), partners' interior points (0)."""
    from autolabel_common import mask_box
    from autolabel_densify import interior_points
    m = masks[o]
    x1, y1, x2, y2 = mask_box(m)
    own = interior_points(m)
    neg = [interior_points(masks[p], 1)[0] for p in masks if p != o and interior_points(masks[p], 1)]
    pts = [[x1/w, y1/h], [x2/w, y2/h]] + [[x/w, y/h] for x, y in own + neg]
    return pts, [2, 3] + [1]*len(own) + [0]*len(neg)


def fuse(pf, pb, wf, wb, bg=0.5):
    """Evidence fusion of two passes for one frame. pf/pb: uint8 prob [n_obj, h, w] or None; wf/wb: pass weights
    (scalar, or one per object).
    Returns (labels [h, w] int: 0 = background, i+1 = object i; per-object agreement IoU between passes or None)."""
    have = [(p.astype(np.float32)/255, np.broadcast_to(np.asarray(w, np.float32), (p.shape[0],))[:, None, None])
            for p, w in ((pf, wf), (pb, wb)) if p is not None and np.any(np.asarray(w) > 0)]
    if not have:
        return None, None
    prob = sum(p*w for p, w in have)/np.maximum(sum(w for _, w in have), 1e-6)
    lab = np.where(prob.max(0) > bg, prob.argmax(0) + 1, 0)
    agree = None
    if pf is not None and pb is not None:
        lf = np.where(pf.max(0) > 127, pf.argmax(0) + 1, 0)
        lb = np.where(pb.max(0) > 127, pb.argmax(0) + 1, 0)
        agree = [float(((lf == i) & (lb == i)).sum()/max(1, ((lf == i) | (lb == i)).sum())) for i in range(1, pf.shape[0] + 1)]
    return lab, agree
