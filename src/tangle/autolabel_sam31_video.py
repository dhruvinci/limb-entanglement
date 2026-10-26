"""Official SAM 3.1 multiplex video predictor, made to fit a 6 GiB GPU. Runs in .venv-sam31.

Working recipe (2026-10-04):
- Detector Linear/Conv weights in bf16, while norms, embeddings, buffers and the whole tracker stay fp32. Casting everything to bf16 kills the
  detector's text-prompt detections (0 people found); fp32 everywhere does not fit.
- A forward pre-hook casts each layer's floating inputs to that layer's weight dtype. Some sam3 code runs outside
  autocast.
- Batched grounding is off and postprocess batch is 1. The builder hard-codes 16-frame batches, which OOM.
- interactive_mode(): point-prompted tracking. Disables the text-detector track lifecycle and fixes upstream
  `_build_sam2_output`, which drops point-prompted objects on frames with no cached text-prompt outputs.
- init_state is shimmed to drop `offload_state_to_cpu`. sam3_base_predictor passes it, but the multiplex model's
  init_state does not accept it (upstream mismatch).
Peak about 4.8 GB on 1280x720 frames.
"""
import torch

from autolabel_common import SAM31_CKPT

MATMUL_TYPES = (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.ConvTranspose2d)
NORM_TYPES = (torch.nn.LayerNorm, torch.nn.GroupNorm)


def _cast_inputs(module, args, kwargs):
    dtype = module.weight.dtype
    cast = lambda x: x.to(dtype) if torch.is_tensor(x) and x.is_floating_point() and x.dtype != dtype else x
    return tuple(cast(a) for a in args), {k: cast(v) for k, v in kwargs.items()}


def build_predictor(max_num_objects=8):
    from sam3.model_builder import build_sam3_multiplex_video_predictor
    original_cuda = torch.nn.Module.cuda

    def mixed_cuda(self, *args, **kwargs):
        # only the detector's matmul weights go to bf16; the (small) tracker stays fp32
        for m in self.detector.modules():
            if isinstance(m, MATMUL_TYPES):
                m.to(dtype=torch.bfloat16)
        return self.to('cuda')

    torch.nn.Module.cuda = mixed_cuda
    try:
        predictor = build_sam3_multiplex_video_predictor(
            checkpoint_path=str(SAM31_CKPT), use_fa3=False, use_rope_real=False,
            async_loading_frames=False, max_num_objects=max_num_objects)
    finally:
        torch.nn.Module.cuda = original_cuda
    model = predictor.model
    for m in model.modules():
        if isinstance(m, MATMUL_TYPES + NORM_TYPES) and getattr(m, 'weight', None) is not None:
            m.register_forward_pre_hook(_cast_inputs, with_kwargs=True)
    model.use_batched_grounding = False
    model.postprocess_batch_size = 1
    init_state = model.init_state
    model.init_state = lambda **kw: init_state(**{k: v for k, v in kw.items() if k != 'offload_state_to_cpu'})
    return predictor


def interactive_mode(predictor):
    """Point-prompted tracking of given objects only: no text detections, so disable the detector-driven track
    lifecycle (hotstart suppression, unmatched-track removal, masklet confirmation, detector reconditioning)."""
    model = predictor.model
    model.hotstart_delay = 0
    model.hotstart_unmatch_thresh = 10**9
    model.hotstart_dup_thresh = 10**9
    model.masklet_confirmation_enable = False
    model.recondition_every_nth_frame = 0
    model.suppress_overlapping_based_on_recent_occlusion_threshold = 1.1

    def build_sam2_output(inference_state, frame_idx, refined_obj_id_to_mask=None):
        # Upstream returns {} when the frame has no cached text-prompt (VG) outputs, which silently drops every
        # point-prompted object on non-prompt frames. Start from the cache when present, else from nothing.
        out = dict(inference_state['cached_frame_outputs'].get(frame_idx, {}))
        out.update(refined_obj_id_to_mask or {})
        return out
    model._build_sam2_output = build_sam2_output

    def tracker_features_only(frame_idx, num_frames, input_batch, geometric_prompt, feature_cache, reverse, **kw):
        # Point tracking needs only the interactive/propagation neck features. Skip the text detector's grounding
        # decoder and segmentation head that upstream runs on every frame.
        from sam3.model.data_misc import NestedTensor
        images = input_batch.img_batch.tensors if hasattr(input_batch.img_batch, 'tensors') else input_batch.img_batch
        image = images[frame_idx].unsqueeze(0).to(dtype=torch.float32, device=model.device)
        feats = model.detector.backbone.forward_image(image, need_sam3_out=False, need_interactive_out=True,
                                                      need_propagation_out=True)
        cache = {}
        inter, prop = feats['interactive'], feats['sam2_backbone_out']
        idec, sdec = model.tracker.interactive_sam_mask_decoder, model.tracker.sam_mask_decoder
        ifpn = [idec.conv_s0(inter['backbone_fpn'][0].tensors), idec.conv_s1(inter['backbone_fpn'][1].tensors),
                inter['backbone_fpn'][2].tensors]
        cache['interactive'] = {'vision_features': ifpn[-1], 'vision_mask': None,
                                'vision_pos_enc': inter['vision_pos_enc'],
                                'backbone_fpn': [NestedTensor(x, None) for x in ifpn]}
        sfpn = [sdec.conv_s0(prop['backbone_fpn'][0].tensors), sdec.conv_s1(prop['backbone_fpn'][1].tensors),
                prop['backbone_fpn'][2].tensors]
        cache['sam2_backbone_out'] = {'vision_features': sfpn[-1], 'vision_mask': None,
                                      'vision_pos_enc': prop['vision_pos_enc'],
                                      'backbone_fpn': [NestedTensor(x, None) for x in sfpn]}
        feature_cache[frame_idx] = (images[frame_idx], cache)
        feature_cache.pop(frame_idx - 1 if not reverse else frame_idx + 1, None)
        return None, None
    model.run_backbone_and_detection = tracker_features_only
    return predictor


def start(predictor, frame_dir):
    return predictor.handle_request(dict(type='start_session', resource_path=str(frame_dir),
                                         offload_video_to_cpu=True))['session_id']


def add_points(predictor, session_id, frame_index, obj_id, points_rel, labels):
    return predictor.handle_request(dict(type='add_prompt', session_id=session_id, frame_index=frame_index,
                                         obj_id=obj_id, points=points_rel, point_labels=labels))


def propagate(predictor, session_id, direction='both', start_frame_idx=None, max_frames=None):
    """Yields (frame_index, {obj_id: bool mask})."""
    request = dict(type='propagate_in_video', session_id=session_id, propagation_direction=direction)
    if start_frame_idx is not None:
        request['start_frame_index'] = start_frame_idx
    if max_frames is not None:
        request['max_frame_num_to_track'] = max_frames
    for response in predictor.handle_stream_request(request):
        o = response['outputs']
        yield response['frame_index'], {int(i): m.astype(bool) for i, m in zip(o['out_obj_ids'], o['out_binary_masks'])}


def close(predictor, session_id):
    predictor.handle_request(dict(type='close_session', session_id=session_id))
