from __future__ import annotations

import gc
import json
import queue
import threading
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from .artifacts import ArtifactStore
from .contracts import CampaignContract
from .launch_packet import MARKER_TOKEN
from .prompts import (
    SYSTEM_PROMPT,
    PromptRunSpec,
    build_actor_queries,
    build_format_repair_prompt,
    build_primary_prompt_runs,
    canonical_grounding_answer,
    grounding_condition,
)


def process_native_video(
    processor,
    serialized: str,
    frames: list[Image.Image],
    frame_indices: list[int],
    source_fps: int,
    total_frames: int,
    width: int,
    height: int,
    duration_seconds: float,
):
    """Use Qwen's native video preprocessor, including temporal patching and time metadata."""
    metadata = [{
        "total_num_frames": total_frames,
        "fps": float(source_fps),
        "width": width,
        "height": height,
        "duration": duration_seconds,
        "video_backend": "frozen-preselected-frames",
        "frames_indices": frame_indices,
    }]
    return processor(
        text=[serialized],
        videos=[frames],
        video_metadata=metadata,
        do_sample_frames=False,
        cap_pixels_per_frame=True,
        return_tensors="pt",
    )


def _output_tensor(output):
    if isinstance(output, tuple):
        return output[0]
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    return output


# KNOWN BUG (found 2026-10-13, see experiments/E03): raw vision tokens come in 2x2-grouped order, so the direct
# reshape below is not image order (14,080 of 14,400 locations misplaced on the 160x90 grid). Kept as run in
# August 2026 for the record; the fix is restore_grid() in src/qwen/final_cache/core.py.
def pool_spatial_tokens(hidden, grid_thw, output_hw: tuple[int, int] = (30, 30)):
    import torch.nn.functional as functional

    grid = grid_thw[0].tolist()
    temporal, height, width = (int(value) for value in grid)
    if hidden.shape[0] != temporal * height * width:
        raise ValueError(f"vision token/grid mismatch: {tuple(hidden.shape)} versus {grid}")
    maps = hidden.reshape(temporal, height, width, hidden.shape[-1]).permute(0, 3, 1, 2).float()
    pooled = functional.adaptive_avg_pool2d(maps, output_hw).permute(0, 2, 3, 1).to(hidden.dtype)
    return pooled.reshape(temporal * output_hw[0] * output_hw[1], hidden.shape[-1])


class BoundedTensorWriter:
    def __init__(self, store: ArtifactStore, max_pending: int = 2):
        self.store = store
        self.items: queue.Queue = queue.Queue(maxsize=max_pending)
        self.errors: list[BaseException] = []
        self.thread = threading.Thread(target=self._run, name="bounded-tensor-writer", daemon=True)
        self.thread.start()

    def submit(self, path: str, tensors: dict, metadata: dict[str, str] | None = None) -> None:
        if self.errors:
            raise RuntimeError("tensor writer failed") from self.errors[0]
        self.items.put((path, tensors, metadata))

    def _run(self) -> None:
        while True:
            item = self.items.get()
            try:
                if item is None:
                    return
                path, tensors, metadata = item
                self.store.write_tensors(path, tensors, metadata)
            except BaseException as exc:
                self.errors.append(exc)
            finally:
                self.items.task_done()

    def close(self) -> None:
        self.items.join()
        self.items.put(None)
        self.items.join()
        self.thread.join()
        if self.errors:
            raise RuntimeError("tensor writer failed") from self.errors[0]


@dataclass
class VisionResult:
    image_features: Any
    image_grid_thw: Any


class QwenRepresentationExtractor:
    """Exact-model extraction without requesting complete LM hidden-state outputs."""

    def __init__(self, model, processor, store: ArtifactStore, device: str = "cuda"):
        self.model = model
        self.processor = processor
        self.store = store
        self.device = device
        self.contract = CampaignContract.default()
        self._validate_structure()

    @classmethod
    def load_exact(cls, model_dir: Path, store: ArtifactStore, use_fast_kernels: bool, device: str = "cuda"):
        import torch
        from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

        config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
        config.use_cache = False

        model = AutoModelForImageTextToText.from_pretrained(
            model_dir,
            config=config,
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
            use_kernels=False,
        ).eval().to(device)
        if use_fast_kernels:
            # FLA functions are imported by Transformers when flash-linear-attention is present.
            # Hub kernelization remains disabled; FlashAttention-4 is never selected.
            pass
        processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
        return cls(model, processor, store, device=device)

    def _validate_structure(self) -> None:
        if len(self.model.model.visual.blocks) != self.contract.vision_layers:
            raise RuntimeError("exact model does not expose the frozen 27 vision blocks")
        if len(self.model.model.language_model.layers) != self.contract.language_layers:
            raise RuntimeError("exact model does not expose the frozen 64 language layers")

    def _device_inputs(self, values: dict[str, Any]) -> dict[str, Any]:
        return {key: value.to(self.device) if hasattr(value, "to") else value for key, value in values.items()}

    def extract_vision(
        self,
        frame_index: int,
        image_path: Path,
        *,
        artifact_prefix: str = "",
        full_layers: tuple[int, ...] | None = None,
    ) -> VisionResult:
        import torch

        image = Image.open(image_path).convert("RGB")
        processed = self.processor(images=[image], return_tensors="pt")
        inputs = self._device_inputs({key: value for key, value in processed.items() if key in {"pixel_values", "image_grid_thw"}})
        grid = inputs["image_grid_thw"].detach().cpu()
        writer = BoundedTensorWriter(self.store, max_pending=2)
        prefix = f"{artifact_prefix.strip('/')}/" if artifact_prefix else ""
        selected_full_layers = set(range(self.contract.vision_layers) if full_layers is None else full_layers)
        if not selected_full_layers.issubset(set(range(self.contract.vision_layers))):
            raise ValueError("selected full-resolution vision layer is outside the model")

        def hook_for(layer_index: int) -> Callable:
            full_path = f"{prefix}spatial/full/layer_{layer_index:02d}/frame_{frame_index:06d}.safetensors"
            pooled_path = f"{prefix}spatial/pooled/layer_{layer_index:02d}/frame_{frame_index:06d}.safetensors"

            def capture(_module, _args, output):
                hidden = _output_tensor(output).detach()
                if layer_index in selected_full_layers and not self.store.is_complete(full_path):
                    writer.submit(full_path, {"hidden": hidden.to("cpu"), "grid_thw": grid}, {"stage": "spatial_full"})
                if not self.store.is_complete(pooled_path):
                    pooled = pool_spatial_tokens(hidden, inputs["image_grid_thw"]).to("cpu")
                    writer.submit(
                        pooled_path,
                        {"hidden": pooled, "grid_thw": torch.tensor([[int(grid[0, 0]), 30, 30]], dtype=torch.int64)},
                        {"stage": "spatial_pooled"},
                    )
            return capture

        with ExitStack() as stack:
            for layer_index, block in enumerate(self.model.model.visual.blocks):
                handle = block.register_forward_hook(hook_for(layer_index))
                stack.callback(handle.remove)
            with torch.inference_mode():
                outputs = self.model.model.get_image_features(**inputs, return_dict=True)
        writer.close()
        image_features = tuple(tensor.detach() for tensor in outputs.pooler_output)
        merged_path = f"{prefix}merged-vision/frame_{frame_index:06d}.safetensors"
        if not self.store.is_complete(merged_path):
            self.store.write_tensors(
                merged_path,
                {"hidden": torch.cat(image_features).to("cpu"), "grid_thw": grid},
                {"stage": "merged_vision"},
            )
        return VisionResult(image_features=image_features, image_grid_thw=inputs["image_grid_thw"])

    def _run_language_with_features(self, processed: dict[str, Any], features: tuple, marker_position: int):
        import torch

        values = self._device_inputs(processed)
        input_ids = values["input_ids"]
        inputs_embeds = self.model.model.get_input_embeddings()(input_ids)
        image_embeds = torch.cat(features, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        image_mask, _ = self.model.model.get_placeholder_mask(input_ids, inputs_embeds, image_features=image_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        position_ids = self.model.model.compute_3d_position_ids(
            input_ids=input_ids,
            image_grid_thw=values.get("image_grid_thw"),
            video_grid_thw=None,
            inputs_embeds=inputs_embeds,
            attention_mask=values.get("attention_mask"),
            past_key_values=None,
            mm_token_type_ids=values.get("mm_token_type_ids"),
        )
        states = []

        def capture(_module, _args, output):
            hidden = _output_tensor(output)
            states.append(hidden[0, marker_position].detach().to("cpu"))

        with ExitStack() as stack:
            for layer in self.model.model.language_model.layers:
                handle = layer.register_forward_hook(capture)
                stack.callback(handle.remove)
            with torch.inference_mode():
                self.model.model.language_model(
                    input_ids=None,
                    inputs_embeds=inputs_embeds,
                    position_ids=position_ids,
                    attention_mask=values.get("attention_mask"),
                    use_cache=False,
                )
        if len(states) != self.contract.language_layers:
            raise RuntimeError("not every language marker hook fired")
        return torch.stack(states)

    def _run_language_with_video_features(self, processed: dict[str, Any], features: tuple, marker_position: int):
        import torch

        values = self._device_inputs(processed)
        input_ids = values["input_ids"]
        inputs_embeds = self.model.model.get_input_embeddings()(input_ids)
        video_embeds = torch.cat(features, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        _, video_mask = self.model.model.get_placeholder_mask(input_ids, inputs_embeds, video_features=video_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
        position_ids = self.model.model.compute_3d_position_ids(
            input_ids=input_ids,
            image_grid_thw=None,
            video_grid_thw=values.get("video_grid_thw"),
            inputs_embeds=inputs_embeds,
            attention_mask=values.get("attention_mask"),
            past_key_values=None,
            mm_token_type_ids=values.get("mm_token_type_ids"),
        )
        states = []

        def capture(_module, _args, output):
            states.append(_output_tensor(output)[0, marker_position].detach().to("cpu"))

        with ExitStack() as stack:
            for layer in self.model.model.language_model.layers:
                handle = layer.register_forward_hook(capture)
                stack.callback(handle.remove)
            with torch.inference_mode():
                self.model.model.language_model(
                    input_ids=None,
                    inputs_embeds=inputs_embeds,
                    position_ids=position_ids,
                    attention_mask=values.get("attention_mask"),
                    use_cache=False,
                )
        if len(states) != self.contract.language_layers:
            raise RuntimeError("not every temporal marker hook fired")
        return torch.stack(states)

    def extract_image_queries(self, frame_index: int, image_path: Path, prompt_records: list[dict], vision: VisionResult) -> None:
        image = Image.open(image_path).convert("RGB")
        marker_id = self.processor.tokenizer.convert_tokens_to_ids(MARKER_TOKEN)
        for prompt in prompt_records:
            path = f"semantic/image/frame_{frame_index:06d}/{prompt['prompt_id']}.safetensors"
            if self.store.is_complete(path):
                continue
            processed = self.processor(text=[prompt["serialized"]], images=[image], return_tensors="pt")
            positions = (processed["input_ids"][0] == marker_id).nonzero(as_tuple=False).flatten()
            if not len(positions):
                raise RuntimeError("processed query lost its marker token")
            states = self._run_language_with_features(dict(processed), vision.image_features, int(positions[-1]))
            self.store.write_tensors(
                path,
                {"marker_states": states},
                {"stage": "semantic_image", "prompt_id": prompt["prompt_id"], "marker_token_id": str(marker_id)},
            )
            del states, processed
            gc.collect()

    def _generate_grounding_tokens(self, processed, *, max_new_tokens: int) -> Any:
        import torch

        values = self._device_inputs(dict(processed))
        input_length = int(values["input_ids"].shape[1])
        with torch.inference_mode():
            generated = self.model.generate(
                **values,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                use_cache=True,
            )
        return generated[0, input_length:].detach().cpu()

    def _serialize_chat(self, messages: list[dict], thinking_mode: str) -> str:
        options: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": thinking_mode == "xhigh",
        }
        if thinking_mode == "xhigh":
            options["reasoning_effort"] = "xhigh"
        return self.processor.apply_chat_template(messages, **options)

    def extract_video_context(
        self,
        context: str,
        frame_indices: list[int],
        frames_dir: Path,
        *,
        frame_paths: list[Path] | None = None,
        source_fps: int | None = None,
        total_frames: int | None = None,
        width: int | None = None,
        height: int | None = None,
        duration_seconds: float | None = None,
        artifact_prefix: str = "",
        prompt_runs: list[PromptRunSpec] | None = None,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> None:
        import torch
        from safetensors.torch import load_file

        if not frame_indices:
            raise ValueError("temporal context cannot be empty")
        resolved_paths = frame_paths or [frames_dir / f"frame_{index + 1:06d}.jpg" for index in frame_indices]
        if len(resolved_paths) != len(frame_indices):
            raise ValueError("frame path count must match temporal frame index count")
        frames = [Image.open(path).convert("RGB") for path in resolved_paths]
        is_video = len(frame_indices) > 1
        modality = "native_video" if is_video else "single_image_baseline"
        prefix = f"{artifact_prefix.strip('/')}/" if artifact_prefix else ""
        source_fps = source_fps or self.contract.video.fps
        total_frames = total_frames or self.contract.video.frame_count
        width = width or self.contract.video.width
        height = height or self.contract.video.height
        duration_seconds = duration_seconds or self.contract.video.duration_seconds
        prompt_runs = list(prompt_runs or build_primary_prompt_runs())
        if not prompt_runs or len(set(prompt_runs)) != len(prompt_runs):
            raise ValueError("prompt run matrix must be non-empty and contain no duplicates")
        total_progress_steps = len(prompt_runs) * 3
        completed_progress_steps = 0

        def report_progress(detail: str) -> None:
            if progress_callback is not None:
                progress_callback(detail, completed_progress_steps, total_progress_steps)

        media = {"type": "video", "video": f"frozen-context:{context}"} if is_video else {
            "type": "image", "image": f"frozen-context:{context}",
        }

        def process(serialized: str):
            if is_video:
                return process_native_video(
                    self.processor, serialized, frames, frame_indices, source_fps,
                    total_frames, width, height, duration_seconds,
                )
            return self.processor(text=[serialized], images=frames, return_tensors="pt")

        answers: dict[PromptRunSpec, str] = {}
        answer_tokens_by_run: dict[PromptRunSpec, Any] = {}
        grounding_messages_by_run: dict[PromptRunSpec, list[dict]] = {}
        for run in prompt_runs:
            report_progress(f"{context}:grounding:{run.condition}:{run.thinking_mode}")
            condition = grounding_condition(run.condition)
            grounding_messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [media, {"type": "text", "text": condition["text"]}],
                },
            ]
            grounding_messages_by_run[run] = grounding_messages
            grounding_path = f"{prefix}grounding/{context}/{run.condition}/{run.thinking_mode}.safetensors"
            if self.store.is_complete(grounding_path):
                cached = load_file(self.store.root / grounding_path)
                canonical_tokens = cached["canonical_answer_token_ids"]
                final_tokens = canonical_tokens if canonical_tokens.numel() else cached["final_answer_token_ids"]
                answers[run] = self.processor.tokenizer.decode(final_tokens, skip_special_tokens=True).strip()
                answer_tokens_by_run[run] = final_tokens
                completed_progress_steps += 1
                continue

            grounding_serialized = self._serialize_chat(grounding_messages, run.thinking_mode)
            grounding_processed = process(grounding_serialized)
            initial_tokens = self._generate_grounding_tokens(
                grounding_processed,
                max_new_tokens=768 if run.thinking_mode == "xhigh" else 192,
            )
            initial_answer = self.processor.tokenizer.decode(initial_tokens, skip_special_tokens=True).strip()
            canonical = canonical_grounding_answer(initial_answer)
            repair_used = canonical is None
            repair_tokens = torch.empty((0,), dtype=torch.int64)
            repair_answer = ""
            if repair_used:
                repair_messages = [
                    *grounding_messages,
                    {"role": "assistant", "content": initial_answer},
                    {"role": "user", "content": build_format_repair_prompt()},
                ]
                repair_serialized = self._serialize_chat(repair_messages, "off")
                repair_processed = process(repair_serialized)
                repair_tokens = self._generate_grounding_tokens(repair_processed, max_new_tokens=192).to(torch.int64)
                repair_answer = self.processor.tokenizer.decode(repair_tokens, skip_special_tokens=True).strip()
                canonical = canonical_grounding_answer(repair_answer)
                del repair_processed
            contract_valid = canonical is not None
            final_answer = canonical if canonical is not None else (repair_answer or initial_answer)
            canonical_tokens = (
                self.processor.tokenizer(canonical, add_special_tokens=False, return_tensors="pt").input_ids[0].to(torch.int64)
                if canonical is not None else torch.empty((0,), dtype=torch.int64)
            )
            final_tokens = self.processor.tokenizer(
                final_answer, add_special_tokens=False, return_tensors="pt",
            ).input_ids[0].to(torch.int64)
            self.store.write_tensors(
                grounding_path,
                {
                    "initial_answer_token_ids": initial_tokens.to(torch.int64).clone(),
                    "repair_answer_token_ids": repair_tokens.clone(),
                    "final_answer_token_ids": final_tokens.clone(),
                    "canonical_answer_token_ids": canonical_tokens.clone(),
                    "frame_indices": torch.tensor(frame_indices, dtype=torch.int64),
                    "contract_valid": torch.tensor([contract_valid], dtype=torch.bool),
                    "repair_used": torch.tensor([repair_used], dtype=torch.bool),
                    "input_token_ids": grounding_processed["input_ids"].cpu(),
                },
                {
                    "stage": "self_grounding",
                    "context": context,
                    "condition": run.condition,
                    "thinking_mode": run.thinking_mode,
                    "modality": modality,
                    "initial_answer": initial_answer,
                    "repair_answer": repair_answer,
                    "canonical_answer": canonical or "",
                    "contract": "valid JSON with A1, A2, relation, and uncertainty; one format-only repair allowed",
                },
            )
            answers[run] = final_answer
            answer_tokens_by_run[run] = final_tokens
            del grounding_processed
            completed_progress_steps += 1

        features = None
        outputs = None
        marker_id = self.processor.tokenizer.convert_tokens_to_ids(MARKER_TOKEN)
        for run in prompt_runs:
            queries = {item["actor_id"]: item for item in build_actor_queries(MARKER_TOKEN, run.condition)}
            for actor in ("A1", "A2"):
                report_progress(f"{context}:semantic:{run.condition}:{run.thinking_mode}:{actor}")
                path = f"{prefix}semantic/video/{context}/{run.condition}/{run.thinking_mode}/{actor}.safetensors"
                if self.store.is_complete(path):
                    completed_progress_steps += 1
                    continue
                messages = [
                    *grounding_messages_by_run[run],
                    {"role": "assistant", "content": answers[run]},
                    {"role": "user", "content": queries[actor]["text"]},
                ]
                serialized = self._serialize_chat(messages, run.thinking_mode)
                processed = process(serialized)
                if features is None:
                    if is_video:
                        feature_inputs = self._device_inputs({
                            "pixel_values_videos": processed["pixel_values_videos"],
                            "video_grid_thw": processed["video_grid_thw"],
                        })
                        with torch.inference_mode():
                            outputs = self.model.model.get_video_features(**feature_inputs, return_dict=True)
                    else:
                        feature_inputs = self._device_inputs({
                            "pixel_values": processed["pixel_values"],
                            "image_grid_thw": processed["image_grid_thw"],
                        })
                        with torch.inference_mode():
                            outputs = self.model.model.get_image_features(**feature_inputs, return_dict=True)
                    features = tuple(tensor.detach() for tensor in outputs.pooler_output)
                positions = (processed["input_ids"][0] == marker_id).nonzero(as_tuple=False).flatten()
                if not len(positions):
                    raise RuntimeError("processed temporal query lost its marker token")
                run_language = self._run_language_with_video_features if is_video else self._run_language_with_features
                states = run_language(dict(processed), features, int(positions[-1]))
                tensors = {
                    "marker_states": states,
                    "frame_indices": torch.tensor(frame_indices, dtype=torch.int64),
                    "token_ids": processed["input_ids"].cpu(),
                    "grounding_answer_token_ids": answer_tokens_by_run[run].to(torch.int64),
                }
                if is_video:
                    tensors["video_grid_thw"] = processed["video_grid_thw"].cpu()
                self.store.write_tensors(
                    path,
                    tensors,
                    {
                        "stage": "semantic_video", "context": context, "actor": actor,
                        "condition": run.condition, "thinking_mode": run.thinking_mode,
                        "marker_token_id": str(marker_id), "modality": modality,
                        "identity_source": "Qwen self-grounding JSON",
                    },
                )
                del processed, states
                completed_progress_steps += 1
        report_progress(f"{context}:complete")
        for image in frames:
            image.close()
        if features is not None:
            del features, outputs
        gc.collect()


def write_run_manifest(run_root: Path, packet_manifest: dict, kernel_policy: dict) -> None:
    manifest = {
        "packet_digests": {key: packet_manifest[key] for key in ("input_digest", "prompt_digest", "label_digest")},
        "model_revision": CampaignContract.default().model_revision,
        "kernel_policy": kernel_policy,
        "use_cache": False,
        "dtype": "bfloat16",
        "batch_size": 1,
        "torch_compile": False,
    }
    path = run_root / "run-manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
