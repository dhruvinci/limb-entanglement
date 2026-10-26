from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path


QWEN_MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
SOURCE_VIDEO_SHA256 = "278f07b036faa79c02fb1fab65e68af96ce25a8062a7f60c3cbd96378b77f8ec"
BREADTH_FULL_LAYERS = (5, 11, 18, 26)


@dataclass(frozen=True)
class VideoContract:
    filename: str
    sha256: str
    frame_count: int
    fps: int
    width: int
    height: int
    duration_seconds: float


@dataclass(frozen=True)
class BreadthClipSpec:
    clip_id: str
    source_filename: str
    source_sha256: str
    start_seconds: float
    duration_seconds: float
    source_fps: int
    sample_fps: int
    width: int
    height: int
    content_tag: str

    def relative_timestamps(self) -> list[float]:
        count = round(self.duration_seconds * self.sample_fps)
        return [index / self.sample_fps for index in range(count)]

    def source_frame_indices(self) -> list[int]:
        start = round(self.start_seconds * self.source_fps)
        step = self.source_fps // self.sample_fps
        return [start + index * step for index in range(len(self.relative_timestamps()))]


def breadth_clip_specs() -> tuple[BreadthClipSpec, ...]:
    """Frozen high-contact windows selected before looking at Qwen outputs.

    The original four windows come from private footage, so their specs are not published. They are read from the
    JSON file named by `LIMB_BREADTH_SPECS` (a list of objects with the `BreadthClipSpec` fields); without it there
    are none.
    """
    path = os.environ.get("LIMB_BREADTH_SPECS")
    if not path:
        return ()
    return tuple(BreadthClipSpec(**row) for row in json.loads(Path(path).read_text()))


@dataclass(frozen=True)
class CampaignContract:
    model_id: str
    model_revision: str
    vision_layers: int
    language_layers: int
    vision_hidden_size: int
    language_hidden_size: int
    video: VideoContract
    actor_definitions: dict[str, str]
    anchor_frame_indices: tuple[int, ...]
    held_out_frame_indices: tuple[int, ...]
    random_seed: int

    @classmethod
    def default(cls) -> "CampaignContract":
        return cls(
            model_id="Qwen/Qwen3.8-27B",
            model_revision=QWEN_MODEL_REVISION,
            vision_layers=27,
            language_layers=64,
            vision_hidden_size=1152,
            language_hidden_size=5120,
            video=VideoContract(
                filename="7B02953C-4050-47F2-9468-9042C755F2C3.MP4",
                sha256=SOURCE_VIDEO_SHA256,
                frame_count=146,
                fps=24,
                width=1440,
                height=2560,
                duration_seconds=6.083333,
            ),
            actor_definitions={
                "A1": "first competitor identity assigned by Qwen from the visual input",
                "A2": "second competitor identity assigned by Qwen from the visual input",
            },
            anchor_frame_indices=(18, 36, 60, 84, 143),
            held_out_frame_indices=(84, 143),
            random_seed=380027,
        )

    def to_dict(self) -> dict:
        return asdict(self)


def _sample_indices(frame_count: int, fps: int, target_fps: int) -> list[int]:
    if frame_count < 2 or fps % target_fps:
        raise ValueError("frame_count must be >=2 and source fps divisible by target fps")
    step = fps // target_fps
    indices = list(range(0, frame_count, step))
    if indices[-1] != frame_count - 1:
        indices.append(frame_count - 1)
    return indices


def build_temporal_matrix(frame_count: int, fps: int, seed: int) -> dict[str, list[int]]:
    two = _sample_indices(frame_count, fps, 2)
    four = _sample_indices(frame_count, fps, 4)
    eight = _sample_indices(frame_count, fps, 8)
    shuffled = four.copy()
    random.Random(seed).shuffle(shuffled)
    return {
        "single_frame": [143],
        "five_frame": [18, 36, 60, 84, 143],
        "2fps": two,
        "4fps": four,
        "4fps_reversed": list(reversed(four)),
        "4fps_shuffled": shuffled,
        "8fps": eight,
    }


def expected_inventory(contract: CampaignContract) -> dict[str, object]:
    temporal = build_temporal_matrix(contract.video.frame_count, contract.video.fps, contract.random_seed)
    mandatory_temporal_contexts = ("single_frame", "five_frame", "2fps", "4fps", "4fps_reversed", "4fps_shuffled")
    breadth_frames = sum(len(clip.relative_timestamps()) for clip in breadth_clip_specs())
    breadth_contexts = len(breadth_clip_specs())
    total_frames = contract.video.frame_count + breadth_frames
    mandatory_contexts = len(mandatory_temporal_contexts) + breadth_contexts
    primary_conditions = 3
    actors = 2
    thinking_ablation_runs = 1
    mandatory_semantic_files = mandatory_contexts * primary_conditions * actors + thinking_ablation_runs * actors
    mandatory_grounding_files = mandatory_contexts * primary_conditions + thinking_ablation_runs
    optional_semantic_files = primary_conditions * actors
    optional_grounding_files = primary_conditions
    return {
        "grounding_audit": mandatory_grounding_files,
        "spatial_full": (contract.video.frame_count * contract.vision_layers) + (breadth_frames * len(BREADTH_FULL_LAYERS)),
        "spatial_pooled": total_frames * contract.vision_layers,
        "merged_vision": total_frames,
        "semantic_image": 0,
        "semantic_video": mandatory_semantic_files * contract.language_layers,
        "8fps": optional_semantic_files * contract.language_layers,
        "semantic_image_files": 0,
        "semantic_video_mandatory_files": mandatory_semantic_files,
        "semantic_video_8fps_files": optional_semantic_files,
        "grounding_answer_mandatory_files": mandatory_grounding_files,
        "grounding_answer_8fps_files": optional_grounding_files,
        "thinking_ablation": {
            "context": "4fps", "condition": "action_relational", "primary": "off", "comparison": "xhigh",
        },
        "breadth_clip_count": breadth_contexts,
        "breadth_frame_count": breadth_frames,
        "breadth_full_layers": list(BREADTH_FULL_LAYERS),
        "approximate_tensor_bytes": {
            "spatial_full": ((contract.video.frame_count * contract.vision_layers) + (breadth_frames * len(BREADTH_FULL_LAYERS))) * 14400 * contract.vision_hidden_size * 2,
            "spatial_pooled": total_frames * contract.vision_layers * 900 * contract.vision_hidden_size * 2,
            "merged_vision": total_frames * 3600 * contract.language_hidden_size * 2,
            "semantic_image": 0,
            "semantic_video_all": (mandatory_semantic_files + optional_semantic_files) * contract.language_layers * contract.language_hidden_size * 2,
        },
        "8fps_optional": True,
    }
