from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from .data import FrameSampleSpec


@dataclass(frozen=True)
class StaticArm:
    full_layers: tuple[int, ...] = ()
    pooled_layers: tuple[int, ...] = ()
    include_merged: bool = False
    use_rgb: bool = False

    @property
    def input_channels(self) -> dict[str, int]:
        channels = {f"layer_{layer:02d}": 1152 for layer in self.full_layers}
        channels.update(
            {f"pooled_{layer:02d}": 1152 for layer in self.pooled_layers}
        )
        if self.include_merged:
            channels["merged"] = 5120
        if self.use_rgb:
            channels["rgb"] = 3
        return channels


STATIC_ARMS = {
    "rgb": StaticArm(use_rgb=True),
    "l11": StaticArm(full_layers=(11,)),
    "p12": StaticArm(pooled_layers=(12,)),
    "merged": StaticArm(include_merged=True),
    "l11_merged": StaticArm(full_layers=(11,), include_merged=True),
    "l05_l11": StaticArm(full_layers=(5, 11)),
    "l11_l18": StaticArm(full_layers=(11, 18)),
    "l05_l11_l18": StaticArm(full_layers=(5, 11, 18)),
    "l05_l11_l18_l26": StaticArm(full_layers=(5, 11, 18, 26)),
    "l05_l11_l18_l26_merged": StaticArm(
        full_layers=(5, 11, 18, 26),
        include_merged=True,
    ),
}


@dataclass(frozen=True)
class ArmbarSplit:
    screen_train: tuple[FrameSampleSpec, ...]
    validation: tuple[FrameSampleSpec, ...]
    final_train: tuple[FrameSampleSpec, ...]
    test: tuple[FrameSampleSpec, ...]


@dataclass(frozen=True)
class LeaveOneClipOutFold:
    heldout_clip: str
    train_clips: tuple[str, ...]


@dataclass(frozen=True)
class NestedLeaveOneClipOutFold:
    heldout_clip: str
    validation_clip: str
    train_clips: tuple[str, ...]


def build_leave_one_clip_out_folds(
    clip_ids: Sequence[str],
) -> tuple[LeaveOneClipOutFold, ...]:
    normalized = tuple(sorted(str(clip_id) for clip_id in clip_ids))
    if len(normalized) < 3:
        raise ValueError("leave-one-clip-out evaluation requires at least three clips")
    if len(set(normalized)) != len(normalized):
        raise ValueError("clip IDs must be unique")
    if any(not clip_id for clip_id in normalized):
        raise ValueError("clip IDs cannot be empty")
    return tuple(
        LeaveOneClipOutFold(
            heldout_clip=heldout,
            train_clips=tuple(clip_id for clip_id in normalized if clip_id != heldout),
        )
        for heldout in normalized
    )


def build_nested_leave_one_clip_out_folds(
    clip_ids: Sequence[str],
) -> tuple[NestedLeaveOneClipOutFold, ...]:
    """Reserve whole clips for both model selection and unbiased outer evaluation."""

    normalized = tuple(sorted(str(clip_id) for clip_id in clip_ids))
    if len(normalized) < 4:
        raise ValueError("nested leave-one-clip-out evaluation requires at least four clips")
    if len(set(normalized)) != len(normalized):
        raise ValueError("clip IDs must be unique")
    if any(not clip_id for clip_id in normalized):
        raise ValueError("clip IDs cannot be empty")
    return tuple(
        NestedLeaveOneClipOutFold(
            heldout_clip=heldout,
            validation_clip=normalized[(index + 1) % len(normalized)],
            train_clips=tuple(
                clip_id
                for clip_id in normalized
                if clip_id not in {heldout, normalized[(index + 1) % len(normalized)]}
            ),
        )
        for index, heldout in enumerate(normalized)
    )


def split_armbar_specs(specs: Sequence[FrameSampleSpec]) -> ArmbarSplit:
    if not specs:
        raise ValueError("armbar split requires frame specifications")
    if any(spec.subset not in {"train", "test"} for spec in specs):
        raise ValueError("armbar subset must be train or test")
    if any(spec.screen_subset not in {"train", "validation", "test"} for spec in specs):
        raise ValueError("armbar screen subset must be train, validation, or test")
    screen_train = tuple(
        spec for spec in specs if spec.subset == "train" and spec.screen_subset == "train"
    )
    validation = tuple(
        spec for spec in specs if spec.subset == "train" and spec.screen_subset == "validation"
    )
    final_train = tuple(spec for spec in specs if spec.subset == "train")
    test = tuple(spec for spec in specs if spec.subset == "test")
    groups = (screen_train, validation, final_train, test)
    if any(not group for group in groups):
        raise ValueError("armbar manifest must populate screen-train, validation, final-train, and test")
    screen_ids = {spec.frame_index for spec in screen_train}
    heldout_ids = {spec.frame_index for spec in validation + test}
    if screen_ids & heldout_ids:
        raise ValueError("screen training overlaps a held-out split")
    return ArmbarSplit(screen_train, validation, final_train, test)
