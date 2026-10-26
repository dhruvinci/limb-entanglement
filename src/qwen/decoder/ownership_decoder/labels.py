from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter


LABEL_BACKGROUND = 0
LABEL_A1 = 1
LABEL_A2 = 2
LABEL_IGNORE = 255
_ALLOWED_LABELS = {LABEL_BACKGROUND, LABEL_A1, LABEL_A2, LABEL_IGNORE}


def _validate_source_masks(*masks: np.ndarray) -> tuple[np.ndarray, ...]:
    converted = tuple(np.asarray(mask, dtype=bool) for mask in masks)
    if not converted or converted[0].ndim != 2:
        raise ValueError("ownership source masks must be two-dimensional")
    if any(mask.ndim != 2 or mask.shape != converted[0].shape for mask in converted):
        raise ValueError("ownership source masks must have one identical shape")
    return converted


def _erode(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius < 1 or radius % 2 == 0:
        raise ValueError("erosion radius must be a positive odd integer")
    if radius == 1:
        return mask.copy()
    image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    return np.asarray(image.filter(ImageFilter.MinFilter(radius))) > 0


def _resize_fraction(mask: np.ndarray, output_hw: tuple[int, int]) -> np.ndarray:
    output_height, output_width = output_hw
    if output_height < 1 or output_width < 1:
        raise ValueError("output dimensions must be positive")
    image = Image.fromarray(mask.astype(np.float32), mode="F")
    return np.asarray(
        image.resize((output_width, output_height), Image.Resampling.BOX),
        dtype=np.float32,
    )


def build_agreement_ownership_labels(
    sam3_a1: np.ndarray,
    sam3_a2: np.ndarray,
    sam31_a1: np.ndarray,
    sam31_a2: np.ndarray,
    *,
    possible_foreground: np.ndarray | None = None,
    output_hw: tuple[int, int],
    erosion_radius: int = 7,
    actor_min_fraction: float = 0.70,
    competing_max_fraction: float = 0.05,
    background_max_fraction: float = 0.02,
) -> np.ndarray:
    """Create conservative A1/A2/background labels from independent backend agreement.

    Actor labels require both SAM versions to occupy most of a target cell after
    erosion, while both versions of the competing actor must be absent. Locations
    inside the guarded foreground region, disagreements, and boundaries remain 255.
    """

    s3_a1, s3_a2, s31_a1, s31_a2 = _validate_source_masks(
        sam3_a1, sam3_a2, sam31_a1, sam31_a2
    )
    if not 0.0 <= competing_max_fraction <= actor_min_fraction <= 1.0:
        raise ValueError("actor thresholds must satisfy 0 <= competing <= actor <= 1")
    if not 0.0 <= background_max_fraction <= 1.0:
        raise ValueError("background threshold must be in [0, 1]")

    if possible_foreground is None:
        possible = s3_a1 | s3_a2 | s31_a1 | s31_a2
    else:
        possible = np.asarray(possible_foreground, dtype=bool)
        if possible.shape != s3_a1.shape:
            raise ValueError("possible_foreground must match source mask shape")

    eroded = tuple(
        _resize_fraction(_erode(mask, erosion_radius), output_hw)
        for mask in (s3_a1, s3_a2, s31_a1, s31_a2)
    )
    raw = tuple(
        _resize_fraction(mask, output_hw)
        for mask in (s3_a1, s3_a2, s31_a1, s31_a2)
    )
    possible_fraction = _resize_fraction(possible, output_hw)

    labels = np.full(output_hw, LABEL_IGNORE, dtype=np.uint8)
    background = possible_fraction <= background_max_fraction
    for fraction in raw:
        background &= fraction <= background_max_fraction
    labels[background] = LABEL_BACKGROUND

    e3_a1, e3_a2, e31_a1, e31_a2 = eroded
    a1_core = (
        (e3_a1 >= actor_min_fraction)
        & (e31_a1 >= actor_min_fraction)
        & (e3_a2 <= competing_max_fraction)
        & (e31_a2 <= competing_max_fraction)
    )
    a2_core = (
        (e3_a2 >= actor_min_fraction)
        & (e31_a2 >= actor_min_fraction)
        & (e3_a1 <= competing_max_fraction)
        & (e31_a1 <= competing_max_fraction)
    )
    conflict = a1_core & a2_core
    labels[a1_core & ~conflict] = LABEL_A1
    labels[a2_core & ~conflict] = LABEL_A2
    return labels


def apply_reviewed_contact_ownership(
    labels: np.ndarray,
    contact_owner: np.ndarray,
    *,
    reviewed: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply explicit per-cell contact ownership after a human review gate."""

    if not reviewed:
        raise ValueError("contact ownership cannot become truth until it is explicitly reviewed")
    values = np.asarray(labels, dtype=np.uint8)
    owners = np.asarray(contact_owner, dtype=np.uint8)
    if values.ndim != 2 or owners.shape != values.shape:
        raise ValueError("contact_owner must match the two-dimensional label grid")
    unknown = set(np.unique(owners).tolist()) - {0, LABEL_A1, LABEL_A2, LABEL_IGNORE}
    if unknown:
        raise ValueError(f"contact_owner contains unsupported values: {sorted(unknown)}")
    corrected = values.copy()
    contact = (owners == LABEL_A1) | (owners == LABEL_A2)
    corrected[owners == LABEL_A1] = LABEL_A1
    corrected[owners == LABEL_A2] = LABEL_A2
    validate_label_ready(corrected, contact, require_both_actors=False)
    return corrected, contact


def validate_label_ready(
    labels: np.ndarray,
    contact: np.ndarray,
    *,
    require_both_actors: bool,
) -> None:
    values = np.asarray(labels)
    contact_values = np.asarray(contact, dtype=bool)
    if values.ndim != 2 or contact_values.shape != values.shape:
        raise ValueError("contact mask must match the two-dimensional label grid")
    unknown = set(np.unique(values).tolist()) - _ALLOWED_LABELS
    if unknown:
        raise ValueError(f"ownership labels contain unsupported values: {sorted(unknown)}")
    if np.any(contact_values & ~np.isin(values, [LABEL_A1, LABEL_A2])):
        raise ValueError("every contact pixel must have A1 or A2 ownership truth")
    if require_both_actors and (
        not np.any(values == LABEL_A1) or not np.any(values == LABEL_A2)
    ):
        raise ValueError("a ready frame must contain truth for both actors")


def label_summary(
    labels: np.ndarray,
    contact: np.ndarray,
    *,
    contact_reviewed: bool,
) -> dict:
    values = np.asarray(labels)
    contact_values = np.asarray(contact, dtype=bool)
    validate_label_ready(values, contact_values, require_both_actors=False)
    counts = {
        str(label): int(np.count_nonzero(values == label))
        for label in (LABEL_BACKGROUND, LABEL_A1, LABEL_A2, LABEL_IGNORE)
    }
    return {
        "pixel_counts": counts,
        "labeled_fraction": float(np.count_nonzero(values != LABEL_IGNORE) / values.size),
        "contact_patch_count": int(np.count_nonzero(contact_values)),
        "contact_reviewed": bool(contact_reviewed),
    }
