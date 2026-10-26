from __future__ import annotations

import math
import itertools
import random
from dataclasses import dataclass
from typing import Mapping


_REQUIRED_METRICS = {
    "macro_actor_iou",
    "contact_accuracy",
    "positive_contact_region_fraction",
    "background_stability",
    "contact_pixel_count",
    "contact_region_count",
}


def _validate_metrics(metrics: Mapping[str, float], *, name: str) -> None:
    missing = sorted(_REQUIRED_METRICS - set(metrics))
    if missing:
        raise ValueError(f"{name} is missing metrics: {missing}")
    bounded = (
        "macro_actor_iou",
        "contact_accuracy",
        "positive_contact_region_fraction",
        "background_stability",
    )
    for key in bounded:
        try:
            value = float(metrics[key])
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} metric {key} is not numeric") from error
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} metric {key} must be finite and in [0, 1]")
    for key in ("contact_pixel_count", "contact_region_count"):
        try:
            value = float(metrics[key])
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} metric {key} is not numeric") from error
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} metric {key} must be finite and non-negative")


def aggregate_clip_metrics(
    by_clip: Mapping[str, Mapping[str, float]],
) -> dict[str, float]:
    """Aggregate held-out folds without allowing large clips to dominate actor IoU."""

    if not by_clip:
        raise ValueError("clip aggregation requires at least one held-out result")
    for clip_id, metrics in by_clip.items():
        _validate_metrics(metrics, name=f"clip {clip_id}")
        if (
            float(metrics["contact_pixel_count"]) <= 0.0
            or float(metrics["contact_region_count"]) <= 0.0
        ):
            raise ValueError(
                f"clip {clip_id} must contain positive reviewed contact evidence"
            )
    values = list(by_clip.values())
    clip_count = len(values)
    contact_pixels = sum(float(item["contact_pixel_count"]) for item in values)
    contact_regions = sum(float(item["contact_region_count"]) for item in values)
    contact_correct = sum(
        float(item["contact_accuracy"]) * float(item["contact_pixel_count"])
        for item in values
    )
    positive_regions = sum(
        float(item["positive_contact_region_fraction"])
        * float(item["contact_region_count"])
        for item in values
    )
    return {
        "clip_count": float(clip_count),
        "macro_actor_iou": sum(float(item["macro_actor_iou"]) for item in values)
        / clip_count,
        "background_stability": sum(
            float(item["background_stability"]) for item in values
        )
        / clip_count,
        "contact_pixel_count": contact_pixels,
        "contact_accuracy": contact_correct / max(1.0, contact_pixels),
        "contact_region_count": contact_regions,
        "positive_contact_region_fraction": positive_regions
        / max(1.0, contact_regions),
    }


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_paired_clip_improvement(
    candidate_by_clip: Mapping[str, Mapping[str, float]],
    baseline_by_clip: Mapping[str, Mapping[str, float]],
    *,
    random_seed: int = 7,
) -> dict[str, object]:
    """Report clip-balanced paired deltas and a deterministic bootstrap interval."""

    if set(candidate_by_clip) != set(baseline_by_clip) or len(candidate_by_clip) < 2:
        raise ValueError("paired uncertainty requires the same two or more clips")
    clip_ids = tuple(sorted(candidate_by_clip))
    for clip_id in clip_ids:
        _validate_metrics(candidate_by_clip[clip_id], name=f"candidate clip {clip_id}")
        _validate_metrics(baseline_by_clip[clip_id], name=f"baseline clip {clip_id}")
    sample_count = len(clip_ids)
    if sample_count <= 6:
        resamples = itertools.product(range(sample_count), repeat=sample_count)
        resample_count = sample_count**sample_count
    else:
        generator = random.Random(random_seed)
        frozen = tuple(
            tuple(generator.randrange(sample_count) for _ in range(sample_count))
            for _ in range(10_000)
        )
        resamples = iter(frozen)
        resample_count = len(frozen)
    index_sets = tuple(resamples)
    output: dict[str, object] = {
        "clip_ids": list(clip_ids),
        "clip_count": sample_count,
        "bootstrap_resample_count": resample_count,
        "bootstrap_method": (
            "exact_paired_clip_resampling"
            if sample_count <= 6
            else "seeded_paired_clip_resampling"
        ),
    }
    for metric in ("macro_actor_iou", "contact_accuracy"):
        deltas = [
            float(candidate_by_clip[clip_id][metric])
            - float(baseline_by_clip[clip_id][metric])
            for clip_id in clip_ids
        ]
        mean = sum(deltas) / sample_count
        population_std = math.sqrt(
            sum((value - mean) ** 2 for value in deltas) / sample_count
        )
        bootstrap_means = [
            sum(deltas[index] for index in indices) / sample_count
            for indices in index_sets
        ]
        output[metric] = {
            "per_clip_delta": {
                clip_id: deltas[index] for index, clip_id in enumerate(clip_ids)
            },
            "mean_delta": mean,
            "population_std": population_std,
            "bootstrap_95_percent_interval": [
                _percentile(bootstrap_means, 0.025),
                _percentile(bootstrap_means, 0.975),
            ],
        }
    return output


@dataclass(frozen=True)
class NorthStarThresholds:
    minimum_macro_actor_iou: float = 0.60
    minimum_iou_improvement: float = 0.03
    minimum_contact_improvement: float = 0.10
    minimum_contact_accuracy: float = 0.70
    minimum_positive_contact_region_fraction: float = 0.75
    minimum_background_stability: float = 0.90
    minimum_real_over_shuffled_iou: float = 0.01
    minimum_real_over_random_iou: float = 0.01
    minimum_real_over_degenerate_iou: float = 0.03
    minimum_swap_flip_fraction: float = 0.75
    maximum_swap_background_delta: float = 0.01


@dataclass(frozen=True)
class NorthStarResult:
    passed: bool
    gates: dict[str, bool]
    evidence: dict[str, float | str]


def evaluate_north_star(
    *,
    candidate: Mapping[str, float],
    strongest_baseline: Mapping[str, float],
    semantic_controls: Mapping[str, Mapping[str, float]],
    swap_metrics: Mapping[str, float],
    thresholds: NorthStarThresholds = NorthStarThresholds(),
) -> NorthStarResult:
    """Evaluate every preregistered spatial and semantic gate without cherry-picking."""

    _validate_metrics(candidate, name="candidate")
    _validate_metrics(strongest_baseline, name="strongest_baseline")
    required_controls = {"real", "shuffled_clip", "random_matched", "zero", "mean"}
    missing_controls = sorted(required_controls - set(semantic_controls))
    if missing_controls:
        raise ValueError(f"semantic controls are missing: {missing_controls}")
    for name in required_controls:
        _validate_metrics(semantic_controls[name], name=f"semantic control {name}")
    for key in ("actor_prediction_flip_fraction", "background_probability_delta"):
        if key not in swap_metrics:
            raise ValueError(f"swap metrics are missing: {key}")
        try:
            value = float(swap_metrics[key])
        except (TypeError, ValueError) as error:
            raise ValueError(f"swap metric {key} is not numeric") from error
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"swap metric {key} must be finite and in [0, 1]")

    iou_gain = float(candidate["macro_actor_iou"]) - float(
        strongest_baseline["macro_actor_iou"]
    )
    contact_gain = float(candidate["contact_accuracy"]) - float(
        strongest_baseline["contact_accuracy"]
    )
    real_iou = float(semantic_controls["real"]["macro_actor_iou"])
    shuffled_iou = float(semantic_controls["shuffled_clip"]["macro_actor_iou"])
    random_iou = float(semantic_controls["random_matched"]["macro_actor_iou"])
    degenerate_iou = max(
        float(semantic_controls["zero"]["macro_actor_iou"]),
        float(semantic_controls["mean"]["macro_actor_iou"]),
    )
    improvement_passed = (
        iou_gain >= thresholds.minimum_iou_improvement
        or contact_gain >= thresholds.minimum_contact_improvement
    )
    gates = {
        "macro_actor_iou": float(candidate["macro_actor_iou"])
        >= thresholds.minimum_macro_actor_iou,
        "baseline_improvement": improvement_passed,
        "contact_accuracy": float(candidate["contact_accuracy"])
        >= thresholds.minimum_contact_accuracy,
        "positive_contact_regions": float(
            candidate["positive_contact_region_fraction"]
        )
        >= thresholds.minimum_positive_contact_region_fraction,
        "background_stability": float(candidate["background_stability"])
        >= thresholds.minimum_background_stability,
        "real_beats_shuffled_clip": (real_iou - shuffled_iou)
        >= thresholds.minimum_real_over_shuffled_iou,
        "real_beats_random_matched": (real_iou - random_iou)
        >= thresholds.minimum_real_over_random_iou,
        "zero_and_mean_fail": (real_iou - degenerate_iou)
        >= thresholds.minimum_real_over_degenerate_iou,
        "actor_query_swap": float(swap_metrics["actor_prediction_flip_fraction"])
        >= thresholds.minimum_swap_flip_fraction,
        "swap_background_stability": float(swap_metrics["background_probability_delta"])
        <= thresholds.maximum_swap_background_delta,
    }
    if iou_gain >= thresholds.minimum_iou_improvement:
        winning_metric = "macro_actor_iou"
    elif contact_gain >= thresholds.minimum_contact_improvement:
        winning_metric = "contact_accuracy"
    else:
        winning_metric = "none"
    evidence: dict[str, float | str] = {
        "macro_actor_iou": float(candidate["macro_actor_iou"]),
        "iou_improvement": iou_gain,
        "contact_accuracy": float(candidate["contact_accuracy"]),
        "contact_improvement": contact_gain,
        "positive_contact_region_fraction": float(
            candidate["positive_contact_region_fraction"]
        ),
        "background_stability": float(candidate["background_stability"]),
        "real_over_shuffled_clip_iou": real_iou - shuffled_iou,
        "real_over_random_iou": real_iou - random_iou,
        "real_over_degenerate_iou": real_iou - degenerate_iou,
        "actor_prediction_flip_fraction": float(
            swap_metrics["actor_prediction_flip_fraction"]
        ),
        "swap_background_probability_delta": float(
            swap_metrics["background_probability_delta"]
        ),
        "winning_improvement_metric": winning_metric,
    }
    return NorthStarResult(passed=all(gates.values()), gates=gates, evidence=evidence)
