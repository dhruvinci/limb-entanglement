from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from itertools import combinations
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence

import torch
from torch.nn import functional as F

from .cache import CacheContractError, _load_actor_states
from .actor_alignment import load_alignment_orders


_DELTA_TARGETS = {
    "action_delta": "action_relational",
    "contact_delta": "contact_ownership",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _load_pair_layers(
    root: Path,
    *,
    clip_id: str,
    condition: str,
    context: str,
    thinking_mode: str,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    condition_root = root / clip_id / "semantic" / "video" / context / condition / thinking_mode
    values = []
    records = []
    for actor in ("A1", "A2"):
        path = condition_root / f"{actor}.safetensors"
        states, actor_metadata = _load_actor_states(path, actor)
        if (
            actor_metadata.get("condition") != condition
            or actor_metadata.get("context") != context
            or actor_metadata.get("thinking_mode") != thinking_mode
        ):
            raise CacheContractError(f"breadth semantic metadata mismatch: {path}")
        values.append(states.float())
        records.append(
            {
                "clip_id": clip_id,
                "condition": condition,
                "actor": actor,
                "path": path.relative_to(root).as_posix(),
                "sha256": _sha256(path),
            }
        )
    if values[0].shape != values[1].shape:
        raise CacheContractError(f"breadth actor state shapes differ: {condition_root}")
    return values[0], values[1], records


def _layer_metrics(role_directions: torch.Tensor) -> list[dict[str, Any]]:
    if role_directions.ndim != 3 or role_directions.shape[0] < 3:
        raise ValueError("role directions must have shape [clips,layers,features] for at least 3 clips")
    directions = F.normalize(role_directions.float(), dim=-1)
    layer_records = []
    clip_count, layer_count, _ = directions.shape
    for layer in range(layer_count):
        current = directions[:, layer]
        pairwise = [
            float(torch.dot(current[left], current[right]))
            for left, right in combinations(range(clip_count), 2)
        ]
        heldout_margins = []
        for heldout in range(clip_count):
            training = torch.stack(
                [current[index] for index in range(clip_count) if index != heldout]
            ).mean(dim=0)
            training = F.normalize(training, dim=0)
            heldout_margins.append(float(torch.dot(training, current[heldout])))
        layer_records.append(
            {
                "layer": layer,
                "leave_one_clip_out_accuracy": (
                    sum(value > 0.0 for value in heldout_margins) / clip_count
                ),
                "leave_one_clip_out_mean_margin": fmean(heldout_margins),
                "leave_one_clip_out_margins": heldout_margins,
                "mean_pairwise_cosine": fmean(pairwise),
                "minimum_pairwise_cosine": min(pairwise),
            }
        )
    return layer_records


def audit_breadth_condition_deltas(
    cache_root: str | Path,
    output_path: str | Path,
    *,
    top_actor_by_clip: Mapping[str, str],
    actor_alignment_manifest: str | Path | None = None,
    reference_packet_sha256: str | None = None,
    target_conditions: Sequence[str] | None = None,
    role_notes_by_clip: Mapping[str, str] | None = None,
    context: str = "4fps",
    thinking_mode: str = "off",
) -> dict[str, Any]:
    """Audit condition deltas in reviewed physical-actor order using caller-supplied roles.

    This is exploratory representation analysis, not a validation of the supplied
    top-actor roles or of the semantic descriptions in the cached conditions.
    """

    if actor_alignment_manifest is None:
        raise ValueError("reviewed actor alignment manifest is required for semantic geometry")
    if not isinstance(reference_packet_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", reference_packet_sha256):
        raise ValueError("a valid reference packet SHA-256 is required for semantic geometry")
    if (
        not isinstance(target_conditions, (tuple, list))
        or not target_conditions
        or any(not isinstance(value, str) for value in target_conditions)
        or len(set(target_conditions)) != len(target_conditions)
        or set(target_conditions) - set(_DELTA_TARGETS.values())
    ):
        raise ValueError("target conditions must be an explicit non-empty unique action/contact subset")
    target_conditions = tuple(target_conditions)
    if len(top_actor_by_clip) < 3:
        raise ValueError("breadth geometry audit requires at least three clips")
    if any(actor not in {"A1", "A2"} for actor in top_actor_by_clip.values()):
        raise ValueError("top-actor mapping must contain only A1 or A2")
    role_notes = dict(role_notes_by_clip or {})
    if set(role_notes) - set(top_actor_by_clip) or any(
        not isinstance(note, str) or not note.strip() for note in role_notes.values()
    ):
        raise ValueError("role notes must be non-empty text for requested clips")
    if not context or not thinking_mode:
        raise ValueError("semantic context and thinking mode cannot be empty")
    root = Path(cache_root)
    if not root.is_dir():
        raise FileNotFoundError(f"breadth semantic cache is missing: {root}")
    clips = sorted(top_actor_by_clip)
    alignment_path = Path(actor_alignment_manifest)
    alignment_sha256 = _sha256(alignment_path)
    requested_conditions = ("identity_only", *target_conditions)
    # Verify every requested condition before allocating any semantic-state tensor.
    alignment_orders = {
        condition: load_alignment_orders(
            alignment_path, reference_packet_sha256=reference_packet_sha256,
            qwen_breadth_root=root, clip_ids=clips, condition=condition,
            context=context, thinking_mode=thinking_mode,
        )
        for condition in requested_conditions
    }
    loaded: dict[tuple[str, str], tuple[torch.Tensor, torch.Tensor]] = {}
    artifacts = []
    for clip_id in clips:
        for condition in requested_conditions:
            a1, a2, records = _load_pair_layers(
                root,
                clip_id=clip_id,
                condition=condition,
                context=context,
                thinking_mode=thinking_mode,
            )
            by_query = {"A1": a1, "A2": a2}
            order = alignment_orders[condition][clip_id]
            loaded[clip_id, condition] = (by_query[order[0]], by_query[order[1]])
            artifacts.extend(records)

    condition_results = {}
    selected_layers = {}
    layer_count = None
    for delta_name, target_condition in _DELTA_TARGETS.items():
        if target_condition not in target_conditions:
            continue
        role_directions = []
        for clip_id in clips:
            target_a1, target_a2 = loaded[clip_id, target_condition]
            identity_a1, identity_a2 = loaded[clip_id, "identity_only"]
            delta = (target_a1 - target_a2) - (identity_a1 - identity_a2)
            if top_actor_by_clip[clip_id] == "A2":
                delta = -delta
            role_directions.append(delta)
        stacked = torch.stack(role_directions)
        if layer_count is None:
            layer_count = int(stacked.shape[1])
        elif layer_count != int(stacked.shape[1]):
            raise CacheContractError("breadth semantic layer counts differ between conditions")
        layers = _layer_metrics(stacked)
        selected = max(
            layers,
            key=lambda value: (
                float(value["leave_one_clip_out_accuracy"]),
                float(value["leave_one_clip_out_mean_margin"]),
                float(value["mean_pairwise_cosine"]),
                -int(value["layer"]),
            ),
        )
        selected_layers[delta_name] = int(selected["layer"])
        condition_results[delta_name] = {
            "target_condition": target_condition,
            "baseline_condition": "identity_only",
            "selected_layer": int(selected["layer"]),
            "selection_order": [
                "leave_one_clip_out_accuracy",
                "leave_one_clip_out_mean_margin",
                "mean_pairwise_cosine",
                "lower_layer_index",
            ],
            "layers": layers,
        }

    artifacts.sort(key=lambda value: (value["clip_id"], value["condition"], value["actor"]))
    result = {
        "format": "breadth-semantic-geometry-audit-v2",
        "implementation_sha256": {
            name: _sha256(Path(__file__).resolve().parent / name)
            for name in ("cache.py", "semantic_geometry.py", "actor_alignment.py", "grounding_audit.py")
        },
        "north_star_eligible": False,
        "decoder_layer_selection_eligible": False,
        "interpretation": (
            "exploratory representation selection in reviewed physical-actor order; "
            "top-actor roles are supplied by the caller and are not validated by this audit. "
            "Valid grounding and actor alignment do not establish correct action/contact semantics "
            "or ownership-mask quality. Layers selected across all supplied clips are diagnostic "
            "and must not select decoder layers outside a nested training-only validation split."
        ),
        "cache_root": str(root.resolve()),
        "context": context,
        "thinking_mode": thinking_mode,
        "reference_packet_sha256": reference_packet_sha256,
        "actor_alignment_manifest": str(alignment_path.resolve()),
        "actor_alignment_sha256": alignment_sha256,
        "actor_alignment_orders": {
            condition: {clip: list(order) for clip, order in orders.items()}
            for condition, orders in alignment_orders.items()
        },
        "target_conditions": list(target_conditions),
        "baseline_condition": "identity_only",
        "actor_order_basis": "reviewed_physical_actors",
        "clip_ids": clips,
        "top_actor_by_clip": {clip_id: top_actor_by_clip[clip_id] for clip_id in clips},
        "role_notes_by_clip": role_notes,
        "layer_count": layer_count,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "selected_layers": selected_layers,
        "conditions": condition_results,
    }
    output = Path(output_path)
    if _sha256(alignment_path) != alignment_sha256:
        raise RuntimeError("actor alignment changed during semantic geometry audit")
    sidecar = output.with_suffix(output.suffix + ".sha256")
    if output.exists() or sidecar.exists():
        if not output.is_file() or not sidecar.is_file():
            raise RuntimeError("breadth semantic geometry audit is only partially committed")
        if sidecar.read_text(encoding="utf-8").strip() != _sha256(output):
            raise RuntimeError("breadth semantic geometry audit checksum mismatch")
        try:
            existing = json.loads(output.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError("breadth semantic geometry audit is invalid") from error
        if existing != result:
            raise RuntimeError("breadth semantic geometry audit inputs or result changed")
        return existing
    _atomic_json(output, result)
    sidecar.write_text(_sha256(output) + "\n", encoding="utf-8")
    return result
