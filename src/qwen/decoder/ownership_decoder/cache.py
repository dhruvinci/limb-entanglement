from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import torch
from safetensors import safe_open


SpatialKind = Literal["full", "pooled", "merged"]


class CacheContractError(ValueError):
    """Raised when a cached representation violates the extraction contract."""


_CONTRACTS = {
    "full": {"stage": "spatial_full", "hidden_size": 1152, "merge_size": 1},
    "pooled": {"stage": "spatial_pooled", "hidden_size": 1152, "merge_size": 1},
    "merged": {"stage": "merged_vision", "hidden_size": 5120, "merge_size": 2},
}


def _campaign_metadata(raw_metadata: dict[str, str] | None, path: Path) -> dict:
    if not raw_metadata or "campaign" not in raw_metadata:
        raise CacheContractError(f"missing campaign metadata: {path}")
    try:
        value = json.loads(raw_metadata["campaign"])
    except (json.JSONDecodeError, TypeError) as error:
        raise CacheContractError(f"invalid campaign metadata: {path}") from error
    if not isinstance(value, dict):
        raise CacheContractError(f"campaign metadata must be an object: {path}")
    return value


def load_spatial_map(path: str | Path, *, kind: SpatialKind) -> torch.Tensor:
    """Return an immutable-use ``[C, H, W]`` view over one frame's cache storage.

    The noncontiguous view retains its storage after the safetensors context
    closes. Callers must not modify it in place; device batching materializes
    the contiguous model input without first copying the entire map on CPU.
    """

    artifact_path = Path(path)
    if kind not in _CONTRACTS:
        raise CacheContractError(f"unsupported spatial kind: {kind}")
    contract = _CONTRACTS[kind]
    try:
        with safe_open(artifact_path, framework="pt", device="cpu") as artifact:
            keys = set(artifact.keys())
            if not {"hidden", "grid_thw"}.issubset(keys):
                raise CacheContractError(f"missing hidden or grid_thw tensor: {artifact_path}")
            metadata = _campaign_metadata(artifact.metadata(), artifact_path)
            hidden = artifact.get_tensor("hidden")
            grid_thw = artifact.get_tensor("grid_thw")
    except CacheContractError:
        raise
    except Exception as error:
        raise CacheContractError(f"could not read safetensor artifact: {artifact_path}") from error

    if metadata.get("stage") != contract["stage"]:
        raise CacheContractError(
            f"stage mismatch for {artifact_path}: expected {contract['stage']}, got {metadata.get('stage')}"
        )
    if grid_thw.dtype != torch.int64 or tuple(grid_thw.shape) != (1, 3):
        raise CacheContractError(f"grid_thw must be int64 [1,3]: {artifact_path}")
    temporal, grid_height, grid_width = (int(value) for value in grid_thw[0].tolist())
    if temporal != 1 or grid_height < 1 or grid_width < 1:
        raise CacheContractError(f"invalid per-frame grid {grid_thw.tolist()}: {artifact_path}")
    if hidden.ndim != 2 or hidden.shape[1] != contract["hidden_size"]:
        raise CacheContractError(
            f"hidden must be [tokens,{contract['hidden_size']}], got {tuple(hidden.shape)}: {artifact_path}"
        )
    if hidden.dtype != torch.bfloat16:
        raise CacheContractError(f"hidden must use bfloat16, got {hidden.dtype}: {artifact_path}")
    if not torch.isfinite(hidden).all():
        raise CacheContractError(f"hidden contains non-finite values: {artifact_path}")

    merge_size = int(contract["merge_size"])
    if grid_height % merge_size or grid_width % merge_size:
        raise CacheContractError(
            f"grid dimensions must be divisible by merge size {merge_size}: {artifact_path}"
        )
    effective_height = grid_height // merge_size
    effective_width = grid_width // merge_size
    expected_tokens = temporal * effective_height * effective_width
    if hidden.shape[0] != expected_tokens:
        raise CacheContractError(
            f"token count mismatch: expected {expected_tokens}, got {hidden.shape[0]}: {artifact_path}"
        )

    # KNOWN BUG for kind='full' (found 2026-10-13, see experiments/E03): raw full-layer tokens are in 2x2-grouped
    # order, so this direct reshape misplaces most locations. 'merged' tokens (one per 2x2 group) were not audited.
    # Kept as run in September 2026; the fix is restore_grid() in src/qwen/final_cache/core.py.
    return hidden.reshape(effective_height, effective_width, hidden.shape[1]).permute(2, 0, 1)


def _load_actor_states(path: Path, expected_actor: str) -> tuple[torch.Tensor, dict]:
    try:
        with safe_open(path, framework="pt", device="cpu") as artifact:
            if "marker_states" not in artifact.keys():
                raise CacheContractError(f"missing marker_states tensor: {path}")
            metadata = _campaign_metadata(artifact.metadata(), path)
            states = artifact.get_tensor("marker_states")
    except CacheContractError:
        raise
    except Exception as error:
        raise CacheContractError(f"could not read semantic safetensor artifact: {path}") from error

    if metadata.get("actor") != expected_actor:
        raise CacheContractError(
            f"semantic actor mismatch for {path}: expected {expected_actor}, got {metadata.get('actor')}"
        )
    if metadata.get("stage") not in {"semantic_image", "semantic_video"}:
        raise CacheContractError(f"invalid semantic stage for {path}: {metadata.get('stage')}")
    if states.ndim != 2 or states.shape[1] != 5120:
        raise CacheContractError(f"marker_states must be [layers,5120], got {tuple(states.shape)}: {path}")
    if states.dtype != torch.bfloat16:
        raise CacheContractError(f"marker_states must use bfloat16, got {states.dtype}: {path}")
    if not torch.isfinite(states).all():
        raise CacheContractError(f"marker_states contains non-finite values: {path}")
    return states, metadata


def load_actor_state_pair(
    a1_path: str | Path,
    a2_path: str | Path,
    *,
    language_layer: int,
    actor_order: tuple[str, str] = ("A1", "A2"),
) -> torch.Tensor:
    """Load the selected marker state in deterministic physical A1/A2 order."""

    if actor_order not in (("A1", "A2"), ("A2", "A1")):
        raise CacheContractError("actor order must map both physical actors exactly once")
    a1_states, a1_metadata = _load_actor_states(Path(a1_path), "A1")
    a2_states, a2_metadata = _load_actor_states(Path(a2_path), "A2")
    comparison_fields = ("stage", "condition", "context", "thinking_mode")
    if any(a1_metadata.get(field) != a2_metadata.get(field) for field in comparison_fields):
        raise CacheContractError("semantic metadata mismatch between A1 and A2 artifacts")
    if language_layer < 0 or language_layer >= a1_states.shape[0] or language_layer >= a2_states.shape[0]:
        raise CacheContractError(
            f"language layer {language_layer} is outside available marker-state layers"
        )
    by_actor = {"A1": a1_states[language_layer], "A2": a2_states[language_layer]}
    return torch.stack(tuple(by_actor[actor] for actor in actor_order)).contiguous()
