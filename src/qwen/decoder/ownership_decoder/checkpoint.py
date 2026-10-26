from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import torch


class CheckpointError(RuntimeError):
    """Raised when an atomic checkpoint cannot be trusted or restored."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".json")


def _checkpoint_artifacts(path: Path) -> list[Path]:
    sidecar_path = _manifest_path(path)
    candidates = [path, sidecar_path]
    candidates.extend(path.parent.glob(f".{path.name}.*.tmp"))
    candidates.extend(path.parent.glob(f".{sidecar_path.name}.*.tmp"))
    return sorted({candidate for candidate in candidates if candidate.exists()})


def _quarantine_invalid_checkpoint(path: Path, artifacts: list[Path]) -> Path:
    failure_root = path.parent / "failures" / "orphaned-checkpoints"
    failure_root.mkdir(parents=True, exist_ok=True)
    quarantine = Path(tempfile.mkdtemp(prefix=f"{path.name}.", dir=failure_root))
    for artifact in artifacts:
        os.replace(artifact, quarantine / artifact.name)
    return quarantine


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def save_checkpoint(
    path: str | Path,
    state: dict[str, Any],
    *,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    checkpoint_path = Path(path)
    sidecar_path = _manifest_path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    existing_artifacts = _checkpoint_artifacts(checkpoint_path)
    if checkpoint_path.exists() and sidecar_path.exists():
        try:
            load_checkpoint(checkpoint_path)
        except CheckpointError:
            pass
        else:
            raise CheckpointError(
                f"refusing to overwrite verified checkpoint: {checkpoint_path}"
            )
    if existing_artifacts:
        _quarantine_invalid_checkpoint(checkpoint_path, existing_artifacts)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=checkpoint_path.parent,
            prefix=f".{checkpoint_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            torch.save(state, handle)
            handle.flush()
            os.fsync(handle.fileno())
        manifest = {
            "format": "ownership-decoder-checkpoint-v1",
            "size_bytes": temporary_path.stat().st_size,
            "sha256": _sha256(temporary_path),
            "metadata": metadata,
        }
        os.replace(temporary_path, checkpoint_path)
        _atomic_json(sidecar_path, manifest)
        return manifest
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def load_checkpoint(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint_path = Path(path)
    sidecar_path = _manifest_path(checkpoint_path)
    try:
        manifest = json.loads(sidecar_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise CheckpointError(f"missing or invalid checkpoint manifest: {sidecar_path}") from error
    if manifest.get("format") != "ownership-decoder-checkpoint-v1":
        raise CheckpointError(f"unsupported checkpoint format: {sidecar_path}")
    try:
        actual_size = checkpoint_path.stat().st_size
        actual_sha256 = _sha256(checkpoint_path)
    except OSError as error:
        raise CheckpointError(f"missing checkpoint data: {checkpoint_path}") from error
    if actual_size != manifest.get("size_bytes"):
        raise CheckpointError(f"checkpoint size mismatch: {checkpoint_path}")
    if actual_sha256 != manifest.get("sha256"):
        raise CheckpointError(f"checkpoint checksum mismatch: {checkpoint_path}")
    try:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise CheckpointError(f"checkpoint payload could not be restored: {checkpoint_path}") from error
    if not isinstance(state, dict):
        raise CheckpointError(f"checkpoint payload must be a dictionary: {checkpoint_path}")
    return state, manifest


def latest_valid_checkpoint(directory: str | Path) -> Path | None:
    for path in sorted(Path(directory).glob("epoch_*.pt"), reverse=True):
        try:
            load_checkpoint(path)
        except CheckpointError:
            continue
        return path
    return None


def prune_checkpoints(
    directory: str | Path,
    *,
    keep_epochs: set[int],
    keep_latest: int = 1,
) -> list[Path]:
    """Retain the scientific best plus recent resume points after new writes validate."""

    if keep_latest < 1:
        raise ValueError("at least one latest checkpoint must be retained")
    valid: list[tuple[int, Path]] = []
    for path in sorted(Path(directory).glob("epoch_*.pt")):
        try:
            epoch = int(path.stem.removeprefix("epoch_"))
            load_checkpoint(path)
        except (ValueError, CheckpointError):
            continue
        valid.append((epoch, path))
    retained_epochs = set(keep_epochs)
    retained_epochs.update(epoch for epoch, _ in valid[-keep_latest:])
    removed = []
    for epoch, path in valid:
        if epoch in retained_epochs:
            continue
        path.unlink()
        _manifest_path(path).unlink()
        removed.append(path)
    return removed
