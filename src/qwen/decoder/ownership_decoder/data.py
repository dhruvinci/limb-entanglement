from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .cache import CacheContractError, SpatialKind, load_actor_state_pair, load_spatial_map


@dataclass(frozen=True)
class SpatialSource:
    path: Path
    kind: SpatialKind


@dataclass(frozen=True)
class FrameSampleSpec:
    clip_id: str
    frame_index: int
    spatial: Mapping[str, SpatialSource]
    subset: str = "unspecified"
    screen_subset: str = "unspecified"
    label_path: Path | None = None
    label_sha256: str | None = None
    contact_path: Path | None = None
    contact_sha256: str | None = None
    actor_state_paths: tuple[Path, Path] | None = None
    language_layer: int | None = None
    rgb_path: Path | None = None
    rgb_sha256: str | None = None
    actor_state_order: tuple[str, str] = ("A1", "A2")
    label_grid_scale: int = 1


@dataclass
class OwnershipSample:
    clip_id: str
    frame_index: int
    spatial: dict[str, torch.Tensor]
    labels: torch.Tensor | None
    contact: torch.Tensor | None
    actor_states: torch.Tensor | None
    rgb: torch.Tensor | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise CacheContractError(f"could not read artifact for checksum: {path}") from error
    return digest.hexdigest()


def _load_mask(path: Path, expected_sha256: str | None, *, role: str) -> torch.Tensor:
    if expected_sha256 is None:
        raise CacheContractError(f"missing {role} checksum: {path}")
    actual_sha256 = _sha256(path)
    if actual_sha256 != expected_sha256:
        raise CacheContractError(
            f"{role} checksum mismatch for {path}: expected {expected_sha256}, got {actual_sha256}"
        )
    try:
        with Image.open(path) as image:
            values = np.asarray(image.convert("L"), dtype=np.uint8).copy()
    except OSError as error:
        raise CacheContractError(f"could not decode {role} image: {path}") from error
    if values.ndim != 2:
        raise CacheContractError(f"{role} image must be two-dimensional: {path}")
    return torch.from_numpy(values)


class OwnershipDataset(Dataset[OwnershipSample]):
    """Lazy, one-frame-at-a-time view over immutable representation caches."""

    def __init__(
        self,
        specs: list[FrameSampleSpec] | tuple[FrameSampleSpec, ...],
        *,
        rgb_output_hw: tuple[int, int] | None = (640, 360),
    ):
        self._specs = tuple(specs)
        if rgb_output_hw is not None and (
            rgb_output_hw[0] < 1 or rgb_output_hw[1] < 1
        ):
            raise ValueError("RGB output dimensions must be positive")
        self._rgb_output_hw = rgb_output_hw

    def __len__(self) -> int:
        return len(self._specs)

    def load_labels(self, index: int) -> torch.Tensor | None:
        """Load and validate only the small ownership mask for one sample."""

        spec = self._specs[index]
        if spec.label_path is None:
            return None
        raw_labels = _load_mask(spec.label_path, spec.label_sha256, role="label")
        allowed = torch.tensor([0, 1, 2, 255], dtype=torch.uint8)
        if not torch.isin(raw_labels, allowed).all():
            raise CacheContractError(
                f"label contains values outside 0,1,2,255: {spec.label_path}"
            )
        return raw_labels.long()

    def __getitem__(self, index: int) -> OwnershipSample:
        spec = self._specs[index]
        spatial = {
            name: load_spatial_map(source.path, kind=source.kind)
            for name, source in spec.spatial.items()
        }

        labels = self.load_labels(index)
        if labels is not None:
            if type(spec.label_grid_scale) is not int or not 1 <= spec.label_grid_scale <= 8:
                raise CacheContractError("label grid scale must be an integer in [1,8]")
            full_shapes = {
                tuple(v * spec.label_grid_scale for v in spatial[name].shape[-2:])
                for name, source in spec.spatial.items()
                if source.kind == "full"
            }
            if full_shapes and (len(full_shapes) != 1 or tuple(labels.shape) not in full_shapes):
                raise CacheContractError(
                    f"label grid {tuple(labels.shape)} does not match declared scaled spatial grid {sorted(full_shapes)}"
                )

        contact = None
        if spec.contact_path is not None:
            raw_contact = _load_mask(spec.contact_path, spec.contact_sha256, role="contact")
            if not torch.isin(raw_contact, torch.tensor([0, 1, 255], dtype=torch.uint8)).all():
                raise CacheContractError(f"contact mask must contain only 0,1,255: {spec.contact_path}")
            contact = raw_contact.bool()
            if labels is None or contact.shape != labels.shape:
                raise CacheContractError("contact mask requires a matching ownership label")
        elif labels is not None:
            contact = torch.zeros_like(labels, dtype=torch.bool)

        actor_states = None
        if spec.actor_state_paths is not None:
            if spec.language_layer is None:
                raise CacheContractError("semantic actor-state paths require a language layer")
            actor_states = load_actor_state_pair(
                spec.actor_state_paths[0],
                spec.actor_state_paths[1],
                language_layer=spec.language_layer,
                actor_order=spec.actor_state_order,
            )

        rgb = None
        if spec.rgb_path is not None:
            if spec.rgb_sha256 is None:
                raise CacheContractError(f"missing RGB checksum: {spec.rgb_path}")
            actual_rgb_sha256 = _sha256(spec.rgb_path)
            if actual_rgb_sha256 != spec.rgb_sha256:
                raise CacheContractError(
                    f"RGB checksum mismatch for {spec.rgb_path}: "
                    f"expected {spec.rgb_sha256}, got {actual_rgb_sha256}"
                )
            if self._rgb_output_hw is None:
                if labels is None:
                    raise CacheContractError(
                        "label-grid RGB resizing requires an ownership label"
                    )
                output_height, output_width = tuple(labels.shape)
            else:
                output_height, output_width = self._rgb_output_hw
            try:
                with Image.open(spec.rgb_path) as image:
                    resized = image.convert("RGB").resize(
                        (output_width, output_height),
                        Image.Resampling.BILINEAR,
                    )
                    rgb_values = np.asarray(resized, dtype=np.float32).copy() / 255.0
            except OSError as error:
                raise CacheContractError(f"could not decode RGB image: {spec.rgb_path}") from error
            rgb = torch.from_numpy(rgb_values).permute(2, 0, 1).contiguous()

        return OwnershipSample(
            clip_id=spec.clip_id,
            frame_index=spec.frame_index,
            spatial=spatial,
            labels=labels,
            contact=contact,
            actor_states=actor_states,
            rgb=rgb,
        )


class ActorStateControlDataset(Dataset[OwnershipSample]):
    """Deterministic semantic controls over an otherwise identical dataset."""

    _CONTROLS = {
        "real",
        "swapped",
        "shuffled_clip",
        "zero",
        "mean",
        "random_matched",
    }

    def __init__(
        self,
        source: Dataset[OwnershipSample] | list[OwnershipSample] | tuple[OwnershipSample, ...],
        *,
        control: str,
        seed: int = 7,
        replacement_actor_states: torch.Tensor | None = None,
    ):
        if control not in self._CONTROLS:
            raise ValueError(f"unsupported actor-state control: {control}")
        if control == "shuffled_clip":
            if (
                replacement_actor_states is None
                or replacement_actor_states.ndim != 2
                or replacement_actor_states.shape[0] != 2
            ):
                raise ValueError(
                    "shuffled-clip control requires replacement actor states shaped [2,D]"
                )
        elif replacement_actor_states is not None:
            raise ValueError("replacement actor states are only valid for shuffled-clip control")
        self.source = source
        self.control = control
        self.seed = seed
        self.replacement_actor_states = (
            replacement_actor_states.detach().clone()
            if replacement_actor_states is not None
            else None
        )

    def __len__(self) -> int:
        return len(self.source)

    def load_labels(self, index: int) -> torch.Tensor | None:
        """Preserve the source's label-only fast path during class-weight scans."""

        label_loader = getattr(self.source, "load_labels", None)
        if callable(label_loader):
            return label_loader(index)
        return self.source[index].labels

    def __getitem__(self, index: int) -> OwnershipSample:
        sample = self.source[index]
        if sample.actor_states is None:
            raise CacheContractError("actor-state control requires semantic states")
        reference = sample.actor_states
        if self.control == "real":
            controlled = reference.clone()
        elif self.control == "swapped":
            controlled = reference.flip(0).clone()
        elif self.control == "shuffled_clip":
            assert self.replacement_actor_states is not None
            if self.replacement_actor_states.shape != reference.shape:
                raise ValueError(
                    "replacement actor states must match the source actor-state shape"
                )
            controlled = self.replacement_actor_states.to(
                device=reference.device,
                dtype=reference.dtype,
            ).clone()
        elif self.control == "zero":
            controlled = torch.zeros_like(reference)
        elif self.control == "mean":
            controlled = reference.mean(dim=0, keepdim=True).expand_as(reference).clone()
        else:
            digest = hashlib.sha256(f"{self.seed}:{sample.clip_id}".encode("utf-8")).digest()
            generator = torch.Generator().manual_seed(int.from_bytes(digest[:8], "big"))
            reference_float = reference.float()
            pair_mean = reference_float.mean(dim=0)
            residuals = reference_float - pair_mean
            residual_magnitude = residuals.norm(dim=-1).mean()
            random_direction = torch.randn(
                reference.shape[-1],
                generator=generator,
                dtype=torch.float32,
            )
            random_direction = torch.nn.functional.normalize(
                random_direction,
                dim=0,
            )
            random_residual = random_direction * residual_magnitude
            controlled = torch.stack(
                (pair_mean + random_residual, pair_mean - random_residual)
            ).to(reference.dtype)
        return replace(sample, actor_states=controlled)


def _safe_manifest_path(root: Path, relative_path: str) -> Path:
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise CacheContractError(f"manifest path escapes its root: {relative_path}") from error
    return candidate


def load_rgb_records(
    frame_manifest_path: str | Path,
    *,
    project_root: str | Path,
) -> dict[int, tuple[Path, str]]:
    """Read the frozen frame inventory without decoding any image data."""

    manifest_path = Path(frame_manifest_path)
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise CacheContractError(f"could not read RGB frame manifest: {manifest_path}") from error
    frames = manifest.get("frames")
    if not isinstance(frames, list) or not frames:
        raise CacheContractError("RGB frame manifest must contain a non-empty frames list")
    root = Path(project_root)
    records: dict[int, tuple[Path, str]] = {}
    for frame in frames:
        try:
            has_frame_index = "frame_index" in frame
            has_clip_frame_index = "clip_frame_index" in frame
            if not has_frame_index and not has_clip_frame_index:
                raise KeyError("frame_index")
            if has_frame_index and has_clip_frame_index and int(
                frame["frame_index"]
            ) != int(frame["clip_frame_index"]):
                raise ValueError("frame_index and clip_frame_index disagree")
            frame_index = int(
                frame["frame_index"]
                if has_frame_index
                else frame["clip_frame_index"]
            )
            path = _safe_manifest_path(root, str(frame["path"]))
            sha256 = str(frame["sha256"])
        except (KeyError, TypeError, ValueError) as error:
            raise CacheContractError("RGB frame manifest record is missing required fields") from error
        if frame_index in records:
            raise CacheContractError(f"duplicate frame index in RGB manifest: {frame_index}")
        records[frame_index] = (path, sha256)
    return records


def build_specs_from_label_manifest(
    manifest_path: str | Path,
    *,
    cache_root: str | Path,
    clip_id: str,
    full_layers: tuple[int, ...] = (),
    pooled_layers: tuple[int, ...] = (),
    include_merged: bool = False,
    actor_state_paths: tuple[Path, Path] | None = None,
    language_layer: int | None = None,
    rgb_records: Mapping[int, tuple[Path, str]] | None = None,
    require_reviewed: bool = False,
    actor_state_order: tuple[str, str] = ("A1", "A2"),
) -> tuple[FrameSampleSpec, ...]:
    """Resolve frozen label records into deterministic, still-lazy frame specs."""

    label_manifest_path = Path(manifest_path)
    if require_reviewed:
        from .breadth_labels import verify_reviewed_label_manifest

        manifest = verify_reviewed_label_manifest(
            label_manifest_path,
            expected_clip_id=clip_id,
        )
    else:
        try:
            manifest = json.loads(label_manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise CacheContractError(f"could not read label manifest: {label_manifest_path}") from error
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise CacheContractError("label manifest must contain a non-empty records list")
    if len(set(full_layers)) != len(full_layers) or any(layer < 0 or layer > 26 for layer in full_layers):
        raise CacheContractError("full vision layers must be unique values in [0,26]")
    if len(set(pooled_layers)) != len(pooled_layers) or any(
        layer < 0 or layer > 26 for layer in pooled_layers
    ):
        raise CacheContractError("pooled vision layers must be unique values in [0,26]")
    if actor_state_paths is None and language_layer is not None:
        raise CacheContractError("a language layer requires semantic actor-state paths")
    if actor_state_paths is not None and language_layer is None:
        raise CacheContractError("semantic actor-state paths require a language layer")

    label_root = label_manifest_path.parent
    representation_root = Path(cache_root)
    seen_frames: set[int] = set()
    specs: list[FrameSampleSpec] = []
    for record in records:
        if not isinstance(record, dict):
            raise CacheContractError("every label manifest record must be an object")
        try:
            frame_index = int(record["frame_index"])
            label_relative = str(record["label_path"])
            contact_relative = str(record["contact_path"])
            label_sha256 = str(record["label_sha256"])
            contact_sha256 = str(record["contact_sha256"])
        except (KeyError, TypeError, ValueError) as error:
            raise CacheContractError("label manifest record is missing required fields") from error
        if frame_index in seen_frames:
            raise CacheContractError(f"duplicate frame index in label manifest: {frame_index}")
        seen_frames.add(frame_index)
        filename = f"frame_{frame_index:06d}.safetensors"
        spatial = {
            f"layer_{layer:02d}": SpatialSource(
                representation_root / f"spatial/full/layer_{layer:02d}" / filename,
                "full",
            )
            for layer in full_layers
        }
        spatial.update(
            {
                f"pooled_{layer:02d}": SpatialSource(
                    representation_root
                    / f"spatial/pooled/layer_{layer:02d}"
                    / filename,
                    "pooled",
                )
                for layer in pooled_layers
            }
        )
        if include_merged:
            spatial["merged"] = SpatialSource(representation_root / "merged-vision" / filename, "merged")
        if rgb_records is not None and frame_index not in rgb_records:
            raise CacheContractError(f"RGB manifest is missing labeled frame {frame_index}")
        rgb_path, rgb_sha256 = rgb_records[frame_index] if rgb_records is not None else (None, None)
        specs.append(
            FrameSampleSpec(
                clip_id=clip_id,
                frame_index=frame_index,
                spatial=spatial,
                subset=str(record.get("subset", "unspecified")),
                screen_subset=str(record.get("screen_subset", record.get("subset", "unspecified"))),
                label_path=_safe_manifest_path(label_root, label_relative),
                label_sha256=label_sha256,
                contact_path=_safe_manifest_path(label_root, contact_relative),
                contact_sha256=contact_sha256,
                actor_state_paths=actor_state_paths,
                actor_state_order=actor_state_order,
                label_grid_scale=manifest.get("label_grid_scale", 1),
                language_layer=language_layer,
                rgb_path=rgb_path,
                rgb_sha256=rgb_sha256,
            )
        )
    return tuple(specs)
