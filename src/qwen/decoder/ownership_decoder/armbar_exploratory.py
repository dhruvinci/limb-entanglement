from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .experiments import STATIC_ARMS


ARMBAR_SUPERVISION_STATUS = "exploratory_legacy_pseudo_labels"
ARMBAR_SPLITS = {"screen", "final"}
ARMBAR_TRAINING_CONTROLS = {"real", "random_matched", "zero", "mean"}
ARMBAR_SEMANTIC_CONDITIONS = {
    "identity_only",
    "action_relational",
    "contact_ownership",
    "action_delta",
    "contact_delta",
}
ARMBAR_DELTA_CONDITIONS = {
    "action_delta": "action_relational",
    "contact_delta": "contact_ownership",
}


@dataclass(frozen=True)
class FixedSubstitution:
    name: str
    context: str
    condition: str
    thinking_mode: str = "off"
    flip_actors: bool = False


ARMBAR_FIXED_SUBSTITUTIONS = (
    FixedSubstitution("identity_ordered_4fps_off", "4fps", "identity_only"),
    FixedSubstitution("contact_ordered_4fps_off", "4fps", "contact_ownership"),
    FixedSubstitution("action_ordered_4fps_off", "4fps", "action_relational"),
    FixedSubstitution(
        "action_ordered_4fps_xhigh",
        "4fps",
        "action_relational",
        thinking_mode="xhigh",
    ),
    FixedSubstitution("action_single_frame_off", "single_frame", "action_relational"),
    FixedSubstitution(
        "action_five_frame_remapped",
        "five_frame",
        "action_relational",
        flip_actors=True,
    ),
    FixedSubstitution(
        "action_2fps_remapped",
        "2fps",
        "action_relational",
        flip_actors=True,
    ),
    FixedSubstitution("action_reversed_4fps_off", "4fps_reversed", "action_relational"),
    FixedSubstitution("action_shuffled_4fps_off", "4fps_shuffled", "action_relational"),
    FixedSubstitution("action_8fps_off", "8fps", "action_relational"),
)


@dataclass(frozen=True)
class ArmbarJobSpec:
    run_name: str
    spatial_arm: str
    split: str
    label_manifest: Path
    cache_root: Path
    frame_manifest: Path
    frame_project_root: Path
    output_root: Path
    semantic_condition: str | None = None
    language_layer: int | None = None
    semantic_context: str = "4fps"
    thinking_mode: str = "off"
    training_control: str = "real"
    width: int = 96
    residual_blocks: int = 2
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    max_epochs: int = 80
    patience: int = 10
    gradient_accumulation: int = 4
    seed: int = 7
    device: str = "cuda"
    use_amp: bool = True
    cuda_memory_fraction: float = 0.60


def validate_armbar_job_spec(spec: ArmbarJobSpec) -> ArmbarJobSpec:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", spec.run_name):
        raise ValueError("armbar run name must be a safe path component")
    if spec.spatial_arm not in STATIC_ARMS:
        raise ValueError(f"unknown armbar spatial arm: {spec.spatial_arm}")
    if spec.split not in ARMBAR_SPLITS:
        raise ValueError("armbar split must be screen or final")
    has_condition = spec.semantic_condition is not None
    has_layer = spec.language_layer is not None
    if has_condition != has_layer:
        raise ValueError("semantic condition and language layer must be supplied together")
    if has_condition and spec.semantic_condition not in ARMBAR_SEMANTIC_CONDITIONS:
        raise ValueError(f"unsupported armbar semantic condition: {spec.semantic_condition}")
    if has_layer and not 0 <= int(spec.language_layer) <= 63:
        raise ValueError("armbar language layer must be in [0, 63]")
    if spec.training_control not in ARMBAR_TRAINING_CONTROLS:
        raise ValueError(f"unsupported armbar training control: {spec.training_control}")
    if not has_condition and spec.training_control != "real":
        raise ValueError("static armbar jobs cannot use a semantic training control")
    if not spec.semantic_context or not spec.thinking_mode:
        raise ValueError("armbar semantic context and thinking mode cannot be empty")
    if not 8 <= spec.width <= 256 or not 0 <= spec.residual_blocks <= 4:
        raise ValueError("armbar decoder architecture is outside its safe range")
    if spec.learning_rate <= 0 or spec.weight_decay < 0:
        raise ValueError("armbar optimizer settings are invalid")
    if min(spec.max_epochs, spec.patience, spec.gradient_accumulation) < 1:
        raise ValueError("armbar epoch, patience, and accumulation settings must be positive")
    if spec.device not in {"cpu", "cuda"}:
        raise ValueError("armbar device must be cpu or cuda")
    if not 0.05 <= spec.cuda_memory_fraction <= 0.90:
        raise ValueError("armbar CUDA memory fraction must be in [0.05, 0.90]")
    return spec


def armbar_job_spec_to_dict(spec: ArmbarJobSpec) -> dict[str, Any]:
    validate_armbar_job_spec(spec)
    return {
        "run_name": spec.run_name,
        "spatial_arm": spec.spatial_arm,
        "split": spec.split,
        "label_manifest": str(spec.label_manifest),
        "cache_root": str(spec.cache_root),
        "frame_manifest": str(spec.frame_manifest),
        "frame_project_root": str(spec.frame_project_root),
        "output_root": str(spec.output_root),
        "semantic_condition": spec.semantic_condition,
        "language_layer": spec.language_layer,
        "semantic_context": spec.semantic_context,
        "thinking_mode": spec.thinking_mode,
        "training_control": spec.training_control,
        "width": spec.width,
        "residual_blocks": spec.residual_blocks,
        "learning_rate": spec.learning_rate,
        "weight_decay": spec.weight_decay,
        "max_epochs": spec.max_epochs,
        "patience": spec.patience,
        "gradient_accumulation": spec.gradient_accumulation,
        "seed": spec.seed,
        "device": spec.device,
        "use_amp": spec.use_amp,
        "cuda_memory_fraction": spec.cuda_memory_fraction,
    }


def armbar_job_spec_from_dict(value: Mapping[str, Any]) -> ArmbarJobSpec:
    expected = set(armbar_job_spec_to_dict(ArmbarJobSpec(
        run_name="schema",
        spatial_arm="l11",
        split="screen",
        label_manifest=Path("label"),
        cache_root=Path("cache"),
        frame_manifest=Path("frames"),
        frame_project_root=Path("project"),
        output_root=Path("output"),
    )))
    if set(value) != expected:
        raise ValueError("armbar job work-item schema is invalid")
    spec = ArmbarJobSpec(
        run_name=str(value["run_name"]),
        spatial_arm=str(value["spatial_arm"]),
        split=str(value["split"]),
        label_manifest=Path(str(value["label_manifest"])),
        cache_root=Path(str(value["cache_root"])),
        frame_manifest=Path(str(value["frame_manifest"])),
        frame_project_root=Path(str(value["frame_project_root"])),
        output_root=Path(str(value["output_root"])),
        semantic_condition=(
            str(value["semantic_condition"])
            if value["semantic_condition"] is not None
            else None
        ),
        language_layer=(
            int(value["language_layer"])
            if value["language_layer"] is not None
            else None
        ),
        semantic_context=str(value["semantic_context"]),
        thinking_mode=str(value["thinking_mode"]),
        training_control=str(value["training_control"]),
        width=int(value["width"]),
        residual_blocks=int(value["residual_blocks"]),
        learning_rate=float(value["learning_rate"]),
        weight_decay=float(value["weight_decay"]),
        max_epochs=int(value["max_epochs"]),
        patience=int(value["patience"]),
        gradient_accumulation=int(value["gradient_accumulation"]),
        seed=int(value["seed"]),
        device=str(value["device"]),
        use_amp=bool(value["use_amp"]),
        cuda_memory_fraction=float(value["cuda_memory_fraction"]),
    )
    return validate_armbar_job_spec(spec)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_armbar_semantic_pair(
    cache_root: str | Path,
    *,
    condition: str,
    context: str,
    thinking_mode: str,
    language_layer: int,
):
    """Load a raw actor pair or a condition-minus-identity pair."""

    from .cache import load_actor_state_pair

    cache_condition = ARMBAR_DELTA_CONDITIONS.get(condition, condition)
    semantic_root = (
        Path(cache_root)
        / "semantic"
        / "video"
        / context
        / cache_condition
        / thinking_mode
    )
    pair = load_actor_state_pair(
        semantic_root / "A1.safetensors",
        semantic_root / "A2.safetensors",
        language_layer=language_layer,
    )
    if condition not in ARMBAR_DELTA_CONDITIONS:
        return pair
    identity_root = (
        Path(cache_root)
        / "semantic"
        / "video"
        / context
        / "identity_only"
        / thinking_mode
    )
    identity = load_actor_state_pair(
        identity_root / "A1.safetensors",
        identity_root / "A2.safetensors",
        language_layer=language_layer,
    )
    return (pair.float() - identity.float()).to(pair.dtype).contiguous()


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


def _source_contract(spec: ArmbarJobSpec) -> dict[str, Any]:
    try:
        label_manifest = json.loads(spec.label_manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("armbar label manifest is unreadable") from error
    if label_manifest.get("status") != (
        "conservative_pseudo_labels_with_manual_final_contact_truth"
    ):
        raise ValueError("armbar job requires the declared legacy pseudo-label protocol")
    if label_manifest.get("training_eligible") is True:
        raise ValueError("legacy armbar labels cannot claim human-reviewed training eligibility")
    records = label_manifest.get("records")
    if not isinstance(records, list) or len(records) < 3:
        raise ValueError("armbar label manifest does not populate all data splits")
    cache_manifest = spec.cache_root / "download-manifest.json"
    return {
        "supervision_status": ARMBAR_SUPERVISION_STATUS,
        "label_manifest_sha256": _sha256(spec.label_manifest),
        "frame_manifest_sha256": _sha256(spec.frame_manifest),
        "cache_root": str(spec.cache_root.resolve()),
        "cache_download_manifest_sha256": (
            _sha256(cache_manifest) if cache_manifest.is_file() else None
        ),
    }


def _job_payload(spec: ArmbarJobSpec) -> dict[str, Any]:
    return {
        "format": "armbar-exploratory-job-spec-v1",
        "job": armbar_job_spec_to_dict(spec),
        "sources": _source_contract(spec),
    }


def _load_completed_job(run_root: Path, expected_spec: Mapping[str, Any]) -> dict[str, Any]:
    spec_path = run_root / "run-spec.json"
    result_path = run_root / "result.json"
    completion_path = run_root / "RUN_COMPLETE"
    try:
        observed_spec = json.loads(spec_path.read_text(encoding="utf-8"))
        result = json.loads(result_path.read_text(encoding="utf-8"))
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"armbar exploratory job is incomplete: {run_root}") from error
    if observed_spec != expected_spec:
        raise RuntimeError("armbar exploratory job specification changed")
    if completion.get("result_sha256") != _sha256(result_path):
        raise RuntimeError("armbar exploratory result checksum mismatch")
    if result.get("run_spec_sha256") != _sha256(spec_path):
        raise RuntimeError("armbar exploratory result is not bound to its run specification")
    return result


def _build_job_datasets(spec: ArmbarJobSpec):
    from .data import (
        ActorStateControlDataset,
        OwnershipDataset,
        build_specs_from_label_manifest,
        load_rgb_records,
    )
    from .experiments import split_armbar_specs

    arm = STATIC_ARMS[spec.spatial_arm]
    rgb_records = None
    if arm.use_rgb:
        rgb_records = load_rgb_records(
            spec.frame_manifest,
            project_root=spec.frame_project_root,
        )
    actor_paths = None
    if spec.semantic_condition is not None:
        cache_condition = ARMBAR_DELTA_CONDITIONS.get(
            spec.semantic_condition,
            spec.semantic_condition,
        )
        semantic_root = (
            spec.cache_root
            / "semantic"
            / "video"
            / spec.semantic_context
            / cache_condition
            / spec.thinking_mode
        )
        actor_paths = (
            semantic_root / "A1.safetensors",
            semantic_root / "A2.safetensors",
        )
        if any(not path.is_file() for path in actor_paths):
            raise FileNotFoundError(f"armbar semantic state pair is incomplete: {semantic_root}")
    frame_specs = build_specs_from_label_manifest(
        spec.label_manifest,
        cache_root=spec.cache_root,
        clip_id="armbar",
        full_layers=arm.full_layers,
        pooled_layers=arm.pooled_layers,
        include_merged=arm.include_merged,
        actor_state_paths=actor_paths,
        language_layer=spec.language_layer,
        rgb_records=rgb_records,
        require_reviewed=False,
    )
    split = split_armbar_specs(frame_specs)
    if spec.split == "screen":
        train_specs = split.screen_train
        evaluation_specs = split.validation
        evaluation_subset = "validation"
    else:
        train_specs = split.final_train
        evaluation_specs = split.test
        evaluation_subset = "test"
    train_dataset = OwnershipDataset(train_specs, rgb_output_hw=None)
    evaluation_dataset = OwnershipDataset(evaluation_specs, rgb_output_hw=None)
    if spec.semantic_condition in ARMBAR_DELTA_CONDITIONS:
        delta_pair = load_armbar_semantic_pair(
            spec.cache_root,
            condition=spec.semantic_condition,
            context=spec.semantic_context,
            thinking_mode=spec.thinking_mode,
            language_layer=int(spec.language_layer),
        )
        train_dataset = ActorStateControlDataset(
            train_dataset,
            control="shuffled_clip",
            replacement_actor_states=delta_pair,
        )
        evaluation_dataset = ActorStateControlDataset(
            evaluation_dataset,
            control="shuffled_clip",
            replacement_actor_states=delta_pair,
        )
    return train_dataset, evaluation_dataset, evaluation_subset


def _fixed_substitution_pairs(spec: ArmbarJobSpec):
    from .cache import load_actor_state_pair

    pairs = {}
    for substitution in ARMBAR_FIXED_SUBSTITUTIONS:
        root = (
            spec.cache_root
            / "semantic"
            / "video"
            / substitution.context
            / substitution.condition
            / substitution.thinking_mode
        )
        pair = load_actor_state_pair(
            root / "A1.safetensors",
            root / "A2.safetensors",
            language_layer=int(spec.language_layer),
        )
        if substitution.flip_actors:
            pair = pair.flip(0).contiguous()
        pairs[substitution.name] = pair
    return pairs


def run_armbar_job(spec: ArmbarJobSpec) -> dict[str, Any]:
    """Train one isolated legacy-armbar job without upgrading its evidence status."""

    validate_armbar_job_spec(spec)
    run_root = spec.output_root / spec.run_name
    expected_spec = _job_payload(spec)
    completion_path = run_root / "RUN_COMPLETE"
    if completion_path.exists():
        return _load_completed_job(run_root, expected_spec)
    run_root.mkdir(parents=True, exist_ok=True)
    spec_path = run_root / "run-spec.json"
    if spec_path.exists():
        try:
            observed_spec = json.loads(spec_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError("partial armbar job has an invalid run specification") from error
        if observed_spec != expected_spec:
            raise RuntimeError("partial armbar job specification changed")
    else:
        _atomic_json(spec_path, expected_spec)

    import torch

    from .data import ActorStateControlDataset
    from .model import OwnershipDecoder, SemanticOwnershipDecoder
    from .training import (
        TrainingConfig,
        evaluate_decoder,
        evaluate_query_swap,
        train_decoder,
    )

    if torch.get_num_threads() != 1:
        torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    device = torch.device(spec.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA armbar job requested but CUDA is unavailable")
        torch.cuda.set_per_process_memory_fraction(spec.cuda_memory_fraction, device=0)
        torch.cuda.reset_peak_memory_stats(device)
    runtime = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device": str(device),
        "gpu_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "gpu_total_memory_bytes": (
            torch.cuda.get_device_properties(device).total_memory
            if device.type == "cuda"
            else None
        ),
        "cuda_memory_fraction": (
            spec.cuda_memory_fraction if device.type == "cuda" else None
        ),
    }
    train_base, evaluation_base, evaluation_subset = _build_job_datasets(spec)
    if spec.semantic_condition is None:
        train_dataset = train_base
        evaluation_dataset = evaluation_base
    else:
        train_dataset = ActorStateControlDataset(
            train_base,
            control=spec.training_control,
            seed=spec.seed,
        )
        evaluation_dataset = ActorStateControlDataset(
            evaluation_base,
            control=spec.training_control,
            seed=spec.seed,
        )
    arm = STATIC_ARMS[spec.spatial_arm]
    if spec.semantic_condition is None:
        model = OwnershipDecoder(
            input_channels=arm.input_channels,
            width=spec.width,
            residual_blocks=spec.residual_blocks,
        )
    else:
        model = SemanticOwnershipDecoder(
            input_channels=arm.input_channels,
            semantic_dim=5120,
            width=spec.width,
            residual_blocks=spec.residual_blocks,
        )
    try:
        training = train_decoder(
            model,
            train_dataset,
            evaluation_dataset,
            config=TrainingConfig(
                learning_rate=spec.learning_rate,
                weight_decay=spec.weight_decay,
                max_epochs=spec.max_epochs,
                patience=spec.patience,
                gradient_accumulation=spec.gradient_accumulation,
                seed=spec.seed,
                device=spec.device,
                use_amp=spec.use_amp,
                checkpoint_directory=run_root / "checkpoints",
            ),
        )
        evaluation_metrics = evaluate_decoder(
            model,
            evaluation_dataset,
            device=device,
            use_amp=spec.use_amp,
        )
        per_frame_evaluation = {}
        for index in range(len(evaluation_dataset)):
            sample = evaluation_dataset[index]
            per_frame_evaluation[str(sample.frame_index)] = evaluate_decoder(
                model,
                [sample],
                device=device,
                use_amp=spec.use_amp,
            )
            del sample
        semantic_controls = None
        swap_metrics = None
        fixed_substitutions = None
        if spec.semantic_condition is not None:
            semantic_controls = {}
            for control in ("real", "random_matched", "zero", "mean"):
                semantic_controls[control] = evaluate_decoder(
                    model,
                    ActorStateControlDataset(
                        evaluation_base,
                        control=control,
                        seed=spec.seed,
                    ),
                    device=device,
                    use_amp=spec.use_amp,
                )
            swap_metrics = evaluate_query_swap(
                model,
                evaluation_base,
                device=device,
                use_amp=spec.use_amp,
            )
            if (
                spec.split == "final"
                and spec.semantic_condition == "action_relational"
                and spec.training_control == "real"
            ):
                fixed_substitutions = {
                    name: evaluate_decoder(
                        model,
                        ActorStateControlDataset(
                            evaluation_base,
                            control="shuffled_clip",
                            replacement_actor_states=pair,
                        ),
                        device=device,
                        use_amp=spec.use_amp,
                    )
                    for name, pair in _fixed_substitution_pairs(spec).items()
                }
        checkpoint_path = training.checkpoint_path
        result = {
            "format": "armbar-exploratory-job-result-v1",
            "supervision_status": ARMBAR_SUPERVISION_STATUS,
            "run_spec_sha256": _sha256(spec_path),
            "run_name": spec.run_name,
            "spatial_arm": spec.spatial_arm,
            "split": spec.split,
            "evaluation_subset": evaluation_subset,
            "semantic_condition": spec.semantic_condition,
            "language_layer": spec.language_layer,
            "semantic_context": spec.semantic_context,
            "thinking_mode": spec.thinking_mode,
            "training_control": spec.training_control,
            "runtime": runtime,
            "sample_counts": {
                "train": len(train_dataset),
                "evaluation": len(evaluation_dataset),
            },
            "training": {
                "best_epoch": training.best_epoch,
                "stopped_epoch": training.stopped_epoch,
                "best_validation_metrics": training.best_metrics,
                "history": training.history,
                "peak_vram_bytes": training.peak_vram_bytes,
                "peak_host_rss_bytes": training.peak_host_rss_bytes,
                "checkpoint_path": (
                    str(checkpoint_path.resolve()) if checkpoint_path else None
                ),
                "checkpoint_sha256": (
                    _sha256(checkpoint_path) if checkpoint_path else None
                ),
            },
            "evaluation_metrics": evaluation_metrics,
            "per_frame_evaluation": per_frame_evaluation,
            "semantic_controls": semantic_controls,
            "swap_metrics": swap_metrics,
            "fixed_substitutions": fixed_substitutions,
        }
        result_path = run_root / "result.json"
        if result_path.exists():
            raise RuntimeError("refusing to overwrite armbar exploratory result")
        _atomic_json(result_path, result)
        _atomic_json(completion_path, {"result_sha256": _sha256(result_path)})
        return _load_completed_job(run_root, expected_spec)
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
