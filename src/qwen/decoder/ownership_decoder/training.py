from __future__ import annotations

import copy
import random
import resource
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import torch
from torch import nn

from .checkpoint import latest_valid_checkpoint, load_checkpoint, prune_checkpoints, save_checkpoint
from .data import OwnershipSample
from .losses import semantic_ownership_loss
from .metrics import OwnershipMetricAccumulator, SwapMetricAccumulator


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    max_epochs: int = 100
    patience: int = 12
    gradient_accumulation: int = 4
    dice_weight: float = 0.5
    seed: int = 7
    device: str = "cuda"
    use_amp: bool = True
    checkpoint_directory: Path | None = None


@dataclass
class TrainingResult:
    history: list[dict]
    best_epoch: int
    best_metrics: dict[str, float]
    stopped_epoch: int
    checkpoint_path: Path | None
    peak_vram_bytes: int
    peak_host_rss_bytes: int


def _validate_config(config: TrainingConfig) -> None:
    if config.learning_rate <= 0 or config.weight_decay < 0:
        raise ValueError("optimizer settings must be non-negative with positive learning rate")
    if config.max_epochs < 1 or config.patience < 1 or config.gradient_accumulation < 1:
        raise ValueError("epochs, patience, and gradient accumulation must be positive")


def _training_signature(config: TrainingConfig, model: nn.Module) -> dict:
    return {
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "gradient_accumulation": config.gradient_accumulation,
        "dice_weight": config.dice_weight,
        "seed": config.seed,
        "model_shapes": {name: list(value.shape) for name, value in model.state_dict().items()},
    }


def _class_weights(dataset: Sequence[OwnershipSample]) -> torch.Tensor:
    counts = torch.zeros(3, dtype=torch.float64)
    valid_total = 0
    label_loader = getattr(dataset, "load_labels", None)
    for index in range(len(dataset)):
        labels = label_loader(index) if callable(label_loader) else dataset[index].labels
        if labels is None:
            raise ValueError("training samples require ownership labels")
        valid = labels != 255
        valid_total += int(valid.sum().item())
        for actor_class in range(3):
            counts[actor_class] += int(((labels == actor_class) & valid).sum().item())
    if valid_total == 0:
        raise ValueError("training data must contain labeled pixels")
    weights = float(valid_total) / (3.0 * counts.clamp_min(1.0))
    return (weights / weights.mean().clamp_min(1e-8)).float()


def _batched_spatial(value: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Preserve dtype and standard NCHW layout, packing cache views on device."""
    if value.ndim == 3 and not value.is_contiguous():
        hwc = value.permute(1, 2, 0)
        if hwc.is_contiguous():
            # Cache storage is already contiguous HWC: transfer it directly,
            # then reorder on the destination, with one source temporary at a time.
            transferred = hwc.to(device=device, non_blocking=False)
            return transferred.permute(2, 0, 1).contiguous().unsqueeze(0)
    return value.unsqueeze(0).to(device=device, non_blocking=False).contiguous()


def _batched_sample(
    sample: OwnershipSample,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if sample.labels is None:
        raise ValueError("decoder samples require ownership labels")
    spatial = {
        name: _batched_spatial(value, device)
        for name, value in sample.spatial.items()
    }
    if sample.rgb is not None:
        if "rgb" in spatial:
            raise ValueError("RGB input may not duplicate a spatial source named 'rgb'")
        spatial["rgb"] = sample.rgb.unsqueeze(0).to(device=device, non_blocking=False)
    labels = sample.labels.unsqueeze(0).to(device=device, non_blocking=False)
    contact_value = sample.contact
    if contact_value is None:
        contact_value = torch.zeros_like(sample.labels, dtype=torch.bool)
    contact = contact_value.unsqueeze(0).to(device=device, non_blocking=False)
    actor_states = (
        sample.actor_states.unsqueeze(0).to(device=device, non_blocking=False)
        if sample.actor_states is not None
        else None
    )
    return spatial, labels, contact, actor_states


def _forward_decoder(
    model: nn.Module,
    spatial: dict[str, torch.Tensor],
    actor_states: torch.Tensor | None,
    output_size: tuple[int, int],
) -> torch.Tensor:
    if getattr(model, "requires_actor_states", False):
        if actor_states is None:
            raise ValueError("semantic decoder requires actor states in every sample")
        return model(spatial, actor_states=actor_states, output_size=output_size)
    return model(spatial, output_size=output_size)


def evaluate_decoder(
    model: nn.Module,
    dataset: Sequence[OwnershipSample],
    *,
    device: torch.device,
    use_amp: bool,
) -> dict[str, float]:
    model.eval()
    accumulator = OwnershipMetricAccumulator()
    amp_enabled = use_amp and device.type == "cuda"
    with torch.inference_mode():
        for index in range(len(dataset)):
            spatial, labels, contact, actor_states = _batched_sample(dataset[index], device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                logits = _forward_decoder(
                    model,
                    spatial,
                    actor_states,
                    tuple(labels.shape[-2:]),
                )
            accumulator.update(logits.float(), labels, contact)
    return accumulator.compute()


def evaluate_query_swap(
    model: nn.Module,
    dataset: Sequence[OwnershipSample],
    *,
    device: torch.device,
    use_amp: bool,
) -> dict[str, float]:
    """Stream exact actor-query swap behavior without retaining frame logits."""

    if not getattr(model, "requires_actor_states", False):
        raise ValueError("query-swap evaluation requires a semantic decoder")
    model.eval()
    accumulator = SwapMetricAccumulator()
    amp_enabled = use_amp and device.type == "cuda"
    with torch.inference_mode():
        for index in range(len(dataset)):
            spatial, labels, _, actor_states = _batched_sample(dataset[index], device)
            if actor_states is None:
                raise ValueError("query-swap evaluation requires actor states")
            output_size = tuple(labels.shape[-2:])
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                ordinary = _forward_decoder(
                    model,
                    spatial,
                    actor_states,
                    output_size,
                )
                swapped = _forward_decoder(
                    model,
                    spatial,
                    actor_states.flip(1),
                    output_size,
                )
            accumulator.update(ordinary.float(), swapped.float())
    return accumulator.compute()


def _cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu() for name, value in model.state_dict().items()}


def _optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def train_decoder(
    model: nn.Module,
    train_dataset: Sequence[OwnershipSample],
    validation_dataset: Sequence[OwnershipSample],
    *,
    config: TrainingConfig,
    epoch_callback: Callable[[dict], None] | None = None,
) -> TrainingResult:
    """Train with batch one, deterministic accumulation, early stopping, and exact resume."""

    _validate_config(config)
    if len(train_dataset) < 1 or len(validation_dataset) < 1:
        raise ValueError("training and validation datasets must both be non-empty")
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.benchmark = False

    device = torch.device(config.device)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    amp_enabled = config.use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    class_weights = _class_weights(train_dataset).to(device)
    signature = _training_signature(config, model)

    history: list[dict] = []
    best_epoch = -1
    best_score = float("-inf")
    best_metrics: dict[str, float] = {}
    bad_epochs = 0
    start_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    checkpoint_path: Path | None = None

    if config.checkpoint_directory is not None:
        checkpoint_path = latest_valid_checkpoint(config.checkpoint_directory)
        if checkpoint_path is not None:
            restored, _ = load_checkpoint(checkpoint_path)
            if restored.get("signature") != signature:
                raise ValueError("checkpoint training signature does not match this run")
            model.load_state_dict(restored["model"])
            optimizer.load_state_dict(restored["optimizer"])
            _optimizer_to_device(optimizer, device)
            scaler.load_state_dict(restored.get("scaler", {}))
            history = list(restored["history"])
            best_epoch = int(restored["best_epoch"])
            best_score = float(restored["best_score"])
            best_metrics = dict(restored["best_metrics"])
            bad_epochs = int(restored["bad_epochs"])
            start_epoch = int(restored["epoch"]) + 1
            del restored

    stopped_epoch = start_epoch - 1
    for epoch in range(start_epoch, config.max_epochs):
        model.train()
        order = list(range(len(train_dataset)))
        random.Random(config.seed + epoch).shuffle(order)
        epoch_loss = 0.0
        epoch_items = 0
        for group_start in range(0, len(order), config.gradient_accumulation):
            group = order[group_start : group_start + config.gradient_accumulation]
            optimizer.zero_grad(set_to_none=True)
            for index in group:
                spatial, labels, contact, actor_states = _batched_sample(train_dataset[index], device)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                    output_size = tuple(labels.shape[-2:])
                    logits = _forward_decoder(model, spatial, actor_states, output_size)
                    swapped_logits = None
                    if getattr(model, "requires_actor_states", False):
                        assert actor_states is not None
                        swapped_logits = _forward_decoder(
                            model,
                            spatial,
                            actor_states.flip(1),
                            output_size,
                        )
                    losses = semantic_ownership_loss(
                        logits,
                        labels,
                        contact,
                        swapped_logits=swapped_logits,
                        class_weights=class_weights,
                        dice_weight=config.dice_weight,
                    )
                    scaled_loss = losses["total"] / len(group)
                scaler.scale(scaled_loss).backward()
                epoch_loss += float(losses["total"].detach())
                epoch_items += 1
            scaler.step(optimizer)
            scaler.update()

        validation_metrics = evaluate_decoder(
            model,
            validation_dataset,
            device=device,
            use_amp=config.use_amp,
        )
        score = validation_metrics["macro_actor_iou"]
        improved = score > best_score + 1e-6
        if improved:
            best_epoch = epoch
            best_score = score
            best_metrics = validation_metrics
            bad_epochs = 0
            if config.checkpoint_directory is None:
                best_state = copy.deepcopy(_cpu_state_dict(model))
        else:
            bad_epochs += 1
        record = {
            "epoch": epoch,
            "train_loss": epoch_loss / max(1, epoch_items),
            "validation": validation_metrics,
            "improved": improved,
        }
        history.append(record)
        stopped_epoch = epoch

        if config.checkpoint_directory is not None:
            checkpoint_path = Path(config.checkpoint_directory) / f"epoch_{epoch:04d}.pt"
            save_checkpoint(
                checkpoint_path,
                {
                    "model": _cpu_state_dict(model),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "epoch": epoch,
                    "history": history,
                    "best_epoch": best_epoch,
                    "best_score": best_score,
                    "best_metrics": best_metrics,
                    "bad_epochs": bad_epochs,
                    "signature": signature,
                },
                metadata={"epoch": epoch, "validation": validation_metrics, "improved": improved},
            )
            prune_checkpoints(
                config.checkpoint_directory,
                keep_epochs={best_epoch},
                keep_latest=1,
            )
        if epoch_callback is not None:
            epoch_callback(record)
        if bad_epochs >= config.patience:
            break

    if best_epoch < 0:
        raise RuntimeError("training produced no evaluable epoch")
    if config.checkpoint_directory is not None:
        best_path = Path(config.checkpoint_directory) / f"epoch_{best_epoch:04d}.pt"
        best_checkpoint, _ = load_checkpoint(best_path)
        model.load_state_dict(best_checkpoint["model"])
        del best_checkpoint
        checkpoint_path = best_path
    elif best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)
    best_metrics = evaluate_decoder(
        model,
        validation_dataset,
        device=device,
        use_amp=config.use_amp,
    )
    peak_vram = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    peak_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    return TrainingResult(
        history=history,
        best_epoch=best_epoch,
        best_metrics=best_metrics,
        stopped_epoch=stopped_epoch,
        checkpoint_path=checkpoint_path,
        peak_vram_bytes=peak_vram,
        peak_host_rss_bytes=peak_rss,
    )
