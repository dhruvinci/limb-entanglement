from __future__ import annotations

import torch
from torch.nn import functional as F


IGNORE_INDEX = 255


def balanced_class_weights(labels: torch.Tensor) -> torch.Tensor:
    valid = labels != IGNORE_INDEX
    if not valid.any():
        raise ValueError("class weights require at least one labeled pixel")
    counts = torch.stack([((labels == actor_class) & valid).sum() for actor_class in range(3)]).float()
    weights = valid.sum().float() / (3.0 * counts.clamp_min(1.0))
    return weights / weights.mean().clamp_min(1e-8)


def ownership_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    class_weights: torch.Tensor | None = None,
    dice_weight: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Ignore-aware FP32 cross-entropy plus multiclass soft Dice."""

    if logits.ndim != 4 or logits.shape[1] != 3:
        raise ValueError("logits must have shape [B,3,H,W]")
    if labels.shape != (logits.shape[0], logits.shape[2], logits.shape[3]):
        raise ValueError("labels must have shape [B,H,W] matching logits")
    valid = labels != IGNORE_INDEX
    if not valid.any():
        raise ValueError("ownership loss requires at least one labeled pixel")
    if dice_weight < 0:
        raise ValueError("dice weight must be non-negative")

    fp32_logits = logits.float()
    labels = labels.long()
    weights = balanced_class_weights(labels) if class_weights is None else class_weights.float()
    weights = weights.to(device=logits.device)
    cross_entropy = F.cross_entropy(
        fp32_logits,
        labels,
        weight=weights,
        ignore_index=IGNORE_INDEX,
    )

    probabilities = torch.softmax(fp32_logits, dim=1)
    safe_labels = labels.clamp(0, 2)
    targets = F.one_hot(safe_labels, num_classes=3).permute(0, 3, 1, 2).float()
    valid_mask = valid.unsqueeze(1).float()
    probabilities = probabilities * valid_mask
    targets = targets * valid_mask
    reduce_dimensions = (0, 2, 3)
    intersection = (probabilities * targets).sum(dim=reduce_dimensions)
    denominator = probabilities.sum(dim=reduce_dimensions) + targets.sum(dim=reduce_dimensions)
    dice = 1.0 - ((2.0 * intersection + 1e-6) / (denominator + 1e-6)).mean()
    total = cross_entropy + float(dice_weight) * dice
    return {"total": total, "cross_entropy": cross_entropy, "dice": dice}


def semantic_ownership_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    contact: torch.Tensor,
    *,
    swapped_logits: torch.Tensor | None = None,
    class_weights: torch.Tensor | None = None,
    dice_weight: float = 0.5,
    contact_weight: float = 2.0,
    swap_weight: float = 0.25,
) -> dict[str, torch.Tensor]:
    """Ownership objective with explicit contact supervision and query-swap consistency."""

    base = ownership_loss(
        logits,
        labels,
        class_weights=class_weights,
        dice_weight=dice_weight,
    )
    if contact.shape != labels.shape:
        raise ValueError("contact must match label shape")
    contact = contact.bool()
    if contact.any():
        contact_labels = labels[contact]
        if not ((contact_labels == 1) | (contact_labels == 2)).all():
            raise ValueError("every contact pixel must have actor-owned truth (A1 or A2)")
        pixel_logits = logits.float().permute(0, 2, 3, 1)[contact]
        contact_cross_entropy = F.cross_entropy(pixel_logits, contact_labels.long())
    else:
        contact_cross_entropy = logits.float().sum() * 0.0

    if swapped_logits is None:
        swap_equivariance = logits.float().sum() * 0.0
    else:
        if swapped_logits.shape != logits.shape:
            raise ValueError("swapped logits must match ordinary logits")
        expected_swap = logits[:, [0, 2, 1]].detach()
        swap_equivariance = F.mse_loss(swapped_logits.float(), expected_swap.float())
    total = (
        base["total"]
        + float(contact_weight) * contact_cross_entropy
        + float(swap_weight) * swap_equivariance
    )
    return {
        "total": total,
        "cross_entropy": base["cross_entropy"],
        "dice": base["dice"],
        "contact_cross_entropy": contact_cross_entropy,
        "swap_equivariance": swap_equivariance,
    }
