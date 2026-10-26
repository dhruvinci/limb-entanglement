from __future__ import annotations

import torch


IGNORE_INDEX = 255


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / max(1.0, float(denominator))


def _connected_component_means(
    values: torch.Tensor,
    selected: torch.Tensor,
) -> list[float]:
    """Return 4-connected component means for a small two-dimensional mask."""

    values = values.detach().float().cpu()
    selected = selected.detach().bool().cpu()
    if values.ndim != 2 or selected.shape != values.shape:
        raise ValueError("contact component values and mask must share a 2D shape")
    visited = torch.zeros_like(selected)
    height, width = selected.shape
    means = []
    for row, column in selected.nonzero(as_tuple=False).tolist():
        if visited[row, column]:
            continue
        stack = [(row, column)]
        visited[row, column] = True
        component_values = []
        while stack:
            current_row, current_column = stack.pop()
            component_values.append(float(values[current_row, current_column]))
            for neighbor_row, neighbor_column in (
                (current_row - 1, current_column),
                (current_row + 1, current_column),
                (current_row, current_column - 1),
                (current_row, current_column + 1),
            ):
                if (
                    0 <= neighbor_row < height
                    and 0 <= neighbor_column < width
                    and selected[neighbor_row, neighbor_column]
                    and not visited[neighbor_row, neighbor_column]
                ):
                    visited[neighbor_row, neighbor_column] = True
                    stack.append((neighbor_row, neighbor_column))
        means.append(sum(component_values) / len(component_values))
    return means


class OwnershipMetricAccumulator:
    """Exact streaming metrics for batches whose spatial grids may differ."""

    def __init__(self) -> None:
        self.confusion = torch.zeros((3, 3), dtype=torch.int64)
        self.contact_pixels = 0
        self.contact_correct = 0
        self.contact_margin_sum = 0.0
        self.contact_positive_pixels = 0
        self.contact_regions = 0
        self.contact_positive_regions = 0

    def update(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        contact: torch.Tensor | None = None,
    ) -> None:
        if logits.ndim != 4 or logits.shape[1] != 3:
            raise ValueError("logits must have shape [B,3,H,W]")
        if labels.shape != (logits.shape[0], logits.shape[2], logits.shape[3]):
            raise ValueError("labels must have shape [B,H,W] matching logits")
        if contact is None:
            contact = torch.zeros_like(labels, dtype=torch.bool)
        if contact.shape != labels.shape:
            raise ValueError("contact must match label shape")
        labels = labels.long()
        contact = contact.bool()
        contact_labels = labels[contact]
        if contact_labels.numel() and not ((contact_labels == 1) | (contact_labels == 2)).all():
            raise ValueError("every contact pixel must have actor-owned truth (A1 or A2)")

        predictions = logits.argmax(dim=1)
        valid = labels != IGNORE_INDEX
        encoded = 3 * labels[valid] + predictions[valid]
        self.confusion += torch.bincount(encoded.detach().cpu(), minlength=9).reshape(3, 3)

        if contact.any():
            probabilities = torch.softmax(logits.float(), dim=1).permute(0, 2, 3, 1)
            owner_indices = labels.clamp(0, 2).unsqueeze(-1)
            other_indices = (3 - labels).clamp(0, 2).unsqueeze(-1)
            margins = (
                probabilities.gather(-1, owner_indices).squeeze(-1)
                - probabilities.gather(-1, other_indices).squeeze(-1)
            )
            contact_margins = margins[contact]
            self.contact_pixels += int(contact_margins.numel())
            self.contact_correct += int((predictions[contact] == labels[contact]).sum().item())
            self.contact_margin_sum += float(contact_margins.sum().item())
            self.contact_positive_pixels += int((contact_margins > 0).sum().item())
            for batch_index in range(labels.shape[0]):
                for actor in (1, 2):
                    actor_contact = contact[batch_index] & (
                        labels[batch_index] == actor
                    )
                    region_margins = _connected_component_means(
                        margins[batch_index],
                        actor_contact,
                    )
                    self.contact_regions += len(region_margins)
                    self.contact_positive_regions += sum(
                        value > 0.0 for value in region_margins
                    )

    def compute(self) -> dict[str, float]:
        total = int(self.confusion.sum().item())
        result: dict[str, float] = {
            "accuracy": _safe_ratio(int(self.confusion.diag().sum().item()), total),
        }
        actor_ious = []
        for actor in (1, 2):
            intersection = int(self.confusion[actor, actor].item())
            true_count = int(self.confusion[actor, :].sum().item())
            predicted_count = int(self.confusion[:, actor].sum().item())
            union = true_count + predicted_count - intersection
            result[f"a{actor}_iou"] = _safe_ratio(intersection, union)
            result[f"a{actor}_dice"] = _safe_ratio(2 * intersection, true_count + predicted_count)
            actor_ious.append(result[f"a{actor}_iou"])
        result["macro_actor_iou"] = sum(actor_ious) / 2.0
        result["background_stability"] = _safe_ratio(
            int(self.confusion[0, 0].item()),
            int(self.confusion[0, :].sum().item()),
        )
        result["contact_pixel_count"] = float(self.contact_pixels)
        result["contact_accuracy"] = _safe_ratio(self.contact_correct, self.contact_pixels)
        result["contact_margin"] = _safe_ratio(self.contact_margin_sum, self.contact_pixels)
        result["positive_contact_margin_fraction"] = _safe_ratio(
            self.contact_positive_pixels,
            self.contact_pixels,
        )
        result["contact_region_count"] = float(self.contact_regions)
        result["positive_contact_region_fraction"] = _safe_ratio(
            self.contact_positive_regions,
            self.contact_regions,
        )
        return result


def ownership_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    contact: torch.Tensor | None = None,
) -> dict[str, float]:
    """Compute globally aggregated ownership and reviewed-contact metrics."""

    accumulator = OwnershipMetricAccumulator()
    accumulator.update(logits, labels, contact)
    return accumulator.compute()


class SwapMetricAccumulator:
    """Pixel-weighted query-swap metrics for frames with different grids."""

    def __init__(self) -> None:
        self.actor_pixels = 0
        self.actor_flips = 0
        self.background_delta_sum = 0.0
        self.background_pixels = 0
        self.actor_swap_error_sum = 0.0
        self.actor_probability_values = 0

    def update(self, logits: torch.Tensor, swapped_logits: torch.Tensor) -> None:
        if logits.shape != swapped_logits.shape or logits.ndim != 4 or logits.shape[1] != 3:
            raise ValueError("ordinary and swapped logits must share shape [B,3,H,W]")
        probabilities = torch.softmax(logits.float(), dim=1)
        swapped_probabilities = torch.softmax(swapped_logits.float(), dim=1)
        predictions = probabilities.argmax(dim=1)
        swapped_predictions = swapped_probabilities.argmax(dim=1)
        actor_pixels = (predictions == 1) | (predictions == 2)
        expected = 3 - predictions
        self.actor_pixels += int(actor_pixels.sum().item())
        self.actor_flips += int(
            (swapped_predictions[actor_pixels] == expected[actor_pixels]).sum().item()
        )
        background_delta = (swapped_probabilities[:, 0] - probabilities[:, 0]).abs()
        self.background_delta_sum += float(background_delta.sum().item())
        self.background_pixels += background_delta.numel()
        actor_error = torch.stack(
            (
                (swapped_probabilities[:, 1] - probabilities[:, 2]).abs(),
                (swapped_probabilities[:, 2] - probabilities[:, 1]).abs(),
            )
        )
        self.actor_swap_error_sum += float(actor_error.sum().item())
        self.actor_probability_values += actor_error.numel()

    def compute(self) -> dict[str, float]:
        return {
            "actor_pixel_count": float(self.actor_pixels),
            "actor_prediction_flip_fraction": _safe_ratio(
                self.actor_flips,
                self.actor_pixels,
            ),
            "background_pixel_count": float(self.background_pixels),
            "background_probability_delta": _safe_ratio(
                self.background_delta_sum,
                self.background_pixels,
            ),
            "actor_probability_value_count": float(self.actor_probability_values),
            "actor_probability_swap_error": _safe_ratio(
                self.actor_swap_error_sum,
                self.actor_probability_values,
            ),
        }


def swap_response_metrics(logits: torch.Tensor, swapped_logits: torch.Tensor) -> dict[str, float]:
    """Measure whether swapping only A1/A2 queries swaps ownership, not background."""

    if logits.shape != swapped_logits.shape or logits.ndim != 4 or logits.shape[1] != 3:
        raise ValueError("ordinary and swapped logits must share shape [B,3,H,W]")
    probabilities = torch.softmax(logits.float(), dim=1)
    swapped_probabilities = torch.softmax(swapped_logits.float(), dim=1)
    predictions = probabilities.argmax(dim=1)
    swapped_predictions = swapped_probabilities.argmax(dim=1)
    actor_pixels = (predictions == 1) | (predictions == 2)
    expected_swapped_predictions = 3 - predictions
    actor_prediction_flip_fraction = (
        float((swapped_predictions[actor_pixels] == expected_swapped_predictions[actor_pixels]).float().mean().item())
        if actor_pixels.any()
        else 0.0
    )
    actor_probability_swap_error = 0.5 * (
        (swapped_probabilities[:, 1] - probabilities[:, 2]).abs().mean()
        + (swapped_probabilities[:, 2] - probabilities[:, 1]).abs().mean()
    )
    return {
        "actor_prediction_flip_fraction": actor_prediction_flip_fraction,
        "background_probability_delta": float(
            (swapped_probabilities[:, 0] - probabilities[:, 0]).abs().mean().item()
        ),
        "actor_probability_swap_error": float(actor_probability_swap_error.item()),
    }
