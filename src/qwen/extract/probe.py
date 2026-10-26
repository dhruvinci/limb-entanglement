from __future__ import annotations

from dataclasses import dataclass


LABEL_BACKGROUND = 0
LABEL_A1 = 1
LABEL_A2 = 2
LABEL_IGNORE = 255


@dataclass
class ProbeResult:
    model: object
    metrics: dict[str, float]
    swapped_metrics: dict[str, float]
    losses: list[float]
    logits: object | None = None
    swapped_logits: object | None = None


def _resize_fraction(mask, output_hw: tuple[int, int]):
    import numpy as np
    from PIL import Image

    values = np.asarray(mask, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("ownership masks must be two-dimensional")
    output_h, output_w = output_hw
    if output_h < 1 or output_w < 1:
        raise ValueError("output dimensions must be positive")
    image = Image.fromarray(values, mode="F")
    return np.asarray(image.resize((output_w, output_h), Image.Resampling.BOX), dtype=np.float32)


def build_coarse_ownership_labels(
    a1_mask,
    a2_mask,
    *,
    possible_foreground=None,
    output_hw: tuple[int, int] = (30, 30),
    actor_min_fraction: float = 0.70,
    competing_max_fraction: float = 0.05,
    background_max_fraction: float = 0.02,
):
    """Convert conservative actor masks into a coarse ownership field.

    ``possible_foreground`` should cover every location where either athlete may
    appear (for example, the union of independently reviewed actor boxes).  A
    missing SAM pixel inside that region remains ignored rather than becoming a
    false background label.
    """
    import numpy as np

    a1 = np.asarray(a1_mask, dtype=bool)
    a2 = np.asarray(a2_mask, dtype=bool)
    if a1.shape != a2.shape or a1.ndim != 2:
        raise ValueError("A1 and A2 masks must have the same two-dimensional shape")
    if not 0.0 <= competing_max_fraction <= actor_min_fraction <= 1.0:
        raise ValueError("actor thresholds must satisfy 0 <= competing <= actor <= 1")
    if not 0.0 <= background_max_fraction <= 1.0:
        raise ValueError("background threshold must be in [0, 1]")
    possible = (a1 | a2) if possible_foreground is None else np.asarray(possible_foreground, dtype=bool)
    if possible.shape != a1.shape:
        raise ValueError("possible_foreground must match the actor-mask shape")

    a1_fraction = _resize_fraction(a1, output_hw)
    a2_fraction = _resize_fraction(a2, output_hw)
    possible_fraction = _resize_fraction(possible, output_hw)
    labels = np.full(output_hw, LABEL_IGNORE, dtype=np.uint8)
    background = (
        (possible_fraction <= background_max_fraction)
        & (a1_fraction <= background_max_fraction)
        & (a2_fraction <= background_max_fraction)
    )
    labels[background] = LABEL_BACKGROUND
    labels[(a1_fraction >= actor_min_fraction) & (a2_fraction <= competing_max_fraction)] = LABEL_A1
    labels[(a2_fraction >= actor_min_fraction) & (a1_fraction <= competing_max_fraction)] = LABEL_A2
    return labels


def _metrics(logits, labels, *, contact_mask=None, contact_actor: int = LABEL_A2) -> dict[str, float]:
    import torch

    predictions = logits.argmax(dim=-1)
    valid = labels != 255
    correct = ((predictions == labels) & valid).sum().item()
    total = valid.sum().item()
    result = {"accuracy": correct / max(1, total)}
    for actor in (1, 2):
        pred = (predictions == actor) & valid
        truth = (labels == actor) & valid
        intersection = (pred & truth).sum().item()
        union = (pred | truth).sum().item()
        result[f"A{actor}_iou"] = intersection / max(1, union)
        result[f"A{actor}_dice"] = 2 * intersection / max(1, pred.sum().item() + truth.sum().item())
    background = labels == 0
    result["background_stability"] = ((predictions == 0) & background).sum().item() / max(1, background.sum().item())
    probability = torch.softmax(logits, dim=-1)
    if contact_mask is not None:
        contact = torch.as_tensor(contact_mask, dtype=torch.bool, device=labels.device)
        if contact.shape != labels.shape:
            raise ValueError("contact mask must match labels")
    else:
        contact = torch.zeros_like(labels, dtype=torch.bool)
    if contact_actor not in (LABEL_A1, LABEL_A2):
        raise ValueError("contact actor must be A1 or A2")
    other_actor = LABEL_A2 if contact_actor == LABEL_A1 else LABEL_A1
    if contact.any():
        result["contact_patch_count"] = float(contact.sum().item())
        result["contact_actor_margin"] = float(
            (probability[..., contact_actor][contact] - probability[..., other_actor][contact]).mean()
        )
        result["contact_actor_accuracy"] = float((predictions[contact] == contact_actor).float().mean())
    else:
        result["contact_patch_count"] = 0.0
        result["contact_actor_margin"] = 0.0
        result["contact_actor_accuracy"] = 0.0
    return result


def _balanced_class_weights(labels):
    import torch

    valid = labels != LABEL_IGNORE
    counts = torch.stack([((labels == index) & valid).sum() for index in range(3)]).float()
    weights = valid.sum().float() / (3.0 * counts.clamp_min(1.0))
    return weights / weights.mean().clamp_min(1e-8)


def train_vision_probe(
    vision,
    labels,
    epochs: int = 300,
    seed: int = 380027,
    rank: int = 32,
    evaluation: tuple | None = None,
    contact_mask=None,
    batch_size: int | None = None,
    device=None,
) -> ProbeResult:
    import torch
    from torch import nn

    if vision.ndim != 3 or labels.ndim != 2 or vision.shape[:2] != labels.shape:
        raise ValueError("expected compatible vision [N,P,D] and labels [N,P]")
    torch.manual_seed(seed)

    class VisionProbe(nn.Module):
        def __init__(self):
            super().__init__()
            self.vision = nn.Linear(vision.shape[-1], rank)
            self.head = nn.Linear(rank, 3)

        def forward(self, spatial):
            return self.head(torch.tanh(self.vision(spatial)))

    target_device = torch.device(device) if device is not None else vision.device
    model = VisionProbe().to(target_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03, weight_decay=1e-4)
    loss_function = nn.CrossEntropyLoss(
        ignore_index=LABEL_IGNORE,
        weight=_balanced_class_weights(labels).to(target_device),
    )
    losses = []
    model.train()
    if batch_size is None:
        for _ in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            logits = model(vision.to(target_device).float())
            loss = loss_function(logits.reshape(-1, 3), labels.to(target_device).reshape(-1).long())
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
    else:
        if batch_size < 1:
            raise ValueError("batch size must be positive")
        flat_vision = vision.reshape(-1, vision.shape[-1])
        flat_labels = labels.reshape(-1).cpu()
        valid_indices = torch.nonzero(flat_labels != LABEL_IGNORE, as_tuple=False).flatten()
        generator = torch.Generator().manual_seed(seed)
        for _ in range(epochs):
            order = valid_indices[torch.randperm(len(valid_indices), generator=generator)]
            total_loss = 0.0
            for start in range(0, len(order), batch_size):
                indices = order[start:start + batch_size]
                optimizer.zero_grad(set_to_none=True)
                logits = model(flat_vision[indices.to(flat_vision.device)].to(target_device).float())
                loss = loss_function(logits, flat_labels[indices].to(target_device).long())
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach()) * len(indices)
            losses.append(total_loss / max(1, len(order)))
    model.eval()
    eval_vision, eval_labels = evaluation or (vision, labels)
    with torch.inference_mode():
        if batch_size is None:
            logits = model(eval_vision.to(target_device).float()).detach().cpu()
        else:
            flat_eval = eval_vision.reshape(-1, eval_vision.shape[-1])
            chunks = [
                model(flat_eval[start:start + batch_size].to(target_device).float()).detach().cpu()
                for start in range(0, len(flat_eval), batch_size)
            ]
            logits = torch.cat(chunks).reshape(*eval_vision.shape[:2], 3)
    metrics = _metrics(logits, eval_labels.cpu(), contact_mask=contact_mask)
    return ProbeResult(
        model=model,
        metrics=metrics,
        swapped_metrics={},
        losses=losses,
        logits=logits.detach().cpu(),
    )


def train_probe(
    vision,
    queries,
    labels,
    epochs: int = 300,
    seed: int = 380027,
    rank: int = 32,
    evaluation: tuple | None = None,
    contact_mask=None,
    batch_size: int | None = None,
    device=None,
) -> ProbeResult:
    import torch
    from torch import nn

    if vision.ndim != 3 or queries.ndim != 3 or labels.ndim != 2:
        raise ValueError("expected vision [N,P,D], queries [N,2,Q], labels [N,P]")
    if vision.shape[:2] != labels.shape or queries.shape[:2] != (vision.shape[0], 2):
        raise ValueError("probe tensors have incompatible sample, pixel, or actor dimensions")
    torch.manual_seed(seed)

    class ActorConditionedProbe(nn.Module):
        def __init__(self):
            super().__init__()
            self.vision = nn.Linear(vision.shape[-1], rank)
            self.query = nn.Linear(queries.shape[-1], rank, bias=False)
            self.context = nn.Linear(queries.shape[-1], rank, bias=False)
            self.actor_head = nn.Linear(rank, 1)
            self.background = nn.Linear(vision.shape[-1], 1)

        def forward(self, spatial, actor_queries):
            pair_mean = actor_queries.mean(dim=1)
            actor_residual = torch.nn.functional.normalize(actor_queries - pair_mean[:, None, :], dim=-1)
            shared_context = torch.nn.functional.normalize(pair_mean, dim=-1)
            visual = torch.tanh(self.vision(spatial) + self.context(shared_context)[:, None, :])
            semantic = torch.tanh(self.query(actor_residual))
            fused = visual[:, None, :, :] * semantic[:, :, None, :]
            actor_logits = self.actor_head(fused).squeeze(-1).transpose(1, 2)
            background_logit = self.background(spatial)
            return torch.cat((background_logit, actor_logits), dim=-1)

    target_device = torch.device(device) if device is not None else vision.device
    model = ActorConditionedProbe().to(target_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03, weight_decay=1e-4)
    loss_function = nn.CrossEntropyLoss(
        ignore_index=LABEL_IGNORE,
        weight=_balanced_class_weights(labels).to(target_device),
    )
    losses = []
    model.train()
    if batch_size is None:
        for _ in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            logits = model(vision.to(target_device).float(), queries.to(target_device).float())
            loss = loss_function(logits.reshape(-1, 3), labels.to(target_device).reshape(-1).long())
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
    else:
        if batch_size < 1:
            raise ValueError("batch size must be positive")
        pixels_per_sample = vision.shape[1]
        flat_vision = vision.reshape(-1, vision.shape[-1])
        flat_labels = labels.reshape(-1).cpu()
        valid_indices = torch.nonzero(flat_labels != LABEL_IGNORE, as_tuple=False).flatten()
        generator = torch.Generator().manual_seed(seed)
        for _ in range(epochs):
            order = valid_indices[torch.randperm(len(valid_indices), generator=generator)]
            total_loss = 0.0
            for start in range(0, len(order), batch_size):
                indices = order[start:start + batch_size]
                sample_indices = torch.div(indices, pixels_per_sample, rounding_mode="floor")
                spatial = flat_vision[indices.to(flat_vision.device)].to(target_device).float()[:, None, :]
                actor_queries = queries[sample_indices.to(queries.device)].to(target_device).float()
                optimizer.zero_grad(set_to_none=True)
                logits = model(spatial, actor_queries).squeeze(1)
                loss = loss_function(logits, flat_labels[indices].to(target_device).long())
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach()) * len(indices)
            losses.append(total_loss / max(1, len(order)))
    model.eval()
    eval_vision, eval_queries, eval_labels = evaluation or (vision, queries, labels)
    with torch.inference_mode():
        if batch_size is None:
            logits = model(eval_vision.to(target_device).float(), eval_queries.to(target_device).float()).detach().cpu()
            swapped = model(
                eval_vision.to(target_device).float(), eval_queries.flip(1).to(target_device).float()
            ).detach().cpu()
        else:
            eval_pixels = eval_vision.shape[1]
            flat_eval = eval_vision.reshape(-1, eval_vision.shape[-1])
            logits_chunks, swapped_chunks = [], []
            for start in range(0, len(flat_eval), batch_size):
                stop = min(len(flat_eval), start + batch_size)
                flat_indices = torch.arange(start, stop)
                sample_indices = torch.div(flat_indices, eval_pixels, rounding_mode="floor")
                spatial = flat_eval[start:stop].to(target_device).float()[:, None, :]
                actor_queries = eval_queries[sample_indices.to(eval_queries.device)].to(target_device).float()
                logits_chunks.append(model(spatial, actor_queries).squeeze(1).detach().cpu())
                swapped_chunks.append(model(spatial, actor_queries.flip(1)).squeeze(1).detach().cpu())
            logits = torch.cat(logits_chunks).reshape(*eval_vision.shape[:2], 3)
            swapped = torch.cat(swapped_chunks).reshape(*eval_vision.shape[:2], 3)
    return ProbeResult(
        model=model,
        metrics=_metrics(logits, eval_labels.cpu(), contact_mask=contact_mask),
        swapped_metrics=_metrics(swapped, eval_labels.cpu(), contact_mask=contact_mask),
        losses=losses,
        logits=logits.detach().cpu(),
        swapped_logits=swapped.detach().cpu(),
    )


def evaluate_actor_probe(
    model,
    vision,
    queries,
    labels,
    *,
    contact_mask=None,
    batch_size: int | None = None,
    device=None,
) -> ProbeResult:
    """Evaluate one trained decoder with a different actor-query pair.

    Keeping the decoder weights fixed separates changes caused by the query
    representation from improvements that arise when each query condition is
    allowed to train its own decoder.
    """
    import torch

    if vision.ndim != 3 or queries.ndim != 3 or labels.ndim != 2:
        raise ValueError("expected vision [N,P,D], queries [N,2,Q], labels [N,P]")
    if vision.shape[:2] != labels.shape or queries.shape[:2] != (vision.shape[0], 2):
        raise ValueError("probe tensors have incompatible sample, pixel, or actor dimensions")
    if batch_size is not None and batch_size < 1:
        raise ValueError("batch size must be positive")

    target_device = torch.device(device) if device is not None else next(model.parameters()).device
    model = model.to(target_device)
    model.eval()
    with torch.inference_mode():
        if batch_size is None:
            logits = model(vision.to(target_device).float(), queries.to(target_device).float()).detach().cpu()
        else:
            pixels_per_sample = vision.shape[1]
            flat_vision = vision.reshape(-1, vision.shape[-1])
            chunks = []
            for start in range(0, len(flat_vision), batch_size):
                stop = min(len(flat_vision), start + batch_size)
                flat_indices = torch.arange(start, stop)
                sample_indices = torch.div(flat_indices, pixels_per_sample, rounding_mode="floor")
                spatial = flat_vision[start:stop].to(target_device).float()[:, None, :]
                actor_queries = queries[sample_indices.to(queries.device)].to(target_device).float()
                chunks.append(model(spatial, actor_queries).squeeze(1).detach().cpu())
            logits = torch.cat(chunks).reshape(*vision.shape[:2], 3)
    return ProbeResult(
        model=model,
        metrics=_metrics(logits, labels.cpu(), contact_mask=contact_mask),
        swapped_metrics={},
        losses=[],
        logits=logits,
    )


def run_cached_probe(run_root, packet_dir, vision_layer: int = 11, language_layer: int = 63, epochs: int = 120):
    import json
    from pathlib import Path

    import numpy as np
    import torch
    from PIL import Image
    from safetensors.torch import load_file, save_file

    run_root, packet_dir = Path(run_root), Path(packet_dir)
    ownership = json.loads((packet_dir / "ownership/ownership-manifest.json").read_text())
    temporal_queries = torch.stack([
        load_file(
            run_root / f"semantic/video/4fps/action_relational/off/{actor}.safetensors"
        )["marker_states"][language_layer].float()
        for actor in ("A1", "A2")
    ])

    def load_subset(subset: str):
        spatial, semantic, labels = [], [], []
        for record in ownership["frames"]:
            if record["subset"] != subset:
                continue
            frame = record["frame_index"]
            spatial.append(load_file(run_root / f"spatial/pooled/layer_{vision_layer:02d}/frame_{frame:06d}.safetensors")["hidden"].float())
            semantic.append(temporal_queries)
            label_image = Image.open(packet_dir / record["label_path"]).resize((30, 30), Image.Resampling.NEAREST)
            labels.append(torch.from_numpy(np.asarray(label_image, dtype=np.uint8).copy()).reshape(-1).long())
        return torch.stack(spatial), torch.stack(semantic), torch.stack(labels)

    train = load_subset("train")
    test = load_subset("test")
    result = train_probe(*train, epochs=epochs, evaluation=test)
    output = run_root / "diagnostics/quick-probe"
    output.mkdir(parents=True, exist_ok=True)
    save_file({key: value.detach().cpu() for key, value in result.model.state_dict().items()}, output / "probe.safetensors")
    report = {
        "vision_layer": vision_layer, "language_layer": language_layer,
        "query_source": "Qwen self-grounded native-video 4fps context",
        "train_frames": 24, "held_out_test_frames": 8,
        "metrics": result.metrics, "swapped_metrics": result.swapped_metrics,
        "final_loss": result.losses[-1],
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report
