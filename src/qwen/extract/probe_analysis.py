from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .launch_packet import atomic_write_json
from .probe import build_coarse_ownership_labels, evaluate_actor_probe, train_probe, train_vision_probe


PHYSICAL_ACTORS = {
    "A1": "light-haired camouflage-rashguard athlete applying the final armbar",
    "A2": "dark-haired solid-dark-rashguard athlete whose arm is trapped",
}
UNTRUSTED_TRAIN_ACTOR_MASKS = {68, 72, 76, 80}
SAM31_SOURCE_OVERRIDES = {84}
VALIDATION_FRAMES = {16, 48, 88, 128}
TRAPPED_ARM_POINTS = ((0.47, 0.58), (0.54, 0.57), (0.65, 0.55))
CANONICAL_FULL_LAYERS = (5, 11, 18, 26)


@dataclass
class LabelBundle:
    frames: list[int]
    subsets: list[str]
    labels: object
    contact: object
    manifest: dict


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _possible_foreground(image_hw: tuple[int, int], boxes_xyxy: np.ndarray, padding_fraction: float = 0.015):
    height, width = image_hw
    possible = np.zeros((height, width), dtype=bool)
    pad_x, pad_y = width * padding_fraction, height * padding_fraction
    for x0, y0, x1, y1 in boxes_xyxy:
        left = max(0, int(np.floor(x0 - pad_x)))
        top = max(0, int(np.floor(y0 - pad_y)))
        right = min(width, int(np.ceil(x1 + pad_x)))
        bottom = min(height, int(np.ceil(y1 + pad_y)))
        possible[top:bottom, left:right] = True
    return possible


def _contact_mask(output_hw: tuple[int, int], radius: int = 0):
    height, width = output_hw
    contact = np.zeros(output_hw, dtype=bool)
    for x, y in TRAPPED_ARM_POINTS:
        row = min(height - 1, int(y * height))
        column = min(width - 1, int(x * width))
        y0, y1 = max(0, row - radius), min(height, row + radius + 1)
        x0, x1 = max(0, column - radius), min(width, column + radius + 1)
        contact[y0:y1, x0:x1] = True
    return contact


def build_label_bundle(
    project_root: Path,
    packet_dir: Path,
    output_dir: Path,
    output_hw: tuple[int, int] = (30, 30),
) -> LabelBundle:
    import torch
    from PIL import Image

    project_root, packet_dir, output_dir = map(Path, (project_root, packet_dir, output_dir))
    ownership = json.loads((packet_dir / "ownership/ownership-manifest.json").read_text())
    frame_manifest = json.loads((packet_dir / "inputs/frame-manifest.json").read_text())
    frame_paths = {item["frame_index"]: project_root / item["path"] for item in frame_manifest["frames"]}
    dense_path = project_root / "runs/openai-prompthmr-video-definitive-20260826T2200IST-v2/dense_tracks.npz"
    with np.load(dense_path, allow_pickle=False) as values:
        boxes_by_frame = {
            int(frame): values["boxes_xyxy"][position]
            for position, frame in enumerate(values["frame_indices"])
        }

    label_dir, contact_dir = output_dir / "labels", output_dir / "contact"
    label_dir.mkdir(parents=True, exist_ok=True)
    contact_dir.mkdir(parents=True, exist_ok=True)
    labels, contacts, frames, subsets, records = [], [], [], [], []
    contact_radius = 0 if max(output_hw) <= 30 else 1
    for record in ownership["frames"]:
        frame = int(record["frame_index"])
        image = Image.open(frame_paths[frame])
        height, width = image.height, image.width
        source_kind = "sam31" if frame in SAM31_SOURCE_OVERRIDES else "sam3"
        source_path = project_root / record["sources"][source_kind]
        with np.load(source_path, allow_pickle=False) as values:
            a1 = values["A1"].astype(bool)
            a2 = values["A2"].astype(bool)
        possible = _possible_foreground((height, width), boxes_by_frame[frame])
        label = build_coarse_ownership_labels(
            a1,
            a2,
            possible_foreground=possible,
            output_hw=output_hw,
            actor_min_fraction=0.70,
            competing_max_fraction=0.05,
            background_max_fraction=0.02,
        )
        actor_source_status = "reviewed_conservative_core"
        if frame in UNTRUSTED_TRAIN_ACTOR_MASKS:
            label[label != 0] = 255
            actor_source_status = "actor_labels_excluded_after_visual_audit"
        contact = np.zeros(output_hw, dtype=bool)
        if frame == 143:
            contact = _contact_mask(output_hw, radius=contact_radius)
            label[contact] = 2

        label_path = label_dir / f"frame_{frame:06d}.png"
        contact_path = contact_dir / f"frame_{frame:06d}.png"
        Image.fromarray(label, mode="L").save(label_path)
        Image.fromarray(contact.astype(np.uint8) * 255, mode="L").save(contact_path)
        counts = {str(value): int((label == value).sum()) for value in (0, 1, 2, 255)}
        records.append({
            "frame_index": frame,
            "subset": record["subset"],
            "screen_subset": "validation" if frame in VALIDATION_FRAMES else record["subset"],
            "actor_source": source_kind,
            "actor_source_status": actor_source_status,
            "label_path": str(label_path.relative_to(output_dir)),
            "label_sha256": _sha256(label_path),
            "contact_path": str(contact_path.relative_to(output_dir)),
            "contact_sha256": _sha256(contact_path),
            "counts": counts,
            "contact_patch_count": int(contact.sum()),
        })
        frames.append(frame)
        subsets.append(record["subset"])
        labels.append(torch.from_numpy(label.copy()).reshape(-1))
        contacts.append(torch.from_numpy(contact.copy()).reshape(-1))

    manifest = {
        "status": "conservative_pseudo_labels_with_manual_final_contact_truth",
        "physical_actors": PHYSICAL_ACTORS,
        "grid_hw": list(output_hw),
        "construction": {
            "actor_min_fraction": 0.70,
            "competing_max_fraction": 0.05,
            "background_max_fraction": 0.02,
            "background_guard": "outside padded union of independently frozen OpenAI actor boxes",
            "untrusted_train_actor_masks": sorted(UNTRUSTED_TRAIN_ACTOR_MASKS),
            "sam31_source_overrides": sorted(SAM31_SOURCE_OVERRIDES),
            "manual_contact_frame": 143,
            "manual_contact_actor": "A2",
            "manual_contact_points_normalized_xy": [list(point) for point in TRAPPED_ARM_POINTS],
            "contact_radius_patches": contact_radius,
        },
        "dense_box_source": str(dense_path.relative_to(project_root)),
        "validation_frames": sorted(VALIDATION_FRAMES),
        "records": records,
    }
    atomic_write_json(output_dir / "label-manifest.json", manifest)
    return LabelBundle(frames, subsets, torch.stack(labels).long(), torch.stack(contacts).bool(), manifest)


def _load_query_pair(run_root: Path, context: str, condition: str, thinking: str, language_layer: int, remap=False):
    import torch
    from safetensors.torch import load_file

    pair = torch.stack([
        load_file(run_root / f"semantic/video/{context}/{condition}/{thinking}/{actor}.safetensors")["marker_states"][language_layer]
        for actor in ("A1", "A2")
    ])
    return pair.flip(0) if remap else pair


def _matched_random_pair(reference, seed: int):
    import torch

    generator = torch.Generator().manual_seed(seed)
    random = torch.randn(reference.shape, generator=generator, dtype=torch.float32)
    random = torch.nn.functional.normalize(random, dim=-1)
    return random.to(reference.dtype) * reference.float().norm(dim=-1, keepdim=True).to(reference.dtype)


def _metric_score(metrics: dict) -> float:
    return 0.5 * (metrics["A1_iou"] + metrics["A2_iou"])


def _indices(bundle: LabelBundle, subset: str):
    if subset == "screen_train":
        return [i for i, (frame, original) in enumerate(zip(bundle.frames, bundle.subsets)) if original == "train" and frame not in VALIDATION_FRAMES]
    if subset == "validation":
        return [i for i, frame in enumerate(bundle.frames) if frame in VALIDATION_FRAMES]
    return [i for i, original in enumerate(bundle.subsets) if original == subset]


def _load_pooled_vision(run_root: Path, frames: list[int], layer: int):
    import torch
    from safetensors.torch import load_file

    return torch.stack([
        load_file(run_root / f"spatial/pooled/layer_{layer:02d}/frame_{frame:06d}.safetensors")["hidden"]
        for frame in frames
    ])


def _load_full_vision(run_root: Path, frames: list[int], layer: int):
    import torch
    from safetensors.torch import load_file

    return torch.stack([
        load_file(run_root / f"spatial/full/layer_{layer:02d}/frame_{frame:06d}.safetensors")["hidden"]
        for frame in frames
    ])


def _run_one(vision, labels, train_indices, eval_indices, *, pair=None, contact=None, epochs=80, seed=380027, device="cuda"):
    import torch

    train_vision = vision[train_indices].to(device)
    train_labels = labels[train_indices].to(device)
    eval_vision = vision[eval_indices].to(device)
    eval_labels = labels[eval_indices].to(device)
    eval_contact = None if contact is None else contact[eval_indices].to(device)
    if pair is None:
        return train_vision_probe(
            train_vision, train_labels, epochs=epochs, seed=seed,
            evaluation=(eval_vision, eval_labels), contact_mask=eval_contact,
        )
    queries = pair.repeat(len(vision), 1, 1)
    return train_probe(
        train_vision,
        queries[train_indices].to(device),
        train_labels,
        epochs=epochs,
        seed=seed,
        evaluation=(eval_vision, queries[eval_indices].to(device), eval_labels),
        contact_mask=eval_contact,
    )


def _run_one_batched(
    vision,
    labels,
    train_indices,
    eval_indices,
    *,
    pair=None,
    contact=None,
    epochs=8,
    seed=380027,
    batch_size=4096,
    device="cuda",
):
    train_vision = vision[train_indices]
    train_labels = labels[train_indices]
    eval_vision = vision[eval_indices]
    eval_labels = labels[eval_indices]
    eval_contact = None if contact is None else contact[eval_indices]
    if pair is None:
        return train_vision_probe(
            train_vision,
            train_labels,
            epochs=epochs,
            seed=seed,
            evaluation=(eval_vision, eval_labels),
            contact_mask=eval_contact,
            batch_size=batch_size,
            device=device,
        )
    queries = pair.repeat(len(vision), 1, 1)
    return train_probe(
        train_vision,
        queries[train_indices],
        train_labels,
        epochs=epochs,
        seed=seed,
        evaluation=(eval_vision, queries[eval_indices], eval_labels),
        contact_mask=eval_contact,
        batch_size=batch_size,
        device=device,
    )


def _run_prepartitioned_batched(
    train_vision,
    train_labels,
    eval_vision,
    eval_labels,
    *,
    pair=None,
    contact=None,
    epochs=8,
    seed=380027,
    batch_size=4096,
    device="cuda",
):
    """Train without making a fresh full-tensor subset copy for every seed."""
    if pair is None:
        return train_vision_probe(
            train_vision,
            train_labels,
            epochs=epochs,
            seed=seed,
            evaluation=(eval_vision, eval_labels),
            contact_mask=contact,
            batch_size=batch_size,
            device=device,
        )
    train_queries = pair.repeat(len(train_vision), 1, 1).to(train_vision.device)
    eval_queries = pair.repeat(len(eval_vision), 1, 1).to(eval_vision.device)
    return train_probe(
        train_vision,
        train_queries,
        train_labels,
        epochs=epochs,
        seed=seed,
        evaluation=(eval_vision, eval_queries, eval_labels),
        contact_mask=contact,
        batch_size=batch_size,
        device=device,
    )


def run_pooled_campaign(project_root: Path, packet_dir: Path, run_root: Path, output_dir: Path, epochs: int = 80):
    import torch

    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = build_label_bundle(project_root, packet_dir, output_dir / "pooled-labels")
    screen_train, validation = _indices(bundle, "screen_train"), _indices(bundle, "validation")
    final_train, test = _indices(bundle, "train"), _indices(bundle, "test")
    query_pairs = {
        "identity_4fps": _load_query_pair(run_root, "4fps", "identity_only", "off", 63),
        "contact_4fps": _load_query_pair(run_root, "4fps", "contact_ownership", "off", 63),
        "action_4fps": _load_query_pair(run_root, "4fps", "action_relational", "off", 63),
    }
    query_pairs["random_matched"] = _matched_random_pair(query_pairs["action_4fps"], 380027)
    layer_screen = []
    for layer in range(27):
        vision = _load_pooled_vision(run_root, bundle.frames, layer)
        for name, pair in (("vision_only", None), *query_pairs.items()):
            result = _run_one(
                vision, bundle.labels, screen_train, validation,
                pair=pair, epochs=epochs, seed=380027,
            )
            layer_screen.append({
                "vision_layer": layer,
                "condition": name,
                "score": _metric_score(result.metrics),
                "metrics": result.metrics,
                "final_loss": result.losses[-1],
            })
        print(f"pooled vision layer {layer:02d}/26 complete", flush=True)
    atomic_write_json(output_dir / "pooled-layer-screen.json", {"records": layer_screen})

    action_records = [item for item in layer_screen if item["condition"] == "action_4fps"]
    best_layer = max(action_records, key=lambda item: item["score"])["vision_layer"]
    vision = _load_pooled_vision(run_root, bundle.frames, best_layer)
    language_screen = []
    for language_layer in range(64):
        pair = _load_query_pair(run_root, "4fps", "action_relational", "off", language_layer)
        result = _run_one(
            vision, bundle.labels, screen_train, validation,
            pair=pair, epochs=epochs, seed=380027,
        )
        language_screen.append({
            "vision_layer": best_layer,
            "language_layer": language_layer,
            "score": _metric_score(result.metrics),
            "metrics": result.metrics,
            "final_loss": result.losses[-1],
        })
    atomic_write_json(output_dir / "pooled-language-screen.json", {"records": language_screen})
    best_language = max(language_screen, key=lambda item: item["score"])["language_layer"]

    variants = {
        "vision_only": None,
        "identity_4fps": _load_query_pair(run_root, "4fps", "identity_only", "off", best_language),
        "contact_4fps": _load_query_pair(run_root, "4fps", "contact_ownership", "off", best_language),
        "action_4fps_off": _load_query_pair(run_root, "4fps", "action_relational", "off", best_language),
        "action_4fps_xhigh": _load_query_pair(run_root, "4fps", "action_relational", "xhigh", best_language),
        "action_2fps_remapped": _load_query_pair(run_root, "2fps", "action_relational", "off", best_language, remap=True),
        "action_five_frame_remapped": _load_query_pair(run_root, "five_frame", "action_relational", "off", best_language, remap=True),
        "action_single_frame": _load_query_pair(run_root, "single_frame", "action_relational", "off", best_language),
        "action_reversed": _load_query_pair(run_root, "4fps_reversed", "action_relational", "off", best_language),
        "action_shuffled": _load_query_pair(run_root, "4fps_shuffled", "action_relational", "off", best_language),
        "action_8fps": _load_query_pair(run_root, "8fps", "action_relational", "off", best_language),
    }
    action_reference = variants["action_4fps_off"]
    variants["random_matched"] = _matched_random_pair(action_reference, 380027)
    variants["mean_query"] = action_reference.mean(0, keepdim=True).repeat(2, 1)
    variants["zero_query"] = torch.zeros_like(action_reference)
    final_records = []
    logits_by_variant = {}
    for name, pair in variants.items():
        for seed in (7, 71, 701, 7001, 70001):
            result = _run_one(
                vision, bundle.labels, final_train, test,
                pair=pair, contact=bundle.contact, epochs=max(epochs, 100), seed=seed,
            )
            final_records.append({
                "condition": name,
                "seed": seed,
                "metrics": result.metrics,
                "swapped_metrics": result.swapped_metrics,
                "score": _metric_score(result.metrics),
                "final_loss": result.losses[-1],
            })
            if seed == 7:
                logits_by_variant[name] = result.logits
        print(f"final pooled condition {name} complete", flush=True)
    atomic_write_json(output_dir / "pooled-final.json", {
        "selected_vision_layer": best_layer,
        "selected_language_layer": best_language,
        "selection_policy": "validation mean actor IoU; held-out test untouched until final comparison",
        "records": final_records,
    })
    torch.save({"frames": [bundle.frames[i] for i in test], "logits": logits_by_variant}, output_dir / "pooled-final-logits.pt")
    return {
        "selected_vision_layer": best_layer,
        "selected_language_layer": best_language,
        "layer_screen_records": len(layer_screen),
        "language_screen_records": len(language_screen),
        "final_records": len(final_records),
    }


def run_full_campaign(project_root: Path, packet_dir: Path, run_root: Path, output_dir: Path, epochs: int = 8):
    import torch
    from safetensors.torch import load_file

    output_dir.mkdir(parents=True, exist_ok=True)
    pooled_selection = json.loads((output_dir / "pooled-final.json").read_text())
    language_layer = int(pooled_selection["selected_language_layer"])
    sample = load_file(run_root / "spatial/full/layer_05/frame_000000.safetensors")
    _, grid_h, grid_w = sample["grid_thw"][0].tolist()
    bundle = build_label_bundle(project_root, packet_dir, output_dir / "full-labels", (grid_h, grid_w))
    screen_train, validation = _indices(bundle, "screen_train"), _indices(bundle, "validation")
    final_train, test = _indices(bundle, "train"), _indices(bundle, "test")
    action_pair = _load_query_pair(run_root, "4fps", "action_relational", "off", language_layer)
    random_pair = _matched_random_pair(action_pair, 380027)
    screen_path = output_dir / "full-layer-screen.json"
    if screen_path.exists():
        screen_records = json.loads(screen_path.read_text())["records"]
        if len(screen_records) != len(CANONICAL_FULL_LAYERS) * 3:
            raise RuntimeError("partial full-resolution screen is not a valid resume checkpoint")
        print("resuming from completed full-resolution layer screen", flush=True)
    else:
        screen_records = []
        for layer in CANONICAL_FULL_LAYERS:
            vision = _load_full_vision(run_root, bundle.frames, layer)
            for name, pair in (("vision_only", None), ("action_4fps_off", action_pair), ("random_matched", random_pair)):
                result = _run_one_batched(
                    vision,
                    bundle.labels,
                    screen_train,
                    validation,
                    pair=pair,
                    epochs=max(4, epochs // 2),
                    seed=380027,
                )
                screen_records.append({
                    "vision_layer": layer,
                    "condition": name,
                    "score": _metric_score(result.metrics),
                    "metrics": result.metrics,
                    "final_loss": result.losses[-1],
                })
                del result
                torch.cuda.empty_cache()
            del vision
            print(f"full-resolution layer {layer:02d} screen complete", flush=True)
        atomic_write_json(screen_path, {
            "language_layer": language_layer,
            "records": screen_records,
        })
    action_records = [item for item in screen_records if item["condition"] == "action_4fps_off"]
    best_layer = max(action_records, key=lambda item: item["score"])["vision_layer"]
    train_frames = [bundle.frames[index] for index in final_train]
    test_frames = [bundle.frames[index] for index in test]
    train_vision = _load_full_vision(run_root, train_frames, best_layer).to("cuda")
    eval_vision = _load_full_vision(run_root, test_frames, best_layer).to("cuda")
    train_labels = bundle.labels[final_train]
    eval_labels = bundle.labels[test]
    eval_contact = bundle.contact[test]
    variants = {
        "vision_only": None,
        "identity_4fps": _load_query_pair(run_root, "4fps", "identity_only", "off", language_layer),
        "contact_4fps": _load_query_pair(run_root, "4fps", "contact_ownership", "off", language_layer),
        "action_4fps_off": action_pair,
        "action_4fps_xhigh": _load_query_pair(run_root, "4fps", "action_relational", "xhigh", language_layer),
        "action_reversed": _load_query_pair(run_root, "4fps_reversed", "action_relational", "off", language_layer),
        "action_shuffled": _load_query_pair(run_root, "4fps_shuffled", "action_relational", "off", language_layer),
        "action_8fps": _load_query_pair(run_root, "8fps", "action_relational", "off", language_layer),
        "random_matched": random_pair,
        "zero_query": torch.zeros_like(action_pair),
    }
    progress_path = output_dir / "full-final-progress.json"
    final_records = json.loads(progress_path.read_text()).get("records", []) if progress_path.exists() else []
    completed = {(record["condition"], int(record["seed"])) for record in final_records}
    logits_dir = output_dir / "full-logits"
    logits_dir.mkdir(exist_ok=True)
    for name, pair in variants.items():
        for seed in (7, 701, 70001):
            if (name, seed) in completed:
                continue
            result = _run_prepartitioned_batched(
                train_vision,
                train_labels,
                eval_vision,
                eval_labels,
                pair=pair,
                contact=eval_contact,
                epochs=epochs,
                seed=seed,
            )
            final_records.append({
                "condition": name,
                "seed": seed,
                "metrics": result.metrics,
                "swapped_metrics": result.swapped_metrics,
                "score": _metric_score(result.metrics),
                "final_loss": result.losses[-1],
            })
            if seed == 7:
                torch.save(result.logits, logits_dir / f"{name}.pt")
            atomic_write_json(progress_path, {"records": final_records})
            del result
            torch.cuda.empty_cache()
        print(f"final full-resolution condition {name} complete", flush=True)
    atomic_write_json(output_dir / "full-final.json", {
        "selected_vision_layer": best_layer,
        "selected_language_layer": language_layer,
        "grid_hw": [grid_h, grid_w],
        "selection_policy": "canonical full layer selected on validation mean actor IoU",
        "records": final_records,
    })
    torch.save({
        "frames": [bundle.frames[i] for i in test],
        "logits": {name: torch.load(logits_dir / f"{name}.pt", map_location="cpu", weights_only=True) for name in variants},
    }, output_dir / "full-final-logits.pt")
    return {
        "selected_vision_layer": best_layer,
        "selected_language_layer": language_layer,
        "grid_hw": [grid_h, grid_w],
        "screen_records": len(screen_records),
        "final_records": len(final_records),
    }


def run_cross_query_control(project_root: Path, packet_dir: Path, run_root: Path, output_dir: Path, epochs: int = 100):
    """Hold the decoder fixed while substituting alternative semantic states."""
    import torch
    import torch.nn.functional as functional

    output_dir.mkdir(parents=True, exist_ok=True)
    target_device = "cuda" if torch.cuda.is_available() else "cpu"
    selection = json.loads((output_dir / "pooled-final.json").read_text())
    vision_layer = int(selection["selected_vision_layer"])
    language_layer = int(selection["selected_language_layer"])
    bundle = build_label_bundle(project_root, packet_dir, output_dir / "pooled-labels")
    final_train, test = _indices(bundle, "train"), _indices(bundle, "test")
    vision = _load_pooled_vision(run_root, bundle.frames, vision_layer)
    pairs = {
        "identity_4fps": _load_query_pair(run_root, "4fps", "identity_only", "off", language_layer),
        "contact_4fps": _load_query_pair(run_root, "4fps", "contact_ownership", "off", language_layer),
        "action_4fps_off": _load_query_pair(run_root, "4fps", "action_relational", "off", language_layer),
        "action_4fps_xhigh": _load_query_pair(run_root, "4fps", "action_relational", "xhigh", language_layer),
        "action_reversed": _load_query_pair(run_root, "4fps_reversed", "action_relational", "off", language_layer),
        "action_shuffled": _load_query_pair(run_root, "4fps_shuffled", "action_relational", "off", language_layer),
    }
    pairs["random_matched"] = _matched_random_pair(pairs["action_4fps_off"], 380027)
    pairs["action_swapped"] = pairs["action_4fps_off"].flip(0)

    reference = pairs["action_4fps_off"].float().reshape(-1)
    pair_diagnostics = {
        name: {
            "cosine_to_action": float(functional.cosine_similarity(pair.float().reshape(-1), reference, dim=0)),
            "actor_state_cosine": float(functional.cosine_similarity(pair[0].float(), pair[1].float(), dim=0)),
            "pair_norm": float(pair.float().norm()),
        }
        for name, pair in pairs.items()
    }

    records, logits = [], {}
    train_anchors = ("identity_4fps", "action_4fps_off")
    for trained_on in train_anchors:
        for seed in (7, 71, 701, 7001, 70001):
            trained = _run_one(
                vision,
                bundle.labels,
                final_train,
                test,
                pair=pairs[trained_on],
                contact=bundle.contact,
                epochs=epochs,
                seed=seed,
                device=target_device,
            )
            for evaluated_with, pair in pairs.items():
                query_batch = pair.repeat(len(test), 1, 1)
                evaluated = evaluate_actor_probe(
                    trained.model,
                    vision[test],
                    query_batch,
                    bundle.labels[test],
                    contact_mask=bundle.contact[test],
                )
                records.append({
                    "trained_on": trained_on,
                    "evaluated_with": evaluated_with,
                    "seed": seed,
                    "score": _metric_score(evaluated.metrics),
                    "metrics": evaluated.metrics,
                })
                if seed == 7:
                    logits[f"{trained_on}__{evaluated_with}"] = evaluated.logits
            del trained
            torch.cuda.empty_cache()
        print(f"cross-query control trained on {trained_on} complete", flush=True)

    atomic_write_json(output_dir / "cross-query-control.json", {
        "selected_vision_layer": vision_layer,
        "selected_language_layer": language_layer,
        "policy": "decoder weights held fixed while only the two actor marker states are substituted",
        "pair_diagnostics": pair_diagnostics,
        "records": records,
    })
    torch.save({
        "frames": [bundle.frames[index] for index in test],
        "logits": logits,
    }, output_dir / "cross-query-control-logits.pt")
    return {"records": len(records), "trained_models": len(train_anchors) * 5}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--packet-dir", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--stage", choices=("pooled", "full", "cross", "all"), default="pooled")
    args = parser.parse_args(argv)
    result = {}
    if args.stage in ("pooled", "all"):
        result["pooled"] = run_pooled_campaign(args.project_root, args.packet_dir, args.run_root, args.output_dir, args.epochs)
    if args.stage in ("full", "all"):
        result["full"] = run_full_campaign(args.project_root, args.packet_dir, args.run_root, args.output_dir, args.epochs)
    if args.stage in ("cross", "all"):
        result["cross"] = run_cross_query_control(args.project_root, args.packet_dir, args.run_root, args.output_dir, max(args.epochs, 100))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
