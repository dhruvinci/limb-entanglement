"""Reviewed physical-actor alignment, independent of decoder predictions."""
from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from .grounding_audit import _sha256, read_grounding_record, require_valid_grounding


CONDITIONS = ("identity_only", "action_relational", "contact_ownership")
ALIGNMENT_ATTESTATION = (
    "I matched cached query identities to the physical athletes in the reference frames, "
    "without using decoder predictions or evaluation scores to choose the mapping."
)


def _write_new(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".alignment-", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(data, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _identity(record):
    try:
        values = tuple(record[k] for k in ("clip_id", "context", "condition", "thinking_mode"))
    except (KeyError, TypeError) as error:
        raise ValueError("alignment record identity is missing") from error
    if any(not isinstance(v, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", v) for v in values):
        raise ValueError("unsafe alignment identity")
    return values


def build_alignment_template(qwen_breadth_root, reference_packet, output_path, *, conditions=CONDITIONS):
    packet = Path(reference_packet)
    if packet.with_suffix(packet.suffix + ".sha256").read_text().strip() != _sha256(packet):
        raise ValueError("reference packet SHA-256 mismatch")
    reference = json.loads(packet.read_text())
    if reference.get("format") not in {"ownership-manual-packet-v1", "ownership-label-candidates-v1"}:
        raise ValueError("alignment requires an ownership reference packet")
    if not conditions or len(set(conditions)) != len(conditions) or set(conditions) - set(CONDITIONS):
        raise ValueError("invalid alignment conditions")
    clips = sorted({r["clip_id"] for r in reference["records"]})
    records = []
    for clip in clips:
        for condition in conditions:
            identity = {"clip_id": clip, "context": "4fps", "condition": condition, "thinking_mode": "off"}
            _identity(identity)
            grounding_path = Path(qwen_breadth_root) / clip / "grounding/4fps" / condition / "off.safetensors"
            grounding = read_grounding_record(grounding_path)
            if any(grounding[k] != identity[k] for k in ("context", "condition", "thinking_mode")):
                raise ValueError("grounding path and metadata disagree")
            records.append({**identity, "grounding_sha256": grounding["sha256"],
                "grounding_contract_valid": grounding["contract_valid"], "answer": grounding["answer"],
                "actor_state_sha256": {a: _sha256(Path(qwen_breadth_root) / clip /
                    f"semantic/video/4fps/{condition}/off/{a}.safetensors") for a in ("A1", "A2")}})
    if not records:
        raise ValueError("empty alignment reference inventory")
    source = {"format": "ownership-actor-alignment-source-v1",
              "reference_packet_sha256": _sha256(packet), "records": records}
    path = Path(output_path)
    if path.exists():
        raise FileExistsError(path)
    source_path = path.with_suffix(".source.json")
    _write_new(source_path, source)
    template = {"format": "ownership-actor-alignment-review-v1", "reviewed": False,
                "source_path": source_path.name, "source_sha256": _sha256(source_path),
                "reviewer": "", "attestation": "",
                "records": [{**r, "decision": "pending", "canonical_actor_order": None,
                             "evidence": ""} for r in records]}
    _write_new(path, template)
    return template


def _load_source(path, review):
    relative = Path(str(review.get("source_path", "")))
    if relative.is_absolute() or len(relative.parts) != 1:
        raise ValueError("unsafe alignment source path")
    source_path = (path.parent / relative).resolve()
    if not source_path.is_relative_to(path.parent.resolve()):
        raise ValueError("alignment source escapes its package")
    if _sha256(source_path) != review.get("source_sha256"):
        raise ValueError("alignment source SHA-256 mismatch")
    source = json.loads(source_path.read_text())
    if source.get("format") != "ownership-actor-alignment-source-v1":
        raise ValueError("invalid alignment source format")
    return source


def _validate_decisions(review, source):
    if review.get("format") != "ownership-actor-alignment-review-v1":
        raise ValueError("invalid actor alignment format")
    originals = {_identity(r): r for r in source["records"]}
    records = review.get("records")
    if not isinstance(records, list) or len(records) != len(originals):
        raise ValueError("alignment inventory is incomplete or duplicated")
    seen = set()
    for r in records:
        identity = _identity(r)
        if identity in seen or identity not in originals:
            raise ValueError("alignment inventory is incomplete or duplicated")
        seen.add(identity)
        if any(r.get(k) != v for k, v in originals[identity].items()):
            raise ValueError("immutable alignment provenance was edited")
        if not isinstance(r.get("evidence"), str) or not r["evidence"].strip():
            raise ValueError("every mapping or exclusion requires visual evidence/reason")
        if r.get("decision") == "include":
            if r.get("grounding_contract_valid") is not True:
                raise ValueError("cannot include invalid grounding")
            if r.get("canonical_actor_order") not in (["A1", "A2"], ["A2", "A1"]):
                raise ValueError("alignment must bijectively map both physical actors")
        elif r.get("decision") == "exclude":
            if r.get("canonical_actor_order") is not None:
                raise ValueError("excluded states must not provide an actor mapping")
        else:
            raise ValueError("alignment review is pending")


def finalize_alignment_review(pending_path, output_path, *, reviewer, attested):
    if not attested or not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError("explicit reviewer and identity-review attestation are required")
    pending, output = Path(pending_path), Path(output_path)
    if output.exists():
        raise FileExistsError(output)
    review = json.loads(pending.read_text())
    source = _load_source(pending, review)
    _validate_decisions(review, source)
    source_path = output.with_suffix(".source.json")
    if source_path.exists():
        if json.loads(source_path.read_text()) != source:
            raise ValueError("existing frozen alignment source differs")
    else:
        _write_new(source_path, source)
    frozen = {**review, "reviewed": True, "reviewer": reviewer.strip(),
              "reviewed_at": datetime.now(timezone.utc).isoformat(),
              "attestation": ALIGNMENT_ATTESTATION, "source_path": source_path.name,
              "source_sha256": _sha256(source_path)}
    _write_new(output, frozen)
    output.with_suffix(output.suffix + ".sha256").write_text(_sha256(output) + "\n")
    return frozen


def load_alignment_orders(path, *, reference_packet_sha256, qwen_breadth_root, clip_ids,
                          condition, context="4fps", thinking_mode="off"):
    path = Path(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.is_file() or sidecar.read_text().strip() != _sha256(path):
        raise ValueError("reviewed alignment SHA-256 is missing or mismatched")
    review = json.loads(path.read_text())
    if (review.get("reviewed") is not True or review.get("attestation") != ALIGNMENT_ATTESTATION
            or not isinstance(review.get("reviewer"), str) or not review["reviewer"].strip()):
        raise ValueError("physical actor alignment has not been reviewed")
    source = _load_source(path, review)
    if source.get("reference_packet_sha256") != reference_packet_sha256:
        raise ValueError("actor alignment belongs to a different reference packet")
    _validate_decisions(review, source)
    by_identity = {_identity(r): r for r in review["records"]}
    orders = {}
    for clip in clip_ids:
        identity = (clip, context, condition, thinking_mode)
        if identity not in by_identity:
            raise ValueError(f"missing physical actor alignment: {identity}")
        r = by_identity[identity]
        if r["decision"] != "include":
            raise ValueError(f"requested semantic condition is excluded: {identity}")
        root = Path(qwen_breadth_root) / clip
        grounding = require_valid_grounding(root / "grounding" / context / condition / f"{thinking_mode}.safetensors",
            context=context, condition=condition, thinking_mode=thinking_mode)
        if grounding["sha256"] != r["grounding_sha256"]:
            raise ValueError("alignment grounding SHA-256 mismatch")
        for actor in ("A1", "A2"):
            artifact = root / "semantic/video" / context / condition / thinking_mode / f"{actor}.safetensors"
            if _sha256(artifact) != r["actor_state_sha256"][actor]:
                raise ValueError("aligned actor-state SHA-256 mismatch")
        orders[clip] = tuple(r["canonical_actor_order"])
    return orders


def verify_campaign_alignment(path, *, label_manifests, qwen_breadth_root, conditions,
                              context="4fps", thinking_mode="off"):
    """Bind reviewed query identities to the actual training-label provenance."""
    if path is None:
        raise ValueError("semantic training requires a reviewed actor alignment manifest")
    from .breadth_labels import verify_reviewed_label_manifest, _safe_package_reference

    references = set()
    for clip, value in label_manifests.items():
        label_path = Path(value)
        label = verify_reviewed_label_manifest(label_path, expected_clip_id=clip)
        candidate_path = _safe_package_reference(label_path.parent, label_path.parents[2],
            label["candidate_manifest_path"], role="candidate reference")
        candidate = json.loads(candidate_path.read_text())
        manual_sha = candidate.get("construction", {}).get("packet_sha256")
        if manual_sha is not None:
            if _sha256(candidate_path.parent / "manual-packet-manifest.json") != manual_sha:
                raise ValueError("manual reference packet SHA-256 mismatch")
            references.add(manual_sha)
        else:
            references.add(_sha256(candidate_path))
    if len(references) != 1:
        raise ValueError("labels must share one independent actor reference packet")
    return {condition: load_alignment_orders(path, reference_packet_sha256=next(iter(references)),
        qwen_breadth_root=qwen_breadth_root, clip_ids=tuple(label_manifests), condition=condition,
        context=context, thinking_mode=thinking_mode) for condition in conditions}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--qwen-breadth-root", required=True)
    prepare.add_argument("--reference-packet", required=True)
    prepare.add_argument("--output", required=True)
    finalize = sub.add_parser("finalize")
    finalize.add_argument("--pending", required=True)
    finalize.add_argument("--output", required=True)
    finalize.add_argument("--reviewer", required=True)
    finalize.add_argument("--attest", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        result = build_alignment_template(args.qwen_breadth_root, args.reference_packet, args.output)
    else:
        result = finalize_alignment_review(args.pending, args.output, reviewer=args.reviewer, attested=args.attest)
    print(json.dumps({"reviewed": result["reviewed"], "records": len(result["records"])}, indent=2))


if __name__ == "__main__":
    main()
