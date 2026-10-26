"""Audit self-grounding provenance without importing Torch or loading activations.

JSON validity cannot establish that independently generated A1/A2 identities refer
to the same physical people. All records explicitly retain that review requirement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import tempfile
from pathlib import Path


MAX_HEADER_BYTES = 1024 * 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_grounding_record(path: str | Path) -> dict:
    path = Path(path)
    with path.open("rb") as handle:
        size = os.fstat(handle.fileno()).st_size
        raw = handle.read(8)
        if len(raw) != 8:
            raise ValueError(f"truncated grounding header: {path}")
        length = struct.unpack("<Q", raw)[0]
        if not 2 <= length <= MAX_HEADER_BYTES or length + 8 > size:
            raise ValueError(f"invalid grounding header length: {path}")
        try:
            header = json.loads(handle.read(length))
            metadata = json.loads(header["__metadata__"]["campaign"])
            flag = header["contract_valid"]
        except (ValueError, KeyError, TypeError) as error:
            raise ValueError(f"invalid grounding header metadata: {path}") from error
        if not isinstance(metadata, dict) or metadata.get("stage") != "self_grounding":
            raise ValueError(f"unexpected grounding stage: {path}")
        if not isinstance(flag, dict) or flag.get("dtype") != "BOOL" or flag.get("shape") != [1]:
            raise ValueError(f"invalid grounding boolean tensor: {path}")
        offsets = flag.get("data_offsets")
        if (not isinstance(offsets, list) or len(offsets) != 2
                or any(type(value) is not int for value in offsets)
                or offsets[0] < 0 or offsets[1] != offsets[0] + 1
                or 8 + length + offsets[1] > size):
            raise ValueError(f"invalid grounding boolean offsets: {path}")
        handle.seek(8 + length + offsets[0])
        raw_flag = handle.read(1)
        if raw_flag not in (b"\x00", b"\x01"):
            raise ValueError(f"invalid grounding boolean payload: {path}")
    contract_valid = raw_flag == b"\x01"
    try:
        answer = json.loads(metadata.get("canonical_answer", ""))
    except (ValueError, TypeError):
        answer = None
    valid_answer = isinstance(answer, dict) and all(
        isinstance(answer.get(key), str) and answer[key].strip()
        for key in ("A1", "A2", "relation", "uncertainty")
    )
    if contract_valid != bool(valid_answer):
        raise ValueError(f"grounding tensor flag and canonical JSON disagree: {path}")
    return {
        "sha256": _sha256(path), "size_bytes": size,
        "context": metadata.get("context"), "condition": metadata.get("condition"),
        "thinking_mode": metadata.get("thinking_mode"),
        "contract_valid": contract_valid, "answer": answer if valid_answer else None,
        "initial_answer": metadata.get("initial_answer", ""),
        "repair_answer": metadata.get("repair_answer", ""),
        "physical_identity_reviewed": False,
        "issues": [] if contract_valid else ["invalid_grounding_contract"],
    }


def require_valid_grounding(path: str | Path, *, context: str, condition: str,
                            thinking_mode: str) -> dict:
    record = read_grounding_record(path)
    if any(record[key] != value for key, value in (
        ("context", context), ("condition", condition), ("thinking_mode", thinking_mode)
    )):
        raise ValueError(f"grounding path and metadata disagree: {path}")
    if not record["contract_valid"]:
        raise ValueError(f"semantic cache has invalid grounding; exclude or reacquire explicitly: {path}")
    return record


def audit_grounding_cache(run_root: str | Path) -> dict:
    root = Path(run_root)
    groups = [("armbar", root / "grounding")]
    breadth = root / "breadth"
    if breadth.is_dir():
        groups.extend((clip.name, clip / "grounding") for clip in sorted(breadth.iterdir())
                      if clip.is_dir())
    records = []
    for clip, directory in groups:
        for path in sorted(directory.glob("*/*/*.safetensors")):
            context, condition, filename = path.relative_to(directory).parts
            record = read_grounding_record(path)
            expected = {"context": context, "condition": condition,
                        "thinking_mode": Path(filename).stem}
            if any(record[key] != value for key, value in expected.items()):
                raise ValueError(f"grounding path and metadata disagree: {path}")
            records.append({"clip_id": clip, "path": path.relative_to(root).as_posix(), **record})
    if not records:
        raise ValueError("no grounding artifacts found")
    return {
        "format": "ownership-grounding-audit-v1", "run_root": str(root.resolve()),
        "record_count": len(records),
        "invalid_contract_count": sum(not r["contract_valid"] for r in records),
        "physical_identity_review_required_count": len(records),
        "interpretation": "Valid JSON is not evidence of physical identity or action correctness. "
                          "Review each condition/context against independent actor references before comparisons.",
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = audit_grounding_cache(args.run_root)
    path = Path(args.output)
    if path.exists():
        raise FileExistsError(f"audit report already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".grounding-audit-", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        # Exclusive creation also prevents a concurrent audit from overwriting a result.
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
