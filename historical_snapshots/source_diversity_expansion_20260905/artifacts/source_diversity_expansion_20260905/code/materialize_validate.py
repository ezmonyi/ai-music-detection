#!/usr/bin/env python3
"""Validate completeness, file inventory, hashes, and view-duration invariants."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
import time
from collections import Counter
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", delete=False) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--materialized", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rehash", action="store_true")
    parser.add_argument(
        "--duplicate-exclusions",
        type=Path,
        help="Optional duplicate-exclusion artifact to fingerprint. Exclusions do not satisfy the physical-materialization gate.",
    )
    args = parser.parse_args()
    with args.frozen.open(encoding="utf-8", newline="") as handle:
        frozen_rows = list(csv.DictReader(handle))
    frozen = {x["item_id"]: x for x in frozen_rows}
    records = [json.loads(x) for x in args.materialized.read_text(encoding="utf-8").splitlines() if x]
    latest = {x["item_id"]: x for x in records}
    errors: list[dict[str, str]] = []

    if len(frozen_rows) != len(frozen):
        errors.append({"item_id": "*", "check": "frozen_unique", "detail": str(len(frozen_rows) - len(frozen))})
    if len(records) != len(latest):
        errors.append({"item_id": "*", "check": "canonical_unique", "detail": str(len(records) - len(latest))})
    if set(latest) != set(frozen):
        errors.append({"item_id": "*", "check": "identity_set",
                       "detail": f"missing={len(set(frozen)-set(latest))}, extra={len(set(latest)-set(frozen))}"})

    success = []
    bytes_by_kind: Counter[str] = Counter()
    for item_id in sorted(frozen):
        record = latest.get(item_id)
        if record is None:
            continue
        if record.get("status") != "success":
            errors.append({"item_id": item_id, "check": "status", "detail": str(record.get("error_message", ""))})
            continue
        success.append(record)
        source = frozen[item_id]
        for key in ("label", "source_id", "role", "source_revision", "group_id",
                    "provenance_grade", "evaluation_allowed"):
            if str(record.get(key)) != str(source.get(key)):
                errors.append({"item_id": item_id, "check": f"frozen_{key}",
                               "detail": f"{record.get(key)!r} != {source.get(key)!r}"})
        native_duration = float(record["native_duration_s"])
        long_duration = float(record["view_max60s_duration_s"])
        if long_duration > 60.03 or long_duration > native_duration + 0.03:
            errors.append({"item_id": item_id, "check": "long_unpadded",
                           "detail": f"native={native_duration}, long={long_duration}"})
        if float(record.get("view_max60s_padded_s", -1)) != 0.0:
            errors.append({"item_id": item_id, "check": "long_padding_flag",
                           "detail": str(record.get("view_max60s_padded_s"))})
        if abs(float(record["view_10s_duration_s"]) - 10.0) > 0.03:
            errors.append({"item_id": item_id, "check": "view_10s_duration",
                           "detail": str(record["view_10s_duration_s"])})
        for prefix in ("native", "view_10s", "view_max60s"):
            path = Path(record[f"{prefix}_path"])
            if not path.is_file():
                errors.append({"item_id": item_id, "check": f"{prefix}_exists", "detail": str(path)})
                continue
            actual_bytes = path.stat().st_size
            bytes_by_kind[prefix] += actual_bytes
            if actual_bytes != int(record[f"{prefix}_bytes"]):
                errors.append({"item_id": item_id, "check": f"{prefix}_bytes",
                               "detail": f"{actual_bytes} != {record[f'{prefix}_bytes']}"})
            if args.rehash:
                actual_hash = sha256(path)
                if actual_hash != record[f"{prefix}_sha256"]:
                    errors.append({"item_id": item_id, "check": f"{prefix}_sha256",
                                   "detail": f"{actual_hash} != {record[f'{prefix}_sha256']}"})

    materialized_ids = {str(x["item_id"]) for x in success}
    expected_items = len(frozen)
    verified_items = len(materialized_ids)
    # The thesis requirement is stricter than an evaluation exclusion: every
    # frozen item must exist physically even if a later evaluator excludes a
    # duplicate.  Keep the union-style counter required by downstream gates,
    # while never allowing an exclusion list to hide a missing file here.
    physical_gate_passed = not errors and verified_items == expected_items
    result = {
        "validation_version": "materialization-validation-v1",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "passed" if physical_gate_passed else "failed",
        "passed": physical_gate_passed,
        "expected": expected_items,
        "verified": verified_items,
        "materialized_or_excluded": verified_items,
        "zero_unexplained_failures": not errors,
        "unexplained_failure_count": len(errors),
        "frozen_manifest_path": str(args.frozen.resolve()),
        "frozen_manifest_sha256": sha256(args.frozen),
        "materialization_manifest_path": str(args.materialized.resolve()),
        "materialization_manifest_sha256": sha256(args.materialized),
        "duplicate_exclusion_manifest_path": (
            str(args.duplicate_exclusions.resolve()) if args.duplicate_exclusions else None
        ),
        "duplicate_exclusion_manifest_sha256": (
            sha256(args.duplicate_exclusions) if args.duplicate_exclusions else None
        ),
        "rehash_enabled": args.rehash, "frozen_items": len(frozen),
        "canonical_records": len(records), "success_items": len(success),
        "source_counts": dict(sorted(Counter(x["source_id"] for x in success).items())),
        "role_counts": dict(sorted(Counter(x["role"] for x in success).items())),
        "label_counts": dict(sorted(Counter(x["label"] for x in success).items())),
        "bytes_by_kind": dict(bytes_by_kind), "error_count": len(errors), "errors": errors,
        "valid": physical_gate_passed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False))
    raise SystemExit(0 if physical_gate_passed else 2)


if __name__ == "__main__":
    main()
