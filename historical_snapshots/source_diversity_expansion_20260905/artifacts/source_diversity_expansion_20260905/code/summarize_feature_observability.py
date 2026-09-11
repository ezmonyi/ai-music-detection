#!/usr/bin/env python3
"""Separate processing failures from natural feature-family unobservability."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--accounting-jsonl", type=Path, required=True)
    parser.add_argument("--duration-view", choices=("10s", "30s"), required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    with args.features.open(encoding="utf-8", newline="") as handle:
        feature_rows = list(csv.DictReader(handle))
    features = {
        str(row.get("item_id") or row.get("track") or row.get("id") or ""): row
        for row in feature_rows
    }
    accounting: dict[str, dict[str, object]] = {}
    with args.accounting_jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                accounting[str(row["item_id"])] = row

    if set(features) != set(accounting) or "" in features:
        raise SystemExit("feature/accounting identity mismatch")

    processing_failures = []
    mapped_failure_ids = set()
    for identifier in sorted(accounting):
        row = accounting[identifier]
        failure = row.get("serialization_failure")
        if failure:
            mapped_failure_ids.add(identifier)
            processing_failures.append({
                "item_id": identifier,
                "overall_status": row.get("feature_status"),
                "failure": failure,
            })

    raw_unavailable: dict[str, list[str]] = {}
    natural_unavailable: dict[str, list[str]] = {}
    serializer_overlap: dict[str, list[str]] = {}
    for family in ("d", "r", "p"):
        unavailable = sorted(
            identifier for identifier, row in features.items()
            if str(row.get(f"{family}_feature_status", "")).startswith("unavailable_")
        )
        overlap = sorted(set(unavailable) & mapped_failure_ids)
        # Beat serialization is a direct processing dependency for R. P's
        # duration/section and downbeat-span gates remain independently natural,
        # so the P overlap is retained in its natural list and disclosed.
        natural = sorted(set(unavailable) - mapped_failure_ids) if family == "r" else unavailable
        raw_unavailable[family] = unavailable
        natural_unavailable[family] = natural
        serializer_overlap[family] = overlap

    payload: dict[str, object] = {
        "schema_version": 1,
        "duration_view": args.duration_view,
        "status": "passed",
        "features": {"path": str(args.features), "sha256": sha256(args.features), "row_count": len(features)},
        "accounting": {"path": str(args.accounting_jsonl), "sha256": sha256(args.accounting_jsonl), "row_count": len(accounting)},
        "processing_failures": {
            "count": len(processing_failures),
            "items": processing_failures,
        },
        "natural_unobservability": {
            "counts": {family: len(ids) for family, ids in natural_unavailable.items()},
            "ids": natural_unavailable,
            "rule": "R excludes mapped Beat This serialization failures; P retains independently satisfied structure/downbeat-span gating even when it overlaps a serializer failure.",
        },
        "raw_unavailable_gating": {
            "counts": {family: len(ids) for family, ids in raw_unavailable.items()},
            "ids": raw_unavailable,
        },
        "mapped_serializer_overlap": {
            "counts": {family: len(ids) for family, ids in serializer_overlap.items()},
            "ids": serializer_overlap,
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(args.output_json, payload)
    print(json.dumps({
        "status": payload["status"],
        "output": str(args.output_json),
        "processing_failure_count": len(processing_failures),
        "natural_counts": payload["natural_unobservability"]["counts"],
        "raw_counts": payload["raw_unavailable_gating"]["counts"],
    }, indent=2))


if __name__ == "__main__":
    main()
