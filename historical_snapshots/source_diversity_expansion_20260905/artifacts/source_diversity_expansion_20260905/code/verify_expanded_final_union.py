#!/usr/bin/env python3
"""Emit the positive, hash-bound completion gate required by formal evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from expanded_feature_schema import D_FEATURES, P_FEATURES, R_FEATURES, S16_FEATURES, S8_FEATURES


FAMILIES = {
    "s16": tuple(f"s16__{name}" for name in S16_FEATURES),
    "s8": tuple(f"s8__{name}" for name in S8_FEATURES),
    "d": tuple(f"d__{name}" for name in D_FEATURES),
    "r": tuple(f"r__{name}" for name in R_FEATURES),
    "p": tuple(f"p__{name}" for name in P_FEATURES),
}


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def id_digest(ids: set[str]) -> str:
    payload = "".join(f"{value}\n" for value in sorted(ids)).encode()
    return hashlib.sha256(payload).hexdigest()


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def row_id(row: dict[str, str]) -> str:
    return row.get("item_id") or row.get("track") or row.get("id") or ""


def duplicates(values: list[str]) -> list[str]:
    counts = Counter(values)
    return sorted(value for value, count in counts.items() if count > 1)


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2); handle.write("\n")
        handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--accounting-jsonl", type=Path, required=True)
    parser.add_argument("--duration-view", choices=("10s", "30s"), required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    metadata, _ = read_csv(args.metadata)
    features, columns = read_csv(args.features)
    metadata_ids_list = [row_id(row) for row in metadata]
    feature_ids_list = [row_id(row) for row in features]
    metadata_ids, feature_ids = set(metadata_ids_list), set(feature_ids_list)
    metadata_duplicates, feature_duplicates = duplicates(metadata_ids_list), duplicates(feature_ids_list)
    missing_ids = sorted(metadata_ids - feature_ids)
    unexpected_ids = sorted(feature_ids - metadata_ids)

    required_columns = {name for names in FAMILIES.values() for name in names}
    absent_columns = sorted(required_columns - set(columns))
    feature_by_id = {row_id(row): row for row in features}
    status_counts = {
        family: dict(sorted(Counter(feature_by_id[identifier].get(f"{family}_feature_status", "") for identifier in feature_ids).items()))
        for family in FAMILIES
    }
    status_counts["overall"] = dict(sorted(Counter(row.get("feature_status", "") for row in features).items()))

    accounting = []
    with args.accounting_jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                accounting.append(json.loads(line))
    accounting_ids = [str(row.get("item_id", "")) for row in accounting]
    accounting_duplicates = duplicates(accounting_ids)
    accounting_id_match = set(accounting_ids) == metadata_ids and not accounting_duplicates
    unexplained = sorted(
        str(row["item_id"]) for row in accounting
        if row.get("unexplained_reasons") or row.get("feature_status") == "unexplained_failure"
    )
    serialization = [row for row in accounting if row.get("serialization_failure")]
    mapped_serialization = [row for row in serialization if row["serialization_failure"].get("mapped")]
    serialization_reasons = Counter(str(row["serialization_failure"].get("reason", "")) for row in serialization)
    natural_prefix = "unavailable_"
    raw_unavailable = {
        family: sum(count for status, count in counts.items() if status.startswith(natural_prefix))
        for family, counts in status_counts.items() if family in {"d", "r", "p"}
    }
    mapped_ids = {str(row["item_id"]) for row in mapped_serialization}
    serializer_unavailable_overlap = {
        family: sum(
            str(feature_by_id[identifier].get(f"{family}_feature_status", "")).startswith(natural_prefix)
            for identifier in mapped_ids
        )
        for family in ("d", "r", "p")
    }
    # R is directly unobservable when Beat This cannot serialize its beat array,
    # so those mapped failures are not natural missingness. P retains its raw
    # structural-gating count: section-duration observability is independently
    # determined by structure boundaries, and bar observability has a separate
    # downbeat-span gate. Do not infer P missingness from the overall failure.
    natural_missing = dict(raw_unavailable)
    natural_missing["r"] = raw_unavailable.get("r", 0) - serializer_unavailable_overlap["r"]

    row_counts_exact = len(metadata) == len(features) == args.expected_rows
    unique_counts_exact = len(metadata_ids) == len(feature_ids) == args.expected_rows
    exact_id_set = metadata_ids == feature_ids
    passed = bool(
        row_counts_exact and unique_counts_exact and exact_id_set
        and not metadata_duplicates and not feature_duplicates and not absent_columns
        and accounting_id_match and not unexplained
        and len(mapped_serialization) == len(serialization)
    )
    payload: dict[str, object] = {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "passed": passed,
        "duration_view": args.duration_view,
        "metadata": {
            "path": str(args.metadata), "sha256": sha256(args.metadata),
            "row_count": len(metadata), "unique_id_count": len(metadata_ids),
            "id_set_sha256": id_digest(metadata_ids),
        },
        "features": {
            "path": str(args.features), "sha256": sha256(args.features),
            "row_count": len(features), "unique_id_count": len(feature_ids),
            "id_set_sha256": id_digest(feature_ids),
        },
        "expected": {"metadata_row_count": args.expected_rows, "feature_row_count": args.expected_rows},
        "identity": {
            "exact_id_set_match": exact_id_set,
            "duplicate_metadata_ids": metadata_duplicates,
            "duplicate_feature_ids": feature_duplicates,
            "missing_feature_ids": missing_ids,
            "unexpected_feature_ids": unexpected_ids,
        },
        "feature_status_counts": status_counts,
        "failure_accounting": {
            "unexplained_failure_count": len(unexplained),
            "unexplained_failure_ids": unexplained,
            "serialization_failure_count": len(serialization),
            "serialization_failure_reason_counts": dict(sorted(serialization_reasons.items())),
            "serialization_failures_mapped_count": len(mapped_serialization),
            "serialization_failures_all_mapped": len(mapped_serialization) == len(serialization),
            "serialization_failure_manifest_path": str(args.accounting_jsonl),
            "serialization_failure_manifest_sha256": sha256(args.accounting_jsonl),
            "natural_missing_counts": natural_missing,
            "raw_unavailable_gating_counts": raw_unavailable,
            "mapped_serializer_unavailable_overlap_counts": serializer_unavailable_overlap,
            "natural_missing_rule": "R excludes mapped Beat This serialization failures; D is unchanged; P retains independent structure/downbeat-span gating rather than being inferred from overall serializer status.",
            "natural_missing_allowed": True,
        },
        "schema": {
            "required_feature_column_count": len(required_columns),
            "missing_required_feature_columns": absent_columns,
            "accounting_exact_id_set_match": accounting_id_match,
        },
        "hash_definition": {
            "id_set_sha256": "sha256 of UTF-8 sorted unique IDs joined by newline with trailing newline",
            "file_sha256": "raw file bytes",
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(args.output_json, payload)
    print(json.dumps(payload, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
