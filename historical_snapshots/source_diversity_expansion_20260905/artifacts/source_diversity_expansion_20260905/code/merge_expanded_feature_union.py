#!/usr/bin/env python3
"""Build one metadata-aligned feature table from frozen and new extractors.

The output intentionally contains measurement values and extraction diagnostics only.
Labels, source names, grouping keys, roles, and evaluation policy stay authoritative in
the separately hashed metadata table used by the evaluator.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from expanded_feature_definitions import D_FEATURES, P_FEATURES, R_FEATURES, S16_FEATURES, S8_FEATURES


FAMILY_COLUMNS = {
    "s16": tuple(f"s16__{name}" for name in S16_FEATURES),
    "s8": tuple(f"s8__{name}" for name in S8_FEATURES),
    "d": tuple(f"d__{name}" for name in D_FEATURES),
    "r": tuple(f"r__{name}" for name in R_FEATURES),
    "p": tuple(f"p__{name}" for name in P_FEATURES),
}
DIAGNOSTIC_COLUMNS = (
    "extractor__native_sample_rate_hz",
    "s16_native_eligible", "s8_native_eligible",
    "s16_vocal_sensitivity_eligible", "s8_vocal_sensitivity_eligible",
    "vocal_analysis_eligible", "vocal_to_mix_rms_db", "vocal_rms_dbfs",
    "vocal_active_frame_count", "vocal_active_frame_ratio",
    "n_beats", "n_downbeats", "n_sections",
    "s16_feature_status", "s8_feature_status", "d_feature_status",
    "r_feature_status", "p_feature_status", "feature_status",
    "feature_error_reason",
)
LEGACY_MAP = {
    **{name: f"d__{name}" for name in D_FEATURES},
    **{name: f"r__{name}" for name in R_FEATURES},
    **{name: f"p__{name}" for name in P_FEATURES},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--legacy-drp", type=Path, action="append", default=[])
    parser.add_argument("--spectral", type=Path, action="append", default=[])
    parser.add_argument("--four-family", type=Path, action="append", default=[])
    parser.add_argument("--serialization-failure-jsonl", type=Path, action="append", default=[])
    parser.add_argument("--allow-extra-input-ids", action="store_true",
                        help="ignore and hash source rows outside authoritative metadata (for the 42 excluded short Heart rows)")
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-accounting-jsonl", type=Path, required=True)
    parser.add_argument("--output-summary-json", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def atomic_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
        handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(value); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def item_id(row: dict[str, str]) -> str:
    return row.get("item_id") or row.get("track") or row.get("id") or ""


def finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def blank_row(identifier: str) -> dict[str, object]:
    row: dict[str, object] = {"item_id": identifier}
    for columns in FAMILY_COLUMNS.values():
        row.update({name: float("nan") for name in columns})
    row.update({name: "" for name in DIAGNOSTIC_COLUMNS})
    return row


def load_unique(paths: list[Path], kind: str) -> dict[str, dict[str, str]]:
    combined: dict[str, dict[str, str]] = {}
    for path in paths:
        for row in read_csv(path):
            identifier = item_id(row)
            if not identifier:
                raise RuntimeError(f"{kind} row without id: {path}")
            if identifier in combined:
                raise RuntimeError(f"duplicate {kind} id {identifier}: {path}")
            combined[identifier] = row
    return combined


def load_serialization_failures(paths: list[Path]) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                raw = json.loads(line)
                identifier = str(raw["item_id"])
                log_path = Path(str(raw.get("log_path", "")))
                log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
                exact = "Not all downbeats are beats" in log_text
                mapped = bool(raw.get("returncode") == 1 and raw.get("output_state") == "missing" and exact)
                result[identifier] = {
                    "item_id": identifier,
                    "reason": "beat_this_serializer_not_all_downbeats_are_beats" if mapped else "unmapped_beat_this_failure",
                    "mapped": mapped,
                    "retry_record_path": str(path),
                    "retry_log_path": str(log_path),
                    "retry_returncode": raw.get("returncode"),
                    "retry_output_state": raw.get("output_state"),
                    "retry_log_sha256": sha256(log_path) if log_path.is_file() else "",
                    "retry_audio_path": raw.get("audio_path", ""),
                    "retry_audio_sha256": raw.get("audio_sha256", ""),
                    "retry_audio_frames": raw.get("audio_frames", ""),
                    "retry_audio_sample_rate_hz": raw.get("audio_sample_rate_hz", ""),
                    "retry_decoded_duration_s": raw.get("decoded_duration_s", ""),
                    "retry_expected_duration_s": raw.get("expected_duration_s", ""),
                }
    return result


def family_has_any(row: dict[str, object], family: str) -> bool:
    return any(finite(row[name]) for name in FAMILY_COLUMNS[family])


def main() -> None:
    args = parse_args()
    metadata = read_csv(args.metadata)
    ids = [item_id(row) for row in metadata]
    if not metadata or "" in ids or len(ids) != len(set(ids)):
        raise RuntimeError("metadata IDs must be nonempty and unique")
    output = {identifier: blank_row(identifier) for identifier in ids}
    authority: dict[str, dict[str, str]] = {identifier: {} for identifier in ids}

    spectral = load_unique(args.spectral, "spectral")
    legacy = load_unique(args.legacy_drp, "legacy-drp")
    current = load_unique(args.four_family, "four-family")
    unexpected = (set(spectral) | set(legacy) | set(current)) - set(ids)
    if unexpected and not args.allow_extra_input_ids:
        raise RuntimeError(f"feature inputs contain {len(unexpected)} IDs absent from metadata; first={sorted(unexpected)[0]}")
    spectral = {key: value for key, value in spectral.items() if key in output}
    legacy = {key: value for key, value in legacy.items() if key in output}
    current = {key: value for key, value in current.items() if key in output}

    for identifier, source in spectral.items():
        target = output[identifier]
        for family in ("s16", "s8"):
            for column in FAMILY_COLUMNS[family]:
                target[column] = source.get(column, "nan")
            computed = str(source.get(f"{family}_computed", "")) == "1" or family_has_any(target, family)
            target[f"{family}_feature_status"] = "computed" if computed else "processing_failure"
            authority[identifier][family] = "spectral"
        for name in DIAGNOSTIC_COLUMNS:
            if name in source and source[name] != "":
                target[name] = source[name]
        if source.get("native_sample_rate_hz"):
            target["extractor__native_sample_rate_hz"] = source["native_sample_rate_hz"]

    for identifier, source in legacy.items():
        target = output[identifier]
        for original, canonical in LEGACY_MAP.items():
            target[canonical] = source.get(original, "nan")
        target["n_beats"] = source.get("n_beats", "")
        target["n_downbeats"] = source.get("n_downbeats", "")
        target["n_sections"] = source.get("n_sections", "")
        target["d_feature_status"] = "eligible" if source.get("dynamics_eligible") == "1" else "unavailable_fewer_than_8_events"
        target["r_feature_status"] = "eligible" if source.get("rhythm_eligible") == "1" else "unavailable_fewer_than_16_beats_or_invalid_intervals"
        target["p_feature_status"] = "eligible" if source.get("structure_eligible") == "1" else "unavailable_fewer_than_3_sections_or_downbeat_spans"
        authority[identifier].update(d="legacy-drp", r="legacy-drp", p="legacy-drp")

    diagnostic_from_current = set(DIAGNOSTIC_COLUMNS) - {"feature_status", "feature_error_reason"}
    for identifier, source in current.items():
        target = output[identifier]
        for family in ("s16", "s8"):
            if family not in authority[identifier]:
                for column in FAMILY_COLUMNS[family]:
                    target[column] = source.get(column, "nan")
                computed = str(source.get(f"{family}_computed", "")) == "1" or family_has_any(target, family)
                target[f"{family}_feature_status"] = "computed" if computed else "processing_failure"
                authority[identifier][family] = "four-family"
        for family in ("d", "r", "p"):
            for column in FAMILY_COLUMNS[family]:
                target[column] = source.get(column, "nan")
            target[f"{family}_feature_status"] = source.get(f"{family}_feature_status", "") or (
                "eligible" if family_has_any(target, family) else "processing_failure"
            )
            authority[identifier][family] = "four-family"
        for name in diagnostic_from_current:
            if name in source and source[name] != "":
                target[name] = source[name]
        if source.get("native_sample_rate_hz"):
            target["extractor__native_sample_rate_hz"] = source["native_sample_rate_hz"]
        target["feature_error_reason"] = source.get("errors", "")

    serialization = load_serialization_failures(args.serialization_failure_jsonl)
    accounting: list[dict[str, object]] = []
    for identifier in ids:
        row = output[identifier]
        missing = []
        for family in FAMILY_COLUMNS:
            if family not in authority[identifier]:
                row[f"{family}_feature_status"] = "missing_feature_source"
                missing.append(family)
        failure = serialization.get(identifier)
        errors = str(row.get("feature_error_reason", ""))
        mapped_failure = failure if failure and "missing_beat_output" in errors else None
        if mapped_failure:
            row["feature_error_reason"] = str(failure["reason"])
        unexplained = []
        for value in [piece.strip() for piece in errors.split("|") if piece.strip()]:
            if value == "missing_beat_output" and mapped_failure and mapped_failure["mapped"]:
                continue
            unexplained.append(value)
        if missing:
            unexplained.append("missing_feature_source:" + ",".join(missing))
        if unexplained:
            row["feature_status"] = "unexplained_failure"
        elif mapped_failure:
            row["feature_status"] = "accounted_serialization_failure"
        else:
            row["feature_status"] = "complete"
        accounting.append({
            "item_id": identifier,
            "family_authority": authority[identifier],
            "feature_status": row["feature_status"],
            "unexplained_reasons": unexplained,
            "serialization_failure": mapped_failure or {},
        })

    fields = ["item_id", *sum((list(FAMILY_COLUMNS[name]) for name in ("s16", "s8", "d", "r", "p")), []), *DIAGNOSTIC_COLUMNS]
    rows = [output[identifier] for identifier in ids]
    atomic_csv(args.output_csv, fields, rows)
    atomic_text(args.output_accounting_jsonl, "".join(json.dumps(row, sort_keys=True) + "\n" for row in accounting))
    summary = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": len(rows),
        "metadata_path": str(args.metadata),
        "metadata_sha256": sha256(args.metadata),
        "feature_csv_path": str(args.output_csv),
        "feature_csv_sha256": sha256(args.output_csv),
        "accounting_path": str(args.output_accounting_jsonl),
        "accounting_sha256": sha256(args.output_accounting_jsonl),
        "feature_status_counts": dict(sorted(Counter(str(row["feature_status"]) for row in rows).items())),
        "family_status_counts": {
            family: dict(sorted(Counter(str(row[f"{family}_feature_status"]) for row in rows).items()))
            for family in FAMILY_COLUMNS
        },
        "input_hashes": {
            str(path): sha256(path)
            for path in [*args.legacy_drp, *args.spectral, *args.four_family, *args.serialization_failure_jsonl]
        },
        "ignored_extra_input_ids": {
            "allowed": args.allow_extra_input_ids,
            "count": len(unexpected),
            "id_set_sha256": hashlib.sha256("".join(f"{value}\n" for value in sorted(unexpected)).encode()).hexdigest(),
            "first_ids": sorted(unexpected)[:20],
        },
    }
    atomic_text(args.output_summary_json, json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
