#!/usr/bin/env python3
"""Build a physically verified fixed-duration cohort for F/H/M extraction.

The input is the preserved 30-second selection registry.  This script never
modifies source audio: it resolves the corresponding long original, verifies
its actual libsndfile frame count, and writes deterministic crop coordinates.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import soundfile as sf


SCHEMA_VERSION = 1
FAMILIES = "F,H,M"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def stable_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def resolve_source(
    row: dict[str, str],
    aime_by_id: dict[str, dict[str, Any]],
    legacy_by_id: dict[str, dict[str, Any]],
    legacy_source_root: Path | None,
) -> tuple[str, str]:
    if row["id"] in aime_by_id:
        return str(aime_by_id[row["id"]]["native_path"]), "aime_restored_native"
    if legacy_source_root is not None and row["id"] in legacy_by_id:
        return str(legacy_source_root / str(legacy_by_id[row["id"]]["path"])), "legacy_selected_original"
    path = row.get("source_audio_path", "").strip()
    return path, "selected_registry_source_audio_path" if path else "unresolved"


def probe_candidate(
    item: tuple[dict[str, str], str, str], target_duration_sec: int
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    row, resolved_path, path_basis = item
    base = {
        "id": row["id"],
        "track": row.get("track", row["id"]),
        "label": int(row["label"]),
        "source_id": row["source_id"],
        "source_group": row["source_group"],
        "role": row["role"],
        "original_role": row.get("original_role", ""),
        "group_id": row["group_id"],
        "condition_id": row.get("condition_id", ""),
        "artist_or_creator": row.get("artist_or_creator", ""),
        "generator_family": row.get("generator_family", ""),
        "provenance_status": row.get("provenance_status", ""),
        "acquisition": row.get("acquisition", ""),
        "evaluation_allowed": row.get("evaluation_allowed", ""),
        "source_locator": row.get("source_locator", ""),
        "registered_native_sample_rate_hz": row.get("native_sample_rate_hz", ""),
        "registered_native_duration_sec": row.get("native_duration_s", ""),
        "registered_raw_sha256": row.get("raw_sha256", ""),
        "registered_30s_crop_start_sec": row.get("crop_start_s", ""),
        "registered_30s_audio_path": row.get("audio_path", ""),
        "registered_source_audio_path": row.get("source_audio_path", ""),
        "resolved_source_audio_path": resolved_path,
        "source_path_basis": path_basis,
        "families": FAMILIES,
        "cohort_duration_sec": target_duration_sec,
    }
    if not resolved_path:
        return None, {**base, "exclusion_reason": "unresolved_source_path", "probe_error": ""}
    path = Path(resolved_path)
    try:
        stat = path.stat()
        info = sf.info(path)
        sample_rate = int(info.samplerate)
        frames = int(info.frames)
        channels = int(info.channels)
        target_frames = int(target_duration_sec * sample_rate)
        physical_sha256 = ""
        hash_verification = "not_rehashed_this_run"
        if path_basis == "legacy_selected_original":
            physical_sha256 = sha256_file(path)
            expected_sha256 = str(row.get("raw_sha256") or "")
            hash_verification = "remote_sha256_matches_registered_raw"
            if not expected_sha256 or physical_sha256 != expected_sha256:
                return None, {
                    **base,
                    "exclusion_reason": "physical_hash_mismatch",
                    "probe_error": "",
                    "physical_sample_rate_hz": sample_rate,
                    "physical_frames": frames,
                    "physical_duration_sec": frames / sample_rate if sample_rate else "",
                    "physical_bytes": stat.st_size,
                    "physical_sha256": physical_sha256,
                }
        elif path_basis == "aime_restored_native":
            hash_verification = "upstream_aime_restoration_checked_registered_raw_sha256"
        if sample_rate <= 0 or frames < target_frames:
            return None, {
                **base,
                "exclusion_reason": "physical_duration_short",
                "probe_error": "",
                "physical_sample_rate_hz": sample_rate,
                "physical_frames": frames,
                "physical_duration_sec": frames / sample_rate if sample_rate else "",
                "physical_bytes": stat.st_size,
            }

        normalized_path = resolved_path.replace("\\", "/")
        resolved_is_materialized_view = int(
            any(token in normalized_path for token in ("/views_max60s/", "/views_60s/", "/views_45s/"))
        )
        anchor = float(row.get("crop_start_s") or 0.0)
        registered_view_duration = float(row.get("duration_sec") or 30.0)
        if resolved_is_materialized_view:
            desired_start = 0.0
            crop_anchor_basis = "materialized_view_origin"
        else:
            desired_start = anchor - (target_duration_sec - registered_view_duration) / 2.0
            crop_anchor_basis = "registered_30s_crop_centered_in_native_original"
        max_start_frame = frames - target_frames
        crop_start_frame = min(max(int(round(desired_start * sample_rate)), 0), max_start_frame)
        crop_end_frame = crop_start_frame + target_frames

        # Verify both crop boundaries are readable without decoding the full file.
        with sf.SoundFile(path) as handle:
            handle.seek(crop_start_frame)
            if len(handle.read(frames=1, dtype="float32", always_2d=True)) != 1:
                raise RuntimeError("crop start frame is not readable")
            handle.seek(crop_end_frame - 1)
            if len(handle.read(frames=1, dtype="float32", always_2d=True)) != 1:
                raise RuntimeError("crop end frame is not readable")

        selected = {
            **base,
            "cohort_view_id": f"{row['id']}__{target_duration_sec}s",
            "audio_path": resolved_path,
            "audio_offset_s": crop_start_frame / sample_rate,
            "duration_sec": target_duration_sec,
            "duration_view": f"{target_duration_sec}s",
            "native_sample_rate_hz": row.get("native_sample_rate_hz", ""),
            "native_duration_s": row.get("native_duration_s", ""),
            "available": 1,
            "eligible_common8": int(float(row.get("native_sample_rate_hz") or 0) >= 16_000),
            "crop_start_frame": crop_start_frame,
            "crop_end_frame_exclusive": crop_end_frame,
            "crop_start_sec": crop_start_frame / sample_rate,
            "crop_end_sec": crop_end_frame / sample_rate,
            "crop_frames": target_frames,
            "physical_sample_rate_hz": sample_rate,
            "physical_frames": frames,
            "physical_duration_sec": frames / sample_rate,
            "physical_channels": channels,
            "physical_format": info.format,
            "physical_subtype": info.subtype,
            "physical_bytes": stat.st_size,
            "physical_mtime_ns": stat.st_mtime_ns,
            "physical_sha256": physical_sha256,
            "content_hash_verification": hash_verification,
            "resolved_source_is_materialized_view": resolved_is_materialized_view,
            "crop_anchor_basis": crop_anchor_basis,
            "boundary_decode_verified": 1,
            "status": "verified",
        }
        return selected, None
    except Exception as exc:  # preserve every failure in the audit ledger
        return None, {
            **base,
            "exclusion_reason": "physical_probe_failed",
            "probe_error": f"{type(exc).__name__}: {exc}",
        }


def count_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(key, "")) for row in rows).items()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--aime-long-manifest", type=Path, required=True)
    parser.add_argument("--legacy-manifest", type=Path)
    parser.add_argument("--legacy-source-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--duration-sec", type=int, default=60)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    if args.duration_sec < 45:
        raise ValueError("Long cohort duration must be at least 45 seconds")

    metadata = read_csv(args.metadata)
    if len(metadata) != len({row["id"] for row in metadata}):
        raise ValueError("metadata ids are not unique")
    aime_rows = read_jsonl(args.aime_long_manifest)
    aime_by_id = {str(row["id"]): row for row in aime_rows}
    if len(aime_by_id) != len(aime_rows):
        raise ValueError("AIME long-manifest ids are not unique")
    legacy_rows = read_jsonl(args.legacy_manifest) if args.legacy_manifest else []
    legacy_by_id = {
        f"{'ai' if int(row['label']) else 'human'}_{row['source']}_{row['id']}": row
        for row in legacy_rows
    }
    if len(legacy_by_id) != len(legacy_rows):
        raise ValueError("legacy manifest ids are not unique after canonicalization")
    if bool(args.legacy_manifest) != bool(args.legacy_source_root):
        raise ValueError("--legacy-manifest and --legacy-source-root must be provided together")

    registered_short: list[dict[str, Any]] = []
    candidates: list[tuple[dict[str, str], str, str]] = []
    for row in metadata:
        duration = float(row.get("native_duration_s") or 0.0)
        resolved, basis = resolve_source(
            row, aime_by_id, legacy_by_id, args.legacy_source_root
        )
        if duration + 1e-9 < args.duration_sec:
            registered_short.append(
                {
                    "id": row["id"], "track": row.get("track", row["id"]),
                    "label": int(row["label"]), "source_id": row["source_id"],
                    "source_group": row["source_group"], "role": row["role"],
                    "original_role": row.get("original_role", ""), "group_id": row["group_id"],
                    "condition_id": row.get("condition_id", ""),
                    "artist_or_creator": row.get("artist_or_creator", ""),
                    "generator_family": row.get("generator_family", ""),
                    "provenance_status": row.get("provenance_status", ""),
                    "acquisition": row.get("acquisition", ""),
                    "evaluation_allowed": row.get("evaluation_allowed", ""),
                    "source_locator": row.get("source_locator", ""),
                    "registered_native_sample_rate_hz": row.get("native_sample_rate_hz", ""),
                    "registered_native_duration_sec": row.get("native_duration_s", ""),
                    "registered_raw_sha256": row.get("raw_sha256", ""),
                    "registered_30s_crop_start_sec": row.get("crop_start_s", ""),
                    "registered_30s_audio_path": row.get("audio_path", ""),
                    "registered_source_audio_path": row.get("source_audio_path", ""),
                    "resolved_source_audio_path": resolved,
                    "source_path_basis": basis,
                    "families": FAMILIES, "cohort_duration_sec": args.duration_sec,
                    "exclusion_reason": "registered_duration_short", "probe_error": "",
                }
            )
        else:
            candidates.append((row, resolved, basis))

    selected: list[dict[str, Any]] = []
    physical_excluded: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for included, excluded in pool.map(
            lambda item: probe_candidate(item, args.duration_sec), candidates
        ):
            if included is not None:
                selected.append(included)
            if excluded is not None:
                physical_excluded.append(excluded)
    selected.sort(key=lambda row: row["id"])
    excluded = sorted(registered_short + physical_excluded, key=lambda row: row["id"])

    selected_fields = [
        "id", "track", "cohort_view_id", "label", "source_id", "source_group", "role",
        "original_role", "group_id", "condition_id", "artist_or_creator", "generator_family",
        "provenance_status", "acquisition", "evaluation_allowed", "source_locator", "families",
        "cohort_duration_sec", "registered_native_sample_rate_hz", "registered_native_duration_sec",
        "registered_raw_sha256", "registered_30s_crop_start_sec", "registered_30s_audio_path",
        "registered_source_audio_path", "resolved_source_audio_path", "source_path_basis",
        "audio_path", "audio_offset_s", "duration_sec", "duration_view", "native_sample_rate_hz",
        "native_duration_s", "available", "eligible_common8",
        "crop_start_frame", "crop_end_frame_exclusive", "crop_start_sec", "crop_end_sec",
        "crop_frames", "physical_sample_rate_hz", "physical_frames", "physical_duration_sec",
        "physical_channels", "physical_format", "physical_subtype", "physical_bytes",
        "physical_mtime_ns", "physical_sha256", "content_hash_verification",
        "resolved_source_is_materialized_view", "crop_anchor_basis", "boundary_decode_verified", "status",
    ]
    excluded_fields = [
        "id", "track", "label", "source_id", "source_group", "role", "original_role",
        "group_id", "condition_id", "artist_or_creator", "generator_family", "provenance_status",
        "acquisition", "evaluation_allowed", "source_locator", "families", "cohort_duration_sec",
        "registered_native_sample_rate_hz", "registered_native_duration_sec", "registered_raw_sha256",
        "registered_30s_crop_start_sec", "registered_30s_audio_path", "registered_source_audio_path",
        "resolved_source_audio_path", "source_path_basis", "physical_sample_rate_hz",
        "physical_frames", "physical_duration_sec", "physical_bytes", "physical_sha256",
        "exclusion_reason", "probe_error",
    ]
    cohort_path = args.output_dir / f"cohort_{args.duration_sec}s.csv"
    excluded_path = args.output_dir / f"excluded_{args.duration_sec}s.csv"
    audit_path = args.output_dir / f"source_audit_{args.duration_sec}s.csv"
    summary_path = args.output_dir / f"summary_{args.duration_sec}s.json"
    write_csv(cohort_path, selected, selected_fields)
    write_csv(excluded_path, excluded, excluded_fields)

    audit_groups: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {"input_rows": 0, "selected_rows": 0, "registered_short": 0,
                 "physical_short": 0, "probe_failed": 0}
    )
    for row in metadata:
        audit_groups[(row["source_group"], row["role"])]["input_rows"] += 1
    for row in selected:
        audit_groups[(row["source_group"], row["role"])]["selected_rows"] += 1
    reason_key = {
        "registered_duration_short": "registered_short",
        "physical_duration_short": "physical_short",
        "physical_probe_failed": "probe_failed",
        "unresolved_source_path": "probe_failed",
        "physical_hash_mismatch": "probe_failed",
    }
    for row in excluded:
        audit_groups[(row["source_group"], row["role"])][reason_key[row["exclusion_reason"]]] += 1
    audit_rows = [
        {"source_group": source, "role": role, **counts}
        for (source, role), counts in sorted(audit_groups.items())
    ]
    write_csv(
        audit_path,
        audit_rows,
        ["source_group", "role", "input_rows", "selected_rows", "registered_short",
         "physical_short", "probe_failed"],
    )

    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass" if len(selected) > 0 and not physical_excluded else "pass_with_exclusions",
        "selection_policy": "all preserved selected rows with a physically verified exact fixed-duration crop",
        "families": FAMILIES.split(","),
        "duration_sec": args.duration_sec,
        "input_rows": len(metadata),
        "nominal_candidate_rows": len(candidates),
        "selected_rows": len(selected),
        "excluded_rows": len(excluded),
        "boundary_decode_verified_rows": sum(int(row["boundary_decode_verified"]) for row in selected),
        "remote_sha256_verified_legacy_original_rows": sum(
            row["content_hash_verification"] == "remote_sha256_matches_registered_raw"
            for row in selected
        ),
        "counts_by_source_group": count_by(selected, "source_group"),
        "counts_by_source_id": count_by(selected, "source_id"),
        "counts_by_role": count_by(selected, "role"),
        "counts_by_label": count_by(selected, "label"),
        "exclusion_counts": count_by(excluded, "exclusion_reason"),
        "input_sha256": {
            "metadata": sha256_file(args.metadata),
            "aime_long_manifest": sha256_file(args.aime_long_manifest),
            **(
                {"legacy_manifest": sha256_file(args.legacy_manifest)}
                if args.legacy_manifest else {}
            ),
        },
        "output_sha256": {
            cohort_path.name: sha256_file(cohort_path),
            excluded_path.name: sha256_file(excluded_path),
            audit_path.name: sha256_file(audit_path),
        },
        "host": os.uname().nodename,
        "audio_mutation": "none; source paths opened read-only",
        "fitting_or_scoring": "none",
    }
    stable_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
