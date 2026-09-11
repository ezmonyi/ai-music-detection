#!/usr/bin/env python3
"""Read-only audit of the frozen expansion's native files and derived views."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--materialized", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ffprobe", required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--verify-view-hashes", action="store_true")
    args = parser.parse_args()
    with args.frozen.open(newline="") as handle:
        frozen = {r["item_id"]: r for r in csv.DictReader(handle)}
    records = [json.loads(line) for line in args.materialized.read_text().splitlines() if line.strip()]
    ids = [r["item_id"] for r in records]
    assert len(ids) == len(set(ids)), "Duplicate canonical ledger items"
    successes = [r for r in records if r["status"] == "success"]
    missing = sorted(set(frozen) - set(r["item_id"] for r in successes))

    def audit(record):
        item = record["item_id"]
        problems = []
        if item not in frozen:
            return {"item_id": item, "issues": ["not_in_frozen_selection"]}
        for field in ("source_id", "label", "role", "provenance_grade", "evaluation_allowed"):
            if str(record.get(field)) != str(frozen[item].get(field)):
                problems.append("frozen_field_mismatch:" + field)
        for kind in ("native", "view_10s", "view_max60s"):
            path = Path(record[kind + "_path"])
            if not path.is_file():
                problems.append("missing_file:" + kind)
                continue
            if path.stat().st_size != int(record[kind + "_bytes"]):
                problems.append("byte_count_mismatch:" + kind)
            if kind == "native":
                continue
            if args.verify_view_hashes and digest(path) != record[kind + "_sha256"]:
                problems.append("sha256_mismatch:" + kind)
            response = subprocess.run(
                [args.ffprobe, "-v", "error", "-select_streams", "a:0", "-show_entries",
                 "stream=sample_rate,channels,duration_ts,time_base,codec_name,bits_per_raw_sample", "-of", "json", str(path)],
                capture_output=True, text=True, timeout=30, check=False)
            if response.returncode:
                problems.append("ffprobe_error:" + kind)
                continue
            try:
                stream = json.loads(response.stdout)["streams"][0]
                numerator, denominator = map(int, stream["time_base"].split("/"))
                duration = int(stream["duration_ts"]) * numerator / denominator
                if (int(stream["sample_rate"]), int(stream["channels"]), stream["codec_name"]) != (44100, 2, "flac"):
                    problems.append("format_mismatch:" + kind)
                if int(stream.get("bits_per_raw_sample", 0)) != 16:
                    problems.append("bit_depth_mismatch:" + kind)
                expected = 10.0 if kind == "view_10s" else float(record["view_max60s_duration_s"])
                if abs(duration - expected) > 2 / 44100:
                    problems.append("duration_mismatch:" + kind)
                if kind == "view_max60s" and duration > min(60.0, float(record["native_duration_s"])) + 0.03:
                    problems.append("longer_than_available_native_audio")
            except (KeyError, ValueError, IndexError):
                problems.append("invalid_probe_metadata:" + kind)
        anchor = float(record["crop_start_s"])
        long_start = float(record["long_crop_start_s"])
        long_duration = float(record["view_max60s_duration_s"])
        native_duration = float(record["native_duration_s"])
        if long_start > anchor + 1e-6 or long_start + long_duration + 1e-3 < min(anchor + 10, native_duration):
            problems.append("long_view_does_not_contain_frozen_anchor")
        if float(record.get("view_max60s_padded_s", 0)) != 0:
            problems.append("long_view_was_padded")
        return {"item_id": item, "issues": problems}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        checked = list(pool.map(audit, successes))
    failures = [r for r in checked if r["issues"]]
    output = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "required_view_format": {"sample_rate_hz": 44100, "channels": 2,
                                 "codec": "flac", "bits_per_raw_sample": 16},
        "frozen_rows": len(frozen), "success_ledger_rows": len(successes),
        "audited_rows": len(checked), "missing_selected_ids": missing,
        "audit_failures": failures, "verified_view_sha256": args.verify_view_hashes,
        "native_hash_policy": "native bytes compared to retained download ledger; native digests not recomputed in this audit",
        "native_files": len(successes), "derived_view_files": 2 * len(successes),
        "selected_recordings": len(successes),
        "unique_native_sha256": len(set(r["native_sha256"] for r in successes)),
        "bytes": {kind: sum(int(r[kind + "_bytes"]) for r in successes)
                  for kind in ("native", "view_10s", "view_max60s")},
        "source_counts": dict(Counter(r["source_id"] for r in successes)),
        "native_sample_rate_counts": dict(Counter(str(r["native_sample_rate_hz"]) for r in successes)),
        "native_duration_ge30": sum(float(r["native_duration_s"]) >= 30 - 1e-6 for r in successes),
        "padded_10s_views": sum(float(r.get("view_10s_padded_s", 0)) > 0 for r in successes),
        "frozen_manifest_sha256": digest(args.frozen),
        "materialization_manifest_sha256": digest(args.materialized),
        "complete_and_valid": not missing and not failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({key: value for key, value in output.items() if key not in {"missing_selected_ids", "audit_failures"}}, indent=2))
    if missing or failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
