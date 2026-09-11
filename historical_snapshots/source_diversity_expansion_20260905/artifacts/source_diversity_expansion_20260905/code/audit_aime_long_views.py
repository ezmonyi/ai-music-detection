#!/usr/bin/env python3
"""Validate the completed 1,000-record cached-AIME long-view restoration."""
import argparse
import hashlib
import json
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--ffprobe", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    rows = [json.loads(s) for s in args.manifest.read_text().splitlines() if s]
    original = {r["id"]: r for r in [json.loads(s) for s in args.original.read_text().splitlines() if s]
                if r["original_duration_s"] >= 30}
    assert len(rows) == len(set(r["id"] for r in rows)) == 1000
    assert set(original) == set(r["id"] for r in rows)

    def check(row):
        issues = []
        old = original[row["id"]]
        for key in ("raw_sha256", "model", "condition_id", "source_parquet", "source_row_number"):
            if row[key] != old[key]:
                issues.append("changed_original_field:" + key)
        for kind in ("native", "view_30s"):
            path = Path(row[kind + "_path"])
            if not path.is_file() or path.stat().st_size != row[kind + "_bytes"]:
                issues.append("absent_or_changed_size:" + kind)
        start = row["view_30s_crop_start_s"]
        if start > old["crop_start_s"] + 1e-6 or start + 30 < old["crop_start_s"] + 10 - 1e-6:
            issues.append("old_anchor_not_contained")
        result = subprocess.run([args.ffprobe, "-v", "error", "-select_streams", "a:0",
                                 "-show_entries", "stream=sample_rate,channels,codec_name,duration_ts,time_base",
                                 "-of", "json", row["view_30s_path"]],
                                capture_output=True, text=True, timeout=30, check=False)
        try:
            stream = json.loads(result.stdout)["streams"][0]
            n, d = map(int, stream["time_base"].split("/"))
            duration = int(stream["duration_ts"]) * n / d
            if (stream["codec_name"], int(stream["sample_rate"]), int(stream["channels"])) != ("flac", 44100, 2):
                issues.append("noncanonical_format")
            if abs(duration - 30) > 1 / 44100:
                issues.append("not_exact_30s")
        except (ValueError, KeyError, IndexError):
            issues.append("ffprobe_failed")
        return {"id": row["id"], "issues": issues}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        failures = [row for row in pool.map(check, rows) if row["issues"]]
    output = {"rows": len(rows), "model_counts": dict(Counter(r["model"] for r in rows)),
              "exact_30s_views_verified": len(rows) - len(failures), "failures": failures,
              "native_bytes": sum(r["native_bytes"] for r in rows),
              "view_30s_bytes": sum(r["view_30s_bytes"] for r in rows),
              "restoration_manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
              "original_manifest_sha256": hashlib.sha256(args.original.read_bytes()).hexdigest(),
              "native_hash_policy": "embedded native bytes were SHA-256 checked against the frozen manifest during restoration",
              "pass": not failures}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
