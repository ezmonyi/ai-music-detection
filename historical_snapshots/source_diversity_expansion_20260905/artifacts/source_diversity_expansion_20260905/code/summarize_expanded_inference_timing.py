#!/usr/bin/env python3
"""Preserve per-stage counts, bytes, and filesystem timing spans for one frozen shard."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--inference-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    with args.manifest.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    ids = [row.get("item_id") or row.get("track") or row.get("id") or "" for row in rows]
    if not rows or "" in ids or len(ids) != len(set(ids)):
        raise RuntimeError("manifest IDs must be nonempty and unique")
    stages: dict[str, list[Path]] = {
        "beats": [args.inference_root/"beats"/f"{identifier}.beats" for identifier in ids],
        "stems": [args.inference_root/"demix"/"htdemucs"/identifier/f"{stem}.wav" for identifier in ids for stem in ("bass", "drums", "other", "vocals")],
        "spectrograms": [args.inference_root/"spec"/f"{identifier}.npy" for identifier in ids],
        "structures": [args.inference_root/"structure"/f"{identifier}.json" for identifier in ids],
    }
    stage_payload = {}
    all_times = []
    for stage, expected in stages.items():
        present = [path for path in expected if path.is_file()]
        times = [path.stat().st_mtime for path in present]
        all_times.extend(times)
        stage_payload[stage] = {
            "expected_files": len(expected), "present_files": len(present),
            "missing_files": len(expected)-len(present),
            "zero_byte_files": sum(path.stat().st_size == 0 for path in present),
            "total_bytes": sum(path.stat().st_size for path in present),
            "first_output_utc": datetime.fromtimestamp(min(times), timezone.utc).isoformat() if times else None,
            "last_output_utc": datetime.fromtimestamp(max(times), timezone.utc).isoformat() if times else None,
            "output_span_s": max(times)-min(times) if times else None,
        }
    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(args.manifest), "manifest_sha256": sha256(args.manifest),
        "inference_root": str(args.inference_root), "item_count": len(ids),
        "stages": stage_payload,
        "first_any_output_utc": datetime.fromtimestamp(min(all_times), timezone.utc).isoformat() if all_times else None,
        "last_any_output_utc": datetime.fromtimestamp(max(all_times), timezone.utc).isoformat() if all_times else None,
        "output_span_s": max(all_times)-min(all_times) if all_times else None,
        "interpretation": "mtime output spans document this preserved realization; they are not controlled cross-source speed benchmarks",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_json.with_name(f".{args.output_json.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2); handle.write("\n")
        handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, args.output_json)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
