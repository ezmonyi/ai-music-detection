#!/usr/bin/env python3
"""Validate one generated batch against its frozen manifest and state logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--state-file", type=Path, nargs="+", required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--generator", choices=("heartmula", "acestep"), required=True)
    parser.add_argument("--expected-sample-rate", type=int, required=True)
    parser.add_argument("--expected-duration", type=float, default=30.0)
    parser.add_argument("--duration-tolerance", type=float, default=0.02)
    parser.add_argument("--minimum-duration", type=float)
    parser.add_argument("--maximum-duration", type=float)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def audio_path(directory: Path, generator: str, item_id: str) -> Path:
    suffix = "wav" if generator == "heartmula" else "flac"
    return directory / f"{generator}_{item_id}.{suffix}"


def main() -> None:
    args = parse_args()
    manifest = read_jsonl(args.manifest)
    state = [row for path in args.state_file for row in read_jsonl(path)]
    generated = [row for row in state if row.get("event") == "generated"]
    historical_errors = [row for row in generated if row.get("status") != "ok"]
    final_event_by_id: dict[str, dict[str, object]] = {}
    error_ids: set[str] = set()
    for row in generated:
        item_id = str(row["id"])
        final_event_by_id[item_id] = row
        if row.get("status") != "ok":
            error_ids.add(item_id)
    effective_ok_ids = {
        item_id
        for item_id, row in final_event_by_id.items()
        if row.get("status") == "ok"
    }
    effective_error_ids = {
        item_id
        for item_id, row in final_event_by_id.items()
        if row.get("status") != "ok"
    }
    recovered_error_ids = error_ids & effective_ok_ids
    issues = []
    valid_audio_ids: set[str] = set()
    durations = []
    peaks = []
    rms_values = []
    for row in manifest:
        item_id = str(row["id"])
        path = audio_path(args.audio_dir, args.generator, item_id)
        if not path.exists():
            issues.append({"id": item_id, "issue": "missing"})
            continue
        audio, sample_rate = sf.read(path, always_2d=True, dtype="float32")
        duration = audio.shape[0] / sample_rate
        durations.append(duration)
        peaks.append(float(np.max(np.abs(audio))))
        rms_values.append(float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))))
        current = []
        if sample_rate != args.expected_sample_rate:
            current.append(f"sample_rate={sample_rate}")
        if audio.shape[1] != 2:
            current.append(f"channels={audio.shape[1]}")
        if args.minimum_duration is not None or args.maximum_duration is not None:
            if args.minimum_duration is not None and duration < args.minimum_duration:
                current.append(f"duration_below_min={duration:.6f}")
            if args.maximum_duration is not None and duration > args.maximum_duration:
                current.append(f"duration_above_max={duration:.6f}")
        elif abs(duration - args.expected_duration) > args.duration_tolerance:
            current.append(f"duration={duration:.6f}")
        if not np.isfinite(audio).all():
            current.append("non_finite")
        if current:
            issues.append({"id": item_id, "issue": ";".join(current)})
        else:
            valid_audio_ids.add(item_id)
    files = list(args.audio_dir.glob(f"{args.generator}_*.*"))
    valid_audio_without_ok_state = valid_audio_ids - effective_ok_ids
    unresolved_effective_error_ids = effective_error_ids - valid_audio_ids
    summary = {
        "generator": args.generator,
        "state_files": [str(path) for path in args.state_file],
        "manifest": len(manifest),
        "generated_events": len(generated),
        "historical_error_events": len(historical_errors),
        "effective_ok_ids": len(effective_ok_ids),
        "effective_error_ids": len(effective_error_ids),
        "recovered_error_ids": sorted(recovered_error_ids),
        "valid_audio_ids": len(valid_audio_ids),
        "valid_audio_without_ok_state": sorted(valid_audio_without_ok_state),
        "unresolved_effective_error_ids": sorted(unresolved_effective_error_ids),
        "audio_files": len(files),
        "validation_issues": len(issues),
        "first_issues": issues[:10],
        "duration_min_s": min(durations) if durations else None,
        "duration_median_s": float(np.median(durations)) if durations else None,
        "duration_max_s": max(durations) if durations else None,
        "peak_min": min(peaks) if peaks else None,
        "peak_median": float(np.median(peaks)) if peaks else None,
        "peak_max": max(peaks) if peaks else None,
        "rms_min": min(rms_values) if rms_values else None,
        "rms_median": float(np.median(rms_values)) if rms_values else None,
        "rms_max": max(rms_values) if rms_values else None,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    if (
        len(manifest) != 500
        or len(valid_audio_ids) != len(manifest)
        or unresolved_effective_error_ids
        or issues
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
