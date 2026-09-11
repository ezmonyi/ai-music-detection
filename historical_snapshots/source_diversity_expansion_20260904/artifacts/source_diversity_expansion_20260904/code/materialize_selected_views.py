#!/usr/bin/env python3
"""Download selected public HF items and create the frozen 10-second view.

This is intentionally an audio-I/O script only: it does not run Demucs, a music
understanding model, or detector feature extraction.  Native sample-rate eligibility
is retained so later high-frequency analysis cannot mistake upsampling for bandwidth.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


DEFAULT_ROOT = Path(__file__).resolve().parents[1]
SEED = "ai-human-source-diversity-v2-catalogue-first-20260904"


def stable_fraction(value: str) -> float:
    digest = hashlib.sha256(f"{SEED}|{value}".encode()).hexdigest()
    return int(digest[:16], 16) / float(16**16 - 1)


def choose_start(item: dict[str, str], duration: float) -> float:
    if item["window_start_s"] not in {"", "deterministic_after_probe"}:
        return float(item["window_start_s"])
    if duration <= 10.0:
        return 0.0
    available = duration - 10.0
    # Use the same deterministic interior-window rule as the manifest builder.
    source_key = item["source_id"].removeprefix("human_")
    identity = item["source_path"] if source_key != "hindustani_raag_hf" else item["group_id"]
    return available * (0.1 + 0.8 * stable_fraction(f"{source_key}|{identity}"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe(ffprobe: str, path: Path) -> dict[str, object]:
    result = subprocess.run([
        ffprobe, "-v", "error", "-select_streams", "a:0",
        "-show_entries", "format=duration:stream=sample_rate,channels,codec_name",
        "-of", "json", str(path),
    ], check=True, capture_output=True, text=True)
    value = json.loads(result.stdout)
    stream = value["streams"][0]
    return {
        "duration_s": float(value["format"]["duration"]),
        "sample_rate_hz": int(stream["sample_rate"]),
        "channels": int(stream["channels"]),
        "codec": stream["codec_name"],
    }


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def materialize_item(item: dict[str, str], args: argparse.Namespace) -> dict[str, object]:
    suffix = Path(item["source_path"]).suffix or ".audio"
    raw = args.temp_root / f"{item['item_id']}{suffix}"
    source_dir = args.output_root / item["source_id"]
    source_dir.mkdir(parents=True, exist_ok=True)
    output = source_dir / f"{item['item_id']}.flac"
    temporary_output = source_dir / f".{item['item_id']}.partial.flac"
    try:
        subprocess.run([
            "curl", "-L", "--fail", "--retry", "4", "--retry-delay", "2",
            "--silent", "--show-error", "-o", str(raw), item["source_locator"],
        ], check=True)
        native = probe(args.ffprobe, raw)
        start = choose_start(item, float(native["duration_s"]))
        subprocess.run([
            args.ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-ss", f"{start:.6f}", "-i", str(raw), "-t", "10.000",
            "-af", "apad=pad_dur=10", "-ar", "44100", "-ac", "2",
            "-sample_fmt", "s16", "-c:a", "flac", str(temporary_output),
        ], check=True)
        standardized = probe(args.ffprobe, temporary_output)
        assert abs(float(standardized["duration_s"]) - 10.0) <= 0.02, standardized
        assert standardized["sample_rate_hz"] == 44100 and standardized["channels"] == 2
        os.replace(temporary_output, output)
        return {
            "item_id": item["item_id"],
            "label": item["label"],
            "source_id": item["source_id"],
            "group_id": item["group_id"],
            "source_revision": item["source_revision"],
            "source_locator": item["source_locator"],
            "native_duration_s": native["duration_s"],
            "native_sample_rate_hz": native["sample_rate_hz"],
            "native_channels": native["channels"],
            "native_codec": native["codec"],
            "crop_start_s": round(start, 6),
            "standardized_duration_s": standardized["duration_s"],
            "standardized_sample_rate_hz": standardized["sample_rate_hz"],
            "standardized_channels": standardized["channels"],
            "standardized_codec": standardized["codec"],
            "eligible_through_8khz": int(int(native["sample_rate_hz"]) >= 16000),
            "eligible_through_10khz": int(int(native["sample_rate_hz"]) >= 20000),
            "eligible_through_20khz": int(int(native["sample_rate_hz"]) >= 40000),
            "raw_sha256": sha256(raw),
            "standardized_sha256": sha256(output),
            "standardized_bytes": output.stat().st_size,
            "standardized_path": str(output.resolve()),
        }
    finally:
        raw.unlink(missing_ok=True)
        temporary_output.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--ffmpeg", required=True)
    parser.add_argument("--ffprobe", required=True)
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--temp-root", type=Path, default=Path("/tmp/source-diversity-v2"))
    args = parser.parse_args()

    manifest = args.root / "manifests" / "v2" / "frozen_item_manifest.csv"
    with manifest.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["source_id"] in set(args.source)]
    assert rows, f"No rows for {args.source}"

    args.output_root.mkdir(parents=True, exist_ok=True)
    args.temp_root.mkdir(parents=True, exist_ok=True)
    state_path = args.output_root / "materialization_manifest.jsonl"
    completed: dict[str, dict[str, object]] = {}
    if state_path.exists():
        for line in state_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                completed[str(item["item_id"])] = item

    pending = []
    for row in rows:
        old = completed.get(row["item_id"])
        if old and Path(str(old["standardized_path"])).exists():
            continue
        pending.append(row)
    if args.limit is not None:
        pending = pending[: args.limit]

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(materialize_item, item, args): item for item in pending}
        for index, future in enumerate(as_completed(futures), 1):
            item = futures[future]
            record = future.result()
            completed[item["item_id"]] = record
            with state_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(json.dumps({
                "progress": f"{index}/{len(pending)}",
                "item_id": item["item_id"],
                "native_hz": record["native_sample_rate_hz"],
                "bytes": record["standardized_bytes"],
            }), flush=True)

    all_records = list(completed.values())
    atomic_json(args.output_root / "materialization_summary.json", {
        "completed_items": len(all_records),
        "source_counts": Counter(str(x["source_id"]) for x in all_records),
        "native_sample_rate_counts": Counter(str(x["native_sample_rate_hz"]) for x in all_records),
        "eligible_through_20khz": sum(int(x["eligible_through_20khz"]) for x in all_records),
        "standardized_bytes": sum(int(x["standardized_bytes"]) for x in all_records),
    })


if __name__ == "__main__":
    main()
