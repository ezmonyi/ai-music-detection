#!/usr/bin/env python3
"""Build a non-destructive unified manifest for the existing four-domain cohort."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path


def read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def track_name(row: dict[str, object]) -> str:
    return f"{row['class_name']}_{row['source']}_{row['id']}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--baseline-audio", type=Path, required=True)
    parser.add_argument("--heartmula-manifest", type=Path, required=True)
    parser.add_argument("--heartmula-audio", type=Path, required=True)
    parser.add_argument("--acestep-manifest", type=Path, required=True)
    parser.add_argument("--acestep-audio", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    sources = [
        (args.baseline_manifest, args.baseline_audio, None),
        (args.heartmula_manifest, args.heartmula_audio, "heartmula"),
        (args.acestep_manifest, args.acestep_audio, "acestep"),
    ]
    audio_dir = args.output_root / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    for manifest_path, source_audio, ai_source in sources:
        for row in read_jsonl(manifest_path):
            if ai_source is not None and int(row["label"]) == 0:
                continue
            name = track_name(row)
            source_path = source_audio / f"{name}.flac"
            if not source_path.exists():
                raise FileNotFoundError(source_path)
            destination = audio_dir / f"{name}.flac"
            if destination.is_symlink():
                if destination.resolve() != source_path.resolve():
                    raise RuntimeError(f"Conflicting symlink: {destination}")
            elif destination.exists():
                raise RuntimeError(f"Refusing to replace non-symlink: {destination}")
            else:
                os.symlink(source_path, destination)
            rows.append({
                "track": name,
                "id": row["id"],
                "label": int(row["label"]),
                "class_name": row["class_name"],
                "source": row["source"],
                "split": row["split"],
                "audio_filename": destination.name,
                "source_audio_path": str(source_path),
            })

    rows.sort(key=lambda row: (str(row["source"]), int(row["label"]), str(row["id"])))
    tracks = [str(row["track"]) for row in rows]
    if len(rows) != 2000 or len(set(tracks)) != 2000:
        raise RuntimeError(f"Expected 2000 unique tracks, got {len(rows)}/{len(set(tracks))}")
    counts = Counter((str(row["source"]), str(row["split"])) for row in rows)
    output = args.output_root / "detector_manifest.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"n_rows": len(rows), "counts": {f"{a}|{b}": n for (a, b), n in sorted(counts.items())}}
    (args.output_root / "detector_manifest_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
