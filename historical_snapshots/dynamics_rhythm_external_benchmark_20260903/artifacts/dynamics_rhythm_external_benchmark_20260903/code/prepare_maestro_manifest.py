#!/usr/bin/env python3
"""Freeze the deterministic MAESTRO v3 test recording and window manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


SEED = "20260903"
AUDIO_ARCHIVE = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.zip"
MIDI_ARCHIVE = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip"
FRACTIONS = (0.2, 0.4, 0.6, 0.8)
WINDOW_SECONDS = 30.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selection_key(filename: str) -> str:
    return hashlib.sha256((SEED + filename).encode("utf-8")).hexdigest()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-recordings", type=int, default=50)
    args = parser.parse_args()

    with args.metadata_csv.open(newline="", encoding="utf-8") as handle:
        all_rows = list(csv.DictReader(handle))
    candidates = [row for row in all_rows if row["split"] == "test"]
    candidates.sort(key=lambda row: selection_key(row["audio_filename"]))
    selected = [row for row in candidates if float(row["duration"]) >= WINDOW_SECONDS][
        : args.n_recordings
    ]
    if len(selected) != args.n_recordings:
        raise RuntimeError(f"Only {len(selected)} eligible test recordings")

    recording_rows: list[dict[str, object]] = []
    window_rows: list[dict[str, object]] = []
    for rank, row in enumerate(selected, start=1):
        duration = float(row["duration"])
        recording_id = Path(row["audio_filename"]).stem
        recording_rows.append(
            {
                "selection_rank": rank,
                "selection_key": selection_key(row["audio_filename"]),
                "recording_id": recording_id,
                "canonical_composer": row["canonical_composer"],
                "canonical_title": row["canonical_title"],
                "year": row["year"],
                "split": row["split"],
                "duration_seconds": f"{duration:.9f}",
                "audio_filename": row["audio_filename"],
                "midi_filename": row["midi_filename"],
                "status": "frozen_selected",
            }
        )
        for index, fraction in enumerate(FRACTIONS):
            start = min(max(duration * fraction - WINDOW_SECONDS / 2.0, 0.0), duration - WINDOW_SECONDS)
            window_rows.append(
                {
                    "recording_id": recording_id,
                    "selection_rank": rank,
                    "window_index": index,
                    "center_fraction": f"{fraction:.1f}",
                    "start_seconds": f"{start:.9f}",
                    "duration_seconds": f"{WINDOW_SECONDS:.3f}",
                    "audio_filename": row["audio_filename"],
                    "midi_filename": row["midi_filename"],
                    "status": "frozen_selected",
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    recording_path = args.output_dir / "maestro_recordings.csv"
    window_path = args.output_dir / "maestro_windows.csv"
    write_csv(recording_path, recording_rows)
    write_csv(window_path, window_rows)
    metadata = {
        "dataset": "MAESTRO v3.0.0",
        "seed": int(SEED),
        "eligible_test_recordings": len(
            [row for row in candidates if float(row["duration"]) >= WINDOW_SECONDS]
        ),
        "selected_recordings": len(recording_rows),
        "selected_windows": len(window_rows),
        "distinct_composers": len({row["canonical_composer"] for row in selected}),
        "selected_source_hours": sum(float(row["duration"]) for row in selected) / 3600.0,
        "metadata_csv_sha256": sha256_file(args.metadata_csv),
        "recording_manifest_sha256": sha256_file(recording_path),
        "window_manifest_sha256": sha256_file(window_path),
        "audio_archive": AUDIO_ARCHIVE,
        "midi_archive": MIDI_ARCHIVE,
        "selection_rule": "ascending SHA256('20260903' || canonical audio_filename)",
        "window_rule": "30 s centered at 20%, 40%, 60%, and 80% of annotated duration",
    }
    (args.output_dir / "maestro_manifest_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

