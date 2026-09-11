#!/usr/bin/env python3
"""Extract the frozen 10 s dynamics, rhythm, and structure feature families."""

from __future__ import annotations

import argparse
import csv
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from workflow_features import (
    DESCRIPTIVE_STRUCTURE_FEATURES,
    DURATION,
    DYNAMICS_FEATURES,
    RHYTHM_FEATURES,
    STRUCTURE_FEATURES,
    allinone_boundaries,
    dynamics_features,
    load_beats,
    read_audio,
    rhythm_features,
    structure_features,
)


def process(row: dict[str, str], beat_dir: Path, structure_dir: Path, demix_dir: Path) -> dict[str, object]:
    output: dict[str, object] = dict(row)
    paths = [demix_dir / "htdemucs" / row["track"] / f"{stem}.wav" for stem in ("bass", "drums", "other")]
    if all(path.exists() for path in paths):
        instrumental = np.sum([read_audio(path, duration=DURATION) for path in paths], axis=0)
        dynamics = dynamics_features(instrumental)
        output["dynamics_signal"] = "demucs_bass_plus_drums_plus_other"
    else:
        dynamics = {name: float("nan") for name in DYNAMICS_FEATURES}
        output["dynamics_signal"] = "missing_nonvocal_stems"
    output.update(dynamics)
    output["dynamics_eligible"] = int(all(math.isfinite(value) for value in dynamics.values()))

    beat_path = beat_dir / f"{row['track']}.beats"
    if beat_path.exists() and beat_path.stat().st_size:
        beat_times, beat_numbers = load_beats(beat_path)
        beat_mask = beat_times <= DURATION + 1e-6
        beat_times, beat_numbers = beat_times[beat_mask], beat_numbers[beat_mask]
        rhythm = rhythm_features(beat_times)
    else:
        beat_times, beat_numbers = np.asarray([]), np.asarray([], dtype=int)
        rhythm = {name: float("nan") for name in RHYTHM_FEATURES}
    output["n_beats"] = len(beat_times)
    output["n_downbeats"] = int((beat_numbers == 1).sum())
    output.update(rhythm)
    output["rhythm_eligible"] = int(all(math.isfinite(value) for value in rhythm.values()))

    structure_path = structure_dir / f"{row['track']}.json"
    if structure_path.exists() and beat_path.exists():
        boundaries = allinone_boundaries(structure_path, duration=DURATION)
        downbeats = beat_times[beat_numbers == 1]
        structure = structure_features(boundaries, downbeats)
        output["n_sections"] = len(boundaries) - 1
    else:
        structure = {name: float("nan") for name in STRUCTURE_FEATURES + DESCRIPTIVE_STRUCTURE_FEATURES}
        output["n_sections"] = 0
    output.update(structure)
    output["structure_eligible"] = int(any(math.isfinite(structure[name]) for name in STRUCTURE_FEATURES))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--beat-dir", type=Path, required=True)
    parser.add_argument("--structure-dir", type=Path, required=True)
    parser.add_argument("--demix-dir", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(lambda row: process(row, args.beat_dir, args.structure_dir, args.demix_dir), rows))
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"wrote {len(results)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
