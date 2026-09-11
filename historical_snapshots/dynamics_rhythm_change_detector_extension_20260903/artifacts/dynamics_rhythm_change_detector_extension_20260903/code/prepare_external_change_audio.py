#!/usr/bin/env python3
"""Create fixed controlled-change audio pairs from external MAESTRO and SALAMI."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample

from workflow_features import DURATION, N_SAMPLES, SR, peak_protect, read_audio


def write_audio(path: Path, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, peak_protect(audio), SR, subtype="PCM_16")


def smooth_gain_envelope() -> np.ndarray:
    anchors_seconds = np.arange(0.0, DURATION + 2.5, 2.5)
    anchors_db = np.asarray([-6, 6, -6, 6, -6, 6, -6, 6, -6, 6, -6, 6, -6], dtype=float)
    sample_times = np.arange(N_SAMPLES) / SR
    gain_db = np.interp(sample_times, anchors_seconds[: len(anchors_db)], anchors_db)
    return np.power(10.0, gain_db / 20.0).astype(np.float32)


def speed_change(audio: np.ndarray, speed: float) -> np.ndarray:
    output_length = max(1, int(round(len(audio) / speed)))
    return resample(audio, output_length, axis=0).astype(np.float32)


def ensure_length(audio: np.ndarray, length: int = N_SAMPLES) -> np.ndarray:
    if len(audio) < length:
        repeats = int(np.ceil(length / max(len(audio), 1)))
        audio = np.tile(audio, (repeats, 1))
    return audio[:length]


def excerpt(audio: np.ndarray, start: int, length: int) -> np.ndarray:
    part = audio[start : start + length]
    return ensure_length(part, length)


def mosaic(source: np.ndarray, lengths_seconds: list[float]) -> np.ndarray:
    max_length = int(round(max(lengths_seconds) * SR))
    available = max(1, len(source) - max_length)
    starts = [int(fraction * available) for fraction in (0.08, 0.34, 0.60, 0.84)]
    pieces = []
    for start, seconds in zip(starts, lengths_seconds):
        length = int(round(seconds * SR))
        pieces.append(excerpt(source, start, length))
    return ensure_length(np.concatenate(pieces, axis=0))


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--maestro-windows", type=Path, required=True)
    parser.add_argument("--maestro-audio-root", type=Path, required=True)
    parser.add_argument("--salami-manifest", type=Path, required=True)
    parser.add_argument("--salami-audio-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    with args.maestro_windows.open(newline="", encoding="utf-8") as handle:
        maestro = [row for row in csv.DictReader(handle) if row["window_index"] == "0"]
    gain = smooth_gain_envelope()
    for row in maestro:
        item_id = row["recording_id"]
        audio = read_audio(
            args.maestro_audio_root / row["audio_filename"],
            float(row["start_seconds"]), DURATION,
        )
        variants = {
            "original": audio,
            "constant_gain": audio * np.float32(10.0 ** (-6.0 / 20.0)),
            "variable_gain": audio * gain[:, None],
        }
        for condition, rendered in variants.items():
            name = f"dyn__{item_id}__{condition}.wav"
            path = args.output_root / "dynamics" / name
            write_audio(path, rendered)
            rows.append({
                "family": "dynamics", "item_id": item_id, "condition": condition,
                "filename": name, "sha256": file_hash(path),
            })

    with args.salami_manifest.open(newline="", encoding="utf-8") as handle:
        salami = list(csv.DictReader(handle))
    for row in salami:
        item_id = row["song_id"]
        candidates = sorted(args.salami_audio_root.glob(f"{item_id}.*"))
        if len(candidates) != 1:
            raise RuntimeError(f"Expected one SALAMI audio file for {item_id}, got {candidates}")
        path = candidates[0]
        source40 = read_audio(path, 0.0, 40.0)
        control = source40[:N_SAMPLES]
        constant = ensure_length(speed_change(source40, 0.8))
        blocks = np.array_split(source40, 10)
        variable = ensure_length(np.concatenate([
            speed_change(block, 0.8 if index % 2 == 0 else 1.25)
            for index, block in enumerate(blocks)
        ], axis=0))
        for condition, rendered in {
            "original": control,
            "constant_tempo": constant,
            "variable_tempo": variable,
        }.items():
            name = f"rhythm__{item_id}__{condition}.wav"
            output = args.output_root / "rhythm" / name
            write_audio(output, rendered)
            rows.append({
                "family": "rhythm", "item_id": item_id, "condition": condition,
                "filename": name, "sha256": file_hash(output),
            })

        source70 = read_audio(path, 0.0, 70.0)
        for condition, rendered in {
            "equal_sections": mosaic(source70, [7.5, 7.5, 7.5, 7.5]),
            "variable_sections": mosaic(source70, [3.0, 5.0, 8.0, 14.0]),
        }.items():
            name = f"structure__{item_id}__{condition}.wav"
            output = args.output_root / "structure" / name
            write_audio(output, rendered)
            rows.append({
                "family": "structure", "item_id": item_id, "condition": condition,
                "filename": name, "sha256": file_hash(output),
            })

    manifest = args.output_root / "controlled_change_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "n_dynamics_items": len(maestro),
        "n_rhythm_items": len(salami),
        "n_structure_items": len(salami),
        "sample_rate": SR,
        "duration_seconds": DURATION,
        "dynamics_gain_anchor_db": [-6, 6],
        "rhythm_speed_factors": [0.8, 1.25],
        "equal_section_lengths_seconds": [7.5, 7.5, 7.5, 7.5],
        "variable_section_lengths_seconds": [3.0, 5.0, 8.0, 14.0],
    }
    (args.output_root / "controlled_change_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
