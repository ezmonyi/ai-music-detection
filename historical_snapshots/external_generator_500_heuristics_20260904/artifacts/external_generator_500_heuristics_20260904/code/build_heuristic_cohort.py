#!/usr/bin/env python3
"""Build a frozen 10 s training-plus-external-test cohort without altering sources."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import soundfile as sf


TRAIN_MANIFEST_SHA256 = "9725a87508bff6cef786ea88d15bf180829292a2d4c0e7ebe809140885377d9e"
TEST_MANIFEST_SHA256 = "49dbe01b8be98857361b3a437db44ee937698e014bf70ccc7d443c24ad342fb4"
SR = 44_100
FRAMES = 441_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate_audio(path: Path) -> None:
    info = sf.info(path)
    if info.samplerate != SR or info.channels != 2 or info.frames != FRAMES:
        raise RuntimeError(f"Unexpected audio standard for {path}: {info}")


def crop_training(source: Path, destination: Path) -> None:
    if destination.exists():
        validate_audio(destination)
        return
    with sf.SoundFile(source) as handle:
        audio = handle.read(FRAMES, dtype="float32", always_2d=True)
        if handle.samplerate != SR or handle.channels != 2 or len(audio) != FRAMES:
            raise RuntimeError(f"Training source cannot supply exact 10 s: {source}")
    temporary = destination.with_suffix(".tmp.flac")
    sf.write(temporary, audio, SR, format="FLAC", subtype="PCM_16")
    validate_audio(temporary)
    os.replace(temporary, destination)


def safe_symlink(source: Path, destination: Path) -> None:
    if destination.is_symlink():
        if destination.resolve() != source.resolve():
            raise RuntimeError(f"Conflicting symlink: {destination}")
    elif destination.exists():
        raise RuntimeError(f"Refusing to replace non-symlink: {destination}")
    else:
        os.symlink(source, destination)
    validate_audio(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--external-manifest", type=Path, required=True)
    parser.add_argument("--external-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    if sha256(args.training_manifest) != TRAIN_MANIFEST_SHA256:
        raise RuntimeError("Training manifest hash does not match the frozen input")
    if sha256(args.external_manifest) != TEST_MANIFEST_SHA256:
        raise RuntimeError("External manifest hash does not match the frozen input")

    args.output_root.mkdir(parents=True, exist_ok=True)
    audio_dir = args.output_root / "audio"
    audio_dir.mkdir(exist_ok=True)
    rows: list[dict[str, object]] = []

    with args.training_manifest.open(newline="", encoding="utf-8") as handle:
        training = [row for row in csv.DictReader(handle) if row["split"] == "development"]
    if len(training) != 1600:
        raise RuntimeError(f"Expected 1600 development rows, got {len(training)}")
    for index, row in enumerate(training, 1):
        track = row["track"]
        source_path = Path(row["source_audio_path"])
        destination = audio_dir / f"{track}.flac"
        crop_training(source_path, destination)
        rows.append({
            "track": track,
            "id": row["id"],
            "label": int(row["label"]),
            "class_name": row["class_name"],
            "source": row["source"],
            "model": row["source"],
            "generator": "human" if int(row["label"]) == 0 else row["source"],
            "split": "development",
            "evaluation_cohort": "training_only_existing_development",
            "condition_id": "",
            "group_id": "",
            "prior_external_exact_track_overlap": 0,
            "audio_filename": destination.name,
            "source_audio_path": str(source_path),
        })
        if index % 200 == 0:
            print(f"training clips {index}/1600", flush=True)

    external = load_jsonl(args.external_manifest)
    if len(external) != 5500:
        raise RuntimeError(f"Expected 5500 external rows, got {len(external)}")
    for row in external:
        track = str(row["id"])
        source_path = args.external_root / str(row["standardized_relpath"])
        destination = audio_dir / f"{track}.flac"
        safe_symlink(source_path, destination)
        rows.append({
            "track": track,
            "id": row["id"],
            "label": int(row["label"]),
            "class_name": row["class_name"],
            "source": row["source"],
            "model": row["model"],
            "generator": "human" if int(row["label"]) == 0 else row["model"],
            "split": "external_test_frozen",
            "evaluation_cohort": row["cohort"],
            "condition_id": row["condition_id"],
            "group_id": row["group_id"],
            "prior_external_exact_track_overlap": int(row["prior_external_exact_track_overlap"]),
            "audio_filename": destination.name,
            "source_audio_path": str(source_path),
        })

    if len(rows) != 7100 or len({str(row["track"]) for row in rows}) != 7100:
        raise RuntimeError("Combined cohort is not 7100 unique tracks")
    fields = list(rows[0])
    manifest = args.output_root / "heuristic_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    counts = Counter((str(row["split"]), str(row["model"])) for row in rows)
    summary = {
        "track_count": len(rows),
        "training_count": len(training),
        "external_test_count": len(external),
        "manifest_sha256": sha256(manifest),
        "training_manifest_sha256": TRAIN_MANIFEST_SHA256,
        "external_manifest_sha256": TEST_MANIFEST_SHA256,
        "counts": {f"{a}|{b}": n for (a, b), n in sorted(counts.items())},
    }
    (args.output_root / "heuristic_manifest_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

