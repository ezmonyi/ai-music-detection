#!/usr/bin/env python3
"""Create exact 30 s, stereo, 44.1 kHz, 16-bit FLAC analysis inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


TARGET_SR = 44_100
TARGET_SECONDS = 30.0
TARGET_FRAMES = int(TARGET_SR * TARGET_SECONDS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--generator", choices=("heartmula", "acestep"), required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-jsonl", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def input_path(input_dir: Path, generator: str, item_id: str) -> Path:
    extension = "wav" if generator == "heartmula" else "flac"
    return input_dir / f"{generator}_{item_id}.{extension}"


def output_path(output_dir: Path, generator: str, item_id: str) -> Path:
    return output_dir / f"ai_{generator}_{item_id}.flac"


def peak_and_rms(waveform: np.ndarray) -> tuple[float, float]:
    peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(waveform, dtype=np.float64)))) if waveform.size else 0.0
    return peak, rms


def standardize(source: Path, target: Path) -> dict[str, object]:
    waveform, source_sr = sf.read(source, always_2d=True, dtype="float32")
    source_frames, source_channels = waveform.shape
    source_peak, source_rms = peak_and_rms(waveform)
    if source_channels == 1:
        waveform = np.repeat(waveform, 2, axis=1)
    elif source_channels > 2:
        waveform = waveform[:, :2]
    if source_sr != TARGET_SR:
        divisor = math.gcd(source_sr, TARGET_SR)
        waveform = resample_poly(
            waveform,
            TARGET_SR // divisor,
            source_sr // divisor,
            axis=0,
        ).astype(np.float32, copy=False)
    frames_before_fix = waveform.shape[0]
    if frames_before_fix < TARGET_FRAMES:
        waveform = np.pad(waveform, ((0, TARGET_FRAMES - frames_before_fix), (0, 0)))
    else:
        waveform = waveform[:TARGET_FRAMES]
    clipped_samples = int((np.abs(waveform) > 1.0).sum())
    waveform = np.clip(waveform, -1.0, 1.0)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f".{target.stem}.partial.flac")
    partial.unlink(missing_ok=True)
    sf.write(partial, waveform, TARGET_SR, format="FLAC", subtype="PCM_16")
    partial.replace(target)
    output_waveform, output_sr = sf.read(target, always_2d=True, dtype="float32")
    output_peak, output_rms = peak_and_rms(output_waveform)
    return {
        "source": str(source.resolve()),
        "source_sha256": sha256_file(source),
        "source_sample_rate": source_sr,
        "source_channels": source_channels,
        "source_frames": source_frames,
        "source_duration_s": source_frames / source_sr,
        "source_peak": source_peak,
        "source_rms": source_rms,
        "resampled_frames_before_fix": frames_before_fix,
        "padding_frames": max(0, TARGET_FRAMES - frames_before_fix),
        "truncated_frames": max(0, frames_before_fix - TARGET_FRAMES),
        "clipped_samples": clipped_samples,
        "output": str(target.resolve()),
        "output_sha256": sha256_file(target),
        "output_sample_rate": output_sr,
        "output_channels": output_waveform.shape[1],
        "output_frames": output_waveform.shape[0],
        "output_duration_s": output_waveform.shape[0] / output_sr,
        "output_peak": output_peak,
        "output_rms": output_rms,
    }


def main() -> None:
    args = parse_args()
    with args.manifest.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    validations = []
    counts: Counter[str] = Counter()
    for index, row in enumerate(rows, 1):
        item_id = str(row["id"])
        source = input_path(args.input_dir, args.generator, item_id)
        target = output_path(args.output_dir, args.generator, item_id)
        if not source.exists():
            validations.append(
                {"generator": args.generator, "id": item_id, "status": "missing", "source": str(source)}
            )
            counts["missing"] += 1
            continue
        try:
            if args.force or not target.exists():
                audio = standardize(source, target)
            else:
                waveform, sr = sf.read(target, always_2d=True, dtype="float32")
                if sr != TARGET_SR or waveform.shape != (TARGET_FRAMES, 2):
                    audio = standardize(source, target)
                else:
                    peak, rms = peak_and_rms(waveform)
                    audio = {
                        "source": str(source.resolve()),
                        "source_sha256": sha256_file(source),
                        "output": str(target.resolve()),
                        "output_sha256": sha256_file(target),
                        "output_sample_rate": sr,
                        "output_channels": waveform.shape[1],
                        "output_frames": waveform.shape[0],
                        "output_duration_s": waveform.shape[0] / sr,
                        "output_peak": peak,
                        "output_rms": rms,
                        "reused": True,
                    }
            validations.append(
                {"generator": args.generator, "id": item_id, "status": "ok", **audio}
            )
            counts["ok"] += 1
        except Exception as exc:
            validations.append(
                {
                    "generator": args.generator,
                    "id": item_id,
                    "status": "error",
                    "source": str(source),
                    "error": repr(exc),
                }
            )
            counts["error"] += 1
        if index == 1 or index % 25 == 0 or index == len(rows):
            print(f"{args.generator} standardized {index}/{len(rows)} {dict(counts)}", flush=True)
    with args.validation_jsonl.open("w", encoding="utf-8") as handle:
        for validation in validations:
            handle.write(json.dumps(validation, ensure_ascii=False, sort_keys=True) + "\n")
    if counts["ok"] != len(rows):
        raise SystemExit(f"not all files standardized: {dict(counts)}")


if __name__ == "__main__":
    main()
