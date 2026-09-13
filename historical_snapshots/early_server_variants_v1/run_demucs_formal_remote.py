#!/usr/bin/env python3
"""Resume-safe four-stem htdemucs runner for the frozen formal experiment."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import torchaudio


STEMS = ("vocals", "drums", "bass", "other")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--vendor", type=Path, required=True)
    parser.add_argument("--hf-home", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=20)
    return parser.parse_args()


def output_paths(output_dir: Path, track: str) -> list[Path]:
    return [output_dir / "htdemucs" / track / f"{stem}.wav" for stem in STEMS]


def valid_audio(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 1_000_000:
        return False
    try:
        metadata = torchaudio.info(path)
        return (
            metadata.sample_rate == 44_100
            and metadata.num_channels == 2
            and metadata.num_frames == 1_323_000
            and metadata.bits_per_sample == 32
            and metadata.encoding == "PCM_F"
        )
    except (OSError, RuntimeError):
        return False


def main() -> None:
    args = parse_args()
    tracks = sorted(args.input_dir.glob("*.flac"))
    if not tracks:
        raise SystemExit(f"no input FLAC files in {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pending = [path for path in tracks if not all(valid_audio(item) for item in output_paths(args.output_dir, path.stem))]
    print(f"inputs={len(tracks)} pending={len(pending)}", flush=True)
    env = os.environ.copy()
    env.update({
        "PYTHONPATH": str(args.vendor),
        "HF_HOME": str(args.hf_home),
        "HF_HUB_OFFLINE": "1",
        "CUDA_VISIBLE_DEVICES": "0",
    })
    started = time.time()
    for offset in range(0, len(pending), args.batch_size):
        batch = pending[offset:offset + args.batch_size]
        command = [
            str(args.python), "-m", "demucs", "-n", "htdemucs",
            "--float32", "--shifts", "0", "--overlap", "0.25",
            "--segment", "7", "-j", "1", "-d", "cuda",
            "-o", str(args.output_dir), *map(str, batch),
        ]
        subprocess.run(command, check=True, env=env)
        invalid = [str(path) for source in batch for path in output_paths(args.output_dir, source.stem) if not valid_audio(path)]
        if invalid:
            raise RuntimeError("invalid outputs: " + ", ".join(invalid[:8]))
        complete = min(offset + len(batch), len(pending))
        print(f"validated={complete}/{len(pending)} elapsed={time.time()-started:.1f}s", flush=True)
    final_invalid = [str(path) for source in tracks for path in output_paths(args.output_dir, source.stem) if not valid_audio(path)]
    if final_invalid:
        raise RuntimeError(f"final validation failed for {len(final_invalid)} files")
    print(f"complete tracks={len(tracks)} stems={len(tracks)*len(STEMS)}", flush=True)


if __name__ == "__main__":
    main()
