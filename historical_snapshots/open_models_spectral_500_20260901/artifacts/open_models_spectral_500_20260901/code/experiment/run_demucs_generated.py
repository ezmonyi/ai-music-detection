#!/usr/bin/env python3
"""Resume-safe htdemucs vocals/no_vocals separation for standardized clips."""

from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


SAMPLE_RATE = 44_100
CHANNELS = 2
FRAMES = 1_323_000
BITS_PER_SAMPLE = 32
STEMS = ("vocals", "no_vocals")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--torch-home", type=Path, required=True)
    parser.add_argument("--state-file", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--gpu-index", type=int, default=0)
    return parser.parse_args()


def wav_metadata(path: Path) -> dict[str, int]:
    with path.open("rb") as handle:
        if handle.read(4) != b"RIFF":
            raise ValueError("not RIFF")
        handle.seek(8)
        if handle.read(4) != b"WAVE":
            raise ValueError("not WAVE")
        metadata: dict[str, int] = {}
        while True:
            chunk_id = handle.read(4)
            if len(chunk_id) != 4:
                break
            size_raw = handle.read(4)
            if len(size_raw) != 4:
                break
            size = struct.unpack("<I", size_raw)[0]
            if chunk_id == b"fmt " and size >= 16:
                payload = handle.read(16)
                audio_format, channels, sample_rate, _, _, bits = struct.unpack(
                    "<HHIIHH", payload
                )
                metadata.update(
                    audio_format=audio_format,
                    channels=channels,
                    sample_rate=sample_rate,
                    bits_per_sample=bits,
                )
                handle.seek(size - 16, 1)
            elif chunk_id == b"data":
                metadata["data_bytes"] = size
                handle.seek(size, 1)
            else:
                handle.seek(size, 1)
            if size % 2:
                handle.seek(1, 1)
        return metadata


def valid_audio(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 10_000_000:
        return False
    try:
        metadata = wav_metadata(path)
    except (OSError, ValueError, struct.error):
        return False
    return (
        metadata.get("audio_format") in {3, 65_534}
        and metadata.get("channels") == CHANNELS
        and metadata.get("sample_rate") == SAMPLE_RATE
        and metadata.get("bits_per_sample") == BITS_PER_SAMPLE
        and metadata.get("data_bytes") == FRAMES * CHANNELS * (BITS_PER_SAMPLE // 8)
    )


def output_paths(output_dir: Path, track: str) -> list[Path]:
    return [output_dir / "htdemucs" / track / f"{stem}.wav" for stem in STEMS]


def append_state(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    args = parse_args()
    tracks = sorted(args.input_dir.glob("*.flac"))
    if not tracks:
        raise SystemExit(f"no FLAC inputs under {args.input_dir}")
    pending = [
        track
        for track in tracks
        if not all(valid_audio(path) for path in output_paths(args.output_dir, track.stem))
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.torch_home.mkdir(parents=True, exist_ok=True)
    print(f"inputs={len(tracks)} pending={len(pending)}", flush=True)
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(args.gpu_index),
            "TORCH_HOME": str(args.torch_home.resolve()),
        }
    )
    started = time.time()
    append_state(
        args.state_file,
        {
            "event": "demucs_start",
            "at_utc": datetime.now(timezone.utc).isoformat(),
            "inputs": len(tracks),
            "pending": len(pending),
            "gpu_index": args.gpu_index,
            "python": str(args.python.resolve()),
        },
    )
    for offset in range(0, len(pending), args.batch_size):
        batch = pending[offset : offset + args.batch_size]
        command = [
            str(args.python),
            "-m",
            "demucs.separate",
            "--name",
            "htdemucs",
            "--two-stems",
            "vocals",
            "--float32",
            "--device",
            "cuda",
            "--shifts",
            "0",
            "--overlap",
            "0.25",
            "--segment",
            "7",
            "--jobs",
            "1",
            "--out",
            str(args.output_dir),
            *map(str, batch),
        ]
        batch_started = time.time()
        subprocess.run(command, check=True, env=environment)
        invalid = [
            str(path)
            for source in batch
            for path in output_paths(args.output_dir, source.stem)
            if not valid_audio(path)
        ]
        if invalid:
            raise RuntimeError("invalid output: " + ", ".join(invalid[:8]))
        append_state(
            args.state_file,
            {
                "event": "demucs_batch",
                "at_utc": datetime.now(timezone.utc).isoformat(),
                "first_track": batch[0].stem,
                "last_track": batch[-1].stem,
                "batch_size": len(batch),
                "batch_elapsed_s": time.time() - batch_started,
                "validated_pending": offset + len(batch),
                "pending_total": len(pending),
            },
        )
        print(
            f"validated={offset + len(batch)}/{len(pending)} "
            f"elapsed={time.time() - started:.1f}s",
            flush=True,
        )
    final_invalid = [
        str(path)
        for source in tracks
        for path in output_paths(args.output_dir, source.stem)
        if not valid_audio(path)
    ]
    if final_invalid:
        raise RuntimeError(f"final validation failed for {len(final_invalid)} outputs")
    append_state(
        args.state_file,
        {
            "event": "demucs_complete",
            "at_utc": datetime.now(timezone.utc).isoformat(),
            "tracks": len(tracks),
            "stems": len(tracks) * len(STEMS),
            "elapsed_s": time.time() - started,
        },
    )
    print(f"complete tracks={len(tracks)} stems={len(tracks) * len(STEMS)}", flush=True)


if __name__ == "__main__":
    main()
