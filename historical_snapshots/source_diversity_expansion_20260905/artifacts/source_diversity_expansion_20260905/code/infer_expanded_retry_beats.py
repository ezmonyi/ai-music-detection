#!/usr/bin/env python3
"""Retry missing Beat This outputs one-by-one and preserve raw error logs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--beat-dir", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument(
        "--expected-duration", type=float,
        help="refuse inputs whose decoded duration differs by more than 20 ms",
    )
    args = parser.parse_args()
    used, utilization = subprocess.check_output([
        "nvidia-smi", f"--id={args.gpu}", "--query-gpu=memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], text=True).strip().split(",")
    if int(used.strip()) > 1024 or int(utilization.strip()) > 5:
        raise RuntimeError(f"Refusing busy GPU {args.gpu}: {used} MiB, {utilization}%")
    with args.manifest.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    args.beat_dir.mkdir(parents=True, exist_ok=True); args.log_dir.mkdir(parents=True, exist_ok=True)
    executable = args.runtime_root/"venv/bin/beat_this"
    checkpoint = args.runtime_root/"checkpoints/hub/checkpoints/beat_this-final0.ckpt"
    state = args.log_dir/"retry_missing_beats.jsonl"
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
    attempted = 0
    for row in rows:
        item_id = row.get("item_id") or row.get("track") or row.get("id") or ""
        audio = row.get("standardized_path") or row.get("audio_path") or row.get("view_10s_path") or ""
        output = args.beat_dir/f"{item_id}.beats"
        if output.exists():
            continue
        audio_path = Path(audio)
        if not audio_path.is_file():
            raise RuntimeError(f"Missing retry audio for {item_id}: {audio_path}")
        import soundfile as sf
        info = sf.info(audio_path)
        duration_s = info.frames / info.samplerate
        if args.expected_duration is not None and abs(duration_s - args.expected_duration) > 0.02:
            raise RuntimeError(
                f"Refusing wrong-duration retry input for {item_id}: "
                f"decoded={duration_s:.9f}s expected={args.expected_duration:g}s path={audio_path}"
            )
        audio_sha256 = sha256_file(audio_path)
        attempted += 1
        command = [str(executable), audio, "-o", str(output), "--model", str(checkpoint), "--no-dbn", "--gpu", "0", "--float16"]
        completed = subprocess.run(command, env=environment, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        log_path = args.log_dir/f"{item_id}.log"
        log_path.write_text(completed.stdout, encoding="utf-8")
        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(), "item_id": item_id,
            "audio_path": audio, "returncode": completed.returncode,
            "audio_sha256": audio_sha256, "decoded_duration_s": duration_s,
            "audio_frames": info.frames, "audio_sample_rate_hz": info.samplerate,
            "expected_duration_s": args.expected_duration,
            "output_state": "nonempty" if output.exists() and output.stat().st_size else "empty" if output.exists() else "missing",
            "log_path": str(log_path), "command": command,
        }
        with state.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True)+"\n")
        print(json.dumps(record, sort_keys=True), flush=True)
    print(json.dumps({"attempted": attempted, "state": str(state)}, sort_keys=True))


if __name__ == "__main__":
    main()
