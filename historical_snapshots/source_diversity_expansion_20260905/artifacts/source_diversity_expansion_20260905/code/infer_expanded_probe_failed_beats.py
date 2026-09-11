#!/usr/bin/env python3
"""Preserve raw Beat This beat/downbeat arrays for serializer failures."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, action="append", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    used, utilization = subprocess.check_output([
        "nvidia-smi", f"--id={args.gpu}", "--query-gpu=memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], text=True).strip().split(",")
    if int(used.strip()) > 1024 or int(utilization.strip()) > 5:
        raise RuntimeError(f"Refusing busy GPU {args.gpu}: {used} MiB, {utilization}%")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import torch
    from beat_this.inference import File2Beats

    estimator = File2Beats(args.checkpoint, torch.device("cuda:0"), float16=True, dbn=False)
    arrays: dict[str, np.ndarray] = {}
    details = []
    for audio in args.audio:
        beats, downbeats = estimator(audio)
        beats = np.asarray(beats, dtype=np.float64)
        downbeats = np.asarray(downbeats, dtype=np.float64)
        key = audio.stem
        arrays[f"{key}__beats"] = beats
        arrays[f"{key}__downbeats"] = downbeats
        details.append({
            "item_id": key,
            "audio_path": str(audio),
            "audio_sha256": sha256(audio),
            "beat_count": int(len(beats)),
            "downbeat_count": int(len(downbeats)),
            "all_downbeats_are_beats": bool(np.all(np.isin(downbeats, beats))),
            "beats": beats.tolist(),
            "downbeats": downbeats.tolist(),
        })
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_prefix.with_suffix(".npz"), **arrays)
    payload = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "raw_probe_only_original_serializer_failure_retained",
        "checkpoint_path": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "options": {"dbn": False, "float16": True, "device": "cuda:0"},
        "npz_path": str(args.output_prefix.with_suffix(".npz")),
        "items": details,
    }
    args.output_prefix.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**{key: value for key, value in payload.items() if key != "items"}, "items": [
        {key: value for key, value in item.items() if key not in {"beats", "downbeats"}} for item in details
    ]}, indent=2))


if __name__ == "__main__":
    main()
