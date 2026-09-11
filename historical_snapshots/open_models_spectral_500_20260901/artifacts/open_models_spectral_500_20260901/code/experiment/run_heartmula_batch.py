#!/usr/bin/env python3
"""Resume-safe HeartMuLa batch inference for a frozen prompt manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from heartlib import HeartMuLaGenPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--rendered-input-dir",
        type=Path,
        help="Directory that preserves the exact per-item lyrics/tags files passed to HeartMuLa.",
    )
    parser.add_argument("--state-file", type=Path, required=True)
    parser.add_argument("--duration-seconds", type=float, default=30.0)
    parser.add_argument("--mula-device", default="cuda:0")
    parser.add_argument("--codec-device", default="cuda:1")
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_state(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def audio_metadata(path: Path) -> dict[str, object]:
    info = sf.info(path)
    return {
        "duration_s": info.frames / info.samplerate,
        "frames": info.frames,
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "format": info.format,
        "subtype": info.subtype,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def valid_existing(path: Path) -> bool:
    try:
        meta = audio_metadata(path)
        return float(meta["duration_s"]) >= 5.0 and int(meta["frames"]) > 0
    except Exception:
        return False


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rendered_input_dir = args.rendered_input_dir or (args.output_dir / "rendered_inputs")
    rendered_input_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(args.manifest)
    stop = len(rows) if args.limit is None else min(len(rows), args.start_index + args.limit)
    rows = rows[args.start_index:stop]

    init_started = time.monotonic()
    pipeline = HeartMuLaGenPipeline.from_pretrained(
        str(args.model_root),
        device={
            "mula": torch.device(args.mula_device),
            "codec": torch.device(args.codec_device),
        },
        dtype={"mula": torch.bfloat16, "codec": torch.float32},
        version="3B",
        lazy_load=False,
    )
    append_state(
        args.state_file,
        {
            "event": "model_initialized",
            "at_utc": utc_now(),
            "elapsed_s": time.monotonic() - init_started,
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "cuda_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "model_root": str(args.model_root.resolve()),
        },
    )

    for offset, row in enumerate(rows, args.start_index):
        item_id = str(row["id"])
        target = args.output_dir / f"heartmula_{item_id}.wav"
        if not args.force and valid_existing(target):
            print(f"[{offset + 1}/{stop}] skip {item_id}", flush=True)
            continue
        partial = args.output_dir / f".heartmula_{item_id}.partial.wav"
        partial.unlink(missing_ok=True)
        seed = int(row["seed"])
        lyrics_path = rendered_input_dir / f"{item_id}.lyrics.txt"
        tags_path = rendered_input_dir / f"{item_id}.tags.txt"
        lyrics_path.write_text(str(row["lyrics_30s"]).strip() + "\n", encoding="utf-8")
        tags_path.write_text(str(row["heartmula_tags"]).strip() + "\n", encoding="utf-8")
        started_at = utc_now()
        started = time.monotonic()
        try:
            set_seed(seed)
            with torch.inference_mode():
                pipeline(
                    {"lyrics": str(lyrics_path), "tags": str(tags_path)},
                    max_audio_length_ms=round(args.duration_seconds * 1000),
                    save_path=str(partial),
                    topk=args.topk,
                    temperature=args.temperature,
                    cfg_scale=args.cfg_scale,
                )
            meta = audio_metadata(partial)
            if float(meta["duration_s"]) < 5.0:
                raise RuntimeError(f"generated audio is too short: {meta['duration_s']:.3f}s")
            os.replace(partial, target)
            meta = audio_metadata(target)
            state = {
                "event": "generated",
                "status": "ok",
                "generator": "heartmula",
                "id": item_id,
                "manifest_index": offset,
                "seed": seed,
                "started_at_utc": started_at,
                "finished_at_utc": utc_now(),
                "elapsed_s": time.monotonic() - started,
                "output": str(target.resolve()),
                "audio": meta,
            }
            append_state(args.state_file, state)
            print(
                f"[{offset + 1}/{stop}] ok {item_id} "
                f"duration={meta['duration_s']:.3f}s elapsed={state['elapsed_s']:.1f}s",
                flush=True,
            )
        except Exception as exc:
            partial.unlink(missing_ok=True)
            append_state(
                args.state_file,
                {
                    "event": "generated",
                    "status": "error",
                    "generator": "heartmula",
                    "id": item_id,
                    "manifest_index": offset,
                    "seed": seed,
                    "started_at_utc": started_at,
                    "finished_at_utc": utc_now(),
                    "elapsed_s": time.monotonic() - started,
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                },
            )
            print(f"[{offset + 1}/{stop}] ERROR {item_id}: {exc!r}", flush=True)


if __name__ == "__main__":
    main()
