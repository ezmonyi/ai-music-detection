#!/usr/bin/env python3
"""Resume-safe ACE-Step 1.5 Turbo inference for a frozen prompt manifest."""

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

from acestep.handler import AceStepHandler
from acestep.inference import GenerationConfig, GenerationParams, generate_music
from acestep.llm_inference import LLMHandler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--state-file", type=Path, required=True)
    parser.add_argument("--duration-seconds", type=float, default=30.0)
    parser.add_argument("--config-path", default="acestep-v15-turbo")
    parser.add_argument("--inference-steps", type=int, default=8)
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


def append_state(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scratch = args.output_dir / ".acestep_scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(args.manifest)
    stop = len(rows) if args.limit is None else min(len(rows), args.start_index + args.limit)
    rows = rows[args.start_index:stop]

    init_started = time.monotonic()
    dit_handler = AceStepHandler()
    status, success = dit_handler.initialize_service(
        project_root=str(args.project_root),
        config_path=args.config_path,
        device="cuda",
        use_flash_attention=False,
        compile_model=False,
        offload_to_cpu=False,
        offload_dit_to_cpu=False,
        quantization=None,
    )
    if not success:
        raise SystemExit(f"ACE-Step initialization failed: {status}")
    llm_handler = LLMHandler()  # Deliberately not initialized: direct, prompt-controlled DiT path.
    append_state(
        args.state_file,
        {
            "event": "model_initialized",
            "at_utc": utc_now(),
            "elapsed_s": time.monotonic() - init_started,
            "status_message": status,
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "cuda_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "project_root": str(args.project_root.resolve()),
            "config_path": args.config_path,
        },
    )

    for offset, row in enumerate(rows, args.start_index):
        item_id = str(row["id"])
        target = args.output_dir / f"acestep_{item_id}.flac"
        if not args.force and valid_existing(target):
            print(f"[{offset + 1}/{stop}] skip {item_id}", flush=True)
            continue
        seed = int(row["seed"])
        started_at = utc_now()
        started = time.monotonic()
        try:
            set_seed(seed)
            params = GenerationParams(
                task_type="text2music",
                caption=str(row["acestep_caption"]),
                lyrics=str(row["lyrics_30s"]),
                instrumental=False,
                vocal_language=str(row["language"]),
                duration=args.duration_seconds,
                enable_normalization=False,
                inference_steps=args.inference_steps,
                seed=seed,
                thinking=False,
                use_cot_metas=False,
                use_cot_caption=False,
                use_cot_lyrics=False,
                use_cot_language=False,
            )
            config = GenerationConfig(
                batch_size=1,
                use_random_seed=False,
                seeds=[seed],
                audio_format="flac",
            )
            with torch.inference_mode():
                result = generate_music(
                    dit_handler,
                    llm_handler,
                    params,
                    config,
                    save_dir=str(scratch),
                )
            if not result.success or not result.audios:
                raise RuntimeError(result.error or result.status_message or "no audio returned")
            generated_path = Path(str(result.audios[0].get("path", "")))
            if not generated_path.exists():
                raise RuntimeError(f"returned audio path does not exist: {generated_path}")
            meta = audio_metadata(generated_path)
            if float(meta["duration_s"]) < 5.0:
                raise RuntimeError(f"generated audio is too short: {meta['duration_s']:.3f}s")
            os.replace(generated_path, target)
            meta = audio_metadata(target)
            state = {
                "event": "generated",
                "status": "ok",
                "generator": "acestep",
                "id": item_id,
                "manifest_index": offset,
                "seed": seed,
                "started_at_utc": started_at,
                "finished_at_utc": utc_now(),
                "elapsed_s": time.monotonic() - started,
                "output": str(target.resolve()),
                "audio": meta,
                "result_status_message": result.status_message,
                "time_costs": result.extra_outputs.get("time_costs", {}),
            }
            append_state(args.state_file, state)
            print(
                f"[{offset + 1}/{stop}] ok {item_id} "
                f"duration={meta['duration_s']:.3f}s elapsed={state['elapsed_s']:.1f}s",
                flush=True,
            )
            del result
        except Exception as exc:
            append_state(
                args.state_file,
                {
                    "event": "generated",
                    "status": "error",
                    "generator": "acestep",
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
