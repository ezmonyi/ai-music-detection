#!/usr/bin/env python3
"""Resume-safe pairwise inference for the frozen Music Flamingo probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import time
from pathlib import Path

import torch
import transformers
from huggingface_hub import HfApi
from transformers import AutoProcessor, MusicFlamingoForConditionalGeneration


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--experiment-root", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--prompts", type=Path, required=True)
    p.add_argument("--state", type=Path, required=True)
    p.add_argument("--model-id", default="nvidia/music-flamingo-2601-hf")
    p.add_argument(
        "--model-path",
        type=Path,
        help="Optional local snapshot path; model-id and Hub revision remain the provenance identity.",
    )
    p.add_argument("--limit", type=int)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=64)
    return p.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        row["query_id"]
        for row in load_jsonl(path)
        if row.get("status") == "ok" and row.get("query_id")
    }


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def exact_json_choice(text: str) -> tuple[str | None, bool]:
    stripped = text.strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return None, False
    if not isinstance(payload, dict) or set(payload) != {"choice"}:
        return None, False
    choice = payload["choice"]
    return (choice, True) if choice in {"A", "B", "TIE"} else (None, False)


def gpu_snapshot() -> str:
    try:
        return subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader",
            ],
            text=True,
            timeout=20,
        ).strip()
    except Exception as exc:
        return f"unavailable: {exc!r}"


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.manifest)[args.start :]
    if args.limit is not None:
        rows = rows[: args.limit]
    prompts = json.loads(args.prompts.read_text(encoding="utf-8"))
    done = completed_ids(args.state)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; this experiment must run on an RTX 5090")
    repo_sha = HfApi().model_info(args.model_id).sha
    load_ref = str(args.model_path.resolve()) if args.model_path else args.model_id
    load_revision = None if args.model_path else repo_sha
    run_metadata = {
        "model_id": args.model_id,
        "model_revision": repo_sha,
        "model_load_path": load_ref,
        "manifest_sha256": sha256_file(args.manifest),
        "prompts_sha256": sha256_file(args.prompts),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "python": platform.python_version(),
        "hostname": platform.node(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "hf_home": os.environ.get("HF_HOME"),
        "gpu_before_load": gpu_snapshot(),
        "decode": {"do_sample": False, "max_new_tokens": args.max_new_tokens},
        "started_unix": time.time(),
    }
    metadata_path = args.state.with_suffix(".metadata.json")
    run_metadata["state_path"] = str(args.state)
    run_metadata["existing_completed_queries"] = len(done)
    metadata_path.write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")

    processor = AutoProcessor.from_pretrained(load_ref, revision=load_revision)
    run_metadata["processor_sampling_rate"] = getattr(processor.feature_extractor, "sampling_rate", None)
    model = MusicFlamingoForConditionalGeneration.from_pretrained(
        load_ref,
        revision=load_revision,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).to("cuda:0")
    model.eval()
    run_metadata["gpu_after_load"] = gpu_snapshot()
    metadata_path.write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")

    for idx, row in enumerate(rows):
        if row["query_id"] in done:
            continue
        paired_audio = (args.experiment_root / row["paired_audio"]).resolve()
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompts[row["task"]]},
                    {"type": "audio", "path": str(paired_audio)},
                ],
            }
        ]
        started = time.time()
        out = {
            **row,
            "manifest_index": args.start + idx,
            "model_id": args.model_id,
            "model_revision": repo_sha,
            "started_unix": started,
        }
        try:
            inputs = processor.apply_chat_template(
                conversation,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
            ).to("cuda:0")
            inputs["input_features"] = inputs["input_features"].to(model.dtype)
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                )
            text = processor.batch_decode(
                generated[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
            )[0]
            choice, parse_ok = exact_json_choice(text)
            out.update(
                {
                    "status": "ok",
                    "raw_response": text,
                    "parsed_choice": choice,
                    "exact_json_parse": parse_ok,
                    "correct": bool(parse_ok and choice == row["correct_choice"]),
                    "elapsed_s": time.time() - started,
                    "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                }
            )
        except Exception as exc:
            out.update(
                {
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": repr(exc),
                    "elapsed_s": time.time() - started,
                }
            )
        append_jsonl(args.state, out)
        print(json.dumps({k: out.get(k) for k in ("query_id", "status", "raw_response", "error", "elapsed_s")}))

    run_metadata["finished_unix"] = time.time()
    run_metadata["gpu_after_run"] = gpu_snapshot()
    metadata_path.write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
