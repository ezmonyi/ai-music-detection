#!/usr/bin/env python3
"""Create auditable state rows for valid outputs orphaned by worker interruption."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import soundfile as sf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--state-file", type=Path, nargs="+", required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--generator", choices=("heartmula", "acestep"), required=True)
    parser.add_argument("--output-state", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def target_path(directory: Path, generator: str, item_id: str) -> Path:
    suffix = "wav" if generator == "heartmula" else "flac"
    return directory / f"{generator}_{item_id}.{suffix}"


def main() -> None:
    args = parse_args()
    manifest = read_jsonl(args.manifest)
    output_resolved = args.output_state.resolve()
    state = [
        row
        for path in args.state_file
        if path.resolve() != output_resolved
        for row in read_jsonl(path)
    ]
    ok_ids = {
        str(row["id"])
        for row in state
        if row.get("event") == "generated" and row.get("status") == "ok"
    }
    reconciled = []
    for index, row in enumerate(manifest):
        item_id = str(row["id"])
        if item_id in ok_ids:
            continue
        path = target_path(args.audio_dir, args.generator, item_id)
        if not path.exists():
            continue
        info = sf.info(path)
        if info.frames <= 0 or info.samplerate <= 0 or info.channels <= 0:
            continue
        reconciled.append(
            {
                "event": "generated",
                "status": "ok",
                "generator": args.generator,
                "id": item_id,
                "manifest_index": index,
                "seed": int(row["seed"]),
                "reconciled_existing": True,
                "reconciliation_reason": "valid atomic target existed without an ok event after controlled worker interruption",
                "reconciled_at_utc": datetime.now(timezone.utc).isoformat(),
                "output": str(path.resolve()),
                "audio": {
                    "duration_s": info.frames / info.samplerate,
                    "frames": info.frames,
                    "sample_rate": info.samplerate,
                    "channels": info.channels,
                    "format": info.format,
                    "subtype": info.subtype,
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                },
            }
        )
    args.output_state.parent.mkdir(parents=True, exist_ok=True)
    with args.output_state.open("w", encoding="utf-8") as handle:
        for row in reconciled:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "manifest": len(manifest),
                "existing_ok_state_ids": len(ok_ids),
                "reconciled": len(reconciled),
                "output_state": str(args.output_state.resolve()),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
