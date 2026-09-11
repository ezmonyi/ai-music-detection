#!/usr/bin/env python3
"""Restore selected long AIME items from already cached pinned parquet shards."""
import argparse
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import pyarrow.parquet as pq


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--parquet", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--ffmpeg", required=True)
    p.add_argument("--workers", type=int, default=6)
    a = p.parse_args()
    rows = [json.loads(s) for s in a.manifest.read_text().splitlines() if s]
    selected = [r for r in rows if r["original_duration_s"] >= 30]
    assert len(selected) == 1000
    assert Counter(r["model"] for r in selected) == {"MTG-Jamendo": 500, "Udio": 500}
    byshard = defaultdict(list)
    for r in selected:
        byshard[r["source_parquet"]].append(r)
    a.output.mkdir(parents=True, exist_ok=True)
    ledger = a.output / "aime_long_manifest.jsonl"
    completed = {r["id"]: r for r in [json.loads(s) for s in ledger.read_text().splitlines() if s]} if ledger.exists() else {}

    def process(row, audio):
        blob = bytes(audio["bytes"])
        if hashlib.sha256(blob).hexdigest() != row["raw_sha256"]:
            raise RuntimeError("AIME native hash mismatch: " + row["id"])
        suffix = Path(audio.get("path") or "").suffix
        if blob[:4] == b"RIFF": suffix = ".wav"
        elif blob[:4] == b"fLaC": suffix = ".flac"
        elif not suffix: suffix = ".mp3"
        native = a.output / "native" / row["source"] / (row["id"] + suffix)
        target = a.output / "views_30s" / row["source"] / (row["id"] + ".flac")
        native.parent.mkdir(parents=True, exist_ok=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not native.exists():
            tmp = native.with_suffix(native.suffix + ".partial")
            tmp.write_bytes(blob)
            os.replace(tmp, native)
        start = max(0.0, min(float(row["crop_start_s"]) - 10, float(row["original_duration_s"]) - 30))
        if not target.exists():
            tmp = target.with_suffix(".partial.flac")
            subprocess.run([a.ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-threads", "1",
                            "-ss", f"{start:.6f}", "-i", str(native), "-t", "30", "-ar", "44100",
                            "-ac", "2", "-sample_fmt", "s16", "-c:a", "flac", str(tmp)], check=True)
            os.replace(tmp, target)
        return {**row, "native_path": str(native), "native_bytes": native.stat().st_size,
                "view_30s_path": str(target), "view_30s_sha256": digest(target),
                "view_30s_bytes": target.stat().st_size, "view_30s_crop_start_s": start,
                "status": "success"}

    with ledger.open("a") as log, ThreadPoolExecutor(max_workers=a.workers) as pool:
        for shard, items in sorted(byshard.items()):
            pending = [r for r in items if r["id"] not in completed]
            if not pending:
                continue
            table = pq.read_table(a.parquet / shard, columns=["id", "model", "audio"])
            futures = []
            for row in pending:
                i = int(row["source_row_number"])
                original = table.slice(i, 1).to_pylist()[0]
                if str(original["id"]) != str(row["hf_original_id"]) or original["model"] != row["model"]:
                    raise RuntimeError("AIME parquet identity mismatch: " + row["id"])
                futures.append(pool.submit(process, row, original["audio"]))
            for f in futures:
                result = f.result()
                completed[result["id"]] = result
                log.write(json.dumps(result) + "\n")
                log.flush()
            print(json.dumps({"completed": len(completed), "total": 1000, "shard": shard}), flush=True)
    summary = {"completed": len(completed), "models": dict(Counter(r["model"] for r in completed.values())),
               "native_bytes": sum(r["native_bytes"] for r in completed.values()),
               "view_30s_bytes": sum(r["view_30s_bytes"] for r in completed.values())}
    (a.output / "aime_long_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
