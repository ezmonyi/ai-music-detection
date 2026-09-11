#!/usr/bin/env python3
"""Resumable, label-blind CPU extraction of the frozen F/H/M measurements.

Does not fit a classifier or modify audio. Per-item JSON is authoritative;
features.csv and summary.json are deterministic consolidations. Null measurements
are unavailable, not zeros. A source-dependent feature error stays visible.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time
import traceback

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np
import scipy
import soundfile as sf
import librosa
from audio_inputs import CONFIG as INPUT_CONFIG, load_exact, sha256_file
from phase_features import extract_phase_features, FEATURE_NAMES as F_NAMES, CONFIG as F_CONFIG
from musical_features import extract_musical_features, H_MEASURE_NAMES, M_MEASURE_NAMES


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def digest(value):
    return hashlib.sha256(json.dumps(clean(value), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w") as f:
        json.dump(clean(value), f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def item_path(out, item_id):
    return Path(out) / "items" / (hashlib.sha256(item_id.encode()).hexdigest() + ".json")


def inspect_input(row, duration):
    path = Path(row.get("audio_path") or row.get("source_audio_path") or "")
    info = sf.info(path)
    offset = float(row.get("audio_offset_s") or 0)
    start, count = round(offset * info.samplerate), round(duration * info.samplerate)
    native = float(row.get("native_sample_rate_hz") or info.samplerate)
    if native < 16000:
        raise ValueError("native_sample_rate_below_16000")
    if offset < 0 or start + count > info.frames:
        raise ValueError(f"insufficient_exact_duration:{info.frames}<{start}+{count}")
    return {"checked_path": str(path), "file_bytes": path.stat().st_size,
            "file_mtime_ns": path.stat().st_mtime_ns, "file_sr": info.samplerate,
            "file_channels": info.channels, "file_frames": info.frames,
            "checked_crop_start_frame": start, "checked_crop_frames": count}


def measure(row, duration, preflight_only, contract_hash):
    started = time.monotonic()
    result = dict(row)
    result.update(extraction_status="started", input_row_hash=digest(row),
                  extraction_contract_hash=contract_hash, requested_duration_sec=duration)
    try:
        result.update(inspect_input(row, duration))
        if preflight_only:
            result["extraction_status"] = "preflight_ok"
        else:
            y, audit = load_exact(row, duration=duration)
            result.update(audit)
            errors = []
            for family, extractor in (("F", extract_phase_features), ("HM", extract_musical_features)):
                try:
                    result.update(extractor(y, 16000))
                except Exception as error:
                    result[family + "_extractor_error"] = repr(error)
                    result[family + "_traceback"] = traceback.format_exc()
                    errors.append(family)
            result["extraction_status"] = "ok" if not errors else "partial_error"
    except Exception as error:
        result.update(extraction_status="input_error", extraction_error=repr(error),
                      extraction_traceback=traceback.format_exc())
    result["elapsed_seconds"] = time.monotonic() - started
    return clean(result)


def consolidate(out, rows, contract):
    records = []
    for row in rows:
        path = item_path(out, row["id"])
        if path.exists():
            record = json.loads(path.read_text())
            if record["input_row_hash"] != digest(row) or record["extraction_contract_hash"] != contract["contract_hash"]:
                raise RuntimeError("Cached result contract mismatch: " + row["id"])
            records.append(record)
    columns = list(rows[0]) if rows else []
    columns += sorted(set().union(*(set(r) for r in records)) - set(columns)) if records else []
    path = Path(out) / "features.csv"
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(records)
    os.replace(temporary, path)
    source_counts = defaultdict(Counter)
    for record in records:
        source_counts[record.get("source_group", "unknown")][record["extraction_status"]] += 1
    summary = {"updated_at": datetime.now(timezone.utc).isoformat(), "expected": len(rows),
               "recorded": len(records), "contract_hash": contract["contract_hash"],
               "status_counts": dict(Counter(r["extraction_status"] for r in records)),
               "family_status_counts": {key: dict(Counter(r.get(key, "not_computed") for r in records))
                                        for key in ("F_status", "H_status", "M_status")},
               "by_source": {k: dict(v) for k, v in source_counts.items()},
               "features_csv_sha256": sha256_file(path),
               "complete_accounting": len(records) == len(rows)}
    atomic_json(Path(out) / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=30)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit-per-source", type=int, default=0)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    args = parser.parse_args()
    if args.duration <= 0 or not math.isfinite(args.duration) or args.workers < 1 or args.limit_per_source < 0:
        parser.error("Invalid duration/workers/limit")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "writer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with args.metadata.open(newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows or any(not r.get("id") for r in rows) or len({r["id"] for r in rows}) != len(rows):
            raise ValueError("Metadata must contain nonempty unique IDs")
        if args.limit_per_source:
            grouped = defaultdict(list)
            for row in rows:
                grouped[row["source_group"]].append(row)
            rows = [row for source in sorted(grouped) for row in sorted(grouped[source], key=lambda r: digest(r["id"]))[:args.limit_per_source]]
        code_dir = Path(__file__).resolve().parent
        contract = {"metadata_path": str(args.metadata.resolve()), "metadata_sha256": sha256_file(args.metadata),
                    "selected_ids": [r["id"] for r in rows], "duration": args.duration,
                    "preflight_only": args.preflight_only, "input_config": INPUT_CONFIG,
                    "F_config": F_CONFIG, "feature_names": {"F": F_NAMES, "H": H_MEASURE_NAMES, "M": M_MEASURE_NAMES},
                    "code_sha256": {name: sha256_file(code_dir / name) for name in
                                    ("audio_inputs.py", "phase_features.py", "musical_features.py", "extract_features.py")},
                    "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                                "scipy": scipy.__version__, "soundfile": sf.__version__, "librosa": librosa.__version__}}
        contract["contract_hash"] = digest(contract)
        existing = args.output / "contract.json"
        if existing.exists() and json.loads(existing.read_text())["contract_hash"] != contract["contract_hash"]:
            raise RuntimeError("Output already contains a different frozen contract; use a new version directory")
        atomic_json(existing, contract)
        atomic_json(args.output / "process.json", {"pid": os.getpid(), "hostname": platform.node(),
                    "started_at": datetime.now(timezone.utc).isoformat(), "state": "running"})
        pending = []
        for row in rows:
            cached_path = item_path(args.output, row["id"])
            if not cached_path.exists():
                pending.append(row)
            else:
                record = json.loads(cached_path.read_text())
                if record["input_row_hash"] != digest(row) or record["extraction_contract_hash"] != contract["contract_hash"]:
                    raise RuntimeError("Stale per-item record: " + row["id"])
                if args.retry_errors and record["extraction_status"] in ("input_error", "partial_error"):
                    pending.append(row)
        print(json.dumps({"event": "start", "pid": os.getpid(), "selected": len(rows), "pending": len(pending)}), flush=True)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(measure, row, args.duration, args.preflight_only, contract["contract_hash"]): row for row in pending}
            for index, future in enumerate(as_completed(futures), 1):
                row = futures[future]
                result = future.result()
                atomic_json(item_path(args.output, row["id"]), result)
                if index % 50 == 0 or index == len(pending):
                    print(json.dumps({"event": "progress", "finished_now": index, "pending_at_start": len(pending),
                                      "last_status": result["extraction_status"]}), flush=True)
        summary = consolidate(args.output, rows, contract)
        atomic_json(args.output / "process.json", {"pid": os.getpid(), "hostname": platform.node(),
                    "finished_at": datetime.now(timezone.utc).isoformat(), "state": "finished",
                    "complete_accounting": summary["complete_accounting"]})
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
