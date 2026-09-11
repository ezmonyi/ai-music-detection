#!/usr/bin/env python3
"""Create exact-native-crop 44.1 kHz stereo FLOAT views for matched S/D/R/P60.

Only the frozen development role is admitted. F/H/M measurements are never used
to choose rows; their existing input audit supplies source/crop identity only.
No neural inference, source modification, padding, gain normalization or limiting.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import tempfile

import numpy as np
import scipy
from scipy.signal import resample_poly
import soundfile as sf

SR = 44100
SECONDS = 60
FRAMES = SR * SECONDS
MANIFEST_SHA = "00ff8671c589ba280f2fc2a5bf3019f2d907fdf7a8933afa573f433112a8ee1b"
FEATURE_SHA = "46ea2a81d4653fa8d830e7f31a858ad75999105e8b6ce813032862743c7e51fa"
CONFIG = {
    "duration_sec": SECONDS, "sample_rate_hz": SR, "channels": 2,
    "mono_policy": "duplicate_channel", "multichannel_policy": "reject_above_2",
    "resampler": "scipy.signal.resample_poly", "window": ["kaiser", 5.0],
    "padtype": "constant", "short_input_padding": False,
    "dc_removal": False, "normalization": False, "limiting": False,
    "output_format": "WAV", "output_subtype": "FLOAT",
    "coordinate_system": "exact_frozen_native_crop_then_resample",
}


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def preserve_json(path, value):
    """Atomic first publication; conflicting prior output is never overwritten."""
    path = Path(path)
    body = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text() != body:
            raise RuntimeError(f"Conflicting frozen JSON: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("x") as f:
        f.write(body)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def safe_id(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value) or value in {".", ".."}:
        raise ValueError(f"Unsafe item ID: {value!r}")
    return value


def select_rows(rows, probe=False):
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate IDs in native cohort")
    selected = [r for r in rows if r["role"] == "development"]
    if not selected:
        raise ValueError("No development rows")
    selected.sort(key=lambda r: (r["source_group"], hashlib.sha256(("20260907:" + r["id"]).encode()).hexdigest()))
    if probe:
        first = {}
        for row in selected:
            first.setdefault(row["source_group"], row)
        selected = list(first.values())
    for row in selected:
        safe_id(row["id"])
    return selected


def validate_native(row, reference):
    if row["id"] != reference["id"] or row["role"] != "development" or reference["role"] != "development":
        raise ValueError("Identity/role mismatch")
    if reference["extraction_status"] != "ok":
        raise ValueError("Reference input was not successfully extracted")
    source = Path(row["audio_path"])
    if str(source) != reference["source_audio_path"]:
        raise ValueError("Native path differs from frozen F/H/M path")
    rate = int(row["physical_sample_rate_hz"])
    start, count = int(row["crop_start_frame"]), int(row["crop_frames"])
    if start < 0 or count != SECONDS * rate or start != round(float(row["audio_offset_s"]) * rate):
        raise ValueError("Native crop coordinates inconsistent")
    if int(row["crop_end_frame_exclusive"]) != start + count:
        raise ValueError("Native crop end inconsistent")
    expected = {
        "source_sample_rate": rate, "source_channels": int(row["physical_channels"]),
        "source_total_frames": int(row["physical_frames"]),
        "crop_start_frame": start, "crop_frames": count,
    }
    for name, value in expected.items():
        if int(reference[name]) != value:
            raise ValueError(f"F/H/M crop audit mismatch: {name}")
    if expected["source_channels"] not in (1, 2):
        raise ValueError("Unsupported channel layout")
    return source, expected


def decode_standardized(row, reference):
    source, expected = validate_native(row, reference)
    before = source.stat()
    source_hash = sha_file(source)
    if source_hash != reference["source_audio_sha256"]:
        raise ValueError("Native source hash differs from frozen F/H/M source")
    with sf.SoundFile(source) as f:
        if (f.samplerate, f.channels, f.frames) != (expected["source_sample_rate"], expected["source_channels"], expected["source_total_frames"]):
            raise ValueError("Physical source properties changed")
        if expected["crop_start_frame"] + expected["crop_frames"] > f.frames:
            raise ValueError("Short source: padding forbidden")
        f.seek(expected["crop_start_frame"])
        y = f.read(expected["crop_frames"], dtype="float64", always_2d=True)
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Source changed while reading")
    if len(y) != expected["crop_frames"] or not np.isfinite(y).all():
        raise ValueError("Incomplete/nonfinite native crop")
    if y.shape[1] == 1:
        y = np.repeat(y, 2, axis=1)
    rate = expected["source_sample_rate"]
    if rate != SR:
        divisor = math.gcd(rate, SR)
        y = resample_poly(y, SR // divisor, rate // divisor, axis=0, window=("kaiser", 5.0), padtype="constant")
    if y.shape != (FRAMES, 2) or not np.isfinite(y).all():
        raise ValueError("Standardized waveform is not exact 60 s stereo")
    y = np.ascontiguousarray(y, dtype="<f4")
    return y, {**expected, "source_audio_path": str(source), "source_audio_sha256": source_hash,
               "native_crop_shared_with_fhm": True,
               "standardized_waveform_sha256": hashlib.sha256(y.tobytes()).hexdigest(),
               "standardized_peak_abs": float(np.max(np.abs(y))),
               "standardized_samples_abs_above_one": int(np.count_nonzero(np.abs(y) > 1)),
               "standardized_rms": float(np.sqrt(np.mean(y.astype(np.float64) ** 2)))}


def materialize(row, reference, root, contract_hash):
    item_id = safe_id(row["id"])
    destination = root / "audio" / f"{item_id}.wav"
    receipt_path = root / "items" / f"{item_id}.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt["contract_sha256"] != contract_hash or receipt["input_row_sha256"] != canonical_hash(row):
            raise ValueError("Resume contract mismatch")
        source, _ = validate_native(row, reference)
        if sha_file(source) != receipt["source_audio_sha256"] or sha_file(destination) != receipt["standardized_file_sha256"]:
            raise ValueError("Resume source/output hash mismatch")
        return receipt
    if destination.exists():
        raise ValueError(f"Unreceipted existing audio is quarantined, not reused: {destination}")
    y, audit = decode_standardized(row, reference)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(root).free < y.nbytes + 5 * (1 << 30):
        raise ValueError("Output storage reserve below 5 GiB")
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".exact60-", suffix=".wav", delete=False) as f:
        temporary = Path(f.name)
    try:
        sf.write(temporary, y, SR, format="WAV", subtype="FLOAT")
        back, rate = sf.read(temporary, dtype="float32", always_2d=True)
        if rate != SR or back.shape != y.shape or not np.array_equal(back, y):
            raise ValueError("FLOAT file roundtrip changed waveform")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    receipt = {"item_id": item_id, "status": "verified", "contract_sha256": contract_hash,
               "input_row_sha256": canonical_hash(row), "native_sr": row["native_sample_rate_hz"],
               "label": row["label"], "source_id": row["source_group"], "group_id": row["group_id"],
               "role": row["role"], "evaluation_allowed": row["evaluation_allowed"],
               "standardized_path": str(destination), "audio_offset_s": 0, "requires_crop": 0,
               "duration": SECONDS, "standardized_sr": SR, "standardized_channels": 2,
               "standardized_frames": FRAMES, "standardized_file_bytes": destination.stat().st_size,
               "standardized_file_sha256": sha_file(destination), **audit}
    preserve_json(receipt_path, receipt)
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--reference-features", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--probe-first-per-source", action="store_true")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--shard-size", type=int, default=24)
    a = p.parse_args()
    if a.workers < 1 or a.shard_size < 1:
        raise ValueError("workers/shard-size must be positive")
    for path, expected in ((a.manifest, MANIFEST_SHA), (a.reference_features, FEATURE_SHA)):
        if sha_file(path) != expected:
            raise ValueError(f"Frozen input SHA mismatch: {path}")
    rows = select_rows(list(csv.DictReader(a.manifest.open())), a.probe_first_per_source)
    refs_list = list(csv.DictReader(a.reference_features.open()))
    refs = {r["id"]: r for r in refs_list}
    if len(refs) != len(refs_list):
        raise ValueError("Duplicate reference IDs")
    contract = {"schema_version": 1, "status": "frozen_before_materialization", "configuration": CONFIG,
                "manifest_sha256": MANIFEST_SHA, "reference_features_sha256": FEATURE_SHA,
                "selection": "development_first_hash_per_source_probe" if a.probe_first_per_source else "all_development",
                "item_ids": [r["id"] for r in rows], "code_sha256": sha_file(__file__),
                "runtime": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "soundfile": sf.__version__},
                "shard_size": a.shard_size}
    contract_hash = canonical_hash(contract)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    preserve_json(a.output_dir / "materialization_contract.json", {**contract, "contract_sha256": contract_hash})
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        receipts = list(pool.map(lambda r: materialize(r, refs[r["id"]], a.output_dir, contract_hash), rows))
    manifest = a.output_dir / "inference_manifest.csv"
    import io
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(receipts[0]))
    writer.writeheader(); writer.writerows(receipts)
    body = stream.getvalue()
    if manifest.exists() and manifest.read_bytes() != body.encode():
        raise ValueError("Conflicting inference manifest")
    if not manifest.exists():
        manifest.write_bytes(body.encode())
    shards = []
    for index, start in enumerate(range(0, len(receipts), a.shard_size)):
        selected = receipts[start:start + a.shard_size]
        path = a.output_dir / f"inference_shard_{index:02d}.txt"
        text = "".join(r["standardized_path"] + "\n" for r in selected)
        if path.exists() and path.read_text() != text:
            raise ValueError("Conflicting inference shard")
        if not path.exists():
            path.write_text(text)
        shards.append({"index": index, "rows": len(selected), "sha256": sha_file(path)})
    summary = {"status": "verified", "contract_sha256": contract_hash, "rows": len(receipts),
               "source_counts": dict(Counter(r["source_id"] for r in receipts)),
               "label_counts": dict(Counter(r["label"] for r in receipts)),
               "role_counts": dict(Counter(r["role"] for r in receipts)),
               "output_bytes": sum(r["standardized_file_bytes"] for r in receipts),
               "manifest_sha256": sha_file(manifest), "shards": shards}
    preserve_json(a.output_dir / "materialization_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
