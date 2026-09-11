#!/usr/bin/env python3
"""Extract parity-preserving S16 plus partial-bandwidth-control S8."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from expanded_feature_definitions import (
    N_FFT, S16_FEATURES, S8_FEATURES, SR, read_audio, read_spectral_audio,
    spectral_families, vocal_activity,
)


def file_hash(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def json_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=True).encode()).hexdigest()


def load_bias(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        frequencies = np.asarray(payload["frequencies_hz"], dtype=float)
        bias = np.asarray(payload["selected_bias_db"], dtype=float)
    if frequencies.shape != np.fft.rfftfreq(N_FFT, 1/SR).shape or not np.allclose(frequencies, np.fft.rfftfreq(N_FFT, 1/SR)):
        raise RuntimeError("Frozen MUSDB bias frequency grid mismatch")
    return bias


def first(row: dict[str, str], names: tuple[str, ...], default: str = "") -> str:
    return next((str(row[name]) for name in names if row.get(name) not in (None, "")), default)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--demix-root", type=Path, action="append", required=True)
    parser.add_argument("--bias", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retry-partial", action="store_true")
    args = parser.parse_args()
    with args.manifest.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    ids = [first(row, ("item_id", "track", "id")) for row in rows]
    if not rows or "" in ids or len(ids) != len(set(ids)):
        raise RuntimeError("Manifest ids must be nonempty and unique")
    bias = load_bias(args.bias); bias_sha = file_hash(args.bias)
    run_payload = {
        "extractor_sha256": file_hash(Path(__file__)),
        "core_sha256": file_hash(Path(__file__).with_name("expanded_feature_definitions.py")),
        "bias_sha256": bias_sha, "duration": args.duration,
        "demix_roots": [str(path.resolve()) for path in args.demix_root],
    }
    run_fingerprint = json_hash(run_payload)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    state = args.output_dir / f"expanded_spectral_{args.duration:g}s.jsonl"
    existing: dict[str, dict[str, object]] = {}
    if state.exists():
        for line in state.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line); existing[str(item["item_id"])] = item
    if any(item.get("run_fingerprint") != run_fingerprint for item in existing.values()):
        raise RuntimeError("Existing state has a different immutable run fingerprint")

    def process(row: dict[str, str]) -> dict[str, object]:
        item_id = first(row, ("item_id", "track", "id"))
        source = Path(first(row, ("standardized_path", "audio_path", "view_10s_path", "source_audio_path")))
        native_text = first(row, ("native_sr", "native_sample_rate_hz", "original_sample_rate"))
        native_sr = int(float(native_text)) if native_text else None
        vocal = next((root/"htdemucs"/item_id/"vocals.wav" for root in args.demix_root if (root/"htdemucs"/item_id/"vocals.wav").is_file()), None)
        result: dict[str, object] = {
            "item_id": item_id, "label": first(row, ("label", "class_name")),
            "source_id": first(row, ("source_id", "source", "model"), "unknown"),
            "group_id": first(row, ("group_id", "source_group", "condition_id"), item_id),
            "native_sample_rate_hz": native_sr if native_sr else "",
            "s16_native_eligible": int(native_sr is not None and native_sr >= 40_000),
            "s8_native_eligible": int(native_sr is not None and native_sr >= 16_000),
            "run_fingerprint": run_fingerprint,
            **{f"s16__{name}": float("nan") for name in S16_FEATURES},
            **{f"s8__{name}": float("nan") for name in S8_FEATURES},
        }
        errors = []
        if vocal is None:
            errors.append("missing_vocal_stem")
        else:
            try:
                vocals = read_spectral_audio(vocal, args.duration)
                s16, s8 = spectral_families(vocals, bias)
                result.update({f"s16__{key}": value for key, value in s16.items()})
                result.update({f"s8__{key}": value for key, value in s8.items()})
                result.update(s16_computed=1, s8_computed=1, vocal_stem_sha256=file_hash(vocal))
                activity = vocal_activity(read_audio(source, args.duration, strict=True), vocals)
                result.update(activity)
                result["s16_vocal_sensitivity_eligible"] = int(bool(activity["vocal_analysis_eligible"]) and bool(result["s16_native_eligible"]))
                result["s8_vocal_sensitivity_eligible"] = int(bool(activity["vocal_analysis_eligible"]) and bool(result["s8_native_eligible"]))
                result["source_audio_sha256"] = file_hash(source)
            except Exception as exc:
                errors.append(f"spectral:{type(exc).__name__}:{exc}")
        if errors:
            result.update(s16_computed=0, s8_computed=0, vocal_analysis_eligible=0,
                          s16_vocal_sensitivity_eligible=0, s8_vocal_sensitivity_eligible=0)
        result["feature_status"] = "complete" if not errors else "partial"
        result["errors"] = " | ".join(errors)
        result["feature_payload_sha256"] = json_hash({key: value for key, value in result.items() if key.startswith(("s16__", "s8__"))})
        return result

    pending = [row for row in rows if first(row, ("item_id", "track", "id")) not in existing or
               (args.retry_partial and existing[first(row, ("item_id", "track", "id"))].get("feature_status") != "complete")]
    started = time.time()
    with state.open("a", encoding="utf-8") as output, ThreadPoolExecutor(max_workers=args.workers) as executor:
        for index, item in enumerate(executor.map(process, pending), 1):
            existing[str(item["item_id"])] = item
            output.write(json.dumps(item, sort_keys=True, allow_nan=True)+"\n"); output.flush()
            if index == 1 or index % 100 == 0 or index == len(pending):
                print(f"spectral {len(existing)}/{len(rows)} new={index}/{len(pending)} elapsed={time.time()-started:.1f}s", flush=True)
    ordered = [existing[item_id] for item_id in ids]
    csv_path = args.output_dir / f"expanded_spectral_{args.duration:g}s.csv"
    fields: list[str] = []
    for row in ordered: fields.extend(key for key in row if key not in fields)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(ordered)
    metadata = {
        "rows": len(ordered), "complete": sum(x["feature_status"] == "complete" for x in ordered),
        "partial": sum(x["feature_status"] != "complete" for x in ordered),
        "run_fingerprint": run_fingerprint, "run_payload": run_payload,
        "s16_features": list(S16_FEATURES), "s8_features": list(S8_FEATURES),
        "s8_control_scope": "only bins below 7.5 kHz after the unchanged Demucs front end; not a low-pass-before-Demucs control",
        "vocal_gate_scope": "diagnostic and vocal-eligible sensitivity subset only; no overall-cohort row exclusion",
    }
    (args.output_dir/f"expanded_spectral_{args.duration:g}s_metadata.json").write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
