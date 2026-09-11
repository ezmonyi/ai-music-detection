#!/usr/bin/env python3
"""Resume-safe extraction of S16, common-band S8, D3, R3 and P6."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from expanded_feature_definitions import (
    D_FEATURES, N_FFT, P_FEATURES, R_FEATURES, S16_FEATURES, S8_FEATURES, SR,
    allinone_boundaries, dynamics_features, load_beats, phrase_features,
    read_audio, read_spectral_audio, rhythm_features, spectral_families, vocal_activity,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--bias", type=Path, required=True)
    parser.add_argument("--demix-root", type=Path, action="append", required=True)
    parser.add_argument("--beat-root", type=Path, action="append", required=True)
    parser.add_argument("--structure-root", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--strict", action="store_true", help="abort if any row is incomplete")
    parser.add_argument("--retry-partial", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def sha256_json(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=True).encode()).hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    ids = [row.get("item_id") or row.get("track") or row.get("id") or "" for row in rows]
    if not rows or "" in ids or len(ids) != len(set(ids)):
        raise RuntimeError("Manifest must contain unique item_id/track/id values")
    return rows


def first(row: dict[str, str], names: tuple[str, ...], default: str = "") -> str:
    return next((str(row[name]) for name in names if row.get(name) not in (None, "")), default)


def resolve_file(roots: list[Path], relatives: list[Path], *, allow_empty: bool = False) -> Path | None:
    for root in roots:
        for relative in relatives:
            candidate = root / relative
            if candidate.is_file() and (allow_empty or candidate.stat().st_size):
                return candidate
    return None


def load_bias(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        frequencies = np.asarray(payload["frequencies_hz"], dtype=np.float64)
        bias = np.asarray(payload["selected_bias_db"], dtype=np.float64)
    expected = np.fft.rfftfreq(N_FFT, 1.0 / SR)
    if frequencies.shape != expected.shape or not np.allclose(frequencies, expected):
        raise RuntimeError("Frozen MUSDB bias frequency grid mismatch")
    if not np.all(np.isfinite(bias)):
        raise RuntimeError("Frozen MUSDB bias contains non-finite values")
    return bias


def empty_features() -> dict[str, float]:
    return {
        **{f"s16__{name}": float("nan") for name in S16_FEATURES},
        **{f"s8__{name}": float("nan") for name in S8_FEATURES},
        **{f"d__{name}": float("nan") for name in D_FEATURES},
        **{f"r__{name}": float("nan") for name in R_FEATURES},
        **{f"p__{name}": float("nan") for name in P_FEATURES},
    }


def process(
    row: dict[str, str], args: argparse.Namespace, bias: np.ndarray, bias_sha256: str,
    run_fingerprint: str,
) -> dict[str, object]:
    item_id = first(row, ("item_id", "track", "id"))
    source_audio = Path(first(row, ("standardized_path", "audio_path", "view_10s_path", "source_audio_path")))
    native_sr_raw = first(row, ("native_sr", "native_sample_rate_hz", "original_sample_rate"))
    native_sr = int(float(native_sr_raw)) if native_sr_raw else None
    result: dict[str, object] = {
        "item_id": item_id, "label": first(row, ("label", "class_name")),
        "source_id": first(row, ("source_id", "source", "model"), "unknown"),
        "group_id": first(row, ("group_id", "source_group", "condition_id"), item_id),
        "duration_sec": args.duration, "native_sample_rate_hz": native_sr if native_sr else "",
        "s16_native_eligible": int(native_sr is not None and native_sr >= 40_000),
        "s8_native_eligible": int(native_sr is not None and native_sr >= 16_000),
        "s16_native_eligibility_reason": "native_sr_ge_40000" if native_sr and native_sr >= 40_000 else "native_sr_lt_40000_or_unknown",
        "s8_native_eligibility_reason": "native_sr_ge_16000" if native_sr and native_sr >= 16_000 else "native_sr_lt_16000_or_unknown",
        "bias_sha256": bias_sha256,
        "run_fingerprint": run_fingerprint,
        **empty_features(),
    }
    errors: list[str] = []
    inputs: dict[str, str] = {}
    if not source_audio.is_file():
        errors.append("missing_source_audio")
        mix = None
    else:
        try:
            mix = read_audio(source_audio, args.duration, strict=True)
            inputs["source_audio_sha256"] = sha256_file(source_audio)
        except Exception as exc:  # retain explicit failure rather than fabricating a value
            mix = None; errors.append(f"source_audio:{type(exc).__name__}:{exc}")

    stem_paths = {
        stem: resolve_file(args.demix_root, [Path("htdemucs") / item_id / f"{stem}.wav"])
        for stem in ("bass", "drums", "other", "vocals")
    }
    if all(stem_paths.values()):
        for stem, path in stem_paths.items():
            assert path is not None
            inputs[f"{stem}_sha256"] = sha256_file(path)
        try:
            nonvocal = {stem: read_audio(stem_paths[stem], args.duration, strict=True) for stem in ("bass", "drums", "other")}  # type: ignore[arg-type]
            d_values = dynamics_features(nonvocal["bass"] + nonvocal["drums"] + nonvocal["other"])
            result.update({f"d__{key}": value for key, value in d_values.items()})
            result["d_eligible"] = int(all(math.isfinite(x) for x in d_values.values()))
            result["d_feature_status"] = "eligible" if result["d_eligible"] else "unavailable_fewer_than_8_events"
        except Exception as exc:
            errors.append(f"dynamics:{type(exc).__name__}:{exc}")
            result.update(d_eligible=0, d_feature_status="processing_failure")
        try:
            spectral_vocals = read_spectral_audio(stem_paths["vocals"], args.duration)  # type: ignore[arg-type]
            s16, s8 = spectral_families(spectral_vocals, bias)
            result.update({f"s16__{key}": value for key, value in s16.items()})
            result.update({f"s8__{key}": value for key, value in s8.items()})
            result["s16_computed"] = result["s8_computed"] = 1
        except Exception as exc:
            errors.append(f"spectral:{type(exc).__name__}:{exc}")
            result.update(s16_computed=0, s8_computed=0)
            spectral_vocals = None
        try:
            if mix is not None and spectral_vocals is not None:
                activity = vocal_activity(mix, spectral_vocals)
                result.update(activity)
                result["s16_vocal_sensitivity_eligible"] = int(bool(activity["vocal_analysis_eligible"]) and bool(result["s16_native_eligible"]))
                result["s8_vocal_sensitivity_eligible"] = int(bool(activity["vocal_analysis_eligible"]) and bool(result["s8_native_eligible"]))
            else:
                raise RuntimeError("mix or vocal stem unavailable")
        except Exception as exc:
            errors.append(f"vocal_activity:{type(exc).__name__}:{exc}")
            result.update(vocal_to_mix_rms_db=float("nan"), vocal_rms_dbfs=float("nan"),
                          vocal_active_frame_count=0, vocal_active_frame_ratio=float("nan"),
                          vocal_analysis_eligible=0, s16_vocal_sensitivity_eligible=0,
                          s8_vocal_sensitivity_eligible=0)
    else:
        missing = [stem for stem, path in stem_paths.items() if path is None]
        errors.append("missing_stems:" + ",".join(missing))
        result.update(d_eligible=0, d_feature_status="missing_inference_output", s16_computed=0, s8_computed=0, vocal_analysis_eligible=0,
                      s16_vocal_sensitivity_eligible=0, s8_vocal_sensitivity_eligible=0)

    beat = resolve_file(args.beat_root, [Path(f"{item_id}.beats")], allow_empty=True)
    if beat is None:
        beat_times, beat_numbers = np.asarray([]), np.asarray([], dtype=int)
        errors.append("missing_beat_output")
    else:
        inputs["beats_sha256"] = sha256_file(beat)
        result["beat_output_status"] = "empty" if beat.stat().st_size == 0 else "nonempty"
        try:
            beat_times, beat_numbers = load_beats(beat, args.duration)
        except Exception as exc:
            beat_times, beat_numbers = np.asarray([]), np.asarray([], dtype=int)
            errors.append(f"beats:{type(exc).__name__}:{exc}")
    r_values = rhythm_features(beat_times)
    result.update({f"r__{key}": value for key, value in r_values.items()})
    result.update(n_beats=len(beat_times), n_downbeats=int((beat_numbers == 1).sum()),
                  r_eligible=int(all(math.isfinite(x) for x in r_values.values())))
    result["r_feature_status"] = "eligible" if result["r_eligible"] else "unavailable_fewer_than_16_beats_or_invalid_intervals"

    structure = resolve_file(args.structure_root, [Path(f"{item_id}.json")])
    if structure is None:
        errors.append("missing_structure_output")
        p_values = {name: float("nan") for name in P_FEATURES}
        n_sections = 0
    else:
        inputs["structure_sha256"] = sha256_file(structure)
        try:
            boundaries = allinone_boundaries(structure, args.duration)
            p_values = phrase_features(boundaries, beat_times[beat_numbers == 1])
            n_sections = len(boundaries) - 1
        except Exception as exc:
            errors.append(f"structure:{type(exc).__name__}:{exc}")
            p_values = {name: float("nan") for name in P_FEATURES}; n_sections = 0
    result.update({f"p__{key}": value for key, value in p_values.items()})
    result.update(n_sections=n_sections,
                  p_eligible=int(any(math.isfinite(p_values[name]) for name in P_FEATURES[:4])))
    result["p_feature_status"] = "eligible" if result["p_eligible"] else "unavailable_fewer_than_3_sections_or_downbeat_spans"
    result["input_hashes"] = json.dumps(inputs, sort_keys=True, separators=(",", ":"))
    result["feature_status"] = "complete" if not errors else "partial"
    result["status"] = result["feature_status"]
    result["errors"] = " | ".join(errors)
    result["feature_payload_sha256"] = sha256_json({
        key: value for key, value in result.items()
        if key.startswith(("s16__", "s8__", "d__", "r__", "p__"))
    })
    return result


def main() -> None:
    args = parse_args()
    if args.duration <= 0 or args.workers < 1:
        raise ValueError("duration and workers must be positive")
    rows = read_rows(args.manifest)
    if args.limit is not None:
        rows = rows[:args.limit]
    bias = load_bias(args.bias); bias_hash = sha256_file(args.bias)
    core_path = Path(__file__).with_name("expanded_feature_definitions.py")
    run_payload = {
        "extractor_sha256": sha256_file(Path(__file__)), "core_sha256": sha256_file(core_path),
        "bias_sha256": bias_hash, "duration": args.duration,
        "demix_roots": [str(path.resolve()) for path in args.demix_root],
        "beat_roots": [str(path.resolve()) for path in args.beat_root],
        "structure_roots": [str(path.resolve()) for path in args.structure_root],
    }
    run_fingerprint = sha256_json(run_payload)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.output_dir / f"expanded_features_{args.duration:g}s.jsonl"
    existing: dict[str, dict[str, object]] = {}
    if state_path.exists():
        for line in state_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line); existing[str(item["item_id"])] = item
    mismatched = [item for item in existing.values() if item.get("run_fingerprint") != run_fingerprint]
    if mismatched:
        raise RuntimeError("Existing state was produced by a different immutable run fingerprint")
    pending = [
        row for row in rows
        if first(row, ("item_id", "track", "id")) not in existing
        or (args.retry_partial and existing[first(row, ("item_id", "track", "id"))].get("status") != "complete")
    ]
    started = time.time()
    with state_path.open("a", encoding="utf-8") as output:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for index, item in enumerate(executor.map(lambda row: process(row, args, bias, bias_hash, run_fingerprint), pending), 1):
                existing[str(item["item_id"])] = item
                output.write(json.dumps(item, sort_keys=True, allow_nan=True) + "\n"); output.flush()
                if index == 1 or index % 50 == 0 or index == len(pending):
                    print(f"features {len(existing)}/{len(rows)} new={index}/{len(pending)} elapsed={time.time()-started:.1f}s", flush=True)
    ordered = [existing[first(row, ("item_id", "track", "id"))] for row in rows]
    csv_path = args.output_dir / f"expanded_features_{args.duration:g}s.csv"
    fieldnames: list[str] = []
    for item in ordered:
        fieldnames.extend(key for key in item if key not in fieldnames)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader(); writer.writerows(ordered)
    failures = [item for item in ordered if item["status"] != "complete"]
    metadata = {
        "rows": len(ordered), "complete": len(ordered)-len(failures), "partial": len(failures),
        "duration_sec": args.duration, "sample_rate_hz": SR, "n_fft": N_FFT,
        "s16_features": list(S16_FEATURES), "s8_features": list(S8_FEATURES),
        "d_features": list(D_FEATURES), "r_features": list(R_FEATURES), "p_features": list(P_FEATURES),
        "bias_sha256": bias_hash,
        "run_fingerprint": run_fingerprint, "run_payload": run_payload,
        "s16_native_rule": "native_sample_rate_hz >= 40000; values retained for audit when ineligible",
        "s8_native_rule": "native_sample_rate_hz >= 16000; feature bins <7500 Hz; partial bandwidth control, not a low-pass-before-Demucs control",
        "vocal_gate": "diagnostic/sensitivity only: raw vocals/mix RMS >= -18 dB, vocal RMS >= -60 dBFS, >=16 selected frames; not an overall-cohort exclusion",
        "frame_selection": "raw vocal RMS >= max(P30, max-35 dB), minimum 16 frames",
        "phase_dependent_metrics": False,
        "state_path": str(state_path.resolve()), "csv_path": str(csv_path.resolve()),
        "pid": os.getpid(),
    }
    (args.output_dir / f"expanded_features_{args.duration:g}s_metadata.json").write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf-8")
    if args.strict and failures:
        raise RuntimeError(f"{len(failures)} incomplete feature rows; first={failures[0]['item_id']}: {failures[0]['errors']}")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
