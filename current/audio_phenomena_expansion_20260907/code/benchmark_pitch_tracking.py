#!/usr/bin/env python3
"""Frozen CPU pYIN benchmark on the selected MIR-1K vocal channels.

This program is deliberately independent of the AI-vs-human classifier.  It
validates only whether a fixed pYIN configuration is usable as a coarse F0
trajectory estimator on isolated MIR-1K vocal channels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


# Avoid nested numerical-library pools inside the explicitly bounded process pool.
for _thread_env_name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_env_name, "1")


SCHEMA_VERSION = 1
EXPECTED_CLIPS = 100
MAX_WORKERS = 8
VOCAL_CHANNEL_INDEX = 1
REFERENCE_FRAME_PERIOD_SEC = 0.02
REFERENCE_FIRST_FRAME_TIME_SEC = 0.02

# Frozen before any MIR-1K pYIN scores were computed.  librosa 0.11.0 marks
# win_length as deprecated, so it is intentionally omitted.  Every other pYIN
# parameter is explicit here, including options that equal upstream defaults.
PYIN_CONFIG: dict[str, Any] = {
    "sr": 16000,
    "fmin": 65.40639132514966,       # C2
    "fmax": 1046.5022612023945,      # C6
    "frame_length": 1024,
    "hop_length": 160,
    "n_thresholds": 100,
    "beta_parameters": [2.0, 18.0],
    "boltzmann_parameter": 2.0,
    "resolution": 0.1,
    "max_transition_rate": 35.92,
    "switch_prob": 0.01,
    "no_trough_prob": 0.01,
    "fill_na": "nan",
    "center": True,
    "pad_mode": "constant",
}

# This criterion is a measurement gate, not a model-selection objective.  It was
# declared before observing benchmark scores and must not be changed post hoc.
PREDECLARED_USABILITY_CRITERION: dict[str, Any] = {
    "required_requested_clips": 100,
    "required_successful_clips": 100,
    "required_failure_count": 0,
    "require_all_input_hashes_verified": True,
    "minimum_aggregate_raw_pitch_accuracy_50c": 0.70,
    "minimum_aggregate_voicing_f1": 0.80,
    "scope_if_passed": "coarse_F0_trajectory_on_isolated_MIR1K_vocal_channel",
    "explicitly_not_validated": [
        "vibrato_rate_or_extent_measurement",
        "source_separation_or_full_mix_robustness",
        "AI_vs_human_class_discrimination",
        "causal_or_scalar_feature_validity",
    ],
}


COUNT_FIELDS = (
    "total_reference_frames",
    "reference_voiced_frames",
    "reference_unvoiced_frames",
    "estimated_voiced_frames",
    "both_voiced_frames",
    "voicing_true_positive_frames",
    "correct_unvoiced_frames",
    "pitch_correct_50c_frames",
    "chroma_correct_50c_frames",
)

METRIC_FIELDS = (
    "raw_pitch_accuracy_50c",
    "raw_chroma_accuracy_50c",
    "voicing_precision",
    "voicing_recall",
    "voicing_f1",
    "overall_accuracy",
    "signed_cents_mean",
    "signed_cents_median",
    "absolute_cents_mean",
    "absolute_cents_median",
    "absolute_cents_rmse",
    "absolute_cents_p90",
    "absolute_cents_p95",
)


@dataclass(frozen=True)
class ClipSpec:
    stem: str
    singer: str
    wav_path: Path
    pitch_path: Path
    expected_wav_sha256: str
    expected_pitch_sha256: str


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def midi_to_hz(midi: np.ndarray) -> np.ndarray:
    midi = np.asarray(midi, dtype=np.float64)
    out = np.zeros_like(midi)
    voiced = midi > 0.0
    out[voiced] = 440.0 * np.power(2.0, (midi[voiced] - 69.0) / 12.0)
    return out


def read_pitch_label(path: Path) -> np.ndarray:
    tokens = path.read_text(encoding="utf-8").split()
    if not tokens:
        raise ValueError(f"empty pitch label: {path}")
    values = np.asarray([float(token) for token in tokens], dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError(f"pitch label must contain finite, nonnegative MIDI values: {path}")
    return values


def reference_times(label_count: int) -> np.ndarray:
    """Return the verified MIR-1K time grid: 0.02 + 0.02*n seconds."""
    if label_count < 0:
        raise ValueError("label_count must be nonnegative")
    return REFERENCE_FIRST_FRAME_TIME_SEC + (
        np.arange(label_count, dtype=np.float64) * REFERENCE_FRAME_PERIOD_SEC
    )


def map_estimate_to_reference(
    reference_time_sec: Sequence[float],
    estimate_time_sec: Sequence[float],
    estimate_f0_hz: Sequence[float],
    estimate_voiced: Sequence[bool],
    *,
    atol_sec: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray]:
    """Map estimator frames to reference times without bridging unvoiced gaps.

    An exact estimator timestamp uses that estimator frame.  A timestamp between
    frames is linearly interpolated only when both adjacent frames are voiced
    and carry finite positive F0; because they are adjacent, they are in the same
    contiguous voiced run.  Out-of-range timestamps and any interval touching an
    unvoiced frame map to unvoiced/NaN.
    """
    ref_t = np.asarray(reference_time_sec, dtype=np.float64)
    est_t = np.asarray(estimate_time_sec, dtype=np.float64)
    est_f0 = np.asarray(estimate_f0_hz, dtype=np.float64)
    est_v = np.asarray(estimate_voiced, dtype=bool)
    if not (len(est_t) == len(est_f0) == len(est_v)):
        raise ValueError("estimate arrays must have equal lengths")
    if len(est_t) == 0 or np.any(np.diff(est_t) <= 0.0):
        raise ValueError("estimate timestamps must be nonempty and strictly increasing")

    mapped_f0 = np.full(len(ref_t), np.nan, dtype=np.float64)
    mapped_voiced = np.zeros(len(ref_t), dtype=bool)
    right_indices = np.searchsorted(est_t, ref_t, side="left")

    for out_index, (time_value, right) in enumerate(zip(ref_t, right_indices)):
        if right < len(est_t) and math.isclose(
            float(est_t[right]), float(time_value), rel_tol=0.0, abs_tol=atol_sec
        ):
            if est_v[right] and np.isfinite(est_f0[right]) and est_f0[right] > 0.0:
                mapped_voiced[out_index] = True
                mapped_f0[out_index] = est_f0[right]
            continue
        left = int(right) - 1
        if left < 0 or right >= len(est_t):
            continue
        if not (
            est_v[left]
            and est_v[right]
            and np.isfinite(est_f0[left])
            and np.isfinite(est_f0[right])
            and est_f0[left] > 0.0
            and est_f0[right] > 0.0
        ):
            continue
        fraction = (time_value - est_t[left]) / (est_t[right] - est_t[left])
        mapped_voiced[out_index] = True
        mapped_f0[out_index] = est_f0[left] + fraction * (est_f0[right] - est_f0[left])

    return mapped_f0, mapped_voiced


def cents_difference(reference_f0_hz: np.ndarray, estimate_f0_hz: np.ndarray) -> np.ndarray:
    return 1200.0 * np.log2(estimate_f0_hz / reference_f0_hz)


def compute_metrics(
    reference_f0_hz: Sequence[float],
    estimate_f0_hz: Sequence[float],
    estimate_voiced: Sequence[bool],
) -> tuple[dict[str, float | int], np.ndarray]:
    """Compute frame-micro pitch and voicing metrics with explicit denominators."""
    ref_f0 = np.asarray(reference_f0_hz, dtype=np.float64)
    est_f0 = np.asarray(estimate_f0_hz, dtype=np.float64)
    est_v = np.asarray(estimate_voiced, dtype=bool)
    if not (len(ref_f0) == len(est_f0) == len(est_v)):
        raise ValueError("metric arrays must have equal lengths")
    if np.any(~np.isfinite(ref_f0)) or np.any(ref_f0 < 0.0):
        raise ValueError("reference F0 must be finite and nonnegative")

    ref_v = ref_f0 > 0.0
    est_v = est_v & np.isfinite(est_f0) & (est_f0 > 0.0)
    both = ref_v & est_v
    ref_unvoiced = ~ref_v
    errors = np.full(len(ref_f0), np.nan, dtype=np.float64)
    errors[both] = cents_difference(ref_f0[both], est_f0[both])
    abs_errors = np.abs(errors[both])
    chroma_errors = np.abs(((errors[both] + 600.0) % 1200.0) - 600.0)
    pitch_correct = int(np.sum(abs_errors <= 50.0))
    chroma_correct = int(np.sum(chroma_errors <= 50.0))
    true_positive = int(np.sum(both))
    correct_unvoiced = int(np.sum(ref_unvoiced & ~est_v))

    counts: dict[str, int] = {
        "total_reference_frames": int(len(ref_f0)),
        "reference_voiced_frames": int(np.sum(ref_v)),
        "reference_unvoiced_frames": int(np.sum(ref_unvoiced)),
        "estimated_voiced_frames": int(np.sum(est_v)),
        "both_voiced_frames": true_positive,
        "voicing_true_positive_frames": true_positive,
        "correct_unvoiced_frames": correct_unvoiced,
        "pitch_correct_50c_frames": pitch_correct,
        "chroma_correct_50c_frames": chroma_correct,
    }
    return metrics_from_counts_and_errors(counts, errors[both]), errors


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def metrics_from_counts_and_errors(
    counts: Mapping[str, int], errors: Sequence[float]
) -> dict[str, float | int]:
    """Recompute metrics from additive counts and jointly voiced cent errors."""
    result: dict[str, float | int] = {field: int(counts[field]) for field in COUNT_FIELDS}
    ref_voiced = result["reference_voiced_frames"]
    est_voiced = result["estimated_voiced_frames"]
    true_positive = result["voicing_true_positive_frames"]
    precision = _safe_ratio(true_positive, est_voiced)
    recall = _safe_ratio(true_positive, ref_voiced)
    f1 = (
        float(2.0 * precision * recall / (precision + recall))
        if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0.0
        else float("nan")
    )
    result.update(
        {
            "raw_pitch_accuracy_50c": _safe_ratio(
                result["pitch_correct_50c_frames"], ref_voiced
            ),
            "raw_chroma_accuracy_50c": _safe_ratio(
                result["chroma_correct_50c_frames"], ref_voiced
            ),
            "voicing_precision": precision,
            "voicing_recall": recall,
            "voicing_f1": f1,
            "overall_accuracy": _safe_ratio(
                result["pitch_correct_50c_frames"] + result["correct_unvoiced_frames"],
                result["total_reference_frames"],
            ),
        }
    )
    error_values = np.asarray(errors, dtype=np.float64)
    error_values = error_values[np.isfinite(error_values)]
    if len(error_values):
        absolute = np.abs(error_values)
        result.update(
            {
                "signed_cents_mean": float(np.mean(error_values)),
                "signed_cents_median": float(np.median(error_values)),
                "absolute_cents_mean": float(np.mean(absolute)),
                "absolute_cents_median": float(np.median(absolute)),
                "absolute_cents_rmse": float(np.sqrt(np.mean(np.square(error_values)))),
                "absolute_cents_p90": float(np.percentile(absolute, 90.0)),
                "absolute_cents_p95": float(np.percentile(absolute, 95.0)),
            }
        )
    else:
        result.update({field: float("nan") for field in METRIC_FIELDS[6:]})
    return result


def _pyin_kwargs() -> dict[str, Any]:
    kwargs = dict(PYIN_CONFIG)
    kwargs["beta_parameters"] = tuple(kwargs["beta_parameters"])
    kwargs["fill_na"] = np.nan
    return kwargs


def estimate_pyin(vocal_audio: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run the frozen pYIN configuration; librosa is imported lazily."""
    import librosa

    f0, voiced, voiced_probability = librosa.pyin(
        np.asarray(vocal_audio, dtype=np.float32), **_pyin_kwargs()
    )
    times = np.arange(len(f0), dtype=np.float64) * (
        PYIN_CONFIG["hop_length"] / PYIN_CONFIG["sr"]
    )
    return (
        times,
        np.asarray(f0, dtype=np.float64),
        np.asarray(voiced, dtype=bool),
        np.asarray(voiced_probability, dtype=np.float64),
    )


def load_clip_specs(manifest_path: Path, data_root: Path) -> list[ClipSpec]:
    specs: list[ClipSpec] = []
    seen: set[str] = set()
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"stem", "singer", "wav_sha256", "pitch_sha256"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"selection manifest lacks required fields: {sorted(required)}")
        for row in reader:
            stem = row["stem"]
            if stem in seen:
                raise ValueError(f"duplicate stem in selection manifest: {stem}")
            seen.add(stem)
            specs.append(
                ClipSpec(
                    stem=stem,
                    singer=row["singer"],
                    wav_path=data_root / "MIR-1K" / "Wavfile" / f"{stem}.wav",
                    pitch_path=data_root / "MIR-1K" / "PitchLabel" / f"{stem}.pv",
                    expected_wav_sha256=row["wav_sha256"],
                    expected_pitch_sha256=row["pitch_sha256"],
                )
            )
    return specs


def process_clip(spec: ClipSpec) -> dict[str, Any]:
    import soundfile as sf

    wav_hash = file_sha256(spec.wav_path)
    pitch_hash = file_sha256(spec.pitch_path)
    if wav_hash != spec.expected_wav_sha256:
        raise ValueError(f"WAV hash mismatch for {spec.stem}: {wav_hash}")
    if pitch_hash != spec.expected_pitch_sha256:
        raise ValueError(f"pitch-label hash mismatch for {spec.stem}: {pitch_hash}")

    audio, sample_rate = sf.read(spec.wav_path, dtype="float32", always_2d=True)
    if sample_rate != PYIN_CONFIG["sr"]:
        raise ValueError(f"{spec.stem}: expected 16000 Hz, received {sample_rate}; no resampling allowed")
    if audio.shape[1] != 2:
        raise ValueError(f"{spec.stem}: expected two channels, received shape {audio.shape}")
    vocal = audio[:, VOCAL_CHANNEL_INDEX]
    pitch_midi = read_pitch_label(spec.pitch_path)
    ref_times = reference_times(len(pitch_midi))
    expected_label_count = math.floor((len(vocal) / sample_rate) / REFERENCE_FRAME_PERIOD_SEC) - 1
    if len(pitch_midi) != expected_label_count:
        raise ValueError(
            f"{spec.stem}: pitch label count {len(pitch_midi)} != verified timing rule "
            f"result {expected_label_count}"
        )
    ref_f0 = midi_to_hz(pitch_midi)
    est_times, est_f0, est_voiced, est_probability = estimate_pyin(vocal)
    mapped_f0, mapped_voiced = map_estimate_to_reference(
        ref_times, est_times, est_f0, est_voiced
    )
    metrics, errors = compute_metrics(ref_f0, mapped_f0, mapped_voiced)
    return {
        "status": "passed",
        "stem": spec.stem,
        "singer": spec.singer,
        "wav_path": str(spec.wav_path),
        "pitch_path": str(spec.pitch_path),
        "wav_sha256": wav_hash,
        "pitch_sha256": pitch_hash,
        "sample_rate_hz": int(sample_rate),
        "channels": int(audio.shape[1]),
        "audio_samples": int(len(vocal)),
        "audio_duration_sec": float(len(vocal) / sample_rate),
        "ref_times": ref_times,
        "ref_pitch_midi": pitch_midi,
        "ref_f0_hz": ref_f0,
        "ref_voiced": ref_f0 > 0.0,
        "est_times": est_times,
        "est_f0_hz": est_f0,
        "est_voiced": est_voiced,
        "est_voiced_probability": est_probability,
        "mapped_est_f0_hz": mapped_f0,
        "mapped_est_voiced": mapped_voiced,
        "cents_error": errors,
        "metrics": metrics,
    }


def aggregate_results(results: Sequence[dict[str, Any]]) -> dict[str, float | int]:
    counts = {field: 0 for field in COUNT_FIELDS}
    errors: list[np.ndarray] = []
    for result in results:
        for field in COUNT_FIELDS:
            counts[field] += int(result["metrics"][field])
        finite_errors = result["cents_error"][np.isfinite(result["cents_error"])]
        if len(finite_errors):
            errors.append(finite_errors)
    joined = np.concatenate(errors) if errors else np.asarray([], dtype=np.float64)
    return metrics_from_counts_and_errors(counts, joined)


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(json_safe(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json_safe(row.get(key)) for key in fieldnames})


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in ("librosa", "numpy", "scipy", "soundfile", "numba"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--selection-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--expected-clips", type=int, default=EXPECTED_CLIPS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not (1 <= args.workers <= MAX_WORKERS):
        raise SystemExit(f"--workers must be between 1 and {MAX_WORKERS}")
    versions = package_versions()
    if versions["librosa"] != "0.11.0":
        raise SystemExit(
            f"frozen benchmark requires librosa 0.11.0, received {versions['librosa']}"
        )
    specs = load_clip_specs(args.selection_manifest, args.data_root)
    if len(specs) != args.expected_clips:
        raise SystemExit(
            f"selection manifest has {len(specs)} clips, expected exactly {args.expected_clips}"
        )

    output_dir = args.output_dir.resolve()
    npz_dir = output_dir / "contours_npz"
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_dir.mkdir(parents=True, exist_ok=True)
    code_path = Path(__file__).resolve()
    frozen_config = {
        "schema_version": SCHEMA_VERSION,
        "estimator": "librosa.pyin",
        "runtime_required_librosa_version": "0.11.0",
        "vocal_channel_index_zero_based": VOCAL_CHANNEL_INDEX,
        "channel_semantics": "channel 0 accompaniment; channel 1 singing voice",
        "resampling": "none; reject any sample rate other than 16000 Hz",
        "estimator_time_grid_sec": "0.00 + 0.01*n (center=True)",
        "reference_time_grid_sec": "0.02 + 0.02*n",
        "alignment": "exact timestamp or linear interpolation only between adjacent voiced frames; never across unvoiced",
        "reference_unvoiced_value": "MIDI 0 means unvoiced, not a silent note",
        "raw_pitch_tolerance_cents": 50.0,
        "raw_chroma_tolerance_cents": 50.0,
        "pyin": PYIN_CONFIG,
        "predeclared_usability_criterion": PREDECLARED_USABILITY_CRITERION,
    }
    write_json(output_dir / "frozen_config.json", frozen_config)

    passed: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_to_spec = {executor.submit(process_clip, spec): spec for spec in specs}
        for future in as_completed(future_to_spec):
            spec = future_to_spec[future]
            try:
                passed.append(future.result())
            except Exception as exc:  # failures are retained and make the gate fail
                failures.append(
                    {
                        "stem": spec.stem,
                        "singer": spec.singer,
                        "wav_path": str(spec.wav_path),
                        "pitch_path": str(spec.pitch_path),
                        "exception_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
                    }
                )
    passed.sort(key=lambda row: row["stem"])
    failures.sort(key=lambda row: row["stem"])

    clip_rows: list[dict[str, Any]] = []
    aligned_rows: list[dict[str, Any]] = []
    raw_estimate_rows: list[dict[str, Any]] = []
    for result in passed:
        npz_path = npz_dir / f"{result['stem']}.npz"
        np.savez_compressed(
            npz_path,
            reference_time_sec=result["ref_times"],
            reference_pitch_midi=result["ref_pitch_midi"],
            reference_f0_hz=result["ref_f0_hz"],
            reference_voiced=result["ref_voiced"],
            estimate_time_sec=result["est_times"],
            estimate_f0_hz=result["est_f0_hz"],
            estimate_voiced=result["est_voiced"],
            estimate_voiced_probability=result["est_voiced_probability"],
            estimate_on_reference_f0_hz=result["mapped_est_f0_hz"],
            estimate_on_reference_voiced=result["mapped_est_voiced"],
            cents_error=result["cents_error"],
        )
        clip_rows.append(
            {
                "status": "passed",
                "stem": result["stem"],
                "singer": result["singer"],
                "wav_path": result["wav_path"],
                "pitch_path": result["pitch_path"],
                "wav_sha256": result["wav_sha256"],
                "pitch_sha256": result["pitch_sha256"],
                "contour_npz_path": str(npz_path),
                "contour_npz_sha256": file_sha256(npz_path),
                "sample_rate_hz": result["sample_rate_hz"],
                "channels": result["channels"],
                "audio_samples": result["audio_samples"],
                "audio_duration_sec": result["audio_duration_sec"],
                **result["metrics"],
            }
        )
        for index in range(len(result["ref_times"])):
            aligned_rows.append(
                {
                    "stem": result["stem"],
                    "singer": result["singer"],
                    "reference_frame_index": index,
                    "reference_time_sec": result["ref_times"][index],
                    "reference_pitch_midi": result["ref_pitch_midi"][index],
                    "reference_f0_hz": result["ref_f0_hz"][index],
                    "reference_voiced": int(result["ref_voiced"][index]),
                    "estimate_f0_hz": result["mapped_est_f0_hz"][index],
                    "estimate_voiced": int(result["mapped_est_voiced"][index]),
                    "cents_error": result["cents_error"][index],
                }
            )
        for index in range(len(result["est_times"])):
            raw_estimate_rows.append(
                {
                    "stem": result["stem"],
                    "singer": result["singer"],
                    "estimate_frame_index": index,
                    "estimate_time_sec": result["est_times"][index],
                    "estimate_f0_hz": result["est_f0_hz"][index],
                    "estimate_voiced": int(result["est_voiced"][index]),
                    "estimate_voiced_probability": result["est_voiced_probability"][index],
                }
            )
    for failure in failures:
        clip_rows.append({"status": "failed", **failure})
    clip_rows.sort(key=lambda row: row["stem"])

    clip_fields = [
        "status", "stem", "singer", "wav_path", "pitch_path", "wav_sha256",
        "pitch_sha256", "contour_npz_path", "contour_npz_sha256", "sample_rate_hz",
        "channels", "audio_samples", "audio_duration_sec", *COUNT_FIELDS, *METRIC_FIELDS,
        "exception_type", "error", "traceback",
    ]
    write_csv(output_dir / "per_clip_metrics.csv", clip_rows, clip_fields)
    write_csv(
        output_dir / "aligned_contours.csv",
        aligned_rows,
        [
            "stem", "singer", "reference_frame_index", "reference_time_sec",
            "reference_pitch_midi", "reference_f0_hz", "reference_voiced",
            "estimate_f0_hz", "estimate_voiced", "cents_error",
        ],
    )
    write_csv(
        output_dir / "estimated_contours_10ms.csv",
        raw_estimate_rows,
        [
            "stem", "singer", "estimate_frame_index", "estimate_time_sec",
            "estimate_f0_hz", "estimate_voiced", "estimate_voiced_probability",
        ],
    )
    write_csv(
        output_dir / "failures.csv",
        failures,
        ["stem", "singer", "wav_path", "pitch_path", "exception_type", "error", "traceback"],
    )

    singers = sorted({spec.singer for spec in specs})
    singer_rows: list[dict[str, Any]] = []
    for singer in singers:
        singer_results = [result for result in passed if result["singer"] == singer]
        singer_rows.append(
            {
                "singer": singer,
                "requested_clips": sum(1 for spec in specs if spec.singer == singer),
                "successful_clips": len(singer_results),
                "failed_clips": sum(1 for row in failures if row["singer"] == singer),
                **aggregate_results(singer_results),
            }
        )
    write_csv(
        output_dir / "per_singer_metrics.csv",
        singer_rows,
        [
            "singer", "requested_clips", "successful_clips", "failed_clips",
            *COUNT_FIELDS, *METRIC_FIELDS,
        ],
    )

    aggregate = aggregate_results(passed)
    all_hashes_verified = all(
        row["wav_sha256"] and row["pitch_sha256"] for row in passed
    ) and not failures
    criterion_checks = {
        "requested_clips_eq_100": len(specs) == PREDECLARED_USABILITY_CRITERION["required_requested_clips"],
        "successful_clips_eq_100": len(passed) == PREDECLARED_USABILITY_CRITERION["required_successful_clips"],
        "failure_count_eq_0": len(failures) == PREDECLARED_USABILITY_CRITERION["required_failure_count"],
        "all_input_hashes_verified": all_hashes_verified,
        "raw_pitch_accuracy_50c_ge_0_70": bool(
            aggregate["raw_pitch_accuracy_50c"]
            >= PREDECLARED_USABILITY_CRITERION["minimum_aggregate_raw_pitch_accuracy_50c"]
        ),
        "voicing_f1_ge_0_80": bool(
            aggregate["voicing_f1"]
            >= PREDECLARED_USABILITY_CRITERION["minimum_aggregate_voicing_f1"]
        ),
    }
    gate_passed = all(criterion_checks.values())
    overall = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete" if not failures and len(passed) == len(specs) else "incomplete_with_failures",
        "requested_clips": len(specs),
        "successful_clips": len(passed),
        "failed_clips": len(failures),
        "singer_count_requested": len({spec.singer for spec in specs}),
        "singer_count_successful": len({result["singer"] for result in passed}),
        "metrics": aggregate,
        "predeclared_usability_criterion": PREDECLARED_USABILITY_CRITERION,
        "criterion_checks": criterion_checks,
        "usable_for_coarse_F0_trajectory": gate_passed,
        "interpretation": (
            "Pass authorizes coarse F0 trajectory measurement on isolated MIR-1K vocal channels only. "
            "It does not validate vibrato rate/extent or mixture/stem robustness."
        ),
    }
    write_json(output_dir / "overall_metrics.json", overall)

    run_manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": overall["status"],
        "argv": sys.argv,
        "workers": args.workers,
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "packages": versions,
        "code_path": str(code_path),
        "code_sha256": file_sha256(code_path),
        "selection_manifest_path": str(args.selection_manifest.resolve()),
        "selection_manifest_sha256": file_sha256(args.selection_manifest),
        "data_root": str(args.data_root.resolve()),
        "output_dir": str(output_dir),
        "frozen_config_sha256": canonical_json_sha256(frozen_config),
        "selected_stem_set_sha256": hashlib.sha256(
            ("\n".join(sorted(spec.stem for spec in specs)) + "\n").encode("utf-8")
        ).hexdigest(),
        "requested_clips": len(specs),
        "successful_clips": len(passed),
        "failed_clips": len(failures),
        "output_file_sha256": {},
    }
    for output_path in sorted(output_dir.rglob("*")):
        if output_path.is_file() and output_path.name != "run_manifest.json":
            run_manifest["output_file_sha256"][str(output_path.relative_to(output_dir))] = file_sha256(output_path)
    write_json(output_dir / "run_manifest.json", run_manifest)
    print(json.dumps(json_safe(overall), indent=2, sort_keys=True))
    return 0 if overall["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
