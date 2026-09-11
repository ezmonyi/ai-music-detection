#!/usr/bin/env python3
"""Conservative 4--8 Hz pitch-microstructure features and external validation.

The three values in FEATURE_NAMES are candidate V-family predictors. Coverage,
rejection, and observability values are diagnostics and must not be used as
numeric classifier predictors.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


for _thread_env_name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_env_name, "1")


VERSION = "v1"
SCHEMA_VERSION = 1
EXPECTED_CLIPS = 100
MAX_WORKERS = 4

# These, and only these, are proposed numeric V-family predictors.
FEATURE_NAMES = (
    "v_modulation_rate_hz",
    "v_modulation_extent_semiamplitude_cents",
    "v_modulation_regularity_r2",
)

DIAGNOSTIC_NAMES = (
    "total_frames",
    "voiced_frames",
    "voiced_run_count",
    "eligible_voiced_run_count",
    "candidate_window_count",
    "observable_window_count",
    "rejected_transition_window_count",
    "rejected_low_variation_window_count",
    "rejected_irregular_window_count",
    "rejected_extent_window_count",
    "observable_window_fraction",
)

# Frozen before inspecting feature-agreement results on the 100 real clips.
MICROSTRUCTURE_CONFIG: dict[str, Any] = {
    "input_frame_period_sec": 0.02,
    "require_uniform_time_grid": True,
    "window_duration_sec": 1.0,
    "window_frames": 50,
    "window_hop_sec": 0.5,
    "window_hop_frames": 25,
    "minimum_contiguous_voiced_duration_sec": 1.0,
    "minimum_contiguous_voiced_frames": 50,
    "detrending": "joint least-squares intercept plus linear cents trajectory in each window",
    "modulation_model": "add sine and cosine at each fixed candidate frequency; select maximum incremental R2",
    "frequency_min_hz": 4.0,
    "frequency_max_hz": 8.0,
    "frequency_grid_step_hz": 0.05,
    "extent_definition": "sqrt(sine_coefficient^2 + cosine_coefficient^2), semi-amplitude in cents",
    "regularity_definition": "1 - sinusoid_plus_linear_SSE / linear_only_SSE",
    "minimum_residual_rms_cents": 6.0,
    "minimum_modulation_semiamplitude_cents": 10.0,
    "maximum_modulation_semiamplitude_cents": 150.0,
    "minimum_modulation_regularity_r2": 0.60,
    "maximum_adjacent_step_cents": 100.0,
    "aggregation": "median across observable windows within each clip",
    "gap_policy": "split at every unvoiced frame; never interpolate or form a window across a gap",
    "transition_policy": "reject any window with an adjacent voiced-frame step over 100 cents",
    "note_overshoot": "not measured because reliable note-onset labels are unavailable",
}

# A pYIN RPA pass is necessary but explicitly insufficient.  This independent
# gate was frozen before computing real-contour V agreement.
PREDECLARED_MEASUREMENT_GATE: dict[str, Any] = {
    "required_clips": 100,
    "required_failures": 0,
    "require_all_synthetic_cases_pass": True,
    "minimum_reference_observable_clips": 25,
    "minimum_paired_observable_clips": 25,
    "minimum_clip_observability_precision": 0.70,
    "minimum_clip_observability_recall": 0.70,
    "maximum_median_absolute_rate_error_hz": 0.50,
    "maximum_p90_absolute_rate_error_hz": 1.50,
    "maximum_median_absolute_extent_error_cents": 12.0,
    "maximum_p90_absolute_extent_error_cents": 35.0,
    "maximum_median_absolute_regularity_error": 0.15,
    "maximum_p90_absolute_regularity_error": 0.35,
    "scope_if_passed": "candidate_V_features_on_isolated_MIR1K_vocal_channel",
    "next_required_gate": "source_separation_or_full_mix_robustness_before_classifier_use",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(json_safe(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: json_safe(row.get(field)) for field in fieldnames})


def hz_to_cents(f0_hz: np.ndarray) -> np.ndarray:
    values = np.asarray(f0_hz, dtype=np.float64)
    if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("voiced F0 values must be finite and positive")
    return 1200.0 * np.log2(values)


def cents_to_hz(cents: np.ndarray, base_hz: float = 1.0) -> np.ndarray:
    return base_hz * np.power(2.0, np.asarray(cents, dtype=np.float64) / 1200.0)


def contiguous_true_runs(mask: Sequence[bool]) -> list[tuple[int, int]]:
    values = np.asarray(mask, dtype=bool)
    changes = np.diff(np.concatenate(([False], values, [False])).astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def window_starts(start: int, end: int, window_frames: int, hop_frames: int) -> list[int]:
    if end - start < window_frames:
        return []
    starts = list(range(start, end - window_frames + 1, hop_frames))
    final_start = end - window_frames
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def analyze_window(time_sec: np.ndarray, f0_hz: np.ndarray) -> dict[str, Any]:
    cents = hz_to_cents(f0_hz)
    adjacent_steps = np.abs(np.diff(cents))
    maximum_step = float(np.max(adjacent_steps)) if len(adjacent_steps) else 0.0
    centered_time = np.asarray(time_sec, dtype=np.float64) - float(np.mean(time_sec))
    trend_design = np.column_stack([np.ones(len(cents)), centered_time])
    trend_coefficients, _, _, _ = np.linalg.lstsq(trend_design, cents, rcond=None)
    trend_residual = cents - trend_design @ trend_coefficients
    trend_sse = float(np.sum(np.square(trend_residual)))
    residual_rms = float(np.sqrt(np.mean(np.square(trend_residual))))
    slope_cents_per_sec = float(trend_coefficients[1])

    base = {
        "maximum_adjacent_step_cents": maximum_step,
        "linear_slope_cents_per_sec": slope_cents_per_sec,
        "detrended_residual_rms_cents": residual_rms,
    }
    if maximum_step > MICROSTRUCTURE_CONFIG["maximum_adjacent_step_cents"]:
        return {**base, "observable": False, "rejection_reason": "transition_or_pitch_jump"}
    if residual_rms < MICROSTRUCTURE_CONFIG["minimum_residual_rms_cents"] or trend_sse <= 1e-12:
        return {**base, "observable": False, "rejection_reason": "low_variation"}

    frequencies = np.arange(
        MICROSTRUCTURE_CONFIG["frequency_min_hz"],
        MICROSTRUCTURE_CONFIG["frequency_max_hz"]
        + MICROSTRUCTURE_CONFIG["frequency_grid_step_hz"] * 0.5,
        MICROSTRUCTURE_CONFIG["frequency_grid_step_hz"],
        dtype=np.float64,
    )
    best: dict[str, float] | None = None
    for frequency in frequencies:
        phase = 2.0 * np.pi * frequency * centered_time
        design = np.column_stack(
            [np.ones(len(cents)), centered_time, np.sin(phase), np.cos(phase)]
        )
        coefficients, _, _, _ = np.linalg.lstsq(design, cents, rcond=None)
        residual = cents - design @ coefficients
        sse = float(np.sum(np.square(residual)))
        regularity = float(np.clip(1.0 - (sse / trend_sse), 0.0, 1.0))
        extent = float(math.hypot(coefficients[2], coefficients[3]))
        candidate = {
            "modulation_rate_hz": float(frequency),
            "modulation_extent_semiamplitude_cents": extent,
            "modulation_regularity_r2": regularity,
        }
        if best is None or candidate["modulation_regularity_r2"] > best["modulation_regularity_r2"]:
            best = candidate
    assert best is not None
    if best["modulation_regularity_r2"] < MICROSTRUCTURE_CONFIG["minimum_modulation_regularity_r2"]:
        return {**base, **best, "observable": False, "rejection_reason": "irregular"}
    if not (
        MICROSTRUCTURE_CONFIG["minimum_modulation_semiamplitude_cents"]
        <= best["modulation_extent_semiamplitude_cents"]
        <= MICROSTRUCTURE_CONFIG["maximum_modulation_semiamplitude_cents"]
    ):
        return {**base, **best, "observable": False, "rejection_reason": "extent_out_of_range"}
    return {**base, **best, "observable": True, "rejection_reason": ""}


def extract_pitch_microstructure(
    time_sec: Sequence[float], f0_hz: Sequence[float], voiced: Sequence[bool]
) -> dict[str, Any]:
    times = np.asarray(time_sec, dtype=np.float64)
    f0 = np.asarray(f0_hz, dtype=np.float64)
    voiced_mask = np.asarray(voiced, dtype=bool) & np.isfinite(f0) & (f0 > 0.0)
    if not (len(times) == len(f0) == len(voiced_mask)):
        raise ValueError("time, F0, and voicing arrays must have equal lengths")
    if len(times) and (
        np.any(~np.isfinite(times))
        or np.any(np.diff(times) <= 0.0)
        or not np.allclose(
            np.diff(times), MICROSTRUCTURE_CONFIG["input_frame_period_sec"], atol=1e-8, rtol=0.0
        )
    ):
        raise ValueError("input must use a strictly increasing uniform 20 ms time grid")

    runs = contiguous_true_runs(voiced_mask)
    eligible_runs = [
        run
        for run in runs
        if run[1] - run[0] >= MICROSTRUCTURE_CONFIG["minimum_contiguous_voiced_frames"]
    ]
    windows: list[dict[str, Any]] = []
    for run_index, (run_start, run_end) in enumerate(eligible_runs):
        for start in window_starts(
            run_start,
            run_end,
            MICROSTRUCTURE_CONFIG["window_frames"],
            MICROSTRUCTURE_CONFIG["window_hop_frames"],
        ):
            end = start + MICROSTRUCTURE_CONFIG["window_frames"]
            window = analyze_window(times[start:end], f0[start:end])
            windows.append(
                {
                    "run_index": run_index,
                    "start_frame": start,
                    "end_frame_exclusive": end,
                    "start_time_sec": float(times[start]),
                    "end_time_sec": float(times[end - 1]),
                    **window,
                }
            )

    observable = [window for window in windows if window["observable"]]
    features = {
        FEATURE_NAMES[0]: float(np.median([w["modulation_rate_hz"] for w in observable]))
        if observable
        else float("nan"),
        FEATURE_NAMES[1]: float(
            np.median([w["modulation_extent_semiamplitude_cents"] for w in observable])
        )
        if observable
        else float("nan"),
        FEATURE_NAMES[2]: float(np.median([w["modulation_regularity_r2"] for w in observable]))
        if observable
        else float("nan"),
    }
    diagnostics = {
        "total_frames": int(len(times)),
        "voiced_frames": int(np.sum(voiced_mask)),
        "voiced_run_count": len(runs),
        "eligible_voiced_run_count": len(eligible_runs),
        "candidate_window_count": len(windows),
        "observable_window_count": len(observable),
        "rejected_transition_window_count": sum(
            w["rejection_reason"] == "transition_or_pitch_jump" for w in windows
        ),
        "rejected_low_variation_window_count": sum(
            w["rejection_reason"] == "low_variation" for w in windows
        ),
        "rejected_irregular_window_count": sum(
            w["rejection_reason"] == "irregular" for w in windows
        ),
        "rejected_extent_window_count": sum(
            w["rejection_reason"] == "extent_out_of_range" for w in windows
        ),
        "observable_window_fraction": float(len(observable) / len(windows))
        if windows
        else 0.0,
    }
    return {"features": features, "diagnostics": diagnostics, "windows": windows}


def _synthetic_contour(
    rate_hz: float,
    extent_cents: float,
    *,
    duration_sec: float = 3.0,
    slope_cents_per_sec: float = 0.0,
    base_hz: float = 220.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times = 0.02 + np.arange(int(round(duration_sec / 0.02)), dtype=np.float64) * 0.02
    centered = times - np.mean(times)
    cents = slope_cents_per_sec * centered + extent_cents * np.sin(2 * np.pi * rate_hz * times)
    f0 = cents_to_hz(cents, base_hz=base_hz)
    return times, f0, np.ones(len(times), dtype=bool)


def run_synthetic_validation() -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    for rate in (4.0, 6.0, 8.0):
        for extent in (20.0, 50.0):
            times, f0, voiced = _synthetic_contour(rate, extent)
            result = extract_pitch_microstructure(times, f0, voiced)
            recovered_rate = result["features"][FEATURE_NAMES[0]]
            recovered_extent = result["features"][FEATURE_NAMES[1]]
            recovered_regularity = result["features"][FEATURE_NAMES[2]]
            passed = bool(
                abs(recovered_rate - rate) <= 0.10
                and abs(recovered_extent - extent) <= 1.0
                and recovered_regularity >= 0.98
            )
            cases.append(
                {
                    "case": f"sinusoid_{rate:g}Hz_{extent:g}c",
                    "passed": passed,
                    "expected_rate_hz": rate,
                    "expected_extent_cents": extent,
                    "features": result["features"],
                    "diagnostics": result["diagnostics"],
                }
            )

    times, _, voiced = _synthetic_contour(6.0, 20.0)
    straight = extract_pitch_microstructure(times, np.full(len(times), 220.0), voiced)
    cases.append(
        {
            "case": "straight_tone_nonobservable",
            "passed": not np.isfinite(straight["features"][FEATURE_NAMES[0]])
            and straight["diagnostics"]["observable_window_count"] == 0,
            "features": straight["features"],
            "diagnostics": straight["diagnostics"],
        }
    )

    times, f0, voiced = _synthetic_contour(6.0, 30.0, slope_cents_per_sec=400.0)
    glide = extract_pitch_microstructure(times, f0, voiced)
    cases.append(
        {
            "case": "linear_glide_plus_6Hz_30c",
            "passed": abs(glide["features"][FEATURE_NAMES[0]] - 6.0) <= 0.10
            and abs(glide["features"][FEATURE_NAMES[1]] - 30.0) <= 1.0,
            "features": glide["features"],
            "diagnostics": glide["diagnostics"],
        }
    )

    times, f0, voiced = _synthetic_contour(6.0, 30.0)
    gap_voiced = voiced.copy()
    gap_voiced[60:80] = False
    gap_f0 = f0.copy()
    gap_f0[~gap_voiced] = np.nan
    gap = extract_pitch_microstructure(times, gap_f0, gap_voiced)
    gap_touched = any(
        not (window["end_frame_exclusive"] <= 60 or window["start_frame"] >= 80)
        for window in gap["windows"]
    )
    cases.append(
        {
            "case": "unvoiced_gap_never_bridged",
            "passed": not gap_touched and gap["diagnostics"]["voiced_run_count"] == 2,
            "features": gap["features"],
            "diagnostics": gap["diagnostics"],
        }
    )

    base = extract_pitch_microstructure(times, f0, voiced)
    octave_constant = extract_pitch_microstructure(times, f0 * 2.0, voiced)
    invariant = all(
        abs(base["features"][name] - octave_constant["features"][name]) <= 1e-8
        for name in FEATURE_NAMES
    )
    cases.append(
        {
            "case": "constant_octave_offset_invariant",
            "passed": invariant,
            "features": octave_constant["features"],
            "diagnostics": octave_constant["diagnostics"],
        }
    )

    jump_f0 = f0.copy()
    jump_f0[75:] *= 2.0
    jump = extract_pitch_microstructure(times, jump_f0, voiced)
    cases.append(
        {
            "case": "internal_octave_jump_rejected",
            "passed": jump["diagnostics"]["rejected_transition_window_count"] >= 1,
            "features": jump["features"],
            "diagnostics": jump["diagnostics"],
        }
    )
    return {
        "case_count": len(cases),
        "passed_case_count": sum(bool(case["passed"]) for case in cases),
        "all_passed": all(bool(case["passed"]) for case in cases),
        "cases": cases,
    }


def load_aligned_contours(path: Path) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, list[Any]]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "stem", "singer", "reference_frame_index", "reference_time_sec",
            "reference_f0_hz", "reference_voiced", "estimate_f0_hz", "estimate_voiced",
        }
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"aligned contour CSV lacks required columns: {sorted(required)}")
        for row in reader:
            key = (row["stem"], row["singer"])
            group = grouped.setdefault(
                key,
                {"indices": [], "times": [], "ref_f0": [], "ref_voiced": [], "est_f0": [], "est_voiced": []},
            )
            group["indices"].append(int(row["reference_frame_index"]))
            group["times"].append(float(row["reference_time_sec"]))
            group["ref_f0"].append(float(row["reference_f0_hz"]))
            group["ref_voiced"].append(bool(int(row["reference_voiced"])))
            group["est_f0"].append(float(row["estimate_f0_hz"]) if row["estimate_f0_hz"] else float("nan"))
            group["est_voiced"].append(bool(int(row["estimate_voiced"])))
    clips: list[dict[str, Any]] = []
    for (stem, singer), group in grouped.items():
        order = np.argsort(np.asarray(group["indices"], dtype=np.int64))
        sorted_indices = np.asarray(group["indices"], dtype=np.int64)[order]
        if not np.array_equal(sorted_indices, np.arange(len(sorted_indices))):
            raise ValueError(f"{stem}: reference frame indices are not contiguous from zero")
        clips.append(
            {
                "stem": stem,
                "singer": singer,
                **{
                    name: np.asarray(values)[order]
                    for name, values in group.items()
                    if name != "indices"
                },
            }
        )
    return sorted(clips, key=lambda clip: clip["stem"])


def process_real_clip(clip: Mapping[str, Any]) -> dict[str, Any]:
    reference = extract_pitch_microstructure(clip["times"], clip["ref_f0"], clip["ref_voiced"])
    estimate = extract_pitch_microstructure(clip["times"], clip["est_f0"], clip["est_voiced"])
    return {
        "status": "passed",
        "stem": clip["stem"],
        "singer": clip["singer"],
        "reference": reference,
        "estimate": estimate,
    }


def _error_summary(reference: Sequence[float], estimate: Sequence[float]) -> dict[str, Any]:
    ref = np.asarray(reference, dtype=np.float64)
    est = np.asarray(estimate, dtype=np.float64)
    valid = np.isfinite(ref) & np.isfinite(est)
    errors = est[valid] - ref[valid]
    absolute = np.abs(errors)
    if not len(errors):
        return {
            "paired_count": 0, "bias_mean": float("nan"), "absolute_error_mean": float("nan"),
            "absolute_error_median": float("nan"), "absolute_error_p90": float("nan"),
            "rmse": float("nan"), "pearson_r": float("nan"),
        }
    pearson = (
        float(np.corrcoef(ref[valid], est[valid])[0, 1])
        if len(errors) >= 2 and np.std(ref[valid]) > 0.0 and np.std(est[valid]) > 0.0
        else float("nan")
    )
    return {
        "paired_count": int(len(errors)),
        "bias_mean": float(np.mean(errors)),
        "absolute_error_mean": float(np.mean(absolute)),
        "absolute_error_median": float(np.median(absolute)),
        "absolute_error_p90": float(np.percentile(absolute, 90.0)),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "pearson_r": pearson,
    }


def summarize_real_results(results: Sequence[dict[str, Any]], failures: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ref_observable = np.asarray(
        [np.isfinite(result["reference"]["features"][FEATURE_NAMES[0]]) for result in results], dtype=bool
    )
    est_observable = np.asarray(
        [np.isfinite(result["estimate"]["features"][FEATURE_NAMES[0]]) for result in results], dtype=bool
    )
    both = ref_observable & est_observable
    true_positive = int(np.sum(both))
    precision = float(true_positive / np.sum(est_observable)) if np.sum(est_observable) else float("nan")
    recall = float(true_positive / np.sum(ref_observable)) if np.sum(ref_observable) else float("nan")
    f1 = (
        float(2 * precision * recall / (precision + recall))
        if np.isfinite(precision) and np.isfinite(recall) and precision + recall > 0
        else float("nan")
    )
    feature_agreement: dict[str, Any] = {}
    for name in FEATURE_NAMES:
        feature_agreement[name] = _error_summary(
            [result["reference"]["features"][name] for result in results],
            [result["estimate"]["features"][name] for result in results],
        )
    return {
        "requested_clips": len(results) + len(failures),
        "successful_clips": len(results),
        "failed_clips": len(failures),
        "reference_observable_clips": int(np.sum(ref_observable)),
        "estimate_observable_clips": int(np.sum(est_observable)),
        "paired_observable_clips": true_positive,
        "clip_observability_true_positive": true_positive,
        "clip_observability_false_positive": int(np.sum(~ref_observable & est_observable)),
        "clip_observability_false_negative": int(np.sum(ref_observable & ~est_observable)),
        "clip_observability_true_negative": int(np.sum(~ref_observable & ~est_observable)),
        "clip_observability_precision": precision,
        "clip_observability_recall": recall,
        "clip_observability_f1": f1,
        "feature_agreement": feature_agreement,
    }


def evaluate_gate(summary: Mapping[str, Any], synthetic: Mapping[str, Any]) -> dict[str, bool]:
    rate = summary["feature_agreement"][FEATURE_NAMES[0]]
    extent = summary["feature_agreement"][FEATURE_NAMES[1]]
    regularity = summary["feature_agreement"][FEATURE_NAMES[2]]
    return {
        "all_100_clips_successful": summary["successful_clips"] == PREDECLARED_MEASUREMENT_GATE["required_clips"]
        and summary["failed_clips"] == PREDECLARED_MEASUREMENT_GATE["required_failures"],
        "all_synthetic_cases_pass": bool(synthetic["all_passed"]),
        "reference_observable_clips_ge_25": summary["reference_observable_clips"]
        >= PREDECLARED_MEASUREMENT_GATE["minimum_reference_observable_clips"],
        "paired_observable_clips_ge_25": summary["paired_observable_clips"]
        >= PREDECLARED_MEASUREMENT_GATE["minimum_paired_observable_clips"],
        "clip_observability_precision_ge_0_70": summary["clip_observability_precision"]
        >= PREDECLARED_MEASUREMENT_GATE["minimum_clip_observability_precision"],
        "clip_observability_recall_ge_0_70": summary["clip_observability_recall"]
        >= PREDECLARED_MEASUREMENT_GATE["minimum_clip_observability_recall"],
        "median_abs_rate_error_le_0_50_hz": rate["absolute_error_median"]
        <= PREDECLARED_MEASUREMENT_GATE["maximum_median_absolute_rate_error_hz"],
        "p90_abs_rate_error_le_1_50_hz": rate["absolute_error_p90"]
        <= PREDECLARED_MEASUREMENT_GATE["maximum_p90_absolute_rate_error_hz"],
        "median_abs_extent_error_le_12_cents": extent["absolute_error_median"]
        <= PREDECLARED_MEASUREMENT_GATE["maximum_median_absolute_extent_error_cents"],
        "p90_abs_extent_error_le_35_cents": extent["absolute_error_p90"]
        <= PREDECLARED_MEASUREMENT_GATE["maximum_p90_absolute_extent_error_cents"],
        "median_abs_regularity_error_le_0_15": regularity["absolute_error_median"]
        <= PREDECLARED_MEASUREMENT_GATE["maximum_median_absolute_regularity_error"],
        "p90_abs_regularity_error_le_0_35": regularity["absolute_error_p90"]
        <= PREDECLARED_MEASUREMENT_GATE["maximum_p90_absolute_regularity_error"],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aligned-contours", required=True, type=Path)
    parser.add_argument("--pitch-tracking-run-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--expected-clips", type=int, default=EXPECTED_CLIPS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not (1 <= args.workers <= MAX_WORKERS):
        raise SystemExit(f"--workers must be between 1 and {MAX_WORKERS}")
    synthetic = run_synthetic_validation()
    if not synthetic["all_passed"]:
        raise SystemExit("synthetic validation failed; real contours were not evaluated")
    tracking_manifest = json.loads(args.pitch_tracking_run_manifest.read_text(encoding="utf-8"))
    if (
        tracking_manifest.get("status") != "complete"
        or tracking_manifest.get("requested_clips") != 100
        or tracking_manifest.get("successful_clips") != 100
        or tracking_manifest.get("failed_clips") != 0
    ):
        raise SystemExit("pitch-tracking run manifest is not a complete 100-clip, zero-failure run")
    expected_aligned_hash = tracking_manifest.get("output_file_sha256", {}).get("aligned_contours.csv")
    actual_aligned_hash = file_sha256(args.aligned_contours)
    if actual_aligned_hash != expected_aligned_hash:
        raise SystemExit(
            f"aligned contour hash does not match pitch-tracking run manifest: "
            f"{actual_aligned_hash} != {expected_aligned_hash}"
        )
    clips = load_aligned_contours(args.aligned_contours)
    if len(clips) != args.expected_clips:
        raise SystemExit(f"received {len(clips)} clips, expected exactly {args.expected_clips}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frozen = {
        "schema_version": SCHEMA_VERSION,
        "version": VERSION,
        "feature_names": FEATURE_NAMES,
        "diagnostic_names_not_predictors": DIAGNOSTIC_NAMES,
        "config": MICROSTRUCTURE_CONFIG,
        "predeclared_measurement_gate": PREDECLARED_MEASUREMENT_GATE,
    }
    write_json(output_dir / "frozen_config.json", frozen)
    write_json(output_dir / "synthetic_validation.json", synthetic)

    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_to_clip = {executor.submit(process_real_clip, clip): clip for clip in clips}
        for future in as_completed(future_to_clip):
            clip = future_to_clip[future]
            try:
                results.append(future.result())
            except Exception as exc:
                failures.append(
                    {
                        "stem": clip["stem"], "singer": clip["singer"],
                        "exception_type": type(exc).__name__, "error": str(exc),
                        "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
                    }
                )
    results.sort(key=lambda result: result["stem"])
    failures.sort(key=lambda failure: failure["stem"])

    per_clip_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    for result in results:
        row: dict[str, Any] = {"status": "passed", "stem": result["stem"], "singer": result["singer"]}
        for contour_name in ("reference", "estimate"):
            contour = result[contour_name]
            for name, value in contour["features"].items():
                row[f"{contour_name}_{name}"] = value
            for name, value in contour["diagnostics"].items():
                row[f"{contour_name}_diag_{name}"] = value
            for window_index, window in enumerate(contour["windows"]):
                window_rows.append(
                    {
                        "stem": result["stem"], "singer": result["singer"],
                        "contour": contour_name, "window_index": window_index, **window,
                    }
                )
        for name in FEATURE_NAMES:
            reference = result["reference"]["features"][name]
            estimate = result["estimate"]["features"][name]
            row[f"error_{name}"] = estimate - reference if np.isfinite(reference) and np.isfinite(estimate) else float("nan")
            row[f"absolute_error_{name}"] = abs(row[f"error_{name}"]) if np.isfinite(row[f"error_{name}"]) else float("nan")
        per_clip_rows.append(row)
    for failure in failures:
        per_clip_rows.append({"status": "failed", **failure})
    per_clip_rows.sort(key=lambda row: row["stem"])

    clip_fields = ["status", "stem", "singer"]
    for contour_name in ("reference", "estimate"):
        clip_fields.extend(f"{contour_name}_{name}" for name in FEATURE_NAMES)
        clip_fields.extend(f"{contour_name}_diag_{name}" for name in DIAGNOSTIC_NAMES)
    for name in FEATURE_NAMES:
        clip_fields.extend([f"error_{name}", f"absolute_error_{name}"])
    clip_fields.extend(["exception_type", "error", "traceback"])
    write_csv(output_dir / "per_clip_feature_agreement.csv", per_clip_rows, clip_fields)
    window_fields = [
        "stem", "singer", "contour", "window_index", "run_index", "start_frame",
        "end_frame_exclusive", "start_time_sec", "end_time_sec", "observable",
        "rejection_reason", "maximum_adjacent_step_cents", "linear_slope_cents_per_sec",
        "detrended_residual_rms_cents", "modulation_rate_hz",
        "modulation_extent_semiamplitude_cents", "modulation_regularity_r2",
    ]
    write_csv(output_dir / "per_window_diagnostics.csv", window_rows, window_fields)
    write_csv(
        output_dir / "failures.csv", failures,
        ["stem", "singer", "exception_type", "error", "traceback"],
    )

    summary = summarize_real_results(results, failures)
    gate_checks = evaluate_gate(summary, synthetic)
    overall = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete" if len(results) == args.expected_clips and not failures else "incomplete_with_failures",
        "synthetic_validation": synthetic,
        "real_contour_summary": summary,
        "predeclared_measurement_gate": PREDECLARED_MEASUREMENT_GATE,
        "gate_checks": gate_checks,
        "measurement_gate_passed": all(gate_checks.values()),
        "interpretation": (
            "A pass validates these conservative descriptors only on isolated MIR-1K vocal contours. "
            "It does not validate note overshoot, causal interpretation, or separated-stem/full-mix robustness."
        ),
    }
    write_json(output_dir / "overall_metrics.json", overall)

    code_path = Path(__file__).resolve()
    run_manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": overall["status"],
        "argv": sys.argv,
        "workers": args.workers,
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "numpy": np.__version__,
        "code_path": str(code_path),
        "code_sha256": file_sha256(code_path),
        "aligned_contours_path": str(args.aligned_contours.resolve()),
        "aligned_contours_sha256": file_sha256(args.aligned_contours),
        "pitch_tracking_run_manifest_path": str(args.pitch_tracking_run_manifest.resolve()),
        "pitch_tracking_run_manifest_sha256": file_sha256(args.pitch_tracking_run_manifest),
        "frozen_config_sha256": canonical_json_sha256(frozen),
        "output_file_sha256": {},
    }
    for output_path in sorted(output_dir.glob("*")):
        if output_path.is_file() and output_path.name != "run_manifest.json":
            run_manifest["output_file_sha256"][output_path.name] = file_sha256(output_path)
    write_json(output_dir / "run_manifest.json", run_manifest)
    print(json.dumps(json_safe(overall), indent=2, sort_keys=True))
    return 0 if overall["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
