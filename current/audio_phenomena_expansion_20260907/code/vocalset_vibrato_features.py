#!/usr/bin/env python3
"""Draft pitch-contour vibrato-presence statistic for the VocalSet assay.

This is intentionally separate from the frozen MIR-1K rate/extent validator.
The primary value is a fraction of independently eligible stable-pitch windows,
so a measured absence is numeric zero.  Clips without enough eligible pitch
windows remain missing and are never coerced to zero.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np


VERSION = "draft-v1-before-vocalset-scoring"
PRIMARY_FEATURE = "V_periodic_modulation_window_fraction"

CONFIG: dict[str, Any] = {
    "input_frame_period_sec": 0.01,
    "window_duration_sec": 1.0,
    "window_frames": 100,
    "window_hop_sec": 0.5,
    "window_hop_frames": 50,
    "run_end_anchor": (
        "include one final window ending exactly at each eligible voiced-run end "
        "when the regular hop grid does not already include it"
    ),
    "minimum_clip_eligible_windows": 3,
    "frequency_min_hz": 4.0,
    "frequency_max_hz": 8.0,
    "frequency_grid_step_hz": 0.05,
    "minimum_detected_semiamplitude_cents": 10.0,
    "maximum_detected_semiamplitude_cents": 150.0,
    "minimum_detected_regularity_r2": 0.60,
    "maximum_adjacent_step_cents": 100.0,
    "eligibility": (
        "one second of contiguous valid F0 with no adjacent step over 100 cents; "
        "eligibility does not depend on modulation amplitude or regularity"
    ),
    "primary_feature": (
        "detected periodic windows divided by eligible stable-pitch windows; "
        "the denominator includes regular-hop and final run-anchored windows exactly once; "
        "numeric zero is allowed only when at least three eligible windows were measured"
    ),
}


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    changes = np.diff(np.concatenate(([False], mask, [False])).astype(np.int8))
    return list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))


def _window_starts(start: int, end: int) -> list[int]:
    size = CONFIG["window_frames"]
    hop = CONFIG["window_hop_frames"]
    if end - start < size:
        return []
    values = list(range(start, end - size + 1, hop))
    final = end - size
    if values[-1] != final:
        values.append(final)
    return values


def _analyze_window(times: np.ndarray, f0_hz: np.ndarray) -> dict[str, Any]:
    cents = 1200.0 * np.log2(f0_hz)
    maximum_step = float(np.max(np.abs(np.diff(cents)))) if len(cents) > 1 else 0.0
    if maximum_step > CONFIG["maximum_adjacent_step_cents"]:
        return {
            "eligible": False,
            "detected": False,
            "rejection_reason": "transition_or_pitch_jump",
            "maximum_adjacent_step_cents": maximum_step,
        }

    centered_time = times - float(np.mean(times))
    trend = np.column_stack((np.ones(len(cents)), centered_time))
    trend_coefficients, _, _, _ = np.linalg.lstsq(trend, cents, rcond=None)
    trend_residual = cents - trend @ trend_coefficients
    trend_sse = float(np.sum(np.square(trend_residual)))
    if trend_sse <= 1e-12:
        # This is a valid measured straight/stable window.  Frequency and
        # conditional extent are undefined, but periodic presence is absent.
        return {
            "eligible": True,
            "detected": False,
            "rejection_reason": "",
            "maximum_adjacent_step_cents": maximum_step,
            "best_rate_hz": math.nan,
            "best_semiamplitude_cents": 0.0,
            "best_regularity_r2": 0.0,
        }

    best: dict[str, float] | None = None
    frequencies = np.arange(
        CONFIG["frequency_min_hz"],
        CONFIG["frequency_max_hz"] + CONFIG["frequency_grid_step_hz"] * 0.5,
        CONFIG["frequency_grid_step_hz"],
    )
    for frequency in frequencies:
        phase = 2.0 * np.pi * frequency * centered_time
        design = np.column_stack(
            (np.ones(len(cents)), centered_time, np.sin(phase), np.cos(phase))
        )
        coefficients, _, _, _ = np.linalg.lstsq(design, cents, rcond=None)
        residual = cents - design @ coefficients
        sse = float(np.sum(np.square(residual)))
        regularity = float(np.clip(1.0 - sse / trend_sse, 0.0, 1.0))
        amplitude = float(math.hypot(coefficients[2], coefficients[3]))
        candidate = {
            "best_rate_hz": float(frequency),
            "best_semiamplitude_cents": amplitude,
            "best_regularity_r2": regularity,
        }
        if best is None or (candidate["best_regularity_r2"], amplitude) > (
            best["best_regularity_r2"],
            best["best_semiamplitude_cents"],
        ):
            best = candidate
    assert best is not None
    detected = bool(
        CONFIG["minimum_detected_semiamplitude_cents"]
        <= best["best_semiamplitude_cents"]
        <= CONFIG["maximum_detected_semiamplitude_cents"]
        and best["best_regularity_r2"] >= CONFIG["minimum_detected_regularity_r2"]
    )
    return {
        "eligible": True,
        "detected": detected,
        "rejection_reason": "",
        "maximum_adjacent_step_cents": maximum_step,
        **best,
    }


def extract_vibrato_presence(
    time_sec: Sequence[float], f0_hz: Sequence[float], voiced: Sequence[bool]
) -> dict[str, Any]:
    times = np.asarray(time_sec, dtype=np.float64)
    f0 = np.asarray(f0_hz, dtype=np.float64)
    voiced_input = np.asarray(voiced, dtype=bool)
    if not (times.ndim == f0.ndim == voiced_input.ndim == 1):
        raise ValueError("time, F0, and voiced inputs must be one-dimensional")
    if not (len(times) == len(f0) == len(voiced_input)):
        raise ValueError("time, F0, and voiced inputs must have equal lengths")
    if len(times) and (
        np.any(~np.isfinite(times))
        or np.any(np.diff(times) <= 0.0)
        or not np.allclose(
            np.diff(times), CONFIG["input_frame_period_sec"], atol=1e-8, rtol=0.0
        )
    ):
        raise ValueError("time input must be a strictly increasing uniform 10 ms grid")
    valid_pitch = voiced_input & np.isfinite(f0) & (f0 > 0.0)
    windows: list[dict[str, Any]] = []
    candidate_count = 0
    for start, end in _runs(valid_pitch):
        for window_start in _window_starts(int(start), int(end)):
            candidate_count += 1
            window_end = window_start + CONFIG["window_frames"]
            result = _analyze_window(times[window_start:window_end], f0[window_start:window_end])
            windows.append({"start_frame": window_start, **result})

    eligible = [window for window in windows if window["eligible"]]
    detected = [window for window in eligible if window["detected"]]
    minimum = CONFIG["minimum_clip_eligible_windows"]
    if len(eligible) < minimum:
        primary = math.nan
        status = "missing"
        missing_reason = "insufficient_stable_pitch_windows"
    else:
        primary = len(detected) / len(eligible)
        status = "ok"
        missing_reason = ""
    conditional_rate = (
        float(np.median([window["best_rate_hz"] for window in detected]))
        if detected
        else math.nan
    )
    conditional_extent = (
        float(np.median([window["best_semiamplitude_cents"] for window in detected]))
        if detected
        else math.nan
    )
    return {
        PRIMARY_FEATURE: float(primary),
        "V_status": status,
        "V_missing_reason": missing_reason,
        "V_pitch_frame_coverage": float(np.mean(valid_pitch)) if len(valid_pitch) else 0.0,
        "V_candidate_window_count": candidate_count,
        "V_eligible_stable_pitch_window_count": len(eligible),
        "V_transition_rejected_window_count": sum(not window["eligible"] for window in windows),
        "V_periodic_window_count": len(detected),
        "V_detected_rate_hz_median_conditional": conditional_rate,
        "V_detected_extent_cents_median_conditional": conditional_extent,
        "V_window_details": windows,
    }
