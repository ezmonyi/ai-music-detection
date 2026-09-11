#!/usr/bin/env python3
"""Isolated CPU temporal envelope-modulation candidate; no classifier integration.

All output is JSON-compatible (missing scalars are None). See the accompanying
implementation note for signal assumptions and the pre-measurement thresholds.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, firwin, hilbert, resample_poly, sosfiltfilt

VERSION = "modulation_candidate_v1_20260907"
SAMPLE_RATE = 16_000
ENVELOPE_RATE = 500
BANDS_HZ = ((250, 1000), (1000, 3000), (3000, 7000))
BAND_NAMES = tuple(f"{lo}_{hi}hz" for lo, hi in BANDS_HZ)
FEATURE_NAMES = tuple(f"Q_{name}_{metric}_median" for name in BAND_NAMES
                      for metric in ("fast_fraction", "entropy"))
CONFIG = {
    "sample_rate_hz": SAMPLE_RATE, "envelope_rate_hz": ENVELOPE_RATE,
    "butterworth_prototype_order": 4, "filter_padtype": "odd",
    "filter_padlen_samples": 27, "edge_discard_seconds_each": 0.25,
    "window_seconds": 2.0, "hop_seconds": 2.0, "window": "periodic_hann",
    "antialias_taps": 1281, "antialias_cutoff_hz": 250.0,
    "antialias_kaiser_beta": 8.6,
    "relative_band_energy_floor": 1e-8,
    "normalized_total_modulation_power_floor": 1e-8,
    "normalized_analysis_modulation_power_floor": 1e-8,
    "maximum_input_absolute_value": 1e150,
    "minimum_summary_windows": 1,
    "fast_interval_hz": [16.0, 64.0], "analysis_interval_hz": [2.0, 128.0],
}
_EDGE = 125
_WINDOW = 1000
_FREQUENCIES = np.fft.rfftfreq(_WINDOW, 1.0 / ENVELOPE_RATE)
_ANALYSIS = (_FREQUENCIES >= 2.0) & (_FREQUENCIES < 128.0)
_FAST = (_FREQUENCIES >= 16.0) & (_FREQUENCIES < 64.0)
_HANN = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(_WINDOW) / _WINDOW)
_FIR = firwin(1281, 250.0, fs=SAMPLE_RATE, window=("kaiser", 8.6))


def _base_result(sample_count: int, reason: str) -> dict:
    return {"version": VERSION, "status": reason, "sample_count": sample_count,
            "sample_rate_hz": SAMPLE_RATE, "config": dict(CONFIG),
            "features": dict.fromkeys(FEATURE_NAMES), "bands": [],
            "frequency_hz": _FREQUENCIES.tolist(), "window_count": 0,
            "valid_band_window_count": 0}


def _spectrum(envelope: np.ndarray) -> tuple[np.ndarray, float]:
    """One-sided per-bin normalized power, summing to Hann-weighted variance."""
    mean = float(np.mean(envelope))
    if mean <= 0.0:
        return np.zeros(_FREQUENCIES.size), mean
    fluctuation = envelope / mean - 1.0
    fluctuation -= np.mean(fluctuation)
    transformed = np.fft.rfft(fluctuation * _HANN)
    power = np.abs(transformed) ** 2 / (_WINDOW * np.sum(_HANN ** 2))
    power[1:-1] *= 2.0
    return power, mean


def extract_modulation_candidate(audio, sample_rate: int, *, band_mask=None) -> dict:
    """Measure exactly 16 kHz finite, real, mono floating-point audio.

    ``band_mask`` is optionally a length-three boolean sequence; False masks
    the corresponding carrier band. Input is never resampled or downmixed.
    Invalid inputs return a stable status with six missing features. Filtering
    uses explicit odd reflection, but analysis windows are never padded.
    """
    if isinstance(sample_rate, (bool, np.bool_)) or sample_rate != SAMPLE_RATE:
        return _base_result(0, "wrong_sample_rate")
    try:
        raw = np.asarray(audio)
    except (TypeError, ValueError):
        return _base_result(0, "invalid_input")
    n = int(raw.size)
    if raw.ndim != 1:
        return _base_result(n, "not_mono")
    if raw.dtype.kind != "f":
        return _base_result(n, "not_floating_point")
    if n == 0:
        return _base_result(0, "empty_input")
    if not np.all(np.isfinite(raw)):
        return _base_result(n, "nonfinite_input")
    peak = np.max(np.abs(raw))
    if peak > CONFIG["maximum_input_absolute_value"]:
        return _base_result(n, "input_magnitude_too_large")
    if band_mask is None:
        mask = np.ones(3, dtype=bool)
    else:
        mask = np.asarray(band_mask)
        if mask.shape != (3,) or mask.dtype.kind != "b":
            return _base_result(n, "invalid_band_mask")
    # Floor excludes a polyphase endpoint whose timestamp reaches past input.
    available = n // 32 - 2 * _EDGE
    count = max(0, available // _WINDOW)
    if count == 0:
        return _base_result(n, "too_short")
    result = _base_result(n, "no_valid_band_windows")
    result["window_count"] = count
    result["discarded_tail_envelope_samples"] = available - count * _WINDOW
    result["covered_seconds"] = count * 2.0
    result["temporal_coverage_fraction"] = count * 2.0 / (n / SAMPLE_RATE)
    # Relative input scaling protects filtering and normalized powers from
    # overflow and underflow. Absolute energy remains in input-unit squared.
    x = np.zeros(n) if peak == 0 else np.asarray(raw / peak, dtype=np.float64)
    peak_squared = float(peak) ** 2
    all_valid = 0
    for index, ((low, high), name) in enumerate(zip(BANDS_HZ, BAND_NAMES)):
        windows = []
        if mask[index] and peak > 0:
            sos = butter(4, [low, high], btype="bandpass", fs=SAMPLE_RATE, output="sos")
            filtered = sosfiltfilt(sos, x, padtype="odd", padlen=27)
            envelope = resample_poly(np.abs(hilbert(filtered)), 1, 32,
                                     window=_FIR, padtype="constant")
        else:
            filtered = np.zeros(n)
            envelope = np.zeros(n // 32 + 1)
        for j in range(count):
            start = _EDGE + j * _WINDOW
            a, b = start * 32, (start + _WINDOW) * 32
            input_energy = float(np.mean(x[a:b] ** 2))
            band_energy = float(np.mean(filtered[a:b] ** 2))
            relative = band_energy / input_energy if input_energy > 0 else 0.0
            power, mean = _spectrum(envelope[start:start + _WINDOW])
            total = float(np.sum(power))
            analysis = float(np.sum(power[_ANALYSIS]))
            reason = ("band_masked" if not mask[index] else
                      "silence" if peak == 0 or input_energy == 0 else
                      "insufficient_band_energy" if relative < 1e-8 else
                      "no_modulation" if total <= 1e-8 else
                      "insufficient_analysis_modulation" if analysis <= 1e-8 else "ok")
            fast, entropy = None, None
            if reason == "ok":
                fast = float(np.clip(np.sum(power[_FAST]) / analysis, 0.0, 1.0))
                probabilities = power[_ANALYSIS] / analysis
                positive = probabilities[probabilities > 0]
                entropy = float(np.clip(-np.sum(positive * np.log(positive)) /
                                        np.log(np.count_nonzero(_ANALYSIS)), 0.0, 1.0))
            windows.append({"index": j, "start_seconds": a / SAMPLE_RATE,
                            "end_seconds": b / SAMPLE_RATE, "status": reason,
                            "sample_count": _WINDOW, "coverage": 1.0,
                            "input_mean_square": input_energy * peak_squared,
                            "band_mean_square": band_energy * peak_squared if mask[index] else None,
                            "relative_band_energy": relative if mask[index] else None,
                            "envelope_mean": mean * float(peak) if mask[index] else None,
                            "modulation_power": None if not mask[index] else power.tolist(),
                            "total_modulation_power": None if not mask[index] else total,
                            "analysis_modulation_power": None if not mask[index] else analysis,
                            "fast_fraction": fast, "entropy": entropy})
        valid = [w for w in windows if w["status"] == "ok"]
        all_valid += len(valid)
        reasons = sorted(set(w["status"] for w in windows if w["status"] != "ok"))
        result["bands"].append({"name": name, "carrier_hz": [low, high],
                                "status": "ok" if len(valid) == count else
                                          "partial" if valid else reasons[0],
                                "missing_reasons": reasons, "windows": windows,
                                "valid_window_count": len(valid),
                                "valid_window_fraction": len(valid) / count})
        for metric in ("fast_fraction", "entropy"):
            if valid:
                result["features"][f"Q_{name}_{metric}_median"] = float(
                    np.median([w[metric] for w in valid]))
    result["valid_band_window_count"] = all_valid
    result["status"] = "ok" if all_valid == count * 3 else "partial" if all_valid else "no_valid_band_windows"
    return result
