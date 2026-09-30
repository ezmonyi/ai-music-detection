#!/usr/bin/env python3
"""Interpretable monaural phase/group-delay measurements.

``extract_phase_features`` returns scalar ``F_`` measurements plus explicit
quality fields.  Model inputs are enumerated by ``FEATURE_NAMES``; callers
should never silently replace unavailable (NaN) values with zero.

Metric units and ranges
-----------------------
``phase_residual_cvar_*`` is weighted circular variance in [0, 1] of
``wrap(phi[k,t+1] - phi[k,t] - 2*pi*f[k]*hop/sr)``.  This formula matches the
SciPy/NumPy forward-FFT convention.  Residuals are aggregated jointly over
valid time-frequency pairs: computing circular variance separately per bin
would make subtraction of that bin's constant expected rotation algebraically
irrelevant.  ``group_delay_iqr_ms_*`` is the median, over frames, of the
within-band IQR of ``-d(phi)/d(omega)`` in milliseconds.
``group_delay_cross_band_iqr_ms_*`` is the corresponding IQR of robust band
centres.  Differences have the units encoded in their names.

Low-energy bins and adjacent time/frequency pairs are masked before phase is
used.  Group delay near spectral zeros is therefore omitted instead of being
clipped into an apparently valid number.  Attack/sustain/decay summaries are
reported only when enough frames of that region are available.

These measurements are exploratory descriptions, not AI-authorship rules.
They remain sensitive to STFT settings, crop/frame alignment, time shifts,
codec history, reverberation, channel folding, and source separation.  The
primary experiment should use the original mono mix resampled to 16 kHz; the
function is nevertheless valid for other positive sample rates.
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np
from scipy.signal import stft


VERSION: Final[str] = "phase_features_v1_20260907"
CONFIG: Final[dict[str, float | int | str]] = {
    "version": VERSION,
    "reference_sample_rate_hz": 16_000,
    "window_duration_s": 0.064,
    "hop_fraction": 0.25,
    "window": "periodic_hann",
    "low_energy_relative_db": -40.0,
    "absolute_floor_dbfs": -100.0,
    "minimum_active_frames": 8,
    "minimum_region_frames": 6,
}

_REGIONS: Final[tuple[str, ...]] = ("all", "attack", "sustain", "decay")
FEATURE_NAMES: Final[tuple[str, ...]] = tuple(
    [f"F_phase_residual_cvar_{region}" for region in _REGIONS]
    + [f"F_group_delay_iqr_ms_{region}" for region in _REGIONS]
    + [f"F_group_delay_cross_band_iqr_ms_{region}" for region in _REGIONS]
    + [
        "F_phase_residual_cvar_decay_minus_sustain",
        "F_group_delay_iqr_ms_decay_minus_sustain",
        "F_group_delay_cross_band_iqr_ms_decay_minus_sustain",
    ]
)

QUALITY_NAMES: Final[tuple[str, ...]] = (
    "F_status",
    "F_quality_eligible",
    "F_duration_s",
    "F_rms_dbfs",
    "F_n_fft",
    "F_hop_length",
    "F_active_frame_count",
    "F_valid_phase_pair_count",
    "F_valid_group_delay_pair_count",
    "F_attack_frame_count",
    "F_sustain_frame_count",
    "F_decay_frame_count",
    "F_attack_eligible",
    "F_sustain_eligible",
    "F_decay_eligible",
)

_EPS: Final[float] = np.finfo(np.float64).tiny
_MIN_ACTIVE_FRAMES: Final[int] = int(CONFIG["minimum_active_frames"])
_MIN_REGION_FRAMES: Final[int] = int(CONFIG["minimum_region_frames"])


def _wrap(angle: np.ndarray) -> np.ndarray:
    """Wrap radians to [-pi, pi) without complex-angle branch ambiguity."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _analysis_size(sr: int) -> tuple[int, int]:
    """Choose the nearest power-of-two window to 64 ms and a quarter hop."""
    target = max(32.0, float(sr) * float(CONFIG["window_duration_s"]))
    n_fft = max(64, 1 << int(round(math.log2(target))))
    return n_fft, n_fft // 4


def _empty_result(y: np.ndarray, sr: int, status: str) -> dict[str, float | int | str]:
    n_fft, hop = _analysis_size(sr)
    rms = float(np.sqrt(np.mean(np.square(y, dtype=np.float64)))) if y.size else 0.0
    result: dict[str, float | int | str] = {name: float("nan") for name in FEATURE_NAMES}
    result.update(
        F_status=status,
        F_quality_eligible=0,
        F_duration_s=float(y.size / sr),
        F_rms_dbfs=float(20.0 * np.log10(max(rms, _EPS))),
        F_n_fft=n_fft,
        F_hop_length=hop,
        F_active_frame_count=0,
        F_valid_phase_pair_count=0,
        F_valid_group_delay_pair_count=0,
        F_attack_frame_count=0,
        F_sustain_frame_count=0,
        F_decay_frame_count=0,
        F_attack_eligible=0,
        F_sustain_eligible=0,
        F_decay_eligible=0,
    )
    return result


def _weighted_circular_variance(angles: np.ndarray, weights: np.ndarray) -> float:
    valid = np.isfinite(angles) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(valid):
        return float("nan")
    angles = angles[valid]
    weights = weights[valid]
    total = float(weights.sum())
    resultant = abs(np.sum(weights * np.exp(1j * angles))) / total
    # Roundoff can make the resultant infinitesimally larger than one.
    return float(np.clip(1.0 - resultant, 0.0, 1.0))


def _nan_iqr(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return float("nan")
    q25, q75 = np.percentile(values, [25.0, 75.0])
    return float(q75 - q25)


def _region_masks(frame_db: np.ndarray, active: np.ndarray) -> dict[str, np.ndarray]:
    """Assign active frames by robust local log-energy direction."""
    smooth = np.convolve(frame_db, np.ones(3, dtype=np.float64) / 3.0, mode="same")
    slope = np.diff(smooth, prepend=smooth[0])
    active_slopes = slope[active]
    if active_slopes.size:
        centre = float(np.median(active_slopes))
        mad = float(np.median(np.abs(active_slopes - centre)))
    else:
        centre = mad = 0.0
    threshold = max(0.5, 2.5 * mad)
    attack = active & (slope > centre + threshold)
    decay = active & (slope < centre - threshold)
    sustain = active & ~(attack | decay)
    return {"all": active, "attack": attack, "sustain": sustain, "decay": decay}


def _band_edges(nyquist: float) -> tuple[tuple[float, float], ...]:
    candidates = ((50.0, 500.0), (500.0, 2_000.0), (2_000.0, 4_000.0),
                  (4_000.0, 0.95 * nyquist))
    return tuple((low, high) for low, high in candidates if high > low + 50.0)


def extract_phase_features(y: np.ndarray, sr: int) -> dict[str, float | int | str]:
    """Extract scalar phase/group-delay features with explicit eligibility.

    Parameters
    ----------
    y:
        One-dimensional, finite, floating-point mono samples.  Amplitudes are
        interpreted as full-scale audio units for the absolute quality floor.
    sr:
        Positive integer sample rate in Hz.  The planned standardized rate is
        16,000 Hz.

    Returns
    -------
    dict
        Keys in ``FEATURE_NAMES`` are candidate model inputs.  Unavailable
        measurements are NaN.  Keys in ``QUALITY_NAMES`` describe coverage;
        ``F_status == 'ok'`` and ``F_quality_eligible == 1`` indicate that the
        all-region phase and group-delay summaries passed minimum coverage.
    """
    if not isinstance(sr, (int, np.integer)) or int(sr) <= 0:
        raise ValueError("sr must be a positive integer")
    sr = int(sr)
    values = np.asarray(y)
    if values.ndim != 1:
        raise ValueError(f"y must be one-dimensional, got shape {values.shape}")
    if not np.issubdtype(values.dtype, np.floating):
        raise TypeError("y must have a floating-point dtype")
    values = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("y must contain only finite samples")

    n_fft, hop = _analysis_size(sr)
    if values.size < n_fft + hop:
        return _empty_result(values, sr, "insufficient_duration")
    values = values - float(np.mean(values))
    rms = float(np.sqrt(np.mean(np.square(values))))
    if rms <= 10.0 ** (float(CONFIG["absolute_floor_dbfs"]) / 20.0):
        return _empty_result(values, sr, "low_energy")

    frequencies, _, spectrum = stft(
        values,
        fs=sr,
        window="hann",
        nperseg=n_fft,
        noverlap=n_fft - hop,
        nfft=n_fft,
        detrend=False,
        return_onesided=True,
        boundary=None,
        padded=False,
    )
    magnitude = np.abs(spectrum)
    power = np.square(magnitude)
    frame_power = np.mean(power, axis=0)
    frame_db = 10.0 * np.log10(np.maximum(frame_power, _EPS))
    peak_db = float(np.max(frame_db))
    spread = peak_db - float(np.percentile(frame_db, 20.0))
    active_floor = peak_db - (50.0 if spread < 12.0 else 45.0)
    active = frame_db >= active_floor
    region_masks = _region_masks(frame_db, active)

    # A bin must be strong relative both to its frame and to the whole excerpt.
    relative_ratio = 10.0 ** (float(CONFIG["low_energy_relative_db"]) / 20.0)
    frame_floor = np.max(magnitude, axis=0, keepdims=True) * relative_ratio
    global_floor = float(np.max(magnitude)) * 10.0 ** (-70.0 / 20.0)
    reliable = (magnitude >= np.maximum(frame_floor, global_floor)) & active[None, :]
    analysis_freq = (frequencies >= 50.0) & (frequencies <= 0.95 * sr / 2.0)
    reliable &= analysis_freq[:, None]

    phase = np.angle(spectrum)
    expected_rotation = 2.0 * np.pi * frequencies * hop / sr
    phase_residual = _wrap(np.diff(phase, axis=1) - expected_rotation[:, None])
    phase_valid = reliable[:, 1:] & reliable[:, :-1]
    phase_weight = np.sqrt(magnitude[:, 1:] * magnitude[:, :-1])
    phase_weight = np.where(phase_valid, phase_weight, 0.0)

    # Forward FFT uses exp(-j*omega*n), hence group delay is -dphi/domega.
    delta_omega = 2.0 * np.pi / n_fft
    group_delay_samples = -_wrap(np.diff(phase, axis=0)) / delta_omega
    group_valid = reliable[1:, :] & reliable[:-1, :]
    pair_frequencies = 0.5 * (frequencies[1:] + frequencies[:-1])

    frame_gd_iqr = np.full(spectrum.shape[1], np.nan, dtype=np.float64)
    frame_cross_band_iqr = np.full(spectrum.shape[1], np.nan, dtype=np.float64)
    bands = _band_edges(sr / 2.0)
    for frame in np.flatnonzero(active):
        band_iqrs: list[float] = []
        band_centres: list[float] = []
        for low, high in bands:
            use = group_valid[:, frame] & (pair_frequencies >= low) & (pair_frequencies < high)
            gd = group_delay_samples[use, frame]
            if gd.size >= 3:
                band_iqrs.append(_nan_iqr(gd))
                band_centres.append(float(np.median(gd)))
        if band_iqrs:
            frame_gd_iqr[frame] = float(np.median(band_iqrs)) * 1_000.0 / sr
        if len(band_centres) >= 2:
            frame_cross_band_iqr[frame] = _nan_iqr(np.asarray(band_centres)) * 1_000.0 / sr

    result = _empty_result(values, sr, "insufficient_phase_coverage")
    result.update(
        F_duration_s=float(values.size / sr),
        F_rms_dbfs=float(20.0 * np.log10(max(rms, _EPS))),
        F_active_frame_count=int(active.sum()),
        F_valid_phase_pair_count=int(phase_valid.sum()),
        F_valid_group_delay_pair_count=int(group_valid.sum()),
    )

    for region, frame_mask in region_masks.items():
        count = int(frame_mask.sum())
        if region != "all":
            result[f"F_{region}_frame_count"] = count
            result[f"F_{region}_eligible"] = int(count >= _MIN_REGION_FRAMES)
        region_ok = count >= (_MIN_ACTIVE_FRAMES if region == "all" else _MIN_REGION_FRAMES)
        time_pair_mask = frame_mask[1:] & frame_mask[:-1]
        residual_mask = phase_valid & time_pair_mask[None, :]
        if region_ok and int(residual_mask.sum()) >= 16:
            result[f"F_phase_residual_cvar_{region}"] = _weighted_circular_variance(
                phase_residual[residual_mask], phase_weight[residual_mask]
            )
        gd_values = frame_gd_iqr[frame_mask]
        gd_values = gd_values[np.isfinite(gd_values)]
        if region_ok and gd_values.size >= max(3, count // 4):
            result[f"F_group_delay_iqr_ms_{region}"] = float(np.median(gd_values))
        cross_values = frame_cross_band_iqr[frame_mask]
        cross_values = cross_values[np.isfinite(cross_values)]
        if region_ok and cross_values.size >= max(3, count // 4):
            result[f"F_group_delay_cross_band_iqr_ms_{region}"] = float(np.median(cross_values))

    for stem in (
        "phase_residual_cvar",
        "group_delay_iqr_ms",
        "group_delay_cross_band_iqr_ms",
    ):
        decay = float(result[f"F_{stem}_decay"])
        sustain = float(result[f"F_{stem}_sustain"])
        if np.isfinite(decay) and np.isfinite(sustain):
            result[f"F_{stem}_decay_minus_sustain"] = decay - sustain

    all_available = all(
        np.isfinite(float(result[name]))
        for name in (
            "F_phase_residual_cvar_all",
            "F_group_delay_iqr_ms_all",
            "F_group_delay_cross_band_iqr_ms_all",
        )
    )
    if int(active.sum()) >= _MIN_ACTIVE_FRAMES and all_available:
        result["F_status"] = "ok"
        result["F_quality_eligible"] = 1

    # Explicit numerical guard: valid metrics must respect their definitions.
    for name in FEATURE_NAMES:
        value = float(result[name])
        if np.isfinite(value) and "phase_residual_cvar" in name and "minus" not in name:
            if not 0.0 <= value <= 1.0:
                raise RuntimeError(f"{name} outside circular-variance range: {value}")
        if np.isinf(value):
            raise RuntimeError(f"Infinite phase metric: {name}")
    return result
