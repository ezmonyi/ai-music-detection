#!/usr/bin/env python3
"""Synthetic-only, no-reference stereo relationship candidate.

This module measures inter-channel relationships in fixed time/frequency bands.
It does not estimate reconstruction fidelity, does not use a clean reference,
and is not an admitted AI/Human detector feature.  ``native_channels`` is a
required provenance argument: a mono signal duplicated upstream must not be
presented as native stereo.

The public result contains per-frame/per-band arrays as well as bounded median
and IQR descriptors.  Missing measurements remain NaN and carry explicit
quality reasons.  In particular, IPD increments are computed only between
adjacent frames whose band cross-spectrum phase is reliable; gaps are never
bridged.
"""

from __future__ import annotations

from typing import Final

import numpy as np
from scipy.signal import stft


VERSION: Final[str] = "stereo_candidate_v1_20260907"
SAMPLE_RATE: Final[int] = 44_100
N_FFT: Final[int] = 4_096
HOP_LENGTH: Final[int] = 1_024
BANDS_HZ: Final[tuple[tuple[int, int], ...]] = (
    (80, 500),
    (500, 2_000),
    (2_000, 6_000),
    (6_000, 12_000),
    (12_000, 20_000),
)

CONFIG: Final[dict[str, object]] = {
    "version": VERSION,
    "sample_rate_hz": SAMPLE_RATE,
    "native_channels_required": 2,
    "window": "periodic_hann",
    "n_fft": N_FFT,
    "hop_length": HOP_LENGTH,
    "center": False,
    "bands_hz": BANDS_HZ,
    "relative_band_energy_floor_db": -60.0,
    "absolute_band_energy_floor_dbfs_like": -100.0,
    "minimum_channel_energy_fraction": 1.0e-8,
    "minimum_ipd_magnitude_coherence": 0.10,
    "minimum_descriptor_frames": 3,
    "iid_clip_db": 60.0,
}

_METRICS: Final[tuple[str, ...]] = (
    "iid_db",
    "signed_real_coherence",
    "magnitude_coherence",
    "side_energy_fraction",
    "ipd_increment_rad",
)
_DESCRIPTOR_METRICS: Final[tuple[str, ...]] = (
    "abs_iid_db",
    "signed_real_coherence",
    "magnitude_coherence",
    "side_energy_fraction",
    "abs_ipd_increment_rad",
)
_SUMMARIES: Final[tuple[str, ...]] = ("median", "iqr")


def _band_name(low: int, high: int) -> str:
    return f"{low}_{high}hz"


FEATURE_NAMES: Final[tuple[str, ...]] = tuple(
    f"SC_{_band_name(low, high)}_{metric}_{summary}"
    for low, high in BANDS_HZ
    for metric in _DESCRIPTOR_METRICS
    for summary in _SUMMARIES
)


def _wrap_radians(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _summary(values: np.ndarray) -> tuple[float, float, int]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    count = int(finite.size)
    if count < int(CONFIG["minimum_descriptor_frames"]):
        return float("nan"), float("nan"), count
    q25, median, q75 = np.percentile(finite, [25.0, 50.0, 75.0])
    return float(median), float(q75 - q25), count


def _descriptor_summary(metric: str, values: np.ndarray) -> tuple[float, float, int]:
    """Summarize one per-frame metric under the scalar invariance contract."""
    if metric in {"iid_db", "ipd_increment_rad"}:
        values = np.abs(values)
    return _summary(values)


def _empty_result(sample_count: int, status: str) -> dict[str, object]:
    n_frames = max(0, 1 + (sample_count - N_FFT) // HOP_LENGTH)
    shape = (len(BANDS_HZ), n_frames)
    per_frame = {metric: np.full(shape, np.nan, dtype=np.float64) for metric in _METRICS}
    masks = {
        "energy_valid": np.zeros(shape, dtype=bool),
        "two_sided_valid": np.zeros(shape, dtype=bool),
        "ipd_phase_valid": np.zeros(shape, dtype=bool),
        "ipd_increment_valid": np.zeros(shape, dtype=bool),
    }
    return {
        "version": VERSION,
        "status": status,
        "quality_eligible": 0,
        "sample_rate_hz": SAMPLE_RATE,
        "native_channels": 2,
        "sample_count": sample_count,
        "frame_count": n_frames,
        "bands": [
            {"low_hz": low, "high_hz": high, "nyquist_eligible": True,
             "status": status, "energy_valid_frames": 0, "two_sided_valid_frames": 0,
             "ipd_phase_valid_frames": 0, "ipd_increment_valid_frames": 0}
            for low, high in BANDS_HZ
        ],
        "per_frame": per_frame,
        "valid_masks": masks,
        "features": {name: float("nan") for name in FEATURE_NAMES},
    }


def extract_stereo_candidate(
    y: np.ndarray,
    sr: int,
    *,
    native_channels: int,
) -> dict[str, object]:
    """Measure fixed-band stereo relationships without a clean reference.

    Parameters
    ----------
    y:
        Finite floating-point samples with shape ``(samples, 2)``.
    sr:
        Must be exactly 44,100 Hz.  Resampling policy belongs upstream.
    native_channels:
        Required provenance value and must equal two.  It prevents silent
        admission of mono audio that was duplicated before this call.

    Returns
    -------
    dict
        ``per_frame`` arrays have shape ``(5 bands, frames)``.  IID is dB;
        coherences and side-energy fraction are unitless; IPD increments are
        wrapped radians.  ``features`` contains bounded median/IQR summaries.
    """
    if not isinstance(sr, (int, np.integer)) or int(sr) != SAMPLE_RATE:
        raise ValueError(f"sr must equal {SAMPLE_RATE}")
    if not isinstance(native_channels, (int, np.integer)) or int(native_channels) != 2:
        raise ValueError("native_channels provenance must equal 2")
    values = np.asarray(y)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError(f"y must have shape (samples, 2), got {values.shape}")
    if not np.issubdtype(values.dtype, np.floating):
        raise TypeError("y must have a floating-point dtype with full-scale units")
    values = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("y must contain only finite samples")
    # Ratios permit ordinary full-scale excursions above one, but finite values
    # so large that the subsequent squared spectral energies can overflow are
    # invalid input, not silence or missing evidence.
    safe_peak = np.sqrt(np.finfo(np.float64).max) / (4.0 * np.sqrt(N_FFT))
    if values.size and float(np.max(np.abs(values))) > safe_peak:
        raise ValueError("y magnitude risks floating-point overflow")
    if values.shape[0] < N_FFT:
        return _empty_result(values.shape[0], "insufficient_duration")

    frequencies, _, left = stft(
        values[:, 0], fs=SAMPLE_RATE, window="hann", nperseg=N_FFT,
        noverlap=N_FFT - HOP_LENGTH, nfft=N_FFT, detrend=False,
        return_onesided=True, boundary=None, padded=False, scaling="spectrum",
    )
    _, _, right = stft(
        values[:, 1], fs=SAMPLE_RATE, window="hann", nperseg=N_FFT,
        noverlap=N_FFT - HOP_LENGTH, nfft=N_FFT, detrend=False,
        return_onesided=True, boundary=None, padded=False, scaling="spectrum",
    )
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("STFT produced nonfinite values from overflowing input")
    n_frames = left.shape[1]
    shape = (len(BANDS_HZ), n_frames)
    per_frame = {metric: np.full(shape, np.nan, dtype=np.float64) for metric in _METRICS}
    masks = {
        "energy_valid": np.zeros(shape, dtype=bool),
        "two_sided_valid": np.zeros(shape, dtype=bool),
        "ipd_phase_valid": np.zeros(shape, dtype=bool),
        "ipd_increment_valid": np.zeros(shape, dtype=bool),
    }
    features = {name: float("nan") for name in FEATURE_NAMES}
    band_quality: list[dict[str, object]] = []
    nyquist = SAMPLE_RATE / 2.0
    absolute_floor = 10.0 ** (float(CONFIG["absolute_band_energy_floor_dbfs_like"]) / 10.0)
    relative_ratio = 10.0 ** (float(CONFIG["relative_band_energy_floor_db"]) / 10.0)
    min_side_fraction = float(CONFIG["minimum_channel_energy_fraction"])
    ipd_coherence_floor = float(CONFIG["minimum_ipd_magnitude_coherence"])

    for band_index, (low, high) in enumerate(BANDS_HZ):
        name = _band_name(low, high)
        eligible = high <= nyquist
        bins = (frequencies >= low) & (frequencies < high)
        quality: dict[str, object] = {
            "low_hz": low,
            "high_hz": high,
            "nyquist_eligible": bool(eligible),
            "bin_count": int(np.count_nonzero(bins)),
        }
        if not eligible or not np.any(bins):
            quality.update(status="ineligible_nyquist", energy_valid_frames=0,
                           two_sided_valid_frames=0, ipd_phase_valid_frames=0,
                           ipd_increment_valid_frames=0)
            band_quality.append(quality)
            continue

        band_left = left[bins]
        band_right = right[bins]
        with np.errstate(over="raise", invalid="raise"):
            try:
                energy_left = np.sum(np.abs(band_left) ** 2, axis=0)
                energy_right = np.sum(np.abs(band_right) ** 2, axis=0)
                total_energy = energy_left + energy_right
                cross = np.sum(band_left * np.conj(band_right), axis=0)
            except FloatingPointError as exc:
                raise ValueError("band energy/cross-spectrum overflow") from exc
        if not (np.all(np.isfinite(energy_left)) and np.all(np.isfinite(energy_right))
                and np.all(np.isfinite(total_energy)) and np.all(np.isfinite(cross))):
            raise ValueError("band energy/cross-spectrum is nonfinite")
        peak = float(np.max(total_energy)) if total_energy.size else 0.0
        threshold = max(absolute_floor, peak * relative_ratio)
        energy_valid = total_energy >= threshold
        two_sided = (
            energy_valid
            & (energy_left >= np.maximum(absolute_floor * 0.5, total_energy * min_side_fraction))
            & (energy_right >= np.maximum(absolute_floor * 0.5, total_energy * min_side_fraction))
        )
        # Multiplying the two energies before sqrt can overflow even when the
        # final geometric mean is representable.
        denominator = np.sqrt(energy_left) * np.sqrt(energy_right)
        if not np.all(np.isfinite(denominator)):
            raise ValueError("coherence denominator is nonfinite")
        with np.errstate(divide="ignore", invalid="ignore"):
            # Log subtraction avoids forming a potentially overflowing ratio;
            # one-sided/zero-energy frames are masked below in either case.
            iid = 10.0 * (np.log10(energy_left) - np.log10(energy_right))
            signed = np.real(cross) / denominator
            magnitude = np.abs(cross) / denominator
        iid = np.clip(iid, -float(CONFIG["iid_clip_db"]), float(CONFIG["iid_clip_db"]))
        signed = np.clip(signed, -1.0, 1.0)
        magnitude = np.clip(magnitude, 0.0, 1.0)
        iid[~two_sided] = np.nan
        signed[~two_sided] = np.nan
        magnitude[~two_sided] = np.nan

        mid = 0.5 * (band_left + band_right)
        side = 0.5 * (band_left - band_right)
        mid_energy = np.sum(np.abs(mid) ** 2, axis=0)
        side_energy = np.sum(np.abs(side) ** 2, axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            side_fraction = side_energy / (mid_energy + side_energy)
        side_fraction = np.clip(side_fraction, 0.0, 1.0)
        side_fraction[~energy_valid] = np.nan

        ipd_phase_valid = two_sided & np.isfinite(magnitude) & (magnitude >= ipd_coherence_floor)
        ipd_phase = np.full(n_frames, np.nan, dtype=np.float64)
        ipd_phase[ipd_phase_valid] = np.angle(cross[ipd_phase_valid])
        increment_valid = np.zeros(n_frames, dtype=bool)
        increment_valid[1:] = ipd_phase_valid[1:] & ipd_phase_valid[:-1]
        ipd_increment = np.full(n_frames, np.nan, dtype=np.float64)
        ipd_increment[1:][increment_valid[1:]] = _wrap_radians(
            ipd_phase[1:][increment_valid[1:]] - ipd_phase[:-1][increment_valid[1:]]
        )

        per_frame["iid_db"][band_index] = iid
        per_frame["signed_real_coherence"][band_index] = signed
        per_frame["magnitude_coherence"][band_index] = magnitude
        per_frame["side_energy_fraction"][band_index] = side_fraction
        per_frame["ipd_increment_rad"][band_index] = ipd_increment
        masks["energy_valid"][band_index] = energy_valid
        masks["two_sided_valid"][band_index] = two_sided
        masks["ipd_phase_valid"][band_index] = ipd_phase_valid
        masks["ipd_increment_valid"][band_index] = increment_valid

        descriptor_counts: dict[str, int] = {}
        for metric in _METRICS:
            descriptor_metric = (
                "abs_iid_db" if metric == "iid_db"
                else "abs_ipd_increment_rad" if metric == "ipd_increment_rad"
                else metric
            )
            median, iqr, count = _descriptor_summary(metric, per_frame[metric][band_index])
            features[f"SC_{name}_{descriptor_metric}_median"] = median
            features[f"SC_{name}_{descriptor_metric}_iqr"] = iqr
            descriptor_counts[descriptor_metric] = count

        energy_count = int(np.count_nonzero(energy_valid))
        two_sided_count = int(np.count_nonzero(two_sided))
        phase_count = int(np.count_nonzero(ipd_phase_valid))
        increment_count = int(np.count_nonzero(increment_valid))
        if energy_count == 0:
            status = "missing_low_energy"
        elif two_sided_count == 0:
            status = "partial_missing_one_side"
        elif increment_count < int(CONFIG["minimum_descriptor_frames"]):
            status = "partial_low_cross_coherence_or_adjacency"
        else:
            status = "ok"
        quality.update(
            status=status,
            energy_threshold=float(threshold),
            energy_valid_frames=energy_count,
            two_sided_valid_frames=two_sided_count,
            ipd_phase_valid_frames=phase_count,
            ipd_increment_valid_frames=increment_count,
            descriptor_valid_frames=descriptor_counts,
        )
        band_quality.append(quality)

    finite_features = sum(np.isfinite(float(value)) for value in features.values())
    any_energy = any(int(band.get("energy_valid_frames", 0)) > 0 for band in band_quality)
    all_ok = all(band["status"] == "ok" for band in band_quality)
    if finite_features == 0:
        status = "insufficient_valid_frames" if any_energy else "low_energy"
    elif all_ok:
        status = "ok"
    else:
        status = "partial"
    return {
        "version": VERSION,
        "status": status,
        "quality_eligible": int(all_ok),
        "sample_rate_hz": SAMPLE_RATE,
        "native_channels": 2,
        "sample_count": int(values.shape[0]),
        "frame_count": int(n_frames),
        "bands": band_quality,
        "per_frame": per_frame,
        "valid_masks": masks,
        "features": features,
    }


__all__ = [
    "BANDS_HZ",
    "CONFIG",
    "FEATURE_NAMES",
    "HOP_LENGTH",
    "N_FFT",
    "SAMPLE_RATE",
    "VERSION",
    "extract_stereo_candidate",
]
