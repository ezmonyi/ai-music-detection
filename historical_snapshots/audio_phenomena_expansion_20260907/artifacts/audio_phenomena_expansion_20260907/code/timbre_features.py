#!/usr/bin/env python3
"""Interpretable timbral-continuity measurements for isolated sources.

The T family describes changes in a low-dimensional spectral envelope.  It is
not a source-identity classifier, and a value computed on a mixture cannot be
interpreted as the continuity of any particular instrument or singer.

MFCC coefficient 0 (level) and coefficient 1 (gross spectral tilt/register)
are deliberately excluded.  Validation additionally matches approximate F0
register and RMS level; the representation itself is not claimed to be pitch
invariant.  Work and memory are linear in the number of frames.
"""

from __future__ import annotations

import math
from typing import TypeAlias

import numpy as np
from scipy.fft import dct
from scipy.signal import stft


FeatureValue: TypeAlias = float | int | str

STANDARD_SR = 16_000
MIN_DURATION_SEC = 4.0
SUMMARY_BLOCK_SEC = 0.5
N_MELS = 40
N_MFCC = 13
MFCC_FIRST = 2
MFCC_LAST = 12
MFCC_DISTANCE_SCALE_DB = 20.0
LARGE_STEP_THRESHOLD = 0.25
EPS = 1e-12

T_MEASURE_NAMES = (
    "T_mfcc2_12_step_median",
    "T_mfcc2_12_step_p90",
    "T_mfcc2_12_step_max",
    "T_mfcc2_12_step_iqr",
    "T_mfcc2_12_large_step_rate",
)


def _base_result(duration: float, sr: int) -> dict[str, FeatureValue]:
    result: dict[str, FeatureValue] = {
        name: float("nan") for name in T_MEASURE_NAMES
    }
    result.update(
        {
            "T_status": "not_evaluated",
            "T_duration_sec": float(duration),
            "T_input_sr_hz": int(sr),
            "T_standardized_sr_16000": int(sr == STANDARD_SR),
            "T_required_scope": "isolated_source",
            "T_active_coverage_fraction": float("nan"),
            "T_eligible_block_fraction": float("nan"),
            "T_quality_score": float("nan"),
            "T_eligible_block_count": 0,
            "T_step_count": 0,
            "T_summary_block_sec": float(SUMMARY_BLOCK_SEC),
            "T_mfcc_first_coefficient": int(MFCC_FIRST),
            "T_mfcc_last_coefficient": int(MFCC_LAST),
        }
    )
    return result


def _analysis_sizes(sr: int) -> tuple[int, int]:
    """Return an approximately 64-ms window and 16-ms hop."""
    target = max(256, int(round(0.064 * sr)))
    n_fft = 1 << int(math.ceil(math.log2(target)))
    n_fft = min(n_fft, 8192)
    hop = max(64, int(round(0.016 * sr)))
    return n_fft, hop


def _mel_filterbank(sr: int, n_fft: int) -> np.ndarray:
    """Construct a deterministic triangular mel filter bank."""
    fmax = min(8_000.0, 0.5 * sr)
    mel_min = 2595.0 * math.log10(1.0 + 80.0 / 700.0)
    mel_max = 2595.0 * math.log10(1.0 + fmax / 700.0)
    mel_points = np.linspace(mel_min, mel_max, N_MELS + 2)
    hz_points = 700.0 * (np.power(10.0, mel_points / 2595.0) - 1.0)
    frequencies = np.fft.rfftfreq(n_fft, 1.0 / sr)
    filters = np.zeros((N_MELS, len(frequencies)), dtype=np.float64)
    for index in range(N_MELS):
        left, centre, right = hz_points[index : index + 3]
        rising = (frequencies - left) / max(centre - left, EPS)
        falling = (right - frequencies) / max(right - centre, EPS)
        filters[index] = np.maximum(0.0, np.minimum(rising, falling))
        # Area normalization prevents wide high-frequency bands dominating.
        filters[index] *= 2.0 / max(right - left, EPS)
    return filters


def _mfcc_blocks(
    y: np.ndarray, sr: int
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """Return robust half-second MFCC-envelope summaries and quality fields."""
    n_fft, hop = _analysis_sizes(sr)
    _, _, spectrum = stft(
        np.asarray(y, dtype=np.float32),
        fs=sr,
        window="hann",
        nperseg=n_fft,
        noverlap=n_fft - hop,
        nfft=n_fft,
        boundary=None,
        padded=False,
    )
    power = np.square(np.abs(spectrum), dtype=np.float64)
    if power.shape[1] == 0:
        return np.empty((0, N_MFCC - MFCC_FIRST)), np.empty(0, bool), 0.0, 0.0, 0.0

    frame_rms = np.sqrt(np.maximum(np.mean(power, axis=0), 0.0))
    peak_rms = float(np.max(frame_rms))
    if peak_rms <= 1e-7 or float(np.sqrt(np.mean(np.square(y)))) <= 1e-7:
        return np.empty((0, N_MFCC - MFCC_FIRST)), np.empty(0, bool), 0.0, 0.0, 0.0
    mel = _mel_filterbank(sr, n_fft) @ power
    # Framewise energy normalization removes level before taking the log.
    mel_fraction = mel / np.maximum(np.sum(mel, axis=0, keepdims=True), EPS)
    log_mel = 10.0 * np.log10(np.maximum(mel_fraction, EPS))
    cepstra = dct(log_mel, type=2, axis=0, norm="ortho")[:N_MFCC].T
    envelope = cepstra[:, MFCC_FIRST:N_MFCC]

    duration = len(y) / sr
    block_count = int(math.floor(duration / SUMMARY_BLOCK_SEC))
    blocks = np.zeros((block_count, envelope.shape[1]), dtype=np.float64)
    eligible = np.zeros(block_count, dtype=bool)
    frame_start = np.arange(envelope.shape[0]) * hop
    frame_centre = frame_start + n_fft // 2
    block_samples = int(round(SUMMARY_BLOCK_SEC * sr))
    frame_block = np.floor_divide(frame_centre, block_samples).astype(int)
    # Do not let an STFT window straddle a summary-block boundary.  This makes
    # a sign flip of one complete block irrelevant to its magnitude envelope.
    within_block = (
        (frame_start >= frame_block * block_samples)
        & (frame_start + n_fft <= (frame_block + 1) * block_samples)
    )
    expected = max(1, int(round(SUMMARY_BLOCK_SEC * sr / hop)))
    minimum_active = max(3, int(math.ceil(0.25 * expected)))
    active = np.zeros(envelope.shape[0], dtype=bool)
    for index in range(block_count):
        candidates = (frame_block == index) & within_block
        if not np.any(candidates):
            continue
        local_peak = float(np.max(frame_rms[candidates]))
        local_active = candidates & (frame_rms >= max(local_peak * 0.02, 1e-7))
        active |= local_active
        use = local_active
        if int(np.sum(use)) < minimum_active:
            continue
        blocks[index] = np.median(envelope[use], axis=0)
        eligible[index] = True

    active_coverage = float(np.sum(active) / max(1, np.sum(within_block)))
    eligible_fraction = float(np.mean(eligible)) if eligible.size else 0.0
    quality = float(math.sqrt(active_coverage * eligible_fraction))
    return blocks, eligible, active_coverage, eligible_fraction, quality


def _step_distances(blocks: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    adjacent = eligible[:-1] & eligible[1:]
    if not np.any(adjacent):
        return np.empty(0, dtype=np.float64)
    delta = blocks[1:] - blocks[:-1]
    # Orthonormal DCT preserves log-envelope distance.  Division by a fixed dB
    # scale keeps the measure interpretable without fitting dataset statistics.
    distance = np.sqrt(np.mean(np.square(delta), axis=1)) / MFCC_DISTANCE_SCALE_DB
    return distance[adjacent]


def timbre_boundary_distance(
    left: np.ndarray, right: np.ndarray, sr: int
) -> dict[str, FeatureValue]:
    """Compare the terminal and initial one-second envelopes of two sources.

    This helper is intended for controlled validation where the boundary is
    known.  It is not part of the per-clip production feature family.
    """
    left_audio = np.asarray(left)
    right_audio = np.asarray(right)
    if left_audio.ndim != 1 or right_audio.ndim != 1:
        raise ValueError("left and right must be one-dimensional")
    if not isinstance(sr, (int, np.integer)) or int(sr) <= 0:
        raise ValueError(f"sr must be a positive integer, got {sr!r}")
    if not np.all(np.isfinite(left_audio)) or not np.all(np.isfinite(right_audio)):
        raise ValueError("boundary inputs contain non-finite samples")
    sr = int(sr)
    span = min(len(left_audio), len(right_audio), sr)
    result: dict[str, FeatureValue] = {
        "T_boundary_mfcc2_12_distance": float("nan"),
        "T_boundary_status": "not_evaluated",
        "T_boundary_context_sec": float(span / sr),
    }
    if span < int(round(SUMMARY_BLOCK_SEC * sr)):
        result["T_boundary_status"] = "missing_short_context"
        return result
    left_blocks, left_ok, _, _, _ = _mfcc_blocks(
        np.asarray(left_audio[-span:], dtype=np.float32), sr
    )
    right_blocks, right_ok, _, _, _ = _mfcc_blocks(
        np.asarray(right_audio[:span], dtype=np.float32), sr
    )
    if not np.any(left_ok) or not np.any(right_ok):
        result["T_boundary_status"] = "missing_low_energy_context"
        return result
    left_summary = np.median(left_blocks[left_ok], axis=0)
    right_summary = np.median(right_blocks[right_ok], axis=0)
    distance = np.sqrt(np.mean(np.square(right_summary - left_summary)))
    result["T_boundary_mfcc2_12_distance"] = float(distance / MFCC_DISTANCE_SCALE_DB)
    result["T_boundary_status"] = "ok"
    return result


def extract_timbre_features(y: np.ndarray, sr: int) -> dict[str, FeatureValue]:
    """Measure spectral-envelope continuity in one mono isolated source.

    Missing measurements remain NaN and are explained by ``T_status``.  The
    caller should supply mono 16-kHz audio; arbitrary positive rates are
    accepted and reported, but this function neither resamples nor pads.
    """
    audio = np.asarray(y)
    if audio.ndim != 1:
        raise ValueError(f"y must be one-dimensional, got shape {audio.shape}")
    if not isinstance(sr, (int, np.integer)) or int(sr) <= 0:
        raise ValueError(f"sr must be a positive integer, got {sr!r}")
    audio = np.asarray(audio, dtype=np.float32)
    if not np.all(np.isfinite(audio)):
        raise ValueError("y contains non-finite samples")
    sr = int(sr)
    duration = float(len(audio) / sr)
    result = _base_result(duration, sr)

    if duration < MIN_DURATION_SEC:
        result["T_status"] = "missing_short_duration"
        return result

    blocks, eligible, active_coverage, eligible_fraction, quality = _mfcc_blocks(audio, sr)
    result.update(
        {
            "T_active_coverage_fraction": active_coverage,
            "T_eligible_block_fraction": eligible_fraction,
            "T_quality_score": quality,
            "T_eligible_block_count": int(np.sum(eligible)),
        }
    )
    if blocks.size == 0 or active_coverage == 0.0:
        result["T_status"] = "missing_low_energy"
        return result

    steps = _step_distances(blocks, eligible)
    result["T_step_count"] = int(len(steps))
    if len(steps) < 3:
        result["T_status"] = "missing_insufficient_active_continuity"
        return result

    q25, q75 = np.quantile(steps, [0.25, 0.75])
    result.update(
        {
            "T_mfcc2_12_step_median": float(np.median(steps)),
            "T_mfcc2_12_step_p90": float(np.percentile(steps, 90)),
            "T_mfcc2_12_step_max": float(np.max(steps)),
            "T_mfcc2_12_step_iqr": float(q75 - q25),
            "T_mfcc2_12_large_step_rate": float(np.mean(steps >= LARGE_STEP_THRESHOLD)),
            "T_status": "ok",
        }
    )
    return result
