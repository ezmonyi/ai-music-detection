#!/usr/bin/env python3
"""CPU descriptors for tonal/chroma paths (H) and long-range recurrence (M).

These are measurement candidates, not chord recognition, motif transcription,
or AI-authorship rules.  The production cohort is expected to supply mono
16 kHz audio, although :func:`extract_musical_features` accepts any positive
sample rate and records whether that contract was met.

The recurrence implementation never constructs a full self-similarity matrix.
It compares one-second chroma summaries only at bounded lags, so memory is
O(T) and work is O(T * number_of_lags * 12) for T one-second blocks.
"""

from __future__ import annotations

import math
from typing import TypeAlias

import librosa
import numpy as np
from scipy.signal import convolve


FeatureValue: TypeAlias = float | int | str

STANDARD_SR = 16_000
H_MIN_DURATION_SEC = 8.0
M_MIN_DURATION_SEC = 45.0
SUMMARY_BLOCK_SEC = 1.0
M_MIN_LAG_SEC = 8
M_MAX_LAG_SEC = 60
M_PATTERN_SEC = 4
EPS = 1e-12

H_MEASURE_NAMES = (
    "H_pitch_class_entropy_norm",
    "H_pc_token_entropy_norm",
    "H_chroma_path_change_median",
    "H_chroma_path_change_iqr",
    "H_pc_path_step_median",
    "H_pc_path_large_step_rate",
)
M_MEASURE_NAMES = (
    "M_recurrence_peak_similarity",
    "M_recurrence_density",
    "M_recurrence_lag_contrast",
    "M_best_lag_sec",
    "M_best_transposition_semitones",
    "M_returning_pattern_count",
)


def _base_result(prefix: str, duration: float, sr: int) -> dict[str, FeatureValue]:
    names = H_MEASURE_NAMES if prefix == "H" else M_MEASURE_NAMES
    result: dict[str, FeatureValue] = {name: float("nan") for name in names}
    result.update(
        {
            f"{prefix}_status": "not_evaluated",
            f"{prefix}_duration_sec": float(duration),
            f"{prefix}_input_sr_hz": int(sr),
            f"{prefix}_standardized_sr_16000": int(sr == STANDARD_SR),
            f"{prefix}_active_coverage_fraction": float("nan"),
            f"{prefix}_tonal_coverage_fraction": float("nan"),
            f"{prefix}_quality_score": float("nan"),
            f"{prefix}_eligible_block_count": 0,
        }
    )
    if prefix == "M":
        result["M_evaluated_lag_count"] = 0
        result["M_pattern_span_sec"] = float(M_PATTERN_SEC)
        result["M_min_lag_sec"] = float(M_MIN_LAG_SEC)
    return result


def _analysis_sizes(sr: int) -> tuple[int, int]:
    """Return a roughly 0.2-second FFT and 64-ms hop for arbitrary ``sr``."""
    target = max(256, int(round(0.20 * sr)))
    n_fft = 1 << int(math.ceil(math.log2(target)))
    n_fft = min(n_fft, 8192)
    hop = max(64, int(round(0.064 * sr)))
    return n_fft, hop


def _chroma_blocks(
    y: np.ndarray, sr: int
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """Return L2-normalized one-second chroma blocks and eligibility mask."""
    n_fft, hop = _analysis_sizes(sr)
    spectrum = librosa.stft(
        y=np.asarray(y, dtype=np.float32),
        n_fft=n_fft,
        hop_length=hop,
        window="hann",
        center=False,
    )
    power = np.square(np.abs(spectrum), dtype=np.float64)
    spectral_rms = np.sqrt(np.mean(power, axis=0) + EPS)
    if spectral_rms.size == 0 or float(np.max(spectral_rms)) <= EPS:
        return np.empty((0, 12)), np.empty(0, dtype=bool), 0.0, 0.0, 0.0

    # Relative activity is scale-invariant; the caller-level RMS gate prevents
    # numerical noise in nominally silent signals from becoming "active".
    active = spectral_rms >= max(float(np.max(spectral_rms)) * 0.01, EPS)
    chroma = librosa.feature.chroma_stft(
        S=power,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop,
        tuning=0.0,
        norm=None,
    ).T
    frame_sum = chroma.sum(axis=1, keepdims=True)
    chroma_l1 = chroma / np.maximum(frame_sum, EPS)
    entropy = -np.sum(chroma_l1 * np.log(chroma_l1 + EPS), axis=1) / math.log(12.0)
    informative = active & np.isfinite(entropy) & (entropy <= 0.92)

    duration = len(y) / sr
    block_count = int(math.floor(duration / SUMMARY_BLOCK_SEC))
    blocks = np.zeros((block_count, 12), dtype=np.float64)
    eligible = np.zeros(block_count, dtype=bool)
    block_index = np.floor(np.arange(chroma.shape[0]) * hop / sr).astype(int)
    expected_frames = max(1, int(round(SUMMARY_BLOCK_SEC * sr / hop)))
    minimum_informative = max(2, int(math.ceil(0.25 * expected_frames)))
    for index in range(block_count):
        use = (block_index == index) & informative
        if int(use.sum()) < minimum_informative:
            continue
        value = np.mean(chroma_l1[use], axis=0)
        norm = float(np.linalg.norm(value))
        if norm > EPS:
            blocks[index] = value / norm
            eligible[index] = True

    active_coverage = float(np.mean(active)) if active.size else 0.0
    tonal_coverage = float(informative.sum() / max(1, active.sum()))
    quality = float(math.sqrt(max(0.0, active_coverage * tonal_coverage)))
    return blocks, eligible, active_coverage, tonal_coverage, quality


def _fill_quality(
    result: dict[str, FeatureValue],
    prefix: str,
    eligible: np.ndarray,
    active_coverage: float,
    tonal_coverage: float,
    quality: float,
) -> None:
    result[f"{prefix}_active_coverage_fraction"] = float(active_coverage)
    result[f"{prefix}_tonal_coverage_fraction"] = float(tonal_coverage)
    result[f"{prefix}_quality_score"] = float(quality)
    result[f"{prefix}_eligible_block_count"] = int(eligible.sum())


def _h_features(blocks: np.ndarray, eligible: np.ndarray) -> dict[str, float]:
    values = blocks[eligible]
    global_chroma = values.sum(axis=0)
    probability = global_chroma / max(float(global_chroma.sum()), EPS)
    pc_entropy = float(-np.sum(probability * np.log(probability + EPS)) / math.log(12.0))

    tokens = np.argmax(blocks, axis=1)
    token_counts = np.bincount(tokens[eligible], minlength=12).astype(np.float64)
    token_probability = token_counts[token_counts > 0] / max(float(token_counts.sum()), EPS)
    token_entropy = float(
        -np.sum(token_probability * np.log(token_probability + EPS)) / math.log(12.0)
    )

    adjacent = eligible[:-1] & eligible[1:]
    similarities = np.sum(blocks[:-1] * blocks[1:], axis=1)[adjacent]
    changes = np.clip(1.0 - similarities, 0.0, 2.0)
    token_steps = np.abs(tokens[1:] - tokens[:-1])[adjacent]
    token_steps = np.minimum(token_steps, 12 - token_steps) / 6.0
    q25, q75 = np.quantile(changes, [0.25, 0.75])
    return {
        "H_pitch_class_entropy_norm": pc_entropy,
        "H_pc_token_entropy_norm": token_entropy,
        "H_chroma_path_change_median": float(np.median(changes)),
        "H_chroma_path_change_iqr": float(q75 - q25),
        "H_pc_path_step_median": float(np.median(token_steps)),
        "H_pc_path_large_step_rate": float(np.mean(token_steps >= 0.5)),
    }


def _signed_pitch_shift(shift: int) -> int:
    return shift if shift <= 6 else shift - 12


def _m_features(
    blocks: np.ndarray, eligible: np.ndarray
) -> tuple[dict[str, float], int]:
    """Evaluate 4-second patterns at lags >=8 s without an O(T^2) SSM."""
    n_blocks = len(blocks)
    max_lag = min(M_MAX_LAG_SEC, n_blocks - M_PATTERN_SEC)
    lag_records: list[tuple[int, float, float, np.ndarray, np.ndarray]] = []

    for lag in range(M_MIN_LAG_SEC, max_lag + 1):
        first = blocks[:-lag]
        second = blocks[lag:]
        pair_valid = eligible[:-lag] & eligible[lag:]
        if len(pair_valid) < M_PATTERN_SEC:
            continue
        window_valid = convolve(
            pair_valid.astype(np.int8), np.ones(M_PATTERN_SEC, dtype=np.int8), mode="valid"
        ) == M_PATTERN_SEC
        if not np.any(window_valid):
            continue

        shift_scores: list[np.ndarray] = []
        for shift in range(12):
            # ``shift`` is the semitone rotation applied to the earlier pattern
            # to align it to the later pattern.
            frame_similarity = np.sum(np.roll(first, shift, axis=1) * second, axis=1)
            pattern_similarity = convolve(
                frame_similarity, np.ones(M_PATTERN_SEC) / M_PATTERN_SEC, mode="valid"
            )
            shift_scores.append(pattern_similarity)
        score_matrix = np.vstack(shift_scores)
        best_shift = np.argmax(score_matrix, axis=0)
        best_score = np.max(score_matrix, axis=0)
        valid_scores = np.clip(best_score[window_valid], 0.0, 1.0)
        valid_shifts = best_shift[window_valid]
        peak = float(np.percentile(valid_scores, 90))
        density = float(np.mean(valid_scores >= 0.88))
        lag_records.append((lag, peak, density, valid_scores, valid_shifts))

    if not lag_records:
        return {}, 0

    # Report the earliest well-supported lag near the global peak.  This treats
    # 16 s as the fundamental return when 32 s is merely a more exact multiple,
    # while the density floor prevents a single accidental match from winning.
    global_peak = max(item[1] for item in lag_records)
    near_peak = [item for item in lag_records if item[1] >= global_peak - 0.01]
    density_floor = max(0.10, 0.50 * max(item[2] for item in near_peak))
    fundamental = [item for item in near_peak if item[2] >= density_floor]
    best = min(fundamental, key=lambda item: item[0]) if fundamental else max(
        lag_records, key=lambda item: (item[1], item[2], -item[0])
    )
    lag, peak, density, scores, shifts = best
    strong = scores >= max(0.88, float(np.percentile(scores, 90)) - 1e-9)
    strong_shifts = shifts[strong]
    if strong_shifts.size:
        counts = np.bincount(strong_shifts, minlength=12)
        transposition = _signed_pitch_shift(int(np.argmax(counts)))
    else:
        transposition = 0
    lag_peaks = np.asarray([item[1] for item in lag_records], dtype=float)
    return (
        {
            "M_recurrence_peak_similarity": peak,
            "M_recurrence_density": density,
            "M_recurrence_lag_contrast": float(peak - np.median(lag_peaks)),
            "M_best_lag_sec": float(lag),
            "M_best_transposition_semitones": float(transposition),
            "M_returning_pattern_count": float(np.sum(scores >= 0.88)),
        },
        len(lag_records),
    )


def extract_musical_features(y: np.ndarray, sr: int) -> dict[str, FeatureValue]:
    """Extract H and M measurements from one mono waveform.

    Missing measurements remain ``NaN`` and are explained by ``H_status`` or
    ``M_status``.  The function does not pad short clips or resample them; the
    integration layer should standardize inputs to mono 16 kHz first.
    """
    audio = np.asarray(y)
    if audio.ndim != 1:
        raise ValueError(f"y must be one-dimensional, got shape {audio.shape}")
    if not isinstance(sr, (int, np.integer)) or int(sr) <= 0:
        raise ValueError(f"sr must be a positive integer, got {sr!r}")
    sr = int(sr)
    audio = np.asarray(audio, dtype=np.float32)
    if not np.all(np.isfinite(audio)):
        raise ValueError("y contains non-finite samples")
    duration = float(len(audio) / sr)
    result = _base_result("H", duration, sr)
    result.update(_base_result("M", duration, sr))

    if duration < H_MIN_DURATION_SEC:
        result["H_status"] = "missing_short_duration"
        result["M_status"] = "missing_short_duration"
        return result

    centered = audio.astype(np.float64) - float(np.mean(audio, dtype=np.float64))
    waveform_rms = float(np.sqrt(np.mean(np.square(centered), dtype=np.float64)))
    if waveform_rms < 1e-6:
        result["H_status"] = "missing_low_energy"
        result["M_status"] = "missing_low_energy"
        return result

    blocks, eligible, active, tonal, quality = _chroma_blocks(centered, sr)
    _fill_quality(result, "H", eligible, active, tonal, quality)
    _fill_quality(result, "M", eligible, active, tonal, quality)
    adjacent_count = int(np.sum(eligible[:-1] & eligible[1:])) if eligible.size > 1 else 0
    if int(eligible.sum()) < 4 or adjacent_count < 3:
        result["H_status"] = "missing_no_informative_pitch"
    else:
        result.update(_h_features(blocks, eligible))
        result["H_status"] = "ok"

    if duration < M_MIN_DURATION_SEC:
        result["M_status"] = "missing_short_duration"
        return result
    if int(eligible.sum()) < max(12, M_PATTERN_SEC * 2):
        result["M_status"] = "missing_no_informative_pitch"
        return result
    m_values, lag_count = _m_features(blocks, eligible)
    result["M_evaluated_lag_count"] = int(lag_count)
    if not m_values:
        result["M_status"] = "missing_insufficient_long_range_coverage"
        return result
    result.update(m_values)
    result["M_status"] = "ok"
    return result


__all__ = [
    "H_MEASURE_NAMES",
    "H_MIN_DURATION_SEC",
    "M_MEASURE_NAMES",
    "M_MIN_DURATION_SEC",
    "STANDARD_SR",
    "extract_musical_features",
]
