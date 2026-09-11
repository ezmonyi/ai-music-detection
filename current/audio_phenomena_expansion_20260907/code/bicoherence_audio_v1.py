"""Fixed continuous-audio development extractor; no external admission.

The pooled statistic is amplitude weighted. Overlapping frames are not asserted
independent, and stable linear oscillators can give b²=1 without a nonlinear
operation. No significance, causal label, null calibration, or AI score follows.

``extract`` returns NumPy arrays (suitable for np.savez) and JSON-safe metadata
(suitable for json.dumps(..., allow_nan=False)). NaN denotes unavailable values
in arrays only; every grid cell, including ineligible cells, has an explicit
record and the unfiltered primitive result in metadata. This module never saves
files, reads audio, resamples, mixes channels, clips, or normalizes the waveform.
"""
from __future__ import annotations

import numpy as np

from bicoherence_primitive_v2 import VERSION as PRIMITIVE_VERSION
from bicoherence_primitive_v2 import estimate

VERSION = "bicoherence_audio_v1"
SAMPLE_RATE = 16000
POOL_SAMPLES = 64000
N_FFT = 1024
HOP = 256
FRAMES_PER_POOL = 247
ENERGY_FRACTION_MIN = 1e-6


def pair_grid() -> np.ndarray:
    """Fixed [f1, f2, f1+f2] integer-bin grid, including sum bin 256."""
    return np.asarray([(a, b, a + b)
                       for a in range(8, 193, 8)
                       for b in range(a, 193, 8) if a + b <= 256],
                      dtype=np.int64)


def _validate(audio: np.ndarray, sample_rate: int) -> None:
    if isinstance(sample_rate, (bool, np.bool_)) or not isinstance(
            sample_rate, (int, np.integer)):
        raise TypeError("sample_rate must be an integer, not a float or bool")
    if sample_rate != SAMPLE_RATE:
        raise ValueError("sample_rate must be exactly 16000 Hz; no implicit resampling")
    if not isinstance(audio, np.ndarray) or audio.dtype != np.dtype(np.float64):
        raise TypeError("audio must be a NumPy float64 array; no implicit conversion")
    if audio.ndim != 1 or not np.isfinite(audio).all():
        raise ValueError("audio must be finite, real, one-dimensional mono audio")


def extract(audio: np.ndarray, sample_rate: int) -> dict:
    """Analyze complete nonoverlapping 4-second pools with a fixed STFT/grid.

    Mean removal precedes the periodic Hann window in every 1024-sample frame.
    Coefficients are rfft(window * (frame - mean(frame))) / sum(window).
    Energy is the sum over frames of squared coefficient magnitudes, without
    one-sided doubling, and its denominator sums bins 1..512 (DC excluded).

    Eligibility requires each triad bin to have fraction >= 1e-6 and a defined
    primitive. Ineligible reported b²/biphase are missing; raw primitive values
    and sufficient sums are retained even when the energy floor fails. Biphase
    uses the primitive's atan2 convention [-pi, pi], descriptive only. A record
    shorter than 64000 samples has explicit insufficient_support and no pools.
    Unsupported float64 arithmetic raises FloatingPointError rather than
    silently turning nonzero energy into zero. Memory use grows with duration.
    """
    _validate(audio, sample_rate)
    count = audio.size // POOL_SAMPLES
    retained = count * POOL_SAMPLES
    grid = pair_grid()
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)
    pool_starts = np.arange(count, dtype=np.int64) * POOL_SAMPLES
    frame_offsets = np.arange(FRAMES_PER_POOL, dtype=np.int64) * HOP
    frame_starts = pool_starts[:, None] + frame_offsets[None, :]
    spectra = np.empty((count, FRAMES_PER_POOL, N_FFT // 2 + 1), np.complex128)
    means = np.empty((count, FRAMES_PER_POOL), np.float64)
    bin_energy = np.empty((count, N_FFT // 2 + 1), np.float64)
    fractions = np.full_like(bin_energy, np.nan)
    total_energy = np.empty(count, np.float64)
    floor_mask = np.zeros((count, len(grid)), dtype=np.bool_)
    primitive_mask = np.zeros_like(floor_mask)
    eligible_mask = np.zeros_like(floor_mask)
    scores = np.full(floor_mask.shape, np.nan, np.float64)
    phases = np.full_like(scores, np.nan)
    primitive_scores = np.full_like(scores, np.nan)
    primitive_phases = np.full_like(scores, np.nan)
    pool_records = []

    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        for p, pool_start in enumerate(pool_starts):
            pool = audio[pool_start:pool_start + POOL_SAMPLES]
            frames = np.lib.stride_tricks.sliding_window_view(pool, N_FFT)[::HOP]
            means[p] = frames.mean(axis=1)
            centered = frames - means[p, :, None]
            windowed = centered * window
            if np.any((centered != 0) & (window != 0) & (windowed == 0)):
                raise FloatingPointError("unsupported_underflow: windowed sample")
            unnormalized = np.fft.rfft(windowed, axis=1)
            spectra[p] = unnormalized / window.sum()
            if np.any((unnormalized != 0) & (spectra[p] == 0)):
                raise FloatingPointError("unsupported_underflow: STFT normalization")
            if np.any(np.any(windowed != 0, axis=1) & ~np.any(spectra[p] != 0, axis=1)):
                raise FloatingPointError("unsupported_underflow: vanished nonzero frame")
            if not np.isfinite(spectra[p]).all():
                raise FloatingPointError("nonfinite STFT coefficients")
            magnitudes = np.abs(spectra[p])
            powers = magnitudes * magnitudes
            if np.any((magnitudes != 0) & (powers == 0)):
                raise FloatingPointError("unsupported_underflow: coefficient energy")
            bin_energy[p] = powers.sum(axis=0)
            total_energy[p] = bin_energy[p, 1:].sum()
            if not np.isfinite(bin_energy[p]).all() or not np.isfinite(total_energy[p]):
                raise FloatingPointError("nonfinite pooled coefficient energy")
            if total_energy[p] > 0:
                fractions[p] = bin_energy[p] / total_energy[p]
                if np.any((bin_energy[p] != 0) & (fractions[p] == 0)):
                    raise FloatingPointError("unsupported_underflow: bin energy fraction")
                floor_mask[p] = np.all(fractions[p, grid] >= ENERGY_FRACTION_MIN, axis=1)
            zero_amplitude = not np.any(pool)
            pool_status = ("zero_amplitude" if zero_amplitude else
                           "zero_non_dc_energy" if total_energy[p] == 0 else "ok")
            cells = []
            for c, (f1, f2, f3) in enumerate(grid):
                primitive = estimate(spectra[p], pair=(int(f1), int(f2)))
                raw_score = primitive["squared_bicoherence"]
                raw_phase = primitive["biphase_radians"]
                primitive_mask[p, c] = primitive["status"] == "ok" and raw_score is not None
                eligible_mask[p, c] = floor_mask[p, c] and primitive_mask[p, c]
                if raw_score is not None:
                    primitive_scores[p, c] = raw_score
                if raw_phase is not None:
                    primitive_phases[p, c] = raw_phase
                if eligible_mask[p, c]:
                    scores[p, c] = raw_score
                    if raw_phase is not None:
                        phases[p, c] = raw_phase
                    status = "ok"
                elif not primitive_mask[p, c]:
                    status = primitive["status"]
                else:
                    status = "below_energy_fraction_floor"
                triad_fractions = ([float(v) for v in fractions[p, [f1, f2, f3]]]
                                   if total_energy[p] > 0 else [None, None, None])
                cells.append({"frequency_bins": [int(f1), int(f2), int(f3)],
                              "status": status,
                              "energy_fractions": triad_fractions,
                              "energy_floor_passed": bool(floor_mask[p, c]),
                              "eligible": bool(eligible_mask[p, c]),
                              "squared_bicoherence": raw_score if eligible_mask[p, c] else None,
                              "biphase_radians": raw_phase if eligible_mask[p, c] else None,
                              "primitive": primitive})
            pool_records.append({"pool_index": p, "start_sample": int(pool_start),
                                 "stop_sample_exclusive": int(pool_start + POOL_SAMPLES),
                                 "status": pool_status, "zero_amplitude": zero_amplitude,
                                 "coefficient_rows": FRAMES_PER_POOL,
                                 "independent_realization_count": None,
                                 "total_non_dc_coefficient_energy": float(total_energy[p]),
                                 "eligible_cell_count": int(eligible_mask[p].sum()),
                                 "cells": cells})

    return {"arrays": {"window": window, "frequency_bins": grid,
                       "frequency_hz": grid.astype(np.float64) * SAMPLE_RATE / N_FFT,
                       "pool_start_samples": pool_starts,
                       "frame_offset_samples": frame_offsets,
                       "frame_start_samples": frame_starts, "frame_means": means,
                       "spectra": spectra, "bin_coefficient_energy": bin_energy,
                       "total_non_dc_coefficient_energy": total_energy,
                       "bin_energy_fraction": fractions,
                       "energy_floor_mask": floor_mask,
                       "primitive_defined_mask": primitive_mask,
                       "eligible_mask": eligible_mask,
                       "squared_bicoherence": scores, "biphase_radians": phases,
                       "primitive_squared_bicoherence": primitive_scores,
                       "primitive_biphase_radians": primitive_phases},
            "metadata": {"version": VERSION, "primitive_version": PRIMITIVE_VERSION,
                         "status": "ok" if count else "insufficient_support",
                         "input_samples": int(audio.size), "sample_rate_hz": SAMPLE_RATE,
                         "input_zero_amplitude": not np.any(audio),
                         "pool_samples": POOL_SAMPLES, "pool_duration_seconds": 4.0,
                         "pool_count": int(count), "analyzed_samples": int(retained),
                         "discarded_tail_samples": int(audio.size - retained),
                         "tail_start_sample": int(retained), "tail_policy": "discard_no_padding",
                         "n_fft": N_FFT, "hop_samples": HOP,
                         "frames_per_pool": FRAMES_PER_POOL,
                         "window": "periodic_hann", "window_sum": float(window.sum()),
                         "transform": "rfft((frame - frame.mean()) * window) / window.sum()",
                         "pool_boundary_policy": "no_frame_crosses_pool_boundary",
                         "energy": "sum_frames(abs(coefficient)**2); no one-sided doubling",
                         "fraction_denominator_bins_inclusive": [1, 512],
                         "energy_fraction_min_inclusive": ENERGY_FRACTION_MIN,
                         "grid_cell_count": int(len(grid)),
                         "grid": "positive parents 8..192 step 8; f1<=f2; f1+f2<=256",
                         "biphase_convention": "atan2(imag,real) in [-pi,pi]; descriptive_only",
                         "array_missing_encoding": "NaN; JSON metadata missing values are null",
                         "independent_realization_count": None,
                         "frames_overlap_and_are_not_asserted_independent": True,
                         "amplitude_weighted_not_phase_only": True,
                         "linear_fixed_phase_tones_can_have_high_bicoherence": True,
                         "null_calibrated": False, "significance_inferred": False,
                         "external_validation_passed": False, "classifier_admitted": False,
                         "pools": pool_records}}
