#!/usr/bin/env python3
"""Synthetic unit tests for the T signal-continuity descriptors."""

from __future__ import annotations

import numpy as np

from timbre_features import T_MEASURE_NAMES, extract_timbre_features, timbre_boundary_distance


SR = 16_000


def _tone(seconds: float, harmonics: tuple[float, ...], phase: float = 0.0, sr: int = SR) -> np.ndarray:
    t = np.arange(int(round(seconds * sr)), dtype=np.float64) / sr
    y = np.zeros_like(t)
    for index, amplitude in enumerate(harmonics, start=1):
        y += amplitude * np.sin(2 * np.pi * 220.0 * index * t + phase * index)
    y /= max(np.max(np.abs(y)), 1e-12)
    return (0.2 * y).astype(np.float32)


def _alternate(a: np.ndarray, b: np.ndarray, repeats: int = 6) -> np.ndarray:
    return np.concatenate([value for _ in range(repeats) for value in (a, b)])


def test_switched_spectral_envelope_orders_above_same_source() -> None:
    a1 = _tone(0.5, (1.0, 0.6, 0.35, 0.2), phase=0.0)
    a2 = _tone(0.5, (1.0, 0.6, 0.35, 0.2), phase=1.1)
    b = _tone(0.5, (0.05, 0.10, 0.25, 1.0, 0.8, 0.5), phase=0.4)
    same = extract_timbre_features(_alternate(a1, a2), SR)
    switched = extract_timbre_features(_alternate(a1, b), SR)
    assert same["T_status"] == "ok"
    assert switched["T_status"] == "ok"
    assert switched["T_mfcc2_12_step_median"] > same["T_mfcc2_12_step_median"] + 0.1
    assert switched["T_mfcc2_12_step_p90"] > same["T_mfcc2_12_step_p90"] + 0.1
    same_boundary = timbre_boundary_distance(a1, a2, SR)
    switched_boundary = timbre_boundary_distance(a1, b, SR)
    assert same_boundary["T_boundary_status"] == "ok"
    assert switched_boundary["T_boundary_mfcc2_12_distance"] > same_boundary[
        "T_boundary_mfcc2_12_distance"
    ] + 0.1


def test_level_and_polarity_are_invariant() -> None:
    a = _tone(0.5, (1.0, 0.4, 0.2, 0.1), phase=0.2)
    b = _tone(0.5, (1.0, 0.55, 0.3, 0.18), phase=0.8)
    base = extract_timbre_features(_alternate(a, b), SR)
    gain = extract_timbre_features(0.37 * _alternate(a, b), SR)
    polarity = extract_timbre_features(-_alternate(a, b), SR)
    for name in T_MEASURE_NAMES:
        assert np.isclose(base[name], gain[name], atol=1e-4)
        assert np.isclose(base[name], polarity[name], atol=2e-6)


def test_short_and_silent_audio_are_missing_not_zero() -> None:
    short = extract_timbre_features(_tone(3.0, (1.0, 0.5)), SR)
    assert short["T_status"] == "missing_short_duration"
    assert np.isnan(short["T_mfcc2_12_step_p90"])

    silence = extract_timbre_features(np.zeros(6 * SR, dtype=np.float32), SR)
    assert silence["T_status"] == "missing_low_energy"
    assert silence["T_step_count"] == 0
    assert np.isnan(silence["T_mfcc2_12_step_median"])
    boundary = timbre_boundary_distance(np.zeros(SR), _tone(1.0, (1.0, 0.5)), SR)
    assert boundary["T_boundary_status"] == "missing_low_energy_context"
    assert np.isnan(boundary["T_boundary_mfcc2_12_distance"])


def test_sparse_activity_has_explicit_coverage_and_missing_status() -> None:
    y = np.zeros(6 * SR, dtype=np.float32)
    y[2 * SR : 2 * SR + SR // 4] = _tone(0.25, (1.0, 0.5))
    measured = extract_timbre_features(y, SR)
    assert measured["T_active_coverage_fraction"] < 0.1
    assert measured["T_status"] == "missing_insufficient_active_continuity"
    assert np.isnan(measured["T_mfcc2_12_step_p90"])


def test_function_is_generic_and_rejects_bad_inputs() -> None:
    measured = extract_timbre_features(_alternate(
        _tone(0.5, (1.0, 0.5), sr=8_000),
        _tone(0.5, (1.0, 0.4), sr=8_000),
    ), 8_000)
    assert measured["T_status"] == "ok"
    assert measured["T_standardized_sr_16000"] == 0
    try:
        extract_timbre_features(np.zeros((2, 100)), SR)
    except ValueError as error:
        assert "one-dimensional" in str(error)
    else:
        raise AssertionError("two-dimensional input was accepted")
