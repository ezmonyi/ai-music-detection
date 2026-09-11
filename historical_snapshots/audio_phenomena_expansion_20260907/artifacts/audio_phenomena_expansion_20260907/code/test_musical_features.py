#!/usr/bin/env python3
"""Synthetic measurement tests for musical_features.py.

The signals have known pitch-class paths and recurrence annotations.  They test
extractor behavior only; they are deliberately unrelated to AI/human labels.
"""

from __future__ import annotations

import numpy as np

from musical_features import extract_musical_features


SR = 16_000


def _render_chord_sequence(
    chords: list[tuple[int, ...]], seconds_per_chord: float = 1.0, sr: int = SR
) -> np.ndarray:
    n = int(round(seconds_per_chord * sr))
    time = np.arange(n, dtype=np.float64) / sr
    fade = min(160, n // 8)
    envelope = np.ones(n, dtype=np.float64)
    envelope[:fade] = np.linspace(0.0, 1.0, fade, endpoint=False)
    envelope[-fade:] = np.linspace(1.0, 0.0, fade, endpoint=False)
    pieces: list[np.ndarray] = []
    for chord in chords:
        value = np.zeros(n, dtype=np.float64)
        for midi in chord:
            frequency = 440.0 * 2.0 ** ((midi - 69) / 12.0)
            value += np.sin(2.0 * np.pi * frequency * time)
            value += 0.20 * np.sin(4.0 * np.pi * frequency * time)
        pieces.append(0.20 * envelope * value / len(chord))
    return np.concatenate(pieces).astype(np.float32)


def _transpose(chord: tuple[int, ...], semitones: int) -> tuple[int, ...]:
    return tuple(note + semitones for note in chord)


def test_h_chroma_path_orders_progression_above_sustained_path() -> None:
    sustained = _render_chord_sequence([(60, 64, 67)] * 12)
    progression = [(60, 64, 67), (62, 65, 69), (59, 62, 67), (65, 69, 72)] * 3
    moving = _render_chord_sequence(progression)

    steady_features = extract_musical_features(sustained, SR)
    moving_features = extract_musical_features(moving, SR)

    assert steady_features["H_status"] == "ok"
    assert moving_features["H_status"] == "ok"
    assert moving_features["H_chroma_path_change_median"] > steady_features[
        "H_chroma_path_change_median"
    ] + 0.05
    assert moving_features["H_pc_token_entropy_norm"] > steady_features[
        "H_pc_token_entropy_norm"
    ] + 0.15


def test_m_recovers_known_long_range_return_under_transposition() -> None:
    phrase = [
        (60, 64, 67), (62, 65, 69), (64, 67, 71), (59, 62, 65),
        (60, 65, 67), (61, 64, 68), (62, 66, 69), (63, 67, 70),
        (60, 63, 67), (62, 65, 68), (64, 68, 71), (59, 64, 67),
        (60, 62, 67), (61, 66, 68), (62, 67, 69), (63, 65, 70),
    ]
    recurring = phrase + [_transpose(chord, 5) for chord in phrase]
    recurring += phrase + [_transpose(chord, 5) for chord in phrase]
    features = extract_musical_features(_render_chord_sequence(recurring), SR)

    assert features["M_status"] == "ok"
    assert features["M_recurrence_peak_similarity"] > 0.96
    assert features["M_recurrence_density"] > 0.40
    assert abs(features["M_best_lag_sec"] - 16.0) <= 1.0
    assert features["M_best_transposition_semitones"] == 5.0


def test_m_recurrence_orders_repeated_phrase_above_nonrepeating_path() -> None:
    phrase = [
        (60, 64, 67), (61, 64, 68), (62, 65, 69), (63, 66, 70),
        (64, 67, 71), (65, 68, 72), (59, 62, 66), (60, 63, 67),
        (61, 66, 68), (62, 64, 69), (63, 67, 70), (64, 69, 71),
        (60, 62, 67), (61, 65, 68), (62, 67, 69), (63, 65, 70),
    ]
    repeated = phrase * 4
    rng = np.random.default_rng(20260907)
    qualities = [(0, 4, 7), (0, 3, 7), (0, 2, 7), (0, 3, 6), (0, 5, 7)]
    unrelated: list[tuple[int, ...]] = []
    for _ in range(64):
        root = int(rng.integers(48, 60))
        intervals = qualities[int(rng.integers(0, len(qualities)))]
        unrelated.append(tuple(root + interval for interval in intervals))

    repeated_features = extract_musical_features(_render_chord_sequence(repeated), SR)
    unrelated_features = extract_musical_features(_render_chord_sequence(unrelated), SR)

    assert repeated_features["M_status"] == "ok"
    assert unrelated_features["M_status"] == "ok"
    assert repeated_features["M_recurrence_peak_similarity"] > unrelated_features[
        "M_recurrence_peak_similarity"
    ] + 0.04
    assert repeated_features["M_recurrence_density"] > unrelated_features[
        "M_recurrence_density"
    ]


def test_duration_gate_keeps_30_second_m_values_missing_not_zero() -> None:
    audio = _render_chord_sequence([(60, 64, 67), (62, 65, 69)] * 15)
    features = extract_musical_features(audio, SR)

    assert features["H_status"] == "ok"
    assert features["M_status"] == "missing_short_duration"
    assert np.isnan(features["M_recurrence_peak_similarity"])
    assert features["M_eligible_block_count"] > 0


def test_silence_and_unpitched_noise_have_explicit_missing_statuses() -> None:
    silence = extract_musical_features(np.zeros(48 * SR, dtype=np.float32), SR)
    assert silence["H_status"] == "missing_low_energy"
    assert silence["M_status"] == "missing_low_energy"
    assert np.isnan(silence["H_pitch_class_entropy_norm"])

    rng = np.random.default_rng(7)
    noise = rng.normal(0.0, 0.03, 48 * SR).astype(np.float32)
    noisy = extract_musical_features(noise, SR)
    assert noisy["H_status"] == "missing_no_informative_pitch"
    assert noisy["M_status"] == "missing_no_informative_pitch"
    assert np.isnan(noisy["M_best_lag_sec"])


def test_function_is_generic_but_records_nonstandard_sample_rate() -> None:
    sr = 8_000
    audio = _render_chord_sequence([(60, 64, 67), (62, 65, 69)] * 6, sr=sr)
    features = extract_musical_features(audio, sr)
    assert features["H_status"] == "ok"
    assert features["H_input_sr_hz"] == sr
    assert features["H_standardized_sr_16000"] == 0
