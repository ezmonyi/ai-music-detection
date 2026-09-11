#!/usr/bin/env python3
"""Deterministic synthetic tests for phase_features.py (stdlib unittest)."""

from __future__ import annotations

import unittest

import numpy as np

from phase_features import FEATURE_NAMES, VERSION, extract_phase_features


SR = 16_000


def _tone(frequency: float = 1_000.0, seconds: float = 2.0, phase: float = 0.0) -> np.ndarray:
    time = np.arange(round(seconds * SR), dtype=np.float64) / SR
    return (0.2 * np.sin(2.0 * np.pi * frequency * time + phase)).astype(np.float64)


def _rich_stationary(seconds: float = 2.0) -> np.ndarray:
    time = np.arange(round(seconds * SR), dtype=np.float64) / SR
    frequencies = np.asarray([137.0, 311.0, 719.0, 1_423.0, 2_809.0, 4_937.0, 6_701.0])
    phases = np.asarray([0.2, -1.1, 2.0, 0.7, -2.4, 1.4, -0.3])
    audio = np.sum(
        np.sin(2.0 * np.pi * frequencies[:, None] * time + phases[:, None]), axis=0
    )
    return np.asarray(0.35 * audio / np.max(np.abs(audio)), dtype=np.float64)


class PhaseFeatureTests(unittest.TestCase):
    def test_silence_is_explicitly_unavailable_not_zero_filled(self) -> None:
        result = extract_phase_features(np.zeros(SR, dtype=np.float64), SR)
        self.assertEqual(result["F_status"], "low_energy")
        self.assertEqual(result["F_quality_eligible"], 0)
        self.assertTrue(all(np.isnan(float(result[name])) for name in FEATURE_NAMES))

    def test_bin_centred_tone_matches_hann_rotation_convention(self) -> None:
        # 1000 Hz lies exactly on a 15.625 Hz bin for the 1024-point/16 kHz STFT.
        # A periodic Hann window puts half-amplitude copies in the two adjacent
        # bins.  Their corrected increments are -pi/2 and +pi/2 while the
        # centre is zero, giving resultant length 0.5 and circular variance 0.5.
        result = extract_phase_features(_tone(), SR)
        value = float(result["F_phase_residual_cvar_all"])
        self.assertTrue(np.isfinite(value))
        self.assertGreaterEqual(value, 0.0)
        self.assertLessEqual(value, 1.0)
        self.assertAlmostEqual(value, 0.5, places=8)

    def test_global_initial_phase_does_not_create_a_false_residual(self) -> None:
        first = extract_phase_features(_tone(phase=0.0), SR)
        second = extract_phase_features(_tone(phase=1.234), SR)
        self.assertAlmostEqual(
            float(first["F_phase_residual_cvar_all"]),
            float(second["F_phase_residual_cvar_all"]),
            places=8,
        )

    def test_equal_global_magnitude_phase_surrogate_changes_phase_metrics(self) -> None:
        original = _rich_stationary()
        spectrum = np.fft.rfft(original)
        # Deterministic phase-only intervention; global Fourier magnitudes match.
        altered = np.fft.irfft(np.abs(spectrum), n=original.size).astype(np.float64)
        self.assertTrue(np.allclose(np.abs(np.fft.rfft(original)),
                                    np.abs(np.fft.rfft(altered)), rtol=1e-10, atol=1e-10))
        before = extract_phase_features(original, SR)
        after = extract_phase_features(altered, SR)
        candidates = (
            "F_phase_residual_cvar_all",
            "F_group_delay_iqr_ms_all",
            "F_group_delay_cross_band_iqr_ms_all",
        )
        differences = [
            abs(float(before[name]) - float(after[name]))
            for name in candidates
            if np.isfinite(float(before[name])) and np.isfinite(float(after[name]))
        ]
        self.assertTrue(differences)
        self.assertGreater(max(differences), 1e-3)

    def test_small_circular_time_shift_is_bounded_for_stationary_signal(self) -> None:
        original = extract_phase_features(_rich_stationary(), SR)
        shifted = extract_phase_features(np.roll(_rich_stationary(), 17), SR)
        for name in (
            "F_phase_residual_cvar_all",
            "F_group_delay_iqr_ms_all",
            "F_group_delay_cross_band_iqr_ms_all",
        ):
            self.assertTrue(np.isfinite(float(original[name])), name)
            self.assertTrue(np.isfinite(float(shifted[name])), name)
        self.assertLess(
            abs(float(original["F_phase_residual_cvar_all"])
                - float(shifted["F_phase_residual_cvar_all"])),
            0.02,
        )

    def test_subhop_time_shift_sensitivity_is_visible_for_transients(self) -> None:
        rng = np.random.default_rng(17)
        audio = np.zeros(3 * SR, dtype=np.float64)
        for start in (2_000, 9_000, 17_000, 27_000, 39_000):
            size = 1_000
            burst = rng.standard_normal(size) * np.exp(-np.arange(size) / 180.0)
            audio[start:start + size] += burst
        audio *= 0.1 / np.max(np.abs(audio))
        shifted = np.pad(audio, (73, 0))[:audio.size]
        first = extract_phase_features(audio, SR)
        second = extract_phase_features(shifted, SR)
        self.assertEqual(first["F_status"], "ok")
        self.assertEqual(second["F_status"], "ok")
        # The descriptor deliberately does not pretend sub-hop crop alignment
        # is invariant; this control quantifies a visible, finite difference.
        difference_ms = abs(float(first["F_group_delay_iqr_ms_all"])
                            - float(second["F_group_delay_iqr_ms_all"]))
        self.assertTrue(np.isfinite(difference_ms))
        self.assertGreater(difference_ms, 0.05)

    def test_finite_outputs_and_regions_on_broadband_enveloped_signal(self) -> None:
        rng = np.random.default_rng(20260907)
        seconds = 3.0
        sample_count = round(seconds * SR)
        noise = rng.standard_normal(sample_count)
        envelope = np.ones(sample_count, dtype=np.float64)
        ramp = round(0.6 * SR)
        envelope[:ramp] = np.linspace(0.01, 1.0, ramp)
        envelope[-ramp:] = np.linspace(1.0, 0.01, ramp)
        result = extract_phase_features(0.08 * noise * envelope, SR)
        self.assertEqual(result["F_status"], "ok")
        self.assertEqual(result["F_quality_eligible"], 1)
        for name in (
            "F_phase_residual_cvar_all",
            "F_group_delay_iqr_ms_all",
            "F_group_delay_cross_band_iqr_ms_all",
        ):
            self.assertTrue(np.isfinite(float(result[name])), name)
        self.assertGreater(int(result["F_attack_frame_count"]), 0)
        self.assertGreater(int(result["F_sustain_frame_count"]), 0)
        self.assertGreater(int(result["F_decay_frame_count"]), 0)
        self.assertTrue(VERSION.startswith("phase_features_v1"))

    def test_input_contract_rejects_ambiguous_or_invalid_arrays(self) -> None:
        with self.assertRaises(ValueError):
            extract_phase_features(np.zeros((10, 2), dtype=np.float64), SR)
        with self.assertRaises(TypeError):
            extract_phase_features(np.zeros(100, dtype=np.int16), SR)
        with self.assertRaises(ValueError):
            extract_phase_features(np.asarray([0.0, np.nan], dtype=np.float64), SR)
        with self.assertRaises(ValueError):
            extract_phase_features(np.zeros(100, dtype=np.float64), 0)


if __name__ == "__main__":
    unittest.main()
