#!/usr/bin/env python3
"""Synthetic tests for conservative V pitch-microstructure features."""

from __future__ import annotations

import unittest

import numpy as np

import pitch_microstructure_features as micro


class KnownModulationTests(unittest.TestCase):
    def test_known_4_6_8hz_and_20_50cent_amplitudes(self) -> None:
        for rate in (4.0, 6.0, 8.0):
            for extent in (20.0, 50.0):
                with self.subTest(rate=rate, extent=extent):
                    times, f0, voiced = micro._synthetic_contour(rate, extent)
                    result = micro.extract_pitch_microstructure(times, f0, voiced)
                    self.assertAlmostEqual(result["features"][micro.FEATURE_NAMES[0]], rate, delta=0.10)
                    self.assertAlmostEqual(result["features"][micro.FEATURE_NAMES[1]], extent, delta=1.0)
                    self.assertGreaterEqual(result["features"][micro.FEATURE_NAMES[2]], 0.98)

    def test_straight_tone_is_nonobservable_not_zero_rate(self) -> None:
        times, _, voiced = micro._synthetic_contour(6.0, 20.0)
        result = micro.extract_pitch_microstructure(times, np.full(len(times), 220.0), voiced)
        self.assertTrue(np.isnan(result["features"][micro.FEATURE_NAMES[0]]))
        self.assertEqual(result["diagnostics"]["observable_window_count"], 0)
        self.assertGreater(result["diagnostics"]["rejected_low_variation_window_count"], 0)

    def test_linear_glide_is_detrended_before_modulation_fit(self) -> None:
        times, f0, voiced = micro._synthetic_contour(
            6.0, 30.0, slope_cents_per_sec=400.0
        )
        result = micro.extract_pitch_microstructure(times, f0, voiced)
        self.assertAlmostEqual(result["features"][micro.FEATURE_NAMES[0]], 6.0, delta=0.10)
        self.assertAlmostEqual(result["features"][micro.FEATURE_NAMES[1]], 30.0, delta=1.0)


class GapAndOctaveTests(unittest.TestCase):
    def test_unvoiced_gap_splits_runs_and_no_window_crosses_it(self) -> None:
        times, f0, voiced = micro._synthetic_contour(6.0, 30.0)
        voiced[60:80] = False
        f0[~voiced] = np.nan
        result = micro.extract_pitch_microstructure(times, f0, voiced)
        self.assertEqual(result["diagnostics"]["voiced_run_count"], 2)
        self.assertTrue(result["windows"])
        self.assertFalse(
            any(
                not (w["end_frame_exclusive"] <= 60 or w["start_frame"] >= 80)
                for w in result["windows"]
            )
        )

    def test_constant_octave_offset_does_not_change_modulation(self) -> None:
        times, f0, voiced = micro._synthetic_contour(6.0, 30.0)
        base = micro.extract_pitch_microstructure(times, f0, voiced)
        octave = micro.extract_pitch_microstructure(times, f0 * 2.0, voiced)
        for name in micro.FEATURE_NAMES:
            self.assertAlmostEqual(base["features"][name], octave["features"][name], delta=1e-8)

    def test_internal_octave_jump_is_transition_rejected(self) -> None:
        times, f0, voiced = micro._synthetic_contour(6.0, 30.0)
        f0[75:] *= 2.0
        result = micro.extract_pitch_microstructure(times, f0, voiced)
        self.assertGreaterEqual(result["diagnostics"]["rejected_transition_window_count"], 1)


class FrozenContractTests(unittest.TestCase):
    def test_feature_registry_excludes_observability_diagnostics(self) -> None:
        self.assertEqual(len(micro.FEATURE_NAMES), 3)
        self.assertTrue(all("observable" not in name for name in micro.FEATURE_NAMES))
        self.assertIn("observable_window_fraction", micro.DIAGNOSTIC_NAMES)

    def test_full_synthetic_validation_passes(self) -> None:
        validation = micro.run_synthetic_validation()
        self.assertEqual(validation["case_count"], 11)
        self.assertEqual(validation["passed_case_count"], 11)
        self.assertTrue(validation["all_passed"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
