#!/usr/bin/env python3
"""Deterministic mathematical tests for stereo_candidate_v1."""

from __future__ import annotations

import unittest
import warnings

import numpy as np

import stereo_candidate_v1 as candidate
from stereo_candidate_v1 import BANDS_HZ, FEATURE_NAMES, SAMPLE_RATE, extract_stereo_candidate


SR = SAMPLE_RATE


def tone(seconds=2.0, frequency=1_000.0, phase=0.0, amplitude=0.2):
    t = np.arange(round(seconds * SR), dtype=np.float64) / SR
    return amplitude * np.sin(2.0 * np.pi * frequency * t + phase)


def stereo(left, right):
    return np.column_stack([left, right]).astype(np.float64)


def feature(result, band, metric, summary="median"):
    descriptor = ("abs_iid_db" if metric == "iid_db" else
                  "abs_ipd_increment_rad" if metric == "ipd_increment_rad" else metric)
    return float(result["features"][f"SC_{band}_{descriptor}_{summary}"])


class StereoCandidateTests(unittest.TestCase):
    def test_duplicate_mono_relationships_are_explicit_not_native_provenance(self):
        x = tone()
        result = extract_stereo_candidate(stereo(x, x), SR, native_channels=2)
        # A single tone does not energize every fixed band, so the excerpt-level
        # status is partial while its occupied band is fully valid.
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["bands"][BANDS_HZ.index((500, 2_000))]["status"], "ok")
        for low, high in BANDS_HZ:
            band = f"{low}_{high}hz"
            # Only the occupied 500--2000 Hz band has enough valid frames.
            if (low, high) == (500, 2_000):
                self.assertAlmostEqual(feature(result, band, "iid_db"), 0.0, places=10)
                self.assertAlmostEqual(feature(result, band, "signed_real_coherence"), 1.0, places=10)
                self.assertAlmostEqual(feature(result, band, "magnitude_coherence"), 1.0, places=10)
                self.assertAlmostEqual(feature(result, band, "side_energy_fraction"), 0.0, places=10)
                self.assertAlmostEqual(feature(result, band, "ipd_increment_rad"), 0.0, places=10)
        with self.assertRaisesRegex(ValueError, "native_channels"):
            extract_stereo_candidate(stereo(x, x), SR, native_channels=1)

    def test_antiphase_separates_signed_and_magnitude_coherence(self):
        x = tone()
        result = extract_stereo_candidate(stereo(x, -x), SR, native_channels=2)
        band = "500_2000hz"
        self.assertAlmostEqual(feature(result, band, "iid_db"), 0.0, places=10)
        self.assertAlmostEqual(feature(result, band, "signed_real_coherence"), -1.0, places=10)
        self.assertAlmostEqual(feature(result, band, "magnitude_coherence"), 1.0, places=10)
        self.assertAlmostEqual(feature(result, band, "side_energy_fraction"), 1.0, places=10)

    def test_known_gain_imbalance_and_common_gain_invariance(self):
        x = tone(amplitude=0.1)
        pair = stereo(x, 0.5 * x)
        first = extract_stereo_candidate(pair, SR, native_channels=2)
        second = extract_stereo_candidate(-3.0 * pair, SR, native_channels=2)
        self.assertAlmostEqual(feature(first, "500_2000hz", "iid_db"), 10.0 * np.log10(4.0), places=9)
        for metric in ("iid_db", "signed_real_coherence", "magnitude_coherence",
                       "side_energy_fraction", "ipd_increment_rad"):
            a = first["per_frame"][metric]
            b = second["per_frame"][metric]
            np.testing.assert_allclose(a, b, atol=2e-12, rtol=2e-12, equal_nan=True)

    def test_high_finite_common_gain_has_no_overflow_or_ratio_drift(self):
        rng = np.random.default_rng(91827)
        left = rng.standard_normal(2 * SR)
        independent = rng.standard_normal(2 * SR)
        right = 0.7 * left + 0.3 * independent
        pair = stereo(left, right)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            normal = extract_stereo_candidate(pair, SR, native_channels=2)
            high = extract_stereo_candidate(pair * 1e100, SR, native_channels=2)
        for mask in normal["valid_masks"]:
            np.testing.assert_array_equal(normal["valid_masks"][mask], high["valid_masks"][mask])
        for metric in normal["per_frame"]:
            a = normal["per_frame"][metric]
            b = high["per_frame"][metric]
            np.testing.assert_allclose(a, b, atol=2e-10, rtol=2e-10, equal_nan=True)
            self.assertTrue(np.all(np.isfinite(b[np.isfinite(a)])))

    def test_channel_swap_changes_iid_and_ipd_sign_only(self):
        t = np.arange(2 * SR, dtype=np.float64) / SR
        left = 0.2 * np.sin(2 * np.pi * 1_000 * t)
        right = 0.1 * np.sin(2 * np.pi * 1_000 * t + 0.7 * np.sin(2 * np.pi * 2.3 * t))
        original = extract_stereo_candidate(stereo(left, right), SR, native_channels=2)
        swapped = extract_stereo_candidate(stereo(right, left), SR, native_channels=2)
        band_index = BANDS_HZ.index((500, 2_000))
        np.testing.assert_allclose(original["per_frame"]["iid_db"][band_index],
                                   -swapped["per_frame"]["iid_db"][band_index],
                                   atol=1e-10, equal_nan=True)
        np.testing.assert_allclose(original["per_frame"]["ipd_increment_rad"][band_index],
                                   -swapped["per_frame"]["ipd_increment_rad"][band_index],
                                   atol=1e-10, equal_nan=True)
        for metric in ("signed_real_coherence", "magnitude_coherence", "side_energy_fraction"):
            np.testing.assert_allclose(original["per_frame"][metric][band_index],
                                       swapped["per_frame"][metric][band_index],
                                       atol=1e-12, equal_nan=True)
        self.assertEqual(len(FEATURE_NAMES), 50)
        self.assertTrue(all("_iid_db_" not in name or "_abs_iid_db_" in name
                            for name in FEATURE_NAMES))
        self.assertTrue(all("_ipd_increment_rad_" not in name or
                            "_abs_ipd_increment_rad_" in name for name in FEATURE_NAMES))
        np.testing.assert_allclose(
            [original["features"][name] for name in FEATURE_NAMES],
            [swapped["features"][name] for name in FEATURE_NAMES],
            atol=1e-10, rtol=1e-10, equal_nan=True,
        )

    def test_absolute_ipd_summary_is_branch_cut_and_swap_invariant(self):
        # These signed shortest-arc increments straddle the +/-pi branch cut.
        increments = np.asarray([np.pi - 0.02, -np.pi + 0.01, np.pi - 0.03])
        median, iqr, count = candidate._descriptor_summary("ipd_increment_rad", increments)
        swapped_median, swapped_iqr, swapped_count = candidate._descriptor_summary(
            "ipd_increment_rad", -increments
        )
        self.assertEqual(count, 3)
        self.assertEqual(swapped_count, 3)
        self.assertAlmostEqual(median, np.pi - 0.02, places=12)
        self.assertLess(iqr, 0.02)
        self.assertAlmostEqual(median, swapped_median, places=12)
        self.assertAlmostEqual(iqr, swapped_iqr, places=12)

    def test_hard_pan_marks_two_sided_metrics_missing_but_retains_side_fraction(self):
        x = tone()
        result = extract_stereo_candidate(stereo(x, np.zeros_like(x)), SR, native_channels=2)
        band = "500_2000hz"
        self.assertTrue(np.isnan(feature(result, band, "iid_db")))
        self.assertTrue(np.isnan(feature(result, band, "signed_real_coherence")))
        self.assertTrue(np.isnan(feature(result, band, "magnitude_coherence")))
        self.assertTrue(np.isnan(feature(result, band, "ipd_increment_rad")))
        self.assertAlmostEqual(feature(result, band, "side_energy_fraction"), 0.5, places=10)
        quality = result["bands"][BANDS_HZ.index((500, 2_000))]
        self.assertEqual(quality["status"], "partial_missing_one_side")

    def test_silence_and_absolute_floor_are_missing_not_zero_filled(self):
        silence = np.zeros((2 * SR, 2), dtype=np.float64)
        result = extract_stereo_candidate(silence, SR, native_channels=2)
        self.assertEqual(result["status"], "low_energy")
        self.assertEqual(result["quality_eligible"], 0)
        self.assertTrue(all(np.isnan(float(result["features"][name])) for name in FEATURE_NAMES))
        below = extract_stereo_candidate(stereo(tone(amplitude=1e-7), tone(amplitude=1e-7)),
                                         SR, native_channels=2)
        self.assertEqual(below["status"], "low_energy")

    def test_one_or_two_energized_frames_are_not_mislabeled_low_energy(self):
        x = tone(seconds=(candidate.N_FFT + candidate.HOP_LENGTH) / SR)
        result = extract_stereo_candidate(stereo(x, x), SR, native_channels=2)
        self.assertEqual(result["frame_count"], 2)
        self.assertEqual(result["status"], "insufficient_valid_frames")
        self.assertEqual(result["quality_eligible"], 0)
        self.assertTrue(any(band["energy_valid_frames"] > 0 for band in result["bands"]))
        self.assertTrue(all(np.isnan(float(result["features"][name])) for name in FEATURE_NAMES))

    def test_delay_changes_no_reference_stereo_relationships(self):
        rng = np.random.default_rng(20260907)
        left = 0.05 * rng.standard_normal(2 * SR)
        delay = 44
        right = np.pad(left, (delay, 0))[:left.size]
        result = extract_stereo_candidate(stereo(left, right), SR, native_channels=2)
        duplicate = extract_stereo_candidate(stereo(left, left), SR, native_channels=2)
        changed = []
        for low, high in BANDS_HZ:
            band = f"{low}_{high}hz"
            a = feature(result, band, "magnitude_coherence")
            b = feature(duplicate, band, "magnitude_coherence")
            if np.isfinite(a) and np.isfinite(b):
                changed.append(b - a)
        self.assertTrue(changed)
        self.assertGreater(max(changed), 0.10)
        self.assertGreater(feature(result, "500_2000hz", "side_energy_fraction"), 0.01)

    def test_phase_jitter_has_nonzero_adjacent_ipd_increments(self):
        t = np.arange(3 * SR, dtype=np.float64) / SR
        left = 0.2 * np.sin(2 * np.pi * 1_000 * t)
        right = 0.2 * np.sin(2 * np.pi * 1_000 * t + 0.8 * np.sin(2 * np.pi * 2.0 * t))
        result = extract_stereo_candidate(stereo(left, right), SR, native_channels=2)
        self.assertGreater(feature(result, "500_2000hz", "ipd_increment_rad", "iqr"), 0.05)

    def test_equal_global_channel_magnitudes_can_have_different_band_coherence(self):
        t = np.arange(2 * SR, dtype=np.float64) / SR
        frequencies = np.asarray([600.0, 900.0, 1_200.0, 1_500.0])
        left = 0.04 * np.sum(np.sin(2 * np.pi * frequencies[:, None] * t), axis=0)
        phases = np.asarray([0.0, np.pi / 2.0, np.pi, -np.pi / 2.0])
        altered = 0.04 * np.sum(np.sin(2 * np.pi * frequencies[:, None] * t + phases[:, None]), axis=0)
        np.testing.assert_allclose(np.abs(np.fft.rfft(left)), np.abs(np.fft.rfft(altered)),
                                   atol=2e-9, rtol=2e-9)
        aligned = extract_stereo_candidate(stereo(left, left), SR, native_channels=2)
        different = extract_stereo_candidate(stereo(left, altered), SR, native_channels=2)
        self.assertGreater(feature(aligned, "500_2000hz", "magnitude_coherence") -
                           feature(different, "500_2000hz", "magnitude_coherence"), 0.5)
        quality = different["bands"][BANDS_HZ.index((500, 2_000))]
        self.assertGreater(quality["two_sided_valid_frames"], 0)
        self.assertEqual(quality["ipd_phase_valid_frames"], 0)
        self.assertEqual(quality["status"], "partial_low_cross_coherence_or_adjacency")
        self.assertTrue(np.isnan(feature(different, "500_2000hz", "ipd_increment_rad")))

    def test_cross_phase_mask_does_not_bridge_invalid_frame(self):
        x = tone(seconds=1.0)
        y = stereo(x, x)
        # A full zero region longer than one window creates invalid intervening frames.
        y[15_000:25_000] = 0.0
        result = extract_stereo_candidate(y, SR, native_channels=2)
        index = BANDS_HZ.index((500, 2_000))
        phase_valid = result["valid_masks"]["ipd_phase_valid"][index]
        increment_valid = result["valid_masks"]["ipd_increment_valid"][index]
        self.assertTrue(np.any(~phase_valid))
        self.assertTrue(np.all(~increment_valid[1:] |
                               (phase_valid[1:] & phase_valid[:-1])))

    def test_invalid_shape_dtype_nan_and_sample_rate(self):
        with self.assertRaises(ValueError):
            extract_stereo_candidate(np.zeros(100, dtype=np.float64), SR, native_channels=2)
        with self.assertRaises(ValueError):
            extract_stereo_candidate(np.zeros((100, 1), dtype=np.float64), SR, native_channels=2)
        with self.assertRaises(TypeError):
            extract_stereo_candidate(np.zeros((5_000, 2), dtype=np.int16), SR, native_channels=2)
        bad = np.zeros((5_000, 2), dtype=np.float64)
        bad[0, 0] = np.nan
        with self.assertRaises(ValueError):
            extract_stereo_candidate(bad, SR, native_channels=2)
        with self.assertRaises(ValueError):
            extract_stereo_candidate(np.zeros((5_000, 2), dtype=np.float64), 48_000,
                                     native_channels=2)
        overflowing = np.full((5_000, 2), 1e200, dtype=np.float64)
        with self.assertRaisesRegex(ValueError, "overflow"):
            extract_stereo_candidate(overflowing, SR, native_channels=2)


if __name__ == "__main__":
    unittest.main()
