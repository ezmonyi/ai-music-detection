#!/usr/bin/env python3
"""Deterministic synthetic tests only; no datasets, inference, or classifier."""
import json
import unittest

import numpy as np

import modulation_candidate_v1 as candidate

SR = candidate.SAMPLE_RATE
CENTERS = (625.0, 2000.0, 5000.0)


def am(carrier, modulation=24.0, depth=0.5, seconds=6.5):
    t = np.arange(round(seconds * SR), dtype=np.float64) / SR
    return (1.0 + depth * np.cos(2.0 * np.pi * modulation * t)) * np.cos(
        2.0 * np.pi * carrier * t)


def extract(x, **kwargs):
    return candidate.extract_modulation_candidate(x, SR, **kwargs)


class ModulationCandidateTests(unittest.TestCase):
    def test_pure_carriers_are_missing_in_occupied_band(self):
        for i, center in enumerate(CENTERS):
            with self.subTest(center=center):
                r = extract(am(center, depth=0))
                for w in r["bands"][i]["windows"]:
                    self.assertEqual(w["status"], "no_modulation")
                    self.assertIsNone(w["fast_fraction"])
                    self.assertIsNone(w["entropy"])
                    self.assertLess(w["total_modulation_power"], 1e-8)

    def test_am_peak_recovery_and_fast_fraction_in_every_band(self):
        for i, center in enumerate(CENTERS):
            low, high = candidate.BANDS_HZ[i]
            for modulation in (8.0, 24.0, 48.0):
                with self.subTest(center=center, modulation=modulation):
                    # Analytic reference requires both AM sidebands in band.
                    self.assertGreater(center - modulation, low)
                    self.assertLess(center + modulation, high)
                    r = extract(am(center, modulation))
                    for w in r["bands"][i]["windows"]:
                        self.assertEqual(w["status"], "ok")
                        peak = r["frequency_hz"][np.argmax(w["modulation_power"])]
                        self.assertEqual(peak, modulation)
                        self.assertAlmostEqual(w["fast_fraction"],
                                               float(modulation >= 16), places=6)
                        # Band filtering is not flat; compare within 0.2%.
                        self.assertAlmostEqual(w["total_modulation_power"], .125, delta=.00025)

    def test_modulation_depth_monotonically_increases_total_power(self):
        for i, center in enumerate(CENTERS):
            powers = [extract(am(center, depth=d))["bands"][i]["windows"][1]
                      ["total_modulation_power"] for d in (.1, .3, .6, .9)]
            self.assertTrue(np.all(np.diff(powers) > 0))
            np.testing.assert_allclose(np.array(powers) / powers[0], [1, 9, 36, 81], rtol=1e-6)

    def test_gain_polarity_invariance_and_energy_scaling(self):
        x = sum(am(c) for c in CENTERS)
        baseline = extract(x)
        for gain in (1e-100, .003, -1.0, -71.0, 1e100):
            with self.subTest(gain=gain):
                result = extract(gain * x)
                self.assertEqual(result["status"], baseline["status"])
                np.testing.assert_allclose(list(result["features"].values()),
                                           list(baseline["features"].values()), atol=1e-12)
                for b, ref in zip(result["bands"], baseline["bands"]):
                    for w, rw in zip(b["windows"], ref["windows"]):
                        np.testing.assert_allclose(w["modulation_power"], rw["modulation_power"], atol=1e-12)
                        self.assertAlmostEqual(w["band_mean_square"] / gain ** 2,
                                               rw["band_mean_square"], places=10)

    def test_silence_and_constant_input_have_no_arbitrary_entropy(self):
        for x in (np.zeros(104000), np.ones(104000)):
            result = extract(x)
            self.assertEqual(result["status"], "no_valid_band_windows")
            self.assertTrue(all(value is None for value in result["features"].values()))
        silence = extract(np.zeros(104000))
        self.assertEqual(silence["bands"][0]["windows"][0]["status"], "silence")

    def test_constant_envelope_has_zero_power(self):
        power, mean = candidate._spectrum(np.full(1000, 7.0))
        self.assertEqual(mean, 7.0)
        np.testing.assert_array_equal(power, 0)

    def test_input_validation_and_overflow_policy(self):
        cases = [(np.zeros(0), "empty_input"), (np.zeros(39999), "too_short"),
                 (np.zeros((40000, 2)), "not_mono"),
                 (np.zeros(40000, dtype=np.int16), "not_floating_point"),
                 (np.zeros(40000, dtype=complex), "not_floating_point"),
                 (np.full(40000, np.nan), "nonfinite_input"),
                 (np.full(40000, np.inf), "nonfinite_input"),
                 (np.full(40000, -np.inf), "nonfinite_input"),
                 (np.full(40000, np.finfo(float).max), "input_magnitude_too_large")]
        for x, status in cases:
            with self.subTest(status=status):
                result = extract(x)
                self.assertEqual(result["status"], status)
                self.assertTrue(all(v is None for v in result["features"].values()))
                json.dumps(result, allow_nan=False)
        self.assertEqual(candidate.extract_modulation_candidate(am(2000), 44100)
                         ["status"], "wrong_sample_rate")

    def test_band_mask_and_mask_validation(self):
        x = sum(am(c) for c in CENTERS)
        baseline, masked = extract(x), extract(x, band_mask=[True, False, True])
        self.assertEqual(masked["bands"][1]["status"], "band_masked")
        self.assertIsNone(masked["bands"][1]["windows"][0]["modulation_power"])
        for name, value in masked["features"].items():
            if "1000_3000hz" in name:
                self.assertIsNone(value)
            else:
                self.assertEqual(value, baseline["features"][name])
        self.assertEqual(extract(x, band_mask=[1, 0, 1])["status"], "invalid_band_mask")
        self.assertEqual(extract(x, band_mask=[True])["status"], "invalid_band_mask")
        self.assertEqual(extract(x, band_mask=[False] * 3)["status"], "no_valid_band_windows")

    def test_antialias_rejects_400_hz_modulation_before_500_hz_sampling(self):
        # Both sidebands (1600, 2400 Hz) are inside this carrier band. An
        # unfiltered decimator would produce a false 100 Hz modulation peak.
        reference = extract(am(2000, 24))["bands"][1]["windows"][1]
        high = extract(am(2000, 400))["bands"][1]["windows"][1]
        self.assertEqual(high["status"], "no_modulation")
        self.assertLess(high["total_modulation_power"] / reference["total_modulation_power"], 1e-8)
        self.assertIsNone(high["entropy"])

    def test_segment_count_boundaries_and_no_padding(self):
        for length, count in ((39999, 0), (40000, 1), (71999, 1), (72000, 2),
                              (104000, 3), (104031, 3)):
            with self.subTest(length=length):
                r = extract(np.zeros(length))
                self.assertEqual(r["window_count"], count)
                if count:
                    windows = r["bands"][0]["windows"]
                    self.assertEqual(len(windows), count)
                    self.assertEqual(windows[0]["start_seconds"], .25)
                    self.assertEqual(windows[-1]["end_seconds"], .25 + 2 * count)
                    self.assertLessEqual(windows[-1]["end_seconds"], length / SR - .25)

    def test_one_sided_power_normalization_and_entropy_hann_reference(self):
        t = np.arange(1000) / 500
        envelope = 1 + .5 * np.cos(2 * np.pi * 24 * t)
        power, _ = candidate._spectrum(envelope)
        self.assertAlmostEqual(np.sum(power), .125, places=12)
        p = power[candidate._ANALYSIS] / np.sum(power[candidate._ANALYSIS])
        positive = p[p > 0]
        h = -np.sum(positive * np.log(positive)) / np.log(252)
        expected = -np.sum(np.array([1 / 6, 2 / 3, 1 / 6]) *
                           np.log([1 / 6, 2 / 3, 1 / 6])) / np.log(252)
        self.assertAlmostEqual(h, expected, places=12)

    def test_finite_bounded_json_and_six_scalar_summary_medians(self):
        rng = np.random.default_rng(7092026)
        x = sum(am(c) for c in CENTERS) + .02 * rng.standard_normal(104000)
        result = extract(x)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(len(result["features"]), 6)
        self.assertEqual(len(result["frequency_hz"]), 501)
        self.assertEqual(result["frequency_hz"][-1], 250)
        json.dumps(result, allow_nan=False)
        for band in result["bands"]:
            for metric in ("fast_fraction", "entropy"):
                value = result["features"][f"Q_{band['name']}_{metric}_median"]
                self.assertGreaterEqual(value, 0)
                self.assertLessEqual(value, 1)
                self.assertEqual(value, np.median([w[metric] for w in band["windows"]]))
            for w in band["windows"]:
                self.assertTrue(np.all(np.asarray(w["modulation_power"]) >= 0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
