import math
import unittest

import numpy as np

from vocalset_vibrato_features import PRIMARY_FEATURE, extract_vibrato_presence


def contour(duration=4.0, base_hz=220.0, rate_hz=0.0, extent_cents=0.0, glide=0.0):
    times = np.arange(0.0, duration, 0.01)
    cents = glide * (times - np.mean(times))
    if rate_hz:
        cents += extent_cents * np.sin(2.0 * np.pi * rate_hz * times)
    f0 = base_hz * np.power(2.0, cents / 1200.0)
    return times, f0, np.ones(len(times), dtype=bool)


class VocalSetVibratoFeaturesTest(unittest.TestCase):
    def test_measured_straight_tone_is_zero_not_missing(self):
        result = extract_vibrato_presence(*contour())
        self.assertEqual(result["V_status"], "ok")
        self.assertEqual(result[PRIMARY_FEATURE], 0.0)
        self.assertTrue(math.isnan(result["V_detected_rate_hz_median_conditional"]))

    def test_known_vibrato_is_detected_in_pitch_not_spectrum(self):
        result = extract_vibrato_presence(*contour(rate_hz=6.0, extent_cents=30.0))
        self.assertGreater(result[PRIMARY_FEATURE], 0.9)
        self.assertAlmostEqual(result["V_detected_rate_hz_median_conditional"], 6.0, places=6)
        self.assertAlmostEqual(result["V_detected_extent_cents_median_conditional"], 30.0, places=5)

    def test_below_threshold_modulation_orders_below_known_vibrato(self):
        low = extract_vibrato_presence(*contour(rate_hz=6.0, extent_cents=5.0))
        high = extract_vibrato_presence(*contour(rate_hz=6.0, extent_cents=20.0))
        self.assertEqual(low[PRIMARY_FEATURE], 0.0)
        self.assertGreater(high[PRIMARY_FEATURE], low[PRIMARY_FEATURE])

    def test_linear_glide_is_detrended(self):
        result = extract_vibrato_presence(
            *contour(rate_hz=5.0, extent_cents=25.0, glide=40.0)
        )
        self.assertGreater(result[PRIMARY_FEATURE], 0.9)
        self.assertAlmostEqual(result["V_detected_rate_hz_median_conditional"], 5.0, places=6)

    def test_no_informative_pitch_is_missing_not_zero(self):
        times = np.arange(0.0, 4.0, 0.01)
        result = extract_vibrato_presence(times, np.full(len(times), np.nan), np.zeros(len(times), bool))
        self.assertEqual(result["V_status"], "missing")
        self.assertTrue(math.isnan(result[PRIMARY_FEATURE]))
        self.assertEqual(result["V_eligible_stable_pitch_window_count"], 0)

    def test_octave_jump_windows_are_rejected_but_other_windows_remain(self):
        times, f0, voiced = contour(duration=5.0, rate_hz=6.0, extent_cents=25.0)
        f0[len(f0) // 2 :] *= 2.0
        result = extract_vibrato_presence(times, f0, voiced)
        self.assertGreater(result["V_transition_rejected_window_count"], 0)
        self.assertEqual(result["V_status"], "ok")
        self.assertGreater(result[PRIMARY_FEATURE], 0.0)

    def test_windows_never_cross_unvoiced_gap(self):
        times, f0, voiced = contour(duration=4.0, rate_hz=6.0, extent_cents=25.0)
        voiced[175:225] = False
        f0[175:225] = np.nan
        result = extract_vibrato_presence(times, f0, voiced)
        for window in result["V_window_details"]:
            start = window["start_frame"]
            self.assertFalse(start < 225 and start + 100 > 175)

    def test_final_run_anchored_window_is_counted_once(self):
        times, f0, voiced = contour(duration=2.2)
        result = extract_vibrato_presence(times, f0, voiced)
        starts = [window["start_frame"] for window in result["V_window_details"]]
        self.assertEqual(starts, [0, 50, 100, 120])
        self.assertEqual(len(starts), len(set(starts)))


if __name__ == "__main__":
    unittest.main()
