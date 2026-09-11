"""Independent coefficient fixtures; no imports from committed producers."""
import io
import json
import math
import unittest

import numpy as np

from bicoherence_primitive_v2 import estimate


def coefficients(phase, amplitude=None):
    phase = np.asarray(phase)
    data = np.zeros((len(phase), 17), dtype=np.complex128)
    data[:, [4, 7, 11]] = np.exp(1j * phase)
    if amplitude is not None:
        data[:, [4, 7, 11]] *= amplitude
    return data


def closed(k=128):
    rng = np.random.default_rng(20260907)
    phase = rng.uniform(-np.pi, np.pi, (k, 2))
    return coefficients(np.column_stack((phase, phase.sum(axis=1))))


def raw_value(encoded):
    return math.ldexp(encoded["mantissa"], encoded["exponent2"])


class TestBicoherencePrimitive(unittest.TestCase):
    def test_closed_and_random_phases(self):
        self.assertAlmostEqual(estimate(closed())["squared_bicoherence"], 1)
        rng = np.random.default_rng(11)
        random = coefficients(rng.uniform(-np.pi, np.pi, (128, 3)))
        self.assertLess(estimate(random)["squared_bicoherence"], .1)
        self.assertEqual(estimate(random)["biphase_status"], "descriptive_only_no_significance")

    def test_amplitude_counterexample_and_raw_sums(self):
        data = coefficients(np.zeros((128, 3)))
        data[64:, 11] = 10
        result = estimate(data)
        self.assertAlmostEqual(result["squared_bicoherence"], 121 / 202)
        self.assertEqual(result["biphase_radians"], 0)
        expected = {"product_energy_sum": 128, "sum_frequency_energy_sum": 6464,
                    "triple_sum_real": 704, "triple_sum_imag": 0}
        for name, value in expected.items():
            self.assertAlmostEqual(raw_value(result["raw_sums"][name]), value)

    def test_static_linear_triad(self):
        result = estimate(coefficients(np.tile([.2, .7, -1.1], (128, 1))))
        self.assertAlmostEqual(result["squared_bicoherence"], 1)
        self.assertAlmostEqual(result["biphase_radians"], 2)

    def test_exact_cancellation_has_undefined_phase(self):
        data = coefficients(np.zeros((2, 3)))
        data[1, 11] = -1
        result = estimate(data)
        self.assertEqual(result["squared_bicoherence"], 0)
        self.assertIsNone(result["biphase_radians"])
        self.assertEqual(result["biphase_status"], "undefined_zero_resultant")
        self.assertEqual(result["status"], "ok")

    def test_small_nonzero_resultant_is_descriptive(self):
        data = coefficients(np.zeros((2, 3)))
        data[1, 11] = -1 + 1e-10j
        result = estimate(data)
        self.assertGreater(result["squared_bicoherence"], 0)
        self.assertLess(result["squared_bicoherence"], 1e-18)
        self.assertIsNotNone(result["biphase_radians"])
        self.assertEqual(result["biphase_status"], "descriptive_only_no_significance")

    def test_gain_polarity_and_delay(self):
        data = closed()
        baseline = estimate(data)
        for gain in (.1, 20, -1):
            result = estimate(data * gain)
            self.assertAlmostEqual(result["squared_bicoherence"], baseline["squared_bicoherence"])
            expected_phase = np.pi if gain < 0 else 0
            difference = np.angle(np.exp(1j * (result["biphase_radians"] - baseline["biphase_radians"] - expected_phase)))
            self.assertAlmostEqual(difference, 0)
        delay = np.exp(-2j * np.pi * np.arange(17) * 5 / 32)
        shifted = estimate(data * delay)
        self.assertAlmostEqual(shifted["squared_bicoherence"], baseline["squared_bicoherence"])
        self.assertAlmostEqual(shifted["biphase_radians"], baseline["biphase_radians"])

    def test_ideal_lti_invariance_and_phase_shift(self):
        data = closed()
        transfer = np.ones(17, dtype=complex)
        transfer[[4, 7, 11]] = np.array([.25, 3, 12]) * np.exp(1j * np.array([.4, -.8, .7]))
        baseline, filtered = estimate(data), estimate(data * transfer)
        self.assertAlmostEqual(filtered["squared_bicoherence"], baseline["squared_bicoherence"])
        difference = np.angle(np.exp(1j * (filtered["biphase_radians"] - baseline["biphase_radians"] + 1.1)))
        self.assertAlmostEqual(difference, 0)

    def test_irrelevant_extreme_bin_does_not_change_triad(self):
        data = closed() * 1e-100
        baseline = estimate(data)
        data[:, 15] = 1e300 + 1e300j
        result = estimate(data)
        self.assertEqual(result, baseline)

    def test_huge_and_tiny_triad_scales_keep_finite_encoded_sums(self):
        data = coefficients(np.zeros((2, 3)))
        for gain in (1e300, 1e-300, np.finfo(float).max, np.nextafter(0., 1.)):
            result = estimate(data * gain)
            self.assertAlmostEqual(result["squared_bicoherence"], 1)
            self.assertNotEqual(result["raw_sums"]["product_energy_sum"]["mantissa"], 0)
            json.dumps(result, allow_nan=False)

    def test_missing_energy_and_disjoint_support(self):
        data = coefficients(np.zeros((2, 3)))
        self.assertEqual(estimate(data * 0)["status"], "zero_energy")
        data[:, 11] = 0
        result = estimate(data)
        self.assertEqual(result["status"], "missing_triad_energy")
        self.assertIsNone(result["squared_bicoherence"])
        data[:, 11] = 1
        data[0, 4], data[1, 7] = 0, 0
        result = estimate(data)
        self.assertEqual(result["status"], "missing_triad_product_energy")
        self.assertIsNone(result["squared_bicoherence"])
        self.assertEqual(result["normalized_sums"]["product_energy_sum"], 0)

    def test_wrong_input_types_shape_and_nonfinite(self):
        for data in (np.ones((2, 17), dtype=bool), [["1"]] * 2, np.ones((2, 17), dtype=object)):
            with self.assertRaises(TypeError):
                estimate(data)
        for data in (np.ones(17), np.ones((1, 17)), np.ones((2, 2, 17)), np.empty((2, 0))):
            with self.assertRaises(ValueError):
                estimate(data)
        for value in (np.nan, np.inf, -np.inf, complex(1, np.nan)):
            data = closed()
            data[0, 0] = value
            with self.assertRaises(ValueError):
                estimate(data)

    def test_invalid_pair_types_and_bounds(self):
        data = closed()
        for pair in ((True, 7), (4., 7), (4, "7"), (4,), (4, 7, 11), "47", None):
            with self.assertRaises(TypeError):
                estimate(data, pair)
        for pair in ((0, 7), (-1, 7), (7, 4), (7, 10), (4, 13)):
            with self.assertRaises(ValueError):
                estimate(data, pair)
        self.assertEqual(estimate(data, (np.int64(4), np.int64(7)))["status"], "ok")

    def test_diagonal_bifrequency(self):
        data = np.zeros((2, 17), complex)
        data[:, 4] = [1, 1j]
        data[:, 8] = [1, -1]
        self.assertAlmostEqual(estimate(data, (4, 4))["squared_bicoherence"], 1)

    def test_underflow_normalization_is_rejected(self):
        data = coefficients(np.zeros((2, 3)))
        data[:, 4] = [1e300, 1e-300]
        with self.assertRaisesRegex(FloatingPointError, "unsupported_underflow"):
            estimate(data)

    def test_underflow_product_is_rejected(self):
        data = coefficients(np.zeros((3, 3)))
        data[:, 4] = [1, 1e-200, 1e-200]
        data[:, 7] = [1e-200, 1, 1e-200]
        with self.assertRaisesRegex(FloatingPointError, "unsupported_underflow"):
            estimate(data)

    def test_underflow_energy_is_rejected(self):
        data = coefficients(np.zeros((2, 3)))
        data[:, 4] = [1, 1e-200]
        data[:, 7] = [1e-200, 1]
        with self.assertRaisesRegex(FloatingPointError, "unsupported_underflow"):
            estimate(data)

    def test_raw_sums_match_direct_independent_calculation(self):
        rng = np.random.default_rng(51)
        data = rng.normal(size=(29, 17)) + 1j * rng.normal(size=(29, 17))
        result = estimate(data)
        u, v = data[:, 4] * data[:, 7], data[:, 11]
        triple = sum(u * v.conj())
        direct = {"product_energy_sum": sum(abs(u) ** 2),
                  "sum_frequency_energy_sum": sum(abs(v) ** 2),
                  "triple_sum_real": triple.real, "triple_sum_imag": triple.imag}
        for name, expected in direct.items():
            self.assertAlmostEqual(raw_value(result["raw_sums"][name]), expected, places=11)
        expected = abs(triple) ** 2 / (direct["product_energy_sum"] * direct["sum_frequency_energy_sum"])
        self.assertAlmostEqual(result["squared_bicoherence"], expected, places=14)

    def test_underflow_nonzero_squared_coherence_is_rejected(self):
        data = coefficients(np.zeros((2, 3)))
        data[1, 11] = -1 + 1e-200j
        with self.assertRaisesRegex(FloatingPointError, "unsupported_underflow: nonzero squared coherence"):
            estimate(data)

    def test_npz_roundtrip_and_json_safe_receipt(self):
        data = closed()
        buffer = io.BytesIO()
        np.savez(buffer, coefficients=data)
        buffer.seek(0)
        with np.load(buffer, allow_pickle=False) as saved:
            np.testing.assert_array_equal(data, saved["coefficients"])
            self.assertEqual(estimate(data), estimate(saved["coefficients"]))
        result = estimate(data)
        self.assertEqual(json.loads(json.dumps(result, allow_nan=False)), result)
        self.assertFalse(result["external_validation_passed"])
        self.assertFalse(result["classifier_admitted"])


if __name__ == "__main__":
    unittest.main()
