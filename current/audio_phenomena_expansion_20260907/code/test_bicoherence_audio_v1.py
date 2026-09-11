"""Continuous synthetic development checks, not music admission thresholds."""
import json
import math
import unittest

import numpy as np

import bicoherence_audio_v1 as audio_bc
from bicoherence_primitive_v2 import estimate


def linear_triad(samples=64000, detuning_hz=0.0, gain=1.0):
    t = np.arange(samples, dtype=np.float64) / 16000
    return gain * (np.cos(2 * np.pi * 500 * t + 0.2)
                   + 0.8 * np.cos(2 * np.pi * 875 * t - 0.6)
                   + 0.6 * np.cos(2 * np.pi * (1375 + detuning_hz) * t + 0.1))


def varying_phase_triad(closure=True):
    """Smooth continuous phases; never concatenate independent coefficient rows."""
    t = np.arange(64000, dtype=np.float64) / 16000
    phase1 = 0.8 * np.sin(2 * np.pi * 0.31 * t)
    phase2 = 0.7 * np.sin(2 * np.pi * 0.47 * t + 0.2)
    phase3 = phase1 + phase2 - 0.4 if closure else 6 * np.sin(2 * np.pi * 1.1 * t + 0.7)
    return (np.cos(2 * np.pi * 500 * t + phase1)
            + 0.8 * np.cos(2 * np.pi * 875 * t + phase2)
            + 0.6 * np.cos(2 * np.pi * 1375 * t + phase3))


def target_cell(result):
    index = np.flatnonzero(np.all(result["arrays"]["frequency_bins"] == [32, 56, 88], axis=1))[0]
    return result["metadata"]["pools"][0]["cells"][index]


class AudioBicoherenceTests(unittest.TestCase):
    def test_transform_independent_loop_and_energy_convention(self):
        rng = np.random.default_rng(1207)
        signal = rng.normal(size=64000).astype(np.float64) + 0.7
        result = audio_bc.extract(signal, 16000)
        actual = result["arrays"]
        # Explicit scalar window definition and loop are independent of the
        # extractor's vectorized window/frame construction.
        window = np.asarray([0.5 - 0.5 * math.cos(2 * math.pi * n / 1024)
                             for n in range(1024)])
        expected = []
        means = []
        for start in range(0, 64000 - 1024 + 1, 256):
            frame = signal[start:start + 1024]
            mean = sum(float(x) for x in frame) / 1024
            means.append(mean)
            expected.append(np.fft.rfft((frame - mean) * window) / sum(window))
        expected = np.asarray(expected)
        self.assertEqual(expected.shape, (247, 513))
        np.testing.assert_allclose(actual["spectra"][0], expected, rtol=2e-12, atol=2e-15)
        np.testing.assert_allclose(actual["frame_means"][0], means, rtol=2e-14, atol=2e-15)
        powers = (expected.real**2 + expected.imag**2).sum(axis=0)
        np.testing.assert_allclose(actual["bin_coefficient_energy"][0], powers, rtol=2e-13)
        np.testing.assert_allclose(actual["bin_energy_fraction"][0], powers / sum(powers[1:]), rtol=2e-13)
        self.assertAlmostEqual(sum(actual["bin_energy_fraction"][0, 1:]), 1.0)
        self.assertEqual(actual["window"][0], 0)
        self.assertGreater(actual["window"][-1], 0)  # Periodic, not symmetric Hann.

    def test_grid_complete_and_boundaries(self):
        expected = [(a, b, a + b) for a in range(1, 257) for b in range(a, 257)
                    if a % 8 == b % 8 == 0 and 8 <= a <= 192 and 8 <= b <= 192 and a + b <= 256]
        self.assertEqual(audio_bc.pair_grid().tolist(), [list(x) for x in expected])
        grid = {tuple(x) for x in audio_bc.pair_grid()}
        self.assertIn((8, 8, 16), grid)
        self.assertIn((64, 192, 256), grid)
        self.assertIn((128, 128, 256), grid)
        self.assertNotIn((136, 136, 272), grid)

    def test_pool_boundaries_and_tail(self):
        signal = np.zeros(128123, np.float64)
        signal[64000:] = 2.0
        signal[128000:] = 50.0
        result = audio_bc.extract(signal, np.int64(16000))
        arrays, meta = result["arrays"], result["metadata"]
        self.assertEqual(meta["pool_count"], 2)
        self.assertEqual(meta["discarded_tail_samples"], 123)
        self.assertEqual(meta["analyzed_samples"], 128000)
        np.testing.assert_array_equal(arrays["pool_start_samples"], [0, 64000])
        self.assertEqual(arrays["frame_start_samples"][0, -1] + 1024, 64000)
        self.assertEqual(arrays["frame_start_samples"][1, 0], 64000)
        self.assertEqual(arrays["frame_start_samples"][1, -1] + 1024, 128000)
        self.assertFalse(np.any(arrays["spectra"]))  # No boundary-spanning steps.
        np.testing.assert_array_equal(arrays["frame_means"][0], 0)
        np.testing.assert_array_equal(arrays["frame_means"][1], 2)
        self.assertEqual(meta["pools"][0]["status"], "zero_amplitude")
        self.assertEqual(meta["pools"][1]["status"], "zero_non_dc_energy")

    def test_short_inputs_explicit_missing_not_zero(self):
        for n in [0, 1024, 63999]:
            with self.subTest(n=n):
                result = audio_bc.extract(np.zeros(n, np.float64), 16000)
                meta = result["metadata"]
                self.assertEqual(meta["status"], "insufficient_support")
                self.assertEqual(meta["discarded_tail_samples"], n)
                self.assertEqual(meta["pools"], [])
                self.assertEqual(result["arrays"]["spectra"].shape, (0, 247, 513))
                json.dumps(meta, allow_nan=False)

    def test_zero_amplitude_all_cells_preserved_and_missing(self):
        result = audio_bc.extract(np.zeros(64000, np.float64), 16000)
        pool = result["metadata"]["pools"][0]
        self.assertEqual(pool["status"], "zero_amplitude")
        self.assertEqual(len(pool["cells"]), len(audio_bc.pair_grid()))
        for cell in pool["cells"]:
            self.assertFalse(cell["eligible"])
            self.assertEqual(cell["status"], "zero_energy")
            self.assertIsNone(cell["squared_bicoherence"])
            self.assertEqual(cell["energy_fractions"], [None] * 3)
        self.assertTrue(np.isnan(result["arrays"]["squared_bicoherence"]).all())
        json.dumps(result["metadata"], allow_nan=False)

    def test_invalid_input_types_and_shapes(self):
        for signal in [[0.0] * 64000, np.zeros(64000, np.float32), np.zeros(64000, np.int64),
                       np.zeros(64000, complex), np.zeros(64000, bool), np.array(["0"]),
                       np.zeros(1, object)]:
            with self.subTest(dtype=type(signal)):
                with self.assertRaises(TypeError):
                    audio_bc.extract(signal, 16000)
        for signal in [np.zeros((64000, 1)), np.array(0.0), np.array([np.nan]), np.array([np.inf])]:
            with self.assertRaises(ValueError):
                audio_bc.extract(signal, 16000)

    def test_exact_integer_sample_rate(self):
        x = np.empty(0, np.float64)
        for fs in [True, np.bool_(True), 16000.0, "16000", None, np.array(16000)]:
            with self.assertRaises(TypeError):
                audio_bc.extract(x, fs)
        for fs in [0, 8000, 44100, -16000]:
            with self.assertRaises(ValueError):
                audio_bc.extract(x, fs)

    def test_low_energy_cells_retained_with_raw_primitive(self):
        x = linear_triad()
        t = np.arange(len(x), dtype=np.float64) / 16000
        x += 1000 * np.cos(2 * np.pi * 6000 * t)
        cell = target_cell(audio_bc.extract(x, 16000))
        self.assertEqual(cell["status"], "below_energy_fraction_floor")
        self.assertFalse(cell["eligible"])
        self.assertIsNone(cell["squared_bicoherence"])
        self.assertGreater(cell["primitive"]["squared_bicoherence"], 0.999999)
        self.assertIsNotNone(cell["primitive"]["raw_sums"])
        self.assertTrue(all(f < 1e-6 for f in cell["energy_fractions"]))

    def test_primitive_replay_all_cells_and_sufficient_sums(self):
        result = audio_bc.extract(linear_triad(), 16000)
        spectra = result["arrays"]["spectra"][0]
        for cell in result["metadata"]["pools"][0]["cells"]:
            f1, f2, _ = cell["frequency_bins"]
            self.assertEqual(cell["primitive"], estimate(spectra, (f1, f2)))
        cell = target_cell(result)
        # Independent direct coefficient formula, not just primitive replay.
        u, v = spectra[:, 32] * spectra[:, 56], spectra[:, 88]
        triple = np.sum(u * v.conj())
        expected = abs(triple)**2 / (np.sum(abs(u)**2) * np.sum(abs(v)**2))
        self.assertAlmostEqual(cell["squared_bicoherence"], expected, places=13)
        raw = cell["primitive"]["raw_sums"]
        decode = lambda key: math.ldexp(raw[key]["mantissa"], raw[key]["exponent2"])
        self.assertAlmostEqual(decode("product_energy_sum"), float(np.sum(abs(u)**2)), places=13)
        self.assertAlmostEqual(decode("sum_frequency_energy_sum"), float(np.sum(abs(v)**2)), places=12)
        self.assertAlmostEqual(decode("triple_sum_real"), triple.real, places=12)
        self.assertAlmostEqual(decode("triple_sum_imag"), triple.imag, places=12)

    def test_static_linear_triad_high_is_not_causality(self):
        cell = target_cell(audio_bc.extract(linear_triad(), 16000))
        self.assertTrue(cell["eligible"])
        self.assertGreater(cell["squared_bicoherence"], 1 - 1e-12)
        self.assertAlmostEqual(cell["biphase_radians"], -0.5, places=10)

    def test_non_sum_detuning_low_coherence(self):
        cell = target_cell(audio_bc.extract(linear_triad(detuning_hz=5.0), 16000))
        self.assertTrue(cell["eligible"])
        self.assertLess(cell["squared_bicoherence"], 0.01)

    def test_gain_and_polarity(self):
        baseline = target_cell(audio_bc.extract(linear_triad(), 16000))
        for gain in [0.1, 10.0, -1.0]:
            other = target_cell(audio_bc.extract(linear_triad(gain=gain), 16000))
            self.assertAlmostEqual(other["squared_bicoherence"], baseline["squared_bicoherence"], places=13)
            np.testing.assert_allclose(other["energy_fractions"], baseline["energy_fractions"], rtol=1e-13)
            shift = other["biphase_radians"] - baseline["biphase_radians"]
            self.assertAlmostEqual(math.cos(shift), -1 if gain < 0 else 1, places=12)

    def test_smooth_continuous_varying_phase_closure_control(self):
        closed = target_cell(audio_bc.extract(varying_phase_triad(True), 16000))
        unrelated = target_cell(audio_bc.extract(varying_phase_triad(False), 16000))
        self.assertTrue(closed["eligible"] and unrelated["eligible"])
        self.assertGreater(closed["squared_bicoherence"], 0.95)
        self.assertLess(unrelated["squared_bicoherence"], 0.2)
        # These inequalities describe this deterministic fixture only.

    def test_unsupported_numeric_range_raises(self):
        subnormal_impulse = np.zeros(64000, np.float64)
        subnormal_impulse[1] = np.nextafter(0.0, 1.0)
        for x in [linear_triad(gain=1e-200), linear_triad(gain=1e200),
                  np.full(64000, np.finfo(np.float64).max), subnormal_impulse]:
            with self.assertRaises(FloatingPointError):
                audio_bc.extract(x, 16000)

    def test_json_safe_metadata_and_no_admission(self):
        meta = audio_bc.extract(linear_triad(), 16000)["metadata"]
        self.assertEqual(json.loads(json.dumps(meta, allow_nan=False)), meta)
        self.assertIsNone(meta["independent_realization_count"])
        self.assertFalse(meta["external_validation_passed"])
        self.assertFalse(meta["classifier_admitted"])
        self.assertFalse(meta["significance_inferred"])
        self.assertFalse(meta["null_calibrated"])


if __name__ == "__main__":
    unittest.main()
