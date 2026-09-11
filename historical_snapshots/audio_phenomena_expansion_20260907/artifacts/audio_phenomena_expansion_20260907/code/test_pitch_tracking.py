#!/usr/bin/env python3
"""Controlled tests for the frozen MIR-1K pYIN benchmark."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

import benchmark_pitch_tracking as benchmark


class TimestampAlignmentTests(unittest.TestCase):
    def test_reference_grid_starts_at_20ms_and_advances_20ms(self) -> None:
        np.testing.assert_allclose(
            benchmark.reference_times(4), np.asarray([0.02, 0.04, 0.06, 0.08]), atol=1e-12
        )

    def test_exact_and_within_voiced_run_interpolation(self) -> None:
        ref_t = np.asarray([0.00, 0.005, 0.01])
        est_t = np.asarray([0.00, 0.01])
        est_f0 = np.asarray([100.0, 110.0])
        est_v = np.asarray([True, True])
        mapped, voiced = benchmark.map_estimate_to_reference(ref_t, est_t, est_f0, est_v)
        np.testing.assert_allclose(mapped, [100.0, 105.0, 110.0])
        np.testing.assert_array_equal(voiced, [True, True, True])

    def test_never_interpolates_across_unvoiced_frame(self) -> None:
        ref_t = np.asarray([0.005, 0.015])
        est_t = np.asarray([0.00, 0.01, 0.02])
        est_f0 = np.asarray([100.0, np.nan, 120.0])
        est_v = np.asarray([True, False, True])
        mapped, voiced = benchmark.map_estimate_to_reference(ref_t, est_t, est_f0, est_v)
        self.assertTrue(np.all(np.isnan(mapped)))
        np.testing.assert_array_equal(voiced, [False, False])


class MetricTests(unittest.TestCase):
    def test_octave_error_fails_raw_pitch_but_passes_raw_chroma(self) -> None:
        ref = np.asarray([220.0, 440.0])
        est = np.asarray([440.0, 880.0])
        metrics, errors = benchmark.compute_metrics(ref, est, [True, True])
        self.assertEqual(metrics["reference_voiced_frames"], 2)
        self.assertEqual(metrics["pitch_correct_50c_frames"], 0)
        self.assertEqual(metrics["chroma_correct_50c_frames"], 2)
        self.assertEqual(metrics["raw_pitch_accuracy_50c"], 0.0)
        self.assertEqual(metrics["raw_chroma_accuracy_50c"], 1.0)
        np.testing.assert_allclose(errors, [1200.0, 1200.0], atol=1e-9)

    def test_zero_reference_is_unvoiced_not_a_pitch(self) -> None:
        ref = np.asarray([0.0, 220.0, 0.0, 220.0])
        est = np.asarray([np.nan, 220.0, 220.0, np.nan])
        voiced = np.asarray([False, True, True, False])
        metrics, _ = benchmark.compute_metrics(ref, est, voiced)
        self.assertEqual(metrics["reference_voiced_frames"], 2)
        self.assertEqual(metrics["reference_unvoiced_frames"], 2)
        self.assertEqual(metrics["pitch_correct_50c_frames"], 1)
        self.assertEqual(metrics["correct_unvoiced_frames"], 1)
        self.assertAlmostEqual(metrics["raw_pitch_accuracy_50c"], 0.5)
        self.assertAlmostEqual(metrics["voicing_precision"], 0.5)
        self.assertAlmostEqual(metrics["voicing_recall"], 0.5)
        self.assertAlmostEqual(metrics["voicing_f1"], 0.5)
        self.assertAlmostEqual(metrics["overall_accuracy"], 0.5)


class ControlledPyinIntegrationTest(unittest.TestCase):
    def test_right_channel_220hz_tone_tracks_known_f0_on_reference_grid(self) -> None:
        try:
            import librosa  # noqa: F401
            import soundfile as sf
        except ImportError as exc:  # pragma: no cover - local lightweight runtimes
            self.skipTest(f"pYIN runtime dependency unavailable: {exc}")

        sample_rate = benchmark.PYIN_CONFIG["sr"]
        duration_sec = 2.0
        samples = int(sample_rate * duration_sec)
        time = np.arange(samples, dtype=np.float64) / sample_rate
        right_vocal = (0.2 * np.sin(2.0 * np.pi * 220.0 * time)).astype(np.float32)
        left_accompaniment = np.zeros_like(right_vocal)
        stereo = np.column_stack([left_accompaniment, right_vocal])
        with tempfile.TemporaryDirectory() as temporary_directory:
            wav_path = Path(temporary_directory) / "known.wav"
            sf.write(wav_path, stereo, sample_rate, subtype="PCM_16")
            decoded, decoded_rate = sf.read(wav_path, always_2d=True, dtype="float32")
            self.assertEqual(decoded_rate, sample_rate)
            est_t, est_f0, est_v, _ = benchmark.estimate_pyin(decoded[:, 1])

        label_count = math.floor(duration_sec / benchmark.REFERENCE_FRAME_PERIOD_SEC) - 1
        ref_t = benchmark.reference_times(label_count)
        mapped_f0, mapped_v = benchmark.map_estimate_to_reference(
            ref_t, est_t, est_f0, est_v
        )
        metrics, _ = benchmark.compute_metrics(
            np.full(label_count, 220.0), mapped_f0, mapped_v
        )
        self.assertGreater(metrics["raw_pitch_accuracy_50c"], 0.95)
        self.assertGreater(metrics["voicing_recall"], 0.95)
        self.assertAlmostEqual(float(np.nanmedian(mapped_f0)), 220.0, delta=2.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
