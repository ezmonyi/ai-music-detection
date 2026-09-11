"""Regression tests for the exact historical crop and read-only input contract."""
import hashlib
from pathlib import Path
import tempfile
import unittest
import numpy as np
import soundfile as sf
from audio_inputs import load_exact, sha256_file


class InputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "input.wav"
        t = np.arange(3 * 16000) / 16000
        self.y = 0.2 * np.sin(2 * np.pi * 330 * t) + 0.03 * t
        sf.write(self.path, np.column_stack([self.y, self.y / 2]), 16000, subtype="DOUBLE")

    def tearDown(self):
        self.temp.cleanup()

    def test_exact_offset_mono_and_source_preserved(self):
        before = sha256_file(self.path)
        y, audit = load_exact({"audio_path": str(self.path), "audio_offset_s": 1}, duration=1)
        expected = self.y[16000:32000] * .75
        expected -= expected.mean()
        np.testing.assert_allclose(y, expected, atol=1e-15)
        self.assertEqual(audit["crop_start_frame"], 16000)
        self.assertEqual(audit["analysis_waveform_sha256"], hashlib.sha256(y.tobytes()).hexdigest())
        self.assertEqual(before, sha256_file(self.path))

    def test_no_padding(self):
        with self.assertRaises(ValueError):
            load_exact({"audio_path": str(self.path), "audio_offset_s": 2.5}, duration=1)

    def test_resample_length(self):
        y, audit = load_exact({"audio_path": str(self.path)}, duration=2, target_sr=8000)
        self.assertEqual(y.shape, (16000,))
        self.assertEqual(audit["analysis_sr"], 8000)

    def test_invalid_parameters(self):
        for duration in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                load_exact({"audio_path": str(self.path)}, duration=duration)
        with self.assertRaises(ValueError):
            load_exact({"audio_path": str(self.path)}, duration=1, target_sr=0)


if __name__ == "__main__":
    unittest.main()
