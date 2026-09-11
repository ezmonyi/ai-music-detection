import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import soundfile as sf

import verify_extract_equal60 as v


class Exact60GuardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def wave(self, frames=4410, channels=2, rate=44100, nonfinite=False):
        p = self.root / 'wave.wav'
        y = np.zeros((frames, channels), dtype=np.float32)
        if nonfinite:
            y[0, 0] = np.nan
        sf.write(p, y, rate, subtype='FLOAT')
        return p

    def test_exact_wave_and_hash(self):
        p = self.wave()
        self.assertEqual(v.inspect_audio(p, v.sha_file(p), frames=4410, subtype='FLOAT')['frames'], 4410)
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            v.inspect_audio(p, 'bad', frames=4410)

    def test_one_sample_short_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Exact-frame'):
            v.inspect_audio(self.wave(frames=4409), frames=4410)

    def test_wrong_rate_or_channels(self):
        for channels, rate in ((1, 44100), (2, 48000)):
            with self.assertRaisesRegex(ValueError, 'Exact-frame'):
                v.inspect_audio(self.wave(channels=channels, rate=rate), frames=4410)

    def test_nonfinite_audio_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            v.inspect_audio(self.wave(nonfinite=True), frames=4410)

    def test_empty_beats_not_missing(self):
        p = self.root / 'empty.beats'
        with self.assertRaisesRegex(ValueError, 'Missing beat'):
            v.inspect_beats(p)
        p.write_text('')
        self.assertEqual(v.inspect_beats(p)['status'], 'empty_unavailable')

    def test_malformed_outside_or_unordered_beats(self):
        p = self.root / 'bad.beats'
        for text in ('1\t1\n0.5\t2\n', '61\t1\n', '1\t1.5\n', 'nan\t1\n'):
            p.write_text(text)
            with self.assertRaises(ValueError):
                v.inspect_beats(p)
        p.write_text('0.5\t1\n1\t2\n')
        self.assertEqual(v.inspect_beats(p)['beat_count'], 2)

    def test_structure_input_and_whole_duration(self):
        p = self.root / 'structure.json'
        payload = {'path': '/exact/input.wav', 'segments': [{'start': 0, 'end': 60, 'label': 'verse'}]}
        p.write_text(json.dumps(payload))
        self.assertEqual(v.inspect_structure(p, '/exact/input.wav')['segment_count'], 1)
        with self.assertRaisesRegex(ValueError, 'provenance'):
            v.inspect_structure(p, '/different/input.wav')
        payload['segments'][0]['end'] = 30
        p.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, 'span exact60'):
            v.inspect_structure(p, '/exact/input.wav')


if __name__ == '__main__':
    unittest.main()
