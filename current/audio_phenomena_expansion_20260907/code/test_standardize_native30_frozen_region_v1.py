"""Synthetic native waveforms only; no corpus or reserved records are read."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf
import standardize_native30_frozen_region_v1 as m


class FrozenRegionTests(unittest.TestCase):
    def fixture(self, directory, rate=100, seconds=100, channels=2):
        path = Path(directory) / 'native.wav'
        time = np.arange(seconds * rate, dtype=np.float64) / rate
        values = np.column_stack([0.25 + np.sin(2*np.pi*3*time), 2*np.cos(2*np.pi*7*time)])
        if channels != 2:
            values = values[:, :1] if channels == 1 else np.column_stack([values, values[:, 0]])
        sf.write(path, values, rate, subtype='DOUBLE')
        return path, values, m.dsp.digest(path)

    def run_fixture(self, path, sha, rate=100, start=1000, historical=None, strength=m.BOUNDS_ONLY):
        return m.standardize(path, sha, rate, 2, region_start_frame=start,
                             region_frames=60*rate, evidence_strength=strength,
                             expected_region_float64_sha256=historical)

    def test_native_context_center_and_no_historical_hash_invention(self):
        with tempfile.TemporaryDirectory() as directory:
            path, values, sha = self.fixture(directory)
            output, audit = self.run_fixture(path, sha)
            self.assertEqual(audit['coordinates']['crop_start_frame'], 2500)
            self.assertNotEqual(audit['coordinates']['crop_start_frame'], 3500)  # full100 center
            self.assertEqual(audit['native_crop_float64_sha256'], m.dsp.pcm_hash(values[2500:5500]))
            self.assertEqual(audit['observed_current_region_float64_sha256'], m.dsp.pcm_hash(values[1000:7000]))
            self.assertIsNone(audit['historical_expected_region_float64_sha256'])
            self.assertIsNone(audit['historical_region_hash_match'])
            self.assertFalse(audit['historical_region_waveform_hash_available'])
            self.assertEqual(output.shape, (1323000, 2))
            self.assertGreater(audit['samples_abs_above_one'], 0)
            self.assertEqual(json.loads(json.dumps(audit, allow_nan=False)), audit)

    def test_historical_hash_checked_before_resampling(self):
        with tempfile.TemporaryDirectory() as directory:
            path, values, sha = self.fixture(directory)
            expected = m.dsp.pcm_hash(values[1000:7000])
            _, audit = self.run_fixture(path, sha, historical=expected, strength=m.HASH_VERIFIED)
            self.assertTrue(audit['historical_region_hash_match'])
            with patch.object(m.dsp, 'resample_crop', side_effect=AssertionError('must not resample')):
                with self.assertRaisesRegex(ValueError, 'historical native60'):
                    self.run_fixture(path, sha, historical='0'*64, strength=m.HASH_VERIFIED)

    def test_missing_mandatory_hash_and_unsupported_strength(self):
        for strength, value in [(m.HASH_VERIFIED, None), (m.HASH_VERIFIED, 'bad'),
                                (m.HASH_VERIFIED, 'A'*64), (m.BOUNDS_ONLY, 'a'*64),
                                ('unknown', None)]:
            with self.assertRaises(ValueError):
                m.region_coordinates(100, 0, 6000, strength, value)

    def test_integer_bounds_and_exact60(self):
        for rate, start, frames in [(True, 0, 60), (100.0, 0, 6000), (0, 0, 0),
                                     (100, -1, 6000), (100, True, 6000), (100, 0.0, 6000),
                                     (100, 0, 6000.0), (100, 0, True), (100, 0, 5999), (100, 0, 6001)]:
            with self.assertRaises(ValueError):
                m.region_coordinates(rate, start, frames, m.BOUNDS_ONLY, None)
        coordinates = m.region_coordinates(44100, 12345, 2646000, m.BOUNDS_ONLY, None)
        self.assertEqual(coordinates['crop_start_frame'], 673845)

    def test_short_interval_rejected_without_padding(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _, sha = self.fixture(directory, seconds=69)
            with self.assertRaisesRegex(ValueError, 'ended before frozen region_end'):
                self.run_fixture(path, sha)

    def test_source_hash_and_channels_rate_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _, sha = self.fixture(directory)
            with self.assertRaisesRegex(ValueError, 'source SHA'):
                self.run_fixture(path, '0'*64)
            with self.assertRaisesRegex(ValueError, 'rate/channels'):
                self.run_fixture(path, sha, rate=101)
            for channels in [True, 1, 3, 2.0]:
                with self.assertRaises(ValueError):
                    m.standardize(path, sha, 100, channels, region_start_frame=0,
                                  region_frames=6000, evidence_strength=m.BOUNDS_ONLY,
                                  expected_region_float64_sha256=None)

    def test_physical_mono_and_multichannel_rejected(self):
        for channels in [1, 3]:
            with tempfile.TemporaryDirectory() as directory:
                path, _, sha = self.fixture(directory, channels=channels)
                with self.assertRaisesRegex(ValueError, 'rate/channels'):
                    self.run_fixture(path, sha)

    def test_source_mutation_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _, sha = self.fixture(directory)
            with patch.object(m.dsp, 'signature', side_effect=[(1,2,3,4,5), (1,2,3,4,6)]):
                with self.assertRaisesRegex(ValueError, 'source changed'):
                    self.run_fixture(path, sha)

    def test_sequential_pass_mutation_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _, sha = self.fixture(directory)
            original = m.read_region
            calls = []
            def mutate(*args):
                values, audit = original(*args); calls.append(1)
                if len(calls) == 2:
                    values[0, 0] += 0.1
                    audit['observed_region_float64_sha256'] = m.dsp.pcm_hash(values)
                return values, audit
            with patch.object(m, 'read_region', side_effect=mutate):
                with self.assertRaisesRegex(ValueError, 'changed across sequential passes'):
                    self.run_fixture(path, sha)

    def test_pass_array_hash_disagreement_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _, sha = self.fixture(directory)
            original = m.read_region
            def mutate(*args):
                values, audit = original(*args); values[0, 0] += 0.1
                return values, audit
            with patch.object(m, 'read_region', side_effect=mutate):
                with self.assertRaisesRegex(ValueError, 'arrays/hash'):
                    self.run_fixture(path, sha)

    def test_nonfinite_prefix_rejected_but_unobserved_tail_not_decoded(self):
        with tempfile.TemporaryDirectory() as directory:
            path, values, _ = self.fixture(directory)
            values[9000, 0] = np.nan  # outside decoded-through boundary7000
            sf.write(path, values, 100, subtype='DOUBLE')
            _, audit = self.run_fixture(path, m.dsp.digest(path))
            self.assertEqual(audit['decoded_through_frame_exclusive'], 7000)
            self.assertFalse(audit['full_stream_completeness_established'])
            self.assertNotIn('actual_frames', audit)
            values[500, 0] = np.nan  # prefix before retained interval must be finite
            sf.write(path, values, 100, subtype='DOUBLE')
            with self.assertRaisesRegex(ValueError, 'nonfinite'):
                self.run_fixture(path, m.dsp.digest(path))

    def test_bounded_read_no_seek_no_eof_probe_even_with_short_nonempty_reads(self):
        class Stream:
            samplerate, channels, format, subtype = 1, 2, 'WAV', 'DOUBLE'
            def __init__(self):
                self.offset = 0; self.requests = []
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def seek(self, *args): raise AssertionError('seek forbidden')
            def read(self, frames, **kwargs):
                self.requests.append(frames)
                if self.offset >= 67: raise AssertionError('post-boundary read forbidden')
                count = min(frames, 13, 67-self.offset)
                values = np.column_stack([np.arange(self.offset, self.offset+count)] * 2).astype(np.float64)
                self.offset += count
                return values
        stream = Stream()
        with patch.object(m.dsp.sf, 'SoundFile', return_value=stream):
            values, audit = m.read_region('synthetic', 1, 7, 60)
        np.testing.assert_array_equal(values[:, 0], np.arange(7, 67))
        self.assertEqual(stream.offset, 67)
        self.assertEqual(stream.requests, [67, 54, 41, 28, 15, 2])
        self.assertFalse(audit['empty_eof_probe_performed'])

    def test_native44100_and48000_exact30_and_frozen_dependency(self):
        for rate in [44100, 48000]:
            with self.subTest(rate=rate), tempfile.TemporaryDirectory() as directory:
                path, values, sha = self.fixture(directory, rate=rate, seconds=61)
                start = rate
                output, audit = self.run_fixture(path, sha, rate=rate, start=start)
                crop = values[16*rate:46*rate]
                expected = m.dsp.resample_crop(crop, rate)
                np.testing.assert_array_equal(output, expected)
                self.assertEqual(audit['resample_crop_dependency']['sha256'], m.DSP_SHA)
                self.assertEqual(audit['output_frames'], 1323000)
                self.assertEqual(audit['runtime']['numpy'], np.__version__)

    def test_config_does_not_mutate_dependency_or_claim_full_eof(self):
        original = copy.deepcopy(m.dsp.CONFIG)
        self.assertEqual(m.CONFIG['decode'], 'two_fresh_sequential_passes_from_zero_through_region_end_only')
        for key in ['normalization', 'limiting', 'dc_removal', 'channel_mixing', 'short_audio_padding']:
            self.assertFalse(m.CONFIG[key])
        self.assertEqual(m.CONFIG['window'], ['kaiser', 5.0])
        self.assertEqual(m.dsp.CONFIG, original)
        self.assertEqual(m.dsp.CONFIG['decode'], 'float64_two_sequential_passes_to_empty_EOF')
        with patch.object(m.dsp, 'digest', return_value='0'*64):
            with self.assertRaisesRegex(ValueError, 'dependency changed'):
                m._dependency_binding()


if __name__ == '__main__':
    unittest.main()
