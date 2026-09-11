from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

import probe_bc_codec_roundtrip_v1 as m


class CodecProbeTests(unittest.TestCase):
    def test_fixed_synthetic_fixture_inventory_and_amplitudes(self):
        signals = m.synthetic_signals()
        self.assertEqual(list(signals), ['silence', 'impulse', 'tone_997hz',
                                         'multitone_500_750_1250hz'])
        for samples in signals.values():
            self.assertEqual(samples.dtype, np.float32)
            self.assertEqual(samples.shape, (64000,))
            self.assertTrue(np.isfinite(samples).all())
            self.assertLess(float(np.max(np.abs(samples))), 1.0)
        self.assertEqual(np.flatnonzero(signals['impulse']).tolist(), [8000, 32000, 56000])
        self.assertEqual(float(np.max(np.abs(signals['multitone_500_750_1250hz']))),
                         float(np.float32(0.85)))

    def test_FLOAT_wav_roundtrip_is_bit_exact(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name).resolve() / 'fixture.wav'
            samples = m.synthetic_signals()['multitone_500_750_1250hz']
            m.write_new(path, m.wav_bytes(samples))
            decoded = m.read_float_wav(path)
            self.assertTrue(np.array_equal(samples, decoded))
            self.assertEqual(path.read_bytes()[20:22], b'\x03\x00')

    def test_cross_correlation_reports_delay_without_modifying_arrays(self):
        source = np.zeros(64, dtype=np.float32); source[10] = 0.75
        decoded = np.zeros(69, dtype=np.float32); decoded[13] = 0.75
        source_before, decoded_before = source.copy(), decoded.copy()
        result = m.compare_raw(source, decoded)
        self.assertEqual(result['decoded_minus_input_samples'], 5)
        self.assertEqual(result['cross_correlation']['peak_lag_samples'], 3)
        self.assertTrue(result['no_alignment_trim_or_padding_applied'])
        self.assertTrue(np.array_equal(source, source_before))
        self.assertTrue(np.array_equal(decoded, decoded_before))

    def test_silence_alignment_is_explicitly_undefined(self):
        result = m.compare_raw(np.zeros(16, dtype=np.float32), np.zeros(20, dtype=np.float32))
        self.assertEqual(result['decoded_minus_input_samples'], 4)
        self.assertEqual(result['cross_correlation'],
                         {'status': 'undefined_zero_energy', 'peak_lag_samples': None,
                          'peak_lag_seconds': None, 'normalized_peak': None})

    def test_amplitude_stats_report_overshoot_without_clipping(self):
        samples = np.array([0.0, -1.1, 1.0, 0.5], dtype=np.float32)
        stats = m.signal_stats(samples)
        self.assertEqual(stats['abs_ge_1_samples'], 2)
        self.assertTrue(stats['peak_over_unity'])
        self.assertGreater(stats['peak_abs'], 1.0)
        self.assertIn('not_inferred', stats['clipping_inference'])
        self.assertEqual(float(samples[1]), float(np.float32(-1.1)))

    def test_codec_commands_are_explicit_and_have_no_alignment_filters(self):
        ffmpeg = '/synthetic/ffmpeg'
        source, encoded, decoded = map(Path, ('/synthetic/in.wav', '/synthetic/out.mp3',
                                              '/synthetic/decoded.wav'))
        mp3 = m.encode_command(ffmpeg, source, encoded, m.CODECS['mp3_128k'])
        decode = m.decode_command(ffmpeg, encoded, decoded)
        self.assertIn('libmp3lame', mp3)
        self.assertEqual(mp3[mp3.index('-b:a') + 1], '128k')
        self.assertEqual(mp3[mp3.index('-ar') + 1], '16000')
        self.assertEqual(mp3[mp3.index('-ac') + 1], '1')
        self.assertEqual(decode[decode.index('-c:a') + 1], 'pcm_f32le')
        forbidden = {'-ss', '-t', '-af', '-filter:a', 'apad', 'atrim'}
        self.assertTrue(forbidden.isdisjoint(mp3))
        self.assertTrue(forbidden.isdisjoint(decode))
        self.assertIn('-n', mp3)
        self.assertIn('-n', decode)

    def test_opus_recipe_freezes_non_bitrate_defaults(self):
        recipe = m.CODECS['opus_96k']
        self.assertEqual(recipe['encoder'], 'libopus')
        self.assertEqual(recipe['bitrate'], '96k')
        self.assertEqual(recipe['encoder_options'], ['-application', 'audio', '-vbr', 'on',
                                                     '-compression_level', '10',
                                                     '-frame_duration', '20'])

    def test_scope_is_non_authorizing_and_synthetic_only(self):
        self.assertTrue(m.SCOPE['synthetic_only'])
        self.assertFalse(m.SCOPE['reserved_audio_read'])
        self.assertFalse(m.SCOPE['development_audio_read'])
        self.assertFalse(m.SCOPE['BC_admitted'])
        self.assertFalse(m.SCOPE['gate_threshold_chosen'])
        self.assertEqual(m.SCOPE['classifier_fits'], 0)
        self.assertFalse(m.SCOPE['automatic_alignment'])
        self.assertFalse(m.SCOPE['automatic_trim'])
        self.assertFalse(m.SCOPE['automatic_padding'])

    def test_malformed_non_FLOAT_wav_is_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name).resolve() / 'bad.wav'
            payload = bytearray(m.wav_bytes(np.zeros(8, dtype=np.float32)))
            payload[20:22] = b'\x01\x00'
            m.write_new(path, bytes(payload))
            with self.assertRaisesRegex(ValueError, 'IEEE float'):
                m.read_float_wav(path)

    def test_WAVE_FORMAT_EXTENSIBLE_FLOAT_is_validated(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name).resolve() / 'extensible.wav'
            samples = np.linspace(-0.5, 0.5, 8, dtype=np.float32)
            raw = samples.astype('<f4').tobytes()
            guid = struct.pack('<IHH8s', 3, 0, 0x10, b'\x80\x00\x00\xaa\x00\x38\x9b\x71')
            fmt = (struct.pack('<HHIIHH', 65534, 1, 16000, 64000, 4, 32)
                   + struct.pack('<HHI', 22, 32, 4) + guid)
            body = b'fmt ' + struct.pack('<I', 40) + fmt + b'data' + struct.pack('<I', len(raw)) + raw
            m.write_new(path, b'RIFF' + struct.pack('<I', len(body) + 4) + b'WAVE' + body)
            decoded, info = m.read_float_wav(path, return_info=True)
            self.assertTrue(np.array_equal(decoded, samples))
            self.assertEqual(info['format'], 'WAVE_FORMAT_EXTENSIBLE_IEEE_FLOAT')
            self.assertEqual(info['valid_bits_per_sample'], 32)


if __name__ == '__main__':
    unittest.main()
