from pathlib import Path
import tempfile
import unittest

import numpy as np

import measure_bc_codec_scalar_synthetic_v1 as m


def scalar(mask, value):
    return {'eligibility_mask': [mask], 'median_squared_bicoherence': value}


def record(ident, signal, codec, mask, value, failure=None):
    return {'id': ident, 'signal': signal, 'codec': codec,
            'scalar': None if failure else scalar(mask, value),
            'processing_failure': failure}


class ScalarCodecTests(unittest.TestCase):
    def test_frozen_BC_module_hashes_and_origins(self):
        loaded = m.modules()
        self.assertEqual(loaded['bicoherence_audio_v1'].VERSION, 'bicoherence_audio_v1')
        self.assertEqual(loaded['bicoherence_scalar_v1'].VERSION, 'bicoherence_scalar_v1')
        self.assertEqual(loaded['bicoherence_primitive_v2'].VERSION, 'bicoherence_primitive_v2')

    def test_exact_twelve_measurement_and_eight_comparison_roster(self):
        ids = []
        for signal_name in m.SIGNALS:
            ids.append('original__' + signal_name)
            ids.extend('decoded__' + signal_name + '__' + codec for codec in m.CODECS)
        self.assertEqual(len(ids), 12)
        self.assertEqual(len(set(ids)), 12)
        self.assertEqual(len(m.SIGNALS) * len(m.CODECS), 8)

    def test_eligible_comparison_delta_direction(self):
        first = record('original__x', 'x', None, True, 0.8)
        second = record('decoded__x__mp3_128k', 'x', 'mp3_128k', True, 0.65)
        result = m.compare_pair(first, second)
        self.assertEqual(result['transition'], 'eligible_to_eligible')
        self.assertTrue(result['mask_agreement'])
        self.assertAlmostEqual(result['scalar_delta_decoded_minus_original'], -0.15)

    def test_null_transitions_are_retained_not_imputed(self):
        missing = record('original__x', 'x', None, False, None)
        still_missing = record('decoded__x__mp3_128k', 'x', 'mp3_128k', False, None)
        becomes_available = record('decoded__x__opus_96k', 'x', 'opus_96k', True, 0.4)
        same = m.compare_pair(missing, still_missing)
        changed = m.compare_pair(missing, becomes_available)
        self.assertEqual(same['transition'], 'missing_to_missing')
        self.assertIsNone(same['scalar_delta_decoded_minus_original'])
        self.assertEqual(changed['transition'], 'missing_to_eligible')
        self.assertFalse(changed['mask_agreement'])
        self.assertIsNone(changed['scalar_delta_decoded_minus_original'])

    def test_processing_failure_is_distinct_from_scientific_null(self):
        first = record('original__x', 'x', None, False, None,
                       {'type': 'FloatingPointError', 'message': 'synthetic'})
        second = record('decoded__x__mp3_128k', 'x', 'mp3_128k', False, None)
        result = m.compare_pair(first, second)
        self.assertEqual(result['status'], 'processing_failure_retained')
        self.assertIsNone(result['transition'])

    def test_measure_entry_preserves_failure_receipt(self):
        class Codec:
            @staticmethod
            def read_float_wav(path):
                return np.zeros(64000, dtype=np.float32)

        class Extractor:
            @staticmethod
            def extract(samples, rate):
                raise FloatingPointError('deliberate synthetic failure')

        class Reducer:
            pass

        with tempfile.TemporaryDirectory() as name:
            root = Path(name).resolve(); source = root / 'source.wav'; source.write_bytes(b'synthetic')
            output = root / 'output'; (output / 'measurements').mkdir(parents=True); (output / 'arrays').mkdir()
            entry = {'id': 'original__silence', 'kind': 'original', 'signal': 'silence',
                     'codec': None, 'input': m.binding(source), 'source_codec_row_sha256': None}
            result = m.measure_entry(entry, output, Codec, Extractor, Reducer)
            self.assertEqual(result['status'], 'processing_failure_retained_not_scientific_missingness')
            self.assertEqual(result['processing_failure']['type'], 'FloatingPointError')
            self.assertIsNone(result['raw_metadata'])
            self.assertFalse((output / 'arrays/original__silence.npz').exists())

    def test_array_manifest_and_npz_preserve_nan_and_complex(self):
        arrays = {'real': np.array([1.0, np.nan]),
                  'complex': np.array([1 + 2j, 3 - 4j], dtype=np.complex128)}
        with tempfile.TemporaryDirectory() as name:
            path = Path(name).resolve() / 'arrays.npz'
            m.save_npz(path, arrays)
            m.verify_npz(path, m.arrays_manifest(arrays), arrays)

    def test_scope_and_interpretation_are_non_authorizing(self):
        self.assertTrue(m.SCOPE['synthetic_codec_package_only'])
        self.assertFalse(m.SCOPE['development_audio_read'])
        self.assertFalse(m.SCOPE['reserved_audio_read'])
        self.assertFalse(m.SCOPE['BC_admitted'])
        self.assertFalse(m.SCOPE['grid_selection'])
        self.assertFalse(m.SCOPE['thresholds_chosen'])
        self.assertFalse(m.SCOPE['causality_inferred'])
        self.assertFalse(m.SCOPE['music_codec_invariance_established'])


if __name__ == '__main__':
    unittest.main()
