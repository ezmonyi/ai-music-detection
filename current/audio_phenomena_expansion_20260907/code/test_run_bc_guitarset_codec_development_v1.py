import importlib.util
import pathlib
import struct
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace

import numpy as np


HERE = pathlib.Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('codec_development_under_test',
    HERE / 'run_bc_guitarset_codec_development_v1.py')
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def reduction(values):
    pools = []
    for index, value in enumerate(values):
        pools.append({'pool_index': index, 'eligible': value is not None,
                      'squared_bicoherence': value})
    finite = [v for v in values if v is not None]
    scalar = sum(finite) / len(finite) if finite else None
    return {'input_samples': 128000, 'pool_count': 2, 'discarded_tail_samples': 0,
            'eligibility_mask': [v is not None for v in values],
            'eligible_pool_count': len(finite), 'missing_pool_count': 2 - len(finite),
            'median_squared_bicoherence': scalar, 'pools': pools}


class ScalarStub:
    @staticmethod
    def reduce_metadata(metadata, *, crop):
        if crop != {'start_sample': 0, 'stop_sample_exclusive': 128000,
                    'source_resampled_samples': 128000}:
            raise ValueError('crop changed')
        return reduction(metadata['values'])


class CodecDevelopmentTests(unittest.TestCase):
    def process_fixture(self):
        scalar = reduction([0.2, 0.4])
        row = {'item_id': 'p_s_comp', 'player_id': 'p', 'score_id': 's',
               'performance': 'comp', 'style_from_score_prefix': 's',
               'split_role': 'development', 'condition': 'baseline',
               'source_float64_wav': {'path': '/accepted.wav', 'bytes': 1, 'sha256': 'a' * 64},
               'source_metadata': {'path': '/accepted.json', 'bytes': 1, 'sha256': 'b' * 64},
               'source_arrays': {'path': '/accepted.npz', 'bytes': 1, 'sha256': 'c' * 64},
               'accepted_float64_reduction': scalar}
        codec = SimpleNamespace(
            read_float_wav=lambda path: np.zeros(m.SAMPLES, dtype=np.float32),
            encode_command=lambda ffmpeg, source, output, recipe: ['encode', str(source), str(output)],
            decode_command=lambda ffmpeg, source, output: ['decode', str(source), str(output)])
        modules = {'codec_probe': codec, 'bicoherence_audio_v1': object(),
                   'bicoherence_scalar_v1': object()}
        recipes = {'_ffmpeg': '/ffmpeg',
            'mp3_128k': {'extension': '.mp3', 'encoder': 'libmp3lame'},
            'opus_96k': {'extension': '.opus', 'encoder': 'libopus'}}
        return row, modules, recipes, scalar

    @staticmethod
    def fake_binding(path, expected=None):
        path = str(path)
        if path == '/accepted.wav':
            return {'path': path, 'bytes': 1, 'sha256': 'a' * 64}
        return {'path': path, 'bytes': 1, 'sha256': 'f' * 64}

    def test_frozen_identifiers_and_scope(self):
        self.assertEqual(m.SOURCE_COMMIT_SHA,
            'cd11c3fb80755260b3908c542f8b8c9c218320c931c4987215936e11b73d639c')
        self.assertEqual(m.AUDIT_SHA,
            '61a926a92c764d6852608f9829395382e6593e56544d1434ec0f28af509ffc93')
        self.assertEqual(m.PINS['codec_probe'],
            'dbf1f548356046568329574ecbd57ddf99aab3164d666e0fa0b02b563d724696')
        self.assertEqual(m.SELECTED_CONDITIONS,
                         ('baseline', 'closed_minus6db', 'independent_minus6db'))
        self.assertEqual(len(m.ALL_SOURCE_CONDITIONS), 7)
        self.assertFalse(m.SCOPE['reserved_recordings_read'])
        self.assertFalse(m.SCOPE['unused_recordings_read'])
        self.assertFalse(m.SCOPE['BC_admitted'])
        self.assertEqual(m.SCOPE['classifier_fits'], 0)

    def test_roster_uses_metadata_only_and_binds_full_source_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            ids = ['p1_s1_comp', 'p1_s1_solo']
            products = {}
            for name in m.expected_source_names(ids):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'x')
                products[name] = {'bytes': 1, 'sha256': 'a' * 64}
            rows = []
            for item in ids:
                rows.append({'item_id': item, 'player_id': 'p1', 'score_id': 's1',
                    'performance': item.rsplit('_', 1)[1], 'style_from_score_prefix': 's',
                    'split_role': 'development',
                    'conditions': {name: {} for name in m.ALL_SOURCE_CONDITIONS}})
            summary = {'version': 'bicoherence_guitarset_development_pilot_v1',
                       'recording_denominator': 2, 'per_recording': rows}
            opened = []
            def metadata_loader(path, sha):
                opened.append(path)
                self.assertEqual(path.suffix, '.json')
                self.assertIn(path.stem, m.SELECTED_CONDITIONS)
                return {'values': [0.2, 0.4]}
            roster = m.build_roster(summary, products, root, ScalarStub,
                                    expected_count=2, metadata_loader=metadata_loader)
            self.assertEqual(len(roster), 6)
            self.assertEqual(len(opened), 6)
            self.assertEqual({row['condition'] for row in roster}, set(m.SELECTED_CONDITIONS))
            self.assertTrue(all(row['source_float64_wav']['sha256'] == 'a' * 64 for row in roster))

    def test_roster_rejects_reserved_and_inventory_mutation(self):
        summary = {'version': 'bicoherence_guitarset_development_pilot_v1',
                   'recording_denominator': 1,
                   'per_recording': [{'item_id': 'x', 'player_id': 'p', 'score_id': 's',
                     'performance': 'comp', 'style_from_score_prefix': 's',
                     'split_role': 'reserved',
                     'conditions': {name: {} for name in m.ALL_SOURCE_CONDITIONS}}]}
        with self.assertRaisesRegex(ValueError, 'inventory'):
            m.build_roster(summary, {}, pathlib.Path('/tmp'), ScalarStub, expected_count=1)
        products = {name: {'bytes': 0, 'sha256': 'a' * 64}
                    for name in m.expected_source_names(['x'])}
        with self.assertRaisesRegex(ValueError, 'development identity'):
            m.build_roster(summary, products, pathlib.Path('/tmp'), ScalarStub, expected_count=1)

    def test_precision_and_codec_comparison_direction_and_transitions(self):
        old = reduction([0.1, None])
        new = reduction([0.3, 0.4])
        result = m.scalar_comparison(old, new, 'float32_minus_float64')
        self.assertAlmostEqual(result['scalar_delta'], 0.25)
        self.assertEqual(result['mask_transition_counts'], {
            'eligible_to_eligible': 1, 'missing_to_missing': 0,
            'missing_to_eligible': 1, 'eligible_to_missing': 0})
        decoded = m.scalar_comparison(new, old, 'decoded_minus_float32')
        self.assertAlmostEqual(decoded['scalar_delta'], -0.25)
        self.assertEqual(decoded['mask_transition_counts']['eligible_to_missing'], 1)

    def test_null_and_failure_are_distinct(self):
        missing = reduction([None, None])
        valid = reduction([0.2, None])
        scientific = m.scalar_comparison(missing, missing, 'decoded_minus_float32')
        self.assertEqual(scientific['status'], 'compared_not_thresholded')
        self.assertIsNone(scientific['scalar_delta'])
        self.assertEqual(scientific['mask_transition_counts']['missing_to_missing'], 2)
        failed = m.scalar_comparison(valid, None, 'decoded_minus_float32')
        self.assertEqual(failed['status'], 'processing_failure_not_scientific_missingness')
        self.assertIsNone(failed['mask_transition_counts'])

    def test_operational_and_paired_pool_contrasts_stay_separate(self):
        closed = reduction([0.9, 0.1])
        independent = reduction([0.4, None])
        result = m.condition_contrast(closed, independent)
        self.assertAlmostEqual(result['operational_median_difference'], 0.1)
        self.assertAlmostEqual(result['paired_pool_mean_difference'], 0.5)
        self.assertEqual(result['paired_pool_count'], 1)
        self.assertEqual(result['pool_denominator'], 2)

    def test_transition_totals_preserve_denominator_and_failures(self):
        values = [
            m.scalar_comparison(reduction([0.1, None]), reduction([0.2, None]),
                                'decoded_minus_float32'),
            m.scalar_comparison(reduction([0.1, None]), None, 'decoded_minus_float32')]
        totals = m.transition_totals(values)
        self.assertEqual(totals['comparison_denominator'], 2)
        self.assertEqual(totals['processing_failures'], 1)
        self.assertEqual(totals['eligible_to_eligible'], 1)
        self.assertEqual(totals['missing_to_missing'], 1)

    def test_complete_summary_denominators_and_grouping(self):
        receipts = []
        base = reduction([0.2, 0.4])
        closed = reduction([0.8, 0.9])
        independent = reduction([0.2, 0.3])
        for index in range(90):
            identity = {'item_id': f'i{index:03}', 'player_id': f'p{index % 3}',
                        'score_id': f's{index % 15}',
                        'performance': 'comp' if index % 2 == 0 else 'solo'}
            for condition, scalar in [('baseline', base), ('closed_minus6db', closed),
                                      ('independent_minus6db', independent)]:
                precision = m.scalar_comparison(scalar, scalar, 'float32_minus_float64')
                receipts.append(identity | {'condition': condition,
                    'representations': {name: scalar for name in m.REPRESENTATIONS},
                    'measurement_status': {name: {'attempted': True, 'status': 'success'}
                                           for name in ('float32_control', *m.CODECS)},
                    'comparisons': {'float32_minus_float64': precision,
                        'decoded_minus_float32': {codec: m.scalar_comparison(
                            scalar, scalar, 'decoded_minus_float32') for codec in m.CODECS}}})
        summary = m.assemble_summary(receipts)
        self.assertEqual(summary['denominators'], {'recordings': 90,
            'selected_conditions': 270, 'precision_comparisons': 270,
            'codec_comparisons': 540})
        self.assertEqual(summary['new_BC_measurements'], {'expected': 810, 'attempted': 810,
            'successful': 810, 'failed_after_attempt': 0, 'skipped_before_attempt': 0})
        self.assertEqual(summary['processing_failures'], 0)
        self.assertEqual(summary['baseline']['precision_pool_transition_counts']
                         ['comparison_denominator'], 90)
        self.assertEqual(summary['baseline']['representation_coverage']['accepted_float64'], {
            'recording_denominator': 90, 'successful_measurements': 90,
            'processing_failures_or_skips': 0, 'pool_denominator': 180,
            'eligible_pools': 180, 'scientifically_missing_pools': 0,
            'unmeasured_pools_due_to_failure_or_skip': 0,
            'covered_recording_scalars': 90, 'scientifically_null_recording_scalars': 0})
        for representation in m.REPRESENTATIONS:
            block = summary['condition_contrasts'][representation]
            self.assertEqual(block['recording_denominator'], 90)
            self.assertEqual(block['paired_pool_denominator'], 180)
            self.assertEqual(block['paired_covered_pools'], 180)
            self.assertEqual(block['metrics']['operational_median_difference']
                             ['equal_score']['group_denominator'], 15)
            self.assertEqual(block['metrics']['operational_median_difference']
                             ['equal_player_secondary']['group_denominator'], 3)
        delta = summary['scalar_delta_distributions']['codec_decoded_minus_float32']['mp3_128k']
        self.assertEqual(delta['all_conditions']['comparison_denominator'], 270)
        self.assertEqual(delta['by_condition']['baseline']['comparison_denominator'], 90)
        self.assertEqual(delta['all_conditions']['p95_quantile_method'],
                         'linear interpolation at h=(n-1)*0.95')

    def test_measurement_accounting_separates_failure_and_skip(self):
        receipts = [{'measurement_status': {
            'float32_control': {'attempted': True, 'status': 'measurement_failure'},
            'mp3_128k': {'attempted': False, 'status': 'skipped_dependency_failure'},
            'opus_96k': {'attempted': True, 'status': 'success'}}}]
        self.assertEqual(m.measurement_counts(receipts), {'expected': 3, 'attempted': 2,
            'successful': 1, 'failed_after_attempt': 1, 'skipped_before_attempt': 1})

    def test_delta_distribution_retains_scientific_null_and_failure(self):
        comparisons = [
            m.scalar_comparison(reduction([0.1, 0.2]), reduction([0.2, 0.4]),
                                'decoded_minus_float32'),
            m.scalar_comparison(reduction([None, None]), reduction([None, None]),
                                'decoded_minus_float32'),
            m.scalar_comparison(reduction([0.1, None]), None, 'decoded_minus_float32')]
        result = m.delta_summary(comparisons)
        self.assertEqual(result['comparison_denominator'], 3)
        self.assertEqual(result['successful_comparisons'], 2)
        self.assertEqual(result['scalar_pair_covered'], 1)
        self.assertEqual(result['scientific_null_pairs'], 1)
        self.assertEqual(result['processing_failure_comparisons'], 1)

    def test_float32_control_writer_is_exact_and_unclipped(self):
        samples = np.linspace(-1.25, 1.25, m.SAMPLES, dtype=np.float32)
        wav = m.wav_float32_bytes(samples)
        self.assertEqual(wav[:4], b'RIFF')
        self.assertEqual(wav[8:12], b'WAVE')
        data = wav.index(b'data')
        n = struct.unpack('<I', wav[data + 4:data + 8])[0]
        decoded = np.frombuffer(wav[data + 8:data + 8 + n], dtype='<f4')
        self.assertTrue(np.array_equal(decoded, samples))
        self.assertGreater(float(np.max(np.abs(decoded))), 1.0)

    def test_process_row_success_attempts_three_measurements(self):
        row, modules, recipes, scalar = self.process_fixture()
        measured = {'status': 'measured_not_admitted', 'scalar': scalar,
                    'metadata': {'path': '/m', 'bytes': 1, 'sha256': 'd' * 64},
                    'arrays': {'path': '/a', 'bytes': 1, 'sha256': 'e' * 64}}
        with tempfile.TemporaryDirectory() as directory, \
             mock.patch.object(m, 'binding', side_effect=self.fake_binding), \
             mock.patch.object(m, 'source_float64', return_value=np.zeros(m.SAMPLES)), \
             mock.patch.object(m, 'wav_float32_bytes', return_value=b'wav'), \
             mock.patch.object(m, 'write_new'), mock.patch.object(m, 'write_json'), \
             mock.patch.object(m, 'command'), \
             mock.patch.object(m, 'measure', return_value=measured) as measurement:
            result = m.process_row(row, pathlib.Path(directory) / 'out', modules, recipes)
        self.assertEqual(measurement.call_count, 3)
        self.assertEqual(result['measurement_status'], {
            'float32_control': {'attempted': True, 'status': 'success'},
            'mp3_128k': {'attempted': True, 'status': 'success'},
            'opus_96k': {'attempted': True, 'status': 'success'}})
        self.assertEqual(set(result['representations']), set(m.REPRESENTATIONS))

    def test_process_row_control_measurement_failure_skips_codecs_without_commands(self):
        row, modules, recipes, _ = self.process_fixture()
        with tempfile.TemporaryDirectory() as directory, \
             mock.patch.object(m, 'binding', side_effect=self.fake_binding), \
             mock.patch.object(m, 'source_float64', return_value=np.zeros(m.SAMPLES)), \
             mock.patch.object(m, 'wav_float32_bytes', return_value=b'wav'), \
             mock.patch.object(m, 'write_new'), mock.patch.object(m, 'write_json'), \
             mock.patch.object(m, 'command') as command, \
             mock.patch.object(m, 'measure', side_effect=RuntimeError('extract failed')) as measurement:
            result = m.process_row(row, pathlib.Path(directory) / 'out', modules, recipes)
        self.assertEqual(measurement.call_count, 1)
        command.assert_not_called()
        self.assertEqual(result['measurement_status']['float32_control'],
                         {'attempted': True, 'status': 'measurement_failure'})
        self.assertEqual(result['measurement_status']['mp3_128k'],
                         {'attempted': False, 'status': 'skipped_dependency_failure'})
        self.assertEqual(result['measurement_status']['opus_96k'],
                         {'attempted': False, 'status': 'skipped_dependency_failure'})

    def test_process_row_wrong_decoded_length_retains_failure_without_codec_measurement(self):
        row, modules, recipes, scalar = self.process_fixture()
        def decoded(path):
            name = str(path)
            if 'mp3_128k.decoded' in name:
                return np.zeros(m.SAMPLES - 1, dtype=np.float32)
            return np.zeros(m.SAMPLES, dtype=np.float32)
        modules['codec_probe'].read_float_wav = decoded
        measured = {'status': 'measured_not_admitted', 'scalar': scalar,
                    'metadata': {'path': '/m', 'bytes': 1, 'sha256': 'd' * 64},
                    'arrays': {'path': '/a', 'bytes': 1, 'sha256': 'e' * 64}}
        with tempfile.TemporaryDirectory() as directory, \
             mock.patch.object(m, 'binding', side_effect=self.fake_binding), \
             mock.patch.object(m, 'source_float64', return_value=np.zeros(m.SAMPLES)), \
             mock.patch.object(m, 'wav_float32_bytes', return_value=b'wav'), \
             mock.patch.object(m, 'write_new'), mock.patch.object(m, 'write_json'), \
             mock.patch.object(m, 'command') as command, \
             mock.patch.object(m, 'measure', return_value=measured) as measurement:
            result = m.process_row(row, pathlib.Path(directory) / 'out', modules, recipes)
        self.assertEqual(command.call_count, 4)
        self.assertEqual(measurement.call_count, 2)  # control and Opus; never wrong-length MP3
        self.assertEqual(result['measurement_status']['mp3_128k'],
                         {'attempted': False, 'status': 'skipped_pipeline_failure'})
        self.assertEqual(result['measurement_status']['opus_96k'],
                         {'attempted': True, 'status': 'success'})
        self.assertIsNone(result['representations']['mp3_128k'])

    def test_freeze_schema_is_exact_not_boolean_authority(self):
        self.assertEqual(m.FREEZE_VERSION, 'bc_guitarset_codec_development_parent_freeze_v1')
        self.assertNotIn('authorized', m.SCOPE)
        authorities = {'bindings': {}, 'codec_results': {'codec_recipes': {}, 'toolchain': {}},
            'raw_root': '/raw', 'codec_root': '/codec', 'output_root': '/new', 'runtime': {},
            'source_commit': {'products': {}}, 'source_products_sha256': 'a' * 64, 'roster': []}
        with mock.patch.object(m, 'binding', return_value={'path': '/x', 'bytes': 1,
                                                          'sha256': 'b' * 64}):
            document = m.draft_document(authorities, '/runner', '/tests', '/protocol')
        self.assertTrue(document['parent_freeze_required'])
        self.assertFalse(document['development_audio_opened'])
        self.assertTrue(document['synthetic_codec_package_replayed'])


if __name__ == '__main__':
    unittest.main()
