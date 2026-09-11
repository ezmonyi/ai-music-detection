import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

import numpy as np


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('reserved_runner_under_test',
    HERE / 'run_bc_guitarset_reserved_admission_v1.py')
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def reducer_fixture(value):
    grid = [[a, b, a + b] for a in range(8, 193, 8)
            for b in range(a, 193, 8) if a + b <= 256]
    metadata = {'version': 'bicoherence_audio_v1',
        'primitive_version': 'bicoherence_primitive_v2', 'status': 'ok',
        'sample_rate_hz': 16000, 'pool_samples': 64000, 'pool_duration_seconds': 4.0,
        'n_fft': 1024, 'hop_samples': 256, 'frames_per_pool': 247,
        'grid_cell_count': 228, 'energy_fraction_min_inclusive': 1e-6,
        'window': 'periodic_hann', 'tail_policy': 'discard_no_padding',
        'pool_boundary_policy': 'no_frame_crosses_pool_boundary', 'input_samples': 128000,
        'pool_count': 2, 'analyzed_samples': 128000, 'discarded_tail_samples': 0,
        'tail_start_sample': 128000, 'construction_status': 'ok', 'pools': []}
    for index in range(2):
        cells = []
        for bins in grid:
            target = bins == [32, 48, 80]
            eligible = not target or value is not None
            score = value if target and value is not None else .3
            cells.append({'frequency_bins': bins, 'eligible': eligible,
                'energy_floor_passed': eligible,
                'energy_fractions': [.01, .01, .01] if eligible else [1e-8, .01, .01],
                'status': 'ok' if eligible else 'below_energy_fraction_floor',
                'squared_bicoherence': score if eligible else None,
                'primitive': {'version': 'bicoherence_primitive_v2', 'realizations': 247,
                    'frequency_bins': bins, 'status': 'ok', 'squared_bicoherence': score}})
        metadata['pools'].append({'pool_index': index, 'start_sample': index * 64000,
            'stop_sample_exclusive': (index + 1) * 64000, 'coefficient_rows': 247,
            'status': 'ok', 'zero_amplitude': False,
            'total_non_dc_coefficient_energy': 1.0, 'cells': cells})
    return metadata


def identity(index=0):
    return {'item_id': f'item{index:03d}', 'player_id': '00', 'score_id': f's{index:03d}',
            'performance': 'comp' if index % 2 == 0 else 'solo',
            'style_from_score_prefix': 'BN', 'split_role': 'reserved'}


def statuses(value):
    return {'float32_control': {'attempted': value == 'success', 'status': value},
            **{name: {'attempted': value == 'success', 'status': value} for name in m.CODECS}}


class ReservedRunnerTests(unittest.TestCase):
    def test_exact_draft_and_frozen_producer_pins(self):
        self.assertEqual(m.DRAFT_COMMIT_SHA,
            '5d0dcb01974cb08f52ca9d7898a38be4a727f5a6776dd8780ec544493c69f4ac')
        self.assertEqual(m.DRAFT_BUILDER_SHA,
            'f27be0836c664b9649aa83877457ccece961378c41da2e9807dd378ff2f27ccd')
        self.assertEqual((len(m.CONDITIONS), len(m.SELECTED), len(m.CODECS)), (7, 3, 2))
        self.assertEqual(m.EXPECTED, {'recordings': 90, 'float64': 630,
            'precision': 270, 'codec': 540, 'measurements': 1440, 'pools': 2880})

    def test_storage_is_conservative_and_separate_freeze_is_required(self):
        self.assertEqual(m.storage_estimate()['required_free_bytes'], 12 * 1024**3)
        self.assertEqual(m.FREEZE_VERSION, 'bc_guitarset_reserved_admission_parent_freeze_v1')
        self.assertFalse(m.SCOPE['BC_admitted'])
        self.assertFalse(m.SCOPE['independent_numerical_replay_passed'])

    def test_actual_loader_shape_satisfies_verify_freeze_entrypoint(self):
        modules = m.load_frozen(HERE)
        self.assertIn('codec_probe', modules)
        self.assertNotIn('probe_bc_codec_roundtrip_v1', modules)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            draft_path = root / 'draft.json'; m.write_json(draft_path, {'synthetic': True})
            synthetic_commit = root / 'synthetic_COMMIT.json'
            m.write_json(synthetic_commit, {'synthetic': True})
            document = {'output_root': str(root / 'result'), 'codec_recipes': {'synthetic': True},
                        'authorities': {'synthetic_codec_COMMIT': m.binding(synthetic_commit)}}
            draft_commit = m.binding(synthetic_commit)
            runtime = {'synthetic': 'runtime'}
            codec_proof = {'results': {'toolchain': {'synthetic': 'toolchain'},
                                       'codec_recipes': document['codec_recipes']}}
            freeze = {'version': m.FREEZE_VERSION,
                'status': 'authorized_reserved_measurement_not_admission',
                'draft_COMMIT': draft_commit, 'draft': m.binding(draft_path),
                'output_root': document['output_root'],
                'authorized_scope': {'reserved_audio_access_authorized': True,
                    'reserved_BC_measurement_authorized': True,
                    'codec_encoding_authorized': True,
                    'classifier_fits_authorized': False,
                    'model_scoring_authorized': False,
                    'BC_admission_authorized': False,
                    'threshold_changes_authorized': False},
                'expected_measurements': m.EXPECTED,
                'storage_estimate': m.storage_estimate(), 'runtime': runtime,
                'toolchain': codec_proof['results']['toolchain'],
                'codec_recipes': document['codec_recipes'],
                'runner': m.binding(Path(m.__file__).resolve()),
                'runner_tests': m.binding(Path(__file__).resolve()),
                'independent_auditor': m.binding(Path(__file__).resolve())}
            freeze_path = root / 'freeze.json'; m.write_json(freeze_path, freeze)
            with mock.patch.object(m, 'verify_draft', return_value=(document, draft_commit)), \
                    mock.patch.object(modules['codec'], 'runtime_snapshot', return_value=runtime), \
                    mock.patch.object(modules['codec_probe'], 'verify_result', return_value=codec_proof):
                verified_document, verified_freeze, entry = m.verify_freeze(
                    root, m.DRAFT_COMMIT_SHA, freeze_path, m.digest(freeze_path), modules)
            self.assertEqual(verified_document, document)
            self.assertEqual(verified_freeze, freeze)
            self.assertEqual(entry, m.binding(freeze_path))

    def test_dependency_receipt_retains_three_skips(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            failure_path = root / 'source.failure.json'
            m.write_json(failure_path, {'status': 'failure'})
            receipt = m.dependency_codec_receipt(identity(), 'baseline', root / 'codec',
                                                  m.binding(failure_path.absolute()))
            self.assertEqual(set(receipt['measurement_status']),
                             {'float32_control', 'mp3_128k', 'opus_96k'})
            self.assertTrue(all(not row['attempted'] for row in
                                receipt['measurement_status'].values()))
            self.assertTrue(all(row['status'] == 'skipped_dependency_failure' for row in
                                receipt['measurement_status'].values()))
            self.assertTrue((root / 'codec/receipt.json').is_file())

    def test_outcome_accounting_exact_all_success(self):
        items = []
        for index in range(90):
            item = identity(index) | {'condition_status': {name: {'attempted': True,
                'status': 'success'} for name in m.CONDITIONS}, '_codec_values': []}
            for condition in m.SELECTED:
                item['_codec_values'].append({'measurement_status': statuses('success')})
            items.append(item)
        result = m.outcome_counts(items)
        self.assertEqual(result['float64']['successful'], 630)
        self.assertEqual(result['float32_precision_control']['successful'], 270)
        self.assertEqual(result['codec_decoded']['successful'], 540)
        self.assertEqual(result['all'], {'expected': 1440, 'attempted': 1440,
            'successful': 1440, 'failed_after_attempt': 0, 'skipped_before_attempt': 0})

    def test_outcome_accounting_keeps_failure_and_dependency_skips(self):
        items = []
        for index in range(90):
            condition_status = {name: {'attempted': True, 'status': 'success'}
                                for name in m.CONDITIONS}
            item = identity(index) | {'condition_status': condition_status,
                                      '_codec_values': []}
            for condition in m.SELECTED:
                item['_codec_values'].append({'measurement_status': statuses('success')})
            items.append(item)
        items[0]['condition_status']['baseline'] = {'attempted': True,
                                                     'status': 'measurement_failure'}
        items[0]['_codec_values'][0]['measurement_status'] = statuses(
            'skipped_dependency_failure')
        result = m.outcome_counts(items)
        self.assertEqual(result['all']['expected'], 1440)
        self.assertEqual(result['all']['successful'], 1436)
        self.assertEqual(result['all']['failed_after_attempt'], 1)
        self.assertEqual(result['all']['skipped_before_attempt'], 3)

    def test_stratum_coverage_preserves_all_six_fixed_denominators(self):
        rows = []
        for player in ('00', '02', '05'):
            for performance in ('comp', 'solo'):
                for index in range(15):
                    rows.append({'player_id': player, 'performance': performance,
                        'value': {'eligible_pool_count': 2} if index < 12 else
                                 {'eligible_pool_count': 0}})
        result = m.per_stratum_coverage(rows, lambda row: row['value'])
        self.assertEqual(len(result), 6)
        self.assertTrue(all(row['pool_denominator'] == 30 for row in result))
        self.assertTrue(all(row['eligible_fraction'] == .8 for row in result))

    def test_contrast_checks_use_both_noninterchangeable_endpoints(self):
        rows = []
        for index in range(90):
            rows.append({**identity(index), 'operational_median_difference': .3,
                         'paired_pool_mean_difference': .25})
        def entry(metric):
            return {'covered_recordings': 90,
                'equal_score': {'equal_group_mean': .3, 'rows': []},
                'equal_player_secondary': {'rows': [
                    {'mean': .3}, {'mean': .3}, {'mean': .3}]},
                'performance_strata': {'comp': {'mean': .3}, 'solo': {'mean': .3}}}
        summary = {'paired_covered_pools': 180, 'per_recording': rows,
                   'metrics': {name: entry(name) for name in
                    ('operational_median_difference', 'paired_pool_mean_difference')}}
        checks = []
        m.contrast_checks(checks, 'x', summary, .2)
        self.assertEqual(len(checks), 11)
        self.assertTrue(all(row['passed'] for row in checks))
        self.assertEqual({row['name'].split('.')[1] for row in checks[1:]},
                         {'operational_median_difference', 'paired_pool_mean_difference'})

    def test_process_item_success_calls_seven_float64_and_three_codec_rows(self):
        row = identity() | {'source_audio': {'path': '/not/read'},
                            'historical_native_float64_pcm_sha256': 'a' * 64,
                            'prospective_preprocessing': {}}
        scalar = {'pool_count': 2, 'eligibility_mask': [True, True],
                  'eligible_pool_count': 2, 'missing_pool_count': 0,
                  'median_squared_bicoherence': .5,
                  'pools': [{'eligible': True, 'squared_bicoherence': .5}] * 2}
        calls = []
        def fake_measure(samples, prefix, modules, construction_status):
            calls.append(Path(prefix).name)
            for suffix in ('.wav', '.metadata.json', '.arrays.npz'):
                Path(str(prefix) + suffix).write_bytes(b'x')
            return {'status': 'measured_reserved_not_admitted',
                'waveform': m.binding(Path(str(prefix) + '.wav').absolute()),
                'metadata': m.binding(Path(str(prefix) + '.metadata.json').absolute()),
                'arrays': m.binding(Path(str(prefix) + '.arrays.npz').absolute()),
                'scalar': scalar, 'descriptor': {'pools': []}}
        def fake_codec(source_row, destination, modules, recipes):
            destination.mkdir(parents=True, exist_ok=False)
            receipt = {**{key: source_row[key] for key in ('item_id', 'player_id', 'score_id',
                'performance', 'style_from_score_prefix', 'split_role', 'condition')},
                'representations': {'accepted_float64': scalar, 'float32_control': scalar,
                                    **{name: scalar for name in m.CODECS}},
                'comparisons': {'float32_minus_float64': {}, 'decoded_minus_float32': {}},
                'artifacts': {}, 'measurement_status': statuses('success')}
            m.write_json(destination / 'receipt.json', receipt)
            return receipt
        original = types.SimpleNamespace(constructions=lambda crop, item: {
            'waveforms': {name: np.zeros(m.SAMPLES, dtype=np.float64) for name in m.CONDITIONS},
            'arrays': {'x': np.zeros(1)}, 'metadata': {'status': 'ok'}},
            save_arrays=lambda path, arrays: Path(path).write_bytes(b'npz'))
        modules = {'original': original,
                   'codec': types.SimpleNamespace(process_row=fake_codec)}
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(m, 'decode_crop', return_value=(
                    np.zeros(m.SAMPLES, dtype=np.float64), {'crop_start': 0})), \
                mock.patch.object(m, 'measure_float64', side_effect=fake_measure):
            result = m.process_item(row, Path(directory) / 'item', modules,
                                    {'mp3_128k': {}, 'opus_96k': {}})
            self.assertEqual(calls, list(m.CONDITIONS))
            self.assertEqual(len(result['_codec_values']), 3)
            self.assertTrue(all(x['status'] == 'success' for x in
                                result['condition_status'].values()))

    def test_process_item_construction_failure_retains_all_sixteen_outcomes(self):
        row = identity() | {'source_audio': {'path': '/not/read'},
                            'historical_native_float64_pcm_sha256': 'a' * 64,
                            'prospective_preprocessing': {}}
        modules = {'original': types.SimpleNamespace(), 'codec': types.SimpleNamespace()}
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(m, 'decode_crop', side_effect=ValueError('bad source')):
            result = m.process_item(row, Path(directory) / 'item', modules, {})
            self.assertTrue(all(x['status'] == 'skipped_dependency_failure'
                                for x in result['condition_status'].values()))
            self.assertEqual(len(result['_codec_values']), 3)
            self.assertTrue(all(not status['attempted'] for receipt in result['_codec_values']
                for status in receipt['measurement_status'].values()))
            self.assertTrue((Path(directory) / 'item/receipt.json').is_file())

    def test_construction_persistence_failure_clears_built_and_skips_all_sixteen(self):
        row = identity() | {'source_audio': {'path': '/not/read'},
                            'historical_native_float64_pcm_sha256': 'a' * 64,
                            'prospective_preprocessing': {}}
        original = types.SimpleNamespace(constructions=lambda crop, item: {
            'waveforms': {name: np.zeros(m.SAMPLES, dtype=np.float64) for name in m.CONDITIONS},
            'arrays': {'x': np.zeros(1)}, 'metadata': {'status': 'ok'}},
            save_arrays=mock.Mock(side_effect=OSError('disk failure')))
        modules = {'original': original, 'codec': types.SimpleNamespace()}
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(m, 'decode_crop', return_value=(
                    np.zeros(m.SAMPLES, dtype=np.float64), {})), \
                mock.patch.object(m, 'measure_float64') as measured:
            result = m.process_item(row, Path(directory) / 'item', modules, {})
            measured.assert_not_called()
            self.assertEqual(result['construction']['status'], 'processing_failure')
            self.assertTrue(all(x['status'] == 'skipped_dependency_failure'
                                for x in result['condition_status'].values()))
            counts = m.outcome_counts([result] * 90)
            self.assertEqual(counts['all']['successful'], 0)
            self.assertEqual(counts['all']['skipped_before_attempt'], 1440)

    def test_construction_json_failure_retains_partial_arrays_and_skips_all_sixteen(self):
        row = identity() | {'source_audio': {'path': '/not/read'},
                            'historical_native_float64_pcm_sha256': 'a' * 64,
                            'prospective_preprocessing': {}}
        original = types.SimpleNamespace(constructions=lambda crop, item: {
            'waveforms': {name: np.zeros(m.SAMPLES, dtype=np.float64) for name in m.CONDITIONS},
            'arrays': {'x': np.zeros(1)}, 'metadata': {'status': 'ok'}},
            save_arrays=lambda path, arrays: Path(path).write_bytes(b'partial-arrays'))
        modules = {'original': original, 'codec': types.SimpleNamespace()}
        real_write_json = m.write_json
        def fail_construction_json(path, value):
            if Path(path).name == 'construction.json':
                raise OSError('metadata persistence failure')
            return real_write_json(path, value)
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(m, 'decode_crop', return_value=(
                    np.zeros(m.SAMPLES, dtype=np.float64), {})), \
                mock.patch.object(m, 'measure_float64') as measured, \
                mock.patch.object(m, 'write_json', side_effect=fail_construction_json):
            item_dir = Path(directory) / 'item'
            result = m.process_item(row, item_dir, modules, {})
            measured.assert_not_called()
            self.assertEqual((item_dir / 'construction.arrays.npz').read_bytes(),
                             b'partial-arrays')
            self.assertFalse((item_dir / 'construction.json').exists())
            self.assertTrue((item_dir / 'construction.failure.json').is_file())
            self.assertEqual(result['construction']['status'], 'processing_failure')
            self.assertTrue(all(x['status'] == 'skipped_dependency_failure'
                                for x in result['condition_status'].values()))
            counts = m.outcome_counts([result] * 90)
            self.assertEqual(counts['all']['successful'], 0)
            self.assertEqual(counts['all']['skipped_before_attempt'], 1440)

    def test_verify_result_rejects_symlink_before_resolution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / 'target'
            target.mkdir()
            link = root / 'link'
            link.symlink_to(target, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, 'safe result root'):
                m.verify_result(link.absolute(), 'a' * 64, HERE)

    def test_float64_measure_failure_does_not_invoke_codec_for_that_condition(self):
        row = identity() | {'source_audio': {'path': '/not/read'},
                            'historical_native_float64_pcm_sha256': 'a' * 64,
                            'prospective_preprocessing': {}}
        calls = []
        scalar = {'pool_count': 2}
        def fake_measure(samples, prefix, modules, construction_status):
            if Path(prefix).name == 'baseline':
                raise ValueError('measurement failed')
            for suffix in ('.wav', '.metadata.json', '.arrays.npz'):
                Path(str(prefix) + suffix).write_bytes(b'x')
            return {'waveform': m.binding(Path(str(prefix) + '.wav').absolute()),
                    'metadata': m.binding(Path(str(prefix) + '.metadata.json').absolute()),
                    'arrays': m.binding(Path(str(prefix) + '.arrays.npz').absolute()),
                    'scalar': scalar, 'descriptor': {}}
        def fake_codec(source_row, destination, modules, recipes):
            calls.append(source_row['condition'])
            destination.mkdir(parents=True, exist_ok=False)
            receipt = {**identity(), 'condition': source_row['condition'],
                       'measurement_status': statuses('success')}
            m.write_json(destination / 'receipt.json', receipt)
            return receipt
        original = types.SimpleNamespace(constructions=lambda crop, item: {
            'waveforms': {name: np.zeros(m.SAMPLES, dtype=np.float64) for name in m.CONDITIONS},
            'arrays': {'x': np.zeros(1)}, 'metadata': {'status': 'ok'}},
            save_arrays=lambda path, arrays: Path(path).write_bytes(b'npz'))
        modules = {'original': original, 'codec': types.SimpleNamespace(process_row=fake_codec)}
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(m, 'decode_crop', return_value=(
                    np.zeros(m.SAMPLES, dtype=np.float64), {})), \
                mock.patch.object(m, 'measure_float64', side_effect=fake_measure):
            result = m.process_item(row, Path(directory) / 'item', modules, {})
            self.assertEqual(calls, ['closed_minus6db', 'independent_minus6db'])
            self.assertEqual(result['_codec_values'][0]['measurement_status']['float32_control']
                             ['status'], 'skipped_dependency_failure')

    def test_real_frozen_reducers_integrate_90_rows_allpass_null_and_failure(self):
        scalar = m.module_from(HERE / 'bicoherence_scalar_v1.py', '_reserved_test_scalar',
                               m.PINS['bicoherence_scalar_v1'])
        original = m.module_from(HERE / 'bicoherence_guitarset_pilot_v1.py',
                                 '_reserved_test_original', m.ORIGINAL_PRODUCER_SHA)
        codec = m.module_from(HERE / 'run_bc_guitarset_codec_development_v1.py',
                              '_reserved_test_codec', m.CODEC_PRODUCER_SHA)
        values = {'baseline': .5, 'common_gain': .5, 'polarity': .5,
                  'closed_minus6db': .9, 'independent_minus6db': .1,
                  'closed_0db': .95, 'independent_0db': .05}
        reduced, descriptors = {}, {}
        for condition, value in values.items():
            metadata = reducer_fixture(value)
            reduced[condition] = scalar.reduce_metadata(metadata, crop={
                'start_sample': 0, 'stop_sample_exclusive': 128000,
                'source_resampled_samples': 128000})
            descriptors[condition] = original.descriptor(metadata)
        items, roster = [], []
        index = 0
        for player in ('00', '02', '05'):
            for score in range(15):
                for performance in ('comp', 'solo'):
                    ident = {'item_id': f'{player}_S{score:02d}_{performance}',
                        'player_id': player, 'score_id': f'S{score:02d}',
                        'performance': performance, 'style_from_score_prefix': 'BN',
                        'split_role': 'reserved'}
                    item = {**ident, 'construction': {'status': 'success'},
                        'condition_status': {condition: {'attempted': True, 'status': 'success'}
                                             for condition in m.CONDITIONS},
                        'conditions': {condition: {'scalar': reduced[condition],
                            'descriptor': descriptors[condition]} for condition in m.CONDITIONS},
                        'codec_receipts': {}, '_codec_values': []}
                    for condition in m.SELECTED:
                        reduction = reduced[condition]
                        item['_codec_values'].append({**ident, 'condition': condition,
                            'representations': {'accepted_float64': reduction,
                                'float32_control': reduction,
                                'mp3_128k': reduction, 'opus_96k': reduction},
                            'comparisons': {'float32_minus_float64': codec.scalar_comparison(
                                reduction, reduction, 'float32_minus_float64'),
                                'decoded_minus_float32': {name: codec.scalar_comparison(
                                    reduction, reduction, 'decoded_minus_float32')
                                    for name in m.CODECS}},
                            'measurement_status': statuses('success')})
                    items.append(item)
                    roster.append(ident)
                    index += 1
        document = {'roster': roster, 'expected_denominators': {},
                    'margins': {'synthetic': 'test'}}
        modules = {'original': original, 'codec': codec}
        complete = m.assemble_summary(items, document, modules)
        self.assertEqual(complete['measurement_counts']['all']['successful'], 1440)
        self.assertTrue(complete['producer_margin_checks']['all_producer_checks_passed'])
        self.assertFalse(complete['BC_admitted'])

        null_meta = reducer_fixture(None)
        null_reduction = scalar.reduce_metadata(null_meta, crop={
            'start_sample': 0, 'stop_sample_exclusive': 128000,
            'source_resampled_samples': 128000})
        null_descriptor = original.descriptor(null_meta)
        for condition in ('baseline', 'common_gain', 'polarity'):
            items[0]['conditions'][condition].update(
                scalar=null_reduction, descriptor=null_descriptor)
        baseline_receipt = items[0]['_codec_values'][0]
        baseline_receipt['representations'] = {name: null_reduction for name in
            ('accepted_float64', 'float32_control', *m.CODECS)}
        baseline_receipt['comparisons']['float32_minus_float64'] = codec.scalar_comparison(
            null_reduction, null_reduction, 'float32_minus_float64')
        baseline_receipt['comparisons']['decoded_minus_float32'] = {name:
            codec.scalar_comparison(null_reduction, null_reduction, 'decoded_minus_float32')
            for name in m.CODECS}
        with_null = m.assemble_summary(items, document, modules)
        self.assertEqual(with_null['measurement_counts']['all']['successful'], 1440)
        self.assertEqual(with_null['original_float64']['baseline_coverage']
                         ['scientifically_null_recording_scalars'], 1)
        self.assertTrue(with_null['producer_margin_checks']['all_producer_checks_passed'])

        items[0]['construction']['status'] = 'processing_failure'
        items[0]['condition_status']['baseline'] = {'attempted': True,
                                                     'status': 'measurement_failure'}
        items[0]['conditions']['baseline'].update(scalar=None, descriptor=None)
        items[0]['_codec_values'][0]['measurement_status'] = statuses(
            'skipped_dependency_failure')
        failed = m.assemble_summary(items, document, modules)
        self.assertEqual(failed['measurement_counts']['all']['successful'], 1436)
        self.assertFalse(failed['producer_margin_checks']['all_producer_checks_passed'])
        self.assertIn('with_retained_failures', failed['status'])


if __name__ == '__main__':
    unittest.main()
