"""Synthetic-only independent reserved auditor tests; no reserved files opened."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.signal import resample_poly
import audit_bc_guitarset_reserved_admission_v1 as r


def numerical_fixture(prefix, samples, construction=None):
    reduced, pools = r.a.reduce_waveform(samples, construction)
    shape = (2, 228); spectra = np.asarray([p['spectra'] for p in pools])
    arrays = {'window': pools[0]['window'], 'frequency_bins': r.GRID, 'frequency_hz': r.GRID.astype(np.float64) * 16000 / 1024,
        'pool_start_samples': np.arange(2, dtype=np.int64) * r.POOL, 'frame_offset_samples': pools[0]['offsets'],
        'frame_start_samples': np.arange(2, dtype=np.int64)[:, None] * r.POOL + pools[0]['offsets'][None, :],
        'frame_means': np.asarray([p['means'] for p in pools]), 'spectra': spectra,
        'bin_coefficient_energy': np.asarray([p['energy'] for p in pools]),
        'total_non_dc_coefficient_energy': np.asarray([p['total_non_dc_coefficient_energy'] for p in pools]),
        'bin_energy_fraction': np.asarray([p['fractions'] for p in pools]),
        **{key: np.zeros(shape, np.bool_) for key in ('energy_floor_mask', 'primitive_defined_mask', 'eligible_mask')},
        **{key: np.full(shape, np.nan, np.float64) for key in ('squared_bicoherence', 'biphase_radians', 'primitive_squared_bicoherence', 'primitive_biphase_radians')}}
    records = []
    for p, pool in enumerate(pools):
        cells = []
        for c, bins in enumerate(r.GRID):
            primitive = r.primitive(pool['spectra'], bins); total = pool['total_non_dc_coefficient_energy']
            fractions = pool['fractions'][bins].tolist() if total else [None] * 3
            floor = bool(total and all(v >= 1e-6 for v in fractions)); defined = primitive['status'] == 'ok'; eligible = floor and defined
            cell = {'frequency_bins': bins.tolist(), 'status': 'ok' if eligible else primitive['status'] if not defined else 'below_energy_fraction_floor',
                'energy_fractions': fractions, 'energy_floor_passed': floor, 'eligible': eligible,
                'squared_bicoherence': primitive['squared_bicoherence'] if eligible else None,
                'biphase_radians': primitive['biphase_radians'] if eligible else None, 'primitive': primitive}
            cells.append(cell)
            for key, value in (('energy_floor_mask', floor), ('primitive_defined_mask', defined), ('eligible_mask', eligible)):
                arrays[key][p, c] = value
            for key, value in (('squared_bicoherence', cell['squared_bicoherence']), ('biphase_radians', cell['biphase_radians']),
                               ('primitive_squared_bicoherence', primitive['squared_bicoherence']), ('primitive_biphase_radians', primitive['biphase_radians'])):
                arrays[key][p, c] = np.nan if value is None else value
        records.append({'pool_index': p, 'start_sample': p * r.POOL, 'stop_sample_exclusive': (p + 1) * r.POOL,
                        'coefficient_rows': 247, 'status': pool['pool_status'], 'cells': cells})
    metadata = {'version': 'bicoherence_audio_v1', 'primitive_version': 'bicoherence_primitive_v2', 'input_samples': r.SAMPLES,
        'sample_rate_hz': 16000, 'pool_samples': 64000, 'pool_count': 2, 'analyzed_samples': r.SAMPLES, 'discarded_tail_samples': 0,
        'n_fft': 1024, 'hop_samples': 256, 'frames_per_pool': 247, 'energy_fraction_min_inclusive': 1e-6,
        'grid_cell_count': 228, 'null_calibrated': False, 'significance_inferred': False, 'external_validation_passed': False,
        'classifier_admitted': False, 'frames_overlap_and_are_not_asserted_independent': True, 'pools': records}
    if construction is not None: metadata['construction_status'] = construction
    Path(str(prefix) + '.metadata.json').write_bytes(r.canonical(metadata))
    np.savez_compressed(str(prefix) + '.arrays.npz', **arrays)
    return reduced, metadata, arrays


def scalar(values):
    pools = [{'pool_index': i, 'eligible': x is not None, 'squared_bicoherence': x} for i, x in enumerate(values)]
    finite = [x for x in values if x is not None]
    return {'pool_count': 2, 'pools': pools, 'eligibility_mask': [x is not None for x in values],
            'median_squared_bicoherence': float(np.median(finite)) if finite else None,
            'eligible_pool_count': len(finite), 'missing_pool_count': 2 - len(finite)}


def descriptive(value):
    return {'pools': [{'pool_index': p, 'pool_status': 'ok',
        'target': {'eligible': True, 'status': 'ok', 'squared_bicoherence': value},
        'grid_cell_count': 228, 'eligible_cell_count': 228, 'grid_eligibility': [True] * 228,
        'grid_status': ['ok'] * 228, 'grid_squared_bicoherence': [value] * 228,
        'grid_raw_squared_bicoherence': [value] * 228} for p in range(2)]}


def summary_fixture():
    items, rows = [], []
    for player in ('00', '02', '05'):
        for score in range(15):
            for performance in ('comp', 'solo'):
                row = {'item_id': f'synthetic_{player}_{score}_{performance}', 'player_id': player, 'score_id': f'S{score:02d}',
                       'performance': performance, 'style_from_score_prefix': 'synthetic', 'split_role': 'reserved'}
                rows.append(row); item = {**row, 'construction': {'status': 'success'}, 'condition_status': {}, 'conditions': {}, '_codec_values': []}
                for c in r.CONDITIONS:
                    value = .6 if c.startswith('closed_') else .1
                    item['conditions'][c] = {'scalar': scalar([value, value]), 'descriptor': descriptive(value)}
                    item['condition_status'][c] = {'attempted': True, 'status': 'success'}
                for c in r.SELECTED:
                    source = item['conditions'][c]['scalar']; reps = {key: copy.deepcopy(source) for key in ('accepted_float64', 'float32_control', *r.CODECS)}
                    receipt = {**row, 'condition': c, 'representations': reps,
                        'measurement_status': {key: {'attempted': True, 'status': 'success'} for key in ('float32_control', *r.CODECS)},
                        'comparisons': {'float32_minus_float64': r.a.compare_scalars(source, reps['float32_control'], 'float32_minus_float64'),
                            'decoded_minus_float32': {key: r.a.compare_scalars(reps['float32_control'], reps[key], 'decoded_minus_float32') for key in r.CODECS}}}
                    item['_codec_values'].append(receipt)
                items.append(item)
    document = {'roster': rows, 'expected_denominators': {'total_BC_measurements': 1440, 'total_BC_pools': 2880}, 'margins': {'synthetic_fixed': True}}
    return items, document


class NumericsTests(unittest.TestCase):
    def test_independent_resampling_bit_exact_synthetic_reference(self):
        rng = np.random.Generator(np.random.PCG64(81))
        signals = [np.zeros(1000, np.float64), np.ones(1001, np.float64),
                   np.r_[1., np.zeros(999)], rng.uniform(-1, 1, 863725),
                   rng.integers(-32768, 32768, 1992140).astype(np.float64) / 32768.0]
        for x in signals:
            actual = r.native_resample(x)
            expected = resample_poly(x, 160, 441, window=('kaiser', 5.0), padtype='constant', cval=0.0)
            self.assertEqual(actual.shape, expected.shape)
            self.assertTrue(np.array_equal(actual, expected), str(np.max(np.abs(actual - expected))))

    def test_pcm16_independent_decode_extremes_and_chunk_rejections(self):
        values = np.array([-32768, -2, -1, 0, 1, 32767], dtype='<i2')
        fmt = struct.pack('<HHIIHH', 1, 1, 44100, 88200, 2, 16)
        body = b'WAVEfmt ' + struct.pack('<I', len(fmt)) + fmt + b'data' + struct.pack('<I', values.nbytes) + values.tobytes()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'native.wav'; path.write_bytes(b'RIFF' + struct.pack('<I', len(body)) + body)
            expected = values.astype(np.float64) / 32768.0
            sha = hashlib.sha256(expected.astype('<f8').tobytes()).hexdigest()
            row = {'source_audio': r.binding(path), 'historical_native_float64_pcm_sha256': sha,
                   'historical_native_decode': {'format': 'WAV', 'subtype': 'PCM_16', 'sample_rate_hz': 44100,
                       'channels': 1, 'header_frames': 6, 'decoded_frames': 6, 'decoded_pcm_sha256': sha}}
            np.testing.assert_array_equal(r.native_pcm16(path, row), expected)
            damaged = bytearray(path.read_bytes()); damaged[22:24] = struct.pack('<H', 2); path.write_bytes(damaged)
            row['source_audio'] = r.binding(path)
            with self.assertRaisesRegex(ValueError, 'PCM16'): r.native_pcm16(path, row)

    def test_construction_exact_relations_and_no_clipping(self):
        x = np.random.default_rng(34).uniform(-1, 1, r.SAMPLES).astype(np.float64)
        waves, arrays, meta = r.construct(x, 'synthetic_only')
        self.assertEqual(tuple(waves), r.CONDITIONS)
        r.exact_array(waves['baseline'], x * .25, 'background')
        r.exact_array(waves['common_gain'], waves['baseline'] * .1, 'gain')
        r.exact_array(waves['polarity'], -waves['baseline'], 'polarity')
        r.exact_array(arrays['closed_phase_trajectories'][2], arrays['independent_phase_trajectories'][0] + arrays['independent_phase_trajectories'][1], 'closure')
        self.assertEqual(arrays['phase_knots_wrapped'].shape, (3, 65))
        for condition in r.CONDITIONS[3:]:
            injection = arrays[condition + '_injection']
            self.assertAlmostEqual(float(np.sqrt(np.mean(injection ** 2))) / meta['background_rms'],
                                   10 ** (-6 / 20) if 'minus6db' in condition else 1., places=14)
            r.exact_array(waves[condition], waves['baseline'] + injection, 'mixture')
        other, _, _ = r.construct(x, 'another_synthetic_only')
        self.assertFalse(np.array_equal(other['closed_0db'], waves['closed_0db']))

    def test_zero_construction_is_scientifically_unsupported_not_failure(self):
        waves, arrays, meta = r.construct(np.zeros(r.SAMPLES, np.float64), 'zero_synthetic')
        self.assertEqual(meta['status'], 'unsupported_zero_background_rms')
        self.assertTrue(all(not np.any(v) for v in waves.values()))
        reduced, pools = r.a.reduce_waveform(waves['baseline'], meta['status'])
        self.assertIsNone(reduced['median_squared_bicoherence']); self.assertEqual(reduced['eligible_pool_count'], 0)

    def test_full_grid_target_and_analytic_primitive(self):
        self.assertEqual(r.GRID.shape, (228, 3)); self.assertEqual(r.GRID[r.TARGET_INDEX].tolist(), [32, 48, 80])
        coeff = np.zeros((247, 513), np.complex128); coeff[:, [32, 48, 80]] = np.array([1 + 1j, 2 - 1j, 3 + 2j])
        result = r.primitive(coeff, [32, 48, 80]); self.assertAlmostEqual(result['squared_bicoherence'], 1., places=14)
        zero = r.primitive(np.zeros_like(coeff), [32, 48, 80]); self.assertEqual(zero['status'], 'zero_energy')
        self.assertIsNone(zero['squared_bicoherence'])

    def test_module_import_boundary(self):
        r.forbid_producer_imports()
        tree = ast.parse(Path(r.__file__).read_text())
        names = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        names += [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
        self.assertFalse(set(names) & set(r.FORBIDDEN))

    def test_complete_npz_shapes_dtypes_offsets_and_full_grid(self):
        with tempfile.TemporaryDirectory() as tmp:
            prefix = Path(tmp) / 'measurement'; samples = np.zeros(r.SAMPLES, np.float64)
            reduced, _, arrays = numerical_fixture(prefix, samples)
            actual, desc, _, _ = r.verify_numeric(samples, prefix, reduced, full_grid=True)
            self.assertIsNone(actual['median_squared_bicoherence']); self.assertEqual(len(desc['pools'][0]['grid_eligibility']), 228)
            for key in ('spectra', 'frame_means', 'bin_coefficient_energy', 'total_non_dc_coefficient_energy', 'eligible_mask', 'squared_bicoherence'):
                changed = dict(arrays); changed[key] = arrays[key][:1]
                np.savez_compressed(str(prefix) + '.arrays.npz', **changed)
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'dtype/shape'):
                    r.verify_numeric(samples, prefix, reduced, full_grid=True)
            changed = dict(arrays); changed['frame_means'] = changed['frame_means'].astype(np.float32)
            np.savez_compressed(str(prefix) + '.arrays.npz', **changed)
            with self.assertRaisesRegex(ValueError, 'dtype/shape'): r.verify_numeric(samples, prefix, reduced)
            for key in ('frequency_hz', 'pool_start_samples', 'frame_offset_samples', 'frame_start_samples'):
                changed = dict(arrays); changed[key] = arrays[key].copy(); changed[key].flat[0] += 1
                np.savez_compressed(str(prefix) + '.arrays.npz', **changed)
                with self.subTest(key=key), self.assertRaises(ValueError): r.verify_numeric(samples, prefix, reduced)

    def test_full_grid_off_target_primitive_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            prefix = Path(tmp) / 'measurement'; t = np.arange(r.SAMPLES) / 16000
            samples = (.2 * np.cos(2 * np.pi * 500 * t) + .2 * np.cos(2 * np.pi * 750 * t) + .2 * np.cos(2 * np.pi * 1250 * t)).astype(np.float64)
            reduced, metadata, arrays = numerical_fixture(prefix, samples)
            r.verify_numeric(samples, prefix, reduced, full_grid=True)
            metadata['pools'][0]['cells'][0]['primitive']['squared_bicoherence'] = .123
            Path(str(prefix) + '.metadata.json').write_bytes(r.canonical(metadata))
            with self.assertRaises(ValueError): r.verify_numeric(samples, prefix, reduced, full_grid=True)

    def test_wrong_length_float32_and_signed_zero_cast_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'bad.wav'; x = np.zeros(r.SAMPLES - 1, '<f4'); fmt = struct.pack('<HHIIHH', 3, 1, 16000, 64000, 4, 32)
            body = b'WAVEfmt ' + struct.pack('<I', 16) + fmt + b'data' + struct.pack('<I', x.nbytes) + x.tobytes()
            path.write_bytes(b'RIFF' + struct.pack('<I', len(body)) + body)
            with self.assertRaisesRegex(ValueError, 'length'): r.a.read_ieee_float_wav(path, bits=32)
        with self.assertRaises(ValueError): r.exact_array(np.array([0.], np.float32), np.array([-0.], np.float32), 'cast')


class ArithmeticTests(unittest.TestCase):
    def test_full_1440_measurement_and_2880_pool_summary(self):
        items, document = summary_fixture(); summary = r.reconstruct_summary(items, document)
        self.assertEqual(summary['measurement_counts']['all'], {'expected': 1440, 'attempted': 1440, 'successful': 1440, 'failed_after_attempt': 0, 'skipped_before_attempt': 0})
        self.assertEqual(sum(summary['measurement_counts'][key]['expected'] * 2 for key in ('float64', 'float32_precision_control', 'codec_decoded')), 2880)
        self.assertEqual(summary['original_float64']['full_grid_historical_summary']['nuisance']['common_gain']['grid_denominator'], 41040)
        self.assertTrue(summary['producer_margin_checks']['all_producer_checks_passed']); self.assertEqual(summary['producer_margin_checks']['check_count'], 72)
        self.assertFalse(summary['BC_admitted']); self.assertIsNone(summary['final_admission_decision'])
        self.assertEqual(summary['original_float64']['levels']['minus6db']['metrics']['operational_median_difference']['equal_score']['group_denominator'], 15)

    def test_failure_is_not_null_and_denominators_are_never_shrunk(self):
        items, document = summary_fixture(); item = items[0]; item['construction']['status'] = 'processing_failure'
        for c in r.CONDITIONS:
            item['conditions'][c] = {'scalar': None, 'descriptor': None}; item['condition_status'][c] = {'attempted': False, 'status': 'skipped_dependency_failure'}
        for receipt in item['_codec_values']:
            receipt['representations'] = dict.fromkeys(('accepted_float64', 'float32_control', *r.CODECS))
            receipt['measurement_status'] = {key: {'attempted': False, 'status': 'skipped_dependency_failure'} for key in ('float32_control', *r.CODECS)}
            receipt['comparisons'] = {'float32_minus_float64': r.a.compare_scalars(None, None, 'float32_minus_float64'),
                'decoded_minus_float32': {key: r.a.compare_scalars(None, None, 'decoded_minus_float32') for key in r.CODECS}}
        summary = r.reconstruct_summary(items, document); count = summary['measurement_counts']['all']
        self.assertEqual(count['expected'], 1440); self.assertEqual(count['successful'], 1424); self.assertEqual(count['skipped_before_attempt'], 16)
        coverage = summary['original_float64']['baseline_coverage']; self.assertEqual(coverage['pool_denominator'], 180)
        self.assertEqual(coverage['unmeasured_pools_due_to_failure_or_skip'], 2); self.assertEqual(coverage['scientifically_missing_pools'], 0)
        self.assertFalse(summary['producer_margin_checks']['all_producer_checks_passed'])
        scientific_null = r.a.coverage([scalar([None, None])]); self.assertEqual(scientific_null['processing_failures_or_skips'], 0)
        self.assertEqual(scientific_null['scientifically_missing_pools'], 2)

    def test_endpoint_distinction_equal_groups_and_ties(self):
        contrast = r.a.condition_difference(scalar([.8, .2]), scalar([.1, None]))
        self.assertAlmostEqual(contrast['operational_median_difference'], .4)
        self.assertAlmostEqual(contrast['paired_pool_mean_difference'], .7)
        rows = [{'score_id': 'A', 'v': 0.} for _ in range(5)] + [{'score_id': 'B', 'v': 1.}]
        self.assertEqual(r.a.equal_group(rows, 'score_id', 'v')['equal_group_mean'], .5)
        self.assertAlmostEqual(r.a.linear_quantile([0., 1., 2., 3.]), 2.85)
        pair = r.a.compare_scalars(scalar([.1, None]), scalar([None, .2]), 'decoded_minus_float32')
        self.assertEqual(pair['mask_transition_counts']['missing_to_eligible'], 1)
        self.assertEqual(pair['mask_transition_counts']['eligible_to_missing'], 1)
        items, document = summary_fixture(); original = r.original_summary(items)
        metric = original['levels']['minus6db']; metric['per_recording'][0]['operational_median_difference'] = 0.
        codec = r.a.reconstruct_summary([v for i in items for v in i['_codec_values']])
        checks = r.margin_checks(items, original, codec, r.counts(items), document)['checks']
        check = next(x for x in checks if x['name'] == 'original.minus6db.operational_median_difference.positive_recordings')
        self.assertEqual(check['observed'], 89)

    def test_inclusive_support_and_stratum_margin_boundaries(self):
        items, document = summary_fixture(); original = r.original_summary(items); codec = r.a.reconstruct_summary([v for i in items for v in i['_codec_values']])
        original['levels']['minus6db']['paired_covered_pools'] = 171
        results = r.margin_checks(items, original, codec, r.counts(items), document)['checks']
        self.assertTrue(next(x for x in results if x['name'] == 'original.minus6db.paired_pool_support')['passed'])
        original['levels']['minus6db']['paired_covered_pools'] = 170
        results = r.margin_checks(items, original, codec, r.counts(items), document)['checks']
        self.assertFalse(next(x for x in results if x['name'] == 'original.minus6db.paired_pool_support')['passed'])
        for invalid in ({'attempted': False, 'status': 'success'}, {'attempted': True, 'status': 'skipped_dependency_failure'}, {'attempted': 1, 'status': 'success'}):
            with self.assertRaises(ValueError): r.outcome(invalid)

    def test_complete_union_linked_library_map_and_byte_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            first, second = Path(tmp) / 'libone.so', Path(tmp) / 'libtwo.so'
            first.write_bytes(b'one'); second.write_bytes(b'two')
            freeze = {'toolchain': {'ffmpeg': {'path': '/synthetic/ffmpeg'}, 'ffprobe': {'path': '/synthetic/ffprobe'}},
                'linked_library_bindings': {str(p): r.binding(p) for p in (first, second)}}
            def ldd(command, **kwargs):
                path = first if command[1].endswith('ffmpeg') else second
                return SimpleNamespace(stdout=f'libexample.so => {path} (0x123)\n', stderr='')
            with patch.object(r.subprocess, 'run', side_effect=ldd) as run:
                self.assertEqual(r.linked_libraries(freeze), freeze['linked_library_bindings']); self.assertEqual(run.call_count, 2)
                second.write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'library map'): r.linked_libraries(freeze)

    def test_independent_scipy_runtime_exact_map_and_mutations(self):
        actual = r.independent_auditor_runtime(); freeze = {'independent_auditor_runtime': actual}
        self.assertEqual(set(actual['modules']), {'scipy', 'scipy.signal._upfirdn', 'scipy.signal._upfirdn_apply', 'scipy.special._ufuncs'})
        self.assertEqual(r.independent_auditor_runtime(freeze), actual)
        for key in actual['modules']:
            changed = copy.deepcopy(freeze); changed['independent_auditor_runtime']['modules'][key]['sha256'] = '0' * 64
            with self.subTest(module=key), self.assertRaisesRegex(ValueError, 'SciPy runtime'): r.independent_auditor_runtime(changed)
        changed = copy.deepcopy(freeze); changed['independent_auditor_runtime']['scipy'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'SciPy runtime'): r.independent_auditor_runtime(changed)
        with self.assertRaisesRegex(ValueError, 'SciPy runtime'): r.independent_auditor_runtime({})

    def test_audit_main_reconstructs_complete_summary_and_rejects_end_source_mutation(self):
        items, document = summary_fixture(); document['source_root'] = '/synthetic/no_reserved_source_access'
        summary = r.reconstruct_summary(items, document); sources = {'synthetic': {'sha256': '0' * 64}}
        commit = {'source_graph_sha256': r.value_hash(sources), 'measurement_counts': summary['measurement_counts'],
                  'summary_sha256': r.value_hash(summary), 'status': 'committed_reserved_measurements_complete_pending_independent_replay_not_admitted'}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'result'; root.mkdir(); (root / 'summary.json').write_bytes(r.canonical(summary))
            context = (document, {}, commit, {}, {})
            with patch.object(r, 'validate_authorities', return_value=context), patch.object(r, 'source_bindings', return_value=sources), \
                 patch.object(r, 'verify_item', side_effect=items), patch('builtins.print'):
                report = r.audit(root, 'synthetic', Path(tmp) / 'draft', Path(tmp) / 'freeze', 'synthetic',
                    Path(tmp) / 'audit.json', 'synthetic', Path(tmp) / 'tests', 'synthetic')
            self.assertEqual(report['planned_fixed_target_pools'], 2880); self.assertFalse(report['BC_admitted'])
            self.assertTrue(report['all_scientific_requirements_satisfied_after_replay'])
            with patch.object(r, 'validate_authorities', return_value=context), patch.object(r, 'source_bindings', side_effect=[sources, {'changed': True}]), \
                 patch.object(r, 'verify_item', side_effect=items), patch('builtins.print'):
                with self.assertRaisesRegex(ValueError, 'changed during audit'):
                    r.audit(root, 'synthetic', Path(tmp) / 'draft', Path(tmp) / 'freeze', 'synthetic',
                        Path(tmp) / 'must_not_exist.json', 'synthetic', Path(tmp) / 'tests', 'synthetic')
            self.assertFalse((Path(tmp) / 'must_not_exist.json').exists())


class ProducerSubprocessIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(); cls.root = Path(cls.temp.name)
        code = Path(r.__file__).parent
        recipe_path = code.parent / 'results/bc_codec_roundtrip_v1/results.json'
        recipe = json.loads(recipe_path.read_text())
        if not Path(recipe['toolchain']['ffmpeg']['path']).is_file():
            cls.temp.cleanup(); raise unittest.SkipTest('pinned remote FFmpeg required for isolated synthetic producer integration')
        # This child exclusively accesses newly generated synthetic PCM16, never the reserved roster/source root.
        script = r'''
import hashlib,json,sys
from pathlib import Path
import numpy as np
from scipy.io import wavfile
import run_bc_guitarset_reserved_admission_v1 as producer
root=Path(sys.argv[1]); code=Path(producer.__file__).parent
modules=producer.load_frozen(code)
package=json.loads((code.parent/'results/bc_codec_roundtrip_v1/results.json').read_text())
recipes={**package['codec_recipes'],'_ffmpeg':package['toolchain']['ffmpeg']['path']}
x=np.random.Generator(np.random.PCG64(901)).integers(-14000,14000,400001,dtype=np.int16)
source=root/'synthetic_native.wav'; wavfile.write(source,44100,x)
decoded=x.astype(np.float64)/32768.; sha=hashlib.sha256(decoded.astype('<f8').tobytes()).hexdigest()
frames=(len(x)*160+440)//441; start=(frames-128000)//2
plan={'native_frames':len(x),'native_sample_rate_hz':44100,'expected_resampled_frames':frames,
 'crop_start_sample':start,'crop_stop_sample_exclusive':start+128000,'padding_samples':0,
 'up':160,'down':441,'window':['kaiser',5.0],'padtype':'constant','cval':0.0,
 'resample_scope':'entire_native_recording_before_crop'}
row={'item_id':'synthetic_independent_audit_only','player_id':'00','score_id':'synthetic_score',
 'performance':'comp','style_from_score_prefix':'synthetic','split_role':'reserved',
 'source_audio':producer.binding(source),'historical_native_float64_pcm_sha256':sha,
 'historical_native_decode':{'format':'WAV','subtype':'PCM_16','sample_rate_hz':44100,'channels':1,
 'header_frames':len(x),'decoded_frames':len(x),'decoded_pcm_sha256':sha},'prospective_preprocessing':plan}
(root/'row.json').write_text(json.dumps(row)); (root/'toolchain.json').write_text(json.dumps({'toolchain':package['toolchain'],'codec_recipes':package['codec_recipes']}))
producer.process_item(row,root/'normal',modules,recipes)
save_arrays=modules['original'].save_arrays
def failed_persistence(path,arrays):
 if Path(path).name=='construction.arrays.npz':
  save_arrays(path,arrays)
  raise OSError('synthetic persistence failure after retained construction arrays')
 return save_arrays(path,arrays)
modules['original'].save_arrays=failed_persistence
producer.process_item(row,root/'construction_failure',modules,recipes)
modules['original'].save_arrays=save_arrays
def failed_codec(*args,**kwargs): raise OSError('synthetic encoder unavailable; no codec subprocess launched')
modules['codec'].command=failed_codec
producer.process_item(row,root/'codec_failure',modules,recipes)
from test_audit_bc_guitarset_reserved_admission_v1 import summary_fixture
synthetic_items,synthetic_document=summary_fixture()
producer.write_json(root/'actual_producer_summary_fixture.json',producer.assemble_summary(synthetic_items,synthetic_document,modules))
print('isolated synthetic normal/construction-failure/codec-failure graphs complete')
'''
        result = subprocess.run([sys.executable, '-c', script, str(cls.root)], cwd=code,
                                capture_output=True, text=True, timeout=180)
        if result.returncode:
            cls.temp.cleanup(); raise AssertionError(result.stdout + result.stderr)
        cls.row = json.loads((cls.root / 'row.json').read_text()); cls.freeze = json.loads((cls.root / 'toolchain.json').read_text())

    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()

    def test_actual_synthetic_producer_item_all16_independently_replayed(self):
        r.forbid_producer_imports()
        item = r.verify_item(self.row, self.root / 'normal', self.freeze)
        count = r.counts([item])['all']
        self.assertEqual(count, {'expected': 16, 'attempted': 16, 'successful': 16, 'failed_after_attempt': 0, 'skipped_before_attempt': 0})
        self.assertEqual(len(item['_codec_values']), 3); r.forbid_producer_imports()

    def test_actual_partial_construction_persistence_failure_accounts_all16(self):
        item = r.verify_item(self.row, self.root / 'construction_failure', self.freeze)
        self.assertTrue((self.root / 'construction_failure/construction.arrays.npz').exists())
        self.assertEqual(r.counts([item])['all'], {'expected': 16, 'attempted': 0, 'successful': 0, 'failed_after_attempt': 0, 'skipped_before_attempt': 16})
        self.assertEqual(item['construction']['status'], 'processing_failure')

    def test_actual_codec_failures_are_not_scientific_nulls(self):
        item = r.verify_item(self.row, self.root / 'codec_failure', self.freeze)
        self.assertEqual(r.counts([item])['all'], {'expected': 16, 'attempted': 10, 'successful': 10, 'failed_after_attempt': 0, 'skipped_before_attempt': 6})
        for row in item['_codec_values']:
            self.assertTrue(all(row['representations'][name] is None for name in r.CODECS))
            self.assertTrue(all(row['measurement_status'][name]['status'] == 'skipped_pipeline_failure' for name in r.CODECS))

    def test_isolated_real_producer_full_summary_schema_and_arithmetic_parity(self):
        items, document = summary_fixture()
        actual = json.loads((self.root / 'actual_producer_summary_fixture.json').read_text())
        expected = r.reconstruct_summary(items, document)
        r.a.compare(actual, expected, 'real isolated producer full summary parity')
        r.forbid_producer_imports()


if __name__ == '__main__': unittest.main()
