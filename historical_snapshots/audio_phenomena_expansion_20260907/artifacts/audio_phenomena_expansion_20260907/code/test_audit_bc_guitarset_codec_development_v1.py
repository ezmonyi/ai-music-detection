import importlib.util
import json
import pathlib
import struct
import tempfile
import unittest

import numpy as np


HERE = pathlib.Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('independent_codec_audit_under_test',
    HERE / 'audit_bc_guitarset_codec_development_v1.py')
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def riff(values, bits, *, extensible=False, subtype=None):
    values = np.asarray(values, dtype='<f4' if bits == 32 else '<f8')
    payload = values.tobytes()
    if extensible:
        guid = m.FLOAT_GUID if subtype is None else subtype
        fmt = struct.pack('<HHIIHHH', 0xfffe, 1, m.RATE, m.RATE * bits // 8,
                          bits // 8, bits, 22)
        fmt += struct.pack('<HI', bits, 0) + guid
    else:
        fmt = struct.pack('<HHIIHH', 3, 1, m.RATE, m.RATE * bits // 8, bits // 8, bits)
    body = b'fmt ' + struct.pack('<I', len(fmt)) + fmt
    body += b'fact' + struct.pack('<I', 4) + struct.pack('<I', len(values))
    body += b'data' + struct.pack('<I', len(payload)) + payload
    return b'RIFF' + struct.pack('<I', len(body) + 4) + b'WAVE' + body


def reduced(values):
    pools = [{'pool_index': i, 'start_sample': i * m.POOL_SAMPLES,
              'stop_sample_exclusive': (i + 1) * m.POOL_SAMPLES,
              'source_start_sample': i * m.POOL_SAMPLES,
              'source_stop_sample_exclusive': (i + 1) * m.POOL_SAMPLES,
              'pool_status': 'ok', 'target_status': 'ok' if value is not None else 'below_energy_fraction_floor',
              'eligible': value is not None, 'squared_bicoherence': value}
             for i, value in enumerate(values)]
    finite = [value for value in values if value is not None]
    return {'version': 'bicoherence_scalar_v1', 'target_frequency_bins': [32, 48, 80],
        'target_frequency_hz': [500, 750, 1250],
        'crop': {'start_sample': 0, 'stop_sample_exclusive': 128000,
                 'source_resampled_samples': 128000},
        'input_samples': 128000, 'pool_count': 2, 'analyzed_samples': 128000,
        'discarded_tail_samples': 0, 'eligible_pool_count': len(finite),
        'missing_pool_count': 2 - len(finite),
        'eligibility_mask': [value is not None for value in values],
        'status': 'ok' if finite else 'no_eligible_target_pools',
        'median_squared_bicoherence': float(np.median(finite)) if finite else None,
        'pools': pools, 'construction_status': None, 'null_calibrated': False,
        'significance_inferred': False, 'external_validation_passed': False,
        'classifier_admitted': False}


def measurement_fixture(root, samples):
    reduction, pools = m.reduce_waveform(samples.astype(np.float64), None)
    grid = np.asarray([(a, b, a + b) for a in range(8, 193, 8)
                       for b in range(a, 193, 8) if a + b <= 256], dtype=np.int64)
    target_index = int(np.flatnonzero(np.all(grid == np.asarray(m.TARGET), axis=1))[0])
    metadata_pools = []
    spectra = np.empty((2, m.FRAMES, 513), dtype=np.complex128)
    means = np.empty((2, m.FRAMES), dtype=np.float64)
    energy = np.empty((2, 513), dtype=np.float64)
    totals = np.empty(2, dtype=np.float64)
    fractions = np.empty((2, 513), dtype=np.float64)
    floor = np.zeros((2, 228), dtype=bool)
    primitive = np.zeros((2, 228), dtype=bool)
    eligible = np.zeros((2, 228), dtype=bool)
    scores = np.full((2, 228), np.nan)
    raw_scores = np.full((2, 228), np.nan)
    for index, pool in enumerate(pools):
        spectra[index], means[index], energy[index] = pool['spectra'], pool['means'], pool['energy']
        totals[index], fractions[index] = pool['total_non_dc_coefficient_energy'], pool['fractions']
        floor[index, target_index] = pool['energy_floor_passed']
        primitive[index, target_index] = pool['primitive_status'] == 'ok'
        eligible[index, target_index] = pool['eligible']
        if pool['squared_bicoherence'] is not None:
            scores[index, target_index] = pool['squared_bicoherence']
        if pool['primitive_squared_bicoherence'] is not None:
            raw_scores[index, target_index] = pool['primitive_squared_bicoherence']
        cells = [{'frequency_bins': bins.tolist()} for bins in grid]
        cells[target_index] = {'frequency_bins': list(m.TARGET), 'status': pool['target_status'],
            'energy_fractions': pool['target_energy_fractions'],
            'energy_floor_passed': pool['energy_floor_passed'], 'eligible': pool['eligible'],
            'squared_bicoherence': pool['squared_bicoherence'],
            'primitive': {'status': pool['primitive_status'],
                          'squared_bicoherence': pool['primitive_squared_bicoherence']}}
        metadata_pools.append({'pool_index': index, 'coefficient_rows': m.FRAMES,
            'status': pool['pool_status'], 'cells': cells})
    metadata = {'version': 'bicoherence_audio_v1', 'primitive_version': 'bicoherence_primitive_v2',
        'input_samples': m.SAMPLES, 'sample_rate_hz': m.RATE, 'pool_samples': m.POOL_SAMPLES,
        'pool_count': 2, 'discarded_tail_samples': 0, 'n_fft': m.N_FFT, 'hop_samples': m.HOP,
        'frames_per_pool': m.FRAMES, 'energy_fraction_min_inclusive': 1e-6,
        'construction_status': None, 'pools': metadata_pools}
    (root / 'x.metadata.json').write_text(json.dumps(metadata, allow_nan=False))
    np.savez_compressed(root / 'x.arrays.npz', window=pools[0]['window'],
        frame_offset_samples=pools[0]['offsets'], frame_means=means, spectra=spectra,
        bin_coefficient_energy=energy, total_non_dc_coefficient_energy=totals,
        bin_energy_fraction=fractions, frequency_bins=grid, energy_floor_mask=floor,
        primitive_defined_mask=primitive, eligible_mask=eligible,
        squared_bicoherence=scores, primitive_squared_bicoherence=raw_scores)
    return reduction, metadata


class IndependentCodecAuditTests(unittest.TestCase):
    def test_frozen_authorities_and_scope(self):
        self.assertEqual(m.PRODUCER_SHA,
            '2c607b1175694b6966b3ce5200c6e306f20c40dd00de8f0229e6236387cc013e')
        self.assertEqual(m.DRAFT_SHA,
            '24d0544582d0992441b856bc00594f52082a97fb6dfce2780e6825bca29aa8f7')
        self.assertEqual(m.FREEZE_SHA,
            '9405118a484d5578b73caff964538571920ae786ea9329e8aa846408ef85e53b')
        self.assertFalse(m.SCOPE['reserved_recordings_read'])
        self.assertFalse(m.SCOPE['BC_admitted'])
        source = (HERE / 'audit_bc_guitarset_codec_development_v1.py').read_text()
        for forbidden in ('import run_bc_guitarset_codec_development_v1',
                          'import bicoherence_audio_v1', 'import bicoherence_scalar_v1',
                          'import bicoherence_primitive_v2'):
            self.assertNotIn(forbidden, source)

    def test_riff_float32_and_float64_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            f32 = np.linspace(-1.25, 1.25, m.SAMPLES, dtype=np.float32)
            f64 = np.linspace(-2, 2, m.SAMPLES, dtype=np.float64)
            (root / 'f32.wav').write_bytes(riff(f32, 32))
            (root / 'f64.wav').write_bytes(riff(f64, 64))
            a, ai = m.read_ieee_float_wav(root / 'f32.wav', bits=32)
            b, bi = m.read_ieee_float_wav(root / 'f64.wav', bits=64)
            self.assertTrue(np.array_equal(a, f32))
            self.assertTrue(np.array_equal(b, f64))
            self.assertEqual(ai['pcm_sha256'], m.hashlib.sha256(f32.astype('<f4').tobytes()).hexdigest())
            self.assertEqual(bi['bits'], 64)

    def test_extensible_float_subtype_and_length_are_strict(self):
        samples = np.zeros(m.SAMPLES, dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / 'x.wav'
            path.write_bytes(riff(samples, 32, extensible=True))
            observed, info = m.read_ieee_float_wav(path, bits=32)
            self.assertEqual(info['format_tag'], 0xfffe)
            self.assertTrue(np.array_equal(observed, samples))
            path.write_bytes(riff(samples, 32, extensible=True, subtype=b'\0' * 16))
            with self.assertRaisesRegex(ValueError, 'extensible'):
                m.read_ieee_float_wav(path, bits=32)
            path.write_bytes(riff(samples[:-1], 32))
            with self.assertRaisesRegex(ValueError, '128000'):
                m.read_ieee_float_wav(path, bits=32)

    def test_exact_float64_to_float32_cast_preserves_out_of_unit_values(self):
        source = np.linspace(-1.5, 1.5, m.SAMPLES, dtype=np.float64)
        cast = source.astype(np.float32)
        self.assertGreater(float(np.max(np.abs(cast))), 1.0)
        self.assertTrue(np.array_equal(cast.astype(np.float64), source.astype(np.float32).astype(np.float64)))

    def test_fixed_harmonic_target_is_eligible_near_one(self):
        t = np.arange(m.SAMPLES, dtype=np.float64) / m.RATE
        samples = (.2 * np.cos(2 * np.pi * 500 * t + .2)
                   + .2 * np.cos(2 * np.pi * 750 * t + .7)
                   + .2 * np.cos(2 * np.pi * 1250 * t + .9))
        reduction, pools = m.reduce_waveform(samples, None)
        self.assertEqual(reduction['eligibility_mask'], [True, True])
        self.assertGreater(reduction['median_squared_bicoherence'], .999999999)
        self.assertEqual([pool['spectra'].shape for pool in pools], [(247, 513), (247, 513)])

    def test_silence_is_scientific_missingness_not_failure_or_zero(self):
        reduction, pools = m.reduce_waveform(np.zeros(m.SAMPLES), None)
        self.assertEqual(reduction['status'], 'no_eligible_target_pools')
        self.assertIsNone(reduction['median_squared_bicoherence'])
        self.assertEqual(reduction['eligibility_mask'], [False, False])
        self.assertEqual([pool['pool_status'] for pool in pools], ['zero_amplitude', 'zero_amplitude'])

    def test_raw_metadata_and_target_arrays_replay_and_mask_mutation(self):
        t = np.arange(m.SAMPLES, dtype=np.float32) / np.float32(m.RATE)
        samples = (np.float32(.2) * np.cos(np.float32(2 * np.pi * 500) * t)
                 + np.float32(.2) * np.cos(np.float32(2 * np.pi * 750) * t)
                 + np.float32(.2) * np.cos(np.float32(2 * np.pi * 1250) * t)).astype(np.float32)
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            reduction, metadata = measurement_fixture(root, samples)
            actual, _ = m.verify_measurement(samples, root / 'x', reduction)
            m.compare(actual, reduction)
            with np.load(root / 'x.arrays.npz', allow_pickle=False) as stored:
                arrays = {key: stored[key] for key in stored.files}
            target_index = int(np.flatnonzero(np.all(
                arrays['frequency_bins'] == np.asarray(m.TARGET), axis=1))[0])
            arrays['primitive_defined_mask'][0, target_index] = not bool(
                arrays['primitive_defined_mask'][0, target_index])
            np.savez_compressed(root / 'x.arrays.npz', **arrays)
            with self.assertRaisesRegex(ValueError, 'primitive-defined'):
                m.verify_measurement(samples, root / 'x', reduction)
            reduction, metadata = measurement_fixture(root, samples)
            target = next(cell for cell in metadata['pools'][0]['cells']
                          if cell['frequency_bins'] == list(m.TARGET))
            target['eligible'] = not target['eligible']
            (root / 'x.metadata.json').write_text(json.dumps(metadata, allow_nan=False))
            with self.assertRaisesRegex(ValueError, 'eligible'):
                m.verify_measurement(samples, root / 'x', reduction)

    def test_nested_numerical_comparison_tolerates_roundoff_but_binding_mutation_fails(self):
        m.compare({'value': 0.25 + 1e-14, 'mask': [True]},
                  {'value': 0.25, 'mask': [True]}, 'near equal')
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / 'bound.bin'
            path.write_bytes(b'first')
            entry = m.binding(path.resolve())
            path.write_bytes(b'second')
            with self.assertRaisesRegex(ValueError, 'binding changed'):
                m.check_file_bindings(entry)

    def test_precision_and_codec_directions_and_nulls(self):
        first, second = reduced([.1, None]), reduced([.3, .5])
        precision = m.compare_scalars(first, second, 'float32_minus_float64')
        codec = m.compare_scalars(second, first, 'decoded_minus_float32')
        self.assertAlmostEqual(precision['scalar_delta'], .3)
        self.assertAlmostEqual(codec['scalar_delta'], -.3)
        self.assertEqual(precision['mask_transition_counts']['missing_to_eligible'], 1)
        missing = m.compare_scalars(reduced([None, None]), reduced([None, None]),
                                    'decoded_minus_float32')
        self.assertEqual(missing['status'], 'compared_not_thresholded')
        self.assertIsNone(missing['scalar_delta'])

    def test_operational_and_paired_contrasts_remain_distinct(self):
        result = m.condition_difference(reduced([.9, .1]), reduced([.4, None]))
        self.assertAlmostEqual(result['operational_median_difference'], .1)
        self.assertAlmostEqual(result['paired_pool_mean_difference'], .5)
        self.assertEqual(result['paired_pool_count'], 1)

    def test_outcome_counts_reject_fake_or_inconsistent_success(self):
        receipt = {'measurement_status': {
            'float32_control': {'attempted': True, 'status': 'success'},
            'mp3_128k': {'attempted': True, 'status': 'measurement_failure'},
            'opus_96k': {'attempted': False, 'status': 'skipped_dependency_failure'}}}
        self.assertEqual(m.outcome_counts([receipt]), {'expected': 3, 'attempted': 2,
            'successful': 1, 'failed_after_attempt': 1, 'skipped_before_attempt': 1})
        receipt['measurement_status']['opus_96k'] = {'attempted': False, 'status': 'success'}
        with self.assertRaisesRegex(ValueError, 'consistency'):
            m.outcome_counts([receipt])

    def test_delta_statistics_fixed_quantile_and_denominators(self):
        comparisons = [m.compare_scalars(reduced([.1, .2]), reduced([.2, .4]),
                                         'decoded_minus_float32'),
                       m.compare_scalars(reduced([None, None]), reduced([None, None]),
                                         'decoded_minus_float32'),
                       m.compare_scalars(reduced([.1, None]), None, 'decoded_minus_float32')]
        result = m.delta_stats(comparisons)
        self.assertEqual(result['comparison_denominator'], 3)
        self.assertEqual(result['scalar_pair_covered'], 1)
        self.assertEqual(result['scientific_null_pairs'], 1)
        self.assertEqual(result['processing_failure_comparisons'], 1)
        self.assertEqual(result['p95_quantile_method'], 'linear interpolation at h=(n-1)*0.95')

    def test_codec_commands_are_exact_frozen_recipes(self):
        draft = {'policy': {'codec_recipes': {'mp3_128k': {'extension': '.mp3',
            'encoder': 'libmp3lame', 'bitrate': '128k',
            'encoder_options': ['-reservoir', '1', '-abr', '0']}}},
            'toolchain': {'ffmpeg': {'path': '/pinned/ffmpeg'}}}
        row = {'item_id': 'id', 'condition': 'baseline'}
        encode, decode, encoded, decoded = m.expected_codec_commands(
            draft, row, pathlib.Path('/out'), 'mp3_128k')
        self.assertIn('libmp3lame', encode)
        self.assertEqual(encode[-1], '/out/items/id/baseline/mp3_128k.mp3')
        self.assertEqual(decode[-1], '/out/items/id/baseline/mp3_128k.decoded.wav')
        self.assertEqual(encoded.suffix, '.mp3')
        self.assertEqual(decoded.name, 'mp3_128k.decoded.wav')

    def test_summary_reconstruction_has_all_fixed_denominators_and_groups(self):
        baseline, closed, independent = reduced([.1, .2]), reduced([.8, .9]), reduced([.2, .3])
        receipts = []
        for index in range(90):
            identity = {'item_id': f'i{index:03}', 'player_id': f'p{index % 3}',
                        'score_id': f's{index % 15}',
                        'performance': 'comp' if index % 2 == 0 else 'solo'}
            for condition, value in (('baseline', baseline), ('closed_minus6db', closed),
                                     ('independent_minus6db', independent)):
                receipts.append(identity | {'condition': condition,
                    'representations': {name: value for name in m.REPRESENTATIONS},
                    'measurement_status': {name: {'attempted': True, 'status': 'success'}
                        for name in ('float32_control', *m.CODECS)},
                    'comparisons': {'float32_minus_float64': m.compare_scalars(
                        value, value, 'float32_minus_float64'),
                        'decoded_minus_float32': {codec: m.compare_scalars(
                            value, value, 'decoded_minus_float32') for codec in m.CODECS}}})
        summary = m.reconstruct_summary(receipts)
        self.assertEqual(summary['new_BC_measurements'], {'expected': 810, 'attempted': 810,
            'successful': 810, 'failed_after_attempt': 0, 'skipped_before_attempt': 0})
        self.assertEqual(summary['baseline']['representation_coverage']['mp3_128k']
                         ['pool_denominator'], 180)
        for representation in m.REPRESENTATIONS:
            block = summary['condition_contrasts'][representation]
            self.assertEqual(block['recording_denominator'], 90)
            self.assertEqual(block['paired_pool_denominator'], 180)
            self.assertEqual(block['metrics']['operational_median_difference']
                             ['equal_score']['group_denominator'], 15)
            self.assertEqual(block['metrics']['paired_pool_mean_difference']
                             ['equal_player_secondary']['group_denominator'], 3)


if __name__ == '__main__':
    unittest.main()
