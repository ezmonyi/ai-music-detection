"""Synthetic fixtures only. Never open actual cohort or reserved waveforms."""
import copy
from contextlib import nullcontext
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

import run_native30_bc_v2 as m


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    m.base.write_new(path, value)
    return m.base.file_binding(path)


def cohort_fixture(root):
    """Actual frozen graph checker, synthetic JSON graph; opaque fake WAV bytes."""
    from test_run_native30_fhsc_cohort_v1 import document_fixture
    plan, screen, new, prior, expected = document_fixture()
    original, old, closure = (root / name for name in ('original', 'prior', 'closure'))
    for output, manifest in ((original, new), (old, prior)):
        for r in manifest['records']:
            path = output / 'audio' / (r['id'] + '.wav'); path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'SYNTHETIC_METADATA_ONLY_NO_AUDIO_DECODER')
            bound = m.base.file_binding(path)
            r.update(standardized_path=str(path), file_bytes=bound['bytes'], file_sha256=bound['sha256'])
        contract = put(output / 'contract.json', {'rows': [r['row'] for r in manifest['records']]})
        manifest['contract_sha256'] = contract['sha256']
        for r in manifest['records']:
            r['contract_sha256'] = contract['sha256']
            put(output / 'items' / (r['id'] + '.json'), {'payload': r, 'receipt_sha256': m.base.value_hash(r)})
    new['original_root'] = str(original)
    put(old / 'manifest.json', prior)
    def product_map(path):
        return {name: {k: v for k, v in m.base.file_binding(path / name).items() if k != 'path'}
                for name in sorted(m.physical.tree_files(path))}
    old_commit = put(old / 'COMMIT.json', {'status': 'committed_prior2174_DSP_only', 'completed': 1, 'products': product_map(old)})
    put(closure / 'manifest.json', new); put(closure / 'summary.json', {'synthetic': True})
    upstream = {'bindings': [m.base.file_binding(original / name) for name in sorted(m.physical.tree_files(original))]}
    put(closure / 'upstream_bindings.json', upstream)
    new_commit = put(closure / 'COMMIT.json', {'version': m.physical.CLOSURE_VERSION, 'status': m.physical.CLOSURE_STATUS,
        'eligible': 1, 'excluded_short': 1, 'original_root': str(original),
        'all_bound_inputs_and_products_end_rehashed': True, 'products': product_map(closure)})
    plan_entry = put(root / 'plan.json', plan); screen_entry = put(root / 'screen.json', screen)
    rows = m.physical.reconcile(plan, screen, new, prior, expected)
    bindings = {e['path']: e for e in (plan_entry, screen_entry)}
    for output in (original, old, closure):
        bindings.update({str(output / n): m.base.file_binding(output / n) for n in m.physical.tree_files(output)})
    cohort = {'version': m.physical.VERSION, 'status': 'frozen_before_any_measurement_audio_reads',
        'input_format': m.INPUT_FORMAT, 'expected_count': 2, 'human': 1, 'ai': 1, 'source_counts': expected['sources'],
        'rows': rows, 'bindings': bindings, 'output_root': str(root / 'unused_fhsc'),
        'upstream_commits': {k: {x: e[x] for x in ('path', 'sha256')} for k, e in (('new', new_commit), ('prior', old_commit))},
        'upstream_inventories': {str(o): sorted(m.physical.tree_files(o)) for o in (original, old, closure)}}
    entry = put(root / 'cohort.json', cohort)
    return {'cohort_contract': entry, 'plan': plan_entry, 'screen': screen_entry}, expected


class AuthorityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()

    def test_pinned_numerical_modules_and_wrong_pin(self):
        draft, audio, scalar = m.numerical_modules()
        self.assertEqual(scalar.TARGET, (32, 48, 80))
        self.assertEqual(audio.FRAMES_PER_POOL, 247)
        self.assertEqual(draft.CROP_SAMPLES, 128000)
        with patch.dict(m.PINS, {'bicoherence_scalar_v1': '0' * 64}), self.assertRaisesRegex(ValueError, 'pinned source'):
            m.numerical_modules()

    def test_actual_metadata_graph_without_waveform_reads(self):
        request, expected = cohort_fixture(self.root)
        original = m.base.digest
        def no_audio(path):
            if Path(path).suffix == '.wav':
                raise AssertionError('preflight read audio bytes')
            return original(path)
        with patch.object(m.base, 'digest', side_effect=no_audio), patch.object(m.graph, 'digest', side_effect=no_audio), \
                patch.object(sf, 'SoundFile', side_effect=AssertionError('preflight decode')):
            cohort, native = m.load_cohort(request, expected=expected,
                plan_sha=request['plan']['sha256'], screen_sha=request['screen']['sha256'])
        self.assertEqual(len(cohort['rows']), 2)
        self.assertEqual({x['sample_rate_hz'] for x in native.values()}, {44100})

    def test_source_graph_changed_identity_rejected(self):
        request, expected = cohort_fixture(self.root)
        c = m.base.read_json(request['cohort_contract']['path']); c['rows'][0]['component_id'] = 'forged'
        request['cohort_contract'] = put(self.root / 'forged.json', c)
        with self.assertRaisesRegex(ValueError, 'do not replay'):
            m.load_cohort(request, expected=expected, plan_sha=request['plan']['sha256'], screen_sha=request['screen']['sha256'])

    def test_fixed_source_graph_pin_required(self):
        request, expected = cohort_fixture(self.root)
        with self.assertRaisesRegex(ValueError, 'fixed SHA'):
            m.load_cohort(request, expected=expected)

    def test_real_source_graph_extra_unbound_product_rejected(self):
        request, expected = cohort_fixture(self.root)
        put(self.root / 'prior/extra.json', {'unbound': True})
        with self.assertRaisesRegex(ValueError, 'exhaustive product inventory'):
            m.load_cohort(request, expected=expected, plan_sha=request['plan']['sha256'], screen_sha=request['screen']['sha256'])

    def test_linux_runtime_binds_numerical_packages_and_live_sndfile(self):
        with patch.object(sf, 'SoundFile', side_effect=AssertionError('runtime opened audio')):
            runtime = m.runtime_snapshot()
        self.assertEqual(runtime['numpy'], '1.26.4')
        self.assertEqual(runtime['scipy'], '1.17.1')
        self.assertEqual(runtime['soundfile'], '0.14.0')
        self.assertTrue(runtime['package_files'])
        self.assertEqual(runtime['libsndfile_resolution']['method'], 'soundfile_CFFI_live_symbol_linux_proc_maps')
        self.assertEqual(set(runtime['import_origins']), {'numpy', 'scipy', 'soundfile', 'cffi', '_cffi_backend'})

    def gate_fixture(self, gate=True):
        from test_summarize_bc_reserved_admission_v2 import CorrectionFixture
        fixture = CorrectionFixture(gate=gate); self.addCleanup(fixture.close)
        decision = {'version': 'bc_native30_transfer_decision_v1',
            'status': 'external_measurement_gate_accepted_prospective_transfer_implementation_only',
            'scope': m.DECISION_SCOPE, 'prospective_cohort_rows': 3830, 'prospective_feature': m.FEATURE,
            'independent_audit': {k: fixture.audit_entry[k] for k in ('path', 'sha256')},
            'producer_commit_sha256': fixture.old.commit_entry['sha256'], 'tolerances_and_scientific_margins_unchanged': True}
        request = {'audit': fixture.audit_entry, 'producer_commit': fixture.old.commit_entry,
                   'decision': put(self.root / 'decision.json', decision)}
        return request

    def gate_patches(self, request):
        from contextlib import ExitStack
        stack = ExitStack()
        for key, entry in (('AUDIT_SHA', 'audit'), ('DECISION_SHA', 'decision'), ('RESERVED_COMMIT_SHA', 'producer_commit')):
            stack.enter_context(patch.object(m, key, request[entry]['sha256']))
        return stack

    def test_real_gate_loader_checks_terminal_counts_and_all_margins(self):
        request = self.gate_fixture()
        with self.gate_patches(request):
            result = m.accepted_gate(request)
        self.assertTrue(result['scientific_gate_satisfied'])
        self.assertFalse(result['actual_execution_authorized'])

    def test_corrected_reporter_module_origin_version_and_proof_contract(self):
        reporter = m.module('summarize_bc_reserved_admission_v2')
        self.assertEqual(reporter.VERSION, 'summarize_bc_reserved_admission_v2')
        self.assertEqual(Path(reporter.__file__).resolve(),
                         m.HERE / 'summarize_bc_reserved_admission_v2.py')
        self.assertIn('summarize_bc_reserved_admission_v2', m.PINS)
        self.assertNotIn('summarize_bc_reserved_admission_v1', m.PINS)
        actual_audit = m.HERE.parent / 'audit/bc_reserved_independent_actual_v2.json'
        self.assertEqual(m.base.file_binding(actual_audit)['sha256'], m.AUDIT_SHA)
        self.assertEqual(m.base.read_json(actual_audit)['version'], reporter.AUDITOR_VERSION)
        request = self.gate_fixture()
        with self.gate_patches(request):
            proof = reporter.load_sources(request['producer_commit']['path'],
                                          request['producer_commit']['sha256'],
                                          request['audit']['path'], request['audit']['sha256'])
            result = m.accepted_gate(request)
        self.assertTrue({'summary_binding', 'complete_measurements',
                         'scientific_gate_satisfied', 'correction_lineage'} <= set(proof))
        self.assertEqual(result['summary'], proof['summary_binding'])
        self.assertTrue(proof['complete_measurements'])
        self.assertTrue(proof['scientific_gate_satisfied'])

    def test_audit_pass_but_scientific_margin_failure_rejected(self):
        request = self.gate_fixture(gate=False)
        with self.gate_patches(request), self.assertRaisesRegex(ValueError, 'not a passed scientific gate'):
            m.accepted_gate(request)

    def test_actual_gate_pin_and_decision_scope_are_required(self):
        request = self.gate_fixture()
        with self.assertRaisesRegex(ValueError, 'exact pins'):
            m.accepted_gate(request)
        value = m.base.read_json(request['decision']['path']); value['scope'] = {**m.DECISION_SCOPE, 'actual_cohort_measurement_authorized_by_this_file': True}
        request['decision'] = put(self.root / 'decision_wrong_scope.json', value)
        with self.gate_patches(request), self.assertRaisesRegex(ValueError, 'implementation-only'):
            m.accepted_gate(request)

    def test_missing_or_wrong_freeze_never_reaches_audio(self):
        entry = put(self.root / 'not_a_freeze.json', {'version': m.VERSION})
        with patch.object(m, 'decode', side_effect=AssertionError('audio')):
            with self.assertRaisesRegex(ValueError, 'parent measurement freeze'):
                m.validate_freeze(entry['path'], entry['sha256'])
            with self.assertRaisesRegex(ValueError, 'caller parent-freeze pin'):
                m.validate_freeze(entry['path'], '0' * 64)

    def test_separate_positive_freeze_and_changed_prepared_contract(self):
        decision = put(self.root / 'implementation_decision.json', {'only': 'synthetic'})
        prepared = {'request': {'decision': decision}, 'runtime': {'synthetic': True},
                    'output_root': str(self.root / 'out')}
        prepared_entry = put(self.root / 'prepared.json', prepared)
        freeze = {'version': m.FREEZE_VERSION, 'status': 'parent_frozen_for_native30_BC_measurement',
            'measurement_authorized': True, 'prepared_contract': prepared_entry,
            'runner': m.base.file_binding(m.HERE / (m.VERSION + '.py')),
            'tests': m.base.file_binding(m.HERE / ('test_' + m.VERSION + '.py')),
            'decision': decision, 'view': m.VIEW, 'runtime': prepared['runtime'], **m.SCOPE}
        entry = put(self.root / 'freeze.json', freeze)
        with patch.object(m, 'prepare', return_value=prepared), patch.object(m, 'decode', side_effect=AssertionError('audio')):
            result = m.validate_freeze(entry['path'], entry['sha256'])
        self.assertEqual(result['status'], 'parent_authorized_BC_measurement_only')
        with patch.object(m, 'prepare', return_value={**prepared, 'changed': True}), self.assertRaisesRegex(ValueError, 'does not replay'):
            m.validate_freeze(entry['path'], entry['sha256'])
        freeze['measurement_authorized'] = False
        entry = put(self.root / 'freeze_not_authorized.json', freeze)
        with self.assertRaisesRegex(ValueError, 'separate parent measurement freeze'):
            m.validate_freeze(entry['path'], entry['sha256'])

    def test_preflight_prepare_no_audio_and_no_output_creation(self):
        request, expected = cohort_fixture(self.root)
        cohort, native = m.load_cohort(request, expected=expected,
            plan_sha=request['plan']['sha256'], screen_sha=request['screen']['sha256'])
        request.update(self.gate_fixture()); output = self.root / 'new_output'
        with self.gate_patches(request), patch.object(m, 'load_cohort', return_value=(cohort, native)), \
                patch.object(m, 'runtime_snapshot', return_value={'synthetic': True}), \
                patch.object(sf, 'SoundFile', side_effect=AssertionError('audio')):
            prepared = m.prepare(request, output)
        self.assertEqual(prepared['audio_reads_in_preflight'], 0)
        self.assertEqual(prepared['view'], m.VIEW)
        self.assertFalse(output.exists())


class NumericalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        t = np.arange(1323000, dtype=np.float64) / 44100
        mono = .2 * np.cos(2*np.pi*500*t) + .16*np.cos(2*np.pi*750*t + .1) + .12*np.cos(2*np.pi*1250*t + .1)
        mono += .015*np.sin(2*np.pi*110*t) + .031
        # Quantize to actual stored FLOAT precision before exact float64 promotion.
        cls.stereo = np.repeat(mono[:, None], 2, axis=1).astype('float32').astype('float64')
        cls.wave, cls.view = m.analysis_view(cls.stereo)
        _, audio, scalar = m.numerical_modules()
        cls.measured = audio.extract(cls.wave, 16000)
        cls.measured['metadata']['construction_status'] = cls.view['construction_status']
        cls.scalar = scalar.reduce_metadata(cls.measured['metadata'], crop=m.CROP)

    def test_exact_external_standardize_and_baseline_parity(self):
        import bicoherence_guitarset_pilot_v1 as pilot
        self.assertEqual(m.base.digest(pilot.__file__), 'dcb8bfd41b89e3ab59bd241214ab64e36dbb7a5264409898bf3e423a03de8d6e')
        draft, _, scalar = m.numerical_modules()
        crop, p = draft.standardize(self.stereo[:, 0], 44100)
        baseline = pilot.constructions(crop, 'synthetic')['waveforms']['baseline']
        np.testing.assert_array_equal(self.wave, baseline)
        self.assertEqual((p['crop_start'], p['crop_stop_exclusive'], p['resampled_frames']), (176000, 304000, 480000))
        self.assertEqual(self.scalar['eligible_pool_count'], 2)
        self.assertGreater(self.scalar['median_squared_bicoherence'], .9)
        self.assertEqual(scalar.reduce_metadata(self.measured['metadata'], crop=m.CROP), self.scalar)
        m.validate_arrays(self.measured['arrays'], self.measured['metadata'])

    def test_no_global_dc_subtraction_and_no_normalization(self):
        self.assertGreater(float(self.wave.mean()), .007)
        self.assertFalse(self.view['global_dc_subtraction'])
        self.assertEqual(self.view['baseline_gain'], .25)
        self.assertEqual(self.view['pool_count'], 2)

    def test_channel_swap_and_antiphase_cancellation_no_fallback(self):
        swapped, _ = m.analysis_view(self.stereo[:, ::-1])
        np.testing.assert_array_equal(swapped, self.wave)
        cancelled, view = m.analysis_view(np.column_stack((self.stereo[:, 0], -self.stereo[:, 0])))
        self.assertFalse(np.any(cancelled))
        self.assertEqual(view['construction_status'], 'unsupported_zero_background_rms')
        _, audio, scalar = m.numerical_modules()
        reduced = scalar.reduce_metadata(audio.extract(cancelled, 16000)['metadata'], crop=m.CROP)
        self.assertIsNone(reduced['median_squared_bicoherence'])

    def test_one_eligible_pool_retained_and_no_complete_pool_rejected(self):
        _, audio, scalar = m.numerical_modules()
        wave = self.wave.copy(); wave[64000:] = 0
        metadata = audio.extract(wave, 16000)['metadata']
        value = scalar.reduce_metadata(metadata, crop=m.CROP)
        self.assertEqual(value['eligible_pool_count'], 1)
        self.assertEqual(value['median_squared_bicoherence'], value['pools'][0]['squared_bicoherence'])
        metadata['pool_count'] = 0
        with self.assertRaises(ValueError):
            scalar.reduce_metadata(metadata, crop=m.CROP)

    def test_exact_energy_floor_inclusive_and_missing_not_failure(self):
        scalar = m.numerical_modules()[2]
        grid_index = [c['frequency_bins'] for c in self.measured['metadata']['pools'][0]['cells']].index([32, 48, 80])
        for fraction, available in ((1e-6, 2), (np.nextafter(1e-6, 0.0), 0)):
            metadata = copy.deepcopy(self.measured['metadata'])
            for pool in metadata['pools']:
                cell = pool['cells'][grid_index]; cell['energy_fractions'] = [float(fraction)] * 3
                cell['energy_floor_passed'] = cell['eligible'] = available == 2
                cell['status'] = 'ok' if available else 'below_energy_fraction_floor'
                if not available:
                    cell['squared_bicoherence'] = None
            result = scalar.reduce_metadata(metadata, crop=m.CROP)
            self.assertEqual(result['eligible_pool_count'], available)
            self.assertEqual(result['median_squared_bicoherence'] is None, available == 0)

    def test_boundary_resample_before_crop_not_crop_before_resample(self):
        stereo = np.zeros_like(self.stereo)
        stereo[11*44100-1] = 1
        wave, _ = m.analysis_view(stereo)
        self.assertTrue(np.any(wave[:32]))
        from scipy.signal import resample_poly
        cropped_first = .25*resample_poly(stereo[11*44100:19*44100].mean(axis=1), 160, 441, window=('kaiser', 5.0), padtype='constant', cval=0)
        self.assertFalse(np.array_equal(wave, cropped_first))

    def test_malformed_nonfinite_and_wrong_duration_fail_closed(self):
        for samples in (self.stereo[:-1], self.stereo.astype('float32')):
            with self.assertRaises(ValueError):
                m.analysis_view(samples)
        samples = self.stereo.copy(); samples[5, 0] = np.nan
        with self.assertRaises(ValueError):
            m.analysis_view(samples)
        metadata = copy.deepcopy(self.measured['metadata']); metadata['pools'][0]['total_non_dc_coefficient_energy'] = float('nan')
        with self.assertRaises(FloatingPointError):
            m.numerical_modules()[2].reduce_metadata(metadata, crop=m.CROP)


class TransactionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        NumericalTests.setUpClass()
        cls.stereo = NumericalTests.stereo
        cls.cached = (NumericalTests.view, NumericalTests.measured['metadata'], NumericalTests.measured['arrays'], NumericalTests.scalar)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve(); self.output = self.root / 'result'
        path = self.root / 'synthetic.wav'; sf.write(path, self.stereo, 44100, subtype='FLOAT')
        self.row = {'id': 'synthetic', 'source_group': 'Synthetic', 'label': '0', 'role': 'development',
                    'group_id': 'g', 'component_id': 'c', 'input': m.base.file_binding(path),
                    'waveform_float32_sha256': hashlib.sha256(self.stereo.astype('<f4').tobytes()).hexdigest()}
        self.origin = {'sample_rate_hz': 48000, 'sha256': 'a'*64}
        self.run = {'prepared': {'output_root': str(self.output), 'expected_count': 1,
                    'rows': [self.row], 'native_origins': {'synthetic': self.origin}}}
        self.check = patch.object(m, 'check_inputs'); self.check.start(); self.addCleanup(self.check.stop)
        self.lock = patch.object(m, 'source_locks', return_value=nullcontext()); self.lock.start(); self.addCleanup(self.lock.stop)

    def run_cached(self):
        with patch.object(m, 'measure', return_value=copy.deepcopy(self.cached)):
            return m.run_contract(self.run)

    def test_real_synthetic_decode_measure_commit_and_verified_resume(self):
        result = m.run_contract(self.run)
        self.assertEqual(result['count'], 1)
        before = m.products(self.output)
        with patch.object(m, 'measure', side_effect=AssertionError('resume measured BC')):
            again = m.run_contract(self.run)
        self.assertEqual(again, result)
        self.assertEqual(before, m.products(self.output))
        receipt = m.base.read_json(self.output / 'items/synthetic.json')['payload']
        self.assertEqual(receipt['features'], {m.FEATURE: NumericalTests.scalar['median_squared_bicoherence']})
        self.assertEqual(receipt['native_origin']['sample_rate_hz'], 48000)
        self.assertEqual(receipt['analysis_view']['resampling']['native_sample_rate_hz'], 44100)
        self.assertEqual(set(receipt['features']), {m.FEATURE})

    def test_all_null_measurement_still_complete_with_source_coverage(self):
        path = Path(self.row['input']['path']); zeros = np.zeros((1323000, 2), dtype='float32')
        sf.write(path, zeros, 44100, subtype='FLOAT')
        self.row['input'] = m.base.file_binding(path)
        self.row['waveform_float32_sha256'] = hashlib.sha256(zeros.astype('<f4').tobytes()).hexdigest()
        result = m.run_contract(self.run)
        self.assertEqual(result['status'], 'verified_complete_native30_BC')
        self.assertEqual(result['source_coverage']['Synthetic']['missing'], 1)
        self.assertEqual(result['source_coverage']['Synthetic']['eligible_pool_counts'], {'0': 1})
        receipt = m.base.read_json(self.output / 'items/synthetic.json')['payload']
        self.assertEqual(receipt['features'], {m.FEATURE: None})
        self.assertEqual(receipt['scalar']['status'], 'no_eligible_target_pools')

    def test_label_source_invariance_on_identical_audio(self):
        changed = {**self.row, 'label': '1', 'source_group': 'Another'}
        a, b = m.measure(self.row), m.measure(changed)
        self.assertEqual(a[0], b[0]); self.assertEqual(a[1], b[1]); self.assertEqual(a[3], b[3])

    def test_expected_failure_retained_then_safe_retry(self):
        with patch.object(m, 'measure', side_effect=FloatingPointError('synthetic expected processing failure')):
            result = m.run_contract(self.run)
        self.assertEqual(result['status'], 'partial_no_COMMIT')
        failures = list((self.output / 'failures').iterdir())
        self.assertEqual(len(failures), 1)
        self.assertFalse(m.base.read_json(failures[0])['scientific_null'])
        self.assertFalse((self.output / 'COMMIT.json').exists())
        self.run_cached()
        self.assertTrue(failures[0].exists())
        self.assertIn(str(failures[0].relative_to(self.output)), m.base.read_json(self.output / 'COMMIT.json')['products'])

    def test_orphan_product_no_overwrite(self):
        with patch.object(m, 'measure', side_effect=ValueError('first failed')):
            m.run_contract(self.run)
        orphan = self.output / 'arrays/synthetic.npz'; orphan.write_bytes(b'orphan')
        with patch.object(m, 'measure', side_effect=AssertionError('must not measure')):
            result = m.run_contract(self.run)
        self.assertEqual(result['failed'], 1)
        self.assertEqual(orphan.read_bytes(), b'orphan')

    def test_array_corruption_and_resealed_scalar_mutation_rejected(self):
        self.run_cached()
        item = self.output / 'items/synthetic.json'
        value = m.base.read_json(item); value['payload']['features'][m.FEATURE] = .1
        value['receipt_sha256'] = m.base.value_hash(value['payload']); item.write_bytes(m.base.canonical(value))
        with self.assertRaisesRegex(ValueError, 'scalar/features replay'):
            m.verify_receipt(self.output, self.row, m.base.value_hash(self.run), self.origin)
        arrays = copy.deepcopy(self.cached[2]); arrays['eligible_mask'][0, 0] = ~arrays['eligible_mask'][0, 0]
        with self.assertRaisesRegex(ValueError, 'mask'):
            m.validate_arrays(arrays, self.cached[1])

    def test_input_hash_mutation_before_and_after_measurement_rejected(self):
        path = Path(self.row['input']['path'])
        before = path.read_bytes()
        path.write_bytes(before[:-1] + bytes([before[-1] ^ 1]))
        with self.assertRaises(ValueError):
            m.decode(self.row)
        path.write_bytes(before)
        original = m.numerical_modules()[1].extract
        def mutate(*args):
            result = original(*args)
            path.write_bytes(before[:-1] + bytes([before[-1] ^ 1]))
            return result
        with patch.object(m.numerical_modules()[1], 'extract', side_effect=mutate), self.assertRaisesRegex(ValueError, 'changed during measurement'):
            m.measure(self.row)

    def test_short_wrong_format_and_nonfinite_files_rejected(self):
        for values, rate, subtype in ((self.stereo[:-1], 44100, 'FLOAT'), (self.stereo, 44100, 'PCM_16'),
                                      (self.stereo, 48000, 'FLOAT')):
            path = self.root / ('bad_' + str(rate) + '_' + subtype + '_' + str(len(values)) + '.wav')
            sf.write(path, values, rate, subtype=subtype)
            row = {**self.row, 'input': m.base.file_binding(path)}
            with self.assertRaisesRegex(ValueError, 'exact FLOAT'):
                m.decode(row)
        invalid = self.stereo.copy(); invalid[9, 1] = np.inf
        path = self.root / 'nonfinite.wav'; sf.write(path, invalid, 44100, subtype='FLOAT')
        with self.assertRaisesRegex(ValueError, 'finite'):
            m.decode({**self.row, 'input': m.base.file_binding(path)})

    def test_float64_decode_exact_float32_promotion_and_pcm_pin(self):
        values = m.decode(self.row)
        np.testing.assert_array_equal(values, self.stereo)
        self.assertEqual(values.dtype, np.dtype('float64'))
        with self.assertRaisesRegex(ValueError, 'PCM hash'):
            m.decode({**self.row, 'waveform_float32_sha256': 'f' * 64})

    def test_conflicting_output_and_exclusive_lock(self):
        with m.base.writer_lock(self.output), self.assertRaises(RuntimeError):
            self.run_cached()
        put(self.output / 'unexpected.json', {})
        with self.assertRaisesRegex(ValueError, 'nonempty output'):
            self.run_cached()

    def test_manifest_coverage_replay_and_exhaustive_commit_inventory(self):
        self.run_cached()
        manifest = m.base.read_json(self.output / 'manifest.json')
        self.assertEqual(manifest['source_coverage']['Synthetic']['available'], 1)
        self.assertEqual(manifest['source_coverage']['Synthetic']['eligible_pool_counts'], {'2': 1})
        put(self.output / 'extra.json', {})
        with self.assertRaisesRegex(ValueError, 'unexpected/orphan'):
            m.verify_output(self.run)


if __name__ == '__main__':
    unittest.main()
