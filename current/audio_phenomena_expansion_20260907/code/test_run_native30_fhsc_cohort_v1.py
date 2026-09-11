"""Synthetic only: frozen joins, metadata-only preflight, receipts and failure recovery."""
import copy
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

import run_native30_fhsc_cohort_v1 as m


def document_fixture():
    plan_rows, screen_rows, components, receipts = [], [], [], {}
    for ident, origin, source, label in [('fresh', 'native30_evidence_v2', 'ACE-Step', '1'),
                                         ('old', 'prior60', 'MTG-Jamendo', '0'),
                                         ('short', 'native30_evidence_v2', 'FMA', '0')]:
        row = {'id': ident, 'source_group': source, 'label': label, 'role': 'development',
               'group_id': 'g_' + ident, 'component_id': 'c_' + ident, 'origin_set': origin,
               'input_occurrences': ['master30'], 'selected_metadata_input': 'master30',
               'source_origin': {'sha256': 'a' * 64, 'channels': 2, 'sample_rate_hz': 44100}}
        screened = {**row, 'duration_exposure_candidate': True, 'exclusion_reasons': []}
        component = {'component_id': row['component_id'], 'members': [ident],
                     'candidate_screen_members': [ident], 'protected_relationships': []}
        receipt_row = {**row, 'native_evidence': row['source_origin'],
                       'origin_plan_row_sha256': m.base.value_hash(row),
                       'screen_row_sha256': m.base.value_hash(screened), 'component_sha256': m.base.value_hash(component)}
        receipts[ident] = {**{k: row[k] for k in m.IDENTITY}, 'row': receipt_row,
                          'row_sha256': m.base.value_hash(receipt_row), 'status': 'materialized_DSP_only',
                          'waveform_bit_exact_roundtrip': True, 'waveform_float32_sha256': 'b' * 64,
                          'audit': {'output_waveform_float32_sha256': 'b' * 64},
                          'standardized_path': '/synthetic/audio/' + ident + '.wav', 'file_bytes': 10, 'file_sha256': 'c' * 64}
        plan_rows.append(row)
        screen_rows.append(screened)
        components.append(component)
    scope = {'classifier_fits': 0, 'cohort_admitted': False, 'feature_extraction_authorized': False}
    plan = {'schema_version': 'native30-origin-plan-v2', 'status': 'draft_not_admitted_not_frozen_for_execution',
            'rows': plan_rows, 'counts': {'audio_files_opened': 0, 'classifier_fits': 0}}
    screen = {'rows': screen_rows, 'components': components, **scope}
    new = {'version': m.CLOSURE_VERSION, 'status': m.CLOSURE_STATUS, 'records': [receipts['fresh']],
           'excluded_short_records': [{**{k: receipts['short'][k] for k in m.IDENTITY}, 'actual_frames': 1322999,
                                       'required_frames': 1323000, 'native_rate_hz': 44100}],
           'eligible_ids': ['fresh'], 'excluded_short_ids': ['short'], 'all_attempted_ids': ['fresh', 'short'],
           'eligible_count': 1, 'excluded_short_count': 1, 'attempted': 2, **scope}
    prior = {'status': 'all_prior2174_DSP_materialized_not_cohort_admitted', 'records': [receipts['old']], 'count': 1, **scope}
    expected = {'plan': 3, 'new': 1, 'prior': 1, 'excluded': 1, 'total': 2, 'human': 1, 'ai': 1,
                'sources': {'ACE-Step': 1, 'MTG-Jamendo': 1}}
    return plan, screen, new, prior, expected


class ReconciliationTests(unittest.TestCase):
    def test_exact_production_roster(self):
        self.assertEqual(sum(m.SOURCES.values()), 3830)
        self.assertEqual(len(m.SOURCES), 11)
        self.assertEqual(sum(v for k, v in m.SOURCES.items() if k in {'FMA', 'MTG-Jamendo'} or k.startswith('human_')), 1664)
        self.assertEqual(m.SOURCES['FMA'], 393 - 39)

    def test_reduced_fixture_join(self):
        rows = m.reconcile(*document_fixture())
        self.assertEqual([r['id'] for r in rows], ['fresh', 'old'])
        self.assertEqual([r['origin_family'] for r in rows], ['new', 'prior'])

    def test_duplicate_receipts_rejected(self):
        plan, screen, new, prior, expected = document_fixture()
        new['records'].append(new['records'][0])
        with self.assertRaisesRegex(ValueError, 'duplicate ID'):
            m.reconcile(plan, screen, new, prior, expected)

    def test_scope_identity_origin_and_protected_mismatches(self):
        for defect in ['role', 'label', 'component', 'source_sha', 'origin', 'prior_hash', 'cohort_scope', 'unknown_excluded', 'not_short']:
            with self.subTest(defect=defect):
                p, s, n, o, e = document_fixture()
                if defect == 'role': n['records'][0]['role'] = 'external'
                if defect == 'label': n['records'][0]['label'] = '0'
                if defect == 'component': s['components'][0]['protected_relationships'] = ['external']
                if defect == 'source_sha': n['records'][0]['row']['native_evidence'] = {'sha256': 'd'*64, 'channels': 2, 'sample_rate_hz': 44100}
                if defect == 'origin': p['rows'][0]['origin_set'] = 'prior60'
                if defect == 'prior_hash': o['records'][0]['row']['origin_plan_row_sha256'] = 'e'*64
                if defect == 'cohort_scope': n['cohort_admitted'] = True
                if defect == 'unknown_excluded': n['excluded_short_records'][0]['source_group'] = 'Suno'
                if defect == 'not_short': n['excluded_short_records'][0]['actual_frames'] = 1323000
                with self.assertRaises(ValueError): m.reconcile(p, s, n, o, e)

    def test_wrong_counts_fail_even_with_valid_rows(self):
        p, s, n, o, e = document_fixture()
        e['human'] = 0
        with self.assertRaisesRegex(ValueError, 'source/label roster'):
            m.reconcile(p, s, n, o, e)


class MetadataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.addCleanup(self.temp.cleanup)

    def upstream(self):
        root = self.root / 'prior'
        root.mkdir()
        (root / 'audio').mkdir()
        (root / 'audio' / 'synthetic.wav').write_bytes(b'no audio decoder should read this')
        m.base.write_new(root / 'manifest.json', {'test': 'metadata'})
        products = {name: {k: v for k, v in m.base.file_binding(root / name).items() if k != 'path'} for name in
                    ('audio/synthetic.wav', 'manifest.json')}
        m.base.write_new(root / 'COMMIT.json', {'status': 'committed_prior2174_DSP_only', 'products': products})
        return root, m.base.digest(root / 'COMMIT.json')

    def test_commit_preflight_never_opens_audio(self):
        root, sha = self.upstream()
        original = m.base.digest
        def metadata_only(path):
            self.assertNotEqual(Path(path).suffix, '.wav')
            return original(path)
        with patch.object(m.base, 'digest', side_effect=metadata_only), patch.object(sf, 'SoundFile', side_effect=AssertionError('audio opened')):
            bindings = {}
            m.bind_commit(root, sha, 'prior', bindings)
        self.assertIn(str(root / 'audio/synthetic.wav'), bindings)

    def test_caller_commit_pin_is_required(self):
        root, sha = self.upstream()
        for wrong in (None, '', 'x', 'f' * 64):
            with self.subTest(wrong=wrong), self.assertRaises(ValueError):
                m.bind_commit(root, wrong, 'prior', {})

    def test_exhaustive_inventory_rejects_extra_file(self):
        root, sha = self.upstream()
        (root / 'extra.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'exhaustive product inventory'):
            m.bind_commit(root, sha, 'prior', {})

    def test_metadata_mutation_fails(self):
        root, sha = self.upstream()
        path = root / 'manifest.json'
        path.write_bytes(path.read_bytes().replace(b'metadata', b'metadatX'))
        with self.assertRaisesRegex(ValueError, 'metadata product changed'):
            m.bind_commit(root, sha, 'prior', {})

    def test_symlink_product_and_redirected_root_rejected(self):
        root, sha = self.upstream()
        alias = self.root / 'alias'
        alias.symlink_to(root, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'unredirected'):
            m.bind_commit(alias, sha, 'prior', {})
        (root / 'redirect.json').symlink_to(root / 'manifest.json')
        with self.assertRaisesRegex(ValueError, 'symlink'):
            m.bind_commit(root, sha, 'prior', {})

    def test_closure_audio_bindings_deferred_but_metadata_rehashed(self):
        root, sha = self.upstream()
        path = root / 'audio' / 'synthetic.wav'
        declared = m.base.file_binding(path)
        declared['signature'] = [1, 2, declared.pop('bytes'), 0, 0]
        bindings = {}
        with patch.object(m.base, 'digest', side_effect=AssertionError('audio hash in preflight')):
            m.collect_upstream({'sources': {'ignored_key': declared}}, bindings)
        self.assertEqual(bindings[str(path)]['sha256'], declared['sha256'])
        contract = {'bindings': bindings}
        m.check_contract(contract, audio=False)
        m.check_contract(contract, audio=True)
        path.write_bytes(b'X' * path.stat().st_size)
        m.check_contract(contract, audio=False)
        with self.assertRaisesRegex(ValueError, 'bound input changed'):
            m.check_contract(contract, audio=True)


class ExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pilot = m.pinned_module('run_native30_fhsc_pilot_v1', m.PILOT_SHA)
        cls.reference_temp = tempfile.TemporaryDirectory()
        cls.reference_path = Path(cls.reference_temp.name) / 'reference.wav'
        sf.write(cls.reference_path, np.zeros((1323000, 2), dtype=np.float32), 44100, subtype='FLOAT')
        cls.measurement, cls.sc = cls.pilot.measure(cls.reference_path)

    @classmethod
    def tearDownClass(cls):
        cls.reference_temp.cleanup()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        path = self.root / 'input.wav'
        path.write_bytes(self.reference_path.read_bytes())
        self.row = {'id': 'synthetic', 'source_group': 'FMA', 'label': '0', 'role': 'development',
                    'group_id': 'synthetic', 'component_id': 'synthetic', 'input': m.base.file_binding(path),
                    'waveform_float32_sha256': m.hashlib.sha256(np.zeros((1323000, 2), dtype='<f4').tobytes()).hexdigest()}
        self.output = self.root / 'measurement'
        self.contract = {'output_root': str(self.output), 'rows': [self.row], 'expected_count': 1,
                         'bindings': {str(path): self.row['input']}, **m.SCOPE}
        self.sha = m.freeze(self.contract)
        def measured(path):
            result = copy.deepcopy(self.measurement)
            result['analysis_view_audit'].update(source_audio_path=str(path), source_audio_sha256=m.base.digest(path))
            return result, copy.deepcopy(self.sc)
        self.fake = SimpleNamespace(SC_KEYS=self.pilot.SC_KEYS, measure=measured)

    def test_real_synthetic_measurement_missingness_and_verified_resume(self):
        result = m.run_contract(self.contract, self.pilot, self.sha)
        self.assertEqual(result['completed'], 1)
        self.assertEqual(result['failed'], 0)
        self.assertEqual(result['availability']['SC_selected_six'], {'missing': 1})
        self.assertEqual(result['availability']['F'], {'low_energy': 1})
        commit_before = (self.output / 'COMMIT.json').read_bytes()
        with patch.object(self.pilot, 'measure', side_effect=AssertionError('resume must not rerun measure')):
            result = m.run_contract(self.contract, self.pilot, self.sha)
        self.assertEqual(result['status'], 'verified_existing_COMMIT')
        self.assertEqual((self.output / 'COMMIT.json').read_bytes(), commit_before)
        item = m.base.read_json(self.output / 'items/synthetic.json')['payload']
        self.assertEqual(item['measurement']['M_diagnostic_not_predictor']['M_status'], 'missing_low_energy')
        self.assertTrue(all(v is None for v in item['measurement']['SC']['features'].values()))
        self.assertFalse(item['cohort_admitted'])
        self.assertEqual(item['classifier_fits'], 0)

    def test_full_grid_partial_does_not_gate_selected_six(self):
        def measured(path):
            result, sc = self.fake.measure(path)
            result['SC'].update(features={k: 0.0 for k in self.pilot.SC_KEYS}, selected_six_finite_count=6,
                                selected_six_status='complete', selected_six_missing=[], full_grid_diagnostic_status='partial')
            return result, sc
        fake = SimpleNamespace(SC_KEYS=self.pilot.SC_KEYS, measure=measured)
        result = m.run_contract(self.contract, fake, self.sha)
        self.assertEqual(result['availability']['SC_selected_six'], {'complete': 1})
        self.assertEqual(result['availability']['SC_full_grid_diagnostic'], {'partial': 1})

    def test_exception_retained_then_successful_resume(self):
        failing = SimpleNamespace(SC_KEYS=self.pilot.SC_KEYS, measure=lambda p: (_ for _ in ()).throw(ValueError('synthetic corruption')))
        result = m.run_contract(self.contract, failing, self.sha)
        self.assertEqual(result['status'], 'partial_no_COMMIT')
        self.assertFalse((self.output / 'COMMIT.json').exists())
        failures = list((self.output / 'failures').iterdir())
        self.assertEqual(len(failures), 1)
        failure_before = failures[0].read_bytes()
        result = m.run_contract(self.contract, self.fake, self.sha)
        self.assertEqual(result['completed'], 1)
        self.assertEqual(failures[0].read_bytes(), failure_before)
        commit = m.base.read_json(self.output / 'COMMIT.json')
        self.assertIn('failures/' + failures[0].name, commit['products'])

    def test_orphan_frames_are_retained_and_never_reused(self):
        for name in ('items', 'frames', 'failures', 'runs'): (self.output / name).mkdir()
        path = self.output / 'frames/synthetic.sc_frames.npz'
        path.write_bytes(b'orphan')
        result = m.run_contract(self.contract, self.fake, self.sha)
        self.assertEqual(result['status'], 'partial_no_COMMIT')
        self.assertIn('unreceipted', result['failures'][0]['message'])
        self.assertEqual(path.read_bytes(), b'orphan')

    def test_invalid_M_is_failure_not_scientific_missingness(self):
        def measured(path):
            result, sc = self.fake.measure(path)
            result['M_diagnostic_not_predictor']['M_status'] = 'complete'
            return result, sc
        result = m.run_contract(self.contract, SimpleNamespace(SC_KEYS=self.pilot.SC_KEYS, measure=measured), self.sha)
        self.assertEqual(result['status'], 'partial_no_COMMIT')
        self.assertIn('M cannot be predictor', result['failures'][0]['message'])

    def test_malformed_feature_dict_is_failure_not_missingness(self):
        def measured(path):
            result, sc = self.fake.measure(path)
            result['F'] = {'F_status': 'low_energy'}
            return result, sc
        result = m.run_contract(self.contract, SimpleNamespace(SC_KEYS=self.pilot.SC_KEYS, measure=measured), self.sha)
        self.assertEqual(result['status'], 'partial_no_COMMIT')
        self.assertIn('malformed F measurement keys', result['failures'][0]['message'])
        self.assertFalse((self.output / 'COMMIT.json').exists())

    def test_nonfinite_frame_products_fail(self):
        arrays = {**copy.deepcopy(self.sc['per_frame']), **copy.deepcopy(self.sc['valid_masks'])}
        arrays['iid_db'][0, 0] = np.inf
        with self.assertRaisesRegex(ValueError, 'infinite SC frame'):
            m.validate_frames(arrays)

    def test_corrupt_frame_resume_fails(self):
        m.run_contract(self.contract, self.fake, self.sha)
        frames = self.output / 'frames/synthetic.sc_frames.npz'
        frames.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'COMMIT changed'):
            m.run_contract(self.contract, self.fake, self.sha)

    def test_input_bytes_changed_rejected_before_measurement(self):
        path = Path(self.row['input']['path'])
        path.write_bytes(b'corrupted')
        with self.assertRaises(ValueError):
            m.run_contract(self.contract, self.fake, self.sha)
        self.assertFalse((self.output / 'COMMIT.json').exists())

    def test_exact_FLOAT_format_required(self):
        path = Path(self.row['input']['path'])
        sf.write(path, np.zeros((1323000, 2), dtype=np.float32), 44100, subtype='PCM_16')
        row = {**self.row, 'input': m.base.file_binding(path)}
        with self.assertRaisesRegex(ValueError, 'exact FLOAT'):
            m.validate_audio(row)

    def test_nonfinite_and_wrong_duration_rejected_without_clipping(self):
        path = Path(self.row['input']['path'])
        for frames, nonfinite in ((1323001, False), (1323000, True)):
            values = np.zeros((frames, 2), dtype=np.float32)
            if nonfinite: values[0, 0] = np.nan
            sf.write(path, values, 44100, subtype='FLOAT')
            with self.subTest(frames=frames), self.assertRaises(ValueError):
                m.validate_audio({**self.row, 'input': m.base.file_binding(path)})

    def test_caller_contract_pin_and_explicit_prepare_required(self):
        with self.assertRaisesRegex(ValueError, 'caller prepared contract'):
            m.run_contract(self.contract, self.fake, '0'*64)
        other = copy.deepcopy(self.contract)
        other['output_root'] = str(self.root / 'not_prepared')
        with self.assertRaises(FileNotFoundError):
            m.run_contract(other, self.fake, m.base.value_hash(other))

    def test_exclusive_writer_lock(self):
        with m.base.writer_lock(self.output):
            with self.assertRaisesRegex(RuntimeError, 'writer lock is held'):
                m.run_contract(self.contract, self.fake, self.sha)

    def test_final_input_rehash_catches_mid_measurement_mutation(self):
        def measured(path):
            result, sc = self.fake.measure(path)
            path.write_bytes(b'changed during extraction')
            return result, sc
        with self.assertRaises(ValueError):
            m.run_contract(self.contract, SimpleNamespace(SC_KEYS=self.pilot.SC_KEYS, measure=measured), self.sha)
        self.assertFalse((self.output / 'COMMIT.json').exists())
        self.assertEqual(len(list((self.output / 'failures').iterdir())), 1)


if __name__ == '__main__':
    unittest.main()
