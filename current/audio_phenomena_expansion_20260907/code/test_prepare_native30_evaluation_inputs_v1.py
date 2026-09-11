import copy
from pathlib import Path
import tempfile
import unittest

import prepare_native30_evaluation_inputs_v1 as m


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    m.write_new(path, value)
    return m.binding(path)


def source_products(root):
    return {relative: {'bytes': path.stat().st_size, 'sha256': m.digest(path)}
            for relative, path in sorted(m.root_files(root).items())}


class AssemblerTests(unittest.TestCase):
    def fixture(self, root, *, missing_sdrp_key=None, schedule_ids=None, recovery=False):
        root = root.resolve()
        audio = root / 'input.wav'; audio.write_bytes(b'not opened by assembler')
        input_entry = m.binding(audio)
        identity = {'id': 'x', 'source_group': 'Synthetic', 'label': '1', 'role': 'development',
                    'group_id': 'g', 'component_id': 'c'}
        plan_row = {**identity, 'source_origin': {'sample_rate_hz': 48000, 'channels': 2}}
        screen_row = {**identity, 'duration_exposure_candidate': True, 'exclusion_reasons': []}
        component = {'component_id': 'c', 'members': ['x'], 'candidate_screen_members': ['x'],
                     'protected_relationships': []}
        row = {**identity, 'origin_family': 'new', 'origin_plan_row_sha256': m.value_hash(plan_row),
               'screen_row_sha256': m.value_hash(screen_row), 'component_sha256': m.value_hash(component),
               'producer_receipt_sha256': '4' * 64, 'input': input_entry,
               'waveform_float32_sha256': '5' * 64}
        plan = put(root / 'plan.json', {'rows': [plan_row]})
        screen = put(root / 'screen.json', {'rows': [screen_row], 'components': [component]})
        cohort = {'version': 'run_native30_fhsc_cohort_v1',
                  'status': 'frozen_before_any_measurement_audio_reads', 'expected_count': 1,
                  'rows': [row], 'bindings': {}, 'source_counts': {'Synthetic': 1}, 'human': 0, 'ai': 1}
        cohort_entry = put(root / 'cohort.json', cohort)

        fhsc_root = root / 'fhsc'; (fhsc_root / 'items').mkdir(parents=True); (fhsc_root / 'frames').mkdir()
        fhsc_contract = {'version': 'run_native30_fhsc_cohort_v1',
                         'status': 'frozen_before_any_measurement_audio_reads', 'expected_count': 1,
                         'output_root': str(fhsc_root), 'rows': [row],
                         'F_measure_names': m.FAMILY_CONFIG['F'], 'H_measure_names': m.FAMILY_CONFIG['H'],
                         'SC_selected_six': m.FAMILY_CONFIG['SC']}
        fhsc_contract_entry = put(fhsc_root / 'contract.json', fhsc_contract)
        frame = fhsc_root / 'frames/x.sc_frames.npz'; frame.write_bytes(b'synthetic-frame-product')
        measurement = {
            'F': {name: (None if index == 0 else float(index)) for index, name in enumerate(m.FAMILY_CONFIG['F'])},
            'H': {name: float(index) for index, name in enumerate(m.FAMILY_CONFIG['H'])},
            'SC': {'features': {name: float(index) for index, name in enumerate(m.FAMILY_CONFIG['SC'])}},
            'M_diagnostic_not_predictor': {'M_status': 'missing_short_duration'},
            'analysis_view_audit': {'source_audio_sha256': input_entry['sha256']},
        }
        fhsc_payload = {'status': 'measured_not_admitted', 'row': row, 'row_sha256': m.value_hash(row),
                        'contract_sha256': fhsc_contract_entry['sha256'], 'measurement': measurement,
                        'frame_product': m.binding(frame), 'classifier_fits': 0, 'cohort_admitted': False}
        put(fhsc_root / 'items/x.json', {'payload': fhsc_payload,
                                        'receipt_sha256': m.value_hash(fhsc_payload)})
        fhsc_commit = {'status': 'committed_native30_F_H_SC_measurements_not_admitted',
                       'contract_sha256': fhsc_contract_entry['sha256'], 'completed': 1,
                       'products': source_products(fhsc_root),
                       'all_bound_inputs_and_products_end_rehashed': True,
                       'classifier_fits': 0, 'cohort_admitted': False, 'no_neural_inference': True,
                       'M_is_diagnostic_only': True, 'source_selection_changed': False}
        fhsc_commit_entry = put(fhsc_root / 'COMMIT.json', fhsc_commit)

        sdrp_root = root / 'sdrp'; (sdrp_root / 'items').mkdir(parents=True)
        mapped = {'cohort_row': row, 'native_sample_rate_hz': 48000,
                  'extractor_row': {'item_id': 'x', 'source_id': 'Synthetic', 'label': '1', 'group_id': 'g',
                                    'standardized_path': str(audio), 'native_sample_rate_hz': '48000',
                                    'duration': '30', 'audio_offset_s': '0'}}
        expanded = m.binding(Path(__file__).resolve().parents[2] /
                             'source_diversity_expansion_20260905/code/expanded_feature_definitions.py')
        sdrp_version = 'extract_native30_sdrp_recovery_v1' if recovery else 'extract_native30_sdrp_v1'
        backend = m.SDRP_BACKENDS[sdrp_version]
        backend_runner = m.binding(Path(m.__file__).resolve().with_name(sdrp_version + '.py'))
        self.assertEqual(backend_runner['sha256'], backend['runner_sha256'])
        if recovery:
            mapped['beat_receipt_path'] = str(root / 'recovery-receipt.json')
            mapped['beat_provenance'] = {'kind': 'recovered_serializer_v1', 'item_status': 'preserved_cli_success'}
        sdrp_contract = {'version': sdrp_version, 'status': backend['contract_status'],
                         'expected_count': 1, 'output_root': str(sdrp_root), 'rows': [mapped],
                         'bindings': {'core': expanded, 'runner': backend_runner}}
        sdrp_contract_entry = put(sdrp_root / 'contract.json', sdrp_contract)
        legacy = {'status': 'complete', 'feature_status': 'complete', 'errors': '', 'item_id': 'x',
                  'source_id': 'Synthetic', 'label': '1', 'group_id': 'g', 'duration_sec': 30,
                  'native_sample_rate_hz': 48000, 's16__excluded_high_band_feature': 123.0,
                  's16_native_eligible': 1, 's8_native_eligible': 1,
                  **{name: float(index) for index, name in enumerate(
                      m.FAMILY_CONFIG['S'] + m.FAMILY_CONFIG['D'] + m.FAMILY_CONFIG['R'] + m.FAMILY_CONFIG['P'])}}
        if missing_sdrp_key:
            legacy.pop(missing_sdrp_key)
        input_evidence = {**input_entry, 'format': 'WAV', 'subtype': 'FLOAT', 'sample_rate_hz': 44100,
                          'channels': 2, 'frames': 1323000, 'finite': True,
                          'waveform_float32_sha256': row['waveform_float32_sha256']}
        sdrp_payload = {'status': backend['item_status'], 'row': mapped,
                        'row_sha256': m.value_hash(mapped), 'contract_sha256': sdrp_contract_entry['sha256'],
                        'inputs': {'input': input_evidence}, 'legacy_result': legacy,
                        **backend['scope']}
        put(sdrp_root / 'items/x.json', {'payload': sdrp_payload,
                                        'receipt_sha256': m.value_hash(sdrp_payload)})
        sdrp_commit = {'version': sdrp_version,
                       'status': backend['commit_status'],
                       'contract_sha256': sdrp_contract_entry['sha256'], 'completed': 1,
                       'products': source_products(sdrp_root),
                       'all_inputs_products_runtime_end_rehashed': True,
                       **backend['scope']}
        sdrp_commit_entry = put(sdrp_root / 'COMMIT.json', sdrp_commit)

        schedule_root = root / 'schedule'; schedule_root.mkdir()
        schedule = {'version': 'plan_native30_evaluation_schedule_v1',
                    'status': 'new_protocol_metadata_draft_not_frozen_for_evaluation',
                    'classifier_fits': 0, 'fitting_authorized': False,
                    'prepared_contract_sha256': cohort_entry['sha256'],
                    'input_population': {'ids': ['x'] if schedule_ids is None else schedule_ids}}
        schedule_entry = put(schedule_root / 'schedule_draft.json', schedule)
        schedule_commit = {'classifier_fits': 0, 'draft_sha256': schedule_entry['sha256'],
                           'fitting_authorized': False, 'products': {'schedule_draft.json': schedule_entry},
                           'status': 'committed_metadata_draft_not_evaluation_authorization',
                           'version': 'plan_native30_evaluation_schedule_v1'}
        schedule_commit_entry = put(schedule_root / 'COMMIT.json', schedule_commit)
        args = dict(cohort_path=cohort_entry['path'], cohort_sha=cohort_entry['sha256'],
                    plan_path=plan['path'], screen_path=screen['path'],
                    fhsc_root=fhsc_root, fhsc_contract_sha=fhsc_contract_entry['sha256'],
                    fhsc_commit_sha=fhsc_commit_entry['sha256'], sdrp_root=sdrp_root,
                    sdrp_contract_sha=sdrp_contract_entry['sha256'], sdrp_commit_sha=sdrp_commit_entry['sha256'],
                    schedule_root=schedule_root, schedule_sha=schedule_entry['sha256'],
                    schedule_commit_sha=schedule_commit_entry['sha256'], expected=1,
                    plan_sha=plan['sha256'], screen_sha=screen['sha256'], expanded_sha=expanded['sha256'])
        return args, row, fhsc_root, sdrp_root

    def test_exact_family_counts_and_order(self):
        self.assertEqual({key: len(value) for key, value in m.FAMILY_CONFIG.items()},
                         {'S': 15, 'D': 3, 'R': 3, 'P': 6, 'F': 15, 'H': 6, 'SC': 6})
        self.assertEqual(len(m.FEATURE_NAMES), 54)
        self.assertEqual(m.FEATURE_NAMES, [name for family in ('S', 'D', 'R', 'P', 'F', 'H', 'SC')
                                           for name in m.FAMILY_CONFIG[family]])

    def test_fhsc_commit_matches_actual_producer_schema_without_version(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name).resolve(); args, _, fhsc_root, _ = self.fixture(root)
            commit = m.read_json(fhsc_root / 'COMMIT.json')
            self.assertNotIn('version', commit)
            self.assertEqual(set(commit), {'status', 'contract_sha256', 'completed', 'products',
                                           'all_bound_inputs_and_products_end_rehashed',
                                           'classifier_fits', 'cohort_admitted', 'no_neural_inference',
                                           'M_is_diagnostic_only', 'source_selection_changed'})
            m.load_authorities(**args)

    def test_fhsc_commit_unexpected_version_is_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name).resolve(); args, _, fhsc_root, _ = self.fixture(root)
            commit = m.read_json(fhsc_root / 'COMMIT.json')
            commit['version'] = 'run_native30_fhsc_cohort_v1'
            (fhsc_root / 'COMMIT.json').write_bytes(m.canonical(commit))
            args['fhsc_commit_sha'] = m.digest(fhsc_root / 'COMMIT.json')
            with self.assertRaisesRegex(ValueError, 'COMMIT scope/status'):
                m.load_authorities(**args)

    def test_assemble_preserves_null_and_excludes_s16_M_and_eligibility(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name).resolve(); args, row, _, _ = self.fixture(root)
            authorities = m.load_authorities(**args, deep=True)
            proof = m.publish(authorities, root / 'package')
            feature = proof['features'][0]
            self.assertEqual(set(feature), {'id', *m.FEATURE_NAMES})
            self.assertIsNone(feature['F_phase_residual_cvar_all'])
            self.assertFalse(any(key.startswith('s16__') or key.startswith('M_') or 'eligible' in key for key in feature))
            self.assertEqual(proof['metadata'][0]['native_sample_rate_hz'], 48000)
            self.assertEqual(proof['metadata'][0]['input_sha256'], row['input']['sha256'])
            self.assertEqual(proof['lineage'][0]['schedule_input_position'], 0)
            self.assertEqual(proof['lineage'][0]['source_input'], row['input'])

    def test_explicit_recovery_backend_preserves_provenance_without_predictor(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name).resolve(); args, _, _, _ = self.fixture(root, recovery=True)
            proof = m.publish(m.load_authorities(**args, deep=True), root / 'package')
            self.assertEqual(proof['contract']['sdrp_backend'], 'extract_native30_sdrp_recovery_v1')
            self.assertEqual(proof['contract']['feature_names'], m.FEATURE_NAMES)
            self.assertNotIn('beat_provenance', proof['features'][0])
            source_receipt = m.read_json(proof['lineage'][0]['sdrp_receipt']['path'])['payload']
            self.assertEqual(source_receipt['row']['beat_provenance']['kind'], 'recovered_serializer_v1')

    def test_unlisted_sdrp_backend_is_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name).resolve(); args, _, _, sdrp_root = self.fixture(root)
            contract = m.read_json(sdrp_root / 'contract.json'); contract['version'] = 'unknown_backend'
            (sdrp_root / 'contract.json').write_bytes(m.canonical(contract))
            args['sdrp_contract_sha'] = m.digest(sdrp_root / 'contract.json')
            commit = m.read_json(sdrp_root / 'COMMIT.json')
            commit['contract_sha256'] = args['sdrp_contract_sha']
            (sdrp_root / 'COMMIT.json').write_bytes(m.canonical(commit))
            args['sdrp_commit_sha'] = m.digest(sdrp_root / 'COMMIT.json')
            with self.assertRaisesRegex(ValueError, 'unapproved S/D/R/P backend'):
                m.load_authorities(**args)

    def test_publish_rehashes_preflight_authorities_before_assembly(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name).resolve(); args, _, _, _ = self.fixture(root)
            authorities = m.load_authorities(**args, deep=False)
            proof = m.publish(authorities, root / 'package')
            commit = m.read_json(proof['commit']['path'])
            self.assertTrue(commit['all_source_COMMIT_products_rehashed_before_assembly'])
            self.assertEqual(commit['source_rehash_before'], commit['source_rehash_end'])

    def test_publish_rejects_output_overlapping_committed_source_root(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name).resolve(); args, _, fhsc_root, _ = self.fixture(root)
            authorities = m.load_authorities(**args, deep=False)
            with self.assertRaisesRegex(ValueError, 'overlaps'):
                m.publish(authorities, fhsc_root / 'bad-output')
            self.assertFalse((fhsc_root / 'bad-output').exists())

    def test_changed_committed_feature_receipt_is_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name).resolve(); args, _, fhsc_root, _ = self.fixture(root)
            (fhsc_root / 'items/x.json').write_text('{"changed":true}\n')
            with self.assertRaisesRegex(ValueError, 'size changed|hash changed'):
                m.load_authorities(**args, deep=True)

    def test_missing_declared_predictor_is_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name).resolve(); args, _, _, _ = self.fixture(root, missing_sdrp_key=m.FAMILY_CONFIG['S'][0])
            authorities = m.load_authorities(**args, deep=True)
            with self.assertRaisesRegex(ValueError, 'predictor set changed'):
                m.assemble_rows(authorities)

    def test_absent_F_key_is_failure_but_explicit_null_is_scientific_missingness(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name).resolve(); args, _, fhsc_root, _ = self.fixture(root)
            envelope = m.read_json(fhsc_root / 'items/x.json')
            envelope['payload']['measurement']['F'].pop(m.FAMILY_CONFIG['F'][0])
            envelope['receipt_sha256'] = m.value_hash(envelope['payload'])
            (fhsc_root / 'items/x.json').write_bytes(m.canonical(envelope))
            commit = m.read_json(fhsc_root / 'COMMIT.json')
            commit['products'] = source_products(fhsc_root)
            (fhsc_root / 'COMMIT.json').write_bytes(m.canonical(commit))
            args['fhsc_commit_sha'] = m.digest(fhsc_root / 'COMMIT.json')
            authorities = m.load_authorities(**args, deep=True)
            with self.assertRaisesRegex(ValueError, 'F primary predictor missing'):
                m.assemble_rows(authorities)

    def test_schedule_ID_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            args, _, _, _ = self.fixture(Path(name), schedule_ids=['wrong'])
            with self.assertRaisesRegex(ValueError, 'schedule/prepared cohort'):
                m.load_authorities(**args)

    def test_component_hash_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name).resolve(); args, _, _, _ = self.fixture(root)
            authorities = m.load_authorities(**args, deep=True)
            authorities['components']['c']['members'] = ['changed']
            with self.assertRaisesRegex(ValueError, 'component/cohort'):
                m.assemble_rows(authorities)

    def test_package_mutation_is_rejected_by_public_verifier(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name).resolve(); args, _, _, _ = self.fixture(root)
            proof = m.publish(m.load_authorities(**args, deep=True), root / 'package')
            (root / 'package/features.json').write_text('[]\n')
            with self.assertRaisesRegex(ValueError, 'product binding changed'):
                m.verify_package(root / 'package', proof['commit']['sha256'])

    def reseal(self, package):
        commit = m.read_json(package / 'COMMIT.json')
        commit['products'] = {name: m.binding(package / name) for name in
                              ('contract.json', 'metadata.json', 'features.json', 'lineage.json')}
        (package / 'COMMIT.json').write_bytes(m.canonical(commit))
        return m.digest(package / 'COMMIT.json')

    def test_resealed_wrong_feature_value_is_rejected_by_source_replay(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name).resolve(); args, _, _, _ = self.fixture(root)
            m.publish(m.load_authorities(**args, deep=True), root / 'package')
            features = m.read_json(root / 'package/features.json')
            features[0][m.FAMILY_CONFIG['S'][0]] = 999.0
            (root / 'package/features.json').write_bytes(m.canonical(features))
            commit_sha = self.reseal(root / 'package')
            with self.assertRaisesRegex(ValueError, 'do not replay'):
                m.verify_package(root / 'package', commit_sha)

    def test_resealed_wrong_lineage_is_rejected_by_source_replay(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name).resolve(); args, _, _, _ = self.fixture(root)
            m.publish(m.load_authorities(**args, deep=True), root / 'package')
            lineage = m.read_json(root / 'package/lineage.json')
            lineage[0]['schedule_input_position'] = 99
            (root / 'package/lineage.json').write_bytes(m.canonical(lineage))
            commit_sha = self.reseal(root / 'package')
            with self.assertRaisesRegex(ValueError, 'do not replay'):
                m.verify_package(root / 'package', commit_sha)

    def test_resealed_wrong_metadata_is_rejected_by_source_replay(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name).resolve(); args, _, _, _ = self.fixture(root)
            m.publish(m.load_authorities(**args, deep=True), root / 'package')
            metadata = m.read_json(root / 'package/metadata.json')
            metadata[0]['native_sample_rate_hz'] = 44100
            (root / 'package/metadata.json').write_bytes(m.canonical(metadata))
            commit_sha = self.reseal(root / 'package')
            with self.assertRaisesRegex(ValueError, 'do not replay'):
                m.verify_package(root / 'package', commit_sha)

    def test_metadata_preflight_does_not_open_item_receipts(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name).resolve(); args, _, fhsc_root, sdrp_root = self.fixture(root)
            for path in (fhsc_root / 'items/x.json', sdrp_root / 'items/x.json'):
                path.write_bytes(b'x' * path.stat().st_size)
            authorities = m.load_authorities(**args, deep=False)
            self.assertEqual(len(authorities['rows']), 1)


if __name__ == '__main__':
    unittest.main()
