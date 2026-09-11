"""Synthetic input adapter tests; no live cohort data, audio decode, or fits."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import prepare_native30_evaluation_inputs_bc_v1 as m
import test_prepare_native30_evaluation_inputs_v3 as parent_tests


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    m.b.write_new(path, value)
    return m.binding(path)


class JoinTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name).resolve()
        args, row, _, _ = parent_tests.AssemblerTests().fixture(self.root)
        self.parent = m.b.publish(m.b.load_authorities(**args, deep=True), self.root / 'parent')
        self.run = {'prepared': {'rows': [row], 'output_root': str(self.root / 'bc'),
                    'native_origins': {'x': {'sample_rate_hz': 48000, 'channels': 2}}}}
        self.schedule = {'input_population': {'ids': ['x']}, 'eligible_population': {'ids': ['x']},
                         'protected_excluded_population': {'rows': 0}}
        self.payload = {'id': 'x', 'version': m.bc.VERSION, 'status': 'measured_BC_not_classifier_admitted',
            'row': row, 'row_sha256': m.value_hash(row), 'contract_sha256': m.value_hash(self.run),
            'native_origin': self.run['prepared']['native_origins']['x'], 'features': {m.bc.FEATURE: .25},
            'scalar': {'median_squared_bicoherence': .25, 'pool_count': 2, 'discarded_tail_samples': 0},
            'metadata': {'path': '/synthetic/metadata.json'}, 'arrays': {'path': '/synthetic/arrays.npz'},
            'analysis_view': {'synthetic': True}, **m.bc.SCOPE}
        self.seal()

    def seal(self):
        path = self.root / 'bc/items/x.json'
        value = {'payload': self.payload, 'receipt_sha256': m.value_hash(self.payload)}
        if path.exists(): path.write_bytes(m.canonical(value))
        else: put(path, value)

    def join(self):
        return m.join_rows(self.parent, self.run, self.schedule)

    def test_real_v3_deep_package_then_exact54_append_and_metadata_lineage_parity(self):
        before = m.canonical(self.parent)
        metadata, features, lineage = self.join()
        self.assertEqual(metadata, self.parent['metadata'])
        self.assertEqual(m.canonical({k: v for k, v in features[0].items() if k != m.bc.FEATURE}),
                         m.canonical(self.parent['features'][0]))
        self.assertEqual(set(features[0]), {'id', *m.FEATURE_NAMES})
        self.assertEqual(len(features[0]), 56)
        self.assertEqual({k: lineage[0][k] for k in m.b.LINEAGE_SCHEMA}, self.parent['lineage'][0])
        self.assertEqual(m.canonical(self.parent), before)
        self.assertFalse(m.SCOPE['fitting_authorized'])

    def test_null_is_retained_and_no_availability_filter(self):
        self.payload['features'][m.bc.FEATURE] = None
        self.payload['scalar']['median_squared_bicoherence'] = None; self.seal()
        metadata, features, _ = self.join()
        self.assertEqual(len(metadata), 1); self.assertIsNone(features[0][m.bc.FEATURE])

    def test_signed_zero_and_integer_types_of_old_values_are_preserved(self):
        self.parent['features'][0][m.b.FEATURE_NAMES[0]] = -0.0
        self.parent['features'][0][m.b.FEATURE_NAMES[1]] = 0
        features = self.join()[1]
        self.assertIn(b'-0.0', m.canonical(features))
        self.assertIs(type(features[0][m.b.FEATURE_NAMES[1]]), int)

    def test_wrong_input_pcm_row_group_rate_or_position_rejected(self):
        mutations = [lambda: self.parent['metadata'][0].update(input_sha256='0'*64),
            lambda: self.parent['metadata'][0].update(waveform_float32_sha256='0'*64),
            lambda: self.parent['metadata'][0].update(group_id='wrong'),
            lambda: self.parent['metadata'][0].update(native_sample_rate_hz=44100),
            lambda: self.parent['lineage'][0].update(schedule_input_position=8),
            lambda: self.payload.update(row_sha256='0'*64)]
        original = copy.deepcopy((self.parent, self.payload))
        for mutate in mutations:
            self.parent, self.payload = copy.deepcopy(original); mutate(); self.seal()
            with self.assertRaises(ValueError): self.join()

    def test_nonfinite_boolean_missing_extra_or_reduction_mismatch_rejected(self):
        original = copy.deepcopy(self.payload)
        for value in (True, '0.2', -1.0, 1.1):
            self.payload = copy.deepcopy(original); self.payload['features'][m.bc.FEATURE] = value; self.seal()
            with self.assertRaises(ValueError): self.join()
        for features in ({}, {m.bc.FEATURE: .25, 'eligible_pool_count': 2}, {m.bc.FEATURE: .5}):
            self.payload = copy.deepcopy(original); self.payload['features'] = features; self.seal()
            with self.assertRaises(ValueError): self.join()
        for value in (float('nan'), float('inf')):
            with self.assertRaises(ValueError): m.b.scalar(value, m.bc.FEATURE)

    def test_unknown_scope_or_status_and_wrong_pool_support_rejected(self):
        original = copy.deepcopy(self.payload)
        for change in ({'version': 'unknown'}, {'status': 'admitted'}, {'classifier_admission': True},
                       {'classifier_fits': 1}, {'scalar': {'median_squared_bicoherence': .25,
                                                        'pool_count': 7, 'discarded_tail_samples': 0}}):
            self.payload = {**copy.deepcopy(original), **change}; self.seal()
            with self.assertRaises(ValueError): self.join()

    def test_missing_duplicate_or_changed_schedule_ids_rejected(self):
        for ids in ([], ['wrong'], ['x', 'x']):
            self.schedule['input_population']['ids'] = ids
            with self.assertRaises(ValueError): self.join()

    def test_resealed_original_package_tamper_rejected_by_unchanged_deep_verifier(self):
        root = self.root / 'parent'; values = m.read(root / 'features.json')
        values[0][m.b.FEATURE_NAMES[0]] = 99.0
        (root / 'features.json').write_bytes(m.canonical(values))
        sha = parent_tests.AssemblerTests().reseal(root)
        with self.assertRaisesRegex(ValueError, 'do not replay'): m.b.verify_package(root, sha)

    def test_production_rejects_reduced_cohort_after_actual_v3_deep_verification(self):
        request = {'parent_commit': self.parent['commit']}
        with patch.object(m, 'validate_request'), \
             patch.object(m, 'verify_bc', side_effect=AssertionError('reduced cohort reached BC')), \
             self.assertRaisesRegex(ValueError, 'exact3830'):
            m.build(request)


class AuthorityTests(unittest.TestCase):
    def test_exact_dependency_pins_and_unknown_request_fail_closed(self):
        self.assertEqual(len(m.code_bindings()), 8)
        for request in ({}, {'latest': '/somewhere'}):
            with self.assertRaises(ValueError): m.validate_request(request)
        with patch.dict(m.PINS, {'run_native30_bc_v2.py': '0'*64}), self.assertRaises(ValueError):
            m.module('run_native30_bc_v2')

    def test_missing_gate_freeze_or_failed_gate_errors_propagate(self):
        request = {'bc_freeze': {'path': '/synthetic/freeze.json', 'sha256': '0'*64}}
        for error in ('missing freeze', 'scientific margins failed', 'prepared contract does not replay'):
            with patch.object(m.bc, 'validate_freeze', side_effect=ValueError(error)), \
                 patch.object(m.bc, 'verify_output', side_effect=AssertionError('must not read measurements')), \
                 self.assertRaisesRegex(ValueError, error): m.verify_bc(request)

    def test_complete_bc_verification_uses_no_audio_and_pinned_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            run = {'prepared': {'output_root': str(root), 'expected_count': 3830,
                               'source_counts': m.bc.physical.SOURCES}}
            commit = {'version': m.bc.VERSION, 'status': 'committed_native30_BC_measurements_not_admitted',
                'contract_sha256': m.value_hash(run), 'completed': 3830, 'products': {},
                'all_bound_inputs_and_products_end_rehashed': True, **m.bc.SCOPE}
            entry = put(root / 'COMMIT.json', commit)
            request = {'bc_freeze': {'path': '/synthetic/freeze.json', 'sha256': '0'*64}, 'bc_commit': entry}
            with patch.object(m.bc, 'validate_freeze', return_value=run), \
                 patch.object(m.bc, 'check_inputs') as inputs, \
                 patch.object(m.bc, 'verify_output', return_value={'commit': entry}) as output:
                m.verify_bc(request)
                self.assertEqual(inputs.call_count, 2)
                for call in inputs.call_args_list: self.assertFalse(call.kwargs['audio'])
                self.assertFalse(output.call_args.kwargs['audio'])
                wrong = copy.deepcopy(request); wrong['bc_commit']['sha256'] = 'f'*64
                with self.assertRaisesRegex(ValueError, 'COMMIT mismatch'): m.verify_bc(wrong)

    def test_schedule_projection_replay_and_tamper_invariants(self):
        from test_plan_native30_evaluation_schedule_bc_v1 import fixture, config
        rows, screen = fixture(); old = m.s.b.make_schedule(rows, screen)
        draft = m.s.derive_schedule(old, old, config())
        self.assertEqual(draft['feature_names'], m.FEATURE_NAMES)
        self.assertEqual(draft['cap_schedules'], old['cap_schedules'])
        self.assertEqual([c for c in draft['combination_schedule_cells'] if 'BC' not in c['combination'].split('+')],
                         old['combination_schedule_cells'])
        changed = copy.deepcopy(old); changed['cap_schedules'][0]['schedule_uid'] = 'changed'
        with self.assertRaises(ValueError): m.s.derive_schedule(changed, old, config())

    def test_source_change_during_publication_leaves_no_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve(); request = {
                key: {'path': str(root / key / (name or 'freeze.json'))} for key, name in m.REQUEST_NAMES.items()}
            values = {name: {'synthetic': True} for name in m.PRODUCT_NAMES}
            cohort = put(root / 'cohort.json', {})
            contract = put(root / 'parent/contract.json', {'cohort_contract': cohort, 'bindings': {
                k: {'path': str(root / k / 'contract.json')} for k in ('fhsc_contract', 'sdrp_contract', 'schedule_draft')}})
            values['contract.json']['parent_products'] = {'contract.json': contract}
            with patch.object(m, 'validate_request'), patch.object(m, 'build', side_effect=[values, {}]), \
                 self.assertRaisesRegex(ValueError, 'changed during assembly'):
                m.publish(request, root / 'output')
            self.assertFalse((root / 'output/COMMIT.json').exists())
            self.assertTrue((root / 'output/features.json').exists())

    def test_resealed_package_payload_change_cannot_bypass_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            request = {'synthetic': 'only the upstream build is stubbed in this unit test'}
            values = {name: {'synthetic': True} for name in m.PRODUCT_NAMES}
            values['contract.json']['request'] = request
            values['features.json'] = [{'id': 'x', m.bc.FEATURE: None}]
            for name, value in values.items(): put(root / name, value)
            def reseal():
                commit = {'version': m.VERSION, 'status': m.STATUS, 'expected_count': 3830,
                    'products': {name: m.binding(root / name) for name in m.PRODUCT_NAMES},
                    'request_sha256': m.value_hash(request), 'all_source_evidence_deep_verified_before_and_after': True,
                    'audio_files_opened': 0, **m.SCOPE}
                path = root / 'COMMIT.json'
                if path.exists(): path.write_bytes(m.canonical(commit))
                else: put(path, commit)
                return m.binding(path)['sha256']
            sha = reseal()
            with patch.object(m, 'build', return_value=values):
                self.assertIsNone(m.verify_package(root, sha)['features'][0][m.bc.FEATURE])
                (root / 'features.json').write_bytes(m.canonical([{'id': 'x', m.bc.FEATURE: .8}]))
                with self.assertRaisesRegex(ValueError, 'does not replay'): m.verify_package(root, reseal())


if __name__ == '__main__': unittest.main()
