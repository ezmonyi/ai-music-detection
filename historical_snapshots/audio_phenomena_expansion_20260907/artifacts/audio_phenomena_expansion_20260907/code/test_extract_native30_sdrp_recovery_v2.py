"""Synthetic waveform/provenance fixtures only; no real cohort or model runs."""
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import extract_native30_sdrp_recovery_v2 as m
ordinary_tests = m.o.load_module(m.HERE / 'test_extract_native30_sdrp_v1.py',
    'fa3c15ca8f3e97197f0379da9a7d4e5e36b7a4c2780ee5928893b1c52790ac69', 'test_extract_native30_sdrp_v1')


class RecoveryExtractionTests(unittest.TestCase):
    write_beat_receipt = ordinary_tests.ExtractionTests.write_beat_receipt
    audit = ordinary_tests.ExtractionTests.audit
    legacy = ordinary_tests.ExtractionTests.legacy

    @classmethod
    def setUpClass(cls):
        ordinary_tests.ExtractionTests.setUpClass.__func__(cls)

    def setUp(self):
        ordinary_tests.ExtractionTests.setUp(self)
        self.recovery = self.root / 'recovery'; self.recovery.mkdir(); (self.recovery / 'writer.lock').touch()
        self.contract.update(version=m.VERSION, recovery_root=str(self.recovery), **m.SCOPE)
        for key in ('recovery_parent_freeze', 'previous_continuation_freeze', 'torch_cuda_source'):
            path = self.root / (key + '.json'); m.base.write_new(path, {'synthetic': key})
            self.contract['bindings'][key] = m.base.file_binding(path)
        self.contract['bindings']['recovery_runner'] = m.base.file_binding(m.HERE / 'continue_native30_empty_beats_v2.py')
        for name in ('driver_run.json', 'driver_run_v2.json'):
            m.base.write_new(self.recovery / name, {'synthetic': name})
        authority = self.contract['bindings']['recovery_parent_freeze']
        self.contract['recovery_epoch'] = {'version': 'continue_native30_empty_beats_v2',
            'continuation_parent_freeze': {'path': authority['path'], 'sha256': authority['sha256']},
            'previous_continuation_freeze': self.contract['bindings']['previous_continuation_freeze'],
            'torch_cuda_source': self.contract['bindings']['torch_cuda_source'],
            'environment_transition': m.load_recovery().ENVIRONMENT_TRANSITION,
            'driver_execution': m.base.file_binding(self.recovery / 'driver_run_v2.json'),
            'historical_driver_execution': m.base.file_binding(self.recovery / 'driver_run.json')}
        self.provenance = {'kind': 'ordinary_v2', 'receipt': m.base.file_binding(self.receipt), 'item_status': 'ordinary_v2_success'}
        self.mapped['beat_provenance'] = self.provenance
        self.audited.update(beat_provenance={'synthetic': self.provenance}, recovery_evidence={}, recovered_ids=[], recovery_epoch=self.contract['recovery_epoch'])

    def recovered(self, missing=True):
        self.receipt = self.recovery / 'beats_000.json'; self.mapped['beat_receipt_path'] = str(self.receipt)
        evidence = {}
        for key in ('request', 'raw_replay', 'replay_log'):
            path = self.recovery / (key + '.json'); m.base.write_new(path, {'synthetic': key})
            evidence[key] = m.base.file_binding(path)
        evidence['missing_ids'] = ['synthetic'] if missing else []
        self.payload = {'status': 'recovered_serializer_v1', 'recovery_applied': True, 'ordinary_success': False,
                        'item_ids': ['synthetic'], **{k: evidence[k] for k in ('request', 'raw_replay', 'replay_log')}}
        authority = self.contract['bindings']['recovery_parent_freeze']
        self.payload['recovery_validation'] = {'validator': self.contract['bindings']['recovery_runner'],
            'parent_freeze': {'path': authority['path'], 'sha256': authority['sha256']}, 'raw_authority_epoch': 'v1_snapshot_bound',
            'environment_transition': self.contract['recovery_epoch']['environment_transition'],
            'torch_cuda_source': self.contract['bindings']['torch_cuda_source']}
        self.write_beat_receipt()
        self.provenance = {'kind': 'recovered_serializer_v1', 'receipt': m.base.file_binding(self.receipt),
                           'raw_replay': evidence['raw_replay'], 'item_status': 'recovered_empty_unavailable' if missing else 'preserved_cli_success'}
        self.mapped['beat_provenance'] = self.provenance
        self.audited = self.audit()
        self.audited.update(beat_provenance={'synthetic': self.provenance}, recovery_evidence={str(self.receipt): evidence},
                            recovered_ids=['synthetic'] if missing else [], recovery_epoch=self.contract['recovery_epoch'])

    def run_fixture(self):
        return m.run_contract(self.contract, self.extractor, self.bias, lambda _: self.audited)

    def test_pinned_backends_unchanged(self):
        self.assertEqual(m.base.digest(m.ordinary_path), m.ORDINARY_SHA)
        self.assertEqual(m.base.digest(m.HERE / 'continue_native30_empty_beats_v2.py'), m.RECOVERY_SHA)
        self.assertEqual(m.base.digest(m.HERE / 'extract_native30_sdrp_recovery_v1.py'), '32c861d3dbe17ac41f0ccfbdc82f63b33babb2d035b94b65fee914f7124107f9')
        self.assertEqual(len(m.o.feature_names(self.extractor)), 43)

    def test_ordinary_branch_reuses_strict_existing_gate(self):
        inputs = m.inspect_inputs(self.contract, self.mapped, self.audited)
        self.assertEqual(inputs['beat_provenance']['kind'], 'ordinary_v2')
        self.payload['recovery_applied'] = True; self.write_beat_receipt()
        with self.assertRaises(ValueError): m.inspect_inputs(self.contract, self.mapped, self.audited)

    def test_ordinary_item_numerics_unchanged_in_separate_backend(self):
        baseline = m.o.clean(self.legacy()); self.assertEqual(self.run_fixture()['completed'], 1)
        path = Path(self.contract['output_root']) / 'items/synthetic.json'
        item = m.base.read_json(path)['payload']
        self.assertEqual(item['legacy_result'], baseline)
        self.assertEqual(item['inputs']['beat_provenance']['item_status'], 'ordinary_v2_success')
        item['inputs']['beat_provenance']['item_status'] = 'recovered_empty_unavailable'
        path.unlink(); m.base.write_new(path, {'payload': item, 'receipt_sha256': m.base.value_hash(item)})
        with self.assertRaisesRegex(ValueError, 'existing COMMIT changed'): self.run_fixture()

    def test_recovered_empty_is_distinct_scientific_missing_no_new_predictors(self):
        self.recovered(); baseline = m.o.clean(self.legacy())
        result = self.run_fixture(); self.assertEqual(result['completed'], 1)
        output = Path(self.contract['output_root']); receipt = m.base.read_json(output / 'items/synthetic.json')['payload']
        self.assertEqual(receipt['legacy_result'], baseline)
        self.assertEqual(receipt['inputs']['beat_provenance']['item_status'], 'recovered_empty_unavailable')
        self.assertEqual(receipt['legacy_result']['r_eligible'], 0)
        self.assertTrue(all(receipt['legacy_result']['r__' + name] is None for name in self.extractor.R_FEATURES))
        self.assertEqual(set(k for k in receipt['legacy_result'] if k.startswith(m.o.PREFIXES)), set(m.o.feature_names(self.extractor)))
        self.assertNotIn('NaN', (output / 'items/synthetic.json').read_text())
        commit = m.base.read_json(output / 'COMMIT.json')
        self.assertEqual(commit['status'], m.COMMIT_STATUS); self.assertFalse(commit['recovery_applied'])
        self.assertTrue(commit['recovery_evidence_consumed']); self.assertFalse(commit['recovery_provenance_is_predictor'])
        self.assertEqual(m.base.read_json(output / 'manifest.json')['recovered_ids'], ['synthetic'])
        before = m.o.products(output)
        with patch.object(self.extractor, 'process', side_effect=AssertionError('must not recompute')):
            self.assertEqual(self.run_fixture()['status'], 'verified_existing_COMMIT')
        self.assertEqual(before, m.o.products(output))

    def test_preserved_success_has_identical_nonempty_numerics(self):
        self.beat.write_text('0.5\t1\n1.0\t2\n1.5\t3\n2.0\t4\n2.5\t1\n3.0\t2\n')
        self.recovered(missing=False)
        baseline = m.o.clean(self.legacy()); self.assertEqual(self.run_fixture()['completed'], 1)
        item = m.base.read_json(Path(self.contract['output_root']) / 'items/synthetic.json')['payload']
        self.assertEqual(item['legacy_result'], baseline)
        self.assertEqual(item['inputs']['beat_provenance']['item_status'], 'preserved_cli_success')

    def test_recovered_cannot_pass_ordinary_gate(self):
        self.recovered()
        with self.assertRaisesRegex(ValueError, 'successful nonrecovery'):
            m.o.inspect_inputs(self.contract, self.mapped, self.audited)

    def test_repaired_nonempty_output_rejected_even_if_hashes_rebound(self):
        self.beat.write_text('1.0\t1\n'); self.recovered()
        with self.assertRaisesRegex(ValueError, 'genuinely empty'): m.inspect_inputs(self.contract, self.mapped, self.audited)

    def test_provenance_type_and_set_mutations_rejected(self):
        self.recovered()
        for mutation in (lambda a: a['beat_provenance']['synthetic'].update(item_status='preserved_cli_success'),
                         lambda a: a['beat_provenance']['synthetic'].update(kind='ordinary_v2'),
                         lambda a: a.update(recovered_ids=[]),
                         lambda a: a['recovery_evidence'][str(self.receipt)].update(missing_ids=[])):
            changed = copy.deepcopy(self.audited); mutation(changed)
            with self.assertRaises(ValueError): m.inspect_inputs(self.contract, self.mapped, changed)

    def test_raw_sidecar_mutation_rejected_after_extraction(self):
        self.recovered(); inputs = m.inspect_inputs(self.contract, self.mapped, self.audited)
        Path(self.provenance['raw_replay']['path']).write_text('changed raw arrays')
        with self.assertRaisesRegex(ValueError, 'recovery evidence changed'): m.recheck_item_inputs(inputs)

    def test_physical_duration_and_input_float_gates_unchanged(self):
        import soundfile as sf
        self.recovered()
        for samples, subtype in ((self.samples[:-1], 'FLOAT'), (self.samples, 'PCM_16')):
            sf.write(self.source, samples, 44100, subtype=subtype)
            with self.assertRaises(ValueError): m.inspect_inputs(self.contract, self.mapped, self.audited)

    def test_failure_is_retained_without_commit(self):
        self.recovered()
        with patch.object(self.extractor, 'process', side_effect=RuntimeError('synthetic failure')):
            result = self.run_fixture()
        self.assertEqual(result['status'], 'partial_no_COMMIT')
        output = Path(self.contract['output_root']); self.assertFalse((output / 'COMMIT.json').exists())
        self.assertEqual(len(list((output / 'failures').iterdir())), 1)

    def test_derivative_verifier_binding_and_provenance_are_not_trusted_without_check(self):
        self.recovered()
        parent = self.root / 'parent.json'; m.base.write_new(parent, {'synthetic': 'authority'}); binding = m.base.file_binding(parent)
        self.contract['bindings'].update(inference_parent_freeze=binding, cohort_contract=binding)
        completion = {**self.audited['evidence'], 'beat_provenance': self.audited['beat_provenance'],
                      'recovery_evidence': self.audited['recovery_evidence'], 'recovered_ids': self.audited['recovered_ids'],
                      'original_parent_freeze': binding, 'cohort_contract': binding, **self.contract['recovery_epoch']}
        path = self.root / 'completion.json'; m.base.write_new(path, completion)
        self.contract['bindings']['inference_completion'] = m.base.file_binding(path)
        proof = {'status': 'verified_complete_native30_inference_with_frozen_recovery', 'rows': 1,
                 'completion': m.base.file_binding(path), 'evidence_sha256': m.base.value_hash(self.audited['evidence']),
                 'beat_provenance': self.audited['beat_provenance']}
        backend = SimpleNamespace(verify_completion=lambda *args: proof, ENVIRONMENT_TRANSITION=m.load_recovery().ENVIRONMENT_TRANSITION)
        with patch.object(m, 'load_recovery', return_value=backend):
            audited = m.verify_completed_inference(self.contract)
            self.assertEqual(audited['recovered_ids'], ['synthetic'])
            proof['evidence_sha256'] = '0' * 64
            with self.assertRaisesRegex(ValueError, 'audited evidence'): m.verify_completed_inference(self.contract)

    def test_completion_epoch_rejects_v1_wrong_freeze_and_swapped_drivers(self):
        valid = self.contract['recovery_epoch']
        self.assertEqual(m.completion_epoch(valid, self.contract['bindings'], self.recovery), valid)
        mutations = (lambda x: x.update(version='continue_native30_empty_beats_v1'),
                     lambda x: x.update(previous_continuation_freeze=self.contract['bindings']['recovery_parent_freeze']),
                     lambda x: x['continuation_parent_freeze'].update(sha256='0' * 64),
                     lambda x: x.update(driver_execution=x['historical_driver_execution']),
                     lambda x: x.update(environment_transition={}))
        for mutation in mutations:
            changed = copy.deepcopy(valid); mutation(changed)
            with self.assertRaises(ValueError): m.completion_epoch(changed, self.contract['bindings'], self.recovery)

    def test_recovered_receipt_rejects_v1_validator_and_wrong_v2_parent(self):
        self.recovered(); baseline = copy.deepcopy(self.payload)
        for mutation in (lambda x: x.pop('recovery_validation'),
                         lambda x: x['recovery_validation'].update(validator=m.base.file_binding(m.HERE / 'continue_native30_empty_beats_v1.py')),
                         lambda x: x['recovery_validation']['parent_freeze'].update(sha256='0' * 64),
                         lambda x: x['recovery_validation'].update(raw_authority_epoch='ordinary_success')):
            self.payload = copy.deepcopy(baseline); mutation(self.payload); self.write_beat_receipt()
            binding = m.base.file_binding(self.receipt)
            self.provenance['receipt'] = binding
            self.audited['evidence']['stage_receipts'][str(self.receipt)] = {'binding': binding, 'payload_sha256': m.base.value_hash(self.payload)}
            with self.assertRaises(ValueError): m.inspect_inputs(self.contract, self.mapped, self.audited)

    def test_new_v2_recorded_raw_epoch_retained_only_as_metadata(self):
        self.recovered(); self.payload['recovery_validation']['raw_authority_epoch'] = 'v2_recorded'; self.write_beat_receipt()
        binding = m.base.file_binding(self.receipt); self.provenance['receipt'] = binding
        self.audited['evidence']['stage_receipts'][str(self.receipt)] = {'binding': binding, 'payload_sha256': m.base.value_hash(self.payload)}
        inputs = m.inspect_inputs(self.contract, self.mapped, self.audited)
        self.assertEqual(inputs['recovery_validation']['raw_authority_epoch'], 'v2_recorded')
        self.assertNotIn('recovery_validation', inputs['extractor_input_hashes'])

    def test_frozen_54_downstream_columns_unchanged_and_new_backend_not_implicitly_allowed(self):
        assembler = m.o.load_module(m.HERE / 'prepare_native30_evaluation_inputs_v1.py',
            'c1b5e223d5ef2951fc93b12da9d82c75f2cf616418d7b7f8704d4225a4c95bb7', 'prepare_native30_evaluation_inputs_v1')
        self.assertEqual(len(assembler.FEATURE_NAMES), 54)
        self.assertEqual({k: len(v) for k, v in assembler.FAMILY_CONFIG.items()}, {'S': 15, 'D': 3, 'R': 3, 'P': 6, 'F': 15, 'H': 6, 'SC': 6})
        descriptor_names = set(m.o.feature_names(self.extractor))
        selected = set(name for family in ('S', 'D', 'R', 'P') for name in assembler.FAMILY_CONFIG[family])
        self.assertEqual(len(selected), 27); self.assertTrue(selected <= descriptor_names)
        self.assertEqual(len(descriptor_names), 43)
        self.assertFalse(any('recovery' in name or 'environment' in name or 'eligible' in name for name in assembler.FEATURE_NAMES))
        self.assertNotIn(m.VERSION, assembler.SDRP_BACKENDS)

    def test_genuine_v2_checker_completion_shape_consumed_without_relaxed_mock_proof(self):
        fixture_module = m.o.load_module(m.HERE / 'test_continue_native30_empty_beats_v2.py',
            'd73cde748119301e2a878301e5ac0309ec0e291fc9f6577e359d8f466b6e7a6f', 'test_continue_native30_empty_beats_v2')
        fixture = fixture_module.EpochTests(); fixture.setUp(); self.addCleanup(fixture.doCleanups)
        driver = m.load_recovery(); frozen = Path(fixture.freeze['_path'])
        driver.b.write_new(frozen, {k: v for k, v in fixture.freeze.items() if not k.startswith('_')})
        fixture.freeze['_sha256'] = driver.b.digest(frozen)
        runtime = {'frozen': True}; cohort = {'rows': fixture.f.rows}
        driver.put(fixture.f.root / 'driver_run_v2.json', driver.driver_record(fixture.freeze, runtime))
        driver.recover_shard(fixture.freeze, fixture.f.prior, fixture.f.shard, fixture.f.rows, fixture.f.root)
        context = (fixture.freeze, fixture.f.prior, cohort, None, runtime)
        with patch.object(driver, 'setup_context', return_value=context), patch.object(driver.v2, 'verify_runtime', return_value=runtime), \
             patch.object(driver.v2, 'rehash_source_graph', return_value={'frozen': True}):
            expected = driver.final_records(*context)
            audit = fixture.f.root / 'final_audit.json'; completion = fixture.f.root / 'completion.json'
            driver.b.write_new(audit, expected); driver.b.write_new(completion, {**expected, 'final_audit': driver.b.binding(audit)})
            bindings = {'recovery_parent_freeze': driver.b.binding(frozen), 'inference_completion': driver.b.binding(completion),
                        'inference_parent_freeze': fixture.freeze['original_parent_freeze'], 'cohort_contract': fixture.f.prior['cohort_contract'],
                        'previous_continuation_freeze': fixture.freeze['previous_continuation_freeze'],
                        'torch_cuda_source': fixture.freeze['torch_cuda_source']}
            contract = {'bindings': bindings, 'expected_count': 2, 'recovery_root': str(fixture.f.root),
                        'rows': [{'cohort_row': row, 'beat_provenance': expected['beat_provenance'][row['id']]} for row in fixture.f.rows],
                        'recovery_epoch': m.completion_epoch(expected, bindings, fixture.f.root)}
            verified = m.verify_completed_inference(contract)
            self.assertEqual(verified['verification']['completion'], driver.b.binding(completion))
            self.assertEqual(verified['recovered_ids'], ['b'])
            self.assertEqual(verified['recovery_epoch']['version'], 'continue_native30_empty_beats_v2')


if __name__ == '__main__':
    unittest.main()
