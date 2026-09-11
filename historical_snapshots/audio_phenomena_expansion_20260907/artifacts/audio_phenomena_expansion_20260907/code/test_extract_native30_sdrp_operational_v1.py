"""Synthetic waveform/provenance fixtures only; no real cohort or model runs."""
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import extract_native30_sdrp_operational_v1 as m
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
        for key in ('recovery_parent_freeze', 'previous_continuation_freeze', 'torch_cuda_source', 'operational_parent_freeze'):
            path = self.root / (key + '.json'); m.base.write_new(path, {'synthetic': key})
            self.contract['bindings'][key] = m.base.file_binding(path)
        self.contract['bindings']['recovery_runner'] = m.base.file_binding(m.HERE / 'continue_native30_empty_beats_v2.py')
        self.operational = self.root / 'operational'; self.operational.mkdir()
        (self.operational / 'writer.lock').touch(); (self.operational / 'dispatch').mkdir()
        self.contract['operational_root'] = str(self.operational)
        self.contract['bindings']['operational_runner'] = m.base.file_binding(m.HERE / 'resume_native30_gpu_guard_v2.py')
        for name in ('driver_run.json', 'blocked_attempt.json'):
            m.base.write_new(self.operational / name, {'synthetic': name})
        authority = self.contract['bindings']['operational_parent_freeze']
        complete = {'version': m.load_operational().VERSION, 'status': m.load_operational().STATUS,
            'base_recovery_v2_status': 'partial_preserved_without_own_completion',
            'operational_parent_freeze': {'path': authority['path'], 'sha256': authority['sha256']},
            'base_freeze': self.contract['bindings']['recovery_parent_freeze'],
            'base_driver': self.contract['bindings']['recovery_runner'], 'placement': m.load_operational().PLACEMENT,
            'driver_execution': m.base.file_binding(self.operational / 'driver_run.json'),
            'blocked_attempt': m.base.file_binding(self.operational / 'blocked_attempt.json'),
            'replacement_attempt_096': {'synthetic': 'replacement'}}
        self.contract['recovery_epoch'] = m.completion_epoch(complete, self.contract['bindings'], self.operational)
        self.provenance = {'execution_epoch': 'preserved_recovery_v2', 'kind': 'ordinary_v2', 'receipt': m.base.file_binding(self.receipt), 'item_status': 'ordinary_v2_success'}
        self.mapped['beat_provenance'] = self.provenance
        self.audited.update(beat_provenance={'synthetic': self.provenance}, recovery_evidence={}, recovered_ids=[], recovery_epoch=self.contract['recovery_epoch'], stage_executions={})

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
        self.provenance = {'execution_epoch': 'preserved_recovery_v2', 'kind': 'recovered_serializer_v1', 'receipt': m.base.file_binding(self.receipt),
                           'raw_replay': evidence['raw_replay'], 'item_status': 'recovered_empty_unavailable' if missing else 'preserved_cli_success'}
        self.mapped['beat_provenance'] = self.provenance
        self.audited = self.audit()
        self.audited.update(beat_provenance={'synthetic': self.provenance}, recovery_evidence={str(self.receipt): evidence},
                            recovered_ids=['synthetic'] if missing else [], recovery_epoch=self.contract['recovery_epoch'], stage_executions={})

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
                      'recovery_evidence': self.audited['recovery_evidence'], 'recovered_ids': self.audited['recovered_ids'], 'stage_executions': {},
                      'original_parent_freeze': binding, 'cohort_contract': binding, **self.contract['recovery_epoch']}
        path = self.root / 'completion.json'; m.base.write_new(path, completion)
        self.contract['bindings']['inference_completion'] = m.base.file_binding(path)
        proof = {'status': 'verified_complete_native30_operational_continuation', 'rows': 1,
                 'completion': m.base.file_binding(path), 'evidence_sha256': m.base.value_hash(self.audited['evidence']),
                 'beat_provenance': self.audited['beat_provenance']}
        backend = m.load_operational()
        with patch.object(backend, 'verify_completion', return_value=proof):
            audited = m.verify_completed_inference(self.contract)
            self.assertEqual(audited['recovered_ids'], ['synthetic'])
            proof['evidence_sha256'] = '0' * 64
            with self.assertRaisesRegex(ValueError, 'audited evidence'): m.verify_completed_inference(self.contract)

    def test_completion_epoch_rejects_old_wrong_parent_placement_or_driver(self):
        valid = self.contract['recovery_epoch']
        self.assertEqual(m.completion_epoch(valid, self.contract['bindings'], self.operational), valid)
        mutations = (lambda x: x.update(version='continue_native30_empty_beats_v2'),
                     lambda x: x.update(base_freeze=self.contract['bindings']['operational_parent_freeze']),
                     lambda x: x['operational_parent_freeze'].update(sha256='0' * 64),
                     lambda x: x.update(driver_execution=x['blocked_attempt']),
                     lambda x: x.update(placement={}))
        for mutation in mutations:
            changed = copy.deepcopy(valid); mutation(changed)
            with self.assertRaises(ValueError): m.completion_epoch(changed, self.contract['bindings'], self.operational)

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


    def operational_item(self, recovered=True):
        if recovered:
            self.recovered()
            authority = self.contract['bindings']['operational_parent_freeze']
            self.payload['recovery_validation'].update(
                validator=self.contract['bindings']['operational_runner'],
                parent_freeze={'path': authority['path'], 'sha256': authority['sha256']},
                raw_authority_epoch='v2_recorded')
            self.write_beat_receipt()
            self.provenance['receipt'] = m.base.file_binding(self.receipt)
            self.audited['evidence']['stage_receipts'][str(self.receipt)] = {
                'binding': self.provenance['receipt'], 'payload_sha256': m.base.value_hash(self.payload)}
        authority = self.contract['bindings']['operational_parent_freeze']
        record = {'status': 'operational_physical_dispatch', 'driver': self.contract['bindings']['operational_runner'],
            'parent_freeze': {'path': authority['path'], 'sha256': authority['sha256']},
            'host': m.load_operational().HOST, 'placement': m.load_operational().PLACEMENT,
            'stage': 'beats', 'item_ids': ['synthetic']}
        path = self.operational / 'dispatch/beats_000.json'
        m.base.write_new(path, {'payload': record, 'receipt_sha256': m.base.value_hash(record)})
        self.provenance.update(execution_epoch='operational_dispatch_v2', physical_dispatch=m.base.file_binding(path))
        self.audited['stage_executions'][str(self.receipt)] = {
            'execution_epoch': 'operational_dispatch_v2', 'dispatch': self.provenance['physical_dispatch']}

    def test_new_operational_empty_numeric_parity_and_dispatch_excluded(self):
        self.operational_item()
        baseline = m.o.clean(self.legacy())
        self.assertEqual(self.run_fixture()['completed'], 1)
        item = m.base.read_json(Path(self.contract['output_root']) / 'items/synthetic.json')['payload']
        self.assertEqual(item['legacy_result'], baseline)
        self.assertEqual(item['inputs']['recovery_validation']['raw_authority_epoch'], 'v2_recorded')
        self.assertIsNotNone(item['inputs']['physical_dispatch'])
        self.assertNotIn('physical_dispatch', item['legacy_result'])
        self.assertNotIn('execution_epoch', item['legacy_result'])

    def test_new_ordinary_dispatch_numeric_parity(self):
        self.operational_item(recovered=False)
        baseline = m.o.clean(self.legacy())
        self.assertEqual(self.run_fixture()['completed'], 1)
        item = m.base.read_json(Path(self.contract['output_root']) / 'items/synthetic.json')['payload']
        self.assertEqual(item['legacy_result'], baseline)
        self.assertEqual(item['inputs']['beat_provenance']['kind'], 'ordinary_v2')

    def test_new_dispatch_and_historical_epoch_mutations_fail(self):
        self.operational_item()
        for mutate in (lambda a: a['stage_executions'].clear(),
                       lambda a: a['beat_provenance']['synthetic'].pop('physical_dispatch'),
                       lambda a: a['beat_provenance']['synthetic'].update(execution_epoch='untyped')):
            changed = copy.deepcopy(self.audited); mutate(changed)
            with self.assertRaises(ValueError): m.inspect_inputs(self.contract, self.mapped, changed)
        self.payload['recovery_validation']['raw_authority_epoch'] = 'v1_snapshot_bound'
        self.write_beat_receipt()
        self.provenance['receipt'] = m.base.file_binding(self.receipt)
        self.audited['evidence']['stage_receipts'][str(self.receipt)] = {
            'binding': self.provenance['receipt'], 'payload_sha256': m.base.value_hash(self.payload)}
        with self.assertRaisesRegex(ValueError, 'must be recorded'): m.inspect_inputs(self.contract, self.mapped, self.audited)

    def test_dispatch_mutation_after_source_check_and_three_upstream_locks(self):
        import fcntl
        self.operational_item()
        inputs = m.inspect_inputs(self.contract, self.mapped, self.audited)
        with (self.operational / 'writer.lock').open('r+b') as writer:
            fcntl.flock(writer, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError): self.run_fixture()
        Path(self.provenance['physical_dispatch']['path']).write_text('changed')
        with self.assertRaisesRegex(ValueError, 'dispatch changed'): m.recheck_item_inputs(inputs)




# Executes the approved source receipt/dispatch/inventory/canonical-completion
# checkers. Only expensive runtime/source-graph fixtures and subprocesses are
# synthetic; no verifier status, final_records, stage_evidence or audit is mocked.
operational_tests = m.o.load_module(m.HERE / 'test_resume_native30_gpu_guard_v2.py',
    '56357e706d14d57bb92e69aa32a07123098750f30fad4b4f34145dd8fd3b149e', 'test_resume_native30_gpu_guard_v2')
s = m.load_operational()


class SourceCheckerTests(unittest.TestCase):
    setUp = operational_tests.OperationalTests.setUp
    launch = operational_tests.OperationalTests.launch
    fake_command = operational_tests.OperationalTests.fake_command

    def test_actual_operational_checker_completion_and_tamper_consumed(self):
        frozen = Path(self.f['_path']); s.b.write_new(frozen, {'synthetic': 'operational authority'})
        self.f['_sha256'] = s.b.digest(frozen)
        self.launch('allinone'); self.launch('beats'); runtime = {'frozen': True}; cohort = {'rows': self.x.rows}
        parent = self.x.base / 'parent.json'; s.b.write_new(parent, {'synthetic': 'parent'})
        self.base.update(original_parent_freeze=s.b.binding(parent), _previous={})
        self.prior['cohort_contract'] = s.b.binding(parent)
        self.f.update(terminal_log=s.b.binding(parent), blocked_intent=s.b.binding(parent))
        # Existing intents retain the actual freeze SHA used for dispatch; the fixture pins the same synthetic value.
        s.r.put(self.oldroot / 'driver_run.json', {'v1': True}); s.r.put(self.oldroot / 'driver_run_v2.json', {'v2': True})
        self.f['stopped_recovery_inventory'] = {p: s.b.binding(p) for p in s.r.files(self.oldroot)}
        s.r.put(self.root / 'driver_run.json', s.driver_record(self.f, runtime))
        context = (self.f, self.prior, self.effective, cohort, None, runtime)
        real_require_hash = s.b.require_hash
        def hash_fixture(path, sha):
            if str(path) == str(frozen) and sha == self.f['_sha256']: return
            return real_require_hash(path, sha)
        with patch.object(s, 'setup_context', return_value=context), patch.object(s.b, 'require_hash', side_effect=hash_fixture), \
             patch.object(s, 'aborted_attempt', return_value=self.certificate), patch.object(s.v2, 'rehash_source_graph', return_value={'frozen': True}), \
             patch.object(s.v2, 'verify_runtime', return_value=runtime), patch.object(s.r, 'driver_record', return_value={'v2': True}), \
             patch.object(s.r, 'load_v1', return_value=SimpleNamespace(driver_record=lambda *a: {'v1': True})):
            expected = s.final_records(*context); audit = self.root / 'final_audit.json'; completion = self.root / 'completion.json'
            s.b.write_new(audit, expected); s.b.write_new(completion, {**expected, 'final_audit': s.b.binding(audit)})
            proof = s.verify_completion(frozen, self.f['_sha256']); self.assertEqual(proof['rows'], 2)
            self.assertEqual(expected['replacement_attempt_096']['intent']['path'], str(self.root / 'executions/beats_096.json'))
            self.assertEqual(expected['base_recovery_v2_status'], 'partial_preserved_without_own_completion')
            self.assertFalse((self.oldroot / 'completion.json').exists())
            bindings = {'operational_parent_freeze': s.b.binding(frozen),
                'operational_runner': self.f['driver'], 'recovery_parent_freeze': self.f['base_freeze'],
                'recovery_runner': self.f['base_driver'], 'inference_parent_freeze': self.base['original_parent_freeze'],
                'cohort_contract': self.prior['cohort_contract'], 'inference_completion': s.b.binding(completion)}
            contract = {'bindings': bindings, 'expected_count': 2, 'operational_root': str(self.root),
                'rows': [{'cohort_row': row, 'beat_provenance': expected['beat_provenance'][row['id']]}
                         for row in self.x.rows], 'recovery_epoch': m.completion_epoch(expected, bindings, self.root)}
            audited = m.verify_completed_inference(contract)
            self.assertEqual(audited['verification'], proof)
            self.assertEqual(audited['recovered_ids'], [])
            for row in self.x.rows:
                provenance = audited['beat_provenance'][row['id']]
                mapped = {'cohort_row': row, 'beat_provenance': provenance, 'beat_receipt_path': provenance['receipt']['path']}
                self.assertEqual(m.item_epoch(contract, mapped, audited)[0], 'operational_dispatch_v2')
            altered = {**expected, 'placement': {}}; audit.unlink(); completion.unlink()
            s.b.write_new(audit, altered); s.b.write_new(completion, {**altered, 'final_audit': s.b.binding(audit)})
            with self.assertRaisesRegex(ValueError, 'JSON value'): s.verify_completion(frozen, self.f['_sha256'])

    def test_historical_and_new_recovered_graph_duplicate_and_inventory_rejections(self):
        x = operational_tests.epoch_fixtures.EpochTests(); x.setUp(); self.addCleanup(x.doCleanups)
        base, prior, oldroot, original = x.freeze, x.f.prior, x.f.root, x.f.original
        s.r.put(oldroot / 'driver_run_v2.json', s.r.driver_record(base, {'frozen': True}))
        s.r.recover_shard(base, prior, x.f.shard, x.f.rows, oldroot)
        root = x.f.base / 'operational'; root.mkdir()
        for folder in ('dispatch', 'executions', 'execution_results', 'recovery_requests', 'raw_replays', 'replay_logs', 'recovered_receipts'):
            (root / folder).mkdir()
        newrows = [{**row, 'id': ident} for row, ident in zip(x.f.rows, ('c', 'd'))]
        shard = {'index': 96, 'rows': 2, 'item_ids': ['c', 'd'], 'item_ids_sha256': s.b.value_hash(['c', 'd'])}
        prior = {**prior, 'shards': [x.f.shard, shard]}; effective = {**prior, 'beat_gpus': [6]}
        f = {**base, 'output_root': str(root), '_base': base, '_path': str(x.f.base / 'operational.json'), '_sha256': '7' * 64,
             'driver': s.b.binding(Path(s.__file__).resolve()), 'resume_continuation_inventory': {},
             'stopped_original_inventory': {p: s.b.binding(p) for p in s.r.files(original)},
             'stopped_recovery_inventory': {p: s.b.binding(p) for p in s.r.files(oldroot)}}
        f['initial_partial_inventory'] = f['stopped_original_inventory']
        s.r.put(root / 'blocked_attempt.json', {'synthetic': 'blocked'}); s.r.put(root / 'driver_run.json', {'synthetic': True})
        def fake(command, **kwargs):
            stream = kwargs['stdout']
            if '--request' in command:
                request_path = Path(command[command.index('--request') + 1]); plan = s.r.get(request_path)
                stream.write('synthetic raw replay\n'); env = s.v2.child_environment(x.f.runtime, 6)
                raw = {**x.raw, 'request': s.b.binding(request_path), 'gpu': 6, 'child_environment_before_cuda_init': env,
                       'child_environment': {**env, 'CUDA_MODULE_LOADING': 'LAZY'}, 'environment_transition': s.r.ENVIRONMENT_TRANSITION,
                       'driver_sha256': f['driver']['sha256'], 'recovery_invocation': command,
                       'records': [{**x.raw['records'][0], 'id': 'd', 'input': plan['inputs']['d']}]}
                s.r.put(root / 'raw_replays/beats_096.json', raw)
            elif 'beat_this' in ' '.join(command):
                (original / 'beats/c.beats').write_text('0.5\t1\n')
                stream.write('Could not process "' + newrows[1]['input']['path'] + '". Rerun with this file alone for details.\n')
            else:
                stream.write('synthetic AIO success\n')
                for row in newrows:
                    for p in s.b.product_paths('allinone', row['id'], original):
                        p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(b'synthetic product')
            return SimpleNamespace(returncode=0)
        with patch.object(s.socket, 'gethostname', return_value=s.HOST), patch.object(s.b, 'require_idle_5090'), \
             patch.object(s.r.subprocess, 'run', side_effect=fake):
            for stage in ('allinone', 'beats'): s.launch_stage(f, prior, effective, stage, shard, newrows)
        cohort = {'rows': x.f.rows + newrows}
        ev, provenance, executions, recovery = s.audit_all(f, prior, effective, cohort)
        self.assertEqual(len(ev['products']), 28); self.assertEqual(len(recovery), 2)
        self.assertEqual(provenance['b']['item_status'], 'recovered_empty_unavailable')
        self.assertEqual(provenance['b']['execution_epoch'], 'preserved_recovery_v2')
        self.assertEqual(provenance['a']['item_status'], 'preserved_cli_success')
        self.assertEqual(provenance['d']['item_status'], 'recovered_empty_unavailable')
        self.assertEqual(provenance['d']['execution_epoch'], 'operational_dispatch_v2')
        self.assertEqual(provenance['c']['item_status'], 'preserved_cli_success')
        self.assertEqual(s.r.get(root / 'recovered_receipts/beats_096.json')['recovery_validation']['validator'], f['driver'])
        bindings = {'operational_parent_freeze': {'path': f['_path'], 'sha256': f['_sha256']},
                    'operational_runner': f['driver']}
        contract = {'bindings': bindings, 'operational_root': str(root)}
        audited = {'stage_executions': executions}
        for row in cohort['rows']:
            mapped = {'cohort_row': row, 'beat_provenance': provenance[row['id']],
                      'beat_receipt_path': provenance[row['id']]['receipt']['path']}
            epoch, dispatch = m.item_epoch(contract, mapped, audited)
            self.assertEqual(epoch, 'preserved_recovery_v2' if row['id'] in ('a', 'b') else 'operational_dispatch_v2')
            self.assertEqual(dispatch is None, row['id'] in ('a', 'b'))
        historical = {p: s.b.digest(p) for p in s.r.files(oldroot)}
        self.assertEqual(s.audit_all(f, prior, effective, cohort)[0], ev)
        self.assertEqual(historical, {p: s.b.digest(p) for p in s.r.files(oldroot)})
        duplicate = s.v2.receipt_path(original, 'beats', 96); s.r.put(duplicate, {})
        with self.assertRaisesRegex(ValueError, 'duplicate'): s.audit_all(f, prior, effective, cohort)
        duplicate.unlink(); extra = original / 'unexpected'; extra.write_bytes(b'x')
        with self.assertRaisesRegex(ValueError, 'exhaustive'): s.audit_all(f, prior, effective, cohort)
        extra.unlink(); missing = original / 'beats/d.beats'; missing.unlink()
        with self.assertRaises((ValueError, FileNotFoundError)): s.audit_all(f, prior, effective, cohort)

if __name__ == '__main__':
    unittest.main()
