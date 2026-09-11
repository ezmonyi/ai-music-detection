"""Synthetic recovery/continuation validation only: no model or audio reads."""
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

import continue_native30_empty_beats_v1 as r


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.original, self.root, self.runtime = [self.base / x for x in ('original', 'continuation', 'runtime')]
        for path in (self.original, self.root, self.runtime): path.mkdir()
        for folder in ('beats', 'structure', 'demix', 'spec', 'logs', 'receipts'):
            (self.original / folder).mkdir()
        for folder in ('executions', 'execution_results', 'recovery_requests', 'raw_replays', 'replay_logs', 'recovered_receipts'):
            (self.root / folder).mkdir()
        (self.original / 'writer.lock').write_bytes(b'')
        (self.root / 'writer.lock').write_bytes(b'')
        checkpoint = self.runtime / 'checkpoints/hub/checkpoints/beat_this-final0.ckpt'
        checkpoint.parent.mkdir(parents=True); checkpoint.write_bytes(b'synthetic checkpoint')
        self.checkpoint = r.b.binding(checkpoint)
        pin = patch.object(r, 'CHECKPOINT_SHA', self.checkpoint['sha256']); pin.start(); self.addCleanup(pin.stop)
        self.rows = []
        for ident in ('a', 'b'):
            source = self.base / (ident + '.wav'); source.write_bytes(('synthetic:' + ident).encode())
            self.rows.append({'id': ident, 'input': r.b.binding(source), 'waveform_float32_sha256': '1' * 64})
        self.shard = {'index': 0, 'rows': 2, 'item_ids': ['a', 'b'], 'item_ids_sha256': r.b.value_hash(['a', 'b'])}
        self.prior = {'output_root': str(self.original), 'runtime_root': str(self.runtime),
                      'aio_gpus': [2], 'beat_gpus': [3], 'shards': [self.shard]}
        run = self.original / 'run_contract.json'; r.b.write_new(run, {'runtime_start': {'frozen': True}, 'source_graph_start': {'frozen': True}})
        log = r.v2.log_path(self.original, 'beats', 0)
        log.write_text('Could not process "' + self.rows[1]['input']['path'] + '". Rerun with this file alone for details.\n')
        (self.original / 'beats/a.beats').write_text('0.5\t1\n1.0\t2\n')
        self.freeze = {'original_run_contract': r.b.binding(run), 'output_root': str(self.root),
                       '_path': str(self.base / 'frozen.json'), '_sha256': 'f' * 64,
                       'driver': r.b.binding(Path(r.__file__).resolve()),
                       'tests': r.b.binding(Path(__file__).resolve()),
                       'v2_runner': r.b.binding(Path(r.v2.__file__).resolve()),
                       'initial_partial_inventory': {str(p): r.b.binding(p) for p in r.files(self.original).values()}}
        self.inputs = {row['id']: self.inspect(row['input']['path'], row, input_audio=True) for row in self.rows}
        p = patch.object(r.b, 'inspect_audio', side_effect=self.inspect); p.start(); self.addCleanup(p.stop)
        p = patch.object(r.b, 'verify_products', side_effect=self.products); p.start(); self.addCleanup(p.stop)

    def inspect(self, path, expected=None, **kwargs):
        binding = r.b.binding(path)
        if expected is not None: r.require(binding == expected['input'], 'synthetic input changed')
        return {**binding, 'format': 'WAV', 'subtype': 'FLOAT', 'sample_rate_hz': 44100, 'channels': 2,
                'frames': 1323000, 'finite': True, 'waveform_float32_sha256': '1' * 64}

    def products(self, stage, rows, output):
        if stage == 'beats':
            return {str(output / 'beats' / (row['id'] + '.beats')): r.b.inspect_beats(output / 'beats' / (row['id'] + '.beats')) for row in rows}
        return {str(p): {**r.b.binding(p), 'synthetic_semantics': 'verified'} for row in rows for p in r.b.product_paths(stage, row['id'], output)}

    def make_request_and_raw(self):
        log = r.v2.log_path(self.original, 'beats', 0)
        missing, existing = r.failed_partition(self.shard, self.rows, self.original, log)
        request = {'status': 'frozen_missing_beat_recovery_transaction', 'policy': r.POLICY,
                   'original_run_contract_sha256': self.freeze['original_run_contract']['sha256'],
                   'shard_index': 0, 'item_ids': ['a', 'b'], 'missing_input_paths': missing, 'missing_ids': ['b'],
                   'original_log': r.b.binding(log), 'original_attempt_kind': 'original_v2_partial', 'execution': None,
                   'preserved_outputs': existing, 'inputs': self.inputs,
                   'original_shard_command': r.b.command('beats', [row['input']['path'] for row in self.rows], self.runtime, self.original)}
        path = self.root / 'recovery_requests/beats_000.json'; r.put(path, request)
        raw = {'status': 'reproduced_exact_empty_beat_serializer_failure', 'request': r.b.binding(path), 'policy': r.POLICY,
               'gpu': 3, 'gpu_name': 'NVIDIA GeForce RTX 5090', 'cuda_device': 'cuda:0',
               'child_environment': r.v2.child_environment(self.runtime, 3), 'checkpoint': self.checkpoint,
               'driver_sha256': self.freeze['driver']['sha256'],
               'runtime_sha256': r.b.value_hash({'frozen': True}),
               'recovery_api': 'File2Beats(checkpoint,device=cuda:0,float16=True,dbn=False)(input_path)',
               'recovery_invocation': r.replay_command(self.freeze, self.prior, path),
               'records': [{'id': 'b', 'input': self.inputs['b'], 'beats': r.array_record(np.array([], dtype=np.float64)),
                            'downbeats': r.array_record(np.array([0.0, 29.9], dtype=np.float64)),
                            'exception_type': 'ValueError', 'exception_message': r.ERROR}], **r.SCOPE}
        raw_path = self.root / 'raw_replays/beats_000.json'; r.put(raw_path, raw)
        (self.root / 'replay_logs/beats_000.log').write_text('synthetic exact predicate replay\n')
        return path, request, raw_path, raw

    def add_ordinary_aio(self):
        for row in self.rows:
            for path in r.b.product_paths('allinone', row['id'], self.original):
                path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b'synthetic product')
        log = r.v2.log_path(self.original, 'allinone', 0); log.write_text('synthetic ordinary success\n')
        payload = {'status': 'passed', 'stage': 'allinone', 'shard_index': 0, 'item_ids': ['a', 'b'],
                   'run_contract_sha256': self.freeze['original_run_contract']['sha256'], 'gpu': 2,
                   'command': r.b.command('allinone', [row['input']['path'] for row in self.rows], self.runtime, self.original),
                   'child_environment': r.v2.child_environment(self.runtime, 2), 'started_utc': 's', 'completed_utc': 'e',
                   'inputs': self.inputs, 'outputs': self.products('allinone', self.rows, self.original), 'log': r.b.binding(log),
                   'ordinary_empty_beat_allowed_only_after_this_successful_command': False, **r.v2.SUCCESS_SCOPE}
        r.put(r.v2.receipt_path(self.original, 'allinone', 0), payload)
        self.freeze['initial_partial_inventory'] = {str(p): r.b.binding(p) for p in r.files(self.original).values()}

    def test_exact_predicate_and_30_second_range(self):
        r.validate_arrays(np.array([]), np.array([0., 30.]))
        for beats, downbeats in (([1.], [1.]), ([], []), ([[]], [0.]), ([], [-0.1]), ([], [30.001]),
                                ([], [0., 0.]), ([], [1., 0.]), ([], [np.nan]), ([], [np.inf])):
            with self.subTest(beats=beats, downbeats=downbeats), self.assertRaises(ValueError):
                r.validate_arrays(np.asarray(beats), np.asarray(downbeats))

    def test_raw_array_roundtrip_and_mutation(self):
        x = np.array([0., 1.25, 30.], dtype=np.float64); record = r.array_record(x)
        np.testing.assert_array_equal(r.read_array(record), x)
        record['values'][1] = 2.0
        with self.assertRaisesRegex(ValueError, 'representation/hash'): r.read_array(record)

    def test_missing_exactly_equals_unique_logged_failure_paths(self):
        log = r.v2.log_path(self.original, 'beats', 0)
        missing, existing = r.failed_partition(self.shard, self.rows, self.original, log)
        self.assertEqual(missing, [self.rows[1]['input']['path']]); self.assertEqual(len(existing), 1)
        initial = log.read_text()
        for changed in (initial + initial, '', initial.replace(self.rows[1]['input']['path'], '/elsewhere.wav')):
            log.write_text(changed)
            with self.assertRaises(ValueError): r.failed_partition(self.shard, self.rows, self.original, log)

    def test_initial_snapshot_mutation_and_unlisted_first_file(self):
        r.validate_initial_snapshot(self.freeze, self.original, audio=True, first=True)
        extra = self.original / 'newfile'; extra.write_bytes(b'new')
        with self.assertRaisesRegex(ValueError, 'first continuation inventory'):
            r.validate_initial_snapshot(self.freeze, self.original, audio=True, first=True)
        extra.unlink(); (self.original / 'beats/a.beats').write_text('0.6\t1\n1.0\t2\n')
        with self.assertRaises(ValueError): r.validate_initial_snapshot(self.freeze, self.original, audio=True)

    def test_raw_replay_rejects_parameter_exception_identity_and_gpu_mutations(self):
        path, request, _, raw = self.make_request_and_raw()
        r.validate_raw(raw, request, r.b.binding(path), self.freeze, self.prior)
        for mutation in (lambda x: x.update(gpu=7), lambda x: x.update(cuda_device='cuda:3'),
                         lambda x: x.update(runtime_sha256='0' * 64), lambda x: x.update(gpu_name='CPU'),
                         lambda x: x.update(feature_extraction_authorized=True),
                         lambda x: x['records'][0].update(exception_message='different failure'),
                         lambda x: x['records'][0].update(id='a'),
                         lambda x: x['records'][0].update(beats=r.array_record(np.array([1.])))):
            changed = copy.deepcopy(raw); mutation(changed)
            with self.assertRaises(ValueError): r.validate_raw(changed, request, r.b.binding(path), self.freeze, self.prior)

    def test_recovery_reuses_verified_raw_transaction_and_preserves_originals(self):
        self.make_request_and_raw()
        initial = {p: r.b.digest(p) for p in self.freeze['initial_partial_inventory']}
        with patch.object(r.subprocess, 'run', side_effect=AssertionError('no model execution in synthetic test')):
            evidence = r.recover_shard(self.freeze, self.prior, self.shard, self.rows, self.root)
            again = r.recover_shard(self.freeze, self.prior, self.shard, self.rows, self.root)
        self.assertEqual(evidence, again)
        self.assertEqual(initial, {p: r.b.digest(p) for p in initial})
        self.assertEqual((self.original / 'beats/b.beats').stat().st_size, 0)
        self.assertFalse(r.v2.receipt_path(self.original, 'beats', 0).exists())
        payload = r.get(self.root / 'recovered_receipts/beats_000.json')
        self.assertTrue(payload['recovery_applied']); self.assertFalse(payload['ordinary_success'])

    def test_interrupt_after_empty_publication_resumes_same_transaction(self):
        self.make_request_and_raw(); (self.original / 'beats/b.beats').write_bytes(b'')
        with patch.object(r.subprocess, 'run', side_effect=AssertionError('must not rerun model')):
            r.recover_shard(self.freeze, self.prior, self.shard, self.rows, self.root)
        self.assertTrue((self.root / 'recovered_receipts/beats_000.json').exists())

    def test_nonempty_recovery_target_is_never_overwritten(self):
        self.make_request_and_raw(); target = self.original / 'beats/b.beats'; target.write_text('1.0\t1\n')
        with self.assertRaises(ValueError): r.recover_shard(self.freeze, self.prior, self.shard, self.rows, self.root)
        self.assertEqual(target.read_text(), '1.0\t1\n')

    def test_original_log_or_preserved_product_mutation_rejected(self):
        _, plan, _, _ = self.make_request_and_raw()
        (self.original / 'beats/a.beats').write_text('0.7\t1\n1.0\t2\n')
        with self.assertRaisesRegex(ValueError, 'preexisting beat product'):
            r.validate_plan(plan, self.freeze, self.prior, self.shard, self.rows, self.root)

    def test_recovered_receipt_cannot_be_admitted_as_ordinary_v2(self):
        self.make_request_and_raw(); r.recover_shard(self.freeze, self.prior, self.shard, self.rows, self.root)
        path = self.root / 'recovered_receipts/beats_000.json'
        with self.assertRaises(ValueError):
            r.v2.validate_receipt(path, self.freeze['original_run_contract']['sha256'], 'beats', self.shard,
                                self.rows, self.prior, self.original)

    def test_combined_audit_preserves_ordinary_and_distinct_recovered_provenance(self):
        self.add_ordinary_aio(); ordinary = r.v2.receipt_path(self.original, 'allinone', 0)
        ordinary_sha = r.b.digest(ordinary)
        self.make_request_and_raw(); r.recover_shard(self.freeze, self.prior, self.shard, self.rows, self.root)
        r.put(self.root / 'driver_run.json', r.driver_record(self.freeze, {'frozen': True}))
        evidence, provenance, recovery, executions = r.audit_all(self.freeze, self.prior, {'rows': self.rows}, self.root)
        self.assertEqual(len(evidence['stage_receipts']), 2); self.assertEqual(len(evidence['products']), 14)
        self.assertEqual(set(provenance), {'a', 'b'}); self.assertEqual(len(recovery), 1); self.assertFalse(executions)
        self.assertEqual(provenance['a']['item_status'], 'preserved_cli_success')
        self.assertEqual(provenance['b']['kind'], 'recovered_serializer_v1')
        self.assertEqual(provenance['b']['item_status'], 'recovered_empty_unavailable')
        self.assertEqual(r.b.digest(ordinary), ordinary_sha)
        self.assertFalse((self.original / 'completion.json').exists())
        self.assertEqual(evidence['products_sha256'], r.b.value_hash(evidence['products']))
        (self.root / 'unbound.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'exhaustive file inventory'):
            r.audit_all(self.freeze, self.prior, {'rows': self.rows}, self.root)

    def test_new_ordinary_execution_has_unchanged_v2_receipt_and_separate_driver_evidence(self):
        def fake_command(command, env, stdout, stderr):
            self.assertEqual(command, r.b.command('allinone', [x['input']['path'] for x in self.rows], self.runtime, self.original))
            self.assertEqual(env, r.v2.child_environment(self.runtime, 2))
            for row in self.rows:
                for path in r.b.product_paths('allinone', row['id'], self.original):
                    path.parent.mkdir(exist_ok=True, parents=True); path.write_bytes(b'synthetic ordinary product')
            stdout.write('synthetic successful command\n')
            return SimpleNamespace(returncode=0)
        with patch.object(r.subprocess, 'run', side_effect=fake_command), patch.object(r.b, 'require_idle_5090') as guard:
            r.ordinary_shard(self.freeze, self.prior, self.shard, self.rows, 'allinone', self.root)
        guard.assert_called_once_with(2)
        ev = r.validate_execution(self.root, 'allinone', self.shard, self.rows, self.prior,
                                  self.freeze['original_run_contract']['sha256'], self.freeze)
        self.assertEqual(set(ev), {'intent', 'result'})
        self.make_request_and_raw(); r.recover_shard(self.freeze, self.prior, self.shard, self.rows, self.root)
        r.put(self.root / 'driver_run.json', r.driver_record(self.freeze, {'frozen': True}))
        _, _, _, executions = r.audit_all(self.freeze, self.prior, {'rows': self.rows}, self.root)
        self.assertEqual(len(executions), 1)
        result_path = Path(ev['result']['path']); payload = r.get(result_path)
        payload['returncode'] = 1; result_path.unlink(); r.put(result_path, payload)
        with self.assertRaisesRegex(ValueError, 'subprocess result/log'):
            r.audit_all(self.freeze, self.prior, {'rows': self.rows}, self.root)

    def test_derivative_completion_reconstructed_not_trusted_and_end_bound(self):
        self.add_ordinary_aio()
        parent = self.base / 'parent.json'; r.b.write_new(parent, {'synthetic': 'original parent authority'})
        self.freeze['original_parent_freeze'] = r.b.binding(parent)
        self.prior['cohort_contract'] = r.b.binding(parent)
        frozen_path = Path(self.freeze['_path'])
        r.b.write_new(frozen_path, {k: v for k, v in self.freeze.items() if not k.startswith('_')})
        self.freeze['_sha256'] = r.b.digest(frozen_path)
        self.make_request_and_raw(); r.recover_shard(self.freeze, self.prior, self.shard, self.rows, self.root)
        runtime = {'frozen': True}; cohort = {'rows': self.rows}
        r.put(self.root / 'driver_run.json', r.driver_record(self.freeze, runtime))
        context = (self.freeze, self.prior, cohort, None, runtime)
        with patch.object(r, 'setup_context', return_value=context), patch.object(r.v2, 'verify_runtime', return_value=runtime), \
             patch.object(r.v2, 'rehash_source_graph', return_value={'frozen': True}):
            expected = r.final_records(*context[:4], runtime)
            audit_path = self.root / 'final_audit.json'; r.b.write_new(audit_path, expected)
            completion_path = self.root / 'completion.json'
            r.b.write_new(completion_path, {**expected, 'final_audit': r.b.binding(audit_path)})
            proof = r.verify_completion(frozen_path, self.freeze['_sha256'])
            self.assertEqual(proof['rows'], 2)
            self.assertEqual(proof['completion'], r.b.binding(completion_path))
            self.assertEqual(proof['beat_provenance']['b']['item_status'], 'recovered_empty_unavailable')
            self.assertEqual(proof['evidence_sha256'], r.b.value_hash({k: expected[k] for key in r.EVIDENCE_KEYS for k in (key, key + '_sha256')}))
            original_audit = audit_path.read_bytes(); audit_path.write_bytes(original_audit + b' ')
            with self.assertRaisesRegex(ValueError, 'encoding/hash'): r.verify_completion(frozen_path, self.freeze['_sha256'])
            audit_path.write_bytes(original_audit)
            altered = copy.deepcopy(expected); altered['recovered_ids'] = []
            audit_path.unlink(); r.b.write_new(audit_path, altered)
            completion_path.unlink(); r.b.write_new(completion_path, {**altered, 'final_audit': r.b.binding(audit_path)})
            with self.assertRaisesRegex(ValueError, 'JSON value'): r.verify_completion(frozen_path, self.freeze['_sha256'])
            parent.write_text('mutated original authority')
            with self.assertRaises(ValueError): r.final_records(*context[:4], runtime)

    def test_interrupted_replay_log_without_raw_evidence_fails_closed(self):
        (self.root / 'replay_logs/beats_000.log').write_text('interrupted replay retained')
        with patch.object(r.subprocess, 'run', side_effect=AssertionError('must not replay without review')):
            with self.assertRaisesRegex(ValueError, 'interrupted/unreceipted replay'):
                r.recover_shard(self.freeze, self.prior, self.shard, self.rows, self.root)
        self.assertFalse((self.original / 'beats/b.beats').exists())


if __name__ == '__main__':
    unittest.main()
