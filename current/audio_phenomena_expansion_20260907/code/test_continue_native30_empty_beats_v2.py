"""Synthetic authority-epoch and exact CUDA environment transition regressions."""
import copy
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import continue_native30_empty_beats_v2 as r
r.b.require_hash(Path(__file__).resolve().with_name('test_continue_native30_empty_beats_v1.py'),
                 '088ef96d6c252d5adc4d7b79924bfb394b94275576357581cc7fbd1b3e691a32')
import test_continue_native30_empty_beats_v1 as fixtures


class EpochTests(unittest.TestCase):
    def setUp(self):
        self.f = fixtures.RecoveryTests(); self.f.setUp(); self.addCleanup(self.f.doCleanups)
        self.f.add_ordinary_aio()
        self.request_path, self.plan, self.raw_path, raw = self.f.make_request_and_raw()
        raw['child_environment']['CUDA_MODULE_LOADING'] = 'LAZY'
        self.raw_path.unlink(); r.put(self.raw_path, raw)
        self.raw = raw
        self.old = copy.deepcopy(self.f.freeze)
        parent = self.f.base / 'parent.json'; r.b.write_new(parent, {'synthetic': 'parent'})
        self.old['original_parent_freeze'] = r.b.binding(parent)
        self.f.prior['cohort_contract'] = r.b.binding(parent)
        old_file = self.f.base / 'old_freeze.json'; r.b.write_new(old_file, self.old)
        self.freeze = {**copy.deepcopy(self.old), 'driver': r.b.binding(Path(r.__file__).resolve()),
                       'tests': r.b.binding(Path(__file__).resolve()), '_previous': self.old,
                       'previous_continuation_freeze': r.b.binding(old_file),
                       '_path': str(self.f.base / 'v2_freeze.json'), '_sha256': '2' * 64}
        r.put(self.f.root / 'driver_run.json', r.load_v1().driver_record(self.old, {'frozen': True}))
        source = self.f.base / 'torch_cuda.py'; source.write_text('synthetic pinned CUDA source')
        self.freeze['torch_cuda_source'] = r.b.binding(source)
        for key, value in (('TORCH_CUDA_SHA', self.freeze['torch_cuda_source']['sha256']), ('CHECKPOINT_SHA', self.f.checkpoint['sha256'])):
            p = patch.object(r, key, value); p.start(); self.addCleanup(p.stop)
        self.freeze['resume_original_inventory'] = {p: r.b.binding(Path(p)) for p in r.files(self.f.original)}
        self.freeze['resume_continuation_inventory'] = {p: r.b.binding(Path(p)) for p in r.files(self.f.root)}

    def test_snapshot_bound_v1_raw_passes_only_exact_transition(self):
        records = r.validate_raw(self.raw, self.plan, r.b.binding(self.request_path), self.freeze, self.f.prior)
        self.assertEqual([x['id'] for x in records], ['b'])
        proof = r.recovery_validation(self.raw, self.plan, self.freeze)
        self.assertEqual(proof['raw_authority_epoch'], 'v1_snapshot_bound')
        self.assertIn('inferred', proof['environment_transition']['v1_before_observation'])

    def test_arbitrary_environment_key_value_and_missing_lazy_rejected(self):
        expected = r.v2.child_environment(self.f.runtime, 3)
        for changed in ({**expected}, {**expected, 'CUDA_MODULE_LOADING': 'EAGER'},
                        {**expected, 'CUDA_MODULE_LOADING': 'LAZY', 'LD_LIBRARY_PATH': '/tmp'},
                        {**expected, 'CUDA_MODULE_LOADING': 'LAZY', 'CUDA_VISIBLE_DEVICES': '2'}):
            raw = {**self.raw, 'child_environment': changed}
            with self.assertRaisesRegex(ValueError, 'environment transition'):
                r.validate_environment(raw, self.f.prior, 3, 'v1_snapshot_bound')

    def test_wrong_torch_source_rejected_even_if_new_binding_is_resealed(self):
        path = Path(self.freeze['torch_cuda_source']['path']); path.write_text('changed source')
        self.freeze['torch_cuda_source'] = r.b.binding(path)
        with self.assertRaisesRegex(ValueError, 'Torch CUDA source pin'):
            r.validate_raw(self.raw, self.plan, r.b.binding(self.request_path), self.freeze, self.f.prior)

    def test_resealed_historical_raw_is_rejected(self):
        changed = copy.deepcopy(self.raw); changed['records'][0]['downbeats'] = r.array_record(np.array([1.0]))
        self.raw_path.unlink(); r.put(self.raw_path, changed)
        with self.assertRaisesRegex(ValueError, 'historical raw changed'):
            r.validate_raw(changed, self.plan, r.b.binding(self.request_path), self.freeze, self.f.prior)

    def test_v1_raw_cannot_be_relabelled_as_recorded_v2(self):
        changed = copy.deepcopy(self.raw)
        changed['child_environment_before_cuda_init'] = r.v2.child_environment(self.f.runtime, 3)
        with self.assertRaisesRegex(ValueError, 'explicitly inferred'):
            r.validate_environment(changed, self.f.prior, 3, 'v1_snapshot_bound')

    def test_recovery_repeated_resume_preserves_all_v1_files(self):
        historical = {p: r.b.digest(p) for p in self.freeze['resume_continuation_inventory']}
        original = {p: r.b.digest(p) for p in self.freeze['resume_original_inventory']}
        with patch.object(r.subprocess, 'run', side_effect=AssertionError('must not replay old successful raw')):
            first = r.recover_shard(self.freeze, self.f.prior, self.f.shard, self.f.rows, self.f.root)
            second = r.recover_shard(self.freeze, self.f.prior, self.f.shard, self.f.rows, self.f.root)
        self.assertEqual(first, second)
        self.assertEqual(historical, {p: r.b.digest(p) for p in historical})
        self.assertEqual(original, {p: r.b.digest(p) for p in original})
        payload = r.get(self.f.root / 'recovered_receipts/beats_000.json')
        self.assertEqual(payload['recovery_validation']['raw_authority_epoch'], 'v1_snapshot_bound')
        self.assertEqual(payload['raw_replay'], r.b.binding(self.raw_path))
        self.assertEqual(payload['recovery_invocation'][1], self.old['driver']['path'])
        self.assertFalse(r.v2.receipt_path(self.f.original, 'beats', 0).exists())

    def test_combined_audit_preserves_both_driver_epoch_markers(self):
        r.put(self.f.root / 'driver_run_v2.json', r.driver_record(self.freeze, {'frozen': True}))
        r.recover_shard(self.freeze, self.f.prior, self.f.shard, self.f.rows, self.f.root)
        evidence, provenance, _, _ = r.audit_all(self.freeze, self.f.prior, {'rows': self.f.rows}, self.f.root)
        self.assertEqual(len(evidence['products']), 14)
        self.assertEqual(provenance['b']['item_status'], 'recovered_empty_unavailable')
        r.validate_epoch_snapshots(self.freeze, self.f.prior, audio=True)
        with self.assertRaisesRegex(ValueError, 'first v2 epoch'):
            r.validate_epoch_snapshots(self.freeze, self.f.prior, audio=True, first=True)

    def test_full_derivative_completion_reconstructs_both_epochs_and_rejects_mutation(self):
        frozen = Path(self.freeze['_path']); r.b.write_new(frozen, {k: v for k, v in self.freeze.items() if not k.startswith('_')})
        self.freeze['_sha256'] = r.b.digest(frozen)
        runtime = {'frozen': True}; cohort = {'rows': self.f.rows}
        r.put(self.f.root / 'driver_run_v2.json', r.driver_record(self.freeze, runtime))
        r.recover_shard(self.freeze, self.f.prior, self.f.shard, self.f.rows, self.f.root)
        context = (self.freeze, self.f.prior, cohort, None, runtime)
        with patch.object(r, 'setup_context', return_value=context), patch.object(r.v2, 'verify_runtime', return_value=runtime), \
             patch.object(r.v2, 'rehash_source_graph', return_value={'frozen': True}):
            expected = r.final_records(*context)
            audit = self.f.root / 'final_audit.json'; completion = self.f.root / 'completion.json'
            r.b.write_new(audit, expected); r.b.write_new(completion, {**expected, 'final_audit': r.b.binding(audit)})
            proof = r.verify_completion(frozen, self.freeze['_sha256'])
            self.assertEqual(proof['rows'], 2)
            self.assertEqual(expected['driver_execution']['path'], str(self.f.root / 'driver_run_v2.json'))
            self.assertEqual(expected['historical_driver_execution']['path'], str(self.f.root / 'driver_run.json'))
            altered = {**expected, 'historical_driver_execution': expected['driver_execution']}
            audit.unlink(); r.b.write_new(audit, altered)
            completion.unlink(); r.b.write_new(completion, {**altered, 'final_audit': r.b.binding(audit)})
            with self.assertRaisesRegex(ValueError, 'JSON value'): r.verify_completion(frozen, self.freeze['_sha256'])

    def test_new_v2_raw_requires_explicit_both_environment_snapshots(self):
        expected = r.v2.child_environment(self.f.runtime, 3)
        valid = {**self.raw, 'child_environment_before_cuda_init': expected, 'environment_transition': r.ENVIRONMENT_TRANSITION}
        r.validate_environment(valid, self.f.prior, 3, 'v2_recorded')
        for changed in ({**valid, 'child_environment_before_cuda_init': None},
                        {**valid, 'child_environment_before_cuda_init': {**expected, 'CUDA_MODULE_LOADING': 'LAZY'}},
                        {**valid, 'environment_transition': {}}):
            with self.assertRaises(ValueError): r.validate_environment(changed, self.f.prior, 3, 'v2_recorded')

    def test_new_worker_records_actual_before_and_after_without_neural_model(self):
        self.raw_path.unlink(); self.freeze['resume_continuation_inventory'].pop(str(self.raw_path))
        expected = r.v2.child_environment(self.f.runtime, 3)
        def available():
            os.environ['CUDA_MODULE_LOADING'] = 'LAZY'; return True
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=available, get_device_name=lambda _: 'NVIDIA GeForce RTX 5090'))
        fake_inference = SimpleNamespace(File2Beats=lambda *a, **k: lambda _: (np.array([]), np.array([0.0])))
        def infer(*args): raise ValueError(r.ERROR)
        with patch.dict(os.environ, expected, clear=True), patch.dict(sys.modules, {'torch': fake_torch,
                'beat_this.inference': fake_inference, 'beat_this.utils': SimpleNamespace(infer_beat_numbers=infer)}), \
                patch.object(r.b, 'require_idle_5090'):
            r.replay_worker(self.freeze, self.f.prior, {'rows': self.f.rows}, self.request_path, r.b.digest(self.request_path))
        raw = r.get(self.raw_path)
        self.assertEqual(raw['child_environment_before_cuda_init'], expected)
        self.assertEqual(raw['child_environment'], {**expected, 'CUDA_MODULE_LOADING': 'LAZY'})
        self.assertEqual(raw['driver_sha256'], self.freeze['driver']['sha256'])
        r.validate_raw(raw, self.plan, r.b.binding(self.request_path), self.freeze, self.f.prior)

    def test_historical_execution_misattribution_to_v2_rejected(self):
        path = self.f.root / 'executions/beats_000.json'; result_path = self.f.root / 'execution_results/beats_000.json'
        intent = {'status': 'continuation_stage_execution_intent', 'stage': 'beats', 'shard_index': 0,
                  'item_ids': ['a', 'b'], 'original_run_contract_sha256': self.old['original_run_contract']['sha256'], 'gpu': 3,
                  'driver': self.old['driver'], 'continuation_freeze_sha256': self.old['_sha256'],
                  'command': r.b.command('beats', [x['input']['path'] for x in self.f.rows], self.f.runtime, self.f.original),
                  'child_environment': r.v2.child_environment(self.f.runtime, 3), 'inputs': self.f.inputs}
        r.put(path, intent)
        result = {'status': 'continuation_stage_subprocess_exited', 'intent': r.b.binding(path), 'returncode': 0,
                  'log': r.b.binding(r.v2.log_path(self.f.original, 'beats', 0)), 'inputs_after': self.f.inputs}
        r.put(result_path, result)
        self.freeze['resume_continuation_inventory'].update({str(p): r.b.binding(p) for p in (path, result_path)})
        r.validate_execution(self.f.root, 'beats', self.f.shard, self.f.rows, self.f.prior,
                             self.old['original_run_contract']['sha256'], self.freeze)
        intent['driver'] = self.freeze['driver']; path.unlink(); r.put(path, intent)
        result['intent'] = r.b.binding(path); result_path.unlink(); r.put(result_path, result)
        self.freeze['resume_continuation_inventory'].update({str(p): r.b.binding(p) for p in (path, result_path)})
        with self.assertRaisesRegex(ValueError, 'execution intent mismatch'):
            r.validate_execution(self.f.root, 'beats', self.f.shard, self.f.rows, self.f.prior,
                                 self.old['original_run_contract']['sha256'], self.freeze)


if __name__ == '__main__': unittest.main()
