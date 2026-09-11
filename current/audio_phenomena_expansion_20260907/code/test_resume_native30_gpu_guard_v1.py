"""Synthetic-only dispatch, immutable history, exact incident and final graph tests."""
import copy
import fcntl
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import resume_native30_gpu_guard_v1 as s
import test_continue_native30_empty_beats_v1 as fixtures
import test_continue_native30_empty_beats_v2 as epoch_fixtures


class OperationalTests(unittest.TestCase):
    def setUp(self):
        self.x = fixtures.RecoveryTests(); self.x.setUp(); self.addCleanup(self.x.doCleanups)
        self.original, self.oldroot = self.x.original, self.x.root
        (self.original / 'logs/beats_000.log').unlink(); (self.original / 'beats/a.beats').unlink()
        self.root = self.x.base / 'operational'; self.root.mkdir(); (self.root / 'writer.lock').write_bytes(b'')
        for folder in ('dispatch', 'executions', 'execution_results', 'recovery_requests', 'raw_replays', 'replay_logs', 'recovered_receipts'):
            (self.root / folder).mkdir()
        self.shard = {**self.x.shard, 'index': 96}
        self.prior = {**self.x.prior, 'aio_gpus': [0, 1, 2, 3], 'beat_gpus': [4, 5], 'shards': [self.shard]}
        self.effective = s.effective_prior(self.prior, s.PLACEMENT)
        self.base = {**self.x.freeze, 'initial_partial_inventory': {p: s.b.binding(p) for p in s.r.files(self.original)},
                     'resume_continuation_inventory': {}}
        parent = self.x.base / 'base_parent.json'; s.b.write_new(parent, {'synthetic': True})
        self.f = {**self.base, 'output_root': str(self.root), '_base': self.base, 'placement': s.PLACEMENT,
                  '_path': str(self.x.base / 'operational_freeze.json'), '_sha256': '9' * 64,
                  'driver': s.b.binding(Path(s.__file__).resolve()), 'tests': s.b.binding(Path(__file__).resolve()),
                  'base_driver': s.b.binding(Path(s.r.__file__).resolve()), 'base_freeze': s.b.binding(parent),
                  'stopped_original_inventory': {p: s.b.binding(p) for p in s.r.files(self.original)},
                  'stopped_recovery_inventory': {p: s.b.binding(p) for p in s.r.files(self.oldroot)}}
        self.certificate = {'synthetic': 'blocked_096_not_launched'}
        s.r.put(self.root / 'blocked_attempt.json', self.certificate)

    def fake_command(self, command, **kwargs):
        stage = 'beats' if 'beat_this' in ' '.join(command) else 'allinone'
        kwargs['stdout'].write('synthetic child succeeded\n')
        for row in self.x.rows:
            for path in s.b.product_paths(stage, row['id'], self.original):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'0.5\t1\n' if stage == 'beats' else b'synthetic')
        return SimpleNamespace(returncode=0)

    def launch(self, stage):
        with patch.object(s.socket, 'gethostname', return_value=s.HOST), patch.object(s.b, 'require_idle_5090'), \
             patch.object(s.r.subprocess, 'run', side_effect=self.fake_command):
            s.launch_stage(self.f, self.prior, self.effective, stage, self.shard, self.x.rows)

    def test_only_explicit_remap_and_disjoint_workers(self):
        self.assertEqual(self.effective['beat_gpus'], [6, 5]); self.assertEqual(self.prior['beat_gpus'], [4, 5])
        self.assertEqual({k: v for k, v in self.prior.items() if not k.endswith('_gpus')},
                         {k: v for k, v in self.effective.items() if not k.endswith('_gpus')})
        for changed in ({**s.PLACEMENT, 'host': 'elsewhere'}, {**s.PLACEMENT, 'beats_physical': [5, 5]},
                        {**s.PLACEMENT, 'allinone_physical': [1, 0, 2, 3]}):
            with self.assertRaises(ValueError): s.effective_prior(self.prior, changed)

    def test_idle_guard_fails_before_any_dispatch_or_intent(self):
        before = set(s.r.files(self.root))
        with patch.object(s.socket, 'gethostname', return_value=s.HOST), patch.object(s.b, 'require_idle_5090', side_effect=ValueError('busy')), \
             patch.object(s.r, 'ordinary_shard') as child:
            with self.assertRaisesRegex(ValueError, 'busy'): s.launch_stage(self.f, self.prior, self.effective, 'beats', self.shard, self.x.rows)
        self.assertEqual(before, set(s.r.files(self.root))); child.assert_not_called()

    def test_wrong_host_rejected_before_child(self):
        with patch.object(s.socket, 'gethostname', return_value='another-host'), patch.object(s.r, 'ordinary_shard') as child:
            with self.assertRaisesRegex(ValueError, 'host'): s.launch_stage(self.f, self.prior, self.effective, 'beats', self.shard, self.x.rows)
        child.assert_not_called()

    def test_new_attempt_records_actual_gpu6_and_separate_old_intent(self):
        old = self.oldroot / 'executions/beats_096.json'; s.r.put(old, {'status': 'synthetic_prelaunch_intent'})
        old_hash = s.b.digest(old); self.launch('beats')
        ev, kind, epoch, execution, dispatch = s.stage_evidence(self.f, self.prior, self.effective, 'beats', self.shard, self.x.rows)
        self.assertEqual(epoch, 'operational_dispatch_v1'); self.assertEqual(s.b.digest(old), old_hash)
        self.assertFalse((self.oldroot / 'execution_results/beats_096.json').exists())
        intent = s.r.get(execution['intent']['path']); self.assertEqual(intent['gpu'], 6)
        self.assertEqual(intent['child_environment']['CUDA_VISIBLE_DEVICES'], '6')
        self.assertEqual(s.r.get(dispatch['path'])['logical_gpu'], 4)
        self.assertEqual(s.r.get(dispatch['path'])['blocked_attempt'], s.b.binding(self.root / 'blocked_attempt.json'))
        with self.assertRaises(ValueError):
            s.v2.validate_receipt(ev['receipt']['path'], self.f['original_run_contract']['sha256'], 'beats', self.shard,
                                  self.x.rows, self.prior, self.original)

    def test_new_receipt_without_dispatch_or_execution_rejected(self):
        self.launch('beats'); (self.root / 'dispatch/beats_096.json').unlink()
        with self.assertRaises((ValueError, FileNotFoundError)):
            s.stage_evidence(self.f, self.prior, self.effective, 'beats', self.shard, self.x.rows)

    def test_resealed_new_receipt_with_old_gpu_environment_rejected(self):
        self.launch('beats'); path = s.v2.receipt_path(self.original, 'beats', 96)
        payload = s.r.get(path); payload.update(gpu=4, child_environment=s.v2.child_environment(self.x.runtime, 4))
        path.unlink(); s.r.put(path, payload)
        with self.assertRaises(ValueError): s.stage_evidence(self.f, self.prior, self.effective, 'beats', self.shard, self.x.rows)

    def test_verification_three_shared_locks_and_writer_contention(self):
        context = (self.f, self.prior, self.effective, {'rows': self.x.rows}, None, {})
        real_flock = fcntl.flock
        with patch.object(s, 'setup_context', return_value=context), patch.object(s, 'verify_locked', return_value={'synthetic': True}) as verify, \
             patch.object(s.fcntl, 'flock', wraps=real_flock) as locks:
            self.assertEqual(s.verify_completion('unused', 'unused'), {'synthetic': True})
            self.assertEqual([c.args[1] for c in locks.call_args_list], [fcntl.LOCK_SH | fcntl.LOCK_NB] * 3)
            verify.assert_called_once_with(context)
        with (self.oldroot / 'writer.lock').open('r+b') as writer:
            real_flock(writer, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with patch.object(s, 'setup_context', return_value=context), patch.object(s, 'verify_locked') as verify:
                with self.assertRaises(BlockingIOError): s.verify_completion('unused', 'unused')
                verify.assert_not_called()

    def test_old_gpu4_receipt_not_relabelled_gpu6(self):
        with patch.object(s.b, 'require_idle_5090'), patch.object(s.r.subprocess, 'run', side_effect=self.fake_command):
            s.r.ordinary_shard(self.base, self.prior, self.shard, self.x.rows, 'beats', self.oldroot)
        receipt = s.v2.receipt_path(self.original, 'beats', 96)
        self.f['stopped_original_inventory'][str(receipt)] = s.b.binding(receipt)
        ev, _, epoch, execution, dispatch = s.stage_evidence(self.f, self.prior, self.effective, 'beats', self.shard, self.x.rows)
        self.assertEqual(epoch, 'preserved_recovery_v2'); self.assertIsNone(dispatch)
        self.assertEqual(s.r.get(ev['receipt']['path'])['gpu'], 4)
        self.assertEqual(s.r.get(execution['intent']['path'])['gpu'], 4)

    def test_snapshots_preserve_old_bytes_and_forbid_new_oldroot_files(self):
        p = self.oldroot / 'old.json'; p.write_text('unchanged')
        self.f['stopped_recovery_inventory'][str(p)] = s.b.binding(p)
        s.snapshots(self.f, self.prior, audio=True)
        p.write_text('different')
        with self.assertRaises(ValueError): s.snapshots(self.f, self.prior, audio=True)
        p.write_text('unchanged'); (self.oldroot / 'unexpected').write_text('x')
        with self.assertRaisesRegex(ValueError, 'file set'): s.snapshots(self.f, self.prior, audio=True)

    def test_repeated_receipt_validation_and_dispatch_put_are_idempotent(self):
        self.launch('beats'); before = {p: s.b.digest(p) for p in s.r.files(self.root)}
        first = s.stage_evidence(self.f, self.prior, self.effective, 'beats', self.shard, self.x.rows)
        second = s.stage_evidence(self.f, self.prior, self.effective, 'beats', self.shard, self.x.rows)
        self.assertEqual(first, second)
        s.immutable_put(self.root / 'dispatch/beats_096.json', s.dispatch_record(self.f, self.prior, self.effective, 'beats', self.shard))
        self.assertEqual(before, {p: s.b.digest(p) for p in s.r.files(self.root)})
        with self.assertRaisesRegex(ValueError, 'conflicting'):
            s.immutable_put(self.root / 'dispatch/beats_096.json', {})

    def test_combined_graph_and_exhaustive_inventory(self):
        self.launch('allinone'); self.launch('beats')
        s.r.put(self.root / 'driver_run.json', {'synthetic': True})
        ev, provenance, executions, recovery = s.audit_all(self.f, self.prior, self.effective, {'rows': self.x.rows})
        self.assertEqual(len(ev['products']), 14); self.assertEqual(len(executions), 2)
        self.assertTrue(all(v['execution_epoch'] == 'operational_dispatch_v1' for v in provenance.values()))
        self.assertEqual(recovery, {})
        (self.root / 'unknown.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'exhaustive'): s.audit_all(self.f, self.prior, self.effective, {'rows': self.x.rows})

    def test_final_records_and_canonical_completion_mutation(self):
        self.launch('allinone'); self.launch('beats'); runtime = {'frozen': True}; cohort = {'rows': self.x.rows}
        parent = self.x.base / 'parent.json'; s.b.write_new(parent, {'synthetic': 'parent'})
        self.base.update(original_parent_freeze=s.b.binding(parent), _previous={})
        self.prior['cohort_contract'] = s.b.binding(parent)
        self.f.update(terminal_log=s.b.binding(parent), blocked_intent=s.b.binding(parent))
        frozen = Path(self.f['_path']); s.b.write_new(frozen, {'synthetic': 'operational authority'})
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
            altered = {**expected, 'placement': {}}; audit.unlink(); completion.unlink()
            s.b.write_new(audit, altered); s.b.write_new(completion, {**altered, 'final_audit': s.b.binding(audit)})
            with self.assertRaisesRegex(ValueError, 'JSON value'): s.verify_completion(frozen, self.f['_sha256'])


class IncidentTests(unittest.TestCase):
    setUp = OperationalTests.setUp
    def test_exact_prelaunch_certificate_and_negative_artifact_tests(self):
        rows = [{**self.x.rows[0], 'id': 'x' + str(i)} for i in range(24)]
        shard = {'index': 96, 'item_ids': [x['id'] for x in rows]}
        prior = {**self.prior, 'shards': [None] * 96 + [shard]}
        self.base['_sha256'] = s.BASE_FREEZE_SHA
        path = self.oldroot / 'executions/beats_096.json'; terminal = self.x.base / 'terminal.log'; terminal.write_text('synthetic pinned guard trace')
        s.r.put(path, {'status': 'continuation_stage_execution_intent', 'stage': 'beats', 'shard_index': 96,
            'item_ids': shard['item_ids'], 'gpu': 4, 'started_utc': '2026-09-08T06:49:42.978609+00:00',
            'driver': self.base['driver'], 'continuation_freeze_sha256': s.BASE_FREEZE_SHA,
            'original_run_contract_sha256': self.base['original_run_contract']['sha256'],
            'command': s.b.command('beats', [x['input']['path'] for x in rows], self.x.runtime, self.original),
            'child_environment': s.v2.child_environment(self.x.runtime, 4)})
        self.f.update(blocked_intent=s.b.binding(path), terminal_log=s.b.binding(terminal))
        self.f['stopped_recovery_inventory'][str(path)] = s.b.binding(path)
        with patch.object(s, 'BLOCKED_SHA', s.b.digest(path)), patch.object(s, 'TERMINAL_SHA', s.b.digest(terminal)):
            cert = s.aborted_attempt(self.f, prior, {'rows': rows}, first=True); self.assertFalse(cert['subprocess_launched'])
            log = self.original / 'logs/beats_096.log'; log.write_text('child may have started')
            with self.assertRaisesRegex(ValueError, 'launch/output'): s.aborted_attempt(self.f, prior, {'rows': rows}, first=True)
            log.unlink(); target = self.original / 'beats/x0.beats'; target.write_text('')
            with self.assertRaises(ValueError): s.aborted_attempt(self.f, prior, {'rows': rows}, first=True)
            target.unlink(); result = self.oldroot / 'execution_results/beats_096.json'; result.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'never gain'): s.aborted_attempt(self.f, prior, {'rows': rows})


class ReplayTests(unittest.TestCase):
    def test_new_worker_actual_invocation_and_exact_gpu_environment(self):
        x = epoch_fixtures.EpochTests(); x.setUp(); self.addCleanup(x.doCleanups)
        x.raw_path.unlink(); x.freeze['resume_continuation_inventory'].pop(str(x.raw_path))
        x.freeze['driver'] = s.b.binding(Path(s.__file__).resolve())
        expected = s.v2.child_environment(x.f.runtime, 3)
        def available(): os.environ['CUDA_MODULE_LOADING'] = 'LAZY'; return True
        cuda = SimpleNamespace(is_available=available, get_device_name=lambda _: 'NVIDIA GeForce RTX 5090')
        calls = []
        def model(*a, **k): calls.append((a, k)); return lambda _: (np.array([]), np.array([0.0]))
        def infer(*a): raise ValueError(s.r.ERROR)
        command = s.r.replay_command(x.freeze, x.f.prior, x.request_path)
        with patch.dict(os.environ, expected, clear=True), patch.dict(sys.modules, {'torch': SimpleNamespace(cuda=cuda),
             'beat_this.inference': SimpleNamespace(File2Beats=model), 'beat_this.utils': SimpleNamespace(infer_beat_numbers=infer)}), \
             patch.object(s.socket, 'gethostname', return_value=s.HOST), patch.object(s.b, 'require_idle_5090'), \
             patch.object(s.sys, 'argv', command[1:]), patch.object(s.sys, 'executable', command[0]):
            s.replay_worker(x.freeze, x.f.prior, {'rows': x.f.rows}, x.request_path, s.b.digest(x.request_path))
        raw = s.r.get(x.raw_path); self.assertEqual(raw['recovery_invocation'], command)
        self.assertEqual(raw['child_environment_before_cuda_init'], expected)
        self.assertEqual(raw['child_environment'], {**expected, 'CUDA_MODULE_LOADING': 'LAZY'})
        self.assertEqual(calls[0][1], {'device': 'cuda:0', 'float16': True, 'dbn': False})
        s.r.validate_raw(raw, x.plan, s.b.binding(x.request_path), x.freeze, x.f.prior)


class MixedRecoveryTests(unittest.TestCase):
    def test_historical_and_new_recovered_graph_duplicate_and_inventory_rejections(self):
        x = epoch_fixtures.EpochTests(); x.setUp(); self.addCleanup(x.doCleanups)
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
        self.assertEqual(provenance['d']['execution_epoch'], 'operational_dispatch_v1')
        self.assertEqual(provenance['c']['item_status'], 'preserved_cli_success')
        self.assertEqual(s.r.get(root / 'recovered_receipts/beats_096.json')['recovery_validation']['validator'], f['driver'])
        historical = {p: s.b.digest(p) for p in s.r.files(oldroot)}
        self.assertEqual(s.audit_all(f, prior, effective, cohort)[0], ev)
        self.assertEqual(historical, {p: s.b.digest(p) for p in s.r.files(oldroot)})
        duplicate = s.v2.receipt_path(original, 'beats', 96); s.r.put(duplicate, {})
        with self.assertRaisesRegex(ValueError, 'duplicate'): s.audit_all(f, prior, effective, cohort)
        duplicate.unlink(); extra = original / 'unexpected'; extra.write_bytes(b'x')
        with self.assertRaisesRegex(ValueError, 'exhaustive'): s.audit_all(f, prior, effective, cohort)
        extra.unlink(); missing = original / 'beats/d.beats'; missing.unlink()
        with self.assertRaises((ValueError, FileNotFoundError)): s.audit_all(f, prior, effective, cohort)


if __name__ == '__main__': unittest.main()
