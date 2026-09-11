import copy
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_native30_inference_batches_v1 as b
import run_native30_inference_batches_v2 as r
import test_run_native30_inference_batches_v1 as v1tests


class V2RunnerTests(unittest.TestCase):
    def build_v2_freeze(self, root):
        entry, expected, helper, plan, screen = v1tests.RunnerTests().build_freeze(root)
        freeze = b.read_json(entry['path'])
        runtime = root / 'runtime'; runtime.mkdir(exist_ok=True)
        freeze.update(version=r.FREEZE_VERSION, recovery_applied=False,
                      runner=b.binding(Path(r.__file__).resolve()),
                      tests=b.binding(Path(__file__).resolve()),
                      v1_dependency=b.binding(Path(b.__file__).resolve()),
                      runtime_root=str(runtime), torch_hub_dir=r.TORCH_HUB,
                      child_environment=r.child_environment_policy(runtime))
        path = root / 'freeze_v2.json'; b.write_new(path, freeze)
        return b.binding(path), expected, helper, plan, screen

    def test_parent_freeze_replays_graph_without_audio(self):
        with tempfile.TemporaryDirectory() as name:
            entry, expected, helper, plan, screen = self.build_v2_freeze(Path(name))
            with patch.object(b, 'sf') as forbidden_audio:
                freeze, cohort, observed_helper = r.validate_freeze(
                    entry['path'], entry['sha256'], expected=expected,
                    helper_loader=lambda record: helper,
                    plan_sha=plan['sha256'], screen_sha=screen['sha256'])
            self.assertEqual(len(cohort['rows']), 1)
            self.assertIs(observed_helper, helper)
            self.assertEqual(freeze['child_environment'], r.child_environment_policy(freeze['runtime_root']))
            forbidden_audio.SoundFile.assert_not_called()

    def test_child_environment_is_exact_and_drops_parent_overrides(self):
        runtime = Path('/runtime')
        with patch.dict(os.environ, {'PYTHONPATH': '/poison', 'HF_HOME': '/wrong',
                                     'LD_LIBRARY_PATH': '/wrong', 'HOME': '/wrong'}, clear=False):
            environment = r.child_environment(runtime, 7)
        self.assertEqual(environment, {**r.CHILD_BASE_ENVIRONMENT,
                                       'PATH': '/runtime/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin',
                                       'CUDA_VISIBLE_DEVICES': '7'})
        self.assertTrue(set(r.CLEARED_CHILD_ENVIRONMENT).isdisjoint(environment))

    @unittest.skipUnless(Path('/mnt/nfs-code/users/yi/dynamics_rhythm_external_benchmark_20260903').is_dir(),
                         'pinned inference runtime is available only on the remote execution host')
    def test_actual_child_runtime_resolution_smoke_no_gpu_or_model(self):
        runtime = Path('/mnt/nfs-code/users/yi/dynamics_rhythm_external_benchmark_20260903')
        observed = r.probe_child_runtime(runtime)
        self.assertEqual(observed['torch_hub_dir'], r.TORCH_HUB)
        self.assertEqual(set(observed['environment']), set(r.CHILD_BASE_ENVIRONMENT) | {'PATH', 'CUDA_VISIBLE_DEVICES'})
        self.assertEqual(observed['launchers']['all-in-one-infer'], str((runtime / 'venv/bin/all-in-one-infer').resolve()))
        self.assertEqual(observed['launchers']['beat_this'], str((runtime / 'venv/bin/beat_this').resolve()))
        self.assertEqual(set(observed['launcher_help']), {'all-in-one-infer', 'beat_this'})
        self.assertTrue(all(record['bytes'] > 0 and len(record['sha256']) == 64
                            for record in observed['launcher_help'].values()))

    def test_changed_final_audit_json_is_rejected_against_original_binding(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / 'final_audit.json'; expected = {'status': 'complete', 'receipts': {'x': '1'}}
            b.write_new(path, expected)
            original = r.bind_canonical_json(path, expected)
            path.write_bytes(b.canonical({'status': 'complete', 'receipts': {'x': '2'}}))
            with self.assertRaisesRegex(ValueError, 'value changed'):
                r.bind_canonical_json(path, expected)
            self.assertNotEqual(b.binding(path), original)

    @staticmethod
    def receipt_fixture(root, stage='beats'):
        output = root / 'output'
        for name in ('beats', 'logs', 'receipts', 'structure', 'spec', 'demix'):
            (output / name).mkdir(parents=True, exist_ok=True)
        runtime = root / 'runtime'; runtime.mkdir(exist_ok=True)
        ident = 'x'
        input_path = root / 'x.wav'; input_path.write_bytes(b'not-opened')
        row = {'id': ident, 'input': b.binding(input_path), 'waveform_float32_sha256': '1' * 64}
        shard = {'index': 0, 'rows': 1, 'item_ids': [ident],
                 'item_ids_sha256': b.value_hash([ident])}
        freeze = {'runtime_root': str(runtime), 'aio_gpus': [2], 'beat_gpus': [3]}
        log = r.log_path(output, stage, 0); log.write_text('successful command\n')
        paths = b.product_paths(stage, ident, output)
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(('product:' + path.name).encode())
        products = {str(path): {**b.binding(path), 'semantic': 'checked'} for path in paths}
        inputs = {ident: {**row['input'], 'format': 'WAV', 'subtype': 'FLOAT',
                          'sample_rate_hz': 44100, 'channels': 2, 'frames': 1323000,
                          'finite': True, 'waveform_float32_sha256': '1' * 64}}
        gpu = r.assigned_gpu(freeze, stage, 0)
        payload = {'status': 'passed', 'stage': stage, 'shard_index': 0, 'item_ids': [ident],
                   'run_contract_sha256': 'a' * 64, 'gpu': gpu,
                   'command': b.command(stage, [str(input_path)], runtime, output),
                   'child_environment': r.child_environment(runtime, gpu),
                   'started_utc': '2026-09-08T00:00:00+00:00',
                   'completed_utc': '2026-09-08T00:00:01+00:00',
                   'inputs': inputs, 'outputs': products, 'log': b.binding(log),
                   'ordinary_empty_beat_allowed_only_after_this_successful_command': stage == 'beats',
                   **r.SUCCESS_SCOPE}
        return output, freeze, row, shard, payload, inputs, products, log

    @staticmethod
    def put_receipt(path, payload):
        b.write_new(path, {'payload': payload, 'receipt_sha256': b.value_hash(payload)})
        return path

    def validate_fixture(self, path, row, shard, freeze, output, inputs, products, stage='beats'):
        with patch.object(b, 'inspect_audio', return_value=inputs['x']), \
             patch.object(b, 'verify_products', return_value=products):
            return r.validate_receipt(path, 'a' * 64, stage, shard, [row], freeze, output)

    def test_complete_receipt_revalidates_semantics_inputs_log_command_gpu_and_environment(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            output, freeze, row, shard, payload, inputs, products, _ = self.receipt_fixture(root)
            path = self.put_receipt(root / 'valid.json', payload)
            evidence = self.validate_fixture(path, row, shard, freeze, output, inputs, products)
            self.assertEqual(evidence['outputs'], products)
            self.assertEqual(evidence['inputs'], inputs)

    def test_empty_and_incomplete_output_maps_are_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            output, freeze, row, shard, payload, inputs, _, _ = self.receipt_fixture(root)
            payload['outputs'] = {}
            path = self.put_receipt(root / 'empty.json', payload)
            with self.assertRaisesRegex(ValueError, 'exact output path set'):
                self.validate_fixture(path, row, shard, freeze, output, inputs, {})

        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            output, freeze, row, shard, payload, inputs, products, _ = self.receipt_fixture(root, 'allinone')
            one = dict([next(iter(products.items()))])
            payload['outputs'] = one
            path = self.put_receipt(root / 'incomplete.json', payload)
            with self.assertRaisesRegex(ValueError, 'exact output path set'):
                self.validate_fixture(path, row, shard, freeze, output, inputs, one, 'allinone')

    def test_changed_log_is_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            output, freeze, row, shard, payload, inputs, products, log = self.receipt_fixture(root)
            path = self.put_receipt(root / 'receipt.json', payload)
            log.write_text('edited after success\n')
            with self.assertRaisesRegex(ValueError, 'log changed'):
                self.validate_fixture(path, row, shard, freeze, output, inputs, products)

    def test_final_revalidation_rejects_changed_earlier_output(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            output, freeze, row, shard, payload, inputs, products, _ = self.receipt_fixture(root)
            path = self.put_receipt(root / 'receipt.json', payload)
            product_path = Path(next(iter(products)))
            product_path.write_bytes(b'edited after shard receipt')
            changed = copy.deepcopy(products)
            changed[str(product_path)].update(b.binding(product_path))
            with self.assertRaisesRegex(ValueError, 'product semantics/hash changed'):
                self.validate_fixture(path, row, shard, freeze, output, inputs, changed)

    def test_audit_all_revalidates_both_stage_receipts_and_changed_earlier_product(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            output, freeze, row, shard, beat_payload, inputs, beat_products, _ = self.receipt_fixture(root, 'beats')
            _, _, _, _, aio_payload, _, aio_products, _ = self.receipt_fixture(root, 'allinone')
            self.put_receipt(r.receipt_path(output, 'beats', 0), beat_payload)
            self.put_receipt(r.receipt_path(output, 'allinone', 0), aio_payload)
            (output / 'writer.lock').write_bytes(b'')
            (output / 'run_contract.json').write_text('{}\n')
            cohort = {'rows': [row]}; freeze['shards'] = [shard]

            def products(stage, rows, out):
                self.assertEqual(rows, [row]); self.assertEqual(out, output)
                return aio_products if stage == 'allinone' else beat_products

            with patch.object(b, 'inspect_audio', return_value=inputs['x']), \
                 patch.object(b, 'verify_products', side_effect=products):
                evidence = r.audit_all('a' * 64, freeze, cohort, output)
            self.assertEqual(len(evidence['stage_receipts']), 2)
            self.assertEqual(len(evidence['logs']), 2)
            self.assertEqual(len(evidence['products']), 7)
            self.assertEqual(set(evidence['inputs']), {'x'})

            changed_path = Path(next(iter(beat_products)))
            changed_path.write_bytes(b'changed after earlier shard completion')
            changed_beats = copy.deepcopy(beat_products)
            changed_beats[str(changed_path)].update(b.binding(changed_path))

            def changed_products(stage, rows, out):
                return aio_products if stage == 'allinone' else changed_beats

            with patch.object(b, 'inspect_audio', return_value=inputs['x']), \
                 patch.object(b, 'verify_products', side_effect=changed_products), \
                 self.assertRaisesRegex(ValueError, 'product semantics/hash changed'):
                r.audit_all('a' * 64, freeze, cohort, output)

    def test_altered_command_and_gpu_are_rejected(self):
        for field in ('command', 'gpu'):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as name:
                root = Path(name)
                output, freeze, row, shard, payload, inputs, products, _ = self.receipt_fixture(root)
                if field == 'command':
                    payload[field] = payload[field] + ['--altered']
                    message = 'command changed'
                else:
                    payload[field] = 99
                    message = 'identity/GPU/scope'
                path = self.put_receipt(root / 'receipt.json', payload)
                with self.assertRaisesRegex(ValueError, message):
                    self.validate_fixture(path, row, shard, freeze, output, inputs, products)

    def test_changed_input_pcm_is_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            output, freeze, row, shard, payload, inputs, products, _ = self.receipt_fixture(root)
            path = self.put_receipt(root / 'receipt.json', payload)
            changed = copy.deepcopy(inputs); changed['x']['waveform_float32_sha256'] = '2' * 64
            with self.assertRaisesRegex(ValueError, 'input PCM/hash changed'):
                self.validate_fixture(path, row, shard, freeze, output, changed, products)

    def test_changed_runtime_loader_is_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            runtime = Path(name) / 'runtime'; target = runtime / 'loader.py'
            target.parent.mkdir(); target.write_text('original\n')
            expected = hashlib.sha256(target.read_bytes()).hexdigest()
            target.write_text('mutated\n')
            with patch.object(b, 'verify_runtime', return_value={'checked_sha256': {}}), \
                 patch.object(r, 'EXTRA_RESOLUTION_CODE', {'loader.py': expected}), \
                 self.assertRaisesRegex(ValueError, 'hash mismatch'):
                r.verify_runtime({'runtime_root': str(runtime)})

    def test_receipt_payload_hash_is_mandatory(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            output, freeze, row, shard, payload, inputs, products, _ = self.receipt_fixture(root)
            path = root / 'receipt.json'
            b.write_new(path, {'payload': payload, 'receipt_sha256': '0' * 64})
            with patch.object(b, 'inspect_audio', return_value=inputs['x']), \
                 patch.object(b, 'verify_products', return_value=products), \
                 self.assertRaisesRegex(ValueError, 'envelope/hash'):
                r.validate_receipt(path, 'a' * 64, 'beats', shard, [row], freeze, output)


if __name__ == '__main__':
    unittest.main()
