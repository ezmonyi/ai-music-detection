from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import prepare_native30_inference_freeze_v2 as d
import run_native30_inference_batches_v1 as b
import run_native30_inference_batches_v2 as r
import test_run_native30_inference_batches_v1 as v1tests


class DraftTests(unittest.TestCase):
    def fixture(self, root, aio_gpus=None, beat_gpus=None):
        cohort_entry, expected, helper, plan, screen = v1tests.RunnerTests().build_freeze(root)
        old_freeze = b.read_json(cohort_entry['path'])
        cohort_path = Path(old_freeze['cohort_contract']['path'])
        cohort = b.read_json(cohort_path)
        Path(plan['path']).write_text('{"kind":"plan","rows":[{"id":"synthetic_1"}]}\n')
        Path(screen['path']).write_text('{"kind":"screen","rows":[{"id":"synthetic_1"}]}\n')
        plan, screen = b.binding(plan['path']), b.binding(screen['path'])
        cohort['bindings'][plan['path']] = plan; cohort['bindings'][screen['path']] = screen
        helper_entry = b.binding(Path(d.__file__).resolve().with_name('run_native30_fhsc_cohort_v1.py'))
        cohort['bindings'][helper_entry['path']] = helper_entry
        prepared = root / 'prepared.json'; b.write_new(prepared, cohort)
        runtime = root / 'runtime'; runtime.mkdir()
        runtime_contract = root / 'runtime_contract.json'; runtime_contract.write_text('{}\n')
        checkpoints = root / 'checkpoints.json'; checkpoints.write_text('{}\n')
        old_code = root / 'old_code'; old_code.mkdir()
        bias = root / 'bias.npz'; bias.write_bytes(b'bias')
        output = root / 'new_inference_output'
        candidate = d.build_candidate(
            prepared, b.digest(prepared), output, aio_gpus or [0], beat_gpus or [1], 1,
            runtime_root=runtime, runtime_contract=runtime_contract,
            checkpoint_manifest=checkpoints, old_code_root=old_code, bias=bias,
            expected=expected, helper_loader=lambda entry: helper,
            plan_sha=plan['sha256'], screen_sha=screen['sha256'],
            runtime_contract_sha=b.digest(runtime_contract),
            checkpoint_manifest_sha=b.digest(checkpoints), bias_sha=b.digest(bias))
        return candidate, expected, helper, plan, screen

    def test_exact_3830_ordered_shards_are_derived_not_hand_built(self):
        rows = [{'id': f'x{index:04d}'} for index in range(3830)]
        shards = d.make_shards(rows, 24)
        self.assertEqual(len(shards), 160)
        self.assertEqual(shards[-1]['rows'], 14)
        self.assertEqual([ident for shard in shards for ident in shard['item_ids']],
                         [row['id'] for row in rows])
        b.validate_shards(shards, rows)

    def test_candidate_replays_metadata_graph_without_audio_and_binds_all_authorities(self):
        with tempfile.TemporaryDirectory() as name, patch.object(b, 'sf') as forbidden_audio:
            candidate, _, _, plan, screen = self.fixture(Path(name))
        forbidden_audio.SoundFile.assert_not_called()
        self.assertEqual(candidate['plan']['sha256'], plan['sha256'])
        self.assertEqual(candidate['screen']['sha256'], screen['sha256'])
        self.assertEqual(candidate['runner']['sha256'], d.INFERENCE_V2_SHA)
        self.assertEqual(candidate['v1_dependency']['sha256'], r.V1_SHA)
        self.assertEqual(candidate['child_environment'], r.child_environment_policy(candidate['runtime_root']))
        self.assertFalse(candidate['feature_extraction_authorized'])
        self.assertFalse(candidate['recovery_applied'])

    def test_wrapper_is_explicitly_non_authorizing_and_not_a_runner_freeze(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); candidate, expected, helper, plan, screen = self.fixture(root)
            draft = d.make_draft(candidate)
            path = root / 'draft.json'; b.write_new(path, draft)
            self.assertEqual(draft['status'], d.STATUS)
            self.assertFalse(draft['production_authorized'])
            self.assertEqual(draft['candidate_rows'], 1)
            self.assertEqual(draft['candidate_sha256'], b.value_hash(candidate))
            with self.assertRaisesRegex(ValueError, 'parent freeze schema/status'):
                r.validate_freeze(path, b.digest(path), expected=expected,
                                  helper_loader=lambda entry: helper,
                                  plan_sha=plan['sha256'], screen_sha=screen['sha256'])

    def test_invalid_gpu_lists_and_overlap_fail_closed(self):
        for value in ('', '0,0', '-1', 'a'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                d.parse_gpus(value)
        with tempfile.TemporaryDirectory() as name:
            with self.assertRaisesRegex(ValueError, 'GPU sets'):
                self.fixture(Path(name), aio_gpus=[0], beat_gpus=[0])


if __name__ == '__main__':
    unittest.main()
