import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import run_equal60_inference_batches as runner


class BatchTests(unittest.TestCase):
    def test_partition_all_67_shards_once(self):
        shards = list(range(67))
        jobs = runner.assignments(shards, [0, 1, 2, 3])
        self.assertEqual(sorted(x for _, items in jobs for x in items), shards)
        self.assertEqual([len(items) for _, items in jobs], [17, 17, 17, 16])

    def test_duplicate_gpu_rejected(self):
        for gpus in ([], [1, 1]):
            with self.assertRaises(ValueError):
                runner.assignments([], gpus)

    def test_gpu_guard(self):
        for output in ('NVIDIA GeForce RTX 4090, 2, 0', 'NVIDIA GeForce RTX 5090, 2000, 0',
                       'NVIDIA GeForce RTX 5090, 2, 20'):
            with patch.object(runner.subprocess, 'check_output', return_value=output):
                with self.assertRaises(RuntimeError):
                    runner.require_idle_5090(0)
        with patch.object(runner.subprocess, 'check_output', return_value='NVIDIA GeForce RTX 5090, 2, 0'):
            runner.require_idle_5090(0)

    def test_command_precision_and_gpu(self):
        base = Path('/runtime')
        aio = runner.command('allinone', ['a.wav', 'b.wav'], base, Path('/out'))
        self.assertEqual(aio[1:3], ['a.wav', 'b.wav'])
        self.assertNotIn('--float16', aio)
        self.assertIn('cuda', aio)
        beat = runner.command('beats', ['a.wav'], base, Path('/out'))
        self.assertIn('--float16', beat)
        self.assertIn('--no-dbn', beat)
        self.assertEqual(beat[beat.index('--gpu')+1], '0')

    def test_output_products_disjoint_between_stages(self):
        aio = runner.product_paths('allinone', 'track', Path('/out'))
        beat = runner.product_paths('beats', 'track', Path('/out'))
        self.assertEqual(len(aio), 6)
        self.assertFalse(set(aio) & set(beat))

    def test_resume_hash_and_identity_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'product'
            path.write_bytes(b'abc')
            receipt = dict(status='passed', run_contract_sha256='contract', stage='beats',
                           shard_index=0, item_ids=['a'], outputs={str(path): {
                               'sha256': hashlib.sha256(b'abc').hexdigest(), 'bytes': 3}})
            runner.validate_resume(receipt, 'contract', 'beats', 0, ['a'])
            with self.assertRaises(ValueError):
                runner.validate_resume(receipt, 'changed', 'beats', 0, ['a'])
            with self.assertRaises(ValueError):
                runner.validate_resume(receipt, 'contract', 'beats', 0, ['b'])
            path.write_bytes(b'abd')
            with self.assertRaises(ValueError):
                runner.validate_resume(receipt, 'contract', 'beats', 0, ['a'])


if __name__ == '__main__':
    unittest.main()
