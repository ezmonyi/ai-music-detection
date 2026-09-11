import hashlib
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import resume_guitarset_archives_v2 as module


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.part = self.root / 'file.part'
        self.payload = b'0123456789'
        self.md5 = hashlib.md5(self.payload).hexdigest()

    def execute(self, run):
        with patch.object(module.subprocess, 'run', side_effect=run), patch.object(module.time, 'sleep'):
            return module.resume_file('https://example.invalid/file', self.part, 10, self.md5, self.root/'logs', 3)

    def test_timeout_then_resume_prefix(self):
        self.part.write_bytes(self.payload[:3])
        lengths = []
        def run(command, **kwargs):
            self.assertIn('--continue-at', command)
            before = self.part.stat().st_size
            lengths.append(before)
            after = 6 if before == 3 else 10
            with self.part.open('ab') as stream:
                stream.write(self.payload[before:after])
            return subprocess.CompletedProcess(command, 28 if after < 10 else 0, '', '')
        self.assertEqual(len(self.execute(run)), 2)
        self.assertEqual(lengths, [3, 6])
        self.assertEqual(self.part.read_bytes(), self.payload)

    def test_already_complete_hash_checked_without_network(self):
        self.part.write_bytes(self.payload)
        self.assertEqual(self.execute(lambda *a, **k: self.fail('network not expected')), [])

    def test_bad_complete_bytes_rejected_unchanged(self):
        self.part.write_bytes(b'x'*10)
        with self.assertRaisesRegex(ValueError, 'wrong MD5'):
            self.execute(lambda *a, **k: self.fail('network not expected'))
        self.assertEqual(self.part.read_bytes(), b'x'*10)

    def test_unsupported_resume_fails_without_restarting(self):
        self.part.write_bytes(self.payload[:3])
        with self.assertRaisesRegex(RuntimeError, 'non-retryable'):
            self.execute(lambda command, **kwargs: subprocess.CompletedProcess(command,33,'','unsupported Range'))
        self.assertEqual(self.part.read_bytes(), self.payload[:3])

    def test_corrupt_final_bytes_rejected(self):
        def run(command, **kwargs):
            self.part.write_bytes(b'x'*10)
            return subprocess.CompletedProcess(command,0,'','')
        with self.assertRaisesRegex(ValueError, 'MD5 mismatch'):
            self.execute(run)

    def test_exhausted_retries_keep_partial(self):
        self.part.write_bytes(self.payload[:3])
        with self.assertRaisesRegex(RuntimeError, 'exhausted'):
            self.execute(lambda command, **kwargs: subprocess.CompletedProcess(command,28,'','timeout'))
        self.assertEqual(self.part.read_bytes(), self.payload[:3])
        self.assertEqual(len(list((self.root/'logs').glob('*.command.json'))), 3)


if __name__ == '__main__':
    unittest.main()
