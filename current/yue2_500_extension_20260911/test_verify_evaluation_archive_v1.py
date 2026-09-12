import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from verify_evaluation_archive_v1 import verify

class ArchiveTests(unittest.TestCase):
    def fixture(self, path, corrupt=False, traversal=False):
        data = b'experiment-result'
        binding = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        commit = json.dumps({'products': {'results.json': binding}}).encode()
        with tarfile.open(path, 'w:gz') as archive:
            for name, raw in [('run/COMMIT.json', commit),
                              ('../escape' if traversal else 'run/results.json', b'wrong' if corrupt else data)]:
                member = tarfile.TarInfo(name)
                member.size = len(raw)
                archive.addfile(member, io.BytesIO(raw))
        return hashlib.sha256(commit).hexdigest()

    def test_verified_members(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'a.tar.gz'
            pin = self.fixture(path)
            self.assertEqual(verify(path, pin)['products'], 1)

    def test_reject_corrupt_product(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'a.tar.gz'
            pin = self.fixture(path, corrupt=True)
            with self.assertRaises(AssertionError):
                verify(path, pin)

    def test_reject_traversal(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'a.tar.gz'
            pin = self.fixture(path, traversal=True)
            with self.assertRaises(AssertionError):
                verify(path, pin)

    def test_expanded_receipt_inventory(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'a.tar.gz'
            data = b'{}'
            entry = dict(bytes=2, sha256=hashlib.sha256(data).hexdigest())
            commit = json.dumps({'model_receipts': {'uid': entry}, 'contract': entry}).encode()
            with tarfile.open(path, 'w:gz') as archive:
                for name, raw in [('run/COMMIT.json', commit), ('run/models/uid.json', data),
                                  ('run/contract.json', data)]:
                    member = tarfile.TarInfo(name)
                    member.size = len(raw)
                    archive.addfile(member, io.BytesIO(raw))
            result = verify(path, hashlib.sha256(commit).hexdigest())
            self.assertEqual(result['products'], 2)

if __name__ == '__main__':
    unittest.main()
