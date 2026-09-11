import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from verify_committed_byte_mirror_v2 import sha, verify


class MirrorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'item').write_bytes(b'example')

    def commit(self, key='files'):
        products = {'item': {'bytes': 7, 'sha256': hashlib.sha256(b'example').hexdigest()}}
        path = self.root / 'COMMIT.json'
        path.write_text(json.dumps({'status': 'committed', key: products}))
        return sha(path)

    def test_both_explicit_schemas(self):
        for key in ('files', 'products'):
            result = verify(self.root, self.commit(key), key)
            self.assertTrue(result['passed'])
            self.assertEqual(result['product_bytes'], 7)

    def test_wrong_hash(self):
        self.commit()
        with self.assertRaises(ValueError):
            verify(self.root, '0' * 64, 'files')

    def test_extra_file(self):
        digest = self.commit()
        (self.root / 'extra').touch()
        with self.assertRaisesRegex(ValueError, 'inventory'):
            verify(self.root, digest, 'files')

    def test_changed_bytes(self):
        digest = self.commit()
        (self.root / 'item').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'byte mismatch'):
            verify(self.root, digest, 'files')

    def test_symlink_rejected(self):
        digest = self.commit()
        (self.root / 'link').symlink_to(self.root / 'item')
        with self.assertRaisesRegex(ValueError, 'symlink'):
            verify(self.root, digest, 'files')


if __name__ == '__main__':
    unittest.main()
