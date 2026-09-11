import tempfile
from pathlib import Path
import unittest
import zipfile
from verify_saraga_archive_v1 import inspect_zip, file_hashes


class ArchiveTests(unittest.TestCase):
    def test_full_members_hashed_and_crc_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'synthetic.zip'
            with zipfile.ZipFile(path, 'w') as z:
                z.writestr('folder/test.mp3', b'synthetic-not-real-audio')
                z.writestr('folder/test.json', b'{}')
            rows = inspect_zip(path)
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(r['crc_verified'] for r in rows))
            self.assertEqual(rows[0]['bytes'], 24)
            self.assertEqual(len(file_hashes(path)['sha256']), 64)

    def test_path_traversal_refused_without_extraction(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'synthetic.zip'
            with zipfile.ZipFile(path, 'w') as z:
                z.writestr('../escape', b'data')
            with self.assertRaisesRegex(ValueError, 'Unsafe'):
                inspect_zip(path)

    def test_corrupt_member_crc_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'synthetic.zip'
            marker = b'UNIQUE_UNCOMPRESSED_PAYLOAD'
            with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_STORED) as z:
                z.writestr('test.bin', marker)
            data = path.read_bytes()
            path.write_bytes(data.replace(marker, b'X'+marker[1:], 1))
            with self.assertRaises(zipfile.BadZipFile):
                inspect_zip(path)


if __name__ == '__main__':
    unittest.main()
