import hashlib
from pathlib import Path
import tempfile
import unittest
from music_detector.rl.verified_restore import stage_archive


class VerifiedStageTests(unittest.TestCase):
    def test_verified_staging_retains_source_and_authenticates_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)/'source.tar';target=Path(folder)/'local.tar';data=b'fixed payload\0'*4096
            source.write_bytes(data)
            expected={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'md5':hashlib.md5(data,usedforsecurity=False).hexdigest()}
            self.assertEqual(stage_archive(source,target,expected),expected)
            self.assertEqual(source.read_bytes(),data);self.assertEqual(target.read_bytes(),data)

    def test_corrupt_source_is_not_promoted(self):
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)/'source.tar';target=Path(folder)/'local.tar';source.write_bytes(b'corrupt')
            expected={'bytes':7,'sha256':'0'*64,'md5':'0'*32}
            with self.assertRaises(ValueError):stage_archive(source,target,expected)
            self.assertFalse(target.exists());self.assertTrue(target.with_name('local.tar.partial').exists())


if __name__=='__main__':unittest.main()
