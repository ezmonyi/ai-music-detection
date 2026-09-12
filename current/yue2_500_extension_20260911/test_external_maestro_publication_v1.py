"""Offline verification edge-case tests; no upload or source-file mutation."""
import hashlib
import io
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import publish_external_maestro_v1 as publication


class VerificationTests(unittest.TestCase):
    def setUp(self):
        self.raw=b'MThd\x00\x00\x00\x06test'
        self.row=dict(path='external_controls/test/sample.midi',bytes=len(self.raw),sha256=hashlib.sha256(self.raw).hexdigest())

    def api(self,lfs=None,size=None,missing=False):
        entry=SimpleNamespace(path=self.row['path'],size=len(self.raw) if size is None else size,lfs=lfs)
        return SimpleNamespace(get_paths_info=lambda *a,**k:[] if missing else [entry])

    def test_lfs(self):
        publication.verify(self.api(SimpleNamespace(sha256=self.row['sha256'])),[self.row],'rev')

    def test_plain_git_midi(self):
        with patch.object(publication.urllib.request,'urlopen',return_value=io.BytesIO(self.raw)):
            publication.verify(self.api(),[self.row],'rev')

    def test_corrupt_plain_git_midi(self):
        with patch.object(publication.urllib.request,'urlopen',return_value=io.BytesIO(b'bad')):
            with self.assertRaises(AssertionError):publication.verify(self.api(),[self.row],'rev')

    def test_wrong_lfs_hash(self):
        with self.assertRaises(AssertionError):
            publication.verify(self.api(SimpleNamespace(sha256='bad')),[self.row],'rev')

    def test_missing_object(self):
        with self.assertRaises(AssertionError):publication.verify(self.api(missing=True),[self.row],'rev')

    def test_wrong_size(self):
        with self.assertRaises(AssertionError):publication.verify(self.api(size=0),[self.row],'rev')


if __name__=='__main__':unittest.main()
