"""Offline checks of immutable receipts and remote metadata validation."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


def load():
    stub = types.ModuleType('huggingface_hub')
    stub.HfApi = object
    stub.CommitOperationAdd = object
    spec = importlib.util.spec_from_file_location('publication_under_test',
        Path(__file__).with_name('publish_yue2_originals_v1.py'))
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {'huggingface_hub': stub}):
        spec.loader.exec_module(module)
    return module


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.module = load()

    def test_receipt_identical_resume_and_conflict(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'receipt.json'
            self.module.save(path, {'revision':'first'})
            original = path.read_bytes()
            self.module.save(path, {'revision':'first'})
            with self.assertRaises(AssertionError):
                self.module.save(path, {'revision':'different'})
            self.assertEqual(path.read_bytes(), original)

    def check_remote(self, size=10, sha='expected', missing=False):
        def info(repo, **kwargs):
            self.assertEqual(kwargs['revision'], 'fixed-revision')
            self.assertEqual(kwargs['paths'], ['audio/test.flac'])
            return [] if missing else [types.SimpleNamespace(path='audio/test.flac',
                size=size,lfs=types.SimpleNamespace(sha256=sha))]
        self.module.verify(types.SimpleNamespace(get_paths_info=info),
            [dict(path='audio/test.flac',bytes=10,sha256='expected')], 'fixed-revision')

    def test_remote_exact_match(self):
        self.check_remote()

    def test_remote_wrong_size(self):
        with self.assertRaises(AssertionError):self.check_remote(size=11)

    def test_remote_wrong_hash(self):
        with self.assertRaises(AssertionError):self.check_remote(sha='wrong')

    def test_remote_missing_file(self):
        with self.assertRaises(KeyError):self.check_remote(missing=True)


if __name__ == '__main__':unittest.main()
