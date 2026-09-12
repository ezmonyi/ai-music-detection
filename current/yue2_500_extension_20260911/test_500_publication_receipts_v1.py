"""Offline acceptance gate tests; no upload, network or real audio required."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


class ReceiptTests(unittest.TestCase):
    def setUp(self):
        stub=types.ModuleType('huggingface_hub');stub.HfApi=object
        spec=importlib.util.spec_from_file_location('verifier',Path(__file__).with_name('verify_500_original_publications_v1.py'))
        self.module=importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules,{'huggingface_hub':stub}):spec.loader.exec_module(self.module)
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.rows=[dict(path=f'audio/{i}.mp3') for i in range(500)]
        for i in range(5):
            (self.root/f'batch_{i:03d}.json').write_text(json.dumps(dict(
                files=[r['path'] for r in self.rows[i*100:(i+1)*100]],sha256_verified=True)))

    def test_complete_receipts_pass(self):self.module.check_receipts(self.root,self.rows)

    def test_partial_receipts_fail(self):
        (self.root/'batch_004.json').unlink()
        with self.assertRaises(AssertionError):self.module.check_receipts(self.root,self.rows)

    def test_wrong_file_order_fails(self):
        self.rows[0],self.rows[1]=self.rows[1],self.rows[0]
        with self.assertRaises(AssertionError):self.module.check_receipts(self.root,self.rows)

    def test_unverified_batch_fails(self):
        p=self.root/'batch_002.json';r=json.loads(p.read_text());r['sha256_verified']=False;p.write_text(json.dumps(r))
        with self.assertRaises(AssertionError):self.module.check_receipts(self.root,self.rows)

    def test_wrong_sequence_fails(self):
        (self.root/'batch_004.json').rename(self.root/'batch_005.json')
        with self.assertRaises(AssertionError):self.module.check_receipts(self.root,self.rows)


if __name__=='__main__':unittest.main()
