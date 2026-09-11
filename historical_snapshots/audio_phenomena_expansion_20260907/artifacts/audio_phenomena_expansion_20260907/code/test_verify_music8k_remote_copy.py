import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from verify_music8k_remote_copy import verify,sha


class CopyTests(unittest.TestCase):
    def fixture(self,root):
        (root/'raw/mureka_v9').mkdir(parents=True);(root/'items').mkdir()
        (root/'contract.json').write_text('{}')
        rows=[];receipts={}
        for i in range(500):
            p=root/'raw/mureka_v9'/f'{i}.mp3';p.write_bytes(b'x')
            item=root/'items'/f'{i}.json';item.write_text('{}');receipts[item.name]=sha(item)
            rows.append(dict(id=str(i),bytes=1,sha256=sha(p)))
        (root/'summary.json').write_text(json.dumps(dict(receipts_sha256=receipts)))
        audit=dict(status='passed',selected=500,all_original_files_rehashed=True,classifier_admission_authorized=False,
                   contract_sha256=sha(root/'contract.json'),summary_sha256=sha(root/'summary.json'),rows=rows,bytes=500)
        p=root/'audit.json';p.write_text(json.dumps(audit));return p
    def test_complete_and_tampered_raw(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);audit=self.fixture(root)
            self.assertEqual(verify(root,audit)['rows'],500)
            (root/'raw/mureka_v9/12.mp3').write_bytes(b'y')
            with self.assertRaisesRegex(ValueError,'Copied raw changed'):verify(root,audit)
    def test_receipt_changed(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);audit=self.fixture(root)
            (root/'items/12.json').write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError,'Copied receipt changed'):verify(root,audit)
    def test_summary_changed(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);audit=self.fixture(root)
            (root/'summary.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'contract/summary changed'):verify(root,audit)


if __name__=='__main__':unittest.main()
