"""Synthetic inventory tests; no network access or research data mutation."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from plan_historical_stem_objects_v1 import main


class InventoryTests(unittest.TestCase):
    def fixture(self, base, bad=False):
        audit, snapshot = base / 'audit', base / 'snapshot'
        audit.mkdir(); snapshot.mkdir()
        rows = [dict(status='verified', sha256='a' * 64, expected_sha256='a' * 64,
            bytes=12, error=None, path='/root/stem.wav', run=r, item_id='track',
            stem='vocals', duration_sec=10) for r in ('one', 'two')]
        if bad:
            rows[1]['status'] = 'hash_mismatch'
        raw = ''.join(json.dumps(r) + '\n' for r in rows).encode()
        (audit / 'records.jsonl').write_bytes(raw)
        (audit / 'COMMIT.json').write_text(json.dumps(dict(records_sha256=hashlib.sha256(raw).hexdigest(),
            memberships=2, statuses={'verified': 1, 'hash_mismatch': 1} if bad else {'verified': 2})))
        raw = json.dumps([dict(sha256='a' * 64, path='audio/stem.wav')]).encode()
        (snapshot / 'files.json').write_bytes(raw)
        (snapshot / 'COMMIT.json').write_text(json.dumps(dict(records_sha256=hashlib.sha256(raw).hexdigest(), revision='fixed')))
        return audit, snapshot

    def test_dedup_preserves_memberships(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            audit, snapshot = self.fixture(base)
            main(audit, snapshot, base / 'out')
            objects = json.loads((base / 'out/objects.json').read_text())
            self.assertEqual(len(objects), 1)
            self.assertEqual(len(objects[0]['memberships']), 2)
            self.assertEqual(objects[0]['source_paths'], ['/root/stem.wav'])
            self.assertEqual(objects[0]['public_paths'], ['audio/stem.wav'])
            self.assertEqual(objects[0]['publication_rights_status'], 'not_assessed_by_this_inventory')

    def test_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            audit, snapshot = self.fixture(base, bad=True)
            with self.assertRaises(AssertionError):
                main(audit, snapshot, base / 'out')
            self.assertFalse((base / 'out').exists())


if __name__ == '__main__':
    unittest.main()
