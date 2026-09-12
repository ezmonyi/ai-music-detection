"""Synthetic, network-free checks; fixtures are not research observations."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from audit_historical_stems_v1 import main, sha


class StemAuditTests(unittest.TestCase):
    def fixture(self, base):
        run = base / 'run'
        run.mkdir()
        roots = [base / 'first', base / 'second']
        for root in roots:
            (root / 'htdemucs' / 'track').mkdir(parents=True)
        (roots[0] / 'htdemucs/track/bass.wav').write_bytes(b'')
        (roots[1] / 'htdemucs/track/bass.wav').write_bytes(b'bass')
        (roots[0] / 'htdemucs/track/drums.wav').write_bytes(b'changed')
        (roots[1] / 'htdemucs/track/drums.wav').write_bytes(b'drums')
        (roots[0] / 'htdemucs/track/vocals.wav').write_bytes(b'unrecorded')
        digest = lambda b: hashlib.sha256(b).hexdigest()
        row = dict(item_id='track', duration_sec=10, run_fingerprint='frozen',
                   input_hashes=json.dumps(dict(bass_sha256=digest(b'bass'),
                       drums_sha256=digest(b'drums'), other_sha256=digest(b'other'))))
        state = run / 'expanded_features_10s.jsonl'
        state.write_text(json.dumps(row) + '\n')
        (run / 'expanded_features_10s_metadata.json').write_text(json.dumps(dict(
            run_fingerprint='frozen', run_payload=dict(demix_roots=list(map(str, roots))))))
        return run, state, row

    def test_statuses_and_root_precedence(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            run, state, _ = self.fixture(base)
            out = base / 'audit'
            main([run], out)
            rows = {r['stem']: r for r in map(json.loads, (out / 'records.jsonl').read_text().splitlines())}
            self.assertEqual({k: r['status'] for k, r in rows.items()}, dict(
                bass='verified', drums='hash_mismatch', other='missing', vocals='not_recorded'))
            self.assertIn('/second/', rows['bass']['path'])
            self.assertIn('/first/', rows['drums']['path'])
            self.assertIsNone(rows['vocals']['sha256'])
            commit = json.loads((out / 'COMMIT.json').read_text())
            self.assertEqual(commit['unique_hashed_paths'], 2)
            self.assertEqual(commit['records_sha256'], sha(out / 'records.jsonl'))
            self.assertEqual(commit['input_sha256'][str(state)], sha(state))
            self.assertFalse(commit['whole_project_complete'])
            with self.assertRaises(FileExistsError):
                main([run], out)

    def test_fingerprint_mismatch_cannot_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            run, state, row = self.fixture(base)
            row['run_fingerprint'] = 'different'
            state.write_text(json.dumps(row) + '\n')
            with self.assertRaises(AssertionError):
                main([run], base / 'audit')
            self.assertFalse((base / 'audit/COMMIT.json').exists())

    def test_duplicate_id_cannot_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            run, state, row = self.fixture(base)
            state.write_text((json.dumps(row) + '\n') * 2)
            with self.assertRaises(AssertionError):
                main([run], base / 'audit')
            self.assertFalse((base / 'audit/COMMIT.json').exists())


if __name__ == '__main__':
    unittest.main()
