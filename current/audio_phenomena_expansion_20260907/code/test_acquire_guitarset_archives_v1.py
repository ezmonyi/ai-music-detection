"""Offline archive-acquisition tests; no network or real archive acceptance."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import acquire_guitarset_archives_v1 as target


class AcquisitionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.payload = b'test archive bytes, not a valid ZIP'
        self.md5 = hashlib.md5(self.payload).hexdigest()
        self.metadata = {'doi': '10.5281/zenodo.3371780', 'metadata': {
            'license': {'id': 'cc-by-4.0'}, 'access_right': 'open', 'version': '1.1.0'},
            'files': [{'key': name, 'size': len(self.payload), 'checksum': 'md5:' + self.md5,
                'links': {'self': f'https://zenodo.org/api/records/3371780/files/{name}/content'}}
                for name in target.EXPECTED]}
        self.path = self.root / 'metadata.json'
        self.output = self.root / 'output'

    def fake_curl(self, command, **kwargs):
        Path(command[command.index('--output') + 1]).write_bytes(self.payload)
        Path(command[command.index('--dump-header') + 1]).write_text('mock HTTP response\n')
        return subprocess.CompletedProcess(command, 0, '', '')

    def run_main(self, run=None, bad_hash=False):
        self.path.write_text(json.dumps(self.metadata))
        metadata_hash = target.digest(self.path) if not bad_hash else '0' * 64
        expected = {name: (len(self.payload), self.md5) for name in target.EXPECTED}
        with patch.object(sys, 'argv', ['acquire', '--metadata', str(self.path),
                '--metadata-sha256', metadata_hash, '--output', str(self.output)]), \
             patch.object(target, 'EXPECTED', expected), \
             patch.object(target.subprocess, 'run', side_effect=run or self.fake_curl) as mocked:
            target.main()
            return mocked.call_count

    def test_success_commits_only_archive_acceptance(self):
        self.assertEqual(self.run_main(), 2)
        commit = json.loads((self.output / 'COMMIT.json').read_text())
        self.assertEqual(len(commit['products']), 8)
        for name, item in commit['products'].items():
            self.assertEqual(target.digest(self.output / name), item['sha256'])
            self.assertEqual((self.output / name).stat().st_size, item['bytes'])
        receipt = json.loads((self.output / 'acquisition_receipt.json').read_text())
        self.assertEqual(receipt['status'], 'archives_verified_not_decoded')
        for key in ('zip_crc_checked', 'audio_decoded', 'external_gate_passed', 'bc_measured'):
            self.assertFalse(receipt[key])

    def test_wrong_metadata_hash_no_output(self):
        with self.assertRaisesRegex(ValueError, 'metadata SHA'):
            self.run_main(bad_hash=True)
        self.assertFalse(self.output.exists())

    def test_wrong_license_no_output(self):
        self.metadata['metadata']['license']['id'] = 'unknown'
        with self.assertRaisesRegex(ValueError, 'provenance/license'):
            self.run_main()
        self.assertFalse(self.output.exists())

    def test_wrong_url_no_output(self):
        self.metadata['files'][0]['links']['self'] = 'https://example.com/archive.zip'
        with self.assertRaisesRegex(ValueError, 'archive identity'):
            self.run_main()
        self.assertFalse(self.output.exists())

    def test_download_failure_preserves_partial_without_commit(self):
        def failing(command, **kwargs):
            self.fake_curl(command, **kwargs)
            return subprocess.CompletedProcess(command, 22, '', 'mock failure')
        with self.assertRaises(RuntimeError):
            self.run_main(run=failing)
        self.assertTrue((self.output / 'audio_mono-mic.zip.part').exists())
        self.assertTrue((self.output / 'EXECUTION_FAILURE.json').exists())
        self.assertFalse((self.output / 'COMMIT.json').exists())

    def test_bad_bytes_fail_closed(self):
        def corrupt(command, **kwargs):
            result = self.fake_curl(command, **kwargs)
            Path(command[command.index('--output') + 1]).write_bytes(b'bad')
            return result
        with self.assertRaisesRegex(ValueError, 'size/MD5'):
            self.run_main(run=corrupt)
        self.assertFalse((self.output / 'COMMIT.json').exists())

    def test_existing_output_is_not_overwritten(self):
        self.output.mkdir()
        sentinel = self.output / 'user_owned.txt'
        sentinel.write_text('preserve')
        with self.assertRaises(FileExistsError):
            self.run_main()
        self.assertEqual(sentinel.read_text(), 'preserve')
        self.assertFalse((self.output / 'COMMIT.json').exists())


if __name__ == '__main__':
    unittest.main()
