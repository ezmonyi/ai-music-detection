import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import soundfile as sf

import materialize_equal60_inputs_v2 as m


class MaterializationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.patch = patch.multiple(m, SECONDS=1, FRAMES=44100)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()

    def fixture(self, channels=1):
        source = self.root / 'native.wav'
        y = 1.1 * np.sin(2 * np.pi * 330 * np.arange(62400) / 48000)
        if channels > 1:
            y = np.column_stack([y] * channels)
        sf.write(source, y, 48000, format='WAV', subtype='FLOAT')
        row = {'id': 'test_001', 'role': 'development', 'audio_path': str(source),
               'physical_sample_rate_hz': '48000', 'physical_channels': str(channels),
               'physical_frames': '62400', 'crop_start_frame': '4800', 'crop_frames': '48000',
               'crop_end_frame_exclusive': '52800', 'audio_offset_s': '0.1',
               'native_sample_rate_hz': '48000', 'label': '0', 'source_group': 'human_test',
               'group_id': 'one_performer', 'evaluation_allowed': 'development'}
        ref = {'id': row['id'], 'role': row['role'], 'extraction_status': 'ok',
               'source_audio_path': str(source), 'source_audio_sha256': m.sha_file(source),
               'source_sample_rate': '48000', 'source_channels': str(channels),
               'source_total_frames': '62400', 'crop_start_frame': '4800', 'crop_frames': '48000',
               'extraction_contract_hash': m.FHM_CONTRACT_SHA, 'requested_duration_sec': '1'}
        return row, ref

    def test_exact_resample_and_mono_duplicate_no_limiter(self):
        row, ref = self.fixture()
        y, audit = m.decode_standardized(row, ref)
        self.assertEqual(y.shape, (44100, 2))
        np.testing.assert_array_equal(y[:, 0], y[:, 1])
        self.assertGreater(audit['standardized_peak_abs'], 1)
        self.assertGreater(audit['standardized_samples_abs_above_one'], 0)

    def test_native_start_must_match_fhm(self):
        row, ref = self.fixture()
        ref['crop_start_frame'] = '4801'
        with self.assertRaisesRegex(ValueError, 'crop audit mismatch'):
            m.decode_standardized(row, ref)

    def test_short_source_rejected_no_padding(self):
        row, ref = self.fixture()
        row.update(crop_start_frame='24000', audio_offset_s='0.5', crop_end_frame_exclusive='72000')
        ref['crop_start_frame'] = '24000'
        with self.assertRaisesRegex(ValueError, 'Short source'):
            m.decode_standardized(row, ref)

    def test_source_hash_mismatch(self):
        row, ref = self.fixture()
        ref['source_audio_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'source hash'):
            m.decode_standardized(row, ref)

    def test_multichannel_rejected(self):
        row, ref = self.fixture(3)
        with self.assertRaisesRegex(ValueError, 'channel layout'):
            m.decode_standardized(row, ref)

    def test_locked_role_rejected(self):
        row, ref = self.fixture()
        row['role'] = 'locked'
        with self.assertRaisesRegex(ValueError, 'role mismatch'):
            m.decode_standardized(row, ref)

    def test_resume_roundtrip_and_corruption_guard(self):
        row, ref = self.fixture(2)
        with patch.object(m.shutil, 'disk_usage', return_value=SimpleNamespace(free=10 * (1 << 30))):
            receipt = m.materialize(row, ref, self.root, 'test-contract')
        self.assertEqual(receipt, m.materialize(row, ref, self.root, 'test-contract'))
        with open(receipt['standardized_path'], 'ab') as f:
            f.write(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'Resume source/output hash mismatch'):
            m.materialize(row, ref, self.root, 'test-contract')

    def test_low_storage_refuses_publication(self):
        row, ref = self.fixture()
        with patch.object(m.shutil, 'disk_usage', return_value=SimpleNamespace(free=1 << 30)):
            with self.assertRaisesRegex(ValueError, 'storage reserve'):
                m.materialize(row, ref, self.root, 'test-contract')
        self.assertFalse((self.root / 'audio' / 'test_001.wav').exists())

    def test_selection_role_group_and_input_order_invariant(self):
        rows = [{'id': name, 'source_group': source, 'role': role} for name, source, role in
                [('a', 'x', 'development'), ('b', 'x', 'development'), ('c', 'y', 'development'), ('d', 'z', 'locked')]]
        first = m.select_rows(rows, True)
        self.assertEqual(first, m.select_rows(list(reversed(rows)), True))
        self.assertEqual(len(first), 2)
        self.assertTrue(all(r['role'] == 'development' for r in first))
        with self.assertRaisesRegex(ValueError, 'Duplicate IDs'):
            m.select_rows(rows + [copy.deepcopy(rows[0])])

    def test_unsafe_filename(self):
        for value in ('../x', '.', '/root/file', 'x/y', ''):
            with self.assertRaises(ValueError):
                m.safe_id(value)

    def test_atomic_publication_and_conflict(self):
        path = self.root / 'manifest.csv'
        body = b'a,b\\r\\n1,2\\r\\n'
        m.publish_bytes(path, body)
        m.publish_bytes(path, body)
        self.assertEqual(path.read_bytes(), body)
        with self.assertRaisesRegex(ValueError, 'Conflicting frozen publication'):
            m.publish_bytes(path, b'wrong')
        self.assertEqual(path.read_bytes(), body)
        self.assertEqual(len(list(self.root.iterdir())), 1)

    def test_reference_extraction_contract_rejected(self):
        row, ref = self.fixture()
        ref['extraction_contract_hash'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'Reference duration/extraction contract'):
            m.decode_standardized(row, ref)

    def test_existing_json_conflict_preserved(self):
        path = self.root / 'receipt.json'
        m.preserve_json(path, {'v': 1})
        original = path.read_bytes()
        with self.assertRaisesRegex(RuntimeError, 'Conflicting frozen JSON'):
            m.preserve_json(path, {'v': 2})
        self.assertEqual(original, path.read_bytes())


if __name__ == '__main__':
    unittest.main()
