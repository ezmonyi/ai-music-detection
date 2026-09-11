"""Synthetic fixtures only; the production CLI never exposes count overrides."""
import copy
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import materialize_native30_new1695_v1 as m


def metadata_fixture():
    candidates, rows, components = [], [], []
    for index, channels in enumerate([2, 1]):
        ident = 'item' + str(index)
        component = 'component' + str(index)
        native = {'path': '/mnt/native/' + ident + '.wav', 'channels': channels,
                  'duration_s': 31.0, 'sample_rate_hz': 100, 'sha256': 'a' * 64}
        candidates.append({'id': ident, 'source_group': 'FMA', 'label': '0', 'component_id': component,
                           'accepted60_freeze_relation': 'absent', 'native_evidence': native,
                           'native_stereo_metadata_eligible': channels == 2,
                           'receipt_binding': 'synthetic', 'receipt_row': index})
        rows.append({'id': ident, 'source_group': 'FMA', 'label': '0', 'component_id': component,
                     'role': 'development', 'group_id': 'group' + str(index), 'exclusion_reasons': [],
                     'duration_exposure_candidate': True, 'full_native_duration_known': True})
        components.append({'component_id': component, 'members': [ident],
                           'candidate_screen_members': [ident], 'protected_relationships': []})
    return {'candidates': candidates}, {'rows': rows, 'components': components}, {
        'candidates': 2, 'selected': 1, 'mono': 1, 'local': 0, 'sources': {'FMA': 1}}


class FakeDSP:
    """In-memory DSP double: publication/resume tests use explicitly fake bytes."""
    def __init__(self):
        self.fail = set()
        self.calls = []

    def standardize(self, path, sha, rate, channels, approved_region=None):
        ident = Path(path).stem
        self.calls.append(ident)
        if ident in self.fail:
            raise ValueError('native region shorter than30s: padding forbidden')
        if m.digest(path) != sha:
            raise ValueError('source SHA256 mismatch')
        values = b'synthetic-WAV-FLOAT-' + ident.encode()
        return values, {'region_kind': 'full_native_sequential_decode', 'native_channels': 2,
                        'sequential_decode': {'actual_frames': rate * 31},
                        'output_waveform_float32_sha256': hashlib.sha256(values).hexdigest(),
                        'peak_absolute': 2.0, 'samples_abs_above_one': 1,
                        'coordinates': {'crop_start_frame': rate // 2, 'crop_frames': rate * 30}}


def fake_publish(dsp, path, values):
    with path.open('xb') as stream:
        stream.write(values)


def contract_fixture(root, count=2):
    rows = []
    for index in range(count):
        ident = 'item' + str(index)
        source = root / (ident + '.wav')
        source.write_bytes(b'synthetic-native-' + ident.encode())
        rows.append({'id': ident, 'source_group': 'FMA', 'group_id': 'group' + str(index),
                     'component_id': 'component' + str(index), 'role': 'development', 'label': '0',
                     'execution_native_path': str(source), 'original_native_path': str(source),
                     'native_evidence': {'sha256': m.digest(source), 'sample_rate_hz': 100, 'channels': 2}})
    return {'version': m.VERSION, 'expected_count': count, 'rows': rows,
            'output_root': str(root / 'output'), 'bindings': {}, 'excluded_native_mono': []}


class MaterializerTests(unittest.TestCase):
    def test_full_reconciliation_and_mono_accountability(self):
        report, screen, expected = metadata_fixture()
        rows, mono = m._reconcile(report, screen, '/tmp/copies', expected)
        self.assertEqual([r['id'] for r in rows], ['item0'])
        self.assertEqual([r['id'] for r in mono], ['item1'])
        self.assertEqual(rows[0]['role'], 'development')
        self.assertEqual(mono[0]['reason'], 'native_mono')

    def test_mono_flag_and_all_exclusion_counts_enforced(self):
        report, screen, expected = metadata_fixture()
        report['candidates'][1]['native_stereo_metadata_eligible'] = True
        with self.assertRaisesRegex(ValueError, 'stereo flag'):
            m._reconcile(report, screen, '/tmp/copies', expected)
        report, screen, expected = metadata_fixture()
        expected['mono'] = 0
        with self.assertRaisesRegex(ValueError, 'split mismatch'):
            m._reconcile(report, screen, '/tmp/copies', expected)

    def test_protected_role_prior_and_source_rejected(self):
        for kind in ['component', 'role', 'prior', 'source']:
            report, screen, expected = metadata_fixture()
            if kind == 'component':
                screen['components'][0]['protected_relationships'] = ['reserved']
            elif kind == 'role':
                screen['rows'][0]['role'] = 'reserved'
            elif kind == 'prior':
                report['candidates'][0]['accepted60_freeze_relation'] = 'selected'
            else:
                report['candidates'][0]['source_group'] = screen['rows'][0]['source_group'] = 'Mureka'
            with self.assertRaises(ValueError):
                m._reconcile(report, screen, '/tmp/copies', expected)

    def test_id_label_component_mismatch(self):
        for key, value in [('id', 'absent'), ('label', '1'), ('component_id', 'wrong')]:
            report, screen, expected = metadata_fixture()
            report['candidates'][0][key] = value
            with self.assertRaises(ValueError):
                m._reconcile(report, screen, '/tmp/copies', expected)

    def test_copy_mapping_keeps_native_relative_paths(self):
        for source, suffix in [('FMA', 'fma_medium/024/024217.mp3'),
                               ('Suno', 'suno-ai-music-dataset-audio/audio/abc.mp3'),
                               ('Suno', 'humair025-suno-audio-mp3/batch_13/abc.mp3')]:
            path, local = m.execution_path(str(m.WORKSPACE / suffix), source, '/mnt/copies')
            self.assertEqual(path, '/mnt/copies/' + suffix)
            self.assertTrue(local)
        self.assertEqual(m.execution_path('/mnt/original/abc.flac', 'Udio', '/mnt/copies'),
                         ('/mnt/original/abc.flac', False))
        for path, source in [('/Users/elsewhere/a.wav', 'FMA'),
                             (str(m.WORKSPACE / 'fma_medium/../a.wav'), 'FMA'),
                             (str(m.WORKSPACE / 'fma_medium/a.wav'), 'Mureka')]:
            with self.assertRaises(ValueError):
                m.execution_path(path, source, '/mnt/copies')

    @patch.object(m, 'publish_waveform', fake_publish)
    def test_complete_commit_and_verified_resume_no_new_processing(self):
        with tempfile.TemporaryDirectory() as directory:
            contract = contract_fixture(Path(directory)); dsp = FakeDSP()
            result = m._run_contract(contract, dsp, workers=2)
            self.assertEqual(result['completed'], 2)
            output = Path(contract['output_root'])
            self.assertTrue((output / 'COMMIT.json').is_file())
            before = m.digest(output / 'COMMIT.json'); calls = len(dsp.calls)
            again = m._run_contract(contract, dsp, workers=1)
            self.assertEqual(again['status'], 'verified_existing_COMMIT')
            self.assertEqual(m.digest(output / 'COMMIT.json'), before)
            self.assertEqual(len(dsp.calls), calls)
            manifest = m.read_json(output / 'manifest.json')
            self.assertEqual([r['row']['id'] for r in manifest['records']], ['item0', 'item1'])

    @patch.object(m, 'publish_waveform', fake_publish)
    def test_short_failure_partial_and_resumption_preserves_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            contract = contract_fixture(Path(directory)); dsp = FakeDSP(); dsp.fail.add('item1')
            first = m._run_contract(contract, dsp)
            output = Path(contract['output_root'])
            self.assertEqual((first['completed'], first['failed']), (1, 1))
            self.assertFalse((output / 'COMMIT.json').exists())
            receipt_before = m.digest(output / 'items/item0.json')
            failure_files = list((output / 'failures').iterdir())
            self.assertEqual(len(failure_files), 1)
            dsp.fail.clear()
            second = m._run_contract(contract, dsp)
            self.assertEqual(second['completed'], 2)
            self.assertEqual(m.digest(output / 'items/item0.json'), receipt_before)
            self.assertTrue(failure_files[0].exists())
            self.assertEqual(dsp.calls.count('item0'), 1)

    @patch.object(m, 'publish_waveform', fake_publish)
    def test_resume_source_or_output_conflict_rejected(self):
        for location in ['source', 'output']:
            with self.subTest(location=location), tempfile.TemporaryDirectory() as directory:
                contract = contract_fixture(Path(directory)); dsp = FakeDSP()
                m._run_contract(contract, dsp)
                path = Path(contract['rows'][0]['execution_native_path']) if location == 'source' else Path(contract['output_root']) / 'audio/item0.wav'
                path.write_bytes(b'changed')
                with self.assertRaises(ValueError):
                    m._run_contract(contract, dsp)

    @patch.object(m, 'publish_waveform', fake_publish)
    def test_resume_contract_conflict_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            contract = contract_fixture(Path(directory)); dsp = FakeDSP()
            m._run_contract(contract, dsp)
            changed = copy.deepcopy(contract); changed['rows'][0]['label'] = '1'
            with self.assertRaisesRegex(ValueError, 'contract conflict'):
                m._run_contract(changed, dsp)

    @patch.object(m, 'publish_waveform', fake_publish)
    def test_unreceipted_waveform_failure_preserves_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            contract = contract_fixture(Path(directory)); dsp = FakeDSP(); dsp.fail.add('item1')
            m._run_contract(contract, dsp)
            output = Path(contract['output_root']); orphan = output / 'audio/item1.wav'
            orphan.write_bytes(b'orphan')
            dsp.fail.clear()
            result = m._run_contract(contract, dsp)
            self.assertEqual(result['failed'], 1)
            self.assertIn('unreceipted', result['failures'][0]['message'])
            self.assertEqual(orphan.read_bytes(), b'orphan')
            self.assertFalse((output / 'COMMIT.json').exists())

    def test_exclusive_lock_and_nonempty_unbound_output(self):
        with tempfile.TemporaryDirectory() as directory:
            contract = contract_fixture(Path(directory)); output = Path(contract['output_root'])
            with m.writer_lock(output):
                with self.assertRaisesRegex(RuntimeError, 'lock is held'):
                    m._run_contract(contract, FakeDSP())
            (output / 'unknown').write_bytes(b'preserve')
            with self.assertRaisesRegex(ValueError, 'nonempty output'):
                m._run_contract(contract, FakeDSP())

    def test_atomic_json_never_overwrites_and_rejects_nan(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'value.json'
            m.write_new(path, {'x': 1})
            with self.assertRaises(FileExistsError):
                m.write_new(path, {'x': 2})
            self.assertEqual(m.read_json(path), {'x': 1})
            with self.assertRaises(ValueError):
                m.write_new(Path(directory) / 'nan.json', {'x': float('nan')})

    def test_worker_limits(self):
        for workers in [0, 17, True]:
            with self.assertRaises(ValueError):
                m._run_contract({}, FakeDSP(), workers)

    def test_library_mapping_uses_symbol_not_candidate_name(self):
        maps = ('1000-2000 r-xp 00001000 00:30 11 /wrong/libsndfile.so\n'
                '3000-4000 r-xp 00002000 00:31 12 /actual/bundled library.so\n')
        actual = m.symbol_mapping(0x3015, maps)
        self.assertEqual(actual['mapped_path'], '/actual/bundled library.so')
        self.assertEqual(actual['symbol_file_offset'], 0x2015)
        self.assertEqual(actual['device_minor'], 0x31)
        shifted = maps.replace('3000-4000', '9000-a000')
        self.assertEqual(m.symbol_mapping(0x9015, shifted), actual)

    def test_library_mapping_rejects_missing_deleted_ambiguous_and_nonexec(self):
        for maps in ['', '1000-2000 r--p 00000000 00:30 11 /lib.so',
                     '1000-2000 r-xp 00000000 00:30 11 /lib.so (deleted)',
                     '1000-2000 r-xp 00000000 00:30 0 [anonymous]',
                     '1000-2000 r-xp 00000000 00:30 11 /one.so\n1000-2000 r-xp 00000000 00:30 12 /two.so']:
            with self.assertRaises(ValueError):
                m.symbol_mapping(0x1500, maps)

    def test_real_loaded_library_resolution_if_available(self):
        if not Path('/proc/self/maps').is_file():
            self.skipTest('Linux mapping test requires /proc/self/maps')
        try:
            import soundfile
        except ModuleNotFoundError:
            self.skipTest('SoundFile unavailable locally')
        binding, proof = m.loaded_sndfile_binding(soundfile)
        self.assertEqual(binding['sha256'], m.digest(binding['path']))
        self.assertEqual(proof['symbol'], 'sf_version_string')
        self.assertNotIn('address', proof)
        self.assertEqual(m.loaded_sndfile_binding(soundfile), (binding, proof))

    @patch.object(m, 'publish_waveform', fake_publish)
    def test_extra_output_entries_rejected_not_added_to_commit(self):
        for relative in ['surprise.json', 'audio/extra.wav', 'items/extra.json', 'runs/bad.json']:
            with self.subTest(relative=relative), tempfile.TemporaryDirectory() as directory:
                contract = contract_fixture(Path(directory)); dsp = FakeDSP(); dsp.fail.add('item1')
                m._run_contract(contract, dsp)
                output = Path(contract['output_root'])
                (output / relative).write_bytes(b'extra')
                dsp.fail.clear()
                with self.assertRaisesRegex(ValueError, 'unexpected'):
                    m._run_contract(contract, dsp)
                self.assertFalse((output / 'COMMIT.json').exists())

    @patch.object(m, 'publish_waveform', fake_publish)
    def test_receipt_payload_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            contract = contract_fixture(Path(directory)); dsp = FakeDSP(); dsp.fail.add('item1')
            m._run_contract(contract, dsp)
            receipt_path = Path(contract['output_root']) / 'items/item0.json'
            receipt = m.read_json(receipt_path); receipt['payload']['row_sha256'] = '0' * 64
            receipt_path.write_bytes(m.canonical(receipt))
            result = m._run_contract(contract, dsp)
            self.assertEqual(result['failed'], 2)
            self.assertIn('receipt payload hash', result['failures'][0]['message'])

    def test_real_synthetic_waveform_pipeline_if_dependencies_available(self):
        try:
            import standardize_native30_v1 as dsp
        except ModuleNotFoundError:
            self.skipTest('numerical DSP dependencies unavailable locally')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract = contract_fixture(root, count=1)
            source = Path(contract['rows'][0]['execution_native_path'])
            t = dsp.np.arange(3100) / 100
            values = dsp.np.column_stack([0.2 + dsp.np.sin(2*dsp.np.pi*3*t),
                                          2*dsp.np.cos(2*dsp.np.pi*7*t)])
            dsp.sf.write(source, values, 100, subtype='DOUBLE')
            contract['rows'][0]['native_evidence']['sha256'] = m.digest(source)
            result = m._run_contract(contract, dsp, workers=1)
            self.assertEqual(result['completed'], 1)
            receipt = m.read_json(Path(contract['output_root']) / 'items/item0.json')['payload']
            self.assertEqual(receipt['audit']['coordinates']['crop_start_frame'], 50)
            self.assertGreater(receipt['audit']['samples_abs_above_one'], 0)
            self.assertEqual(receipt['audit']['native_crop_float64_sha256'], dsp.pcm_hash(values[50:3050]))


if __name__ == '__main__':
    unittest.main()
