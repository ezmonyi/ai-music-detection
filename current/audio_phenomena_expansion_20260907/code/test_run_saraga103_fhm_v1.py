"""Synthetic checks only; never freezes receipts or runs feature extraction."""
import copy
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_saraga103_fhm_v1 as runner


def write_json(path, value):
    path.write_bytes(runner.canonical(value) + b'\n')


def write_csv(path, records):
    names = list(dict.fromkeys(k for r in records for k in r))
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=names)
        writer.writeheader()
        writer.writerows(records)


def native_fixture(index=0, sr=44100):
    iid = 'saraga_hindustani_synthetic_' + str(index)
    mbid = 'synthetic_' + str(index)
    frames, start = sr * 60, 1234567
    source = str(runner.RD / 'saraga_hindustani_physical_v1/raw' / (mbid + '.mp3'))
    digest = hashlib.sha256(iid.encode()).hexdigest()
    row = dict(id=iid, role=runner.ROLE, label='0', source_id=runner.SOURCE,
               source_group=runner.SOURCE, group_id='synthetic_group_' + str(index // 4),
               classifier_admission_authorized='False', evaluation_allowed='False',
               audio_path=source, source_audio_sha256=digest, registered_raw_sha256=digest,
               native_sample_rate_hz=str(sr), crop_start_frame=str(start), crop_frames=str(frames),
               crop_end_frame_exclusive=str(start + frames), sf_header_frames=str(2 * start + frames + 7),
               sf_actual_read_frames=str(2 * start + frames + 1), audio_offset_s=str(start / sr),
               native_crop_float64_sha256='a' * 64)
    selected = dict(item_id=iid, role=runner.ROLE, label='0', source_id=runner.SOURCE,
                    group_id=row['group_id'], source_audio_path=source, acquisition_raw_sha256=digest)
    q = dict(mbid=mbid, source_audio_path=source, raw_hashes_after={'sha256': digest},
             native_sample_rate_hz=sr, native_channels=2, crop_start_frame=start,
             crop_frames=frames, crop_end_frame_exclusive=start + frames,
             observed_header={'header_frames': int(row['sf_header_frames'])},
             observed_actual_frames=int(row['sf_actual_read_frames']), native_float64_sha256='a' * 64,
             real_empty_read_observed=True, seek_proof={'exact_bytes_equal': True})
    proof = dict(item_id=iid, role=runner.ROLE, classifier_admission=False,
                 status='passed_interval_materialization_not_classifier_admission', interval_proof=q)
    return row, proof, selected


class NativeProvenanceTests(unittest.TestCase):
    def test_native_counts_and_rounding_for_both_rates(self):
        for sr in (44100, 48000):
            row, proof, selected = native_fixture(sr=sr)
            runner.validate_native_row(row, proof, selected)
            self.assertEqual(int(row['crop_frames']), sr * 60)
            if sr == 48000:
                row['crop_frames'] = '2646000'
                with self.assertRaisesRegex(ValueError, 'crop_frames'):
                    runner.validate_native_row(row, proof, selected)

    def test_exact_metadata_boundaries_reject_mutations(self):
        for key in ('id', 'role', 'label', 'source_id', 'source_group', 'group_id',
                    'classifier_admission_authorized', 'evaluation_allowed', 'audio_path',
                    'source_audio_sha256', 'registered_raw_sha256', 'native_sample_rate_hz',
                    'crop_start_frame', 'crop_frames', 'crop_end_frame_exclusive',
                    'sf_header_frames', 'sf_actual_read_frames', 'native_crop_float64_sha256'):
            with self.subTest(key=key):
                row, proof, selected = native_fixture()
                row[key] += '_tampered'
                with self.assertRaises(ValueError):
                    runner.validate_native_row(row, proof, selected)

    def test_offset_and_proof_flags_reject_mutations(self):
        row, proof, selected = native_fixture()
        row['audio_offset_s'] = str(float(row['audio_offset_s']) + 1 / 44100)
        with self.assertRaisesRegex(ValueError, 'Offset'):
            runner.validate_native_row(row, proof, selected)
        for scope, key, value in (('proof', 'classifier_admission', True), ('proof', 'status', 'draft'),
                                  ('interval', 'real_empty_read_observed', False),
                                  ('seek', 'exact_bytes_equal', False)):
            row, proof, selected = native_fixture()
            target = proof if scope == 'proof' else proof['interval_proof']
            if scope == 'seek':
                target = target['seek_proof']
            target[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                runner.validate_native_row(row, proof, selected)

    def test_live_synced_manifest_and_all_proofs_without_audio_access(self):
        base = runner.ROOT / 'manifests/saraga_external103_intervals_v1'
        if not base.exists():
            base = runner.INTERVALS
        if not base.exists():
            self.skipTest('Synced manifest not present at this test root')
        selection = runner.read(runner.ROOT / 'preregistration/saraga_external103_selection_frozen_v1.json')
        selected = {r['item_id']: r for r in runner.rows(runner.ROOT / selection['selection_draft_directory'] / 'metadata.csv')}
        native = runner.rows(base / 'final/native_metadata_60s.csv')
        inventory = runner.read(base / 'final/item_proof_inventory.json')
        self.assertEqual(len(native), 103)
        self.assertEqual(sum(r['native_sample_rate_hz'] == '48000' for r in native), 7)
        for row in native:
            path = base / 'items' / row['id'] / 'proof.json'
            self.assertEqual(runner.sha(path), inventory[row['id']])
            runner.validate_native_row(row, runner.read(path), selected[row['id']])


class ResultTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.features = self.base / 'features'
        (self.features / 'items').mkdir(parents=True)
        self.metadata = self.base / 'metadata.csv'
        self.native = [native_fixture(i, 48000 if i >= 96 else 44100)[0] for i in range(103)]
        write_csv(self.metadata, self.native)
        self.ref = {k: copy.deepcopy(v) for k, v in runner.read(runner.ROOT / 'results/features_60s_v1/contract.json').items()
                    if k in ('duration', 'preflight_only', 'input_config', 'F_config', 'feature_names', 'code_sha256', 'runtime')}
        self.contract = dict(self.ref, selected_ids=[r['id'] for r in self.native], metadata_sha256=runner.sha(self.metadata))
        self.contract['contract_hash'] = runner.seal(self.contract)
        write_json(self.features / 'contract.json', self.contract)
        self.records = []
        for i, row in enumerate(self.native):
            item = dict(row, input_row_hash=runner.seal(row), extraction_contract_hash=self.contract['contract_hash'],
                        extraction_status='ok', source_audio_path=row['audio_path'],
                        crop_start_frame=int(row['crop_start_frame']), crop_frames=int(row['crop_frames']),
                        source_total_frames=int(row['sf_header_frames']), analysis_frames=960000, analysis_sr=16000,
                        analysis_waveform_sha256='b' * 64, F_status='ok' if i else 'insufficient_regions',
                        H_status='ok', M_status='ok' if i else 'insufficient_recurrence')
            for names in self.ref['feature_names'].values():
                for name in names:
                    item[name] = None if i == 0 else 0.12345678901234568
            self.records.append(item)
            write_json(self.item_path(i), item)
        self.refresh()

    def item_path(self, i):
        return self.features / 'items' / (hashlib.sha256(self.native[i]['id'].encode()).hexdigest() + '.json')

    def refresh(self):
        write_csv(self.features / 'features.csv', self.records)
        counts = {k: dict(runner.Counter(r[k] for r in self.records)) for k in ('F_status', 'H_status', 'M_status')}
        write_json(self.features / 'summary.json', dict(expected=103, recorded=103, complete_accounting=True,
                   contract_hash=self.contract['contract_hash'], features_csv_sha256=runner.sha(self.features / 'features.csv'),
                   status_counts=dict(runner.Counter(r['extraction_status'] for r in self.records)), family_status_counts=counts))
        write_json(self.features / 'process.json', dict(state='finished', complete_accounting=True))

    def test_optional_unavailable_descriptors_stay_in_103_rows(self):
        result = runner.validate_results(self.features, self.metadata, self.ref)
        self.assertEqual(result['rows'], 103)
        self.assertEqual(result['family_status_counts']['F_status']['insufficient_regions'], 1)
        self.assertEqual(runner.rows(self.features / 'features.csv')[0]['M_best_lag_sec'], '')

    def test_csv_item_scalar_none_and_nan_tampering_rejected(self):
        for key, value in [('F_phase_residual_cvar_all', 999), ('M_best_lag_sec', 'nan'), ('F_status', 'ok'),
                           ('analysis_waveform_sha256', 'c' * 64), ('group_id', 'other_group')]:
            original = self.records[0][key]
            self.records[0][key] = value
            self.refresh()
            with self.subTest(key=key), self.assertRaises(ValueError):
                runner.validate_results(self.features, self.metadata, self.ref)
            self.records[0][key] = original

    def test_manifest_change_rejected(self):
        self.native[0]['group_id'] = 'different_group'
        write_csv(self.metadata, self.native)
        with self.assertRaisesRegex(ValueError, 'Feature identity mismatch'):
            runner.validate_results(self.features, self.metadata, self.ref)

    def test_missing_row_rejected(self):
        self.records.pop()
        self.refresh()
        with self.assertRaisesRegex(ValueError, 'Feature identity mismatch'):
            runner.validate_results(self.features, self.metadata, self.ref)

    def test_cached_item_row_binding_rejected(self):
        item = runner.read(self.item_path(0))
        item['input_row_hash'] = '0' * 64
        write_json(self.item_path(0), item)
        with self.assertRaisesRegex(ValueError, 'Item binding mismatch'):
            runner.validate_results(self.features, self.metadata, self.ref)

    def test_48000_fixed_44100_crop_rejected(self):
        item = runner.read(self.item_path(102))
        item['crop_frames'] = 2646000
        write_json(self.item_path(102), item)
        with self.assertRaisesRegex(ValueError, 'Native interval mismatch'):
            runner.validate_results(self.features, self.metadata, self.ref)

    def test_extractor_error_not_silently_filtered(self):
        item = runner.read(self.item_path(0))
        item['extraction_status'] = 'partial_error'
        write_json(self.item_path(0), item)
        with self.assertRaisesRegex(ValueError, 'Extraction failed; keep row and inspect'):
            runner.validate_results(self.features, self.metadata, self.ref)
        self.assertTrue(self.item_path(0).exists())


class ReceiptTests(unittest.TestCase):
    def test_publish_never_replaces_existing_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'receipt.json'
            runner.publish(path, {'status': 'draft'})
            original, inode = path.read_bytes(), path.stat().st_ino
            runner.publish(path, {'status': 'draft'})
            self.assertEqual(path.stat().st_ino, inode)
            with self.assertRaisesRegex(ValueError, 'immutable receipt differs'):
                runner.publish(path, {'status': 'frozen'})
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(Path(temp).iterdir()), [path])

    def test_false_frozen_and_hash_tampering_cannot_reach_extraction(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'receipt.json'
            body = {'synthetic': True}
            variants = [dict(status='draft', authorized_stage=None, contract=body, contract_sha256=runner.seal(body)),
                        dict(status='frozen', authorized_stage=None, contract=body, contract_sha256=runner.seal(body)),
                        dict(status='frozen', authorized_stage=runner.STAGE, contract={'synthetic': False}, contract_sha256=runner.seal(body))]
            for receipt in variants:
                write_json(path, receipt)
                with patch.object(runner, 'build_contract', return_value=body), patch.object(runner, 'OUTPUT', Path(temp) / 'never_created'), \
                     patch.object(runner.sys, 'argv', ['runner', 'run', '--receipt', str(path), '--receipt-sha256', runner.sha(path)]), \
                     patch.object(runner.subprocess, 'run') as launch:
                    with self.assertRaisesRegex(ValueError, 'Frozen contract mismatch'):
                        runner.main()
                    launch.assert_not_called()
                    self.assertFalse((Path(temp) / 'never_created').exists())
            with patch.object(runner, 'build_contract', return_value=body), \
                 patch.object(runner.sys, 'argv', ['runner', 'run', '--receipt', str(path), '--receipt-sha256', '0' * 64]):
                with self.assertRaisesRegex(ValueError, 'Frozen receipt hash mismatch'):
                    runner.main()


class NumericalInputTests(unittest.TestCase):
    def test_synthetic_float64_seek_and_analysis_parity(self):
        try:
            import numpy as np
            import soundfile as sf
            from scipy.signal import resample_poly
            from audio_inputs import load_exact
        except ImportError as error:
            self.skipTest(str(error))
        import math
        with tempfile.TemporaryDirectory() as temp:
            for sr in (44100, 48000):
                count, start = sr * 60, 23
                t = np.arange(count + start + 31, dtype=np.float64)
                audio = np.column_stack((np.sin(t * .011) * .17, np.cos(t * .019) * .13))
                path = Path(temp) / ('synthetic_' + str(sr) + '.wav')
                sf.write(path, audio, sr, subtype='DOUBLE')
                row = native_fixture(sr=sr)[0]
                row.update(audio_path=str(path), crop_start_frame=str(start), sf_header_frames=str(len(audio)),
                           audio_offset_s=str(start / sr),
                           native_crop_float64_sha256=hashlib.sha256(audio[start:start + count].astype('<f8').tobytes()).hexdigest())
                runner.verify_native_pcm([row])
                actual, audit = load_exact(row, duration=60)
                expected = audio[start:start + count].mean(axis=1)
                expected -= expected.mean()
                gcd = math.gcd(sr, 16000)
                expected = np.ascontiguousarray(resample_poly(expected, 16000 // gcd, sr // gcd,
                    window=('kaiser', 5.0), padtype='constant'), dtype='<f8')
                self.assertEqual(actual.tobytes(), expected.tobytes())
                self.assertEqual(audit['analysis_frames'], 960000)
                self.assertEqual(audit['crop_frames'], count)
                row['native_crop_float64_sha256'] = '0' * 64
                with self.assertRaisesRegex(ValueError, 'Native seek crop'):
                    runner.verify_native_pcm([row])


if __name__ == '__main__':
    unittest.main(verbosity=2)
