"""Synthetic-only checks. Never accesses the acquired MUSIC8K media."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf
import prepare_mureka60_inputs as M


class Info:
    def __init__(self, path):
        self.samplerate, self.channels, self.frames = M.SR, 2, M.FRAMES + 3
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(M.json_bytes(value))


def acquisition_fixture(root):
    ranked = [dict(id=str(i), path=f'mureka_v9/{i}.mp3', bytes=1,
                   sha256=hashlib.sha256(b'x').hexdigest(), role='reserved_unscored', source='Mureka v9',
                   reference_group_id=f'music8k_reference_{i}',
                   rank=hashlib.sha256((M.SEED+'|'+str(i)).encode()).hexdigest()) for i in range(650)]
    ranked.sort(key=lambda r:r['rank'])
    c = dict(repo='homura23/MUSIC8K', revision=M.REVISION, seed=M.SEED,
             status='frozen_for_acquisition_only', classifier_authorized=False,
             selected_count=500, ranked_candidates=ranked, selected=ranked[:500],
             excluded_probe_ids=[str(i) for i in range(1000, 1012)], intended_bytes=500)
    put(root/'contract.json', c)
    ch = M.sha_file(root/'contract.json')
    receipts, physical = {}, []
    for row in c['selected']:
        path = root/'raw'/row['path']
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'x')
        receipt = dict(row, contract_sha256=ch, status='passed', eligible_native60=True, all_samples_finite=True,
                       sample_rate=M.SR, channels=2, decoded_frames=M.FRAMES+3, decoded_duration_s=(M.FRAMES+3)/M.SR,
                       decoded_native_float32_sha256='f'*64)
        put(root/'items'/(row['id']+'.json'), receipt)
        receipts[row['id']+'.json'] = M.sha_file(root/'items'/(row['id']+'.json'))
        physical.append(dict(id=row['id'], sha256=row['sha256'], bytes=1, sample_rate=M.SR, channels=2,
                             frames=M.FRAMES+3, eligible_native60=True, decoded_native_float32_sha256='f'*64))
    s = dict(status='completed', selected=500, passed=500, eligible_native60=500, contract_sha256=ch,
             bytes=500, classifiers_fitted=0, neural_inference=False, receipts_sha256=receipts)
    put(root/'summary.json', s)
    a = dict(status='passed', selected=500, eligible_native60=500, all_original_files_rehashed=True,
             selection_independently_reconstructed=True, classifier_admission_authorized=False,
             source_revision=M.REVISION, contract_sha256=ch, summary_sha256=M.sha_file(root/'summary.json'),
             bytes=500, rows=physical)
    put(root/'local_physical_audit_v1.json', a)
    r = dict(status='passed', rows=500, local_audit_sha256=M.sha_file(root/'local_physical_audit_v1.json'),
             remote_root=str(root), all_raw_files_rehashed=True, all_item_receipts_rehashed=True,
             classifier_fitted=False, bytes=500)
    put(root/'remote_copy_audit_v1.json', r)
    return c, ch


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)/'source'
        self.out = Path(self.tmp.name)/'output'
        self.c, ch = acquisition_fixture(self.root)
        self.h = patch.object(M, 'ACQUISITION_SHA', ch)
        self.info = patch.object(M.sf, 'SoundFile', Info)
        self.h.start(); self.info.start()
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(self.h.stop); self.addCleanup(self.info.stop)

    def test_full500_freeze_and_recheck(self):
        c = M.freeze(self.root, self.out)
        rows = M.validate_contract(c, self.root, self.out)
        self.assertEqual(len(rows), 500)
        self.assertEqual([r['id'] for r in rows], [r['id'] for r in self.c['selected']])
        self.assertTrue(all(r['role'] == M.ROLE and r['acquisition_role']=='reserved_unscored' for r in rows))
        self.assertTrue(all(r['crop_start_frame'] == 1 for r in rows))
        with self.assertRaisesRegex(ValueError, 'already frozen'):
            M.freeze(self.root, self.out)

    def test_missing_remote_audit(self):
        (self.root/'remote_copy_audit_v1.json').unlink()
        with self.assertRaisesRegex(ValueError, 'Missing regular input'):
            M.freeze(self.root, self.out)
        self.assertFalse((self.out/'materialization_contract.json').exists())

    def test_freeze_foreign_directory_rejected(self):
        self.out.mkdir()
        (self.out/'unrelated.txt').write_text('preserve me')
        with self.assertRaisesRegex(ValueError, 'foreign/unreceipted'):
            M.freeze(self.root, self.out)
        self.assertEqual((self.out/'unrelated.txt').read_text(), 'preserve me')

    def test_freeze_and_resume_initialize_decoder_before_runtime(self):
        events = []
        audit, runtime = M.audit_inputs, M.runtime
        def recorded_audit(source):
            events.append('decode')
            return audit(source)
        def recorded_runtime():
            events.append('runtime')
            return runtime()
        with patch.object(M, 'audit_inputs', side_effect=recorded_audit), patch.object(M, 'runtime', side_effect=recorded_runtime):
            c = M.freeze(self.root, self.out)
            M.validate_contract(c, self.root, self.out)
        self.assertEqual(events, ['decode','runtime','decode','runtime'])

    def test_remote_pending_or_admitted(self):
        path = self.root/'remote_copy_audit_v1.json'
        original = M.read_json(path)
        for k,v in [('status','pending'),('classifier_fitted',True),('all_raw_files_rehashed',False)]:
            put(path, dict(original, **{k:v}))
            with self.assertRaisesRegex(ValueError, 'remote copy audit'):
                M.freeze(self.root, self.out)

    def test_decoder_frame_mismatch_and_short_source(self):
        for frames in (M.FRAMES+2, M.FRAMES-1):
            class WrongInfo(Info):
                def __init__(self, path):
                    super().__init__(path)
                    self.frames = frames
            with patch.object(M.sf, 'SoundFile', WrongInfo):
                with self.assertRaisesRegex(ValueError, 'Decoder frames'):
                    M.freeze(self.root, self.out)

    def test_native_hash_mutation(self):
        c = M.freeze(self.root, self.out)
        (self.root/'raw'/self.c['selected'][0]['path']).write_bytes(b'y')
        with self.assertRaisesRegex(ValueError, 'bytes/hash'):
            M.validate_contract(c, self.root, self.out)

    def test_hash_and_runtime_contract_mutation(self):
        c = M.freeze(self.root, self.out)
        broken = copy.deepcopy(c); broken['role'] = 'development'
        with self.assertRaisesRegex(ValueError, 'canonical hash'):
            M.validate_contract(broken, self.root, self.out)
        for key, value, error in [('code_sha256','0'*64,'code changed'), ('runtime',{},'runtime changed'),
                                   ('configuration',{},'configuration changed')]:
            broken = copy.deepcopy(c); broken[key] = value
            broken['contract_sha256'] = M.canonical_hash({k:v for k,v in broken.items() if k!='contract_sha256'})
            with self.assertRaisesRegex(ValueError, error):
                M.validate_contract(broken, self.root, self.out)

    def test_audit_mutation_after_freeze(self):
        c = M.freeze(self.root, self.out)
        path = self.root/'remote_copy_audit_v1.json'
        put(path, dict(M.read_json(path), extra='changed'))
        with self.assertRaisesRegex(ValueError, 'evidence changed'):
            M.validate_contract(c, self.root, self.out)

    def test_selection_ranking_probes_source_admission(self):
        mutations = [lambda c:c['selected'].reverse(),
                     lambda c:c['ranked_candidates'].reverse(),
                     lambda c:c['excluded_probe_ids'].__setitem__(0,c['selected'][0]['id']),
                     lambda c:c.__setitem__('repo','other/MUSIC8K'),
                     lambda c:c.__setitem__('classifier_authorized',True),
                     lambda c:c['selected'][0].__setitem__('role','development')]
        for mutate in mutations:
            bad = copy.deepcopy(self.c); mutate(bad)
            with self.assertRaises(ValueError):
                M.selected_rows(bad)

    def test_no_publication_before_all_pass(self):
        M.freeze(self.root, self.out)
        with patch.object(M, 'one_item', side_effect=ValueError('synthetic failure')):
            with self.assertRaisesRegex(ValueError, 'synthetic failure'):
                M.materialize(self.root, self.out)
        self.assertFalse((self.out/'inference_manifest.csv').exists())
        self.assertFalse(list(self.out.glob('inference_shard_*.txt')))

    def test_manifest_and_shard_interface(self):
        M.freeze(self.root, self.out)
        def fake_item(source, output, row, contract_hash):
            i = 'music8k_mureka_v9_'+row['id']
            put(output/'items'/(i+'.json'), {'synthetic':True})
            return dict(item_id=i, standardized_path=str(output/'audio'/(i+'.wav')),
                        standardized_file_sha256='a'*64, source_audio_path=str(source/'raw'/row['path']),
                        crop_start_frame=1, source_total_frames=M.FRAMES+3, crop_end_frame_exclusive=M.FRAMES+1,
                        group_id=row['reference_group_id'], source_audio_sha256=row['sha256'])
        with patch.object(M, 'one_item', side_effect=fake_item):
            summary = M.materialize(self.root, self.out)
        self.assertEqual(summary['rows'], 500)
        self.assertEqual([s['rows'] for s in summary['shards']], [24]*20+[20])
        self.assertEqual(len(list(self.out.glob('inference_shard_*.txt'))), 21)
        for s in summary['shards']:
            paths = Path(s['path']).read_text().splitlines()
            self.assertTrue(all(Path(p).is_absolute() and p.endswith('.wav') for p in paths))


class CropTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root, self.out = Path(self.tmp.name)/'source', Path(self.tmp.name)/'output'
        path = self.root/'raw/mureka_v9/1.mp3'
        path.parent.mkdir(parents=True)
        # A synthetic FLOAT WAV with an MP3 suffix exercises samples without acquiring audio.
        # Deliberately includes DC and values above 1 to expose unintended gain/limiting.
        self.data = np.empty((M.FRAMES+3,2), dtype=np.float32)
        self.data[:,0] = np.linspace(-1.5, 1.5, len(self.data), dtype=np.float32)
        self.data[:,1] = .25
        sf.write(path, self.data, M.SR, format='WAV', subtype='FLOAT')
        self.row = dict(id='1', path='mureka_v9/1.mp3', sha256=M.sha_file(path), bytes=path.stat().st_size,
                        native_frames=len(self.data), crop_start_frame=1, crop_frames=M.FRAMES,
                        reference_group_id='music8k_reference_1', acquisition_decoded_float32_sha256='f'*64)

    def test_exact_center_odd_frames_and_resume(self):
        receipt = M.one_item(self.root, self.out, self.row, 'c'*64)
        back, sr = sf.read(receipt['standardized_path'], dtype='float32', always_2d=True)
        self.assertEqual(sr, M.SR)
        self.assertTrue(np.array_equal(back,self.data[1:1+M.FRAMES]))
        self.assertGreater(np.max(back[:,0]),1)
        self.assertTrue(np.all(back[:,1]==.25))
        self.assertEqual(receipt,M.one_item(self.root,self.out,self.row,'c'*64))

    def test_exact_even_frame_center(self):
        self.row['native_frames'] -= 1
        path = self.root/'raw'/self.row['path']
        sf.write(path,self.data[:-1],M.SR,format='WAV',subtype='FLOAT')
        self.row.update(sha256=M.sha_file(path),bytes=path.stat().st_size)
        self.assertTrue(np.array_equal(M.decode_crop(self.root,self.row),self.data[1:1+M.FRAMES]))

    def test_short_source(self):
        self.row['native_frames'] = M.FRAMES-1
        path = self.root/'raw'/self.row['path']
        sf.write(path,self.data[:M.FRAMES-1],M.SR,format='WAV',subtype='FLOAT')
        self.row.update(sha256=M.sha_file(path),bytes=path.stat().st_size)
        with self.assertRaisesRegex(ValueError,'Short source'):
            M.decode_crop(self.root,self.row)

    def test_resume_output_and_receipt_mutation(self):
        receipt = M.one_item(self.root,self.out,self.row,'c'*64)
        with self.assertRaisesRegex(ValueError,'receipt/contract'):
            M.one_item(self.root,self.out,self.row,'d'*64)
        path = Path(receipt['standardized_path'])
        with path.open('r+b') as f:
            f.seek(-4,2); f.write(b'\x00'*4)
        with self.assertRaisesRegex(ValueError,'output hash'):
            M.one_item(self.root,self.out,self.row,'c'*64)

    def test_orphan_no_overwrite_and_publish_conflict(self):
        path = self.out/'audio/music8k_mureka_v9_1.wav'
        path.parent.mkdir(parents=True); path.write_bytes(b'owned')
        with self.assertRaisesRegex(ValueError,'Unreceipted output'):
            M.one_item(self.root,self.out,self.row,'c'*64)
        self.assertEqual(path.read_bytes(), b'owned')
        with self.assertRaisesRegex(ValueError,'Conflicting immutable'):
            M.publish(path,b'new')

    def test_destination_boundary(self):
        with self.assertRaisesRegex(ValueError,'NFS-data source'):
            M.validate_paths(self.root,self.out)
        with self.assertRaisesRegex(ValueError,'destination'):
            M.validate_paths(M.SOURCE,Path('/mnt/nfs-code/not-allowed'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
